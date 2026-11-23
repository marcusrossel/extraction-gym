"""Shared loaders and DataFrame helpers for extraction-gym benchmark results."""

import json
import math
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "output"

# Extractors expected to produce identical optimal tree costs (matches Optimal::Tree in src/main.rs).
TREE_EXTRACTORS: tuple[str, ...] = (
    "bottom-up",
    "faster-bottom-up",
    "prio-queue",
    "uniform-cost-search",
    "a-star-two-phase",
    "a-star-interleaved",
)

STRUCTURE_FEATURES: dict[str, str] = {
    "reach_enodes": "Reachable E-Nodes",
    "reach_enodes_pct": "Reachable E-Nodes (%)",
    "enodes_per_eclass": "OR-Branching (E-Nodes / E-Class)",
    "avg_arity": "AND-Branching (Avg Arity)",
    "avg_child_eclasses": "E-Class Out-Degree",
    "avg_parent_enodes": "Parent Fan-In / E-Class",
    "leaf_pct": "Leaf E-Nodes (%)",
    "max_depth": "Max BFS Depth",
    "cyclic_eclasses_pct": "Cyclic E-Classes (%)",
}


@dataclass(frozen=True)
class EGraphInfo:
    """Precomputed per-eclass adjacency, size, arity, and cycle info for an e-graph."""

    eclass_children: dict[str, tuple[str, ...]]
    eclass_enodes: dict[str, int]
    eclass_child_refs: dict[str, int]
    eclass_leaves: dict[str, int]
    cyclic_eclasses: frozenset[str]
    total_enodes: int
    total_eclasses: int


@dataclass(frozen=True)
class EGraphStats:
    """Total, reachability, density, branching, depth, and cycle stats for one (e-graph, roots)."""

    total_enodes: int | None = None
    total_eclasses: int | None = None
    reach_enodes: int | None = None
    reach_eclasses: int | None = None
    reach_enodes_pct: float | None = None
    reach_eclasses_pct: float | None = None
    enodes_per_eclass: float | None = None
    avg_arity: float | None = None
    avg_child_eclasses: float | None = None
    avg_parent_enodes: float | None = None
    leaf_pct: float | None = None
    max_depth: int | None = None
    cyclic_eclasses_pct: float | None = None


_RESULTS_SCHEMA: pl.Schema = pl.Schema(
    {
        "name": pl.String,
        "suite": pl.String,
        "benchmark": pl.String,
        "roots": pl.String,
        "label": pl.String,
        "n_roots": pl.Int64,
        "result_index": pl.Int64,
        "extractor": pl.String,
        "micros": pl.Int64,
        "tree": pl.Float64,
        "dag": pl.Float64,
    }
)

_STATS_SCHEMA: pl.Schema = pl.Schema(
    {
        "name": pl.String,
        "roots": pl.String,
        "total_enodes": pl.Int64,
        "total_eclasses": pl.Int64,
        "reach_enodes": pl.Int64,
        "reach_eclasses": pl.Int64,
        "reach_enodes_pct": pl.Float64,
        "reach_eclasses_pct": pl.Float64,
        "enodes_per_eclass": pl.Float64,
        "avg_arity": pl.Float64,
        "avg_child_eclasses": pl.Float64,
        "avg_parent_enodes": pl.Float64,
        "leaf_pct": pl.Float64,
        "max_depth": pl.Int64,
        "cyclic_eclasses_pct": pl.Float64,
    }
)


def find_result_files(
    files: Sequence[str | Path] = (), output_dir: str | Path | None = None
) -> list[Path]:
    """Return benchmark result JSON files, defaulting to output/**/*.json."""
    if files:
        return [Path(f) for f in files]
    if output_dir is not None:
        return sorted(Path(output_dir).rglob("*.json"))
    local_out = Path("output")
    found = sorted(local_out.rglob("*.json")) if local_out.exists() else []
    if found:
        return found
    return sorted(DEFAULT_OUTPUT_DIR.rglob("*.json")) if DEFAULT_OUTPUT_DIR.exists() else []


def load_entries(files: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load raw benchmark result entries from JSON files."""
    entries: list[dict[str, Any]] = []
    for file in files:
        try:
            with Path(file).open() as f:
                data = json.load(f)
            for idx, result in enumerate(data["results"]):
                entries.append(
                    {
                        "name": data["name"],
                        "extractor": data["extractor"],
                        "roots": tuple(result["roots"]),
                        "result_index": idx,
                        "micros": int(result["micros"]),
                        "tree": float(result["tree"]),
                        "dag": float(result["dag"]),
                    }
                )
        except OSError, json.JSONDecodeError, KeyError, TypeError, ValueError:
            print(f"Error loading {file}", file=sys.stderr)
    return entries


def _find_cyclic_eclasses(eclass_children: dict[str, tuple[str, ...]]) -> frozenset[str]:
    """Return all e-classes belonging to at least one directed cycle via iterative Tarjan SCC."""
    idx_counter = 0
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    scc_stack: list[str] = []
    cyclic: set[str] = set()

    for start in eclass_children:
        if start in index:
            continue
        call_stack: list[tuple[str, int]] = [(start, 0)]
        index[start] = lowlink[start] = idx_counter
        idx_counter += 1
        scc_stack.append(start)
        on_stack.add(start)

        while call_stack:
            u, ci = call_stack[-1]
            children = eclass_children.get(u, ())
            if ci < len(children):
                call_stack[-1] = (u, ci + 1)
                v = children[ci]
                if v not in index:
                    index[v] = lowlink[v] = idx_counter
                    idx_counter += 1
                    scc_stack.append(v)
                    on_stack.add(v)
                    call_stack.append((v, 0))
                elif v in on_stack and index[v] < lowlink[u]:
                    lowlink[u] = index[v]
            else:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[u])
                if lowlink[u] == index[u]:
                    comp: list[str] = []
                    while True:
                        w = scc_stack.pop()
                        on_stack.remove(w)
                        comp.append(w)
                        if w == u:
                            break
                    if len(comp) > 1 or u in children:
                        cyclic.update(comp)

    return frozenset(cyclic)


def load_egraph(name: str) -> EGraphInfo | None:
    """Load an e-graph JSON and precompute per-eclass adjacency, arity, leaf, and cycle info."""
    try:
        path = Path(name)
        if not path.exists():
            path = REPO_ROOT / name
        with path.open() as f:
            nodes: dict[str, dict[str, Any]] = json.load(f)["nodes"]
        eclass_child_sets: dict[str, set[str]] = {}
        eclass_enodes: dict[str, int] = {}
        eclass_child_refs: dict[str, int] = {}
        eclass_leaves: dict[str, int] = {}
        for n in nodes.values():
            ec: str = n["eclass"]
            eclass_enodes[ec] = eclass_enodes.get(ec, 0) + 1
            ch: list[str] = n["children"]
            eclass_child_refs[ec] = eclass_child_refs.get(ec, 0) + len(ch)
            if not ch:
                eclass_leaves[ec] = eclass_leaves.get(ec, 0) + 1
            child_set = eclass_child_sets.setdefault(ec, set())
            for child_id in ch:
                child_set.add(nodes[child_id]["eclass"])
        eclass_children = {ec: tuple(ch_set) for ec, ch_set in eclass_child_sets.items()}
        return EGraphInfo(
            eclass_children=eclass_children,
            eclass_enodes=eclass_enodes,
            eclass_child_refs=eclass_child_refs,
            eclass_leaves=eclass_leaves,
            cyclic_eclasses=_find_cyclic_eclasses(eclass_children),
            total_enodes=len(nodes),
            total_eclasses=len(eclass_children),
        )
    except OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError:
        return None


def egraph_stats(egraph: EGraphInfo | None, roots: Sequence[str]) -> EGraphStats:
    """Compute reachability, density, branching, depth, and cycle stats for `roots`."""
    if egraph is None:
        return EGraphStats()

    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for r in roots:
        if r not in depth:
            depth[r] = 0
            queue.append(r)

    max_depth = 0
    while queue:
        ec = queue.popleft()
        d = depth[ec]
        max_depth = max(max_depth, d)
        for child_ec in egraph.eclass_children.get(ec, ()):
            if child_ec not in depth:
                depth[child_ec] = d + 1
                queue.append(child_ec)

    reach_eclasses = len(depth)
    reach_enodes = sum(egraph.eclass_enodes.get(ec, 0) for ec in depth)
    reach_child_refs = sum(egraph.eclass_child_refs.get(ec, 0) for ec in depth)
    reach_leaves = sum(egraph.eclass_leaves.get(ec, 0) for ec in depth)
    reach_unique_edges = sum(len(egraph.eclass_children.get(ec, ())) for ec in depth)
    reach_cyclic = sum(1 for ec in depth if ec in egraph.cyclic_eclasses)

    return EGraphStats(
        total_enodes=egraph.total_enodes,
        total_eclasses=egraph.total_eclasses,
        reach_enodes=reach_enodes,
        reach_eclasses=reach_eclasses,
        reach_enodes_pct=100.0 * reach_enodes / egraph.total_enodes
        if egraph.total_enodes
        else None,
        reach_eclasses_pct=(
            100.0 * reach_eclasses / egraph.total_eclasses if egraph.total_eclasses else None
        ),
        enodes_per_eclass=reach_enodes / reach_eclasses if reach_eclasses else None,
        avg_arity=reach_child_refs / reach_enodes if reach_enodes else None,
        avg_child_eclasses=reach_unique_edges / reach_eclasses if reach_eclasses else None,
        avg_parent_enodes=reach_child_refs / reach_eclasses if reach_eclasses else None,
        leaf_pct=100.0 * reach_leaves / reach_enodes if reach_enodes else None,
        max_depth=max_depth if reach_eclasses else None,
        cyclic_eclasses_pct=100.0 * reach_cyclic / reach_eclasses if reach_eclasses else None,
    )


def load_results_df(
    files: Sequence[str | Path] = (),
    *,
    output_dir: str | Path | None = None,
    with_egraph_stats: bool = True,
) -> pl.DataFrame:
    """Load benchmark results into a tidy Polars DataFrame.

    One row per (name, roots, extractor), optionally enriched with total and
    root-reachable e-node/e-class counts computed once per unique (name, roots).
    """
    resolved = find_result_files(files, output_dir=output_dir)
    entries = load_entries(resolved)
    if not entries:
        empty = pl.DataFrame(schema=_RESULTS_SCHEMA)
        if with_egraph_stats:
            return empty.join(pl.DataFrame(schema=_STATS_SCHEMA), on=["name", "roots"], how="left")
        return empty

    roots_per_name: dict[str, set[tuple[str, ...]]] = {}
    for entry in entries:
        roots_per_name.setdefault(entry["name"], set()).add(entry["roots"])

    rows: list[dict[str, Any]] = []
    unique_instances: dict[tuple[str, str], tuple[str, ...]] = {}
    for entry in entries:
        name: str = entry["name"]
        roots_tuple: tuple[str, ...] = entry["roots"]
        roots_str = ", ".join(roots_tuple)
        benchmark = name.removeprefix("data/").removesuffix(".json")
        suite = benchmark.split("/", 1)[0] if "/" in benchmark else "other"
        label = f"{benchmark} [{roots_str}]" if len(roots_per_name[name]) > 1 else benchmark
        rows.append(
            {
                "name": name,
                "suite": suite,
                "benchmark": benchmark,
                "roots": roots_str,
                "label": label,
                "n_roots": len(roots_tuple),
                "result_index": entry["result_index"],
                "extractor": entry["extractor"],
                "micros": entry["micros"],
                "tree": entry["tree"],
                "dag": entry["dag"],
            }
        )
        unique_instances.setdefault((name, roots_str), roots_tuple)

    df = pl.DataFrame(rows, schema=_RESULTS_SCHEMA)
    if not with_egraph_stats:
        return df

    egraph_cache: dict[str, EGraphInfo | None] = {}
    stat_rows: list[dict[str, Any]] = []
    for (name, roots_str), roots_tuple in unique_instances.items():
        if name not in egraph_cache:
            egraph_cache[name] = load_egraph(name)
        st = egraph_stats(egraph_cache[name], roots_tuple)
        stat_rows.append(
            {
                "name": name,
                "roots": roots_str,
                "total_enodes": st.total_enodes,
                "total_eclasses": st.total_eclasses,
                "reach_enodes": st.reach_enodes,
                "reach_eclasses": st.reach_eclasses,
                "reach_enodes_pct": st.reach_enodes_pct,
                "reach_eclasses_pct": st.reach_eclasses_pct,
                "enodes_per_eclass": st.enodes_per_eclass,
                "avg_arity": st.avg_arity,
                "avg_child_eclasses": st.avg_child_eclasses,
                "avg_parent_enodes": st.avg_parent_enodes,
                "leaf_pct": st.leaf_pct,
                "max_depth": st.max_depth,
                "cyclic_eclasses_pct": st.cyclic_eclasses_pct,
            }
        )

    stats_df = pl.DataFrame(stat_rows, schema=_STATS_SCHEMA)
    return df.join(stats_df, on=["name", "roots"], how="left")


def filter_egraph_stats(
    df: pl.DataFrame,
    *,
    min_enodes: int | None = None,
    min_enodes_pct: float | None = None,
    min_eclasses: int | None = None,
    min_eclasses_pct: float | None = None,
) -> pl.DataFrame:
    """Filter a results or pairwise DataFrame by reachable e-node/e-class thresholds."""
    if min_enodes is not None:
        df = df.filter(
            pl.col("reach_enodes").is_not_null() & (pl.col("reach_enodes") >= min_enodes)
        )
    if min_enodes_pct is not None:
        df = df.filter(
            pl.col("reach_enodes_pct").is_not_null()
            & (pl.col("reach_enodes_pct") >= min_enodes_pct)
        )
    if min_eclasses is not None:
        df = df.filter(
            pl.col("reach_eclasses").is_not_null() & (pl.col("reach_eclasses") >= min_eclasses)
        )
    if min_eclasses_pct is not None:
        df = df.filter(
            pl.col("reach_eclasses_pct").is_not_null()
            & (pl.col("reach_eclasses_pct") >= min_eclasses_pct)
        )
    return df


def _safe_cost_ratio(col1: str, col2: str) -> pl.Expr:
    return (
        pl.when((pl.col(col1) == 0.0) & (pl.col(col2) == 0.0))
        .then(pl.lit(1.0))
        .when(pl.col(col2) == 0.0)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(col1) / pl.col(col2))
    )


def pairwise_comparison_df(
    df: pl.DataFrame,
    e1: str,
    e2: str,
    *,
    min_enodes: int | None = None,
    min_enodes_pct: float | None = None,
    min_eclasses: int | None = None,
    min_eclasses_pct: float | None = None,
) -> pl.DataFrame:
    """Join two extractors on (name, roots) and compute speedups, winners, and cost ratios.

    `speedup` is `max(micros_2, 1) / max(micros_1, 1)` (> 1 means `e1` is faster).
    `micros_ratio` is `max(micros_1, 1) / max(micros_2, 1)` (< 1 means `e1` is faster).
    """
    meta_cols = [
        c
        for c in (
            "name",
            "suite",
            "benchmark",
            "roots",
            "label",
            "n_roots",
            "total_enodes",
            "total_eclasses",
            "reach_enodes",
            "reach_eclasses",
            "reach_enodes_pct",
            "reach_eclasses_pct",
            "enodes_per_eclass",
            "avg_arity",
            "avg_child_eclasses",
            "avg_parent_enodes",
            "leaf_pct",
            "max_depth",
            "cyclic_eclasses_pct",
        )
        if c in df.columns
    ]

    d1 = df.filter(pl.col("extractor") == e1).select(
        *meta_cols,
        pl.col("micros").alias("micros_1"),
        pl.col("tree").alias("tree_1"),
        pl.col("dag").alias("dag_1"),
    )
    d2 = df.filter(pl.col("extractor") == e2).select(
        "name",
        "roots",
        pl.col("micros").alias("micros_2"),
        pl.col("tree").alias("tree_2"),
        pl.col("dag").alias("dag_2"),
    )

    joined = d1.join(d2, on=["name", "roots"], how="inner")
    joined = filter_egraph_stats(
        joined,
        min_enodes=min_enodes,
        min_enodes_pct=min_enodes_pct,
        min_eclasses=min_eclasses,
        min_eclasses_pct=min_eclasses_pct,
    )

    t1c = pl.max_horizontal(pl.col("micros_1"), pl.lit(1)).cast(pl.Float64)
    t2c = pl.max_horizontal(pl.col("micros_2"), pl.lit(1)).cast(pl.Float64)

    return joined.with_columns(
        (t2c / t1c).alias("speedup"),
        (t1c / t2c).alias("micros_ratio"),
        pl.when(t1c < t2c)
        .then(pl.lit(e1))
        .when(t2c < t1c)
        .then(pl.lit(e2))
        .otherwise(pl.lit("tie"))
        .alias("winner"),
        pl.when(t1c < t2c)
        .then(t2c / t1c)
        .when(t2c < t1c)
        .then(t1c / t2c)
        .otherwise(pl.lit(1.0))
        .alias("winner_ratio"),
        _safe_cost_ratio("tree_1", "tree_2").alias("tree_ratio"),
        _safe_cost_ratio("dag_1", "dag_2").alias("dag_ratio"),
        (pl.col("tree_1") - pl.col("tree_2")).alias("tree_diff"),
        (pl.col("dag_1") - pl.col("dag_2")).alias("dag_diff"),
    )


def all_pairs_summary_df(df: pl.DataFrame) -> pl.DataFrame:
    """Compute pairwise geometric means, quantiles, and win rates for all extractor pairs.

    For each ordered pair (e1, e2):
    - `geomean_speedup` > 1 means `e1` is faster than `e2` (geomean of e2/e1).
    - `geomean_micros_ratio` < 1 means `e1` is faster than `e2` (geomean of e1/e2).
    """
    extractors = sorted(df["extractor"].unique().to_list())
    rows: list[dict[str, Any]] = []

    for e1 in extractors:
        for e2 in extractors:
            if e1 == e2:
                n = df.filter(pl.col("extractor") == e1).height
                rows.append(
                    {
                        "e1": e1,
                        "e2": e2,
                        "n_benchmarks": n,
                        "geomean_speedup": 1.0,
                        "geomean_micros_ratio": 1.0,
                        "q25_speedup": 1.0,
                        "median_speedup": 1.0,
                        "q75_speedup": 1.0,
                        "geomean_tree_ratio": 1.0,
                        "geomean_dag_ratio": 1.0,
                        "tree_diffs": 0,
                        "dag_diffs": 0,
                        "e1_win_rate": 0.0,
                        "tie_rate": 100.0,
                    }
                )
                continue

            pair = pairwise_comparison_df(df, e1, e2)
            if pair.is_empty():
                continue

            pos_tree = pair.filter(
                pl.col("tree_ratio").is_not_null()
                & pl.col("tree_ratio").is_finite()
                & (pl.col("tree_ratio") > 0)
            )
            pos_dag = pair.filter(
                pl.col("dag_ratio").is_not_null()
                & pl.col("dag_ratio").is_finite()
                & (pl.col("dag_ratio") > 0)
            )

            stats = pair.select(
                pl.col("speedup").log().mean().exp().alias("gm_speedup"),
                pl.col("micros_ratio").log().mean().exp().alias("gm_ratio"),
                pl.col("speedup").quantile(0.25).alias("q25"),
                pl.col("speedup").median().alias("median"),
                pl.col("speedup").quantile(0.75).alias("q75"),
                (pl.col("tree_diff").abs() > 1e-9).sum().alias("tree_diffs"),
                (pl.col("dag_diff").abs() > 1e-9).sum().alias("dag_diffs"),
                (pl.col("winner") == e1).mean().alias("win_rate"),
                (pl.col("winner") == "tie").mean().alias("tie_rate"),
            ).row(0, named=True)

            gm_tree = (
                pos_tree.select(pl.col("tree_ratio").log().mean().exp()).item()
                if not pos_tree.is_empty()
                else None
            )
            gm_dag = (
                pos_dag.select(pl.col("dag_ratio").log().mean().exp()).item()
                if not pos_dag.is_empty()
                else None
            )

            rows.append(
                {
                    "e1": e1,
                    "e2": e2,
                    "n_benchmarks": pair.height,
                    "geomean_speedup": float(stats["gm_speedup"]),
                    "geomean_micros_ratio": float(stats["gm_ratio"]),
                    "q25_speedup": float(stats["q25"]),
                    "median_speedup": float(stats["median"]),
                    "q75_speedup": float(stats["q75"]),
                    "geomean_tree_ratio": float(gm_tree) if gm_tree is not None else None,
                    "geomean_dag_ratio": float(gm_dag) if gm_dag is not None else None,
                    "tree_diffs": int(stats["tree_diffs"]),
                    "dag_diffs": int(stats["dag_diffs"]),
                    "e1_win_rate": 100.0 * float(stats["win_rate"]),
                    "tie_rate": 100.0 * float(stats["tie_rate"]),
                }
            )

    return pl.DataFrame(rows)


def filter_results(
    df: pl.DataFrame,
    *,
    exclude_suites: Sequence[str] = (),
    include_suites: Sequence[str] = (),
    extractors: Sequence[str] = (),
    min_enodes: int | None = None,
    min_enodes_pct: float | None = None,
    min_eclasses: int | None = None,
    min_eclasses_pct: float | None = None,
) -> pl.DataFrame:
    """Filter benchmark results by suites, extractors, and e-graph reachability thresholds."""
    if include_suites:
        df = df.filter(pl.col("suite").is_in(list(include_suites)))
    if exclude_suites:
        df = df.filter(~pl.col("suite").is_in(list(exclude_suites)))
    if extractors:
        df = df.filter(pl.col("extractor").is_in(list(extractors)))
    return filter_egraph_stats(
        df,
        min_enodes=min_enodes,
        min_enodes_pct=min_enodes_pct,
        min_eclasses=min_eclasses,
        min_eclasses_pct=min_eclasses_pct,
    )


def suite_counts_df(df: pl.DataFrame) -> pl.DataFrame:
    """Count unique benchmark instances per suite, sorted descending."""
    return (
        df.select("suite", "name", "roots")
        .unique()
        .group_by("suite")
        .len(name="instances")
        .sort("instances", descending=True)
    )


def extractor_summary_df(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate runtime and cost summary metrics per extractor, sorted by total runtime."""
    return (
        df.group_by("extractor")
        .agg(
            pl.len().alias("benchmarks"),
            (pl.col("micros").sum() / 1_000.0).alias("total_ms"),
            pl.col("micros").median().alias("median_us"),
            pl.max_horizontal(pl.col("micros"), pl.lit(1))
            .cast(pl.Float64)
            .log()
            .mean()
            .exp()
            .alias("geomean_us"),
            pl.col("tree").sum().alias("total_tree_cost"),
            pl.col("dag").sum().alias("total_dag_cost"),
        )
        .sort("total_ms")
    )


def speedup_vs_baseline_df(df: pl.DataFrame, baseline: str) -> pl.DataFrame:
    """Compute per-benchmark speedup of each non-baseline extractor against `baseline`."""
    base_times = df.filter(pl.col("extractor") == baseline).select(
        "name",
        "roots",
        pl.max_horizontal(pl.col("micros"), pl.lit(1)).cast(pl.Float64).alias("base_micros"),
    )
    return (
        df.filter(pl.col("extractor") != baseline)
        .join(base_times, on=["name", "roots"], how="inner")
        .with_columns(
            (
                pl.col("base_micros")
                / pl.max_horizontal(pl.col("micros"), pl.lit(1)).cast(pl.Float64)
            ).alias("speedup_vs_base")
        )
    )


def cost_comparison_df(df: pl.DataFrame, baseline: str) -> pl.DataFrame:
    """Compare tree and DAG costs of each non-baseline extractor against `baseline`."""
    base_costs = df.filter(pl.col("extractor") == baseline).select(
        "name", "roots", pl.col("tree").alias("base_tree"), pl.col("dag").alias("base_dag")
    )
    return (
        df.filter(pl.col("extractor") != baseline)
        .join(base_costs, on=["name", "roots"], how="inner")
        .with_columns(
            (pl.col("tree") - pl.col("base_tree")).abs().alias("abs_tree_diff"),
            (pl.col("dag") - pl.col("base_dag")).abs().alias("abs_dag_diff"),
            _safe_cost_ratio("tree", "base_tree").alias("tree_ratio"),
            _safe_cost_ratio("dag", "base_dag").alias("dag_ratio"),
        )
    )


def cost_summary_df(df: pl.DataFrame, baseline: str) -> pl.DataFrame:
    """Summarize tree and DAG cost differences and geomean ratios vs. `baseline`."""
    cost_cmp = cost_comparison_df(df, baseline)
    return (
        cost_cmp.group_by("extractor")
        .agg(
            (pl.col("abs_tree_diff") > 1e-9).sum().alias("tree_diffs"),
            (pl.col("abs_dag_diff") > 1e-9).sum().alias("dag_diffs"),
            pl.col("tree_ratio")
            .filter(pl.col("tree_ratio") > 0)
            .log()
            .mean()
            .exp()
            .alias("geomean_tree_ratio"),
            pl.col("dag_ratio")
            .filter(pl.col("dag_ratio") > 0)
            .log()
            .mean()
            .exp()
            .alias("geomean_dag_ratio"),
        )
        .sort("extractor")
    )


@dataclass(frozen=True)
class PairSummary:
    """Headline win/loss and speedup statistics for a pairwise extractor comparison."""

    e1: str
    e2: str
    n_total: int
    e1_wins: int
    e2_wins: int
    ties: int
    geomean_speedup: float
    median_speedup: float
    q25_speedup: float
    q75_speedup: float

    def format(self) -> str:
        n = self.n_total
        return (
            f"{self.e1} vs {self.e2}  ({n:,} benchmarks)\n"
            f"  Wins: {self.e1}: {self.e1_wins:,} ({100 * self.e1_wins / n:.1f}%), "
            f"{self.e2}: {self.e2_wins:,} ({100 * self.e2_wins / n:.1f}%), "
            f"ties: {self.ties:,} ({100 * self.ties / n:.1f}%)\n"
            f"  Speedup ({self.e2} / {self.e1}): "
            f"geomean = {self.geomean_speedup:.2f}x, "
            f"median = {self.median_speedup:.2f}x "
            f"(IQR: {self.q25_speedup:.2f}x – {self.q75_speedup:.2f}x)"
        )


def summarize_pair(pair_df: pl.DataFrame, e1: str, e2: str) -> PairSummary:
    """Compute headline win/loss and speedup statistics from a `pairwise_comparison_df`."""
    if pair_df.is_empty():
        msg = f"No shared benchmarks found for '{e1}' and '{e2}'"
        raise ValueError(msg)

    stats = pair_df.select(
        (pl.col("winner") == e1).sum().alias("e1_wins"),
        (pl.col("winner") == e2).sum().alias("e2_wins"),
        (pl.col("winner") == "tie").sum().alias("ties"),
        pl.col("speedup").log().mean().exp().alias("gm_speedup"),
        pl.col("speedup").median().alias("med_speedup"),
        pl.col("speedup").quantile(0.25).alias("q25_speedup"),
        pl.col("speedup").quantile(0.75).alias("q75_speedup"),
    ).row(0, named=True)

    return PairSummary(
        e1=e1,
        e2=e2,
        n_total=pair_df.height,
        e1_wins=int(stats["e1_wins"]),
        e2_wins=int(stats["e2_wins"]),
        ties=int(stats["ties"]),
        geomean_speedup=float(stats["gm_speedup"]),
        median_speedup=float(stats["med_speedup"]),
        q25_speedup=float(stats["q25_speedup"]),
        q75_speedup=float(stats["q75_speedup"]),
    )


def top_outliers_df(
    pair_df: pl.DataFrame, e1: str, e2: str, *, top_n: int = 15, max_label_len: int = 45
) -> pl.DataFrame:
    """Return the top `top_n` biggest speedups and slowdowns with unique short display labels."""
    top_wins = pair_df.filter(pl.col("winner") == e1).sort("speedup", descending=True).head(top_n)
    top_losses = pair_df.filter(pl.col("winner") == e2).sort("speedup").head(top_n)

    def shorten(label: str) -> str:
        if len(label) <= max_label_len:
            return label
        return "..." + label[-(max_label_len - 3) :]

    outliers = (
        pl.concat([top_wins, top_losses])
        .with_columns(
            pl.col("speedup").log(base=10).alias("log10_speedup"),
            pl.col("label").map_elements(shorten, return_dtype=pl.String).alias("short_label"),
        )
        .sort("log10_speedup")
    )
    return outliers.with_columns(
        pl.when(pl.col("short_label").is_duplicated())
        .then(pl.col("short_label") + " (#" + pl.int_range(pl.len()).cast(pl.String) + ")")
        .otherwise(pl.col("short_label"))
        .alias("short_label")
    )


def tree_cost_mismatches_df(
    df: pl.DataFrame, *, extractors: Sequence[str] = TREE_EXTRACTORS, tol: float = 1e-9
) -> pl.DataFrame:
    """Return benchmark instances where optimal tree extractors disagree on `tree` cost."""
    tree_df = df.filter(pl.col("extractor").is_in(list(extractors)))
    if tree_df.is_empty():
        return pl.DataFrame(
            schema={"label": pl.String, "min_tree": pl.Float64, "max_tree": pl.Float64}
        )

    spread = (
        tree_df.group_by("name", "roots", "label")
        .agg(
            pl.col("tree").min().alias("min_tree"),
            pl.col("tree").max().alias("max_tree"),
            pl.col("tree").is_nan().any().alias("has_nan"),
        )
        .filter(pl.col("has_nan") | ((pl.col("max_tree") - pl.col("min_tree")) > tol))
        .with_columns((pl.col("max_tree") - pl.col("min_tree")).alias("diff"))
        .drop("has_nan")
    )
    if spread.is_empty():
        return spread.select("label", "min_tree", "max_tree", "diff")

    pivoted = tree_df.join(spread.select("name", "roots"), on=["name", "roots"], how="semi").pivot(
        on="extractor", index=["name", "roots", "label"], values="tree"
    )
    return (
        spread.join(pivoted, on=["name", "roots", "label"], how="left")
        .drop("name", "roots")
        .sort("diff", descending=True)
    )


def check_tree_extractors(
    df: pl.DataFrame, *, extractors: Sequence[str] = TREE_EXTRACTORS, tol: float = 1e-9
) -> pl.DataFrame | None:
    """Sanity-check that all present tree extractors agree on `tree` cost.

    Displays a loud warning banner and returns the mismatching rows if any disagree;
    otherwise displays a brief confirmation and returns `None`.
    """
    present = [e for e in extractors if e in set(df["extractor"].unique().to_list())]
    mismatches = tree_cost_mismatches_df(df, extractors=present, tol=tol)
    n_instances = (
        df.filter(pl.col("extractor").is_in(present)).select("name", "roots").n_unique()
        if present
        else 0
    )

    try:
        from IPython.core.getipython import get_ipython
        from IPython.display import HTML, display

        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False
        HTML = None
        display = None

    if mismatches.is_empty():
        msg = (
            f"Tree cost sanity check passed: all {len(present)} tree extractors "
            f"({', '.join(present)}) agree across {n_instances:,} benchmarks."
        )
        if in_notebook and display is not None and HTML is not None:
            display(
                HTML(
                    "<div style='padding:10px 14px;border-left:5px solid #2e7d32;"
                    "background:#edf7ed;color:#1b5e20;font-weight:500;border-radius:4px'>"
                    f"&#10004; {msg}</div>"
                )
            )
        else:
            print(f"OK: {msg}")
        return None

    banner = (
        f"TREE COST MISMATCH DETECTED: {mismatches.height:,} / {n_instances:,} benchmarks "
        f"differ across tree extractors ({', '.join(present)})!"
    )
    if in_notebook and display is not None and HTML is not None:
        display(
            HTML(
                "<div style='padding:14px 18px;border:2px solid #b71c1c;"
                "border-left:8px solid #b71c1c;background:#fdecea;color:#7f0000;"
                "font-size:15px;font-weight:700;border-radius:4px'>"
                f"&#9888;&#65039; {banner}</div>"
            )
        )
    else:
        border = "!" * max(len(banner) + 8, 60)
        print(f"\n{border}\n!!! {banner} !!!\n{border}\n", file=sys.stderr)
    return mismatches


def pair_table_df(
    pair_df: pl.DataFrame,
    e1: str,
    e2: str,
    *,
    sort_by: str = "ratio_x",
    descending: bool = True,
    head: int | None = None,
) -> pl.DataFrame:
    """Format a pairwise comparison DataFrame as a compact per-benchmark table."""
    cols: list[pl.Expr | str] = [
        "label",
        "winner",
        pl.col("winner_ratio").round(2).alias("ratio_x"),
        pl.col("micros_1").alias(f"{e1}_us"),
        pl.col("micros_2").alias(f"{e2}_us"),
    ]
    if "reach_enodes" in pair_df.columns:
        cols.append("reach_enodes")
    if "reach_enodes_pct" in pair_df.columns:
        cols.append(pl.col("reach_enodes_pct").round(1).alias("reach_enodes_pct"))
    if "reach_eclasses" in pair_df.columns:
        cols.append("reach_eclasses")
    if "reach_eclasses_pct" in pair_df.columns:
        cols.append(pl.col("reach_eclasses_pct").round(1).alias("reach_eclasses_pct"))

    table = pair_df.select(cols).sort(sort_by, descending=descending)
    return table.head(head) if head is not None else table


def pair_feature_correlations_df(
    pair_df: pl.DataFrame, *, features: Mapping[str, str] = STRUCTURE_FEATURES
) -> pl.DataFrame:
    """Compute Spearman rank correlation between each structural feature and pairwise speedup."""
    rows: list[dict[str, Any]] = []
    for col, label in features.items():
        if col not in pair_df.columns:
            continue
        valid = pair_df.filter(pl.col(col).is_not_null() & pl.col(col).cast(pl.Float64).is_finite())
        if valid.height < 3:
            continue
        # Constant features have undefined Spearman correlation
        if valid[col].n_unique() < 2:
            continue
        corr = valid.select(
            pl.corr(pl.col(col).cast(pl.Float64), pl.col("speedup"), method="spearman")
        ).item()
        if not isinstance(corr, int | float) or math.isnan(corr):
            continue
        rows.append(
            {"feature": col, "label": label, "spearman_r": float(corr), "n_valid": valid.height}
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "label": pl.String,
                "spearman_r": pl.Float64,
                "n_valid": pl.Int64,
            }
        )
    return pl.DataFrame(rows).sort("spearman_r")
