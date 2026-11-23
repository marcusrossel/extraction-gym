#!/usr/bin/env python3

"""Compare the speed of two extractors across all benchmarks.

If no JSON files are given, all files in output/**/*.json are used.
"""

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import tyro

REPO_ROOT: Path = Path(__file__).resolve().parent.parent


class Row(NamedTuple):
    label: str
    winner: str | None
    ratio: float
    t1: int
    t2: int
    total_enodes: int | None
    total_eclasses: int | None
    reach_enodes: int | None
    reach_eclasses: int | None


type EClassChildren = dict[str, list[str]]
type EClassSize = dict[str, int]
type EGraphData = tuple[EClassChildren, EClassSize] | tuple[None, None]
type EGraphStats = tuple[int, int, int, int] | tuple[None, None, None, None]


def load_jsons(files: Sequence[str | Path]) -> list[dict[str, Any]]:
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
                        "micros": result["micros"],
                    }
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error loading {file}: {e}", file=sys.stderr)
    return entries


def load_egraph(name: str) -> EGraphData:
    try:
        path = Path(name)
        if not path.exists():
            path = REPO_ROOT / name
        with path.open() as f:
            nodes: dict[str, dict[str, Any]] = json.load(f)["nodes"]
        # Pre-build index: eclass -> list of child eclasses (across all member nodes)
        eclass_children: EClassChildren = {}
        for n in nodes.values():
            ec: str = n["eclass"]
            children = eclass_children.setdefault(ec, [])
            for child_id in n["children"]:
                children.append(nodes[child_id]["eclass"])
        # eclass -> node count, for reachable enodes computation
        eclass_size: EClassSize = {}
        for n in nodes.values():
            ec = n["eclass"]
            eclass_size[ec] = eclass_size.get(ec, 0) + 1
        return eclass_children, eclass_size
    except OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError:
        return None, None


def egraph_stats(
    eclass_children: EClassChildren | None, eclass_size: EClassSize | None, roots: Sequence[str]
) -> EGraphStats:
    if eclass_children is None or eclass_size is None:
        return None, None, None, None
    total_eclasses = len(eclass_children)
    total_enodes = sum(eclass_size.values())

    visited = set(roots)
    queue = list(roots)
    while queue:
        ec = queue.pop()
        for child_ec in eclass_children.get(ec, []):
            if child_ec not in visited:
                visited.add(child_ec)
                queue.append(child_ec)
    reach_eclasses = len(visited)
    reach_enodes = sum(eclass_size.get(ec, 0) for ec in visited)

    return total_enodes, total_eclasses, reach_enodes, reach_eclasses


def compare(
    entries: Sequence[dict[str, Any]],
    e1: str,
    e2: str,
    min_enodes: int | None = None,
    min_enodes_pct: float | None = None,
    min_eclasses: int | None = None,
    min_eclasses_pct: float | None = None,
    sort_by: str = "benchmark",
) -> None:
    # Index by (name, roots) -> extractor -> micros
    index: dict[tuple[str, tuple[str, ...]], dict[str, int]] = {}
    for entry in entries:
        if entry["extractor"] not in (e1, e2):
            continue
        key: tuple[str, tuple[str, ...]] = (entry["name"], entry["roots"])
        index.setdefault(key, {})[entry["extractor"]] = entry["micros"]

    roots_per_name: dict[str, set[tuple[str, ...]]] = {}
    for name, roots in index:
        roots_per_name.setdefault(name, set()).add(roots)

    nodes_cache = {}

    rows = []
    skipped = 0
    for key, times in index.items():
        if e1 not in times or e2 not in times:
            skipped += 1
            continue
        t1, t2 = times[e1], times[e2]
        # Avoid division by zero; treat 0 micros as 1 for ratio purposes
        t1c, t2c = max(t1, 1), max(t2, 1)
        if t1c < t2c:
            winner = e1
            ratio = t2c / t1c
        elif t2c < t1c:
            winner = e2
            ratio = t1c / t2c
        else:
            winner = None
            ratio = 1.0
        name, roots = key
        if name not in nodes_cache:
            nodes_cache[name] = load_egraph(name)
        eclass_children, eclass_size = nodes_cache[name]
        total_enodes, total_eclasses, reach_enodes, reach_eclasses = egraph_stats(
            eclass_children, eclass_size, roots
        )
        if min_enodes is not None and (reach_enodes is None or reach_enodes < min_enodes):
            continue
        if min_enodes_pct is not None and (
            reach_enodes is None
            or total_enodes is None
            or total_enodes == 0
            or 100 * reach_enodes / total_enodes < min_enodes_pct
        ):
            continue
        if min_eclasses is not None and (reach_eclasses is None or reach_eclasses < min_eclasses):
            continue
        if min_eclasses_pct is not None and (
            reach_eclasses is None
            or total_eclasses is None
            or total_eclasses == 0
            or 100 * reach_eclasses / total_eclasses < min_eclasses_pct
        ):
            continue
        display_name = name.removeprefix("data/").removesuffix(".json")
        roots_str = " [" + ", ".join(roots) + "]" if len(roots_per_name[name]) > 1 else ""
        label = f"{display_name}{roots_str}"
        rows.append(
            Row(
                label,
                winner,
                ratio,
                t1,
                t2,
                total_enodes,
                total_eclasses,
                reach_enodes,
                reach_eclasses,
            )
        )

    if skipped:
        print(f"(skipped {skipped} benchmarks missing one extractor)\n", file=sys.stderr)

    if not rows:
        print(f"No benchmarks found for both '{e1}' and '{e2}'.")
        return

    match sort_by:
        case "benchmark":
            rows.sort(key=lambda r: r.label)
        case "winner":
            rows.sort(key=lambda r: (r.winner is None, r.winner or ""))
        case "speedup":
            rows.sort(key=lambda r: r.ratio)
        case _ if sort_by == e1:
            rows.sort(key=lambda r: r.t1)
        case _ if sort_by == e2:
            rows.sort(key=lambda r: r.t2)
        case "e-nodes":
            rows.sort(key=lambda r: (r.reach_enodes is None, r.reach_enodes or 0))
        case "e-nodes-pct":
            rows.sort(
                key=lambda r: (
                    r.reach_enodes is None or not r.total_enodes,
                    r.reach_enodes / r.total_enodes
                    if r.reach_enodes is not None and r.total_enodes
                    else 0.0,
                )
            )
        case "e-classes":
            rows.sort(key=lambda r: (r.reach_eclasses is None, r.reach_eclasses or 0))
        case "e-classes-pct":
            rows.sort(
                key=lambda r: (
                    r.reach_eclasses is None or not r.total_eclasses,
                    r.reach_eclasses / r.total_eclasses
                    if r.reach_eclasses is not None and r.total_eclasses
                    else 0.0,
                )
            )
        case _:
            msg = f"Invalid sort key: {sort_by}"
            raise ValueError(msg)

    e1_wins = sum(1 for r in rows if r.winner == e1)
    e2_wins = sum(1 for r in rows if r.winner == e2)
    ties = sum(1 for r in rows if r.winner is None)

    def fmt_reach(reach: int | None, total: int | None) -> str:
        if reach is None or total is None or total == 0:
            return "?"
        pct_val = 100.0 * reach / total
        return f"{reach} ({pct_val:.1f}%)"

    label_w = max(len(r.label) for r in rows)
    name_w = max(len(e1), len(e2), len("tie"))
    en_w = max(len(fmt_reach(r.reach_enodes, r.total_enodes)) for r in rows)
    ec_w = max(len(fmt_reach(r.reach_eclasses, r.total_eclasses)) for r in rows)
    en_w = max(en_w, len("e-nodes (reach %)"))
    ec_w = max(ec_w, len("e-classes (reach %)"))

    header = f"{'Benchmark':<{label_w}}  {'Winner':<{name_w}}  {'Speedup':>10}  {e1 + ' µs':>12}  {e2 + ' µs':>12}  {'e-nodes (reach %)':>{en_w}}  {'e-classes (reach %)':>{ec_w}}"
    print(header)
    print("-" * len(header))

    for label, winner, ratio, t1, t2, total_en, total_ec, reach_en, reach_ec in rows:
        w = winner if winner is not None else "tie"
        en = fmt_reach(reach_en, total_en)
        ec = fmt_reach(reach_ec, total_ec)
        print(
            f"{label:<{label_w}}  {w:<{name_w}}  {ratio:>9.2f}x  {t1:>12}  {t2:>12}  {en:>{en_w}}  {ec:>{ec_w}}"
        )

    total = len(rows)

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%"

    print()
    print(f"Wins: {e1}: {e1_wins} ({pct(e1_wins)}), {e2}: {e2_wins} ({pct(e2_wins)})")
    if ties:
        print(f"Ties: {ties} ({pct(ties)})")
    print(f"Total: {total} benchmarks")


@dataclass(frozen=True)
class CompareArgs:
    extractor1: tyro.conf.Positional[str]
    """First extractor name."""

    extractor2: tyro.conf.Positional[str]
    """Second extractor name."""

    files: tyro.conf.Positional[tuple[Path, ...]] = ()
    """JSON result files to compare (default: output/**/*.json)."""

    min_enodes: int | None = None
    """Only include benchmarks with at least N reachable e-nodes."""

    min_enodes_pct: float | None = None
    """Only include benchmarks where at least P% of e-nodes are reachable."""

    min_eclasses: int | None = None
    """Only include benchmarks with at least N reachable e-classes."""

    min_eclasses_pct: float | None = None
    """Only include benchmarks where at least P% of e-classes are reachable."""

    sort: str = "benchmark"
    """Sort by: benchmark, winner, speedup, e-nodes, e-nodes-pct, e-classes, e-classes-pct, or an extractor name."""


def main(args: CompareArgs) -> None:
    valid_sort: set[str] = {
        "benchmark",
        "winner",
        "speedup",
        "e-nodes",
        "e-nodes-pct",
        "e-classes",
        "e-classes-pct",
        args.extractor1,
        args.extractor2,
    }
    if args.sort not in valid_sort:
        print(f"error: --sort must be one of: {', '.join(sorted(valid_sort))}", file=sys.stderr)
        sys.exit(2)

    files: list[Path] = (
        list(args.files)
        or list(Path("output").rglob("*.json"))
        or list((REPO_ROOT / "output").rglob("*.json"))
    )
    entries = load_jsons(files)
    if not entries:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    compare(
        entries,
        args.extractor1,
        args.extractor2,
        args.min_enodes,
        args.min_enodes_pct,
        args.min_eclasses,
        args.min_eclasses_pct,
        args.sort,
    )


if __name__ == "__main__":
    main(tyro.cli(CompareArgs, description=__doc__))
