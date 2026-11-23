"""Reusable plotnine charts for extraction-gym benchmark analysis."""

import copy
import math
import operator
from collections.abc import Callable, Sequence
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
import polars as pl
from plotnine.composition import Compose

from scripts.results import (
    STRUCTURE_FEATURES,
    extractor_summary_df,
    pair_feature_correlations_df,
    speedup_vs_baseline_df,
    suite_extractor_summary_df,
    suite_labels_df,
    top_outliers_df,
)

DEFAULT_PLOTS_DIR: Path = Path("plots")

_COLOR_FASTER = "#2b83ba"
_COLOR_SLOWER = "#d7191c"
_COLOR_TIE = "#bababa"
_COLOR_BASELINE = "#777777"

_DEFAULT_REGIME_FEATURES: tuple[tuple[str, bool], ...] = (
    ("enodes_per_eclass", True),
    ("avg_arity", True),
    ("avg_child_eclasses", True),
    ("reach_enodes_pct", False),
    ("max_depth", True),
    ("cyclic_eclasses_pct", False),
)


# --- Theme, Saving & Shared Helpers ---


def save_plot(
    plot: p9.ggplot | Compose,
    name: str,
    *,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    fmt: str = "pdf",
) -> Path:
    """Save a plotnine `ggplot` or `Compose` figure to `plots_dir/<name>.<fmt>`."""
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{fmt}"
    out_path = out_dir / (name if name.endswith(suffix) else f"{name}{suffix}")
    if isinstance(plot, p9.ggplot):
        plot.save(out_path, format=fmt, verbose=False)
    else:
        fig = copy.deepcopy(plot).draw()
        try:
            fig.savefig(out_path, format=fmt)
        finally:
            plt.close(fig)
    return out_path


def to_pd(
    frame: pl.DataFrame, *, ordered_col: tuple[str, Sequence[str]] | None = None
) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas without pyarrow, optionally ordering one column."""
    pdf = pd.DataFrame(frame.to_dict(as_series=False))
    if ordered_col is not None:
        col, categories = ordered_col
        pdf[col] = pd.Categorical(pdf[col], categories=list(categories), ordered=True)
    return pdf


def default_theme() -> p9.theme:
    """Return the default plotnine theme used across analysis plots."""
    return p9.theme_minimal(base_size=11) + p9.theme(
        plot_title=p9.element_text(weight="bold", size=13),
        plot_subtitle=p9.element_text(color="#555555", size=10),
        legend_position="bottom",
    )


def set_default_theme() -> None:
    """Install the default extraction-gym plotnine theme globally."""
    p9.theme_set(default_theme())


def winner_palette(e1: str, e2: str) -> tuple[dict[str, str], list[str]]:
    """Return consistent color mapping and legend ordering for pairwise winner plots."""
    return {e1: _COLOR_FASTER, e2: _COLOR_SLOWER, "tie": _COLOR_TIE}, [e1, "tie", e2]


def _resolve_extractor_order(df: pl.DataFrame, extractor_order: Sequence[str] | None) -> list[str]:
    if extractor_order is not None:
        return list(extractor_order)
    if "extractor" in df.columns:
        return extractor_summary_df(df)["extractor"].to_list()
    return sorted(df["e1"].unique().to_list())


def _map_str(col: str, fn: Callable[[float], str]) -> pl.Expr:
    return pl.col(col).map_elements(fn, return_dtype=pl.String)


def _format_multiplier(value: float) -> str:
    """Format a speedup or runtime multiplier compactly for plot labels."""
    if value >= 100.0:
        return f"{value:.0f}x"
    if value >= 10.0:
        return f"{value:.1f}x"
    if value >= 0.01:
        return f"{value:.2f}x"
    if value >= 0.001:
        return f"{value:.3f}x"
    return "<0.001x"


def _format_duration_ms(ms: float) -> str:
    """Format a cumulative duration in milliseconds as a concise human-readable string."""
    if ms >= 10_000.0:
        return f"{ms / 1_000.0:.0f}s"
    if ms >= 1_000.0:
        return f"{ms / 1_000.0:.1f}s"
    if ms >= 10.0:
        return f"{ms:.0f}ms"
    if ms >= 1.0:
        return f"{ms:.1f}ms"
    return f"{ms * 1_000.0:.0f}µs"


def _diverging_fill(name: str, clamp_log2: float) -> p9.scale_fill_gradient2:
    return p9.scale_fill_gradient2(
        low="#d6604d",
        mid="#f7f7f7",
        high="#4393c3",
        midpoint=0.0,
        limits=(-clamp_log2, clamp_log2),
        name=name,
    )


def _filter_finite(
    df: pl.DataFrame, cols: Sequence[str], *, positive_cols: Sequence[str] = ()
) -> pl.DataFrame:
    for col in cols:
        df = df.filter(pl.col(col).is_not_null() & pl.col(col).cast(pl.Float64).is_finite())
    for col in positive_cols:
        df = df.filter(pl.col(col) > 0)
    return df


# --- Part 1: All-Extractor Overview ---


def plot_pairwise_heatmap(
    pairs_df: pl.DataFrame, *, extractor_order: Sequence[str] | None = None, clamp_log2: float = 4.0
) -> p9.ggplot:
    """Combined heatmap of pairwise geometric-mean speedups and strict win rates."""
    order = _resolve_extractor_order(pairs_df, extractor_order)
    heatmap_df = pairs_df.with_columns(
        pl.col("geomean_speedup")
        .log(base=2)
        .clip(-clamp_log2, clamp_log2)
        .alias("log2_speedup_clamped"),
        pl.when(pl.col("e1") == pl.col("e2"))
        .then(pl.lit("—"))
        .otherwise(
            _map_str("geomean_speedup", lambda v: f"{v:.2f}x")
            + "\n("
            + _map_str("e1_win_rate", lambda v: f"{v:.0f}%")
            + ")"
        )
        .alias("label"),
    )

    return (
        p9.ggplot(to_pd(heatmap_df), p9.aes(x="e2", y="e1", fill="log2_speedup_clamped"))
        + p9.geom_tile(color="white", size=0.8)
        + p9.geom_text(p9.aes(label="label"), size=8.5, color="#111111", va="center")
        + p9.scale_x_discrete(limits=order)
        + p9.scale_y_discrete(limits=list(reversed(order)))
        + _diverging_fill("log2(speedup)", clamp_log2)
        + p9.labs(
            title="Pairwise Geometric Mean Speedup & Win Rate (Row vs. Column)",
            subtitle="Cell: geomean speedup and (row strict win %); > 1.0x (blue) means row is faster",
            x="Baseline (Column)",
            y="Extractor (Row)",
        )
        + p9.theme(
            figure_size=(10.5, 7.0),
            axis_text_x=p9.element_text(rotation=30, ha="right"),
            panel_grid=p9.element_blank(),
        )
    )


def plot_performance_profile(df: pl.DataFrame) -> p9.ggplot:
    """Dolan–Moré performance profile across all extractors."""
    clamped = df.with_columns(pl.col("micros").clip(lower_bound=1).cast(pl.Float64).alias("_us"))
    best_per_inst = clamped.group_by("name", "roots").agg(pl.col("_us").min().alias("_best"))
    profile_df = (
        clamped.join(best_per_inst, on=["name", "roots"], how="inner")
        .with_columns((pl.col("_us") / pl.col("_best")).alias("tau"))
        .sort("extractor", "tau")
        .with_columns(
            (pl.col("tau").cum_count().over("extractor") / pl.len().over("extractor")).alias(
                "fraction_solved"
            )
        )
    )

    return (
        p9.ggplot(to_pd(profile_df), p9.aes(x="tau", y="fraction_solved", color="extractor"))
        + p9.geom_step(size=1.0, alpha=0.9)
        + p9.scale_x_log10()
        + p9.labs(
            title="Dolan–Moré Performance Profile",
            subtitle="Fraction of benchmarks solved within factor τ of the fastest extractor (higher & further left is better)",
            x="Slowdown Factor τ vs. Fastest Extractor (log scale)",
            y="Fraction of Benchmarks",
            color="Extractor",
        )
        + p9.theme(figure_size=(10, 5.5))
    )


def _plot_single_regime_panel(
    rel_df: pl.DataFrame,
    baseline: str,
    feature: str,
    *,
    log_x: bool,
    n_bins: int,
    min_bin_count: int,
    show_y_label: bool,
    legend_position: str,
) -> p9.ggplot:
    feat_df = _filter_finite(rel_df, [feature], positive_cols=[feature] if log_x else ())

    # Assign quantile bins on unique benchmark instances so every extractor shares the same bins
    inst_bins = (
        feat_df.select("name", "roots", pl.col(feature).cast(pl.Float64).alias("_feat"))
        .unique(subset=["name", "roots"])
        .with_columns(
            ((pl.col("_feat").rank("average") - 1.0) * n_bins / pl.len())
            .floor()
            .clip(0, n_bins - 1)
            .cast(pl.Int64)
            .alias("bin")
        )
    )
    bin_meta = (
        inst_bins.group_by("bin")
        .agg(pl.len().alias("n_inst"), pl.col("_feat").median().alias("x_mid"))
        .filter(pl.col("n_inst") >= min_bin_count)
    )
    binned = (
        feat_df.join(inst_bins.select("name", "roots", "bin"), on=["name", "roots"], how="inner")
        .join(bin_meta, on="bin", how="inner")
        .group_by("bin", "x_mid", "extractor")
        .agg(pl.col("speedup_vs_base").log().mean().exp().alias("gm_speedup"))
        .sort("extractor", "x_mid")
    )

    label = STRUCTURE_FEATURES.get(feature, feature)
    p = (
        p9.ggplot(to_pd(binned), p9.aes(x="x_mid", y="gm_speedup", color="extractor"))
        + p9.geom_hline(yintercept=1.0, linetype="dashed", color="#333333")
        + p9.geom_line(size=0.9, alpha=0.9)
        + p9.geom_point(size=2.0)
        + p9.scale_y_log10()
        + p9.labs(
            title=label,
            x=f"{label} (log)" if log_x else label,
            y=f"Geomean Speedup vs. {baseline}" if show_y_label else "",
            color="",
        )
        + p9.theme(legend_position=legend_position)
    )
    return p + p9.scale_x_log10() if log_x else p


def plot_structure_regimes(
    df: pl.DataFrame,
    baseline: str,
    *,
    features: Sequence[tuple[str, bool]] = _DEFAULT_REGIME_FEATURES,
    n_bins: int = 8,
    min_bin_count: int = 10,
) -> Compose:
    """Grid of binned geomean speedup curves vs. e-graph structural metrics across extractors."""
    rel_df = speedup_vs_baseline_df(df, baseline)
    ncols = 3 if len(features) > 4 else 2
    nrows = math.ceil(len(features) / ncols)
    legend_idx = min((nrows - 1) * ncols + (ncols // 2), len(features) - 1)

    panels = [
        _plot_single_regime_panel(
            rel_df,
            baseline,
            feat,
            log_x=log_x,
            n_bins=n_bins,
            min_bin_count=min_bin_count,
            show_y_label=(idx % ncols == 0),
            legend_position="bottom" if idx == legend_idx else "none",
        )
        for idx, (feat, log_x) in enumerate(features)
    ]
    rows: list[p9.ggplot | Compose] = [
        reduce(operator.or_, panels[r * ncols : (r + 1) * ncols]) for r in range(nrows)
    ]
    grid = reduce(operator.truediv, rows)
    assert isinstance(grid, Compose)
    return grid & p9.theme(figure_size=(5.0 * ncols, 4.2 * nrows))


# --- Part 2: Per-Suite Performance Breakdown ---


def plot_suite_extractor_heatmap(
    df: pl.DataFrame,
    baseline: str,
    *,
    extractor_order: Sequence[str] | None = None,
    clamp_log2: float = 4.0,
) -> p9.ggplot:
    """Heatmap of per-suite geometric-mean speedup vs. `baseline` and fastest-extractor share."""
    ext_order = _resolve_extractor_order(df, extractor_order)
    suite_order = suite_labels_df(df)["suite_label"].to_list()
    heatmap_df = (
        suite_extractor_summary_df(df, baseline)
        .filter(pl.col("geomean_speedup").is_not_null())
        .with_columns(
            pl.col("geomean_speedup")
            .log(base=2)
            .clip(-clamp_log2, clamp_log2)
            .alias("log2_speedup_clamped"),
            (
                _map_str("geomean_speedup", _format_multiplier)
                + "\n("
                + _map_str("fastest_rate", lambda v: f"{v:.0f}%")
                + ")"
            ).alias("label"),
        )
    )

    return (
        p9.ggplot(
            to_pd(heatmap_df), p9.aes(x="extractor", y="suite_label", fill="log2_speedup_clamped")
        )
        + p9.geom_tile(color="white", size=0.8)
        + p9.geom_text(p9.aes(label="label"), size=8.0, color="#111111", va="center")
        + p9.scale_x_discrete(limits=ext_order)
        + p9.scale_y_discrete(limits=list(reversed(suite_order)))
        + _diverging_fill(f"log2(speedup vs. {baseline})", clamp_log2)
        + p9.labs(
            title=f"Per-Suite Performance vs. {baseline}",
            subtitle=(
                "Cell: geomean speedup and (% fastest in suite, incl. ties); "
                "> 1.0x (blue) is faster than baseline"
            ),
            x="Extractor",
            y="Suite",
        )
        + p9.theme(
            figure_size=(
                max(9.5, 1.35 * len(ext_order) + 2.0),
                max(5.5, 0.58 * len(suite_order) + 1.8),
            ),
            axis_text_x=p9.element_text(rotation=25, ha="right"),
            panel_grid=p9.element_blank(),
        )
    )


def plot_suite_speedup_distributions(
    df: pl.DataFrame,
    baseline: str,
    *,
    extractor_order: Sequence[str] | None = None,
    small_suite_n: int = 15,
    ncols: int = 3,
) -> p9.ggplot:
    """Faceted speedup distributions vs. `baseline` per suite.

    Large suites (`n > small_suite_n`) show horizontal p10–p90 whiskers, IQR bars, and median
    ticks; smaller suites show raw per-instance points. A white diamond marks the geometric mean.
    """
    rel_df = speedup_vs_baseline_df(df, baseline)
    ext_order = [e for e in _resolve_extractor_order(df, extractor_order) if e != baseline]
    suite_map = suite_labels_df(rel_df)
    suite_order = suite_map["suite_label"].to_list()
    ordered = ("suite_label", suite_order)
    rel_labeled = rel_df.join(suite_map, on="suite", how="inner")

    gm_df = rel_labeled.group_by("suite_label", "suite_instances", "extractor").agg(
        pl.col("speedup_vs_base").log().mean().exp().alias("geomean")
    )
    large_summary = (
        rel_labeled.filter(pl.col("suite_instances") > small_suite_n)
        .group_by("suite_label", "extractor")
        .agg(
            pl.col("speedup_vs_base").quantile(0.10).alias("q10"),
            pl.col("speedup_vs_base").quantile(0.25).alias("q25"),
            pl.col("speedup_vs_base").median().alias("median"),
            pl.col("speedup_vs_base").quantile(0.75).alias("q75"),
            pl.col("speedup_vs_base").quantile(0.90).alias("q90"),
        )
    )
    small_multi = rel_labeled.filter(
        (pl.col("suite_instances") <= small_suite_n) & (pl.col("suite_instances") > 1)
    )
    small_single = rel_labeled.filter(pl.col("suite_instances") == 1)
    gm_multi = gm_df.filter(pl.col("suite_instances") > 1)

    p = p9.ggplot(
        to_pd(gm_df, ordered_col=ordered), p9.aes(x="geomean", y="extractor")
    ) + p9.geom_vline(xintercept=1.0, linetype="dashed", color="#333333")

    if not large_summary.is_empty():
        large_pd = to_pd(large_summary, ordered_col=ordered)
        p = (
            p
            + p9.geom_segment(
                p9.aes(x="q10", xend="q90", y="extractor", yend="extractor", color="extractor"),
                data=large_pd,
                inherit_aes=False,
                size=0.9,
                alpha=0.75,
                show_legend=False,
            )
            + p9.geom_segment(
                p9.aes(x="q25", xend="q75", y="extractor", yend="extractor", color="extractor"),
                data=large_pd,
                inherit_aes=False,
                size=4.8,
                alpha=0.82,
                show_legend=False,
            )
            + p9.geom_point(
                p9.aes(x="median", y="extractor"),
                data=large_pd,
                inherit_aes=False,
                shape="|",
                size=4.2,
                color="#111111",
            )
        )
    if not small_multi.is_empty():
        p = p + p9.geom_jitter(
            p9.aes(x="speedup_vs_base", y="extractor", color="extractor"),
            data=to_pd(small_multi, ordered_col=ordered),
            inherit_aes=False,
            width=0.0,
            height=0.18,
            alpha=0.72,
            size=1.8,
            show_legend=False,
        )
    if not small_single.is_empty():
        p = p + p9.geom_point(
            p9.aes(x="speedup_vs_base", y="extractor", color="extractor"),
            data=to_pd(small_single, ordered_col=ordered),
            inherit_aes=False,
            alpha=0.95,
            size=3.2,
            show_legend=False,
        )
    if not gm_multi.is_empty():
        p = p + p9.geom_point(
            p9.aes(x="geomean", y="extractor"),
            data=to_pd(gm_multi, ordered_col=ordered),
            inherit_aes=False,
            shape="D",
            size=2.5,
            color="#111111",
            fill="white",
        )

    nrows = math.ceil(len(suite_order) / ncols)
    return (
        p
        + p9.scale_y_discrete(limits=list(reversed(ext_order)))
        + p9.scale_x_log10(expand=(0.08, 0.0))
        + p9.facet_wrap("~suite_label", ncol=ncols, scales="free_x")
        + p9.labs(
            title=f"Speedup Distribution vs. {baseline} by Suite",
            subtitle=(
                f"n > {small_suite_n}: thick bar = IQR, whisker = p10–p90, tick = median; "
                "smaller suites: raw points; diamond = geomean (> 1.0x is faster)"
            ),
            x=f"Speedup ({baseline} / extractor, log scale)",
            y="",
        )
        + p9.theme(
            figure_size=(5.0 * ncols, max(4.0, 2.75 * nrows + 1.2)),
            panel_spacing_x=0.03,
            panel_spacing_y=0.04,
            strip_text=p9.element_text(weight="bold", size=10),
        )
    )


def plot_suite_normalized_runtime(
    df: pl.DataFrame, baseline: str, *, extractor_order: Sequence[str] | None = None, ncols: int = 3
) -> p9.ggplot:
    """Faceted diverging lollipop chart of total suite runtime relative to `baseline`."""
    ext_order = _resolve_extractor_order(df, extractor_order)
    suite_order = suite_labels_df(df)["suite_label"].to_list()
    ordered = ("suite_label", suite_order)

    plot_df = (
        suite_extractor_summary_df(df, baseline)
        .filter(pl.col("total_runtime_ratio").is_not_null())
        .with_columns(
            pl.col("total_runtime_ratio").log(base=10).alias("_log_ratio"),
            pl.when(pl.col("extractor") == baseline)
            .then(pl.lit("baseline"))
            .when(pl.col("total_runtime_ratio") < 1.0)
            .then(pl.lit("faster"))
            .otherwise(pl.lit("slower"))
            .alias("status"),
            (
                _map_str("total_runtime_ratio", _format_multiplier)
                + " ("
                + _map_str("total_ms", _format_duration_ms)
                + ")"
            ).alias("label"),
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("_log_ratio").max().over("suite_label")
                - pl.col("_log_ratio").min().over("suite_label"),
                pl.lit(0.3),
            ).alias("_log_span")
        )
        .with_columns(
            pl.when(pl.col("total_runtime_ratio") < 1.0)
            .then(10.0 ** (pl.col("_log_ratio") - 0.045 * pl.col("_log_span")))
            .otherwise(10.0 ** (pl.col("_log_ratio") + 0.045 * pl.col("_log_span")))
            .alias("label_x")
        )
    )

    p = (
        p9.ggplot(
            to_pd(plot_df, ordered_col=ordered),
            p9.aes(x="total_runtime_ratio", y="extractor", color="status"),
        )
        + p9.geom_vline(xintercept=1.0, linetype="dashed", color="#444444")
        + p9.geom_segment(
            p9.aes(x=1.0, xend="total_runtime_ratio", y="extractor", yend="extractor"), size=1.1
        )
        + p9.geom_point(size=2.6)
    )
    for subset, ha in (
        (plot_df.filter(pl.col("total_runtime_ratio") < 1.0), "right"),
        (plot_df.filter(pl.col("total_runtime_ratio") >= 1.0), "left"),
    ):
        if not subset.is_empty():
            p = p + p9.geom_text(
                p9.aes(x="label_x", y="extractor", label="label"),
                data=to_pd(subset, ordered_col=ordered),
                inherit_aes=False,
                ha=ha,
                size=7.2,
                color="#222222",
            )

    nrows = math.ceil(len(suite_order) / ncols)
    return (
        p
        + p9.scale_y_discrete(limits=list(reversed(ext_order)))
        + p9.scale_x_log10(expand=(0.50, 0.0))
        + p9.scale_color_manual(
            values={"faster": _COLOR_FASTER, "baseline": _COLOR_BASELINE, "slower": _COLOR_SLOWER},
            limits=["faster", "baseline", "slower"],
            labels={
                "faster": f"Faster than {baseline}",
                "baseline": baseline,
                "slower": f"Slower than {baseline}",
            },
        )
        + p9.facet_wrap("~suite_label", ncol=ncols, scales="free_x")
        + p9.labs(
            title=f"Normalized Cumulative Runtime by Suite (vs. {baseline})",
            subtitle=(
                "Total suite wall-clock time / baseline total time (log scale); "
                "< 1.0x (left, blue) is faster; label shows ratio (total time)"
            ),
            x=f"Total Runtime Ratio (extractor / {baseline}, log scale)",
            y="",
            color="",
        )
        + p9.theme(
            figure_size=(5.4 * ncols, max(4.2, 2.9 * nrows + 1.2)),
            panel_spacing_x=0.03,
            panel_spacing_y=0.04,
            strip_text=p9.element_text(weight="bold", size=10),
        )
    )


# --- Part 3: Pairwise Extractor Deep-Dive ---


def plot_pair_suite_wins(pair_df: pl.DataFrame, e1: str, e2: str) -> p9.ggplot:
    """Stacked horizontal bar chart of win/tie/loss share by benchmark suite."""
    winner_colors, winner_order = winner_palette(e1, e2)
    suite_wins = (
        pair_df.group_by("suite", "winner")
        .len(name="count")
        .with_columns((100.0 * pl.col("count") / pl.col("count").sum().over("suite")).alias("pct"))
    )
    suite_order = (
        pair_df.group_by("suite")
        .agg((pl.col("winner") == e1).mean().alias("e1_rate"))
        .sort("e1_rate")["suite"]
        .to_list()
    )

    return (
        p9.ggplot(to_pd(suite_wins), p9.aes(x="suite", y="pct", fill="winner"))
        + p9.geom_col(width=0.72)
        + p9.scale_x_discrete(limits=suite_order)
        + p9.scale_fill_manual(values=winner_colors, limits=winner_order)
        + p9.coord_flip()
        + p9.labs(
            title=f"Win / Loss Breakdown by Suite: {e1} vs {e2}",
            subtitle=f"Suites sorted by win rate for {e1}",
            x="Suite",
            y="Share of Benchmarks (%)",
            fill="Winner",
        )
        + p9.theme(figure_size=(10, max(3.8, 0.45 * len(suite_order) + 1.5)))
    )


def plot_pair_scatter(pair_df: pl.DataFrame, e1: str, e2: str) -> p9.ggplot:
    """Log–log runtime scatter plot comparing `e1` against `e2`."""
    scatter_df = pair_df.with_columns(
        pl.col("micros_1").clip(lower_bound=1).cast(pl.Float64).alias("t1_us"),
        pl.col("micros_2").clip(lower_bound=1).cast(pl.Float64).alias("t2_us"),
    )
    ref_lines = (
        (0.0, "solid", "#222222", 0.7),
        (-math.log10(2), "dashed", "#777777", 0.5),
        (math.log10(2), "dashed", "#777777", 0.5),
        (-1.0, "dotted", "#999999", 0.5),
        (1.0, "dotted", "#999999", 0.5),
    )

    p = p9.ggplot(to_pd(scatter_df), p9.aes(x="t2_us", y="t1_us", color="suite"))
    for intercept, linetype, color, size in ref_lines:
        p = p + p9.geom_abline(
            slope=1, intercept=intercept, linetype=linetype, color=color, size=size
        )

    return (
        p
        + p9.geom_point(alpha=0.55, size=1.8)
        + p9.scale_x_log10()
        + p9.scale_y_log10()
        + p9.labs(
            title=f"Runtime Scatter: {e1} vs {e2}",
            subtitle=f"Below diagonal: {e1} is faster (dashed: 2x, dotted: 10x; 0 µs clamped to 1 µs)",
            x=f"{e2} Time (µs, log scale)",
            y=f"{e1} Time (µs, log scale)",
            color="Suite",
        )
        + p9.theme(figure_size=(8.5, 7.0))
    )


def plot_pair_feature_correlations(pair_df: pl.DataFrame, e1: str, e2: str) -> p9.ggplot:
    """Horizontal bar chart of Spearman rank correlations between e-graph features and speedup."""
    corr_df = pair_feature_correlations_df(pair_df).with_columns(
        pl.when(pl.col("spearman_r") >= 0).then(pl.lit(e1)).otherwise(pl.lit(e2)).alias("favors"),
        _map_str("spearman_r", lambda r: f"{r:+.2f}").alias("r_label"),
    )
    winner_colors, _ = winner_palette(e1, e2)
    order = corr_df["label"].to_list()
    raw_max = corr_df.select(pl.col("spearman_r").abs().max()).item()
    lim = min(1.08, max(0.35, (float(raw_max) if isinstance(raw_max, int | float) else 0.5) + 0.14))

    p = (
        p9.ggplot(to_pd(corr_df), p9.aes(x="label", y="spearman_r", fill="favors"))
        + p9.geom_hline(yintercept=0.0, linetype="dashed", color="#333333")
        + p9.geom_col(width=0.65, alpha=0.88)
    )
    for subset, ha, nudge in (
        (corr_df.filter(pl.col("spearman_r") >= 0), "left", 0.02),
        (corr_df.filter(pl.col("spearman_r") < 0), "right", -0.02),
    ):
        if not subset.is_empty():
            p = p + p9.geom_text(
                p9.aes(x="label", y="spearman_r", label="r_label"),
                data=to_pd(subset),
                inherit_aes=False,
                ha=ha,
                nudge_y=nudge,
                size=9,
            )

    return (
        p
        + p9.scale_x_discrete(limits=order)
        + p9.scale_y_continuous(limits=(-lim, lim))
        + p9.scale_fill_manual(
            values={e1: winner_colors[e1], e2: winner_colors[e2]}, limits=[e1, e2]
        )
        + p9.coord_flip()
        + p9.labs(
            title=f"Structural Drivers of Speedup: {e1} vs {e2}",
            subtitle=f"Spearman ρ with speedup ({e2} / {e1}); positive favors {e1}, negative favors {e2}",
            x="",
            y="Spearman Rank Correlation (ρ)",
            fill="Higher value favors",
        )
        + p9.theme(figure_size=(10, max(4.0, 0.48 * len(order) + 1.5)))
    )


def plot_pair_regime_map(
    pair_df: pl.DataFrame,
    e1: str,
    e2: str,
    *,
    x_col: str = "enodes_per_eclass",
    y_col: str = "avg_arity",
    log_x: bool = True,
    log_y: bool = True,
    clamp_log2: float = 4.0,
) -> p9.ggplot:
    """2D scatter map of e-graph structure colored by pairwise `log2(speedup)`."""
    map_df = pair_df.select(
        pl.col(x_col).cast(pl.Float64).alias("_x"),
        pl.col(y_col).cast(pl.Float64).alias("_y"),
        pl.col("speedup").log(base=2).clip(-clamp_log2, clamp_log2).alias("log2_speedup"),
    )
    pos_cols = [c for c, is_log in (("_x", log_x), ("_y", log_y)) if is_log]
    map_df = _filter_finite(map_df, ["_x", "_y"], positive_cols=pos_cols)

    x_label = STRUCTURE_FEATURES.get(x_col, x_col)
    y_label = STRUCTURE_FEATURES.get(y_col, y_col)
    p = (
        p9.ggplot(to_pd(map_df), p9.aes(x="_x", y="_y", color="log2_speedup"))
        + p9.geom_point(alpha=0.65, size=2.0)
        + p9.scale_color_gradient2(
            low="#d6604d",
            mid="#cccccc",
            high="#2166ac",
            midpoint=0.0,
            limits=(-clamp_log2, clamp_log2),
            name="log2(speedup)",
        )
        + p9.labs(
            title=f"Pairwise Speedup Regime Map: {e1} vs {e2}",
            subtitle=f"Blue (> 0): {e1} faster; red (< 0): {e2} faster",
            x=f"{x_label} (log scale)" if log_x else x_label,
            y=f"{y_label} (log scale)" if log_y else y_label,
        )
        + p9.theme(figure_size=(10, 6.2))
    )
    if log_x:
        p = p + p9.scale_x_log10()
    if log_y:
        p = p + p9.scale_y_log10()
    return p


def _plot_pair_speedup_binned(
    pair_df: pl.DataFrame,
    e1: str,
    e2: str,
    *,
    x_col: str,
    bin_expr: pl.Expr,
    x_from_bin: pl.Expr,
    log_x: bool,
    title: str,
    subtitle: str,
    x_label: str,
    min_bin_count: int,
) -> p9.ggplot:
    sub = _filter_finite(pair_df, [x_col], positive_cols=[x_col] if log_x else ())
    binned = (
        sub.with_columns(bin_expr.alias("_bin"))
        .group_by("_bin")
        .agg(pl.len().alias("n"), pl.col("speedup").median().alias("median_speedup"))
        .filter(pl.col("n") >= min_bin_count)
        .with_columns(x_from_bin.alias("_x_bin"))
        .sort("_x_bin")
    )
    p = (
        p9.ggplot(to_pd(sub), p9.aes(x=x_col, y="speedup", color="suite"))
        + p9.geom_hline(yintercept=1.0, linetype="dashed", color="#333333")
        + p9.geom_point(alpha=0.5, size=1.8)
        + p9.geom_line(
            p9.aes(x="_x_bin", y="median_speedup"),
            data=to_pd(binned),
            inherit_aes=False,
            color="#111111",
            size=1.0,
        )
        + p9.scale_y_log10()
        + p9.labs(
            title=title,
            subtitle=subtitle,
            x=x_label,
            y=f"Speedup ({e2} / {e1}, log scale)",
            color="Suite",
        )
        + p9.theme(figure_size=(10, 5.5))
    )
    return p + p9.scale_x_log10() if log_x else p


def plot_pair_speedup_vs_reachability(
    pair_df: pl.DataFrame, e1: str, e2: str, *, min_bin_count: int = 3
) -> Compose:
    """Side-by-side speedup scatter plots vs. reachable e-node count and reachable percentage."""
    left = _plot_pair_speedup_binned(
        pair_df,
        e1,
        e2,
        x_col="reach_enodes",
        bin_expr=(pl.col("reach_enodes").cast(pl.Float64).log(base=10) * 2.0).round() / 2.0,
        x_from_bin=10.0 ** pl.col("_bin"),
        log_x=True,
        title="vs. Reachable E-Nodes",
        subtitle=f"> 1.0x: {e1} faster; black line: binned median",
        x_label="Reachable E-Nodes (log scale)",
        min_bin_count=min_bin_count,
    ) + p9.theme(legend_position="none")

    right = (
        _plot_pair_speedup_binned(
            pair_df,
            e1,
            e2,
            x_col="reach_enodes_pct",
            bin_expr=(pl.col("reach_enodes_pct") / 10.0).floor().clip(0, 9) * 10.0 + 5.0,
            x_from_bin=pl.col("_bin"),
            log_x=False,
            title="vs. Reachable E-Node Fraction (%)",
            subtitle="Low % = small subgraph reachable from root",
            x_label="Reachable E-Nodes (% of Total E-Graph)",
            min_bin_count=min_bin_count,
        )
        + p9.labs(y="")
        + p9.theme(legend_position="right")
    )
    return (left | right) & p9.theme(figure_size=(14, 5.2))


def plot_pair_outliers(
    pair_df: pl.DataFrame, e1: str, e2: str, *, top_n: int = 15, max_label_len: int = 45
) -> p9.ggplot:
    """Lollipop chart of the top `top_n` biggest speedups and slowdowns."""
    winner_colors, winner_order = winner_palette(e1, e2)
    outliers = top_outliers_df(pair_df, e1, e2, top_n=top_n, max_label_len=max_label_len)
    outlier_order = outliers["short_label"].to_list()

    return (
        p9.ggplot(to_pd(outliers), p9.aes(x="short_label", y="log10_speedup", color="winner"))
        + p9.geom_hline(yintercept=0.0, linetype="dashed", color="#333333")
        + p9.geom_segment(p9.aes(xend="short_label", y=0.0, yend="log10_speedup"), size=0.9)
        + p9.geom_point(size=2.8)
        + p9.scale_x_discrete(limits=outlier_order)
        + p9.scale_color_manual(values=winner_colors, limits=winner_order)
        + p9.coord_flip()
        + p9.labs(
            title=f"Top {top_n} Biggest Speedups & Slowdowns: {e1} vs {e2}",
            subtitle=f"> 0 (right): {e1} faster; < 0 (left): {e2} faster",
            x="",
            y=f"log10(Speedup) = log10({e2} / {e1})",
            color="Winner",
        )
        + p9.theme(figure_size=(11, max(4.5, 0.32 * len(outlier_order) + 1.5)))
    )
