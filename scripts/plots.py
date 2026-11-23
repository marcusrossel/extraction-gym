"""Reusable plotnine charts for extraction-gym benchmark analysis."""

import copy
import math
from collections.abc import Sequence
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
    top_outliers_df,
)

DEFAULT_PLOTS_DIR: Path = Path("plots")


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
    filename = name if name.endswith(suffix) else f"{name}{suffix}"
    out_path = out_dir / filename
    if isinstance(plot, p9.ggplot):
        plot.save(out_path, format=fmt, verbose=False)
    else:
        fig = copy.deepcopy(plot).draw()
        try:
            fig.savefig(out_path, format=fmt)
        finally:
            plt.close(fig)
    return out_path


def to_pd(frame: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas without requiring pyarrow."""
    return pd.DataFrame(frame.to_dict(as_series=False))


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
    return {e1: "#2b83ba", e2: "#d7191c", "tie": "#bababa"}, [e1, "tie", e2]


def _resolve_extractor_order(df: pl.DataFrame, extractor_order: Sequence[str] | None) -> list[str]:
    if extractor_order is not None:
        return list(extractor_order)
    if "extractor" in df.columns:
        return extractor_summary_df(df)["extractor"].to_list()
    return sorted(df["e1"].unique().to_list())


def plot_cumulative_runtime(
    df: pl.DataFrame, *, extractor_order: Sequence[str] | None = None
) -> p9.ggplot:
    """Stacked horizontal bar chart of cumulative runtime by extractor and suite."""
    summary = extractor_summary_df(df)
    order = list(extractor_order) if extractor_order is not None else summary["extractor"].to_list()

    by_suite_ms = df.group_by("extractor", "suite").agg(
        (pl.col("micros").sum() / 1_000.0).alias("total_ms")
    )
    totals_label = summary.with_columns(
        pl.col("total_ms")
        .map_elements(lambda v: f"{v:,.0f} ms", return_dtype=pl.String)
        .alias("label")
    )

    raw_max = summary["total_ms"].max()
    max_total_ms = float(raw_max) if isinstance(raw_max, int | float) else 0.0

    return (
        p9.ggplot(to_pd(by_suite_ms), p9.aes(x="extractor", y="total_ms", fill="suite"))
        + p9.geom_col(width=0.72)
        + p9.geom_text(
            p9.aes(x="extractor", y="total_ms", label="label"),
            data=to_pd(totals_label),
            inherit_aes=False,
            ha="left",
            nudge_y=max(max_total_ms * 0.015, 1.0),
            size=9,
        )
        + p9.scale_x_discrete(limits=list(reversed(order)))
        + p9.coord_flip()
        + p9.labs(
            title="Cumulative Runtime by Extractor (Stacked by Suite)",
            subtitle="Lower is faster; sorted by total runtime across all benchmarks",
            x="",
            y="Cumulative Time (ms)",
            fill="Suite",
        )
        + p9.theme(figure_size=(10, max(4.0, 0.55 * len(order) + 1.5)))
    )


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
            pl.col("geomean_speedup").map_elements(lambda v: f"{v:.2f}x", return_dtype=pl.String)
            + "\n("
            + pl.col("e1_win_rate").map_elements(lambda v: f"{v:.0f}%", return_dtype=pl.String)
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
        + p9.scale_fill_gradient2(
            low="#d6604d",
            mid="#f7f7f7",
            high="#4393c3",
            midpoint=0.0,
            limits=(-clamp_log2, clamp_log2),
            name="log2(speedup)",
        )
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


def plot_speedup_vs_baseline(df: pl.DataFrame, baseline: str) -> p9.ggplot:
    """Boxplot + geometric-mean diamonds of speedup distributions relative to `baseline`."""
    rel_df = speedup_vs_baseline_df(df, baseline)
    gm_vs_base = (
        rel_df.group_by("extractor")
        .agg(pl.col("speedup_vs_base").log().mean().exp().alias("geomean"))
        .sort("geomean", descending=True)
    )
    rel_order = gm_vs_base["extractor"].to_list()

    return (
        p9.ggplot(to_pd(rel_df), p9.aes(x="extractor", y="speedup_vs_base", fill="extractor"))
        + p9.geom_hline(yintercept=1.0, linetype="dashed", color="#333333")
        + p9.geom_boxplot(alpha=0.75, outlier_alpha=0.25, outlier_size=1.2, show_legend=False)
        + p9.geom_point(
            p9.aes(x="extractor", y="geomean"),
            data=to_pd(gm_vs_base),
            inherit_aes=False,
            shape="D",
            size=3.0,
            color="#111111",
            fill="white",
        )
        + p9.scale_x_discrete(limits=list(reversed(rel_order)))
        + p9.scale_y_log10()
        + p9.coord_flip()
        + p9.labs(
            title=f"Speedup Distribution vs. {baseline} (All Benchmarks)",
            subtitle="Box: IQR & median, diamond: geometric mean; > 1.0x is faster than baseline",
            x="",
            y=f"Speedup ({baseline} / extractor, log scale)",
        )
        + p9.theme(figure_size=(10, max(4.0, 0.55 * len(rel_order) + 1.5)))
    )


def plot_performance_profile(df: pl.DataFrame) -> p9.ggplot:
    """Dolan–Moré performance profile across all extractors."""
    clamped = df.with_columns(
        pl.max_horizontal(pl.col("micros"), pl.lit(1)).cast(pl.Float64).alias("micros_c")
    )
    best_per_inst = clamped.group_by("name", "roots").agg(
        pl.col("micros_c").min().alias("best_micros")
    )
    profile_df = (
        clamped.join(best_per_inst, on=["name", "roots"], how="inner")
        .with_columns((pl.col("micros_c") / pl.col("best_micros")).alias("tau"))
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
        pl.max_horizontal(pl.col("micros_1"), pl.lit(1)).cast(pl.Float64).alias("t1_us"),
        pl.max_horizontal(pl.col("micros_2"), pl.lit(1)).cast(pl.Float64).alias("t2_us"),
    )

    return (
        p9.ggplot(to_pd(scatter_df), p9.aes(x="t2_us", y="t1_us", color="suite"))
        + p9.geom_abline(slope=1, intercept=0, color="#222222", size=0.7)
        + p9.geom_abline(
            slope=1, intercept=-math.log10(2), linetype="dashed", color="#777777", size=0.5
        )
        + p9.geom_abline(
            slope=1, intercept=math.log10(2), linetype="dashed", color="#777777", size=0.5
        )
        + p9.geom_abline(slope=1, intercept=-1.0, linetype="dotted", color="#999999", size=0.5)
        + p9.geom_abline(slope=1, intercept=1.0, linetype="dotted", color="#999999", size=0.5)
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


def plot_pair_speedup_vs_enodes(
    pair_df: pl.DataFrame, e1: str, e2: str, *, min_bin_count: int = 3
) -> p9.ggplot:
    """Scatter plot of speedup vs. reachable e-node count with a log-binned median trendline."""
    reach_df = pair_df.filter(pl.col("reach_enodes").is_not_null() & (pl.col("reach_enodes") > 0))
    binned_size = (
        reach_df.with_columns(
            ((pl.col("reach_enodes").cast(pl.Float64).log(base=10) * 2.0).round() / 2.0).alias(
                "log10_bin"
            )
        )
        .group_by("log10_bin")
        .agg(pl.len().alias("n"), pl.col("speedup").median().alias("median_speedup"))
        .filter(pl.col("n") >= min_bin_count)
        .with_columns((10.0 ** pl.col("log10_bin")).alias("reach_enodes"))
        .sort("reach_enodes")
    )

    return (
        p9.ggplot(to_pd(reach_df), p9.aes(x="reach_enodes", y="speedup", color="suite"))
        + p9.geom_hline(yintercept=1.0, linetype="dashed", color="#333333")
        + p9.geom_point(alpha=0.5, size=1.8)
        + p9.geom_line(
            p9.aes(x="reach_enodes", y="median_speedup"),
            data=to_pd(binned_size),
            inherit_aes=False,
            color="#111111",
            size=1.0,
        )
        + p9.scale_x_log10()
        + p9.scale_y_log10()
        + p9.labs(
            title="vs. Reachable E-Nodes",
            subtitle=f"> 1.0x: {e1} faster; black line: binned median",
            x="Reachable E-Nodes (log scale)",
            y=f"Speedup ({e2} / {e1}, log scale)",
            color="Suite",
        )
        + p9.theme(figure_size=(10, 5.5))
    )


def plot_pair_speedup_vs_reach_pct(
    pair_df: pl.DataFrame, e1: str, e2: str, *, min_bin_count: int = 3
) -> p9.ggplot:
    """Scatter plot of speedup vs. reachable e-node percentage with a binned median trendline."""
    reach_pct_df = pair_df.filter(pl.col("reach_enodes_pct").is_not_null())
    binned_pct = (
        reach_pct_df.with_columns(
            ((pl.col("reach_enodes_pct") / 10.0).floor().clip(0, 9) * 10.0 + 5.0).alias("pct_bin")
        )
        .group_by("pct_bin")
        .agg(pl.len().alias("n"), pl.col("speedup").median().alias("median_speedup"))
        .filter(pl.col("n") >= min_bin_count)
        .sort("pct_bin")
    )

    return (
        p9.ggplot(to_pd(reach_pct_df), p9.aes(x="reach_enodes_pct", y="speedup", color="suite"))
        + p9.geom_hline(yintercept=1.0, linetype="dashed", color="#333333")
        + p9.geom_point(alpha=0.5, size=1.8)
        + p9.geom_line(
            p9.aes(x="pct_bin", y="median_speedup"),
            data=to_pd(binned_pct),
            inherit_aes=False,
            color="#111111",
            size=1.0,
        )
        + p9.scale_y_log10()
        + p9.labs(
            title="vs. Reachable E-Node Fraction (%)",
            subtitle="Low % = small subgraph reachable from root",
            x="Reachable E-Nodes (% of Total E-Graph)",
            y=f"Speedup ({e2} / {e1}, log scale)",
            color="Suite",
        )
        + p9.theme(figure_size=(10, 5.5))
    )


def plot_pair_speedup_vs_reachability(
    pair_df: pl.DataFrame, e1: str, e2: str, *, min_bin_count: int = 3
) -> Compose:
    """Side-by-side speedup scatter plots vs. reachable e-node count and reachable percentage."""
    left = plot_pair_speedup_vs_enodes(pair_df, e1, e2, min_bin_count=min_bin_count) + p9.theme(
        legend_position="none"
    )
    right = (
        plot_pair_speedup_vs_reach_pct(pair_df, e1, e2, min_bin_count=min_bin_count)
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


_DEFAULT_REGIME_FEATURES: tuple[tuple[str, bool], ...] = (
    ("enodes_per_eclass", True),
    ("avg_arity", True),
    ("avg_child_eclasses", True),
    ("reach_enodes_pct", False),
    ("max_depth", True),
    ("cyclic_eclasses_pct", False),
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
    feat_df = rel_df.filter(
        pl.col(feature).is_not_null() & pl.col(feature).cast(pl.Float64).is_finite()
    )
    if log_x:
        feat_df = feat_df.filter(pl.col(feature) > 0)

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
    if log_x:
        p = p + p9.scale_x_log10()
    return p


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
    legend_idx = (nrows - 1) * ncols + (ncols // 2)
    legend_idx = min(legend_idx, len(features) - 1)

    panels: list[p9.ggplot] = []
    for idx, (feat, log_x) in enumerate(features):
        panels.append(
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
        )

    rows: list[p9.ggplot | Compose] = []
    for r in range(nrows):
        row_panels = panels[r * ncols : (r + 1) * ncols]
        row_comp: p9.ggplot | Compose = row_panels[0]
        for panel in row_panels[1:]:
            row_comp = row_comp | panel
        rows.append(row_comp)

    grid: p9.ggplot | Compose = rows[0]
    for row_comp in rows[1:]:
        grid = grid / row_comp

    assert isinstance(grid, Compose)
    return grid & p9.theme(figure_size=(5.0 * ncols, 4.2 * nrows))


def plot_regime_map(
    df: pl.DataFrame,
    *,
    x_col: str = "enodes_per_eclass",
    y_col: str = "avg_arity",
    log_x: bool = True,
    log_y: bool = True,
) -> p9.ggplot:
    """2D scatter map of e-graph structure colored by the fastest extractor on each benchmark."""
    min_times = df.group_by("name", "roots").agg(pl.col("micros").min().alias("min_micros"))
    winners = (
        df.join(min_times, on=["name", "roots"], how="inner")
        .filter(pl.col("micros") == pl.col("min_micros"))
        .group_by("name", "roots")
        .agg(
            pl.when(pl.len() == 1)
            .then(pl.col("extractor").first())
            .otherwise(pl.lit("tie"))
            .alias("fastest"),
            pl.col(x_col).first().cast(pl.Float64).alias("_x"),
            pl.col(y_col).first().cast(pl.Float64).alias("_y"),
        )
        .filter(
            pl.col("_x").is_not_null()
            & pl.col("_x").is_finite()
            & pl.col("_y").is_not_null()
            & pl.col("_y").is_finite()
        )
    )
    if log_x:
        winners = winners.filter(pl.col("_x") > 0)
    if log_y:
        winners = winners.filter(pl.col("_y") > 0)

    x_label = STRUCTURE_FEATURES.get(x_col, x_col)
    y_label = STRUCTURE_FEATURES.get(y_col, y_col)
    p = (
        p9.ggplot(to_pd(winners), p9.aes(x="_x", y="_y", color="fastest"))
        + p9.geom_point(alpha=0.55, size=1.8)
        + p9.labs(
            title=f"Fastest Extractor Regime Map: {x_label} vs. {y_label}",
            subtitle="Each point is one benchmark instance, colored by its fastest extractor",
            x=f"{x_label} (log scale)" if log_x else x_label,
            y=f"{y_label} (log scale)" if log_y else y_label,
            color="Fastest",
        )
        + p9.theme(figure_size=(10, 6.2))
    )
    if log_x:
        p = p + p9.scale_x_log10()
    if log_y:
        p = p + p9.scale_y_log10()
    return p


def plot_pair_feature_correlations(pair_df: pl.DataFrame, e1: str, e2: str) -> p9.ggplot:
    """Horizontal bar chart of Spearman rank correlations between e-graph features and speedup."""
    corr_df = pair_feature_correlations_df(pair_df).with_columns(
        pl.when(pl.col("spearman_r") >= 0).then(pl.lit(e1)).otherwise(pl.lit(e2)).alias("favors"),
        pl.col("spearman_r")
        .map_elements(lambda r: f"{r:+.2f}", return_dtype=pl.String)
        .alias("r_label"),
    )
    winner_colors, _ = winner_palette(e1, e2)
    order = corr_df["label"].to_list()
    pos_df = corr_df.filter(pl.col("spearman_r") >= 0)
    neg_df = corr_df.filter(pl.col("spearman_r") < 0)

    raw_max = corr_df.select(pl.col("spearman_r").abs().max()).item()
    lim = min(1.08, max(0.35, (float(raw_max) if isinstance(raw_max, int | float) else 0.5) + 0.14))

    return (
        p9.ggplot(to_pd(corr_df), p9.aes(x="label", y="spearman_r", fill="favors"))
        + p9.geom_hline(yintercept=0.0, linetype="dashed", color="#333333")
        + p9.geom_col(width=0.65, alpha=0.88)
        + p9.geom_text(
            p9.aes(x="label", y="spearman_r", label="r_label"),
            data=to_pd(pos_df),
            inherit_aes=False,
            ha="left",
            nudge_y=0.02,
            size=9,
        )
        + p9.geom_text(
            p9.aes(x="label", y="spearman_r", label="r_label"),
            data=to_pd(neg_df),
            inherit_aes=False,
            ha="right",
            nudge_y=-0.02,
            size=9,
        )
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
    ).filter(
        pl.col("_x").is_not_null()
        & pl.col("_x").is_finite()
        & pl.col("_y").is_not_null()
        & pl.col("_y").is_finite()
    )
    if log_x:
        map_df = map_df.filter(pl.col("_x") > 0)
    if log_y:
        map_df = map_df.filter(pl.col("_y") > 0)

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
