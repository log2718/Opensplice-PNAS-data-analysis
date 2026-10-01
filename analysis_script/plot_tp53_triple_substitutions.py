"""
Plot PNAS pre-tuner scores against AlphaGenome splice-site scores for the
TP53_e6 / TP53_e7 randomly-sampled triple exonic substitution experiment
produced by ``run_tp53_triple_substitutions.py --merge``, in the context of
the single- and double-substitution experiments that preceded it.

Input
-----
``outputs/tp53_hybrid_scores_with_alphagenome.csv`` -- exhaustive single-SNV
saturation (``construct_type == "wt"``, ``mutation_region == "exon"``; the WT
reference row per exon has ``mutation_region == "reference"``).

``outputs/double_substitutions/TP53_e{6,7}_merged.csv`` -- exhaustive
pairwise (double) substitution tables (one row per ``(i, j, alt_i, alt_j)``
variant plus one WT reference row per exon, ``is_reference`` flagged).

``outputs/triple_substitutions/TP53_e{6,7}_merged.csv`` -- 50,000 randomly
sampled triple-substitution variants (seed=42) plus one WT reference row per
exon, ``is_reference`` flagged. Triples are a random sample, NOT exhaustive.

This script only reads the CSVs above and draws figures / writes summary
CSVs -- no prediction code (PNAS pre-tuner, AlphaGenome) is imported, rerun,
or modified, and no result CSV is rewritten.

Output -- exactly three PNGs, nothing else
--------------------------------------------------------------------------
1. ``outputs/plots/triple_substitutions/TP53_e6_pnas_vs_alphagenome.png``
2. ``outputs/plots/triple_substitutions/TP53_e7_pnas_vs_alphagenome.png``

   Triple-substitutions-only, same panel-drawing code as
   ``plot_hybrid_pnas_vs_alphagenome.py`` / ``plot_tp53_double_substitutions.py``:
   hist2d + ``Blues`` colormap + LogNorm colour scale + "Count" colorbar +
   red OLS line + white n / slope / Pearson r / Spearman rho stats box, no
   scatter, no WT marker. Left panel = mean_logit, right panel = prod_logit.

3. ``outputs/triple_substitutions/plots/TP53_single_double_triple_overlay.png``

   One figure, two panels (left = TP53_e6, right = TP53_e7), each an
   alpha-composited density overlay of single (green) / double (blue) /
   triple (violet) substitutions -- PNAS pre-tuner vs AlphaGenome product
   logit. Each order is its own single-hue RGBA layer (rectangular
   histogram bins, per-cell alpha scaled by log(count) so empty bins are
   transparent and dense bins are saturated) so the three overlaid clouds
   stay distinguishable instead of turning into a muddy block. One OLS line
   + n/slope/Pearson r/Spearman rho per order, legend identifying each
   order, no WT marker.

Everything else previously produced by this script (PNAS-distribution
histograms, the triple-only duplicate scatter, the e6-vs-e7 triple scatter,
the binned-response plot, the PNAS-range bar chart, the common-PNAS-range
diagnostic figure, the mean-logit variant of the overlay) has been removed.
This script deletes any stale PNGs from both output directories before
regenerating, so only the three files above are ever present.

Summary CSVs (kept -- still computed from the same underlying stats used by
the figures above, even though several of them no longer have a matching
plot):

* ``outputs/tp53_triple_substitutions_slope_summary.csv``
* ``outputs/tp53_triple_substitutions_pnas_range_comparison.csv``
* ``outputs/triple_substitutions/triple_analysis_summary.csv``

Reference (unmutated) rows are excluded from every regression and every
distribution statistic; no figure in this script marks the WT point.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb
from scipy.stats import linregress, pearsonr, spearmanr

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reuse the exact hist2d/LogNorm/OLS/Pearson/Spearman panel drawing code used
# by the single-SNV hybrid plots, instead of re-implementing it -- except for
# the stats textbox placement, which is configurable here (see _panel), same
# pattern as plot_tp53_double_substitutions.py.
import plot_hybrid_pnas_vs_alphagenome as hybrid_plot  # noqa: E402

TRIPLE_SUB_DIR = ROOT / "outputs" / "triple_substitutions"
DOUBLE_SUB_DIR = ROOT / "outputs" / "double_substitutions"
SINGLE_SNV_CSV = ROOT / "outputs" / "tp53_hybrid_scores_with_alphagenome.csv"

# Main hist2d plot location (figures 1 and 2).
OUTPUT_DIR = ROOT / "outputs" / "plots" / "triple_substitutions"
SUMMARY_CSV = ROOT / "outputs" / "tp53_triple_substitutions_slope_summary.csv"
RANGE_CSV = ROOT / "outputs" / "tp53_triple_substitutions_pnas_range_comparison.csv"

# Single-vs-double-vs-triple overlay location (figure 3).
NEW_PLOT_DIR = TRIPLE_SUB_DIR / "plots"
OVERLAY_PNG = NEW_PLOT_DIR / "TP53_single_double_triple_overlay.png"
NEW_SUMMARY_CSV = TRIPLE_SUB_DIR / "triple_analysis_summary.csv"

EXONS = ("TP53_e6", "TP53_e7")
ORDERS = ("single", "double", "triple")

X_COL = hybrid_plot.X_COL
X_LABEL = hybrid_plot.X_LABEL
ALPHAGENOME_METRICS = hybrid_plot.ALPHAGENOME_METRICS  # [(col, short, label), ...]
_METRIC_BY_SHORT = {short: (col, short, label) for col, short, label in ALPHAGENOME_METRICS}

# single = green, double = blue, triple = violet.
CATEGORICAL = {
    "single": "#2ca85a",
    "double": "#2a78d6",
    "triple": "#8e44ad",
}
ORDER_LABELS_FULL = {
    "single": "single (exhaustive)",
    "double": "double (exhaustive)",
    "triple": "triple (random 50k sample, seed=42)",
}


def _clean_output_dirs() -> None:
    """Delete stale PNGs from both output directories so only the three
    figures this script still produces are ever present."""
    for d in (OUTPUT_DIR, NEW_PLOT_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Figures 1 & 2: triple-substitutions-only hist2d panels. Reuses
# plot_hybrid_pnas_vs_alphagenome's hist2d/LogNorm/OLS/stats-box panel code
# directly (same as plot_tp53_double_substitutions.py's ``_panel``); only the
# stats-box corner is configurable so the TP53_e7 panel (densest in the
# upper-left) can move it out of the point cloud, same trick the
# double-substitution script uses.
# --------------------------------------------------------------------------

_STATS_BOX_ANCHORS = {
    "upper left": (0.02, 0.98, "left", "top"),
    "lower right": (0.98, 0.02, "right", "bottom"),
}


def _panel(ax, x: np.ndarray, y: np.ndarray, y_label: str, fig, *, stats_loc: str = "upper left") -> dict:
    """Same hist2d + OLS regression panel as ``plot_hybrid_pnas_vs_alphagenome._panel``,
    with a configurable stats-box corner (that module's version is pinned to
    upper-left)."""
    lin = linregress(x, y)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    hist = ax.hist2d(x, y, bins=hybrid_plot.BINS, cmap="Blues", cmin=1, norm=LogNorm())
    fig.colorbar(hist[3], ax=ax, label="Count")

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, lin.intercept + lin.slope * x_line, color="crimson", lw=1.5, label="OLS fit")

    box_x, box_y, ha, va = _STATS_BOX_ANCHORS[stats_loc]
    ax.text(
        box_x, box_y,
        f"n = {len(x):,}\n"
        f"Slope = {lin.slope:.3f}\n"
        f"Pearson r = {pearson.statistic:.3f}\n"
        f"Spearman ρ = {spearman.statistic:.3f}",
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )
    ax.set_xlabel(hybrid_plot.X_LABEL)
    ax.set_ylabel(y_label)
    ax.set_box_aspect(1)

    return {
        "n": int(len(x)),
        "slope": lin.slope,
        "intercept": lin.intercept,
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
    }


def _load_merged(sub_dir: Path, exon_id: str) -> pd.DataFrame | None:
    path = sub_dir / f"{exon_id}_merged.csv"
    if not path.exists():
        print(f"[missing] {path} not found.")
        return None
    return pd.read_csv(path)


def _plot_exon_main(exon_id: str, df: pd.DataFrame) -> list[dict]:
    variants = df[df["is_reference"].fillna(False).astype(bool).ne(True)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        f"{exon_id}  --  randomly sampled triple exonic substitutions\n"
        f"PNAS pre-tuner vs AlphaGenome  (n={len(variants):,} triple mutants)"
    )

    stats_loc = "lower right" if exon_id == "TP53_e7" else "upper left"

    rows: list[dict] = []
    for ax, (col, short, y_label) in zip(axes, ALPHAGENOME_METRICS):
        pair = variants[[X_COL, col]].replace([np.inf, -np.inf], np.nan).dropna()
        x = pair[X_COL].to_numpy()
        y = pair[col].to_numpy()

        stats = _panel(ax, x, y, y_label, fig, stats_loc=stats_loc)
        rows.append({
            "exon_id": exon_id,
            "alphagenome_metric": short,
            "alphagenome_column": col,
            **stats,
        })

    out = OUTPUT_DIR / f"{exon_id}_pnas_vs_alphagenome.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return rows


def _pnas_range_row(x: pd.Series, exon_id: str, mutation_kind: str) -> dict:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "exon_id": exon_id, "mutation_kind": mutation_kind,
            "n": 0, "pnas_min": np.nan, "pnas_max": np.nan, "pnas_range": np.nan,
        }
    return {
        "exon_id": exon_id,
        "mutation_kind": mutation_kind,
        "n": int(len(x)),
        "pnas_min": float(x.min()),
        "pnas_max": float(x.max()),
        "pnas_range": float(x.max() - x.min()),
    }


def _report_pnas_ranges(triple_dfs: dict[str, pd.DataFrame]) -> None:
    rows: list[dict] = []

    for exon_id, df in triple_dfs.items():
        variants = df[df["is_reference"].fillna(False).astype(bool).ne(True)]
        rows.append(_pnas_range_row(variants[X_COL], exon_id, "triple_snv"))

    if DOUBLE_SUB_DIR.exists():
        for exon_id in EXONS:
            path = DOUBLE_SUB_DIR / f"{exon_id}_merged.csv"
            if not path.exists():
                continue
            double = pd.read_csv(path)
            double_variants = double[double["is_reference"].fillna(False).astype(bool).ne(True)]
            rows.append(_pnas_range_row(double_variants[X_COL], exon_id, "double_snv"))
    else:
        print(f"[note] {DOUBLE_SUB_DIR} not found -- skipping double-SNV PNAS range comparison.")

    if SINGLE_SNV_CSV.exists():
        single = pd.read_csv(SINGLE_SNV_CSV)
        single = single[
            (single["construct_type"] == "wt")
            & (single["mutation_region"] == "exon")
            & single["is_reference"].fillna(False).astype(bool).ne(True)
        ]
        for exon_id in EXONS:
            sub = single[single["construct_label"] == exon_id]
            rows.append(_pnas_range_row(sub[X_COL], exon_id, "single_snv"))
    else:
        print(f"[note] {SINGLE_SNV_CSV} not found -- skipping single-SNV PNAS range comparison.")

    if not rows:
        return

    kind_order = {"single_snv": 0, "double_snv": 1, "triple_snv": 2}
    table = pd.DataFrame(rows)
    table["_order"] = table["mutation_kind"].map(kind_order)
    table = table.sort_values(["exon_id", "_order"]).drop(columns="_order")

    RANGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RANGE_CSV, index=False)
    print(f"\nSaved {RANGE_CSV}  ({len(table)} rows)\n")
    print(
        "PNAS pretuner range, single-SNV vs double-SNV vs triple-SNV "
        "(a widening range/spread from single -> double -> triple would show "
        "that each added mutation pushes SR balance substantially farther):\n"
    )
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))


def _run_main_plots(triple_dfs: dict[str, pd.DataFrame]) -> None:
    summary_rows: list[dict] = []
    for exon_id, df in triple_dfs.items():
        summary_rows.extend(_plot_exon_main(exon_id, df))

    summary = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}  ({len(summary)} rows)\n")
    print(
        summary[["exon_id", "alphagenome_metric", "n", "slope", "pearson_r", "spearman_rho"]]
        .to_string(index=False)
    )

    _report_pnas_ranges(triple_dfs)


# --------------------------------------------------------------------------
# Single vs double vs triple: shared data loading / stats helpers
# --------------------------------------------------------------------------


def _load_single(exon_id: str) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if not SINGLE_SNV_CSV.exists():
        print(f"[missing] {SINGLE_SNV_CSV} not found.")
        return None, None
    df = pd.read_csv(SINGLE_SNV_CSV)
    sub = df[(df["construct_type"] == "wt") & (df["construct_label"] == exon_id)]
    variants = sub[sub["mutation_region"] == "exon"].copy()
    ref_rows = sub[sub["mutation_region"] == "reference"]
    ref_row = ref_rows.iloc[0] if len(ref_rows) else None
    return variants, ref_row


def _load_merged_split(sub_dir: Path, exon_id: str) -> tuple[pd.DataFrame | None, pd.Series | None]:
    df = _load_merged(sub_dir, exon_id)
    if df is None:
        return None, None
    is_ref = df["is_reference"].fillna(False).astype(bool)
    variants = df[~is_ref].copy()
    ref_rows = df[is_ref]
    ref_row = ref_rows.iloc[0] if len(ref_rows) else None
    return variants, ref_row


def _load_all_orders(exon_id: str) -> dict[str, dict]:
    data: dict[str, dict] = {}

    v, r = _load_single(exon_id)
    if v is not None and len(v):
        data["single"] = {"variants": v, "ref": r}

    v, r = _load_merged_split(DOUBLE_SUB_DIR, exon_id)
    if v is not None and len(v):
        data["double"] = {"variants": v, "ref": r}

    v, r = _load_merged_split(TRIPLE_SUB_DIR, exon_id)
    if v is not None and len(v):
        data["triple"] = {"variants": v, "ref": r}

    return data


def _clean_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    pair = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return pair[x_col].to_numpy(), pair[y_col].to_numpy()


def _clean_1d(x: pd.Series) -> np.ndarray:
    return x.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def _regression_stats(x: np.ndarray, y: np.ndarray) -> dict:
    lin = linregress(x, y)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": int(len(x)),
        "slope": float(lin.slope),
        "intercept": float(lin.intercept),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _distribution_stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        keys = ["n", "pnas_min", "pnas_max", "pnas_range", "pnas_mean", "pnas_median",
                "pnas_p1", "pnas_p5", "pnas_p95", "pnas_p99"]
        return {k: (0 if k == "n" else np.nan) for k in keys}
    return {
        "n": int(len(x)),
        "pnas_min": float(np.min(x)),
        "pnas_max": float(np.max(x)),
        "pnas_range": float(np.max(x) - np.min(x)),
        "pnas_mean": float(np.mean(x)),
        "pnas_median": float(np.median(x)),
        "pnas_p1": float(np.percentile(x, 1)),
        "pnas_p5": float(np.percentile(x, 5)),
        "pnas_p95": float(np.percentile(x, 95)),
        "pnas_p99": float(np.percentile(x, 99)),
    }


def _distribution_table(all_data: dict[str, dict]) -> pd.DataFrame:
    """PNAS-distribution stats, single vs double vs triple, per exon.

    CSV only -- no standalone distribution histogram is plotted any more.
    """
    rows = []
    for exon_id in EXONS:
        data = all_data[exon_id]
        for order in ORDERS:
            if order not in data:
                continue
            stats = _distribution_stats(_clean_1d(data[order]["variants"][X_COL]))
            rows.append({"exon_id": exon_id, "mutation_order": order, **stats})
    return pd.DataFrame(rows)


def _build_regression_table(all_data: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    lookup: dict = {}
    for exon_id in EXONS:
        data = all_data[exon_id]
        for order in ORDERS:
            if order not in data:
                continue
            for col, short, _label in ALPHAGENOME_METRICS:
                x, y = _clean_xy(data[order]["variants"], X_COL, col)
                if len(x) == 0:
                    continue
                stats = _regression_stats(x, y)
                lookup[(exon_id, order, col)] = stats
                rows.append({
                    "exon_id": exon_id, "mutation_order": order,
                    "alphagenome_metric": short, "alphagenome_column": col, **stats,
                })
    return pd.DataFrame(rows), lookup


def _common_range_table(all_data: dict[str, dict], metric_col: str) -> pd.DataFrame | None:
    """e6-vs-e7 triple-mutant regression stats restricted to their common PNAS
    range. CSV only -- no standalone figure."""
    if "triple" not in all_data["TP53_e6"] or "triple" not in all_data["TP53_e7"]:
        print("[skip] missing triple data for e6 or e7 -- skipping common-range diagnostic.")
        return None

    x6 = _clean_1d(all_data["TP53_e6"]["triple"]["variants"][X_COL])
    x7 = _clean_1d(all_data["TP53_e7"]["triple"]["variants"][X_COL])
    lo, hi = max(x6.min(), x7.min()), min(x6.max(), x7.max())

    if lo >= hi:
        print("[note] TP53_e6 and TP53_e7 triple-mutant PNAS ranges do not overlap "
              "-- skipping common-range diagnostic.")
        return None

    rows = []
    for exon_id in EXONS:
        df = all_data[exon_id]["triple"]["variants"]
        common = df[(df[X_COL] >= lo) & (df[X_COL] <= hi)]
        x, y = _clean_xy(common, X_COL, metric_col)
        stats = _regression_stats(x, y)
        rows.append({
            "exon_id": exon_id, "common_pnas_min": lo, "common_pnas_max": hi,
            "alphagenome_metric": metric_col, **stats,
        })

    table = pd.DataFrame(rows)
    print(f"\nCommon PNAS range across TP53_e6/TP53_e7 triple mutants: [{lo:.4f}, {hi:.4f}]")
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))
    return table


# --------------------------------------------------------------------------
# Figure 3: single vs double vs triple density overlay, one figure, two
# panels (TP53_e6 left, TP53_e7 right). Each order is rendered as its own
# single-hue RGBA image built from a 2D count histogram, with per-cell alpha
# scaled by log(count) rather than a flat per-layer alpha, so overlapping
# orders stay readable instead of turning into an opaque muddy block.
# Composited against the white figure background this reproduces the same
# "near-white at low count -> saturated at high count" look as the Blues
# hist2d used in figures 1/2.
# --------------------------------------------------------------------------

_ALPHA_RANGE = (0.12, 0.88)
_DENSITY_BINS = 40


def _alpha_density_rgba(
    x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, color: str,
) -> np.ndarray | None:
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    counts = counts.T  # row 0 = lowest y, matching imshow(..., origin="lower")
    mask = counts > 0
    if not mask.any():
        return None

    log_counts = np.log1p(counts)
    peak = log_counts[mask].max()
    density = log_counts / peak if peak > 0 else log_counts

    alpha_lo, alpha_hi = _ALPHA_RANGE
    alpha = np.where(mask, alpha_lo + (alpha_hi - alpha_lo) * density, 0.0)

    r, g, b = to_rgb(color)
    rgba = np.zeros((*counts.shape, 4))
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = alpha
    return rgba


def _draw_order_overlay_panel(
    ax, exon_id: str, data: dict[str, dict], metric_col: str, reg_lookup: dict,
) -> None:
    xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for order in ORDERS:
        if order not in data:
            continue
        x, y = _clean_xy(data[order]["variants"], X_COL, metric_col)
        if len(x):
            xy[order] = (x, y)

    if not xy:
        ax.set_title(f"{exon_id}  (no data)")
        return

    all_x = np.concatenate([xy[o][0] for o in xy])
    all_y = np.concatenate([xy[o][1] for o in xy])
    x_edges = np.linspace(all_x.min(), all_x.max(), _DENSITY_BINS + 1)
    y_edges = np.linspace(all_y.min(), all_y.max(), _DENSITY_BINS + 1)
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])

    stats_lines = []
    # Largest / densest series drawn first (bottom) so the sparser series
    # stay visible on top.
    for order in ("triple", "double", "single"):
        if order not in xy:
            continue
        x, y = xy[order]
        rgba = _alpha_density_rgba(x, y, x_edges, y_edges, CATEGORICAL[order])
        if rgba is not None:
            ax.imshow(
                rgba, origin="lower", extent=extent, aspect="auto",
                interpolation="nearest", zorder=2,
            )
        stats = reg_lookup.get((exon_id, order, metric_col))
        if stats is not None:
            x_line = np.array([x.min(), x.max()])
            ax.plot(
                x_line, stats["intercept"] + stats["slope"] * x_line,
                color=CATEGORICAL[order], linewidth=2.2, zorder=5,
                label=f"{ORDER_LABELS_FULL[order]}  (n={stats['n']:,})",
            )
            stats_lines.append(
                f"{order:6s} n={stats['n']:>6,}  slope={stats['slope']:+.3f}  "
                f"r={stats['pearson_r']:+.3f}  rho={stats['spearman_rho']:+.3f}"
            )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(X_LABEL)
    ax.set_title(exon_id)

    if stats_lines:
        ax.text(
            0.02, 0.98, "\n".join(stats_lines), transform=ax.transAxes,
            fontsize=7.5, family="monospace", ha="left", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9), zorder=11,
        )

    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


def _plot_single_double_triple_overlay(
    all_data: dict[str, dict], metric_col: str, metric_label: str, reg_lookup: dict, out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)

    for ax, exon_id in zip(axes, EXONS):
        _draw_order_overlay_panel(ax, exon_id, all_data[exon_id], metric_col, reg_lookup)

    axes[0].set_ylabel(metric_label)
    fig.suptitle(
        f"Single vs double vs triple substitutions  --  PNAS pre-tuner vs AlphaGenome ({metric_label.splitlines()[0]})\n"
        "single (green) + double (blue) + triple (violet), density overlay"
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---- orchestration -----------------------------------------------------------

def _run_comparison_analyses(all_data: dict[str, dict]) -> None:
    # --- distribution stats (CSV only) --------------------------------------
    dist_table = _distribution_table(all_data)
    print("\n" + "=" * 78)
    print("PNAS pre-tuner distribution: single vs double vs triple, per exon")
    print("=" * 78)
    with pd.option_context("display.width", 220):
        print(dist_table.round(4).to_string(index=False))

    # --- regression stats (single source of truth for annotations + CSV) --
    reg_table, reg_lookup = _build_regression_table(all_data)
    print("\n" + "=" * 78)
    print("Regression stats (PNAS pretuner vs AlphaGenome), per exon x order x metric")
    print("=" * 78)
    with pd.option_context("display.width", 220):
        print(
            reg_table[["exon_id", "mutation_order", "alphagenome_metric", "n", "slope",
                       "pearson_r", "spearman_rho"]]
            .to_string(index=False)
        )

    # --- figure 3: single vs double vs triple overlay (prod_logit only) ----
    prod_col, _prod_short, prod_label = _METRIC_BY_SHORT["prod_logit"]
    _plot_single_double_triple_overlay(all_data, prod_col, prod_label, reg_lookup, OVERLAY_PNG)

    # --- common-range diagnostic (CSV only) ---------------------------------
    common_range_table = _common_range_table(all_data, prod_col)

    # --- consolidated CSV summary -------------------------------------------
    summary_rows: list[dict] = []
    for _, r in dist_table.iterrows():
        row = r.to_dict()
        row["section"] = "pnas_distribution"
        summary_rows.append(row)
    for _, r in reg_table.iterrows():
        row = r.to_dict()
        row["section"] = "regression"
        summary_rows.append(row)
    if common_range_table is not None:
        for _, r in common_range_table.iterrows():
            row = r.to_dict()
            row["section"] = "common_range_diagnostic"
            row["mutation_order"] = "triple"
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    front_cols = ["section", "exon_id", "mutation_order", "alphagenome_metric"]
    other_cols = [c for c in summary.columns if c not in front_cols]
    summary = summary[front_cols + other_cols]

    NEW_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(NEW_SUMMARY_CSV, index=False)
    print(f"\nSaved {NEW_SUMMARY_CSV}  ({len(summary)} rows)")

    # --- final headline report (console only) --------------------------------
    print("\n" + "=" * 78)
    print("HEADLINE: does adding a 3rd simultaneous substitution expand the PNAS")
    print("x-axis beyond doubles, and does AlphaGenome respond in TP53_e7?")
    print("=" * 78)
    for exon_id in EXONS:
        sub = dist_table[dist_table["exon_id"] == exon_id].set_index("mutation_order")
        for order in ORDERS:
            if order in sub.index:
                row = sub.loc[order]
                print(f"  {exon_id:8s} {order:6s}: n={int(row['n']):>6,}  "
                      f"PNAS [{row['pnas_min']:.3f}, {row['pnas_max']:.3f}]  "
                      f"range={row['pnas_range']:.3f}")
    print()
    for exon_id in EXONS:
        row = reg_table[
            (reg_table["exon_id"] == exon_id)
            & (reg_table["mutation_order"] == "triple")
            & (reg_table["alphagenome_metric"] == "prod_logit")
        ]
        if len(row):
            row = row.iloc[0]
            print(f"  {exon_id:8s} triple prod_logit: slope={row['slope']:+.4f}  "
                  f"Pearson r={row['pearson_r']:+.4f}  Spearman rho={row['spearman_rho']:+.4f}  "
                  f"n={int(row['n']):,}")


def main() -> None:
    triple_dfs: dict[str, pd.DataFrame] = {}
    for exon_id in EXONS:
        df = _load_merged(TRIPLE_SUB_DIR, exon_id)
        if df is not None:
            triple_dfs[exon_id] = df

    if not triple_dfs:
        print("No merged triple-substitution CSVs found; nothing to plot.")
        return

    _clean_output_dirs()

    _run_main_plots(triple_dfs)

    all_data = {exon_id: _load_all_orders(exon_id) for exon_id in EXONS}
    _run_comparison_analyses(all_data)


if __name__ == "__main__":
    main()
