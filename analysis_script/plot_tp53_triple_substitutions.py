"""
Plot PNAS pre-tuner scores against AlphaGenome splice-site scores for the
TP53_e6 / TP53_e7 randomly-sampled triple exonic substitution experiment
produced by ``run_tp53_triple_substitutions.py --merge``.

Input
-----
``outputs/triple_substitutions/TP53_e6_merged.csv`` and
``outputs/triple_substitutions/TP53_e7_merged.csv`` -- the merged, verified
triple-substitution tables (one row per sampled ``(i, j, k, alt_i, alt_j,
alt_k)`` variant plus one WT reference row per exon).

Output
------
* ``outputs/plots/triple_substitutions/<exon>_pnas_vs_alphagenome.png`` --
  one 2-panel figure per exon:

    left  : pnas_pretuner  vs  alphagenome_mean_logit
    right : pnas_pretuner  vs  alphagenome_prod_logit

  hist2d density + LogNorm colour scale + OLS regression line + n / slope /
  Pearson r / Spearman rho, matching the style of
  ``plot_hybrid_pnas_vs_alphagenome.py`` (panel drawing is imported from that
  module rather than re-implemented).

* ``outputs/tp53_triple_substitutions_slope_summary.csv`` -- slope /
  intercept / Pearson / Spearman for both AlphaGenome definitions, one row
  per (exon, alphagenome_metric).

* ``outputs/tp53_triple_substitutions_pnas_range_comparison.csv`` -- PNAS
  pretuner min/max/range for single-SNV vs double-SNV vs triple-SNV variants
  of TP53_e6 / TP53_e7 (single-SNV from
  ``outputs/tp53_hybrid_scores_with_alphagenome.csv``, double-SNV from
  ``outputs/double_substitutions/<exon>_merged.csv``, when present), so the
  triple-mutant push on SR balance can be judged against both the
  single-mutant and double-mutant baselines -- this is the central
  scientific question the triple-mutant experiment was designed to answer.

Reference (unmutated) rows are excluded from every regression, matching the
single-SNV and double-SNV plots. This script only reads CSVs and draws
figures -- no prediction code is imported or modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
OUTPUT_DIR = ROOT / "outputs" / "plots" / "triple_substitutions"
SUMMARY_CSV = ROOT / "outputs" / "tp53_triple_substitutions_slope_summary.csv"
RANGE_CSV = ROOT / "outputs" / "tp53_triple_substitutions_pnas_range_comparison.csv"
SINGLE_SNV_CSV = ROOT / "outputs" / "tp53_hybrid_scores_with_alphagenome.csv"

EXONS = ("TP53_e6", "TP53_e7")

X_COL = hybrid_plot.X_COL
X_LABEL = hybrid_plot.X_LABEL
ALPHAGENOME_METRICS = hybrid_plot.ALPHAGENOME_METRICS


# (x, y, ha, va) anchors in axes-fraction coords, keyed by corner name.
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


def _load_merged(exon_id: str) -> pd.DataFrame | None:
    path = TRIPLE_SUB_DIR / f"{exon_id}_merged.csv"
    if not path.exists():
        print(f"[skip] {path} not found -- run run_tp53_triple_substitutions.py --merge first.")
        return None
    return pd.read_csv(path)


def _plot_exon(exon_id: str, df: pd.DataFrame) -> list[dict]:
    variants = df[df["is_reference"].fillna(False).astype(bool).ne(True)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        f"{exon_id}  --  randomly sampled triple exonic substitutions\n"
        f"PNAS pre-tuner vs AlphaGenome  (n={len(variants):,} triple mutants)"
    )

    # Match the double-substitution plot: move the stats box out of the
    # densest corner for TP53_e7.
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    triple_dfs: dict[str, pd.DataFrame] = {}
    for exon_id in EXONS:
        df = _load_merged(exon_id)
        if df is not None:
            triple_dfs[exon_id] = df

    if not triple_dfs:
        print("No merged triple-substitution CSVs found; nothing to plot.")
        return

    summary_rows: list[dict] = []
    for exon_id, df in triple_dfs.items():
        summary_rows.extend(_plot_exon(exon_id, df))

    summary = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}  ({len(summary)} rows)\n")
    print(
        summary[["exon_id", "alphagenome_metric", "n", "slope", "pearson_r", "spearman_rho"]]
        .to_string(index=False)
    )

    _report_pnas_ranges(triple_dfs)


if __name__ == "__main__":
    main()
