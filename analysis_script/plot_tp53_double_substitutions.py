"""
Plot PNAS pre-tuner scores against AlphaGenome splice-site scores for the
TP53_e6 / TP53_e7 pairwise (double) exonic substitution experiment produced by
``run_tp53_double_substitutions.py --merge``.

Input
-----
``outputs/double_substitutions/TP53_e6_merged.csv`` and
``outputs/double_substitutions/TP53_e7_merged.csv`` -- the merged,
verified double-substitution tables (one row per (i, j, alt_i, alt_j) variant
plus one WT reference row per exon).

Output
------
* ``outputs/plots/double_substitutions/<exon>_pnas_vs_alphagenome.png`` --
  one 2-panel figure per exon:

    left  : pnas_pretuner  vs  alphagenome_mean_logit
    right : pnas_pretuner  vs  alphagenome_prod_logit

  hist2d density + LogNorm colour scale + OLS regression line + n / slope /
  Pearson r / Spearman rho, matching the style of
  ``plot_hybrid_pnas_vs_alphagenome.py`` (panel drawing is imported from that
  module rather than re-implemented).

* ``outputs/tp53_double_substitutions_slope_summary.csv`` -- slope /
  intercept / Pearson / Spearman for both AlphaGenome definitions, one row
  per (exon, alphagenome_metric).

* ``outputs/tp53_double_substitutions_pnas_range_comparison.csv`` -- PNAS
  pretuner min/max/range for single-SNV vs double-SNV variants of TP53_e6 /
  TP53_e7 (single-SNV side read from
  ``outputs/tp53_hybrid_scores_with_alphagenome.csv`` when present), so the
  double-mutant push on SR balance can be judged against the single-mutant
  baseline.

Reference (unmutated) rows are excluded from every regression, matching the
single-SNV hybrid plots. This script only reads CSVs and draws figures -- no
prediction code is imported or modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reuse the exact hist2d/LogNorm/OLS/Pearson/Spearman panel drawing code used
# by the single-SNV hybrid plots, instead of re-implementing it.
import plot_hybrid_pnas_vs_alphagenome as hybrid_plot  # noqa: E402

DOUBLE_SUB_DIR = ROOT / "outputs" / "double_substitutions"
OUTPUT_DIR = ROOT / "outputs" / "plots" / "double_substitutions"
SUMMARY_CSV = ROOT / "outputs" / "tp53_double_substitutions_slope_summary.csv"
RANGE_CSV = ROOT / "outputs" / "tp53_double_substitutions_pnas_range_comparison.csv"
SINGLE_SNV_CSV = ROOT / "outputs" / "tp53_hybrid_scores_with_alphagenome.csv"

EXONS = ("TP53_e6", "TP53_e7")

X_COL = hybrid_plot.X_COL
X_LABEL = hybrid_plot.X_LABEL
ALPHAGENOME_METRICS = hybrid_plot.ALPHAGENOME_METRICS


def _load_merged(exon_id: str) -> pd.DataFrame | None:
    path = DOUBLE_SUB_DIR / f"{exon_id}_merged.csv"
    if not path.exists():
        print(f"[skip] {path} not found -- run run_tp53_double_substitutions.py --merge first.")
        return None
    return pd.read_csv(path)


def _plot_exon(exon_id: str, df: pd.DataFrame) -> list[dict]:
    variants = df[df["is_reference"].fillna(False).astype(bool).ne(True)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        f"{exon_id}  --  pairwise (double) exonic substitutions\n"
        f"PNAS pre-tuner vs AlphaGenome  (n={len(variants):,} double mutants)"
    )

    rows: list[dict] = []
    for ax, (col, short, y_label) in zip(axes, ALPHAGENOME_METRICS):
        pair = variants[[X_COL, col]].replace([np.inf, -np.inf], np.nan).dropna()
        x = pair[X_COL].to_numpy()
        y = pair[col].to_numpy()

        stats = hybrid_plot._panel(ax, x, y, y_label, fig)
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


def _report_pnas_ranges(double_dfs: dict[str, pd.DataFrame]) -> None:
    rows: list[dict] = []

    for exon_id, df in double_dfs.items():
        variants = df[df["is_reference"].fillna(False).astype(bool).ne(True)]
        rows.append(_pnas_range_row(variants[X_COL], exon_id, "double_snv"))

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

    table = pd.DataFrame(rows).sort_values(["exon_id", "mutation_kind"])
    RANGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RANGE_CSV, index=False)
    print(f"\nSaved {RANGE_CSV}  ({len(table)} rows)\n")
    print(
        "PNAS pretuner range, single-SNV vs double-SNV "
        "(wider double_snv range/spread => double mutations push SR balance "
        "substantially farther than single mutations):\n"
    )
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    double_dfs: dict[str, pd.DataFrame] = {}
    for exon_id in EXONS:
        df = _load_merged(exon_id)
        if df is not None:
            double_dfs[exon_id] = df

    if not double_dfs:
        print("No merged double-substitution CSVs found; nothing to plot.")
        return

    summary_rows: list[dict] = []
    for exon_id, df in double_dfs.items():
        summary_rows.extend(_plot_exon(exon_id, df))

    summary = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}  ({len(summary)} rows)\n")
    print(
        summary[["exon_id", "alphagenome_metric", "n", "slope", "pearson_r", "spearman_rho"]]
        .to_string(index=False)
    )

    _report_pnas_ranges(double_dfs)


if __name__ == "__main__":
    main()
