"""
Plot PNAS pre-tuner scores against AlphaGenome splice-site scores for a
single-motif (GGA or UAG) tandem-repeat exonic substitution experiment run
out to 1x-15x copies, e.g. produced by

    python analysis_script/run_tp53_motif_substitutions.py \\
        --exon TP53_e6 --motifs UAG --copy-counts 1-15 --alphagenome

Generalizes ``plot_tp53_e7_uag_1x_15x.py`` (which is left untouched -- it
still plots the original TP53_e7 UAG 1x-15x run and nothing else) to any
(exon, motif) pair sharing the same single-motif 1x-15x CSV shape, so the
two new TP53_e6 UAG/GGA 1x-15x experiments don't need their own dedicated
one-off plotting scripts. Panel drawing and stats helpers are imported from
``plot_tp53_motif_substitutions`` rather than re-implemented.

Input
-----
``outputs/motif_substitutions/{exon}_{motif}_1x_15x.csv`` -- exhaustive
single-motif tandem-repeat placements (1x-15x) + 1 WT reference row, written
by ``run_tp53_motif_substitutions.py``. Only exonic sequence differs between
rows; upstream/downstream flanks and splice sites are the untouched WT
flanks in every row.

This script only reads a CSV and draws one figure -- no prediction code is
imported, rerun, or modified.

Output -- exactly one PNG per invocation
------------------------------------------
``outputs/plots/motif_substitutions/{exon}_{motif}_1x_15x_pnas_vs_alphagenome.png``
-- 2-panel hist2d (left: mean_logit, right: prod_logit), the same
Blues/LogNorm/"Count"-colorbar/crimson-OLS/white-stats-box style used
throughout this repo (drawn by the exact same shared ``_panel`` helper as
every other hist2d figure in this repo). No scatter overlay, no WT marker,
no boxplots, no copy-count plots, no comparison panels -- just this one
figure.

Usage
-----
    python analysis_script/plot_tp53_e6_motif_1x_15x.py --exon TP53_e6 --motif UAG
    python analysis_script/plot_tp53_e6_motif_1x_15x.py --exon TP53_e6 --motif GGA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reuse the hist2d panel + stats helpers from the TP53_e7 1x-4x motif-plotting
# script, rather than re-implementing them.
import plot_tp53_motif_substitutions as motif_plot  # noqa: E402

plt = motif_plot.plt

MOTIF_DIR = ROOT / "outputs" / "motif_substitutions"
OUTPUT_DIR = ROOT / "outputs" / "plots" / "motif_substitutions"

X_COL = motif_plot.X_COL
ALPHAGENOME_METRICS = motif_plot.ALPHAGENOME_METRICS
_METRIC_BY_SHORT = motif_plot._METRIC_BY_SHORT
_panel = motif_plot._panel
_clean_xy = motif_plot._clean_xy
_regression_stats = motif_plot._regression_stats

COPY_COUNTS = tuple(range(1, 16))


# --------------------------------------------------------------------------
# The one figure: 2-panel hist2d (mean_logit | prod_logit), all copy counts
# pooled -- no other plots.
# --------------------------------------------------------------------------

def _plot_main_response(variants: pd.DataFrame, exon: str, motif: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        f"{exon}  --  {motif} motif substitutions, 1x-15x copies\n"
        f"PNAS pre-tuner vs AlphaGenome  (n={len(variants):,} motif variants)"
    )

    for ax, (col, _short, y_label) in zip(axes, ALPHAGENOME_METRICS):
        x, y = _clean_xy(variants, X_COL, col)
        _panel(ax, x, y, y_label, fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Printed summary (text only -- no figures)
# --------------------------------------------------------------------------

def _print_summary(variants: pd.DataFrame, wt_row: pd.Series, prod_col: str) -> None:
    print("\n" + "=" * 100)
    print("Per copy_count summary")
    print("=" * 100)

    rows = []
    for cc in COPY_COUNTS:
        sub = variants[variants["copy_count"] == cc]
        if sub.empty:
            continue
        x, y = _clean_xy(sub, X_COL, prod_col)
        reg = _regression_stats(x, y) if len(x) > 1 else {
            "slope": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan,
        }
        rows.append({
            "copy_count": cc,
            "n": len(sub),
            "pnas_min": sub[X_COL].min(),
            "pnas_max": sub[X_COL].max(),
            "pnas_mean": sub[X_COL].mean(),
            "mean_delta_from_wt": (sub[X_COL] - wt_row[X_COL]).mean(),
            "ols_slope": reg["slope"],
            "pearson_r": reg["pearson_r"],
            "spearman_rho": reg["spearman_rho"],
        })
    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string(index=False))

    wt_pnas = float(wt_row[X_COL])
    all_pnas = variants[X_COL]
    print(f"\nWT PNAS pre-tuner: {wt_pnas:.4f}")
    print(
        f"Overall PNAS range: [{all_pnas.min():.4f}, {all_pnas.max():.4f}]  "
        f"(n={len(all_pnas):,})"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the single main PNAS-vs-AlphaGenome hist2d figure for a "
            "single-motif 1x-15x exonic motif-substitution CSV."
        )
    )
    parser.add_argument("--exon", required=True, help="e.g. TP53_e6, TP53_e7")
    parser.add_argument("--motif", required=True, choices=("UAG", "GGA"))
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="override input CSV (default: outputs/motif_substitutions/{exon}_{motif}_1x_15x.csv)",
    )
    args = parser.parse_args(argv)

    motif_csv = args.csv if args.csv is not None else (
        MOTIF_DIR / f"{args.exon}_{args.motif}_1x_15x.csv"
    )
    if not motif_csv.exists():
        print(
            f"[missing] {motif_csv} not found. Run:\n"
            f"  python analysis_script/run_tp53_motif_substitutions.py "
            f"--exon {args.exon} --motifs {args.motif} --copy-counts 1-15 --alphagenome"
        )
        return 1

    df = pd.read_csv(motif_csv)
    is_ref = df["is_reference"].fillna(False).astype(bool)
    wt_row = df[is_ref].iloc[0]
    variants = df[~is_ref].copy()

    prod_col, _prod_short, _prod_label = _METRIC_BY_SHORT["prod_logit"]

    out_path = OUTPUT_DIR / f"{args.exon}_{args.motif}_1x_15x_pnas_vs_alphagenome.png"
    _plot_main_response(variants, args.exon, args.motif, out_path)
    _print_summary(variants, wt_row, prod_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
