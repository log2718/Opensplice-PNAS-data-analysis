"""
Plot PNAS pre-tuner scores against AlphaGenome splice-site scores for the
TP53_e7 UAG tandem-motif substitution experiment extended out to 1x-15x
copies, produced by
``run_tp53_motif_substitutions.py --exon TP53_e7 --motifs UAG --copy-counts 1-15``.

Scientific question
--------------------
The original TP53_e7 GGA/UAG 1x-4x experiment already pushed UAG's PNAS
minimum below the random-triple-mutant TP53_e7 floor. This script asks
whether pushing UAG much farther (up to 15x, 45 nt of the 110-nt exon)
keeps moving the PNAS distribution lower, or whether it saturates -- and
whether AlphaGenome's still-mostly-flat response starts to change as PNAS
is pushed further down. It only reports what the two models produce; it
does not assume UAG has its intended biological effect, and it does not
draw conclusions beyond what the printed statistics show.

Input
-----
``outputs/motif_substitutions/TP53_e7_UAG_1x_15x.csv`` -- 1,305 exhaustive
UAG tandem-repeat placements (1x-15x) + 1 WT reference row, written by
``run_tp53_motif_substitutions.py``.

``outputs/triple_substitutions/TP53_e7_merged.csv`` -- read only, for the
established random-triple-mutant TP53_e7 PNAS range.

``outputs/motif_substitutions/TP53_e7_GGA_UAG_1x_4x.csv`` -- read only, for
the established UAG 1x-4x PNAS range from the original motif experiment.

Neither existing CSV is modified. This script only reads CSVs and draws
figures -- no prediction code is imported, rerun, or modified. Panel drawing
and stats helpers are imported from ``plot_tp53_motif_substitutions`` rather
than re-implemented.

Output -- exactly one PNG
----------------------------
1. ``outputs/plots/motif_substitutions/TP53_e7_UAG_1x_15x_pnas_vs_alphagenome.png``
   -- 2-panel hist2d (left: mean_logit, right: prod_logit), same
   Blues/LogNorm/"Count"-colorbar/red-OLS/white-stats-box style used
   throughout this repo (drawn by the exact same shared ``_panel`` helper as
   every other hist2d figure in this repo). No scatter, no WT marker.

A printed summary (per-copy-count stats, WT PNAS, overall UAG 1x-15x PNAS
range, comparisons against the triple-mutant and UAG 1x-4x ranges, and
whether AlphaGenome responsiveness changes at higher copy counts) follows
the plot. The per-copy-count PNAS and AlphaGenome-response breakdowns are
reported in this printed table rather than as separate figures, to keep the
plotting scope to the one essential figure.
"""

from __future__ import annotations

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

MOTIF_CSV = ROOT / "outputs" / "motif_substitutions" / "TP53_e7_UAG_1x_15x.csv"
TRIPLE_CSV = ROOT / "outputs" / "triple_substitutions" / "TP53_e7_merged.csv"
UAG_1X_4X_CSV = ROOT / "outputs" / "motif_substitutions" / "TP53_e7_GGA_UAG_1x_4x.csv"

OUTPUT_DIR = ROOT / "outputs" / "plots" / "motif_substitutions"
MAIN_PNG = OUTPUT_DIR / "TP53_e7_UAG_1x_15x_pnas_vs_alphagenome.png"

X_COL = motif_plot.X_COL
X_LABEL = motif_plot.X_LABEL
ALPHAGENOME_METRICS = motif_plot.ALPHAGENOME_METRICS
_METRIC_BY_SHORT = motif_plot._METRIC_BY_SHORT
_panel = motif_plot._panel
_clean_xy = motif_plot._clean_xy
_regression_stats = motif_plot._regression_stats

COPY_COUNTS = tuple(range(1, 16))


# --------------------------------------------------------------------------
# Figure 1: 2-panel hist2d (mean_logit | prod_logit), all copy counts pooled
# --------------------------------------------------------------------------

def _plot_main_response(variants: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        f"TP53_e7  --  UAG motif substitutions, 1x-15x copies\n"
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
# Printed summary
# --------------------------------------------------------------------------

def _print_summary(variants: pd.DataFrame, wt_row: pd.Series, prod_col: str) -> None:
    print("\n" + "=" * 100)
    print("Per copy_count summary (UAG 1x-15x)")
    print("=" * 100)

    rows = []
    for cc in COPY_COUNTS:
        sub = variants[variants["copy_count"] == cc]
        if sub.empty:
            continue
        x, y = _clean_xy(sub, X_COL, prod_col)
        reg = _regression_stats(x, y) if len(x) > 1 else {"slope": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
        rows.append({
            "copy_count": cc,
            "n": len(sub),
            "pnas_min": sub[X_COL].min(),
            "pnas_max": sub[X_COL].max(),
            "pnas_mean": sub[X_COL].mean(),
            "pnas_median": sub[X_COL].median(),
            "mean_delta_from_wt": (sub[X_COL] - wt_row[X_COL]).mean(),
            "ag_prod_logit_min": sub[prod_col].min(),
            "ag_prod_logit_max": sub[prod_col].max(),
            "ols_slope": reg["slope"],
            "pearson_r": reg["pearson_r"],
            "spearman_rho": reg["spearman_rho"],
        })
    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 220):
        print(summary.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("Overall ranges")
    print("=" * 100)
    wt_pnas = float(wt_row[X_COL])
    print(f"WT e7 PNAS pre-tuner: {wt_pnas:.4f}")

    all_pnas = variants[X_COL]
    overall_lo, overall_hi = float(all_pnas.min()), float(all_pnas.max())
    print(f"Overall UAG 1x-15x PNAS range: [{overall_lo:.4f}, {overall_hi:.4f}]  (n={len(all_pnas):,})")
    print(f"Lowest PNAS value reached: {overall_lo:.4f}")

    # ---- vs random-triple-mutant floor ------------------------------------
    if TRIPLE_CSV.exists():
        triple = pd.read_csv(TRIPLE_CSV)
        triple_variants = triple[triple["is_reference"].fillna(False).astype(bool).ne(True)]
        t_lo = float(triple_variants[X_COL].min())
        print(f"\nRandom triple-mutant TP53_e7 PNAS min (from {TRIPLE_CSV.relative_to(ROOT)}): {t_lo:.4f}")
        first_cc = None
        for cc in COPY_COUNTS:
            sub = variants[variants["copy_count"] == cc]
            if not sub.empty and sub[X_COL].min() < t_lo:
                first_cc = cc
                break
        print(f"First copy count with min PNAS below the triple-mutant floor ({t_lo:.4f}): "
              f"{first_cc if first_cc is not None else 'none reached'}")
    else:
        print(f"\n[note] {TRIPLE_CSV} not found -- skipping triple-mutant floor comparison.")

    # ---- vs original UAG 1x-4x floor --------------------------------------
    if UAG_1X_4X_CSV.exists():
        orig = pd.read_csv(UAG_1X_4X_CSV)
        orig_uag = orig[(orig["motif"] == "UAG") & (orig["is_reference"].fillna(False).astype(bool).ne(True))]
        u_lo = float(orig_uag[X_COL].min())
        print(f"\nOriginal UAG 1x-4x PNAS min (from {UAG_1X_4X_CSV.relative_to(ROOT)}): {u_lo:.4f}")
        first_cc = None
        for cc in COPY_COUNTS:
            sub = variants[variants["copy_count"] == cc]
            if not sub.empty and sub[X_COL].min() < u_lo:
                first_cc = cc
                break
        print(f"First copy count with min PNAS below the original UAG 1x-4x floor ({u_lo:.4f}): "
              f"{first_cc if first_cc is not None else 'none reached'}")
    else:
        print(f"\n[note] {UAG_1X_4X_CSV} not found -- skipping UAG 1x-4x floor comparison.")

    # ---- does PNAS keep shifting lower, or saturate? ----------------------
    print("\n" + "=" * 100)
    print("Does PNAS keep shifting lower through 15x, or does it saturate?")
    print("=" * 100)
    print("Per-copy-count change in mean/min PNAS vs the previous copy count "
          "(consistently negative = still shifting lower; near zero / positive = leveling off):")
    prev_mean = prev_min = None
    for cc in COPY_COUNTS:
        sub = variants[variants["copy_count"] == cc]
        if sub.empty:
            continue
        mean_pnas = float(sub[X_COL].mean())
        min_pnas = float(sub[X_COL].min())
        if prev_mean is None:
            print(f"  {cc:>2}x: mean={mean_pnas:.4f}  min={min_pnas:.4f}  (first copy count)")
        else:
            print(f"  {cc:>2}x: mean={mean_pnas:.4f} (Δ={mean_pnas - prev_mean:+.4f})  "
                  f"min={min_pnas:.4f} (Δ={min_pnas - prev_min:+.4f})")
        prev_mean, prev_min = mean_pnas, min_pnas

    # ---- does AlphaGenome responsiveness change at larger copy counts? ----
    print("\n" + "=" * 100)
    print("Does AlphaGenome responsiveness change at lower PNAS / larger UAG copy counts?")
    print("=" * 100)
    print("Per-copy-count OLS slope and Pearson r vs PNAS (from the table above); "
          "also pooled over the low-copy-count half (1x-7x) vs high-copy-count half (8x-15x):")
    low_half = variants[variants["copy_count"] <= 7]
    high_half = variants[variants["copy_count"] >= 8]
    x_lo, y_lo = _clean_xy(low_half, X_COL, prod_col)
    x_hi, y_hi = _clean_xy(high_half, X_COL, prod_col)
    reg_lo = _regression_stats(x_lo, y_lo)
    reg_hi = _regression_stats(x_hi, y_hi)
    print(f"  1x-7x  (n={reg_lo['n']:,}): slope={reg_lo['slope']:+.4f}  Pearson r={reg_lo['pearson_r']:+.4f}  "
          f"Spearman rho={reg_lo['spearman_rho']:+.4f}")
    print(f"  8x-15x (n={reg_hi['n']:,}): slope={reg_hi['slope']:+.4f}  Pearson r={reg_hi['pearson_r']:+.4f}  "
          f"Spearman rho={reg_hi['spearman_rho']:+.4f}")


def main() -> None:
    if not MOTIF_CSV.exists():
        print(f"[missing] {MOTIF_CSV} not found. Run:\n"
              f"  python analysis_script/run_tp53_motif_substitutions.py "
              f"--exon TP53_e7 --motifs UAG --copy-counts 1-15 --alphagenome")
        return

    df = pd.read_csv(MOTIF_CSV)
    is_ref = df["is_reference"].fillna(False).astype(bool)
    wt_row = df[is_ref].iloc[0]
    variants = df[~is_ref].copy()

    prod_col, _prod_short, _prod_label = _METRIC_BY_SHORT["prod_logit"]

    _plot_main_response(variants, MAIN_PNG)

    _print_summary(variants, wt_row, prod_col)


if __name__ == "__main__":
    main()
