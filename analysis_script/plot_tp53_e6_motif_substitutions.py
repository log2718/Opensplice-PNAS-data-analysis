"""
Plot PNAS pre-tuner scores against AlphaGenome splice-site scores for the
TP53_e6 targeted GGA / UAG tandem-motif substitution experiment produced by
``run_tp53_motif_substitutions.py --exon TP53_e6``.

Same experiment shape as the original TP53_e7 motif experiment
(``plot_tp53_motif_substitutions.py``): GGA and UAG, each tested as 1x-4x
contiguous tandem repeats, exhaustively slid across every valid exon-only
start position (flanks / splice sites untouched).

Scientific question
--------------------
TP53_e6's AlphaGenome response is currently *responsive* (not flat, unlike
TP53_e7) across the single/double/triple random-substitution experiments.
This script asks whether targeted motifs -- GGA (upward-targeted) and UAG
(downward-targeted) -- push that relationship further, and whether it stays
responsive or begins to flatten. It only reports what the two models
produce; it does not assume GGA/UAG have their intended biological effect.

Input
-----
``outputs/motif_substitutions/TP53_e6_GGA_UAG_1x_4x.csv`` -- 852 exhaustive
GGA/UAG tandem-repeat placements (1x-4x) + 1 WT reference row, written by
``run_tp53_motif_substitutions.py``.

``outputs/triple_substitutions/TP53_e6_merged.csv`` -- read only, for the
already-established random-triple-mutant TP53_e6 PNAS range; not modified.

This script only reads CSVs and draws figures -- no prediction code is
imported, rerun, or modified. Panel drawing and stats helpers are imported
from ``plot_tp53_motif_substitutions`` (the TP53_e7 motif-plotting script)
rather than re-implemented.

Output -- exactly three PNGs
------------------------------
1. ``outputs/plots/motif_substitutions/TP53_e6_GGA_UAG_pnas_vs_alphagenome.png``
   -- both motifs pooled, 2-panel hist2d (left: mean_logit, right: prod_logit),
   same Blues/LogNorm/"Count"-colorbar/red-OLS/white-stats-box style used
   throughout this repo. No scatter, no WT marker.
2. ``outputs/plots/motif_substitutions/TP53_e6_GGA_vs_UAG.png`` -- 2-panel
   hist2d, GGA (green sequential colormap) vs UAG (red/violet sequential
   colormap), PNAS vs AlphaGenome prod_logit, red OLS line, no WT marker.
3. ``outputs/plots/motif_substitutions/TP53_e6_motif_copycount_pnas.png`` --
   one compact grouped boxplot of PNAS pretuner by motif x copy_count (1x-4x),
   with a dashed WT reference line.

A printed summary (per motif x copy_count stats, WT PNAS, GGA/UAG overall
PNAS ranges, and whether either exceeds the established random-triple-mutant
TP53_e6 PNAS range) follows the plots.
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

# Reuse the hist2d panel + stats helpers + boxplot/color conventions from the
# TP53_e7 motif-plotting script, rather than re-implementing them.
import plot_tp53_motif_substitutions as motif_plot  # noqa: E402

plt = motif_plot.plt
Patch = motif_plot.Patch

MOTIF_CSV = ROOT / "outputs" / "motif_substitutions" / "TP53_e6_GGA_UAG_1x_4x.csv"
TRIPLE_CSV = ROOT / "outputs" / "triple_substitutions" / "TP53_e6_merged.csv"

OUTPUT_DIR = ROOT / "outputs" / "plots" / "motif_substitutions"
MAIN_PNG = OUTPUT_DIR / "TP53_e6_GGA_UAG_pnas_vs_alphagenome.png"
GGA_VS_UAG_PNG = OUTPUT_DIR / "TP53_e6_GGA_vs_UAG.png"
COPYCOUNT_PNG = OUTPUT_DIR / "TP53_e6_motif_copycount_pnas.png"

X_COL = motif_plot.X_COL
X_LABEL = motif_plot.X_LABEL
ALPHAGENOME_METRICS = motif_plot.ALPHAGENOME_METRICS
_METRIC_BY_SHORT = motif_plot._METRIC_BY_SHORT
_panel = motif_plot._panel
_clean_xy = motif_plot._clean_xy
_regression_stats = motif_plot._regression_stats
MOTIF_COLOR = motif_plot.MOTIF_COLOR
COPY_COUNTS = motif_plot.COPY_COUNTS  # (1, 2, 3, 4)


# --------------------------------------------------------------------------
# Figure 1: both motifs pooled, 2-panel hist2d (mean_logit | prod_logit)
# --------------------------------------------------------------------------

def _plot_main_response(variants: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        f"TP53_e6  --  targeted GGA + UAG motif substitutions\n"
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
# Figure 2: GGA vs UAG, side by side, PNAS vs prod_logit
# --------------------------------------------------------------------------

def _plot_gga_vs_uag(variants: pd.DataFrame, prod_col: str, prod_label: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    cmaps = {"GGA": "Greens", "UAG": "RdPu"}
    for ax, motif in zip(axes, ("GGA", "UAG")):
        sub = variants[variants["motif"] == motif]
        x, y = _clean_xy(sub, X_COL, prod_col)
        _panel(ax, x, y, prod_label, fig, cmap=cmaps[motif])
        ax.set_title(f"{motif}  (n={len(x):,})")

    fig.suptitle(
        "TP53_e6  --  GGA (upward-targeted) vs UAG (downward-targeted) motif substitutions\n"
        "PNAS pre-tuner vs AlphaGenome product logit"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Figure 3: copy-count progression, grouped boxplot
# --------------------------------------------------------------------------

def _plot_copycount_progression(variants: pd.DataFrame, wt_pnas: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    offset = 0.18
    for motif, dx in (("GGA", -offset), ("UAG", offset)):
        data = [
            variants.loc[
                (variants["motif"] == motif) & (variants["copy_count"] == cc), X_COL
            ].to_numpy()
            for cc in COPY_COUNTS
        ]
        positions = [cc + dx for cc in COPY_COUNTS]
        bp = ax.boxplot(
            data, positions=positions, widths=0.3, patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for box in bp["boxes"]:
            box.set_facecolor(MOTIF_COLOR[motif])
            box.set_alpha(0.6)

    ax.axhline(wt_pnas, color="crimson", linestyle="--", linewidth=1.4,
               label=f"WT reference (PNAS = {wt_pnas:.3f})")

    ax.set_xticks(COPY_COUNTS)
    ax.set_xticklabels([f"{cc}x" for cc in COPY_COUNTS])
    ax.set_xlabel("Motif copy count")
    ax.set_ylabel(X_LABEL)
    ax.set_title(
        "TP53_e6  --  PNAS pre-tuner vs motif copy count\n"
        "does GGA push PNAS up and UAG push PNAS down as copies increase?"
    )

    handles = [
        Patch(facecolor=MOTIF_COLOR["GGA"], alpha=0.6, label="GGA"),
        Patch(facecolor=MOTIF_COLOR["UAG"], alpha=0.6, label="UAG"),
    ]
    handles.append(ax.get_legend_handles_labels()[0][-1])  # WT reference line
    ax.legend(handles=handles, fontsize=9, loc="best", framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Printed summary
# --------------------------------------------------------------------------

def _print_summary(variants: pd.DataFrame, wt_row: pd.Series, prod_col: str) -> None:
    print("\n" + "=" * 90)
    print("Per motif x copy_count summary")
    print("=" * 90)

    rows = []
    for motif in ("GGA", "UAG"):
        for cc in COPY_COUNTS:
            sub = variants[(variants["motif"] == motif) & (variants["copy_count"] == cc)]
            if sub.empty:
                continue
            x, y = _clean_xy(sub, X_COL, prod_col)
            reg = _regression_stats(x, y) if len(x) > 1 else {"slope": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
            rows.append({
                "motif": motif,
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
    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string(index=False))

    print("\n" + "=" * 90)
    print("Overall ranges")
    print("=" * 90)
    wt_pnas = float(wt_row[X_COL])
    print(f"WT e6 PNAS pre-tuner: {wt_pnas:.4f}")

    gga = variants[variants["motif"] == "GGA"][X_COL]
    uag = variants[variants["motif"] == "UAG"][X_COL]
    print(f"GGA overall PNAS range: [{gga.min():.4f}, {gga.max():.4f}]  (n={len(gga):,})")
    print(f"UAG overall PNAS range: [{uag.min():.4f}, {uag.max():.4f}]  (n={len(uag):,})")

    if TRIPLE_CSV.exists():
        triple = pd.read_csv(TRIPLE_CSV)
        triple_variants = triple[triple["is_reference"].fillna(False).astype(bool).ne(True)]
        t_lo, t_hi = float(triple_variants[X_COL].min()), float(triple_variants[X_COL].max())
        print(f"\nRandom triple-mutant TP53_e6 PNAS range (from {TRIPLE_CSV.relative_to(ROOT)}): "
              f"[{t_lo:.4f}, {t_hi:.4f}]  (n={len(triple_variants):,})")

        gga_exceeds_hi = gga.max() > t_hi
        gga_exceeds_lo = gga.min() < t_lo
        uag_exceeds_hi = uag.max() > t_hi
        uag_exceeds_lo = uag.min() < t_lo

        print(
            f"GGA exceeds triple range on the high end: {gga_exceeds_hi} "
            f"({gga.max():.4f} vs triple max {t_hi:.4f})"
        )
        print(
            f"GGA exceeds triple range on the low end:  {gga_exceeds_lo} "
            f"({gga.min():.4f} vs triple min {t_lo:.4f})"
        )
        print(
            f"UAG exceeds triple range on the high end: {uag_exceeds_hi} "
            f"({uag.max():.4f} vs triple max {t_hi:.4f})"
        )
        print(
            f"UAG exceeds triple range on the low end:  {uag_exceeds_lo} "
            f"({uag.min():.4f} vs triple min {t_lo:.4f})"
        )
        any_exceeds = gga_exceeds_hi or gga_exceeds_lo or uag_exceeds_hi or uag_exceeds_lo
        print(
            f"\n=> Targeted motif substitutions {'DO' if any_exceeds else 'do NOT'} "
            f"extend beyond the random triple-mutant TP53_e6 PNAS range."
        )
    else:
        print(f"\n[note] {TRIPLE_CSV} not found -- skipping triple-mutant range comparison.")


def main() -> None:
    if not MOTIF_CSV.exists():
        print(f"[missing] {MOTIF_CSV} not found. Run:\n"
              f"  python analysis_script/run_tp53_motif_substitutions.py "
              f"--exon TP53_e6 --motifs GGA,UAG --copy-counts 1-4 --alphagenome")
        return

    df = pd.read_csv(MOTIF_CSV)
    is_ref = df["is_reference"].fillna(False).astype(bool)
    wt_row = df[is_ref].iloc[0]
    variants = df[~is_ref].copy()

    prod_col, _prod_short, prod_label = _METRIC_BY_SHORT["prod_logit"]

    _plot_main_response(variants, MAIN_PNG)
    _plot_gga_vs_uag(variants, prod_col, prod_label, GGA_VS_UAG_PNG)
    _plot_copycount_progression(variants, float(wt_row[X_COL]), COPYCOUNT_PNG)

    _print_summary(variants, wt_row, prod_col)


if __name__ == "__main__":
    main()
