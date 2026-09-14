"""
Plot PNAS pre-tuner scores against AlphaGenome splice-site scores for every
TP53 hybrid construct produced by ``run_hybrid_pnas_alphagenome.py``.

Input
-----
``outputs/tp53_hybrid_scores_with_alphagenome.csv`` -- the saturation-mutagenesis
table with both models scored (one row per variant, columns ``construct_label``,
``pnas_pretuner``, ``alphagenome_mean_logit``, ``alphagenome_prod_logit`` ...).

Output
------
* ``outputs/plots/hybrid/<construct>_pnas_vs_alphagenome.png`` -- one 2-panel
  figure per ``construct_label``:

    left  : pnas_pretuner  vs  alphagenome_mean_logit
    right : pnas_pretuner  vs  alphagenome_prod_logit

  Each panel is a 2D histogram / density plot (``hist2d``, ``Blues``,
  ``cmin=1``, ``LogNorm`` colour scale -- the style used by the existing
  OpenSplice / SpliceAI notebooks in this repo) with an OLS regression line
  overlaid and a text box reporting ``n``, regression slope, Pearson r and
  Spearman rho.

* ``outputs/tp53_hybrid_slope_summary.csv`` -- slope / intercept / Pearson /
  Spearman for both AlphaGenome definitions, one row per
  (construct, alphagenome_metric).

WT variant subset (matching the original TP53 AlphaGenome-vs-PNAS plots)
----------------------------------------------------------------------
The earlier TP53 plots (``outputs/plots/alphagenome_vs_pnas/TP53_e{6,7}_alphagenome_vs_pnas.png``,
from ``analysis_script/alphagenome_minigene_figshare_analysis.ipynb``) reported

    TP53_e6: n = 326      TP53_e7: n = 318

Those n are NOT a full in-silico saturation. The old pipeline merged the PNAS
pre-tuner table with the AlphaGenome minigene table on ``variant_id`` and then
kept only rows with ``region == "Exon"`` and ``mut_type == "sub"`` -- i.e. the
**experimentally measured exonic single-nucleotide substitutions** of the
OpenSplice / Vaz-Drago minigene saturation library. That designed library does
not contain every exon position x every alternate allele:

* the first exonic nt is annotated ``region == "3'SS"`` (not ``"Exon"``) and is
  dropped;
* the last few exon positions are annotated ``region == "5'SS"`` and carry no
  ``Exon`` substitutions;
* a handful of interior positions are missing one or two of the three alt
  alleles in the library design.

Full saturation would be 113*3 = 339 (e6) and 110*3 = 330 (e7); the library
covers 326 and 318 of those. The new hybrid pipeline emits the *complete*
in-silico saturation plus a reference row (340 / 331), a strict superset.

To keep the WT slopes directly comparable, this script restricts the
``construct_type == "wt"`` groups (TP53_e6, TP53_e7) to exactly that library
subset, matched on ``(exon position, ref base, alt base)`` read from
``outputs/test_exons_with_pretuner.csv`` (``region == "Exon"``,
``mut_type == "sub"``).

Hybrid constructs: same library rule, mapped through the construct geometry
--------------------------------------------------------------------------
``run_hybrid_pnas_alphagenome`` builds the swap exons as (verified base-by-base
against the scored CSV, not assumed from the nominal 3-nt SS margins):

* ``3ss_swap``  exon = ``e6[:3] + e7[3:]``   -> exon positions >= 3 are the
  recipient TP53_e7 exon **at the same index**;
* ``5ss_swap``  exon = ``e7[:-3] + e6[-3:]`` -> exon positions <= 106 are the
  recipient TP53_e7 exon **at the same index**;
* ``3ss_5ss_swap`` exon = ``e6[:3] + e7[3:-3] + e6[-3:]`` -> the exon middle
  (positions 3..106) is the recipient TP53_e7 exon **at the same index**; only
  the two 3-nt SS ends are donor-derived;
* ``full_internal_swap`` exon = ``e7[:3] + e6[3:110] + e7[107:110]``; only the
  donor middle is mutagenised (positions 3..109), and those are the donor
  TP53_e6 exon **at the same index**.

Every head/prefix length is 3, so no coordinate offset is needed; the
``(position, ref, alt)`` triple is matched directly against the relevant WT
library key set -- TP53_e7 for the 3'SS / 5'SS / combined-SS swaps, TP53_e6 for
the internal swap. A donor-derived position in an SS swap only survives if its
base happens to equal the recipient's there *and* that exact triple is a real e7
library variant (i.e. it genuinely "would have been in the e7 set").

Reference (unmutated) rows are dropped from every regression, matching the old
scatter plots.

This script only reads CSVs and draws figures. No prediction code is imported
or modified.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import linregress, pearsonr, spearmanr

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT / "outputs" / "tp53_hybrid_scores_with_alphagenome.csv"
OUTPUT_DIR = ROOT / "outputs" / "plots" / "hybrid"
SUMMARY_CSV = ROOT / "outputs" / "tp53_hybrid_slope_summary.csv"

# Experimental minigene saturation library (PNAS-side table used by the original
# plots). Its ``start`` column is a 1-based index into ``nt_seq`` = upstream
# intron (70 nt) + exon + downstream intron (25 nt), so exon position 0 == start
# UPSTREAM_LEN + 1.
PNAS_LIBRARY_CSV = ROOT / "outputs" / "test_exons_with_pretuner.csv"
UPSTREAM_LEN = 70  # matches run_hybrid_pnas_alphagenome.DEFAULT_UPSTREAM_LEN

X_COL = "pnas_pretuner"
X_LABEL = "PNAS pre-tuner score\n(energy_seq_struct)"

# (column, short name, axis label) for the two AlphaGenome definitions.
ALPHAGENOME_METRICS = [
    (
        "alphagenome_mean_logit",
        "mean_logit",
        "AlphaGenome mean logit:\n(logit P(acc) + logit P(don)) / 2",
    ),
    (
        "alphagenome_prod_logit",
        "prod_logit",
        "AlphaGenome product logit:\nlogit(P(acc) × P(don))",
    ),
]

BINS = 40


def _slug(text: str) -> str:
    """Filesystem-safe token for a construct label."""
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", text)).strip("_")


@lru_cache(maxsize=1)
def _library_exonic_subs() -> pd.DataFrame:
    """Experimental exonic single-nt substitutions, one row per (exon, variant).

    Columns: ``exon_id``, ``pos0`` (0-based exon index), ``ref``, ``alt``
    (transcript-sense, T not U).
    """
    lib = pd.read_csv(
        PNAS_LIBRARY_CSV,
        usecols=["exon_id", "start", "wt", "mut", "mut_type", "region"],
    )
    lib = lib[(lib["region"] == "Exon") & (lib["mut_type"] == "sub")].copy()
    lib["pos0"] = lib["start"].astype(int) - (UPSTREAM_LEN + 1)
    lib["ref"] = lib["wt"].astype(str).str.upper().str.replace("U", "T", regex=False)
    lib["alt"] = lib["mut"].astype(str).str.upper().str.replace("U", "T", regex=False)
    return lib[["exon_id", "pos0", "ref", "alt"]]


def _experimental_exonic_sub_keys(exon_id: str, exon_seq: str) -> set[tuple[int, str, str]]:
    """``(pos0, ref, alt)`` keys of the library exonic subs for ``exon_id``.

    ``ref`` is checked against ``exon_seq`` so a coordinate mismatch fails loudly
    rather than silently dropping every variant.
    """
    lib = _library_exonic_subs()
    lib = lib[lib["exon_id"] == exon_id]
    if lib.empty:
        raise ValueError(
            f"No exonic substitutions for {exon_id!r} in {PNAS_LIBRARY_CSV.name}."
        )

    keys: set[tuple[int, str, str]] = set()
    for pos0, ref, alt in zip(lib["pos0"], lib["ref"], lib["alt"]):
        pos0 = int(pos0)
        if not 0 <= pos0 < len(exon_seq):
            raise ValueError(f"{exon_id}: library pos0 {pos0} outside exon.")
        if exon_seq[pos0] != ref:
            raise ValueError(
                f"{exon_id}: library ref {ref} at pos0 {pos0} != exon_seq "
                f"base {exon_seq[pos0]} -- coordinate systems disagree."
            )
        keys.add((pos0, ref, alt))
    return keys


def _exon_snv_rows(group: pd.DataFrame) -> pd.DataFrame:
    """Non-reference exon-body single-nt variants of a construct group."""
    snv = group[
        (group["mutation_region"] == "exon")
        & group["is_reference"].ne(True)
    ].dropna(subset=["position0", "ref", "alt"]).copy()
    snv["position0"] = snv["position0"].astype(int)
    return snv


def _library_subset(
    group: pd.DataFrame, keys: set[tuple[int, str, str]]
) -> pd.DataFrame:
    """Keep only exon SNVs whose ``(position0, ref, alt)`` is a library key.

    ``keys`` is built from a WT exon (:func:`_experimental_exonic_sub_keys`);
    for hybrid constructs the construct's own exon indices line up with that WT
    exon over the relevant region (see module docstring), so membership is a
    direct triple test with no offset.
    """
    snv = _exon_snv_rows(group)
    keep = [
        (p, r, a) in keys
        for p, r, a in zip(snv["position0"], snv["ref"], snv["alt"])
    ]
    return snv[keep]


def _panel(ax, x: np.ndarray, y: np.ndarray, y_label: str, fig) -> dict:
    """Draw one hist2d + regression panel; return the fit statistics."""
    lin = linregress(x, y)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    hist = ax.hist2d(
        x,
        y,
        bins=BINS,
        cmap="Blues",
        cmin=1,
        norm=LogNorm(),
    )
    fig.colorbar(hist[3], ax=ax, label="Count")

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        lin.intercept + lin.slope * x_line,
        color="crimson",
        lw=1.5,
        label="OLS fit",
    )

    ax.text(
        0.02,
        0.98,
        f"n = {len(x):,}\n"
        f"Slope = {lin.slope:.3f}\n"
        f"Pearson r = {pearson.statistic:.3f}\n"
        f"Spearman ρ = {spearman.statistic:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )
    ax.set_xlabel(X_LABEL)
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


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Library key sets, one per WT exon, validated against that WT exon body.
    wt_exon_seq = {
        row["construct_label"]: row["exon_seq"]
        for _, row in df[
            (df["construct_type"] == "wt") & (df["is_reference"] == True)  # noqa: E712
        ].iterrows()
    }
    lib_keys = {
        exon_id: _experimental_exonic_sub_keys(exon_id, seq)
        for exon_id, seq in wt_exon_seq.items()
    }

    summary_rows: list[dict] = []

    for construct_label, group in df.groupby("construct_label", sort=False):
        construct_type = group["construct_type"].iloc[0]
        recipient_exon = group["recipient_exon"].iloc[0]
        donor_exon = group["donor_exon"].iloc[0]

        # Which WT library governs this construct, and over which sequence.
        if construct_type == "wt":
            lib_exon = construct_label
            subset = (
                f"experimental exonic subs — {lib_exon} library "
                f"(region==Exon & mut_type==sub)"
            )
        elif construct_type in ("3ss_swap", "5ss_swap", "3ss_5ss_swap"):
            lib_exon = recipient_exon  # recipient TP53_e7, same exon indices
            subset = (
                f"experimental exonic subs — recipient {lib_exon} library "
                f"(region==Exon & mut_type==sub)"
            )
        elif construct_type == "full_internal_swap":
            lib_exon = donor_exon  # donor TP53_e6, over the internal region
            subset = (
                f"experimental exonic subs — donor {lib_exon} library, "
                f"internal region (region==Exon & mut_type==sub)"
            )
        else:
            lib_exon = None
            subset = "all exon SNVs (no library filter)"

        if lib_exon is not None and lib_exon in lib_keys:
            plot_df = _library_subset(group, lib_keys[lib_exon])
        else:
            plot_df = _exon_snv_rows(group)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        fig.suptitle(
            f"{construct_label}   ({construct_type})\n"
            f"PNAS pre-tuner vs AlphaGenome  —  {subset}"
        )

        for ax, (col, short, y_label) in zip(axes, ALPHAGENOME_METRICS):
            pair = (
                plot_df[[X_COL, col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            x = pair[X_COL].to_numpy()
            y = pair[col].to_numpy()

            stats = _panel(ax, x, y, y_label, fig)
            summary_rows.append(
                {
                    "construct_label": construct_label,
                    "construct_type": construct_type,
                    "library_exon": lib_exon,
                    "variant_subset": subset,
                    "alphagenome_metric": short,
                    "alphagenome_column": col,
                    **stats,
                }
            )

        out = OUTPUT_DIR / f"{_slug(construct_label)}_pnas_vs_alphagenome.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    summary = pd.DataFrame(summary_rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved {SUMMARY_CSV}  ({len(summary)} rows)\n")
    print(
        summary[
            ["construct_label", "library_exon", "alphagenome_metric", "n",
             "slope", "pearson_r", "spearman_rho"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
