"""
Compare the AlphaGenome SPLICE_SITES-derived ``prod_logit`` against
*experimental logit(PSI)* for the PNAS three-exon cassette.

Motivation
----------
Against raw PSI, ``prod_logit`` looked compressed at high inclusion. Raw PSI
is itself compressed near 0 and 1, so that apparent ceiling may be an artifact
of the x-axis rather than a property of the AlphaGenome metric. Putting the
experimental measurement on the logit scale -- which expands both extremes --
tests whether the compression survives.

``prod_logit`` is NOT PSI. It is::

    prod_logit = logit(acceptor_probability * donor_probability)

with both probabilities taken from AlphaGenome ``SPLICE_SITES``.

Input (read-only, no API calls)
--------------------------------
``outputs/alphagenome_pnas_sanity/pnas_fulltrack_splice_site_sanity.csv``

Exact zeros and ones
--------------------
``metadata_PSI`` contains exact 0 and exact 1 values, whose logit is infinite.
They are **not clipped**. They are excluded from the finite-logit analysis and
reported separately, with counts, so nothing is silently discarded. The
original PSI column is never modified.

Output
------
``outputs/plots/alphagenome_pnas_sanity/logit_psi_vs_prod_logit.png``
-- 2-panel hist2d (train | test), shared axes, repo Blues/LogNorm/"Count"
style, with a binned-median curve overlaid instead of a global OLS fit.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

CSV = ROOT / "outputs" / "alphagenome_pnas_sanity" / "pnas_fulltrack_splice_site_sanity.csv"
OUT_PNG = ROOT / "outputs" / "plots" / "alphagenome_pnas_sanity" / "logit_psi_vs_prod_logit.png"

N_SMOOTH_BINS = 20   # equal-count bins for the overlaid median curve
N_TABLE_BINS = 10    # deciles for the quantitative compression table


def load() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    psi = d["metadata_PSI"]
    print(f"loaded {len(d)} rows from {CSV.relative_to(ROOT)}")
    print("\n=== exact 0 / 1 PSI values (infinite logit -- NOT clipped) ===")
    for split in ("train", "test"):
        s = d[d["split"] == split]
        print(f"  {split:5}: PSI==0 -> {(s['metadata_PSI']==0).sum():4d}   "
              f"PSI==1 -> {(s['metadata_PSI']==1).sum():4d}   "
              f"finite -> {((s['metadata_PSI']>0)&(s['metadata_PSI']<1)).sum():4d}   (n={len(s)})")
    print(f"  TOTAL: PSI==0 -> {(psi==0).sum()}, PSI==1 -> {(psi==1).sum()}, "
          f"finite -> {((psi>0)&(psi<1)).sum()}")
    print("  -> exact 0/1 excluded from the finite-logit analysis and plot; "
          "reported separately below.")
    return d


def add_logit(d: pd.DataFrame) -> pd.DataFrame:
    """Add logit_psi for strictly interior PSI only; original column untouched."""
    out = d.copy()
    psi = out["metadata_PSI"]
    interior = (psi > 0) & (psi < 1)
    out["psi_is_exact_0"] = psi == 0
    out["psi_is_exact_1"] = psi == 1
    out["logit_psi"] = np.where(interior, np.log(psi.where(interior) / (1 - psi.where(interior))), np.nan)
    return out


def correlations(d: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("4. Relationship strength on the logit scale (finite logit(PSI) only)")
    print("=" * 100)
    for split in ("train", "test"):
        s = d[(d["split"] == split) & d["logit_psi"].notna()]
        x, y = s["logit_psi"].to_numpy(), s["prod_logit"].to_numpy()
        rho, _ = spearmanr(x, y)
        r, _ = pearsonr(x, y)
        print(f"\n[{split}]  n={len(s)}")
        print(f"  Spearman(logit_psi, prod_logit) = {rho:+.4f}")
        print(f"  Pearson (logit_psi, prod_logit) = {r:+.4f}")
        for nm, v in (("logit(PSI)", s["logit_psi"]), ("prod_logit", s["prod_logit"])):
            q1, q3 = v.quantile(.25), v.quantile(.75)
            print(f"  {nm:11}: range=[{v.min():+.3f}, {v.max():+.3f}] (width {v.max()-v.min():.3f})   "
                  f"IQR=[{q1:+.3f}, {q3:+.3f}] (width {q3-q1:.3f})")


def compression_table(d: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("5/6. Compression by experimental logit(PSI) decile (equal-count bins)")
    print("=" * 100)
    for split in ("train", "test"):
        s = d[(d["split"] == split) & d["logit_psi"].notna()].copy()
        s["q"] = pd.qcut(s["logit_psi"], N_TABLE_BINS, duplicates="drop")
        g = s.groupby("q", observed=True).agg(
            n=("logit_psi", "size"),
            lp_med=("logit_psi", "median"),
            lp_iqr=("logit_psi", lambda v: v.quantile(.75) - v.quantile(.25)),
            pl_med=("prod_logit", "median"),
            pl_iqr=("prod_logit", lambda v: v.quantile(.75) - v.quantile(.25)),
        ).reset_index(drop=True)
        # local slope between consecutive decile medians
        g["d_logit_psi"] = g["lp_med"].diff()
        g["d_prod_logit"] = g["pl_med"].diff()
        g["local_slope"] = g["d_prod_logit"] / g["d_logit_psi"]
        print(f"\n[{split}]  (local_slope = Δ median prod_logit / Δ median logit(PSI))")
        print(g.round(4).to_string(index=False))


def exact_extremes(d: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("Exact PSI==0 and PSI==1 observations (excluded above; reported here)")
    print("=" * 100)
    for split in ("train", "test"):
        s = d[d["split"] == split]
        for lab, m in (("PSI==0", s["psi_is_exact_0"]), ("PSI==1", s["psi_is_exact_1"])):
            v = s.loc[m, "prod_logit"]
            if v.empty:
                continue
            print(f"  {split:5} {lab}: n={len(v):4d}  prod_logit median={v.median():+.4f}  "
                  f"IQR=[{v.quantile(.25):+.4f}, {v.quantile(.75):+.4f}]")


def make_plot(d: pd.DataFrame) -> None:
    fin = d[d["logit_psi"].notna()]
    xlim = (fin["logit_psi"].min(), fin["logit_psi"].max())
    ylim = (fin["prod_logit"].min(), fin["prod_logit"].max())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    for ax, split in zip(axes, ("train", "test")):
        s = fin[fin["split"] == split]
        x, y = s["logit_psi"].to_numpy(), s["prod_logit"].to_numpy()

        h = ax.hist2d(x, y, bins=60, range=[xlim, ylim], cmap="Blues", cmin=1, norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label="Count")

        # binned median curve (nonparametric) -- not a global regression
        q = pd.qcut(s["logit_psi"], N_SMOOTH_BINS, duplicates="drop")
        g = s.groupby(q, observed=True).agg(bx=("logit_psi", "median"), by=("prod_logit", "median"))
        ax.plot(g["bx"], g["by"], "-o", color="crimson", lw=1.8, ms=4, label="binned median")

        rho, _ = spearmanr(x, y)
        r, _ = pearsonr(x, y)
        ax.text(0.02, 0.98,
                f"n = {len(x):,}\nSpearman ρ = {rho:.3f}\nPearson r = {r:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("Experimental logit(PSI)   [PNAS metadata_PSI]")
        ax.set_ylabel("AlphaGenome prod_logit\nlogit(P(acceptor) × P(donor))")
        ax.set_title(f"{split}  (n={len(x):,})")
        ax.legend(fontsize=8.5, loc="lower right", framealpha=.9)
        ax.set_box_aspect(1)

    fig.suptitle(
        "PNAS three-exon cassette — experimental logit(PSI) vs AlphaGenome SPLICE_SITES prod_logit\n"
        "exact PSI = 0 / 1 excluded (infinite logit); binned median overlaid, no global fit"
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_PNG}")


def main() -> int:
    d = add_logit(load())
    correlations(d)
    compression_table(d)
    exact_extremes(d)
    make_plot(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
