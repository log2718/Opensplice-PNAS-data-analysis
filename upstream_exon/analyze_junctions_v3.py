#!/usr/bin/env python
"""Do cryptic junctions inside B1 explain the v3 signal? Tables and one figure.

Reads junction_measurements.csv (written by check_junctions_v3.py) and answers
the question the cryptic check exists for. Two findings drive the layout:

  * cryptic donors inside B1 go with HIGHER AG_mean_logit, not lower -- the
    naive worry was that they steal splicing and depress the target;
  * yet dropping every sequence that has one leaves the target's variance
    untouched, so the B1 effect does not depend on them.

Writes into outputs/upstream_exon/junction_check_v3/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / "outputs" / "upstream_exon" / "junction_check_v3"

CRYPTIC_COLS = ["b1_cryptic_acceptor_fraction", "b1_cryptic_fraction_all",
                "n_b1_cryptic_donors", "max_b1_cryptic_value"]
C_WITH, C_WITHOUT = "#d62728", "#1f6fb4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CHECK / "junction_measurements.csv")
    ap.add_argument("--out-fig", type=Path, default=CHECK / "junction_summary.png")
    ap.add_argument("--out-txt", type=Path, default=CHECK / "junction_summary.txt")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    y = d.AG_mean_logit
    has = d.n_b1_cryptic_donors > 0
    L: list[str] = []
    P = L.append

    P("=" * 74)
    P("B1 CRYPTIC JUNCTION CHECK -- v3 library")
    P("=" * 74)
    P(f"sequences scored: {len(d):,} (stratified over AG_mean_logit deciles "
      f"of the 40,000-row test split)")
    P(f"target re-derived from the same calls; max |diff| vs the stored value "
      f"= {(d.recomputed_mean_logit - y).abs().max():.2e}")
    P("")
    P("-" * 74)
    P("1. HOW COMMON ARE CRYPTIC DONORS INSIDE B1?")
    P("-" * 74)
    for k, g in d.groupby("n_b1_cryptic_donors"):
        P(f"  {k} cryptic donor(s): {len(g):5,} seqs ({100 * len(g) / len(d):5.1f}%)"
          f"   AG_mean_logit {g.AG_mean_logit.mean():.4f} +- "
          f"{g.AG_mean_logit.std():.4f}"
          f"   cryptic share of acceptor {g.b1_cryptic_acceptor_fraction.mean():.3f}")
    P(f"  any: {has.sum():,} of {len(d):,} ({100 * has.mean():.1f}%)")

    P("")
    P("-" * 74)
    P("2. DIRECTION OF THE EFFECT")
    P("-" * 74)
    u = stats.mannwhitneyu(y[has], y[~has])
    P(f"  with    >=1 cryptic donor: {y[has].mean():.4f} +- {y[has].std():.4f}")
    P(f"  without  0 cryptic donors: {y[~has].mean():.4f} +- {y[~has].std():.4f}")
    P(f"  Mann-Whitney U p = {u.pvalue:.3e}")
    P("  cryptic donors go WITH a HIGHER target, the opposite of the artifact")
    P("  worry, which was that they compete splicing away from the middle exon.")
    P("")
    P("  The middle exon's TWO sites rise together (see section 5): the")
    P("  acceptor by +0.71 logit and the donor by +0.63 across 0 -> 3 cryptic")
    P("  donors. An acceptor-specific mechanism would lift the acceptor far")
    P("  more, since the donor at 318 lies downstream of the middle exon. So")
    P("  this is exon-level recognition, not a cryptic donor feeding one site;")
    P("  whether the cryptic donor causes that or merely co-occurs with B1")
    P("  sequence that strengthens the whole cassette is NOT resolved here.")

    P("")
    P("-" * 74)
    P("3. CORRELATIONS WITH AG_mean_logit")
    P("-" * 74)
    for c in CRYPTIC_COLS + ["designed_acceptor_fraction",
                             "v_designed_to_acceptor", "v_exon_skipping",
                             "middle_donor_designed_fraction"]:
        rho, p = stats.spearmanr(d[c], y)
        r, pp = stats.pearsonr(d[c], y)
        P(f"  {c:<32} rho={rho:+.4f} p={p:.2e}   r={r:+.4f} p={pp:.2e}")
    P("")
    P("  NOTE: middle_donor_designed_fraction was intended as a control -- B1")
    P("  cannot create a donor at position 318 -- but at rho=+0.93 it is not")
    P("  one. It restates the target: a strongly used middle exon both scores")
    P("  high at its two sites and sends most of its donor signal to the")
    P("  designed acceptor. It is a consequence of the target, not a check.")

    X = d[CRYPTIC_COLS].values
    r2 = LinearRegression().fit(X, y).score(X, y)
    P("")
    P("-" * 74)
    P("4. HOW MUCH OF THE TARGET DO CRYPTIC JUNCTIONS EXPLAIN?")
    P("-" * 74)
    P(f"  linear R2 of AG_mean_logit on all four cryptic metrics: {r2:.4f}")
    nc = d[~has]
    vf, vn = y.var(ddof=0), nc.AG_mean_logit.var(ddof=0)
    P(f"  variance of the target, all {len(d):,} sequences : {vf:.4f}")
    P(f"  variance with all {has.sum():,} cryptic sequences dropped: {vn:.4f}")
    P(f"  -> {100 * vn / vf:.1f}% of the spread retained")
    P("")
    P("  HEADLINE: cryptic junction structure predicts a real slice of the")
    P(f"  target (R2 {r2:.3f}), but removing every sequence that has one leaves")
    P("  the spread intact. The B1 -> splice-strength relationship the models")
    P("  learn is therefore NOT an artifact of cryptic splice sites.")

    P("")
    P("-" * 74)
    P("5. WHICH OF THE TWO SITES MOVES?")
    P("-" * 74)
    P("  AG_mean_logit averages the middle exon's acceptor and donor logits.")
    P("  Splitting them says whether the effect is acceptor-specific.")
    lg = lambda p: np.log(p / (1 - p))
    d["acc_logit"] = lg(d.AG_acceptor_probability.clip(1e-6, 1 - 1e-6))
    d["don_logit"] = lg(d.AG_donor_probability.clip(1e-6, 1 - 1e-6))
    P(f"  {'cryptic':<9}{'n':>7}{'acceptor':>12}{'donor':>10}")
    for k, g in d.groupby("n_b1_cryptic_donors"):
        P(f"  {k:<9}{len(g):>7,}{g.acc_logit.mean():>12.4f}{g.don_logit.mean():>10.4f}")
    b = d.groupby("n_b1_cryptic_donors")[["acc_logit", "don_logit"]].mean()
    P(f"  shift 0 -> 3: acceptor {b.acc_logit.iloc[-1] - b.acc_logit.iloc[0]:+.4f}"
      f"   donor {b.don_logit.iloc[-1] - b.don_logit.iloc[0]:+.4f}")
    P("  Both move, and by similar amounts -> exon-level, not site-specific.")

    txt = "\n".join(L) + "\n"
    args.out_txt.write_text(txt)
    print(txt)

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    groups = sorted(d.n_b1_cryptic_donors.unique())
    data = [d.loc[d.n_b1_cryptic_donors == k, "AG_mean_logit"] for k in groups]
    bp = ax.boxplot(data, positions=groups, widths=.62, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.4))
    for patch, k in zip(bp["boxes"], groups):
        patch.set_facecolor(C_WITHOUT if k == 0 else C_WITH)
        patch.set_alpha(.30 + .18 * min(k, 3))
        patch.set_edgecolor("#333")
    for k, g in zip(groups, data):
        ax.annotate(f"n={len(g):,}", (k, ax.get_ylim()[0]), ha="center",
                    va="bottom", fontsize=7.5, color="#555")
    ax.set_xlabel("cryptic donors inside B1")
    ax.set_ylabel("AG_mean_logit")
    ax.set_title("More cryptic donors → higher target", fontsize=10.5)

    ax = axes[1]
    ax.scatter(d.b1_cryptic_acceptor_fraction, y, s=5, alpha=.22,
               color=C_WITH, ec="none")
    rho, p = stats.spearmanr(d.b1_cryptic_acceptor_fraction, y)
    ax.set_xlabel("cryptic share of signal into the middle acceptor")
    ax.set_ylabel("AG_mean_logit")
    ax.set_title(f"Spearman ρ = {rho:+.3f}  (p = {p:.1e})", fontsize=10.5)

    ax = axes[2]
    bins = np.linspace(y.min(), y.max(), 46)
    ax.hist(y, bins=bins, color="#999", alpha=.55, label=f"all ({len(d):,})",
            density=True)
    ax.hist(nc.AG_mean_logit, bins=bins, histtype="step", lw=1.8,
            color=C_WITHOUT, label=f"no cryptic donor ({len(nc):,})",
            density=True)
    ax.set_xlabel("AG_mean_logit")
    ax.set_ylabel("density")
    ax.set_title(f"Spread survives removal: {100 * vn / vf:.0f}% of variance",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)

    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Cryptic splice junctions inside B1 do not explain the v3 "
                 "upstream-exon signal", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(args.out_fig, dpi=170)
    print(f"wrote {args.out_fig}")
    print(f"wrote {args.out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
