#!/usr/bin/env python
"""What the v3 junction result means, in three panels.

(a) what a cryptic donor IS: a GT inside the randomized block that AlphaGenome
    splices to the middle acceptor, alongside the designed donor;
(b) what it does: BOTH of the middle exon's sites strengthen, by similar
    amounts -- so the effect is exon-level, not a cryptic donor feeding one
    site, and causation is not established;
(c) what it does not do: removing every sequence that has one leaves the
    target's spread untouched, so the B1 effect is not a cryptic artifact.

    python upstream_exon/plot_junction_mechanism.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / "outputs" / "upstream_exon" / "junction_check_v3"

C_DES, C_CRY, C_NONE = "#1f6fb4", "#d62728", "#6a9fd4"


def panel_schematic(ax) -> None:
    """B1 with and without a cryptic donor, drawn to the same scale."""
    ax.set_xlim(-8, 270)
    ax.set_ylim(-0.3, 2.35)
    ax.axis("off")

    for row, (label, cryptic) in enumerate([("no cryptic donor  (68% of sequences)", None),
                                            ("cryptic donor in B1  (32%)", 99)]):
        y = 1.35 - 1.25 * row
        ax.add_patch(plt.Rectangle((0, y), 113, .17, color="#e8a33d"))
        ax.add_patch(plt.Rectangle((113, y), 3, .17, color="#444"))
        ax.add_patch(plt.Rectangle((243, y), 27, .17, color="#444"))
        ax.plot([116, 243], [y + .085, y + .085], color="#444", lw=.9, zorder=0)
        ax.annotate("B1 (113 nt, randomized)", (56, y - .04), ha="center",
                    va="top", fontsize=7.5, color="#7a4e00")
        ax.annotate("middle\nexon", (256, y - .04), ha="center", va="top",
                    fontsize=7.5, color="#444")
        ax.annotate(label, (0, y + .46), ha="left", va="bottom", fontsize=9,
                    color="#222", weight="bold")

        def arc(a, b, colour, lw, h):
            t = np.linspace(0, np.pi, 160)
            ax.plot((a + b) / 2 + (b - a) / 2 * np.cos(t),
                    y + .17 + h * np.sin(t), color=colour, lw=lw,
                    solid_capstyle="round")

        arc(116, 243, C_DES, 2.6, .22)
        ax.annotate("designed donor", (116, y + .20), ha="center", va="bottom",
                    fontsize=7, color=C_DES)
        if cryptic:
            arc(cryptic + 1, 243, C_CRY, 2.0, .34)
            ax.plot([cryptic, cryptic], [y, y + .17], color=C_CRY, lw=1.4)
            # Above the block, not below: the B1 caption occupies that space.
            ax.annotate("cryptic GT", (cryptic - 6, y + .36), ha="right",
                        va="center", fontsize=7.5, color=C_CRY, weight="bold")
    ax.set_title("(a)  a cryptic donor is a GT inside B1 that AlphaGenome\n"
                 "splices to the same acceptor", fontsize=10, loc="left")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CHECK / "junction_measurements.csv")
    ap.add_argument("--out", type=Path, default=CHECK / "junction_mechanism.png")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    lg = lambda p: np.log(p / (1 - p))
    d["acc_logit"] = lg(d.AG_acceptor_probability.clip(1e-6, 1 - 1e-6))
    d["don_logit"] = lg(d.AG_donor_probability.clip(1e-6, 1 - 1e-6))
    has = d.n_b1_cryptic_donors > 0

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=.28)
    panel_schematic(fig.add_subplot(gs[0]))

    # (b) both sites move together --------------------------------------
    ax = fig.add_subplot(gs[1])
    g = d.groupby("n_b1_cryptic_donors")
    ks = sorted(d.n_b1_cryptic_donors.unique())
    w = .36
    for off, col, lab, c in ((-w / 2, "acc_logit", "acceptor (243)", "#2a7f5f"),
                             (+w / 2, "don_logit", "donor (318)", "#9a6fb0")):
        ax.bar(np.array(ks) + off, [g[col].mean()[k] for k in ks], w,
               yerr=[g[col].sem()[k] for k in ks], color=c, label=lab,
               capsize=2.5)
    b = g[["acc_logit", "don_logit"]].mean()
    ax.annotate(f"0→3 shift\nacceptor {b.acc_logit.iloc[-1] - b.acc_logit.iloc[0]:+.2f}\n"
                f"donor      {b.don_logit.iloc[-1] - b.don_logit.iloc[0]:+.2f}",
                (.03, .97), xycoords="axes fraction", va="top", fontsize=8,
                bbox=dict(fc="white", ec="#ccc"))
    ax.set_xticks(ks)
    ax.set_xlabel("cryptic donors inside B1")
    ax.set_ylabel("splice-site logit")
    ax.set_ylim(2.8, 4.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("(b)  both of the exon's sites strengthen,\nby similar "
                 "amounts → exon-level, not site-specific", fontsize=10,
                 loc="left")

    # (c) the spread survives removal ------------------------------------
    ax = fig.add_subplot(gs[2])
    y = d.AG_mean_logit
    nc = d.loc[~has, "AG_mean_logit"]
    bins = np.linspace(y.min(), y.max(), 42)
    ax.hist(y, bins=bins, color="#aaa", alpha=.6, density=True,
            label=f"all ({len(d):,})")
    ax.hist(nc, bins=bins, histtype="step", lw=1.9, color=C_NONE, density=True,
            label=f"no cryptic donor ({len(nc):,})")
    ax.annotate(f"variance retained\n{100 * nc.var(ddof=0) / y.var(ddof=0):.0f}%",
                (.97, .97), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, bbox=dict(fc="white", ec="#ccc"))
    ax.set_xlabel("AG_mean_logit")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title("(c)  drop every cryptic sequence and the\nsignal is "
                 "unchanged → not an artifact", fontsize=10, loc="left")

    for a in fig.axes[1:]:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
