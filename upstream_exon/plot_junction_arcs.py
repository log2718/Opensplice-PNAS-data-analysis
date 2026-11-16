#!/usr/bin/env python
"""Draw AlphaGenome's predicted splice junctions as an arc diagram.

Reads the CSV written by probe_junction_output.py (core coordinates, one row
per junction) and draws the minigene with every predicted junction as an arc.
Junctions are coloured by what they mean for the experiment:

  designed     the two intended junctions, 116->243 and 319->1168
  B1 cryptic   a junction whose DONOR falls inside B1 [0, 113) -- the thing
               the cryptic-site check exists to detect
  other        everything else (exon skipping, cryptic acceptors outside B1)

Arc width is log-scaled: the values span roughly 60x, so linear widths would
render the weak junctions invisible.

    python upstream_exon/plot_junction_arcs.py
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
CHECK = ROOT / "outputs" / "upstream_exon" / "cryptic_check"

B1 = (0, 113)
#: exon spans as [start, end); the upstream exon ends at the donor + 1
EXONS = [(0, 116, "upstream exon"), (243, 319, "middle exon"),
         (1168, 1309, "downstream exon")]
SITES = {115: ("upstream\ndonor", "donor"), 243: ("middle\nacceptor", "acceptor"),
         318: ("middle\ndonor", "donor"), 1168: ("downstream\nacceptor", "acceptor")}
DESIGNED = {(116, 243), (319, 1168)}

C_DESIGNED = "#1f6fb4"
C_CRYPTIC = "#d62728"
C_OTHER = "#999999"


def classify(row) -> tuple[str, str]:
    key = (int(row.start_core), int(row.end_core))
    if key in DESIGNED:
        return "designed", C_DESIGNED
    if B1[0] <= row.start_core < B1[1]:
        return "B1 cryptic", C_CRYPTIC
    return "other", C_OTHER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=CHECK / "junction_probe_junctions.csv")
    ap.add_argument("--out", type=Path,
                    default=CHECK / "junction_arcs.png")
    ap.add_argument("--core-len", type=int, default=1309)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    vcol = "value_t0" if "value_t0" in df.columns else "value"
    df = df.sort_values(vcol).reset_index(drop=True)

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(13, 7.2), gridspec_kw={"height_ratios": [3.1, 1],
                                              "hspace": .45})

    # ---- arc panel --------------------------------------------------------
    grand = df[vcol].sum()
    vmin, vmax = df[vcol].min(), df[vcol].max()
    lw = lambda v: 0.8 + 5.2 * (np.log10(v) - np.log10(vmin)) / (
        np.log10(vmax) - np.log10(vmin))

    # Labels are placed at staggered points ALONG each arc rather than all at
    # the apex, which is where they collided when several arcs share an endpoint.
    for i, r in enumerate(df.itertuples()):
        label, colour = classify(r)
        a, b = int(r.start_core), int(r.end_core)
        mid, half = (a + b) / 2, (b - a) / 2
        t = np.linspace(0, np.pi, 200)
        # Height scales with span so long arcs do not flatten onto short ones.
        h = 0.22 + 0.78 * (half / (args.core_len / 2)) ** 0.55
        ax.plot(mid + half * np.cos(t), h * np.sin(t), color=colour,
                lw=lw(r._asdict()[vcol]), alpha=.85, solid_capstyle="round",
                zorder=3 if label == "designed" else 2)
        tl = np.pi * (0.50 if label == "designed" else (0.72, 0.28, 0.62)[i % 3])
        dy = 0.055 if label == "designed" else 0.012
        # The raw value is a predicted RNA-seq junction abundance, not a
        # probability, so the share of this construct's total is what reads.
        ax.annotate(f"{100 * r._asdict()[vcol] / grand:.1f}%",
                    (mid + half * np.cos(tl), h * np.sin(tl) + dy),
                    ha="center", va="bottom", fontsize=7, color=colour,
                    zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=.72, pad=.6))

    for s, e, name in EXONS:
        ax.add_patch(plt.Rectangle((s, -0.16), e - s, 0.12, color="#444",
                                   zorder=4))
    # B1 is the first 113 nt OF the upstream exon, so it overlays that box
    # rather than sitting beside it; one combined label avoids the collision.
    ax.add_patch(plt.Rectangle((B1[0], -0.16), B1[1] - B1[0], 0.12,
                               color="#e8a33d", zorder=5))
    ax.annotate("upstream exon\nB1 = nt 0–112, randomized", (58, -0.185),
                ha="center", va="top", fontsize=8, color="#7a4e00")
    ax.annotate("middle exon", (281, -0.185), ha="center", va="top",
                fontsize=8, color="#444")
    ax.annotate("downstream exon", (1238, -0.185), ha="center", va="top",
                fontsize=8, color="#444")
    ax.plot([0, args.core_len], [-0.10, -0.10], color="#444", lw=1, zorder=1)

    for pos, (name, kind) in SITES.items():
        ax.plot([pos, pos], [-0.16, -0.33], color="#222", lw=.9, zorder=4)
        ax.annotate(f"{name}\n{pos}", (pos, -0.345), ha="center", va="top",
                    fontsize=7.5, color="#222")
    # 897 carries no annotated site but receives two junctions; the pilot's
    # per-base check flagged the same position as the top off-target acceptor.
    ax.plot([897, 897], [-0.10, -0.33], color=C_OTHER, lw=.9, ls=":", zorder=4)
    ax.annotate("cryptic\nacceptor\n897", (897, -0.345), ha="center", va="top",
                fontsize=7.5, color="#777")

    ax.set_xlim(-30, args.core_len + 30)
    ax.set_ylim(-0.60, 1.18)
    ax.set_yticks([])
    ax.set_xlabel("position in the 1,309-nt construct core")
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_title("AlphaGenome predicted splice junctions — native PNAS B1\n"
                 "arc width ∝ log(junction value); CL:0002518, total RNA-seq",
                 fontsize=11)
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=l) for l, c in
               (("designed junction", C_DESIGNED),
                ("cryptic donor inside B1", C_CRYPTIC),
                ("other", C_OTHER))]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)

    # ---- composition panel ------------------------------------------------
    df = df.assign(kind=[classify(r)[0] for r in df.itertuples()])
    acc = df[df.end_core == 243]

    # Two denominators, each summing to 100%. Keeping them on separate rows
    # stops the two percentages being read as parts of one total.
    rows = [
        (1, df, "every junction in the construct"),
        (0, acc, "junctions arriving at the middle acceptor (243)"),
    ]
    for y, sub, caption in rows:
        total = sub[vcol].sum()
        left = 0.0
        for kind, colour in (("designed", C_DESIGNED), ("B1 cryptic", C_CRYPTIC),
                             ("other", C_OTHER)):
            v = sub.loc[sub.kind == kind, vcol].sum()
            if v == 0:
                continue
            frac = v / total
            axb.barh(y, frac, left=left, color=colour, height=.52)
            axb.annotate(f"{kind} {frac:.1%}" if frac > .09 else f"{frac:.1%}",
                         (left + frac / 2, y), ha="center", va="center",
                         fontsize=8.5,
                         color="white" if frac > .09 else colour)
            left += frac
        axb.annotate(caption, (0, y + .40), ha="left", va="bottom",
                     fontsize=8, color="#444")
    axb.set_xlim(0, 1)
    axb.set_ylim(-.55, 1.75)
    axb.set_yticks([])
    axb.set_xlabel("share of predicted junction signal (each row sums to 100%)")
    for sp in ("left", "right", "top"):
        axb.spines[sp].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"\njunctions: {len(df)}   total value {df[vcol].sum():.6f}")
    for kind in ("designed", "B1 cryptic", "other"):
        sub = df[[classify(r)[0] == kind for r in df.itertuples()]]
        print(f"  {kind:<12} n={len(sub):<3} sum={sub[vcol].sum():.6f}  "
              f"({sub[vcol].sum() / df[vcol].sum():.1%} of all)")
    print(f"\nmiddle acceptor (243): designed keeps "
          f"{acc.loc[acc.kind == 'designed', vcol].sum() / total:.1%}, "
          f"B1 cryptic takes "
          f"{acc.loc[acc.kind == 'B1 cryptic', vcol].sum() / total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
