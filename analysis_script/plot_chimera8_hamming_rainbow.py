"""
Hamming-1 neighbourhood figure, recoloured by which block changes.

Same data, same 256 contexts per arm, same 1,024 Hamming-1 edges per arm, same
x values and reference lines as ``plot_chimera8_hamming.py`` -- that module is
imported and its data path and edge construction reused unchanged, so the two
figures cannot drift apart.

Two changes only:

1. Edges are coloured by the block that differs across them. The context space
   is an 8-dimensional hypercube, so every Hamming-1 edge flips exactly one
   block; B1..B8 get a red -> violet ramp.
2. The vertical jitter is widened substantially (see JITTER_LO/HI below), so
   nodes separate instead of clumping. y remains deterministic and meaningless.

Writes a NEW png and never touches the existing one.

Usage::

    python analysis_script/plot_chimera8_hamming_rainbow.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import plot_chimera8_hamming as H  # noqa: E402

#: red -> violet, one per block, in B1..B8 order
BLOCK_COLOR = {
    "B1": "#e11d1d",   # red
    "B2": "#f57c00",   # orange
    "B3": "#e8c400",   # yellow
    "B4": "#2fa84f",   # green
    "B5": "#00bcd4",   # cyan
    "B6": "#1e5fd0",   # blue
    "B7": "#4b2ea8",   # indigo
    "B8": "#9b30c9",   # violet
}

#: the previous figure used uniform(0.15, 0.85) -- a 0.70 band.
#: 0.03..0.97 is a 0.94 band, 1.34x wider, with the panel ylim opened to match.
JITTER_LO, JITTER_HI = 0.03, 0.97
JITTER_SEED = 0

DOT_SIZE = 34          # was 46 -- smaller so dense regions stay readable
EDGE_ALPHA = 0.42      # was 0.14 -- raised because colour now carries meaning
EDGE_LW = 0.62


def plot_rainbow(med, edges, ep_med, path: Path) -> dict:
    rng = np.random.default_rng(JITTER_SEED)
    pos: dict[str, dict[str, tuple[float, float]]] = {}
    for arm in H.ARMS:
        labels = sorted(med[arm])
        yy = rng.uniform(JITTER_LO, JITTER_HI, len(labels))
        pos[arm] = {lab: (med[arm][lab], float(y))
                    for lab, y in zip(labels, yy)}

    allv = [v for arm in H.ARMS for v in med[arm].values()]
    lo, hi = min(allv + [ep_med["PNAS"]]), max(allv + [ep_med["PNAS"]])
    pad = (hi - lo) * 0.04
    xlim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, 1, figsize=(13, 12.0), sharex=True)
    fig.subplots_adjust(top=0.895, bottom=0.115, left=0.055, right=0.985,
                        hspace=0.17)

    for ax, arm in zip(axes, H.ARMS):
        e = edges[edges.comparison == arm]
        # draw block by block so each colour group is contiguous in z
        for b in H.BLOCKS:
            eb = e[e.block_changed == b]
            if eb.empty:
                continue
            segs = [[pos[arm][r.label_P_analysis], pos[arm][r.label_F_analysis]]
                    for r in eb.itertuples()]
            ax.add_collection(LineCollection(
                segs, colors=BLOCK_COLOR[b], linewidths=EDGE_LW,
                alpha=EDGE_ALPHA, zorder=1))

        ax.axvline(ep_med[arm], color="black", lw=1.4, ls="--", zorder=2)
        ax.axvline(ep_med["PNAS"], color=H.MUTED, lw=1.2, ls=":", zorder=2)

        xs = [pos[arm][l][0] for l in sorted(med[arm])]
        ys = [pos[arm][l][1] for l in sorted(med[arm])]
        ax.scatter(xs, ys, s=DOT_SIZE, color=H.ARM_COLOR[arm],
                   edgecolor="black", linewidth=.5, zorder=3)

        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_ylabel("contexts", fontsize=9)
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=H.MUTED)
        ax.set_title(
            f"{arm}   ·   {len(med[arm])} contexts, {len(e):,} Hamming-1 edges"
            f"   ·   dashed = 10k {arm} endpoint median ({ep_med[arm]:.2f})"
            f", dotted = 10k PNAS endpoint median ({ep_med['PNAS']:.2f})",
            fontsize=9.5, color=H.MUTED, loc="left", pad=6)

    axes[-1].set_xlabel("context median AlphaGenome mean-logit", fontsize=11)

    fig.text(0.5, 0.972, "Hamming-distance-1 neighbourhood of the 8-block "
             "context space", ha="center", fontsize=14, fontweight="bold",
             color=H.INK)
    fig.text(0.5, 0.952,
             "every edge joins two contexts whose P/F labels differ at exactly "
             "one block, and is coloured by THAT block   ·   y is deterministic "
             "jitter with no meaning, so only HORIZONTAL separation is "
             "interpretable",
             ha="center", fontsize=9, color=H.MUTED)

    handles = [Line2D([0], [0], color=BLOCK_COLOR[b], lw=2.6,
                      label=H.BLOCK_SHORT[b]) for b in H.BLOCKS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, handlelength=2.0, columnspacing=1.8,
               bbox_to_anchor=(0.5, 0.004),
               title="edge colour = block that changes across the edge",
               title_fontsize=9)

    fig.savefig(path, dpi=190)
    plt.close(fig)
    return pos


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    H.P_HAM.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(H.RESULTS_CSV,
                    usecols=["comparison", "context_label_8", "sequence_id",
                             H.Y],
                    low_memory=False)
    med = {arm: d[d.comparison == arm].groupby("context_label_8")[H.Y]
                 .median().to_dict() for arm in H.ARMS}
    ep = pd.read_csv(H.OLD_ENDPOINT_CSV)
    ep_med = ep.groupby("context")[H.Y].median().to_dict()

    edges = H.build_edge_table(med)

    out = (H.P_HAM
           / "chimera8_hamming1_jitter_rainbow_widejitter_e6_e7.png")
    if out.exists():
        print(f"NOTE: overwriting {out.name} (regenerated figure)")
    plot_rainbow(med, edges, ep_med, out)

    for arm in H.ARMS:
        n = len(edges[edges.comparison == arm])
        print(f"  {arm}: {len(med[arm])} contexts, {n:,} Hamming-1 edges")
    print("\n  edges per block (per arm):")
    print(edges.groupby(["block_changed", "comparison"]).size()
          .unstack().to_string())
    print(f"\n  jitter: uniform({JITTER_LO}, {JITTER_HI}) "
          f"seed={JITTER_SEED}  (previous: uniform(0.15, 0.85))")
    print(f"  figure -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
