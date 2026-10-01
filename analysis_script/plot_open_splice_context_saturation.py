"""
Plots for the OpenSplice/FAS-context AlphaGenome saturation experiment.

For each (exon, motif family) produces three views of the same relationship --
x = PNAS pre-tuner, y = AlphaGenome prod_logit -- differing only in which copy
levels are included:

1. ``all``   -- 2x,4x,6x,8x,10x,12x,14x (whichever are feasible)
2. ``low``   -- 2x,4x,6x
3. ``high``  -- 6x,8x,10x,12x,14x

No regression line is drawn: these figures feed the saturation detector, and a
global fit is exactly the wrong primary interpretation for a curve that may
plateau. A binned-median curve is overlaid instead so the shape is visible.
Repo-standard blue hist2d/LogNorm/"Count" styling is used when a panel has
enough points for a density view; sparse panels fall back to points coloured
by copy level.

Output
------
``outputs/plots/open_splice_context_saturation/<exon>_<family>_<view>.png``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent

CSV = ROOT / "outputs" / "open_splice_context_saturation" / "open_splice_context_saturation_variants.csv"
PLOT_DIR = ROOT / "outputs" / "plots" / "open_splice_context_saturation"

X, Y = "PNAS_pretuner", "AG_prod_logit"
VIEWS = {"all": [2, 4, 6, 8, 10, 12, 14], "low": [2, 4, 6], "high": [6, 8, 10, 12, 14]}
DENSITY_MIN_N = 400      # below this, a hist2d is mostly empty cells
N_MEDIAN_BINS = 12


def _panel(ax, sub: pd.DataFrame, wt: pd.DataFrame, fig, title: str) -> None:
    x, y = sub[X].to_numpy(), sub[Y].to_numpy()
    if len(x) >= DENSITY_MIN_N:
        h = ax.hist2d(x, y, bins=40, cmap="Blues", cmin=1, norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label="Count")
    else:
        sc = ax.scatter(x, y, c=sub["copy_number"], cmap="viridis", s=14,
                        alpha=.75, edgecolor="none")
        fig.colorbar(sc, ax=ax, label="Copy number")

    if len(x) >= 24:
        nb = min(N_MEDIAN_BINS, max(4, len(x) // 20))
        q = pd.qcut(sub[X], nb, duplicates="drop")
        g = sub.groupby(q, observed=True).agg(bx=(X, "median"), by=(Y, "median"))
        ax.plot(g["bx"], g["by"], "-o", color="crimson", lw=1.6, ms=4,
                label="binned median")

    if not wt.empty:
        ax.scatter(wt[X], wt[Y], marker="*", s=190, color="black",
                   zorder=5, label="WT")

    ax.set_xlabel("PNAS pre-tuner score")
    ax.set_ylabel("AlphaGenome prod_logit\nlogit(P(acceptor) × P(donor))")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best", framealpha=.9)
    ax.grid(alpha=.25, linewidth=.7)
    ax.set_axisbelow(True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plot OpenSplice/FAS context-saturation results.")
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--views", default="all,low,high")
    args = ap.parse_args(argv)

    if not args.csv.exists():
        print(f"[missing] {args.csv} -- run the scoring job first.")
        return 1

    df = pd.read_csv(args.csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[X, Y])
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    n = 0
    for exon_id, ex in df.groupby("exon_id"):
        wt = ex[ex["motif_family"] == "WT"]
        for family, fam in ex[ex["motif_family"] != "WT"].groupby("motif_family"):
            for view in views:
                sub = fam[fam["copy_number"].isin(VIEWS[view])]
                if sub.empty:
                    continue
                levels = sorted(sub["copy_number"].unique())
                fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
                _panel(ax, sub, wt, fig,
                       f"{exon_id} — {family} ({view}: "
                       f"{','.join(f'{c}x' for c in levels)})  n={len(sub):,}")
                out = PLOT_DIR / f"{exon_id}_{family}_{view}.png"
                fig.savefig(out, dpi=180, bbox_inches="tight")
                plt.close(fig)
                n += 1
    print(f"wrote {n} figures -> {PLOT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
