#!/usr/bin/env python
"""2D histogram of predicted vs measured AlphaGenome mean-logit.

A scatter of 40,000 points is a solid blob and hides where the mass is, so
density is binned and shown on a log colour scale.

Two things about this target make a plain scatter actively misleading:

  * y_true is QUANTIZED. AlphaGenome returns probabilities on a ~1/256 grid,
    so the measured axis lands on a discrete ladder (~0.04 logit per step near
    p=0.95) and the cloud is banded. That banding is the assay, not the model.
  * Regression to the mean is expected under MSE and shows up as a cloud
    flatter than the identity line. The binned conditional mean of y_pred
    given y_true is overlaid so that tilt can be read off directly rather
    than eyeballed against y=x.

    python upstream_exon/plot_pred_vs_true.py \
        --run-dir outputs/upstream_exon/models/full_v3 --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_CAL_BINS = 28


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bins", type=int, default=160)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--clip", type=float, default=0.999,
                    help="central quantile span to plot (outliers still "
                         "counted in the stats, and reported)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    src = args.run_dir / "predictions.csv"
    if not src.exists():
        print(f"FATAL: {src} not found", file=sys.stderr)
        return 1
    df = pd.read_csv(src, usecols=["split", "y_true", "y_pred"])
    d = df[df.split == args.split]
    if d.empty:
        print(f"FATAL: no rows in split {args.split!r}", file=sys.stderr)
        return 1
    y, p = d.y_true.to_numpy(), d.y_pred.to_numpy()
    n = len(y)

    resid = p - y
    mse = float((resid ** 2).mean())
    r2 = 1.0 - mse / float(y.var())
    r = float(np.corrcoef(y, p)[0, 1])
    print(f"{args.split}: n={n}  R2={r2:+.4f}  r={r:+.4f}  "
          f"RMSE={mse ** .5:.4f}  bias={resid.mean():+.4f}")

    # The measured axis is a quantization ladder; say how coarse it is.
    # The target averages the acceptor and donor logits, so TWO ~1/256
    # ladders interleave and the minimum gap is much finer than the spacing
    # you can see. The median gap is the one that describes the striping.
    uniq = np.unique(y)
    if len(uniq) > 1:
        gaps = np.diff(uniq)
        print(f"measured axis: {len(uniq)} distinct values, "
              f"median gap {np.median(gaps):.4f} logit "
              f"(min {gaps.min():.5f}) -- the two averaged ladders interleave, "
              f"so the visible striping is coarser than the min")

    lo = min(np.quantile(y, 1 - args.clip), np.quantile(p, 1 - args.clip))
    hi = max(np.quantile(y, args.clip), np.quantile(p, args.clip))
    pad = 0.04 * (hi - lo)
    lim = (lo - pad, hi + pad)
    shown = int(((y >= lim[0]) & (y <= lim[1])
                 & (p >= lim[0]) & (p <= lim[1])).sum())
    if shown < n:
        print(f"note: {n - shown} of {n} points ({100 * (n - shown) / n:.2f}%) "
              f"fall outside the plotted range; all are included in the stats")

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    h = ax.hist2d(y, p, bins=args.bins, range=[lim, lim],
                  norm=LogNorm(), cmap="viridis")
    cb = fig.colorbar(h[3], ax=ax, pad=.02)
    cb.set_label("sequences per bin (log scale)", fontsize=9)

    ax.plot(lim, lim, color="#d62728", lw=1.1, ls="--", label="identity")

    # conditional mean of prediction given measurement
    edges = np.linspace(*lim, N_CAL_BINS + 1)
    idx = np.digitize(y, edges) - 1
    ok = (idx >= 0) & (idx < N_CAL_BINS)
    cx, cy = [], []
    for b in range(N_CAL_BINS):
        m = ok & (idx == b)
        if m.sum() >= 25:
            cx.append(.5 * (edges[b] + edges[b + 1]))
            cy.append(p[m].mean())
    ax.plot(cx, cy, color="#ffffff", lw=2.6, zorder=4)
    ax.plot(cx, cy, color="#111111", lw=1.4, zorder=5,
            label="binned mean prediction")

    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("measured AlphaGenome mean logit", fontsize=10)
    ax.set_ylabel("predicted AlphaGenome mean logit", fontsize=10)
    ax.set_title(f"{args.run_dir.name} -- {args.split} "
                 f"(n={n:,})", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=.9)
    ax.text(.985, .035,
            f"R² = {r2:+.4f}\nPearson r = {r:+.4f}\n"
            f"RMSE = {mse ** .5:.4f} logit\nbias = {resid.mean():+.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(fc="white", ec="#bbb", alpha=.9, boxstyle="round,pad=.4"))
    ax.grid(alpha=.18, lw=.5)

    dest = args.out or args.run_dir / f"pred_vs_true_{args.split}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
