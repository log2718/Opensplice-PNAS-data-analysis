#!/usr/bin/env python
"""Information-content sequence logos for every conv filter, via Logomaker.

Method (Schneider & Stephens 1990, doi:10.1093/nar/18.20.6097):

  1. Score all 4^k hexamers against the filter: the activation of hexamer h is
     sum_p W[base_h[p], p] + bias.
  2. Keep only hexamers activating above HALF THE MAXIMUM activation. The kept
     count is reported as n. This is what turns a weight matrix into a set of
     sequences the filter actually responds to -- without it the logo shows
     the weights, not the preference.
  3. Build a position frequency matrix from those n hexamers.
  4. Information content at position i is R_i = 2 - H_i, with
     H_i = -sum_b p(b,i) log2 p(b,i)   (0 log 0 taken as 0).
  5. Letter height = p(b,i) * R_i, which is what Logomaker draws.

Note what n means. The filters are dense, not sparse motif matchers, so a
broad filter can keep a large fraction of all 4,096 hexamers; a logo built
from n near 4,096 is close to the uniform background and carries little
information by construction. Read R_i (the y axis) before reading the letters.

    python upstream_exon/plot_filter_logos.py \
        --run-dir outputs/upstream_exon/models/full_v3
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_upstream import UpstreamExonModel  # noqa: E402

BASES = "ACGT"
BASE_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
MAX_BITS = 2.0


def load_model(run_dir: Path, ckpt: str = "best.pt"):
    state = torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("config", {}) or {}
    sd = state["model_state_dict"]
    model = UpstreamExonModel(
        input_length=sd["position_bias_incl"].shape[1]
        + sd["conv_incl.weight"].shape[2] - 1,
        num_seq_filters=sd["conv_incl.weight"].shape[0],
        seq_kernel_size=sd["conv_incl.weight"].shape[2],
        use_structure=bool(cfg.get("use_structure", False)),
        head=cfg.get("head", "linear"))
    model.load_state_dict(sd)
    model.eval()
    return model, state


def all_kmers(k: int) -> np.ndarray:
    """(4**k, k) array of base indices, every k-mer exactly once."""
    return np.array(list(itertools.product(range(4), repeat=k)), dtype=np.int64)


def ic_matrix(weight: np.ndarray, bias: float, kmers: np.ndarray,
              use_bias: bool = True) -> tuple[pd.DataFrame, int, dict]:
    """Information-content letter heights for one filter.

    Returns ``(df, n_kept, info)`` where df is (k, 4) in Logomaker's layout.
    """
    k = weight.shape[1]
    act = weight[kmers, np.arange(k)[None, :]].sum(axis=1)
    if use_bias:
        act = act + bias

    hi = float(act.max())
    degenerate = hi <= 0
    if degenerate:
        # "half the maximum" is meaningless when the maximum is negative --
        # the threshold would sit ABOVE the max and keep nothing. Fall back to
        # the midpoint of the activation range and say so, rather than
        # silently emitting an empty logo.
        thr = float(act.min()) + 0.5 * (hi - float(act.min()))
    else:
        thr = 0.5 * hi
    keep = kmers[act > thr]
    n = len(keep)
    if n == 0:                       # cannot happen for the midpoint rule
        return (pd.DataFrame(0.0, index=range(k), columns=list(BASES)),
                0, {"degenerate": degenerate, "max_act": hi, "thr": thr,
                    "mean_bits": 0.0})

    # position frequency matrix
    p = np.zeros((k, 4))
    for b in range(4):
        p[:, b] = (keep == b).sum(axis=0)
    p /= n

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p, where=p > 0), 0.0)
    H = -(p * logp).sum(axis=1)
    R = MAX_BITS - H                                   # bits at each position
    df = pd.DataFrame(p * R[:, None], index=range(k), columns=list(BASES))
    return df, n, {"degenerate": degenerate, "max_act": hi, "thr": thr,
                   "mean_bits": float(R.mean())}


def plot_bank(model, arm: str, kmers: np.ndarray, dest: Path,
              run_name: str, use_bias: bool, ncol: int = 5) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logomaker

    conv = model.conv_incl if arm == "incl" else model.conv_skip
    W = conv.weight.detach().numpy()               # (F, 4, k)
    B = conv.bias.detach().numpy()
    F, _, k = W.shape

    mats, rows = [], []
    for f in range(F):
        df, n, info = ic_matrix(W[f], float(B[f]), kmers, use_bias)
        mats.append((df, n, info))
        rows.append({"arm": arm, "filter": f, "n_hexamers": n,
                     "mean_bits": info["mean_bits"],
                     "max_activation": info["max_act"],
                     "threshold": info["thr"],
                     "degenerate_threshold": info["degenerate"]})

    nrow = int(np.ceil(F / ncol))
    # One shared y limit so panels are comparable; 2 bits is the ceiling.
    ymax = max(float(df.sum(axis=1).max()) for df, _, _ in mats)
    ymax = min(MAX_BITS, max(0.05, ymax * 1.15))

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 1.95 * nrow),
                             squeeze=False)
    for f, (df, n, info) in enumerate(mats):
        ax = axes[f // ncol][f % ncol]
        logomaker.Logo(df, ax=ax, color_scheme=BASE_COLORS,
                       show_spines=False)
        ax.set_ylim(0, ymax)
        ax.set_xticks(range(k))
        ax.set_xticklabels(range(1, k + 1), fontsize=7)
        ax.tick_params(labelsize=7)
        flag = " *" if info["degenerate"] else ""
        ax.set_title(f"filter {f}  (n={n}){flag}", fontsize=9)
        if f % ncol == 0:
            ax.set_ylabel("bits", fontsize=8)
    for j in range(F, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    sub = ("activation includes the filter bias" if use_bias
           else "activation excludes the filter bias")
    fig.suptitle(
        f"{run_name} -- {arm} filters: information-content logos\n"
        f"hexamers above half-maximum activation; {sub}; "
        f"R = 2 - H (Schneider & Stephens 1990)",
        fontsize=11, y=.998)
    fig.tight_layout(rect=(0, 0, 1, .96 if nrow > 2 else .90))
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--no-bias", action="store_true",
                    help="score hexamers on the weight matrix alone")
    args = ap.parse_args()

    try:
        import logomaker  # noqa: F401
    except ImportError:
        print("FATAL: logomaker is not installed in this interpreter.\n"
              "       <python> -m pip install --no-input logomaker",
              file=sys.stderr)
        return 1

    out_dir = args.out_dir or args.run_dir / "filters"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, state = load_model(args.run_dir, args.ckpt)
    k = model.seq_kernel_size
    kmers = all_kmers(k)
    print(f"{args.run_dir.name}: epoch {state.get('epoch')}  "
          f"{model.num_seq_filters} filters x 2 arms  k={k}  "
          f"({len(kmers)} hexamers enumerated)")

    allrows = []
    for arm in ("incl", "skip"):
        dest = out_dir / f"logos_{arm}.png"
        rows = plot_bank(model, arm, kmers, dest, args.run_dir.name,
                         use_bias=not args.no_bias)
        allrows += rows
        n = np.array([r["n_hexamers"] for r in rows])
        bits = np.array([r["mean_bits"] for r in rows])
        print(f"  {arm}: n kept per filter min={n.min()} median="
              f"{int(np.median(n))} max={n.max()} of {len(kmers)}   "
              f"mean bits/position {bits.mean():.3f}")
        deg = [r["filter"] for r in rows if r["degenerate_threshold"]]
        if deg:
            print(f"        * filters {deg} had max activation <= 0; "
                  "threshold fell back to the activation midpoint")
        print(f"  wrote {dest}")

    tbl = out_dir / "logo_hexamer_counts.csv"
    pd.DataFrame(allrows).to_csv(tbl, index=False)
    print(f"wrote {tbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
