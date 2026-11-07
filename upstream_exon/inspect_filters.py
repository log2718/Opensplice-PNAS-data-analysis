#!/usr/bin/env python
"""Rank and plot the learned filters of a trained UpstreamExonModel.

Importance is CONTRIBUTION VARIANCE ACROSS REAL SEQUENCES, not weight norm.
That distinction is the whole point here: the prediction is

    y = mu + sd * (head_w * (w * (sum(incl) - sum(skip)) + b) + head_b)

so one filter-arm's contribution, in original logit units, is

    contrib[n,arm,f] = s(arm) * c * sum_p softplus(conv[f] * x_n + pbias[f,p])
    c = sd * head_w * w        s(incl) = +1,  s(skip) = -1

c folds in the learned sign of the output head. When head_w * w is negative --
as it is in both full_v2 and full_v3 -- the arm NAMED "incl" pushes the
predicted logit DOWN, so reading the arm names as inclusion/skipping would
invert every conclusion. Everything below reports effective signed logit units.

A filter whose contribution is large but CONSTANT across sequences cannot
explain any variation in the response; in the v3 library, where all 250,000
sequences share one exact composition, that is the main failure mode worth
guarding against. So filters are ranked by the SD of contrib over sequences,
and the mean is reported alongside as a separate (offset) quantity.

    python upstream_exon/inspect_filters.py \
        --run-dir outputs/upstream_exon/models/full_v3 --top 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_upstream as D  # noqa: E402
from model_upstream import UpstreamExonModel  # noqa: E402
from plot_filter_logos import all_kmers, ic_matrix  # noqa: E402

BASES = "ACGT"
BASE_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
SAMPLE_SEED = 20261107


# ======================================================================
# contributions
# ======================================================================

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


@torch.no_grad()
def contributions(model, seqs: list[str], scaler_sd: float,
                  batch: int = 512) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-sequence, per-arm-filter contribution in signed logit units.

    Returns ``(contrib, profile, c)``:
      contrib (N, 2, F)       -- summed over position, signed
      profile (2, F, L-5)     -- mean over sequences, signed, per position
      c                       -- the effective coefficient sd * head_w * w
    """
    w = float(model.energy_seq_struct.w.item())
    head_w = (float(model.head.weight.item())
              if model.head_kind == "linear" else 1.0)
    c = scaler_sd * head_w * w
    sign = np.array([1.0, -1.0])            # incl adds, skip subtracts

    F = model.num_seq_filters
    L = model.conv_out_length
    out = np.empty((len(seqs), 2, F), dtype=np.float64)
    prof = np.zeros((2, F, L), dtype=np.float64)

    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        x = torch.from_numpy(D.encode_sequences(chunk))
        a_incl, a_skip = model.activations(x)           # (n, F, L) each
        a = torch.stack([a_incl, a_skip], dim=1).numpy()   # (n, 2, F, L)
        a = a[:, :, :F, :]        # drop structure filters if any are appended
        signed = c * sign[None, :, None, None] * a
        out[i:i + len(chunk)] = signed.sum(axis=3)
        prof += signed.sum(axis=0)
    prof /= len(seqs)
    return out, prof, c


def logo_matrix(weight: np.ndarray) -> np.ndarray:
    """(4, k) conv weights -> per-column mean-centered preference.

    A constant added to one column of a conv filter shifts every window's
    score by the same amount, so only the SPREAD within a column discriminates
    between sequences. Centering removes the part that cannot.
    """
    return weight - weight.mean(axis=0, keepdims=True)


def describe_filter(logo: np.ndarray, profile: np.ndarray) -> dict:
    """Separate composition-like filters from position-specific motifs."""
    # Composition-like: the same base is preferred at every kernel offset, so
    # the per-base row sums dominate the total centered weight.
    row = np.abs(logo.sum(axis=1))
    denom = np.abs(logo).sum()
    single_base = float(row.max() / denom) if denom > 0 else 0.0
    pref = BASES[int(np.argmax(logo.sum(axis=1)))]

    # Position-specific: the mean positional profile has structure rather than
    # being flat across the exon body.
    m = np.abs(profile).mean()
    peak = float(np.abs(profile).max() / m) if m > 0 else 0.0
    flatness = float(profile.std() / m) if m > 0 else 0.0
    # Positional flatness is the clean discriminator. A filter whose mean
    # contribution is uniform along the exon is reading composition (or a
    # motif with no preferred location); one with structure along position is
    # reading WHERE a motif sits. Base preference is reported as a descriptor,
    # not used as a gate -- single_base_frac runs ~0.25 for an unbiased filter,
    # so the informative range is narrow and thresholding it is unreliable.
    # Named for what is MEASURED -- positional flatness -- not for an
    # interpretation that only holds in one library. In v2 (Dirichlet
    # composition) a flat filter was reading base composition; in v3
    # composition is identical across every sequence by construction, so a
    # flat filter there is a motif with no preferred location. Calling both
    # "composition-like" would be wrong for v3.
    kind = ("position-independent" if flatness < 0.15
            else "position-specific" if flatness >= 0.50
            else "mixed")
    return {"preferred_base": pref, "single_base_frac": single_base,
            "position_peak_over_mean": peak, "position_cv": flatness,
            "kind": kind}


# ======================================================================
# plotting
# ======================================================================

def draw_ic_logo(ax, df, n_kept: int) -> None:
    """Schneider & Stephens logo, identical to plot_filter_logos.py."""
    import logomaker
    logomaker.Logo(df, ax=ax, color_scheme=BASE_COLORS, show_spines=False)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(range(1, len(df) + 1), fontsize=7)
    ax.tick_params(labelsize=7)


def draw_logo(ax, logo: np.ndarray) -> None:
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    k = logo.shape[1]
    for p in range(k):
        col = logo[:, p]
        # Stack each column outward from zero, smallest letter innermost.
        for pos in (True, False):
            idx = np.where(col > 0)[0] if pos else np.where(col < 0)[0]
            idx = idx[np.argsort(np.abs(col[idx]))]
            run = 0.0
            for b in idx:
                h = abs(float(col[b]))
                letter = BASES[b]
                tp = TextPath((0, 0), letter, size=1, prop=None)
                bb = tp.get_extents()
                if bb.width <= 0 or bb.height <= 0:
                    continue
                tr = (Affine2D()
                      .translate(-bb.x0, -bb.y0)
                      .scale(0.82 / bb.width, h / bb.height)
                      .translate(p + 0.09, run if pos else -(run + h)))
                ax.add_patch(PathPatch(tr.transform_path(tp),
                                       fc=BASE_COLORS[letter], ec="none"))
                run += h
    lim = np.abs(logo).sum(axis=0).max() * 0.6 + 1e-9
    ax.set_xlim(0, k)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#444", lw=.6)
    ax.set_xticks(np.arange(k) + .5)
    ax.set_xticklabels(range(1, k + 1), fontsize=7)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make_figure(rows: list[dict], prof: np.ndarray, kernel: int,
                total_sd: float, dest: Path, run_name: str,
                logo_mode: str = "ic") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(13, 1.55 * n),
                             gridspec_kw={"width_ratios": [1, 3.1]},
                             squeeze=False)
    # Consistent scales so panels are comparable at a glance.
    ic = logo_mode == "ic"
    logo_lim = (min(2.0, max(float(r["ic_df"].sum(axis=1).max())
                             for r in rows) * 1.12) if ic
                else max(np.abs(r["logo"]).sum(axis=0).max() for r in rows) * 0.6)
    prof_lim = max(np.abs(prof[r["arm_i"], r["filter"]]).max() for r in rows) * 1.12

    for i, r in enumerate(rows):
        la, pa = axes[i][0], axes[i][1]
        if ic:
            draw_ic_logo(la, r["ic_df"], r["n_hexamers"])
            la.set_ylim(0, logo_lim)
        else:
            draw_logo(la, r["logo"])
            la.set_ylim(-logo_lim, logo_lim)
        la.set_ylabel(f"#{r['filter']} {r['arm']}\n(n={r['n_hexamers']})",
                      fontsize=8, rotation=0, ha="right", va="center",
                      labelpad=30)

        y = prof[r["arm_i"], r["filter"]]
        x = np.arange(len(y)) + kernel / 2.0      # window centre in B1 coords
        pa.axhline(0, color="#444", lw=.6)
        pa.fill_between(x, 0, y,
                        color="#d62728" if r["mean_contrib"] < 0 else "#1f77b4",
                        alpha=.45, lw=0)
        pa.plot(x, y, color="#222", lw=.8)
        pa.set_ylim(-prof_lim, prof_lim)
        pa.set_xlim(0, 113)
        pa.tick_params(labelsize=7)
        for s in ("top", "right"):
            pa.spines[s].set_visible(False)
        pa.text(.995, .93,
                f"importance (SD) {r['contrib_sd']:+.4f}   "
                f"mean {r['mean_contrib']:+.3f} logit   {r['kind']}",
                transform=pa.transAxes, ha="right", va="top", fontsize=7.5,
                color="#222")
        if i == 0:
            la.set_title("6-nt filter: information content (bits)" if ic
                         else "6-nt filter (mean-centered weights)", fontsize=9)
            pa.set_title("mean signed contribution per position "
                         "(logit units)", fontsize=9)
        if i == n - 1:
            la.set_xlabel("kernel offset" + ("  (bits; n = hexamers above "
                          "half-max activation)" if ic else ""), fontsize=8)
            pa.set_xlabel("position in 113-nt upstream exon (window centre)",
                          fontsize=8)

    fig.suptitle(
        f"{run_name}: top {n} filter-arms by contribution SD across test "
        f"sequences   (total prediction SD {total_sd:.3f} logit)",
        fontsize=10.5, y=.998)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(dest, dpi=200)
    plt.close(fig)


# ======================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <run-dir>/filters")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-seqs", type=int, default=4000)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--logo-mode", choices=("ic", "weights"), default="ic",
                    help="ic: Schneider & Stephens information content, "
                         "matching logos_{incl,skip}.png; weights: raw "
                         "mean-centered conv weights")
    args = ap.parse_args()

    out_dir = args.out_dir or args.run_dir / "filters"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, state = load_model(args.run_dir, args.ckpt)
    scaler = state.get("scaler") or {}
    s_sd = float(scaler.get("sd", 1.0)) if scaler.get("active", True) else 1.0
    print(f"{args.run_dir.name}: epoch {state.get('epoch')}  "
          f"filters={model.num_seq_filters}  k={model.seq_kernel_size}  "
          f"target sd={s_sd:.4f}")

    preds = pd.read_csv(args.run_dir / "predictions.csv",
                        usecols=["split", "b1", "y_true", "y_pred"])
    pool = preds[preds.split == args.split]
    if pool.empty:
        print(f"FATAL: no rows in split {args.split!r}", file=sys.stderr)
        return 1
    n = min(args.n_seqs, len(pool))
    samp = pool.sample(n=n, random_state=SAMPLE_SEED)
    seqs = samp.b1.tolist()
    print(f"sampling {n} of {len(pool)} {args.split} sequences")

    contrib, prof, c = contributions(model, seqs, s_sd)
    note = ("NEGATIVE -- the arm named incl LOWERS the logit" if c < 0
            else "positive -- arm names read literally")
    print(f"effective coefficient c = sd * head_w * w = {c:+.6f}   ({note})")

    # Reconstructing the model's own spread from the parts is the check that
    # the decomposition is exact: these must agree.
    recon = contrib.sum(axis=(1, 2))
    actual = samp.y_pred.to_numpy()
    print(f"reconstruction vs predictions.csv: "
          f"sd {recon.std():.4f} vs {actual.std():.4f}  "
          f"max|offset-free diff| "
          f"{np.abs((recon - recon.mean()) - (actual - actual.mean())).max():.2e}")

    sd_mat = contrib.std(axis=0)                 # (2, F)
    mean_mat = contrib.mean(axis=0)
    order = np.dstack(np.unravel_index(
        np.argsort(-sd_mat, axis=None), sd_mat.shape))[0]

    arms = ("incl", "skip")
    kmers = all_kmers(model.seq_kernel_size)
    rows, table = [], []
    for arm_i, f in order:
        conv = model.conv_incl if arm_i == 0 else model.conv_skip
        logo = logo_matrix(conv.weight[f].detach().numpy())
        ic_df, n_kept, _ = ic_matrix(conv.weight[f].detach().numpy(),
                                     float(conv.bias[f].detach()), kmers)
        d = describe_filter(logo, prof[arm_i, f])
        rec = {"filter": int(f), "arm": arms[arm_i], "arm_i": int(arm_i),
               "n_hexamers": int(n_kept),
               "contrib_sd": float(sd_mat[arm_i, f]),
               "mean_contrib": float(mean_mat[arm_i, f]),
               "effective_sign": "raises logit" if mean_mat[arm_i, f] > 0
                                 else "lowers logit", **d}
        table.append(rec)
        if len(rows) < args.top:
            rows.append({**rec, "logo": logo, "ic_df": ic_df})

    total_sd = float(recon.std())
    sum_abs_sd = float(sd_mat.sum())
    print(f"\ncancellation: sum of per-filter SDs = {sum_abs_sd:.3f} logit, "
          f"but total prediction SD = {total_sd:.3f} logit "
          f"-> ratio {total_sd / sum_abs_sd:.3f}")
    flat = contrib.reshape(len(seqs), -1)
    cm = np.corrcoef(flat.T)
    iu = np.triu_indices_from(cm, k=1)
    print(f"pairwise contribution correlation: median {np.median(cm[iu]):+.3f}, "
          f"{(cm[iu] < -0.5).sum()} of {len(cm[iu])} pairs below -0.5")

    print(f"\ntop {args.top} filter-arms by contribution SD:")
    hdr = f"{'arm':>5} {'f':>3} {'SD':>9} {'mean':>9}  {'base':>4} " \
          f"{'1base':>6} {'posCV':>6} {'n':>5}  kind"
    print(hdr)
    for r in table[:args.top]:
        print(f"{r['arm']:>5} {r['filter']:>3} {r['contrib_sd']:>9.4f} "
              f"{r['mean_contrib']:>9.3f}  {r['preferred_base']:>4} "
              f"{r['single_base_frac']:>6.2f} {r['position_cv']:>6.2f} "
              f"{r['n_hexamers']:>5}  {r['kind']}")

    tdf = pd.DataFrame(table).drop(columns=["arm_i"])
    tdf.to_csv(out_dir / "filter_importance.csv", index=False)
    summary = {"run": args.run_dir.name, "epoch": state.get("epoch"),
               "split": args.split, "n_seqs": n,
               "effective_coefficient": c, "target_sd": s_sd,
               "total_prediction_sd": total_sd,
               "sum_per_filter_sd": sum_abs_sd,
               "cancellation_ratio": total_sd / sum_abs_sd,
               "median_pair_corr": float(np.median(cm[iu])),
               "n_pairs_corr_below_-0.5": int((cm[iu] < -0.5).sum())}
    (out_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2))

    dest = out_dir / f"top{args.top}_filters_logo_position.png"
    make_figure(rows, prof, model.seq_kernel_size, total_sd, dest,
                args.run_dir.name, logo_mode=args.logo_mode)
    print(f"\nwrote {dest}")
    print(f"wrote {out_dir / 'filter_importance.csv'}")
    print(f"wrote {out_dir / 'filter_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
