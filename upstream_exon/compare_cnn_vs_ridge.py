#!/usr/bin/env python
"""Paired comparison of the CNN against the 6-mer ridge on the test split.

baseline_comparison.csv reports each model's test R2 independently, which
cannot say whether a small gap is real: both models predict the SAME 40,000
rows, so their errors are strongly correlated and the unpaired standard
errors are far too wide. In full_v3 the gap is +0.009 R2 (0.6806 vs 0.6716),
which is exactly the regime where the distinction matters.

This refits the ridge from saved artifacts -- predictions.csv carries b1 and
y_true for every split -- so nothing is retrained and the CNN's predictions
are used exactly as the run produced them. The ridge fit reproduces
train_upstream.py's: fitted on the train rows, penalty chosen on the real
val rows.

    python upstream_exon/compare_cnn_vs_ridge.py \
        --run-dir outputs/upstream_exon/models/full_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as BL  # noqa: E402

N_BOOT = 10_000
BOOT_SEED = 20261107


def paired_stats(se_a: np.ndarray, se_b: np.ndarray, var_y: float,
                 n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Bootstrap the paired difference in squared error.

    Resampling ROWS (not models) keeps each row's two errors together, which
    is what makes the interval paired. dR2 follows from dMSE because both
    models are scored against the same target variance.
    """
    d = se_a - se_b                      # negative => A better
    rng = np.random.default_rng(seed)
    n = len(d)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "delta_mse": float(d.mean()),
        "delta_mse_ci95": [float(lo), float(hi)],
        "delta_r2": float(-d.mean() / var_y),
        "delta_r2_ci95": [float(-hi / var_y), float(-lo / var_y)],
        "boot_frac_favoring_a": float((boot < 0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    preds_path = args.run_dir / "predictions.csv"
    if not preds_path.exists():
        print(f"FATAL: {preds_path} not found", file=sys.stderr)
        return 1

    preds = pd.read_csv(preds_path, usecols=["split", "b1", "y_true", "y_pred"])
    need = {"train", "val", args.split}
    missing = need - set(preds.split.unique())
    if missing:
        print(f"FATAL: predictions.csv lacks split(s): {sorted(missing)}",
              file=sys.stderr)
        return 1

    tr = preds[preds.split == "train"]
    va = preds[preds.split == "val"]
    te = preds[preds.split == args.split].reset_index(drop=True)
    print(f"rows: train={len(tr)}  val={len(va)}  {args.split}={len(te)}")

    ridge = BL.kmer_gram_baseline(args.k)
    print(f"refitting {ridge.name} on {len(tr)} train rows "
          f"(penalty chosen on {len(va)} val rows) ...")
    ridge.fit(tr.b1.tolist(), tr.y_true.to_numpy(),
              va.b1.tolist(), va.y_true.to_numpy())
    print(f"  alpha={ridge.alpha_:g}  (selected on {ridge.alpha_selected_on_})")

    y = te.y_true.to_numpy()
    cnn = te.y_pred.to_numpy()
    rdg = np.asarray(ridge.predict(te.b1.tolist()), dtype=float)

    var_y = float(y.var())
    se_cnn, se_rdg = (cnn - y) ** 2, (rdg - y) ** 2
    r2 = lambda se: 1.0 - se.mean() / var_y  # noqa: E731

    # If the refit does not land on the number the run reported, the
    # comparison is not describing the same ridge and nothing below is valid.
    print(f"\nrefit ridge R2 = {r2(se_rdg):+.4f}  "
          "(compare against baseline_comparison.csv; a mismatch means the "
          "refit differs from the run's own fit)")

    err_corr = float(np.corrcoef(cnn - y, rdg - y)[0, 1])
    st = paired_stats(se_cnn, se_rdg, var_y, n_boot=args.n_boot)

    print(f"\n=== {args.split}: CNN vs {ridge.name} (n={len(te)}) ===")
    print(f"  CNN   R2={r2(se_cnn):+.4f}  MSE={se_cnn.mean():.4f}")
    print(f"  ridge R2={r2(se_rdg):+.4f}  MSE={se_rdg.mean():.4f}")
    print(f"  residual correlation between the two models: r={err_corr:+.4f}")
    print(f"\n  paired dR2 (CNN - ridge) = {st['delta_r2']:+.4f}  "
          f"95% CI [{st['delta_r2_ci95'][0]:+.4f}, {st['delta_r2_ci95'][1]:+.4f}]")
    print(f"  paired dMSE              = {st['delta_mse']:+.5f}  "
          f"95% CI [{st['delta_mse_ci95'][0]:+.5f}, {st['delta_mse_ci95'][1]:+.5f}]")
    print(f"  bootstrap replicates favoring CNN: {st['boot_frac_favoring_a']:.4f}")
    print(f"  rows where CNN is closer: {(se_cnn < se_rdg).mean():.4f}")

    try:
        from scipy.stats import wilcoxon
        w = wilcoxon(se_cnn, se_rdg)
        print(f"  Wilcoxon signed-rank on squared errors: "
              f"stat={w.statistic:.4g}  p={w.pvalue:.3g}")
        st["wilcoxon_p"] = float(w.pvalue)
    except ImportError:
        print("  (scipy unavailable -- bootstrap only)")

    excludes_zero = st["delta_r2_ci95"][0] > 0 or st["delta_r2_ci95"][1] < 0
    print("\nVERDICT: " + (
        "the paired interval excludes zero -- the difference is resolvable "
        "at this sample size."
        if excludes_zero else
        "the paired interval includes zero -- the two models are not "
        "distinguishable on this test set; do not claim one beats the other."))

    out = {"split": args.split, "n": int(len(te)), "k": args.k,
           "ridge_alpha": float(ridge.alpha_),
           "ridge_r2_refit": r2(se_rdg), "cnn_r2": r2(se_cnn),
           "residual_corr": err_corr, "n_boot": args.n_boot, **st}
    dest = args.run_dir / f"paired_cnn_vs_ridge_{args.split}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
