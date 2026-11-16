#!/usr/bin/env python
"""Turn Arush's infer.py output into the predictions.csv plot_pred_vs_true wants.

His infer.py appends a ``pred_0`` column to whatever CSV it was given, so the
file it writes for our test split holds ``b1``, ``AG_mean_logit`` and
``pred_0`` -- all in CENTERED units, because export_fnbm_csv.py was run with
--center and subtracted the train mean (3.276854) from every split.

Adding that shift back to both columns puts the figure on the same absolute
AG_mean_logit axis as the CNN and BiLSTM plots. The shift cancels in MSE, R2
and correlation, so no metric changes -- only the axis labels.

    python upstream_exon/make_fnbm_predictions_csv.py \
        --infer-csv ~/Downloads/fnbm_test_predictions.csv \
        --out-dir outputs/upstream_exon/models/fnbm_o1_b1_v3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: train-split mean subtracted by export_fnbm_csv.py --center, as printed by
#: job 23142864: "center_shift=3.276854".
DEFAULT_SHIFT = 3.276854
#: test MSE this run reported, for the round-trip check
EXPECTED_MSE = 0.120584


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shift", type=float, default=DEFAULT_SHIFT,
                    help="train mean added back to both columns")
    ap.add_argument("--pred-col", default="pred_0")
    ap.add_argument("--target-col", default="AG_mean_logit")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    if not args.infer_csv.exists():
        sys.exit(f"FATAL: {args.infer_csv} not found")
    d = pd.read_csv(args.infer_csv)
    for c in (args.pred_col, args.target_col):
        if c not in d.columns:
            sys.exit(f"FATAL: column {c!r} missing; got {list(d.columns)}")

    y = d[args.target_col].to_numpy(dtype=float)
    p = d[args.pred_col].to_numpy(dtype=float)

    # The run's own test_loss must come back out, or the checkpoint, the split
    # or the centering convention is not the one that produced the metric.
    mse = float(((p - y) ** 2).mean())
    print(f"n={len(d):,}  recomputed test MSE {mse:.6f}  "
          f"(run reported {EXPECTED_MSE:.6f}, diff {abs(mse - EXPECTED_MSE):.2e})")
    if abs(mse - EXPECTED_MSE) > 2e-3:
        print("WARNING: MSE does not match the run's test_metrics.csv -- check "
              "that --infer-csv came from best.pt on the test split",
              file=sys.stderr)
    print(f"R2 = {1 - mse / y.var():.4f}   r = {np.corrcoef(y, p)[0, 1]:+.4f}")

    out = pd.DataFrame({"split": args.split,
                        "y_true": y + args.shift,
                        "y_pred": p + args.shift})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / "predictions.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  (shift +{args.shift:.6f} applied to both columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
