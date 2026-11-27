#!/usr/bin/env python
"""Turn Arush's infer.py output into the predictions.csv F1 wants, for v4.

Why this is not make_fnbm_predictions_csv.py: the v3 export ran
``export_fnbm_csv.py --center``, so undoing it is a single additive shift.
The v4 export ran ``--center --standardize``, so the stored target is

    y_std = (y_logit - center_shift) / target_scale

and both columns need the scale as well as the shift to come back to logit
units.  Applying only the shift -- what the v3 script does -- would leave the
axes in units of training-set standard deviations, silently.

Both constants are read from the export's own export_meta.json, never typed
in by hand, so the figure cannot be put on the wrong axis by a typo.

    python upstream_exon/make_fnbm_predictions_csv_v4.py \
        --infer-csv ~/Downloads/upstream_test_pred.csv \
        --export-meta ~/Downloads/upstream_export_meta.json \
        --out-dir outputs/upstream_exon/models/fnbm_o1_b1_v4 \
        --expected-mse 0.6013
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-csv", type=Path, required=True)
    ap.add_argument("--export-meta", type=Path, required=True,
                    help="export_meta.json from the SAME export the run trained on")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pred-col", default="pred_0")
    ap.add_argument("--target-col", default="logit_average")
    ap.add_argument("--split", default="test")
    ap.add_argument("--expected-mse", type=float, default=None,
                    help="test_loss from the run's test_metrics.csv, in "
                         "STANDARDIZED units, for the round-trip check")
    args = ap.parse_args()

    for f in (args.infer_csv, args.export_meta):
        if not f.exists():
            sys.exit(f"FATAL: {f} not found")

    meta = json.loads(args.export_meta.read_text())
    shift = float(meta["center_shift"])
    scale = float(meta.get("target_scale", 1.0))
    if meta.get("target_column") != args.target_col:
        print(f"WARNING: export_meta target_column is "
              f"{meta.get('target_column')!r}, not {args.target_col!r}",
              file=sys.stderr)

    d = pd.read_csv(args.infer_csv)
    for c in (args.pred_col, args.target_col):
        if c not in d.columns:
            sys.exit(f"FATAL: column {c!r} missing; got {list(d.columns)}")

    y_std = d[args.target_col].to_numpy(dtype=float)
    p_std = d[args.pred_col].to_numpy(dtype=float)

    # The run's own test_loss must come back out of the STANDARDIZED columns,
    # or this is not the checkpoint/split that produced the metric.
    mse_std = float(((p_std - y_std) ** 2).mean())
    r2 = 1.0 - mse_std / y_std.var()
    r = float(np.corrcoef(y_std, p_std)[0, 1])
    print(f"n={len(d):,}  center_shift={shift:.6f}  target_scale={scale:.6f}")
    print(f"standardized: MSE {mse_std:.6f}  R2 {r2:+.4f}  r {r:+.4f}")
    if args.expected_mse is not None:
        diff = abs(mse_std - args.expected_mse)
        print(f"run reported {args.expected_mse:.6f}  (diff {diff:.2e})")
        if diff > 2e-3:
            print("WARNING: MSE does not match test_metrics.csv -- check that "
                  "--infer-csv came from best.pt on the test split, and that "
                  "--export-meta is that run's export", file=sys.stderr)

    y = y_std * scale + shift
    p = p_std * scale + shift
    # R2, r and the normalized MSE are invariant to this affine map; only the
    # axis units change. Assert it rather than trust it.
    mse_logit = float(((p - y) ** 2).mean())
    assert abs(mse_logit - mse_std * scale ** 2) < 1e-9 * max(1.0, mse_logit)
    print(f"logit units:  MSE {mse_logit:.6g}  RMSE {np.sqrt(mse_logit):.6g}  "
          f"y range {y.min():.4f}..{y.max():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / "predictions.csv"
    pd.DataFrame({"split": args.split, "y_true": y, "y_pred": p}).to_csv(
        dest, index=False)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
