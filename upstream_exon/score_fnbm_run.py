#!/usr/bin/env python
"""Turn an FNBM run's test_metrics.csv into an R2 comparable to our baselines.

His loop records test_loss and test_pearson_r but no R2. Because the loss is
plain MSE on the raw target, test_loss IS the mean squared error in target
units, so

    R2 = 1 - test_loss / var_pop(y_test)

with var_pop the population variance (ddof=0) that losses.r2's SST/n implies.
export_fnbm_csv.py records that variance in export_meta.json, so nothing has
to re-read the 40,000-row test CSV.

Centering the target does not affect this: it shifts the mean, leaving both
the squared errors and the variance untouched.

    python upstream_exon/score_fnbm_run.py \
        --run-dir /scratch/yl12316/results/upstream-exon-fnbm/fnbm_o1_b1_v3 \
        --export-meta /scratch/yl12316/data/b1_v3_fnbm/export_meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

#: test-split numbers from outputs/upstream_exon/models/*/baseline_comparison.csv
REFERENCE = [
    ("constant", 0.414383, -0.0000),
    ("dinucleotide_ridge", 0.250261, 0.3961),
    ("kmer6_ridge", 0.136086, 0.6716),
    ("interpretable CNN (full_v3)", 0.132336, 0.6806),
    ("BiLSTM (lstm_v3)", 0.074623, 0.8199),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--export-meta", type=Path, required=True)
    ap.add_argument("--var", type=float, default=None,
                    help="override the test population variance")
    args = ap.parse_args()

    tm = args.run_dir / "test_metrics.csv"
    if not tm.exists():
        print(f"no {tm} -- the run has not finished its test pass",
              file=sys.stderr)
        return 1

    with tm.open(newline="") as f:
        row = next(iter(csv.DictReader(f)))
    test_loss = float(row["test_loss"])
    pearson = float(row["test_pearson_r"])
    best_epoch = row.get("best_epoch", "?")

    if args.var is not None:
        var = args.var
    else:
        meta = json.loads(args.export_meta.read_text())
        var = float(meta["splits"]["test"]["var_pop"])

    r2 = 1.0 - test_loss / var

    print(f"run        : {args.run_dir.name}  (best epoch {best_epoch})")
    print(f"test MSE   : {test_loss:.6f}   over var_pop {var:.6f}")
    print(f"test R2    : {r2:+.4f}")
    print(f"test r     : {pearson:+.4f}   (r^2 = {pearson ** 2:.4f})")
    # r2 < r^2 is the signature of a miscalibrated slope or offset rather than
    # of weak correlation, and is worth seeing separately from R2.
    if pearson ** 2 - r2 > 0.01:
        print(f"note: r^2 exceeds R2 by {pearson ** 2 - r2:.4f}; the "
              f"predictions correlate better than they calibrate")

    print("\n%-30s %10s %8s" % ("model", "test MSE", "test R2"))
    for name, mse, ref_r2 in REFERENCE:
        print("%-30s %10.6f %+8.4f" % (name, mse, ref_r2))
    print("%-30s %10.6f %+8.4f  <- this run" % ("FNBM order-1", test_loss, r2))

    better = [n for n, _, rr in REFERENCE if r2 > rr]
    print(f"\nbeats: {', '.join(better) if better else 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
