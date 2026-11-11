#!/usr/bin/env python
"""Join manifest + AlphaGenome scores into one modelling-ready CSV to share.

The dataset lives in two files joined on row_id: the manifest holds the B1
sequence and its composition QC, the results file holds the AlphaGenome
scores. A collaborator wants one table, so this emits sequence + target +
split and nothing else.

The split column is carried through deliberately. It is assigned by sequence
hash upstream, so anyone who re-splits randomly gets a different test set and
their R2 is no longer comparable to ours.

    python upstream_exon/export_dataset_share.py \
        --dataset-dir outputs/upstream_exon/dataset_v3 \
        --out b1_v3_upstream_exon_250k.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KEEP_MANIFEST = ["row_id", "b1_id", "b1", "split", "gc", "entropy_bits",
                 "max_homopolymer", "count_A", "count_C", "count_G", "count_T"]
KEEP_RESULTS = ["row_id", "AG_acceptor_probability", "AG_donor_probability",
                "AG_mean_prob", "AG_mean_logit"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", default="b1_primary_manifest.csv")
    ap.add_argument("--results", default="b1_primary_ag_results.csv")
    ap.add_argument("--gzip", action="store_true")
    args = ap.parse_args()

    man_p = args.dataset_dir / args.manifest
    res_p = args.dataset_dir / args.results
    for p in (man_p, res_p):
        if not p.exists():
            print(f"FATAL: {p} not found", file=sys.stderr)
            return 1

    man = pd.read_csv(man_p, usecols=lambda c: c in KEEP_MANIFEST)
    res = pd.read_csv(res_p, usecols=lambda c: c in KEEP_RESULTS)
    print(f"manifest {len(man):,} rows   results {len(res):,} rows")

    df = man.merge(res, on="row_id", how="inner", validate="one_to_one")
    print(f"joined   {len(df):,} rows")
    if len(df) != len(res):
        print(f"WARNING: {len(res) - len(df):,} scored rows had no manifest "
              f"entry and were dropped", file=sys.stderr)

    # A row with no target is useless downstream; say so rather than ship it.
    bad = df.AG_mean_logit.isna().sum()
    if bad:
        print(f"dropping {bad:,} rows with a missing target", file=sys.stderr)
        df = df[df.AG_mean_logit.notna()]

    lens = df.b1.str.len().unique()
    print(f"B1 length(s): {sorted(lens)}")
    print(f"split sizes : {df.split.value_counts().to_dict()}")
    print(f"target      : mean {df.AG_mean_logit.mean():.4f}  "
          f"sd {df.AG_mean_logit.std():.4f}  "
          f"range [{df.AG_mean_logit.min():.3f}, "
          f"{df.AG_mean_logit.max():.3f}]")

    out = args.out
    if args.gzip and out.suffix != ".gz":
        out = out.with_suffix(out.suffix + ".gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
