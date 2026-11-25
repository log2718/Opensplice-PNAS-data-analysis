#!/usr/bin/env python
"""Write the v3 splits as three CSVs in the FNBM repo's schema.

Arush's loader (``data/sequence.py``) wants one CSV per split with a sequence
column and a scalar target column, so the v3 dataset's two-file manifest +
results layout is joined and split here rather than in his code.

Two choices are deliberate and should not be "fixed" later:

  * The split comes from the dataset's own ``split`` column. His config also
    offers ``valid_fraction``, which carves a random validation set out of the
    training CSV using ``run.seed`` -- that would produce a different val set
    per seed and break comparability with the CNN / ridge / BiLSTM numbers.
    Writing valid.csv separately and pointing ``data.valid_csv`` at it keeps
    the exact 200k / 10k / 40k partition.
  * The target is the raw ``AG_mean_logit``, NOT standardized. ``test_loss``
    in his ``test_metrics.csv`` is then an MSE directly comparable to the
    ``mse`` column of our baseline_comparison.csv, and test R2 falls out as
    ``1 - test_loss / var(y_test)``.

``--center`` subtracts the TRAIN mean (scale untouched). MSE, variance and R2
are all unchanged by it; it only spares the model an epoch or two spent
walking its output bias from 0 up to the target mean of ~3.28.

    python upstream_exon/export_fnbm_csv.py \
        --dataset-dir outputs/upstream_exon/dataset_v3 \
        --out-dir /scratch/yl12316/data/b1_v3_fnbm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

#: v3 defaults. Both are overridable so the same exporter serves the v4
#: full-context libraries, whose results CSVs carry b1/syn70 + logit_average
#: in ONE file (no manifest join needed -- use --from-share).
SEQ_COL = "b1"
TGT_COL = "AG_mean_logit"
SPLITS = ("train", "valid", "test")
#: the dataset labels its validation split "val"; his config says "valid"
_SPLIT_ALIASES = {"valid": ("valid", "val")}


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    """Rows with <seq> / <target> / split, from a share CSV or the raw pair."""
    seq_col, tgt_col = args.sequence_column, args.target_column
    if args.from_share:
        return pd.read_csv(args.from_share,
                           usecols=[seq_col, tgt_col, "split"])

    man_p = args.dataset_dir / args.manifest
    res_p = args.dataset_dir / args.results
    for p in (man_p, res_p):
        if not p.exists():
            sys.exit(f"FATAL: {p} not found")

    man = pd.read_csv(man_p, usecols=["row_id", seq_col, "split"])
    res = pd.read_csv(res_p, usecols=["row_id", tgt_col])
    df = man.merge(res, on="row_id", how="inner", validate="one_to_one")
    print(f"joined {len(df):,} of {len(res):,} scored rows")
    if len(df) != len(res):
        print(f"WARNING: {len(res) - len(df):,} scored rows had no manifest row",
              file=sys.stderr)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-dir", type=Path)
    src.add_argument("--from-share", type=Path,
                     help="a CSV already written by export_dataset_share.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--manifest", default="b1_primary_manifest.csv")
    ap.add_argument("--results", default="b1_primary_ag_results.csv")
    ap.add_argument("--sequence-column", default=SEQ_COL,
                    help=f"default {SEQ_COL!r}; use 'syn70' for the middle "
                         f"library")
    ap.add_argument("--target-column", default=TGT_COL,
                    help=f"default {TGT_COL!r}; use 'logit_average' for v4")
    ap.add_argument("--center", action="store_true",
                    help="subtract the train mean from every split's target")
    ap.add_argument("--standardize", action="store_true",
                    help="ALSO divide by the train sd. Required for the v4 "
                         "libraries: the FNBM config's L2 penalties are "
                         "absolute and were tuned at var(target) ~ 0.414, so "
                         "an un-scaled target whose variance is 0.0021 "
                         "(upstream) or 8.5 (middle) changes the effective "
                         "regularization by 195x / 20x. R2 is scale-invariant "
                         "and var_pop is written post-scaling, so the reported "
                         "R2 is unaffected.")
    args = ap.parse_args()
    seq_col, tgt_col = args.sequence_column, args.target_column

    df = load_rows(args)
    df = df.dropna(subset=[seq_col, tgt_col])

    lengths = sorted(df[seq_col].str.len().unique())
    if len(lengths) != 1:
        sys.exit(f"FATAL: sequences are not all one length: {lengths}")
    bad = sorted(set("".join(df[seq_col].head(2000))) - set("ACGTN"))
    if bad:
        sys.exit(f"FATAL: non-ACGTN characters in {seq_col}: {bad}")

    parts: dict[str, pd.DataFrame] = {}
    for name in SPLITS:
        wanted = _SPLIT_ALIASES.get(name, (name,))
        part = df[df.split.isin(wanted)]
        if part.empty:
            sys.exit(f"FATAL: no rows for split {name!r} (looked for {wanted})")
        parts[name] = part

    shift = 0.0
    if args.center:
        shift = float(parts["train"][tgt_col].mean())
        print(f"centering: subtracting train mean {shift:.6f}")
    scale = 1.0
    if args.standardize:
        # ddof=0 to match losses.r2's SST/n, and computed on TRAIN only -- the
        # val/test sd must not leak into the transform.
        scale = float(parts["train"][tgt_col].std(ddof=0))
        if not scale > 0:
            sys.exit(f"FATAL: train sd of {tgt_col} is {scale}; cannot scale")
        print(f"standardizing: dividing by train sd {scale:.6f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"sequence_column": seq_col, "target_column": tgt_col,
            "input_length": int(lengths[0]), "center_shift": shift,
            # multiply a prediction by this to return to target units
            "target_scale": scale,
            "source": str(args.from_share or args.dataset_dir), "splits": {}}

    for name, part in parts.items():
        out = part[[seq_col, tgt_col]].copy()
        if shift:
            out[tgt_col] = out[tgt_col] - shift
        if scale != 1.0:
            out[tgt_col] = out[tgt_col] / scale
        dest = args.out_dir / f"{name}.csv"
        out.to_csv(dest, index=False)
        # var is the population variance: it is the SST/n that test R2 divides
        # by, so it has to match losses.r2's ddof=0, not pandas' default 1.
        v = float(out[tgt_col].var(ddof=0))
        meta["splits"][name] = {"n": len(out), "mean": float(out[tgt_col].mean()),
                                "var_pop": v}
        print(f"{name:5s} {len(out):>7,} rows  mean {out[tgt_col].mean():+.4f}  "
              f"var {v:.6f}  -> {dest}")

    meta_p = args.out_dir / "export_meta.json"
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_p}")
    print(f"\nset in the config:  input_length: {lengths[0]}   "
          f"sequence_column: {seq_col}   target_column: {tgt_col}")
    print(f"test R2 will be:    1 - test_loss / {meta['splits']['test']['var_pop']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
