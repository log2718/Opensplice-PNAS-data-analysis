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

SEQ_COL = "b1"
TGT_COL = "AG_mean_logit"
SPLITS = ("train", "valid", "test")
#: the dataset labels its validation split "val"; his config says "valid"
_SPLIT_ALIASES = {"valid": ("valid", "val")}


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    """Rows with b1 / AG_mean_logit / split, from a share CSV or the raw pair."""
    if args.from_share:
        return pd.read_csv(args.from_share,
                           usecols=[SEQ_COL, TGT_COL, "split"])

    man_p = args.dataset_dir / args.manifest
    res_p = args.dataset_dir / args.results
    for p in (man_p, res_p):
        if not p.exists():
            sys.exit(f"FATAL: {p} not found")

    man = pd.read_csv(man_p, usecols=["row_id", SEQ_COL, "split"])
    res = pd.read_csv(res_p, usecols=["row_id", TGT_COL])
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
    ap.add_argument("--center", action="store_true",
                    help="subtract the train mean from every split's target")
    args = ap.parse_args()

    df = load_rows(args)
    df = df.dropna(subset=[SEQ_COL, TGT_COL])

    lengths = sorted(df[SEQ_COL].str.len().unique())
    if len(lengths) != 1:
        sys.exit(f"FATAL: sequences are not all one length: {lengths}")
    bad = sorted(set("".join(df[SEQ_COL].head(2000))) - set("ACGTN"))
    if bad:
        sys.exit(f"FATAL: non-ACGTN characters in b1: {bad}")

    parts: dict[str, pd.DataFrame] = {}
    for name in SPLITS:
        wanted = _SPLIT_ALIASES.get(name, (name,))
        part = df[df.split.isin(wanted)]
        if part.empty:
            sys.exit(f"FATAL: no rows for split {name!r} (looked for {wanted})")
        parts[name] = part

    shift = 0.0
    if args.center:
        shift = float(parts["train"][TGT_COL].mean())
        print(f"centering: subtracting train mean {shift:.6f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"sequence_column": SEQ_COL, "target_column": TGT_COL,
            "input_length": int(lengths[0]), "center_shift": shift,
            "source": str(args.from_share or args.dataset_dir), "splits": {}}

    for name, part in parts.items():
        out = part[[SEQ_COL, TGT_COL]].copy()
        if shift:
            out[TGT_COL] = out[TGT_COL] - shift
        dest = args.out_dir / f"{name}.csv"
        out.to_csv(dest, index=False)
        # var is the population variance: it is the SST/n that test R2 divides
        # by, so it has to match losses.r2's ddof=0, not pandas' default 1.
        v = float(out[TGT_COL].var(ddof=0))
        meta["splits"][name] = {"n": len(out), "mean": float(out[TGT_COL].mean()),
                                "var_pop": v}
        print(f"{name:5s} {len(out):>7,} rows  mean {out[TGT_COL].mean():+.4f}  "
              f"var {v:.6f}  -> {dest}")

    meta_p = args.out_dir / "export_meta.json"
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_p}")
    print(f"\nset in the config:  input_length: {lengths[0]}   "
          f"sequence_column: {SEQ_COL}   target_column: {TGT_COL}")
    print(f"test R2 will be:    1 - test_loss / {meta['splits']['test']['var_pop']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
