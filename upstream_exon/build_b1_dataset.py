"""
Build and validate the B1 training-library manifest.

Generates Dirichlet B1 sequences, inserts each into the fixed PNAS 1,309-nt
cassette around a fixed middle exon, assigns reproducible alpha-stratified
train/val/test splits, and writes a manifest ready for AlphaGenome scoring.

Makes NO AlphaGenome calls. Writes only under ``outputs/upstream_exon/``.

Two modes:

``primary``      the training library for ``syn_a5.0_00482``.
``replication``  the SAME B1 sequences as the primary split's TEST rows, but
                 inserted around ``syn_a0.5_00475``. This satisfies "held-out
                 B1 sequences with no training leakage" while keeping the two
                 datasets PAIRED on B1, which is what makes a ranking
                 comparison across middle exons possible at all.

Usage::

    python upstream_exon/build_b1_dataset.py --n 10000
    python upstream_exon/build_b1_dataset.py --n 10000 --validate-only
    python upstream_exon/build_b1_dataset.py --mode replication
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b1_dirichlet as G  # noqa: E402
import b1_lib as B  # noqa: E402
import splice_site_partition as SP  # noqa: E402

PRIMARY_EXON = "syn_a5.0_00482"
REPLICATION_EXON = "syn_a0.5_00475"

OUT = ROOT / "outputs" / "upstream_exon"
DATA = OUT / "dataset_v1"

SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLIT_SEED = 7_20261020


def exon_record(sequence_id: str) -> dict:
    lib = pd.read_csv(ROOT / "outputs" / "synthetic_context" /
                      "synthetic_exon_library.csv")
    row = lib[lib.sequence_id == sequence_id]
    if row.empty:
        raise SystemExit(f"unknown exon {sequence_id}")
    r = row.iloc[0]
    return {"sequence_id": sequence_id, "alpha_exon": float(r.alpha),
            "synthetic_70mer": r.synthetic_70mer,
            "full_middle_exon": r.full_middle_exon,
            "PNAS_common": float(r.PNAS_common)}


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Alpha-stratified, deterministic, keyed on b1_sha256."""
    df = df.copy()
    df["split"] = ""
    for alpha, idx in df.groupby("alpha").groups.items():
        idx = np.array(sorted(idx))
        # blake2b, not hash(): Python string hashing is salted per process
        r = G._rng(SPLIT_SEED, "split", float(alpha))
        order = r.permutation(len(idx))
        n = len(idx)
        n_tr = int(round(n * SPLIT_FRACTIONS["train"]))
        n_va = int(round(n * SPLIT_FRACTIONS["val"]))
        lab = np.array(["test"] * n, dtype=object)
        lab[order[:n_tr]] = "train"
        lab[order[n_tr:n_tr + n_va]] = "val"
        df.loc[idx, "split"] = lab
    return df


def build(mode: str, n_total: int, seed: int) -> tuple[pd.DataFrame, dict]:
    if mode == "primary":
        exon = exon_record(PRIMARY_EXON)
        panel = pd.read_csv(OUT / "b1_variant_panel.csv")
        recs, report = G.generate(n_total, seed=seed,
                                  exclude=set(panel.b1))
        df = pd.DataFrame(recs)
        for k, v in pd.DataFrame([G.qc_flags(r) for r in recs]).items():
            df[k] = v
        df = assign_splits(df)
    else:
        prim = DATA / "b1_primary_manifest.csv"
        if not prim.exists():
            raise SystemExit(f"missing {prim}; build the primary dataset first")
        df = pd.read_csv(prim)
        df = df[df.split == "test"].copy()
        df["split"] = "replication"
        exon = exon_record(REPLICATION_EXON)
        report = {"source": "primary test split", "produced": len(df),
                  "unique": df.b1_sha256.nunique()}

    rows = []
    for r in df.itertuples():
        core, start = B.build_construct(r.b1, exon["synthetic_70mer"])
        rows.append({"core_sha256": B.sha(core), "core_length": len(core),
                     "exon_start": start,
                     "exon_len": len(exon["full_middle_exon"])})
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(rows)], axis=1)
    df["sequence_id"] = exon["sequence_id"]
    df["PNAS_common"] = exon["PNAS_common"]
    df["row_id"] = df.sequence_id + "|" + df.b1_id
    report["exon"] = exon["sequence_id"]
    report["mode"] = mode
    return df, report


def validate(df: pd.DataFrame, exon: dict, mode: str) -> list[str]:
    fails: list[str] = []
    lines: list[str] = []

    def chk(name, ok, detail=""):
        line = f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else "")
        print(line)
        lines.append(line)
        if not ok:
            fails.append(name)

    native = SP.pnas_blocks8()

    print("\n1. Sequences")
    chk("1a. every B1 is exactly 113 nt",
        (df.length == G.B1_LEN).all() if "length" in df else
        all(len(s) == G.B1_LEN for s in df.b1))
    chk("1b. alphabet is ACGT only",
        all(not (set(s) - set(G.ALPHABET)) for s in df.b1))
    chk("1c. all B1 sequences unique", df.b1_sha256.nunique() == len(df),
        f"{df.b1_sha256.nunique()}/{len(df)}")
    chk("1d. b1_sha256 matches the sequence",
        all(G.sha(r.b1) == r.b1_sha256 for r in df.itertuples()))

    print("\n2. Alpha mixture")
    got = df.alpha.value_counts(normalize=True).sort_index().round(3).to_dict()
    if mode == "primary":
        want = {a: round(p, 3) for a, p in G.ALPHA_MIX.items()}
        chk("2a. realized alpha mixture matches 20/40/40",
            all(abs(got.get(a, 0) - p) < 0.01 for a, p in want.items()),
            f"got {got}")
    else:
        chk("2a. replication inherits the primary alpha mixture", True, f"got {got}")

    print("\n3. Constructs")
    sample = df if len(df) <= 400 else df.sample(400, random_state=0)
    bad = []
    for r in sample.itertuples():
        core, start = B.build_construct(r.b1, exon["synthetic_70mer"])
        f = B.validate_construct(core, start, r.b1, exon["synthetic_70mer"])
        if G.sha(core) != r.core_sha256 or start != r.exon_start:
            f.append("manifest digest/coordinate mismatch")
        if f:
            bad.append((r.b1_id, f))
    chk(f"3a. {len(sample)} sampled constructs pass every structural assertion",
        not bad, f"{len(bad)} bad" + (f"; first {bad[0]}" if bad else ""))
    chk("3b. all constructs are 1,309 nt", set(df.core_length) == {1309},
        str(sorted(set(df.core_length))))
    chk("3c. exon_start is constant and consistent",
        df.exon_start.nunique() == 1, f"{sorted(set(df.exon_start))}")
    chk("3d. middle exon is the intended one",
        df.sequence_id.nunique() == 1
        and df.sequence_id.iloc[0] == exon["sequence_id"],
        exon["sequence_id"])
    chk("3e. B2-B8 are the native PNAS blocks",
        all(len(native[b]) > 0 for b in SP.BLOCK_ORDER_8[1:]))

    print("\n4. Splits")
    if mode == "primary":
        s = df.split.value_counts(normalize=True).round(3).to_dict()
        chk("4a. split fractions are ~80/10/10",
            abs(s.get("train", 0) - .8) < .02 and abs(s.get("val", 0) - .1) < .02,
            str(s))
        overlap = False
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            sa = set(df[df.split == a].b1_sha256)
            sb = set(df[df.split == b].b1_sha256)
            overlap |= bool(sa & sb)
        chk("4b. no B1 sequence appears in two splits", not overlap)
        strat = df.groupby("split").alpha.value_counts(normalize=True).unstack()
        chk("4c. every split carries all three alpha groups",
            strat.notna().all().all(), f"\n{strat.round(3)}")
    else:
        prim = pd.read_csv(DATA / "b1_primary_manifest.csv")
        train_val = set(prim[prim.split.isin(["train", "val"])].b1_sha256)
        chk("4a. replication B1s are disjoint from primary train+val",
            not (set(df.b1_sha256) & train_val), "no leakage")
        chk("4b. replication B1s are exactly the primary test split",
            set(df.b1_sha256) == set(prim[prim.split == "test"].b1_sha256),
            f"n={len(df)} (paired with primary test)")

    print("\n5. Quality statistics are recorded, not filtered")
    for c in ("gc", "max_homopolymer", "distinct_3mers", "entropy_bits",
              "max_base_frac"):
        chk(f"5. {c} recorded for every row", c in df.columns and df[c].notna().all())
    return fails, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["primary", "replication"],
                    default="primary")
    ap.add_argument("--n", type=int, default=10_000,
                    help="library size (primary mode only); configurable")
    ap.add_argument("--seed", type=int, default=G.DATASET_SEED)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    print(f"Building B1 dataset -- mode={args.mode} "
          f"{'n=' + str(args.n) if args.mode == 'primary' else ''}")
    print("=" * 72)
    df, report = build(args.mode, args.n, args.seed)
    exon = exon_record(report["exon"])
    print(f"  {len(df)} rows for middle exon {exon['sequence_id']} "
          f"(pre-tuner {exon['PNAS_common']:+.3f})")
    if args.mode == "primary":
        print(f"  alpha counts: {df.alpha.value_counts().sort_index().to_dict()}")
        print(f"  splits:       {df.split.value_counts().to_dict()}")
        print(f"  duplicate redraws: {report['duplicate_redraws']}, "
              f"pilot-panel collisions excluded: {report['excluded_collisions']}")

    fails, _ = validate(df, exon, args.mode)

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} FAILURES: {fails}")
        return 1
    print("ALL CHECKS PASSED")

    flags = [c for c in df.columns if c.startswith("flag_")]
    if flags:
        print("\nAdvisory QC flags (recorded, nothing excluded):")
        for c in flags:
            print(f"  {c}: {int(df[c].sum())} / {len(df)}")

    if args.validate_only:
        print("--validate-only: nothing written")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    name = ("b1_primary_manifest.csv" if args.mode == "primary"
            else "b1_replication_manifest.csv")
    path = DATA / name
    if path.exists():
        print(f"REFUSING to overwrite existing {path}")
        return 1
    df.to_csv(path, index=False)
    meta = {**report, "exon_detail": exon,
            "alpha_mix": G.ALPHA_MIX, "split_fractions": SPLIT_FRACTIONS,
            "split_seed": SPLIT_SEED, "b1_len": G.B1_LEN,
            "n_rows": len(df), "qc_flag_counts":
                {c: int(df[c].sum()) for c in flags}}
    (DATA / name.replace(".csv", "_meta.json")).write_text(
        json.dumps(meta, indent=2, default=str))
    print(f"\nwrote {path}")
    print(f"wrote {DATA / name.replace('.csv', '_meta.json')}")
    print(f"\nAlphaGenome calls this manifest will require: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
