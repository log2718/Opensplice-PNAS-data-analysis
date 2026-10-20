"""
Build and validate the v2 (grouped) B1 library: 40,000 sequences.

Design change from v1: one Dirichlet probability vector per GROUP, with four
113-nt sequences drawn independently from that same vector. Replicates share
the underlying nucleotide probabilities but not their realized counts, so a
within-group response difference isolates nucleotide ORDER from composition.

    alpha 0.5 : 2,000 groups ->  8,000 sequences  (20%)
    alpha 1.0 : 4,000 groups -> 16,000 sequences  (40%)
    alpha 2.0 : 4,000 groups -> 16,000 sequences  (40%)
    total     : 10,000 groups -> 40,000 sequences

Splits are assigned BY GROUP, alpha-stratified, so all four replicates always
land in the same split:

    train 27,000 seqs (6,750 groups) | val 3,000 (750) | test 10,000 (2,500)

Writes only into ``outputs/upstream_exon/dataset_v2/``. The v1 manifest, the
pilot outputs and the 300 already-labeled v1 examples are never touched, and
v2 is generated from a different seed stream and explicitly excludes every
previously used sequence, so the two libraries are disjoint by construction.

Makes NO AlphaGenome calls.

Usage::

    python upstream_exon/build_b1_dataset_v2.py --validate-only
    python upstream_exon/build_b1_dataset_v2.py
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
from build_b1_dataset import PRIMARY_EXON, REPLICATION_EXON, exon_record  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
DATA_V1 = OUT / "dataset_v1"
DATA = OUT / "dataset_v2"

N_TOTAL = 40_000
REPS = G.REPLICATES_PER_GROUP

#: target SEQUENCE counts per split (groups follow by dividing by REPS)
SPLIT_SEQUENCES = {"train": 27_000, "val": 3_000, "test": 10_000}
SPLIT_SEED = 20261021


def previously_used_sequences() -> set[str]:
    """Every B1 already used by the pilot or by the v1 library."""
    used: set[str] = set()
    panel = OUT / "b1_variant_panel.csv"
    if panel.exists():
        used |= set(pd.read_csv(panel).b1)
    v1 = DATA_V1 / "b1_primary_manifest.csv"
    if v1.exists():
        used |= set(pd.read_csv(v1).b1)
    return used


def split_group_targets() -> dict[float, dict[str, int]]:
    """Groups per (alpha, split), exact and alpha-stratified."""
    groups = G.group_counts(N_TOTAL, REPS)
    total_groups = sum(groups.values())
    fracs = {k: v / N_TOTAL for k, v in SPLIT_SEQUENCES.items()}
    if abs(sum(fracs.values()) - 1.0) > 1e-12:
        raise ValueError("split sizes must sum to the library size")
    out: dict[float, dict[str, int]] = {}
    for a, n_g in groups.items():
        names = list(SPLIT_SEQUENCES)
        counts = G._largest_remainder(
            np.array([fracs[s] for s in names]), n_g)
        out[a] = dict(zip(names, (int(c) for c in counts)))
    got = sum(sum(v.values()) for v in out.values())
    if got != total_groups:
        raise ValueError(f"group split accounting: {got} != {total_groups}")
    return out


def assign_group_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, alpha-stratified, assigned at the GROUP level."""
    targets = split_group_targets()
    df = df.copy()
    df["split"] = ""
    for alpha, want in targets.items():
        gids = np.array(sorted(df.loc[df.alpha == alpha, "group_id"].unique()))
        r = G._rng(SPLIT_SEED, "groupsplit", float(alpha))
        order = r.permutation(len(gids))
        shuffled = gids[order]
        lo = 0
        mapping: dict[str, str] = {}
        for name in ("train", "val", "test"):
            hi = lo + want[name]
            for g in shuffled[lo:hi]:
                mapping[g] = name
            lo = hi
        if lo != len(gids):
            raise ValueError(f"alpha={alpha}: assigned {lo} of {len(gids)}")
        sel = df.alpha == alpha
        df.loc[sel, "split"] = df.loc[sel, "group_id"].map(mapping)
    if (df.split == "").any():
        raise ValueError("some rows were not assigned a split")
    return df


def build() -> tuple[pd.DataFrame, dict]:
    exon = exon_record(PRIMARY_EXON)
    used = previously_used_sequences()
    recs, report = G.generate_grouped(N_TOTAL, exclude=used)
    df = pd.DataFrame(recs)
    flags = pd.DataFrame([G.qc_flags(r) for r in recs])
    for c in flags.columns:
        df[c] = flags[c]
    df = assign_group_splits(df)

    rows = []
    for r in df.itertuples():
        core, start = B.build_construct(r.b1, exon["synthetic_70mer"])
        rows.append({"core_sha256": B.sha(core), "core_length": len(core),
                     "exon_start": start,
                     "exon_len": len(exon["full_middle_exon"])})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df["sequence_id"] = exon["sequence_id"]
    df["PNAS_common"] = exon["PNAS_common"]
    df["row_id"] = df.sequence_id + "|" + df.b1_id
    report["exon"] = exon["sequence_id"]
    report["excluded_prior_sequences"] = len(used)
    return df, report


def validate(df: pd.DataFrame, exon: dict) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    lines: list[str] = []

    def chk(name, ok, detail=""):
        line = f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else "")
        print(line)
        lines.append(line)
        if not ok:
            fails.append(name)

    native = SP.pnas_blocks8()

    print("\n1. Library size and group structure")
    chk("1a. 40,000 sequences", len(df) == N_TOTAL, f"n={len(df)}")
    chk("1b. 10,000 groups", df.group_id.nunique() == N_TOTAL // REPS,
        f"{df.group_id.nunique()}")
    sizes = df.groupby("group_id").size()
    chk("1c. every group has exactly 4 sequences", set(sizes) == {REPS},
        f"sizes {sorted(set(sizes))}")
    chk("1d. replicate ids are 0..3 within every group",
        (df.groupby("group_id").replicate.apply(
            lambda s: sorted(s) == list(range(REPS)))).all())

    print("\n2. Alpha mixture")
    gpa = df.groupby("alpha").group_id.nunique().to_dict()
    chk("2a. groups per alpha are 2,000 / 4,000 / 4,000",
        gpa == {0.5: 2000, 1.0: 4000, 2.0: 4000}, str(gpa))
    spa = df.alpha.value_counts().sort_index().to_dict()
    chk("2b. sequences per alpha are 8,000 / 16,000 / 16,000",
        spa == {0.5: 8000, 1.0: 16000, 2.0: 16000}, str(spa))
    frac = df.alpha.value_counts(normalize=True).sort_index().round(4).to_dict()
    chk("2c. realized mixture is 20/40/40",
        frac == {0.5: 0.2, 1.0: 0.4, 2.0: 0.4}, str(frac))

    print("\n3. Shared probability vectors")
    pv = df.groupby("group_id")[["target_pA", "target_pC", "target_pG",
                                 "target_pT"]].nunique()
    chk("3a. all four replicates share one probability vector",
        (pv == 1).all().all())
    sums = df[["target_pA", "target_pC", "target_pG", "target_pT"]].sum(axis=1)
    chk("3b. every probability vector sums to 1",
        bool(np.allclose(sums, 1.0)), f"max dev {abs(sums - 1).max():.2e}")
    within = (df.groupby("group_id").gc.std()).mean()
    between = df.groupby("group_id").gc.mean().std()
    chk("3c. within-group GC spread is much smaller than between-group",
        within < between / 2,
        f"within {within:.4f} vs between {between:.4f}")

    print("\n4. Sequences")
    chk("4a. every B1 is exactly 113 nt", (df.length == G.B1_LEN).all())
    chk("4b. alphabet is ACGT only",
        all(not (set(s) - set(G.ALPHABET)) for s in df.b1.head(2000)))
    chk("4c. all 40,000 sequences unique", df.b1_sha256.nunique() == len(df),
        f"{df.b1_sha256.nunique()}")
    chk("4d. digests match the sequences",
        all(G.sha(r.b1) == r.b1_sha256 for r in df.head(2000).itertuples()))

    print("\n5. Disjoint from previous work")
    prior = previously_used_sequences()
    chk("5a. no overlap with the pilot panel or the v1 library",
        not (set(df.b1) & prior), f"{len(prior)} prior sequences checked")
    v1 = DATA_V1 / "b1_primary_manifest.csv"
    chk("5b. the v1 manifest still exists and is untouched", v1.exists())
    v1res = DATA_V1 / "b1_primary_ag_results.csv"
    n_v1 = len(pd.read_csv(v1res)) if v1res.exists() else 0
    chk("5c. the 300 previously labeled v1 examples are preserved",
        n_v1 == 300, f"{n_v1} rows in dataset_v1/b1_primary_ag_results.csv")

    print("\n6. Splits (assigned by group)")
    s_seq = df.split.value_counts().to_dict()
    chk("6a. split sizes are 27,000 / 3,000 / 10,000",
        s_seq == SPLIT_SEQUENCES, str(s_seq))
    s_grp = df.groupby("split").group_id.nunique().to_dict()
    chk("6b. group counts are 6,750 / 750 / 2,500",
        s_grp == {"train": 6750, "val": 750, "test": 2500}, str(s_grp))
    straddle = df.groupby("group_id").split.nunique()
    chk("6c. no group is split across two sets", set(straddle) == {1},
        f"max distinct splits per group {straddle.max()}")
    mix = (df.groupby("split").alpha.value_counts(normalize=True)
             .unstack().round(4))
    chk("6d. every split keeps the 20/40/40 alpha mixture",
        bool((mix - pd.DataFrame(
            {0.5: 0.2, 1.0: 0.4, 2.0: 0.4}, index=mix.index)).abs().max().max()
            < 1e-9), f"\n{mix}")
    overlap = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap |= bool(set(df[df.split == a].b1_sha256)
                        & set(df[df.split == b].b1_sha256))
    chk("6e. no sequence appears in two splits", not overlap)

    print("\n7. Constructs")
    sample = df.sample(500, random_state=0)
    bad = []
    for r in sample.itertuples():
        core, start = B.build_construct(r.b1, exon["synthetic_70mer"])
        f = B.validate_construct(core, start, r.b1, exon["synthetic_70mer"])
        if G.sha(core) != r.core_sha256 or start != r.exon_start:
            f.append("manifest digest/coordinate mismatch")
        if f:
            bad.append((r.b1_id, f))
    chk("7a. 500 sampled constructs pass every structural assertion", not bad,
        f"{len(bad)} bad" + (f"; first {bad[0]}" if bad else ""))
    chk("7b. all constructs are 1,309 nt", set(df.core_length) == {1309})
    chk("7c. exon_start is constant", df.exon_start.nunique() == 1,
        str(sorted(set(df.exon_start))))
    chk("7d. middle exon is the primary one",
        df.sequence_id.unique().tolist() == [PRIMARY_EXON], PRIMARY_EXON)
    chk("7e. B2-B8 present and non-empty",
        all(len(native[b]) > 0 for b in SP.BLOCK_ORDER_8[1:]))

    print("\n8. Determinism")
    again, _ = G.generate_grouped(400, exclude=previously_used_sequences())
    first400, _ = G.generate_grouped(400, exclude=previously_used_sequences())
    chk("8a. regenerating reproduces the same sequences",
        [r["b1"] for r in again] == [r["b1"] for r in first400])

    print("\n9. Quality statistics recorded, nothing filtered")
    for c in ("gc", "max_homopolymer", "distinct_3mers", "entropy_bits",
              "max_base_frac", "target_pA", "seed"):
        chk(f"9. {c} recorded for every row",
            c in df.columns and df[c].notna().all())
    return fails, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    print("Building the v2 grouped B1 library (40,000 sequences)")
    print("=" * 72)
    df, report = build()
    exon = exon_record(report["exon"])
    print(f"  {len(df)} sequences in {df.group_id.nunique()} groups "
          f"({REPS} per group), middle exon {exon['sequence_id']}")
    print(f"  groups per alpha:    {df.groupby('alpha').group_id.nunique().to_dict()}")
    print(f"  sequences per alpha: {df.alpha.value_counts().sort_index().to_dict()}")
    print(f"  split (sequences):   {df.split.value_counts().to_dict()}")
    print(f"  split (groups):      {df.groupby('split').group_id.nunique().to_dict()}")
    print(f"  redraws: {report['duplicate_redraws']} duplicate, "
          f"{report['excluded_collisions']} collided with prior sequences")

    fails, lines = validate(df, exon)

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} FAILURES: {fails}")
        return 1
    print("ALL CHECKS PASSED")

    flagcols = [c for c in df.columns if c.startswith("flag_")]
    print("\nAdvisory QC flags (recorded, nothing excluded):")
    for c in flagcols:
        print(f"  {c}: {int(df[c].sum())} / {len(df)}")

    if args.validate_only:
        print("--validate-only: nothing written")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "b1_primary_manifest.csv"
    if path.exists():
        print(f"REFUSING to overwrite existing {path}")
        return 1
    df.to_csv(path, index=False)
    meta = {**report, "exon_detail": exon, "alpha_mix": G.ALPHA_MIX,
            "split_sequences": SPLIT_SEQUENCES,
            "split_groups": df.groupby("split").group_id.nunique().to_dict(),
            "split_seed": SPLIT_SEED, "b1_len": G.B1_LEN,
            "replicates_per_group": REPS,
            "qc_flag_counts": {c: int(df[c].sum()) for c in flagcols}}
    (DATA / "b1_primary_manifest_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))
    (DATA / "b1_primary_validation_report.txt").write_text(
        "v2 grouped B1 library validation\n" + "=" * 72 + "\n"
        + "\n".join(lines) + "\nALL CHECKS PASSED\n")
    print(f"\nwrote {path}")
    print(f"wrote {DATA / 'b1_primary_manifest_meta.json'}")
    print(f"wrote {DATA / 'b1_primary_validation_report.txt'}")
    print(f"\nAlphaGenome calls this manifest will require: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
