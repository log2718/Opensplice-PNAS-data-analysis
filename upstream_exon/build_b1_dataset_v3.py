"""
Build and validate the v3 (fixed-composition) B1 library: 250,000 sequences.

Design change from v2: every sequence is a random PERMUTATION of the same
multiset -- 29 A, 28 C, 28 G, 28 T. Nucleotide composition is therefore
identical across the whole library, not merely matched in expectation, so any
response variation is attributable to nucleotide ORDER.

    train 200,000 | val 10,000 | test 40,000     (250,000 total)
    pilot          1,000 sequences, drawn from TRAIN

The pilot is a marked subset of the manifest, not a separate library, so its
AlphaGenome labels are written into the same results CSV and count toward the
full job instead of being re-spent.

Writes only into ``outputs/upstream_exon/dataset_v3/``. The v1 and v2
libraries, the pilot panel and the full_v2 model run are never touched, and
v3 is generated from a different seed stream and explicitly excludes every
previously used sequence.

The manifest is ~150 MB and is NOT committed; it is reproducible from the seed
alone. ``manifest_fingerprint.json`` (small, committed) carries a SHA-256 over
the ordered sequences so a manifest rebuilt on Torch can be proven identical
to the one built here.

Makes NO AlphaGenome calls.

Usage::

    python upstream_exon/build_b1_dataset_v3.py --validate-only
    python upstream_exon/build_b1_dataset_v3.py --validate-only -n 5000
    python upstream_exon/build_b1_dataset_v3.py
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

import b1_lib as B  # noqa: E402
import b1_permutation as P  # noqa: E402
import splice_site_partition as SP  # noqa: E402
from build_b1_dataset import PRIMARY_EXON, exon_record  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
DATA_V1 = OUT / "dataset_v1"
DATA_V2 = OUT / "dataset_v2"
DATA = OUT / "dataset_v3"

N_TOTAL = 250_000
SPLIT_SEQUENCES = {"train": 200_000, "val": 10_000, "test": 40_000}
N_PILOT = 1_000
SPLIT_SEED = 20261101
PILOT_SEED = 20261102


def previously_used_sequences() -> set[str]:
    """Every B1 already used by the pilot panel or by the v1 / v2 libraries."""
    used: set[str] = set()
    panel = OUT / "b1_variant_panel.csv"
    if panel.exists():
        used |= set(pd.read_csv(panel, usecols=["b1"]).b1)
    for d in (DATA_V1, DATA_V2):
        man = d / "b1_primary_manifest.csv"
        if man.exists():
            used |= set(pd.read_csv(man, usecols=["b1"]).b1)
    return used


def assign_splits(df: pd.DataFrame, sizes: dict[str, int]) -> pd.DataFrame:
    """Deterministic split assignment over a shuffled index."""
    if sum(sizes.values()) != len(df):
        raise ValueError(f"split sizes {sizes} do not sum to {len(df)}")
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(df))
    split = np.empty(len(df), dtype=object)
    lo = 0
    for name in ("train", "val", "test"):
        hi = lo + sizes[name]
        split[order[lo:hi]] = name
        lo = hi
    df = df.copy()
    df["split"] = split
    return df


def mark_pilot(df: pd.DataFrame, n_pilot: int) -> pd.DataFrame:
    """Flag a deterministic subset of TRAIN as the pilot.

    Taken from train so that (a) the pilot spends no test-set budget and
    (b) its labels are reused by the full run rather than re-called.
    """
    df = df.copy()
    df["pilot"] = False
    train_idx = np.flatnonzero((df.split == "train").to_numpy())
    if n_pilot > len(train_idx):
        raise ValueError(f"pilot {n_pilot} exceeds train split {len(train_idx)}")
    rng = np.random.default_rng(PILOT_SEED)
    chosen = train_idx[rng.permutation(len(train_idx))[:n_pilot]]
    df.iloc[np.sort(chosen), df.columns.get_loc("pilot")] = True
    return df


def scale_sizes(n_total: int) -> tuple[dict[str, int], int]:
    """Split sizes for a reduced ``-n`` rehearsal build, keeping the ratios."""
    if n_total == N_TOTAL:
        return dict(SPLIT_SEQUENCES), N_PILOT
    frac = n_total / N_TOTAL
    sizes = {k: int(round(v * frac)) for k, v in SPLIT_SEQUENCES.items()}
    sizes["train"] += n_total - sum(sizes.values())      # absorb rounding
    return sizes, max(1, int(round(N_PILOT * frac)))


def build(n_total: int) -> tuple[pd.DataFrame, dict, dict, int]:
    exon = exon_record(PRIMARY_EXON)
    used = previously_used_sequences()
    print(f"  excluding {len(used):,} previously used sequences")
    recs, report = P.generate(n_total, exclude=used)
    df = pd.DataFrame(recs)

    sizes, n_pilot = scale_sizes(n_total)
    df = assign_splits(df, sizes)
    df = mark_pilot(df, n_pilot)

    syn70 = exon["synthetic_70mer"]
    cores, starts = [], []
    for s in df.b1:
        core, start = B.build_construct(s, syn70)
        cores.append(B.sha(core))
        starts.append(start)
    df["core_sha256"] = cores
    df["core_length"] = 1309
    df["exon_start"] = starts
    df["exon_len"] = len(exon["full_middle_exon"])
    df["sequence_id"] = exon["sequence_id"]
    df["PNAS_common"] = exon["PNAS_common"]
    df["row_id"] = df.sequence_id + "|" + df.b1_id

    report["exon"] = exon["sequence_id"]
    report["excluded_prior_sequences"] = len(used)
    report["fingerprint_sha256"] = P.library_fingerprint(df.b1)
    report["split_sequences"] = sizes
    report["n_pilot"] = n_pilot
    report["split_seed"] = SPLIT_SEED
    report["pilot_seed"] = PILOT_SEED
    return df, report, exon, n_pilot


def validate(df: pd.DataFrame, exon: dict, sizes: dict[str, int],
             n_pilot: int) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    lines: list[str] = []

    def chk(name, ok, detail=""):
        line = f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else "")
        print(line, flush=True)
        lines.append(line)
        if not ok:
            fails.append(name)

    n = len(df)
    native = SP.pnas_blocks8()

    print("\n1. Library size and uniqueness")
    chk("1a. requested number of sequences", n == sum(sizes.values()),
        f"n={n:,}")
    chk("1b. all sequences unique", df.b1.nunique() == n,
        f"{df.b1.nunique():,} distinct")
    chk("1c. all digests unique", df.b1_sha256.nunique() == n)
    chk("1d. all row_ids unique", df.row_id.nunique() == n)

    print("\n2. EXACT composition (the whole point of v3)")
    cols = [f"count_{b}" for b in P.ALPHABET]
    recorded = df[cols].drop_duplicates()
    chk("2a. one recorded composition for the entire library",
        len(recorded) == 1, str(recorded.to_dict("records")))
    chk("2b. recorded composition is 29 A / 28 C / 28 G / 28 T",
        recorded.iloc[0].to_dict() == {f"count_{b}": v
                                       for b, v in P.TARGET_COUNTS.items()})
    # recount from the sequences themselves -- the recorded columns are
    # constants, so they would agree with themselves even if generation broke
    sample = df.b1 if n <= 20_000 else df.b1.sample(20_000, random_state=0)
    recount = {b: sample.str.count(b).unique().tolist() for b in P.ALPHABET}
    chk("2c. recounted from the sequences, every base count is the target",
        all(recount[b] == [P.TARGET_COUNTS[b]] for b in P.ALPHABET),
        f"{len(sample):,} sequences recounted: {recount}")
    chk("2d. GC is a single value", df.gc.nunique() == 1,
        f"GC = {df.gc.iloc[0]:.6f} = 56/113")

    print("\n3. Sequences")
    chk("3a. every B1 is exactly 113 nt", (df.length == P.B1_LEN).all()
        and bool((sample.str.len() == P.B1_LEN).all()))
    chk("3b. alphabet is ACGT only",
        all(not (set(s) - set(P.ALPHABET)) for s in sample.head(2000)))
    chk("3c. digests match the sequences",
        all(P.sha(r.b1) == r.b1_sha256 for r in df.head(2000).itertuples()))

    print("\n4. Order statistics vary (there is something to learn)")
    chk("4a. max homopolymer varies", df.max_homopolymer.nunique() > 1,
        f"range {df.max_homopolymer.min()}-{df.max_homopolymer.max()}, "
        f"mean {df.max_homopolymer.mean():.2f}")
    chk("4b. distinct 3-mer count varies", df.distinct_3mers.nunique() > 1,
        f"range {df.distinct_3mers.min()}-{df.distinct_3mers.max()}")
    chk("4c. entropy is constant (composition is fixed)",
        df.entropy_bits.nunique() == 1, f"{df.entropy_bits.iloc[0]:.6f} bits")

    print("\n5. Disjoint from previous work")
    prior = previously_used_sequences()
    chk("5a. no overlap with the pilot panel or the v1 / v2 libraries",
        not (set(df.b1) & prior), f"{len(prior):,} prior sequences checked")
    for name, d in (("v1", DATA_V1), ("v2", DATA_V2)):
        man = d / "b1_primary_manifest.csv"
        chk(f"5b. the {name} manifest still exists and is untouched",
            man.exists(), str(man.relative_to(ROOT)) if man.exists() else "MISSING")
    v2res = DATA_V2 / "b1_primary_ag_results.csv"
    n_v2 = sum(1 for _ in v2res.open()) - 1 if v2res.exists() else 0
    chk("5c. the 40,000 v2 labels are preserved", n_v2 == 40_000,
        f"{n_v2:,} rows in dataset_v2/b1_primary_ag_results.csv")

    print("\n6. Splits")
    got = df.split.value_counts().to_dict()
    chk("6a. split sizes are as designed", got == sizes, str(got))
    overlap = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap |= bool(set(df[df.split == a].b1_sha256)
                        & set(df[df.split == b].b1_sha256))
    chk("6b. no sequence appears in two splits", not overlap)
    chk("6c. every row has a split", df.split.notna().all()
        and (df.split != "").all())

    print("\n7. Pilot subset")
    chk("7a. pilot has the requested size", int(df.pilot.sum()) == n_pilot,
        f"{int(df.pilot.sum()):,}")
    chk("7b. every pilot sequence is in TRAIN",
        bool((df[df.pilot].split == "train").all()),
        str(df[df.pilot].split.value_counts().to_dict()))

    print("\n8. Constructs")
    syn70 = exon["synthetic_70mer"]
    bad = []
    for r in df.sample(min(500, n), random_state=0).itertuples():
        core, start = B.build_construct(r.b1, syn70)
        f = B.validate_construct(core, start, r.b1, syn70)
        if P.sha(core) != r.core_sha256 or start != r.exon_start:
            f.append("manifest digest/coordinate mismatch")
        if f:
            bad.append((r.b1_id, f))
    chk("8a. 500 sampled constructs pass every structural assertion", not bad,
        f"{len(bad)} bad" + (f"; first {bad[0]}" if bad else ""))
    chk("8b. all constructs are 1,309 nt", set(df.core_length) == {1309})
    chk("8c. exon_start is constant", df.exon_start.nunique() == 1,
        str(sorted(set(df.exon_start))))
    chk("8d. middle exon is the primary one",
        df.sequence_id.unique().tolist() == [PRIMARY_EXON], PRIMARY_EXON)
    chk("8e. B2-B8 are the native PNAS blocks, present and non-empty",
        all(len(native[b]) > 0 for b in SP.BLOCK_ORDER_8[1:]))

    print("\n9. Determinism")
    again, _ = P.generate(500, exclude=previously_used_sequences(),
                          progress_every=0)
    chk("9a. regenerating from the seed reproduces the same sequences",
        [r["b1"] for r in again] == df.b1.head(500).tolist())
    return fails, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("-n", "--n-total", type=int, default=N_TOTAL,
                    help="build a smaller rehearsal library (splits are "
                         "scaled); the default is the real 250,000")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing v3 manifest")
    args = ap.parse_args(argv)

    print(f"Building the v3 fixed-composition B1 library ({args.n_total:,} sequences)")
    print("=" * 72)
    df, report, exon, n_pilot = build(args.n_total)
    sizes = report["split_sequences"]
    print(f"  {len(df):,} sequences, middle exon {exon['sequence_id']}")
    print(f"  composition: " + "  ".join(
        f"{b}={P.TARGET_COUNTS[b]}" for b in P.ALPHABET)
        + f"   GC={df.gc.iloc[0]:.4f}")
    print(f"  split:       {df.split.value_counts().to_dict()}")
    print(f"  pilot:       {int(df.pilot.sum()):,} (from train)")
    print(f"  redraws:     {report['duplicate_redraws']} duplicate, "
          f"{report['excluded_collisions']} collided with prior sequences")
    print(f"  fingerprint: {report['fingerprint_sha256']}")

    fails, lines = validate(df, exon, sizes, n_pilot)

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} FAILURES: {fails}")
        return 1
    print("ALL CHECKS PASSED")

    if args.validate_only:
        print("--validate-only: nothing written")
        return 0
    if args.n_total != N_TOTAL:
        print(f"REFUSING to write a partial library (-n {args.n_total:,}); "
              "use --validate-only for rehearsals")
        return 1

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "b1_primary_manifest.csv"
    if path.exists() and not args.force:
        print(f"REFUSING to overwrite existing {path} (pass --force)")
        return 1
    df.to_csv(path, index=False)
    meta = {**report, "exon_detail": exon}
    (DATA / "b1_primary_manifest_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))
    (DATA / "manifest_fingerprint.json").write_text(json.dumps({
        "design": "fixed_composition_permutation",
        "n_total": len(df), "seed": P.PERM_SEED,
        "split_seed": SPLIT_SEED, "pilot_seed": PILOT_SEED,
        "target_counts": dict(P.TARGET_COUNTS),
        "split_sequences": sizes, "n_pilot": n_pilot,
        "fingerprint_sha256": report["fingerprint_sha256"],
        "manifest_rows": len(df),
    }, indent=2))
    (DATA / "b1_primary_validation_report.txt").write_text(
        "v3 fixed-composition B1 library validation\n" + "=" * 72 + "\n"
        + "\n".join(lines) + "\nALL CHECKS PASSED\n")
    print(f"\nwrote {path}  ({path.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {DATA / 'b1_primary_manifest_meta.json'}")
    print(f"wrote {DATA / 'manifest_fingerprint.json'}")
    print(f"wrote {DATA / 'b1_primary_validation_report.txt'}")
    print(f"\nAlphaGenome calls this manifest will require: {len(df):,} "
          f"({n_pilot:,} of them in the pilot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
