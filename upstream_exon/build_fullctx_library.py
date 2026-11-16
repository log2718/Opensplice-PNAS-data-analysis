#!/usr/bin/env python
"""Build the two full-context permutation libraries. Makes NO AlphaGenome calls.

    upstream   1,000,000 permutations of the 113-nt upstream exon body (B1),
               middle exon held at the primary synthetic exon
    middle       500,000 permutations of the 70-nt middle exon body,
               B1 held at the native PNAS sequence   (the sanity check)

Both libraries are EXACT-composition: every sequence is a permutation of one
fixed multiset, so composition cannot vary and any response variation is
attributable to nucleotide ORDER. This is the v3 design, carried over.

  B1,     113 nt:  29 A, 28 C, 28 G, 28 T
      113 is not divisible by 4, so one base takes the extra copy. A takes it,
      leaving G = C = 28 exactly, so GC content is one number for the whole
      library and there is no G/C asymmetry for a model to read.

  exon,    70 nt:  18 A, 18 T, 17 C, 17 G
      70 leaves a remainder of 2. Giving the two extras to A and T keeps
      G = C = 17 AND A = T = 18, so the library is balanced on both the GC
      axis and the purine/pyrimidine axis at once.

Why the middle-exon library is the sanity check: B1 sits 130 nt upstream of
the middle exon's acceptor and is not part of the exon being measured, so its
effect on the exon's splice-site usage is the surprising result. Permuting the
exon body itself is the positive control -- sequence inside the exon, between
the two sites being read, SHOULD move the readout. If it does not move it at
least as much as B1 does, the upstream result is not what it appears to be.

Splits are 80 / 4 / 16 percent, matching v3's 200k / 10k / 40k exactly.

The manifests are large (~120 MB and ~45 MB) and are NOT committed; they are
reproducible from the seeds alone, and ``manifest_fingerprint.json`` carries a
SHA-256 over the ordered sequences so a manifest rebuilt on Torch can be
proven identical to one built here.

    python upstream_exon/build_fullctx_library.py --library upstream --validate-only
    python upstream_exon/build_fullctx_library.py --library upstream
    python upstream_exon/build_fullctx_library.py --library middle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fullctx_lib as F  # noqa: E402
import splice_site_partition as SP  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"

LIBRARIES = {
    "upstream": {
        "slot": "b1",
        "length": F.B1_LEN,
        "counts": {"A": 29, "C": 28, "G": 28, "T": 28},
        "n_total": 1_000_000,
        "splits": {"train": 800_000, "val": 40_000, "test": 160_000},
        "seed": 20261115,
        "split_seed": 20261116,
        "dirname": "fullctx_upstream_1m",
        "prefix": "u",
    },
    "middle": {
        "slot": "syn70",
        "length": F.SYN_LEN,
        "counts": {"A": 18, "T": 18, "C": 17, "G": 17},
        "n_total": 500_000,
        "splits": {"train": 400_000, "val": 20_000, "test": 80_000},
        "seed": 20261117,
        "split_seed": 20261118,
        "dirname": "fullctx_middle_500k",
        "prefix": "m",
    },
}

CHUNK = 50_000


def pool_array(counts: dict[str, int], length: int) -> np.ndarray:
    if sum(counts.values()) != length:
        raise ValueError(f"counts sum to {sum(counts.values())}, not {length}")
    return np.frombuffer(
        "".join(b * n for b, n in sorted(counts.items())).encode("ascii"),
        dtype="S1")


def max_homopolymer(seq: str) -> int:
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def assign_splits(n: int, splits: dict[str, int], seed: int) -> np.ndarray:
    """A shuffled split label per row, with exact per-split counts."""
    if sum(splits.values()) != n:
        raise ValueError(f"splits sum to {sum(splits.values())}, not {n}")
    labels = np.concatenate([np.full(c, k, dtype=object)
                             for k, c in splits.items()])
    np.random.default_rng(seed).shuffle(labels)
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", choices=sorted(LIBRARIES), required=True)
    ap.add_argument("--validate-only", action="store_true",
                    help="generate -n sequences, check them, write nothing")
    ap.add_argument("-n", type=int, default=5_000,
                    help="sample size for --validate-only")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = LIBRARIES[args.library]
    dest = args.out_dir or (OUT / cfg["dirname"])
    n_total = args.n if args.validate_only else cfg["n_total"]
    length, slot = cfg["length"], cfg["slot"]

    exons = pd.read_csv(OUT / "b1_pilot_exons.csv")
    syn_primary = dict(zip(exons.sequence_id, exons.synthetic_70mer))[F.PRIMARY_EXON]
    native_b1 = SP.pnas_blocks8()["B1"]

    print("=" * 72)
    print(f"FULL-CONTEXT LIBRARY: {args.library}")
    print("=" * 72)
    print(f"  variable slot      {slot}  ({length} nt)")
    print(f"  fixed composition  {cfg['counts']}")
    print(f"  sequences          {n_total:,}"
          + ("   [VALIDATE-ONLY SAMPLE]" if args.validate_only else ""))
    if not args.validate_only:
        print(f"  splits             " + "  ".join(
            f"{k} {v:,}" for k, v in cfg["splits"].items()))
    print(f"  held fixed         "
          + (f"middle exon = {F.PRIMARY_EXON}" if slot == "b1"
             else "B1 = native PNAS"))
    print(f"  context            {F.CONTEXT_LEN:,} nt, "
          f"acceptor {F.GEOMETRY['acceptor']}, donor {F.GEOMETRY['donor']}")
    print(f"  target             logit_average = "
          f"logit((usage_acc + usage_don) / 2)   [SPLICE_SITE_USAGE]")
    print()

    pool = pool_array(cfg["counts"], length)
    rng = np.random.default_rng(cfg["seed"])
    digest = hashlib.sha256()
    seen: set[str] = set()
    n_dupes = 0

    manifest = dest / "manifest.csv"
    if not args.validate_only:
        # Only now, once we know something will actually be written: a
        # --validate-only run must leave no trace on disk.
        if manifest.exists():
            sys.exit(f"FATAL: {manifest} already exists. Delete it to rebuild; "
                     f"overwriting a manifest would orphan any labels keyed "
                     f"to it.")
        dest.mkdir(parents=True, exist_ok=True)

    splits = (assign_splits(n_total, cfg["splits"], cfg["split_seed"])
              if not args.validate_only else np.full(n_total, "train",
                                                     dtype=object))

    t0 = time.time()
    written = 0
    first = True
    for lo in range(0, n_total, CHUNK):
        hi = min(lo + CHUNK, n_total)
        recs = []
        for i in range(lo, hi):
            seq = rng.permutation(pool).tobytes().decode("ascii")
            while seq in seen:                 # ~1e66 possibilities; defensive
                n_dupes += 1
                seq = rng.permutation(pool).tobytes().decode("ascii")
            seen.add(seq)
            digest.update(seq.encode("ascii"))
            digest.update(b"\n")
            recs.append({
                "row_id": f"{cfg['prefix']}{i:07d}",
                "draw_index": i,
                slot: seq,
                "split": splits[i],
                "max_homopolymer": max_homopolymer(seq),
                "distinct_6mers": len({seq[j:j + 6]
                                       for j in range(len(seq) - 5)}),
            })
        if not args.validate_only:
            pd.DataFrame(recs).to_csv(manifest, mode="a", index=False,
                                      header=first)
            first = False
        written += len(recs)
        if written % 200_000 == 0 or hi == n_total:
            print(f"  {written:,}/{n_total:,}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- structural validation on a sample of assembled constructs --------
    print("\n  validating assembled constructs ...")
    check_idx = sorted(np.random.default_rng(7).choice(
        len(recs), size=min(25, len(recs)), replace=False))
    for j in check_idx:
        r = recs[j]
        b1 = r[slot] if slot == "b1" else native_b1
        s70 = r[slot] if slot == "syn70" else syn_primary
        full = F.build_construct(b1, s70)
        fails = F.validate_construct(full, b1, s70)
        if fails:
            sys.exit(f"FATAL {r['row_id']}: {fails}")
        if len(set(r[slot])) != 4:
            sys.exit(f"FATAL {r['row_id']}: slot is not over all four bases")
        for base, want in cfg["counts"].items():
            if r[slot].count(base) != want:
                sys.exit(f"FATAL {r['row_id']}: {base} count "
                         f"{r[slot].count(base)} != {want}")
    print(f"  {len(check_idx)} constructs assembled and validated  [PASS]")
    print(f"  duplicate redraws: {n_dupes}")

    if args.validate_only:
        print("\n--validate-only: nothing written")
        return 0

    fp = {
        "library": args.library,
        "slot": slot,
        "length": length,
        "counts": cfg["counts"],
        "n_total": n_total,
        "splits": cfg["splits"],
        "seed": cfg["seed"],
        "split_seed": cfg["split_seed"],
        "sequence_sha256": digest.hexdigest(),
        "context_file": F.CONTEXT_TXT.name,
        "context_sha256": hashlib.sha256(
            F.load_context().encode("ascii")).hexdigest(),
        "context_len": F.CONTEXT_LEN,
        "geometry": F.GEOMETRY,
        "fixed_middle_exon": F.PRIMARY_EXON if slot == "b1" else None,
        "fixed_b1": "native_pnas" if slot == "syn70" else None,
        "target": "logit_average = logit((usage_acceptor + usage_donor)/2)",
        "output_type": "SPLICE_SITE_USAGE",
        "barcode": "left as N (20 nt)",
    }
    (dest / "manifest_fingerprint.json").write_text(json.dumps(fp, indent=2) + "\n")

    print(f"\nwrote {manifest}  ({manifest.stat().st_size / 1e6:.0f} MB, "
          f"{written:,} rows)")
    print(f"wrote {dest / 'manifest_fingerprint.json'}")
    print(f"sequence sha256 {fp['sequence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
