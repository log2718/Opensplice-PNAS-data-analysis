"""
JOB 2b (scoring step) -- score one deterministic shard of the 8-block manifest.

Reads ``chimera8_job_manifest.csv`` (built by ``build_chimera8_manifest.py``),
takes a contiguous shard of it, rebuilds each construct from
``splice_site_partition``, VERIFIES the construct against the manifest's
SHA-256 before spending a call, and scores it with AlphaGenome SPLICE_SITES.

One AlphaGenome stream. Shards are contiguous and disjoint, so running them
back to back (``--array=0-11%1``) never produces two concurrent streams.

Resume safety
-------------
Each shard owns exactly one output file, ``chimera8_ag_results_shard{NN}.csv``.
``synthetic_context_lib.run_scoring`` keys on ``row_id`` and skips any row
already present in that file, so a requeued shard re-scores nothing. Shards
never write to each other's file and never write to the combined file.

Usage::

    python analysis_script/run_chimera8_experiment.py --shard 0 --n-shards 12
    python analysis_script/run_chimera8_experiment.py --shard 0 --n-shards 12 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import synthetic_context_lib as L  # noqa: E402
import splice_site_partition as SP  # noqa: E402

NEW = ROOT / "outputs" / "synthetic_context_8block"
LIB_CSV = ROOT / "outputs" / "synthetic_context" / "synthetic_exon_library.csv"
MANIFEST_CSV = NEW / "chimera8_job_manifest.csv"
SHARD_DIR = NEW / "shards"


def shard_bounds(n_rows: int, shard: int, n_shards: int) -> tuple[int, int]:
    """Contiguous, disjoint, deterministic; earlier shards take the remainder."""
    if not 0 <= shard < n_shards:
        raise SystemExit(f"--shard must be in [0, {n_shards})")
    base, rem = divmod(n_rows, n_shards)
    lo = shard * base + min(shard, rem)
    hi = lo + base + (1 if shard < rem else 0)
    return lo, hi


def build_shard_jobs(manifest: pd.DataFrame, lib: pd.DataFrame) -> list[dict]:
    """Rebuild each construct and check it against the manifest digest."""
    syn = lib.set_index("sequence_id").synthetic_70mer
    pb = SP.pnas_blocks8()
    fbs = {a: SP.fas_blocks8(*L.FAS_TARGETS[a]) for a in L.FAS_TARGETS}

    jobs, mismatches = [], []
    for r in manifest.itertuples():
        core, start = SP.assemble8(r.canon_label, syn[r.sequence_id], pb,
                                   fbs[r.canon_comparison])
        digest = hashlib.sha256(core.encode()).hexdigest()
        if digest != r.core_sha256 or start != r.exon_start \
                or len(core) != r.core_length:
            mismatches.append(r.job_row_id)
            continue
        jobs.append({
            "row_id": r.job_row_id,
            "job_index": r.job_index,
            "sequence_id": r.sequence_id,
            "canon_comparison": r.canon_comparison,
            "canon_label": r.canon_label,
            "core_sha256": r.core_sha256,
            "core": core,
            "exon_start": start,
            "exon_len": r.exon_len,
        })
    if mismatches:
        raise SystemExit(
            f"[abort] {len(mismatches)} constructs do not match the manifest "
            f"digest, e.g. {mismatches[:3]}. The manifest and the partition "
            "code are out of sync -- rebuild the manifest before scoring.")
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=12)
    ap.add_argument("--env-path", default=".env.alphagenome")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--call-timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="verify every construct; make NO AlphaGenome calls")
    args = ap.parse_args(argv)

    if not MANIFEST_CSV.exists():
        print(f"[missing] {MANIFEST_CSV} -- run build_chimera8_manifest.py first",
              file=sys.stderr)
        return 1

    manifest = pd.read_csv(MANIFEST_CSV)
    lo, hi = shard_bounds(len(manifest), args.shard, args.n_shards)
    part = manifest.iloc[lo:hi]
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SHARD_DIR / f"chimera8_ag_results_shard{args.shard:02d}.csv"
    failed_csv = SHARD_DIR / f"chimera8_failed_shard{args.shard:02d}.csv"

    print(f"[shard] {args.shard + 1}/{args.n_shards}  rows [{lo:,}, {hi:,}) "
          f"= {len(part):,} of {len(manifest):,} calls")
    print(f"[shard] output -> {out_csv.relative_to(ROOT)}")

    lib = pd.read_csv(LIB_CSV)
    t0 = time.time()
    jobs = build_shard_jobs(part, lib)
    print(f"[verify] {len(jobs):,} constructs rebuilt and SHA-256 matched "
          f"in {time.time() - t0:.1f}s")

    done = 0
    if out_csv.exists():
        done = len(pd.read_csv(out_csv))
        print(f"[resume] {done:,} rows already scored in this shard; "
              f"{len(jobs) - done:,} remain")

    if args.dry_run:
        for r in (0.38, 0.51, 0.60):
            print(f"[dry-run] {r:.2f} s/call -> "
                  f"{(len(jobs) - done) * r / 3600:.2f} h for this shard")
        print("[dry-run] stopping before AlphaGenome. No call was made.")
        return 0

    L.run_scoring(jobs, out_csv, args.env_path,
                  flush_every=args.flush_every,
                  call_timeout=args.call_timeout,
                  max_retries=args.max_retries,
                  failed_csv=failed_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
