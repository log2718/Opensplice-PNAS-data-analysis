#!/usr/bin/env python
"""Label a full-context library with AlphaGenome SPLICE_SITE_USAGE.

Target: ``logit_average = logit((usage_acceptor + usage_donor) / 2)``.

Resumable on ``row_id`` and safe to run as a serialized SLURM array: each
shard takes the next ``--max-new-calls`` unlabeled rows, appends them, and
exits. A shard that is killed loses at most ``--flush-every`` calls.

Only ONE shard may run at a time. The AlphaGenome quota is not known to
permit concurrent streams, and two scorers appending to the same CSV would
interleave rows mid-line. A lock file enforces this.

    python upstream_exon/score_fullctx.py --library upstream --dry-run
    python upstream_exon/score_fullctx.py --library upstream --pilot 2000
    python upstream_exon/score_fullctx.py --library upstream --max-new-calls 25000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fullctx_lib as F  # noqa: E402
import splice_site_partition as SP  # noqa: E402
from build_fullctx_library import LIBRARIES  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"

#: build this many constructs at a time. 8,612 nt x 2,000 is ~17 MB of
#: strings held at once; the whole shard at once would be hundreds of MB for
#: no benefit, since the calls are serial anyway.
BATCH = 20_000


class Lock:
    """An advisory lock so two shards cannot append to one CSV."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - self.path.stat().st_mtime
            sys.exit(
                f"FATAL: {self.path} exists (age {age / 60:.0f} min) -- another "
                f"scorer holds the lock.\n"
                f"       If you are certain no scorer is running (check "
                f"`squeue -u $USER`), delete it:\n"
                f"         rm {self.path}")
        os.write(self.fd, f"{os.getpid()} {time.time():.0f} "
                          f"{os.environ.get('SLURM_JOB_ID', 'local')}\n".encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", choices=sorted(LIBRARIES), required=True)
    ap.add_argument("--max-new-calls", type=int, default=0,
                    help="cap this shard at N rows; 0 (default) means the "
                         "whole remaining library. The old 25,000 default was "
                         "sized for the API's single-stream rate and is "
                         "pointless locally -- the loop is resumable, so one "
                         "job can just run until its walltime expires")
    ap.add_argument("--pilot", type=int, default=0,
                    help="score only this many rows from the head of TRAIN, "
                         "as a go/no-go before committing the full run")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be called, make no call")
    ap.add_argument("--flush-every", type=int, default=2_000)
    ap.add_argument("--call-timeout", type=float, default=90.0)
    ap.add_argument("--also-splice-sites", action="store_true",
                    help="also record the legacy SPLICE_SITES metric from the "
                         "same call, for the metric-comparison figure")
    ap.add_argument("--backend", default="local", choices=list(F.BACKENDS),
                    help="'local' (default) = weights held on the cluster, "
                         "batched; 'api' = the DeepMind gRPC client "
                         "(quota-limited, single stream)")
    ap.add_argument("--env-path", default=None)
    # --- local backend only -------------------------------------------
    ap.add_argument("--weights",
                    default="/scratch/yl12316/models/alphagenome/"
                            "model_all_folds.safetensors")
    ap.add_argument("--catalog", default=str(F.CATALOG_PATH),
                    help="track_metadata_human.parquet, used to re-derive the "
                         "CL:0002518 track index instead of trusting the "
                         "hard-coded one")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="ceiling; lowered automatically on CUDA OOM")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-strict-preflight", action="store_true",
                    help="report preflight problems but score anyway")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the native-construct check entirely (don't: "
                         "it is one forward pass and it is what catches a "
                         "wrong track index before it corrupts every label)")
    args = ap.parse_args()

    cfg = LIBRARIES[args.library]
    dsdir = OUT / cfg["dirname"]
    manifest = dsdir / "manifest.csv"
    results = dsdir / "ag_results.csv"
    failed = dsdir / "ag_failed.csv"
    lockfile = dsdir / ".scoring.lock"
    slot = cfg["slot"]

    if not manifest.exists():
        sys.exit(f"FATAL: {manifest} not found.\n"
                 f"       Build it first (not committed, reproducible from "
                 f"the seed):\n"
                 f"         python upstream_exon/build_fullctx_library.py "
                 f"--library {args.library}")

    print("=" * 72)
    print(f"SCORING {args.library}  ->  logit_average  [SPLICE_SITE_USAGE]")
    print("=" * 72)

    man = pd.read_csv(manifest, usecols=["row_id", slot, "split"])
    man["row_id"] = man.row_id.astype(str)
    print(f"  manifest    {len(man):,} rows")

    done: set[str] = set()
    if results.exists():
        done = set(pd.read_csv(results, usecols=["row_id"]).row_id.astype(str))
    print(f"  labeled     {len(done):,}")

    pending = man[~man.row_id.isin(done)]
    if args.pilot:
        pending = pending[pending.split == "train"].head(args.pilot)
        print(f"  PILOT       first {args.pilot:,} unlabeled TRAIN rows")
    todo = pending if args.max_new_calls <= 0 else pending.head(args.max_new_calls)
    print(f"  pending     {len(man) - len(done):,}")
    print(f"  this shard  {len(todo):,} calls")

    print(f"  backend     {args.backend}")
    if len(todo) and args.backend == "api":
        lo = 0.49 * len(todo) / 3600
        hi = 0.66 * len(todo) / 3600
        print(f"  estimate    {lo:.1f}-{hi:.1f} h at this repo's observed "
              f"0.49-0.66 s/call over the API")
    elif len(todo):
        # No honest prior: the only measured local rate is 20.5 seq/s on a
        # B200 at 131,072 nt, which is a different GPU and an 8x longer
        # input. The loop prints its own rate, so the run measures itself.
        print("  estimate    unmeasured on this GPU; the loop reports seq/s "
              "and a live ETA. Resumable, so a short walltime is safe")

    if args.dry_run:
        print("\n--dry-run: no AlphaGenome call made")
        return 0
    if not len(todo):
        print("\nnothing to do")
        return 0

    env = args.env_path
    if args.backend == "api":
        if env is None:
            for cand in (ROOT / ".env.alphagenome", ROOT / ".env"):
                if cand.exists():
                    env = str(cand)
                    break
        if env is None:
            sys.exit("FATAL: no .env.alphagenome or .env found at the repo "
                     "root (needed by --backend api)")

    exons = pd.read_csv(OUT / "b1_pilot_exons.csv")
    syn_primary = dict(zip(exons.sequence_id, exons.synthetic_70mer))[F.PRIMARY_EXON]
    native_b1 = SP.pnas_blocks8()["B1"]
    context = F.load_context()

    dsdir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    t0 = time.time()

    # The local scorer holds 921 MB of weights, so it is built ONCE here and
    # reused across every block; rebuilding it per block (as the API path does
    # on timeout) would dominate the run.
    scorer = None
    if args.backend == "local":
        import local_backend as LB

        scorer = LB.LocalScorer(args.weights, device=args.device,
                                catalog_path=args.catalog,
                                batch_size=args.batch_size)
        pre = None
        if not args.skip_preflight:
            pre = LB.preflight(scorer, strict=not args.no_strict_preflight)
        LB.write_provenance(
            scorer, dsdir / "provenance.json",
            extra={"library": args.library, "slot": slot,
                   "manifest_rows": int(len(man)), "preflight": pre})

    with Lock(lockfile):
        for lo_i in range(0, len(todo), BATCH):
            block = todo.iloc[lo_i:lo_i + BATCH]
            jobs = []
            for r in block.itertuples():
                var = getattr(r, slot)
                b1 = var if slot == "b1" else native_b1
                s70 = var if slot == "syn70" else syn_primary
                seq = F.build_construct(b1, s70, context=context)
                # Validate the FIRST construct of every batch rather than all
                # of them: assembly is pure string slicing with no per-row
                # branch, so a failure would be systematic, not sporadic.
                if not jobs:
                    fails = F.validate_construct(seq, b1, s70)
                    if fails:
                        sys.exit(f"FATAL {r.row_id}: {fails}")
                jobs.append({"row_id": r.row_id, slot: var,
                             "split": r.split, "seq": seq})

            if scorer is not None:
                total_written += LB.run_scoring_local(
                    jobs, results, scorer,
                    flush_every=args.flush_every,
                    failed_csv=failed,
                    progress_every=args.flush_every,
                    skip_ids=set(),
                    done_before=lo_i, shard_total=len(todo))
            else:
                total_written += F.run_scoring(
                    jobs, results, env,
                    flush_every=args.flush_every,
                    call_timeout=args.call_timeout,
                    failed_csv=failed,
                    also_splice_sites=args.also_splice_sites,
                    skip_ids=set(),
                    progress_every=args.flush_every,
                    done_before=lo_i, shard_total=len(todo),
                    backend=args.backend)

    el = time.time() - t0
    now = len(done) + total_written
    print(f"\nshard wrote {total_written:,} rows in {el / 3600:.2f} h")
    print(f"library now {now:,} / {cfg['n_total']:,} "
          f"({100 * now / cfg['n_total']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
