"""
Label the B1 dataset with AlphaGenome.

Rebuilds every construct from the manifest, verifies its SHA-256 before
spending a call, and scores through ``synthetic_context_lib.run_scoring`` --
the repo's existing resumable, single-stream, watchdogged harness, unmodified.
``AG_mean_logit`` therefore has exactly the same definition as the endpoint,
8-block and pilot runs.

Resumability: ``run_scoring`` keys on ``row_id`` and skips rows already in the
output CSV. A ``--limit`` smoke run writes into the same file, so those calls
count toward the full job instead of being repeated.

Usage::

    python upstream_exon/score_b1_dataset.py --dry-run
    python upstream_exon/score_b1_dataset.py --limit 8        # smoke test
    python upstream_exon/score_b1_dataset.py                  # full job
    python upstream_exon/score_b1_dataset.py --mode replication
"""

from __future__ import annotations

import argparse
import atexit
import os
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
import synthetic_context_lib as L  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
DEFAULT_DATASET = "dataset_v2"   # v1 stays on disk, untouched

NAMES = {
    "primary": ("b1_primary_manifest.csv", "b1_primary_ag_results.csv",
                "b1_primary_failed.csv"),
    "replication": ("b1_replication_manifest.csv",
                    "b1_replication_ag_results.csv",
                    "b1_replication_failed.csv"),
}


def exon_syn70(sequence_id: str) -> str:
    lib = pd.read_csv(ROOT / "outputs" / "synthetic_context" /
                      "synthetic_exon_library.csv")
    row = lib[lib.sequence_id == sequence_id]
    if row.empty:
        raise SystemExit(f"unknown exon {sequence_id}")
    return row.iloc[0].synthetic_70mer


def stratified_head(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Deterministic alpha-stratified subset, for a representative smoke run."""
    if limit >= len(df):
        return df
    per = {a: int(round(limit * len(g) / len(df)))
           for a, g in df.groupby("alpha")}
    frames = [g.sort_values("b1_id").head(per.get(a, 0))
              for a, g in df.groupby("alpha")]
    out = pd.concat(frames).sort_index()
    return out.head(limit)


def build_jobs(man: pd.DataFrame, syn70: str) -> list[dict]:
    jobs = []
    for r in man.itertuples():
        core, start = B.build_construct(r.b1, syn70)
        digest = B.sha(core)
        if digest != r.core_sha256:
            raise SystemExit(
                f"{r.row_id}: rebuilt construct does not match the manifest "
                f"digest -- refusing to spend a call")
        if start != r.exon_start or len(core) != r.core_length:
            raise SystemExit(f"{r.row_id}: coordinate mismatch vs manifest")
        fails = B.validate_construct(core, start, r.b1, syn70)
        if fails:
            raise SystemExit(f"{r.row_id}: structural validation failed: {fails}")
        jobs.append({
            "row_id": r.row_id, "b1_id": r.b1_id,
            "sequence_id": r.sequence_id, "alpha": r.alpha,
            "split": r.split, "b1_sha256": r.b1_sha256,
            "gc": r.gc, "max_homopolymer": r.max_homopolymer,
            "core": core, "exon_start": start, "exon_len": r.exon_len,
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=list(NAMES), default="primary")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset directory under outputs/upstream_exon/ "
                         "(dataset_v2 = the 40,000-sequence grouped library; "
                         "dataset_v1 = the earlier 10,000-sequence library)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only this many rows (alpha-stratified); "
                         "results append to the same CSV, so a smoke run's "
                         "calls count toward the full job")
    ap.add_argument("--max-new-calls", type=int, default=None,
                    help="score at most this many NOT-YET-SCORED rows, taken "
                         "in manifest order. Lets one SLURM job fit inside a "
                         "walltime cap; chain jobs sequentially to finish.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env-path", default=str(ROOT / ".env.alphagenome"))
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the concurrency lock (not recommended)")
    args = ap.parse_args()

    man_name, res_name, fail_name = NAMES[args.mode]
    DATA = OUT / args.dataset
    man_path = DATA / man_name
    if not man_path.exists():
        raise SystemExit(f"missing {man_path}; run build_b1_dataset.py first")

    man = pd.read_csv(man_path)
    syn70 = exon_syn70(man.sequence_id.iloc[0])
    res_path = DATA / res_name

    # --- what is already done ------------------------------------------
    done_ids: set[str] = set()
    if res_path.exists():
        done_ids = set(pd.read_csv(res_path, usecols=["row_id"]).row_id)
    total = len(man)
    pending = man[~man.row_id.isin(done_ids)]

    print(f"[state]  manifest {total} rows | already scored {len(done_ids)} "
          f"| remaining {len(pending)}")

    # --- select this invocation's slice --------------------------------
    subset = stratified_head(pending, args.limit) if args.limit else pending
    if args.max_new_calls is not None:
        subset = subset.head(args.max_new_calls)

    if subset.empty:
        print("[plan]   nothing left to score -- dataset is complete")
        print(f"[plan]   MAXIMUM NEW ALPHAGENOME CALLS: 0")
        return 0

    # Only unscored constructs are rebuilt, so a resumed job does not pay to
    # re-verify the whole manifest before making its first call.
    jobs = build_jobs(subset, syn70)
    print(f"[verify] {len(jobs)} constructs rebuilt and SHA-256 verified")
    print(f"[plan]   middle exon {man.sequence_id.iloc[0]}, "
          f"alpha mix {subset.alpha.value_counts().sort_index().to_dict()}")
    print(f"[plan]   MAXIMUM NEW ALPHAGENOME CALLS: {len(jobs)}")
    print(f"[plan]   after this job: "
          f"{len(done_ids) + len(jobs)}/{total} labeled")

    if args.dry_run:
        print("[dry-run] no AlphaGenome calls made")
        return 0

    # --- refuse to run two scorers against one results CSV -------------
    lock = res_path.with_suffix(".lock")
    if not args.no_lock:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(
                f"FATAL: {lock} exists -- another scoring process is writing "
                f"to {res_path.name}.\n"
                "       Wait for it, or delete the lock if it is stale.")
        os.write(fd, f"pid={os.getpid()} host={os.uname().nodename}\n"
                     f"slurm_job={os.environ.get('SLURM_JOB_ID', 'none')}\n"
                     .encode())
        os.close(fd)
        atexit.register(lambda: lock.unlink(missing_ok=True))

    df = L.run_scoring(jobs, res_path, args.env_path,
                       flush_every=args.flush_every,
                       failed_csv=DATA / fail_name)
    n_now = len(df)
    print(f"\n[score] {n_now} rows now in {res_path}")
    if n_now >= total:
        print(f"[score] COMPLETE: {n_now}/{total} labeled")
    else:
        print(f"[score] INCOMPLETE: {n_now}/{total} labeled, "
              f"{total - n_now} remaining -- submit another scoring job")
    if D_ok(df):
        print(f"[score] AG_mean_logit: mean {df.AG_mean_logit.mean():+.4f}  "
              f"sd {df.AG_mean_logit.std():.4f}  "
              f"range [{df.AG_mean_logit.min():+.3f}, "
              f"{df.AG_mean_logit.max():+.3f}]")
    if (DATA / fail_name).exists():
        print(f"[score] FAILURES recorded in {DATA / fail_name}")
    return 0


def D_ok(df) -> bool:
    return "AG_mean_logit" in df and len(df) and np.isfinite(
        df.AG_mean_logit).all()


if __name__ == "__main__":
    raise SystemExit(main())
