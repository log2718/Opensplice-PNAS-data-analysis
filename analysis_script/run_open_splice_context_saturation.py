"""
Score the OpenSplice/FAS-context saturation variants with the PNAS pre-tuner
and AlphaGenome SPLICE_SITES.

Reuses, unchanged:
* ``run_hybrid_pnas_alphagenome.load_opensplice_exon`` -- WT exon + 70/25 flanks
* ``pnas_prediction_hybrid_input.score_pnas_pretuner_batch`` -- PNAS pre-tuner
  on ``upstream[-7:] + exon + downstream[:7]`` (NOT the FAS cassette)
* ``alphagenome_local_prediction_pipeline_minigene.score_exon`` -- the SAME FAS
  minigene construction used in every previous TP53/OpenSplice experiment
  (FAS_E5 + FAS_I5 + 70 nt + exon + 25 nt + FAS_I6 + FAS_E7, centre-padded to
  16,384) and the SAME ``mean_logit`` / ``prod_logit`` definitions.

The PNAS three-exon calibration cassette is deliberately NOT used here.

Robustness
----------
PNAS runs first, locally and in one batch (~13 min for 20k variants), so an
AlphaGenome failure never costs PNAS work. AlphaGenome then runs as a single
stream (one process -- no concurrent API usage) with:

* a per-call **watchdog**: each ``score_exon`` runs in a worker thread and is
  abandoned if it exceeds ``--call-timeout``. This exists because a previous
  run blocked indefinitely inside the gRPC client, which retries errors but has
  no timeout for a call that simply never returns. On timeout the AlphaGenome
  client is rebuilt so a poisoned channel cannot stall the rest of the run.
* bounded retries for transient failures, with backoff;
* checkpointing every ``--flush-every`` successful calls;
* resume keyed on the deterministic ``variant_id``, so a job killed by the
  Slurm time limit loses nothing;
* failures logged separately to ``failed_variants.csv`` rather than aborting.

Usage
-----
    python analysis_script/run_open_splice_context_saturation.py \\
        --env-path .env.alphagenome
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

OUT_DIR = ROOT / "outputs" / "open_splice_context_saturation"
UNSCORED = OUT_DIR / "variants_unscored.csv"
PNAS_CSV = OUT_DIR / "variants_with_pnas.csv"
MAIN_CSV = OUT_DIR / "open_splice_context_saturation_variants.csv"
FAILED_CSV = OUT_DIR / "failed_variants.csv"


def _logit(p: float, eps: float = 1e-7) -> float:
    """Identical convention to ``run_hybrid_pnas_alphagenome._logit``."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _log(msg: str) -> None:
    print(msg, flush=True)


def stage_pnas(args) -> pd.DataFrame:
    """Score every variant with the PNAS pre-tuner (local, batched)."""
    if PNAS_CSV.exists() and not args.force_pnas:
        df = pd.read_csv(PNAS_CSV)
        _log(f"[pnas] reusing {len(df)} rows from {PNAS_CSV.name}")
        return df

    import run_hybrid_pnas_alphagenome as hybrid
    import pnas_prediction_hybrid_input as pnas

    df = pd.read_csv(UNSCORED)
    flanks = {}
    for exon_id in df["exon_id"].unique():
        wt = hybrid.load_opensplice_exon(exon_id, hybrid.DEFAULT_OPENSPLICE_CSV)
        flanks[exon_id] = (wt.upstream_flank, wt.downstream_flank)

    triples = [
        (r.mutated_exon_sequence, *flanks[r.exon_id])
        for r in df.itertuples()
    ]
    _log(f"[pnas] scoring {len(triples)} variants ...")
    t0 = time.time()
    df["PNAS_pretuner"] = pnas.score_pnas_pretuner_batch(
        triples, num_threads=args.pnas_threads
    )
    _log(f"[pnas] done in {time.time()-t0:.1f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PNAS_CSV, index=False)
    return df


def stage_alphagenome(df: pd.DataFrame, args) -> None:
    import alphagenome_local_prediction_pipeline_minigene as ag
    import run_hybrid_pnas_alphagenome as hybrid

    flanks = {}
    for exon_id in df["exon_id"].unique():
        wt = hybrid.load_opensplice_exon(exon_id, hybrid.DEFAULT_OPENSPLICE_CSV)
        flanks[exon_id] = (wt.upstream_flank, wt.downstream_flank)

    done: dict[str, dict] = {}
    if MAIN_CSV.exists():
        prev = pd.read_csv(MAIN_CSV)
        done = {r["variant_id"]: r.to_dict() for _, r in prev.iterrows()}
        _log(f"[resume] {len(done)} variants already scored in {MAIN_CSV.name}")

    todo = df[~df["variant_id"].isin(done.keys())]
    total = len(todo)
    _log(f"[alphagenome] {total} variants to score (single stream)\n")

    records = list(done.values())
    failures: list[dict] = []
    model = ag.create_model(env_path=args.env_path)
    pool = ThreadPoolExecutor(max_workers=1)
    t_start = time.time()
    n_done = 0

    for row in todo.itertuples():
        up, dn = flanks[row.exon_id]
        result = None
        for attempt in range(1, args.max_retries + 1):
            try:
                fut = pool.submit(
                    ag.score_exon, model, row.mutated_exon_sequence, up, dn,
                    require_opensplice_flank_lengths=True,
                    ontology_terms=("CL:0002518",),
                )
                result = fut.result(timeout=args.call_timeout)
                break
            except FutureTimeout:
                _log(f"  [timeout {attempt}/{args.max_retries}] {row.variant_id} "
                     f"(> {args.call_timeout}s) -- rebuilding client")
                # Abandon the stuck worker; a poisoned channel must not stall the run.
                pool.shutdown(wait=False)
                pool = ThreadPoolExecutor(max_workers=1)
                model = ag.create_model(env_path=args.env_path)
            except Exception as exc:  # transient API/network error
                _log(f"  [error {attempt}/{args.max_retries}] {row.variant_id}: "
                     f"{type(exc).__name__}: {exc}")
                time.sleep(min(2 ** attempt, 30))

        if result is None:
            failures.append({"variant_id": row.variant_id, "exon_id": row.exon_id,
                             "motif_family": row.motif_family,
                             "copy_number": row.copy_number,
                             "reason": "exhausted retries"})
            pd.DataFrame(failures).to_csv(FAILED_CSV, index=False)
            continue

        acc, don = float(result["acceptor"]), float(result["donor"])
        rec = {c: getattr(row, c) for c in df.columns}
        rec.update({
            "AG_acceptor_probability": acc,
            "AG_donor_probability": don,
            "AG_mean_prob": (acc + don) / 2.0,
            "AG_mean_logit": (_logit(acc) + _logit(don)) / 2.0,
            "AG_prod_logit": _logit(acc * don),
        })
        records.append(rec)
        n_done += 1

        if n_done % args.flush_every == 0 or n_done == total:
            pd.DataFrame(records).to_csv(MAIN_CSV, index=False)
            el = time.time() - t_start
            rate = el / n_done
            eta = (total - n_done) * rate
            _log(f"  [{n_done}/{total}] {row.exon_id} {row.motif_family} "
                 f"{row.copy_number}x | {rate:.2f}s/call | elapsed {el/60:.1f}m "
                 f"| ETA {eta/60:.1f}m | {len(records)} rows saved")

    pd.DataFrame(records).to_csv(MAIN_CSV, index=False)
    pool.shutdown(wait=False)
    _log(f"\n[done] {len(records)} scored rows -> {MAIN_CSV}")
    if failures:
        pd.DataFrame(failures).to_csv(FAILED_CSV, index=False)
        _log(f"[done] {len(failures)} failures -> {FAILED_CSV}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score OpenSplice/FAS context-saturation variants.")
    ap.add_argument("--env-path", default=".env.alphagenome")
    ap.add_argument("--pnas-threads", type=int, default=8)
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--call-timeout", type=float, default=90.0,
                    help="seconds before a single AlphaGenome call is abandoned")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--force-pnas", action="store_true")
    ap.add_argument("--pnas-only", action="store_true")
    args = ap.parse_args(argv)

    if not UNSCORED.exists():
        _log(f"[missing] {UNSCORED}\n  run: python analysis_script/generate_context_saturation_variants.py")
        return 1

    df = stage_pnas(args)
    if args.pnas_only:
        _log("[pnas-only] stopping before AlphaGenome.")
        return 0
    stage_alphagenome(df, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
