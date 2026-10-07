"""
JOB 1 -- synthetic-exon endpoint experiment.

Builds the 10,000-exon Dirichlet synthetic library, scores every exon once with
the PNAS pre-tuner (``PNAS_common``), then scores all 10,000 exons with
AlphaGenome SPLICE_SITES in three full contexts:

    PPPPPP   PNAS reporter (incl. the 130-nt upstream-extra)
    FFFFFF   TP53_e6 / FAS
    FFFFFF   TP53_e7 / FAS

= 30,000 AlphaGenome calls, one stream, checkpointed and resumable.

``PNAS_common`` is deliberately computed ONCE per exon from the PNAS splice-site
context (CATCCAG + exon + GTCTGAC) and reused as the x-coordinate under every
AlphaGenome context, so the same exon has the same x everywhere and any y shift
is attributable to context rather than to the PNAS model.

No existing scoring definition is modified.

Usage
-----
    python analysis_script/run_synthetic_endpoint_experiment.py --env-path .env.alphagenome
    python analysis_script/run_synthetic_endpoint_experiment.py --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import synthetic_context_lib as L  # noqa: E402

OUT_DIR = ROOT / "outputs" / "synthetic_context"
LIB_CSV = OUT_DIR / "synthetic_exon_library.csv"
MAIN_CSV = OUT_DIR / "synthetic_endpoint_ag_results.csv"
FAILED_CSV = OUT_DIR / "synthetic_endpoint_failed.csv"

CONTEXTS = ["PNAS", "TP53_e6", "TP53_e7"]


def build_library(args) -> pd.DataFrame:
    if LIB_CSV.exists() and not args.force_library:
        lib = pd.read_csv(LIB_CSV)
        print(f"[library] reusing {len(lib)} rows from {LIB_CSV.name}", flush=True)
        return lib
    lib = L.generate_synthetic_library(seed=args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lib.to_csv(LIB_CSV, index=False)
    print(f"[library] generated {len(lib)} synthetic exons -> {LIB_CSV}", flush=True)
    return lib


def add_pnas_common(lib: pd.DataFrame, args) -> pd.DataFrame:
    if "PNAS_common" in lib.columns and not args.force_library:
        return lib
    import pnas_prediction_hybrid_input as pnas
    print("[pnas] scoring PNAS_common for all exons ...", flush=True)
    triples = [(e, L.PNAS_COMMON_UP7, L.PNAS_COMMON_DN7) for e in lib.full_middle_exon]
    lib["PNAS_common"] = pnas.score_pnas_pretuner_batch(
        triples, num_threads=args.pnas_threads)
    lib.to_csv(LIB_CSV, index=False)
    print(f"[pnas] done, range [{lib.PNAS_common.min():.2f}, {lib.PNAS_common.max():.2f}]",
          flush=True)
    return lib


def build_jobs(lib: pd.DataFrame) -> list[dict]:
    pb = L.pnas_blocks()
    jobs: list[dict] = []
    for ctx in CONTEXTS:
        if ctx == "PNAS":
            blocks, label = pb, "PPPPPP"
        else:
            u70, d25 = L.FAS_TARGETS[ctx]
            blocks, label = L.fas_blocks(u70, d25), "FFFFFF"
        for r in lib.itertuples():
            core, start = L.assemble(label, r.full_middle_exon, pb, blocks)
            jobs.append({
                "row_id": f"{ctx}|{label}|{r.sequence_id}",
                "sequence_id": r.sequence_id, "context": ctx,
                "context_label": label, "alpha": r.alpha,
                "PNAS_common": r.PNAS_common,
                "core": core, "exon_start": start,
                "exon_len": len(r.full_middle_exon),
            })
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="JOB 1: synthetic endpoint experiment.")
    ap.add_argument("--env-path", default=".env.alphagenome")
    ap.add_argument("--seed", type=int, default=L.DEFAULT_SEED)
    ap.add_argument("--pnas-threads", type=int, default=8)
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--call-timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--force-library", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)

    lib = build_library(args)
    print("\n=== STRUCTURAL VALIDATION ===", flush=True)
    fails = L.validate(lib)
    if fails:
        print(f"\nVALIDATION FAILED: {fails}", file=sys.stderr)
        return 1
    print("all structural checks passed\n", flush=True)

    lib = add_pnas_common(lib, args)
    jobs = build_jobs(lib)
    print(f"\nexpected AlphaGenome calls: {len(jobs)} "
          f"({len(lib)} exons x {len(CONTEXTS)} contexts)", flush=True)
    if args.validate_only:
        print("[validate-only] stopping before AlphaGenome.")
        return 0

    L.run_scoring(jobs, MAIN_CSV, args.env_path, flush_every=args.flush_every,
                  call_timeout=args.call_timeout, max_retries=args.max_retries,
                  failed_csv=FAILED_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
