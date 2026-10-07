"""
JOB 2 -- 2^6 chimera experiment (depends on JOB 1).

For each of the two comparisons (PNAS <-> TP53_e6/FAS, PNAS <-> TP53_e7/FAS)
all 64 six-block contexts are scored on a deterministic, balanced 400-exon
subset (100 per Dirichlet alpha), using the identical synthetic exon in every
construct.

The two endpoint contexts of each comparison -- ``PPPPPP`` and ``FFFFFF`` -- were
already scored for these exons by JOB 1 and are reused, not rescored. ``PPPPPP``
is physically the same construct in both comparisons, so:

    (64 - 2) x 2 comparisons = 124 new contexts x 400 exons = 49,600 calls

Block order is ``L1 L2 L3 | exon | R3 R2 R1`` (see ``synthetic_context_lib``).

``PNAS_common`` stays the primary x-coordinate so an exon keeps one x-value in
every context. ``PNAS_contextual`` is computed as a secondary quantity from the
chimera's actual L3/R3 7-mers, to separate "AlphaGenome moved because of
context" from "the PNAS model itself would have predicted a shift".

Usage
-----
    python analysis_script/run_chimera_experiment.py --env-path .env.alphagenome
    python analysis_script/run_chimera_experiment.py --dry-run
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
ENDPOINT_CSV = OUT_DIR / "synthetic_endpoint_ag_results.csv"
SUBSET_CSV = OUT_DIR / "chimera_subset_ids.csv"
CHIMERA_CSV = OUT_DIR / "chimera_ag_results.csv"
FAILED_CSV = OUT_DIR / "chimera_failed.csv"

COMPARISONS = ["TP53_e6", "TP53_e7"]
ENDPOINTS = {"PPPPPP", "FFFFFF"}


def build_jobs(lib: pd.DataFrame, subset_ids: list[str]) -> list[dict]:
    """One job per (comparison, non-endpoint label, exon)."""
    import pnas_prediction_hybrid_input as pnas

    sub = lib[lib.sequence_id.isin(subset_ids)].copy()
    pb = L.pnas_blocks()
    labels = [l for l in L.all_labels() if l not in ENDPOINTS]

    # PNAS_contextual depends only on (L3, R3) and the exon, so score each
    # distinct 7-mer pair once across the whole subset rather than per chimera.
    jobs: list[dict] = []
    for target in COMPARISONS:
        u70, d25 = L.FAS_TARGETS[target]
        fb = L.fas_blocks(u70, d25)
        pair_cache: dict[tuple[str, str], dict[str, float]] = {}
        for lab in labels:
            l3 = (pb if lab[2] == "P" else fb)["L3"]
            r3 = (pb if lab[3] == "P" else fb)["R3"]
            key = (l3, r3)
            if key not in pair_cache:
                scores = pnas.score_pnas_pretuner_batch(
                    [(e, l3, r3) for e in sub.full_middle_exon], num_threads=8)
                pair_cache[key] = dict(zip(sub.sequence_id, scores))
            ctxmap = pair_cache[key]
            for r in sub.itertuples():
                core, start = L.assemble(lab, r.full_middle_exon, pb, fb)
                jobs.append({
                    "row_id": f"{target}|{lab}|{r.sequence_id}",
                    "sequence_id": r.sequence_id, "comparison": target,
                    "context_label": lab, "alpha": r.alpha,
                    "L1": lab[0], "L2": lab[1], "L3": lab[2],
                    "R3": lab[3], "R2": lab[4], "R1": lab[5],
                    "n_F_blocks": lab.count("F"),
                    "PNAS_common": r.PNAS_common,
                    "PNAS_contextual": ctxmap[r.sequence_id],
                    "core": core, "exon_start": start,
                    "exon_len": len(r.full_middle_exon),
                })
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="JOB 2: 2^6 chimera experiment.")
    ap.add_argument("--env-path", default=".env.alphagenome")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--call-timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not LIB_CSV.exists():
        print(f"[missing] {LIB_CSV} -- run JOB 1 first.", file=sys.stderr)
        return 1
    lib = pd.read_csv(LIB_CSV)
    if "PNAS_common" not in lib.columns:
        print("[missing] PNAS_common -- JOB 1 did not finish its PNAS stage.",
              file=sys.stderr)
        return 1

    subset_ids = L.balanced_subset(lib)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sequence_id": subset_ids}).to_csv(SUBSET_CSV, index=False)
    per_alpha = lib[lib.sequence_id.isin(subset_ids)].alpha.value_counts().sort_index()
    print(f"[subset] {len(subset_ids)} exons, per alpha {per_alpha.to_dict()}", flush=True)

    if ENDPOINT_CSV.exists():
        ep = pd.read_csv(ENDPOINT_CSV)
        have = ep[ep.sequence_id.isin(subset_ids)]
        print(f"[endpoints] JOB 1 already scored {len(have)} endpoint rows for "
              f"these exons (reused, not rescored)", flush=True)
    else:
        print("[warn] endpoint results not found; endpoint contexts will be "
              "missing from the chimera analysis", flush=True)

    jobs = build_jobs(lib, subset_ids)
    n_ctx = len([l for l in L.all_labels() if l not in ENDPOINTS]) * len(COMPARISONS)
    print(f"\nnew contexts: {n_ctx}  x  {len(subset_ids)} exons "
          f"= {len(jobs)} AlphaGenome calls", flush=True)
    if args.dry_run:
        print("[dry-run] stopping before AlphaGenome.")
        return 0

    L.run_scoring(jobs, CHIMERA_CSV, args.env_path, flush_every=args.flush_every,
                  call_timeout=args.call_timeout, max_retries=args.max_retries,
                  failed_csv=FAILED_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
