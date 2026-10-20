"""
Score the upstream-exon (B1) pilot with AlphaGenome.

Rebuilds every construct from ``b1_lib`` and VERIFIES it against the manifest's
SHA-256 before spending a call, then scores it through
``synthetic_context_lib.run_scoring`` -- the repo's existing resumable,
single-stream, watchdogged harness. The metric is
``synthetic_context_lib.score_core`` unchanged, so ``AG_mean_logit`` is
numerically comparable to the endpoint and 8-block runs.

Resume safety: ``run_scoring`` keys on ``row_id`` and skips anything already in
the output CSV, so a re-run never re-spends a call. The smoke test writes into
the same CSV and is therefore counted, not duplicated.

Usage::

    python upstream_exon/run_b1_pilot.py --smoke     # 1 call, verified against
                                                     # the completed endpoint run
    python upstream_exon/run_b1_pilot.py             # the full pilot (resumes)
    python upstream_exon/run_b1_pilot.py --dry-run   # rebuild + verify, no calls
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
if str(ROOT / "analysis_script") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_script"))

import b1_lib as B  # noqa: E402
import synthetic_context_lib as L  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
MANIFEST_CSV = OUT / "b1_pilot_manifest.csv"
PANEL_CSV = OUT / "b1_variant_panel.csv"
EXONS_CSV = OUT / "b1_pilot_exons.csv"
RESULTS_CSV = OUT / "b1_pilot_ag_results.csv"
FAILED_CSV = OUT / "b1_pilot_failed.csv"

ENDPOINT_CSV = (ROOT / "outputs" / "synthetic_context" /
                "synthetic_endpoint_ag_results.csv")


def build_jobs(man: pd.DataFrame, panel: pd.DataFrame,
               exons: pd.DataFrame) -> list[dict]:
    """Rebuild every construct and verify it against the manifest digest."""
    seqs = dict(zip(panel.b1_id, panel.b1))
    syn = dict(zip(exons.sequence_id, exons.synthetic_70mer))

    jobs: list[dict] = []
    for r in man.itertuples():
        b1 = seqs[r.b1_id]
        core, start = B.build_construct(b1, syn[r.sequence_id])
        digest = B.sha(core)
        if digest != r.core_sha256:
            raise SystemExit(
                f"{r.row_id}: rebuilt construct does not match the manifest "
                f"digest ({digest[:12]} != {r.core_sha256[:12]}) -- refusing "
                "to spend a call")
        if start != r.exon_start or len(core) != r.core_length:
            raise SystemExit(f"{r.row_id}: coordinate mismatch vs manifest")
        fails = B.validate_construct(
            core, start, b1, syn[r.sequence_id],
            require_length=B.B1_LEN if r.fixed_length else None)
        if fails:
            raise SystemExit(f"{r.row_id}: structural validation failed: {fails}")
        jobs.append({
            "row_id": r.row_id,
            "sequence_id": r.sequence_id,
            "exon_class": r.exon_class,
            "alpha": r.alpha,
            "PNAS_common": r.PNAS_common,
            "b1_id": r.b1_id,
            "arm": r.arm,
            "replicate": r.replicate,
            "fixed_length": r.fixed_length,
            "b1_len": r.b1_len,
            "b1_gc": r.b1_gc,
            "b1_sha256": r.b1_sha256,
            "core": core,
            "exon_start": start,
            "exon_len": r.exon_len,
        })
    return jobs


def check_smoke(df: pd.DataFrame, row_id: str) -> bool:
    """Compare a native-B1 result against the completed endpoint run."""
    row = df[df.row_id == row_id]
    if row.empty:
        print(f"[smoke] FAIL: {row_id} produced no result")
        return False
    row = row.iloc[0]
    eid = row_id.split("|")[0]
    ep = pd.read_csv(ENDPOINT_CSV)
    ref = ep[(ep.context == "PNAS") & (ep.sequence_id == eid)]
    if ref.empty:
        print(f"[smoke] no stored endpoint value for {eid}; API call succeeded "
              f"(AG_mean_logit={row.AG_mean_logit:+.4f}) but unverified")
        return True
    ref = ref.iloc[0]
    d_acc = abs(row.AG_acceptor_probability - ref.AG_acceptor_probability)
    d_don = abs(row.AG_donor_probability - ref.AG_donor_probability)
    d_ml = abs(row.AG_mean_logit - ref.AG_mean_logit)
    print(f"[smoke] {eid} native B1, this run vs the 2026-10-07 endpoint run:")
    print(f"          acceptor  {row.AG_acceptor_probability:.5f} vs "
          f"{ref.AG_acceptor_probability:.5f}   |d|={d_acc:.5f}")
    print(f"          donor     {row.AG_donor_probability:.5f} vs "
          f"{ref.AG_donor_probability:.5f}   |d|={d_don:.5f}")
    print(f"          mean_logit {row.AG_mean_logit:+.4f} vs "
          f"{ref.AG_mean_logit:+.4f}   |d|={d_ml:.4f}")
    ok = d_acc < 1e-6 and d_don < 1e-6
    print("[smoke] " + ("PASS -- bit-identical reproduction of the stored result"
                        if ok else
                        "WARNING -- values differ from the stored result"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="score only the first exon's native-B1 construct")
    ap.add_argument("--dry-run", action="store_true",
                    help="rebuild and verify every construct, make no calls")
    ap.add_argument("--env-path", default=str(ROOT / ".env.alphagenome"))
    ap.add_argument("--flush-every", type=int, default=10)
    args = ap.parse_args()

    for p in (MANIFEST_CSV, PANEL_CSV, EXONS_CSV):
        if not p.exists():
            raise SystemExit(f"missing {p}; run build_b1_manifest.py first")

    man = pd.read_csv(MANIFEST_CSV)
    panel = pd.read_csv(PANEL_CSV)
    exons = pd.read_csv(EXONS_CSV)

    jobs = build_jobs(man, panel, exons)
    print(f"[verify] {len(jobs)} constructs rebuilt and SHA-256 verified "
          "against the manifest")

    if args.dry_run:
        print("[dry-run] no AlphaGenome calls made")
        return 0

    if args.smoke:
        smoke_id = f"{exons.sequence_id.iloc[0]}|native_PNAS|r1"
        jobs = [j for j in jobs if j["row_id"] == smoke_id]
        print(f"[smoke] scoring 1 construct: {smoke_id}")

    df = L.run_scoring(jobs, RESULTS_CSV, args.env_path,
                       flush_every=args.flush_every, failed_csv=FAILED_CSV)

    if args.smoke:
        return 0 if check_smoke(df, jobs[0]["row_id"]) else 1

    print(f"\n[pilot] {len(df)} of {len(man)} manifest rows scored")
    if FAILED_CSV.exists():
        print(f"[pilot] failures recorded in {FAILED_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
