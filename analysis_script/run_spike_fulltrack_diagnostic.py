"""
PREPARED, NOT RUN -- full SPLICE_SITES track for the spike representatives.

The endpoint CSV stores only the *intended* middle-exon acceptor/donor values.
It therefore cannot answer the second hypothesis for the far-left spikes: that a
randomised exon creates a strong alternative donor/acceptor somewhere else in
the construct. This script requests the complete SPLICE_SITES track for a small
representative set and looks for peaks away from the intended sites.

It makes AlphaGenome calls, so it refuses to do anything unless explicitly
unlocked:

    python analysis_script/run_spike_fulltrack_diagnostic.py --dry-run     # safe
    python analysis_script/run_spike_fulltrack_diagnostic.py --execute ... # calls AG

``--dry-run`` builds every construct, verifies the geometry, and prints the call
plan without contacting AlphaGenome.

Inputs
------
``outputs/synthetic_context/spike_diagnosis/fulltrack_representatives.csv``
(written by ``diagnose_endpoint_spikes.py``).

Design
------
Contexts reuse ``synthetic_context_lib`` unchanged, so a construct here is
byte-identical to the one the endpoint run scored:

* ``PNAS``    -- ``assemble("PPPPPP", ...)`` with ``pnas_blocks()``
* ``TP53_e6`` -- ``assemble("FFFFFF", ...)`` with ``fas_blocks(*FAS_TARGETS["TP53_e6"])``
* ``TP53_e7`` -- ``assemble("FFFFFF", ...)`` with ``fas_blocks(*FAS_TARGETS["TP53_e7"])``

For each construct the script records, over the unpadded core only:

* the intended acceptor / donor probability (reproducing ``score_core``)
* the top-k off-target acceptor and donor peaks, with position, probability,
  distance to the intended site, and which region they fall in
  (upstream exon / upstream intron / middle exon / downstream intron /
  downstream exon)
* the ratio ``max(off-target) / intended``, which is the quantity that decides
  whether a cryptic site is plausibly competing

Nothing here calls a site functional; a peak in the SPLICE_SITES track is
evidence about the model, not about splicing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import synthetic_context_lib as L  # noqa: E402

DIAG = ROOT / "outputs" / "synthetic_context" / "spike_diagnosis"
REPS_CSV = DIAG / "fulltrack_representatives.csv"
OUT_CSV = DIAG / "fulltrack_splice_sites.csv"
PEAKS_CSV = DIAG / "fulltrack_offtarget_peaks.csv"

CONTEXTS = ["PNAS", "TP53_e6", "TP53_e7"]
TOP_K = 5
#: ignore peaks within this many nt of the intended site -- they are the
#: shoulder of the intended peak, not an alternative site
EXCLUSION_NT = 10


def build_jobs(reps: pd.DataFrame) -> list[dict]:
    """One job per (representative, context). No network access."""
    pb = L.pnas_blocks()
    fbs = {t: L.fas_blocks(u, d) for t, (u, d) in L.FAS_TARGETS.items()}

    jobs: list[dict] = []
    for _, r in reps.iterrows():
        exon = L.EXON_PREFIX + r.synthetic_70mer + L.EXON_SUFFIX
        for ctx in CONTEXTS:
            if ctx == "PNAS":
                core, start = L.assemble("PPPPPP", exon, pb, pb)
            else:
                core, start = L.assemble("FFFFFF", exon, pb, fbs[ctx])
            assert core[start:start + len(exon)] == exon, "exon misplaced"
            jobs.append({
                "row_id": f"{r.sequence_id}|{ctx}",
                "sequence_id": r.sequence_id,
                "role": r.role,
                "context": ctx,
                "PNAS_common": r.PNAS_common,
                "homopolymer_class": r.homopolymer_class,
                "core": core,
                "exon_start": start,
                "exon_len": len(exon),
            })
    return jobs


def region_of(pos: int, exon_start: int, exon_len: int, core_len: int,
              ctx: str) -> str:
    """Coarse region label for an off-target peak inside the core."""
    if pos < exon_start:
        return "upstream_of_exon"
    if pos < exon_start + exon_len:
        return "middle_exon"
    return "downstream_of_exon"


def analyse_track(track: np.ndarray, intended: int, exon_start: int,
                  exon_len: int, core_len: int, left_pad: int,
                  ctx: str, kind: str) -> list[dict]:
    """Top-k peaks in the core, excluding the intended site's neighbourhood."""
    core_slice = track[left_pad:left_pad + core_len]
    mask = np.ones(core_len, dtype=bool)
    lo = max(0, intended - left_pad - EXCLUSION_NT)
    hi = min(core_len, intended - left_pad + EXCLUSION_NT + 1)
    mask[lo:hi] = False
    cand = np.where(mask)[0]
    order = cand[np.argsort(-core_slice[cand])][:TOP_K]
    out = []
    for rank, p in enumerate(order, start=1):
        out.append({
            "kind": kind,
            "rank": rank,
            "core_position": int(p),
            "probability": float(core_slice[p]),
            "distance_to_intended": int(p - (intended - left_pad)),
            "region": region_of(int(p), exon_start, exon_len, core_len, ctx),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="build and validate everything; make NO AlphaGenome calls")
    g.add_argument("--execute", action="store_true",
                   help="actually call AlphaGenome (requires --yes-i-mean-it)")
    ap.add_argument("--yes-i-mean-it", action="store_true")
    ap.add_argument("--env-path", default=str(ROOT / ".env"))
    args = ap.parse_args(argv)

    if not REPS_CSV.exists():
        print(f"missing {REPS_CSV}; run diagnose_endpoint_spikes.py first")
        return 1
    reps = pd.read_csv(REPS_CSV)
    jobs = build_jobs(reps)

    print(f"representatives : {len(reps)}")
    print(f"contexts        : {', '.join(CONTEXTS)}")
    print(f"AlphaGenome calls that WOULD be made: {len(jobs)}")
    print(f"core lengths    : "
          f"{sorted({(j['context'], len(j['core'])) for j in jobs})}")
    by_role = reps.groupby("role").sequence_id.apply(list)
    for role, ids in by_role.items():
        print(f"  {role:8s} ({len(ids)}): {', '.join(ids)}")

    if args.dry_run:
        print("\n[dry-run] geometry verified for every job; no AlphaGenome call "
              "was made and no output file was written.")
        return 0

    if not args.yes_i_mean_it:
        print("\nRefusing to call AlphaGenome without --yes-i-mean-it.")
        return 2

    # ---- from here on, network calls happen -------------------------------
    import alphagenome_local_prediction_pipeline_minigene as ag
    from alphagenome.models import dna_client

    model = ag.create_model(env_path=args.env_path)
    rows, peaks = [], []
    for i, job in enumerate(jobs, start=1):
        padded, left_pad, _ = ag.center_pad(ag.clean_seq(job["core"]))
        acc_pos = left_pad + job["exon_start"]
        don_pos = left_pad + job["exon_start"] + job["exon_len"] - 1
        pred = model.predict_sequence(
            sequence=padded,
            organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=[dna_client.OutputType.SPLICE_SITES],
            ontology_terms=["CL:0002518"],
        )
        donor_track, acceptor_track = ag.get_positive_strand_tracks(pred)
        acc, don = float(acceptor_track[acc_pos]), float(donor_track[don_pos])

        off_a = analyse_track(acceptor_track, acc_pos, job["exon_start"],
                              job["exon_len"], len(job["core"]), left_pad,
                              job["context"], "acceptor")
        off_d = analyse_track(donor_track, don_pos, job["exon_start"],
                              job["exon_len"], len(job["core"]), left_pad,
                              job["context"], "donor")
        for p in off_a + off_d:
            peaks.append({**{k: job[k] for k in
                             ("row_id", "sequence_id", "role", "context")}, **p})

        rows.append({
            **{k: job[k] for k in ("row_id", "sequence_id", "role", "context",
                                   "PNAS_common", "homopolymer_class",
                                   "exon_start", "exon_len")},
            "intended_acceptor": acc,
            "intended_donor": don,
            "max_offtarget_acceptor": max(p["probability"] for p in off_a),
            "max_offtarget_donor": max(p["probability"] for p in off_d),
            "offtarget_acceptor_ratio": max(p["probability"] for p in off_a) / max(acc, 1e-12),
            "offtarget_donor_ratio": max(p["probability"] for p in off_d) / max(don, 1e-12),
        })
        if i % 10 == 0 or i == len(jobs):
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            pd.DataFrame(peaks).to_csv(PEAKS_CSV, index=False)
            print(f"  [{i}/{len(jobs)}] {job['row_id']}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(peaks).to_csv(PEAKS_CSV, index=False)
    print(f"\nwrote {OUT_CSV}\nwrote {PEAKS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
