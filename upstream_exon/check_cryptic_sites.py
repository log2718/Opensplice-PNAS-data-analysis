"""
Does the B1 effect measured in the pilot come from cryptic splice sites?

Re-scores a subset of the pilot's constructs with FULL positive-strand splice
tracks (``cryptic_lib.score_core_fulltrack``) instead of the two-position read
``synthetic_context_lib.score_core`` performs, then measures, per construct:

(a) the strongest donor and acceptor probability anywhere in the randomized B1
    region and anywhere else in the construct, excluding the four annotated
    splice sites;
(b) how many such off-target positions clear 0.1 / 0.3 / 0.5;
(c) whether the middle exon's own acceptor and donor peak have drifted from the
    positions ``score_core`` reads (argmax within +/-5 nt).

Before any call it rebuilds every construct with ``b1_lib.build_construct`` and
verifies the SHA-256 against ``b1_pilot_manifest.csv``, exactly as
``run_b1_pilot.py`` does, and it asserts that the full-track scorer reproduces
``score_core`` to floating-point equality on one construct.

Writes only into ``outputs/upstream_exon/cryptic_check/``. Resumable on
``row_id``; a re-run never re-spends a call.

Usage::

    python upstream_exon/check_cryptic_sites.py --dry-run   # verify, no calls
    python upstream_exon/check_cryptic_sites.py --parity    # parity check only
    python upstream_exon/check_cryptic_sites.py             # the full check
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b1_lib as B  # noqa: E402
import cryptic_lib as C  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
CHECK = OUT / "cryptic_check"
MANIFEST_CSV = OUT / "b1_pilot_manifest.csv"
PANEL_CSV = OUT / "b1_variant_panel.csv"
EXONS_CSV = OUT / "b1_pilot_exons.csv"
ANNOT_CSV = OUT / "b1_pilot_results_annotated.csv"

MEASURE_CSV = CHECK / "cryptic_measurements.csv"
TRACKS_NPZ = CHECK / "cryptic_tracks.npz"
PARITY_TXT = CHECK / "parity_check.txt"

#: the primary exon -- pilot rank 1 on every spread measure, and the one the
#: results document recommends for the scale-up
PRIMARY_EXON = "syn_a5.0_00482"
#: one further plateau exon and one rising control, native + the most extreme
#: up and down variant each, as a cross-exon spot check
SECONDARY_EXONS = ["syn_a0.5_00475", "syn_a1.0_00359"]


# ==========================================================================
# subset selection
# ==========================================================================

def select_rows(man: pd.DataFrame, annot: pd.DataFrame) -> pd.DataFrame:
    """The constructs to re-score. Replicate 1 only; fixed-length only."""
    man = man[(man.replicate == 1) & man.fixed_length].copy()

    picked = [man[man.sequence_id == PRIMARY_EXON]]

    fixed = annot[(annot.replicate == 1) & annot.fixed_length]
    for eid in SECONDARY_EXONS:
        sub = fixed[fixed.sequence_id == eid]
        extremes = {"native_PNAS",
                    sub.loc[sub.delta_vs_native.idxmax()].b1_id,
                    sub.loc[sub.delta_vs_native.idxmin()].b1_id}
        picked.append(man[(man.sequence_id == eid) & man.b1_id.isin(extremes)])

    sel = pd.concat(picked, ignore_index=True)
    if sel.row_id.duplicated().any():
        raise SystemExit("duplicate row_id in the selected subset")
    return sel


def build_jobs(sel: pd.DataFrame, panel: pd.DataFrame,
               exons: pd.DataFrame) -> list[dict]:
    """Rebuild and SHA-256-verify every selected construct before any call."""
    seqs = dict(zip(panel.b1_id, panel.b1))
    syn = dict(zip(exons.sequence_id, exons.synthetic_70mer))

    jobs: list[dict] = []
    for r in sel.itertuples():
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
        fails = B.validate_construct(core, start, b1, syn[r.sequence_id],
                                     require_length=B.B1_LEN)
        if fails:
            raise SystemExit(f"{r.row_id}: structural validation failed: {fails}")
        # also confirm the four annotated sites sit where the block layout says
        C.known_sites(len(b1), start, r.exon_len)
        jobs.append({
            "row_id": r.row_id, "sequence_id": r.sequence_id,
            "exon_class": r.exon_class, "b1_id": r.b1_id, "arm": r.arm,
            "b1_len": int(r.b1_len), "b1_gc": float(r.b1_gc),
            "core": core, "exon_start": int(start), "exon_len": int(r.exon_len),
        })
    return jobs


# ==========================================================================
# the run loop -- single stream, watchdogged, resumable (same shape as
# synthetic_context_lib.run_scoring, which is hardcoded to score_core)
# ==========================================================================

def run(jobs: list[dict], env_path: str, parity_job: dict | None,
        call_timeout: float = 90.0, max_retries: int = 3) -> pd.DataFrame:
    import alphagenome_local_prediction_pipeline_minigene as ag

    CHECK.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    tracks: dict[str, np.ndarray] = {}
    if MEASURE_CSV.exists():
        prev = pd.read_csv(MEASURE_CSV)
        done = {r["row_id"]: r.to_dict() for _, r in prev.iterrows()}
        print(f"[resume] {len(done)} rows already measured", flush=True)
    if TRACKS_NPZ.exists():
        with np.load(TRACKS_NPZ) as z:
            tracks = {k: z[k] for k in z.files}

    model = ag.create_model(env_path=env_path)
    pool = ThreadPoolExecutor(max_workers=1)
    calls = 0

    # --- parity assertion, and its full-track result is kept, not wasted ------
    if parity_job is not None and parity_job["row_id"] not in done:
        print(f"[parity] score_core vs score_core_fulltrack on "
              f"{parity_job['row_id']}", flush=True)
        p = C.assert_parity(model, parity_job["core"], parity_job["exon_start"],
                            parity_job["exon_len"])
        calls += 2
        lines = [f"parity check on {parity_job['row_id']}", ""]
        for k in C._SCALAR_KEYS:
            a, b = p["reference"][k], p["fulltrack"][k]
            lines.append(f"  {k:28s} score_core={a!r:24s} fulltrack={b!r:24s} "
                         f"{'EQUAL' if a == b else 'MISMATCH'}")
        lines.append("")
        lines.append("RESULT: " + ("PASS -- every shared scalar identical"
                                   if not p["mismatches"] else
                                   f"FAIL -- {p['mismatches']}"))
        PARITY_TXT.write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        if p["mismatches"]:
            raise SystemExit("parity check FAILED -- stopping before any "
                             "further calls; the full-track path does not "
                             "reproduce score_core")
        done[parity_job["row_id"]] = _record(parity_job, p["fulltrack"])
        tracks[parity_job["row_id"] + "|donor"] = p["fulltrack"]["donor_track"]
        tracks[parity_job["row_id"] + "|acceptor"] = p["fulltrack"]["acceptor_track"]

    todo = [j for j in jobs if j["row_id"] not in done]
    print(f"[scoring] {len(todo)} of {len(jobs)} constructs to call", flush=True)
    t0 = time.time()

    for i, job in enumerate(todo, 1):
        res = None
        for attempt in range(1, max_retries + 1):
            try:
                fut = pool.submit(C.score_core_fulltrack, model, job["core"],
                                  job["exon_start"], job["exon_len"])
                res = fut.result(timeout=call_timeout)
                calls += 1
                break
            except FutureTimeout:
                print(f"  [timeout {attempt}/{max_retries}] {job['row_id']} "
                      "-- rebuilding client", flush=True)
                pool.shutdown(wait=False)
                pool = ThreadPoolExecutor(max_workers=1)
                model = ag.create_model(env_path=env_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error {attempt}/{max_retries}] {job['row_id']}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                time.sleep(2 * attempt)
        if res is None:
            raise SystemExit(f"{job['row_id']}: failed after {max_retries} tries")

        done[job["row_id"]] = _record(job, res)
        tracks[job["row_id"] + "|donor"] = res["donor_track"]
        tracks[job["row_id"] + "|acceptor"] = res["acceptor_track"]
        if i % 10 == 0 or i == len(todo):
            _flush(done, tracks)
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {el:.0f}s  "
                  f"offtarget_max={done[job['row_id']]['offtarget_max']:.4f}",
                  flush=True)

    _flush(done, tracks)
    print(f"[done] {calls} AlphaGenome calls this invocation", flush=True)
    (CHECK / "call_count.txt").write_text(f"{calls}\n")
    return pd.DataFrame(list(done.values()))


def _record(job: dict, res: dict) -> dict:
    m = C.measure(res["donor_track"], res["acceptor_track"], job["b1_len"],
                  job["exon_start"], job["exon_len"])
    rec = {k: job[k] for k in ("row_id", "sequence_id", "exon_class", "b1_id",
                               "arm", "b1_len", "b1_gc", "exon_start",
                               "exon_len")}
    rec.update({k: res[k] for k in C._SCALAR_KEYS})
    rec.update({k: v for k, v in m.items() if k not in ("b1_len", "core_length")})
    return rec


def _flush(done: dict, tracks: dict) -> None:
    pd.DataFrame(list(done.values())).to_csv(MEASURE_CSV, index=False)
    np.savez_compressed(TRACKS_NPZ, **tracks)


# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parity", action="store_true",
                    help="run only the score_core parity assertion (2 calls)")
    ap.add_argument("--env-path", default=str(ROOT / ".env.alphagenome"))
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST_CSV)
    panel = pd.read_csv(PANEL_CSV)
    exons = pd.read_csv(EXONS_CSV)
    annot = pd.read_csv(ANNOT_CSV)

    sel = select_rows(man, annot)
    jobs = build_jobs(sel, panel, exons)
    print(f"[verify] {len(jobs)} constructs rebuilt and SHA-256 verified "
          f"against the manifest")
    for eid, n in sel.sequence_id.value_counts().items():
        print(f"           {eid}: {n}")

    parity_job = next(j for j in jobs
                      if j["row_id"] == f"{PRIMARY_EXON}|native_PNAS|r1")
    if args.dry_run:
        ks = C.known_sites(parity_job["b1_len"], parity_job["exon_start"],
                           parity_job["exon_len"])
        print("[dry-run] annotated splice sites in the 1,309-nt core:")
        for k, v in ks.items():
            print(f"           {k:22s} {v['kind']:9s} pos {v['pos']}")
        print(f"[dry-run] expected calls: {len(jobs)} full-track + 1 score_core "
              f"= {len(jobs) + 1}")
        print("[dry-run] no AlphaGenome calls made")
        return 0

    if args.parity:
        jobs = [parity_job]
    run(jobs, args.env_path, parity_job)
    print(f"[out] {MEASURE_CSV}")
    print(f"[out] {TRACKS_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
