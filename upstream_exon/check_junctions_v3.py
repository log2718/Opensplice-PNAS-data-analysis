#!/usr/bin/env python
"""Do cryptic junctions inside B1 explain the v3 upstream-exon signal?

The v3 library's target, ``AG_mean_logit``, is read from the per-base
SPLICE_SITES track at two fixed positions. That track says a position *looks*
like a splice site; it never says the site is *used*. SPLICE_JUNCTIONS does:
it returns discrete donor->acceptor pairs with a predicted abundance each.

The probe (probe_junction_output.py) established the convention on the native
PNAS construct: junction start = donor + 1, end = acceptor, both exact
nucleotides, in padded coordinates that ``left_pad`` converts to core
coordinates. There, 13.8% of all junction signal arriving at the middle
acceptor came from cryptic donors inside B1 -- in the UNMODIFIED sequence.
This script turns that single number into a distribution over v3.

ONE call per sequence requests BOTH outputs, so every row also re-derives
``AG_mean_logit`` and compares it to the value the dataset already holds. A
mismatch means the construct or the scoring path differs from the one that
built the dataset, and the run stops rather than producing numbers that cannot
be joined back.

Resumable on ``row_id``; a re-run never re-spends a call. Writes only into
``outputs/upstream_exon/junction_check_v3/``.

    python upstream_exon/check_junctions_v3.py --dry-run          # no calls
    python upstream_exon/check_junctions_v3.py --n 50             # 50 calls
    python upstream_exon/check_junctions_v3.py --n 3000
"""
from __future__ import annotations

import argparse
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

import b1_lib as B  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
SHARE_CSV = OUT / "b1_v3_upstream_exon_250k.csv.gz"
EXONS_CSV = OUT / "b1_pilot_exons.csv"
CHECK = OUT / "junction_check_v3"
MEASURE_CSV = CHECK / "junction_measurements.csv"
META_JSON = CHECK / "run_meta.json"

PRIMARY_EXON = "syn_a5.0_00482"
B1_LO, B1_HI = 0, 113          #: B1 occupies core [0, 113)
DESIGNED_DONOR = 115           #: last exonic base of the upstream exon
DESIGNED_START = 116           #: therefore the designed junction's start
MIDDLE_ACCEPTOR = 243
MIDDLE_DONOR = 318
DOWNSTREAM_ACCEPTOR = 1168
ONTOLOGY = ("CL:0002518",)

#: tolerance for the AG_mean_logit re-derivation. AlphaGenome was measured
#: bit-identical within a session in the pilot, but this runs across sessions,
#: so an exact == would be the wrong bar; 1e-6 still catches a real divergence.
LOGIT_TOL = 1e-6


def pick_subset(df: pd.DataFrame, n: int, split: str, seed: int) -> pd.DataFrame:
    """A subset stratified across AG_mean_logit deciles of one split.

    Uniform sampling would spend most calls in the middle of the distribution,
    where the sequences are least informative about whether cryptic junctions
    drive the extremes. Equal counts per decile keeps both tails populated.
    """
    pool = df[df.split == split]
    if pool.empty:
        sys.exit(f"FATAL: no rows with split={split!r}")
    n = min(n, len(pool))
    q = pd.qcut(pool.AG_mean_logit, 10, labels=False, duplicates="drop")
    per = max(1, n // int(q.nunique()))
    # Explicit per-decile sampling rather than groupby.apply: pandas versions
    # disagree about whether the grouping column reaches the callable.
    parts = [pool[q == d].sample(min(per, int((q == d).sum())),
                                 random_state=seed)
             for d in sorted(q.unique())]
    picked = pd.concat(parts)
    if len(picked) > n:
        picked = picked.sample(n, random_state=seed)
    return picked.sort_values("row_id").reset_index(drop=True)


def junction_metrics(jd, left_pad: int) -> dict:
    """Collapse JunctionData into the per-sequence numbers we compare."""
    starts = np.array([int(iv.start) - left_pad for iv in jd.junctions])
    ends = np.array([int(iv.end) - left_pad for iv in jd.junctions])
    vals = np.asarray(jd.values, dtype=np.float64).reshape(len(starts), -1)[:, 0]

    total = float(vals.sum())
    # A donor at core position d produces a junction starting at d + 1, so a
    # donor anywhere in B1 [0, 113) gives a start in [1, 113].
    from_b1 = (starts >= B1_LO + 1) & (starts <= B1_HI)
    to_acc = ends == MIDDLE_ACCEPTOR
    designed = (starts == DESIGNED_START) & to_acc

    acc_total = float(vals[to_acc].sum())
    v_designed = float(vals[designed].sum())
    v_b1_to_acc = float(vals[from_b1 & to_acc].sum())

    # middle donor's own fidelity, as a control: B1 cannot create a donor there
    from_md = starts == MIDDLE_DONOR + 1
    md_total = float(vals[from_md].sum())
    v_md_designed = float(vals[from_md & (ends == DOWNSTREAM_ACCEPTOR)].sum())

    skip = (starts == DESIGNED_START) & (ends == DOWNSTREAM_ACCEPTOR)

    b1_vals = vals[from_b1]
    return {
        "n_junctions": len(starts),
        "junction_total": total,
        # the headline: competition for the middle acceptor
        "acceptor_total": acc_total,
        "v_designed_to_acceptor": v_designed,
        "v_b1_cryptic_to_acceptor": v_b1_to_acc,
        "designed_acceptor_fraction": v_designed / acc_total if acc_total else np.nan,
        "b1_cryptic_acceptor_fraction": v_b1_to_acc / acc_total if acc_total else np.nan,
        # whole-construct view
        "b1_cryptic_fraction_all": float(b1_vals.sum()) / total if total else np.nan,
        "n_b1_cryptic_junctions": int(from_b1.sum()),
        "n_b1_cryptic_donors": int(len(np.unique(starts[from_b1]))),
        "max_b1_cryptic_value": float(b1_vals.max()) if len(b1_vals) else 0.0,
        "argmax_b1_cryptic_donor": (int(starts[from_b1][int(np.argmax(b1_vals))]) - 1
                                    if len(b1_vals) else -1),
        # controls
        "middle_donor_designed_fraction": (v_md_designed / md_total
                                           if md_total else np.nan),
        "v_exon_skipping": float(vals[skip].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="select and validate constructs, make no call")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="hard ceiling on calls this invocation may spend")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--flush-every", type=int, default=25)
    ap.add_argument("--env-path", default=None)
    args = ap.parse_args()

    if not SHARE_CSV.exists():
        sys.exit(f"FATAL: {SHARE_CSV} not found")
    df = pd.read_csv(SHARE_CSV,
                     usecols=["row_id", "b1", "split", "AG_mean_logit"])
    sel = pick_subset(df, args.n, args.split, args.seed)
    print(f"selected {len(sel):,} of {(df.split == args.split).sum():,} "
          f"{args.split} rows, stratified by AG_mean_logit decile")

    exons = pd.read_csv(EXONS_CSV)
    syn = dict(zip(exons.sequence_id, exons.synthetic_70mer))[PRIMARY_EXON]

    # Rebuild and structurally validate every construct BEFORE any call.
    jobs = []
    for r in sel.itertuples():
        core, start = B.build_construct(r.b1, syn)
        fails = B.validate_construct(core, start, r.b1, syn,
                                     require_length=B.B1_LEN)
        if fails:
            sys.exit(f"FATAL {r.row_id}: structural validation failed: {fails}")
        if start != MIDDLE_ACCEPTOR:
            sys.exit(f"FATAL {r.row_id}: exon_start {start} != {MIDDLE_ACCEPTOR}")
        jobs.append({"row_id": r.row_id, "core": core, "exon_start": start,
                     "stored_mean_logit": float(r.AG_mean_logit)})
    print(f"all {len(jobs):,} constructs rebuilt and validated")

    done: set[str] = set()
    if MEASURE_CSV.exists():
        done = set(pd.read_csv(MEASURE_CSV, usecols=["row_id"]).row_id)
        print(f"resuming: {len(done):,} rows already measured")
    todo = [j for j in jobs if j["row_id"] not in done]
    budget = len(todo) if args.max_calls is None else min(len(todo), args.max_calls)
    print(f"{len(todo):,} remaining; this run will spend at most {budget:,} calls")

    if args.dry_run:
        print("--dry-run: no AlphaGenome call made")
        return 0
    if not budget:
        print("nothing to do")
        return 0

    import alphagenome_local_prediction_pipeline_minigene as ag
    import synthetic_context_lib as L
    from alphagenome.models import dna_client

    env = args.env_path
    if env is None:
        for cand in (ROOT / ".env.alphagenome", ROOT / ".env"):
            if cand.exists():
                env = str(cand)
                break
    model = ag.create_model(env_path=env)

    CHECK.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    spent = 0
    t0 = time.time()

    def flush():
        if not rows:
            return
        out = pd.DataFrame(rows)
        hdr = not MEASURE_CSV.exists()
        out.to_csv(MEASURE_CSV, mode="a", header=hdr, index=False)
        rows.clear()

    for job in todo[:budget]:
        padded, left_pad, _ = ag.center_pad(ag.clean_seq(job["core"]))
        acc_pos = left_pad + job["exon_start"]
        don_pos = acc_pos + 76 - 1
        pred = model.predict_sequence(
            sequence=padded,
            organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=[dna_client.OutputType.SPLICE_SITES,
                               dna_client.OutputType.SPLICE_JUNCTIONS],
            ontology_terms=list(ONTOLOGY),
        )
        spent += 1

        # Re-derive the dataset's own target from the same response.
        donor_t, acceptor_t = ag.get_positive_strand_tracks(pred)
        acc, don = float(acceptor_t[acc_pos]), float(donor_t[don_pos])
        mean_logit = (L._logit(acc) + L._logit(don)) / 2.0
        diff = abs(mean_logit - job["stored_mean_logit"])
        if diff > LOGIT_TOL:
            flush()
            sys.exit(
                f"FATAL {job['row_id']}: re-derived AG_mean_logit "
                f"{mean_logit:.9f} != stored {job['stored_mean_logit']:.9f} "
                f"(|diff| {diff:.2e}). The construct or scoring path differs "
                f"from the one that built the dataset; refusing to continue.")

        m = junction_metrics(pred.splice_junctions, left_pad)
        m.update(row_id=job["row_id"], AG_mean_logit=job["stored_mean_logit"],
                 recomputed_mean_logit=mean_logit,
                 AG_acceptor_probability=acc, AG_donor_probability=don)
        rows.append(m)

        if len(rows) >= args.flush_every:
            flush()
            rate = spent / max(1e-9, time.time() - t0)
            print(f"  {spent:,}/{budget:,} calls  {rate * 60:.0f}/min", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    flush()
    META_JSON.write_text(json.dumps(
        {"split": args.split, "n_requested": args.n, "seed": args.seed,
         "calls_this_run": spent, "ontology": list(ONTOLOGY),
         "b1_span": [B1_LO, B1_HI], "designed_start": DESIGNED_START,
         "middle_acceptor": MIDDLE_ACCEPTOR}, indent=2) + "\n")
    print(f"\nspent {spent:,} calls in {time.time() - t0:.0f} s -> {MEASURE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
