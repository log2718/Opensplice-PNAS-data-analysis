"""
Shared library for the synthetic-exon context experiments.

Two experiments are built on this module:

1. **Endpoints** -- 10,000 synthetic middle exons scored in three full contexts
   (PNAS reporter, TP53_e6/FAS, TP53_e7/FAS) to test whether the very different
   AlphaGenome saturation levels are caused by the surrounding context.
2. **Chimeras** -- all 2^6 block swaps between PNAS and each FAS context, on a
   balanced 400-exon subset, to identify which block(s) are responsible.

The synthetic exon is NEVER swapped: every construct carries the identical
``GTT + <70 synthetic nt> + CAG``.

Block order (six-bit label, left to right)
------------------------------------------
``L1 L2 L3 | exon | R3 R2 R1``  -- e.g. ``PPFFPP``

===  ==========================================================
L1   upstream outer exon / scaffold
     P: Exon1                           F: FAS_E5
L2   upstream DISTAL intron
     P: Intron1 minus its last 7 nt     F: FAS_I5 + upstream70[:63]
L3   proximal 3'SS intronic 7-mer
     P: CATCCAG                         F: upstream70[-7:]
R3   proximal 5'SS intronic 7-mer
     P: GTCTGAC                         F: downstream25[:7]
R2   downstream DISTAL intron
     P: Intron2 minus its first 7 nt    F: downstream25[7:] + FAS_I6
R1   downstream outer exon / scaffold
     P: Exon3                           F: FAS_E7
===  ==========================================================

The proximal splice-site 7-mers (L3, R3) are deliberately SEPARATE blocks from
the distal introns (L2, R2), so a splice-site swap can be distinguished from an
intron-body swap.

``PPPPPP`` reconstructs the full PNAS context byte-for-byte and ``FFFFFF``
reconstructs the corresponding full FAS context byte-for-byte; both identities
are asserted by :func:`validate`.

The PNAS context is EXACTLY the original 1,309-nt construct from
``run_alphagenome_pnas_fulltrack_sanity`` (Exon1 + Intron1 + exon + Intron2 +
Exon3). No extra upstream sequence is included anywhere in this experiment, so
values remain directly comparable to that earlier experiment.

Scoring definitions are reused unchanged: AlphaGenome ``SPLICE_SITES`` only,
middle-exon acceptor/donor read from the positive-strand tracks, and the same
``mean_logit`` / ``prod_logit`` formulas as every other experiment in the repo.
"""

from __future__ import annotations

import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ==========================================================================
# fixed sequences
# ==========================================================================

PNAS_EXON1 = (
    "gtaccgcaacctcaaacagacaccatggtgcacctgactcctgaggagaagtctgccgttactgcc"
    "ctgtggggcaaggtgaacgtggatgaagttggtggtgaggccctgggcag"
).upper()

PNAS_INTRON1 = (
    "gttggtatcaaggttacaagacaggtttaaggagaccaatagaaactgggcatatggagacagagaag"
    "actcttgggtttctgataggcactgactctctctgcctatgtctttctctgccatccag"
).upper()

PNAS_INTRON2 = (
    "gtctgactatgggacccttgatgttttctttccccttcttttctatggttaagttcatgtcataggaagggg"
    "agaagtaacagggtacagtttagaatgggaaacagacgaatgattgcatcagtgtggaagtctcaggatcgt"
    "tttagtttcttttatttgctgttcataacaattgttttcttttgtttaattcttgctttcttttttttcttc"
    "tccgcaatttttactattatacttaatgccttaacattgtgtataacaaaaggaaatatctctgagataca"
    "ttaagtaacttaaaaaaaaactttacacagtctgcctagtacattactatttggaatatatgtgtgcttat"
    "ttgcatattcataatctccctactttattttcttttatttttaattgatacataatcattatacatattta"
    "tgggttaaagtgtaatgttttaatatcgatacacatattgaccaaatcagggtaattttgcatttgtaatt"
    "ttaaaaaatgctttcttcttttaatatacttttttgtttatcttatttctaatactttccctaatctcttt"
    "ctttcagggcaataatgatacaatgtatcatgcctctttgcaccattctaaagaataacagtgataatttc"
    "tgggttaaggcaatagcaatatttctgcatataaatatttctgcatataaattgtaactgatgtaagaggt"
    "ttcatattgctaatagcagctacaatccagctaccattctgcttttattttatggttgggataaggctgga"
    "ttattctgagtccaagctaggcccttttgctaatcatgttcatacctcttatcttcctcccacag"
).upper()

PNAS_EXON3 = (
    "ctcctgggcaacgtgctggtctgtgtgctggcccatcactttggcaaagaattcaccccaccagtgcagg"
    "ctgcctatcagaaagtggtggctggtgtggctaatgccctggcccacaagtatcactaagctcgctctaga"
).upper()

EXON_PREFIX, EXON_SUFFIX = "GTT", "CAG"
SYN_LEN = 70
ALPHAS = (0.1, 0.5, 1.0, 5.0)
N_PER_ALPHA = 2500
DEFAULT_SEED = 20261007
SUBSET_PER_ALPHA = 100

BLOCK_ORDER = ("L1", "L2", "L3", "R3", "R2", "R1")

#: AlphaGenome SPLICE_SITES clip from the shared _logit(eps=1e-7)
CENSOR_VALUE = math.log(1e-7 / (1 - 1e-7))


def _logit(p: float, eps: float = 1e-7) -> float:
    """Identical convention to ``run_hybrid_pnas_alphagenome._logit``."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


# ==========================================================================
# A. synthetic exon library
# ==========================================================================

def _largest_remainder(props: np.ndarray, total: int) -> np.ndarray:
    """Deterministically turn proportions into integer counts summing to total."""
    raw = props * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem:
        order = np.argsort(-(raw - base))
        base[order[:rem]] += 1
    return base


def generate_synthetic_library(seed: int = DEFAULT_SEED,
                               n_per_alpha: int = N_PER_ALPHA) -> pd.DataFrame:
    """Dirichlet-composition synthetic 70-mers, ``n_per_alpha`` per regime.

    Sequences are deduplicated on the 70-mer; selection never depends on any
    PNAS or AlphaGenome score.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alpha in ALPHAS:
        rng = np.random.default_rng(
            int.from_bytes(hashlib.blake2b(f"{seed}|{alpha}".encode(),
                                           digest_size=8).digest(), "big") % 2**32)
        made, attempts = 0, 0
        while made < n_per_alpha and attempts < n_per_alpha * 200:
            attempts += 1
            props = rng.dirichlet([alpha] * 4)
            counts = _largest_remainder(props, SYN_LEN)
            bases = np.array(list("ACGT")).repeat(counts)
            rng.shuffle(bases)
            syn = "".join(bases)
            if syn in seen:
                continue
            seen.add(syn)
            rows.append({
                "sequence_id": f"syn_a{alpha}_{made:05d}",
                "alpha": alpha,
                "target_pA": props[0], "target_pC": props[1],
                "target_pG": props[2], "target_pT": props[3],
                "count_A": int(counts[0]), "count_C": int(counts[1]),
                "count_G": int(counts[2]), "count_T": int(counts[3]),
                "synthetic_70mer": syn,
                "full_middle_exon": EXON_PREFIX + syn + EXON_SUFFIX,
            })
            made += 1
    return pd.DataFrame(rows)


def balanced_subset(lib: pd.DataFrame, per_alpha: int = SUBSET_PER_ALPHA,
                    seed: int = DEFAULT_SEED) -> list[str]:
    """Deterministic balanced chimera subset; never uses any score."""
    ids: list[str] = []
    for alpha in ALPHAS:
        sub = lib[lib.alpha == alpha].sort_values("sequence_id")
        rng = np.random.default_rng(
            int.from_bytes(hashlib.blake2b(f"subset|{seed}|{alpha}".encode(),
                                           digest_size=8).digest(), "big") % 2**32)
        pick = rng.choice(len(sub), size=min(per_alpha, len(sub)), replace=False)
        ids.extend(sub.iloc[np.sort(pick)].sequence_id.tolist())
    return ids


# ==========================================================================
# F/G. context blocks + assembly
# ==========================================================================

def pnas_blocks() -> dict[str, str]:
    return {
        "L1": PNAS_EXON1,
        "L2": PNAS_INTRON1[:-7],
        "L3": PNAS_INTRON1[-7:],
        "R3": PNAS_INTRON2[:7],
        "R2": PNAS_INTRON2[7:],
        "R1": PNAS_EXON3,
    }


def fas_blocks(upstream70: str, downstream25: str) -> dict[str, str]:
    import alphagenome_local_prediction_pipeline_minigene as ag
    if len(upstream70) != 70 or len(downstream25) != 25:
        raise ValueError("FAS flanks must be 70 / 25 nt")
    return {
        "L1": ag.FAS_E5,
        "L2": ag.FAS_I5 + upstream70[:63],
        "L3": upstream70[-7:],
        "R3": downstream25[:7],
        "R2": downstream25[7:] + ag.FAS_I6,
        "R1": ag.FAS_E7,
    }


def assemble(label: str, exon: str, pb: dict[str, str],
             fb: dict[str, str]) -> tuple[str, int]:
    """Assemble a core from a six-character P/F label.

    Returns ``(core, exon_start)`` where ``exon_start`` is the 0-based index of
    the middle exon's first base within the unpadded core.
    """
    if len(label) != 6 or set(label) - set("PF"):
        raise ValueError(f"bad context label {label!r}")
    pick = {b: (pb if c == "P" else fb)[b] for b, c in zip(BLOCK_ORDER, label)}
    left = pick["L1"] + pick["L2"] + pick["L3"]
    right = pick["R3"] + pick["R2"] + pick["R1"]
    return left + exon + right, len(left)


def all_labels() -> list[str]:
    return [f"{i:06b}".replace("0", "P").replace("1", "F") for i in range(64)]


# ==========================================================================
# B/I. PNAS pre-tuner inputs
# ==========================================================================

def pnas_input(exon: str, up7: str, dn7: str) -> str:
    """``up7 + exon + dn7`` -- the exact 90-nt PNAS model input for a 76-nt exon."""
    return up7 + exon + dn7


PNAS_COMMON_UP7, PNAS_COMMON_DN7 = PNAS_INTRON1[-7:], PNAS_INTRON2[:7]


# ==========================================================================
# E. AlphaGenome scoring of an arbitrary assembled core
# ==========================================================================

def score_core(model, core: str, exon_start: int, exon_len: int,
               ontology_terms=("CL:0002518",)) -> dict[str, float]:
    """Pad to 16,384 and read middle-exon acceptor/donor from SPLICE_SITES.

    Uses the repo's own ``center_pad`` and ``get_positive_strand_tracks`` and
    the same prediction arguments as every other AlphaGenome experiment here.
    """
    import alphagenome_local_prediction_pipeline_minigene as ag
    from alphagenome.models import dna_client

    padded, left_pad, _ = ag.center_pad(ag.clean_seq(core))
    acc_pos = left_pad + exon_start
    don_pos = left_pad + exon_start + exon_len - 1

    pred = model.predict_sequence(
        sequence=padded,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_client.OutputType.SPLICE_SITES],
        ontology_terms=list(ontology_terms) if ontology_terms is not None else None,
    )
    donor_track, acceptor_track = ag.get_positive_strand_tracks(pred)
    acc, don = float(acceptor_track[acc_pos]), float(donor_track[don_pos])
    return {
        "AG_acceptor_probability": acc,
        "AG_donor_probability": don,
        "AG_mean_prob": (acc + don) / 2.0,
        "AG_mean_logit": (_logit(acc) + _logit(don)) / 2.0,
        "AG_prod_logit": _logit(acc * don),
        "core_length": len(core),
    }


# ==========================================================================
# K. resumable single-stream scoring loop
# ==========================================================================

def run_scoring(jobs: Iterable[dict[str, Any]], out_csv: Path, env_path: str,
                flush_every: int = 50, call_timeout: float = 90.0,
                max_retries: int = 3, failed_csv: Path | None = None) -> pd.DataFrame:
    """Score ``jobs`` (each with row_id / core / exon_start / exon_len) one at a
    time, checkpointing and resuming on ``row_id``.

    A single stream only -- AlphaGenome quota is not known to permit concurrent
    use. Each call runs in a worker thread with a watchdog, because the gRPC
    client retries errors but has no timeout for a call that never returns; on
    timeout the worker is abandoned and the client rebuilt.
    """
    import alphagenome_local_prediction_pipeline_minigene as ag

    jobs = list(jobs)
    done: dict[str, dict] = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = {r["row_id"]: r.to_dict() for _, r in prev.iterrows()}
        print(f"[resume] {len(done)} rows already scored in {out_csv.name}", flush=True)

    todo = [j for j in jobs if j["row_id"] not in done]
    total = len(todo)
    print(f"[scoring] {total} of {len(jobs)} rows to call", flush=True)

    records = list(done.values())
    failures: list[dict] = []
    model = ag.create_model(env_path=env_path)
    pool = ThreadPoolExecutor(max_workers=1)
    t0 = time.time()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(todo, 1):
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                fut = pool.submit(score_core, model, job["core"],
                                  job["exon_start"], job["exon_len"])
                result = fut.result(timeout=call_timeout)
                break
            except FutureTimeout:
                print(f"  [timeout {attempt}/{max_retries}] {job['row_id']} "
                      f"(>{call_timeout}s) -- rebuilding client", flush=True)
                pool.shutdown(wait=False)
                pool = ThreadPoolExecutor(max_workers=1)
                model = ag.create_model(env_path=env_path)
            except Exception as exc:
                print(f"  [error {attempt}/{max_retries}] {job['row_id']}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(2 ** attempt, 30))
        if result is None:
            failures.append({k: job[k] for k in job if k not in ("core",)})
            if failed_csv:
                pd.DataFrame(failures).to_csv(failed_csv, index=False)
            continue

        rec = {k: v for k, v in job.items() if k != "core"}
        rec.update(result)
        records.append(rec)

        if i % flush_every == 0 or i == total:
            pd.DataFrame(records).to_csv(out_csv, index=False)
            el = time.time() - t0
            rate = el / i
            print(f"  [{i}/{total}] {job['row_id']} | {rate:.2f}s/call | "
                  f"elapsed {el/60:.1f}m | ETA {(total-i)*rate/60:.1f}m | "
                  f"{len(records)} rows", flush=True)

    pd.DataFrame(records).to_csv(out_csv, index=False)
    pool.shutdown(wait=False)
    print(f"\n[done] {len(records)} rows -> {out_csv}", flush=True)
    if failures:
        print(f"[done] {len(failures)} failures -> {failed_csv}", flush=True)
    return pd.DataFrame(records)


# ==========================================================================
# M. validation
# ==========================================================================

FAS_TARGETS = {
    "TP53_e6": ("GGGCTGGAGAGACGACAGGGCTGGTTGCCCAGGGTCCCCAGGCCTCTGATTCCTCACTGATTGCTCTTAG",
                "GTCTGGTTTGCAACTGGGGTCTCTG"),
    "TP53_e7": ("GGCCTCCCCTGCTTGCCACAGGTCTCCCCAAGGCGCACTGGCCTCATCTTGGGCCTGTGTTATCTCCTAG",
                "GTCAGGAGCCACTTGCCACCCTGCA"),
}


def validate(lib: pd.DataFrame, verbose: bool = True) -> list[str]:
    """Run every structural check. Returns a list of failures (empty == OK)."""
    import alphagenome_local_prediction_pipeline_minigene as ag
    import run_hybrid_pnas_alphagenome as hybrid

    fails: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    # 1-4 library
    chk("1. all synthetic cores exactly 70 nt",
        bool((lib.synthetic_70mer.str.len() == SYN_LEN).all()))
    chk("2. full middle exon exactly 76 nt",
        bool((lib.full_middle_exon.str.len() == 76).all()))
    counts = lib.alpha.value_counts().to_dict()
    chk("3. 2500 sequences per alpha", all(counts.get(a) == N_PER_ALPHA for a in ALPHAS),
        str(counts))
    chk("4. no duplicate sequence_id / 70-mer",
        lib.sequence_id.is_unique and lib.synthetic_70mer.is_unique)
    chk("4b. exon = GTT + syn + CAG",
        bool((lib.full_middle_exon == EXON_PREFIX + lib.synthetic_70mer + EXON_SUFFIX).all()))
    chk("4c. ACGT only", bool(lib.synthetic_70mer.str.match(r"^[ACGT]+$").all()))

    exon = lib.full_middle_exon.iloc[0]

    # 5 PNAS_common input
    pi = pnas_input(exon, PNAS_COMMON_UP7, PNAS_COMMON_DN7)
    chk("5. PNAS_common input exactly 90 nt", len(pi) == 90, f"len={len(pi)}")
    chk("5b. PNAS_common 7-mers are CATCCAG / GTCTGAC",
        PNAS_COMMON_UP7 == "CATCCAG" and PNAS_COMMON_DN7 == "GTCTGAC",
        f"{PNAS_COMMON_UP7}/{PNAS_COMMON_DN7}")

    # 6 PNAS construct concatenation
    pb = pnas_blocks()
    # authoritative reference: the existing sanity script's own builder
    import run_alphagenome_pnas_fulltrack_sanity as PS
    manual = PS.build_construct(lib.synthetic_70mer.iloc[0])
    manual_core = manual["padded"][manual["left_pad"]:
                                    manual["left_pad"] + manual["core_length"]]
    core_p, start_p = assemble("PPPPPP", exon, pb, pb)
    chk("6/9. PPPPPP reconstructs the ORIGINAL 1,309-nt PNAS construct "
        "byte-for-byte (vs run_alphagenome_pnas_fulltrack_sanity)",
        core_p == manual_core and len(core_p) == 1309, f"core={len(core_p)}")
    chk("6b. exon_start locates the exon in PPPPPP",
        core_p[start_p:start_p + len(exon)] == exon)

    for tgt, (u70, d25) in FAS_TARGETS.items():
        w = hybrid.load_opensplice_exon(tgt, hybrid.DEFAULT_OPENSPLICE_CSV)
        chk(f"7. {tgt} flanks match wt_exons.csv",
            w.upstream_flank == u70 and w.downstream_flank == d25)
        fb = fas_blocks(u70, d25)
        manual_f = (ag.FAS_E5 + ag.FAS_I5 + u70 + exon + d25 + ag.FAS_I6 + ag.FAS_E7)
        core_f, start_f = assemble("FFFFFF", exon, pb, fb)
        chk(f"10. FFFFFF reconstructs the {tgt}/FAS context byte-for-byte",
            core_f == manual_f, f"core={len(core_f)}")
        # the repo pipeline's own builder must agree
        pipe = ag.build_fas_minigene(ag.build_variable_region(exon, u70, d25))
        chk(f"7b. {tgt} FFFFFF == repo build_fas_minigene output", core_f == pipe)
        chk(f"10b. {tgt} exon_start locates the exon in FFFFFF",
            core_f[start_f:start_f + len(exon)] == exon)
        # 8 no omitted/duplicated bases across every chimera
        bad = []
        for lab in all_labels():
            core, st = assemble(lab, exon, pb, fb)
            expect = sum(len((pb if c == "P" else fb)[b])
                         for b, c in zip(BLOCK_ORDER, lab)) + len(exon)
            if len(core) != expect or core[st:st + len(exon)] != exon:
                bad.append(lab)
        chk(f"8. {tgt}: all 64 chimeras concatenate with no lost/duplicated bases",
            not bad, f"{len(bad)} bad")
        # 13 contextual 7-mers
        l3 = {"P": pb["L3"], "F": fb["L3"]}
        r3 = {"P": pb["R3"], "F": fb["R3"]}
        ok13 = all(len(v) == 7 for v in (*l3.values(), *r3.values()))
        chk(f"13. {tgt}: L3/R3 blocks are 7 nt", ok13,
            f"L3 P={l3['P']} F={l3['F']} | R3 P={r3['P']} F={r3['F']}")
        # 14 padding
        core, st = assemble("PFPFPF", exon, pb, fb)
        padded, lp, rp = ag.center_pad(core)
        chk(f"14. {tgt}: padded length is 16,384", len(padded) == 16384,
            f"core={len(core)} left={lp} right={rp}")

    # 11 all 64 unique
    labs = all_labels()
    chk("11. 64 unique block combinations", len(labs) == 64 and len(set(labs)) == 64)
    chk("11b. label endpoints present", "PPPPPP" in labs and "FFFFFF" in labs)
    return fails
