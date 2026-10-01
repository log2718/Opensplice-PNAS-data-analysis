"""
Variant generation for the OpenSplice/FAS-context AlphaGenome saturation
experiment.

Scientific question (Oded): why does AlphaGenome's response saturate at
different ``prod_logit`` levels for different OpenSplice/FAS exon contexts?
To probe that we push each exon's internal sequence hard with four motif
families at increasing copy number and look at where the AlphaGenome response
plateaus.

This module ONLY generates and validates variants -- it performs no scoring and
makes no API calls. Scoring lives in
``run_open_splice_context_saturation.py``.

Motif families
--------------
* ``UAG``      -- 3 nt (``TAG`` in DNA space, matching the repo-wide convention
                  of scoring transcript-sense sequence in a DNA alphabet),
                  with a REQUIRED minimum of 2 untouched nt between copies.
* ``CNNC``     -- 4 nt, drawn from exactly {CTAC, CATC, CAAC}, composition as
                  close to 1/3 each as possible, remainder rotated across
                  replicates so no single motif is systematically favoured.
* ``GGA``      -- 3 nt, non-overlapping (adjacent permitted).
* ``GGAGGAC``  -- 7 nt, a distinct family: each copy is the full 7-mer and is
                  counted as ONE copy, never as overlapping GGA motifs.

Placement
---------
Placements are sampled by an exact stars-and-bars construction rather than
rejection sampling, so a valid placement is always produced when one exists and
tight cells (e.g. 10x UAG in a 54-nt exon, which leaves only 6 nt of slack) do
not fail or bias toward loose packings. For ``k`` copies of an ``m``-mer with a
required gap ``g``::

    required = k*m + (k-1)*g
    slack    = L - required          # infeasible if negative

``slack`` is distributed uniformly at random over the ``k+1`` gaps (before,
between, after), which samples uniformly from all valid placements.

Feasibility
-----------
Maximum feasible copy number per (exon, family) is computed in closed form::

    UAG      k_max = (L + 2) // 5        # 3k + 2(k-1) <= L
    CNNC     k_max = L // 4
    GGA      k_max = L // 3
    GGAGGAC  k_max = L // 7

Requested levels above ``k_max`` are recorded as ``feasible=False`` with a
reason and are never generated with overlaps, reduced spacing, truncated
motifs, or a silently lowered copy number.

Only the internal target exon is ever modified. Exon length is invariant by
construction (substitution only), so the FAS scaffold, both intronic flanks and
all splice sites are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

EXONS = [
    "TP53_e6", "TP53_e7", "COL6A1_e9", "DMXL2_e36",
    "LDLR_e7", "COL6A1_e14", "GFPT1_e9", "MEGF8_e41",
]
COPY_LEVELS = [2, 4, 6, 8, 10, 12, 14]
N_REPLICATES = 100
DEFAULT_SEED = 20261001

CNNC_MOTIFS = ("CTAC", "CATC", "CAAC")

#: family -> (motif length, required untouched gap between copies)
FAMILY_SPEC: dict[str, tuple[int, int]] = {
    "UAG": (3, 2),
    "CNNC": (4, 0),
    "GGA": (3, 0),
    "GGAGGAC": (7, 0),
}

OUT_DIR = ROOT / "outputs" / "open_splice_context_saturation"


def max_copies(exon_length: int, family: str) -> int:
    """Largest k with k*m + (k-1)*g <= exon_length."""
    m, g = FAMILY_SPEC[family]
    return (exon_length + g) // (m + g)


def _sample_gaps(rng: np.random.Generator, slack: int, k: int) -> list[int]:
    """Uniformly sample k+1 non-negative integers summing to ``slack``.

    Standard stars-and-bars bijection: choose k bar positions out of slack+k.
    """
    if slack == 0:
        return [0] * (k + 1)
    bars = np.sort(rng.choice(slack + k, size=k, replace=False))
    gaps = [int(bars[0])]
    for i in range(1, k):
        gaps.append(int(bars[i] - bars[i - 1] - 1))
    gaps.append(int(slack + k - 1 - bars[-1]))
    return gaps


def _cnnc_composition(k: int, replicate_id: int) -> list[str]:
    """k motif identities, as close to 1/3 each as possible.

    The ``k % 3`` remainder is assigned to a window of motifs that rotates with
    ``replicate_id``, so across replicates no CNNC variant is systematically
    overrepresented.
    """
    base, rem = divmod(k, 3)
    counts = {mot: base for mot in CNNC_MOTIFS}
    for j in range(rem):
        counts[CNNC_MOTIFS[(replicate_id + j) % 3]] += 1
    out: list[str] = []
    for mot, c in counts.items():
        out.extend([mot] * c)
    return out


def place_motifs(
    exon_seq: str, family: str, k: int, rng: np.random.Generator,
    replicate_id: int,
) -> dict[str, Any] | None:
    """Build one variant. Returns None if (k, family) does not fit."""
    L = len(exon_seq)
    m, g = FAMILY_SPEC[family]
    required = k * m + (k - 1) * g
    slack = L - required
    if slack < 0:
        return None

    gaps = _sample_gaps(rng, slack, k)

    if family == "CNNC":
        motifs = _cnnc_composition(k, replicate_id)
        rng.shuffle(motifs)
    elif family == "UAG":
        motifs = ["TAG"] * k           # U -> T, repo-wide DNA-space convention
    else:
        motifs = [family] * k

    starts: list[int] = []
    pos = gaps[0]
    seq = list(exon_seq)
    for i in range(k):
        starts.append(pos)
        seq[pos:pos + m] = list(motifs[i])
        pos += m + (g if i < k - 1 else 0) + gaps[i + 1]

    mutated = "".join(seq)
    assert len(mutated) == L, "exon length changed -- generator bug"
    return {
        "mutated_exon_sequence": mutated,
        "motif_start_positions": ",".join(map(str, starts)),
        "motif_composition": ",".join(motifs),
    }


def generate_cell(
    exon_id: str, exon_seq: str, family: str, k: int, seed: int,
    n_replicates: int = N_REPLICATES, max_attempts_factor: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate up to ``n_replicates`` unique variants for one cell."""
    L = len(exon_seq)
    kmax = max_copies(L, family)
    if k > kmax:
        m, g = FAMILY_SPEC[family]
        return [], {
            "exon_id": exon_id, "exon_length": L, "motif_family": family,
            "copy_number": k, "feasible": False, "n_generated": 0,
            "max_feasible_copies": kmax,
            "reason": (f"{k} copies need {k*m + (k-1)*g} nt "
                       f"({m}-mer, {g}-nt required gap) > exon length {L}"),
        }

    # Deterministic per-cell stream so a cell reproduces independently of order
    # AND across processes. Python's built-in hash() is salted per interpreter
    # (PYTHONHASHSEED), which would silently hand the same variant_id different
    # sequences on a resumed run -- so derive the seed from a stable digest.
    digest = hashlib.blake2b(
        f"{seed}|{exon_id}|{family}|{k}".encode(), digest_size=8
    ).digest()
    cell_seed = int.from_bytes(digest, "big") % (2 ** 32)
    rng = np.random.default_rng(cell_seed)

    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    attempts = 0
    while len(records) < n_replicates and attempts < n_replicates * max_attempts_factor:
        attempts += 1
        v = place_motifs(exon_seq, family, k, rng, replicate_id=len(records))
        if v is None:
            break
        if v["mutated_exon_sequence"] in seen:
            continue
        seen.add(v["mutated_exon_sequence"])
        records.append({
            "variant_id": f"{exon_id}|{family}|{k}x|{len(records):03d}",
            "exon_id": exon_id, "exon_length": L,
            "motif_family": family, "copy_number": k,
            "replicate_id": len(records), "random_seed": cell_seed,
            "feasible": True, **v,
        })

    summary = {
        "exon_id": exon_id, "exon_length": L, "motif_family": family,
        "copy_number": k, "feasible": True, "n_generated": len(records),
        "max_feasible_copies": kmax, "attempts": attempts,
        "reason": "" if len(records) == n_replicates
                  else f"only {len(records)} unique variants reachable",
    }
    return records, summary


def build_all(seed: int = DEFAULT_SEED, n_replicates: int = N_REPLICATES):
    import run_hybrid_pnas_alphagenome as hybrid

    variants: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    feas_rows: list[dict[str, Any]] = []

    for exon_id in EXONS:
        wt = hybrid.load_opensplice_exon(exon_id, hybrid.DEFAULT_OPENSPLICE_CSV)
        L = wt.exon_length
        feas_rows.append({
            "exon_id": exon_id, "exon_length": L,
            **{f"{fam}_max": max_copies(L, fam) for fam in FAMILY_SPEC},
            **{f"{fam}_feasible_levels": ",".join(
                str(n) for n in COPY_LEVELS if n <= max_copies(L, fam)
            ) for fam in FAMILY_SPEC},
        })
        # WT reference row for every exon
        variants.append({
            "variant_id": f"{exon_id}|WT|0x|REF",
            "exon_id": exon_id, "exon_length": L,
            "motif_family": "WT", "copy_number": 0, "replicate_id": -1,
            "random_seed": seed, "feasible": True,
            "mutated_exon_sequence": wt.exon_seq,
            "motif_start_positions": "", "motif_composition": "",
        })
        for family in FAMILY_SPEC:
            for k in COPY_LEVELS:
                recs, summ = generate_cell(exon_id, wt.exon_seq, family, k,
                                           seed, n_replicates)
                variants.extend(recs)
                summaries.append(summ)

    return pd.DataFrame(variants), pd.DataFrame(summaries), pd.DataFrame(feas_rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate OpenSplice/FAS context-saturation variants (no scoring).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    variants, summary, feas = build_all(args.seed, args.n_replicates)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants.to_csv(args.out_dir / "variants_unscored.csv", index=False)
    summary.to_csv(args.out_dir / "variant_generation_summary.csv", index=False)
    feas.to_csv(args.out_dir / "feasibility_by_exon_motif.csv", index=False)

    n_wt = (variants.motif_family == "WT").sum()
    print(f"variants: {len(variants)} ({len(variants)-n_wt} mutants + {n_wt} WT reference rows)")
    print(f"feasible cells: {int(summary.feasible.sum())} / {len(summary)}")
    short = summary[(summary.feasible) & (summary.n_generated < args.n_replicates)]
    if len(short):
        print(f"\ncells short of {args.n_replicates} unique variants:")
        print(short[["exon_id","motif_family","copy_number","n_generated","reason"]].to_string(index=False))
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
