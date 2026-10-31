"""
Exact-composition (permutation) B1 generator for the v3 experiment.

Every sequence is a random permutation of the SAME multiset of nucleotides::

    29 A, 28 C, 28 G, 28 T   ->   113 nt,  GC = 56/113 = 0.4956

This is the control the v2 analysis asked for. In v2 the composition varied
(Dirichlet), and ~70% of the response variance turned out to be explained by
base frequencies alone -- a gradient-boosting model on three numbers beat both
the CNN and the 6-mer ridge. Holding the counts EXACTLY fixed removes that
axis by construction, so whatever signal remains is attributable to nucleotide
ORDER: dinucleotide structure, k-mers, and position.

Why 29/28/28/28 and not 28/28/28/29 or some other assignment: 113 is not
divisible by 4, so one base must take the extra copy. A is chosen because it
leaves GC exactly balanced between G and C (28 each), which keeps the GC
content of the library at a single value and avoids a G/C asymmetry that the
v2 model's dominant filter (a G-minus-C counter) could still read.

What this does NOT control:
  * dinucleotide frequencies -- those still vary, and SHOULD: they are part
    of the "order" signal the experiment is trying to measure. A library that
    also fixed dinucleotide counts would be a different, stricter experiment.
  * the rest of the construct, which is fixed in every arm already.

Sampling is a single deterministic RNG stream seeded from :data:`PERM_SEED`,
so the library is reproducible from the seed alone and never has to be
committed to git.

Makes NO AlphaGenome calls.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

B1_LEN = 113
ALPHABET = "ACGT"

#: exact nucleotide counts every sequence must have, summing to B1_LEN
TARGET_COUNTS: dict[str, int] = {"A": 29, "C": 28, "G": 28, "T": 28}

#: master seed for the v3 fixed-composition library (a different stream from
#: v1's DATASET_SEED and v2's GROUPED_SEED, so the libraries cannot coincide)
PERM_SEED = 20261031

_POOL = np.frombuffer(
    "".join(b * n for b, n in TARGET_COUNTS.items()).encode("ascii"),
    dtype="S1")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _check_pool() -> None:
    if sum(TARGET_COUNTS.values()) != B1_LEN:
        raise ValueError(
            f"TARGET_COUNTS sums to {sum(TARGET_COUNTS.values())}, not {B1_LEN}")
    if set(TARGET_COUNTS) != set(ALPHABET):
        raise ValueError("TARGET_COUNTS must cover exactly ACGT")


# ==========================================================================
# per-sequence statistics
# ==========================================================================

def max_homopolymer(seq: str) -> int:
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _constant_stats() -> dict:
    """Statistics that are identical for every sequence, by construction."""
    _check_pool()
    n = B1_LEN
    f = {b: TARGET_COUNTS[b] / n for b in ALPHABET}
    p = np.array(list(f.values()), dtype=float)
    return {
        **{f"count_{b}": TARGET_COUNTS[b] for b in ALPHABET},
        **{f"freq_{b}": f[b] for b in ALPHABET},
        "gc": f["G"] + f["C"],
        "max_base_frac": max(f.values()),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "length": n,
    }


CONSTANT_STATS = _constant_stats()


def variable_stats(seq: str) -> dict:
    """The only statistics that can differ between sequences here."""
    return {
        "max_homopolymer": max_homopolymer(seq),
        "distinct_3mers": len({seq[i:i + 3] for i in range(len(seq) - 2)}),
        "distinct_6mers": len({seq[i:i + 6] for i in range(len(seq) - 5)}),
    }


def exact_counts_ok(seq: str) -> bool:
    return all(seq.count(b) == TARGET_COUNTS[b] for b in ALPHABET)


# ==========================================================================
# generation
# ==========================================================================

def generate(n_total: int, *, seed: int = PERM_SEED,
             exclude: Iterable[str] = (),
             progress_every: int = 50_000) -> tuple[list[dict], dict]:
    """``n_total`` unique fixed-composition sequences.

    Args:
        n_total: how many sequences to produce.
        seed: master seed for the single RNG stream.
        exclude: sequences that must not appear (the pilot panel and the v1 /
            v2 libraries), so v3 cannot reuse an already-labeled sequence.
        progress_every: print a line every this many sequences.

    Returns ``(records, report)``. Collisions are redrawn and counted; nothing
    is ever filtered on sequence quality.
    """
    _check_pool()
    rng = np.random.default_rng(seed)
    banned = set(exclude)
    seen: set[str] = set()
    records: list[dict] = []
    report = {
        "design": "fixed_composition_permutation",
        "requested": n_total,
        "target_counts": dict(TARGET_COUNTS),
        "seed": seed,
        "length": B1_LEN,
        "duplicate_redraws": 0,
        "excluded_collisions": 0,
    }

    # The number of distinct permutations is 113!/(29!*28!^3) ~ 1e66, so
    # collisions are not expected; the loop handles them anyway rather than
    # assuming.
    while len(records) < n_total:
        seq = rng.permutation(_POOL).tobytes().decode("ascii")
        if seq in banned:
            report["excluded_collisions"] += 1
            continue
        if seq in seen:
            report["duplicate_redraws"] += 1
            continue
        seen.add(seq)
        i = len(records)
        records.append({
            "b1_id": f"b1p_{i:06d}",
            "draw_index": i,
            "b1": seq,
            "b1_sha256": sha(seq),
            "seed": seed,
            **CONSTANT_STATS,
            **variable_stats(seq),
        })
        if progress_every and len(records) % progress_every == 0:
            print(f"  generated {len(records):,}/{n_total:,}", flush=True)

    report["produced"] = len(records)
    report["unique"] = len({r["b1"] for r in records})
    return records, report


def library_fingerprint(seqs: Iterable[str]) -> str:
    """One digest over the ordered sequences.

    Lets a manifest regenerated on another machine be proven byte-identical
    to this one without shipping the 150 MB CSV through git.
    """
    h = hashlib.sha256()
    for s in seqs:
        h.update(s.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()
