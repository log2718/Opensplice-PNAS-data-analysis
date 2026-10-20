"""
Dirichlet B1 (upstream exon body) generator for the full modeling experiment.

Sampling, per the finalized design:

    (pA, pC, pG, pT) ~ Dirichlet(alpha, alpha, alpha, alpha)
    113 nucleotides drawn i.i.d. from that categorical

with a mixture over alpha:  0.5 -> 20%,  1 -> 40%,  2 -> 40%.

This is deliberately NOT the ``_largest_remainder`` exact-count scheme used by
``synthetic_context_lib.generate_synthetic_library`` for middle exons: here the
realized composition fluctuates around the sampled probabilities, which is what
the design calls for.

No composition matching and no dinucleotide shuffling -- those were pilot
control arms and are not part of the training library.

Quality statistics are RECORDED, never used to filter. Any exclusion is
explicit, counted and reported.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable

import numpy as np

B1_LEN = 113
ALPHABET = "ACGT"
_ALPHA_ARR = np.array(list(ALPHABET))

#: alpha -> proportion of the library
ALPHA_MIX: dict[float, float] = {0.5: 0.20, 1.0: 0.40, 2.0: 0.40}

#: master seed for the v1 (flat) training library
DATASET_SEED = 20261020

#: master seed for the v2 (grouped) training library -- a different stream, so
#: v2 never reproduces v1 sequences by construction
GROUPED_SEED = 20261021

#: sequences drawn from each shared Dirichlet probability vector
REPLICATES_PER_GROUP = 4


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big") % 2**32)


def _largest_remainder(props: np.ndarray, total: int) -> np.ndarray:
    """Deterministic proportions -> integer counts summing to ``total``."""
    raw = props * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem:
        base[np.argsort(-(raw - base))[:rem]] += 1
    return base


def alpha_counts(n_total: int, mix: dict[float, float] | None = None
                 ) -> dict[float, int]:
    """Split ``n_total`` across alpha groups, exactly summing to n_total."""
    mix = mix or ALPHA_MIX
    alphas = sorted(mix)
    props = np.array([mix[a] for a in alphas], dtype=float)
    if not np.isclose(props.sum(), 1.0):
        raise ValueError(f"alpha mixture must sum to 1, got {props.sum()}")
    counts = _largest_remainder(props, n_total)
    return {a: int(c) for a, c in zip(alphas, counts)}


# ==========================================================================
# generation
# ==========================================================================

def sample_one(alpha: float, index: int, seed: int = DATASET_SEED,
               length: int = B1_LEN) -> tuple[str, np.ndarray]:
    """One Dirichlet-composition sequence and its sampled probability vector."""
    r = _rng(seed, "b1", alpha, index)
    probs = r.dirichlet([alpha] * 4)
    idx = r.choice(4, size=length, p=probs)
    return "".join(_ALPHA_ARR[idx]), probs


def generate(n_total: int, *, seed: int = DATASET_SEED,
             mix: dict[float, float] | None = None,
             length: int = B1_LEN,
             exclude: Iterable[str] = (),
             max_attempts_factor: int = 50) -> tuple[list[dict], dict]:
    """Generate the B1 library.

    Args:
        n_total: how many unique sequences to produce.
        seed: master seed; every sequence's RNG derives from it.
        mix: alpha -> proportion. Defaults to :data:`ALPHA_MIX`.
        length: sequence length (113).
        exclude: sequences that must not appear (e.g. the pilot panel), so the
            training library cannot leak into pilot comparisons.
        max_attempts_factor: give up after this many attempts per requested
            sequence.

    Returns:
        ``(records, report)``. Every exclusion is counted in ``report``; no
        sequence is ever dropped for being compositionally extreme.
    """
    want = alpha_counts(n_total, mix)
    banned = {s for s in exclude}
    seen: set[str] = set()
    records: list[dict] = []
    report = {
        "requested": n_total,
        "per_alpha_requested": dict(want),
        "duplicate_redraws": 0,
        "excluded_collisions": 0,
        "seed": seed,
        "length": length,
    }

    for alpha in sorted(want):
        made = 0
        attempts = 0
        limit = max(want[alpha] * max_attempts_factor, 1000)
        while made < want[alpha] and attempts < limit:
            seq, probs = sample_one(alpha, attempts, seed=seed, length=length)
            attempts += 1
            if seq in banned:
                report["excluded_collisions"] += 1
                continue
            if seq in seen:
                report["duplicate_redraws"] += 1
                continue
            seen.add(seq)
            records.append({
                "b1_id": f"b1_a{alpha}_{made:06d}",
                "alpha": alpha,
                "draw_index": attempts - 1,
                "b1": seq,
                "b1_sha256": sha(seq),
                "target_pA": float(probs[0]), "target_pC": float(probs[1]),
                "target_pG": float(probs[2]), "target_pT": float(probs[3]),
                **sequence_stats(seq),
            })
            made += 1
        if made < want[alpha]:
            raise RuntimeError(
                f"alpha={alpha}: only {made}/{want[alpha]} unique sequences "
                f"after {attempts} attempts")
        report[f"per_alpha_made_{alpha}"] = made

    report["produced"] = len(records)
    report["unique"] = len({r["b1"] for r in records})
    return records, report


# ==========================================================================
# grouped generation (v2 design)
# ==========================================================================

def group_counts(n_total: int, reps: int = REPLICATES_PER_GROUP,
                 mix: dict[float, float] | None = None) -> dict[float, int]:
    """Groups per alpha, given a total SEQUENCE count.

    ``n_total`` must divide evenly by ``reps`` and the per-alpha sequence
    counts must too, so that no group is left partially filled.
    """
    if n_total % reps:
        raise ValueError(f"n_total={n_total} is not divisible by reps={reps}")
    seqs = alpha_counts(n_total, mix)
    out = {}
    for a, n in seqs.items():
        if n % reps:
            raise ValueError(
                f"alpha={a}: {n} sequences is not divisible by reps={reps}; "
                "choose a total that splits evenly across the mixture")
        out[a] = n // reps
    return out


def generate_grouped(n_total: int, *, reps: int = REPLICATES_PER_GROUP,
                     seed: int = GROUPED_SEED,
                     mix: dict[float, float] | None = None,
                     length: int = B1_LEN,
                     exclude: Iterable[str] = (),
                     max_redraws: int = 1000) -> tuple[list[dict], dict]:
    """Grouped Dirichlet library: one probability vector, ``reps`` sequences.

    For each group a single ``p ~ Dirichlet(alpha, alpha, alpha, alpha)`` is
    drawn, then ``reps`` sequences of ``length`` nt are sampled independently
    from that same ``p``. The replicates therefore share the underlying
    nucleotide probabilities but not their exact realized counts.

    This is what replaces a separate composition-preserving arm: within a
    group, composition is matched in expectation, so any response difference
    among the four is attributable to nucleotide ORDER rather than content.

    Returns ``(records, report)``. Nothing is filtered on composition; the
    only redraws are for exact-duplicate or excluded sequences, and both are
    counted in ``report``.
    """
    want = group_counts(n_total, reps, mix)
    banned = set(exclude)
    seen: set[str] = set()
    records: list[dict] = []
    report = {
        "requested_sequences": n_total,
        "replicates_per_group": reps,
        "per_alpha_groups": dict(want),
        "duplicate_redraws": 0,
        "excluded_collisions": 0,
        "seed": seed,
        "length": length,
        "design": "grouped",
    }

    for alpha in sorted(want):
        for g in range(want[alpha]):
            r = _rng(seed, "group", alpha, g)
            probs = r.dirichlet([alpha] * 4)
            group_id = f"g_a{alpha}_{g:05d}"
            for k in range(reps):
                seq = None
                for _ in range(max_redraws):
                    cand = "".join(_ALPHA_ARR[r.choice(4, size=length, p=probs)])
                    if cand in banned:
                        report["excluded_collisions"] += 1
                        continue
                    if cand in seen:
                        report["duplicate_redraws"] += 1
                        continue
                    seq = cand
                    break
                if seq is None:  # pragma: no cover
                    raise RuntimeError(
                        f"{group_id} replicate {k}: no unique sequence after "
                        f"{max_redraws} redraws")
                seen.add(seq)
                records.append({
                    "b1_id": f"{group_id}_r{k}",
                    "group_id": group_id,
                    "replicate": k,
                    "alpha": alpha,
                    "b1": seq,
                    "b1_sha256": sha(seq),
                    "target_pA": float(probs[0]), "target_pC": float(probs[1]),
                    "target_pG": float(probs[2]), "target_pT": float(probs[3]),
                    "seed": seed,
                    **sequence_stats(seq),
                })

    report["produced"] = len(records)
    report["unique"] = len({r["b1"] for r in records})
    report["groups"] = len({r["group_id"] for r in records})
    return records, report


# ==========================================================================
# quality statistics (recorded, never used to filter)
# ==========================================================================

def max_homopolymer(seq: str) -> int:
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def shannon_entropy(seq: str) -> float:
    n = len(seq)
    c = Counter(seq)
    p = np.array([c.get(b, 0) / n for b in ALPHABET], dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sequence_stats(seq: str) -> dict:
    """Composition and complexity statistics for one sequence."""
    n = len(seq)
    counts = {b: seq.count(b) for b in ALPHABET}
    gc = (counts["G"] + counts["C"]) / n
    k3 = len({seq[i:i + 3] for i in range(n - 2)})
    k6 = len({seq[i:i + 6] for i in range(n - 5)})
    return {
        "count_A": counts["A"], "count_C": counts["C"],
        "count_G": counts["G"], "count_T": counts["T"],
        "freq_A": counts["A"] / n, "freq_C": counts["C"] / n,
        "freq_G": counts["G"] / n, "freq_T": counts["T"] / n,
        "gc": gc,
        "max_base_frac": max(counts.values()) / n,
        "max_homopolymer": max_homopolymer(seq),
        "distinct_3mers": k3,
        "distinct_6mers": k6,
        "entropy_bits": shannon_entropy(seq),
        "length": n,
    }


def qc_flags(rec: dict) -> dict:
    """Advisory flags. Recorded only -- nothing is filtered on them."""
    return {
        "flag_long_homopolymer": rec["max_homopolymer"] >= 10,
        "flag_extreme_base_frac": rec["max_base_frac"] >= 0.60,
        "flag_low_complexity": rec["distinct_3mers"] < 30,
        "flag_extreme_gc": not (0.20 <= rec["gc"] <= 0.80),
    }
