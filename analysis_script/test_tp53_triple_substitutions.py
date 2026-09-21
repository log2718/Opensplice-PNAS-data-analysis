"""
Lightweight, no-AlphaGenome, no-PNAS sanity checks for
``run_tp53_triple_substitutions.py``'s sampling / chunking / merge logic.

Not a pytest suite (this repo's analysis_script/ has none) -- a plain script
of small, fast assertions, runnable directly:

    python3 analysis_script/test_tp53_triple_substitutions.py

Covers exactly what was asked for and nothing else: uniqueness, distinct
positions, alt != ref, reproducibility from --seed, different seeds diverge,
chunk coverage is exact-once, and --merge catches duplicate/missing chunks.
No AlphaGenome or PNAS model is instantiated -- these tests only exercise
pure Python sampling/chunk-bounds logic and CSV-level merge validation with
hand-built fixture rows.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import pandas as pd  # noqa: E402

import run_tp53_triple_substitutions as triple  # noqa: E402

# A short synthetic exon is enough for the pure sampling/positions/alts
# checks -- no need to touch arush_data/wt_exons.csv for those.
_FAKE_EXON_SEQ = "ACGTACGTACGTACGTACGTACGTACGT"  # 28 nt, plenty of C(n,3)*27 space
_FAKE_EXON_ID = "FAKE_EXON"


def _fake_wt(exon_seq: str = _FAKE_EXON_SEQ) -> SimpleNamespace:
    return SimpleNamespace(
        exon_seq=exon_seq,
        upstream_flank="N" * 70,
        downstream_flank="N" * 25,
    )


# ==========================================================================
# Sampling checks
# ==========================================================================

def test_sampled_variants_unique() -> None:
    keys = triple.sample_triple_keys(_FAKE_EXON_SEQ, 500, seed=1, exon_id=_FAKE_EXON_ID)
    assert len(keys) == 500, f"expected 500 keys, got {len(keys)}"
    assert len(set(keys)) == 500, "sampled keys are not all unique"


def test_positions_distinct_and_sorted() -> None:
    keys = triple.sample_triple_keys(_FAKE_EXON_SEQ, 300, seed=2, exon_id=_FAKE_EXON_ID)
    for i, j, k, *_ in keys:
        assert i < j < k, f"positions not strictly increasing/distinct: {(i, j, k)}"
        assert 0 <= i and k < len(_FAKE_EXON_SEQ), f"position out of range: {(i, j, k)}"


def test_alts_differ_from_reference() -> None:
    keys = triple.sample_triple_keys(_FAKE_EXON_SEQ, 300, seed=3, exon_id=_FAKE_EXON_ID)
    for i, j, k, alt_i, alt_j, alt_k in keys:
        assert alt_i != _FAKE_EXON_SEQ[i], f"alt1 equals ref at pos {i}"
        assert alt_j != _FAKE_EXON_SEQ[j], f"alt2 equals ref at pos {j}"
        assert alt_k != _FAKE_EXON_SEQ[k], f"alt3 equals ref at pos {k}"
        for alt in (alt_i, alt_j, alt_k):
            assert alt in ("A", "C", "G", "T"), f"alt {alt!r} not a base"


def test_reproducible_with_same_seed() -> None:
    keys_a = triple.sample_triple_keys(_FAKE_EXON_SEQ, 400, seed=42, exon_id=_FAKE_EXON_ID)
    keys_b = triple.sample_triple_keys(_FAKE_EXON_SEQ, 400, seed=42, exon_id=_FAKE_EXON_ID)
    assert keys_a == keys_b, "same seed produced different samples"

    # Also reproducible across exons sharing the same nominal --seed value,
    # each exon's derived RNG seed is still a deterministic function of
    # (seed, exon_id).
    keys_c = triple.sample_triple_keys(_FAKE_EXON_SEQ, 400, seed=42, exon_id="OTHER_EXON")
    assert keys_a == triple.sample_triple_keys(_FAKE_EXON_SEQ, 400, seed=42, exon_id=_FAKE_EXON_ID)
    assert keys_a != keys_c, "different exon_id with same seed should derive a different RNG stream"


def test_different_seeds_produce_different_samples() -> None:
    keys_a = triple.sample_triple_keys(_FAKE_EXON_SEQ, 400, seed=42, exon_id=_FAKE_EXON_ID)
    keys_b = triple.sample_triple_keys(_FAKE_EXON_SEQ, 400, seed=43, exon_id=_FAKE_EXON_ID)
    assert keys_a != keys_b, "different seeds produced identical samples"
    overlap = len(set(keys_a) & set(keys_b))
    assert overlap < len(keys_a), "different seeds' samples are suspiciously identical"


def test_sample_size_exceeding_full_space_raises() -> None:
    tiny_seq = "ACGT"  # C(4,3)*27 = 108
    full = triple.total_triple_variants(len(tiny_seq))
    try:
        triple.sample_triple_keys(tiny_seq, full + 1, seed=1, exon_id="TINY")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when sample_size exceeds full space")


# ==========================================================================
# Chunking checks
# ==========================================================================

def test_chunk_coverage_exactly_once() -> None:
    for total, num_chunks in [(1000, 7), (50_000, 20), (13, 5), (1, 1)]:
        seen: set[int] = set()
        for chunk_index in range(num_chunks):
            start, end = triple.chunk_bounds(total, num_chunks, chunk_index)
            idx = set(range(start, end))
            assert not (idx & seen), f"overlap detected for total={total}, num_chunks={num_chunks}"
            seen |= idx
        assert seen == set(range(total)), (
            f"chunks don't cover [0, {total}) exactly once "
            f"(total={total}, num_chunks={num_chunks}); missing={sorted(set(range(total)) - seen)[:5]}"
        )


def test_chunk_bounds_rejects_bad_index() -> None:
    try:
        triple.chunk_bounds(100, 5, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range chunk_index")


# ==========================================================================
# Merge validation checks (CSV-level fixtures; no scoring)
# ==========================================================================

_REAL_EXON = "TP53_e7"


def _write_chunk_csv(out_dir: Path, exon_id: str, chunk_index: int, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{exon_id}_chunk_{chunk_index:03d}.csv", index=False)


def _variant_row(exon_id: str, key: tuple[int, int, int, str, str, str]) -> dict:
    i, j, k, alt_i, alt_j, alt_k = key
    return {
        "exon_id": exon_id, "is_reference": False, "mutation_region": "triple_exon",
        "pos1": i, "pos2": j, "pos3": k,
        "position1": i + 1, "position2": j + 1, "position3": k + 1,
        "ref1": "N", "alt1": alt_i, "ref2": "N", "alt2": alt_j, "ref3": "N", "alt3": alt_k,
        "exon_seq": "N", "pnas_pretuner": 0.0,
    }


def _reference_row(exon_id: str) -> dict:
    return {
        "exon_id": exon_id, "is_reference": True, "mutation_region": "reference",
        "pos1": None, "pos2": None, "pos3": None,
        "position1": None, "position2": None, "position3": None,
        "ref1": None, "alt1": None, "ref2": None, "alt2": None, "ref3": None, "alt3": None,
        "exon_seq": "N", "pnas_pretuner": 0.0, "pnas_pretuner_delta": 0.0,
    }


def test_merge_succeeds_on_clean_chunks() -> None:
    wt = triple.hybrid.load_opensplice_exon(_REAL_EXON)
    keys = triple.sample_triple_keys(wt.exon_seq, 6, seed=999, exon_id=_REAL_EXON)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        s0, e0 = triple.chunk_bounds(6, 2, 0)
        s1, e1 = triple.chunk_bounds(6, 2, 1)

        rows0 = [_reference_row(_REAL_EXON)] + [_variant_row(_REAL_EXON, key) for key in keys[s0:e0]]
        rows1 = [_variant_row(_REAL_EXON, key) for key in keys[s1:e1]]
        _write_chunk_csv(out_dir, _REAL_EXON, 0, rows0)
        _write_chunk_csv(out_dir, _REAL_EXON, 1, rows1)

        args = Namespace(
            exon=_REAL_EXON, csv=triple.DEFAULT_OPENSPLICE_CSV,
            sample_size=6, seed=999, out_dir=out_dir, num_chunks=2, force=True,
        )
        rc = triple.run_merge(args)
        assert rc == 0, "expected clean merge to succeed"

        merged = pd.read_csv(out_dir / f"{_REAL_EXON}_merged.csv")
        n_variants = (~merged["is_reference"].fillna(False).astype(bool)).sum()
        assert n_variants == 6, f"expected 6 merged variant rows, got {n_variants}"


def test_merge_detects_missing_chunk() -> None:
    wt = triple.hybrid.load_opensplice_exon(_REAL_EXON)
    keys = triple.sample_triple_keys(wt.exon_seq, 6, seed=999, exon_id=_REAL_EXON)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        s0, e0 = triple.chunk_bounds(6, 2, 0)
        rows0 = [_reference_row(_REAL_EXON)] + [_variant_row(_REAL_EXON, key) for key in keys[s0:e0]]
        _write_chunk_csv(out_dir, _REAL_EXON, 0, rows0)
        # chunk 1 is deliberately never written

        args = Namespace(
            exon=_REAL_EXON, csv=triple.DEFAULT_OPENSPLICE_CSV,
            sample_size=6, seed=999, out_dir=out_dir, num_chunks=2, force=True,
        )
        rc = triple.run_merge(args)
        assert rc != 0, "expected merge to fail with a missing chunk file"
        assert not (out_dir / f"{_REAL_EXON}_merged.csv").exists(), "merged CSV should not be written on failure"


def test_merge_detects_duplicate_keys() -> None:
    wt = triple.hybrid.load_opensplice_exon(_REAL_EXON)
    keys = triple.sample_triple_keys(wt.exon_seq, 6, seed=999, exon_id=_REAL_EXON)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        # Both chunks carry the FULL key set, i.e. every key is duplicated.
        rows0 = [_reference_row(_REAL_EXON)] + [_variant_row(_REAL_EXON, key) for key in keys]
        rows1 = [_variant_row(_REAL_EXON, key) for key in keys]
        _write_chunk_csv(out_dir, _REAL_EXON, 0, rows0)
        _write_chunk_csv(out_dir, _REAL_EXON, 1, rows1)

        args = Namespace(
            exon=_REAL_EXON, csv=triple.DEFAULT_OPENSPLICE_CSV,
            sample_size=6, seed=999, out_dir=out_dir, num_chunks=2, force=True,
        )
        rc = triple.run_merge(args)
        assert rc != 0, "expected merge to fail on duplicate keys"
        assert not (out_dir / f"{_REAL_EXON}_merged.csv").exists(), "merged CSV should not be written on failure"


# ==========================================================================
# Runner
# ==========================================================================

_TESTS = [
    test_sampled_variants_unique,
    test_positions_distinct_and_sorted,
    test_alts_differ_from_reference,
    test_reproducible_with_same_seed,
    test_different_seeds_produce_different_samples,
    test_sample_size_exceeding_full_space_raises,
    test_chunk_coverage_exactly_once,
    test_chunk_bounds_rejects_bad_index,
    test_merge_succeeds_on_clean_chunks,
    test_merge_detects_missing_chunk,
    test_merge_detects_duplicate_keys,
]


def main() -> int:
    n_pass = 0
    n_fail = 0
    for test in _TESTS:
        name = test.__name__
        try:
            test()
        except Exception:
            n_fail += 1
            print(f"FAIL: {name}")
            traceback.print_exc()
        else:
            n_pass += 1
            print(f"PASS: {name}")

    print(f"\n{n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
