"""
Unit tests for the v3 fixed-composition upstream-exon experiment.

Run from the repository root::

    python -m pytest upstream_exon/tests/test_upstream_v3.py -q

No AlphaGenome calls, no network, no GPU, and nothing written outside tmp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "upstream_exon"), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b1_lib as B  # noqa: E402
import b1_permutation as P  # noqa: E402
import baselines as BL  # noqa: E402
import build_b1_dataset_v3 as V3  # noqa: E402
import score_b1_dataset as SC  # noqa: E402


# ==========================================================================
# generator
# ==========================================================================

@pytest.fixture(scope="module")
def recs():
    r, _ = P.generate(400, progress_every=0)
    return r


def test_counts_sum_to_length():
    assert sum(P.TARGET_COUNTS.values()) == P.B1_LEN == 113
    assert set(P.TARGET_COUNTS) == set("ACGT")


def test_every_sequence_has_the_exact_target_composition(recs):
    for r in recs:
        for base, want in P.TARGET_COUNTS.items():
            assert r["b1"].count(base) == want
        assert len(r["b1"]) == P.B1_LEN
        assert P.exact_counts_ok(r["b1"])


def test_gc_and_entropy_are_constant(recs):
    assert len({r["gc"] for r in recs}) == 1
    assert len({r["entropy_bits"] for r in recs}) == 1
    assert recs[0]["gc"] == pytest.approx(56 / 113)


def test_order_statistics_actually_vary(recs):
    # if these did not vary there would be nothing for a model to learn
    assert len({r["max_homopolymer"] for r in recs}) > 1
    assert len({r["distinct_3mers"] for r in recs}) > 1


def test_sequences_are_unique_and_digests_match(recs):
    seqs = [r["b1"] for r in recs]
    assert len(set(seqs)) == len(seqs)
    assert all(P.sha(r["b1"]) == r["b1_sha256"] for r in recs)


def test_generation_is_deterministic():
    a, _ = P.generate(120, progress_every=0)
    b, _ = P.generate(120, progress_every=0)
    assert [r["b1"] for r in a] == [r["b1"] for r in b]


def test_a_different_seed_gives_a_different_library():
    a, _ = P.generate(50, progress_every=0)
    b, _ = P.generate(50, seed=P.PERM_SEED + 1, progress_every=0)
    assert [r["b1"] for r in a] != [r["b1"] for r in b]


def test_excluded_sequences_never_appear(recs):
    banned = {recs[0]["b1"], recs[5]["b1"]}
    out, report = P.generate(60, exclude=banned, progress_every=0)
    got = {r["b1"] for r in out}
    assert not (got & banned)
    assert report["excluded_collisions"] >= 1


def test_fingerprint_is_order_sensitive(recs):
    seqs = [r["b1"] for r in recs]
    assert P.library_fingerprint(seqs) == P.library_fingerprint(list(seqs))
    assert P.library_fingerprint(seqs) != P.library_fingerprint(seqs[::-1])


# ==========================================================================
# splits and pilot
# ==========================================================================

@pytest.fixture(scope="module")
def split_df(recs):
    df = pd.DataFrame(recs)
    sizes = {"train": 320, "val": 16, "test": 64}
    return V3.mark_pilot(V3.assign_splits(df, sizes), 10), sizes


def test_split_sizes_are_exact(split_df):
    df, sizes = split_df
    assert df.split.value_counts().to_dict() == sizes


def test_splits_are_disjoint(split_df):
    df, _ = split_df
    groups = {k: set(g.b1) for k, g in df.groupby("split")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (groups[a] & groups[b])


def test_split_assignment_is_deterministic(recs):
    sizes = {"train": 320, "val": 16, "test": 64}
    a = V3.assign_splits(pd.DataFrame(recs), sizes).split.tolist()
    b = V3.assign_splits(pd.DataFrame(recs), sizes).split.tolist()
    assert a == b


def test_split_sizes_must_sum_to_the_library(recs):
    with pytest.raises(ValueError):
        V3.assign_splits(pd.DataFrame(recs), {"train": 1, "val": 1, "test": 1})


def test_pilot_is_the_right_size_and_lives_in_train(split_df):
    df, _ = split_df
    assert int(df.pilot.sum()) == 10
    assert (df[df.pilot].split == "train").all()


def test_pilot_cannot_exceed_train(recs):
    df = V3.assign_splits(pd.DataFrame(recs),
                          {"train": 320, "val": 16, "test": 64})
    with pytest.raises(ValueError):
        V3.mark_pilot(df, 999)


def test_scaled_sizes_keep_the_total(recs):
    sizes, n_pilot = V3.scale_sizes(5000)
    assert sum(sizes.values()) == 5000
    assert n_pilot == 20
    assert V3.scale_sizes(V3.N_TOTAL) == (dict(V3.SPLIT_SEQUENCES), V3.N_PILOT)


# ==========================================================================
# constructs
# ==========================================================================

def test_construct_round_trips_for_a_fixed_composition_sequence(recs):
    syn70 = "A" * 70
    core, start = B.build_construct(recs[0]["b1"], syn70)
    assert B.validate_construct(core, start, recs[0]["b1"], syn70) == []
    assert len(core) == 1309
    assert core[:113] == recs[0]["b1"]        # B1 is the first block


# ==========================================================================
# baselines
# ==========================================================================

@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(7)
    seqs = ["".join(rng.choice(list("ACGT"), 60)) for _ in range(500)]
    x = BL.kmer_counts(seqs, 3)
    w = rng.normal(size=x.shape[1])
    y = (x - x.mean(0)) @ w * 0.05 + rng.normal(size=len(seqs)) * 0.1
    return seqs, y


def test_dinucleotide_features_are_frequencies(toy):
    seqs, _ = toy
    x = BL.dinucleotide_features(seqs)
    assert x.shape == (len(seqs), 16)
    assert np.allclose(x.sum(1), 1.0, atol=1e-5)
    assert x.std(0).max() > 0


def test_dinucleotide_varies_even_at_fixed_composition(recs):
    x = BL.dinucleotide_features([r["b1"] for r in recs])
    # this is the premise of the whole v3 experiment
    assert x.std(0).max() > 0


def test_composition_features_are_degenerate_at_fixed_composition(recs):
    """Why the composition ridge is replaced for v3, asserted rather than assumed."""
    x = BL.composition_features([r["b1"] for r in recs])
    # float32 rounding leaves ~1e-7 of noise in the constant columns; the one
    # genuinely varying column (max homopolymer) is four orders larger
    varying = x.std(0) > 1e-4
    assert varying.sum() == 1
    assert x.std(0)[:6].max() < 1e-5


def test_gram_ridge_matches_sklearn_ridge(toy):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    seqs, y = toy
    g = BL.GramRidgeBaseline(lambda s: BL.kmer_counts(s, 3), "t",
                             alphas=(10.0,), chunk=64)
    g.fit(seqs, y, val_seqs=seqs[:40], val_y=y[:40])
    x = BL.kmer_counts(seqs, 3)
    sc = StandardScaler().fit(x)
    r = Ridge(alpha=10.0).fit(sc.transform(x), y)
    assert np.abs(g.predict(seqs) - r.predict(sc.transform(x))).max() < 1e-4


def test_gram_ridge_is_chunk_size_invariant(toy):
    seqs, y = toy
    preds = []
    for chunk in (37, 128, 10_000):
        g = BL.GramRidgeBaseline(lambda s: BL.kmer_counts(s, 3), "t",
                                 alphas=(100.0,), chunk=chunk)
        g.fit(seqs, y, val_seqs=seqs[:40], val_y=y[:40])
        preds.append(g.predict(seqs))
    assert np.abs(preds[0] - preds[1]).max() < 1e-4
    assert np.abs(preds[0] - preds[2]).max() < 1e-4


def test_gram_ridge_selects_alpha_on_the_validation_split(toy):
    seqs, y = toy
    g = BL.GramRidgeBaseline(lambda s: BL.kmer_counts(s, 3), "t",
                             alphas=(1.0, 1e3, 1e6), chunk=200)
    g.fit(seqs, y, val_seqs=seqs[:60], val_y=y[:60])
    assert g.alpha_selected_on_ == "explicit validation split"
    assert len(g.alpha_path_) == 3
    assert g.alpha_ in (1.0, 1e3, 1e6)


def test_gram_ridge_falls_back_to_a_holdout(toy):
    seqs, y = toy
    g = BL.GramRidgeBaseline(lambda s: BL.kmer_counts(s, 2), "t",
                             alphas=(1.0, 1e3), chunk=200)
    g.fit(seqs, y)
    assert g.alpha_selected_on_ == "10% holdout of train"


def test_baseline_sets(recs):
    assert [b.name for b in BL.all_baselines_for("fixed_composition")] == [
        "constant", "dinucleotide_ridge", "kmer6_ridge"]
    assert [b.name for b in BL.all_baselines_for("dirichlet")] == [
        "constant", "composition_ridge", "kmer6_ridge"]
    with pytest.raises(ValueError):
        BL.all_baselines_for("nonsense")


# ==========================================================================
# scorer plumbing
# ==========================================================================

def test_stratified_head_falls_back_without_an_alpha_axis():
    df = pd.DataFrame({"b1_id": [f"x{i}" for i in range(50)]})
    assert len(SC.stratified_head(df, 10)) == 10
    df["alpha"] = 1.0                      # present but constant
    assert len(SC.stratified_head(df, 10)) == 10


def test_stratified_head_still_stratifies_when_alpha_varies():
    df = pd.DataFrame({"b1_id": [f"x{i}" for i in range(100)],
                       "alpha": [0.5] * 20 + [1.0] * 40 + [2.0] * 40})
    out = SC.stratified_head(df.sort_values("b1_id"), 20)
    assert len(out) <= 20
    assert out.alpha.nunique() == 3
