"""
Unit tests for the upstream-exon (B1) experiment.

Run from the repository root::

    python -m pytest upstream_exon/tests/test_upstream.py -q

No AlphaGenome calls, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "upstream_exon"), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b1_dirichlet as G  # noqa: E402
import b1_lib as B  # noqa: E402
import baselines as BL  # noqa: E402
import splice_site_partition as SP  # noqa: E402
import synthetic_context_lib as L  # noqa: E402
from model_upstream import UpstreamExonModel  # noqa: E402


# ==========================================================================
# generator
# ==========================================================================

def test_alpha_counts_sum_exactly():
    for n in (1, 7, 100, 9999, 10_000):
        c = G.alpha_counts(n)
        assert sum(c.values()) == n
    c = G.alpha_counts(10_000)
    assert c[0.5] == 2000 and c[1.0] == 4000 and c[2.0] == 4000


def test_alpha_mixture_sums_to_one():
    assert abs(sum(G.ALPHA_MIX.values()) - 1.0) < 1e-12


def test_generated_sequences_are_valid_and_unique():
    recs, report = G.generate(300)
    assert len(recs) == 300
    assert report["produced"] == report["unique"] == 300
    for r in recs:
        assert len(r["b1"]) == G.B1_LEN
        assert not (set(r["b1"]) - set("ACGT"))
        assert G.sha(r["b1"]) == r["b1_sha256"]


def test_generation_is_deterministic():
    a, _ = G.generate(120, seed=1234)
    b, _ = G.generate(120, seed=1234)
    assert [r["b1"] for r in a] == [r["b1"] for r in b]


def test_different_seed_gives_different_sequences():
    a, _ = G.generate(60, seed=1)
    b, _ = G.generate(60, seed=2)
    assert {r["b1"] for r in a} != {r["b1"] for r in b}


def test_exclude_is_respected():
    first, _ = G.generate(40, seed=99)
    banned = {first[0]["b1"], first[5]["b1"]}
    recs, report = G.generate(40, seed=99, exclude=banned)
    assert not ({r["b1"] for r in recs} & banned)
    assert report["excluded_collisions"] >= len(banned)


def test_lower_alpha_is_more_compositionally_skewed():
    """alpha=0.5 should give more extreme compositions than alpha=2."""
    lo = [G.sequence_stats(G.sample_one(0.5, i)[0])["max_base_frac"]
          for i in range(300)]
    hi = [G.sequence_stats(G.sample_one(2.0, i)[0])["max_base_frac"]
          for i in range(300)]
    assert np.mean(lo) > np.mean(hi)


def test_sequence_stats_are_consistent():
    s = "AAAACCCGGT" * 11 + "AAA"
    st = G.sequence_stats(s)
    assert st["length"] == len(s)
    assert st["count_A"] + st["count_C"] + st["count_G"] + st["count_T"] == len(s)
    assert abs(st["gc"] - (s.count("G") + s.count("C")) / len(s)) < 1e-12
    assert st["max_homopolymer"] >= 4
    assert 0 <= st["entropy_bits"] <= 2.0


def test_max_homopolymer():
    assert G.max_homopolymer("ACGT") == 1
    assert G.max_homopolymer("AACCCGT") == 3
    assert G.max_homopolymer("TTTTTTT") == 7


# ==========================================================================
# grouped (v2) generator
# ==========================================================================

def test_group_counts_for_the_40k_design():
    g = G.group_counts(40_000)
    assert g == {0.5: 2000, 1.0: 4000, 2.0: 4000}
    assert sum(g.values()) * G.REPLICATES_PER_GROUP == 40_000


def test_group_counts_rejects_indivisible_totals():
    with pytest.raises(ValueError):
        G.group_counts(4001)
    with pytest.raises(ValueError):
        G.group_counts(40_000, reps=3)


def test_grouped_structure():
    recs, report = G.generate_grouped(400)
    df_groups = {}
    for r in recs:
        df_groups.setdefault(r["group_id"], []).append(r)
    assert len(recs) == 400
    assert report["groups"] == 100
    assert all(len(v) == 4 for v in df_groups.values())
    for v in df_groups.values():
        assert sorted(r["replicate"] for r in v) == [0, 1, 2, 3]


def test_replicates_share_one_probability_vector():
    recs, _ = G.generate_grouped(400)
    by_group = {}
    for r in recs:
        by_group.setdefault(r["group_id"], []).append(r)
    for v in by_group.values():
        for key in ("target_pA", "target_pC", "target_pG", "target_pT"):
            assert len({r[key] for r in v}) == 1


def test_replicates_differ_in_sequence():
    """Shared p, independent draws -- the sequences must not be identical."""
    recs, _ = G.generate_grouped(400)
    by_group = {}
    for r in recs:
        by_group.setdefault(r["group_id"], []).append(r["b1"])
    assert all(len(set(v)) == 4 for v in by_group.values())


def test_within_group_composition_is_tighter_than_between():
    """The point of the design: within a group composition is ~matched."""
    recs, _ = G.generate_grouped(2000)
    by_group = {}
    for r in recs:
        by_group.setdefault(r["group_id"], []).append(r["gc"])
    within = np.mean([np.std(v) for v in by_group.values()])
    between = np.std([np.mean(v) for v in by_group.values()])
    assert within < between / 2


def test_grouped_generation_is_deterministic():
    a, _ = G.generate_grouped(200, seed=777)
    b, _ = G.generate_grouped(200, seed=777)
    assert [r["b1"] for r in a] == [r["b1"] for r in b]
    assert [r["group_id"] for r in a] == [r["group_id"] for r in b]


def test_grouped_sequences_are_unique_and_valid():
    recs, report = G.generate_grouped(800)
    assert report["produced"] == report["unique"] == 800
    for r in recs:
        assert len(r["b1"]) == G.B1_LEN
        assert not (set(r["b1"]) - set("ACGT"))
        assert G.sha(r["b1"]) == r["b1_sha256"]


def test_grouped_excludes_prior_sequences():
    first, _ = G.generate_grouped(80, seed=31)
    banned = {first[0]["b1"], first[7]["b1"]}
    recs, report = G.generate_grouped(80, seed=31, exclude=banned)
    assert not ({r["b1"] for r in recs} & banned)
    assert report["excluded_collisions"] >= len(banned)


def test_v1_and_v2_seed_streams_differ():
    assert G.DATASET_SEED != G.GROUPED_SEED
    flat, _ = G.generate(200)
    grouped, _ = G.generate_grouped(200)
    assert not ({r["b1"] for r in flat} & {r["b1"] for r in grouped})


# ==========================================================================
# grouped splits
# ==========================================================================

def test_split_group_targets_are_exact():
    import build_b1_dataset_v2 as V2
    t = V2.split_group_targets()
    assert t[0.5] == {"train": 1350, "val": 150, "test": 500}
    assert t[1.0] == {"train": 2700, "val": 300, "test": 1000}
    assert t[2.0] == {"train": 2700, "val": 300, "test": 1000}
    seqs = {s: sum(t[a][s] for a in t) * G.REPLICATES_PER_GROUP
            for s in ("train", "val", "test")}
    assert seqs == {"train": 27_000, "val": 3_000, "test": 10_000}


def test_assign_group_splits_keeps_groups_together():
    import pandas as pd
    import build_b1_dataset_v2 as V2
    recs, _ = G.generate_grouped(40_000)
    df = V2.assign_group_splits(pd.DataFrame(recs))
    assert df.split.value_counts().to_dict() == V2.SPLIT_SEQUENCES
    assert set(df.groupby("group_id").split.nunique()) == {1}
    mix = df.groupby("split").alpha.value_counts(normalize=True).unstack()
    for s in ("train", "val", "test"):
        assert abs(mix.loc[s, 0.5] - 0.2) < 1e-9
        assert abs(mix.loc[s, 1.0] - 0.4) < 1e-9
        assert abs(mix.loc[s, 2.0] - 0.4) < 1e-9


# ==========================================================================
# construct assembly -- B1 only varies
# ==========================================================================

@pytest.fixture(scope="module")
def syn70():
    import pandas as pd
    lib = pd.read_csv(ROOT / "outputs" / "synthetic_context" /
                      "synthetic_exon_library.csv")
    return lib[lib.sequence_id == "syn_a5.0_00482"].iloc[0].synthetic_70mer


def test_only_b1_changes(syn70):
    native = SP.pnas_blocks8()
    variant = G.sample_one(1.0, 0)[0]
    core, start = B.build_construct(variant, syn70)
    assert not B.validate_construct(core, start, variant, syn70)
    assert len(core) == 1309

    pos = len(variant)
    for b in ("B2", "B3", "B4"):
        assert core[pos:pos + len(native[b])] == native[b]
        pos += len(native[b])
    assert core[pos:pos + 70] == syn70
    pos += 70
    for b in ("B5", "B6", "B7", "B8"):
        assert core[pos:pos + len(native[b])] == native[b]
        pos += len(native[b])
    assert pos == len(core)


def test_middle_exon_is_locatable(syn70):
    exon = L.EXON_PREFIX + syn70 + L.EXON_SUFFIX
    for i in range(5):
        v = G.sample_one(2.0, i)[0]
        core, start = B.build_construct(v, syn70)
        assert core[start:start + len(exon)] == exon
        assert core.count(syn70) == 1


def test_exon_start_is_independent_of_b1_content(syn70):
    starts = {B.build_construct(G.sample_one(1.0, i)[0], syn70)[1]
              for i in range(10)}
    assert len(starts) == 1


def test_native_b1_reproduces_the_canonical_construct(syn70):
    core, start = B.build_construct(B.native_b1(), syn70)
    pb = SP.pnas_blocks8()
    ref, ref_start = SP.assemble8("PPPPPPPP", syn70, pb, pb)
    assert core == ref and start == ref_start


def test_validate_construct_catches_tampering(syn70):
    v = G.sample_one(1.0, 3)[0]
    core, start = B.build_construct(v, syn70)
    broken = core[:len(v)] + "A" + core[len(v) + 1:]   # corrupt B2
    assert B.validate_construct(broken, start, v, syn70)
    assert B.validate_construct(core, start, v[:-1], syn70)  # wrong length


# ==========================================================================
# model
# ==========================================================================

def test_model_shapes_and_param_count():
    m = UpstreamExonModel(input_length=113)
    assert m.conv_out_length == 108
    assert m.position_bias_incl.shape == (20, 108)
    x = torch.zeros(5, 4, 113)
    assert m(x).shape == (5,)
    y, e = m(x, return_energy=True)
    assert y.shape == e.shape == (5,)
    # 2*(20*4*6+20) conv + 2*20*108 bias + 2 SumDiff + 2 head
    assert m.describe()["n_params"] == 1000 + 4320 + 2 + 2


def test_position_biases_start_at_zero():
    m = UpstreamExonModel()
    assert torch.allclose(m.position_bias_incl, torch.zeros(1))
    assert torch.allclose(m.position_bias_skip, torch.zeros(1))


def test_energy_is_not_astronomically_large_at_init():
    """Regression test: SumDiff sums ~2*F*(L-5) softplus terms."""
    m = UpstreamExonModel()
    x = torch.rand(8, 4, 113).round()
    e = m.energy(x)
    assert torch.isfinite(e).all()
    assert e.abs().max() < 50.0


def test_structure_branch_runs_and_crops():
    m = UpstreamExonModel(use_structure=True)
    x = torch.zeros(3, 4, 113)
    s = torch.zeros(3, 3, 113)
    w = torch.zeros(3, 1, 113)
    a_incl, a_skip = m.activations(x, s, w)
    assert a_incl.shape == (3, 28, 108)      # 20 seq + 8 struct filters
    assert m(x, s, w).shape == (3,)


def test_structure_branch_requires_its_inputs():
    m = UpstreamExonModel(use_structure=True)
    with pytest.raises(ValueError):
        m(torch.zeros(2, 4, 113))


def test_head_variants_all_produce_scalars():
    x = torch.rand(4, 4, 113).round()
    for head in ("linear", "residual_tuner", "identity"):
        m = UpstreamExonModel(head=head, use_batchnorm=False)
        assert m(x).shape == (4,)


def test_calibrate_head_centres_predictions():
    torch.manual_seed(0)
    m = UpstreamExonModel()
    x = torch.rand(64, 4, 113).round()
    y = torch.randn(64) * 0.7 + 3.8
    m.calibrate_head_(x, y)
    pred = m(x)
    assert abs(float(pred.mean()) - float(y.mean())) < 0.25


def test_gradients_flow_to_filters_and_biases():
    m = UpstreamExonModel()
    x = torch.rand(6, 4, 113).round()
    m(x).sum().backward()
    assert m.conv_incl.weight.grad is not None
    assert m.position_bias_incl.grad is not None
    assert torch.isfinite(m.conv_incl.weight.grad).all()


def test_regularizers_are_finite_and_nonnegative():
    m = UpstreamExonModel()
    assert float(m.l1_filters()) >= 0
    assert float(m.l2_position_smoothness()) == 0.0   # biases start flat
    with torch.no_grad():
        m.position_bias_incl.normal_()
    assert float(m.l2_position_smoothness()) > 0


def test_model_accepts_other_lengths():
    m = UpstreamExonModel(input_length=90)
    assert m(torch.zeros(2, 4, 90)).shape == (2,)
    with pytest.raises(ValueError):
        UpstreamExonModel(input_length=4)


# ==========================================================================
# baselines
# ==========================================================================

def test_kmer_counts_shape_and_total():
    seqs = ["ACGT" * 10, "AAAA" * 10]
    x = BL.kmer_counts(seqs, k=3)
    assert x.shape == (2, 64)
    assert x[0].sum() == len(seqs[0]) - 2


def test_constant_baseline_predicts_the_mean():
    b = BL.ConstantBaseline().fit(["A" * 113], np.array([1.0, 3.0]))
    assert abs(b.predict(["A" * 113, "C" * 113])[0] - 2.0) < 1e-6


def test_baselines_fit_and_predict():
    recs, _ = G.generate(60, seed=5)
    seqs = [r["b1"] for r in recs]
    y = np.array([r["gc"] for r in recs], dtype=float)
    for b in BL.all_baselines(k=3):
        b.fit(seqs, y)
        p = b.predict(seqs)
        assert p.shape == (60,) and np.isfinite(p).all()


def test_composition_baseline_recovers_gc():
    """Sanity: a GC target must be learnable from composition features."""
    recs, _ = G.generate(200, seed=11)
    seqs = [r["b1"] for r in recs]
    y = np.array([r["gc"] for r in recs], dtype=float)
    b = BL.composition_baseline().fit(seqs, y)
    assert np.corrcoef(b.predict(seqs), y)[0, 1] > 0.95


# ==========================================================================
# splits / leakage
# ==========================================================================

def test_assert_no_leakage_detects_a_shared_sequence():
    import pandas as pd
    import dataset_upstream as D
    df = pd.DataFrame({"b1_id": ["a", "b"], "b1_sha256": ["x", "x"],
                       "split": ["train", "test"]})
    with pytest.raises(ValueError):
        D.assert_no_leakage(df)
    df.loc[1, "b1_sha256"] = "y"
    D.assert_no_leakage(df)


def test_within_group_metrics_detects_order_signal():
    """A model that only tracks group means must score ~0 within groups."""
    import pandas as pd
    import train_upstream as T
    rng = np.random.default_rng(0)
    gid = np.repeat([f"g{i}" for i in range(50)], 4)
    gmean = np.repeat(rng.normal(0, 1, 50), 4)
    within = rng.normal(0, 0.3, 200)
    y = gmean + within

    only_group = pd.DataFrame({"group_id": gid, "y_true": y, "y_pred": gmean})
    m = T.within_group_metrics(only_group)
    assert abs(m["within_group_pearson_r"]) < 0.2

    perfect = pd.DataFrame({"group_id": gid, "y_true": y, "y_pred": y})
    m2 = T.within_group_metrics(perfect)
    assert m2["within_group_pearson_r"] > 0.99
    assert m2["within_group_sign_agreement"] > 0.99


def test_beyond_composition_metrics_runs():
    import pandas as pd
    import train_upstream as T
    recs, _ = G.generate_grouped(200)
    df = pd.DataFrame(recs)
    df["y_true"] = df.gc * 3.0 + np.random.default_rng(1).normal(0, .1, len(df))
    df["y_pred"] = df.y_true
    out = T.beyond_composition_metrics(df.head(160), df.tail(40))
    assert "composition_only_r2" in out
    assert "model_r2_on_composition_residual" in out


def test_metrics_by_composition_bins():
    import pandas as pd
    import train_upstream as T
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"gc": rng.uniform(0, 1, 200),
                       "y_true": rng.normal(0, 1, 200)})
    df["y_pred"] = df.y_true + rng.normal(0, .2, 200)
    t = T.metrics_by_composition(df, n_bins=5)
    assert len(t) == 5
    assert {"gc_bin", "gc_mid", "r2", "n"} <= set(t.columns)
    assert t.n.sum() == 200


def test_eval_loaders_preserve_manifest_order():
    import dataset_upstream as D
    import pandas as pd
    recs, _ = G.generate_grouped(80)
    df = pd.DataFrame(recs)
    df["split"] = "train"
    df[D.TARGET_COL] = np.arange(len(df), dtype=float)
    ds = D.build_datasets(df, splits=("train",))
    ld = D.make_loaders(ds, batch_size=16, shuffle_train=False)["train"]
    ys = np.concatenate([b[3].numpy() for b in ld])
    assert np.allclose(ys, df[D.TARGET_COL].to_numpy())


def test_encode_sequences_is_one_hot():
    import dataset_upstream as D
    x = D.encode_sequences(["ACGT" * 28 + "A"])
    assert x.shape == (1, 4, 113)
    assert np.allclose(x.sum(axis=1), 1.0)
    assert x[0, 0, 0] == 1.0 and x[0, 1, 1] == 1.0
