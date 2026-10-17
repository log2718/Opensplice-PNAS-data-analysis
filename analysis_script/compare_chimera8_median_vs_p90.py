"""
Median vs p90: effect comparison with a paired, stratified bootstrap.

Part 5 of the p90 follow-up. Compares the FACTORIAL EFFECT ESTIMATES and SHAP
importances between the two context statistics, rather than their overall
levels (p90 is trivially higher than the median and that is not the question).

Makes NO AlphaGenome calls and fits no surrogate model.

Bootstrap design
----------------
The 400-exon subset is balanced across four Dirichlet strata (100 each), so
the resample is STRATIFIED: 100 sequence IDs drawn with replacement within
each alpha stratum. The SAME resampled IDs are used in

  * every one of the 512 contexts,
  * both arms (the two arms share the identical 400 IDs), and
  * both statistics,

so median-vs-p90 and e7-vs-e6 contrasts are fully paired and the bootstrap
reflects only sampling of the synthetic exon library.

Three algebraic identities make 2,000 replicates cheap and exact. With
``S = 2X - 1`` the +/-1 coded design over the 256 contexts:

  main_effect_i   = (S[:, i] . y) / 128
  interaction_ij  = (S[:, i] * S[:, j] . y) / 64         (i != j)
  exact SHAP      = A @ y, where A is the Shapley operator built once by
                    applying the exact coalition enumeration to each basis
                    vector -- it depends only on the design, not on y

All three were verified against the saved median analysis before use.

Inference
---------
Percentile 95% CIs on each paired difference, plus a two-sided bootstrap
p-value ``2 * min(P(delta* <= 0), P(delta* >= 0))`` with the usual (1 + count)
/ (B + 1) correction, then Benjamini-Hochberg across the declared family of
72 effect comparisons (8 main + 28 pairwise, per arm). Nominal-CI findings and
FDR-adjusted findings are reported separately.

The bootstrap measures uncertainty over the sampled synthetic exon library.
It is not biological replication, and a significant contrast is a statement
about AlphaGenome's behaviour on these constructs.

Usage::

    python analysis_script/compare_chimera8_median_vs_p90.py
    python analysis_script/compare_chimera8_median_vs_p90.py --n-boot 200
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import analyze_chimera8_shap as SH          # noqa: E402

NEW = ROOT / "outputs" / "synthetic_context_8block"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
MED_FACT, MED_SHAP = NEW / "factorial", NEW / "shap"
MED_TREE = NEW / "decision_tree"
P90_FACT, P90_SHAP = NEW / "p90" / "factorial", NEW / "p90" / "shap"
P90_TREE = NEW / "p90" / "decision_tree"
CMP = NEW / "median_vs_p90"
P_CMP = NEW / "plots" / "median_vs_p90"

Y = "AG_mean_logit"
ARMS = ["TP53_e6", "TP53_e7"]
ARM_SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
BLOCKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
BLOCK_SHORT = {
    "B1": "B1 Up exon body", "B2": "B2 Up 5'SS", "B3": "B3 Up intron",
    "B4": "B4 Mid 3'SS", "B5": "B5 Mid 5'SS", "B6": "B6 Down intron",
    "B7": "B7 Down 3'SS", "B8": "B8 Down exon body",
}
PAIRS = [(i, j) for i, j in itertools.combinations(range(8), 2)]
N_BOOT = 2000
SEED = 20261017
FDR_Q = 0.05

ARM_COLOR = {"TP53_e6": "#f0c000", "TP53_e7": "#2f6fd0"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
DIVERGING = LinearSegmentedColormap.from_list(
    "effect_div", ["#1b4f9c", "#2a78d6", "#9dc3ec", "#f2f1ed",
                   "#f7b48f", "#eb6834", "#a83c12"])


# ==========================================================================
# data
# ==========================================================================

def load_matrix(d: pd.DataFrame) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    """(256 contexts x 400 exons) per arm, with a shared exon column order."""
    labels = sorted(d.context_label_8.unique())
    seqs = sorted(d[d.comparison == ARMS[0]].sequence_id.unique())
    idx = {s: i for i, s in enumerate(seqs)}
    lab_idx = {l: i for i, l in enumerate(labels)}

    mats = {}
    for arm in ARMS:
        M = np.full((len(labels), len(seqs)), np.nan)
        s = d[d.comparison == arm]
        M[s.context_label_8.map(lab_idx).to_numpy(),
          s.sequence_id.map(idx).to_numpy()] = s[Y].to_numpy()
        if np.isnan(M).any():
            raise SystemExit(f"[abort] {arm}: incomplete context x exon matrix")
        mats[arm] = M

    alpha = (d[d.comparison == ARMS[0]]
             .drop_duplicates("sequence_id").set_index("sequence_id")
             .loc[seqs, "alpha"].to_numpy())
    X = np.array([[1 if l[i] == "F" else 0 for i in range(8)] for l in labels],
                 dtype=float)
    return mats, labels, X, alpha


def effects_from_y(y: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8 main effects and 28 pairwise interactions, as linear functionals."""
    main = (S.T @ y) / 128.0
    M = (S.T @ (S * y[:, None])) / 64.0
    inter = np.array([M[i, j] for i, j in PAIRS])
    return main, inter


def shap_operator() -> np.ndarray:
    """A with phi = A @ y. Depends only on the design, so built once."""
    A = np.zeros((8, 256, 256))
    for k in range(256):
        e = np.zeros(256)
        e[k] = 1.0
        f = e.reshape((2,) * 8)
        A[:, :, k] = SH.exact_shap(f, SH.conditional_means(f)).reshape(8, 256)
    return A


def bh_fdr(p: np.ndarray, q: float = FDR_Q) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg: adjusted p-values and the reject flags at level q."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out, out <= q


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    CMP.mkdir(parents=True, exist_ok=True)
    P_CMP.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(RESULTS_CSV,
                    usecols=["comparison", "context_label_8", "sequence_id",
                             "alpha", Y], low_memory=False)
    mats, labels, X, alpha = load_matrix(d)
    S = 2 * X - 1
    print(f"context x exon matrices: "
          f"{ {ARM_SLUG[a]: mats[a].shape for a in ARMS} }")

    strata = sorted(set(alpha))
    strat_idx = [np.where(alpha == a)[0] for a in strata]
    print(f"alpha strata: { {float(a): int(len(i)) for a, i in zip(strata, strat_idx)} }")

    # ---- observed point estimates ---------------------------------------
    obs = {}
    for arm in ARMS:
        M = mats[arm]
        y_med = np.median(M, axis=1)
        y_p90 = np.quantile(M, 0.90, axis=1)
        obs[arm] = {"median": effects_from_y(y_med, S),
                    "p90": effects_from_y(y_p90, S),
                    "y_median": y_med, "y_p90": y_p90}

    # cross-check against the saved analyses before going further
    worst = 0.0
    for arm in ARMS:
        slug = ARM_SLUG[arm]
        for stat, path in (("median", MED_FACT / f"{slug}_main_pairwise_effects.csv"),
                           ("p90", P90_FACT / f"{slug}_p90_main_pairwise_effects.csv")):
            ref = pd.read_csv(path)
            r1 = ref[ref.order == 1].set_index("term").effect.reindex(BLOCKS).to_numpy()
            worst = max(worst, float(np.abs(obs[arm][stat][0] - r1).max()))
            r2 = ref[ref.order == 2].set_index("term").effect
            got = {f"{BLOCKS[i]}:{BLOCKS[j]}": v
                   for (i, j), v in zip(PAIRS, obs[arm][stat][1])}
            worst = max(worst, max(abs(r2[k] - got[k]) for k in got))
    print(f"recomputed effects match the saved median and p90 tables to "
          f"{worst:.3e}")

    A = shap_operator()
    print(f"exact-SHAP operator built {A.shape}")

    # ---- paired stratified bootstrap ------------------------------------
    B = args.n_boot
    rng = np.random.default_rng(SEED)
    n_main, n_int = 8, len(PAIRS)
    boot = {arm: {"d_main": np.zeros((B, n_main)),
                  "d_int": np.zeros((B, n_int)),
                  "d_shap": np.zeros((B, 8))} for arm in ARMS}
    boot_arm = {"p90_main": np.zeros((B, n_main)),
                "p90_int": np.zeros((B, n_int))}

    t0 = time.time()
    for b in range(B):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in strat_idx])
        per_arm_p90 = {}
        for arm in ARMS:
            Mb = mats[arm][:, take]                 # identical IDs everywhere
            y_med = np.median(Mb, axis=1)
            y_p90 = np.quantile(Mb, 0.90, axis=1)
            m_med, i_med = effects_from_y(y_med, S)
            m_p90, i_p90 = effects_from_y(y_p90, S)
            boot[arm]["d_main"][b] = m_p90 - m_med
            boot[arm]["d_int"][b] = i_p90 - i_med
            shap_med = np.abs(A @ y_med).mean(axis=1)
            shap_p90 = np.abs(A @ y_p90).mean(axis=1)
            boot[arm]["d_shap"][b] = shap_p90 - shap_med
            per_arm_p90[arm] = (m_p90, i_p90)
        boot_arm["p90_main"][b] = per_arm_p90["TP53_e7"][0] - per_arm_p90["TP53_e6"][0]
        boot_arm["p90_int"][b] = per_arm_p90["TP53_e7"][1] - per_arm_p90["TP53_e6"][1]
        if (b + 1) % 500 == 0:
            print(f"  [{b + 1}/{B}] {time.time() - t0:.1f}s", flush=True)
    print(f"bootstrap done in {time.time() - t0:.1f}s")

    def summarise(draws: np.ndarray, point: np.ndarray, names: list[str],
                  arm: str, kind: str) -> pd.DataFrame:
        lo = np.percentile(draws, 2.5, axis=0)
        hi = np.percentile(draws, 97.5, axis=0)
        n = draws.shape[0]
        p_le = (1 + (draws <= 0).sum(axis=0)) / (n + 1)
        p_ge = (1 + (draws >= 0).sum(axis=0)) / (n + 1)
        pval = np.minimum(1.0, 2 * np.minimum(p_le, p_ge))
        return pd.DataFrame({
            "comparison": arm, "kind": kind, "term": names,
            "point_change": point, "boot_mean_change": draws.mean(axis=0),
            "boot_SE": draws.std(axis=0, ddof=1),
            "ci95_low": lo, "ci95_high": hi,
            "ci_excludes_zero": (lo > 0) | (hi < 0),
            "p_two_sided": pval,
        })

    pair_names = [f"{BLOCKS[i]}:{BLOCKS[j]}" for i, j in PAIRS]
    frames = []
    for arm in ARMS:
        dm = obs[arm]["p90"][0] - obs[arm]["median"][0]
        di = obs[arm]["p90"][1] - obs[arm]["median"][1]
        frames.append(summarise(boot[arm]["d_main"], dm, list(BLOCKS), arm,
                                "main_effect_change"))
        frames.append(summarise(boot[arm]["d_int"], di, pair_names, arm,
                                "interaction_change"))
    changes = pd.concat(frames, ignore_index=True)
    # BH across the declared family: 72 effect comparisons
    adj, rej = bh_fdr(changes.p_two_sided.to_numpy(), FDR_Q)
    changes["p_BH_adjusted"] = adj
    changes["significant_FDR_5pct"] = rej
    changes["family_size"] = len(changes)
    changes.to_csv(CMP / "median_vs_p90_effect_changes.csv", index=False)

    # SHAP importance changes (bootstrapped, same resamples)
    shap_frames = []
    for arm in ARMS:
        slug = ARM_SLUG[arm]
        med_s = pd.read_csv(MED_SHAP / f"{slug}_shap_feature_summary.csv"
                            ).set_index("block").mean_abs_SHAP.reindex(BLOCKS)
        p90_s = pd.read_csv(P90_SHAP / f"{slug}_p90_shap_feature_summary.csv"
                            ).set_index("block").mean_abs_SHAP.reindex(BLOCKS)
        pt = (p90_s - med_s).to_numpy()
        f = summarise(boot[arm]["d_shap"], pt, list(BLOCKS), arm,
                      "mean_abs_SHAP_change")
        f["mean_abs_SHAP_median"] = med_s.to_numpy()
        f["mean_abs_SHAP_p90"] = p90_s.to_numpy()
        shap_frames.append(f)
    shap_changes = pd.concat(shap_frames, ignore_index=True)
    adj, rej = bh_fdr(shap_changes.p_two_sided.to_numpy(), FDR_Q)
    shap_changes["p_BH_adjusted"] = adj
    shap_changes["significant_FDR_5pct"] = rej
    shap_changes.to_csv(CMP / "median_vs_p90_shap_importance_changes.csv",
                        index=False)

    # e7 - e6 contrasts on the p90 scale (paired: same exons in both arms)
    arm_pt_main = obs["TP53_e7"]["p90"][0] - obs["TP53_e6"]["p90"][0]
    arm_pt_int = obs["TP53_e7"]["p90"][1] - obs["TP53_e6"]["p90"][1]
    arm_cmp = pd.concat([
        summarise(boot_arm["p90_main"], arm_pt_main, list(BLOCKS),
                  "e7_minus_e6", "p90_main_effect_arm_difference"),
        summarise(boot_arm["p90_int"], arm_pt_int, pair_names,
                  "e7_minus_e6", "p90_interaction_arm_difference"),
    ], ignore_index=True)
    adj, rej = bh_fdr(arm_cmp.p_two_sided.to_numpy(), FDR_Q)
    arm_cmp["p_BH_adjusted"] = adj
    arm_cmp["significant_FDR_5pct"] = rej
    arm_cmp.to_csv(CMP / "p90_e7_minus_e6_contrasts.csv", index=False)

    # ---- side-by-side effect tables -------------------------------------
    rows = []
    for arm in ARMS:
        for k, b in enumerate(BLOCKS):
            rows.append({"comparison": arm, "kind": "main_effect", "term": b,
                         "median": obs[arm]["median"][0][k],
                         "p90": obs[arm]["p90"][0][k],
                         "change": obs[arm]["p90"][0][k] - obs[arm]["median"][0][k]})
        for k, nm in enumerate(pair_names):
            rows.append({"comparison": arm, "kind": "interaction", "term": nm,
                         "median": obs[arm]["median"][1][k],
                         "p90": obs[arm]["p90"][1][k],
                         "change": obs[arm]["p90"][1][k] - obs[arm]["median"][1][k]})
    side = pd.DataFrame(rows)
    side = side.merge(changes[["comparison", "term", "ci95_low", "ci95_high",
                               "ci_excludes_zero", "p_two_sided",
                               "p_BH_adjusted", "significant_FDR_5pct"]],
                      on=["comparison", "term"], how="left")
    side["sign_flip"] = np.sign(side["median"]) != np.sign(side["p90"])
    side.to_csv(CMP / "median_vs_p90_side_by_side.csv", index=False)

    top = side.reindex(side.change.abs().sort_values(ascending=False).index)
    top.head(25).to_csv(CMP / "median_vs_p90_largest_changes.csv", index=False)

    # ---- figures ---------------------------------------------------------
    def eff_matrix(main: np.ndarray, inter: np.ndarray) -> np.ndarray:
        M = np.zeros((8, 8))
        for k in range(8):
            M[k, k] = main[k]
        for (i, j), v in zip(PAIRS, inter):
            M[i, j] = M[j, i] = v
        return M

    delta = {a: eff_matrix(obs[a]["p90"][0] - obs[a]["median"][0],
                           obs[a]["p90"][1] - obs[a]["median"][1])
             for a in ARMS}
    vmax = float(max(np.abs(m).max() for m in delta.values()))
    sig = {a: {(r.term): bool(r.significant_FDR_5pct)
               for _, r in changes[changes.comparison == a].iterrows()}
           for a in ARMS}
    for arm in ARMS:
        fig, ax = plt.subplots(figsize=(9.6, 8.4), dpi=190)
        im = ax.imshow(delta[arm], cmap=DIVERGING,
                       norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        for i in range(8):
            for j in range(8):
                v = delta[arm][i, j]
                term = (BLOCKS[i] if i == j
                        else f"{BLOCKS[min(i, j)]}:{BLOCKS[max(i, j)]}")
                star = "*" if sig[arm].get(term) else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center",
                        fontsize=8.8,
                        color="white" if abs(v) > vmax * .62 else INK,
                        fontweight="bold" if i == j else "normal")
            ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                       edgecolor=INK, lw=1.8, zorder=4))
        labs = [BLOCK_SHORT[b] for b in BLOCKS]
        ax.set_xticks(range(8), labs, rotation=38, ha="right", fontsize=9)
        ax.set_yticks(range(8), labs, fontsize=9)
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(colors=MUTED)
        cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
        cb.set_label("p90 effect  −  median effect", fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title("diagonal: change in main effect   ·   off-diagonal: "
                     "change in pairwise interaction\n"
                     "* = paired-bootstrap change significant at BH-FDR 5% "
                     f"(family of {len(changes)} comparisons)",
                     fontsize=9, color=MUTED, pad=12, linespacing=1.6)
        ax.annotate(f"{arm} — p90 minus median factorial effects",
                    xy=(.5, 1.0), xycoords="axes fraction", xytext=(0, 34),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=INK)
        fig.tight_layout()
        fig.savefig(P_CMP / f"{ARM_SLUG[arm]}_p90_minus_median_effects.png",
                    bbox_inches="tight")
        plt.close(fig)
    print(f"  shared delta-heatmap scale: +/- {vmax:.4f}")

    # tree depth vs R2, median and p90 together
    mt = pd.read_csv(MED_TREE / "tree_depth_metrics.csv")
    mt["statistic"] = "median"
    pt = pd.read_csv(P90_TREE / "p90_tree_depth_metrics.csv")
    both = pd.concat([mt, pt], ignore_index=True)
    both.to_csv(CMP / "tree_depth_metrics_median_and_p90.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=190)
    for arm in ARMS:
        for stat, ls, mk in (("median", "-", "o"), ("p90", "--", "s")):
            s = both[(both.comparison == arm) & (both.statistic == stat)]
            ax.plot(s.max_depth, s.train_R2, ls, marker=mk, ms=7, lw=2,
                    color=ARM_COLOR[arm], mec="black", mew=.5,
                    alpha=1.0 if stat == "median" else .8,
                    label=f"{ARM_SLUG[arm]} {stat} training R²")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("maximum tree depth", fontsize=10.5)
    ax.set_ylabel("training R²", fontsize=10.5)
    ax.set_title("solid = median response, dashed = p90 response",
                 fontsize=8.8, color=MUTED, pad=10)
    ax.annotate("Tree fit to the 256-context response: median vs p90",
                xy=(.5, 1.0), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(P_CMP / "tree_depth_vs_R2_median_vs_p90.png",
                bbox_inches="tight")
    plt.close(fig)

    # ---- console report ---------------------------------------------------
    print("\n" + "=" * 78 + "\nMAIN EFFECTS: median vs p90\n" + "=" * 78)
    me = side[side.kind == "main_effect"]
    for arm in ARMS:
        s = me[me.comparison == arm].set_index("term").reindex(BLOCKS)
        print(f"\n--- {ARM_SLUG[arm]} ---")
        print(s[["median", "p90", "change", "ci95_low", "ci95_high",
                 "p_BH_adjusted", "significant_FDR_5pct", "sign_flip"]]
              .round(4).to_string())

    print("\n" + "=" * 78 + "\nLARGEST EFFECT CHANGES (|p90 - median|)\n" + "=" * 78)
    print(top.head(15)[["comparison", "kind", "term", "median", "p90",
                        "change", "ci95_low", "ci95_high",
                        "significant_FDR_5pct"]].round(4).to_string(index=False))

    nsig = int(changes.significant_FDR_5pct.sum())
    nci = int(changes.ci_excludes_zero.sum())
    print(f"\nfamily of {len(changes)} comparisons: {nci} with 95% CI excluding "
          f"zero, {nsig} significant at BH-FDR {FDR_Q:.0%}")

    print("\n" + "=" * 78 + "\nSHAP IMPORTANCE CHANGES\n" + "=" * 78)
    print(shap_changes[["comparison", "term", "mean_abs_SHAP_median",
                        "mean_abs_SHAP_p90", "point_change", "ci95_low",
                        "ci95_high", "significant_FDR_5pct"]]
          .round(4).to_string(index=False))

    print("\n" + "=" * 78 + "\nB3:B4 FOCUS\n" + "=" * 78)
    b34 = side[(side.term == "B3:B4")]
    print(b34[["comparison", "median", "p90", "change", "ci95_low",
               "ci95_high", "p_BH_adjusted", "significant_FDR_5pct"]]
          .round(4).to_string(index=False))

    print(f"\nwrote -> {CMP.relative_to(ROOT)} and {P_CMP.relative_to(ROOT)}")
    print("No AlphaGenome calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
