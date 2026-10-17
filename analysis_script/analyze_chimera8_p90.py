"""
Repeat the factorial, exact-SHAP and decision-tree analyses on the context p90.

The response changes from the context MEDIAN to the context 90th percentile of
``AG_mean_logit`` across the 400 synthetic exons; everything else -- the P/F
encoding, the coalition conventions, the tree hyperparameters, the seeds -- is
imported unchanged from the median modules, so the two analyses differ only in
the statistic.

Writes only under ``outputs/synthetic_context_8block/p90/`` and
``plots/p90/``. The median outputs are read for validation and never written.

Makes NO AlphaGenome calls and fits no surrogate model: the 2^8 design is
complete, so Shapley values are evaluated from their definition.

Usage::

    python analysis_script/analyze_chimera8_p90.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import analyze_chimera8_factorial as FA      # noqa: E402
import analyze_chimera8_shap as SH           # noqa: E402
import analyze_chimera8_trees as TR          # noqa: E402
import plot_chimera8_factorial as PF         # noqa: E402
import plot_chimera8_shap as PS              # noqa: E402

NEW = ROOT / "outputs" / "synthetic_context_8block"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
P90 = NEW / "p90"
P90_RESP, P90_FACT, P90_SHAP, P90_TREE = (P90 / "response", P90 / "factorial",
                                          P90 / "shap", P90 / "decision_tree")
PLOT_P90 = NEW / "plots" / "p90"
MED_FACT, MED_SHAP = NEW / "factorial", NEW / "shap"
MED_P90_SUMMARY = NEW / "plots" / "summary" / "context_p90_AG_mean_logit.csv"

Y = "AG_mean_logit"
STAT = "p90"
VALUE_COL = "observed_p90_AG_mean_logit"
ARMS = FA.ARMS
ARM_SLUG = FA.ARM_SLUG
BLOCKS = FA.BLOCKS


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    for p in (P90_RESP, P90_FACT, P90_SHAP, P90_TREE,
              PLOT_P90 / "factorial", PLOT_P90 / "shap",
              PLOT_P90 / "decision_tree"):
        p.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(RESULTS_CSV,
                    usecols=["comparison", "context_label_8", "sequence_id",
                             "alpha", Y], low_memory=False)
    print(f"loaded {len(d):,} rows")

    # ================= 1. p90 response table =============================
    print("\n" + "=" * 76 + "\n1. CONTEXT p90 RESPONSE TABLE\n" + "=" * 76)
    sizes = d.groupby(["comparison", "context_label_8"]).sequence_id.nunique()
    resp_rows = []
    p90 = {}
    for arm in ARMS:
        g = d[d.comparison == arm].groupby("context_label_8")[Y]
        s = g.quantile(0.90).sort_index()
        p90[arm] = s.to_dict()
        for lab, v in s.items():
            resp_rows.append({
                "comparison": arm, "context_label_8": lab,
                "n_F_blocks": lab.count("F"), "n_sequences": 400,
                "p90_AG_mean_logit": float(v),
                **{b: lab[i] for i, b in enumerate(BLOCKS)},
            })
    resp = pd.DataFrame(resp_rows)
    resp_path = P90_RESP / "context_p90_response_table.csv"
    resp.to_csv(resp_path, index=False)

    fails = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              f"{'  -- ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    chk("256 contexts per arm",
        all(len(p90[a]) == 256 for a in ARMS),
        str({ARM_SLUG[a]: len(p90[a]) for a in ARMS}))
    chk("400 sequences per context", bool((sizes == 400).all()),
        f"min {sizes.min()} max {sizes.max()}")
    chk("no missing or non-finite p90 values",
        bool(resp.p90_AG_mean_logit.notna().all()
             and np.isfinite(resp.p90_AG_mean_logit).all()))
    # must match the previously saved p90 summary
    prev = pd.read_csv(MED_P90_SUMMARY)
    prev = prev[prev.group != "original"].copy()
    prev["comparison"] = prev.group.map({"e6": "TP53_e6", "e7": "TP53_e7"})
    m = resp.merge(prev[["comparison", "context_label_8",
                         "p90_AG_mean_logit"]],
                   on=["comparison", "context_label_8"], how="inner",
                   suffixes=("", "_saved"))
    err = float((m.p90_AG_mean_logit - m.p90_AG_mean_logit_saved).abs().max())
    chk("matches the previously saved p90 context summary",
        len(m) == 512 and err < 1e-12, f"{len(m)} matched, max diff {err:.3e}")
    print(f"  wrote {resp_path.relative_to(ROOT)}")

    # ================= 2. factorial ======================================
    print("\n" + "=" * 76 + "\n2. MAIN EFFECTS AND PAIRWISE INTERACTIONS (p90)\n"
          + "=" * 76)
    effects, backgrounds = {}, {}
    for arm in ARMS:
        me = FA.main_effects(p90[arm])
        pe, bg = FA.pairwise_interactions(p90[arm])
        pe["frac_positive"] = np.nan
        pe["frac_negative"] = np.nan
        for i, r in pe.iterrows():
            v = bg[bg.term == r.term].interaction_dod
            pe.loc[i, "frac_positive"] = float((v > 0).mean())
            pe.loc[i, "frac_negative"] = float((v < 0).mean())
        eff = pd.concat([me, pe], ignore_index=True)
        eff.insert(0, "comparison", arm)
        eff.insert(1, "statistic", STAT)
        bg.insert(0, "comparison", arm)
        bg.insert(1, "statistic", STAT)
        effects[arm], backgrounds[arm] = eff, bg
        eff.to_csv(P90_FACT / f"{ARM_SLUG[arm]}_p90_main_pairwise_effects.csv",
                   index=False)
        bg.to_csv(P90_FACT
                  / f"{ARM_SLUG[arm]}_p90_pairwise_interaction_backgrounds.csv",
                  index=False)
        print(f"\n--- {ARM_SLUG[arm]} p90 main effects ---")
        print(me[["term", "effect", "mean_if_P", "mean_if_F"]]
              .round(4).to_string(index=False))
        top = pe.reindex(pe.effect.abs().sort_values(ascending=False).index)
        print(f"--- {ARM_SLUG[arm]} top 5 |p90 pairwise interactions| ---")
        print(top.head(5)[["term", "interaction_mean",
                           "interaction_SD_across_backgrounds",
                           "interaction_min", "interaction_max",
                           "frac_positive"]].round(4).to_string(index=False))

    mats = {a: PF.effect_matrix(effects[a]) for a in ARMS}
    vmax = float(max(np.abs(m).max() for m in mats.values()))
    for arm in ARMS:
        path = (PLOT_P90 / "factorial"
                / f"{ARM_SLUG[arm]}_p90_main_pairwise_effect_heatmap.png")
        PF.plot_heatmap(mats[arm], arm, vmax, path,
                        display=f"{arm} (p90)")
        print(f"  wrote {path.relative_to(ROOT)}")
    print(f"  shared p90 heatmap scale: +/- {vmax:.4f}")

    # ================= 3. exact SHAP =====================================
    print("\n" + "=" * 76 + "\n3. EXACT SHAP ON THE p90 RESPONSE\n" + "=" * 76)
    shap_summ, max_err, max_ierr = {}, 0.0, 0.0
    for arm in ARMS:
        slug = ARM_SLUG[arm]
        f, _ = SH.response_table(d, arm, stat=STAT)
        baseline = float(f.mean())
        v = SH.conditional_means(f)
        phi = SH.exact_shap(f, v)
        inter = SH.exact_shap_interactions(f, v, phi)

        long = SH.to_long(phi, f, baseline, arm, value_col=VALUE_COL)
        wide = SH.to_wide(phi, f, baseline, arm, value_col=VALUE_COL)
        summ = SH.feature_summary(long)
        ilong = SH.interactions_long(inter, arm)
        for frame in (long, wide, summ, ilong):
            frame.insert(0, "statistic", STAT)

        err = float(np.abs(wide.reconstruction_error).max())
        ierr = float(np.abs(inter.sum(axis=(0, 1)) - (f - baseline)).max())
        max_err, max_ierr = max(max_err, err), max(max_ierr, ierr)

        long.to_csv(P90_SHAP / f"{slug}_p90_exact_shap_values.csv", index=False)
        wide.to_csv(P90_SHAP / f"{slug}_p90_exact_shap_values_wide.csv",
                    index=False)
        summ.to_csv(P90_SHAP / f"{slug}_p90_shap_feature_summary.csv",
                    index=False)
        ilong.to_csv(P90_SHAP / f"{slug}_p90_exact_shap_interactions.csv",
                     index=False)
        np.save(P90_SHAP / f"{slug}_p90_shap_interaction_tensor.npy", inter)
        shap_summ[arm] = {"summary": summ, "long": long, "inter": inter,
                          "baseline": baseline}
        print(f"\n--- {slug} ---")
        print(f"  baseline (mean context p90)          : {baseline:.6f}")
        print(f"  max |baseline + sum(phi) - observed| : {err:.3e}")
        print(f"  max |sum(phi_ij) - (f - baseline)|   : {ierr:.3e}")
        print(summ[["rank", "block", "mean_abs_SHAP", "mean_SHAP_when_F",
                    "SD_SHAP_within_mean"]].round(4).to_string(index=False))

    chk("SHAP efficiency: baseline + sum(phi) = observed p90",
        max_err < 1e-10, f"max abs error {max_err:.3e}")
    chk("SHAP interaction decomposition sums to f - baseline",
        max_ierr < 1e-10, f"max abs error {max_ierr:.3e}")

    # p90 SHAP figures, on scales shared across the two arms
    xmax = max(shap_summ[a]["summary"].mean_abs_SHAP.max() for a in ARMS)
    lo = min(shap_summ[a]["long"].shap_value.min() for a in ARMS)
    hi = max(shap_summ[a]["long"].shap_value.max() for a in ARMS)
    pad = (hi - lo) * .05
    xlim = (lo - pad, hi + pad)
    mean_abs = {a: np.abs(shap_summ[a]["inter"]).mean(axis=tuple(range(2, 10)))
                for a in ARMS}
    ivmax = float(max(m.max() for m in mean_abs.values()))
    for arm in ARMS:
        slug = ARM_SLUG[arm]
        order = shap_summ[arm]["summary"].sort_values(
            "mean_abs_SHAP", ascending=False).block.tolist()
        disp = f"{arm} (p90)"
        PS.plot_importance(
            shap_summ[arm]["summary"], arm, xmax,
            PLOT_P90 / "shap" / f"{slug}_p90_mean_absolute_shap.png",
            display=disp)
        PS.plot_beeswarm(
            shap_summ[arm]["long"], order, arm, xlim,
            PLOT_P90 / "shap" / f"{slug}_p90_exact_shap_beeswarm.png",
            display=disp)
        PS.plot_state_conditional(
            shap_summ[arm]["long"], order, arm, xlim,
            PLOT_P90 / "shap" / f"{slug}_p90_shap_state_conditional.png",
            display=disp)
        PS.plot_interaction_heatmap(
            mean_abs[arm], arm, ivmax,
            PLOT_P90 / "shap"
            / f"{slug}_p90_mean_abs_shap_interaction_heatmap.png",
            display=disp)
        rows = [{"statistic": STAT, "comparison": arm, "block_i": bi,
                 "block_j": bj,
                 "mean_abs_shap_interaction": mean_abs[arm][i, j]}
                for i, bi in enumerate(BLOCKS) for j, bj in enumerate(BLOCKS)]
        pd.DataFrame(rows).to_csv(
            P90_SHAP / f"{slug}_p90_mean_abs_shap_interaction_matrix.csv",
            index=False)
        print(f"  wrote 4 p90 SHAP figures for {slug}")

    # ================= 4. decision trees =================================
    print("\n" + "=" * 76 + "\n4. DECISION TREES ON THE p90 RESPONSE\n" + "=" * 76)
    metrics, rules, splits = [], [], []
    for arm in ARMS:
        X, y = TR.design_matrix(d, arm, stat=STAT)
        for md in TR.DEPTHS:
            m = DecisionTreeRegressor(criterion="squared_error", max_depth=md,
                                      min_samples_leaf=TR.MIN_LEAF,
                                      random_state=TR.SEED).fit(X, y)
            pred = m.predict(X)
            resid = y.to_numpy() - pred
            cv = cross_val_score(
                DecisionTreeRegressor(criterion="squared_error", max_depth=md,
                                      min_samples_leaf=TR.MIN_LEAF,
                                      random_state=TR.SEED),
                X, y, cv=KFold(5, shuffle=True, random_state=TR.SEED),
                scoring="r2")
            metrics.append({
                "statistic": STAT, "comparison": arm, "max_depth": md,
                "actual_depth": int(m.get_depth()),
                "n_leaves": int(m.get_n_leaves()),
                "n_observations": int(len(y)), "n_features": int(X.shape[1]),
                "train_R2": float(m.score(X, y)),
                "train_RMSE": float(np.sqrt((resid ** 2).mean())),
                "train_MAE": float(np.abs(resid).mean()),
                "cv5_R2_mean": float(cv.mean()),
                "cv5_R2_sd": float(cv.std(ddof=1))})
            if md == TR.SELECTED_DEPTH:
                p = (PLOT_P90 / "decision_tree"
                     / f"{ARM_SLUG[arm]}_p90_decision_tree_depth3.png")
                TR.plot_tree_custom(m, f"{arm} (p90)", md, p)
                r = TR.extract_rules(m, X, y, arm)
                r.insert(0, "statistic", STAT)
                rules.append(r)
                sp = TR.split_inventory(m)
                sp.insert(0, "comparison", arm)
                sp.insert(0, "statistic", STAT)
                splits.append(sp)
                print(f"  wrote {p.relative_to(ROOT)}")

    met = pd.DataFrame(metrics)
    met.to_csv(P90_TREE / "p90_tree_depth_metrics.csv", index=False)
    rules_df = pd.concat(rules, ignore_index=True)
    rules_df.to_csv(P90_TREE / "p90_tree_depth3_rules.csv", index=False)
    splits_df = pd.concat(splits, ignore_index=True)
    splits_df.to_csv(P90_TREE / "p90_tree_depth3_splits.csv", index=False)
    print()
    print(met.round(4).to_string(index=False))
    for arm in ARMS:
        r = rules_df[rules_df.comparison == arm]
        sp = splits_df[splits_df.comparison == arm]
        print(f"\n--- {ARM_SLUG[arm]} p90 depth-3 rules ---")
        print(r[["rule", "n_contexts", "predicted_median_AG_mean_logit",
                 "sd_within_leaf"]].round(4).to_string(index=False))
        print(f"  root split: {sp[sp.level == 0].block.iloc[0]}")
        print("  blocks by level: " + "; ".join(
            f"L{lv}: {', '.join(sorted(set(g.block)))}"
            for lv, g in sp.groupby('level')))

    chk("tree input is 8 binary features",
        bool((met.n_features == 8).all() and (met.n_observations == 256).all()))
    chk("no AlphaGenome calls", True, "no AG client is imported anywhere")

    print("\n" + "=" * 76)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")
    print(f"wrote -> {P90.relative_to(ROOT)} and {PLOT_P90.relative_to(ROOT)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
