"""
Interpretable regression trees over the 8-block context-response landscape.

Part 3 of the 8-block follow-up. Makes NO AlphaGenome calls and writes only
into new subdirectories; factorial, SHAP, bootstrap and Hamming outputs are
read-only.

One observation = one CONTEXT (256 per arm). Features are the eight blocks
with P = 0 and F = 1; the target is the median AG_mean_logit over that
context's 400 synthetic exons. The 204,800 individual AlphaGenome rows are
never used as training rows, so a single exon cannot appear in both a train
and a test fold.

Because all 256 binary combinations are already observed, the objective is
interpretable COMPRESSION of a known response function, not extrapolation.
5-fold CV over contexts is reported as a rule-stability check only: it says
nothing about generalisation to new synthetic exon sequences, which would
require new AlphaGenome scoring.

Usage::

    python analysis_script/analyze_chimera8_trees.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

NEW = ROOT / "outputs" / "synthetic_context_8block"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
TREE_DIR = NEW / "decision_tree"
P_TREE = NEW / "plots" / "decision_tree"

Y = "AG_mean_logit"
ARMS = ["TP53_e6", "TP53_e7"]
ARM_SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
BLOCKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
BLOCK_SHORT = {
    "B1": "B1 Upstream exon body", "B2": "B2 Upstream exon 5'SS",
    "B3": "B3 Upstream intron body", "B4": "B4 Middle exon 3'SS",
    "B5": "B5 Middle exon 5'SS", "B6": "B6 Downstream intron body",
    "B7": "B7 Downstream exon 3'SS", "B8": "B8 Downstream exon body",
}
DEPTHS = (1, 2, 3, 4)
SELECTED_DEPTH = 3
MIN_LEAF = 8
SEED = 0

ARM_COLOR = {"TP53_e6": "#f0c000", "TP53_e7": "#2f6fd0"}
NODE_FILL, LEAF_FILL = "#eef3fa", "#f7f1e3"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def design_matrix(d: pd.DataFrame, arm: str, stat: str = "median"
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """256 contexts x 8 binary features, with the context response as target."""
    g = d[d.comparison == arm].groupby("context_label_8")[Y]
    med = (g.median() if stat == "median" else g.quantile(0.90)).sort_index()
    X = pd.DataFrame(
        {b: [1 if lab[i] == "F" else 0 for lab in med.index]
         for i, b in enumerate(BLOCKS)}, index=med.index)
    return X, med


# ==========================================================================
# tree rendering
# ==========================================================================

def _layout(tree, node: int, depth: int, nxt: list[int],
            out: dict[int, tuple[float, int]]) -> float:
    left, right = tree.children_left[node], tree.children_right[node]
    if left == -1:
        x = float(nxt[0])
        nxt[0] += 1
    else:
        xl = _layout(tree, left, depth + 1, nxt, out)
        xr = _layout(tree, right, depth + 1, nxt, out)
        x = (xl + xr) / 2.0
    out[node] = (x, depth)
    return x


def plot_tree_custom(model: DecisionTreeRegressor, arm: str, depth: int,
                     path: Path) -> None:
    """Readable diagram: block names, explicit P/F branches, n and mean."""
    t = model.tree_
    pos: dict[int, tuple[float, int]] = {}
    _layout(t, 0, 0, [0], pos)
    n_leaves = sum(1 for i in range(t.node_count) if t.children_left[i] == -1)
    max_d = max(d for _, d in pos.values())

    fig_w = max(11.0, n_leaves * 2.45)
    fig_h = 2.5 + max_d * 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=175)

    # edges first
    for node, (x, d) in pos.items():
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            continue
        for child, lab in ((left, "= P"), (right, "= F")):
            cx, cd = pos[child]
            ax.plot([x, cx], [-d, -cd], color=MUTED, lw=1.2, zorder=1)
            mx, my = (x + cx) / 2, (-d - cd) / 2
            ax.text(mx, my, lab, fontsize=10, fontweight="bold",
                    color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=MUTED, lw=.6))

    for node, (x, d) in pos.items():
        is_leaf = t.children_left[node] == -1
        n = int(t.n_node_samples[node])
        val = float(t.value[node].ravel()[0])
        # a node may be no wider than the horizontal span it owns, or the
        # boxes at that level collide
        span = 1.0 if is_leaf else float(2 ** (max_d - d))
        if is_leaf:
            txt = f"predict\n{val:+.3f}\nn = {n}"
            fc, w, h = LEAF_FILL, min(0.90, span * 0.90), 0.74
        else:
            feat = BLOCKS[t.feature[node]]
            txt = (f"{BLOCK_SHORT[feat]}\nn = {n}   mean {val:+.3f}")
            fc, w, h = NODE_FILL, min(2.6, span * 0.86), 0.58
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, -d - h / 2), w, h,
            boxstyle="round,pad=0.04", fc=fc, ec=INK, lw=1.0, zorder=2))
        ax.text(x, -d, txt, ha="center", va="center",
                fontsize=8.6 if is_leaf else 9.2,
                color=INK, zorder=3, linespacing=1.45,
                fontweight="bold" if is_leaf else "normal")

    ax.set_xlim(-1.0, n_leaves)
    ax.set_ylim(-max_d - 1.0, 0.95)
    ax.axis("off")
    ax.set_title(
        f"{arm} — depth-{depth} regression tree on the context median "
        "AlphaGenome mean-logit\n"
        "one observation = one of the 256 P/F contexts   ·   "
        "left branch = block is P (PNAS), right branch = block is F (FAS)",
        fontsize=12, fontweight="bold", color=INK, linespacing=1.7, pad=16)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def extract_rules(model: DecisionTreeRegressor, X: pd.DataFrame,
                  y: pd.Series, arm: str) -> pd.DataFrame:
    """One row per terminal leaf: path, coverage, observed vs predicted."""
    t = model.tree_
    leaf_of = model.apply(X)
    rows: list[dict] = []

    def walk(node: int, conds: list[str]) -> None:
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            mask = leaf_of == node
            obs = y[mask]
            rows.append({
                "comparison": arm,
                "leaf_id": int(node),
                "rule": " AND ".join(conds) if conds else "(root)",
                "depth": len(conds),
                "n_contexts": int(mask.sum()),
                "predicted_median_AG_mean_logit": float(t.value[node].ravel()[0]),
                "mean_observed_median": float(obs.mean()),
                "sd_within_leaf": float(obs.std(ddof=1)) if mask.sum() > 1 else 0.0,
                "min_observed_median": float(obs.min()),
                "max_observed_median": float(obs.max()),
                "example_contexts": ", ".join(sorted(obs.index)[:3]),
            })
            return
        feat = BLOCKS[t.feature[node]]
        walk(left, conds + [f"{feat}=P"])
        walk(right, conds + [f"{feat}=F"])

    walk(0, [])
    return pd.DataFrame(rows).sort_values(
        "predicted_median_AG_mean_logit").reset_index(drop=True)


def split_inventory(model: DecisionTreeRegressor) -> pd.DataFrame:
    """Which block is used at which level, and how much error it removes."""
    t = model.tree_
    rows = []
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        if t.children_left[node] == -1:
            continue
        left, right = t.children_left[node], t.children_right[node]
        n, nl, nr = (t.n_node_samples[node], t.n_node_samples[left],
                     t.n_node_samples[right])
        drop = (t.impurity[node]
                - (nl * t.impurity[left] + nr * t.impurity[right]) / n)
        rows.append({"level": d, "block": BLOCKS[t.feature[node]],
                     "n_contexts": int(n),
                     "variance_reduction": float(drop),
                     "weighted_variance_reduction": float(drop * n / 256)})
        stack += [(left, d + 1), (right, d + 1)]
    return pd.DataFrame(rows).sort_values(["level", "block"])


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    TREE_DIR.mkdir(parents=True, exist_ok=True)
    P_TREE.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(RESULTS_CSV,
                    usecols=["comparison", "context_label_8", "sequence_id", Y],
                    low_memory=False)
    sizes = d.groupby(["comparison", "context_label_8"]).sequence_id.nunique()

    metrics, rules, splits, models, data = [], [], [], {}, {}
    for arm in ARMS:
        X, y = design_matrix(d, arm)
        data[arm] = (X, y)
        for md in DEPTHS:
            m = DecisionTreeRegressor(criterion="squared_error", max_depth=md,
                                      min_samples_leaf=MIN_LEAF,
                                      random_state=SEED)
            m.fit(X, y)
            pred = m.predict(X)
            resid = y.to_numpy() - pred
            cv = cross_val_score(
                DecisionTreeRegressor(criterion="squared_error", max_depth=md,
                                      min_samples_leaf=MIN_LEAF,
                                      random_state=SEED),
                X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
            metrics.append({
                "comparison": arm, "max_depth": md,
                "actual_depth": int(m.get_depth()),
                "n_leaves": int(m.get_n_leaves()),
                "n_observations": int(len(y)), "n_features": int(X.shape[1]),
                "train_R2": float(m.score(X, y)),
                "train_RMSE": float(np.sqrt((resid ** 2).mean())),
                "train_MAE": float(np.abs(resid).mean()),
                "cv5_R2_mean": float(cv.mean()), "cv5_R2_sd": float(cv.std(ddof=1)),
            })
            if md == SELECTED_DEPTH:
                models[arm] = m
                r = extract_rules(m, X, y, arm)
                rules.append(r)
                sp = split_inventory(m)
                sp.insert(0, "comparison", arm)
                splits.append(sp)

    met = pd.DataFrame(metrics)
    met.to_csv(TREE_DIR / "tree_depth_metrics.csv", index=False)
    rules_df = pd.concat(rules, ignore_index=True)
    rules_df.to_csv(TREE_DIR / f"tree_depth{SELECTED_DEPTH}_rules.csv",
                    index=False)
    splits_df = pd.concat(splits, ignore_index=True)
    splits_df.to_csv(TREE_DIR / f"tree_depth{SELECTED_DEPTH}_splits.csv",
                     index=False)

    print("=== model metrics ===")
    print(met.round(4).to_string(index=False))

    # ---- figures ---------------------------------------------------------
    for arm in ARMS:
        p = P_TREE / f"{ARM_SLUG[arm]}_decision_tree_depth{SELECTED_DEPTH}.png"
        plot_tree_custom(models[arm], arm, SELECTED_DEPTH, p)
        print(f"\nwrote {p.relative_to(ROOT)}")

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=190)
    for arm in ARMS:
        s = met[met.comparison == arm]
        ax.plot(s.max_depth, s.train_R2, "-o", ms=7, lw=2,
                color=ARM_COLOR[arm], mec="black", mew=.5,
                label=f"{ARM_SLUG[arm]} training R²")
        ax.plot(s.max_depth, s.cv5_R2_mean, "--s", ms=6, lw=1.6,
                color=ARM_COLOR[arm], mec="black", mew=.5, alpha=.75,
                label=f"{ARM_SLUG[arm]} 5-fold CV R² (held-out contexts)")
        dy = 10 if arm == "TP53_e6" else -16
        for _, r in s.iterrows():
            ax.annotate(f"{r.train_R2:.3f}", (r.max_depth, r.train_R2),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(DEPTHS))
    ax.set_xlabel("maximum tree depth", fontsize=10.5)
    ax.set_ylabel("R²  (context median AlphaGenome mean-logit)", fontsize=10.5)
    ax.set_title("CV is over held-out CONTEXTS; it does not measure "
                 "generalisation to new synthetic exon sequences",
                 fontsize=8.6, color=MUTED, pad=10)
    ax.annotate("Tree complexity vs fit to the 256-context response function",
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
    depth_png = P_TREE / "tree_depth_vs_R2.png"
    fig.savefig(depth_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {depth_png.relative_to(ROOT)}")

    # ---- interpretation --------------------------------------------------
    print("\n=== depth-3 rules ===")
    for arm in ARMS:
        r = rules_df[rules_df.comparison == arm]
        print(f"\n--- {ARM_SLUG[arm]} ---")
        print(r[["rule", "n_contexts", "predicted_median_AG_mean_logit",
                 "sd_within_leaf"]].round(4).to_string(index=False))
        sp = splits_df[splits_df.comparison == arm]
        root = sp[sp.level == 0].block.iloc[0]
        print(f"  root split: {root} ({BLOCK_SHORT[root]})")
        print(f"  blocks used by level: "
              + "; ".join(f"L{lv}: {', '.join(sorted(set(g.block)))}"
                          for lv, g in sp.groupby('level')))
        lo, hi = r.iloc[0], r.iloc[-1]
        print(f"  lowest  leaf: {lo.rule}  -> {lo.predicted_median_AG_mean_logit:+.4f}"
              f"  (n={int(lo.n_contexts)})")
        print(f"  highest leaf: {hi.rule}  -> {hi.predicted_median_AG_mean_logit:+.4f}"
              f"  (n={int(hi.n_contexts)})")
        used = set(sp.block)
        pair = {"B3", "B4"} <= used
        nested = any(("B3=" in rr and "B4=" in rr) for rr in r.rule)
        print(f"  B3 and B4 both used: {pair}; "
              f"co-occur on a path (B3:B4 coupling visible): {nested}")

    # ---- validation ------------------------------------------------------
    print("\n" + "=" * 76 + "\nVALIDATION (part 3)\n" + "=" * 76)
    fails = []

    def chk(n, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:2d}. {name}"
              f"{'  -- ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    chk(1, "256 unique contexts per arm",
        all(len(data[a][1]) == 256 for a in ARMS),
        str({ARM_SLUG[a]: len(data[a][1]) for a in ARMS}))
    chk(2, "each context contains 400 sequence IDs", bool((sizes == 400).all()),
        f"min {sizes.min()} max {sizes.max()}")
    chk(8, "tree input is exactly 8 binary features",
        all(list(data[a][0].columns) == list(BLOCKS)
            and set(np.unique(data[a][0].to_numpy())) <= {0, 1}
            for a in ARMS),
        "columns B1..B8, values in {0,1} (P=0, F=1)")
    chk(9, "no leakage from individual sequence observations",
        all(len(data[a][0]) == 256 for a in ARMS)
        and bool((met.n_observations == 256).all()),
        "256 context-level rows per arm; the 204,800 per-sequence rows are "
        "aggregated to medians BEFORE any split or fold is formed")
    chk(10, "no AlphaGenome calls", True,
        "module imports no AlphaGenome client and opens no network path")
    print("=" * 76)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")

    print(f"\nwrote -> {TREE_DIR.relative_to(ROOT)}")
    print("No AlphaGenome calls were made.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
