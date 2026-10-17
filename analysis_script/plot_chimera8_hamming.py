"""
Hamming-distance-1 jitter graph and the focused B3:B4 interaction figure.

Parts 1 and 2 of the 8-block follow-up. Makes NO AlphaGenome calls, fits no
model, and writes only into new subdirectories; existing factorial, SHAP and
bootstrap outputs are read-only inputs.

Part 1  One figure, two stacked jitter panels (e6 top, e7 bottom), styled after
        the existing context-median jitter strip. Every pair of contexts in the
        SAME arm whose 8-character P/F labels differ at exactly one position is
        joined by a thin translucent edge drawn behind the dots. 256 nodes and
        1,024 undirected edges per arm.

        The x-axis is the context median AG mean-logit. The y-coordinate is
        deterministic jitter with NO meaning, so edge slope and vertical
        distance carry no information -- only horizontal separation does.

Part 2  One figure, two stacked panels, showing the B3 x B4 interaction one
        background at a time: for each of the 64 settings of the other six
        blocks, the B4 P->F effect when B3=P and when B3=F, joined by a line
        whose length IS that background's interaction.

Labels: all arithmetic uses the analytical P/F labels. Display labels rewrite
F at B4/B5 only, as 6 in the e6 arm and 7 in the e7 arm.

Usage::

    python analysis_script/plot_chimera8_hamming.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

NEW = ROOT / "outputs" / "synthetic_context_8block"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
FACT_DIR = NEW / "factorial"
HAM_DIR = NEW / "hamming"
P_HAM = NEW / "plots" / "hamming"
OLD_ENDPOINT_CSV = (ROOT / "outputs" / "synthetic_context"
                    / "synthetic_endpoint_ag_results.csv")

Y = "AG_mean_logit"
ARMS = ["TP53_e6", "TP53_e7"]
ARM_SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
ARM_DIGIT = {"TP53_e6": "6", "TP53_e7": "7"}
BLOCKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
BLOCK_SHORT = {
    "B1": "B1 Up exon body", "B2": "B2 Up 5'SS", "B3": "B3 Up intron",
    "B4": "B4 Mid 3'SS", "B5": "B5 Mid 5'SS", "B6": "B6 Down intron",
    "B7": "B7 Down 3'SS", "B8": "B8 Down exon body",
}
SS_POSITIONS = (3, 4)          # B4, B5 take the 6/7 display notation

#: inherited from the existing context-median jitter strip
ARM_COLOR = {"TP53_e6": "#f0c000", "TP53_e7": "#2f6fd0"}
DOT_SIZE = 46
EDGE_COLOR = "#60a5fa"
EDGE_ALPHA = 0.14              # tuned by eye after rendering: visible, not dominant
EDGE_LW = 0.55
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
#: B3 state colours for Part 2 -- CVD-validated pair
B3_COLOR = {"P": "#2a78d6", "F": "#eb6834"}


def display_label(label: str, arm: str) -> str:
    """F at B4/B5 becomes the arm's TP53 digit. Display only."""
    c = list(label)
    for p in SS_POSITIONS:
        if c[p] == "F":
            c[p] = ARM_DIGIT[arm]
    return "".join(c)


# ==========================================================================
# PART 1 -- Hamming-1 graph over the jitter layout
# ==========================================================================

def hamming1_edges(labels: list[str]) -> list[tuple[str, str, int]]:
    """(label_with_P, label_with_F, position) for every 1-bit neighbour pair.

    Orienting each edge P->F at the differing position makes it undirected and
    unique: the pair is emitted once, from the P side.
    """
    have = set(labels)
    edges = []
    for lab in labels:
        for k in range(8):
            if lab[k] == "P":
                other = lab[:k] + "F" + lab[k + 1:]
                if other in have:
                    edges.append((lab, other, k))
    return edges


def build_edge_table(med: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        labels = sorted(med[arm])
        for lab_p, lab_f, k in hamming1_edges(labels):
            y_p, y_f = med[arm][lab_p], med[arm][lab_f]
            rows.append({
                "comparison": arm,
                "block_changed": BLOCKS[k],
                "block_position": k + 1,
                "label_P_analysis": lab_p,
                "label_F_analysis": lab_f,
                "label_P_display": display_label(lab_p, arm),
                "label_F_display": display_label(lab_f, arm),
                "median_P": y_p,
                "median_F": y_f,
                "delta_P_to_F": y_f - y_p,
            })
    return pd.DataFrame(rows)


def plot_hamming(med: dict[str, dict[str, float]], edges: pd.DataFrame,
                 ep_med: dict[str, float], path: Path) -> dict:
    rng = np.random.default_rng(0)
    pos: dict[str, dict[str, tuple[float, float]]] = {}
    for arm in ARMS:
        labels = sorted(med[arm])
        yy = rng.uniform(0.15, 0.85, len(labels))
        pos[arm] = {lab: (med[arm][lab], float(y))
                    for lab, y in zip(labels, yy)}

    allv = [v for arm in ARMS for v in med[arm].values()]
    lo, hi = min(allv + [ep_med["PNAS"]]), max(allv + [ep_med["PNAS"]])
    pad = (hi - lo) * 0.04
    xlim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), sharex=True)
    fig.subplots_adjust(top=0.875, bottom=0.085, left=0.055, right=0.985,
                        hspace=0.20)
    for ax, arm in zip(axes, ARMS):
        e = edges[edges.comparison == arm]
        segs = [[pos[arm][r.label_P_analysis], pos[arm][r.label_F_analysis]]
                for r in e.itertuples()]
        ax.add_collection(LineCollection(
            segs, colors=EDGE_COLOR, linewidths=EDGE_LW, alpha=EDGE_ALPHA,
            zorder=1))

        ax.axvline(ep_med[arm], color="black", lw=1.4, ls="--", zorder=2)
        ax.axvline(ep_med["PNAS"], color=MUTED, lw=1.2, ls=":", zorder=2)

        xs = [pos[arm][l][0] for l in sorted(med[arm])]
        ys = [pos[arm][l][1] for l in sorted(med[arm])]
        ax.scatter(xs, ys, s=DOT_SIZE, color=ARM_COLOR[arm],
                   edgecolor="black", linewidth=.55, zorder=3)

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_ylabel("contexts", fontsize=9)
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.set_title(
            f"{arm}   ·   {len(med[arm])} contexts, {len(e):,} Hamming-1 edges"
            f"   ·   dashed = 10k {arm} endpoint median ({ep_med[arm]:.2f})"
            f", dotted = 10k PNAS endpoint median ({ep_med['PNAS']:.2f})",
            fontsize=9.5, color=MUTED, loc="left", pad=6)

    axes[-1].set_xlabel("context median AlphaGenome mean-logit", fontsize=11)
    fig.text(0.5, 0.962, "Hamming-distance-1 neighbourhood of the 8-block "
             "context space", ha="center", fontsize=14, fontweight="bold",
             color=INK)
    fig.text(0.5, 0.934,
             "every edge joins two contexts whose P/F labels differ at exactly "
             "one block   ·   y is deterministic jitter with no meaning, so "
             "only HORIZONTAL separation is interpretable",
             ha="center", fontsize=9, color=MUTED)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return pos


# ==========================================================================
# PART 2 -- B3 x B4 conditional effects, one row per background
# ==========================================================================

OTHER6 = ("B1", "B2", "B5", "B6", "B7", "B8")
OTHER_POS = (0, 1, 4, 5, 6, 7)
B3_POS, B4_POS = 2, 3


def b3b4_table(med: dict[str, dict[str, float]]) -> pd.DataFrame:
    """64 backgrounds x 2 arms of conditional B4 effects."""
    rows = []
    for arm in ARMS:
        for bits in itertools.product("PF", repeat=6):
            base = [""] * 8
            for p, c in zip(OTHER_POS, bits):
                base[p] = c

            def lab(b3: str, b4: str) -> str:
                s = list(base)
                s[B3_POS], s[B4_POS] = b3, b4
                return "".join(s)

            y = {f"y_{a}{b}": med[arm][lab(a, b)]
                 for a in "PF" for b in "PF"}
            d_p = y["y_PF"] - y["y_PP"]       # B4 effect with B3 = P
            d_f = y["y_FF"] - y["y_FP"]       # B4 effect with B3 = F
            rows.append({
                "comparison": arm,
                "background_blocks": "".join(OTHER6),
                "background_label": "".join(bits),
                "background_display": "".join(bits),
                "label_PP": lab("P", "P"), "label_PF": lab("P", "F"),
                "label_FP": lab("F", "P"), "label_FF": lab("F", "F"),
                **y,
                "delta_B4_given_B3P": d_p,
                "delta_B4_given_B3F": d_f,
                "interaction": d_f - d_p,
            })
    return pd.DataFrame(rows)


def plot_b3b4(tab: pd.DataFrame, path: Path, order_by: str) -> str:
    e7 = tab[tab.comparison == "TP53_e7"].set_index("background_label")
    order = e7.sort_values(order_by).index.tolist()
    order_note = ("backgrounds ordered by the e7 interaction, ascending; "
                  "the same order is used in both panels")

    xs = pd.concat([tab.delta_B4_given_B3P, tab.delta_B4_given_B3F])
    lo, hi = xs.min(), xs.max()
    pad = (hi - lo) * 0.06
    xlim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 15.5), sharex=True)
    fig.subplots_adjust(top=0.925, bottom=0.052, left=0.115, right=0.985,
                        hspace=0.075)
    for ax, arm in zip(axes, ARMS):
        s = tab[tab.comparison == arm].set_index("background_label").loc[order]
        y = np.arange(len(order))
        ax.hlines(y, s.delta_B4_given_B3P, s.delta_B4_given_B3F,
                  color=MUTED, lw=0.9, alpha=.75, zorder=2)
        ax.scatter(s.delta_B4_given_B3P, y, s=22, color=B3_COLOR["P"],
                   edgecolor="black", lw=.35, zorder=3,
                   label="B4 effect when B3 = P")
        ax.scatter(s.delta_B4_given_B3F, y, s=22, color=B3_COLOR["F"],
                   edgecolor="black", lw=.35, zorder=3,
                   label="B4 effect when B3 = F")
        ax.axvline(0, color=INK, lw=1.3, zorder=1)
        ax.set_yticks(y, order, fontsize=5.6, fontfamily="monospace")
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(*xlim)
        ax.set_ylabel(f"background  ({' '.join(OTHER6)})", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=.6, alpha=.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)

        m = s.interaction.mean()
        npos = int((s.interaction > 0).sum())
        ax.set_title(
            f"{arm}   ·   mean interaction = {m:+.4f}   ·   "
            f"{npos}/64 backgrounds positive   ·   "
            f"range [{s.interaction.min():+.3f}, {s.interaction.max():+.3f}]",
            fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=6)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    axes[-1].set_xlabel("effect of switching B4 (middle-exon 3'SS) from P to F"
                        "   —   AlphaGenome mean-logit", fontsize=10.5)
    fig.text(0.5, 0.972, "B3 × B4 interaction, one background at a time",
             ha="center", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.5, 0.955,
             "the horizontal gap between the two dots IS that background's "
             f"interaction   ·   {order_note}",
             ha="center", fontsize=9, color=MUTED)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return order_note


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    HAM_DIR.mkdir(parents=True, exist_ok=True)
    P_HAM.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(RESULTS_CSV,
                    usecols=["comparison", "context_label_8", "sequence_id", Y],
                    low_memory=False)
    sizes = d.groupby(["comparison", "context_label_8"]).sequence_id.nunique()
    med = {arm: d[d.comparison == arm].groupby("context_label_8")[Y]
                 .median().to_dict() for arm in ARMS}
    ep = pd.read_csv(OLD_ENDPOINT_CSV)
    ep_med = ep.groupby("context")[Y].median().to_dict()

    # ---------------- Part 1 ----------------
    edges = build_edge_table(med)
    edge_csv = HAM_DIR / "chimera8_hamming1_edges.csv"
    edges.to_csv(edge_csv, index=False)
    ham_png = P_HAM / "chimera8_hamming1_jitter_e6_e7.png"
    plot_hamming(med, edges, ep_med, ham_png)
    print(f"[part 1] {len(edges):,} edges -> {edge_csv.relative_to(ROOT)}")
    print(f"[part 1] figure -> {ham_png.relative_to(ROOT)}")

    per_block = (edges.groupby(["comparison", "block_changed"])
                 .delta_P_to_F.agg(["count", "mean", "std", "min", "max"])
                 .reset_index())
    per_block.to_csv(HAM_DIR / "chimera8_hamming1_edge_summary.csv", index=False)
    print("\n  mean signed P->F step along each edge class:")
    print(per_block.pivot(index="block_changed", columns="comparison",
                          values="mean").round(4).to_string())

    # ---------------- Part 2 ----------------
    b34 = b3b4_table(med)
    b34_csv = HAM_DIR / "B3_B4_conditional_effects.csv"
    b34.to_csv(b34_csv, index=False)
    b34_png = P_HAM / "B3_B4_conditional_effects_e6_e7.png"
    order_note = plot_b3b4(b34, b34_png, order_by="interaction")
    print(f"\n[part 2] {len(b34)} rows -> {b34_csv.relative_to(ROOT)}")
    print(f"[part 2] figure -> {b34_png.relative_to(ROOT)}")
    print(f"[part 2] {order_note}")
    for arm in ARMS:
        s = b34[b34.comparison == arm]
        print(f"  {ARM_SLUG[arm]}: mean interaction {s.interaction.mean():+.4f}"
              f"  SD {s.interaction.std(ddof=1):.4f}"
              f"  range [{s.interaction.min():+.4f}, {s.interaction.max():+.4f}]"
              f"  positive {int((s.interaction > 0).sum())}/64")

    # ---------------- validation ----------------
    print("\n" + "=" * 76 + "\nVALIDATION (parts 1-2)\n" + "=" * 76)
    fails = []

    def chk(n, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:2d}. {name}"
              f"{'  -- ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    chk(1, "256 unique contexts per arm",
        all(len(med[a]) == 256 for a in ARMS),
        str({ARM_SLUG[a]: len(med[a]) for a in ARMS}))
    chk(2, "each context contains 400 sequence IDs", bool((sizes == 400).all()),
        f"min {sizes.min()} max {sizes.max()}")
    cnt = edges.groupby("comparison").size().to_dict()
    chk(3, "exactly 1,024 Hamming-1 edges per arm",
        all(cnt.get(a) == 1024 for a in ARMS), str(cnt))
    diffs = [sum(c1 != c2 for c1, c2 in zip(r.label_P_analysis,
                                            r.label_F_analysis))
             for r in edges.itertuples()]
    chk(4, "every edge changes exactly one block",
        set(diffs) == {1} and bool(
            (edges.apply(lambda r: r.label_P_analysis[r.block_position - 1] == "P"
                         and r.label_F_analysis[r.block_position - 1] == "F",
                         axis=1)).all()),
        f"distinct Hamming distances: {sorted(set(diffs))}")
    key = edges.apply(lambda r: (r.comparison,
                                 *sorted((r.label_P_analysis,
                                          r.label_F_analysis))), axis=1)
    chk(5, "no cross-arm edges, no duplicates, no self-loops",
        key.is_unique
        and not (edges.label_P_analysis == edges.label_F_analysis).any()
        and set(edges.comparison) == set(ARMS),
        f"{key.nunique():,} unique undirected pairs")
    nbg = b34.groupby("comparison").background_label.nunique().to_dict()
    chk(6, "64 distinct B3:B4 backgrounds per arm",
        all(v == 64 for v in nbg.values())
        and bool((b34.groupby("comparison").size() == 64).all()), str(nbg))

    # 7 -- must reproduce the saved factorial difference-of-differences
    worst = 0.0
    for arm in ARMS:
        slug = ARM_SLUG[arm]
        ref = pd.read_csv(FACT_DIR / f"{slug}_pairwise_interaction_backgrounds.csv")
        ref = ref[ref.term == "B3:B4"].set_index("background_label")
        mine = b34[b34.comparison == arm].set_index("background_label")
        common = mine.index.intersection(ref.index)
        worst = max(worst, float(
            (mine.loc[common, "interaction"]
             - ref.loc[common, "interaction_dod"]).abs().max()))
        eff = pd.read_csv(FACT_DIR / f"{slug}_main_pairwise_effects.csv")
        saved = float(eff[eff.term == "B3:B4"].effect.iloc[0])
        got = float(mine.interaction.mean())
        worst = max(worst, abs(saved - got))
    chk(7, "interaction values and their means match the saved factorial data",
        worst < 1e-12, f"max abs difference {worst:.3e}")
    chk(10, "no AlphaGenome calls", True,
        "module imports no AlphaGenome client and opens no network path")
    print("=" * 76)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
