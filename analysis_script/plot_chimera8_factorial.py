"""
Figures for the 8-block factorial analysis: ranked median+SE, and effect heatmaps.

Parts 2 and 5. Makes NO AlphaGenome calls; reads only the CSVs written by
``analyze_chimera8_factorial.py`` plus the 6-block endpoint file (read-only).

Part 2  one ranked forest figure per arm. All 256 contexts appear exactly once,
        sorted ascending by observed median AG mean-logit. The grid is filled
        left to right, then top to bottom, so reading order IS rank order.
        Dot = observed median, horizontal bar = +/- 1 bootstrap SE (horizontal
        because AG mean-logit is the x quantity). Every row is labelled with
        ``context_label_8_display``; nothing is omitted.

Part 5  one 8x8 symmetric heatmap per arm: main effects on the diagonal,
        pairwise interactions off-diagonal, every cell annotated. Both arms
        share ONE diverging colour scale centred at 0, so the two figures are
        directly comparable.

Usage::

    python analysis_script/plot_chimera8_factorial.py
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

NEW = ROOT / "outputs" / "synthetic_context_8block"
BOOT_CSV = NEW / "bootstrap" / "context_median_bootstrap_SE.csv"
FACT_DIR = NEW / "factorial"
P_SUM = NEW / "plots" / "summary"
P_FACT = NEW / "plots" / "factorial"
OLD_ENDPOINT_CSV = (ROOT / "outputs" / "synthetic_context"
                    / "synthetic_endpoint_ag_results.csv")

ARMS = ["TP53_e6", "TP53_e7"]
ARM_SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
BLOCKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
BLOCK_SHORT = {
    "B1": "B1 Up exon body", "B2": "B2 Up 5'SS", "B3": "B3 Up intron",
    "B4": "B4 Mid 3'SS", "B5": "B5 Mid 5'SS", "B6": "B6 Down intron",
    "B7": "B7 Down 3'SS", "B8": "B8 Down exon body",
}

#: carried over from every other figure in this series
ARM_COLOR = {"TP53_e6": "#f0c000", "TP53_e7": "#2f6fd0"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

N_COLS = 4          # 4 columns x 64 rows = 256 contexts
#: diverging: cool for negative, warm for positive, neutral (not a hue) at 0
DIVERGING = LinearSegmentedColormap.from_list(
    "effect_div", ["#1b4f9c", "#2a78d6", "#9dc3ec", "#f2f1ed",
                   "#f7b48f", "#eb6834", "#a83c12"])


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(direction="out", colors=MUTED)
    return ax


# ==========================================================================
# PART 2 -- ranked median + SE forest grid
# ==========================================================================

def plot_ranked(boot: pd.DataFrame, arm: str, ref: dict[str, float],
                path: Path) -> None:
    """256 contexts, ascending by median, filled left->right then top->bottom.

    Rank k lands at column ``k % 4``, row ``k // 4``, so reading the grid the
    way you read text gives the ranks in order. Each column therefore holds
    every 4th rank, which its header states explicitly.
    """
    s = (boot[boot.comparison == arm]
         .sort_values("observed_median_AG_mean_logit", ascending=True)
         .reset_index(drop=True))
    n = len(s)
    col = ARM_COLOR[arm]

    # one shared x across every column, with room for the error bars
    lo = (s.observed_median_AG_mean_logit - s.bootstrap_SE_median).min()
    hi = (s.observed_median_AG_mean_logit + s.bootstrap_SE_median).max()
    lo, hi = min(lo, ref["FAS"], ref["PNAS"]), max(hi, ref["FAS"], ref["PNAS"])
    pad = (hi - lo) * 0.05
    xlim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, N_COLS, figsize=(19, 14), dpi=150, sharex=True)
    fig.subplots_adjust(top=0.905, bottom=0.075, left=0.05, right=0.995,
                        wspace=0.40)

    for c, ax in enumerate(axes):
        idx = [k for k in range(n) if k % N_COLS == c]
        g = s.iloc[idx]
        y = np.arange(len(idx))

        ax.axvline(ref["FAS"], color="black", lw=1.3, ls="--", zorder=1)
        ax.axvline(ref["PNAS"], color=MUTED, lw=1.2, ls=":", zorder=1)
        ax.errorbar(g.observed_median_AG_mean_logit, y,
                    xerr=g.bootstrap_SE_median, fmt="o", ms=4.4,
                    mfc=col, mec="black", mew=.5,
                    ecolor=MUTED, elinewidth=1.0, capsize=1.6, zorder=3)
        ax.set_yticks(y, g.context_label_8_display, fontsize=6.5,
                      fontfamily="monospace")
        ax.set_ylim(len(idx) - 0.5, -0.5)          # best rank at the top
        ax.set_xlim(*xlim)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", length=2, pad=1.5)
        ax.grid(axis="x", color=GRID, lw=.6, alpha=.8)
        ax.set_axisbelow(True)
        ax.set_title(f"ranks {idx[0] + 1}, {idx[1] + 1}, {idx[2] + 1}, …, "
                     f"{idx[-1] + 1}", fontsize=8.5, color=MUTED, pad=5)
        _despine(ax)

    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=6, mfc=col,
                   mec="black", mew=.5,
                   label=f"{ARM_SLUG[arm]} context median ± 1 bootstrap SE "
                         "(2,000 replicates)"),
        plt.Line2D([], [], color="black", lw=1.3, ls="--",
                   label=f"10k {arm} endpoint median ({ref['FAS']:.2f})"),
        plt.Line2D([], [], color=MUTED, lw=1.2, ls=":",
                   label=f"10k PNAS endpoint median ({ref['PNAS']:.2f})"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.006))
    fig.text(0.5, 0.040, "median AlphaGenome mean-logit", ha="center",
             fontsize=10.5)
    fig.text(0.5, 0.973,
             f"{arm} — 256 corrected 8-block contexts ranked by median "
             "AlphaGenome mean-logit",
             ha="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.5, 0.952,
             "ascending rank fills left→right, then top→bottom, so each column "
             f"holds every 4th rank   ·   B4/B5 shown as "
             f"'{ARM_SLUG[arm][-1]}' where that splice site is {arm}-derived "
             "(display only)",
             ha="center", fontsize=9.5, color=MUTED)
    fig.savefig(path)
    plt.close(fig)


# ==========================================================================
# PART 5 -- 8x8 main-effect / interaction heatmap
# ==========================================================================

def effect_matrix(eff: pd.DataFrame) -> np.ndarray:
    """Diagonal = main effect, off-diagonal = pairwise interaction."""
    m = np.full((8, 8), np.nan)
    pos = {b: i for i, b in enumerate(BLOCKS)}
    for r in eff.itertuples():
        if r.order == 1:
            m[pos[r.block_i], pos[r.block_i]] = r.effect
        else:
            i, j = pos[r.block_i], pos[r.block_j]
            m[i, j] = m[j, i] = r.effect
    return m


def plot_heatmap(m: np.ndarray, arm: str, vmax: float, path: Path) -> None:
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9.6, 8.4), dpi=190)
    im = ax.imshow(m, cmap=DIVERGING, norm=norm)

    for i in range(8):
        for j in range(8):
            v = m[i, j]
            # white text only where the fill is dark enough to need it
            tc = "white" if abs(v) > vmax * 0.62 else INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9.2, color=tc,
                    fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=INK, lw=1.8, zorder=4))

    labels = [BLOCK_SHORT[b] for b in BLOCKS]
    ax.set_xticks(range(8), labels, rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(8), labels, fontsize=9)
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=MUTED)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("effect on context median AlphaGenome mean-logit",
                 fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    digit = ARM_SLUG[arm][-1]
    ax.set_title(
        "diagonal: mean(median | Bi=F) − mean(median | Bi=P)   ·   "
        "off-diagonal: mean over 64 backgrounds of "
        "y$_{FF}$ − y$_{FP}$ − y$_{PF}$ + y$_{PP}$\n"
        "negative = swapping in the FAS block lowers the context median   ·   "
        f"F at B4/B5 is the TP53_e{digit} splice site "
        "(display note only; all maths uses P/F)",
        fontsize=9, color=MUTED, pad=13, linespacing=1.6)
    ax.title.set_color(MUTED)
    # the first title line carries the figure name, so re-draw it bold on top
    ax.annotate(f"{arm} — block main effects (boxed diagonal) and pairwise "
                "interactions",
                xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 42), textcoords="offset points",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    P_SUM.mkdir(parents=True, exist_ok=True)
    P_FACT.mkdir(parents=True, exist_ok=True)

    boot = pd.read_csv(BOOT_CSV)
    ep = pd.read_csv(OLD_ENDPOINT_CSV)
    ep_med = ep.groupby("context").AG_mean_logit.median().to_dict()

    # ---- Part 2 ----------------------------------------------------------
    print("[part 2] ranked median + SE forest grids")
    for arm in ARMS:
        ref = {"FAS": ep_med[arm], "PNAS": ep_med["PNAS"]}
        path = P_SUM / f"{ARM_SLUG[arm]}_ranked_context_median_SE.png"
        plot_ranked(boot, arm, ref, path)
        s = boot[boot.comparison == arm]
        assert len(s) == 256 and s.context_label_8_analysis.is_unique
        print(f"  {ARM_SLUG[arm]}: 256 contexts, each drawn once -> "
              f"{path.relative_to(ROOT)}")

    # ---- Part 5 ----------------------------------------------------------
    print("\n[part 5] 8x8 effect heatmaps (one shared colour scale)")
    mats, effs = {}, {}
    for arm in ARMS:
        effs[arm] = pd.read_csv(FACT_DIR / f"{ARM_SLUG[arm]}_main_pairwise_effects.csv")
        mats[arm] = effect_matrix(effs[arm])
        assert not np.isnan(mats[arm]).any(), "incomplete effect matrix"
        assert np.allclose(mats[arm], mats[arm].T), "matrix not symmetric"
    vmax = float(max(np.abs(m).max() for m in mats.values()))
    print(f"  shared symmetric colour scale: [-{vmax:.4f}, +{vmax:.4f}] "
          "centred at 0")
    for arm in ARMS:
        path = P_FACT / f"{ARM_SLUG[arm]}_main_pairwise_effect_heatmap.png"
        plot_heatmap(mats[arm], arm, vmax, path)
        print(f"  {ARM_SLUG[arm]}: -> {path.relative_to(ROOT)}")

    # ---- SE distribution summary ----------------------------------------
    print("\n[part 2] bootstrap SE distribution")
    rows = []
    for arm in ARMS:
        s = boot[boot.comparison == arm].bootstrap_SE_median
        rows.append({"comparison": arm, "n_contexts": len(s),
                     "SE_min": s.min(), "SE_median": s.median(),
                     "SE_max": s.max(), "SE_mean": s.mean()})
    se = pd.DataFrame(rows)
    se_path = NEW / "bootstrap" / "bootstrap_SE_summary.csv"
    se.round(6).to_csv(se_path, index=False)
    print(se.round(4).to_string(index=False))
    print(f"  wrote {se_path.relative_to(ROOT)}")
    print("\nNo AlphaGenome calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
