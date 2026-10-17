"""
Figures for the exact-Shapley analysis of the 8-block chimera design.

Reads only the CSVs written by ``analyze_chimera8_shap.py``. Makes NO
AlphaGenome calls and fits no model.

Per arm:
  1. ``*_mean_absolute_shap.png``      global importance, mean(|phi|), sorted
  2. ``*_exact_shap_beeswarm.png``     every context as a point, P/F encoded
  3. ``*_shap_state_conditional.png``  phi split by block state (box + strip)
  4. ``*_mean_abs_shap_interaction_heatmap.png``
                                       mean |phi_ij| -- a DIFFERENT quantity
                                       from the factorial difference-of-
                                       differences heatmap, kept separate

Figures 1-3 share one x range across e6 and e7; figure 4 shares one colour
scale. All block states are the analytical P/F labels.

Usage::

    python analysis_script/plot_chimera8_shap.py
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

NEW = ROOT / "outputs" / "synthetic_context_8block"
SHAP_DIR = NEW / "shap"
P_SHAP = NEW / "plots" / "shap"

ARMS = ["TP53_e6", "TP53_e7"]
ARM_SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
BLOCKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
BLOCK_SHORT = {
    "B1": "B1 Up exon body", "B2": "B2 Up 5'SS", "B3": "B3 Up intron",
    "B4": "B4 Mid 3'SS", "B5": "B5 Mid 5'SS", "B6": "B6 Down intron",
    "B7": "B7 Down 3'SS", "B8": "B8 Down exon body",
}

#: arm identity, carried over from every other figure in this series
ARM_COLOR = {"TP53_e6": "#f0c000", "TP53_e7": "#2f6fd0"}
#: block STATE -- validated CVD-safe pair (normal 33.6, protan 24.7)
STATE_COLOR = {"P": "#2a78d6", "F": "#eb6834"}
STATE_LABEL = {"P": "P  (PNAS-derived block)", "F": "F  (FAS-derived block)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

SEQ_RAMP = LinearSegmentedColormap.from_list(
    "shap_mag", ["#f7f7f5", "#cfe0f3", "#8fb8e4", "#4a8ed2", "#2a78d6",
                 "#1b4f9c"])


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(direction="out", colors=MUTED)
    return ax


# ==========================================================================
# 1. global importance
# ==========================================================================

def plot_importance(summ: pd.DataFrame, arm: str, xmax: float,
                    path: Path, display: str | None = None) -> None:
    s = summ.sort_values("mean_abs_SHAP", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
    y = np.arange(len(s))
    ax.barh(y, s.mean_abs_SHAP, height=.66, color=ARM_COLOR[arm],
            edgecolor="black", linewidth=.5)
    for yi, v in zip(y, s.mean_abs_SHAP):
        ax.text(v + xmax * .012, yi, f"{v:.3f}", va="center", fontsize=9,
                color=INK)
    ax.set_yticks(y, [BLOCK_SHORT[b] for b in s.block], fontsize=9.5)
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel("mean |exact Shapley value|   (context median AG mean-logit)",
                  fontsize=10)
    ax.set_title(f"{display or arm} — global block importance, exact Shapley",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# 2. beeswarm-like distribution
# ==========================================================================

def plot_beeswarm(long: pd.DataFrame, order: list[str], arm: str,
                  xlim: tuple[float, float], path: Path,
                  display: str | None = None) -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=200)
    for k, b in enumerate(order):
        g = long[long.block == b]
        for st in ("P", "F"):
            gs = g[g.block_state == st]
            jitter = rng.uniform(-0.3, 0.3, len(gs))
            ax.scatter(gs.shap_value, k + jitter, s=11,
                       c=STATE_COLOR[st], alpha=.72, lw=0, zorder=3)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(range(len(order)), [BLOCK_SHORT[b] for b in order],
                  fontsize=9.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("exact Shapley value   (context median AG mean-logit)",
                  fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=7,
                          mfc=STATE_COLOR[s], mec="none", label=STATE_LABEL[s])
               for s in ("P", "F")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    ax.set_title("blocks ordered by mean |Shapley|; one point per context, "
                 "vertical jitter only",
                 fontsize=8.8, color=MUTED, pad=10)
    ax.annotate(f"{display or arm} — exact Shapley value per block, "
                "all 256 contexts",
                xy=(.5, 1.0), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# 3. state-conditional summary
# ==========================================================================

def plot_state_conditional(long: pd.DataFrame, order: list[str], arm: str,
                           xlim: tuple[float, float], path: Path,
                           display: str | None = None) -> None:
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=200)
    offs = {"P": -0.19, "F": 0.19}
    for k, b in enumerate(order):
        for st in ("P", "F"):
            gs = long[(long.block == b) & (long.block_state == st)].shap_value
            pos = k + offs[st]
            bp = ax.boxplot([gs.to_numpy()], positions=[pos], vert=False,
                            widths=0.3, patch_artist=True, manage_ticks=False,
                            medianprops=dict(color=INK, lw=1.4),
                            flierprops=dict(marker="", ms=0))
            for patch in bp["boxes"]:
                patch.set(facecolor=STATE_COLOR[st], alpha=.45,
                          edgecolor=STATE_COLOR[st])
            for w in bp["whiskers"] + bp["caps"]:
                w.set_color(MUTED)
            ax.scatter(gs, pos + rng.uniform(-0.07, 0.07, len(gs)), s=5,
                       c=STATE_COLOR[st], alpha=.5, lw=0, zorder=3)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(range(len(order)), [BLOCK_SHORT[b] for b in order],
                  fontsize=9.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("exact Shapley value   (context median AG mean-logit)",
                  fontsize=10)
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=8,
                          mfc=STATE_COLOR[s], mec="none", alpha=.7,
                          label=STATE_LABEL[s]) for s in ("P", "F")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    ax.set_title("a wide box within one state means the block's contribution "
                 "depends strongly on the other seven blocks",
                 fontsize=8.8, color=MUTED, pad=10)
    ax.annotate(f"{display or arm} — Shapley value split by block state",
                xy=(.5, 1.0), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# 4. mean |SHAP interaction| heatmap
# ==========================================================================

def plot_interaction_heatmap(m: np.ndarray, arm: str, vmax: float,
                             path: Path, display: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 8.2), dpi=190)
    im = ax.imshow(m, cmap=SEQ_RAMP, vmin=0, vmax=vmax)
    for i in range(8):
        for j in range(8):
            v = m[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.6,
                    color="white" if v > vmax * .60 else INK,
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
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
    cb.set_label("mean |exact SHAP interaction value| over 256 contexts",
                 fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("mean |exact SHAP interaction|   ·   boxed diagonal = "
                 "mean |main SHAP interaction term| phi_ii\n"
                 "NOT the factorial difference-of-differences heatmap — a "
                 "different quantity, on a different scale",
                 fontsize=9, color=MUTED, pad=12, linespacing=1.6)
    ax.annotate(f"{display or arm} — exact SHAP interaction importance",
                xy=(.5, 1.0), xycoords="axes fraction", xytext=(0, 34),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    P_SHAP.mkdir(parents=True, exist_ok=True)

    longs, summs, inters = {}, {}, {}
    for arm in ARMS:
        slug = ARM_SLUG[arm]
        longs[arm] = pd.read_csv(SHAP_DIR / f"{slug}_exact_shap_values.csv")
        summs[arm] = pd.read_csv(SHAP_DIR / f"{slug}_shap_feature_summary.csv")
        inters[arm] = np.load(SHAP_DIR / f"{slug}_shap_interaction_tensor.npy")

    # shared scales
    xmax = max(s.mean_abs_SHAP.max() for s in summs.values())
    lo = min(l.shap_value.min() for l in longs.values())
    hi = max(l.shap_value.max() for l in longs.values())
    pad = (hi - lo) * .05
    xlim = (lo - pad, hi + pad)
    mean_abs = {a: np.abs(inters[a]).mean(axis=tuple(range(2, 10)))
                for a in ARMS}
    vmax = float(max(m.max() for m in mean_abs.values()))
    print(f"shared mean|SHAP| x max : {xmax:.4f}")
    print(f"shared SHAP x range     : [{xlim[0]:.4f}, {xlim[1]:.4f}]")
    print(f"shared interaction vmax : {vmax:.4f}")

    for arm in ARMS:
        slug = ARM_SLUG[arm]
        order = summs[arm].sort_values("mean_abs_SHAP",
                                       ascending=False).block.tolist()
        plot_importance(summs[arm], arm, xmax,
                        P_SHAP / f"{slug}_mean_absolute_shap.png")
        plot_beeswarm(longs[arm], order, arm, xlim,
                      P_SHAP / f"{slug}_exact_shap_beeswarm.png")
        plot_state_conditional(longs[arm], order, arm, xlim,
                               P_SHAP / f"{slug}_shap_state_conditional.png")
        plot_interaction_heatmap(
            mean_abs[arm], arm, vmax,
            P_SHAP / f"{slug}_mean_abs_shap_interaction_heatmap.png")
        print(f"  {slug}: 4 figures -> {P_SHAP.relative_to(ROOT)}")

        # the mean |phi_ij| matrix as a CSV too
        rows = []
        for i, bi in enumerate(BLOCKS):
            for j, bj in enumerate(BLOCKS):
                rows.append({"comparison": arm, "block_i": bi, "block_j": bj,
                             "mean_abs_shap_interaction": mean_abs[arm][i, j]})
        pd.DataFrame(rows).to_csv(
            SHAP_DIR / f"{slug}_mean_abs_shap_interaction_matrix.csv",
            index=False)

    print("\nNo AlphaGenome calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
