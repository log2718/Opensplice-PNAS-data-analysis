"""
Mean-logit plotting + saturation analysis for the synthetic-context experiments.

READ-ONLY with respect to scoring: no AlphaGenome calls, no regeneration of the
synthetic library. Everything here is computed from the completed result CSVs.

SCIENTIFIC NOTE: this module uses AlphaGenome **mean_logit** exclusively --
``(logit(P_acceptor) + logit(P_donor)) / 2``. ``prod_logit`` appears nowhere in
any plot, estimate, summary or label produced here. The earlier prod_logit
analysis remains reproducible through ``saturation_detector`` defaults.

Outputs
-------
``outputs/synthetic_context/plots/original_context/``  3-panel original-context figure
``outputs/synthetic_context/plots/chimeras/e6/`` 64 individual figures
``outputs/synthetic_context/plots/chimeras/e7/`` 64 individual figures
``outputs/synthetic_context/plots/summary/``     classification + distribution figures
``outputs/synthetic_context/mean_logit_saturation/`` saturation + factorial CSVs

All 128 chimera figures and the 3 endpoint panels share ONE set of x/y limits
computed from the pooled endpoint + chimera data, so plateau heights can be
compared by eye across contexts.

Block label order is ``L1 L2 L3 R3 R2 R1`` -- verified against the explicit
per-block columns in the result CSV, not assumed.
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
from matplotlib.colors import LogNorm  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import synthetic_context_lib as L  # noqa: E402
import saturation_detector as sd  # noqa: E402

OUT = ROOT / "outputs" / "synthetic_context"
ENDPOINT_CSV = OUT / "synthetic_endpoint_ag_results.csv"
CHIMERA_CSV = OUT / "chimera_ag_results.csv"
SAT_DIR = OUT / "mean_logit_saturation"
P_END = OUT / "plots" / "original_context"
P_CHI = OUT / "plots" / "chimeras"
P_SUM = OUT / "plots" / "summary"

X = "PNAS_common"
Y = "AG_mean_logit"                      # mean-logit ONLY
X_LABEL = "PNAS pre-tuner score (common)\n(energy_seq_struct)"
Y_LABEL = "AlphaGenome mean-logit\n(logit P(acceptor) + logit P(donor)) / 2"
BINS = 40                                 # repo convention
ENDPOINTS = {"PNAS": "PPPPPP", "TP53_e6": "FFFFFF", "TP53_e7": "FFFFFF"}
SERIES = {"TP53_e6": "e6", "TP53_e7": "e7"}


# ==========================================================================
# data assembly
# ==========================================================================

def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Endpoint frame, and the full 128-context chimera frame with endpoints folded in."""
    ep = pd.read_csv(ENDPOINT_CSV)
    ch = pd.read_csv(CHIMERA_CSV)

    # verify the six-bit label really maps to L1 L2 L3 R3 R2 R1
    bad = [b for i, b in enumerate(L.BLOCK_ORDER)
           if not (ch.context_label.str[i] == ch[b]).all()]
    if bad:
        raise AssertionError(f"block label order mismatch at {bad}")

    subset = set(ch.sequence_id)
    add = []
    for tgt in SERIES:
        for ctx, lab in (("PNAS", "PPPPPP"), (tgt, "FFFFFF")):
            s = ep[(ep.context == ctx) & (ep.sequence_id.isin(subset))].copy()
            s["comparison"] = tgt
            s["context_label"] = lab
            for i, b in enumerate(L.BLOCK_ORDER):
                s[b] = lab[i]
            s["n_F_blocks"] = lab.count("F")
            add.append(s)
    full = pd.concat([ch] + add, ignore_index=True)
    return ep, full


def global_limits(ep: pd.DataFrame, full: pd.DataFrame) -> tuple[tuple, tuple]:
    xs = pd.concat([ep[X], full[X]])
    ys = pd.concat([ep[Y], full[Y]])
    return ((xs.min(), xs.max()), (ys.min(), ys.max()))


def _hist_panel(ax, s: pd.DataFrame, fig, xlim, ylim, title: str,
                cbar: bool = True) -> None:
    h = ax.hist2d(s[X], s[Y], bins=BINS, range=[xlim, ylim],
                  cmap="Blues", cmin=1, norm=LogNorm())
    if cbar:
        fig.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    # Axes title carries the explanatory line; the main title is a suptitle, so
    # constrained_layout reserves space for both and neither can overlap the plot.
    ax.set_title(title, fontsize=9, color="0.25", linespacing=1.35)
    ax.set_box_aspect(1)


# ==========================================================================
# PART 1 -- endpoint three-panel
# ==========================================================================

def part1_endpoints(ep: pd.DataFrame, xlim, ylim) -> int:
    P_END.mkdir(parents=True, exist_ok=True)
    order = [("PNAS", "PNAS_original_1309"), ("TP53_e6", "FAS_TP53_e6"),
             ("TP53_e7", "FAS_TP53_e7")]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.6), constrained_layout=True)
    for ax, (ctx, nice) in zip(axes, order):
        s = ep[ep.context == ctx]
        _hist_panel(ax, s, fig, xlim, ylim,
                    f"{nice}\nn = {len(s):,}")
    fig.suptitle("Original context — AlphaGenome mean-logit",
                 fontsize=14, fontweight="bold")
    out = P_END / "original_context_three_panel_mean_logit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[part1] wrote {out}")
    return 1


# ==========================================================================
# PART 2 -- 128 individual chimera figures
# ==========================================================================

def part2_chimeras(full: pd.DataFrame, xlim, ylim) -> dict[str, int]:
    counts = {}
    for tgt, tag in SERIES.items():
        d = P_CHI / tag
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for lab in L.all_labels():
            s = full[(full.comparison == tgt) & (full.context_label == lab)]
            if s.empty:
                print(f"[part2] MISSING {tag} {lab}")
                continue
            fig, ax = plt.subplots(figsize=(6.8, 6.6), constrained_layout=True)
            subtitle = f"P = PNAS block, F = FAS block   |   n = {len(s):,}"
            _hist_panel(ax, s, fig, xlim, ylim, subtitle)
            fig.suptitle(f"{tgt} chimera context: {lab}",
                         fontsize=13, fontweight="bold")
            fig.savefig(d / f"{tag}_{lab}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            n += 1
        counts[tag] = n
        print(f"[part2] wrote {n} figures -> {d}")
    return counts


# ==========================================================================
# PART 4 -- saturation detection on all 128 contexts
# ==========================================================================

def part4_saturation(full: pd.DataFrame) -> pd.DataFrame:
    t = sd.Thresholds.for_metric("mean_logit", min_levels=0)
    rows = []
    for (tgt, lab), g in full.groupby(["comparison", "context_label"]):
        d = g.rename(columns={X: sd.X_COL})
        r = sd.detect(d, tgt, lab, t, y_col=Y)
        glob = r.classification == "GLOBAL_SATURATION"
        rows.append({
            "comparison": tgt, "series": SERIES[tgt], "context_label": lab,
            **{b: lab[i] for i, b in enumerate(L.BLOCK_ORDER)},
            "n_F_blocks": lab.count("F"), "n_points": len(g),
            "classification": r.classification, "subtype": r.subtype,
            "left_plateau_present": r.left.present and not glob,
            "left_plateau_mean_logit": r.left.level if (r.left.present and not glob) else np.nan,
            "left_CI_low": r.left.ci_low if (r.left.present and not glob) else np.nan,
            "left_CI_high": r.left.ci_high if (r.left.present and not glob) else np.nan,
            "left_x_min": r.left.x_min if (r.left.present and not glob) else np.nan,
            "left_x_max": r.left.x_max if (r.left.present and not glob) else np.nan,
            "left_onset_x": r.left.onset_x if (r.left.present and not glob) else np.nan,
            "left_censored": r.left.censored if (r.left.present and not glob) else False,
            "right_plateau_present": r.right.present,
            "right_plateau_mean_logit": r.right.level if r.right.present else np.nan,
            "right_CI_low": r.right.ci_low if r.right.present else np.nan,
            "right_CI_high": r.right.ci_high if r.right.present else np.nan,
            "right_x_min": r.right.x_min if r.right.present else np.nan,
            "right_x_max": r.right.x_max if r.right.present else np.nan,
            "right_onset_x": r.right.onset_x if r.right.present else np.nan,
            "right_censored": r.right.censored if r.right.present else False,
            "global_plateau_mean_logit": r.left.level if glob else np.nan,
            "global_CI_low": r.left.ci_low if glob else np.nan,
            "global_CI_high": r.left.ci_high if glob else np.nan,
            "response_y_range": r.global_response_range,
            "global_slope": r.global_slope,
            "branch_score": r.branch_score,
            "multimodal_flag": r.multimodal_flag,
            "flatness_score": r.flatness_score,
            "confidence": r.confidence, "warnings": r.warnings,
            "median_mean_logit": g[Y].median(),
            "IQR_mean_logit": g[Y].quantile(.75) - g[Y].quantile(.25),
        })
    sat = pd.DataFrame(rows)
    SAT_DIR.mkdir(parents=True, exist_ok=True)
    sat.to_csv(SAT_DIR / "chimera_mean_logit_saturation.csv", index=False)
    print(f"[part4] wrote {len(sat)} rows -> "
          f"{SAT_DIR / 'chimera_mean_logit_saturation.csv'}")
    return sat


# ==========================================================================
# PART 5 -- single median-distribution summary
# ==========================================================================

def part5_median_distribution(ep: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Histogram of the median mean-logit of all 131 contexts, plus the table.

    131 = 128 chimera contexts (64 per series, each on the 400-exon subset)
    + the 3 original full contexts (each on all 10,000 exons). The chimera set
    already contains PPPPPP / FFFFFF on the subset, so the three originals are
    the independent full-library versions of those same constructs, not extras.
    """
    P_SUM.mkdir(parents=True, exist_ok=True)

    rows = []
    for ctx, nice in (("PNAS", "original_PNAS_1309"),
                      ("TP53_e6", "original_FAS_TP53_e6"),
                      ("TP53_e7", "original_FAS_TP53_e7")):
        g = ep[ep.context == ctx]
        rows.append({"context_name": nice, "group": "original",
                     "median_AG_mean_logit": g[Y].median(), "n": len(g)})
    for (tgt, lab), g in full.groupby(["comparison", "context_label"]):
        rows.append({"context_name": f"{SERIES[tgt]}_{lab}", "group": SERIES[tgt],
                     "median_AG_mean_logit": g[Y].median(), "n": len(g)})
    tab = pd.DataFrame(rows).sort_values("median_AG_mean_logit").reset_index(drop=True)
    out_csv = P_SUM / "context_median_mean_logit.csv"
    tab[["context_name", "median_AG_mean_logit", "group", "n"]].to_csv(out_csv, index=False)

    style = {"e6": ("#f0c000", 46, "e6 chimeras"),
             "e7": ("#2f6fd0", 46, "e7 chimeras"),
             "original": ("black", 120, "10k endpoint reference runs")}
    chim = tab[tab.group != "original"]
    endp = tab[tab.group == "original"]

    # Two rows on a shared x. The y axes mean different things, so they get
    # separate panels rather than being overlaid: counts on top, one dot per
    # context on the bottom strip (its y is categorical jitter only).
    fig, (ax_h, ax_s) = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        constrained_layout=True)

    ax_h.hist(chim.median_AG_mean_logit, bins=26, color="0.84",
              edgecolor="white", linewidth=.8, zorder=1)
    for _, r in endp.iterrows():
        ax_h.axvline(r.median_AG_mean_logit, color="black", lw=1.4,
                     ls="--", zorder=3)
    ax_h.plot([], [], color="black", lw=1.4, ls="--",
              label=f"{style['original'][2]} (n={len(endp)})")
    ax_h.set_ylabel("number of contexts")
    ax_h.set_title(f"chimera contexts, n = {len(chim)}", fontsize=9, color="0.25")
    ax_h.legend(frameon=False, fontsize=9, loc="upper left")
    ax_h.grid(axis="y", alpha=.25)
    ax_h.set_axisbelow(True)

    rng = np.random.default_rng(0)
    for grp in ("e6", "e7", "original"):
        g = tab[tab.group == grp]
        col, size, lab = style[grp]
        yy = rng.uniform(0.15, 0.85, len(g))
        ax_s.scatter(g.median_AG_mean_logit, yy, s=size, color=col,
                     edgecolor="black", linewidth=.55 if grp != "original" else .9,
                     zorder=4 if grp == "original" else 3,
                     label=f"{lab} (n={len(g)})")
    ax_s.set_ylim(0, 1)
    ax_s.set_yticks([])
    ax_s.set_ylabel("contexts", fontsize=9)
    ax_s.set_xlabel("median AlphaGenome mean-logit")
    ax_s.legend(frameon=False, fontsize=8.5, loc="upper center",
                bbox_to_anchor=(0.5, -0.32), ncol=3)
    ax_s.grid(axis="x", alpha=.25)
    ax_s.set_axisbelow(True)

    fig.suptitle("Distribution of context median AlphaGenome mean-logit",
                 fontsize=14, fontweight="bold")
    fig.savefig(P_SUM / "context_median_mean_logit_distribution.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[part5] wrote 1 figure + {out_csv.name} ({len(tab)} contexts) -> {P_SUM}")
    return tab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mean-logit analysis of the synthetic-context experiments.")
    ap.add_argument("--y-col", default=Y,
                    help="AlphaGenome metric (default AG_mean_logit)")
    ap.add_argument("--skip-chimera-figures", action="store_true")
    ap.add_argument("--saturation", action="store_true",
                    help="also re-run the saturation detector (separate folder)")
    args = ap.parse_args(argv)
    globals()["Y"] = args.y_col

    ep, full = load_all()
    xlim, ylim = global_limits(ep, full)
    print(f"global limits  x=({xlim[0]:.2f}, {xlim[1]:.2f})  "
          f"y=({ylim[0]:.2f}, {ylim[1]:.2f})  [metric={Y}]\n")

    n_end = part1_endpoints(ep, xlim, ylim)
    n_chi = {"e6": 0, "e7": 0}
    if not args.skip_chimera_figures:
        n_chi = part2_chimeras(full, xlim, ylim)
    part5_median_distribution(ep, full)
    if args.saturation:
        part4_saturation(full)

    print(f"\nendpoint figures: {n_end}   e6 chimera figures: {n_chi.get('e6',0)}   "
          f"e7 chimera figures: {n_chi.get('e7',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
