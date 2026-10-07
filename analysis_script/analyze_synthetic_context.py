"""
Analysis + figures for the synthetic-exon context experiments (JOB 1 and JOB 2).

Read-only with respect to scoring: no AlphaGenome calls, no regeneration.

``--endpoints``
    3-panel blue hist2d with IDENTICAL x/y limits (PNAS | TP53_e6/FAS |
    TP53_e7/FAS), paired context scatters (FAS prod_logit vs PNAS-context
    prod_logit), and an exploratory saturation-detector run per context.

``--chimeras``
    Per-context summary across all 64 labels x 2 comparisons, a factorial
    summary of each block's marginal effect plus pairwise interactions, a
    compact sorted plateau-level figure, and hist2d diagnostics for a few
    informative contexts.

Saturation detector note
------------------------
The detector is run with ``min_levels=0`` only. That gate exists to refuse
copy-number series with too few doses; this experiment has no dose axis at all
(x is continuous PNAS score), so the gate is inapplicable rather than being
relaxed. Every scientific threshold -- flatness, branch, slope, plateau extent
-- is left at its calibrated default.
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

OUT_DIR = ROOT / "outputs" / "synthetic_context"
PLOT_DIR = ROOT / "outputs" / "plots" / "synthetic_context"
ENDPOINT_CSV = OUT_DIR / "synthetic_endpoint_ag_results.csv"
CHIMERA_CSV = OUT_DIR / "chimera_ag_results.csv"

X, Y = "PNAS_common", "AG_prod_logit"
X_LABEL = "PNAS pre-tuner (PNAS_common)"
Y_LABEL = "AlphaGenome prod_logit\nlogit(P(acceptor) × P(donor))"
CONTEXTS = ["PNAS", "TP53_e6", "TP53_e7"]


def _detect(df: pd.DataFrame, name: str) -> sd.Result:
    d = df.rename(columns={X: sd.X_COL})
    return sd.detect(d, name, "", sd.Thresholds(min_levels=0))


# ==========================================================================
# endpoints
# ==========================================================================

def endpoints() -> None:
    d = pd.read_csv(ENDPOINT_CSV).replace([np.inf, -np.inf], np.nan).dropna(subset=[X, Y])
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    xlim = (d[X].min(), d[X].max())
    ylim = (d[Y].min(), d[Y].max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    for ax, ctx in zip(axes, CONTEXTS):
        s = d[d.context == ctx]
        h = ax.hist2d(s[X], s[Y], bins=50, range=[xlim, ylim],
                      cmap="Blues", cmin=1, norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label="Count")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(X_LABEL); ax.set_ylabel(Y_LABEL)
        ax.set_title(f"{ctx}   (n={len(s):,})", fontsize=11)
        ax.set_box_aspect(1)
    fig.suptitle("Same 10,000 synthetic exons, three full contexts — "
                 "does context set the AlphaGenome ceiling?", fontsize=13)
    fig.savefig(PLOT_DIR / "endpoints_3panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # paired: FAS context vs PNAS context, same exon
    piv = d.pivot_table(index="sequence_id", columns="context", values=Y)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)
    for ax, tgt in zip(axes, ["TP53_e6", "TP53_e7"]):
        p = piv[[tgt, "PNAS"]].dropna()
        h = ax.hist2d(p[tgt], p["PNAS"], bins=50, cmap="Blues", cmin=1, norm=LogNorm())
        fig.colorbar(h[3], ax=ax, label="Count")
        lo = min(p.min().min(), p.min().min()); hi = max(p.max().max(), p.max().max())
        ax.plot([lo, hi], [lo, hi], color="crimson", lw=1.4, ls="--", label="y = x")
        ax.set_xlabel(f"{tgt}/FAS context prod_logit")
        ax.set_ylabel("PNAS context prod_logit")
        ax.set_title(f"PNAS vs {tgt}/FAS  (n={len(p):,})", fontsize=11)
        ax.legend(fontsize=8); ax.set_box_aspect(1)
    fig.suptitle("Paired by synthetic exon: context shift in AlphaGenome prod_logit",
                 fontsize=13)
    fig.savefig(PLOT_DIR / "endpoints_paired.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows = []
    print("\n=== endpoint summary (exploratory saturation detector) ===")
    for ctx in CONTEXTS:
        s = d[d.context == ctx]
        r = _detect(s, ctx)
        rows.append({
            "context": ctx, "n": len(s),
            "AG_median": s[Y].median(), "AG_p05": s[Y].quantile(.05),
            "AG_p95": s[Y].quantile(.95),
            "AG_min": s[Y].min(), "AG_max": s[Y].max(),
            "classification": r.classification,
            "response_range": r.global_response_range,
            "branch_score": r.branch_score,
            "left_level": r.left.level, "left_censored": r.left.censored,
            "right_level": r.right.level, "right_censored": r.right.censored,
            "confidence": r.confidence, "warnings": r.warnings,
        })
        print(f"  {ctx:9} n={len(s):6,}  median={s[Y].median():7.3f}  "
              f"{r.classification:24} L={r.left.level if r.left.present else float('nan'):7.3f} "
              f"R={r.right.level if r.right.present else float('nan'):7.3f}  {r.confidence}")
        if r.warnings:
            print(f"            warnings: {r.warnings}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "endpoint_summary.csv", index=False)
    print(f"\nfigures -> {PLOT_DIR}")


# ==========================================================================
# chimeras
# ==========================================================================

def chimeras() -> None:
    ch = pd.read_csv(CHIMERA_CSV)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # fold in the endpoint contexts already scored by JOB 1
    if ENDPOINT_CSV.exists():
        ep = pd.read_csv(ENDPOINT_CSV)
        ids = set(ch.sequence_id)
        add = []
        for tgt in ["TP53_e6", "TP53_e7"]:
            for ctx, lab in ((("PNAS"), "PPPPPP"), (tgt, "FFFFFF")):
                s = ep[(ep.context == ctx) & (ep.sequence_id.isin(ids))].copy()
                s["comparison"] = tgt
                s["context_label"] = lab
                for i, b in enumerate(L.BLOCK_ORDER):
                    s[b] = lab[i]
                s["n_F_blocks"] = lab.count("F")
                add.append(s)
        ch = pd.concat([ch] + add, ignore_index=True)

    rows = []
    for (cmp_, lab), g in ch.groupby(["comparison", "context_label"]):
        r = _detect(g, f"{cmp_}|{lab}")
        rows.append({
            "comparison": cmp_, "context_id": lab,
            **{b: lab[i] for i, b in enumerate(L.BLOCK_ORDER)},
            "n_F_blocks": lab.count("F"), "n": len(g),
            "AG_median": g[Y].median(),
            "AG_IQR": g[Y].quantile(.75) - g[Y].quantile(.25),
            "AG_response_range": r.global_response_range,
            "classification": r.classification,
            "left_level": r.left.level, "left_censored": r.left.censored,
            "right_level": r.right.level, "right_censored": r.right.censored,
            "plateau_estimate": r.left.level if r.left.present else r.right.level,
            "branch_score": r.branch_score,
            "confidence": r.confidence, "warnings": r.warnings,
            "PNAS_contextual_mean": g.PNAS_contextual.mean()
                if "PNAS_contextual" in g else np.nan,
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT_DIR / "chimera_context_summary.csv", index=False)

    # factorial effects on the context median
    eff = []
    for cmp_, g in summ.groupby("comparison"):
        base = g.AG_median
        for b in L.BLOCK_ORDER:
            f_ = g[g[b] == "F"].AG_median.mean()
            p_ = g[g[b] == "P"].AG_median.mean()
            eff.append({"comparison": cmp_, "term": b, "order": 1,
                        "effect_F_minus_P": f_ - p_})
        for b1, b2 in itertools.combinations(L.BLOCK_ORDER, 2):
            m = {(a, c): g[(g[b1] == a) & (g[b2] == c)].AG_median.mean()
                 for a in "PF" for c in "PF"}
            inter = (m[("F", "F")] - m[("F", "P")]) - (m[("P", "F")] - m[("P", "P")])
            eff.append({"comparison": cmp_, "term": f"{b1}x{b2}", "order": 2,
                        "effect_F_minus_P": inter})
    effdf = pd.DataFrame(eff)
    effdf.to_csv(OUT_DIR / "chimera_factorial_effects.csv", index=False)

    print("\n=== marginal block effects on context median AG prod_logit "
          "(F minus P; negative = moves toward FAS-like) ===")
    piv = (effdf[effdf.order == 1]
           .pivot(index="term", columns="comparison", values="effect_F_minus_P")
           .reindex(L.BLOCK_ORDER))
    print(piv.round(3).to_string())
    print("\n=== strongest pairwise interactions ===")
    top = (effdf[effdf.order == 2].reindex(
        effdf[effdf.order == 2].effect_F_minus_P.abs().sort_values(ascending=False).index))
    print(top.head(8).round(3).to_string(index=False))

    # compact sorted plateau/median figure, one panel per comparison
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    for ax, cmp_ in zip(axes, ["TP53_e6", "TP53_e7"]):
        g = summ[summ.comparison == cmp_].sort_values("AG_median")
        colors = ["#2f6fd0" if n == 0 else "#d62828" if n == 6 else "#9aa0a6"
                  for n in g.n_F_blocks]
        ax.bar(range(len(g)), g.AG_median, color=colors)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.context_id, rotation=90, fontsize=6, family="monospace")
        ax.set_ylabel("median AG prod_logit")
        ax.set_title(f"PNAS ↔ {cmp_}/FAS — all 64 contexts "
                     f"(blue = PPPPPP, red = FFFFFF), block order {' '.join(L.BLOCK_ORDER)}",
                     fontsize=11)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    fig.savefig(PLOT_DIR / "chimera_sorted_levels.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # hist2d diagnostics for informative contexts
    want = ["PPPPPP", "FFFFFF", "PPFPPP", "PPPFPP", "PPFFPP", "FFPPPP", "PPPPFF"]
    for cmp_ in ["TP53_e6", "TP53_e7"]:
        sel = [w for w in want if w in set(summ[summ.comparison == cmp_].context_id)]
        if not sel:
            continue
        fig, axes = plt.subplots(1, len(sel), figsize=(4.3 * len(sel), 4.8),
                                 constrained_layout=True, squeeze=False)
        sub = ch[ch.comparison == cmp_]
        xlim = (sub[X].min(), sub[X].max()); ylim = (sub[Y].min(), sub[Y].max())
        for ax, lab in zip(axes[0], sel):
            s = sub[sub.context_label == lab]
            ax.hist2d(s[X], s[Y], bins=30, range=[xlim, ylim],
                      cmap="Blues", cmin=1, norm=LogNorm())
            ax.set_title(lab, fontsize=10, family="monospace")
            ax.set_xlabel(X_LABEL, fontsize=8)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        axes[0][0].set_ylabel(Y_LABEL, fontsize=8)
        fig.suptitle(f"PNAS ↔ {cmp_}/FAS — selected chimera contexts", fontsize=12)
        fig.savefig(PLOT_DIR / f"chimera_diagnostics_{cmp_}.png", dpi=170,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"\nwrote chimera summaries -> {OUT_DIR}")
    print(f"figures -> {PLOT_DIR}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyse the synthetic-context experiments.")
    ap.add_argument("--endpoints", action="store_true")
    ap.add_argument("--chimeras", action="store_true")
    args = ap.parse_args(argv)
    if not (args.endpoints or args.chimeras):
        args.endpoints = args.chimeras = True
    if args.endpoints and ENDPOINT_CSV.exists():
        endpoints()
    elif args.endpoints:
        print(f"[skip] {ENDPOINT_CSV} not found")
    if args.chimeras and CHIMERA_CSV.exists():
        chimeras()
    elif args.chimeras:
        print(f"[skip] {CHIMERA_CSV} not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
