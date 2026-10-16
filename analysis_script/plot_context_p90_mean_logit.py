"""
90th-percentile companion to the context median-distribution summary.

Same inputs, same 131 contexts, same layout and style as
``analyze_synthetic_mean_logit.part5_median_distribution`` -- the only change is
the per-context summary statistic:

    median  ->  90th percentile  of AG_mean_logit

Writes to NEW filenames; the median version is never touched::

    plots/summary/context_p90_mean_logit.csv
    plots/summary/context_p90_mean_logit_distribution.png

READ-ONLY with respect to AlphaGenome: no calls are made. Inputs are the
existing ``synthetic_endpoint_ag_results.csv`` and ``chimera_ag_results.csv``,
loaded through ``analyze_synthetic_mean_logit.load_all`` so the context set is
identical to the median figure's by construction.

Usage::

    python analysis_script/plot_context_p90_mean_logit.py
    python analysis_script/plot_context_p90_mean_logit.py --quantile 0.95
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

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import analyze_synthetic_mean_logit as A  # noqa: E402

#: the three original full-library contexts, in the median figure's order
ORIGINALS = (("PNAS", "original_PNAS_1309"),
             ("TP53_e6", "original_FAS_TP53_e6"),
             ("TP53_e7", "original_FAS_TP53_e7"))


def context_quantile_table(ep: pd.DataFrame, full: pd.DataFrame,
                           q: float, stat_col: str) -> pd.DataFrame:
    """One row per context: the q-quantile of AG_mean_logit over its exons.

    Mirrors the median table exactly -- 3 original full contexts (10,000 exons
    each) + 128 chimera contexts (64 per series on the 400-exon subset).
    """
    rows = []
    for ctx, nice in ORIGINALS:
        g = ep[ep.context == ctx]
        rows.append({"context_name": nice, "group": "original",
                     stat_col: g[A.Y].quantile(q), "n": len(g)})
    for (tgt, lab), g in full.groupby(["comparison", "context_label"]):
        rows.append({"context_name": f"{A.SERIES[tgt]}_{lab}",
                     "group": A.SERIES[tgt],
                     stat_col: g[A.Y].quantile(q), "n": len(g)})
    return pd.DataFrame(rows).sort_values(stat_col).reset_index(drop=True)


def plot_distribution(tab: pd.DataFrame, stat_col: str, pct_label: str,
                      png_path: Path) -> None:
    """The median figure's layout and style, driven by ``stat_col``."""
    style = {"e6": ("#f0c000", 46, "e6 chimeras"),
             "e7": ("#2f6fd0", 46, "e7 chimeras"),
             "original": ("black", 120, "10k endpoint reference runs")}
    chim = tab[tab.group != "original"]
    endp = tab[tab.group == "original"]

    fig, (ax_h, ax_s) = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        constrained_layout=True)

    ax_h.hist(chim[stat_col], bins=26, color="0.84",
              edgecolor="white", linewidth=.8, zorder=1)
    for _, r in endp.iterrows():
        ax_h.axvline(r[stat_col], color="black", lw=1.4, ls="--", zorder=3)
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
        ax_s.scatter(g[stat_col], yy, s=size, color=col,
                     edgecolor="black", linewidth=.55 if grp != "original" else .9,
                     zorder=4 if grp == "original" else 3,
                     label=f"{lab} (n={len(g)})")
    ax_s.set_ylim(0, 1)
    ax_s.set_yticks([])
    ax_s.set_ylabel("contexts", fontsize=9)
    ax_s.set_xlabel(f"{pct_label} AlphaGenome mean-logit")
    ax_s.legend(frameon=False, fontsize=8.5, loc="upper center",
                bbox_to_anchor=(0.5, -0.32), ncol=3)
    ax_s.grid(axis="x", alpha=.25)
    ax_s.set_axisbelow(True)

    fig.suptitle(f"Distribution of context {pct_label} AlphaGenome mean-logit",
                 fontsize=14, fontweight="bold")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="per-context quantile of AG_mean_logit (default 0.90)")
    args = ap.parse_args(argv)
    q = args.quantile
    if not 0.0 < q < 1.0:
        raise SystemExit("--quantile must be strictly between 0 and 1")

    pct = q * 100
    pct_txt = f"{pct:g}"
    stat_col = f"p{pct_txt.replace('.', '_')}_AG_mean_logit"
    pct_label = f"{pct_txt}th-percentile"
    slug = f"context_p{pct_txt.replace('.', '_')}_mean_logit"

    A.P_SUM.mkdir(parents=True, exist_ok=True)
    ep, full = A.load_all()
    tab = context_quantile_table(ep, full, q, stat_col)

    out_csv = A.P_SUM / f"{slug}.csv"
    out_png = A.P_SUM / f"{slug}_distribution.png"
    tab[["context_name", stat_col, "group", "n"]].to_csv(out_csv, index=False)
    plot_distribution(tab, stat_col, pct_label, out_png)

    # ---- report ----------------------------------------------------------
    print(f"statistic           : {pct_label} of {A.Y}")
    print(f"contexts            : {len(tab)}  "
          f"({(tab.group == 'e6').sum()} e6 + {(tab.group == 'e7').sum()} e7 "
          f"+ {(tab.group == 'original').sum()} originals)")
    print(f"\nendpoint reference values ({pct_label}):")
    for _, r in tab[tab.group == "original"].iterrows():
        print(f"   {r.context_name:24s} {r[stat_col]:9.4f}   (n={int(r.n):,})")
    print(f"\nchimera spread      : min {tab[tab.group != 'original'][stat_col].min():.4f}"
          f"  median {tab[tab.group != 'original'][stat_col].median():.4f}"
          f"  max {tab[tab.group != 'original'][stat_col].max():.4f}")
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
