"""
Summary figures for the corrected 8-block chimera run.

Two figures only -- context median and context 90th percentile of
``AG_mean_logit`` -- in the same layout and style as the 6-block summary
figures in ``analyze_synthetic_mean_logit.part5_median_distribution``:

    top    grey histogram of the statistic over the 512 chimera contexts
    bottom horizontal jitter strip, e6 yellow / e7 blue / endpoints black,
           with dashed black verticals at the three endpoint references

The three 10,000-exon endpoint runs are reference OVERLAYS only: they are drawn
as dashed lines and black markers but are never part of the histogram.

Makes NO AlphaGenome calls. Reads the merged 8-block results and the 6-block
endpoint CSV (read-only) and writes only under
``outputs/synthetic_context_8block/plots/summary/``.

Usage::

    python analysis_script/plot_chimera8_summary.py
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

import splice_site_partition as SP  # noqa: E402

NEW = ROOT / "outputs" / "synthetic_context_8block"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
OLD_ENDPOINT_CSV = ROOT / "outputs" / "synthetic_context" / "synthetic_endpoint_ag_results.csv"
P_SUM = NEW / "plots" / "summary"

Y = "AG_mean_logit"
SERIES = {"TP53_e6": "e6", "TP53_e7": "e7"}
ORIGINALS = (("PNAS", "original_PNAS_1309"),
             ("TP53_e6", "original_FAS_TP53_e6"),
             ("TP53_e7", "original_FAS_TP53_e7"))

#: inherited verbatim from the 6-block summary figure
STYLE = {"e6": ("#f0c000", 46, "e6 chimeras"),
         "e7": ("#2f6fd0", 46, "e7 chimeras"),
         "original": ("black", 120, "10k endpoint reference runs")}
FIGSIZE = (11, 6.4)
HIST_BINS = 26


def validate(d: pd.DataFrame) -> None:
    """Fail loudly rather than plot a malformed design."""
    problems = []
    if len(d) != 204_800:
        problems.append(f"{len(d):,} rows, expected 204,800")
    per_arm = d.groupby("comparison").context_label_8.nunique().to_dict()
    for arm in SERIES:
        if per_arm.get(arm) != 256:
            problems.append(f"{arm}: {per_arm.get(arm)} contexts, expected 256")
    sizes = d.groupby(["comparison", "context_label_8"]).sequence_id.nunique()
    if not (sizes == 400).all():
        problems.append(f"sequences per context range "
                        f"{sizes.min()}..{sizes.max()}, expected 400")
    if not (d.context_label_8.str.len() == 8).all():
        problems.append("context_label_8 is not 8 characters everywhere")
    bad = [b for i, b in enumerate(SP.BLOCK_ORDER_8)
           if not (d.context_label_8.str[i] == d[b]).all()]
    if bad:
        problems.append(f"label positions do not match block columns at {bad}")
    if Y not in d.columns:
        problems.append(f"missing response column {Y}")
    if d[Y].isna().any():
        problems.append(f"{int(d[Y].isna().sum())} null {Y}")
    if problems:
        raise SystemExit("[abort] merged dataset failed validation:\n  - "
                         + "\n  - ".join(problems))


def stat_table(d: pd.DataFrame, ep: pd.DataFrame, q: float | None,
               stat_col: str) -> pd.DataFrame:
    """One row per context; endpoints appended as reference overlays."""
    def agg(s: pd.Series) -> float:
        return s.median() if q is None else s.quantile(q)

    rows = []
    for (arm, lab), g in d.groupby(["comparison", "context_label_8"]):
        rows.append({"context_name": f"{SERIES[arm]}_{lab}",
                     "context_label_8": lab, "group": SERIES[arm],
                     "n": int(g.sequence_id.nunique()),
                     "n_F_blocks": int(lab.count("F")),
                     stat_col: agg(g[Y])})
    for ctx, nice in ORIGINALS:
        g = ep[ep.context == ctx]
        rows.append({"context_name": nice, "context_label_8": pd.NA,
                     "group": "original", "n": int(len(g)),
                     "n_F_blocks": pd.NA, stat_col: agg(g[Y])})
    return pd.DataFrame(rows).sort_values(stat_col).reset_index(drop=True)


def plot(tab: pd.DataFrame, stat_col: str, stat_label: str,
         png_path: Path) -> None:
    """The 6-block summary layout, driven by ``stat_col``."""
    chim = tab[tab.group != "original"]
    endp = tab[tab.group == "original"]

    fig, (ax_h, ax_s) = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        constrained_layout=True)

    # histogram: the 512 chimera contexts ONLY
    ax_h.hist(chim[stat_col], bins=HIST_BINS, color="0.84",
              edgecolor="white", linewidth=.8, zorder=1)
    for _, r in endp.iterrows():
        ax_h.axvline(r[stat_col], color="black", lw=1.4, ls="--", zorder=3)
    ax_h.plot([], [], color="black", lw=1.4, ls="--",
              label=f"{STYLE['original'][2]} (n={len(endp)})")
    ax_h.set_ylabel("number of contexts")
    ax_h.set_title(f"8-block chimera contexts, n = {len(chim)}",
                   fontsize=9, color="0.25")
    ax_h.legend(frameon=False, fontsize=9, loc="upper left")
    ax_h.grid(axis="y", alpha=.25)
    ax_h.set_axisbelow(True)

    rng = np.random.default_rng(0)
    for grp in ("e6", "e7", "original"):
        g = tab[tab.group == grp]
        col, size, lab = STYLE[grp]
        yy = rng.uniform(0.15, 0.85, len(g))
        ax_s.scatter(g[stat_col], yy, s=size, color=col,
                     edgecolor="black", linewidth=.55 if grp != "original" else .9,
                     zorder=4 if grp == "original" else 3,
                     label=f"{lab} (n={len(g)})")
    ax_s.set_ylim(0, 1)
    ax_s.set_yticks([])
    ax_s.set_ylabel("contexts", fontsize=9)
    ax_s.set_xlabel(f"{stat_label} AlphaGenome mean-logit")
    ax_s.legend(frameon=False, fontsize=8.5, loc="upper center",
                bbox_to_anchor=(0.5, -0.32), ncol=3)
    ax_s.grid(axis="x", alpha=.25)
    ax_s.set_axisbelow(True)

    # x limits from the new data (chimeras + overlays), not the old figure's
    lo, hi = tab[stat_col].min(), tab[stat_col].max()
    pad = (hi - lo) * 0.04
    ax_s.set_xlim(lo - pad, hi + pad)

    fig.suptitle(f"Distribution of context {stat_label} AlphaGenome mean-logit",
                 fontsize=14, fontweight="bold")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    d = pd.read_csv(RESULTS_CSV, low_memory=False)
    ep = pd.read_csv(OLD_ENDPOINT_CSV)
    validate(d)
    P_SUM.mkdir(parents=True, exist_ok=True)

    print(f"rows                : {len(d):,}")
    print(f"contexts per arm    : "
          f"{d.groupby('comparison').context_label_8.nunique().to_dict()}")
    print(f"sequences per context: "
          f"{d.groupby(['comparison', 'context_label_8']).sequence_id.nunique().unique()}")
    print(f"response            : {Y}  (prod_logit not used)")

    specs = [(None, "median_AG_mean_logit", "median",
              "context_median_AG_mean_logit"),
             (0.90, "p90_AG_mean_logit", "90th-percentile",
              "context_p90_AG_mean_logit")]

    for q, stat_col, stat_label, slug in specs:
        tab = stat_table(d, ep, q, stat_col)
        csv_path = P_SUM / f"{slug}.csv"
        png_path = P_SUM / f"{slug}_distribution.png"
        tab[["context_name", "context_label_8", "group", "n", "n_F_blocks",
             stat_col]].to_csv(csv_path, index=False)
        plot(tab, stat_col, stat_label, png_path)

        chim = tab[tab.group != "original"]
        print(f"\n--- {stat_label} ---")
        for grp in ("e6", "e7"):
            g = chim[chim.group == grp][stat_col]
            print(f"  {grp}: n={len(g):3d}  min {g.min():8.4f}  "
                  f"median {g.median():8.4f}  max {g.max():8.4f}")
        print("  endpoint references (10,000 exons each):")
        for _, r in tab[tab.group == "original"].iterrows():
            print(f"    {r.context_name:24s} {r[stat_col]:9.4f}")
        print(f"  wrote {csv_path.relative_to(ROOT)}")
        print(f"  wrote {png_path.relative_to(ROOT)}")

    print("\nNo AlphaGenome calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
