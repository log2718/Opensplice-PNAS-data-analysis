"""
Per-context scatter panels for the 8-block chimera run: 256 contexts x 2 arms.

The equivalent figures for the 6-block run live in
``outputs/synthetic_context/plots/chimeras/{e6,e7}/`` and were produced by
``analyze_synthetic_mean_logit.part2_chimeras``. The 8-block run never got
them. This script fills that gap using the SAME panel helper
(``analyze_synthetic_mean_logit._hist_panel``) and the same global axis limits,
so an 8-block panel is visually comparable to a 6-block one.

Writes 512 PNGs into ``outputs/synthetic_context_8block/plots/contexts/{e6,e7}/``
plus an index CSV used by the manual labelling tool.

Usage::

    python analysis_script/make_chimera8_context_plots.py
    python analysis_script/make_chimera8_context_plots.py --limit 4   # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import analyze_synthetic_mean_logit as A  # noqa: E402
import splice_site_partition as SP  # noqa: E402

RESULTS_CSV = (ROOT / "outputs" / "synthetic_context_8block"
               / "chimera8_ag_results.csv")
ENDPOINT_CSV = (ROOT / "outputs" / "synthetic_context"
                / "synthetic_endpoint_ag_results.csv")
OUT_DIR = ROOT / "outputs" / "synthetic_context_8block" / "plots" / "contexts"
INDEX_CSV = OUT_DIR / "context_plot_index.csv"

ARMS = ["TP53_e6", "TP53_e7"]
SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
LABEL_COL = "context_label_8"


def block_summary(label: str) -> str:
    """'B1=F B4=F ...' for the blocks that are FAS in this context."""
    f = [b for b, c in zip(SP.BLOCK_ORDER_8, label) if c == "F"]
    return "all PNAS" if not f else " ".join(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="only render this many contexts per arm (smoke test)")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args(argv)

    if not RESULTS_CSV.exists():
        raise SystemExit(f"missing {RESULTS_CSV}")

    full = pd.read_csv(RESULTS_CSV,
                       usecols=["comparison", LABEL_COL, "sequence_id",
                                "n_F_blocks", A.X, A.Y],
                       low_memory=False)
    ep = pd.read_csv(ENDPOINT_CSV)

    # Same global limits as the 6-block figures, so panels are comparable.
    xlim, ylim = A.global_limits(ep, full)
    print(f"axis limits  x={xlim}  y={ylim}")

    rows = []
    for arm in ARMS:
        tag = SLUG[arm]
        d = OUT_DIR / tag
        d.mkdir(parents=True, exist_ok=True)
        labels = sorted(full.loc[full.comparison == arm, LABEL_COL].unique())
        if args.limit:
            labels = labels[:args.limit]
        n = 0
        for lab in labels:
            s = full[(full.comparison == arm) & (full[LABEL_COL] == lab)]
            if s.empty:
                print(f"  MISSING {tag} {lab}")
                continue
            fig, ax = plt.subplots(figsize=(6.8, 6.6), constrained_layout=True)
            subtitle = (f"P = PNAS block, F = FAS block   |   n = {len(s):,}"
                        f"\nFAS blocks: {block_summary(lab)}")
            A._hist_panel(ax, s, fig, xlim, ylim, subtitle)
            fig.suptitle(f"{arm} chimera context: {lab}",
                         fontsize=13, fontweight="bold")
            path = d / f"{tag}_{lab}.png"
            fig.canvas.draw()
            bb = ax.get_window_extent()
            fig_h = fig.get_size_inches()[1] * fig.dpi
            fig.savefig(path, dpi=args.dpi)
            plt.close(fig)
            scale = args.dpi / fig.dpi          # savefig dpi vs canvas dpi
            rows.append({
                "img_w": round(fig.get_size_inches()[0] * args.dpi),
                "img_h": round(fig.get_size_inches()[1] * args.dpi),
                "ax_left_px": round(bb.x0 * scale, 2),
                "ax_right_px": round(bb.x1 * scale, 2),
                "ax_top_px": round((fig_h - bb.y1) * scale, 2),
                "ax_bottom_px": round((fig_h - bb.y0) * scale, 2),
                "x_min": xlim[0], "x_max": xlim[1],
                "y_min": ylim[0], "y_max": ylim[1],
                "comparison": arm,
                "arm": tag,
                LABEL_COL: lab,
                "n_F_blocks": int(s.n_F_blocks.iloc[0]),
                "n_points": len(s),
                "median_mean_logit": float(s[A.Y].median()),
                "image": str(path.relative_to(ROOT)),
            })
            n += 1
            if n % 50 == 0:
                print(f"  {tag}: {n}/{len(labels)}")
        print(f"[{tag}] wrote {n} figures -> {d.relative_to(ROOT)}")

    # Review order: within each arm, fewest FAS blocks first, so the
    # sweep runs all-PNAS -> one block swapped -> ... -> all-FAS.
    idx = (pd.DataFrame(rows)
           .sort_values(["comparison", "n_F_blocks", LABEL_COL])
           .reset_index(drop=True))
    idx.to_csv(INDEX_CSV, index=False)
    print(f"\nindex ({len(idx)} rows) -> {INDEX_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
