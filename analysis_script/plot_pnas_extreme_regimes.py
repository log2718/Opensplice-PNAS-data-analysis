"""
Standalone figure: AlphaGenome ``prod_logit`` for experimentally skipped
(PSI < 0.05) vs experimentally included (PSI > 0.95) PNAS cassette exons.

Train and test are pooled -- this figure is about the separation between the
two experimental extremes, not about split-level validation (which the
per-split tables already cover).

``prod_logit`` is NOT PSI. It is ``logit(P(acceptor) x P(donor))`` with both
probabilities from AlphaGenome ``SPLICE_SITES``.

Input (read-only, no API calls)
--------------------------------
``outputs/alphagenome_pnas_sanity/pnas_fulltrack_splice_site_sanity.csv``

Output
------
``outputs/plots/alphagenome_pnas_sanity/extreme_regimes_prod_logit.png``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "alphagenome_pnas_sanity" / "pnas_fulltrack_splice_site_sanity.csv"
OUT_PNG = ROOT / "outputs" / "plots" / "alphagenome_pnas_sanity" / "extreme_regimes_prod_logit.png"

LOW_COLOR = "#2f6fd0"   # blue  -- experimentally skipped
HIGH_COLOR = "#d62828"  # red   -- experimentally included


def main() -> int:
    d = pd.read_csv(CSV)

    low = d[d["psi_bin"] == "<0.05"]["prod_logit"]
    high = d[d["psi_bin"] == ">0.95"]["prod_logit"]
    print(f"pooled train+test:  PSI<0.05 n={len(low)}   PSI>0.95 n={len(high)}")

    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    bins = np.linspace(
        min(low.min(), high.min()), max(low.max(), high.max()), 60
    )
    for v, color, label in (
        (low, LOW_COLOR, f"Experimentally skipped — PSI < 0.05  (n={len(low):,})"),
        (high, HIGH_COLOR, f"Experimentally included — PSI > 0.95  (n={len(high):,})"),
    ):
        ax.hist(v, bins=bins, density=True, color=color, alpha=0.40)
        ax.hist(v, bins=bins, density=True, histtype="step", color=color, linewidth=1.9,
                label=label)
        ax.axvline(v.median(), color=color, linestyle="--", linewidth=1.4)

    ax.annotate(f"median {low.median():.2f}", xy=(low.median(), ax.get_ylim()[1] * 0.93),
                color=LOW_COLOR, ha="right", va="top", fontsize=9, fontweight="bold",
                xytext=(-4, 0), textcoords="offset points")
    ax.annotate(f"median {high.median():.2f}", xy=(high.median(), ax.get_ylim()[1] * 0.80),
                color=HIGH_COLOR, ha="right", va="top", fontsize=9, fontweight="bold",
                xytext=(-4, 0), textcoords="offset points")

    ax.set_xlabel("AlphaGenome middle-exon prod_logit\nlogit(P(acceptor) × P(donor))")
    ax.set_ylabel("Density")
    ax.set_title(
        "PNAS three-exon cassette — AlphaGenome SPLICE_SITES at the experimental extremes\n"
        "train + test pooled",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")

    print(f"  PSI<0.05 : median={low.median():+.4f}  IQR=[{low.quantile(.25):+.4f}, {low.quantile(.75):+.4f}]")
    print(f"  PSI>0.95 : median={high.median():+.4f}  IQR=[{high.quantile(.25):+.4f}, {high.quantile(.75):+.4f}]")
    print(f"  median separation: {high.median() - low.median():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
