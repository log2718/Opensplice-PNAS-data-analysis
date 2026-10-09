"""
Per-context endpoint figures split by Dirichlet regime.

Three PNGs -- one per full context -- each a 2x2 grid of blue 2D histograms,
one subplot per Dirichlet alpha (0.1, 0.5, 1.0, 5.0):

    endpoint_dirichlet_panels_PNAS_original_1309.png
    endpoint_dirichlet_panels_FAS_TP53_e6.png
    endpoint_dirichlet_panels_FAS_TP53_e7.png

READ-ONLY: no AlphaGenome calls, no CSV is written or modified. The existing
3-panel figure in ``plots/original_context/`` is left untouched; these go to
``plots/endpoints/``.

Axes are identical across all 12 subplots AND across all three PNGs, reusing
the same pooled endpoint+chimera limits as the other mean-logit figures, so
every panel in the whole set is directly comparable.

Style (Blues / LogNorm / cmin=1 / "Count" colorbar / 40 bins) is inherited from
``analyze_synthetic_mean_logit._hist_panel`` rather than re-specified here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import analyze_synthetic_mean_logit as A  # noqa: E402

LIB_CSV = A.OUT / "synthetic_exon_library.csv"
OUT_DIR = A.OUT / "plots" / "endpoints"

#: (context value in the results CSV, display name used in the filename/title)
CONTEXTS = [("PNAS", "PNAS_original_1309"),
            ("TP53_e6", "FAS_TP53_e6"),
            ("TP53_e7", "FAS_TP53_e7")]
#: 2x2 reading order: top-left, top-right, bottom-left, bottom-right
ALPHA_ORDER = [0.1, 0.5, 1.0, 5.0]


def _regime_column(lib: pd.DataFrame) -> str:
    """Detect the Dirichlet regime column rather than assuming its name."""
    for c in ("alpha", "dirichlet_alpha", "regime", "dirichlet_regime"):
        if c in lib.columns:
            return c
    cand = [c for c in lib.columns
            if any(k in c.lower() for k in ("alpha", "dirichlet", "regime"))]
    if not cand:
        raise SystemExit(
            f"No Dirichlet regime column found in {LIB_CSV.name}. "
            f"Columns are: {list(lib.columns)}")
    return cand[0]


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    lib = pd.read_csv(LIB_CSV)
    col = _regime_column(lib)

    # Reuse the pooled endpoint+chimera limits so these panels line up exactly
    # with the existing mean-logit figures.
    ep, full = A.load_all()
    xlim, ylim = A.global_limits(ep, full)

    # Merge on sequence_id to recover the regime, per the stated requirement.
    merged = ep.merge(lib[["sequence_id", col]], on="sequence_id",
                      how="left", suffixes=("", "_lib"))
    reg = f"{col}_lib" if f"{col}_lib" in merged.columns else col
    if merged[reg].isna().any():
        raise SystemExit(f"{merged[reg].isna().sum()} endpoint rows had no "
                         f"matching sequence_id in {LIB_CSV.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"regime column used: '{col}' (from {LIB_CSV.name}, merged on sequence_id)")
    print(f"shared axes: x={tuple(round(v, 2) for v in xlim)}  "
          f"y={tuple(round(v, 2) for v in ylim)}")
    print("\nrows per alpha group (per context):")

    written = []
    for ctx, nice in CONTEXTS:
        sub = merged[merged.context == ctx]
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 11.4),
                                 constrained_layout=True)
        counts = []
        for ax, a in zip(axes.ravel(), ALPHA_ORDER):
            s = sub[sub[reg] == a]
            counts.append(f"alpha={a}: {len(s):,}")
            A._hist_panel(ax, s, fig, xlim, ylim,
                          f"Dirichlet alpha = {a}\nn = {len(s):,}")
        fig.suptitle(f"{nice} — AlphaGenome mean-logit by Dirichlet regime",
                     fontsize=15, fontweight="bold")
        out = OUT_DIR / f"endpoint_dirichlet_panels_{nice}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"  {nice:20} " + "   ".join(counts))

    print("\nwrote:")
    for p in written:
        print(f"  {p}")
    print(f"\nexisting figure left untouched: "
          f"{A.P_END / 'original_context_three_panel_mean_logit.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
