"""
Runner for the saturation detector: classifies every (exon, motif family) in
the completed OpenSplice/FAS context-saturation dataset, draws diagnostic
figures for the development cases, and reports threshold sensitivity.

Read-only with respect to the experiment: no AlphaGenome calls, no variant
regeneration, no change to any scoring definition.

Outputs
-------
* ``outputs/open_splice_context_saturation/saturation_detector_results.csv``
* ``outputs/plots/open_splice_context_saturation/detector/<exon>_<family>.png``

Usage
-----
    python analysis_script/run_saturation_detector.py            # dev cases + all 32
    python analysis_script/run_saturation_detector.py --calibrate
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
from matplotlib.colors import LogNorm  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import saturation_detector as sd  # noqa: E402

CSV = ROOT / "outputs" / "open_splice_context_saturation" / "open_splice_context_saturation_variants.csv"
OUT_CSV = ROOT / "outputs" / "open_splice_context_saturation" / "saturation_detector_results.csv"
PLOT_DIR = ROOT / "outputs" / "plots" / "open_splice_context_saturation" / "detector"

#: (exon, family) -> expected label, from visual inspection of the exploratory plots
DEV_CASES = [
    ("LDLR_e7", "UAG", "BOTH_SIDES"),
    ("TP53_e7", "UAG", "BOTH_SIDES"),
    ("DMXL2_e36", "GGA", "GLOBAL_SATURATION"),
    ("TP53_e7", "GGA", "GLOBAL_SATURATION"),
    ("LDLR_e7", "GGAGGAC", "LEFT_SATURATION"),
    ("TP53_e7", "GGAGGAC", "LEFT_SATURATION"),
    ("TP53_e6", "CNNC", "AMBIGUOUS_OR_MULTIMODAL"),
    ("MEGF8_e41", "GGA", "AMBIGUOUS_OR_MULTIMODAL"),
    ("COL6A1_e9", "GGAGGAC", "INSUFFICIENT_RANGE"),
    ("TP53_e6", "UAG", "LEFT_SATURATION"),
]


def _load() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    return d[d.motif_family != "WT"]


def _plot(df: pd.DataFrame, res: sd.Result, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    x, y = df[sd.X_COL].to_numpy(), df[sd.Y_COL].to_numpy()
    h = ax.hist2d(x, y, bins=40, cmap="Blues", cmin=1, norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label="Count")

    c = res.curve
    if c is not None:
        ax.plot(c.bx, c.by, "-", color="gold", lw=1.2, alpha=.65, label="binned median")
        ax.plot(c.bx, c.sy, "-o", color="darkorange", lw=2.0, ms=4,
                alpha=.9, label="smoothed curve")

    # censored region
    if np.any(np.abs(y - sd.CENSOR_VALUE) < sd.CENSOR_TOL):
        ax.axhline(sd.CENSOR_VALUE, color="black", ls=":", lw=1.4,
                   label=f"clip ({sd.CENSOR_VALUE:.2f})")

    for side, p, col in (("left", res.left, "#2f6fd0"), ("right", res.right, "#d62828")):
        if not p.present:
            continue
        ax.axvspan(p.x_min, p.x_max, color=col, alpha=.12)
        ax.hlines(p.level, p.x_min, p.x_max, color=col, lw=2.6, zorder=7,
                  label=f"{side} plateau = {p.level:.2f}"
                        + (" (CENSORED)" if p.censored else ""))
        if np.isfinite(p.ci_low):
            ax.fill_between([p.x_min, p.x_max], p.ci_low, p.ci_high,
                            color=col, alpha=.30, zorder=6)
        if np.isfinite(p.onset_x):
            ax.axvline(p.onset_x, color=col, ls="--", lw=1.3, alpha=.8)

    ax.set_xlabel("PNAS pre-tuner score\n(energy_seq_struct)")
    ax.set_ylabel("AlphaGenome prod_logit\nlogit(P(acceptor) × P(donor))")
    ax.set_title(f"{res.exon_id} — {res.motif_family}", fontsize=12, pad=34)
    ax.text(0.0, 1.015,
            f"{res.classification}"
            + (f"  [{res.subtype}]" if res.subtype else "")
            + f"    conf={res.confidence}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            fontweight="bold")
    ax.text(1.0, 1.015,
            f"range={res.global_response_range:.2f}  branch={res.branch_score:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
    ax.legend(fontsize=8, loc="best", framealpha=.9)
    ax.set_box_aspect(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_all(d: pd.DataFrame, t: sd.Thresholds, plot_only: set | None = None):
    rows, results = [], {}
    for (e, f), g in d.groupby(["exon_id", "motif_family"]):
        r = sd.detect(g, e, f, t)
        results[(e, f)] = r
        rows.append(sd.to_row(r))
        if plot_only is None or (e, f) in plot_only:
            _plot(g, r, PLOT_DIR / f"{e}_{f}.png")
    return pd.DataFrame(rows), results


def calibrate(d: pd.DataFrame) -> None:
    """Sensitivity of every development-case label to each threshold."""
    dev = {(e, f): exp for e, f, exp in DEV_CASES}
    grids = {
        "flat_range":        [0.6, 0.8, 1.0, 1.2, 1.5, 1.9],
        "branch_frac":       [0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
        "slope_frac":        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "min_plateau_xfrac": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "min_plateau_bins":  [2, 3, 4],
        "target_bin_n":      [20, 25, 30, 40, 50],
    }
    base = sd.Thresholds()
    for name, grid in grids.items():
        print(f"\n--- {name} (default {getattr(base, name)}) ---")
        for v in grid:
            t = sd.Thresholds(**{**vars(base), name: v})
            _, res = run_all(d, t, plot_only=set())
            agree = sum(1 for k, exp in dev.items() if res[k].classification == exp)
            changed = [f"{k[0]}/{k[1]}→{res[k].classification}"
                       for k, exp in dev.items() if res[k].classification != exp]
            mark = " <-- default" if v == getattr(base, name) else ""
            print(f"  {name}={v:<5} dev agreement {agree}/10{mark}")
            if changed:
                print(f"      mismatches: {', '.join(changed)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the saturation detector.")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--dev-only", action="store_true")
    args = ap.parse_args(argv)

    d = _load()
    if args.calibrate:
        calibrate(d)
        return 0

    t = sd.Thresholds()
    dev_keys = {(e, f) for e, f, _ in DEV_CASES}
    df, results = run_all(d, t, plot_only=dev_keys if args.dev_only else None)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("=== DEVELOPMENT CASES ===")
    print(f"{'exon':11} {'fam':8} {'expected':24} {'got':24} {'ok':>3}  detail")
    ok = 0
    for e, f, exp in DEV_CASES:
        r = results[(e, f)]
        match = r.classification == exp
        ok += match
        detail = []
        if r.left.present:
            detail.append(f"L={r.left.level:.2f}" + ("*" if r.left.censored else ""))
        if r.right.present:
            detail.append(f"R={r.right.level:.2f}" + ("*" if r.right.censored else ""))
        print(f"{e:11} {f:8} {exp:24} {r.classification:24} {'Y' if match else 'N':>3}  "
              f"{' '.join(detail)}")
    print(f"\nagreement: {ok}/10")
    print(f"\nwrote {len(df)} rows -> {OUT_CSV}")
    print(f"figures -> {PLOT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
