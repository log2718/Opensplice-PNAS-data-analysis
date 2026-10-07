"""
Exploratory analysis of the completed 8-exon OpenSplice/FAS context-saturation
experiment. Read-only: no AlphaGenome calls, no variant regeneration, no
changes to any scoring definition, and no saturation detector.

Produces
--------
* ``outputs/plots/open_splice_context_saturation/exploratory/<exon>_<family>.png``
  -- one figure per exon x motif family (up to 32), all feasible copy levels
  pooled into a single blue hist2d density (Blues / LogNorm / "Count"
  colorbar, matching ``plot_hybrid_pnas_vs_alphagenome``), with the
  copy-number median trajectory (2x -> 14x) and the WT reference overlaid.
  No OLS line: the point is the response *shape*, not a single slope.
* ``context_saturation_summary_by_exon_motif.csv``  -- section 3
* ``context_saturation_high_enrichment_endpoints.csv`` -- section 4
* ``context_saturation_convergence.csv`` -- section 5

``prod_logit`` is logit(P_acceptor x P_donor) from AlphaGenome SPLICE_SITES.
It is a splice-site recognisability quantity, not PSI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.stats import linregress, pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "open_splice_context_saturation" / "open_splice_context_saturation_variants.csv"
OUT_DIR = ROOT / "outputs" / "open_splice_context_saturation"
PLOT_DIR = ROOT / "outputs" / "plots" / "open_splice_context_saturation" / "exploratory"

X, Y = "PNAS_pretuner", "AG_prod_logit"
X_LABEL = "PNAS pre-tuner score\n(energy_seq_struct)"
Y_LABEL = "AlphaGenome prod_logit\nlogit(P(acceptor) × P(donor))"
BINS = 40
LEVELS = [2, 4, 6, 8, 10, 12, 14]
FAMILIES = ["UAG", "CNNC", "GGA", "GGAGGAC"]


#: candidate label offsets in points, tried in order until one is free
_OFFSETS = [(7, 7), (7, -11), (-20, 7), (-20, -11), (7, 16), (7, -20),
            (-22, 16), (-22, -20), (18, 0), (-30, 0)]


def _label_levels(ax, med: pd.DataFrame, min_sep: float = 0.052) -> None:
    """Annotate each Nx median, nudging labels apart when points collide.

    On flat curves consecutive medians can land almost on the same pixel, which
    hides a label under its neighbour and makes a level look like it is missing.
    Labels are placed greedily: the first free candidate offset (in axes
    fraction) that is at least ``min_sep`` from every already-placed label wins,
    and a leader line is drawn whenever the label had to move far.
    """
    inv = ax.transAxes.inverted()
    placed: list[tuple[float, float]] = []
    for lvl, r in med.iterrows():
        px, py = ax.transAxes.inverted().transform(
            ax.transData.transform((r.mx, r.my)))
        chosen = _OFFSETS[-1]
        for dx, dy in _OFFSETS:
            # convert the point offset into axes fraction for the distance test
            disp = ax.transData.transform((r.mx, r.my)) + np.array(
                [dx * ax.figure.dpi / 72.0, dy * ax.figure.dpi / 72.0])
            ax_xy = inv.transform(disp)
            if all(np.hypot(ax_xy[0] - qx, ax_xy[1] - qy) >= min_sep
                   for qx, qy in placed):
                chosen = (dx, dy)
                placed.append((ax_xy[0], ax_xy[1]))
                break
        else:
            disp = ax.transData.transform((r.mx, r.my)) + np.array(
                [chosen[0] * ax.figure.dpi / 72.0,
                 chosen[1] * ax.figure.dpi / 72.0])
            placed.append(tuple(inv.transform(disp)))

        far = abs(chosen[0]) > 12 or abs(chosen[1]) > 12
        ax.annotate(
            f"{lvl}x", (r.mx, r.my), textcoords="offset points",
            xytext=chosen, fontsize=8, color="darkgoldenrod",
            fontweight="bold", zorder=9,
            arrowprops=dict(arrowstyle="-", color="darkgoldenrod",
                            lw=0.6, alpha=0.8) if far else None,
        )


def _plot_cell(sub: pd.DataFrame, wt_row: pd.Series, exon: str, family: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)

    x, y = sub[X].to_numpy(), sub[Y].to_numpy()
    h = ax.hist2d(x, y, bins=BINS, cmap="Blues", cmin=1, norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label="Count")

    # copy-number median trajectory, 2x -> 14x
    med = (sub.groupby("copy_number")
              .agg(mx=(X, "median"), my=(Y, "median"))
              .reindex([l for l in LEVELS if l in set(sub.copy_number)]))
    ax.plot(med.mx, med.my, "-o", color="gold", lw=2.2, ms=7, alpha=0.75,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6,
            label="median per Nx (2x→14x)")

    # global OLS fit -- descriptive only, NOT the saturation criterion
    lr = linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, lr.intercept + lr.slope * xl, color="red", lw=1.8,
            zorder=5, label="OLS fit")

    if wt_row is not None:
        ax.scatter([wt_row[X]], [wt_row[Y]], marker="*", s=320, color="black",
                   edgecolor="white", linewidth=1.2, zorder=8, label="WT")

    # Keep the view on the data: the OLS line is allowed to run off-panel
    # rather than stretching the axes and flattening the density.
    ypad = 0.04 * (y.max() - y.min() or 1.0)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"{exon} — {family}", fontsize=12, pad=46)
    # stats block: top-left, OUTSIDE the axes
    ax.text(0.0, 1.015,
            f"n = {len(x):,}    Slope = {lr.slope:.3f}    "
            f"Pearson r = {pearsonr(x, y).statistic:.3f}    "
            f"Spearman ρ = {spearmanr(x, y).statistic:.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    # own line above the stats so it can never collide with them on a narrow panel
    ax.text(1.0, 1.085, f"exon length = {int(sub.exon_length.iloc[0])} nt",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.legend(fontsize=8.5, loc="best", framealpha=0.9)
    ax.set_box_aspect(1)

    # Labels last: the de-collision logic works in display space, so it needs
    # the final axis limits and aspect to already be set.
    fig.canvas.draw()
    _label_levels(ax, med)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{exon}_{family}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Exploratory analysis of context-saturation results.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    d = pd.read_csv(CSV)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[X, Y])
    wt = d[d.motif_family == "WT"].set_index("exon_id")
    mut = d[d.motif_family != "WT"]

    summary, endpoints, convergence = [], [], []
    n_plots = 0

    for exon in sorted(mut.exon_id.unique()):
        for family in FAMILIES:
            sub = mut[(mut.exon_id == exon) & (mut.motif_family == family)]
            if sub.empty:
                continue
            levels = sorted(sub.copy_number.unique())
            wt_row = wt.loc[exon] if exon in wt.index else None

            if not args.no_plots:
                _plot_cell(sub, wt_row, exon, family)
                n_plots += 1

            x, y = sub[X].to_numpy(), sub[Y].to_numpy()
            lr = linregress(x, y)
            row = {
                "exon_id": exon, "motif_family": family,
                "exon_length": int(sub.exon_length.iloc[0]),
                "n_variants": len(sub),
                "feasible_copy_levels": ",".join(f"{l}x" for l in levels),
                "WT_PNAS": float(wt_row[X]) if wt_row is not None else np.nan,
                "WT_AG": float(wt_row[Y]) if wt_row is not None else np.nan,
                "PNAS_min": x.min(), "PNAS_max": x.max(), "PNAS_range": x.max() - x.min(),
                "AG_min": y.min(), "AG_max": y.max(), "AG_range": y.max() - y.min(),
                "global_OLS_slope": lr.slope,
                "Pearson_r": pearsonr(x, y).statistic,
                "Spearman_rho": spearmanr(x, y).statistic,
            }
            med_by_level = {}
            for lvl in LEVELS:
                g = sub[sub.copy_number == lvl]
                if g.empty:
                    for s in ("median_PNAS", "median_AG", "AG_IQR", "AG_std", "AG_min", "AG_max"):
                        row[f"{lvl}x_{s}"] = np.nan
                    continue
                q1, q3 = g[Y].quantile(.25), g[Y].quantile(.75)
                med_by_level[lvl] = g[Y].median()
                row[f"{lvl}x_median_PNAS"] = g[X].median()
                row[f"{lvl}x_median_AG"] = g[Y].median()
                row[f"{lvl}x_AG_IQR"] = q3 - q1
                row[f"{lvl}x_AG_std"] = g[Y].std()
                row[f"{lvl}x_AG_min"] = g[Y].min()
                row[f"{lvl}x_AG_max"] = g[Y].max()
            med_range = (max(med_by_level.values()) - min(med_by_level.values())
                         if med_by_level else np.nan)
            row["median_response_range"] = med_range
            summary.append(row)

            # --- section 4: provisional high-enrichment endpoint -------------
            hi_levels = levels[-3:] if len(levels) >= 3 else levels
            hi = sub[sub.copy_number.isin(hi_levels)]
            endpoints.append({
                "exon_id": exon, "motif_family": family,
                "high_enrichment_levels_used": ",".join(f"{l}x" for l in hi_levels),
                "high_enrichment_n": len(hi),
                "high_enrichment_AG_median": hi[Y].median(),
                "high_enrichment_AG_IQR": hi[Y].quantile(.75) - hi[Y].quantile(.25),
                "high_enrichment_AG_std": hi[Y].std(),
                "high_enrichment_AG_range": hi[Y].max() - hi[Y].min(),
                "high_enrichment_PNAS_median": hi[X].median(),
            })

            # --- section 5: convergence at the highest feasible level --------
            top = sub[sub.copy_number == levels[-1]]
            iqr = top[Y].quantile(.75) - top[Y].quantile(.25)
            convergence.append({
                "exon_id": exon, "motif_family": family,
                "highest_level": f"{levels[-1]}x", "n": len(top),
                "top_AG_median": top[Y].median(), "top_AG_IQR": iqr,
                "top_AG_std": top[Y].std(),
                "top_AG_range": top[Y].max() - top[Y].min(),
                "median_response_range": med_range,
                "IQR_over_response_range": iqr / med_range if med_range and med_range > 0 else np.nan,
            })

    pd.DataFrame(summary).to_csv(OUT_DIR / "context_saturation_summary_by_exon_motif.csv", index=False)
    pd.DataFrame(endpoints).to_csv(OUT_DIR / "context_saturation_high_enrichment_endpoints.csv", index=False)
    pd.DataFrame(convergence).to_csv(OUT_DIR / "context_saturation_convergence.csv", index=False)
    print(f"wrote 3 summary CSVs -> {OUT_DIR}")
    if not args.no_plots:
        print(f"wrote {n_plots} figures -> {PLOT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
