"""
Analyze the B1 cryptic-splice-site check.

Joins ``cryptic_measurements.csv`` to the pilot's ``delta_vs_native`` and asks
the question the pilot left open: is the measured B1 effect explained by novel
splice sites appearing inside the randomized block?

Writes tables and one scatter plot into ``outputs/upstream_exon/cryptic_check/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import cryptic_lib as C  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
CHECK = OUT / "cryptic_check"
PRIMARY = "syn_a5.0_00482"


def main() -> int:
    m = pd.read_csv(CHECK / "cryptic_measurements.csv")
    annot = pd.read_csv(OUT / "b1_pilot_results_annotated.csv")
    a = annot[annot.replicate == 1][["row_id", "delta_vs_native",
                                     "native_mean_logit"]]
    d = m.merge(a, on="row_id", how="left", validate="one_to_one")
    assert d.delta_vs_native.notna().all(), "unmatched row_id in the join"

    # the full-track re-score must reproduce the pilot's own numbers ----------
    p = annot[annot.replicate == 1][["row_id", "AG_mean_logit"]].rename(
        columns={"AG_mean_logit": "pilot_mean_logit"})
    d = d.merge(p, on="row_id", validate="one_to_one")
    d["rescore_diff"] = (d.AG_mean_logit - d.pilot_mean_logit).abs()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say("=" * 74)
    say("B1 CRYPTIC SPLICE SITE CHECK")
    say("=" * 74)
    say(f"constructs re-scored with full tracks: {len(d)}")
    say(f"  {PRIMARY} (primary): {(d.sequence_id == PRIMARY).sum()}")
    for eid in sorted(set(d.sequence_id) - {PRIMARY}):
        say(f"  {eid}: {(d.sequence_id == eid).sum()}")
    say(f"exclusion half-width around annotated sites: "
        f"{int(d.exclude_halfwidth.iloc[0])} nt")
    say()
    say(f"re-score vs pilot AG_mean_logit: max |diff| = "
        f"{d.rescore_diff.max():.2e}  (exact for "
        f"{(d.rescore_diff == 0).sum()}/{len(d)})")
    say()

    # --- 1. the four annotated sites -----------------------------------------
    say("-" * 74)
    say("1. ANNOTATED SPLICE SITES (sanity: all four should stay strong)")
    say("-" * 74)
    for s in ("upstream_donor", "middle_acceptor", "middle_donor",
              "downstream_acceptor"):
        col = f"site_{s}_prob"
        say(f"  {s:22s} pos {int(d[f'site_{s}_pos'].iloc[0]):5d}   "
            f"prob min {d[col].min():.4f}  median {d[col].median():.4f}  "
            f"max {d[col].max():.4f}")
    say()

    # --- 2. drift -------------------------------------------------------------
    say("-" * 74)
    say("2. MIDDLE-EXON PEAK DRIFT (argmax within +/-5 nt of the read position)")
    say("-" * 74)
    for k in ("acceptor", "donor"):
        dr = d[f"middle_{k}_drift"]
        say(f"  {k}: nonzero drift in {int((dr != 0).sum())}/{len(d)} "
            f"constructs; drift values {sorted(set(dr.tolist()))}")
        gap = d[f"middle_{k}_local_max"] - d[f"site_middle_{k}_prob"]
        say(f"            local max exceeds the read position by at most "
            f"{gap.max():.2e}")
    say(f"  middle acceptor is the construct's global acceptor argmax in "
        f"{int((d.global_argmax_acceptor == d.site_middle_acceptor_pos).sum())}"
        f"/{len(d)}")
    say(f"  middle donor    is the construct's global donor argmax in "
        f"{int((d.global_argmax_donor == d.site_middle_donor_pos).sum())}"
        f"/{len(d)}")
    say()

    # --- 3. off-target peaks --------------------------------------------------
    say("-" * 74)
    say("3. OFF-TARGET PEAKS (annotated sites excluded)")
    say("-" * 74)
    nat = d[d.b1_id == "native_PNAS"]
    say(f"  strongest off-target peak anywhere, over all {len(d)} constructs:")
    say(f"      min {d.offtarget_max.min():.4f}   median "
        f"{d.offtarget_max.median():.4f}   max {d.offtarget_max.max():.4f}")
    say(f"  NATIVE B1 constructs (n={len(nat)}) off-target max: "
        f"{', '.join(f'{v:.4f}' for v in nat.offtarget_max)}")
    say(f"  inside the randomized B1 region:  median "
        f"{d.offtarget_max_b1.median():.4f}  max {d.offtarget_max_b1.max():.4f}")
    say(f"  outside B1 (rest of the construct): median "
        f"{d.offtarget_max_rest.median():.4f}  max "
        f"{d.offtarget_max_rest.max():.4f}")
    say(f"  with only the 4 exact positions excluded: max "
        f"{d.offtarget_max_exactexcl.max():.4f}")
    say()
    say("  constructs with ANY off-target position above a threshold:")
    for thr in C.THRESHOLDS:
        tag = str(thr).replace(".", "p")
        n_any = int((d[f"n_offtarget_gt{tag}"] > 0).sum())
        n_b1 = int(((d[f"n_donor_b1_gt{tag}"] + d[f"n_acceptor_b1_gt{tag}"]) > 0
                    ).sum())
        say(f"      > {thr}:  {n_any:2d}/{len(d)} constructs "
            f"({n_b1} of them with the peak inside B1); "
            f"max positions in one construct = "
            f"{int(d[f'n_offtarget_gt{tag}'].max())}")
    say()
    strongest = d.loc[d.offtarget_max.idxmax()]
    say(f"  single strongest off-target peak in the whole experiment: "
        f"{strongest.offtarget_max:.4f}")
    say(f"      {strongest.row_id}  (vs its annotated sites at "
        f"{strongest.site_middle_acceptor_prob:.3f} / "
        f"{strongest.site_middle_donor_prob:.3f})")
    say()

    # --- 4. correlation with the measured B1 effect ---------------------------
    say("-" * 74)
    say("4. DOES OFF-TARGET PEAK STRENGTH EXPLAIN THE B1 EFFECT?")
    say("-" * 74)
    prim = d[d.sequence_id == PRIMARY].copy()
    say(f"  primary exon {PRIMARY}, n={len(prim)} "
        f"(native + 36 fixed-length variants)")
    say(f"  delta_vs_native range {prim.delta_vs_native.min():+.3f} .. "
        f"{prim.delta_vs_native.max():+.3f}")
    say()
    rows = []
    for xcol, label in (("offtarget_max", "strongest off-target peak anywhere"),
                        ("offtarget_max_b1", "strongest peak inside B1"),
                        ("max_donor_b1", "strongest donor peak inside B1"),
                        ("max_acceptor_b1", "strongest acceptor peak inside B1")):
        for ycol, ylabel in (("delta_vs_native", "delta"),
                             ("abs_delta", "|delta|")):
            if ycol == "abs_delta":
                prim[ycol] = prim.delta_vs_native.abs()
            rho, pv = stats.spearmanr(prim[xcol], prim[ycol])
            r, pr = stats.pearsonr(prim[xcol], prim[ycol])
            rows.append({"x": label, "y": ylabel, "n": len(prim),
                         "spearman_rho": rho, "spearman_p": pv,
                         "pearson_r": r, "pearson_p": pr})
            say(f"  {label:38s} vs {ylabel:8s}  Spearman rho={rho:+.3f} "
                f"p={pv:.3f}   Pearson r={r:+.3f} p={pr:.3f}")
    pd.DataFrame(rows).to_csv(CHECK / "cryptic_correlations.csv", index=False)
    say()

    # the headline test
    rho, pv = stats.spearmanr(prim.offtarget_max, prim.delta_vs_native)
    say(f"  HEADLINE: Spearman(strongest off-target peak, delta_vs_native) = "
        f"{rho:+.4f}, p = {pv:.4f}, n = {len(prim)}")
    # how much of the spread could the strongest-peak variable account for
    sl, ic, rv, pvl, se = stats.linregress(prim.offtarget_max,
                                           prim.delta_vs_native)
    say(f"            linear fit r^2 = {rv**2:.4f} "
        f"(slope {sl:+.3f} logit per unit peak probability, p = {pvl:.3f})")
    say()

    # --- 5. cross-exon spot check --------------------------------------------
    say("-" * 74)
    say("5. CROSS-EXON SPOT CHECK (native + most extreme up/down variant)")
    say("-" * 74)
    sec = d[d.sequence_id != PRIMARY].sort_values(["sequence_id",
                                                   "delta_vs_native"])
    say(f"  {'exon':16s} {'b1_id':16s} {'delta':>8s} {'offtgt':>8s} "
        f"{'in_B1':>8s} {'drift_a':>8s} {'drift_d':>8s}")
    for r in sec.itertuples():
        say(f"  {r.sequence_id:16s} {r.b1_id:16s} {r.delta_vs_native:+8.3f} "
            f"{r.offtarget_max:8.4f} {r.offtarget_max_b1:8.4f} "
            f"{r.middle_acceptor_drift:8d} {r.middle_donor_drift:8d}")
    say()

    (CHECK / "cryptic_summary.txt").write_text("\n".join(lines) + "\n")
    d.to_csv(CHECK / "cryptic_measurements_annotated.csv", index=False)

    # --- plot -----------------------------------------------------------------
    _plot(prim, d, rho, pv)
    print(f"[out] {CHECK / 'cryptic_summary.txt'}")
    return 0


def _plot(prim: pd.DataFrame, d: pd.DataFrame, rho: float, pv: float) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    arms = {"native": ("k", "*", 220), "uniform": ("#1f77b4", "o", 55),
            "compmatched": ("#ff7f0e", "s", 50), "dinuc": ("#2ca02c", "^", 55)}

    for arm, (c, mk, s) in arms.items():
        sub = prim[prim.arm == arm]
        if sub.empty:
            continue
        ax[0].scatter(sub.offtarget_max, sub.delta_vs_native, c=c, marker=mk,
                      s=s, alpha=0.85, edgecolor="white", linewidth=0.6,
                      label=f"{arm} (n={len(sub)})", zorder=3)
    sl, ic, *_ = stats.linregress(prim.offtarget_max, prim.delta_vs_native)
    xs = np.linspace(prim.offtarget_max.min(), prim.offtarget_max.max(), 50)
    ax[0].plot(xs, sl * xs + ic, "r--", lw=1.4, zorder=2,
               label=f"fit (r$^2$={stats.linregress(prim.offtarget_max, prim.delta_vs_native)[2]**2:.3f})")
    ax[0].axhline(0, color="gray", lw=0.8, zorder=1)
    for thr in (0.1, 0.3, 0.5):
        ax[0].axvline(thr, color="gray", ls=":", lw=0.8, zorder=1)
        ax[0].text(thr, ax[0].get_ylim()[1], f" {thr}", fontsize=7,
                   color="gray", va="top")
    ax[0].set_xlabel("strongest off-target splice-site peak\n"
                     "(annotated sites excluded)")
    ax[0].set_ylabel(r"$\Delta$ AG_mean_logit vs native B1")
    ax[0].set_title(f"{PRIMARY}: off-target peak vs B1 effect\n"
                    rf"Spearman $\rho$={rho:+.3f}, p={pv:.3f}, n={len(prim)}",
                    fontsize=10)
    ax[0].legend(fontsize=7, loc="best")

    ax[1].hist(d.offtarget_max_b1, bins=18, color="#1f77b4", alpha=0.75,
               label="inside randomized B1")
    ax[1].hist(d.offtarget_max_rest, bins=18, color="#d62728", alpha=0.6,
               label="rest of construct")
    natv = d[d.b1_id == "native_PNAS"].offtarget_max_b1
    for v in natv:
        ax[1].axvline(v, color="k", ls="--", lw=1.2)
    ax[1].axvline(np.nan, color="k", ls="--", lw=1.2, label="native B1")
    for thr in (0.1, 0.3, 0.5):
        ax[1].axvline(thr, color="gray", ls=":", lw=0.9)
    ax[1].set_xlabel("strongest off-target peak probability")
    ax[1].set_ylabel("constructs")
    ax[1].set_title(f"Off-target peak distribution, all n={len(d)}\n"
                    "annotated sites (0.83-0.95) excluded", fontsize=10)
    ax[1].legend(fontsize=7)

    ax[2].scatter(prim.site_middle_acceptor_prob, prim.site_middle_donor_prob,
                  c=prim.offtarget_max, cmap="viridis", s=60,
                  edgecolor="k", linewidth=0.4)
    cb = fig.colorbar(ax[2].collections[0], ax=ax[2])
    cb.set_label("strongest off-target peak", fontsize=8)
    ax[2].set_xlabel("P(acceptor) at the middle exon")
    ax[2].set_ylabel("P(donor) at the middle exon")
    ax[2].set_title("Middle-exon probabilities, coloured by\n"
                    "off-target peak strength", fontsize=10)

    fig.tight_layout()
    p = CHECK / "cryptic_offtarget_vs_effect.png"
    fig.savefig(p, dpi=150)
    print(f"[out] {p}")


if __name__ == "__main__":
    raise SystemExit(main())
