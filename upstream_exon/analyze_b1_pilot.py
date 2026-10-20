"""
Analyse the upstream-exon (B1) pilot.

Reads ``outputs/upstream_exon/b1_pilot_ag_results.csv`` and writes tables and
plots into ``outputs/upstream_exon/``. Makes no AlphaGenome calls.

The fixed-length analysis uses the 36 randomized 113-nt B1 variants plus the
native PNAS B1. The 59-nt FAS B1 changes the construct length and is reported
separately as a reference point, never pooled into the distributions.

Usage::

    python upstream_exon/analyze_b1_pilot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

OUT = ROOT / "outputs" / "upstream_exon"
PLOTS = OUT / "plots"
RESULTS_CSV = OUT / "b1_pilot_ag_results.csv"
ENDPOINT_CSV = (ROOT / "outputs" / "synthetic_context" /
                "synthetic_endpoint_ag_results.csv")

RANDOM_ARMS = ["uniform", "compmatched", "dinuc"]
ARM_COLOR = {"uniform": "#2f6f9f", "compmatched": "#c98a2b",
             "dinuc": "#5a9e6f", "native": "#333333", "fas_reference": "#b4474e"}
#: AlphaGenome SPLICE_SITES probabilities arrive on a 1/256 grid
QUANTUM = 1 / 256


def load() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"missing {RESULTS_CSV}; run run_b1_pilot.py first")
    df = pd.read_csv(RESULTS_CSV)
    df["fixed_length"] = df.fixed_length.astype(bool)
    return df


# ==========================================================================
# tables
# ==========================================================================

def reproducibility(df: pd.DataFrame) -> pd.DataFrame:
    """Reproducibility of the native-B1 construct, two ways.

    ``abs_diff`` is within-run: the two deliberate replicate calls made back to
    back in this session. ``cross_session_diff`` compares this run's native
    value against the same construct's value stored by the 2026-10-07 endpoint
    run, which costs nothing because that construct is the ``PPPPPPPP``
    endpoint construct.
    """
    ep = pd.read_csv(ENDPOINT_CSV)
    ep = ep[ep.context == "PNAS"].set_index("sequence_id")
    nat = df[df.arm == "native"]
    rows = []
    for eid, g in nat.groupby("sequence_id"):
        g = g.sort_values("replicate")
        if len(g) < 2:
            continue
        row = {
            "sequence_id": eid,
            "r1_mean_logit": g.AG_mean_logit.iloc[0],
            "r2_mean_logit": g.AG_mean_logit.iloc[1],
            "abs_diff": abs(g.AG_mean_logit.iloc[0] - g.AG_mean_logit.iloc[1]),
            "acceptor_diff": abs(g.AG_acceptor_probability.iloc[0]
                                 - g.AG_acceptor_probability.iloc[1]),
            "donor_diff": abs(g.AG_donor_probability.iloc[0]
                              - g.AG_donor_probability.iloc[1]),
        }
        if eid in ep.index:
            row["endpoint_2026_10_07"] = float(ep.loc[eid, "AG_mean_logit"])
            row["cross_session_diff"] = abs(g.AG_mean_logit.iloc[0]
                                            - row["endpoint_2026_10_07"])
        rows.append(row)
    return pd.DataFrame(rows)


def native_levels(df: pd.DataFrame) -> pd.DataFrame:
    nat = df[df.arm == "native"].groupby("sequence_id").agg(
        native_mean_logit=("AG_mean_logit", "mean"),
        native_acceptor=("AG_acceptor_probability", "mean"),
        native_donor=("AG_donor_probability", "mean"),
        n_native=("AG_mean_logit", "size")).reset_index()
    return nat


def per_arm_summary(df: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(nat, on="sequence_id")
    d["delta_vs_native"] = d.AG_mean_logit - d.native_mean_logit
    meta = (d.groupby("sequence_id")
             .agg(exon_class=("exon_class", "first"),
                  alpha=("alpha", "first"),
                  PNAS_common=("PNAS_common", "first")).reset_index())
    g = (d[d.arm.isin(RANDOM_ARMS) | (d.arm == "fas_reference")]
         .groupby(["sequence_id", "arm"])
         .agg(n=("AG_mean_logit", "size"),
              median=("AG_mean_logit", "median"),
              mean=("AG_mean_logit", "mean"),
              sd=("AG_mean_logit", "std"),
              iqr=("AG_mean_logit", lambda s: s.quantile(.75) - s.quantile(.25)),
              lo=("AG_mean_logit", "min"),
              hi=("AG_mean_logit", "max"),
              median_delta=("delta_vs_native", "median"),
              frac_up=("delta_vs_native", lambda s: float((s > 0).mean())),
              ).reset_index().merge(meta, on="sequence_id"))
    return g.merge(nat[["sequence_id", "native_mean_logit"]], on="sequence_id"), d


def ranking(d: pd.DataFrame, noise: float) -> pd.DataFrame:
    """Rank exons by how much their response varies with B1 (fixed-length only)."""
    fl = d[d.fixed_length & d.arm.isin(RANDOM_ARMS)]
    rows = []
    for eid, g in fl.groupby("sequence_id"):
        sd = g.AG_mean_logit.std()
        spread = g.AG_mean_logit.quantile(.95) - g.AG_mean_logit.quantile(.05)
        iqr = g.AG_mean_logit.quantile(.75) - g.AG_mean_logit.quantile(.25)
        rows.append({
            "sequence_id": eid,
            "exon_class": g.exon_class.iloc[0],
            "alpha": g.alpha.iloc[0],
            "PNAS_common": g.PNAS_common.iloc[0],
            "native_mean_logit": g.native_mean_logit.iloc[0],
            "n_variants": len(g),
            "sd": sd,
            "iqr": iqr,
            "p95_minus_p5": spread,
            "min": g.AG_mean_logit.min(),
            "max": g.AG_mean_logit.max(),
            "median_delta": g.delta_vs_native.median(),
            "max_up": g.delta_vs_native.max(),
            "max_down": g.delta_vs_native.min(),
            "frac_up": float((g.delta_vs_native > 0).mean()),
            "frac_beyond_noise": float((g.delta_vs_native.abs() > 2 * noise).mean()),
            "sd_over_noise": sd / noise if noise > 0 else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("sd", ascending=False)
    out["rank_by_sd"] = range(1, len(out) + 1)
    return out


# ==========================================================================
# plots
# ==========================================================================

def plot_distributions(d: pd.DataFrame, nat: pd.DataFrame) -> Path:
    exons = list(nat.sequence_id)
    order = (d.groupby("sequence_id").PNAS_common.first()
             .sort_values(ascending=False).index.tolist())
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 5.2),
                             sharey=True)
    rng = np.random.default_rng(0)
    for ax, eid in zip(np.atleast_1d(axes), order):
        g = d[d.sequence_id == eid]
        for i, arm in enumerate(RANDOM_ARMS):
            s = g[g.arm == arm].AG_mean_logit.values
            if not len(s):
                continue
            x = i + rng.normal(0, 0.07, len(s))
            ax.scatter(x, s, s=26, alpha=.8, color=ARM_COLOR[arm],
                       edgecolor="white", linewidth=.5, zorder=3)
            ax.hlines(np.median(s), i - .28, i + .28, color=ARM_COLOR[arm],
                      linewidth=2.4, zorder=4)
        nv = g.native_mean_logit.iloc[0]
        ax.axhline(nv, color=ARM_COLOR["native"], linestyle="--", linewidth=1.4,
                   zorder=2, label="native PNAS B1")
        fas = g[g.arm == "fas_reference"].AG_mean_logit
        if len(fas):
            ax.axhline(fas.iloc[0], color=ARM_COLOR["fas_reference"],
                       linestyle=":", linewidth=1.6, zorder=2,
                       label="FAS B1 (59 nt)")
        ax.set_xticks(range(len(RANDOM_ARMS)))
        ax.set_xticklabels(["uniform", "comp-\nmatched", "dinuc\nshuffle"],
                           fontsize=9)
        cls = g.exon_class.iloc[0]
        ax.set_title(f"{eid}\npre-tuner {g.PNAS_common.iloc[0]:+.2f}  ({cls})",
                     fontsize=9)
        ax.grid(axis="y", alpha=.25)
    np.atleast_1d(axes)[0].set_ylabel(
        "AlphaGenome mean-logit\n(logit P(acceptor) + logit P(donor)) / 2")
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Middle-exon AlphaGenome response across randomized "
                 "upstream exon bodies (B1)", fontsize=12)
    fig.tight_layout()
    p = PLOTS / "b1_distributions_by_exon.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_deltas(d: pd.DataFrame, noise: float) -> Path:
    fl = d[d.fixed_length & d.arm.isin(RANDOM_ARMS)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, cls in zip(axes, ["plateau", "rising_control"]):
        g = fl[fl.exon_class == cls]
        for arm in RANDOM_ARMS:
            s = g[g.arm == arm].delta_vs_native.values
            if not len(s):
                continue
            ax.hist(s, bins=np.arange(-3, 3.01, 0.2), alpha=.55,
                    label=f"{arm} (n={len(s)})", color=ARM_COLOR[arm])
        ax.axvline(0, color="#333", linewidth=1.2)
        ax.axvspan(-2 * noise, 2 * noise, color="#999", alpha=.25,
                   label=f"±2× noise ({2*noise:.3f})")
        ax.set_title(f"{cls}  (n exons = {g.sequence_id.nunique()})")
        ax.set_xlabel("Δ AlphaGenome mean-logit  vs native PNAS B1")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("variants")
    fig.suptitle("Effect of replacing the upstream exon body, by middle-exon regime",
                 fontsize=12)
    fig.tight_layout()
    p = PLOTS / "b1_delta_distributions.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_on_saturation_curve(d: pd.DataFrame) -> Path:
    ep = pd.read_csv(ENDPOINT_CSV)
    ep = ep[ep.context == "PNAS"]
    bins = np.arange(-50, 16, 2.5)
    ep["xb"] = pd.cut(ep.PNAS_common, bins)
    med = ep.groupby("xb", observed=True).agg(
        x=("PNAS_common", "median"), y=("AG_mean_logit", "median"),
        n=("AG_mean_logit", "size")).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ep.PNAS_common, ep.AG_mean_logit, s=2, alpha=.07,
               color="#9fb6c9", zorder=1, label="10,000 exons, native context")
    ax.plot(med.x, med.y, color="#2f4858", linewidth=2.2, zorder=4,
            label="binned median")
    ax.axvline(5.0, color="#777", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(5.6, -1.2, "pre-tuner slope\nindistinguishable from\nflat beyond here",
            fontsize=8, color="#555")

    fl = d[d.fixed_length & d.arm.isin(RANDOM_ARMS)]
    for eid, g in fl.groupby("sequence_id"):
        x = g.PNAS_common.iloc[0]
        lo, hi = g.AG_mean_logit.min(), g.AG_mean_logit.max()
        col = "#b4474e" if g.exon_class.iloc[0] == "plateau" else "#2f6f9f"
        ax.vlines(x, lo, hi, color=col, linewidth=5, alpha=.55, zorder=5)
        ax.scatter([x], [g.native_mean_logit.iloc[0]], marker="_", s=320,
                   color="black", zorder=6, linewidth=2)
    ax.vlines([], [], [], color="#b4474e", linewidth=5, alpha=.55,
              label="B1 range, plateau exon")
    ax.vlines([], [], [], color="#2f6f9f", linewidth=5, alpha=.55,
              label="B1 range, rising control")
    ax.scatter([], [], marker="_", s=320, color="black", linewidth=2,
               label="native PNAS B1")
    ax.set_xlabel("PNAS pre-tuner score (common)  (energy_seq_struct)")
    ax.set_ylabel("AlphaGenome mean-logit")
    ax.set_title("B1 variation spans more of the response axis than the "
                 "pre-tuner plateau does")
    ax.set_xlim(-50, 16)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.25)
    fig.tight_layout()
    p = PLOTS / "b1_range_on_saturation_curve.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_effect_vs_strength(rank: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for cls, col in [("plateau", "#b4474e"), ("rising_control", "#2f6f9f")]:
        g = rank[rank.exon_class == cls]
        ax.scatter(g.PNAS_common, g.sd, s=110, color=col, label=cls,
                   edgecolor="white", linewidth=1, zorder=3)
        for r in g.itertuples():
            ax.annotate(r.sequence_id.replace("syn_", ""),
                        (r.PNAS_common, r.sd), fontsize=7,
                        xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("middle-exon PNAS pre-tuner score")
    ax.set_ylabel("SD of AlphaGenome mean-logit across 36 B1 variants")
    ax.set_title("Does B1 sensitivity depend on middle-exon strength?")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = PLOTS / "b1_effect_vs_exon_strength.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def plot_acceptor_donor(d: pd.DataFrame, nat_full: pd.DataFrame) -> Path:
    fl = d[d.fixed_length & d.arm.isin(RANDOM_ARMS)].copy()
    eps = 1e-7

    def lg(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    fl["d_acc"] = lg(fl.AG_acceptor_probability) - lg(fl.native_acceptor)
    fl["d_don"] = lg(fl.AG_donor_probability) - lg(fl.native_donor)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for arm in RANDOM_ARMS:
        g = fl[fl.arm == arm]
        ax.scatter(g.d_acc, g.d_don, s=30, alpha=.75, color=ARM_COLOR[arm],
                   label=arm, edgecolor="white", linewidth=.5)
    lim = max(fl.d_acc.abs().max(), fl.d_don.abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="#999", linestyle="--", linewidth=1)
    ax.axhline(0, color="#333", linewidth=.8)
    ax.axvline(0, color="#333", linewidth=.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δ logit P(acceptor)")
    ax.set_ylabel("Δ logit P(donor)")
    ax.set_title("Do both splice sites move together?")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    r = np.corrcoef(fl.d_acc, fl.d_don)[0, 1]
    ax.text(.04, .96, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            fontsize=10, va="top")
    fig.tight_layout()
    p = PLOTS / "b1_acceptor_vs_donor.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p, r


def main() -> int:
    df = load()
    PLOTS.mkdir(parents=True, exist_ok=True)

    print(f"[load] {len(df)} scored rows, {df.sequence_id.nunique()} exons, "
          f"{df.b1_id.nunique()} B1 variants")

    rep = reproducibility(df)
    nat = native_levels(df)
    summary, d = per_arm_summary(df, nat)
    # Practical noise floor: within-run replicates are exact, so the binding
    # constraint is cross-session drift on AlphaGenome's 1/256 probability grid.
    within = rep.abs_diff.max() if len(rep) else 0.0
    cross = (rep.cross_session_diff.max()
             if "cross_session_diff" in rep else 0.0)
    noise = max(within, cross, QUANTUM)
    rank = ranking(d, noise)

    rep.to_csv(OUT / "b1_pilot_reproducibility.csv", index=False)
    summary.to_csv(OUT / "b1_pilot_arm_summary.csv", index=False)
    rank.to_csv(OUT / "b1_pilot_candidate_ranking.csv", index=False)
    d.to_csv(OUT / "b1_pilot_results_annotated.csv", index=False)

    print("\n=== within-run reproducibility (duplicated native constructs) ===")
    print(rep.round(5).to_string(index=False))
    print(f"\nnoise floor (max |r1-r2|) = {noise:.4f} logit units")

    print("\n=== per-exon, per-arm summary ===")
    cols = ["sequence_id", "exon_class", "PNAS_common", "arm", "n",
            "native_mean_logit", "median", "sd", "lo", "hi",
            "median_delta", "frac_up"]
    print(summary[cols].round(3).to_string(index=False))

    print("\n=== candidate ranking (fixed-length 113-nt variants only) ===")
    rcols = ["rank_by_sd", "sequence_id", "exon_class", "PNAS_common",
             "native_mean_logit", "n_variants", "sd", "iqr", "p95_minus_p5",
             "min", "max", "max_up", "max_down", "frac_up", "frac_beyond_noise"]
    print(rank[rcols].round(3).to_string(index=False))

    p1 = plot_distributions(d, nat.merge(
        d.groupby("sequence_id").agg(PNAS_common=("PNAS_common", "first"))
        .reset_index(), on="sequence_id"))
    p2 = plot_deltas(d, noise)
    p3 = plot_on_saturation_curve(d)
    p4 = plot_effect_vs_strength(rank)
    p5, r_ad = plot_acceptor_donor(d, nat)

    print(f"\nacceptor/donor Δ correlation: r = {r_ad:.3f}")
    print("\nplots:")
    for p in (p1, p2, p3, p4, p5):
        print(f"  {p}")
    print("\ntables:")
    for n in ("b1_pilot_reproducibility.csv", "b1_pilot_arm_summary.csv",
              "b1_pilot_candidate_ranking.csv",
              "b1_pilot_results_annotated.csv"):
        print(f"  {OUT / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
