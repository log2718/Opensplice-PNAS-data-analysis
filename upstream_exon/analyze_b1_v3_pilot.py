"""
Go / no-go analysis of the v3 fixed-composition pilot.

The question this answers is narrow and it is the only one worth asking
before spending ~250,000 AlphaGenome calls:

    with nucleotide composition held EXACTLY fixed, does AlphaGenome's
    response still vary enough to be modelled -- or does it collapse into
    the output quantization?

That risk is real and specific. AlphaGenome returns splice-site probabilities
on a grid of about 1/256, so near p = 0.95 one step is roughly 0.04 in
mean-logit units. The v2 library spanned 0.68 logit of total SD, but ~70% of
that was composition, which v3 removes on purpose.

Three numbers decide it:

  1. the SD of AG_mean_logit across the pilot, in logit units;
  2. the quantization step implied by the distinct probabilities returned;
  3. their ratio -- the effective resolution of the experiment.

A reference value is recomputed from the EXISTING v2 data rather than assumed:
the pooled SD of AG_mean_logit among v2 sequences that happen to share exactly
the same base counts. That is the same quantity the pilot measures, estimated
from data already paid for, so the pilot either confirms it or contradicts it.

Makes NO AlphaGenome calls. Writes only into
``outputs/upstream_exon/dataset_v3/pilot_analysis/``.

Usage::

    python upstream_exon/analyze_b1_v3_pilot.py
    python upstream_exon/analyze_b1_v3_pilot.py --no-baseline   # skip sklearn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

OUT = ROOT / "outputs" / "upstream_exon"
DATA = OUT / "dataset_v3"
V2 = OUT / "dataset_v2"
OUTDIR = DATA / "pilot_analysis"

#: below this, the response is mostly quantization noise and the full run is
#: not worth its API cost
MIN_EFFECTIVE_STEPS = 3.0
#: below this many distinct response values the target is too coarse to regress
MIN_DISTINCT_VALUES = 20


def quantization_step(probs: np.ndarray) -> float:
    """Smallest positive gap between distinct returned probabilities."""
    u = np.unique(probs)
    if len(u) < 2:
        return float("nan")
    d = np.diff(u)
    d = d[d > 1e-12]
    return float(np.min(d)) if len(d) else float("nan")


def logit_step_at(p: float, dp: float) -> float:
    """How much one probability step moves the logit, near ``p``."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(dp / (p * (1 - p)))


def v2_fixed_composition_sd() -> dict:
    """Pooled within-exact-composition SD from the v2 library, for reference.

    v2 varied composition, but by chance many of its sequences share exactly
    the same base counts. The spread WITHIN such a set is, by definition,
    response variation at fixed composition -- an estimate of what v3 should
    see, obtained from calls already made.
    """
    res = V2 / "b1_primary_ag_results.csv"
    man = V2 / "b1_primary_manifest.csv"
    if not (res.exists() and man.exists()):
        return {}
    r = pd.read_csv(res, usecols=["row_id", "AG_mean_logit"])
    m = pd.read_csv(man, usecols=["row_id", "count_A", "count_C",
                                  "count_G", "count_T"])
    d = r.merge(m, on="row_id")
    key = ["count_A", "count_C", "count_G", "count_T"]
    g = d.groupby(key).AG_mean_logit
    size = g.transform("size")
    sub = d[size >= 2]
    if sub.empty:
        return {}
    gs = sub.groupby(key).AG_mean_logit
    ss = float(((sub.AG_mean_logit - gs.transform("mean")) ** 2).sum())
    dof = len(sub) - gs.ngroups
    return {
        "v2_total_sd": float(d.AG_mean_logit.std()),
        "v2_within_exact_composition_sd": float(np.sqrt(ss / dof)),
        "v2_cells_used": int(gs.ngroups),
        "v2_sequences_used": int(len(sub)),
        "v2_order_variance_share": float((ss / dof) / d.AG_mean_logit.var()),
    }


def order_feature_fit(df: pd.DataFrame) -> dict:
    """Can dinucleotide frequency alone predict the pilot response?

    Fitted on 80% of the pilot and scored on the held-out 20%. With so few
    rows this is indicative only -- it says whether there is any order signal
    at all, not how well the full experiment will do.
    """
    try:
        import baselines as BL
    except ImportError as e:
        return {"order_fit_skipped": str(e)}
    if len(df) < 100:
        return {"order_fit_skipped": f"only {len(df)} rows"}
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    y_te = te.AG_mean_logit.to_numpy()
    out = {}
    for name, model in (("dinucleotide_ridge", BL.dinucleotide_baseline()),
                        ("kmer3_ridge", BL.kmer_baseline(3))):
        model.fit(tr.b1.tolist(), tr.AG_mean_logit.to_numpy())
        pred = model.predict(te.b1.tolist())
        ss_res = float(((y_te - pred) ** 2).sum())
        ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
        out[f"{name}_holdout_r2"] = float(1 - ss_res / ss_tot) if ss_tot else float("nan")
        out[f"{name}_alpha"] = float(getattr(model, "alpha_", float("nan")))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-csv",
                    default=str(DATA / "b1_primary_ag_results.csv"))
    ap.add_argument("--manifest-csv",
                    default=str(DATA / "b1_primary_manifest.csv"))
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the dinucleotide / 3-mer fit (no sklearn)")
    args = ap.parse_args(argv)

    res_path, man_path = Path(args.results_csv), Path(args.manifest_csv)
    if not res_path.exists():
        raise SystemExit(
            f"missing {res_path}\n"
            "  Run the pilot scoring job first:\n"
            "    sbatch upstream_exon/slurm/score_b1_v3.sbatch   # PILOT=1")
    if not man_path.exists():
        raise SystemExit(f"missing {man_path}; run build_b1_dataset_v3.py")

    res = pd.read_csv(res_path)
    man = pd.read_csv(man_path, usecols=["row_id", "b1", "split", "pilot",
                                         "max_homopolymer", "distinct_3mers",
                                         "distinct_6mers"])
    df = res.merge(man, on="row_id", how="inner", validate="one_to_one")
    pilot = df[df.pilot.astype(bool)] if "pilot" in df else df

    y = df.AG_mean_logit.to_numpy()
    step_acc = quantization_step(df.AG_acceptor_probability.to_numpy())
    step_don = quantization_step(df.AG_donor_probability.to_numpy())
    p_acc = float(df.AG_acceptor_probability.mean())
    p_don = float(df.AG_donor_probability.mean())
    # AG_mean_logit averages the two arms, so one probability step on a single
    # arm moves the target by half its logit step
    step_logit = 0.5 * max(logit_step_at(p_acc, step_acc),
                           logit_step_at(p_don, step_don))
    sd = float(y.std(ddof=1))
    eff_steps = sd / step_logit if step_logit > 0 else float("nan")

    stats = {
        "rows_scored": int(len(df)),
        "pilot_rows": int(len(pilot)),
        "target_mean": float(y.mean()),
        "target_sd": sd,
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_range": float(y.max() - y.min()),
        "target_iqr": float(np.subtract(*np.percentile(y, [75, 25]))),
        "distinct_target_values": int(df.AG_mean_logit.nunique()),
        "distinct_acceptor_probs": int(df.AG_acceptor_probability.nunique()),
        "distinct_donor_probs": int(df.AG_donor_probability.nunique()),
        "mean_acceptor_prob": p_acc,
        "mean_donor_prob": p_don,
        "frac_acceptor_at_ceiling": float((df.AG_acceptor_probability >= 0.999).mean()),
        "frac_donor_at_ceiling": float((df.AG_donor_probability >= 0.999).mean()),
        "prob_quantization_step_acceptor": step_acc,
        "prob_quantization_step_donor": step_don,
        "logit_quantization_step": step_logit,
        "effective_resolution_steps": float(eff_steps),
    }
    stats.update(v2_fixed_composition_sd())
    if not args.no_baseline:
        stats.update(order_feature_fit(df))

    # correlation of the response with the few order statistics we recorded
    corr = {}
    for c in ("max_homopolymer", "distinct_3mers", "distinct_6mers"):
        if c in df and df[c].nunique() > 1:
            corr[f"pearson_r_vs_{c}"] = float(np.corrcoef(df[c], y)[0, 1])
    stats.update(corr)

    # ---------------- verdict ----------------
    reasons = []
    if stats["distinct_target_values"] < MIN_DISTINCT_VALUES:
        reasons.append(
            f"only {stats['distinct_target_values']} distinct response values "
            f"(need >= {MIN_DISTINCT_VALUES})")
    if not np.isfinite(eff_steps) or eff_steps < MIN_EFFECTIVE_STEPS:
        reasons.append(
            f"response SD is {eff_steps:.1f} quantization steps "
            f"(need >= {MIN_EFFECTIVE_STEPS})")
    if stats["frac_acceptor_at_ceiling"] > 0.9 and stats["frac_donor_at_ceiling"] > 0.9:
        reasons.append("both splice sites are saturated at p ~ 1")
    verdict = "GO" if not reasons else "NO-GO"
    stats["verdict"] = verdict
    stats["verdict_reasons"] = reasons

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "pilot_summary.json").write_text(json.dumps(stats, indent=2))
    df[["row_id", "split", "AG_acceptor_probability", "AG_donor_probability",
        "AG_mean_logit", "max_homopolymer", "distinct_3mers",
        "distinct_6mers"]].to_csv(OUTDIR / "pilot_rows.csv", index=False)
    pd.Series(y).value_counts().sort_index().rename("n").to_csv(
        OUTDIR / "pilot_response_histogram.csv", header=True)

    # ---------------- report ----------------
    W = 72
    print("=" * W)
    print(" v3 FIXED-COMPOSITION PILOT")
    print("=" * W)
    print(f" rows scored              : {stats['rows_scored']:,}")
    print(f" AG_mean_logit            : mean {stats['target_mean']:+.4f}   "
          f"sd {sd:.4f}")
    print(f"                            range [{stats['target_min']:+.3f}, "
          f"{stats['target_max']:+.3f}]   IQR {stats['target_iqr']:.4f}")
    print(f" distinct response values : {stats['distinct_target_values']} "
          f"(acceptor {stats['distinct_acceptor_probs']}, "
          f"donor {stats['distinct_donor_probs']} distinct probabilities)")
    print(f" mean probabilities       : acceptor {p_acc:.4f}  donor {p_don:.4f}")
    print(f" saturated at p>=0.999    : acceptor "
          f"{stats['frac_acceptor_at_ceiling']*100:.1f}%  donor "
          f"{stats['frac_donor_at_ceiling']*100:.1f}%")
    print()
    print(f" quantization step        : {step_logit:.4f} logit")
    print(f" EFFECTIVE RESOLUTION     : {eff_steps:.1f} steps of response SD")
    if "v2_within_exact_composition_sd" in stats:
        print()
        print(" reference, recomputed from the existing v2 data:")
        print(f"   v2 total SD                        {stats['v2_total_sd']:.4f}")
        print(f"   v2 SD within exact composition     "
              f"{stats['v2_within_exact_composition_sd']:.4f}  "
              f"({stats['v2_sequences_used']:,} sequences in "
              f"{stats['v2_cells_used']:,} composition cells)")
        print(f"   -> order-driven variance share     "
              f"{stats['v2_order_variance_share']*100:.1f}%")
        print(f"   pilot SD / v2 reference            "
              f"{sd / stats['v2_within_exact_composition_sd']:.2f}x")
    for k, v in stats.items():
        if k.endswith("_holdout_r2"):
            print(f" {k:40s} {v:+.4f}")
    if corr:
        print()
        for k, v in corr.items():
            print(f" {k:40s} {v:+.4f}")
    print()
    print("=" * W)
    print(f" VERDICT: {verdict}")
    for r in reasons:
        print(f"   - {r}")
    if verdict == "GO":
        print("   response varies well above the quantization floor;")
        print("   the full 250,000-sequence run is justified.")
    else:
        print("   Do NOT launch the full run on this evidence.")
    print("=" * W)
    print(f"\nwrote {OUTDIR}/pilot_summary.json")
    print(f"wrote {OUTDIR}/pilot_rows.csv")
    print(f"wrote {OUTDIR}/pilot_response_histogram.csv")
    return 0 if verdict == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
