"""
Summarise the manual saturation labels for the 8-block chimera contexts.

Reads ``outputs/synthetic_context_8block/manual_saturation_labels.csv`` (written
by ``label_saturation.py``) and produces:

``manual_saturation_summary.csv``        overall + per-arm means
``manual_saturation_by_block.csv``       per-block main effect on the manual value
``manual_saturation_by_nF.csv``          mean by number of FAS blocks
``manual_saturation_vs_measured.csv``    manual label joined to the measured
                                         per-context median AG mean-logit

Makes no AlphaGenome calls and never modifies the label file.

Usage::

    python analysis_script/summarize_manual_saturation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import splice_site_partition as SP  # noqa: E402

NEW = ROOT / "outputs" / "synthetic_context_8block"
LABELS_CSV = NEW / "manual_saturation_labels.csv"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
OUT_DIR = NEW / "manual_saturation"

VAL = "saturation_value"
BLOCK_ROLE = {
    "B1": "upstream exon body", "B2": "upstream 5'SS",
    "B3": "upstream intron", "B4": "middle 3'SS", "B5": "middle 5'SS",
    "B6": "downstream intron", "B7": "downstream 3'SS",
    "B8": "downstream exon body",
}


def load() -> pd.DataFrame:
    if not LABELS_CSV.exists():
        raise SystemExit(f"missing {LABELS_CSV}")
    d = pd.read_csv(LABELS_CSV)
    d["no_saturation"] = d.no_saturation.fillna(0).astype(float).astype(int)
    d["low_confidence"] = d.low_confidence.fillna(0).astype(float).astype(int)
    for i, b in enumerate(SP.BLOCK_ORDER_8):
        d[b] = d.context_label_8.str[i]
    return d


def describe(s: pd.Series) -> dict:
    return {
        "n": int(s.notna().sum()),
        "mean": float(s.mean()),
        "sd": float(s.std()),
        "median": float(s.median()),
        "q25": float(s.quantile(.25)),
        "q75": float(s.quantile(.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load()
    sat = d[d.no_saturation == 0]

    # ---- 1. overall + per arm -----------------------------------------
    rows = [{"scope": "ALL", **describe(sat[VAL]),
             "n_no_saturation": int(d.no_saturation.sum()),
             "n_low_confidence": int(d.low_confidence.sum())}]
    for arm, g in d.groupby("comparison"):
        gs = g[g.no_saturation == 0]
        rows.append({"scope": arm, **describe(gs[VAL]),
                     "n_no_saturation": int(g.no_saturation.sum()),
                     "n_low_confidence": int(g.low_confidence.sum())})
    # excluding low-confidence panels, as a sensitivity check
    hc = sat[sat.low_confidence == 0]
    rows.append({"scope": "ALL (high-confidence only)", **describe(hc[VAL]),
                 "n_no_saturation": 0, "n_low_confidence": 0})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "manual_saturation_summary.csv", index=False)

    # ---- 2. per-block main effect -------------------------------------
    brows = []
    for arm, g in sat.groupby("comparison"):
        for b in SP.BLOCK_ORDER_8:
            f = g.loc[g[b] == "F", VAL]
            p = g.loc[g[b] == "P", VAL]
            brows.append({
                "comparison": arm, "block": b, "role": BLOCK_ROLE[b],
                "mean_if_F": float(f.mean()), "mean_if_P": float(p.mean()),
                "effect_F_minus_P": float(f.mean() - p.mean()),
                "n_F": int(len(f)), "n_P": int(len(p)),
            })
    by_block = pd.DataFrame(brows)
    by_block.to_csv(OUT_DIR / "manual_saturation_by_block.csv", index=False)

    # ---- 3. by number of FAS blocks -----------------------------------
    by_nf = (sat.groupby(["comparison", "n_F_blocks"])[VAL]
             .agg(["count", "mean", "std", "median"]).reset_index())
    by_nf.to_csv(OUT_DIR / "manual_saturation_by_nF.csv", index=False)

    # ---- 4. manual vs measured median ---------------------------------
    joined = pd.DataFrame()
    if RESULTS_CSV.exists():
        meas = (pd.read_csv(RESULTS_CSV,
                            usecols=["comparison", "context_label_8",
                                     "AG_mean_logit"], low_memory=False)
                .groupby(["comparison", "context_label_8"]).AG_mean_logit
                .agg(measured_median="median",
                     measured_p90=lambda s: s.quantile(.90),
                     measured_max="max").reset_index())
        joined = sat.merge(meas, on=["comparison", "context_label_8"],
                           how="left")
        joined[["comparison", "context_label_8", "n_F_blocks", VAL,
                "low_confidence", "measured_median", "measured_p90",
                "measured_max"]].to_csv(
            OUT_DIR / "manual_saturation_vs_measured.csv", index=False)

    # ================= report =================
    print("=" * 70)
    print(" MANUAL SATURATION LABELS")
    print("=" * 70)
    print(f" panels labelled      : {len(d)}  "
          f"({d.comparison.value_counts().to_dict()})")
    print(f" marked 'no saturation': {int(d.no_saturation.sum())}")
    print(f" flagged low confidence: {int(d.low_confidence.sum())} "
          f"({100*d.low_confidence.mean():.1f}%)")
    print()
    print(summary[["scope", "n", "mean", "sd", "median", "q25", "q75",
                   "min", "max"]].round(3).to_string(index=False))

    print("\n--- mean by arm ---")
    for arm, g in sat.groupby("comparison"):
        print(f"  {arm}:  mean {g[VAL].mean():+.4f}   "
              f"median {g[VAL].median():+.4f}   sd {g[VAL].std():.4f}   "
              f"n {len(g)}")
    a, b = [g[VAL] for _, g in sat.groupby("comparison")]
    print(f"  difference (e6 - e7): {a.mean() - b.mean():+.4f}")

    print("\n--- per-block main effect on the manual value (F minus P) ---")
    piv = by_block.pivot(index="block", columns="comparison",
                         values="effect_F_minus_P")
    piv["role"] = [BLOCK_ROLE[b] for b in piv.index]
    print(piv.round(4).to_string())

    print("\n--- mean by number of FAS blocks ---")
    print(by_nf.pivot(index="n_F_blocks", columns="comparison",
                      values="mean").round(3).to_string())

    if not joined.empty:
        from scipy import stats
        print("\n--- manual label vs measured AG mean-logit ---")
        for name, col in (("median", "measured_median"),
                          ("p90", "measured_p90"),
                          ("max", "measured_max")):
            g = joined.dropna(subset=[col])
            r = stats.pearsonr(g[VAL], g[col])
            sp = stats.spearmanr(g[VAL], g[col])
            print(f"  vs per-context {name:6s}: Pearson {r.statistic:+.3f} "
                  f"(p={r.pvalue:.2e})   Spearman {sp.statistic:+.3f}")

    print(f"\nwrote 4 tables -> {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
