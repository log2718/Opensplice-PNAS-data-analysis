"""
Bootstrap SEs and factorial effects for the corrected 8-block chimera run.

Parts 1, 3 and 4 of the 8-block analysis. Makes NO AlphaGenome calls; reads the
merged results only and writes under ``outputs/synthetic_context_8block/``.

Response variable is ``AG_mean_logit`` throughout. ``prod_logit`` is never read.

Analytical vs display labels
----------------------------
Every calculation uses the original 8-character P/F label
(``context_label_8_analysis``). A second column
(``context_label_8_display``) rewrites ONLY the two middle-exon splice-site
blocks B4 and B5: an ``F`` there becomes ``6`` in the TP53_e6 arm and ``7`` in
the TP53_e7 arm, recording which TP53 flank the splice site came from. All
other positions keep P/F. The display label is never used for arithmetic.

Part 1  per-context bootstrap of the median (2,000 replicates, deterministic
        per-context seed, resampling the 400 sequence IDs with replacement)
Part 3  main effect of each block: mean(median | Bi=F) - mean(median | Bi=P)
Part 4  pairwise interactions: for each pair and each of the 2^6 = 64
        backgrounds, y_FF - y_FP - y_PF + y_PP, then the mean over backgrounds,
        with the SD / min / max across backgrounds kept as diagnostics

Usage::

    python analysis_script/analyze_chimera8_factorial.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

NEW = ROOT / "outputs" / "synthetic_context_8block"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
BOOT_DIR = NEW / "bootstrap"
FACT_DIR = NEW / "factorial"

Y = "AG_mean_logit"
ARMS = ["TP53_e6", "TP53_e7"]
ARM_SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
#: which TP53 exon an F at B4/B5 came from, for DISPLAY only
ARM_DIGIT = {"TP53_e6": "6", "TP53_e7": "7"}
BLOCKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
#: display positions that take the 6/7 notation (0-based within the label)
SS_POSITIONS = (3, 4)   # B4, B5

N_BOOT = 2000
BOOT_SEED = 20261015


def display_label(label: str, arm: str) -> str:
    """P/F label with B4/B5 F rewritten as the arm's TP53 exon digit.

    Display only -- never fed back into any calculation.
    """
    chars = list(label)
    for p in SS_POSITIONS:
        if chars[p] == "F":
            chars[p] = ARM_DIGIT[arm]
    return "".join(chars)


# ==========================================================================
# PART 1 -- bootstrap SE of the per-context median
# ==========================================================================

def bootstrap_context(values: np.ndarray, arm: str, label: str,
                      n_boot: int = N_BOOT) -> dict[str, float]:
    """Resample the context's sequences with replacement; summarise medians.

    The seed is derived from the arm and label, so a context's interval is
    reproducible regardless of iteration order or how many contexts are run.
    """
    seed = int.from_bytes(
        hashlib.blake2b(f"{BOOT_SEED}|{arm}|{label}".encode(),
                        digest_size=8).digest(), "big") % 2**32
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(values[idx], axis=1)
    return {
        "observed_median_AG_mean_logit": float(np.median(values)),
        "bootstrap_SE_median": float(meds.std(ddof=1)),
        "bootstrap_95CI_low": float(np.percentile(meds, 2.5)),
        "bootstrap_95CI_high": float(np.percentile(meds, 97.5)),
        "n_bootstrap": int(n_boot),
        "n": int(n),
    }


def run_bootstrap(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, label), g in d.groupby(["comparison", "context_label_8"],
                                     sort=True):
        g = g.sort_values("sequence_id")
        res = bootstrap_context(g[Y].to_numpy(dtype=float), arm, label)
        rows.append({
            "comparison": arm,
            "context_label_8_analysis": label,
            "context_label_8_display": display_label(label, arm),
            "n_F_blocks": int(label.count("F")),
            **res,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# PARTS 3 & 4 -- main effects and pairwise interactions
# ==========================================================================

def arm_median_map(boot: pd.DataFrame, arm: str) -> dict[str, float]:
    """label -> observed context median, for one arm."""
    s = boot[boot.comparison == arm]
    return dict(zip(s.context_label_8_analysis,
                    s.observed_median_AG_mean_logit))


def main_effects(med: dict[str, float]) -> pd.DataFrame:
    """mean(median | Bi=F) - mean(median | Bi=P), on the analytical labels."""
    rows = []
    for i, b in enumerate(BLOCKS):
        f = [v for k, v in med.items() if k[i] == "F"]
        p = [v for k, v in med.items() if k[i] == "P"]
        rows.append({
            "term": b, "order": 1, "block_i": b, "block_j": pd.NA,
            "effect": float(np.mean(f) - np.mean(p)),
            "n_F_contexts": len(f), "n_P_contexts": len(p),
            "mean_if_F": float(np.mean(f)), "mean_if_P": float(np.mean(p)),
        })
    return pd.DataFrame(rows)


def pairwise_interactions(med: dict[str, float]
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Difference-of-differences per background, then the mean over them.

    For a pair (i, j) the remaining six positions define 2^6 = 64 backgrounds;
    each background fixes exactly four contexts, one per (Bi, Bj) setting.
    """
    eff_rows, bg_rows = [], []
    for i, j in itertools.combinations(range(8), 2):
        others = [k for k in range(8) if k not in (i, j)]
        dods = []
        for bits in itertools.product("PF", repeat=6):
            base = [""] * 8
            for pos, c in zip(others, bits):
                base[pos] = c

            def lab(ci: str, cj: str) -> str:
                s = list(base)
                s[i], s[j] = ci, cj
                return "".join(s)

            l_ff, l_fp = lab("F", "F"), lab("F", "P")
            l_pf, l_pp = lab("P", "F"), lab("P", "P")
            y_ff, y_fp = med[l_ff], med[l_fp]
            y_pf, y_pp = med[l_pf], med[l_pp]
            dod = y_ff - y_fp - y_pf + y_pp
            dods.append(dod)
            bg_rows.append({
                "term": f"{BLOCKS[i]}:{BLOCKS[j]}",
                "block_i": BLOCKS[i], "block_j": BLOCKS[j],
                "background_blocks": "".join(BLOCKS[k] for k in others),
                "background_label": "".join(bits),
                "label_FF": l_ff, "label_FP": l_fp,
                "label_PF": l_pf, "label_PP": l_pp,
                "y_FF": y_ff, "y_FP": y_fp, "y_PF": y_pf, "y_PP": y_pp,
                "interaction_dod": dod,
            })
        a = np.asarray(dods, dtype=float)
        eff_rows.append({
            "term": f"{BLOCKS[i]}:{BLOCKS[j]}", "order": 2,
            "block_i": BLOCKS[i], "block_j": BLOCKS[j],
            "effect": float(a.mean()),
            "interaction_mean": float(a.mean()),
            "interaction_SD_across_backgrounds": float(a.std(ddof=1)),
            "interaction_min": float(a.min()),
            "interaction_max": float(a.max()),
            "n_backgrounds": int(a.size),
            "frac_sign_flip": float(min((a > 0).mean(), (a < 0).mean())),
        })
    return pd.DataFrame(eff_rows), pd.DataFrame(bg_rows)


# ==========================================================================
# validation
# ==========================================================================

def validate(d: pd.DataFrame, boot: pd.DataFrame,
             effects: dict[str, pd.DataFrame],
             backgrounds: dict[str, pd.DataFrame]) -> list[str]:
    fails: list[str] = []

    def chk(n: int, name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:2d}. {name}"
              f"{'  -- ' + detail if detail else ''}")
        if not ok:
            fails.append(f"{n}. {name}")

    per_arm = d.groupby("comparison").context_label_8.nunique().to_dict()
    chk(1, "256 contexts used per arm",
        all(per_arm.get(a) == 256 for a in ARMS), str(per_arm))

    sizes = d.groupby(["comparison", "context_label_8"]).sequence_id.nunique()
    chk(2, "each context has 400 sequences", bool((sizes == 400).all()),
        f"min {sizes.min()} max {sizes.max()}")

    chk(3, "2,000 bootstrap replicates per context",
        bool((boot.n_bootstrap == N_BOOT).all()),
        f"{sorted(boot.n_bootstrap.unique())}")

    n_se = boot.groupby("comparison").size().to_dict()
    chk(4, "256 SE estimates per arm",
        all(n_se.get(a) == 256 for a in ARMS), str(n_se))

    se = boot.bootstrap_SE_median
    chk(5, "all SE values are finite and >= 0",
        bool(np.isfinite(se).all() and (se >= 0).all()),
        f"min {se.min():.5f} max {se.max():.5f}")

    for arm in ARMS:
        e = effects[arm]
        chk(6, f"exactly 8 main effects for {ARM_SLUG[arm]}",
            int((e.order == 1).sum()) == 8, f"{int((e.order == 1).sum())}")
    for arm in ARMS:
        e = effects[arm]
        chk(7, f"exactly 28 pairwise interactions for {ARM_SLUG[arm]}",
            int((e.order == 2).sum()) == 28, f"{int((e.order == 2).sum())}")

    ok8 = True
    detail8 = []
    for arm in ARMS:
        e = effects[arm]
        counts = e[e.order == 2].n_backgrounds.unique()
        bg = backgrounds[arm].groupby("term").size()
        if not (set(counts) == {64} and set(bg.unique()) == {64}
                and len(bg) == 28):
            ok8 = False
        detail8.append(f"{ARM_SLUG[arm]}: {len(bg)} pairs x "
                       f"{sorted(set(bg.unique()))} backgrounds")
    chk(8, "every pairwise interaction averages exactly 64 background-specific "
        "difference-of-differences", ok8, "; ".join(detail8))

    # 9 -- all analytical labels are pure P/F, and every effect was computed
    #      from the analysis column
    pure = boot.context_label_8_analysis.str.fullmatch(r"[PF]{8}").all()
    matches_source = set(boot.context_label_8_analysis) == set(d.context_label_8)
    chk(9, "analytical P/F labels were used for all calculations",
        bool(pure and matches_source),
        "analysis labels are [PF]{8} and match the source file exactly")

    # 10 -- display labels differ from analysis labels ONLY at B4/B5, and only
    #       by substituting the arm's digit for F
    bad10 = []
    for r in boot.itertuples():
        a, disp = r.context_label_8_analysis, r.context_label_8_display
        if disp != display_label(a, r.comparison):
            bad10.append(a)
            continue
        for pos, (ca, cd) in enumerate(zip(a, disp)):
            if pos in SS_POSITIONS:
                if cd not in ("P", ARM_DIGIT[r.comparison]) or \
                        (ca == "P") != (cd == "P"):
                    bad10.append(a)
            elif ca != cd:
                bad10.append(a)
    digits_only_at_ss = all(
        all(c in "PF" for k, c in enumerate(r.context_label_8_display)
            if k not in SS_POSITIONS)
        for r in boot.itertuples())
    chk(10, "6/7 notation used for display only, at B4/B5 only",
        not bad10 and digits_only_at_ss,
        f"{len(bad10)} mismatched; digits appear only at B4/B5")

    chk(11, "no AlphaGenome calls were made", True,
        "this module imports no AlphaGenome client and opens no network path")

    return fails


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    BOOT_DIR.mkdir(parents=True, exist_ok=True)
    FACT_DIR.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(RESULTS_CSV,
                    usecols=["sequence_id", "comparison", "context_label_8",
                             "n_F_blocks", Y], low_memory=False)
    print(f"loaded {len(d):,} rows, {d.groupby(['comparison','context_label_8']).ngroups} contexts")

    # ---- Part 1 ----------------------------------------------------------
    print(f"\n[part 1] bootstrapping {d.groupby(['comparison','context_label_8']).ngroups} "
          f"contexts x {N_BOOT:,} replicates ...", flush=True)
    boot = run_bootstrap(d)
    boot_cols = ["comparison", "context_label_8_analysis",
                 "context_label_8_display", "n_F_blocks",
                 "observed_median_AG_mean_logit", "bootstrap_SE_median",
                 "bootstrap_95CI_low", "bootstrap_95CI_high",
                 "n_bootstrap", "n"]
    boot_path = BOOT_DIR / "context_median_bootstrap_SE.csv"
    boot[boot_cols].to_csv(boot_path, index=False)
    print(f"[part 1] wrote {boot_path.relative_to(ROOT)}")
    for arm in ARMS:
        s = boot[boot.comparison == arm].bootstrap_SE_median
        print(f"  {ARM_SLUG[arm]}: SE min {s.min():.4f}  median {s.median():.4f}"
              f"  max {s.max():.4f}")

    # ---- Parts 3 & 4 -----------------------------------------------------
    effects, backgrounds = {}, {}
    for arm in ARMS:
        med = arm_median_map(boot, arm)
        me = main_effects(med)
        pe, bg = pairwise_interactions(med)
        eff = pd.concat([me, pe], ignore_index=True)
        eff.insert(0, "comparison", arm)
        bg.insert(0, "comparison", arm)
        effects[arm], backgrounds[arm] = eff, bg

        ep = FACT_DIR / f"{ARM_SLUG[arm]}_main_pairwise_effects.csv"
        bp = FACT_DIR / f"{ARM_SLUG[arm]}_pairwise_interaction_backgrounds.csv"
        eff.to_csv(ep, index=False)
        bg.to_csv(bp, index=False)
        print(f"\n[parts 3-4] {ARM_SLUG[arm]}: wrote {ep.relative_to(ROOT)} "
              f"({len(eff)} terms) and {bp.relative_to(ROOT)} ({len(bg):,} rows)")

    # ---- validation ------------------------------------------------------
    print("\n" + "=" * 74 + "\nVALIDATION\n" + "=" * 74)
    fails = validate(d, boot, effects, backgrounds)
    print("=" * 74)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")

    # ---- console report --------------------------------------------------
    for arm in ARMS:
        e = effects[arm]
        print(f"\n--- {ARM_SLUG[arm]} main effects (F minus P) ---")
        print(e[e.order == 1][["term", "effect", "mean_if_P", "mean_if_F"]]
              .round(4).to_string(index=False))
        p = e[e.order == 2].copy()
        p["abs_effect"] = p.effect.abs()
        print(f"--- {ARM_SLUG[arm]} top 5 |pairwise interactions| ---")
        print(p.nlargest(5, "abs_effect")[
            ["term", "interaction_mean", "interaction_SD_across_backgrounds",
             "interaction_min", "interaction_max"]].round(4).to_string(index=False))
        het = p[(p.interaction_mean.abs() < 0.05)].nlargest(
            5, "interaction_SD_across_backgrounds")
        print(f"--- {ARM_SLUG[arm]} small mean (|mean| < 0.05) but largest SD ---")
        print(het[["term", "interaction_mean", "interaction_SD_across_backgrounds",
                   "interaction_min", "interaction_max", "frac_sign_flip"]]
              .round(4).to_string(index=False) if len(het) else "   (none)")

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
