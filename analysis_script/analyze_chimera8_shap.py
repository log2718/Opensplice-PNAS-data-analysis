"""
EXACT Shapley values for the corrected 8-block chimera design.

No surrogate model is fitted and no coalition is sampled. The design is a
COMPLETE 2^8 factorial per arm, so the response function

    f(B1..B8) = median AG_mean_logit over the 400 synthetic exons

is known at every one of its 256 points, and the Shapley value can be
evaluated from its definition by enumerating all coalitions.

For a context x and block i, with M = 8::

    phi_i(x) = sum_{S subset of N\\{i}}  |S|!(M-|S|-1)!/M!
                                        * [ v_x(S + {i}) - v_x(S) ]

and ``v_x(S)`` is the observed response averaged over every block NOT in S,
with the blocks in S held at their values in x. With the full factorial that
average is exact -- it is a mean over the relevant slice of the lookup table,
not an estimate. ``v_x({}) = baseline`` is the mean context median over all
256 contexts of the arm, as specified.

Exact SHAP interaction values are computed the same way::

    phi_ij(x) = sum_{S subset of N\\{i,j}}  |S|!(M-|S|-2)!/(2(M-1)!)
                                           * [ v_x(S+{i,j}) - v_x(S+{i})
                                               - v_x(S+{j}) + v_x(S) ]

with the diagonal set to phi_ii = phi_i - sum_{j != i} phi_ij, so the whole
matrix sums to f(x) - baseline.

Everything uses the original P/F analytical labels. The 6/7 notation for
B4/B5 is display only and is never used as a variable here.

Makes NO AlphaGenome calls.

Usage::

    python analysis_script/analyze_chimera8_shap.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

NEW = ROOT / "outputs" / "synthetic_context_8block"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
FACT_DIR = NEW / "factorial"
SHAP_DIR = NEW / "shap"

Y = "AG_mean_logit"
ARMS = ["TP53_e6", "TP53_e7"]
ARM_SLUG = {"TP53_e6": "e6", "TP53_e7": "e7"}
BLOCKS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
M = 8
#: bit encoding used for the lookup table axes: P -> 0, F -> 1
STATE_BIT = {"P": 0, "F": 1}
BIT_STATE = {0: "P", 1: "F"}


# ==========================================================================
# the response function as a complete lookup table
# ==========================================================================

def aggregate(g, stat: str = "median") -> float:
    """Context-level response: the median, or the 90th percentile."""
    if stat == "median":
        return g.median()
    if stat == "p90":
        return g.quantile(0.90)
    raise ValueError(f"unknown statistic {stat!r}")


def response_table(d: pd.DataFrame, arm: str,
                   stat: str = "median") -> tuple[np.ndarray, list[str]]:
    """``f`` as an 8-dimensional array of shape (2,)*8, axis k = block k+1.

    Returns the array and the 256 labels in bit order, so the array and the
    label list can be checked against each other.
    """
    s = d[d.comparison == arm]
    med = aggregate(s.groupby("context_label_8")[Y], stat)
    if len(med) != 256:
        raise SystemExit(f"[abort] {arm}: {len(med)} contexts, expected 256")

    f = np.full((2,) * M, np.nan)
    labels: list[str] = []
    for bits in itertools.product((0, 1), repeat=M):
        lab = "".join(BIT_STATE[b] for b in bits)
        if lab not in med.index:
            raise SystemExit(f"[abort] {arm}: missing context {lab}")
        f[bits] = med[lab]
        labels.append(lab)
    if np.isnan(f).any():
        raise SystemExit(f"[abort] {arm}: incomplete response table")
    return f, labels


def conditional_means(f: np.ndarray) -> dict[int, np.ndarray]:
    """``v_x(S)`` for every coalition S, as a full (2,)*8 array per S.

    ``S`` is a bitmask over the 8 blocks. Averaging over the axes NOT in S and
    broadcasting back gives an array whose value at x is exactly v_x(S).
    """
    out: dict[int, np.ndarray] = {}
    for mask in range(1 << M):
        drop = tuple(k for k in range(M) if not (mask >> k) & 1)
        out[mask] = (f.mean(axis=drop, keepdims=True) if drop
                     else f).copy() if drop else f.copy()
        if drop:
            out[mask] = np.broadcast_to(f.mean(axis=drop, keepdims=True),
                                        f.shape).copy()
    return out


# ==========================================================================
# exact Shapley values and exact SHAP interaction values
# ==========================================================================

def shapley_weights() -> np.ndarray:
    """|S|!(M-|S|-1)!/M! indexed by |S|."""
    return np.array([factorial(s) * factorial(M - s - 1) / factorial(M)
                     for s in range(M)])


def interaction_weights() -> np.ndarray:
    """|S|!(M-|S|-2)!/(2(M-1)!) indexed by |S|."""
    return np.array([factorial(s) * factorial(M - s - 2)
                     / (2 * factorial(M - 1)) for s in range(M - 1)])


def exact_shap(f: np.ndarray, v: dict[int, np.ndarray]) -> np.ndarray:
    """phi of shape (8,) + (2,)*8 -- one full value array per block."""
    w = shapley_weights()
    phi = np.zeros((M,) + f.shape)
    for i in range(M):
        bit_i = 1 << i
        others = [k for k in range(M) if k != i]
        for r in range(M):                       # |S| = r
            for comb in itertools.combinations(others, r):
                mask = 0
                for k in comb:
                    mask |= 1 << k
                phi[i] += w[r] * (v[mask | bit_i] - v[mask])
    return phi


def exact_shap_interactions(f: np.ndarray, v: dict[int, np.ndarray],
                            phi: np.ndarray) -> np.ndarray:
    """phi_ij of shape (8, 8) + (2,)*8, symmetric, with the standard diagonal."""
    w = interaction_weights()
    inter = np.zeros((M, M) + f.shape)
    for i, j in itertools.combinations(range(M), 2):
        bi, bj = 1 << i, 1 << j
        others = [k for k in range(M) if k not in (i, j)]
        acc = np.zeros(f.shape)
        for r in range(M - 1):                   # |S| = 0 .. M-2
            for comb in itertools.combinations(others, r):
                mask = 0
                for k in comb:
                    mask |= 1 << k
                acc += w[r] * (v[mask | bi | bj] - v[mask | bi]
                               - v[mask | bj] + v[mask])
        inter[i, j] = acc
        inter[j, i] = acc
    for i in range(M):
        off = sum(inter[i, j] for j in range(M) if j != i)
        inter[i, i] = phi[i] - off
    return inter


# ==========================================================================
# tidy output
# ==========================================================================

def to_long(phi: np.ndarray, f: np.ndarray, baseline: float,
            arm: str, value_col: str = "observed_median_AG_mean_logit"
            ) -> pd.DataFrame:
    rows = []
    for bits in itertools.product((0, 1), repeat=M):
        lab = "".join(BIT_STATE[b] for b in bits)
        obs = float(f[bits])
        for i, b in enumerate(BLOCKS):
            rows.append({
                "comparison": arm,
                "context_label_8": lab,
                value_col: obs,
                "baseline": baseline,
                "block": b,
                "block_state": BIT_STATE[bits[i]],
                "shap_value": float(phi[i][bits]),
            })
    return pd.DataFrame(rows)


def to_wide(phi: np.ndarray, f: np.ndarray, baseline: float,
            arm: str, value_col: str = "observed_median_AG_mean_logit"
            ) -> pd.DataFrame:
    rows = []
    for bits in itertools.product((0, 1), repeat=M):
        lab = "".join(BIT_STATE[b] for b in bits)
        r = {"comparison": arm, "context_label_8": lab,
             value_col: float(f[bits]),
             "baseline": baseline,
             "n_F_blocks": int(sum(bits))}
        for i, b in enumerate(BLOCKS):
            r[f"shap_{b}"] = float(phi[i][bits])
        r["shap_sum"] = sum(r[f"shap_{b}"] for b in BLOCKS)
        r["reconstruction"] = baseline + r["shap_sum"]
        r["reconstruction_error"] = r["reconstruction"] - r[value_col]
        rows.append(r)
    return pd.DataFrame(rows)


def feature_summary(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("block")
    out = pd.DataFrame({
        "mean_abs_SHAP": g.shap_value.apply(lambda s: s.abs().mean()),
        "mean_SHAP_when_F": long[long.block_state == "F"].groupby("block")
                                .shap_value.mean(),
        "mean_SHAP_when_P": long[long.block_state == "P"].groupby("block")
                                .shap_value.mean(),
        "SD_SHAP": g.shap_value.std(ddof=1),
        "min_SHAP": g.shap_value.min(),
        "max_SHAP": g.shap_value.max(),
        "n": g.shap_value.size(),
    }).reset_index()
    # within-state dispersion: between-state variation IS the main effect, so
    # the spread of phi_i WITHIN a state is what measures how much block i's
    # attribution depends on the other seven blocks.
    wf = long[long.block_state == "F"].groupby("block").shap_value.std(ddof=1)
    wp = long[long.block_state == "P"].groupby("block").shap_value.std(ddof=1)
    out["SD_SHAP_within_F"] = out.block.map(wf)
    out["SD_SHAP_within_P"] = out.block.map(wp)
    out["SD_SHAP_within_mean"] = (out.SD_SHAP_within_F + out.SD_SHAP_within_P) / 2
    # does the attribution ever flip sign against its own state's mean?
    sign_flip = {}
    for b, g in long.groupby("block"):
        bad = 0
        for st, gs in g.groupby("block_state"):
            m = gs.shap_value.mean()
            bad += int((np.sign(gs.shap_value) != np.sign(m)).sum()) if m != 0 else 0
        sign_flip[b] = bad / len(g)
    out["frac_sign_against_state_mean"] = out.block.map(sign_flip)
    out["range_SHAP"] = out.max_SHAP - out.min_SHAP
    out = out.sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def interactions_long(inter: np.ndarray, arm: str) -> pd.DataFrame:
    rows = []
    for bits in itertools.product((0, 1), repeat=M):
        lab = "".join(BIT_STATE[b] for b in bits)
        for i, j in itertools.combinations(range(M), 2):
            rows.append({
                "comparison": arm, "context_label_8": lab,
                "block_i": BLOCKS[i], "block_j": BLOCKS[j],
                "term": f"{BLOCKS[i]}:{BLOCKS[j]}",
                "state_i": BIT_STATE[bits[i]], "state_j": BIT_STATE[bits[j]],
                "shap_interaction_value": float(inter[i, j][bits]),
            })
    return pd.DataFrame(rows)


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(RESULTS_CSV,
                    usecols=["comparison", "context_label_8", "sequence_id", Y],
                    low_memory=False)
    print(f"loaded {len(d):,} rows")

    report: dict[str, dict] = {}
    max_err_all, max_inter_err_all = 0.0, 0.0

    for arm in ARMS:
        slug = ARM_SLUG[arm]
        f, labels = response_table(d, arm)
        baseline = float(f.mean())
        v = conditional_means(f)
        assert np.isclose(v[0].flat[0], baseline), "v(empty) must be the baseline"

        phi = exact_shap(f, v)
        inter = exact_shap_interactions(f, v, phi)

        long = to_long(phi, f, baseline, arm)
        wide = to_wide(phi, f, baseline, arm)
        summ = feature_summary(long)
        ilong = interactions_long(inter, arm)

        err = float(np.abs(wide.reconstruction_error).max())
        max_err_all = max(max_err_all, err)
        # interaction efficiency: the whole 8x8 matrix sums to f - baseline
        isum = inter.sum(axis=(0, 1))
        ierr = float(np.abs(isum - (f - baseline)).max())
        max_inter_err_all = max(max_inter_err_all, ierr)

        long.to_csv(SHAP_DIR / f"{slug}_exact_shap_values.csv", index=False)
        wide.to_csv(SHAP_DIR / f"{slug}_exact_shap_values_wide.csv", index=False)
        summ.to_csv(SHAP_DIR / f"{slug}_shap_feature_summary.csv", index=False)
        ilong.to_csv(SHAP_DIR / f"{slug}_exact_shap_interactions.csv", index=False)
        np.save(SHAP_DIR / f"{slug}_shap_interaction_tensor.npy", inter)

        report[arm] = {"baseline": baseline, "recon_err": err,
                       "inter_err": ierr, "summary": summ, "wide": wide,
                       "long": long, "inter": inter, "f": f}

        print(f"\n=== {slug} ===")
        print(f"  baseline (mean context median)       : {baseline:.6f}")
        print(f"  observed median range                : "
              f"{f.min():.4f} .. {f.max():.4f}")
        print(f"  max |baseline + sum(phi) - observed| : {err:.3e}")
        print(f"  max |sum(phi_ij) - (f - baseline)|   : {ierr:.3e}")
        print(summ.round(4).to_string(index=False))

    # ---------------- validation ----------------
    print("\n" + "=" * 76 + "\nVALIDATION\n" + "=" * 76)
    fails = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              f"{'  -- ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    for arm in ARMS:
        w = report[arm]["wide"]
        lg = report[arm]["long"]
        chk(f"{ARM_SLUG[arm]}: 256 contexts", len(w) == 256, f"{len(w)}")
        chk(f"{ARM_SLUG[arm]}: all 256 P/F combinations appear exactly once",
            w.context_label_8.is_unique and len(set(w.context_label_8)) == 256
            and w.context_label_8.str.fullmatch(r"[PF]{8}").all())
        chk(f"{ARM_SLUG[arm]}: 8 SHAP values per context",
            bool((lg.groupby('context_label_8').size() == 8).all()))
        chk(f"{ARM_SLUG[arm]}: no missing values",
            not lg.shap_value.isna().any() and np.isfinite(lg.shap_value).all())
        chk(f"{ARM_SLUG[arm]}: efficiency baseline + sum(phi) = observed",
            report[arm]["recon_err"] < 1e-10,
            f"max abs error {report[arm]['recon_err']:.3e}")
        chk(f"{ARM_SLUG[arm]}: interaction matrix sums to f - baseline",
            report[arm]["inter_err"] < 1e-10,
            f"max abs error {report[arm]['inter_err']:.3e}")
        it = report[arm]["inter"]
        chk(f"{ARM_SLUG[arm]}: interaction matrix symmetric",
            bool(np.allclose(it, np.swapaxes(it, 0, 1))))
    chk("analytical P/F labels used throughout (no 6/7 encoding)",
        all(report[a]["long"].block_state.isin(["P", "F"]).all() for a in ARMS)
        and all(report[a]["wide"].context_label_8.str.fullmatch(r"[PF]{8}").all()
                for a in ARMS))
    chk("no AlphaGenome calls were made", True,
        "module imports no AlphaGenome client and opens no network path")
    print("=" * 76)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")

    # ---------------- relation to the factorial analysis ----------------
    print("\n" + "=" * 76)
    print("SHAP vs factorial main effects (DIFFERENT quantities, not expected "
          "to match)\n" + "=" * 76)
    for arm in ARMS:
        slug = ARM_SLUG[arm]
        fe = pd.read_csv(FACT_DIR / f"{slug}_main_pairwise_effects.csv")
        me = fe[fe.order == 1].set_index("term").effect
        s = report[arm]["summary"].set_index("block")
        cmp_ = pd.DataFrame({
            "factorial_main_effect": me,
            "mean_abs_SHAP": s.mean_abs_SHAP,
            "mean_SHAP_when_F": s.mean_SHAP_when_F,
            "mean_SHAP_when_P": s.mean_SHAP_when_P,
            "F_minus_P_mean_SHAP": s.mean_SHAP_when_F - s.mean_SHAP_when_P,
            "SD_SHAP": s.SD_SHAP,
        }).reindex(BLOCKS)
        cmp_.to_csv(SHAP_DIR / f"{slug}_shap_vs_factorial.csv")
        print(f"\n--- {slug} ---")
        print(cmp_.round(4).to_string())

    print(f"\nmax reconstruction error across both arms: {max_err_all:.3e}")
    print(f"max interaction-efficiency error:          {max_inter_err_all:.3e}")
    print(f"\nwrote -> {SHAP_DIR.relative_to(ROOT)}")
    print("No AlphaGenome calls were made.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
