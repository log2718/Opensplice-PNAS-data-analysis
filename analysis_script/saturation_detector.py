"""
Automatic saturation detector for PNAS-pre-tuner vs AlphaGenome prod_logit
relationships.

Given one (exon, motif family) dataset it answers: does the response contain a
plateau, on which side, at what AlphaGenome level, and how confident are we?

Design, and why it is in this order
------------------------------------
The gates run in a deliberate sequence, because each later step is only
meaningful if the earlier ones passed:

1. **INSUFFICIENT_RANGE** -- fewer than ``min_levels`` distinct copy-number
   doses. With 3 doses you cannot separate "plateau + transition + plateau"
   from "three points on a line", whatever the PNAS span happens to be.
2. **AMBIGUOUS_OR_MULTIMODAL** -- a substantial secondary population exists, so
   a single median curve is not a faithful summary and a single plateau level
   would be a fiction.
3. **GLOBAL_SATURATION** -- the central curve barely moves. This MUST precede
   any relative-slope logic: on a flat curve the largest local slope is itself
   noise, so normalising by it rescales noise to look like structure.
4. Only then: local slopes and edge-anchored plateau detection.

Everything operates on ``|local slope|``, so increasing sigmoids (LDLR_e7 UAG)
and decreasing/inverted ones (LDLR_e7 GGAGGAC) use identical machinery. The
detector never assumes the transition has positive slope, and never assumes
which side is the high one.

Thresholds
----------
Defaults were calibrated against the 32 completed exon x motif datasets rather
than chosen a priori; ``calibrate`` in the runner reports the sensitivity of
every classification to each threshold. Each default sits inside a range over
which the development-case labels do not change.

Censoring
---------
``AG_prod_logit`` is hard-clipped at ``logit(1e-7) = -16.1181`` by the scoring
pipeline. A plateau sitting on that value is an artifact of clipping, not an
AlphaGenome saturation level, so it is reported as ``censored=True`` /
``OBSERVED_LOWER_BOUND`` with no confidence interval.

Noise floor
-----------
Repeat scoring of identical WT sequences differed by ~0.02-0.03 logit, so
differences below ~0.05 logit are not treated as signal anywhere here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

X_COL = "PNAS_pretuner"
Y_COL = "AG_prod_logit"

#: hard clip applied by the scoring pipeline's _logit(eps=1e-7)
CENSOR_VALUE = math.log(1e-7 / (1 - 1e-7))      # -16.1181
CENSOR_TOL = 1e-3
#: repeat-scoring reproducibility of AlphaGenome on identical input
AG_NOISE = 0.05


@dataclass
class Thresholds:
    # --- gating -----------------------------------------------------------
    min_levels: int = 4            # distinct Nx doses needed to speak of a plateau
    min_points: int = 200
    min_x_range: float = 2.0
    # --- branch / multimodality ------------------------------------------
    branch_drop: float = 3.0       # logit below local median to count as "other branch"
    branch_frac: float = 0.20      # mean per-bin fraction that forces AMBIGUOUS
    # --- global flatness --------------------------------------------------
    flat_range: float = 1.0        # smoothed central-curve span, in logit units
    # --- plateau detection ------------------------------------------------
    slope_frac: float = 0.35       # |slope| / P90(|slope|) to count as flat
    # 4 of a ~23-bin curve (~17%), consistent with min_plateau_xfrac. NOTE: this
    # is the least robust default -- 3 admits a 3-bin right "plateau" in
    # LDLR_e7/GGAGGAC that is really dose mixing, while 5 rejects the genuine
    # narrow left plateau of TP53_e7/UAG. Stable in every other dimension.
    min_plateau_bins: int = 4
    min_plateau_xfrac: float = 0.10
    # --- binning / smoothing ---------------------------------------------
    target_bin_n: int = 30         # points per quantile bin
    max_bins: int = 24
    min_bins: int = 6
    smooth_window: int = 3         # centred rolling median
    # --- supporting -------------------------------------------------------
    still_changing_delta: float = 0.30   # logit change between top two Nx levels
    max_plateau_dose_span: float = 4.0   # per-dose spread inside a plateau before warning
    n_boot: int = 2000


    @classmethod
    def for_metric(cls, metric: str = "prod_logit", **overrides) -> "Thresholds":
        """Thresholds rescaled for a different AlphaGenome y-metric.

        Only the thresholds expressed in LOGIT UNITS are rescaled; ratios and
        counts (``branch_frac``, ``slope_frac``, ``min_plateau_bins``,
        ``min_plateau_xfrac``, ``target_bin_n``) are scale-free and unchanged.

        ``mean_logit = (logit(p_acc) + logit(p_don)) / 2`` while
        ``prod_logit = logit(p_acc * p_don) ~= logit(p_acc) + logit(p_don)`` in
        the small-probability regime, so mean_logit occupies roughly half the
        scale. The factor below is MEASURED on the synthetic-context data
        (IQR ratio 0.513 endpoint / 0.518 chimera; smoothed-response-range
        ratio 0.560), not assumed.
        """
        factor = {"prod_logit": 1.0, "mean_logit": 0.55}.get(metric)
        if factor is None:
            raise ValueError(f"unknown metric {metric!r}")
        t = cls()
        for f in ("branch_drop", "flat_range", "still_changing_delta",
                  "max_plateau_dose_span"):
            setattr(t, f, round(getattr(t, f) * factor, 4))
        for k, v in overrides.items():
            setattr(t, k, v)
        return t


@dataclass
class Plateau:
    present: bool = False
    level: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    iqr: float = np.nan
    x_min: float = np.nan
    x_max: float = np.nan
    onset_x: float = np.nan
    n_points: int = 0
    x_fraction: float = np.nan
    mean_abs_slope: float = np.nan
    censored: bool = False
    nx_levels: str = ""
    nx_median: float = np.nan
    #: spread of per-dose medians INSIDE the plateau window. A real plateau
    #: should look the same at every dose it contains; a large value means the
    #: flatness is an artifact of PNAS-space bins mixing doses whose AG levels
    #: differ, not a converged response.
    dose_span: float = np.nan


@dataclass
class Result:
    exon_id: str = ""
    motif_family: str = ""
    classification: str = ""
    subtype: str = ""
    n_variants: int = 0
    n_levels: int = 0
    left: Plateau = field(default_factory=Plateau)
    right: Plateau = field(default_factory=Plateau)
    global_response_range: float = np.nan
    global_slope: float = np.nan
    branch_score: float = np.nan
    multimodal_flag: bool = False
    flatness_score: float = np.nan
    highest_nx: int = 0
    still_changing_at_highest_nx: bool = False
    confidence: str = ""
    warnings: str = ""
    # retained for plotting / inspection, not written to CSV
    curve: pd.DataFrame | None = None


# ==========================================================================
# building blocks
# ==========================================================================

def _bin_curve(df: pd.DataFrame, t: Thresholds, y_col: str = Y_COL) -> pd.DataFrame:
    """Equal-count (quantile) bins along PNAS, with a lightly smoothed median.

    Quantile rather than equal-width bins because PNAS density is very uneven
    across a motif series -- equal-width bins would leave the sparse tails with
    one or two points and dominate the slope estimate with noise.
    """
    nb = int(np.clip(len(df) // t.target_bin_n, t.min_bins, t.max_bins))
    q = pd.qcut(df[X_COL], nb, duplicates="drop")
    g = (df.groupby(q, observed=True)
           .agg(bx=(X_COL, "median"), by=(y_col, "median"),
                lo=(y_col, lambda v: v.quantile(.25)),
                hi=(y_col, lambda v: v.quantile(.75)),
                n=(y_col, "size"))
           .reset_index(drop=True))
    # Rolling median of 3: removes single-bin spikes while preserving a sharp
    # transition. A spline/LOWESS would round off exactly the sharp 4x->8x
    # transitions this experiment is designed to find.
    g["sy"] = g.by.rolling(t.smooth_window, center=True, min_periods=1).median()
    g["slope"] = np.gradient(g.sy.to_numpy(), g.bx.to_numpy())
    g["abs_slope"] = np.abs(g.slope)
    return g


def _branch_score(df: pd.DataFrame, curve_bins: pd.Series, t: Thresholds,
                  y_col: str = Y_COL) -> float:
    """Mean per-bin fraction of variants far below their own local median.

    A fixed logit drop (not a multiple of the local IQR) because when a branch
    exists it inflates the IQR, which would mask exactly what we are looking
    for. ``branch_drop = 3`` logit is ~100x the AlphaGenome repeat-scoring
    noise and ~3/4 of the full skipped-to-included AG range.
    """
    fracs = []
    for _, s in df.groupby(curve_bins, observed=True):
        med = s[y_col].median()
        fracs.append((s[y_col] < med - t.branch_drop).mean())
    return float(np.mean(fracs)) if fracs else 0.0


def _edge_run(rel: np.ndarray, thr: float, from_left: bool) -> int:
    idx = range(len(rel)) if from_left else range(len(rel) - 1, -1, -1)
    n = 0
    for i in idx:
        if rel[i] <= thr:
            n += 1
        else:
            break
    return n


def _bootstrap_median(values: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """Percentile CI for the plateau level.

    Resamples the *raw variants* inside the plateau region with replacement.
    Each variant is an independently randomised motif placement, so they are
    exchangeable within a plateau; no stratification by Nx is applied, which
    means the CI reflects both placement variability and any residual
    dose variation still present inside the plateau.
    """
    if len(values) < 5:
        return (np.nan, np.nan)
    boots = np.median(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _describe_plateau(df: pd.DataFrame, curve: pd.DataFrame, lo_i: int, hi_i: int,
                      side: str, t: Thresholds, rng, y_col: str = Y_COL) -> Plateau:
    seg = curve.iloc[lo_i:hi_i + 1]
    x_lo, x_hi = seg.bx.min(), seg.bx.max()
    total_x = curve.bx.max() - curve.bx.min()
    # level estimated from the RAW variants in the plateau x-window, not the
    # smoothed bin medians
    raw = df[(df[X_COL] >= x_lo) & (df[X_COL] <= x_hi)]
    vals = raw[y_col].to_numpy()

    n_cens = int(np.sum(np.abs(vals - CENSOR_VALUE) < CENSOR_TOL))
    censored = n_cens / max(len(vals), 1) > 0.10

    p = Plateau(
        present=True,
        level=float(np.median(vals)),
        iqr=float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        x_min=float(x_lo), x_max=float(x_hi),
        n_points=len(raw),
        x_fraction=float((x_hi - x_lo) / total_x) if total_x > 0 else np.nan,
        mean_abs_slope=float(seg.abs_slope.mean()),
        censored=censored,
    )
    if not censored:
        p.ci_low, p.ci_high = _bootstrap_median(vals, t.n_boot, rng)
    if "copy_number" in raw and len(raw):
        lv = sorted(raw.copy_number.unique())
        p.nx_levels = ",".join(f"{int(v)}x" for v in lv)
        p.nx_median = float(np.median(raw.copy_number))
        if len(lv) > 1:
            per_dose = raw.groupby("copy_number")[y_col].median()
            p.dose_span = float(per_dose.max() - per_dose.min())
    # onset = first bin beyond the plateau (where the curve starts moving)
    if side == "left" and hi_i + 1 < len(curve):
        p.onset_x = float(curve.bx.iloc[hi_i + 1])
    elif side == "right" and lo_i - 1 >= 0:
        p.onset_x = float(curve.bx.iloc[lo_i - 1])
    return p


# ==========================================================================
# main entry point
# ==========================================================================

def detect(df: pd.DataFrame, exon_id: str = "", motif_family: str = "",
           t: Thresholds | None = None, seed: int = 0,
           y_col: str = Y_COL) -> Result:
    """Classify one x-vs-y relationship.

    ``y_col`` selects the AlphaGenome metric; pair it with
    ``Thresholds.for_metric(...)`` so the logit-unit thresholds match the
    metric's scale. Defaults reproduce the original prod_logit analysis.
    """
    t = t or Thresholds()
    rng = np.random.default_rng(seed)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[X_COL, y_col])
    res_y_col = y_col
    res = Result(exon_id=exon_id, motif_family=motif_family, n_variants=len(df))
    warn: list[str] = []

    levels = sorted(df.copy_number.unique()) if "copy_number" in df else []
    res.n_levels = len(levels)
    res.highest_nx = int(max(levels)) if levels else 0

    x = df[X_COL].to_numpy()
    y = df[y_col].to_numpy()
    if len(df) >= 3:
        res.global_slope = float(np.polyfit(x, y, 1)[0])

    # --- gate 1: insufficient range --------------------------------------
    if (res.n_levels < t.min_levels or len(df) < t.min_points
            or (x.max() - x.min()) < t.min_x_range):
        res.classification = "INSUFFICIENT_RANGE"
        res.subtype = f"{res.n_levels} dose levels, n={len(df)}"
        res.confidence = "low"
        res.warnings = (f"only {res.n_levels} feasible Nx levels; cannot separate "
                        f"plateau from transition")
        return res

    curve = _bin_curve(df, t, y_col)
    res.curve = curve
    res.global_response_range = float(curve.sy.max() - curve.sy.min())
    res.flatness_score = res.global_response_range

    # --- gate 2: multimodality -------------------------------------------
    nb = len(curve)
    q = pd.qcut(df[X_COL], nb, duplicates="drop")
    res.branch_score = _branch_score(df, q, t, y_col)
    if res.branch_score >= t.branch_frac:
        res.multimodal_flag = True
        res.classification = "AMBIGUOUS_OR_MULTIMODAL"
        res.subtype = f"branch_score={res.branch_score:.3f}"
        res.confidence = "low"
        res.warnings = ("substantial secondary branch; a single median curve is "
                        "not a faithful summary and no plateau level is reported")
        return res
    if res.branch_score >= 0.6 * t.branch_frac:
        warn.append(f"moderate branching (branch_score={res.branch_score:.3f})")

    # --- still changing at the top dose? ---------------------------------
    if len(levels) >= 2:
        m_top = df[df.copy_number == levels[-1]][y_col].median()
        m_prev = df[df.copy_number == levels[-2]][y_col].median()
        res.still_changing_at_highest_nx = abs(m_top - m_prev) > t.still_changing_delta
        if res.still_changing_at_highest_nx:
            warn.append(f"still changing at {int(levels[-1])}x "
                        f"(Δmedian={m_top - m_prev:+.2f}); far plateau may be outside range")

    # --- gate 3: global flatness (BEFORE any relative-slope logic) --------
    if res.global_response_range < t.flat_range:
        res.classification = "GLOBAL_SATURATION"
        res.subtype = "ALREADY_FLAT / LOW_RESPONSE_RANGE"
        lvl = float(np.median(y))
        res.left = Plateau(present=True, level=lvl,
                           iqr=float(np.percentile(y, 75) - np.percentile(y, 25)),
                           x_min=float(x.min()), x_max=float(x.max()),
                           n_points=len(df), x_fraction=1.0,
                           mean_abs_slope=float(curve.abs_slope.mean()))
        res.left.ci_low, res.left.ci_high = _bootstrap_median(y, t.n_boot, rng)
        res.confidence = "high" if res.global_response_range < 0.5 * t.flat_range else "medium"
        warn.append(f"central curve spans only {res.global_response_range:.2f} logit "
                    f"across {x.max()-x.min():.1f} PNAS units")
        res.warnings = "; ".join(warn)
        return res

    # --- plateau detection ------------------------------------------------
    rel = curve.abs_slope.to_numpy() / max(np.percentile(curve.abs_slope, 90), 1e-9)
    nl = _edge_run(rel, t.slope_frac, from_left=True)
    nr = _edge_run(rel, t.slope_frac, from_left=False)
    total_x = curve.bx.max() - curve.bx.min()

    def _accept(n: int, lo_i: int, hi_i: int) -> bool:
        if n < t.min_plateau_bins:
            return False
        span = curve.bx.iloc[hi_i] - curve.bx.iloc[lo_i]
        return (span / total_x) >= t.min_plateau_xfrac if total_x > 0 else False

    left_ok = nl > 0 and _accept(nl, 0, nl - 1)
    right_ok = nr > 0 and _accept(nr, len(curve) - nr, len(curve) - 1)
    # a run covering the whole curve is flatness, already handled above
    if nl >= len(curve) or nr >= len(curve):
        left_ok = right_ok = False

    if left_ok:
        res.left = _describe_plateau(df, curve, 0, nl - 1, "left", t, rng, y_col)
    if right_ok:
        res.right = _describe_plateau(df, curve, len(curve) - nr, len(curve) - 1,
                                      "right", t, rng, y_col)

    res.classification = {(True, True): "BOTH_SIDES",
                          (True, False): "LEFT_SATURATION",
                          (False, True): "RIGHT_SATURATION",
                          (False, False): "NO_SATURATION_IN_RANGE"}[(left_ok, right_ok)]

    for side, p in (("left", res.left), ("right", res.right)):
        if p.present and p.censored:
            warn.append(f"{side} plateau sits on the {CENSOR_VALUE:.3f} clip: "
                        f"OBSERVED_LOWER_BOUND, not a biological saturation level")
        if p.present and np.isfinite(p.dose_span) and p.dose_span > t.max_plateau_dose_span:
            warn.append(f"{side} plateau spans {p.dose_span:.1f} logit across the doses it "
                        f"contains: likely PNAS-space dose mixing, not a converged plateau")

    n_sides = int(left_ok) + int(right_ok)
    if res.classification == "NO_SATURATION_IN_RANGE":
        res.confidence = "medium"
        warn.append("curve still responding at both edges of the sampled PNAS range")
    else:
        wide = all(p.x_fraction >= 0.15 for p in (res.left, res.right) if p.present)
        res.confidence = "high" if (n_sides and wide and not res.still_changing_at_highest_nx) else "medium"
    res.warnings = "; ".join(warn)
    return res


def to_row(res: Result) -> dict[str, Any]:
    """Flatten a Result into one CSV row."""
    row: dict[str, Any] = {
        "exon_id": res.exon_id, "motif_family": res.motif_family,
        "classification": res.classification, "subtype": res.subtype,
        "n_variants": res.n_variants, "n_levels": res.n_levels,
    }
    for side, p in (("left", res.left), ("right", res.right)):
        row.update({
            f"{side}_plateau_present": p.present,
            f"{side}_plateau_level": p.level,
            f"{side}_plateau_CI_low": p.ci_low,
            f"{side}_plateau_CI_high": p.ci_high,
            f"{side}_plateau_IQR": p.iqr,
            f"{side}_plateau_x_min": p.x_min,
            f"{side}_plateau_x_max": p.x_max,
            f"{side}_plateau_onset_x": p.onset_x,
            f"{side}_plateau_n_points": p.n_points,
            f"{side}_plateau_x_fraction": p.x_fraction,
            f"{side}_plateau_censored": p.censored,
            f"{side}_plateau_nx_levels": p.nx_levels,
            f"{side}_plateau_dose_span": p.dose_span,
        })
    row.update({
        "global_response_range": res.global_response_range,
        "global_slope": res.global_slope,
        "branch_score": res.branch_score,
        "multimodal_flag": res.multimodal_flag,
        "flatness_score": res.flatness_score,
        "highest_Nx": res.highest_nx,
        "still_changing_at_highest_Nx": res.still_changing_at_highest_nx,
        "confidence": res.confidence,
        "warnings": res.warnings,
    })
    return row
