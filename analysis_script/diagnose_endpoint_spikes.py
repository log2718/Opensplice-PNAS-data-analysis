"""
Diagnose the far-left spikes in the alpha=0.1 synthetic-exon endpoint plots.

READ-ONLY with respect to AlphaGenome: this script makes **no** AG calls. It
reads only

    outputs/synthetic_context/synthetic_exon_library.csv
    outputs/synthetic_context/synthetic_endpoint_ag_results.csv

and writes CSVs + PNGs under ``outputs/synthetic_context/spike_diagnosis/``.

Steps
-----
A1  Locate the far-left modes of ``PNAS_common`` for alpha=0.1 objectively.
    The spikes are *narrow* excesses sitting on a *broad* bulk, so a mode is
    bracketed by where a lightly-smoothed histogram rises above a heavily
    smoothed estimate of the bulk itself. Valley-bracketing and full-width-at-
    half-prominence are reported alongside as cross-checks; no range is
    hard-coded.
A2  Composition of each spike (base fractions, dominant base, Shannon entropy,
    longest homopolymer run, GC) vs non-spike alpha=0.1 and vs every alpha.
A3  Candidate splice-site dinucleotides (AG / GT) and their positions. The repo
    contains no splice-site PWM / MaxEnt / SpliceAI scorer, so nothing here is
    called a functional cryptic site -- these are *candidate motifs* only. The
    null is a per-sequence composition-preserving shuffle, which keeps base
    composition exactly and destroys arrangement.
A4  Per-spike AlphaGenome endpoint values in all three contexts, paired on
    ``sequence_id`` (``PNAS_common`` is a property of the exon, so the same IDs
    define the spikes in every context).
A5  Pick representatives for a later full-SPLICE_SITES-track diagnostic and
    write the request plan. Nothing is submitted.

Usage::

    python analysis_script/diagnose_endpoint_spikes.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.signal import find_peaks, peak_widths  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

OUT = ROOT / "outputs" / "synthetic_context"
LIB_CSV = OUT / "synthetic_exon_library.csv"
END_CSV = OUT / "synthetic_endpoint_ag_results.csv"
DIAG = OUT / "spike_diagnosis"
PLOTS = DIAG / "plots"

SPIKE_ALPHA = 0.1
BASES = "ACGT"

#: validated categorical slots (dataviz reference palette, light mode):
#: blue / orange / aqua / violet -- CVD-safe at every adjacent pair.
GROUP_COLOR = {
    "spike1": "#2a78d6",
    "spike2": "#eb6834",
    "spike3": "#1baf7a",
    "control": "#4a3aa7",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

#: histogram resolution and the two smoothing scales, in score units
BIN_W = 0.25
SPIKE_SIGMA = 0.50   # resolves a spike
BULK_SIGMA = 4.00    # estimates the bulk underneath it
MIN_PROMINENCE = 5.0

#: dominant-base fraction at or above which a 70-mer is called near-homopolymeric
HOMOPOLYMER_CUT = 0.95

CONTEXT_ORDER = ["PNAS", "TP53_e6", "TP53_e7"]
N_REPS = 5


# ==========================================================================
# sequence features
# ==========================================================================

def base_fractions(seqs: pd.Series) -> pd.DataFrame:
    """Per-sequence A/C/G/T fractions."""
    n = seqs.str.len()
    return pd.DataFrame(
        {f"f{b}": seqs.str.count(b) / n for b in BASES}, index=seqs.index
    )


def shannon_entropy(frac: pd.DataFrame) -> pd.Series:
    """Shannon entropy (bits) of the base-composition vector, 0..2."""
    f = frac.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(f > 0, f * np.log2(f), 0.0)
    return pd.Series(-terms.sum(axis=1), index=frac.index)


def longest_run(seq: str) -> int:
    """Longest single-base homopolymer run."""
    return max((len(m.group(0)) for m in re.finditer(r"(.)\1*", seq)), default=0)


def sequence_features(seqs: pd.Series) -> pd.DataFrame:
    """Every composition feature requested in step A2."""
    frac = base_fractions(seqs)
    arr = frac.to_numpy()
    out = frac.copy()
    out["dominant_base"] = np.array(list(BASES))[arr.argmax(axis=1)]
    out["dominant_fraction"] = arr.max(axis=1)
    out["entropy_bits"] = shannon_entropy(frac)
    out["longest_homopolymer"] = seqs.map(longest_run)
    out["gc_fraction"] = frac["fG"] + frac["fC"]
    out["n_AG"] = seqs.str.count("(?=AG)")
    out["n_GT"] = seqs.str.count("(?=GT)")
    #: composition-resolved class. A PNAS_common window is a proxy for this;
    #: this column is the thing itself, and the two disagree where a window
    #: overlaps the bulk.
    out["homopolymer_class"] = np.where(
        out["dominant_fraction"] >= HOMOPOLYMER_CUT,
        "poly" + out["dominant_base"], "mixed")
    return out


# ==========================================================================
# A1 -- objective spike detection
# ==========================================================================

def detect_spikes(x: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Find narrow modes of ``x`` as excesses over a smooth bulk baseline.

    Returns ``(table, curves)``: one row per detected mode with its centre and
    three independently-derived ranges, plus the curves for plotting.
    """
    edges = np.arange(np.floor(x.min()) - 1.0,
                      np.ceil(x.max()) + 1.0 + BIN_W, BIN_W)
    cent = (edges[:-1] + edges[1:]) / 2.0
    counts, _ = np.histogram(x, bins=edges)

    fine = gaussian_filter1d(counts.astype(float), SPIKE_SIGMA / BIN_W)
    bulk = gaussian_filter1d(counts.astype(float), BULK_SIGMA / BIN_W)

    peaks, props = find_peaks(fine, prominence=MIN_PROMINENCE)
    valleys, _ = find_peaks(-fine)
    widths = peak_widths(fine, peaks, rel_height=0.5)

    # contiguous runs where the spike-scale curve exceeds the bulk estimate
    excess = fine > bulk
    region = np.zeros(len(fine), dtype=int)
    cur, prev = 0, False
    for i, v in enumerate(excess):
        if v and not prev:
            cur += 1
        region[i] = cur if v else 0
        prev = v

    rows = []
    for i, p in enumerate(peaks):
        # primary: baseline-excess bracket
        if region[p]:
            idx = np.where(region == region[p])[0]
            lo, hi = cent[idx[0]] - BIN_W / 2, cent[idx[-1]] + BIN_W / 2
        else:
            lo, hi = np.nan, np.nan
        # cross-check 1: adjacent local minima
        left = [cent[v] for v in valleys if v < p]
        right = [cent[v] for v in valleys if v > p]
        v_lo = left[-1] if left else cent[0] - BIN_W / 2
        v_hi = right[0] if right else cent[-1] + BIN_W / 2
        # cross-check 2: full width at half prominence
        f_lo, f_hi = np.interp([widths[2][i], widths[3][i]],
                               np.arange(len(cent)), cent)
        rows.append({
            "center": cent[p],
            "prominence": props["prominences"][i],
            "lo": lo, "hi": hi,
            "n": int(((x >= lo) & (x < hi)).sum()) if np.isfinite(lo) else 0,
            "valley_lo": v_lo, "valley_hi": v_hi,
            "fwhm_lo": f_lo, "fwhm_hi": f_hi,
        })
    table = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    curves = {"centers": cent, "counts": counts, "fine": fine, "bulk": bulk}
    return table, curves


# ==========================================================================
# A3 -- candidate splice-site dinucleotides
# ==========================================================================

def motif_positions(seq: str, dinuc: str) -> list[int]:
    """0-based start indices of every (overlapping) ``dinuc`` occurrence."""
    return [m.start() for m in re.finditer(f"(?={dinuc})", seq)]


def candidate_motif_table(seqs: pd.Series, rng: np.random.Generator,
                          n_shuffles: int = 20) -> pd.DataFrame:
    """Candidate AG/GT counts against a composition-preserving shuffle null.

    Shuffling each sequence keeps its base composition *exactly* and randomises
    only the arrangement, so a count that matches the shuffle mean is fully
    explained by composition alone.
    """
    rows = []
    for sid, s in seqs.items():
        arr = np.frombuffer(s.encode(), dtype="S1")
        obs_ag, obs_gt = len(motif_positions(s, "AG")), len(motif_positions(s, "GT"))
        null_ag, null_gt = [], []
        for _ in range(n_shuffles):
            perm = rng.permutation(arr).tobytes().decode()
            null_ag.append(len(motif_positions(perm, "AG")))
            null_gt.append(len(motif_positions(perm, "GT")))
        rows.append({
            "sequence_id": sid,
            "n_AG": obs_ag, "n_GT": obs_gt,
            "null_AG_mean": float(np.mean(null_ag)),
            "null_GT_mean": float(np.mean(null_gt)),
            "excess_AG": obs_ag - float(np.mean(null_ag)),
            "excess_GT": obs_gt - float(np.mean(null_gt)),
        })
    return pd.DataFrame(rows).set_index("sequence_id")


def donor_windows(full_exon: str, max_report: int = 3) -> list[str]:
    """9-nt windows (-3..+6) around each internal GT -- the span a donor spans.

    Descriptive only: these are candidate dinucleotide contexts, not scores.
    """
    out = []
    for p in motif_positions(full_exon, "GT"):
        if p - 3 < 0 or p + 6 > len(full_exon):
            continue
        out.append(f"{full_exon[p - 3:p]}|{full_exon[p:p + 6]}")
        if len(out) >= max_report:
            break
    return out


def acceptor_windows(full_exon: str, max_report: int = 3) -> list[str]:
    """20 nt upstream + AG + 3 nt, with the upstream pyrimidine fraction.

    Descriptive only -- a real acceptor needs a branch point and a
    polypyrimidine tract, neither of which is scored here.
    """
    out = []
    for p in motif_positions(full_exon, "AG"):
        if p - 20 < 0 or p + 5 > len(full_exon):
            continue
        up = full_exon[p - 20:p]
        py = (up.count("C") + up.count("T")) / 20.0
        out.append(f"{up}|AG|{full_exon[p + 2:p + 5]} (py={py:.2f})")
        if len(out) >= max_report:
            break
    return out


# ==========================================================================
# plots
# ==========================================================================

def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(direction="out", colors=MUTED, labelsize=9)
    return ax


def plot_spike_ranges(x: np.ndarray, curves: dict, spikes: pd.DataFrame,
                      label_hint: list[str], path: Path) -> None:
    """alpha=0.1 PNAS_common distribution with the three spike ranges marked."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=200)
    ax.hist(x, bins=np.arange(x.min() - 1, x.max() + 1 + BIN_W, BIN_W),
            color="#c9d8ee", edgecolor="none", label="count (0.25-unit bins)")
    ax.plot(curves["centers"], curves["fine"], color=INK, lw=1.4,
            label=f"smoothed, sigma={SPIKE_SIGMA}")
    ax.plot(curves["centers"], curves["bulk"], color=MUTED, lw=1.4, ls="--",
            label=f"bulk baseline, sigma={BULK_SIGMA}")

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.18)
    for i, r in spikes.iterrows():
        c = GROUP_COLOR[f"spike{i + 1}"]
        ax.axvspan(r.lo, r.hi, color=c, alpha=0.16, lw=0)
        ax.axvline(r.center, color=c, lw=1.8)
        # sit each label just above its own peak, never at the figure top where
        # the legend lives
        peak_h = curves["fine"][np.argmin(np.abs(curves["centers"] - r.center))]
        ax.annotate(f"spike{i + 1}  centre {r.center:.1f}\n"
                    f"[{r.lo:.1f}, {r.hi:.1f})  n={int(r.n)}\n{label_hint[i]}",
                    xy=(r.center, peak_h), xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=c,
                    weight="bold", linespacing=1.35)

    ax.set_xlabel("PNAS pre-tuner score (common)  (energy_seq_struct)", fontsize=10)
    ax.set_ylabel("Sequences", fontsize=10)
    ax.set_title("Dirichlet alpha = 0.1 — three far-left modes of PNAS_common",
                 fontsize=12, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_composition(assign: pd.DataFrame, path: Path) -> None:
    """Dominant-base make-up and entropy of each group."""
    groups = ["spike1", "spike2", "spike3", "control"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=200)

    ax = axes[0]
    bottom = np.zeros(len(groups))
    for b in BASES:
        vals = [(assign[assign.group == g].dominant_base == b).mean() * 100
                for g in groups]
        ax.bar(groups, vals, bottom=bottom, label=f"{b}-dominant",
               color={"A": "#2a78d6", "C": "#eb6834",
                      "G": "#1baf7a", "T": "#4a3aa7"}[b],
               edgecolor="white", linewidth=2)
        for i, (v, bo) in enumerate(zip(vals, bottom)):
            if v > 7:
                ax.text(i, bo + v / 2, f"{b}\n{v:.0f}%", ha="center",
                        va="center", fontsize=8, color="white", weight="bold")
        bottom += vals
    ax.set_ylabel("% of group", fontsize=10)
    ax.set_title("Dominant base", fontsize=11, weight="bold", color=INK)
    ax.set_ylim(0, 100)

    for ax, col, lab in ((axes[1], "entropy_bits", "Shannon entropy (bits)"),
                         (axes[2], "longest_homopolymer",
                          "Longest homopolymer run (nt)")):
        data = [assign[assign.group == g][col].to_numpy() for g in groups]
        bp = ax.boxplot(data, tick_labels=groups, widths=0.55,
                        patch_artist=True, medianprops=dict(color=INK, lw=1.6),
                        flierprops=dict(marker=".", ms=3, mfc=MUTED,
                                        mec="none", alpha=0.5))
        for patch, g in zip(bp["boxes"], groups):
            patch.set(facecolor=GROUP_COLOR[g], alpha=0.65, edgecolor=GROUP_COLOR[g])
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color(MUTED)
        ax.set_ylabel(lab, fontsize=10)
        ax.set_title(lab, fontsize=11, weight="bold", color=INK)

    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        _despine(ax)
    fig.suptitle("Composition of the three spikes vs non-spike alpha=0.1",
                 fontsize=12.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ag_by_group(merged: pd.DataFrame, path: Path) -> None:
    """AG mean-logit per group, paired across the three contexts."""
    groups = ["spike1", "spike2", "spike3", "control"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=200, sharey=True)
    for ax, ctx in zip(axes, CONTEXT_ORDER):
        sub = merged[merged.context == ctx]
        data = [sub[sub.group == g].AG_mean_logit.dropna().to_numpy() for g in groups]
        bp = ax.boxplot(data, tick_labels=groups, widths=0.55, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.6),
                        flierprops=dict(marker=".", ms=3, mfc=MUTED,
                                        mec="none", alpha=0.5))
        for patch, g in zip(bp["boxes"], groups):
            patch.set(facecolor=GROUP_COLOR[g], alpha=0.65, edgecolor=GROUP_COLOR[g])
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color(MUTED)
        for i, d in enumerate(data, start=1):
            if len(d):
                ax.text(i, np.median(d), f" {np.median(d):.2f}", fontsize=8,
                        color=INK, va="bottom", ha="center")
        ax.set_title(ctx, fontsize=11, weight="bold", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        _despine(ax)
    axes[0].set_ylabel("AlphaGenome mean-logit\n(logit P(acc) + logit P(don)) / 2",
                       fontsize=10)
    fig.suptitle("AlphaGenome endpoint values by spike group "
                 "(same sequence IDs in all three contexts)",
                 fontsize=12.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_highlight(merged: pd.DataFrame, path: Path) -> None:
    """The endpoint scatter itself, with spike members coloured."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=200, sharex=True, sharey=True)
    for ax, ctx in zip(axes, CONTEXT_ORDER):
        sub = merged[merged.context == ctx]
        ctl = sub[sub.group == "control"]
        ax.scatter(ctl.PNAS_common, ctl.AG_mean_logit, s=5, c="#c9c8c3",
                   lw=0, alpha=0.7, label="non-spike alpha=0.1")
        for g in ("spike1", "spike2", "spike3"):
            s = sub[sub.group == g]
            ax.scatter(s.PNAS_common, s.AG_mean_logit, s=7,
                       c=GROUP_COLOR[g], lw=0, alpha=0.85, label=g)
        ax.set_title(ctx, fontsize=11, weight="bold", color=INK)
        ax.set_xlabel("PNAS pre-tuner score (common)", fontsize=9.5)
        ax.grid(color=GRID, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        _despine(ax)
    axes[0].set_ylabel("AlphaGenome mean-logit", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.8)
    fig.suptitle("alpha=0.1 endpoint scatter with spike membership marked",
                 fontsize=12.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_context_sensitivity(merged: pd.DataFrame, path: Path) -> None:
    """Per-sequence AG mean-logit across contexts, split by composition class.

    One line per class, paired on sequence_id -- this is the view that separates
    "the spike is only an x-axis effect" from "AlphaGenome also behaves oddly".
    """
    order = ["polyA", "polyC", "polyG", "polyT", "mixed"]
    colors = {"polyA": "#2a78d6", "polyC": "#eb6834", "polyG": "#1baf7a",
              "polyT": "#4a3aa7", "mixed": "#9a9892"}
    wide = merged.pivot_table(index=["sequence_id", "homopolymer_class"],
                              columns="context", values="AG_mean_logit")
    wide = wide.reset_index()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=200)
    xs = np.arange(len(CONTEXT_ORDER))
    ends: list[tuple[float, str]] = []
    for cls in order:
        sub = wide[wide.homopolymer_class == cls]
        if not len(sub):
            continue
        med = [sub[c].median() for c in CONTEXT_ORDER]
        q1 = [sub[c].quantile(0.25) for c in CONTEXT_ORDER]
        q3 = [sub[c].quantile(0.75) for c in CONTEXT_ORDER]
        ax.fill_between(xs, q1, q3, color=colors[cls], alpha=0.15, lw=0)
        ax.plot(xs, med, color=colors[cls], lw=2, marker="o", ms=7,
                label=f"{cls} (n={len(sub)})")
        ends.append((med[-1], cls))
    # fan the right-edge labels apart so near-identical lines stay readable
    ends.sort()
    span = (max(e[0] for e in ends) - min(e[0] for e in ends)) or 1.0
    min_gap = span * 0.07
    placed: list[float] = []
    for y, cls in ends:
        while placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(cls, xy=(xs[-1], y), xytext=(9, 0),
                    textcoords="offset points", fontsize=9, color=colors[cls],
                    weight="bold", va="center")
    ax.set_xticks(xs, CONTEXT_ORDER)
    ax.set_xlim(-0.25, len(xs) - 0.45)
    ax.set_ylabel("AlphaGenome mean-logit  (median, IQR band)", fontsize=10)
    ax.set_title("Same exon, three contexts", fontsize=11, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    # context effect size per class
    wide["delta_e6"] = wide["PNAS"] - wide["TP53_e6"]
    cls_present = [c for c in order if (wide.homopolymer_class == c).any()]
    data = [wide[wide.homopolymer_class == c].delta_e6.dropna() for c in cls_present]
    bp = ax2.boxplot(data, tick_labels=cls_present, widths=0.55,
                     patch_artist=True, medianprops=dict(color=INK, lw=1.6),
                     flierprops=dict(marker=".", ms=3, mfc=MUTED, mec="none",
                                     alpha=0.4))
    for patch, c in zip(bp["boxes"], cls_present):
        patch.set(facecolor=colors[c], alpha=0.65, edgecolor=colors[c])
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(MUTED)
    for i, d in enumerate(data, start=1):
        if len(d):
            ax2.text(i + 0.33, np.median(d), f"{np.median(d):.2f}",
                     fontsize=8.5, color=INK, va="center", ha="left",
                     weight="bold")
    ax2.axhline(0, color=MUTED, lw=1, ls="--")
    ax2.set_ylabel("Context effect:  mean-logit(PNAS) - mean-logit(TP53_e6)",
                   fontsize=10)
    ax2.set_title("Context effect collapses for homopolymers",
                  fontsize=11, weight="bold", color=INK)

    for a_ in (ax, ax2):
        a_.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        a_.set_axisbelow(True)
        _despine(a_)
    fig.suptitle("alpha=0.1 — AlphaGenome response by composition class "
                 "(paired on sequence_id)",
                 fontsize=12.5, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args(argv)

    DIAG.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lib = pd.read_csv(LIB_CSV)
    end = pd.read_csv(END_CSV)
    print(f"library {lib.shape} | endpoint results {end.shape}")

    # ---------------- A1 ----------------
    print("\n" + "=" * 74 + "\nA1  spike detection (alpha = 0.1)\n" + "=" * 74)
    a = lib[lib.alpha == SPIKE_ALPHA].copy()
    x = a.PNAS_common.to_numpy()
    spikes, curves = detect_spikes(x)
    spikes = spikes.head(3).reset_index(drop=True)  # three far-left modes
    for i, r in spikes.iterrows():
        print(f"  spike{i + 1}: centre {r.center:7.2f}  prominence {r.prominence:5.1f}"
              f"  range [{r.lo:7.2f}, {r.hi:7.2f})  n={int(r.n):4d}"
              f"   [valley {r.valley_lo:.1f},{r.valley_hi:.1f}]"
              f"  [FWHM {r.fwhm_lo:.1f},{r.fwhm_hi:.1f}]")

    def assign_group(v: float) -> str:
        for i, r in spikes.iterrows():
            if r.lo <= v < r.hi:
                return f"spike{i + 1}"
        return "control"

    a["group"] = a.PNAS_common.map(assign_group)
    print("\n  group sizes:", a.group.value_counts().to_dict())

    # the same IDs must define the spikes in every context, because x is a
    # property of the exon, not of the context
    piv = end.pivot_table(index="sequence_id", columns="context",
                          values="PNAS_common")
    identical = bool((piv.std(axis=1).fillna(0) < 1e-9).all())
    print(f"  PNAS_common identical across all three contexts: {identical}")
    per_ctx = {
        ctx: set(end[(end.context == ctx)
                     & (end.sequence_id.isin(a[a.group != 'control'].sequence_id))]
                 .sequence_id)
        for ctx in CONTEXT_ORDER
    }
    same_ids = len({frozenset(v) for v in per_ctx.values()}) == 1
    print(f"  identical spike sequence_id sets in all three contexts: {same_ids}")

    # ---------------- A2 ----------------
    print("\n" + "=" * 74 + "\nA2  composition\n" + "=" * 74)
    feats = sequence_features(a.synthetic_70mer)
    a = pd.concat([a.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    all_feats = sequence_features(lib.synthetic_70mer)
    lib_f = pd.concat([lib.reset_index(drop=True),
                       all_feats.reset_index(drop=True)], axis=1)

    agg = {
        "n": ("sequence_id", "count"),
        "PNAS_common_mean": ("PNAS_common", "mean"),
        "PNAS_common_sd": ("PNAS_common", "std"),
        "fA": ("fA", "mean"), "fC": ("fC", "mean"),
        "fG": ("fG", "mean"), "fT": ("fT", "mean"),
        "gc_fraction": ("gc_fraction", "mean"),
        "dominant_fraction_mean": ("dominant_fraction", "mean"),
        "entropy_bits_mean": ("entropy_bits", "mean"),
        "longest_homopolymer_mean": ("longest_homopolymer", "mean"),
        "longest_homopolymer_max": ("longest_homopolymer", "max"),
        "n_AG_mean": ("n_AG", "mean"), "n_GT_mean": ("n_GT", "mean"),
    }
    by_group = a.groupby("group").agg(**agg).reset_index()
    by_group["dominant_base_mix"] = [
        ", ".join(f"{k}:{v}" for k, v in
                  a[a.group == g].dominant_base.value_counts().items())
        for g in by_group.group
    ]
    by_group["n_unique_sequences"] = [
        a[a.group == g].synthetic_70mer.nunique() for g in by_group.group
    ]
    by_group.insert(0, "stratum", "alpha=0.1 " + by_group.group)
    by_group = by_group.drop(columns="group")

    by_alpha = lib_f.groupby("alpha").agg(**agg).reset_index()
    by_alpha["dominant_base_mix"] = [
        ", ".join(f"{k}:{v}" for k, v in
                  lib_f[lib_f.alpha == al].dominant_base.value_counts().items())
        for al in by_alpha.alpha
    ]
    by_alpha["n_unique_sequences"] = [
        lib_f[lib_f.alpha == al].synthetic_70mer.nunique() for al in by_alpha.alpha
    ]
    by_alpha.insert(0, "stratum", "alpha=" + by_alpha.alpha.astype(str) + " (all)")
    by_alpha = by_alpha.drop(columns="alpha")

    comp = pd.concat([by_group, by_alpha], ignore_index=True)
    comp_path = DIAG / "spike_composition_summary.csv"
    comp.round(4).to_csv(comp_path, index=False)
    print(comp[["stratum", "n", "fA", "fC", "fG", "fT", "entropy_bits_mean",
                "longest_homopolymer_mean", "dominant_base_mix"]]
          .round(3).to_string(index=False))

    # near-homopolymer score by dominant base -- the key explanatory table
    nh = a[a.dominant_fraction >= 0.95]
    print("\n  PNAS_common of near-homopolymers (dominant_fraction >= 0.95), "
          "alpha=0.1:")
    print(nh.groupby("dominant_base").PNAS_common
          .agg(["count", "mean", "std", "min", "max"]).round(2).to_string())

    # ---------------- A3 ----------------
    print("\n" + "=" * 74 + "\nA3  candidate splice-site dinucleotides\n" + "=" * 74)
    print("  NOTE: the repo contains no splice-site PWM / MaxEnt / SpliceAI "
          "scorer.\n  Counts below are CANDIDATE dinucleotides only; none is "
          "called functional.")
    mot = candidate_motif_table(a.set_index("sequence_id").synthetic_70mer, rng)
    a = a.merge(mot.reset_index(), on="sequence_id", suffixes=("", "_dup"))
    a = a.drop(columns=[c for c in a.columns if c.endswith("_dup")])
    mot_sum = a.groupby("group").agg(
        n=("sequence_id", "count"),
        n_AG_mean=("n_AG", "mean"), null_AG_mean=("null_AG_mean", "mean"),
        excess_AG_mean=("excess_AG", "mean"),
        n_GT_mean=("n_GT", "mean"), null_GT_mean=("null_GT_mean", "mean"),
        excess_GT_mean=("excess_GT", "mean"),
        frac_zero_AG=("n_AG", lambda s: float((s == 0).mean())),
        frac_zero_GT=("n_GT", lambda s: float((s == 0).mean())),
    ).reset_index()
    print(mot_sum.round(3).to_string(index=False))
    mot_sum.round(4).to_csv(DIAG / "spike_candidate_motifs.csv", index=False)

    # ---------------- A4 ----------------
    print("\n" + "=" * 74 + "\nA4  AlphaGenome endpoint values per spike\n" + "=" * 74)
    merged = end.merge(a[["sequence_id", "group", "dominant_base",
                          "dominant_fraction", "homopolymer_class",
                          "entropy_bits", "longest_homopolymer",
                          "n_AG", "n_GT"]],
                       on="sequence_id", how="inner")
    ag_cols = ["PNAS_common", "AG_acceptor_probability", "AG_donor_probability",
               "AG_mean_logit", "AG_prod_logit"]
    ag_sum = (merged.groupby(["group", "context"])[ag_cols]
              .agg(["median", "mean", "std", "count"]))
    ag_sum.columns = ["_".join(c) for c in ag_sum.columns]
    ag_sum = ag_sum.reset_index()
    ag_sum.round(4).to_csv(DIAG / "spike_alphagenome_summary.csv", index=False)
    show = ag_sum[["group", "context", "PNAS_common_median",
                   "AG_acceptor_probability_median", "AG_donor_probability_median",
                   "AG_mean_logit_median", "AG_mean_logit_std",
                   "AG_mean_logit_count"]]
    print(show.round(4).to_string(index=False))

    # paired across contexts: does spike membership survive the context change?
    wide = merged.pivot_table(index=["sequence_id", "group"], columns="context",
                              values="AG_mean_logit").reset_index()
    print("\n  paired per-sequence AG mean-logit, median by group:")
    print(wide.groupby("group")[CONTEXT_ORDER].median().round(3).to_string())

    # ---- A4b: the composition-resolved view -------------------------------
    # A PNAS_common window is only a proxy for "homopolymer". Where the window
    # overlaps the bulk it is impure, so report the classes directly too.
    print("\n  purity of each window (near-homopolymer = dominant_fraction "
          f">= {HOMOPOLYMER_CUT}):")
    for g in ["spike1", "spike2", "spike3"]:
        s_ = a[a.group == g]
        if not len(s_):
            continue
        print(f"    {g}: n={len(s_):4d}  near-homopolymer "
              f"{int((s_.homopolymer_class != 'mixed').sum()):4d} "
              f"({(s_.homopolymer_class != 'mixed').mean() * 100:3.0f}%)   "
              + ", ".join(f"{k}:{v}" for k, v in
                          s_.homopolymer_class.value_counts().items()))
    print("\n  where each homopolymer class lands (window x class):")
    print(pd.crosstab(a.homopolymer_class, a.group).to_string())

    cls = (merged.groupby(["homopolymer_class", "context"])
           .agg(n=("AG_mean_logit", "count"),
                PNAS_common_median=("PNAS_common", "median"),
                acceptor_median=("AG_acceptor_probability", "median"),
                donor_median=("AG_donor_probability", "median"),
                AG_mean_logit_median=("AG_mean_logit", "median"),
                AG_mean_logit_sd=("AG_mean_logit", "std"))
           .reset_index())
    print("\n  AlphaGenome by composition class:")
    print(cls.round(4).to_string(index=False))
    cls.round(4).to_csv(DIAG / "spike_alphagenome_by_class.csv", index=False)

    wide_cls = merged.pivot_table(index=["sequence_id", "homopolymer_class"],
                                  columns="context",
                                  values="AG_mean_logit").reset_index()
    wide_cls["delta_PNAS_minus_e6"] = wide_cls["PNAS"] - wide_cls["TP53_e6"]
    wide_cls["delta_PNAS_minus_e7"] = wide_cls["PNAS"] - wide_cls["TP53_e7"]
    ctx_eff = (wide_cls.groupby("homopolymer_class")
               [["delta_PNAS_minus_e6", "delta_PNAS_minus_e7"]]
               .agg(["median", "std", "count"]))
    ctx_eff.columns = ["_".join(c) for c in ctx_eff.columns]
    print("\n  context effect (positive = higher in PNAS than in FAS/TP53_e6):")
    print(ctx_eff.round(3).to_string())
    ctx_eff.round(4).to_csv(DIAG / "spike_context_effect_by_class.csv")

    # ---------------- A5 ----------------
    print("\n" + "=" * 74 + "\nA5  representative selection (NO AG CALLS MADE)\n" + "=" * 74)
    reps = []
    for g in ["spike1", "spike2", "spike3", "control"]:
        sub = a[a.group == g].sort_values("PNAS_common")
        if not len(sub):
            continue
        # span the internal variation: evenly spaced quantiles of PNAS_common
        qs = np.linspace(0.05, 0.95, N_REPS)
        idx = np.unique((qs * (len(sub) - 1)).round().astype(int))
        pick = sub.iloc[idx].copy()
        pick["role"] = g
        reps.append(pick)
    reps = pd.concat(reps, ignore_index=True)
    rep_cols = ["sequence_id", "role", "group", "alpha", "PNAS_common",
                "dominant_base", "dominant_fraction", "homopolymer_class",
                "entropy_bits",
                "longest_homopolymer", "n_AG", "n_GT", "synthetic_70mer"]
    reps[rep_cols].to_csv(DIAG / "fulltrack_representatives.csv", index=False)
    n_calls = len(reps) * len(CONTEXT_ORDER)
    print(f"  {len(reps)} representatives x {len(CONTEXT_ORDER)} contexts "
          f"= {n_calls} AlphaGenome calls (NOT submitted)")
    for g in reps.role.unique():
        s = reps[reps.role == g]
        print(f"   {g:8s}: " + ", ".join(s.sequence_id))

    plan = {
        "status": "PREPARED_NOT_RUN",
        "n_representatives": int(len(reps)),
        "contexts": CONTEXT_ORDER,
        "n_alphagenome_calls": int(n_calls),
        "requested_output": "SPLICE_SITES (full track, not only intended positions)",
        "sequence_ids": reps.sequence_id.tolist(),
        "runner": "analysis_script/run_spike_fulltrack_diagnostic.py",
    }
    (DIAG / "fulltrack_request_plan.json").write_text(json.dumps(plan, indent=2))

    # ---------------- outputs ----------------
    assign_cols = ["sequence_id", "alpha", "PNAS_common", "group", "fA", "fC",
                   "fG", "fT", "dominant_base", "dominant_fraction",
                   "homopolymer_class", "entropy_bits",
                   "longest_homopolymer", "gc_fraction",
                   "n_AG", "n_GT", "excess_AG", "excess_GT", "synthetic_70mer"]
    a[assign_cols].to_csv(DIAG / "spike_assignments.csv", index=False)

    spikes.round(4).to_csv(DIAG / "spike_ranges.csv", index=False)

    hints = []
    for i in range(len(spikes)):
        g = a[a.group == f"spike{i + 1}"]
        top = g[g.homopolymer_class != "mixed"].homopolymer_class.value_counts()
        pure = (g.homopolymer_class != "mixed").mean() * 100
        hints.append(f"{' + '.join(top.index[:2])}  ({pure:.0f}% near-homopolymer)"
                     if len(top) else "mixed")
    plot_spike_ranges(x, curves, spikes, hints,
                      PLOTS / "alpha01_pnas_spike_ranges.png")
    plot_context_sensitivity(merged, PLOTS / "spike_context_sensitivity.png")
    plot_composition(a, PLOTS / "spike_composition.png")
    plot_ag_by_group(merged, PLOTS / "spike_alphagenome_by_context.png")
    plot_scatter_highlight(merged, PLOTS / "spike_endpoint_scatter.png")

    print("\n" + "=" * 74)
    print(f"wrote -> {DIAG}")
    for p in sorted(DIAG.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
