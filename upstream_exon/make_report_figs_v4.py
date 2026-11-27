#!/usr/bin/env python
"""Report figures for the v4 (full-plasmid / SPLICE_SITE_USAGE) experiments.

Everything here reads files already committed to the repo -- the two cryptic
scans and the two FNBM interpretation dumps -- so it runs on a laptop with no
cluster access and no AlphaGenome calls:

    outputs/upstream_exon/fullctx_{upstream_1m,middle_500k}/cryptic_fullctx_10000.csv
    outputs/upstream_exon/fnbm_interp_{upstream_b1_v4,middle_syn70_v4}/

Usage:
    python upstream_exon/make_report_figs_v4.py

Writes PNG + PDF into outputs/upstream_exon/report_figs_v4/.

COLOR: slots 1 and 2 of the validated categorical palette, blue #2a78d6 for the
UPSTREAM library and orange #eb6834 for the MIDDLE one, used consistently in
every panel so the two libraries are the same color throughout the deck. The
same pair doubles as the diverging pair where a figure encodes sign (E1), which
is legitimate because no figure uses both encodings at once.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "upstream_exon" / "report_figs_v4"

# --- design tokens --------------------------------------------------------
UP, MID = "#2a78d6", "#eb6834"          # categorical slots 1, 2
NEG, POS = "#2a78d6", "#eb6834"          # diverging poles (E1)
INK, INK2, INK3 = "#1a1a19", "#5c5b55", "#8a8980"
GRID = "#e3e2dd"
DEAD, UNINF, LIVE = "#cfcec8", "#9ab8dc", "#2a78d6"   # ordered quality scale

# Sequential ramp for 2-D density panels: ONE hue, light -> dark, surface at
# the zero end so empty bins read as background. Counts are a magnitude, so a
# single-hue sequential scale is the correct job here -- not the categorical
# library hues, which stay for everything that encodes identity.
BLUES = mpl.colors.LinearSegmentedColormap.from_list(
    "report_blues",
    ["#f4f7fc", "#c9dbf2", "#8fb8e6", "#4f93dc", "#2a78d6", "#14407a"])
BLUES.set_under((0, 0, 0, 0))           # empty bins -> transparent


def dens2d(ax, x, y, xbins, ybins, vmax=None):
    """2-D histogram with a log count scale; returns the QuadMesh."""
    h, xe, ye = np.histogram2d(x, y, bins=[xbins, ybins])
    m = ax.pcolormesh(xe, ye, h.T, cmap=BLUES, rasterized=True,
                      norm=mpl.colors.LogNorm(vmin=1, vmax=vmax or max(h.max(), 2)))
    return m

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.edgecolor": INK3,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
})

LIBS = [
    ("upstream", "fullctx_upstream_1m", "upstream_b1_v4",
     "Upstream exon body (B1, 113 nt)", UP),
    ("middle", "fullctx_middle_500k", "middle_syn70_v4",
     "Middle exon body (syn70, 70 nt)", MID),
]

# measured R2 / filter counts for the sweep figure (A2). Source: test_metrics.csv
# on the cluster, reproduced in the session log. target sd 0.0460 (standardized).
SWEEP = [
    ("baseline\n1e-4, 64",   0.3647, 15,  64),
    ("_c5\n1e-5, 64",        0.3770,  5,  64),
    ("_f128\n1e-4, 128",     0.3881, 39, 128),
    ("_c5f128\n1e-5, 128",   0.3934,  6, 128),
]
SWEEP_SEED1 = 0.3633   # baseline at seed 1 -- the noise yardstick


def grid(ax, axis="y"):
    ax.grid(axis=axis, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def note(ax, text, y=0.97, x=0.03, color=None, **kw):
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.6, color=color or INK2, linespacing=1.45, **kw)


def load_cryptic(d: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "upstream_exon" / d /
                       "cryptic_fullctx_10000.csv")


def load_interp(d: str):
    base = ROOT / "outputs" / "upstream_exon" / f"fnbm_interp_{d}"
    return (pd.read_csv(base / "metrics.csv"),
            pd.read_csv(base / "dedup" / "cluster_effects.csv"),
            json.load(open(base / "dedup" / "summary.json"))["diagnostics"])


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# =========================================================================
# C1  the dissociation figure: cryptic motif strength vs usage at that site
# =========================================================================
def fig_c1():
    FLOOR = 1e-7
    C1_VMAX = 400
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 4.1), sharex=True, sharey=True)
    meshes = []
    for ax, (key, cdir, _, title, color) in zip(axes, LIBS):
        d = load_cryptic(cdir)
        x = d["cryptic_donor_inslot_max"].to_numpy()
        y = np.clip(d["cryptic_donor_inslot_usage_at_argmax"].to_numpy(), FLOOR, None)
        ax.set_yscale("log")
        mesh = dens2d(ax, x, y,
                      np.linspace(0.0, 1.0, 61),
                      np.logspace(np.log10(FLOOR), np.log10(1.0), 61),
                      vmax=C1_VMAX)
        meshes.append(mesh)
        ax.axvline(0.5, color=INK3, lw=0.9, ls="--", zorder=1)
        ax.axhline(0.05, color=INK3, lw=0.9, ls="--", zorder=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(FLOOR / 2, 2.0)
        strong = x > 0.5
        used = y > 0.05
        ax.set_title(title)
        ax.set_xlabel("cryptic donor MOTIF score  (splice_sites head)")
        grid(ax, "both")
        # quadrant readout
        note(ax, f"motif > 0.5:  {100 * strong.mean():.1f}%\n"
                 f"usage > 0.05: {100 * used.mean():.1f}%\n"
                 f"both:         {100 * (strong & used).mean():.1f}%",
             y=0.985, color=INK, family="monospace")
        ax.text(0.97, 0.03, f"median usage\nat motif argmax\n"
                            f"{np.median(y):.0e}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.4, color=INK2, linespacing=1.4)
    axes[0].set_ylabel("USAGE at that same position  (splice_site_usage head)")
    cb = fig.colorbar(meshes[0], ax=axes, fraction=0.035, pad=0.02,
                      extend="max")
    cb.set_label("sequences per bin", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=INK3, labelcolor=INK2)
    fig.suptitle("Strong cryptic donor motifs are common; AlphaGenome declines "
                 "to splice at them—except in the middle exon",
                 fontsize=11, y=1.005, color=INK, ha="center")
    fig.text(0.5, -0.045,
             "2-D density, 60 x 60 bins, log count scale; n = 10,000 permuted sequences per library. "
             "Dashed lines: motif 0.5 and usage 0.05.\nSame motif prevalence in both libraries "
             "(38.1% vs 35.8%), opposite usage outcome — the motif/usage dissociation.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "C1_motif_vs_usage_dissociation")


# =========================================================================
# C2  the same two quantities as marginal distributions
# =========================================================================
def fig_c2():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    # left: motif score
    bins = np.linspace(0, 1, 51)
    for key, cdir, _, title, color in LIBS:
        d = load_cryptic(cdir)
        axes[0].hist(d["cryptic_donor_inslot_max"], bins=bins, histtype="step",
                     lw=1.8, color=color, label=title.split(" (")[0])
    axes[0].axvline(0.5, color=INK3, lw=0.9, ls="--")
    axes[0].set_xlabel("cryptic donor motif score")
    axes[0].set_ylabel("sequences")
    axes[0].set_title("Motif strength: comparable prevalence\n(38.1% vs 35.8% above 0.5)")
    axes[0].legend(loc="upper left")
    grid(axes[0])

    # right: usage at argmax, log x
    lb = np.logspace(-7, 0, 57)
    for key, cdir, _, title, color in LIBS:
        d = load_cryptic(cdir)
        v = np.clip(d["cryptic_donor_inslot_usage_at_argmax"], 1e-7, None)
        axes[1].hist(v, bins=lb, histtype="step", lw=1.8, color=color,
                     label=title.split(" (")[0])
    axes[1].axvline(0.05, color=INK3, lw=0.9, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("usage at the cryptic motif argmax")
    axes[1].set_title("Usage: they diverge by orders\nof magnitude")
    axes[1].legend(loc="upper left")
    grid(axes[1])
    fig.text(0.5, -0.19,
             "Left: near-identical motif distributions. Right: upstream usage piles up at the "
             "floor (median 1.1e-04, p95 0.013);\nmiddle reaches p95 0.967. The model recognizes "
             "the same motifs and acts on them only in the middle exon.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "C2_motif_and_usage_marginals")


# =========================================================================
# C3  variance retained after dropping strong-motif sequences
# =========================================================================
def fig_c3():
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    labels, vals, colors, ann = [], [], [], []
    for key, cdir, _, title, color in LIBS:
        d = load_cryptic(cdir)
        strong = d["cryptic_donor_inslot_max"] > 0.5
        v_all = d["logit_average"].var(ddof=0)
        v_keep = d.loc[~strong, "logit_average"].var(ddof=0)
        labels.append(title.split(" (")[0])
        vals.append(100 * v_keep / v_all)
        colors.append(color)
        ann.append(f"n dropped\n{int(strong.sum()):,}")
    xs = np.arange(len(vals))
    ax.bar(xs, vals, width=0.5, color=colors, zorder=2)
    ax.axhline(100, color=INK3, lw=1.0, ls="--", zorder=3)
    ax.text(1.46, 100, "  no change", va="center", fontsize=7.6, color=INK2)
    for x, v, a in zip(xs, vals, ann):
        ax.text(x, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9.5,
                color=INK, fontweight="semibold")
        ax.text(x, 6, a, ha="center", fontsize=7.4, color="white",
                linespacing=1.4)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 128)
    ax.set_ylabel("target variance retained (%)")
    ax.set_title("Dropping every strong-cryptic-motif sequence\nloses no target variance")
    grid(ax)
    fig.text(0.5, -0.21,
             "Variance of logit_average after removing all sequences with a cryptic donor motif > 0.5.\n"
             "Above 100% because the dropped subset was slightly below average variance — so these\n"
             "sequences were never filtered out of training, and the v3 cassette check agrees (101%).",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "C3_variance_retained")


# =========================================================================
# B1  the two-regime target distribution
# =========================================================================
def fig_b1():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for ax, (key, cdir, _, title, color) in zip(axes, LIBS):
        d = load_cryptic(cdir)
        la = d["logit_average"]
        ax.hist(la, bins=60, color=color, alpha=0.85, lw=0, zorder=2)
        ax.set_title(title)
        ax.set_xlabel("logit_average")
        grid(ax)
        note(ax, f"sd      {la.std(ddof=0):.4f}\n"
                 f"range   [{la.min():+.2f}, {la.max():+.2f}]\n"
                 f"usage   [{d.usage_acceptor.min():.3f}, {d.usage_acceptor.max():.3f}] acc\n"
                 f"        [{d.usage_donor.min():.3f}, {d.usage_donor.max():.3f}] don",
             family="monospace", color=INK)
    axes[0].set_ylabel("sequences")
    fig.suptitle("Two regimes: the upstream library sits at saturation, "
                 "the middle library spans the full range",
                 fontsize=11, y=1.01, color=INK)
    fig.text(0.5, -0.18,
             "Target spread differs 65× in sd (0.045 vs 2.93) and ~4,200× in variance. This is why the target "
             "is standardized\nbefore training — Arush's L2 penalties are absolute and were tuned at "
             "var ≈ 0.414 — and why R² is not comparable\nacross libraries without saying so.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "B1_target_two_regimes")


# =========================================================================
# D1  PSI vs the target
# =========================================================================
def fig_d1():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax, (key, cdir, _, title, color) in zip(axes, LIBS):
        d = load_cryptic(cdir)
        r = d["logit_average"].corr(d["psi"])
        mesh = dens2d(ax, d["psi"].to_numpy(), d["logit_average"].to_numpy(),
                      np.linspace(0.0, 1.0, 61), 61)
        cb = ax.figure.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
        if ax is axes[-1]:
            cb.set_label("sequences per bin", fontsize=7.4, color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, color=INK3, labelcolor=INK2)
        ax.set_title(title)
        ax.set_xlabel("junction PSI")
        # same 0-1 scale in both panels: the upstream library collapsing to a
        # single stripe at PSI = 1 IS the finding, and an auto-zoom hides it
        ax.set_xlim(-0.03, 1.03)
        grid(ax, "both")
        ax.text(0.03, 0.97, f"r = {r:+.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, color=INK,
                fontweight="semibold")
        note(ax, f"PSI median {d.psi.median():.4f}\n"
                 f"    p05–p95 {d.psi.quantile(.05):.4f}–{d.psi.quantile(.95):.4f}",
             y=0.86, family="monospace")
    axes[0].set_ylabel("logit_average  (the trained target)")
    fig.suptitle("Junction PSI is unusable in the upstream library and "
                 "excellent in the middle one",
                 fontsize=11, y=1.005, color=INK)
    fig.text(0.5, -0.14,
             "2-D density, 60 bins in PSI x 60 in the target, log count scale, each panel on its own target range.\n"
             "Upstream PSI is saturated — median 1.0000, p05 0.9986 — so it has almost no dynamic range, and what little "
             "it has\ncorrelates with the WRONG sign. Middle PSI spans 0.02–1.00 and tracks the target at r = +0.911.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "D1_psi_vs_target")


# =========================================================================
# D2  why: the inclusion-junction asymmetry
# =========================================================================
def fig_d2():
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bins = np.linspace(-1, 1, 81)
    for key, cdir, _, title, color in LIBS:
        d = load_cryptic(cdir)
        ax.hist(d["junc_incl_asym"], bins=bins, histtype="step", lw=1.9,
                color=color, label=f"{title.split(' (')[0]}  "
                                   f"(median {d.junc_incl_asym.median():+.3f})",
                zorder=2)
    ax.axvline(0, color=INK3, lw=0.9, ls="--", zorder=1)
    ax.set_xlabel("junc_incl_asym   =   (I₅′ − I₃′) / (I₅′ + I₃′)")
    ax.set_ylabel("sequences")
    ax.set_title("The upstream library suppresses its 5′ inclusion junction")
    ax.legend(loc="upper left")
    grid(ax)
    fig.text(0.5, -0.22,
             "Both libraries have IDENTICAL geometry (same 128-nt intron), so intron length cannot explain this.\n"
             "B1 sits immediately 5′ of the I₅′ donor, and permuting it collapses that junction; the middle library\n"
             "leaves B1 native and stays symmetric. A collapsed I₅′ is what breaks the PSI denominator in D1.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "D2_junction_asymmetry")


# =========================================================================
# A1  training curves -- the underfitting signature
# =========================================================================
def fig_a1():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))
    for ax, (key, _, idir, title, color) in zip(axes, LIBS):
        m, _, _ = load_interp(idir)
        ax.plot(m.epoch, m.train_loss, lw=1.9, color=color, label="train")
        ax.plot(m.epoch, m.val_loss, lw=1.6, color="#000000", label="validation")
        best = int(m.val_loss.idxmin())
        ax.scatter([m.epoch[best]], [m.val_loss[best]], s=34, facecolor="white",
                   edgecolor=INK, lw=1.4, zorder=5)
        gap = m.train_loss.iloc[-1] - m.val_loss.iloc[-1]
        ax.set_title(f"{title}\nbest epoch {int(m.epoch[best])}  ·  val "
                     f"{m.val_loss.min():.4f}  ·  train−val {gap:+.4f}",
                     fontsize=9, pad=9)
        ax.set_xlabel("epoch")
        # zoom past the epoch-0 drop, otherwise the train/val gap is invisible
        hi = max(m.train_loss.iloc[4], m.val_loss.iloc[4])
        lo = min(m.train_loss.min(), m.val_loss.min())
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.10 * (hi - lo))
        ax.legend(loc="upper right")
        grid(ax)
    axes[0].set_ylabel("MSE (standardized target units)")
    fig.suptitle("Validation loss sits BELOW training loss: the model underfits",
                 fontsize=11, y=1.08, color=INK)
    fig.text(0.5, -0.15,
             "Training loss ABOVE validation loss is the signature of a model that cannot fit even its training "
             "set, so the\nlimit is capacity and regularization, not data. Best epoch 97–98 is the bottom of the final "
             "cosine restart cycle (period 11 → 11, 22 … 99),\nnot a truncated run: another 100 epochs projects "
             "roughly +0.003 R².",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "A1_training_curves_underfit")


# =========================================================================
# A2  the 2x2 sweep
# =========================================================================
def fig_a2():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    xs = np.arange(len(SWEEP))
    r2 = [s[1] for s in SWEEP]
    bars = ax.bar(xs, r2, width=0.56, color=[UP] * 3 + [MID], zorder=2)
    bars[3].set_color(MID)
    for x, v in zip(xs, r2):
        ax.text(x, v + 0.004, f"{v:.4f}", ha="center", fontsize=9,
                color=INK, fontweight="semibold")
    # the seed-noise yardstick
    ax.plot([-0.42, -0.42], [SWEEP_SEED1, SWEEP[0][1]], color=INK, lw=2.4,
            solid_capstyle="butt", zorder=4)
    ax.annotate(f"seed spread\n{abs(SWEEP[0][1] - SWEEP_SEED1):.4f}",
                xy=(-0.42, (SWEEP_SEED1 + SWEEP[0][1]) / 2),
                xytext=(0.02, 0.22), textcoords="axes fraction",
                fontsize=7.6, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=INK3))
    ax.set_xticks(xs, [s[0] for s in SWEEP])
    ax.set_ylim(0.34, 0.405)
    ax.set_ylabel("test R²  (upstream library)")
    ax.set_title("Loosening the conv penalty and doubling\nfilters gains +0.029 R²")
    grid(ax)

    # right: dead-filter fraction
    dead_frac = [100 * s[2] / s[3] for s in SWEEP]
    ax2.bar(xs, dead_frac, width=0.56, color=[DEAD] * len(xs), zorder=2,
            edgecolor=INK3, lw=0.7)
    for x, s, f in zip(xs, SWEEP, dead_frac):
        ax2.text(x, f + 0.9, f"{s[2]}/{s[3]}", ha="center", fontsize=8.4,
                 color=INK)
    ax2.set_xticks(xs, [s[0] for s in SWEEP])
    ax2.set_ylim(0, 36)
    ax2.set_ylabel("dead conv filters (%)")
    ax2.set_title("Capacity alone kills filters;\nrelaxing the penalty rescues them")
    grid(ax2)
    fig.text(0.5, -0.20,
             "conv_kernel_l2_penalty × filter count, a full 2×2. Penalty relief alone fixes filter death but barely "
             "moves R²;\ncapacity alone raises R² and triples dead filters; together you get both. The +0.029 gain is "
             "~20× the seed-to-seed\nspread (0.3647 vs 0.3633), so it is a real effect — but it is only a tenth of the "
             "0.709 → 0.365 drop it was meant to explain.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "A2_regularization_sweep")


# =========================================================================
# E1  ranked motif effects
# =========================================================================
def fig_e1():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharex=True)
    for ax, (key, _, idir, title, color) in zip(axes, LIBS):
        _, ce, _ = load_interp(idir)
        ce = ce.reindex(ce.realized_effect.abs().sort_values().index)
        ys = np.arange(len(ce))
        # one hue per LIBRARY, matching every other figure in the deck. Sign is
        # already encoded by which side of zero the bar falls on, so colouring
        # by sign would collide with the library mapping for no information.
        ax.barh(ys, ce.realized_effect, height=0.68, color=color, zorder=2)
        ax.axvline(0, color=INK3, lw=0.8)
        ax.set_yticks(ys, ce.label, fontsize=6.8)
        ax.set_title(f"{title}\n{len(ce)} motif clusters")
        ax.set_xlabel("realized effect on the standardized target\n"
                      "(left = lowers the target, right = raises it)")
        grid(ax, "x")
    fig.suptitle("Motif clusters ranked by realized effect", fontsize=11,
                 y=1.0, color=INK)
    fig.text(0.5, -0.11,
             "Deduplicated filter clusters from Arush's interpretation hook, sorted by |realized effect|. "
             "Note the scale: the largest\nsingle motif moves the standardized target by 0.062 (middle) and "
             "0.026 (upstream), where sd = 1. No single motif\ncarries the model — R² comes from many small "
             "contributions plus the positional and tuner components.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "E1_ranked_motif_effects")


# =========================================================================
# E2  filter accounting -- the real interpretability bottleneck
# =========================================================================
def fig_e2():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    rows = []
    for key, _, idir, title, _c in LIBS:
        _, _, dg = load_interp(idir)
        rows.append((title.split(" (")[0], dg["n_dead"], dg["n_uninformative"],
                     dg["n_filters"] - dg["n_dead"] - dg["n_uninformative"],
                     dg["n_filters"], dg["n_clusters"]))
    ys = np.arange(len(rows))
    h = 0.42
    for i, (lbl, dead, uninf, live, tot, ncl) in enumerate(rows):
        left = 0.0
        for val, col, name in ((dead, DEAD, "dead"),
                               (uninf, UNINF, "uninformative"),
                               (live, LIVE, "clustered into motifs")):
            ax.barh(i, val, left=left, height=h, color=col, zorder=2,
                    edgecolor="white", lw=2.0,
                    label=name if i == 0 else None)
            ax.text(left + val / 2, i, str(val), ha="center", va="center",
                    fontsize=9, color="white" if col == LIVE else INK,
                    fontweight="semibold")
            left += val
        ax.text(tot + 1.2, i, f"{ncl} clusters", va="center", fontsize=8,
                color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 78)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("conv filters (of 64)")
    ax.set_title("Most filters are not interpretable — and 'dead' is the smaller problem")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3)
    grid(ax, "x")
    fig.text(0.5, -0.40,
             "'Dead' = near-zero weight norm. 'Uninformative' = fails the hook's ic_min = 2.0 bits test. "
             "Uninformative filters\noutnumber dead ones nearly 2:1, so relaxing the L2 penalty — which revives dead "
             "filters — is not guaranteed to\nbuy readable motifs. The number to watch in the sweep is "
             "n_uninformative, not the weight norms.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "E2_filter_accounting")




# =========================================================================
# E3  learned effect vs realized effect -- the dissociation inside the model
# =========================================================================
def fig_e3():
    """effect (model coefficient) vs realized_effect (mean order-1
    contribution on real sequences, = z1_per_cluster.mean(axis=0) in Arush's
    hook). A motif can be strongly learned and still contribute nothing."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    for ax, (key, _, idir, title, color) in zip(axes, LIBS):
        _, ce, _ = load_interp(idir)
        x, y = ce.effect.abs(), ce.realized_effect.abs()
        ax.scatter(x, y, s=46, c=color, alpha=0.75, lw=0, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("|effect|   (learned coefficient)")
        grid(ax, "both")
        rho = x.rank().corr(y.rank(), method="pearson")
        ax.text(0.03, 0.96, f"Spearman ρ = {rho:+.2f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK,
                fontweight="semibold")
        if key == "upstream":
            r = ce.loc[ce.label == "motif_16"].iloc[0]
            ax.scatter([abs(r.effect)], [abs(r.realized_effect)], s=150,
                       facecolor="none", edgecolor=INK, lw=1.8, zorder=5)
            ax.annotate("motif_16 = GUAAGU\nthe 5′ splice-site consensus\n"
                        "rank 1 of 21 by effect\nrank 21 of 21 by realized",
                        xy=(abs(r.effect), abs(r.realized_effect)),
                        xytext=(0.30, 0.16), textcoords="axes fraction",
                        fontsize=7.8, color=INK, linespacing=1.5,
                        arrowprops=dict(arrowstyle="->", lw=1.0, color=INK3))
    axes[0].set_ylabel("|realized effect|   (mean contribution on real data)")
    fig.suptitle("A motif can be strongly learned and still do nothing",
                 fontsize=11, y=1.005, color=INK)
    fig.text(0.5, -0.19,
             "'effect' is the model's learned coefficient for the motif cluster; 'realized effect' is its "
             "first-order contribution\naveraged over real sequences (z1_per_cluster.mean, per Arush's hook). "
             "In the upstream library these barely agree,\nand the extreme case is the donor consensus itself: "
             "the single best-learned filter in the model, contributing less\nthan any other cluster. This is "
             "the motif/usage dissociation of figure C1, seen inside the model's parameters.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "E3_learned_vs_realized_effect")




# =========================================================================
# F1  predicted vs actual on the held-out test split, as 2-D density
#
# Needs per-sequence predictions, which the training run writes on the
# cluster and which are NOT part of the committed interpretation dump. Drop
#     outputs/upstream_exon/models/fnbm_o1_b1_v4_c5f128/predictions.csv
#     outputs/upstream_exon/models/fnbm_o1_syn70_v4/predictions.csv
# (columns split,y_true,y_pred, in LOGIT units) and this renders. Missing
# files are skipped with a message rather than crashing the whole run.
# =========================================================================
PRED_RUNS = [
    ("models/fnbm_o1_b1_v4_c5f128/predictions.csv",
     "Upstream exon body (B1, 113 nt)", 0.3934, "best run: _c5f128"),
    ("models/fnbm_o1_syn70_v4/predictions.csv",
     "Middle exon body (syn70, 70 nt)", 0.8326, "baseline config"),
]


def fig_f1():
    found = []
    for rel, title, r2_run, tag in PRED_RUNS:
        f = ROOT / "outputs" / "upstream_exon" / rel
        if f.exists():
            found.append((pd.read_csv(f), title, r2_run, tag))
        else:
            print(f"  SKIP F1: {f} not found")
    if not found:
        print("  F1 not drawn -- no predictions.csv available")
        return

    fig, axes = plt.subplots(1, len(found), figsize=(5.2 * len(found), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (d, title, r2_run, tag) in zip(axes, found):
        d = d[d["split"] == "test"] if "split" in d.columns else d
        y = d["y_true"].to_numpy(dtype=float)
        p_ = d["y_pred"].to_numpy(dtype=float)
        mse = float(((p_ - y) ** 2).mean())
        r2 = 1.0 - mse / y.var()
        r = float(np.corrcoef(y, p_)[0, 1])

        # one shared square frame: a predicted-vs-actual panel is only honest
        # on equal axes, because the identity line must be the 45 degrees the
        # eye expects. Padded 3% so edge bins are not clipped.
        lo, hi = float(min(y.min(), p_.min())), float(max(y.max(), p_.max()))
        pad = 0.03 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        bins = np.linspace(lo, hi, 71)

        mesh = dens2d(ax, y, p_, bins, bins)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, ls="--", zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{tag}", fontsize=9.5, pad=8)
        ax.set_xlabel("ACTUAL logit_average  (AlphaGenome)")
        grid(ax, "both")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("sequences per bin", fontsize=7.4, color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, color=INK3, labelcolor=INK2)
        ax.text(0.04, 0.96, f"R\u00b2 = {r2:+.4f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=11, color=INK,
                fontweight="semibold")
        note(ax, f"r    = {r:+.4f}\n"
                 f"MSE  = {mse:.4g}\n"
                 f"n    = {len(d):,}\n"
                 f"sd(y)= {y.std():.4g}",
             y=0.885, family="monospace")
        if abs(r2 - r2_run) > 5e-3:
            note(ax, f"!! R\u00b2 here differs from the run's reported "
                     f"{r2_run:+.4f}\n   -- wrong checkpoint, split or units?",
                 y=0.13, color="#b3391c", family="monospace")
    axes[0].set_ylabel("PREDICTED logit_average  (FNBM)")
    fig.suptitle("Predicted vs actual on the held-out test split",
                 fontsize=11, y=1.01, color=INK)
    fig.text(0.5, -0.07,
             "2-D density, 70 x 70 bins, log count scale, equal axes, dashed identity line. "
             "Each panel is on its OWN logit range:\nthe upstream target spans ~0.3 logit "
             "and the middle one ~14, so the clouds are not to a common scale and the two "
             "R\u00b2 are not comparable.",
             ha="center", fontsize=7.8, color=INK2, linespacing=1.5)
    save(fig, "F1_pred_vs_actual")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    for fn in (fig_c1, fig_c2, fig_c3, fig_b1, fig_d1, fig_d2,
               fig_a1, fig_a2, fig_e1, fig_e2, fig_e3, fig_f1):
        fn()
    print("done")
