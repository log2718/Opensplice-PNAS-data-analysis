"""
Full-track AlphaGenome SPLICE_SITES sanity check on the real PNAS three-exon
minigene cassette.

Question
--------
Do the four *known* splice sites of the PNAS reporter stand out from ordinary
intronic positions in AlphaGenome's full donor/acceptor tracks? If they do,
the SPLICE_SITES-derived metric we have been using may be defensible as a
splice-site strength / recognizability score -- which is a separate question
from whether it is calibrated to PSI.

This uses the same ``SPLICE_SITES`` output as the existing pipeline. It does
NOT use SPLICE_SITE_USAGE or SPLICE_JUNCTIONS, and it never runs the PNAS
model -- PNAS data is used only as a source of exon sequences and their
measured PSI / read counts.

Construct
---------
The fixed experimental cassette, with only the 70-nt PNAS exon varying::

    EXON1 + INTRON1 + [ gtt + <70-nt PNAS exon> + cag ] + INTRON2 + EXON3

The ``gtt`` / ``cag`` are fixed construct bases, not exon-derived. This was
verified against the PNAS model's own hardcoded flanks
(``PNAS_model/utils.py``): ``INTRON1[-7:] + "gtt" == LEFT_FLANK ==
"CATCCAGGTT"`` and ``"cag" + INTRON2[:7] == RIGHT_FLANK == "CAGGTCTGAC"``.
Those identities are asserted at import time.

The core is then centered in 16,384 nt of N padding by the *existing*
pipeline's ``center_pad``, exactly as the FAS minigene pipeline does.

Site coordinates are derived from segment lengths (never hardcoded) and every
construct is checked for canonical GT/AG at all four boundaries before any
prediction is made.

Background rules (per the analysis spec)
----------------------------------------
* N padding excluded entirely -- only the real core is considered.
* Intronic background = INTRON1 + INTRON2 positions only; all three exon
  interiors are excluded.
* A +/-10 nt window around all four known sites is excluded, so a true site's
  own shoulder never contaminates the background.
* Donor sites are ranked against the donor track, acceptor sites against the
  acceptor track -- computed separately, never pooled.

Outputs
-------
* ``outputs/alphagenome_pnas_sanity/pnas_fulltrack_splice_site_sanity.csv``
  -- one row per construct (wide: four sites x site statistics).
* ``outputs/plots/alphagenome_pnas_sanity/true_site_percentile_distributions.png``
* ``outputs/plots/alphagenome_pnas_sanity/psi_vs_prod_logit_calibration.png``

The run is resumable: results are flushed to the CSV periodically and an
existing CSV is reused, so an interrupted 7,500-call run is not lost.

Usage
-----
    python analysis_script/run_alphagenome_pnas_fulltrack_sanity.py \\
        --env-path .env.alphagenome
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import alphagenome_local_prediction_pipeline_minigene as ag  # noqa: E402
from alphagenome.models import dna_client  # noqa: E402

OUT_CSV = ROOT / "outputs" / "alphagenome_pnas_sanity" / "pnas_fulltrack_splice_site_sanity.csv"
PLOT_DIR = ROOT / "outputs" / "plots" / "alphagenome_pnas_sanity"
NPZ_DIR = ROOT / "PNAS_model" / "data"

# ==========================================================================
# The fixed experimental cassette
# ==========================================================================

EXON1 = (
    "gtaccgcaacctcaaacagacaccatggtgcacctgactcctgaggagaagtctgccgttactgcc"
    "ctgtggggcaaggtgaacgtggatgaagttggtggtgaggccctgggcag"
).upper()

INTRON1 = (
    "gttggtatcaaggttacaagacaggtttaaggagaccaatagaaactgggcatatggagacagagaag"
    "actcttgggtttctgataggcactgactctctctgcctatgtctttctctgccatccag"
).upper()

EXON2_PREFIX = "GTT"   # fixed construct bases, not exon-derived
EXON2_SUFFIX = "CAG"   # fixed construct bases, not exon-derived

INTRON2 = (
    "gtctgactatgggacccttgatgttttctttccccttcttttctatggttaagttcatgtcataggaagggg"
    "agaagtaacagggtacagtttagaatgggaaacagacgaatgattgcatcagtgtggaagtctcaggatcgt"
    "tttagtttcttttatttgctgttcataacaattgttttcttttgtttaattcttgctttcttttttttcttc"
    "tccgcaatttttactattatacttaatgccttaacattgtgtataacaaaaggaaatatctctgagataca"
    "ttaagtaacttaaaaaaaaactttacacagtctgcctagtacattactatttggaatatatgtgtgcttat"
    "ttgcatattcataatctccctactttattttcttttatttttaattgatacataatcattatacatattta"
    "tgggttaaagtgtaatgttttaatatcgatacacatattgaccaaatcagggtaattttgcatttgtaatt"
    "ttaaaaaatgctttcttcttttaatatacttttttgtttatcttatttctaatactttccctaatctcttt"
    "ctttcagggcaataatgatacaatgtatcatgcctctttgcaccattctaaagaataacagtgataatttc"
    "tgggttaaggcaatagcaatatttctgcatataaatatttctgcatataaattgtaactgatgtaagaggt"
    "ttcatattgctaatagcagctacaatccagctaccattctgcttttattttatggttgggataaggctgga"
    "ttattctgagtccaagctaggcccttttgctaatcatgttcatacctcttatcttcctcccacag"
).upper()

EXON3 = (
    "ctcctgggcaacgtgctggtctgtgtgctggcccatcactttggcaaagaattcaccccaccagtgcagg"
    "ctgcctatcagaaagtggtggctggtgtggctaatgccctggcccacaagtatcactaagctcgctctaga"
).upper()

PNAS_EXON_LEN = 70
WINDOW = 3          # +/- nt peak check
EXCLUDE_RADIUS = 10  # +/- nt excluded around every true site from background

PSI_BINS = [
    (-0.001, 0.05, "<0.05"),
    (0.05, 0.25, "0.05-0.25"),
    (0.25, 0.75, "0.25-0.75"),
    (0.75, 0.95, "0.75-0.95"),
    (0.95, 1.001, ">0.95"),
]
BIN_LABELS = [b[2] for b in PSI_BINS]

#: site key -> (label, which positive-strand track)
SITES = (
    ("exon1_donor", "Exon1 donor (5'SS)", "donor"),
    ("middle_acceptor", "Middle exon acceptor (3'SS)", "acceptor"),
    ("middle_donor", "Middle exon donor (5'SS)", "donor"),
    ("exon3_acceptor", "Exon3 acceptor (3'SS)", "acceptor"),
)


def _verify_construct_design() -> None:
    """Assert the cassette matches the PNAS model's own fixed flanks."""
    sys.path.insert(0, str(ROOT / "PNAS_model"))
    from utils import LEFT_FLANK, RIGHT_FLANK  # noqa: E402

    left = INTRON1[-7:] + EXON2_PREFIX
    right = EXON2_SUFFIX + INTRON2[:7]
    if left != LEFT_FLANK.upper():
        raise AssertionError(f"INTRON1[-7:]+gtt = {left!r} != PNAS LEFT_FLANK {LEFT_FLANK!r}")
    if right != RIGHT_FLANK.upper():
        raise AssertionError(f"cag+INTRON2[:7] = {right!r} != PNAS RIGHT_FLANK {RIGHT_FLANK!r}")


_verify_construct_design()


def _logit(p: float, eps: float = 1e-7) -> float:
    """Same convention as ``run_hybrid_pnas_alphagenome._logit``."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


# ==========================================================================
# Construct assembly + coordinate derivation
# ==========================================================================

def build_construct(pnas_exon: str) -> dict:
    """Assemble the cassette and derive all four site positions from lengths."""
    pnas_exon = ag.clean_seq(pnas_exon)
    if len(pnas_exon) != PNAS_EXON_LEN:
        raise ValueError(f"expected {PNAS_EXON_LEN}-nt PNAS exon, got {len(pnas_exon)}")

    exon2 = EXON2_PREFIX + pnas_exon + EXON2_SUFFIX
    core = ag.clean_seq(EXON1 + INTRON1 + exon2 + INTRON2 + EXON3)
    padded, left_pad, right_pad = ag.center_pad(core)

    # segment offsets within the core
    e1_start = 0
    i1_start = len(EXON1)
    e2_start = i1_start + len(INTRON1)
    i2_start = e2_start + len(exon2)
    e3_start = i2_start + len(INTRON2)
    core_end = e3_start + len(EXON3)

    construct = {
        "exon1_donor": i1_start - 1,        # last base of EXON1
        "middle_acceptor": e2_start,        # first base of exon2
        "middle_donor": i2_start - 1,       # last base of exon2
        "exon3_acceptor": e3_start,         # first base of EXON3
    }

    # canonical dinucleotide checks at all four boundaries
    checks = {
        "INTRON1 begins GT": core[i1_start:i1_start + 2] == "GT",
        "INTRON1 ends AG": core[e2_start - 2:e2_start] == "AG",
        "INTRON2 begins GT": core[i2_start:i2_start + 2] == "GT",
        "INTRON2 ends AG": core[e3_start - 2:e3_start] == "AG",
        "exon2 length is 76": len(exon2) == PNAS_EXON_LEN + 6,
        "core fully assembled": core_end == len(core),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(f"construct geometry check(s) failed: {bad}")

    return {
        "padded": padded,
        "core_length": len(core),
        "left_pad": left_pad,
        "construct_pos0": construct,
        "padded_pos0": {k: left_pad + v for k, v in construct.items()},
        # intronic background, in *padded* coordinates
        "intron_spans": [
            (left_pad + i1_start, left_pad + e2_start),   # INTRON1
            (left_pad + i2_start, left_pad + e3_start),   # INTRON2
        ],
    }


def _background_mask(geom: dict, total_len: int) -> np.ndarray:
    """Boolean mask over the padded input: intronic positions only, with a
    +/-EXCLUDE_RADIUS window around every known true site removed."""
    mask = np.zeros(total_len, dtype=bool)
    for lo, hi in geom["intron_spans"]:
        mask[lo:hi] = True
    for pos in geom["padded_pos0"].values():
        mask[max(0, pos - EXCLUDE_RADIUS):pos + EXCLUDE_RADIUS + 1] = False
    return mask


def site_stats(track: np.ndarray, pos: int, background: np.ndarray) -> dict:
    """True-site value, +/-3 peak, and rank statistics vs intronic background."""
    at_pos = float(track[pos])
    win = np.asarray(track[pos - WINDOW:pos + WINDOW + 1], dtype=float)
    arg = int(np.argmax(win))

    n_bg = background.size
    n_above = int((background > at_pos).sum())
    return {
        "probability": at_pos,
        "max_pm3": float(win[arg]),
        "peak_offset_pm3": arg - WINDOW,
        "percentile_vs_intron": float((background < at_pos).sum()) / n_bg * 100.0,
        "intron_bg_median": float(np.median(background)),
        "intron_bg_max": float(background.max()),
        "n_intron_above": n_above,
        "frac_intron_above": n_above / n_bg,
        "n_background": n_bg,
    }


# ==========================================================================
# Sampling
# ==========================================================================

def _bin_label(p: float) -> str | None:
    for lo, hi, lab in PSI_BINS:
        if lo <= p < hi:
            return lab
    return None


def load_and_sample(seed: int, n_train: int, n_test: int, min_depth: int) -> pd.DataFrame:
    frames = []
    for split, want in (("train", n_train), ("test", n_test)):
        d = np.load(NPZ_DIR / f"{split}_data.npz", allow_pickle=True)
        df = pd.DataFrame({
            "exon": d["exon"],
            "metadata_PSI": d["metadata_PSI"],
            "metadata_num_exon_inclusion": d["metadata_num_exon_inclusion"],
            "metadata_num_exon_skipping": d["metadata_num_exon_skipping"],
        })
        df["total_depth"] = (
            df["metadata_num_exon_inclusion"] + df["metadata_num_exon_skipping"]
        )
        n0 = len(df)
        df = df[df["total_depth"] >= min_depth].drop_duplicates("exon")
        df["psi_bin"] = df["metadata_PSI"].map(_bin_label)
        df["split"] = split

        picked = []
        print(f"\n[{split}] rows={n0} -> depth>={min_depth} & unique: {len(df)}")
        for lab in BIN_LABELS:
            sub = df[df["psi_bin"] == lab]
            take = min(len(sub), want)
            if take < want:
                print(f"  {lab:10} available={len(sub):6d}  take={take:5d}  << SHORT of {want}")
            else:
                print(f"  {lab:10} available={len(sub):6d}  take={take:5d}")
            picked.append(sub.sample(n=take, random_state=seed))
        frames.append(pd.concat(picked))

    train = frames[0]
    test = frames[1]
    # remove any sampled test sequence that also occurs in the sampled train set
    overlap = set(train["exon"]) & set(test["exon"])
    if overlap:
        print(f"\n[overlap] {len(overlap)} sampled test sequence(s) also in sampled train -- removed from test")
        test = test[~test["exon"].isin(overlap)]
    else:
        print("\n[overlap] no sampled test sequence occurs in the sampled train set")

    out = pd.concat([train, test], ignore_index=True)
    print(f"\nTOTAL constructs to score: {len(out)}  (train={len(train)}, test={len(test)})")
    return out


# ==========================================================================
# Scoring
# ==========================================================================

def score_all(model, sample: pd.DataFrame, flush_every: int) -> pd.DataFrame:
    done: dict[str, dict] = {}
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done = {r["exon"]: r.to_dict() for _, r in prev.iterrows()}
        print(f"[resume] reusing {len(done)} already-scored constructs from {OUT_CSV}")

    records: list[dict] = list(done.values())
    todo = sample[~sample["exon"].isin(done.keys())]
    print(f"[scoring] {len(todo)} construct(s) to call\n")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        geom = build_construct(row["exon"])
        pred = model.predict_sequence(
            sequence=geom["padded"],
            organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=[dna_client.OutputType.SPLICE_SITES],
            ontology_terms=["CL:0002518"],
        )
        donor_track, acceptor_track = ag.get_positive_strand_tracks(pred)
        tracks = {"donor": donor_track, "acceptor": acceptor_track}

        bg_mask = _background_mask(geom, len(donor_track))
        backgrounds = {
            "donor": donor_track[bg_mask],
            "acceptor": acceptor_track[bg_mask],
        }

        rec = {
            "exon": row["exon"],
            "split": row["split"],
            "psi_bin": row["psi_bin"],
            "metadata_PSI": row["metadata_PSI"],
            "metadata_num_exon_inclusion": row["metadata_num_exon_inclusion"],
            "metadata_num_exon_skipping": row["metadata_num_exon_skipping"],
            "total_depth": row["total_depth"],
            "core_length": geom["core_length"],
            "left_pad": geom["left_pad"],
        }
        probs = {}
        for key, _label, track_name in SITES:
            pos = geom["padded_pos0"][key]
            st = site_stats(tracks[track_name], pos, backgrounds[track_name])
            probs[key] = st["probability"]
            rec[f"{key}_expected_position"] = pos
            for stat, val in st.items():
                rec[f"{key}_{stat}"] = val

        acc, don = probs["middle_acceptor"], probs["middle_donor"]
        rec["acceptor_probability"] = acc
        rec["donor_probability"] = don
        rec["mean_logit"] = (_logit(acc) + _logit(don)) / 2.0
        rec["prod_logit"] = _logit(acc * don)
        records.append(rec)

        if i % flush_every == 0 or i == len(todo):
            pd.DataFrame(records).to_csv(OUT_CSV, index=False)
            print(f"  [{i}/{len(todo)}] flushed ({len(records)} total rows)")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(df)} rows -> {OUT_CSV}")
    return df


# ==========================================================================
# Analyses A / B / C
# ==========================================================================

def analysis_A(df: pd.DataFrame) -> None:
    print("\n" + "=" * 104)
    print("A. SPLICE_SITES sanity -- true-site percentile vs intronic background")
    print("=" * 104)
    for split in ("train", "test"):
        s = df[df["split"] == split]
        if s.empty:
            continue
        print(f"\n[{split}]  n={len(s)}")
        rows = []
        for key, label, _t in SITES:
            pc = s[f"{key}_percentile_vs_intron"]
            rows.append({
                "site": label,
                "median_prob": s[f"{key}_probability"].median(),
                "median_pctile": pc.median(),
                "p5_pctile": pc.quantile(.05),
                "frac_top1%": (pc >= 99).mean(),
                "frac_top5%": (pc >= 95).mean(),
                "frac_top10%": (pc >= 90).mean(),
                "frac_any_intron_above": (s[f"{key}_n_intron_above"] > 0).mean(),
                "median_peak_offset": s[f"{key}_peak_offset_pm3"].median(),
            })
        print(pd.DataFrame(rows).round(4).to_string(index=False))


def analysis_B(df: pd.DataFrame) -> None:
    print("\n" + "=" * 104)
    print("B. Experimental calibration of the existing middle-exon metric")
    print("=" * 104)
    for split in ("train", "test"):
        s = df[df["split"] == split]
        if s.empty:
            continue
        print(f"\n[{split}]")
        rows = []
        for lab in BIN_LABELS:
            b = s[s["psi_bin"] == lab]
            if b.empty:
                continue
            r = {"psi_bin": lab, "n": len(b)}
            for col, nm in (("acceptor_probability", "acc"), ("donor_probability", "don"),
                            ("mean_logit", "mean_logit"), ("prod_logit", "prod_logit")):
                r[f"{nm}_median"] = b[col].median()
                r[f"{nm}_IQR"] = b[col].quantile(.75) - b[col].quantile(.25)
            rows.append(r)
        print(pd.DataFrame(rows).round(4).to_string(index=False))


def analysis_C(df: pd.DataFrame) -> None:
    print("\n" + "=" * 104)
    print("C. Extreme regimes: PSI < 0.05  vs  PSI > 0.95  (prod_logit)")
    print("=" * 104)
    for split in ("train", "test"):
        s = df[df["split"] == split]
        if s.empty:
            continue
        lo = s[s["psi_bin"] == "<0.05"]["prod_logit"]
        hi = s[s["psi_bin"] == ">0.95"]["prod_logit"]
        if lo.empty or hi.empty:
            continue
        # overlap: fraction of the low group above the high group's 25th pct, and vice versa
        hi_q25, lo_q75 = hi.quantile(.25), lo.quantile(.75)
        # probability a random low-PSI construct scores >= a random high-PSI one
        rng = np.random.default_rng(0)
        a = rng.choice(lo.to_numpy(), size=20000)
        b = rng.choice(hi.to_numpy(), size=20000)
        print(f"\n[{split}]  n_low={len(lo)}  n_high={len(hi)}")
        print(f"  PSI<0.05  prod_logit: median={lo.median():.4f}  IQR=[{lo.quantile(.25):.4f}, {lo.quantile(.75):.4f}]")
        print(f"  PSI>0.95  prod_logit: median={hi.median():.4f}  IQR=[{hi.quantile(.25):.4f}, {hi.quantile(.75):.4f}]")
        print(f"  separation (median difference): {hi.median() - lo.median():+.4f}")
        print(f"  P(random low >= random high)  : {(a >= b).mean():.4f}   (0.5 = no separation)")
        print(f"  low IQR-top ({lo_q75:.3f}) vs high IQR-bottom ({hi_q25:.3f}): "
              f"{'OVERLAP' if lo_q75 > hi_q25 else 'disjoint IQRs'}")


# ==========================================================================
# Plots (exactly two)
# ==========================================================================

def make_plots(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#3b6fd6", "test": "#e07b39"}

    # ---- Plot 1: true-site percentile distributions, one panel per site ----
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True, sharey=True)
    bins = np.linspace(0, 100, 51)
    for ax, (key, label, _t) in zip(axes, SITES):
        for split in ("train", "test"):
            s = df[df["split"] == split]
            if s.empty:
                continue
            ax.hist(s[f"{key}_percentile_vs_intron"], bins=bins, histtype="step",
                    linewidth=1.8, color=colors[split], label=f"{split} (n={len(s)})",
                    density=True)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Percentile vs intronic background")
        ax.grid(alpha=.25, linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("AlphaGenome SPLICE_SITES -- do known PNAS cassette splice sites stand out "
                 "from intronic background?", fontsize=12)
    p1 = PLOT_DIR / "true_site_percentile_distributions.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")

    # ---- Plot 2: PSI vs prod_logit calibration + extreme-regime overlap ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    ax = axes[0]
    for split in ("train", "test"):
        s = df[df["split"] == split]
        if s.empty:
            continue
        # nonparametric: median prod_logit in 20 equal-count PSI bins
        q = pd.qcut(s["metadata_PSI"], 20, duplicates="drop")
        g = s.groupby(q, observed=True).agg(psi=("metadata_PSI", "median"),
                                            pl=("prod_logit", "median"),
                                            lo=("prod_logit", lambda v: v.quantile(.25)),
                                            hi=("prod_logit", lambda v: v.quantile(.75)))
        ax.plot(g["psi"], g["pl"], "-o", ms=4, color=colors[split], label=f"{split} (median)")
        ax.fill_between(g["psi"], g["lo"], g["hi"], color=colors[split], alpha=.15)
    ax.set_xlabel("Experimental PSI (PNAS metadata_PSI)")
    ax.set_ylabel("AlphaGenome middle-exon prod_logit")
    ax.set_title("Nonparametric calibration (20 equal-count PSI bins, IQR shaded)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.25, linewidth=.7); ax.set_axisbelow(True)

    ax = axes[1]
    for split, ls in (("train", "-"), ("test", "--")):
        s = df[df["split"] == split]
        if s.empty:
            continue
        for lab, col in (("<0.05", "#b5179e"), (">0.95", "#2ca85a")):
            v = s[s["psi_bin"] == lab]["prod_logit"]
            if v.empty:
                continue
            ax.hist(v, bins=40, histtype="step", linewidth=1.7, linestyle=ls,
                    color=col, density=True, label=f"{split} PSI {lab} (n={len(v)})")
    ax.set_xlabel("AlphaGenome middle-exon prod_logit")
    ax.set_ylabel("Density")
    ax.set_title("Extreme regimes: experimentally skipped vs included", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.25, linewidth=.7); ax.set_axisbelow(True)

    fig.suptitle("PNAS cassette -- experimental PSI vs the existing SPLICE_SITES-derived metric",
                 fontsize=12)
    p2 = PLOT_DIR / "psi_vs_prod_logit_calibration.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p2}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Full-track SPLICE_SITES sanity check on the PNAS cassette.")
    ap.add_argument("--env-path", default=".env.alphagenome")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1000, help="per PSI bin (default 1000)")
    ap.add_argument("--n-test", type=int, default=500, help="per PSI bin (default 500)")
    ap.add_argument("--min-depth", type=int, default=100)
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip all API calls; re-run analyses/plots from the existing CSV")
    args = ap.parse_args(argv)

    if args.analyze_only:
        df = pd.read_csv(OUT_CSV)
        print(f"[analyze-only] loaded {len(df)} rows from {OUT_CSV}")
    else:
        sample = load_and_sample(args.seed, args.n_train, args.n_test, args.min_depth)
        print(f"\nexpected API calls: {len(sample)}")
        model = ag.create_model(env_path=args.env_path)
        df = score_all(model, sample, args.flush_every)

    analysis_A(df)
    analysis_B(df)
    analysis_C(df)
    make_plots(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
