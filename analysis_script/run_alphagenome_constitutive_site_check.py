"""
AlphaGenome-only sanity check on the constitutive FAS splice sites of the
OpenSplice three-exon reporter.

Question
--------
The construct AlphaGenome is given is a three-exon cassette::

    FAS_E5 -- intron -- variable/test exon -- intron -- FAS_E7

FAS_E5 and FAS_E7 are the fixed, constitutively included flanking exons of
the reporter; only the middle exon is alternatively included/skipped. Every
existing experiment in this repo reads out *only* the middle exon's acceptor
and donor. This script asks whether AlphaGenome assigns strong splice-site
usage to the two outer constitutive sites -- a sanity check on whether the
middle-exon probabilities we build ``mean_logit`` / ``prod_logit`` from are
behaving sensibly at all.

The expectation is NOT that the FAS sites equal 1.0. The question is whether
they are consistently strong/high-confidence relative to the variable exon's
sites, and whether they stay stable when the middle exon is mutated.

No PNAS
-------
No PNAS model is instantiated or run. ``load_opensplice_exon`` is imported
from ``run_hybrid_pnas_alphagenome`` purely as the canonical WT-sequence
loader; for the two mutant constructs the ``exon_seq`` strings are read
verbatim out of existing motif-substitution CSVs (read-only -- no existing
output file is modified). Any ``pnas_pretuner`` value read from those CSVs is
used only to *pick* which stored variant to score, never recomputed.

Construct is unchanged
----------------------
The biological construct is built by the existing pipeline's own functions
(``build_variable_region`` -> ``build_fas_minigene`` -> ``center_pad``) and
the prediction uses the same arguments as
``score_variable_region`` (HOMO_SAPIENS / SPLICE_SITES / CL:0002518) and the
same ``get_positive_strand_tracks`` extraction. The only difference is that
this script indexes four positions in the returned full-length tracks instead
of two -- ``score_variable_region`` does not return the tracks, so the
``predict_sequence`` call is made here rather than modifying that validated
module.

Site coordinates are derived, not hardcoded
-------------------------------------------
All four positions are recomputed from the construct itself:

* middle exon acceptor / donor -- ``canonical_construct_positions``
* FAS_E5 donor    -- ``len(FAS_E5) - 1``        (FAS_E5 begins at ``core[0]``)
* FAS_E7 acceptor -- ``len(core) - len(FAS_E7)`` (FAS_E7 ends the core)

each shifted by ``left_pad`` into padded coordinates. Both a base-identity
check and a comparison against the independently verified expected positions
run before any prediction, so a geometry change fails loudly instead of
silently scoring the wrong index.

Outputs
-------
* ``outputs/alphagenome_sanity/constitutive_splice_site_check.csv``
* ``outputs/plots/alphagenome_sanity/constitutive_splice_site_check.png``
  -- one grouped bar figure, WT e6 vs WT e7, four sites.

Usage
-----
    python analysis_script/run_alphagenome_constitutive_site_check.py \\
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

OUT_CSV = ROOT / "outputs" / "alphagenome_sanity" / "constitutive_splice_site_check.csv"
OUT_PNG = ROOT / "outputs" / "plots" / "alphagenome_sanity" / "constitutive_splice_site_check.png"

E7_UAG_CSV = ROOT / "outputs" / "motif_substitutions" / "TP53_e7_UAG_1x_15x.csv"
E7_GGA_CSV = ROOT / "outputs" / "motif_substitutions" / "TP53_e7_GGA_UAG_1x_4x.csv"

WINDOW = 3  # +/- nt peak-check half-width

#: Padded 0-based positions verified independently from the construct code.
#: Derived values are asserted against these; they are a cross-check, not the
#: source of truth.
EXPECTED_PADDED = {
    "TP53_e6": {"fas_e5_donor": 7999, "middle_acceptor": 8154,
                "middle_donor": 8266, "fas_e7_acceptor": 8399},
    "TP53_e7": {"fas_e5_donor": 8000, "middle_acceptor": 8155,
                "middle_donor": 8264, "fas_e7_acceptor": 8397},
}

#: site key -> (human label, which positive-strand track it lives in)
SITES = (
    ("fas_e5_donor", "FAS_E5 donor (5'SS)", "donor"),
    ("middle_acceptor", "Middle exon acceptor (3'SS)", "acceptor"),
    ("middle_donor", "Middle exon donor (5'SS)", "donor"),
    ("fas_e7_acceptor", "FAS_E7 acceptor (3'SS)", "acceptor"),
)


def _logit(p: float, eps: float = 1e-7) -> float:
    """Identical convention to ``run_hybrid_pnas_alphagenome._logit``."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def derive_site_positions(exon_seq: str, upstream_flank: str, downstream_flank: str) -> dict:
    """Recompute all four splice-site positions from the construct itself.

    Returns construct-relative and padded 0-based positions, plus the core /
    padding geometry. Raises if a base-identity check fails.
    """
    variable_region = ag.build_variable_region(exon_seq, upstream_flank, downstream_flank)
    core = ag.build_fas_minigene(variable_region)
    padded, left_pad, right_pad = ag.center_pad(core)

    exon_length = len(ag.clean_seq(exon_seq))
    acc0, don0 = ag.canonical_construct_positions(
        exon_length=exon_length, upstream_len=ag.DEFAULT_UPSTREAM_LEN,
    )
    fas_e5_donor0 = len(ag.FAS_E5) - 1
    fas_e7_acceptor0 = len(core) - len(ag.FAS_E7)

    # -- base-identity checks: catch any off-by-one / geometry drift --------
    exon_clean = ag.clean_seq(exon_seq)
    checks = {
        "FAS_E5 donor is FAS_E5's last base": core[fas_e5_donor0] == ag.FAS_E5[-1],
        "intron after FAS_E5 starts GT": core[fas_e5_donor0 + 1:fas_e5_donor0 + 3] == "GT",
        "middle acceptor is exon's first base": core[acc0] == exon_clean[0],
        "intron before middle exon ends AG": core[acc0 - 2:acc0] == "AG",
        "middle donor is exon's last base": core[don0] == exon_clean[-1],
        "intron after middle exon starts GT": core[don0 + 1:don0 + 3] == "GT",
        "FAS_E7 acceptor is FAS_E7's first base": core[fas_e7_acceptor0] == ag.FAS_E7[0],
        "intron before FAS_E7 ends AG": core[fas_e7_acceptor0 - 2:fas_e7_acceptor0] == "AG",
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"construct geometry check(s) failed: {failed}")

    construct = {
        "fas_e5_donor": fas_e5_donor0,
        "middle_acceptor": acc0,
        "middle_donor": don0,
        "fas_e7_acceptor": fas_e7_acceptor0,
    }
    return {
        "padded_sequence": padded,
        "core_length": len(core),
        "left_pad": left_pad,
        "right_pad": right_pad,
        "exon_length": exon_length,
        "construct_pos0": construct,
        "padded_pos0": {k: left_pad + v for k, v in construct.items()},
    }


def peak_check(track: np.ndarray, pos: int, window: int = WINDOW) -> tuple[float, float, int]:
    """Value at ``pos``, max within +/-``window``, and the max's offset from ``pos``."""
    lo = pos - window
    hi = pos + window
    sub = np.asarray(track[lo:hi + 1], dtype=float)
    arg = int(np.argmax(sub))
    return float(track[pos]), float(sub[arg]), arg - window


def score_construct(model, exon_id: str, variant_id: str, condition: str,
                    exon_seq: str, upstream_flank: str, downstream_flank: str,
                    ontology_terms=("CL:0002518",)) -> dict:
    geom = derive_site_positions(exon_seq, upstream_flank, downstream_flank)

    expected = EXPECTED_PADDED.get(exon_id)
    if expected is not None and geom["padded_pos0"] != expected:
        raise AssertionError(
            f"{exon_id}: derived padded positions {geom['padded_pos0']} != "
            f"independently verified {expected}"
        )

    pred_out = model.predict_sequence(
        sequence=geom["padded_sequence"],
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_client.OutputType.SPLICE_SITES],
        ontology_terms=list(ontology_terms) if ontology_terms is not None else None,
    )
    donor_track, acceptor_track = ag.get_positive_strand_tracks(pred_out)
    tracks = {"donor": donor_track, "acceptor": acceptor_track}

    row: dict = {
        "exon_id": exon_id,
        "variant_id": variant_id,
        "condition": condition,
        "exon_length": geom["exon_length"],
        "core_length": geom["core_length"],
        "left_pad": geom["left_pad"],
    }
    probs: dict[str, float] = {}
    for key, _label, track_name in SITES:
        pos = geom["padded_pos0"][key]
        at_pos, max_pm3, offset = peak_check(tracks[track_name], pos)
        probs[key] = at_pos
        row[f"{key}_expected_position"] = pos
        row[f"{key}_probability_at_expected_position"] = at_pos
        row[f"{key}_max_probability_pm3"] = max_pm3
        row[f"{key}_peak_offset_pm3"] = offset
        row[f"{key}_logit_at_expected_position"] = _logit(at_pos)

    acc = probs["middle_acceptor"]
    don = probs["middle_donor"]
    row["middle_mean_logit"] = (_logit(acc) + _logit(don)) / 2.0
    row["middle_prod_logit"] = _logit(acc * don)
    return row


def _load_wt_constructs() -> list[dict]:
    # Canonical WT loader; importing it does not run any PNAS inference.
    import run_hybrid_pnas_alphagenome as hybrid

    out = []
    for exon_id in ("TP53_e6", "TP53_e7"):
        wt = hybrid.load_opensplice_exon(exon_id, hybrid.DEFAULT_OPENSPLICE_CSV)
        out.append({
            "exon_id": exon_id,
            "variant_id": f"{exon_id}_WT",
            "condition": "WT",
            "exon_seq": wt.exon_seq,
            "upstream_flank": wt.upstream_flank,
            "downstream_flank": wt.downstream_flank,
        })
    return out


def _load_e7_motif_extremes() -> list[dict]:
    """One UAG-heavy (lowest stored PNAS) and one GGA-heavy (highest stored
    PNAS) TP53_e7 middle-exon variant, read verbatim from existing CSVs."""
    out = []
    if E7_UAG_CSV.exists():
        df = pd.read_csv(E7_UAG_CSV)
        df = df[~df["is_reference"].fillna(False).astype(bool)]
        r = df.loc[df["pnas_pretuner"].idxmin()]
        out.append({
            "exon_id": "TP53_e7",
            "variant_id": f"UAG_{int(r['copy_count'])}x@{int(r['start0'])}",
            "condition": "e7 UAG-heavy (lowest stored PNAS)",
            "exon_seq": r["exon_seq"],
            "upstream_flank": r["upstream_flank"],
            "downstream_flank": r["downstream_flank"],
        })
    else:
        print(f"[note] {E7_UAG_CSV} not found -- skipping UAG-heavy variant.")

    if E7_GGA_CSV.exists():
        df = pd.read_csv(E7_GGA_CSV)
        df = df[(~df["is_reference"].fillna(False).astype(bool)) & (df["motif"] == "GGA")]
        r = df.loc[df["pnas_pretuner"].idxmax()]
        out.append({
            "exon_id": "TP53_e7",
            "variant_id": f"GGA_{int(r['copy_count'])}x@{int(r['start0'])}",
            "condition": "e7 GGA-heavy (highest stored PNAS)",
            "exon_seq": r["exon_seq"],
            "upstream_flank": r["upstream_flank"],
            "downstream_flank": r["downstream_flank"],
        })
    else:
        print(f"[note] {E7_GGA_CSV} not found -- skipping GGA-heavy variant.")
    return out


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    """One grouped bar figure: WT e6 vs WT e7 across the four splice sites."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wt = df[df["condition"] == "WT"]
    series = [("TP53_e6", "#3b6fd6"), ("TP53_e7", "#e07b39")]
    labels = [label for _k, label, _t in SITES]
    x = np.arange(len(SITES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for i, (exon_id, color) in enumerate(series):
        row = wt[wt["exon_id"] == exon_id]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [row[f"{k}_probability_at_expected_position"] for k, _l, _t in SITES]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.94, label=exon_id,
                      color=color, edgecolor="white", linewidth=1.0)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AlphaGenome splice-site probability (+ strand)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "AlphaGenome sanity check -- constitutive FAS sites vs variable middle exon\n"
        "WT three-exon OpenSplice reporter (FAS_E5 - test exon - FAS_E7)",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AlphaGenome-only constitutive-splice-site sanity check (no PNAS)."
    )
    parser.add_argument("--env-path", default=".env.alphagenome")
    parser.add_argument("--wt-only", action="store_true",
                        help="score only WT e6 / WT e7 (skip the two e7 motif variants)")
    args = parser.parse_args(argv)

    constructs = _load_wt_constructs()
    if not args.wt_only:
        constructs += _load_e7_motif_extremes()

    model = ag.create_model(env_path=args.env_path)

    rows = []
    for c in constructs:
        print(f"scoring {c['exon_id']} / {c['variant_id']} ({c['condition']}) ...")
        rows.append(score_construct(
            model, c["exon_id"], c["variant_id"], c["condition"],
            c["exon_seq"], c["upstream_flank"], c["downstream_flank"],
        ))

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(df)} rows -> {OUT_CSV}")

    _plot(df, OUT_PNG)

    # ---- printed summary ------------------------------------------------
    print("\n" + "=" * 100)
    print("Splice-site probabilities (value at expected position | max within +/-3 nt @ offset)")
    print("=" * 100)
    for _, r in df.iterrows():
        print(f"\n{r['exon_id']}  {r['variant_id']}  [{r['condition']}]")
        for key, label, _t in SITES:
            print(
                f"  {label:<30} p={r[f'{key}_probability_at_expected_position']:.6f}   "
                f"max±3={r[f'{key}_max_probability_pm3']:.6f} @ offset "
                f"{int(r[f'{key}_peak_offset_pm3']):+d}   "
                f"logit={r[f'{key}_logit_at_expected_position']:+.4f}"
            )
        print(f"  middle_mean_logit={r['middle_mean_logit']:+.4f}  "
              f"middle_prod_logit={r['middle_prod_logit']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
