"""
Full-track AlphaGenome retrieval for the B1 cryptic-splice-site check.

``synthetic_context_lib.score_core`` reads only the middle exon's two splice-site
positions, so the B1 pilot could not see whether a randomized 113-mer created a
NOVEL splice site elsewhere in the construct. This module makes the same
AlphaGenome call with the same arguments and returns the WHOLE positive-strand
donor and acceptor tracks cropped to the unpadded core, alongside the identical
scalar dictionary ``score_core`` produces.

Nothing under ``analysis_script/`` or ``PNAS_model/`` is modified or
reimplemented: ``create_model`` / ``clean_seq`` / ``center_pad`` /
``get_positive_strand_tracks`` and ``synthetic_context_lib._logit`` are imported
and used as-is, and :func:`score_core_fulltrack` is deliberately a line-for-line
echo of ``score_core`` plus the track crop, so :func:`assert_parity` can show the
two agree to floating-point equality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
_ANALYSIS = ROOT / "analysis_script"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import synthetic_context_lib as L  # noqa: E402
import splice_site_partition as SP  # noqa: E402

#: how many nt either side of a KNOWN splice site are not counted as off-target.
#: AlphaGenome's splice-site peaks are a few nt wide, so the exact position alone
#: is too tight an exclusion -- the shoulder of a real peak would masquerade as a
#: cryptic site. Reported alongside the exact-position-only exclusion.
EXCLUDE_HALFWIDTH = 3

#: window searched for the middle exon's own peak when testing for drift
DRIFT_HALFWIDTH = 5

#: off-target thresholds reported
THRESHOLDS = (0.1, 0.3, 0.5)


# ==========================================================================
# the scoring call
# ==========================================================================

def score_core_fulltrack(model, core: str, exon_start: int, exon_len: int,
                         ontology_terms=("CL:0002518",)) -> dict:
    """``score_core``'s scalars PLUS the full positive-strand tracks.

    Same padding, same prediction arguments, same track extraction, same
    acceptor/donor indices. The returned ``donor_track`` / ``acceptor_track`` are
    cropped to the unpadded core, so index ``i`` is core base ``i``.
    """
    import alphagenome_local_prediction_pipeline_minigene as ag
    from alphagenome.models import dna_client

    padded, left_pad, _ = ag.center_pad(ag.clean_seq(core))
    acc_pos = left_pad + exon_start
    don_pos = left_pad + exon_start + exon_len - 1

    pred = model.predict_sequence(
        sequence=padded,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_client.OutputType.SPLICE_SITES],
        ontology_terms=list(ontology_terms) if ontology_terms is not None else None,
    )
    donor_track, acceptor_track = ag.get_positive_strand_tracks(pred)
    acc, don = float(acceptor_track[acc_pos]), float(donor_track[don_pos])
    return {
        "AG_acceptor_probability": acc,
        "AG_donor_probability": don,
        "AG_mean_prob": (acc + don) / 2.0,
        "AG_mean_logit": (L._logit(acc) + L._logit(don)) / 2.0,
        "AG_prod_logit": L._logit(acc * don),
        "core_length": len(core),
        # extras, not part of score_core's contract
        "left_pad": int(left_pad),
        "donor_track": np.asarray(donor_track[left_pad:left_pad + len(core)],
                                  dtype=np.float32),
        "acceptor_track": np.asarray(acceptor_track[left_pad:left_pad + len(core)],
                                     dtype=np.float32),
    }


_SCALAR_KEYS = ("AG_acceptor_probability", "AG_donor_probability",
                "AG_mean_prob", "AG_mean_logit", "AG_prod_logit", "core_length")


def assert_parity(model, core: str, exon_start: int, exon_len: int) -> dict:
    """Call ``score_core`` and :func:`score_core_fulltrack` on one construct and
    require every shared scalar to match exactly (``==``, not ``isclose``).

    Costs two AlphaGenome calls. AlphaGenome was measured bit-identical within a
    session in the pilot, so exact equality is the right bar; a mismatch means
    the two code paths differ and the check must stop.
    """
    ref = L.score_core(model, core, exon_start, exon_len)
    got = score_core_fulltrack(model, core, exon_start, exon_len)
    diffs = {k: (ref[k], got[k]) for k in _SCALAR_KEYS if ref[k] != got[k]}
    return {"reference": ref, "fulltrack": got, "mismatches": diffs}


# ==========================================================================
# where the four KNOWN splice sites are
# ==========================================================================

def known_sites(b1_len: int, exon_start: int, exon_len: int) -> dict[str, dict]:
    """The four annotated splice-site positions in a B1-substituted core.

    Offsets follow ``assemble8``: ``B1 B2 B3 B4 | exon | B5 B6 B7 B8`` with the
    middle exon starting 3 nt before the synthetic 70-mer.

    ====================  =====  =========================================
    site                  kind   position
    ====================  =====  =========================================
    upstream exon 5'SS    donor  last base of the upstream exon
    middle exon 3'SS      acc    ``exon_start``          (score_core's acc_pos)
    middle exon 5'SS      donor  ``exon_start+exon_len-1`` (score_core's don_pos)
    downstream exon 3'SS  acc    first base of the downstream exon
    ====================  =====  =========================================

    The upstream exon has no acceptor and the downstream exon no donor inside the
    construct -- they are the cassette's outer boundaries -- so four is the
    complete set.
    """
    nb = SP.pnas_blocks8()
    off_b2 = b1_len
    off_b3 = off_b2 + len(nb["B2"])
    off_b4 = off_b3 + len(nb["B3"])
    left_len = off_b4 + len(nb["B4"])
    off_b5 = left_len + L.SYN_LEN
    off_b6 = off_b5 + len(nb["B5"])
    off_b7 = off_b6 + len(nb["B6"])

    assert left_len - len(L.EXON_PREFIX) == exon_start, (
        f"block offsets put the exon at {left_len - len(L.EXON_PREFIX)}, "
        f"manifest says {exon_start}")

    return {
        "upstream_donor": {"kind": "donor",
                           "pos": off_b2 + SP.DONOR_EXONIC - 1},
        "middle_acceptor": {"kind": "acceptor", "pos": exon_start},
        "middle_donor": {"kind": "donor", "pos": exon_start + exon_len - 1},
        "downstream_acceptor": {"kind": "acceptor",
                                "pos": off_b7 + SP.ACCEPTOR_INTRONIC},
    }


# ==========================================================================
# off-target peak measurement
# ==========================================================================

def _masked_peak(track: np.ndarray, lo: int, hi: int,
                 blocked: np.ndarray) -> tuple[float, int]:
    """(max, argmax) of ``track[lo:hi]`` ignoring positions where ``blocked``."""
    idx = np.arange(lo, hi)
    idx = idx[~blocked[lo:hi]]
    if idx.size == 0:
        return float("nan"), -1
    vals = track[idx]
    j = int(np.argmax(vals))
    return float(vals[j]), int(idx[j])


def measure(donor: np.ndarray, acceptor: np.ndarray, b1_len: int,
            exon_start: int, exon_len: int,
            exclude_halfwidth: int = EXCLUDE_HALFWIDTH) -> dict:
    """All off-target / drift quantities for one construct's tracks.

    ``donor`` and ``acceptor`` are core-cropped, so index ``i`` is core base
    ``i``. Regions: ``b1`` is ``[0, b1_len)``; ``rest`` is everything after it.
    With the native 113-nt B1 the nearest known site (the upstream donor at 115)
    falls outside B1, so the B1 region carries no annotated site at all and any
    peak in it is novel by construction.
    """
    n = len(donor)
    sites = known_sites(b1_len, exon_start, exon_len)

    blocked_wide = np.zeros(n, dtype=bool)
    blocked_exact = np.zeros(n, dtype=bool)
    for s in sites.values():
        p = s["pos"]
        blocked_exact[p] = True
        blocked_wide[max(0, p - exclude_halfwidth):
                     min(n, p + exclude_halfwidth + 1)] = True

    out: dict = {"b1_len": b1_len, "core_length": n,
                 "exclude_halfwidth": exclude_halfwidth}
    for name, s in sites.items():
        tr = donor if s["kind"] == "donor" else acceptor
        out[f"site_{name}_pos"] = s["pos"]
        out[f"site_{name}_prob"] = float(tr[s["pos"]])

    regions = {"b1": (0, b1_len), "rest": (b1_len, n)}
    for rname, (lo, hi) in regions.items():
        for tname, tr in (("donor", donor), ("acceptor", acceptor)):
            v, p = _masked_peak(tr, lo, hi, blocked_wide)
            out[f"max_{tname}_{rname}"] = v
            out[f"argmax_{tname}_{rname}"] = p
            ve, pe = _masked_peak(tr, lo, hi, blocked_exact)
            out[f"max_{tname}_{rname}_exactexcl"] = ve
            out[f"argmax_{tname}_{rname}_exactexcl"] = pe

    out["offtarget_max"] = max(out[f"max_{t}_{r}"]
                               for t in ("donor", "acceptor")
                               for r in ("b1", "rest"))
    out["offtarget_max_b1"] = max(out["max_donor_b1"], out["max_acceptor_b1"])
    out["offtarget_max_rest"] = max(out["max_donor_rest"], out["max_acceptor_rest"])
    out["offtarget_max_exactexcl"] = max(out[f"max_{t}_{r}_exactexcl"]
                                         for t in ("donor", "acceptor")
                                         for r in ("b1", "rest"))

    keep = ~blocked_wide
    for thr in THRESHOLDS:
        tag = str(thr).replace(".", "p")
        for rname, (lo, hi) in regions.items():
            k = keep[lo:hi]
            out[f"n_donor_{rname}_gt{tag}"] = int(
                ((donor[lo:hi] > thr) & k).sum())
            out[f"n_acceptor_{rname}_gt{tag}"] = int(
                ((acceptor[lo:hi] > thr) & k).sum())
        out[f"n_offtarget_gt{tag}"] = sum(
            out[f"n_{t}_{r}_gt{tag}"] for t in ("donor", "acceptor")
            for r in ("b1", "rest"))

    # drift of the middle exon's own two peaks
    for name, tr in (("acceptor", acceptor), ("donor", donor)):
        expect = sites[f"middle_{name}"]["pos"]
        lo = max(0, expect - DRIFT_HALFWIDTH)
        hi = min(n, expect + DRIFT_HALFWIDTH + 1)
        w = tr[lo:hi]
        j = int(np.argmax(w))
        out[f"middle_{name}_local_argmax"] = lo + j
        out[f"middle_{name}_local_max"] = float(w[j])
        out[f"middle_{name}_drift"] = (lo + j) - expect

    # is the middle exon's site still the global best of its kind in the core?
    out["global_argmax_acceptor"] = int(np.argmax(acceptor))
    out["global_max_acceptor"] = float(acceptor.max())
    out["global_argmax_donor"] = int(np.argmax(donor))
    out["global_max_donor"] = float(donor.max())
    return out
