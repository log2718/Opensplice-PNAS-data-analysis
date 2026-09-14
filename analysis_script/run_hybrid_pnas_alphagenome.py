"""
Experiment driver: score stubborn-exon **hybrid** constructs with BOTH the PNAS
pretuner and AlphaGenome, from a single shared biological representation.

One representation, two models
------------------------------
Everything is expressed as a :class:`HybridExon`::

    exon_seq          -- exon body, transcript 5'->3'
    upstream_flank    -- upstream intron context (may be the full OpenSplice 70 nt)
    downstream_flank  -- downstream intron context (may be the full 25 nt)

Each model then slices what *it* needs, using its own existing code:

* PNAS  -> ``pnas_prediction_hybrid_input.score_pnas_pretuner`` which internally
  uses only ``upstream_flank[-7:] + exon_seq + downstream_flank[:7]``
  (``add_flanks=False``, ``temperature=37.0``, pre-tuner ``energy_seq_struct``).
* AlphaGenome -> ``alphagenome_local_prediction_pipeline_minigene.score_exon``
  which internally uses the full ``upstream_flank + exon_seq + downstream_flank``,
  wraps it in the FAS minigene, centers it in 16,384 nt of N padding, and reads
  the positive-strand SPLICE_SITES acceptor/donor probabilities. That module's
  minigene / FAS / padding / position logic is reused verbatim -- not duplicated.

No sequence is ever reverse-complemented or reoriented here.

Hybrid constructors (all return a new :class:`HybridExon`)
---------------------------------------------------------
* :func:`make_internal_swap`           -- swap an internal exon-body window
* :func:`make_full_internal_exon_swap` -- swap the *entire* internal exon body
                                          (``exon_seq[3:-3]``), keeping the
                                          recipient's exonic SS nt + flanks;
                                          exon length may change
* :func:`make_3ss_swap`                -- swap the 3' splice site region (configurable
                                          ``n_intronic_3ss`` / ``n_exonic_3ss``)
* :func:`make_5ss_swap`                -- swap the 5' splice site region (configurable
                                          ``n_exonic_5ss`` / ``n_intronic_5ss``)
* :func:`make_3ss_5ss_swap`            -- swap both splice site regions at once
                                          (3'SS + 5'SS from the same donor)
* :func:`make_crossover_chimera`       -- "first X nt of exon A + remainder of exon B"

Saturation + scoring
--------------------
* :func:`generate_single_nt_exonic_variants` -- exon-body (and optionally
  swapped-flank) saturation mutagenesis, carrying full metadata per variant.
* :func:`score_hybrid_variants` -- run every variant through the PNAS pretuner
  and (optionally) AlphaGenome, returning a tidy ``pandas.DataFrame``.

Reusable dataset paths (driver only -- never in the PNAS module):
``ROOT`` and the optional ``outputs/`` destination live here.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Local imports (both new + existing modules live in analysis_script/)
# --------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import pnas_prediction_hybrid_input as pnas             # noqa: E402
# import alphagenome_local_prediction_pipeline_minigene as ag  # noqa: E402

DEFAULT_OPENSPLICE_CSV = ROOT / "arush_data" / "wt_exons.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

DEFAULT_UPSTREAM_LEN = 70
DEFAULT_DOWNSTREAM_LEN = 25

_BASES = ("A", "C", "G", "T")

__all__ = [
    "ROOT",
    "HybridExon",
    "load_opensplice_exon",
    "make_internal_swap",
    "make_full_internal_exon_swap",
    "make_3ss_swap",
    "make_5ss_swap",
    "make_3ss_5ss_swap",
    "make_crossover_chimera",
    "generate_single_nt_exonic_variants",
    "score_hybrid_variants",
]


# ==========================================================================
# Shared biological representation
# ==========================================================================

@dataclass(frozen=True)
class HybridExon:
    """A single exon-in-context construct, transcript-sense 5'->3'.

    ``upstream_flank`` / ``downstream_flank`` may carry the full OpenSplice
    context (70 nt / 25 nt) or more. Both downstream models slice from this
    same object, so a construct is defined exactly once.
    """

    exon_seq: str
    upstream_flank: str
    downstream_flank: str
    construct_type: str = "wt"
    label: str = ""
    #: free-form record of how this construct was built (parents, swap sizes...)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exon_seq", _clean(self.exon_seq))
        object.__setattr__(self, "upstream_flank", _clean(self.upstream_flank))
        object.__setattr__(self, "downstream_flank", _clean(self.downstream_flank))
        if not self.exon_seq:
            raise ValueError(f"{self.label or 'HybridExon'}: exon_seq is empty.")

    # -- convenience views -------------------------------------------------
    @property
    def exon_length(self) -> int:
        return len(self.exon_seq)

    def pnas_input(self, flank_len: int = pnas.DEFAULT_FLANK_LEN) -> str:
        """The exact string the PNAS module will score (7 + exon + 7)."""
        return pnas.build_pnas_input(
            self.exon_seq,
            self.upstream_flank,
            self.downstream_flank,
            flank_len=flank_len,
        )

    def base_record(self) -> dict[str, Any]:
        """Metadata dict shared by every variant derived from this construct.

        Provenance is flattened into explicit columns (``recipient_exon``,
        ``donor_exon``, the four ``n_*ss`` swap sizes) so downstream consumers
        never have to parse the ``provenance`` dict. The absolute ``csv_path`` is
        dropped -- ``source`` in ``provenance`` is enough to trace the origin.
        """
        prov = dict(self.provenance)
        prov.pop("csv_path", None)
        return {
            "construct_type": self.construct_type,
            "construct_label": self.label,
            "recipient_exon": prov.get("recipient"),
            "donor_exon": prov.get("donor"),
            "exon_length": self.exon_length,
            "exon_seq": self.exon_seq,
            "upstream_flank": self.upstream_flank,
            "downstream_flank": self.downstream_flank,
            "n_intronic_3ss": prov.get("n_intronic_3ss"),
            "n_exonic_3ss": prov.get("n_exonic_3ss"),
            "n_exonic_5ss": prov.get("n_exonic_5ss"),
            "n_intronic_5ss": prov.get("n_intronic_5ss"),
            "provenance": prov,
        }


def _clean(seq: str) -> str:
    """Upper-case and strip whitespace. Alphabet checks happen in each model."""
    return "".join(str(seq).split()).upper()


# ==========================================================================
# Loading OpenSplice wild-type exons (driver-side dataset access)
# ==========================================================================

def load_opensplice_exon(
    exon_id: str,
    csv_path: str | Path = DEFAULT_OPENSPLICE_CSV,
    *,
    upstream_len: int = DEFAULT_UPSTREAM_LEN,
    downstream_len: int = DEFAULT_DOWNSTREAM_LEN,
) -> HybridExon:
    """Build a wild-type :class:`HybridExon` from an OpenSplice ``nt_seq``.

    ``nt_seq`` = ``upstream_len`` nt upstream intron + exon + ``downstream_len``
    nt downstream intron, so::

        upstream_flank   = nt_seq[:upstream_len]
        exon_seq         = nt_seq[upstream_len:-downstream_len]
        downstream_flank = nt_seq[-downstream_len:]

    Uses only the Python standard library (no pandas dependency).
    """
    csv_path = Path(csv_path)
    csv.field_size_limit(10**7)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next((r for r in reader if r.get("exon_id") == exon_id), None)

    if row is None:
        raise KeyError(f"exon_id {exon_id!r} not found in {csv_path}.")

    nt_seq = _clean(row["nt_seq"])
    if len(nt_seq) <= upstream_len + downstream_len:
        raise ValueError(
            f"{exon_id}: nt_seq length {len(nt_seq)} too short for "
            f"{upstream_len} + {downstream_len} flanks."
        )

    exon_seq = nt_seq[upstream_len:-downstream_len]
    upstream_flank = nt_seq[:upstream_len]
    downstream_flank = nt_seq[-downstream_len:]

    # Cross-check against the precomputed 'exon' column when present.
    if row.get("exon") and _clean(row["exon"]) != exon_seq:
        raise ValueError(
            f"{exon_id}: reconstructed exon disagrees with the CSV 'exon' column."
        )

    return HybridExon(
        exon_seq=exon_seq,
        upstream_flank=upstream_flank,
        downstream_flank=downstream_flank,
        construct_type="wt",
        label=exon_id,
        provenance={
            "source": "opensplice_wt",
            "exon_id": exon_id,
            "recipient": exon_id,
            "upstream_len": upstream_len,
            "downstream_len": downstream_len,
        },
    )


# ==========================================================================
# Hybrid constructors
# ==========================================================================
#
# Design note: every constructor returns a plain HybridExon. The scoring code
# (score_hybrid_variants / score_pnas_pretuner / ag.score_exon) only ever sees
# (exon_seq, upstream_flank, downstream_flank), so a NEW kind of chimera only
# needs a new make_* function here -- no scoring code changes.

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def make_internal_swap(
    recipient: HybridExon,
    donor: HybridExon,
    *,
    start: int,
    end: int,
    construct_type: str = "internal_swap",
    label: str | None = None,
) -> HybridExon:
    """Replace ``recipient.exon_seq[start:end]`` with ``donor.exon_seq[start:end]``.

    Flanks are untouched. Positions are 0-based, half-open, and indexed into the
    *recipient* exon body.
    """
    n = recipient.exon_length
    _require(0 <= start < end <= n, f"need 0 <= start < end <= {n}, got {start}:{end}")
    _require(
        len(donor.exon_seq) >= end,
        f"donor exon ({len(donor.exon_seq)} nt) shorter than swap end {end}.",
    )

    new_exon = recipient.exon_seq[:start] + donor.exon_seq[start:end] + recipient.exon_seq[end:]
    return HybridExon(
        exon_seq=new_exon,
        upstream_flank=recipient.upstream_flank,
        downstream_flank=recipient.downstream_flank,
        construct_type=construct_type,
        label=label or f"{recipient.label}<-{donor.label}:int[{start}:{end}]",
        provenance={
            "recipient": recipient.label,
            "donor": donor.label,
            "swap": "internal",
            "start": start,
            "end": end,
        },
    )


def make_full_internal_exon_swap(
    recipient: HybridExon,
    donor: HybridExon,
    *,
    n_exonic_3ss: int = 3,
    n_exonic_5ss: int = 3,
    construct_type: str = "full_internal_swap",
    label: str | None = None,
) -> HybridExon:
    """Swap the **entire internal exon body** from ``donor`` into ``recipient``.

    Partition used in Zhenqi's experiment (transcript-sense 5'->3')::

        3'SS         = upstream_flank[-7:]  +  exon_seq[:n_exonic_3ss]
        internal exon = exon_seq[n_exonic_3ss : -n_exonic_5ss]
        5'SS         = exon_seq[-n_exonic_5ss:]  +  downstream_flank[:7]

    The construct keeps::

        recipient.exon_seq[:n_exonic_3ss]     (exonic 3'SS nt)
        recipient.exon_seq[-n_exonic_5ss:]    (exonic 5'SS nt)
        recipient.upstream_flank / downstream_flank

    and replaces the middle with ``donor.exon_seq[n_exonic_3ss:-n_exonic_5ss]``::

        new_exon = recipient.exon_seq[:n_exonic_3ss]
                 + donor.exon_seq[n_exonic_3ss:-n_exonic_5ss]
                 + recipient.exon_seq[-n_exonic_5ss:]

    Equal exon lengths are **not** required -- the resulting exon may be longer or
    shorter than the recipient. Both the original (recipient) and resulting exon
    lengths are recorded in ``provenance``. Downstream, PNAS instantiates a fresh
    ``PNASModel(input_length=L)`` and AlphaGenome is told the exon length
    explicitly, so a new length is fine.
    """
    _require(n_exonic_3ss >= 0 and n_exonic_5ss >= 0, "SS margin sizes must be >= 0")
    _require(
        recipient.exon_length > n_exonic_3ss + n_exonic_5ss,
        f"recipient exon ({recipient.exon_length} nt) too short for "
        f"{n_exonic_3ss} + {n_exonic_5ss} exonic SS margins.",
    )
    _require(
        len(donor.exon_seq) > n_exonic_3ss + n_exonic_5ss,
        f"donor exon ({len(donor.exon_seq)} nt) too short for "
        f"{n_exonic_3ss} + {n_exonic_5ss} exonic SS margins.",
    )

    head = recipient.exon_seq[:n_exonic_3ss]
    tail = recipient.exon_seq[-n_exonic_5ss:] if n_exonic_5ss else ""
    donor_internal = (
        donor.exon_seq[n_exonic_3ss:-n_exonic_5ss] if n_exonic_5ss
        else donor.exon_seq[n_exonic_3ss:]
    )
    new_exon = head + donor_internal + tail

    return HybridExon(
        exon_seq=new_exon,
        upstream_flank=recipient.upstream_flank,
        downstream_flank=recipient.downstream_flank,
        construct_type=construct_type,
        label=label or f"{recipient.label}<-{donor.label}:internal",
        provenance={
            "recipient": recipient.label,
            "donor": donor.label,
            "swap": "full_internal",
            "n_exonic_3ss": n_exonic_3ss,
            "n_exonic_5ss": n_exonic_5ss,
            "recipient_exon_length": recipient.exon_length,
            "donor_exon_length": len(donor.exon_seq),
            "result_exon_length": len(new_exon),
        },
    )


def make_3ss_swap(
    recipient: HybridExon,
    donor: HybridExon,
    *,
    n_intronic_3ss: int,
    n_exonic_3ss: int,
    construct_type: str = "3ss_swap",
    label: str | None = None,
) -> HybridExon:
    """Swap the 3' splice site (acceptor) region from ``donor`` into ``recipient``.

    The swapped block spans the intron->exon boundary at the exon's 5' end:

    * ``n_intronic_3ss`` nt taken from the *end* of ``donor.upstream_flank``
      (the intronic nt immediately upstream of the exon, incl. the ``AG``).
    * ``n_exonic_3ss`` nt taken from the *start* of ``donor.exon_seq``.

    Flank lengths and exon length are preserved (equal-length replacement), so
    the result still satisfies the OpenSplice 70 / 25 geometry when the parents
    did.
    """
    _require(n_intronic_3ss >= 0 and n_exonic_3ss >= 0, "swap sizes must be >= 0")
    _require(
        n_exonic_3ss <= recipient.exon_length and n_exonic_3ss <= len(donor.exon_seq),
        "n_exonic_3ss exceeds an exon length.",
    )
    _require(
        n_intronic_3ss <= len(recipient.upstream_flank)
        and n_intronic_3ss <= len(donor.upstream_flank),
        "n_intronic_3ss exceeds an upstream flank length.",
    )

    if n_intronic_3ss:
        new_up = recipient.upstream_flank[:-n_intronic_3ss] + donor.upstream_flank[-n_intronic_3ss:]
    else:
        new_up = recipient.upstream_flank

    new_exon = donor.exon_seq[:n_exonic_3ss] + recipient.exon_seq[n_exonic_3ss:]

    return HybridExon(
        exon_seq=new_exon,
        upstream_flank=new_up,
        downstream_flank=recipient.downstream_flank,
        construct_type=construct_type,
        label=label or f"{recipient.label}<-{donor.label}:3ss(i{n_intronic_3ss},e{n_exonic_3ss})",
        provenance={
            "recipient": recipient.label,
            "donor": donor.label,
            "swap": "3ss",
            "n_intronic_3ss": n_intronic_3ss,
            "n_exonic_3ss": n_exonic_3ss,
            # positions (0-based) inside the new construct that came from the donor:
            "swapped_upstream_tail_nt": n_intronic_3ss,
            "swapped_exon_head_nt": n_exonic_3ss,
        },
    )


def make_5ss_swap(
    recipient: HybridExon,
    donor: HybridExon,
    *,
    n_exonic_5ss: int,
    n_intronic_5ss: int,
    construct_type: str = "5ss_swap",
    label: str | None = None,
) -> HybridExon:
    """Swap the 5' splice site (donor) region from ``donor`` into ``recipient``.

    The swapped block spans the exon->intron boundary at the exon's 3' end:

    * ``n_exonic_5ss`` nt taken from the *end* of ``donor.exon_seq``.
    * ``n_intronic_5ss`` nt taken from the *start* of ``donor.downstream_flank``
      (the intronic nt immediately downstream of the exon, incl. the ``GT``).
    """
    _require(n_exonic_5ss >= 0 and n_intronic_5ss >= 0, "swap sizes must be >= 0")
    _require(
        n_exonic_5ss <= recipient.exon_length and n_exonic_5ss <= len(donor.exon_seq),
        "n_exonic_5ss exceeds an exon length.",
    )
    _require(
        n_intronic_5ss <= len(recipient.downstream_flank)
        and n_intronic_5ss <= len(donor.downstream_flank),
        "n_intronic_5ss exceeds a downstream flank length.",
    )

    if n_exonic_5ss:
        new_exon = recipient.exon_seq[:-n_exonic_5ss] + donor.exon_seq[-n_exonic_5ss:]
    else:
        new_exon = recipient.exon_seq

    if n_intronic_5ss:
        new_down = donor.downstream_flank[:n_intronic_5ss] + recipient.downstream_flank[n_intronic_5ss:]
    else:
        new_down = recipient.downstream_flank

    return HybridExon(
        exon_seq=new_exon,
        upstream_flank=recipient.upstream_flank,
        downstream_flank=new_down,
        construct_type=construct_type,
        label=label or f"{recipient.label}<-{donor.label}:5ss(e{n_exonic_5ss},i{n_intronic_5ss})",
        provenance={
            "recipient": recipient.label,
            "donor": donor.label,
            "swap": "5ss",
            "n_exonic_5ss": n_exonic_5ss,
            "n_intronic_5ss": n_intronic_5ss,
            "swapped_exon_tail_nt": n_exonic_5ss,
            "swapped_downstream_head_nt": n_intronic_5ss,
        },
    )


def make_3ss_5ss_swap(
    recipient: HybridExon,
    donor: HybridExon,
    *,
    n_intronic_3ss: int,
    n_exonic_3ss: int,
    n_exonic_5ss: int,
    n_intronic_5ss: int,
    construct_type: str = "3ss_5ss_swap",
    label: str | None = None,
) -> HybridExon:
    """Swap **both** splice-site regions (3'SS and 5'SS) from ``donor``.

    Equivalent to :func:`make_3ss_swap` followed by :func:`make_5ss_swap` with the
    same ``donor`` and ``recipient``: the exon's middle stays recipient-derived at
    unchanged indices, while the intron->exon and exon->intron boundary blocks
    both come from the donor. Exon length is preserved (equal-length replacement
    at each end).
    """
    step = make_3ss_swap(
        recipient, donor,
        n_intronic_3ss=n_intronic_3ss, n_exonic_3ss=n_exonic_3ss,
    )
    step = make_5ss_swap(
        step, donor,
        n_exonic_5ss=n_exonic_5ss, n_intronic_5ss=n_intronic_5ss,
    )
    return HybridExon(
        exon_seq=step.exon_seq,
        upstream_flank=step.upstream_flank,
        downstream_flank=step.downstream_flank,
        construct_type=construct_type,
        label=label or (
            f"{recipient.label}<-{donor.label}:3ss+5ss"
            f"(i{n_intronic_3ss},e{n_exonic_3ss}|e{n_exonic_5ss},i{n_intronic_5ss})"
        ),
        provenance={
            "recipient": recipient.label,
            "donor": donor.label,
            "swap": "3ss+5ss",
            "n_intronic_3ss": n_intronic_3ss,
            "n_exonic_3ss": n_exonic_3ss,
            "n_exonic_5ss": n_exonic_5ss,
            "n_intronic_5ss": n_intronic_5ss,
            "swapped_upstream_tail_nt": n_intronic_3ss,
            "swapped_exon_head_nt": n_exonic_3ss,
            "swapped_exon_tail_nt": n_exonic_5ss,
            "swapped_downstream_head_nt": n_intronic_5ss,
        },
    )


def make_crossover_chimera(
    five_prime: HybridExon,
    three_prime: HybridExon,
    *,
    exon_crossover: int,
    upstream_from: str = "five_prime",
    downstream_from: str = "three_prime",
    construct_type: str = "crossover",
    label: str | None = None,
) -> HybridExon:
    """Chimera: first ``exon_crossover`` nt of ``five_prime`` exon + the rest of
    ``three_prime`` exon.

    ``upstream_from`` / ``downstream_from`` pick which parent supplies each flank
    (``"five_prime"`` or ``"three_prime"``). Exon length becomes
    ``exon_crossover + (len(three_prime.exon_seq) - exon_crossover)`` -- possibly
    new, which is fine: PNAS instantiates a fresh ``PNASModel(input_length=L)``
    and AlphaGenome is told the exon length explicitly.
    """
    _require(
        0 <= exon_crossover <= five_prime.exon_length,
        f"exon_crossover must be in [0, {five_prime.exon_length}].",
    )
    _require(
        exon_crossover <= len(three_prime.exon_seq),
        "exon_crossover exceeds the 3' parent exon length.",
    )
    parents = {"five_prime": five_prime, "three_prime": three_prime}
    _require(upstream_from in parents and downstream_from in parents,
             "upstream_from / downstream_from must be 'five_prime' or 'three_prime'.")

    new_exon = five_prime.exon_seq[:exon_crossover] + three_prime.exon_seq[exon_crossover:]

    return HybridExon(
        exon_seq=new_exon,
        upstream_flank=parents[upstream_from].upstream_flank,
        downstream_flank=parents[downstream_from].downstream_flank,
        construct_type=construct_type,
        label=label or f"{five_prime.label}|{three_prime.label}:x{exon_crossover}",
        provenance={
            "five_prime": five_prime.label,
            "three_prime": three_prime.label,
            "swap": "crossover",
            "exon_crossover": exon_crossover,
            "upstream_from": upstream_from,
            "downstream_from": downstream_from,
        },
    )


# ==========================================================================
# Saturation mutagenesis
# ==========================================================================

def _swapped_flank_spans(hybrid: HybridExon) -> tuple[int, int]:
    """(upstream_tail_nt, downstream_head_nt) that a swap brought in, else (0, 0)."""
    prov = hybrid.provenance or {}
    up = int(prov.get("swapped_upstream_tail_nt", 0) or 0)
    down = int(prov.get("swapped_downstream_head_nt", 0) or 0)
    return up, down


def generate_single_nt_exonic_variants(
    hybrid: HybridExon,
    *,
    positions: Iterable[int] | None = None,
    alphabet: Sequence[str] = _BASES,
    include_reference: bool = True,
    include_swapped_flank_positions: bool = False,
) -> list[dict[str, Any]]:
    """Saturation single-nucleotide mutagenesis over a construct.

    By default every position of ``hybrid.exon_seq`` gets the other 3
    nucleotides; the intronic flanks are left unchanged.

    Set ``include_swapped_flank_positions=True`` to also saturate the intronic
    nt that a 3'ss / 5'ss swap pulled in (recorded in ``provenance`` by
    :func:`make_3ss_swap` / :func:`make_5ss_swap`); those variants mutate the
    flank string and leave ``exon_seq`` unchanged.

    Args:
        hybrid: The construct to mutagenize.
        positions: Optional explicit 0-based indices into ``exon_seq``
            (e.g. a sub-window). Default: all exon-body positions.
        alphabet: Allowed alleles. Default ``("A", "C", "G", "T")``.
        include_reference: Prepend one unmutated row (``ref``/``alt``/``position``
            = ``None``, ``is_reference=True``) so downstream code can compute
            deltas within the same scoring batch.
        include_swapped_flank_positions: Also mutate swapped-in intronic flank nt.

    Returns:
        ``list[dict]``. Every row carries: ``construct_type``, ``construct_label``,
        ``recipient_exon``, ``donor_exon``, ``is_reference``, ``mutation_region``
        (``"reference"`` | ``"exon"`` | ``"upstream_flank"`` |
        ``"downstream_flank"``), ``position`` (1-based in its region),
        ``position0``, ``ref``, ``alt``, ``exon_length``, ``exon_seq``,
        ``upstream_flank``, ``downstream_flank``, the four ``n_*ss`` swap sizes,
        and ``provenance``.
    """
    alphabet = tuple(a.upper() for a in alphabet)
    base = hybrid.base_record()
    rows: list[dict[str, Any]] = []

    if include_reference:
        rows.append({
            **base,
            "is_reference": True,
            "mutation_region": "reference",
            "position": None,
            "position0": None,
            "ref": None,
            "alt": None,
        })

    def _emit(region: str, seq_attr: str, seq: str, idx: int) -> None:
        ref = seq[idx]
        if ref not in _BASES:
            return
        for alt in alphabet:
            if alt == ref:
                continue
            mutated = seq[:idx] + alt + seq[idx + 1:]
            row = {**base,
                   "is_reference": False,
                   "mutation_region": region,
                   "position": idx + 1,
                   "position0": idx,
                   "ref": ref,
                   "alt": alt}
            row[seq_attr] = mutated
            rows.append(row)

    exon_positions = range(hybrid.exon_length) if positions is None else list(positions)
    for idx in exon_positions:
        if not 0 <= idx < hybrid.exon_length:
            raise IndexError(f"exon position {idx} out of range 0..{hybrid.exon_length - 1}")
        _emit("exon", "exon_seq", hybrid.exon_seq, idx)

    if include_swapped_flank_positions:
        up_n, down_n = _swapped_flank_spans(hybrid)
        for k in range(up_n):
            idx = len(hybrid.upstream_flank) - up_n + k
            _emit("upstream_flank", "upstream_flank", hybrid.upstream_flank, idx)
        for idx in range(down_n):
            _emit("downstream_flank", "downstream_flank", hybrid.downstream_flank, idx)

    return rows


# ==========================================================================
# Scoring
# ==========================================================================

def _logit(p: float, eps: float = 1e-7) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def score_hybrid_variants(
    variants: Sequence[Mapping[str, Any]],
    dna_model: Any = None,
    *,
    run_pnas: bool = True,
    run_alphagenome: bool = True,
    require_opensplice_flank_lengths: bool = True,
    ontology_terms: Iterable[str] | None = ("CL:0002518",),
    pnas_kwargs: Mapping[str, Any] | None = None,
    add_deltas: bool = True,
):
    """Score a list of variant records with the PNAS pretuner and AlphaGenome.

    Each record must expose ``exon_seq``, ``upstream_flank`` and
    ``downstream_flank`` (the output of
    :func:`generate_single_nt_exonic_variants`, or hand-built dicts).

    PNAS: all records are scored with a single
    ``pnas.score_pnas_pretuner_batch`` call (internally grouped by final input
    length). AlphaGenome: one ``ag.score_exon`` call per record, reusing that
    module's FAS minigene + padding + splice-site extraction.

    Score columns added
    -------------------
    * ``pnas_pretuner``            -- pre-tuner ``energy_seq_struct`` scalar
    * ``alphagenome_acceptor``     -- +strand SPLICE_SITES prob at the first exon base
    * ``alphagenome_donor``        -- +strand SPLICE_SITES prob at the last exon base
    * ``alphagenome_mean_prob``    -- ``(acceptor + donor) / 2``
    * ``alphagenome_mean_logit``   -- ``(logit(acceptor) + logit(donor)) / 2``
    * ``alphagenome_prod_logit``   -- ``logit(acceptor * donor)`` (matches the
      existing OpenSplice analysis)
    * ``alphagenome_exon_length``, ``alphagenome_acceptor_padded_pos0``,
      ``alphagenome_donor_padded_pos0`` -- provenance passthrough from ``score_exon``

    When ``add_deltas`` is set and a group (keyed by
    ``(construct_type, construct_label)``) contains a reference row, ``*_delta``
    columns are added relative to that reference.

    Args:
        variants: Variant records.
        dna_model: AlphaGenome client from
            ``alphagenome_local_prediction_pipeline_minigene.create_model``.
            If ``None``, AlphaGenome scoring is skipped.
        run_pnas / run_alphagenome: Per-model toggles.
        require_opensplice_flank_lengths: Passed to ``ag.score_exon``; keep
            ``True`` for standard 70 / 25 constructs.
        ontology_terms: Passed to ``ag.score_exon``.
        pnas_kwargs: Extra kwargs for ``pnas.score_pnas_pretuner_batch``
            (``flank_len``, ``temperature``, ``num_threads``, ``weights_path``,
            ``device``).

    Returns:
        ``pandas.DataFrame`` with one row per input variant. Columns are ordered
        sequence / metadata first (``construct_type``, ``construct_label``,
        ``recipient_exon``, ``donor_exon``, ``mutation_region``, ``position``,
        ``exon_seq`` ...) and prediction columns last (``pnas_pretuner`` and, when
        AlphaGenome ran, the ``alphagenome_*`` columns), each followed by its
        ``*_delta`` vs the group reference row.
    """
    import pandas as pd  # lazy: constructors above stay pandas-free

    records = [dict(v) for v in variants]
    if not records:
        return pd.DataFrame()

    for r in records:
        for key in ("exon_seq", "upstream_flank", "downstream_flank"):
            if key not in r:
                raise KeyError(f"variant record missing required field {key!r}")

    # ---- PNAS (batched, grouped by length inside the module) ------------
    if run_pnas:
        pnas_scores = pnas.score_pnas_pretuner_batch(
            [
                (r["exon_seq"], r["upstream_flank"], r["downstream_flank"])
                for r in records
            ],
            **(dict(pnas_kwargs) if pnas_kwargs else {}),
        )
        for r, s in zip(records, pnas_scores):
            r["pnas_pretuner"] = s

    # ---- AlphaGenome (one call per record, reusing score_exon) ---------
    if run_alphagenome and dna_model is not None:
        import alphagenome_local_prediction_pipeline_minigene as ag  # lazy

        for r in records:
            out = ag.score_exon(
                dna_model,
                r["exon_seq"],
                r["upstream_flank"],
                r["downstream_flank"],
                require_opensplice_flank_lengths=require_opensplice_flank_lengths,
                ontology_terms=ontology_terms,
            )
            acc = float(out["acceptor"])
            don = float(out["donor"])
            r["alphagenome_acceptor"] = acc
            r["alphagenome_donor"] = don
            r["alphagenome_mean_prob"] = (acc + don) / 2.0
            r["alphagenome_mean_logit"] = (_logit(acc) + _logit(don)) / 2.0
            r["alphagenome_prod_logit"] = _logit(acc * don)
            r["alphagenome_exon_length"] = out["exon_length"]
            r["alphagenome_acceptor_padded_pos0"] = out["acceptor_padded_pos0"]
            r["alphagenome_donor_padded_pos0"] = out["donor_padded_pos0"]

    df = pd.DataFrame(records)

    # ---- deltas vs the reference row of each group --------------------
    if add_deltas and "is_reference" in df.columns:
        delta_cols = [
            c for c in (
                "pnas_pretuner",
                "alphagenome_acceptor",
                "alphagenome_donor",
                "alphagenome_mean_prob",
                "alphagenome_mean_logit",
                "alphagenome_prod_logit",
            ) if c in df.columns
        ]
        group_keys = [
            k for k in ("construct_type", "construct_label") if k in df.columns
        ]
        is_ref = df["is_reference"].fillna(False).astype(bool)

        def _apply_deltas(idx: "pd.Index") -> None:
            """Fill ``*_delta`` for rows ``idx``, relative to their one ref row."""
            refs = df.loc[idx][is_ref.loc[idx]]
            if len(refs) != 1:
                return
            ref_row = refs.iloc[0]
            for c in delta_cols:
                df.loc[idx, f"{c}_delta"] = df.loc[idx, c] - ref_row[c]

        if delta_cols:
            if not group_keys:
                _apply_deltas(df.index)
            else:
                for _, grp in df.groupby(group_keys, sort=False):
                    _apply_deltas(grp.index)

    return _order_output_columns(df)


# Preferred output column order: sequence / metadata first, predictions last.
_META_COL_ORDER = (
    "construct_type",
    "construct_label",
    "recipient_exon",
    "donor_exon",
    "is_reference",
    "mutation_region",
    "position",
    "position0",
    "ref",
    "alt",
    "exon_length",
    "exon_seq",
    "upstream_flank",
    "downstream_flank",
    "n_intronic_3ss",
    "n_exonic_3ss",
    "n_exonic_5ss",
    "n_intronic_5ss",
    "provenance",
)
_PRED_COL_ORDER = (
    "pnas_pretuner",
    "pnas_pretuner_delta",
    "alphagenome_acceptor",
    "alphagenome_acceptor_delta",
    "alphagenome_donor",
    "alphagenome_donor_delta",
    "alphagenome_mean_prob",
    "alphagenome_mean_prob_delta",
    "alphagenome_mean_logit",
    "alphagenome_mean_logit_delta",
    "alphagenome_prod_logit",
    "alphagenome_prod_logit_delta",
    "alphagenome_exon_length",
    "alphagenome_acceptor_padded_pos0",
    "alphagenome_donor_padded_pos0",
)


def _order_output_columns(df):
    """Reorder columns to (metadata..., predictions...); keep any extras at end."""
    preferred = [c for c in (*_META_COL_ORDER, *_PRED_COL_ORDER) if c in df.columns]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]


# ==========================================================================
# CLI demo  (PNAS-only unless --alphagenome and an API key are available)
# ==========================================================================

def _build_demo_constructs(exon_a_id: str, exon_b_id: str, csv_path: Path) -> list[HybridExon]:
    a = load_opensplice_exon(exon_a_id, csv_path)
    b = load_opensplice_exon(exon_b_id, csv_path)

    constructs = [a, b]  # exon_a WT, exon_b WT
    # Zhenqi's proposed initial swap region (not the OpenSplice region
    # definition): 3'ss = 7 intronic (5 + AG) + 3 exonic; sizes stay
    # configurable via make_3ss_swap / make_5ss_swap.
    constructs.append(make_3ss_swap(a, b, n_intronic_3ss=7, n_exonic_3ss=3))
    # Proposed swap region: 5'ss = 3 exonic + 7 intronic (GT + 5).
    constructs.append(make_5ss_swap(a, b, n_exonic_5ss=3, n_intronic_5ss=7))
    # Both splice-site regions swapped at once (3'ss + 5'ss), same sizes.
    constructs.append(make_3ss_5ss_swap(
        a, b,
        n_intronic_3ss=7, n_exonic_3ss=3,
        n_exonic_5ss=3, n_intronic_5ss=7,
    ))
    # Full internal-exon swap: keep exon_a's 3 exonic nt at each SS, replace the
    # whole middle (exon_seq[3:-3]) with exon_b's. Exon length may change.
    constructs.append(make_full_internal_exon_swap(a, b, n_exonic_3ss=3, n_exonic_5ss=3))
    return constructs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--exon-a", default="TP53_e7", help="recipient / 5' parent exon_id")
    parser.add_argument("--exon-b", default="TP53_e6", help="donor / 3' parent exon_id")
    parser.add_argument("--csv", type=Path, default=DEFAULT_OPENSPLICE_CSV)
    parser.add_argument("--out", type=Path, default=None,
                        help="optional CSV path (created under outputs/ if bare name)")
    parser.add_argument("--alphagenome", action="store_true",
                        help="also run AlphaGenome (needs ALPHA_GENOME_API_KEY)")
    parser.add_argument("--env-path", default="../.env")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    constructs = _build_demo_constructs(args.exon_a, args.exon_b, args.csv)

    variants: list[dict[str, Any]] = []
    for c in constructs:
        # For a full internal-exon swap, only the donor-derived middle is
        # informative -- the first/last exonic SS nt still come from the
        # recipient, so restrict saturation to exon_seq[3:-3].
        if c.construct_type == "full_internal_swap":
            head = int((c.provenance or {}).get("n_exonic_3ss", 3) or 0)
            tail = int((c.provenance or {}).get("n_exonic_5ss", 3) or 0)
            positions: Iterable[int] | None = range(head, c.exon_length - tail)
        else:
            positions = None  # full exon
        variants.extend(generate_single_nt_exonic_variants(c, positions=positions))
    print(f"{len(constructs)} constructs -> {len(variants)} variant rows")

    dna_model = None
    if args.alphagenome:
        import alphagenome_local_prediction_pipeline_minigene as ag  # lazy

        dna_model = ag.create_model(env_path=args.env_path)

    df = score_hybrid_variants(
        variants,
        dna_model,
        run_alphagenome=args.alphagenome,
        pnas_kwargs={"device": args.device} if args.device else None,
    )

    cols = [c for c in ("construct_type", "construct_label", "mutation_region",
                        "position", "ref", "alt", "pnas_pretuner", "pnas_pretuner_delta",
                        "alphagenome_mean_logit", "alphagenome_mean_logit_delta")
            if c in df.columns]
    print(df[cols].head(12).to_string(index=False))

    if args.out is not None:
        out = args.out if args.out.is_absolute() or args.out.parent != Path(".") \
            else DEFAULT_OUTPUT_DIR / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows -> {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
