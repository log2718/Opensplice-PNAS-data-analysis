"""
AlphaGenome minigene splice-site pipeline for OpenSplice-style experiments.

Main use case
-------------
Score an arbitrary target exon in the OpenSplice FAS minigene context:

    70-nt upstream intron
    + target exon
    + 25-nt downstream intron

The full sequence sent to AlphaGenome is:

    FAS exon 5 + FAS intron 5
    + upstream flank + target exon + downstream flank
    + FAS intron 6 + FAS exon 7

The construct is centered in a 16,384-nt sequence padded with Ns, and
the positive-strand SPLICE_SITES probabilities are extracted at the
canonical acceptor (first exon base) and donor (last exon base).

Important
---------
Input sequences are assumed to already be in assay/transcript 5'->3'
orientation, matching the OpenSplice minigene convention.

This file uses the public AlphaGenome API client (`dna_client.create`).
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alphagenome.models import dna_client


# ---------------------------------------------------------------------
# OpenSplice FAS minigene context
# ---------------------------------------------------------------------

FAS_E5 = (
    "ATGTGAACATGGAATCATCAAGGAATGCACACTCACCAGCAACACCAAGTGCAAAGAGGAAG"
)

FAS_I5 = (
    "GTAATTATTTTTTTACGGTTATATTCTCCTTTCCCCCAACCCCATGGAAAGATGTGAAG"
    "AAAAACCAATCACTCTTGATTACTA"
)

FAS_I6 = (
    "CAGATTGAAATAACTTGGGAAGTAGTTTCTCTTAGTGTGAAAGTATGTTCTCACATGCATT"
    "CTACAAGGCTGAGACCTGAGTTGATAAAATTTCTTTGTTCTTTCAG"
)

FAS_E7 = (
    "TGAAGAGAAAGGAAGTACAGAAAACATGCAGAAAGCACAGAAAGGAA"
)

PRE_MANUAL = FAS_E5 + FAS_I5
POST_MANUAL = FAS_I6 + FAS_E7

DEFAULT_UPSTREAM_LEN = 70
DEFAULT_DOWNSTREAM_LEN = 25
TARGET_LEN = 2**14  # 16,384


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------

def create_model(
    api_key: str | None = None,
    env_path: str = "../.env",
    env_var: str = "ALPHA_GENOME_API_KEY",
):
    """
    Create the public AlphaGenome API client.

    If api_key is omitted, load it from env_path. This default assumes
    this file/notebook is run from analysis_script/ and .env is one
    directory above it.
    """
    if api_key is None:
        load_dotenv(env_path)
        api_key = os.environ[env_var]

    return dna_client.create(api_key)


# ---------------------------------------------------------------------
# Sequence / construct helpers
# ---------------------------------------------------------------------

def clean_seq(seq: str) -> str:
    """Upper-case sequence and replace non-ACGTN characters with N."""
    seq = str(seq).upper()
    return "".join(
        base if base in {"A", "C", "G", "T", "N"} else "N"
        for base in seq
    )


def build_variable_region(
    exon_seq: str,
    upstream_flank: str,
    downstream_flank: str,
    *,
    require_opensplice_flank_lengths: bool = True,
) -> str:
    """
    Build:
        upstream flank + exon + downstream flank

    By default, enforce the OpenSplice 70-nt upstream / 25-nt downstream
    flank lengths. Set require_opensplice_flank_lengths=False for custom
    synthetic flank lengths.
    """
    exon_seq = clean_seq(exon_seq)
    upstream_flank = clean_seq(upstream_flank)
    downstream_flank = clean_seq(downstream_flank)

    if require_opensplice_flank_lengths:
        if len(upstream_flank) != DEFAULT_UPSTREAM_LEN:
            raise ValueError(
                f"Expected {DEFAULT_UPSTREAM_LEN}-nt upstream flank, "
                f"got {len(upstream_flank)}."
            )
        if len(downstream_flank) != DEFAULT_DOWNSTREAM_LEN:
            raise ValueError(
                f"Expected {DEFAULT_DOWNSTREAM_LEN}-nt downstream flank, "
                f"got {len(downstream_flank)}."
            )

    if len(exon_seq) == 0:
        raise ValueError("exon_seq must not be empty.")

    return upstream_flank + exon_seq + downstream_flank


def build_fas_minigene(variable_region: str) -> str:
    """Insert a variable region into the fixed FAS minigene context."""
    return clean_seq(PRE_MANUAL + clean_seq(variable_region) + POST_MANUAL)


def center_pad(
    core_seq: str,
    target_len: int = TARGET_LEN,
    pad_base: str = "N",
) -> tuple[str, int, int]:
    """Center a construct in a fixed-length AlphaGenome input."""
    core_seq = clean_seq(core_seq)

    if len(core_seq) > target_len:
        raise ValueError(
            f"Construct length {len(core_seq)} exceeds target length "
            f"{target_len}."
        )

    left_pad = (target_len - len(core_seq)) // 2
    right_pad = target_len - len(core_seq) - left_pad

    padded = pad_base * left_pad + core_seq + pad_base * right_pad
    return padded, left_pad, right_pad


def canonical_construct_positions(
    exon_length: int,
    upstream_len: int = DEFAULT_UPSTREAM_LEN,
) -> tuple[int, int]:
    """
    Return 0-based canonical acceptor and donor positions in the full
    unpadded FAS construct.

    OpenSplice geometry with a 70-nt upstream flank:
        acceptor = len(PRE_MANUAL) + 70 = 216
        donor    = acceptor + exon_length - 1
    """
    if exon_length <= 0:
        raise ValueError("exon_length must be positive.")

    acceptor_pos0 = len(PRE_MANUAL) + upstream_len
    donor_pos0 = acceptor_pos0 + exon_length - 1
    return acceptor_pos0, donor_pos0


# ---------------------------------------------------------------------
# AlphaGenome prediction helpers
# ---------------------------------------------------------------------

def get_positive_strand_tracks(pred_out) -> tuple[np.ndarray, np.ndarray]:
    """Return (donor_track, acceptor_track) for the positive strand."""
    splice_sites = pred_out.splice_sites.filter_to_positive_strand()

    values = np.asarray(splice_sites.values)
    names = [str(name).lower() for name in splice_sites.names]

    donor_idx = names.index("donor")
    acceptor_idx = names.index("acceptor")

    donor = values[:, donor_idx]
    acceptor = values[:, acceptor_idx]
    return donor, acceptor


def score_variable_region(
    model,
    variable_region: str,
    exon_length: int,
    *,
    upstream_len: int = DEFAULT_UPSTREAM_LEN,
    ontology_terms: Iterable[str] | None = ("CL:0002518",),
) -> dict:
    """
    Score canonical acceptor/donor probabilities for one variable region.

    variable_region should be:
        upstream flank + target exon + downstream flank

    exon_length and upstream_len determine where the target exon lies.
    """
    variable_region = clean_seq(variable_region)

    if len(variable_region) < upstream_len + exon_length:
        raise ValueError(
            "variable_region is too short for the supplied upstream_len "
            "and exon_length."
        )

    core = build_fas_minigene(variable_region)
    padded, left_pad, right_pad = center_pad(core)

    acceptor_construct_pos0, donor_construct_pos0 = (
        canonical_construct_positions(
            exon_length=exon_length,
            upstream_len=upstream_len,
        )
    )

    acceptor_padded_pos0 = left_pad + acceptor_construct_pos0
    donor_padded_pos0 = left_pad + donor_construct_pos0

    pred_out = model.predict_sequence(
        sequence=padded,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=[dna_client.OutputType.SPLICE_SITES],
        ontology_terms=list(ontology_terms) if ontology_terms is not None else None,
    )

    donor_track, acceptor_track = get_positive_strand_tracks(pred_out)

    acceptor = float(acceptor_track[acceptor_padded_pos0])
    donor = float(donor_track[donor_padded_pos0])

    return {
        "acceptor": acceptor,
        "donor": donor,
        "exon_length": int(exon_length),
        "upstream_len": int(upstream_len),
        "core_length": len(core),
        "left_pad": left_pad,
        "right_pad": right_pad,
        "acceptor_construct_pos0": acceptor_construct_pos0,
        "donor_construct_pos0": donor_construct_pos0,
        "acceptor_padded_pos0": acceptor_padded_pos0,
        "donor_padded_pos0": donor_padded_pos0,
    }


def score_exon(
    model,
    exon_seq: str,
    upstream_flank: str,
    downstream_flank: str,
    *,
    require_opensplice_flank_lengths: bool = True,
    ontology_terms: Iterable[str] | None = ("CL:0002518",),
) -> dict:
    """
    Convenience interface for scoring an exon with flanks of your choice.

    Example
    -------
    scores = score_exon(
        dna_model,
        exon_seq=my_exon,
        upstream_flank=my_upstream_70nt,
        downstream_flank=my_downstream_25nt,
    )
    """
    exon_seq = clean_seq(exon_seq)
    upstream_flank = clean_seq(upstream_flank)
    downstream_flank = clean_seq(downstream_flank)

    variable_region = build_variable_region(
        exon_seq,
        upstream_flank,
        downstream_flank,
        require_opensplice_flank_lengths=require_opensplice_flank_lengths,
    )

    return score_variable_region(
        model,
        variable_region,
        exon_length=len(exon_seq),
        upstream_len=len(upstream_flank),
        ontology_terms=ontology_terms,
    )


def score_pair(
    model,
    wt_exon_seq: str,
    mut_exon_seq: str,
    upstream_flank: str,
    downstream_flank: str,
    *,
    require_opensplice_flank_lengths: bool = True,
    ontology_terms: Iterable[str] | None = ("CL:0002518",),
) -> dict:
    """
    Score WT and mutant exon sequences in the same flank/minigene context.

    Returns OpenSplice-style WT, mutant, and delta splice-site values.
    """
    wt = score_exon(
        model,
        wt_exon_seq,
        upstream_flank,
        downstream_flank,
        require_opensplice_flank_lengths=require_opensplice_flank_lengths,
        ontology_terms=ontology_terms,
    )

    mut = score_exon(
        model,
        mut_exon_seq,
        upstream_flank,
        downstream_flank,
        require_opensplice_flank_lengths=require_opensplice_flank_lengths,
        ontology_terms=ontology_terms,
    )

    delta_acceptor = mut["acceptor"] - wt["acceptor"]
    delta_donor = mut["donor"] - wt["donor"]

    return {
        "acceptor_wt": wt["acceptor"],
        "donor_wt": wt["donor"],
        "acceptor_mut": mut["acceptor"],
        "donor_mut": mut["donor"],
        "delta_acceptor": delta_acceptor,
        "delta_donor": delta_donor,
        "delta_mean": (delta_acceptor + delta_donor) / 2,
    }


# ---------------------------------------------------------------------
# Optional: all exonic SNVs for one exon
# ---------------------------------------------------------------------

def make_all_exonic_snvs(exon_seq: str) -> list[dict]:
    """
    Generate all 3 substitutions at every exon position.

    Positions are returned 0-based and 1-based.
    """
    exon_seq = clean_seq(exon_seq)
    variants = []

    for pos0, ref in enumerate(exon_seq):
        if ref not in {"A", "C", "G", "T"}:
            continue

        for alt in "ACGT":
            if alt == ref:
                continue

            mut = exon_seq[:pos0] + alt + exon_seq[(pos0 + 1):]

            variants.append(
                {
                    "pos0": pos0,
                    "pos1": pos0 + 1,
                    "ref": ref,
                    "alt": alt,
                    "mut_exon_seq": mut,
                }
            )

    return variants


def score_all_exonic_snvs(
    model,
    exon_seq: str,
    upstream_flank: str,
    downstream_flank: str,
    *,
    require_opensplice_flank_lengths: bool = True,
    ontology_terms: Iterable[str] | None = ("CL:0002518",),
) -> pd.DataFrame:
    """
    Run all possible exonic single-nucleotide substitutions.

    WT is predicted once. Each mutant is then scored in the same FAS
    minigene + flank context.
    """
    exon_seq = clean_seq(exon_seq)
    upstream_flank = clean_seq(upstream_flank)
    downstream_flank = clean_seq(downstream_flank)

    wt = score_exon(
        model,
        exon_seq,
        upstream_flank,
        downstream_flank,
        require_opensplice_flank_lengths=require_opensplice_flank_lengths,
        ontology_terms=ontology_terms,
    )

    rows = []

    for variant in make_all_exonic_snvs(exon_seq):
        mut = score_exon(
            model,
            variant["mut_exon_seq"],
            upstream_flank,
            downstream_flank,
            require_opensplice_flank_lengths=require_opensplice_flank_lengths,
            ontology_terms=ontology_terms,
        )

        delta_acceptor = mut["acceptor"] - wt["acceptor"]
        delta_donor = mut["donor"] - wt["donor"]

        rows.append(
            {
                "pos0": variant["pos0"],
                "pos1": variant["pos1"],
                "ref": variant["ref"],
                "alt": variant["alt"],
                "acceptor_wt": wt["acceptor"],
                "donor_wt": wt["donor"],
                "acceptor_mut": mut["acceptor"],
                "donor_mut": mut["donor"],
                "delta_acceptor": delta_acceptor,
                "delta_donor": delta_donor,
                "delta_mean": (delta_acceptor + delta_donor) / 2,
            }
        )

    return pd.DataFrame(rows)
