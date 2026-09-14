"""
Reusable PNAS *pretuner* scoring for hybrid / synthetic exon sequences.

Scope of this module: **sequence in -> pretuner scalar out.**
No CSV paths, no dataset filtering, no minigene / FAS logic, no plotting.

Established repo conventions reproduced here
-------------------------------------------
* OpenSplice ``nt_seq`` = 70 nt upstream intron + exon + 25 nt downstream intron.
* The PNAS pretuner input used across the OpenSplice analysis is::

      upstream_flank[-7:] + exon_seq + downstream_flank[:7]

  which, for the full 70 / 25 OpenSplice flanks, is exactly ``nt_seq[63:-18]``.
* Features are built with ``create_input_data(..., add_flanks=False,
  temperature=37.0)`` -- the fixed PNAS vector flanks are *not* added.
* "pretuner" is the pre-tuner ``energy_seq_struct(...)`` scalar, i.e. the exact
  forward-pass semantics of ``analysis_script/predict_pretuner.py:get_pretuner``
  (weighted sum-difference of softplus inclusion/skipping energies, before the
  ``ResidualTuner`` and the output sigmoid).
* Each distinct final input length ``L`` needs its own
  ``PNASModel(input_length=L)``; the checkpoint's position-bias tensors are
  resized by the existing ``PNASModel.load_state_dict`` (Lanczos) logic.

Public API
----------
* :func:`build_pnas_input`            -- biological triple -> 7+exon+7 string
* :func:`score_pnas_pretuner`         -- one hybrid exon -> pretuner float
* :func:`score_pnas_pretuner_batch`   -- many hybrid exons -> list[float]
* :data:`DEFAULT_WEIGHTS`, :data:`DEFAULT_FLANK_LEN`, :data:`DEFAULT_TEMPERATURE`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import sys

import numpy as np
import torch

# --------------------------------------------------------------------------
# Make ``PNAS_model`` importable no matter the working directory, exactly the
# way analysis_script/predict_pretuner.py does it.
# --------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PNAS_model.model import PNASModel          # noqa: E402
from PNAS_model.utils import create_input_data  # noqa: E402


DEFAULT_WEIGHTS = _ROOT / "PNAS_model" / "model_weights.pt"
DEFAULT_FLANK_LEN = 7
DEFAULT_TEMPERATURE = 37.0

_ALLOWED_BASES = set("ACGTU")

# Cache one loaded model per (input_length, weights_path, device). Inference
# only; identical to instantiating + load_state_dict + eval on every call, but
# avoids re-reading the checkpoint and re-resampling the position biases.
_MODEL_CACHE: dict[tuple[int, str, str], PNASModel] = {}

__all__ = [
    "DEFAULT_WEIGHTS",
    "DEFAULT_FLANK_LEN",
    "DEFAULT_TEMPERATURE",
    "build_pnas_input",
    "score_pnas_pretuner",
    "score_pnas_pretuner_batch",
]


# --------------------------------------------------------------------------
# Sequence construction
# --------------------------------------------------------------------------

def _norm(seq: str, *, name: str) -> str:
    """Upper-case, strip whitespace, and sanity-check the alphabet."""
    out = "".join(str(seq).split()).upper()
    bad = sorted(set(out) - _ALLOWED_BASES)
    if bad:
        raise ValueError(
            f"{name} contains non-ACGTU characters {bad}. "
            "Sequences must be plain nucleotides in transcript 5'->3' orientation."
        )
    return out


def build_pnas_input(
    exon_seq: str,
    upstream_flank: str,
    downstream_flank: str,
    flank_len: int = DEFAULT_FLANK_LEN,
) -> str:
    """Build the PNAS pretuner input string for one hybrid exon.

    Returns ``upstream_flank[-flank_len:] + exon_seq + downstream_flank[:flank_len]``.

    The flanks may carry the *full* OpenSplice context (70 nt upstream /
    25 nt downstream) or more; only the ``flank_len`` nt closest to each
    splice junction are used. Orientation is assumed transcript-sense 5'->3'
    and is never changed here.

    Args:
        exon_seq: Exon body, 5'->3'. Must be non-empty.
        upstream_flank: Upstream (intronic) context, 5'->3'. Needs >= ``flank_len`` nt.
        downstream_flank: Downstream (intronic) context, 5'->3'. Needs >= ``flank_len`` nt.
        flank_len: Number of intronic nt kept on each side. Default 7
            (the OpenSplice convention: 5 intronic nt + the AG / GT dinucleotide).

    Raises:
        ValueError: If ``flank_len`` is not positive, the exon is empty, or
            either flank is shorter than ``flank_len``.
    """
    if flank_len <= 0:
        raise ValueError(f"flank_len must be positive, got {flank_len}.")

    exon_seq = _norm(exon_seq, name="exon_seq")
    upstream_flank = _norm(upstream_flank, name="upstream_flank")
    downstream_flank = _norm(downstream_flank, name="downstream_flank")

    if not exon_seq:
        raise ValueError("exon_seq must not be empty.")
    if len(upstream_flank) < flank_len:
        raise ValueError(
            f"upstream_flank has {len(upstream_flank)} nt; need >= {flank_len}."
        )
    if len(downstream_flank) < flank_len:
        raise ValueError(
            f"downstream_flank has {len(downstream_flank)} nt; need >= {flank_len}."
        )

    return upstream_flank[-flank_len:] + exon_seq + downstream_flank[:flank_len]


# --------------------------------------------------------------------------
# Model loading + pretuner forward pass
# --------------------------------------------------------------------------

def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_model(
    input_length: int,
    weights_path: str | Path,
    device: torch.device,
) -> PNASModel:
    """Instantiate + load a ``PNASModel`` for a given input length (cached).

    Mirrors ``analysis_script/predict_pretuner.py``: load the checkpoint on CPU,
    take ``model_state_dict`` if present, then ``load_state_dict`` (which
    Lanczos-resamples the position biases when ``input_length != 90``).
    """
    key = (int(input_length), str(Path(weights_path)), str(device))
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    model = PNASModel(input_length=int(input_length))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _MODEL_CACHE[key] = model
    return model


@torch.no_grad()
def _pretuner_forward(
    model: PNASModel,
    x_seq: torch.Tensor,
    x_struct: torch.Tensor,
    x_wobble: torch.Tensor,
) -> torch.Tensor:
    """Pre-tuner energy, byte-for-byte equivalent to
    ``analysis_script/predict_pretuner.py:get_pretuner``.

    Returns a tensor of shape ``(batch_size,)``.
    """
    # Sequence activations
    conv_skip_out = model.conv_skip(x_seq) + model.position_bias_skip.unsqueeze(0)
    conv_incl_out = model.conv_incl(x_seq) + model.position_bias_incl.unsqueeze(0)

    # Structure activations
    struct_input = torch.cat([x_seq, x_struct, x_wobble], dim=1)
    conv_struct_skip_out = (
        model.conv_struct_skip(struct_input)
        + model.position_bias_skip_struct.unsqueeze(0)
    )
    conv_struct_incl_out = (
        model.conv_struct_incl(struct_input)
        + model.position_bias_incl_struct.unsqueeze(0)
    )

    # Crop structure activations to match the valid-padding sequence conv output
    conv_struct_skip_out = conv_struct_skip_out[:, :, 2:-3]
    conv_struct_incl_out = conv_struct_incl_out[:, :, 2:-3]

    activations_skip = model.energy_activation_skip(
        torch.cat([conv_skip_out, conv_struct_skip_out], dim=1)
    )
    activations_incl = model.energy_activation_incl(
        torch.cat([conv_incl_out, conv_struct_incl_out], dim=1)
    )

    energy_in = torch.stack([activations_incl, activations_skip], dim=1)
    energy_out = model.energy_seq_struct(energy_in)  # (batch_size,)
    return energy_out


# --------------------------------------------------------------------------
# Public scoring entry points
# --------------------------------------------------------------------------

def _as_triple(record: Any) -> tuple[str, str, str]:
    """Coerce a record to ``(exon_seq, upstream_flank, downstream_flank)``.

    Accepts a mapping with those keys or a 3-item sequence in that order.
    """
    if isinstance(record, Mapping):
        return (
            record["exon_seq"],
            record["upstream_flank"],
            record["downstream_flank"],
        )
    if isinstance(record, Sequence) and not isinstance(record, str):
        if len(record) != 3:
            raise ValueError(
                "Sequence records must be (exon_seq, upstream_flank, downstream_flank)."
            )
        return tuple(record)  # type: ignore[return-value]
    raise TypeError(
        "Each record must be a mapping with exon_seq/upstream_flank/"
        "downstream_flank keys or a 3-item (exon, upstream, downstream) sequence."
    )


def score_pnas_pretuner_batch(
    records: Iterable[Any],
    *,
    flank_len: int = DEFAULT_FLANK_LEN,
    temperature: float = DEFAULT_TEMPERATURE,
    num_threads: int = 8,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    device: str | torch.device | None = None,
) -> list[float]:
    """Score many hybrid exons, returning pretuner values in input order.

    Sequences are grouped by their *final* PNAS input length
    (``flank_len + len(exon) + flank_len``); each length group is one-hot
    encoded, folded, and forwarded as a single batch through the matching
    ``PNASModel(input_length=L)``. Only equal-length sequences are ever
    batched together (``utils.one_hot_batch`` requires this).

    Args:
        records: Iterable of ``(exon_seq, upstream_flank, downstream_flank)``
            tuples or mappings with those keys.
        flank_len: Intronic nt kept per side (see :func:`build_pnas_input`).
        temperature: RNAfold folding temperature (Celsius). Keep at 37.0 for
            consistency with the existing OpenSplice analysis.
        num_threads: RNAfold worker threads.
        weights_path: PNAS checkpoint. Defaults to ``PNAS_model/model_weights.pt``.
        device: Torch device string/obj, or ``None`` to auto-select CUDA/CPU.

    Returns:
        ``list[float]`` of pretuner scalars, aligned to ``records``.
    """
    triples = [_as_triple(r) for r in records]
    seqs = [
        build_pnas_input(exon, up, down, flank_len=flank_len)
        for (exon, up, down) in triples
    ]
    if not seqs:
        return []

    device = _resolve_device(device)
    results: list[float | None] = [None] * len(seqs)

    groups: dict[int, list[int]] = {}
    for idx, seq in enumerate(seqs):
        groups.setdefault(len(seq), []).append(idx)

    for input_length, idxs in groups.items():
        group_seqs = [seqs[i] for i in idxs]

        seq_oh, struct_oh, wobbles = create_input_data(
            group_seqs,
            add_flanks=False,
            temperature=temperature,
            num_threads=num_threads,
        )

        x_seq = torch.tensor(seq_oh, dtype=torch.float32, device=device)
        x_struct = torch.tensor(struct_oh, dtype=torch.float32, device=device)
        x_wobble = torch.tensor(wobbles, dtype=torch.float32, device=device)

        model = _get_model(input_length, weights_path, device)
        pretuner = _pretuner_forward(model, x_seq, x_struct, x_wobble)
        pretuner = pretuner.reshape(-1).detach().cpu().numpy()

        for local_i, global_i in enumerate(idxs):
            results[global_i] = float(pretuner[local_i])

    # No None can remain: every index was assigned in its length group.
    return [float(v) for v in results]  # type: ignore[arg-type]


def score_pnas_pretuner(
    exon_seq: str,
    upstream_flank: str,
    downstream_flank: str,
    *,
    flank_len: int = DEFAULT_FLANK_LEN,
    temperature: float = DEFAULT_TEMPERATURE,
    num_threads: int = 8,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    device: str | torch.device | None = None,
) -> float:
    """Score one hybrid exon and return its pretuner scalar.

    Thin wrapper over :func:`score_pnas_pretuner_batch`.

    Steps: build ``upstream_flank[-7:] + exon_seq + downstream_flank[:7]``,
    call ``create_input_data(..., add_flanks=False, temperature=37.0)``,
    instantiate ``PNASModel(input_length=L)``, load
    ``PNAS_model/model_weights.pt``, and return the pre-tuner
    ``energy_seq_struct`` value.
    """
    (value,) = score_pnas_pretuner_batch(
        [(exon_seq, upstream_flank, downstream_flank)],
        flank_len=flank_len,
        temperature=temperature,
        num_threads=num_threads,
        weights_path=weights_path,
        device=device,
    )
    return value


if __name__ == "__main__":  # pragma: no cover - tiny smoke check
    # Minimal self-test: two synthetic exons of different lengths.
    demo = [
        ("ACGTACGTACGTACGTACGTACGT", "TTTTTCTCCTAG", "GTCAGGATTTTT"),
        ("ACGTACGTACGTACGTACGTACGTAAAA", "TTTTTCTCCTAG", "GTCAGGATTTTT"),
    ]
    print("input lengths:",
          [len(build_pnas_input(*d)) for d in demo])
    print("pretuner:", score_pnas_pretuner_batch(demo, device="cpu"))
