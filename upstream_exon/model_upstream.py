"""
PNAS-derived interpretable regression model for upstream-exon (B1) sequence.

The architecture is the original PNAS model's, with exactly three changes, each
forced by the new task:

1. INPUT is the 113-nt upstream exon body, not the middle exon. The position
   axis is therefore redefined: index ``p`` means "p nucleotides into the
   upstream exon body", not "p nucleotides into the exon being scored". For
   that reason the position biases are initialized at ZERO and learned from
   scratch -- resampling the pretrained ones would carry over a prior about a
   different coordinate system (see UPSTREAM_EXON_B1_DESIGN_REVIEW.md section D).
2. OUTPUT is an unbounded AlphaGenome mean-logit, so the final sigmoid is
   dropped. The default head is an affine ``Linear(1, 1)`` on the SumDiff
   scalar, which keeps the prediction monotone in the energy and so preserves
   the pre-tuner's interpretability.
3. LOSS is regression (MSE / Huber), not the Bernoulli KL.

Everything that carries the interpretability is kept verbatim, and two pieces
are imported from the untouched original rather than re-implemented:
``SumDiff`` and ``ResidualTuner`` come from ``PNAS_model.model``.

Structure is OPTIONAL and off by default. With ``use_structure=True`` the
original structure branch is reproduced exactly, including the ``[2:-3]`` crop
that aligns a ``padding='same'`` length-L output with the valid-padded
sequence branch of length L-5.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported, never modified -- same semantics as the published model.
from PNAS_model.model import ResidualTuner, SumDiff  # noqa: E402

logger = logging.getLogger(__name__)

B1_LEN = 113


class UpstreamExonModel(nn.Module):
    """Interpretable B1-sequence -> AlphaGenome mean-logit regressor.

    Forward pass (sequence-only default)::

        x_seq (N, 4, L)
          -> conv_incl / conv_skip : Conv1d(4 -> F_seq, k=6, valid)   (N, F, L-5)
          -> + position_bias_incl / position_bias_skip                (F, L-5)
          -> Softplus
          -> SumDiff : w * (sum(incl) - sum(skip)) + b                (N,)
          -> head                                                      (N,)

    The SumDiff scalar is exposed as the ``energy`` (the "pre-tuner") so that
    filter activations remain readable as additive contributions to it.
    """

    def __init__(self, input_length: int = B1_LEN, *,
                 seq_in_channels: int = 4,
                 struct_in_channels: int = 3,
                 wobble_in_channels: int = 1,
                 num_seq_filters: int = 20,
                 num_struct_filters: int = 8,
                 seq_kernel_size: int = 6,
                 struct_kernel_size: int = 30,
                 use_structure: bool = False,
                 head: str = "linear",
                 tuner_hidden: int = 4,
                 use_batchnorm: bool = False,
                 init_energy_scale: float = 1e-3):
        super().__init__()
        if head not in ("linear", "residual_tuner", "identity"):
            raise ValueError(f"unknown head {head!r}")
        self.input_length = input_length
        self.use_structure = use_structure
        self.head_kind = head
        self.seq_kernel_size = seq_kernel_size
        self.struct_kernel_size = struct_kernel_size
        self.num_seq_filters = num_seq_filters
        self.num_struct_filters = num_struct_filters

        conv_out = input_length - seq_kernel_size + 1
        if conv_out < 1:
            raise ValueError(
                f"input_length={input_length} is shorter than the "
                f"{seq_kernel_size}-nt sequence kernel")
        self.conv_out_length = conv_out

        # --- sequence branch (always on) ---
        self.conv_incl = nn.Conv1d(seq_in_channels, num_seq_filters,
                                   seq_kernel_size, padding=0)
        self.conv_skip = nn.Conv1d(seq_in_channels, num_seq_filters,
                                   seq_kernel_size, padding=0)
        self.position_bias_incl = nn.Parameter(
            torch.zeros(num_seq_filters, conv_out))
        self.position_bias_skip = nn.Parameter(
            torch.zeros(num_seq_filters, conv_out))

        # --- structure branch (optional) ---
        if use_structure:
            if input_length < struct_kernel_size:
                raise ValueError(
                    f"input_length={input_length} is shorter than the "
                    f"{struct_kernel_size}-nt structure kernel")
            total_ch = seq_in_channels + struct_in_channels + wobble_in_channels
            self.conv_struct_incl = nn.Conv1d(total_ch, num_struct_filters,
                                              struct_kernel_size,
                                              padding="same")
            self.conv_struct_skip = nn.Conv1d(total_ch, num_struct_filters,
                                              struct_kernel_size,
                                              padding="same")
            self.position_bias_incl_struct = nn.Parameter(
                torch.zeros(num_struct_filters, input_length))
            self.position_bias_skip_struct = nn.Parameter(
                torch.zeros(num_struct_filters, input_length))

        self.energy_activation_incl = nn.Softplus()
        self.energy_activation_skip = nn.Softplus()
        self.energy_seq_struct = SumDiff()
        # SumDiff adds ~2 * F * (L-5) softplus terms, so a N(0,1) weight makes
        # the untrained energy enormous and the regression loss unoptimizable.
        # Start small; the head and training recover the needed scale.
        with torch.no_grad():
            self.energy_seq_struct.w.fill_(init_energy_scale)
            self.energy_seq_struct.b.zero_()

        if head == "linear":
            self.head = nn.Linear(1, 1)
            with torch.no_grad():  # start as the identity on the energy
                self.head.weight.fill_(1.0)
                self.head.bias.zero_()
        elif head == "residual_tuner":
            self.head = ResidualTuner(hidden_units=tuner_hidden,
                                      use_batchnorm=use_batchnorm)
        else:
            self.head = nn.Identity()

        logger.info(
            "UpstreamExonModel: L=%d, structure=%s, head=%s, params=%d",
            input_length, use_structure, head,
            sum(p.numel() for p in self.parameters()))

    # ------------------------------------------------------------------
    def activations(self, x_seq, x_struct=None, x_wobble=None):
        """Softplus filter activations for the inclusion and skipping arms."""
        incl = self.conv_incl(x_seq) + self.position_bias_incl.unsqueeze(0)
        skip = self.conv_skip(x_seq) + self.position_bias_skip.unsqueeze(0)

        if not self.use_structure:
            return (self.energy_activation_incl(incl),
                    self.energy_activation_skip(skip))

        if x_struct is None or x_wobble is None:
            raise ValueError(
                "use_structure=True requires x_struct and x_wobble")
        struct_input = torch.cat([x_seq, x_struct, x_wobble], dim=1)
        s_incl = (self.conv_struct_incl(struct_input)
                  + self.position_bias_incl_struct.unsqueeze(0))
        s_skip = (self.conv_struct_skip(struct_input)
                  + self.position_bias_skip_struct.unsqueeze(0))
        # same crop as the original, aligning 'same' padding to valid padding
        s_incl = s_incl[:, :, 2:-3]
        s_skip = s_skip[:, :, 2:-3]
        return (self.energy_activation_incl(torch.cat([incl, s_incl], dim=1)),
                self.energy_activation_skip(torch.cat([skip, s_skip], dim=1)))

    def energy(self, x_seq, x_struct=None, x_wobble=None):
        """The SumDiff scalar -- this model's analogue of the pre-tuner."""
        a_incl, a_skip = self.activations(x_seq, x_struct, x_wobble)
        return self.energy_seq_struct(torch.stack([a_incl, a_skip], dim=1))

    def forward(self, x_seq, x_struct=None, x_wobble=None,
                return_energy: bool = False):
        e = self.energy(x_seq, x_struct, x_wobble)          # (N,)
        if self.head_kind == "identity":
            y = e
        else:
            y = self.head(e.unsqueeze(-1)).squeeze(-1)      # (N,)
        return (y, e) if return_energy else y

    @torch.no_grad()
    def calibrate_head_(self, x_seq, y, x_struct=None, x_wobble=None):
        """One-shot least-squares fit of the affine head at initialization.

        Puts the untrained model at roughly the target's mean and scale, so
        the first gradient step is informative instead of recovering from a
        huge offset. Only the head is touched; filters and position biases
        stay at their random / zero init.
        """
        if self.head_kind != "linear":
            return self
        e = self.energy(x_seq, x_struct, x_wobble).flatten().double()
        t = torch.as_tensor(y, dtype=torch.float64, device=e.device).flatten()
        var = torch.var(e)
        if torch.isfinite(var) and var > 1e-12:
            slope = torch.cov(torch.stack([e, t]))[0, 1] / var
        else:
            slope = torch.zeros((), dtype=torch.float64, device=e.device)
        intercept = t.mean() - slope * e.mean()
        self.head.weight.fill_(float(slope))
        self.head.bias.fill_(float(intercept))
        logger.info("calibrate_head_: slope=%.6g intercept=%.6g",
                    float(slope), float(intercept))
        return self

    # ------------------------------------------------------------------
    def sr_balance(self, x_seq, agg: str = "mean"):
        """Sequence-only inclusion-minus-skipping score, as in the original."""
        incl = F.softplus(self.conv_incl(x_seq)
                          + self.position_bias_incl.unsqueeze(0))
        skip = F.softplus(self.conv_skip(x_seq)
                          + self.position_bias_skip.unsqueeze(0))
        red = torch.mean if agg == "mean" else torch.sum
        return red(incl, dim=2).sum(1) - red(skip, dim=2).sum(1)

    def l1_filters(self) -> torch.Tensor:
        """L1 over sequence filter weights (sparser, more readable motifs)."""
        return self.conv_incl.weight.abs().sum() + self.conv_skip.weight.abs().sum()

    def l2_position_smoothness(self) -> torch.Tensor:
        """Squared first differences of every position-bias tensor."""
        loss = torch.zeros((), device=self.position_bias_incl.device)
        for name, p in self.named_parameters():
            if "position_bias" in name and p.dim() == 2 and p.shape[-1] > 1:
                loss = loss + ((p[:, 1:] - p[:, :-1]) ** 2).sum()
        return loss

    def describe(self) -> dict:
        return {
            "input_length": self.input_length,
            "conv_out_length": self.conv_out_length,
            "use_structure": self.use_structure,
            "head": self.head_kind,
            "num_seq_filters": self.num_seq_filters,
            "num_struct_filters": (self.num_struct_filters
                                   if self.use_structure else 0),
            "n_params": sum(p.numel() for p in self.parameters()),
            "n_trainable": sum(p.numel() for p in self.parameters()
                               if p.requires_grad),
        }
