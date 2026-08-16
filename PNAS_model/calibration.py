"""Bias-only calibration helpers for pretrained PNAS models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy import optimize
from scipy.special import expit
import torch

from model import PNASModel


@dataclass(frozen=True)
class BiasCalibrationResult:
    """Summary of a fitted output-logit bias."""

    original_bias: float
    bias_delta: float
    calibrated_bias: float
    n_examples: int
    target_mean: float
    raw_prediction_mean: float
    calibrated_prediction_mean: float
    rmse_before: float
    rmse_after: float
    at_bound: bool


def _as_vector(
    value: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a detached, one-dimensional CPU tensor."""
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(-1)


def _validate_calibration_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor | None,
) -> None:
    if logits.numel() == 0:
        raise ValueError("Cannot calibrate output bias on an empty dataset.")
    if logits.shape != targets.shape:
        raise ValueError(
            "Logits and targets must have the same number of elements, got "
            f"{logits.numel()} and {targets.numel()}."
        )
    if not torch.isfinite(logits).all():
        raise ValueError("Calibration logits must all be finite.")
    if not torch.isfinite(targets).all():
        raise ValueError("Calibration targets must all be finite.")
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("Calibration targets must lie in the interval [0, 1].")

    if sample_weight is not None:
        if sample_weight.shape != targets.shape:
            raise ValueError(
                "Sample weights and targets must have the same number of "
                f"elements, got {sample_weight.numel()} and {targets.numel()}."
            )
        if not torch.isfinite(sample_weight).all():
            raise ValueError("Calibration sample weights must all be finite.")
        if torch.any(sample_weight < 0):
            raise ValueError("Calibration sample weights cannot be negative.")
        if sample_weight.sum() <= 0:
            raise ValueError("Calibration sample weights must have a positive sum.")


def fit_logit_bias(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    max_abs_delta: float = 20.0,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> tuple[float, bool]:
    """Fit a constant logit offset by minimizing probability-space RMSE.

    This follows the basal-shift fitting used in the original analysis: add a
    scalar to every pretrained logit, apply the sigmoid, and minimize prediction
    error against measured PSI. SciPy's bounded scalar optimizer is used because
    there is only one parameter.

    Args:
        logits: Frozen, uncalibrated model logits.
        targets: PSI targets in the interval ``[0, 1]``.
        sample_weight: Optional non-negative weight per example.
        max_abs_delta: Absolute bound for the fitted offset.
        tolerance: Absolute tolerance for the scalar optimizer.
        max_iterations: Maximum scalar-optimizer iterations.

    Returns:
        A pair ``(delta, at_bound)``. ``at_bound`` is true when the fitted
        optimum reaches the configured search boundary.

    Raises:
        RuntimeError: If SciPy does not successfully complete the fit.
    """
    if not math.isfinite(max_abs_delta) or max_abs_delta <= 0:
        raise ValueError("max_abs_delta must be finite and greater than zero.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and greater than zero.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero.")

    logits = _as_vector(logits)
    targets = _as_vector(targets)
    weights = (
        torch.ones_like(targets)
        if sample_weight is None
        else _as_vector(sample_weight)
    )
    _validate_calibration_inputs(
        logits,
        targets,
        None if sample_weight is None else weights,
    )
    logits_array = logits.numpy()
    targets_array = targets.numpy()
    weights_array = weights.numpy()
    weight_sum = weights_array.sum()

    def objective(delta: float) -> float:
        probabilities = expit(logits_array + delta)
        squared_errors = (probabilities - targets_array) ** 2
        return float(np.sqrt(np.sum(weights_array * squared_errors) / weight_sum))

    result = optimize.minimize_scalar(
        objective,
        bounds=(-float(max_abs_delta), float(max_abs_delta)),
        method="bounded",
        options={
            "xatol": tolerance,
            "maxiter": max_iterations,
        },
    )
    if not result.success or not math.isfinite(result.x):
        raise RuntimeError(f"Output-bias optimization failed: {result.message}")

    delta = float(result.x)
    boundary_tolerance = max(10 * tolerance, 1e-6)
    at_bound = abs(delta) >= max_abs_delta - boundary_tolerance
    return delta, at_bound


def _weighted_rmse(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
) -> float:
    squared_errors = (probabilities - targets) ** 2
    weighted_mse = (squared_errors * sample_weight).sum() / sample_weight.sum()
    return torch.sqrt(weighted_mse).item()


def _unpack_calibration_batch(
    batch: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(batch, Sequence) or len(batch) not in (4, 5):
        raise ValueError(
            "Each calibration batch must contain "
            "(sequence, structure, wobble, target) and may optionally include "
            "a fifth sample-weight tensor."
        )
    seq, struct, wobble, target = batch[:4]
    weight = batch[4] if len(batch) == 5 else None
    return seq, struct, wobble, target, weight


@torch.inference_mode()
def collect_logits(
    model: PNASModel,
    loader: Iterable[Sequence[torch.Tensor]],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute CPU logits for a loader whose first three tensors are inputs."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)

    was_training = model.training
    model.eval()
    logits_parts: list[torch.Tensor] = []
    try:
        for batch in loader:
            if not isinstance(batch, Sequence) or len(batch) < 3:
                raise ValueError(
                    "Each prediction batch must begin with sequence, structure, "
                    "and wobble tensors."
                )
            seq, struct, wobble = batch[:3]
            logits = model(
                seq.to(device),
                struct.to(device),
                wobble.to(device),
                return_logits=True,
            )
            logits_parts.append(logits.detach().reshape(-1).cpu())
    finally:
        model.train(was_training)

    if not logits_parts:
        raise ValueError("Cannot compute predictions for an empty dataset.")
    return torch.cat(logits_parts)


@torch.inference_mode()
def tune_output_bias_(
    model: PNASModel,
    calibration_loader: Iterable[Sequence[torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    max_abs_delta: float = 20.0,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> BiasCalibrationResult:
    """Fit and apply only ``model.tuner.fc3.bias`` on labeled data.

    Calibration batches must contain ``(sequence, structure, wobble, target)``
    and may optionally contain a fifth sample-weight tensor. The model is
    evaluated with frozen BatchNorm statistics, and its prior training/eval
    mode is restored afterward.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)

    was_training = model.training
    model.eval()
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    any_weight = False
    any_unweighted = False

    try:
        for batch in calibration_loader:
            seq, struct, wobble, target, weight = _unpack_calibration_batch(batch)
            logits = model(
                seq.to(device),
                struct.to(device),
                wobble.to(device),
                return_logits=True,
            )
            logits_parts.append(logits.detach().reshape(-1).cpu())
            target_parts.append(target.detach().reshape(-1).cpu())
            if weight is None:
                any_unweighted = True
            else:
                any_weight = True
                weight_parts.append(weight.detach().reshape(-1).cpu())
    finally:
        model.train(was_training)

    if not logits_parts:
        raise ValueError("Cannot calibrate output bias on an empty dataset.")
    if any_weight and any_unweighted:
        raise ValueError(
            "Calibration batches must either all provide sample weights or all "
            "omit them."
        )

    logits = _as_vector(torch.cat(logits_parts))
    targets = _as_vector(torch.cat(target_parts))
    weights = (
        _as_vector(torch.cat(weight_parts))
        if any_weight
        else torch.ones_like(targets)
    )
    _validate_calibration_inputs(
        logits,
        targets,
        weights if any_weight else None,
    )

    delta, at_bound = fit_logit_bias(
        logits,
        targets,
        sample_weight=weights if any_weight else None,
        max_abs_delta=max_abs_delta,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    original_bias = model.tuner.fc3.bias.detach().item()
    model.shift_output_bias_(delta)
    calibrated_bias = model.tuner.fc3.bias.detach().item()
    applied_delta = calibrated_bias - original_bias
    calibrated_logits = logits + applied_delta
    raw_probabilities = torch.sigmoid(logits)
    calibrated_probabilities = torch.sigmoid(calibrated_logits)
    rmse_before = _weighted_rmse(raw_probabilities, targets, weights)
    rmse_after = _weighted_rmse(calibrated_probabilities, targets, weights)

    return BiasCalibrationResult(
        original_bias=original_bias,
        bias_delta=applied_delta,
        calibrated_bias=calibrated_bias,
        n_examples=targets.numel(),
        target_mean=((targets * weights).sum() / weights.sum()).item(),
        raw_prediction_mean=(
            (raw_probabilities * weights).sum() / weights.sum()
        ).item(),
        calibrated_prediction_mean=(
            (calibrated_probabilities * weights).sum() / weights.sum()
        ).item(),
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        at_bound=at_bound,
    )
