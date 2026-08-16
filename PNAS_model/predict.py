"""Run pretrained PNAS predictions with optional bias-only calibration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from calibration import BiasCalibrationResult, collect_logits, tune_output_bias_
from model import PNASModel
from utils import create_input_data

logger = logging.getLogger(__name__)

FEATURE_KEYS = ("seq_oh", "struct_oh", "wobbles")


def _load_npz(
    path: str | Path,
    *,
    optional_keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in FEATURE_KEYS if key not in archive]
        if missing:
            raise ValueError(
                f"Dataset {path} is missing required fields: {', '.join(missing)}."
            )
        requested_keys = set(FEATURE_KEYS) | set(optional_keys)
        data = {
            key: archive[key]
            for key in requested_keys
            if key in archive
        }

    n_examples = len(data["seq_oh"])
    for key in FEATURE_KEYS[1:]:
        if len(data[key]) != n_examples:
            raise ValueError(
                f"Dataset field {key!r} has {len(data[key])} examples; "
                f"expected {n_examples}."
            )
    return data


def _feature_loader(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    target_key: str | None = None,
    weight_key: str | None = None,
) -> DataLoader:
    tensors = [torch.as_tensor(data[key]) for key in FEATURE_KEYS]
    if target_key is not None:
        if target_key not in data:
            raise ValueError(
                f"Calibration dataset does not contain target field {target_key!r}."
            )
        tensors.append(torch.as_tensor(data[target_key], dtype=torch.float32))
    if weight_key is not None:
        if target_key is None:
            raise ValueError("A weight field can only be used with a target field.")
        if weight_key not in data:
            raise ValueError(
                f"Calibration dataset does not contain weight field {weight_key!r}."
            )
        tensors.append(torch.as_tensor(data[weight_key], dtype=torch.float32))

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _load_model(
    checkpoint_path: str | Path,
    *,
    input_length: int,
    use_batchnorm: bool,
    device: torch.device,
) -> PNASModel:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint must be a state dict or contain a 'model_state_dict' mapping."
        )

    model = PNASModel(
        input_length=input_length,
        use_batchnorm=use_batchnorm,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(
    df: pd.DataFrame,
    *,
    calibrate: bool = False,
    model: PNASModel | None = None,
    checkpoint: str | Path = "model_weights.pt",
    batch_size: int = 512,
    device: torch.device | str | None = None,
    use_batchnorm: bool = True,
    sequence_column: str = "flanked_sequence",
    target_column: str = "PSI",
    rnafold_bin: str = "RNAfold",
    temperature: float = 37.0,
    max_bp_span: int = 0,
    commands_file: str = "",
    num_threads: int = 8,
    max_abs_bias_delta: float = 20.0,
) -> pd.DataFrame:
    """Predict PSI directly from a dataframe of already-flanked sequences.

    The sequences are one-hot encoded and folded with RNAfold using
    :func:`utils.create_input_data`. Because ``sequence_column`` contains the
    complete flanked model input, no additional flanks are added.

    When ``calibrate`` is true, ``target_column`` must contain measured PSI for
    every row. One output-logit bias is fitted over the entire dataframe by
    minimizing PSI-space RMSE, then used for all returned predictions. The
    supplied model's original bias is restored before this function returns.

    Args:
        df: Input dataframe containing a ``flanked_sequence`` column by default.
        calibrate: Fit the final dense-layer bias using this dataframe's PSI.
        model: Optional preloaded model. If omitted, ``checkpoint`` is loaded.
        checkpoint: Raw or training-style model checkpoint used when ``model``
            is omitted.
        batch_size: Batch size for calibration and prediction.
        device: Torch device. CUDA is selected automatically when available.
        use_batchnorm: ResidualTuner configuration when loading a model.
        sequence_column: Dataframe column containing complete flanked sequences.
        target_column: Dataframe column containing measured PSI.
        rnafold_bin: Executable name or path for ViennaRNA RNAfold.
        temperature: RNAfold temperature in Celsius.
        max_bp_span: Optional RNAfold maximum base-pair span.
        commands_file: Optional ViennaRNA commands file.
        num_threads: Number of RNAfold worker threads.
        max_abs_bias_delta: Absolute bound for the fitted logit offset.

    Returns:
        A copy of ``df`` with a ``predicted_PSI`` column. Calibrated calls also
        include ``predicted_PSI_uncalibrated`` and store the calibration report
        in ``result.attrs["bias_calibration"]``.

    Raises:
        ValueError: If required columns or values are missing, the dataframe is
            empty, sequences have inconsistent lengths, or the supplied model
            has an incompatible input length.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if sequence_column not in df.columns:
        raise ValueError(
            f"Input dataframe must contain a {sequence_column!r} column."
        )
    if df.empty:
        raise ValueError("Cannot predict an empty dataframe.")
    if df[sequence_column].isna().any():
        raise ValueError(f"Column {sequence_column!r} cannot contain missing values.")
    if calibrate and target_column not in df.columns:
        raise ValueError(
            f"calibrate=True requires a {target_column!r} column containing true PSI."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    targets: np.ndarray | None = None
    if calibrate:
        targets = pd.to_numeric(df[target_column], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(targets).all():
            raise ValueError(
                f"Column {target_column!r} must contain finite numeric PSI values."
            )
        if np.any((targets < 0) | (targets > 1)):
            raise ValueError(
                f"Column {target_column!r} must contain PSI values in [0, 1]."
            )

    sequences = df[sequence_column].astype(str).tolist()
    sequence_lengths = {len(sequence) for sequence in sequences}
    if len(sequence_lengths) != 1:
        raise ValueError("All flanked sequences must have the same length.")
    input_length = sequence_lengths.pop()
    if input_length <= 0:
        raise ValueError("Flanked sequences cannot be empty.")

    seq_oh, struct_oh, wobbles = create_input_data(
        sequences,
        add_flanks=False,
        rnafold_bin=rnafold_bin,
        temperature=temperature,
        maxBPspan=max_bp_span,
        commands_file=commands_file,
        num_threads=num_threads,
    )
    feature_data = {
        "seq_oh": seq_oh,
        "struct_oh": struct_oh,
        "wobbles": wobbles,
    }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if model is None:
        model = _load_model(
            checkpoint,
            input_length=input_length,
            use_batchnorm=use_batchnorm,
            device=device,
        )
    elif model.input_length != input_length:
        raise ValueError(
            f"Supplied model expects input length {model.input_length}, but "
            f"{sequence_column!r} contains length-{input_length} sequences."
        )
    else:
        model.to(device)

    prediction_loader = _feature_loader(
        feature_data,
        batch_size=batch_size,
    )
    calibration_result: BiasCalibrationResult | None = None
    original_bias = model.tuner.fc3.bias.detach().clone()

    try:
        if calibrate:
            assert targets is not None
            calibration_data = dict(feature_data)
            calibration_data[target_column] = targets
            calibration_loader = _feature_loader(
                calibration_data,
                batch_size=batch_size,
                target_key=target_column,
            )
            calibration_result = tune_output_bias_(
                model,
                calibration_loader,
                device=device,
                max_abs_delta=max_abs_bias_delta,
            )

        final_logits = collect_logits(model, prediction_loader, device=device)
    finally:
        with torch.no_grad():
            model.tuner.fc3.bias.copy_(original_bias)

    bias_delta = (
        calibration_result.bias_delta if calibration_result is not None else 0.0
    )
    raw_predictions = torch.sigmoid(final_logits - bias_delta).numpy()
    final_predictions = torch.sigmoid(final_logits).numpy()

    result_df = df.copy()
    if calibration_result is not None:
        result_df["predicted_PSI_uncalibrated"] = raw_predictions
        result_df.attrs["bias_calibration"] = asdict(calibration_result)
    result_df["predicted_PSI"] = final_predictions
    return result_df


def _write_predictions(
    output_path: str | Path,
    data: dict[str, np.ndarray],
    raw_predictions: np.ndarray,
    final_predictions: np.ndarray,
    *,
    target_key: str,
    calibrated: bool,
) -> Path:
    columns: dict[str, np.ndarray] = {}
    if "exon" in data:
        columns["exon"] = data["exon"]
    if target_key in data:
        columns[target_key] = data[target_key]
    if calibrated:
        columns["prediction_raw"] = raw_predictions
        columns["prediction_calibrated"] = final_predictions
    columns["prediction"] = final_predictions

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the prediction command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict PSI with a pretrained PNAS model, optionally fitting only "
            "the final dense-layer bias on a labeled dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="NPZ dataset to predict.")
    parser.add_argument("--checkpoint", required=True, help="Pretrained model checkpoint.")
    parser.add_argument("--output", required=True, help="Output prediction CSV.")
    parser.add_argument(
        "--tune-bias",
        action="store_true",
        help="Fit the output bias using labels in the prediction dataset.",
    )
    parser.add_argument(
        "--calibration-dataset",
        default=None,
        metavar="NPZ",
        help=(
            "Fit the output bias on this labeled NPZ before prediction. Providing "
            "this option implies bias tuning and overrides --tune-bias."
        ),
    )
    parser.add_argument(
        "--target-key",
        default="metadata_PSI",
        help="NPZ field containing calibration targets.",
    )
    parser.add_argument(
        "--weight-key",
        default=None,
        help="Optional NPZ field containing non-negative calibration weights.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help="Model input length. Inferred from the prediction dataset if omitted.",
    )
    parser.add_argument(
        "--max-abs-bias-delta",
        type=float,
        default=20.0,
        help="Absolute bound on the fitted logit offset.",
    )
    parser.add_argument(
        "--no-batchnorm",
        action="store_true",
        help="Instantiate ResidualTuner without BatchNorm layers.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cpu, cuda, or cuda:1. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--calibration-report",
        default=None,
        metavar="JSON",
        help=(
            "Calibration report path. By default, a '.calibration.json' sidecar "
            "is written next to the output CSV when calibration is enabled."
        ),
    )
    parser.add_argument(
        "--save-calibrated-checkpoint",
        default=None,
        metavar="PATH",
        help="Optionally save the calibrated model as a separate checkpoint.",
    )
    return parser


def main() -> None:
    """Run prediction and optional bias-only calibration."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    calibration_enabled = bool(args.tune_bias or args.calibration_dataset)
    if args.weight_key and not calibration_enabled:
        raise ValueError("--weight-key requires bias calibration.")
    if args.calibration_report and not calibration_enabled:
        raise ValueError("--calibration-report requires bias calibration.")
    if args.save_calibrated_checkpoint and not calibration_enabled:
        raise ValueError(
            "--save-calibrated-checkpoint requires bias calibration."
        )
    calibration_path = (
        args.calibration_dataset
        if args.calibration_dataset is not None
        else (args.dataset if args.tune_bias else None)
    )
    report_path = (
        Path(
            args.calibration_report
            if args.calibration_report is not None
            else f"{args.output}.calibration.json"
        )
        if calibration_enabled
        else None
    )

    input_paths = {
        Path(args.dataset).resolve(),
        Path(args.checkpoint).resolve(),
    }
    if calibration_path is not None:
        input_paths.add(Path(calibration_path).resolve())
    generated_paths = {Path(args.output).resolve()}
    if report_path is not None:
        generated_paths.add(report_path.resolve())
    if args.save_calibrated_checkpoint is not None:
        generated_paths.add(Path(args.save_calibrated_checkpoint).resolve())

    if input_paths & generated_paths:
        raise ValueError(
            "Output, report, and calibrated-checkpoint paths must not overwrite "
            "an input dataset or the pretrained checkpoint."
        )
    expected_generated_count = 1 + int(report_path is not None) + int(
        args.save_calibrated_checkpoint is not None
    )
    if len(generated_paths) != expected_generated_count:
        raise ValueError(
            "Output, report, and calibrated-checkpoint paths must be distinct."
        )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    optional_dataset_keys = ["exon", args.target_key]
    if args.weight_key is not None:
        optional_dataset_keys.append(args.weight_key)
    prediction_data = _load_npz(
        args.dataset,
        optional_keys=tuple(optional_dataset_keys),
    )
    inferred_length = int(prediction_data["seq_oh"].shape[-1])
    input_length = (
        inferred_length if args.input_length is None else args.input_length
    )
    if input_length <= 0:
        raise ValueError("--input-length must be greater than zero.")
    if input_length != inferred_length:
        raise ValueError(
            f"Prediction data length is {inferred_length}, but model input length "
            f"is {input_length}."
        )

    logger.info("Loading checkpoint %s on %s.", args.checkpoint, device)
    model = _load_model(
        args.checkpoint,
        input_length=input_length,
        use_batchnorm=not args.no_batchnorm,
        device=device,
    )

    calibration_result: BiasCalibrationResult | None = None
    if calibration_path is not None:
        calibration_data = (
            prediction_data
            if Path(calibration_path).resolve() == Path(args.dataset).resolve()
            else _load_npz(
                calibration_path,
                optional_keys=tuple(
                    key
                    for key in (args.target_key, args.weight_key)
                    if key is not None
                ),
            )
        )
        calibration_length = int(calibration_data["seq_oh"].shape[-1])
        if calibration_length != input_length:
            raise ValueError(
                f"Calibration data length is {calibration_length}, but model "
                f"input length is {input_length}."
            )
        calibration_loader = _feature_loader(
            calibration_data,
            batch_size=args.batch_size,
            target_key=args.target_key,
            weight_key=args.weight_key,
        )
        calibration_result = tune_output_bias_(
            model,
            calibration_loader,
            device=device,
            max_abs_delta=args.max_abs_bias_delta,
        )
        logger.info(
            "Fitted output bias on %d examples: delta=%+.8f, "
            "RMSE %.6f -> %.6f, weighted mean prediction %.6f -> %.6f "
            "(target %.6f).",
            calibration_result.n_examples,
            calibration_result.bias_delta,
            calibration_result.rmse_before,
            calibration_result.rmse_after,
            calibration_result.raw_prediction_mean,
            calibration_result.calibrated_prediction_mean,
            calibration_result.target_mean,
        )
        if calibration_result.at_bound:
            logger.warning(
                "The fitted bias reached the configured bound of +/-%.3f.",
                args.max_abs_bias_delta,
            )

    prediction_loader = _feature_loader(
        prediction_data,
        batch_size=args.batch_size,
    )
    final_logits = collect_logits(model, prediction_loader, device=device)
    bias_delta = (
        calibration_result.bias_delta if calibration_result is not None else 0.0
    )
    raw_predictions = torch.sigmoid(final_logits - bias_delta).numpy()
    final_predictions = torch.sigmoid(final_logits).numpy()
    output_path = _write_predictions(
        args.output,
        prediction_data,
        raw_predictions,
        final_predictions,
        target_key=args.target_key,
        calibrated=calibration_result is not None,
    )
    logger.info("Saved %d predictions to %s.", len(final_predictions), output_path)

    if calibration_result is not None:
        assert report_path is not None
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(calibration_result), handle, indent=2)
            handle.write("\n")
        logger.info("Saved calibration report to %s.", report_path)

        if args.save_calibrated_checkpoint is not None:
            calibrated_checkpoint_path = Path(args.save_calibrated_checkpoint)
            calibrated_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "calibration": asdict(calibration_result),
                    "base_checkpoint": str(Path(args.checkpoint)),
                },
                calibrated_checkpoint_path,
            )
            logger.info(
                "Saved calibrated checkpoint to %s.",
                calibrated_checkpoint_path,
            )


if __name__ == "__main__":
    main()
