"""Local prediction pipeline for Hospital A diabetes inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    load_simple_yaml_config,
    resolve_path,
)
from FedLite_Project.Shared_Assets.data_preprocessing_helpers.preprocessing_utils import (
    load_csv_records,
    transform_records,
)
from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import (
    load_checkpoint,
    predict_probabilities,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "client_config.yaml"


def _resolve_prediction_paths(settings: dict[str, Any], config_path: Path) -> dict[str, Path]:
    hospital_root = config_path.parent.parent
    uploads_dir = resolve_path(hospital_root, str(settings["uploads_dir"]))
    models_dir = resolve_path(hospital_root, str(settings["models_dir"]))
    return {"hospital_root": hospital_root, "uploads_dir": uploads_dir, "models_dir": models_dir}


def predict_from_csv(
    input_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Generate diabetes predictions from a local CSV file."""
    settings = load_simple_yaml_config(config_path)
    paths = _resolve_prediction_paths(settings, config_path)

    resolved_input_path = input_path or (paths["uploads_dir"] / str(settings["dataset_filename"]))
    resolved_model_path = model_path or (paths["models_dir"] / str(settings["model_filename"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(resolved_model_path, device)

    records = load_csv_records(resolved_input_path)
    features, labels = transform_records(records, checkpoint["preprocessing"])
    probabilities = predict_probabilities(model, features, device)
    threshold = 0.5

    prediction_rows: list[dict[str, Any]] = []
    for index, probability in enumerate(probabilities.tolist(), start=1):
        row_result: dict[str, Any] = {
            "row": index,
            "probability": probability,
            "predicted_label": int(probability >= threshold),
        }
        if labels is not None:
            row_result["actual_label"] = int(labels[index - 1].item())
        prediction_rows.append(row_result)

    accuracy = None
    if labels is not None:
        predicted_tensor = torch.tensor(
            [row["predicted_label"] for row in prediction_rows], dtype=torch.float32
        ).unsqueeze(1)
        accuracy = float((predicted_tensor == labels).float().mean().item())

    return {
        "input_path": resolved_input_path,
        "model_path": resolved_model_path,
        "predictions": prediction_rows,
        "accuracy": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hospital A diabetes predictions from a CSV file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hospital A config file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV file to score. Defaults to the configured uploads dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional model checkpoint path. Defaults to the configured models path.",
    )
    args = parser.parse_args()

    result = predict_from_csv(input_path=args.input, config_path=args.config, model_path=args.model)
    print(f"Model loaded from: {result['model_path']}")
    print(f"Input file: {result['input_path']}")
    if result["accuracy"] is not None:
        print(f"Accuracy on input file: {result['accuracy']:.4f}")

    for row in result["predictions"]:
        line = (
            f"Row {row['row']}: probability={row['probability']:.4f}, "
            f"predicted_label={row['predicted_label']}"
        )
        if "actual_label" in row:
            line += f", actual_label={row['actual_label']}"
        print(line)


if __name__ == "__main__":
    main()
