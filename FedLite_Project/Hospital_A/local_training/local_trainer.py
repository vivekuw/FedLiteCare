"""Local PyTorch training pipeline for Hospital A diabetes prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    ensure_directory,
    load_simple_yaml_config,
    resolve_path,
)
from FedLite_Project.Shared_Assets.data_preprocessing_helpers.preprocessing_utils import (
    fit_preprocessor,
    load_csv_records,
    split_tensor_dataset,
)
from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import (
    DiabetesClassifier,
    save_checkpoint,
    train_classifier,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "client_config.yaml"


def _resolve_training_paths(settings: dict[str, Any], config_path: Path) -> dict[str, Path]:
    hospital_root = config_path.parent.parent
    uploads_dir = resolve_path(hospital_root, str(settings["uploads_dir"]))
    models_dir = ensure_directory(resolve_path(hospital_root, str(settings["models_dir"])))
    return {"hospital_root": hospital_root, "uploads_dir": uploads_dir, "models_dir": models_dir}


def train_local_model(
    config_path: Path = DEFAULT_CONFIG_PATH,
    dataset_filename: str | None = None,
) -> dict[str, Any]:
    """Train a lightweight local model from a CSV file in Hospital_A/uploads."""
    settings = load_simple_yaml_config(config_path)
    paths = _resolve_training_paths(settings, config_path)

    seed = int(settings.get("random_seed", 42))
    torch.manual_seed(seed)

    dataset_name = dataset_filename or str(settings["dataset_filename"])
    dataset_path = paths["uploads_dir"] / dataset_name
    model_path = paths["models_dir"] / str(settings["model_filename"])

    records = load_csv_records(dataset_path)
    features, labels, preprocessing = fit_preprocessor(
        records,
        target_column=str(settings.get("target_column", "Outcome")),
    )
    train_dataset, validation_dataset = split_tensor_dataset(
        features,
        labels,
        validation_ratio=float(settings.get("validation_ratio", 0.25)),
        seed=seed,
    )

    batch_size = int(settings.get("batch_size", 8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = DiabetesClassifier(
        input_dim=features.shape[1],
        hidden_dim=int(settings.get("hidden_dim", 16)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, history, validation_metrics = train_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=int(settings.get("epochs", 80)),
        learning_rate=float(settings.get("learning_rate", 0.001)),
        device=device,
    )

    save_checkpoint(
        checkpoint_path=model_path,
        model=trained_model,
        preprocessing=preprocessing,
        model_config={
            "input_dim": features.shape[1],
            "hidden_dim": int(settings.get("hidden_dim", 16)),
        },
        training_metrics=validation_metrics,
        history=history,
    )

    return {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "device": str(device),
        "training_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Hospital A diabetes classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hospital A config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional CSV filename inside the uploads directory.",
    )
    args = parser.parse_args()

    result = train_local_model(config_path=args.config, dataset_filename=args.dataset)
    print(f"Dataset loaded from: {result['dataset_path']}")
    print(f"Training device: {result['device']}")
    print(f"Rows used - train: {result['training_rows']}, validation: {result['validation_rows']}")
    print(f"Validation loss: {result['validation_loss']:.4f}")
    print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    print(f"Model saved to: {result['model_path']}")


if __name__ == "__main__":
    main()
