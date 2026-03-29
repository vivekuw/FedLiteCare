"""Hospital A local PyTorch training entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    train_local_model,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "client_config.yaml"

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
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="Optional starting global model checkpoint path.",
    )
    parser.add_argument(
        "--local-update-path",
        type=Path,
        default=None,
        help="Optional output path for a round-specific local update checkpoint.",
    )
    parser.add_argument(
        "--round-name",
        type=str,
        default=None,
        help="Optional federated round name, such as round_001.",
    )
    args = parser.parse_args()

    result = train_local_model(
        config_path=args.config,
        dataset_filename=args.dataset,
        initial_model_path=args.initial_model,
        local_update_path=args.local_update_path,
        round_name=args.round_name,
    )
    print(f"Hospital: {result['hospital_name']}")
    print(f"Dataset loaded from: {result['dataset_path']}")
    if result["initial_model_path"] is not None:
        print(f"Starting from global model: {result['initial_model_path']}")
    print(f"Training device: {result['device']}")
    print(f"Rows used - train: {result['training_rows']}, validation: {result['validation_rows']}")
    print(f"Validation loss: {result['validation_loss']:.4f}")
    print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    print(f"Model saved to: {result['model_path']}")
    if result["local_update_path"] is not None:
        print(f"Local update saved to: {result['local_update_path']}")
    print(f"Training log updated: {result['log_path']}")


if __name__ == "__main__":
    main()
