"""Hospital C local PyTorch training entrypoint."""

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
    parser = argparse.ArgumentParser(description="Train the Hospital C diabetes classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Hospital C config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional CSV filename inside the uploads directory.",
    )
    args = parser.parse_args()

    result = train_local_model(config_path=args.config, dataset_filename=args.dataset)
    print(f"Hospital: {result['hospital_name']}")
    print(f"Dataset loaded from: {result['dataset_path']}")
    print(f"Training device: {result['device']}")
    print(f"Rows used - train: {result['training_rows']}, validation: {result['validation_rows']}")
    print(f"Validation loss: {result['validation_loss']:.4f}")
    print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
    print(f"Model saved to: {result['model_path']}")
    print(f"Training log updated: {result['log_path']}")


if __name__ == "__main__":
    main()
