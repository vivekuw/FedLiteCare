"""Convenience CLI wrapper for Hospital B local ML workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Hospital_B.local_training.local_trainer import (
    DEFAULT_CONFIG_PATH as TRAIN_CONFIG_PATH,
    train_local_model,
)
from FedLite_Project.Hospital_B.prediction.predict_diabetes import (
    DEFAULT_CONFIG_PATH as PREDICT_CONFIG_PATH,
    predict_from_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hospital B local training and prediction commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the local diabetes model.")
    train_parser.add_argument("--config", type=Path, default=TRAIN_CONFIG_PATH, help="Optional config path.")
    train_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional CSV filename inside Hospital_B/uploads.",
    )

    predict_parser = subparsers.add_parser("predict", help="Run predictions from a CSV file.")
    predict_parser.add_argument(
        "--config",
        type=Path,
        default=PREDICT_CONFIG_PATH,
        help="Optional config path.",
    )
    predict_parser.add_argument("--input", type=Path, default=None, help="Optional CSV path to score.")
    predict_parser.add_argument("--model", type=Path, default=None, help="Optional checkpoint path.")

    args = parser.parse_args()

    if args.command == "train":
        result = train_local_model(config_path=args.config, dataset_filename=args.dataset)
        print(f"Hospital: {result['hospital_name']}")
        print(f"Model saved to: {result['model_path']}")
        print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
        print(f"Training log updated: {result['log_path']}")
        return

    result = predict_from_csv(
        config_path=args.config,
        input_path=args.input,
        model_path=args.model,
    )
    print(f"Hospital: {result['hospital_name']}")
    print(f"Scored file: {result['input_path']}")
    for row in result["predictions"]:
        print(
            f"Row {row['row']}: probability={row['probability']:.4f}, "
            f"predicted_label={row['predicted_label']}"
        )
    print(f"Prediction log updated: {result['log_path']}")


if __name__ == "__main__":
    main()
