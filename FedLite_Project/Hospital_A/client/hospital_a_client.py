"""Convenience CLI wrapper for Hospital A local ML workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Hospital_A.local_training.local_trainer import (
    DEFAULT_CONFIG_PATH as TRAIN_CONFIG_PATH,
    train_local_model,
)
from FedLite_Project.Hospital_A.communication.ltx_transfer import (
    receive_global_model_via_ltx,
    send_local_update_via_ltx,
)
from FedLite_Project.Hospital_A.prediction.predict_diabetes import (
    DEFAULT_CONFIG_PATH as PREDICT_CONFIG_PATH,
    predict_from_csv,
)
from FedLite_Project.Shared_Assets.common_utilities.federated_hospital_node import (
    run_hospital_federated_round,
)
from FedLite_Project.Shared_Assets.common_utilities.hospital_quality_reports import (
    validate_training_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hospital A local training and prediction commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the local diabetes model.")
    train_parser.add_argument("--config", type=Path, default=TRAIN_CONFIG_PATH, help="Optional config path.")
    train_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional CSV filename inside Hospital_A/uploads.",
    )

    federated_parser = subparsers.add_parser(
        "federated-round",
        help="Receive the global model, train locally, and send the update back.",
    )
    federated_parser.add_argument(
        "--config",
        type=Path,
        default=TRAIN_CONFIG_PATH,
        help="Optional config path.",
    )
    federated_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional CSV filename inside Hospital_A/uploads.",
    )

    validate_parser = subparsers.add_parser("validate-dataset", help="Validate a training CSV before model training.")
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=TRAIN_CONFIG_PATH,
        help="Optional config path.",
    )
    validate_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional CSV filename inside Hospital_A/uploads.",
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
        print(f"Validation status: {result['validation_result']['status']}")
        print(f"Validation report: {result['validation_report_path']}")
        print(f"Model saved to: {result['model_path']}")
        print(f"Validation accuracy: {result['validation_accuracy']:.4f}")
        print(f"Training log updated: {result['log_path']}")
        return

    if args.command == "federated-round":
        result = run_hospital_federated_round(
            config_path=args.config,
            dataset_filename=args.dataset,
            receive_global_model_callable=receive_global_model_via_ltx,
            send_local_update_callable=send_local_update_via_ltx,
            progress_callback=print,
        )
        print(f"Hospital: {result['hospital_name']}")
        print(f"Completed round: {result['round_name']}")
        print(f"Base global model round: {result['base_round_name']}")
        print(f"Cached global model path: {result['current_global_model_path']}")
        print(f"Validation status: {result['training_result']['validation_result']['status']}")
        print(f"Validation report: {result['training_result']['validation_report_path']}")
        print(f"Local model saved to: {result['training_result']['model_path']}")
        print(f"Local update sent from: {result['training_result']['local_update_path']}")
        print(f"Refreshed global model saved to: {result['refreshed_global_model_path']}")
        print(f"Validation accuracy: {result['training_result']['validation_accuracy']:.4f}")
        print(f"Hospital runtime log updated: {result['runtime_log_path']}")
        print(f"Transfer log updated: {result['transfer_log_path']}")
        return

    if args.command == "validate-dataset":
        result = validate_training_dataset(config_path=args.config, dataset_filename=args.dataset)
        print(f"Hospital: {result['hospital_name']}")
        print(f"Dataset path: {result['dataset_path']}")
        print(f"Validation status: {result['status']}")
        print(f"Validation report: {result['report_path']}")
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
