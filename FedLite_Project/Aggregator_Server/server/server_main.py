"""Command-line entrypoint for the full local FedLiteCare federated simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Aggregator_Server.server.global_model_manager import (
    DEFAULT_CONFIG_PATH,
    run_distributed_federated_round,
    run_full_federated_round,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one FedLiteCare federated round.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the aggregator server config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="distributed",
        choices=["distributed", "single-process"],
        help="Choose 'distributed' for the 4-terminal demo or 'single-process' for the old one-terminal flow.",
    )
    parser.add_argument(
        "--no-confirm-wait",
        action="store_true",
        help="Do not wait for manual Enter confirmation before distributing the global model.",
    )
    parser.add_argument(
        "--startup-delay-seconds",
        type=int,
        default=None,
        help="Optional startup delay before distributing the global model in distributed mode.",
    )
    args = parser.parse_args()

    print("FedLiteCare Local Federated Learning Simulation")
    print("LTX Transfer Backbone on 127.0.0.1")
    print("------------------------------------------------")
    if args.mode == "distributed":
        print("Mode: distributed 4-terminal demo")
        result = run_distributed_federated_round(
            config_path=args.config,
            progress_callback=print,
            wait_for_hospital_confirmation=False if args.no_confirm_wait else None,
            startup_delay_seconds=args.startup_delay_seconds,
        )
    else:
        print("Mode: single-process fallback")
        result = run_full_federated_round(config_path=args.config, progress_callback=print)
    print("------------------------------------------------")

    print(f"Completed round: {result['round_name']}")
    print(f"Starting global model: {result['current_global_model_path']}")
    if args.mode == "single-process":
        for hospital_name, training_result in result["hospital_training_results"].items():
            print(
                f"{hospital_name} local update ready at: {training_result['local_update_path']}"
            )
    for hospital_name, update_paths in result["received_updates"].items():
        print(f"{hospital_name} update collected by aggregator at: {update_paths['received_path']}")
    print(f"Versioned global model saved to: {result['global_model_path']}")
    print(f"Latest global model saved to: {result['latest_model_path']}")
    print(f"Version history updated: {result['version_history_path']}")
    print(f"Round state updated: {result['round_state_path']}")
    if args.mode == "distributed":
        print(f"Aggregator runtime log updated: {result['node_log_path']}")
    print(f"Aggregator log updated: {result['aggregation_log_path']}")
    print(f"Round log updated: {result['round_log_path']}")
    print("Hospital validation summary:")
    for hospital_name, hospital_summary in result["hospital_update_summaries"].items():
        accuracy = hospital_summary["validation_accuracy"]
        accuracy_text = "N/A" if accuracy is None else f"{accuracy:.4f}"
        loss = hospital_summary["validation_loss"]
        loss_text = "N/A" if loss is None else f"{loss:.4f}"
        print(
            f"{hospital_name}: status={hospital_summary['validation_status']} | "
            f"accuracy={accuracy_text} | loss={loss_text}"
        )
    print(f"Demo summary text exported to: {result['demo_summary_text_path']}")
    print(f"Demo summary JSON exported to: {result['demo_summary_json_path']}")


if __name__ == "__main__":
    main()
