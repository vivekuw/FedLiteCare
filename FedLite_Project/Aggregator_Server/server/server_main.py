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
    run_full_federated_round,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full FedLiteCare local federated round.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the aggregator server config file.",
    )
    args = parser.parse_args()

    print("FedLiteCare Local Federated Learning Simulation")
    print("LTX Transfer Backbone on 127.0.0.1")
    print("------------------------------------------------")
    result = run_full_federated_round(config_path=args.config, progress_callback=print)
    print("------------------------------------------------")

    print(f"Completed round: {result['round_name']}")
    print(f"Starting global model: {result['current_global_model_path']}")
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
    print(f"Aggregator log updated: {result['aggregation_log_path']}")
    print(f"Round log updated: {result['round_log_path']}")


if __name__ == "__main__":
    main()
