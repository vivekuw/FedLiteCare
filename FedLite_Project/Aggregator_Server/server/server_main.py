"""Command-line entrypoint for local FedLiteCare aggregation rounds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Aggregator_Server.server.global_model_manager import (
    DEFAULT_CONFIG_PATH,
    run_aggregation_round,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one FedLiteCare aggregation round.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the aggregator server config file.",
    )
    parser.add_argument("--hospital-a-model", type=Path, default=None, help="Optional Hospital A model override.")
    parser.add_argument("--hospital-b-model", type=Path, default=None, help="Optional Hospital B model override.")
    parser.add_argument("--hospital-c-model", type=Path, default=None, help="Optional Hospital C model override.")
    args = parser.parse_args()

    result = run_aggregation_round(
        config_path=args.config,
        hospital_model_overrides={
            "Hospital_A": args.hospital_a_model,
            "Hospital_B": args.hospital_b_model,
            "Hospital_C": args.hospital_c_model,
        },
    )

    print(f"Completed round: {result['round_name']}")
    for hospital_name, update_paths in result["received_updates"].items():
        print(f"{hospital_name} update stored at: {update_paths['received_path']}")
    print(f"Versioned global model saved to: {result['global_model_path']}")
    print(f"Latest global model saved to: {result['latest_model_path']}")
    print(f"Version history updated: {result['version_history_path']}")
    print(f"Round state updated: {result['round_state_path']}")
    print(f"Aggregator log updated: {result['log_path']}")


if __name__ == "__main__":
    main()
