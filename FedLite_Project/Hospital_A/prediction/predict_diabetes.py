"""Hospital A local diabetes prediction entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    predict_from_csv,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "client_config.yaml"


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

    result = predict_from_csv(config_path=args.config, input_path=args.input, model_path=args.model)
    print(f"Hospital: {result['hospital_name']}")
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
    print(f"Prediction log updated: {result['log_path']}")


if __name__ == "__main__":
    main()
