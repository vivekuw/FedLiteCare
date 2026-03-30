"""Tkinter GUI launcher for Hospital C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Hospital_C.communication.ltx_transfer import (
    receive_global_model_via_ltx,
    send_local_update_via_ltx,
)
from FedLite_Project.Shared_Assets.common_utilities.tkinter_hospital_app import (
    launch_hospital_gui,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "client_config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Hospital C FedLiteCare desktop GUI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Optional Hospital C config path.",
    )
    args = parser.parse_args()

    launch_hospital_gui(
        config_path=args.config,
        receive_global_model_callable=receive_global_model_via_ltx,
        send_local_update_callable=send_local_update_via_ltx,
    )


if __name__ == "__main__":
    main()
