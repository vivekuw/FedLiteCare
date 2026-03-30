"""Packaging-friendly GUI entrypoint for the Hospital A desktop client."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
for candidate in (CURRENT_FILE.parent, *CURRENT_FILE.parents):
    package_parent = candidate / "FedLite_Project"
    if package_parent.is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from FedLite_Project.Shared_Assets.common_utilities.runtime_paths import (
    bootstrap_project_imports,
)

bootstrap_project_imports(CURRENT_FILE)

from FedLite_Project.Hospital_A.communication.ltx_transfer import (  # noqa: E402
    receive_global_model_via_ltx,
    send_local_update_via_ltx,
)
from FedLite_Project.Shared_Assets.common_utilities.runtime_paths import (  # noqa: E402
    resolve_hospital_runtime_paths,
)
from FedLite_Project.Shared_Assets.common_utilities.tkinter_hospital_app import (  # noqa: E402
    launch_hospital_gui,
)

DEFAULT_RUNTIME_PATHS = resolve_hospital_runtime_paths("Hospital_A", CURRENT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the FedLiteCare Hospital A desktop client."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUNTIME_PATHS.config_path,
        help="Optional Hospital A client config path.",
    )
    parser.add_argument(
        "--transfer-config",
        type=Path,
        default=DEFAULT_RUNTIME_PATHS.transfer_config_path,
        help="Optional Hospital A LTX transfer config path.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    transfer_config_path = args.transfer_config.resolve()

    launch_hospital_gui(
        config_path=config_path,
        receive_global_model_callable=partial(
            receive_global_model_via_ltx,
            transfer_config_path=transfer_config_path,
        ),
        send_local_update_callable=partial(
            send_local_update_via_ltx,
            transfer_config_path=transfer_config_path,
        ),
    )


if __name__ == "__main__":
    main()
