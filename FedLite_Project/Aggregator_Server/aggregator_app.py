"""GUI entrypoint for the FedLiteCare aggregator dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
for candidate in (CURRENT_FILE.parent, *CURRENT_FILE.parents):
    package_parent = candidate / "FedLite_Project"
    if package_parent.is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from FedLite_Project.Shared_Assets.common_utilities.runtime_paths import (  # noqa: E402
    bootstrap_project_imports,
)

bootstrap_project_imports(CURRENT_FILE)

from FedLite_Project.Aggregator_Server.gui.tkinter_aggregator_dashboard import (  # noqa: E402
    launch_aggregator_dashboard,
)
from FedLite_Project.Aggregator_Server.server.global_model_manager import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the FedLiteCare aggregator dashboard."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Optional aggregator server config path.",
    )
    args = parser.parse_args()
    launch_aggregator_dashboard(config_path=args.config.resolve())


if __name__ == "__main__":
    main()
