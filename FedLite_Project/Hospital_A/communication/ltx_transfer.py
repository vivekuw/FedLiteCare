"""Hospital A LTX communication wrappers."""

from __future__ import annotations

import threading
from pathlib import Path

from FedLite_Project.Shared_Assets.common_utilities.ltx_hospital_client import (
    receive_global_model_via_ltx as _receive_global_model_via_ltx,
    send_local_update_via_ltx as _send_local_update_via_ltx,
)

DEFAULT_TRANSFER_CONFIG_PATH = Path(__file__).resolve().parent / "transfer_config.yaml"


def receive_global_model_via_ltx(
    destination_path: Path,
    round_name: str | None = None,
    transfer_config_path: Path = DEFAULT_TRANSFER_CONFIG_PATH,
    ready_event: threading.Event | None = None,
) -> dict[str, object]:
    return _receive_global_model_via_ltx(
        destination_path=destination_path,
        round_name=round_name,
        transfer_config_path=transfer_config_path,
        ready_event=ready_event,
    )


def send_local_update_via_ltx(
    source_path: Path,
    round_name: str,
    transfer_config_path: Path = DEFAULT_TRANSFER_CONFIG_PATH,
) -> dict[str, object]:
    return _send_local_update_via_ltx(
        source_path=source_path,
        round_name=round_name,
        transfer_config_path=transfer_config_path,
    )
