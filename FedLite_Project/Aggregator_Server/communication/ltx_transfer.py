"""Aggregator-side LTX communication helpers for localhost model transfer."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    ensure_directory,
    load_simple_yaml_config,
    resolve_path,
)
from FedLite_Project.Shared_Assets.common_utilities.ltx_core import (
    receive_file_chunked,
    send_file_chunked,
)
from FedLite_Project.Shared_Assets.common_utilities.transfer_logging import (
    log_transfer_event,
)

DEFAULT_TRANSFER_CONFIG_PATH = Path(__file__).resolve().parent / "transfer_config.yaml"

MODEL_TRANSFER_SUFFIXES = {".pt", ".pth", ".json", ".yaml", ".yml"}
HOSPITAL_PORT_KEYS = {
    "Hospital_A": {
        "global_model_port": "hospital_a_receive_port",
        "local_update_port": "hospital_a_update_receive_port",
    },
    "Hospital_B": {
        "global_model_port": "hospital_b_receive_port",
        "local_update_port": "hospital_b_update_receive_port",
    },
    "Hospital_C": {
        "global_model_port": "hospital_c_receive_port",
        "local_update_port": "hospital_c_update_receive_port",
    },
}


def _validate_model_transfer_path(path: Path) -> Path:
    resolved_path = path.resolve()
    if resolved_path.suffix.lower() not in MODEL_TRANSFER_SUFFIXES:
        raise ValueError(
            f"LTX transfers are limited to model-related files, got '{resolved_path.name}'."
        )
    return resolved_path


def _load_transfer_context(
    transfer_config_path: Path = DEFAULT_TRANSFER_CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Path]]:
    resolved_config_path = transfer_config_path.resolve()
    settings = load_simple_yaml_config(resolved_config_path)
    aggregator_root = resolved_config_path.parent.parent
    logs_dir = ensure_directory(aggregator_root / "logs")
    transfer_log_path = resolve_path(
        logs_dir,
        str(settings.get("transfer_log_filename", "transfer.log")),
    )

    return settings, {
        "config_path": resolved_config_path,
        "aggregator_root": aggregator_root,
        "logs_dir": logs_dir,
        "transfer_log_path": transfer_log_path,
    }


def _resolve_hospital_port(settings: dict[str, Any], hospital_name: str, port_type: str) -> int:
    hospital_ports = HOSPITAL_PORT_KEYS.get(hospital_name)
    if hospital_ports is None:
        raise ValueError(f"Unsupported hospital name '{hospital_name}'.")
    return int(settings[hospital_ports[port_type]])


def send_global_model_to_hospital(
    hospital_name: str,
    source_path: Path,
    round_name: str,
    transfer_config_path: Path = DEFAULT_TRANSFER_CONFIG_PATH,
) -> dict[str, Any]:
    """Send the current global model to one hospital over localhost LTX."""
    settings, paths = _load_transfer_context(transfer_config_path)
    resolved_source_path = _validate_model_transfer_path(source_path)
    target_host = str(settings.get("host", "127.0.0.1"))
    target_port = _resolve_hospital_port(settings, hospital_name, "global_model_port")

    log_transfer_event(
        paths["transfer_log_path"],
        title="Sending global model via LTX",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "source_path": resolved_source_path,
            "target_host": target_host,
            "target_port": target_port,
        },
    )

    result = send_file_chunked(
        source_path=resolved_source_path,
        target_host=target_host,
        target_port=target_port,
        chunk_size=int(settings.get("chunk_size_bytes", 65536)),
        socket_timeout_seconds=float(settings.get("socket_timeout_seconds", 30)),
        metadata={
            "transfer_type": "global_model",
            "hospital_name": hospital_name,
            "round_name": round_name,
            "sender": "Aggregator_Server",
            "receiver": hospital_name,
        },
    )

    log_transfer_event(
        paths["transfer_log_path"],
        title="Global model sent via LTX",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "source_path": result["source_path"],
            "bytes_sent": result["bytes_sent"],
            "file_size": result["file_size"],
            "target_host": result["target_host"],
            "target_port": result["target_port"],
        },
    )
    return {
        **result,
        "transfer_log_path": paths["transfer_log_path"],
    }


def receive_local_update_from_hospital(
    hospital_name: str,
    destination_path: Path,
    round_name: str,
    transfer_config_path: Path = DEFAULT_TRANSFER_CONFIG_PATH,
    ready_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Receive one hospital model update over localhost LTX."""
    settings, paths = _load_transfer_context(transfer_config_path)
    resolved_destination_path = _validate_model_transfer_path(destination_path)
    bind_host = str(settings.get("host", "127.0.0.1"))
    bind_port = _resolve_hospital_port(settings, hospital_name, "local_update_port")

    log_transfer_event(
        paths["transfer_log_path"],
        title="Waiting for local update via LTX",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "bind_host": bind_host,
            "bind_port": bind_port,
            "destination_path": resolved_destination_path,
        },
    )

    result = receive_file_chunked(
        bind_host=bind_host,
        bind_port=bind_port,
        destination_path=resolved_destination_path,
        chunk_size=int(settings.get("chunk_size_bytes", 65536)),
        socket_timeout_seconds=float(settings.get("socket_timeout_seconds", 30)),
        ready_event=ready_event,
    )

    log_transfer_event(
        paths["transfer_log_path"],
        title="Local update received via LTX",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "sender_address": result["sender_address"],
            "sender_port": result["sender_port"],
            "received_bytes": result["bytes_received"],
            "destination_path": result["destination_path"],
            "file_name": result["header"]["file_name"],
        },
    )
    return {
        **result,
        "transfer_log_path": paths["transfer_log_path"],
    }
