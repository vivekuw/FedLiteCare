"""Reusable LTX helpers for hospital-side localhost model transfers."""

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

MODEL_TRANSFER_SUFFIXES = {".pt", ".pth", ".json", ".yaml", ".yml"}


def _validate_model_transfer_path(path: Path) -> Path:
    resolved_path = path.resolve()
    if resolved_path.suffix.lower() not in MODEL_TRANSFER_SUFFIXES:
        raise ValueError(
            f"LTX transfers are limited to model-related files, got '{resolved_path.name}'."
        )
    return resolved_path


def load_hospital_transfer_context(
    transfer_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load a hospital LTX config and resolve its log location."""
    resolved_config_path = transfer_config_path.resolve()
    settings = load_simple_yaml_config(resolved_config_path)
    hospital_root = resolved_config_path.parent.parent
    logs_dir = ensure_directory(hospital_root / "logs")
    transfer_log_path = resolve_path(
        logs_dir,
        str(settings.get("transfer_log_filename", "transfer.log")),
    )

    return settings, {
        "config_path": resolved_config_path,
        "hospital_root": hospital_root,
        "logs_dir": logs_dir,
        "transfer_log_path": transfer_log_path,
    }


def receive_global_model_via_ltx(
    destination_path: Path,
    round_name: str,
    transfer_config_path: Path,
    ready_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Receive the current global model over localhost LTX."""
    settings, paths = load_hospital_transfer_context(transfer_config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    resolved_destination_path = _validate_model_transfer_path(destination_path)

    log_transfer_event(
        paths["transfer_log_path"],
        title="Receiving global model via LTX",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "bind_host": str(settings.get("host", "127.0.0.1")),
            "bind_port": int(settings["receive_port"]),
            "destination_path": resolved_destination_path,
        },
    )

    result = receive_file_chunked(
        bind_host=str(settings.get("host", "127.0.0.1")),
        bind_port=int(settings["receive_port"]),
        destination_path=resolved_destination_path,
        chunk_size=int(settings.get("chunk_size_bytes", 65536)),
        socket_timeout_seconds=float(settings.get("socket_timeout_seconds", 30)),
        ready_event=ready_event,
    )

    log_transfer_event(
        paths["transfer_log_path"],
        title="Global model received via LTX",
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


def send_local_update_via_ltx(
    source_path: Path,
    round_name: str,
    transfer_config_path: Path,
) -> dict[str, Any]:
    """Send a locally trained hospital update to the aggregator over localhost LTX."""
    settings, paths = load_hospital_transfer_context(transfer_config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    resolved_source_path = _validate_model_transfer_path(source_path)

    log_transfer_event(
        paths["transfer_log_path"],
        title="Sending local update via LTX",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "target_host": str(settings.get("aggregator_host", "127.0.0.1")),
            "target_port": int(settings["aggregator_receive_port"]),
            "source_path": resolved_source_path,
        },
    )

    result = send_file_chunked(
        source_path=resolved_source_path,
        target_host=str(settings.get("aggregator_host", "127.0.0.1")),
        target_port=int(settings["aggregator_receive_port"]),
        chunk_size=int(settings.get("chunk_size_bytes", 65536)),
        socket_timeout_seconds=float(settings.get("socket_timeout_seconds", 30)),
        metadata={
            "transfer_type": "local_update",
            "hospital_name": hospital_name,
            "round_name": round_name,
            "sender": hospital_name,
            "receiver": "Aggregator_Server",
        },
    )

    log_transfer_event(
        paths["transfer_log_path"],
        title="Local update sent via LTX",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "bytes_sent": result["bytes_sent"],
            "file_size": result["file_size"],
            "source_path": result["source_path"],
            "target_host": result["target_host"],
            "target_port": result["target_port"],
        },
    )
    return {
        **result,
        "transfer_log_path": paths["transfer_log_path"],
    }
