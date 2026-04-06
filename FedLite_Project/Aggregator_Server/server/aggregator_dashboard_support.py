"""Status helpers for the FedLiteCare aggregator dashboard GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from FedLite_Project.Aggregator_Server.server.global_model_manager import (
    DEFAULT_CONFIG_PATH,
    HOSPITAL_ORDER,
    load_hospital_config_paths,
    load_server_context,
)
from FedLite_Project.Aggregator_Server.server.round_manager import RoundManager
from FedLite_Project.Shared_Assets.common_utilities.hospital_gui_support import (
    get_research_node_status,
    read_recent_log_lines,
)
from FedLite_Project.Shared_Assets.common_utilities.common_utils import resolve_path


def _format_bytes(num_bytes: int | None) -> str:
    if not num_bytes:
        return "N/A"

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _load_json_if_exists(path: Path, default_value: Any) -> Any:
    if not path.exists():
        return default_value
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None

    candidates = [
        file_path
        for file_path in directory.glob(pattern)
        if file_path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda file_path: file_path.stat().st_mtime)


def _resolve_aggregator_log_paths(settings: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "runtime": resolve_path(paths["logs_dir"], str(settings.get("node_log_filename", "aggregator_runtime.log"))),
        "aggregation": resolve_path(paths["logs_dir"], str(settings.get("aggregation_log_filename", "aggregator.log"))),
        "round": resolve_path(paths["logs_dir"], str(settings.get("round_log_filename", "round_log.log"))),
    }


def _resolve_latest_received_update(received_models_dir: Path, hospital_name: str) -> Path | None:
    update_name = f"{hospital_name.lower()}_update.pt"
    latest_match: Path | None = None

    for round_dir in received_models_dir.glob("round_*"):
        candidate = round_dir / update_name
        if not candidate.exists():
            continue
        if latest_match is None or candidate.stat().st_mtime > latest_match.stat().st_mtime:
            latest_match = candidate

    return latest_match


def _build_node_summaries(hospital_config_paths: dict[str, Path], received_models_dir: Path) -> dict[str, dict[str, Any]]:
    node_summaries: dict[str, dict[str, Any]] = {}

    for hospital_name in HOSPITAL_ORDER:
        node_status = get_research_node_status(hospital_config_paths[hospital_name])
        latest_received_update = _resolve_latest_received_update(received_models_dir, hospital_name)
        readiness = "Ready"
        if node_status["dataset_row_count"] in {None, 0}:
            readiness = "Missing dataset"
        elif node_status["local_model_version"] == "Not trained":
            readiness = "Needs local model"

        node_summaries[hospital_name] = {
            **node_status,
            "readiness": readiness,
            "latest_received_update_path": latest_received_update,
            "latest_received_update_size_text": (
                _format_bytes(latest_received_update.stat().st_size)
                if latest_received_update is not None
                else "N/A"
            ),
        }

    return node_summaries


def get_aggregator_dashboard_status(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Return a compact aggregator + node summary for the Tkinter dashboard."""
    settings, paths = load_server_context(config_path)
    round_state_path = resolve_path(paths["aggregator_root"], str(settings["round_state_filename"]))
    version_history_path = resolve_path(paths["global_models_dir"], str(settings["version_history_filename"]))
    latest_global_model_path = resolve_path(paths["global_models_dir"], str(settings["latest_global_model_filename"]))
    hospital_config_paths = load_hospital_config_paths(settings, paths["aggregator_root"])
    round_manager = RoundManager(round_state_path)
    round_state = round_manager.load_state()
    version_history = _load_json_if_exists(version_history_path, {"versions": []})
    version_entries = list(version_history.get("versions", []))
    latest_version_entry = version_entries[-1] if version_entries else None
    log_paths = _resolve_aggregator_log_paths(settings, paths)

    return {
        "server_name": str(settings.get("server_name", "FedLiteCare_Aggregator")),
        "config_path": config_path.resolve(),
        "latest_completed_round": int(round_state.get("latest_completed_round", 0)),
        "next_round_number": round_manager.next_round_number(),
        "latest_round_name": None if latest_version_entry is None else latest_version_entry.get("round_name"),
        "latest_round_completed_at": None if latest_version_entry is None else latest_version_entry.get("aggregated_at"),
        "total_saved_versions": len(version_entries),
        "latest_global_model_path": latest_global_model_path,
        "latest_global_model_size_text": (
            _format_bytes(latest_global_model_path.stat().st_size)
            if latest_global_model_path.exists()
            else "N/A"
        ),
        "round_state_path": round_state_path,
        "version_history_path": version_history_path,
        "logs": log_paths,
        "nodes": _build_node_summaries(hospital_config_paths, paths["received_models_dir"]),
    }


def get_aggregator_log_text(
    config_path: Path = DEFAULT_CONFIG_PATH,
    log_name: str = "runtime",
    max_lines: int = 60,
) -> str:
    """Read one of the aggregator logs for GUI display."""
    status = get_aggregator_dashboard_status(config_path)
    log_path = status["logs"].get(log_name)
    if log_path is None:
        return f"Unknown log: {log_name}"
    return read_recent_log_lines(log_path, max_lines=max_lines)
