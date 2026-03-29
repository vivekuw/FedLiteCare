"""Core aggregator orchestration for local federated rounds."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Aggregator_Server.aggregation.federated_averaging import (
    aggregate_hospital_checkpoints,
)
from FedLite_Project.Aggregator_Server.server.model_version_saver import (
    append_version_history,
    save_global_model_versions,
)
from FedLite_Project.Aggregator_Server.server.round_manager import RoundManager
from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    append_log_entry,
    ensure_directory,
    load_simple_yaml_config,
    resolve_path,
)
from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import load_checkpoint

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "server_config.yaml"

HOSPITAL_SOURCE_KEYS = {
    "Hospital_A": "hospital_a_model_source",
    "Hospital_B": "hospital_b_model_source",
    "Hospital_C": "hospital_c_model_source",
}


def load_server_context(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load aggregator config and resolve the key directories."""
    resolved_config_path = config_path.resolve()
    settings = load_simple_yaml_config(resolved_config_path)
    aggregator_root = resolved_config_path.parent.parent

    received_models_dir = ensure_directory(
        resolve_path(aggregator_root, str(settings["received_models_dir"]))
    )
    global_models_dir = ensure_directory(
        resolve_path(aggregator_root, str(settings["global_models_dir"]))
    )
    logs_dir = ensure_directory(resolve_path(aggregator_root, str(settings.get("logs_dir", "logs"))))

    return settings, {
        "config_path": resolved_config_path,
        "aggregator_root": aggregator_root,
        "received_models_dir": received_models_dir,
        "global_models_dir": global_models_dir,
        "logs_dir": logs_dir,
    }


def _resolve_hospital_sources(
    settings: dict[str, Any],
    aggregator_root: Path,
    overrides: dict[str, Path | None] | None = None,
) -> dict[str, Path]:
    resolved_sources: dict[str, Path] = {}
    overrides = overrides or {}

    for hospital_name, config_key in HOSPITAL_SOURCE_KEYS.items():
        raw_override = overrides.get(hospital_name)
        if raw_override is not None:
            resolved_sources[hospital_name] = raw_override.resolve()
            continue

        resolved_sources[hospital_name] = resolve_path(aggregator_root, str(settings[config_key]))

    return resolved_sources


def receive_hospital_updates(
    settings: dict[str, Any],
    paths: dict[str, Path],
    round_name: str,
    overrides: dict[str, Path | None] | None = None,
) -> dict[str, dict[str, Path]]:
    """Copy local hospital checkpoints into the aggregator's received-model store."""
    source_paths = _resolve_hospital_sources(settings, paths["aggregator_root"], overrides)
    round_received_dir = ensure_directory(paths["received_models_dir"] / round_name)

    received_updates: dict[str, dict[str, Path]] = {}
    for hospital_name, source_path in source_paths.items():
        if not source_path.exists():
            raise FileNotFoundError(
                f"Expected model update for {hospital_name} at '{source_path}', but the file does not exist."
            )

        destination_path = round_received_dir / f"{hospital_name.lower()}_update.pt"
        shutil.copy2(source_path, destination_path)
        received_updates[hospital_name] = {
            "source_path": source_path,
            "received_path": destination_path,
        }

    manifest_path = round_received_dir / "received_manifest.json"
    manifest_payload = {
        "round_name": round_name,
        "received_at": datetime.now().isoformat(timespec="seconds"),
        "hospitals": {
            hospital_name: {
                "source_path": str(update_paths["source_path"]),
                "received_path": str(update_paths["received_path"]),
            }
            for hospital_name, update_paths in received_updates.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return received_updates


def _load_received_checkpoint_payloads(
    received_updates: dict[str, dict[str, Path]],
) -> dict[str, dict[str, Any]]:
    device = torch.device("cpu")
    hospital_checkpoints: dict[str, dict[str, Any]] = {}

    for hospital_name, update_paths in received_updates.items():
        _, checkpoint = load_checkpoint(update_paths["received_path"], device)
        hospital_checkpoints[hospital_name] = checkpoint

    return hospital_checkpoints


def run_aggregation_round(
    config_path: Path = DEFAULT_CONFIG_PATH,
    hospital_model_overrides: dict[str, Path | None] | None = None,
) -> dict[str, Any]:
    """Run one complete local federated aggregation round."""
    settings, paths = load_server_context(config_path)
    round_manager = RoundManager(
        resolve_path(paths["aggregator_root"], str(settings["round_state_filename"]))
    )
    round_number = round_manager.next_round_number()
    round_name = round_manager.format_round_name(round_number)

    received_updates = receive_hospital_updates(
        settings=settings,
        paths=paths,
        round_name=round_name,
        overrides=hospital_model_overrides,
    )
    aggregated_checkpoint = aggregate_hospital_checkpoints(
        _load_received_checkpoint_payloads(received_updates)
    )
    aggregated_checkpoint["aggregation_metadata"].update(
        {
            "round_number": round_number,
            "round_name": round_name,
            "aggregated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    global_model_path, latest_model_path = save_global_model_versions(
        global_models_dir=paths["global_models_dir"],
        latest_filename=str(settings["latest_global_model_filename"]),
        round_name=round_name,
        checkpoint_payload=aggregated_checkpoint,
    )

    version_entry = {
        "round_number": round_number,
        "round_name": round_name,
        "global_model_path": str(global_model_path),
        "latest_model_path": str(latest_model_path),
        "hospital_names": list(received_updates.keys()),
        "aggregated_at": aggregated_checkpoint["aggregation_metadata"]["aggregated_at"],
    }
    version_history_path = resolve_path(
        paths["global_models_dir"],
        str(settings["version_history_filename"]),
    )
    append_version_history(version_history_path, version_entry)

    round_summary = {
        "round_number": round_number,
        "round_name": round_name,
        "global_model_path": str(global_model_path),
        "received_round_dir": str(paths["received_models_dir"] / round_name),
        "hospital_names": list(received_updates.keys()),
        "completed_at": aggregated_checkpoint["aggregation_metadata"]["aggregated_at"],
    }
    round_manager.record_completed_round(round_summary)

    aggregation_log_path = resolve_path(
        paths["logs_dir"],
        str(settings.get("aggregation_log_filename", "aggregator.log")),
    )
    append_log_entry(
        aggregation_log_path,
        title="Aggregation round completed",
        details={
            "round_number": round_number,
            "round_name": round_name,
            "global_model_path": global_model_path,
            "latest_model_path": latest_model_path,
            "hospitals": ", ".join(received_updates.keys()),
        },
    )

    return {
        "round_number": round_number,
        "round_name": round_name,
        "received_updates": received_updates,
        "global_model_path": global_model_path,
        "latest_model_path": latest_model_path,
        "version_history_path": version_history_path,
        "round_state_path": round_manager.state_path,
        "log_path": aggregation_log_path,
    }
