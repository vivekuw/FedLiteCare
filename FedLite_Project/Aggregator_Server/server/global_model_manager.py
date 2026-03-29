"""Core aggregator orchestration for local federated rounds."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from FedLite_Project.Aggregator_Server.aggregation.federated_averaging import (
    aggregate_hospital_checkpoints,
)
from FedLite_Project.Aggregator_Server.model_definition.diabetes_model import (
    build_global_diabetes_model,
)
from FedLite_Project.Aggregator_Server.server.model_version_saver import (
    append_version_history,
    load_version_history,
    save_global_model_versions,
)
from FedLite_Project.Aggregator_Server.server.round_manager import RoundManager
from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    append_log_entry,
    ensure_directory,
    load_simple_yaml_config,
    resolve_path,
)
from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    get_hospital_round_transfer_paths,
    load_hospital_context,
    train_local_model,
)
from FedLite_Project.Shared_Assets.data_preprocessing_helpers.preprocessing_utils import (
    load_csv_records,
)
from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import load_checkpoint

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "server_config.yaml"

HOSPITAL_SOURCE_KEYS = {
    "Hospital_A": "hospital_a_model_source",
    "Hospital_B": "hospital_b_model_source",
    "Hospital_C": "hospital_c_model_source",
}

HOSPITAL_CONFIG_KEYS = {
    "Hospital_A": "hospital_a_config_path",
    "Hospital_B": "hospital_b_config_path",
    "Hospital_C": "hospital_c_config_path",
}


def _emit(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


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


def load_hospital_config_paths(settings: dict[str, Any], aggregator_root: Path) -> dict[str, Path]:
    """Resolve the three hospital config files from server settings."""
    return {
        hospital_name: resolve_path(aggregator_root, str(settings[config_key]))
        for hospital_name, config_key in HOSPITAL_CONFIG_KEYS.items()
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


def _load_hospital_layout(config_path: Path) -> dict[str, Any]:
    settings, paths = load_hospital_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    target_column = str(settings.get("target_column", "Outcome"))
    dataset_path = resolve_path(paths["uploads_dir"], str(settings["dataset_filename"]))
    records = load_csv_records(dataset_path)
    feature_columns = [column for column in records[0].keys() if column != target_column]

    return {
        "hospital_name": hospital_name,
        "dataset_path": dataset_path,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "model_config": {
            "input_dim": len(feature_columns),
            "hidden_dim": int(settings.get("hidden_dim", 16)),
        },
    }


def validate_hospital_layouts(hospital_config_paths: dict[str, Path]) -> dict[str, Any]:
    """Ensure all hospitals use the same feature layout and model dimensions."""
    layouts = [_load_hospital_layout(config_path) for config_path in hospital_config_paths.values()]
    reference = layouts[0]

    for candidate in layouts[1:]:
        if candidate["feature_columns"] != reference["feature_columns"]:
            raise ValueError("Hospital datasets do not share the same feature columns.")
        if candidate["target_column"] != reference["target_column"]:
            raise ValueError("Hospital datasets do not share the same target column.")
        if candidate["model_config"] != reference["model_config"]:
            raise ValueError("Hospital configs do not share the same model settings.")

    return reference


def _build_initial_global_checkpoint(layout: dict[str, Any]) -> dict[str, Any]:
    model = build_global_diabetes_model(**layout["model_config"])
    input_dim = int(layout["model_config"]["input_dim"])

    return {
        "model_state_dict": model.state_dict(),
        "model_config": layout["model_config"],
        "preprocessing": {
            "feature_columns": layout["feature_columns"],
            "target_column": layout["target_column"],
            "impute_values": [0.0] * input_dim,
            "feature_means": [0.0] * input_dim,
            "feature_stds": [1.0] * input_dim,
        },
        "training_metrics": {},
        "history": [],
        "aggregation_metadata": {
            "algorithm": "InitialSeed",
            "hospital_names": [],
            "num_hospitals": 0,
            "round_number": 0,
            "round_name": "round_000",
            "aggregated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def ensure_current_global_model(
    settings: dict[str, Any],
    paths: dict[str, Path],
    hospital_config_paths: dict[str, Path],
) -> dict[str, Any]:
    """Load the latest global model or create the initial seed model."""
    latest_model_path = resolve_path(
        paths["global_models_dir"],
        str(settings["latest_global_model_filename"]),
    )
    layout = validate_hospital_layouts(hospital_config_paths)

    if latest_model_path.exists():
        _, checkpoint = load_checkpoint(latest_model_path, torch.device("cpu"))
        if checkpoint["model_config"] != layout["model_config"]:
            raise ValueError("The saved global model does not match the current hospital model configuration.")
        return {
            "global_model_path": latest_model_path,
            "created_new_model": False,
            "layout": layout,
        }

    checkpoint_payload = _build_initial_global_checkpoint(layout)
    round_zero_path, refreshed_latest_path = save_global_model_versions(
        global_models_dir=paths["global_models_dir"],
        latest_filename=str(settings["latest_global_model_filename"]),
        round_name="round_000",
        checkpoint_payload=checkpoint_payload,
    )
    version_history_path = resolve_path(
        paths["global_models_dir"],
        str(settings["version_history_filename"]),
    )
    version_history = load_version_history(version_history_path)
    if not any(int(entry.get("round_number", -1)) == 0 for entry in version_history.get("versions", [])):
        append_version_history(
            version_history_path,
            {
                "round_number": 0,
                "round_name": "round_000",
                "global_model_path": str(round_zero_path),
                "latest_model_path": str(refreshed_latest_path),
                "hospital_names": [],
                "aggregated_at": checkpoint_payload["aggregation_metadata"]["aggregated_at"],
                "source": "initializer",
            },
        )

    return {
        "global_model_path": refreshed_latest_path,
        "created_new_model": True,
        "layout": layout,
        "initial_version_path": round_zero_path,
    }


def distribute_global_model_to_hospitals(
    hospital_config_paths: dict[str, Path],
    global_model_path: Path,
    round_name: str,
) -> dict[str, Path]:
    """Copy the current global model into each hospital's local communication folder."""
    distributed_paths: dict[str, Path] = {}
    for hospital_name, config_path in hospital_config_paths.items():
        transfer_paths = get_hospital_round_transfer_paths(config_path, round_name)
        ensure_directory(transfer_paths["received_global_model_path"].parent)
        shutil.copy2(global_model_path, transfer_paths["received_global_model_path"])
        distributed_paths[hospital_name] = transfer_paths["received_global_model_path"]
    return distributed_paths


def run_hospital_training_rounds(
    hospital_config_paths: dict[str, Path],
    distributed_models: dict[str, Path],
    round_name: str,
) -> dict[str, dict[str, Any]]:
    """Train all hospitals locally from the distributed global model."""
    training_results: dict[str, dict[str, Any]] = {}
    for hospital_name, config_path in hospital_config_paths.items():
        transfer_paths = get_hospital_round_transfer_paths(config_path, round_name)
        training_results[hospital_name] = train_local_model(
            config_path=config_path,
            initial_model_path=distributed_models[hospital_name],
            local_update_path=transfer_paths["local_update_path"],
            round_name=round_name,
        )
    return training_results


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
    round_number: int | None = None,
    round_name: str | None = None,
) -> dict[str, Any]:
    """Run one local federated aggregation pass from available hospital updates."""
    settings, paths = load_server_context(config_path)
    round_manager = RoundManager(
        resolve_path(paths["aggregator_root"], str(settings["round_state_filename"]))
    )
    resolved_round_number = round_number or round_manager.next_round_number()
    resolved_round_name = round_name or round_manager.format_round_name(resolved_round_number)

    received_updates = receive_hospital_updates(
        settings=settings,
        paths=paths,
        round_name=resolved_round_name,
        overrides=hospital_model_overrides,
    )
    aggregated_checkpoint = aggregate_hospital_checkpoints(
        _load_received_checkpoint_payloads(received_updates)
    )
    aggregated_checkpoint["aggregation_metadata"].update(
        {
            "round_number": resolved_round_number,
            "round_name": resolved_round_name,
            "aggregated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    global_model_path, latest_model_path = save_global_model_versions(
        global_models_dir=paths["global_models_dir"],
        latest_filename=str(settings["latest_global_model_filename"]),
        round_name=resolved_round_name,
        checkpoint_payload=aggregated_checkpoint,
    )

    version_entry = {
        "round_number": resolved_round_number,
        "round_name": resolved_round_name,
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
        "round_number": resolved_round_number,
        "round_name": resolved_round_name,
        "global_model_path": str(global_model_path),
        "received_round_dir": str(paths["received_models_dir"] / resolved_round_name),
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
            "round_number": resolved_round_number,
            "round_name": resolved_round_name,
            "global_model_path": global_model_path,
            "latest_model_path": latest_model_path,
            "hospitals": ", ".join(received_updates.keys()),
        },
    )

    return {
        "round_number": resolved_round_number,
        "round_name": resolved_round_name,
        "received_updates": received_updates,
        "global_model_path": global_model_path,
        "latest_model_path": latest_model_path,
        "version_history_path": version_history_path,
        "round_state_path": round_manager.state_path,
        "aggregation_log_path": aggregation_log_path,
    }


def run_full_federated_round(
    config_path: Path = DEFAULT_CONFIG_PATH,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the full one-laptop federated learning simulation round."""
    settings, paths = load_server_context(config_path)
    hospital_config_paths = load_hospital_config_paths(settings, paths["aggregator_root"])
    round_manager = RoundManager(
        resolve_path(paths["aggregator_root"], str(settings["round_state_filename"]))
    )
    round_number = round_manager.next_round_number()
    round_name = round_manager.format_round_name(round_number)

    _emit(progress_callback, f"[1/8] Loading or creating the current global model for {round_name}...")
    global_model_info = ensure_current_global_model(settings, paths, hospital_config_paths)
    current_global_model_path = Path(global_model_info["global_model_path"])
    if global_model_info["created_new_model"]:
        _emit(progress_callback, f"Created initial global model: {current_global_model_path}")
    else:
        _emit(progress_callback, f"Loaded existing global model: {current_global_model_path}")

    _emit(progress_callback, f"[2/8] Distributing the global model to all hospitals for {round_name}...")
    distributed_models = distribute_global_model_to_hospitals(
        hospital_config_paths=hospital_config_paths,
        global_model_path=current_global_model_path,
        round_name=round_name,
    )
    for hospital_name, distributed_path in distributed_models.items():
        _emit(progress_callback, f"{hospital_name} received global model copy: {distributed_path}")

    hospital_order = ["Hospital_A", "Hospital_B", "Hospital_C"]
    step_labels = {
        "Hospital_A": "[3/8]",
        "Hospital_B": "[4/8]",
        "Hospital_C": "[5/8]",
    }
    training_results: dict[str, dict[str, Any]] = {}
    for hospital_name in hospital_order:
        _emit(
            progress_callback,
            f"{step_labels[hospital_name]} {hospital_name} is training locally from the distributed global model...",
        )
        training_result = train_local_model(
            config_path=hospital_config_paths[hospital_name],
            initial_model_path=distributed_models[hospital_name],
            local_update_path=get_hospital_round_transfer_paths(
                hospital_config_paths[hospital_name], round_name
            )["local_update_path"],
            round_name=round_name,
        )
        training_results[hospital_name] = training_result
        _emit(
            progress_callback,
            (
                f"{hospital_name} finished training. Local model: {training_result['model_path']} | "
                f"Local update: {training_result['local_update_path']} | "
                f"Validation accuracy: {training_result['validation_accuracy']:.4f}"
            ),
        )

    _emit(progress_callback, f"[6/8] Aggregator collecting hospital model updates for {round_name}...")
    update_overrides = {
        hospital_name: Path(training_results[hospital_name]["local_update_path"])
        for hospital_name in hospital_order
    }

    _emit(progress_callback, f"[7/8] Performing federated averaging for {round_name}...")
    aggregation_result = run_aggregation_round(
        config_path=config_path,
        hospital_model_overrides=update_overrides,
        round_number=round_number,
        round_name=round_name,
    )

    _emit(progress_callback, f"[8/8] Saving the new global model version for {round_name}...")
    _emit(progress_callback, f"New versioned global model: {aggregation_result['global_model_path']}")
    _emit(progress_callback, f"Latest global model refreshed: {aggregation_result['latest_model_path']}")

    round_log_path = resolve_path(
        paths["logs_dir"],
        str(settings.get("round_log_filename", "round_log.log")),
    )
    round_log_details: dict[str, Any] = {
        "round_name": round_name,
        "round_number": round_number,
        "starting_global_model": current_global_model_path,
        "new_global_model": aggregation_result["global_model_path"],
        "latest_global_model": aggregation_result["latest_model_path"],
    }
    for hospital_name in hospital_order:
        round_log_details[f"{hospital_name.lower()}_distributed_model"] = distributed_models[hospital_name]
        round_log_details[f"{hospital_name.lower()}_local_model"] = training_results[hospital_name]["model_path"]
        round_log_details[f"{hospital_name.lower()}_local_update"] = training_results[hospital_name]["local_update_path"]

    append_log_entry(
        round_log_path,
        title="Full federated round completed",
        details=round_log_details,
    )

    return {
        "round_number": round_number,
        "round_name": round_name,
        "current_global_model_path": current_global_model_path,
        "distributed_models": distributed_models,
        "hospital_training_results": training_results,
        "received_updates": aggregation_result["received_updates"],
        "global_model_path": aggregation_result["global_model_path"],
        "latest_model_path": aggregation_result["latest_model_path"],
        "version_history_path": aggregation_result["version_history_path"],
        "round_state_path": aggregation_result["round_state_path"],
        "aggregation_log_path": aggregation_result["aggregation_log_path"],
        "round_log_path": round_log_path,
    }
