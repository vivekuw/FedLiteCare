"""Hospital-side runtime for one distributed FedLiteCare round."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import torch

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    append_log_entry,
    ensure_directory,
    resolve_path,
)
from FedLite_Project.Shared_Assets.common_utilities.hospital_quality_reports import (
    validate_training_dataset,
)
from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    get_hospital_global_model_paths,
    get_hospital_round_transfer_paths,
    load_hospital_context,
    train_local_model,
)
from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import load_checkpoint


def _emit(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _read_checkpoint_round_info(checkpoint_path: Path) -> tuple[int, str]:
    _, checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    aggregation_metadata = dict(checkpoint.get("aggregation_metadata", {}))
    round_number = int(aggregation_metadata.get("round_number", 0))
    round_name = str(aggregation_metadata.get("round_name", "")).strip() or f"round_{round_number:03d}"
    return round_number, round_name


def run_hospital_federated_round(
    config_path: Path,
    receive_global_model_callable: Callable[..., dict[str, Any]],
    send_local_update_callable: Callable[..., dict[str, Any]],
    dataset_filename: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one hospital node flow: receive, train, and send."""
    settings, paths = load_hospital_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    selected_dataset_name = dataset_filename or str(settings["dataset_filename"])
    runtime_log_path = resolve_path(
        paths["logs_dir"],
        str(settings.get("federated_log_filename", "federated_client.log")),
    )
    current_global_model_path = get_hospital_global_model_paths(config_path)[
        "current_global_model_path"
    ]
    staging_global_model_path = resolve_path(
        paths["received_global_models_dir"],
        str(settings.get("incoming_global_model_filename", "incoming_global_model.pt")),
    )

    append_log_entry(
        runtime_log_path,
        title="Hospital federated node started",
        details={
            "hospital_name": hospital_name,
            "config_path": config_path.resolve(),
            "current_global_model_path": current_global_model_path,
            "staging_global_model_path": staging_global_model_path,
            "status": "started",
        },
    )
    try:
        validation_result = validate_training_dataset(
            config_path=config_path,
            dataset_filename=selected_dataset_name,
        )
        if not validation_result["is_valid"]:
            append_log_entry(
                runtime_log_path,
                title="Federated round preflight failed",
                details={
                    "hospital_name": hospital_name,
                    "dataset_filename": selected_dataset_name,
                    "validation_status": validation_result["status"],
                    "validation_report_path": validation_result["report_path"],
                    "status": "failed_preflight",
                },
            )
            raise ValueError(
                f"{hospital_name} cannot start the federated round because the selected dataset is missing or invalid. "
                f"See validation report: {validation_result['report_path']}"
            )

        bootstrap_receive_result: dict[str, Any] | None = None
        if current_global_model_path.exists():
            base_round_number, base_round_name = _read_checkpoint_round_info(current_global_model_path)
            append_log_entry(
                runtime_log_path,
                title="Cached global model located",
                details={
                    "hospital_name": hospital_name,
                    "base_round_name": base_round_name,
                    "base_round_number": base_round_number,
                    "current_global_model_path": current_global_model_path,
                    "status": "using_cached_global_model",
                },
            )
            _emit(
                progress_callback,
                (
                    f"{hospital_name}: using cached global model {current_global_model_path.name} "
                    f"from {base_round_name}."
                ),
            )
        else:
            _emit(
                progress_callback,
                f"{hospital_name}: no cached global model found. Waiting for bootstrap model from the aggregator...",
            )
            bootstrap_receive_result = receive_global_model_callable(
                destination_path=staging_global_model_path,
                round_name=None,
            )
            shutil.copy2(staging_global_model_path, current_global_model_path)
            base_round_number, base_round_name = _read_checkpoint_round_info(current_global_model_path)
            bootstrap_archive_path = get_hospital_global_model_paths(
                config_path,
                base_round_name,
            )["round_received_global_model_path"]
            ensure_directory(bootstrap_archive_path.parent)
            shutil.copy2(current_global_model_path, bootstrap_archive_path)

            append_log_entry(
                runtime_log_path,
                title="Bootstrap global model received",
                details={
                    "hospital_name": hospital_name,
                    "base_round_name": base_round_name,
                    "base_round_number": base_round_number,
                    "received_bytes": bootstrap_receive_result["bytes_received"],
                    "staging_global_model_path": staging_global_model_path,
                    "current_global_model_path": current_global_model_path,
                    "bootstrap_archive_path": bootstrap_archive_path,
                    "status": "received_bootstrap_global_model",
                },
            )
            _emit(
                progress_callback,
                (
                    f"{hospital_name}: bootstrap global model {base_round_name} received and cached at "
                    f"{current_global_model_path}"
                ),
            )

        round_number = base_round_number + 1
        round_name = f"round_{round_number:03d}"
        transfer_paths = get_hospital_round_transfer_paths(config_path, round_name)
        refreshed_global_archive_path = get_hospital_global_model_paths(
            config_path,
            round_name,
        )["round_received_global_model_path"]

        _emit(
            progress_callback,
            f"{hospital_name}: starting local training for {round_name} from the cached global model...",
        )
        training_result = train_local_model(
            config_path=config_path,
            dataset_filename=selected_dataset_name,
            initial_model_path=current_global_model_path,
            local_update_path=transfer_paths["local_update_path"],
            round_name=round_name,
        )
        append_log_entry(
            runtime_log_path,
            title="Local training completed for federated round",
            details={
                "hospital_name": hospital_name,
                "round_name": round_name,
                "base_round_name": base_round_name,
                "dataset_path": training_result["dataset_path"],
                "validation_status": training_result["validation_result"]["status"],
                "validation_report_path": training_result["validation_report_path"],
                "local_model_path": training_result["model_path"],
                "local_update_path": training_result["local_update_path"],
                "validation_accuracy": round(training_result["validation_accuracy"], 6),
                "status": "training_completed",
            },
        )
        _emit(
            progress_callback,
            (
                f"{hospital_name}: local training completed. Update ready at "
                f"{training_result['local_update_path']}"
            ),
        )

        _emit(progress_callback, f"{hospital_name}: sending local update back to the aggregator...")
        send_result = send_local_update_callable(
            source_path=Path(training_result["local_update_path"]),
            round_name=round_name,
        )
        append_log_entry(
            runtime_log_path,
            title="Local update sent for federated round",
            details={
                "hospital_name": hospital_name,
                "round_name": round_name,
                "local_update_path": training_result["local_update_path"],
                "bytes_sent": send_result["bytes_sent"],
                "target_host": send_result["target_host"],
                "target_port": send_result["target_port"],
                "status": "completed",
            },
        )
        _emit(progress_callback, f"{hospital_name}: local update sent for {round_name}.")

        _emit(
            progress_callback,
            f"{hospital_name}: waiting for the refreshed global model for {round_name}...",
        )
        refreshed_receive_result = receive_global_model_callable(
            destination_path=staging_global_model_path,
            round_name=round_name,
        )
        shutil.copy2(staging_global_model_path, current_global_model_path)
        ensure_directory(refreshed_global_archive_path.parent)
        shutil.copy2(current_global_model_path, refreshed_global_archive_path)

        append_log_entry(
            runtime_log_path,
            title="Refreshed global model received",
            details={
                "hospital_name": hospital_name,
                "round_name": round_name,
                "received_bytes": refreshed_receive_result["bytes_received"],
                "staging_global_model_path": staging_global_model_path,
                "current_global_model_path": current_global_model_path,
                "round_global_model_path": refreshed_global_archive_path,
                "status": "received_refreshed_global_model",
            },
        )
        _emit(
            progress_callback,
            (
                f"{hospital_name}: refreshed global model for {round_name} cached at "
                f"{current_global_model_path}"
            ),
        )

        return {
            "hospital_name": hospital_name,
            "round_name": round_name,
            "base_round_name": base_round_name,
            "bootstrap_receive_result": bootstrap_receive_result,
            "current_global_model_path": current_global_model_path,
            "refreshed_global_model_path": refreshed_global_archive_path,
            "training_result": training_result,
            "send_result": send_result,
            "receive_result": refreshed_receive_result,
            "runtime_log_path": runtime_log_path,
            "transfer_log_path": send_result["transfer_log_path"],
        }
    except Exception as error:
        append_log_entry(
            runtime_log_path,
            title="Federated sync failed",
            details={
                "hospital_name": hospital_name,
                "status": "failed",
                "error": str(error),
            },
        )
        raise
