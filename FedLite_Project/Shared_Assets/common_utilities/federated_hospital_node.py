"""Hospital-side runtime for one distributed FedLiteCare round."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    append_log_entry,
    ensure_directory,
    resolve_path,
)
from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import (
    get_hospital_round_transfer_paths,
    load_hospital_context,
    train_local_model,
)


def _emit(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


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
    runtime_log_path = resolve_path(
        paths["logs_dir"],
        str(settings.get("federated_log_filename", "federated_client.log")),
    )
    staging_global_model_path = resolve_path(
        paths["received_global_models_dir"],
        str(settings.get("current_global_model_filename", "current_global_model.pt")),
    )

    append_log_entry(
        runtime_log_path,
        title="Hospital federated node started",
        details={
            "hospital_name": hospital_name,
            "config_path": config_path.resolve(),
            "staging_global_model_path": staging_global_model_path,
        },
    )

    _emit(
        progress_callback,
        f"{hospital_name}: waiting for the current global model from the aggregator...",
    )
    receive_result = receive_global_model_callable(
        destination_path=staging_global_model_path,
        round_name=None,
    )
    round_name = str(receive_result["header"].get("round_name", "round_unknown"))
    transfer_paths = get_hospital_round_transfer_paths(config_path, round_name)
    ensure_directory(transfer_paths["received_global_model_path"].parent)
    if staging_global_model_path != transfer_paths["received_global_model_path"]:
        shutil.copy2(staging_global_model_path, transfer_paths["received_global_model_path"])

    append_log_entry(
        runtime_log_path,
        title="Global model received for federated round",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "received_bytes": receive_result["bytes_received"],
            "staging_global_model_path": staging_global_model_path,
            "round_global_model_path": transfer_paths["received_global_model_path"],
        },
    )
    _emit(
        progress_callback,
        f"{hospital_name}: received {round_name} global model at {transfer_paths['received_global_model_path']}",
    )

    _emit(progress_callback, f"{hospital_name}: starting local training for {round_name}...")
    training_result = train_local_model(
        config_path=config_path,
        dataset_filename=dataset_filename,
        initial_model_path=transfer_paths["received_global_model_path"],
        local_update_path=transfer_paths["local_update_path"],
        round_name=round_name,
    )
    append_log_entry(
        runtime_log_path,
        title="Local training completed for federated round",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name,
            "dataset_path": training_result["dataset_path"],
            "local_model_path": training_result["model_path"],
            "local_update_path": training_result["local_update_path"],
            "validation_accuracy": round(training_result["validation_accuracy"], 6),
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
        },
    )
    _emit(progress_callback, f"{hospital_name}: local update sent for {round_name}.")

    return {
        "hospital_name": hospital_name,
        "round_name": round_name,
        "received_global_model_path": transfer_paths["received_global_model_path"],
        "training_result": training_result,
        "send_result": send_result,
        "runtime_log_path": runtime_log_path,
        "transfer_log_path": send_result["transfer_log_path"],
    }
