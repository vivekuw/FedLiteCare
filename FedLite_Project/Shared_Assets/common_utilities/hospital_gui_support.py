"""Shared support helpers for the FedLiteCare hospital Tkinter client."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import torch

from FedLite_Project.Shared_Assets.common_utilities.common_utils import ensure_directory, resolve_path
from FedLite_Project.Shared_Assets.common_utilities.hospital_quality_reports import (
    get_latest_report_file,
    read_labeled_report_value,
)
from FedLite_Project.Shared_Assets.common_utilities.local_ml_pipeline import load_hospital_context
from FedLite_Project.Shared_Assets.data_preprocessing_helpers.preprocessing_utils import load_csv_records
from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import load_checkpoint

DEFAULT_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def list_available_dataset_files(config_path: Path) -> list[str]:
    """List CSV files inside the hospital uploads directory."""
    _, paths = load_hospital_context(config_path)
    return sorted(
        file_path.name
        for file_path in paths["uploads_dir"].glob("*.csv")
        if file_path.is_file()
    )


def copy_dataset_into_uploads(config_path: Path, source_path: Path) -> Path:
    """Copy a chosen CSV into the hospital uploads folder without overwriting by default."""
    _, paths = load_hospital_context(config_path)
    ensure_directory(paths["uploads_dir"])

    destination_path = paths["uploads_dir"] / source_path.name
    if destination_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_path = paths["uploads_dir"] / f"{source_path.stem}_{timestamp}{source_path.suffix}"

    shutil.copy2(source_path, destination_path)
    return destination_path


def read_recent_log_lines(log_path: Path, max_lines: int = 40) -> str:
    """Return the last lines from a log file for display in the GUI."""
    if not log_path.exists():
        return f"No log available yet: {log_path.name}"

    lines = log_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-max_lines:])


def _load_checkpoint_if_available(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    _, checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    return checkpoint


def get_feature_columns_for_hospital(config_path: Path) -> list[str]:
    """Resolve the patient feature fields expected by the hospital model."""
    settings, paths = load_hospital_context(config_path)
    model_path = resolve_path(paths["models_dir"], str(settings["model_filename"]))
    checkpoint = _load_checkpoint_if_available(model_path)
    if checkpoint is not None:
        return list(checkpoint["preprocessing"]["feature_columns"])

    dataset_path = resolve_path(paths["uploads_dir"], str(settings["dataset_filename"]))
    if dataset_path.exists():
        records = load_csv_records(dataset_path)
        target_column = str(settings.get("target_column", "Outcome"))
        return [column for column in records[0].keys() if column != target_column]

    return list(DEFAULT_FEATURE_COLUMNS)


def _resolve_local_model_version(checkpoint: dict[str, Any] | None, model_path: Path) -> str:
    if checkpoint is None:
        return "Not trained"

    metadata = dict(checkpoint.get("metadata", {}))
    round_name = str(metadata.get("round_name", "")).strip()
    if round_name:
        return round_name

    if model_path.exists():
        return datetime.fromtimestamp(model_path.stat().st_mtime).strftime("local %Y-%m-%d %H:%M")
    return "Available"


def _resolve_global_model_version(received_global_models_dir: Path) -> str:
    model_files = sorted(
        (
            file_path for file_path in received_global_models_dir.glob("*.pt")
            if file_path.is_file()
        ),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )
    if not model_files:
        return "Not synced"

    checkpoint = _load_checkpoint_if_available(model_files[0])
    if checkpoint is None:
        return model_files[0].stem

    aggregation_metadata = dict(checkpoint.get("aggregation_metadata", {}))
    round_name = str(aggregation_metadata.get("round_name", "")).strip()
    return round_name or model_files[0].stem


def get_hospital_dashboard_status(config_path: Path) -> dict[str, Any]:
    """Build a lightweight dashboard summary for the hospital GUI."""
    settings, paths = load_hospital_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    model_path = resolve_path(paths["models_dir"], str(settings["model_filename"]))
    local_checkpoint = _load_checkpoint_if_available(model_path)
    training_metrics = dict(local_checkpoint.get("training_metrics", {})) if local_checkpoint else {}

    logs = {
        "training": resolve_path(paths["logs_dir"], str(settings.get("training_log_filename", "training.log"))),
        "prediction": resolve_path(paths["logs_dir"], str(settings.get("prediction_log_filename", "prediction.log"))),
        "sync": resolve_path(paths["logs_dir"], str(settings.get("federated_log_filename", "federated_client.log"))),
        "transfer": resolve_path(paths["logs_dir"], "transfer.log"),
    }
    latest_validation_report = get_latest_report_file(paths["validation_reports_dir"])
    latest_prediction_report = get_latest_report_file(paths["prediction_reports_dir"])
    latest_validation_status = (
        None
        if latest_validation_report is None
        else read_labeled_report_value(latest_validation_report, "Validation Status")
    )

    return {
        "hospital_name": hospital_name,
        "active_dataset": str(settings["dataset_filename"]),
        "dataset_files": list_available_dataset_files(config_path),
        "model_path": model_path,
        "model_exists": model_path.exists(),
        "local_model_version": _resolve_local_model_version(local_checkpoint, model_path),
        "current_global_version": _resolve_global_model_version(paths["received_global_models_dir"]),
        "training_accuracy": training_metrics.get("accuracy"),
        "training_loss": training_metrics.get("loss"),
        "latest_validation_report": latest_validation_report,
        "latest_validation_status": latest_validation_status,
        "latest_prediction_report": latest_prediction_report,
        "logs": logs,
    }
