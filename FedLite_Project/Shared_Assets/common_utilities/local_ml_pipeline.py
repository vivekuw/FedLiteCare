"""Hospital-agnostic local ML training and prediction workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.utils.data import DataLoader

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    append_log_entry,
    ensure_directory,
    load_simple_yaml_config,
    resolve_path,
)
from FedLite_Project.Shared_Assets.common_utilities.hospital_quality_reports import (
    create_prediction_report,
    validate_training_dataset,
)
from FedLite_Project.Shared_Assets.data_preprocessing_helpers.preprocessing_utils import (
    fit_preprocessor,
    load_csv_records,
    split_tensor_dataset,
    transform_records,
)
from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import (
    DiabetesClassifier,
    load_checkpoint,
    predict_probabilities,
    save_checkpoint,
    train_classifier,
)


def _resolve_existing_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return path.resolve()


def load_hospital_context(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load config and resolve the hospital-specific directories."""
    resolved_config_path = _resolve_existing_path(config_path)
    settings = load_simple_yaml_config(resolved_config_path)
    hospital_root = resolved_config_path.parent.parent

    uploads_dir = ensure_directory(resolve_path(hospital_root, str(settings["uploads_dir"])))
    models_dir = ensure_directory(resolve_path(hospital_root, str(settings["models_dir"])))
    logs_dir = ensure_directory(
        resolve_path(hospital_root, str(settings.get("logs_dir", "logs")))
    )
    received_global_models_dir = ensure_directory(
        resolve_path(
            hospital_root,
            str(settings.get("received_global_models_dir", "communication/received_global_models")),
        )
    )
    local_updates_dir = ensure_directory(
        resolve_path(
            hospital_root,
            str(settings.get("local_updates_dir", "communication/local_model_updates")),
        )
    )
    reports_dir = ensure_directory(
        resolve_path(
            hospital_root,
            str(settings.get("reports_dir", "reports")),
        )
    )
    validation_reports_dir = ensure_directory(
        resolve_path(
            hospital_root,
            str(settings.get("validation_reports_dir", "reports/validation")),
        )
    )
    prediction_reports_dir = ensure_directory(
        resolve_path(
            hospital_root,
            str(settings.get("prediction_reports_dir", "reports/predictions")),
        )
    )

    return settings, {
        "config_path": resolved_config_path,
        "hospital_root": hospital_root,
        "uploads_dir": uploads_dir,
        "models_dir": models_dir,
        "logs_dir": logs_dir,
        "received_global_models_dir": received_global_models_dir,
        "local_updates_dir": local_updates_dir,
        "reports_dir": reports_dir,
        "validation_reports_dir": validation_reports_dir,
        "prediction_reports_dir": prediction_reports_dir,
    }


def _resolve_training_dataset_path(uploads_dir: Path, dataset_filename: str) -> Path:
    return resolve_path(uploads_dir, dataset_filename)


def _resolve_optional_cli_path(path: Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path
    return _resolve_existing_path(path)


def get_hospital_round_transfer_paths(config_path: Path, round_name: str) -> dict[str, Path]:
    """Return round-specific inbound and outbound model paths for a hospital."""
    settings, paths = load_hospital_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    hospital_slug = hospital_name.lower()

    return {
        "received_global_model_path": paths["received_global_models_dir"] / f"{round_name}_global_model.pt",
        "local_update_path": paths["local_updates_dir"] / f"{round_name}_{hospital_slug}_update.pt",
    }


def train_local_model(
    config_path: Path,
    dataset_filename: str | None = None,
    initial_model_path: Path | None = None,
    local_update_path: Path | None = None,
    round_name: str | None = None,
) -> dict[str, Any]:
    """Train a local diabetes model using hospital-specific config and data."""
    settings, paths = load_hospital_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    seed = int(settings.get("random_seed", 42))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_name = dataset_filename or str(settings["dataset_filename"])
    dataset_path = _resolve_training_dataset_path(paths["uploads_dir"], dataset_name)
    model_path = resolve_path(paths["models_dir"], str(settings["model_filename"]))
    training_log_path = resolve_path(
        paths["logs_dir"],
        str(settings.get("training_log_filename", "training.log")),
    )
    resolved_initial_model_path = _resolve_existing_path(initial_model_path) if initial_model_path else None
    resolved_local_update_path = (
        _resolve_existing_path(local_update_path) if local_update_path else None
    )
    if round_name and resolved_local_update_path is None:
        resolved_local_update_path = get_hospital_round_transfer_paths(config_path, round_name)[
            "local_update_path"
        ]

    validation_result = validate_training_dataset(
        config_path=config_path,
        dataset_filename=dataset_name,
    )
    append_log_entry(
        training_log_path,
        title="Dataset validation completed",
        details={
            "hospital_name": hospital_name,
            "dataset_path": dataset_path,
            "validation_status": validation_result["status"],
            "validation_report_path": validation_result["report_path"],
            "missing_columns": ", ".join(validation_result["missing_columns"]) or "None",
            "invalid_row_count": len(validation_result["invalid_rows"]),
            "suspicious_row_count": len(validation_result["suspicious_rows"]),
            "rows_with_missing_features": validation_result["rows_with_missing_features"],
        },
    )
    if not validation_result["is_valid"]:
        append_log_entry(
            training_log_path,
            title="Training blocked by dataset validation",
            details={
                "hospital_name": hospital_name,
                "dataset_path": dataset_path,
                "validation_status": validation_result["status"],
                "validation_report_path": validation_result["report_path"],
                "status": "failed",
            },
        )
        raise ValueError(
            f"Dataset validation failed for {hospital_name}. See report: {validation_result['report_path']}"
        )

    try:
        records = load_csv_records(dataset_path)
        features, labels, preprocessing = fit_preprocessor(
            records,
            target_column=str(settings.get("target_column", "Outcome")),
        )
        train_dataset, validation_dataset = split_tensor_dataset(
            features,
            labels,
            validation_ratio=float(settings.get("validation_ratio", 0.25)),
            seed=seed,
        )

        batch_size = int(settings.get("batch_size", 8))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        model = DiabetesClassifier(
            input_dim=features.shape[1],
            hidden_dim=int(settings.get("hidden_dim", 16)),
        )
        if resolved_initial_model_path is not None:
            _, base_checkpoint = load_checkpoint(resolved_initial_model_path, torch.device("cpu"))
            model.load_state_dict(base_checkpoint["model_state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model, history, validation_metrics = train_classifier(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=int(settings.get("epochs", 80)),
            learning_rate=float(settings.get("learning_rate", 0.001)),
            device=device,
        )

        save_checkpoint(
            checkpoint_path=model_path,
            model=trained_model,
            preprocessing=preprocessing,
            model_config={
                "input_dim": features.shape[1],
                "hidden_dim": int(settings.get("hidden_dim", 16)),
            },
            training_metrics=validation_metrics,
            history=history,
            metadata={
                "hospital_name": hospital_name,
                "round_name": round_name or "standalone",
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_filename": dataset_name,
            },
        )
        if resolved_local_update_path is not None:
            ensure_directory(resolved_local_update_path.parent)
            shutil.copy2(model_path, resolved_local_update_path)
    except Exception as error:
        append_log_entry(
            training_log_path,
            title="Training failed",
            details={
                "hospital_name": hospital_name,
                "round_name": round_name or "standalone",
                "status": "failed",
                "dataset_path": dataset_path,
                "validation_status": validation_result["status"],
                "validation_report_path": validation_result["report_path"],
                "error": str(error),
            },
        )
        raise

    append_log_entry(
        training_log_path,
        title="Training completed",
        details={
            "hospital_name": hospital_name,
            "round_name": round_name or "standalone",
            "status": "completed",
            "dataset_path": dataset_path,
            "validation_status": validation_result["status"],
            "validation_report_path": validation_result["report_path"],
            "initial_model_path": "None" if resolved_initial_model_path is None else resolved_initial_model_path,
            "model_path": model_path,
            "local_update_path": "None" if resolved_local_update_path is None else resolved_local_update_path,
            "device": device,
            "training_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "validation_loss": round(validation_metrics["loss"], 6),
            "validation_accuracy": round(validation_metrics["accuracy"], 6),
        },
    )

    return {
        "hospital_name": hospital_name,
        "dataset_path": dataset_path,
        "model_path": model_path,
        "initial_model_path": resolved_initial_model_path,
        "local_update_path": resolved_local_update_path,
        "round_name": round_name,
        "log_path": training_log_path,
        "device": str(device),
        "training_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_result": validation_result,
        "validation_report_path": validation_result["report_path"],
    }


def predict_from_csv(
    config_path: Path,
    input_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Generate diabetes predictions for a hospital using its own config and checkpoint."""
    settings, paths = load_hospital_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))

    default_input_path = resolve_path(paths["uploads_dir"], str(settings["dataset_filename"]))
    default_model_path = resolve_path(paths["models_dir"], str(settings["model_filename"]))
    resolved_input_path = _resolve_optional_cli_path(input_path, default_input_path)
    resolved_model_path = _resolve_optional_cli_path(model_path, default_model_path)
    prediction_log_path = resolve_path(
        paths["logs_dir"],
        str(settings.get("prediction_log_filename", "prediction.log")),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(resolved_model_path, device)

    records = load_csv_records(resolved_input_path)
    features, labels = transform_records(records, checkpoint["preprocessing"])
    probabilities = predict_probabilities(model, features, device)
    threshold = 0.5

    prediction_rows: list[dict[str, Any]] = []
    for index, probability in enumerate(probabilities.tolist(), start=1):
        row_result: dict[str, Any] = {
            "row": index,
            "probability": probability,
            "predicted_label": int(probability >= threshold),
        }
        if labels is not None:
            row_result["actual_label"] = int(labels[index - 1].item())
        prediction_rows.append(row_result)

    accuracy = None
    if labels is not None:
        predicted_tensor = torch.tensor(
            [row["predicted_label"] for row in prediction_rows], dtype=torch.float32
        ).unsqueeze(1)
        accuracy = float((predicted_tensor == labels).float().mean().item())

    append_log_entry(
        prediction_log_path,
        title="Prediction run",
        details={
            "hospital_name": hospital_name,
            "input_path": resolved_input_path,
            "model_path": resolved_model_path,
            "rows_scored": len(prediction_rows),
            "positive_predictions": sum(row["predicted_label"] for row in prediction_rows),
            "accuracy": "N/A" if accuracy is None else round(accuracy, 6),
            "status": "completed",
        },
    )

    return {
        "hospital_name": hospital_name,
        "input_path": resolved_input_path,
        "model_path": resolved_model_path,
        "log_path": prediction_log_path,
        "predictions": prediction_rows,
        "accuracy": accuracy,
    }


def predict_from_patient_values(
    config_path: Path,
    patient_values: dict[str, str | float],
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Generate a diabetes prediction for one patient record from in-memory field values."""
    settings, paths = load_hospital_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    default_model_path = resolve_path(paths["models_dir"], str(settings["model_filename"]))
    resolved_model_path = _resolve_optional_cli_path(model_path, default_model_path)
    prediction_log_path = resolve_path(
        paths["logs_dir"],
        str(settings.get("prediction_log_filename", "prediction.log")),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(resolved_model_path, device)
    feature_columns = list(checkpoint["preprocessing"]["feature_columns"])
    record = {
        column: str(patient_values.get(column, "")).strip()
        for column in feature_columns
    }

    features, _ = transform_records([record], checkpoint["preprocessing"])
    probability = float(predict_probabilities(model, features, device).view(-1)[0].item())
    predicted_label = int(probability >= 0.5)

    append_log_entry(
        prediction_log_path,
        title="Single-patient prediction",
        details={
            "hospital_name": hospital_name,
            "model_path": resolved_model_path,
            "probability": round(probability, 6),
            "predicted_label": predicted_label,
            "status": "completed",
        },
    )
    result_label = "High diabetes risk" if predicted_label == 1 else "Lower diabetes risk"
    confidence_score = probability if predicted_label == 1 else 1.0 - probability
    report_path = create_prediction_report(
        config_path=config_path,
        patient_values=record,
        prediction_result={
            "model_path": resolved_model_path,
            "predicted_label": predicted_label,
            "result_label": result_label,
            "confidence_score": confidence_score,
        },
    )
    append_log_entry(
        prediction_log_path,
        title="Prediction report generated",
        details={
            "hospital_name": hospital_name,
            "report_path": report_path,
            "result_label": result_label,
            "confidence_score": round(confidence_score, 6),
        },
    )

    return {
        "hospital_name": hospital_name,
        "model_path": resolved_model_path,
        "log_path": prediction_log_path,
        "feature_columns": feature_columns,
        "probability": probability,
        "predicted_label": predicted_label,
        "confidence_score": confidence_score,
        "result_label": result_label,
        "report_path": report_path,
    }
