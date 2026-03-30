"""Hospital-side dataset validation and readable report helpers."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from FedLite_Project.Shared_Assets.common_utilities.common_utils import (
    ensure_directory,
    load_simple_yaml_config,
    resolve_path,
)

DIABETES_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
DIABETES_TARGET_COLUMN = "Outcome"
DIABETES_REQUIRED_COLUMNS = DIABETES_FEATURE_COLUMNS + [DIABETES_TARGET_COLUMN]
DIABETES_UPPER_BOUNDS = {
    "Pregnancies": 25.0,
    "Glucose": 400.0,
    "BloodPressure": 250.0,
    "SkinThickness": 120.0,
    "Insulin": 1200.0,
    "BMI": 80.0,
    "DiabetesPedigreeFunction": 5.0,
    "Age": 120.0,
}


def _load_report_context(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    resolved_config_path = config_path.resolve()
    settings = load_simple_yaml_config(resolved_config_path)
    hospital_root = resolved_config_path.parent.parent

    uploads_dir = ensure_directory(resolve_path(hospital_root, str(settings["uploads_dir"])))
    reports_dir = ensure_directory(resolve_path(hospital_root, str(settings.get("reports_dir", "reports"))))
    validation_reports_dir = ensure_directory(
        resolve_path(hospital_root, str(settings.get("validation_reports_dir", "reports/validation")))
    )
    prediction_reports_dir = ensure_directory(
        resolve_path(hospital_root, str(settings.get("prediction_reports_dir", "reports/predictions")))
    )

    return settings, {
        "config_path": resolved_config_path,
        "hospital_root": hospital_root,
        "uploads_dir": uploads_dir,
        "reports_dir": reports_dir,
        "validation_reports_dir": validation_reports_dir,
        "prediction_reports_dir": prediction_reports_dir,
    }


def _get_report_path(directory: Path, prefix: str, suffix: str = ".txt") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{prefix}_{timestamp}{suffix}"


def _write_report(report_path: Path, lines: list[str]) -> Path:
    ensure_directory(report_path.parent)
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


def validate_training_dataset(
    config_path: Path,
    dataset_filename: str | None = None,
) -> dict[str, Any]:
    """Validate a hospital training CSV and save a readable report."""
    settings, paths = _load_report_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    dataset_name = dataset_filename or str(settings["dataset_filename"])
    dataset_path = resolve_path(paths["uploads_dir"], dataset_name)
    report_path = _get_report_path(
        paths["validation_reports_dir"],
        prefix=f"{dataset_path.stem}_validation",
    )

    generated_at = datetime.now().isoformat(timespec="seconds")
    try:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise ValueError("Dataset file has no header row.")
            records = list(reader)
            if not records:
                raise ValueError("Dataset file contains no data rows.")
    except Exception as error:
        result = {
            "hospital_name": hospital_name,
            "dataset_path": dataset_path,
            "generated_at": generated_at,
            "status": "FAIL",
            "is_valid": False,
            "row_count": 0,
            "missing_columns": list(DIABETES_REQUIRED_COLUMNS),
            "missing_value_counts": {},
            "rows_with_missing_features": 0,
            "invalid_rows": [str(error)],
            "suspicious_rows": [],
            "report_path": report_path,
        }
        _write_validation_report(result)
        return result

    missing_columns = [column for column in DIABETES_REQUIRED_COLUMNS if column not in fieldnames]
    missing_value_counts = {
        column: 0
        for column in DIABETES_REQUIRED_COLUMNS
        if column in fieldnames
    }
    invalid_rows: list[str] = []
    suspicious_rows: list[str] = []
    rows_with_missing_features = 0

    for row_index, record in enumerate(records, start=2):
        row_issues: list[str] = []
        row_warnings: list[str] = []
        row_has_missing_feature = False

        for column in DIABETES_REQUIRED_COLUMNS:
            if column not in fieldnames:
                continue

            raw_value = (record.get(column) or "").strip()
            if not raw_value:
                missing_value_counts[column] += 1
                if column == DIABETES_TARGET_COLUMN:
                    row_issues.append("Outcome is missing")
                else:
                    row_has_missing_feature = True
                continue

            try:
                numeric_value = float(raw_value)
            except ValueError:
                row_issues.append(f"{column} is non-numeric ('{raw_value}')")
                continue

            if column != DIABETES_TARGET_COLUMN and numeric_value < 0:
                row_issues.append(f"{column} is negative ({numeric_value})")

            if column == DIABETES_TARGET_COLUMN and numeric_value not in {0.0, 1.0}:
                row_issues.append(f"Outcome should be 0 or 1, got {numeric_value}")

            upper_bound = DIABETES_UPPER_BOUNDS.get(column)
            if upper_bound is not None and numeric_value > upper_bound:
                row_warnings.append(
                    f"{column}={numeric_value} exceeds expected upper bound {upper_bound}"
                )

        if row_has_missing_feature:
            rows_with_missing_features += 1
        if row_issues:
            invalid_rows.append(f"Row {row_index}: " + "; ".join(row_issues))
        elif row_warnings:
            suspicious_rows.append(f"Row {row_index}: " + "; ".join(row_warnings))

    has_warnings = any(count > 0 for count in missing_value_counts.values()) or bool(suspicious_rows)
    is_valid = not missing_columns and not invalid_rows
    status = "PASS" if is_valid and not has_warnings else "PASS WITH WARNINGS" if is_valid else "FAIL"

    result = {
        "hospital_name": hospital_name,
        "dataset_path": dataset_path,
        "generated_at": generated_at,
        "status": status,
        "is_valid": is_valid,
        "row_count": len(records),
        "missing_columns": missing_columns,
        "missing_value_counts": missing_value_counts,
        "rows_with_missing_features": rows_with_missing_features,
        "invalid_rows": invalid_rows,
        "suspicious_rows": suspicious_rows,
        "report_path": report_path,
    }
    _write_validation_report(result)
    return result


def _write_validation_report(validation_result: dict[str, Any]) -> Path:
    missing_counts = {
        column: count
        for column, count in dict(validation_result["missing_value_counts"]).items()
        if count > 0
    }
    lines = [
        "FedLiteCare Dataset Validation Report",
        "===================================",
        "",
        f"Hospital ID: {validation_result['hospital_name']}",
        f"Generated At: {validation_result['generated_at']}",
        f"Dataset Path: {validation_result['dataset_path']}",
        f"Validation Status: {validation_result['status']}",
        f"Ready For Training: {'Yes' if validation_result['is_valid'] else 'No'}",
        f"Total Rows: {validation_result['row_count']}",
        "",
        "Required Columns",
        "----------------",
        ", ".join(DIABETES_REQUIRED_COLUMNS),
        "",
        "Missing Required Columns",
        "------------------------",
        "None" if not validation_result["missing_columns"] else "\n".join(
            f"- {column}" for column in validation_result["missing_columns"]
        ),
        "",
        "Missing Values",
        "--------------",
        "None" if not missing_counts else "\n".join(
            f"- {column}: {count}" for column, count in missing_counts.items()
        ),
        "",
        f"Rows With Missing Feature Values: {validation_result['rows_with_missing_features']}",
        "",
        "Invalid Rows",
        "------------",
        "None" if not validation_result["invalid_rows"] else "\n".join(
            f"- {issue}" for issue in validation_result["invalid_rows"]
        ),
        "",
        "Suspicious Rows",
        "---------------",
        "None" if not validation_result["suspicious_rows"] else "\n".join(
            f"- {issue}" for issue in validation_result["suspicious_rows"]
        ),
        "",
        "Recommendation",
        "--------------",
        _build_validation_recommendation(validation_result),
    ]
    return _write_report(Path(validation_result["report_path"]), lines)


def _build_validation_recommendation(validation_result: dict[str, Any]) -> str:
    if not validation_result["is_valid"]:
        return (
            "Fix the invalid rows or missing required columns before training. "
            "The current dataset should not be used yet."
        )

    if validation_result["status"] == "PASS WITH WARNINGS":
        return (
            "Training can continue, but review the missing values and suspicious rows. "
            "Missing feature values will be imputed by the preprocessing step."
        )

    return "Dataset looks ready for local training."


def create_prediction_report(
    config_path: Path,
    patient_values: dict[str, str | float],
    prediction_result: dict[str, Any],
) -> Path:
    """Create a readable single-patient prediction report for demo use."""
    settings, paths = _load_report_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    report_path = _get_report_path(paths["prediction_reports_dir"], prefix="patient_prediction")
    result_label = str(prediction_result.get("result_label", "Unknown result"))
    confidence_score = prediction_result.get("confidence_score")

    lines = [
        "FedLiteCare Prediction Report",
        "============================",
        "",
        f"Hospital ID: {hospital_name}",
        f"Date/Time: {datetime.now().isoformat(timespec='seconds')}",
        f"Model Path: {prediction_result.get('model_path', 'N/A')}",
        "",
        "Patient Input Values",
        "--------------------",
    ]
    for field_name, raw_value in patient_values.items():
        lines.append(f"- {field_name}: {raw_value}")

    lines.extend(
        [
            "",
            "Prediction Outcome",
            "------------------",
            f"Prediction Result: {result_label}",
            f"Predicted Label: {prediction_result.get('predicted_label', 'N/A')}",
            "Confidence Score: N/A" if confidence_score is None else f"Confidence Score: {confidence_score:.4f}",
        ]
    )
    return _write_report(report_path, lines)


def get_latest_report_file(directory: Path, pattern: str = "*.txt") -> Path | None:
    """Return the newest readable report file inside a directory."""
    if not directory.exists():
        return None

    files = [
        file_path
        for file_path in directory.glob(pattern)
        if file_path.is_file()
    ]
    if not files:
        return None
    return max(files, key=lambda file_path: file_path.stat().st_mtime)


def read_labeled_report_value(report_path: Path, label: str) -> str | None:
    """Read a simple `Label: value` entry from a text report."""
    if not report_path.exists():
        return None

    prefix = f"{label}:"
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None
