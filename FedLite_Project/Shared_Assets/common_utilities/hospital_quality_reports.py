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
PATIENT_METADATA_FIELDS = [
    "patient_case_id",
    "first_name",
    "last_name",
    "gender",
    "date_of_birth",
    "contact_number",
    "department",
    "attending_doctor",
    "address",
    "visit_notes",
]
PATIENT_METADATA_LABELS = {
    "patient_case_id": "Patient Case ID",
    "first_name": "First Name",
    "last_name": "Last Name",
    "gender": "Gender",
    "date_of_birth": "Date of Birth",
    "contact_number": "Contact Number",
    "department": "Department",
    "attending_doctor": "Attending Doctor",
    "address": "Address",
    "visit_notes": "Visit Notes",
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
    prediction_registry_dir = ensure_directory(
        resolve_path(
            hospital_root,
            str(settings.get("prediction_registry_dir", "reports/predictions")),
        )
    )

    return settings, {
        "config_path": resolved_config_path,
        "hospital_root": hospital_root,
        "uploads_dir": uploads_dir,
        "reports_dir": reports_dir,
        "validation_reports_dir": validation_reports_dir,
        "prediction_reports_dir": prediction_reports_dir,
        "prediction_registry_dir": prediction_registry_dir,
    }


def _get_report_path(directory: Path, prefix: str, suffix: str = ".txt") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{prefix}_{timestamp}{suffix}"


def _write_report(report_path: Path, lines: list[str]) -> Path:
    ensure_directory(report_path.parent)
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


def generate_patient_case_id() -> str:
    """Create a readable local patient case identifier for demo use."""
    return "CASE-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def normalize_patient_metadata(patient_metadata: dict[str, Any] | None) -> dict[str, str]:
    """Normalize patient metadata values and ensure a case ID exists."""
    normalized = {
        field_name: str((patient_metadata or {}).get(field_name, "") or "").strip()
        for field_name in PATIENT_METADATA_FIELDS
    }
    if not normalized["patient_case_id"]:
        normalized["patient_case_id"] = generate_patient_case_id()
    return normalized


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
    patient_metadata = normalize_patient_metadata(prediction_result.get("patient_metadata"))

    lines = [
        "FedLiteCare Prediction Report",
        "============================",
        "",
        f"Hospital ID: {hospital_name}",
        f"Date/Time: {datetime.now().isoformat(timespec='seconds')}",
        f"Patient Case ID: {patient_metadata['patient_case_id'] or 'N/A'}",
        f"Model Path: {prediction_result.get('model_path', 'N/A')}",
        "",
        "Patient Details",
        "---------------",
    ]
    for field_name in PATIENT_METADATA_FIELDS:
        if field_name == "patient_case_id":
            continue
        raw_value = patient_metadata.get(field_name, "")
        lines.append(f"- {PATIENT_METADATA_LABELS[field_name]}: {raw_value or 'N/A'}")

    lines.extend(
        [
            "",
        "Patient Input Values",
        "--------------------",
        ]
    )
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


def append_prediction_registry_row(
    config_path: Path,
    patient_values: dict[str, str | float],
    prediction_result: dict[str, Any],
) -> Path:
    """Append one patient prediction intake row to a hospital CSV registry."""
    settings, paths = _load_report_context(config_path)
    registry_path = resolve_path(
        paths["prediction_registry_dir"],
        str(settings.get("prediction_registry_filename", "patient_prediction_registry.csv")),
    )
    return _append_prediction_row_to_csv(
        csv_path=registry_path,
        hospital_name=str(settings.get("hospital_name", paths["hospital_root"].name)),
        patient_values=patient_values,
        prediction_result=prediction_result,
    )


def append_predicted_patients_row(
    config_path: Path,
    patient_values: dict[str, str | float],
    prediction_result: dict[str, Any],
) -> Path:
    """Append one screened patient row to a hospital-facing predicted-patients CSV."""
    settings, paths = _load_report_context(config_path)
    predicted_patients_path = resolve_path(
        paths["prediction_registry_dir"],
        str(settings.get("predicted_patients_filename", "predicted_patients.csv")),
    )
    return _append_prediction_row_to_csv(
        csv_path=predicted_patients_path,
        hospital_name=str(settings.get("hospital_name", paths["hospital_root"].name)),
        patient_values=patient_values,
        prediction_result=prediction_result,
    )


def _append_prediction_row_to_csv(
    csv_path: Path,
    hospital_name: str,
    patient_values: dict[str, str | float],
    prediction_result: dict[str, Any],
) -> Path:
    """Append one prediction row to a CSV file, expanding headers safely when needed."""
    ensure_directory(csv_path.parent)
    patient_metadata = normalize_patient_metadata(prediction_result.get("patient_metadata"))

    base_fieldnames = [
        "hospital_id",
        "date_time",
        "patient_case_id",
    ]
    patient_metadata_fieldnames = [
        field_name for field_name in PATIENT_METADATA_FIELDS if field_name != "patient_case_id"
    ]
    feature_fieldnames = list(patient_values.keys())
    result_fieldnames = [
        "prediction_result",
        "predicted_label",
        "confidence_score",
        "model_path",
        "prediction_report_path",
        "confirmed_outcome",
        "eligible_for_training",
        "notes",
    ]
    fieldnames = base_fieldnames + patient_metadata_fieldnames + feature_fieldnames + result_fieldnames

    row = {
        "hospital_id": hospital_name,
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "patient_case_id": patient_metadata["patient_case_id"],
        "prediction_result": str(prediction_result.get("result_label", "")),
        "predicted_label": str(prediction_result.get("predicted_label", "")),
        "confidence_score": (
            ""
            if prediction_result.get("confidence_score") is None
            else f"{float(prediction_result['confidence_score']):.4f}"
        ),
        "model_path": str(prediction_result.get("model_path", "")),
        "prediction_report_path": str(prediction_result.get("report_path", "")),
        "confirmed_outcome": "",
        "eligible_for_training": "No - needs confirmed outcome",
        "notes": "Prediction intake only. Do not use for training until the true outcome is confirmed.",
    }
    for field_name in patient_metadata_fieldnames:
        row[field_name] = patient_metadata.get(field_name, "")
    for field_name, raw_value in patient_values.items():
        row[field_name] = str(raw_value)

    existing_fieldnames: list[str] = []
    existing_rows: list[dict[str, str]] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    merged_fieldnames = (
        fieldnames
        if not existing_fieldnames
        else existing_fieldnames + [
            field_name for field_name in fieldnames if field_name not in existing_fieldnames
        ]
    )

    if existing_fieldnames and merged_fieldnames != existing_fieldnames:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow({field_name: existing_row.get(field_name, "") for field_name in merged_fieldnames})
            writer.writerow({field_name: row.get(field_name, "") for field_name in merged_fieldnames})
        return csv_path

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
        if not existing_fieldnames:
            writer.writeheader()
        writer.writerow({field_name: row.get(field_name, "") for field_name in merged_fieldnames})

    return csv_path


def export_csv_prediction_results(
    config_path: Path,
    input_path: Path,
    input_records: list[dict[str, str]],
    prediction_rows: list[dict[str, Any]],
    model_path: Path,
) -> Path:
    """Save CSV prediction results for demo review and screenshots."""
    settings, paths = _load_report_context(config_path)
    hospital_name = str(settings.get("hospital_name", paths["hospital_root"].name))
    output_path = _get_report_path(
        paths["prediction_reports_dir"],
        prefix=f"{input_path.stem}_predictions",
        suffix=".csv",
    )

    base_fieldnames = list(input_records[0].keys()) if input_records else []
    extra_fieldnames = [
        "prediction_probability",
        "predicted_label",
        "prediction_result",
        "hospital_id",
        "model_path",
        "scored_at",
    ]
    fieldnames = base_fieldnames + [
        field_name for field_name in extra_fieldnames if field_name not in base_fieldnames
    ]

    ensure_directory(output_path.parent)
    scored_at = datetime.now().isoformat(timespec="seconds")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, prediction_row in zip(input_records, prediction_rows, strict=True):
            predicted_label = int(prediction_row["predicted_label"])
            row = dict(record)
            row["prediction_probability"] = f"{float(prediction_row['probability']):.4f}"
            row["predicted_label"] = str(predicted_label)
            row["prediction_result"] = (
                "High diabetes risk" if predicted_label == 1 else "Lower diabetes risk"
            )
            row["hospital_id"] = hospital_name
            row["model_path"] = str(model_path)
            row["scored_at"] = scored_at
            writer.writerow(row)

    return output_path


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
