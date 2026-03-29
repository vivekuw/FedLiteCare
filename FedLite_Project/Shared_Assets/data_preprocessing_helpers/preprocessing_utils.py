"""CSV loading and preprocessing helpers for diabetes prediction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import TensorDataset

DEFAULT_TARGET_COLUMN = "Outcome"


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    """Load CSV rows into memory while preserving column order."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {csv_path}")
        records = list(reader)

    if not records:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")

    return records


def _get_feature_columns(records: list[dict[str, str]], target_column: str) -> list[str]:
    feature_columns = [column for column in records[0].keys() if column != target_column]
    if not feature_columns:
        raise ValueError("No feature columns were found in the dataset.")
    return feature_columns


def _safe_float(raw_value: str, column_name: str) -> float:
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"Column '{column_name}' contains a non-numeric value: {raw_value}") from error


def fit_preprocessor(
    records: list[dict[str, str]],
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit preprocessing statistics and return normalized tensors."""
    feature_columns = _get_feature_columns(records, target_column)
    collected_values: dict[str, list[float]] = {column: [] for column in feature_columns}
    labels: list[float] = []

    for record in records:
        for column in feature_columns:
            raw_value = (record.get(column) or "").strip()
            if raw_value:
                collected_values[column].append(_safe_float(raw_value, column))

        raw_target = (record.get(target_column) or "").strip()
        if not raw_target:
            raise ValueError(f"Training data must include the target column '{target_column}'.")
        labels.append(_safe_float(raw_target, target_column))

    impute_values = [
        sum(collected_values[column]) / len(collected_values[column]) if collected_values[column] else 0.0
        for column in feature_columns
    ]

    feature_rows: list[list[float]] = []
    for record in records:
        row: list[float] = []
        for index, column in enumerate(feature_columns):
            raw_value = (record.get(column) or "").strip()
            if raw_value:
                row.append(_safe_float(raw_value, column))
            else:
                row.append(impute_values[index])
        feature_rows.append(row)

    feature_tensor = torch.tensor(feature_rows, dtype=torch.float32)
    means = feature_tensor.mean(dim=0)
    stds = feature_tensor.std(dim=0, unbiased=False)
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)
    normalized_features = (feature_tensor - means) / stds
    label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    preprocessing = {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "impute_values": impute_values,
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
    }
    return normalized_features, label_tensor, preprocessing


def transform_records(
    records: list[dict[str, str]],
    preprocessing: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply saved preprocessing settings to new CSV rows."""
    feature_columns = list(preprocessing["feature_columns"])
    target_column = str(preprocessing["target_column"])
    impute_values = list(preprocessing["impute_values"])
    means = torch.tensor(preprocessing["feature_means"], dtype=torch.float32)
    stds = torch.tensor(preprocessing["feature_stds"], dtype=torch.float32)

    feature_rows: list[list[float]] = []
    labels: list[float] = []
    rows_with_target = 0

    for record in records:
        row: list[float] = []
        for index, column in enumerate(feature_columns):
            raw_value = (record.get(column) or "").strip()
            if raw_value:
                row.append(_safe_float(raw_value, column))
            else:
                row.append(float(impute_values[index]))
        feature_rows.append(row)

        raw_target = (record.get(target_column) or "").strip()
        if raw_target:
            rows_with_target += 1
            labels.append(_safe_float(raw_target, target_column))

    if rows_with_target not in {0, len(records)}:
        raise ValueError("Prediction CSV must either include the target column for every row or omit it entirely.")

    features = torch.tensor(feature_rows, dtype=torch.float32)
    normalized_features = (features - means) / stds
    label_tensor = (
        torch.tensor(labels, dtype=torch.float32).unsqueeze(1) if rows_with_target == len(records) else None
    )
    return normalized_features, label_tensor


def split_tensor_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    validation_ratio: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    """Create deterministic train/validation splits."""
    total_rows = features.size(0)
    if total_rows < 2:
        raise ValueError("At least two rows are required to create train and validation splits.")

    validation_size = max(1, int(total_rows * validation_ratio))
    validation_size = min(validation_size, total_rows - 1)

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_rows, generator=generator)
    validation_indices = shuffled_indices[:validation_size]
    training_indices = shuffled_indices[validation_size:]

    training_dataset = TensorDataset(features[training_indices], labels[training_indices])
    validation_dataset = TensorDataset(features[validation_indices], labels[validation_indices])
    return training_dataset, validation_dataset
