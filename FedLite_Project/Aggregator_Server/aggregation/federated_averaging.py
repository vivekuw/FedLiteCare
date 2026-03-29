"""FedAvg aggregation helpers for combining hospital model updates."""

from __future__ import annotations

from typing import Any

import torch


def _validate_matching_keys(state_dicts: list[dict[str, torch.Tensor]]) -> None:
    reference_keys = list(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        if list(state_dict.keys()) != reference_keys:
            raise ValueError("Hospital model updates do not share the same parameter structure.")


def _average_tensor_group(tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total_weight = sum(weights)
    base_dtype = tensors[0].dtype
    averaged_tensor = tensors[0].detach().clone().float() * weights[0]

    for tensor, weight in zip(tensors[1:], weights[1:]):
        averaged_tensor += tensor.detach().float() * weight

    averaged_tensor = averaged_tensor / total_weight
    if base_dtype.is_floating_point:
        return averaged_tensor.to(base_dtype)
    return averaged_tensor.round().to(base_dtype)


def federated_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Average compatible model state dictionaries."""
    if not state_dicts:
        raise ValueError("At least one hospital model update is required for aggregation.")

    _validate_matching_keys(state_dicts)
    normalized_weights = weights or [1.0] * len(state_dicts)
    if len(normalized_weights) != len(state_dicts):
        raise ValueError("The number of aggregation weights must match the number of state dictionaries.")

    aggregated_state: dict[str, torch.Tensor] = {}
    for parameter_name in state_dicts[0]:
        parameter_group = [state_dict[parameter_name] for state_dict in state_dicts]
        aggregated_state[parameter_name] = _average_tensor_group(parameter_group, normalized_weights)
    return aggregated_state


def _validate_shared_metadata(items: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    reference = items[0]
    for candidate in items[1:]:
        if candidate != reference:
            raise ValueError(f"Hospital checkpoints contain mismatched '{field_name}' metadata.")
    return reference


def _average_numeric_lists(values: list[list[float]]) -> list[float]:
    if not values:
        return []

    length = len(values[0])
    for item in values[1:]:
        if len(item) != length:
            raise ValueError("Preprocessing statistics use inconsistent lengths across hospital updates.")

    averaged: list[float] = []
    for index in range(length):
        averaged.append(sum(item[index] for item in values) / len(values))
    return averaged


def average_preprocessing_metadata(preprocessing_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a compatible global preprocessing payload from hospital metadata."""
    if not preprocessing_items:
        raise ValueError("At least one preprocessing payload is required.")

    feature_columns = _validate_shared_metadata(
        [dict(feature_columns=item["feature_columns"]) for item in preprocessing_items],
        "feature_columns",
    )["feature_columns"]
    target_column = _validate_shared_metadata(
        [dict(target_column=item["target_column"]) for item in preprocessing_items],
        "target_column",
    )["target_column"]

    return {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "impute_values": _average_numeric_lists([list(item["impute_values"]) for item in preprocessing_items]),
        "feature_means": _average_numeric_lists([list(item["feature_means"]) for item in preprocessing_items]),
        "feature_stds": _average_numeric_lists([list(item["feature_stds"]) for item in preprocessing_items]),
    }


def aggregate_hospital_checkpoints(
    hospital_checkpoints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate compatible hospital checkpoints into a new global checkpoint payload."""
    if len(hospital_checkpoints) < 2:
        raise ValueError("At least two hospital checkpoints are required for federated averaging.")

    hospital_names = list(hospital_checkpoints.keys())
    checkpoints = [hospital_checkpoints[name] for name in hospital_names]
    reference_model_config = _validate_shared_metadata(
        [dict(model_config=checkpoint["model_config"]) for checkpoint in checkpoints],
        "model_config",
    )["model_config"]

    aggregated_state_dict = federated_average_state_dicts(
        [checkpoint["model_state_dict"] for checkpoint in checkpoints]
    )
    aggregated_preprocessing = average_preprocessing_metadata(
        [checkpoint["preprocessing"] for checkpoint in checkpoints]
    )

    training_metrics: dict[str, float] = {}
    metric_names = {"loss", "accuracy"}
    for metric_name in metric_names:
        metric_values = [
            checkpoint.get("training_metrics", {}).get(metric_name)
            for checkpoint in checkpoints
            if checkpoint.get("training_metrics", {}).get(metric_name) is not None
        ]
        if metric_values:
            training_metrics[metric_name] = sum(metric_values) / len(metric_values)

    return {
        "model_state_dict": aggregated_state_dict,
        "model_config": reference_model_config,
        "preprocessing": aggregated_preprocessing,
        "training_metrics": training_metrics,
        "history": [],
        "aggregation_metadata": {
            "algorithm": "FedAvg",
            "hospital_names": hospital_names,
            "num_hospitals": len(hospital_names),
        },
    }
