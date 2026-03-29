"""Aggregator-facing wrapper around the shared diabetes model."""

from __future__ import annotations

from FedLite_Project.Shared_Assets.shared_model_helpers.model_helpers import DiabetesClassifier


def build_global_diabetes_model(input_dim: int, hidden_dim: int = 16) -> DiabetesClassifier:
    """Build the shared diabetes model used by hospitals and the aggregator."""
    return DiabetesClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
