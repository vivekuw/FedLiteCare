"""Reusable PyTorch model and checkpoint helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn


class DiabetesClassifier(nn.Module):
    """A lightweight feed-forward classifier for tabular diabetes data."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        reduced_dim = max(hidden_dim // 2, 4)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate loss and accuracy on a dataset."""
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((predictions == batch_labels).sum().item())
            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return {"loss": average_loss, "accuracy": accuracy}


def train_classifier(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    global_model_state: dict[str, torch.Tensor] | None = None,
    mu: float = 0.0,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float]]:
    """Train the model using FedProx-style proximal optimization if enabled."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = {"loss": float("inf"), "accuracy": 0.0}

    # Pre-calculate global tensors on device once if mu > 0
    global_tensors = {}
    if global_model_state is not None and mu > 0.0:
        for name, param in global_model_state.items():
            global_tensors[name] = param.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_examples = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)

            # Add FedProx proximal term: (mu / 2) * ||w - w_t||^2
            if global_tensors and mu > 0.0:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_tensors:
                        proximal_term += ((param - global_tensors[name]) ** 2).sum()
                loss += (mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()

            batch_size = batch_labels.size(0)
            running_loss += loss.item() * batch_size
            seen_examples += batch_size

        train_loss = running_loss / seen_examples
        validation_metrics = evaluate_model(model, validation_loader, device)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_metrics)

        if validation_metrics["loss"] < best_metrics["loss"]:
            best_metrics = validation_metrics
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history, best_metrics


def predict_probabilities(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run model inference and return probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        return torch.sigmoid(logits).cpu().squeeze(1)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    preprocessing: dict[str, Any],
    model_config: dict[str, Any],
    training_metrics: dict[str, float],
    history: list[dict[str, float]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the trained model and preprocessing metadata for later inference."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "preprocessing": preprocessing,
            "training_metrics": training_metrics,
            "history": history,
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved model checkpoint in a PyTorch-version-tolerant way."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DiabetesClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
