"""Version saving helpers for global model checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from FedLite_Project.Shared_Assets.common_utilities.common_utils import ensure_directory


def save_global_model_versions(
    global_models_dir: Path,
    latest_filename: str,
    round_name: str,
    checkpoint_payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Save a round-specific global checkpoint and refresh the latest checkpoint."""
    ensure_directory(global_models_dir)
    round_checkpoint_path = global_models_dir / f"{round_name}_global_model.pt"
    latest_checkpoint_path = global_models_dir / latest_filename

    torch.save(checkpoint_payload, round_checkpoint_path)
    torch.save(checkpoint_payload, latest_checkpoint_path)
    return round_checkpoint_path, latest_checkpoint_path


def load_version_history(history_path: Path) -> dict[str, Any]:
    """Load version history metadata if present."""
    if not history_path.exists():
        return {"versions": []}
    return json.loads(history_path.read_text(encoding="utf-8"))


def append_version_history(history_path: Path, version_entry: dict[str, Any]) -> dict[str, Any]:
    """Persist a new global model version entry."""
    history = load_version_history(history_path)
    history.setdefault("versions", []).append(version_entry)
    ensure_directory(history_path.parent)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
