"""Helpers for logging LTX transfer events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from FedLite_Project.Shared_Assets.common_utilities.common_utils import append_log_entry


def log_transfer_event(log_path: Path, title: str, details: dict[str, Any]) -> Path:
    """Append a human-readable transfer event entry."""
    return append_log_entry(log_path, title=title, details=details)
