"""Round tracking helpers for the FedLiteCare aggregator server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from FedLite_Project.Shared_Assets.common_utilities.common_utils import ensure_directory


class RoundManager:
    """Track federated aggregation rounds using a lightweight JSON state file."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    @staticmethod
    def format_round_name(round_number: int) -> str:
        """Create a human-readable round identifier."""
        return f"round_{round_number:03d}"

    def _default_state(self) -> dict[str, Any]:
        return {"latest_completed_round": 0, "rounds": []}

    def load_state(self) -> dict[str, Any]:
        """Load persisted round state if present."""
        if not self.state_path.exists():
            return self._default_state()

        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: dict[str, Any]) -> None:
        """Persist the round state to disk."""
        ensure_directory(self.state_path.parent)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def next_round_number(self) -> int:
        """Return the next round number based on completed history."""
        state = self.load_state()
        return int(state.get("latest_completed_round", 0)) + 1

    def record_completed_round(self, round_summary: dict[str, Any]) -> dict[str, Any]:
        """Append a completed round summary and update the latest round counter."""
        state = self.load_state()
        round_number = int(round_summary["round_number"])

        state["latest_completed_round"] = max(
            int(state.get("latest_completed_round", 0)),
            round_number,
        )
        state.setdefault("rounds", []).append(round_summary)
        self.save_state(state)
        return state
