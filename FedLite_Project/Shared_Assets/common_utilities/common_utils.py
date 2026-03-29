"""Common helpers shared across local FedLiteCare workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Return the FedLite_Project root directory."""
    return Path(__file__).resolve().parents[2]


def ensure_directory(path: Path) -> Path:
    """Create a directory when needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    """Resolve absolute or base-relative paths in a reusable way."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""

    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a flat YAML file containing simple key/value pairs."""
    config: dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_scalar(value)
    return config


def append_log_entry(log_path: Path, title: str, details: dict[str, Any]) -> Path:
    """Append a readable timestamped entry to a hospital log file."""
    ensure_directory(log_path.parent)
    timestamp = datetime.now().isoformat(timespec="seconds")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {title}\n")
        for key, value in details.items():
            rendered_value = value if not isinstance(value, Path) else str(value)
            handle.write(f"{key}: {rendered_value}\n")
        handle.write("\n")

    return log_path
