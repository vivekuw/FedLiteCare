"""Runtime path helpers for source and packaged FedLiteCare clients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT_NAME = "FedLite_Project"


@dataclass(frozen=True)
class HospitalRuntimePaths:
    """Resolved runtime paths for a hospital node."""

    mode: str
    runtime_base_dir: Path
    hospital_root: Path
    config_dir: Path
    communication_dir: Path
    config_path: Path
    transfer_config_path: Path


def is_frozen_app() -> bool:
    """Return True when running from a bundled executable."""
    return bool(getattr(sys, "frozen", False))


def _looks_like_project_root(path: Path) -> bool:
    return (
        path.name == PROJECT_ROOT_NAME
        and (path / "Shared_Assets" / "common_utilities").is_dir()
        and (path / "Hospital_A").is_dir()
    )


def find_project_root(current_path: Path) -> Path:
    """Walk upward from a file or directory until the FedLiteCare root is found."""
    start_path = current_path.resolve()
    if start_path.is_file():
        start_path = start_path.parent

    for candidate in (start_path, *start_path.parents):
        if _looks_like_project_root(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate the '{PROJECT_ROOT_NAME}' project root from: {current_path}"
    )


def bootstrap_project_imports(current_path: Path) -> Path | None:
    """Add the project parent to sys.path when running directly from source."""
    if is_frozen_app():
        return None

    project_root = find_project_root(current_path)
    workspace_root = project_root.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    return project_root


def _resolve_packaged_hospital_root(runtime_base_dir: Path, hospital_name: str) -> Path:
    candidates = [
        runtime_base_dir / hospital_name,
        runtime_base_dir / PROJECT_ROOT_NAME / hospital_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_hospital_runtime_paths(
    hospital_name: str,
    current_path: Path,
) -> HospitalRuntimePaths:
    """Resolve hospital config and communication paths for source or EXE runtime."""
    if is_frozen_app():
        runtime_base_dir = Path(sys.executable).resolve().parent
        hospital_root = _resolve_packaged_hospital_root(runtime_base_dir, hospital_name)
        mode = "packaged"
    else:
        project_root = find_project_root(current_path)
        runtime_base_dir = project_root
        hospital_root = project_root / hospital_name
        mode = "source"

    config_dir = hospital_root / "config"
    communication_dir = hospital_root / "communication"
    return HospitalRuntimePaths(
        mode=mode,
        runtime_base_dir=runtime_base_dir,
        hospital_root=hospital_root,
        config_dir=config_dir,
        communication_dir=communication_dir,
        config_path=config_dir / "client_config.yaml",
        transfer_config_path=communication_dir / "transfer_config.yaml",
    )
