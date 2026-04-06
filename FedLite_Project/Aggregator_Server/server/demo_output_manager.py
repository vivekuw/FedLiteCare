"""Helpers for exporting demo-friendly round summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from FedLite_Project.Shared_Assets.common_utilities.common_utils import ensure_directory


def _render_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _render_path(value: str | Path | None) -> str:
    if value is None:
        return "N/A"
    return str(value)


def build_demo_summary_text(summary_payload: dict[str, Any]) -> str:
    """Render a readable per-round demo summary."""
    lines = [
        "FedLiteCare Demo Round Summary",
        "==============================",
        "",
        f"Generated At: {summary_payload['generated_at']}",
        f"Round Name: {summary_payload['round_name']}",
        f"Round Number: {summary_payload['round_number']}",
        f"Mode: {summary_payload['mode']}",
        "",
        "Global Model Paths",
        "------------------",
        f"Starting Global Model: {summary_payload['starting_global_model']}",
        f"Versioned Global Model: {summary_payload['new_global_model']}",
        f"Latest Global Model: {summary_payload['latest_global_model']}",
        "",
        "Hospital Results",
        "----------------",
    ]

    for hospital_name in summary_payload["hospital_order"]:
        hospital_summary = summary_payload["hospitals"][hospital_name]
        lines.extend(
            [
                hospital_name,
                f"  Validation Status: {hospital_summary['validation_status']}",
                f"  Validation Accuracy: {_render_metric(hospital_summary['validation_accuracy'])}",
                f"  Validation Loss: {_render_metric(hospital_summary['validation_loss'])}",
                f"  Dataset: {hospital_summary['dataset_filename']}",
                f"  Validation Report: {_render_path(hospital_summary['validation_report_path'])}",
                f"  Local Model: {_render_path(hospital_summary['local_model_path'])}",
                f"  Aggregator Received Update: {_render_path(hospital_summary['received_update_path'])}",
                f"  Bytes Received By Aggregator: {hospital_summary['bytes_received']}",
                "",
            ]
        )

    lines.extend(
        [
            "Key Logs",
            "--------",
            f"Aggregator Runtime Log: {summary_payload['aggregator_runtime_log']}",
            f"Aggregator Log: {summary_payload['aggregator_log']}",
            f"Round Log: {summary_payload['round_log']}",
            "",
            "Demo Export Files",
            "-----------------",
            f"Summary Text: {summary_payload['summary_text_path']}",
            f"Summary JSON: {summary_payload['summary_json_path']}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def export_demo_round_artifacts(
    project_root: Path,
    summary_payload: dict[str, Any],
) -> dict[str, Path]:
    """Write screenshot-friendly and machine-readable demo artifacts."""
    demo_root = ensure_directory(project_root / "Demo_Outputs")
    demo_logs_dir = ensure_directory(demo_root / "demo_logs")
    test_outputs_dir = ensure_directory(demo_root / "test_outputs")
    round_dir = ensure_directory(test_outputs_dir / summary_payload["round_name"])

    summary_text_path = demo_logs_dir / f"{summary_payload['round_name']}_demo_summary.txt"
    summary_json_path = round_dir / "demo_summary.json"
    export_manifest_path = round_dir / "artifact_paths.json"

    enriched_payload = dict(summary_payload)
    enriched_payload["summary_text_path"] = str(summary_text_path)
    enriched_payload["summary_json_path"] = str(summary_json_path)
    enriched_payload["exported_at"] = datetime.now().isoformat(timespec="seconds")

    summary_text_path.write_text(
        build_demo_summary_text(enriched_payload),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(enriched_payload, indent=2),
        encoding="utf-8",
    )
    export_manifest_path.write_text(
        json.dumps(
            {
                "round_name": summary_payload["round_name"],
                "summary_text_path": str(summary_text_path),
                "summary_json_path": str(summary_json_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "summary_text_path": summary_text_path,
        "summary_json_path": summary_json_path,
        "export_manifest_path": export_manifest_path,
    }
