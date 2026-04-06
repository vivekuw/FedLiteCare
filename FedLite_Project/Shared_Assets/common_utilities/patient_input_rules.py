"""Prediction input rules for safe hospital-side patient scoring."""

from __future__ import annotations

from typing import Any

PATIENT_INPUT_RULES: dict[str, dict[str, Any]] = {
    "Pregnancies": {
        "minimum": 0.0,
        "maximum": 20.0,
        "unit": "count",
        "integer_only": True,
    },
    "Glucose": {
        "minimum": 40.0,
        "maximum": 300.0,
        "unit": "mg/dL",
        "integer_only": False,
    },
    "BloodPressure": {
        "minimum": 40.0,
        "maximum": 180.0,
        "unit": "mmHg",
        "integer_only": False,
    },
    "SkinThickness": {
        "minimum": 5.0,
        "maximum": 99.0,
        "unit": "mm",
        "integer_only": False,
    },
    "Insulin": {
        "minimum": 0.0,
        "maximum": 900.0,
        "unit": "uU/mL",
        "integer_only": False,
    },
    "BMI": {
        "minimum": 10.0,
        "maximum": 70.0,
        "unit": "kg/m^2",
        "integer_only": False,
    },
    "DiabetesPedigreeFunction": {
        "minimum": 0.05,
        "maximum": 3.0,
        "unit": "score",
        "integer_only": False,
    },
    "Age": {
        "minimum": 21.0,
        "maximum": 90.0,
        "unit": "years",
        "integer_only": True,
    },
}


def get_patient_input_rules(feature_columns: list[str]) -> dict[str, dict[str, Any]]:
    """Return the known manual-prediction rules for the provided feature columns."""
    return {
        column: PATIENT_INPUT_RULES[column]
        for column in feature_columns
        if column in PATIENT_INPUT_RULES
    }


def format_rule_for_display(column: str, rule: dict[str, Any]) -> str:
    """Render a compact single-line rule description."""
    minimum = float(rule["minimum"])
    maximum = float(rule["maximum"])
    unit = str(rule.get("unit", "")).strip()
    integer_suffix = " | whole number only" if bool(rule.get("integer_only", False)) else ""
    unit_suffix = f" {unit}" if unit else ""
    return f"{column}: {minimum:g}-{maximum:g}{unit_suffix}{integer_suffix}"


def build_range_guide_text(feature_columns: list[str]) -> str:
    """Render a multiline range guide for the GUI."""
    lines = [
        format_rule_for_display(column, rule)
        for column, rule in get_patient_input_rules(feature_columns).items()
    ]
    return "\n".join(lines)
