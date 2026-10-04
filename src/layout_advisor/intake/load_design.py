"""Load and validate design intake files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from layout_advisor.models.design import DesignIntake


def _load_dict(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON/YAML object at top level")
    return data


def load_design_intake(design_dir: Path) -> DesignIntake:
    """Load design intake from *design_dir* (first matching JSON/YAML file)."""
    design_dir = Path(design_dir)
    candidates = sorted(
        p
        for p in design_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"No design intake JSON/YAML found in {design_dir}. "
            "Expected design_intake.json or similar."
        )
    return DesignIntake.model_validate(_load_dict(candidates[0]))
