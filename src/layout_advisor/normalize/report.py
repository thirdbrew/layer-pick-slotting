"""Write normalization_report.json alongside every run."""

from __future__ import annotations

import json
from pathlib import Path

from layout_advisor.models.provenance import NormalizationReport


def write_normalization_report(report: NormalizationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
