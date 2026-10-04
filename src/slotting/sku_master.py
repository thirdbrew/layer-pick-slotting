"""SKU master helpers for setup ranking (dims, ABC, pickline velocity)."""

from __future__ import annotations

import csv
from pathlib import Path

_DEFAULT_SKUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "site" / "skus.csv"
)

_cache: dict[str, dict[str, dict[str, float | int | str]]] = {}


def _load_rows(path: Path | None = None, *, reload: bool = False) -> dict[str, dict[str, float | int | str]]:
    path = path or _DEFAULT_SKUS_PATH
    key = str(path.resolve())
    if not reload and key in _cache:
        return _cache[key]

    rows: dict[str, dict[str, float | int | str]] = {}
    if not path.is_file():
        _cache[key] = rows
        return rows

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue
            rows[item_id] = row
    _cache[key] = rows
    return rows


def cases_per_layer(item_id: str, *, default: float = 10.0, path: Path | None = None) -> float:
    row = _load_rows(path).get(item_id, {})
    raw = row.get("cases_per_layer")
    try:
        val = float(raw)
        return val if val > 0 else default
    except (TypeError, ValueError):
        return default


def cp_velocity_cases(item_id: str, *, path: Path | None = None) -> int:
    """Recent case-pick volume (e.g. a 6-week case-pick pickline history)."""
    row = _load_rows(path).get(item_id, {})
    raw = row.get("cp_cases_6w") or row.get("cp_velocity_cases")
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return 0


def has_pallet_specs(item_id: str, *, path: Path | None = None) -> bool:
    row = _load_rows(path).get(item_id, {})
    layers = row.get("layers_per_pallet")
    cases = row.get("cases_per_layer")
    return bool(str(layers or "").strip() and str(cases or "").strip())
