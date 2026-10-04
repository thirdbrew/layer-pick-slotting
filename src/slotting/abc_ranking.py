"""ABC classification helpers for setup / slotting priority."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

_ABC_ORDER = {"A": 0, "B": 1, "C": 2}

_DEFAULT_SKUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "site" / "skus.csv"
)

_abc_cache: dict[str, dict[str, str]] = {}


def abc_rank_key(abc_class: str | None) -> int:
    """Lower sort key = higher priority (A before B before C)."""
    if not abc_class:
        return len(_ABC_ORDER)
    return _ABC_ORDER.get(str(abc_class).strip().upper(), len(_ABC_ORDER))


def load_abc_by_item(path: Path | None = None, *, reload: bool = False) -> dict[str, str]:
    """Load ``item_id -> abc_class`` from ``skus.csv`` (``abc_class`` column)."""
    path = path or _DEFAULT_SKUS_PATH
    key = str(path.resolve())
    if not reload and key in _abc_cache:
        return _abc_cache[key]

    result: dict[str, str] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames and "abc_class" in reader.fieldnames:
                for row in reader:
                    item_id = (row.get("item_id") or "").strip()
                    abc = (row.get("abc_class") or "").strip().upper()
                    if item_id and abc in _ABC_ORDER:
                        result[item_id] = abc
    _abc_cache[key] = result
    return result


def setup_sort_key(
    item_id: str,
    layers: int,
    due,
    *,
    abc_by_item: Mapping[str, str] | None = None,
    waved_at_setup_layers: int = 0,
    early_wave_layers: int = 0,
    visible_at_setup_layers: int = 0,
    peak_wave_layers: int = 0,
    order_line_count: int = 0,
    total_cases: int = 0,
    cp_velocity_cases: int = 0,
    has_specs: bool = True,
) -> tuple:
    """
    Sort key for setup ranking.

    Priority: waved-at-setup → early waves → visible-at-setup → peak single-wave
    spike → repeat order lines → total cases → historical CP velocity → specs
    present → ABC → due → id.
    """
    abc = (abc_by_item or {}).get(item_id)
    return (
        -waved_at_setup_layers,
        -early_wave_layers,
        -visible_at_setup_layers,
        -peak_wave_layers,
        -order_line_count,
        -total_cases,
        -layers,
        -cp_velocity_cases,
        0 if has_specs else 1,
        abc_rank_key(abc),
        due,
        item_id,
    )
