"""Tests for floor miss log vs sim shortage comparison."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from slotting.floor_validation import (
    aggregate_floor_by_sku,
    load_floor_miss_log,
    overlap_metrics,
    sim_shortage_by_sku,
)
from slotting.shortage_analysis import ShortageKind, SkuShortageRow


def test_load_floor_miss_log(tmp_path: Path) -> None:
    path = tmp_path / "miss.csv"
    path.write_text(
        "item_id,ship_date,hour,miss_layers,miss_type\n"
        "SKU1,2026-02-23,21,5,true_shortage\n"
        "SKU2,2026-02-23,20,3,wait\n"
        "SKU1,2026-02-23,19,2,true_shortage\n"
    )
    rows = load_floor_miss_log(path, ship_date=date(2026, 2, 23))
    assert len(rows) == 3
    ranked = aggregate_floor_by_sku(rows, hours=(19, 20, 21))
    assert ranked["SKU1"] == 7
    assert ranked["SKU2"] == 3


def test_overlap_metrics() -> None:
    sim = {"A": 10, "B": 8, "C": 6, "D": 4}
    floor = {"B": 9, "C": 7, "E": 5, "F": 3}
    metrics = overlap_metrics(sim, floor, top_n=3)
    assert metrics.overlap_skus == ["B", "C"]
    assert metrics.overlap_count == 2
    assert metrics.recall_pct == round(100 * 2 / 3, 1)


def test_sim_shortage_filter() -> None:
    rows = [
        SkuShortageRow(
            item_id="A",
            waved_layers=4,
            face_layers=0,
            rack_layers=0,
            unslotted_layers=0,
            total_inventory=0,
            ship_day_layers=4,
            late_shift_layers=4,
            kind=ShortageKind.TRUE_SHORTAGE,
            shortage_layers=4,
        ),
        SkuShortageRow(
            item_id="B",
            waved_layers=3,
            face_layers=0,
            rack_layers=8,
            unslotted_layers=0,
            total_inventory=8,
            ship_day_layers=3,
            late_shift_layers=0,
            kind=ShortageKind.RACK_ONLY,
            shortage_layers=3,
        ),
    ]
    true_only = sim_shortage_by_sku(rows, kinds={ShortageKind.TRUE_SHORTAGE})
    all_short = sim_shortage_by_sku(
        rows, kinds={ShortageKind.TRUE_SHORTAGE, ShortageKind.RACK_ONLY}
    )
    assert true_only == {"A": 4}
    assert all_short == {"A": 4, "B": 3}
