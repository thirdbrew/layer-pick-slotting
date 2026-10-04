"""Tests for CSV data loading."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from slotting.data_loader import CsvDataSource
from slotting.models import OrderArea, SlotType
from conftest import AS_OF, SAMPLE_DIR, load_sample_state


def test_load_sample_csvs() -> None:
    state = load_sample_state()
    assert len(state.pallets) == 10
    assert len(state.source_slots) == 10
    assert len(state.exclusions) == 1


def test_blank_slot_parsed_as_none() -> None:
    source = CsvDataSource(SAMPLE_DIR)
    state = source.load_world_state(as_of=AS_OF)
    unslotted = [p for p in state.pallets if p.current_slot is None]
    assert any(p.pallet_id == "P020" for p in unslotted)


def test_slot_types() -> None:
    state = load_sample_state()
    dest = [s for s in state.slots if s.slot_type == SlotType.DESTINATION]
    assert len(dest) == 4


def test_order_areas() -> None:
    state = load_sample_state()
    other = [o for o in state.orders if o.area == OrderArea.OTHER_AREA]
    assert len(other) == 1


def test_orders_without_arrived_at_column_load_as_none() -> None:
    # Regression: making arrived_at required broke every pre-censor feed.
    state = load_sample_state()
    assert all(o.arrived_at is None for o in state.orders)


def test_orders_arrived_at_parsed_when_present(tmp_path: Path) -> None:
    for name in ("pallets.csv", "slots.csv", "orders.csv", "exclusions.csv"):
        shutil.copy(SAMPLE_DIR / name, tmp_path / name)
    orders_path = tmp_path / "orders.csv"
    lines = orders_path.read_text(encoding="utf-8").splitlines()
    stamped = [lines[0] + ",arrived_at"]
    stamped += [f"{row},2026-06-11 09:30:00" for row in lines[1:]]
    # Blank arrived_at on the last row must stay None
    stamped[-1] = lines[-1] + ","
    orders_path.write_text("\n".join(stamped) + "\n", encoding="utf-8")

    state = CsvDataSource(tmp_path).load_world_state(as_of=AS_OF)
    assert state.orders[0].arrived_at == datetime(2026, 6, 11, 9, 30)
    assert state.orders[-1].arrived_at is None
