"""Tests for parameterized slot generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from slotting.generate_slots import build_slots, donor_slot_ids, source_slot_ids, write_slots_csv


def test_source_and_donor_slot_ids() -> None:
    assert source_slot_ids(3) == ["S01", "S02", "S03"]
    assert donor_slot_ids(2) == ["D01", "D02"]


def test_build_slots_counts_and_types() -> None:
    slots = build_slots(source_count=4, donor_count=2)
    assert len(slots) == 6
    assert sum(1 for s in slots if s.slot_type.value == "source") == 4
    assert sum(1 for s in slots if s.slot_type.value == "destination") == 2
    assert all(s.occupied_by_pallet_id is None for s in slots)


def test_write_slots_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "slots.csv"
    write_slots_csv(path, source_count=10, donor_count=4)
    df = pd.read_csv(path)
    assert list(df["slot_id"]) == [f"S{i:02d}" for i in range(1, 11)] + [
        f"D{i:02d}" for i in range(1, 5)
    ]
    assert df["occupied_by_pallet_id"].fillna("").eq("").all()
