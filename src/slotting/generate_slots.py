"""Parameterized slot layout generation for layout sweep."""

from __future__ import annotations

import csv
from pathlib import Path

from slotting.models import Slot, SlotType


def source_slot_ids(count: int) -> list[str]:
    """Return ``S01``..``S{count}`` slot identifiers."""
    if count < 1:
        raise ValueError("source_count must be >= 1")
    return [f"S{i:02d}" for i in range(1, count + 1)]


def donor_slot_ids(count: int) -> list[str]:
    """Return ``D01``..``D{count}`` destination slot identifiers."""
    if count < 1:
        raise ValueError("donor_count must be >= 1")
    return [f"D{i:02d}" for i in range(1, count + 1)]


def build_slots(*, source_count: int, donor_count: int) -> list[Slot]:
    """Build empty source + destination slots for a candidate layout."""
    slots: list[Slot] = []
    for slot_id in source_slot_ids(source_count):
        slots.append(Slot(slot_id=slot_id, slot_type=SlotType.SOURCE, occupied_by_pallet_id=None))
    for slot_id in donor_slot_ids(donor_count):
        slots.append(
            Slot(slot_id=slot_id, slot_type=SlotType.DESTINATION, occupied_by_pallet_id=None)
        )
    return slots


def write_slots_csv(path: Path, *, source_count: int, donor_count: int) -> None:
    """Write ``slots.csv`` with ``S01..S{n}`` and ``D01..D{m}`` rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    slots = build_slots(source_count=source_count, donor_count=donor_count)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["slot_id", "slot_type", "occupied_by_pallet_id"])
        writer.writeheader()
        for slot in slots:
            writer.writerow(
                {
                    "slot_id": slot.slot_id,
                    "slot_type": slot.slot_type.value,
                    "occupied_by_pallet_id": slot.occupied_by_pallet_id or "",
                }
            )
