"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from slotting.data_loader import CsvDataSource
from slotting.models import (
    Exclusion,
    OrderLine,
    Pallet,
    Slot,
    SlotType,
    WorldState,)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
SYNTHETIC_ORDERS = (
    Path(__file__).resolve().parent.parent / "data" / "synthetic" / "orders.csv"
)
AS_OF = datetime(2026, 6, 12, 8, 0, 0)


def load_sample_state() -> WorldState:
    return CsvDataSource(SAMPLE_DIR).load_world_state(as_of=AS_OF)


def minimal_state(
    *,
    pallets: list[Pallet] | None = None,
    orders: list[OrderLine] | None = None,
    exclusions: list[Exclusion] | None = None,
    slots: list[Slot] | None = None,
    as_of: datetime = AS_OF,
) -> WorldState:
    default_slots = [
        Slot(slot_id=f"S{i:02d}", slot_type=SlotType.SOURCE, occupied_by_pallet_id=None)
        for i in range(1, 11)
    ]
    return WorldState(
        pallets=pallets or [],
        slots=slots or default_slots,
        orders=orders or [],
        exclusions=exclusions or [],
        as_of=as_of,
    )


def slotted_pallet(
    pallet_id: str,
    item_id: str,
    layers: int,
    slot_id: str = "S01",
) -> tuple[Pallet, Slot]:
    pallet = Pallet(
        pallet_id=pallet_id,
        item_id=item_id,
        layers_remaining=layers,
        current_slot=slot_id,
        arrived_at=AS_OF,
    )
    slot = Slot(slot_id=slot_id, slot_type=SlotType.SOURCE, occupied_by_pallet_id=pallet_id)
    return pallet, slot
