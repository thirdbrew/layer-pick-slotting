"""Tests for backup-row swap physics."""

from __future__ import annotations

from datetime import timedelta

from slotting.backup_row import empty_resupply_slots, plan_backup_swap
from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    Slot,
    SlotType,
    WorldState,
)

from conftest import AS_OF


def _source(slot_id: str, *, face: str | None = None, backup: str | None = None) -> Slot:
    return Slot(
        slot_id=slot_id,
        slot_type=SlotType.SOURCE,
        occupied_by_pallet_id=face,
        backup_pallet_id=backup,
    )


def test_empty_resupply_requires_no_backup_pallet() -> None:
    slots = [
        _source("S01", face="P1", backup="P2"),
        _source("S02", face="P3"),
        _source("S03"),
    ]
    assert [s.slot_id for s in empty_resupply_slots(slots)] == ["S02", "S03"]


def test_backup_swap_relocates_face_to_empty_resupply() -> None:
    pallets = [
        Pallet(pallet_id="FACE", item_id="SKU-A", layers_remaining=5, current_slot="S01", arrived_at=AS_OF),
        Pallet(pallet_id="BACK", item_id="SKU-B", layers_remaining=6, current_slot=None, arrived_at=AS_OF),
    ]
    slots = [
        _source("S01", face="FACE", backup="BACK"),
        _source("S02"),
    ]
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-B",
            qty_layers=2,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = WorldState(pallets=pallets, slots=slots, orders=orders, exclusions=[], as_of=AS_OF)
    plan = plan_backup_swap(state, slots[0], waved_items={"SKU-B"})
    assert plan is not None
    assert plan.resupply_slot_id == "S02"
    assert plan.to_storage is False
    assert plan.face_pallet_id == "FACE"


def test_backup_swap_blocked_when_no_resupply_and_no_storage() -> None:
    pallets = [
        Pallet(pallet_id="FACE", item_id="SKU-A", layers_remaining=5, current_slot="S01", arrived_at=AS_OF),
        Pallet(pallet_id="BACK", item_id="SKU-B", layers_remaining=6, current_slot=None, arrived_at=AS_OF),
    ]
    slots = [_source("S01", face="FACE", backup="BACK")]
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-B",
            qty_layers=2,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = WorldState(pallets=pallets, slots=slots, orders=orders, exclusions=[], as_of=AS_OF)
    plan = plan_backup_swap(
        state, slots[0], waved_items={"SKU-B"}, allow_storage_fallback=False
    )
    assert plan is None
