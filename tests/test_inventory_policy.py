"""Pending-completion tracking and reserve discipline."""

from __future__ import annotations

from datetime import datetime

from slotting.inventory_policy import (
    is_pending_completion,
    mark_removed_from_pick,
    prune_orphan_reserve,
    select_best_supply_pallet,
)
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet

from conftest import AS_OF, slotted_pallet


def test_mark_removed_tracks_partial_pallet() -> None:
    pallet, slot = slotted_pallet("P1", "SKU-A", layers=4)
    mark_removed_from_pick(pallet, as_of=AS_OF, source_slot_id=slot.slot_id)
    assert is_pending_completion(pallet)
    assert pallet.last_source_slot_id == slot.slot_id


def test_select_best_supply_prefers_pending_lowest_layers() -> None:
    pending = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=2,
        current_slot=None,
        arrived_at=AS_OF,
        removed_from_pick_at=AS_OF,
    )
    full = Pallet(
        pallet_id="P2",
        item_id="A",
        layers_remaining=20,
        current_slot=None,
        arrived_at=AS_OF,
    )
    chosen = select_best_supply_pallet([full, pending])
    assert chosen is not None
    assert chosen.pallet_id == "P1"


def test_prune_orphan_drops_no_demand_keeps_pending() -> None:
    from datetime import date

    ship = date(2026, 3, 3)
    pending = Pallet(
        pallet_id="P1",
        item_id="ORPHAN",
        layers_remaining=3,
        current_slot=None,
        arrived_at=AS_OF,
        removed_from_pick_at=AS_OF,
    )
    live = Pallet(
        pallet_id="P2",
        item_id="LIVE",
        layers_remaining=8,
        current_slot=None,
        arrived_at=AS_OF,
    )
    orders = [
        OrderLine(
            order_id="O1",
            item_id="LIVE",
            qty_layers=5,
            due_at=datetime(2026, 3, 3, 12, 0),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    kept = prune_orphan_reserve([pending, live], orders, ship_date=ship, as_of=AS_OF)
    ids = {p.pallet_id for p in kept}
    assert "P1" in ids
    assert "P2" in ids
