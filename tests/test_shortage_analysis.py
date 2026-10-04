"""Per-SKU shortage classification."""

from __future__ import annotations

from datetime import timedelta

from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, Slot, SlotType
from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits
from slotting.shortage_analysis import (
    ShortageKind,
    analyze_waved_shortage,
    due_window_layers_by_item,
    summarize_shortage,
)

from conftest import AS_OF, minimal_state, slotted_pallet


def test_true_shortage_when_no_inventory() -> None:
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=10,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(pallets=[], slots=[], orders=orders)
    rows = analyze_waved_shortage(state, ship_date=AS_OF.date())
    assert len(rows) == 1
    assert rows[0].kind == ShortageKind.TRUE_SHORTAGE
    summary = summarize_shortage(rows)
    assert summary["true_shortage_layers"] == 10


def test_lp_ready_when_on_face() -> None:
    pallet, slot = slotted_pallet("P1", "SKU-A", layers=10)
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=5,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(pallets=[pallet], slots=[slot], orders=orders)
    rows = analyze_waved_shortage(state, ship_date=AS_OF.date())
    assert rows[0].kind == ShortageKind.LP_READY


def test_height_blocked_when_face_exceeds_pick_reach() -> None:
    face_slot = Slot(
        slot_id="S01",
        slot_type=SlotType.SOURCE,
        occupied_by_pallet_id="P1",
    )
    face = Pallet(
        pallet_id="P1",
        item_id="SKU-A",
        layers_remaining=8,
        current_slot="S01",
        arrived_at=AS_OF,
    )
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=8,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(pallets=[face], slots=[face_slot], orders=orders)
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=48.0)):
        rows = analyze_waved_shortage(state, ship_date=AS_OF.date())
    assert rows[0].kind == ShortageKind.HEIGHT_BLOCKED
    assert rows[0].shortage_layers == 3


def test_partial_miss_when_pickable_equals_face_depth() -> None:
    pallet, slot = slotted_pallet("P1", "SKU-A", layers=5)
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=8,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(pallets=[pallet], slots=[slot], orders=orders)
    rows = analyze_waved_shortage(state, ship_date=AS_OF.date())
    assert rows[0].kind == ShortageKind.LP_PARTIAL_MISS
    assert rows[0].shortage_layers == 3


def test_due_hours_filter_excludes_early_waves() -> None:
    orders = [
        OrderLine(
            order_id="O1",
            item_id="EARLY",
            qty_layers=8,
            due_at=AS_OF.replace(hour=7),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id="O2",
            item_id="LATE",
            qty_layers=6,
            due_at=AS_OF.replace(hour=21),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[], slots=[], orders=orders, as_of=AS_OF.replace(hour=21))
    assert due_window_layers_by_item(state, (21,)) == {"LATE": 6}
    rows = analyze_waved_shortage(state, ship_date=AS_OF.date(), due_hours=(21,))
    assert [r.item_id for r in rows] == ["LATE"]
