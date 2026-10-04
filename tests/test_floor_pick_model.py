"""Floor partial / top-off pick decomposition."""

from __future__ import annotations

from slotting.floor_pick_model import PickPolicy, decompose_lp_pick_fulfillment
from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.shortage_analysis import ShortageKind, analyze_waved_shortage

from conftest import AS_OF, minimal_state, slotted_pallet


def test_partial_miss_when_face_below_demand() -> None:
    policy = PickPolicy(partial_pick_model=True)
    lp, partial, top_off = decompose_lp_pick_fulfillment(
        face_layers=8,
        waved_layers=12,
        as_of_hour=21,
        policy=policy,
    )
    assert lp == 8
    assert partial == 4
    assert top_off == 0


def test_top_off_when_face_thinner_than_min_lp() -> None:
    policy = PickPolicy(partial_pick_model=True)
    lp, partial, top_off = decompose_lp_pick_fulfillment(
        face_layers=2,
        waved_layers=6,
        as_of_hour=21,
        policy=policy,
    )
    assert lp == 0
    assert partial == 0
    assert top_off == 6


def test_small_late_shift_pick_top_off() -> None:
    policy = PickPolicy(partial_pick_model=True, late_shift_small_pick_topoff_max_layers=6)
    lp, partial, top_off = decompose_lp_pick_fulfillment(
        face_layers=10,
        waved_layers=4,
        as_of_hour=21,
        policy=policy,
    )
    assert lp == 0
    assert partial == 0
    assert top_off == 4


def test_analyze_waved_shortage_partial_kind() -> None:
    pallet, slot = slotted_pallet("P1", "SKU-A", layers=8)
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=12,
            due_at=AS_OF.replace(hour=21),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(pallets=[pallet], slots=[slot], orders=orders, as_of=AS_OF.replace(hour=21))
    rows = analyze_waved_shortage(
        state,
        ship_date=AS_OF.date(),
        due_hours=(21,),
        pick_policy=PickPolicy(partial_pick_model=True),
    )
    # 8 face layers @ 9in default → 5 LP-pickable; 12 waved → 7 height-blocked
    assert rows[0].kind == ShortageKind.HEIGHT_BLOCKED
    assert rows[0].shortage_layers == 7


def test_due_hour_uses_bleed_waved_for_partial_decompose() -> None:
    """Hour-21 due slice uses total 4h waved depth for partial pick decompose."""
    pallet, slot = slotted_pallet("P1", "SKU-A", layers=16)
    ship = AS_OF.date()
    as_of = AS_OF.replace(hour=21)
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=12,
            due_at=AS_OF.replace(hour=22),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id="O2",
            item_id="SKU-A",
            qty_layers=8,
            due_at=as_of,
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[pallet], slots=[slot], orders=orders, as_of=as_of)
    rows = analyze_waved_shortage(
        state,
        ship_date=ship,
        due_hours=(21,),
        pick_policy=PickPolicy(partial_pick_model=True),
    )
    assert len(rows) == 1
    assert rows[0].waved_layers == 8
    assert rows[0].kind == ShortageKind.HEIGHT_BLOCKED
    assert rows[0].shortage_layers == 15
