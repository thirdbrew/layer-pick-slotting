"""Idle-assist opportunistic freezer→rack staging."""

from __future__ import annotations

from datetime import date, datetime

from slotting.generate_slots import build_slots
from slotting.inventory import apply_idle_assist_staging, idle_assist_pull_candidates
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, SlotType, WorldState
from slotting.resupply_rack import build_rack_slots_aligned


def _slots_with_rack() -> list:
    return build_slots(source_count=4, donor_count=2) + build_rack_slots_aligned(
        stations_per_flank=2
    )


def test_idle_assist_candidates_require_thin_active_face() -> None:
    as_of = datetime(2026, 2, 23, 10, 0)
    slots = _slots_with_rack()
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face = Pallet(
        pallet_id="FACE-1",
        item_id="HOT",
        layers_remaining=3,
        current_slot=source.slot_id,
        arrived_at=as_of,
    )
    source.occupied_by_pallet_id = face.pallet_id
    orders = [
        OrderLine(
            order_id="O1",
            item_id="HOT",
            qty_layers=20,
            due_at=datetime(2026, 2, 23, 12, 0),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = WorldState(
        pallets=[face],
        slots=slots,
        orders=orders,
        exclusions=[],
        as_of=as_of,
    )
    cands = idle_assist_pull_candidates(state, face_layer_threshold=4)
    assert cands and cands[0][0] == "HOT"

    face.layers_remaining = 8
    state = WorldState(
        pallets=[face],
        slots=slots,
        orders=orders,
        exclusions=[],
        as_of=as_of,
    )
    assert idle_assist_pull_candidates(state, face_layer_threshold=4) == []


def test_apply_idle_assist_stages_to_rack() -> None:
    ship = date(2026, 2, 23)
    as_of = datetime(2026, 2, 23, 10, 0)
    slots = _slots_with_rack()
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face = Pallet(
        pallet_id="FACE-1",
        item_id="HOT",
        layers_remaining=2,
        current_slot=source.slot_id,
        arrived_at=as_of,
    )
    source.occupied_by_pallet_id = face.pallet_id
    orders = [
        OrderLine(
            order_id="O1",
            item_id="HOT",
            qty_layers=20,
            due_at=datetime(2026, 2, 23, 12, 0),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = WorldState(
        pallets=[face],
        slots=slots,
        orders=orders,
        exclusions=[],
        as_of=as_of,
    )
    pallets, out_slots, n, counts, events = apply_idle_assist_staging(
        [face],
        slots,
        state,
        orders,
        ship,
        pulled_today_by_item={},
        max_per_hour=2,
        prefer_rack=True,
    )
    assert n == 1
    assert counts["HOT"] == 1
    idle = next(p for p in pallets if p.pallet_id.startswith("IDLE-"))
    assert idle.current_slot is not None
    rack_ids = {
        s.slot_id for s in out_slots if s.slot_type == SlotType.RESUPPLY_RACK
    }
    assert idle.current_slot in rack_ids
    assert events[0][0] == "HOT"


def test_idle_assist_never_consumes_last_per_sku_pull() -> None:
    """Opportunistic staging yields the final per-SKU pull to reactive shortage
    pulls. Otherwise idle pulls can exhaust the 3/SKU/day budget in calm
    hours, cap-blocking emergency pulls when the afternoon shortage hits."""
    ship = date(2026, 2, 23)
    as_of = datetime(2026, 2, 23, 10, 0)
    slots = _slots_with_rack()
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face = Pallet(
        pallet_id="FACE-1",
        item_id="HOT",
        layers_remaining=2,
        current_slot=source.slot_id,
        arrived_at=as_of,
    )
    source.occupied_by_pallet_id = face.pallet_id
    orders = [
        OrderLine(
            order_id="O1",
            item_id="HOT",
            qty_layers=20,
            due_at=datetime(2026, 2, 23, 12, 0),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = WorldState(
        pallets=[face], slots=slots, orders=orders, exclusions=[], as_of=as_of
    )
    # HOT already pulled twice today; max_per_sku=3 -> one allowance left,
    # and idle assist must leave it for a reactive shortage pull
    _, _, n, counts, _ = apply_idle_assist_staging(
        [face],
        slots,
        state,
        orders,
        ship,
        pulled_today_by_item={"HOT": 2},
        max_per_hour=2,
        max_per_sku=3,
        prefer_rack=True,
    )
    assert n == 0
    assert counts["HOT"] == 2  # untouched


def test_unconsumed_idle_pallets_return_to_freezer() -> None:
    """Rack-staged IDLE leftovers never enter carry; face-active ones stay."""
    from slotting.backtest import return_unconsumed_idle_pallets

    as_of = datetime(2026, 2, 23, 22, 0)
    slots = _slots_with_rack()
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    rack = next(s for s in slots if s.slot_type == SlotType.RESUPPLY_RACK)
    on_face = Pallet(
        pallet_id="IDLE-0001",
        item_id="A",
        layers_remaining=5,
        current_slot=source.slot_id,
        arrived_at=as_of,
    )
    source.occupied_by_pallet_id = on_face.pallet_id
    leftover = Pallet(
        pallet_id="IDLE-0002",
        item_id="B",
        layers_remaining=8,
        current_slot=rack.slot_id,
        arrived_at=as_of,
    )
    rack.occupied_by_pallet_id = leftover.pallet_id
    ordinary = Pallet(
        pallet_id="RES-1",
        item_id="C",
        layers_remaining=8,
        current_slot=None,
        arrived_at=as_of,
    )
    survivors = return_unconsumed_idle_pallets([on_face, leftover, ordinary], slots)
    ids = {p.pallet_id for p in survivors}
    assert ids == {"IDLE-0001", "RES-1"}  # leftover returned, face + ordinary kept
    assert rack.occupied_by_pallet_id is None


def test_rack_headroom_guard_blocks_staging_when_rack_tight() -> None:
    """min_free_rack_slots keeps rack slack for displacement / promotes.

    Failure mode: idle pallets congest the rack, starving the displacement
    lever of landing slots and multiplying the day's waits."""
    ship = date(2026, 2, 23)
    as_of = datetime(2026, 2, 23, 10, 0)
    slots = _slots_with_rack()
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face = Pallet(
        pallet_id="FACE-1",
        item_id="HOT",
        layers_remaining=2,
        current_slot=source.slot_id,
        arrived_at=as_of,
    )
    source.occupied_by_pallet_id = face.pallet_id
    # fill rack down to exactly 2 free slots
    rack = [s for s in slots if s.slot_type == SlotType.RESUPPLY_RACK]
    fillers = []
    for i, slot in enumerate(rack[:-2]):
        filler = Pallet(
            pallet_id=f"FILL-{i}",
            item_id=f"COLD-{i}",
            layers_remaining=8,
            current_slot=slot.slot_id,
            arrived_at=as_of,
        )
        slot.occupied_by_pallet_id = filler.pallet_id
        fillers.append(filler)
    orders = [
        OrderLine(
            order_id="O1",
            item_id="HOT",
            qty_layers=20,
            due_at=datetime(2026, 2, 23, 12, 0),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    all_pallets = [face, *fillers]
    state = WorldState(
        pallets=all_pallets,
        slots=slots,
        orders=orders,
        exclusions=[],
        as_of=as_of,
    )
    _, _, n_guarded, _, _ = apply_idle_assist_staging(
        all_pallets,
        slots,
        state,
        orders,
        ship,
        pulled_today_by_item={},
        max_per_hour=2,
        prefer_rack=True,
        min_free_rack_slots=2,
    )
    assert n_guarded == 0  # only 2 free rack slots — reserved for promotes
    _, _, n_unguarded, _, _ = apply_idle_assist_staging(
        all_pallets,
        slots,
        state,
        orders,
        ship,
        pulled_today_by_item={},
        max_per_hour=2,
        prefer_rack=True,
        min_free_rack_slots=0,
    )
    assert n_unguarded == 1
