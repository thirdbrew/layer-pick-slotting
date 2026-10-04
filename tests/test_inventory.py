"""Pallet inventory seeding and hourly depletion."""

from __future__ import annotations

from datetime import date, datetime, time

from slotting.generate_slots import build_slots
from slotting.inventory import (
    apply_hourly_layer_picks,
    build_reserve_pallets,
    reserve_pallet_layers,
)
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, SlotType, WorldState


def _order(item_id: str, layers: int, due: datetime) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=due,
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def test_reserve_pallet_layers_from_orders() -> None:
    ship = date(2026, 3, 3)
    due = datetime.combine(ship, time(10, 0))
    orders = [_order("A", 29, due)]
    # Default 9in/layer → floor(100/9)=11, not legacy 24-layer cap
    assert reserve_pallet_layers(orders, "A", ship, default_layers=8) == 11


def test_reserve_pallet_layers_respects_height_cap(monkeypatch) -> None:
    ship = date(2026, 3, 3)
    due = datetime.combine(ship, time(10, 0))
    orders = [_order("A", 20, due)]
    monkeypatch.setattr(
        "slotting.inventory.max_physical_pallet_layers_for_item",
        lambda item_id: 5,
    )
    assert reserve_pallet_layers(orders, "A", ship, default_layers=8) == 5


def test_donor_build_layers_respects_height_cap(monkeypatch) -> None:
    from slotting.inventory import _donor_build_layers

    monkeypatch.setattr(
        "slotting.inventory.max_physical_pallet_layers_for_item",
        lambda item_id: 5,
    )
    slots = build_slots(source_count=2, donor_count=1)
    order = _order("A", 12, datetime(2026, 3, 3, 10, 0))
    state = WorldState(
        pallets=[],
        slots=slots,
        orders=[order],
        exclusions=[],
        as_of=datetime(2026, 3, 3, 9, 0),
    )
    assert _donor_build_layers(state, "A") == 5


def test_build_reserve_pallets_reuses_carried_stock() -> None:
    ship = date(2026, 3, 3)
    carried = [
        Pallet(
            pallet_id="SWP-0001",
            item_id="A",
            layers_remaining=5,
            current_slot=None,
            arrived_at=datetime(2026, 3, 2, 22, 0),
        )
    ]
    pallets = build_reserve_pallets(
        ["A"],
        [],
        ship,
        as_of=datetime(2026, 3, 2, 22, 0),
        existing=carried,
    )
    assert len(pallets) == 1
    assert pallets[0].pallet_id == "SWP-0001"
    assert pallets[0].layers_remaining == 5


def test_build_reserve_pallets_keeps_slotted_carried() -> None:
    """Slotted carry must persist — dropping it leaves ghost slot refs."""
    ship = date(2026, 3, 3)
    on_face = Pallet(
        pallet_id="SWP-0007",
        item_id="A",
        layers_remaining=6,
        current_slot="S01",
        arrived_at=datetime(2026, 3, 2, 22, 0),
    )
    pallets = build_reserve_pallets(
        ["B"],
        [],
        ship,
        as_of=datetime(2026, 3, 2, 22, 0),
        existing=[on_face],
    )
    kept = [p for p in pallets if p.item_id == "A"]
    assert len(kept) == 1
    assert kept[0].pallet_id == "SWP-0007"
    assert kept[0].current_slot == "S01"


def test_build_reserve_pallets_never_reuses_carried_ids() -> None:
    """A minted ID colliding with a carried pallet makes stale slot refs resolve wrong."""
    ship = date(2026, 3, 3)
    carried = [
        Pallet(
            pallet_id="SWP-0001",
            item_id="A",
            layers_remaining=6,
            current_slot="S01",
            arrived_at=datetime(2026, 3, 2, 22, 0),
        ),
        Pallet(
            pallet_id="SWP-0002",
            item_id="B",
            layers_remaining=0,  # depleted: dropped, but its ID must stay reserved
            current_slot=None,
            arrived_at=datetime(2026, 3, 2, 22, 0),
        ),
    ]
    pallets = build_reserve_pallets(
        ["C", "D"],
        [],
        ship,
        as_of=datetime(2026, 3, 2, 22, 0),
        existing=carried,
    )
    minted = [p.pallet_id for p in pallets if p.item_id in ("C", "D")]
    assert minted == ["SWP-0003", "SWP-0004"]
    ids = [p.pallet_id for p in pallets]
    assert len(ids) == len(set(ids))


def test_apply_hourly_layer_picks_depletes_face_pallet() -> None:
    slots = build_slots(source_count=2, donor_count=1)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face.occupied_by_pallet_id = "P1"
    pallet = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=10,
        current_slot=face.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    as_of = datetime(2026, 3, 3, 9, 0)
    orders = [_order("A", 6, datetime(2026, 3, 3, 12, 0))]
    state = WorldState(
        pallets=[pallet],
        slots=slots,
        orders=orders,
        exclusions=[],
        as_of=as_of,
    )
    picked = apply_hourly_layer_picks([pallet], slots, state, layers_per_hour=200.0)
    # 10 layers @ 9in default — only 5 within 48in clamp reach
    assert picked == 5
    assert pallet.layers_remaining == 5


def test_apply_hourly_layer_picks_reports_layers_per_item() -> None:
    """Callers need per-SKU picks to credit them back to the order lines.

    Without per-item attribution the order ledger cannot be decremented, and
    demand is re-requested every tick.
    """
    slots = build_slots(source_count=3, donor_count=1)
    sources = [s for s in slots if s.slot_type == SlotType.SOURCE]
    pallets = []
    for idx, (item_id, layers) in enumerate((("A", 4), ("B", 3))):
        sources[idx].occupied_by_pallet_id = f"P{idx}"
        pallets.append(
            Pallet(
                pallet_id=f"P{idx}",
                item_id=item_id,
                layers_remaining=layers,
                current_slot=sources[idx].slot_id,
                arrived_at=datetime(2026, 3, 3, 8, 0),
            )
        )
    as_of = datetime(2026, 3, 3, 9, 0)
    due = datetime(2026, 3, 3, 12, 0)
    orders = [_order("A", 4, due), _order("B", 3, due)]
    state = WorldState(
        pallets=pallets, slots=slots, orders=orders, exclusions=[], as_of=as_of
    )
    by_item: dict[str, int] = {}
    picked = apply_hourly_layer_picks(
        pallets, slots, state, layers_per_hour=200.0, picked_by_item=by_item
    )
    assert sum(by_item.values()) == picked
    assert by_item == {"A": 4, "B": 3}


def test_apply_hourly_layer_picks_depletes_thickest_face_first() -> None:
    slots = build_slots(source_count=3, donor_count=1)
    sources = [s for s in slots if s.slot_type == SlotType.SOURCE]
    thin = Pallet(
        pallet_id="THIN",
        item_id="A",
        layers_remaining=3,
        current_slot=sources[0].slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    thick = Pallet(
        pallet_id="THICK",
        item_id="A",
        layers_remaining=8,
        current_slot=sources[1].slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    sources[0].occupied_by_pallet_id = thin.pallet_id
    sources[1].occupied_by_pallet_id = thick.pallet_id
    as_of = datetime(2026, 3, 3, 9, 0)
    orders = [_order("A", 4, datetime(2026, 3, 3, 12, 0))]
    state = WorldState(
        pallets=[thin, thick],
        slots=slots,
        orders=orders,
        exclusions=[],
        as_of=as_of,
    )
    picked = apply_hourly_layer_picks([thin, thick], slots, state, layers_per_hour=4.0)
    assert picked == 4
    assert thick.layers_remaining == 4
    assert thin.layers_remaining == 3


def test_backtest_chained_final_state_includes_inventory() -> None:
    from datetime import timedelta

    from slotting.backtest import BacktestMode, backtest_historical
    from slotting.rules import DEFAULT_RULE_CONFIG

    slots = build_slots(source_count=2, donor_count=1)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face.occupied_by_pallet_id = "P1"
    pallet = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=10,
        current_slot=face.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    start = datetime(2026, 3, 3, 9, 0)
    orders = [_order("A", 6, datetime(2026, 3, 3, 12, 0))]
    initial = WorldState(
        pallets=[pallet],
        slots=slots,
        orders=orders,
        exclusions=[],
        as_of=start,
    )
    report = backtest_historical(
        initial,
        start=start,
        end=start,
        step=timedelta(hours=1),
        mode=BacktestMode.CHAINED,
        rule_config=DEFAULT_RULE_CONFIG,
        inventory_layers_per_hour=200.0,
    )
    assert report.final_state is not None
    assert report.final_state.pallet_by_id("P1").layers_remaining == 5


def test_finalize_depleted_pallets_removes_empty() -> None:
    from slotting.inventory import finalize_depleted_pallets

    slots = build_slots(source_count=2, donor_count=1)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face.occupied_by_pallet_id = "P1"
    empty = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=0,
        current_slot=face.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    live = Pallet(
        pallet_id="P2",
        item_id="B",
        layers_remaining=5,
        current_slot=None,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    kept = finalize_depleted_pallets([empty, live], slots, repurpose_as_donor=False)
    assert face.occupied_by_pallet_id is None
    assert len(kept) == 1
    assert kept[0].pallet_id == "P2"


def test_depleted_pallet_repurposed_as_donor_build() -> None:
    from slotting.inventory import dispose_depleted_pallets

    slots = build_slots(source_count=2, donor_count=2)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    donor = next(s for s in slots if s.slot_type == SlotType.DESTINATION)
    face.occupied_by_pallet_id = "P1"
    empty = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=0,
        current_slot=face.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    as_of = datetime(2026, 3, 3, 9, 0)
    orders = [
        _order("A", 6, datetime(2026, 3, 3, 11, 0)),
    ]
    state = WorldState(pallets=[empty], slots=slots, orders=orders, exclusions=[], as_of=as_of)
    kept = dispose_depleted_pallets([empty], slots, state=state)
    assert len(kept) == 1
    assert kept[0].pallet_id == f"BUILD-{donor.slot_id}"
    assert kept[0].layers_remaining > 0
    assert donor.occupied_by_pallet_id == kept[0].pallet_id


def test_depleted_pallet_exits_when_no_donor_need() -> None:
    from slotting.inventory import dispose_depleted_pallets

    slots = build_slots(source_count=2, donor_count=2)
    donor = next(s for s in slots if s.slot_type == SlotType.DESTINATION)
    empty = Pallet(
        pallet_id="P1",
        item_id="DEAD",
        layers_remaining=0,
        current_slot=None,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    as_of = datetime(2026, 3, 3, 9, 0)
    state = WorldState(pallets=[empty], slots=slots, orders=[], exclusions=[], as_of=as_of)
    kept = dispose_depleted_pallets([empty], slots, state=state)
    assert kept == []
    assert donor.occupied_by_pallet_id is None


def test_item_needs_face_resupply_when_low() -> None:
    from slotting.inventory import item_needs_face_resupply

    slots = build_slots(source_count=2, donor_count=1)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face.occupied_by_pallet_id = "P1"
    pallet = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=2,
        current_slot=face.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    state = WorldState(pallets=[pallet], slots=slots, orders=[], exclusions=[], as_of=datetime(2026, 3, 3, 9, 0))
    assert item_needs_face_resupply(state, "A", threshold=3)
    assert not item_needs_face_resupply(state, "A", threshold=1)


def test_item_needs_face_resupply_uses_pickable_not_total_layers() -> None:
    """8 layers on face but only 5 pickable at 48in — resupply at pickable threshold."""
    from slotting.inventory import item_needs_face_resupply
    from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits

    slots = build_slots(source_count=2, donor_count=1)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face.occupied_by_pallet_id = "P1"
    pallet = Pallet(
        pallet_id="P1",
        item_id="TALL",
        layers_remaining=8,
        current_slot=face.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    state = WorldState(pallets=[pallet], slots=slots, orders=[], exclusions=[], as_of=datetime(2026, 3, 3, 9, 0))
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=48.0)):
        # Total face = 8 would not trigger at threshold 6; pickable = 5 should.
        assert not item_needs_face_resupply(state, "TALL", threshold=3)
        assert item_needs_face_resupply(state, "TALL", threshold=5)
        assert item_needs_face_resupply(state, "TALL", threshold=6)
