"""Low-layer resupply threshold rules."""

from __future__ import annotations

from datetime import datetime, timedelta

from slotting.inventory import (
    ANY_DEPTH_RESUPPLY_THRESHOLD,
    face_stock_needs_resupply,
    item_needs_face_resupply,
)
from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    RecommendationAction,
    RuleId,
    Slot,
    SlotType,
    WorldState,
)
from slotting.resupply_rack import build_rack_slots_aligned, plan_rack_to_face
from slotting.rules import RESUPPLY_RACK_RULE_CONFIG, RuleConfig, rule_add_waved_demand
from slotting.simulation import simulate_once

from conftest import AS_OF, minimal_state, slotted_pallet


def _waved_order(item_id: str, layers: int = 10, *, as_of: datetime | None = None) -> OrderLine:
    due = (as_of or AS_OF) + timedelta(hours=2)
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=due,
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def test_face_stock_needs_resupply_zero_means_any_depth() -> None:
    assert face_stock_needs_resupply(8, ANY_DEPTH_RESUPPLY_THRESHOLD) is True
    assert face_stock_needs_resupply(5, ANY_DEPTH_RESUPPLY_THRESHOLD) is True
    assert face_stock_needs_resupply(9, ANY_DEPTH_RESUPPLY_THRESHOLD) is False
    assert face_stock_needs_resupply(2, 3) is True
    assert face_stock_needs_resupply(4, 3) is False


def test_effective_resupply_threshold_late_shift_sentinel() -> None:
    cfg = RuleConfig(resupply_layer_threshold=3, late_shift_hour=18, late_shift_resupply_threshold=0)
    day = minimal_state(as_of=datetime(2026, 3, 3, 14, 0))
    late = minimal_state(as_of=datetime(2026, 3, 3, 19, 0))
    assert cfg.effective_resupply_threshold(day) == 3
    assert cfg.effective_resupply_threshold(late) == 0


def test_item_needs_face_resupply_any_depth_with_stock_remaining() -> None:
    face, slot = slotted_pallet("P1", "SKU-A", layers=5)
    state = minimal_state(pallets=[face], slots=[slot], orders=[_waved_order("SKU-A")])
    assert item_needs_face_resupply(state, "SKU-A", threshold=3) is False
    assert item_needs_face_resupply(state, "SKU-A", threshold=0) is True


def test_plan_rack_to_face_any_depth_refreshes_stocked_face() -> None:
    """threshold=0 must refresh a same-SKU face that still has layers (not empty-only)."""
    face_slot = Slot(slot_id="S01", slot_type=SlotType.SOURCE, occupied_by_pallet_id="FACE")
    racks = build_rack_slots_aligned(stations_per_flank=2, depth_tiers=1)
    rack = next(s for s in racks if s.slot_id == "R-L01")
    rack.occupied_by_pallet_id = "RACK"
    as_of = datetime(2026, 3, 3, 19, 0)
    face = Pallet(
        pallet_id="FACE",
        item_id="SKU-A",
        layers_remaining=5,
        current_slot="S01",
        arrived_at=as_of,
    )
    rack_pallet = Pallet(
        pallet_id="RACK",
        item_id="SKU-A",
        layers_remaining=10,
        current_slot=rack.slot_id,
        arrived_at=as_of,
    )
    state = WorldState(
        pallets=[face, rack_pallet],
        slots=[face_slot, *racks],
        orders=[_waved_order("SKU-A", as_of=as_of)],
        exclusions=[],
        as_of=as_of,
    )
    blocked = plan_rack_to_face(
        state, item_id="SKU-A", waved_items={"SKU-A"}, resupply_layer_threshold=3
    )
    assert blocked is None  # 5-layer face above day threshold; no empty face

    plan = plan_rack_to_face(
        state, item_id="SKU-A", waved_items={"SKU-A"}, resupply_layer_threshold=0
    )
    assert plan is not None
    assert plan.face_slot_id == "S01"
    assert plan.face_pallet_id == "FACE"
    assert plan.rack_pallet_id == "RACK"


def test_execute_rack_promote_uses_late_shift_effective_threshold() -> None:
    """Apply path must honor late-shift sentinel 0, not the static day threshold."""
    as_of = datetime(2026, 3, 3, 19, 0)
    face_slot = Slot(slot_id="S01", slot_type=SlotType.SOURCE, occupied_by_pallet_id="FACE")
    racks = build_rack_slots_aligned(stations_per_flank=2, depth_tiers=1)
    rack = next(s for s in racks if s.slot_id == "R-L01")
    rack.occupied_by_pallet_id = "RACK"
    face = Pallet(
        pallet_id="FACE",
        item_id="SKU-A",
        layers_remaining=5,
        current_slot="S01",
        arrived_at=as_of,
    )
    rack_pallet = Pallet(
        pallet_id="RACK",
        item_id="SKU-A",
        layers_remaining=10,
        current_slot=rack.slot_id,
        arrived_at=as_of,
    )
    state = WorldState(
        pallets=[face, rack_pallet],
        slots=[face_slot, *racks],
        orders=[_waved_order("SKU-A", as_of=as_of)],
        exclusions=[],
        as_of=as_of,
    )
    cfg = RuleConfig(
        disabled=frozenset({RuleId.LOW_LAYERS, RuleId.BACKUP_PROMOTE, RuleId.DEST_CLOSE_IDLE}),
        include_rack_aisle=True,
        rack_only_reserve=True,
        resupply_layer_threshold=3,
        late_shift_hour=18,
        late_shift_resupply_threshold=0,
        low_layer_refresh_enabled=False,
        visible_prestage_enabled=False,
        proactive_zero_demand_remove=False,
    )
    result = simulate_once(state, rule_config=cfg)
    promotes = [a for a in result.applied if a.action == RecommendationAction.FLAG_RACK_PROMOTE]
    assert promotes, "late-shift any-depth should apply rack promote onto a 5-layer face"
    assert result.projected_state.pallet_by_id("RACK").current_slot == "S01"


def test_add_skips_when_face_has_adequate_layers() -> None:
    pallet, slot = slotted_pallet("P1", "SKU-A", layers=8)
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[_waved_order("SKU-A")])
    recs = rule_add_waved_demand(state, set(), RESUPPLY_RACK_RULE_CONFIG)
    assert [r for r in recs if r.rule == RuleId.ADD_WAVED_DEMAND] == []


def test_add_fires_when_face_empty() -> None:
    reserve = slotted_pallet("P2", "SKU-A", layers=10)[0]
    reserve.current_slot = None
    slot = slotted_pallet("P1", "SKU-B", layers=8)[1]
    state = minimal_state(pallets=[reserve], slots=[slot], orders=[_waved_order("SKU-A")])
    recs = rule_add_waved_demand(state, set(), RESUPPLY_RACK_RULE_CONFIG)
    assert any(r.rule == RuleId.ADD_WAVED_DEMAND and r.item_id == "SKU-A" for r in recs)


def test_add_skips_when_partial_face_remains() -> None:
    """Do not open a second face while the current pallet still has pick layers."""
    face, slot = slotted_pallet("P1", "SKU-A", layers=2)
    reserve = slotted_pallet("P2", "SKU-A", layers=10)[0]
    reserve.current_slot = None
    empty = slotted_pallet("P3", "SKU-B", layers=8)[1]
    state = minimal_state(
        pallets=[face, reserve],
        slots=[slot, empty],
        orders=[_waved_order("SKU-A")],
    )
    recs = rule_add_waved_demand(state, set(), RESUPPLY_RACK_RULE_CONFIG)
    assert [r for r in recs if r.rule == RuleId.ADD_WAVED_DEMAND and r.item_id == "SKU-A"] == []


def test_face_refresh_prioritizes_pending_completion() -> None:
    face, slot = slotted_pallet("P1", "SKU-A", layers=2)
    pending = Pallet(
        pallet_id="P2",
        item_id="SKU-A",
        layers_remaining=4,
        current_slot=None,
        arrived_at=AS_OF,
        removed_from_pick_at=AS_OF - timedelta(hours=1),
    )
    full = slotted_pallet("P3", "SKU-A", layers=20)[0]
    full.current_slot = None
    state = minimal_state(pallets=[face, pending, full], slots=[slot], orders=[_waved_order("SKU-A")])
    result = simulate_once(state, rule_config=RESUPPLY_RACK_RULE_CONFIG)
    refreshed = [a for a in result.applied if a.action == RecommendationAction.FLAG_FACE_REFRESH]
    assert refreshed
    assert result.projected_state.pallet_by_id("P2").current_slot == slot.slot_id
    assert result.projected_state.pallet_by_id("P1").current_slot is None
    assert result.projected_state.pallet_by_id("P1").removed_from_pick_at is not None


def test_waved_resupply_sort_prioritizes_lowest_face_layers() -> None:
    from slotting.inventory import face_layers_for_item
    from slotting.rules import _items_with_waved_demand

    low_face, slot_a = slotted_pallet("LOW", "SKU-A", layers=2)
    high_face, slot_b = slotted_pallet("HIGH", "SKU-B", layers=10)
    orders = [_waved_order("SKU-A"), _waved_order("SKU-B")]
    state = minimal_state(pallets=[low_face, high_face], slots=[slot_a, slot_b], orders=orders)
    waved = _items_with_waved_demand(state)
    ordered = sorted(waved, key=lambda item_id: (face_layers_for_item(state, item_id), item_id))
    assert ordered[0] == "SKU-A"
