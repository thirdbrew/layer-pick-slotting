"""Demand-required slot reclaim and travel-aware slotting."""

from __future__ import annotations

from datetime import datetime, timedelta

from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_face_geography
from layout_advisor.planning.wave_slotting import assign_faces_wave_clustered, primary_wave_release_by_item
from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    Slot,
    SlotType,
    WorldState,
)
from slotting.rules import (
    RESUPPLY_RACK_RULE_CONFIG,
    RuleId,
    rule_aggressive_zero_demand_reclaim,
    rule_zero_demand,
    run_rules,
)
from slotting.slotting_policy import (
    best_empty_source_slot,
    rest_of_ship_day_layers,
    select_slot_eviction_victim,
)

AS_OF = datetime(2026, 3, 3, 10, 0)


def _waved_order(item_id: str, *, layers: int = 8, hours: float = 2.0) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=AS_OF + timedelta(hours=hours),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def _visible_later_order(item_id: str, *, layers: int = 20, hour: int = 20) -> OrderLine:
    return OrderLine(
        order_id=f"O-late-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=AS_OF.replace(hour=hour),
        status=OrderStatus.VISIBLE,
        area=OrderArea.THIS_LINE,
    )


def _slotted(pallet_id: str, item_id: str, slot_id: str, *, layers: int = 8) -> tuple[Pallet, Slot]:
    pallet = Pallet(
        pallet_id=pallet_id,
        item_id=item_id,
        layers_remaining=layers,
        current_slot=slot_id,
        arrived_at=AS_OF,
    )
    slot = Slot(slot_id=slot_id, slot_type=SlotType.SOURCE, occupied_by_pallet_id=pallet_id)
    return pallet, slot


def test_reference_demand_required_slot_reclaim_policy() -> None:
    """Reference config: demand-required reclaim only; no proactive short-horizon REMOVEs."""
    from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG

    assert not REFERENCE_REPLAY_CONFIG.proactive_zero_demand_remove
    assert not REFERENCE_REPLAY_CONFIG.proactive_zero_demand_realignment_only
    assert REFERENCE_REPLAY_CONFIG.slot_reclaim_rest_of_day
    assert REFERENCE_REPLAY_CONFIG.aggressive_zero_demand_reclaim
    assert REFERENCE_REPLAY_CONFIG.zero_demand_reclaim_horizon_hours == 4


def test_proactive_zero_demand_skipped_outside_realignment_hours() -> None:
    from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG

    cfg = REFERENCE_REPLAY_CONFIG.rule_config.with_policy_flags(
        proactive_zero_demand_remove=True,
        proactive_zero_demand_realignment_only=True,
        proactive_zero_demand_realignment_hours=(11, 14),
    )
    state = WorldState(
        pallets=[_slotted("P1", "STALE", "S01")[0]],
        slots=[_slotted("P1", "STALE", "S01")[1]],
        orders=[],
        exclusions=[],
        as_of=AS_OF.replace(hour=10),
    )
    assert run_rules(state, cfg) == []
    at_realign = state.model_copy(update={"as_of": AS_OF.replace(hour=11)})
    assert run_rules(at_realign, cfg) != []


def test_proactive_zero_demand_remove_when_enabled() -> None:
    from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG

    cfg = REFERENCE_REPLAY_CONFIG.rule_config.with_policy_flags(
        proactive_zero_demand_remove=True,
    )
    state = WorldState(
        pallets=[_slotted("P1", "STALE", "S01")[0]],
        slots=[_slotted("P1", "STALE", "S01")[1]],
        orders=[],
        exclusions=[],
        as_of=AS_OF,
    )
    assert cfg.proactive_zero_demand_remove
    assert rule_zero_demand(state, cfg) != []
    assert run_rules(state, cfg) != []


def test_reclaim_skips_occupant_with_rest_of_day_demand() -> None:
    blocker, face = _slotted("P1", "BUSY", "S01")
    config = RESUPPLY_RACK_RULE_CONFIG.with_slotting_maps(
        preferred_faces={"NEED": "S01"},
        setup_ranks={"NEED": 1, "BUSY": 50},
    )
    state = WorldState(
        pallets=[blocker],
        slots=[face],
        orders=[_waved_order("NEED"), _visible_later_order("BUSY")],
        exclusions=[],
        as_of=AS_OF,
    )
    assert rest_of_ship_day_layers(state, "BUSY") > 0
    recs = rule_aggressive_zero_demand_reclaim(state, config)
    assert recs == []


def test_reclaim_evicts_lowest_value_blocker() -> None:
    low_rank, face_low = _slotted("P-LOW", "STALE-A", "S01", layers=4)
    high_rank, face_high = _slotted("P-HIGH", "STALE-B", "S02", layers=10)
    config = RESUPPLY_RACK_RULE_CONFIG.with_slotting_maps(
        preferred_faces={"NEED": "S01"},
        setup_ranks={"NEED": 1, "STALE-A": 80, "STALE-B": 90},
    )
    state = WorldState(
        pallets=[high_rank, low_rank],
        slots=[face_high, face_low],
        orders=[_waved_order("NEED")],
        exclusions=[],
        as_of=AS_OF,
    )
    recs = rule_aggressive_zero_demand_reclaim(state, config)
    assert len(recs) == 1
    assert recs[0].pallet_id == "P-LOW"
    assert recs[0].rule == RuleId.ZERO_DEMAND_RECLAIM


def test_reclaim_when_no_empty_faces_picks_lowest_rank() -> None:
    p1, s1 = _slotted("P1", "STALE-A", "S01", layers=8)
    p2, s2 = _slotted("P2", "STALE-B", "S02", layers=8)
    config = RESUPPLY_RACK_RULE_CONFIG.with_slotting_maps(
        preferred_faces={"NEED": "S99"},
        setup_ranks={"NEED": 1, "STALE-A": 60, "STALE-B": 90},
    )
    state = WorldState(
        pallets=[p1, p2],
        slots=[s1, s2],
        orders=[_waved_order("NEED")],
        exclusions=[],
        as_of=AS_OF,
    )
    victim = select_slot_eviction_victim(state, config)
    assert victim is not None
    assert victim[0].pallet_id == "P2"


def test_high_touch_sku_gets_donor_colocated_face() -> None:
    ship = AS_OF.date()
    wave = AS_OF.replace(hour=8)
    orders = [
        OrderLine(
            order_id="O-HIGH",
            item_id="HIGH",
            qty_layers=40,
            due_at=wave + timedelta(hours=4),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id="O-LOW",
            item_id="LOW",
            qty_layers=5,
            due_at=wave + timedelta(hours=4),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
    ]
    layout, aisle_map = build_aisle_slot_map(source_count=8, donor_count=4)
    primary = primary_wave_release_by_item(orders, ship)
    assignment = assign_faces_wave_clustered(
        ["HIGH", "LOW"],
        primary,
        aisle_map=aisle_map,
        layout=layout,
        source_count=8,
        donor_count=4,
        order_affinity=False,
    )
    donor_stations = {
        mapped.station for mapped in aisle_map.values() if mapped.zone == "donor_face"
    }
    high_station = aisle_map[assignment["HIGH"]].station
    low_station = aisle_map[assignment["LOW"]].station
    assert high_station in donor_stations
    assert high_station <= low_station or aisle_map[assignment["HIGH"]].station in donor_stations


def test_add_picks_nearest_empty_face_to_preferred() -> None:
    _, aisle_map = build_aisle_slot_map(source_count=8, donor_count=4)
    stations, _ = source_face_geography(aisle_map)
    preferred = "S03"
    config = RESUPPLY_RACK_RULE_CONFIG.with_slotting_maps(
        preferred_faces={"SKU-A": preferred},
        setup_ranks={"SKU-A": 1},
        face_station_by_id=stations,
    )
    empty = ["S01", "S05", "S07"]
    anchor_station = stations[preferred]
    expected = min(
        empty,
        key=lambda slot_id: (abs(stations[slot_id] - anchor_station), slot_id),
    )
    chosen = best_empty_source_slot(
        empty,
        item_id="SKU-A",
        preferred_slot=None,
        config=config,
    )
    assert chosen == expected
