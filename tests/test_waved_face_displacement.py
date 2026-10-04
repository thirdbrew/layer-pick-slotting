"""Ranked waved-face displacement in plan_rack_to_face.

Default behavior (flag off) hard-skips every waved face, which on a heavy day
can become the main WAIT_SUPPLY_SHORTAGE driver. With waved_displacement=True the
planner may evict the waved face with the LOWEST near-term waved layers
(WAVED_HORIZON window), but only when strictly below the incoming item's.
"""

from __future__ import annotations

from datetime import datetime

from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_face_geography
from slotting.generate_slots import source_slot_ids
from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    Slot,
    SlotType,
    WorldState,
)
from slotting.resupply_rack import build_rack_slots_aligned, plan_rack_to_face

AS_OF = datetime(2026, 3, 3, 10, 0)


def _order(item_id: str, layers: int) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=datetime(2026, 3, 3, 12, 0),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def _world(*, incoming_layers: int, face_layers: dict[str, int]):
    """All faces occupied by waved items; incoming HOT SKU sits on rack."""
    source_count = len(face_layers)
    _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=2)
    face_station_by_id, face_flank_by_id = source_face_geography(aisle_map)
    face_ids = source_slot_ids(source_count)
    sources = [
        Slot(slot_id=fid, slot_type=SlotType.SOURCE, occupied_by_pallet_id=f"P-{fid}")
        for fid in face_ids
    ]
    racks = build_rack_slots_aligned(stations_per_flank=4, depth_tiers=2)
    rack_slot = next(s for s in racks if s.slot_id == "R-L02")
    rack_slot.occupied_by_pallet_id = "RACK-HOT"
    face_items = dict(zip(face_ids, face_layers))
    pallets = [
        Pallet(
            pallet_id="RACK-HOT",
            item_id="HOT",
            layers_remaining=8,
            current_slot=rack_slot.slot_id,
            arrived_at=AS_OF,
        ),
        *[
            Pallet(
                pallet_id=f"P-{fid}",
                item_id=item,
                layers_remaining=8,
                current_slot=fid,
                arrived_at=AS_OF,
            )
            for fid, item in face_items.items()
        ],
    ]
    orders = [_order("HOT", incoming_layers)] + [
        _order(item, layers) for item, layers in face_layers.items()
    ]
    state = WorldState(
        pallets=pallets,
        slots=sources + racks,
        orders=orders,
        exclusions=[],
        as_of=AS_OF,
    )
    waved = {"HOT", *face_layers}
    kwargs = dict(
        item_id="HOT",
        waved_items=waved,
        face_station_by_id=face_station_by_id,
        face_flank_by_id=face_flank_by_id,
    )
    return state, kwargs


def test_flag_off_keeps_hard_skip() -> None:
    state, kwargs = _world(incoming_layers=20, face_layers={"LOW": 2, "MID": 9})
    assert plan_rack_to_face(state, **kwargs) is None


def test_displaces_lowest_value_waved_face() -> None:
    state, kwargs = _world(
        incoming_layers=20, face_layers={"MID": 9, "LOW": 2, "HIGH": 30}
    )
    plan = plan_rack_to_face(state, waved_displacement=True, **kwargs)
    assert plan is not None
    displaced = state.pallet_by_id(plan.face_pallet_id)
    assert displaced is not None and displaced.item_id == "LOW"
    # displaced pallet must have somewhere to go
    assert plan.relocate_to_slot_id is not None or plan.to_storage


def test_no_displacement_when_incoming_not_hotter() -> None:
    state, kwargs = _world(incoming_layers=5, face_layers={"A": 6, "B": 9})
    assert plan_rack_to_face(state, waved_displacement=True, **kwargs) is None


def _fill_rack(state: WorldState) -> WorldState:
    """Occupy every rack slot (stale filler SKUs) except those already used."""
    pallets = list(state.pallets)
    slots = [s.model_copy(deep=True) for s in state.slots]
    for i, slot in enumerate(s for s in slots if s.slot_type == SlotType.RESUPPLY_RACK):
        if slot.occupied_by_pallet_id:
            continue
        pid = f"FILL-{i:03d}"
        pallets.append(
            Pallet(
                pallet_id=pid,
                item_id=f"STALE{i:03d}",
                layers_remaining=8,
                current_slot=slot.slot_id,
                arrived_at=AS_OF,
            )
        )
        slot.occupied_by_pallet_id = pid
    return WorldState(
        pallets=pallets,
        slots=slots,
        orders=state.orders,
        exclusions=state.exclusions,
        as_of=AS_OF,
    )


def test_full_rack_blocks_displacement_without_swap() -> None:
    state, kwargs = _world(incoming_layers=20, face_layers={"COLD": 2, "IDLE": 1})
    kwargs["waved_items"] = {"HOT"}  # both faces non-waved squatters
    state = _fill_rack(state)
    assert plan_rack_to_face(state, waved_displacement=True, **kwargs) is None


def test_full_rack_swap_lands_displaced_in_vacated_rack_slot() -> None:
    state, kwargs = _world(incoming_layers=20, face_layers={"COLD": 2, "IDLE": 1})
    kwargs["waved_items"] = {"HOT"}
    state = _fill_rack(state)
    plan = plan_rack_to_face(
        state,
        waved_displacement=True,
        waved_displacement_rack_swap=True,
        **kwargs,
    )
    assert plan is not None
    assert plan.relocate_to_slot_id == plan.rack_slot_id == "R-L02"
    assert plan.rack_pallet_id == "RACK-HOT"


def test_full_rack_swap_applies_without_ghost_ref() -> None:
    from slotting.models import Recommendation, RecommendationAction, RuleId
    from slotting.rules import RuleConfig
    from slotting.simulation import _execute_rack_promotes

    state, kwargs = _world(incoming_layers=20, face_layers={"COLD": 2, "IDLE": 1})
    kwargs["waved_items"] = {"HOT"}
    state = _fill_rack(state)
    rec = Recommendation(
        action=RecommendationAction.FLAG_RACK_PROMOTE,
        pallet_id="RACK-HOT",
        item_id="HOT",
        slot_id="S01",
        reason="test",
        rule=RuleId.RACK_PROMOTE,
    )
    pallets = [p.model_copy(deep=True) for p in state.pallets]
    slots = [s.model_copy(deep=True) for s in state.slots]
    cfg = RuleConfig(
        waved_displacement=True,
        waved_displacement_rack_swap=True,
        source_face_station_by_id=kwargs["face_station_by_id"],
        source_face_flank_by_id=kwargs["face_flank_by_id"],
    )
    applied = _execute_rack_promotes(state, pallets, slots, [rec], rule_config=cfg)
    assert applied
    slot_by_id = {s.slot_id: s for s in slots}
    pallet_by_id = {p.pallet_id: p for p in pallets}
    hot = pallet_by_id["RACK-HOT"]
    face_slot = slot_by_id[hot.current_slot]
    assert face_slot.slot_type == SlotType.SOURCE
    assert face_slot.occupied_by_pallet_id == "RACK-HOT"
    displaced_slot = slot_by_id["R-L02"]
    displaced = pallet_by_id[displaced_slot.occupied_by_pallet_id]
    assert displaced.current_slot == "R-L02"
    assert displaced.pallet_id != "RACK-HOT"
    # no ghost: every occupied ref resolves to a pallet standing there
    for s in slots:
        if s.occupied_by_pallet_id:
            p = pallet_by_id.get(s.occupied_by_pallet_id)
            assert p is not None and p.current_slot == s.slot_id, s.slot_id


def test_non_waved_face_still_preferred_over_displacement() -> None:
    state, kwargs = _world(incoming_layers=20, face_layers={"COLD": 2, "IDLE": 1})
    # IDLE loses its waved status -> it is a plain non-waved face and must be
    # evicted by the ordinary path before any waved-face displacement happens
    kwargs["waved_items"] = {"HOT", "COLD"}
    plan = plan_rack_to_face(state, waved_displacement=True, **kwargs)
    assert plan is not None
    displaced = state.pallet_by_id(plan.face_pallet_id)
    assert displaced is not None and displaced.item_id == "IDLE"
