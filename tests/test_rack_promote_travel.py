"""Rack promote and colocation travel tests."""

from __future__ import annotations

from datetime import datetime

from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_face_geography
from slotting.generate_slots import source_slot_ids
from slotting.models import Pallet, Slot, SlotType, WorldState
from slotting.resupply_rack import build_rack_slots_aligned, plan_rack_to_face
from slotting.rack_depth import RankedSku, assign_rack_slots_colocated, rack_station_index


def _world(
  *,
  source_count: int = 4,
  donor_count: int = 2,
  rack_depth: int = 2,
) -> tuple[WorldState, dict[str, int], dict[str, str], dict[str, str]]:
    _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=donor_count)
    face_station_by_id, face_flank_by_id = source_face_geography(aisle_map)
    sources = [
        Slot(slot_id=face_id, slot_type=SlotType.SOURCE, occupied_by_pallet_id=f"P-{face_id}")
        for face_id in source_slot_ids(source_count)
    ]
    for slot in sources:
        if slot.slot_id == "S01":
            slot.occupied_by_pallet_id = None
    racks = build_rack_slots_aligned(stations_per_flank=4, depth_tiers=rack_depth)
    rack_slot = next(s for s in racks if s.slot_id == "R-L02")
    rack_slot.occupied_by_pallet_id = "RACK-P1"
    pallets = [
        Pallet(
            pallet_id="RACK-P1",
            item_id="WAVED",
            layers_remaining=10,
            current_slot=rack_slot.slot_id,
            arrived_at=datetime(2026, 3, 3, 6, 0),
        ),
        *[
            Pallet(
                pallet_id=f"P-{slot.slot_id}",
                item_id=f"ITEM-{slot.slot_id}",
                layers_remaining=10,
                current_slot=slot.slot_id,
                arrived_at=datetime(2026, 3, 3, 6, 0),
            )
            for slot in sources
            if slot.occupied_by_pallet_id
        ],
    ]
    state = WorldState(
        pallets=pallets,
        slots=sources + racks,
        orders=[],
        exclusions=[],
        as_of=datetime(2026, 3, 3, 10, 0),
    )
    preferred = {"WAVED": "S01"}
    return state, face_station_by_id, face_flank_by_id, preferred


def test_plan_rack_to_face_prefers_preferred_empty_face() -> None:
    state, face_station_by_id, face_flank_by_id, preferred = _world()
    plan = plan_rack_to_face(
        state,
        item_id="WAVED",
        waved_items={"WAVED"},
        preferred_face_by_item=preferred,
        face_station_by_id=face_station_by_id,
        face_flank_by_id=face_flank_by_id,
    )
    assert plan is not None
    assert plan.face_slot_id == "S01"


def test_assign_rack_fallback_prefers_nearest_station() -> None:
    from slotting.rack_depth import rack_slot_at_station

    source_count = 2
    donor_count = 1
    _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=donor_count)
    face_order = source_slot_ids(source_count)
    slots = build_rack_slots_aligned(stations_per_flank=3, depth_tiers=1)
    slot_ids = [s.slot_id for s in slots]
    near_mapped = aisle_map["S02"]
    near_flank = "L" if near_mapped.side.value == "left" else "R"
    occupied_front = rack_slot_at_station(near_mapped.station, near_flank, tier=0)
    overflow = [
        RankedSku(rank=3, item_id="NEAR", layers=1, earliest_due=datetime(2026, 3, 3, 8)),
    ]
    preferred = {"NEAR": "S02"}
    result = assign_rack_slots_colocated(
        overflow,
        [sid for sid in slot_ids if sid != occupied_front],
        face_order=face_order,
        aisle_map=aisle_map,
        virtual_face_by_item=preferred,
    )
    assert abs(rack_station_index(result["NEAR"]) - near_mapped.station) <= 1
