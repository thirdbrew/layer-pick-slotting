"""Tests for value-based slotting and operator-simple rack rules."""

from __future__ import annotations

from datetime import datetime

from layout_advisor.layout.slot_geometry import source_faces_by_value
from slotting.rack_depth import (
    RankedSku,
    assign_rack_slots,
    assign_rack_slots_colocated,
    donor_staging_slot_ids,
    rack_bay_pairs,
    rack_depth_tier,
    rack_slot_at_station,
    rack_station_index,
    select_rack_slot,
)


def _sku(rank: int, item: str, due_hour: int) -> RankedSku:
    return RankedSku(
        rank=rank,
        item_id=item,
        layers=100 - rank,
        earliest_due=datetime(2026, 3, 3, due_hour, 0),
    )


from slotting.resupply_rack import build_rack_slots_aligned


def test_build_rack_slots_aligned_50_stations() -> None:
    slots = build_rack_slots_aligned(stations_per_flank=50, depth_tiers=2)
    assert len(slots) == 200
    ids = [s.slot_id for s in slots]
    assert ids[0] == "R-L01"
    assert ids[1] == "R-L02"
    assert ids[99] == "R-L100"
    assert ids[100] == "R-R01"
    assert ids[199] == "R-R100"


def test_build_rack_slots_front_row_only() -> None:
    slots = build_rack_slots_aligned(stations_per_flank=50, depth_tiers=1)
    assert len(slots) == 100
    ids = [s.slot_id for s in slots]
    assert ids[0] == "R-L01"
    assert ids[49] == "R-L50"
    assert ids[50] == "R-R01"
    assert ids[99] == "R-R50"


def test_rack_bay_pairs_two_deep_per_flank() -> None:
    slots = ["R-L01", "R-L02", "R-L03", "R-R01", "R-R02"]
    bays = rack_bay_pairs(slots)
    assert ("R-L01", "R-L02") in bays
    assert ("R-L03", None) in bays
    assert ("R-R01", "R-R02") in bays


def test_rack_slot_at_station() -> None:
    assert rack_slot_at_station(0, "L", tier=0) == "R-L01"
    assert rack_slot_at_station(5, "R", tier=0) == "R-R06"
    from layout_advisor.layout import reserve_policy as rp

    old = rp.DEFAULT_RACK_DEPTH_TIERS
    rp.DEFAULT_RACK_DEPTH_TIERS = 2
    try:
        assert rack_slot_at_station(0, "L", tier=1) == "R-L02"
    finally:
        rp.DEFAULT_RACK_DEPTH_TIERS = old


def test_assign_rack_hungarian_shares_station_fairly() -> None:
    """Three SKUs targeting the same face station get nearest distinct rack slots."""
    from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_faces_by_value

    source_count = 4
    donor_count = 2
    face_order = source_faces_by_value(source_count=source_count, donor_count=donor_count)
    _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=donor_count)
    slots = build_rack_slots_aligned(stations_per_flank=6, depth_tiers=1)
    slot_ids = [s.slot_id for s in slots]
    target_face = face_order[2]
    mapped = aisle_map[target_face]
    preferred = {f"SKU{i}": target_face for i in range(3)}
    overflow = [
        RankedSku(rank=i, item_id=f"SKU{i}", layers=10, earliest_due=datetime(2026, 3, 3, 8))
        for i in range(3)
    ]
    result = assign_rack_slots_colocated(
        overflow,
        slot_ids,
        face_order=face_order,
        aisle_map=aisle_map,
        virtual_face_by_item=preferred,
    )
    stations = {rack_station_index(result[f"SKU{i}"]) for i in range(3)}
    assert len(stations) == 3
    assert mapped.station in stations
    assert all(abs(s - mapped.station) <= 2 for s in stations)


def test_assign_rack_colocated_with_virtual_face() -> None:
    from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_faces_by_value

    source_count = 4
    donor_count = 2
    face_order = source_faces_by_value(source_count=source_count, donor_count=donor_count)
    _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=donor_count)
    slots = build_rack_slots_aligned(stations_per_flank=4)
    slot_ids = [s.slot_id for s in slots]

    overflow = [_sku(5, "OVERFLOW", 8)]
    preferred = {"OVERFLOW": face_order[2]}
    result = assign_rack_slots_colocated(
        overflow,
        slot_ids,
        face_order=face_order,
        aisle_map=aisle_map,
        virtual_face_by_item=preferred,
    )
    virtual_face = preferred["OVERFLOW"]
    mapped = aisle_map[virtual_face]
    flank = "L" if mapped.side.value == "left" else "R"
    expected = rack_slot_at_station(mapped.station, flank, tier=0)
    assert result["OVERFLOW"] == expected


def test_assign_rack_colocated_front_then_back_at_same_station() -> None:
    from layout_advisor.layout.reserve_policy import DEFAULT_RACK_DEPTH_TIERS
    from layout_advisor.layout import reserve_policy as rp

    rp.DEFAULT_RACK_DEPTH_TIERS = 2
    try:
        from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_faces_by_value

        source_count = 1
        donor_count = 1
        face_order = source_faces_by_value(source_count=source_count, donor_count=donor_count)
        _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=donor_count)
        slots = build_rack_slots_aligned(stations_per_flank=2, depth_tiers=2)
        slot_ids = [s.slot_id for s in slots]

        overflow = [_sku(2, "A", 8), _sku(3, "B", 10)]
        result = assign_rack_slots_colocated(
            overflow,
            slot_ids,
            face_order=face_order,
            aisle_map=aisle_map,
        )
        virtual_face = face_order[0]
        mapped = aisle_map[virtual_face]
        flank = "L" if mapped.side.value == "left" else "R"
        front = rack_slot_at_station(mapped.station, flank, tier=0)
        # Hungarian prefers two nearby front tiers over front+back at one bay
        # when both SKUs share the same promote target.
        assert rack_depth_tier(result["A"]) == 0
        assert rack_depth_tier(result["B"]) == 0
        assert result["A"] == front
        assert abs(rack_station_index(result["B"]) - mapped.station) <= 1
    finally:
        rp.DEFAULT_RACK_DEPTH_TIERS = DEFAULT_RACK_DEPTH_TIERS


def test_assign_rack_fills_fronts_before_backs() -> None:
    from layout_advisor.layout import reserve_policy as rp

    old = rp.DEFAULT_RACK_DEPTH_TIERS
    rp.DEFAULT_RACK_DEPTH_TIERS = 2
    try:
        overflow = [_sku(1, "A", 8), _sku(2, "B", 10), _sku(3, "C", 12), _sku(4, "D", 14)]
        slots = ["R-L01", "R-L02", "R-L03", "R-L04"]
        result = assign_rack_slots(overflow, slots)
        assert result["A"] == "R-L01"
        assert result["B"] == "R-L03"
        assert result["C"] == "R-L02"
        assert result["D"] == "R-L04"
        assert rack_depth_tier(result["A"]) == 0
        assert rack_depth_tier(result["B"]) == 0
    finally:
        rp.DEFAULT_RACK_DEPTH_TIERS = old


def test_donor_staging_uses_tail_slots() -> None:
    slots = [f"R-L{i:02d}" for i in range(1, 6)]
    reserved = donor_staging_slot_ids(slots, reserve=2)
    assert reserved == ["R-L04", "R-L05"]


def test_select_rack_slot_skips_back_blocked_by_different_sku() -> None:
    from layout_advisor.layout import reserve_policy as rp

    old = rp.DEFAULT_RACK_DEPTH_TIERS
    rp.DEFAULT_RACK_DEPTH_TIERS = 2
    try:
        chosen = select_rack_slot(
            candidate_slot_ids=["R-L01", "R-L02"],
            occupied={"R-L01": "BLOCKER"},
            incoming_item_id="OTHER",
        )
        assert chosen is None
    finally:
        rp.DEFAULT_RACK_DEPTH_TIERS = old


def test_select_rack_slot_allows_back_for_same_sku_replenishment() -> None:
    from layout_advisor.layout import reserve_policy as rp

    old = rp.DEFAULT_RACK_DEPTH_TIERS
    rp.DEFAULT_RACK_DEPTH_TIERS = 2
    try:
        chosen = select_rack_slot(
            candidate_slot_ids=["R-L01", "R-L02"],
            occupied={"R-L01": "BLOCKER"},
            incoming_item_id="BLOCKER",
        )
        assert chosen == "R-L02"
    finally:
        rp.DEFAULT_RACK_DEPTH_TIERS = old


def test_select_rack_slot_never_back_while_front_open() -> None:
    from layout_advisor.layout import reserve_policy as rp

    old = rp.DEFAULT_RACK_DEPTH_TIERS
    rp.DEFAULT_RACK_DEPTH_TIERS = 2
    try:
        chosen = select_rack_slot(
            candidate_slot_ids=["R-L01", "R-L02", "R-L03", "R-L04"],
            occupied={"R-L02": "IN_BACK"},
            incoming_item_id="NEW",
        )
        assert chosen == "R-L01"
        assert rack_depth_tier(chosen) == 0
    finally:
        rp.DEFAULT_RACK_DEPTH_TIERS = old


def test_select_rack_slot_prefers_open_front_over_back_bay() -> None:
    chosen = select_rack_slot(
        candidate_slot_ids=["R-L01", "R-L02", "R-L03", "R-L04"],
        occupied={"R-L02": "IN_BACK"},
        incoming_item_id="NEW",
    )
    assert chosen == "R-L01"
    assert rack_depth_tier(chosen) == 0


def test_select_rack_slot_prefers_empty_front() -> None:
    chosen = select_rack_slot(
        candidate_slot_ids=["R-L01", "R-L02", "R-L03"],
        occupied={},
        incoming_item_id="NEW",
    )
    assert chosen == "R-L01"


def test_source_faces_prefer_non_donor_stations() -> None:
    faces = source_faces_by_value(source_count=4, donor_count=2)
    assert len(faces) == 4
    assert len(set(faces)) == 4


def test_assign_rack_many_skus_fronts_before_any_back() -> None:
    from layout_advisor.layout import reserve_policy as rp

    old = rp.DEFAULT_RACK_DEPTH_TIERS
    rp.DEFAULT_RACK_DEPTH_TIERS = 2
    try:
        overflow = [_sku(i, f"SKU{i}", 6 + (i % 10)) for i in range(1, 7)]
        slots = ["R-L01", "R-L02", "R-L03", "R-L04", "R-L05", "R-L06"]
        result = assign_rack_slots(overflow, slots)
        fronts = [sid for sid in result.values() if rack_depth_tier(sid) == 0]
        backs = [sid for sid in result.values() if rack_depth_tier(sid) == 1]
        assert len(fronts) == 3
        assert len(backs) == 3
    finally:
        rp.DEFAULT_RACK_DEPTH_TIERS = old
