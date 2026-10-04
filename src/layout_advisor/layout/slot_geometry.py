"""Map simulation slot IDs (S01, D01) to dual-sided aisle geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from layout_advisor.layout.aisle_plan import AisleLayoutPlan, SlotRole, SlotSide, build_aisle_layout
from slotting.generate_slots import donor_slot_ids, source_slot_ids

@dataclass(frozen=True)
class MappedSlot:
    """Simulation slot placed on the physical aisle grid."""

    sim_slot_id: str
    aisle_slot_id: str
    side: SlotSide
    station: int
    role: SlotRole
    zone: str  # source_face, source_backup, donor_face, donor_staging, rack


def build_aisle_slot_map(
    *,
    source_count: int,
    donor_count: int,
    rack_only_reserve: bool = True,
    center_gap_ft: float = 0.0,
    picker_count: int = 1,
) -> tuple[AisleLayoutPlan, dict[str, MappedSlot]]:
    """
    Map sim IDs ``S01..S{n}`` and ``D01..D{m}`` onto interleaved left/right aisle pads.

    100 total positions at 70:30 → 50 pads per side (25 stations × 2 flanks).
    (Legacy 75:25 used the same 50/50 pad split with a different source/donor mix.)
    """
    layout = build_aisle_layout(
        source_count=source_count,
        donor_count=donor_count,
        rack_only_reserve=rack_only_reserve,
        center_gap_ft=center_gap_ft,
        picker_count=picker_count,
    )
    sources = source_slot_ids(source_count)
    donors = donor_slot_ids(donor_count)
    mapping: dict[str, MappedSlot] = {}
    s_i = 0
    d_i = 0
    for aisle_slot in layout.slots:
        if aisle_slot.role == SlotRole.SOURCE:
            sim_id = sources[s_i]
            s_i += 1
            zone = "source_face"
        else:
            sim_id = donors[d_i]
            d_i += 1
            zone = "donor_face"
        mapping[sim_id] = MappedSlot(
            sim_slot_id=sim_id,
            aisle_slot_id=aisle_slot.slot_id,
            side=aisle_slot.side,
            station=aisle_slot.position_index,
            role=aisle_slot.role,
            zone=zone,
        )
        if not rack_only_reserve and aisle_slot.has_backup_row:
            mapping[f"{sim_id}-BK"] = MappedSlot(
                sim_slot_id=f"{sim_id}-BK",
                aisle_slot_id=f"{aisle_slot.slot_id}-BK",
                side=aisle_slot.side,
                station=aisle_slot.position_index,
                role=aisle_slot.role,
                zone="source_backup",
            )
        if not rack_only_reserve and aisle_slot.requires_ship_access:
            mapping[f"{sim_id}-ST"] = MappedSlot(
                sim_slot_id=f"{sim_id}-ST",
                aisle_slot_id=f"{aisle_slot.slot_id}-ST",
                side=aisle_slot.side,
                station=aisle_slot.position_index,
                role=aisle_slot.role,
                zone="donor_staging",
            )
    return layout, mapping


def stations_per_side(layout: AisleLayoutPlan) -> int:
    left = max((s.position_index for s in layout.slots if s.side == SlotSide.LEFT), default=0)
    right = max((s.position_index for s in layout.slots if s.side == SlotSide.RIGHT), default=0)
    return max(left, right) + 1


def rack_station_index(slot_id: str, *, stations: int, per_side: int) -> int:
    """Spread rack slots along the aisle next to their flank (legacy single-row view)."""
    try:
        num = int("".join(c for c in slot_id if c.isdigit()) or "0")
    except ValueError:
        num = 0
    if per_side <= 0:
        return 0
    return min(stations - 1, (num - 1) * stations // per_side)


def face_travel_rank_key(
    *,
    station: int,
    donor_stations: set[int],
    center_station: float,
) -> tuple[int, float, int]:
    """
    Sort key for source faces — lower is better for high pick-frequency SKUs.

    Donor-colocated stations first, then distance to *center_station*. Dual-picker
    flank scoping happens in the caller (``rank_face_for_travel``) by passing
    per-flank *donor_stations* and the flank's own center.
    """
    at_donor_station = station in donor_stations
    center_dist = abs(station - center_station)
    return (0 if at_donor_station else 1, center_dist, station)


def flank_donor_stations(
    aisle_map: dict[str, MappedSlot],
    *,
    flank: str,
) -> set[int]:
    """Donor stations on LEFT or RIGHT flank (``left`` / ``right``)."""
    return {
        mapped.station
        for mapped in aisle_map.values()
        if mapped.zone == "donor_face" and mapped.side.value == flank
    }


def donor_slots_interleaved_by_flank(
    donor_slot_ids: list[str],
    aisle_map: dict[str, MappedSlot],
) -> list[str]:
    """Order donor pads L,R,L,R for even prebuild spread across pick zones."""
    left = sorted(
        [slot_id for slot_id in donor_slot_ids if aisle_map.get(slot_id) and aisle_map[slot_id].side == SlotSide.LEFT],
        key=lambda slot_id: aisle_map[slot_id].station,
    )
    right = sorted(
        [slot_id for slot_id in donor_slot_ids if aisle_map.get(slot_id) and aisle_map[slot_id].side == SlotSide.RIGHT],
        key=lambda slot_id: aisle_map[slot_id].station,
    )
    out: list[str] = []
    li = ri = 0
    pick_left = True
    while li < len(left) or ri < len(right):
        if pick_left and li < len(left):
            out.append(left[li])
            li += 1
        elif ri < len(right):
            out.append(right[ri])
            ri += 1
        elif li < len(left):
            out.append(left[li])
            li += 1
        pick_left = not pick_left
    unknown = [slot_id for slot_id in donor_slot_ids if slot_id not in out]
    return out + unknown


def travel_rank_context(
    aisle_map: dict[str, MappedSlot],
    layout: AisleLayoutPlan,
) -> tuple[bool, float, dict[str, set[int]]]:
    """Dual-picker flag, per-flank center station, per-flank donor stations."""
    stations = stations_per_side(layout)
    center = (stations - 1) / 2.0
    dual = layout.dual_picker
    donors_by_flank = {
        "left": flank_donor_stations(aisle_map, flank="left"),
        "right": flank_donor_stations(aisle_map, flank="right"),
    }
    return dual, center, donors_by_flank


def gap_buffer_station_count(layout: AisleLayoutPlan) -> int:
    """How many aisle stations sit under the physical center gap (no staging)."""
    if not layout.dual_picker or layout.center_gap_ft <= 0:
        return 0
    return max(1, math.ceil(layout.center_gap_ft / max(layout.slot_width_ft, 1.0)))


def gap_excluded_stations(layout: AisleLayoutPlan) -> set[int]:
    """
    Center dead zone — no source/donor/rack staging so pickers are not blocked in.

    Applies when dual-picker mode has a positive center gap.
    """
    buf = gap_buffer_station_count(layout)
    if buf <= 0:
        return set()
    stations = stations_per_side(layout)
    center = (stations - 1) / 2.0
    half = max(0, (buf - 1) // 2)
    low = max(0, int(math.floor(center - half)))
    high = min(stations - 1, int(math.ceil(center + half)))
    return set(range(low, high + 1))


def slot_in_gap_zone(
    mapped: MappedSlot | None,
    *,
    layout: AisleLayoutPlan,
) -> bool:
    if mapped is None or not layout.dual_picker or layout.center_gap_ft <= 0:
        return False
    return mapped.station in gap_excluded_stations(layout)


def rank_face_for_travel(
    face_id: str,
    *,
    aisle_map: dict[str, MappedSlot],
    dual_picker: bool,
    center_station: float,
    donors_by_flank: dict[str, set[int]],
) -> tuple[int, float, int]:
    mapped = aisle_map[face_id]
    flank = mapped.side.value
    donor_stations = donors_by_flank.get(flank, set()) if dual_picker else {
        mapped.station for mapped in aisle_map.values() if mapped.zone == "donor_face"
    }
    return face_travel_rank_key(
        station=mapped.station,
        donor_stations=donor_stations,
        center_station=center_station,
    )


def source_faces_by_travel(
    *,
    source_count: int,
    donor_count: int,
    rack_only_reserve: bool = True,
    center_gap_ft: float = 0.0,
    picker_count: int = 1,
) -> list[str]:
    """
    Order source face sim IDs best-first for travel-aware dynamic slotting.

    Highest-ranked (most picked) SKUs should take faces earlier in this list.
    """
    layout, aisle_map = build_aisle_slot_map(
        source_count=source_count,
        donor_count=donor_count,
        rack_only_reserve=rack_only_reserve,
        center_gap_ft=center_gap_ft,
        picker_count=picker_count,
    )
    faces = source_slot_ids(source_count)
    dual_picker, center, donors_by_flank = travel_rank_context(aisle_map, layout)

    def rank_key(face_id: str) -> tuple[int, float, int]:
        return rank_face_for_travel(
            face_id,
            aisle_map=aisle_map,
            dual_picker=dual_picker,
            center_station=center,
            donors_by_flank=donors_by_flank,
        )

    ranked = sorted(faces, key=rank_key)
    return filter_slots_outside_gap(ranked, aisle_map, layout)


def filter_slots_outside_gap(
    slot_ids: list[str],
    aisle_map: dict[str, MappedSlot],
    layout: AisleLayoutPlan,
) -> list[str]:
    """Drop pad IDs that fall in the center gap dead zone."""
    excluded = gap_excluded_stations(layout)
    if not excluded:
        return list(slot_ids)
    return [
        slot_id
        for slot_id in slot_ids
        if (mapped := aisle_map.get(slot_id)) is not None and mapped.station not in excluded
    ]


def source_faces_by_value(
    *,
    source_count: int,
    donor_count: int,
    rack_only_reserve: bool = True,
) -> list[str]:
    """
    Order source face sim IDs from most to least valuable for slotting.

    Uses travel-aware ranking: donor-colocated stations near the aisle center first.
    """
    return source_faces_by_travel(
        source_count=source_count,
        donor_count=donor_count,
        rack_only_reserve=rack_only_reserve,
    )


def donor_face_at_station(
    aisle_map: dict[str, MappedSlot],
    station: int,
) -> str | None:
    """Return the native donor sim slot ID at *station*, if any."""
    for mapped in aisle_map.values():
        if mapped.zone == "donor_face" and mapped.station == station:
            return mapped.sim_slot_id
    return None


def source_face_geography(
    aisle_map: dict[str, MappedSlot],
) -> tuple[dict[str, int], dict[str, str]]:
    """Station index and rack flank (L/R) for each source face sim ID."""
    station_by_id: dict[str, int] = {}
    flank_by_id: dict[str, str] = {}
    for sim_id, mapped in aisle_map.items():
        if mapped.zone != "source_face":
            continue
        station_by_id[sim_id] = mapped.station
        flank_by_id[sim_id] = "L" if mapped.side.value == "left" else "R"
    return station_by_id, flank_by_id


def donor_slot_at_station(
    aisle_map: dict[str, MappedSlot],
    station: int,
    donor_slot_ids: set[str],
    *,
    flank: str | None = None,
) -> str | None:
    """
    Return donor-role sim slot at *station*.

    When *flank* is set (``left`` / ``right``), only pads on that flank match —
    dual-picker builds never cross the center gap.
    """
    matches = [
        slot_id
        for slot_id in donor_slot_ids
        if (mapped := aisle_map.get(slot_id)) is not None
        and mapped.station == station
        and mapped.zone == "donor_face"
        and (flank is None or mapped.side.value == flank)
    ]
    return sorted(matches)[0] if matches else None


def nearest_donor_on_flank(
    aisle_map: dict[str, MappedSlot],
    station: int,
    flank: str,
    donor_slot_ids: set[str],
    *,
    gap_excluded: set[int] | None = None,
) -> str | None:
    """Closest donor pad on the same flank (no gap crossing)."""
    gap_excluded = gap_excluded or set()
    ranked: list[tuple[int, int, str]] = []
    for slot_id in donor_slot_ids:
        mapped = aisle_map.get(slot_id)
        if mapped is None:
            continue
        if mapped.zone != "donor_face" or mapped.side.value != flank:
            continue
        if mapped.station in gap_excluded:
            continue
        ranked.append((abs(mapped.station - station), mapped.station, slot_id))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]


def staging_anchor_for_assignment(
    assignment_slot_id: str,
    *,
    location: str,
    aisle_map: dict[str, MappedSlot],
    stations: int,
) -> tuple[int | None, str | None]:
    """Physical aisle station and flank for a staged source face or rack slot."""
    if location == "source_face":
        mapped = aisle_map.get(assignment_slot_id)
        if mapped is not None:
            return mapped.station, mapped.side.value
    if location == "rack":
        station, _ = rack_slot_position(assignment_slot_id, stations=stations)
        flank = "left" if "-L" in assignment_slot_id.upper() else "right"
        return station, flank
    return None, None


def donor_slot_colocated(
    aisle_map: dict[str, MappedSlot],
    station: int,
    flank: str,
    donor_slot_ids: set[str],
) -> str | None:
    """Same station + same flank — picker/assist never crosses the gap."""
    return donor_slot_at_station(
        aisle_map,
        station,
        donor_slot_ids,
        flank=flank,
    )


def staging_station_for_assignment(
    assignment_slot_id: str,
    *,
    location: str,
    aisle_map: dict[str, MappedSlot],
    stations: int,
) -> int | None:
    """Physical aisle station for a staged source face or colocated rack slot."""
    if location == "source_face":
        mapped = aisle_map.get(assignment_slot_id)
        return mapped.station if mapped else None
    if location == "rack":
        station, _ = rack_slot_position(assignment_slot_id, stations=stations)
        return station
    return None


def rack_slot_position(
    slot_id: str,
    *,
    stations: int,
    slots_per_side: int | None = None,
    depth_tiers: int | None = None,
) -> tuple[int, int]:
    """
    Map a rack slot to display grid position beside its aisle station.

    Uses ``DEFAULT_RACK_DEPTH_TIERS`` (reference = 1): ``R-L01`` → station 0,
    ``R-L02`` → station 1, … so rack pallets sit beside the matching face.
    With depth 2: ``R-L01`` front + ``R-L02`` back share station 0.

    Returns ``(station_index, depth_tier)`` where tier 0 is aisle-adjacent (front).
    """
    from layout_advisor.layout.reserve_policy import DEFAULT_RACK_DEPTH_TIERS
    from slotting.rack_depth import rack_slot_number

    depth = max(1, depth_tiers if depth_tiers is not None else DEFAULT_RACK_DEPTH_TIERS)
    num = rack_slot_number(slot_id)
    bay = max(0, (num - 1) // depth)
    tier = 0 if depth <= 1 else (num - 1) % depth
    if slots_per_side is not None and slots_per_side != stations * depth:
        bays_per_side = max(1, (slots_per_side + depth - 1) // depth)
        station = min(stations - 1, bay * stations // bays_per_side)
    else:
        station = min(stations - 1, bay)
    return station, tier
