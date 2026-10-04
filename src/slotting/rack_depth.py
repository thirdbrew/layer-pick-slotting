"""Rack bay logic — aligned 1:1 with aisle stations (front row default)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from layout_advisor.layout import reserve_policy as _reserve_policy

if TYPE_CHECKING:
    from layout_advisor.layout.slot_geometry import MappedSlot


@dataclass(frozen=True)
class RankedSku:
    """One SKU in setup priority order."""

    rank: int
    item_id: str
    layers: int
    earliest_due: datetime


def rack_slot_number(slot_id: str) -> int:
    digits = "".join(c for c in slot_id if c.isdigit())
    return int(digits) if digits else 0


def rack_flank(slot_id: str) -> str:
    return "L" if "-L" in slot_id else "R"


def rack_depth_tier(slot_id: str) -> int:
    """0 = aisle-adjacent front, 1 = second deep (unused when depth tiers is 1)."""
    if _reserve_policy.DEFAULT_RACK_DEPTH_TIERS <= 1:
        return 0
    return (rack_slot_number(slot_id) - 1) % _reserve_policy.DEFAULT_RACK_DEPTH_TIERS


def rack_station_index(slot_id: str) -> int:
    """0-based aisle station index for this rack bay."""
    depth = max(1, _reserve_policy.DEFAULT_RACK_DEPTH_TIERS)
    return (rack_slot_number(slot_id) - 1) // depth


def rack_bay_pairs(slot_ids: list[str]) -> list[tuple[str, str | None]]:
    """
    Group rack IDs into 2-deep bays per flank.

    Consecutive IDs form a bay: R-L01 front, R-L02 back, R-L03 front, …
    """
    by_flank: dict[str, list[str]] = {"L": [], "R": []}
    for slot_id in slot_ids:
        by_flank[rack_flank(slot_id)].append(slot_id)
    bays: list[tuple[str, str | None]] = []
    for flank in ("L", "R"):
        ordered = sorted(by_flank[flank], key=rack_slot_number)
        for index in range(0, len(ordered), 2):
            front = ordered[index]
            back = ordered[index + 1] if index + 1 < len(ordered) else None
            bays.append((front, back))
    return bays


def rack_front_slot(slot_id: str) -> str:
    """Front (aisle) slot for the bay containing *slot_id*."""
    if rack_depth_tier(slot_id) == 0:
        return slot_id
    num = rack_slot_number(slot_id) - 1
    prefix = slot_id.rsplit(f"{rack_slot_number(slot_id):02d}", 1)[0]
    return f"{prefix}{num:02d}"


def rack_back_slot(slot_id: str) -> str | None:
    """Back slot paired with *slot_id*, or None if *slot_id* is already back tier."""
    if rack_depth_tier(slot_id) == 1:
        return slot_id
    num = rack_slot_number(slot_id) + 1
    prefix = slot_id.rsplit(f"{rack_slot_number(slot_id):02d}", 1)[0]
    return f"{prefix}{num:02d}"


def donor_staging_slot_ids(rack_slot_ids: list[str], *, reserve: int) -> list[str]:
    """Rack positions held empty at setup for legacy donor staging on rack."""
    if reserve <= 0:
        return []
    return list(rack_slot_ids[-reserve:])


def _flank_ordered_tiers(rack_slot_ids: list[str]) -> tuple[list[str], list[str]]:
    """L fronts → R fronts → L backs → R backs (station order within each)."""
    fronts: list[str] = []
    backs: list[str] = []
    for flank in ("L", "R"):
        flank_ids = sorted(
            (slot_id for slot_id in rack_slot_ids if f"-{flank}" in slot_id),
            key=rack_slot_number,
        )
        for slot_id in flank_ids:
            if rack_depth_tier(slot_id) == 0:
                fronts.append(slot_id)
            else:
                backs.append(slot_id)
    return fronts, backs


def rack_slot_at_station(station: int, flank: str, *, tier: int = 0) -> str:
    """Rack slot ID for a 0-based aisle station on flank ``L`` or ``R``."""
    depth = max(1, _reserve_policy.DEFAULT_RACK_DEPTH_TIERS)
    num = station * depth + tier + 1
    return f"R-{flank}{num:02d}"


def _rack_slot_pick_key(
    slot_id: str,
    *,
    prefer_station: int | None,
    prefer_flank: str | None,
) -> tuple[int, int, int, str, int]:
    station = rack_station_index(slot_id)
    flank = rack_flank(slot_id)
    tier = rack_depth_tier(slot_id)
    station_dist = abs(station - prefer_station) if prefer_station is not None else 0
    flank_penalty = 0 if prefer_flank is None or flank == prefer_flank else 1
    return (tier, station_dist, flank_penalty, flank, station)


def _rack_match_cost(
    slot_id: str,
    *,
    prefer_station: int,
    prefer_flank: str,
) -> float:
    """Scalar cost for Hungarian rack→promote-face matching."""
    tier = rack_depth_tier(slot_id)
    station = rack_station_index(slot_id)
    flank = rack_flank(slot_id)
    flank_penalty = 0 if flank == prefer_flank else 1
    return tier * 1000.0 + flank_penalty * 100.0 + abs(station - prefer_station)


def assign_rack_slots_colocated(
    overflow: list[RankedSku],
    rack_slot_ids: list[str],
    *,
    face_order: list[str],
    aisle_map: dict[str, MappedSlot],
    virtual_face_by_item: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Assign overflow SKUs to rack bays colocated with their promote target face.

    Each SKU uses *virtual_face_by_item* when provided (preferred entry face);
    otherwise falls back to ``face_order[index % len(face_order)]``.
    Uses minimum-cost matching (Hungarian) so competing SKUs share rack stations
    optimally rather than first-come-first-served greedy colocation.
    """
    if not overflow or not rack_slot_ids or not face_order:
        return assign_rack_slots(overflow, rack_slot_ids)

    allowed = set(rack_slot_ids)
    fronts, backs = _flank_ordered_tiers(rack_slot_ids)
    candidates = fronts if _reserve_policy.DEFAULT_RACK_DEPTH_TIERS <= 1 else fronts + backs
    candidates = [slot_id for slot_id in candidates if slot_id in allowed]
    if not candidates:
        return assign_rack_slots(overflow, rack_slot_ids)

    targets: list[tuple[str, int, str]] = []
    for index, sku in enumerate(overflow):
        if virtual_face_by_item and sku.item_id in virtual_face_by_item:
            virtual_face = virtual_face_by_item[sku.item_id]
        else:
            from layout_advisor.planning.wave_slotting import overflow_promote_face

            virtual_face = overflow_promote_face(
                sku.rank - 1,
                face_order,
                source_count=len(face_order),
            )
        mapped = aisle_map[virtual_face]
        flank = "L" if mapped.side.value == "left" else "R"
        targets.append((sku.item_id, mapped.station, flank))

    from layout_advisor.planning.build_anchored_slotting import hungarian_assignment

    n_slots = len(candidates)
    matrix = [
        [
            _rack_match_cost(
                slot_id,
                prefer_station=station,
                prefer_flank=flank,
            )
            for slot_id in candidates
        ]
        for _item_id, station, flank in targets
    ]
    col_assign = hungarian_assignment(matrix)
    assignments: dict[str, str] = {}
    used_cols: set[int] = set()
    for row_idx, (item_id, _station, _flank) in enumerate(targets):
        col = col_assign[row_idx]
        # col < 0 is UNASSIGNED (more targets than candidate slots). It used to
        # arrive as a fabricated colliding index and get caught by the
        # `in used_cols` arm; naming it keeps the same first-free fallback but
        # stops a negative index from ever reaching a list lookup.
        if col < 0 or col in used_cols or col >= n_slots:
            free = next((j for j in range(n_slots) if j not in used_cols), None)
            col = free if free is not None else col
        if col < n_slots:
            used_cols.add(col)
            assignments[item_id] = candidates[col]
    return assignments


def assign_rack_slots(
    overflow: list[RankedSku],
    rack_slot_ids: list[str],
) -> dict[str, str]:
    """
    Assign overflow SKUs to rack slots.

    1. Fill every front tier on L, then every front on R (50 + 50 columns).
    Back tiers only when rack depth tiers > 1.
    """
    if not overflow or not rack_slot_ids:
        return {}

    remaining = list(overflow)
    assignments: dict[str, str] = {}
    fronts, backs = _flank_ordered_tiers(rack_slot_ids)

    for front in fronts:
        if not remaining:
            break
        assignments[remaining[0].item_id] = front
        remaining.pop(0)

    for back in backs:
        if _reserve_policy.DEFAULT_RACK_DEPTH_TIERS <= 1 or not remaining:
            break
        assignments[remaining[0].item_id] = back
        remaining.pop(0)

    return assignments


def select_rack_slot(
    *,
    candidate_slot_ids: list[str],
    occupied: dict[str, str],
    incoming_item_id: str,
    prefer_front: bool = True,
    prefer_station: int | None = None,
    prefer_flank: str | None = None,
) -> str | None:
    """
    Pick an empty rack slot.

    1. Any open front tier, nearest *prefer_station* / *prefer_flank* when set.
    2. Back tier only when depth > 1 and the front of that bay is already occupied.
    """
    if not candidate_slot_ids:
        return None

    open_fronts = [
        slot_id
        for slot_id in candidate_slot_ids
        if rack_depth_tier(slot_id) == 0 and slot_id not in occupied
    ]
    if prefer_front and open_fronts:
        return min(
            open_fronts,
            key=lambda slot_id: _rack_slot_pick_key(
                slot_id,
                prefer_station=prefer_station,
                prefer_flank=prefer_flank,
            ),
        )

    if _reserve_policy.DEFAULT_RACK_DEPTH_TIERS <= 1:
        return None

    open_backs = [
        slot_id
        for slot_id in candidate_slot_ids
        if rack_depth_tier(slot_id) == 1
        and slot_id not in occupied
        and rack_front_slot(slot_id) in occupied
        and occupied.get(rack_front_slot(slot_id)) == incoming_item_id
    ]
    if open_backs:
        return min(
            open_backs,
            key=lambda slot_id: _rack_slot_pick_key(
                slot_id,
                prefer_station=prefer_station,
                prefer_flank=prefer_flank,
            ),
        )

    return None
