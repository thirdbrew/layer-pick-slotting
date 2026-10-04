"""Resupply rack aisle — deep staging beside the pick aisle (not 1:1 backup rows)."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from slotting.models import Pallet, Recommendation, RecommendationAction, RuleId, Slot, SlotType, WorldState
from slotting.rack_depth import rack_flank, rack_station_index, select_rack_slot


def rack_slot_ids(count: int) -> list[str]:
    if count < 0:
        raise ValueError("rack count must be >= 0")
    return [f"R{i:02d}" for i in range(1, count + 1)]


def aisle_stations_per_flank(*, source_count: int, donor_count: int) -> int:
    """Aisle stations per L/R flank — matches pick pad count on that side."""
    return (source_count + donor_count + 1) // 2


def build_rack_slots_aligned(
    *,
    stations_per_flank: int,
    depth_tiers: int = 1,
) -> list[Slot]:
    """
    One bay per aisle station on each flank (front row by default).

    ``depth_tiers=1``: R-L01..R-L50 and R-R01..R-R50 (100 slots).
    ``depth_tiers=2``: front + back per station (200 physical positions).
    """
    if stations_per_flank < 0:
        raise ValueError("stations_per_flank must be >= 0")
    if depth_tiers < 1:
        raise ValueError("depth_tiers must be >= 1")
    slots: list[Slot] = []
    for label in ("L", "R"):
        for station in range(stations_per_flank):
            for tier in range(depth_tiers):
                num = station * depth_tiers + tier + 1
                slots.append(
                    Slot(
                        slot_id=f"R-{label}{num:02d}",
                        slot_type=SlotType.RESUPPLY_RACK,
                        occupied_by_pallet_id=None,
                    )
                )
    return slots


def overflow_rack_slot_ids(
    *,
    source_count: int,
    donor_count: int,
    donor_staging_reserve: int = 0,
) -> list[str]:
    """Rack positions for source overflow (all aligned slots when donor builds exit to wrap)."""
    stations = aisle_stations_per_flank(source_count=source_count, donor_count=donor_count)
    all_ids = [s.slot_id for s in build_rack_slots_aligned(stations_per_flank=stations)]
    reserve = min(donor_staging_reserve, len(all_ids))
    return all_ids[: max(0, len(all_ids) - reserve)]


def empty_rack_slots(state: WorldState) -> list[Slot]:
    return sorted(
        (s for s in state.resupply_rack_slots if not s.occupied_by_pallet_id),
        key=lambda s: s.slot_id,
    )


def pick_rack_slot(
    state: WorldState,
    *,
    item_id: str,
    for_donor_build: bool = False,
    donor_staging_reserve: int = 0,
    prefer_station: int | None = None,
    prefer_flank: str | None = None,
    prefer_rack_slot_id: str | None = None,
) -> Slot | None:
    """
    Choose an empty rack slot using floor-simple rules.

    Source/resupply pallets: front tier only when racks are 1-deep.
    Donor builds: prefer the reserved tail slots kept empty at setup.
    """
    from slotting.rack_depth import donor_staging_slot_ids

    racks = state.resupply_rack_slots
    occupied: dict[str, str] = {}
    for slot in racks:
        if not slot.occupied_by_pallet_id:
            continue
        pallet = state.pallet_by_id(slot.occupied_by_pallet_id)
        if pallet is None:
            continue
        occupied[slot.slot_id] = pallet.item_id

    all_ids = [s.slot_id for s in racks]
    if for_donor_build and donor_staging_reserve > 0:
        candidates = [
            slot_id
            for slot_id in donor_staging_slot_ids(all_ids, reserve=donor_staging_reserve)
            if slot_id not in occupied
        ]
    else:
        candidates = [s.slot_id for s in racks if not s.occupied_by_pallet_id]

    if prefer_rack_slot_id and prefer_rack_slot_id in candidates:
        return next(s for s in racks if s.slot_id == prefer_rack_slot_id)

    chosen = select_rack_slot(
        candidate_slot_ids=candidates,
        occupied=occupied,
        incoming_item_id=item_id,
        prefer_station=prefer_station,
        prefer_flank=prefer_flank,
    )
    if chosen is None:
        return None
    return next(s for s in racks if s.slot_id == chosen)


def resupply_relocation_targets(
    state: WorldState,
    *,
    exclude: str | None = None,
    rack_only_reserve: bool = True,
    prefer_station: int | None = None,
    prefer_flank: str | None = None,
) -> list[Slot]:
    """
    Places a displaced face pallet can go without a storage trip.

    Rack-only: side racks only. Legacy: per-face backup rows, then racks.
    """
    targets: list[Slot] = []
    if not rack_only_reserve:
        from slotting.backup_row import empty_resupply_slots

        targets.extend(empty_resupply_slots(state.slots, exclude=exclude))
    targets.extend(empty_rack_slots(state))

    def sort_key(slot: Slot) -> tuple[int, int, int, str]:
        if slot.slot_type != SlotType.RESUPPLY_RACK:
            return (1, 0, 0, slot.slot_id)
        station = rack_station_index(slot.slot_id)
        flank = rack_flank(slot.slot_id)
        station_dist = abs(station - prefer_station) if prefer_station is not None else 0
        flank_penalty = 0 if prefer_flank is None or flank == prefer_flank else 1
        return (0, station_dist, flank_penalty, slot.slot_id)

    return sorted(
        (t for t in targets if t.slot_id != exclude),
        key=sort_key,
    )


def _place_pallet_on_target(pallet: Pallet, target: Slot) -> None:
    if target.slot_type == SlotType.RESUPPLY_RACK:
        target.occupied_by_pallet_id = pallet.pallet_id
        pallet.current_slot = target.slot_id
        return
    if target.slot_type == SlotType.SOURCE:
        target.backup_pallet_id = pallet.pallet_id
        pallet.current_slot = None


@dataclass(frozen=True)
class RackPromotePlan:
    """Pull a staged rack pallet onto a source face."""

    rack_slot_id: str
    rack_pallet_id: str
    item_id: str
    face_slot_id: str
    face_pallet_id: str | None
    relocate_to_slot_id: str | None
    to_storage: bool


def _entry_colocation_hints(
    item_id: str,
    *,
    preferred_face_by_item: Mapping[str, str] | None,
    face_station_by_id: Mapping[str, int] | None,
    face_flank_by_id: Mapping[str, str] | None,
) -> tuple[str | None, int | None, str | None]:
    if not preferred_face_by_item:
        return None, None, None
    preferred_face = preferred_face_by_item.get(item_id)
    if not preferred_face:
        return None, None, None
    station = face_station_by_id.get(preferred_face) if face_station_by_id else None
    flank = face_flank_by_id.get(preferred_face) if face_flank_by_id else None
    return preferred_face, station, flank


def _face_promote_sort_key(
    face_slot_id: str,
    *,
    preferred_face: str | None,
    rack_slot_id: str,
    face_station_by_id: Mapping[str, int] | None,
) -> tuple[int, int, int, str]:
    if preferred_face and face_slot_id == preferred_face:
        return (0, 0, 0, face_slot_id)
    rack_station = rack_station_index(rack_slot_id)
    if face_station_by_id and face_slot_id in face_station_by_id:
        face_station = face_station_by_id[face_slot_id]
        return (1, abs(face_station - rack_station), 0, face_slot_id)
    return (2, 0, 0, face_slot_id)


def plan_rack_to_face(
    state: WorldState,
    *,
    item_id: str,
    waved_items: set[str],
    allow_storage_fallback: bool = False,
    preferred_face_by_item: Mapping[str, str] | None = None,
    face_station_by_id: Mapping[str, int] | None = None,
    face_flank_by_id: Mapping[str, str] | None = None,
    rack_slot_by_item: Mapping[str, str] | None = None,
    resupply_layer_threshold: int = 3,
    waved_displacement: bool = False,
    waved_displacement_rack_swap: bool = False,
) -> RackPromotePlan | None:
    """Promote a waved SKU from the resupply rack aisle onto a source face.

    With ``waved_displacement`` (experiment lever, default off) a promote that
    finds no empty/non-waved face may evict the waved face holding the LOWEST
    near-term waved layers — only when strictly below the incoming item's.

    With ``waved_displacement_rack_swap`` (DISP-RACK-SWAP arm, default off) a
    displacement that finds no landing slot may put the displaced pallet in
    the rack slot the promoting pallet is vacating — a 1-for-1 swap.
    """
    if item_id not in waved_items or item_id in state.excluded_item_ids:
        return None

    assigned_rack_id = rack_slot_by_item.get(item_id) if rack_slot_by_item else None
    rack_slot = None
    if assigned_rack_id:
        rack_slot = next(
            (
                s
                for s in state.resupply_rack_slots
                if s.slot_id == assigned_rack_id
                and s.occupied_by_pallet_id
                and (p := state.pallet_by_id(s.occupied_by_pallet_id)) is not None
                and p.item_id == item_id
                and state.is_eligible_for_line(p)
            ),
            None,
        )
    if rack_slot is None:
        rack_slot = next(
            (
                s
                for s in state.resupply_rack_slots
                if s.occupied_by_pallet_id
                and (p := state.pallet_by_id(s.occupied_by_pallet_id)) is not None
                and p.item_id == item_id
                and state.is_eligible_for_line(p)
            ),
            None,
        )
    if rack_slot is None:
        return None

    rack_pallet = state.pallet_by_id(rack_slot.occupied_by_pallet_id)  # type: ignore[arg-type]
    assert rack_pallet is not None

    preferred_face, entry_station, entry_flank = _entry_colocation_hints(
        item_id,
        preferred_face_by_item=preferred_face_by_item,
        face_station_by_id=face_station_by_id,
        face_flank_by_id=face_flank_by_id,
    )

    for face_slot in state.source_slots:
        if not face_slot.occupied_by_pallet_id:
            continue
        face = state.pallet_by_id(face_slot.occupied_by_pallet_id)
        if face is None or face.item_id != item_id:
            continue
        from slotting.inventory import face_stock_needs_resupply

        if not face_stock_needs_resupply(face.layers_remaining, resupply_layer_threshold):
            continue
        rack_target = pick_rack_slot(
            state,
            item_id=item_id,
            prefer_station=entry_station,
            prefer_flank=entry_flank,
        )
        relocate_id = None
        if rack_target is not None and rack_target.slot_id != rack_slot.slot_id:
            if not rack_target.occupied_by_pallet_id:
                relocate_id = rack_target.slot_id
            else:
                targets = resupply_relocation_targets(
                    state,
                    exclude=face_slot.slot_id,
                    prefer_station=entry_station,
                    prefer_flank=entry_flank,
                )
                targets = [t for t in targets if t.slot_id != rack_slot.slot_id]
                if targets:
                    relocate_id = targets[0].slot_id
        return RackPromotePlan(
            rack_slot_id=rack_slot.slot_id,
            rack_pallet_id=rack_pallet.pallet_id,
            item_id=item_id,
            face_slot_id=face_slot.slot_id,
            face_pallet_id=face.pallet_id,
            relocate_to_slot_id=relocate_id,
            to_storage=False,
        )

    empty_faces = [s for s in state.source_slots if not s.occupied_by_pallet_id]
    if empty_faces:
        face_slot_id = empty_faces[0].slot_id
        if preferred_face:
            preferred_empty = next(
                (s.slot_id for s in empty_faces if s.slot_id == preferred_face),
                None,
            )
            if preferred_empty:
                face_slot_id = preferred_empty
            else:
                face_slot_id = min(
                    (s.slot_id for s in empty_faces),
                    key=lambda slot_id: _face_promote_sort_key(
                        slot_id,
                        preferred_face=preferred_face,
                        rack_slot_id=rack_slot.slot_id,
                        face_station_by_id=face_station_by_id,
                    ),
                )
        return RackPromotePlan(
            rack_slot_id=rack_slot.slot_id,
            rack_pallet_id=rack_pallet.pallet_id,
            item_id=item_id,
            face_slot_id=face_slot_id,
            face_pallet_id=None,
            relocate_to_slot_id=None,
            to_storage=False,
        )

    occupied_faces = [s for s in state.source_slots if s.occupied_by_pallet_id]
    occupied_faces.sort(
        key=lambda face_slot: _face_promote_sort_key(
            face_slot.slot_id,
            preferred_face=preferred_face,
            rack_slot_id=rack_slot.slot_id,
            face_station_by_id=face_station_by_id,
        )
    )
    for face_slot in occupied_faces:
        face = state.pallet_by_id(face_slot.occupied_by_pallet_id)
        if face is None or face.item_id in waved_items:
            continue
        rack_target = pick_rack_slot(
            state,
            item_id=face.item_id,
            prefer_station=entry_station,
            prefer_flank=entry_flank,
        )
        if rack_target and rack_target.slot_id != rack_slot.slot_id:
            return RackPromotePlan(
                rack_slot_id=rack_slot.slot_id,
                rack_pallet_id=rack_pallet.pallet_id,
                item_id=item_id,
                face_slot_id=face_slot.slot_id,
                face_pallet_id=face.pallet_id,
                relocate_to_slot_id=rack_target.slot_id,
                to_storage=False,
            )
        targets = resupply_relocation_targets(
            state,
            exclude=face_slot.slot_id,
            prefer_station=entry_station,
            prefer_flank=entry_flank,
        )
        targets = [t for t in targets if t.slot_id != rack_slot.slot_id]
        if targets:
            return RackPromotePlan(
                rack_slot_id=rack_slot.slot_id,
                rack_pallet_id=rack_pallet.pallet_id,
                item_id=item_id,
                face_slot_id=face_slot.slot_id,
                face_pallet_id=face.pallet_id,
                relocate_to_slot_id=targets[0].slot_id,
                to_storage=False,
            )
        if waved_displacement_rack_swap:
            return RackPromotePlan(
                rack_slot_id=rack_slot.slot_id,
                rack_pallet_id=rack_pallet.pallet_id,
                item_id=item_id,
                face_slot_id=face_slot.slot_id,
                face_pallet_id=face.pallet_id,
                relocate_to_slot_id=rack_slot.slot_id,
                to_storage=False,
            )
        if allow_storage_fallback:
            return RackPromotePlan(
                rack_slot_id=rack_slot.slot_id,
                rack_pallet_id=rack_pallet.pallet_id,
                item_id=item_id,
                face_slot_id=face_slot.slot_id,
                face_pallet_id=face.pallet_id,
                relocate_to_slot_id=None,
                to_storage=True,
            )

    if waved_displacement:
        from slotting.inventory import waved_layers_by_item

        # value = near-term waved layers (WAVED_HORIZON window), the same
        # signal the wave rules run on — NOT full rest-of-day demand.
        demand = waved_layers_by_item(state)
        incoming_layers = demand.get(item_id, 0)
        candidates = sorted(
            (
                (demand.get(face.item_id, 0), face_slot, face)
                for face_slot in occupied_faces
                if (face := state.pallet_by_id(face_slot.occupied_by_pallet_id))
                is not None
                and face.item_id in waved_items
                and face.item_id != item_id
                and demand.get(face.item_id, 0) < incoming_layers
            ),
            key=lambda c: (
                c[0],
                _face_promote_sort_key(
                    c[1].slot_id,
                    preferred_face=preferred_face,
                    rack_slot_id=rack_slot.slot_id,
                    face_station_by_id=face_station_by_id,
                ),
            ),
        )
        for _value, face_slot, face in candidates:
            rack_target = pick_rack_slot(
                state,
                item_id=face.item_id,
                prefer_station=entry_station,
                prefer_flank=entry_flank,
            )
            relocate_id = None
            if rack_target and rack_target.slot_id != rack_slot.slot_id:
                relocate_id = rack_target.slot_id
            else:
                targets = resupply_relocation_targets(
                    state,
                    exclude=face_slot.slot_id,
                    prefer_station=entry_station,
                    prefer_flank=entry_flank,
                )
                targets = [t for t in targets if t.slot_id != rack_slot.slot_id]
                if targets:
                    relocate_id = targets[0].slot_id
            if relocate_id is None and waved_displacement_rack_swap:
                relocate_id = rack_slot.slot_id  # 1-for-1 swap into vacated slot
            if relocate_id is None:
                continue  # never displace without a landing slot (no storage loss)
            return RackPromotePlan(
                rack_slot_id=rack_slot.slot_id,
                rack_pallet_id=rack_pallet.pallet_id,
                item_id=item_id,
                face_slot_id=face_slot.slot_id,
                face_pallet_id=face.pallet_id,
                relocate_to_slot_id=relocate_id,
                to_storage=False,
            )
    return None


def item_rack_ready(
    state: WorldState,
    item_id: str,
    *,
    allow_storage_fallback: bool = False,
) -> bool:
    from slotting.rules import _items_with_waved_demand

    waved = _items_with_waved_demand(state)
    return plan_rack_to_face(
        state,
        item_id=item_id,
        waved_items=waved,
        allow_storage_fallback=allow_storage_fallback,
    ) is not None


def rule_rack_promote(
    state: WorldState,
    held_ids: set[str],
    *,
    preferred_face_by_item: Mapping[str, str] | None = None,
    face_station_by_id: Mapping[str, int] | None = None,
    face_flank_by_id: Mapping[str, str] | None = None,
    rack_slot_by_item: Mapping[str, str] | None = None,
    resupply_layer_threshold: int = 3,
    waved_displacement: bool = False,
    waved_displacement_rack_swap: bool = False,
    late_shift_hour: int = 18,
    near_term_hour: int = 14,
    near_term_hours: int = 4,
    flank_local_batching: bool = False,
    morning_batch_end_hour: int = 11,
) -> list[Recommendation]:
    from slotting.inventory import item_needs_face_resupply
    from slotting.inventory_policy import rack_promote_sort_key
    from slotting.rules import _items_with_waved_demand

    recs: list[Recommendation] = []
    waved = _items_with_waved_demand(state)
    for item_id in sorted(
        waved,
        key=lambda sku: rack_promote_sort_key(
            state,
            sku,
            late_shift_hour=late_shift_hour,
            near_term_hour=near_term_hour,
            near_term_hours=near_term_hours,
            preferred_face_by_item=preferred_face_by_item,
            face_station_by_id=face_station_by_id,
            face_flank_by_id=face_flank_by_id,
            flank_local_batching=flank_local_batching,
            morning_batch_end_hour=morning_batch_end_hour,
        ),
    ):
        if not item_needs_face_resupply(state, item_id, threshold=resupply_layer_threshold):
            continue
        plan = plan_rack_to_face(
            state,
            item_id=item_id,
            waved_items=waved,
            preferred_face_by_item=preferred_face_by_item,
            face_station_by_id=face_station_by_id,
            face_flank_by_id=face_flank_by_id,
            rack_slot_by_item=rack_slot_by_item,
            resupply_layer_threshold=resupply_layer_threshold,
            waved_displacement=waved_displacement,
            waved_displacement_rack_swap=waved_displacement_rack_swap,
        )
        if plan is None:
            continue
        if plan.rack_pallet_id in held_ids:
            continue
        if plan.face_pallet_id and plan.face_pallet_id in held_ids:
            continue
        dest = (
            f"rack/resupply {plan.relocate_to_slot_id}"
            if plan.relocate_to_slot_id
            else "empty face"
        )
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_RACK_PROMOTE,
                pallet_id=plan.rack_pallet_id,
                item_id=item_id,
                slot_id=plan.face_slot_id,
                reason=(
                    f"Waved demand for {item_id}; promote rack pallet {plan.rack_pallet_id} "
                    f"to face {plan.face_slot_id} ({dest})."
                ),
                rule=RuleId.RACK_PROMOTE,
            )
        )
    return recs
