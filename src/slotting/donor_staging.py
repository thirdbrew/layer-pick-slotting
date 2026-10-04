"""Completed donor builds — exit to wrapping (rack-only) or stage behind face / rack (legacy)."""

from __future__ import annotations

from slotting.models import (
    OrderArea,
    OrderStatus,
    Recommendation,
    RecommendationAction,
    RuleId,
    Slot,
    WorldState,
)
from slotting.rules import WAVED_HORIZON


def _items_with_waved_demand(state: WorldState) -> set[str]:
    cutoff = state.as_of + WAVED_HORIZON
    return {
        o.item_id
        for o in state.orders
        if o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    }


def empty_donor_backup_slots(state: WorldState) -> list[Slot]:
    """Behind-donor staging row (completed build pallet)."""
    return sorted(
        (s for s in state.destination_slots if not s.backup_pallet_id),
        key=lambda s: s.slot_id,
    )


def destination_slots_available(state: WorldState, *, include_rack_aisle: bool = False, rack_only_reserve: bool = True) -> list[str]:
    """
    Donor faces open now or freeable by clearing a completed build off the face.

    Rack-only: completed builds exit to wrapping immediately — no rack slot required.
    """
    from slotting.resupply_rack import empty_rack_slots

    waved = _items_with_waved_demand(state)
    available: list[str] = []
    has_rack = bool(empty_rack_slots(state)) if include_rack_aisle else False

    for slot in state.destination_slots:
        if not slot.occupied_by_pallet_id:
            available.append(slot.slot_id)
            continue
        pallet = state.pallet_by_id(slot.occupied_by_pallet_id)
        if pallet is None or pallet.has_waved_demand(waved):
            continue
        if rack_only_reserve or not slot.backup_pallet_id or has_rack:
            available.append(slot.slot_id)
    return sorted(set(available))


def rule_dest_stage_completed_build(
    state: WorldState,
    *,
    include_rack_aisle: bool = False,
    rack_only_reserve: bool = True,
) -> list[Recommendation]:
    """
    Clear completed donor builds from build faces.

    Rack-only: assist removes the pallet to wrapping (out of the LP module).
    Legacy: stage behind donor or on side rack when backup rows exist.
    """
    from slotting.resupply_rack import empty_rack_slots

    waved = _items_with_waved_demand(state)
    recs: list[Recommendation] = []
    rack_targets = empty_rack_slots(state) if include_rack_aisle else []

    for slot in state.destination_slots:
        if not slot.occupied_by_pallet_id:
            continue
        pallet = state.pallet_by_id(slot.occupied_by_pallet_id)
        if pallet is None or pallet.has_waved_demand(waved):
            continue
        if rack_only_reserve:
            recs.append(
                Recommendation(
                    action=RecommendationAction.FLAG_REMOVE,
                    pallet_id=pallet.pallet_id,
                    item_id=pallet.item_id,
                    slot_id=slot.slot_id,
                    reason=(
                        f"Completed build {pallet.pallet_id} on {slot.slot_id}; "
                        f"remove to wrapping and free build face."
                    ),
                    rule=RuleId.DEST_STAGE_COMPLETE,
                )
            )
        elif not slot.backup_pallet_id:
            recs.append(
                Recommendation(
                    action=RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY,
                    pallet_id=pallet.pallet_id,
                    item_id=pallet.item_id,
                    slot_id=slot.slot_id,
                    reason=(
                        f"Completed build {pallet.pallet_id} on {slot.slot_id}; "
                        f"stage behind donor to free build face."
                    ),
                    rule=RuleId.DEST_STAGE_COMPLETE,
                )
            )
        elif rack_targets:
            recs.append(
                Recommendation(
                    action=RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY,
                    pallet_id=pallet.pallet_id,
                    item_id=pallet.item_id,
                    slot_id=slot.slot_id,
                    reason=(
                        f"Completed build {pallet.pallet_id} on {slot.slot_id}; "
                        f"stage to side rack: donor backup full."
                    ),
                    rule=RuleId.DEST_STAGE_COMPLETE,
                )
            )
    return recs
