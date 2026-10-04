"""Destination (donor/build-face) slot rules — assign and close waved builds."""

from __future__ import annotations

from slotting.models import (
    OrderArea,
    OrderStatus,
    Recommendation,
    RecommendationAction,
    RuleId,
    WorldState,
)
from slotting.rules import WAVED_HORIZON


def _waved_orders_for_item(state: WorldState, item_id: str) -> list:
    cutoff = state.as_of + WAVED_HORIZON
    return [
        o
        for o in state.orders
        if o.item_id == item_id
        and o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    ]


def _waved_layers_for_item(state: WorldState, item_id: str) -> int:
    return sum(o.qty_layers for o in _waved_orders_for_item(state, item_id))


def _items_with_waved_demand(state: WorldState) -> set[str]:
    cutoff = state.as_of + WAVED_HORIZON
    return {
        o.item_id
        for o in state.orders
        if o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    }


def _empty_destination_slot_ids(state: WorldState) -> list[str]:
    return sorted(
        s.slot_id
        for s in state.destination_slots
        if not s.occupied_by_pallet_id
    )


def rule_dest_close_idle(state: WorldState) -> list[Recommendation]:
    """
    FLAG_REMOVE build pallets on destination slots with no waved demand.

    Frees donor/build faces when the current wave no longer needs them.
    """
    recs: list[Recommendation] = []
    waved_items = _items_with_waved_demand(state)
    for pallet in state.pallets_in_destination_slots():
        if pallet.has_waved_demand(waved_items):
            continue
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_REMOVE,
                pallet_id=pallet.pallet_id,
                item_id=pallet.item_id,
                slot_id=pallet.current_slot,
                reason=(
                    f"Close build pallet {pallet.pallet_id} on {pallet.current_slot} "
                    f"({pallet.item_id}): no waved demand in the next 4 hours."
                ),
                rule=RuleId.DEST_CLOSE_IDLE,
            )
        )
    return recs


def rule_dest_assign_waved(
    state: WorldState, *, include_rack_aisle: bool = False, rack_only_reserve: bool = True
) -> list[Recommendation]:
    """
    FLAG_ADD — open a destination build face for waved demand with staged supply.

    Requires the item to already occupy a source slot (layer-pick supply ready).
    """
    from slotting.donor_staging import destination_slots_available

    recs: list[Recommendation] = []
    if include_rack_aisle:
        candidate_slots = destination_slots_available(
            state, include_rack_aisle=True, rack_only_reserve=rack_only_reserve
        )
        waved = _items_with_waved_demand(state)
        slot_ids: list[str] = []
        for slot_id in candidate_slots:
            slot = next(s for s in state.destination_slots if s.slot_id == slot_id)
            if not slot.occupied_by_pallet_id:
                slot_ids.append(slot_id)
                continue
            pallet = state.pallet_by_id(slot.occupied_by_pallet_id)
            if pallet and pallet.item_id not in waved:
                slot_ids.append(slot_id)
    else:
        slot_ids = _empty_destination_slot_ids(state)
    if not slot_ids:
        return recs

    slot_iter = iter(slot_ids)
    for item_id in sorted(_items_with_waved_demand(state)):
        if state.item_on_destination_slot(item_id):
            continue
        if not state.item_in_source_slot(item_id):
            continue
        if item_id in state.excluded_item_ids:
            continue
        try:
            slot_id = next(slot_iter)
        except StopIteration:
            break
        layers = _waved_layers_for_item(state, item_id)
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_ADD,
                pallet_id=None,
                item_id=item_id,
                slot_id=slot_id,
                reason=(
                    f"Waved demand for {item_id} ({layers} layers) with supply in source; "
                    f"open build face on {slot_id}."
                ),
                rule=RuleId.DEST_ASSIGN_WAVED,
            )
        )
    return recs


def rule_dest_wait_no_face(
    state: WorldState, *, include_rack_aisle: bool = False, rack_only_reserve: bool = True
) -> list[Recommendation]:
    """
    FLAG_WAIT for waved, supply-ready items beyond available build-face capacity.

    Continuous queue signal: with F available (empty or freeable) destination
    slots and N supply-ready waved items not yet on a build face, the first F
    items in assignment order are not flagged (DEST_ASSIGN_WAVED seats them);
    the remaining N−F wait. The previous all-or-nothing gate (any available
    slot → zero waits) made wait counts discontinuous in free-pad count: a
    two-pad difference in setup occupancy could flip a day's waits by an
    order of magnitude.
    """
    from slotting.donor_staging import destination_slots_available

    available = len(
        destination_slots_available(
            state, include_rack_aisle=include_rack_aisle, rack_only_reserve=rack_only_reserve
        )
    )
    queued: list[str] = []
    for item_id in sorted(_items_with_waved_demand(state)):
        if state.item_on_destination_slot(item_id):
            continue
        if not state.item_in_source_slot(item_id):
            continue
        if item_id in state.excluded_item_ids:
            continue
        queued.append(item_id)
    recs: list[Recommendation] = []
    for item_id in queued[available:]:
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_WAIT,
                pallet_id=None,
                item_id=item_id,
                slot_id=None,
                reason=(
                    f"Waved demand for {item_id} with staged supply, but no empty "
                    f"destination slot: wait for a build face."
                ),
                rule=RuleId.DEST_WAIT_NO_FACE,
            )
        )
    return recs


def run_destination_rules(state: WorldState, config) -> list[Recommendation]:
    """Evaluate destination-slot rules."""
    from slotting.donor_staging import rule_dest_stage_completed_build

    include_rack = getattr(config, "include_rack_aisle", False)
    rack_only = getattr(config, "rack_only_reserve", True)
    recs: list[Recommendation] = []
    if include_rack and config.is_enabled(RuleId.DEST_STAGE_COMPLETE):
        recs.extend(
            rule_dest_stage_completed_build(
                state, include_rack_aisle=include_rack, rack_only_reserve=rack_only
            )
        )
    if config.is_enabled(RuleId.DEST_CLOSE_IDLE):
        recs.extend(rule_dest_close_idle(state))
    if config.is_enabled(RuleId.DEST_ASSIGN_WAVED):
        recs.extend(
            rule_dest_assign_waved(state, include_rack_aisle=include_rack, rack_only_reserve=rack_only)
        )
    if config.is_enabled(RuleId.DEST_WAIT_NO_FACE):
        recs.extend(
            rule_dest_wait_no_face(state, include_rack_aisle=include_rack, rack_only_reserve=rack_only)
        )
    return recs
