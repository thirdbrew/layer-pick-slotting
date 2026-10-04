"""Dynamic slotting helpers — rest-of-day demand, eviction scoring, travel-aware ADD."""

from __future__ import annotations

from datetime import datetime, time
from typing import TYPE_CHECKING

from slotting.models import OrderArea, Pallet, WorldState

if TYPE_CHECKING:
    from slotting.rules import RuleConfig


def rest_of_ship_day_end(as_of: datetime) -> datetime:
    """Last moment of the operational ship day containing *as_of*."""
    return datetime.combine(as_of.date(), time(23, 59, 59))


def rest_of_ship_day_layers_by_item(state: WorldState) -> dict[str, int]:
    """Per-item layer demand from *state.as_of* through end of ship day.

    Single O(orders) pass — use this (not per-item calls) inside loops.
    """
    cutoff = rest_of_ship_day_end(state.as_of)
    totals: dict[str, int] = {}
    for o in state.orders:
        if o.area == OrderArea.THIS_LINE and state.as_of <= o.due_at <= cutoff:
            totals[o.item_id] = totals.get(o.item_id, 0) + o.qty_layers
    return totals


def rest_of_ship_day_layers(state: WorldState, item_id: str) -> int:
    """Layer demand for *item_id* from *state.as_of* through end of ship day."""
    cutoff = rest_of_ship_day_end(state.as_of)
    return sum(
        o.qty_layers
        for o in state.orders
        if o.area == OrderArea.THIS_LINE
        and o.item_id == item_id
        and state.as_of <= o.due_at <= cutoff
    )


def items_without_rest_of_ship_day_demand(state: WorldState) -> set[str]:
    """SKUs with no remaining pick demand today after *state.as_of*."""
    face_items = {p.item_id for p in state.pallets_in_source_slots()}
    order_items = {
        o.item_id
        for o in state.orders
        if o.area == OrderArea.THIS_LINE and o.due_at >= state.as_of
    }
    candidates = face_items | order_items
    layers_by_item = rest_of_ship_day_layers_by_item(state)
    return {item_id for item_id in candidates if layers_by_item.get(item_id, 0) <= 0}


def pallet_eviction_key(
    state: WorldState,
    pallet: Pallet,
    config: RuleConfig,
    *,
    slot_id: str | None = None,
    layers_by_item: dict[str, int] | None = None,
) -> tuple[int, int, int, float, str]:
    """
    Sort key for eviction victims — lower tuple = remove first.

    Prefers SKUs with no rest-of-day demand, then higher setup rank (less critical),
    then slots farther from the SKU's preferred face (free prime travel positions).
    Pass *layers_by_item* (from :func:`rest_of_ship_day_layers_by_item`) when
    scoring many pallets to avoid rescanning orders per pallet.
    """
    if layers_by_item is not None:
        layers = layers_by_item.get(pallet.item_id, 0)
    else:
        layers = rest_of_ship_day_layers(state, pallet.item_id)
    rank = config.setup_rank_by_item.get(pallet.item_id, 10**6)
    stations = config.source_face_station_by_id
    travel_penalty = 0.0
    if slot_id and stations:
        pref = config.preferred_source_face_by_item.get(pallet.item_id)
        pref_station = stations.get(pref or "", stations.get(slot_id, 0))
        slot_station = stations.get(slot_id, pref_station)
        travel_penalty = abs(slot_station - pref_station)
    return (layers, -rank, pallet.layers_remaining, travel_penalty, pallet.pallet_id)


def select_slot_eviction_victim(
    state: WorldState,
    config: RuleConfig,
    *,
    slot_ids: set[str] | None = None,
    exclude_pallet_ids: set[str] | None = None,
) -> tuple[Pallet, str] | None:
    """
    Lowest-value source face pallet safe to evict (no rest-of-day demand).

    Returns ``(pallet, slot_id)`` or ``None`` when no eligible victim exists.
    """
    exclude = exclude_pallet_ids or set()
    slot_by_pallet = {
        s.occupied_by_pallet_id: s.slot_id
        for s in state.source_slots
        if s.occupied_by_pallet_id
    }
    layers_by_item = rest_of_ship_day_layers_by_item(state)
    face_items = {p.item_id for p in state.pallets_in_source_slots()}
    stale_items = {i for i in face_items if layers_by_item.get(i, 0) <= 0}
    candidates: list[tuple[tuple[int, int, int, float, str], Pallet, str]] = []
    for pallet in state.pallets_in_source_slots():
        if pallet.pallet_id in exclude:
            continue
        if pallet.item_id not in stale_items:
            continue
        slot_id = slot_by_pallet.get(pallet.pallet_id)
        if slot_id is None:
            continue
        if slot_ids is not None and slot_id not in slot_ids:
            continue
        key = pallet_eviction_key(
            state, pallet, config, slot_id=slot_id, layers_by_item=layers_by_item
        )
        candidates.append((key, pallet, slot_id))
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[0])
    _, pallet, slot_id = candidates[0]
    return pallet, slot_id


def best_empty_source_slot(
    empty_slots: list[str],
    *,
    item_id: str | None,
    preferred_slot: str | None,
    config: RuleConfig | None,
) -> str:
    """
    Pick the travel-optimal empty source face for an ADD.

    Prefers the planned face, then nearest station to that face, then lowest station id.
    """
    if not empty_slots:
        raise ValueError("empty_slots must be non-empty")
    if preferred_slot and preferred_slot in empty_slots:
        return preferred_slot
    cfg = config
    if cfg is None:
        return sorted(empty_slots)[0]
    stations = cfg.source_face_station_by_id
    if not stations:
        return sorted(empty_slots)[0]
    anchor = preferred_slot or (cfg.preferred_source_face_by_item.get(item_id or "") if item_id else None)
    anchor_station = stations.get(anchor or "", sum(stations.values()) // max(len(stations), 1))
    return min(
        empty_slots,
        key=lambda slot_id: (abs(stations.get(slot_id, anchor_station) - anchor_station), slot_id),
    )
