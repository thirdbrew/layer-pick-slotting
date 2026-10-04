"""Pallet layer inventory — seeding, hourly pick depletion, carryover helpers."""

from __future__ import annotations

from collections.abc import Callable

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, Slot, SlotType, WorldState
from slotting.pallet_geometry import (
    face_lp_pickable_layers_for_item,
    lp_pickable_layers_on_pallet,
    max_physical_pallet_layers_for_item,
)
from slotting.wave_release import order_ship_date

if TYPE_CHECKING:
    pass

WAVED_HORIZON = timedelta(hours=4)
RESUPPLY_LAYER_THRESHOLD = 3
# Late-shift sentinel: ``0`` means “any face depth” — more aggressive than the
# daytime threshold of 3. Resolved to a full standard pallet so a freshly
# promoted face (typically 8–10 layers) does not thrash on the second apply pass.
ANY_DEPTH_RESUPPLY_THRESHOLD = 0
ANY_DEPTH_RESUPPLY_LAYERS = 8


def resolve_resupply_threshold(threshold: int) -> int:
    """Expand the late-shift ``0`` sentinel into a concrete layer threshold."""
    if threshold == ANY_DEPTH_RESUPPLY_THRESHOLD:
        return ANY_DEPTH_RESUPPLY_LAYERS
    return threshold


def face_stock_needs_resupply(layers: int, threshold: int) -> bool:
    """True when face stock is low enough to warrant resupply / rack promote.

    ``threshold == 0`` is the documented late-shift sentinel meaning *any*
    practical face depth (resolved to :data:`ANY_DEPTH_RESUPPLY_LAYERS`).
    Positive thresholds use ``layers <= threshold``.
    """
    return layers <= resolve_resupply_threshold(threshold)


def ship_day_layers_for_item(
    orders: list[OrderLine],
    item_id: str,
    ship_date,
) -> int:
    """Total layer demand for *item_id* on a ship date."""
    return sum(
        o.qty_layers
        for o in orders
        if o.item_id == item_id
        and o.area == OrderArea.THIS_LINE
        and order_ship_date(o) == ship_date
    )


def reserve_pallet_layers(
    orders: list[OrderLine],
    item_id: str,
    ship_date,
    *,
    default_layers: int = 8,
) -> int:
    """Layers to place on a reserve pallet for setup staging (height-capped per SKU)."""
    cap = max_physical_pallet_layers_for_item(item_id)
    total = ship_day_layers_for_item(orders, item_id, ship_date)
    if total <= 0:
        return min(default_layers, cap)
    return min(max(default_layers, total), cap)


def build_reserve_pallets(
    item_ids: list[str],
    orders: list[OrderLine],
    ship_date,
    *,
    as_of: datetime,
    default_layers: int = 8,
    pallets_per_sku: int = 1,
    pallets_per_sku_by_item: dict[str, int] | None = None,
    prefix: str = "SWP",
    existing: list[Pallet] | None = None,
    retain_unplanned_carried: bool = True,
    topoff_carried_to_target: bool = False,
) -> list[Pallet]:
    """
    Reserve pool pallets with order-weighted layer counts.

    Slotted carried pallets from *existing* always persist (pallets on faces,
    racks, and pads don't evaporate overnight), and freshly minted IDs never
    collide with a carried pallet's ID — a reused ID makes stale slot refs
    resolve to the wrong physical pallet (ghost-occupied faces block ADD
    admission and teleport mid-day pallets).

    Reuses carried pallets from *existing* when the SKU already has unslotted stock.
    When *retain_unplanned_carried* is False, unslotted carry pallets for SKUs outside
    *item_ids* are dropped (spike-day reserve rebootstrap).
    """
    carried_by_item: dict[str, list[Pallet]] = {}
    slotted_carried: list[Pallet] = []
    used_ids: set[str] = set()
    for pallet in existing or []:
        used_ids.add(pallet.pallet_id)
        if pallet.layers_remaining <= 0:
            continue
        if pallet.current_slot:
            slotted_carried.append(pallet.model_copy(deep=True))
            continue
        carried_by_item.setdefault(pallet.item_id, []).append(pallet)

    pallets: list[Pallet] = list(slotted_carried)
    seq = 1

    def _next_id() -> str:
        nonlocal seq
        while True:
            pid = f"{prefix}-{seq:04d}"
            seq += 1
            if pid not in used_ids:
                used_ids.add(pid)
                return pid

    handled: set[str] = set()
    for item_id in item_ids:
        reuse = carried_by_item.get(item_id, [])
        count = max(1, (pallets_per_sku_by_item or {}).get(item_id, pallets_per_sku))
        if reuse:
            pallets.extend(p.model_copy(deep=True) for p in reuse)
            handled.add(item_id)
            if not topoff_carried_to_target or len(reuse) >= count:
                continue
            mint = count - len(reuse)
        else:
            mint = count
        layers = reserve_pallet_layers(
            orders, item_id, ship_date, default_layers=default_layers
        )
        for copy in range(mint):
            pallets.append(
                Pallet(
                    pallet_id=_next_id(),
                    item_id=item_id,
                    layers_remaining=layers,
                    current_slot=None,
                    arrived_at=as_of,
                )
            )
        handled.add(item_id)
    if retain_unplanned_carried:
        for item_id, pool in carried_by_item.items():
            if item_id in handled:
                continue
            pallets.extend(p.model_copy(deep=True) for p in pool)
    return pallets


def append_staging_pick_boost(
    pallets: list[Pallet],
    orders: list[OrderLine],
    ship_date,
    *,
    boost_by_item: dict[str, int],
    as_of: datetime,
    default_layers: int = 8,
    prefix: str = "BOOST",
) -> list[Pallet]:
    """Add manual staging pallets for SKUs on a staging pick list."""
    if not boost_by_item:
        return pallets
    existing_ids = {p.pallet_id for p in pallets}
    seq = 1
    extra: list[Pallet] = []
    for item_id, count in sorted(boost_by_item.items()):
        if count <= 0:
            continue
        layers = reserve_pallet_layers(orders, item_id, ship_date, default_layers=default_layers)
        for _ in range(count):
            while True:
                pid = f"{prefix}-{seq:04d}"
                seq += 1
                if pid not in existing_ids:
                    break
            extra.append(
                Pallet(
                    pallet_id=pid,
                    item_id=item_id,
                    layers_remaining=layers,
                    current_slot=None,
                    arrived_at=as_of,
                )
            )
            existing_ids.add(pid)
    return pallets + extra


def shortage_pull_candidates(
    state: WorldState,
    *,
    include_rack_aisle: bool = True,
    resupply_threshold: int | None = None,
) -> list[tuple[str, int]]:
    """
    SKUs that would FLAG_WAIT — waved demand, no unslotted stock, no rack ready.

    Returns (item_id, waved_layers) sorted by waved layers descending.
    """
    from slotting.inventory import item_needs_face_add
    from slotting.resupply_rack import item_rack_ready
    from slotting.rules import _eligible_unslotted_pallets, _held_pallet_ids, _items_with_waved_demand
    from slotting.throughput import _waved_demand_by_item

    held = _held_pallet_ids(state)
    threshold = resupply_threshold if resupply_threshold is not None else 2
    waved = _waved_demand_by_item(state)
    candidates: list[tuple[str, int]] = []
    for item_id in _items_with_waved_demand(state):
        if not item_needs_face_add(state, item_id, threshold=threshold):
            continue
        if _eligible_unslotted_pallets(state, item_id, held):
            continue
        if include_rack_aisle and item_rack_ready(state, item_id):
            continue
        candidates.append((item_id, waved.get(item_id, 0)))
    candidates.sort(key=lambda row: (-row[1], row[0]))
    return candidates


def apply_shift_pallet_pulls(
    pallets: list[Pallet],
    slots: list[Slot],
    state: WorldState,
    orders: list[OrderLine],
    ship_date,
    *,
    pulled_today_by_item: dict[str, int],
    max_per_hour: int,
    max_per_sku_for_item: Callable[[str], int] | None = None,
    default_layers: int = 8,
    prefix: str = "PULL",
    include_rack_aisle: bool = True,
    resupply_threshold: int | None = None,
    stage_after_pull: bool = False,
    preferred_face_by_item: dict[str, str] | None = None,
) -> tuple[list[Pallet], list[Slot], int, dict[str, int], list[tuple[str, str, str | None]]]:
    """
    Pull unslotted pallets from upstream storage when waved demand has no floor stock.

    When *stage_after_pull* is True, place each pulled pallet on an empty source face
    immediately (preferred face first).
    """
    events: list[tuple[str, str, str | None]] = []
    if max_per_hour <= 0:
        return pallets, slots, 0, pulled_today_by_item, events

    def sku_cap(item_id: str) -> int:
        if max_per_sku_for_item is not None:
            return max_per_sku_for_item(item_id)
        return 2

    existing_ids = {p.pallet_id for p in pallets}
    seq = max(
        (
            int(pid.rsplit("-", 1)[-1])
            for pid in existing_ids
            if pid.startswith(f"{prefix}-") and pid.rsplit("-", 1)[-1].isdigit()
        ),
        default=0,
    )
    updated = dict(pulled_today_by_item)
    extra: list[Pallet] = []
    pulled_ids: list[str] = []
    pulls = 0

    for item_id, _waved in shortage_pull_candidates(
        state,
        include_rack_aisle=include_rack_aisle,
        resupply_threshold=resupply_threshold,
    ):
        if pulls >= max_per_hour:
            break
        cap = sku_cap(item_id)
        if cap <= 0 or updated.get(item_id, 0) >= cap:
            continue
        layers = reserve_pallet_layers(orders, item_id, ship_date, default_layers=default_layers)
        seq += 1
        pid = f"{prefix}-{seq:04d}"
        while pid in existing_ids:
            seq += 1
            pid = f"{prefix}-{seq:04d}"
        extra.append(
            Pallet(
                pallet_id=pid,
                item_id=item_id,
                layers_remaining=layers,
                current_slot=None,
                arrived_at=state.as_of,
            )
        )
        existing_ids.add(pid)
        pulled_ids.append(pid)
        updated[item_id] = updated.get(item_id, 0) + 1
        pulls += 1

    if not extra:
        return pallets, slots, 0, updated, events

    pallets = pallets + extra
    if stage_after_pull and pulled_ids:
        pallets, slots, staged_map = _stage_pulled_pallets_to_faces(
            pallets,
            slots,
            pallet_ids=pulled_ids,
            preferred_face_by_item=preferred_face_by_item,
        )
        for pid in pulled_ids:
            pallet = next(p for p in pallets if p.pallet_id == pid)
            events.append((pallet.item_id, pid, staged_map.get(pid)))
    else:
        for pid in pulled_ids:
            pallet = next(p for p in pallets if p.pallet_id == pid)
            events.append((pallet.item_id, pid, None))

    return pallets, slots, pulls, updated, events


def idle_assist_pull_candidates(
    state: WorldState,
    *,
    face_layer_threshold: int = 4,
    include_rack_aisle: bool = True,
) -> list[tuple[str, int]]:
    """
    Waved SKUs whose *current face is still picking* but the next pallet is missing.

    Unlike :func:`shortage_pull_candidates` (no active face → WAIT), this targets
    assist slack: face has stock ≤ *face_layer_threshold*, no unslotted reserve,
    no rack ready — bring the next freezer pallet into the aisle (prefer rack).
    """
    from slotting.resupply_rack import item_rack_ready
    from slotting.rules import _eligible_unslotted_pallets, _held_pallet_ids, _items_with_waved_demand
    from slotting.throughput import _waved_demand_by_item

    held = _held_pallet_ids(state)
    waved = _waved_demand_by_item(state)
    threshold = resolve_resupply_threshold(face_layer_threshold)
    candidates: list[tuple[str, int]] = []
    for item_id in _items_with_waved_demand(state):
        if active_face_pallet_count(state, item_id) <= 0:
            continue
        if face_lp_pickable_layers_for_item(state, item_id) > threshold:
            continue
        if _eligible_unslotted_pallets(state, item_id, held):
            continue
        if include_rack_aisle and item_rack_ready(state, item_id):
            continue
        candidates.append((item_id, waved.get(item_id, 0)))
    candidates.sort(key=lambda row: (-row[1], row[0]))
    return candidates


def apply_idle_assist_staging(
    pallets: list[Pallet],
    slots: list[Slot],
    state: WorldState,
    orders: list[OrderLine],
    ship_date,
    *,
    pulled_today_by_item: dict[str, int],
    max_per_hour: int,
    max_per_sku: int = 3,
    face_layer_threshold: int = 4,
    default_layers: int = 8,
    prefer_rack: bool = True,
    include_rack_aisle: bool = True,
    min_free_rack_slots: int = 0,
    prefix: str = "IDLE",
) -> tuple[list[Pallet], list[Slot], int, dict[str, int], list[tuple[str, str, str | None]]]:
    """Pull next pallets for thin active faces; stage onto rack when possible.

    ``min_free_rack_slots`` keeps that much rack headroom for displacement /
    rack promotes — idle staging never congests the aisle (failure mode: idle
    pallets fill the rack and promotes lose their landing slots)."""
    from slotting.inventory_policy import stage_items_to_rack
    from slotting.resupply_rack import pick_rack_slot

    events: list[tuple[str, str, str | None]] = []
    if min_free_rack_slots > 0:
        free_rack = sum(
            1 for s in state.resupply_rack_slots if not s.occupied_by_pallet_id
        )
        max_per_hour = min(max_per_hour, max(0, free_rack - min_free_rack_slots))
    if max_per_hour <= 0:
        return pallets, slots, 0, pulled_today_by_item, events

    existing_ids = {p.pallet_id for p in pallets}
    seq = max(
        (
            int(pid.rsplit("-", 1)[-1])
            for pid in existing_ids
            if pid.startswith(f"{prefix}-") and pid.rsplit("-", 1)[-1].isdigit()
        ),
        default=0,
    )
    updated = dict(pulled_today_by_item)
    extra: list[Pallet] = []
    pulled_ids: list[str] = []
    pulls = 0

    for item_id, _waved in idle_assist_pull_candidates(
        state,
        face_layer_threshold=face_layer_threshold,
        include_rack_aisle=include_rack_aisle,
    ):
        if pulls >= max_per_hour:
            break
        # yield the FINAL per-SKU allowance to reactive shortage pulls —
        # opportunistic staging must never cap-block an emergency pull
        # (otherwise idle pulls eat the budget and the afternoon shortage starves)
        if max_per_sku <= 0 or updated.get(item_id, 0) >= max_per_sku - 1:
            continue
        if prefer_rack:
            probe = WorldState(
                pallets=pallets + extra,
                slots=slots,
                orders=state.orders,
                exclusions=state.exclusions,
                as_of=state.as_of,
            )
            if pick_rack_slot(probe, item_id=item_id) is None:
                continue
        layers = reserve_pallet_layers(orders, item_id, ship_date, default_layers=default_layers)
        seq += 1
        pid = f"{prefix}-{seq:04d}"
        while pid in existing_ids:
            seq += 1
            pid = f"{prefix}-{seq:04d}"
        extra.append(
            Pallet(
                pallet_id=pid,
                item_id=item_id,
                layers_remaining=layers,
                current_slot=None,
                arrived_at=state.as_of,
            )
        )
        existing_ids.add(pid)
        pulled_ids.append(pid)
        updated[item_id] = updated.get(item_id, 0) + 1
        pulls += 1

    if not extra:
        return pallets, slots, 0, updated, events

    pallets = pallets + extra
    staged_items = {next(p for p in pallets if p.pallet_id == pid).item_id for pid in pulled_ids}
    if prefer_rack:
        stage_items_to_rack(pallets, slots, staged_items, as_of=state.as_of)
    pallet_by_id = {p.pallet_id: p for p in pallets}
    for pid in pulled_ids:
        pallet = pallet_by_id[pid]
        events.append((pallet.item_id, pid, pallet.current_slot))
    return pallets, slots, pulls, updated, events


def _stage_pulled_pallets_to_faces(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    pallet_ids: list[str],
    preferred_face_by_item: dict[str, str] | None = None,
) -> tuple[list[Pallet], list[Slot], dict[str, str | None]]:
    """Place freshly pulled pallets onto empty source faces (preferred first)."""
    slot_by_id = {s.slot_id: s for s in slots}
    pallet_by_id = {p.pallet_id: p for p in pallets}
    empty_sources = [
        s for s in slots if s.slot_type == SlotType.SOURCE and not s.occupied_by_pallet_id
    ]
    empty_set = {s.slot_id for s in empty_sources}
    staged: dict[str, str | None] = {}

    for pid in pallet_ids:
        pallet = pallet_by_id.get(pid)
        if pallet is None or pallet.current_slot:
            staged[pid] = pallet.current_slot if pallet else None
            continue
        if active_face_pallet_count_for_pallets(pallets, slots, pallet.item_id) > 0:
            staged[pid] = None
            continue
        pref = (preferred_face_by_item or {}).get(pallet.item_id)
        slot_id: str | None = pref if pref and pref in empty_set else None
        if slot_id is None and empty_sources:
            slot = empty_sources.pop(0)
            slot_id = slot.slot_id
        if slot_id is None:
            staged[pid] = None
            continue
        slot = slot_by_id[slot_id]
        from slotting.inventory_policy import clear_pending_completion

        clear_pending_completion(pallet)
        slot.occupied_by_pallet_id = pid
        pallet.current_slot = slot_id
        empty_set.discard(slot_id)
        staged[pid] = slot_id
    return pallets, slots, staged


def _waved_layers_by_item(state: WorldState) -> dict[str, int]:
    cutoff = state.as_of + WAVED_HORIZON
    demand: dict[str, int] = {}
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.status != OrderStatus.WAVED:
            continue
        if not (state.as_of <= order.due_at <= cutoff):
            continue
        demand[order.item_id] = demand.get(order.item_id, 0) + order.qty_layers
    return demand


waved_layers_by_item = _waved_layers_by_item


def _pickable_pallets(
    pallets: list[Pallet],
    slots: list[Slot],
    item_id: str,
) -> list[Pallet]:
    source_ids = {s.slot_id for s in slots if s.slot_type == SlotType.SOURCE}
    picked: list[Pallet] = []
    for pallet in pallets:
        if pallet.item_id != item_id or pallet.layers_remaining <= 0:
            continue
        if pallet.current_slot and pallet.current_slot in source_ids:
            picked.append(pallet)
    picked.sort(
        key=lambda p: (p.layers_remaining, lp_pickable_layers_on_pallet(p)),
        reverse=True,
    )
    return picked


def apply_hourly_layer_picks(
    pallets: list[Pallet],
    slots: list[Slot],
    state: WorldState,
    *,
    layers_per_hour: float,
    late_shift_retain: bool = True,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
    pick_events: list | None = None,
    demand_by_item: dict[str, int] | None = None,
    picked_by_item: dict[str, int] | None = None,
) -> int:
    """
    Deplete ``layers_remaining`` on face pallets for this hour's waved demand.

    When *pick_events* is set, append :class:`LayerPickEvent` rows (source face
    → donor build when a build face is open for the SKU).

    *demand_by_item* overrides the default waved-demand rollup (order-schedule path).

    *picked_by_item*, when given, is filled with layers picked per SKU so callers
    can credit them back to the order lines they served. Cheaper than reading
    *pick_events*, which builds an object and resolves a donor build per pick.

    Returns total layers picked. Clears depleted pallets from source slots.
    """
    from slotting.layer_pick_log import LayerPickEvent, active_donor_build

    demand = demand_by_item if demand_by_item is not None else _waved_layers_by_item(state)
    if not demand:
        return 0
    # Whole layers only — callers with sub-hour float budgets (e.g. 10-min ticks)
    # must accumulate fractional remainders before calling (see backtest_historical).
    budget = min(max(0, int(layers_per_hour)), sum(demand.values()))
    if budget <= 0:
        return 0

    slot_by_id = {s.slot_id: s for s in slots}
    picked_total = 0
    from slotting.inventory_policy import resupply_priority_key

    items = sorted(
        demand.items(),
        key=lambda row: resupply_priority_key(state, row[0], waved_layers=demand),
    )
    remaining_budget = budget

    for item_id, item_layers in items:
        if remaining_budget <= 0:
            break
        share = min(item_layers, remaining_budget)
        face_pallets = _pickable_pallets(pallets, slots, item_id)
        face_total = sum(
            lp_pickable_layers_on_pallet(pallet) for pallet in face_pallets
        )
        if late_shift_retain and state.as_of.hour < late_shift_hour:
            from slotting.inventory_policy import cap_face_pick_for_late_reserve

            share = cap_face_pick_for_late_reserve(
                state,
                item_id,
                face_layers=face_total,
                take_requested=share,
                late_shift_hour=late_shift_hour,
                final_shift_hour=final_shift_hour,
            )
        for pallet in face_pallets:
            if share <= 0:
                break
            take = min(share, lp_pickable_layers_on_pallet(pallet))
            if take <= 0:
                continue
            pallet.layers_remaining -= take
            share -= take
            remaining_budget -= take
            picked_total += take
            if picked_by_item is not None:
                picked_by_item[item_id] = picked_by_item.get(item_id, 0) + take
            if take > 0 and pick_events is not None and pallet.current_slot:
                build = active_donor_build(pallets, item_id)
                from layout_advisor.layout.pick_order_visual import resolve_order_key_for_pick

                pick_events.append(
                    LayerPickEvent(
                        as_of=state.as_of,
                        item_id=item_id,
                        layers=take,
                        from_pallet_id=pallet.pallet_id,
                        from_slot=pallet.current_slot,
                        to_pallet_id=build.pallet_id if build else None,
                        to_slot=build.current_slot if build else None,
                        order_key=resolve_order_key_for_pick(state, item_id),
                    )
                )
            if pallet.layers_remaining <= 0 and pallet.current_slot:
                slot = slot_by_id.get(pallet.current_slot)
                if slot is not None and slot.occupied_by_pallet_id == pallet.pallet_id:
                    slot.occupied_by_pallet_id = None
                pallet.current_slot = None

    dispose_depleted_pallets(pallets, slots, state=state)

    return picked_total


def floor_inventory_layers(pallets: list[Pallet]) -> tuple[int, int]:
    """Return (layers on slotted pallets, layers in reserve pool)."""
    on_floor = 0
    reserve = 0
    for pallet in pallets:
        if pallet.layers_remaining <= 0:
            continue
        if pallet.current_slot:
            on_floor += pallet.layers_remaining
        else:
            reserve += pallet.layers_remaining
    return on_floor, reserve


def face_layers_for_item(state: WorldState, item_id: str) -> int:
    """Layers on source face(s) for *item_id*; 0 when not on a face."""
    return sum(
        p.layers_remaining
        for p in state.pallets_in_source_slots()
        if p.item_id == item_id
    )


def active_face_pallet_count(state: WorldState, item_id: str) -> int:
    """Source faces with a non-depleted pallet for *item_id*."""
    return sum(
        1
        for p in state.pallets_in_source_slots()
        if p.item_id == item_id and p.layers_remaining > 0
    )


def active_face_pallet_count_for_pallets(
    pallets: list[Pallet],
    slots: list[Slot],
    item_id: str,
) -> int:
    """Like :func:`active_face_pallet_count` without building a full WorldState."""
    source_ids = {s.slot_id for s in slots if s.slot_type == SlotType.SOURCE}
    return sum(
        1
        for p in pallets
        if p.item_id == item_id
        and p.layers_remaining > 0
        and p.current_slot in source_ids
    )


def item_needs_face_resupply(
    state: WorldState,
    item_id: str,
    *,
    threshold: int = RESUPPLY_LAYER_THRESHOLD,
) -> bool:
    """True when LP-pickable face stock warrants resupply (see face_stock_needs_resupply)."""
    return face_stock_needs_resupply(
        face_lp_pickable_layers_for_item(state, item_id),
        threshold,
    )


def item_needs_face_add(
    state: WorldState,
    item_id: str,
    *,
    threshold: int = RESUPPLY_LAYER_THRESHOLD,
) -> bool:
    """
    True when a storage ADD should open a pick face for *item_id*.

    Requires no active face pallet — finish or refresh the current face before
    staging a second pallet of the same SKU onto another slot.
    """
    if active_face_pallet_count(state, item_id) > 0:
        return False
    return face_stock_needs_resupply(
        face_lp_pickable_layers_for_item(state, item_id),
        threshold,
    )


def lowest_face_pallet(state: WorldState, item_id: str) -> tuple[Pallet | None, str | None]:
    """Return (pallet, source_slot_id) for the lowest-layer face pallet of *item_id*."""
    best: Pallet | None = None
    best_slot: str | None = None
    slot_by_pallet = {
        s.occupied_by_pallet_id: s.slot_id
        for s in state.source_slots
        if s.occupied_by_pallet_id
    }
    for pallet in state.pallets_in_source_slots():
        if pallet.item_id != item_id:
            continue
        if best is None or pallet.layers_remaining < best.layers_remaining:
            best = pallet
            best_slot = slot_by_pallet.get(pallet.pallet_id)
    return best, best_slot


def _clear_pallet_from_slots(pallet: Pallet, slots: list[Slot]) -> None:
    """Release *pallet* from any face, rack, or backup position."""
    for slot in slots:
        if slot.occupied_by_pallet_id == pallet.pallet_id:
            slot.occupied_by_pallet_id = None
        if slot.backup_pallet_id == pallet.pallet_id:
            slot.backup_pallet_id = None
    pallet.current_slot = None


def scrub_stale_slot_refs(pallets: list[Pallet], slots: list[Slot]) -> int:
    """
    Clear slot refs to missing or depleted pallets (ghost occupancy).

    Returns count of refs cleared.
    """
    pallet_by_id = {p.pallet_id: p for p in pallets}
    cleared = 0
    for slot in slots:
        for attr in ("occupied_by_pallet_id", "backup_pallet_id"):
            pid = getattr(slot, attr)
            if not pid:
                continue
            pallet = pallet_by_id.get(pid)
            if pallet is None or pallet.layers_remaining <= 0:
                setattr(slot, attr, None)
                cleared += 1
            elif pallet.current_slot == slot.slot_id and pallet.layers_remaining <= 0:
                pallet.current_slot = None
    for pallet in pallets:
        if pallet.layers_remaining <= 0 and pallet.current_slot:
            _clear_pallet_from_slots(pallet, slots)
            cleared += 1
    return cleared


def _donor_build_layers(state: WorldState, item_id: str) -> int:
    """Layers to assign when repurposing an empty shell as a donor build (100in cap)."""
    cap = max_physical_pallet_layers_for_item(item_id)
    demand = _waved_layers_by_item(state).get(item_id, 0)
    if demand > 0:
        return min(max(demand, 1), cap)
    return min(8, cap)


def _eligible_donor_repurpose(state: WorldState, item_id: str) -> bool:
    """True when *item_id* may occupy a donor pad as a build after the pick face empties."""
    if item_id in state.excluded_item_ids:
        return False
    if _waved_layers_by_item(state).get(item_id, 0) > 0:
        return True
    cutoff = state.as_of + WAVED_HORIZON
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.item_id != item_id:
            continue
        if order.status not in (OrderStatus.WAVED, OrderStatus.VISIBLE):
            continue
        if state.as_of <= order.due_at <= cutoff + timedelta(hours=20):
            return True
    return False


def _pick_colocated_donor_slot(
    slots: list[Slot],
    *,
    last_source_slot_id: str | None,
    existing_ids: set[str],
) -> Slot | None:
    """Prefer a donor pad colocated with the pallet's last pick face."""
    from layout_advisor.layout.slot_geometry import build_aisle_slot_map, donor_slot_colocated

    donor_slots = [
        s
        for s in slots
        if s.slot_type == SlotType.DESTINATION
        and not s.occupied_by_pallet_id
        and f"BUILD-{s.slot_id}" not in existing_ids
    ]
    if not donor_slots:
        return None

    if last_source_slot_id:
        source_count = sum(1 for s in slots if s.slot_type == SlotType.SOURCE)
        donor_count = sum(1 for s in slots if s.slot_type == SlotType.DESTINATION)
        _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=donor_count)
        mapped = aisle_map.get(last_source_slot_id)
        if mapped is not None and mapped.zone == "source_face":
            flank = mapped.side.value
            donor_ids = {s.slot_id for s in donor_slots}
            colocated_id = donor_slot_colocated(
                aisle_map,
                mapped.station,
                flank,
                donor_ids,
            )
            if colocated_id:
                match = next((s for s in donor_slots if s.slot_id == colocated_id), None)
                if match is not None:
                    return match
            from layout_advisor.layout.slot_geometry import nearest_donor_on_flank

            near_id = nearest_donor_on_flank(
                aisle_map,
                mapped.station,
                flank,
                donor_ids,
            )
            if near_id:
                match = next((s for s in donor_slots if s.slot_id == near_id), None)
                if match is not None:
                    return match

    return donor_slots[0]


def _try_repurpose_as_donor_build(
    pallet: Pallet,
    slots: list[Slot],
    *,
    state: WorldState,
    existing_ids: set[str],
    colocated: bool = True,
) -> bool:
    """Turn a zero-layer shell into a donor build, preferring colocated donor pads."""
    if not _eligible_donor_repurpose(state, pallet.item_id):
        return False
    if colocated:
        donor_slot = _pick_colocated_donor_slot(
            slots,
            last_source_slot_id=pallet.last_source_slot_id,
            existing_ids=existing_ids,
        )
    else:
        donor_slot = next(
            (
                s
                for s in slots
                if s.slot_type == SlotType.DESTINATION
                and not s.occupied_by_pallet_id
                and f"BUILD-{s.slot_id}" not in existing_ids
            ),
            None,
        )
    if donor_slot is None:
        return False
    build_id = f"BUILD-{donor_slot.slot_id}"
    pallet.pallet_id = build_id
    pallet.layers_remaining = _donor_build_layers(state, pallet.item_id)
    pallet.current_slot = donor_slot.slot_id
    pallet.arrived_at = state.as_of
    donor_slot.occupied_by_pallet_id = build_id
    existing_ids.add(build_id)
    return True


def dispose_depleted_pallets(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    state: WorldState | None = None,
    repurpose_as_donor: bool = True,
    colocated_donor_repurpose: bool = True,
) -> list[Pallet]:
    """
    Remove zero-layer pallets from the floor, optionally repurposing as donor builds.

    Empty pick-face pallets exit immediately unless an empty donor pad needs a build
    for the same SKU (then the shell is restaged as ``BUILD-{donor_slot}``).
    """
    existing_ids = {p.pallet_id for p in pallets}
    kept: list[Pallet] = []
    for pallet in pallets:
        if pallet.layers_remaining > 0:
            kept.append(pallet)
            continue
        _clear_pallet_from_slots(pallet, slots)
        if (
            repurpose_as_donor
            and state is not None
            and _try_repurpose_as_donor_build(
                pallet,
                slots,
                state=state,
                existing_ids=existing_ids,
                colocated=colocated_donor_repurpose,
            )
        ):
            kept.append(pallet)
    scrub_stale_slot_refs(kept, slots)
    return kept


def finalize_depleted_pallets(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    state: WorldState | None = None,
    repurpose_as_donor: bool = True,
    colocated_donor_repurpose: bool = True,
) -> list[Pallet]:
    """Alias for :func:`dispose_depleted_pallets`."""
    return dispose_depleted_pallets(
        pallets,
        slots,
        state=state,
        repurpose_as_donor=repurpose_as_donor,
        colocated_donor_repurpose=colocated_donor_repurpose,
    )


def prune_depleted_pallets(pallets: list[Pallet]) -> list[Pallet]:
    """Drop empty pallets from the reserve pool."""
    return [p for p in pallets if p.layers_remaining > 0]
