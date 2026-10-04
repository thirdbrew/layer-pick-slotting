"""Reserve discipline, JIT staging, and pending-completion pallet priority."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from collections.abc import Mapping

from slotting.inventory import WAVED_HORIZON, ship_day_layers_for_item
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, Slot, SlotType, WorldState
from slotting.wave_release import order_ship_date


def mark_removed_from_pick(
    pallet: Pallet,
    *,
    as_of: datetime,
    source_slot_id: str | None = None,
) -> None:
    """Track inventory for a pallet pulled off an active pick face."""
    if pallet.layers_remaining <= 0:
        return
    pallet.removed_from_pick_at = as_of
    if source_slot_id:
        pallet.last_source_slot_id = source_slot_id


def clear_pending_completion(pallet: Pallet) -> None:
    """Clear pending flag when pallet returns to an active pick face."""
    pallet.removed_from_pick_at = None


def is_pending_completion(pallet: Pallet) -> bool:
    return pallet.removed_from_pick_at is not None and pallet.layers_remaining > 0


def supply_pallet_sort_key(pallet: Pallet) -> tuple[int, int, datetime]:
    """Pending partial pallets first, then lowest layers (finish-to-empty)."""
    pending_rank = 0 if is_pending_completion(pallet) else 1
    return (pending_rank, pallet.layers_remaining, pallet.arrived_at)


def select_best_supply_pallet(candidates: list[Pallet]) -> Pallet | None:
    if not candidates:
        return None
    return min(candidates, key=supply_pallet_sort_key)


def pending_completion_layers(pallets: list[Pallet]) -> int:
    return sum(p.layers_remaining for p in pallets if is_pending_completion(p))


def pending_completion_count(pallets: list[Pallet]) -> int:
    return sum(1 for p in pallets if is_pending_completion(p))


def next_ship_day_after(orders: list[OrderLine], ship_date: date) -> date | None:
    """Next calendar ship day with THIS_LINE demand after *ship_date*."""
    from slotting.models import OrderArea

    days = sorted(
        {
            order_ship_date(o)
            for o in orders
            if o.area == OrderArea.THIS_LINE and order_ship_date(o) > ship_date
        }
    )
    return days[0] if days else None


def _item_has_ship_demand(
    orders: list[OrderLine],
    item_id: str,
    ship_date,
) -> bool:
    return ship_day_layers_for_item(orders, item_id, ship_date) > 0


def _item_has_near_demand(state: WorldState, item_id: str, *, horizon: timedelta) -> bool:
    cutoff = state.as_of + horizon
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.item_id != item_id:
            continue
        if order.status not in (OrderStatus.WAVED, OrderStatus.VISIBLE):
            continue
        if state.as_of <= order.due_at <= cutoff:
            return True
    return False


def demand_weighted_pallets_per_sku(
    orders: list[OrderLine],
    item_id: str,
    ship_date,
    *,
    default_layers: int = 8,
    min_per_sku: int = 1,
    max_per_sku: int = 4,
    spike_multiplier: float = 1.5,
    is_spike_day: bool = False,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
) -> int:
    """Pallet count to stage from ship-day layer demand (not a flat cap per SKU)."""
    layers = ship_day_layers_for_item(orders, item_id, ship_date)
    if layers <= 0:
        return max(1, min_per_sku)
    needed = math.ceil(layers / max(1, default_layers))
    if is_spike_day:
        needed = math.ceil(needed * spike_multiplier)
    from datetime import time

    late_start = datetime.combine(ship_date, time(late_shift_hour))
    day_end = datetime.combine(ship_date, time(final_shift_hour))
    late_layers = sum(
        o.qty_layers
        for o in orders
        if o.item_id == item_id
        and o.area == OrderArea.THIS_LINE
        and late_start <= o.due_at <= day_end
    )
    if late_layers > 0:
        needed = max(needed, math.ceil(late_layers / max(1, default_layers)))
    return max(min_per_sku, min(needed, max_per_sku))


def _max_per_sku_for_item(
    item_id: str,
    *,
    max_per_sku: int,
    demand_weighted: bool,
    orders: list[OrderLine] | None,
    ship_date,
    default_layers: int,
    demand_weighted_max: int,
    is_spike_day: bool,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
) -> int:
    if demand_weighted and orders is not None and ship_date is not None:
        return demand_weighted_pallets_per_sku(
            orders,
            item_id,
            ship_date,
            default_layers=default_layers,
            min_per_sku=max_per_sku,
            max_per_sku=demand_weighted_max,
            is_spike_day=is_spike_day,
            late_shift_hour=late_shift_hour,
            final_shift_hour=final_shift_hour,
        )
    return max(1, max_per_sku)


def apply_reserve_layer_budget(
    pallets: list[Pallet],
    *,
    max_unslotted_layers: int,
    keep_pending: bool = True,
) -> list[Pallet]:
    """Cap total unslotted reserve layers; pending-completion pallets are exempt."""
    if max_unslotted_layers <= 0:
        return pallets

    slotted = [p for p in pallets if p.current_slot or p.layers_remaining <= 0]
    unslotted = [p for p in pallets if not p.current_slot and p.layers_remaining > 0]
    pending = [p for p in unslotted if keep_pending and is_pending_completion(p)]
    droppable = [p for p in unslotted if p not in pending]

    pending_layers = sum(p.layers_remaining for p in pending)
    budget = max(0, max_unslotted_layers - pending_layers)

    kept: list[Pallet] = []
    used = 0
    for pallet in sorted(droppable, key=supply_pallet_sort_key):
        if used + pallet.layers_remaining <= budget:
            kept.append(pallet)
            used += pallet.layers_remaining

    return slotted + pending + kept


def prune_orphan_reserve(
    pallets: list[Pallet],
    orders: list[OrderLine],
    *,
    ship_date,
    as_of: datetime,
    max_per_sku: int = 1,
    keep_pending: bool = True,
    demand_weighted: bool = False,
    default_layers: int = 8,
    demand_weighted_max: int = 4,
    is_spike_day: bool = False,
    max_unslotted_layers: int = 0,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
) -> list[Pallet]:
    """
    Drop unslotted reserve pallets with no ship-day demand.

    Pending-completion partial pallets are kept until depleted.
    """
    by_item: dict[str, list[Pallet]] = {}
    kept_slotted: list[Pallet] = []
    for pallet in pallets:
        if pallet.current_slot:
            kept_slotted.append(pallet)
            continue
        if pallet.layers_remaining <= 0:
            continue
        if keep_pending and is_pending_completion(pallet):
            kept_slotted.append(pallet)
            continue
        if ship_date is not None and not _item_has_ship_demand(orders, pallet.item_id, ship_date):
            continue
        by_item.setdefault(pallet.item_id, []).append(pallet)

    reserve: list[Pallet] = []
    for item_id, pool in by_item.items():
        pool.sort(key=supply_pallet_sort_key)
        cap = _max_per_sku_for_item(
            item_id,
            max_per_sku=max_per_sku,
            demand_weighted=demand_weighted,
            orders=orders,
            ship_date=ship_date,
            default_layers=default_layers,
            demand_weighted_max=demand_weighted_max,
            is_spike_day=is_spike_day,
            late_shift_hour=late_shift_hour,
            final_shift_hour=final_shift_hour,
        )
        reserve.extend(pool[:cap])

    result = kept_slotted + reserve
    if max_unslotted_layers > 0:
        result = apply_reserve_layer_budget(result, max_unslotted_layers=max_unslotted_layers)
    return result


def jit_rack_item_ids(
    ranked_item_ids: list[str],
    state: WorldState,
    *,
    source_face_count: int,
    jit_horizon_hours: int = 8,
) -> set[str]:
    """
    SKUs that should occupy rack at setup — overflow ranks with near-term demand only.
    """
    overflow = ranked_item_ids[source_face_count:]
    horizon = timedelta(hours=jit_horizon_hours)
    return {
        item_id
        for item_id in overflow
        if _item_has_near_demand(state, item_id, horizon=horizon)
    }


def resupply_priority_key(
    state: WorldState,
    item_id: str,
    *,
    waved_layers: dict[str, int],
) -> tuple[int, int, int, int, str]:
    """Sort resupply: pending reserve layers, lowest LP-pickable face, highest waved demand."""
    from slotting.pallet_geometry import face_lp_pickable_layers_for_item

    pending = sum(
        p.layers_remaining
        for p in state.pallets
        if p.item_id == item_id and is_pending_completion(p) and not p.current_slot
    )
    face = face_lp_pickable_layers_for_item(state, item_id)
    demand = waved_layers.get(item_id, 0)
    zero_face = 0 if face == 0 else 1
    return (-pending, zero_face, face, -demand, item_id)


def future_same_day_layers_by_item(
    state: WorldState,
    *,
    after: datetime | None = None,
    final_shift_hour: int = 22,
) -> dict[str, int]:
    """
    Layer demand for orders due after *after* through end of pick day.

    Used to retain inventory for late waves not yet in the 4h waved window.
    """
    from datetime import time

    cutoff = after if after is not None else state.as_of + WAVED_HORIZON
    day_end = datetime.combine(state.as_of.date(), time(final_shift_hour))
    demand: dict[str, int] = {}
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.status not in (OrderStatus.WAVED, OrderStatus.VISIBLE):
            continue
        if not (cutoff < order.due_at <= day_end):
            continue
        demand[order.item_id] = demand.get(order.item_id, 0) + order.qty_layers
    return demand


def late_shift_layers_by_item(
    state: WorldState,
    *,
    late_shift_hour: int,
    final_shift_hour: int = 22,
) -> dict[str, int]:
    """Layers due at or after *late_shift_hour* on the pick day."""
    from datetime import time

    start = datetime.combine(state.as_of.date(), time(late_shift_hour))
    day_end = datetime.combine(state.as_of.date(), time(final_shift_hour))
    demand: dict[str, int] = {}
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.status not in (OrderStatus.WAVED, OrderStatus.VISIBLE):
            continue
        if not (start <= order.due_at <= day_end):
            continue
        demand[order.item_id] = demand.get(order.item_id, 0) + order.qty_layers
    return demand


def inventory_layers_by_item(state: WorldState) -> dict[str, int]:
    """Total layers on floor (face, rack, unslotted) per SKU."""
    totals: dict[str, int] = {}
    for pallet in state.pallets:
        if pallet.layers_remaining <= 0:
            continue
        totals[pallet.item_id] = totals.get(pallet.item_id, 0) + pallet.layers_remaining
    return totals


def cap_face_pick_for_late_reserve(
    state: WorldState,
    item_id: str,
    *,
    face_layers: int,
    take_requested: int,
    late_shift_hour: int,
    final_shift_hour: int = 22,
) -> int:
    """
    Limit face depletion so rack/unslotted stock can cover late-day waves.

    Before *late_shift_hour*, hold back face picks when future same-day demand
    exceeds off-face inventory.

    *face_layers* is LP-pickable depth on the face (may be less than total face
    layers when the clamp pick reach blocks upper layers).
    """
    if state.as_of.hour >= late_shift_hour or take_requested <= 0:
        return take_requested
    future = future_same_day_layers_by_item(
        state, final_shift_hour=final_shift_hour
    ).get(item_id, 0)
    if future <= 0:
        return take_requested
    from slotting.inventory import face_layers_for_item

    inv = inventory_layers_by_item(state).get(item_id, 0)
    total_face = face_layers_for_item(state, item_id)
    off_face = max(0, inv - total_face)
    reserve = max(0, future - off_face)
    max_from_face = max(0, face_layers - reserve)
    return min(take_requested, max_from_face)


def late_rack_promote_sort_key(
    state: WorldState,
    item_id: str,
    *,
    late_shift_hour: int,
    final_shift_hour: int = 22,
    near_term_hour: int = 14,
    near_term_hours: int = 4,
) -> tuple[int, int, int, int, str]:
    """Prioritize rack promotes: evening late-shift demand, then near-term waved layers.

    Face depth uses LP-pickable layers so height-blocked stock does not look
    healthier than it is when ranking who gets the next rack promote.
    """
    from slotting.pallet_geometry import face_lp_pickable_layers_for_item

    late = late_shift_layers_by_item(
        state, late_shift_hour=late_shift_hour, final_shift_hour=final_shift_hour
    ).get(item_id, 0)
    near = near_term_waved_layers_by_item(state, hours=near_term_hours).get(item_id, 0)
    face = face_lp_pickable_layers_for_item(state, item_id)
    if state.as_of.hour >= late_shift_hour:
        return (-late, -near, face, 0, item_id)
    if state.as_of.hour >= near_term_hour:
        return (-near, -late, face, 0, item_id)
    return (0, face, -late, -near, item_id)


def rack_promote_sort_key(
    state: WorldState,
    item_id: str,
    *,
    late_shift_hour: int,
    final_shift_hour: int = 22,
    near_term_hour: int = 14,
    near_term_hours: int = 4,
    preferred_face_by_item: Mapping[str, str] | None = None,
    face_station_by_id: Mapping[str, int] | None = None,
    face_flank_by_id: Mapping[str, str] | None = None,
    flank_local_batching: bool = False,
    morning_batch_end_hour: int = 11,
) -> tuple:
    """Sort rack promotes: demand first, optional flank-local path as tiebreak.

    Layer-picker readiness wins — a hotter SKU is never delayed to shorten the
    assistant's aisle walk. When *flank_local_batching* is on, equal-demand
    SKUs are ordered left→right by station (assist path batching only among
    ties). *morning_batch_end_hour* is retained for call-site compatibility;
    geometry tiebreak applies whenever batching is enabled.
    """
    demand_key = late_rack_promote_sort_key(
        state,
        item_id,
        late_shift_hour=late_shift_hour,
        final_shift_hour=final_shift_hour,
        near_term_hour=near_term_hour,
        near_term_hours=near_term_hours,
    )
    if not flank_local_batching:
        return demand_key
    _ = morning_batch_end_hour  # API compat; geometry tiebreak is all-day

    from slotting.resupply_rack import _entry_colocation_hints

    preferred = preferred_face_by_item or {}
    stations = face_station_by_id or {}
    flanks = face_flank_by_id or {}
    _, entry_station, entry_flank = _entry_colocation_hints(
        item_id,
        preferred_face_by_item=preferred,
        face_station_by_id=stations,
        face_flank_by_id=flanks,
    )
    flank_ord = 2
    flank_norm = (entry_flank or "").upper()
    if flank_norm in ("L", "LEFT"):
        flank_ord = 0
    elif flank_norm in ("R", "RIGHT"):
        flank_ord = 1
    station = entry_station if entry_station is not None else 999
    return (*demand_key, flank_ord, station)


def near_term_waved_layers_by_item(
    state: WorldState,
    *,
    hours: int = 4,
) -> dict[str, int]:
    """Waved layer demand due within *hours* of *state.as_of*."""
    cutoff = state.as_of + timedelta(hours=hours)
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


def maximize_setup_rack_fill(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    setup_ranks: dict[str, int],
    rack_by_item: dict[str, str],
    as_of: datetime,
) -> int:
    """
    Fill empty rack positions at setup — planned overflow slots first, then spare bays.

    Returns count of pallets staged onto racks.
    """
    from slotting.resupply_rack import pick_rack_slot

    slot_by_id = {s.slot_id: s for s in slots}
    staged = 0

    def _stage(pallet: Pallet, rack: Slot) -> None:
        nonlocal staged
        clear_pending_completion(pallet)
        rack.occupied_by_pallet_id = pallet.pallet_id
        pallet.current_slot = rack.slot_id
        staged += 1

    for item_id in sorted(rack_by_item, key=lambda i: (setup_ranks.get(i, 10**9), i)):
        rack_id = rack_by_item[item_id]
        rack = slot_by_id.get(rack_id)
        if rack is None or rack.occupied_by_pallet_id:
            continue
        pool = [
            p
            for p in pallets
            if p.item_id == item_id and not p.current_slot and p.layers_remaining > 0
        ]
        if not pool:
            continue
        _stage(min(pool, key=supply_pallet_sort_key), rack)

    state = WorldState(pallets=pallets, slots=slots, orders=[], exclusions=[], as_of=as_of)
    unslotted = sorted(
        [p for p in pallets if not p.current_slot and p.layers_remaining > 0],
        key=lambda p: (setup_ranks.get(p.item_id, 10**9), *supply_pallet_sort_key(p)),
    )
    spare_ids = sorted(
        s.slot_id
        for s in slots
        if s.slot_type == SlotType.RESUPPLY_RACK and not s.occupied_by_pallet_id
    )
    for rack_id in spare_ids:
        if not unslotted:
            break
        rack = slot_by_id[rack_id]
        pick_idx = 0
        for i, pallet in enumerate(unslotted):
            target = pick_rack_slot(state, item_id=pallet.item_id)
            if target is not None and target.slot_id == rack_id:
                pick_idx = i
                break
        pallet = unslotted.pop(pick_idx)
        _stage(pallet, rack)
        state = WorldState(pallets=pallets, slots=slots, orders=[], exclusions=[], as_of=as_of)
    return staged


def early_wave_priority_item_ids(
    orders: list[OrderLine],
    ship_date: date,
    *,
    as_of: datetime,
    horizon_hours: int = 4,
    pick_start_hour: int = 6,
    due_offset_hours: float = 4.0,
    top_n: int = 40,
    overnight_only: bool = False,
) -> list[str]:
    """SKUs with overnight or early-morning waved demand on *ship_date* (carry setup)."""
    from datetime import time

    from slotting.visibility_planning import rank_ship_day_setup_skus

    ranked = rank_ship_day_setup_skus(
        orders,
        ship_date,
        setup_as_of=as_of,
        pick_start_hour=pick_start_hour,
        early_wave_hours=horizon_hours,
        due_offset_hours=due_offset_hours,
    )
    day_start = datetime.combine(ship_date, time(pick_start_hour))
    early_cutoff = day_start + timedelta(hours=horizon_hours)
    picked: list[str] = []
    for item_id, _layers, due in ranked:
        if due.date() != ship_date:
            continue
        overnight = due < day_start
        early_morning = day_start <= due <= early_cutoff
        if overnight_only and not overnight:
            continue
        if not overnight_only and not (overnight or early_morning):
            continue
        picked.append(item_id)
        if len(picked) >= top_n:
            break
    return picked


def early_wave_as_of_priority_item_ids(
    orders: list[OrderLine],
    ship_date: date,
    *,
    as_of: datetime,
    horizon_hours: int = 4,
    pick_start_hour: int = 6,
    top_n: int = 40,
    overnight_only: bool = False,
) -> list[str]:
    """
    Overnight/early SKUs on *ship_date* ranked from the *as_of* visibility window.

    Unlike ``early_wave_priority_item_ids``, does not scan the full next-day
    order book — only dues already inside the 24h visible horizon at prep time.
    """
    from datetime import time

    from slotting.visibility_planning import rank_setup_skus

    ranked = rank_setup_skus(orders, as_of)
    day_start = datetime.combine(ship_date, time(pick_start_hour))
    early_cutoff = day_start + timedelta(hours=horizon_hours)
    picked: list[str] = []
    for item_id, _layers, due in ranked:
        if due.date() != ship_date:
            continue
        overnight = due < day_start
        early_morning = day_start <= due <= early_cutoff
        if overnight_only and not overnight:
            continue
        if not overnight_only and not (overnight or early_morning):
            continue
        picked.append(item_id)
        if len(picked) >= top_n:
            break
    return picked


def stage_carry_early_wave_unslotted_to_rack(
    pallets: list[Pallet],
    slots: list[Slot],
    orders: list[OrderLine],
    ship_date: date,
    *,
    as_of: datetime,
    horizon_hours: int = 4,
    pick_start_hour: int = 6,
    due_offset_hours: float = 4.0,
    top_n: int = 40,
    overnight_only: bool = False,
) -> int:
    """
    At carry setup, stage unslotted reserve for overnight/early-wave SKUs onto racks.

    Runs after generic bootstrap; prioritizes first-wave demand over setup rank alone.
    """
    from slotting.resupply_rack import pick_rack_slot

    priority = early_wave_priority_item_ids(
        orders,
        ship_date,
        as_of=as_of,
        horizon_hours=horizon_hours,
        pick_start_hour=pick_start_hour,
        due_offset_hours=due_offset_hours,
        top_n=top_n,
        overnight_only=overnight_only,
    )
    if not priority:
        return 0

    staged = 0
    state = WorldState(pallets=pallets, slots=slots, orders=orders, exclusions=[], as_of=as_of)
    for item_id in priority:
        pool = [
            p
            for p in pallets
            if p.item_id == item_id and not p.current_slot and p.layers_remaining > 0
        ]
        if not pool:
            continue
        pallet = min(pool, key=supply_pallet_sort_key)
        rack = pick_rack_slot(state, item_id=item_id)
        if rack is None or rack.occupied_by_pallet_id:
            continue
        clear_pending_completion(pallet)
        rack.occupied_by_pallet_id = pallet.pallet_id
        pallet.current_slot = rack.slot_id
        staged += 1
        state = WorldState(pallets=pallets, slots=slots, orders=orders, exclusions=[], as_of=as_of)
    return staged


def bootstrap_unslotted_to_rack(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    rack_by_item: dict[str, str],
    setup_ranks: dict[str, int],
    as_of: datetime,
) -> None:
    """
    Slot unslotted carried reserve into empty rack positions at setup.

    Planned rack slots are filled first (by setup rank), then any remaining
    unslotted pallets take spare rack capacity (pending-completion first).
    """
    slot_by_id = {s.slot_id: s for s in slots}
    unslotted = [
        p
        for p in pallets
        if not p.current_slot and p.layers_remaining > 0
    ]
    if not unslotted:
        return

    by_item: dict[str, list[Pallet]] = {}
    for pallet in unslotted:
        by_item.setdefault(pallet.item_id, []).append(pallet)
    for pool in by_item.values():
        pool.sort(key=supply_pallet_sort_key)

    def _take(item_id: str) -> Pallet | None:
        pool = by_item.get(item_id, [])
        while pool:
            pallet = pool.pop(0)
            if pallet.current_slot:
                continue
            if not pool:
                by_item.pop(item_id, None)
            return pallet
        return None

    def _assign(pallet: Pallet, rack: Slot) -> None:
        clear_pending_completion(pallet)
        rack.occupied_by_pallet_id = pallet.pallet_id
        pallet.current_slot = rack.slot_id

    ranked_items = sorted(
        rack_by_item.keys(),
        key=lambda item_id: (setup_ranks.get(item_id, 10**9), item_id),
    )
    for item_id in ranked_items:
        rack_id = rack_by_item[item_id]
        rack = slot_by_id.get(rack_id)
        if rack is None or rack.slot_type != SlotType.RESUPPLY_RACK:
            continue
        if rack.occupied_by_pallet_id:
            continue
        pallet = _take(item_id)
        if pallet is None:
            continue
        _assign(pallet, rack)

    spare_racks = sorted(
        s.slot_id
        for s in slots
        if s.slot_type == SlotType.RESUPPLY_RACK and not s.occupied_by_pallet_id
    )
    remaining = sorted(
        (p for pool in by_item.values() for p in pool if not p.current_slot),
        key=supply_pallet_sort_key,
    )
    for rack_id in spare_racks:
        if not remaining:
            break
        rack = slot_by_id[rack_id]
        if rack.occupied_by_pallet_id:
            continue
        pallet = remaining.pop(0)
        _assign(pallet, rack)


def spike_reserve_keep_item_ids(
    orders: list[OrderLine],
    ship_date,
    *,
    setup_rank_by_item: dict[str, int],
    staging_boost: dict[str, int] | None = None,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
) -> set[str]:
    """SKUs to retain in reserve when rebootstraping carry inventory on a spike day."""
    keep = set(setup_rank_by_item.keys())
    if staging_boost:
        keep |= set(staging_boost.keys())
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        keep.add(order.item_id)
    return keep


def drop_unslotted_carry_not_in_plan(
    pallets: list[Pallet],
    keep_item_ids: set[str],
) -> list[Pallet]:
    """Remove unslotted carry pallets for SKUs outside the spike-day keep set."""
    kept: list[Pallet] = []
    for pallet in pallets:
        if pallet.layers_remaining <= 0:
            continue
        if pallet.current_slot:
            kept.append(pallet)
            continue
        if is_pending_completion(pallet):
            kept.append(pallet)
            continue
        if pallet.item_id in keep_item_ids:
            kept.append(pallet)
    return kept


def _late_shift_layers_for_item(
    orders: list[OrderLine],
    item_id: str,
    ship_date,
    *,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
) -> int:
    from datetime import time

    start = datetime.combine(ship_date, time(late_shift_hour))
    day_end = datetime.combine(ship_date, time(final_shift_hour))
    total = 0
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.item_id != item_id:
            continue
        if order.status not in (OrderStatus.WAVED, OrderStatus.VISIBLE):
            continue
        if start <= order.due_at <= day_end:
            total += order.qty_layers
    return total


def prune_carry_stale_unslotted(
    pallets: list[Pallet],
    orders: list[OrderLine],
    *,
    ship_date,
    setup_rank_by_item: dict[str, int],
    max_rank_keep: int,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
    also_keep_ship_dates: tuple[date, ...] = (),
) -> list[Pallet]:
    """
    Drop carried unslotted pallets for SKUs outside today's setup plan with no demand.

    Keeps slotted pallets, pending-completion partials, and unslotted stock for
    SKUs ranked within *max_rank_keep* or with ship-day / late-shift demand.
    """
    kept: list[Pallet] = []
    for pallet in pallets:
        if pallet.current_slot or pallet.layers_remaining <= 0:
            kept.append(pallet)
            continue
        if is_pending_completion(pallet):
            kept.append(pallet)
            continue
        rank = setup_rank_by_item.get(pallet.item_id, 10**9)
        if rank <= max_rank_keep:
            kept.append(pallet)
            continue
        if ship_date is not None and _item_has_ship_demand(orders, pallet.item_id, ship_date):
            kept.append(pallet)
            continue
        if any(
            _item_has_ship_demand(orders, pallet.item_id, extra_day)
            for extra_day in also_keep_ship_dates
        ):
            kept.append(pallet)
            continue
        late = _late_shift_layers_for_item(
            orders,
            pallet.item_id,
            ship_date,
            late_shift_hour=late_shift_hour,
            final_shift_hour=final_shift_hour,
        )
        if late > 0:
            kept.append(pallet)
    return kept


def build_next_ship_day_staging_boost(
    orders: list[OrderLine],
    ship_date: date,
    *,
    top_n: int = 25,
    early_wave_hours: int = 8,
    pick_start_hour: int = 6,
    due_offset_hours: float = 4.0,
) -> tuple[date | None, dict[str, int]]:
    """
    Evening carry prep: +1 pallet for next ship day's early-wave SKUs.

    Returns ``(next_ship_day, boost_by_item)``; empty when no next day demand.
    """
    from datetime import time

    from slotting.visibility_planning import final_shift_as_of, rank_ship_day_setup_skus

    next_day = next_ship_day_after(orders, ship_date)
    if next_day is None or top_n <= 0:
        return None, {}
    ranked = rank_ship_day_setup_skus(
        orders,
        next_day,
        setup_as_of=final_shift_as_of(next_day),
        pick_start_hour=pick_start_hour,
        early_wave_hours=early_wave_hours,
        due_offset_hours=due_offset_hours,
    )
    day_start = datetime.combine(next_day, time(pick_start_hour))
    early_cutoff = day_start + timedelta(hours=early_wave_hours)
    boost: dict[str, int] = {}
    for item_id, _layers, due in ranked:
        if due > early_cutoff:
            continue
        boost[item_id] = 1
        if len(boost) >= top_n:
            break
    return next_day, boost


def build_next_morning_wave_forecast_boost(
    orders: list[OrderLine],
    ship_date: date,
    *,
    horizon_hours: int = 4,
    pick_start_hour: int = 6,
    due_offset_hours: float = 4.0,
    top_n: int = 25,
    overnight_only: bool = False,
) -> tuple[date | None, dict[str, int]]:
    """
    Evening carry forecast: mint reserve for next ship day's overnight/early waves.

    Uses order visibility at next-day setup (not live waved shortage at 17:00).
    Looks ahead at the full next ship-day book via ``rank_ship_day_setup_skus``.
    """
    next_day = next_ship_day_after(orders, ship_date)
    if next_day is None or top_n <= 0:
        return None, {}
    from slotting.visibility_planning import final_shift_as_of

    items = early_wave_priority_item_ids(
        orders,
        next_day,
        as_of=final_shift_as_of(next_day),
        horizon_hours=horizon_hours,
        pick_start_hour=pick_start_hour,
        due_offset_hours=due_offset_hours,
        top_n=top_n,
        overnight_only=overnight_only,
    )
    return next_day, dict.fromkeys(items, 1)


def build_next_morning_ols_forecast_boost(
    orders: list[OrderLine],
    ship_date: date,
    *,
    as_of: datetime,
    horizon_hours: int = 4,
    pick_start_hour: int = 6,
    top_n: int = 25,
    overnight_only: bool = False,
    min_train_days: int = 14,
    early_sku_fraction: float = 0.25,
) -> tuple[date | None, dict[str, int]]:
    """
    Honesty-mode evening mint: OLS sizes the count; SKUs come from as-of visibility.

    Trains on ship days strictly before the next ship day (no tomorrow leakage).
    SKU identity uses ``rank_setup_skus`` at *as_of*, not the full next-day book.
    Falls back to the as-of top_n list when the OLS fit lacks enough history.
    """
    import math

    from layout_advisor.forecast.demand_model import build_day_observations, fit_demand_forecast
    from slotting.wave_release import order_ship_date

    next_day = next_ship_day_after(orders, ship_date)
    if next_day is None or top_n <= 0:
        return None, {}

    train_orders = [o for o in orders if order_ship_date(o) < next_day]
    n = top_n
    try:
        model = fit_demand_forecast(train_orders, min_days=min_train_days)
        obs = build_day_observations(train_orders)
        lag1 = float(obs[-1].day_layers) if obs else None
        lag7 = float(obs[-8].day_layers) if len(obs) >= 8 else lag1
        pred = model.predict(
            next_day,
            lag_day_layers=lag1,
            lag7_day_layers=lag7,
        )
        frac = max(0.05, min(1.0, float(early_sku_fraction)))
        n = min(top_n, max(1, int(math.ceil(pred.sku_count * frac))))
    except ValueError:
        n = top_n

    items = early_wave_as_of_priority_item_ids(
        orders,
        next_day,
        as_of=as_of,
        horizon_hours=horizon_hours,
        pick_start_hour=pick_start_hour,
        top_n=n,
        overnight_only=overnight_only,
    )
    return next_day, dict.fromkeys(items, 1)


def build_dynamic_evening_shortage_boosts(
    state: WorldState,
    orders: list[OrderLine],
    ship_date: date,
    *,
    top_n: int = 30,
    min_waved_layers: int = 2,
    default_layers: int = 8,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
) -> dict[date, dict[str, int]]:
    """
    Shortage-driven 17:00 carry prep from live floor state.

    Returns ``mint_ship_date -> {item_id: extra_pallets}`` for today and next ship day.
    """
    from slotting.shortage_analysis import ShortageKind, analyze_waved_shortage

    shortage_kinds = {
        ShortageKind.TRUE_SHORTAGE,
        ShortageKind.UNSTAGED,
        ShortageKind.RACK_ONLY,
    }
    picks: list[tuple[int, date, str, int]] = []
    ship_dates = [ship_date]
    next_day = next_ship_day_after(orders, ship_date)
    if next_day is not None:
        ship_dates.append(next_day)

    for target_day in ship_dates:
        rows = analyze_waved_shortage(
            state,
            ship_date=target_day,
            late_shift_hour=late_shift_hour,
            final_shift_hour=final_shift_hour,
        )
        for row in rows:
            if row.kind not in shortage_kinds:
                continue
            if row.late_shift_layers < 4 and row.waved_layers < min_waved_layers:
                continue
            if row.total_inventory >= row.waved_layers and row.kind != ShortageKind.RACK_ONLY:
                continue
            gap = max(row.shortage_layers, row.waved_layers - row.total_inventory)
            pallets = max(1, (gap + default_layers - 1) // default_layers)
            priority = row.late_shift_layers * 2 + row.waved_layers
            picks.append((priority, target_day, row.item_id, pallets))

    picks.sort(key=lambda row: (-row[0], row[2]))
    boosts: dict[date, dict[str, int]] = {}
    seen: set[tuple[date, str]] = set()
    for _priority, target_day, item_id, pallets in picks[:top_n]:
        key = (target_day, item_id)
        if key in seen:
            continue
        seen.add(key)
        boosts.setdefault(target_day, {})[item_id] = pallets
    return boosts


def stage_items_to_rack(
    pallets: list[Pallet],
    slots: list[Slot],
    item_ids: set[str],
    *,
    as_of: datetime,
) -> int:
    """Move unslotted reserve for *item_ids* onto spare rack bays."""
    from slotting.resupply_rack import pick_rack_slot

    if not item_ids:
        return 0
    state = WorldState(pallets=pallets, slots=slots, orders=[], exclusions=[], as_of=as_of)
    staged = 0
    for item_id in sorted(item_ids):
        pool = [
            p
            for p in pallets
            if p.item_id == item_id and not p.current_slot and p.layers_remaining > 0
        ]
        if not pool:
            continue
        pallet = min(pool, key=supply_pallet_sort_key)
        rack = pick_rack_slot(state, item_id=item_id)
        if rack is None or rack.occupied_by_pallet_id:
            continue
        clear_pending_completion(pallet)
        rack.occupied_by_pallet_id = pallet.pallet_id
        pallet.current_slot = rack.slot_id
        staged += 1
        state = WorldState(pallets=pallets, slots=slots, orders=[], exclusions=[], as_of=as_of)
    return staged


def apply_dynamic_evening_carry_prep(
    pallets: list[Pallet],
    slots: list[Slot],
    orders: list[OrderLine],
    ship_date: date,
    *,
    as_of: datetime,
    top_n: int = 30,
    default_layers: int = 8,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
    exclusions: list | None = None,
    next_morning_forecast: bool = False,
    next_morning_ols_forecast: bool = False,
    forecast_horizon_hours: int = 4,
    forecast_top_n: int = 25,
    forecast_overnight_only: bool = False,
    pick_start_hour: int = 6,
    due_offset_hours: float = 4.0,
    ols_early_sku_fraction: float = 0.25,
) -> tuple[list[Pallet], set[str]]:
    """Mint shortage-driven staging pallets from live floor state at prep hour.

    Next-morning mint modes (mutually preferred: OLS wins when both True):

    * ``next_morning_ols_forecast`` — OLS-sized count + as-of-visible SKUs
    * ``next_morning_forecast`` — foresight early-wave SKUs from full next-day book
    """
    from slotting.backtest import orders_for_as_of
    from slotting.inventory import append_staging_pick_boost

    step_orders = orders_for_as_of(orders, as_of)
    state = WorldState(
        pallets=pallets,
        slots=slots,
        orders=step_orders,
        exclusions=list(exclusions or []),
        as_of=as_of,
    )
    boosts_by_day = build_dynamic_evening_shortage_boosts(
        state,
        orders,
        ship_date,
        top_n=top_n,
        default_layers=default_layers,
        late_shift_hour=late_shift_hour,
        final_shift_hour=final_shift_hour,
    )
    forecast_day: date | None = None
    forecast: dict[str, int] = {}
    if next_morning_ols_forecast:
        forecast_day, forecast = build_next_morning_ols_forecast_boost(
            orders,
            ship_date,
            as_of=as_of,
            horizon_hours=forecast_horizon_hours,
            pick_start_hour=pick_start_hour,
            top_n=forecast_top_n,
            overnight_only=forecast_overnight_only,
            early_sku_fraction=ols_early_sku_fraction,
        )
    elif next_morning_forecast:
        forecast_day, forecast = build_next_morning_wave_forecast_boost(
            orders,
            ship_date,
            horizon_hours=forecast_horizon_hours,
            pick_start_hour=pick_start_hour,
            due_offset_hours=due_offset_hours,
            top_n=forecast_top_n,
            overnight_only=forecast_overnight_only,
        )
    if forecast_day is not None and forecast:
        day_boost = dict(boosts_by_day.get(forecast_day, {}))
        for item_id, pallets_needed in forecast.items():
            day_boost[item_id] = max(day_boost.get(item_id, 0), pallets_needed)
        boosts_by_day[forecast_day] = day_boost
    boosted_items: set[str] = set()
    for mint_day, boost in boosts_by_day.items():
        if not boost:
            continue
        boosted_items |= set(boost.keys())
        pallets = append_staging_pick_boost(
            pallets,
            orders,
            mint_day,
            boost_by_item=boost,
            as_of=as_of,
            default_layers=default_layers,
            prefix="DYN",
        )
    return pallets, boosted_items


def stage_evening_next_day_to_rack(
    pallets: list[Pallet],
    slots: list[Slot],
    orders: list[OrderLine],
    ship_date: date,
    *,
    as_of: datetime,
    top_n: int = 25,
    early_wave_hours: int = 8,
    pick_start_hour: int = 6,
    due_offset_hours: float = 4.0,
) -> int:
    """Slot next-ship-day unslotted reserve onto spare rack bays before evening waves."""
    from slotting.resupply_rack import pick_rack_slot

    _next_day, boost = build_next_ship_day_staging_boost(
        orders,
        ship_date,
        top_n=top_n,
        early_wave_hours=early_wave_hours,
        pick_start_hour=pick_start_hour,
        due_offset_hours=due_offset_hours,
    )
    if not boost:
        return 0
    state = WorldState(pallets=pallets, slots=slots, orders=orders, exclusions=[], as_of=as_of)
    staged = 0
    for item_id in sorted(boost.keys()):
        pool = [
            p
            for p in pallets
            if p.item_id == item_id and not p.current_slot and p.layers_remaining > 0
        ]
        if not pool:
            continue
        pallet = min(pool, key=supply_pallet_sort_key)
        rack = pick_rack_slot(state, item_id=item_id)
        if rack is None or rack.occupied_by_pallet_id:
            continue
        clear_pending_completion(pallet)
        rack.occupied_by_pallet_id = pallet.pallet_id
        pallet.current_slot = rack.slot_id
        staged += 1
        state = WorldState(pallets=pallets, slots=slots, orders=orders, exclusions=[], as_of=as_of)
    return staged


def _release_slotted_pallet(
    pallet: Pallet,
    slot: Slot,
    *,
    as_of: datetime,
) -> None:
    if pallet.layers_remaining <= 0:
        slot.occupied_by_pallet_id = None
        return
    if slot.slot_type == SlotType.SOURCE:
        mark_removed_from_pick(pallet, as_of=as_of, source_slot_id=slot.slot_id)
    slot.occupied_by_pallet_id = None
    pallet.current_slot = None


def release_slotted_outside_keep_set(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    keep_item_ids: set[str],
    as_of: datetime,
) -> None:
    """
    Spike transition without a full floor flush: release slotted carry for SKUs
    outside today's spike keep set; retain in-plan stock on faces, racks, and donors.
    """
    pallet_by_id = {p.pallet_id: p for p in pallets}
    for slot in slots:
        if slot.backup_pallet_id:
            backup = pallet_by_id.get(slot.backup_pallet_id)
            if backup is not None and backup.item_id not in keep_item_ids:
                backup.current_slot = None
                slot.backup_pallet_id = None
        pid = slot.occupied_by_pallet_id
        if not pid:
            continue
        pallet = pallet_by_id.get(pid)
        if pallet is None:
            slot.occupied_by_pallet_id = None
            continue
        if pallet.item_id in keep_item_ids:
            continue
        _release_slotted_pallet(pallet, slot, as_of=as_of)


def flush_carry_sunday_reset(
    pallets: list[Pallet],
    slots: list[Slot],
    *,
    as_of: datetime,
) -> None:
    """
    Legacy Sunday-night carry reset: return every slotted pallet to unslotted reserve.

    Prefer ``release_slotted_outside_keep_set`` + setup reslot for spike transitions.
    """
    pallet_by_id = {p.pallet_id: p for p in pallets}
    for slot in slots:
        if slot.backup_pallet_id:
            backup = pallet_by_id.get(slot.backup_pallet_id)
            if backup is not None:
                backup.current_slot = None
            slot.backup_pallet_id = None
        pid = slot.occupied_by_pallet_id
        if not pid:
            continue
        pallet = pallet_by_id.get(pid)
        if pallet is None or pallet.layers_remaining <= 0:
            slot.occupied_by_pallet_id = None
            continue
        _release_slotted_pallet(pallet, slot, as_of=as_of)
