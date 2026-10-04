"""Apply scheduled donor build opens to simulation state."""

from __future__ import annotations

from datetime import datetime, timedelta

from layout_advisor.planning.dynamic_slotting import WaveDonorOpen
from slotting.models import OrderArea, OrderStatus, Pallet, SlotType, WorldState
from slotting.rules import WAVED_HORIZON


def _waved_items_at(state: WorldState) -> set[str]:
    cutoff = state.as_of + WAVED_HORIZON
    return {
        o.item_id
        for o in state.orders
        if o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    }


def _empty_destination_slots(state: WorldState) -> list[str]:
    return sorted(
        s.slot_id
        for s in state.slots
        if s.slot_type == SlotType.DESTINATION and not s.occupied_by_pallet_id
    )


def _items_with_open_builds(state: WorldState) -> set[str]:
    """SKUs that already have an in-progress build pallet on a donor pad.

    Minimum-pallets rule: an order/SKU may span several build pallets over the
    day, but never two OPEN ones at once — a later open defers until the
    earlier build closes."""
    pallet_by_id = {p.pallet_id: p for p in state.pallets}
    out: set[str] = set()
    for slot in state.slots:
        if slot.slot_type != SlotType.DESTINATION or not slot.occupied_by_pallet_id:
            continue
        pallet = pallet_by_id.get(slot.occupied_by_pallet_id)
        if pallet is not None:
            out |= pallet.destination_skus()
    return out


def apply_donor_preopen(
    state: WorldState,
    preopen: list[tuple[str, int, str]],
) -> tuple[WorldState, list[tuple[str, int, str]]]:
    """Stage donor build pallets on empty destination pads.

    Returns ``(new_state, staged)`` where *staged* lists tuples actually placed.
    """
    if not preopen:
        return state, []
    pallets = [p.model_copy(deep=True) for p in state.pallets]
    slots = [s.model_copy(deep=True) for s in state.slots]
    slot_by_id = {s.slot_id: s for s in slots}
    existing_builds = {p.pallet_id for p in pallets}
    staged: list[tuple[str, int, str]] = []

    for item_id, layers, donor_slot in preopen:
        slot = slot_by_id.get(donor_slot)
        if slot is None or slot.slot_type != SlotType.DESTINATION:
            continue
        if slot.occupied_by_pallet_id:
            continue
        build_id = f"BUILD-{donor_slot}"
        if build_id in existing_builds:
            stale = next((p for p in pallets if p.pallet_id == build_id), None)
            if stale is not None and stale.current_slot is None:
                pallets = [p for p in pallets if p.pallet_id != build_id]
                existing_builds.discard(build_id)
            else:
                continue
        build_pallet = Pallet(
            pallet_id=build_id,
            item_id=item_id,
            layers_remaining=max(layers, 1),
            current_slot=donor_slot,
            arrived_at=state.as_of,
        )
        pallets.append(build_pallet)
        slot.occupied_by_pallet_id = build_id
        existing_builds.add(build_id)
        staged.append((item_id, layers, donor_slot))

    if not staged:
        return state, []
    return (
        WorldState(
            pallets=pallets,
            slots=slots,
            orders=state.orders,
            exclusions=state.exclusions,
            as_of=state.as_of,
        ),
        staged,
    )


def apply_multi_sku_donor_preopen(
    state: WorldState,
    donor_slot: str,
    build_lines: dict[str, int],
) -> WorldState:
    """Stage one physical build pallet with multiple SKU lines on a donor pad."""
    if not build_lines:
        return state
    if len(build_lines) == 1:
        item_id, layers = next(iter(build_lines.items()))
        new_state, _ = apply_donor_preopen(state, [(item_id, layers, donor_slot)])
        return new_state

    pallets = [p.model_copy(deep=True) for p in state.pallets]
    slots = [s.model_copy(deep=True) for s in state.slots]
    slot_by_id = {s.slot_id: s for s in slots}
    slot = slot_by_id.get(donor_slot)
    if slot is None or slot.slot_type != SlotType.DESTINATION or slot.occupied_by_pallet_id:
        return state

    build_id = f"BUILD-{donor_slot}"
    if any(p.pallet_id == build_id for p in pallets):
        return state

    total_layers = sum(max(layers, 0) for layers in build_lines.values())
    primary_item = min(build_lines.keys())
    build_pallet = Pallet(
        pallet_id=build_id,
        item_id=primary_item,
        layers_remaining=max(total_layers, 1),
        build_lines={sku: max(layers, 1) for sku, layers in build_lines.items()},
        current_slot=donor_slot,
        arrived_at=state.as_of,
    )
    pallets.append(build_pallet)
    slot.occupied_by_pallet_id = build_id
    return WorldState(
        pallets=pallets,
        slots=slots,
        orders=state.orders,
        exclusions=state.exclusions,
        as_of=state.as_of,
    )


def _hour_key(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _wave_open_rows_due(
    opens: list[WaveDonorOpen],
    as_of: datetime,
    *,
    step: timedelta | None,
    applied_item_ids: set[str],
    retry_overdue: bool,
) -> list[WaveDonorOpen]:
    """Wave opens to attempt now — first-time window plus overdue backlog."""
    as_of_norm = as_of.replace(second=0, microsecond=0)
    due: list[WaveDonorOpen] = []
    seen: set[str] = set()

    for row in opens:
        # Dedupe by open identity (item@wave) — a SKU may legitimately open
        # again in a later wave. `seen` stays per-item so one batch never
        # opens two pallets for the same SKU (minimum-pallets rule).
        if row.open_key in applied_item_ids or row.item_id in seen:
            continue
        scheduled = row.scheduled_open.replace(second=0, microsecond=0)
        if step is not None:
            window_start = as_of_norm - step
            first_attempt = window_start < scheduled <= as_of_norm
        else:
            first_attempt = _hour_key(row.scheduled_open) == _hour_key(as_of)
        if first_attempt:
            due.append(row)
            seen.add(row.item_id)

    if retry_overdue:
        backlog = [
            row
            for row in opens
            if row.open_key not in applied_item_ids
            and row.item_id not in seen
            and row.scheduled_open.replace(second=0, microsecond=0) < as_of_norm
        ]
        backlog.sort(key=lambda r: (r.scheduled_open, r.setup_rank, r.item_id))
        for row in backlog:
            due.append(row)
            seen.add(row.item_id)
    return due


def wave_donor_opens_at_hour(
    opens: list[WaveDonorOpen],
    as_of: datetime,
    *,
    applied_item_ids: set[str],
    retry_overdue: bool = False,
) -> list[tuple[str, int, str]]:
    """Donor opens scheduled for *as_of* (hour precision).

    When *retry_overdue* is False (legacy), marks selected SKUs applied even if
    the pad is still occupied. When True, selection is read-only — callers mark
    applied only after ``apply_donor_preopen`` succeeds.
    """
    rows = _wave_open_rows_due(
        opens,
        as_of,
        step=None,
        applied_item_ids=applied_item_ids,
        retry_overdue=retry_overdue,
    )
    batch = [(row.item_id, row.layers, row.donor_slot) for row in rows]
    if not retry_overdue:
        for row in rows:
            applied_item_ids.add(row.open_key)
    return batch


def wave_donor_opens_at_step(
    opens: list[WaveDonorOpen],
    as_of: datetime,
    *,
    step: timedelta,
    applied_item_ids: set[str],
    retry_overdue: bool = False,
) -> list[tuple[str, int, str]]:
    """Donor opens whose scheduled time falls in (as_of - step, as_of], plus backlog."""
    rows = _wave_open_rows_due(
        opens,
        as_of,
        step=step,
        applied_item_ids=applied_item_ids,
        retry_overdue=retry_overdue,
    )
    batch = [(row.item_id, row.layers, row.donor_slot) for row in rows]
    if not retry_overdue:
        for row in rows:
            applied_item_ids.add(row.open_key)
    return batch


def _apply_wave_open_rows(
    state: WorldState,
    rows: list[WaveDonorOpen],
    *,
    applied_item_ids: set[str],
    retry_failed: bool,
) -> tuple[WorldState, list[tuple[str, int, str]]]:
    """Stage due rows, deferring SKUs that already have an open build pallet."""
    if not rows:
        return state, []
    open_now = _items_with_open_builds(state)
    rows = [row for row in rows if not (row.sku_ids & open_now)]
    if not rows:
        return state, []

    new_state = state
    staged: list[tuple[str, int, str]] = []
    for row in rows:
        lines = row.line_map
        if len(lines) > 1:
            before = {s.slot_id: s.occupied_by_pallet_id for s in new_state.slots}
            new_state = apply_multi_sku_donor_preopen(new_state, row.donor_slot, lines)
            after = {s.slot_id: s.occupied_by_pallet_id for s in new_state.slots}
            if before.get(row.donor_slot) != after.get(row.donor_slot):
                staged.append((row.item_id, row.layers, row.donor_slot))
                if retry_failed:
                    applied_item_ids.add(row.open_key)
            continue
        batch_state, batch_staged = apply_donor_preopen(
            new_state, [(row.item_id, row.layers, row.donor_slot)]
        )
        new_state = batch_state
        staged.extend(batch_staged)
        if retry_failed:
            staged_pairs = {(item_id, slot) for item_id, _, slot in batch_staged}
            if (row.item_id, row.donor_slot) in staged_pairs:
                applied_item_ids.add(row.open_key)
    return new_state, staged


def apply_wave_donor_opens_at_step(
    state: WorldState,
    opens: list[WaveDonorOpen],
    *,
    as_of: datetime,
    step: timedelta,
    applied_item_ids: set[str],
    retry_failed: bool = True,
) -> tuple[WorldState, list[tuple[str, int, str]]]:
    """Open colocated donor builds due in this sim step."""
    rows = _wave_open_rows_due(
        opens,
        as_of,
        step=step,
        applied_item_ids=applied_item_ids,
        retry_overdue=retry_failed,
    )
    if not retry_failed:
        for row in rows:
            applied_item_ids.add(row.open_key)
    return _apply_wave_open_rows(
        state, rows, applied_item_ids=applied_item_ids, retry_failed=retry_failed
    )


def apply_wave_donor_opens_at_hour(
    state: WorldState,
    opens: list[WaveDonorOpen],
    *,
    as_of: datetime,
    applied_item_ids: set[str],
    retry_failed: bool = True,
) -> tuple[WorldState, list[tuple[str, int, str]]]:
    """Open colocated donor builds due at *as_of*."""
    rows = _wave_open_rows_due(
        opens,
        as_of,
        step=None,
        applied_item_ids=applied_item_ids,
        retry_overdue=retry_failed,
    )
    if not retry_failed:
        for row in rows:
            applied_item_ids.add(row.open_key)
    return _apply_wave_open_rows(
        state, rows, applied_item_ids=applied_item_ids, retry_failed=retry_failed
    )


def apply_pad_turnover_queue(
    state: WorldState,
    queue: list,
    *,
    applied_item_ids: set[str],
    afternoon_hour: int = 12,
    min_waved_layers: int = 0,
    waved_layers: int = 0,
    lead_hours: int = 1,
) -> WorldState:
    """
    Fill empty donor pads from the build queue (pad turnover).

    Runs when *waved_layers* >= *min_waved_layers*. Only stages SKUs that are
    currently waved or whose wave sends within *lead_hours* — avoids staging
    afternoon builds at dawn and exhausting pads before peak.
    """
    if not queue:
        return state
    if waved_layers < min_waved_layers:
        return state
    from layout_advisor.planning.donor_build_queue import DonorBuildQueueEntry

    free_slots = _empty_destination_slots(state)
    if not free_slots:
        return state
    waved_items = _waved_items_at(state)
    as_of = state.as_of.replace(minute=0, second=0, microsecond=0)
    free_set = set(free_slots)
    horizon = as_of + timedelta(hours=lead_hours)

    def _eligible(entry: DonorBuildQueueEntry) -> bool:
        if entry.item_id in waved_items:
            return True
        send = entry.wave_send.replace(minute=0, second=0, microsecond=0)
        return as_of <= send <= horizon

    open_now = _items_with_open_builds(state)
    pending = [
        entry
        for entry in queue
        if isinstance(entry, DonorBuildQueueEntry)
        and entry.open_key not in applied_item_ids
        and entry.item_id not in open_now
        and _eligible(entry)
    ]
    if not pending:
        return state
    pending.sort(
        key=lambda e: e.turnover_priority_key(as_of, waved_items=waved_items, afternoon_hour=afternoon_hour)
    )

    batch: list[tuple[str, int, str]] = []
    entry_by_pair: dict[tuple[str, str], DonorBuildQueueEntry] = {}
    for entry in pending:
        slot = entry.donor_slot if entry.donor_slot in free_set else None
        if slot is None:
            slot = next((s for s in free_slots if s in free_set), None)
        if slot is None:
            break
        batch.append((entry.item_id, entry.layers, slot))
        entry_by_pair[(entry.item_id, slot)] = entry
        free_set.discard(slot)
        if not free_set:
            break
    new_state, staged = apply_donor_preopen(state, batch)
    for item_id, _, slot in staged:
        entry = entry_by_pair.get((item_id, slot))
        if entry is not None:
            applied_item_ids.add(entry.open_key)
    return new_state


def _empty_destination_slots_excluding(
    state: WorldState,
    *,
    exclude: set[str],
) -> list[str]:
    return sorted(
        s.slot_id
        for s in state.slots
        if s.slot_type == SlotType.DESTINATION
        and not s.occupied_by_pallet_id
        and s.slot_id not in exclude
    )


def _pick_donor_for_order_open(
    state: WorldState,
    open_row,
    *,
    used_slots: set[str],
) -> str | None:
    """Prefer a donor pad colocated with the SKU's staged source face."""
    from layout_advisor.layout.slot_geometry import (
        build_aisle_slot_map,
        donor_slot_colocated,
        nearest_donor_on_flank,
    )

    free = _empty_destination_slots_excluding(state, exclude=used_slots)
    if not free:
        return None
    free_set = set(free)
    source_count = sum(1 for s in state.slots if s.slot_type == SlotType.SOURCE)
    donor_count = sum(1 for s in state.slots if s.slot_type == SlotType.DESTINATION)
    _, aisle_map = build_aisle_slot_map(source_count=source_count, donor_count=donor_count)
    source_slot = open_row.source_slot
    if source_slot:
        mapped = aisle_map.get(source_slot)
        if mapped is not None and mapped.zone == "source_face":
            flank = mapped.side.value
            colocated = donor_slot_colocated(
                aisle_map,
                mapped.station,
                flank,
                free_set,
            )
            if colocated:
                return colocated
            near = nearest_donor_on_flank(
                aisle_map,
                mapped.station,
                flank,
                free_set,
            )
            if near:
                return near
    return free[0]


def order_donor_opens_due(
    opens: list,
    as_of: datetime,
    *,
    step: timedelta,
    applied_keys: set[str],
    include_overdue: bool = False,
) -> list:
    """Order donor opens whose scheduled time falls in (as_of - step, as_of].

    With ``include_overdue`` every unapplied open at or before *as_of* stays
    eligible — required when a gate may defer an open past its window so it
    retries on later steps.
    """
    window_start = as_of - step
    due: list = []
    for row in opens:
        if row.open_key in applied_keys:
            continue
        scheduled = row.scheduled_open.replace(second=0, microsecond=0)
        if scheduled <= as_of if include_overdue else window_start < scheduled <= as_of:
            due.append(row)
    return due


def apply_order_donor_opens_at_step(
    state: WorldState,
    opens: list,
    *,
    as_of: datetime,
    step: timedelta,
    applied_keys: set[str],
    require_supply_complete: bool = False,
    due_override_hours: float = 0.0,
) -> WorldState:
    """Open colocated donor builds for customer orders due in this sim step.

    ``require_supply_complete``: an order opens NO donor pad until every line
    of the order (across all due rows sharing its order_key) has pickable
    supply — short orders hold no pad and retry next tick; buildable orders
    take the capacity instead. ``due_override_hours``: inside that window of
    build due the gate stops deferring (waiting guarantees a miss); 0 = never.
    """
    from itertools import groupby

    due = order_donor_opens_due(
        opens,
        as_of,
        step=step,
        applied_keys=applied_keys,
        include_overdue=require_supply_complete,
    )
    if require_supply_complete and due:
        # Build-face-independent shortage: the open CREATES the build face, so
        # item_pick_shortage_layers(require_build_face=True) would deadlock.
        from slotting.throughput import _classify_demand

        def _line_short(item_id: str, layers: int) -> bool:
            if item_id in state.excluded_item_ids:
                return False
            m = _classify_demand(state, {item_id: layers}, require_build_face=False)
            return (
                m.hand_pick_shortage_layers
                + m.low_layer_slotted_layers
                + m.hand_pick_unstaged_layers
            ) > 0

        by_order: dict[str, list] = {}
        for row in due:
            by_order.setdefault(row.order_key, []).append(row)
        short_orders = {
            key
            for key, rows in by_order.items()
            if any(_line_short(r.item_id, r.layers) for r in rows)
            and not (
                due_override_hours > 0
                and min(
                    (r.build_due - as_of).total_seconds() / 3600.0 for r in rows
                )
                <= due_override_hours
            )
        }
        due = [r for r in due if r.order_key not in short_orders]
    if not due:
        return state

    due_sorted = sorted(due, key=lambda row: (row.order_key, row.pallet_index, row.line_index))
    new_state = state
    used_slots: set[str] = set()
    for (_order_key, _pallet_index), group in groupby(
        due_sorted,
        key=lambda row: (row.order_key, row.pallet_index),
    ):
        rows = list(group)
        donor_slot = _pick_donor_for_order_open(new_state, rows[0], used_slots=used_slots)
        if donor_slot is None:
            continue
        build_lines = {row.item_id: row.layers for row in rows}
        new_state = apply_multi_sku_donor_preopen(new_state, donor_slot, build_lines)
        used_slots.add(donor_slot)
        for row in rows:
            applied_keys.add(row.open_key)
    return new_state
