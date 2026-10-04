"""Time-of-day pad role schedule — swap source ↔ donor on the same 100 pads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from layout_advisor.layout.slot_geometry import build_aisle_slot_map, source_faces_by_travel
from slotting.generate_slots import donor_slot_ids, source_slot_ids
from slotting.models import OrderArea, OrderLine, OrderStatus, Slot, SlotType, WorldState


@dataclass(frozen=True)
class PadRolePhase:
    """Target source:donor counts effective from *effective_at* (inclusive, hour precision)."""

    effective_at: datetime
    source_count: int
    donor_count: int
    notes: str

    @property
    def ratio(self) -> str:
        return f"{self.source_count}:{self.donor_count}"


@dataclass(frozen=True)
class PadRoleSwap:
    """Convert one aisle pad between source and donor roles."""

    effective_at: datetime
    slot_id: str
    from_role: str  # source | donor
    to_role: str
    station: int
    flank: str
    notes: str


def waved_sku_count_at(orders: list[OrderLine], as_of: datetime) -> int:
    """Distinct this-line SKUs in the 4-hour waved window at *as_of*."""
    from slotting.backtest import orders_for_as_of

    visible = orders_for_as_of(orders, as_of)
    return len(
        {
            o.item_id
            for o in visible
            if o.area == OrderArea.THIS_LINE and o.status == OrderStatus.WAVED
        }
    )


def waved_layer_count_at(orders: list[OrderLine], as_of: datetime) -> int:
    """Waved layer demand in the 4-hour window at *as_of*."""
    from slotting.backtest import orders_for_as_of

    visible = orders_for_as_of(orders, as_of)
    return sum(
        o.qty_layers
        for o in visible
        if o.area == OrderArea.THIS_LINE and o.status == OrderStatus.WAVED
    )


def target_source_count_for_demand(
    *,
    waved_skus: int,
    waved_layers: int,
    physical_source: int,
    early_source_delta: int = 5,
    waved_sku_buffer: int = 3,
    lull_layer_threshold: int = 110,
    peak_layer_threshold: int = 175,
    peak_waved_sku_threshold: int = 58,
    transitional_layer_high: int = 145,
) -> int:
    """
    Map hourly waved demand to a source-face count on fixed 100 pads.

    Peak windows use max sources; lulls free donor pads for ahead builds.
    """
    min_source = physical_source - early_source_delta
    max_source = physical_source
    if waved_layers >= peak_layer_threshold or waved_skus >= peak_waved_sku_threshold:
        return max_source
    if waved_layers <= lull_layer_threshold and waved_skus < peak_waved_sku_threshold - 5:
        return min_source
    if waved_layers >= transitional_layer_high:
        return max_source
    return max(min_source, min(max_source, waved_skus + waved_sku_buffer))


def build_hourly_demand_phases(
    orders: list[OrderLine],
    ship_date: date,
    *,
    physical_source: int = 70,
    physical_donor: int = 30,
    setup_hour: int = 22,
    early_source_delta: int = 5,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    waved_sku_buffer: int = 3,
    lull_layer_threshold: int = 110,
    peak_layer_threshold: int = 175,
    peak_waved_sku_threshold: int = 58,
    transitional_layer_high: int = 145,
    min_phase_hours: int = 2,
) -> tuple[PadRolePhase, ...]:
    """
  Hourly pad ratio schedule driven by waved SKU/layer demand.

  Setup is donor-heavy; pick day walks source count up for peaks and down
  during lulls so assist can pre-build without extra labor.
    """
    total = physical_source + physical_donor
    early_src = physical_source - early_source_delta
    prep_day = ship_date - timedelta(days=1)
    phases: list[PadRolePhase] = [
        PadRolePhase(
            datetime.combine(prep_day, time(setup_hour)),
            early_src,
            total - early_src,
            f"Setup donor-heavy ({early_src}:{total - early_src}) for pre-open builds.",
        )
    ]
    current_source = early_src
    last_change_at: datetime | None = None

    for hour in range(pick_start_hour, final_shift_hour + 1):
        as_of = datetime.combine(ship_date, time(hour))
        waved_skus = waved_sku_count_at(orders, as_of)
        waved_layers = waved_layer_count_at(orders, as_of)
        target = target_source_count_for_demand(
            waved_skus=waved_skus,
            waved_layers=waved_layers,
            physical_source=physical_source,
            early_source_delta=early_source_delta,
            waved_sku_buffer=waved_sku_buffer,
            lull_layer_threshold=lull_layer_threshold,
            peak_layer_threshold=peak_layer_threshold,
            peak_waved_sku_threshold=peak_waved_sku_threshold,
            transitional_layer_high=transitional_layer_high,
        )
        if target == current_source:
            continue
        if last_change_at is not None and (as_of - last_change_at) < timedelta(hours=min_phase_hours):
            continue
        donor_count = total - target
        phases.append(
            PadRolePhase(
                as_of,
                target,
                donor_count,
                (
                    f"Demand @ {as_of:%H:%M} - {target}:{donor_count} "
                    f"({waved_skus} waved SKUs, {waved_layers} layers)."
                ),
            )
        )
        current_source = target
        last_change_at = as_of

    return tuple(phases)


def resolve_pad_role_phases(
    orders: list[OrderLine] | None,
    ship_date: date,
    *,
    physical_source: int = 70,
    physical_donor: int = 30,
    setup_hour: int = 22,
    prepeak_hour: int = 14,
    early_source_delta: int = 5,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    hourly_demand: bool = True,
    demand_triggered: bool = True,
    restore_waved_sku_threshold: int = 65,
    waved_sku_buffer: int = 3,
    lull_layer_threshold: int = 110,
    peak_layer_threshold: int = 175,
    peak_waved_sku_threshold: int = 58,
    transitional_layer_high: int = 145,
    min_phase_hours: int = 2,
) -> tuple[PadRolePhase, ...]:
    """Pick pad-role phases for planning or simulation."""
    if orders and hourly_demand:
        return build_hourly_demand_phases(
            orders,
            ship_date,
            physical_source=physical_source,
            physical_donor=physical_donor,
            setup_hour=setup_hour,
            early_source_delta=early_source_delta,
            pick_start_hour=pick_start_hour,
            final_shift_hour=final_shift_hour,
            waved_sku_buffer=waved_sku_buffer,
            lull_layer_threshold=lull_layer_threshold,
            peak_layer_threshold=peak_layer_threshold,
            peak_waved_sku_threshold=peak_waved_sku_threshold,
            min_phase_hours=min_phase_hours,
            transitional_layer_high=transitional_layer_high,
        )
    if orders and demand_triggered:
        return build_demand_triggered_phases(
            orders,
            ship_date,
            physical_source=physical_source,
            physical_donor=physical_donor,
            setup_hour=setup_hour,
            early_source_delta=early_source_delta,
            restore_waved_sku_threshold=restore_waved_sku_threshold,
            pick_start_hour=pick_start_hour,
            final_shift_hour=final_shift_hour,
            fallback_prepeak_hour=prepeak_hour,
        )
    early_keep = physical_source - early_source_delta
    return default_operating_phases(
        ship_date,
        physical_source=physical_source,
        physical_donor=physical_donor,
        setup_hour=setup_hour,
        prepeak_hour=prepeak_hour,
        early_source=early_keep,
    )


def build_demand_triggered_phases(
    orders: list[OrderLine],
    ship_date: date,
    *,
    physical_source: int = 70,
    physical_donor: int = 30,
    setup_hour: int = 22,
    early_source_delta: int = 5,
    restore_waved_sku_threshold: int = 65,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    fallback_prepeak_hour: int = 14,
) -> tuple[PadRolePhase, ...]:
    """
    Donor-heavy at setup; restore max sources when waved SKU count crosses threshold.

    Scans hourly on *ship_date* from *pick_start_hour* through *final_shift_hour*.
    Falls back to *fallback_prepeak_hour* if threshold is never reached.
    """
    total = physical_source + physical_donor
    early_src = physical_source - early_source_delta
    early_don = total - early_src
    prep_day = ship_date - timedelta(days=1)
    setup_phase = PadRolePhase(
        datetime.combine(prep_day, time(setup_hour)),
        early_src,
        early_don,
        f"Donor-heavy ({early_src}:{early_don}) for setup pre-open builds.",
    )
    restore_at: datetime | None = None
    for hour in range(pick_start_hour, final_shift_hour + 1):
        as_of = datetime.combine(ship_date, time(hour))
        if waved_sku_count_at(orders, as_of) >= restore_waved_sku_threshold:
            restore_at = as_of
            break
    if restore_at is None:
        restore_at = datetime.combine(ship_date, time(fallback_prepeak_hour))
    restore_phase = PadRolePhase(
        restore_at,
        physical_source,
        physical_donor,
        (
            f"Pre-peak ({physical_source}:{physical_donor}): "
            f"waved SKUs reached {restore_waved_sku_threshold}+ at {restore_at:%H:%M}."
        ),
    )
    return (setup_phase, restore_phase)


def default_operating_phases(
    ship_date: date,
    *,
    physical_source: int = 70,
    physical_donor: int = 30,
    setup_hour: int = 22,
    prepeak_hour: int = 14,
    early_source: int | None = None,
) -> tuple[PadRolePhase, ...]:
    """
    Default operating schedule on a fixed ``physical_source:physical_donor`` module.

    Evening setup runs donor-heavy for pre-open; pre-peak restores max source faces.
    """
    total = physical_source + physical_donor
    early_src = early_source if early_source is not None else physical_source - 5
    early_don = total - early_src
    prep_day = ship_date - timedelta(days=1)
    return (
        PadRolePhase(
            datetime.combine(prep_day, time(setup_hour)),
            early_src,
            early_don,
            f"Donor-heavy ({early_src}:{early_don}) for setup pre-open builds.",
        ),
        PadRolePhase(
            datetime.combine(ship_date, time(prepeak_hour)),
            physical_source,
            physical_donor,
            f"Pre-peak ({physical_source}:{physical_donor}): restore source faces before spike.",
        ),
    )


def _source_to_donor_candidates(
    *,
    source_count: int,
    donor_count: int,
    count: int,
) -> list[str]:
    """Low-priority source faces to convert to donor (tail of travel ranking)."""
    order = source_faces_by_travel(source_count=source_count, donor_count=donor_count)
    return list(reversed(order))[:count]


def build_pad_role_swaps(
    ship_date: date,
    *,
    physical_source: int = 70,
    physical_donor: int = 30,
    phases: tuple[PadRolePhase, ...] | None = None,
    source_to_donor_slots: tuple[str, ...] | None = None,
) -> list[PadRoleSwap]:
    """
    Emit ordered pad role conversions to walk through *phases* on *ship_date*.

    Starts from the built physical ratio ``physical_source:physical_donor``.
    When *source_to_donor_slots* is set, those exact source faces convert to donor
    during the first donor-heavy phase (typically lowest setup ranks 71–75).
    """
    if phases is None:
        phases = default_operating_phases(
            ship_date,
            physical_source=physical_source,
            physical_donor=physical_donor,
        )
    layout, aisle_map = build_aisle_slot_map(
        source_count=physical_source,
        donor_count=physical_donor,
    )
    _ = layout

    current_source = physical_source
    converted_to_donor: list[str] = []
    swaps: list[PadRoleSwap] = []
    explicit_donor = list(source_to_donor_slots or ())

    for phase in sorted(phases, key=lambda p: p.effective_at):
        delta = phase.source_count - current_source
        if delta == 0:
            continue
        if delta < 0:
            n = -delta
            # Explicit low-rank faces first, then the travel tail — skipping
            # pads already converted, so a second donor-heavy phase after a
            # partial restore never emits a duplicate (no-op) conversion.
            pool = explicit_donor + _source_to_donor_candidates(
                source_count=physical_source,
                donor_count=physical_donor,
                count=physical_source,
            )
            already = set(converted_to_donor)
            candidates: list[str] = []
            for slot_id in pool:
                if slot_id in already:
                    continue
                already.add(slot_id)
                candidates.append(slot_id)
                if len(candidates) == n:
                    break
            for slot_id in candidates:
                mapped = aisle_map[slot_id]
                swaps.append(
                    PadRoleSwap(
                        effective_at=phase.effective_at,
                        slot_id=slot_id,
                        from_role="source",
                        to_role="donor",
                        station=mapped.station,
                        flank=mapped.side.value.upper(),
                        notes=phase.notes,
                    )
                )
                converted_to_donor.append(slot_id)
            # Track what was actually emitted — claiming the phase target when
            # fewer pads converted would silently drift the ratio.
            current_source -= len(candidates)
        else:
            n = delta
            restore = converted_to_donor[-n:]
            converted_to_donor = converted_to_donor[:-n]
            for slot_id in restore:
                mapped = aisle_map[slot_id]
                swaps.append(
                    PadRoleSwap(
                        effective_at=phase.effective_at,
                        slot_id=slot_id,
                        from_role="donor",
                        to_role="source",
                        station=mapped.station,
                        flank=mapped.side.value.upper(),
                        notes=phase.notes,
                    )
                )
            current_source += len(restore)

    return swaps


def effective_donor_slot_ids(
    *,
    physical_source: int,
    physical_donor: int,
    swaps_applied: list[PadRoleSwap],
    as_of: datetime,
) -> list[str]:
    """Donor sim slot IDs after scheduled swaps through *as_of*."""
    donor_ids = set(donor_slot_ids(physical_donor))
    hour = as_of.replace(minute=0, second=0, microsecond=0)
    for swap in sorted(swaps_applied, key=lambda s: s.effective_at):
        if swap.effective_at > hour:
            break
        if swap.from_role == "source" and swap.to_role == "donor":
            donor_ids.add(swap.slot_id)
        elif swap.from_role == "donor" and swap.to_role == "source":
            donor_ids.discard(swap.slot_id)
    return sorted(donor_ids)


def effective_source_slot_ids(
    *,
    physical_source: int,
    physical_donor: int,
    swaps_applied: list[PadRoleSwap],
    as_of: datetime,
) -> list[str]:
    """Active source sim slot IDs after scheduled swaps through *as_of*."""
    source_ids = set(source_slot_ids(physical_source))
    hour = as_of.replace(minute=0, second=0, microsecond=0)
    for swap in sorted(swaps_applied, key=lambda s: s.effective_at):
        if swap.effective_at > hour:
            break
        if swap.from_role == "source" and swap.to_role == "donor":
            source_ids.discard(swap.slot_id)
        elif swap.from_role == "donor" and swap.to_role == "source":
            source_ids.add(swap.slot_id)
    return sorted(source_ids)


def _slot_is_empty(slot: Slot) -> bool:
    return not slot.occupied_by_pallet_id and not slot.backup_pallet_id


def _seat_starved_waved_items(state: WorldState) -> bool:
    """True when a waved item has zero face presence but eligible on-site
    supply (unslotted or rack) — the evening admission deadlock."""
    from slotting.rules import (
        _eligible_unslotted_pallets,
        _held_pallet_ids,
        _items_with_waved_demand,
    )

    waved = _items_with_waved_demand(state)
    if not waved:
        return False
    held = _held_pallet_ids(state)
    rack_items = {
        p.item_id
        for s in state.resupply_rack_slots
        if s.occupied_by_pallet_id
        and (p := state.pallet_by_id(s.occupied_by_pallet_id)) is not None
        and p.layers_remaining > 0
        and state.is_eligible_for_line(p)
    }
    for item_id in waved:
        if state.item_in_source_slot(item_id):
            continue
        if item_id in rack_items or _eligible_unslotted_pallets(state, item_id, held):
            return True
    return False


def apply_pad_role_swaps(
    state: WorldState,
    swaps: list[PadRoleSwap],
    *,
    restore_on_starve: bool = False,
) -> tuple[WorldState, list[PadRoleSwap]]:
    """
    Apply swaps scheduled at or before ``state.as_of`` (hour precision).

    For each pad only the LATEST scheduled swap counts, and it is applied only
    while the pad's current role still differs from its target. Restores
    (donor→source) retry on later steps when blocked by an in-progress build —
    dropping one silently starves the peak of a source face. Conversions
    (source→donor) fire only in their scheduled hour: retrying a missed one
    mid-day steals a live source face the planner's restore list never targets,
    which costs layer-pick rate rather than saving it.
    Skips occupied pads; returns applied swaps.
    """
    hour = state.as_of.replace(minute=0, second=0, microsecond=0)
    latest: dict[str, PadRoleSwap] = {}
    for swap in sorted(swaps, key=lambda s: s.effective_at):
        if swap.effective_at <= hour:
            latest[swap.slot_id] = swap
    target = {"source": SlotType.SOURCE, "donor": SlotType.DESTINATION}
    current_type = {s.slot_id: s.slot_type for s in state.slots}
    due = [
        s
        for s in latest.values()
        if current_type.get(s.slot_id) is not None
        and current_type[s.slot_id] != target[s.to_role]
        and (s.to_role == "source" or s.effective_at == hour)
    ]
    if restore_on_starve:
        # Pull a SCHEDULED donor->source restore forward when its pad idles
        # in donor role while a waved SKU is seat-starved (the evening
        # admission deadlock). Only the pad's NEXT scheduled swap
        # qualifies - never jump over an intervening conversion.
        due_slots = {s.slot_id for s in due}
        next_swap: dict[str, PadRoleSwap] = {}
        for swap in sorted(swaps, key=lambda s: s.effective_at):
            if swap.effective_at > hour and swap.slot_id not in next_swap:
                next_swap[swap.slot_id] = swap
        empty_by_id = {s.slot_id: _slot_is_empty(s) for s in state.slots}
        candidates = [
            s
            for s in next_swap.values()
            if s.to_role == "source"
            and s.slot_id not in due_slots
            and current_type.get(s.slot_id) == SlotType.DESTINATION
            and empty_by_id.get(s.slot_id, False)
        ]
        if candidates and _seat_starved_waved_items(state):
            due.extend(candidates)
    if not due:
        return state, []

    slots = [s.model_copy(deep=True) for s in state.slots]
    pallets = [p.model_copy(deep=True) for p in state.pallets]
    slot_by_id = {s.slot_id: s for s in slots}
    pallet_by_id = {p.pallet_id: p for p in pallets}
    applied: list[PadRoleSwap] = []
    stale_items: set[str] | None = None  # computed once, only if an occupied swap needs it

    for swap in due:
        slot = slot_by_id.get(swap.slot_id)
        if slot is None:
            continue
        if not _slot_is_empty(slot):
            if swap.from_role == "source" and swap.to_role == "donor":
                pid = slot.occupied_by_pallet_id
                pallet = pallet_by_id.get(pid) if pid else None
                if pallet is None:
                    continue
                if stale_items is None:
                    from slotting.slotting_policy import items_without_rest_of_ship_day_demand

                    stale_items = items_without_rest_of_ship_day_demand(state)
                if pallet.item_id not in stale_items:
                    continue
                from slotting.inventory_policy import mark_removed_from_pick

                mark_removed_from_pick(
                    pallet,
                    as_of=state.as_of,
                    source_slot_id=slot.slot_id,
                )
                slot.occupied_by_pallet_id = None
                pallet.current_slot = None
            else:
                continue
        if swap.from_role == "source" and swap.to_role == "donor":
            if slot.slot_type != SlotType.SOURCE:
                continue
            slot.slot_type = SlotType.DESTINATION
            applied.append(swap)
        elif swap.from_role == "donor" and swap.to_role == "source":
            if slot.slot_type != SlotType.DESTINATION:
                continue
            slot.slot_type = SlotType.SOURCE
            applied.append(swap)

    if not applied:
        return state, []

    return (
        WorldState(
            pallets=pallets,
            slots=slots,
            orders=state.orders,
            exclusions=state.exclusions,
            as_of=state.as_of,
        ),
        applied,
    )


def build_pad_swaps_from_plan(
    plan,
    ship_date: date,
    *,
    source_slots: int = 70,
    donor_slots: int = 30,
    setup_hour: int = 22,
    prepeak_hour: int = 14,
    early_source_delta: int = 5,
    orders: list[OrderLine] | None = None,
    hourly_demand: bool = True,
    demand_triggered: bool = True,
    restore_waved_sku_threshold: int = 65,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    waved_sku_buffer: int = 3,
    lull_layer_threshold: int = 110,
    peak_layer_threshold: int = 175,
    peak_waved_sku_threshold: int = 58,
    transitional_layer_high: int = 145,
    min_phase_hours: int = 2,
) -> list[PadRoleSwap]:
    """Build pad schedule from a :class:`SetupSlottingPlan` (ranks 71+ faces → donor)."""
    early_keep = source_slots - early_source_delta
    low_faces = tuple(
        plan.face_by_item[item_id]
        for item_id, rank in sorted(plan.setup_ranks.items(), key=lambda row: row[1])
        if rank > early_keep and item_id in plan.face_by_item
    )
    phases = resolve_pad_role_phases(
        orders,
        ship_date,
        physical_source=source_slots,
        physical_donor=donor_slots,
        setup_hour=setup_hour,
        prepeak_hour=prepeak_hour,
        early_source_delta=early_source_delta,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
        hourly_demand=hourly_demand,
        demand_triggered=demand_triggered,
        restore_waved_sku_threshold=restore_waved_sku_threshold,
        waved_sku_buffer=waved_sku_buffer,
        lull_layer_threshold=lull_layer_threshold,
        peak_layer_threshold=peak_layer_threshold,
        peak_waved_sku_threshold=peak_waved_sku_threshold,
        transitional_layer_high=transitional_layer_high,
        min_phase_hours=min_phase_hours,
    )
    return build_pad_role_swaps(
        ship_date,
        physical_source=source_slots,
        physical_donor=donor_slots,
        phases=phases,
        source_to_donor_slots=low_faces or None,
    )


def build_pad_swaps_for_day(
    ship_date: date,
    assignments: list,
    *,
    source_slots: int = 70,
    donor_slots: int = 30,
    setup_hour: int = 22,
    prepeak_hour: int = 14,
    early_source_delta: int = 5,
    orders: list[OrderLine] | None = None,
    hourly_demand: bool = True,
    demand_triggered: bool = True,
    restore_waved_sku_threshold: int = 65,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    waved_sku_buffer: int = 3,
    lull_layer_threshold: int = 110,
    peak_layer_threshold: int = 175,
    peak_waved_sku_threshold: int = 58,
    transitional_layer_high: int = 145,
    min_phase_hours: int = 2,
) -> list[PadRoleSwap]:
    """Build the pad schedule using lowest setup-ranked source faces."""
    early_keep = source_slots - early_source_delta
    low_faces = tuple(
        a.slot_id
        for a in sorted(assignments, key=lambda row: row.rank)
        if getattr(a, "location", None) == "source_face"
        and a.rank > early_keep
        and a.slot_id
    )
    phases = resolve_pad_role_phases(
        orders,
        ship_date,
        physical_source=source_slots,
        physical_donor=donor_slots,
        setup_hour=setup_hour,
        prepeak_hour=prepeak_hour,
        early_source_delta=early_source_delta,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
        hourly_demand=hourly_demand,
        demand_triggered=demand_triggered,
        restore_waved_sku_threshold=restore_waved_sku_threshold,
        waved_sku_buffer=waved_sku_buffer,
        lull_layer_threshold=lull_layer_threshold,
        peak_layer_threshold=peak_layer_threshold,
        peak_waved_sku_threshold=peak_waved_sku_threshold,
        transitional_layer_high=transitional_layer_high,
        min_phase_hours=min_phase_hours,
    )
    return build_pad_role_swaps(
        ship_date,
        physical_source=source_slots,
        physical_donor=donor_slots,
        phases=phases,
        source_to_donor_slots=low_faces or None,
    )


def pad_role_swaps_to_wave_actions(
    swaps: list[PadRoleSwap],
    ship_date: date,
    *,
    sequence_start: int = 1,
) -> list[tuple[int, PadRoleSwap]]:
    """Return ``(sequence, swap)`` pairs for plan assembly."""
    return [(sequence_start + i, swap) for i, swap in enumerate(swaps)]
