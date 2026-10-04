"""Donor build queue priority and layer-triggered rolling pre-open on spike ship days."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta

from layout_advisor.planning.dynamic_slotting import SetupAssignment, WaveDonorOpen, wave_colocated_donor_opens
from layout_advisor.pick_throughput import PICK_LAYERS_PER_HOUR
from slotting.models import OrderLine
from slotting.visibility_planning import visibility_snapshot


def _hour_key(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def ship_day_peak_waved_demand(
    orders: list[OrderLine],
    ship_date: date,
    *,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
) -> tuple[int, int]:
    """Peak distinct waved SKUs and waved layers in the 4h window across the pick day."""
    peak_skus = 0
    peak_layers = 0
    for hour in range(pick_start_hour, final_shift_hour + 1):
        as_of = datetime.combine(ship_date, time(hour))
        snap = visibility_snapshot(orders, as_of)
        peak_skus = max(peak_skus, snap.waved_skus)
        peak_layers = max(peak_layers, snap.waved_layers)
    return peak_skus, peak_layers


def is_spike_ship_day(
    orders: list[OrderLine],
    ship_date: date,
    *,
    sku_threshold: int,
    layer_threshold: int,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
) -> bool:
    """True when peak waved SKUs or layers on *ship_date* cross spike thresholds."""
    peak_skus, peak_layers = ship_day_peak_waved_demand(
        orders,
        ship_date,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
    )
    return peak_skus >= sku_threshold or peak_layers >= layer_threshold


@dataclass(frozen=True)
class DonorBuildQueueEntry:
    """One staged SKU needing a donor build, with queue priority metadata."""

    wave_index: int
    wave_send: datetime
    wave_due: datetime
    item_id: str
    layers: int
    setup_rank: int
    donor_slot: str
    source_slot: str | None
    build_key: str = ""
    build_lines: tuple[tuple[str, int], ...] = ()

    @property
    def priority_key(self) -> tuple:
        return (self.wave_send, -self.layers, self.setup_rank, self.item_id)

    @property
    def open_key(self) -> str:
        """Dedupe identity shared with WaveDonorOpen (item@wave)."""
        if self.build_key:
            return f"{self.build_key}@{self.wave_index}"
        return f"{self.item_id}@{self.wave_index}"

    def spike_priority_key(self, *, afternoon_hour: int = 12) -> tuple:
        """Afternoon waves and heavy layers first — for spike-day setup pre-open."""
        afternoon = 0 if self.wave_send.hour >= afternoon_hour else 1
        return (afternoon, self.wave_send, -self.layers, self.setup_rank, self.item_id)

    def turnover_priority_key(
        self,
        as_of: datetime,
        *,
        waved_items: set[str],
        afternoon_hour: int = 12,
    ) -> tuple:
        """Pad-turnover drain: waved now, then imminent send, then spike order."""
        hour = as_of.replace(minute=0, second=0, microsecond=0)
        send_hour = self.wave_send.replace(minute=0, second=0, microsecond=0)
        waved_now = 0 if self.item_id in waved_items else 1
        overdue = 1 if send_hour < hour else 0
        imminent = 0 if send_hour <= hour + timedelta(hours=4) else 1
        return (waved_now, overdue, imminent, *self.spike_priority_key(afternoon_hour=afternoon_hour))


def build_spike_setup_preopen(
    orders: list[OrderLine],
    ship_date: date,
    setup_assignments: list[SetupAssignment],
    *,
    wave_count: int,
    max_builds: int,
    afternoon_hour: int = 12,
    donor_count: int = 30,
    source_count: int = 70,
    donor_slot_ids: list[str] | None = None,
    setup_at: datetime | None = None,
    due_offset_hours: float = 4.0,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
) -> list[tuple[str, int, str]]:
    """
    Setup pre-open capped at *max_builds*, prioritizing afternoon / heavy waves.

    Scans *wave_count* post-setup waves, ranks all staged SKUs, stages the top
    *max_builds* on colocated donor pads.
    """
    from layout_advisor.planning.dynamic_slotting import (
        _assign_colocated_donors,
        build_aisle_slot_map,
        pick_waves_after_setup,
        stations_per_side,
    )
    from slotting.generate_slots import donor_slot_ids as default_donor_ids
    from slotting.wave_release import order_ship_date, split_day_by_native_due

    day_orders = [o for o in orders if order_ship_date(o) == ship_date]
    if not day_orders or max_builds <= 0:
        return []
    buckets = split_day_by_native_due(
        day_orders,
        ship_date=ship_date,
        due_offset_hours=due_offset_hours,
    )
    layout, aisle_map = build_aisle_slot_map(
        source_count=source_count,
        donor_count=donor_count,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )
    stations = stations_per_side(layout)
    donors = list(donor_slot_ids or default_donor_ids(donor_count))
    if picker_count > 1 or center_gap_ft > 0:
        from layout_advisor.layout.slot_geometry import donor_slots_interleaved_by_flank, gap_excluded_stations

        donors = donor_slots_interleaved_by_flank(donors, aisle_map)
    gap_zone = gap_excluded_stations(layout) if center_gap_ft > 0 or picker_count > 1 else set()
    staged_rows = [
        a
        for a in setup_assignments
        if a.location in {"source_face", "rack"} and a.slot_id
    ]
    eligible = pick_waves_after_setup(buckets, setup_at=setup_at or buckets[0].release_at)
    end = min(len(eligible), max(1, wave_count))
    ranked: list[tuple[tuple, str, int, str]] = []
    seen: set[str] = set()

    for idx in range(end):
        wave = eligible[idx]
        wave_items = {o.item_id for o in wave.orders}
        # Full pad pool per wave ON PURPOSE: this pass only generates
        # candidates with their best colocated pad. Pad conflicts are resolved
        # below in spike-priority order — accumulating used pads here would
        # hand prime pads to early waves and starve the afternoon peak.
        wave_used_donors: set[str] = set()
        pairs = _assign_colocated_donors(
            staged_rows,
            wave_items,
            aisle_map=aisle_map,
            stations=stations,
            donor_slot_ids=donors,
            used_donors=wave_used_donors,
            used_items=seen,
            gap_excluded=gap_zone,
        )
        for row, donor_slot in pairs:
            layers = sum(o.qty_layers for o in wave.orders if o.item_id == row.item_id)
            entry = DonorBuildQueueEntry(
                wave_index=wave.wave_index,
                wave_send=wave.release_at,
                wave_due=wave.due_at,
                item_id=row.item_id,
                layers=layers,
                setup_rank=row.rank,
                donor_slot=donor_slot,
                source_slot=row.slot_id,
            )
            ranked.append((entry.spike_priority_key(afternoon_hour=afternoon_hour), row.item_id, layers, donor_slot))
            seen.add(row.item_id)

    ranked.sort(key=lambda row: row[0])
    out: list[tuple[str, int, str]] = []
    used: set[str] = set()
    used_pads: set[str] = set()
    for _, item_id, layers, donor_slot in ranked:
        if item_id in used or donor_slot in used_pads:
            # One pad, one build: the whole batch lands simultaneously at
            # setup. Deduping here (in priority order) replaces the old
            # behavior where the sim silently dropped the duplicate at apply
            # time AND the doomed entry wasted a max_builds slot.
            continue
        out.append((item_id, layers, donor_slot))
        used.add(item_id)
        used_pads.add(donor_slot)
        if len(out) >= max_builds:
            break
    return out


def queue_from_wave_opens(opens: list[WaveDonorOpen]) -> list[DonorBuildQueueEntry]:
    """Derive a priority queue from colocated wave donor opens."""
    entries = [
        DonorBuildQueueEntry(
            wave_index=row.wave_index,
            wave_send=row.wave_send,
            wave_due=row.wave_due,
            item_id=row.item_id,
            layers=row.layers,
            setup_rank=row.setup_rank,
            donor_slot=row.donor_slot,
            source_slot=row.source_slot,
            build_key=row.build_key,
            build_lines=row.build_lines,
        )
        for row in opens
    ]
    entries.sort(key=lambda e: e.priority_key)
    return entries


def estimate_pad_free_at(
    entry: DonorBuildQueueEntry,
    *,
    open_at: datetime,
    pad_free_on_build_complete: bool,
    pick_layers_per_hour: float,
) -> datetime:
    """When the planner assumes a donor pad becomes reusable."""
    if not pad_free_on_build_complete:
        return entry.wave_due
    rate = max(pick_layers_per_hour, 1.0)
    dwell = timedelta(hours=entry.layers / rate)
    return max(open_at, open_at + dwell)


def schedule_rolling_preopens(
    entries: list[DonorBuildQueueEntry],
    *,
    setup_at: datetime,
    donor_pad_count: int,
    lead_hours: int,
    afternoon_start_hour: int = 12,
    pad_free_on_build_complete: bool = True,
    pick_layers_per_hour: float = PICK_LAYERS_PER_HOUR,
) -> list[WaveDonorOpen]:
    """
    Pull donor opens earlier on spike days; stagger when demand exceeds pad count.

    Only waves sending at or after *afternoon_start_hour* are pulled earlier;
    morning waves keep opens at wave send so builds are not exhausted before peak.
    """
    if not entries or donor_pad_count <= 0:
        return []
    ordered = sorted(entries, key=lambda e: e.priority_key)
    scheduled_per_hour: dict[datetime, int] = {}
    pad_free_at: dict[str, datetime] = {}
    opens: list[WaveDonorOpen] = []

    for entry in ordered:
        if entry.wave_send.hour < afternoon_start_hour:
            hour = _hour_key(entry.wave_send)
        else:
            target = max(setup_at, entry.wave_send - timedelta(hours=lead_hours))
            pad_busy = pad_free_at.get(entry.donor_slot, setup_at)
            hour = _hour_key(max(target, pad_busy))
            wave_hour = _hour_key(entry.wave_send)
            while scheduled_per_hour.get(hour, 0) >= donor_pad_count and hour < wave_hour:
                hour += timedelta(hours=1)
            if scheduled_per_hour.get(hour, 0) >= donor_pad_count:
                hour = wave_hour
            pad_free_at[entry.donor_slot] = max(
                pad_free_at.get(entry.donor_slot, setup_at),
                estimate_pad_free_at(
                    entry,
                    open_at=hour,
                    pad_free_on_build_complete=pad_free_on_build_complete,
                    pick_layers_per_hour=pick_layers_per_hour,
                ),
            )
        scheduled_per_hour[hour] = scheduled_per_hour.get(hour, 0) + 1
        opens.append(
            WaveDonorOpen(
                wave_index=entry.wave_index,
                wave_send=entry.wave_send,
                wave_due=entry.wave_due,
                item_id=entry.item_id,
                layers=entry.layers,
                donor_slot=entry.donor_slot,
                source_slot=entry.source_slot,
                setup_rank=entry.setup_rank,
                open_at=hour,
                build_key=entry.build_key,
                build_lines=entry.build_lines,
            )
        )
    return opens


def _hourly_waved_layers(
    orders: list[OrderLine],
    ship_date: date,
    *,
    pick_start_hour: int,
    final_shift_hour: int,
) -> list[tuple[datetime, int]]:
    rows: list[tuple[datetime, int]] = []
    for hour in range(pick_start_hour, final_shift_hour + 1):
        as_of = datetime.combine(ship_date, time(hour))
        snap = visibility_snapshot(orders, as_of)
        rows.append((as_of, snap.waved_layers))
    return rows


def schedule_lull_ahead_preopens(
    opens: list[WaveDonorOpen],
    *,
    orders: list[OrderLine],
    ship_date: date,
    layers_per_hour: float,
    utilization_pct: float = 0.55,
    max_lookahead_hours: int = 8,
    max_builds_per_hour: int = 5,
    afternoon_hour: int = 12,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
) -> list[WaveDonorOpen]:
    """
    Pull donor opens into slack hours when pick demand is below capacity.

    When waved layers at an hour are under *utilization_pct* × hourly pick
    capacity, stage future-wave donor builds early so crews can get ahead.
    """
    if not opens or layers_per_hour <= 0:
        return opens
    lull_cap = max(1, int(layers_per_hour * utilization_pct))
    hourly = _hourly_waved_layers(
        orders,
        ship_date,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
    )
    scheduled = list(opens)
    horizon = timedelta(hours=max_lookahead_hours)

    for hour_dt, waved_layers in hourly:
        if waved_layers >= lull_cap:
            continue
        builds_this_hour = 0
        candidates: list[tuple[tuple, int]] = []
        for idx, row in enumerate(scheduled):
            if row.scheduled_open <= hour_dt:
                continue
            if row.wave_send <= hour_dt:
                continue
            if row.wave_send > hour_dt + horizon:
                continue
            afternoon = 0 if row.wave_send.hour >= afternoon_hour else 1
            candidates.append(
                ((afternoon, row.wave_send, -row.layers, row.setup_rank, row.item_id), idx)
            )
        candidates.sort(key=lambda pair: pair[0])
        for _, idx in candidates:
            if builds_this_hour >= max_builds_per_hour:
                break
            row = scheduled[idx]
            if row.scheduled_open <= hour_dt:
                continue
            scheduled[idx] = replace(row, open_at=hour_dt, ahead_prep=True)
            builds_this_hour += 1
    return scheduled


def build_donor_build_queue(
    orders: list[OrderLine],
    ship_date: date,
    setup_assignments: list[SetupAssignment],
    wave_buckets: list,
    *,
    source_count: int = 70,
    donor_count: int = 30,
    donor_slot_ids: list[str] | None = None,
    setup_at: datetime | None = None,
    preopen_wave_count: int = 0,
    due_offset_hours: float = 4.0,
    initial_used_items: set[str] | None = None,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
    allow_repeat_opens: bool = True,
) -> list[DonorBuildQueueEntry]:
    """Priority-ordered donor builds for waves beyond the setup pre-open window."""
    opens = wave_colocated_donor_opens(
        orders,
        ship_date,
        setup_assignments,
        wave_buckets,
        source_count=source_count,
        donor_count=donor_count,
        donor_slot_ids=donor_slot_ids,
        setup_at=setup_at,
        preopen_wave_count=preopen_wave_count,
        due_offset_hours=due_offset_hours,
        initial_used_items=initial_used_items,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
        allow_repeat_opens=allow_repeat_opens,
    )
    return queue_from_wave_opens(opens)


def build_wave_donor_opens(
    orders: list[OrderLine],
    ship_date: date,
    setup_assignments: list[SetupAssignment],
    wave_buckets: list,
    *,
    source_count: int = 70,
    donor_count: int = 30,
    donor_slot_ids: list[str] | None = None,
    setup_at: datetime | None = None,
    preopen_wave_count: int = 0,
    due_offset_hours: float = 4.0,
    initial_used_items: set[str] | None = None,
    rolling_preopen_enabled: bool = True,
    spike_sku_threshold: int = 50,
    spike_layer_threshold: int = 280,
    rolling_lead_hours: int = 2,
    rolling_afternoon_hour: int = 12,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    effective_donor_count: int | None = None,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
    lull_ahead_prep_enabled: bool = True,
    lull_ahead_utilization_pct: float = 0.55,
    lull_ahead_max_lookahead_hours: int = 8,
    lull_ahead_max_builds_per_hour: int = 5,
    pick_layers_per_hour: float = PICK_LAYERS_PER_HOUR,
    pad_free_on_build_complete: bool = True,
    allow_repeat_opens: bool = True,
    multi_pallet_slices: bool = False,
    skus_path=None,
) -> list[WaveDonorOpen]:
    """
    Colocated per-wave donor opens, with optional rolling pre-open on spike days.

    On spike ship days (peak waved SKUs/layers cross thresholds), waves beyond
  the setup pre-open window are scheduled ``rolling_lead_hours`` before send.
    """
    opens = wave_colocated_donor_opens(
        orders,
        ship_date,
        setup_assignments,
        wave_buckets,
        source_count=source_count,
        donor_count=donor_count,
        donor_slot_ids=donor_slot_ids,
        setup_at=setup_at,
        preopen_wave_count=preopen_wave_count,
        due_offset_hours=due_offset_hours,
        initial_used_items=initial_used_items,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
        allow_repeat_opens=allow_repeat_opens,
        multi_pallet_slices=multi_pallet_slices,
        skus_path=skus_path,
    )
    if not opens:
        return opens
    if rolling_preopen_enabled and is_spike_ship_day(
        orders,
        ship_date,
        sku_threshold=spike_sku_threshold,
        layer_threshold=spike_layer_threshold,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
    ):
        pad_count = effective_donor_count if effective_donor_count is not None else donor_count
        queue = queue_from_wave_opens(opens)
        opens = schedule_rolling_preopens(
            queue,
            setup_at=setup_at or opens[0].wave_send,
            donor_pad_count=pad_count,
            lead_hours=rolling_lead_hours,
            afternoon_start_hour=rolling_afternoon_hour,
            pad_free_on_build_complete=pad_free_on_build_complete,
            pick_layers_per_hour=pick_layers_per_hour,
        )
    if lull_ahead_prep_enabled:
        opens = schedule_lull_ahead_preopens(
            opens,
            orders=orders,
            ship_date=ship_date,
            layers_per_hour=pick_layers_per_hour * max(1, picker_count),
            utilization_pct=lull_ahead_utilization_pct,
            max_lookahead_hours=lull_ahead_max_lookahead_hours,
            max_builds_per_hour=lull_ahead_max_builds_per_hour,
            afternoon_hour=rolling_afternoon_hour,
            pick_start_hour=pick_start_hour,
            final_shift_hour=final_shift_hour,
        )
    return opens
