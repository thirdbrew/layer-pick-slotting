"""Collect floor snapshots at setup and after each wave."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time

from layout_advisor.layout.floor_snapshot import LayerPickOverlay, WaveFloorSnapshot, extract_floor_snapshot
from layout_advisor.simulate.replay import (
    PickDayReplay,
    ReplayConfig,
    _spread_orders,
    replay_step_rows,
    run_pick_day_backtest,
)
from slotting.models import OrderLine
from slotting.backtest import HourlyStagingBoost
from slotting.wave_release import order_ship_date, split_day_by_native_due, split_day_into_wave_buckets, wave_pick_hour_range


@dataclass(frozen=True)
class WaveFloorTimeline:
    ship_date: date
    layout: str
    source_count: int
    donor_count: int
    frames: tuple[WaveFloorSnapshot, ...]
    picker_count: int = 1
    center_gap_ft: float = 0.0
    realignment_hours: tuple[int, ...] = ()
    minute_resolution: bool = False
    tick_minutes: int = 60


PEAK_HOURLY_HOURS = (16, 17, 18, 21)


def _overlay_picks_from_events(
    events: tuple,
    station_by_slot: dict[str, int] | None = None,
) -> tuple[LayerPickOverlay, ...]:
    if not events:
        return ()
    ordered = list(events)
    if station_by_slot:
        from layout_advisor.planning.wave_actions import resequence_picks_min_travel

        ordered = resequence_picks_min_travel(ordered, station_by_slot)
    return tuple(
        LayerPickOverlay(
            item_id=event.item_id,
            layers=event.layers,
            from_slot=event.from_slot,
            to_slot=event.to_slot,
            order_key=getattr(event, "order_key", None),
            to_pallet_id=getattr(event, "to_pallet_id", None),
        )
        for event in ordered
    )


def _station_by_slot_for_cfg(cfg: ReplayConfig) -> dict[str, int]:
    from layout_advisor.layout.slot_geometry import build_aisle_slot_map
    from slotting.rack_depth import rack_station_index

    _, aisle_map = build_aisle_slot_map(
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
    )
    stations = {slot_id: mapped.station for slot_id, mapped in aisle_map.items()}
    n_stations = max((m.station for m in aisle_map.values()), default=-1) + 1
    for flank in ("L", "R"):
        for bay in range(1, n_stations + 1):
            rack_id = f"R-{flank}{bay:02d}"
            stations[rack_id] = min(n_stations - 1, rack_station_index(rack_id))
    return stations


def _picks_for_step_index(
    report,
    step_index: int,
    station_by_slot: dict[str, int] | None = None,
) -> tuple[LayerPickOverlay, ...]:
    if step_index < 0 or step_index >= len(report.layer_pick_events_per_step):
        return ()
    return _overlay_picks_from_events(
        report.layer_pick_events_per_step[step_index],
        station_by_slot,
    )


def _step_index_for_as_of(report, as_of: datetime) -> int:
    key = as_of.replace(minute=0, second=0, microsecond=0)
    for index, result in enumerate(report.steps):
        if result.as_of.replace(minute=0, second=0, microsecond=0) == key:
            return index
    return -1


def _picks_for_wave_window(
    report,
    buckets: list,
    bucket,
    *,
    ship_date: date,
    final_shift_hour: int,
    station_by_slot: dict[str, int] | None = None,
) -> tuple[LayerPickOverlay, ...]:
    """Aggregate layer picks across the wave send window (send hour through next send)."""
    overlays: list[LayerPickOverlay] = []
    for hour in wave_pick_hour_range(buckets, bucket, final_shift_hour=final_shift_hour):
        as_of = datetime.combine(ship_date, time(hour))
        step_index = _step_index_for_as_of(report, as_of)
        overlays.extend(_picks_for_step_index(report, step_index, station_by_slot))
    return tuple(overlays)


def _cfg_realignment_hours(cfg: ReplayConfig) -> tuple[int, ...]:
    return cfg.realignment_hours


def _hours_between_exclusive(prev: datetime, curr: datetime) -> list[int]:
    """Whole hours strictly between two same-day snapshots (for bridge frames)."""
    if prev.date() != curr.date():
        return []
    if curr.hour - prev.hour <= 1:
        return []
    return list(range(prev.hour + 1, curr.hour))


def _source_face_items(snapshot: WaveFloorSnapshot) -> dict[str, str | None]:
    return {s.slot_id: s.item_id for s in snapshot.slots if s.zone == "source_face"}


def _face_diff(prev: WaveFloorSnapshot, curr: WaveFloorSnapshot) -> tuple[int, int, int]:
    """Return (sku_swaps, faces_cleared, faces_filled) on source faces."""
    before = _source_face_items(prev)
    after = _source_face_items(curr)
    all_slots = set(before) | set(after)
    cleared = filled = swapped = 0
    for slot_id in all_slots:
        old, new = before.get(slot_id), after.get(slot_id)
        if old and not new:
            cleared += 1
        elif not old and new:
            filled += 1
        elif old and new and old != new:
            swapped += 1
    return swapped, cleared, filled


def _bridge_label(hour: int, *, is_realignment: bool) -> str:
    if is_realignment:
        return f"Hour {hour:02d}:00 · face realignment"
    return f"Hour {hour:02d}:00: picks & completions"


def _insert_bridge_frames(
    frames: list[WaveFloorSnapshot],
    *,
    ship_date: date,
    step_by_hour: dict[datetime, object],
    report,
    realign_hours: set[int],
    station_by_slot: dict[str, int] | None = None,
) -> list[WaveFloorSnapshot]:
    """Insert hourly bridge snapshots when the scrubber would skip 2+ hours."""
    if len(frames) <= 1:
        return frames
    out: list[WaveFloorSnapshot] = [frames[0]]
    for curr in frames[1:]:
        prev = out[-1]
        for hour in _hours_between_exclusive(prev.as_of, curr.as_of):
            as_of = datetime.combine(ship_date, time(hour))
            step = step_by_hour.get(as_of)
            if step is None:
                continue
            step_index = _step_index_for_as_of(report, as_of)
            out.append(
                extract_floor_snapshot(
                    step.projected_state,
                    label=_bridge_label(hour, is_realignment=hour in realign_hours),
                    wave_index=None,
                    wave_send=as_of,
                    layer_picks=_picks_for_step_index(report, step_index, station_by_slot),
                )
            )
        out.append(curr)
    return out


def _enrich_frames(
    frames: list[WaveFloorSnapshot],
    *,
    ship_date: date,
    row_by_hour: dict[int, object],
    realign_hours: set[int],
) -> tuple[WaveFloorSnapshot, ...]:
    enriched: list[WaveFloorSnapshot] = []
    for index, frame in enumerate(frames):
        hour = frame.as_of.hour if frame.as_of.date() == ship_date else None
        row = row_by_hour.get(hour) if hour is not None else None
        gap = 0
        swaps = cleared = filled = None
        if index > 0:
            prev = frames[index - 1]
            if prev.as_of.date() == frame.as_of.date():
                gap = max(0, frame.as_of.hour - prev.as_of.hour - 1)
            swaps, cleared, filled = _face_diff(prev, frame)
        is_realign = hour in realign_hours if hour is not None else False
        enriched.append(
            replace(
                frame,
                hourly_adds=row.adds if row else 0,
                hourly_removes=row.removes if row else 0,
                hourly_donor_exits=row.donor_stages if row else 0,
                hourly_rack_promotes=row.rack_promotes if row else 0,
                is_realignment_hour=is_realign or frame.is_realignment_hour,
                gap_hours_since_prior=gap,
                face_swaps_from_prior=swaps,
                faces_cleared_from_prior=cleared,
                faces_filled_from_prior=filled,
            )
        )
    return tuple(enriched)


def _timeline_from_tick_frames(
    replay,
    *,
    ship_date: date,
    cfg: ReplayConfig,
    tick_minutes: int,
    station_by_slot: dict[str, int] | None = None,
) -> list[WaveFloorSnapshot]:
    """One floor SVG per sim step (true minute/sub-hour snapshots)."""
    report = replay.report
    initial = replay.initial
    setup_at = replay.setup_at
    frames: list[WaveFloorSnapshot] = [
        extract_floor_snapshot(
            initial,
            label=f"Setup {setup_at:%m/%d %H:%M}",
            wave_index=None,
            wave_send=setup_at,
        )
    ]
    for step_index, result in enumerate(report.steps):
        as_of = result.as_of
        if as_of <= setup_at:
            continue
        if as_of.date() > ship_date:
            continue
        if as_of.date() == ship_date and as_of.hour > cfg.final_shift_hour:
            continue
        frames.append(
            extract_floor_snapshot(
                result.projected_state,
                label=f"{as_of:%H:%M} - floor",
                wave_index=None,
                wave_send=as_of,
                layer_picks=_picks_for_step_index(report, step_index, station_by_slot),
            )
        )
    return frames


def build_wave_floor_timeline(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig | None = None,
    peak_hourly_hours: tuple[int, ...] | None = None,
    hourly_frames: bool = False,
    minute_frames: bool = False,
    minute_tick_minutes: int = 10,
    staging_pick_boost: dict[str, int] | None = None,
    carried_state=None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...] = (),
    replay: PickDayReplay | None = None,
) -> WaveFloorTimeline:
    """
    Build interactive floor frames from the full replay (dynamic pads, slotting, pulls).

    Frames: setup @ 22:00, each native wave send, bridge hours for multi-hour gaps,
    peak hourly snapshots, plus activity deltas for the scrubber.

    When *minute_frames* is True, runs replay at *minute_tick_minutes* and emits
    one snapshot per sim step (true sub-hour floor state, not UI offset).

    Pass *replay* (from ``build_daily_wave_plan_with_replay``) to reuse an
    hourly backtest already run with identical inputs instead of re-simulating.
    Ignored when *minute_frames* is True — that path needs a finer-tick replay.
    """
    cfg = cfg or ReplayConfig(resupply_rack_enabled=True)
    replay_cfg = (
        replace(cfg, sim_step_minutes=minute_tick_minutes) if minute_frames else cfg
    )
    spread = _spread_orders(orders, cfg)
    day_orders = [o for o in spread if order_ship_date(o) == ship_date]
    if not day_orders:
        raise ValueError(f"No orders on ship date {ship_date}")

    if minute_frames:
        replay = None
    if replay is None:
        replay = run_pick_day_backtest(
            orders,
            ship_date,
            cfg=replay_cfg,
            staging_pick_boost=staging_pick_boost,
            carried_state=carried_state,
            hourly_staging_boosts=hourly_staging_boosts,
        )
    report = replay.report
    initial = replay.initial
    setup_at = replay.setup_at
    station_by_slot = _station_by_slot_for_cfg(cfg)

    if minute_frames:
        frames = _timeline_from_tick_frames(
            replay,
            ship_date=ship_date,
            cfg=cfg,
            tick_minutes=minute_tick_minutes,
            station_by_slot=station_by_slot,
        )
        row_by_hour = {
            row.as_of.hour: row
            for row in replay_step_rows(replay, cfg=cfg, ship_date=ship_date)
            if row.as_of.date() == ship_date
        }
        enriched = _enrich_frames(
            frames,
            ship_date=ship_date,
            row_by_hour=row_by_hour,
            realign_hours=set(_cfg_realignment_hours(cfg)),
        )
        return WaveFloorTimeline(
            ship_date=ship_date,
            layout=f"{cfg.source_slots}:{cfg.donor_slots}",
            source_count=cfg.source_slots,
            donor_count=cfg.donor_slots,
            frames=enriched,
            picker_count=cfg.picker_count,
            center_gap_ft=cfg.center_gap_ft,
            realignment_hours=_cfg_realignment_hours(cfg),
            minute_resolution=True,
            tick_minutes=minute_tick_minutes,
        )

    realign_hours = set(_cfg_realignment_hours(cfg))

    row_by_hour = {
        row.as_of.hour: row
        for row in replay_step_rows(replay, cfg=cfg, ship_date=ship_date)
        if row.as_of.date() == ship_date
    }

    if cfg.synthetic_wave_spread:
        buckets = split_day_into_wave_buckets(
            day_orders,
            ship_date=ship_date,
            waves_per_day=cfg.waves_per_day,
            shift_start_hour=cfg.send_start_hour,
            due_offset_hours=cfg.due_offset_hours,
        )
    else:
        buckets = split_day_by_native_due(
            day_orders,
            ship_date=ship_date,
            due_offset_hours=cfg.due_offset_hours,
        )
    bucket_by_send = {b.release_at.replace(minute=0, second=0, microsecond=0): b for b in buckets}

    frames: list[WaveFloorSnapshot] = [
        extract_floor_snapshot(
            initial,
            label=f"Setup {setup_at:%m/%d %H:%M}",
            wave_index=None,
            wave_send=setup_at,
        )
    ]

    step_by_hour = {
        result.as_of.replace(minute=0, second=0, microsecond=0): result
        for result in report.steps
    }

    for result in report.steps:
        as_of = result.as_of.replace(minute=0, second=0, microsecond=0)
        if as_of.date() != ship_date:
            continue
        if cfg.synthetic_wave_spread and (
            as_of.hour < cfg.send_start_hour or as_of.hour >= cfg.send_start_hour + cfg.waves_per_day
        ):
            continue
        bucket = bucket_by_send.get(as_of)
        if bucket is None:
            continue
        step_index = _step_index_for_as_of(report, as_of)
        label = f"Wave {bucket.wave_index + 1} - send {bucket.release_at:%H:%M}"
        if as_of.hour in realign_hours:
            label += " · face realign"
        frames.append(
            extract_floor_snapshot(
                result.projected_state,
                label=label,
                wave_index=bucket.wave_index,
                wave_send=bucket.release_at,
                wave_due=bucket.due_at,
                layer_picks=_picks_for_wave_window(
                    report,
                    buckets,
                    bucket,
                    ship_date=ship_date,
                    final_shift_hour=cfg.final_shift_hour,
                    station_by_slot=station_by_slot,
                ),
            )
        )

    seen_as_of = {frame.as_of.replace(minute=0, second=0, microsecond=0) for frame in frames}
    peak_hours = PEAK_HOURLY_HOURS if peak_hourly_hours is None else peak_hourly_hours
    hours_to_add: set[int] = set(peak_hours)
    if hourly_frames:
        hours_to_add.update(range(cfg.send_start_hour, cfg.final_shift_hour + 1))

    for hour in sorted(hours_to_add):
        as_of = datetime.combine(ship_date, time(hour))
        if as_of in seen_as_of:
            continue
        step = step_by_hour.get(as_of)
        if step is None:
            continue
        step_index = _step_index_for_as_of(report, as_of)
        label = _bridge_label(hour, is_realignment=hour in realign_hours)
        frames.append(
            extract_floor_snapshot(
                step.projected_state,
                label=label,
                wave_index=None,
                wave_send=as_of,
                layer_picks=_picks_for_step_index(report, step_index, station_by_slot),
            )
        )
        seen_as_of.add(as_of)

    setup_frame = frames[0]
    rest = sorted(frames[1:], key=lambda frame: frame.as_of)
    frames = [setup_frame, *rest]
    frames = _insert_bridge_frames(
        frames,
        ship_date=ship_date,
        step_by_hour=step_by_hour,
        report=report,
        realign_hours=realign_hours,
        station_by_slot=station_by_slot,
    )
    enriched = _enrich_frames(
        frames,
        ship_date=ship_date,
        row_by_hour=row_by_hour,
        realign_hours=realign_hours,
    )

    return WaveFloorTimeline(
        ship_date=ship_date,
        layout=f"{cfg.source_slots}:{cfg.donor_slots}",
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        frames=enriched,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
        realignment_hours=_cfg_realignment_hours(cfg),
    )
