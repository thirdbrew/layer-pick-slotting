"""Hourly (or N-per-day) wave release models for layout sizing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache

from slotting.models import OrderArea, OrderLine


@lru_cache(maxsize=None)
def _ship_date_from_order_id(order_id: str) -> date | None:
    tail = order_id.rsplit("-", 1)[-1]
    if len(tail) == 8 and tail.isdigit():
        return date(int(tail[:4]), int(tail[4:6]), int(tail[6:8]))
    return None


def order_ship_date(order: OrderLine) -> date:
    """Calendar ship day from the order id suffix ``...-YYYYMMDD``."""
    parsed = _ship_date_from_order_id(order.order_id)
    return parsed if parsed is not None else order.due_at.date()


def wave_release_at(due_at: datetime, *, due_offset_hours: float = 4.0) -> datetime:
    """Infer WMS send time from native due (floored to the hour)."""
    release = due_at - timedelta(hours=due_offset_hours)
    return release.replace(minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class WaveBucket:
    """One released wave on a ship day."""

    ship_date: date
    wave_index: int
    release_at: datetime
    due_at: datetime
    orders: tuple[OrderLine, ...]

    @property
    def layer_demand(self) -> int:
        return sum(o.qty_layers for o in self.orders)

    @property
    def sku_count(self) -> int:
        return len({o.item_id for o in self.orders})


@dataclass(frozen=True)
class HourlyWaveDemand:
    """Peak staging demand under an hourly (or N-per-day) release cadence."""

    waves_per_day: int
    shift_start_hour: int
    due_offset_hours: float
    peak_ship_date: date
    peak_day_layers: int
    peak_day_skus: int
    max_layers_per_wave: int
    max_skus_per_wave: int
    pick_hours_per_wave: float
    sla_layers_clearable: int
    largest_wave_meets_sla: bool
    in_flight_waves: int
    peak_in_flight_layers: int
    peak_in_flight_skus: int
    peak_due_window_layers: int
    peak_due_window_skus: int
    waves_in_peak_due_window: int
    source_slots_single_wave: int
    donor_slots_single_wave: int
    source_slots_in_flight: int
    donor_slots_in_flight: int
    source_slots_due_window: int
    donor_slots_due_window: int


def _wave_release_datetime(ship_date: date, *, shift_start_hour: int, wave_index: int) -> datetime:
    hour = shift_start_hour + wave_index
    day = ship_date
    while hour >= 24:
        hour -= 24
        day += timedelta(days=1)
    return datetime.combine(day, time(hour))


def bucket_orders_by_ship_day(orders: list[OrderLine]) -> dict[date, list[OrderLine]]:
    by_day: dict[date, list[OrderLine]] = defaultdict(list)
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        by_day[order_ship_date(order)].append(order)
    return dict(by_day)


def split_day_into_wave_buckets(
    day_orders: list[OrderLine],
    *,
    ship_date: date,
    waves_per_day: int,
    shift_start_hour: int,
    due_offset_hours: float = 0.0,
) -> list[WaveBucket]:
    """Balance layer demand across wave buckets (greedy min-load)."""
    sorted_lines = sorted(day_orders, key=lambda o: (-o.qty_layers, o.item_id))
    buckets: list[list[OrderLine]] = [[] for _ in range(waves_per_day)]
    totals = [0] * waves_per_day
    for order in sorted_lines:
        idx = min(range(waves_per_day), key=lambda i: totals[i])
        buckets[idx].append(order)
        totals[idx] += order.qty_layers

    offset = timedelta(hours=due_offset_hours)
    return [
        WaveBucket(
            ship_date=ship_date,
            wave_index=i,
            release_at=_wave_release_datetime(ship_date, shift_start_hour=shift_start_hour, wave_index=i),
            due_at=_wave_release_datetime(ship_date, shift_start_hour=shift_start_hour, wave_index=i) + offset,
            orders=tuple(buckets[i]),
        )
        for i in range(waves_per_day)
    ]


def split_day_by_native_due(
    day_orders: list[OrderLine],
    *,
    ship_date: date,
    due_offset_hours: float = 4.0,
) -> list[WaveBucket]:
    """
    Group orders into hourly waves using each line's ``due_at``.

    Infers wave send as ``due_at - due_offset_hours`` (floored to the hour).
    """
    if not day_orders:
        return []
    by_release: dict[datetime, list[OrderLine]] = defaultdict(list)
    for order in day_orders:
        release = wave_release_at(order.due_at, due_offset_hours=due_offset_hours)
        by_release[release].append(order)

    buckets: list[WaveBucket] = []
    for wave_index, release_at in enumerate(sorted(by_release)):
        orders = tuple(by_release[release_at])
        due_at = max(o.due_at for o in orders)
        buckets.append(
            WaveBucket(
                ship_date=ship_date,
                wave_index=wave_index,
                release_at=release_at,
                due_at=due_at,
                orders=orders,
            )
        )
    return buckets


def spread_orders_to_hourly_waves(
    orders: list[OrderLine],
    *,
    waves_per_day: int = 12,
    shift_start_hour: int = 6,
    due_offset_hours: float = 0.0,
) -> list[OrderLine]:
    """
    Spread each ship day into hourly releases.

    ``due_at`` on each line is ``release_at + due_offset_hours`` (0 = due when sent).
    """
    spread: list[OrderLine] = []
    for ship_date, day_orders in sorted(bucket_orders_by_ship_day(orders).items()):
        for bucket in split_day_into_wave_buckets(
            day_orders,
            ship_date=ship_date,
            waves_per_day=waves_per_day,
            shift_start_hour=shift_start_hour,
            due_offset_hours=due_offset_hours,
        ):
            for order in bucket.orders:
                spread.append(
                    OrderLine(
                        order_id=order.order_id,
                        item_id=order.item_id,
                        qty_layers=order.qty_layers,
                        due_at=bucket.due_at,
                        status=order.status,
                        area=order.area,
                    )
                )
    return spread


def _donor_hint(sku_count: int) -> int:
    return max(1, (sku_count + 2) // 3)


def _in_flight_peak(buckets: list[WaveBucket], *, pick_hours_per_wave: float) -> tuple[int, int]:
    """Peak floor staging while waves sent hourly are still being picked."""
    peak_layers = 0
    peak_skus = 0
    horizon = timedelta(hours=pick_hours_per_wave)
    for tick in range(len(buckets)):
        release_at = buckets[0].release_at + timedelta(hours=tick)
        active: list[OrderLine] = []
        for bucket in buckets:
            if bucket.release_at <= release_at < bucket.release_at + horizon:
                active.extend(bucket.orders)
        peak_layers = max(peak_layers, sum(o.qty_layers for o in active))
        peak_skus = max(peak_skus, len({o.item_id for o in active}))
    return peak_layers, peak_skus


def _due_window_peak(
    buckets: list[WaveBucket],
    *,
    due_horizon_hours: float,
) -> tuple[int, int, int]:
    """
    Peak demand in a rolling due window (matches the rules engine's ``WAVED_HORIZON`` scoring).

    Uses half-open ``[as_of, as_of + horizon)`` on ``due_at``.
    """
    if not buckets:
        return 0, 0, 0

    horizon = timedelta(hours=due_horizon_hours)
    peak_layers = 0
    peak_skus = 0
    peak_wave_count = 0
    for as_of in sorted({b.due_at for b in buckets}):
        cutoff = as_of + horizon
        window_buckets = [b for b in buckets if as_of <= b.due_at < cutoff]
        orders = [o for b in window_buckets for o in b.orders]
        layers = sum(o.qty_layers for o in orders)
        skus = len({o.item_id for o in orders})
        if layers > peak_layers:
            peak_layers = layers
            peak_wave_count = len(window_buckets)
        peak_skus = max(peak_skus, skus)
    return peak_layers, peak_skus, peak_wave_count


def compute_hourly_wave_demand(
    orders: list[OrderLine],
    *,
    waves_per_day: int = 12,
    shift_start_hour: int = 6,
    due_offset_hours: float = 4.0,
    layers_per_hour_picker: float,
    source_buffer_slots: int = 2,
) -> HourlyWaveDemand:
    """
    Size aisle pads for hourly wave cadence on historical demand.

    Waves are **sent** every hour; ``due_at`` is ``release + due_offset_hours``.
    *layers_per_hour_picker* is the serial clamp pick rate (one picker line).
    """
    by_day = bucket_orders_by_ship_day(orders)
    if not by_day:
        return HourlyWaveDemand(
            waves_per_day=waves_per_day,
            shift_start_hour=shift_start_hour,
            due_offset_hours=due_offset_hours,
            peak_ship_date=date.today(),
            peak_day_layers=0,
            peak_day_skus=0,
            max_layers_per_wave=0,
            max_skus_per_wave=0,
            pick_hours_per_wave=0.0,
            sla_layers_clearable=0,
            largest_wave_meets_sla=False,
            in_flight_waves=0,
            peak_in_flight_layers=0,
            peak_in_flight_skus=0,
            peak_due_window_layers=0,
            peak_due_window_skus=0,
            waves_in_peak_due_window=0,
            source_slots_single_wave=source_buffer_slots,
            donor_slots_single_wave=1,
            source_slots_in_flight=source_buffer_slots,
            donor_slots_in_flight=1,
            source_slots_due_window=source_buffer_slots,
            donor_slots_due_window=1,
        )

    peak_ship_date = max(by_day, key=lambda d: sum(o.qty_layers for o in by_day[d]))
    day_orders = by_day[peak_ship_date]
    buckets = split_day_into_wave_buckets(
        day_orders,
        ship_date=peak_ship_date,
        waves_per_day=waves_per_day,
        shift_start_hour=shift_start_hour,
        due_offset_hours=due_offset_hours,
    )

    peak_day_layers = sum(o.qty_layers for o in day_orders)
    peak_day_skus = len({o.item_id for o in day_orders})
    max_layers_per_wave = max(b.layer_demand for b in buckets)
    max_skus_per_wave = max(b.sku_count for b in buckets)

    pick_hours = max_layers_per_wave / max(layers_per_hour_picker, 1.0)
    sla_layers = int(layers_per_hour_picker * due_offset_hours)
    in_flight_waves = max(1, int(pick_hours) + (1 if pick_hours % 1 > 1e-9 else 0))
    peak_in_flight_layers, peak_in_flight_skus = _in_flight_peak(buckets, pick_hours_per_wave=pick_hours)
    peak_due_layers, peak_due_skus, peak_due_wave_count = _due_window_peak(
        buckets,
        due_horizon_hours=due_offset_hours,
    )

    return HourlyWaveDemand(
        waves_per_day=waves_per_day,
        shift_start_hour=shift_start_hour,
        due_offset_hours=due_offset_hours,
        peak_ship_date=peak_ship_date,
        peak_day_layers=peak_day_layers,
        peak_day_skus=peak_day_skus,
        max_layers_per_wave=max_layers_per_wave,
        max_skus_per_wave=max_skus_per_wave,
        pick_hours_per_wave=pick_hours,
        sla_layers_clearable=sla_layers,
        largest_wave_meets_sla=max_layers_per_wave <= sla_layers,
        in_flight_waves=in_flight_waves,
        peak_in_flight_layers=peak_in_flight_layers,
        peak_in_flight_skus=peak_in_flight_skus,
        peak_due_window_layers=peak_due_layers,
        peak_due_window_skus=peak_due_skus,
        waves_in_peak_due_window=peak_due_wave_count,
        source_slots_single_wave=max(1, max_skus_per_wave + source_buffer_slots),
        donor_slots_single_wave=_donor_hint(max_skus_per_wave),
        source_slots_in_flight=max(1, peak_in_flight_skus + source_buffer_slots),
        donor_slots_in_flight=_donor_hint(peak_in_flight_skus),
        source_slots_due_window=max(1, peak_due_skus + source_buffer_slots),
        donor_slots_due_window=_donor_hint(peak_due_skus),
    )


def peak_spread_wave_window(
    spread_orders: list[OrderLine],
    *,
    wave_horizon: timedelta,
) -> tuple[int, int, datetime | None]:
    """
    Peak layers/SKUs in a single release window on spread orders.

    Uses a half-open interval ``[as_of, as_of + horizon)`` so adjacent hourly
    releases are not double-counted.
    """
    this_line = [o for o in spread_orders if o.area == OrderArea.THIS_LINE]
    if not this_line:
        return 0, 0, None

    starts = sorted({o.due_at for o in this_line})
    peak_layers = 0
    peak_skus = 0
    peak_start: datetime | None = None
    for as_of in starts:
        cutoff = as_of + wave_horizon
        window = [o for o in this_line if as_of <= o.due_at < cutoff]
        layers = sum(o.qty_layers for o in window)
        skus = len({o.item_id for o in window})
        if layers > peak_layers:
            peak_layers = layers
            peak_start = as_of
        peak_skus = max(peak_skus, skus)
    return peak_layers, peak_skus, peak_start


def active_wave_bucket(
    buckets: list[WaveBucket],
    as_of: datetime,
) -> WaveBucket | None:
    """Latest wave whose send hour is at or before *as_of* (hour precision)."""
    hour = as_of.replace(minute=0, second=0, microsecond=0)
    active: WaveBucket | None = None
    for bucket in sorted(buckets, key=lambda b: b.release_at):
        send = bucket.release_at.replace(minute=0, second=0, microsecond=0)
        if send <= hour:
            active = bucket
        else:
            break
    return active


def wave_pick_hour_range(
    buckets: list[WaveBucket],
    bucket: WaveBucket,
    *,
    final_shift_hour: int,
) -> range:
    """Pick hours ``[start, end)`` attributed to one wave send window."""
    start_hour = bucket.release_at.hour
    later = sorted(
        (b for b in buckets if b.release_at > bucket.release_at),
        key=lambda b: b.release_at,
    )
    end_hour = later[0].release_at.hour if later else final_shift_hour + 1
    return range(start_hour, end_hour)
