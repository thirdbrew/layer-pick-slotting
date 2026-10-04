"""Required throughput profile derived from order history (no floor state)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from slotting.jda_columns import ORDER_ALIASES, resolve_columns
from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.rules import WAVED_HORIZON
from slotting.wave_release import order_ship_date


@dataclass(frozen=True)
class RequiredThroughputProfile:
    """
    Planning targets for Mode A layout sweep.

    Derived from orders alone — no pallets, slots, or simulation required.
    """

    peak_wave_layer_demand: int
    peak_wave_window_start: datetime | None
    peak_concurrent_skus: int
    layers_per_day_p95: float
    min_layer_pick_ready_pct: float
    min_source_slots_hint: int
    min_donor_slots_hint: int
    order_count: int
    period_days: int
    #: Share of ship days whose orders all carry ONE due timestamp. Every field
    #: above is derived by sliding a 4h window over `due_at`, so on such a day
    #: the window swallows the whole book and "peak wave" IS the day total.
    single_wave_days_pct: float = 0.0
    mean_due_times_per_day: float = 0.0

    @property
    def wave_structure_degenerate(self) -> bool:
        """True when the feed has no intra-day wave structure to find.

        An order feed whose due column is a DATE (no time of day) gets every
        order stamped with `default_wave_pick_time` on normalisation, so each
        ship day collapses to a single wave even when the same orders carry
        native ship times elsewhere.

        On such a feed `peak_wave_layer_demand` is a DAY total and
        `min_source_slots_hint` sizes for the entire day's SKUs arriving at once,
        which can be more than double the faces a real wave cadence needs.
        """
        return self.single_wave_days_pct >= 0.5

    @property
    def wave_structure_message(self) -> str | None:
        if not self.wave_structure_degenerate:
            return None
        return (
            f"NO WAVE STRUCTURE: {self.single_wave_days_pct:.0%} of ship days carry a "
            f"single due timestamp ({self.mean_due_times_per_day:.1f} distinct times per "
            f"day). A 4h wave window therefore spans the whole day, so "
            f"peak_wave_layer_demand, peak_concurrent_skus and min_source_slots_hint are "
            f"DAY totals, not wave peaks: sizing on them oversizes the room by roughly "
            f"the number of real waves. The order feed needs due TIMES; a feed carrying "
            f"only ship dates cannot answer a wave question."
        )


_REQUIRED_ORDER_ALIASES = {
    "order_id": ORDER_ALIASES["order_id"],
    "item_id": ORDER_ALIASES["item_id"],
    "qty_layers": ORDER_ALIASES["qty_layers"],
    "due_at": ORDER_ALIASES["due_at"],
}


def _optional_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    for name in aliases:
        if name in df.columns:
            return df[name]
    return None


def _load_orders(path: Path) -> list[OrderLine]:
    df = pd.read_csv(path, dtype=str)
    cols = resolve_columns(df, _REQUIRED_ORDER_ALIASES)
    status_series = _optional_series(df, ORDER_ALIASES["status"])
    area_series = _optional_series(df, ORDER_ALIASES["area"])
    arrived_series = _optional_series(df, ORDER_ALIASES["arrived_at"])

    # One vectorized datetime parse. The old per-row pd.to_datetime() re-guessed
    # the format on every row and dominated a pick-day sim's runtime on large feeds.
    try:
        due_ats = pd.to_datetime(cols["due_at"])
    except ValueError:
        due_ats = pd.to_datetime(cols["due_at"], format="mixed")
    due_list = [ts.to_pydatetime() for ts in due_ats]

    order_ids = [str(v).strip() for v in cols["order_id"]]
    item_ids = [str(v).strip() for v in cols["item_id"]]
    qty_list = [int(float(v)) for v in cols["qty_layers"]]

    if status_series is not None:
        statuses = []
        for raw in status_series:
            try:
                statuses.append(OrderStatus(str(raw).strip().lower()))
            except ValueError:
                statuses.append(OrderStatus.VISIBLE)
    else:
        statuses = [OrderStatus.VISIBLE] * len(df)

    if area_series is not None:
        areas = [OrderArea(str(v).strip()) for v in area_series]
    else:
        areas = [OrderArea.THIS_LINE] * len(df)

    if arrived_series is not None:
        try:
            arrived_parsed = pd.to_datetime(arrived_series)
        except ValueError:
            arrived_parsed = pd.to_datetime(arrived_series, format="mixed")
        arrived_list = [
            None if pd.isna(ts) else ts.to_pydatetime() for ts in arrived_parsed
        ]
    else:
        arrived_list = [None] * len(df)

    return [
        OrderLine(
            order_id=oid,
            item_id=iid,
            qty_layers=qty,
            due_at=due,
            status=status,
            area=area,
            arrived_at=arrived,
        )
        # strict=True: seven parallel columns off one DataFrame, so they cannot
        # legitimately differ in length. Without it a short column silently
        # TRUNCATES the order book - the sim would run on fewer orders than the
        # feed holds and every KPI would be quietly wrong but plausible. Raising
        # is the only safe behaviour for a loader.
        for oid, iid, qty, due, status, area, arrived in zip(
            order_ids, item_ids, qty_list, due_list, statuses, areas, arrived_list,
            strict=True,
        )
    ]


def load_orders(path: Path) -> list[OrderLine]:
    """Load ``orders.csv`` with engine or WMS-export column aliases."""
    return _load_orders(path)


def _this_line_orders(orders: list[OrderLine]) -> list[OrderLine]:
    return [o for o in orders if o.area == OrderArea.THIS_LINE]


def _wave_window_candidates(
    orders: list[OrderLine],
    *,
    wave_horizon: timedelta,
) -> list[datetime]:
    """Candidate wave start times that can maximize demand in *wave_horizon*."""
    if not orders:
        return []
    starts = {o.due_at for o in orders}
    starts.update(o.due_at - wave_horizon for o in orders)
    return sorted(starts)


def _layers_in_wave_window(
    orders: list[OrderLine],
    as_of: datetime,
    *,
    wave_horizon: timedelta,
) -> int:
    cutoff = as_of + wave_horizon
    return sum(o.qty_layers for o in orders if as_of <= o.due_at <= cutoff)


def _skus_in_wave_window(
    orders: list[OrderLine],
    as_of: datetime,
    *,
    wave_horizon: timedelta,
) -> int:
    cutoff = as_of + wave_horizon
    return len({o.item_id for o in orders if as_of <= o.due_at <= cutoff})


def _layers_per_day_p95(orders: list[OrderLine]) -> float:
    if not orders:
        return 0.0
    by_day: dict[date, int] = defaultdict(int)
    for order in orders:
        by_day[order_ship_date(order)] += order.qty_layers
    daily = sorted(by_day.values())
    if len(daily) == 1:
        return float(daily[0])
    rank = 0.95 * (len(daily) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(daily) - 1)
    weight = rank - lower
    return daily[lower] * (1 - weight) + daily[upper] * weight


def compute_required_throughput(
    orders_path: Path,
    *,
    min_layer_pick_ready_pct: float = 0.85,
    source_buffer_slots: int = 2,
    wave_horizon: timedelta | None = None,
) -> RequiredThroughputProfile:
    """
    Derive Mode A sweep targets from historical order demand.

    Parameters
    ----------
    orders_path:
        ``orders.csv`` with engine or WMS-export column names.
    min_layer_pick_ready_pct:
        Target wave layer-pick-ready % for the smallest feasible layout.
    source_buffer_slots:
        Added to peak concurrent SKUs for a lower-bound source slot hint.
    wave_horizon:
        Rolling wave window for peak demand. Defaults to ``WAVED_HORIZON`` (4h).
        Override via CLI ``--wave-hours`` when ops cadence is known (e.g. ~2h).
    """
    horizon = wave_horizon or WAVED_HORIZON
    orders = _load_orders(Path(orders_path))
    this_line = _this_line_orders(orders)

    peak_layers = 0
    peak_window_start: datetime | None = None
    peak_skus = 0

    for as_of in _wave_window_candidates(this_line, wave_horizon=horizon):
        window_layers = _layers_in_wave_window(this_line, as_of, wave_horizon=horizon)
        window_skus = _skus_in_wave_window(this_line, as_of, wave_horizon=horizon)
        if window_layers > peak_layers:
            peak_layers = window_layers
            peak_window_start = as_of
        peak_skus = max(peak_skus, window_skus)

    dates = [o.due_at.date() for o in this_line]
    period_days = max(1, (max(dates) - min(dates)).days + 1) if dates else 0

    # Wave structure. Every peak_* field above slides a window over due_at, so a
    # feed whose orders share one timestamp per day reports the day as its peak.
    due_times_by_day: dict[date, set[datetime]] = {}
    for order in this_line:
        due_times_by_day.setdefault(order.due_at.date(), set()).add(order.due_at)
    if due_times_by_day:
        per_day = [len(times) for times in due_times_by_day.values()]
        single_wave_pct = sum(1 for c in per_day if c <= 1) / len(per_day)
        mean_due_times = sum(per_day) / len(per_day)
    else:
        single_wave_pct = 0.0
        mean_due_times = 0.0

    min_source = max(1, peak_skus + source_buffer_slots)
    min_donor = max(1, (peak_skus + 2) // 3)

    return RequiredThroughputProfile(
        peak_wave_layer_demand=peak_layers,
        peak_wave_window_start=peak_window_start,
        peak_concurrent_skus=peak_skus,
        layers_per_day_p95=_layers_per_day_p95(this_line),
        min_layer_pick_ready_pct=min_layer_pick_ready_pct,
        min_source_slots_hint=min_source,
        min_donor_slots_hint=min_donor,
        order_count=len(this_line),
        period_days=period_days,
        single_wave_days_pct=round(single_wave_pct, 4),
        mean_due_times_per_day=round(mean_due_times, 2),
    )


def compute_required_throughput_from_dir(
    data_dir: Path,
    *,
    min_layer_pick_ready_pct: float = 0.85,
    source_buffer_slots: int = 2,
) -> RequiredThroughputProfile:
    """Load ``orders.csv`` from an engine data directory."""
    path = Path(data_dir) / "orders.csv"
    return compute_required_throughput(
        path,
        min_layer_pick_ready_pct=min_layer_pick_ready_pct,
        source_buffer_slots=source_buffer_slots,
    )
