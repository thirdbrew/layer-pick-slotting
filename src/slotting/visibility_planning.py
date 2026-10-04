"""24-hour visible / 4-hour waved planning snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from slotting.abc_ranking import load_abc_by_item, setup_sort_key
from slotting.backtest import orders_for_as_of
from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.rules import VISIBLE_HORIZON, WAVED_HORIZON
from slotting.sku_master import cases_per_layer, cp_velocity_cases, has_pallet_specs
from slotting.wave_release import order_ship_date, wave_release_at


@dataclass(frozen=True)
class VisibilitySnapshot:
    """Order demand seen from *as_of* under the engine's horizons."""

    as_of: datetime
    visible_horizon: timedelta
    wave_horizon: timedelta
    visible_layers: int
    visible_skus: int
    waved_layers: int
    waved_skus: int
    visible_only_layers: int
    visible_only_skus: int
    order_lines_visible: int

    @property
    def setup_skus(self) -> int:
        """SKUs visible but not yet waved — final-shift dynamic setup targets."""
        return self.visible_only_skus

    @property
    def pick_now_skus(self) -> int:
        """SKUs in the 4-hour waved window — immediate pick/stage."""
        return self.waved_skus


def visibility_snapshot(
    orders: list[OrderLine],
    as_of: datetime,
    *,
    visible_horizon: timedelta = VISIBLE_HORIZON,
    wave_horizon: timedelta = WAVED_HORIZON,
) -> VisibilitySnapshot:
    """Summarize visible vs waved demand at a point in time."""
    visible_orders = orders_for_as_of(
        orders,
        as_of,
        wave_horizon=wave_horizon,
        visible_horizon=visible_horizon,
    )
    this_line = [o for o in visible_orders if o.area == OrderArea.THIS_LINE]

    waved = [o for o in this_line if o.status == OrderStatus.WAVED]
    visible_only = [o for o in this_line if o.status == OrderStatus.VISIBLE]

    return VisibilitySnapshot(
        as_of=as_of,
        visible_horizon=visible_horizon,
        wave_horizon=wave_horizon,
        visible_layers=sum(o.qty_layers for o in this_line),
        visible_skus=len({o.item_id for o in this_line}),
        waved_layers=sum(o.qty_layers for o in waved),
        waved_skus=len({o.item_id for o in waved}),
        visible_only_layers=sum(o.qty_layers for o in visible_only),
        visible_only_skus=len({o.item_id for o in visible_only}),
        order_lines_visible=len(this_line),
    )


def final_shift_as_of(ship_date, *, final_shift_start_hour: int = 22) -> datetime:
    """Evening before *ship_date* when shift 3 begins dynamic setup."""
    from datetime import time

    prep_day = ship_date - timedelta(days=1)
    return datetime.combine(prep_day, time(final_shift_start_hour))


def pick_day_milestones(
    ship_date,
    *,
    pick_start_hour: int = 6,
    final_shift_start_hour: int = 22,
) -> dict[str, datetime]:
    from datetime import time

    return {
        "final_shift_setup": final_shift_as_of(
            ship_date,
            final_shift_start_hour=final_shift_start_hour,
        ),
        "pick_shift_1_start": datetime.combine(ship_date, time(pick_start_hour)),
        "pick_shift_2_start": datetime.combine(ship_date, time(pick_start_hour + 8)),
        "pick_shift_3_start": datetime.combine(ship_date, time(pick_start_hour + 16)),
    }


@dataclass(frozen=True)
class SetupSkuMetrics:
    """Layer demand buckets used to rank final-shift slotting."""

    item_id: str
    total_layers: int
    total_cases: int
    waved_at_setup_layers: int
    early_wave_layers: int
    visible_at_setup_layers: int
    peak_wave_layers: int
    order_line_count: int
    cp_velocity_cases: int
    has_specs: bool
    earliest_due: datetime


def build_ship_day_setup_metrics(
    orders: list[OrderLine],
    ship_date,
    *,
    setup_as_of: datetime,
    pick_start_hour: int = 6,
    early_wave_hours: int = 6,
    due_offset_hours: float = 4.0,
    wave_horizon: timedelta = WAVED_HORIZON,
    visible_horizon: timedelta = VISIBLE_HORIZON,
) -> dict[str, SetupSkuMetrics]:
    """
    Per-SKU demand buckets for wave-aware setup ranking on *ship_date*.

    Includes waved/early/visible windows, peak single-wave spike, cases,
    and optional case-pick pickline velocity from ``skus.csv``.
    """
    from datetime import time

    waved_cutoff = setup_as_of + wave_horizon
    visible_cutoff = setup_as_of + visible_horizon
    early_cutoff = datetime.combine(ship_date, time(pick_start_hour)) + timedelta(
        hours=early_wave_hours
    )
    day_orders = [
        o
        for o in orders
        if o.area == OrderArea.THIS_LINE and order_ship_date(o) == ship_date
    ]
    wave_layers: dict[str, dict[datetime, int]] = {}
    for order in day_orders:
        release = wave_release_at(order.due_at, due_offset_hours=due_offset_hours)
        wave_layers.setdefault(order.item_id, {})
        wave_layers[order.item_id][release] = (
            wave_layers[order.item_id].get(release, 0) + order.qty_layers
        )

    by_item: dict[str, SetupSkuMetrics] = {}
    for order in day_orders:
        release = wave_release_at(order.due_at, due_offset_hours=due_offset_hours)
        waved_layers = order.qty_layers if setup_as_of <= order.due_at <= waved_cutoff else 0
        visible_layers = (
            order.qty_layers if setup_as_of <= order.due_at <= visible_cutoff else 0
        )
        early_layers = order.qty_layers if release < early_cutoff else 0
        cases = int(round(order.qty_layers * cases_per_layer(order.item_id)))
        peak = max(wave_layers.get(order.item_id, {0: 0}).values())
        existing = by_item.get(order.item_id)
        if existing is None:
            by_item[order.item_id] = SetupSkuMetrics(
                item_id=order.item_id,
                total_layers=order.qty_layers,
                total_cases=cases,
                waved_at_setup_layers=waved_layers,
                early_wave_layers=early_layers,
                visible_at_setup_layers=visible_layers,
                peak_wave_layers=peak,
                order_line_count=1,
                cp_velocity_cases=cp_velocity_cases(order.item_id),
                has_specs=has_pallet_specs(order.item_id),
                earliest_due=order.due_at,
            )
        else:
            by_item[order.item_id] = SetupSkuMetrics(
                item_id=order.item_id,
                total_layers=existing.total_layers + order.qty_layers,
                total_cases=existing.total_cases + cases,
                waved_at_setup_layers=existing.waved_at_setup_layers + waved_layers,
                early_wave_layers=existing.early_wave_layers + early_layers,
                visible_at_setup_layers=existing.visible_at_setup_layers + visible_layers,
                peak_wave_layers=peak,
                order_line_count=existing.order_line_count + 1,
                cp_velocity_cases=existing.cp_velocity_cases,
                has_specs=existing.has_specs,
                earliest_due=min(existing.earliest_due, order.due_at),
            )
    return by_item


def prioritize_wave_skus_for_staging(
    ranked: list[tuple[str, int, datetime]],
    metrics: dict[str, SetupSkuMetrics],
) -> list[tuple[str, int, datetime]]:
    """Keep wave-critical SKUs ahead of afternoon-only bulk (stable within bands)."""

    def is_wave_critical(item_id: str) -> bool:
        m = metrics[item_id]
        return m.waved_at_setup_layers > 0 or m.early_wave_layers > 0

    critical = [row for row in ranked if is_wave_critical(row[0])]
    bulk = [row for row in ranked if not is_wave_critical(row[0])]
    return critical + bulk


def _sort_key_for_metrics(
    row: tuple[str, int, datetime],
    metrics: dict[str, SetupSkuMetrics],
    *,
    abc_by_item: dict[str, str],
) -> tuple:
    m = metrics[row[0]]
    return setup_sort_key(
        row[0],
        row[1],
        row[2],
        abc_by_item=abc_by_item,
        waved_at_setup_layers=m.waved_at_setup_layers,
        early_wave_layers=m.early_wave_layers,
        visible_at_setup_layers=m.visible_at_setup_layers,
        peak_wave_layers=m.peak_wave_layers,
        order_line_count=m.order_line_count,
        total_cases=m.total_cases,
        cp_velocity_cases=m.cp_velocity_cases,
        has_specs=m.has_specs,
    )


def rank_setup_skus(
    orders: list[OrderLine],
    as_of: datetime,
    *,
    abc_by_item: dict[str, str] | None = None,
) -> list[tuple[str, int, datetime]]:
    """
    Rank SKUs for final-shift setup: layer demand first, then ABC, then earliest due.

    Returns ``(item_id, layers_in_visible_window, earliest_due)``.
    """
    if abc_by_item is None:
        abc_by_item = load_abc_by_item()
    visible_orders = orders_for_as_of(orders, as_of)
    by_item: dict[str, tuple[int, datetime]] = {}
    for order in visible_orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        layers, earliest = by_item.get(order.item_id, (0, order.due_at))
        by_item[order.item_id] = (
            layers + order.qty_layers,
            min(earliest, order.due_at),
        )
    ranked = [(item, layers, due) for item, (layers, due) in by_item.items()]
    ranked.sort(
        key=lambda row: setup_sort_key(row[0], row[1], row[2], abc_by_item=abc_by_item)
    )
    return ranked


def rank_ship_day_setup_skus(
    orders: list[OrderLine],
    ship_date,
    *,
    setup_as_of: datetime | None = None,
    pick_start_hour: int = 6,
    early_wave_hours: int = 6,
    due_offset_hours: float = 4.0,
    abc_by_item: dict[str, str] | None = None,
) -> list[tuple[str, int, datetime]]:
    """
    Rank every SKU on *ship_date* for final-shift staging.

    Wave-aware priority through cases, peak-wave spike, pickline velocity, ABC.
    """
    if abc_by_item is None:
        abc_by_item = load_abc_by_item()
    if setup_as_of is None:
        setup_as_of = final_shift_as_of(ship_date)
    metrics = build_ship_day_setup_metrics(
        orders,
        ship_date,
        setup_as_of=setup_as_of,
        pick_start_hour=pick_start_hour,
        early_wave_hours=early_wave_hours,
        due_offset_hours=due_offset_hours,
    )
    ranked = [
        (m.item_id, m.total_layers, m.earliest_due)
        for m in metrics.values()
    ]
    ranked.sort(key=lambda row: _sort_key_for_metrics(row, metrics, abc_by_item=abc_by_item))
    return prioritize_wave_skus_for_staging(ranked, metrics)


def orders_arrived_by(orders: list[OrderLine], as_of: datetime) -> list[OrderLine]:
    """Order lines known at *as_of* — missing ``arrived_at`` treated as always known."""
    return [o for o in orders if o.arrived_at is None or o.arrived_at <= as_of]


def ship_day_orders_arrived_by_setup(
    orders: list[OrderLine],
    ship_date,
    *,
    setup_as_of: datetime | None = None,
    final_shift_start_hour: int = 22,
) -> list[OrderLine]:
    """Ship-day this-line orders censored by ``arrived_at <= setup_as_of``."""
    if setup_as_of is None:
        setup_as_of = final_shift_as_of(ship_date, final_shift_start_hour=final_shift_start_hour)
    day_orders = [
        o
        for o in orders
        if o.area == OrderArea.THIS_LINE and order_ship_date(o) == ship_date
    ]
    return orders_arrived_by(day_orders, setup_as_of)


def rank_for_final_shift_setup(
    orders: list[OrderLine],
    ship_date,
    *,
    setup_as_of: datetime | None = None,
    setup_ship_day_foresight: bool = True,
    setup_arrival_censor: bool = False,
    pick_start_hour: int = 6,
    early_wave_hours: int = 6,
    due_offset_hours: float = 4.0,
    abc_by_item: dict[str, str] | None = None,
) -> list[tuple[str, int, datetime]]:
    """Setup ranking — full ship-day foresight, visible-window, or arrival censor."""
    if setup_as_of is None:
        setup_as_of = final_shift_as_of(ship_date)
    if setup_arrival_censor:
        day_orders = ship_day_orders_arrived_by_setup(
            orders,
            ship_date,
            setup_as_of=setup_as_of,
        )
        return rank_ship_day_setup_skus(
            day_orders,
            ship_date,
            setup_as_of=setup_as_of,
            pick_start_hour=pick_start_hour,
            early_wave_hours=early_wave_hours,
            due_offset_hours=due_offset_hours,
            abc_by_item=abc_by_item,
        )
    if setup_ship_day_foresight:
        return rank_ship_day_setup_skus(
            orders,
            ship_date,
            setup_as_of=setup_as_of,
            pick_start_hour=pick_start_hour,
            early_wave_hours=early_wave_hours,
            due_offset_hours=due_offset_hours,
            abc_by_item=abc_by_item,
        )
    if setup_as_of is None:
        setup_as_of = final_shift_as_of(ship_date)
    day_orders = [
        o
        for o in orders
        if o.area == OrderArea.THIS_LINE and order_ship_date(o) == ship_date
    ]
    return rank_setup_skus(day_orders, setup_as_of, abc_by_item=abc_by_item)


def affinity_orders_at_setup(
    orders: list[OrderLine],
    ship_date,
    *,
    setup_as_of: datetime | None = None,
    setup_ship_day_foresight: bool = True,
    setup_arrival_censor: bool = False,
) -> list[OrderLine]:
    """Order lines feeding setup affinity weights at final shift."""
    if setup_arrival_censor:
        if setup_as_of is None:
            setup_as_of = final_shift_as_of(ship_date)
        return ship_day_orders_arrived_by_setup(
            orders,
            ship_date,
            setup_as_of=setup_as_of,
        )
    if setup_ship_day_foresight:
        return [
            o
            for o in orders
            if o.area == OrderArea.THIS_LINE and order_ship_date(o) == ship_date
        ]
    if setup_as_of is None:
        setup_as_of = final_shift_as_of(ship_date)
    visible = orders_for_as_of(orders, setup_as_of)
    return [
        o
        for o in visible
        if o.area == OrderArea.THIS_LINE and order_ship_date(o) == ship_date
    ]


def setup_coverage(
    orders: list[OrderLine],
    as_of: datetime,
    *,
    source_slots: int,
) -> dict[str, int | float]:
    """
    How much of the visible 24h window fits on *source_slots* by layer priority.
    """
    ranked = rank_setup_skus(orders, as_of)
    visible_layers = sum(row[1] for row in ranked)
    staged_layers = 0
    staged_skus = 0
    for _item, layers, _due in ranked[:source_slots]:
        staged_skus += 1
        staged_layers += layers
    return {
        "visible_skus": len(ranked),
        "visible_layers": visible_layers,
        "staged_skus": staged_skus,
        "staged_layers": staged_layers,
        "staged_layer_pct": 100.0 * staged_layers / visible_layers if visible_layers else 0.0,
        "unstaged_skus": max(0, len(ranked) - staged_skus),
    }
