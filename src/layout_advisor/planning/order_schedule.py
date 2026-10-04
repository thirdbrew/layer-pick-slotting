"""Order-ahead scheduling — minute ticks, shift windows, build-completion pad turnover.

Parallel to the wave-based path (``wave_release.py`` / ``daily_plan.py``). Waves
remain the default dispatch unit; this module defines the order schedule
contract for the play-by-the-minute operating model.

Policy (v1):
  - Work may start ``order_lookahead_hours`` before the order due (configurable).
  - Simulation ticks every ``tick_minutes`` (default 10) within shift windows.
  - Short inventory blocks the order (``WAITING_SHORT``) until stock is available;
    set ``order_whole_order_shortage_wait=False`` to pick in-stock lines only.
    the order must still complete before ``due_at``.
  - Build pallets are filled to order demand, capped by physical layers and weight.
  - Multi-SKU build pallets share one donor pad per ``BuildPalletSlice`` (``build_lines`` on sim pallet).
  - Donor pads free on **build completion**, not wave due.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from collections.abc import Iterator, Mapping

from layout_advisor.planning.order_affinity import true_order_key
from slotting.models import OrderArea, OrderLine, OrderStatus, WorldState
from slotting.pallet_geometry import (
    MAX_ORDER_PALLET_HEIGHT_IN,
    max_build_layers_for_height,
    sku_layer_height_in,
    take_layers_for_build_pallet,
)
from slotting.sku_master import _load_rows, cases_per_layer
from slotting.throughput import item_pick_shortage_layers
from slotting.wave_release import order_ship_date


class OrderWorkStatus(str, Enum):
    """Lifecycle for a customer-order work item."""

    SCHEDULED = "scheduled"  # before start_at
    ACTIVE = "active"  # in work window, stock available
    WAITING_SHORT = "waiting_short"  # blocked on floor/reserve stock
    COMPLETE = "complete"
    LATE = "late"  # past due_at without completion


@dataclass(frozen=True)
class PickShift:
    """Named shift window on a calendar day (hours 0–23 inclusive start)."""

    name: str
    start_hour: int
    end_hour: int  # exclusive upper bound except setup wraps midnight


DEFAULT_PICK_SHIFTS: tuple[PickShift, ...] = (
    PickShift("day", 6, 14),
    PickShift("afternoon", 14, 18),
    PickShift("evening", 18, 22),
    PickShift("setup", 22, 24),
)

# GMA outbound build pallet — weight-limited cap per pallet (lb).
BUILD_PALLET_MAX_WEIGHT_LB = 2500.0


@dataclass(frozen=True)
class OrderScheduleConfig:
    """Configurable order-ahead scheduling knobs (not hardcoded)."""

    order_lookahead_hours: float = 24.0
    tick_minutes: int = 10
    pick_start_hour: int = 6
    final_shift_hour: int = 22
    final_shift_start_hour: int = 22
    max_build_weight_lb: float = BUILD_PALLET_MAX_WEIGHT_LB
    max_build_height_in: float = MAX_ORDER_PALLET_HEIGHT_IN
    default_layers_per_pallet: int = 8
    pad_free_on_build_complete: bool = True
    order_whole_order_shortage_wait: bool = True
    order_shortage_policy: str = "whole_order"  # whole_order | partial_line | late_block
    order_shortage_late_block_hours: float = 2.0
    shifts: tuple[PickShift, ...] = DEFAULT_PICK_SHIFTS

    def __post_init__(self) -> None:
        if self.order_lookahead_hours <= 0:
            raise ValueError("order_lookahead_hours must be positive")
        if self.tick_minutes <= 0 or 60 % self.tick_minutes != 0:
            raise ValueError("tick_minutes must divide 60 evenly")


@dataclass(frozen=True)
class BuildPalletSlice:
    """One build pallet's layer allocation within a customer order."""

    pallet_index: int
    lines: tuple[OrderLineDemand, ...]
    target_layers: int


@dataclass(frozen=True)
class OrderLineDemand:
    """One SKU line on a customer build."""

    order_id: str
    item_id: str
    layers: int
    due_at: datetime
    layers_per_pallet: int | None = None
    layer_weight_lb: float | None = None
    layer_height_in: float | None = None


@dataclass(frozen=True)
class OrderWorkItem:
    """One customer order in the minute-ahead schedule."""

    order_key: str
    ship_date: date
    start_at: datetime
    due_at: datetime
    lines: tuple[OrderLineDemand, ...]
    target_build_layers: int
    total_demand_layers: int
    build_pallets: tuple[BuildPalletSlice, ...] = ()
    build_pallet_count: int = 1
    layers_per_pallet_cap: int = 0

    @property
    def layer_demand(self) -> int:
        return self.total_demand_layers

    @property
    def sku_count(self) -> int:
        return len({line.item_id for line in self.lines})


@dataclass(frozen=True)
class DonorPadEpisode:
    """Donor pad occupancy: open until build completes (not wave due)."""

    donor_slot: str
    order_key: str
    open_at: datetime
    close_at: datetime


def floor_to_tick(dt: datetime, tick_minutes: int) -> datetime:
    """Floor *dt* to the nearest tick boundary."""
    minute = (dt.minute // tick_minutes) * tick_minutes
    return dt.replace(minute=minute, second=0, microsecond=0)


def tick_range(
    start: datetime,
    end: datetime,
    *,
    tick_minutes: int,
) -> Iterator[datetime]:
    """Yield tick timestamps from *start* through *end* inclusive."""
    if start > end:
        return
    step = timedelta(minutes=tick_minutes)
    current = floor_to_tick(start, tick_minutes)
    if current < start:
        current += step
    while current <= end:
        yield current
        current += step


def shift_at(dt: datetime, shifts: tuple[PickShift, ...] = DEFAULT_PICK_SHIFTS) -> PickShift | None:
    """Return the shift containing *dt*'s hour, if any."""
    hour = dt.hour
    for shift in shifts:
        if shift.start_hour <= hour < shift.end_hour:
            return shift
    return None


def order_start_at(due_at: datetime, *, lookahead_hours: float) -> datetime:
    """Earliest work start for an order (lookahead before due)."""
    return due_at - timedelta(hours=lookahead_hours)


def layer_weight_lb(item_id: str, *, skus_path: Path | None = None) -> float | None:
    """Estimated pounds per layer from SKU master (case weight × cases/layer)."""
    row = _load_rows(skus_path).get(item_id, {})
    raw_weight = row.get("weight_lb")
    try:
        case_weight = float(raw_weight)
    except (TypeError, ValueError):
        return None
    if case_weight <= 0:
        return None
    return case_weight * cases_per_layer(item_id, path=skus_path)


def layers_per_pallet_for(item_id: str, *, skus_path: Path | None = None) -> int | None:
    row = _load_rows(skus_path).get(item_id, {})
    raw = row.get("layers_per_pallet")
    try:
        val = int(float(raw))
        return val if val > 0 else None
    except (TypeError, ValueError):
        return None


def layer_height_in_for(item_id: str, *, skus_path: Path | None = None) -> float:
    return sku_layer_height_in(item_id, skus_path=skus_path)


def max_build_layers_for_order(
    lines: tuple[OrderLineDemand, ...],
    *,
    max_weight_lb: float,
    default_layers_per_pallet: int = 8,
    max_build_height_in: float = MAX_ORDER_PALLET_HEIGHT_IN,
    skus_path: Path | None = None,
) -> int:
    """
    Build-pallet layer cap: min(order demand, weight limit, physical height, 100in cap).

    When demand is below the cap the order builds to demand (cannot over-fill).
    """
    if not lines:
        return 0
    total_demand = sum(line.layers for line in lines)
    physical_cap = max(
        line.layers_per_pallet or default_layers_per_pallet for line in lines
    )
    weight_cap = total_demand
    if max_weight_lb > 0:
        weighted_layers = 0.0
        total_weight = 0.0
        for line in lines:
            w = line.layer_weight_lb
            if w is None or w <= 0:
                continue
            weighted_layers += line.layers
            total_weight += line.layers * w
        if weighted_layers > 0 and total_weight > max_weight_lb:
            avg_w = total_weight / weighted_layers
            weight_cap = int(max_weight_lb // avg_w)
    height_cap = max_build_layers_for_height(
        lines,
        max_build_height_in=max_build_height_in,
        skus_path=skus_path,
    )
    return min(total_demand, physical_cap, weight_cap, height_cap)


def split_order_into_build_pallets(
    lines: tuple[OrderLineDemand, ...],
    *,
    max_weight_lb: float,
    default_layers_per_pallet: int = 8,
    max_build_height_in: float = MAX_ORDER_PALLET_HEIGHT_IN,
    skus_path: Path | None = None,
) -> tuple[BuildPalletSlice, ...]:
    """Greedy multi-pallet split — each pallet respects weight and physical caps."""
    if not lines:
        return ()
    remaining: dict[str, int] = {line.item_id: line.layers for line in lines}
    by_id = {line.item_id: line for line in lines}
    pallets: list[BuildPalletSlice] = []
    pallet_index = 0
    while any(v > 0 for v in remaining.values()):
        active = tuple(
            replace(by_id[iid], layers=remaining[iid])
            for iid in sorted(remaining)
            if remaining[iid] > 0
        )
        cap = max_build_layers_for_order(
            active,
            max_weight_lb=max_weight_lb,
            default_layers_per_pallet=default_layers_per_pallet,
            max_build_height_in=max_build_height_in,
            skus_path=skus_path,
        )
        taken, placed = take_layers_for_build_pallet(
            active,
            layer_cap=cap,
            max_weight_lb=max_weight_lb,
            max_build_height_in=max_build_height_in,
            skus_path=skus_path,
        )
        if placed == 0:
            iid = max(remaining, key=lambda k: remaining[k])
            taken = {iid: 1}
            placed = 1
        pallet_lines = tuple(replace(by_id[iid], layers=n) for iid, n in taken.items())
        pallets.append(
            BuildPalletSlice(pallet_index=pallet_index, lines=pallet_lines, target_layers=placed)
        )
        for iid, n in taken.items():
            remaining[iid] -= n
        pallet_index += 1
    return tuple(pallets)


def order_work_status(
    item: OrderWorkItem,
    as_of: datetime,
    *,
    short: bool = False,
    complete: bool = False,
) -> OrderWorkStatus:
    """Derive order status at *as_of* given inventory and completion flags."""
    if complete:
        return OrderWorkStatus.COMPLETE
    if as_of > item.due_at:
        return OrderWorkStatus.LATE
    if short and item.start_at <= as_of <= item.due_at:
        return OrderWorkStatus.WAITING_SHORT
    if as_of < item.start_at:
        return OrderWorkStatus.SCHEDULED
    if as_of <= item.due_at:
        return OrderWorkStatus.ACTIVE
    return OrderWorkStatus.LATE


def _line_demand(line: OrderLine, *, skus_path: Path | None) -> OrderLineDemand:
    return OrderLineDemand(
        order_id=line.order_id,
        item_id=line.item_id,
        layers=line.qty_layers,
        due_at=line.due_at,
        layers_per_pallet=layers_per_pallet_for(line.item_id, skus_path=skus_path),
        layer_weight_lb=layer_weight_lb(line.item_id, skus_path=skus_path),
        layer_height_in=layer_height_in_for(line.item_id, skus_path=skus_path),
    )


def build_order_schedule(
    orders: list[OrderLine],
    ship_date: date,
    *,
    config: OrderScheduleConfig | None = None,
    skus_path: Path | None = None,
) -> list[OrderWorkItem]:
    """
    Group THIS_LINE lines into customer orders with start/due windows.

    ``start_at`` = min line due − lookahead; ``due_at`` = max line due on the order.
    """
    cfg = config or OrderScheduleConfig()
    by_key: dict[str, list[OrderLine]] = {}
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        by_key.setdefault(true_order_key(order.order_id), []).append(order)

    items: list[OrderWorkItem] = []
    for order_key, lines in sorted(by_key.items()):
        demands = tuple(_line_demand(line, skus_path=skus_path) for line in lines)
        due_at = max(line.due_at for line in lines)
        start_at = min(
            order_start_at(line.due_at, lookahead_hours=cfg.order_lookahead_hours)
            for line in lines
        )
        target = sum(d.layers for d in demands)
        layers_per_pallet_cap = max_build_layers_for_order(
            demands,
            max_weight_lb=cfg.max_build_weight_lb,
            default_layers_per_pallet=cfg.default_layers_per_pallet,
            max_build_height_in=cfg.max_build_height_in,
            skus_path=skus_path,
        )
        build_pallets = split_order_into_build_pallets(
            demands,
            max_weight_lb=cfg.max_build_weight_lb,
            default_layers_per_pallet=cfg.default_layers_per_pallet,
            max_build_height_in=cfg.max_build_height_in,
            skus_path=skus_path,
        )
        items.append(
            OrderWorkItem(
                order_key=order_key,
                ship_date=ship_date,
                start_at=start_at,
                due_at=due_at,
                lines=demands,
                target_build_layers=target,
                total_demand_layers=target,
                build_pallets=build_pallets,
                build_pallet_count=len(build_pallets),
                layers_per_pallet_cap=layers_per_pallet_cap,
            )
        )
    items.sort(key=lambda w: (w.start_at, w.due_at, w.order_key))
    return items


def pick_day_tick_window(
    ship_date: date,
    *,
    config: OrderScheduleConfig | None = None,
) -> tuple[datetime, datetime]:
    """Sim window: evening setup before ship day through final shift hour."""
    cfg = config or OrderScheduleConfig()
    setup_day = ship_date - timedelta(days=1)
    start = datetime.combine(setup_day, time(cfg.final_shift_start_hour))
    end = datetime.combine(ship_date, time(cfg.final_shift_hour))
    return start, end


def donor_episode_build_complete(
    *,
    donor_slot: str,
    order_key: str,
    open_at: datetime,
    build_complete_at: datetime,
) -> DonorPadEpisode:
    """Donor pad busy from open until build completion (not wave due)."""
    return DonorPadEpisode(
        donor_slot=donor_slot,
        order_key=order_key,
        open_at=open_at,
        close_at=max(build_complete_at, open_at),
    )


def write_order_schedule_csv(
    items: list[OrderWorkItem],
    path: Path,
    *,
    config: OrderScheduleConfig | None = None,
) -> Path:
    """Export one row per customer order for portal / analysis."""
    cfg = config or OrderScheduleConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order_key",
        "ship_date",
        "start_at",
        "due_at",
        "sku_count",
        "total_demand_layers",
        "target_build_layers",
        "build_pallet_count",
        "layers_per_pallet_cap",
        "lookahead_hours",
        "tick_minutes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "order_key": item.order_key,
                    "ship_date": item.ship_date.isoformat(),
                    "start_at": item.start_at.isoformat(sep=" "),
                    "due_at": item.due_at.isoformat(sep=" "),
                    "sku_count": item.sku_count,
                    "total_demand_layers": item.total_demand_layers,
                    "target_build_layers": item.target_build_layers,
                    "build_pallet_count": item.build_pallet_count,
                    "layers_per_pallet_cap": item.layers_per_pallet_cap,
                    "lookahead_hours": cfg.order_lookahead_hours,
                    "tick_minutes": cfg.tick_minutes,
                }
            )
    return path


def orders_active_at(
    items: list[OrderWorkItem],
    as_of: datetime,
) -> list[OrderWorkItem]:
    """Customer orders in their work window [start_at, due_at]."""
    return [item for item in items if item.start_at <= as_of <= item.due_at]


def _effective_shortage_policy(cfg: OrderScheduleConfig) -> str:
    """Resolve legacy bool flag into policy name."""
    if not cfg.order_whole_order_shortage_wait and cfg.order_shortage_policy == "whole_order":
        return "partial_line"
    return cfg.order_shortage_policy


def order_remaining_by_item(
    item: OrderWorkItem,
    picked_by_line: Mapping[tuple[str, str], int] | None = None,
) -> dict[str, int]:
    """Layers still owed on *item*, by SKU, net of what has already been picked."""
    picked = picked_by_line or {}
    remaining: dict[str, int] = {}
    for line in item.lines:
        remaining[line.item_id] = remaining.get(line.item_id, 0) + line.layers
    for item_id in list(remaining):
        remaining[item_id] = max(
            0, remaining[item_id] - picked.get((item.order_key, item_id), 0)
        )
    return remaining


def credit_picks_to_lines(
    items: list[OrderWorkItem],
    picked_by_item: Mapping[str, int],
    ledger: dict[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    """Credit picked layers to the order lines they served, earliest due first.

    The picker consumes by SKU and cannot say which customer order a layer went
    to, so attribute in the order the floor works them. Layers beyond what any
    active order still owes are dropped rather than over-crediting.
    """
    ordered_items = sorted(items, key=lambda w: (w.due_at, w.order_key))
    for item_id, layers in picked_by_item.items():
        left = layers
        for work in ordered_items:
            if left <= 0:
                break
            owed = order_remaining_by_item(work, ledger).get(item_id, 0)
            if owed <= 0:
                continue
            take = min(owed, left)
            key = (work.order_key, item_id)
            ledger[key] = ledger.get(key, 0) + take
            left -= take
    return ledger


def pick_demand_respecting_order_waits(
    state: WorldState,
    orders: list[OrderLine],
    ship_date: date,
    *,
    config: OrderScheduleConfig | None = None,
    picked_by_line: Mapping[tuple[str, str], int] | None = None,
    schedule: list[OrderWorkItem] | None = None,
) -> dict[str, int]:
    """
    Layer pick demand by SKU for active orders, respecting shortage policy.

    Policies:
      - ``whole_order``: any short line blocks the entire order (default).
      - ``partial_line``: pick in-stock lines only.
      - ``late_block``: partial-line until ``due_at - late_block_hours``, then whole-order.

    *picked_by_line* maps ``(order_key, item_id)`` to layers already picked for
    that order line. Demand is what is LEFT: without it the same line is
    re-requested on every tick of the ship day and the floor is drawn down past
    the book.
    """
    cfg = config or OrderScheduleConfig()
    policy = _effective_shortage_policy(cfg)
    if schedule is None:
        schedule = build_order_schedule(orders, ship_date, config=cfg)
    demand: dict[str, int] = {}
    for item in orders_active_at(schedule, state.as_of):
        remaining = order_remaining_by_item(item, picked_by_line)
        owed = {iid: n for iid, n in remaining.items() if n > 0}
        if not owed:
            continue  # order complete — nothing left to pick
        hours_to_due = (item.due_at - state.as_of).total_seconds() / 3600.0
        use_whole_order = policy == "whole_order" or (
            policy == "late_block" and hours_to_due <= cfg.order_shortage_late_block_hours
        )
        if use_whole_order:
            # require_build_face=False: this asks whether SUPPLY exists, not
            # whether the SKU is pickable this instant. The donor build pad is
            # opened as part of starting the order, so gating the start on the
            # pad already existing is unsatisfiable — it blocked nearly every
            # tick of the day. Same reasoning
            # donor_builds.py applies to the pad-open gate.
            if any(
                item_pick_shortage_layers(state, iid, n, require_build_face=False) > 0
                for iid, n in owed.items()
            ):
                continue
            for item_id, layers in owed.items():
                demand[item_id] = demand.get(item_id, 0) + layers
        else:
            for item_id, layers in owed.items():
                short = item_pick_shortage_layers(
                    state, item_id, layers, require_build_face=False
                )
                pickable = max(0, layers - short)
                if pickable > 0:
                    demand[item_id] = demand.get(item_id, 0) + pickable
    return demand


def active_order_line_demand(
    state: WorldState,
    orders: list[OrderLine],
    ship_date: date,
    *,
    config: OrderScheduleConfig | None = None,
    schedule: list[OrderWorkItem] | None = None,
) -> dict[str, int]:
    """Full layer demand by SKU for every active order line (no shortage wait).

    Used for the operator pick-list emission path. Inventory mutation under
    ``whole_order`` stays on :func:`pick_demand_respecting_order_waits`; the plan
    still needs pick rows ≈ active book even while orders wait for complete stock.
    """
    cfg = config or OrderScheduleConfig()
    if schedule is None:
        schedule = build_order_schedule(orders, ship_date, config=cfg)
    demand: dict[str, int] = {}
    for item in orders_active_at(schedule, state.as_of):
        for line in item.lines:
            demand[line.item_id] = demand.get(line.item_id, 0) + line.layers
    return demand


def operator_plan_pick_demand(
    state: WorldState,
    orders: list[OrderLine],
    ship_date: date,
    *,
    emitted_by_item: dict[str, int],
    config: OrderScheduleConfig | None = None,
    schedule: list[OrderWorkItem] | None = None,
) -> dict[str, int]:
    """Residual active-order demand not yet stamped onto the operator pick list."""
    active = active_order_line_demand(
        state, orders, ship_date, config=config, schedule=schedule
    )
    demand: dict[str, int] = {}
    for item_id, total in active.items():
        left = total - emitted_by_item.get(item_id, 0)
        if left > 0:
            demand[item_id] = left
    return demand


def orders_for_order_schedule(
    orders: list[OrderLine],
    as_of: datetime,
    ship_date: date,
    *,
    config: OrderScheduleConfig | None = None,
) -> list[OrderLine]:
    """
    Derive line status from order work windows (replaces 4h waved bucket when enabled).

    Lines on active orders are WAVED; lines visible within lookahead but not yet active
    are VISIBLE; lines outside the lookahead window are omitted.
    """
    cfg = config or OrderScheduleConfig()
    horizon = timedelta(hours=cfg.order_lookahead_hours)
    schedule_by_key = {item.order_key: item for item in build_order_schedule(orders, ship_date, config=cfg)}
    derived: list[OrderLine] = []
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        if order.due_at < as_of or order.due_at > as_of + horizon:
            continue
        key = true_order_key(order.order_id)
        work = schedule_by_key.get(key)
        if work is None:
            continue
        if work.start_at <= as_of <= work.due_at:
            status = OrderStatus.WAVED
        elif as_of < work.start_at:
            status = OrderStatus.VISIBLE
        else:
            continue
        derived.append(
            OrderLine(
                order_id=order.order_id,
                item_id=order.item_id,
                qty_layers=order.qty_layers,
                due_at=order.due_at,
                status=status,
                area=order.area,
            )
        )
    return derived


@dataclass(frozen=True)
class OrderDonorOpen:
    """Scheduled donor build for one SKU line on a customer-order build pallet."""

    order_key: str
    pallet_index: int
    line_index: int
    scheduled_open: datetime
    build_due: datetime
    item_id: str
    layers: int
    source_slot: str | None = None

    @property
    def open_key(self) -> str:
        return f"{self.order_key}|{self.pallet_index}|{self.line_index}|{self.item_id}"


def build_order_donor_opens(
    orders: list[OrderLine],
    ship_date: date,
    setup_assignments: list,
    *,
    config: OrderScheduleConfig | None = None,
) -> list[OrderDonorOpen]:
    """
    One donor open per SKU line on each build pallet slice.

    Opens fire at the order ``start_at`` (tick-aligned). Donor pads are chosen
    dynamically at apply time so pads freed by build completion can be reused.
    """
    cfg = config or OrderScheduleConfig()
    schedule = build_order_schedule(orders, ship_date, config=cfg)
    source_by_item = {
        row.item_id: row.slot_id
        for row in setup_assignments
        if getattr(row, "location", None) in {"source_face", "rack"}
        and getattr(row, "slot_id", None)
    }
    opens: list[OrderDonorOpen] = []
    for item in schedule:
        open_at = floor_to_tick(item.start_at, cfg.tick_minutes)
        for pallet in item.build_pallets:
            for line_index, line in enumerate(pallet.lines):
                opens.append(
                    OrderDonorOpen(
                        order_key=item.order_key,
                        pallet_index=pallet.pallet_index,
                        line_index=line_index,
                        scheduled_open=open_at,
                        build_due=item.due_at,
                        item_id=line.item_id,
                        layers=line.layers,
                        source_slot=source_by_item.get(line.item_id),
                    )
                )
    opens.sort(key=lambda row: (row.scheduled_open, row.order_key, row.pallet_index, row.line_index))
    return opens
