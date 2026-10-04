"""Historical backtest — measure rule impact over time with real order history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum

from slotting.models import OrderLine, OrderStatus, Pallet, Slot, SlotType, WorldState
from slotting.rules import VISIBLE_HORIZON, WAVED_HORIZON
from slotting.simulation import SimulationResult, simulate_once
from slotting.rules import DEFAULT_RULE_CONFIG, RuleConfig


class BacktestMode(str, Enum):
    """
    How floor state evolves across backtest steps.

    STATIC: Each step uses the same initial pallets/slots (orders shift only).
            Answers: "At each point in history, how much would one round of rules help?"

    CHAINED: After each step, the projected (rules-applied) floor becomes the next baseline.
             Answers: "If operators had followed rules every day, how would KPIs trend?"
    """

    STATIC = "static"
    CHAINED = "chained"


@dataclass(frozen=True)
class ShiftPullConfig:
    """Mid-shift pallet pulls from upstream storage when floor stock runs out."""

    ship_date: date
    orders: list[OrderLine]
    max_per_sku: int = 2
    max_per_sku_late_shift: int = 3
    late_shift_cap_min_layers: int = 6
    max_per_hour_day: int = 4
    max_per_hour_afternoon: int = 6
    max_per_hour_evening: int = 10
    max_per_hour_default: int = 6
    default_layers: int = 8
    include_rack_aisle: bool = True
    resupply_threshold: int | None = None
    stage_after_pull: bool = True
    preferred_faces: dict[str, str] = field(default_factory=dict)
    late_shift_hour: int = 18
    final_shift_hour: int = 22
    # late-shift residual starve (a known sim failure mode): a small residual
    # (< cap-min) after late_shift_hour keeps the day cap, which may already
    # be consumed — the SKU's last wave then starves with zero pickable
    # supply. When enabled, grant ONE extra pull in exactly that state.
    late_residual_rescue: bool = False

    def max_per_hour_at(self, hour: int) -> int:
        if hour >= self.late_shift_hour:
            return self.max_per_hour_evening
        if hour >= 14:
            return self.max_per_hour_afternoon
        if hour >= 6:
            return self.max_per_hour_day
        return self.max_per_hour_default

    def max_per_sku_for(self, item_id: str, *, hour: int | None = None) -> int:
        from slotting.inventory_policy import _late_shift_layers_for_item

        late = _late_shift_layers_for_item(
            self.orders,
            item_id,
            self.ship_date,
            late_shift_hour=self.late_shift_hour,
            final_shift_hour=self.final_shift_hour,
        )
        if late >= self.late_shift_cap_min_layers:
            return self.max_per_sku_late_shift
        if (
            self.late_residual_rescue
            and hour is not None
            and hour >= self.late_shift_hour
            and late > 0
        ):
            return self.max_per_sku + 1
        return self.max_per_sku


def return_unconsumed_idle_pallets(pallets, slots):
    """End-of-day freezer return for idle-assist staging (intra-shift only).

    Unconsumed IDLE pallets not on a source face never enter carry — staged
    leftovers in carry can blow up the next day's waits even when that day has
    no idle activity of its own. Mutates *slots* to vacate returned pallets;
    returns the surviving pallet list.
    """
    on_face = {
        s.occupied_by_pallet_id
        for s in slots
        if s.slot_type == SlotType.SOURCE and s.occupied_by_pallet_id
    }
    returned_ids = {
        p.pallet_id
        for p in pallets
        if p.pallet_id.startswith("IDLE-") and p.pallet_id not in on_face
    }
    if not returned_ids:
        return pallets
    for slot in slots:
        if slot.occupied_by_pallet_id in returned_ids:
            slot.occupied_by_pallet_id = None
    return [p for p in pallets if p.pallet_id not in returned_ids]


@dataclass(frozen=True)
class IdleAssistConfig:
    """Opportunistic freezer→rack staging when reactive pull budget is unused.

    Models an assist-role pattern: when the assistant is not busy with shortage
    pulls, bring the *next* pallet into the aisle (prefer rack) so the picker
    keeps throughput without waiting for a WAIT.
    """

    ship_date: date
    orders: list[OrderLine]
    max_per_hour: int = 2
    face_layer_threshold: int = 4
    default_layers: int = 8
    max_per_sku: int = 3
    prefer_rack: bool = True
    include_rack_aisle: bool = True
    # rack headroom kept free for displacement / promotes (0 = no guard)
    min_free_rack_slots: int = 0
    # Only spend when reactive shortage pulls left unused hourly capacity.
    require_unused_pull_budget: bool = True


@dataclass(frozen=True)
class HourlyStagingBoost:
    """Inject staging pallets at a specific hour (e.g. 17:00 late-shift prep)."""

    ship_date: date
    hour: int
    boost_by_item: dict[str, int]
    default_layers: int = 8
    prefix: str = "LATE"


@dataclass(frozen=True)
class BacktestReport:
    """Aggregated results from replaying rules over a historical period."""

    mode: BacktestMode
    start: datetime
    end: datetime
    step: timedelta
    steps: list[SimulationResult]
    final_state: WorldState | None = None
    shift_pulls_per_step: tuple[int, ...] = ()
    shift_pull_events_per_step: tuple[tuple[tuple[str, str, str | None], ...], ...] = ()
    layer_pick_events_per_step: tuple[tuple, ...] = ()
    donor_pad_events: tuple = ()
    # Layers taken off face pallets by the inventory-mutating pick path. NOT the
    # same as summing layer_pick_events_per_step: under the order schedule those
    # events come from the operator plan path, which picks against throwaway
    # copies. Only this number tracks floor depletion.
    layers_picked: int = 0
    # (order_key, item_id) -> layers picked for that order line. Empty on the
    # wave path, which has no order ledger. Lets callers ask which ORDERS
    # completed; ``layers_picked`` only answers whether the floor moved.
    picked_by_line: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def avg_baseline_dead_slot_rate_pct(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.baseline.dead_slot_rate for s in self.steps) / len(self.steps) * 100

    @property
    def avg_projected_dead_slot_rate_pct(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.projected.dead_slot_rate for s in self.steps) / len(self.steps) * 100

    @property
    def avg_improvement_pp(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.dead_slot_rate_improvement_pct for s in self.steps) / len(self.steps)

    @property
    def total_improvement_pp(self) -> float:
        return sum(s.dead_slot_rate_improvement_pct for s in self.steps)

    @property
    def steps_with_dead_slot_improvement(self) -> int:
        return sum(1 for s in self.steps if s.dead_slot_rate_improvement_pct > 0)

    @property
    def total_shortage_reduction(self) -> int:
        return sum(s.shortage_improvement for s in self.steps)

    @property
    def total_removes_applied(self) -> int:
        return sum(s.removes_applied for s in self.steps)

    @property
    def total_adds_applied(self) -> int:
        return sum(s.adds_applied for s in self.steps)

    @property
    def avg_baseline_layer_pick_pct(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.baseline_throughput.layer_pick_pct for s in self.steps) / len(self.steps) * 100

    @property
    def avg_projected_layer_pick_pct(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.projected_throughput.layer_pick_pct for s in self.steps) / len(self.steps) * 100

    @property
    def avg_layer_pick_improvement_pp(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.layer_pick_improvement_pp for s in self.steps) / len(self.steps)

    @property
    def total_layers_shifted_to_layer_pick(self) -> int:
        return sum(s.layers_shifted_to_layer_pick for s in self.steps)

    @property
    def avg_baseline_wave_layer_pick_pct(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.baseline_wave_layer_pick_pct for s in self.steps) / len(self.steps) * 100

    @property
    def avg_projected_wave_layer_pick_pct(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.wave_layer_pick_pct for s in self.steps) / len(self.steps) * 100

    @property
    def avg_wave_layer_pick_improvement_pp(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.wave_layer_pick_improvement_pp for s in self.steps) / len(self.steps)

    @property
    def total_wave_layers_shifted_to_layer_pick(self) -> int:
        return sum(s.wave_layers_shifted_to_layer_pick for s in self.steps)

    @property
    def final_unmanaged_dead_slot_rate_pct(self) -> float | None:
        """Last step baseline dead slot rate (do-nothing path in static mode)."""
        if not self.steps:
            return None
        return self.steps[-1].baseline.dead_slot_rate * 100

    @property
    def final_managed_dead_slot_rate_pct(self) -> float | None:
        """Last step projected dead slot rate after rules (chained end state)."""
        if not self.steps:
            return None
        return self.steps[-1].projected.dead_slot_rate * 100

    @property
    def cumulative_dead_slot_gap_pp(self) -> float | None:
        """
        Difference between final unmanaged vs managed dead slot rate (chained mode).

        Positive => sustained rule-following ends with fewer dead slots.
        """
        if self.mode != BacktestMode.CHAINED or not self.steps:
            return None
        first_baseline = self.steps[0].baseline.dead_slot_rate * 100
        last_projected = self.steps[-1].projected.dead_slot_rate * 100
        return first_baseline - last_projected


def derive_order_status(
    order: OrderLine,
    as_of: datetime,
    *,
    wave_horizon: timedelta = WAVED_HORIZON,
) -> OrderStatus:
    """Waved if due within *wave_horizon* of *as_of*, else visible."""
    if as_of <= order.due_at <= as_of + wave_horizon:
        return OrderStatus.WAVED
    return OrderStatus.VISIBLE


def _derive_order_status(order: OrderLine, as_of: datetime) -> OrderStatus:
    return derive_order_status(order, as_of)


def orders_for_as_of(
    orders: list[OrderLine],
    as_of: datetime,
    *,
    wave_horizon: timedelta = WAVED_HORIZON,
    visible_horizon: timedelta = VISIBLE_HORIZON,
) -> list[OrderLine]:
    """Keep orders in the visible window with status derived from *as_of*."""
    cutoff = as_of + visible_horizon
    derived: list[OrderLine] = []
    for order in orders:
        if order.due_at < as_of or order.due_at > cutoff:
            continue
        derived.append(
            OrderLine(
                order_id=order.order_id,
                item_id=order.item_id,
                qty_layers=order.qty_layers,
                due_at=order.due_at,
                status=derive_order_status(order, as_of, wave_horizon=wave_horizon),
                area=order.area,
            )
        )
    return derived


def _orders_for_as_of(orders: list[OrderLine], as_of: datetime) -> list[OrderLine]:
    return orders_for_as_of(orders, as_of)


def _state_at(
    *,
    pallets: list[Pallet],
    slots: list[Slot],
    orders: list[OrderLine],
    exclusions: list,
    as_of: datetime,
) -> WorldState:
    return WorldState(
        pallets=[p.model_copy(deep=True) for p in pallets],
        slots=[s.model_copy(deep=True) for s in slots],
        orders=orders,
        exclusions=exclusions,
        as_of=as_of,
    )


def backtest_historical(
    initial: WorldState,
    *,
    start: datetime,
    end: datetime,
    step: timedelta = timedelta(hours=1),
    mode: BacktestMode = BacktestMode.CHAINED,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
    pad_role_swaps: list | None = None,
    pad_role_restore_on_starve: bool = False,
    wave_donor_opens: list | None = None,
    rolling_donor_queue: list | None = None,
    rolling_layer_threshold: int | None = None,
    pad_turnover_afternoon_hour: int = 12,
    pad_turnover_lead_hours: int = 1,
    inventory_layers_per_hour: float | None = None,
    pick_shift_hours: tuple[int, int] | None = None,
    inventory_late_shift_retain: bool = True,
    inventory_late_shift_hour: int = 18,
    inventory_final_shift_hour: int = 22,
    realignment_hour: int = 0,
    realignment_hours: list[int] | None = None,
    realignment_fn=None,
    rule_config_refresh_fn=None,
    shift_pull: ShiftPullConfig | None = None,
    idle_assist: IdleAssistConfig | None = None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...] = (),
    dynamic_evening_prep_hours: tuple[int, ...] = (),
    dynamic_evening_prep_fn=None,
    evening_carry_hours: tuple[int, ...] = (),
    evening_carry_fn=None,
    order_schedule_enabled: bool = False,
    order_schedule_ship_date: date | None = None,
    order_schedule_config=None,
    order_donor_opens: list | None = None,
    order_supply_complete_gate: bool = False,
    order_gate_due_override_hours: float = 0.0,
    donor_open_retry_enabled: bool = True,
) -> BacktestReport:
    """
    Replay rules across a date range using historical orders in *initial.orders*.

    At each step, order visibility windows are evaluated relative to *as_of*
    (same 24h visible / 4h waved logic as production rules).

    Parameters
    ----------
    initial:
        Floor snapshot (pallets, slots, exclusions) plus the full historical
        ``orders`` list (all due dates in the backtest window).
    mode:
        ``CHAINED`` carries forward simulated slot moves; ``STATIC`` resets
        the floor to the initial snapshot each step.
    """
    if start > end:
        raise ValueError("start must be before or equal to end")

    all_orders = list(initial.orders)
    exclusions = list(initial.exclusions)
    static_pallets = [p.model_copy(deep=True) for p in initial.pallets]
    static_slots = [s.model_copy(deep=True) for s in initial.slots]

    managed_pallets = [p.model_copy(deep=True) for p in initial.pallets]
    managed_slots = [s.model_copy(deep=True) for s in initial.slots]

    results: list[SimulationResult] = []
    shift_pulls_log: list[int] = []
    shift_events_log: list[tuple[tuple[str, str, str | None], ...]] = []
    layer_pick_log: list[tuple] = []
    donor_pad_log: list = []
    pulled_by_item: dict[str, int] = {}
    # Layers the MUTATION pick path actually took off face pallets. The operator
    # plan path below picks against throwaway deep copies, so its volume says
    # nothing about floor depletion — keep the two apart.
    layers_picked_total = 0
    # Residual tracker for operator pick-list emission under order schedule
    # (see operator_plan_pick_demand). Independent of whole_order inventory.
    _operator_pick_emitted: dict[str, int] = {}
    # Per-order-line progress: (order_key, item_id) -> layers picked. Demand is
    # what is LEFT on the order; without this the same line is re-requested on
    # every tick and the floor is drawn down past the book.
    picked_by_line: dict[tuple[str, str], int] = {}
    order_schedule = None  # built lazily on first pick tick, then reused
    current = start
    applied_wave_open_items: set[str] = set()
    applied_order_open_keys: set[str] = set()
    from layout_advisor.planning.donor_pad_sim import (
        destination_slot_ids_from_state,
        log_donor_closes_from_applied,
        log_donor_opens,
        wave_index_by_item,
    )

    destination_slot_ids = destination_slot_ids_from_state(initial)
    wave_item_index = wave_index_by_item(wave_donor_opens or [])
    setup_preopen: list[tuple[str, int, str]] = []
    for slot in initial.slots:
        if slot.slot_type != SlotType.DESTINATION or not slot.occupied_by_pallet_id:
            continue
        pallet = next((p for p in initial.pallets if p.pallet_id == slot.occupied_by_pallet_id), None)
        if pallet is None:
            continue
        setup_preopen.append((pallet.item_id, max(pallet.layers_remaining, 1), slot.slot_id))
    if setup_preopen:
        log_donor_opens(
            donor_pad_log,
            at=start,
            batch=setup_preopen,
            wave_index_by_item=wave_item_index,
            detail="setup_preopen",
        )
    realign_hours = (
        list(realignment_hours)
        if realignment_hours
        else ([realignment_hour] if realignment_hour else [])
    )
    from slotting.schedule_ticks import (
        consume_whole_layers_budget,
        hour_in_schedule,
        is_hour_boundary,
        layers_budget_for_step,
    )

    def _orders_at(as_of: datetime) -> list[OrderLine]:
        if order_schedule_enabled and order_schedule_ship_date is not None:
            from layout_advisor.planning.order_schedule import orders_for_order_schedule

            return orders_for_order_schedule(
                all_orders,
                as_of,
                order_schedule_ship_date,
                config=order_schedule_config,
            )
        return _orders_for_as_of(all_orders, as_of)

    pick_budget_remainder = 0.0
    pick_rate = float(inventory_layers_per_hour) if inventory_layers_per_hour else 0.0

    while current <= end:
        if mode == BacktestMode.STATIC:
            pallets, slots = static_pallets, static_slots
        else:
            pallets, slots = managed_pallets, managed_slots

        if (
            realign_hours
            and hour_in_schedule(current, realign_hours, step)
            and realignment_fn is not None
            and mode == BacktestMode.CHAINED
        ):
            realign_state = _state_at(
                pallets=pallets,
                slots=slots,
                orders=_orders_at(current),
                exclusions=exclusions,
                as_of=current,
            )
            realigned = realignment_fn(realign_state)
            managed_pallets = [p.model_copy(deep=True) for p in realigned.pallets]
            managed_slots = [s.model_copy(deep=True) for s in realigned.slots]
            pallets, slots = managed_pallets, managed_slots

        hour_pulls = 0
        hour_events: tuple[tuple[str, str, str | None], ...] = ()
        if hourly_staging_boosts and mode == BacktestMode.CHAINED and is_hour_boundary(current, step):
            for boost in hourly_staging_boosts:
                if boost.hour != current.hour or not boost.boost_by_item:
                    continue
                from slotting.inventory import append_staging_pick_boost

                pallets = append_staging_pick_boost(
                    pallets,
                    all_orders,
                    boost.ship_date,
                    boost_by_item=boost.boost_by_item,
                    as_of=current,
                    default_layers=boost.default_layers,
                    prefix=boost.prefix,
                )
                managed_pallets = pallets

        if (
            dynamic_evening_prep_hours
            and hour_in_schedule(current, dynamic_evening_prep_hours, step)
            and dynamic_evening_prep_fn is not None
            and mode == BacktestMode.CHAINED
        ):
            pallets = dynamic_evening_prep_fn(pallets, slots, all_orders, current)
            managed_pallets = pallets

        if (
            evening_carry_hours
            and hour_in_schedule(current, evening_carry_hours, step)
            and evening_carry_fn is not None
            and mode == BacktestMode.CHAINED
        ):
            pallets, slots = evening_carry_fn(pallets, slots, current)
            managed_pallets = pallets
            managed_slots = slots

        step_state = _state_at(
            pallets=pallets,
            slots=slots,
            orders=_orders_at(current),
            exclusions=exclusions,
            as_of=current,
        )
        if pad_role_swaps:
            from layout_advisor.planning.pad_role_schedule import apply_pad_role_swaps

            step_state, _ = apply_pad_role_swaps(
                step_state,
                pad_role_swaps,
                restore_on_starve=pad_role_restore_on_starve,
            )
        if order_donor_opens:
            from layout_advisor.simulate.donor_builds import apply_order_donor_opens_at_step

            step_state = apply_order_donor_opens_at_step(
                step_state,
                order_donor_opens,
                as_of=current,
                step=step,
                applied_keys=applied_order_open_keys,
                require_supply_complete=order_supply_complete_gate,
                due_override_hours=order_gate_due_override_hours,
            )
        elif wave_donor_opens:
            from layout_advisor.simulate.donor_builds import apply_wave_donor_opens_at_step

            step_state, staged = apply_wave_donor_opens_at_step(
                step_state,
                wave_donor_opens,
                as_of=current,
                step=step,
                applied_item_ids=applied_wave_open_items,
                retry_failed=donor_open_retry_enabled,
            )
            if staged:
                log_donor_opens(
                    donor_pad_log,
                    at=current,
                    batch=staged,
                    wave_index_by_item=wave_item_index,
                    detail="wave_open",
                )
        if (
            rolling_donor_queue is not None
            and rolling_layer_threshold is not None
            and is_hour_boundary(current, step)
        ):
            from layout_advisor.simulate.donor_builds import apply_pad_turnover_queue
            from slotting.visibility_planning import visibility_snapshot

            snap = visibility_snapshot(all_orders, current)
            before_turnover = {
                s.slot_id
                for s in step_state.slots
                if s.slot_id in destination_slot_ids and s.occupied_by_pallet_id
            }
            step_state = apply_pad_turnover_queue(
                step_state,
                rolling_donor_queue,
                applied_item_ids=applied_wave_open_items,
                afternoon_hour=pad_turnover_afternoon_hour,
                min_waved_layers=rolling_layer_threshold,
                waved_layers=snap.waved_layers,
                lead_hours=pad_turnover_lead_hours,
            )
            turnover_batch: list[tuple[str, int, str]] = []
            for slot in step_state.slots:
                if slot.slot_id not in destination_slot_ids or not slot.occupied_by_pallet_id:
                    continue
                if slot.slot_id in before_turnover:
                    continue
                pallet = next(
                    (p for p in step_state.pallets if p.pallet_id == slot.occupied_by_pallet_id),
                    None,
                )
                if pallet is None:
                    continue
                turnover_batch.append(
                    (pallet.item_id, max(pallet.layers_remaining, 1), slot.slot_id)
                )
            if turnover_batch:
                log_donor_opens(
                    donor_pad_log,
                    at=current,
                    batch=turnover_batch,
                    wave_index_by_item=wave_item_index,
                    detail="pad_turnover",
                )
        if shift_pull is not None and mode == BacktestMode.CHAINED and is_hour_boundary(current, step):
            from slotting.inventory import apply_shift_pallet_pulls

            new_pallets, new_slots, hour_pulls, pulled_by_item, events = apply_shift_pallet_pulls(
                pallets,
                slots,
                step_state,
                shift_pull.orders,
                shift_pull.ship_date,
                pulled_today_by_item=pulled_by_item,
                max_per_hour=shift_pull.max_per_hour_at(current.hour),
                max_per_sku_for_item=(
                    lambda item_id, _h=current.hour: shift_pull.max_per_sku_for(
                        item_id, hour=_h
                    )
                ),
                default_layers=shift_pull.default_layers,
                include_rack_aisle=shift_pull.include_rack_aisle,
                resupply_threshold=shift_pull.resupply_threshold,
                stage_after_pull=shift_pull.stage_after_pull,
                preferred_face_by_item=shift_pull.preferred_faces or None,
            )
            hour_events = tuple(events)
            if hour_pulls:
                pallets = new_pallets
                slots = new_slots
                managed_pallets = new_pallets
                managed_slots = new_slots
                step_state = _state_at(
                    pallets=pallets,
                    slots=slots,
                    orders=_orders_at(current),
                    exclusions=exclusions,
                    as_of=current,
                )
            if idle_assist is not None:
                from slotting.inventory import apply_idle_assist_staging

                reactive_cap = shift_pull.max_per_hour_at(current.hour)
                unused = max(0, reactive_cap - hour_pulls)
                idle_budget = idle_assist.max_per_hour
                if idle_assist.require_unused_pull_budget:
                    idle_budget = min(idle_budget, unused)
                if idle_budget > 0:
                    (
                        idle_pallets,
                        idle_slots,
                        idle_n,
                        pulled_by_item,
                        idle_events,
                    ) = apply_idle_assist_staging(
                        pallets,
                        slots,
                        step_state,
                        idle_assist.orders,
                        idle_assist.ship_date,
                        pulled_today_by_item=pulled_by_item,
                        max_per_hour=idle_budget,
                        max_per_sku=idle_assist.max_per_sku,
                        face_layer_threshold=idle_assist.face_layer_threshold,
                        default_layers=idle_assist.default_layers,
                        prefer_rack=idle_assist.prefer_rack,
                        include_rack_aisle=idle_assist.include_rack_aisle,
                        min_free_rack_slots=idle_assist.min_free_rack_slots,
                    )
                    if idle_n:
                        pallets = idle_pallets
                        slots = idle_slots
                        managed_pallets = idle_pallets
                        managed_slots = idle_slots
                        hour_pulls += idle_n
                        hour_events = hour_events + tuple(idle_events)
                        step_state = _state_at(
                            pallets=pallets,
                            slots=slots,
                            orders=_orders_at(current),
                            exclusions=exclusions,
                            as_of=current,
                        )
        shift_pulls_log.append(hour_pulls)
        shift_events_log.append(hour_events)
        if (
            rule_config_refresh_fn is not None
            and mode == BacktestMode.CHAINED
            and is_hour_boundary(current, step)
        ):
            refreshed = rule_config_refresh_fn(current, rule_config)
            if refreshed is not None:
                rule_config = refreshed
        result = simulate_once(step_state, rule_config=rule_config)
        log_donor_closes_from_applied(
            donor_pad_log,
            at=current,
            applied=result.applied,
            destination_slot_ids=destination_slot_ids,
        )
        results.append(result)

        if mode == BacktestMode.CHAINED:
            managed_pallets = [p.model_copy(deep=True) for p in result.projected_state.pallets]
            managed_slots = [s.model_copy(deep=True) for s in result.projected_state.slots]
            pick_on_shift = (
                pick_shift_hours is None
                or pick_shift_hours[0] <= current.hour < pick_shift_hours[1]
            )
            pick_budget = 0
            if pick_rate > 0 and pick_on_shift:
                pick_budget, pick_budget_remainder = consume_whole_layers_budget(
                    layers_budget_for_step(pick_rate, step),
                    pick_budget_remainder,
                )
            if pick_budget > 0 and pick_on_shift:
                from slotting.inventory import apply_hourly_layer_picks, finalize_depleted_pallets

                pick_state = _state_at(
                    pallets=managed_pallets,
                    slots=managed_slots,
                    orders=_orders_at(current),
                    exclusions=exclusions,
                    as_of=current,
                )
                hour_pick_events: list = []
                if order_schedule_enabled and order_schedule_ship_date is not None:
                    from layout_advisor.planning.order_schedule import (
                        build_order_schedule,
                        credit_picks_to_lines,
                        operator_plan_pick_demand,
                        orders_active_at,
                        pick_demand_respecting_order_waits,
                    )

                    # Orders, ship date and config are fixed for the run, so the
                    # schedule is too — build it once (226 ms a pop, and it was
                    # being rebuilt 2-3x on every one of ~145 ticks).
                    if order_schedule is None:
                        order_schedule = build_order_schedule(
                            all_orders,
                            order_schedule_ship_date,
                            config=order_schedule_config,
                        )
                    # Inventory mutation follows shortage policy (whole_order by
                    # default). Events for the operator CSV / portal come from the
                    # residual active-order plan path below — not from waits.
                    pick_demand = pick_demand_respecting_order_waits(
                        pick_state,
                        all_orders,
                        order_schedule_ship_date,
                        config=order_schedule_config,
                        picked_by_line=picked_by_line,
                        schedule=order_schedule,
                    )
                    tick_picks: dict[str, int] = {}
                    layers_picked_total += apply_hourly_layer_picks(
                        managed_pallets,
                        managed_slots,
                        pick_state,
                        layers_per_hour=pick_budget,
                        late_shift_retain=inventory_late_shift_retain,
                        late_shift_hour=inventory_late_shift_hour,
                        final_shift_hour=inventory_final_shift_hour,
                        pick_events=None,
                        demand_by_item=pick_demand,
                        picked_by_item=tick_picks,
                    )
                    if tick_picks:
                        credit_picks_to_lines(
                            orders_active_at(order_schedule, current),
                            tick_picks,
                            picked_by_line,
                        )
                    end_state = _state_at(
                        pallets=managed_pallets,
                        slots=managed_slots,
                        orders=_orders_at(current),
                        exclusions=exclusions,
                        as_of=current,
                    )
                    managed_pallets = finalize_depleted_pallets(
                        managed_pallets, managed_slots, state=end_state
                    )

                    plan_state = _state_at(
                        pallets=managed_pallets,
                        slots=managed_slots,
                        orders=_orders_at(current),
                        exclusions=exclusions,
                        as_of=current,
                    )
                    plan_demand = operator_plan_pick_demand(
                        plan_state,
                        all_orders,
                        order_schedule_ship_date,
                        emitted_by_item=_operator_pick_emitted,
                        config=order_schedule_config,
                        schedule=order_schedule,
                    )
                    if plan_demand:
                        plan_pallets = [p.model_copy(deep=True) for p in managed_pallets]
                        plan_slots = [s.model_copy(deep=True) for s in managed_slots]
                        apply_hourly_layer_picks(
                            plan_pallets,
                            plan_slots,
                            plan_state,
                            layers_per_hour=pick_budget,
                            late_shift_retain=inventory_late_shift_retain,
                            late_shift_hour=inventory_late_shift_hour,
                            final_shift_hour=inventory_final_shift_hour,
                            pick_events=hour_pick_events,
                            demand_by_item=plan_demand,
                        )
                        for event in hour_pick_events:
                            _operator_pick_emitted[event.item_id] = (
                                _operator_pick_emitted.get(event.item_id, 0) + event.layers
                            )
                else:
                    layers_picked_total += apply_hourly_layer_picks(
                        managed_pallets,
                        managed_slots,
                        pick_state,
                        layers_per_hour=pick_budget,
                        late_shift_retain=inventory_late_shift_retain,
                        late_shift_hour=inventory_late_shift_hour,
                        final_shift_hour=inventory_final_shift_hour,
                        pick_events=hour_pick_events,
                        demand_by_item=None,
                    )
                    end_state = _state_at(
                        pallets=managed_pallets,
                        slots=managed_slots,
                        orders=_orders_at(current),
                        exclusions=exclusions,
                        as_of=current,
                    )
                    managed_pallets = finalize_depleted_pallets(
                        managed_pallets, managed_slots, state=end_state
                    )
                layer_pick_log.append(tuple(hour_pick_events))
            else:
                layer_pick_log.append(())
        else:
            layer_pick_log.append(())

        current += step

    final_state: WorldState | None = None
    if mode == BacktestMode.CHAINED:
        if idle_assist is not None:
            managed_pallets = return_unconsumed_idle_pallets(
                managed_pallets, managed_slots
            )
        final_state = _state_at(
            pallets=managed_pallets,
            slots=managed_slots,
            orders=all_orders,
            exclusions=exclusions,
            as_of=end,
        )

    return BacktestReport(
        mode=mode,
        start=start,
        end=end,
        step=step,
        steps=results,
        final_state=final_state,
        shift_pulls_per_step=tuple(shift_pulls_log),
        shift_pull_events_per_step=tuple(shift_events_log),
        layer_pick_events_per_step=tuple(layer_pick_log),
        donor_pad_events=tuple(donor_pad_log),
        layers_picked=layers_picked_total,
        picked_by_line=picked_by_line,
    )


def backtest_snapshot_dir(
    snapshots: list[tuple[datetime, WorldState]],
    *,
    mode: BacktestMode = BacktestMode.STATIC,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> BacktestReport:
    """
    Backtest from a list of timestamped world states (gold standard).

    Each tuple is ``(as_of, WorldState)`` from a JDA export folder per shift/day.
    """
    if not snapshots:
        raise ValueError("snapshots list is empty")

    ordered = sorted(snapshots, key=lambda t: t[0])
    start = ordered[0][0]
    end = ordered[-1][0]
    step = end - start if len(ordered) == 1 else ordered[1][0] - ordered[0][0]

    if mode == BacktestMode.CHAINED:
        raise ValueError(
            "CHAINED mode requires a single initial floor with historical orders; "
            "use backtest_historical or run STATIC mode on snapshot dirs."
        )

    results = [simulate_once(state, rule_config=rule_config) for _, state in ordered]
    return BacktestReport(mode=mode, start=start, end=end, step=step, steps=results)
