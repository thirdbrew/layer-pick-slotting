"""Throughput / labor mix — layer pick vs hand pick demand."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from slotting.models import OrderArea, OrderStatus, WorldState
from slotting.pallet_geometry import face_lp_pickable_layers_for_item, lp_pickable_layers_on_pallet
from slotting.rules import (
    VISIBLE_HORIZON,
    WAVED_HORIZON,
    _eligible_unslotted_pallets,
    _held_pallet_ids,
    _this_line_orders_in_window,
)


@dataclass(frozen=True)
class ThroughputMetrics:
    """
    Layer demand split by fulfillment path (labor proxy).

    Percentages are of total layer demand in the evaluated window.
    """

    as_of: datetime
    total_layer_demand: int
    layer_pick_ready_layers: int
    hand_pick_unstaged_layers: int
    hand_pick_shortage_layers: int
    hand_pick_excluded_layers: int
    low_layer_slotted_layers: int

    @property
    def layer_pick_pct(self) -> float:
        if self.total_layer_demand == 0:
            return 0.0
        return self.layer_pick_ready_layers / self.total_layer_demand

    @property
    def hand_pick_pct(self) -> float:
        if self.total_layer_demand == 0:
            return 0.0
        return self.hand_pick_total_layers / self.total_layer_demand

    @property
    def hand_pick_total_layers(self) -> int:
        return (
            self.hand_pick_unstaged_layers
            + self.hand_pick_shortage_layers
            + self.hand_pick_excluded_layers
            + self.low_layer_slotted_layers
        )

    @property
    def fixable_by_rules_layers(self) -> int:
        """Demand that ADD/REMOVE rules can shift toward layer pick."""
        return self.hand_pick_unstaged_layers + self.low_layer_slotted_layers

    def as_row(self, prefix: str = "") -> dict[str, str | float | int]:
        p = prefix
        return {
            f"{p}total_layer_demand": self.total_layer_demand,
            f"{p}layer_pick_pct": round(self.layer_pick_pct * 100, 2),
            f"{p}hand_pick_pct": round(self.hand_pick_pct * 100, 2),
            f"{p}layer_pick_ready_layers": self.layer_pick_ready_layers,
            f"{p}hand_unstaged_layers": self.hand_pick_unstaged_layers,
            f"{p}hand_shortage_layers": self.hand_pick_shortage_layers,
            f"{p}hand_excluded_layers": self.hand_pick_excluded_layers,
            f"{p}low_layer_slotted_layers": self.low_layer_slotted_layers,
            f"{p}fixable_by_rules_layers": self.fixable_by_rules_layers,
        }


def _demand_by_item(state: WorldState) -> dict[str, int]:
    demand: dict[str, int] = {}
    for order in _this_line_orders_in_window(state, VISIBLE_HORIZON):
        if order.area != OrderArea.THIS_LINE:
            continue
        demand[order.item_id] = demand.get(order.item_id, 0) + order.qty_layers
    return demand


def _waved_demand_by_item(state: WorldState, horizon: timedelta = WAVED_HORIZON) -> dict[str, int]:
    """Layer demand from waved orders due within *horizon* of *state.as_of*."""
    cutoff = state.as_of + horizon
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


def _waved_demand_by_item_for_ship_date(
    state: WorldState,
    ship_date: date,
    horizon: timedelta = WAVED_HORIZON,
) -> dict[str, int]:
    """Waved demand limited to orders on *ship_date* (excludes next-day wave bleed)."""
    from slotting.wave_release import order_ship_date

    cutoff = state.as_of + horizon
    demand: dict[str, int] = {}
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.status != OrderStatus.WAVED:
            continue
        if order_ship_date(order) != ship_date:
            continue
        if not (state.as_of <= order.due_at <= cutoff):
            continue
        demand[order.item_id] = demand.get(order.item_id, 0) + order.qty_layers
    return demand


def item_pick_shortage_layers(
    state: WorldState,
    item_id: str,
    layers: int,
    *,
    require_build_face: bool = True,
) -> int:
    """Layers of *layers* demand for *item_id* that cannot be layer-picked now.

    ``require_build_face=True`` answers "is this pickable *right now*" — it needs
    an open donor build listing the SKU. Callers gating whether work may *start*
    must pass False: the build pad is opened as part of starting the order, so
    demanding it already exist is unsatisfiable (see the same reasoning at
    ``donor_builds.py`` for the pad-open gate).
    """
    if item_id in state.excluded_item_ids:
        return 0
    metrics = _classify_demand(
        state, {item_id: layers}, require_build_face=require_build_face
    )
    return (
        metrics.hand_pick_shortage_layers
        + metrics.low_layer_slotted_layers
        + metrics.hand_pick_unstaged_layers
    )


def _swap_ready_pickable_layers_for_item(
    state: WorldState,
    item_id: str,
    *,
    allow_aisle_staging: bool = False,
    include_rack_aisle: bool = False,
) -> tuple[int, int]:
    """
    LP-pickable and total layers on backup/rack supply reachable via in-aisle swap.

    Used when demand is not yet on a pick face but ``item_swap_ready`` is true.
    """
    from slotting.backup_row import plan_backup_swap
    from slotting.models import SlotType

    pickable = 0
    total = 0
    for slot in state.source_slots:
        if not slot.backup_pallet_id:
            continue
        backup = state.pallet_by_id(slot.backup_pallet_id)
        if backup is None or backup.item_id != item_id:
            continue
        if not state.is_eligible_for_line(backup):
            continue
        plan = plan_backup_swap(
            state,
            slot,
            waved_items={item_id},
            allow_storage_fallback=False,
            allow_aisle_staging=allow_aisle_staging,
            include_rack_aisle=include_rack_aisle,
        )
        if plan is None:
            continue
        pickable += lp_pickable_layers_on_pallet(backup)
        total += backup.layers_remaining

    if include_rack_aisle:
        from slotting.resupply_rack import item_rack_ready

        if item_rack_ready(state, item_id, allow_storage_fallback=False):
            rack_ids = {s.slot_id for s in state.slots if s.slot_type == SlotType.RESUPPLY_RACK}
            for pallet in state.pallets:
                if pallet.item_id != item_id or pallet.current_slot not in rack_ids:
                    continue
                if pallet.layers_remaining <= 0:
                    continue
                pickable += lp_pickable_layers_on_pallet(pallet)
                total += pallet.layers_remaining

    return pickable, total


def _classify_demand_layers_by_pickable_supply(
    layers: int,
    *,
    pickable: int,
    total_supply: int,
) -> tuple[int, int, int]:
    """Split demand into (lp_ready, height_blocked, shortage) given pickable supply."""
    lp_part = min(layers, pickable)
    rem = layers - lp_part
    height_blocked = 0
    if rem > 0:
        height_blocked = min(rem, max(0, total_supply - pickable))
        rem -= height_blocked
    shortage = rem
    return lp_part, height_blocked, shortage


def item_has_pickable_face_supply(state: WorldState, item_id: str) -> bool:
    """True when a source face has LP-pickable layers for *item_id*."""
    return face_lp_pickable_layers_for_item(state, item_id) > 0


def item_has_immediate_lp_supply(
    state: WorldState,
    item_id: str,
    *,
    allow_swap: bool = True,
) -> bool:
    """True when LP demand could be served from pickable face or swap-ready backup/rack."""
    if item_has_pickable_face_supply(state, item_id):
        return True
    if not allow_swap:
        return False
    pickable, _ = _swap_ready_pickable_layers_for_item(
        state,
        item_id,
        allow_aisle_staging=True,
        include_rack_aisle=bool(state.resupply_rack_slots),
    )
    return pickable > 0


def _classify_demand(
    state: WorldState,
    demand: dict[str, int],
    *,
    require_build_face: bool = False,
) -> ThroughputMetrics:
    held = _held_pallet_ids(state)
    excluded = state.excluded_item_ids

    layer_pick_ready = 0
    hand_unstaged = 0
    hand_shortage = 0
    hand_excluded = 0
    low_layer_slotted = 0

    # Hoisted item sets — the per-item helpers each rescanned every pallet,
    # making this loop O(items x pallets) on every hourly sim step.
    source_items = {
        p.item_id for p in state.pallets_in_source_slots() if p.layers_remaining > 0
    }
    dest_items: set[str] = set()
    if require_build_face:
        for pallet in state.pallets_in_destination_slots():
            dest_items.update(pallet.destination_skus())

    for item_id, layers in demand.items():
        if item_id in excluded:
            hand_excluded += layers
            continue
        on_face = item_id in source_items
        via_swap = not on_face and _eligible_via_resupply_swap(
            state,
            item_id,
            allow_aisle_staging=True,
            include_rack_aisle=bool(state.resupply_rack_slots),
        )
        if on_face or via_swap:
            if require_build_face and item_id not in dest_items:
                hand_unstaged += layers
            elif on_face:
                from slotting.inventory import face_layers_for_item

                pickable = face_lp_pickable_layers_for_item(state, item_id)
                total_face = face_layers_for_item(state, item_id)
                lp_part, height_blocked, rem = _classify_demand_layers_by_pickable_supply(
                    layers, pickable=pickable, total_supply=total_face
                )
                layer_pick_ready += lp_part
                low_layer_slotted += height_blocked
                hand_shortage += rem
            else:
                swap_pickable, swap_total = _swap_ready_pickable_layers_for_item(
                    state,
                    item_id,
                    allow_aisle_staging=True,
                    include_rack_aisle=bool(state.resupply_rack_slots),
                )
                lp_part, height_blocked, rem = _classify_demand_layers_by_pickable_supply(
                    layers, pickable=swap_pickable, total_supply=swap_total
                )
                layer_pick_ready += lp_part
                low_layer_slotted += height_blocked
                hand_shortage += rem
        elif _eligible_unslotted_pallets(state, item_id, held):
            hand_unstaged += layers
        else:
            hand_shortage += layers

    return ThroughputMetrics(
        as_of=state.as_of,
        total_layer_demand=sum(demand.values()),
        layer_pick_ready_layers=layer_pick_ready,
        hand_pick_unstaged_layers=hand_unstaged,
        hand_pick_shortage_layers=hand_shortage,
        hand_pick_excluded_layers=hand_excluded,
        low_layer_slotted_layers=low_layer_slotted,
    )


def _eligible_in_source_slot(state: WorldState, item_id: str) -> bool:
    """Any non-empty source-slot pallet for *item_id* can serve layer pick (clear to empty)."""
    for pallet in state.pallets_in_source_slots():
        if pallet.item_id != item_id:
            continue
        if pallet.layers_remaining <= 0:
            continue
        if item_id in state.excluded_item_ids:
            continue
        return True
    return False


def _eligible_via_resupply_swap(state: WorldState, item_id: str, *, allow_aisle_staging: bool = False, include_rack_aisle: bool = False) -> bool:
    """Eligible on backup row or resupply rack with in-aisle swap path (cross-SKU OK)."""
    from slotting.backup_row import item_swap_ready

    return item_swap_ready(
        state,
        item_id,
        allow_storage_fallback=False,
        allow_aisle_staging=allow_aisle_staging,
        include_rack_aisle=include_rack_aisle,
    )


def _only_low_layer_in_source(state: WorldState, item_id: str) -> bool:
    """Legacy bucket — unused when LOW_LAYERS rule is disabled."""
    _ = (state, item_id)
    return False


def compute_throughput(state: WorldState) -> ThroughputMetrics:
    """
    Classify 24h layer demand by fulfillment path.

    - layer_pick_ready: eligible pallet already in a source slot
    - hand_pick_unstaged: eligible pallet exists but not slotted (ADD fixes)
    - hand_pick_shortage: no eligible pallet (WAIT / hand pick elsewhere)
    - hand_pick_excluded: item on exclusion list
    - low_layer_slotted: face stock above clamp pick reach (48in)
    """
    return _classify_demand(state, _demand_by_item(state))


def compute_wave_throughput(
    state: WorldState,
    *,
    horizon: timedelta = WAVED_HORIZON,
    require_build_face: bool = False,
    ship_date: date | None = None,
) -> ThroughputMetrics:
    """
    Classify waved layer demand (default 4h window) by fulfillment path.

    Uses the same buckets as :func:`compute_throughput`, but only counts
    ``OrderStatus.WAVED`` orders due within *horizon* of *state.as_of*.

    When *ship_date* is set, only orders on that ship day are counted (carry-day
    LP without next-day wave bleed in the 4h window).

    When *require_build_face* is True, demand counts as layer-pick-ready only
    if eligible supply is in a source slot **and** the item has a build face
    on a destination slot.
    """
    demand = (
        _waved_demand_by_item_for_ship_date(state, ship_date, horizon)
        if ship_date is not None
        else _waved_demand_by_item(state, horizon)
    )
    return _classify_demand(
        state,
        demand,
        require_build_face=require_build_face,
    )


def compute_wave_throughput_bleed(
    state: WorldState,
    ship_date: date,
    *,
    horizon: timedelta = WAVED_HORIZON,
) -> ThroughputMetrics:
    """Waved demand from orders not on *ship_date* (next-day bleed in the 4h window)."""
    all_demand = _waved_demand_by_item(state, horizon)
    ship_demand = _waved_demand_by_item_for_ship_date(state, ship_date, horizon)
    bleed_demand = {
        item_id: layers - ship_demand.get(item_id, 0)
        for item_id, layers in all_demand.items()
        if layers > ship_demand.get(item_id, 0)
    }
    return _classify_demand(state, bleed_demand)
