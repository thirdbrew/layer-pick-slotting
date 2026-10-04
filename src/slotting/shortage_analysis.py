"""Per-SKU waved-demand vs supply classification for floor validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from slotting.inventory import face_layers_for_item, ship_day_layers_for_item
from slotting.inventory_policy import inventory_layers_by_item, late_shift_layers_by_item
from slotting.floor_pick_model import PickPolicy, decompose_lp_pick_fulfillment
from slotting.models import OrderArea, OrderStatus, SlotType, WorldState
from slotting.pallet_geometry import face_lp_pickable_layers_for_item
from slotting.resupply_rack import item_rack_ready
from slotting.rules import WAVED_HORIZON, _eligible_unslotted_pallets, _held_pallet_ids
from slotting.throughput import _waved_demand_by_item


class ShortageKind(str, Enum):
    LP_READY = "lp_ready"
    LP_PARTIAL_MISS = "lp_partial_miss"
    HEIGHT_BLOCKED = "height_blocked"  # supply on face/swap but above clamp pick reach
    TOP_OFF_HAND_PICK = "top_off_hand_pick"
    UNSTAGED = "unstaged"  # inventory exists; not on pick-ready face
    RACK_ONLY = "rack_only"  # rack stock; face needs promote
    TRUE_SHORTAGE = "true_shortage"  # no eligible inventory left
    EXCLUDED = "excluded"


FLOOR_PICK_MISS_KINDS = frozenset(
    {
        ShortageKind.LP_PARTIAL_MISS,
        ShortageKind.HEIGHT_BLOCKED,
        ShortageKind.TOP_OFF_HAND_PICK,
    }
)


@dataclass(frozen=True)
class SkuShortageRow:
    item_id: str
    waved_layers: int
    face_layers: int
    rack_layers: int
    unslotted_layers: int
    total_inventory: int
    ship_day_layers: int
    late_shift_layers: int
    kind: ShortageKind
    shortage_layers: int
    due_window_layers: int = 0

    def as_dict(self) -> dict[str, str | int]:
        return {
            "item_id": self.item_id,
            "waved_layers": self.waved_layers,
            "face_layers": self.face_layers,
            "rack_layers": self.rack_layers,
            "unslotted_layers": self.unslotted_layers,
            "total_inventory": self.total_inventory,
            "ship_day_layers": self.ship_day_layers,
            "late_shift_layers": self.late_shift_layers,
            "due_window_layers": self.due_window_layers,
            "kind": self.kind.value,
            "shortage_layers": self.shortage_layers,
        }


def _rack_layers_for_item(state: WorldState, item_id: str) -> int:
    rack_ids = {s.slot_id for s in state.slots if s.slot_type == SlotType.RESUPPLY_RACK}
    return sum(
        p.layers_remaining
        for p in state.pallets
        if p.item_id == item_id and p.current_slot in rack_ids
    )


def _unslotted_layers_for_item(state: WorldState, item_id: str) -> int:
    return sum(
        p.layers_remaining
        for p in state.unslotted_pallets()
        if p.item_id == item_id
    )


def due_window_layers_by_item(
    state: WorldState,
    due_hours: tuple[int, ...],
) -> dict[str, int]:
    """Waved layer demand with due_at hour in *due_hours* and still in waved horizon."""
    cutoff = state.as_of + WAVED_HORIZON
    demand: dict[str, int] = {}
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.status != OrderStatus.WAVED:
            continue
        if order.due_at.hour not in due_hours:
            continue
        if not (state.as_of <= order.due_at <= cutoff):
            continue
        demand[order.item_id] = demand.get(order.item_id, 0) + order.qty_layers
    return demand


def _pick_supply_shortage_kind(
    waved_layers: int,
    pickable: int,
    total_supply: int,
) -> tuple[ShortageKind, int]:
    """
    Classify immediate LP gap for supply already on face or swap-ready.

    When *total_supply* exceeds *pickable*, the gap is height-blocked stack;
    otherwise inventory depth is the limit (partial miss).
    """
    if pickable >= waved_layers:
        return ShortageKind.LP_READY, 0
    shortage = waved_layers - pickable
    if total_supply > pickable:
        return ShortageKind.HEIGHT_BLOCKED, shortage
    return ShortageKind.LP_PARTIAL_MISS, shortage


def _classify_sku(
    state: WorldState,
    item_id: str,
    waved_layers: int,
    *,
    ship_date,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
    due_window_layers: int = 0,
    pick_demand_layers: int | None = None,
    pick_policy: PickPolicy | None = None,
) -> SkuShortageRow:
    if item_id in state.excluded_item_ids:
        return SkuShortageRow(
            item_id=item_id,
            waved_layers=waved_layers,
            face_layers=0,
            rack_layers=0,
            unslotted_layers=0,
            total_inventory=0,
            ship_day_layers=0,
            late_shift_layers=0,
            due_window_layers=0,
            kind=ShortageKind.EXCLUDED,
            shortage_layers=waved_layers,
        )

    face = face_layers_for_item(state, item_id)
    rack = _rack_layers_for_item(state, item_id)
    unslot = _unslotted_layers_for_item(state, item_id)
    total = inventory_layers_by_item(state).get(item_id, 0)
    held = _held_pallet_ids(state)
    ship_day = ship_day_layers_for_item(state.orders, item_id, ship_date) if ship_date else 0
    late = late_shift_layers_by_item(
        state, late_shift_hour=late_shift_hour, final_shift_hour=final_shift_hour
    ).get(item_id, 0)

    from slotting.throughput import (
        _eligible_in_source_slot,
        _eligible_via_resupply_swap,
        _swap_ready_pickable_layers_for_item,
    )

    policy = pick_policy or PickPolicy()
    pick_kw = (
        {"max_pick_height_in": policy.max_lp_pick_height_in}
        if policy.max_lp_pick_height_in is not None
        else {}
    )
    pickable_face = face_lp_pickable_layers_for_item(state, item_id, **pick_kw)
    if policy.partial_pick_model and pickable_face > 0:
        decompose_layers = (
            pick_demand_layers
            if pick_demand_layers is not None and pick_demand_layers > waved_layers
            else waved_layers
        )
        lp_ready, partial, top_off = decompose_lp_pick_fulfillment(
            face_layers=pickable_face,
            waved_layers=decompose_layers,
            as_of_hour=state.as_of.hour,
            policy=policy,
        )
        if top_off > 0:
            kind = ShortageKind.TOP_OFF_HAND_PICK
            shortage = top_off
        elif partial > 0:
            kind = (
                ShortageKind.HEIGHT_BLOCKED
                if face > pickable_face
                else ShortageKind.LP_PARTIAL_MISS
            )
            shortage = partial
        else:
            kind = ShortageKind.LP_READY
            shortage = 0
        return SkuShortageRow(
            item_id=item_id,
            waved_layers=waved_layers,
            face_layers=face,
            rack_layers=rack,
            unslotted_layers=unslot,
            total_inventory=total,
            ship_day_layers=ship_day,
            late_shift_layers=late,
            due_window_layers=due_window_layers,
            kind=kind,
            shortage_layers=shortage,
        )

    on_face = _eligible_in_source_slot(state, item_id) and pickable_face > 0
    via_swap = _eligible_via_resupply_swap(
        state, item_id, include_rack_aisle=bool(state.resupply_rack_slots)
    )
    if on_face:
        kind, shortage = _pick_supply_shortage_kind(waved_layers, pickable_face, face)
    elif via_swap:
        swap_pickable, swap_total = _swap_ready_pickable_layers_for_item(
            state,
            item_id,
            include_rack_aisle=bool(state.resupply_rack_slots),
        )
        kind, shortage = _pick_supply_shortage_kind(waved_layers, swap_pickable, swap_total)
    elif _eligible_unslotted_pallets(state, item_id, held):
        kind = ShortageKind.UNSTAGED
        shortage = waved_layers
    elif rack > 0 and item_rack_ready(state, item_id):
        kind = ShortageKind.RACK_ONLY
        shortage = waved_layers
    elif total <= 2:
        kind = ShortageKind.TRUE_SHORTAGE
        shortage = waved_layers
    else:
        kind = ShortageKind.TRUE_SHORTAGE
        shortage = waved_layers

    return SkuShortageRow(
        item_id=item_id,
        waved_layers=waved_layers,
        face_layers=face,
        rack_layers=rack,
        unslotted_layers=unslot,
        total_inventory=total,
        ship_day_layers=ship_day,
        late_shift_layers=late,
        due_window_layers=due_window_layers,
        kind=kind,
        shortage_layers=shortage,
    )


def analyze_waved_shortage(
    state: WorldState,
    *,
    ship_date,
    late_shift_hour: int = 18,
    final_shift_hour: int = 22,
    due_hours: tuple[int, ...] | None = None,
    pick_policy: PickPolicy | None = None,
) -> list[SkuShortageRow]:
    """Classify each waved SKU at *state.as_of*.

    When *due_hours* is set, only SKUs with waved demand due in those hours are
    included (aligns sim validation with floor miss logs for a pick hour).
    """
    waved = _waved_demand_by_item(state)
    due_window = due_window_layers_by_item(state, due_hours) if due_hours else {}
    item_ids = sorted(due_window.keys() if due_hours else waved.keys())
    rows: list[SkuShortageRow] = []
    use_bleed_pick_demand = bool(
        due_hours
        and pick_policy
        and pick_policy.partial_pick_model
    )
    for item_id in item_ids:
        layers = due_window.get(item_id, 0) if due_hours else waved.get(item_id, 0)
        if layers <= 0:
            continue
        total_waved = waved.get(item_id, layers)
        rows.append(
            _classify_sku(
                state,
                item_id,
                layers,
                ship_date=ship_date,
                late_shift_hour=late_shift_hour,
                final_shift_hour=final_shift_hour,
                due_window_layers=due_window.get(item_id, 0),
                pick_demand_layers=total_waved if use_bleed_pick_demand else None,
                pick_policy=pick_policy,
            )
        )
    rows.sort(key=lambda row: -row.waved_layers)
    return rows


def filter_shortage_rows(
    rows: list[SkuShortageRow],
    *,
    kinds: set[ShortageKind] | None = None,
    min_late_shift_layers: int = 0,
    require_due_window: bool = False,
) -> list[SkuShortageRow]:
    """Narrow shortage rows for floor-validation comparisons."""
    filtered: list[SkuShortageRow] = []
    for row in rows:
        if kinds is not None and row.kind not in kinds:
            continue
        if min_late_shift_layers > 0 and row.late_shift_layers < min_late_shift_layers:
            continue
        if require_due_window and row.due_window_layers <= 0:
            continue
        filtered.append(row)
    return filtered


def summarize_shortage(rows: list[SkuShortageRow]) -> dict[str, int | float]:
    total_waved = sum(r.waved_layers for r in rows)
    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[row.kind.value] = by_kind.get(row.kind.value, 0) + row.waved_layers
    return {
        "sku_count": len(rows),
        "total_waved_layers": total_waved,
        **{f"{k}_layers": v for k, v in by_kind.items()},
        **{f"{k}_pct": round(100 * v / total_waved, 1) if total_waved else 0 for k, v in by_kind.items()},
    }
