"""Dynamic source:destination active ratio per wave."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from slotting.throughput import item_has_pickable_face_supply
from slotting.models import (
    OrderArea,
    OrderStatus,
    Recommendation,
    RecommendationAction,
    RuleId,
    WorldState,
)
from slotting.rules import WAVED_HORIZON


@dataclass(frozen=True)
class WaveRatioAdvice:
    """
    Recommended active slot counts for the current wave window.

    Physical layout may be larger than the active set during lulls; during
    heavy waves the active set approaches (or hits) the built layout.
    """

    as_of: datetime
    physical_source_slots: int
    physical_destination_slots: int
    physical_ratio: str
    active_source_slots: int
    active_destination_slots: int
    active_ratio: str
    waved_sku_count: int
    waved_layer_demand: int
    supply_ready_sku_count: int
    build_faces_in_use: int

    @property
    def source_headroom(self) -> int:
        return self.physical_source_slots - self.active_source_slots

    @property
    def destination_headroom(self) -> int:
        return self.physical_destination_slots - self.active_destination_slots


def _waved_item_ids(
    state: WorldState,
    *,
    horizon: timedelta,
) -> set[str]:
    cutoff = state.as_of + horizon
    return {
        o.item_id
        for o in state.orders
        if o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
        and o.item_id not in state.excluded_item_ids
    }


def _waved_layer_demand(
    state: WorldState,
    waved_items: set[str],
    *,
    horizon: timedelta,
) -> int:
    cutoff = state.as_of + horizon
    return sum(
        o.qty_layers
        for o in state.orders
        if o.item_id in waved_items
        and o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    )


def _ratio_string(source: int, destination: int) -> str:
    return f"{source}:{destination}"


def compute_wave_ratio(
    state: WorldState,
    *,
    horizon: timedelta = WAVED_HORIZON,
    source_buffer_slots: int = 2,
) -> WaveRatioAdvice:
    """
    Derive active source:destination counts from waved demand.

    Uses the same donor hint as ``demand_throughput`` — ``(skus + 2) // 3`` —
    raised to cover supply-ready SKUs and build faces already in use.
    """
    physical_source = len(state.source_slots)
    physical_dest = len(state.destination_slots)
    waved_items = _waved_item_ids(state, horizon=horizon)
    waved_sku_count = len(waved_items)
    waved_layers = _waved_layer_demand(state, waved_items, horizon=horizon)
    supply_ready = {
        item for item in waved_items if item_has_pickable_face_supply(state, item)
    }
    supply_ready_count = len(supply_ready)
    build_faces_in_use = len(state.pallets_in_destination_slots())

    if waved_sku_count == 0:
        active_source = min(physical_source, max(1, build_faces_in_use))
        active_dest = min(physical_dest, build_faces_in_use)
    else:
        active_source = min(physical_source, max(1, waved_sku_count + source_buffer_slots))
        donor_hint = max(1, (waved_sku_count + 2) // 3)
        faces_needed = max(supply_ready_count, build_faces_in_use, donor_hint)
        active_dest = min(physical_dest, max(1, faces_needed))

    return WaveRatioAdvice(
        as_of=state.as_of,
        physical_source_slots=physical_source,
        physical_destination_slots=physical_dest,
        physical_ratio=_ratio_string(physical_source, physical_dest),
        active_source_slots=active_source,
        active_destination_slots=active_dest,
        active_ratio=_ratio_string(active_source, active_dest),
        waved_sku_count=waved_sku_count,
        waved_layer_demand=waved_layers,
        supply_ready_sku_count=supply_ready_count,
        build_faces_in_use=build_faces_in_use,
    )


def rule_wave_deactivate_dest(
    state: WorldState,
    advice: WaveRatioAdvice,
) -> list[Recommendation]:
    """Advise standing down destination slots beyond the active wave ratio."""
    if advice.active_destination_slots >= advice.physical_destination_slots:
        return []

    active_ids = {
        s.slot_id
        for s in sorted(state.destination_slots, key=lambda slot: slot.slot_id)[
            : advice.active_destination_slots
        ]
    }
    recs: list[Recommendation] = []
    for slot in sorted(state.destination_slots, key=lambda s: s.slot_id):
        if slot.slot_id in active_ids:
            continue
        pallet = state.pallet_by_id(slot.occupied_by_pallet_id) if slot.occupied_by_pallet_id else None
        detail = (
            f"Stand down {slot.slot_id} for this wave "
            f"(active ratio {advice.active_ratio} vs built {advice.physical_ratio})."
        )
        if pallet is not None:
            detail = (
                f"Close {pallet.pallet_id} on {slot.slot_id} and stand down slot: "
                f"active destination count is {advice.active_destination_slots}."
            )
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_DEACTIVATE_DEST,
                pallet_id=pallet.pallet_id if pallet else None,
                item_id=pallet.item_id if pallet else None,
                slot_id=slot.slot_id,
                reason=detail,
                rule=RuleId.WAVE_DEACTIVATE_DEST,
            )
        )
    return recs


def rule_wave_activate_dest(
    state: WorldState,
    advice: WaveRatioAdvice,
) -> list[Recommendation]:
    """
    Advise bringing empty destination slots online within the active set.

    Complements ``DEST_ASSIGN_WAVED`` — flags capacity before a build face opens.
    """
    if advice.waved_sku_count == 0:
        return []

    dest_slots = sorted(state.destination_slots, key=lambda s: s.slot_id)
    active_slots = dest_slots[: advice.active_destination_slots]
    waiting_items = [
        item
        for item in sorted(_waved_item_ids(state, horizon=WAVED_HORIZON))
        if item_has_pickable_face_supply(state, item) and not state.item_on_destination_slot(item)
    ]
    if not waiting_items:
        return []

    recs: list[Recommendation] = []
    wait_iter = iter(waiting_items)
    for slot in active_slots:
        if slot.occupied_by_pallet_id:
            continue
        try:
            item_id = next(wait_iter)
        except StopIteration:
            break
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_ACTIVATE_DEST,
                pallet_id=None,
                item_id=item_id,
                slot_id=slot.slot_id,
                reason=(
                    f"Activate {slot.slot_id} for waved {item_id} "
                    f"(active ratio {advice.active_ratio})."
                ),
                rule=RuleId.WAVE_ACTIVATE_DEST,
            )
        )
    return recs


def run_wave_ratio_rules(state: WorldState, config) -> list[Recommendation]:
    """Evaluate wave-ratio rules and return advisory recommendations."""
    advice = compute_wave_ratio(state)
    recs: list[Recommendation] = []
    if config.is_enabled(RuleId.WAVE_DEACTIVATE_DEST):
        recs.extend(rule_wave_deactivate_dest(state, advice))
    if config.is_enabled(RuleId.WAVE_ACTIVATE_DEST):
        recs.extend(rule_wave_activate_dest(state, advice))
    return recs
