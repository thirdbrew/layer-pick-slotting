"""Tests for dynamic wave source:destination ratio."""

from __future__ import annotations

from datetime import timedelta

from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    RecommendationAction,
    RuleId,
    Slot,
    SlotType,
)
from slotting.rules import RuleConfig, run_rules
from slotting.simulation import simulate_once
from slotting.wave_ratio import compute_wave_ratio, rule_wave_activate_dest, rule_wave_deactivate_dest

from conftest import AS_OF, minimal_state, slotted_pallet


def _dest_slot(slot_id: str = "D01", *, occupied: str | None = None) -> Slot:
    return Slot(slot_id=slot_id, slot_type=SlotType.DESTINATION, occupied_by_pallet_id=occupied)


def _source_slots(count: int) -> list[Slot]:
    return [
        Slot(slot_id=f"S{i:02d}", slot_type=SlotType.SOURCE, occupied_by_pallet_id=None)
        for i in range(1, count + 1)
    ]


def _dest_slots(count: int) -> list[Slot]:
    return [_dest_slot(f"D{i:02d}") for i in range(1, count + 1)]


def _waved_order(item_id: str, *, layers: int = 2, hours: float = 2) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=AS_OF + timedelta(hours=hours),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def test_compute_wave_ratio_heavy_wave_caps_at_physical_layout() -> None:
    orders = [_waved_order(f"SKU-{index:02d}") for index in range(1, 9)]
    state = minimal_state(
        slots=_source_slots(10) + _dest_slots(4),
        orders=orders,
    )
    advice = compute_wave_ratio(state, source_buffer_slots=2)
    assert advice.physical_ratio == "10:4"
    assert advice.active_ratio == "10:3"
    assert advice.waved_sku_count == 8
    assert advice.waved_layer_demand == 16
    assert advice.destination_headroom == 1


def test_compute_wave_ratio_light_wave_reduces_active_dest() -> None:
    pallet, source = slotted_pallet("P1", "SKU-A", layers=6, slot_id="S01")
    orders = [_waved_order("SKU-A"), _waved_order("SKU-B"), _waved_order("SKU-C")]
    state = minimal_state(
        pallets=[pallet],
        slots=[source] + _dest_slots(4),
        orders=orders,
    )
    advice = compute_wave_ratio(state, source_buffer_slots=0)
    assert advice.physical_ratio == "1:4"
    assert advice.active_source_slots == 1
    assert advice.active_destination_slots == 1
    assert advice.active_ratio == "1:1"
    assert advice.supply_ready_sku_count == 1


def test_compute_wave_ratio_lull_with_no_waved_demand() -> None:
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="SKU-IDLE",
        layers_remaining=1,
        current_slot="D01",
        arrived_at=AS_OF,
    )
    state = minimal_state(
        pallets=[build],
        slots=_source_slots(6) + [_dest_slot("D01", occupied="BUILD-D01"), _dest_slot("D02")],
        orders=[],
    )
    advice = compute_wave_ratio(state)
    assert advice.waved_sku_count == 0
    assert advice.active_destination_slots == 1
    assert advice.build_faces_in_use == 1


def test_wave_deactivate_dest_on_excess_slots() -> None:
    orders = [_waved_order("SKU-A"), _waved_order("SKU-B"), _waved_order("SKU-C")]
    state = minimal_state(slots=_source_slots(4) + _dest_slots(4), orders=orders)
    advice = compute_wave_ratio(state, source_buffer_slots=0)
    recs = rule_wave_deactivate_dest(state, advice)
    assert len(recs) == 3
    assert all(r.action == RecommendationAction.FLAG_DEACTIVATE_DEST for r in recs)
    assert recs[0].slot_id == "D02"
    assert recs[-1].slot_id == "D04"


def test_wave_activate_dest_for_supply_ready_items() -> None:
    pallet, source = slotted_pallet("P1", "SKU-A", layers=6, slot_id="S01")
    orders = [_waved_order("SKU-A"), _waved_order("SKU-B")]
    state = minimal_state(
        pallets=[pallet],
        slots=[source] + _dest_slots(3),
        orders=orders,
    )
    advice = compute_wave_ratio(state, source_buffer_slots=0)
    recs = rule_wave_activate_dest(state, advice)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_ACTIVATE_DEST
    assert recs[0].slot_id == "D01"
    assert recs[0].item_id == "SKU-A"
    assert recs[0].rule == RuleId.WAVE_ACTIVATE_DEST


def test_wave_ratio_rules_wired_in_run_rules() -> None:
    pallet, source = slotted_pallet("P1", "SKU-A", layers=6, slot_id="S01")
    orders = [_waved_order("SKU-A"), _waved_order("SKU-B"), _waved_order("SKU-C")]
    state = minimal_state(
        pallets=[pallet],
        slots=[source] + _dest_slots(4),
        orders=orders,
    )
    recs = run_rules(state)
    rules_fired = {r.rule for r in recs}
    assert RuleId.WAVE_ACTIVATE_DEST in rules_fired
    assert RuleId.WAVE_DEACTIVATE_DEST in rules_fired


def test_wave_ratio_rules_respect_disabled_config() -> None:
    pallet, source = slotted_pallet("P1", "SKU-A", layers=6, slot_id="S01")
    orders = [_waved_order("SKU-A"), _waved_order("SKU-B")]
    state = minimal_state(
        pallets=[pallet],
        slots=[source] + _dest_slots(3),
        orders=orders,
    )
    config = RuleConfig(
        disabled=frozenset({RuleId.WAVE_ACTIVATE_DEST, RuleId.WAVE_DEACTIVATE_DEST})
    )
    recs = run_rules(state, config)
    wave_rules = {r.rule for r in recs if r.rule.value.startswith("WAVE_")}
    assert wave_rules == set()


def test_simulation_includes_wave_ratio_advice() -> None:
    pallet, source = slotted_pallet("P1", "SKU-A", layers=6, slot_id="S01")
    orders = [_waved_order("SKU-A")]
    state = minimal_state(
        pallets=[pallet],
        slots=[source, _dest_slot("D01")],
        orders=orders,
    )
    result = simulate_once(state)
    assert result.wave_ratio is not None
    assert result.wave_ratio.active_ratio == "1:1"
    assert result.wave_ratio.waved_sku_count == 1
