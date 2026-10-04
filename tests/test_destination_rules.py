"""Tests for destination (donor/build-face) slot rules."""

from __future__ import annotations

from datetime import timedelta

from slotting.destination_rules import (
    rule_dest_assign_waved,
    rule_dest_close_idle,
    rule_dest_wait_no_face,
    run_destination_rules,
)
from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    Recommendation,
    RecommendationAction,
    RuleId,
    Slot,
    SlotType,
)
from slotting.rules import DEFAULT_RULE_CONFIG, run_rules
from slotting.simulation import apply_recommendations

from conftest import AS_OF, minimal_state, slotted_pallet


def _dest_slot(slot_id: str = "D01", *, occupied: str | None = None) -> Slot:
    return Slot(slot_id=slot_id, slot_type=SlotType.DESTINATION, occupied_by_pallet_id=occupied)


def _waved_order(item_id: str, *, layers: int = 2, hours: float = 2) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=AS_OF + timedelta(hours=hours),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def test_dest_assign_waved_opens_empty_face() -> None:
    pallet, source = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    state = minimal_state(
        pallets=[pallet],
        slots=[source, _dest_slot("D01")],
        orders=[_waved_order("ITEM-A", layers=3)],
    )
    recs = rule_dest_assign_waved(state)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.action == RecommendationAction.FLAG_ADD
    assert rec.item_id == "ITEM-A"
    assert rec.slot_id == "D01"
    assert rec.rule == RuleId.DEST_ASSIGN_WAVED


def test_dest_assign_skips_without_source_supply() -> None:
    state = minimal_state(
        slots=[_dest_slot("D01")],
        orders=[_waved_order("ITEM-A")],
    )
    assert rule_dest_assign_waved(state) == []


def test_dest_assign_skips_excluded_item() -> None:
    from slotting.models import Exclusion

    pallet, source = slotted_pallet("P1", "ITEM-EXCL", layers=5)
    state = minimal_state(
        pallets=[pallet],
        slots=[source, _dest_slot("D01")],
        orders=[_waved_order("ITEM-EXCL")],
        exclusions=[Exclusion(item_id="ITEM-EXCL", reason="test")],
    )
    assert rule_dest_assign_waved(state) == []


def test_dest_assign_skips_item_already_on_destination() -> None:
    pallet, source = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="ITEM-A",
        layers_remaining=2,
        current_slot="D01",
        arrived_at=AS_OF,
    )
    state = minimal_state(
        pallets=[pallet, build],
        slots=[source, _dest_slot("D01", occupied="BUILD-D01")],
        orders=[_waved_order("ITEM-A")],
    )
    assert rule_dest_assign_waved(state) == []


def test_dest_close_idle_removes_stale_build() -> None:
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="ITEM-IDLE",
        layers_remaining=1,
        current_slot="D01",
        arrived_at=AS_OF,
    )
    state = minimal_state(
        pallets=[build],
        slots=[_dest_slot("D01", occupied="BUILD-D01")],
        orders=[_waved_order("ITEM-OTHER")],
    )
    recs = rule_dest_close_idle(state)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_REMOVE
    assert recs[0].pallet_id == "BUILD-D01"
    assert recs[0].slot_id == "D01"
    assert recs[0].rule == RuleId.DEST_CLOSE_IDLE


def test_dest_close_idle_keeps_waved_build() -> None:
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="ITEM-A",
        layers_remaining=2,
        current_slot="D01",
        arrived_at=AS_OF,
    )
    state = minimal_state(
        pallets=[build],
        slots=[_dest_slot("D01", occupied="BUILD-D01")],
        orders=[_waved_order("ITEM-A")],
    )
    assert rule_dest_close_idle(state) == []


def test_dest_wait_no_face_when_faces_full() -> None:
    pallet, source = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="ITEM-B",
        layers_remaining=1,
        current_slot="D01",
        arrived_at=AS_OF,
    )
    state = minimal_state(
        pallets=[pallet, build],
        slots=[source, _dest_slot("D01", occupied="BUILD-D01")],
        orders=[_waved_order("ITEM-A"), _waved_order("ITEM-B")],
    )
    recs = rule_dest_wait_no_face(state)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_WAIT
    assert recs[0].item_id == "ITEM-A"
    assert recs[0].rule == RuleId.DEST_WAIT_NO_FACE


def test_dest_wait_counts_queue_overflow_past_free_faces() -> None:
    """Continuous queue signal: with 1 free face and 2 supply-ready
    waved items, the overflow item waits — the old any-free-slot gate reported
    zero and made day waits discontinuous in pad count (a knife-edge)."""
    pallet_a, source_a = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    pallet_b, source_b = slotted_pallet("P2", "ITEM-B", layers=5, slot_id="S02")
    state = minimal_state(
        pallets=[pallet_a, pallet_b],
        slots=[source_a, source_b, _dest_slot("D01")],
        orders=[_waved_order("ITEM-A"), _waved_order("ITEM-B")],
    )
    recs = rule_dest_wait_no_face(state)
    assert len(recs) == 1
    # sorted assignment order: ITEM-A takes the free face, ITEM-B queues
    assert recs[0].item_id == "ITEM-B"
    assert recs[0].rule == RuleId.DEST_WAIT_NO_FACE


def test_dest_wait_silent_when_capacity_covers_queue() -> None:
    pallet, source = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    state = minimal_state(
        pallets=[pallet],
        slots=[source, _dest_slot("D01"), _dest_slot("D02")],
        orders=[_waved_order("ITEM-A")],
    )
    assert rule_dest_wait_no_face(state) == []


def test_run_destination_rules_wired_in_run_rules() -> None:
    pallet, source = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    state = minimal_state(
        pallets=[pallet],
        slots=[source, _dest_slot("D01")],
        orders=[_waved_order("ITEM-A")],
    )
    recs = run_rules(state)
    dest_rules = {r.rule for r in recs if r.rule.value.startswith("DEST_")}
    assert RuleId.DEST_ASSIGN_WAVED in dest_rules


def test_apply_opens_build_pallet_on_destination() -> None:
    pallet, source = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    dest = _dest_slot("D01")
    state = minimal_state(
        pallets=[pallet],
        slots=[source, dest],
        orders=[_waved_order("ITEM-A", layers=4)],
    )
    rec = Recommendation(
        action=RecommendationAction.FLAG_ADD,
        pallet_id=None,
        item_id="ITEM-A",
        slot_id="D01",
        reason="test",
        rule=RuleId.DEST_ASSIGN_WAVED,
    )
    projected, applied = apply_recommendations(state, [rec])
    dest_proj = next(s for s in projected.slots if s.slot_id == "D01")
    assert dest_proj.occupied_by_pallet_id == "BUILD-D01"
    build = projected.pallet_by_id("BUILD-D01")
    assert build is not None
    assert build.item_id == "ITEM-A"
    assert build.layers_remaining == 4
    assert build.current_slot == "D01"
    assert len(applied) == 1
    assert applied[0].pallet_id == "BUILD-D01"


def test_apply_closes_idle_build_pallet() -> None:
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="ITEM-IDLE",
        layers_remaining=1,
        current_slot="D01",
        arrived_at=AS_OF,
    )
    state = minimal_state(
        pallets=[build],
        slots=[_dest_slot("D01", occupied="BUILD-D01")],
        orders=[],
    )
    rec = Recommendation(
        action=RecommendationAction.FLAG_REMOVE,
        pallet_id="BUILD-D01",
        item_id="ITEM-IDLE",
        slot_id="D01",
        reason="test",
        rule=RuleId.DEST_CLOSE_IDLE,
    )
    projected, applied = apply_recommendations(state, [rec])
    dest_proj = next(s for s in projected.slots if s.slot_id == "D01")
    assert dest_proj.occupied_by_pallet_id is None
    build = projected.pallet_by_id("BUILD-D01")
    assert build is not None
    assert build.current_slot is None
    assert len(applied) == 1


def test_destination_rules_respect_disabled_config() -> None:
    from slotting.rules import RuleConfig

    pallet, source = slotted_pallet("P1", "ITEM-A", layers=5, slot_id="S01")
    state = minimal_state(
        pallets=[pallet],
        slots=[source, _dest_slot("D01")],
        orders=[_waved_order("ITEM-A")],
    )
    config = RuleConfig(
        disabled=frozenset(
            {
                RuleId.DEST_ASSIGN_WAVED,
                RuleId.DEST_CLOSE_IDLE,
                RuleId.DEST_WAIT_NO_FACE,
            }
        )
    )
    assert run_destination_rules(state, config) == []


def test_sample_integration_dest_rules_optional() -> None:
    """Sample data exercises source rules; dest rules may be absent at baseline."""
    from conftest import load_sample_state

    state = load_sample_state()
    recs = run_rules(state, DEFAULT_RULE_CONFIG)
    dest_recs = [r for r in recs if r.rule.value.startswith("DEST_")]
    # No waved+staged item at 08:00 on sample — dest assign/wait should not fire.
    assert dest_recs == []
