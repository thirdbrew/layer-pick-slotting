"""Tests for rules 4–6: cross-area hold, add waved demand, wait for supply."""

from __future__ import annotations

from datetime import timedelta

from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    RecommendationAction,
    RuleId,
)
from slotting.rules import (
    apply_suppression,
    rule_add_waved_demand,
    rule_cross_area_hold,
    rule_low_layers,
    rule_wait_supply_shortage,
    run_rules,
    _held_pallet_ids,
)

from conftest import AS_OF, minimal_state, slotted_pallet


def test_cross_area_hold_emits_do_not_touch() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-X", layers=5)
    orders = [
        OrderLine(
            order_id="O-OTHER",
            item_id="ITEM-X",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.OTHER_AREA,
        ),
        OrderLine(
            order_id="O-THIS",
            item_id="ITEM-X",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=6),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[pallet], slots=[slot], orders=orders)
    recs = rule_cross_area_hold(state)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.DO_NOT_TOUCH
    assert recs[0].rule == RuleId.CROSS_AREA_HOLD


def test_cross_area_hold_suppresses_removal() -> None:
    """Pallet with <=2 layers would be removed, but hold blocks it."""
    pallet, slot = slotted_pallet("P1", "ITEM-X", layers=2)
    orders = [
        OrderLine(
            order_id="O-OTHER",
            item_id="ITEM-X",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.OTHER_AREA,
        ),
        OrderLine(
            order_id="O-THIS",
            item_id="ITEM-X",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=6),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[pallet], slots=[slot], orders=orders)
    assert rule_low_layers(state)  # would remove
    final = run_rules(state)
    remove_recs = [r for r in final if r.action == RecommendationAction.FLAG_REMOVE]
    hold_recs = [r for r in final if r.action == RecommendationAction.DO_NOT_TOUCH]
    assert hold_recs
    assert not any(r.pallet_id == "P1" for r in remove_recs)


def test_add_waved_demand_when_unslotted_pallet_exists() -> None:
    unslotted = Pallet(
        pallet_id="P-NEW",
        item_id="ITEM-Y",
        layers_remaining=5,
        current_slot=None,
        arrived_at=AS_OF,
    )
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-Y",
            qty_layers=2,
            due_at=AS_OF + timedelta(hours=3),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[unslotted], orders=orders)
    held = _held_pallet_ids(state)
    recs = rule_add_waved_demand(state, held)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_ADD
    assert recs[0].pallet_id == "P-NEW"


def test_wait_when_waved_demand_but_no_eligible_pallet() -> None:
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-Z",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[], orders=orders)
    held = _held_pallet_ids(state)
    recs = rule_wait_supply_shortage(state, held)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_WAIT
    assert recs[0].rule == RuleId.WAIT_SUPPLY_SHORTAGE


def test_thin_unslotted_pallet_eligible_for_add() -> None:
    thin = Pallet(
        pallet_id="P-THIN",
        item_id="ITEM-Z",
        layers_remaining=2,
        current_slot=None,
        arrived_at=AS_OF,
    )
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-Z",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[thin], orders=orders)
    held = _held_pallet_ids(state)
    assert rule_add_waved_demand(state, held)
    assert rule_wait_supply_shortage(state, held) == []


def test_wait_when_only_depleted_unslotted_pallet() -> None:
    depleted = Pallet(
        pallet_id="P-EMPTY",
        item_id="ITEM-Z",
        layers_remaining=0,
        current_slot=None,
        arrived_at=AS_OF,
    )
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-Z",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[depleted], orders=orders)
    held = _held_pallet_ids(state)
    assert rule_add_waved_demand(state, held) == []
    recs = rule_wait_supply_shortage(state, held)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_WAIT


def test_apply_suppression_filters_held_pallet_add() -> None:
    from slotting.models import Recommendation

    recs = [
        Recommendation(
            action=RecommendationAction.DO_NOT_TOUCH,
            pallet_id="P1",
            item_id="ITEM-X",
            reason="hold",
            rule=RuleId.CROSS_AREA_HOLD,
        ),
        Recommendation(
            action=RecommendationAction.FLAG_REMOVE,
            pallet_id="P1",
            item_id="ITEM-X",
            reason="remove",
            rule=RuleId.LOW_LAYERS,
        ),
    ]
    filtered = apply_suppression(recs, {"P1"})
    assert len(filtered) == 1
    assert filtered[0].action == RecommendationAction.DO_NOT_TOUCH
