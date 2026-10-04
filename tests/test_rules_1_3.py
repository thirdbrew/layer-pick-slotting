"""Tests for rules 1–3: low layers, zero demand, excluded items."""

from __future__ import annotations

from datetime import timedelta

from slotting.models import Exclusion, OrderArea, OrderLine, OrderStatus, RecommendationAction, RuleId
from slotting.rules import DEFAULT_RULE_CONFIG, rule_excluded_item, rule_low_layers, rule_zero_demand

from conftest import AS_OF, minimal_state, slotted_pallet


def test_low_layers_exactly_two_triggers_remove() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-A", layers=2)
    state = minimal_state(pallets=[pallet], slots=[slot])
    recs = rule_low_layers(state)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_REMOVE
    assert recs[0].rule == RuleId.LOW_LAYERS
    assert recs[0].pallet_id == "P1"


def test_low_layers_exactly_three_does_not_trigger() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-A", layers=3)
    state = minimal_state(pallets=[pallet], slots=[slot])
    assert rule_low_layers(state) == []


def test_zero_demand_triggers_remove() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-DEAD", layers=5)
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[])
    cfg = DEFAULT_RULE_CONFIG.with_policy_flags(proactive_zero_demand_remove=True)
    recs = rule_zero_demand(state, cfg)
    assert len(recs) == 1
    assert recs[0].rule == RuleId.ZERO_DEMAND


def test_zero_demand_with_visible_order_does_not_trigger() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-LIVE", layers=5)
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-LIVE",
        qty_layers=1,
        due_at=AS_OF + timedelta(hours=12),
        status=OrderStatus.VISIBLE,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[order])
    cfg = DEFAULT_RULE_CONFIG.with_policy_flags(
        proactive_zero_demand_remove=True,
        zero_demand_reclaim_horizon_hours=24,
    )
    assert rule_zero_demand(state, cfg) == []


def test_excluded_item_triggers_remove() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-BAD", layers=6)
    state = minimal_state(
        pallets=[pallet],
        slots=[slot],
        exclusions=[Exclusion(item_id="ITEM-BAD", reason="Test exclusion")],
    )
    recs = rule_excluded_item(state)
    assert len(recs) == 1
    assert recs[0].rule == RuleId.EXCLUDED_ITEM
    assert "Test exclusion" in recs[0].reason
