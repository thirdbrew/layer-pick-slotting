"""Integration tests: sample data exercises all six rules."""

from __future__ import annotations

from slotting.models import RecommendationAction, RuleId
from slotting.rules import run_rules

from conftest import load_sample_state


def test_sample_data_fires_core_rules() -> None:
    state = load_sample_state()
    recs = run_rules(state)
    rules_fired = {r.rule for r in recs}
    assert RuleId.ZERO_DEMAND in rules_fired
    assert RuleId.EXCLUDED_ITEM in rules_fired
    assert RuleId.CROSS_AREA_HOLD in rules_fired
    assert RuleId.ADD_WAVED_DEMAND in rules_fired
    assert RuleId.WAIT_SUPPLY_SHORTAGE in rules_fired
    assert RuleId.LOW_LAYERS not in rules_fired


def test_sample_hold_suppresses_p004_removal() -> None:
    state = load_sample_state()
    recs = run_rules(state)
    p004_removals = [
        r
        for r in recs
        if r.pallet_id == "P004" and r.action == RecommendationAction.FLAG_REMOVE
    ]
    assert p004_removals == []
    holds = [r for r in recs if r.pallet_id == "P004" and r.action == RecommendationAction.DO_NOT_TOUCH]
    assert holds


def test_sample_boundary_pallet_not_removed_for_layers() -> None:
    """P011 has exactly 3 layers — above the 2-layer cutoff."""
    state = load_sample_state()
    recs = run_rules(state)
    p011_low = [r for r in recs if r.pallet_id == "P011" and r.rule == RuleId.LOW_LAYERS]
    assert p011_low == []
