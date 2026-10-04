"""Tests for disabling rules and LOW_LAYERS comparison."""

from __future__ import annotations

from slotting.models import RuleId
from slotting.rules import DEFAULT_RULE_CONFIG, WITH_LOW_LAYERS, parse_disabled_rules, run_rules
from slotting.simulation import simulate_once

from conftest import load_sample_state


def test_default_skips_low_layers_remove() -> None:
    state = load_sample_state()
    default = run_rules(state, DEFAULT_RULE_CONFIG)
    with_low = run_rules(state, WITH_LOW_LAYERS)
    assert [r for r in default if r.rule == RuleId.LOW_LAYERS] == []
    assert len([r for r in with_low if r.rule == RuleId.LOW_LAYERS]) >= 1


def test_simulate_compare_low_layers_fewer_removes_by_default() -> None:
    state = load_sample_state()
    default = simulate_once(state, rule_config=DEFAULT_RULE_CONFIG)
    with_low = simulate_once(state, rule_config=WITH_LOW_LAYERS)
    assert default.low_layer_removes_applied == 0
    assert with_low.low_layer_removes_applied >= 1
    assert with_low.removes_applied > default.removes_applied


def test_parse_disabled_rules() -> None:
    config = parse_disabled_rules(["LOW_LAYERS", "zero_demand"])
    assert RuleId.LOW_LAYERS in config.disabled
    assert RuleId.ZERO_DEMAND in config.disabled
