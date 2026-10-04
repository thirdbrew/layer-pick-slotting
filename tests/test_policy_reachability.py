"""Guards for the two ways the promote half of the engine went missing.

Both failures were silent and both produced plausible-looking output, which is
the only reason they survived: the floor still reported a large dead-slot
improvement while the layer-pick rate — the metric the operation is paid on —
did not move at all.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from slotting.data_loader import CsvDataSource
from slotting.models import RuleId
from slotting.rules import (
    DEFAULT_RULE_CONFIG,
    LIBRARY_POLICIES,
    RESUPPLY_RACK_RULE_CONFIG,
    parse_disabled_rules,
    run_rules,
    with_rules_disabled,
)
from slotting.simulation import simulate_once

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
SITE_AS_OF = datetime(2026, 3, 2, 6, 0, 0)


def _synthetic_state():
    return CsvDataSource(SYNTHETIC_DIR).load_world_state(as_of=SITE_AS_OF)


# --------------------------------------------------------------------------
# 1. --without-rules is a subtraction, never an addition.
# --------------------------------------------------------------------------
# DEFAULT_RULE_CONFIG disables three rules. A parse that builds a fresh config
# from the named rules alone silently ENABLES those three, so the attribution
# delta the README sells measures four changes while claiming to measure one.


def test_without_rules_keeps_the_base_policy_disabled_set() -> None:
    config = parse_disabled_rules(["ZERO_DEMAND"])
    assert RuleId.ZERO_DEMAND in config.disabled
    for still_off in DEFAULT_RULE_CONFIG.disabled:
        assert still_off in config.disabled, f"{still_off.value} was silently enabled"


def test_without_rules_never_enables_a_rule_on_a_named_policy() -> None:
    base = RESUPPLY_RACK_RULE_CONFIG
    narrowed = with_rules_disabled(base, ["CROSS_AREA_HOLD"])
    assert base.disabled <= narrowed.disabled
    assert RuleId.CROSS_AREA_HOLD in narrowed.disabled


def test_without_rules_is_a_no_op_when_nothing_is_named() -> None:
    assert with_rules_disabled(RESUPPLY_RACK_RULE_CONFIG, None) is RESUPPLY_RACK_RULE_CONFIG


# --------------------------------------------------------------------------
# 2. The promote path is reachable from a shipped policy.
# --------------------------------------------------------------------------
# RACK_PROMOTE is gated twice: by the disabled set AND by include_rack_aisle.
# DEFAULT_RULE_CONFIG closes both, and only a full replay profile opened them,
# so before LIBRARY_POLICIES existed the library engine could evict and never promote.


def test_default_policy_cannot_promote() -> None:
    assert RuleId.RACK_PROMOTE in DEFAULT_RULE_CONFIG.disabled
    assert DEFAULT_RULE_CONFIG.include_rack_aisle is False


def test_a_shipped_policy_opens_both_promote_gates() -> None:
    openers = [
        name
        for name, cfg in LIBRARY_POLICIES.items()
        if RuleId.RACK_PROMOTE not in cfg.disabled and cfg.include_rack_aisle
    ]
    assert openers, "no named policy can reach RACK_PROMOTE: the promote path is dead code"


def test_resupply_rack_policy_actually_fires_rack_promote() -> None:
    state = _synthetic_state()
    fired = {r.rule for r in run_rules(state, LIBRARY_POLICIES["resupply-rack"])}
    assert RuleId.RACK_PROMOTE in fired


def test_promote_policy_moves_layer_pick_not_just_dead_slots() -> None:
    """The floor can look better while picking exactly as slowly as before.

    Dead-slot rate falls on eviction alone, so it cannot tell you whether the
    freed faces went to anything worth picking. Assert the throughput metric
    moves too, or the engine is decorating the floor.
    """
    state = _synthetic_state()
    promote = simulate_once(state, rule_config=LIBRARY_POLICIES["resupply-rack"])
    assert promote.wave_layer_pick_improvement_pp > 0.0
    assert promote.wave_layers_shifted_to_layer_pick > 0


# --------------------------------------------------------------------------
# 3. The generator supplies the promote path with an input.
# --------------------------------------------------------------------------
# Drawing rack backing stock from the same popularity curve that filled the
# faces backs SKUs that already hold a face, so "waved SKU, no face, stock
# upstream" is never generated and RACK_PROMOTE has nothing to fire on.


def test_rack_backs_demanded_skus_that_hold_no_face() -> None:
    slots = {
        row["slot_id"]: row["slot_type"]
        for row in csv.DictReader(open(SYNTHETIC_DIR / "slots.csv", encoding="utf-8"))
    }
    pallets = list(csv.DictReader(open(SYNTHETIC_DIR / "pallets.csv", encoding="utf-8")))
    orders = list(csv.DictReader(open(SYNTHETIC_DIR / "orders.csv", encoding="utf-8")))

    faced = {p["item_id"] for p in pallets if slots[p["current_slot"]] == "source"}
    racked = {p["item_id"] for p in pallets if slots[p["current_slot"]] == "resupply_rack"}
    demanded = {o["item_id"] for o in orders}

    promote_candidates = racked & demanded - faced
    # A smoke guard against the zero-candidate state, not a tuned threshold — the
    # binding check is the behavioural one above. Sizing stock to demand once cut this
    # from 21 to 5 by facing nearly every demanded SKU at t=0, and only a count caught it.
    assert len(promote_candidates) >= 5, (
        f"only {len(promote_candidates)} rack SKUs lack a face; the promote rules "
        "have nothing to act on and the demo can only evict"
    )


def test_floor_stock_is_sized_against_demand() -> None:
    """Stock drawn from the popularity curve is not stock sized to demand.

    Drawing it gave the head SKU one 1-to-8 layer pallet whether it wanted 1 layer in
    the horizon or 151, so a third of demand was unservable for a reason no slotting
    decision could fix, and the engine looked weak because the site was mis-stocked.
    Coverage is bounded on both sides: a floor that can serve everything never
    exercises WAIT_SUPPLY_SHORTAGE.
    """
    from datetime import timedelta

    pallets = list(csv.DictReader(open(SYNTHETIC_DIR / "pallets.csv", encoding="utf-8")))
    orders = list(csv.DictReader(open(SYNTHETIC_DIR / "orders.csv", encoding="utf-8")))
    excluded = {
        r["item_id"]
        for r in csv.DictReader(open(SYNTHETIC_DIR / "exclusions.csv", encoding="utf-8"))
    }

    stock: dict[str, int] = {}
    for p in pallets:
        stock[p["item_id"]] = stock.get(p["item_id"], 0) + int(p["layers_remaining"])

    horizon_end = SITE_AS_OF + timedelta(hours=24)
    demand: dict[str, int] = {}
    for o in orders:
        if o["area"] != "this_line" or o["item_id"] in excluded:
            continue
        if SITE_AS_OF <= datetime.fromisoformat(o["due_at"]) <= horizon_end:
            demand[o["item_id"]] = demand.get(o["item_id"], 0) + int(o["qty_layers"])

    total = sum(demand.values())
    covered = sum(min(want, stock.get(item, 0)) for item, want in demand.items())
    pct = 100 * covered / total
    assert 65.0 <= pct <= 92.0, f"horizon demand coverage is {pct:.1f}%"
