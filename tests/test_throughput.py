"""Tests for throughput / labor mix metrics."""

from __future__ import annotations

from datetime import timedelta

from slotting.simulation import simulate_once
from slotting.throughput import compute_throughput

from conftest import AS_OF, load_sample_state, minimal_state, slotted_pallet
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet


def test_sample_throughput_baseline_hand_vs_layer() -> None:
    state = load_sample_state()
    tp = compute_throughput(state)
    assert tp.total_layer_demand == 13
    assert tp.layer_pick_ready_layers == 8
    assert tp.hand_pick_unstaged_layers == 2
    assert tp.hand_pick_excluded_layers == 1
    assert tp.low_layer_slotted_layers == 0
    assert tp.hand_pick_shortage_layers == 2
    assert abs(tp.layer_pick_pct - 8 / 13) < 0.001
    assert abs(tp.hand_pick_pct - 5 / 13) < 0.001


def test_simulation_improves_layer_pick_throughput_on_sample() -> None:
    result = simulate_once(load_sample_state())
    assert result.projected_throughput.layer_pick_ready_layers > result.baseline_throughput.layer_pick_ready_layers
    assert result.layer_pick_improvement_pp > 0
    assert result.layers_shifted_to_layer_pick == 2
    assert result.projected_throughput.hand_pick_unstaged_layers == 0


def test_add_moves_unstaged_demand_to_layer_pick() -> None:
    fresh = Pallet(
        pallet_id="P-NEW",
        item_id="ITEM-NEED",
        layers_remaining=6,
        current_slot=None,
        arrived_at=AS_OF,
    )
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-NEED",
        qty_layers=4,
        due_at=AS_OF + timedelta(hours=2),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[fresh], orders=[order])
    baseline = compute_throughput(state)
    assert baseline.hand_pick_unstaged_layers == 4
    assert baseline.layer_pick_ready_layers == 0

    result = simulate_once(state)
    assert result.adds_applied == 1
    assert result.projected_throughput.layer_pick_ready_layers == 4
    assert result.projected_throughput.hand_pick_unstaged_layers == 0


def test_low_layer_on_source_counts_pickable_face_layers_only() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-A", layers=2)
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-A",
        qty_layers=3,
        due_at=AS_OF + timedelta(hours=6),
        status=OrderStatus.VISIBLE,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[order])
    tp = compute_throughput(state)
    assert tp.low_layer_slotted_layers == 0
    assert tp.layer_pick_ready_layers == 2
    assert tp.hand_pick_shortage_layers == 1
