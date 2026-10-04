"""Tests for wave-scoped throughput (4h) and required throughput profile."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from slotting.demand_throughput import compute_required_throughput
from slotting.simulation import simulate_once
from slotting.throughput import compute_wave_throughput

from conftest import AS_OF, load_sample_state, minimal_state, slotted_pallet
from slotting.models import OrderArea, OrderLine, OrderStatus


def test_sample_wave_throughput_baseline() -> None:
    state = load_sample_state()
    wave = compute_wave_throughput(state)
    assert wave.total_layer_demand == 4
    assert wave.layer_pick_ready_layers == 0
    assert wave.hand_pick_unstaged_layers == 2
    assert wave.hand_pick_excluded_layers == 1
    assert wave.hand_pick_shortage_layers == 1
    assert abs(wave.layer_pick_pct - 0.0) < 0.001


def test_simulation_improves_wave_layer_pick_on_sample() -> None:
    result = simulate_once(load_sample_state())
    assert result.baseline_wave_throughput.total_layer_demand == 4
    assert result.projected_wave_throughput.layer_pick_ready_layers == 2
    assert result.wave_layer_pick_pct == 0.5
    assert result.wave_layer_pick_improvement_pp == 50.0
    assert result.wave_layers_shifted_to_layer_pick == 2
    assert result.baseline_wave_shortage_count == 1
    assert result.wave_shortage_count == 1


def test_wave_throughput_ignores_visible_only_orders_in_window() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-VISIBLE", layers=6)
    waved = OrderLine(
        order_id="O-W",
        item_id="ITEM-WAVED",
        qty_layers=2,
        due_at=AS_OF + timedelta(hours=2),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    visible = OrderLine(
        order_id="O-V",
        item_id="ITEM-VISIBLE",
        qty_layers=5,
        due_at=AS_OF + timedelta(hours=3),
        status=OrderStatus.VISIBLE,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[waved, visible])
    wave = compute_wave_throughput(state)
    assert wave.total_layer_demand == 2
    assert wave.layer_pick_ready_layers == 0


def test_wave_throughput_counts_slotted_waved_demand() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-WAVED", layers=6)
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-WAVED",
        qty_layers=3,
        due_at=AS_OF + timedelta(hours=1),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[order])
    wave = compute_wave_throughput(state)
    assert wave.total_layer_demand == 3
    assert wave.layer_pick_ready_layers == 3


def test_wave_throughput_requires_build_face_when_configured() -> None:
    pallet, source = slotted_pallet("P1", "ITEM-WAVED", layers=6)
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-WAVED",
        qty_layers=3,
        due_at=AS_OF + timedelta(hours=1),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[pallet], slots=[source], orders=[order])
    source_only = compute_wave_throughput(state)
    build_face = compute_wave_throughput(state, require_build_face=True)
    assert source_only.layer_pick_ready_layers == 3
    assert build_face.layer_pick_ready_layers == 0
    assert build_face.hand_pick_unstaged_layers == 3


def test_wave_throughput_height_blocked_layers_on_tall_face() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-WAVED", layers=8)
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-WAVED",
        qty_layers=8,
        due_at=AS_OF + timedelta(hours=1),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[order])
    wave = compute_wave_throughput(state)
    assert wave.total_layer_demand == 8
    assert wave.layer_pick_ready_layers == 5
    assert wave.low_layer_slotted_layers == 3


def test_required_throughput_peak_wave_window(tmp_path: Path) -> None:
    orders_csv = tmp_path / "orders.csv"
    orders_csv.write_text(
        "order_id,item_id,qty_layers,due_at,status,area\n"
        "O1,SKU-A,10,2026-06-12T08:00:00,visible,this_line\n"
        "O2,SKU-B,5,2026-06-12T09:00:00,visible,this_line\n"
        "O3,SKU-C,8,2026-06-12T10:00:00,visible,this_line\n"
        "O4,SKU-D,20,2026-06-13T08:00:00,visible,this_line\n",
        encoding="utf-8",
    )
    profile = compute_required_throughput(orders_csv)
    assert profile.peak_wave_layer_demand == 23
    assert profile.peak_concurrent_skus == 3
    assert profile.min_source_slots_hint == 5
    assert profile.min_donor_slots_hint == 1
    assert profile.order_count == 4
