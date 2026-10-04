"""Tests for layout sweep optimizer."""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta

from slotting.optimize import (
    LayoutConstraints,
    OptimizeMode,
    enumerate_layout_pairs,
    run_sweep,
)


def _write_orders(path: Path, rows: str) -> None:
    path.write_text(
        "order_id,item_id,qty_layers,due_at,status,area\n" + rows,
        encoding="utf-8",
    )


def test_enumerate_layout_pairs_sorted_by_footprint() -> None:
    constraints = LayoutConstraints(
        area_sqft=1000,
        source_slot_sqft=60,
        donor_slot_sqft=48,
        source_range=(2, 3),
        donor_range=(1, 2),
    )
    pairs = enumerate_layout_pairs(constraints)
    footprints = [row[2] for row in pairs]
    assert footprints == sorted(footprints)
    assert (2, 1, 168.0, True) in pairs


def test_combined_footprint_decides_area_fit() -> None:
    """A pair within each per-dimension cap but over the usable area is infeasible."""
    constraints = LayoutConstraints(
        area_sqft=1000,  # usable = 850 after 15% aisle reserve
        source_slot_sqft=60,
        donor_slot_sqft=48,
        source_range=(10, 14),  # 14 x 60 = 840 <= 850 alone
        donor_range=(5, 17),  # 17 x 48 = 816 <= 850 alone
    )
    pairs = enumerate_layout_pairs(constraints)
    by_pair = {(s, d): fits for s, d, _, fits in pairs}
    assert by_pair[(10, 5)] is True  # 600 + 240 = 840 <= 850
    assert by_pair[(14, 17)] is False  # 840 + 816 = 1656 > 850
    assert by_pair[(10, 6)] is False  # 600 + 288 = 888 > 850


def test_mode_b_picks_highest_wave_throughput(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    _write_orders(
        orders,
        "O1,SKU-A,5,2026-06-12T08:00:00,visible,this_line\n"
        "O2,SKU-B,5,2026-06-12T09:00:00,visible,this_line\n"
        "O3,SKU-C,5,2026-06-12T10:00:00,visible,this_line\n",
    )
    constraints = LayoutConstraints(
        area_sqft=2000,
        source_slot_sqft=60,
        donor_slot_sqft=48,
        source_range=(1, 3),
        donor_range=(1, 1),
    )
    result = run_sweep(
        orders_path=orders,
        constraints=constraints,
        mode=OptimizeMode.MAX_THROUGHPUT,
    )
    assert result.winner is not None
    assert result.winner.source_slots == 3
    assert result.winner.projected_wave_layer_pick_pct == 100.0


def test_mode_a_picks_smallest_footprint_meeting_target(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    _write_orders(
        orders,
        "O1,SKU-A,5,2026-06-12T08:00:00,visible,this_line\n"
        "O2,SKU-B,5,2026-06-12T09:00:00,visible,this_line\n",
    )
    constraints = LayoutConstraints(
        area_sqft=2000,
        source_slot_sqft=60,
        donor_slot_sqft=48,
        source_range=(1, 3),
        donor_range=(1, 2),
    )
    result = run_sweep(
        orders_path=orders,
        constraints=constraints,
        mode=OptimizeMode.MIN_FOOTPRINT,
        target_wave_lp_pct=100.0,
        max_shortage_count=0,
        source_buffer_slots=0,
    )
    assert result.winner is not None
    assert result.winner.source_slots == 2
    assert result.winner.meets_target is True

def test_mode_a_unlimited_area_finds_min_source(tmp_path: Path) -> None:
    rows = []
    for index in range(1, 11):
        rows.append(
            f"O{index},SKU-{index:02d},5,2026-06-12T08:00:00,visible,this_line\n"
        )
    orders = tmp_path / "orders.csv"
    _write_orders(orders, "".join(rows))
    constraints = LayoutConstraints(
        area_sqft=None,
        source_slot_sqft=60,
        donor_slot_sqft=48,
        source_range=(1, 20),
        donor_range=(2, 2),
    )
    result = run_sweep(
        orders_path=orders,
        constraints=constraints,
        mode=OptimizeMode.MIN_FOOTPRINT,
        target_wave_lp_pct=100.0,
        max_shortage_count=0,
        source_buffer_slots=0,
        wave_horizon=timedelta(hours=4),
    )
    assert result.winner is not None
    assert result.winner.source_slots == 10
    assert result.unlimited_area is True


def test_wave_hours_changes_peak_profile(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    _write_orders(
        orders,
        "O1,SKU-A,5,2026-06-12T08:00:00,visible,this_line\n"
        "O2,SKU-B,5,2026-06-12T11:00:00,visible,this_line\n",
    )
    from slotting.demand_throughput import compute_required_throughput

    wide = compute_required_throughput(orders, wave_horizon=timedelta(hours=4))
    narrow = compute_required_throughput(orders, wave_horizon=timedelta(hours=2))
    assert wide.peak_wave_layer_demand >= narrow.peak_wave_layer_demand
