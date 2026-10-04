"""Tests for donor sweep at fixed source count."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from slotting.optimize import LayoutConstraints, run_donor_sweep, write_donor_sweep_csv


def _write_orders(path: Path, rows: str) -> None:
    path.write_text(
        "order_id,item_id,qty_layers,due_at,status,area\n" + rows,
        encoding="utf-8",
    )


def test_donor_sweep_build_face_lp_rises_with_more_donors(tmp_path: Path) -> None:
    rows = []
    for index in range(1, 6):
        rows.append(
            f"O{index},SKU-{index:02d},5,2026-06-12T08:00:00,visible,this_line\n"
        )
    orders = tmp_path / "orders.csv"
    _write_orders(orders, "".join(rows))
    constraints = LayoutConstraints(
        area_sqft=None,
        source_slot_sqft=60,
        donor_slot_sqft=48,
        source_range=(5, 5),
        donor_range=(1, 5),
    )
    result = run_donor_sweep(
        orders_path=orders,
        fixed_source_slots=5,
        constraints=constraints,
        target_build_face_wave_lp_pct=100.0,
        wave_horizon=timedelta(hours=4),
    )
    by_donor = {c.donor_slots: c for c in result.candidates}
    assert by_donor[1].projected_build_face_wave_lp_pct < by_donor[5].projected_build_face_wave_lp_pct
    assert by_donor[5].meets_build_face_target is True
    assert result.winner is not None
    assert result.winner.donor_slots == 5


def test_write_donor_sweep_csv(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    _write_orders(
        orders,
        "O1,SKU-A,5,2026-06-12T08:00:00,visible,this_line\n"
        "O2,SKU-B,5,2026-06-12T09:00:00,visible,this_line\n",
    )
    constraints = LayoutConstraints(
        area_sqft=None,
        source_slot_sqft=60,
        donor_slot_sqft=48,
        source_range=(2, 2),
        donor_range=(1, 2),
    )
    result = run_donor_sweep(
        orders_path=orders,
        fixed_source_slots=2,
        constraints=constraints,
    )
    out = tmp_path / "donor_sweep_results.csv"
    write_donor_sweep_csv(result, out)
    text = out.read_text(encoding="utf-8")
    assert "projected_build_face_wave_lp_pct" in text
    assert "dest_blocked_sku_count" in text
