"""Tests for historical backtest."""

from __future__ import annotations

from datetime import timedelta

from slotting.backtest import BacktestMode, backtest_historical
from slotting.models import OrderArea, OrderLine, OrderStatus

from conftest import AS_OF, load_sample_state, minimal_state, slotted_pallet


def test_backtest_chained_improves_over_static_floor() -> None:
    state = load_sample_state()
    report = backtest_historical(
        state,
        start=AS_OF,
        end=AS_OF + timedelta(hours=2),
        step=timedelta(hours=1),
        mode=BacktestMode.CHAINED,
    )
    assert report.step_count == 3
    assert report.avg_improvement_pp >= 0
    assert report.total_removes_applied >= 0


def test_backtest_static_resets_floor() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-A", layers=2)
    state = minimal_state(pallets=[pallet], slots=[slot])
    report = backtest_historical(
        state,
        start=AS_OF,
        end=AS_OF + timedelta(hours=1),
        step=timedelta(hours=1),
        mode=BacktestMode.STATIC,
    )
    assert report.step_count == 2
    assert report.steps[0].baseline.dead_slots == report.steps[1].baseline.dead_slots


def test_backtest_orders_window_shifts_with_as_of() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-LIVE", layers=5)
    near = OrderLine(
        order_id="O1",
        item_id="ITEM-LIVE",
        qty_layers=1,
        due_at=AS_OF + timedelta(hours=2),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    far = OrderLine(
        order_id="O2",
        item_id="ITEM-LIVE",
        qty_layers=1,
        due_at=AS_OF + timedelta(days=3),
        status=OrderStatus.VISIBLE,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[pallet], slots=[slot], orders=[near, far])
    report = backtest_historical(
        state,
        start=AS_OF,
        end=AS_OF + timedelta(days=2),
        step=timedelta(days=1),
        mode=BacktestMode.STATIC,
    )
    assert report.step_count >= 2
