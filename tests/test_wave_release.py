"""Tests for hourly wave release modeling."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from slotting.demand_throughput import load_orders
from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.wave_release import (
    compute_hourly_wave_demand,
    order_ship_date,
    peak_spread_wave_window,
    spread_orders_to_hourly_waves,
)
from conftest import SYNTHETIC_ORDERS


@pytest.fixture
def synthetic_orders_path() -> Path:
    if not SYNTHETIC_ORDERS.is_file():
        pytest.skip(f"synthetic orders missing: {SYNTHETIC_ORDERS}")
    return SYNTHETIC_ORDERS


def test_spread_reduces_single_wave_skus_vs_clumped_due_at() -> None:
    """Synthetic spread still lowers 4h concurrent SKUs vs all lines due at once."""
    day = date(2026, 3, 3)
    clumped = [
        OrderLine(
            order_id=f"o-{i}-20260303",
            item_id=f"SKU{i}",
            qty_layers=10 + i,
            due_at=datetime.combine(day, time(6, 0)),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        )
        for i in range(30)
    ]
    raw_skus = len({o.item_id for o in clumped})

    spread = spread_orders_to_hourly_waves(
        clumped,
        waves_per_day=12,
        shift_start_hour=6,
        due_offset_hours=4,
    )
    peak_layers, peak_skus, _ = peak_spread_wave_window(spread, wave_horizon=timedelta(hours=4))

    assert raw_skus == 30
    assert peak_skus < raw_skus
    assert peak_layers < sum(o.qty_layers for o in clumped)


def test_hourly_demand_send_plus_4h_due(synthetic_orders_path):
    orders = load_orders(synthetic_orders_path)
    this_line = [o for o in orders if o.area == OrderArea.THIS_LINE]
    peak_day = max(
        {order_ship_date(o) for o in this_line},
        key=lambda d: sum(o.qty_layers for o in this_line if order_ship_date(o) == d),
    )
    demand = compute_hourly_wave_demand(
        this_line,
        waves_per_day=12,
        shift_start_hour=6,
        due_offset_hours=4,
        layers_per_hour_picker=160.0,
    )

    assert demand.peak_ship_date == peak_day
    assert demand.max_skus_per_wave >= 1
    assert demand.sla_layers_clearable == 640


def test_due_at_is_release_plus_offset(synthetic_orders_path):
    orders = load_orders(synthetic_orders_path)
    spread = spread_orders_to_hourly_waves(
        orders,
        waves_per_day=12,
        shift_start_hour=6,
        due_offset_hours=4,
    )
    assert spread[0].due_at.hour == 10
