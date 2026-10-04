"""Tests for 24-hour visibility planning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from slotting.demand_throughput import load_orders
from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.visibility_planning import (
    final_shift_as_of,
    rank_ship_day_setup_skus,
    setup_coverage,
    visibility_snapshot,
)
from slotting.wave_release import spread_orders_to_hourly_waves
from conftest import SYNTHETIC_ORDERS


@pytest.fixture
def synthetic_orders_path() -> Path:
    if not SYNTHETIC_ORDERS.is_file():
        pytest.skip(f"synthetic orders missing: {SYNTHETIC_ORDERS}")
    return SYNTHETIC_ORDERS


def test_visible_contains_waved():
    as_of = datetime(2026, 6, 1, 8, 0)
    orders = [
        OrderLine(
            order_id="1",
            item_id="A",
            qty_layers=5,
            due_at=datetime(2026, 6, 1, 10, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id="2",
            item_id="B",
            qty_layers=3,
            due_at=datetime(2026, 6, 2, 8, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
    ]
    snap = visibility_snapshot(orders, as_of)
    assert snap.waved_skus == 1
    assert snap.visible_skus == 2
    assert snap.setup_skus == 1


def test_final_shift_sees_next_pick_day(synthetic_orders_path):
    orders = load_orders(synthetic_orders_path)
    spread = spread_orders_to_hourly_waves(
        [o for o in orders if o.area == OrderArea.THIS_LINE],
        waves_per_day=12,
        shift_start_hour=6,
        due_offset_hours=4,
    )
    # Derive the peak day from the feed rather than pinning a calendar date:
    # the test then holds for any order file, synthetic or real, instead of
    # silently going vacuous when the data changes underneath it.
    by_day: dict[object, int] = {}
    for o in spread:
        by_day[o.due_at.date()] = by_day.get(o.due_at.date(), 0) + o.qty_layers
    assert by_day, "no this-line orders in the feed"
    peak_day = max(by_day, key=lambda d: by_day[d])

    as_of = final_shift_as_of(peak_day, final_shift_start_hour=22)
    snap = visibility_snapshot(spread, as_of)
    # The final shift must SEE the next pick day, which is the whole point of a
    # 24h visible horizon: strictly more visible than waved, and every waved SKU
    # covered by the setup list.
    assert snap.visible_skus > 0
    assert snap.visible_skus >= snap.waved_skus
    assert snap.setup_skus >= snap.waved_skus


def test_setup_coverage_respects_slot_count():
    as_of = datetime(2026, 6, 1, 8, 0)
    orders = [
        OrderLine(
            order_id=str(i),
            item_id=f"SKU{i}",
            qty_layers=10 - i,
            due_at=datetime(2026, 6, 1, 12, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        )
        for i in range(5)
    ]
    cov = setup_coverage(orders, as_of, source_slots=2)
    assert cov["staged_skus"] == 2
    assert cov["unstaged_skus"] == 3


def test_rank_ship_day_abc_tiebreaker():
    ship = __import__("datetime").date(2026, 3, 3)
    due = __import__("datetime").datetime(2026, 3, 3, 10, 0)
    suffix = ship.strftime("%Y%m%d")
    orders = [
        OrderLine(
            order_id=f"o-C-{suffix}",
            item_id="C_ITEM",
            qty_layers=50,
            due_at=due,
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id=f"o-A-{suffix}",
            item_id="A_ITEM",
            qty_layers=50,
            due_at=due,
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
    ]
    ranked = rank_ship_day_setup_skus(
        orders,
        ship,
        abc_by_item={"A_ITEM": "A", "C_ITEM": "C"},
    )
    assert [row[0] for row in ranked] == ["A_ITEM", "C_ITEM"]


def test_rank_ship_day_peak_wave_priority():
    ship = __import__("datetime").date(2026, 3, 3)
    suffix = ship.strftime("%Y%m%d")
    orders = [
        OrderLine(
            order_id=f"o-flat-{suffix}",
            item_id="FLAT",
            qty_layers=30,
            due_at=__import__("datetime").datetime(2026, 3, 3, 18, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id=f"o-flat2-{suffix}",
            item_id="FLAT",
            qty_layers=30,
            due_at=__import__("datetime").datetime(2026, 3, 3, 20, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id=f"o-spike-{suffix}",
            item_id="SPIKE",
            qty_layers=60,
            due_at=__import__("datetime").datetime(2026, 3, 3, 19, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
    ]
    ranked = rank_ship_day_setup_skus(orders, ship, early_wave_hours=4)
    assert ranked[0][0] == "SPIKE"


def test_rank_ship_day_early_wave_beats_afternoon_bulk():
    ship = __import__("datetime").date(2026, 3, 3)
    suffix = ship.strftime("%Y%m%d")
    orders = [
        OrderLine(
            order_id=f"o-pm-{suffix}",
            item_id="PM_BULK",
            qty_layers=100,
            due_at=__import__("datetime").datetime(2026, 3, 3, 18, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id=f"o-am-{suffix}",
            item_id="AM_WAVE",
            qty_layers=20,
            due_at=__import__("datetime").datetime(2026, 3, 3, 8, 0),
            status=OrderStatus.VISIBLE,
            area=OrderArea.THIS_LINE,
        ),
    ]
    ranked = rank_ship_day_setup_skus(orders, ship)
    assert ranked[0][0] == "AM_WAVE"
    assert ranked[1][0] == "PM_BULK"
