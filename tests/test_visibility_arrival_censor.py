"""Tests for order arrived_at foresight censor."""

from __future__ import annotations

from datetime import datetime

from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.visibility_planning import (
    affinity_orders_at_setup,
    orders_arrived_by,
    ship_day_orders_arrived_by_setup,
)


def _line(order_id: str, due: datetime, *, arrived: datetime | None = None) -> OrderLine:
    return OrderLine(
        order_id=order_id,
        item_id="SKU-A",
        qty_layers=1,
        due_at=due,
        status=OrderStatus.VISIBLE,
        area=OrderArea.THIS_LINE,
        arrived_at=arrived,
    )


def test_orders_arrived_by_treats_missing_as_known() -> None:
    setup = datetime(2026, 3, 2, 22, 0)
    orders = [
        _line("a", datetime(2026, 3, 3, 6, 0)),
        _line("b", datetime(2026, 3, 3, 8, 0), arrived=datetime(2026, 3, 3, 1, 0)),
    ]
    assert len(orders_arrived_by(orders, setup)) == 1


def test_ship_day_arrival_censor_at_setup() -> None:
    setup = datetime(2026, 3, 2, 22, 0)
    orders = [
        _line("early", datetime(2026, 3, 3, 6, 0), arrived=datetime(2026, 3, 2, 20, 0)),
        _line("late", datetime(2026, 3, 3, 12, 0), arrived=datetime(2026, 3, 3, 2, 0)),
        _line("other", datetime(2026, 3, 4, 6, 0), arrived=datetime(2026, 3, 2, 21, 0)),
    ]
    censored = ship_day_orders_arrived_by_setup(
        orders,
        datetime(2026, 3, 3).date(),
        setup_as_of=setup,
    )
    assert [o.order_id for o in censored] == ["early"]
    aff = affinity_orders_at_setup(
        orders,
        datetime(2026, 3, 3).date(),
        setup_as_of=setup,
        setup_arrival_censor=True,
    )
    assert [o.order_id for o in aff] == ["early"]
