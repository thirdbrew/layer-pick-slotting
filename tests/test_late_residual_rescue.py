"""Late-shift residual rescue — the late-shift residual starve case.

A waved SKU with a small (<cap-min) residual after late_shift_hour keeps the
day cap (2); if both pulls were consumed earlier, its last wave starves with
zero pickable supply. The rescue grants ONE extra pull in exactly that state.
Off by default; evaluate it as an arm against the chain benchmark.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from slotting.backtest import ShiftPullConfig
from slotting.models import OrderArea, OrderLine, OrderStatus

SHIP = date(2026, 2, 17)


def _order(item: str, layers: int, hour: int) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item}-{hour}",
        item_id=item,
        qty_layers=layers,
        due_at=datetime.combine(SHIP, datetime.min.time()) + timedelta(hours=hour),
        status=OrderStatus.VISIBLE,
        area=OrderArea.THIS_LINE,
    )


def _cfg(orders, **kw) -> ShiftPullConfig:
    return ShiftPullConfig(ship_date=SHIP, orders=orders, **kw)


def test_small_residual_keeps_day_cap_without_rescue() -> None:
    orders = [_order("SKU-A", 4, 19)]  # 4-layer residual due 19:00 — below cap-min 6
    cfg = _cfg(orders)
    assert cfg.max_per_sku_for("SKU-A") == cfg.max_per_sku
    assert cfg.max_per_sku_for("SKU-A", hour=19) == cfg.max_per_sku


def test_rescue_grants_one_extra_pull_late_shift_only() -> None:
    orders = [_order("SKU-A", 4, 19)]
    cfg = _cfg(orders, late_residual_rescue=True)
    # before late shift: unchanged
    assert cfg.max_per_sku_for("SKU-A", hour=12) == cfg.max_per_sku
    # after late shift with small nonzero residual: one rescue pull
    assert cfg.max_per_sku_for("SKU-A", hour=19) == cfg.max_per_sku + 1


def test_rescue_does_not_apply_to_large_residuals_or_zero() -> None:
    big = [_order("SKU-B", 10, 19)]  # >= cap-min: normal late cap already applies
    cfg_big = _cfg(big, late_residual_rescue=True)
    assert cfg_big.max_per_sku_for("SKU-B", hour=19) == cfg_big.max_per_sku_late_shift

    none = [_order("SKU-C", 5, 10)]  # demand due before late shift: residual 0
    cfg_none = _cfg(none, late_residual_rescue=True)
    assert cfg_none.max_per_sku_for("SKU-C", hour=19) == cfg_none.max_per_sku


def test_rescue_hour_unknown_is_conservative() -> None:
    orders = [_order("SKU-A", 4, 19)]
    cfg = _cfg(orders, late_residual_rescue=True)
    assert cfg.max_per_sku_for("SKU-A") == cfg.max_per_sku
