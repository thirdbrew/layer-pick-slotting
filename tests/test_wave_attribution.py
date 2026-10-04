"""Wave send window attribution for layer picks."""

from __future__ import annotations

from datetime import date, datetime

from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.wave_release import (
    WaveBucket,
    active_wave_bucket,
    wave_pick_hour_range,
)


def _bucket(wave_index: int, send_hour: int, due_hour: int) -> WaveBucket:
    ship = date(2026, 3, 3)
    return WaveBucket(
        ship_date=ship,
        wave_index=wave_index,
        release_at=datetime(2026, 3, 3, send_hour),
        due_at=datetime(2026, 3, 3, due_hour),
        orders=(
            OrderLine(
                order_id=f"o-{wave_index}",
                item_id=f"SKU-{wave_index}",
                qty_layers=1,
                due_at=datetime(2026, 3, 3, due_hour),
                status=OrderStatus.VISIBLE,
                area=OrderArea.THIS_LINE,
            ),
        ),
    )


def test_active_wave_bucket_uses_latest_send_at_or_before_hour() -> None:
    buckets = [_bucket(0, 15, 19), _bucket(1, 17, 21)]
    assert active_wave_bucket(buckets, datetime(2026, 3, 3, 16)).wave_index == 0
    assert active_wave_bucket(buckets, datetime(2026, 3, 3, 17)).wave_index == 1


def test_wave_pick_hour_range_spans_until_next_send() -> None:
    buckets = [_bucket(0, 15, 19), _bucket(1, 17, 21)]
    hours = list(wave_pick_hour_range(buckets, buckets[0], final_shift_hour=22))
    assert hours == [15, 16]
