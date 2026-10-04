"""Sub-hour simulation tick helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

from slotting.schedule_ticks import (
    hour_in_schedule,
    is_hour_boundary,
    layers_budget_for_step,
    sim_step_delta,
)


def test_sim_step_delta_validates_divisor() -> None:
    assert sim_step_delta(minutes=10) == timedelta(minutes=10)


def test_is_hour_boundary_sub_hour() -> None:
    step = timedelta(minutes=10)
    assert is_hour_boundary(datetime(2026, 3, 3, 8, 0), step)
    assert not is_hour_boundary(datetime(2026, 3, 3, 8, 10), step)


def test_layers_budget_scales_with_step() -> None:
    assert layers_budget_for_step(60.0, timedelta(minutes=10)) == 10.0


def test_consume_whole_layers_budget_accumulates_fraction() -> None:
    from slotting.schedule_ticks import consume_whole_layers_budget

    # 200 lyr/h × 10 min — six ticks must total exactly 200 layers
    step = layers_budget_for_step(200.0, timedelta(minutes=10))
    rem = 0.0
    wholes: list[int] = []
    for _ in range(6):
        whole, rem = consume_whole_layers_budget(step, rem)
        wholes.append(whole)
    assert sum(wholes) == 200
    assert abs(rem) < 1e-6


def test_hour_in_schedule_respects_boundary() -> None:
    step = timedelta(minutes=10)
    assert hour_in_schedule(datetime(2026, 3, 3, 11, 0), (11,), step)
    assert not hour_in_schedule(datetime(2026, 3, 3, 11, 10), (11,), step)
