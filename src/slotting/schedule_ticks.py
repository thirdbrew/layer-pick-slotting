"""Simulation tick helpers — sub-hour steps with hourly hook boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta


def sim_step_delta(*, minutes: int) -> timedelta:
    if minutes <= 0 or 60 % minutes != 0:
        raise ValueError("minutes must divide 60 evenly")
    return timedelta(minutes=minutes)


def is_hour_boundary(as_of: datetime, step: timedelta) -> bool:
    """True when hourly-only hooks (realign, staging boost, shift pull) should run."""
    if step >= timedelta(hours=1):
        return True
    return as_of.minute == 0


def hour_in_schedule(as_of: datetime, hours: tuple[int, ...] | list[int], step: timedelta) -> bool:
    return is_hour_boundary(as_of, step) and as_of.hour in hours


def layers_budget_for_step(layers_per_hour: float, step: timedelta) -> float:
    """Scale pick budget for sub-hour simulation steps."""
    if layers_per_hour <= 0:
        return 0.0
    return layers_per_hour * (step.total_seconds() / 3600.0)


def consume_whole_layers_budget(raw_budget: float, remainder: float) -> tuple[int, float]:
    """Split a float pick budget into whole layers plus carried fractional remainder.

    Sub-hour ticks (e.g. 200 lyr/h × 10 min = 33.333) would otherwise lose ~0.33
    layers per tick when truncated with ``int()``. Accumulate the fraction so the
    long-run rate matches ``layers_per_hour``. A tiny epsilon absorbs binary-float
    dust so six 10-min ticks at 200 lyr/h sum to 200.
    """
    total = float(raw_budget) + float(remainder)
    if total <= 0:
        return 0, 0.0
    whole = int(total + 1e-9)
    return whole, total - whole
