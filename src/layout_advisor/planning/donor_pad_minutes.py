"""Donor pads as pad-MINUTES (measurement half).

Reads plan ``wave_actions.csv`` rows, reconstructs per-pad occupancy
episodes, computes hourly concurrency vs dynamic donor capacity, and
estimates the resource-leveling headroom from shifting wave releases
earlier within the due SLA.

Occupancy model = the planner's own assumption (donor_build_queue.py
``pad_free_at``): a donor pad is busy from its scheduled open until the
serving wave's due. Plan artifacts record no pad-free event, so this is
the best model available from committed plans.

The leveling estimate is an UPPER BOUND on benefit: it assumes an episode's
duration is invariant under the shift and ignores inventory knock-on
effects. A real gain claim needs the shift implemented in the replay sim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

DONOR_OPEN_ACTIONS = {
    "PREOPEN_DONOR",
    "ROLLING_PREOPEN_DONOR",
    "LULL_AHEAD_PREP",
    "OPEN_DONOR_BUILD",
    "OPEN_DONOR_COLOCATED",
}

DONOR_BUILD_COMPLETE_ACTIONS = {
    "EXIT_DONOR_TO_WRAP",
    "STAGE_DONOR_COMPLETE",
    "CLOSE_IDLE_BUILD",
}


@dataclass(frozen=True)
class DonorEpisode:
    slot: str
    open_at: datetime
    close_at: datetime
    wave_index: int | None
    wave_due: datetime | None


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def donor_episodes(
    rows: list[dict],
    *,
    day_end: datetime,
    default_due: datetime | None = None,
) -> list[DonorEpisode]:
    """Occupancy episodes for donor open actions: open → serving wave's due.

    Setup pre-opens carry no wave_due in the CSV; those close at
    *default_due* (the first wave's due), falling back to *day_end*.
    """
    episodes: list[DonorEpisode] = []
    for row in rows:
        if row["action"] not in DONOR_OPEN_ACTIONS:
            continue
        open_at = _parse_dt(row["wave_send"])
        wave_due = _parse_dt(row["wave_due"]) if row.get("wave_due") else None
        close_at = wave_due or default_due or day_end
        wave_index = int(row["wave_index"]) if row.get("wave_index") else None
        episodes.append(
            DonorEpisode(
                slot=row["to_slot"],
                open_at=open_at,
                close_at=max(close_at, open_at),
                wave_index=wave_index,
                wave_due=wave_due,
            )
        )
    episodes.sort(key=lambda e: (e.open_at, e.slot))
    return episodes


def donor_episodes_build_complete(
    rows: list[dict],
    *,
    day_end: datetime,
) -> list[DonorEpisode]:
    """Donor pad busy from open until build-completion action frees the face.

    Pair each donor open on *to_slot* with the next wrap/complete on the same
    *from_slot* (build face). Unclosed opens run to *day_end*.
    """
    opens: dict[str, tuple[datetime, int | None, datetime | None]] = {}
    episodes: list[DonorEpisode] = []
    for row in sorted(rows, key=lambda r: r.get("wave_send", "")):
        action = row.get("action", "")
        slot = row.get("to_slot") or row.get("from_slot") or ""
        if not slot:
            continue
        at = _parse_dt(row["wave_send"])
        if action in DONOR_OPEN_ACTIONS and row.get("to_slot"):
            opens[row["to_slot"]] = (
                at,
                int(row["wave_index"]) if row.get("wave_index") else None,
                _parse_dt(row["wave_due"]) if row.get("wave_due") else None,
            )
        elif action in DONOR_BUILD_COMPLETE_ACTIONS and row.get("from_slot"):
            face = row["from_slot"]
            if face not in opens:
                continue
            open_at, wave_index, wave_due = opens.pop(face)
            episodes.append(
                DonorEpisode(
                    slot=face,
                    open_at=open_at,
                    close_at=max(at, open_at),
                    wave_index=wave_index,
                    wave_due=wave_due,
                )
            )
    for face, (open_at, wave_index, wave_due) in opens.items():
        episodes.append(
            DonorEpisode(
                slot=face,
                open_at=open_at,
                close_at=day_end,
                wave_index=wave_index,
                wave_due=wave_due,
            )
        )
    episodes.sort(key=lambda e: (e.open_at, e.slot))
    return episodes


def hourly_concurrency(episodes: list[DonorEpisode]) -> dict[datetime, int]:
    """Occupied donor pads per hour bucket; close hour is exclusive."""
    counts: dict[datetime, int] = {}
    for ep in episodes:
        hour = ep.open_at.replace(minute=0, second=0, microsecond=0)
        while hour < ep.close_at:
            counts[hour] = counts.get(hour, 0) + 1
            hour += timedelta(hours=1)
    return counts


def level_wave_shifts(
    episodes: list[DonorEpisode],
    *,
    max_shift_hours: int,
    earliest: datetime,
) -> tuple[list[DonorEpisode], dict[int | None, int]]:
    """Greedy resource leveling: shift whole waves earlier to cut peak concurrency.

    Decision variable: integer shift (hours) per wave_index, 0..max_shift_hours,
    earlier only (due SLA holds trivially). Objective: min peak concurrency.
    Greedy on the peak hour's biggest contributor; NP-hard exactly — this is a
    heuristic lower bound on achievable peak reduction, not the optimum.
    """
    shifts: dict[int | None, int] = {}
    current = list(episodes)

    def _apply(eps: list[DonorEpisode], wave: int | None, hours: int) -> list[DonorEpisode]:
        delta = timedelta(hours=hours)
        return [
            replace(e, open_at=e.open_at - delta, close_at=e.close_at - delta)
            if e.wave_index == wave
            else e
            for e in eps
        ]

    improved = True
    while improved:
        improved = False
        conc = hourly_concurrency(current)
        if not conc:
            break
        peak = max(conc.values())
        peak_hours = [h for h, c in conc.items() if c == peak]
        # Waves contributing to the peak hour, largest pad-hour footprint first.
        candidates: dict[int | None, int] = {}
        for ep in current:
            if ep.wave_index is None:
                continue
            if any(ep.open_at <= h < ep.close_at for h in peak_hours):
                candidates[ep.wave_index] = candidates.get(ep.wave_index, 0) + 1
        for wave, _ in sorted(candidates.items(), key=lambda kv: -kv[1]):
            used = shifts.get(wave, 0)
            for extra in range(1, max_shift_hours - used + 1):
                trial = _apply(current, wave, extra)
                if any(e.open_at < earliest for e in trial if e.wave_index == wave):
                    break
                if max(hourly_concurrency(trial).values()) < peak:
                    current = trial
                    shifts[wave] = used + extra
                    improved = True
                    break
            if improved:
                break
    return current, shifts
