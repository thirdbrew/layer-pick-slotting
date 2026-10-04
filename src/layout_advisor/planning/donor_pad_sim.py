"""Sim-actual donor pad occupancy (action half).

Reconstructs donor pad open/close episodes from replay events so pad-minute
reports can compare plan assumptions (close at wave due) vs sim truth (free at
build completion via ``DEST_STAGE_COMPLETE``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from layout_advisor.planning.donor_pad_minutes import DonorEpisode, hourly_concurrency
from slotting.models import RecommendationAction, SlotType

DONOR_PAD_EVENT_FIELDS = (
    "at",
    "event",
    "slot",
    "item_id",
    "wave_index",
    "layers",
    "detail",
)


@dataclass(frozen=True)
class DonorPadSimEvent:
    at: datetime
    event: str  # "open" | "close"
    slot: str
    item_id: str | None = None
    wave_index: int | None = None
    layers: int | None = None
    detail: str = ""


def log_donor_opens(
    log: list[DonorPadSimEvent],
    *,
    at: datetime,
    batch: list[tuple[str, int, str]],
    wave_index_by_item: dict[str, int] | None = None,
    detail: str = "",
) -> None:
    """Append open events for ``(item_id, layers, donor_slot)`` tuples."""
    wave_index_by_item = wave_index_by_item or {}
    for item_id, layers, slot in batch:
        log.append(
            DonorPadSimEvent(
                at=at,
                event="open",
                slot=slot,
                item_id=item_id,
                wave_index=wave_index_by_item.get(item_id),
                layers=layers,
                detail=detail,
            )
        )


def log_donor_closes_from_applied(
    log: list[DonorPadSimEvent],
    *,
    at: datetime,
    applied: list,
    destination_slot_ids: set[str],
) -> None:
    """Record pad frees when completed builds exit to wrapping."""
    for action in applied:
        if action.action != RecommendationAction.FLAG_REMOVE:
            continue
        slot_id = action.slot_id
        if not slot_id or slot_id not in destination_slot_ids:
            continue
        detail = action.detail or ""
        if "wrapping" not in detail.lower() and "completed build" not in detail.lower():
            continue
        log.append(
            DonorPadSimEvent(
                at=at,
                event="close",
                slot=slot_id,
                item_id=action.item_id,
                detail=detail,
            )
        )


def donor_episodes_from_sim_events(
    events: list[DonorPadSimEvent],
    *,
    day_end: datetime,
) -> list[DonorEpisode]:
    """Pair opens with the next close on the same donor slot."""
    opens: dict[str, tuple[datetime, int | None, int | None, str | None]] = {}
    episodes: list[DonorEpisode] = []
    for ev in sorted(events, key=lambda e: (e.at, e.event != "open", e.slot)):
        if ev.event == "open":
            opens[ev.slot] = (ev.at, ev.wave_index, ev.layers, ev.item_id)
        elif ev.event == "close" and ev.slot in opens:
            open_at, wave_index, _layers, _item = opens.pop(ev.slot)
            episodes.append(
                DonorEpisode(
                    slot=ev.slot,
                    open_at=open_at,
                    close_at=max(ev.at, open_at),
                    wave_index=wave_index,
                    wave_due=None,
                )
            )
    for slot, (open_at, wave_index, _layers, _item) in opens.items():
        episodes.append(
            DonorEpisode(
                slot=slot,
                open_at=open_at,
                close_at=day_end,
                wave_index=wave_index,
                wave_due=None,
            )
        )
    episodes.sort(key=lambda e: (e.open_at, e.slot))
    return episodes


def wave_index_by_item(wave_opens: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in wave_opens:
        out[row.item_id] = row.wave_index
    return out


def destination_slot_ids_from_state(state) -> set[str]:
    return {s.slot_id for s in state.slots if s.slot_type == SlotType.DESTINATION}


def write_donor_pad_sim_csv(events: list[DonorPadSimEvent], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DONOR_PAD_EVENT_FIELDS)
        writer.writeheader()
        for ev in events:
            writer.writerow(
                {
                    "at": ev.at.isoformat(sep=" "),
                    "event": ev.event,
                    "slot": ev.slot,
                    "item_id": ev.item_id or "",
                    "wave_index": ev.wave_index if ev.wave_index is not None else "",
                    "layers": ev.layers if ev.layers is not None else "",
                    "detail": ev.detail,
                }
            )
    return path


def load_donor_pad_sim_csv(path: Path) -> list[DonorPadSimEvent]:
    events: list[DonorPadSimEvent] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            events.append(
                DonorPadSimEvent(
                    at=datetime.fromisoformat(row["at"]),
                    event=row["event"],
                    slot=row["slot"],
                    item_id=row["item_id"] or None,
                    wave_index=int(row["wave_index"]) if row.get("wave_index") else None,
                    layers=int(row["layers"]) if row.get("layers") else None,
                    detail=row.get("detail") or "",
                )
            )
    return events


def summarize_episodes(
    episodes: list[DonorEpisode],
    *,
    label: str,
) -> dict:
    if not episodes:
        return {"label": label, "episodes": 0}
    conc = hourly_concurrency(episodes)
    durs = sorted((e.close_at - e.open_at).total_seconds() / 60 for e in episodes)
    pad_minutes = sum(durs)
    return {
        "label": label,
        "episodes": len(episodes),
        "pad_minutes": int(pad_minutes),
        "peak_concurrent": max(conc.values()) if conc else 0,
        "median_dwell_min": round(durs[len(durs) // 2], 1) if durs else 0.0,
        "mean_dwell_min": round(pad_minutes / len(episodes), 1) if episodes else 0.0,
    }


def compare_occupancy_models(
    rows: list[dict],
    *,
    day_end: datetime,
    default_due: datetime | None,
    sim_events: list[DonorPadSimEvent] | None = None,
) -> dict:
    """Compare wave-due plan model, build-complete plan rows, and sim truth."""
    from layout_advisor.planning.donor_pad_minutes import donor_episodes, donor_episodes_build_complete

    due_eps = donor_episodes(rows, day_end=day_end, default_due=default_due)
    bld_eps = donor_episodes_build_complete(rows, day_end=day_end)
    sim_eps = (
        donor_episodes_from_sim_events(sim_events, day_end=day_end) if sim_events else []
    )
    out = {
        "wave_due": summarize_episodes(due_eps, label="wave_due"),
        "build_complete_csv": summarize_episodes(bld_eps, label="build_complete_csv"),
    }
    if sim_eps:
        out["sim_truth"] = summarize_episodes(sim_eps, label="sim_truth")
    return out
