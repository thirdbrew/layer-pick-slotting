"""Operational KPIs and run-to-run trend logging."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from slotting.models import WorldState
from slotting.rules import (
    VISIBLE_HORIZON,
    _eligible_unslotted_pallets,
    _held_pallet_ids,
    _items_with_demand,
    _items_with_waved_demand,
)
from slotting.throughput import item_has_immediate_lp_supply


@dataclass(frozen=True)
class RunMetrics:
    """Snapshot KPIs for a single rules-engine run."""

    as_of: datetime
    dead_slot_rate: float
    slot_utilization: float
    shortage_count: int
    source_slots_total: int
    source_slots_occupied: int
    dead_slots: int

    def as_row(self) -> dict[str, str | float | int]:
        return {
            "as_of": self.as_of.isoformat(sep=" ", timespec="seconds"),
            "dead_slot_rate_pct": round(self.dead_slot_rate * 100, 2),
            "slot_utilization_pct": round(self.slot_utilization * 100, 2),
            "shortage_count": self.shortage_count,
            "source_slots_total": self.source_slots_total,
            "source_slots_occupied": self.source_slots_occupied,
            "dead_slots": self.dead_slots,
        }


def _is_dead_slot_pallet(state: WorldState, pallet_id: str, demand_items: set[str]) -> bool:
    """Pallet occupies a dead slot: it cannot productively serve layer-pick demand.

    Dead when the pallet is empty, too thin to layer-pick (<= 2 layers, the
    hand-pick threshold), carries an excluded item, its item has zero
    this-line demand within the 24h visible horizon (*demand_items*), or the
    stack has **zero LP-pickable layers** right now (e.g. height-blocked above
    the clamp pick zone while still occupying the face).
    """
    from slotting.pallet_geometry import lp_pickable_layers_on_pallet

    pallet = state.pallet_by_id(pallet_id)
    if pallet is None:
        return False
    if pallet.layers_remaining <= 2:
        return True
    if pallet.item_id in state.excluded_item_ids:
        return True
    if pallet.item_id not in demand_items:
        return True
    return lp_pickable_layers_on_pallet(pallet) <= 0


def compute_metrics(state: WorldState) -> RunMetrics:
    """
    Compute baseline and ongoing KPIs for the current world state.

    - dead_slot_rate: % of source slots occupied by dead pallets (no 24h demand,
      empty/thin, excluded, or 0 LP-pickable right now)
    - slot_utilization: % of source slots occupied
    - shortage_count: items with waved demand and no immediate LP-pickable supply
    """
    source_slots = state.source_slots
    total = len(source_slots)
    occupied_ids = [s.occupied_by_pallet_id for s in source_slots if s.occupied_by_pallet_id]
    occupied = len(occupied_ids)
    demand_items = _items_with_demand(state, VISIBLE_HORIZON)
    dead = sum(1 for pid in occupied_ids if _is_dead_slot_pallet(state, pid, demand_items))

    utilization = occupied / total if total else 0.0
    dead_rate = dead / total if total else 0.0

    held_ids = _held_pallet_ids(state)
    excluded = state.excluded_item_ids
    shortage = 0
    for item_id in _items_with_waved_demand(state):
        if item_id in excluded:
            continue
        if item_has_immediate_lp_supply(state, item_id):
            continue
        if not _eligible_unslotted_pallets(state, item_id, held_ids):
            shortage += 1

    return RunMetrics(
        as_of=state.as_of,
        dead_slot_rate=dead_rate,
        slot_utilization=utilization,
        shortage_count=shortage,
        source_slots_total=total,
        source_slots_occupied=occupied,
        dead_slots=dead,
    )


METRICS_COLUMNS = [
    "as_of",
    "dead_slot_rate_pct",
    "slot_utilization_pct",
    "shortage_count",
    "source_slots_total",
    "source_slots_occupied",
    "dead_slots",
]


def append_metrics_log(metrics: RunMetrics, log_path: Path) -> None:
    """Append one metrics row to *log_path*, creating the file with headers if needed."""
    log_path = Path(log_path)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics.as_row())
