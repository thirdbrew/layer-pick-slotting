"""Hourly travel KPIs — picker station span and flank balance for waved demand."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from slotting.models import WorldState
from slotting.throughput import WAVED_HORIZON, _waved_demand_by_item


@dataclass(frozen=True)
class WaveTravelKpi:
    """Travel proxy for current waved window (default 4h horizon)."""

    waved_layers: int
    waved_layers_left: int
    waved_layers_right: int
    flank_balance_pct: float
    lp_ready_station_span: int
    planned_station_span: int
    lp_ready_pick_faces: int


def preferred_faces_from_assignments(assignments) -> dict[str, str]:
    """Build item → source face map from setup assignments."""
    out: dict[str, str] = {}
    for row in assignments:
        if getattr(row, "location", None) == "source_face" and getattr(row, "slot_id", None):
            out[row.item_id] = row.slot_id
    return out


def _face_for_item(
    state: WorldState,
    item_id: str,
    *,
    preferred_face_by_item: Mapping[str, str],
) -> str | None:
    """Occupied source face for *item_id*, else planned preferred face."""
    for slot in state.source_slots:
        if not slot.occupied_by_pallet_id:
            continue
        pallet = state.pallet_by_id(slot.occupied_by_pallet_id)
        if pallet is None or pallet.item_id != item_id:
            continue
        if state.is_eligible_for_line(pallet):
            return slot.slot_id
    pref = preferred_face_by_item.get(item_id)
    return pref if pref else None


def _item_lp_ready_on_face(state: WorldState, item_id: str) -> bool:
    for slot in state.source_slots:
        if not slot.occupied_by_pallet_id:
            continue
        pallet = state.pallet_by_id(slot.occupied_by_pallet_id)
        if pallet is None or pallet.item_id != item_id:
            continue
        if state.is_eligible_for_line(pallet) and pallet.layers_remaining > 0:
            return True
    return False


def compute_wave_travel_kpi(
    state: WorldState,
    *,
    preferred_face_by_item: Mapping[str, str] | None = None,
    face_station_by_id: Mapping[str, int] | None = None,
    face_flank_by_id: Mapping[str, str] | None = None,
) -> WaveTravelKpi:
    """
    Station span and flank balance for waved layer demand in the WAVED horizon.

    *lp_ready_station_span* — max−min station among LP-ready waved SKUs on face.
    *planned_station_span* — same using planned/occupied pick faces for all waved SKUs.
    *flank_balance_pct* — 100 × min(L,R layers) / max(L,R layers).
    """
    preferred = preferred_face_by_item or {}
    stations = face_station_by_id or {}
    flanks = face_flank_by_id or {}

    waved = _waved_demand_by_item(state, horizon=WAVED_HORIZON)
    if not waved:
        return WaveTravelKpi(
            waved_layers=0,
            waved_layers_left=0,
            waved_layers_right=0,
            flank_balance_pct=100.0,
            lp_ready_station_span=0,
            planned_station_span=0,
            lp_ready_pick_faces=0,
        )

    left_layers = 0
    right_layers = 0
    planned_stations: list[int] = []
    ready_stations: list[int] = []

    for item_id, layers in waved.items():
        if item_id in state.excluded_item_ids:
            continue
        face_id = _face_for_item(state, item_id, preferred_face_by_item=preferred)
        if face_id is None:
            continue
        station = stations.get(face_id)
        flank = (flanks.get(face_id) or "L").upper()
        if flank in ("R", "RIGHT"):
            right_layers += layers
        else:
            left_layers += layers
        if station is not None:
            planned_stations.append(station)
            if _item_lp_ready_on_face(state, item_id):
                ready_stations.append(station)

    total = sum(waved.values())
    max_flank = max(left_layers, right_layers)
    balance = round(100.0 * min(left_layers, right_layers) / max_flank, 1) if max_flank else 100.0

    def _span(vals: list[int]) -> int:
        if len(vals) < 2:
            return 0
        return max(vals) - min(vals)

    return WaveTravelKpi(
        waved_layers=total,
        waved_layers_left=left_layers,
        waved_layers_right=right_layers,
        flank_balance_pct=balance,
        lp_ready_station_span=_span(ready_stations),
        planned_station_span=_span(planned_stations),
        lp_ready_pick_faces=len(set(ready_stations)),
    )
