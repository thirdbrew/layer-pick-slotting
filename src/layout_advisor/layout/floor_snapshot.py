"""Extract pick-module occupancy from a simulation world state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from slotting.models import OrderArea, OrderStatus, WorldState
from slotting.visibility_planning import visibility_snapshot


@dataclass(frozen=True)
class SlotOccupancy:
    slot_id: str
    zone: str
    item_id: str | None
    waved: bool
    is_build: bool


@dataclass(frozen=True)
class LayerPickOverlay:
    """Layer-picker transfer drawn on the floor grid (source face → donor build)."""

    item_id: str
    layers: int
    from_slot: str
    to_slot: str | None
    order_key: str | None = None
    to_pallet_id: str | None = None


@dataclass(frozen=True)
class WaveFloorSnapshot:
    label: str
    as_of: datetime
    wave_index: int | None
    wave_send: datetime | None
    wave_due: datetime | None
    waved_skus: int
    waved_layers: int
    slots: tuple[SlotOccupancy, ...]
    layer_picks: tuple[LayerPickOverlay, ...] = ()
    # Scrubber context — populated by build_wave_floor_timeline for operator clarity.
    hourly_adds: int = 0
    hourly_removes: int = 0
    hourly_donor_exits: int = 0
    hourly_rack_promotes: int = 0
    is_realignment_hour: bool = False
    gap_hours_since_prior: int = 0
    face_swaps_from_prior: int | None = None
    faces_cleared_from_prior: int | None = None
    faces_filled_from_prior: int | None = None

    def by_slot_id(self) -> dict[str, SlotOccupancy]:
        return {s.slot_id: s for s in self.slots}

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self.slots if s.item_id)

    @property
    def empty_source_faces(self) -> int:
        return sum(1 for s in self.slots if s.zone == "source_face" and not s.item_id)

    @property
    def empty_donor_pads(self) -> int:
        return sum(1 for s in self.slots if s.zone == "donor_face" and not s.item_id)


def _item_on_pallet(state: WorldState, pallet_id: str | None) -> str | None:
    if not pallet_id:
        return None
    pallet = state.pallet_by_id(pallet_id)
    if pallet is None or pallet.layers_remaining <= 0:
        return None
    return pallet.item_id


def extract_floor_snapshot(
    state: WorldState,
    *,
    label: str,
    wave_index: int | None = None,
    wave_send: datetime | None = None,
    wave_due: datetime | None = None,
    layer_picks: tuple[LayerPickOverlay, ...] | None = None,
) -> WaveFloorSnapshot:
    snap = visibility_snapshot(state.orders, state.as_of)
    waved = {
        o.item_id
        for o in state.orders
        if o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= state.as_of + snap.wave_horizon
    }
    rows: list[SlotOccupancy] = []

    for slot in sorted(state.source_slots, key=lambda s: s.slot_id):
        face_item = _item_on_pallet(state, slot.occupied_by_pallet_id)
        rows.append(
            SlotOccupancy(
                slot_id=slot.slot_id,
                zone="source_face",
                item_id=face_item,
                waved=face_item in waved if face_item else False,
                is_build=False,
            )
        )
        backup_item = _item_on_pallet(state, slot.backup_pallet_id)
        if backup_item or slot.backup_pallet_id:
            rows.append(
                SlotOccupancy(
                    slot_id=f"{slot.slot_id}-BK",
                    zone="source_backup",
                    item_id=backup_item,
                    waved=backup_item in waved if backup_item else False,
                    is_build=False,
                )
            )

    for slot in sorted(state.resupply_rack_slots, key=lambda s: s.slot_id):
        from slotting.rack_depth import rack_depth_tier

        item = _item_on_pallet(state, slot.occupied_by_pallet_id)
        tier = rack_depth_tier(slot.slot_id)
        rows.append(
            SlotOccupancy(
                slot_id=slot.slot_id,
                zone="rack_front" if tier == 0 else "rack_back",
                item_id=item,
                waved=item in waved if item else False,
                is_build=False,
            )
        )

    for slot in sorted(state.destination_slots, key=lambda s: s.slot_id):
        face_item = _item_on_pallet(state, slot.occupied_by_pallet_id)
        is_build = bool(face_item and slot.occupied_by_pallet_id and slot.occupied_by_pallet_id.startswith("BUILD-"))
        rows.append(
            SlotOccupancy(
                slot_id=slot.slot_id,
                zone="donor_face",
                item_id=face_item,
                waved=face_item in waved if face_item else False,
                is_build=is_build,
            )
        )
        stage_item = _item_on_pallet(state, slot.backup_pallet_id)
        if stage_item or slot.backup_pallet_id:
            rows.append(
                SlotOccupancy(
                    slot_id=f"{slot.slot_id}-ST",
                    zone="donor_staging",
                    item_id=stage_item,
                    waved=stage_item in waved if stage_item else False,
                    is_build=True,
                )
            )

    return WaveFloorSnapshot(
        label=label,
        as_of=state.as_of,
        wave_index=wave_index,
        wave_send=wave_send,
        wave_due=wave_due,
        waved_skus=snap.waved_skus,
        waved_layers=snap.waved_layers,
        slots=tuple(rows),
        layer_picks=layer_picks or (),
    )
