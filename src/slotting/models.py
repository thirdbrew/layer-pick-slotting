"""Pydantic domain models for pallet slot decision support."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SlotType(str, Enum):
    """Physical slot role in the layer-pick area."""

    SOURCE = "source"
    DESTINATION = "destination"
    RESUPPLY_RACK = "resupply_rack"


class OrderStatus(str, Enum):
    """Order commitment horizon."""

    VISIBLE = "visible"
    WAVED = "waved"


class OrderArea(str, Enum):
    """Which fulfillment area owns the order line."""

    THIS_LINE = "this_line"
    OTHER_AREA = "other_area"


class RecommendationAction(str, Enum):
    """Recommended operator action (advisory only)."""

    FLAG_REMOVE = "FLAG_REMOVE"
    DO_NOT_TOUCH = "DO_NOT_TOUCH"
    FLAG_ADD = "FLAG_ADD"
    FLAG_WAIT = "FLAG_WAIT"
    FLAG_ACTIVATE_DEST = "FLAG_ACTIVATE_DEST"
    FLAG_DEACTIVATE_DEST = "FLAG_DEACTIVATE_DEST"
    FLAG_BACKUP_PROMOTE = "FLAG_BACKUP_PROMOTE"
    FLAG_RELOCATE_TO_RESUPPLY = "FLAG_RELOCATE_TO_RESUPPLY"
    FLAG_STORAGE_RETURN = "FLAG_STORAGE_RETURN"
    FLAG_AISLE_STAGE = "FLAG_AISLE_STAGE"
    FLAG_RACK_PROMOTE = "FLAG_RACK_PROMOTE"
    FLAG_FACE_REFRESH = "FLAG_FACE_REFRESH"


class RuleId(str, Enum):
    """Identifier for the rule that produced a recommendation."""

    LOW_LAYERS = "LOW_LAYERS"
    ZERO_DEMAND = "ZERO_DEMAND"
    EXCLUDED_ITEM = "EXCLUDED_ITEM"
    CROSS_AREA_HOLD = "CROSS_AREA_HOLD"
    ADD_WAVED_DEMAND = "ADD_WAVED_DEMAND"
    WAIT_SUPPLY_SHORTAGE = "WAIT_SUPPLY_SHORTAGE"
    DEST_ASSIGN_WAVED = "DEST_ASSIGN_WAVED"
    DEST_CLOSE_IDLE = "DEST_CLOSE_IDLE"
    DEST_WAIT_NO_FACE = "DEST_WAIT_NO_FACE"
    DEST_STAGE_COMPLETE = "DEST_STAGE_COMPLETE"
    WAVE_ACTIVATE_DEST = "WAVE_ACTIVATE_DEST"
    WAVE_DEACTIVATE_DEST = "WAVE_DEACTIVATE_DEST"
    BACKUP_PROMOTE = "BACKUP_PROMOTE"
    RACK_PROMOTE = "RACK_PROMOTE"
    FACE_REFRESH = "FACE_REFRESH"
    ADD_VISIBLE_PRESTAGE = "ADD_VISIBLE_PRESTAGE"
    ZERO_DEMAND_RECLAIM = "ZERO_DEMAND_RECLAIM"
    VELOCITY_TWIN_RECLAIM = "VELOCITY_TWIN_RECLAIM"


class Pallet(BaseModel):
    """A physical pallet that may or may not occupy a source slot."""

    pallet_id: str
    item_id: str
    layers_remaining: int = Field(ge=0)
    current_slot: str | None = None
    arrived_at: datetime
    removed_from_pick_at: datetime | None = None
    last_source_slot_id: str | None = None
    # Multi-SKU outbound build pallet (order-ahead): one physical pallet, many lines.
    build_lines: dict[str, int] | None = None

    def destination_skus(self) -> set[str]:
        if self.build_lines:
            return {sku for sku, layers in self.build_lines.items() if layers > 0}
        if self.layers_remaining > 0:
            return {self.item_id}
        return set()

    def serves_item(self, item_id: str) -> bool:
        if self.build_lines is not None:
            return self.build_lines.get(item_id, 0) > 0
        return self.item_id == item_id and self.layers_remaining > 0

    def has_waved_demand(self, waved_items: set[str]) -> bool:
        return bool(self.destination_skus() & waved_items)


class Slot(BaseModel):
    """A source or destination position in the layer-pick area."""

    slot_id: str
    slot_type: SlotType
    occupied_by_pallet_id: str | None = None
    backup_pallet_id: str | None = None


class OrderLine(BaseModel):
    """A single order line visible in the planning horizon."""

    order_id: str
    item_id: str
    qty_layers: int = Field(ge=1)
    due_at: datetime
    status: OrderStatus
    area: OrderArea
    arrived_at: datetime | None = None


class Exclusion(BaseModel):
    """Item that must not be used in this line."""

    item_id: str
    reason: str


class Recommendation(BaseModel):
    """Advisory output for a human operator."""

    action: RecommendationAction
    pallet_id: str | None = None
    item_id: str | None = None
    slot_id: str | None = None
    reason: str
    rule: RuleId


class WorldState(BaseModel):
    """Snapshot of pallets, slots, orders, and exclusions at a point in time."""

    pallets: list[Pallet]
    slots: list[Slot]
    orders: list[OrderLine]
    exclusions: list[Exclusion]
    as_of: datetime

    model_config = {"frozen": True}

    @property
    def excluded_item_ids(self) -> set[str]:
        return {e.item_id for e in self.exclusions}

    @property
    def source_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.slot_type == SlotType.SOURCE]

    @property
    def destination_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.slot_type == SlotType.DESTINATION]

    @property
    def resupply_rack_slots(self) -> list[Slot]:
        return [s for s in self.slots if s.slot_type == SlotType.RESUPPLY_RACK]

    def pallet_by_id(self, pallet_id: str) -> Pallet | None:
        for pallet in self.pallets:
            if pallet.pallet_id == pallet_id:
                return pallet
        return None

    def pallets_in_source_slots(self) -> list[Pallet]:
        """Pallets currently assigned to a source slot."""
        source_slot_ids = {s.slot_id for s in self.source_slots}
        return [
            p
            for p in self.pallets
            if p.current_slot is not None and p.current_slot in source_slot_ids
        ]

    def pallets_in_destination_slots(self) -> list[Pallet]:
        """Build pallets currently assigned to a destination slot."""
        dest_slot_ids = {s.slot_id for s in self.destination_slots}
        return [
            p
            for p in self.pallets
            if p.current_slot is not None and p.current_slot in dest_slot_ids
        ]

    def unslotted_pallets(self) -> list[Pallet]:
        return [p for p in self.pallets if p.current_slot is None or p.current_slot == ""]

    def item_in_source_slot(self, item_id: str) -> bool:
        return any(
            p.item_id == item_id and p.layers_remaining > 0
            for p in self.pallets_in_source_slots()
        )

    def item_on_destination_slot(self, item_id: str) -> bool:
        return any(p.serves_item(item_id) for p in self.pallets_in_destination_slots())

    def is_eligible_for_line(self, pallet: Pallet) -> bool:
        """Pallet can serve layer-pick demand (non-empty, not excluded, not held)."""
        if pallet.layers_remaining <= 0:
            return False
        if pallet.item_id in self.excluded_item_ids:
            return False
        return True
