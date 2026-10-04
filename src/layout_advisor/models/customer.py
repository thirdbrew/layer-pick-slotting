"""Customer / operations intake models."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class PickMethod(str, Enum):
    """Historical or current pick method."""

    LAYER = "layer"
    CASE = "case"
    PALLET = "pallet"
    MIXED = "mixed"


class OrderIntakeRow(BaseModel):
    """Single order line from customer intake."""

    item_id: str
    ship_date: date
    qty_layers: int = Field(ge=0)
    qty_cases: int | None = Field(default=None, ge=0)
    order_id: str


class PickHistoryRow(BaseModel):
    """Optional picking history row."""

    item_id: str
    method_used: PickMethod
    layers_picked: int = Field(ge=0)
    labor_seconds: float | None = Field(default=None, ge=0)


class PolicyIntake(BaseModel):
    """Layer-pick policy thresholds."""

    min_layers_to_layer_pick: int = Field(default=3, ge=1)
    max_layers_hand_pick: int = Field(default=2, ge=0)


class SkuMasterRow(BaseModel):
    """Partial SKU master data."""

    item_id: str
    layers_per_pallet: int | None = Field(default=None, ge=1)
    cases_per_layer: int | None = Field(default=None, ge=1)
    length_in: float | None = Field(default=None, gt=0)
    width_in: float | None = Field(default=None, gt=0)
    height_in: float | None = Field(default=None, gt=0)
    weight_lb: float | None = Field(default=None, gt=0)
    abc_class: str | None = Field(default=None, description="A/B/C from velocity analysis")
    abc_shipped_cases: int | None = Field(default=None, ge=0)


class ExclusionRow(BaseModel):
    """SKU excluded from layer pick."""

    item_id: str
    reason: str


class VolumeIntake(BaseModel):
    """Optional facility volume metadata."""

    facility: str | None = None
    date_range_start: date | None = None
    date_range_end: date | None = None
    peak_days_note: str | None = None


class CustomerIntake(BaseModel):
    """Normalized customer / operations intake bundle."""

    site_id: str | None = None
    facility_name: str | None = None
    operator: str | None = None
    contracted_customer: str | None = None
    orders: list[OrderIntakeRow] = Field(default_factory=list)
    pick_history: list[PickHistoryRow] = Field(default_factory=list)
    policy: PolicyIntake = Field(default_factory=PolicyIntake)
    current_method: PickMethod | None = None
    current_method_notes: str | None = None
    skus: list[SkuMasterRow] = Field(default_factory=list)
    exclusions: list[ExclusionRow] = Field(default_factory=list)
    volume: VolumeIntake | None = None
