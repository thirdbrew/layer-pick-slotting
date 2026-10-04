"""Merged site profile — canonical layout-advisor input for analyze + recommend."""

from __future__ import annotations

from typing import Literal

from datetime import date

from pydantic import BaseModel, field_validator

from layout_advisor.models.design import LineType, SlotRange, ThroughputTargets
from layout_advisor.models.provenance import ProvenanceField


class SiteProfile(BaseModel):
    """Single canonical model after dual-intake normalization."""

    site_id: str
    facility_name: str | None = None
    operator: str | None = None
    contracted_customer: str | None = None
    order_date_range: tuple[date, date] | None = None

    min_layers_to_layer_pick: ProvenanceField[int]
    max_layers_hand_pick: ProvenanceField[int]

    area_sqft: ProvenanceField[float]
    source_slot_count_range: ProvenanceField[tuple[int, int]]
    donor_slot_count_range: ProvenanceField[tuple[int, int]]
    source_slot_sqft: ProvenanceField[float]
    donor_slot_sqft: ProvenanceField[float]
    allowed_line_types: ProvenanceField[list[LineType]]

    # Optional Mode A throughput goals from design_intake.json (when present).
    throughput: ThroughputTargets | None = None

    # What the orders file's quantity column actually holds. A column named
    # like layers (e.g. `pick_layers`) can carry CASES, which silently inflates
    # every demand number by the cases-per-layer ratio. "auto" detects the
    # mismatch against the SKU master and converts; set explicitly to
    # "layers" or "cases" when the feed's contract is known.
    orders_quantity_unit: Literal["auto", "layers", "cases"] = "auto"

    orders_path: str
    orders_slotting_path: str | None = None
    skus_path: str | None = None
    picks_path: str | None = None
    exclusions_path: str | None = None

    @field_validator("order_date_range", mode="before")
    @classmethod
    def _coerce_date_range(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return tuple(value)
        return value

    # Convenience accessors for slot ranges (design intake shape).
    @property
    def source_range(self) -> SlotRange:
        lo, hi = self.source_slot_count_range.value
        return SlotRange(min_count=lo, max_count=hi)

    @property
    def donor_range(self) -> SlotRange:
        lo, hi = self.donor_slot_count_range.value
        return SlotRange(min_count=lo, max_count=hi)
