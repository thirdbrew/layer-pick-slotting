"""Recommendation outputs from the layout-advisor analyze + recommend pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from layout_advisor.models.design import LineType
from layout_advisor.models.provenance import Confidence


class PickMethodRecommendation(str, Enum):
    """Recommended fulfillment method mix."""

    LAYER_PRIMARY = "layer_primary"
    LAYER_CASE_SPLIT = "layer+case_split"
    PALLET_FORWARD = "pallet_forward"
    DYNAMIC_BY_LAYERS = "dynamic_by_layers"


class LayoutRecommendation(BaseModel):
    """Source:donor slot layout."""

    source_slots: int = Field(ge=0)
    donor_slots: int = Field(ge=0)
    ratio: str
    feasible: bool = True
    notes: str | None = None


class RotationMetrics(BaseModel):
    """Pallet rotation estimates for source slots."""

    source_turns_per_day: float | None = None
    peak_replenishments_per_hour: float | None = None
    top_sku_concentration_pct: float | None = None
    notes: str | None = None


class AlternativeRecommendation(BaseModel):
    """Secondary viable configuration with tradeoffs."""

    pick_line_type: LineType
    pick_method: PickMethodRecommendation
    layout: LayoutRecommendation
    tradeoffs: str


class Recommendation(BaseModel):
    """Primary layout-advisor recommendation package."""

    site_id: str
    pick_line_type: LineType
    pick_method: PickMethodRecommendation
    layout: LayoutRecommendation
    rotation: RotationMetrics
    confidence: Confidence
    confidence_reasons: list[str] = Field(default_factory=list)
    alternatives: list[AlternativeRecommendation] = Field(default_factory=list)
