"""Design team intake models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class LineType(str, Enum):
    """Allowed pick line configurations."""

    CLAMP_LAYER_PICK = "clamp_layer_pick"
    HYBRID_MANUAL_LP = "hybrid_manual_lp"
    MANUAL_LP_ZONE = "manual_lp_zone"
    NOT_VIABLE = "not_viable"


class AreaIntake(BaseModel):
    """Physical area constraints."""

    length_ft: float = Field(gt=0)
    width_ft: float = Field(gt=0)
    height_ft: float | None = Field(default=None, gt=0)
    obstructions_notes: str | None = None


class SlotRange(BaseModel):
    """Min/max slot counts from design."""

    min_count: int = Field(ge=0)
    max_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _min_lte_max(self) -> SlotRange:
        if self.min_count > self.max_count:
            raise ValueError("min_count must be <= max_count")
        return self


class FootprintIntake(BaseModel):
    """Per-slot footprint defaults."""

    source_slot_sqft: float | None = Field(default=None, gt=0)
    donor_slot_sqft: float | None = Field(default=None, gt=0)


class ThroughputTargets(BaseModel):
    """Optional throughput goals."""

    target_layers_per_hour: float | None = Field(default=None, gt=0)
    target_orders_per_day: int | None = Field(default=None, ge=1)
    #: Hours per ship day the aisle is actually worked — the denominator for
    #: every picker-hour, utilization and throughput figure the frontier
    #: publishes. Undeclared means the pack assumes a single 8 h shift and says
    #: so. Declare it when order-ahead dispatch spreads ship-day work well past
    #: one shift, or every per-hour figure is overstated.
    operating_window_hours: float | None = Field(default=None, gt=0)


class EquipmentIntake(BaseModel):
    """Optional equipment details."""

    clamp_model: str | None = None
    has_conveyor: bool | None = None
    automation_level: str | None = None


class DesignIntake(BaseModel):
    """Normalized design team intake bundle."""

    project_id: str | None = None
    area: AreaIntake
    allowed_line_types: list[LineType] = Field(min_length=1)
    source_slot_range: SlotRange
    donor_slot_range: SlotRange
    footprint: FootprintIntake = Field(default_factory=FootprintIntake)
    throughput: ThroughputTargets | None = None
    equipment: EquipmentIntake | None = None
