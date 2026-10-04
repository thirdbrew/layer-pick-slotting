"""Provenance tagging for normalized site profile fields."""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ProvenanceSource(str, Enum):
    """Where a normalized value originated."""

    CUSTOMER = "customer"
    DESIGN = "design"
    DERIVED = "derived"
    DEFAULT = "default"
    ASSUMED = "assumed"


class Confidence(str, Enum):
    """Confidence in a normalized value."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProvenanceField(BaseModel, Generic[T]):
    """A value with documented origin and confidence."""

    value: T
    source: ProvenanceSource
    confidence: Confidence = Confidence.HIGH
    note: str | None = None


class NormalizationTier(str, Enum):
    """Mandatory normalization behavior tier."""

    PROVIDED = "provided"
    DERIVED = "derived"
    DEFAULT = "default"
    ASSUMED = "assumed"
    BLOCKING = "blocking"


class NormalizationGap(BaseModel):
    """A field that blocks recommendation until resolved."""

    field: str
    tier: NormalizationTier = NormalizationTier.BLOCKING
    message: str


class ExclusionJoin(BaseModel):
    """How well a customer exclusion list joined to the study SKUs.

    Gate on this before trusting the list (e.g. >=95% matched); an id that matches no
    ordered SKU excludes nothing, so a list keyed to the wrong window or item
    scheme would otherwise land silently.
    """

    total_ids: int = 0
    matched_ids: int = 0
    unmatched_ids: list[str] = Field(default_factory=list)
    join_rate: float = 0.0


class NormalizationReport(BaseModel):
    """Emitted alongside every normalize run."""

    site_id: str | None = None
    gaps: list[NormalizationGap] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    derived_fields: list[str] = Field(default_factory=list)
    defaults_applied: list[str] = Field(default_factory=list)
    exclusion_join: ExclusionJoin | None = None
