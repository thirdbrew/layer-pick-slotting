"""Layout-advisor configuration defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from layout_advisor.config.ops_profile import list_ops_profiles, load_ops_profile, ops_config_from_site

__all__ = [
    "LayoutAdvisorDefaults",
    "list_ops_profiles",
    "load_defaults",
    "load_ops_profile",
    "ops_config_from_site",
]


class LayoutAdvisorDefaults(BaseModel):
    """v1 defaults — override per project via intake."""

    min_layers_to_layer_pick: int = 3
    max_layers_hand_pick: int = 2
    source_slot_sqft: float = 60.0
    donor_slot_sqft: float = 48.0
    avg_layers_per_pallet: float = 6.0
    benchmark_seconds_per_layer: float = 45.0
    wave_horizon_hours: int = 4
    planning_horizon_hours: int = 24
    top_sku_buffer_slots: int = 2
    source_utilization: float = Field(default=0.85, ge=0.1, le=1.0)
    default_wave_pick_time: str = "06:00:00"


_DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.yaml"


@lru_cache(maxsize=1)
def load_defaults() -> LayoutAdvisorDefaults:
    """Load packaged defaults.yaml (cached)."""
    if not _DEFAULTS_PATH.exists():
        return LayoutAdvisorDefaults()
    raw: dict[str, Any] = yaml.safe_load(_DEFAULTS_PATH.read_text(encoding="utf-8")) or {}
    return LayoutAdvisorDefaults.model_validate(raw)
