"""Named ops profiles — load ReplayConfig without dissolving REFERENCE_REPLAY_CONFIG.

* ``reference`` — alias of the frozen ``REFERENCE_REPLAY_CONFIG`` (byte-identical).
* Other profiles — YAML with ``base`` + ``overrides`` (or a flat field map).

Hash stability: ``reference`` must remain ``is``-equal field-wise to
``REFERENCE_REPLAY_CONFIG`` so ``replay_config_hash`` / plan freshness stay green.
"""

from __future__ import annotations

from dataclasses import fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from layout_advisor.simulate.replay import ReplayConfig

PROFILES_DIR = Path(__file__).resolve().parents[3] / "profiles"
_VALID_FIELDS = {f.name for f in fields(ReplayConfig)}


def _coerce_value(name: str, value: Any) -> Any:
    """Coerce YAML lists to tuples for frozen ReplayConfig tuple fields."""
    if value is None:
        return value
    if isinstance(value, list):
        # spike_prep_pull_weekdays, proactive_zero_demand_realignment_hours, etc.
        return tuple(value)
    return value


def _apply_overrides(base: ReplayConfig, overrides: dict[str, Any]) -> ReplayConfig:
    unknown = sorted(set(overrides) - _VALID_FIELDS)
    if unknown:
        raise ValueError(f"Unknown ReplayConfig fields in ops profile: {unknown}")
    coerced = {k: _coerce_value(k, v) for k, v in overrides.items()}
    return replace(base, **coerced)


def _load_yaml_profile(path: Path) -> ReplayConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_id = raw.get("base", "reference")
    overrides = dict(raw.get("overrides") or {})
    # Allow flat top-level fields as overrides (excluding meta keys)
    for key, value in raw.items():
        if key in {"base", "overrides", "id", "description", "display_name", "notes"}:
            continue
        if key in _VALID_FIELDS:
            overrides[key] = value
    base = load_ops_profile(base_id) if base_id != path.stem else _reference()
    if base_id == path.stem:
        # Full dump profile — build from library defaults + all fields
        from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG

        return _apply_overrides(REFERENCE_REPLAY_CONFIG, overrides) if overrides else REFERENCE_REPLAY_CONFIG
    return _apply_overrides(base, overrides)


def _reference() -> ReplayConfig:
    from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG

    return REFERENCE_REPLAY_CONFIG


@lru_cache(maxsize=16)
def load_ops_profile(profile_id: str = "reference") -> ReplayConfig:
    """Resolve a named ops profile to a :class:`ReplayConfig`.

    ``reference`` always returns the frozen reference object (not a YAML rebuild).
    """
    if profile_id == "reference":
        return _reference()

    path = PROFILES_DIR / f"{profile_id}.yaml"
    if path.is_file():
        return _load_yaml_profile(path)

    raise FileNotFoundError(
        f"Ops profile {profile_id!r} not found (tried alias and {path})"
    )


def list_ops_profiles() -> list[str]:
    names = {"reference"}
    if PROFILES_DIR.is_dir():
        names.update(p.stem for p in PROFILES_DIR.glob("*.yaml"))
    return sorted(names)


def ops_config_from_site(
    cfg: ReplayConfig,
    *,
    min_layers_to_layer_pick: int | None = None,
    max_layers_hand_pick: int | None = None,
    layer_pick_sku_fraction: float | None = None,
    lp_ineligible_items_path: str | None = None,
) -> ReplayConfig:
    """Partial SiteProfile → ReplayConfig bridge (additive overrides only)."""
    updates: dict[str, Any] = {}
    if min_layers_to_layer_pick is not None:
        updates["min_layers_to_layer_pick"] = min_layers_to_layer_pick
    if max_layers_hand_pick is not None:
        updates["max_layers_hand_pick"] = max_layers_hand_pick
    if layer_pick_sku_fraction is not None:
        updates["layer_pick_sku_fraction"] = layer_pick_sku_fraction
    if lp_ineligible_items_path is not None and "lp_ineligible_items_path" in _VALID_FIELDS:
        updates["lp_ineligible_items_path"] = lp_ineligible_items_path
    return replace(cfg, **updates) if updates else cfg
