"""Physical pallet height limits — outbound build caps and clamp pick reach."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from slotting.models import Pallet, WorldState
from slotting.sku_master import _load_rows

# GMA outbound build pallet and clamp layer-pick reach (inches).
MAX_ORDER_PALLET_HEIGHT_IN = 100.0
MAX_LP_PICK_HEIGHT_IN = 48.0
DEFAULT_LAYER_HEIGHT_IN = 9.0  # illustrative default when height_in is missing


@dataclass(frozen=True)
class PalletHeightLimits:
    """Active height limits for build caps and source-face LP reach."""

    max_order_pallet_height_in: float = MAX_ORDER_PALLET_HEIGHT_IN
    max_lp_pick_height_in: float = MAX_LP_PICK_HEIGHT_IN
    skus_path: Path | None = None
    # Pick-depth model: False = bottom pick zone (layers above the reach cap
    # are unpickable — library default); True = clamp takes <=cap bites off the
    # TOP, so every layer is eventually pickable. Which is physically right
    # depends on the clamp; it is a configurable assumption, not a finding.
    top_pick_model: bool = False


_DEFAULT_LIMITS = PalletHeightLimits()
_active_limits: contextvars.ContextVar[PalletHeightLimits] = contextvars.ContextVar(
    "pallet_height_limits",
    default=_DEFAULT_LIMITS,
)


def active_height_limits() -> PalletHeightLimits:
    return _active_limits.get()


@contextmanager
def pallet_height_limits(limits: PalletHeightLimits | None = None) -> Iterator[None]:
    """Scope sim / replay height policy (per-SKU heights come from the SKU master)."""
    token = _active_limits.set(limits or PalletHeightLimits())
    try:
        yield
    finally:
        _active_limits.reset(token)


def default_site_skus_path() -> Path | None:
    """Repo-relative local SKU master (``data/site/skus.csv``) when present."""
    candidate = Path(__file__).resolve().parents[2] / "data" / "site" / "skus.csv"
    return candidate if candidate.is_file() else None


def resolve_height_limits(skus_path: Path | None = None) -> PalletHeightLimits:
    """Height policy for CLI / optimize scoring (local SKU master when available)."""
    path = skus_path if skus_path is not None else default_site_skus_path()
    return PalletHeightLimits(skus_path=path)


class _LineWithLayers(Protocol):
    item_id: str
    layers: int
    layer_height_in: float | None
    layer_weight_lb: float | None


def sku_layer_height_in(
    item_id: str,
    *,
    skus_path: Path | None = None,
    default: float = DEFAULT_LAYER_HEIGHT_IN,
) -> float:
    """Per-layer product height from SKU master (case/layer height in inches)."""
    path = skus_path if skus_path is not None else active_height_limits().skus_path
    row = _load_rows(path).get(item_id, {})
    raw = row.get("height_in")
    try:
        val = float(raw)
        return val if val > 0 else default
    except (TypeError, ValueError):
        return default


def max_physical_pallet_layers_for_item(
    item_id: str,
    *,
    max_height_in: float | None = None,
    skus_path: Path | None = None,
) -> int:
    """Max layers on one physical pallet for *item_id* without exceeding *max_height_in*."""
    limits = active_height_limits()
    cap = limits.max_order_pallet_height_in if max_height_in is None else max_height_in
    h = sku_layer_height_in(item_id, skus_path=skus_path)
    if cap <= 0 or h <= 0:
        return 1
    return max(1, int(cap // h))


def lp_pickable_layers_on_pallet(
    pallet: Pallet,
    *,
    layer_height: float | None = None,
    max_pick_height_in: float | None = None,
    skus_path: Path | None = None,
) -> int:
    """
    Layers on *pallet* reachable by the clamp (bottom pick zone only).

    When the remaining stack fits within the pick zone, all layers are pickable.
    """
    if pallet.layers_remaining <= 0:
        return 0
    limits = active_height_limits()
    pick_cap = limits.max_lp_pick_height_in if max_pick_height_in is None else max_pick_height_in
    path = skus_path if skus_path is not None else limits.skus_path
    h = layer_height if layer_height is not None and layer_height > 0 else sku_layer_height_in(
        pallet.item_id, skus_path=path
    )
    if pick_cap <= 0 or limits.top_pick_model:
        return pallet.layers_remaining
    total_height = pallet.layers_remaining * h
    if total_height <= pick_cap:
        return pallet.layers_remaining
    return min(pallet.layers_remaining, int(pick_cap // h))


def face_lp_pickable_layers_for_item(
    state: WorldState,
    item_id: str,
    *,
    max_pick_height_in: float | None = None,
    skus_path: Path | None = None,
) -> int:
    """Sum of LP-pickable layers on source face(s) for *item_id*."""
    return sum(
        lp_pickable_layers_on_pallet(
            pallet,
            max_pick_height_in=max_pick_height_in,
            skus_path=skus_path,
        )
        for pallet in state.pallets_in_source_slots()
        if pallet.item_id == item_id and pallet.layers_remaining > 0
    )


def _line_heights_and_weights(
    lines: tuple[_LineWithLayers, ...],
    *,
    skus_path: Path | None,
    default_layer_height_in: float,
) -> tuple[dict[str, float], dict[str, float]]:
    path = skus_path if skus_path is not None else active_height_limits().skus_path
    heights: dict[str, float] = {}
    weights: dict[str, float] = {}
    for line in lines:
        if line.item_id in heights:
            continue
        h = line.layer_height_in
        heights[line.item_id] = (
            h if h is not None and h > 0 else sku_layer_height_in(line.item_id, skus_path=path)
        )
        if heights[line.item_id] <= 0:
            heights[line.item_id] = default_layer_height_in
        w = line.layer_weight_lb
        weights[line.item_id] = w if w is not None and w > 0 else 0.0
    return heights, weights


def take_layers_for_build_pallet(
    lines: tuple[_LineWithLayers, ...],
    *,
    layer_cap: int,
    max_weight_lb: float,
    max_build_height_in: float | None = None,
    skus_path: Path | None = None,
) -> tuple[dict[str, int], int]:
    """
    Layer-by-layer greedy fill for one build pallet (height + weight aware).

    Interleaves SKUs by largest remaining count — matches height-cap math and
    avoids block-stacking tall SKUs past the 100″ cap.
    """
    if not lines or layer_cap <= 0:
        return {}, 0

    limits = active_height_limits()
    height_cap = (
        limits.max_order_pallet_height_in if max_build_height_in is None else max_build_height_in
    )
    remaining = {line.item_id: line.layers for line in lines}
    heights, weights = _line_heights_and_weights(lines, skus_path=skus_path, default_layer_height_in=DEFAULT_LAYER_HEIGHT_IN)
    taken = {iid: 0 for iid in remaining}
    placed = 0
    height_used = 0.0
    weight_used = 0.0

    while placed < layer_cap and any(v > 0 for v in remaining.values()):
        progressed = False
        for iid in sorted(remaining, key=lambda k: remaining[k], reverse=True):
            if remaining[iid] <= 0:
                continue
            h = heights[iid]
            w = weights[iid]
            if height_cap > 0 and height_used + h > height_cap:
                continue
            if max_weight_lb > 0 and w > 0 and weight_used + w > max_weight_lb:
                continue
            height_used += h
            weight_used += w
            taken[iid] += 1
            remaining[iid] -= 1
            placed += 1
            progressed = True
            if placed >= layer_cap:
                break
        if not progressed:
            break

    return {iid: count for iid, count in taken.items() if count > 0}, placed


def build_pallet_height_in(
    lines: tuple[_LineWithLayers, ...],
    taken: dict[str, int],
    *,
    skus_path: Path | None = None,
) -> float:
    """Total inches for a taken layer mix on one build pallet."""
    heights, _ = _line_heights_and_weights(lines, skus_path=skus_path, default_layer_height_in=DEFAULT_LAYER_HEIGHT_IN)
    return sum(heights[iid] * count for iid, count in taken.items())


def max_build_layers_for_height(
    lines: tuple[_LineWithLayers, ...],
    *,
    max_build_height_in: float | None = None,
    default_layer_height_in: float = DEFAULT_LAYER_HEIGHT_IN,
    skus_path: Path | None = None,
) -> int:
    """
    Max layers on one outbound build pallet without exceeding the height cap.

    Greedy placement (largest remaining SKU first) for mixed-SKU pallets.
    """
    if not lines:
        return 0
    total_demand = sum(line.layers for line in lines)
    taken, placed = take_layers_for_build_pallet(
        lines,
        layer_cap=total_demand,
        max_weight_lb=0.0,
        max_build_height_in=max_build_height_in,
        skus_path=skus_path,
    )
    _ = taken
    return placed
