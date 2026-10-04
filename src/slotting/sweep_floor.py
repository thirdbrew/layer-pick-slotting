"""Synthetic floor snapshots for layout sweep scoring (no JDA required)."""

from __future__ import annotations

from datetime import datetime

from slotting.generate_slots import build_slots
from slotting.models import Pallet
from slotting.pallet_geometry import max_physical_pallet_layers_for_item


def build_sweep_pallets(
    item_ids: list[str],
    *,
    as_of: datetime,
    default_layers: int = 8,
    prefix: str = "SWP",
    pallets_per_sku: int = 1,
) -> list[Pallet]:
    """
    Unslotted reserve pallets — one or more eligible pallets per waved SKU.

    Layout sweep uses the same reserve pool for every candidate; source slot
    count is what limits how many ADD recommendations can succeed.
    When *pallets_per_sku* is 2+, extra pallets support backup-row staging.
    """
    if pallets_per_sku < 1:
        raise ValueError("pallets_per_sku must be >= 1")
    pallets: list[Pallet] = []
    seq = 1
    for item_id in item_ids:
        layers = min(default_layers, max_physical_pallet_layers_for_item(item_id))
        for copy in range(pallets_per_sku):
            suffix = "" if copy == 0 else f"-B{copy}"
            pallets.append(
                Pallet(
                    pallet_id=f"{prefix}-{seq:04d}{suffix}",
                    item_id=item_id,
                    layers_remaining=layers,
                    current_slot=None,
                    arrived_at=as_of,
                )
            )
            seq += 1
    return pallets


def build_sweep_floor(
    *,
    source_count: int,
    donor_count: int,
    waved_item_ids: list[str],
    as_of: datetime,
    default_layers: int = 8,
    reserve_pallets_per_sku: int = 1,
) -> tuple[list[Pallet], list]:
    """Return pallets + slots for a candidate layout at peak wave time."""
    slots = build_slots(source_count=source_count, donor_count=donor_count)
    pallets = build_sweep_pallets(
        waved_item_ids,
        as_of=as_of,
        default_layers=default_layers,
        pallets_per_sku=reserve_pallets_per_sku,
    )
    return pallets, slots
