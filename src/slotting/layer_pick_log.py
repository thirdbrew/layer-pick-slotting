"""Layer-picker work log — layers taken from source faces onto donor builds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from slotting.models import Pallet, WorldState


@dataclass(frozen=True)
class LayerPickEvent:
    """One layer-pick transfer recorded during hourly inventory depletion."""

    as_of: datetime
    item_id: str
    layers: int
    from_pallet_id: str
    from_slot: str
    to_pallet_id: str | None
    to_slot: str | None
    order_key: str | None = None


def active_donor_build(pallets: list[Pallet], item_id: str) -> Pallet | None:
    """Donor build pallet on a destination face for *item_id*, if any."""
    for pallet in pallets:
        if not pallet.serves_item(item_id):
            continue
        if not pallet.current_slot or pallet.layers_remaining <= 0:
            continue
        if pallet.pallet_id.startswith("BUILD-"):
            return pallet
    return None


def donor_build_for_item(state: WorldState, item_id: str) -> tuple[Pallet | None, str | None]:
    build = active_donor_build(state.pallets, item_id)
    if build is None:
        return None, None
    return build, build.current_slot
