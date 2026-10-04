"""Aisle-adjacent slot layout — source and donor roles share the same spine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SlotSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class SlotRole(str, Enum):
    SOURCE = "source"
    DONOR = "donor"


@dataclass(frozen=True)
class AisleSlot:
    """One pallet position directly beside the aisle."""

    slot_id: str
    side: SlotSide
    position_index: int
    role: SlotRole
    has_backup_row: bool
    requires_ship_access: bool

    @property
    def swappable(self) -> bool:
        """All aisle-adjacent positions may change role (source ↔ donor)."""
        return True

    @property
    def can_become_donor_in_place(self) -> bool:
        """A depleted source pallet may become the build/donor on the same pad."""
        return True


@dataclass(frozen=True)
class AisleLayoutPlan:
    """Top-down concept plan for one or two clamp pick lines."""

    source_count: int
    donor_count: int
    ratio: str
    slots: tuple[AisleSlot, ...]
    slot_width_ft: float
    slot_depth_ft: float
    aisle_width_ft: float
    backup_depth_ft: float
    aisle_length_ft: float
    footprint_sqft: float
    center_gap_ft: float = 0.0
    picker_count: int = 1
    title: str = "Layer-pick aisle concept"
    notes: tuple[str, ...] = ()

    @property
    def total_aisle_slots(self) -> int:
        return self.source_count + self.donor_count

    @property
    def dual_picker(self) -> bool:
        return self.picker_count > 1 or self.center_gap_ft > 0


def _interleave_roles(source_count: int, donor_count: int, total: int) -> list[SlotRole]:
    """
    Assign donor roles evenly along the aisle (cascade-friendly spacing).

    Remaining positions are source. Order is pick-station sequence along the aisle.
    """
    roles = [SlotRole.SOURCE] * total
    if donor_count <= 0:
        return roles
    if donor_count >= total:
        return [SlotRole.DONOR] * total

    # Place donors at roughly even indices (including ends when count is small).
    if donor_count == 1:
        donor_indices = [total // 2]
    else:
        donor_indices = [
            round(i * (total - 1) / (donor_count - 1)) for i in range(donor_count)
        ]
    seen: set[int] = set()
    for idx in donor_indices:
        idx = max(0, min(total - 1, idx))
        while idx in seen and idx < total - 1:
            idx += 1
        while idx in seen and idx > 0:
            idx -= 1
        seen.add(idx)
        roles[idx] = SlotRole.DONOR

    placed = sum(1 for r in roles if r == SlotRole.DONOR)
    if placed < donor_count:
        for i in range(total):
            if roles[i] == SlotRole.SOURCE and placed < donor_count:
                roles[i] = SlotRole.DONOR
                placed += 1
    return roles


def build_aisle_layout(
    *,
    source_count: int,
    donor_count: int,
    slot_sqft: float = 60.0,
    donor_slot_sqft: float | None = None,
    aisle_width_ft: float = 8.0,
    backup_depth_ft: float = 4.0,
    aisle_reserve_pct: float = 0.12,
    rack_only_reserve: bool = True,
    center_gap_ft: float = 0.0,
    picker_count: int = 1,
    title: str | None = None,
) -> AisleLayoutPlan:
    """
    Build a concept layout where every aisle-adjacent position is either source or donor.

    Positions are split across left and right sides of one aisle.

    Source slots may have one backup row behind the active face when *rack_only_reserve* is False.
    Donor slots require clear ship access behind the pad — no backup row may block removal.
    A depleted source pallet may become donor on the same pad without a replacement pallet.
    """
    if source_count < 0 or donor_count < 0:
        raise ValueError("source_count and donor_count must be non-negative")
    if source_count + donor_count == 0:
        raise ValueError("At least one aisle-adjacent slot is required")

    donor_sqft = donor_slot_sqft if donor_slot_sqft is not None else slot_sqft
    # Same pad size beside aisle for source and donor positions.
    pitch_sqft = max(slot_sqft, donor_sqft)
    slot_depth_ft = math.sqrt(pitch_sqft * 0.6)
    slot_width_ft = pitch_sqft / slot_depth_ft

    total = source_count + donor_count
    roles_along_aisle = _interleave_roles(source_count, donor_count, total)

    slots: list[AisleSlot] = []
    for index, role in enumerate(roles_along_aisle):
        side = SlotSide.LEFT if index % 2 == 0 else SlotSide.RIGHT
        station = index // 2
        slot_id = f"{'S' if role == SlotRole.SOURCE else 'D'}-{index + 1:02d}"
        is_source = role == SlotRole.SOURCE
        slots.append(
            AisleSlot(
                slot_id=slot_id,
                side=side,
                position_index=station,
                role=role,
                has_backup_row=is_source and not rack_only_reserve,
                requires_ship_access=not is_source,
            )
        )

    stations_per_side = max(
        1,
        max((s.position_index for s in slots if s.side == SlotSide.LEFT), default=0) + 1,
        max((s.position_index for s in slots if s.side == SlotSide.RIGHT), default=0) + 1,
    )
    aisle_length_ft = stations_per_side * slot_width_ft

    active_sqft = total * pitch_sqft
    backup_sqft = 0.0 if rack_only_reserve else source_count * slot_width_ft * backup_depth_ft
    aisle_lanes = 2 if picker_count > 1 or center_gap_ft > 0 else 1
    aisle_sqft = aisle_length_ft * aisle_width_ft * aisle_lanes
    gap_sqft = aisle_length_ft * max(0.0, center_gap_ft) if aisle_lanes > 1 else 0.0
    footprint = (active_sqft + backup_sqft + aisle_sqft + gap_sqft) * (1 + aisle_reserve_pct)

    ratio = f"{source_count}:{donor_count}"
    default_title = f"Aisle concept {ratio} ({total} positions beside aisle)"
    if rack_only_reserve:
        reserve_note = "All reserve pallets staged in 2-deep side racks: no face-adjacent rows."
    else:
        reserve_note = "Backup row only behind active source faces; clear ship access behind every donor."
    if picker_count > 1 or center_gap_ft > 0:
        team_note = (
            f"{picker_count} clamp pickers: {center_gap_ft:.0f} ft center gap between L and R pick aisles; "
            "each picker serves one flank only."
        )
    else:
        team_note = "One aisle picker + one assist (resupply): a 2-person team."
    notes = (
        "All positions are directly adjacent to the aisle.",
        "Each position may be source (pick from) or donor/build (pick to); roles swappable.",
        "Depleted source pallet may become donor on the same pad: no replacement needed.",
        reserve_note,
        "Assist must reach donor builds for outbound: resupply must not block ship path.",
        team_note,
    )

    return AisleLayoutPlan(
        source_count=source_count,
        donor_count=donor_count,
        ratio=ratio,
        slots=tuple(slots),
        slot_width_ft=round(slot_width_ft, 2),
        slot_depth_ft=round(slot_depth_ft, 2),
        aisle_width_ft=aisle_width_ft,
        backup_depth_ft=backup_depth_ft,
        aisle_length_ft=round(aisle_length_ft, 2),
        footprint_sqft=round(footprint, 1),
        center_gap_ft=max(0.0, center_gap_ft),
        picker_count=max(1, picker_count),
        title=title or default_title,
        notes=notes,
    )
