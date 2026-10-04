"""Reserve budget — 1:1 cap; all staging in 2-deep side racks (no face-adjacent rows)."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_RESERVE_CAP_RATIO = 1.0
DEFAULT_RACK_DEPTH_TIERS = 1


@dataclass(frozen=True)
class ReserveBudget:
    """Staging pallet slots in the pick module (excludes active aisle faces)."""

    pick_positions: int
    reserve_cap: int
    reserve_cap_ratio: float
    rack_only_reserve: bool
    source_backup_slots: int
    donor_staging_slots: int
    face_adjacent_reserve: int
    max_side_rack_slots: int
    rack_depth_tiers: int

    @property
    def total_module_slots(self) -> int:
        """Faces + all reserve staging positions."""
        return self.pick_positions + self.reserve_cap


def pick_positions(*, source_count: int, donor_count: int) -> int:
    return source_count + donor_count


def reserve_cap(
    *,
    source_count: int,
    donor_count: int,
    ratio: float = DEFAULT_RESERVE_CAP_RATIO,
) -> int:
    return int(pick_positions(source_count=source_count, donor_count=donor_count) * ratio)


def face_adjacent_reserve(
    *,
    source_count: int,
    donor_count: int,
    rack_only_reserve: bool = True,
) -> int:
    """Source backup + donor staging behind faces (0 when all reserve is in racks)."""
    if rack_only_reserve:
        return 0
    return source_count + donor_count


def max_side_rack_slots(
    *,
    source_count: int,
    donor_count: int,
    ratio: float = DEFAULT_RESERVE_CAP_RATIO,
    rack_only_reserve: bool = True,
) -> int:
    """Pallet slots on L/R flank racks."""
    cap = reserve_cap(source_count=source_count, donor_count=donor_count, ratio=ratio)
    if rack_only_reserve:
        return cap
    return max(
        0,
        cap - face_adjacent_reserve(source_count=source_count, donor_count=donor_count, rack_only_reserve=False),
    )


def side_rack_slots_for_peak(
    *,
    visible_skus: int,
    source_faces: int,
    source_count: int,
    donor_count: int,
    ratio: float = DEFAULT_RESERVE_CAP_RATIO,
    rack_only_reserve: bool = True,
) -> int:
    """Rack slots to model at setup — peak source overflow under the reserve cap."""
    needed = max(0, visible_skus - source_faces)
    return min(
        needed,
        max_side_rack_slots(
            source_count=source_count,
            donor_count=donor_count,
            ratio=ratio,
            rack_only_reserve=rack_only_reserve,
        ),
    )


def rack_bay_count(*, pallet_slots: int, depth_tiers: int = DEFAULT_RACK_DEPTH_TIERS) -> int:
    if pallet_slots <= 0:
        return 0
    if depth_tiers < 1:
        raise ValueError("depth_tiers must be >= 1")
    return int(math.ceil(pallet_slots / depth_tiers))


def build_reserve_budget(
    *,
    source_count: int,
    donor_count: int,
    ratio: float = DEFAULT_RESERVE_CAP_RATIO,
    rack_depth_tiers: int = DEFAULT_RACK_DEPTH_TIERS,
    rack_only_reserve: bool = True,
) -> ReserveBudget:
    picks = pick_positions(source_count=source_count, donor_count=donor_count)
    cap = reserve_cap(source_count=source_count, donor_count=donor_count, ratio=ratio)
    adjacent = face_adjacent_reserve(
        source_count=source_count, donor_count=donor_count, rack_only_reserve=rack_only_reserve
    )
    rack_slots = max_side_rack_slots(
        source_count=source_count,
        donor_count=donor_count,
        ratio=ratio,
        rack_only_reserve=rack_only_reserve,
    )
    return ReserveBudget(
        pick_positions=picks,
        reserve_cap=cap,
        reserve_cap_ratio=ratio,
        rack_only_reserve=rack_only_reserve,
        source_backup_slots=0 if rack_only_reserve else source_count,
        donor_staging_slots=0 if rack_only_reserve else donor_count,
        face_adjacent_reserve=adjacent,
        max_side_rack_slots=rack_slots,
        rack_depth_tiers=rack_depth_tiers,
    )
