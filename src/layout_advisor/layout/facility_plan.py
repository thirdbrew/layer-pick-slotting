"""Pick aisle + 2-deep side racks — footprint-minimized, 1:1 reserve cap."""

from __future__ import annotations

import math
from dataclasses import dataclass

from layout_advisor.layout.aisle_plan import AisleLayoutPlan, build_aisle_layout
from layout_advisor.layout.reserve_policy import (
    DEFAULT_RACK_DEPTH_TIERS,
    DEFAULT_RESERVE_CAP_RATIO,
    build_reserve_budget,
    rack_bay_count,
    side_rack_slots_for_peak,
)


@dataclass(frozen=True)
class FacilityLayoutPlan:
    """Top-down concept: cascade pick aisle with 2-deep resupply racks on both flanks."""

    pick: AisleLayoutPlan
    reserve: ReserveBudgetView
    rack_count: int
    rack_sides: int
    rack_depth_tiers: int
    rack_bay_count: int
    rack_per_side: int
    rack_bays_per_side: int
    rack_cols: int
    rack_slot_sqft: float
    rack_aisle_width_ft: float
    rack_depth_per_side_ft: float
    rack_length_ft: float
    rack_footprint_sqft: float
    total_footprint_sqft: float
    title: str = "LP facility concept"
    notes: tuple[str, ...] = ()

    @property
    def ratio(self) -> str:
        return self.pick.ratio

    @property
    def source_count(self) -> int:
        return self.pick.source_count

    @property
    def donor_count(self) -> int:
        return self.pick.donor_count


@dataclass(frozen=True)
class ReserveBudgetView:
    pick_positions: int
    reserve_cap: int
    reserve_cap_ratio: float
    face_adjacent_reserve: int
    max_side_rack_slots: int
    rack_slots_used: int
    rack_slots_headroom: int


def build_facility_layout(
    *,
    source_count: int,
    donor_count: int,
    rack_count: int | None = None,
    visible_skus: int | None = None,
    slot_sqft: float = 60.0,
    aisle_width_ft: float = 8.0,
    backup_depth_ft: float = 4.0,
    rack_aisle_width_ft: float = 8.0,
    aisle_reserve_pct: float = 0.12,
    rack_sides: int = 2,
    reserve_cap_ratio: float = DEFAULT_RESERVE_CAP_RATIO,
    rack_depth_tiers: int = DEFAULT_RACK_DEPTH_TIERS,
    rack_only_reserve: bool = True,
    title: str | None = None,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
) -> FacilityLayoutPlan:
    """
    Build pick aisle plus 2-deep flank racks under the reserve cap (default 1:1).

    *rack_count* defaults from *visible_skus* (peak overflow capped by reserve budget).
    """
    effective_backup_depth = 0.0 if rack_only_reserve else backup_depth_ft
    pick = build_aisle_layout(
        source_count=source_count,
        donor_count=donor_count,
        slot_sqft=slot_sqft,
        aisle_width_ft=aisle_width_ft,
        backup_depth_ft=effective_backup_depth,
        aisle_reserve_pct=0.0,
        rack_only_reserve=rack_only_reserve,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
        title=f"Pick aisle {source_count}:{donor_count}",
    )
    budget = build_reserve_budget(
        source_count=source_count,
        donor_count=donor_count,
        ratio=reserve_cap_ratio,
        rack_depth_tiers=rack_depth_tiers,
        rack_only_reserve=rack_only_reserve,
    )
    if rack_count is None:
        vis = visible_skus if visible_skus is not None else source_count
        rack_count = side_rack_slots_for_peak(
            visible_skus=vis,
            source_faces=source_count,
            source_count=source_count,
            donor_count=donor_count,
            ratio=reserve_cap_ratio,
            rack_only_reserve=rack_only_reserve,
        )
    if rack_count < 0:
        raise ValueError("rack_count must be >= 0")
    if rack_count > budget.max_side_rack_slots:
        raise ValueError(
            f"rack_count {rack_count} exceeds side-rack budget "
            f"({budget.max_side_rack_slots} slots for {budget.reserve_cap_ratio}:1 cap)"
        )
    if rack_sides < 1:
        raise ValueError("rack_sides must be >= 1")

    slot_width_ft = pick.slot_width_ft
    slot_depth_ft = pick.slot_depth_ft
    bays_total = rack_bay_count(pallet_slots=rack_count, depth_tiers=rack_depth_tiers)
    rack_per_side = int(math.ceil(rack_count / rack_sides)) if rack_count else 0
    rack_bays_per_side = int(math.ceil(rack_per_side / rack_depth_tiers)) if rack_per_side else 0
    rack_cols = max(1, rack_bays_per_side) if rack_count else 0
    rack_length_ft = rack_cols * slot_width_ft
    rack_depth_per_side_ft = rack_depth_tiers * slot_depth_ft if rack_count else 0.0

    pick_active = pick.total_aisle_slots * slot_sqft
    if rack_only_reserve:
        pick_backup = 0.0
        donor_staging = 0.0
    else:
        pick_backup = source_count * slot_width_ft * backup_depth_ft
        donor_staging = donor_count * slot_width_ft * backup_depth_ft
    aisle_lanes = 2 if picker_count > 1 or center_gap_ft > 0 else 1
    pick_aisle_sqft = pick.aisle_length_ft * aisle_width_ft * aisle_lanes
    gap_sqft = pick.aisle_length_ft * max(0.0, center_gap_ft) if aisle_lanes > 1 else 0.0
    pick_footprint = (pick_active + pick_backup + donor_staging + pick_aisle_sqft + gap_sqft) * (
        1 + aisle_reserve_pct
    )

    rack_active_sqft = rack_count * slot_sqft
    rack_access_sqft = rack_length_ft * rack_aisle_width_ft * rack_sides
    rack_footprint = (rack_active_sqft + rack_access_sqft) * (1 + aisle_reserve_pct)

    ratio = f"{source_count}:{donor_count}"
    default_title = (
        f"LP layout: {ratio} pick aisle + {rack_count} rack slots "
        f"({rack_depth_tiers}-deep, {reserve_cap_ratio:.0f}:1 reserve cap)"
    )
    reserve_view = ReserveBudgetView(
        pick_positions=budget.pick_positions,
        reserve_cap=budget.reserve_cap,
        reserve_cap_ratio=budget.reserve_cap_ratio,
        face_adjacent_reserve=budget.face_adjacent_reserve,
        max_side_rack_slots=budget.max_side_rack_slots,
        rack_slots_used=rack_count,
        rack_slots_headroom=budget.max_side_rack_slots - rack_count,
    )
    if rack_only_reserve:
        reserve_detail = f"{budget.max_side_rack_slots} rack slots (all reserve in 2-deep side racks)"
    else:
        reserve_detail = (
            f"{budget.face_adjacent_reserve} face-adjacent + {budget.max_side_rack_slots} rack max"
        )
    notes = (
        f"Pick aisle: {source_count} source + {donor_count} donor = {pick.total_aisle_slots} LP pads.",
        f"Reserve cap {budget.reserve_cap_ratio:.0f}:1 → {budget.reserve_cap} staging slots ({reserve_detail}).",
        f"Side racks: {rack_count} pallet slots in {bays_total} bays "
        f"({rack_depth_tiers}-deep per flank, {rack_bays_per_side} bays/side × {rack_cols} along aisle).",
        "Setup @ 22:00: faces + capped rack pool; mid-day face↔rack swaps only.",
        f"Pick ~{pick_footprint:,.0f} sqft | Racks ~{rack_footprint:,.0f} sqft | "
        f"Total ~{pick_footprint + rack_footprint:,.0f} sqft.",
    )
    return FacilityLayoutPlan(
        pick=pick,
        reserve=reserve_view,
        rack_count=rack_count,
        rack_sides=rack_sides,
        rack_depth_tiers=rack_depth_tiers,
        rack_bay_count=bays_total,
        rack_per_side=rack_per_side,
        rack_bays_per_side=rack_bays_per_side,
        rack_cols=rack_cols,
        rack_slot_sqft=slot_sqft,
        rack_aisle_width_ft=rack_aisle_width_ft,
        rack_depth_per_side_ft=rack_depth_per_side_ft,
        rack_length_ft=rack_length_ft,
        rack_footprint_sqft=round(rack_footprint, 1),
        total_footprint_sqft=round(pick_footprint + rack_footprint, 1),
        title=title or default_title,
        notes=notes,
    )


def peak_rack_count(*, visible_skus: int, source_faces: int, donor_count: int = 25) -> int:
    """Side-rack pallet slots for peak visible SKUs under the reserve cap."""
    return side_rack_slots_for_peak(
        visible_skus=visible_skus,
        source_faces=source_faces,
        source_count=source_faces,
        donor_count=donor_count,
        ratio=DEFAULT_RESERVE_CAP_RATIO,
    )
