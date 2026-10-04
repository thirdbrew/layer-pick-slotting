"""Layout sweep optimizer — Mode A (min footprint) and Mode B (max throughput)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from slotting.backtest import derive_order_status, orders_for_as_of
from slotting.demand_throughput import RequiredThroughputProfile, compute_required_throughput, load_orders
from slotting.models import Exclusion, OrderArea, OrderLine, OrderStatus, WorldState
from slotting.pallet_geometry import pallet_height_limits, resolve_height_limits
from slotting.rules import WAVED_HORIZON
from slotting.simulation import simulate_once
from slotting.sweep_floor import build_sweep_floor
from slotting.throughput import compute_wave_throughput, item_has_pickable_face_supply


class OptimizeMode(str, Enum):
    MIN_FOOTPRINT = "min-footprint"
    MAX_THROUGHPUT = "max-throughput"


@dataclass(frozen=True)
class LayoutConstraints:
    source_slot_sqft: float
    donor_slot_sqft: float
    source_range: tuple[int, int]
    donor_range: tuple[int, int]
    area_sqft: float | None = None
    aisle_reserve_pct: float = 0.15

    @property
    def unlimited_area(self) -> bool:
        return self.area_sqft is None


@dataclass(frozen=True)
class SweepCandidateScore:
    source_slots: int
    donor_slots: int
    ratio: str
    footprint_sqft: float
    feasible: bool
    projected_wave_layer_pick_pct: float
    projected_shortage_count: int
    layers_shifted_to_layer_pick: int
    adds_applied: int
    meets_target: bool
    notes: str = ""


@dataclass(frozen=True)
class DonorSweepScore:
    """Donor sweep at a fixed source count — includes build-face-aware wave LP."""

    source_slots: int
    donor_slots: int
    ratio: str
    footprint_sqft: float
    projected_wave_layer_pick_pct: float
    projected_build_face_wave_lp_pct: float
    dest_blocked_sku_count: int
    build_faces_opened: int
    meets_build_face_target: bool
    notes: str = ""


@dataclass(frozen=True)
class DonorSweepResult:
    fixed_source_slots: int
    target_build_face_wave_lp_pct: float
    candidates: list[DonorSweepScore]
    throughput_profile: RequiredThroughputProfile
    score_as_of: datetime
    wave_horizon: timedelta
    winner: DonorSweepScore | None = None


DONOR_SWEEP_COLUMNS = [
    "source_slots",
    "donor_slots",
    "ratio",
    "footprint_sqft",
    "projected_wave_layer_pick_pct",
    "projected_build_face_wave_lp_pct",
    "dest_blocked_sku_count",
    "build_faces_opened",
    "meets_build_face_target",
    "notes",
]


@dataclass(frozen=True)
class SweepResult:
    mode: OptimizeMode
    winner: SweepCandidateScore | None
    candidates: list[SweepCandidateScore]
    throughput_profile: RequiredThroughputProfile
    score_as_of: datetime
    wave_horizon: timedelta
    unlimited_area: bool


SWEEP_COLUMNS = [
    "source_slots",
    "donor_slots",
    "ratio",
    "footprint_sqft",
    "feasible",
    "projected_wave_layer_pick_pct",
    "projected_shortage_count",
    "layers_shifted_to_layer_pick",
    "adds_applied",
    "meets_target",
    "notes",
]


def footprint_sqft(source_slots: int, donor_slots: int, *, constraints: LayoutConstraints) -> float:
    return (
        source_slots * constraints.source_slot_sqft
        + donor_slots * constraints.donor_slot_sqft
    )


def _usable_area_sqft(constraints: LayoutConstraints) -> float | None:
    if constraints.unlimited_area or constraints.area_sqft is None:
        return None
    return constraints.area_sqft * (1 - constraints.aisle_reserve_pct)


def _area_capacity(constraints: LayoutConstraints) -> tuple[int, int]:
    """Per-dimension UPPER BOUNDS for enumeration only.

    A pair can satisfy both bounds and still not fit — feasibility must check
    the combined footprint against the usable area, not these caps.
    """
    usable = _usable_area_sqft(constraints)
    if usable is None:
        return 10_000, 10_000
    max_source = max(0, math.floor(usable / constraints.source_slot_sqft))
    max_donor = max(0, math.floor(usable / constraints.donor_slot_sqft))
    return max_source, max_donor


def enumerate_layout_pairs(
    constraints: LayoutConstraints,
    *,
    profile: RequiredThroughputProfile | None = None,
    source_buffer_slots: int = 2,
) -> list[tuple[int, int, float, bool]]:
    """
    Candidate (source, donor) pairs sorted by footprint ascending.

    When ``area_sqft`` is None (unlimited area), upper bounds come from peak
    wave demand — not from a physical zone cap.
    """
    max_source_cap, max_donor_cap = _area_capacity(constraints)
    src_lo, src_hi = constraints.source_range
    don_lo, don_hi = constraints.donor_range

    if constraints.unlimited_area and profile is not None:
        src_hi = max(src_hi, profile.peak_concurrent_skus + source_buffer_slots)
        don_hi = max(don_hi, profile.min_donor_slots_hint)
        src_lo = 1
        don_lo = constraints.donor_range[0]

    usable = _usable_area_sqft(constraints)
    pairs: list[tuple[int, int, float, bool]] = []
    for source in range(src_lo, min(src_hi, max_source_cap) + 1):
        for donor in range(don_lo, min(don_hi, max_donor_cap) + 1):
            sqft = footprint_sqft(source, donor, constraints=constraints)
            # Combined footprint decides fit — per-dimension caps alone would
            # pass layouts whose source+donor total exceeds the zone.
            fits_area = usable is None or sqft <= usable
            pairs.append((source, donor, sqft, fits_area))
    pairs.sort(key=lambda row: row[2])
    return pairs


def enumerate_mode_a_pairs(
    constraints: LayoutConstraints,
    profile: RequiredThroughputProfile,
    *,
    source_buffer_slots: int = 2,
) -> list[tuple[int, int, float, bool]]:
    """
    Demand-driven Mode A search: grow source slots with minimum donor until target met.

    Donor slots do not affect Part 2 throughput today, so hold donor at design min
    to minimize footprint while finding required source capacity.
    """
    max_source = profile.peak_concurrent_skus + source_buffer_slots
    donor = constraints.donor_range[0]
    pairs = [
        (
            source,
            donor,
            footprint_sqft(source, donor, constraints=constraints),
            True,
        )
        for source in range(1, max_source + 1)
    ]
    return pairs


def _items_in_wave_window(
    orders: list[OrderLine],
    as_of: datetime,
    *,
    wave_horizon: timedelta,
) -> list[str]:
    cutoff = as_of + wave_horizon
    items: set[str] = set()
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order.due_at < as_of or order.due_at > cutoff:
            continue
        if derive_order_status(order, as_of, wave_horizon=wave_horizon) != OrderStatus.WAVED:
            continue
        items.add(order.item_id)
    return sorted(items)


def _pick_score_as_of(profile: RequiredThroughputProfile, orders: list[OrderLine]) -> datetime:
    if profile.peak_wave_window_start is not None:
        return profile.peak_wave_window_start
    if not orders:
        return datetime.now().replace(microsecond=0)
    return min(o.due_at for o in orders)


def _load_exclusions(path: Path | None) -> list[Exclusion]:
    if path is None or not path.exists():
        return []
    import pandas as pd

    from slotting.jda_columns import EXCLUSION_ALIASES, resolve_columns

    df = pd.read_csv(path, dtype=str)
    cols = resolve_columns(df, EXCLUSION_ALIASES)
    return [
        Exclusion(
            item_id=str(cols["item_id"].iloc[i]).strip(),
            reason=str(cols["reason"].iloc[i]).strip(),
        )
        for i in range(len(df))
    ]


def score_layout(
    *,
    source_slots: int,
    donor_slots: int,
    orders: list[OrderLine],
    exclusions: list[Exclusion],
    as_of: datetime,
    constraints: LayoutConstraints,
    target_wave_lp_pct: float,
    max_shortage_count: int,
    source_buffer_slots: int,
    wave_horizon: timedelta,
) -> SweepCandidateScore:
    feasible = True
    notes: list[str] = []
    usable = _usable_area_sqft(constraints)
    if usable is not None and footprint_sqft(
        source_slots, donor_slots, constraints=constraints
    ) > usable:
        feasible = False
        notes.append("Exceeds area capacity")

    waved_items = _items_in_wave_window(orders, as_of, wave_horizon=wave_horizon)
    pallets, slots = build_sweep_floor(
        source_count=source_slots,
        donor_count=donor_slots,
        waved_item_ids=waved_items,
        as_of=as_of,
    )
    step_orders = orders_for_as_of(orders, as_of, wave_horizon=wave_horizon)
    state = WorldState(
        pallets=pallets,
        slots=slots,
        orders=step_orders,
        exclusions=exclusions,
        as_of=as_of,
    )
    result = simulate_once(state)
    sqft = footprint_sqft(source_slots, donor_slots, constraints=constraints)
    wave_pct = result.wave_layer_pick_pct * 100
    shortage = result.wave_shortage_count

    meets_target = (
        feasible
        and wave_pct >= target_wave_lp_pct
        and shortage <= max_shortage_count
        and source_slots + source_buffer_slots >= len(waved_items)
    )
    if not meets_target and feasible and source_slots + source_buffer_slots < len(waved_items):
        notes.append("Insufficient source slots for waved SKU count")

    return SweepCandidateScore(
        source_slots=source_slots,
        donor_slots=donor_slots,
        ratio=f"{source_slots}:{donor_slots}",
        footprint_sqft=sqft,
        feasible=feasible,
        projected_wave_layer_pick_pct=round(wave_pct, 2),
        projected_shortage_count=shortage,
        layers_shifted_to_layer_pick=result.wave_layers_shifted_to_layer_pick,
        adds_applied=result.adds_applied,
        meets_target=meets_target,
        notes="; ".join(notes),
    )


def _dest_blocked_sku_count(state: WorldState, waved_item_ids: list[str]) -> int:
    """Waved SKUs with staged source supply but no destination build face."""
    return sum(
        1
        for item_id in waved_item_ids
        if item_has_pickable_face_supply(state, item_id) and not state.item_on_destination_slot(item_id)
    )


def score_layout_donor_aware(
    *,
    source_slots: int,
    donor_slots: int,
    orders: list[OrderLine],
    exclusions: list[Exclusion],
    as_of: datetime,
    constraints: LayoutConstraints,
    target_build_face_wave_lp_pct: float,
    wave_horizon: timedelta,
) -> DonorSweepScore:
    """Score a layout with both source-only and build-face-aware wave throughput."""
    waved_items = _items_in_wave_window(orders, as_of, wave_horizon=wave_horizon)
    pallets, slots = build_sweep_floor(
        source_count=source_slots,
        donor_count=donor_slots,
        waved_item_ids=waved_items,
        as_of=as_of,
    )
    step_orders = orders_for_as_of(orders, as_of, wave_horizon=wave_horizon)
    state = WorldState(
        pallets=pallets,
        slots=slots,
        orders=step_orders,
        exclusions=exclusions,
        as_of=as_of,
    )
    result = simulate_once(state)
    sqft = footprint_sqft(source_slots, donor_slots, constraints=constraints)
    source_wave_pct = result.wave_layer_pick_pct * 100
    build_face_wave = compute_wave_throughput(
        result.projected_state,
        horizon=wave_horizon,
        require_build_face=True,
    )
    build_face_pct = build_face_wave.layer_pick_pct * 100
    dest_blocked = _dest_blocked_sku_count(result.projected_state, waved_items)
    build_faces = len(result.projected_state.pallets_in_destination_slots())
    meets = build_face_pct >= target_build_face_wave_lp_pct

    notes: list[str] = []
    if dest_blocked > 0:
        notes.append(f"{dest_blocked} waved SKUs blocked by donor capacity")

    return DonorSweepScore(
        source_slots=source_slots,
        donor_slots=donor_slots,
        ratio=f"{source_slots}:{donor_slots}",
        footprint_sqft=sqft,
        projected_wave_layer_pick_pct=round(source_wave_pct, 2),
        projected_build_face_wave_lp_pct=round(build_face_pct, 2),
        dest_blocked_sku_count=dest_blocked,
        build_faces_opened=build_faces,
        meets_build_face_target=meets,
        notes="; ".join(notes),
    )


def run_donor_sweep(
    *,
    orders_path: Path,
    fixed_source_slots: int,
    constraints: LayoutConstraints,
    exclusions_path: Path | None = None,
    target_build_face_wave_lp_pct: float = 85.0,
    wave_horizon: timedelta | None = None,
    donor_range: tuple[int, int] | None = None,
    skus_path: Path | None = None,
) -> DonorSweepResult:
    """
    Sweep donor slot counts at a fixed source count.

    Uses build-face-aware wave LP (source staged **and** destination build open).
    """
    horizon = wave_horizon or WAVED_HORIZON
    all_orders = load_orders(Path(orders_path))
    exclusions = _load_exclusions(exclusions_path)
    profile = compute_required_throughput(
        orders_path,
        wave_horizon=horizon,
    )
    as_of = _pick_score_as_of(profile, all_orders)

    don_lo, don_hi = donor_range or constraints.donor_range
    don_hi = max(don_hi, profile.min_donor_slots_hint)

    candidates: list[DonorSweepScore] = []
    winner: DonorSweepScore | None = None

    with pallet_height_limits(resolve_height_limits(skus_path)):
        for donor in range(don_lo, don_hi + 1):
            score = score_layout_donor_aware(
                source_slots=fixed_source_slots,
                donor_slots=donor,
                orders=all_orders,
                exclusions=exclusions,
                as_of=as_of,
                constraints=constraints,
                target_build_face_wave_lp_pct=target_build_face_wave_lp_pct,
                wave_horizon=horizon,
            )
            candidates.append(score)
            if winner is None and score.meets_build_face_target:
                winner = score

    if winner is None and candidates:
        winner = max(candidates, key=lambda c: c.projected_build_face_wave_lp_pct)

    return DonorSweepResult(
        fixed_source_slots=fixed_source_slots,
        target_build_face_wave_lp_pct=target_build_face_wave_lp_pct,
        candidates=candidates,
        throughput_profile=profile,
        score_as_of=as_of,
        wave_horizon=horizon,
        winner=winner,
    )


def write_donor_sweep_csv(result: DonorSweepResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DONOR_SWEEP_COLUMNS)
        writer.writeheader()
        for candidate in result.candidates:
            writer.writerow(
                {
                    "source_slots": candidate.source_slots,
                    "donor_slots": candidate.donor_slots,
                    "ratio": candidate.ratio,
                    "footprint_sqft": round(candidate.footprint_sqft, 2),
                    "projected_wave_layer_pick_pct": candidate.projected_wave_layer_pick_pct,
                    "projected_build_face_wave_lp_pct": candidate.projected_build_face_wave_lp_pct,
                    "dest_blocked_sku_count": candidate.dest_blocked_sku_count,
                    "build_faces_opened": candidate.build_faces_opened,
                    "meets_build_face_target": candidate.meets_build_face_target,
                    "notes": candidate.notes,
                }
            )


def run_sweep(
    *,
    orders_path: Path,
    constraints: LayoutConstraints,
    mode: OptimizeMode,
    exclusions_path: Path | None = None,
    target_wave_lp_pct: float = 85.0,
    max_shortage_count: int = 0,
    source_buffer_slots: int = 2,
    wave_horizon: timedelta | None = None,
    skus_path: Path | None = None,
) -> SweepResult:
    """
    Score layouts and pick a winner.

    Mode A: smallest footprint meeting target wave LP % and shortage constraints.
    When ``constraints.area_sqft`` is None, sweep is demand-driven (no zone cap).

    Mode B: highest projected wave LP %; tie-break fewer total slots then lower sqft.
    """
    horizon = wave_horizon or WAVED_HORIZON
    all_orders = load_orders(Path(orders_path))
    exclusions = _load_exclusions(exclusions_path)
    profile = compute_required_throughput(
        orders_path,
        min_layer_pick_ready_pct=target_wave_lp_pct / 100,
        source_buffer_slots=source_buffer_slots,
        wave_horizon=horizon,
    )
    as_of = _pick_score_as_of(profile, all_orders)

    if mode == OptimizeMode.MIN_FOOTPRINT and constraints.unlimited_area:
        pair_list = enumerate_mode_a_pairs(
            constraints,
            profile,
            source_buffer_slots=source_buffer_slots,
        )
    else:
        pair_list = enumerate_layout_pairs(
            constraints,
            profile=profile,
            source_buffer_slots=source_buffer_slots,
        )

    candidates: list[SweepCandidateScore] = []
    winner: SweepCandidateScore | None = None

    with pallet_height_limits(resolve_height_limits(skus_path)):
        for source, donor, _sqft, fits in pair_list:
            if not fits:
                candidates.append(
                    SweepCandidateScore(
                        source_slots=source,
                        donor_slots=donor,
                        ratio=f"{source}:{donor}",
                        footprint_sqft=footprint_sqft(source, donor, constraints=constraints),
                        feasible=False,
                        projected_wave_layer_pick_pct=0.0,
                        projected_shortage_count=0,
                        layers_shifted_to_layer_pick=0,
                        adds_applied=0,
                        meets_target=False,
                        notes="Does not fit area",
                    )
                )
                continue

            score = score_layout(
                source_slots=source,
                donor_slots=donor,
                orders=all_orders,
                exclusions=exclusions,
                as_of=as_of,
                constraints=constraints,
                target_wave_lp_pct=target_wave_lp_pct,
                max_shortage_count=max_shortage_count,
                source_buffer_slots=source_buffer_slots,
                wave_horizon=horizon,
            )
            candidates.append(score)

            if mode == OptimizeMode.MIN_FOOTPRINT and winner is None and score.meets_target:
                winner = score
                if constraints.unlimited_area:
                    break

    if mode == OptimizeMode.MIN_FOOTPRINT and winner is None:
        for candidate in sorted(candidates, key=lambda c: c.footprint_sqft):
            if candidate.meets_target:
                winner = candidate
                break
    elif mode == OptimizeMode.MAX_THROUGHPUT:
        feasible_scored = [c for c in candidates if c.feasible]
        if feasible_scored:
            winner = max(
                feasible_scored,
                key=lambda c: (
                    c.projected_wave_layer_pick_pct,
                    -(c.source_slots + c.donor_slots),
                    -c.footprint_sqft,
                ),
            )

    return SweepResult(
        mode=mode,
        winner=winner,
        candidates=candidates,
        throughput_profile=profile,
        score_as_of=as_of,
        wave_horizon=horizon,
        unlimited_area=constraints.unlimited_area,
    )


def write_sweep_results_csv(result: SweepResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        for candidate in result.candidates:
            writer.writerow(
                {
                    "source_slots": candidate.source_slots,
                    "donor_slots": candidate.donor_slots,
                    "ratio": candidate.ratio,
                    "footprint_sqft": round(candidate.footprint_sqft, 2),
                    "feasible": candidate.feasible,
                    "projected_wave_layer_pick_pct": candidate.projected_wave_layer_pick_pct,
                    "projected_shortage_count": candidate.projected_shortage_count,
                    "layers_shifted_to_layer_pick": candidate.layers_shifted_to_layer_pick,
                    "adds_applied": candidate.adds_applied,
                    "meets_target": candidate.meets_target,
                    "notes": candidate.notes,
                }
            )


def constraints_from_site_profile(
    profile,
    *,
    unlimited_area: bool = False,
) -> LayoutConstraints:
    """Build sweep constraints from a layout-advisor ``SiteProfile``."""
    return LayoutConstraints(
        area_sqft=None if unlimited_area else float(profile.area_sqft.value),
        source_slot_sqft=float(profile.source_slot_sqft.value),
        donor_slot_sqft=float(profile.donor_slot_sqft.value),
        source_range=tuple(profile.source_slot_count_range.value),
        donor_range=tuple(profile.donor_slot_count_range.value),
    )
