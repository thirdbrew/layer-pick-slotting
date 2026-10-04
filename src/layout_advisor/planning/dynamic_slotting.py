"""Final-shift dynamic slotting — rank visible SKUs into faces, racks, or freezer hold."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from layout_advisor.layout.slot_geometry import (
    build_aisle_slot_map,
    donor_slot_colocated,
    nearest_donor_on_flank,
    staging_anchor_for_assignment,
    stations_per_side,
)
from layout_advisor.planning.wave_slotting import (
    build_face_by_item,
    preferred_source_faces,
    primary_wave_release_by_item,
    setup_rank_by_item,
)
from slotting.generate_slots import source_slot_ids
from slotting.models import OrderLine
from slotting.rack_depth import RankedSku, assign_rack_slots_colocated, rack_depth_tier
from slotting.resupply_rack import overflow_rack_slot_ids
from slotting.visibility_planning import (
    affinity_orders_at_setup,
    final_shift_as_of,
    rank_for_final_shift_setup,
)


@dataclass(frozen=True)
class SetupAssignment:
    """One SKU placement decision at final-shift setup."""

    rank: int
    item_id: str
    layers: int
    earliest_due: datetime
    location: str  # source_face | rack | freezer_hold
    slot_id: str | None
    role: str  # setup_crew
    notes: str


def _to_ranked(ranked: list[tuple[str, int, datetime]]) -> list[RankedSku]:
    return [
        RankedSku(rank=index, item_id=item_id, layers=layers, earliest_due=due)
        for index, (item_id, layers, due) in enumerate(ranked, start=1)
    ]


def _rack_slots_for_overflow(
    *,
    source_count: int,
    donor_count: int,
) -> list[str]:
    """Rack pallet positions for source overflow (excludes donor staging tail reserve)."""
    return overflow_rack_slot_ids(source_count=source_count, donor_count=donor_count)


@dataclass(frozen=True)
class SetupSlottingPlan:
    """Maps produced by final-shift slotting (shared by planning and simulation)."""

    setup_as_of: datetime
    ranked_skus: tuple[RankedSku, ...]
    face_by_item: dict[str, str]
    rack_by_item: dict[str, str]
    face_order: list[str]
    preferred_faces: dict[str, str]
    setup_ranks: dict[str, int]
    # Bartholdi-Hackman velocity allocation: second concurrent face for the
    # top-k layer-velocity SKUs (empty unless velocity_twin_faces_top_k > 0).
    twin_face_by_item: dict[str, str] = field(default_factory=dict)


def build_setup_slotting_plan(
    orders: list[OrderLine],
    ship_date: date,
    *,
    source_count: int,
    donor_count: int,
    final_shift_hour: int = 22,
    early_wave_hours: int = 8,
    due_offset_hours: float = 4.0,
    pick_start_hour: int = 6,
    wave_clustering: bool = True,
    order_affinity: bool = True,
    rank_anchored_chain: bool = False,
    build_anchored_slotting: bool = False,
    order_anchored_window_assign: bool = False,
    prev_face_by_item: dict[str, str] | None = None,
    velocity_twin_faces_top_k: int = 0,
    build_anchored_light_day_sku_threshold: int = 0,
    build_anchored_light_day_stickiness: float = 4.0,
    build_anchored_prelight_stickiness: bool = False,
    setup_ship_day_foresight: bool = True,
    setup_arrival_censor: bool = False,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
    skus_path: Path | None = None,
) -> SetupSlottingPlan:
    """Rank ship-day SKUs and compute face / rack / preferred-face maps."""
    setup_as_of = final_shift_as_of(ship_date, final_shift_start_hour=final_shift_hour)
    ranked = rank_for_final_shift_setup(
        orders,
        ship_date,
        setup_as_of=setup_as_of,
        setup_ship_day_foresight=setup_ship_day_foresight,
        setup_arrival_censor=setup_arrival_censor,
        pick_start_hour=pick_start_hour,
        early_wave_hours=early_wave_hours,
        due_offset_hours=due_offset_hours,
    )
    ranked_skus = _to_ranked(ranked)
    affinity_orders = affinity_orders_at_setup(
        orders,
        ship_date,
        setup_as_of=setup_as_of,
        setup_ship_day_foresight=setup_ship_day_foresight,
        setup_arrival_censor=setup_arrival_censor,
    )
    layout, aisle_map = build_aisle_slot_map(
        source_count=source_count,
        donor_count=donor_count,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )
    twin_k = max(0, min(velocity_twin_faces_top_k, source_count - 1))
    face_by_item, face_order = build_face_by_item(
        ranked_skus,
        orders,
        ship_date,
        source_count=source_count,
        donor_count=donor_count,
        aisle_map=aisle_map,
        layout=layout,
        due_offset_hours=due_offset_hours,
        wave_clustering=wave_clustering,
        order_affinity=order_affinity,
        rank_anchored_chain=rank_anchored_chain,
        build_anchored_slotting=build_anchored_slotting,
        order_anchored_window_assign=order_anchored_window_assign,
        prev_face_by_item=prev_face_by_item,
        build_anchored_light_day_sku_threshold=build_anchored_light_day_sku_threshold,
        build_anchored_light_day_stickiness=build_anchored_light_day_stickiness,
        build_anchored_prelight_stickiness=build_anchored_prelight_stickiness,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
        stage_item_budget=(source_count - twin_k) if twin_k else None,
        affinity_orders=affinity_orders,
        skus_path=skus_path,
    )
    twin_face_by_item: dict[str, str] = {}
    if twin_k:
        # B-H velocity allocation: the k hottest staged SKUs (by ship-day
        # layers) get a second concurrent face on the nearest free station,
        # displacing the k lowest-ranked SKUs to rack. One pallet holds 8
        # layers; SKUs above that force mid-day restocks under 1 face each.
        free = sorted(
            (f for f in source_slot_ids(source_count)
             if f in aisle_map and f not in set(face_by_item.values())),
            key=lambda f: aisle_map[f].station,
        )
        staged_by_layers = sorted(
            (s for s in ranked_skus if s.item_id in face_by_item),
            key=lambda s: -s.layers,
        )
        for sku in staged_by_layers[:twin_k]:
            if not free:
                break
            primary_station = aisle_map[face_by_item[sku.item_id]].station
            twin = min(free, key=lambda f: abs(aisle_map[f].station - primary_station))
            free.remove(twin)
            twin_face_by_item[sku.item_id] = twin
    overflow_rack_ids = _rack_slots_for_overflow(
        source_count=source_count,
        donor_count=donor_count,
    )
    primary_wave = (
        primary_wave_release_by_item(
            orders,
            ship_date,
            due_offset_hours=due_offset_hours,
        )
        if wave_clustering
        else None
    )
    preferred = preferred_source_faces(
        ranked_skus,
        face_by_item,
        face_order,
        source_count=source_count,
        primary_wave=primary_wave,
    )
    rack_virtual_faces = dict(preferred)
    rack_virtual_faces.update(face_by_item)
    # With twin faces the staged-item budget is below source_count, so
    # overflow = everything without a staged face; otherwise keep the
    # original index slice so baseline behavior is byte-identical.
    overflow_skus = (
        [s for s in ranked_skus if s.item_id not in face_by_item]
        if twin_k
        else list(ranked_skus[source_count:])
    )
    rack_by_item = assign_rack_slots_colocated(
        overflow_skus,
        overflow_rack_ids,
        face_order=face_order,
        aisle_map=aisle_map,
        virtual_face_by_item=rack_virtual_faces,
    )
    return SetupSlottingPlan(
        setup_as_of=setup_as_of,
        ranked_skus=tuple(ranked_skus),
        face_by_item=face_by_item,
        rack_by_item=rack_by_item,
        face_order=face_order,
        preferred_faces=preferred,
        setup_ranks=setup_rank_by_item(ranked_skus),
        twin_face_by_item=twin_face_by_item,
    )


def build_setup_slotting(
    orders: list[OrderLine],
    ship_date: date,
    *,
    source_count: int,
    donor_count: int,
    visible_skus: int | None = None,
    reserve_cap_ratio: float = 1.0,
    final_shift_hour: int = 22,
    early_wave_hours: int = 8,
    due_offset_hours: float = 4.0,
    pick_start_hour: int = 6,
    preopen_wave_count: int = 3,
    wave_clustering: bool = True,
    order_affinity: bool = True,
    rank_anchored_chain: bool = False,
    build_anchored_slotting: bool = False,
    order_anchored_window_assign: bool = False,
    prev_face_by_item: dict[str, str] | None = None,
    velocity_twin_faces_top_k: int = 0,
    build_anchored_light_day_sku_threshold: int = 0,
    build_anchored_light_day_stickiness: float = 4.0,
    build_anchored_prelight_stickiness: bool = False,
    setup_ship_day_foresight: bool = True,
    setup_arrival_censor: bool = False,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
    plan: SetupSlottingPlan | None = None,
    skus_path: Path | None = None,
) -> tuple[datetime, list[SetupAssignment]]:
    """
    Rank ship-day SKUs and assign pick-module homes under the reserve cap.

    Wave-aware ranking with travel + wave-affinity face clustering; rack overflow
    colocated with virtual source face; preferred faces for mid-day swaps.
    Pass a prebuilt *plan* (same orders/policy) to skip re-deriving it — the
    replay previously ran the full slotting twice per sim day.
    """
    if plan is None:
        plan = build_setup_slotting_plan(
            orders,
            ship_date,
            source_count=source_count,
            donor_count=donor_count,
            final_shift_hour=final_shift_hour,
            early_wave_hours=early_wave_hours,
            due_offset_hours=due_offset_hours,
            pick_start_hour=pick_start_hour,
            wave_clustering=wave_clustering,
            order_affinity=order_affinity,
            rank_anchored_chain=rank_anchored_chain,
            build_anchored_slotting=build_anchored_slotting,
            order_anchored_window_assign=order_anchored_window_assign,
            prev_face_by_item=prev_face_by_item,
            velocity_twin_faces_top_k=velocity_twin_faces_top_k,
            build_anchored_light_day_sku_threshold=build_anchored_light_day_sku_threshold,
            build_anchored_light_day_stickiness=build_anchored_light_day_stickiness,
            build_anchored_prelight_stickiness=build_anchored_prelight_stickiness,
            setup_ship_day_foresight=setup_ship_day_foresight,
            setup_arrival_censor=setup_arrival_censor,
            picker_count=picker_count,
            center_gap_ft=center_gap_ft,
            skus_path=skus_path,
        )
    _, aisle_map = build_aisle_slot_map(
        source_count=source_count,
        donor_count=donor_count,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )
    ranked_skus = list(plan.ranked_skus)
    face_by_item = plan.face_by_item
    rack_by_item = plan.rack_by_item
    preferred_faces = plan.preferred_faces

    assignments: list[SetupAssignment] = []
    for sku in ranked_skus:
        pref = preferred_faces.get(sku.item_id)
        if sku.item_id in face_by_item:
            face_id = face_by_item[sku.item_id]
            flank = aisle_map[face_id].side.value.upper()
            assignments.append(
                SetupAssignment(
                    rank=sku.rank,
                    item_id=sku.item_id,
                    layers=sku.layers,
                    earliest_due=sku.earliest_due,
                    location="source_face",
                    slot_id=face_id,
                    role="setup_crew",
                    notes=f"Stage to {flank} source face {face_id} (rank {sku.rank}).",
                )
            )
            twin_id = plan.twin_face_by_item.get(sku.item_id)
            if twin_id:
                twin_flank = aisle_map[twin_id].side.value.upper()
                assignments.append(
                    SetupAssignment(
                        rank=sku.rank,
                        item_id=sku.item_id,
                        layers=sku.layers,
                        earliest_due=sku.earliest_due,
                        location="source_face",
                        slot_id=twin_id,
                        role="setup_crew",
                        notes=(
                            f"Stage TWIN to {twin_flank} source face {twin_id} "
                            f"(velocity top-{len(plan.twin_face_by_item)}; primary {face_id})."
                        ),
                    )
                )
        elif sku.item_id in rack_by_item:
            rack_id = rack_by_item[sku.item_id]
            flank = "LEFT" if "-L" in rack_id else "RIGHT"
            depth = "front" if rack_depth_tier(rack_id) == 0 else "back"
            assignments.append(
                SetupAssignment(
                    rank=sku.rank,
                    item_id=sku.item_id,
                    layers=sku.layers,
                    earliest_due=sku.earliest_due,
                    location="rack",
                    slot_id=rack_id,
                    role="setup_crew",
                    notes=(
                        f"Stage to {flank} side rack {rack_id} ({depth}, rank {sku.rank}; "
                        f"virtual face {pref}; colocated with virtual source face)."
                    ),
                )
            )
        else:
            assignments.append(
                SetupAssignment(
                    rank=sku.rank,
                    item_id=sku.item_id,
                    layers=sku.layers,
                    earliest_due=sku.earliest_due,
                    location="freezer_hold",
                    slot_id=None,
                    role="supervisor",
                    notes=(
                        f"Exceeds reserve cap: keep in freezer until promoted mid-day "
                        f"to preferred face {pref} when a face opens."
                    ),
                )
            )
    return plan.setup_as_of, assignments


def face_swap_candidate_for_slot(
    freed_face: str,
    setup_assignments: list[SetupAssignment],
    preferred_faces: dict[str, str],
) -> SetupAssignment | None:
    """Best freezer-hold SKU whose preferred face matches *freed_face*."""
    candidates = [
        row
        for row in setup_assignments
        if row.location == "freezer_hold" and preferred_faces.get(row.item_id) == freed_face
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.rank, row.item_id))


@dataclass(frozen=True)
class WaveDonorOpen:
    """Colocated donor build to open for one waved SKU or one build-pallet slice."""

    wave_index: int
    wave_send: datetime
    wave_due: datetime
    item_id: str
    layers: int
    donor_slot: str
    source_slot: str | None
    setup_rank: int = 0
    open_at: datetime | None = None  # sim/plan apply time; defaults to wave_send
    ahead_prep: bool = False  # pulled into a lull window before wave send
    # Multi-pallet wave opens: one pad per outbound build slice (order path mirror).
    build_key: str = ""
    build_lines: tuple[tuple[str, int], ...] = ()

    @property
    def scheduled_open(self) -> datetime:
        return self.open_at or self.wave_send

    @property
    def open_key(self) -> str:
        """Dedupe identity — one SKU may legitimately open in several waves."""
        if self.build_key:
            return f"{self.build_key}@{self.wave_index}"
        return f"{self.item_id}@{self.wave_index}"

    @property
    def line_map(self) -> dict[str, int]:
        if self.build_lines:
            return {sku: layers for sku, layers in self.build_lines}
        return {self.item_id: self.layers}

    @property
    def sku_ids(self) -> set[str]:
        return set(self.line_map)


def _assign_colocated_donors(
    staged_rows: list[SetupAssignment],
    wave_items: set[str],
    *,
    aisle_map: dict,
    stations: int,
    donor_slot_ids: list[str],
    used_donors: set[str],
    used_items: set[str],
    gap_excluded: set[int] | None = None,
) -> list[tuple[SetupAssignment, str]]:
    """Pair staged SKUs with donor pads on the **same flank** (no gap crossing)."""
    gap_excluded = gap_excluded or set()
    donor_set = {
        d
        for d in donor_slot_ids
        if (mapped := aisle_map.get(d)) is not None and mapped.station not in gap_excluded
    }
    flank_fallback: dict[str, list[str]] = {"left": [], "right": []}
    for slot_id in donor_slot_ids:
        mapped = aisle_map.get(slot_id)
        if mapped is None or slot_id in used_donors or slot_id not in donor_set:
            continue
        flank_fallback.setdefault(mapped.side.value, []).append(slot_id)
    for flank in flank_fallback:
        flank_fallback[flank].sort(key=lambda slot_id: aisle_map[slot_id].station)
    flank_cursor: dict[str, int] = {"left": 0, "right": 0}
    out: list[tuple[SetupAssignment, str]] = []

    def _next_flank_fallback(flank: str) -> str | None:
        pool = flank_fallback.get(flank, [])
        idx = flank_cursor.get(flank, 0)
        while idx < len(pool):
            slot = pool[idx]
            idx += 1
            if slot not in used_donors:
                flank_cursor[flank] = idx
                return slot
        flank_cursor[flank] = idx
        return None

    candidates = [
        a for a in staged_rows if a.item_id in wave_items and a.item_id not in used_items
    ]
    candidates.sort(key=lambda a: (a.rank, a.item_id))
    for row in candidates:
        station, flank = staging_anchor_for_assignment(
            row.slot_id or "",
            location=row.location,
            aisle_map=aisle_map,
            stations=stations,
        )
        donor_slot: str | None = None
        if station is not None and flank is not None and station not in gap_excluded:
            donor_slot = donor_slot_colocated(
                aisle_map, station, flank, donor_set - used_donors
            )
            if donor_slot is None:
                donor_slot = nearest_donor_on_flank(
                    aisle_map,
                    station,
                    flank,
                    donor_set - used_donors,
                    gap_excluded=gap_excluded,
                )
        if donor_slot is None and flank is not None:
            donor_slot = _next_flank_fallback(flank)
        if donor_slot is None:
            # This candidate's flank is out of donors — skip it; candidates on
            # the other flank may still have free pads (break starved them).
            continue
        out.append((row, donor_slot))
        used_items.add(row.item_id)
        used_donors.add(donor_slot)
    return out


def pick_waves_after_setup(
    buckets: list,
    *,
    setup_at: datetime,
) -> list:
    """Wave buckets that release on or after final-shift setup."""
    return [b for b in buckets if b.release_at >= setup_at]


def donor_preopen_candidates(
    orders: list[OrderLine],
    ship_date: date,
    setup_assignments: list[SetupAssignment],
    *,
    wave_index: int = 0,
    wave_count: int = 1,
    waves_per_day: int = 12,
    shift_start_hour: int = 6,
    due_offset_hours: float = 4.0,
    donor_count: int = 30,
    source_count: int = 70,
    donor_slot_ids: list[str] | None = None,
    setup_at: datetime | None = None,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
) -> list[tuple[str, int, str]]:
    """
    Pre-open donor builds for the first *wave_count* pick waves after setup.

    Uses *donor_slot_ids* when set (e.g. 30 effective donors after pad-role conversion).
    Donor pads are colocated with staged source/rack station when possible.

    Returns ``(item_id, layers, suggested_donor_slot)``.
    """
    from slotting.generate_slots import donor_slot_ids as default_donor_ids
    from slotting.wave_release import order_ship_date, split_day_by_native_due

    day_orders = [o for o in orders if order_ship_date(o) == ship_date]
    if not day_orders:
        return []
    buckets = split_day_by_native_due(
        day_orders,
        ship_date=ship_date,
        due_offset_hours=due_offset_hours,
    )
    if wave_index >= len(buckets):
        return []

    layout, aisle_map = build_aisle_slot_map(
        source_count=source_count,
        donor_count=donor_count,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )
    stations = stations_per_side(layout)
    donors = list(donor_slot_ids or default_donor_ids(donor_count))
    if picker_count > 1 or center_gap_ft > 0:
        from layout_advisor.layout.slot_geometry import donor_slots_interleaved_by_flank, gap_excluded_stations

        donors = donor_slots_interleaved_by_flank(donors, aisle_map)
    gap_zone = gap_excluded_stations(layout) if center_gap_ft > 0 or picker_count > 1 else set()

    staged_rows = [
        a
        for a in setup_assignments
        if a.location in {"source_face", "rack"} and a.slot_id
    ]
    out: list[tuple[str, int, str]] = []
    seen: set[str] = set()

    eligible = pick_waves_after_setup(buckets, setup_at=setup_at or buckets[0].release_at)
    end = min(len(eligible), wave_index + max(1, wave_count))
    # All pre-opens are applied SIMULTANEOUSLY at setup, so a pad may be used
    # once across the whole batch — resetting this per wave double-booked pads
    # and the sim silently dropped the second build (occupied-pad skip).
    used_donors: set[str] = set()
    for idx in range(wave_index, end):
        wave = eligible[idx]
        wave_items = {o.item_id for o in wave.orders}
        pairs = _assign_colocated_donors(
            staged_rows,
            wave_items,
            aisle_map=aisle_map,
            stations=stations,
            donor_slot_ids=donors,
            used_donors=used_donors,
            used_items=seen,
            gap_excluded=gap_zone,
        )
        for row, donor_slot in pairs:
            layers = sum(o.qty_layers for o in wave.orders if o.item_id == row.item_id)
            out.append((row.item_id, layers, donor_slot))
            seen.add(row.item_id)
    return out


def wave_colocated_donor_opens(
    orders: list[OrderLine],
    ship_date: date,
    setup_assignments: list[SetupAssignment],
    wave_buckets: list,
    *,
    source_count: int = 70,
    donor_count: int = 30,
    donor_slot_ids: list[str] | None = None,
    setup_at: datetime | None = None,
    preopen_wave_count: int = 0,
    due_offset_hours: float = 4.0,
    initial_used_items: set[str] | None = None,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
    allow_repeat_opens: bool = True,
    multi_pallet_slices: bool = False,
    skus_path=None,
) -> list[WaveDonorOpen]:
    """
    Per-wave colocated donor opens for pick waves after setup.

    Skips the first *preopen_wave_count* eligible waves (covered at setup).
    With *allow_repeat_opens* a SKU waved in several waves gets a row PER
    wave — the sim enforces at most one open build pallet per SKU at a time
    (minimum-pallets rule), deferring later opens until the earlier build
    closes. Gated off (carry days, where repeat opens added waits), the day-long
    dedupe applies: one open per SKU, *initial_used_items* excluded entirely.

    When *multi_pallet_slices* is True, opens one donor pad per outbound
    build pallet (100″ / 2500 lb split) whose SKUs appear in the wave —
    mirroring order-ahead ``build_order_donor_opens`` on the wave path.
    """
    from slotting.generate_slots import donor_slot_ids as default_donor_ids
    from slotting.wave_release import order_ship_date, split_day_by_native_due

    if wave_buckets:
        buckets = wave_buckets
    else:
        day_orders = [o for o in orders if order_ship_date(o) == ship_date]
        buckets = split_day_by_native_due(
            day_orders,
            ship_date=ship_date,
            due_offset_hours=due_offset_hours,
        )
    eligible = pick_waves_after_setup(buckets, setup_at=setup_at or buckets[0].release_at)
    skip_indices = {w.wave_index for w in eligible[: max(0, preopen_wave_count)]}
    layout, aisle_map = build_aisle_slot_map(
        source_count=source_count,
        donor_count=donor_count,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )
    stations = stations_per_side(layout)
    donors = list(donor_slot_ids or default_donor_ids(donor_count))
    if picker_count > 1 or center_gap_ft > 0:
        from layout_advisor.layout.slot_geometry import donor_slots_interleaved_by_flank, gap_excluded_stations

        donors = donor_slots_interleaved_by_flank(donors, aisle_map)
    gap_zone = gap_excluded_stations(layout) if center_gap_ft > 0 or picker_count > 1 else set()
    staged_rows = [
        a
        for a in setup_assignments
        if a.location in {"source_face", "rack"} and a.slot_id
    ]
    staged_by_item = {row.item_id: row for row in staged_rows}
    opens: list[WaveDonorOpen] = []
    day_used_items: set[str] = set(initial_used_items or ())

    pallet_maps: dict[str, dict[str, int]] = {}
    if multi_pallet_slices:
        from layout_advisor.planning.order_affinity import build_pallet_layer_maps
        from slotting.pallet_geometry import default_site_skus_path

        pallet_maps = build_pallet_layer_maps(
            orders,
            ship_date,
            skus_path=skus_path if skus_path is not None else default_site_skus_path(),
        )

    for wave in buckets:
        if wave.wave_index in skip_indices:
            continue
        if setup_at is not None and wave.release_at < setup_at:
            continue
        wave_items = {o.item_id for o in wave.orders}
        wave_used_donors: set[str] = set()
        used_items = set() if allow_repeat_opens else day_used_items

        if multi_pallet_slices and pallet_maps:
            # One open per build pallet intersecting this wave's SKUs.
            for build_key, lines in sorted(pallet_maps.items()):
                slice_lines = {
                    sku: lyr for sku, lyr in lines.items() if sku in wave_items
                }
                if not slice_lines:
                    continue
                if any(sku in used_items for sku in slice_lines):
                    continue
                anchor_rows = [
                    staged_by_item[sku]
                    for sku in slice_lines
                    if sku in staged_by_item
                ]
                if not anchor_rows:
                    continue
                anchor = min(anchor_rows, key=lambda row: (row.rank, row.item_id))
                pairs = _assign_colocated_donors(
                    [anchor],
                    {anchor.item_id},
                    aisle_map=aisle_map,
                    stations=stations,
                    donor_slot_ids=donors,
                    used_donors=wave_used_donors,
                    used_items=set(),
                    gap_excluded=gap_zone,
                )
                if not pairs:
                    continue
                row, donor_slot = pairs[0]
                primary = min(slice_lines.keys())
                total = sum(slice_lines.values())
                opens.append(
                    WaveDonorOpen(
                        wave_index=wave.wave_index,
                        wave_send=wave.release_at,
                        wave_due=wave.due_at,
                        item_id=primary,
                        layers=total,
                        donor_slot=donor_slot,
                        source_slot=row.slot_id,
                        setup_rank=row.rank,
                        build_key=build_key,
                        build_lines=tuple(sorted(slice_lines.items())),
                    )
                )
                for sku in slice_lines:
                    used_items.add(sku)
                    day_used_items.add(sku)
            continue

        pairs = _assign_colocated_donors(
            staged_rows,
            wave_items,
            aisle_map=aisle_map,
            stations=stations,
            donor_slot_ids=donors,
            used_donors=wave_used_donors,
            used_items=used_items,
            gap_excluded=gap_zone,
        )
        for row, donor_slot in pairs:
            layers = sum(o.qty_layers for o in wave.orders if o.item_id == row.item_id)
            opens.append(
                WaveDonorOpen(
                    wave_index=wave.wave_index,
                    wave_send=wave.release_at,
                    wave_due=wave.due_at,
                    item_id=row.item_id,
                    layers=layers,
                    donor_slot=donor_slot,
                    source_slot=row.slot_id,
                    setup_rank=row.rank,
                )
            )
            day_used_items.add(row.item_id)
    return opens
