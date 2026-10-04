"""Wave-affinity face clustering and preferred-face maps for dynamic slotting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

from layout_advisor.layout.aisle_plan import AisleLayoutPlan
from layout_advisor.layout.slot_geometry import (
    MappedSlot,
    gap_excluded_stations,
    rank_face_for_travel,
    source_faces_by_travel,
    travel_rank_context,
)
from slotting.generate_slots import source_slot_ids
from slotting.models import OrderArea, OrderLine
from slotting.rack_depth import RankedSku
from slotting.wave_release import order_ship_date, wave_release_at


def primary_wave_release_by_item(
    orders: list[OrderLine],
    ship_date: date,
    *,
    due_offset_hours: float = 4.0,
) -> dict[str, datetime]:
    """Primary native wave (most layers) for each SKU on *ship_date*."""
    layers_by_item_wave: dict[str, dict[datetime, int]] = defaultdict(lambda: defaultdict(int))
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        release = wave_release_at(order.due_at, due_offset_hours=due_offset_hours)
        layers_by_item_wave[order.item_id][release] += order.qty_layers

    primary: dict[str, datetime] = {}
    for item_id, waves in layers_by_item_wave.items():
        primary[item_id] = max(waves, key=lambda rel: (waves[rel], -rel.timestamp()))
    return primary


def _window_travel_score(
    window_faces: list[str],
    *,
    aisle_map: dict[str, MappedSlot],
    dual_picker: bool,
    center_station: float,
    donors_by_flank: dict[str, set[int]],
    continuity_station: float | None = None,
) -> tuple[float, float, int, int]:
    """
    Lower is better. Prefer donor-colocated travel rank, then proximity to the
    previous wave's station centroid (*continuity_station*) so adjacent waves
    stay nearby, then aisle-center / donor hits / span.
    """
    keys = [
        rank_face_for_travel(
            face_id,
            aisle_map=aisle_map,
            dual_picker=dual_picker,
            center_station=center_station,
            donors_by_flank=donors_by_flank,
        )
        for face_id in window_faces
    ]
    avg_center = sum(key[1] for key in keys) / len(keys)
    donor_hits = sum(1 for key in keys if key[0] == 0)
    span = aisle_map[window_faces[-1]].station - aisle_map[window_faces[0]].station
    window_mid = (
        aisle_map[window_faces[0]].station + aisle_map[window_faces[-1]].station
    ) / 2.0
    continuity = (
        abs(window_mid - continuity_station) if continuity_station is not None else 0.0
    )
    return (continuity, avg_center, -donor_hits, span)


def _flank_face_pool(
    faces_by_station: list[str],
    aisle_map: dict[str, MappedSlot],
    *,
    flank: str,
    used_faces: set[str],
    gap_stations: set[int],
) -> list[str]:
    return [
        face_id
        for face_id in faces_by_station
        if aisle_map[face_id].side.value == flank
        and face_id not in used_faces
        and aisle_map[face_id].station not in gap_stations
    ]


def _best_contiguous_window(
    size: int,
    *,
    faces_by_station: list[str],
    used_faces: set[str],
    aisle_map: dict[str, MappedSlot],
    donor_stations: set[int],
    center_station: float,
    travel_fallback: list[str],
    dual_picker: bool = False,
    donors_by_flank: dict[str, set[int]] | None = None,
    continuity_station: float | None = None,
) -> list[str]:
    if size <= 0:
        return []
    candidates: list[tuple[tuple[float, float, int, int], list[str]]] = []
    for start in range(len(faces_by_station) - size + 1):
        window = faces_by_station[start : start + size]
        if any(face_id in used_faces for face_id in window):
            continue
        score = _window_travel_score(
            window,
            aisle_map=aisle_map,
            dual_picker=dual_picker,
            center_station=center_station,
            donors_by_flank=donors_by_flank or {"left": donor_stations, "right": set()},
            continuity_station=continuity_station,
        )
        candidates.append((score, window))
    if candidates:
        candidates.sort(key=lambda row: row[0])
        return candidates[0][1]
    return [face_id for face_id in travel_fallback if face_id not in used_faces][:size]


def _stations_centroid(
    face_ids: list[str],
    aisle_map: dict[str, MappedSlot],
) -> float | None:
    stations = [
        float(aisle_map[face_id].station)
        for face_id in face_ids
        if face_id in aisle_map
    ]
    if not stations:
        return None
    return sum(stations) / len(stations)


def _faces_for_chain_group(
    window: list[str],
    *,
    aisle_map: dict[str, MappedSlot],
    dual_picker: bool,
    center_station: float,
    donors_by_flank: dict[str, set[int]],
    rank_anchored: bool,
) -> list[str]:
    """Station-order faces for chain zip; orient so chain[0] gets best travel rank."""
    ordered = sorted(window, key=lambda face_id: aisle_map[face_id].station)
    if not rank_anchored or len(ordered) < 2:
        return ordered
    first_key = rank_face_for_travel(
        ordered[0],
        aisle_map=aisle_map,
        dual_picker=dual_picker,
        center_station=center_station,
        donors_by_flank=donors_by_flank,
    )
    last_key = rank_face_for_travel(
        ordered[-1],
        aisle_map=aisle_map,
        dual_picker=dual_picker,
        center_station=center_station,
        donors_by_flank=donors_by_flank,
    )
    if last_key < first_key:
        return list(reversed(ordered))
    return ordered


# Seed-stickiness weight for the order-anchored window Hungarian: moving an
# item N stations off its chain-seed face costs weight x layers x N, so only
# order-coupled travel savings that beat it justify a move. The chain seed
# carries wait-critical structure the anchor objective can't see (rank →
# donor-colocated station orientation); dropping this term trades fewer travel
# steps for more waits. A/B scripts override this module constant per arm.
WINDOW_SEED_STICKINESS = 1.0


def _assign_window_order_anchored(
    group: list[str],
    window: list[str],
    seed_faces: list[str],
    *,
    assignment: dict[str, str],
    order_layers: dict[str, dict[str, int]],
    aisle_map: dict[str, MappedSlot],
    max_iterations: int = 4,
) -> dict[str, str]:
    """Arrange a wave group on its contiguous window minimizing order travel.

    Coordinate descent on sum(layers x |face_station - order_anchor|) plus a
    seed-stickiness term (see WINDOW_SEED_STICKINESS): anchors are
    layer-weighted median stations over ALL placed order-mates (prior waves
    + this window's tentative), the assignment step is a Hungarian over the
    window. Seeded with the affinity-chain zip, so the result never regresses
    the chain heuristic on this objective. Single-item orders carry no coupling
    and are excluded from the cost.
    """
    from layout_advisor.planning.build_anchored_slotting import (
        UNASSIGNED,
        compute_build_anchors,
        hungarian_assignment,
    )

    if len(group) < 2 or len(window) < len(group):
        return dict(zip(group, seed_faces))

    multi_orders = {key: items for key, items in order_layers.items() if len(items) >= 2}
    orders_by_item: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for order_key, items in multi_orders.items():
        for item_id, layers in items.items():
            orders_by_item[item_id].append((order_key, layers))
    if not any(orders_by_item.get(item_id) for item_id in group):
        return dict(zip(group, seed_faces))

    window_stations = [float(aisle_map[f].station) for f in window]
    default_anchor = sum(window_stations) / len(window_stations)
    seed_station = {
        item_id: float(aisle_map[face_id].station)
        for item_id, face_id in zip(group, seed_faces)
    }
    item_layers = {
        item_id: sum(layers for _, layers in orders_by_item.get(item_id, ())) or 1
        for item_id in group
    }
    tentative = dict(zip(group, seed_faces))
    for _ in range(max_iterations):
        anchors = compute_build_anchors(
            face_by_item={**assignment, **tentative},
            order_layers=multi_orders,
            aisle_map=aisle_map,
            default_anchor=default_anchor,
        )
        cost = [
            [
                sum(
                    layers * abs(float(aisle_map[face_id].station) - anchors.get(order_key, default_anchor))
                    for order_key, layers in orders_by_item.get(item_id, ())
                )
                + WINDOW_SEED_STICKINESS
                * item_layers[item_id]
                * abs(float(aisle_map[face_id].station) - seed_station[item_id])
                for face_id in window
            ]
            for item_id in group
        ]
        cols = hungarian_assignment(cost)
        # UNASSIGNED means more items than faces in this window, so some item
        # genuinely has nowhere to go. Checked BEFORE indexing: `window[-1]`
        # is a valid Python index and would silently hand that item the LAST
        # face. Same outcome as before (keep the seed) — this used to arrive
        # here disguised as a duplicate, because the solver fabricated a
        # colliding column index instead of admitting the row was unplaceable.
        if any(col == UNASSIGNED for col in cols):
            return dict(zip(group, seed_faces))  # more items than faces — keep chain seed
        new = {item_id: window[cols[i]] for i, item_id in enumerate(group)}
        if len(set(new.values())) < len(new):
            return dict(zip(group, seed_faces))  # degenerate matching — keep chain seed
        if new == tentative:
            break
        tentative = new
    return tentative


def assign_faces_wave_clustered(
    ranked_item_ids: list[str],
    primary_wave: dict[str, datetime],
    *,
    aisle_map: dict[str, MappedSlot],
    layout: AisleLayoutPlan,
    source_count: int,
    donor_count: int,
    order_freq: dict[str, int] | None = None,
    pair_weights: dict[frozenset[str], int] | None = None,
    order_affinity: bool = True,
    rank_anchored_chain: bool = False,
    dual_picker: bool | None = None,
    order_layers_for_anchoring: dict[str, dict[str, int]] | None = None,
    stay_put_faces: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Assign source faces so SKUs in the same wave occupy contiguous aisle stations.

    When *order_affinity* is set, complements (same-wave co-occurrence) are chained
    on adjacent stations; high order-frequency SKUs seed each wave block unless
    *rank_anchored_chain* seeds by setup rank and orients the window toward
    donor-colocated stations.
    When *dual_picker* is set, wave SKUs alternate onto L/R faces for balanced zones.
    """
    if not ranked_item_ids:
        return {}

    dual = layout.dual_picker if dual_picker is None else dual_picker
    dual_picker_flag, center, donors_by_flank = travel_rank_context(aisle_map, layout)
    if dual_picker is not None:
        dual_picker_flag = dual
    donor_stations = {
        mapped.station for mapped in aisle_map.values() if mapped.zone == "donor_face"
    }
    faces_by_station = sorted(
        source_slot_ids(source_count),
        key=lambda face_id: aisle_map[face_id].station,
    )
    gap_stations = gap_excluded_stations(layout)
    if gap_stations:
        faces_by_station = [
            face_id for face_id in faces_by_station if aisle_map[face_id].station not in gap_stations
        ]
    travel_fallback = source_faces_by_travel(
        source_count=source_count,
        donor_count=donor_count,
        center_gap_ft=layout.center_gap_ft,
        picker_count=layout.picker_count,
    )

    rank_index = {item_id: index for index, item_id in enumerate(ranked_item_ids)}
    assignment: dict[str, str] = {}
    used_faces: set[str] = set()

    if stay_put_faces:
        # Stay-put pinning: a SKU already sitting on a valid source face keeps
        # it — the nightly full re-slot pays an unmodeled physical relocation
        # per repeat SKU, and without pinning almost no repeat SKU keeps its face.
        valid_faces = set(faces_by_station)
        for item_id in ranked_item_ids:
            face_id = stay_put_faces.get(item_id)
            if face_id and face_id in valid_faces and face_id not in used_faces:
                assignment[item_id] = face_id
                used_faces.add(face_id)

    unpinned = [i for i in ranked_item_ids if i not in assignment]
    wave_groups: dict[datetime, list[str]] = defaultdict(list)
    for item_id in unpinned:
        wave_groups[primary_wave[item_id]].append(item_id)
    for wave_time in wave_groups:
        wave_groups[wave_time].sort(key=lambda item_id: rank_index[item_id])

    prev_centroid: float | None = None
    # Chronological waves + continuity: each block prefers stations near the
    # previous wave so mid-day transitions stay local (adjacent waves, adjacent blocks).
    for wave_time in sorted(wave_groups):
        group = wave_groups[wave_time]
        if order_affinity and pair_weights is not None and order_freq is not None:
            from layout_advisor.planning.order_affinity import affinity_chain_order

            group = affinity_chain_order(
                group,
                order_freq=order_freq,
                pair_weights=pair_weights,
                rank_index=rank_index,
                seed_by_rank=rank_anchored_chain,
            )
        else:
            group.sort(key=lambda item_id: rank_index[item_id])
        wave_faces: list[str] = []
        if dual_picker_flag:
            n_left = (len(group) + 1) // 2
            n_right = len(group) - n_left
            left_pool = _flank_face_pool(
                faces_by_station,
                aisle_map,
                flank="left",
                used_faces=used_faces,
                gap_stations=gap_stations,
            )
            right_pool = _flank_face_pool(
                faces_by_station,
                aisle_map,
                flank="right",
                used_faces=used_faces,
                gap_stations=gap_stations,
            )
            left_window = _best_contiguous_window(
                n_left,
                faces_by_station=left_pool,
                used_faces=used_faces,
                aisle_map=aisle_map,
                donor_stations=donor_stations,
                center_station=center,
                travel_fallback=[f for f in travel_fallback if aisle_map[f].side.value == "left"],
                dual_picker=True,
                donors_by_flank=donors_by_flank,
                continuity_station=prev_centroid,
            )
            right_window = _best_contiguous_window(
                n_right,
                faces_by_station=right_pool,
                used_faces=used_faces,
                aisle_map=aisle_map,
                donor_stations=donor_stations,
                center_station=center,
                travel_fallback=[f for f in travel_fallback if aisle_map[f].side.value == "right"],
                dual_picker=True,
                donors_by_flank=donors_by_flank,
                continuity_station=prev_centroid,
            )
            left_faces_by_travel = sorted(
                left_window,
                key=lambda face_id: rank_face_for_travel(
                    face_id,
                    aisle_map=aisle_map,
                    dual_picker=True,
                    center_station=center,
                    donors_by_flank=donors_by_flank,
                ),
            )
            right_faces_by_travel = sorted(
                right_window,
                key=lambda face_id: rank_face_for_travel(
                    face_id,
                    aisle_map=aisle_map,
                    dual_picker=True,
                    center_station=center,
                    donors_by_flank=donors_by_flank,
                ),
            )
            if order_affinity and pair_weights is not None:
                # Preserve the affinity chain under the flank split: contiguous
                # chain segments stay with one picker (same-order SKUs on one
                # flank) and chain order maps to station order. Alternating
                # [0::2]/[1::2] here put chain NEIGHBORS on opposite flanks and
                # re-sorting by rank discarded the chain (same no-op bug fixed
                # in the single-picker branch below).
                left_items = group[:n_left]
                right_items = group[n_left:]
                left_faces = _faces_for_chain_group(
                    left_window,
                    aisle_map=aisle_map,
                    dual_picker=True,
                    center_station=center,
                    donors_by_flank=donors_by_flank,
                    rank_anchored=rank_anchored_chain,
                )
                right_faces = _faces_for_chain_group(
                    right_window,
                    aisle_map=aisle_map,
                    dual_picker=True,
                    center_station=center,
                    donors_by_flank=donors_by_flank,
                    rank_anchored=rank_anchored_chain,
                )
            else:
                left_items = sorted(group[0::2], key=lambda item_id: rank_index[item_id])
                right_items = sorted(group[1::2], key=lambda item_id: rank_index[item_id])
                left_faces = left_faces_by_travel
                right_faces = right_faces_by_travel
            for item_id, face_id in zip(left_items, left_faces):
                assignment[item_id] = face_id
                used_faces.add(face_id)
                wave_faces.append(face_id)
            for item_id, face_id in zip(right_items, right_faces):
                assignment[item_id] = face_id
                used_faces.add(face_id)
                wave_faces.append(face_id)
            prev_centroid = _stations_centroid(wave_faces, aisle_map)
            continue
        window = _best_contiguous_window(
            len(group),
            faces_by_station=faces_by_station,
            used_faces=used_faces,
            aisle_map=aisle_map,
            donor_stations=donor_stations,
            center_station=center,
            travel_fallback=travel_fallback,
            dual_picker=dual_picker_flag,
            donors_by_flank=donors_by_flank,
            continuity_station=prev_centroid,
        )
        window_sorted = sorted(
            window,
            key=lambda face_id: rank_face_for_travel(
                face_id,
                aisle_map=aisle_map,
                dual_picker=dual_picker_flag,
                center_station=center,
                donors_by_flank=donors_by_flank,
            ),
        )
        if order_affinity and pair_weights is not None:
            # Preserve the affinity-chain order onto station-ordered faces so
            # chain adjacency (same-order / co-waved SKUs) becomes PHYSICAL
            # adjacency. Re-sorting the group by rank here (the old behavior)
            # silently discarded the chain and made order affinity a no-op.
            window_by_station = _faces_for_chain_group(
                window,
                aisle_map=aisle_map,
                dual_picker=dual_picker_flag,
                center_station=center,
                donors_by_flank=donors_by_flank,
                rank_anchored=rank_anchored_chain,
            )
            if order_layers_for_anchoring is not None:
                # Known limitation: single-picker only — dual-picker keeps the
                # chain zip; extend if a deployment runs two pickers.
                block = _assign_window_order_anchored(
                    group,
                    window,
                    window_by_station,
                    assignment=assignment,
                    order_layers=order_layers_for_anchoring,
                    aisle_map=aisle_map,
                )
            else:
                block = dict(zip(group, window_by_station))
            for item_id, face_id in block.items():
                assignment[item_id] = face_id
                used_faces.add(face_id)
                wave_faces.append(face_id)
            prev_centroid = _stations_centroid(wave_faces, aisle_map)
            continue
        group_by_touch = sorted(group, key=lambda item_id: rank_index[item_id])
        for item_id, face_id in zip(group_by_touch, window_sorted):
            assignment[item_id] = face_id
            used_faces.add(face_id)
            wave_faces.append(face_id)
        prev_centroid = _stations_centroid(wave_faces, aisle_map)
    if len(assignment) < len(ranked_item_ids):
        # A window smaller than its group (flank pool exhausted, gap-excluded
        # stations) zip-truncates silently — top-ranked SKUs would lose their
        # face and demote to rack/freezer while faces sit free. Sweep leftovers
        # onto remaining faces in travel order.
        leftovers = [i for i in ranked_item_ids if i not in assignment]
        free_faces = [f for f in faces_by_station if f not in used_faces]
        free_faces.sort(
            key=lambda face_id: rank_face_for_travel(
                face_id,
                aisle_map=aisle_map,
                dual_picker=dual_picker_flag,
                center_station=center,
                donors_by_flank=donors_by_flank,
            )
        )
        for item_id, face_id in zip(leftovers, free_faces):
            assignment[item_id] = face_id
            used_faces.add(face_id)
    return assignment


def assign_faces_travel_only(
    ranked_item_ids: list[str],
    *,
    source_count: int,
    donor_count: int,
) -> dict[str, str]:
    """Legacy travel-only assignment (rank index → travel-ordered face)."""
    face_order = source_faces_by_travel(source_count=source_count, donor_count=donor_count)
    return {
        item_id: face_order[index]
        for index, item_id in enumerate(ranked_item_ids[:source_count])
    }


def build_face_by_item(
    ranked_skus: list[RankedSku],
    orders: list[OrderLine],
    ship_date: date,
    *,
    source_count: int,
    donor_count: int,
    aisle_map: dict[str, MappedSlot],
    layout: AisleLayoutPlan,
    due_offset_hours: float = 4.0,
    wave_clustering: bool = True,
    order_affinity: bool = True,
    rank_anchored_chain: bool = False,
    build_anchored_slotting: bool = False,
    order_anchored_window_assign: bool = False,
    prev_face_by_item: dict[str, str] | None = None,
    build_anchored_light_day_sku_threshold: int = 0,
    build_anchored_light_day_stickiness: float = 4.0,
    build_anchored_prelight_stickiness: bool = False,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    stage_item_budget: int | None = None,
    affinity_orders: list[OrderLine] | None = None,
    skus_path: Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    """
    Map top-ranked SKUs to source faces.

    Returns ``(face_by_item, face_order)`` where *face_order* is the travel-ranked
    face list used for rack colocation and virtual preferred faces.

    *skus_path* is the site's SKU master, used for build-pallet affinity weights.
    ``None`` falls back to the default local SKU master (``data/site/skus.csv``),
    which is wrong for any other site — pass ``cfg.skus_path`` through.
    """
    face_order = source_faces_by_travel(
        source_count=source_count,
        donor_count=donor_count,
        center_gap_ft=layout.center_gap_ft,
        picker_count=layout.picker_count,
    )
    top_items = [sku.item_id for sku in ranked_skus[: (stage_item_budget or source_count)]]
    if wave_clustering and top_items:
        primary_wave = primary_wave_release_by_item(
            orders,
            ship_date,
            due_offset_hours=due_offset_hours,
        )
        order_freq = None
        pair_weights = None
        if order_affinity:
            from layout_advisor.planning.order_affinity import (
                combined_affinity_weights,
                ship_day_order_frequency,
            )

            affinity_src = affinity_orders if affinity_orders is not None else orders
            order_freq = ship_day_order_frequency(affinity_src, ship_date)
            # Same *build pallet* dominates same-order / same-wave: the picker
            # shuttles one pad ↔ that pallet's SKU faces (multi-pallet aware).
            from slotting.pallet_geometry import default_site_skus_path

            # Honour the caller's SKU master. Reading `default_site_skus_path()`
            # unconditionally computed order-affinity weights for ANY site from
            # one site's pallet geometry — silently, since a
            # wrong-but-present SKU file produces plausible weights. The three
            # sibling call sites (dynamic_slotting, build_anchored_slotting,
            # daily_plan) already used this override form; this one did not.
            pair_weights = combined_affinity_weights(
                affinity_src,
                ship_date,
                due_offset_hours=due_offset_hours,
                skus_path=skus_path if skus_path is not None else default_site_skus_path(),
            )
        if build_anchored_slotting:
            from layout_advisor.planning.build_anchored_slotting import assign_faces_build_anchored

            face_by_item = assign_faces_build_anchored(
                top_items,
                primary_wave,
                orders,
                ship_date,
                aisle_map=aisle_map,
                layout=layout,
                source_count=source_count,
                donor_count=donor_count,
                dual_picker=layout.dual_picker,
                stay_put_faces=prev_face_by_item,
                light_day_sku_threshold=build_anchored_light_day_sku_threshold,
                light_day_stickiness=build_anchored_light_day_stickiness,
                prelight_stickiness=build_anchored_prelight_stickiness,
                pick_start_hour=pick_start_hour,
                final_shift_hour=final_shift_hour,
            )
        else:
            order_layers = None
            if order_anchored_window_assign:
                from layout_advisor.planning.build_anchored_slotting import _order_layers_on_ship_day

                affinity_src = affinity_orders if affinity_orders is not None else orders
                order_layers = _order_layers_on_ship_day(affinity_src, ship_date)
            face_by_item = assign_faces_wave_clustered(
                top_items,
                primary_wave,
                aisle_map=aisle_map,
                layout=layout,
                source_count=source_count,
                donor_count=donor_count,
                order_freq=order_freq,
                pair_weights=pair_weights,
                order_affinity=order_affinity,
                rank_anchored_chain=rank_anchored_chain,
                dual_picker=layout.dual_picker,
                order_layers_for_anchoring=order_layers,
                stay_put_faces=prev_face_by_item,
            )
    else:
        face_by_item = assign_faces_travel_only(
            top_items,
            source_count=source_count,
            donor_count=donor_count,
        )
    return face_by_item, face_order


def wave_anchor_faces(
    face_by_item: dict[str, str],
    primary_wave: dict[str, datetime],
    *,
    rank_index: dict[str, int] | None = None,
) -> dict[datetime, str]:
    """Best staged source face per native wave (lowest setup rank wins)."""
    anchors: dict[datetime, str] = {}
    best_rank: dict[datetime, int] = {}
    for item_id, face_id in face_by_item.items():
        wave_time = primary_wave.get(item_id)
        if wave_time is None:
            continue
        rank = rank_index.get(item_id, 10**9) if rank_index is not None else 0
        if wave_time not in anchors or rank < best_rank.get(wave_time, 10**9):
            anchors[wave_time] = face_id
            best_rank[wave_time] = rank
    return anchors


def overflow_promote_face(
    rank_index: int,
    face_order: list[str],
    *,
    source_count: int,
) -> str:
    """
    Promote target for freezer-hold overflow — tail-biased, not modulo wrap.

    Modulo mapping (``index % len(face_order)``) lands high-rank overflow SKUs
    on prime faces; clamp to the single travel-worst face instead. Spreading
    overflow across the tail quarter was tried and rejected: it lowered sim LP,
    because collapsed colocation keeps anchorless overflow rack traffic in one
    station block.
    """
    if not face_order:
        return ""
    if rank_index < len(face_order):
        return face_order[rank_index]
    return face_order[-1]


def preferred_source_faces(
    ranked_skus: list[RankedSku],
    face_by_item: dict[str, str],
    face_order: list[str],
    *,
    source_count: int,
    primary_wave: dict[str, datetime] | None = None,
) -> dict[str, str]:
    """
    Target source face per SKU — staged faces for top ranks, virtual faces for overflow.

    Overflow virtual faces prefer the staged wave anchor (same aisle block as pick entry).
    Mid-day face swaps promote freezer-hold SKUs onto their preferred face when it opens.
    """
    preferred = dict(face_by_item)
    rank_index = {sku.item_id: index for index, sku in enumerate(ranked_skus)}
    wave_anchors = (
        wave_anchor_faces(face_by_item, primary_wave, rank_index=rank_index)
        if primary_wave
        else {}
    )
    for index, sku in enumerate(ranked_skus):
        if sku.item_id in preferred:
            continue
        wave_time = primary_wave.get(sku.item_id) if primary_wave else None
        if wave_time is not None and wave_time in wave_anchors:
            preferred[sku.item_id] = wave_anchors[wave_time]
        else:
            preferred[sku.item_id] = overflow_promote_face(
                index,
                face_order,
                source_count=source_count,
            )
    return preferred


def refresh_preferred_faces_for_window(
    preferred: dict[str, str],
    *,
    orders: list[OrderLine],
    ship_date: date,
    as_of: datetime,
    aisle_map: dict[str, MappedSlot],
    face_by_item: dict[str, str],
    horizon_hours: float = 4.0,
    due_offset_hours: float = 4.0,
) -> dict[str, str]:
    """
    Soft mid-day prefer refresh: relocate *virtual* (not physically staged) preferred
    faces for SKUs due in the next *horizon_hours* toward the active wave block.

    Staged ``face_by_item`` assignments are left alone (no floor move). Empty/virtual
    preferred targets for upcoming waved SKUs snap to the station centroid of
    already-staged SKUs sharing that primary wave — so ADD picks land nearer the
    current pick zone. When no same-wave stage exists, fall back to the centroid of
    all staged faces in the upcoming window.
    """
    from datetime import timedelta

    from slotting.models import OrderArea
    from slotting.wave_release import order_ship_date

    cutoff = as_of + timedelta(hours=horizon_hours)
    upcoming: set[str] = set()
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        if as_of <= order.due_at <= cutoff:
            upcoming.add(order.item_id)
    if not upcoming:
        return preferred

    primary = primary_wave_release_by_item(
        orders, ship_date, due_offset_hours=due_offset_hours
    )
    # Station centroid of staged SKUs in each primary wave.
    wave_stations: dict[datetime, list[float]] = defaultdict(list)
    all_staged: list[float] = []
    for item_id, face_id in face_by_item.items():
        if face_id not in aisle_map:
            continue
        st = float(aisle_map[face_id].station)
        all_staged.append(st)
        wave = primary.get(item_id)
        if wave is not None:
            wave_stations[wave].append(st)
    wave_centroid = {
        wave: sum(vals) / len(vals) for wave, vals in wave_stations.items() if vals
    }
    global_centroid = sum(all_staged) / len(all_staged) if all_staged else None

    # Faces currently claimed as physical stage destinations. Candidates are
    # source faces only — aisle_map also holds donor pads, and "D.." even wins
    # station ties against "S.." lexicographically.
    staged_faces = set(face_by_item.values())
    candidate_faces = [
        face_id
        for face_id in {*(preferred.values()), *aisle_map}
        if (mapped := aisle_map.get(face_id)) is not None
        and mapped.zone == "source_face"
        and face_id not in staged_faces
    ]
    if not candidate_faces:
        candidate_faces = [f for f in face_by_item.values() if f in aisle_map]
    if not candidate_faces:
        return preferred

    updated = dict(preferred)
    for item_id in upcoming:
        if item_id in face_by_item:
            continue
        wave = primary.get(item_id)
        target = wave_centroid.get(wave) if wave is not None else None
        if target is None:
            target = global_centroid
        if target is None:
            continue
        best = min(
            candidate_faces,
            key=lambda face_id: (
                abs(float(aisle_map[face_id].station) - target),
                face_id,
            ),
        )
        updated[item_id] = best
    return updated


def nearest_empty_face(
    preferred_face: str | None,
    empty_faces: list[str],
    *,
    face_station_by_id: Mapping[str, int],
    claimed: set[str] | None = None,
) -> str | None:
    """Pick empty face closest in station index to *preferred_face* (travel soft landing)."""
    claimed = claimed or set()
    free = [f for f in empty_faces if f not in claimed]
    if not free:
        return None
    if preferred_face is None or preferred_face not in face_station_by_id:
        return free[0]
    pref_st = face_station_by_id[preferred_face]
    return min(
        free,
        key=lambda face_id: (
            abs(face_station_by_id.get(face_id, pref_st) - pref_st),
            face_id,
        ),
    )


def setup_rank_by_item(ranked_skus: list[RankedSku]) -> dict[str, int]:
    return {sku.item_id: sku.rank for sku in ranked_skus}


def wave_station_span(
    item_ids: list[str],
    face_by_item: dict[str, str],
    aisle_map: dict[str, MappedSlot],
) -> int:
    """Max − min station index for *item_ids* (0 when empty or one SKU)."""
    stations = [
        aisle_map[face_by_item[item_id]].station
        for item_id in item_ids
        if item_id in face_by_item
    ]
    if len(stations) < 2:
        return 0
    return max(stations) - min(stations)
