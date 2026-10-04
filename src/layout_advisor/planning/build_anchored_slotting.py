"""Build-anchored face assignment — layer-weighted median + Hungarian per wave.

True picker travel for a multi-line *build pallet* is approximately::

    sum(layers x |face_station - build_station|)

where *build_station* is the layer-weighted median of that pallet's face
stations (1-median / Weber problem on a line). Large customer orders split
into multiple outbound pallets (100″ / 2500 lb); each pad is its own travel
unit. This module alternates:

1. Fix build anchors from the current face assignment.
2. Solve a linear assignment (Hungarian) per wave block to minimize travel.
3. Repeat until stable or *max_iterations*.

Replaces the station-span proxy used by wave clustering + affinity chains when
``build_anchored_slotting`` is enabled on ``ReplayConfig``.
"""

from __future__ import annotations

from collections import defaultdict
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
from layout_advisor.planning.order_affinity import true_order_key
from slotting.generate_slots import source_slot_ids
from slotting.models import OrderArea, OrderLine

# Compare-script hook: (sku_threshold, stickiness). None = use call kwargs /
# defaults. Keeps ReplayConfig hash frozen until an intentional adoption.
LIGHT_DAY_SAFE_OVERRIDE: tuple[int, float] | None = None


def weighted_median(values: list[float], weights: list[int]) -> float:
    """Layer-weighted median station index (Weber 1-median on a line)."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2:
            return value
    return pairs[-1][0]


def _order_layers_on_ship_day(
    orders: list[OrderLine],
    ship_date: date,
) -> dict[str, dict[str, int]]:
    """order_key -> {item_id: layers} for THIS_LINE on *ship_date*."""
    by_order: dict[str, dict[str, int]] = defaultdict(dict)
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        from slotting.wave_release import order_ship_date

        if order_ship_date(order) != ship_date:
            continue
        key = true_order_key(order.order_id)
        by_order[key][order.item_id] = by_order[key].get(order.item_id, 0) + order.qty_layers
    return dict(by_order)


def _item_to_units(
    unit_layers: dict[str, dict[str, int]],
) -> dict[str, list[tuple[str, int]]]:
    """item_id -> [(unit_key, layers_on_unit), ...] across build pallets / orders."""
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for unit_key, items in unit_layers.items():
        for item_id, layers in items.items():
            out[item_id].append((unit_key, layers))
    return dict(out)


def _item_to_order(
    order_layers: dict[str, dict[str, int]],
) -> dict[str, str]:
    """Backward-compat: first unit key wins per item (prefer :func:`_item_to_units`)."""
    out: dict[str, str] = {}
    for order_key, items in order_layers.items():
        for item_id in items:
            out.setdefault(item_id, order_key)
    return out


def compute_build_anchors(
    *,
    face_by_item: dict[str, str],
    order_layers: dict[str, dict[str, int]],
    aisle_map: dict[str, MappedSlot],
    default_anchor: float,
) -> dict[str, float]:
    """Layer-weighted median station per travel unit (order or build pallet)."""
    anchors: dict[str, float] = {}
    for order_key, items in order_layers.items():
        stations: list[float] = []
        weights: list[int] = []
        for item_id, layers in items.items():
            face_id = face_by_item.get(item_id)
            if face_id is None or face_id not in aisle_map:
                continue
            stations.append(float(aisle_map[face_id].station))
            weights.append(layers)
        if len(stations) >= 2:
            anchors[order_key] = weighted_median(stations, weights)
        elif len(stations) == 1:
            anchors[order_key] = stations[0]
        else:
            anchors[order_key] = default_anchor
    return anchors


def order_travel_cost(
    *,
    item_id: str,
    face_id: str,
    layers: int,
    item_to_order: dict[str, str] | None = None,
    item_to_units: dict[str, list[tuple[str, int]]] | None = None,
    build_anchors: dict[str, float],
    aisle_map: dict[str, MappedSlot],
    default_anchor: float,
) -> float:
    """
    Travel cost of placing *item_id* on *face_id*.

    When *item_to_units* is provided (build-pallet slices), sums
    ``unit_layers x |face - unit_anchor|`` across every unit containing the
    item. Otherwise falls back to a single order anchor via *item_to_order*.
    """
    station = float(aisle_map[face_id].station)
    if item_to_units is not None:
        units = item_to_units.get(item_id)
        if not units:
            return float(layers) * abs(station - default_anchor)
        return sum(
            unit_layers * abs(station - build_anchors.get(unit_key, default_anchor))
            for unit_key, unit_layers in units
        )
    order_key = (item_to_order or {}).get(item_id, item_id)
    anchor = build_anchors.get(order_key, default_anchor)
    return layers * abs(station - anchor)


def total_order_travel(
    face_by_item: dict[str, str],
    *,
    order_layers: dict[str, dict[str, int]],
    aisle_map: dict[str, MappedSlot],
    default_anchor: float,
) -> float:
    """Sum of per-order weighted travel for multi-line orders (measurement helper)."""
    anchors = compute_build_anchors(
        face_by_item=face_by_item,
        order_layers=order_layers,
        aisle_map=aisle_map,
        default_anchor=default_anchor,
    )
    total = 0.0
    for order_key, items in order_layers.items():
        if len(items) < 2:
            continue
        for item_id, layers in items.items():
            face_id = face_by_item.get(item_id)
            if face_id is None or face_id not in aisle_map:
                continue
            station = float(aisle_map[face_id].station)
            anchor = anchors.get(order_key, default_anchor)
            total += layers * abs(station - anchor)
    return total


def total_build_pallet_travel(
    face_by_item: dict[str, str],
    *,
    orders: list,
    ship_date: date,
    aisle_map: dict[str, MappedSlot],
    default_anchor: float,
    skus_path: Path | None = None,
) -> float:
    """
    Weighted travel summed per outbound *build pallet* (not whole customer order).

    Large orders split across multiple pads; each pad has its own median face
    station. This is the travel the picker actually walks when building pallet
    by pallet.
    """
    from layout_advisor.planning.order_affinity import build_pallet_layer_maps

    return total_order_travel(
        face_by_item,
        order_layers=build_pallet_layer_maps(
            orders, ship_date, skus_path=skus_path
        ),
        aisle_map=aisle_map,
        default_anchor=default_anchor,
    )

#: Returned for a row that has no column to take (more rows than columns).
#: Callers MUST check for it — it is deliberately not a usable index, and in
#: particular ``window[UNASSIGNED]`` would silently select the LAST element.
UNASSIGNED = -1


def hungarian_assignment(cost: list[list[float]]) -> list[int]:
    """
    Minimum-cost assignment: row *i* → column ``result[i]``, or ``UNASSIGNED``.

    Classic Kuhn-Munkres for square matrices. Pads rectangular inputs internally.

    When there are more rows than columns some rows cannot be placed at all;
    those come back as :data:`UNASSIGNED` rather than a fabricated index.

    Costs may be any sign. The padding sentinel is derived from the observed
    range rather than assuming non-negative costs, so a score-style matrix
    (where lower is better and values are negative) is handled correctly.
    """
    if not cost:
        return []
    n_rows = len(cost)
    n_cols = len(cost[0]) if cost else 0
    n = max(n_rows, n_cols)
    # Padding sentinel: strictly dearer than any complete real assignment, so a
    # padded cell is never preferred over a real one. Derived from the observed
    # RANGE, not from `max * n + 1`, which assumed costs are non-negative — with
    # an all-negative (score-style) matrix that formula produces a sentinel
    # CHEAPER than real cells and the solver prefers the padding. Every current
    # caller happens to build non-negative costs from abs() distances; nothing
    # enforced it.
    lo = min(min(row) for row in cost)
    hi = max(max(row) for row in cost)
    big = hi + n * (hi - lo) + 1.0
    padded = [[big] * n for _ in range(n)]
    for i in range(n_rows):
        for j in range(n_cols):
            padded[i][j] = cost[i][j]

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = padded[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = [UNASSIGNED] * n_rows
    for j in range(1, n + 1):
        if p[j] != 0 and p[j] - 1 < n_rows and j - 1 < n_cols:
            result[p[j] - 1] = j - 1
    # Rows left over when there are more rows than columns are genuinely
    # unplaceable, and they stay UNASSIGNED.
    #
    # This block used to fabricate `min(i, n_cols - 1)`, which is not a real
    # assignment: with 3 rows and 2 columns it returned [0, 1, 1], handing column
    # 1 to two rows at once. Both call sites already defended against the result
    # — wave_slotting discards the whole optimisation as a "degenerate matching"
    # and reverts to the seed, rack_depth reassigns to the first free slot — so
    # the visible symptom was the optimiser quietly doing nothing precisely when
    # items outnumber faces, which is when it is worth most. The invisible one
    # was worse: when a real column happened to be free, the fabricated index
    # could slip past the duplicate guard as a plausible wrong assignment.
    return result


def light_day_softness(peak_skus: int, threshold: int) -> float:
    """0 on heavy days; 1 on light days (peak waved SKUs below *threshold*).

    Binary on purpose — mirrors scaled twin-k, which zeroes *k* on light days
    without flipping the policy family. A soft ramp leaves days just under the
    threshold almost undamped. Per-day BA on/off stays forbidden: toggling the
    policy family between days poisons the carried floor state.
    """
    if threshold <= 0 or peak_skus >= threshold:
        return 0.0
    return 1.0


def next_ship_day(orders: list[OrderLine], ship_date: date) -> date | None:
    """Earliest ship day in *orders* strictly after *ship_date* (prelight look-ahead)."""
    from slotting.wave_release import order_ship_date

    later = sorted(
        {order_ship_date(order) for order in orders if order_ship_date(order) > ship_date}
    )
    return later[0] if later else None


def resolve_build_anchored_stickiness(
    orders: list[OrderLine],
    ship_date: date,
    *,
    light_day_sku_threshold: int,
    light_day_stickiness: float,
    prelight_stickiness: bool = False,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
) -> float:
    """Stickiness weight for BA on *ship_date* (0 = pure travel BA).

    Applies full *light_day_stickiness* when this day is light, or (when
    *prelight_stickiness*) when the next ship day in the book is light.
    """
    if light_day_sku_threshold <= 0 or light_day_stickiness <= 0:
        return 0.0
    from layout_advisor.planning.donor_build_queue import ship_day_peak_waved_demand

    peak_skus, _ = ship_day_peak_waved_demand(
        orders,
        ship_date,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
    )
    if light_day_softness(peak_skus, light_day_sku_threshold) > 0:
        return light_day_stickiness
    if not prelight_stickiness:
        return 0.0
    nxt = next_ship_day(orders, ship_date)
    if nxt is None:
        return 0.0
    next_peak, _ = ship_day_peak_waved_demand(
        orders,
        nxt,
        pick_start_hour=pick_start_hour,
        final_shift_hour=final_shift_hour,
    )
    if light_day_softness(next_peak, light_day_sku_threshold) > 0:
        return light_day_stickiness
    return 0.0


def _assign_block_hungarian(
    items: list[str],
    faces: list[str],
    *,
    item_layers: dict[str, int],
    item_to_order: dict[str, str] | None = None,
    item_to_units: dict[str, list[tuple[str, int]]] | None = None,
    build_anchors: dict[str, float],
    aisle_map: dict[str, MappedSlot],
    default_anchor: float,
    seed_by_item: dict[str, str] | None = None,
    stickiness: float = 0.0,
) -> dict[str, str]:
    if not items or not faces:
        return {}
    n = len(items)
    m = len(faces)
    if n > m:
        items = items[:m]
        n = m
    seed = seed_by_item or {}

    def _cell(item_id: str, face_id: str) -> float:
        layers = item_layers.get(item_id, 1)
        cost = order_travel_cost(
            item_id=item_id,
            face_id=face_id,
            layers=layers,
            item_to_order=item_to_order,
            item_to_units=item_to_units,
            build_anchors=build_anchors,
            aisle_map=aisle_map,
            default_anchor=default_anchor,
        )
        if stickiness <= 0.0:
            return cost
        seed_face = seed.get(item_id)
        if seed_face is None or seed_face not in aisle_map or face_id not in aisle_map:
            return cost
        seed_station = float(aisle_map[seed_face].station)
        station = float(aisle_map[face_id].station)
        return cost + stickiness * float(layers) * abs(station - seed_station)

    matrix = [[_cell(item_id, face_id) for face_id in faces] for item_id in items]
    col_assign = hungarian_assignment(matrix)
    used_cols: set[int] = set()
    out: dict[str, str] = {}
    for row_idx, item_id in enumerate(items):
        col = col_assign[row_idx]
        if col in used_cols or col >= len(faces):
            free = next((j for j in range(len(faces)) if j not in used_cols), None)
            col = free if free is not None else col
        used_cols.add(col)
        out[item_id] = faces[col]
    return out


def _alternating_wave_assign(
    items: list[str],
    faces: list[str],
    *,
    order_layers: dict[str, dict[str, int]],
    item_layers: dict[str, int],
    aisle_map: dict[str, MappedSlot],
    default_anchor: float,
    max_iterations: int = 8,
    seed: dict[str, str] | None = None,
    stickiness: float = 0.0,
    stickiness_seed: dict[str, str] | None = None,
) -> dict[str, str]:
    """Alternate build-anchor update + Hungarian assignment until stable."""
    item_to_units = _item_to_units(order_layers)
    item_to_order = _item_to_order(order_layers)
    # Default continuity seed = travel-ranked zip. Callers may pass carried
    # faces as stickiness_seed (stay-put) without changing the BA algorithm.
    travel_seed = {item_id: face_id for item_id, face_id in zip(items, faces)}
    continuity_seed = dict(travel_seed)
    if stickiness_seed:
        free = set(faces)
        for item_id in items:
            prior = stickiness_seed.get(item_id)
            if prior in free:
                continuity_seed[item_id] = prior
                free.discard(prior)
    assignment = dict(seed or {})
    for item_id in items:
        assignment.pop(item_id, None)
    if not assignment:
        assignment.update(continuity_seed)
    for _ in range(max_iterations):
        build_anchors = compute_build_anchors(
            face_by_item=assignment,
            order_layers=order_layers,
            aisle_map=aisle_map,
            default_anchor=default_anchor,
        )
        for order_key, order_items in order_layers.items():
            if len(order_items) == 1:
                item_id = next(iter(order_items))
                if item_id in items:
                    build_anchors[order_key] = default_anchor
        new_assign = _assign_block_hungarian(
            items,
            faces,
            item_layers=item_layers,
            item_to_order=item_to_order,
            item_to_units=item_to_units,
            build_anchors=build_anchors,
            aisle_map=aisle_map,
            default_anchor=default_anchor,
            seed_by_item=continuity_seed,
            stickiness=stickiness,
        )
        if new_assign == {k: assignment[k] for k in items if k in assignment}:
            break
        assignment.update(new_assign)
    return {k: assignment[k] for k in items if k in assignment}


def assign_faces_build_anchored(
    ranked_item_ids: list[str],
    primary_wave: dict[str, datetime],
    orders: list[OrderLine],
    ship_date: date,
    *,
    aisle_map: dict[str, MappedSlot],
    layout: AisleLayoutPlan,
    source_count: int,
    donor_count: int,
    dual_picker: bool | None = None,
    max_iterations: int = 8,
    skus_path: Path | None = None,
    use_build_pallet_slices: bool = True,
    light_day_sku_threshold: int = 0,
    light_day_stickiness: float = 4.0,
    prelight_stickiness: bool = False,
    pick_start_hour: int = 6,
    final_shift_hour: int = 22,
    stay_put_faces: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Assign source faces minimizing build-anchored picker travel per wave block.

    Processes waves in chronological order; each wave gets a Hungarian assignment
    against the remaining free faces (travel-ranked). Dual-picker mode splits each
    wave's SKUs across L/R flanks before assignment.

    When *use_build_pallet_slices* is True (default), travel units are outbound
    build pallets (100″ / 2500 lb split) rather than whole customer orders — the
    picker walks one pad at a time.

    When *light_day_sku_threshold* > 0, light (and optionally pre-light) days
    add stickiness toward the continuity seed (travel zip, or *stay_put_faces*
    when provided) inside Hungarian — same BA algorithm every day, not a mode toggle.
    """
    if not ranked_item_ids:
        return {}

    dual = layout.dual_picker if dual_picker is None else dual_picker
    dual_picker_flag, center, donors_by_flank = travel_rank_context(aisle_map, layout)
    if dual_picker is not None:
        dual_picker_flag = dual

    gap_stations = gap_excluded_stations(layout)
    faces_by_station = sorted(
        source_slot_ids(source_count),
        key=lambda face_id: aisle_map[face_id].station,
    )
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

    if use_build_pallet_slices:
        from layout_advisor.planning.order_affinity import build_pallet_layer_maps
        from slotting.pallet_geometry import default_site_skus_path

        order_layers = build_pallet_layer_maps(
            orders,
            ship_date,
            skus_path=skus_path if skus_path is not None else default_site_skus_path(),
        )
        if not order_layers:
            order_layers = _order_layers_on_ship_day(orders, ship_date)
    else:
        order_layers = _order_layers_on_ship_day(orders, ship_date)
    item_layers: dict[str, int] = {}
    for _order_key, items in order_layers.items():
        for item_id, layers in items.items():
            item_layers[item_id] = item_layers.get(item_id, 0) + layers

    if LIGHT_DAY_SAFE_OVERRIDE is not None:
        threshold, stickiness_weight = LIGHT_DAY_SAFE_OVERRIDE
        # Compare scripts pin sticky days with a high threshold (e.g. 999) so
        # softness is always 1 for that day; otherwise resolve from peak SKUs.
        stickiness = resolve_build_anchored_stickiness(
            orders,
            ship_date,
            light_day_sku_threshold=threshold,
            light_day_stickiness=stickiness_weight,
            prelight_stickiness=False,
            pick_start_hour=pick_start_hour,
            final_shift_hour=final_shift_hour,
        )
    else:
        stickiness = resolve_build_anchored_stickiness(
            orders,
            ship_date,
            light_day_sku_threshold=light_day_sku_threshold,
            light_day_stickiness=light_day_stickiness,
            prelight_stickiness=prelight_stickiness,
            pick_start_hour=pick_start_hour,
            final_shift_hour=final_shift_hour,
        )

    rank_index = {item_id: index for index, item_id in enumerate(ranked_item_ids)}
    wave_groups: dict[datetime, list[str]] = defaultdict(list)
    for item_id in ranked_item_ids:
        wave_groups[primary_wave[item_id]].append(item_id)
    for wave_time in wave_groups:
        wave_groups[wave_time].sort(key=lambda item_id: rank_index[item_id])

    assignment: dict[str, str] = {}
    used_faces: set[str] = set()

    def _free_faces(flank: str | None = None) -> list[str]:
        pool = [
            face_id
            for face_id in faces_by_station
            if face_id not in used_faces
            and (flank is None or aisle_map[face_id].side.value == flank)
        ]
        pool.sort(
            key=lambda face_id: rank_face_for_travel(
                face_id,
                aisle_map=aisle_map,
                dual_picker=dual_picker_flag,
                center_station=center,
                donors_by_flank=donors_by_flank,
            )
        )
        return pool

    def _block(items: list[str], faces: list[str]) -> dict[str, str]:
        return _alternating_wave_assign(
            items,
            faces,
            order_layers=order_layers,
            item_layers=item_layers,
            aisle_map=aisle_map,
            default_anchor=center,
            max_iterations=max_iterations,
            stickiness=stickiness,
            stickiness_seed=stay_put_faces,
        )

    for wave_time in sorted(wave_groups):
        group = wave_groups[wave_time]
        if dual_picker_flag:
            n_left = (len(group) + 1) // 2
            left_items = group[:n_left]
            right_items = group[n_left:]
            left_faces = _free_faces("left")
            right_faces = _free_faces("right")
            left_assign = _block(
                left_items,
                left_faces[: len(left_items)] if len(left_faces) >= len(left_items) else left_faces,
            )
            right_assign = _block(
                right_items,
                right_faces[: len(right_items)] if len(right_faces) >= len(right_items) else right_faces,
            )
            assignment.update(left_assign)
            assignment.update(right_assign)
            used_faces.update(left_assign.values())
            used_faces.update(right_assign.values())
            continue

        free = _free_faces()
        block_faces = free[: len(group)] if len(free) >= len(group) else free
        block_assign = _block(group, block_faces)
        assignment.update(block_assign)
        used_faces.update(block_assign.values())

    if len(assignment) < len(ranked_item_ids):
        leftovers = [i for i in ranked_item_ids if i not in assignment]
        free_faces = _free_faces()
        for item_id, face_id in zip(leftovers, free_faces):
            assignment[item_id] = face_id
            used_faces.add(face_id)

    if len(assignment) < len(ranked_item_ids):
        leftovers = [i for i in ranked_item_ids if i not in assignment]
        for item_id, face_id in zip(leftovers, travel_fallback):
            if face_id not in used_faces:
                assignment[item_id] = face_id
                used_faces.add(face_id)

    return assignment
