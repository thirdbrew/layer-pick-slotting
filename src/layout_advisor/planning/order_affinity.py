"""Order-frequency and wave co-occurrence affinity for travel-aware slotting."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from slotting.models import OrderArea, OrderLine
from slotting.wave_release import order_ship_date, split_day_by_native_due

if TYPE_CHECKING:
    from layout_advisor.planning.order_schedule import OrderScheduleConfig


def ship_day_order_frequency(
    orders: list[OrderLine],
    ship_date: date,
) -> dict[str, int]:
    """Distinct order lines per SKU on *ship_date* (repeat picks → prime faces)."""
    freq: dict[str, int] = defaultdict(int)
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        freq[order.item_id] += 1
    return dict(freq)


def true_order_key(order_id: str) -> str:
    """Customer-order grouping key from an order-line id.

    The order feed carries one row per LINE with ids shaped
    ``{order_hash}-{sku}-{yyyymmdd}`` — the hash prefix identifies the physical
    customer order. Multi-pallet splits within an order are handled by
    :func:`build_pallet_cooccurrence_weights`.
    Ids without that shape (short first token) fall back to the full id.
    """
    head = order_id.split("-", 1)[0]
    return head if len(head) >= 8 else order_id


def order_cooccurrence_weights(
    orders: list[OrderLine],
    ship_date: date,
) -> dict[frozenset[str], int]:
    """
    Pair weights for SKUs appearing in the SAME CUSTOMER ORDER on *ship_date*.

    Stronger than same-wave co-occurrence, but weaker than same *build pallet*
    (large orders split across multiple outbound pallets — see
    :func:`build_pallet_cooccurrence_weights`).
    """
    by_order: dict[str, set[str]] = defaultdict(set)
    for order in orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if order_ship_date(order) != ship_date:
            continue
        by_order[true_order_key(order.order_id)].add(order.item_id)
    weights: dict[frozenset[str], int] = defaultdict(int)
    for items in by_order.values():
        if len(items) < 2:
            continue
        for left, right in combinations(sorted(items), 2):
            weights[frozenset((left, right))] += 1
    return dict(weights)


def build_pallet_cooccurrence_weights(
    orders: list[OrderLine],
    ship_date: date,
    *,
    config: OrderScheduleConfig | None = None,
    skus_path: Path | None = None,
) -> dict[frozenset[str], int]:
    """
    Pair weights for SKUs sharing the SAME outbound build pallet on *ship_date*.

    Uses the order-schedule multi-pallet split (100in / 2500 lb caps). The picker
    shuttles between one build pad and that pallet's SKU faces — those faces are
    the strongest adjacency signal for travel.
    """
    from layout_advisor.planning.order_schedule import build_order_schedule

    weights: dict[frozenset[str], int] = defaultdict(int)
    for work in build_order_schedule(
        orders, ship_date, config=config, skus_path=skus_path
    ):
        for pallet in work.build_pallets:
            items = sorted(
                {line.item_id for line in pallet.lines if line.layers > 0}
            )
            if len(items) < 2:
                continue
            for left, right in combinations(items, 2):
                weights[frozenset((left, right))] += 1
    return dict(weights)


def build_pallet_layer_maps(
    orders: list[OrderLine],
    ship_date: date,
    *,
    config: OrderScheduleConfig | None = None,
    skus_path: Path | None = None,
) -> dict[str, dict[str, int]]:
    """
    Map ``{order_key}:{pallet_index}`` → ``{item_id: layers}`` for travel KPIs.

    Compatible with :func:`layout_advisor.planning.build_anchored_slotting.total_order_travel`
    (each build pallet is treated as its own travel unit with its own median).
    """
    from layout_advisor.planning.order_schedule import build_order_schedule

    maps: dict[str, dict[str, int]] = {}
    for work in build_order_schedule(
        orders, ship_date, config=config, skus_path=skus_path
    ):
        for pallet in work.build_pallets:
            layers = {
                line.item_id: line.layers
                for line in pallet.lines
                if line.layers > 0
            }
            if not layers:
                continue
            maps[f"{work.order_key}:{pallet.pallet_index}"] = layers
    return maps


def combined_affinity_weights(
    orders: list[OrderLine],
    ship_date: date,
    *,
    due_offset_hours: float = 4.0,
    order_weight_scale: int = 10,
    build_pallet_weight_scale: int = 20,
    skus_path: Path | None = None,
    config: OrderScheduleConfig | None = None,
) -> dict[frozenset[str], int]:
    """
    Blend affinity signals for face adjacency (strongest → weakest):

    1. Same *build pallet* (multi-pallet split) — ``build_pallet_weight_scale``
    2. Same customer order — ``order_weight_scale``
    3. Same wave bucket — unscaled
    """
    combined: dict[frozenset[str], int] = dict(
        wave_cooccurrence_weights(orders, ship_date, due_offset_hours=due_offset_hours)
    )
    for pair, weight in order_cooccurrence_weights(orders, ship_date).items():
        combined[pair] = combined.get(pair, 0) + order_weight_scale * weight
    for pair, weight in build_pallet_cooccurrence_weights(
        orders, ship_date, config=config, skus_path=skus_path
    ).items():
        combined[pair] = combined.get(pair, 0) + build_pallet_weight_scale * weight
    return combined


def wave_cooccurrence_weights(
    orders: list[OrderLine],
    ship_date: date,
    *,
    due_offset_hours: float = 4.0,
) -> dict[frozenset[str], int]:
    """
    Pair weights for SKUs picked in the same native wave on *ship_date*.

    Complements (frequently waved together) should land on adjacent aisle stations.
    """
    day_orders = [
        o
        for o in orders
        if o.area == OrderArea.THIS_LINE and order_ship_date(o) == ship_date
    ]
    buckets = split_day_by_native_due(
        day_orders,
        ship_date=ship_date,
        due_offset_hours=due_offset_hours,
    )
    weights: dict[frozenset[str], int] = defaultdict(int)
    for bucket in buckets:
        items = sorted({o.item_id for o in bucket.orders})
        for left, right in combinations(items, 2):
            weights[frozenset((left, right))] += 1
    return dict(weights)


def pair_weight(
    item_a: str,
    item_b: str,
    weights: dict[frozenset[str], int],
) -> int:
    if item_a == item_b:
        return 0
    return weights.get(frozenset((item_a, item_b)), 0)


def affinity_chain_order(
    items: list[str],
    *,
    order_freq: dict[str, int],
    pair_weights: dict[frozenset[str], int],
    rank_index: dict[str, int],
    seed_by_rank: bool = False,
) -> list[str]:
    """
    Order SKUs along an aisle so complements sit next to each other.

    Greedy path: seed one SKU, then repeatedly append the remaining SKU with
    strongest co-occurrence to the chain. Default seed is highest order
    frequency; *seed_by_rank* uses setup rank instead.
    """
    if len(items) <= 1:
        return list(items)
    remaining = set(items)
    if seed_by_rank:
        seed = min(
            remaining,
            key=lambda item_id: (
                rank_index.get(item_id, 10_000),
                -order_freq.get(item_id, 0),
                item_id,
            ),
        )
    else:
        seed = min(
            remaining,
            key=lambda item_id: (
                -order_freq.get(item_id, 0),
                rank_index.get(item_id, 10_000),
                item_id,
            ),
        )
    chain = [seed]
    remaining.remove(seed)
    while remaining:
        best = max(
            remaining,
            key=lambda item_id: (
                sum(pair_weight(item_id, other, pair_weights) for other in chain),
                order_freq.get(item_id, 0),
                -rank_index.get(item_id, 10_000),
                item_id,
            ),
        )
        chain.append(best)
        remaining.remove(best)
    return chain
