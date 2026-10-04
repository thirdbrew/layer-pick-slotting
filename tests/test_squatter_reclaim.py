"""Squatter reclaim — evict the nearest zero-demand face, not only the preferred one.

The failure this fixes: in the afternoon most faces can sit at zero demand while
a large share of waved layers is classified ``hand_unstaged`` (no face, no
swap-ready supply). Cause:
``rule_aggressive_zero_demand_reclaim`` only ever inspects the incoming SKU's own
preferred face, so a stale occupant one station away is invisible to it.

Flag off = old behavior (demand-required reclaim only). Flag on = when
faces are in deficit, evict the nearest stale occupant to the waved SKU's home
station.
"""

from __future__ import annotations

from datetime import datetime

from slotting.models import (
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    RecommendationAction,
    RuleId,
    Slot,
    SlotType,
    WorldState,
)
from slotting.rules import RuleConfig, rule_aggressive_zero_demand_reclaim

AS_OF = datetime(2026, 3, 3, 14, 0)
FACES = ["S01", "S02", "S03", "S04"]


def _order(item_id: str, layers: int, *, hours: int) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=AS_OF.replace(hour=AS_OF.hour + hours),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def _world() -> WorldState:
    """4 faces, all occupied; S01..S03 stale, S04 still has demand. HOT wants a face."""
    slots = [
        Slot(slot_id=f, slot_type=SlotType.SOURCE, occupied_by_pallet_id=f"P-{f}")
        for f in FACES
    ]
    pallets = [
        Pallet(
            pallet_id=f"P-{f}",
            item_id=f"OLD{i}",
            layers_remaining=8 - i,  # S03 is the emptiest stale pallet
            current_slot=f,
            arrived_at=AS_OF,
        )
        for i, f in enumerate(FACES)
    ]
    orders = [
        _order("HOT", 20, hours=1),  # waved, no face, no supply on the floor
        _order("OLD3", 6, hours=2),  # S04's occupant still owes layers today
    ]
    # HOT's own supply must exist off-face or item_needs_face_add is moot.
    pallets.append(
        Pallet(pallet_id="P-HOT", item_id="HOT", layers_remaining=8, current_slot=None,
               arrived_at=AS_OF)
    )
    return WorldState(
        as_of=AS_OF, pallets=pallets, slots=slots, orders=orders, exclusions=[]
    )


def _config(**over) -> RuleConfig:
    return RuleConfig(
        disabled=frozenset(),
        preferred_source_face_by_item={"HOT": "S04"},  # blocked by a LIVE occupant
        source_face_station_by_id={f: i for i, f in enumerate(FACES)},
        slot_reclaim_rest_of_day=True,
    ).with_policy_flags(**over)


def test_preferred_face_only_reclaim_leaves_squatters_alone() -> None:
    """Flag off: HOT's preferred face has live demand, so nothing is freed."""
    recs = rule_aggressive_zero_demand_reclaim(_world(), _config())
    assert recs == []


def test_squatter_reclaim_evicts_nearest_stale_face() -> None:
    """Flag on: S03 (stale, adjacent to preferred S04) is reclaimed for HOT."""
    recs = rule_aggressive_zero_demand_reclaim(
        _world(), _config(zero_demand_any_face_reclaim=True)
    )
    assert [r.action for r in recs] == [RecommendationAction.FLAG_REMOVE]
    assert [r.rule for r in recs] == [RuleId.ZERO_DEMAND_RECLAIM]
    assert recs[0].pallet_id == "P-S03"  # nearest stale face to preferred S04


def test_squatter_reclaim_respects_step_budget() -> None:
    """Budget caps evictions even when the deficit is larger."""
    world = _world()
    world.orders.extend([_order(f"HOT{i}", 9, hours=1) for i in range(3)])
    world.pallets.extend(
        Pallet(pallet_id=f"P-HOT{i}", item_id=f"HOT{i}", layers_remaining=8,
               current_slot=None, arrived_at=AS_OF)
        for i in range(3)
    )
    recs = rule_aggressive_zero_demand_reclaim(
        world, _config(zero_demand_any_face_reclaim=True, zero_demand_any_face_max_per_step=1)
    )
    assert len(recs) == 1


if __name__ == "__main__":  # pragma: no cover - manual run
    test_preferred_face_only_reclaim_leaves_squatters_alone()
    test_squatter_reclaim_evicts_nearest_stale_face()
    test_squatter_reclaim_respects_step_budget()
    print("ok")
