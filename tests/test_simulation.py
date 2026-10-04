"""Tests for what-if simulation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet
from slotting.simulation import simulate_once, simulate_timeline

from conftest import AS_OF, load_sample_state, minimal_state, slotted_pallet


def test_simulation_improves_dead_slot_rate_on_sample() -> None:
    result = simulate_once(load_sample_state())
    assert result.baseline.dead_slot_rate > result.projected.dead_slot_rate
    assert result.dead_slot_rate_improvement_pct > 0
    assert result.removes_applied >= 2
    assert result.adds_applied >= 1


def test_simulation_improves_waved_readiness_on_sample() -> None:
    result = simulate_once(load_sample_state())
    # ITEM-ADD-WAVE staged into a slot; ITEM-WAIT still has no eligible supply
    assert result.waved_ready_projected >= result.waved_ready_baseline
    assert result.adds_applied >= 1
    assert result.projected.shortage_count == result.baseline.shortage_count


def test_hold_pallet_not_removed_in_simulation() -> None:
    result = simulate_once(load_sample_state())
    p004_still_slotted = result.projected_state.pallet_by_id("P004")
    assert p004_still_slotted is not None
    assert p004_still_slotted.current_slot == "S04"


def test_apply_remove_clears_slot() -> None:
    pallet, slot = slotted_pallet("P1", "ITEM-A", layers=2)
    state = minimal_state(pallets=[pallet], slots=[slot])
    result = simulate_once(state)
    projected_pallet = result.projected_state.pallet_by_id("P1")
    assert projected_pallet is not None
    assert projected_pallet.current_slot is None
    projected_slot = next(s for s in result.projected_state.slots if s.slot_id == "S01")
    assert projected_slot.occupied_by_pallet_id is None


def test_apply_add_uses_freed_slot() -> None:
    dead, dead_slot = slotted_pallet("P-DEAD", "ITEM-OLD", layers=1, slot_id="S01")
    fresh = Pallet(
        pallet_id="P-NEW",
        item_id="ITEM-NEW",
        layers_remaining=6,
        current_slot=None,
        arrived_at=AS_OF,
    )
    slots = [
        dead_slot,
        *[s for s in minimal_state().slots if s.slot_id != "S01"],
    ]
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-NEW",
        qty_layers=2,
        due_at=AS_OF + timedelta(hours=2),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[dead, fresh], slots=slots, orders=[order])
    result = simulate_once(state)
    assert result.removes_applied >= 1
    assert result.adds_applied == 1
    added = result.projected_state.pallet_by_id("P-NEW")
    assert added is not None
    assert added.current_slot == "S01"


def test_timeline_produces_multiple_results() -> None:
    state = load_sample_state()
    results = simulate_timeline(
        state,
        start=AS_OF,
        end=AS_OF + timedelta(hours=3),
        step=timedelta(hours=1),
    )
    assert len(results) == 4
    assert all(r.as_of >= AS_OF for r in results)


def test_blocked_add_when_no_empty_slots(tmp_path: Path) -> None:
    """All source slots full; removes may free some but adds compete for slots."""
    pallets = []
    slots = []
    for i in range(1, 4):
        p, s = slotted_pallet(f"P{i}", f"ITEM-{i}", layers=1, slot_id=f"S{i:02d}")
        pallets.append(p)
        slots.append(s)
    slots.extend([s for s in minimal_state().slots if int(s.slot_id[1:]) > 3])

    unslotted = Pallet(
        pallet_id="P-NEW",
        item_id="ITEM-NEED",
        layers_remaining=5,
        current_slot=None,
        arrived_at=AS_OF,
    )
    pallets.append(unslotted)
    order = OrderLine(
        order_id="O1",
        item_id="ITEM-NEED",
        qty_layers=1,
        due_at=AS_OF + timedelta(hours=2),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=pallets, slots=slots, orders=[order])
    result = simulate_once(state)
    assert result.removes_applied == 3
    assert result.adds_applied == 1
