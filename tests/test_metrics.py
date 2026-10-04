"""Tests for KPI computation and metrics logging."""

from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

from slotting.generate_slots import build_slots
from slotting.metrics import append_metrics_log, compute_metrics
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, SlotType
from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits
from slotting.throughput import item_has_immediate_lp_supply

from conftest import AS_OF, load_sample_state, minimal_state


def test_sample_metrics() -> None:
    state = load_sample_state()
    metrics = compute_metrics(state)
    assert metrics.source_slots_total == 10
    assert metrics.source_slots_occupied == 7
    assert metrics.dead_slots >= 3  # P001, P002, P003 at minimum
    assert 0 < metrics.dead_slot_rate <= 1.0
    assert metrics.shortage_count >= 1


def test_shortage_count_for_waved_item_without_supply() -> None:
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-MISSING",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(orders=orders)
    metrics = compute_metrics(state)
    assert metrics.shortage_count == 1


def test_dead_slot_includes_zero_pickable_height_blocked_face() -> None:
    """Occupied face with demand but 0 LP-pickable layers counts as dead."""
    from slotting.models import Slot

    face = Slot(
        slot_id="S01",
        slot_type=SlotType.SOURCE,
        occupied_by_pallet_id="P-TALL",
    )
    tall = Pallet(
        pallet_id="P-TALL",
        item_id="ITEM-TALL",
        layers_remaining=5,
        current_slot="S01",
        arrived_at=AS_OF,
    )
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-TALL",
            qty_layers=2,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[tall], slots=[face], orders=orders)
    # Default layer height 9″; pick zone below one layer → 0 pickable.
    with pallet_height_limits(
        PalletHeightLimits(
            max_order_pallet_height_in=100.0,
            max_lp_pick_height_in=8.0,
        )
    ):
        metrics = compute_metrics(state)
    assert metrics.dead_slots == 1
    assert metrics.source_slots_occupied == 1


def test_append_metrics_log_creates_and_appends(tmp_path: Path) -> None:
    state = load_sample_state()
    metrics = compute_metrics(state)
    log_path = tmp_path / "metrics_log.csv"
    append_metrics_log(metrics, log_path)
    append_metrics_log(metrics, log_path)
    with log_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert "dead_slot_rate_pct" in rows[0]


def test_shortage_count_ignores_depleted_face_pallet() -> None:
    slots = build_slots(source_count=2, donor_count=1)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face.occupied_by_pallet_id = "P1"
    pallet = Pallet(
        pallet_id="P1",
        item_id="ITEM-MISSING",
        layers_remaining=0,
        current_slot=face.slot_id,
        arrived_at=AS_OF,
    )
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-MISSING",
            qty_layers=1,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[pallet], slots=slots, orders=orders)
    assert not state.item_in_source_slot("ITEM-MISSING")
    assert compute_metrics(state).shortage_count == 1


def test_shortage_cleared_by_swap_ready_pickable_backup() -> None:
    from slotting.models import Slot

    backup = Pallet(
        pallet_id="BACK",
        item_id="ITEM-MISSING",
        layers_remaining=6,
        current_slot=None,
        arrived_at=AS_OF,
    )
    slots = [
        Slot(
            slot_id="S01",
            slot_type=SlotType.SOURCE,
            occupied_by_pallet_id=None,
            backup_pallet_id="BACK",
        )
    ]
    orders = [
        OrderLine(
            order_id="O1",
            item_id="ITEM-MISSING",
            qty_layers=2,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[backup], slots=slots, orders=orders)
    assert item_has_immediate_lp_supply(state, "ITEM-MISSING")
    assert compute_metrics(state).shortage_count == 0
