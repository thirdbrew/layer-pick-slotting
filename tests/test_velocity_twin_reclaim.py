"""Demand-horizon reclaim of velocity twin faces."""

from __future__ import annotations

from datetime import datetime, timedelta

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
from slotting.rules import RuleConfig, rule_velocity_twin_face_reclaim

AS_OF = datetime(2026, 3, 3, 10, 0)


def _waved(item_id: str, *, layers: int = 8, hours: float = 2.0) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=AS_OF + timedelta(hours=hours),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def _state(
    *,
    face_pallets: list[tuple[str, str, int]],
    orders: list[OrderLine],
    empty_faces: int = 0,
    source_slots: int | None = None,
) -> WorldState:
    """face_pallets: (pallet_id, item_id, layers) on S01, S02, ..."""
    total_slots = source_slots or (len(face_pallets) + empty_faces)
    slots: list[Slot] = []
    pallets: list[Pallet] = []
    for i, (pid, item, layers) in enumerate(face_pallets, start=1):
        sid = f"S{i:02d}"
        slots.append(
            Slot(slot_id=sid, slot_type=SlotType.SOURCE, occupied_by_pallet_id=pid)
        )
        pallets.append(
            Pallet(
                pallet_id=pid,
                item_id=item,
                layers_remaining=layers,
                current_slot=sid,
                arrived_at=AS_OF - timedelta(hours=12),
            )
        )
    for j in range(total_slots - len(face_pallets)):
        slots.append(
            Slot(
                slot_id=f"S{len(face_pallets) + j + 1:02d}",
                slot_type=SlotType.SOURCE,
                occupied_by_pallet_id=None,
            )
        )
    return WorldState(
        as_of=AS_OF, pallets=pallets, slots=slots, orders=orders, exclusions=[]
    )


def test_reclaim_when_preferred_face_collision() -> None:
    """Twin slot is preferred face for a horizon waiter that needs ADD."""
    state = _state(
        face_pallets=[
            ("P-HOT-1", "HOT", 8),  # primary
            ("P-HOT-2", "HOT", 8),  # twin on S02
        ],
        orders=[
            _waved("HOT", layers=20, hours=6),
            _waved("WAIT", layers=8, hours=2),
        ],
        empty_faces=0,
    )
    cfg = RuleConfig(
        velocity_twin_face_reclaim=True,
        velocity_twin_reclaim_horizon_hours=4,
        twin_source_face_by_item={"HOT": "S02"},
        preferred_source_face_by_item={"HOT": "S01", "WAIT": "S02"},
    )
    recs = rule_velocity_twin_face_reclaim(state, cfg)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_REMOVE
    assert recs[0].pallet_id == "P-HOT-2"
    assert recs[0].rule == RuleId.VELOCITY_TWIN_RECLAIM


def test_reclaim_on_empty_face_deficit() -> None:
    """No empty faces + waved needer → free least-needed twin."""
    state = _state(
        face_pallets=[
            ("P-HOT-1", "HOT", 8),
            ("P-HOT-2", "HOT", 8),
        ],
        orders=[
            _waved("HOT", layers=12, hours=8),
            _waved("NEED", layers=8, hours=1),
        ],
        empty_faces=0,
    )
    cfg = RuleConfig(
        velocity_twin_face_reclaim=True,
        twin_source_face_by_item={"HOT": "S02"},
        preferred_source_face_by_item={"HOT": "S01", "NEED": "S99"},
    )
    recs = rule_velocity_twin_face_reclaim(state, cfg)
    assert len(recs) == 1
    assert recs[0].pallet_id == "P-HOT-2"


def test_no_reclaim_when_disabled_or_no_pressure() -> None:
    state = _state(
        face_pallets=[
            ("P-HOT-1", "HOT", 8),
            ("P-HOT-2", "HOT", 8),
        ],
        orders=[_waved("HOT", layers=20, hours=6)],
        empty_faces=2,
    )
    off = RuleConfig(
        velocity_twin_face_reclaim=False,
        twin_source_face_by_item={"HOT": "S02"},
        preferred_source_face_by_item={"HOT": "S01"},
    )
    assert rule_velocity_twin_face_reclaim(state, off) == []

    on = RuleConfig(
        velocity_twin_face_reclaim=True,
        twin_source_face_by_item={"HOT": "S02"},
        preferred_source_face_by_item={"HOT": "S01"},
    )
    # Hot SKU still needs two faces; empty faces available; no waiters → keep twin.
    assert rule_velocity_twin_face_reclaim(state, on) == []


def test_surplus_twin_reclaimed_when_face_needers_exist() -> None:
    """Rest-of-day demand fits one face → drop twin when others need faces."""
    state = _state(
        face_pallets=[
            ("P-HOT-1", "HOT", 8),
            ("P-HOT-2", "HOT", 8),
        ],
        orders=[
            _waved("HOT", layers=4, hours=3),  # fits primary alone
            _waved("NEED", layers=8, hours=1),
        ],
        empty_faces=0,
    )
    cfg = RuleConfig(
        velocity_twin_face_reclaim=True,
        twin_source_face_by_item={"HOT": "S02"},
        preferred_source_face_by_item={"HOT": "S01", "NEED": "S03"},
    )
    recs = rule_velocity_twin_face_reclaim(state, cfg)
    assert len(recs) == 1
    assert recs[0].pallet_id == "P-HOT-2"


def test_horizon_surplus_reclaims_after_peak_window() -> None:
    """Deficit reclaim prefers twins whose near-term demand fits primary only."""
    state = _state(
        face_pallets=[
            ("P-HOT-1", "HOT", 8),
            ("P-HOT-2", "HOT", 8),
        ],
        orders=[
            _waved("HOT", layers=20, hours=10),  # heavy rest-of-day
            _waved("HOT", layers=4, hours=2),  # light near-term
            _waved("NEED", layers=8, hours=1),
        ],
        empty_faces=0,
    )
    cfg = RuleConfig(
        velocity_twin_face_reclaim=True,
        velocity_twin_reclaim_horizon_hours=4,
        twin_source_face_by_item={"HOT": "S02"},
        preferred_source_face_by_item={"HOT": "S01", "NEED": "S03"},
    )
    recs = rule_velocity_twin_face_reclaim(state, cfg)
    assert len(recs) == 1
    assert recs[0].pallet_id == "P-HOT-2"
    assert "horizon face needers" in recs[0].reason


def test_no_surplus_reclaim_when_primary_pickable_below_demand() -> None:
    """8 layers on face but only 5 LP-pickable at 48in — keep twin for horizon demand."""
    from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits

    state = _state(
        face_pallets=[
            ("P-HOT-1", "HOT", 8),
            ("P-HOT-2", "HOT", 8),
        ],
        orders=[
            _waved("HOT", layers=6, hours=2),  # exceeds 5 pickable on primary alone
        ],
        empty_faces=2,
    )
    cfg = RuleConfig(
        velocity_twin_face_reclaim=True,
        velocity_twin_reclaim_horizon_hours=4,
        twin_source_face_by_item={"HOT": "S02"},
        preferred_source_face_by_item={"HOT": "S01"},
    )
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=48.0)):
        recs = rule_velocity_twin_face_reclaim(state, cfg)
    assert recs == []


def test_deficit_frees_multiple_least_needed_twins() -> None:
    state = _state(
        face_pallets=[
            ("P-A-1", "A", 8),
            ("P-A-2", "A", 8),
            ("P-B-1", "B", 8),
            ("P-B-2", "B", 8),
        ],
        orders=[
            _waved("A", layers=20, hours=8),
            _waved("B", layers=20, hours=8),
            _waved("N1", layers=8, hours=1),
            _waved("N2", layers=8, hours=1),
        ],
        source_slots=4,
    )
    cfg = RuleConfig(
        velocity_twin_face_reclaim=True,
        twin_source_face_by_item={"A": "S02", "B": "S04"},
        preferred_source_face_by_item={
            "A": "S01",
            "B": "S03",
            "N1": "S05",
            "N2": "S06",
        },
    )
    recs = rule_velocity_twin_face_reclaim(state, cfg)
    assert len(recs) == 2
    assert {r.pallet_id for r in recs} == {"P-A-2", "P-B-2"}
