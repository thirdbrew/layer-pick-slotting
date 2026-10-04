"""Completed donor build exit to wrapping."""

from __future__ import annotations

from datetime import datetime

from slotting.donor_staging import destination_slots_available, rule_dest_stage_completed_build
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


AS_OF = datetime(2026, 3, 3, 8, 0)


def _state_with_completed_build() -> WorldState:
    return WorldState(
        pallets=[
            Pallet(
                pallet_id="BUILD-D01",
                item_id="SKU-A",
                layers_remaining=20,
                current_slot="D01",
                arrived_at=AS_OF,
            )
        ],
        slots=[
            Slot(slot_id="D01", slot_type=SlotType.DESTINATION, occupied_by_pallet_id="BUILD-D01"),
        ],
        orders=[
            OrderLine(
                order_id="1",
                item_id="SKU-B",
                qty_layers=10,
                due_at=datetime(2026, 3, 3, 12, 0),
                area=OrderArea.THIS_LINE,
                status=OrderStatus.WAVED,
            )
        ],
        exclusions=[],
        as_of=AS_OF,
    )


def test_completed_build_exits_to_wrapping_not_rack() -> None:
    recs = rule_dest_stage_completed_build(_state_with_completed_build(), include_rack_aisle=True)
    assert len(recs) == 1
    assert recs[0].action == RecommendationAction.FLAG_REMOVE
    assert recs[0].rule == RuleId.DEST_STAGE_COMPLETE


def test_completed_build_frees_donor_face_without_rack_space() -> None:
    state = _state_with_completed_build()
    available = destination_slots_available(state, include_rack_aisle=True)
    assert "D01" in available
