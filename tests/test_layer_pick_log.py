"""Layer pick event logging."""

from __future__ import annotations

from datetime import datetime

from layout_advisor.planning.wave_actions import layer_pick_actions_from_events
from slotting.generate_slots import build_slots
from slotting.inventory import apply_hourly_layer_picks
from slotting.layer_pick_log import LayerPickEvent
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, SlotType, WorldState


def _waved_order(item_id: str, *, layers: int = 6) -> OrderLine:
    return OrderLine(
        order_id=f"O-{item_id}",
        item_id=item_id,
        qty_layers=layers,
        due_at=datetime(2026, 3, 3, 12, 0),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )


def test_layer_pick_logs_source_and_donor_build() -> None:
    slots = build_slots(source_count=2, donor_count=1)
    face = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    donor = next(s for s in slots if s.slot_type == SlotType.DESTINATION)
    face_pallet = Pallet(
        pallet_id="P1",
        item_id="SKU-A",
        layers_remaining=10,
        current_slot=face.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    build = Pallet(
        pallet_id=f"BUILD-{donor.slot_id}",
        item_id="SKU-A",
        layers_remaining=6,
        current_slot=donor.slot_id,
        arrived_at=datetime(2026, 3, 3, 8, 0),
    )
    donor.occupied_by_pallet_id = build.pallet_id
    face.occupied_by_pallet_id = face_pallet.pallet_id
    as_of = datetime(2026, 3, 3, 9, 0)
    state = WorldState(
        pallets=[face_pallet, build],
        slots=slots,
        orders=[_waved_order("SKU-A")],
        exclusions=[],
        as_of=as_of,
    )
    events: list[LayerPickEvent] = []
    picked = apply_hourly_layer_picks(
        [face_pallet, build],
        slots,
        state,
        layers_per_hour=200.0,
        pick_events=events,
    )
    assert picked == 5
    assert len(events) == 1
    assert events[0].from_slot == face.slot_id
    assert events[0].to_slot == donor.slot_id
    assert events[0].layers == 5


def test_layer_pick_actions_describe_route() -> None:
    event = LayerPickEvent(
        as_of=datetime(2026, 3, 3, 10, 0),
        item_id="SKU-A",
        layers=4,
        from_pallet_id="P1",
        from_slot="S01",
        to_pallet_id="BUILD-D01",
        to_slot="D01",
    )
    actions = layer_pick_actions_from_events(
        [event],
        ship_date=datetime(2026, 3, 3).date(),
        wave_send=datetime(2026, 3, 3, 10, 0),
    )
    assert len(actions) == 1
    assert actions[0].action == "LAYER_PICK_TO_BUILD"
    assert actions[0].role == "picker"
    assert actions[0].from_slot == "S01"
    assert actions[0].to_slot == "D01"
    assert "from face S01" in actions[0].detail
    assert "D01" in actions[0].detail
