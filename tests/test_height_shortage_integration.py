"""Integration tests — height limits vs throughput / order-schedule pick demand."""

from __future__ import annotations

from datetime import timedelta

from layout_advisor.planning.order_schedule import OrderScheduleConfig, pick_demand_respecting_order_waits
from slotting.generate_slots import build_slots
from slotting.models import OrderArea, OrderLine, OrderStatus, Pallet, SlotType
from slotting.throughput import _classify_demand, item_pick_shortage_layers

from conftest import AS_OF, minimal_state


def test_item_pick_shortage_counts_height_blocked_layers() -> None:
    slots = build_slots(source_count=2, donor_count=1)
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    donor = next(s for s in slots if s.slot_type == SlotType.DESTINATION)
    face = Pallet(
        pallet_id="P1",
        item_id="SKU-A",
        layers_remaining=8,
        current_slot=source.slot_id,
        arrived_at=AS_OF,
    )
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="SKU-A",
        layers_remaining=1,
        current_slot=donor.slot_id,
        arrived_at=AS_OF,
    )
    source.occupied_by_pallet_id = face.pallet_id
    donor.occupied_by_pallet_id = build.pallet_id
    order = OrderLine(
        order_id="O1",
        item_id="SKU-A",
        qty_layers=8,
        due_at=AS_OF + timedelta(hours=1),
        status=OrderStatus.WAVED,
        area=OrderArea.THIS_LINE,
    )
    state = minimal_state(pallets=[face, build], slots=slots, orders=[order], as_of=AS_OF)
    assert item_pick_shortage_layers(state, "SKU-A", 8) == 3


def test_pick_demand_caps_at_pickable_face_without_monkeypatch() -> None:
    slots = build_slots(source_count=2, donor_count=1)
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    donor = next(s for s in slots if s.slot_type == SlotType.DESTINATION)
    face = Pallet(
        pallet_id="P1",
        item_id="SKU-A",
        layers_remaining=10,
        current_slot=source.slot_id,
        arrived_at=AS_OF,
    )
    build = Pallet(
        pallet_id="BUILD-D01",
        item_id="SKU-A",
        layers_remaining=1,
        current_slot=donor.slot_id,
        arrived_at=AS_OF,
    )
    source.occupied_by_pallet_id = face.pallet_id
    donor.occupied_by_pallet_id = build.pallet_id
    due = AS_OF + timedelta(hours=4)
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=8,
            due_at=due,
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(
        pallets=[face, build],
        slots=slots,
        orders=orders,
        as_of=AS_OF,
    )
    demand = pick_demand_respecting_order_waits(
        state,
        orders,
        AS_OF.date(),
        config=OrderScheduleConfig(order_lookahead_hours=24, order_whole_order_shortage_wait=False),
    )
    assert demand.get("SKU-A", 0) == 5


def test_pick_demand_releases_when_supply_on_face_but_no_build_open() -> None:
    """Supply on the face releases the order even before its build pad is open.

    The build pad is opened *as part of starting the order* (see
    ``apply_order_donor_opens_at_step``), so a release gate that demands the build
    already exist can never be satisfied — the same deadlock ``donor_builds.py``
    already avoids for the pad-open gate. Regression for a fulfilment defect
    where almost none of a stress day's layers were picked.
    """
    slots = build_slots(source_count=2, donor_count=1)
    source = next(s for s in slots if s.slot_type == SlotType.SOURCE)
    face = Pallet(
        pallet_id="P1",
        item_id="SKU-A",
        layers_remaining=10,
        current_slot=source.slot_id,
        arrived_at=AS_OF,
    )
    source.occupied_by_pallet_id = face.pallet_id
    # No pallet in the donor slot: nothing is building SKU-A yet.
    # Order 5 layers, not 8: only 5 of the face's 10 are within pick reach (see
    # test_pick_demand_caps_at_pickable_face_without_monkeypatch), and height
    # blocking would otherwise mask the build-face effect this test is isolating.
    orders = [
        OrderLine(
            order_id="O1",
            item_id="SKU-A",
            qty_layers=5,
            due_at=AS_OF + timedelta(hours=4),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(pallets=[face], slots=slots, orders=orders, as_of=AS_OF)
    demand = pick_demand_respecting_order_waits(
        state,
        orders,
        AS_OF.date(),
        config=OrderScheduleConfig(
            order_lookahead_hours=24, order_whole_order_shortage_wait=True
        ),
    )
    assert demand.get("SKU-A", 0) == 5


def test_via_swap_classifies_height_blocked_backup_supply() -> None:
    """Backup-row supply via swap must respect 48in pick reach, not count all layers LP-ready."""
    from slotting.models import Slot, SlotType
    from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits

    backup = Pallet(
        pallet_id="BACK",
        item_id="TALL",
        layers_remaining=8,
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
            item_id="TALL",
            qty_layers=8,
            due_at=AS_OF + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        )
    ]
    state = minimal_state(pallets=[backup], slots=slots, orders=orders, as_of=AS_OF)
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=48.0)):
        metrics = _classify_demand(state, {"TALL": 8})
        assert metrics.layer_pick_ready_layers == 5
        assert metrics.low_layer_slotted_layers == 3
        assert metrics.hand_pick_shortage_layers == 0


def test_late_rack_promote_sort_key_uses_pickable_face_depth() -> None:
    """Face sort component is LP-pickable depth, not total layers on the face."""
    from slotting.inventory_policy import late_rack_promote_sort_key
    from slotting.models import Slot, SlotType
    from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits

    afternoon = AS_OF.replace(hour=15, minute=0)
    tall = Pallet(
        pallet_id="P-TALL",
        item_id="TALL",
        layers_remaining=8,
        current_slot="S01",
        arrived_at=AS_OF,
    )
    slots = [
        Slot(slot_id="S01", slot_type=SlotType.SOURCE, occupied_by_pallet_id="P-TALL"),
    ]
    orders = [
        OrderLine(
            order_id="O-T",
            item_id="TALL",
            qty_layers=4,
            due_at=afternoon + timedelta(hours=2),
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(pallets=[tall], slots=slots, orders=orders, as_of=afternoon)
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=9999.0)):
        unlimited = late_rack_promote_sort_key(state, "TALL", late_shift_hour=18)
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=48.0)):
        capped = late_rack_promote_sort_key(state, "TALL", late_shift_hour=18)
    # Afternoon key: (-near, -late, face, ...) — face is index 2.
    assert unlimited[2] == 8
    assert capped[2] == 5
    assert capped < unlimited  # lower pickable face → higher promote priority


def test_resupply_priority_key_prefers_lower_pickable_face() -> None:
    from slotting.inventory_policy import resupply_priority_key
    from slotting.models import Slot, SlotType
    from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits

    tall = Pallet(
        pallet_id="P-TALL",
        item_id="TALL",
        layers_remaining=8,
        current_slot="S01",
        arrived_at=AS_OF,
    )
    deep = Pallet(
        pallet_id="P-DEEP",
        item_id="DEEP",
        layers_remaining=3,
        current_slot="S02",
        arrived_at=AS_OF,
    )
    slots = [
        Slot(slot_id="S01", slot_type=SlotType.SOURCE, occupied_by_pallet_id="P-TALL"),
        Slot(slot_id="S02", slot_type=SlotType.SOURCE, occupied_by_pallet_id="P-DEEP"),
    ]
    state = minimal_state(pallets=[tall, deep], slots=slots, orders=[], as_of=AS_OF)
    demand = {"TALL": 4, "DEEP": 4}
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=48.0)):
        tall_key = resupply_priority_key(state, "TALL", waved_layers=demand)
        deep_key = resupply_priority_key(state, "DEEP", waved_layers=demand)
    # TALL: 5 pickable; DEEP: 3 pickable → DEEP sorts first (lower face).
    assert deep_key < tall_key
    assert tall_key[2] == 5
    assert deep_key[2] == 3


def test_whole_order_blocks_on_height_blocked_line() -> None:
    """HEIGHT_BLOCKED on one line waits the whole order under whole_order policy."""
    from datetime import datetime

    from slotting.models import Slot, SlotType
    from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits

    as_of = datetime(2026, 3, 3, 8, 0)
    due = as_of + timedelta(hours=4)
    tall_face = Pallet(
        pallet_id="P-TALL",
        item_id="TALL",
        layers_remaining=8,
        current_slot="S01",
        arrived_at=as_of,
    )
    ok_face = Pallet(
        pallet_id="P-OK",
        item_id="OK",
        layers_remaining=5,
        current_slot="S02",
        arrived_at=as_of,
    )
    tall_build = Pallet(
        pallet_id="BUILD-TALL",
        item_id="TALL",
        layers_remaining=1,
        current_slot="D01",
        arrived_at=as_of,
    )
    ok_build = Pallet(
        pallet_id="BUILD-OK",
        item_id="OK",
        layers_remaining=1,
        current_slot="D02",
        arrived_at=as_of,
    )
    slots = [
        Slot(slot_id="S01", slot_type=SlotType.SOURCE, occupied_by_pallet_id="P-TALL"),
        Slot(slot_id="S02", slot_type=SlotType.SOURCE, occupied_by_pallet_id="P-OK"),
        Slot(slot_id="D01", slot_type=SlotType.DESTINATION, occupied_by_pallet_id="BUILD-TALL"),
        Slot(slot_id="D02", slot_type=SlotType.DESTINATION, occupied_by_pallet_id="BUILD-OK"),
    ]
    oid = "abcd1234efgh5678"
    orders = [
        OrderLine(
            order_id=f"{oid}-TALL-20260303",
            item_id="TALL",
            qty_layers=8,
            due_at=due,
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
        OrderLine(
            order_id=f"{oid}-OK-20260303",
            item_id="OK",
            qty_layers=3,
            due_at=due,
            status=OrderStatus.WAVED,
            area=OrderArea.THIS_LINE,
        ),
    ]
    state = minimal_state(
        pallets=[tall_face, ok_face, tall_build, ok_build],
        slots=slots,
        orders=orders,
        as_of=as_of,
    )
    with pallet_height_limits(PalletHeightLimits(max_lp_pick_height_in=48.0)):
        whole = pick_demand_respecting_order_waits(
            state,
            orders,
            as_of.date(),
            config=OrderScheduleConfig(
                order_lookahead_hours=24,
                order_whole_order_shortage_wait=True,
                order_shortage_policy="whole_order",
            ),
        )
        partial = pick_demand_respecting_order_waits(
            state,
            orders,
            as_of.date(),
            config=OrderScheduleConfig(
                order_lookahead_hours=24,
                order_whole_order_shortage_wait=False,
                order_shortage_policy="partial_line",
            ),
        )
    assert whole == {}  # entire order waits on height-blocked TALL line
    assert partial.get("OK", 0) == 3
    assert partial.get("TALL", 0) == 5  # pickable only
