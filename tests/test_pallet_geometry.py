"""Physical pallet height limits — build cap and clamp pick reach."""

from __future__ import annotations


from layout_advisor.planning.order_schedule import OrderLineDemand, max_build_layers_for_order
from slotting.models import Pallet
from slotting.pallet_geometry import (
    MAX_LP_PICK_HEIGHT_IN,
    MAX_ORDER_PALLET_HEIGHT_IN,
    build_pallet_height_in,
    face_lp_pickable_layers_for_item,
    lp_pickable_layers_on_pallet,
    max_build_layers_for_height,
    max_physical_pallet_layers_for_item,
    sku_layer_height_in,
    take_layers_for_build_pallet,
)

from conftest import AS_OF, minimal_state, slotted_pallet


def test_lp_pickable_layers_full_stack_within_reach() -> None:
    pallet = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=5,
        current_slot="S01",
        arrived_at=AS_OF,
    )
    # 5 × 9in = 45in ≤ 48in
    assert lp_pickable_layers_on_pallet(pallet, layer_height=9.0) == 5


def test_lp_pickable_layers_tall_stack_capped_at_48in() -> None:
    pallet = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=8,
        current_slot="S01",
        arrived_at=AS_OF,
    )
    # 8 × 9in = 72in; floor(48/9) = 5
    assert lp_pickable_layers_on_pallet(pallet, layer_height=9.0) == 5


def test_lp_pickable_layers_shrinks_as_stack_depletes() -> None:
    pallet = Pallet(
        pallet_id="P1",
        item_id="A",
        layers_remaining=3,
        current_slot="S01",
        arrived_at=AS_OF,
    )
    assert lp_pickable_layers_on_pallet(pallet, layer_height=9.0) == 3


def test_face_lp_pickable_layers_for_item() -> None:
    pallet, slot = slotted_pallet("P1", "SKU-A", layers=10)
    state = minimal_state(pallets=[pallet], slots=[slot])
    assert face_lp_pickable_layers_for_item(state, "SKU-A") == 5


def test_max_build_layers_for_height_single_sku() -> None:
    lines = (
        OrderLineDemand(
            order_id="o1",
            item_id="A",
            layers=20,
            due_at=AS_OF,
            layer_height_in=9.0,
        ),
    )
    # floor(100/9) = 11
    assert max_build_layers_for_height(lines, max_build_height_in=100.0) == 11


def test_max_build_layers_for_order_includes_height_cap() -> None:
    lines = (
        OrderLineDemand(
            order_id="o1",
            item_id="A",
            layers=20,
            due_at=AS_OF,
            layers_per_pallet=14,
            layer_height_in=9.0,
            layer_weight_lb=50.0,
        ),
    )
    cap = max_build_layers_for_order(
        lines,
        max_weight_lb=2500.0,
        max_build_height_in=MAX_ORDER_PALLET_HEIGHT_IN,
    )
    assert cap == 11


def test_constants() -> None:
    assert MAX_ORDER_PALLET_HEIGHT_IN == 100.0
    assert MAX_LP_PICK_HEIGHT_IN == 48.0
    assert sku_layer_height_in("missing-sku") == 9.0


def test_take_layers_mixed_sku_height_interleaves() -> None:
    from layout_advisor.planning.order_schedule import OrderLineDemand

    lines = (
        OrderLineDemand(
            order_id="o1",
            item_id="TALL",
            layers=6,
            due_at=AS_OF,
            layer_height_in=20.0,
            layer_weight_lb=50.0,
        ),
        OrderLineDemand(
            order_id="o1",
            item_id="SHORT",
            layers=6,
            due_at=AS_OF,
            layer_height_in=4.0,
            layer_weight_lb=10.0,
        ),
    )
    taken, placed = take_layers_for_build_pallet(
        lines,
        layer_cap=20,
        max_weight_lb=10_000.0,
        max_build_height_in=100.0,
    )
    assert placed == 9  # 5×20 + 4×4 interleaved before height fills
    assert build_pallet_height_in(lines, taken) <= 100.0


def test_oversized_single_layer_returns_zero_height_cap() -> None:
    from layout_advisor.planning.order_schedule import OrderLineDemand

    lines = (
        OrderLineDemand(
            order_id="o1",
            item_id="HUGE",
            layers=3,
            due_at=AS_OF,
            layer_height_in=120.0,
        ),
    )
    assert max_build_layers_for_height(lines, max_build_height_in=100.0) == 0


def test_max_physical_pallet_layers_for_item(monkeypatch) -> None:
    monkeypatch.setattr(
        "slotting.pallet_geometry.sku_layer_height_in",
        lambda item_id, **kw: 20.0,
    )
    assert max_physical_pallet_layers_for_item("X", max_height_in=100.0) == 5


def test_resolve_height_limits_uses_site_skus_when_present() -> None:
    from slotting.pallet_geometry import default_site_skus_path, resolve_height_limits

    site = default_site_skus_path()
    limits = resolve_height_limits()
    if site is not None:
        assert limits.skus_path == site
    else:
        assert limits.skus_path is None


def test_top_pick_model_makes_tall_stack_fully_pickable() -> None:
    """Top-pick clamp takes <=48in bites off the top - height caps the grab, not availability."""
    from slotting.pallet_geometry import PalletHeightLimits, pallet_height_limits

    tall = Pallet(
        pallet_id="P-TALL",
        item_id="A",
        layers_remaining=10,  # 10 x 9in default = 90in > 48in reach
        current_slot="S01",
        arrived_at=AS_OF,
    )
    # bottom-pick (library default): only the bottom 48//9 = 5 layers count
    assert lp_pickable_layers_on_pallet(tall) == 5
    with pallet_height_limits(PalletHeightLimits(top_pick_model=True)):
        assert lp_pickable_layers_on_pallet(tall) == 10
    # context restored
    assert lp_pickable_layers_on_pallet(tall) == 5
