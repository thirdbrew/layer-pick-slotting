"""Order-aware colors and labels for LP route visualization."""

from __future__ import annotations

import hashlib

from layout_advisor.planning.order_affinity import true_order_key
from slotting.models import OrderArea, OrderStatus, WorldState
from slotting.rules import WAVED_HORIZON

# Distinct hues for customer orders on the floor (readable on white / green pads).
ORDER_ROUTE_COLORS: tuple[str, ...] = (
    "#1565C0",
    "#C62828",
    "#6A1B9A",
    "#EF6C00",
    "#00838F",
    "#AD1457",
    "#558B2F",
    "#4527A0",
    "#D84315",
    "#0277BD",
    "#7B1FA2",
    "#F9A825",
    "#00695C",
    "#5D4037",
    "#283593",
    "#C2185B",
)

DONOR_SLOT_COLORS: tuple[str, ...] = (
    "#0D47A1",
    "#B71C1C",
    "#4A148C",
    "#E65100",
    "#006064",
    "#880E4F",
    "#33691E",
    "#311B92",
    "#BF360C",
    "#01579B",
)


def short_order_key(order_key: str | None, *, width: int = 8) -> str:
    if not order_key:
        return "-"
    return order_key[:width]


def resolve_order_key_for_pick(state: WorldState, item_id: str) -> str | None:
    """Best-effort customer order for a layer pick at *state.as_of*."""
    cutoff = state.as_of + WAVED_HORIZON
    candidates = [
        o
        for o in state.orders
        if o.item_id == item_id
        and o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: (row.due_at, row.order_id))
    return true_order_key(best.order_id)


def color_key_for_pick(
    *,
    order_key: str | None,
    to_slot: str | None,
    from_slot: str,
) -> str:
    if order_key:
        return f"order:{order_key}"
    if to_slot:
        return f"donor:{to_slot}"
    return f"face:{from_slot}"


def color_for_key(color_key: str) -> str:
    if color_key.startswith("order:"):
        digest = hashlib.md5(color_key.encode(), usedforsecurity=False).hexdigest()
        idx = int(digest[:8], 16) % len(ORDER_ROUTE_COLORS)
        return ORDER_ROUTE_COLORS[idx]
    if color_key.startswith("donor:"):
        slot = color_key.split(":", 1)[1]
        digits = "".join(ch for ch in slot if ch.isdigit())
        idx = int(digits or "0", 10) % len(DONOR_SLOT_COLORS)
        return DONOR_SLOT_COLORS[idx]
    digest = hashlib.md5(color_key.encode(), usedforsecurity=False).hexdigest()
    idx = int(digest[:8], 16) % len(ORDER_ROUTE_COLORS)
    return ORDER_ROUTE_COLORS[idx]


def pick_route_label(
    *,
    from_slot: str,
    to_slot: str | None,
    layers: int,
    item_id: str,
    order_key: str | None,
) -> str:
    dest = to_slot or "face"
    ord_bit = f"Ord {short_order_key(order_key)} · " if order_key else ""
    return f"{ord_bit}{from_slot}→{dest} · {layers}L"
