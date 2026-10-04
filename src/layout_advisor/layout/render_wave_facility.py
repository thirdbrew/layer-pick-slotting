"""Render per-wave 2D floor occupancy — dual-sided aisle (50 pads per flank)."""

from __future__ import annotations

import html
import json
from pathlib import Path

from layout_advisor.layout.aisle_plan import SlotSide
from layout_advisor.layout.floor_snapshot import LayerPickOverlay, WaveFloorSnapshot
from layout_advisor.layout.pick_order_visual import (
    color_for_key,
    color_key_for_pick,
    pick_route_label,
    short_order_key,
)
from layout_advisor.layout.slot_geometry import (
    build_aisle_slot_map,
    gap_excluded_stations,
    stations_per_side,
)
from layout_advisor.layout.reserve_policy import DEFAULT_RACK_DEPTH_TIERS
from layout_advisor.planning.wave_floor_timeline import WaveFloorTimeline
from slotting.rack_depth import rack_depth_tier, rack_station_index

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_relative(path: Path) -> str:
    """Path relative to the repo root, POSIX-style; absolute if it sits outside."""
    try:
        return Path(path).resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


CELL_W = 34
CELL_H = 30
GAP = 2
MARGIN = 20
HEADER_FULL = 72
HEADER_COMPACT = 8
LABEL_W = 72
DEFAULT_ZOOM = 1.2


def _short_sku(item_id: str | None, *, max_len: int = 8) -> str:
    if not item_id:
        return ""
    return item_id if len(item_id) <= max_len else item_id[-max_len:]


def _fill(occ) -> str:
    if not occ or not occ.item_id:
        if occ and occ.zone == "donor_face":
            return "#E8EAF6"  # empty donor pad (not a pick-face gap)
        if occ and occ.zone == "source_face":
            return "#FFEBEE"  # empty source face — needs ADD / resupply
        return "#EEEEEE"
    if occ.zone == "rack_front":
        return "#E65100" if occ.waved else "#FFB74D"
    if occ.zone == "rack_back":
        return "#BF360C" if occ.waved else "#FFA726"
    if occ.zone == "rack":
        return "#E65100" if occ.waved else "#FFB74D"
    if occ.zone == "source_backup":
        return "#A5D6A7" if occ.waved else "#C8E6C9"
    if occ.zone == "source_face":
        return "#1B5E20" if occ.waved else "#43A047"
    if occ.zone == "donor_staging":
        return "#64B5F6"
    if occ.zone == "donor_face":
        return "#0D47A1" if occ.waved else "#1565C0"
    return "#BDBDBD"


def _state_label(occ) -> str:
    """Redundant text for the color state (colorblind + screen-reader path)."""
    if not occ or not occ.item_id:
        if occ and occ.zone == "donor_face":
            return "empty donor pad"
        if occ and occ.zone == "source_face":
            return "empty source face: needs resupply"
        return "empty"
    zone = {
        "rack_front": "rack (front)", "rack_back": "rack (back)", "rack": "rack",
        "source_backup": "source backup", "source_face": "source face",
        "donor_staging": "donor staging", "donor_face": "donor face",
    }.get(occ.zone, occ.zone)
    return f"{zone}{': waved' if occ.waved else ''}"


def _rack_bay_index(slot_id: str, *, stations: int) -> int:
    """Aisle station column for a rack slot (1:1 with face stations under default depth)."""
    if stations <= 0:
        return 0
    return min(stations - 1, max(0, rack_station_index(slot_id)))


def _rack_flank_slots(snapshot: WaveFloorSnapshot, flank: str) -> list:
    token = f"-{flank}"
    return [
        o
        for o in snapshot.slots
        if o.zone in {"rack", "rack_front", "rack_back"} and token in o.slot_id
    ]


def _rack_bay_count(flank_slots: list, *, stations: int) -> int:
    """Rack columns always match aisle station count (1:1 bays per flank)."""
    return stations


def _build_station_grids(
    snapshot: WaveFloorSnapshot,
    *,
    source_count: int,
    donor_count: int,
    rack_only_reserve: bool = True,
    picker_count: int = 1,
    center_gap_ft: float = 0.0,
) -> tuple[int, int, int, dict[str, list], set[int]]:
    layout, mapping = build_aisle_slot_map(
        source_count=source_count,
        donor_count=donor_count,
        rack_only_reserve=rack_only_reserve,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )
    stations = stations_per_side(layout)
    occ_by_sim = snapshot.by_slot_id()

    rack_l = _rack_flank_slots(snapshot, "L")
    rack_r = _rack_flank_slots(snapshot, "R")
    bays_l = _rack_bay_count(rack_l, stations=stations)
    bays_r = _rack_bay_count(rack_r, stations=stations)

    grids: dict[str, list] = {
        "rack_l_back": [None] * bays_l,
        "rack_l_front": [None] * bays_l,
        "left_staging": [None] * stations,
        "left_backup": [None] * stations,
        "left_face": [None] * stations,
        "right_face": [None] * stations,
        "right_staging": [None] * stations,
        "rack_r_front": [None] * bays_r,
        "rack_r_back": [None] * bays_r,
    }

    for sim_id, mapped in mapping.items():
        occ = occ_by_sim.get(sim_id)
        if occ is None:
            continue
        st = mapped.station
        if mapped.zone == "source_face":
            key = "left_face" if mapped.side == SlotSide.LEFT else "right_face"
            grids[key][st] = occ
        elif mapped.zone == "source_backup":
            if mapped.side == SlotSide.LEFT:
                grids["left_backup"][st] = occ
        elif mapped.zone == "donor_face":
            key = "left_face" if mapped.side == SlotSide.LEFT else "right_face"
            grids[key][st] = occ
        elif mapped.zone == "donor_staging":
            if mapped.side == SlotSide.LEFT:
                grids["left_staging"][st] = occ
            else:
                grids["right_staging"][st] = occ

    for occ in rack_l:
        bay = _rack_bay_index(occ.slot_id, stations=bays_l)
        if bay >= bays_l:
            continue
        key = "rack_l_front" if rack_depth_tier(occ.slot_id) == 0 else "rack_l_back"
        grids[key][bay] = occ
    for occ in rack_r:
        bay = _rack_bay_index(occ.slot_id, stations=bays_r)
        if bay >= bays_r:
            continue
        key = "rack_r_front" if rack_depth_tier(occ.slot_id) == 0 else "rack_r_back"
        grids[key][bay] = occ

    gap_zone = gap_excluded_stations(layout)
    for st in gap_zone:
        for key in ("left_face", "right_face", "left_staging", "right_staging", "left_backup"):
            if st < len(grids[key]):
                grids[key][st] = None
        if st < bays_l:
            grids["rack_l_front"][st] = None
            grids["rack_l_back"][st] = None
        if st < bays_r:
            grids["rack_r_front"][st] = None
            grids["rack_r_back"][st] = None

    return stations, bays_l, bays_r, grids, gap_zone


def _draw_station_row(
    lines: list[str],
    *,
    y: int,
    label: str,
    cells: list,
    x0: int,
    stations: int,
    blocked_stations: set[int] | None = None,
    slot_centers: dict[str, tuple[float, float]] | None = None,
    orders_by_slot: dict[str, str] | None = None,
) -> int:
    blocked = blocked_stations or set()
    orders_by_slot = orders_by_slot or {}
    lines.append(
        f'<text x="{MARGIN}" y="{y + 14}" font-family="Segoe UI,Arial,sans-serif" '
        f'font-size="12" font-weight="bold" fill="#37474F">{html.escape(label)}</text>'
    )
    row_y = y + 18
    for i in range(stations):
        x = x0 + i * (CELL_W + GAP)
        if i in blocked:
            lines.append(
                f'<rect x="{x}" y="{row_y}" width="{CELL_W}" height="{CELL_H}" '
                f'fill="#FFF8E1" stroke="#FFB300" stroke-dasharray="3,2"/>'
            )
            lines.append(
                f'<text x="{x + CELL_W / 2}" y="{row_y + CELL_H / 2 + 3}" text-anchor="middle" '
                f'font-size="7" fill="#F57F17"> - </text>'
            )
            continue
        occ = cells[i] if i < len(cells) else None
        fill = _fill(occ)
        slot_id = occ.slot_id if occ else ""
        item = occ.item_id if occ else ""
        title = html.escape(f"{slot_id}: {item or 'empty'} - {_state_label(occ)}")
        stroke = "#C62828" if occ and occ.zone == "source_face" and not item else "#455A64"
        slot_attr = f' data-slot-id="{html.escape(slot_id)}" class="slot-cell"' if slot_id else ""
        lines.append(
            f'<rect x="{x}" y="{row_y}" width="{CELL_W}" height="{CELL_H}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.8" rx="1"{slot_attr}>'
            f'<title>{title}</title></rect>'
        )
        if item:
            order_short = orders_by_slot.get(slot_id, "")
            if order_short:
                lines.append(
                    f'<text x="{x + CELL_W / 2}" y="{row_y + 10}" text-anchor="middle" '
                    f'font-family="Consolas,monospace" font-size="6.5" fill="#FFE082">'
                    f'#{html.escape(order_short)}</text>'
                )
            # 6-char SKU tail fits the 34px cell; ids share long prefixes so the
            # tail is the distinctive part. Full id stays in the cell tooltip.
            lines.append(
                f'<text x="{x + CELL_W / 2}" y="{row_y + CELL_H - 6}" text-anchor="middle" '
                f'font-family="Consolas,monospace" font-size="7.5" fill="white">'
                f'{html.escape(_short_sku(item, max_len=6))}</text>'
            )
        if slot_id and slot_centers is not None:
            slot_centers[slot_id] = (x + CELL_W / 2, row_y + CELL_H / 2)
    return row_y + CELL_H + 10


PEAK_LP_ROUTE_CAP = 40
DEFAULT_LP_ROUTE_CAP = 28


def _aggregate_picks_for_display(
    picks: tuple[LayerPickOverlay, ...],
    *,
    max_routes: int | None = None,
) -> tuple[list[LayerPickOverlay], int]:
    """Merge repeated SKU routes; return displayed routes and hidden route count."""
    merged: dict[tuple[str, str | None, str, str | None], tuple[int, str | None, str | None]] = {}
    for pick in picks:
        key = (pick.from_slot, pick.to_slot, pick.item_id, pick.order_key)
        layers, order_key, to_pallet = merged.get(key, (0, pick.order_key, pick.to_pallet_id))
        merged[key] = (layers + pick.layers, order_key or pick.order_key, to_pallet or pick.to_pallet_id)
    cap = max_routes
    if cap is None:
        cap = PEAK_LP_ROUTE_CAP if len(picks) > DEFAULT_LP_ROUTE_CAP else DEFAULT_LP_ROUTE_CAP
    ranked = sorted(merged.items(), key=lambda row: row[1][0], reverse=True)[:cap]
    hidden = max(0, len(merged) - len(ranked))
    routes = [
        LayerPickOverlay(
            item_id=item_id,
            layers=layers,
            from_slot=from_slot,
            to_slot=to_slot,
            order_key=order_key,
            to_pallet_id=to_pallet,
        )
        for (from_slot, to_slot, item_id, _order_key), (layers, order_key, to_pallet) in ranked
    ]
    return routes, hidden


def _route_points(
    src: tuple[float, float],
    dst: tuple[float, float],
    *,
    aisle_lane_y: float | None,
    lane_offset: float = 0.0,
    stroke: str = "#2E7D32",
) -> tuple[str, float, float]:
    """Return SVG route markup and label anchor for an LP route."""
    if abs(dst[0] - src[0]) <= 2 and abs(dst[1] - src[1]) <= 2:
        return "", src[0], src[1] - 10
    lane = (aisle_lane_y + lane_offset) if aisle_lane_y is not None else None
    cross_aisle = lane is not None and (
        (src[1] < lane < dst[1]) or (dst[1] < lane < src[1])
    )
    width = 2.2 + min(2.0, abs(lane_offset) * 0.15)
    if cross_aisle and lane is not None:
        points = (
            f"{src[0]:.1f},{src[1]:.1f} "
            f"{src[0]:.1f},{lane:.1f} "
            f"{dst[0]:.1f},{lane:.1f} "
            f"{dst[0]:.1f},{dst[1]:.1f}"
        )
        return (
            f'<polyline points="{points}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:.1f}" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>',
            (src[0] + dst[0]) / 2,
            lane - 8,
        )
    return (
        f'<line x1="{src[0]:.1f}" y1="{src[1]:.1f}" x2="{dst[0]:.1f}" y2="{dst[1]:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}" stroke-linecap="round" opacity="0.95"/>',
        (src[0] + dst[0]) / 2,
        (src[1] + dst[1]) / 2 - 4,
    )


def _src_slot_tag(src: tuple[float, float], from_slot: str, color: str) -> str:
    """Small pill naming the individual slot (face or rack) a pick grabs from."""
    tag_w = max(26, len(from_slot) * 6 + 8)
    tx, ty = src[0], src[1] + 16
    return (
        f'<rect x="{tx - tag_w / 2:.1f}" y="{ty - 8}" width="{tag_w}" height="12" rx="6" '
        f'fill="#ffffff" stroke="{color}" stroke-width="1" opacity="0.95" class="lp-src-tag"/>'
        f'<text x="{tx:.1f}" y="{ty + 1.5}" text-anchor="middle" '
        f'font-family="Consolas,monospace" font-size="7.5" font-weight="bold" fill="{color}">'
        f'{html.escape(from_slot)}</text>'
    )


def _draw_layer_picks(
    lines: list[str],
    picks: list[LayerPickOverlay],
    centers: dict[str, tuple[float, float]],
    *,
    aisle_lane_y: float | None = None,
) -> None:
    if not picks:
        return
    lines.append('<g id="layer-picks" opacity="0.98">')
    lane_offsets: dict[float, int] = {}
    for index, pick in enumerate(picks):
        src = centers.get(pick.from_slot)
        if src is None:
            continue
        dst = centers.get(pick.to_slot) if pick.to_slot else None
        color_key = color_key_for_pick(
            order_key=pick.order_key,
            to_slot=pick.to_slot,
            from_slot=pick.from_slot,
        )
        color = color_for_key(color_key)
        route_id = html.escape(
            f"{pick.order_key or ''}|{pick.from_slot}|{pick.to_slot or ''}|{pick.item_id}"
        )
        label = pick_route_label(
            from_slot=pick.from_slot,
            to_slot=pick.to_slot,
            layers=pick.layers,
            item_id=pick.item_id,
            order_key=pick.order_key,
        )
        sku = _short_sku(pick.item_id, max_len=10)
        lane_offset = 0.0
        if aisle_lane_y is not None:
            bucket = round(aisle_lane_y, 1)
            slot = lane_offsets.get(bucket, 0)
            lane_offsets[bucket] = slot + 1
            lane_offset = (slot - (len(picks) - 1) / 2.0) * 5.0
        if dst and (abs(dst[0] - src[0]) > 2 or abs(dst[1] - src[1]) > 2):
            route_svg, mid_x, mid_y = _route_points(
                src,
                dst,
                aisle_lane_y=aisle_lane_y,
                lane_offset=lane_offset,
                stroke=color,
            )
            lines.append(
                f'<g class="lp-route" data-route-id="{route_id}" '
                f'data-order-key="{html.escape(pick.order_key or "")}" '
                f'data-color-key="{html.escape(color_key)}" '
                f'data-color="{color}" '
                f'data-from-slot="{html.escape(pick.from_slot)}" '
                f'data-to-slot="{html.escape(pick.to_slot or "")}">{route_svg}'
                f'<circle cx="{dst[0]:.1f}" cy="{dst[1]:.1f}" r="5.2" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1.2" class="lp-dest-marker"/>'
                f'<circle cx="{src[0]:.1f}" cy="{src[1]:.1f}" r="5" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1.4" opacity="0.9" class="lp-src-marker"/>'
                f'{_src_slot_tag(src, pick.from_slot, color)}</g>'
            )
        else:
            mid_x, mid_y = src[0], src[1] - 12
            lines.append(
                f'<circle cx="{src[0]:.1f}" cy="{src[1]:.1f}" r="4.2" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1" class="lp-src-marker"/>'
            )
        label_w = max(104, len(label) * 6.2)
        ord_line = (
            f"Ord {short_order_key(pick.order_key, width=6)}"
            if pick.order_key
            else (pick.to_slot or "no build")
        )
        lines.append(
            f'<g class="lp-route-label" data-route-id="{route_id}" data-color="{color}">'
            f'<rect x="{mid_x - label_w / 2}" y="{mid_y - 14}" width="{label_w:.0f}" height="31" rx="4" '
            f'fill="#ffffff" stroke="{color}" stroke-width="1.2" opacity="0.96"/>'
            f'<text x="{mid_x}" y="{mid_y - 2}" text-anchor="middle" '
            f'font-family="Consolas,monospace" font-size="8" font-weight="bold" fill="{color}">'
            f'{html.escape(label)}</text>'
            f'<text x="{mid_x}" y="{mid_y + 10}" text-anchor="middle" '
            f'font-family="Segoe UI,Arial,sans-serif" font-size="7" fill="#455A64">'
            f'{html.escape(ord_line)} · {html.escape(sku)}</text></g>'
        )
    lines.append("</g>")


def render_wave_floor_svg(
    snapshot: WaveFloorSnapshot,
    *,
    source_count: int = 70,
    donor_count: int = 25,
    rack_only_reserve: bool = True,
    rack_depth_tiers: int = DEFAULT_RACK_DEPTH_TIERS,
    center_gap_ft: float = 0.0,
    picker_count: int = 1,
    aisle_width_ft: float = 8.0,
    compact: bool = False,
) -> str:
    """Top-down dual-flank grid: source + donor pads along aisle, racks on flanks."""
    rack_depth_tiers = max(1, rack_depth_tiers)
    two_deep_rack = rack_depth_tiers > 1
    header = HEADER_COMPACT if compact else HEADER_FULL
    stations, bays_l, bays_r, grids, gap_zone = _build_station_grids(
        snapshot,
        source_count=source_count,
        donor_count=donor_count,
        rack_only_reserve=rack_only_reserve,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )
    total_pads = source_count + donor_count
    rack_cols = stations
    aisle_cols = stations

    row_count = (7 if rack_only_reserve else 9) - (0 if two_deep_rack else 2)
    grid_w = max(aisle_cols, rack_cols) * (CELL_W + GAP)
    svg_w = MARGIN * 2 + LABEL_W + grid_w + 24
    svg_h = header + row_count * (CELL_H + 28) + (24 if compact else 90)
    x0_aisle = MARGIN + LABEL_W
    x0_rack_l = x0_aisle
    x0_rack_r = x0_aisle

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" role="img" '
        f'aria-label="Warehouse floor map: source faces, donor pads, and rack slots; '
        f'each cell tooltip names the slot, SKU, and state (waved / empty / resupply).">',
        '<defs><marker id="lp-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#2E7D32"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
    ]
    slot_centers: dict[str, tuple[float, float]] = {}
    pick_routes, hidden_routes = _aggregate_picks_for_display(snapshot.layer_picks)
    # Order tag inside destination build pads (donor cells) — from the pick overlay.
    # 4 chars so the tag stays inside a 34px cell instead of bleeding into
    # neighbors; the cell tooltip and route labels carry longer forms.
    orders_by_slot = {
        p.to_slot: short_order_key(p.order_key, width=4)
        for p in snapshot.layer_picks
        if p.to_slot and p.order_key
    }
    pick_layers = sum(p.layers for p in snapshot.layer_picks)
    if not compact:
        pick_note = f"Layer picks: {len(snapshot.layer_picks)} transfers · {pick_layers:,} layers"
        if hidden_routes:
            pick_note += f" · +{hidden_routes} routes hidden"
        rack_desc = (
            f"{rack_depth_tiers}-deep racks each flank"
            if two_deep_rack
            else "1-deep front rack each flank"
        )
        lines.extend(
            [
                f'<text x="{MARGIN}" y="18" font-family="Segoe UI,Arial,sans-serif" font-size="14" '
                f'font-weight="600">{html.escape(snapshot.label)}</text>',
                f'<text x="{MARGIN}" y="34" font-family="Segoe UI,Arial,sans-serif" font-size="10" fill="#546E7A">'
                f'{total_pads} pads ({source_count} source + {donor_count} donor) | {rack_desc} | '
                f'waved {snapshot.waved_skus} SKUs / {snapshot.waved_layers:,} lyr</text>',
            ]
        )
        if snapshot.wave_send and snapshot.wave_due:
            lines.append(
                f'<text x="{MARGIN}" y="48" font-family="Segoe UI,Arial,sans-serif" font-size="10" fill="#78909C">'
                f'Send {snapshot.wave_send:%H:%M} → due {snapshot.wave_due:%H:%M}</text>'
            )
        if pick_layers:
            lines.append(
                f'<text x="{MARGIN}" y="62" font-family="Segoe UI,Arial,sans-serif" font-size="10" fill="#2E7D32">'
                f'{html.escape(pick_note)} ({len(pick_routes)} routes shown)</text>'
            )

    y = header
    # Every row registers slot_centers so routes can reach rack drops, not just faces.
    if two_deep_rack:
        y = _draw_station_row(
            lines, y=y, label="L rack back", cells=grids["rack_l_back"], x0=x0_rack_l, stations=bays_l, slot_centers=slot_centers
        )
    y = _draw_station_row(
        lines, y=y, label="L rack", cells=grids["rack_l_front"], x0=x0_rack_l, stations=bays_l, slot_centers=slot_centers
    )
    if not rack_only_reserve:
        y = _draw_station_row(lines, y=y, label="L stage", cells=grids["left_staging"], x0=x0_aisle, stations=stations, slot_centers=slot_centers)
        y = _draw_station_row(lines, y=y, label="L backup", cells=grids["left_backup"], x0=x0_aisle, stations=stations, slot_centers=slot_centers)
    y = _draw_station_row(lines, y=y, label="LEFT", cells=grids["left_face"], x0=x0_aisle, stations=stations, blocked_stations=gap_zone, slot_centers=slot_centers, orders_by_slot=orders_by_slot)

    aisle_y = y + 4
    aisle_w = aisle_cols * (CELL_W + GAP) - GAP
    if center_gap_ft > 0 or picker_count > 1:
        gap_px = max(12, int((center_gap_ft / max(aisle_width_ft, 1.0)) * 24)) if center_gap_ft > 0 else 12
        lane_w = max(20, (aisle_w - gap_px) // 2)
        lines.append(
            f'<rect x="{x0_aisle}" y="{aisle_y}" width="{lane_w}" height="14" fill="#E3F2FD" stroke="#1565C0"/>'
        )
        lines.append(
            f'<text x="{x0_aisle + lane_w / 2}" y="{aisle_y + 11}" text-anchor="middle" '
            f'font-size="9" fill="#1565C0">L PICKER</text>'
        )
        gap_x = x0_aisle + lane_w
        lines.append(
            f'<rect x="{gap_x}" y="{aisle_y}" width="{gap_px}" height="14" fill="#FFF8E1" stroke="#FFB300"/>'
        )
        lines.append(
            f'<text x="{gap_x + gap_px / 2}" y="{aisle_y + 11}" text-anchor="middle" '
            f'font-size="8" fill="#F57F17">{int(center_gap_ft)}ft GAP: NO STAGING</text>'
        )
        lines.append(
            f'<rect x="{gap_x + gap_px}" y="{aisle_y}" width="{lane_w}" height="14" fill="#E8F5E9" stroke="#2E7D32"/>'
        )
        lines.append(
            f'<text x="{gap_x + gap_px + lane_w / 2}" y="{aisle_y + 11}" text-anchor="middle" '
            f'font-size="9" fill="#2E7D32">R PICKER</text>'
        )
    else:
        lines.append(
            f'<rect x="{x0_aisle}" y="{aisle_y}" width="{aisle_w}" height="14" fill="#ECEFF1" stroke="#607D8B"/>'
        )
        lines.append(
            f'<text x="{x0_aisle + aisle_w / 2}" y="{aisle_y + 11}" text-anchor="middle" '
            f'font-size="10" fill="#546E7A">PICK AISLE</text>'
        )
    y = aisle_y + 24

    y = _draw_station_row(lines, y=y, label="RIGHT", cells=grids["right_face"], x0=x0_aisle, stations=stations, blocked_stations=gap_zone, slot_centers=slot_centers, orders_by_slot=orders_by_slot)
    if not rack_only_reserve:
        y = _draw_station_row(lines, y=y, label="R stage", cells=grids["right_staging"], x0=x0_aisle, stations=stations, slot_centers=slot_centers)
    y = _draw_station_row(
        lines, y=y, label="R rack", cells=grids["rack_r_front"], x0=x0_rack_r, stations=bays_r, slot_centers=slot_centers
    )
    if two_deep_rack:
        y = _draw_station_row(
            lines, y=y, label="R rack back", cells=grids["rack_r_back"], x0=x0_rack_r, stations=bays_r, slot_centers=slot_centers
        )

    _draw_layer_picks(lines, pick_routes, slot_centers, aisle_lane_y=aisle_y + 7)

    if not compact:
        ly = y + 6
        legend = [
            ("#1B5E20", "Waved source"),
            ("#43A047", "Visible source"),
            ("#0D47A1", "Waved donor"),
            ("#1565C0", "Donor build"),
            ("#2E7D32", "LP pick route"),
            ("#FFB74D", "Rack"),
            ("#EEEEEE", "Empty"),
        ]
        for i, (color, text) in enumerate(legend):
            lx = MARGIN + (i % 3) * 180
            yy = ly + (i // 3) * 14
            lines.append(f'<rect x="{lx}" y="{yy}" width="10" height="10" fill="{color}" stroke="#333"/>')
            lines.append(
                f'<text x="{lx + 14}" y="{yy + 9}" font-size="9">{html.escape(text)}</text>'
            )
    lines.append("</svg>")
    return "\n".join(lines)


def render_wave_floor_html(timeline: WaveFloorTimeline) -> str:
    """Interactive HTML — scrub setup, waves, and peak hours on the floor grid."""
    layout_kwargs = {
        "source_count": timeline.source_count,
        "donor_count": timeline.donor_count,
        "picker_count": timeline.picker_count,
        "center_gap_ft": timeline.center_gap_ft,
        "compact": True,
    }
    frames = [
        {
            "id": i,
            "label": f.label,
            "time": f.as_of.strftime("%H:%M"),
            "waved_skus": f.waved_skus,
            "waved_layers": f.waved_layers,
            "occupied": f.occupied_count,
            "empty_source": f.empty_source_faces,
            "empty_donor": f.empty_donor_pads,
            "layer_pick_count": len(f.layer_picks),
            "layer_pick_layers": sum(p.layers for p in f.layer_picks),
            "hourly_adds": f.hourly_adds,
            "hourly_removes": f.hourly_removes,
            "hourly_donor_exits": f.hourly_donor_exits,
            "hourly_rack_promotes": f.hourly_rack_promotes,
            "is_realignment_hour": f.is_realignment_hour,
            "gap_hours_since_prior": f.gap_hours_since_prior,
            "face_swaps_from_prior": f.face_swaps_from_prior,
            "faces_cleared_from_prior": f.faces_cleared_from_prior,
            "faces_filled_from_prior": f.faces_filled_from_prior,
            "svg": render_wave_floor_svg(f, **layout_kwargs),
        }
        for i, f in enumerate(timeline.frames)
    ]
    payload = json.dumps(frames)
    total_pads = timeline.source_count + timeline.donor_count
    pads_per_side = total_pads // 2
    picker_note = (
        f"Dual pickers · {int(timeline.center_gap_ft)} ft gap"
        if timeline.picker_count > 1 or timeline.center_gap_ft > 0
        else "Single picker"
    )
    title = f"LP floor: {timeline.ship_date} ({timeline.layout})"
    realign_note = ""
    if timeline.realignment_hours:
        hrs = ", ".join(f"{h:02d}:00" for h in timeline.realignment_hours)
        realign_note = (
            f"Face realign @ {hrs} - evict wrong-SKU faces, restage empties (no rack bootstrap)"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #e9edf0;
      --panel: #ffffff;
      --ink: #141b22;
      --muted: #607d8b;
      --accent: #1c6fb0; --source: #1c6fb0; --source-ink: #0e4c7d; --donor: #b9761c;
      --border: #cfd6dc;
      --mono: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--ink); }}
    header {{
      background: var(--ink); color: #eef2f5; padding: 18px 20px 16px;
      display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
      background-image: linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
      background-size: 22px 22px; border-bottom: 3px solid var(--source);
    }}
    .tb-mark {{ font-family: var(--mono); font-weight: 700; font-size: 16px; letter-spacing: .12em;
      color: #eaf2fa; border: 1.5px solid var(--source); padding: 6px 10px; border-radius: 2px;
      background: rgba(28,111,176,.16); }}
    header h1 {{ margin: 0; font-size: 17px; font-weight: 600; }}
    header .sub {{ font-size: 12px; color: #aab6c0; }}
    .toolbar {{
      display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
      padding: 10px 20px; background: var(--panel); border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 5;
    }}
    .toolbar button, .toolbar select {{
      font: inherit; font-size: 13px; padding: 6px 12px;
      border: 1px solid var(--border); border-radius: 4px; background: #fff; cursor: pointer;
    }}
    .toolbar button:hover {{ border-color: var(--accent); }}
    .toolbar button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .toolbar input[type=range] {{ flex: 1; min-width: 140px; max-width: 420px; }}
    .toolbar .frame-count {{ font-family: var(--mono); font-size: 12px; color: var(--muted); min-width: 72px; text-align: center; }}
    .legend {{
      display: flex; flex-wrap: wrap; gap: 12px 18px; padding: 8px 20px;
      background: #f5f7f8; border-bottom: 1px solid var(--border); font-size: 12px;
    }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; border: 1px solid #90a4ae; }}
    #meta {{
      padding: 10px 20px; font-size: 14px; font-weight: 500;
      background: var(--panel); border-bottom: 1px solid var(--border);
    }}
    #meta .detail {{ font-size: 12px; font-weight: 400; color: var(--muted); margin-top: 2px; }}
    #delta {{
      padding: 8px 20px 10px; font-size: 12px; line-height: 1.45;
      background: #fff8e1; border-bottom: 1px solid #ffe082; color: #5d4037;
      display: none;
    }}
    #delta.realign {{ background: #e3f2fd; border-bottom-color: #90caf9; color: #0d47a1; }}
    #delta ul {{ margin: 4px 0 0; padding-left: 18px; }}
    #delta li {{ margin: 2px 0; }}
    .policy-note {{
      margin: 6px 20px; padding: 6px 10px; font-size: 12px;
      background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    }}
    #viewport {{
      padding: 16px 20px 32px; overflow: auto;
    }}
    #viewport svg {{
      display: block; background: #fff; border: 1px solid var(--border);
      border-radius: 6px; width: max-content; height: auto;
    }}
    .zoom-row {{
      display: flex; align-items: center; gap: 8px; padding: 0 20px 10px;
      font-size: 12px; color: var(--muted); background: var(--panel);
      border-bottom: 1px solid var(--border);
    }}
    .zoom-row input {{ width: 120px; }}
  </style>
</head>
<body>
  <header>
    <span class="tb-mark">Layer Pick</span>
    <div>
    <h1>{html.escape(title)}</h1>
    <span class="sub">{total_pads} pads · {timeline.source_count} source + {timeline.donor_count} donor · {pads_per_side}/flank · {html.escape(picker_note)}</span>
    </div>
  </header>

  <div class="legend">
    <span><i class="swatch" style="background:#1B5E20"></i>Waved source</span>
    <span><i class="swatch" style="background:#43A047"></i>Visible source</span>
    <span><i class="swatch" style="background:#0D47A1"></i>Waved donor</span>
    <span><i class="swatch" style="background:#1565C0"></i>Donor build</span>
    <span><i class="swatch" style="background:#FFB74D"></i>Rack reserve</span>
    <span><i class="swatch" style="background:#FFEBEE;border-color:#C62828"></i>Empty source (needs stock)</span>
    <span><i class="swatch" style="background:#E8EAF6"></i>Empty donor pad</span>
    <span><i class="swatch" style="background:#FFF8E1;border-color:#FFB300"></i>Gap / blocked</span>
    <span style="color:#2E7D32">▸ green arrows = layer pick (source → build)</span>
  </div>
  {f'<div class="policy-note" id="policy-note">{html.escape(realign_note)}</div>' if realign_note else ''}

  <div class="toolbar">
    <button type="button" id="prev" title="Previous frame">Prev</button>
    <button type="button" id="play" class="primary" title="Play / pause">Play</button>
    <button type="button" id="next" title="Next frame">Next</button>
    <input type="range" id="slider" min="0" max="{len(frames) - 1}" value="0" aria-label="Timeline"/>
    <span class="frame-count" id="counter">1 / {len(frames)}</span>
    <select id="jump" aria-label="Jump to frame"></select>
  </div>
  <div class="zoom-row">
    <label for="zoom">Zoom</label>
    <input type="range" id="zoom" min="80" max="200" value="{int(DEFAULT_ZOOM * 100)}"/>
    <span id="zoom-label">{int(DEFAULT_ZOOM * 100)}%</span>
  </div>

  <div id="meta"></div>
  <div id="delta"></div>
  <div id="viewport"></div>

  <script>
    const frames = {payload};
    const slider = document.getElementById('slider');
    const jump = document.getElementById('jump');
    const zoom = document.getElementById('zoom');
    const zoomLabel = document.getElementById('zoom-label');
    const meta = document.getElementById('meta');
    const delta = document.getElementById('delta');
    const viewport = document.getElementById('viewport');
    const counter = document.getElementById('counter');
    let active = 0;
    let timer = null;

    frames.forEach((f, i) => {{
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = f.label;
      jump.appendChild(opt);
    }});

    function applyZoom() {{
      const pct = Number(zoom.value);
      viewport.style.zoom = (pct / 100).toString();
      zoomLabel.textContent = pct + '%';
    }}

    function buildDelta(f) {{
      const items = [];
      if (f.gap_hours_since_prior > 0) {{
        items.push('Timeline skipped ' + f.gap_hours_since_prior + ' hour(s) since the prior frame: use bridge frames to step through.');
      }}
      if (f.is_realignment_hour) {{
        items.push('<strong>Face realignment hour</strong>: afternoon slotting plan evicts wrong-SKU faces and restages empties from reserve (evicted pallets stay unslotted, not auto-racked).');
      }}
      const hourParts = [];
      if (f.hourly_adds) hourParts.push(f.hourly_adds + ' face add(s)');
      if (f.hourly_donor_exits) hourParts.push(f.hourly_donor_exits + ' completed build(s) exited to wrapping');
      const otherRemoves = Math.max(0, (f.hourly_removes || 0) - (f.hourly_donor_exits || 0));
      if (otherRemoves) hourParts.push(otherRemoves + ' face pull-off(s)');
      if (f.hourly_rack_promotes) hourParts.push(f.hourly_rack_promotes + ' rack promote(s)');
      if (hourParts.length) items.push('This hour: ' + hourParts.join(' · '));
      if (f.face_swaps_from_prior != null) {{
        const faceParts = [];
        if (f.face_swaps_from_prior) faceParts.push(f.face_swaps_from_prior + ' SKU swap(s)');
        if (f.faces_cleared_from_prior) faceParts.push(f.faces_cleared_from_prior + ' face(s) cleared');
        if (f.faces_filled_from_prior) faceParts.push(f.faces_filled_from_prior + ' face(s) filled');
        if (faceParts.length) items.push('Vs prior frame: ' + faceParts.join(' · '));
      }}
      return items;
    }}

    function show(i) {{
      active = Math.max(0, Math.min(frames.length - 1, i));
      slider.value = active;
      jump.value = active;
      const f = frames[active];
      viewport.innerHTML = f.svg;
      counter.textContent = (active + 1) + ' / ' + frames.length;
      meta.innerHTML =
        '<div>' + f.label + '</div>' +
        '<div class="detail">Waved ' + f.waved_skus + ' SKUs · ' +
        f.waved_layers.toLocaleString() + ' layers · ' +
        f.occupied + ' occupied · ' +
        '<span style="color:#c62828">' + f.empty_source + ' empty source</span> · ' +
        f.empty_donor + ' empty donor pads · ' + f.time +
        (f.layer_pick_layers ? ' · <span style="color:#2E7D32">LP ' +
        f.layer_pick_count + ' picks / ' + f.layer_pick_layers.toLocaleString() + ' layers</span>' : '') +
        '</div>';
      const deltaItems = buildDelta(f);
      if (deltaItems.length) {{
        delta.style.display = 'block';
        delta.className = f.is_realignment_hour ? 'realign' : '';
        delta.innerHTML = '<strong>What changed</strong><ul>' +
          deltaItems.map((t) => '<li>' + t + '</li>').join('') + '</ul>';
      }} else {{
        delta.style.display = 'none';
        delta.innerHTML = '';
      }}
    }}

    function step(delta) {{ show(active + delta); }}

    document.getElementById('prev').onclick = () => step(-1);
    document.getElementById('next').onclick = () => step(1);
    slider.oninput = () => show(Number(slider.value));
    jump.onchange = () => show(Number(jump.value));
    zoom.oninput = applyZoom;

    document.getElementById('play').onclick = () => {{
      const btn = document.getElementById('play');
      if (timer) {{
        clearInterval(timer);
        timer = null;
        btn.textContent = 'Play';
        return;
      }}
      btn.textContent = 'Pause';
      timer = setInterval(() => {{
        if (active >= frames.length - 1) {{
          clearInterval(timer);
          timer = null;
          btn.textContent = 'Play';
          return;
        }}
        step(1);
      }}, 900);
    }};

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
      if (e.key === ' ') {{ e.preventDefault(); document.getElementById('play').click(); }}
    }});

    applyZoom();
    show(0);
  </script>
</body>
</html>"""


def write_wave_floor_views(
    timeline: WaveFloorTimeline,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir) / timeline.ship_date.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "wave_floor_timeline.html"
    html_path.write_text(render_wave_floor_html(timeline), encoding="utf-8")

    frames_dir = out_dir / "wave_frames"
    frames_dir.mkdir(exist_ok=True)
    layout_kwargs = {
        "source_count": timeline.source_count,
        "donor_count": timeline.donor_count,
        "picker_count": timeline.picker_count,
        "center_gap_ft": timeline.center_gap_ft,
    }
    frame_entries: list[dict[str, object]] = []
    for i, frame in enumerate(timeline.frames):
        if i == 0:
            safe = "setup"
        elif timeline.minute_resolution:
            safe = f"tick_{frame.as_of:%H%M}"
        elif frame.wave_index is not None:
            safe = f"wave_{frame.wave_index:02d}"
        else:
            safe = f"hour_{frame.as_of:%H%M}"
        svg_rel = f"wave_frames/{safe}.svg"
        (frames_dir / f"{safe}.svg").write_text(
            render_wave_floor_svg(frame, **layout_kwargs),
            encoding="utf-8",
        )
        frame_entries.append(
            {
                "label": frame.label,
                "svg": svg_rel,
                "as_of": frame.as_of.isoformat(sep=" "),
                "waved_skus": frame.waved_skus,
                "waved_layers": frame.waved_layers,
                "occupied": frame.occupied_count,
                "layer_pick_count": len(frame.layer_picks),
                "layer_pick_layers": sum(p.layers for p in frame.layer_picks),
                "layer_picks": [
                    {
                        "from_slot": pick.from_slot,
                        "to_slot": pick.to_slot,
                        "layers": pick.layers,
                        "item_id": pick.item_id,
                        "order_key": pick.order_key,
                    }
                    for pick in frame.layer_picks
                ],
                "wave_index": frame.wave_index,
                "is_realignment_hour": frame.is_realignment_hour,
                "hourly_removes": frame.hourly_removes,
                "hourly_donor_exits": frame.hourly_donor_exits,
                "face_swaps_from_prior": frame.face_swaps_from_prior,
            }
        )

    index = {
        "ship_date": timeline.ship_date.isoformat(),
        "layout": timeline.layout,
        "picker_count": timeline.picker_count,
        "center_gap_ft": timeline.center_gap_ft,
        "pads_per_side": (timeline.source_count + timeline.donor_count) // 2,
        "frame_count": len(timeline.frames),
        "minute_resolution": timeline.minute_resolution,
        "tick_minutes": timeline.tick_minutes,
        "realignment_hours": list(timeline.realignment_hours),
        # Repo-relative: these are informational metadata (nothing reads them
        # back), and absolute paths baked one developer's checkout location
        # into every
        # generated index. The repo resolves its own root; artifacts must too.
        "html": _repo_relative(html_path),
        "frames_dir": _repo_relative(frames_dir),
        "labels": [f.label for f in timeline.frames],
        "frames": frame_entries,
    }
    index_path = out_dir / "wave_floor_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {"html": html_path, "index": index_path, "frames_dir": frames_dir}
