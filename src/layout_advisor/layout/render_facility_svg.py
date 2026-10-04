"""SVG rendering for pick aisle + dual-side resupply rack facility plan."""

from __future__ import annotations

import html
from pathlib import Path

from layout_advisor.layout.aisle_plan import SlotRole, SlotSide
from layout_advisor.layout.facility_plan import FacilityLayoutPlan


def render_facility_layout_svg(plan: FacilityLayoutPlan) -> str:
    """Top-down: resupply racks north/south flanks; pick aisle center; donor staging south."""
    px_per_ft = 8.0
    margin = 48
    legend_h = 320
    pick = plan.pick

    slot_x0 = pick.backup_depth_ft
    north_rack_ft = plan.rack_depth_per_side_ft + plan.rack_aisle_width_ft if plan.rack_count else 0.0
    south_rack_ft = plan.rack_depth_per_side_ft + plan.rack_aisle_width_ft if plan.rack_count else 0.0
    pick_h_ft = pick.aisle_width_ft + pick.slot_depth_ft * 2 + pick.backup_depth_ft
    pick_w_ft = pick.aisle_length_ft + pick.backup_depth_ft + pick.slot_depth_ft
    total_w_ft = max(pick_w_ft, plan.rack_length_ft + pick.backup_depth_ft + 8.0)
    total_h_ft = north_rack_ft + pick_h_ft + south_rack_ft + pick.backup_depth_ft

    width = int(total_w_ft * px_per_ft + margin * 2)
    height = int(total_h_ft * px_per_ft + margin * 2 + legend_h)

    def x_ft(x: float) -> float:
        return margin + x * px_per_ft

    def y_ft(y: float) -> float:
        return margin + 28 + y * px_per_ft

    aisle_y0 = north_rack_ft + pick.slot_depth_ft + pick.backup_depth_ft
    aisle_y1 = aisle_y0 + pick.aisle_width_ft

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f5f5"/>',
        f'<text x="{margin}" y="22" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="18" font-weight="bold">{html.escape(plan.title)}</text>',
        f'<text x="{margin}" y="40" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#444">'
        f'Ratio {plan.ratio} | {plan.rack_count} rack slots ({plan.rack_depth_tiers}-deep) | '
        f'{plan.reserve.reserve_cap_ratio:.0f}:1 reserve cap | '
        f'Total ~{plan.total_footprint_sqft:,.0f} sqft</text>',
    ]

    def draw_rack_band(*, side_label: str, y0: float, access_below: bool) -> None:
        if not plan.rack_count:
            return
        per_side = plan.rack_per_side
        lines.append(
            f'<text x="{x_ft(slot_x0)}" y="{y_ft(y0 - 0.4)}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11" font-weight="bold" fill="#E65100">'
            f'RESUPPLY RACK: {side_label} ({plan.rack_depth_tiers}-deep)</text>'
        )
        drawn = 0
        for bay in range(plan.rack_bays_per_side):
            for tier in range(plan.rack_depth_tiers):
                if drawn >= per_side:
                    break
                rx = slot_x0 + bay * pick.slot_width_ft
                ry = y0 + tier * pick.slot_depth_ft
                fill = "#FFB74D" if tier == 0 else "#FFA726"
                lines.append(
                    f'<rect x="{x_ft(rx)}" y="{y_ft(ry)}" '
                    f'width="{pick.slot_width_ft * px_per_ft}" height="{pick.slot_depth_ft * px_per_ft}" '
                    f'fill="{fill}" fill-opacity="0.9" stroke="#E65100" stroke-width="0.8" rx="1"/>'
                )
                if drawn % 8 == 0:
                    depth = "F" if tier == 0 else "B"
                    lines.append(
                        f'<text x="{x_ft(rx + pick.slot_width_ft / 2)}" '
                        f'y="{y_ft(ry + pick.slot_depth_ft / 2 + 0.2)}" text-anchor="middle" '
                        f'font-family="Segoe UI, Arial, sans-serif" font-size="7" fill="#4E342E">'
                        f'R-{side_label[0]}{depth}</text>'
                    )
                drawn += 1
        access_y = y0 + plan.rack_depth_per_side_ft + (0.3 if access_below else -plan.rack_aisle_width_ft - 0.3)
        lines.append(
            f'<rect x="{x_ft(slot_x0)}" y="{y_ft(access_y)}" '
            f'width="{plan.rack_length_ft * px_per_ft}" '
            f'height="{plan.rack_aisle_width_ft * px_per_ft}" '
            f'fill="#FFF3E0" stroke="#FF9800" stroke-width="1" stroke-dasharray="6,3"/>'
        )
        lines.append(
            f'<text x="{x_ft(slot_x0 + plan.rack_length_ft / 2)}" '
            f'y="{y_ft(access_y + plan.rack_aisle_width_ft / 2 + 0.3)}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="9" fill="#E65100">rack access</text>'
        )

    draw_rack_band(side_label="LEFT (source flank)", y0=0.5, access_below=True)

    ax0 = slot_x0
    ax1 = slot_x0 + pick.aisle_length_ft
    lines.append(
        f'<rect x="{x_ft(ax0)}" y="{y_ft(aisle_y0)}" '
        f'width="{(ax1 - ax0) * px_per_ft}" height="{(aisle_y1 - aisle_y0) * px_per_ft}" '
        f'fill="#ECEFF1" stroke="#607D8B" stroke-width="1"/>'
    )
    lines.append(
        f'<text x="{x_ft((ax0 + ax1) / 2)}" y="{y_ft((aisle_y0 + aisle_y1) / 2 + 0.3)}" '
        f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#546E7A">'
        f'PICK AISLE: clamp picker</text>'
    )

    def slot_fill(role: SlotRole) -> str:
        return "#2E7D32" if role == SlotRole.SOURCE else "#1565C0"

    for slot in pick.slots:
        x = slot_x0 + slot.position_index * pick.slot_width_ft
        if slot.side == SlotSide.LEFT:
            y = north_rack_ft + pick.backup_depth_ft
            backup_y = y - pick.backup_depth_ft
        else:
            y = aisle_y1
            backup_y = aisle_y1 + pick.slot_depth_ft

        fill = slot_fill(slot.role)
        label = "S" if slot.role == SlotRole.SOURCE else "D"
        lines.append(
            f'<rect x="{x_ft(x)}" y="{y_ft(y)}" '
            f'width="{pick.slot_width_ft * px_per_ft}" height="{pick.slot_depth_ft * px_per_ft}" '
            f'fill="{fill}" fill-opacity="0.9" stroke="#263238" stroke-width="1" rx="2"/>'
        )
        lines.append(
            f'<text x="{x_ft(x + pick.slot_width_ft / 2)}" y="{y_ft(y + pick.slot_depth_ft / 2 + 0.35)}" '
            f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" '
            f'font-weight="bold" fill="white">{label}</text>'
        )
        if slot.has_backup_row:
            lines.append(
                f'<rect x="{x_ft(x)}" y="{y_ft(backup_y)}" '
                f'width="{pick.slot_width_ft * px_per_ft}" height="{pick.backup_depth_ft * px_per_ft}" '
                f'fill="#90A4AE" fill-opacity="0.75" stroke="#607D8B" stroke-width="1" stroke-dasharray="4,2"/>'
            )
        elif slot.requires_ship_access:
            lines.append(
                f'<rect x="{x_ft(x)}" y="{y_ft(backup_y)}" '
                f'width="{pick.slot_width_ft * px_per_ft}" height="{pick.backup_depth_ft * px_per_ft}" '
                f'fill="#64B5F6" fill-opacity="0.55" stroke="#1565C0" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            lines.append(
                f'<text x="{x_ft(x + pick.slot_width_ft / 2)}" y="{y_ft(backup_y + pick.backup_depth_ft / 2 + 0.25)}" '
                f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="7" fill="#0D47A1">'
                f'donor stage</text>'
            )

    south_y0 = north_rack_ft + pick_h_ft + pick.backup_depth_ft * 0.5
    draw_rack_band(side_label="RIGHT (donor flank)", y0=south_y0, access_below=False)

    lines.append(
        f'<rect x="{x_ft(0)}" y="{y_ft(0)}" width="{total_w_ft * px_per_ft}" '
        f'height="{(north_rack_ft - 0.2) * px_per_ft}" fill="none" stroke="#FF9800" stroke-width="2"/>'
    )
    lines.append(
        f'<rect x="{x_ft(0)}" y="{y_ft(north_rack_ft)}" width="{total_w_ft * px_per_ft}" '
        f'height="{(pick_h_ft + pick.backup_depth_ft) * px_per_ft}" fill="none" stroke="#455A64" stroke-width="2"/>'
    )
    if plan.rack_count:
        lines.append(
            f'<rect x="{x_ft(0)}" y="{y_ft(south_y0 - 0.5)}" width="{total_w_ft * px_per_ft}" '
            f'height="{(south_rack_ft + 1) * px_per_ft}" fill="none" stroke="#FF9800" stroke-width="2"/>'
        )

    ly = total_h_ft * px_per_ft + margin + 36
    lx = margin
    legend = [
        ("#FFB74D", f"Side rack ({plan.rack_depth_tiers}-deep): capped @ {plan.reserve.reserve_cap_ratio:.0f}:1 reserve"),
        ("#2E7D32", "Source face: pick layers FROM"),
        ("#1565C0", "Donor face: build pallet, pick layers TO"),
        ("#ECEFF1", "Pick aisle: clamp picker travel"),
    ]
    if pick.backup_depth_ft > 0:
        legend.insert(2, ("#90A4AE", "Source backup: one deep behind each source"))
        legend.insert(4, ("#64B5F6", "Donor staging: completed build behind face (no WAIT)"))
    for i, (color, text) in enumerate(legend):
        yy = ly + i * 18
        lines.append(f'<rect x="{lx}" y="{yy}" width="14" height="14" fill="{color}" stroke="#333"/>')
        lines.append(
            f'<text x="{lx + 20}" y="{yy + 12}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11">{html.escape(text)}</text>'
        )
    for i, note in enumerate(plan.notes):
        lines.append(
            f'<text x="{lx}" y="{ly + 120 + i * 14}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="10" fill="#555">• {html.escape(note)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def write_facility_layout_svg(plan: FacilityLayoutPlan, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_facility_layout_svg(plan), encoding="utf-8")
    return path
