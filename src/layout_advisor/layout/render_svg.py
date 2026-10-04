"""SVG rendering for aisle layout concept plans."""

from __future__ import annotations

import html
from pathlib import Path

from layout_advisor.layout.aisle_plan import AisleLayoutPlan, AisleSlot, SlotRole, SlotSide


def _slot_fill(role: SlotRole) -> str:
    return "#2E7D32" if role == SlotRole.SOURCE else "#1565C0"


def _slot_label(role: SlotRole) -> str:
    return "S" if role == SlotRole.SOURCE else "D"


def render_aisle_layout_svg(plan: AisleLayoutPlan) -> str:
    """Return SVG document string (top-down view, aisle horizontal)."""
    px_per_ft = 8.0
    margin = 40
    legend_h = 260

    # The legend used to promise backup rows unconditionally. Under the reference
    # config's `rack_only_reserve=True` none are drawn, so the sheet described a row that
    # is not on it — the same self-contradicting-prose class as the "1-deep per
    # flank" / "2-deep side racks" pair in facility_plan.py.
    has_backup = any(s.has_backup_row for s in plan.slots)
    backup_swatch = "#B0BEC5" if has_backup else "none"
    backup_legend = (
        "Backup staging behind source only (forklift resupply)"
        if has_backup
        else "No face-adjacent backup: all reserve in side racks"
    )

    w_ft = plan.aisle_length_ft + plan.slot_depth_ft * 2 + plan.backup_depth_ft
    h_ft = plan.aisle_width_ft + plan.slot_depth_ft * 2
    width = int(w_ft * px_per_ft + margin * 2)
    height = int(h_ft * px_per_ft + margin * 2 + legend_h)

    def x_ft(x: float) -> float:
        return margin + x * px_per_ft

    def y_ft(y: float) -> float:
        return margin + y * px_per_ft

    aisle_y0 = plan.slot_depth_ft
    aisle_y1 = aisle_y0 + plan.aisle_width_ft
    slot_x0 = plan.backup_depth_ft

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{margin}" y="24" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="18" font-weight="bold">{html.escape(plan.title)}</text>',
        f'<text x="{margin}" y="44" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="12" fill="#444">Ratio {plan.ratio} | '
        f'{plan.source_count} source + {plan.donor_count} donor | '
        f'~{plan.footprint_sqft:,.0f} sqft (incl. aisle{" + backup" if has_backup else ""} + reserve)</text>',
    ]

    # Aisle
    ax0 = slot_x0
    ax1 = slot_x0 + plan.aisle_length_ft
    lines.append(
        f'<rect x="{x_ft(ax0)}" y="{y_ft(aisle_y0)}" '
        f'width="{(ax1 - ax0) * px_per_ft}" height="{(aisle_y1 - aisle_y0) * px_per_ft}" '
        f'fill="#ECEFF1" stroke="#607D8B" stroke-width="1"/>'
    )
    lines.append(
        f'<text x="{x_ft((ax0 + ax1) / 2)}" y="{y_ft((aisle_y0 + aisle_y1) / 2 + 0.3)}" '
        f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="11" fill="#546E7A">AISLE: picker</text>'
    )

    def draw_slot(slot: AisleSlot) -> None:
        x = slot_x0 + slot.position_index * plan.slot_width_ft
        if slot.side == SlotSide.LEFT:
            y = 0
            backup_y = -plan.backup_depth_ft
        else:
            y = aisle_y1
            backup_y = aisle_y1 + plan.slot_depth_ft

        fill = _slot_fill(slot.role)
        label = _slot_label(slot.role)
        lines.append(
            f'<rect x="{x_ft(x)}" y="{y_ft(y)}" '
            f'width="{plan.slot_width_ft * px_per_ft}" height="{plan.slot_depth_ft * px_per_ft}" '
            f'fill="{fill}" fill-opacity="0.85" stroke="#263238" stroke-width="1" rx="2"/>'
        )
        lines.append(
            f'<text x="{x_ft(x + plan.slot_width_ft / 2)}" '
            f'y="{y_ft(y + plan.slot_depth_ft / 2 + 0.35)}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="13" '
            f'font-weight="bold" fill="white">{label}</text>'
        )
        lines.append(
            f'<text x="{x_ft(x + plan.slot_width_ft / 2)}" '
            f'y="{y_ft(y + plan.slot_depth_ft / 2 + 1.1)}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="8" fill="#ECEFF1">'
            f'{html.escape(slot.slot_id)}</text>'
        )
        if slot.has_backup_row:
            lines.append(
                f'<rect x="{x_ft(x)}" y="{y_ft(backup_y)}" '
                f'width="{plan.slot_width_ft * px_per_ft}" '
                f'height="{plan.backup_depth_ft * px_per_ft}" '
                f'fill="#B0BEC5" fill-opacity="0.7" stroke="#78909C" stroke-width="1" '
                f'stroke-dasharray="4,2" rx="1"/>'
            )
            lines.append(
                f'<text x="{x_ft(x + plan.slot_width_ft / 2)}" '
                f'y="{y_ft(backup_y + plan.backup_depth_ft / 2 + 0.25)}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="7" fill="#37474F">backup</text>'
            )
        elif slot.requires_ship_access:
            lines.append(
                f'<rect x="{x_ft(x)}" y="{y_ft(backup_y)}" '
                f'width="{plan.slot_width_ft * px_per_ft}" '
                f'height="{plan.backup_depth_ft * px_per_ft}" '
                f'fill="none" stroke="#1565C0" stroke-width="1" stroke-dasharray="3,3" rx="1"/>'
            )
            lines.append(
                f'<text x="{x_ft(x + plan.slot_width_ft / 2)}" '
                f'y="{y_ft(backup_y + plan.backup_depth_ft / 2 + 0.25)}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="7" fill="#1565C0">ship access</text>'
            )

    for slot in plan.slots:
        draw_slot(slot)

    # Legend
    ly = h_ft * px_per_ft + margin + 10
    lx = margin
    lines.extend(
        [
            f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#2E7D32"/>',
            f'<text x="{lx + 20}" y="{ly + 12}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11">Source: pick layers FROM'
            f'{" (backup row behind)" if has_backup else ""}</text>',
            f'<rect x="{lx}" y="{ly + 22}" width="14" height="14" fill="#1565C0"/>',
            f'<text x="{lx + 20}" y="{ly + 34}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11">Donor: build pallet, pick layers TO</text>',
            f'<rect x="{lx}" y="{ly + 44}" width="14" height="14" fill="{backup_swatch}" '
            f'stroke="#B0BEC5" stroke-dasharray="4,2"/>',
            f'<text x="{lx + 20}" y="{ly + 56}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11">{backup_legend}</text>',
            f'<rect x="{lx}" y="{ly + 66}" width="14" height="14" fill="none" stroke="#1565C0" stroke-dasharray="3,3"/>',
            f'<text x="{lx + 20}" y="{ly + 78}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11">Ship access behind donor: must stay clear for assist</text>',
        ]
    )
    for i, note in enumerate(plan.notes[:5]):
        lines.append(
            f'<text x="{lx}" y="{ly + 96 + i * 14}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="10" fill="#555">• {html.escape(note)}</text>'
        )
    lines.append(
        f'<text x="{lx}" y="{ly + 168}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="10" fill="#555" font-style="italic">'
        f'Depleted source may become donor on same pad'
        f'{"; clear backup before ship path is required" if has_backup else ""}.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def write_aisle_layout_svg(plan: AisleLayoutPlan, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_aisle_layout_svg(plan), encoding="utf-8")
    return path
