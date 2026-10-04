"""Group-facing pick-day portal — floor timeline + KPIs + role views + wave actions."""

from __future__ import annotations

import csv
import html
import json
import re
from datetime import date
from pathlib import Path

from layout_advisor.pick_throughput import DEFAULT_PICKER_COUNT, PICK_CASES_PER_HOUR


class PortalDateConflict(RuntimeError):
    """Raised when a portal build would overwrite another ship day's artifact."""


def _sim_lp_display(sim_kpis: dict) -> str:
    """Layer-weighted sim LP (the day headline metric), falling back to the
    unweighted average only if the weighted value is absent."""
    val = sim_kpis.get("avg_lp_sim_pct_weighted")
    if val is None:
        val = sim_kpis.get("avg_lp_sim_pct")
    return f"{val:.1f}%" if val is not None else "-"


def _waits_display(sim_kpis: dict) -> str:
    """Prefer starve-hours when the sim step is known; otherwise show the raw
    count honestly as tick 'wait events', never a bare 'waits' headline (a tick
    count changes meaning with the tick length)."""
    waits = sim_kpis.get("total_waits")
    if waits is None:
        return "-"
    step = sim_kpis.get("sim_step_minutes")
    if step:
        return f"{waits * step / 60:.0f} starve-h"
    return f"{waits} wait events"


def _fill_display(sim_kpis: dict) -> str:
    """Fulfilment: layers picked vs the day's book, or '' when not measured.

    Every other KPI on this strip is READINESS — whether a SKU was on a face when
    its layers waved. Readiness cannot distinguish a floor that served its book
    from one that picked almost none of its book, which is how such a defect
    can survive unnoticed. Returns '' rather than a placeholder on
    the wave path, where no order ledger backs the number.
    """
    fill = sim_kpis.get("fill_rate_pct")
    if fill is None:
        return ""
    picked = sim_kpis.get("layers_picked")
    ordered = sim_kpis.get("layers_ordered")
    if picked is None or not ordered:
        return ""
    return f" · Shipped {fill:.0f}% of book ({picked:,}/{ordered:,} layers)"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hourly_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _peak_hour(rows: list[dict[str, str]], *, ship_date: date) -> dict[str, str] | None:
    if not rows:
        return None
    prefix = ship_date.isoformat()
    day_rows = [r for r in rows if r.get("as_of", "").startswith(prefix)]
    if not day_rows:
        day_rows = rows
    return max(day_rows, key=lambda r: int(r.get("waved_layers") or 0))


def _pending_completion_display(rows: list[dict[str, str]], *, ship_date: date) -> str:
    """Work-in-progress that never closes out, or '' when the column is absent.

    ``pending_completion_layers`` counts layers sitting on partial pallets pulled
    off pick faces (``removed_from_pick_at`` set) and never finished to empty.
    On a bad day it climbs all shift and ends at its maximum: nothing closes out.
    The column was written to the CSV but aggregated into no KPI and shown on no
    surface, which is how a floor that never finished a pallet still read healthy. Sourced from
    the hourly CSV the portal already loads, so it needs no chain regen.
    """
    prefix = ship_date.isoformat()
    day_rows = [r for r in rows if r.get("as_of", "").startswith(prefix)] or rows
    vals = []
    for row in day_rows:
        raw = row.get("pending_completion_layers")
        if raw in (None, ""):
            return ""
        try:
            vals.append(int(float(raw)))
        except (TypeError, ValueError):
            return ""
    if not vals or max(vals) <= 0:
        return ""
    tail = " (ends at max: nothing closed out)" if vals[-1] == max(vals) else ""
    return f" · WIP {vals[-1]:,} layers on unfinished pallets{tail}"


def _fmt_thruput(layers_per_hour: object, cases_per_hour: object) -> str:
    """Portal KPI text: ``12.3 lyr/h · 123 c/h`` or em dash when missing."""
    if layers_per_hour is None:
        return "-"
    try:
        lyr = float(layers_per_hour)
    except (TypeError, ValueError):
        return "-"
    if cases_per_hour is None:
        return f"{lyr:g} lyr/h"
    try:
        cph = float(cases_per_hour)
    except (TypeError, ValueError):
        return f"{lyr:g} lyr/h"
    return f"{lyr:g} lyr/h | {cph:,.0f} c/h"

def _frame_manifest(index: dict, *, ship_date: date) -> list[dict[str, object]]:
    frames = index.get("frames")
    if frames:
        return list(frames)
    labels = index.get("labels") or []
    return [{"label": label, "svg": "", "waved_skus": 0, "waved_layers": 0, "occupied": 0} for label in labels]


def resolve_day_dir(plans_root: Path, ship_date: date) -> Path:
    """Plan artifacts live at ``plans/<date>/``; ``plans/<date>/<date>/`` is legacy.

    **Flat wins.** The nested layout came from a fixed `write_daily_plan` bug
    (it appends ``<ship_date>/`` itself, so passing ``day_dir`` doubled it). The
    leftovers are frozen at whatever config was live when the bug was fixed —
    a doubled `plans/<date>/<date>/` path sat at a stale config hash with no fill
    measured, and this resolver used to prefer it over the current artifacts
    in the directory above. Nested is a fallback for old trees only.
    """
    flat = plans_root / ship_date.isoformat()
    if (flat / "wave_floor_index.json").is_file():
        return flat
    nested = flat / ship_date.isoformat()
    if (nested / "wave_floor_index.json").is_file():
        return nested
    return flat


def _load_wave_actions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _frame_filter_meta(frame: dict[str, object]) -> dict[str, object]:
    svg = str(frame.get("svg") or "")
    as_of = frame.get("as_of")
    if "tick_" in svg:
        meta: dict[str, object] = {"kind": "tick"}
        if as_of:
            meta["as_of"] = as_of
        return meta
    if "setup" in svg:
        return {"kind": "setup"}
    wave_index = frame.get("wave_index")
    if wave_index is not None:
        return {"kind": "wave", "wave_index": int(wave_index)}
    wave_match = re.search(r"wave_(\d+)", svg)
    if wave_match:
        return {"kind": "wave", "wave_index": int(wave_match.group(1))}
    hour_match = re.search(r"hour_(\d{2})(\d{2})", svg)
    if hour_match:
        return {"kind": "hour", "hour": int(hour_match.group(1))}
    return {"kind": "all"}


def resolve_hourly_sim_csv(
    day_dir: Path,
    ship_date: date,
    *,
    plans_root: Path | None = None,
) -> Path | None:
    """Find hourly sim CSV co-located with plan artifacts or legacy sim_runs paths."""
    candidates = [
        day_dir / f"pick_day_{ship_date.isoformat()}_hourly.csv",
    ]
    if plans_root is not None:
        candidates.append(
            plans_root.parent
            / "sim_runs"
            / "facility_75_25_rack_only"
            / f"pick_day_{ship_date.isoformat()}_hourly.csv"
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_order_schedule(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _frame_index_for_time(frames: list[dict[str, object]], target: str) -> int | None:
    """Nearest frame at or after *target* ISO timestamp (ship-day scrub)."""
    if not target or not frames:
        return None
    best_idx: int | None = None
    best_key = ""
    for i, frame in enumerate(frames):
        as_of = str(frame.get("as_of") or "")
        if as_of and as_of <= target and as_of >= best_key:
            best_key = as_of
            best_idx = i
    if best_idx is not None:
        return best_idx
    for i, frame in enumerate(frames):
        label = str(frame.get("label") or "")
        if target[:16] in label or target[:13] in label:
            return i
        if "Hour " in label and target[11:13] in label:
            return i
    return None


def _frame_index_for_wave(frames: list[dict[str, object]], wave_index: int) -> int | None:
    for i, frame in enumerate(frames):
        wi = frame.get("wave_index")
        if wi is not None and int(wi) == wave_index:
            return i
    token = f"wave_{wave_index:02d}"
    for i, frame in enumerate(frames):
        if token in str(frame.get("svg") or ""):
            return i
    return None


def render_pick_day_portal(
    *,
    ship_date: date,
    plans_root: Path,
    sim_csv: Path | None = None,
    display_name: str | None = None,
) -> str:
    day_dir = resolve_day_dir(plans_root, ship_date)
    if sim_csv is None:
        sim_csv = resolve_hourly_sim_csv(day_dir, ship_date, plans_root=plans_root)
    day_rel = day_dir.relative_to(plans_root).as_posix()
    summary = _load_json(day_dir / "daily_summary.json") or {}
    index = _load_json(day_dir / "wave_floor_index.json") or {}
    frames = _frame_manifest(index, ship_date=ship_date)
    for frame in frames:
        frame.update(_frame_filter_meta(frame))
    actions = _load_wave_actions(day_dir / "wave_actions.csv")
    order_schedule = _load_order_schedule(day_dir / "order_schedule.csv")
    hourly = _load_hourly_rows(sim_csv) if sim_csv else []
    peak = _peak_hour(hourly, ship_date=ship_date)

    # No site-shaped default: the portal renders any site, and a missing
    # layout must read as unknown rather than silently asserting a 70:30
    # module.
    layout_label = summary.get("layout") or "layout - "

    site_label = display_name or summary.get("display_name") or "Layer-pick site"
    title = f"{site_label} - Pick Day {ship_date.isoformat()}"
    ship_iso = ship_date.isoformat()
    picker_count = int(index.get("picker_count", DEFAULT_PICKER_COUNT))
    picker_label = (
        "1 layer-picker"
        if picker_count == 1
        else f"{picker_count} layer-pickers"
    )
    throughput_label = f"{PICK_CASES_PER_HOUR:,.0f} cases/hr"
    timeline_page = f"../{day_rel}/wave_floor_timeline.html"
    waves = summary.get("waves", [])
    max_layers = max((w.get("layer_demand", 0) for w in waves), default=1) or 1
    frames_json = json.dumps(frames)
    actions_json = json.dumps(actions)
    orders_json = json.dumps(order_schedule)
    hourly_json = json.dumps(
        [row for row in hourly if str(row.get("as_of", "")).startswith(ship_iso)]
    )
    minute_resolution = bool(index.get("minute_resolution"))
    tick_minutes_index = int(index.get("tick_minutes") or 60)

    peak_line = ""
    if peak:
        # projected wave LP (assumes every planned action fires) — label it so a
        # reader does not take it as the simulated LP for the peak hour.
        peak_line = (
            f"Sim peak @ {peak['as_of'][11:16]} - "
            f"{peak['waved_skus']} waved SKUs, {int(peak['waved_layers']):,} layers, "
            f"proj LP {float(peak.get('wave_lp_projected_pct', 0)):.1f}%"
        )
    peak_line += _pending_completion_display(hourly, ship_date=ship_date)

    sim_kpis = summary.get("sim_kpis") or {}
    sim_lp_line = ""
    if sim_kpis.get("avg_lp_sim_pct") is not None:
        travel_bit = ""
        if sim_kpis.get("peak_lp_ready_station_span") is not None:
            travel_bit = (
                f" · span {sim_kpis.get('peak_lp_ready_station_span', 0)}"
                f" · flank {sim_kpis.get('avg_flank_balance_pct', 100):.0f}%"
            )
        sim_lp_line = (
            f" · Sim LP {_sim_lp_display(sim_kpis)} (layer-wtd, carry-chain diagnostic: "
            f"not a service level) · {_waits_display(sim_kpis)}{travel_bit}"
            f"{_fill_display(sim_kpis)}"
        )

    wave_rows = []
    for display_idx, wave in enumerate(waves, start=1):
        send = wave.get("wave_send", "")[11:16]
        due = wave.get("wave_due", "")[11:16]
        layers = int(wave.get("layer_demand", 0))
        pct = int(100 * layers / max_layers)
        wave_index = int(wave.get("wave_index", -1))
        frame_idx = _frame_index_for_wave(frames, wave_index)
        jump = f' data-frame="{frame_idx}" class="wave-jump"' if frame_idx is not None else ""
        wave_rows.append(
            f"""<tr{jump}>
              <td>W{display_idx}</td>
              <td>{html.escape(send)}</td>
              <td>{html.escape(due)}</td>
              <td>{wave.get('sku_count', 0)}</td>
              <td>{layers:,}</td>
              <td><span class="bar" style="width:{pct}%"></span></td>
            </tr>"""
        )

    order_rows = []
    for row in order_schedule[:40]:
        start = row.get("start_at", "")[11:16]
        due = row.get("due_at", "")[11:16]
        key = row.get("order_key", "")[:12]
        frame_idx = _frame_index_for_time(frames, row.get("start_at", ""))
        jump = f' data-frame="{frame_idx}" class="order-jump"' if frame_idx is not None else ""
        order_rows.append(
            f"""<tr{jump}>
              <td title="{html.escape(row.get('order_key', ''))}">{html.escape(key)}…</td>
              <td>{html.escape(start)}</td>
              <td>{html.escape(due)}</td>
              <td>{row.get('sku_count', '-')}</td>
              <td>{row.get('total_demand_layers', '-')}</td>
              <td>{row.get('build_pallet_count', '-')}</td>
            </tr>"""
        )
    order_panel = ""
    if order_schedule:
        extra = len(order_schedule) - 40
        more = f"<p class='muted' style='font-size:11px'>+ {extra} more orders</p>" if extra > 0 else ""
        order_panel = f"""
      <div class="panel">
        <h2>Order schedule <span style="font-weight:400;color:#7e909c"> - click start</span></h2>
        <div class="panel-body" style="max-height:220px;overflow:auto">
          <table id="order-table">
            <thead><tr><th>Order</th><th>Start</th><th>Due</th><th>SKU</th><th>Lyr</th><th>Pallets</th></tr></thead>
            <tbody>{''.join(order_rows)}</tbody>
          </table>
          {more}
        </div>
      </div>"""

    hourly_bars = []
    for row in hourly:
        if not row.get("as_of", "").startswith(ship_iso):
            continue
        layers = int(row.get("waved_layers") or 0)
        if layers <= 0:
            continue
        hour = row["as_of"][11:16]
        pct = int(100 * layers / max(int(peak["waved_layers"]) if peak else layers, 1))
        span = row.get("lp_ready_station_span") or row.get("planned_station_span") or ""
        balance = row.get("flank_balance_pct") or ""
        tip_extra = ""
        if span:
            tip_extra = f" · span {span}"
        if balance:
            tip_extra += f" · flank {balance}%"
        hourly_bars.append(
            f'<div class="hour" title="{hour} - {layers:,} layers{tip_extra}">'
            f'<div class="hour-bar" style="height:{pct}%"></div>'
            f'<span>{hour[:2]}</span></div>'
        )

    face_swaps = (
        (sim_kpis.get("total_adds") or 0) + (sim_kpis.get("total_removes") or 0)
        if sim_kpis.get("total_adds") is not None or sim_kpis.get("total_removes") is not None
        else "-"
    )
    no_frames_msg = (
        '<p class="warn">Floor frames missing: run '
        "<code>python scripts/generate_daily_wave_plans.py --date "
        f"{ship_iso} --wave-layout</code></p>"
        if not frames or not frames[0].get("svg")
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #e9edf0; --ink: #141b22; --ink-2: #4a5e6c; --muted: #7e909c;
      --accent: #1c6fb0; --source: #1c6fb0; --source-ink: #0e4c7d; --donor: #b9761c;
      --panel: #ffffff; --border: #cfd6dc;
      --mono: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI Variable", "Segoe UI", Arial, sans-serif;
            background: var(--bg); color: var(--ink); }}
    header {{ background: var(--ink); color: #eef2f5; padding: 20px 24px 16px;
              display: flex; align-items: flex-start; gap: 16px;
              background-image: linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
              background-size: 22px 22px; border-bottom: 3px solid var(--source); }}
    .tb-mark {{ font-family: var(--mono); font-weight: 700; font-size: 18px; letter-spacing: .12em;
                color: #eaf2fa; border: 1.5px solid var(--source); padding: 7px 11px;
                border-radius: 2px; background: rgba(28,111,176,.16); }}
    header h1 {{ margin: 0; font-size: 20px; font-weight: 650; letter-spacing: -0.01em; }}
    header p {{ margin: 6px 0 0; font-size: 13px; color: #aab6c0; }}
    .banner {{ display: none; margin: 12px 24px 0; padding: 10px 14px; background: #fff3e0;
               border: 1px solid #ffb74d; border-radius: 6px; font-size: 13px; }}
    .banner.visible {{ display: block; }}
    .kpi-groups {{ display: grid; grid-template-columns: 5fr 6fr 2fr; gap: 16px; padding: 16px 24px 4px; }}
    @media (max-width: 1100px) {{ .kpi-groups {{ grid-template-columns: 1fr; }} }}
    .kpi-group > h3 {{ margin: 0 0 6px; font-size: 11px; text-transform: uppercase;
                       letter-spacing: 0.09em; color: var(--ink-2); font-weight: 700;
                       font-family: var(--mono); }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(108px, 1fr)); gap: 10px; }}
    .kpi {{ background: var(--panel); border: 1px solid var(--border); border-radius: 3px; padding: 10px 12px; }}
    .kpi label {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
                  color: var(--muted); }}
    .kpi strong {{ display: block; margin-top: 4px; font-size: 21px; font-weight: 650;
                   font-family: var(--mono); font-variant-numeric: tabular-nums; }}
    .roles {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 0 24px 12px; }}
    .roles button {{ padding: 8px 14px; border: 1px solid var(--border); border-radius: 2px;
                     background: var(--panel); cursor: pointer; font-size: 13px; }}
    .roles button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .role-note {{ margin: 0 24px 12px; padding: 10px 14px; background: #e7f0f7; border-radius: 8px;
                  font-size: 13px; line-height: 1.45; color: var(--ink-2); }}
    .layout {{ display: grid; grid-template-columns: minmax(340px, 400px) 1fr; gap: 16px; padding: 0 24px 24px;
               min-height: calc(100vh - 280px); }}
    @media (max-width: 1100px) {{ .layout {{ grid-template-columns: 1fr; }} }}
    .left-stack {{ display: flex; flex-direction: column; gap: 12px; }}
    tr.wave-jump {{ cursor: pointer; }}
    tr.wave-jump:hover {{ background: #e7f0f7; }}
    tr.wave-jump.active {{ background: #bbdefb; }}
    tr.order-jump {{ cursor: pointer; }}
    tr.order-jump:hover {{ background: #e8f5e9; }}
    .crew-list .action {{ padding: 8px 0 8px 8px; border-bottom: 1px solid #e6ecf0; line-height: 1.5;
                            display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }}
    .crew-list .muted {{ color: var(--muted); }}
    .action-chip {{ font-size: 10px; font-weight: 700; letter-spacing: 0.03em; padding: 2px 8px;
                    border-radius: 10px; background: #e7edf1; color: #4a5e6c; white-space: nowrap; }}
    .action-role-setup_crew .action-chip {{ background: #fff3e0; color: #e65100; }}
    .action-role-picker .action-chip {{ background: #e8f5e9; color: #1b5e20; }}
    .action-role-assist .action-chip {{ background: #f3e5f5; color: #6a1b9a; }}
    .crew-list .lyr {{ font-weight: 700; }}
    .crew-list .sku {{ font-family: Consolas, monospace; font-size: 11px; color: #455a64; }}
    .crew-list .route {{ font-family: Consolas, monospace; font-size: 11px; font-weight: 600; color: var(--accent); }}
    .trip-badge {{ font-size: 10px; padding: 1px 6px; border-radius: 8px; border: 1px dashed #8d6e63;
                   color: #6d4c41; white-space: nowrap; }}
    .crew-panel {{ margin: 0 24px 16px; }}
    .crew-cols {{ display: grid; grid-template-columns: 1fr 1fr; }}
    .crew-cols.solo {{ grid-template-columns: 1fr; }}
    @media (max-width: 900px) {{ .crew-cols {{ grid-template-columns: 1fr; }} }}
    .crew-col {{ padding: 10px 14px; min-width: 0; }}
    .crew-col + .crew-col {{ border-left: 1px solid var(--border); }}
    .crew-col.hidden {{ display: none; }}
    .crew-list {{ max-height: 300px; overflow: auto; font-size: 12px; }}
    .crew-head {{ margin: 0 0 6px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
                  font-weight: 700; padding: 4px 8px; border-radius: 6px; }}
    .picker-head {{ background: #e8f5e9; color: #1b5e20; }}
    .assist-head {{ background: #fff3e0; color: #e65100; }}
    .build-head {{ margin: 6px 0 2px; padding: 3px 8px; font-size: 11px; font-weight: 650;
                   color: #1b5e20; border-left: 3px solid #2e7d32; background: #f1f8f1;
                   border-radius: 0 6px 6px 0; cursor: pointer; }}
    .build-head:hover {{ background: #e0f0e0; }}
    .crew-list .detail {{ flex-basis: 100%; font-size: 11px; color: var(--muted); }}
    .action-role-setup_crew {{ border-left: 3px solid #fb8c00; padding-left: 8px; }}
    .action-role-picker {{ border-left: 3px solid #2e7d32; padding-left: 8px; }}
    .action-role-assist {{ border-left: 3px solid #6a1b9a; padding-left: 8px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 3px; overflow: hidden; }}
    .panel h2 {{ margin: 0; padding: 12px 14px; font-size: 13px; font-weight: 650;
                 border-bottom: 1px solid var(--border); }}
    .panel-body {{ padding: 12px 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ text-align: left; padding: 6px 4px; border-bottom: 1px solid #e6ecf0; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .bar {{ display: inline-block; height: 8px; background: var(--accent); border-radius: 4px; min-width: 4px; }}
    .hourly {{ display: flex; align-items: flex-end; gap: 4px; height: 80px; margin-top: 8px; }}
    .hour {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; height: 100%; }}
    .hour-bar {{ width: 100%; background: #86b6ef; border-radius: 3px 3px 0 0; min-height: 2px; }}
    .hour span {{ font-size: 10px; color: var(--muted); }}
    .floor-controls {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 14px; border-bottom: 1px solid var(--border); }}
    .floor-controls button {{ padding: 6px 10px; border: 1px solid var(--border); border-radius: 4px;
                              background: #fff; cursor: pointer; font-size: 11px; }}
    .floor-controls button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .floor-controls button.play-btn {{ background: #2e7d32; color: #fff; border-color: #2e7d32; font-weight: 600; }}
    .floor-controls button.play-btn.paused {{ background: #fff; color: #2e7d32; }}
    .floor-controls input[type=range] {{ flex: 1; min-width: 160px; }}
    .floor-controls select {{ font-size: 11px; padding: 5px 8px; border: 1px solid var(--border); border-radius: 4px; }}
    #floor-meta {{ padding: 8px 14px; font-size: 15px; font-weight: 600; color: var(--ink);
                   border-bottom: 1px solid var(--border); }}
    .roles button:focus-visible, .floor-controls button:focus-visible {{
      outline: 2px solid var(--accent); outline-offset: 2px; }}
    #path-panel {{ padding: 8px 14px 10px; font-size: 12px; border-bottom: 1px solid var(--border); background: #f1f8e9; }}
    #path-panel h3 {{ margin: 0 0 6px; font-size: 12px; color: #33691e; font-weight: 600; }}
    #path-list {{ display: flex; flex-wrap: wrap; gap: 6px; max-height: 88px; overflow: auto; }}
    #path-list .path-chip {{
      padding: 4px 8px; border-radius: 14px; border: 1px solid #a5d6a7; background: #fff;
      font-family: Consolas, monospace; font-size: 10px; cursor: pointer;
    }}
    #path-list .path-chip:hover,     #path-list .path-chip.active {{
      background: #c8e6c9; border-color: #2e7d32; box-shadow: 0 0 0 1px #2e7d32;
    }}
    #order-legend {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 8px; }}
    #order-legend .legend-chip {{
      display: inline-flex; align-items: center; gap: 5px; font-size: 10px;
      padding: 3px 8px; border-radius: 12px; border: 1px solid var(--border); background: #fff;
      cursor: pointer; font-family: Consolas, monospace;
    }}
    #order-legend .legend-chip .sw {{ width: 10px; height: 10px; border-radius: 50%; }}
    #floor-viewport .slot-highlight-order {{ stroke-width: 3 !important; }}
    #floor-viewport .lp-route.dimmed {{ opacity: 0.1 !important; }}
    #floor-viewport .lp-route-label.dimmed {{ opacity: 0.15 !important; }}
    #path-list .muted {{ color: var(--muted); font-size: 11px; }}
    #floor-viewport {{ padding: 12px 14px 20px; overflow: auto; min-height: 480px; position: relative; }}
    #floor-viewport svg {{ display: block; max-width: none; border: 1px solid var(--border); border-radius: 4px; }}
    #floor-viewport .slot-highlight {{ stroke: #ff6f00 !important; stroke-width: 2.5 !important; }}
    #floor-viewport .lp-route-active line,
    #floor-viewport .lp-route-active polyline {{
      stroke: #1b5e20 !important; stroke-width: 3.8 !important;
      filter: drop-shadow(0 0 2px #66bb6a);
    }}
    @keyframes lp-pulse {{
      0%, 100% {{ opacity: 0.75; }}
      50% {{ opacity: 1; }}
    }}
    #floor-viewport.playing .lp-route line,
    #floor-viewport.playing .lp-route polyline {{
      animation: lp-pulse 0.9s ease-in-out infinite;
    }}
    .hour {{ cursor: pointer; }}
    .hour:hover .hour-bar {{ background: #3987e5; }}
    .warn {{ color: #e65100; font-size: 13px; }}
    .links {{ padding: 8px 24px 16px; font-size: 12px; color: var(--muted); }}
    .links a {{ color: var(--accent); }}
  </style>
</head>
<body data-ship-date="{html.escape(ship_iso)}" data-plans-base="{html.escape(day_rel)}">
  <header>
    <span class="tb-mark">Layer Pick</span>
    <div>
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(layout_label)} rack · {picker_label} · {throughput_label} · {summary.get('visible_skus', '-')} SKUs · {int(summary.get('visible_layers') or 0):,} layers · setup @ {summary.get('setup_at', '22:00')[:16]}{html.escape(sim_lp_line)} · {peak_line or 'Hourly KPIs from co-located sim CSV'}</p>
    </div>
  </header>

  <div id="serve-banner" class="banner">
    For the full floor scrub, serve the portal's parent directory locally:
    <code>python -m http.server 8765 --bind 127.0.0.1</code>
    then open <a id="serve-url" href="http://127.0.0.1:8765/portal/index.html">http://127.0.0.1:8765/portal/index.html</a>
  </div>

  <div class="roles" id="roles">
    <button type="button" class="active" data-role="all">Whole crew</button>
    <button type="button" data-role="picker">Picker</button>
    <button type="button" data-role="assistant">Assistant (SKU moves)</button>
    <select id="order-filter" title="Filter actions by customer order" style="margin-left:8px;padding:6px 10px;border-radius:20px;font-size:13px">
      <option value="">All orders</option>
    </select>
  </div>
  <p class="role-note" id="role-note">
    <strong>Roles shown: 1 picker + 1 assistant, concurrent, not the roster</strong> (crew sizing is 3 shifts over the configured operating window). Picker rows = clamp picks (face → build pad); assistant
    rows = live dynamic slotting (staging, face swaps, rack promotes). <strong>Play</strong> steps pick hours;
    select an order to light its path; click a row to highlight slots.
  </p>

  <div class="layout">
    <div class="left-stack">
      <div class="panel">
        <h2>Wave schedule <span style="font-weight:400;color:#7e909c"> - click row</span></h2>
        <div class="panel-body" style="max-height:260px;overflow:auto">
          <table id="wave-table">
            <thead><tr><th>Wave</th><th>Send</th><th>Due</th><th>SKU</th><th>Lyr</th><th></th></tr></thead>
            <tbody>{''.join(wave_rows)}</tbody>
          </table>
        </div>
      </div>
      {order_panel}
      <div class="panel">
        <h2>Hourly waved demand (sim)</h2>
        <div class="panel-body">
          <div class="hourly">{''.join(hourly_bars) or '<p style="font-size:12px;color:#7e909c">No sim CSV linked.</p>'}</div>
        </div>
      </div>
    </div>
    <div class="panel">
      <h2>Floor: LP pathing <span style="font-weight:400;color:#7e909c">play through time</span></h2>
      {no_frames_msg}
      <div class="floor-controls">
        <button type="button" id="prev-frame" title="Previous frame">◀</button>
        <button type="button" id="play-frame" class="play-btn" title="Play / pause (Space)">▶ Play</button>
        <button type="button" id="next-frame" title="Next frame">▶</button>
        <select id="frame-mode" title="Frame set for scrub and play">
          <option value="time" selected>Time (hours)</option>
          <option value="all">All frames</option>
          <option value="waves">Waves only</option>
        </select>
        <select id="play-granularity" title="Play step size">
          <option value="pick" selected>By pick</option>
          <option value="hour">By hour</option>
        </select>
        <select id="play-speed" title="Play speed">
          <option value="250">Fast</option>
          <option value="500" selected>Normal</option>
          <option value="900">Slow</option>
        </select>
        <span id="frame-counter" style="font-size:11px;color:#7e909c;min-width:64px"></span>
      </div>
      <div class="floor-controls" id="frame-buttons"></div>
      <div class="floor-controls">
        <label for="frame-slider">Scrub:</label>
        <input type="range" id="frame-slider" min="0" max="{max(0, len(frames) - 1)}" value="0"/>
      </div>
      <div id="floor-meta"></div>
      <div id="path-panel">
        <h3>Layer-pick routes this frame <span id="path-mode-note" style="font-weight:400;color:#7e909c"></span></h3>
        <div id="order-legend"></div>
        <div id="path-list"><span class="muted">Scrub or play to see routes.</span></div>
      </div>
      <div id="floor-viewport"></div>
    </div>
  </div>

  <div class="kpi-groups">
    <section class="kpi-group">
      <h3>Setup &amp; plan</h3>
      <div class="kpis">
        <div class="kpi"><label>Faces staged</label><strong>{summary.get('setup_face_count', '-')}</strong></div>
        <div class="kpi"><label>Rack overflow</label><strong>{summary.get('setup_rack_count', '-')}</strong></div>
        <div class="kpi"><label>Pre-open targets</label><strong>{summary.get('preopen_donor_count', '-')}</strong></div>
        <div class="kpi"><label>Waves</label><strong>{summary.get('wave_count', '-')}</strong></div>
        <div class="kpi"><label>Planned actions</label><strong>{summary.get('total_actions', '-')}</strong></div>
      </div>
    </section>
    <section class="kpi-group">
      <h3>Sim day</h3>
      <div class="kpis">
        <div class="kpi" title="Layer-weighted sim LP: carry-chain diagnostic, not a customer service level"><label>Sim LP (layer-wtd)</label><strong>{_sim_lp_display(sim_kpis)}</strong></div>
        <div class="kpi" title="Worst single hour, material-weighted: the trough that drives service risk"><label>Sim worst hr (material)</label><strong>{sim_kpis.get('min_lp_sim_pct_material', '-')}{'%' if sim_kpis.get('min_lp_sim_pct_material') is not None else ''}</strong></div>
        <div class="kpi" title="Starve-hours if the sim step is known, else raw wait-event count (10-min ticks); a wait event is not a missed order"><label>Sim starve</label><strong>{_waits_display(sim_kpis)}</strong></div>
        <div class="kpi" title="ADD + REMOVE face swaps the dynamic slotter ordered: the assistant's reslot work"><label>Dyn slotting moves</label><strong>{face_swaps}</strong></div>
        <div class="kpi"><label>Peak pick span</label><strong>{sim_kpis.get('peak_lp_ready_station_span', '-')}</strong></div>
      </div>
    </section>
    <section class="kpi-group">
      <h3>Throughput</h3>
      <div class="kpis">
        <div class="kpi"><label>Avg LP thruput</label><strong>{_fmt_thruput(sim_kpis.get('avg_lp_ready_layers_per_hour'), sim_kpis.get('avg_lp_ready_cases_per_hour'))}</strong></div>
        <div class="kpi"><label>Peak LP thruput</label><strong>{_fmt_thruput(sim_kpis.get('peak_lp_ready_layers_per_hour'), sim_kpis.get('peak_lp_ready_cases_per_hour'))}</strong></div>
      </div>
    </section>
  </div>

  <div class="panel crew-panel">
    <h2>Crew queue <span id="action-count" style="font-weight:400;color:#7e909c"></span></h2>
    <div class="crew-cols">
      <div class="crew-col" id="crew-picker">
        <div class="crew-head picker-head" id="picker-count-head">Picker</div>
        <div class="crew-list" id="picker-list"><p class="muted">Scrub a wave to see picks.</p></div>
      </div>
      <div class="crew-col" id="crew-assist">
        <div class="crew-head assist-head" id="assist-count-head">Assistant (SKU moves = dynamic slotting)</div>
        <div class="crew-list" id="assist-list"><p class="muted">Scrub a wave to see moves.</p></div>
      </div>
    </div>
  </div>

  <div class="links">
    Full-screen floor: <a href="{html.escape(timeline_page)}" target="_blank" rel="noopener">wave_floor_timeline.html</a>
  </div>

  <script>
    const SHIP_DATE = document.body.dataset.shipDate;
    const PLANS_BASE = document.body.dataset.plansBase;
    const FRAMES = {frames_json};
    const ACTIONS = {actions_json};
    const ORDERS = {orders_json};
    const HOURLY = {hourly_json};
    const MINUTE_RESOLUTION = {'true' if minute_resolution else 'false'};
    const INDEX_TICK_MINUTES = {tick_minutes_index};
    const TICK_MINUTES = ORDERS.length && ORDERS[0].tick_minutes ? Number(ORDERS[0].tick_minutes) : INDEX_TICK_MINUTES;
    const notes = {{
      all: "Shared pick-day view: Play steps through pick hours; green arrows = LP routes on the floor.",
      setup: "Setup: the assistant's 22:00 shift (no separate setup crew): faces + rack overflow staged before waves.",
      pick: "Pick team: green routes show source face → target build. Click path chips to highlight slots.",
      assist: "Assist / reslot: rack promotes, face swaps: purple border in action list.",
      supervisor: "Supervisors: hourly chart + play through peak hours for LP span."
    }};

    function plansBase() {{
      if (location.protocol === 'http:' || location.protocol === 'https:') {{
        return '/plans/' + PLANS_BASE;
      }}
      return '../' + PLANS_BASE;
    }}

    let activeRole = 'all';
    let activeOrderKey = '';

    const orderFilter = document.getElementById('order-filter');
    if (ORDERS.length && orderFilter) {{
      ORDERS.forEach(o => {{
        const opt = document.createElement('option');
        opt.value = o.order_key;
        opt.textContent = (o.order_key || '').slice(0, 14) + '… @ ' + (o.start_at || '').slice(11, 16);
        orderFilter.appendChild(opt);
      }});
      orderFilter.onchange = () => {{
        activeOrderKey = orderFilter.value;
        renderActions(FRAMES[active]);
        renderPathPanel(FRAMES[active]);
        if (activeOrderKey) {{
          highlightOrder(activeOrderKey, pathsForFrame(FRAMES[active]));
        }} else {{
          clearFloorHighlights();
        }}
      }};
    }}

    function isPickerAction(action) {{
      return action.role === 'picker';
    }}

    function actionMatchesFrame(action, frame) {{
      if (!frame) return false;
      const wi = action.wave_index;
      if (frame.kind === 'setup') {{
        if (wi === '' || wi === '0') return true;
        // v4 order-ahead: work starts at order lock, so most picks execute the
        // evening BEFORE the ship day. The setup frame owns that overnight
        // window: otherwise 100+ picker actions match no Time-mode frame.
        const send = action.wave_send || '';
        return Boolean(send) && send.slice(0, 10) < SHIP_DATE;
      }}
      if (frame.kind === 'wave') {{
        return String(wi) === String(frame.wave_index);
      }}
      if (frame.kind === 'hour') {{
        const send = action.wave_send || '';
        // Overnight + setup-only actions live on the setup frame, not here.
        if (!send.startsWith(SHIP_DATE)) return false;
        const hour = parseInt(send.slice(11, 13), 10);
        if (Number.isNaN(hour)) return false;
        // Hour frames are sparse (every ~3h): each frame claims every send
        // since the previous hour frame, so no ship-day action falls between.
        let prev = -1, maxH = -1;
        FRAMES.forEach(f => {{
          if (f.kind !== 'hour' || typeof f.hour !== 'number') return;
          if (f.hour < frame.hour && f.hour > prev) prev = f.hour;
          if (f.hour > maxH) maxH = f.hour;
        }});
        // Final hour frame also claims late sends past its own hour.
        return hour > prev && (hour <= frame.hour || frame.hour === maxH);
      }}
      if (frame.kind === 'tick') {{
        const send = action.wave_send || '';
        const asOf = frame.as_of || '';
        if (!send || !asOf) return false;
        const tickEnd = new Date(asOf.replace(' ', 'T'));
        tickEnd.setMinutes(tickEnd.getMinutes() + TICK_MINUTES);
        const sendDt = new Date(send.replace(' ', 'T'));
        return sendDt >= new Date(asOf.replace(' ', 'T')) && sendDt < tickEnd;
      }}
      return true;
    }}

    function actionMatchesOrder(action) {{
      if (!activeOrderKey) return true;
      const order = ORDERS.find(o => o.order_key === activeOrderKey);
      if (!order) return true;
      const start = order.start_at || '';
      const due = order.due_at || '';
      const send = action.wave_send || '';
      if (send && start && due && send >= start && send <= due) return true;
      return false;
    }}

    function enrichPath(p) {{
      const base = {{
        from_slot: p.from_slot,
        to_slot: p.to_slot || '',
        layers: Number(p.layers) || 0,
        item_id: p.item_id || '',
        order_key: p.order_key || ''
      }};
      if (base.order_key) return base;
      const match = ACTIONS.find(a =>
        (a.action === 'LAYER_PICK' || a.action === 'LAYER_PICK_TO_BUILD') &&
        a.from_slot === base.from_slot &&
        (a.to_slot || '') === base.to_slot &&
        a.item_id === base.item_id
      );
      if (match && match.order_key) base.order_key = match.order_key;
      return base;
    }}

    const ORDER_COLORS = [
      '#1565C0','#C62828','#6A1B9A','#EF6C00','#00838F','#AD1457','#558B2F','#4527A0',
      '#D84315','#0277BD','#7B1FA2','#F9A825','#00695C','#5D4037','#283593','#C2185B'
    ];

    function colorForPath(p) {{
      const key = p.order_key ? ('order:' + p.order_key) : (p.to_slot ? ('donor:' + p.to_slot) : ('face:' + p.from_slot));
      let h = 0;
      for (let i = 0; i < key.length; i++) h = ((h << 5) - h) + key.charCodeAt(i);
      return ORDER_COLORS[Math.abs(h) % ORDER_COLORS.length];
    }}

    function shortOrder(key) {{
      return key ? key.slice(0, 8) : ' - ';
    }}

    function pathsForFrame(frame) {{
      if (!frame) return [];
      let paths = [];
      if (frame.layer_picks && frame.layer_picks.length) paths = frame.layer_picks;
      else {{
        paths = ACTIONS.filter(a =>
          actionMatchesFrame(a, frame) &&
          (a.action === 'LAYER_PICK' || a.action === 'LAYER_PICK_TO_BUILD') &&
          a.from_slot
        ).map(a => ({{
          from_slot: a.from_slot,
          to_slot: a.to_slot || '',
          layers: Number(a.layers) || 0,
          item_id: a.item_id || '',
          order_key: a.order_key || ''
        }}));
      }}
      return paths.map(enrichPath);
    }}

    function hourlySpanForFrame(frame) {{
      if (!frame || !frame.as_of) return null;
      const hourKey = frame.as_of.slice(0, 13);
      const row = HOURLY.find(h => (h.as_of || '').startsWith(hourKey));
      if (!row) return null;
      return {{
        span: row.lp_ready_station_span || row.planned_station_span || '',
        flank: row.flank_balance_pct || ''
      }};
    }}

    function routeKey(p) {{
      return (p.order_key || '') + '|' + (p.from_slot || '') + '|' + (p.to_slot || '') + '|' + (p.item_id || '');
    }}

    function clearFloorHighlights() {{
      const root = viewport.querySelector('svg');
      if (!root) return;
      root.querySelectorAll('.slot-highlight, .slot-highlight-order').forEach(el => {{
        el.classList.remove('slot-highlight', 'slot-highlight-order');
        el.style.stroke = '';
      }});
      root.querySelectorAll('.lp-route-active').forEach(el => el.classList.remove('lp-route-active'));
      root.querySelectorAll('.lp-route.dimmed, .lp-route-label.dimmed').forEach(el => el.classList.remove('dimmed'));
    }}

    function highlightRoute(fromSlot, toSlot, itemId, orderKey, soloOnly) {{
      const root = viewport.querySelector('svg');
      if (!root) return;
      clearFloorHighlights();
      const rid = routeKey({{
        from_slot: fromSlot,
        to_slot: toSlot || '',
        item_id: itemId || '',
        order_key: orderKey || ''
      }});
      const color = colorForPath({{
        order_key: orderKey || '',
        to_slot: toSlot || '',
        from_slot: fromSlot || ''
      }});
      [fromSlot, toSlot].filter(Boolean).forEach(slotId => {{
        const cell = root.querySelector('[data-slot-id="' + slotId + '"]');
        if (cell) {{
          cell.classList.add('slot-highlight', 'slot-highlight-order');
          cell.style.stroke = color;
        }}
      }});
      root.querySelectorAll('.lp-route, .lp-route-label').forEach(el => {{
        const match = el.getAttribute('data-route-id') === rid;
        if (match) {{
          el.classList.add('lp-route-active');
          el.classList.remove('dimmed');
        }} else if (soloOnly) {{
          el.classList.add('dimmed');
        }}
      }});
      document.querySelectorAll('#path-list .path-chip').forEach(chip => {{
        chip.classList.toggle('active', chip.dataset.routeId === rid);
      }});
    }}

    function highlightOrder(orderKey, paths) {{
      // Whole-order view: every route of the order lights up: source faces AND
      // destination build pads: so the full pick path is visible at once.
      const root = viewport.querySelector('svg');
      if (!root) return;
      clearFloorHighlights();
      const matches = paths.filter(p => (p.order_key || '') === (orderKey || ''));
      if (!matches.length) return;
      matches.forEach(p => {{
        const color = colorForPath(p);
        [p.from_slot, p.to_slot].filter(Boolean).forEach(slotId => {{
          const cell = root.querySelector('[data-slot-id="' + slotId + '"]');
          if (cell) {{
            cell.classList.add('slot-highlight', 'slot-highlight-order');
            cell.style.stroke = color;
          }}
        }});
      }});
      const rids = new Set(matches.map(routeKey));
      root.querySelectorAll('.lp-route, .lp-route-label').forEach(el => {{
        if (rids.has(el.getAttribute('data-route-id'))) {{
          el.classList.add('lp-route-active');
          el.classList.remove('dimmed');
        }} else {{
          el.classList.add('dimmed');
        }}
      }});
      document.querySelectorAll('#path-list .path-chip').forEach(chip => {{
        chip.classList.toggle('active', rids.has(chip.dataset.routeId));
      }});
    }}

    function renderOrderLegend(paths) {{
      const legend = document.getElementById('order-legend');
      if (!legend) return;
      const byOrder = new Map();
      paths.forEach(p => {{
        const key = p.order_key || ('dest:' + (p.to_slot || p.from_slot));
        if (!byOrder.has(key)) byOrder.set(key, {{ order: p.order_key, color: colorForPath(p), count: 0 }});
        byOrder.get(key).count += 1;
      }});
      if (!byOrder.size) {{
        legend.innerHTML = '';
        return;
      }}
      legend.innerHTML = Array.from(byOrder.values()).map(row => {{
        const label = row.order ? ('Ord ' + shortOrder(row.order)) : 'No order';
        return '<button type="button" class="legend-chip" data-order="' + (row.order || '') + '">' +
          '<span class="sw" style="background:' + row.color + '"></span>' + label +
          ' (' + row.count + ')</button>';
      }}).join('');
      legend.querySelectorAll('.legend-chip').forEach(chip => {{
        chip.onclick = () => highlightOrder(chip.dataset.order, paths);
      }});
    }}

    function renderPathPanel(frame, soloPick) {{
      const list = document.getElementById('path-list');
      const note = document.getElementById('path-mode-note');
      const paths = pathsForFrame(frame);
      if (note) {{
        note.textContent = paths.length ? (' - ' + paths.length + ' route(s), colors = order → build') : '';
      }}
      renderOrderLegend(paths);
      if (!paths.length) {{
        list.innerHTML = '<span class="muted">No layer picks this frame.</span>';
        return;
      }}
      const sorted = paths.slice().sort((a, b) => (b.layers || 0) - (a.layers || 0));
      list.innerHTML = sorted.map(p => {{
        const color = colorForPath(p);
        const ord = p.order_key ? ('Ord ' + shortOrder(p.order_key) + ' · ') : '';
        const label = ord + p.from_slot + (p.to_slot ? ' → ' + p.to_slot : '') +
          ' · ' + (p.layers || 0) + 'L · ' + (p.item_id || '');
        const rid = routeKey(p);
        return '<button type="button" class="path-chip" data-route-id="' + rid + '" ' +
          'style="border-color:' + color + '; border-left-width:4px" ' +
          'data-from="' + (p.from_slot || '') + '" data-to="' + (p.to_slot || '') + '" ' +
          'data-item="' + (p.item_id || '') + '" data-order="' + (p.order_key || '') + '">' + label + '</button>';
      }}).join('');
      list.querySelectorAll('.path-chip').forEach(chip => {{
        chip.onclick = () => highlightRoute(
          chip.dataset.from, chip.dataset.to, chip.dataset.item, chip.dataset.order, true
        );
      }});
      if (soloPick) {{
        highlightRoute(soloPick.from_slot, soloPick.to_slot, soloPick.item_id, soloPick.order_key, true);
      }} else if (sorted.length) {{
        const top = sorted[0];
        highlightRoute(top.from_slot, top.to_slot, top.item_id, top.order_key, false);
      }}
    }}

    function renderActions(frame) {{
      const pickerEl = document.getElementById('picker-list');
      const assistEl = document.getElementById('assist-list');
      const count = document.getElementById('action-count');
      const rows = ACTIONS.filter(a => actionMatchesFrame(a, frame) && actionMatchesOrder(a));
      count.textContent = '(' + rows.length + ')';
      const ACTION_LABELS = {{
        STAGE_SOURCE_FACE: 'Stage face', STAGE_RACK: 'Stage rack', CONVERT_PAD: 'Convert pad',
        LAYER_PICK: 'Pick layers', LAYER_PICK_TO_BUILD: 'Pick → build',
        RELOCATE_TO_RACK: 'Relocate to rack', RACK_PROMOTE: 'Promote from rack',
        EXIT_DONOR_TO_WRAP: 'Target → wrap', CLOSE_SOURCE: 'Close face',
        OPEN_DONOR_BUILD: 'Open target build', FREEZER_PULL: 'Freezer pull',
      }};
      const renderRows = rs => rs.map(a => renderActionRow(a)).join('');
      // Picker constructs one order at a time: group picks under the build
      // they feed (order + destination pad), in work sequence.
      const renderPickerGroups = rs => {{
        if (!rs.length) return '';
        const groups = new Map();
        rs.forEach(a => {{
          const key = (a.order_key || '') + '|' + (a.to_slot || '');
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key).push(a);
        }});
        return Array.from(groups.values()).map(g => {{
          const first = g[0];
          const layers = g.reduce((s, a) => s + (Number(a.layers) || 0), 0);
          const ordBit = first.order_key ? 'Ord ' + first.order_key.slice(0, 6) : 'No order';
          const dest = first.to_slot ? ' @ ' + first.to_slot : '';
          return '<div class="build-head" data-order="' + (first.order_key || '') + '" ' +
            'title="Click to light this order\\'s full path on the floor">Building ' + ordBit + dest +
            ' · ' + g.length + ' pick(s) · ' + layers + ' lyr</div>' +
            g.map(renderActionRow).join('');
        }}).join('');
      }};
      // Two-worker split: picker column groups picks by the build under
      // construction; assistant column lists SKU moves (= dynamic slotting).
      const pickerRows = rows.filter(isPickerAction);
      const assistRows = rows.filter(a => !isPickerAction(a));
      pickerEl.innerHTML = renderPickerGroups(pickerRows) || '<p class="muted">No picks this frame.</p>';
      assistEl.innerHTML = renderRows(assistRows) || '<p class="muted">No moves this frame.</p>';
      document.getElementById('picker-count-head').textContent =
        'Picker: ' + pickerRows.length + ' action(s)';
      document.getElementById('assist-count-head').textContent =
        'Assistant (SKU moves = dynamic slotting): ' + assistRows.length + ' action(s)';
      // Role buttons show/hide columns instead of filtering rows.
      document.getElementById('crew-picker').classList.toggle('hidden', activeRole === 'assistant');
      document.getElementById('crew-assist').classList.toggle('hidden', activeRole === 'picker');
      document.querySelector('.crew-cols').classList.toggle('solo', activeRole !== 'all');
      const list = document.querySelector('.crew-cols');
      list.querySelectorAll('.build-head').forEach(head => {{
        head.onclick = () => highlightOrder(head.dataset.order, pathsForFrame(frame));
      }});
      function renderActionRow(a) {{
        const role = a.role || 'ops';
        const fromTo = [a.from_slot, a.to_slot].filter(Boolean).join(' → ');
        const isPick = a.action === 'LAYER_PICK' || a.action === 'LAYER_PICK_TO_BUILD';
        const pickAttrs = isPick && a.from_slot
          ? ' data-from="' + a.from_slot + '" data-to="' + (a.to_slot || '') + '" data-item="' + (a.item_id || '') + '" data-order="' + (a.order_key || '') + '" class="pick-action"'
          : '';
        const label = ACTION_LABELS[a.action] || a.action;
        const trip = a.trip
          ? '<span class="trip-badge" title="Clamp trip: combine bites before drop">trip ' + a.trip + '</span>'
          : '';
        return '<div class="action action-role-' + role + '"' + pickAttrs + '>' +
          '<span class="action-chip" data-code="' + a.action + '" title="' + a.action + ' · ' + role + '">' + label + '</span>' +
          (a.layers ? '<span class="lyr">' + a.layers + ' lyr</span>' : '') +
          (a.item_id ? '<span class="sku">' + a.item_id + '</span>' : '') +
          (fromTo ? '<span class="route">' + fromTo + '</span>' : '') +
          trip +
          (a.detail ? '<div class="detail">' + a.detail + '</div>' : '') +
          '</div>';
      }}
      list.querySelectorAll('.pick-action').forEach(row => {{
        row.style.cursor = 'pointer';
        row.onclick = () => highlightRoute(
          row.dataset.from, row.dataset.to, row.dataset.item, row.dataset.order || '', true
        );
      }});
    }}

    function assetUrl(relativePath) {{
      return plansBase() + '/' + relativePath;
    }}

    function shortLabel(frame, index) {{
      if (index === 0) return 'Setup';
      const label = frame.label || '';
      if (frame.kind === 'tick') return label.split(' - ')[0] || String(index);
      if (label.includes('Hour ')) return label.replace('Hour ', 'H').split(' - ')[0];
      if (label.includes('Wave ')) return label.replace('Wave ', 'W').split(' - ')[0];
      return String(index);
    }}

    function frameModeMatches(frame, mode) {{
      if (mode === 'all') return true;
      if (mode === 'waves') return frame.kind === 'wave' || frame.kind === 'setup';
      // time: setup + hourly bridge + tick: skip wave-send snapshots
      return frame.kind === 'setup' || frame.kind === 'hour' || frame.kind === 'tick';
    }}

    function buildPlayIndices(mode) {{
      return FRAMES.map((f, i) => i).filter(i => frameModeMatches(FRAMES[i], mode));
    }}

    let frameMode = 'time';
    let playIndices = buildPlayIndices(frameMode);
    let playPos = 0;
    let active = 0;
    let playTimer = null;
    let pickSteps = [];
    let pickStepPos = 0;
    let soloPick = null;

    const buttons = document.getElementById('frame-buttons');
    const slider = document.getElementById('frame-slider');
    const meta = document.getElementById('floor-meta');
    const viewport = document.getElementById('floor-viewport');
    const frameCounter = document.getElementById('frame-counter');
    const playBtn = document.getElementById('play-frame');
    const frameModeSelect = document.getElementById('frame-mode');
    const playSpeedSelect = document.getElementById('play-speed');
    const playGranularitySelect = document.getElementById('play-granularity');

    function playGranularity() {{
      return playGranularitySelect ? playGranularitySelect.value : 'pick';
    }}

    function rebuildPickSteps() {{
      pickSteps = [];
      playIndices.forEach(fi => {{
        pathsForFrame(FRAMES[fi]).forEach(p => pickSteps.push({{ frameIndex: fi, pick: p }}));
      }});
      pickStepPos = Math.max(0, Math.min(pickStepPos, Math.max(0, pickSteps.length - 1)));
    }}

    function syncPlayPos() {{
      if (playGranularity() === 'pick' && pickSteps.length) {{
        if (frameCounter) {{
          frameCounter.textContent = (pickStepPos + 1) + ' / ' + pickSteps.length + ' picks';
        }}
        return;
      }}
      const idx = playIndices.indexOf(active);
      playPos = idx >= 0 ? idx : 0;
      if (frameCounter) {{
        frameCounter.textContent = (playPos + 1) + ' / ' + playIndices.length + ' frames';
      }}
    }}

    function rebuildFrameButtons() {{
      buttons.innerHTML = '';
      playIndices.forEach((frameIdx, pos) => {{
        const frame = FRAMES[frameIdx];
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = shortLabel(frame, frameIdx);
        btn.title = frame.label || '';
        btn.onclick = () => showFrame(frameIdx);
        btn.dataset.pos = String(pos);
        buttons.appendChild(btn);
      }});
    }}

    function refreshFrameMode() {{
      frameMode = frameModeSelect ? frameModeSelect.value : 'time';
      playIndices = buildPlayIndices(frameMode);
      if (!playIndices.length) playIndices = FRAMES.map((_, i) => i);
      slider.max = String(Math.max(0, playIndices.length - 1));
      rebuildFrameButtons();
      rebuildPickSteps();
      const idx = playIndices.indexOf(active);
      if (idx < 0) showFrame(playIndices[0]);
      else syncPlayPos();
    }}

    function mountSvg(svgText, pickFocus) {{
      viewport.innerHTML = svgText;
      viewport.classList.toggle('playing', !!playTimer);
      renderPathPanel(FRAMES[active], pickFocus || soloPick);
    }}

    function showFrame(index, options) {{
      options = options || {{}};
      soloPick = options.soloPick || null;
      if (!FRAMES.length || !FRAMES[index] || !FRAMES[index].svg) {{
        meta.textContent = 'No floor frames: regenerate wave layout.';
        return;
      }}
      active = index;
      const pos = playIndices.indexOf(index);
      if (pos >= 0) {{
        playPos = pos;
        slider.value = pos;
      }}
      syncPlayPos();
      const frame = FRAMES[index];
      const url = assetUrl(frame.svg);
      const spanInfo = hourlySpanForFrame(frame);
      let metaText = frame.label + ' | waved ' + frame.waved_skus + ' SKUs, ' +
        Number(frame.waved_layers).toLocaleString() + ' layers | ' + frame.occupied + ' occupied';
      if (frame.layer_pick_layers) {{
        metaText += ' | LP ' + frame.layer_pick_count + ' picks, ' +
          Number(frame.layer_pick_layers).toLocaleString() + ' layers (colored by order → build)';
      }}
      if (spanInfo && spanInfo.span) {{
        metaText += ' | pick span ' + spanInfo.span + ' stations';
        if (spanInfo.flank) metaText += ' · flank ' + spanInfo.flank + '%';
      }}
      meta.textContent = metaText;
      document.querySelectorAll('#frame-buttons button').forEach(btn => {{
        btn.classList.toggle('active', Number(btn.dataset.pos) === playPos);
      }});
      document.querySelectorAll('#wave-table tr.wave-jump').forEach(row => {{
        row.classList.toggle('active', Number(row.dataset.frame) === index);
      }});
      renderActions(frame);
      renderPathPanel(frame, soloPick);

      function showError() {{
        viewport.innerHTML = '<p class="warn">Could not load floor SVG at ' + url +
          '. Serve the portal over HTTP (python -m http.server 8765) for ' + SHIP_DATE + '.</p>';
        document.getElementById('serve-banner').classList.add('visible');
      }}

      if (location.protocol === 'http:' || location.protocol === 'https:') {{
        fetch(url)
          .then(resp => {{
            if (!resp.ok) throw new Error(String(resp.status));
            return resp.text();
          }})
          .then(svg => mountSvg(svg, soloPick))
          .catch(showError);
        return;
      }}

      viewport.innerHTML = '';
      const obj = document.createElement('object');
      obj.type = 'image/svg+xml';
      obj.data = url;
      obj.style.minWidth = '900px';
      obj.style.minHeight = '400px';
      viewport.appendChild(obj);
      renderPathPanel(frame, soloPick);
    }}

    function stepPlay(delta) {{
      if (playGranularity() === 'pick' && pickSteps.length) {{
        pickStepPos = Math.max(0, Math.min(pickSteps.length - 1, pickStepPos + delta));
        const step = pickSteps[pickStepPos];
        showFrame(step.frameIndex, {{ soloPick: step.pick }});
        return;
      }}
      if (!playIndices.length) return;
      playPos = Math.max(0, Math.min(playIndices.length - 1, playPos + delta));
      soloPick = null;
      showFrame(playIndices[playPos]);
    }}

    function stopPlay() {{
      if (playTimer) {{
        clearInterval(playTimer);
        playTimer = null;
      }}
      viewport.classList.remove('playing');
      if (playBtn) {{
        playBtn.textContent = '▶ Play';
        playBtn.classList.remove('paused');
      }}
    }}

    function startPlay() {{
      stopPlay();
      if (!playIndices.length) return;
      if (playBtn) {{
        playBtn.textContent = '⏸ Pause';
        playBtn.classList.add('paused');
      }}
      viewport.classList.add('playing');
      const ms = Number(playSpeedSelect ? playSpeedSelect.value : 700);
      playTimer = setInterval(() => {{
        if (playGranularity() === 'pick' && pickSteps.length) {{
          if (pickStepPos >= pickSteps.length - 1) {{
            stopPlay();
            return;
          }}
          stepPlay(1);
          return;
        }}
        if (playPos >= playIndices.length - 1) {{
          stopPlay();
          return;
        }}
        stepPlay(1);
      }}, ms);
    }}

    if (frameModeSelect) frameModeSelect.onchange = () => refreshFrameMode();
    if (playGranularitySelect) playGranularitySelect.onchange = () => {{
      rebuildPickSteps();
      syncPlayPos();
    }};
    document.getElementById('prev-frame').onclick = () => stepPlay(-1);
    document.getElementById('next-frame').onclick = () => stepPlay(1);
    playBtn.onclick = () => {{
      if (playTimer) stopPlay();
      else startPlay();
    }};

    slider.oninput = () => {{
      const pos = Number(slider.value);
      if (playIndices[pos] != null) showFrame(playIndices[pos]);
    }};

    document.querySelectorAll('#roles button').forEach(btn => {{
      btn.onclick = () => {{
        document.querySelectorAll('#roles button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeRole = btn.dataset.role;
        document.getElementById('role-note').textContent = notes[btn.dataset.role];
        renderActions(FRAMES[active]);
      }};
    }});

    function keyActivatable(el, fn) {{
      el.onclick = fn;
      el.tabIndex = 0;
      el.setAttribute('role', 'button');
      el.addEventListener('keydown', ev => {{
        if (ev.key === 'Enter' || ev.key === ' ') {{ ev.preventDefault(); fn(); }}
      }});
    }}

    document.querySelectorAll('#wave-table tr.wave-jump').forEach(row => {{
      keyActivatable(row, () => showFrame(Number(row.dataset.frame)));
    }});

    document.querySelectorAll('#order-table tr.order-jump').forEach(row => {{
      keyActivatable(row, () => showFrame(Number(row.dataset.frame)));
    }});

    document.querySelectorAll('.hour').forEach((bar, i) => {{
      keyActivatable(bar, () => {{
        const hour = bar.querySelector('span');
        if (!hour) return;
        const hh = hour.textContent.padStart(2, '0');
        const target = FRAMES.findIndex(f =>
          (f.as_of || '').includes(' ' + hh + ':') || (f.label || '').includes(hh + ':00')
        );
        if (target >= 0) showFrame(target);
      }});
    }});

    document.addEventListener('keydown', (e) => {{
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
      if (e.key === 'ArrowLeft') stepPlay(-1);
      if (e.key === 'ArrowRight') stepPlay(1);
      if (e.key === ' ') {{
        e.preventDefault();
        playBtn.click();
      }}
    }});

    if (location.protocol === 'file:') {{
      document.getElementById('serve-banner').classList.add('visible');
    }}

    refreshFrameMode();
    rebuildPickSteps();
    if (FRAMES.length) {{
      const start = playIndices.find(i => FRAMES[i].kind === 'hour') ?? playIndices[0];
      showFrame(start);
      if (playGranularity() === 'pick' && pickSteps.length) {{
        pickStepPos = 0;
        showFrame(pickSteps[0].frameIndex, {{ soloPick: pickSteps[0].pick }});
      }}
    }}
  </script>
</body>
</html>"""


_PORTAL_SHIP_DATE_RE = re.compile(r'data-ship-date="(\d{4}-\d{2}-\d{2})"')


def portal_ship_date(index_path: Path) -> date | None:
    """Ship date a built portal ``index.html`` belongs to, or ``None``."""
    try:
        # Whole file: the <body> tag sits behind a large inline <style> block,
        # so any fixed head slice is a bug waiting for the CSS to grow.
        text = Path(index_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = _PORTAL_SHIP_DATE_RE.search(text)
    return date.fromisoformat(match.group(1)) if match else None


def write_pick_day_portal(
    *,
    ship_date: date,
    plans_root: Path,
    portal_root: Path,
    sim_csv: Path | None = None,
    display_name: str | None = None,
    force: bool = False,
) -> Path:
    """Build the portal into *portal_root*, refusing to clobber another day.

    ``portal_root/index.html`` is an artifact pinned to one ship date (the
    stress day). Building a different date used to overwrite it
    silently — a `git checkout` was needed to notice and undo. Pass ``force`` to
    replace it deliberately, or point ``portal_root`` somewhere scratch.
    """
    out = portal_root / "index.html"
    existing = portal_ship_date(out)
    if existing is not None and existing != ship_date and not force:
        raise PortalDateConflict(
            f"{out} is the built portal for {existing}, not {ship_date}. "
            f"Use --portal-dir to write elsewhere, or --force to replace it."
        )
    portal_root.mkdir(parents=True, exist_ok=True)
    content = render_pick_day_portal(
        ship_date=ship_date,
        plans_root=plans_root,
        sim_csv=sim_csv,
        display_name=display_name,
    )
    out.write_text(content, encoding="utf-8")
    day_dir = resolve_day_dir(plans_root, ship_date)
    day_portal = day_dir / "portal.html"
    day_portal.write_text(content, encoding="utf-8")
    return out
