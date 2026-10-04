"""Build and export a full pick-day plan: setup slotting + per-wave actions."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from layout_advisor.planning.dynamic_slotting import SetupAssignment
from layout_advisor.planning.pad_role_schedule import PadRoleSwap
from layout_advisor.planning.wave_actions import (
    WaveAction,
    actions_from_simulation_step,
    layer_pick_actions_from_events,
    resequence_assist_min_travel,
)
from layout_advisor.simulate.replay import (
    PickDayReplay,
    ReplayConfig,
    _spread_orders,
    replay_sim_kpis,
    run_pick_day_backtest,
    write_pick_day_hourly_csv,
)
from slotting.backtest import HourlyStagingBoost
from slotting.models import OrderLine
from slotting.wave_release import bucket_orders_by_ship_day, order_ship_date, split_day_by_native_due, split_day_into_wave_buckets, active_wave_bucket


def replay_config_hash(cfg: ReplayConfig) -> str:
    """Stable hash of the effective replay config, stamped into
    daily_summary.json so tests can flag committed plans generated
    under a different policy (staleness guard)."""
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def provenance(cfg: ReplayConfig) -> dict[str, object]:
    """Which config and which code produced this artifact.

    `plans/*/daily_summary.json` carries `config_hash`. Everything written under
    `sim_runs/` carried nothing, so a JSON there could not be told apart from one
    produced by a completely different policy — and a script that builds a bare
    `ReplayConfig` from argparse defaults gets the legacy 75:25 module. Stamp
    this into any payload a person or another script will later read as if it
    were the reference config.
    """
    from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG

    cfg_hash = replay_config_hash(cfg)
    return {
        "config_hash": cfg_hash,
        "code_fingerprint": replay_code_fingerprint(),
        "is_reference_config": cfg_hash == replay_config_hash(REFERENCE_REPLAY_CONFIG),
        "source_slots": cfg.source_slots,
        "donor_slots": cfg.donor_slots,
    }


def warn_if_not_reference(cfg: ReplayConfig, *, label: str) -> dict[str, object]:
    """Print a loud banner when *cfg* is not the reference config. Returns the provenance.

    Two fixes exist for bare-config scripts: rewire them to
    `REFERENCE_REPLAY_CONFIG`, or make them print the config hash so a stale run
    is loud. This is the second — it changes no behaviour, so no KPI can move,
    and it is one call per script rather than a policy migration that has to be
    measured against the benchmark first.
    """
    prov = provenance(cfg)
    if not prov["is_reference_config"]:
        print(
            f"\n!! {label}: NOT the reference config "
            f"(config {prov['config_hash'][:8]}, {cfg.source_slots}:{cfg.donor_slots}).\n"
            "!! Numbers from this run are NOT comparable to the chain benchmark. Use\n"
            "!! scripts/generate_daily_wave_plans.py for reference-config runs.\n",
            flush=True,
        )
    return prov


_SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src


def replay_code_fingerprint(roots: tuple[Path, ...] | None = None) -> str:
    """Fingerprint of src/ *.py bytes, stamped next to config_hash.

    Catches what config_hash cannot: a sim-behavior code change without a
    chain regen, which leaves committed artifacts stale while they still look
    current (and can send a later investigation bisecting a fake regression)."""
    if roots is None:
        roots = (_SRC_ROOT / "slotting", _SRC_ROOT / "layout_advisor")
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class WavePlan:
    wave_index: int
    wave_send: datetime
    wave_due: datetime
    sku_count: int
    layer_demand: int
    actions: tuple[WaveAction, ...]


@dataclass(frozen=True)
class DailyWavePlan:
    ship_date: date
    setup_at: datetime
    layout: str
    visible_skus: int
    visible_layers: int
    setup_assignments: tuple[SetupAssignment, ...]
    preopen_donors: tuple[tuple[str, int, str], ...]
    pad_role_swaps: tuple[PadRoleSwap, ...]
    waves: tuple[WavePlan, ...]
    all_actions: tuple[WaveAction, ...]

    @property
    def wave_count(self) -> int:
        return len(self.waves)

    @property
    def total_actions(self) -> int:
        return len(self.all_actions)


def _setup_actions(
    ship_date: date,
    setup_at: datetime,
    assignments: list[SetupAssignment],
    preopen: list[tuple[str, int, str]],
) -> list[WaveAction]:
    actions: list[WaveAction] = []
    seq = 1
    for row in assignments:
        code = {
            "source_face": "STAGE_SOURCE_FACE",
            "rack": "STAGE_RACK",
            "freezer_hold": "FREEZER_HOLD",
        }[row.location]
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=setup_at,
                wave_due=None,
                wave_index=None,
                sequence=seq,
                action=code,
                role=row.role,
                item_id=row.item_id,
                layers=row.layers,
                from_slot=None,
                to_slot=row.slot_id,
                detail=row.notes,
            )
        )
        seq += 1
    for item_id, layers, donor_slot in preopen:
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=setup_at,
                wave_due=None,
                wave_index=0,
                sequence=seq,
                action="PREOPEN_DONOR",
                role="setup_crew",
                item_id=item_id,
                layers=layers,
                from_slot=None,
                to_slot=donor_slot,
                detail=f"Pre-open donor build on {donor_slot} before first wave.",
            )
        )
        seq += 1
    return actions


def _pad_role_actions(
    ship_date: date,
    swaps: list[PadRoleSwap],
    *,
    sequence_start: int,
) -> list[WaveAction]:
    actions: list[WaveAction] = []
    seq = sequence_start
    for swap in swaps:
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=swap.effective_at,
                wave_due=None,
                wave_index=None,
                sequence=seq,
                action="CONVERT_PAD",
                role="setup_crew",
                item_id=None,
                layers=None,
                from_slot=swap.slot_id if swap.from_role == "source" else swap.slot_id,
                to_slot=f"{swap.to_role}@{swap.flank}-ST{swap.station}",
                detail=(
                    f"Convert {swap.slot_id} from {swap.from_role} to {swap.to_role} "
                    f"at station {swap.station} ({swap.flank}). Pad must be empty. {swap.notes}"
                ),
            )
        )
        seq += 1
    return actions


def _wave_donor_open_actions(
    ship_date: date,
    opens: list,
    *,
    sequence_start: int,
) -> list[WaveAction]:
    """Colocated donor opens (rolling pre-open uses earlier scheduled time)."""
    actions: list[WaveAction] = []
    seq = sequence_start
    for row in opens:
        scheduled = row.scheduled_open
        rolling = row.open_at is not None and row.open_at < row.wave_send and not row.ahead_prep
        if row.ahead_prep:
            code = "LULL_AHEAD_PREP"
            lead_h = int((row.wave_send - scheduled).total_seconds() // 3600)
            detail = (
                f"Lull-window prep: open donor build on {row.donor_slot} "
                f"(source {row.source_slot}) for wave {row.wave_index} "
                f"@ {scheduled:%H:%M} ({lead_h}h before send: slack hour)."
            )
        elif rolling:
            code = "ROLLING_PREOPEN_DONOR"
            lead_h = int((row.wave_send - row.open_at).total_seconds() // 3600)
            detail = (
                f"Rolling pre-open donor build on {row.donor_slot} "
                f"(source {row.source_slot}) for wave {row.wave_index} "
                f"@ {scheduled:%H:%M} ({lead_h}h before send)."
            )
        else:
            code = "OPEN_DONOR_COLOCATED"
            detail = (
                f"Open colocated donor build on {row.donor_slot} "
                f"(source {row.source_slot}) for wave {row.wave_index}."
            )
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=scheduled,
                wave_due=row.wave_due,
                wave_index=row.wave_index,
                sequence=seq,
                action=code,
                role="setup_crew",
                item_id=row.item_id,
                layers=row.layers,
                from_slot=row.source_slot,
                to_slot=row.donor_slot,
                detail=detail,
            )
        )
        seq += 1
    return actions


def _shift_pull_actions(
    ship_date: date,
    as_of: datetime,
    events: tuple[tuple[str, str, str | None], ...],
    *,
    sequence_start: int,
) -> list[WaveAction]:
    actions: list[WaveAction] = []
    seq = sequence_start
    for item_id, pallet_id, staged_slot in events:
        if staged_slot:
            detail = (
                f"Fetch pallet {pallet_id} for {item_id} from upstream and stage to {staged_slot}."
            )
            to_slot = staged_slot
        else:
            detail = f"Fetch pallet {pallet_id} for {item_id} from upstream (stage when face opens)."
            to_slot = None
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=as_of,
                wave_due=None,
                wave_index=None,
                sequence=seq,
                action="FREEZER_PULL",
                role="assist",
                item_id=item_id,
                layers=None,
                from_slot=None,
                to_slot=to_slot,
                detail=detail,
            )
        )
        seq += 1
    return actions


@dataclass(frozen=True)
class DailyWavePlanBuild:
    """Plan plus replay bundle — one backtest, no duplicate sim for CSV export."""

    plan: DailyWavePlan
    replay: PickDayReplay


def build_daily_wave_plan(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig | None = None,
    carried_state=None,
    staging_pick_boost: dict[str, int] | None = None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...] = (),
) -> DailyWavePlan:
    """Dynamic slotting at setup plus hourly wave action plan for one ship date."""
    return build_daily_wave_plan_with_replay(
        orders,
        ship_date,
        cfg=cfg,
        carried_state=carried_state,
        staging_pick_boost=staging_pick_boost,
        hourly_staging_boosts=hourly_staging_boosts,
    ).plan


def build_daily_wave_plan_with_replay(
    orders: list[OrderLine],
    ship_date: date,
    *,
    cfg: ReplayConfig | None = None,
    carried_state=None,
    staging_pick_boost: dict[str, int] | None = None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...] = (),
) -> DailyWavePlanBuild:
    """Build plan and return replay for hourly CSV / sim KPI export."""
    cfg = cfg or ReplayConfig(resupply_rack_enabled=True)
    spread = _spread_orders(orders, cfg)
    day_orders = [o for o in spread if order_ship_date(o) == ship_date]
    if not day_orders:
        raise ValueError(f"No orders on ship date {ship_date}")

    replay = run_pick_day_backtest(
        orders,
        ship_date,
        cfg=cfg,
        carried_state=carried_state,
        staging_pick_boost=staging_pick_boost,
        hourly_staging_boosts=hourly_staging_boosts,
    )
    report = replay.report
    setup_at = replay.setup_at
    assignments = list(replay.assignments)
    preopen = list(replay.preopen)
    pad_swaps = list(replay.pad_swaps)
    wave_donor_opens = list(replay.wave_donor_opens)
    snap_visible_skus = replay.visible_skus
    snap_visible_layers = replay.visible_layers

    if cfg.synthetic_wave_spread:
        buckets = split_day_into_wave_buckets(
            day_orders,
            ship_date=ship_date,
            waves_per_day=cfg.waves_per_day,
            shift_start_hour=cfg.send_start_hour,
            due_offset_hours=cfg.due_offset_hours,
        )
    else:
        buckets = split_day_by_native_due(
            day_orders,
            ship_date=ship_date,
            due_offset_hours=cfg.due_offset_hours,
        )

    bucket_by_send = {b.release_at.replace(minute=0, second=0, microsecond=0): b for b in buckets}

    # Slot → station map (faces AND donor pads) for empty-leg pick resequencing.
    from layout_advisor.planning.dynamic_slotting import build_aisle_slot_map

    _, aisle_map = build_aisle_slot_map(
        source_count=cfg.source_slots,
        donor_count=cfg.donor_slots,
        picker_count=cfg.picker_count,
        center_gap_ft=cfg.center_gap_ft,
    )
    station_by_slot = {slot_id: mapped.station for slot_id, mapped in aisle_map.items()}
    # Rack bays sit beside the matching aisle station (1-deep default) so assist
    # RELOCATE_TO_RACK / RACK_PROMOTE can resequence on true rack geometry.
    from slotting.rack_depth import rack_station_index as _rack_st

    n_stations = max((m.station for m in aisle_map.values()), default=-1) + 1
    for flank in ("L", "R"):
        for bay in range(1, n_stations + 1):
            rack_id = f"R-{flank}{bay:02d}"
            station_by_slot[rack_id] = min(n_stations - 1, _rack_st(rack_id))

    all_actions: list[WaveAction] = []
    all_actions.extend(_setup_actions(ship_date, setup_at, assignments, preopen))
    all_actions.extend(_pad_role_actions(ship_date, pad_swaps, sequence_start=len(all_actions) + 1))

    # One WavePlan per wave: 10-min sim steps share a send hour, so per-step
    # appends would duplicate each wave (6x at 10-min ticks). Accumulate
    # actions per bucket and materialize after the loop.
    wave_actions_by_index: dict[int, list[WaveAction]] = {}
    wave_bucket_by_index: dict[int, object] = {}
    seq = len(all_actions) + 1
    next_trip = 1  # unique across 10-min steps that share a wave_send hour
    wave_open_by_hour: dict[datetime, list] = {}
    for row in wave_donor_opens:
        hour = row.scheduled_open.replace(minute=0, second=0, microsecond=0)
        wave_open_by_hour.setdefault(hour, []).append(row)

    for i, result in enumerate(report.steps):
        step_as_of = result.as_of
        as_of = step_as_of.replace(minute=0, second=0, microsecond=0)
        # Picks may land on setup evening (ship_date - 1) under order-ahead /
        # overnight windows — include through end of ship day, not ship_date only.
        include_picks = step_as_of.date() <= ship_date
        pick_events = (
            report.layer_pick_events_per_step[i]
            if include_picks and i < len(report.layer_pick_events_per_step)
            else ()
        )
        bucket = active_wave_bucket(buckets, as_of) if include_picks else None
        if pick_events:
            from slotting.pallet_geometry import sku_layer_height_in

            from pathlib import Path as _Path

            from slotting.pallet_geometry import default_site_skus_path

            _skus = (
                _Path(cfg.skus_path)
                if getattr(cfg, "skus_path", None)
                else default_site_skus_path()
            )
            heights = {
                e.item_id: sku_layer_height_in(e.item_id, skus_path=_skus)
                for e in pick_events
            }
            pick_actions = layer_pick_actions_from_events(
                list(pick_events),
                ship_date=ship_date,
                wave_send=bucket.release_at if bucket else step_as_of,
                wave_due=bucket.due_at if bucket else None,
                wave_index=bucket.wave_index if bucket else None,
                sequence_start=seq,
                trip_start=next_trip,
                station_by_slot=station_by_slot,
                layer_height_by_item=heights,
            )
            all_actions.extend(pick_actions)
            seq += len(pick_actions)
            stamped = [a.trip for a in pick_actions if a.trip is not None]
            if stamped:
                next_trip = max(stamped) + 1
        if as_of.date() != ship_date:
            continue
        pull_events = (
            report.shift_pull_events_per_step[i]
            if i < len(report.shift_pull_events_per_step)
            else ()
        )
        if pull_events:
            pull_actions = _shift_pull_actions(
                ship_date, as_of, pull_events, sequence_start=seq
            )
            all_actions.extend(pull_actions)
            seq += len(pull_actions)
        if cfg.synthetic_wave_spread and (
            as_of.hour < cfg.send_start_hour or as_of.hour >= cfg.send_start_hour + cfg.waves_per_day
        ):
            continue
        wave_opens = wave_open_by_hour.pop(as_of, [])
        send_bucket = bucket_by_send.get(as_of)
        if send_bucket is None and not wave_opens:
            continue
        step_actions: list[WaveAction] = []
        if send_bucket is not None:
            step_actions = actions_from_simulation_step(
                result,
                ship_date=ship_date,
                wave_send=send_bucket.release_at,
                wave_due=send_bucket.due_at,
                wave_index=send_bucket.wave_index,
                sequence_start=seq,
                include_aggregate_pick=not cfg.track_pallet_inventory,
            )
        if wave_opens:
            open_actions = _wave_donor_open_actions(
                ship_date,
                wave_opens,
                sequence_start=seq + len(step_actions),
            )
            step_actions = list(step_actions) + open_actions
        if step_actions and station_by_slot:
            step_actions = resequence_assist_min_travel(step_actions, station_by_slot)
        if send_bucket is not None:
            wave_actions_by_index.setdefault(send_bucket.wave_index, []).extend(step_actions)
            wave_bucket_by_index[send_bucket.wave_index] = send_bucket
        all_actions.extend(step_actions)
        seq += len(step_actions)

    wave_plans = [
        WavePlan(
            wave_index=idx,
            wave_send=bucket.release_at,
            wave_due=bucket.due_at,
            sku_count=bucket.sku_count,
            layer_demand=bucket.layer_demand,
            actions=tuple(wave_actions_by_index[idx]),
        )
        for idx, bucket in sorted(wave_bucket_by_index.items())
    ]

    layout = f"{cfg.source_slots}:{cfg.donor_slots}"
    plan = DailyWavePlan(
        ship_date=ship_date,
        setup_at=setup_at,
        layout=layout,
        visible_skus=snap_visible_skus,
        visible_layers=snap_visible_layers,
        setup_assignments=tuple(assignments),
        preopen_donors=tuple(preopen),
        pad_role_swaps=tuple(pad_swaps),
        waves=tuple(wave_plans),
        all_actions=tuple(all_actions),
    )
    return DailyWavePlanBuild(plan=plan, replay=replay)


def build_month_wave_plans(
    orders: list[OrderLine],
    *,
    year: int,
    month: int,
    cfg: ReplayConfig | None = None,
) -> list[DailyWavePlan]:
    """One daily wave plan per ship day in a calendar month."""
    spread = _spread_orders(orders, cfg or ReplayConfig())
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)
    by_day = bucket_orders_by_ship_day(
        [o for o in spread if start_date <= order_ship_date(o) <= end_date]
    )
    plans: list[DailyWavePlan] = []
    for day in sorted(by_day):
        plans.append(build_daily_wave_plan(orders, day, cfg=cfg))
    return plans


def write_daily_plan(
    plan: DailyWavePlan,
    out_dir: Path,
    *,
    replay: PickDayReplay | None = None,
    cfg: ReplayConfig | None = None,
    carry_warm: bool | None = None,
) -> dict[str, Path]:
    """Write setup slotting, wave actions, summary JSON, and optional hourly sim CSV.

    *carry_warm* stamps whether the sim started from carried chain inventory —
    an isolated single-day regen writes carry-cold KPIs that must not be cited
    for chain evaluation.
    """
    out_dir = Path(out_dir) / plan.ship_date.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_path = out_dir / "setup_slotting.csv"
    with setup_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "item_id",
                "layers",
                "earliest_due",
                "location",
                "slot_id",
                "role",
                "notes",
            ],
        )
        writer.writeheader()
        for row in plan.setup_assignments:
            writer.writerow(
                {
                    "rank": row.rank,
                    "item_id": row.item_id,
                    "layers": row.layers,
                    "earliest_due": row.earliest_due.isoformat(sep=" "),
                    "location": row.location,
                    "slot_id": row.slot_id or "",
                    "role": row.role,
                    "notes": row.notes,
                }
            )

    wave_path = out_dir / "wave_actions.csv"
    with wave_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "ship_date",
                "wave_index",
                "wave_send",
                "wave_due",
                "sequence",
                "action",
                "role",
                "item_id",
                "layers",
                "from_slot",
                "to_slot",
                "trip",
                "order_key",
                "detail",
            ],
        )
        writer.writeheader()
        for row in plan.all_actions:
            writer.writerow(
                {
                    "ship_date": row.ship_date.isoformat(),
                    "wave_index": row.wave_index if row.wave_index is not None else "",
                    "wave_send": row.wave_send.isoformat(sep=" "),
                    "wave_due": row.wave_due.isoformat(sep=" ") if row.wave_due else "",
                    "sequence": row.sequence,
                    "action": row.action,
                    "role": row.role,
                    "item_id": row.item_id or "",
                    "layers": row.layers if row.layers is not None else "",
                    "from_slot": row.from_slot or "",
                    "to_slot": row.to_slot or "",
                    "trip": row.trip if row.trip is not None else "",
                    "order_key": row.order_key or "",
                    "detail": row.detail,
                }
            )

    waves_summary = [
        {
            "wave_index": w.wave_index,
            "wave_send": w.wave_send.isoformat(sep=" "),
            "wave_due": w.wave_due.isoformat(sep=" "),
            "sku_count": w.sku_count,
            "layer_demand": w.layer_demand,
            "action_count": len(w.actions),
        }
        for w in plan.waves
    ]
    sim_kpis = None
    hourly_path: Path | None = None
    pad_sim_path: Path | None = None
    if replay is not None and cfg is not None:
        hourly_path = out_dir / f"pick_day_{plan.ship_date.isoformat()}_hourly.csv"
        write_pick_day_hourly_csv(replay, cfg=cfg, ship_date=plan.ship_date, path=hourly_path)
        sim_kpis = replay_sim_kpis(replay, cfg=cfg, ship_date=plan.ship_date)
        if replay.donor_pad_events:
            from layout_advisor.planning.donor_pad_sim import write_donor_pad_sim_csv

            pad_sim_path = out_dir / f"donor_pad_sim_{plan.ship_date.isoformat()}.csv"
            write_donor_pad_sim_csv(list(replay.donor_pad_events), pad_sim_path)

    summary_payload: dict = {
        "ship_date": plan.ship_date.isoformat(),
        "setup_at": plan.setup_at.isoformat(sep=" "),
        "layout": plan.layout,
        "visible_skus": plan.visible_skus,
        "visible_layers": plan.visible_layers,
        "setup_face_count": sum(1 for a in plan.setup_assignments if a.location == "source_face"),
        "setup_rack_count": sum(1 for a in plan.setup_assignments if a.location == "rack"),
        "freezer_hold_count": sum(1 for a in plan.setup_assignments if a.location == "freezer_hold"),
        "preopen_donor_count": len(plan.preopen_donors),
        "pad_role_swap_count": len(plan.pad_role_swaps),
        "dynamic_pad_schedule": [
            {
                "at": s.effective_at.isoformat(sep=" "),
                "slot_id": s.slot_id,
                "from": s.from_role,
                "to": s.to_role,
                "station": s.station,
            }
            for s in plan.pad_role_swaps
        ],
        "wave_count": plan.wave_count,
        "total_actions": plan.total_actions,
        "waves": waves_summary,
    }
    if sim_kpis is not None:
        summary_payload["sim_kpis"] = sim_kpis
    if cfg is not None:
        summary_payload["config_hash"] = replay_config_hash(cfg)
        summary_payload["code_fingerprint"] = replay_code_fingerprint()
    if carry_warm is not None:
        summary_payload["carry_warm"] = carry_warm

    summary_path = out_dir / "daily_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    paths = {
        "setup_slotting": setup_path,
        "wave_actions": wave_path,
        "daily_summary": summary_path,
    }
    if replay is not None:
        from layout_advisor.planning.order_schedule import OrderScheduleConfig, build_order_schedule, write_order_schedule_csv

        osc = OrderScheduleConfig(
            order_lookahead_hours=getattr(cfg, "order_lookahead_hours", 24.0) if cfg else 24.0,
            tick_minutes=getattr(cfg, "sim_step_minutes", 60) if cfg else 60,
            max_build_weight_lb=getattr(cfg, "max_build_weight_lb", 2500.0) if cfg else 2500.0,
        )
        order_items = build_order_schedule(list(replay.initial.orders), plan.ship_date, config=osc)
        order_path = write_order_schedule_csv(order_items, out_dir / "order_schedule.csv", config=osc)
        paths["order_schedule"] = order_path
    if hourly_path is not None:
        paths["hourly_sim"] = hourly_path
    if pad_sim_path is not None:
        paths["donor_pad_sim"] = pad_sim_path
    return paths


def write_month_index(plans: list[DailyWavePlan], out_dir: Path) -> Path:
    """Write a month-level index CSV of all ship days."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "month_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "ship_date",
                "visible_skus",
                "visible_layers",
                "waves",
                "total_actions",
                "face_staged",
                "rack_staged",
                "freezer_hold",
                "preopen_donors",
            ],
        )
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "ship_date": plan.ship_date.isoformat(),
                    "visible_skus": plan.visible_skus,
                    "visible_layers": plan.visible_layers,
                    "waves": plan.wave_count,
                    "total_actions": plan.total_actions,
                    "face_staged": sum(1 for a in plan.setup_assignments if a.location == "source_face"),
                    "rack_staged": sum(1 for a in plan.setup_assignments if a.location == "rack"),
                    "freezer_hold": sum(1 for a in plan.setup_assignments if a.location == "freezer_hold"),
                    "preopen_donors": len(plan.preopen_donors),
                }
            )
    return index_path
