"""Generate dynamic slotting + per-wave operator action plans.

Runs on the synthetic site by default:

    python scripts/generate_daily_wave_plans.py --date 2026-03-03
    python scripts/generate_daily_wave_plans.py --carry-week --wave-layout

Artifacts land under ``data/plans/<ship_date>/`` (ignored by git).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from layout_advisor.planning.daily_plan import (
    build_daily_wave_plan_with_replay,
    build_month_wave_plans,
    write_daily_plan,
    write_month_index,
)
from layout_advisor.pick_throughput import DEFAULT_CENTER_GAP_FT, DEFAULT_PICKER_COUNT
from layout_advisor.simulate.replay import ReplayConfig, simulate_pick_day
from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG
from slotting.backtest import HourlyStagingBoost
from slotting.demand_throughput import load_orders
from slotting.models import OrderArea
from slotting.wave_release import bucket_orders_by_ship_day, order_ship_date

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data" / "plans"
DEFAULT_ORDERS = REPO / "data" / "synthetic" / "orders.csv"
# Optional manual staging pick lists (item_id, suggested_extra_pallets). None by
# default: the synthetic site ships without one.
DEFAULT_PICK_LIST: Path | None = None
LATE_PICK_LIST: Path | None = None
# Chain windows over the seed-7 synthetic week (Mon 2 Mar .. Fri 6 Mar 2026).
CARRY_WEEK_START = date(2026, 3, 2)
CARRY_WEEK_END = date(2026, 3, 6)
STRESS_CHAIN_START = date(2026, 3, 2)
STRESS_CHAIN_END = date(2026, 3, 6)


def _picker_layout(args) -> tuple[int, float]:
    picker_count = (
        args.picker_count
        if args.picker_count is not None
        else (2 if args.dual_picker else DEFAULT_PICKER_COUNT)
    )
    center_gap_ft = (
        args.center_gap_ft
        if args.center_gap_ft is not None
        else (20.0 if args.dual_picker else DEFAULT_CENTER_GAP_FT)
    )
    return picker_count, center_gap_ft


def _default_cfg(args) -> ReplayConfig:
    from layout_advisor.config import load_ops_profile

    picker_count, center_gap_ft = _picker_layout(args)
    profile_id = getattr(args, "ops_profile", None) or "reference"
    if args.reference:
        base = load_ops_profile(profile_id)
        return replace(
            base,
            picker_count=picker_count,
            center_gap_ft=center_gap_ft,
        )
    return ReplayConfig(
        source_slots=args.source,
        donor_slots=args.donor,
        setup_source_slots=args.source,
        resupply_rack_enabled=args.resupply_rack,
        reserve_cap_ratio=args.reserve_ratio,
        rack_depth_tiers=args.rack_depth,
        picker_count=picker_count,
        center_gap_ft=center_gap_ft,
    )


def load_staging_pick_boost(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {}
    boost: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            item_id = row["item_id"]
            count = int(row.get("suggested_extra_pallets") or 1)
            boost[item_id] = boost.get(item_id, 0) + count
    return boost


def _hourly_boosts_for_day(
    cfg: ReplayConfig,
    ship: date,
    *,
    carried_state,
    late_pick_list: Path | None,
) -> tuple[HourlyStagingBoost, ...]:
    if carried_state is None or not cfg.late_shift_staging_boost_enabled:
        return ()
    if ship != CARRY_WEEK_END:
        return ()
    late_boost = load_staging_pick_boost(late_pick_list)
    if not late_boost:
        return ()
    return (
        HourlyStagingBoost(
            ship_date=ship,
            hour=cfg.late_shift_staging_hour,
            boost_by_item=late_boost,
            default_layers=cfg.reserve_default_layers,
        ),
    )


def _staging_boost_for_day(cfg: ReplayConfig, ship: date, pick_list: Path | None) -> dict[str, int] | None:
    if not cfg.spike_prep_pull_enabled:
        return None
    if ship.weekday() in cfg.spike_prep_pull_weekdays or ship == CARRY_WEEK_END:
        return load_staging_pick_boost(pick_list) or None
    return None


def _print_plan_summary(plan, cfg: ReplayConfig, *, reference: bool) -> None:
    print(f"=== Daily wave plan: {plan.ship_date} ===")
    print(
        f"  Config: {'REFERENCE_REPLAY_CONFIG' if reference else 'legacy minimal'}  "
        f"|  layout {plan.layout}"
    )
    print(f"  Pickers: {cfg.picker_count} layer-picker  |  center gap: {cfg.center_gap_ft:.0f} ft")
    print(f"  Layout: {plan.layout} | visible {plan.visible_skus} SKUs / {plan.visible_layers:,} layers")
    print(f"  Setup @ {plan.setup_at:%Y-%m-%d %H:%M}")
    print(
        f"  Slotting: {sum(1 for a in plan.setup_assignments if a.location == 'source_face')} faces | "
        f"{sum(1 for a in plan.setup_assignments if a.location == 'rack')} racks | "
        f"{sum(1 for a in plan.setup_assignments if a.location == 'freezer_hold')} freezer hold"
    )
    print(
        f"  Pre-open donors: {len(plan.preopen_donors)} | Waves: {plan.wave_count} | "
        f"Actions: {plan.total_actions}"
    )


def _generate_one_day(
    orders,
    ship: date,
    cfg: ReplayConfig,
    *,
    output_dir: Path,
    wave_layout: bool,
    minute_frames: bool = False,
    minute_tick_minutes: int = 10,
    carried_state,
    staging_pick_boost: dict[str, int] | None,
    hourly_staging_boosts: tuple[HourlyStagingBoost, ...] = (),
    reference: bool,
):
    build = build_daily_wave_plan_with_replay(
        orders,
        ship,
        cfg=cfg,
        carried_state=carried_state,
        staging_pick_boost=staging_pick_boost,
        hourly_staging_boosts=hourly_staging_boosts,
    )
    paths = write_daily_plan(
        build.plan,
        output_dir,
        replay=build.replay,
        cfg=cfg,
        carry_warm=carried_state is not None,
    )
    plan = build.plan
    _print_plan_summary(plan, cfg, reference=reference)
    print(f"  Setup CSV:  {paths['setup_slotting']}")
    print(f"  Wave CSV:   {paths['wave_actions']}")
    print(f"  Summary:    {paths['daily_summary']}")
    if paths.get("hourly_sim"):
        print(f"  Hourly sim: {paths['hourly_sim']}")
    print()
    print("=== Waves ===")
    for wave in plan.waves:
        waits = sum(1 for a in wave.actions if a.action.startswith("WAIT"))
        promotes = sum(1 for a in wave.actions if "PROMOTE" in a.action or a.action == "OPEN_DONOR_BUILD")
        print(
            f"  {wave.wave_send:%H:%M} -> due {wave.wave_due:%H:%M}  "
            f"{wave.sku_count:3d} SKU  {wave.layer_demand:4,} lyr  "
            f"{len(wave.actions):3d} actions  ({promotes} moves, {waits} waits)"
        )
    if wave_layout:
        from layout_advisor.layout.render_wave_facility import write_wave_floor_views
        from layout_advisor.planning.wave_floor_timeline import build_wave_floor_timeline

        timeline = build_wave_floor_timeline(
            orders,
            ship,
            cfg=cfg,
            carried_state=carried_state,
            staging_pick_boost=staging_pick_boost,
            hourly_staging_boosts=hourly_staging_boosts,
            minute_frames=minute_frames,
            minute_tick_minutes=minute_tick_minutes,
            replay=build.replay,
        )
        layout_paths = write_wave_floor_views(timeline, output_dir)
        print(f"  Wave floor HTML: {layout_paths['html']}")
    print()
    return build.replay


def _carry_cache_path(output_dir: Path, day: date, cfg: ReplayConfig) -> Path:
    """End-of-day carry state cache, keyed so it can never serve stale state:
    any src or config change rotates the key exactly like the freshness guard."""
    from layout_advisor.planning.daily_plan import replay_code_fingerprint, replay_config_hash

    key = f"{day.isoformat()}_{replay_config_hash(cfg)}_{replay_code_fingerprint()}"
    return output_dir.parent / ".carry_cache" / f"{key}.pkl"


def _guard_canonical_plans(
    output_dir: Path, cfg: ReplayConfig, *, force: bool = False, chain_run: bool = False
) -> None:
    """Refuse to overwrite the canonical plans with non-reference artifacts.

    `data/plans/` is the chain benchmark in file form: `lp_upper_bound.py` and
    any chain comparison read it. `--no-reference` builds a bare `ReplayConfig`
    from argparse defaults, and `--output-dir` defaults *here*, so one flag would
    silently replace reference-config output with a legacy-config regen. A
    freshness check would catch it afterwards; this catches it before the write.
    """
    from layout_advisor.planning.daily_plan import provenance

    if force or output_dir.resolve() != DEFAULT_OUT.resolve():
        return

    # Second, subtler path to the same damage, found by tripping it while
    # testing the config guard below. A reference-config single-day run passes
    # every check here and still destroys the stored artifacts, because the
    # stored ones are CARRY-WARM chain output and a lone `--date X` starts from
    # a cold floor. It rewrites a whole ship day under an identical config hash,
    # so a hash-based freshness check stays green on a plan that no longer
    # matches the benchmark it feeds.
    warm_days = [
        path.parent.name
        for path in sorted(DEFAULT_OUT.glob("*/daily_summary.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("carry_warm") is True
    ]
    if warm_days and not chain_run:
        raise SystemExit(
            "REFUSING to overwrite carry-warm plans with a cold single-day run.\n"
            f"  {output_dir}\n"
            f"  {len(warm_days)} committed day(s) are carry_warm=True "
            f"({warm_days[0]}..{warm_days[-1]})\n\n"
            "A lone --date run starts from a cold floor, so its artifacts differ from\n"
            "the chain output the benchmark reads even under identical config - the\n"
            "config hash matches, so the freshness test would NOT catch it.\n"
            "Regenerate the chain (--chain / --carry-week), or write elsewhere:\n"
            "  --output-dir <scratch>\n"
            "--force overrides."
        )

    prov = provenance(cfg)
    if prov["is_reference_config"]:
        return
    raise SystemExit(
        f"REFUSING to write non-reference artifacts into the canonical plans directory.\n"
        f"  {output_dir}\n"
        f"  config {prov['config_hash'][:8]} "
        f"({prov['source_slots']}:{prov['donor_slots']}), reference is "
        f"{provenance(REFERENCE_REPLAY_CONFIG)['config_hash'][:8]} (70:30)\n\n"
        "This directory holds the chain benchmark. Write somewhere else:\n"
        "  --output-dir <scratch>\n"
        "or drop --no-reference to run the reference config. --force overrides."
    )


def _carry_cache_prune(output_dir: Path, cfg: ReplayConfig, *, quiet: bool = False) -> int:
    """Drop cache entries from retired configs. Returns bytes freed.

    The key embeds the config hash AND the src fingerprint, so every code change
    rotates it — which is what makes the cache safe, and also what makes it grow
    without bound: nothing ever removed the orphans, and after a few config
    changes the whole directory can be dead weight with no current entry in it.

    Entries for the live key are kept; everything else is regenerable by
    definition (a miss just re-simulates). Runs once per invocation.
    """
    from layout_advisor.planning.daily_plan import replay_code_fingerprint, replay_config_hash

    cache_dir = output_dir.parent / ".carry_cache"
    if not cache_dir.is_dir():
        return 0
    live_suffix = f"_{replay_config_hash(cfg)}_{replay_code_fingerprint()}.pkl"
    freed = 0
    for path in cache_dir.glob("*.pkl"):
        if path.name.endswith(live_suffix):
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            freed += size
        except OSError:
            continue  # best-effort, same as the rest of the cache
    if freed and not quiet:
        print(f"  Carry cache: pruned {freed / 1e6:.0f} MB of retired-config entries")
    return freed


def _carry_cache_load(output_dir: Path, day: date, cfg: ReplayConfig):
    import pickle

    path = _carry_cache_path(output_dir, day, cfg)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _carry_cache_save(output_dir: Path, day: date, cfg: ReplayConfig, state) -> None:
    import pickle

    if state is None:
        return
    path = _carry_cache_path(output_dir, day, cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(state, fh)
    except Exception:
        pass  # cache is best-effort; a miss just re-simulates


def _day_artifacts_fresh(output_dir: Path, day: date, cfg: ReplayConfig, *, wave_layout: bool) -> bool:
    """True when the day's committed artifacts were generated under the CURRENT
    config hash + code fingerprint (same staleness test the suite enforces)."""
    import json

    from layout_advisor.planning.daily_plan import replay_code_fingerprint, replay_config_hash

    summary = output_dir / day.isoformat() / "daily_summary.json"
    if not summary.is_file():
        return False
    if wave_layout:
        # Existence is not freshness: the visuals are written in the same call
        # as the summary, so an index OLDER than the summary is left over from a
        # run that skipped --wave-layout. Treating it as fresh let the portal and
        # floor scrubber render a prior sim beside regenerated KPIs.
        index = output_dir / day.isoformat() / "wave_floor_index.json"
        if not index.is_file() or index.stat().st_mtime < summary.stat().st_mtime:
            return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("config_hash") == replay_config_hash(cfg)
        and payload.get("code_fingerprint") == replay_code_fingerprint()
    )


def _run_carry_chain(
    orders,
    cfg: ReplayConfig,
    targets: list[date],
    pick_list: Path,
    *,
    output_dir: Path,
    wave_layout: bool,
    reference: bool,
    carry_from: date | None = None,
) -> None:
    carried = None
    if carry_from is not None:
        prep_end = min(targets)
        prep_days = sorted(
            bucket_orders_by_ship_day(
                [o for o in orders if carry_from <= order_ship_date(o) < prep_end]
            )
        )
        # Resume from the deepest cached prefix: cache holds each prep day's
        # END state, so an aborted/repeated run skips already-simulated days.
        resume_index = 0
        for index, day in enumerate(prep_days):
            cached = _carry_cache_load(output_dir, day, cfg)
            if cached is None:
                break
            carried = cached
            resume_index = index + 1
        if resume_index:
            print(f"  Carry cache: resumed after {prep_days[resume_index - 1]} ({resume_index}/{len(prep_days)} prep days skipped)")
        for day in prep_days[resume_index:]:
            boost = _staging_boost_for_day(cfg, day, pick_list)
            rep, _ = simulate_pick_day(
                orders,
                day,
                cfg=cfg,
                write_csv=False,
                carried_state=carried,
                staging_pick_boost=boost,
            )
            carried = rep.final_state
            _carry_cache_save(output_dir, day, cfg, carried)

    for day in sorted(targets):
        # Restart resume: a target day whose artifacts are stamped with the
        # CURRENT config+code and whose end state is cached can be skipped —
        # an aborted chain run continues where it stopped instead of at zero.
        cached = _carry_cache_load(output_dir, day, cfg)
        if cached is not None and _day_artifacts_fresh(output_dir, day, cfg, wave_layout=wave_layout):
            print(f"=== Daily wave plan: {day} === (cached: artifacts fresh, skipping)")
            carried = cached
            continue
        boost = _staging_boost_for_day(cfg, day, pick_list)
        hourly_boosts = _hourly_boosts_for_day(
            cfg, day, carried_state=carried, late_pick_list=LATE_PICK_LIST
        )
        # Reuse the plan build's backtest for carry state — the old extra
        # simulate_pick_day here re-ran the identical sim per chain day.
        replay = _generate_one_day(
            orders,
            day,
            cfg,
            output_dir=output_dir,
            wave_layout=wave_layout,
            carried_state=carried,
            staging_pick_boost=boost,
            hourly_staging_boosts=hourly_boosts,
            reference=reference,
        )
        report = replay.report
        eod_state = report.steps[-1].projected_state if report.steps else carried
        carried = report.final_state or eod_state
        _carry_cache_save(output_dir, day, cfg, carried)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=("day", "month"), default="day")
    parser.add_argument("--date", default="2026-03-03", help="Ship date for --period day")
    parser.add_argument("--month", default="2026-03", help="YYYY-MM for --period month")
    parser.add_argument(
        "--orders",
        type=Path,
        default=DEFAULT_ORDERS,
        help="orders.csv to plan from (default: the synthetic site)",
    )
    parser.add_argument(
        "--ops-profile",
        default=None,
        help="Ops profile id (default: reference = REFERENCE_REPLAY_CONFIG)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        default=True,
        help="Use the frozen ops profile (default reference / REFERENCE_REPLAY_CONFIG)",
    )
    parser.add_argument(
        "--no-reference",
        action="store_false",
        dest="reference",
        help="Legacy minimal ReplayConfig (--source/--donor only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing NON-REFERENCE artifacts into the canonical plans directory. "
        "That directory holds the chain benchmark; prefer --output-dir <scratch>.",
    )
    parser.add_argument(
        "--carry-week",
        action="store_true",
        help=f"Regenerate carry chain {CARRY_WEEK_START}..{CARRY_WEEK_END} with floor visuals",
    )
    parser.add_argument(
        "--stress-chain",
        action="store_true",
        help=f"Regenerate full carry chain {STRESS_CHAIN_START}..{STRESS_CHAIN_END} "
        f"(carry week through the stress day)",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        help="Multiple ship dates (ISO); use with --carry-from to chain inventory",
    )
    parser.add_argument(
        "--carry-from",
        default=None,
        help="Chain carry inventory from this date through each --dates target",
    )
    parser.add_argument(
        "--pick-list",
        type=Path,
        default=DEFAULT_PICK_LIST,
        help="Optional morning staging pick list CSV for spike prep / eval days",
    )
    parser.add_argument(
        "--late-pick-list",
        type=Path,
        default=LATE_PICK_LIST,
        help="Optional late-shift pull list CSV for the carry-week eval day",
    )
    parser.add_argument("--source", type=int, default=75)
    parser.add_argument("--donor", type=int, default=25)
    parser.add_argument("--resupply-rack", action="store_true", default=True)
    parser.add_argument("--no-resupply-rack", action="store_false", dest="resupply_rack")
    parser.add_argument("--reserve-ratio", type=float, default=1.0)
    parser.add_argument("--rack-depth", type=int, default=1)
    parser.add_argument(
        "--dual-picker",
        action="store_true",
        help="Dual clamp pickers with 20 ft center gap",
    )
    parser.add_argument("--center-gap-ft", type=float, default=None)
    parser.add_argument("--picker-count", type=int, default=None, help=f"Layer-picker count (default {DEFAULT_PICKER_COUNT})")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--wave-layout",
        action="store_true",
        help="Also render interactive 2D floor layout per wave (HTML + SVG frames)",
    )
    parser.add_argument(
        "--minute-frames",
        action="store_true",
        help="Sub-hour floor SVGs via 10-min sim replay (single --period day only; slow)",
    )
    parser.add_argument("--minute-tick", type=int, default=10)
    args = parser.parse_args()

    orders_path = args.orders
    output_dir = args.output_dir
    pick_list = args.pick_list
    late_pick_list = args.late_pick_list
    stress_start = STRESS_CHAIN_START
    stress_end = STRESS_CHAIN_END
    carry_week_start = CARRY_WEEK_START
    carry_week_end = CARRY_WEEK_END

    this_line = [o for o in load_orders(orders_path) if o.area == OrderArea.THIS_LINE]
    cfg = _default_cfg(args)
    args.output_dir = output_dir
    args.pick_list = pick_list
    globals()["LATE_PICK_LIST"] = late_pick_list

    # A chain regen is the ONLY thing that may rewrite the canonical plans:
    # it reproduces the carry-warm state the committed artifacts hold.
    #
    # NOT `or args.month`: --month has a default and is therefore always
    # truthy, which silently disabled this guard on its first outing. The month
    # branch writes to its own `month_YYYY_MM/` subdirectory anyway, so it never
    # touches the canonical day folders.
    chain_run = bool(args.carry_week or args.stress_chain or args.dates)
    _guard_canonical_plans(output_dir, cfg, force=args.force, chain_run=chain_run)

    # Once per run, before anything writes to the cache. Entries keyed to a
    # retired config or src fingerprint can never be served again — they only
    # consume disk (and sync bandwidth, if the checkout is in a synced folder).
    _carry_cache_prune(output_dir, cfg)

    if args.carry_week:
        targets = sorted(
            bucket_orders_by_ship_day(
                [o for o in this_line if carry_week_start <= order_ship_date(o) <= carry_week_end]
            )
        )
        print(f"=== Carry week regen: {len(targets)} days ===")
        _run_carry_chain(
            this_line,
            cfg,
            targets,
            args.pick_list,
            output_dir=args.output_dir,
            wave_layout=args.wave_layout,
            reference=args.reference,
        )
        return

    if args.stress_chain:
        targets = sorted(
            bucket_orders_by_ship_day(
                [o for o in this_line if stress_start <= order_ship_date(o) <= stress_end]
            )
        )
        print(f"=== Stress chain regen: {len(targets)} days ({stress_start}..{stress_end}) ===")
        _run_carry_chain(
            this_line,
            cfg,
            targets,
            args.pick_list,
            output_dir=args.output_dir,
            wave_layout=args.wave_layout,
            reference=args.reference,
            carry_from=stress_start,
        )
        return

    if args.dates:
        targets = sorted(date.fromisoformat(d) for d in args.dates)
        carry_from = date.fromisoformat(args.carry_from) if args.carry_from else None
        _run_carry_chain(
            this_line,
            cfg,
            targets,
            args.pick_list,
            output_dir=args.output_dir,
            wave_layout=args.wave_layout,
            reference=args.reference,
            carry_from=carry_from,
        )
        return

    if args.period == "day":
        ship = date.fromisoformat(args.date)
        boost = _staging_boost_for_day(cfg, ship, args.pick_list) if args.reference else None
        hourly_boosts = (
            _hourly_boosts_for_day(
                cfg, ship, carried_state=None, late_pick_list=LATE_PICK_LIST
            )
            if args.reference
            else ()
        )
        _generate_one_day(
            this_line,
            ship,
            cfg,
            output_dir=args.output_dir,
            wave_layout=args.wave_layout,
            minute_frames=args.minute_frames and args.wave_layout,
            minute_tick_minutes=args.minute_tick,
            carried_state=None,
            staging_pick_boost=boost,
            hourly_staging_boosts=hourly_boosts,
            reference=args.reference,
        )
    else:
        year_s, month_s = args.month.split("-")
        year, month = int(year_s), int(month_s)
        month_dir = args.output_dir / f"month_{year}_{month:02d}"
        plans = build_month_wave_plans(this_line, year=year, month=month, cfg=cfg)
        for plan in plans:
            write_daily_plan(plan, month_dir)
        index = write_month_index(plans, month_dir)
        print(f"=== Month wave plans: {year}-{month:02d} ===")
        print(f"  Ship days: {len(plans)}")
        print(f"  Index: {index}")
        print(f"  Per-day folders under: {month_dir}")


if __name__ == "__main__":
    main()
