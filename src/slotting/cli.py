"""Command-line entry point for the slotting engine."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console

from slotting.data_loader import CsvDataSource
from slotting.metrics import append_metrics_log, compute_metrics
from slotting.backtest import BacktestMode, backtest_historical, backtest_snapshot_dir
from slotting.report import (
    print_metrics,
    print_recommendations,
    print_backtest_summary,
    print_backtest_low_layers_comparison,
    print_simulation,
    print_wave_ratio,
    print_low_layers_comparison,
    print_simulation_timeline,
)
from slotting.rules import (
    DEFAULT_RULE_CONFIG,
    LIBRARY_POLICIES,
    WITH_LOW_LAYERS,
    RuleConfig,
    parse_disabled_rules,
    parse_rule_ids,
    run_rules,
    with_rules_disabled,
)
from slotting.wave_ratio import compute_wave_ratio
from slotting.simulation import SimulationResult, simulate_once, simulate_timeline
from slotting.optimize import (
    LayoutConstraints,
    OptimizeMode,
    constraints_from_site_profile,
    run_donor_sweep,
    run_sweep,
    write_donor_sweep_csv,
    write_sweep_results_csv,
)
from slotting.pallet_geometry import pallet_height_limits, resolve_height_limits


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slotting",
        description="Rules-based layer-pick slot recommendations (advisory only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Load data, evaluate rules, print report")
    run_cmd.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing pallets.csv, slots.csv, orders.csv, exclusions.csv",
    )
    run_cmd.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Reference timestamp (ISO 8601). Defaults to now.",
    )
    run_cmd.add_argument(
        "--metrics-log",
        type=Path,
        default=None,
        help="Append run metrics to this CSV (default: <data-dir>/metrics_log.csv)",
    )
    run_cmd.add_argument(
        "--profile",
        default=None,
        help="Ops profile whose policy to evaluate under, e.g. 'reference'. Without it "
        "the general-engine library defaults apply, NOT the reference replay policy.",
    )
    run_cmd.add_argument(
        "--policy",
        default=None,
        choices=sorted(LIBRARY_POLICIES),
        help="Named library policy. 'default' can only evict: it disables RACK_PROMOTE "
        "and leaves include_rack_aisle False. Use 'resupply-rack' to reach the promote "
        "path. Mutually exclusive with --profile.",
    )
    run_cmd.add_argument(
        "--without-rules",
        nargs="+",
        default=None,
        metavar="RULE",
        help="Disable these rules on top of the active policy, for attribution.",
    )

    sim_cmd = sub.add_parser(
        "simulate",
        help="Compare baseline KPIs vs projected state if recommendations were followed",
    )
    sim_cmd.add_argument("--data-dir", type=Path, required=True)
    sim_cmd.add_argument("--as-of", type=str, default=None, help="Reference timestamp (ISO 8601)")
    sim_cmd.add_argument(
        "--profile",
        default=None,
        help="Ops profile whose policy to simulate under, e.g. 'reference'. Without it "
        "the general-engine library defaults apply, NOT the reference replay policy.",
    )
    sim_cmd.add_argument(
        "--policy",
        default=None,
        choices=sorted(LIBRARY_POLICIES),
        help="Named library policy. 'default' can only evict: it disables RACK_PROMOTE "
        "and leaves include_rack_aisle False. Use 'resupply-rack' to reach the promote "
        "path. Mutually exclusive with --profile.",
    )
    sim_cmd.add_argument(
        "--from",
        dest="from_time",
        type=str,
        default=None,
        help="Timeline replay start (ISO 8601). Requires --to.",
    )
    sim_cmd.add_argument(
        "--to",
        dest="to_time",
        type=str,
        default=None,
        help="Timeline replay end (ISO 8601). Requires --from.",
    )
    sim_cmd.add_argument(
        "--step-hours",
        type=float,
        default=1.0,
        help="Hours between timeline steps (default: 1)",
    )
    sim_cmd.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write timeline results to CSV",
    )
    sim_cmd.add_argument(
        "--without-rules",
        action="append",
        default=None,
        metavar="RULE",
        help="Disable rule(s), e.g. LOW_LAYERS (repeatable)",
    )
    sim_cmd.add_argument(
        "--compare-low-layers",
        action="store_true",
        help="Run twice: all rules vs all rules except LOW_LAYERS (>2 layer rule)",
    )

    bt_cmd = sub.add_parser(
        "backtest",
        help="Replay rules over historical orders to estimate sustained improvement",
    )
    bt_cmd.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Floor snapshot: pallets.csv, slots.csv, exclusions.csv, orders.csv (historical)",
    )
    bt_cmd.add_argument("--from", dest="from_time", type=str, required=True)
    bt_cmd.add_argument("--to", dest="to_time", type=str, required=True)
    bt_cmd.add_argument("--step-hours", type=float, default=24.0)
    bt_cmd.add_argument(
        "--mode",
        choices=["chained", "static"],
        default="chained",
        help="chained=carry slot moves forward; static=reset floor each step",
    )
    bt_cmd.add_argument(
        "--snapshots-dir",
        type=Path,
        default=None,
        help="Optional: subfolder per timestamp with full CSV sets (JDA daily exports)",
    )
    bt_cmd.add_argument("--output", type=Path, default=None, help="Write step detail CSV")
    bt_cmd.add_argument(
        "--without-rules",
        action="append",
        default=None,
        metavar="RULE",
        help="Disable rule(s), e.g. LOW_LAYERS (repeatable)",
    )
    bt_cmd.add_argument(
        "--compare-low-layers",
        action="store_true",
        help="Run backtest twice: with vs without LOW_LAYERS (>2 layer rule)",
    )
    bt_cmd.add_argument(
        "--skus-path",
        type=Path,
        default=None,
        help="SKU master CSV for per-SKU layer heights (default: data/site/skus.csv when present)",
    )

    opt_cmd = sub.add_parser(
        "optimize",
        help="Sweep (source, donor) layouts and rank by wave throughput",
    )
    opt_cmd.add_argument(
        "--orders",
        type=Path,
        default=None,
        help="Historical orders.csv (optional if --profile provides orders_slotting_path)",
    )
    opt_cmd.add_argument(
        "--mode",
        choices=["min-footprint", "max-throughput"],
        required=True,
        help="Mode A: smallest sqft meeting target; Mode B: max wave LP %%",
    )
    opt_cmd.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Layout-advisor site_profile.yaml (area, ranges, orders path override)",
    )
    opt_cmd.add_argument("--area-sqft", type=float, default=None)
    opt_cmd.add_argument("--source-slot-sqft", type=float, default=60.0)
    opt_cmd.add_argument("--donor-slot-sqft", type=float, default=48.0)
    opt_cmd.add_argument("--source-range", type=int, nargs=2, metavar=("MIN", "MAX"))
    opt_cmd.add_argument("--donor-range", type=int, nargs=2, metavar=("MIN", "MAX"))
    opt_cmd.add_argument(
        "--target-wave-lp-pct",
        type=float,
        default=85.0,
        help="Mode A target projected wave layer pick %% (default 85)",
    )
    opt_cmd.add_argument(
        "--max-shortage",
        type=int,
        default=0,
        help="Mode A max waved item shortages after rules",
    )
    opt_cmd.add_argument("--exclusions", type=Path, default=None)
    opt_cmd.add_argument(
        "--no-area-limit",
        action="store_true",
        help="Mode A: ignore physical area, find min footprint for throughput only",
    )
    opt_cmd.add_argument(
        "--wave-hours",
        type=float,
        default=None,
        help="Wave demand window in hours (default: layout_advisor wave_horizon_hours or 4). "
        "Set it when the operation's wave cadence is known; do not assume.",
    )
    opt_cmd.add_argument(
        "--fix-source",
        type=int,
        default=None,
        metavar="N",
        help="Donor sweep: hold source at N slots and sweep donor range",
    )
    opt_cmd.add_argument("--out", type=Path, required=True, help="Output directory for sweep_results.csv")
    opt_cmd.add_argument(
        "--skus-path",
        type=Path,
        default=None,
        help="SKU master CSV for per-SKU layer heights (default: data/site/skus.csv when present)",
    )
    return parser


def resolve_rule_config(args: argparse.Namespace, base: RuleConfig) -> tuple[RuleConfig, str]:
    """Apply ``--profile`` to *base*, returning (config, provenance label).

    `slotting run --data-dir <floor export>` evaluates under DEFAULT_RULE_CONFIG
    unless told otherwise — the general-engine library default, not the
    reference replay policy. The two differ where it matters:

        RACK_PROMOTE      disabled by default, ENABLED under `reference`
        DEST_CLOSE_IDLE   enabled by default, DISABLED under `reference`
        include_rack_aisle  False by default, True under `reference`

    So the default path cannot recommend a rack promote at all — the move the
    resupply-rack design exists to make — while emitting idle-close moves the
    reference policy drops. Nothing in the output used to say which policy
    produced the list.

    `--policy` selects one of the named library policies for callers with no ops
    profile to point at; `--profile reference` uses the reference replay policy.

    Omitting both keeps the library default (unchanged behaviour) but says so,
    because silence is what made this invisible.

    `--without-rules` is a debugging SUBTRACTION and unions on top of whichever
    policy is in force. It must not replace it: DEFAULT_RULE_CONFIG already
    disables three rules, so a replacing parse silently *enables* RACK_PROMOTE,
    BACKUP_PROMOTE and LOW_LAYERS while claiming to have removed one rule.
    """
    profile_id = getattr(args, "profile", None)
    policy_id = getattr(args, "policy", None)
    requested = getattr(args, "without_rules", None)

    if profile_id and policy_id:
        raise SystemExit(
            "--profile and --policy are mutually exclusive: a site profile already "
            "carries its own policy."
        )

    if profile_id:
        from layout_advisor.config import load_ops_profile
        from layout_advisor.planning.daily_plan import replay_config_hash

        replay_cfg = load_ops_profile(profile_id)
        resolved = replay_cfg.rule_config
        label = f"profile {profile_id} (config hash {replay_config_hash(replay_cfg)})"
    elif policy_id:
        resolved = LIBRARY_POLICIES[policy_id]
        label = f"library policy '{policy_id}'"
    else:
        resolved = base
        label = "library defaults (DEFAULT_RULE_CONFIG), no --policy or --profile given"

    if requested:
        resolved = with_rules_disabled(resolved, requested)
        removed = ", ".join(sorted(r.value for r in parse_rule_ids(requested)))
        label = f"{label}, minus {removed}"
    return resolved, label


def cmd_run(args: argparse.Namespace) -> int:
    console = Console()
    data_dir: Path = args.data_dir
    as_of = _parse_as_of(args.as_of)

    source = CsvDataSource(data_dir)
    state = source.load_world_state(as_of=as_of)

    rule_config, policy_label = resolve_rule_config(args, DEFAULT_RULE_CONFIG)

    metrics = compute_metrics(state)
    recommendations = run_rules(state, rule_config)
    wave_ratio = compute_wave_ratio(state)

    console.print(f"[bold]Slotting engine[/bold]: data from [cyan]{data_dir}[/cyan]")
    # Stamp the policy on the output. A recommendation list that does not say
    # which policy produced it cannot be traced back, and the live path was
    # silently running library defaults.
    console.print(f"Policy: [cyan]{policy_label}[/cyan]")
    console.print(f"As of:  [cyan]{state.as_of:%Y-%m-%d %H:%M}[/cyan]\n")
    print_metrics(console, metrics)
    print_wave_ratio(console, wave_ratio)
    print_recommendations(console, recommendations)

    log_path = args.metrics_log or (data_dir / "metrics_log.csv")
    append_metrics_log(metrics, log_path)
    console.print(f"\n[dim]Metrics appended to {log_path}[/dim]")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    console = Console()
    data_dir: Path = args.data_dir
    as_of = _parse_as_of(args.as_of)
    rule_config, policy_label = resolve_rule_config(args, DEFAULT_RULE_CONFIG)

    source = CsvDataSource(data_dir)
    state = source.load_world_state(as_of=as_of)

    console.print(f"[bold]Slotting Simulation[/bold]: data from [cyan]{data_dir}[/cyan]")
    console.print(f"Policy: [cyan]{policy_label}[/cyan]\n")

    if args.from_time and args.to_time:
        start = datetime.fromisoformat(args.from_time)
        end = datetime.fromisoformat(args.to_time)
        step = timedelta(hours=args.step_hours)
        base = source.load_world_state(as_of=start)
        if args.compare_low_layers:
            console.print("[yellow]Note:[/yellow] --compare-low-layers applies to single-run simulate only.\n")
        results = simulate_timeline(base, start=start, end=end, step=step, rule_config=rule_config)
        print_simulation_timeline(console, results)
        if args.output:
            _write_timeline_csv(results, args.output)
            console.print(f"[dim]Timeline written to {args.output}[/dim]")
        return 0

    if args.compare_low_layers:
        without_result = simulate_once(state, rule_config=DEFAULT_RULE_CONFIG)
        with_result = simulate_once(state, rule_config=WITH_LOW_LAYERS)
        print_low_layers_comparison(console, with_result, without_result)
        console.print("[bold]Full detail: with LOW_LAYERS rule[/bold]\n")
        print_simulation(console, with_result)
        console.print("[bold]Full detail: without LOW_LAYERS rule[/bold]\n")
        print_simulation(console, without_result)
        if args.output:
            _write_low_layers_compare_csv(with_result, without_result, args.output)
            console.print(f"[dim]Comparison written to {args.output}[/dim]")
        return 0

    result = simulate_once(state, rule_config=rule_config)
    if args.without_rules:
        disabled = ", ".join(rule_config.disabled)
        console.print(f"[dim]Disabled rules: {disabled}[/dim]\n")
    print_simulation(console, result)
    console.print("[dim]Advisory recommendations that drove this simulation:[/dim]\n")
    print_recommendations(console, result.recommendations)
    return 0


TIMELINE_COLUMNS = [
    "as_of",
    "baseline_dead_slot_rate_pct",
    "projected_dead_slot_rate_pct",
    "improvement_pp",
    "baseline_shortage_count",
    "projected_shortage_count",
    "baseline_layer_pick_pct",
    "projected_layer_pick_pct",
    "baseline_hand_pick_pct",
    "projected_hand_pick_pct",
    "layer_pick_improvement_pp",
    "baseline_wave_layer_pick_pct",
    "projected_wave_layer_pick_pct",
    "wave_layer_pick_improvement_pp",
    "baseline_wave_shortage_count",
    "projected_wave_shortage_count",
    "wave_layers_shifted_to_layer_pick",
    "baseline_layer_demand",
    "projected_layer_demand",
    "layers_shifted_to_layer_pick",
    "removes_applied",
    "adds_applied",
    "actions_total",
]


def _write_timeline_csv(results: list, path: Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMELINE_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "as_of": r.as_of.isoformat(sep=" ", timespec="seconds"),
                    "baseline_dead_slot_rate_pct": round(r.baseline.dead_slot_rate * 100, 2),
                    "projected_dead_slot_rate_pct": round(r.projected.dead_slot_rate * 100, 2),
                    "improvement_pp": round(r.dead_slot_rate_improvement_pct, 2),
                    "baseline_shortage_count": r.baseline.shortage_count,
                    "projected_shortage_count": r.projected.shortage_count,
                    "baseline_layer_pick_pct": round(r.baseline_throughput.layer_pick_pct * 100, 2),
                    "projected_layer_pick_pct": round(r.projected_throughput.layer_pick_pct * 100, 2),
                    "baseline_hand_pick_pct": round(r.baseline_throughput.hand_pick_pct * 100, 2),
                    "projected_hand_pick_pct": round(r.projected_throughput.hand_pick_pct * 100, 2),
                    "layer_pick_improvement_pp": round(r.layer_pick_improvement_pp, 2),
                    "baseline_wave_layer_pick_pct": round(
                        r.baseline_wave_layer_pick_pct * 100, 2
                    ),
                    "projected_wave_layer_pick_pct": round(r.wave_layer_pick_pct * 100, 2),
                    "wave_layer_pick_improvement_pp": round(
                        r.wave_layer_pick_improvement_pp, 2
                    ),
                    "baseline_wave_shortage_count": r.baseline_wave_shortage_count,
                    "projected_wave_shortage_count": r.wave_shortage_count,
                    "wave_layers_shifted_to_layer_pick": r.wave_layers_shifted_to_layer_pick,
                    "baseline_layer_demand": r.baseline_throughput.total_layer_demand,
                    "projected_layer_demand": r.projected_throughput.total_layer_demand,
                    "layers_shifted_to_layer_pick": r.layers_shifted_to_layer_pick,
                    "removes_applied": r.removes_applied,
                    "adds_applied": r.adds_applied,
                    "actions_total": len(r.applied),
                }
            )


LOW_LAYERS_COMPARE_COLUMNS = [
    "scenario",
    "as_of",
    "projected_dead_slot_rate_pct",
    "projected_layer_pick_pct",
    "projected_hand_pick_pct",
    "low_layer_slotted_layers",
    "removes_applied",
    "low_layer_removes_applied",
    "adds_applied",
    "layers_shifted_to_layer_pick",
]


def _write_low_layers_compare_csv(
    # Annotated `object` at one point, which silently disabled type checking
    # on all 13 attribute reads below - `result.projected_throughput.<anything>`
    # was unverifiable. Callers pass `simulate_once(...)`, which returns a
    # SimulationResult; saying so is what makes the body checkable.
    with_result: SimulationResult,
    without_result: SimulationResult,
    path: Path,
) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOW_LAYERS_COMPARE_COLUMNS)
        writer.writeheader()
        for scenario, result in (("with_low_layers", with_result), ("without_low_layers", without_result)):
            writer.writerow(
                {
                    "scenario": scenario,
                    "as_of": result.as_of.isoformat(sep=" ", timespec="seconds"),
                    "projected_dead_slot_rate_pct": round(result.projected.dead_slot_rate * 100, 2),
                    "projected_layer_pick_pct": round(
                        result.projected_throughput.layer_pick_pct * 100, 2
                    ),
                    "projected_hand_pick_pct": round(
                        result.projected_throughput.hand_pick_pct * 100, 2
                    ),
                    "low_layer_slotted_layers": result.projected_throughput.low_layer_slotted_layers,
                    "removes_applied": result.removes_applied,
                    "low_layer_removes_applied": result.low_layer_removes_applied,
                    "adds_applied": result.adds_applied,
                    "layers_shifted_to_layer_pick": result.layers_shifted_to_layer_pick,
                }
            )


BACKTEST_COMPARE_COLUMNS = [
    "as_of",
    "with_projected_dead_slot_rate_pct",
    "without_projected_dead_slot_rate_pct",
    "with_projected_layer_pick_pct",
    "without_projected_layer_pick_pct",
    "with_low_layer_removes",
    "without_low_layer_removes",
    "with_removes_applied",
    "without_removes_applied",
]


def _write_backtest_compare_csv(with_steps: list, without_steps: list, path: Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BACKTEST_COMPARE_COLUMNS)
        writer.writeheader()
        for with_step, without_step in zip(with_steps, without_steps, strict=True):
            writer.writerow(
                {
                    "as_of": with_step.as_of.isoformat(sep=" ", timespec="seconds"),
                    "with_projected_dead_slot_rate_pct": round(
                        with_step.projected.dead_slot_rate * 100, 2
                    ),
                    "without_projected_dead_slot_rate_pct": round(
                        without_step.projected.dead_slot_rate * 100, 2
                    ),
                    "with_projected_layer_pick_pct": round(
                        with_step.projected_throughput.layer_pick_pct * 100, 2
                    ),
                    "without_projected_layer_pick_pct": round(
                        without_step.projected_throughput.layer_pick_pct * 100, 2
                    ),
                    "with_low_layer_removes": with_step.low_layer_removes_applied,
                    "without_low_layer_removes": without_step.low_layer_removes_applied,
                    "with_removes_applied": with_step.removes_applied,
                    "without_removes_applied": without_step.removes_applied,
                }
            )


def _load_snapshot_folders(snapshots_dir: Path) -> list[tuple[datetime, object]]:
    """Load each ISO-named subfolder as a WorldState."""
    snapshots: list[tuple[datetime, object]] = []
    for child in sorted(snapshots_dir.iterdir()):
        if not child.is_dir():
            continue
        as_of = datetime.fromisoformat(child.name.replace("_", " "))
        state = CsvDataSource(child).load_world_state(as_of=as_of)
        snapshots.append((as_of, state))
    return snapshots


def cmd_backtest(args: argparse.Namespace) -> int:
    console = Console()
    start = datetime.fromisoformat(args.from_time)
    end = datetime.fromisoformat(args.to_time)
    step = timedelta(hours=args.step_hours)
    mode = BacktestMode(args.mode)
    rule_config = parse_disabled_rules(args.without_rules)

    console.print("[bold]Slotting Historical Backtest[/bold]\n")
    height_limits = resolve_height_limits(args.skus_path)

    if args.compare_low_layers:
        if args.snapshots_dir:
            console.print("[red]Error:[/red] --compare-low-layers requires --data-dir mode, not snapshots.")
            return 1
        source = CsvDataSource(args.data_dir)
        initial = source.load_world_state(as_of=start)
        with pallet_height_limits(height_limits):
            report_without = backtest_historical(
                initial, start=start, end=end, step=step, mode=mode, rule_config=DEFAULT_RULE_CONFIG
            )
            report_with = backtest_historical(
                initial, start=start, end=end, step=step, mode=mode, rule_config=WITH_LOW_LAYERS
            )
        print_backtest_low_layers_comparison(console, report_with, report_without)
        console.print("[bold]Full backtest: with LOW_LAYERS rule[/bold]\n")
        print_backtest_summary(console, report_with)
        console.print("[bold]Full backtest: without LOW_LAYERS rule[/bold]\n")
        print_backtest_summary(console, report_without)
        if args.output:
            _write_backtest_compare_csv(report_with.steps, report_without.steps, args.output)
            console.print(f"[dim]Step comparison written to {args.output}[/dim]")
        return 0

    if args.snapshots_dir:
        snapshots = _load_snapshot_folders(args.snapshots_dir)
        if mode == BacktestMode.CHAINED:
            console.print("[yellow]Warning:[/yellow] CHAINED mode ignored for snapshot dirs; using STATIC.")
        report = backtest_snapshot_dir(snapshots, mode=BacktestMode.STATIC, rule_config=rule_config)
    else:
        source = CsvDataSource(args.data_dir)
        initial = source.load_world_state(as_of=start)
        with pallet_height_limits(height_limits):
            report = backtest_historical(
                initial, start=start, end=end, step=step, mode=mode, rule_config=rule_config
            )

    if args.without_rules:
        disabled = ", ".join(rule_config.disabled)
        console.print(f"[dim]Disabled rules: {disabled}[/dim]\n")

    print_backtest_summary(console, report)

    if args.output:
        _write_timeline_csv(report.steps, args.output)
        console.print(f"[dim]Step detail written to {args.output}[/dim]")
    return 0


def _resolve_optimize_inputs(args: argparse.Namespace) -> tuple[Path, LayoutConstraints, timedelta | None]:
    orders_path = args.orders
    wave_hours = args.wave_hours
    if args.profile:
        from layout_advisor.normalize.io import read_site_profile
        from layout_advisor.config import load_defaults

        profile_path = Path(args.profile).resolve()
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile not found: {profile_path}\n"
                "Run from the repo root, or pass an absolute path to site_profile.yaml."
            )
        profile = read_site_profile(profile_path)
        if orders_path is None:
            if profile.orders_slotting_path and Path(profile.orders_slotting_path).exists():
                orders_path = Path(profile.orders_slotting_path)
            elif profile.orders_path and Path(profile.orders_path).exists():
                orders_path = Path(profile.orders_path)
            else:
                slotting_orders = profile.orders_slotting_path or "(not set)"
                raw_orders = profile.orders_path or "(not set)"
                raise FileNotFoundError(
                    "No orders file found. Profile lists:\n"
                    f"  orders_slotting_path: {slotting_orders}\n"
                    f"  orders_path: {raw_orders}\n"
                    "Pass --orders explicitly, or run the layout-advisor normalize step first."
                )
        constraints = constraints_from_site_profile(
            profile,
            unlimited_area=args.no_area_limit,
        )
        if args.area_sqft is not None:
            constraints = LayoutConstraints(
                area_sqft=args.area_sqft,
                source_slot_sqft=constraints.source_slot_sqft,
                donor_slot_sqft=constraints.donor_slot_sqft,
                source_range=constraints.source_range,
                donor_range=constraints.donor_range,
            )
        if wave_hours is None:
            wave_hours = float(load_defaults().wave_horizon_hours)
        return orders_path, constraints, timedelta(hours=wave_hours) if wave_hours else None

    if orders_path is None:
        raise ValueError("--orders is required when --profile is not provided.")

    if not args.no_area_limit and (
        args.area_sqft is None or args.source_range is None or args.donor_range is None
    ):
        raise ValueError(
            "Provide --profile, or --no-area-limit with --orders, "
            "or all of --area-sqft, --source-range, and --donor-range"
        )
    constraints = LayoutConstraints(
        area_sqft=None if args.no_area_limit else args.area_sqft,
        source_slot_sqft=args.source_slot_sqft,
        donor_slot_sqft=args.donor_slot_sqft,
        source_range=tuple(args.source_range or (1, 200)),
        donor_range=tuple(args.donor_range or (3, 12)),
    )
    if wave_hours is None:
        wave_hours = 4.0
    return orders_path, constraints, timedelta(hours=wave_hours)


def cmd_optimize(args: argparse.Namespace) -> int:
    console = Console()
    try:
        orders_path, constraints, wave_horizon = _resolve_optimize_inputs(args)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    mode = OptimizeMode(args.mode)
    console.print(f"[bold]Slotting Layout Sweep[/bold]: {mode.value}\n")
    console.print(f"Orders: [cyan]{orders_path}[/cyan]")
    area_label = (
        "none (demand-driven)"
        if constraints.unlimited_area
        else f"{constraints.area_sqft:.0f} sqft"
    )
    console.print(
        f"Area limit: {area_label} | "
        f"wave window: {wave_horizon.total_seconds() / 3600:.1f}h | "
        f"source {constraints.source_range[0]}-{constraints.source_range[1]} | "
        f"donor {constraints.donor_range[0]}-{constraints.donor_range[1]}\n"
    )

    out_dir = Path(args.out)

    if args.fix_source is not None:
        console.print(
            f"[bold]Donor sweep[/bold] at fixed source={args.fix_source} "
            f"(build-face-aware wave LP, target {args.target_wave_lp_pct:.0f}%)\n"
        )
        donor_result = run_donor_sweep(
            orders_path=orders_path,
            fixed_source_slots=args.fix_source,
            constraints=constraints,
            exclusions_path=args.exclusions,
            target_build_face_wave_lp_pct=args.target_wave_lp_pct,
            wave_horizon=wave_horizon,
            donor_range=tuple(args.donor_range) if args.donor_range else None,
            skus_path=args.skus_path,
        )
        csv_path = out_dir / "donor_sweep_results.csv"
        write_donor_sweep_csv(donor_result, csv_path)
        profile = donor_result.throughput_profile
        console.print(
            f"Peak wave demand: {profile.peak_wave_layer_demand} layers | "
            f"peak SKUs: {profile.peak_concurrent_skus} | "
            f"min donor hint: {profile.min_donor_slots_hint} | "
            f"score as_of: {donor_result.score_as_of.isoformat(sep=' ', timespec='seconds')}\n"
        )
        if donor_result.winner is None:
            console.print("[yellow]No donor count met the build-face target.[/yellow]")
        else:
            donor_winner = donor_result.winner
            console.print(
                f"[green]Winner:[/green] {donor_winner.ratio} "
                f"({donor_winner.footprint_sqft:.0f} sqft): "
                f"source LP {donor_winner.projected_wave_layer_pick_pct:.1f}% | "
                f"build-face LP {donor_winner.projected_build_face_wave_lp_pct:.1f}% | "
                f"dest blocked {donor_winner.dest_blocked_sku_count} SKUs | "
                f"{donor_winner.build_faces_opened} build faces"
            )
            if not donor_winner.meets_build_face_target:
                console.print(
                    "[yellow]Note:[/yellow] best available layout still below "
                    f"{args.target_wave_lp_pct:.0f}% build-face target."
                )
        console.print(f"\n[dim]Donor sweep written to {csv_path}[/dim]")
        return 0

    result = run_sweep(
        orders_path=orders_path,
        constraints=constraints,
        mode=mode,
        exclusions_path=args.exclusions,
        target_wave_lp_pct=args.target_wave_lp_pct,
        max_shortage_count=args.max_shortage,
        wave_horizon=wave_horizon,
        skus_path=args.skus_path,
    )

    out_dir = Path(args.out)
    csv_path = out_dir / "sweep_results.csv"
    write_sweep_results_csv(result, csv_path)

    profile = result.throughput_profile
    console.print(
        f"Peak wave demand: {profile.peak_wave_layer_demand} layers | "
        f"peak SKUs: {profile.peak_concurrent_skus} | "
        f"score as_of: {result.score_as_of.isoformat(sep=' ', timespec='seconds')}\n"
    )

    if result.winner is None:
        console.print("[yellow]No layout met the target constraints.[/yellow]")
    else:
        w = result.winner
        console.print(
            f"[green]Winner:[/green] {w.ratio} ({w.footprint_sqft:.0f} sqft): "
            f"wave LP {w.projected_wave_layer_pick_pct:.1f}% | "
            f"shortage {w.projected_shortage_count} | "
            f"+{w.layers_shifted_to_layer_pick} layers shifted"
        )

    console.print(f"\n[dim]Full sweep written to {csv_path}[/dim]")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(cmd_run(args))
    if args.command == "simulate":
        raise SystemExit(cmd_simulate(args))
    if args.command == "backtest":
        raise SystemExit(cmd_backtest(args))
    if args.command == "optimize":
        raise SystemExit(cmd_optimize(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
