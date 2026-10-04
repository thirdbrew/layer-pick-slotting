"""Rich console report for recommendations, KPIs, and simulation output."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from slotting.metrics import RunMetrics
from slotting.models import Recommendation
from slotting.backtest import BacktestReport
from slotting.simulation import SimulationResult
from slotting.wave_ratio import WaveRatioAdvice


def print_metrics(console: Console, metrics: RunMetrics) -> None:
    """Render KPI summary at the top of a run report."""
    table = Table(title="Run Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("As of", metrics.as_of.isoformat(sep=" ", timespec="seconds"))
    table.add_row("Dead slot rate", f"{metrics.dead_slot_rate * 100:.1f}%")
    table.add_row("Slot utilization", f"{metrics.slot_utilization * 100:.1f}%")
    table.add_row("Shortage count (waved, no supply)", str(metrics.shortage_count))
    table.add_row(
        "Source slots",
        f"{metrics.source_slots_occupied}/{metrics.source_slots_total} occupied "
        f"({metrics.dead_slots} dead)",
    )
    console.print(table)
    console.print()


def print_recommendations(console: Console, recommendations: list[Recommendation]) -> None:
    """Render advisory recommendations in a table."""
    table = Table(title="Recommendations (advisory only)", show_header=True, header_style="bold")
    table.add_column("Action", style="bold")
    table.add_column("Pallet")
    table.add_column("Item")
    table.add_column("Slot")
    table.add_column("Rule")
    table.add_column("Reason")

    action_style = {
        "FLAG_REMOVE": "red",
        "DO_NOT_TOUCH": "yellow",
        "FLAG_ADD": "green",
        "FLAG_WAIT": "magenta",
        "FLAG_ACTIVATE_DEST": "cyan",
        "FLAG_DEACTIVATE_DEST": "blue",
    }

    if not recommendations:
        console.print("[dim]No recommendations for the current state.[/dim]")
        return

    for rec in recommendations:
        style = action_style.get(rec.action.value, "")
        table.add_row(
            f"[{style}]{rec.action.value}[/{style}]" if style else rec.action.value,
            rec.pallet_id or "-",
            rec.item_id or "-",
            rec.slot_id or "-",
            rec.rule.value,
            rec.reason,
        )
    console.print(table)


def _delta_str(baseline: float | int, projected: float | int, *, lower_is_better: bool = True) -> str:
    diff = float(projected) - float(baseline)
    if diff == 0:
        return "[dim]no change[/dim]"
    improved = diff < 0 if lower_is_better else diff > 0
    sign = "+" if diff > 0 else ""
    color = "green" if improved else "red"
    if isinstance(baseline, float):
        return f"[{color}]{sign}{diff:.1f}[/{color}]"
    return f"[{color}]{sign}{int(diff)}[/{color}]"


def print_wave_ratio(console: Console, advice: WaveRatioAdvice) -> None:
    """Render physical vs active source:destination ratio for the current wave."""
    table = Table(title="Wave Ratio", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Built layout", advice.physical_ratio)
    table.add_row("Active this wave", f"[bold]{advice.active_ratio}[/bold]")
    table.add_row("Waved SKUs", str(advice.waved_sku_count))
    table.add_row("Waved layer demand", str(advice.waved_layer_demand))
    table.add_row("Supply-ready SKUs", str(advice.supply_ready_sku_count))
    table.add_row("Build faces in use", str(advice.build_faces_in_use))
    table.add_row("Source headroom", str(advice.source_headroom))
    table.add_row("Destination headroom", str(advice.destination_headroom))
    console.print(table)
    console.print()


def print_simulation(console: Console, result: SimulationResult) -> None:
    """Render baseline vs projected KPI comparison and actions taken."""
    if result.wave_ratio is not None:
        print_wave_ratio(console, result.wave_ratio)

    table = Table(title="Simulation: Baseline vs If Recommendations Applied", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Baseline", justify="right")
    table.add_column("Projected", justify="right")
    table.add_column("Delta", justify="right")

    b, p = result.baseline, result.projected
    table.add_row(
        "Dead slot rate",
        f"{b.dead_slot_rate * 100:.1f}%",
        f"{p.dead_slot_rate * 100:.1f}%",
        _delta_str(b.dead_slot_rate * 100, p.dead_slot_rate * 100),
    )
    table.add_row(
        "Dead slots",
        str(b.dead_slots),
        str(p.dead_slots),
        _delta_str(b.dead_slots, p.dead_slots),
    )
    table.add_row(
        "Slot utilization",
        f"{b.slot_utilization * 100:.1f}%",
        f"{p.slot_utilization * 100:.1f}%",
        _delta_str(b.slot_utilization * 100, p.slot_utilization * 100, lower_is_better=False),
    )
    table.add_row(
        "Shortage count (waved, no supply)",
        str(b.shortage_count),
        str(p.shortage_count),
        _delta_str(b.shortage_count, p.shortage_count),
    )
    table.add_row(
        "Waved items ready in slot",
        str(result.waved_ready_baseline),
        str(result.waved_ready_projected),
        _delta_str(result.waved_ready_baseline, result.waved_ready_projected, lower_is_better=False),
    )
    console.print(table)
    console.print()

    bt, pt = result.baseline_throughput, result.projected_throughput
    labor = Table(title="Throughput / Labor Mix (24h layer demand)", show_header=True)
    labor.add_column("Metric", style="dim")
    labor.add_column("Baseline", justify="right")
    labor.add_column("Projected", justify="right")
    labor.add_column("Delta", justify="right")

    labor.add_row(
        "Total layer demand",
        str(bt.total_layer_demand),
        str(pt.total_layer_demand),
        _delta_str(bt.total_layer_demand, pt.total_layer_demand, lower_is_better=False),
    )
    labor.add_row(
        "Layer pick throughput %",
        f"{bt.layer_pick_pct * 100:.1f}%",
        f"{pt.layer_pick_pct * 100:.1f}%",
        _delta_str(bt.layer_pick_pct * 100, pt.layer_pick_pct * 100, lower_is_better=False),
    )
    labor.add_row(
        "Hand pick throughput %",
        f"{bt.hand_pick_pct * 100:.1f}%",
        f"{pt.hand_pick_pct * 100:.1f}%",
        _delta_str(bt.hand_pick_pct * 100, pt.hand_pick_pct * 100),
    )
    labor.add_row(
        "Layer-pick-ready layers",
        str(bt.layer_pick_ready_layers),
        str(pt.layer_pick_ready_layers),
        _delta_str(bt.layer_pick_ready_layers, pt.layer_pick_ready_layers, lower_is_better=False),
    )
    labor.add_row(
        "Hand pick - unstaged (ADD fixes)",
        str(bt.hand_pick_unstaged_layers),
        str(pt.hand_pick_unstaged_layers),
        _delta_str(bt.hand_pick_unstaged_layers, pt.hand_pick_unstaged_layers),
    )
    labor.add_row(
        "Hand pick - low-layer in slot (REMOVE fixes)",
        str(bt.low_layer_slotted_layers),
        str(pt.low_layer_slotted_layers),
        _delta_str(bt.low_layer_slotted_layers, pt.low_layer_slotted_layers),
    )
    labor.add_row(
        "Hand pick - excluded items",
        str(bt.hand_pick_excluded_layers),
        str(pt.hand_pick_excluded_layers),
        _delta_str(bt.hand_pick_excluded_layers, pt.hand_pick_excluded_layers),
    )
    labor.add_row(
        "Hand pick - supply shortage",
        str(bt.hand_pick_shortage_layers),
        str(pt.hand_pick_shortage_layers),
        _delta_str(bt.hand_pick_shortage_layers, pt.hand_pick_shortage_layers),
    )
    console.print(labor)
    console.print()

    bwt, pwt = result.baseline_wave_throughput, result.projected_wave_throughput
    wave = Table(title="Wave Throughput / Labor Mix (4h waved demand)", show_header=True)
    wave.add_column("Metric", style="dim")
    wave.add_column("Baseline", justify="right")
    wave.add_column("Projected", justify="right")
    wave.add_column("Delta", justify="right")

    wave.add_row(
        "Total waved layer demand",
        str(bwt.total_layer_demand),
        str(pwt.total_layer_demand),
        _delta_str(bwt.total_layer_demand, pwt.total_layer_demand, lower_is_better=False),
    )
    wave.add_row(
        "Wave layer pick throughput %",
        f"{bwt.layer_pick_pct * 100:.1f}%",
        f"{pwt.layer_pick_pct * 100:.1f}%",
        _delta_str(bwt.layer_pick_pct * 100, pwt.layer_pick_pct * 100, lower_is_better=False),
    )
    wave.add_row(
        "Wave layer-pick-ready layers",
        str(bwt.layer_pick_ready_layers),
        str(pwt.layer_pick_ready_layers),
        _delta_str(
            bwt.layer_pick_ready_layers,
            pwt.layer_pick_ready_layers,
            lower_is_better=False,
        ),
    )
    wave.add_row(
        "Wave hand pick - unstaged (ADD fixes)",
        str(bwt.hand_pick_unstaged_layers),
        str(pwt.hand_pick_unstaged_layers),
        _delta_str(bwt.hand_pick_unstaged_layers, pwt.hand_pick_unstaged_layers),
    )
    wave.add_row(
        "Wave hand pick - supply shortage",
        str(bwt.hand_pick_shortage_layers),
        str(pwt.hand_pick_shortage_layers),
        _delta_str(bwt.hand_pick_shortage_layers, pwt.hand_pick_shortage_layers),
    )
    wave.add_row(
        "Waved item shortage count",
        str(result.baseline_wave_shortage_count),
        str(result.wave_shortage_count),
        _delta_str(result.baseline_wave_shortage_count, result.wave_shortage_count),
    )
    console.print(wave)
    console.print()

    summary = Table(title="Expected Improvement Summary", show_header=False)
    summary.add_column("Label", style="bold")
    summary.add_column("Value")
    dead_imp = result.dead_slot_rate_improvement_pct
    dead_color = "green" if dead_imp > 0 else "dim" if dead_imp == 0 else "red"
    summary.add_row(
        "Dead slot rate improvement",
        f"[{dead_color}]{dead_imp:+.1f} pp[/{dead_color}]",
    )
    sh_imp = result.shortage_improvement
    sh_color = "green" if sh_imp > 0 else "dim" if sh_imp == 0 else "red"
    summary.add_row("Shortage reduction", f"[{sh_color}]{sh_imp:+d}[/{sh_color}]")
    lp_imp = result.layer_pick_improvement_pp
    lp_color = "green" if lp_imp > 0 else "dim" if lp_imp == 0 else "red"
    summary.add_row(
        "Layer pick throughput gain",
        f"[{lp_color}]{lp_imp:+.1f} pp[/{lp_color}]",
    )
    wlp_imp = result.wave_layer_pick_improvement_pp
    wlp_color = "green" if wlp_imp > 0 else "dim" if wlp_imp == 0 else "red"
    summary.add_row(
        "Wave layer pick throughput gain",
        f"[{wlp_color}]{wlp_imp:+.1f} pp[/{wlp_color}]",
    )
    summary.add_row(
        "Layers shifted to layer pick",
        f"{result.layers_shifted_to_layer_pick:+d}",
    )
    summary.add_row(
        "Wave layers shifted to layer pick",
        f"{result.wave_layers_shifted_to_layer_pick:+d}",
    )
    summary.add_row("Removes applied", str(result.removes_applied))
    summary.add_row("Adds applied", str(result.adds_applied))
    if result.blocked_adds:
        summary.add_row(
            "Adds blocked (no empty slot)",
            str(len(result.blocked_adds)),
        )
    console.print(summary)
    console.print()

    if result.applied:
        actions = Table(title="Simulated Actions", show_header=True)
        actions.add_column("Action")
        actions.add_column("Pallet")
        actions.add_column("Item")
        actions.add_column("Slot")
        actions.add_column("Detail")
        for act in result.applied:
            actions.add_row(
                act.action.value,
                act.pallet_id or "-",
                act.item_id or "-",
                act.slot_id or "-",
                act.detail,
            )
        console.print(actions)
        console.print()

    if result.blocked_adds:
        console.print(
            f"[yellow]Warning:[/yellow] {len(result.blocked_adds)} FLAG_ADD recommendation(s) "
            f"could not be simulated: no empty source slot available after removals."
        )
        console.print()


def print_simulation_timeline(console: Console, results: list[SimulationResult]) -> None:
    """Render improvement potential across a time replay."""
    if not results:
        console.print("[dim]No timeline results.[/dim]")
        return

    table = Table(title="Timeline Simulation", show_header=True, header_style="bold cyan")
    table.add_column("As of")
    table.add_column("Baseline dead %", justify="right")
    table.add_column("Projected dead %", justify="right")
    table.add_column("Improvement pp", justify="right")
    table.add_column("Baseline shortage", justify="right")
    table.add_column("Projected shortage", justify="right")
    table.add_column("Baseline LP %", justify="right")
    table.add_column("Projected LP %", justify="right")
    table.add_column("Actions", justify="right")

    for r in results:
        imp = r.dead_slot_rate_improvement_pct
        imp_style = "green" if imp > 0 else "dim"
        table.add_row(
            r.as_of.strftime("%Y-%m-%d %H:%M"),
            f"{r.baseline.dead_slot_rate * 100:.1f}",
            f"{r.projected.dead_slot_rate * 100:.1f}",
            f"[{imp_style}]{imp:+.1f}[/{imp_style}]",
            str(r.baseline.shortage_count),
            str(r.projected.shortage_count),
            f"{r.baseline_throughput.layer_pick_pct * 100:.1f}",
            f"{r.projected_throughput.layer_pick_pct * 100:.1f}",
            str(len(r.applied)),
        )
    console.print(table)


def print_backtest_summary(console: Console, report: BacktestReport) -> None:
    """Render aggregate historical backtest results."""
    from slotting.backtest import BacktestMode

    table = Table(
        title="Historical Backtest: Rules Impact Summary",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Mode", report.mode.value)
    table.add_row("Period", f"{report.start.date()} -> {report.end.date()}")
    table.add_row("Steps", str(report.step_count))
    table.add_row("Avg baseline dead slot rate", f"{report.avg_baseline_dead_slot_rate_pct:.1f}%")
    table.add_row("Avg projected dead slot rate", f"{report.avg_projected_dead_slot_rate_pct:.1f}%")
    table.add_row("Avg improvement per step", f"{report.avg_improvement_pp:+.1f} pp")
    table.add_row(
        "Steps with dead-slot improvement",
        f"{report.steps_with_dead_slot_improvement}/{report.step_count}",
    )
    table.add_row("Total shortage reduction (sum of steps)", str(report.total_shortage_reduction))
    table.add_row("Total simulated removes", str(report.total_removes_applied))
    table.add_row("Total simulated adds", str(report.total_adds_applied))
    table.add_row(
        "Avg baseline layer pick throughput",
        f"{report.avg_baseline_layer_pick_pct:.1f}%",
    )
    table.add_row(
        "Avg projected layer pick throughput",
        f"{report.avg_projected_layer_pick_pct:.1f}%",
    )
    lp_imp = report.avg_layer_pick_improvement_pp
    lp_color = "green" if lp_imp > 0 else "dim"
    table.add_row(
        "Avg layer pick gain per step",
        f"[{lp_color}]{lp_imp:+.1f} pp[/{lp_color}]",
    )
    table.add_row(
        "Total layers shifted to layer pick (sum of steps)",
        str(report.total_layers_shifted_to_layer_pick),
    )
    table.add_row(
        "Avg baseline wave layer pick throughput",
        f"{report.avg_baseline_wave_layer_pick_pct:.1f}%",
    )
    table.add_row(
        "Avg projected wave layer pick throughput",
        f"{report.avg_projected_wave_layer_pick_pct:.1f}%",
    )
    wlp_imp = report.avg_wave_layer_pick_improvement_pp
    wlp_color = "green" if wlp_imp > 0 else "dim"
    table.add_row(
        "Avg wave layer pick gain per step",
        f"[{wlp_color}]{wlp_imp:+.1f} pp[/{wlp_color}]",
    )
    table.add_row(
        "Total wave layers shifted to layer pick (sum of steps)",
        str(report.total_wave_layers_shifted_to_layer_pick),
    )

    if report.mode == BacktestMode.CHAINED and report.cumulative_dead_slot_gap_pp is not None:
        gap = report.cumulative_dead_slot_gap_pp
        color = "green" if gap > 0 else "dim"
        table.add_row(
            "Cumulative dead-slot gap (start -> end managed)",
            f"[{color}]{gap:+.1f} pp[/{color}]",
        )
        if report.final_managed_dead_slot_rate_pct is not None:
            table.add_row(
                "Final managed dead slot rate",
                f"{report.final_managed_dead_slot_rate_pct:.1f}%",
            )

    console.print(table)
    console.print()
    print_simulation_timeline(console, report.steps)


def _rule_compare_delta(with_value: float, without_value: float, *, higher_is_better: bool = True) -> str:
    diff = with_value - without_value
    if abs(diff) < 0.05:
        return "no change"
    improved = diff > 0 if higher_is_better else diff < 0
    sign = "+" if diff > 0 else ""
    color = "green" if improved else "red"
    if abs(with_value) <= 100 and abs(without_value) <= 100 and isinstance(with_value, float):
        return f"[{color}]{sign}{diff:.1f}[/{color}]"
    return f"[{color}]{sign}{diff:.0f}[/{color}]"


def print_low_layers_comparison(
    console: Console,
    with_rule: SimulationResult,
    without_rule: SimulationResult,
) -> None:
    """Side-by-side simulation: all rules vs all rules except LOW_LAYERS."""
    table = Table(
        title="LOW_LAYERS Rule Comparison (>2 layers required for layer pick)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric (projected after rules)", style="dim")
    table.add_column("With LOW_LAYERS", justify="right")
    table.add_column("Without LOW_LAYERS", justify="right")
    table.add_column("Delta (with - without)", justify="right")

    wp, wop = with_rule.projected, without_rule.projected
    wtp, wotp = with_rule.projected_throughput, without_rule.projected_throughput

    def row(
        label: str,
        with_num: float,
        without_num: float,
        *,
        higher_is_better: bool = True,
        fmt: str = ".1f",
        suffix: str = "",
    ) -> None:
        table.add_row(
            label,
            f"{with_num:{fmt}}{suffix}",
            f"{without_num:{fmt}}{suffix}",
            _rule_compare_delta(with_num, without_num, higher_is_better=higher_is_better),
        )

    row(
        "Dead slot rate",
        wp.dead_slot_rate * 100,
        wop.dead_slot_rate * 100,
        higher_is_better=False,
        suffix="%",
    )
    row("Dead slots", float(wp.dead_slots), float(wop.dead_slots), higher_is_better=False, fmt=".0f")
    row(
        "Layer pick throughput",
        wtp.layer_pick_pct * 100,
        wotp.layer_pick_pct * 100,
        suffix="%",
    )
    row(
        "Hand pick throughput",
        wtp.hand_pick_pct * 100,
        wotp.hand_pick_pct * 100,
        higher_is_better=False,
        suffix="%",
    )
    row(
        "Low-layer slotted (hand pick layers)",
        float(wtp.low_layer_slotted_layers),
        float(wotp.low_layer_slotted_layers),
        higher_is_better=False,
        fmt=".0f",
    )
    row(
        "Total removes applied",
        float(with_rule.removes_applied),
        float(without_rule.removes_applied),
        higher_is_better=False,
        fmt=".0f",
    )
    row(
        "LOW_LAYERS removes",
        float(with_rule.low_layer_removes_applied),
        float(without_rule.low_layer_removes_applied),
        higher_is_better=False,
        fmt=".0f",
    )
    row(
        "Total adds applied",
        float(with_rule.adds_applied),
        float(without_rule.adds_applied),
        fmt=".0f",
    )
    row(
        "Layers shifted to layer pick",
        float(with_rule.layers_shifted_to_layer_pick),
        float(without_rule.layers_shifted_to_layer_pick),
        fmt=".0f",
    )

    console.print(table)
    console.print()
    console.print(
        "[dim]Baseline floor is identical in both runs; only LOW_LAYERS "
        "FLAG_REMOVE recommendations differ.[/dim]"
    )
    console.print()


def print_backtest_low_layers_comparison(
    console: Console,
    with_rule: BacktestReport,
    without_rule: BacktestReport,
) -> None:
    """Aggregate backtest comparison with vs without LOW_LAYERS."""
    table = Table(
        title="Backtest LOW_LAYERS Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="dim")
    table.add_column("With LOW_LAYERS", justify="right")
    table.add_column("Without LOW_LAYERS", justify="right")
    table.add_column("Delta", justify="right")

    def row(
        label: str,
        with_num: float,
        without_num: float,
        *,
        higher_is_better: bool = True,
        suffix: str = "",
    ) -> None:
        table.add_row(
            label,
            f"{with_num:.1f}{suffix}",
            f"{without_num:.1f}{suffix}",
            _rule_compare_delta(with_num, without_num, higher_is_better=higher_is_better),
        )

    row(
        "Avg projected dead slot rate",
        with_rule.avg_projected_dead_slot_rate_pct,
        without_rule.avg_projected_dead_slot_rate_pct,
        higher_is_better=False,
        suffix="%",
    )
    row(
        "Avg projected layer pick throughput",
        with_rule.avg_projected_layer_pick_pct,
        without_rule.avg_projected_layer_pick_pct,
        suffix="%",
    )
    row(
        "Avg layer pick gain per step",
        with_rule.avg_layer_pick_improvement_pp,
        without_rule.avg_layer_pick_improvement_pp,
        suffix=" pp",
    )
    row(
        "Total LOW_LAYERS removes (sum of steps)",
        float(sum(s.low_layer_removes_applied for s in with_rule.steps)),
        float(sum(s.low_layer_removes_applied for s in without_rule.steps)),
        higher_is_better=False,
    )
    row(
        "Total removes (sum of steps)",
        float(with_rule.total_removes_applied),
        float(without_rule.total_removes_applied),
        higher_is_better=False,
    )
    row(
        "Total adds (sum of steps)",
        float(with_rule.total_adds_applied),
        float(without_rule.total_adds_applied),
    )
    row(
        "Total layers shifted to layer pick",
        float(with_rule.total_layers_shifted_to_layer_pick),
        float(without_rule.total_layers_shifted_to_layer_pick),
    )

    console.print(table)
    console.print()
