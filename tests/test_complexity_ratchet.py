"""Complexity ratchet: the worst functions may shrink, never grow.

53 functions in `src/` exceed 100 lines and 15 exceed cyclomatic complexity 25.
That is where the expensive bugs have actually lived — `analyze_layout` shipped a
feasibility check that could not fail, and it survived review because it sat
inside a function nobody reads end to end.

**This does not demand a rewrite.** Splitting 563 lines of frozen-KPI simulation
into six functions called once each moves risk around rather than removing it,
and every number in this repo is pinned — a refactor that shifts behaviour by
accident is the exact failure mode the project keeps paying for. So the ratchet
only stops the bleeding: current offenders are recorded with their measured
size, and the suite fails if any grows or a new one appears.

Lowering a budget after a real simplification is the point. Raising one is a
decision that needs a reason in the commit message.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# Anything at or under these is unremarkable; above, it goes in the table below.
LINE_BUDGET = 100
COMPLEXITY_BUDGET = 25

# Measured at adoption. name -> (max_lines, max_complexity). Only the functions
# above BOTH thresholds are listed - the long-but-simple ones (a 1,639-line HTML
# template at complexity 4) are boring, not dangerous, and pinning them would
# just generate churn.
BUDGETS: dict[str, tuple[int, int]] = {
    "backtest_historical": (563, 64),
    "_apply_setup_shift": (319, 58),
    "run_pick_day_backtest": (425, 46),
    "_build_initial_state": (319, 45),
    "render_pick_day_portal": (1147, 41),
    "plan_rack_to_face": (264, 38),
    "plan_batched_trips": (132, 33),
    "build_daily_wave_plan_with_replay": (202, 30),
    "apply_pad_role_swaps": (113, 29),
    "apply_recommendations": (195, 27),
    "load_orders_template": (123, 27),
    "assign_faces_wave_clustered": (271, 26),
    "wave_colocated_donor_opens": (166, 26),
}


def _complexity(node: ast.AST) -> int:
    return 1 + sum(
        isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp, ast.IfExp))
        for child in ast.walk(node)
    )


def _measure() -> dict[str, tuple[int, int, str]]:
    """name -> (lines, complexity, path) for every function in src/."""
    out: dict[str, tuple[int, int, str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            lines = (node.end_lineno or node.lineno) - node.lineno
            cc = _complexity(node)
            # Same name in two modules: keep the worse one, which is the one a
            # budget has to bound anyway.
            prior = out.get(node.name)
            if prior is None or (cc, lines) > (prior[1], prior[0]):
                out[node.name] = (lines, cc, path.as_posix())
    return out


MEASURED = _measure()


@pytest.mark.parametrize("name", sorted(BUDGETS))
def test_a_recorded_offender_does_not_grow(name: str) -> None:
    max_lines, max_cc = BUDGETS[name]
    assert name in MEASURED, f"{name} no longer exists - drop its row from BUDGETS"
    lines, cc, path = MEASURED[name]
    assert lines <= max_lines, (
        f"{name} ({path}) grew to {lines} lines, budget {max_lines}. "
        "Extract something, or raise the budget with a reason in the commit."
    )
    assert cc <= max_cc, (
        f"{name} ({path}) grew to complexity {cc}, budget {max_cc}. "
        "Extract something, or raise the budget with a reason in the commit."
    )


def test_no_new_function_is_both_long_and_complex() -> None:
    offenders = {
        name: (lines, cc, path)
        for name, (lines, cc, path) in MEASURED.items()
        if lines > LINE_BUDGET and cc > COMPLEXITY_BUDGET and name not in BUDGETS
    }
    assert not offenders, (
        "New long-and-complex function(s) - split them, or add a budget row "
        "with a reason:\n"
        + "\n".join(
            f"  {n}: {ln} lines, complexity {cc} ({p})" for n, (ln, cc, p) in sorted(offenders.items())
        )
    )


def test_the_budgets_are_not_quietly_slack() -> None:
    """A budget far above the real size stops ratcheting. Tighten it instead.

    Slack accumulates the moment someone simplifies a function and leaves the
    old number in place, and a ratchet that does not ratchet is worse than none
    because it reads as protection.
    """
    slack = {
        name: (BUDGETS[name][0], MEASURED[name][0])
        for name in BUDGETS
        if name in MEASURED and MEASURED[name][0] < BUDGETS[name][0] - 40
    }
    assert not slack, (
        "Budgets have gone slack after a simplification - lower them:\n"
        + "\n".join(f"  {n}: budget {b} lines, actual {a}" for n, (b, a) in sorted(slack.items()))
    )
