"""How much LP% is left on the table: the clairvoyant ceiling, and what binds.

Rule tuning needs a stopping rule. When an arm fails its test there is
otherwise no way to tell "the policy is bad" from "the problem is already
solved". A once-a-day re-slot has a hindsight-optimal ceiling that is
straightforward to compute. Nothing answered the same question for an intraday
system.

Three reference points, each computed **exactly from the order feed** — no
simulation, no solver, no dependency:

* **B_day** — best fixed 70 SKUs for the whole ship day, chosen with hindsight.
  What a perfect nightly re-slot achieves. (Computed here alongside the
  others so all three sit on one denominator.)
* **B_wave** — best 70 SKUs *per wave*, chosen with hindsight, **moves free and
  instant**. The absolute ceiling of any intraday policy on this face count.
* **moves required** — how many face changes per wave B_wave actually needs.

Read it as a decomposition:

    B_wave - B_day     = the prize for moving at all, if moving were free
    B_wave - sim       = policy headroom (what better rules could still win)
    moves required     = the price of collecting it

Layer weighting makes this exact rather than approximate: the bar's headline is
a layer-weighted mean of per-tick ratios, and a layer-weighted mean of ratios
equals total-covered over total-demanded. So a coverage fraction computed on
totals *is* the bar's metric, with no reconciliation needed.

**What the bound assumes away, deliberately.** Inventory is assumed available
whenever a SKU is slotted — no replenishment, no reserve depth, no pallet
geometry. So B_wave is a true ceiling for the *slotting decision only*. A gap
between B_wave and the sim is not automatically winnable; part of it is
inventory physics the sim charges and this does not.

Usage:
    python scripts/lp_upper_bound.py                     # stress day, 70 faces
    python scripts/lp_upper_bound.py --chain             # every chain day
    python scripts/lp_upper_bound.py --faces 45,55,70    # against the sizing sweep
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from slotting.models import OrderArea, OrderLine, OrderStatus
from slotting.wave_release import order_ship_date, split_day_by_native_due

ORDERS = REPO / "data" / "synthetic" / "orders.csv"
PLANS = REPO / "data" / "plans"  # written by scripts/generate_daily_wave_plans.py
OUT = REPO / "data" / "sim_runs" / "lp_upper_bound.json"

# Scored days. These defaults match the seed-7 synthetic site (one Mon-Fri week).
CHAIN_DAYS = [date(2026, 3, d) for d in range(2, 7)]
STRESS = date(2026, 3, 3)
DEFAULT_FACES = 70
DUE_OFFSET_HOURS = 4.0


def load_orders(path: Path = ORDERS) -> list[OrderLine]:
    """THIS_LINE order lines, exactly the population the sim runs on."""
    orders: list[OrderLine] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("area") != "this_line":
                continue
            orders.append(
                OrderLine(
                    order_id=row["order_id"],
                    item_id=row["item_id"],
                    qty_layers=int(row["qty_layers"]),
                    due_at=datetime.fromisoformat(row["due_at"]),
                    status=OrderStatus.VISIBLE,
                    area=OrderArea.THIS_LINE,
                )
            )
    return orders


def wave_demand(orders: list[OrderLine], ship_day: date) -> list[dict[str, int]]:
    """Per-wave {item_id: layers} for one ship day, on the simulator's own wave rule."""
    day_orders = [o for o in orders if order_ship_date(o) == ship_day]
    buckets = split_day_by_native_due(
        day_orders, ship_date=ship_day, due_offset_hours=DUE_OFFSET_HOURS
    )
    waves: list[dict[str, int]] = []
    for bucket in buckets:
        demand: dict[str, int] = defaultdict(int)
        for order in bucket.orders:
            demand[order.item_id] += order.qty_layers
        waves.append(dict(demand))
    return waves


def top_k_set(demand: dict[str, int], faces: int) -> set[str]:
    """The `faces` highest-volume SKUs. Ties broken by item_id so runs are stable."""
    ranked = sorted(demand.items(), key=lambda kv: (-kv[1], kv[0]))
    return {item for item, _ in ranked[:faces]}


def covered_layers(demand: dict[str, int], slotted: set[str]) -> int:
    return sum(layers for item, layers in demand.items() if item in slotted)


def day_bounds(waves: list[dict[str, int]], faces: int) -> dict:
    """B_day, B_wave and the move bill for one ship day."""
    total = sum(sum(w.values()) for w in waves)
    if not total:
        return {
            "total_layers": 0,
            "b_day_pct": None,
            "b_wave_pct": None,
            "waves": 0,
            "max_wave_skus": 0,
            "day_skus": 0,
            "moves_total": 0,
            "moves_peak": 0,
        }

    # B_day: one fixed set for the whole day, chosen on day totals.
    day_demand: dict[str, int] = defaultdict(int)
    for wave in waves:
        for item, layers in wave.items():
            day_demand[item] += layers
    fixed = top_k_set(dict(day_demand), faces)
    b_day = 100.0 * sum(covered_layers(w, fixed) for w in waves) / total

    # B_wave: re-choose every wave, free moves. Move bill is what that costs.
    covered = 0
    previous: set[str] = set()
    moves: list[int] = []
    for wave in waves:
        chosen = top_k_set(wave, faces)
        covered += covered_layers(wave, chosen)
        moves.append(len(chosen - previous))
        previous = chosen
    b_wave = 100.0 * covered / total

    return {
        "total_layers": total,
        "b_day_pct": round(b_day, 1),
        "b_wave_pct": round(b_wave, 1),
        "waves": len(waves),
        "max_wave_skus": max(len(w) for w in waves),
        "day_skus": len(day_demand),
        "moves_total": sum(moves),
        "moves_peak": max(moves),
    }


def sim_day(ship_day: date) -> dict:
    """Committed sim results for this day: LP% and the face changes it spent."""
    path = PLANS / ship_day.isoformat() / "daily_summary.json"
    if not path.is_file():
        return {}
    kpis = json.loads(path.read_text(encoding="utf-8")).get("sim_kpis") or {}
    return {
        "sim_lp_pct": kpis.get("avg_lp_sim_pct_weighted"),
        # ADDs are face-set changes, the same currency as the bound's `moves`.
        # rack promotes are resupply of an already-slotted SKU, so they are
        # reported separately rather than folded in.
        "sim_adds": kpis.get("total_adds"),
        "sim_rack_promotes": kpis.get("total_rack_promotes"),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", action="store_true", help="Every scored chain day")
    ap.add_argument("--faces", default=str(DEFAULT_FACES), help="Comma-separated face counts")
    ap.add_argument("--orders", type=Path, default=ORDERS, help="orders.csv (default: synthetic site)")
    args = ap.parse_args()

    face_counts = [int(x) for x in args.faces.split(",")]
    days = CHAIN_DAYS if args.chain else [STRESS]
    orders = load_orders(args.orders)

    payload: dict = {"faces": face_counts, "days": {}}
    for faces in face_counts:
        print(f"=== LP upper bound at {faces} source faces ===\n")
        print(
            f"{'day':12} {'layers':>7} {'waves':>6} {'SKUs':>5} {'pk/wave':>7}  "
            f"{'B_day':>6} {'B_wave':>7} {'sim':>6} {'headroom':>9}  "
            f"{'moves':>6} {'pk':>4} {'simADD':>7}"
        )
        rows = []
        for day in days:
            waves = wave_demand(orders, day)
            bounds = day_bounds(waves, faces)
            if not bounds["total_layers"]:
                continue
            simmed_day = sim_day(day)
            sim = simmed_day.get("sim_lp_pct")
            headroom = None if sim is None else round(bounds["b_wave_pct"] - float(sim), 1)
            rows.append({"ship_date": day.isoformat(), **simmed_day, **bounds})
            adds = simmed_day.get("sim_adds")
            print(
                f"{day.isoformat():12} {bounds['total_layers']:7} {bounds['waves']:6} "
                f"{bounds['day_skus']:5} {bounds['max_wave_skus']:7}  "
                f"{bounds['b_day_pct']:6.1f} {bounds['b_wave_pct']:7.1f} "
                f"{(sim if sim is not None else float('nan')):6.1f} "
                f"{(headroom if headroom is not None else float('nan')):9.1f}  "
                f"{bounds['moves_total']:6} {bounds['moves_peak']:4} "
                f"{(adds if adds is not None else 0):7}"
            )
        payload["days"][faces] = rows

        # Chain aggregates on the bar's denominator: layer-weighted, which for a
        # coverage fraction is just total-covered over total-demanded.
        if rows:
            simmed = [r for r in rows if r.get("sim_lp_pct") is not None]

            def lw(key: str, subset: list[dict]) -> float:
                n = sum(r["total_layers"] for r in subset)
                return sum(float(r[key]) * r["total_layers"] for r in subset) / n

            # BOTH aggregations, always. One is layer-weighted across the chain,
            # the other a plain mean of per-day weighted values, and they are not
            # the same number. Quoting one against the other invents a headroom
            # figure that matches neither.
            print(
                f"\nLayer-weighted across days:  B_day {lw('b_day_pct', rows):.1f}%   "
                f"B_wave {lw('b_wave_pct', rows):.1f}%",
                end="",
            )
            if simmed:
                print(
                    f"   sim {lw('sim_lp_pct', simmed):.1f}%   "
                    f"headroom {lw('b_wave_pct', simmed) - lw('sim_lp_pct', simmed):+.1f}pp"
                )
                mean_sim = sum(float(r["sim_lp_pct"]) for r in simmed) / len(simmed)
                mean_wave = sum(r["b_wave_pct"] for r in simmed) / len(simmed)
                print(
                    f"Mean of days (the bar's rule): B_day "
                    f"{sum(r['b_day_pct'] for r in rows) / len(rows):.1f}%   "
                    f"B_wave {mean_wave:.1f}%   sim {mean_sim:.1f}%   "
                    f"headroom {mean_wave - mean_sim:+.1f}pp"
                )
                print(
                    f"Prize for moving intraday at all (B_wave - B_day, layer-weighted): "
                    f"{lw('b_wave_pct', rows) - lw('b_day_pct', rows):+.1f}pp"
                )
            else:
                print()

            need = max(r["max_wave_skus"] for r in rows)
            print(
                f"\nFaces for 100% coverage with free moves: {need} "
                f"(the busiest single wave's SKU count) against {faces} built."
            )
            bound_moves = sum(r["moves_total"] for r in rows)
            print(
                f"Price of B_wave: {bound_moves} face changes over "
                f"{sum(r['waves'] for r in rows)} waves, peak "
                f"{max(r['moves_peak'] for r in rows)} in one wave."
            )
            spent = [r for r in rows if r.get("sim_adds") is not None]
            if spent:
                sim_adds = sum(int(r["sim_adds"]) for r in spent)
                bound_on_same = sum(r["moves_total"] for r in spent)
                print(
                    f"The sim spends {sim_adds} ADDs on the same days - "
                    f"{100 * sim_adds / bound_on_same:.0f}% of what the bound needs "
                    f"(plus {sum(int(r['sim_rack_promotes'] or 0) for r in spent)} rack "
                    "promotes, which resupply an already-slotted SKU and are not face changes)."
                )
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        "\nCaveat: inventory is assumed available wherever a SKU is slotted. These are\n"
        "ceilings for the SLOTTING decision only - part of any gap to the sim is\n"
        "replenishment physics the sim charges and this does not."
    )


if __name__ == "__main__":
    main()
