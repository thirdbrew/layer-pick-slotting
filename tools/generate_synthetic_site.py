#!/usr/bin/env python
"""Generate a synthetic layer-pick site: SKUs, orders, slots, pallets, exclusions.

WHY THIS EXISTS
---------------
The engine in `src/` was developed against a proprietary order feed that cannot be
published. Without data it is unrunnable code, and unrunnable code cannot be evaluated.
This generator replaces that feed with a statistically plausible one, so every command
in the README executes end to end on a fresh clone.

WHAT "PLAUSIBLE" MEANS HERE, AND WHAT IT DOES NOT
-------------------------------------------------
The generator reproduces the *structural* features the slotting rules actually respond
to. It is not fitted to any real site, and no number in it was measured anywhere:

  1. ABC VELOCITY SKEW. Order lines are drawn against a Zipf-shaped popularity curve,
     so a small head of SKUs carries most of the layer demand. This is the feature that
     makes slotting a decision at all -- under uniform demand every face is worth the
     same and the optimal policy is "do nothing".

  2. WAVE STRUCTURE. Demand is not spread evenly across the day. Orders are released in
     discrete waves with a configurable ramp, because the binding constraint in a
     layer-pick operation is concurrent build-face throughput at the peak, not daily volume.

  3. A VISIBLE/WAVED HORIZON SPLIT. An order is `waved` once it is committed to the
     floor and `visible` while it is still only forecast. Several rules exist purely to
     distinguish "needed now" from "will be needed", so a generator that emitted only
     one status would leave them untested.

  4. LUMPY LINE SIZES. Layer counts per line are geometric rather than uniform: many
     small lines, a thin tail of large ones. The `min_layers_to_layer_pick` gate only
     does something when small lines exist.

  5. A FEW INELIGIBLE SKUs. Oversized items that must be hand-picked, so the exclusion
     path is exercised.

  6. BACKING STOCK BEHIND UNFACED DEMAND. A share of the resupply rack is reserved for
     SKUs that have demand but hold no source face. Without it every rack pallet backs a
     SKU that is already faced, so the promote rules have nothing to promote: the floor's
     dead-slot rate improves while the layer-pick rate -- the metric the operation is
     actually paid on -- does not move at all.

  7. STOCK DEPTH SCALED TO DEMAND. Pallets are allocated against horizon demand rather
     than drawn from the popularity curve a second time. Drawing them put one 1-to-8
     layer pallet behind the head SKU whether it wanted 1 layer that day or 151, so
     two thirds of demand was unservable from a face for a reason no slotting decision
     could fix -- the engine looked weak because the site was mis-stocked, not because
     the rules were wrong. A share of demand is left uncovered on purpose so the
     shortage rule still has an input.

Everything is seeded. The same --seed gives byte-identical output on any machine, which
is what lets a result in the README be checked rather than taken on trust.

USAGE
-----
    python tools/generate_synthetic_site.py --out data/synthetic --seed 7
    PYTHONPATH=src python -m slotting.cli run --data-dir data/synthetic --as-of ...
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults. Round numbers on purpose -- they are a starting point to vary, not
# a calibration. Nothing here is derived from any real facility.
# --------------------------------------------------------------------------
DEFAULT_SKUS = 180
DEFAULT_SOURCE_SLOTS = 100
DEFAULT_DEST_SLOTS = 40
DEFAULT_RACK_SLOTS = 60
DEFAULT_DAYS = 5
DEFAULT_WAVES_PER_DAY = 6
DEFAULT_LINES_PER_WAVE = 45
DEFAULT_EXCLUDED = 6

SHIFT_START_HOUR = 6          # first wave releases here
WAVE_SPACING_HOURS = 2
DUE_LEAD_HOURS = 4            # a wave is due this long after it releases
WAVED_HORIZON_HOURS = 8       # inside this, an order is committed; outside, forecast

ZIPF_SHAPE = 1.15             # >1; higher = sharper head. 1.15 gives ~top 20% of SKUs
                              # carrying roughly 60% of lines, the usual warehouse shape.
GEOMETRIC_P = 0.45            # line size: P(1 layer) = .45, mean ~2.2, tail to ~10
MAX_LAYERS_PER_LINE = 12
PALLET_LAYERS_FULL = 8        # a full inbound pallet, in layers
RACK_UNFACED_SHARE = 0.40     # share of rack slots reserved for backing stock behind
                              # demanded SKUs that hold no face (feature 6 above)
MEASUREMENT_HORIZON_HOURS = 24  # the window throughput is classified over
SUPPLY_COVERAGE = 0.85        # target: this share of horizon layer demand has eligible
                              # stock somewhere on the floor. Not 1.0 -- a floor that
                              # can serve everything never exercises WAIT_SUPPLY_SHORTAGE.
SQUATTER_FACE_SHARE = 0.25    # share of source faces deliberately given to slow movers
MAX_PALLETS_PER_ITEM = 5      # depth cap. The head SKU wants ~19 pallets of cover in a
                              # day; a pick rack does not hold that, it gets replenished
                              # from bulk. The uncovered remainder is real shortage.
UNFACED_DEMAND_SHARE = 0.30   # share of demanded SKUs that start the shift with stock in
                              # the rack and no face. Sizing stock to demand otherwise
                              # faces nearly every demanded SKU at t=0 and quietly
                              # deletes the promote condition (feature 6).


def _zipf_weights(n: int, shape: float) -> list[float]:
    """Popularity weights over n SKUs, normalised. Rank 1 is the fastest mover."""
    raw = [1.0 / ((i + 1) ** shape) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def _line_layers(rng: random.Random) -> int:
    """Geometric line size, clamped. Many small lines, a thin heavy tail."""
    n = 1
    while n < MAX_LAYERS_PER_LINE and rng.random() > GEOMETRIC_P:
        n += 1
    return n


def generate(
    out_dir: Path,
    seed: int,
    n_skus: int,
    n_source: int,
    n_dest: int,
    n_rack: int,
    days: int,
    waves_per_day: int,
    lines_per_wave: int,
    n_excluded: int,
    start_date: datetime,
) -> dict[str, int]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    skus = [f"SKU-{i:04d}" for i in range(1, n_skus + 1)]
    weights = _zipf_weights(n_skus, ZIPF_SHAPE)

    # ---- exclusions: drawn from the SLOW half, because an oversized fast mover
    # would be a facility design error rather than a slotting problem.
    excluded = sorted(rng.sample(skus[n_skus // 2:], k=min(n_excluded, n_skus // 2)))

    # ---- orders -------------------------------------------------------------
    orders: list[dict[str, object]] = []
    order_seq = 0
    horizon_start = start_date + timedelta(hours=SHIFT_START_HOUR)

    for day in range(days):
        day_start = start_date + timedelta(days=day, hours=SHIFT_START_HOUR)
        # A ramp across the day: the middle waves are the heavy ones. This is what
        # creates a peak for the build-face constraint to bind against.
        for w in range(waves_per_day):
            phase = (w + 0.5) / waves_per_day           # 0..1 across the shift
            ramp = 0.55 + 0.9 * (1.0 - abs(2.0 * phase - 1.0))   # tent, peaks mid-shift
            n_lines = max(1, int(round(lines_per_wave * ramp)))

            release = day_start + timedelta(hours=w * WAVE_SPACING_HOURS)
            due = release + timedelta(hours=DUE_LEAD_HOURS)

            for _ in range(n_lines):
                item = rng.choices(skus, weights=weights, k=1)[0]
                order_seq += 1
                orders.append({
                    "order_id": f"ORD-{order_seq:06d}",
                    "item_id": item,
                    "qty_layers": _line_layers(rng),
                    "due_at": due.isoformat(timespec="seconds"),
                    # committed if it falls inside the waved horizon from the very
                    # start of the run, otherwise still forecast
                    "status": ("waved"
                               if due - horizon_start <= timedelta(hours=WAVED_HORIZON_HOURS)
                               else "visible"),
                    # a minority of lines belong to a neighbouring fulfilment area and
                    # must NOT be picked here -- that is what CROSS_AREA guards
                    "area": "other_area" if rng.random() < 0.06 else "this_line",
                })

    # ---- slots --------------------------------------------------------------
    slots: list[dict[str, object]] = []
    for i in range(1, n_source + 1):
        slots.append({"slot_id": f"SRC-{i:03d}", "slot_type": "source",
                      "occupied_by_pallet_id": ""})
    for i in range(1, n_dest + 1):
        slots.append({"slot_id": f"DST-{i:03d}", "slot_type": "destination",
                      "occupied_by_pallet_id": ""})
    for i in range(1, n_rack + 1):
        slots.append({"slot_id": f"RCK-{i:03d}", "slot_type": "resupply_rack",
                      "occupied_by_pallet_id": ""})

    # ---- pallets ------------------------------------------------------------
    # Stock is allocated AGAINST DEMAND. The previous version drew face items and rack
    # items from the same Zipf curve that generated the orders, which is not the same
    # thing: it gave the head SKU a single partially depleted pallet whether that SKU
    # wanted 1 layer in the horizon or 151. Aggregate stock was ample (932 layers against
    # 539 of demand) and still only 27.6% of demand sat on a pick face, because the
    # depth was in the wrong places and 398 layers sat on SKUs with no demand at all.
    #
    # Three things have to survive this, and they pull against each other:
    #   - a deliberately imperfect floor, or the reclaim rules have nothing to reclaim;
    #   - backing stock behind unfaced demand, or the promote rules have nothing to
    #     promote (feature 6);
    #   - demand that nothing on the floor can serve, or WAIT_SUPPLY_SHORTAGE never
    #     fires. Hence SUPPLY_COVERAGE < 1 and the per-item depth cap.
    pallets: list[dict[str, object]] = []
    pallet_seq = 0
    arrival = start_date - timedelta(hours=6)

    horizon_end = horizon_start + timedelta(hours=MEASUREMENT_HORIZON_HOURS)
    horizon_demand: dict[str, int] = {}
    for o in orders:
        if o["area"] != "this_line" or o["item_id"] in excluded:
            continue
        if horizon_start <= datetime.fromisoformat(str(o["due_at"])) <= horizon_end:
            key = str(o["item_id"])
            horizon_demand[key] = horizon_demand.get(key, 0) + int(o["qty_layers"])

    n_squatter_faces = int(round(n_source * SQUATTER_FACE_SHARE))
    n_demand_faces = n_source - n_squatter_faces
    target_layers = int(round(SUPPLY_COVERAGE * sum(horizon_demand.values())))

    # Allocate one pallet at a time to whichever demanded SKU is furthest from covered.
    # Its first pallet takes a source face (partially depleted, so resupply thresholds
    # trigger); the rest go to the rack as backing stock. A SKU whose turn comes after
    # the face budget is spent is faceless with stock upstream -- feature 6, earned
    # rather than sampled.
    # A slotting engine that opens the shift with every needed SKU already on a face has
    # nothing to slot. Hold a share of them back with their stock in the rack: that is
    # the "waved SKU, no face, supply upstream" state RACK_PROMOTE exists for, and
    # sizing stock to demand erases it if you do not reserve it explicitly.
    demanded_ranked = sorted(horizon_demand, key=lambda i: (-horizon_demand[i], i))
    start_faceless = set(
        rng.sample(demanded_ranked, k=int(round(len(demanded_ranked) * UNFACED_DEMAND_SHARE)))
    )

    face_items: list[str] = []
    rack_plan: list[tuple[str, int]] = []
    face_plan: list[tuple[str, int]] = []
    allocated: dict[str, int] = {}
    covered: dict[str, int] = {}

    while sum(covered.values()) < target_layers:
        pending = [
            item for item, want in horizon_demand.items()
            if allocated.get(item, 0) < MAX_PALLETS_PER_ITEM and covered.get(item, 0) < want
        ]
        if not pending or len(face_plan) + len(rack_plan) >= n_demand_faces + n_rack:
            break
        item = max(pending, key=lambda i: (horizon_demand[i] - covered.get(i, 0), i))
        allocated[item] = allocated.get(item, 0) + 1
        if (
            item not in start_faceless
            and item not in face_items
            and len(face_plan) < n_demand_faces
        ):
            layers = rng.randint(1, PALLET_LAYERS_FULL)
            face_items.append(item)
            face_plan.append((item, layers))
        elif len(rack_plan) < n_rack:
            layers = PALLET_LAYERS_FULL
            rack_plan.append((item, layers))
        else:
            break
        covered[item] = covered.get(item, 0) + layers

    # Squatters: slow movers with no horizon demand, parked on the remaining faces.
    # This is the condition ZERO_DEMAND exists to resolve, and without it the opening
    # floor would be close to correct and the engine would look inert.
    idle = [s for s in skus[n_skus // 2:] if s not in horizon_demand]
    squatters = rng.sample(idle, k=min(n_source - len(face_plan), len(idle)))
    for item in squatters:
        face_plan.append((item, rng.randint(1, PALLET_LAYERS_FULL)))
    rng.shuffle(face_plan)

    for idx, (item, layers) in enumerate(face_plan[:n_source]):
        pallet_seq += 1
        pallets.append({
            "pallet_id": f"PLT-{pallet_seq:05d}",
            "item_id": item,
            "layers_remaining": layers,
            "current_slot": f"SRC-{idx + 1:03d}",
            "arrived_at": arrival.isoformat(timespec="seconds"),
        })
        slots[idx]["occupied_by_pallet_id"] = f"PLT-{pallet_seq:05d}"

    # Any rack slot the demand allocation did not claim goes to a demanded SKU that
    # holds no face if one is left, and to the popularity curve otherwise.
    faced_items = {item for item, _ in face_plan[:n_source]}
    promote_pool = [
        s for s in skus
        if s not in faced_items and s not in excluded and horizon_demand.get(s, 0) > 0
    ]
    promote_weights = [float(horizon_demand[s]) for s in promote_pool]
    n_unfaced = min(int(round(n_rack * RACK_UNFACED_SHARE)), len(promote_pool))

    while len(rack_plan) < n_rack:
        if len(rack_plan) < n_unfaced and promote_pool:
            item = rng.choices(promote_pool, weights=promote_weights, k=1)[0]
        else:
            item = rng.choices(skus, weights=weights, k=1)[0]
        rack_plan.append((item, PALLET_LAYERS_FULL))

    for i, (item, layers) in enumerate(rack_plan[:n_rack], start=1):
        pallet_seq += 1
        pallets.append({
            "pallet_id": f"PLT-{pallet_seq:05d}",
            "item_id": item,
            "layers_remaining": layers,
            "current_slot": f"RCK-{i:03d}",
            "arrived_at": arrival.isoformat(timespec="seconds"),
        })
        slots[n_source + n_dest + i - 1]["occupied_by_pallet_id"] = f"PLT-{pallet_seq:05d}"

    achieved = sum(
        min(want, sum(n for it, n in face_plan[:n_source] + rack_plan[:n_rack] if it == item))
        for item, want in horizon_demand.items()
    )
    horizon_total = sum(horizon_demand.values())

    # ---- write --------------------------------------------------------------
    def _write(name: str, rows: list[dict[str, object]], cols: list[str]) -> None:
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    _write("orders.csv", orders,
           ["order_id", "item_id", "qty_layers", "due_at", "status", "area"])
    _write("slots.csv", slots, ["slot_id", "slot_type", "occupied_by_pallet_id"])
    _write("pallets.csv", pallets,
           ["pallet_id", "item_id", "layers_remaining", "current_slot", "arrived_at"])
    _write("exclusions.csv",
           [{"item_id": s, "reason": "Oversized - hand pick only"} for s in excluded],
           ["item_id", "reason"])

    return {
        "skus": n_skus,
        "order_lines": len(orders),
        "layers": sum(int(o["qty_layers"]) for o in orders),
        "waved_lines": sum(1 for o in orders if o["status"] == "waved"),
        "slots": len(slots),
        "pallets": len(pallets),
        "exclusions": len(excluded),
        "horizon_demand_layers": horizon_total,
        "horizon_demand_covered_pct": round(100 * achieved / horizon_total, 1) if horizon_total else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--seed", type=int, default=7,
                   help="same seed = byte-identical output (default: 7)")
    p.add_argument("--skus", type=int, default=DEFAULT_SKUS)
    p.add_argument("--source-slots", type=int, default=DEFAULT_SOURCE_SLOTS)
    p.add_argument("--dest-slots", type=int, default=DEFAULT_DEST_SLOTS)
    p.add_argument("--rack-slots", type=int, default=DEFAULT_RACK_SLOTS)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--waves-per-day", type=int, default=DEFAULT_WAVES_PER_DAY)
    p.add_argument("--lines-per-wave", type=int, default=DEFAULT_LINES_PER_WAVE)
    p.add_argument("--excluded", type=int, default=DEFAULT_EXCLUDED)
    p.add_argument("--start", default="2026-03-02",
                   help="first shift date, YYYY-MM-DD (default: 2026-03-02, a Monday)")
    args = p.parse_args()

    stats = generate(
        out_dir=args.out, seed=args.seed, n_skus=args.skus,
        n_source=args.source_slots, n_dest=args.dest_slots, n_rack=args.rack_slots,
        days=args.days, waves_per_day=args.waves_per_day,
        lines_per_wave=args.lines_per_wave, n_excluded=args.excluded,
        start_date=datetime.fromisoformat(args.start),
    )

    print(f"wrote synthetic site to {args.out}/  (seed={args.seed})")
    for k, v in stats.items():
        print(f"  {k:14s} {v:>8,}")
    first_wave = (datetime.fromisoformat(args.start)
                  + timedelta(hours=SHIFT_START_HOUR)).isoformat(timespec="seconds")
    print(f"\nrun it:\n  PYTHONPATH=src python -m slotting.cli run "
          f"--data-dir {args.out} --as-of {first_wave}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
