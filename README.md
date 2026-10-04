# Warehouse Layer-Pick Slotting Engine

[![tests](https://github.com/thirdbrew/layer-pick-slotting/actions/workflows/tests.yml/badge.svg)](https://github.com/thirdbrew/layer-pick-slotting/actions/workflows/tests.yml)

Decision support for **layer picking** in a distribution centre: which pallet should occupy
which pick face, right now, and what should move next.

A rules engine that emits advisory `ADD` / `REMOVE` / `WAIT` recommendations against a live
floor state, a discrete-event simulator that replays a full pick day against those rules,
and the slotting and wave-planning machinery underneath both.

**84 modules · ~30k lines · 295 tests · runs end to end on generated data, no external services.**

> **Synthetic data only.** This engine was developed against a proprietary order feed that
> cannot be published. That feed is not in this repository in any form, at any point in its
> history. Everything here runs on
> [`tools/generate_synthetic_site.py`](tools/generate_synthetic_site.py), which builds a
> plausible site from a seed.

---

## The problem

In a layer-pick operation a picker takes whole **layers** off a pallet rather than
individual cases. It is far faster than case picking, but only for SKUs that are actually
*on a pick face* when their order waves. A SKU without a face gets hand-picked instead, at
a fraction of the rate.

Faces are scarce. A layout might have 100 source faces against several hundred active SKUs,
so the floor is a continuous eviction problem: every face given to one SKU is denied to
another, and the demand that decides who deserves it arrives in waves through the day.

Both naive policies fail:

- **Never evict**: the floor fills with slow movers that were needed once, and the
  layer-pick rate decays as the day goes on.
- **Always evict for the next wave**: pallets churn constantly and the movement cost
  exceeds the picking saved.

This engine is the middle: independently testable rules that each answer one narrow
question about one pallet, combined into a recommendation the floor can act on or ignore.

---

## Repository layout

| Path | What it is |
|---|---|
| [`src/slotting/`](src/slotting) | The rules engine, the what-if simulator (`simulate`), the historical backtest, and the layout sweep (`optimize`). CLI: `slotting`. |
| [`src/layout_advisor/`](src/layout_advisor) | The pick-day replay simulator, daily wave planning (setup slotting, Hungarian face assignment, pad-role schedules, clamp-trip packing), floor visualisation, and demand forecasting for layout sizing. |
| [`scripts/`](scripts) | `generate_daily_wave_plans.py` (per-wave operator plans), `lp_upper_bound.py` (clairvoyant ceilings), `chain_ab.py` (exact paired Wilcoxon / Hodges-Lehmann for comparing policy arms). |
| [`tools/generate_synthetic_site.py`](tools/generate_synthetic_site.py) | The seeded site generator every number below is computed from. |

---

## Quick start

```bash
pip install -e .
python tools/generate_synthetic_site.py --out data/synthetic --seed 7
```

Then run the engine (`pytest` runs the suite; both work from a bare checkout):

```bash
slotting run --data-dir data/synthetic --as-of 2026-03-02T06:00:00
```

That prints floor metrics, the wave ratio, and the ranked recommendations. To replay a full
day hour by hour under the **default** policy:

```bash
slotting simulate --data-dir data/synthetic --from 2026-03-02T06:00:00 --to 2026-03-03T06:00:00 --output timeline.csv
```

On the seed-7 site the opening floor carries a **77% dead-slot rate** (faces held by SKUs
with no live demand), and left alone it would layer-pick **26.3%** of the day's waved
demand. Following the default policy's recommendations takes dead slots to **12%** and the
waved layer-pick rate to **40.0%**; the opt-in `resupply-rack` policy below reaches **10%**
and **48.3%**. The timeline CSV shows every hour.

How the two rates are aggregated over the 25 hourly steps:

- **Waved layer-pick rate** is weighted by the layers actually waved in each hour, not
  averaged across hours. 1,455 layers wave over the simulated day and they are not spread
  evenly: 16 of the 25 hours have no waved demand at all. Averaging hourly percentages lets
  those empty hours vote, which understates the throughput effect by more than half
  (+5.0pp instead of +13.7pp under `default`).
- **Dead-slot rate** is a property of the floor, not of the demand, so it is the plain mean
  of the 25 hourly floor snapshots.

**Those two numbers come from different rules, and only one of them is what the operation
is paid on.** Freeing a face is not the same as picking a layer off it. The rule that fills
a freed face, `RACK_PROMOTE`, is disabled in `default` and gated a second time behind
`include_rack_aisle`, so the default policy can evict and not much else. Open both:

```bash
slotting simulate --data-dir data/synthetic --policy resupply-rack \
  --from 2026-03-02T06:00:00 --to 2026-03-03T06:00:00
```

| Policy | Dead slots (hourly mean) | Waved layer-pick (layer-weighted) | Layers shifted to layer pick* |
|---|---|---|---|
| `default` | 77% to 12% (65.3pp) | 26.3% to 40.0% (+13.7pp) | 367 |
| `resupply-rack` (opt-in) | 77% to 10% (67.3pp) | **26.3% to 48.3% (+22.1pp)** | **412** |

\* 24-hour-horizon layer demand moved from hand pick to layer pick, summed over the 25
hourly steps (the `layers_shifted_to_layer_pick` column of the timeline CSV).

Eviction earns nearly all of the dead-slot number and none of the throughput number. That
split is the actual thesis: the floor is nearly always worse than it looks, and freeing a
face only pays once something worth picking goes into it. Reporting the dead-slot swing
alone would have hidden a version of this engine that could not pick any faster than the
floor it replaced. That is exactly what an earlier revision shipped.

---

## How the rules engine is built

Each rule is a **pure function** over a `WorldState` snapshot returning zero or more
`Recommendation`s. No rule mutates state, no rule calls another, and every one is
individually disableable:

```bash
slotting simulate --data-dir data/synthetic --policy resupply-rack \
  --from 2026-03-02T06:00:00 --to 2026-03-03T06:00:00 --without-rules RACK_PROMOTE
```

That structure exists for one reason: it makes *attribution* possible. Re-run with one rule
removed and read what that rule was worth, instead of arguing about it. `--without-rules`
subtracts from whatever policy is in force and never adds a rule that policy had switched
off, so the delta measures the one change you asked for.

Dropped from `resupply-rack` one at a time, over the seed-7 day (same aggregation as above):

| Rule dropped | Dead-slot improvement | Waved layer-pick | Layers shifted |
|---|---|---|---|
| *(none)* | 67.3pp | 48.3% (+22.1pp) | 412 |
| `ZERO_DEMAND` | **8.0pp** | 48.3% (+22.1pp) | 412 |
| `RACK_PROMOTE` | 65.3pp | **40.0% (+13.7pp)** | **367** |
| `CROSS_AREA_HOLD` | 67.3pp | 48.3% (+22.1pp) | 412 |

The two load-bearing rules are orthogonal and neither substitutes for the other:
`ZERO_DEMAND` is essentially the whole dead-slot result and contributes nothing to
throughput, `RACK_PROMOTE` the reverse. `CROSS_AREA_HOLD` is a guard: it changes which
pallets are legal to touch, not how many get touched on this site.

`RuleConfig` carries ~30 policy fields. `--policy` selects a named combination;
`slotting simulate --help` lists them.

Representative rules:

| Rule | Question | In `default`? |
|---|---|---|
| `ZERO_DEMAND` | Nothing waved needs this SKU. Reclaim the face, or will it be needed shortly? | on |
| `CROSS_AREA_HOLD` | Another fulfilment area owns this pallet's earliest order. Do not touch it. | on |
| `EXCLUDED_ITEM` | This SKU cannot be layer-picked at all. Never spend a face on it. | on |
| `WAIT_SUPPLY_SHORTAGE` | Waved demand with no supply anywhere. Flag it; the engine cannot conjure stock. | on |
| `RACK_PROMOTE` | Backing stock exists for a waved SKU with no face. Promote it, and evict what? | **off**, use `--policy resupply-rack` |
| `LOW_LAYERS` | A face is nearly empty. Resupply it, or give the slot away? | **off**, use `--policy with-low-layers` |

The three rules `default` switches off (`LOW_LAYERS`, `BACKUP_PROMOTE`, `RACK_PROMOTE`)
each move pallets on a judgement the general engine should not make unasked. They are
opt-in, not unfinished.

The engine is **advisory by construction**. It returns recommendations; it does not move
pallets, and there is no code path that talks to a WMS.

---

## The simulator

[`src/layout_advisor/simulate/replay.py`](src/layout_advisor/simulate/replay.py) replays a
pick day on a discrete clock: demand waves in, pickers draw layers off faces, faces deplete,
rules fire between ticks, pallets move. It reports layer-pick percentage, hand-pick fallback,
waits, shortages, and dead-slot rate per hour.
[`scripts/generate_daily_wave_plans.py`](scripts/generate_daily_wave_plans.py) drives it to
produce a setup slotting plan and per-wave operator actions for a ship day:

```bash
python scripts/generate_daily_wave_plans.py --date 2026-03-03 --output-dir /tmp/plans
```

The design lesson worth stealing is a bug the project shipped and then found: for a long
period the simulated floor **never depleted**, because the fulfilment path did not decrement
what it picked. Every metric still looked plausible. The floor simply stayed full, so
layer-pick rates were measured against a room that could not run out. Figures produced
before the fix are not comparable to any after it.

The guard is now structural: `tests/test_inventory.py` asserts that the layers a pick step
reports equal the layers removed from the face pallets, so a floor that does not deplete
fails the suite rather than quietly reporting good numbers.

---

## The synthetic generator

[`tools/generate_synthetic_site.py`](tools/generate_synthetic_site.py) reproduces the
*structural* features the rules respond to, and nothing else:

- **ABC velocity skew** (Zipf): under uniform demand every face is worth the same and
  slotting is not a decision at all.
- **Wave structure with a mid-shift peak**: the binding constraint is concurrent
  throughput at the peak, not daily volume.
- **A visible/waved horizon split**: several rules exist only to tell "needed now" from
  "will be needed".
- **Lumpy line sizes** (geometric): the minimum-layers gate only matters when small lines
  exist.
- **A deliberately imperfect opening floor**: slow movers are seeded onto source faces so
  the reclaim rules have something to do. A perfectly slotted start would make the engine
  look inert.
- **Backing stock behind unfaced demand**: 40% of the resupply rack is reserved for SKUs that have demand but hold no source face. Drawing the whole rack from the same popularity curve that filled the faces backs SKUs that are already faced, and then the promote rules have nothing to promote: the dead-slot rate improves while the layer-pick rate does not move at all. That was the state this repository's first synthetic site was in.
- **Stock depth scaled to demand**: pallets are allocated against horizon demand rather than drawn from the popularity curve a second time, to a stated coverage target (`--seed 7` reaches **78.1%** of horizon layer demand; the generator prints it). Drawing them gave the head SKU one partially depleted pallet whether it wanted 1 layer that day or 151, so most demand was unservable for a reason no slotting decision could fix. A share is left uncovered on purpose, or `WAIT_SUPPLY_SHORTAGE` never fires, and the generator targets 30% of demanded SKUs starting the shift faced by nothing, or `RACK_PROMOTE` has nothing to promote.

Seeded throughout: the same `--seed` gives byte-identical output anywhere, which is what
makes every figure above checkable rather than a claim. CI fails if `--seed 7` stops
reproducing the committed site, so a stale figure cannot sit here quietly.

---

## Configuration and hard-coded values

Operational rates live in
[`src/layout_advisor/pick_throughput.py`](src/layout_advisor/pick_throughput.py) and are
**illustrative placeholders, not measurements**. Every one reads an environment override,
and anything derived is recomputed rather than hand-scaled:

```bash
LP_PICK_CASES_PER_HOUR=1500 slotting simulate --data-dir data/synthetic \
  --from 2026-03-02T06:00:00 --to 2026-03-03T06:00:00
```

[`src/layout_advisor/simulate/reference_config.py`](src/layout_advisor/simulate/reference_config.py)
supplies one internally consistent arrangement of the ~90 simulator switches
(`--profile reference` on the CLI), as a starting point to vary, not a tuned or optimal
configuration. The switch *names* and their interactions are the reusable part.

---

## What is deliberately not here

The general mechanism is published; anything tied to the data it was developed against is
not, and was removed rather than rewritten with invented numbers:

- the order feed, SKU master, and site profiles
- WMS export queries, which were written against a specific production schema
- labour, capacity and ROI models, and their commercial figures
- pick rates and cost bases from any real operation
- performance figures measured on real data, and the experiment results quoted against them
- real ship dates

What remains is the mechanism: the rules, the simulators, the slotting and wave-planning
logic, and a generator that gives them something to run on. Dates in the code are
synthetic, and the operating parameters are defaults to vary rather than measurements.

## License

MIT. See [LICENSE](LICENSE).
