"""Translate simulation steps into operator wave actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass, replace
from datetime import date, datetime

from slotting.models import RecommendationAction, RuleId
from slotting.simulation import SimulationResult


@dataclass(frozen=True)
class WaveAction:
    ship_date: date
    wave_send: datetime
    wave_due: datetime | None
    wave_index: int | None
    sequence: int
    action: str
    role: str
    item_id: str | None
    layers: int | None
    from_slot: str | None
    to_slot: str | None
    detail: str
    order_key: str | None = None
    trip: int | None = None  # clamp trip group (multi-bite trips, 48in stack cap)


def _role_for_action(action: str) -> str:
    if action in {"STAGE_SOURCE_FACE", "STAGE_RACK", "STAGE_SOURCE_BACKUP", "PREOPEN_DONOR", "OPEN_DONOR_COLOCATED", "ROLLING_PREOPEN_DONOR", "LULL_AHEAD_PREP", "CONVERT_PAD"}:
        return "setup_crew"
    if action in {"RACK_PROMOTE", "BACKUP_PROMOTE", "RELOCATE_TO_RACK", "STAGE_DONOR_COMPLETE", "EXIT_DONOR_TO_WRAP"}:
        return "assist"
    if action in {"OPEN_DONOR_BUILD", "CLOSE_IDLE_BUILD"}:
        return "assist"
    if action in {"WAIT_DONOR_FACE", "WAIT_SOURCE", "FREEZER_PULL"}:
        return "supervisor"
    if action in {"LAYER_PICK", "LAYER_PICK_TO_BUILD", "PICK_WAVED"}:
        return "picker"
    return "assist"


def _map_recommendation(rec) -> tuple[str, str, str | None, str | None]:
    rule = rec.rule
    if rec.action == RecommendationAction.FLAG_RACK_PROMOTE:
        return "RACK_PROMOTE", rec.item_id or "", rec.slot_id, None
    if rec.action == RecommendationAction.FLAG_BACKUP_PROMOTE:
        return "BACKUP_PROMOTE", rec.item_id or "", None, rec.slot_id
    if rec.action == RecommendationAction.FLAG_REMOVE and rule == RuleId.DEST_STAGE_COMPLETE:
        return "EXIT_DONOR_TO_WRAP", rec.item_id or "", rec.slot_id, None
    if rec.action == RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY:
        if rule == RuleId.DEST_STAGE_COMPLETE:
            return "STAGE_DONOR_COMPLETE", rec.item_id or "", rec.slot_id, None
        return "RELOCATE_TO_RACK", rec.item_id or "", rec.slot_id, None
    if rec.action == RecommendationAction.FLAG_ADD and rule == RuleId.DEST_ASSIGN_WAVED:
        return "OPEN_DONOR_BUILD", rec.item_id or "", None, rec.slot_id
    if rec.action == RecommendationAction.FLAG_ADD and rec.pallet_id:
        return "FREEZER_PULL", rec.item_id or "", None, rec.slot_id
    if rec.action == RecommendationAction.FLAG_REMOVE and rule == RuleId.DEST_CLOSE_IDLE:
        return "CLOSE_IDLE_BUILD", rec.item_id or "", rec.slot_id, None
    if rec.action == RecommendationAction.FLAG_REMOVE:
        return "CLOSE_SOURCE", rec.item_id or "", rec.slot_id, None
    if rec.action == RecommendationAction.FLAG_WAIT and rule == RuleId.DEST_WAIT_NO_FACE:
        return "WAIT_DONOR_FACE", rec.item_id or "", None, None
    if rec.action == RecommendationAction.FLAG_WAIT:
        return "WAIT_SOURCE", rec.item_id or "", None, None
    if rec.action == RecommendationAction.FLAG_STORAGE_RETURN:
        return "STORAGE_RETURN", rec.item_id or "", rec.slot_id, None
    return "ADVISORY", rec.item_id or "", rec.slot_id, None


def _layers_for_item(result: SimulationResult, item_id: str | None) -> int | None:
    if not item_id:
        return None
    cutoff = result.as_of
    from slotting.rules import WAVED_HORIZON

    end = cutoff + WAVED_HORIZON
    total = 0
    for order in result.baseline_state.orders:
        if order.item_id != item_id:
            continue
        if cutoff <= order.due_at <= end:
            total += order.qty_layers
    return total or None


def actions_from_simulation_step(
    result: SimulationResult,
    *,
    ship_date: date,
    wave_send: datetime,
    wave_due: datetime | None = None,
    wave_index: int | None = None,
    sequence_start: int = 1,
    include_aggregate_pick: bool = True,
) -> list[WaveAction]:
    """Convert applied simulation moves + outstanding waits into wave actions."""
    actions: list[WaveAction] = []
    seq = sequence_start

    seen: set[tuple[str, str, str | None]] = set()
    for applied in result.applied:
        key = (applied.action.value, applied.item_id or "", applied.slot_id)
        if key in seen:
            continue
        seen.add(key)
        act = applied.action.value
        if act == "FLAG_RACK_PROMOTE":
            code = "RACK_PROMOTE"
        elif act == "FLAG_BACKUP_PROMOTE":
            code = "BACKUP_PROMOTE"
        elif act == "FLAG_REMOVE" and applied.detail.startswith("Sent completed build"):
            code = "EXIT_DONOR_TO_WRAP"
        elif act == "FLAG_RELOCATE_TO_RESUPPLY" and applied.detail.startswith("Staged completed build"):
            code = "STAGE_DONOR_COMPLETE"
        elif act == "FLAG_RELOCATE_TO_RESUPPLY":
            code = "RELOCATE_TO_RACK"
        elif act == "FLAG_ADD" and applied.detail.startswith("Opened build"):
            code = "OPEN_DONOR_BUILD"
        elif act == "FLAG_ADD":
            code = "FREEZER_PULL"
        elif act == "FLAG_REMOVE" and "BUILD" in (applied.pallet_id or ""):
            code = "CLOSE_IDLE_BUILD"
        elif act == "FLAG_REMOVE":
            code = "CLOSE_SOURCE"
        elif act == "FLAG_STORAGE_RETURN":
            code = "STORAGE_RETURN"
        else:
            continue
        layers = _layers_for_item(result, applied.item_id)
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=wave_send,
                wave_due=wave_due,
                wave_index=wave_index,
                sequence=seq,
                action=code,
                role=_role_for_action(code),
                item_id=applied.item_id,
                layers=layers,
                from_slot=None,
                to_slot=applied.slot_id,
                detail=applied.detail,
            )
        )
        seq += 1

    for rec in result.recommendations:
        if rec.action != RecommendationAction.FLAG_WAIT:
            continue
        code, item_id, _from, _to = _map_recommendation(rec)
        key = (code, item_id, None)
        if key in seen:
            continue
        seen.add(key)
        layers = _layers_for_item(result, item_id)
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=wave_send,
                wave_due=wave_due,
                wave_index=wave_index,
                sequence=seq,
                action=code,
                role=_role_for_action(code),
                item_id=item_id,
                layers=layers,
                from_slot=None,
                to_slot=None,
                detail=rec.reason,
            )
        )
        seq += 1

    waved_ready = result.projected_wave_throughput.layer_pick_ready_layers
    if (
        include_aggregate_pick
        and waved_ready
        and result.projected_wave_throughput.total_layer_demand
    ):
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=wave_send,
                wave_due=wave_due,
                wave_index=wave_index,
                sequence=seq,
                action="PICK_WAVED",
                role="picker",
                item_id=None,
                layers=int(result.projected_wave_throughput.layer_pick_ready_layers),
                from_slot=None,
                to_slot=None,
                detail=(
                    f"Layer-pick {result.projected_wave_throughput.layer_pick_ready_layers:,} of "
                    f"{result.projected_wave_throughput.total_layer_demand:,} waved layers "
                    f"({100 * result.projected_wave_throughput.layer_pick_pct:.0f}% wave LP)."
                ),
            )
        )
    return actions


def resequence_picks_min_travel(events: list, station_by_slot: Mapping[str, int]) -> list:
    """Order picks to minimize the picker's empty legs (drop point → next face).

    Loaded legs (face → build) are fixed by slotting; only the walk back from a
    drop to the next face is schedulable. Greedy nearest-neighbor from each
    drop point, then 2-opt on empty-leg cost; deterministic tie-break by station
    then item.
    """
    if not station_by_slot or len(events) < 2:
        return list(events)

    def _st(slot: str | None) -> int | None:
        return station_by_slot.get(slot) if slot else None

    remaining = list(events)
    if any(_st(e.from_slot) is None for e in remaining):
        return remaining  # unknown geometry — keep sim emission order

    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]
        at = _st(last.to_slot)
        if at is None:
            at = _st(last.from_slot)
        nxt = min(
            remaining,
            key=lambda e: (abs(_st(e.from_slot) - at), _st(e.from_slot), e.item_id),
        )
        remaining.remove(nxt)
        ordered.append(nxt)
    return _two_opt_empty_legs(ordered, station_by_slot)


def _empty_leg_cost(events: list, station_by_slot: Mapping[str, int]) -> int:
    def _st(slot: str | None) -> int | None:
        return station_by_slot.get(slot) if slot else None

    total = 0
    for prev, nxt in zip(events, events[1:]):
        at = _st(prev.to_slot)
        if at is None:
            at = _st(prev.from_slot)
        face = _st(nxt.from_slot)
        if at is None or face is None:
            continue
        total += abs(face - at)
    return total


def _two_opt_empty_legs(events: list, station_by_slot: Mapping[str, int]) -> list:
    """Improve a pick sequence by reversing subsequences when empty-leg cost drops."""
    if len(events) < 4:
        return list(events)
    best = list(events)
    best_cost = _empty_leg_cost(best, station_by_slot)
    improved = True
    # Bound passes — hour blocks are small; avoid pathological loops.
    for _ in range(len(best)):
        if not improved:
            break
        improved = False
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                # Keep the first pick fixed as the wave's entry (matches greedy seed).
                if i == 0:
                    continue
                candidate = best[:i] + list(reversed(best[i : k + 1])) + best[k + 1 :]
                cost = _empty_leg_cost(candidate, station_by_slot)
                if cost < best_cost:
                    best, best_cost = candidate, cost
                    improved = True
                    break
            if improved:
                break
    return best


def _assist_anchor_station(action: WaveAction, station_by_slot: Mapping[str, int]) -> int | None:
    for slot in (action.to_slot, action.from_slot):
        if slot and slot in station_by_slot:
            return station_by_slot[slot]
    return None


def resequence_assist_min_travel(
    actions: list[WaveAction], station_by_slot: Mapping[str, int]
) -> list[WaveAction]:
    """Order assist-role wave actions to cut aisle zigzag without touching picks.

    Contiguous assist blocks are greedy nearest-neighbor by face/donor station.
    Picker / supervisor / setup rows stay put — LP sequence is never reordered.
    Actions with unknown geometry keep relative order at the end of their block.
    """
    if not station_by_slot or len(actions) < 2:
        return list(actions)

    out: list[WaveAction] = []
    i = 0
    while i < len(actions):
        if actions[i].role != "assist":
            out.append(actions[i])
            i += 1
            continue
        j = i
        while j < len(actions) and actions[j].role == "assist":
            j += 1
        block = actions[i:j]
        known = [a for a in block if _assist_anchor_station(a, station_by_slot) is not None]
        unknown = [a for a in block if _assist_anchor_station(a, station_by_slot) is None]
        if len(known) < 2:
            out.extend(block)
            i = j
            continue
        remaining = list(known)
        ordered = [remaining.pop(0)]
        while remaining:
            at = _assist_anchor_station(ordered[-1], station_by_slot)
            nxt = min(
                remaining,
                key=lambda a: (
                    abs(_assist_anchor_station(a, station_by_slot) - at),
                    _assist_anchor_station(a, station_by_slot),
                    a.action,
                    a.item_id or "",
                ),
            )
            remaining.remove(nxt)
            ordered.append(nxt)
        # Renumber sequences to match the new order within the block.
        seq0 = block[0].sequence
        renumbered = [
            replace(a, sequence=seq0 + k) for k, a in enumerate(ordered + unknown)
        ]
        out.extend(renumbered)
        i = j
    return out


def _event_with_layers(event, layers: int):
    """Copy a pick event dataclass with a new layer count."""
    if is_dataclass(event) and not isinstance(event, type):
        return replace(event, layers=layers)
    raise TypeError(f"cannot resize pick event of type {type(event)!r}")


def split_events_to_clamp_bites(
    events: list,
    *,
    layer_height_by_item: Mapping[str, float],
    max_trip_height_in: float = 48.0,
    default_layer_height_in: float = 9.0,
    unit_layers: bool = True,
) -> list:
    """Split aggregate picks into clamp-legal bites (<= *max_trip_height_in* each).

    Default *unit_layers=True* emits one layer per bite so the trip packer can
    interleave SKUs under the 48″ stack — max-size bites were starving combines
    (a 5-layer grab fills the clamp before a mate face can join). Operator CSVs
    coalesce contiguous same-face layers back into one row after packing.
    """
    bites: list = []
    for event in events:
        unit = layer_height_by_item.get(event.item_id, default_layer_height_in)
        if unit <= 0:
            unit = default_layer_height_in
        layers_left = max(1, int(event.layers))
        # Unit layer packs tighter; fall back to max legal bite only when
        # a single layer alone would already exceed the cap (pathological SKU).
        if unit_layers:
            max_layers = 1 if unit <= max_trip_height_in else max(1, int(max_trip_height_in // unit))
        else:
            max_layers = max(1, int(max_trip_height_in // unit))
        while layers_left > 0:
            take = min(layers_left, max_layers)
            bites.append(_event_with_layers(event, take))
            layers_left -= take
    return bites


def coalesce_trip_bites(trip: list) -> list:
    """Merge contiguous same-face / same-item bites within one clamp trip."""
    if not trip:
        return trip
    merged: list = []
    for event in trip:
        if (
            merged
            and merged[-1].item_id == event.item_id
            and merged[-1].from_slot == event.from_slot
            and merged[-1].to_slot == event.to_slot
            and getattr(merged[-1], "order_key", None) == getattr(event, "order_key", None)
        ):
            merged[-1] = _event_with_layers(
                merged[-1], int(merged[-1].layers) + int(event.layers)
            )
        else:
            merged.append(event)
    return merged


def plan_batched_trips(
    events: list,
    station_by_slot: Mapping[str, int],
    *,
    layer_height_by_item: Mapping[str, float],
    max_trip_height_in: float = 48.0,
    default_layer_height_in: float = 9.0,
) -> list[list]:
    """Group picks into clamp trips: multi-source bites, combined stack <=cap.

    Modelling assumption (configurable via *max_trip_height_in*): the clamp can
    bite face A then face B before one drop, and the combined stacked height is
    the cap. Combining is only practical onto the same build pad. This planner:

    1. Splits aggregates to unit layers so mates can share the clamp stack.
    2. Seeds trips on builds that still have packable mate layers.
    3. Packs **same build pad only** (no multi-drop trips).
    4. Coalesces contiguous same-face layers per trip for a compact pick list.
    """
    def _st(slot: str | None) -> int | None:
        return station_by_slot.get(slot) if slot else None

    def _h(e) -> float:
        h = layer_height_by_item.get(e.item_id, default_layer_height_in)
        return max(1, e.layers) * (h if h > 0 else default_layer_height_in)

    if any(_st(e.from_slot) is None for e in events):
        return [[e] for e in events]

    pending = split_events_to_clamp_bites(
        events,
        layer_height_by_item=layer_height_by_item,
        max_trip_height_in=max_trip_height_in,
        default_layer_height_in=default_layer_height_in,
        unit_layers=True,
    )
    trips: list[list] = []
    pos: int | None = None

    def _mate_count(seed) -> int:
        """Count other pending unit-layers sharing this seed's build pad."""
        to_slot = seed.to_slot
        if not to_slot:
            return 0
        return sum(1 for other in pending if other is not seed and other.to_slot == to_slot)

    def _seed() -> object:
        return min(
            pending,
            key=lambda e: (
                -_mate_count(e),
                0 if pos is None else abs((_st(e.from_slot) or 0) - pos),
                _st(e.from_slot),
                e.item_id,
            ),
        )

    def _pack_key(e, *, at: int, faces_open: set[str]):
        # A trip combines at most two faces (A then B), then drops — not a 5-face wander.
        # Open a second face once, then drain those two before starting another.
        if len(faces_open) < 2:
            # Prefer opening a nearby *new* face so A alone doesn't fill 48".
            novelty = 0 if e.from_slot not in faces_open else 1
        else:
            # Stick to the two faces already on the clamp route.
            novelty = 0 if e.from_slot in faces_open else 1
        return (
            novelty,
            abs((_st(e.from_slot) or 0) - at),
            _st(e.from_slot),
            e.item_id,
        )

    def _regroup_by_face(trip_events: list) -> list:
        """Visit each source once per trip (grab coalesced layers), NN order."""
        if not trip_events:
            return trip_events
        buckets: dict[str, list] = {}
        for e in trip_events:
            buckets.setdefault(e.from_slot, []).append(e)
        ordered_slots: list[str] = []
        at = _st(trip_events[0].from_slot) or 0
        remaining = set(buckets)
        while remaining:
            nxt = min(
                remaining,
                key=lambda s: (abs((_st(s) or 0) - at), _st(s) or 0, s),
            )
            ordered_slots.append(nxt)
            remaining.remove(nxt)
            at = _st(nxt) or at
        regrouped: list = []
        for slot in ordered_slots:
            regrouped.extend(buckets[slot])
        return regrouped

    while pending:
        first = _seed()
        pending.remove(first)
        trip = [first]
        height = _h(first)
        pos = _st(first.from_slot)
        anchor_to = first.to_slot
        faces_open = {first.from_slot}
        while True:
            fits = [
                e
                for e in pending
                if height + _h(e) <= max_trip_height_in
                and (
                    (anchor_to and e.to_slot == anchor_to)
                    or (not anchor_to and not e.to_slot)
                )
                and (len(faces_open) < 2 or e.from_slot in faces_open)
            ]
            if not fits:
                break
            nxt = min(
                fits,
                key=lambda e: _pack_key(e, at=pos or 0, faces_open=faces_open),
            )
            pending.remove(nxt)
            trip.append(nxt)
            height += _h(nxt)
            pos = _st(nxt.from_slot)
            faces_open.add(nxt.from_slot)
            if not anchor_to and nxt.to_slot:
                anchor_to = nxt.to_slot
        drop = _st(anchor_to) if anchor_to else None
        if drop is not None:
            pos = drop
        trips.append(coalesce_trip_bites(_regroup_by_face(trip)))
    return trips


def trip_walk_stations(trips: list[list], station_by_slot: Mapping[str, int]) -> int:
    """Total walk (station-units) over trips: bites in order, then drops nearest-first."""
    def _st(slot: str | None) -> int | None:
        return station_by_slot.get(slot) if slot else None

    total = 0
    pos: int | None = None
    for trip in trips:
        for e in trip:
            st = _st(e.from_slot)
            if st is None:
                continue
            if pos is not None:
                total += abs(st - pos)
            pos = st
        drops = []
        for e in trip:
            st = _st(e.to_slot)
            if st is not None and st not in drops:
                drops.append(st)
        while drops:
            nxt = min(drops, key=lambda s: abs(s - pos))
            drops.remove(nxt)
            total += abs(nxt - pos)
            pos = nxt
    return total


def layer_pick_actions_from_events(
    events: list,
    *,
    ship_date: date,
    wave_send: datetime,
    wave_due: datetime | None = None,
    wave_index: int | None = None,
    sequence_start: int = 1,
    trip_start: int = 1,
    station_by_slot: Mapping[str, int] | None = None,
    layer_height_by_item: Mapping[str, float] | None = None,
    max_trip_height_in: float = 48.0,
) -> list[WaveAction]:
    """Convert hourly layer-pick log rows into operator-facing picker actions.

    With *station_by_slot*, picks are resequenced to minimize empty-leg travel
    instead of raw sim emission order. With *layer_height_by_item* as well,
    picks are grouped into clamp trips (multi-source bites under the
    *max_trip_height_in* stack cap, an assumption) and the `trip` column is stamped.

    *trip_start* should continue across sim steps that share the same
    ``wave_send`` (order-ahead 10-min ticks) so trip numbers stay unique in the
    printed CSV.
    """
    trip_by_id: dict[int, int] = {}
    if station_by_slot and layer_height_by_item is not None:
        trips = plan_batched_trips(
            events,
            station_by_slot,
            layer_height_by_item=layer_height_by_item,
            max_trip_height_in=max_trip_height_in,
        )
        events = [e for trip in trips for e in trip]
        for t_idx, trip in enumerate(trips, start=max(1, trip_start)):
            for e in trip:
                trip_by_id[id(e)] = t_idx
    elif station_by_slot:
        events = resequence_picks_min_travel(events, station_by_slot)
    actions: list[WaveAction] = []
    seq = sequence_start
    for event in events:
        to_slot = event.to_slot
        code = "LAYER_PICK_TO_BUILD" if to_slot else "LAYER_PICK"
        trip_no = trip_by_id.get(id(event))
        detail_parts = [
            f"Pick {event.layers} layer(s) from face {event.from_slot}",
        ]
        if to_slot:
            detail_parts.append(f"onto build {event.to_pallet_id or 'BUILD'} @ {to_slot}")
        else:
            detail_parts.append("(no donor build face open: face depletion only)")
        if trip_no is not None:
            detail_parts.append(f"[clamp trip {trip_no} - combine bites before drop]")
        actions.append(
            WaveAction(
                ship_date=ship_date,
                wave_send=wave_send,
                wave_due=wave_due,
                wave_index=wave_index,
                sequence=seq,
                action=code,
                role="picker",
                item_id=event.item_id,
                layers=event.layers,
                from_slot=event.from_slot,
                to_slot=to_slot,
                detail=" ".join(detail_parts) + ".",
                order_key=getattr(event, "order_key", None),
                trip=trip_no,
            )
        )
        seq += 1
    return actions
