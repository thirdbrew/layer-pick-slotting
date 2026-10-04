"""Rules engine — each rule is a separate, testable function."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from collections.abc import Mapping

from slotting.models import (
    OrderArea,
    OrderStatus,
    Pallet,
    Recommendation,
    RecommendationAction,
    RuleId,
    SlotType,
    WorldState,
)

from slotting.inventory import RESUPPLY_LAYER_THRESHOLD, WAVED_HORIZON

VISIBLE_HORIZON = timedelta(hours=24)


@dataclass(frozen=True)
class RuleConfig:
    """Which rules to evaluate. Disabled rules emit no recommendations."""

    disabled: frozenset[RuleId] = field(default_factory=frozenset)
    allow_aisle_staging: bool = False
    include_rack_aisle: bool = False
    rack_only_reserve: bool = True
    preferred_source_face_by_item: Mapping[str, str] = field(default_factory=dict)
    rack_slot_by_item: Mapping[str, str] = field(default_factory=dict)
    setup_rank_by_item: Mapping[str, int] = field(default_factory=dict)
    source_face_station_by_id: Mapping[str, int] = field(default_factory=dict)
    source_face_flank_by_id: Mapping[str, str] = field(default_factory=dict)
    # Velocity twin faces (item_id → twin source slot); empty unless setup staged twins.
    twin_source_face_by_item: Mapping[str, str] = field(default_factory=dict)
    resupply_layer_threshold: int = RESUPPLY_LAYER_THRESHOLD
    low_layer_refresh_enabled: bool = True
    visible_prestage_enabled: bool = True
    aggressive_zero_demand_reclaim: bool = True
    zero_demand_reclaim_horizon_hours: int = 4
    # Library default keeps the documented general-engine behavior (rule 6 fires).
    # The reference config's "demand-required reclaim only" policy is an explicit
    # override in REFERENCE_REPLAY_CONFIG / ReplayConfig — do not change it here.
    proactive_zero_demand_remove: bool = True
    proactive_zero_demand_realignment_only: bool = False
    proactive_zero_demand_realignment_hours: tuple[int, ...] = ()
    slot_reclaim_rest_of_day: bool = True
    velocity_twin_face_reclaim: bool = False
    velocity_twin_reclaim_horizon_hours: int = 4
    # Rack promote may evict the coldest waved face when all faces are
    # waved-locked (on in the reference config; library default off).
    waved_displacement: bool = False
    # DISP-RACK-SWAP arm: displaced pallet may land in the rack slot the
    # promoting pallet vacates when no other landing exists (default off).
    waved_displacement_rack_swap: bool = False
    late_shift_hour: int = 18
    # 0 = ANY_DEPTH_RESUPPLY_THRESHOLD sentinel — resolved to a full-pallet
    # layer count (any practical face depth) after late_shift_hour
    late_shift_resupply_threshold: int = 0
    rack_promote_near_term_hour: int = 14
    rack_promote_near_term_hours: int = 4
    flank_local_rack_promote_batching: bool = True
    morning_rack_promote_batch_end_hour: int = 11
    # Squatter reclaim: when a waved SKU needs a face and its *preferred* face is
    # not reclaimable, evict the nearest zero-demand occupant instead of waiting.
    # Off by default — the library's reclaim is deliberately demand-required.
    zero_demand_any_face_reclaim: bool = False
    # Evictions per step. 4 is a conservative cap, 0 = uncapped. Small
    # differences between caps over a carried chain can be path noise (carry
    # inheritance) rather than merit, so compare them on fresh, no-carry days.
    zero_demand_any_face_max_per_step: int = 4
    # Don't spend an eviction seating a SKU that cannot be layer-picked anyway.
    # With min_layers_to_layer_pick=3, a SKU owing <3 layers is hand-picked
    # whether or not it holds a face, so the move yields zero layer picks —
    # wasted movement by definition, not a tuning threshold. 0 = no gate.
    zero_demand_any_face_min_layers: int = 0

    def is_enabled(self, rule: RuleId) -> bool:
        return rule not in self.disabled

    def with_slotting_maps(
        self,
        *,
        preferred_faces: Mapping[str, str],
        setup_ranks: Mapping[str, int],
        face_station_by_id: Mapping[str, int] | None = None,
        face_flank_by_id: Mapping[str, str] | None = None,
        rack_slot_by_item: Mapping[str, str] | None = None,
        twin_faces: Mapping[str, str] | None = None,
    ) -> RuleConfig:
        """Return a copy with mid-day face-swap maps from final-shift slotting."""
        from dataclasses import replace as _replace

        return _replace(
            self,
            preferred_source_face_by_item=dict(preferred_faces),
            setup_rank_by_item=dict(setup_ranks),
            source_face_station_by_id=dict(
                face_station_by_id or self.source_face_station_by_id
            ),
            source_face_flank_by_id=dict(face_flank_by_id or self.source_face_flank_by_id),
            rack_slot_by_item=dict(rack_slot_by_item or self.rack_slot_by_item),
            twin_source_face_by_item=dict(
                twin_faces if twin_faces is not None else self.twin_source_face_by_item
            ),
        )

    def effective_resupply_threshold(self, state: WorldState) -> int:
        """Resupply threshold for the current hour.

        After ``late_shift_hour``, returns ``late_shift_resupply_threshold``.
        The value ``0`` is a sentinel meaning *any practical face depth*
        (resolved via ``face_stock_needs_resupply`` to a full-pallet layer
        count) — not "only when empty". That pulls rack stock forward more
        aggressively than the daytime threshold without thrashing a freshly
        promoted full face on the second apply pass.
        """
        if state.as_of.hour >= self.late_shift_hour:
            return self.late_shift_resupply_threshold
        return self.resupply_layer_threshold

    def with_policy_flags(self, **overrides) -> RuleConfig:
        """Return a copy with the named policy fields overridden.

        ``None`` means "leave as is", so callers can pass optional knobs through
        without branching. Unknown names raise — a typo used to be silently
        dropped by the old hand-written copy constructor.
        """
        from dataclasses import fields, replace as _replace

        known = {f.name for f in fields(self)}
        alias = {"low_layer_refresh": "low_layer_refresh_enabled",
                 "visible_prestage": "visible_prestage_enabled"}
        changes = {}
        for key, value in overrides.items():
            name = alias.get(key, key)
            if name not in known:
                raise TypeError(f"with_policy_flags: unknown field {key!r}")
            if value is not None:
                changes[name] = value
        return _replace(self, **changes)


DEFAULT_RULE_CONFIG = RuleConfig(disabled=frozenset({RuleId.LOW_LAYERS, RuleId.BACKUP_PROMOTE, RuleId.RACK_PROMOTE}))
WITHOUT_LOW_LAYERS = DEFAULT_RULE_CONFIG
WITH_LOW_LAYERS = RuleConfig(disabled=frozenset({RuleId.BACKUP_PROMOTE}))
BACKUP_ROW_RULE_CONFIG = RuleConfig(disabled=frozenset({RuleId.LOW_LAYERS}))
BACKUP_ROW_AISLE_RULE_CONFIG = RuleConfig(
    disabled=frozenset({RuleId.LOW_LAYERS}),
    allow_aisle_staging=True,
)
RESUPPLY_RACK_RULE_CONFIG = RuleConfig(
    disabled=frozenset({RuleId.LOW_LAYERS, RuleId.DEST_CLOSE_IDLE, RuleId.BACKUP_PROMOTE}),
    include_rack_aisle=True,
    rack_only_reserve=True,
)
RESUPPLY_RACK_AISLE_RULE_CONFIG = RuleConfig(
    disabled=frozenset({RuleId.LOW_LAYERS, RuleId.DEST_CLOSE_IDLE, RuleId.BACKUP_PROMOTE}),
    include_rack_aisle=True,
    allow_aisle_staging=True,
    rack_only_reserve=True,
)


LIBRARY_POLICIES: dict[str, RuleConfig] = {
    "default": DEFAULT_RULE_CONFIG,
    "with-low-layers": WITH_LOW_LAYERS,
    "backup-row": BACKUP_ROW_RULE_CONFIG,
    "backup-row-aisle": BACKUP_ROW_AISLE_RULE_CONFIG,
    "resupply-rack": RESUPPLY_RACK_RULE_CONFIG,
    "resupply-rack-aisle": RESUPPLY_RACK_AISLE_RULE_CONFIG,
}
"""Named library policies selectable with ``--policy``.

``DEFAULT_RULE_CONFIG`` disables RACK_PROMOTE *and* leaves ``include_rack_aisle``
False, which gates the promote path twice. Before this map existed the only way
to reach that path was ``--profile``, which loads a whole replay policy rather
than a library rule set — so the library engine alone could evict and never
promote. Selecting
``resupply-rack`` turns both gates off together, which is the only combination
that actually reaches :data:`RuleId.RACK_PROMOTE`.
"""


def parse_rule_ids(values: list[str] | None) -> frozenset[RuleId]:
    """Parse rule names into ids. Raises ValueError naming the valid set."""
    if not values:
        return frozenset()
    parsed: set[RuleId] = set()
    for raw in values:
        try:
            parsed.add(RuleId(raw.strip().upper()))
        except ValueError as exc:
            valid = ", ".join(r.value for r in RuleId)
            raise ValueError(f"Unknown rule {raw!r}. Valid values: {valid}") from exc
    return frozenset(parsed)


def parse_disabled_rules(values: list[str] | None) -> RuleConfig:
    """Parse CLI ``--without-rules`` values into a :class:`RuleConfig`.

    ``--without-rules`` is a SUBTRACTION from whatever policy is in force, so the
    named rules union onto the base policy's own disabled set rather than
    replacing it. Replacing it inverts the flag: ``DEFAULT_RULE_CONFIG`` already
    disables three rules, so ``--without-rules ZERO_DEMAND`` against a fresh
    config would *enable* LOW_LAYERS, BACKUP_PROMOTE and RACK_PROMOTE while
    claiming to have removed one rule — and the attribution delta would then
    measure four changes at once.
    """
    return with_rules_disabled(DEFAULT_RULE_CONFIG, values)


def with_rules_disabled(base: RuleConfig, values: list[str] | None) -> RuleConfig:
    """Return *base* with the named rules additionally disabled."""
    from dataclasses import replace as _replace

    extra = parse_rule_ids(values)
    if not extra:
        return base
    return _replace(base, disabled=base.disabled | extra)


def _this_line_orders_in_window(state: WorldState, horizon: timedelta) -> list:
    """Order lines for this line due within *horizon* of *state.as_of*."""
    cutoff = state.as_of + horizon
    return [
        o
        for o in state.orders
        if o.area == OrderArea.THIS_LINE and state.as_of <= o.due_at <= cutoff
    ]


def _items_with_demand(state: WorldState, horizon: timedelta) -> set[str]:
    return {o.item_id for o in _this_line_orders_in_window(state, horizon)}


def _items_with_waved_demand(state: WorldState) -> set[str]:
    cutoff = state.as_of + WAVED_HORIZON
    return {
        o.item_id
        for o in state.orders
        if o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    }


def _items_with_visible_only_demand(state: WorldState) -> set[str]:
    """SKUs visible in 24h but not yet in the 4h waved pick window."""
    visible_cutoff = state.as_of + VISIBLE_HORIZON
    waved_cutoff = state.as_of + WAVED_HORIZON
    items: set[str] = set()
    for order in state.orders:
        if order.area != OrderArea.THIS_LINE:
            continue
        if not (state.as_of <= order.due_at <= visible_cutoff):
            continue
        if order.status == OrderStatus.VISIBLE or order.status == OrderStatus.WAVED and order.due_at > waved_cutoff:
            items.add(order.item_id)
    return items


def _earliest_this_line_due(state: WorldState, item_id: str):
    """Earliest due_at for this_line orders for *item_id*, or None."""
    lines = [o for o in state.orders if o.item_id == item_id and o.area == OrderArea.THIS_LINE]
    if not lines:
        return None
    return min(o.due_at for o in lines)


def _is_held_by_other_area(state: WorldState, item_id: str) -> bool:
    """
    True when another area has a claim on *item_id* that is due before this
    line's earliest order for the same item.
    """
    earliest_this = _earliest_this_line_due(state, item_id)
    if earliest_this is None:
        return False
    other_lines = [
        o for o in state.orders if o.item_id == item_id and o.area == OrderArea.OTHER_AREA
    ]
    return any(o.due_at < earliest_this for o in other_lines)


def _held_pallet_ids(state: WorldState) -> set[str]:
    """Pallets whose items are subject to a cross-area hold."""
    held: set[str] = set()
    item_ids = {p.item_id for p in state.pallets}
    for item_id in item_ids:
        if _is_held_by_other_area(state, item_id):
            for pallet in state.pallets:
                if pallet.item_id == item_id:
                    held.add(pallet.pallet_id)
    return held


def rule_low_layers(state: WorldState) -> list[Recommendation]:
    """
    Legacy: FLAG_REMOVE when a source-slot pallet has <= 2 layers remaining.

    Disabled in all replay configs — the floor finishes thin partials on the face
    instead of freeing the slot early. Kept for CLI ``--compare-low-layers`` only.
    """
    recs: list[Recommendation] = []
    for pallet in state.pallets_in_source_slots():
        if pallet.layers_remaining <= 2:
            recs.append(
                Recommendation(
                    action=RecommendationAction.FLAG_REMOVE,
                    pallet_id=pallet.pallet_id,
                    item_id=pallet.item_id,
                    reason=(
                        f"Pallet {pallet.pallet_id} ({pallet.item_id}) has "
                        f"{pallet.layers_remaining} layer(s) remaining: at or below "
                        f"the 2-layer minimum for layer picking."
                    ),
                    rule=RuleId.LOW_LAYERS,
                )
            )
    return recs


def rule_zero_demand(state: WorldState, config: RuleConfig) -> list[Recommendation]:
    """
    FLAG_REMOVE when a source-slot pallet's item has no this-line demand
    within *zero_demand_reclaim_horizon_hours* (proactive path only).
    """
    horizon = timedelta(hours=config.zero_demand_reclaim_horizon_hours)
    demand_items = _items_with_demand(state, horizon)
    recs: list[Recommendation] = []
    for pallet in state.pallets_in_source_slots():
        if pallet.item_id not in demand_items:
            recs.append(
                Recommendation(
                    action=RecommendationAction.FLAG_REMOVE,
                    pallet_id=pallet.pallet_id,
                    item_id=pallet.item_id,
                    reason=(
                        f"Pallet {pallet.pallet_id} ({pallet.item_id}) has zero "
                        f"visible this-line demand in the next "
                        f"{config.zero_demand_reclaim_horizon_hours} hours."
                    ),
                    rule=RuleId.ZERO_DEMAND,
                )
            )
    return recs


def rule_excluded_item(state: WorldState) -> list[Recommendation]:
    """
    FLAG_REMOVE when a source-slot pallet carries an item on the exclusion list.
    """
    excluded = state.excluded_item_ids
    reasons = {e.item_id: e.reason for e in state.exclusions}
    recs: list[Recommendation] = []
    for pallet in state.pallets_in_source_slots():
        if pallet.item_id in excluded:
            recs.append(
                Recommendation(
                    action=RecommendationAction.FLAG_REMOVE,
                    pallet_id=pallet.pallet_id,
                    item_id=pallet.item_id,
                    reason=(
                        f"Pallet {pallet.pallet_id} ({pallet.item_id}) is excluded "
                        f"from this line: {reasons[pallet.item_id]}."
                    ),
                    rule=RuleId.EXCLUDED_ITEM,
                )
            )
    return recs


def rule_cross_area_hold(state: WorldState) -> list[Recommendation]:
    """
    DO_NOT_TOUCH when another area's order for the same item is due before
    this line's earliest order — the pallet must be held for that claim.
    """
    recs: list[Recommendation] = []
    seen_items: set[str] = set()
    for pallet in state.pallets:
        if pallet.item_id in seen_items:
            continue
        if not _is_held_by_other_area(state, pallet.item_id):
            continue
        seen_items.add(pallet.item_id)
        earliest_this = _earliest_this_line_due(state, pallet.item_id)
        other_due = min(
            o.due_at
            for o in state.orders
            if o.item_id == pallet.item_id and o.area == OrderArea.OTHER_AREA
        )
        recs.append(
            Recommendation(
                action=RecommendationAction.DO_NOT_TOUCH,
                pallet_id=pallet.pallet_id,
                item_id=pallet.item_id,
                reason=(
                    f"Hold pallet for {pallet.item_id}: other-area order due "
                    f"{other_due.isoformat(sep=' ', timespec='minutes')} before "
                    f"this line's earliest order "
                    f"({earliest_this.isoformat(sep=' ', timespec='minutes')})."
                ),
                rule=RuleId.CROSS_AREA_HOLD,
            )
        )
    return recs


def _eligible_unslotted_pallets(state: WorldState, item_id: str, held_ids: set[str]) -> list[Pallet]:
    """Unslotted pallets matching *item_id* that may be added to a source slot."""
    candidates: list[Pallet] = []
    for pallet in state.unslotted_pallets():
        if pallet.item_id != item_id:
            continue
        if pallet.pallet_id in held_ids:
            continue
        if pallet.layers_remaining <= 0:
            continue
        if pallet.item_id in state.excluded_item_ids:
            continue
        candidates.append(pallet)
    return candidates


def rule_backup_swap(state: WorldState, held_ids: set[str]) -> list[Recommendation]:
    """Delegate to backup-row physics module."""
    from slotting.backup_row import rule_backup_swap as _rule_backup_swap

    return _rule_backup_swap(state, held_ids, allow_aisle_staging=False)


def rule_backup_swap_with_config(state: WorldState, held_ids: set[str], config: RuleConfig) -> list[Recommendation]:
    from slotting.backup_row import rule_backup_swap as _rule_backup_swap

    return _rule_backup_swap(
        state,
        held_ids,
        allow_aisle_staging=config.allow_aisle_staging,
        include_rack_aisle=config.include_rack_aisle,
    )


def rule_rack_promote_with_config(state: WorldState, held_ids: set[str], config: RuleConfig) -> list[Recommendation]:
    from slotting.resupply_rack import rule_rack_promote

    threshold = config.effective_resupply_threshold(state)
    return rule_rack_promote(
        state,
        held_ids,
        preferred_face_by_item=config.preferred_source_face_by_item,
        face_station_by_id=config.source_face_station_by_id,
        face_flank_by_id=config.source_face_flank_by_id,
        rack_slot_by_item=config.rack_slot_by_item,
        resupply_layer_threshold=threshold,
        waved_displacement=config.waved_displacement,
        waved_displacement_rack_swap=config.waved_displacement_rack_swap,
        late_shift_hour=config.late_shift_hour,
        near_term_hour=config.rack_promote_near_term_hour,
        near_term_hours=config.rack_promote_near_term_hours,
        flank_local_batching=config.flank_local_rack_promote_batching,
        morning_batch_end_hour=config.morning_rack_promote_batch_end_hour,
    )


def rule_face_refresh(
    state: WorldState,
    held_ids: set[str],
    config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[Recommendation]:
    """
    Swap a low-layer face pallet for fresher unslotted reserve when rack promote is unavailable.
    """
    from slotting.inventory import item_needs_face_resupply, lowest_face_pallet
    from slotting.resupply_rack import item_rack_ready

    threshold = config.effective_resupply_threshold(state)
    recs: list[Recommendation] = []
    for item_id in sorted(_items_with_waved_demand(state)):
        if not item_needs_face_resupply(state, item_id, threshold=threshold):
            continue
        face_pallet, face_slot = lowest_face_pallet(state, item_id)
        if face_pallet is None or face_slot is None:
            continue
        if face_pallet.pallet_id in held_ids:
            continue
        if config.include_rack_aisle and item_rack_ready(state, item_id):
            continue
        candidates = _eligible_unslotted_pallets(state, item_id, held_ids)
        from slotting.inventory_policy import is_pending_completion, select_best_supply_pallet

        pending = [p for p in candidates if is_pending_completion(p)]
        if pending:
            fresh = select_best_supply_pallet(pending)
        else:
            better = [
                p
                for p in candidates
                if p.layers_remaining > face_pallet.layers_remaining
            ]
            if not better:
                continue
            fresh = max(better, key=lambda p: p.layers_remaining)
        if fresh is None:
            continue
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_FACE_REFRESH,
                pallet_id=fresh.pallet_id,
                item_id=item_id,
                slot_id=face_slot,
                reason=(
                    f"Face {face_slot} low ({face_pallet.layers_remaining} layers); refresh "
                    f"with reserve pallet {fresh.pallet_id} ({fresh.layers_remaining} layers)."
                ),
                rule=RuleId.FACE_REFRESH,
            )
        )
    return recs


def rule_aggressive_zero_demand_reclaim(state: WorldState, config: RuleConfig) -> list[Recommendation]:
    """
    Evict a source face pallet only when that **specific slot** is required.

    Removes run when a waved SKU needs an ADD onto its preferred face and a stale
    occupant (no rest-of-ship-day demand) blocks that face. Does **not** bulk-evict
    to satisfy a generic empty-slot deficit.
    """
    if not config.aggressive_zero_demand_reclaim:
        return []
    preferred = config.preferred_source_face_by_item
    if not preferred:
        return []

    from slotting.inventory import item_needs_face_add
    from slotting.slotting_policy import rest_of_ship_day_layers_by_item

    waved = _items_with_waved_demand(state)
    if not waved:
        return []

    threshold = config.effective_resupply_threshold(state)
    slot_by_pallet = {
        s.occupied_by_pallet_id: s.slot_id
        for s in state.source_slots
        if s.occupied_by_pallet_id
    }
    recs: list[Recommendation] = []
    claimed_pallets: set[str] = set()

    # Hoisted out of the per-face loop: one O(orders) pass covers both the
    # staleness test (face occupants are always demand candidates) and the
    # rest-of-day layer counts used in the reason strings.
    layers_by_item = rest_of_ship_day_layers_by_item(state)
    if config.slot_reclaim_rest_of_day:
        horizon_demand_items: set[str] | None = None
    else:
        horizon = timedelta(hours=config.zero_demand_reclaim_horizon_hours)
        horizon_demand_items = _items_with_demand(state, horizon)

    def _item_stale(item_id: str) -> bool:
        if horizon_demand_items is None:
            return layers_by_item.get(item_id, 0) <= 0
        return item_id not in horizon_demand_items

    def _append_remove(pallet: Pallet, slot_id: str, *, reason: str) -> None:
        if pallet.pallet_id in claimed_pallets:
            return
        claimed_pallets.add(pallet.pallet_id)
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_REMOVE,
                pallet_id=pallet.pallet_id,
                item_id=pallet.item_id,
                reason=reason,
                rule=RuleId.ZERO_DEMAND_RECLAIM,
            )
        )

    blocked_faces: dict[str, list[str]] = {}
    for item_id in waved:
        if not item_needs_face_add(state, item_id, threshold=threshold):
            continue
        face_id = preferred.get(item_id)
        if face_id:
            blocked_faces.setdefault(face_id, []).append(item_id)

    for face_id, waiting in sorted(blocked_faces.items()):
        occupant = next(
            (
                p
                for p in state.pallets_in_source_slots()
                if slot_by_pallet.get(p.pallet_id) == face_id
            ),
            None,
        )
        if occupant is None:
            continue
        if not _item_stale(occupant.item_id):
            continue
        layers = layers_by_item.get(occupant.item_id, 0)
        _append_remove(
            occupant,
            face_id,
            reason=(
                f"Pallet {occupant.pallet_id} ({occupant.item_id}, {layers} layers rest-of-day) "
                f"blocks preferred face {face_id} for waved {', '.join(sorted(waiting))} - "
                f"reclaim slot (low-value, no remaining ship-day demand)."
            ),
        )

    if not config.zero_demand_any_face_reclaim:
        return recs

    # Squatter reclaim. The loop above only frees a waved SKU's OWN preferred
    # face, so a floor can sit with most faces at zero demand while a large
    # share of waved layers has nowhere to land. When faces are in deficit,
    # evict the nearest stale occupant to
    # the SKU's home station instead of waiting for its one face to come free.
    from slotting.throughput import _waved_demand_by_item

    waved_layers = _waved_demand_by_item(state)
    # Urgency (waved layers) orders the queue; VALUE (layers still owed today)
    # decides whether the seat is worth an assist move at all — the seat lasts
    # the rest of the day, so rest-of-day demand is what it earns.
    min_layers = config.zero_demand_any_face_min_layers
    unmet = sorted(
        (
            (waved_layers.get(item_id, 0), item_id)
            for item_id in waved
            if item_needs_face_add(state, item_id, threshold=threshold)
            and layers_by_item.get(item_id, 0) >= min_layers
        ),
        reverse=True,
    )
    free_faces = sum(1 for s in state.source_slots if not s.occupied_by_pallet_id)
    deficit = len(unmet) - free_faces - len(recs)
    if deficit <= 0:
        return recs

    stations = config.source_face_station_by_id
    stale = [
        (slot_by_pallet[p.pallet_id], p)
        for p in state.pallets_in_source_slots()
        if p.pallet_id in slot_by_pallet
        and p.pallet_id not in claimed_pallets
        and _item_stale(p.item_id)
    ]
    cap = config.zero_demand_any_face_max_per_step
    budget = deficit if cap <= 0 else min(deficit, cap)
    for _layers, item_id in unmet:
        if budget <= 0 or not stale:
            break
        home = stations.get(preferred.get(item_id) or "")
        face_id, occupant = min(
            stale,
            key=lambda fp: (
                abs(stations.get(fp[0], 0) - home) if home is not None else 0,
                fp[1].layers_remaining,
                fp[0],
            ),
        )
        stale.remove((face_id, occupant))
        budget -= 1
        _append_remove(
            occupant,
            face_id,
            reason=(
                f"Pallet {occupant.pallet_id} ({occupant.item_id}, no remaining ship-day "
                f"demand) squats face {face_id} while waved {item_id} has no face and "
                f"{free_faces} faces are free: reclaim nearest stale slot."
            ),
        )

    return recs


def _layers_in_horizon_by_item(state: WorldState, horizon: timedelta) -> dict[str, int]:
    """Layer demand by SKU due within *horizon* of *state.as_of*."""
    layers: dict[str, int] = {}
    for order in _this_line_orders_in_window(state, horizon):
        layers[order.item_id] = layers.get(order.item_id, 0) + order.qty_layers
    return layers


def _primary_face_pickable_layers_for_twin(
    state: WorldState,
    *,
    twin_item: str,
    twin_slot: str,
    primary_face_id: str | None,
    slot_by_pallet: dict[str, str],
) -> int:
    """LP-pickable layers on the primary face only (exclude the twin slot occupant)."""
    from slotting.pallet_geometry import lp_pickable_layers_on_pallet

    def _pickable(p: Pallet) -> int:
        return lp_pickable_layers_on_pallet(p)

    if primary_face_id:
        return sum(
            _pickable(p)
            for p in state.pallets_in_source_slots()
            if p.item_id == twin_item and slot_by_pallet.get(p.pallet_id) == primary_face_id
        )
    return sum(
        _pickable(p)
        for p in state.pallets_in_source_slots()
        if p.item_id == twin_item and slot_by_pallet.get(p.pallet_id) != twin_slot
    )


def _reserve_pickable_capacity(item_id: str, *, reserve_layers: int = 8) -> int:
    """Pickable layers on a standard reserve-depth face pallet for *item_id*."""
    from datetime import datetime, timezone

    from slotting.pallet_geometry import lp_pickable_layers_on_pallet

    return lp_pickable_layers_on_pallet(
        Pallet(
            pallet_id="__cap__",
            item_id=item_id,
            layers_remaining=reserve_layers,
            arrived_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
    )


def rule_velocity_twin_face_reclaim(state: WorldState, config: RuleConfig) -> list[Recommendation]:
    """
    Free velocity-twin source faces early when the spot will be needed soon.

    Twins keep hot SKUs pick-ready, but they displace lower-ranked SKUs to rack.
    This rule removes the *twin* pallet (never the primary) when either:

    1. **Preferred-face collision** — a horizon-demand SKU needs ADD onto that
       exact twin slot as its preferred face.
    2. **Empty-face deficit** — SKUs with horizon demand need a face and empty
       source slots are short, while the twin SKU still has a primary face.
    3. **Surplus twin** — reclaim-horizon (or rest-of-day) demand for the twin
       SKU fits on the primary face alone while other SKUs need faces soon.

    Leaves the primary face intact so the hot SKU stays LP-ready on one face.
    """
    if not config.velocity_twin_face_reclaim:
        return []
    twins = config.twin_source_face_by_item
    if not twins:
        return []

    from slotting.inventory import (
        active_face_pallet_count,
        item_needs_face_add,
    )
    from slotting.pallet_geometry import lp_pickable_layers_on_pallet
    from slotting.slotting_policy import rest_of_ship_day_layers_by_item

    horizon = timedelta(hours=max(1, config.velocity_twin_reclaim_horizon_hours))
    horizon_items = _items_with_demand(state, horizon)
    horizon_layers_by_item = _layers_in_horizon_by_item(state, horizon)
    preferred = config.preferred_source_face_by_item
    threshold = config.effective_resupply_threshold(state)
    layers_by_item = rest_of_ship_day_layers_by_item(state)

    slot_by_pallet = {
        s.occupied_by_pallet_id: s.slot_id
        for s in state.source_slots
        if s.occupied_by_pallet_id
    }
    empty_faces = sum(1 for s in state.source_slots if not s.occupied_by_pallet_id)
    horizon_face_needers = [
        item_id
        for item_id in horizon_items
        if item_id not in twins and item_needs_face_add(state, item_id, threshold=threshold)
    ]
    slots_needed = max(0, len(horizon_face_needers) - empty_faces)
    deficit = slots_needed > 0

    # Reverse preferred map: face → waiters with horizon demand needing ADD.
    waiters_by_face: dict[str, list[str]] = {}
    for item_id in horizon_face_needers:
        face_id = preferred.get(item_id)
        if face_id:
            waiters_by_face.setdefault(face_id, []).append(item_id)

    recs: list[Recommendation] = []
    claimed: set[str] = set()

    # Score twins for deficit reclaim: prefer freeing twins whose remaining
    # demand is smallest (least hurt by dropping to one face).
    deficit_candidates: list[tuple[int, int, int, str, Pallet, str]] = []

    for twin_item, twin_slot in sorted(twins.items()):
        if active_face_pallet_count(state, twin_item) < 2:
            continue
        occupant = next(
            (
                p
                for p in state.pallets_in_source_slots()
                if p.item_id == twin_item and slot_by_pallet.get(p.pallet_id) == twin_slot
            ),
            None,
        )
        if occupant is None:
            continue
        if occupant.pallet_id in claimed:
            continue

        waiters = waiters_by_face.get(twin_slot, [])
        rest_layers = layers_by_item.get(twin_item, 0)
        horizon_layers = horizon_layers_by_item.get(twin_item, 0)
        primary_face_id = preferred.get(twin_item)
        primary_pickable = _primary_face_pickable_layers_for_twin(
            state,
            twin_item=twin_item,
            twin_slot=twin_slot,
            primary_face_id=primary_face_id,
            slot_by_pallet=slot_by_pallet,
        )
        pickable_cap = max(primary_pickable, _reserve_pickable_capacity(twin_item))
        surplus_horizon = horizon_layers <= pickable_cap

        reason: str | None = None
        if waiters:
            reason = (
                f"Twin face {twin_slot} for {twin_item} blocks preferred face for "
                f"horizon demand {', '.join(sorted(waiters))} - reclaim twin "
                f"(primary face kept; horizon {config.velocity_twin_reclaim_horizon_hours}h)."
            )
        elif deficit:
            # Prefer freeing twins whose near-term demand already fits the primary
            # face; keep twins that still need two faces for the reclaim window.
            deficit_candidates.append(
                (
                    0 if surplus_horizon else 1,
                    rest_layers,
                    -lp_pickable_layers_on_pallet(occupant),
                    twin_item,
                    occupant,
                    twin_slot,
                )
            )
            continue

        if reason is None:
            continue
        claimed.add(occupant.pallet_id)
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_REMOVE,
                pallet_id=occupant.pallet_id,
                item_id=occupant.item_id,
                reason=reason,
                rule=RuleId.VELOCITY_TWIN_RECLAIM,
            )
        )

    if deficit and deficit_candidates:
        freed = 0
        needers = ", ".join(sorted(horizon_face_needers)[:4])
        more = "" if len(horizon_face_needers) <= 4 else f" (+{len(horizon_face_needers) - 4} more)"
        for rank, rest_layers, _neg_lyr, twin_item, occupant, twin_slot in sorted(
            deficit_candidates
        ):
            if occupant.pallet_id in claimed:
                continue
            if freed >= slots_needed:
                break
            claimed.add(occupant.pallet_id)
            freed += 1
            recs.append(
                Recommendation(
                    action=RecommendationAction.FLAG_REMOVE,
                    pallet_id=occupant.pallet_id,
                    item_id=occupant.item_id,
                    reason=(
                        f"Twin face {twin_slot} for {twin_item} (rest-of-day {rest_layers} ly) "
                        f"reclaimed: {len(horizon_face_needers)} horizon face needers, "
                        f"{empty_faces} empty faces; {needers}{more}."
                    ),
                    rule=RuleId.VELOCITY_TWIN_RECLAIM,
                )
            )

    return recs


def rule_add_visible_prestage(
    state: WorldState,
    held_ids: set[str],
    config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[Recommendation]:
    """
    Pre-stage visible (not yet waved) demand onto preferred empty faces.
    """
    if not config.visible_prestage_enabled:
        return []
    from slotting.inventory import item_needs_face_add

    preferred = config.preferred_source_face_by_item
    ranks = config.setup_rank_by_item
    threshold = config.effective_resupply_threshold(state)
    visible_items = _items_with_visible_only_demand(state)
    if not visible_items:
        return []

    empty_slots = sorted(
        s.slot_id
        for s in state.slots
        if s.slot_type == SlotType.SOURCE and not s.occupied_by_pallet_id
    )
    empty_set = set(empty_slots)
    claimed_items: set[str] = set()
    claimed_slots: set[str] = set()
    recs: list[Recommendation] = []

    def _best_pallet(item_id: str) -> Pallet | None:
        from slotting.inventory_policy import select_best_supply_pallet

        return select_best_supply_pallet(_eligible_unslotted_pallets(state, item_id, held_ids))

    sorted_items = sorted(
        visible_items,
        key=lambda item_id: (ranks.get(item_id, 10**9), item_id),
    )

    for slot_id in empty_slots:
        best_item: str | None = None
        best_rank = 10**9
        for item_id in sorted_items:
            if item_id in claimed_items:
                continue
            if not item_needs_face_add(state, item_id, threshold=threshold):
                continue
            if preferred.get(item_id) != slot_id:
                continue
            if _best_pallet(item_id) is None:
                continue
            rank = ranks.get(item_id, 10**9)
            if rank < best_rank:
                best_rank = rank
                best_item = item_id
        if best_item is None:
            continue
        pallet = _best_pallet(best_item)
        assert pallet is not None
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_ADD,
                pallet_id=pallet.pallet_id,
                item_id=best_item,
                slot_id=slot_id,
                reason=(
                    f"Visible demand for {best_item}; pre-stage to preferred {slot_id} "
                    f"with pallet {pallet.pallet_id} ({pallet.layers_remaining} layers)."
                ),
                rule=RuleId.ADD_VISIBLE_PRESTAGE,
            )
        )
        claimed_items.add(best_item)
        claimed_slots.add(slot_id)

    for item_id in sorted_items:
        if item_id in claimed_items or not item_needs_face_add(state, item_id, threshold=threshold):
            continue
        pallet = _best_pallet(item_id)
        if pallet is None:
            continue
        slot_id: str | None = None
        pref = preferred.get(item_id)
        if pref and pref in empty_set and pref not in claimed_slots:
            slot_id = pref
        detail = (
            f"Visible demand for {item_id}; pre-stage unslotted pallet "
            f"{pallet.pallet_id} ({pallet.layers_remaining} layers) "
        )
        detail += (
            f"to preferred source slot {slot_id}."
            if slot_id
            else "to a source slot."
        )
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_ADD,
                pallet_id=pallet.pallet_id,
                item_id=item_id,
                slot_id=slot_id,
                reason=detail,
                rule=RuleId.ADD_VISIBLE_PRESTAGE,
            )
        )
        claimed_items.add(item_id)
        if slot_id:
            claimed_slots.add(slot_id)
    return recs


def rule_add_waved_demand(
    state: WorldState,
    held_ids: set[str],
    config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[Recommendation]:
    """
    FLAG_ADD when an item has waved demand in the next 4 hours, face stock is at or
    below the resupply threshold, and an eligible unslotted pallet exists.

    When *config.preferred_source_face_by_item* is set, promotes pallets onto
    their preferred face (mid-day face swap) instead of the first empty slot.
    """
    from slotting.inventory import item_needs_face_add

    preferred = config.preferred_source_face_by_item
    ranks = config.setup_rank_by_item
    threshold = config.effective_resupply_threshold(state)
    empty_slots = sorted(
        s.slot_id
        for s in state.slots
        if s.slot_type == SlotType.SOURCE and not s.occupied_by_pallet_id
    )
    empty_set = set(empty_slots)
    claimed_items: set[str] = set()
    claimed_slots: set[str] = set()
    recs: list[Recommendation] = []

    def _best_pallet(item_id: str) -> Pallet | None:
        from slotting.inventory_policy import select_best_supply_pallet

        candidates = _eligible_unslotted_pallets(state, item_id, held_ids)
        return select_best_supply_pallet(candidates)

    from slotting.inventory import waved_layers_by_item
    from slotting.inventory_policy import resupply_priority_key

    waved = _items_with_waved_demand(state)
    waved_layers = waved_layers_by_item(state)
    waved_items = sorted(
        waved,
        key=lambda item_id: resupply_priority_key(state, item_id, waved_layers=waved_layers),
    )

    for slot_id in empty_slots:
        best_item: str | None = None
        best_rank = 10**9
        for item_id in waved_items:
            if item_id in claimed_items or not item_needs_face_add(state, item_id, threshold=threshold):
                continue
            if preferred.get(item_id) != slot_id:
                continue
            if _best_pallet(item_id) is None:
                continue
            rank = ranks.get(item_id, 10**9)
            if rank < best_rank:
                best_rank = rank
                best_item = item_id
        if best_item is None:
            continue
        pallet = _best_pallet(best_item)
        assert pallet is not None
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_ADD,
                pallet_id=pallet.pallet_id,
                item_id=best_item,
                slot_id=slot_id,
                reason=(
                    f"Waved demand for {best_item}; face swap to preferred {slot_id} "
                    f"with pallet {pallet.pallet_id} ({pallet.layers_remaining} layers)."
                ),
                rule=RuleId.ADD_WAVED_DEMAND,
            )
        )
        claimed_items.add(best_item)
        claimed_slots.add(slot_id)

    for item_id in waved_items:
        if item_id in claimed_items or not item_needs_face_add(state, item_id, threshold=threshold):
            continue
        pallet = _best_pallet(item_id)
        if pallet is None:
            continue
        slot_id: str | None = None
        pref = preferred.get(item_id)
        if pref and pref in empty_set and pref not in claimed_slots:
            slot_id = pref
        elif config.source_face_station_by_id:
            # Soft travel landing: nearest empty face to preferred when preferred
            # is occupied — keeps mid-day ADDs in the active wave zone.
            from layout_advisor.planning.wave_slotting import nearest_empty_face

            slot_id = nearest_empty_face(
                pref,
                empty_slots,
                face_station_by_id=config.source_face_station_by_id,
                claimed=claimed_slots,
            )
        detail = (
            f"Waved demand for {item_id} in the next 4 hours; add unslotted "
            f"pallet {pallet.pallet_id} ({pallet.layers_remaining} layers) "
        )
        if slot_id and pref and slot_id == pref:
            detail += f"to preferred source slot {slot_id}."
        elif slot_id and pref:
            detail += (
                f"to travel-nearest empty slot {slot_id} "
                f"(preferred {pref} busy)."
            )
        elif slot_id:
            detail += f"to source slot {slot_id}."
        else:
            detail += "to a source slot."
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_ADD,
                pallet_id=pallet.pallet_id,
                item_id=item_id,
                slot_id=slot_id,
                reason=detail,
                rule=RuleId.ADD_WAVED_DEMAND,
            )
        )
        claimed_items.add(item_id)
        if slot_id:
            claimed_slots.add(slot_id)
    return recs


def rule_wait_supply_shortage(state: WorldState, held_ids: set[str], config: RuleConfig = DEFAULT_RULE_CONFIG) -> list[Recommendation]:
    """
    FLAG_WAIT when an item has waved demand but no eligible pallet is available.

    Uses the daytime resupply threshold (not the late-shift any-depth sentinel):
    a healthy face above the day threshold should not raise WAIT just because
    evening promote policy is more aggressive.
    """
    from slotting.inventory import item_needs_face_resupply
    from slotting.resupply_rack import item_rack_ready

    threshold = config.resupply_layer_threshold
    recs: list[Recommendation] = []
    for item_id in sorted(_items_with_waved_demand(state)):
        if not item_needs_face_resupply(state, item_id, threshold=threshold):
            continue
        if _eligible_unslotted_pallets(state, item_id, held_ids):
            continue
        if config.include_rack_aisle and item_rack_ready(state, item_id):
            continue
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_WAIT,
                pallet_id=None,
                item_id=item_id,
                reason=(
                    f"Waved demand for {item_id} in the next 4 hours but no eligible "
                    f"pallet available (not excluded, not held): wait for supply."
                ),
                rule=RuleId.WAIT_SUPPLY_SHORTAGE,
            )
        )
    return recs


def apply_suppression(recs: list[Recommendation], held_pallet_ids: set[str]) -> list[Recommendation]:
    """
    DO_NOT_TOUCH suppresses FLAG_REMOVE and FLAG_ADD for the same pallet.

    Returns recommendations in stable order: holds first, then removes, adds, waits.
    """
    filtered: list[Recommendation] = []
    for rec in recs:
        if rec.action in (RecommendationAction.FLAG_REMOVE, RecommendationAction.FLAG_ADD):
            if rec.pallet_id and rec.pallet_id in held_pallet_ids:
                continue
        filtered.append(rec)

    order = {
        RecommendationAction.DO_NOT_TOUCH: 0,
        RecommendationAction.FLAG_DEACTIVATE_DEST: 1,
        RecommendationAction.FLAG_REMOVE: 2,
        RecommendationAction.FLAG_BACKUP_PROMOTE: 3,
        RecommendationAction.FLAG_RACK_PROMOTE: 3,
        RecommendationAction.FLAG_FACE_REFRESH: 3,
        RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY: 3,
        RecommendationAction.FLAG_STORAGE_RETURN: 3,
        RecommendationAction.FLAG_AISLE_STAGE: 3,
        RecommendationAction.FLAG_ACTIVATE_DEST: 4,
        RecommendationAction.FLAG_ADD: 5,
        RecommendationAction.FLAG_WAIT: 6,
    }
    return sorted(filtered, key=lambda r: (order[r.action], r.item_id or "", r.pallet_id or ""))


def run_rules(state: WorldState, config: RuleConfig = DEFAULT_RULE_CONFIG) -> list[Recommendation]:
    """Evaluate enabled rules and return non-conflicting recommendations."""
    held_ids = _held_pallet_ids(state)
    all_recs: list[Recommendation] = []
    if config.is_enabled(RuleId.LOW_LAYERS):
        all_recs.extend(rule_low_layers(state))
    if config.is_enabled(RuleId.ZERO_DEMAND) and config.proactive_zero_demand_remove:
        if not config.proactive_zero_demand_realignment_only or (
            state.as_of.hour in config.proactive_zero_demand_realignment_hours
        ):
            all_recs.extend(rule_zero_demand(state, config))
    if config.is_enabled(RuleId.ZERO_DEMAND_RECLAIM):
        all_recs.extend(rule_aggressive_zero_demand_reclaim(state, config))
    if config.is_enabled(RuleId.VELOCITY_TWIN_RECLAIM):
        all_recs.extend(rule_velocity_twin_face_reclaim(state, config))
    if config.is_enabled(RuleId.EXCLUDED_ITEM):
        all_recs.extend(rule_excluded_item(state))
    if config.is_enabled(RuleId.CROSS_AREA_HOLD):
        all_recs.extend(rule_cross_area_hold(state))
    if config.is_enabled(RuleId.BACKUP_PROMOTE):
        all_recs.extend(rule_backup_swap_with_config(state, held_ids, config))
    if config.include_rack_aisle and config.is_enabled(RuleId.RACK_PROMOTE):
        all_recs.extend(rule_rack_promote_with_config(state, held_ids, config))
    if config.low_layer_refresh_enabled:
        all_recs.extend(rule_face_refresh(state, held_ids, config))
    if config.is_enabled(RuleId.ADD_VISIBLE_PRESTAGE):
        all_recs.extend(rule_add_visible_prestage(state, held_ids, config))
    if config.is_enabled(RuleId.ADD_WAVED_DEMAND):
        all_recs.extend(rule_add_waved_demand(state, held_ids, config))
    if config.is_enabled(RuleId.WAIT_SUPPLY_SHORTAGE):
        all_recs.extend(rule_wait_supply_shortage(state, held_ids, config))

    from slotting.destination_rules import run_destination_rules
    from slotting.wave_ratio import run_wave_ratio_rules

    all_recs.extend(run_destination_rules(state, config))
    all_recs.extend(run_wave_ratio_rules(state, config))
    return apply_suppression(all_recs, held_ids)
