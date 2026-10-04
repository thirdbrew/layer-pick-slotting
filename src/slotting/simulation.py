"""What-if simulation: baseline state vs. projected state after applying recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from slotting.metrics import RunMetrics, compute_metrics
from slotting.throughput import (
    ThroughputMetrics,
    compute_throughput,
    compute_wave_throughput,
    item_has_pickable_face_supply,
)
from slotting.wave_ratio import WaveRatioAdvice, compute_wave_ratio
from slotting.models import (
    OrderArea,
    OrderStatus,
    Pallet,
    Recommendation,
    RecommendationAction,
    RuleId,
    Slot,
    SlotType,
    WorldState,
)
from slotting.rules import _items_with_waved_demand, run_rules
from slotting.rules import DEFAULT_RULE_CONFIG, RuleConfig, WAVED_HORIZON


@dataclass(frozen=True)
class AppliedAction:
    """Record of a simulated slot move applied to the projected state."""

    action: RecommendationAction
    pallet_id: str | None
    item_id: str | None
    slot_id: str | None
    detail: str


@dataclass(frozen=True)
class SimulationResult:
    """
    Compare KPIs before and after hypothetically following recommendations.

    *baseline* is the current world (do nothing).
    *projected* is the state after applying FLAG_REMOVE and FLAG_ADD actions.
    DO_NOT_TOUCH and FLAG_WAIT do not change slot assignments.
    """

    as_of: datetime
    baseline: RunMetrics
    projected: RunMetrics
    baseline_throughput: ThroughputMetrics
    projected_throughput: ThroughputMetrics
    baseline_wave_throughput: ThroughputMetrics
    projected_wave_throughput: ThroughputMetrics
    baseline_state: WorldState
    projected_state: WorldState
    recommendations: list[Recommendation]
    applied: list[AppliedAction] = field(default_factory=list)
    blocked_adds: list[Recommendation] = field(default_factory=list)
    wave_ratio: WaveRatioAdvice | None = None

    @property
    def dead_slot_rate_improvement_pct(self) -> float:
        """Positive means fewer dead slots after following recommendations."""
        return (self.baseline.dead_slot_rate - self.projected.dead_slot_rate) * 100

    @property
    def shortage_improvement(self) -> int:
        """Positive means fewer supply shortages after following recommendations."""
        return self.baseline.shortage_count - self.projected.shortage_count

    @property
    def dead_slots_freed(self) -> int:
        return self.baseline.dead_slots - self.projected.dead_slots

    @property
    def removes_applied(self) -> int:
        return sum(1 for a in self.applied if a.action == RecommendationAction.FLAG_REMOVE)

    @property
    def adds_applied(self) -> int:
        return sum(1 for a in self.applied if a.action == RecommendationAction.FLAG_ADD)

    @property
    def backup_promotes_applied(self) -> int:
        return sum(1 for a in self.applied if a.action == RecommendationAction.FLAG_BACKUP_PROMOTE)

    @property
    def rack_promotes_applied(self) -> int:
        return sum(1 for a in self.applied if a.action == RecommendationAction.FLAG_RACK_PROMOTE)

    @property
    def resupply_relocates_applied(self) -> int:
        return sum(1 for a in self.applied if a.action == RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY)

    @property
    def donor_stages_applied(self) -> int:
        return sum(
            1
            for a in self.applied
            if (
                a.action == RecommendationAction.FLAG_REMOVE
                and a.detail.startswith("Sent completed build")
            )
            or (
                a.action == RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY
                and a.detail.startswith("Staged completed build")
            )
        )

    @property
    def storage_returns_applied(self) -> int:
        return sum(1 for a in self.applied if a.action == RecommendationAction.FLAG_STORAGE_RETURN)

    @property
    def waved_ready_baseline(self) -> int:
        return _waved_ready_count(self.baseline_state)

    @property
    def waved_ready_projected(self) -> int:
        return _waved_ready_count(self.projected_state)

    @property
    def layer_pick_improvement_pp(self) -> float:
        """Positive means more demand can be layer-picked after following recommendations."""
        return (self.projected_throughput.layer_pick_pct - self.baseline_throughput.layer_pick_pct) * 100

    @property
    def hand_pick_reduction_pp(self) -> float:
        """Positive means less demand routed to hand pick."""
        return (self.baseline_throughput.hand_pick_pct - self.projected_throughput.hand_pick_pct) * 100

    @property
    def layers_shifted_to_layer_pick(self) -> int:
        return (
            self.projected_throughput.layer_pick_ready_layers
            - self.baseline_throughput.layer_pick_ready_layers
        )

    @property
    def low_layer_removes_applied(self) -> int:
        """FLAG_REMOVE actions driven by the >2-layer (LOW_LAYERS) rule."""
        applied_remove_pallets = {
            a.pallet_id for a in self.applied if a.action == RecommendationAction.FLAG_REMOVE
        }
        return sum(
            1
            for r in self.recommendations
            if r.rule == RuleId.LOW_LAYERS
            and r.pallet_id in applied_remove_pallets
        )

    @property
    def wave_layer_pick_pct(self) -> float:
        """Projected layer-pick % for waved demand (4h window)."""
        return self.projected_wave_throughput.layer_pick_pct

    @property
    def baseline_wave_layer_pick_pct(self) -> float:
        return self.baseline_wave_throughput.layer_pick_pct

    @property
    def wave_layer_pick_improvement_pp(self) -> float:
        return (
            self.projected_wave_throughput.layer_pick_pct
            - self.baseline_wave_throughput.layer_pick_pct
        ) * 100

    @property
    def wave_shortage_count(self) -> int:
        """Waved items with no eligible supply after projected rules."""
        return self.projected.shortage_count

    @property
    def baseline_wave_shortage_count(self) -> int:
        return self.baseline.shortage_count

    @property
    def wave_layers_shifted_to_layer_pick(self) -> int:
        return (
            self.projected_wave_throughput.layer_pick_ready_layers
            - self.baseline_wave_throughput.layer_pick_ready_layers
        )


def _copy_state(state: WorldState) -> tuple[list[Pallet], list[Slot]]:
    pallets = [p.model_copy(deep=True) for p in state.pallets]
    slots = [s.model_copy(deep=True) for s in state.slots]
    return pallets, slots


def _empty_source_slot_ids(slots: list[Slot]) -> list[str]:
    return sorted(
        s.slot_id
        for s in slots
        if s.slot_type == SlotType.SOURCE and not s.occupied_by_pallet_id
    )


def _waved_ready_count(state: WorldState) -> int:
    """Items with waved demand that have LP-pickable supply on a source face."""
    waved = _items_with_waved_demand(state)
    return sum(1 for item_id in waved if item_has_pickable_face_supply(state, item_id))


def _waved_layers_for_item(state: WorldState, item_id: str) -> int:
    cutoff = state.as_of + WAVED_HORIZON
    return sum(
        o.qty_layers
        for o in state.orders
        if o.item_id == item_id
        and o.area == OrderArea.THIS_LINE
        and o.status == OrderStatus.WAVED
        and state.as_of <= o.due_at <= cutoff
    )


def _relocate_face_to_target(
    *,
    face: Pallet | None,
    face_slot: Slot,
    target: Slot,
    applied: list[AppliedAction],
    face_pallet_id: str,
) -> None:
    if target.slot_type == SlotType.RESUPPLY_RACK:
        target.occupied_by_pallet_id = face_pallet_id
        if face is not None:
            face.current_slot = target.slot_id
    else:
        target.backup_pallet_id = face_pallet_id
        if face is not None:
            face.current_slot = None
    face_slot.occupied_by_pallet_id = None
    applied.append(
        AppliedAction(
            action=RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY,
            pallet_id=face_pallet_id,
            item_id=face.item_id if face else None,
            slot_id=target.slot_id,
            detail=f"Relocated face {face_pallet_id} to resupply on {target.slot_id}",
        )
    )


def _execute_backup_swaps(
    state: WorldState,
    pallets: list[Pallet],
    slots: list[Slot],
    recommendations: list[Recommendation],
    *,
    allow_storage_fallback: bool = True,
    allow_aisle_staging: bool = False,
    include_rack_aisle: bool = False,
) -> list[AppliedAction]:
    """
    Relocate blocking face pallets, then promote backup → face.

    Face pallets go to an empty resupply row (preferred) or storage when none free.
    """
    from slotting.backup_row import plan_backup_swap

    waved_items = _items_with_waved_demand(state)
    step_state = WorldState(
        pallets=pallets,
        slots=slots,
        orders=state.orders,
        exclusions=state.exclusions,
        as_of=state.as_of,
    )
    slot_by_id = {s.slot_id: s for s in slots}
    applied: list[AppliedAction] = []
    promote_slots = {
        r.slot_id
        for r in recommendations
        if r.action == RecommendationAction.FLAG_BACKUP_PROMOTE and r.slot_id
    }
    seen: set[str] = set()

    for slot_id in sorted(promote_slots):
        if slot_id in seen:
            continue
        slot = slot_by_id.get(slot_id)
        if slot is None or not slot.backup_pallet_id:
            continue
        plan = plan_backup_swap(
            step_state,
            slot,
            waved_items=waved_items,
            allow_storage_fallback=allow_storage_fallback,
            allow_aisle_staging=allow_aisle_staging,
            include_rack_aisle=include_rack_aisle,
        )
        if plan is None:
            continue

        if plan.face_pallet_id:
            face = next((p for p in pallets if p.pallet_id == plan.face_pallet_id), None)
            if plan.resupply_slot_id:
                target = slot_by_id[plan.resupply_slot_id]
                _relocate_face_to_target(
                    face=face,
                    face_slot=slot,
                    target=target,
                    applied=applied,
                    face_pallet_id=plan.face_pallet_id,
                )
            elif allow_aisle_staging:
                if face is not None:
                    face.current_slot = None
                slot.occupied_by_pallet_id = None
                applied.append(
                    AppliedAction(
                        action=RecommendationAction.FLAG_AISLE_STAGE,
                        pallet_id=plan.face_pallet_id,
                        item_id=face.item_id if face else None,
                        slot_id=slot.slot_id,
                        detail=f"Staged face {plan.face_pallet_id} in aisle during backup swap",
                    )
                )
            elif plan.to_storage:
                if face is not None:
                    face.current_slot = None
                slot.occupied_by_pallet_id = None
                applied.append(
                    AppliedAction(
                        action=RecommendationAction.FLAG_STORAGE_RETURN,
                        pallet_id=plan.face_pallet_id,
                        item_id=face.item_id if face else None,
                        slot_id=slot.slot_id,
                        detail=(
                            f"Returned face {plan.face_pallet_id} to storage: "
                            f"no empty resupply row for backup swap"
                        ),
                    )
                )
            else:
                continue

        backup = next((p for p in pallets if p.pallet_id == plan.backup_pallet_id), None)
        if backup is None:
            continue
        slot.occupied_by_pallet_id = plan.backup_pallet_id
        slot.backup_pallet_id = None
        backup.current_slot = slot.slot_id
        seen.add(slot_id)
        applied.append(
            AppliedAction(
                action=RecommendationAction.FLAG_BACKUP_PROMOTE,
                pallet_id=backup.pallet_id,
                item_id=backup.item_id,
                slot_id=slot.slot_id,
                detail=f"Promoted backup {backup.pallet_id} to face on {slot.slot_id}",
            )
        )
        step_state = WorldState(
            pallets=pallets,
            slots=slots,
            orders=state.orders,
            exclusions=state.exclusions,
            as_of=state.as_of,
        )
    return applied


def _execute_rack_promotes(
    state: WorldState,
    pallets: list[Pallet],
    slots: list[Slot],
    recommendations: list[Recommendation],
    *,
    rule_config: RuleConfig | None = None,
) -> list[AppliedAction]:
    from slotting.resupply_rack import plan_rack_to_face

    cfg = rule_config

    waved_items = _items_with_waved_demand(state)
    slot_by_id = {s.slot_id: s for s in slots}
    applied: list[AppliedAction] = []
    promote_faces = {
        r.slot_id
        for r in recommendations
        if r.action == RecommendationAction.FLAG_RACK_PROMOTE and r.slot_id
    }
    seen: set[str] = set()

    for face_slot_id in sorted(promote_faces):
        if face_slot_id in seen:
            continue
        rec = next(
            r
            for r in recommendations
            if r.action == RecommendationAction.FLAG_RACK_PROMOTE and r.slot_id == face_slot_id
        )
        step_state = WorldState(
            pallets=pallets,
            slots=slots,
            orders=state.orders,
            exclusions=state.exclusions,
            as_of=state.as_of,
        )
        # Use the same effective threshold rules already applied when emitting
        # FLAG_RACK_PROMOTE (late-shift sentinel 0 = any face depth).
        promote_threshold = (
            cfg.effective_resupply_threshold(step_state)
            if cfg is not None
            else 3
        )
        plan = plan_rack_to_face(
            step_state,
            item_id=rec.item_id or "",
            waved_items=waved_items,
            preferred_face_by_item=cfg.preferred_source_face_by_item if cfg else None,
            face_station_by_id=cfg.source_face_station_by_id if cfg else None,
            face_flank_by_id=cfg.source_face_flank_by_id if cfg else None,
            rack_slot_by_item=cfg.rack_slot_by_item if cfg else None,
            resupply_layer_threshold=promote_threshold,
            waved_displacement=cfg.waved_displacement if cfg else False,
            waved_displacement_rack_swap=(
                cfg.waved_displacement_rack_swap if cfg else False
            ),
        )
        if plan is None:
            continue

        face_slot = slot_by_id.get(plan.face_slot_id)
        rack_slot = slot_by_id.get(plan.rack_slot_id)
        if face_slot is None or rack_slot is None:
            continue

        if plan.face_pallet_id:
            face = next((p for p in pallets if p.pallet_id == plan.face_pallet_id), None)
            if plan.relocate_to_slot_id:
                target = slot_by_id.get(plan.relocate_to_slot_id)
                if target is None:
                    continue
                _relocate_face_to_target(
                    face=face,
                    face_slot=face_slot,
                    target=target,
                    applied=applied,
                    face_pallet_id=plan.face_pallet_id,
                )
            elif face is not None:
                face_slot.occupied_by_pallet_id = None
                face.current_slot = None

        rack_pallet = next((p for p in pallets if p.pallet_id == plan.rack_pallet_id), None)
        if rack_pallet is None:
            continue
        if plan.relocate_to_slot_id != plan.rack_slot_id:
            # 1-for-1 swap keeps the displaced pallet's ref on the vacated
            # rack slot — clearing it here would ghost the swapped-in pallet.
            rack_slot.occupied_by_pallet_id = None
        face_slot.occupied_by_pallet_id = plan.rack_pallet_id
        rack_pallet.current_slot = face_slot.slot_id
        seen.add(face_slot_id)
        applied.append(
            AppliedAction(
                action=RecommendationAction.FLAG_RACK_PROMOTE,
                pallet_id=rack_pallet.pallet_id,
                item_id=rack_pallet.item_id,
                slot_id=face_slot.slot_id,
                detail=f"Promoted rack pallet {rack_pallet.pallet_id} to face {face_slot.slot_id}",
            )
        )
    return applied


def _execute_donor_staging(
    state: WorldState,
    pallets: list[Pallet],
    slots: list[Slot],
    recommendations: list[Recommendation],
    *,
    include_rack_aisle: bool,
    rack_only_reserve: bool = True,
) -> list[AppliedAction]:
    """Stage completed donor builds behind face or on side rack (legacy layout only)."""
    from slotting.resupply_rack import pick_rack_slot

    if not include_rack_aisle or rack_only_reserve:
        return []

    slot_by_id = {s.slot_id: s for s in slots}
    applied: list[AppliedAction] = []
    stage_recs = [
        r
        for r in recommendations
        if r.action == RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY
        and r.rule == RuleId.DEST_STAGE_COMPLETE
        and r.pallet_id
        and r.slot_id
    ]
    seen: set[str] = set()
    for rec in stage_recs:
        if rec.pallet_id in seen:
            continue
        donor_slot = slot_by_id.get(rec.slot_id or "")
        if donor_slot is None or donor_slot.slot_type != SlotType.DESTINATION:
            continue
        pallet = next((p for p in pallets if p.pallet_id == rec.pallet_id), None)
        if pallet is None:
            continue
        target = None
        dest_label = ""
        rack_state = WorldState(
            pallets=pallets, slots=slots, orders=state.orders, exclusions=state.exclusions, as_of=state.as_of
        )
        if not donor_slot.backup_pallet_id:
            target = donor_slot
            dest_label = f"behind {donor_slot.slot_id}"
        else:
            target = pick_rack_slot(rack_state, item_id=pallet.item_id)
            if target is not None:
                dest_label = target.slot_id
        if target is None:
            continue
        if target.slot_type == SlotType.DESTINATION:
            target.backup_pallet_id = pallet.pallet_id
            pallet.current_slot = None
        else:
            target.occupied_by_pallet_id = pallet.pallet_id
            pallet.current_slot = target.slot_id
        donor_slot.occupied_by_pallet_id = None
        seen.add(rec.pallet_id)
        applied.append(
            AppliedAction(
                action=RecommendationAction.FLAG_RELOCATE_TO_RESUPPLY,
                pallet_id=pallet.pallet_id,
                item_id=pallet.item_id,
                slot_id=target.slot_id,
                detail=f"Staged completed build {pallet.pallet_id} to {dest_label}",
            )
        )
    return applied


def _execute_face_refresh(
    pallets: list[Pallet],
    slots: list[Slot],
    recommendations: list[Recommendation],
    *,
    as_of: datetime | None = None,
) -> list[AppliedAction]:
    slot_by_id = {s.slot_id: s for s in slots}
    pallet_by_id = {p.pallet_id: p for p in pallets}
    applied: list[AppliedAction] = []
    seen: set[str] = set()

    for rec in recommendations:
        if rec.action != RecommendationAction.FLAG_FACE_REFRESH:
            continue
        if not rec.pallet_id or not rec.slot_id or rec.pallet_id in seen:
            continue
        face_slot = slot_by_id.get(rec.slot_id)
        fresh = pallet_by_id.get(rec.pallet_id)
        if face_slot is None or fresh is None or fresh.current_slot:
            continue
        old_id = face_slot.occupied_by_pallet_id
        old = pallet_by_id.get(old_id) if old_id else None
        if old is not None:
            from slotting.inventory_policy import mark_removed_from_pick

            if as_of is not None:
                mark_removed_from_pick(old, as_of=as_of, source_slot_id=face_slot.slot_id)
            old.current_slot = None
        from slotting.inventory_policy import clear_pending_completion

        clear_pending_completion(fresh)
        face_slot.occupied_by_pallet_id = fresh.pallet_id
        fresh.current_slot = face_slot.slot_id
        seen.add(rec.pallet_id)
        applied.append(
            AppliedAction(
                action=RecommendationAction.FLAG_FACE_REFRESH,
                pallet_id=fresh.pallet_id,
                item_id=fresh.item_id,
                slot_id=face_slot.slot_id,
                detail=(
                    f"Refreshed face {face_slot.slot_id} with {fresh.pallet_id}"
                    + (f" (replaced {old_id})" if old_id else "")
                ),
            )
        )
    return applied


def apply_recommendations(
    state: WorldState,
    recommendations: list[Recommendation],
    *,
    allow_aisle_staging: bool = False,
    include_rack_aisle: bool = False,
    rack_only_reserve: bool = True,
    rule_config: RuleConfig | None = None,
) -> tuple[WorldState, list[AppliedAction]]:
    """
    Apply actionable recommendations to produce a projected world state.

    Processing order:
    1. FLAG_REMOVE — clear source slot assignments (frees capacity)
    2. Backup swap — relocate face to empty resupply (or storage), promote backup
    3. FLAG_ADD — place unslotted pallets into empty source slots (storage trip)
    4. Destination build face open/close

    DO_NOT_TOUCH and FLAG_WAIT are advisory only and do not mutate state.
    """
    pallets, slots = _copy_state(state)
    pallet_by_id = {p.pallet_id: p for p in pallets}
    slot_by_id = {s.slot_id: s for s in slots}
    applied: list[AppliedAction] = []

    remove_recs = [
        r for r in recommendations if r.action == RecommendationAction.FLAG_REMOVE and r.pallet_id
    ]
    seen_remove: set[str] = set()
    for rec in remove_recs:
        pid = rec.pallet_id
        assert pid is not None
        if pid in seen_remove:
            continue
        seen_remove.add(pid)
        pallet = pallet_by_id.get(pid)
        if pallet is None or not pallet.current_slot:
            continue
        slot_id = pallet.current_slot
        slot = slot_by_id.get(slot_id)
        if slot is None:
            continue
        slot.occupied_by_pallet_id = None
        pallet.current_slot = None
        from slotting.inventory_policy import mark_removed_from_pick

        mark_removed_from_pick(pallet, as_of=state.as_of, source_slot_id=slot_id)
        rec = next((r for r in remove_recs if r.pallet_id == pid), None)
        if rec and rec.rule == RuleId.DEST_STAGE_COMPLETE:
            detail = f"Sent completed build {pid} from {slot_id} to wrapping"
        else:
            detail = f"Removed {pid} from {slot_id}"
        applied.append(
            AppliedAction(
                action=RecommendationAction.FLAG_REMOVE,
                pallet_id=pid,
                item_id=pallet.item_id,
                slot_id=slot_id,
                detail=detail,
            )
        )

    applied.extend(
        _execute_backup_swaps(
            state,
            pallets,
            slots,
            recommendations,
            allow_aisle_staging=allow_aisle_staging,
            include_rack_aisle=include_rack_aisle,
        )
    )
    if include_rack_aisle:
        applied.extend(
            _execute_rack_promotes(state, pallets, slots, recommendations, rule_config=rule_config)
        )
        applied.extend(
            _execute_donor_staging(
                state,
                pallets,
                slots,
                recommendations,
                include_rack_aisle=True,
                rack_only_reserve=rack_only_reserve,
            )
        )
    applied.extend(_execute_face_refresh(pallets, slots, recommendations, as_of=state.as_of))

    add_recs = [
        r
        for r in recommendations
        if r.action == RecommendationAction.FLAG_ADD
        and r.pallet_id
        and r.rule != RuleId.DEST_ASSIGN_WAVED
    ]
    claimed_add_slots: set[str] = set()
    for rec in add_recs:
        pid = rec.pallet_id
        assert pid is not None
        pallet = pallet_by_id.get(pid)
        if pallet is None or pallet.current_slot:
            continue
        if rec.item_id:
            from slotting.inventory import active_face_pallet_count

            if active_face_pallet_count(
                WorldState(
                    pallets=pallets,
                    slots=slots,
                    orders=state.orders,
                    exclusions=state.exclusions,
                    as_of=state.as_of,
                ),
                rec.item_id,
            ) > 0:
                continue
        empty_slots = [
            sid
            for sid in _empty_source_slot_ids(slots)
            if sid not in claimed_add_slots
        ]
        if not empty_slots:
            continue
        if rec.slot_id and rec.slot_id in empty_slots:
            slot_id = rec.slot_id
        else:
            from slotting.slotting_policy import best_empty_source_slot

            slot_id = best_empty_source_slot(
                empty_slots,
                item_id=rec.item_id,
                preferred_slot=rec.slot_id,
                config=rule_config,
            )
        slot = slot_by_id[slot_id]
        slot.occupied_by_pallet_id = pid
        pallet.current_slot = slot_id
        claimed_add_slots.add(slot_id)
        detail = f"Added {pid} ({rec.item_id}) to {slot_id}"
        if rec.slot_id == slot_id and "preferred" in rec.reason.lower():
            detail = f"Face swap: {detail}"
        applied.append(
            AppliedAction(
                action=RecommendationAction.FLAG_ADD,
                pallet_id=pid,
                item_id=rec.item_id,
                slot_id=slot_id,
                detail=detail,
            )
        )

    dest_add_recs = [
        r
        for r in recommendations
        if r.action == RecommendationAction.FLAG_ADD
        and r.slot_id
        and r.rule == RuleId.DEST_ASSIGN_WAVED
    ]
    for rec in dest_add_recs:
        slot_id = rec.slot_id
        assert slot_id is not None and rec.item_id is not None
        slot = slot_by_id.get(slot_id)
        if slot is None or slot.slot_type != SlotType.DESTINATION or slot.occupied_by_pallet_id:
            continue
        build_id = f"BUILD-{slot_id}"
        if build_id in pallet_by_id:
            continue
        layers = _waved_layers_for_item(state, rec.item_id)
        build_pallet = Pallet(
            pallet_id=build_id,
            item_id=rec.item_id,
            layers_remaining=max(layers, 1),
            current_slot=slot_id,
            arrived_at=state.as_of,
        )
        pallets.append(build_pallet)
        pallet_by_id[build_id] = build_pallet
        slot.occupied_by_pallet_id = build_id
        applied.append(
            AppliedAction(
                action=RecommendationAction.FLAG_ADD,
                pallet_id=build_id,
                item_id=rec.item_id,
                slot_id=slot_id,
                detail=f"Opened build {build_id} ({rec.item_id}) on {slot_id}",
            )
        )

    projected = WorldState(
        pallets=pallets,
        slots=slots,
        orders=state.orders,
        exclusions=state.exclusions,
        as_of=state.as_of,
    )
    return projected, applied


def simulate_once(
    state: WorldState,
    *,
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> SimulationResult:
    """
    Run the rules engine and compare baseline KPIs to a projected after-state.

    Returns a :class:`SimulationResult` with deltas suitable for estimating
    improvement if operators follow recommendations.
    """
    baseline_metrics = compute_metrics(state)
    baseline_throughput = compute_throughput(state)
    baseline_wave_throughput = compute_wave_throughput(state)
    recommendations = run_rules(state, rule_config)
    wave_ratio = compute_wave_ratio(state)
    projected_state, applied = apply_recommendations(
        state,
        recommendations,
        allow_aisle_staging=rule_config.allow_aisle_staging,
        include_rack_aisle=rule_config.include_rack_aisle,
        rack_only_reserve=rule_config.rack_only_reserve,
        rule_config=rule_config,
    )

    # Destination rules need staged source supply — re-run after source ADD/REMOVE.
    second_pass_recs = run_rules(projected_state, rule_config)
    projected_state, applied_second = apply_recommendations(
        projected_state,
        second_pass_recs,
        allow_aisle_staging=rule_config.allow_aisle_staging,
        include_rack_aisle=rule_config.include_rack_aisle,
        rack_only_reserve=rule_config.rack_only_reserve,
        rule_config=rule_config,
    )
    applied = applied + applied_second
    recommendations = recommendations + second_pass_recs

    from slotting.inventory import finalize_depleted_pallets

    cleaned_pallets = finalize_depleted_pallets(
        list(projected_state.pallets),
        list(projected_state.slots),
        state=projected_state,
    )
    projected_state = WorldState(
        pallets=cleaned_pallets,
        slots=projected_state.slots,
        orders=projected_state.orders,
        exclusions=projected_state.exclusions,
        as_of=projected_state.as_of,
    )

    add_recs = [r for r in recommendations if r.action == RecommendationAction.FLAG_ADD and r.pallet_id]
    applied_add_pallets = {a.pallet_id for a in applied if a.action == RecommendationAction.FLAG_ADD}
    blocked_adds = [r for r in add_recs if r.pallet_id not in applied_add_pallets]

    projected_metrics = compute_metrics(projected_state)
    projected_throughput = compute_throughput(projected_state)
    projected_wave_throughput = compute_wave_throughput(projected_state)

    return SimulationResult(
        as_of=state.as_of,
        baseline=baseline_metrics,
        projected=projected_metrics,
        baseline_throughput=baseline_throughput,
        projected_throughput=projected_throughput,
        baseline_wave_throughput=baseline_wave_throughput,
        projected_wave_throughput=projected_wave_throughput,
        baseline_state=state,
        projected_state=projected_state,
        recommendations=recommendations,
        applied=applied,
        blocked_adds=blocked_adds,
        wave_ratio=wave_ratio,
    )


def simulate_timeline(
    state: WorldState,
    *,
    start: datetime,
    end: datetime,
    step: timedelta = timedelta(hours=1),
    rule_config: RuleConfig = DEFAULT_RULE_CONFIG,
) -> list[SimulationResult]:
    """
    Replay simulation at regular time steps using the same CSV snapshot.

    Order visibility windows shift with *as_of*, so this shows how much
    improvement the rules could deliver across a planning day without
    needing multiple data files.
    """
    results: list[SimulationResult] = []
    current = start
    while current <= end:
        step_state = WorldState(
            pallets=state.pallets,
            slots=state.slots,
            orders=state.orders,
            exclusions=state.exclusions,
            as_of=current,
        )
        results.append(simulate_once(step_state, rule_config=rule_config))
        current += step
    return results
