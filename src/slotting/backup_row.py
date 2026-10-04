"""Backup-row / resupply physics beside source faces."""

from __future__ import annotations

from dataclasses import dataclass

from slotting.models import Recommendation, RecommendationAction, RuleId, Slot, SlotType, WorldState


@dataclass(frozen=True)
class BackupSwapPlan:
    """Atomic in-aisle swap: relocate face pallet, promote backup to face."""

    slot_id: str
    backup_pallet_id: str
    item_id: str
    face_pallet_id: str | None
    resupply_slot_id: str | None
    to_storage: bool


def empty_resupply_slots(slots: list[Slot], *, exclude: str | None = None) -> list[Slot]:
    """
    Source slots with an empty backup/resupply row.

    A loaded resupply pallet cannot be pushed — relocation targets must be empty.
    """
    return sorted(
        (
            s
            for s in slots
            if s.slot_type == SlotType.SOURCE
            and s.slot_id != exclude
            and not s.backup_pallet_id
        ),
        key=lambda s: s.slot_id,
    )


def resupply_targets_for_state(
    state: WorldState,
    *,
    exclude: str | None = None,
    include_rack_aisle: bool = False,
) -> list[Slot]:
    """Relocation targets: backup rows, optionally plus resupply rack aisle."""
    targets = empty_resupply_slots(state.slots, exclude=exclude)
    if include_rack_aisle:
        from slotting.resupply_rack import empty_rack_slots

        targets = targets + empty_rack_slots(state)
        targets = sorted(targets, key=lambda s: (s.slot_type.value, s.slot_id))
    return targets


def backup_is_accessible(slot: Slot) -> bool:
    """
    Backup pallet exists and is one row behind the face (single-depth model).

    The face physically blocks the backup until the face pallet is moved.
    """
    return bool(slot.backup_pallet_id)


def plan_backup_swap(
    state: WorldState,
    slot: Slot,
    *,
    waved_items: set[str],
    allow_storage_fallback: bool = True,
    allow_aisle_staging: bool = False,
    include_rack_aisle: bool = False,
) -> BackupSwapPlan | None:
    """
    Plan promote backup → face when waved demand matches backup SKU.

    If a face pallet is present it must move to an empty resupply row elsewhere,
    to temporary aisle staging, or to storage (discouraged).
    """
    if slot.slot_type != SlotType.SOURCE or not backup_is_accessible(slot):
        return None

    backup = state.pallet_by_id(slot.backup_pallet_id)  # type: ignore[arg-type]
    if backup is None or backup.item_id not in waved_items:
        return None
    if not state.is_eligible_for_line(backup):
        return None

    face_id = slot.occupied_by_pallet_id
    if face_id:
        face = state.pallet_by_id(face_id)
        if face is None:
            return None
        resupply_targets = resupply_targets_for_state(
            state, exclude=slot.slot_id, include_rack_aisle=include_rack_aisle
        )
        if resupply_targets:
            return BackupSwapPlan(
                slot_id=slot.slot_id,
                backup_pallet_id=backup.pallet_id,
                item_id=backup.item_id,
                face_pallet_id=face.pallet_id,
                resupply_slot_id=resupply_targets[0].slot_id,
                to_storage=False,
            )
        if allow_aisle_staging:
            return BackupSwapPlan(
                slot_id=slot.slot_id,
                backup_pallet_id=backup.pallet_id,
                item_id=backup.item_id,
                face_pallet_id=face.pallet_id,
                resupply_slot_id=None,
                to_storage=False,
            )
        if allow_storage_fallback:
            return BackupSwapPlan(
                slot_id=slot.slot_id,
                backup_pallet_id=backup.pallet_id,
                item_id=backup.item_id,
                face_pallet_id=face.pallet_id,
                resupply_slot_id=None,
                to_storage=True,
            )
        return None

    return BackupSwapPlan(
        slot_id=slot.slot_id,
        backup_pallet_id=backup.pallet_id,
        item_id=backup.item_id,
        face_pallet_id=None,
        resupply_slot_id=None,
        to_storage=False,
    )


def item_swap_ready(
    state: WorldState,
    item_id: str,
    *,
    allow_storage_fallback: bool = False,
    allow_aisle_staging: bool = False,
    include_rack_aisle: bool = False,
) -> bool:
    """
    True when *item_id* sits on a backup row and can promote to face in-aisle.

    Requires an empty resupply row to relocate a blocking face pallet, unless
    the face is already empty. Cross-SKU backup behind another face is OK.
    """
    if item_id in state.excluded_item_ids:
        return False
    for slot in state.source_slots:
        if not slot.backup_pallet_id:
            continue
        backup = state.pallet_by_id(slot.backup_pallet_id)
        if backup is None or backup.item_id != item_id:
            continue
        if not state.is_eligible_for_line(backup):
            continue
        plan = plan_backup_swap(
            state,
            slot,
            waved_items={item_id},
            allow_storage_fallback=allow_storage_fallback,
            allow_aisle_staging=allow_aisle_staging,
            include_rack_aisle=include_rack_aisle,
        )
        if plan is not None:
            return True
    if include_rack_aisle:
        from slotting.resupply_rack import item_rack_ready

        if item_rack_ready(state, item_id, allow_storage_fallback=allow_storage_fallback):
            return True
    return False


def rule_backup_swap(state: WorldState, held_ids: set[str], *, allow_aisle_staging: bool = False, include_rack_aisle: bool = False) -> list[Recommendation]:
    """
    Recommend in-aisle backup swap when waved demand is on the backup row.

    Face pallet (if any) must relocate to an empty resupply position or storage.
    """
    from slotting.rules import _items_with_waved_demand

    recs: list[Recommendation] = []
    waved = _items_with_waved_demand(state)
    for slot in state.source_slots:
        if slot.backup_pallet_id in held_ids:
            continue
        plan = plan_backup_swap(
            state, slot, waved_items=waved, allow_aisle_staging=allow_aisle_staging, include_rack_aisle=include_rack_aisle
        )
        if plan is None:
            continue
        if plan.face_pallet_id and plan.face_pallet_id in held_ids:
            continue
        face = state.pallet_by_id(plan.face_pallet_id) if plan.face_pallet_id else None
        if face is not None and face.item_id in waved:
            continue
        if plan.face_pallet_id:
            dest = (
                f"resupply {plan.resupply_slot_id}"
                if plan.resupply_slot_id
                else "storage (no empty resupply)"
            )
            reason = (
                f"Waved demand for {plan.item_id}; relocate face {plan.face_pallet_id} "
                f"to {dest}, promote backup {plan.backup_pallet_id} to {plan.slot_id}."
            )
        else:
            reason = (
                f"Waved demand for {plan.item_id}; promote backup "
                f"{plan.backup_pallet_id} to empty face {plan.slot_id}."
            )
        recs.append(
            Recommendation(
                action=RecommendationAction.FLAG_BACKUP_PROMOTE,
                pallet_id=plan.backup_pallet_id,
                item_id=plan.item_id,
                slot_id=plan.slot_id,
                reason=reason,
                rule=RuleId.BACKUP_PROMOTE,
            )
        )
    return recs
