"""Resolve engine CSV columns from WMS export headers (JDA-style names)."""

from __future__ import annotations

import pandas as pd

# Engine logical name -> accepted WMS / export column names (first match wins).
PALLET_ALIASES: dict[str, list[str]] = {
    "pallet_id": ["pallet_id", "lodnum", "lpn"],
    "item_id": ["item_id", "prtnum", "material_code"],
    "layers_remaining": ["layers_remaining", "layers_rem", "rem_layers"],
    "current_slot": ["current_slot", "stoloc"],
    "arrived_at": ["arrived_at", "adddte", "date_added"],
}

SLOT_ALIASES: dict[str, list[str]] = {
    "slot_id": ["slot_id", "stoloc"],
    "slot_type": ["slot_type", "slot_role"],
    "occupied_by_pallet_id": ["occupied_by_pallet_id", "lodnum", "lpn"],
}

ORDER_ALIASES: dict[str, list[str]] = {
    "order_id": ["order_id", "order_line_id", "ord_line_key"],
    "item_id": ["item_id", "prtnum", "material_code"],
    "qty_layers": ["qty_layers", "pick_layers", "rem_layers"],
    "due_at": ["due_at", "late_shpdte", "ship_by_date", "ship_date"],
    "status": ["status", "plan_status"],
    "area": ["area", "pick_area"],
    "arrived_at": ["arrived_at", "adddte", "insdte", "order_adddte", "created_at"],
}

EXCLUSION_ALIASES: dict[str, list[str]] = {
    "item_id": ["item_id", "prtnum", "invt_lev1", "material_code"],
    "reason": ["reason", "exclusion_reason", "comments"],
}


def normalize_header(name: object) -> str:
    """Fold a CSV header to its comparable core: lowercase alphanumerics only.

    Real exports do not spell headers the way a schema does. ``Item ID``,
    ``item-id``, ``Item_Id``, ``ITEM #`` and ``item id `` all normalize to
    ``itemid``, so one alias entry covers the spelling variants instead of the
    alias list having to enumerate them.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def match_column(columns: list, candidates: list[str]) -> object | None:
    """First column matching any *candidate*: exact wins, then normalized.

    Exact-first matters when a sheet carries both ``item_id`` and ``Item ID``
    — the schema spelling wins rather than whichever pandas happened to order
    first.
    """
    for candidate in candidates:
        if candidate in columns:
            return candidate
    norm_map: dict[str, object] = {}
    for col in columns:
        norm_map.setdefault(normalize_header(col), col)
    for candidate in candidates:
        hit = norm_map.get(normalize_header(candidate))
        if hit is not None:
            return hit
    return None


def missing_column_error(logical: str, candidates: list[str], columns: list) -> str:
    """One actionable message: what is needed, what is accepted, what you gave."""
    return (
        f"Missing column for '{logical}'.\n"
        f"  Accepted names (case/spacing/punctuation ignored): {', '.join(candidates)}\n"
        f"  Columns found in the file: {', '.join(str(c) for c in columns) or '(none)'}\n"
        f"  Fix: rename one of your columns to '{logical}', or add its spelling to "
        f"the alias list in slotting/jda_columns.py or layout_advisor/intake/columns.py."
    )


def resolve_columns(
    df: pd.DataFrame,
    aliases: dict[str, list[str]],
    *,
    optional: tuple[str, ...] = (),
) -> dict[str, pd.Series]:
    """
    Map a DataFrame to the engine's logical column names.

    Accepts either friendly names (pallet_id) or JDA export names (lodnum), in
    any capitalization or punctuation (see :func:`normalize_header`).
    Logical names in *optional* are omitted from the result when absent
    instead of raising (e.g. ``arrived_at`` on pre-censor order feeds).
    """
    columns = list(df.columns)
    resolved: dict[str, pd.Series] = {}
    for logical, candidates in aliases.items():
        hit = match_column(columns, candidates)
        if hit is not None:
            resolved[logical] = df[hit]
        elif logical not in optional:
            raise KeyError(missing_column_error(logical, candidates, columns))
    return resolved
