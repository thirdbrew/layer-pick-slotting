"""Column alias resolution for customer intake CSVs.

Header matching and the missing-column message are shared with the JDA path so
the two intake surfaces cannot drift apart on what they accept or how they
explain a rejection.
"""

from __future__ import annotations

import pandas as pd

from slotting.jda_columns import match_column, missing_column_error, normalize_header

__all__ = [
    "ORDER_ALIASES",
    "SKU_ALIASES",
    "EXCLUSION_ALIASES",
    "PICK_HISTORY_ALIASES",
    "normalize_header",
    "resolve_columns",
    "require_columns",
]

ORDER_ALIASES: dict[str, list[str]] = {
    "item_id": ["item_id", "prtnum", "sku", "item", "product_id"],
    "ship_date": ["ship_date", "late_shpdte", "due_date", "ship_dt", "shpdte"],
    "qty_layers": ["qty_layers", "pick_layers", "layers", "layer_qty"],
    "qty_cases": ["qty_cases", "pick_cases", "cases", "case_qty"],
    "order_id": ["order_id", "order_line_id", "order_num", "ordnum"],
}

SKU_ALIASES: dict[str, list[str]] = {
    "item_id": ["item_id", "prtnum", "sku"],
    "layers_per_pallet": ["layers_per_pallet", "lyr_per_plt", "layers_per_plt"],
    "cases_per_layer": ["cases_per_layer", "cases_per_lyr", "cs_per_layer"],
    "length_in": ["length_in", "len_in", "length"],
    "width_in": ["width_in", "wid_in", "width"],
    "height_in": ["height_in", "hgt_in", "height"],
    "weight_lb": ["weight_lb", "weight", "wt_lb"],
    "abc_class": ["abc_class", "abc", "abc_classification"],
    "abc_shipped_cases": ["abc_shipped_cases", "shipped_cases"],
}

EXCLUSION_ALIASES: dict[str, list[str]] = {
    "item_id": ["item_id", "prtnum", "sku"],
    "reason": ["reason", "exclusion_reason", "note"],
}

PICK_HISTORY_ALIASES: dict[str, list[str]] = {
    "item_id": ["item_id", "prtnum", "sku"],
    "method_used": ["method_used", "pick_method", "method"],
    "layers_picked": ["layers_picked", "layers", "qty_layers"],
    "labor_seconds": ["labor_seconds", "seconds", "labor_sec"],
}


def resolve_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str]:
    """Map canonical field names to actual CSV column names.

    Matching ignores case, spacing and punctuation, so ``Item ID``, ``item-id``
    and ``ITEM #`` all resolve to ``item_id`` (see
    :func:`slotting.jda_columns.normalize_header`). Unmatched fields are simply
    absent from the result — callers decide which are required and report via
    :func:`require_columns`.
    """
    columns = list(df.columns)
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        hit = match_column(columns, candidates)
        if hit is not None:
            resolved[canonical] = hit
    return resolved


def require_columns(
    df: pd.DataFrame,
    resolved: dict[str, str],
    aliases: dict[str, list[str]],
    required: tuple[str, ...],
    *,
    path: object = "",
) -> None:
    """Raise with an actionable message if any *required* field is unresolved.

    The old message was ``missing order columns for ['item_id']`` — it named what
    was absent but neither what spellings are accepted nor what the file actually
    contained, so the reader had to open the source to act on it.
    """
    missing = [k for k in required if k not in resolved]
    if not missing:
        return
    prefix = f"{path}: " if path else ""
    blocks = [
        missing_column_error(name, aliases.get(name, []), list(df.columns))
        for name in missing
    ]
    raise ValueError(prefix + "\n".join(blocks))
