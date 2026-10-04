"""Load and validate customer intake files."""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from layout_advisor.intake.columns import (
    EXCLUSION_ALIASES,
    ORDER_ALIASES,
    PICK_HISTORY_ALIASES,
    SKU_ALIASES,
    require_columns,
    resolve_columns,
)
from layout_advisor.models.customer import (
    CustomerIntake,
    ExclusionRow,
    OrderIntakeRow,
    PickHistoryRow,
    PickMethod,
    PolicyIntake,
    SkuMasterRow,
    VolumeIntake,
)


#: Fraction of rows that may be dropped silently before the loader complains.
#: A file that parses 10% of its rows must not look like a successful load —
#: that is data loss wearing a success message.
SKIP_WARN_FRACTION = 0.10


def _warn_skipped(path: Path, skipped: int, total: int, *, what: str) -> None:
    """Surface dropped rows. Loud on the common case, fatal on the absurd one."""
    if not skipped or not total:
        return
    share = skipped / total
    message = (
        f"{path}: skipped {skipped:,} of {total:,} rows ({share:.1%}), {what}."
    )
    if share >= 0.90:
        raise ValueError(
            message
            + "\n  Almost nothing parsed. This is usually a column mismatch (the "
            "right header matched, but the values are in a different column) or a "
            "file exported with the wrong sheet selected."
        )
    warnings.warn(message, stacklevel=3)


def _parse_date(value: object) -> date:
    return pd.to_datetime(value).date()


def _parse_int(value: object, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    if text == "":
        return default
    return int(float(text))


def _parse_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _parse_optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(float(text))


def _load_orders(path: Path) -> list[OrderIntakeRow]:
    df = pd.read_csv(path, dtype=str)
    cols = resolve_columns(df, ORDER_ALIASES)
    require_columns(
        df, cols, ORDER_ALIASES, ("item_id", "ship_date", "qty_layers", "order_id"), path=path
    )

    rows: list[OrderIntakeRow] = []
    skipped = 0
    for idx, row in df.iterrows():
        item_id = str(row[cols["item_id"]]).strip()
        order_id = str(row[cols["order_id"]]).strip()
        if not item_id or not order_id or item_id.lower() == "nan" or order_id.lower() == "nan":
            skipped += 1
            continue
        qty_cases = None
        if "qty_cases" in cols:
            raw = row[cols["qty_cases"]]
            if raw is not None and str(raw).strip() != "":
                qty_cases = _parse_int(raw)
        # Locate a bad cell instead of surfacing a bare pandas error. CSV line =
        # 0-based index + 2 (header row + 1-based numbering), which is what the
        # reader sees in a spreadsheet.
        try:
            ship_date = _parse_date(row[cols["ship_date"]])
            qty_layers = _parse_int(row[cols["qty_layers"]])
        except (ValueError, TypeError, pd.errors.ParserError) as exc:
            raise ValueError(
                f"{path} line {int(idx) + 2} (order_id={order_id!r}, item_id={item_id!r}): "
                f"could not parse ship_date={row[cols['ship_date']]!r} / "
                f"qty_layers={row[cols['qty_layers']]!r} -- {exc}"
            ) from exc
        rows.append(
            OrderIntakeRow(
                item_id=item_id,
                ship_date=ship_date,
                qty_layers=qty_layers,
                qty_cases=qty_cases,
                order_id=order_id,
            )
        )
    _warn_skipped(path, skipped, len(df), what="order rows missing item_id/order_id")
    return rows


def _load_policy(path: Path, defaults: PolicyIntake) -> PolicyIntake:
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return defaults
    row = df.iloc[0]
    min_col = next((c for c in df.columns if "min_layers" in c.lower()), None)
    max_col = next((c for c in df.columns if "max_layers" in c.lower()), None)
    return PolicyIntake(
        min_layers_to_layer_pick=_parse_int(row[min_col], defaults.min_layers_to_layer_pick)
        if min_col
        else defaults.min_layers_to_layer_pick,
        max_layers_hand_pick=_parse_int(row[max_col], defaults.max_layers_hand_pick)
        if max_col
        else defaults.max_layers_hand_pick,
    )


def _load_skus(path: Path) -> list[SkuMasterRow]:
    df = pd.read_csv(path, dtype=str)
    cols = resolve_columns(df, SKU_ALIASES)
    require_columns(df, cols, SKU_ALIASES, ("item_id",), path=path)
    rows: list[SkuMasterRow] = []
    skipped = 0
    for _, row in df.iterrows():
        item_id = str(row[cols["item_id"]]).strip()
        if not item_id or item_id.lower() == "nan":
            skipped += 1
            continue
        rows.append(
            SkuMasterRow(
                item_id=item_id,
                layers_per_pallet=_parse_optional_int(row[cols["layers_per_pallet"]])
                if "layers_per_pallet" in cols
                else None,
                cases_per_layer=_parse_optional_int(row[cols["cases_per_layer"]])
                if "cases_per_layer" in cols
                else None,
                length_in=_parse_float(row[cols["length_in"]]) if "length_in" in cols else None,
                width_in=_parse_float(row[cols["width_in"]]) if "width_in" in cols else None,
                height_in=_parse_float(row[cols["height_in"]]) if "height_in" in cols else None,
                weight_lb=_parse_float(row[cols["weight_lb"]]) if "weight_lb" in cols else None,
                abc_class=str(row[cols["abc_class"]]).strip().upper()
                if "abc_class" in cols and pd.notna(row.get(cols["abc_class"]))
                else None,
                abc_shipped_cases=_parse_optional_int(row[cols["abc_shipped_cases"]])
                if "abc_shipped_cases" in cols
                else None,
            )
        )
    _warn_skipped(path, skipped, len(df), what="SKU rows missing item_id")
    return rows


def _load_exclusions(path: Path) -> list[ExclusionRow]:
    df = pd.read_csv(path, dtype=str)
    cols = resolve_columns(df, EXCLUSION_ALIASES)
    require_columns(df, cols, EXCLUSION_ALIASES, ("item_id",), path=path)
    rows: list[ExclusionRow] = []
    skipped = 0
    for _, row in df.iterrows():
        item_id = str(row[cols["item_id"]]).strip()
        if not item_id or item_id.lower() == "nan":
            skipped += 1
            continue
        reason = str(row[cols["reason"]]).strip() if "reason" in cols else "excluded"
        rows.append(ExclusionRow(item_id=item_id, reason=reason))
    _warn_skipped(path, skipped, len(df), what="exclusion rows missing item_id")
    return rows


def _load_pick_history(path: Path) -> list[PickHistoryRow]:
    df = pd.read_csv(path, dtype=str)
    cols = resolve_columns(df, PICK_HISTORY_ALIASES)
    require_columns(
        df, cols, PICK_HISTORY_ALIASES, ("item_id", "method_used", "layers_picked"), path=path
    )
    rows: list[PickHistoryRow] = []
    skipped = 0
    for idx, row in df.iterrows():
        item_id = str(row[cols["item_id"]]).strip()
        if not item_id or item_id.lower() == "nan":
            skipped += 1
            continue
        method_raw = str(row[cols["method_used"]]).strip().lower()
        try:
            method = PickMethod(method_raw)
        except ValueError as exc:
            raise ValueError(
                f"{path} line {int(idx) + 2} (item_id={item_id!r}): "
                f"method_used={method_raw!r} is not a known pick method. "
                f"Accepted: {', '.join(m.value for m in PickMethod)}."
            ) from exc
        rows.append(
            PickHistoryRow(
                item_id=item_id,
                method_used=method,
                layers_picked=_parse_int(row[cols["layers_picked"]]),
                labor_seconds=_parse_float(row[cols["labor_seconds"]])
                if "labor_seconds" in cols
                else None,
            )
        )
    _warn_skipped(path, skipped, len(df), what="pick-history rows missing item_id")
    return rows


def _load_volume(path: Path) -> VolumeIntake:
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return VolumeIntake()
    row = df.iloc[0]
    col_map = {str(c).strip().lower(): c for c in df.columns}

    def _col(*names: str) -> str | None:
        for name in names:
            if name in col_map:
                return col_map[name]
        return None

    facility_col = _col("facility", "facility_name", "site")
    start_col = _col("date_range_start", "start_date", "from_date")
    end_col = _col("date_range_end", "end_date", "to_date")
    peak_col = _col("peak_days_note", "peak_note", "peak_days")

    return VolumeIntake(
        facility=str(row[facility_col]).strip() if facility_col else None,
        date_range_start=_parse_date(row[start_col]) if start_col and str(row[start_col]).strip() else None,
        date_range_end=_parse_date(row[end_col]) if end_col and str(row[end_col]).strip() else None,
        peak_days_note=str(row[peak_col]).strip() if peak_col and str(row[peak_col]).strip() else None,
    )


def load_customer_intake(
    customer_dir: Path,
    *,
    default_policy: PolicyIntake | None = None,
) -> CustomerIntake:
    """Load all customer intake files from *customer_dir*."""
    customer_dir = Path(customer_dir)
    policy_defaults = default_policy or PolicyIntake()

    orders_path = customer_dir / "orders.csv"
    if not orders_path.exists():
        raise FileNotFoundError(f"Customer intake requires orders.csv in {customer_dir}")

    orders = _load_orders(orders_path)
    policy = policy_defaults
    policy_path = customer_dir / "policy.csv"
    if policy_path.exists():
        policy = _load_policy(policy_path, policy_defaults)

    skus: list[SkuMasterRow] = []
    skus_path = customer_dir / "skus.csv"
    if skus_path.exists():
        skus = _load_skus(skus_path)

    exclusions: list[ExclusionRow] = []
    exclusions_path = customer_dir / "exclusions.csv"
    if exclusions_path.exists():
        exclusions = _load_exclusions(exclusions_path)

    pick_history: list[PickHistoryRow] = []
    picks_path = customer_dir / "pick_history.csv"
    if picks_path.exists():
        pick_history = _load_pick_history(picks_path)

    volume: VolumeIntake | None = None
    volume_path = customer_dir / "volume.csv"
    if volume_path.exists():
        volume = _load_volume(volume_path)

    site_id: str | None = None
    facility_name: str | None = None
    meta_path = customer_dir / "site.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        site_id = meta.get("site_id")
        facility_name = meta.get("facility_name")
        operator = meta.get("operator")
        contracted_customer = meta.get("contracted_customer") or meta.get("end_customer")
    else:
        operator = None
        contracted_customer = None
    if volume and volume.facility and not facility_name:
        facility_name = volume.facility

    return CustomerIntake(
        site_id=site_id,
        facility_name=facility_name,
        operator=operator,
        contracted_customer=contracted_customer,
        orders=orders,
        pick_history=pick_history,
        policy=policy,
        skus=skus,
        exclusions=exclusions,
        volume=volume,
    )
