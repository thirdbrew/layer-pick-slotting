"""Published orders-extract template: validate and normalize to canonical orders CSV.

Template columns (see templates/design_intake/orders_template.csv):
  required: sku, order_id, due_ts
  one of:   layers  OR  cases (+ cases_per_layer column or default arg)
  optional: ship_date (defaults to due_ts date)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

MIN_SPAN_DAYS = 28
MIN_DISTINCT_SKUS = 20
TEMPLATE_REQUIRED = ("sku", "order_id", "due_ts")
CANONICAL_COLUMNS = ("order_id", "item_id", "qty_layers", "due_at", "status", "area")


@dataclass(frozen=True)
class OrdersValidationError:
    column: str | None
    message: str


@dataclass
class OrdersTemplateResult:
    frame: pd.DataFrame
    errors: list[OrdersValidationError] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _err(column: str | None, message: str) -> OrdersValidationError:
    return OrdersValidationError(column=column, message=message)


def load_orders_template(
    path: Path, *, default_cases_per_layer: float | None = None
) -> OrdersTemplateResult:
    errors: list[OrdersValidationError] = []
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # unreadable file is one clear error, not a crash
        return OrdersTemplateResult(pd.DataFrame(), [_err(None, f"Could not read CSV: {exc}")])

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    # Case-variant duplicates (e.g. 'SKU' + 'sku') collapse to one name here and
    # would make raw[col] a DataFrame downstream -> AttributeError -> HTTP 500.
    # Fail cleanly instead.
    cols = list(raw.columns)
    dupes = sorted({c for c in cols if cols.count(c) > 1})
    if dupes:
        return OrdersTemplateResult(pd.DataFrame(), [
            _err(None, f"Duplicate/case-variant column(s) after normalizing headers: "
                       f"{', '.join(dupes)}. Rename so each maps to a distinct name.")
        ])
    found = ", ".join(raw.columns) or "(none)"
    for col in TEMPLATE_REQUIRED:
        if col not in raw.columns:
            errors.append(_err(col, f"Required column '{col}' missing; found columns: {found}"))
    has_layers = "layers" in raw.columns
    has_cases = "cases" in raw.columns
    if not has_layers and not has_cases:
        errors.append(_err("layers", f"Need 'layers' or 'cases' column; found columns: {found}"))
    if errors:
        return OrdersTemplateResult(pd.DataFrame(), errors)

    if raw.empty:
        return OrdersTemplateResult(raw, [_err(None, "File has headers but zero data rows")])

    due = pd.to_datetime(raw["due_ts"], errors="coerce")
    bad_due = int(due.isna().sum())
    if bad_due:
        errors.append(_err("due_ts", f"{bad_due} rows have unparseable due_ts"))

    # trust boundary: blank sku/order_id must not silently become the literal
    # string "nan" in the canonical CSV downstream.
    for col in ("sku", "order_id"):
        blank = raw[col].isna() | (raw[col].astype(str).str.strip() == "")
        bad_blank = int(blank.sum())
        if bad_blank:
            errors.append(_err(col, f"{bad_blank} rows have blank/missing {col}"))

    if has_layers:
        layers = pd.to_numeric(raw["layers"], errors="coerce")
        bad = int(layers.isna().sum())
        if bad:
            errors.append(_err("layers", f"{bad} rows have non-numeric layers"))
        nonpos = int((layers.notna() & (layers <= 0)).sum())
        if nonpos:
            errors.append(_err("layers", f"{nonpos} rows have non-positive layers"))
    else:
        cases = pd.to_numeric(raw["cases"], errors="coerce")
        bad = int(cases.isna().sum())
        if bad:
            errors.append(_err("cases", f"{bad} rows have non-numeric cases"))
        if "cases_per_layer" in raw.columns:
            cpl = pd.to_numeric(raw["cases_per_layer"], errors="coerce")
            bad_cpl = int((cpl.isna() | (cpl <= 0)).sum())
            if bad_cpl:
                errors.append(
                    _err(
                        "cases_per_layer",
                        f"{bad_cpl} rows have non-numeric or non-positive cases_per_layer",
                    )
                )
                cpl = None
        elif default_cases_per_layer is not None and default_cases_per_layer > 0:
            cpl = pd.Series(default_cases_per_layer, index=raw.index)
        else:
            errors.append(
                _err(
                    "cases_per_layer",
                    "Extract has 'cases' but no 'cases_per_layer' column and no "
                    "intake-form default cases-per-layer was provided",
                )
            )
            cpl = None
        if cpl is not None and not errors:
            layers = (cases / cpl).round().clip(lower=1)

    if errors:
        return OrdersTemplateResult(pd.DataFrame(), errors)

    span_days = int((due.max().normalize() - due.min().normalize()).days) + 1
    distinct_skus = int(raw["sku"].astype(str).nunique())
    if span_days < MIN_SPAN_DAYS:
        errors.append(
            _err(None, f"Extract spans {span_days} days; minimum {MIN_SPAN_DAYS} required for the quantile fit")
        )
    if distinct_skus < MIN_DISTINCT_SKUS:
        errors.append(
            _err(None, f"Extract has {distinct_skus} distinct SKUs; minimum {MIN_DISTINCT_SKUS} required")
        )
    if errors:
        return OrdersTemplateResult(pd.DataFrame(), errors)

    frame = pd.DataFrame(
        {
            "order_id": raw["order_id"].astype(str),
            "item_id": raw["sku"].astype(str),
            # clip matches the cases-path convention: any positive quantity
            # is at least one layer (0.4 must not round to a zero-qty line)
            "qty_layers": layers.round().clip(lower=1).astype(int),
            "due_at": due.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "visible",
            "area": "this_line",
        }
    )
    stats = {
        "rows_in": int(len(raw)),
        "rows_out": int(len(frame)),
        "date_span_days": span_days,
        "distinct_skus": distinct_skus,
        "ship_days": int(due.dt.normalize().nunique()),
        "first_due": due.min().strftime("%Y-%m-%d"),
        "last_due": due.max().strftime("%Y-%m-%d"),
        "total_layers": int(frame["qty_layers"].sum()),
    }
    return OrdersTemplateResult(frame, [], stats)


def write_canonical_orders(result: OrdersTemplateResult, out_path: Path) -> None:
    if not result.ok:
        raise ValueError("Refusing to write canonical orders from a failed validation")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(out_path, index=False, columns=list(CANONICAL_COLUMNS))
