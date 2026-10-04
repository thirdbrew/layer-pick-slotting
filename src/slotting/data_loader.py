"""CSV-backed data loading behind a swappable DataSource interface."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd

from slotting.jda_columns import (
    EXCLUSION_ALIASES,
    ORDER_ALIASES,
    PALLET_ALIASES,
    SLOT_ALIASES,
    resolve_columns,
)
from slotting.models import (
    Exclusion,
    OrderArea,
    OrderLine,
    OrderStatus,
    Pallet,
    Slot,
    SlotType,
    WorldState,
)


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def _blank_to_none(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text == "" else text


def _row_error(path: Path, i: int, column: str, value: object, problem: str) -> ValueError:
    """Locate a bad cell. CSV line = 0-based row + 2 (header + 1-based numbering)."""
    return ValueError(
        f"{path} line {i + 2}, column '{column}': {problem} (value={value!r}). "
        f"A WMS export with this row is not loadable; fix the export or the "
        f"query that produced it rather than letting the row through."
    )


def _require_int(path: Path, i: int, column: str, value: object) -> int:
    """Parse a required integer cell, refusing blanks instead of coercing them.

    Blank `layers_remaining` used to reach `int(float(nan))` and raise an
    unlocated ValueError. Worse, an export that falls back to dividing a
    quantity by itself when a SKU has no layer unit-of-measure emits 1, so a
    whole pallet silently counts as one layer. A blank is refused by name.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise _row_error(path, i, column, value, "required value is blank")
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        raise _row_error(path, i, column, value, "required value is blank")
    try:
        return int(float(text))
    except (TypeError, ValueError) as exc:
        raise _row_error(path, i, column, value, "not a number") from exc


def _reject_duplicate_keys(path: Path, series: pd.Series, column: str) -> None:
    """A repeated key silently produced two objects and double-counted stock.

    A WMS export can emit a duplicate pallet id whenever a pallet spans rows; nothing
    downstream de-duplicates, so the floor looks like it holds twice the stock.
    """
    keys = series.astype(str).str.strip()
    dupes = keys[keys.duplicated(keep=False) & (keys != "")]
    if dupes.empty:
        return
    sample = sorted(dupes.unique())[:5]
    raise ValueError(
        f"{path}: duplicate {column} - {len(dupes)} rows across "
        f"{dupes.nunique()} repeated keys, e.g. {sample}. Each row becomes its own "
        f"object, so duplicates double-count stock. De-duplicate the export."
    )


class DataSource(ABC):
    """Abstract data access — swap CSV for DB/API without touching rules."""

    @abstractmethod
    def load_world_state(self, as_of: datetime | None = None) -> WorldState:
        """Return a validated snapshot of the operational world."""


class CsvDataSource(DataSource):
    """
    Load domain objects from CSV files in a directory.

    CSVs may use the engine's column names *or* common WMS export names
    (e.g. ``lodnum`` instead of ``pallet_id``); see ``jda_columns.py``.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def load_world_state(self, as_of: datetime | None = None) -> WorldState:
        pallets = self._load_pallets()
        slots = self._load_slots()
        orders = self._load_orders()
        exclusions = self._load_exclusions()
        if as_of is None:
            # Every horizon (24h visible, 4h waved) is measured from as_of, so a
            # wall-clock default silently reinterprets a stale export as current.
            # `slotting run --data-dir ...` takes --as-of but does not require
            # it (cli.py), which is how a morning export gets evaluated at 5pm
            # against horizons that have already passed.
            warnings.warn(
                f"{self.data_dir}: no --as-of given, defaulting to wall-clock "
                f"{datetime.now():%Y-%m-%d %H:%M}. Visible/waved horizons are measured "
                f"from this instant, so a stale export will be read as current. Pass "
                f"--as-of with the export time.",
                stacklevel=2,
            )
        reference = as_of or datetime.now()
        return WorldState(
            pallets=pallets,
            slots=slots,
            orders=orders,
            exclusions=exclusions,
            as_of=reference,
        )

    def _load_pallets(self) -> list[Pallet]:
        path = self.data_dir / "pallets.csv"
        df = pd.read_csv(path, dtype=str)
        cols = resolve_columns(df, PALLET_ALIASES)
        _reject_duplicate_keys(path, cols["pallet_id"], "pallet_id")
        records: list[Pallet] = []
        for i in range(len(df)):
            try:
                arrived_at = _parse_datetime(cols["arrived_at"].iloc[i])
            except (TypeError, ValueError) as exc:
                raise _row_error(
                    path, i, "arrived_at", cols["arrived_at"].iloc[i], "not a datetime"
                ) from exc
            records.append(
                Pallet(
                    pallet_id=str(cols["pallet_id"].iloc[i]).strip(),
                    item_id=str(cols["item_id"].iloc[i]).strip(),
                    layers_remaining=_require_int(
                        path, i, "layers_remaining", cols["layers_remaining"].iloc[i]
                    ),
                    current_slot=_blank_to_none(cols["current_slot"].iloc[i]),
                    arrived_at=arrived_at,
                )
            )
        return records

    def _load_slots(self) -> list[Slot]:
        path = self.data_dir / "slots.csv"
        df = pd.read_csv(path, dtype=str)
        cols = resolve_columns(df, SLOT_ALIASES)
        records: list[Slot] = []
        for i in range(len(df)):
            records.append(
                Slot(
                    slot_id=str(cols["slot_id"].iloc[i]).strip(),
                    slot_type=SlotType(str(cols["slot_type"].iloc[i]).strip()),
                    occupied_by_pallet_id=_blank_to_none(cols["occupied_by_pallet_id"].iloc[i]),
                )
            )
        return records

    def _load_orders(self) -> list[OrderLine]:
        path = self.data_dir / "orders.csv"
        df = pd.read_csv(path, dtype=str)
        cols = resolve_columns(df, ORDER_ALIASES, optional=("arrived_at",))
        arrived_col = cols.get("arrived_at")
        records: list[OrderLine] = []
        for i in range(len(df)):
            arrived_raw = _blank_to_none(arrived_col.iloc[i]) if arrived_col is not None else None
            records.append(
                OrderLine(
                    order_id=str(cols["order_id"].iloc[i]).strip(),
                    item_id=str(cols["item_id"].iloc[i]).strip(),
                    qty_layers=int(float(cols["qty_layers"].iloc[i])),
                    due_at=_parse_datetime(cols["due_at"].iloc[i]),
                    status=OrderStatus(str(cols["status"].iloc[i]).strip()),
                    area=OrderArea(str(cols["area"].iloc[i]).strip()),
                    arrived_at=_parse_datetime(arrived_raw) if arrived_raw else None,
                )
            )
        return records

    def _load_exclusions(self) -> list[Exclusion]:
        path = self.data_dir / "exclusions.csv"
        df = pd.read_csv(path, dtype=str)
        cols = resolve_columns(df, EXCLUSION_ALIASES)
        return [
            Exclusion(
                item_id=str(cols["item_id"].iloc[i]).strip(),
                reason=str(cols["reason"].iloc[i]).strip(),
            )
            for i in range(len(df))
        ]
