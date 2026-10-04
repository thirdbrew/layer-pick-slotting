"""A live WMS export must fail loudly, not load wrong.

Three loader defects, all on the path a floor export takes into the engine:

1. An export that divides a quantity by ITSELF when a SKU has no layer
   unit-of-measure makes a whole pallet silently count as exactly 1 layer.
   The export must emit a blank instead, and the loader must refuse it.
2. The loader built objects in a bare loop — a blank or non-numeric cell raised
   an unlocated `ValueError` with no file, line, or column.
3. Nothing checked key uniqueness, so a duplicated `lodnum` became two Pallets
   and double-counted floor stock.

All three are silent-wrongness failures: plausible numbers, wrong floor.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path

import pytest

from slotting.data_loader import CsvDataSource

REPO = Path(__file__).resolve().parent.parent
HEADER = "pallet_id,item_id,layers_remaining,current_slot,arrived_at\n"
GOOD = "P1,SKU1,5,A01,2026-03-03T06:00:00\n"


def _dir(tmp_path: Path, pallets_body: str) -> Path:
    (tmp_path / "pallets.csv").write_text(HEADER + pallets_body, encoding="utf-8")
    (tmp_path / "slots.csv").write_text("slot_id,slot_type,occupied_by_pallet_id\nA01,source,P1\n", encoding="utf-8")
    (tmp_path / "orders.csv").write_text(
        "order_id,item_id,qty_layers,due_at,status,area\nO1,SKU1,2,2026-03-03T12:00:00,waved,this_line\n",
        encoding="utf-8",
    )
    (tmp_path / "exclusions.csv").write_text("item_id,reason\n", encoding="utf-8")
    return tmp_path


def _load(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CsvDataSource(path).load_world_state(as_of=datetime(2026, 3, 3, 6, 0))


def test_blank_layers_remaining_is_refused_with_file_line_and_column(tmp_path: Path) -> None:
    """This is what a correct export emits when the layer UOM is missing."""
    d = _dir(tmp_path, GOOD + "P2,SKU2,,A02,2026-03-03T06:00:00\n")
    with pytest.raises(ValueError) as excinfo:
        _load(d)
    msg = str(excinfo.value)
    assert "line 3" in msg, "must locate the row"
    assert "layers_remaining" in msg, "must name the column"
    assert "blank" in msg


def test_non_numeric_layers_remaining_is_refused(tmp_path: Path) -> None:
    d = _dir(tmp_path, GOOD + "P2,SKU2,not-a-number,A02,2026-03-03T06:00:00\n")
    with pytest.raises(ValueError, match=r"line 3.*layers_remaining.*not a number"):
        _load(d)


def test_duplicate_pallet_id_is_refused_not_double_counted(tmp_path: Path) -> None:
    d = _dir(tmp_path, GOOD + "P1,SKU1,5,A02,2026-03-03T06:00:00\n")
    with pytest.raises(ValueError) as excinfo:
        _load(d)
    msg = str(excinfo.value)
    assert "duplicate pallet_id" in msg and "P1" in msg
    assert "double-count" in msg


def test_unparseable_arrived_at_is_located(tmp_path: Path) -> None:
    d = _dir(tmp_path, "P1,SKU1,5,A01,not-a-date\n")
    with pytest.raises(ValueError, match=r"line 2.*arrived_at"):
        _load(d)


def test_missing_as_of_warns_that_horizons_move(tmp_path: Path) -> None:
    d = _dir(tmp_path, GOOD)
    with pytest.warns(UserWarning, match="no --as-of given"):
        CsvDataSource(d).load_world_state()


def test_clean_export_loads_silently(tmp_path: Path) -> None:
    d = _dir(tmp_path, GOOD + "P2,SKU2,3,A02,2026-03-03T06:00:00\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state = CsvDataSource(d).load_world_state(as_of=datetime(2026, 3, 3, 6, 0))
    assert [p.layers_remaining for p in state.pallets] == [5, 3]


# The WMS export queries themselves are not part of this repository; the
# loader-level guards above cover the behaviour that matters (a missing layer
# UOM must not divide a quantity by itself).
