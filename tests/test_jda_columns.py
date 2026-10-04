"""Tests that JDA-style WMS export column names load correctly."""

from __future__ import annotations

from pathlib import Path

from slotting.data_loader import CsvDataSource

JDA_DIR = Path(__file__).resolve().parent.parent / "data" / "jda-headers"


def test_load_jda_header_csvs() -> None:
    state = CsvDataSource(JDA_DIR).load_world_state()
    assert len(state.pallets) == 1
    assert state.pallets[0].pallet_id == "LPN-TEST-001"
    assert state.pallets[0].item_id == "900002"
    assert state.pallets[0].current_slot == "CLMP-01"

    assert len(state.source_slots) == 2
    assert state.orders[0].order_id == "ORD-000001-1"
    assert state.orders[0].status.value == "waved"
    assert state.exclusions[0].item_id == "900001"
