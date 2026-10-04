"""Tests for ABC setup ranking."""

from __future__ import annotations

from slotting.abc_ranking import abc_rank_key, load_abc_by_item


def test_abc_rank_key_order():
    assert abc_rank_key("A") < abc_rank_key("B") < abc_rank_key("C")
    assert abc_rank_key(None) > abc_rank_key("C")


def test_load_abc_from_site_skus():
    abc = load_abc_by_item(reload=True)
    if not abc:
        return
    assert all(v in {"A", "B", "C"} for v in abc.values())
    assert len(abc) >= 100
