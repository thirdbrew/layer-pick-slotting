"""Clairvoyant LP ceilings — every expectation derivable by hand.

These bounds are what tells the project when to STOP tuning rules, so a wrong
bound is worse than no bound: it would either license endless tuning or end it
early.
"""

from __future__ import annotations

import pytest

from scripts.lp_upper_bound import covered_layers, day_bounds, top_k_set


def test_top_k_takes_the_highest_volume_skus():
    assert top_k_set({"A": 10, "B": 5, "C": 1}, 2) == {"A", "B"}


def test_top_k_breaks_ties_deterministically():
    # Same layers: item_id ascending, so two runs cannot disagree.
    assert top_k_set({"B": 5, "A": 5, "C": 5}, 2) == {"A", "B"}


def test_top_k_larger_than_the_demand_takes_everything():
    assert top_k_set({"A": 3, "B": 2}, 70) == {"A", "B"}


def test_covered_layers_counts_only_slotted_skus():
    assert covered_layers({"A": 10, "B": 5}, {"A"}) == 10


def test_b_wave_is_one_hundred_when_every_wave_fits_on_the_faces():
    # 2 faces, and no wave ever demands more than 2 SKUs.
    waves = [{"A": 10, "B": 5}, {"C": 7, "D": 3}]
    out = day_bounds(waves, faces=2)
    assert out["b_wave_pct"] == 100.0


def test_b_day_is_worse_than_b_wave_when_the_mix_churns():
    # One fixed pair can hold only 15 of 25 layers; re-choosing per wave holds all.
    waves = [{"A": 10, "B": 5}, {"C": 7, "D": 3}]
    out = day_bounds(waves, faces=2)
    assert out["b_day_pct"] == pytest.approx(100.0 * 17 / 25, abs=0.05)
    assert out["b_wave_pct"] > out["b_day_pct"]


def test_b_day_equals_b_wave_when_the_mix_is_stable():
    # Same two SKUs every wave: nothing to gain from moving.
    waves = [{"A": 10, "B": 5}, {"A": 8, "B": 4}]
    out = day_bounds(waves, faces=2)
    assert out["b_day_pct"] == out["b_wave_pct"] == 100.0
    assert out["moves_total"] == 2  # staged once at the first wave, never again


def test_moves_count_the_initial_stage():
    # The first wave has no predecessor, so its whole face set is charged. That
    # is the complete accounting: those pallets do have to be staged at setup.
    waves = [{"A": 10, "B": 5}]
    assert day_bounds(waves, faces=2)["moves_total"] == 2


def test_moves_count_only_what_changes_between_waves():
    waves = [{"A": 10, "B": 5}, {"A": 9, "C": 4}]
    out = day_bounds(waves, faces=2)
    assert out["moves_total"] == 3  # A,B staged, then C swaps in for B
    assert out["moves_peak"] == 2


def test_a_tight_face_count_caps_both_bounds():
    waves = [{"A": 10, "B": 5}, {"C": 7, "D": 3}]
    out = day_bounds(waves, faces=1)
    # Per wave the best single face holds 10 of 15, then 7 of 10 -> 17 of 25.
    assert out["b_wave_pct"] == pytest.approx(100.0 * 17 / 25, abs=0.05)
    assert out["b_day_pct"] <= out["b_wave_pct"]


def test_peak_wave_skus_is_the_faces_needed_for_full_coverage():
    waves = [{"A": 1, "B": 1, "C": 1}, {"A": 1}]
    assert day_bounds(waves, faces=70)["max_wave_skus"] == 3


def test_an_empty_day_reports_nothing_rather_than_dividing_by_zero():
    out = day_bounds([], faces=70)
    assert out["total_layers"] == 0
    assert out["b_wave_pct"] is None


def test_day_skus_is_the_union_not_the_per_wave_max():
    waves = [{"A": 1, "B": 1}, {"C": 1}]
    out = day_bounds(waves, faces=70)
    assert out["day_skus"] == 3
    assert out["max_wave_skus"] == 2
