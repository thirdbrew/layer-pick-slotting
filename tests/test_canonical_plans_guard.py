"""The canonical plans directory is the chain benchmark in file form. Guard it.

`data/plans/` is what the chain tooling reads. Two ways to destroy it — the
second found by tripping it while testing the first:

1. `--no-reference` builds a bare ReplayConfig (defaults to the legacy 75:25)
   and `--output-dir` defaults *here*.
2. A **reference-config** lone `--date X` run starts from a cold floor, while
   the stored artifacts are carry-warm chain output. It rewrites a whole ship
   day under an *identical config hash*, so a hash-based freshness check would
   stay green on a plan that no longer matched its own benchmark.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG
from scripts.generate_daily_wave_plans import _guard_canonical_plans


def test_a_scratch_directory_is_never_guarded(tmp_path):
    # The escape hatch has to work, or people --force everything out of habit.
    bare = replace(REFERENCE_REPLAY_CONFIG, source_slots=75, donor_slots=25, setup_source_slots=75)
    _guard_canonical_plans(tmp_path, bare, force=False, chain_run=False)


def test_force_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.generate_daily_wave_plans.DEFAULT_OUT", tmp_path)
    bare = replace(REFERENCE_REPLAY_CONFIG, source_slots=75, donor_slots=25, setup_source_slots=75)
    _guard_canonical_plans(tmp_path, bare, force=True, chain_run=False)


def test_a_non_reference_config_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.generate_daily_wave_plans.DEFAULT_OUT", tmp_path)
    bare = replace(REFERENCE_REPLAY_CONFIG, source_slots=75, donor_slots=25, setup_source_slots=75)
    with pytest.raises(SystemExit, match="non-reference artifacts"):
        _guard_canonical_plans(tmp_path, bare, force=False, chain_run=True)


def test_a_cold_run_over_carry_warm_plans_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.generate_daily_wave_plans.DEFAULT_OUT", tmp_path)
    day = tmp_path / "2026-03-03"
    day.mkdir()
    (day / "daily_summary.json").write_text(json.dumps({"carry_warm": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="carry-warm"):
        _guard_canonical_plans(tmp_path, REFERENCE_REPLAY_CONFIG, force=False, chain_run=False)


def test_a_chain_run_may_rewrite_carry_warm_plans(tmp_path, monkeypatch):
    # A chain regen reproduces the carry-warm state, so it is the one thing
    # allowed to replace it.
    monkeypatch.setattr("scripts.generate_daily_wave_plans.DEFAULT_OUT", tmp_path)
    day = tmp_path / "2026-03-03"
    day.mkdir()
    (day / "daily_summary.json").write_text(json.dumps({"carry_warm": True}), encoding="utf-8")
    _guard_canonical_plans(tmp_path, REFERENCE_REPLAY_CONFIG, force=False, chain_run=True)


def test_cold_plans_may_be_rewritten_by_a_cold_run(tmp_path, monkeypatch):
    # Nothing carry-warm to protect: a fresh-vs-fresh overwrite is like-for-like.
    monkeypatch.setattr("scripts.generate_daily_wave_plans.DEFAULT_OUT", tmp_path)
    day = tmp_path / "2026-03-03"
    day.mkdir()
    (day / "daily_summary.json").write_text(json.dumps({"carry_warm": False}), encoding="utf-8")
    _guard_canonical_plans(tmp_path, REFERENCE_REPLAY_CONFIG, force=False, chain_run=False)


def test_an_empty_plans_directory_is_not_guarded(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.generate_daily_wave_plans.DEFAULT_OUT", tmp_path)
    _guard_canonical_plans(tmp_path, REFERENCE_REPLAY_CONFIG, force=False, chain_run=False)


def test_the_real_committed_plans_are_carry_warm():
    """If this ever fails the guard above has quietly stopped protecting anything."""
    from scripts.generate_daily_wave_plans import DEFAULT_OUT

    summaries = sorted(DEFAULT_OUT.glob("*/daily_summary.json"))
    if not summaries:
        pytest.skip("plan artifacts not present")
    warm = [
        p for p in summaries
        if json.loads(p.read_text(encoding="utf-8")).get("carry_warm") is True
    ]
    assert warm, "no committed plan is carry_warm - the cold-run guard is inert"
