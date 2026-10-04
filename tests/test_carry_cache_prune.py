"""Carry cache prunes retired-config entries instead of growing without bound.

The cache key embeds the config hash and the src fingerprint, so every code
change rotates it — which is what makes it safe to serve, and also what made it
grow forever: nothing removed the orphans, and after a few config changes the
cache can hold gigabytes with **no current entry in it**.
"""

from __future__ import annotations

from layout_advisor.planning.daily_plan import replay_code_fingerprint, replay_config_hash
from layout_advisor.simulate.reference_config import REFERENCE_REPLAY_CONFIG
from scripts.generate_daily_wave_plans import _carry_cache_prune


def _cache(tmp_path):
    """(output_dir, cache_dir) laid out the way the script expects."""
    out = tmp_path / "plans"
    out.mkdir()
    cache = tmp_path / ".carry_cache"
    cache.mkdir()
    return out, cache


def _live_name(day: str = "2026-03-03") -> str:
    cfg_hash = replay_config_hash(REFERENCE_REPLAY_CONFIG)
    return f"{day}_{cfg_hash}_{replay_code_fingerprint()}.pkl"


def test_a_live_entry_survives(tmp_path):
    out, cache = _cache(tmp_path)
    live = cache / _live_name()
    live.write_bytes(b"x" * 100)
    _carry_cache_prune(out, REFERENCE_REPLAY_CONFIG, quiet=True)
    assert live.is_file()


def test_a_retired_config_entry_is_removed(tmp_path):
    out, cache = _cache(tmp_path)
    stale = cache / f"2026-03-03_deadbeefdeadbeef_{replay_code_fingerprint()}.pkl"
    stale.write_bytes(b"x" * 100)
    freed = _carry_cache_prune(out, REFERENCE_REPLAY_CONFIG, quiet=True)
    assert not stale.exists()
    assert freed == 100


def test_a_retired_code_fingerprint_entry_is_removed(tmp_path):
    # config_hash alone was never enough - a src change that moves behaviour
    # without touching config is exactly what the fingerprint exists to catch.
    out, cache = _cache(tmp_path)
    cfg_hash = replay_config_hash(REFERENCE_REPLAY_CONFIG)
    stale = cache / f"2026-03-03_{cfg_hash}_0000000000000000.pkl"
    stale.write_bytes(b"x" * 250)
    assert _carry_cache_prune(out, REFERENCE_REPLAY_CONFIG, quiet=True) == 250
    assert not stale.exists()


def test_it_keeps_every_day_of_the_live_chain(tmp_path):
    out, cache = _cache(tmp_path)
    days = [f"2026-03-{d:02d}" for d in range(12, 28)]
    for day in days:
        (cache / _live_name(day)).write_bytes(b"x")
    _carry_cache_prune(out, REFERENCE_REPLAY_CONFIG, quiet=True)
    assert len(list(cache.glob("*.pkl"))) == len(days)


def test_it_reports_the_bytes_it_freed(tmp_path):
    out, cache = _cache(tmp_path)
    for i in range(3):
        (cache / f"2026-03-{12 + i}_oldoldoldoldold0_{replay_code_fingerprint()}.pkl").write_bytes(
            b"x" * 1000
        )
    assert _carry_cache_prune(out, REFERENCE_REPLAY_CONFIG, quiet=True) == 3000


def test_a_missing_cache_directory_is_not_an_error(tmp_path):
    out = tmp_path / "plans"
    out.mkdir()
    assert _carry_cache_prune(out, REFERENCE_REPLAY_CONFIG, quiet=True) == 0


def test_non_pickle_files_are_left_alone(tmp_path):
    # The prune must never reach outside its own file type; a stray README or a
    # half-written temp file is not its business.
    out, cache = _cache(tmp_path)
    keep = cache / "README.txt"
    keep.write_text("not a cache entry", encoding="utf-8")
    _carry_cache_prune(out, REFERENCE_REPLAY_CONFIG, quiet=True)
    assert keep.is_file()
