"""Profiles + kill switches: the hard limits must be enforceable, the M3 Pro hard cap must hold
even against a generous budget, and clamps must never exceed the cap."""

import pytest

from mop.studio.profiles import M3PRO_LOCAL_MAX, STUDIO, get_profile, list_profiles


def test_get_profile_by_name_and_alias():
    assert get_profile("studio-1tb") is STUDIO
    assert get_profile("m3pro-local-max") is M3PRO_LOCAL_MAX
    # aliases resolve to the same object
    assert get_profile("studio") is STUDIO
    assert get_profile("local-max") is M3PRO_LOCAL_MAX
    assert get_profile("M3PRO") is M3PRO_LOCAL_MAX  # case-insensitive


def test_get_profile_unknown_raises():
    with pytest.raises(ValueError):
        get_profile("does-not-exist")


def test_usable_is_total_minus_reserve():
    assert STUDIO.usable_gb == STUDIO.disk_total_gb - STUDIO.reserve_gb
    assert M3PRO_LOCAL_MAX.usable_gb == M3PRO_LOCAL_MAX.disk_total_gb - M3PRO_LOCAL_MAX.reserve_gb


def test_within_budget_enforces_hard_cap():
    # the m3pro hard cap is 25 GB: a request above it is rejected REGARDLESS of any budget argument
    ok, _ = M3PRO_LOCAL_MAX.within_budget(10.0)
    assert ok
    ok, msg = M3PRO_LOCAL_MAX.within_budget(50.0)
    assert not ok and "hard cap" in msg


def test_effective_budget_clamps_to_hard_cap():
    # a generous --budget-gb cannot push a real download past the hard cap (the kill switch)
    assert M3PRO_LOCAL_MAX.effective_budget_gb(1000.0) == M3PRO_LOCAL_MAX.download_hard_cap_gb
    # None falls back to the profile default budget
    assert M3PRO_LOCAL_MAX.effective_budget_gb(None) == M3PRO_LOCAL_MAX.download_budget_gb
    # a modest request passes through unchanged
    assert M3PRO_LOCAL_MAX.effective_budget_gb(5.0) == 5.0


def test_effective_budget_rejects_negative():
    # a negative --budget-gb is a bad input, not a quiet no-op (mirrors within_budget)
    with pytest.raises(ValueError):
        M3PRO_LOCAL_MAX.effective_budget_gb(-5.0)


def test_clamp_clips_and_runs_never_exceed_caps():
    assert M3PRO_LOCAL_MAX.clamp_clips(10_000) == M3PRO_LOCAL_MAX.max_cache_clips
    assert M3PRO_LOCAL_MAX.clamp_clips(16) == 16
    assert M3PRO_LOCAL_MAX.clamp_clips(-5) == 0
    assert M3PRO_LOCAL_MAX.clamp_runs(10_000) == M3PRO_LOCAL_MAX.max_run_count


def test_tier_allowed_matches_profile():
    assert M3PRO_LOCAL_MAX.tier_allowed("C")
    assert not M3PRO_LOCAL_MAX.tier_allowed("E")  # no env tier on the laptop
    assert not M3PRO_LOCAL_MAX.tier_allowed("R")
    assert STUDIO.tier_allowed("E")
    assert not STUDIO.tier_allowed("R")  # rented CUDA stays opt-in even on the Studio


def test_m3pro_is_the_documented_envelope():
    p = M3PRO_LOCAL_MAX
    assert p.download_budget_gb == 10.0
    assert p.download_hard_cap_gb == 25.0
    assert p.fixture_budget_gb == 2.0
    assert p.max_cache_clips == 128
    assert p.min_free_disk_gb == 60.0
    assert p.max_wall_min == 90
    assert not p.allow_manual_auth  # laptop never auto-selects signed-terms sources


def test_studio_is_900gb_usable():
    assert STUDIO.usable_gb == 900.0
    assert STUDIO.allow_manual_auth  # once the user has signed access on the Studio


def test_free_disk_ok_returns_bool_and_value(tmp_path):
    ok, free_gb = M3PRO_LOCAL_MAX.free_disk_ok(tmp_path)
    assert isinstance(ok, bool)
    assert free_gb > 0


def test_list_profiles_includes_both():
    names = {p["name"] for p in list_profiles()}
    assert names == {"studio-1tb", "m3pro-local-max"}
    # the serialized envelope carries the caps a manifest needs
    studio = next(p for p in list_profiles() if p["name"] == "studio-1tb")
    assert studio["usable_gb"] == 900.0
    assert "allowed_tiers" in studio and "min_free_disk_gb" in studio
