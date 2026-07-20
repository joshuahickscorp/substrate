from types import SimpleNamespace

import pytest

from mop.studio.profiles import (
    M1_ULTRA,
    M3PRO_LOCAL_MAX,
    M3PRO_LOCAL_MIN_FREE_DISK_GB,
    STUDIO,
    get_profile,
    list_profiles,
)


def test_get_profile_by_name_and_alias():
    assert get_profile("studio-1tb") is STUDIO
    assert get_profile("m3pro-local-max") is M3PRO_LOCAL_MAX
    assert get_profile("studio-m1ultra") is M1_ULTRA
    assert get_profile("studio") is STUDIO
    assert get_profile("local-max") is M3PRO_LOCAL_MAX
    assert get_profile("M3PRO") is M3PRO_LOCAL_MAX  # case-insensitive
    assert get_profile("m1ultra") is M1_ULTRA
    assert get_profile("8tb") is M1_ULTRA


def test_get_profile_unknown_raises():
    with pytest.raises(ValueError):
        get_profile("does-not-exist")


def test_usable_is_total_minus_reserve():
    assert STUDIO.usable_gb == STUDIO.disk_total_gb - STUDIO.reserve_gb
    assert M3PRO_LOCAL_MAX.usable_gb == M3PRO_LOCAL_MAX.disk_total_gb - M3PRO_LOCAL_MAX.reserve_gb


def test_m3pro_is_the_documented_envelope():
    p = M3PRO_LOCAL_MAX
    assert p.download_budget_gb == 10.0
    assert p.download_hard_cap_gb == 25.0
    assert p.fixture_budget_gb == 2.0
    assert p.max_cache_clips == 128
    assert p.min_free_disk_gb == 40.0
    assert p.min_free_disk_gb == M3PRO_LOCAL_MIN_FREE_DISK_GB
    assert p.min_free_disk_gb == p.reserve_gb + p.download_hard_cap_gb + 5.0
    assert p.max_wall_min == 300
    assert not p.allow_manual_auth  # laptop never auto-selects signed-terms sources


def test_studio_is_900gb_usable():
    assert STUDIO.usable_gb == 900.0
    assert STUDIO.allow_manual_auth  # once the user has signed access on the Studio


def test_m1ultra_is_the_legacy_8tb_scenario_envelope():
    p = M1_ULTRA
    assert p.usable_gb == 7200.0
    assert p.download_hard_cap_gb == 6000.0
    assert p.min_free_disk_gb == 250.0
    assert p.max_wall_min == 60 * 24 * 7  # a week of unattended gated queue
    assert p.allowed_tiers == frozenset({"C", "E"})
    assert p.allow_manual_auth
    assert p.dry_run_default  # heavy actions still default to dry-run, even at 8 TB
    assert p.procurement_status == "unverified-procurement-scenario"
    assert p.min_host_unified_memory_gb == 120.0
    assert p.min_host_disk_gb == 7000.0


def test_profile_host_compatibility_uses_resources_not_chip_name(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mop.studio.profiles.shutil.disk_usage",
        lambda root: SimpleNamespace(total=500 * 1e9),
    )
    host = {
        "is_apple_silicon": True,
        "chip": "Apple Future",
        "unified_memory_gb": 128.0,
    }
    ok, problems, measured = STUDIO.host_compatibility(host=host, disk_root=tmp_path)
    assert not ok
    assert any("disk" in problem for problem in problems)
    assert measured["chip"] == "Apple Future"


def test_m3_host_contract_rejects_tiny_synthetic_host(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mop.studio.profiles.shutil.disk_usage",
        lambda root: SimpleNamespace(total=500 * 1e9),
    )
    host = {"is_apple_silicon": True, "chip": "x", "unified_memory_gb": 8.0}
    ok, problems, _ = M3PRO_LOCAL_MAX.host_compatibility(host=host, disk_root=tmp_path)
    assert not ok
    assert any("unified memory" in problem for problem in problems)


def test_free_disk_ok_returns_bool_and_value(tmp_path):
    ok, free_gb = M3PRO_LOCAL_MAX.free_disk_ok(tmp_path)
    assert isinstance(ok, bool)
    assert free_gb > 0


def test_list_profiles_includes_all():
    names = {p["name"] for p in list_profiles()}
    assert names == {"studio-1tb", "m3pro-local-max", "studio-m1ultra"}
    studio = next(p for p in list_profiles() if p["name"] == "studio-1tb")
    assert studio["usable_gb"] == 900.0
    assert "allowed_tiers" in studio and "min_free_disk_gb" in studio
    assert studio["procurement_status"] == "unverified-procurement-scenario"
