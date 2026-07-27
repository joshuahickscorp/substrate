from types import SimpleNamespace

import pytest

from mop.evidence import canonical_sha256
from mop.studio.profiles import PROFILES, get_profile, list_profiles


@pytest.mark.parametrize(
    "alias,target",
    [
        ("studio-1tb", "studio-1tb"),
        ("studio", "studio-1tb"),
        ("studio-m1ultra", "studio-m1ultra"),
        ("m1ultra", "studio-m1ultra"),
        ("8tb", "studio-m1ultra"),
        ("m3pro-local-max", "m3pro-local-max"),
        ("M3PRO", "m3pro-local-max"),
        ("local-max", "m3pro-local-max"),
    ],
)
def test_profile_aliases_resolve_to_singular_records(alias, target):
    assert get_profile(alias) is PROFILES[target]


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="unknown profile"):
        get_profile("does-not-exist")


def test_profile_projection_is_byte_exact():
    assert canonical_sha256(list_profiles()) == (
        "d602bfa3db7b337868139b0d01092710a63c3306110444c6855c43b6265d4067"
    )
    assert [row["name"] for row in list_profiles()] == ["studio-m1ultra", "studio-1tb", "m3pro-local-max"]


def test_profile_scenarios_retain_honest_scope():
    m3, studio, ultra = (PROFILES[name] for name in ("m3pro-local-max", "studio-1tb", "studio-m1ultra"))
    assert m3.min_free_disk_gb == m3.reserve_gb + m3.download_hard_cap_gb + 5.0 == 40.0
    assert not m3.allow_manual_auth and m3.procurement_status == "measured-current-host-envelope"
    assert studio.usable_gb == 900.0 and studio.allow_manual_auth
    assert ultra.usable_gb == 7200.0 and ultra.allowed_tiers == frozenset({"C", "E"})
    assert studio.procurement_status == ultra.procurement_status == "unverified-procurement-scenario"


def test_host_compatibility_uses_resources_not_chip_name(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mop.studio.profiles.shutil.disk_usage", lambda _root: SimpleNamespace(total=500 * 1e9)
    )
    host = {"is_apple_silicon": True, "chip": "Apple Future", "unified_memory_gb": 128.0}
    ok, problems, measured = PROFILES["studio-1tb"].host_compatibility(host=host, disk_root=tmp_path)
    assert not ok and any("disk" in problem for problem in problems)
    assert measured["chip"] == "Apple Future"
    host["unified_memory_gb"] = 8.0
    assert not PROFILES["m3pro-local-max"].host_compatibility(host=host, disk_root=tmp_path)[0]


def test_free_disk_probe_returns_measured_value(tmp_path):
    ok, free_gb = PROFILES["m3pro-local-max"].free_disk_ok(tmp_path)
    assert isinstance(ok, bool) and free_gb > 0
