
import json
from pathlib import Path

import pytest

from mop.provenance import RESULT_TAGS
from mop.studio import pipeline


def _skip_if_disk_below_floor():
    profile = pipeline.get_profile("m3pro-local-max")
    ok, free_gb = profile.free_disk_ok()
    if not ok:
        pytest.skip(
            f"free-disk kill switch active on this machine ({free_gb:.0f} GB free, floor "
            f"{profile.min_free_disk_gb:.0f} GB): cache/local-max execute paths refuse by design"
        )


def test_plan_writes_artifacts_and_latest(tmp_path):
    out = pipeline.cmd_plan("studio-1tb", budget_gb=900, label="t", out_root=tmp_path)
    run_dir = Path(out["run_dir"])
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "plan.md").exists()
    assert (run_dir / "license_ledger.md").exists()
    assert (run_dir / "datacards").is_dir()
    resolved = pipeline._resolve_latest(tmp_path / "latest" / "plan.json")
    assert resolved.exists()
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["provenance"]["result_tag"] in RESULT_TAGS
    assert plan["totals"]["selected_sources"] >= 1


def test_acquire_dry_run_then_validate_then_cache_estimate(tmp_path):
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=10, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"

    acq = pipeline.cmd_acquire(plan_path, execute=False)
    assert acq["manifest"]["mode"] == "dry-run"
    assert acq["manifest"]["totals"]["bytes_spent"] == 0

    val = pipeline.cmd_validate(plan_path)
    assert all(r["validation"].get("ok") in (None, True, False) for r in val["results"])

    cache = pipeline.cmd_cache(plan_path, execute=False)
    assert any("estimate" in r for r in cache["results"])
    assert all(r.get("cache", {}).get("ok") in (None,) for r in cache["results"] if "cache" in r)


def test_acquire_execute_generates_controls_then_caches(tmp_path):
    _skip_if_disk_below_floor()
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=10, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"
    acq = pipeline.cmd_acquire(plan_path, execute=True, accept_license=False)
    by = {s["slug"]: s for s in acq["manifest"]["sources"]}
    assert by["synthetic_controls"]["status"] == "complete"
    assert by["synthetic_controls"]["bytes"] > 0
    assert by.get("epic_kitchens_subset", {}).get("status") in ("blocked", "skipped-budget", "needs-license")

    cache = pipeline.cmd_cache(plan_path, execute=True)
    syn = next(r for r in cache["results"] if r["slug"] == "synthetic_controls")
    assert syn["cache"]["ok"] is True
    assert syn["cache"]["count"] > 0
    assert syn["cache"]["backend"] == "frozen_random"


def test_gated_conveyor_stops_on_disallowed_tier(tmp_path):
    out = pipeline.cmd_run(
        gated=True, tiers={"E"}, full=False, profile_name="m3pro-local-max", label="t", out_root=tmp_path
    )
    assert out["ran"] is False
    assert any(g["name"] == "tier_allowed" and not g["ok"] for g in out["gates"])
    assert "reason" in out


def test_gated_conveyor_stops_full_over_run_cap(tmp_path):
    out = pipeline.cmd_run(
        gated=True, tiers={"C"}, full=True, profile_name="m3pro-local-max", label="t2", out_root=tmp_path
    )
    assert out["ran"] is False
    assert any(g["name"] == "run_count_cap" and not g["ok"] for g in out["gates"])


def test_gated_conveyor_runs_bounded_toy(tmp_path, monkeypatch):
    from mop.studio.profiles import Profile

    monkeypatch.setattr(Profile, "free_disk_ok", lambda self, root=None: (True, 1000.0))
    out = pipeline.cmd_run(
        gated=True,
        tiers={"C"},
        full=False,
        profile_name="m3pro-local-max",
        label="t3",
        out_root=tmp_path,
        max_legs=1,
    )
    assert out["ran"] is True
    assert all(g["ok"] for g in out["gates"])


def test_report_aggregates_run(tmp_path):
    out = pipeline.cmd_plan("studio-1tb", budget_gb=900, label="t", out_root=tmp_path)
    run_dir = Path(out["run_dir"])
    pipeline.cmd_acquire(run_dir / "plan.json", execute=False)
    summary = pipeline.cmd_report(run_dir=run_dir)
    assert "plan" in summary["stages_present"]
    assert (run_dir / "report.md").exists()
    assert summary["plan_totals"]["selected_sources"] >= 1


def test_optimize_writes_report(tmp_path):
    pipeline.cmd_plan("m3pro-local-max", budget_gb=5, label="t", out_root=tmp_path)
    out = pipeline.cmd_optimize(cache="none", profile_name="m3pro-local-max", reps=1)
    assert out["benches"]  # microbenchmarks ran
    assert "not science" in out["note"]


def test_validate_families_ok_after_generate(tmp_path):
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=10, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"
    pipeline.cmd_acquire(plan_path, execute=True)
    val = pipeline.cmd_validate(plan_path)
    syn = next(r for r in val["results"] if r["slug"] == "synthetic_controls")
    assert syn["validation"]["ok"] is True
    assert all(f["ok"] for f in syn["validation"]["families"])


def test_gated_conveyor_stops_on_invalid_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.registry, "validate_registry", lambda *a, **k: ["injected problem"])
    out = pipeline.cmd_run(
        tiers={"C"}, full=False, profile_name="m3pro-local-max", label="rg", out_root=tmp_path, max_legs=1
    )
    assert out["ran"] is False
    assert any(g["name"] == "registry_valid" and not g["ok"] for g in out["gates"])


def test_gated_conveyor_stops_on_low_free_disk(tmp_path, monkeypatch):
    from mop.studio.profiles import Profile

    monkeypatch.setattr(Profile, "free_disk_ok", lambda self, root=None: (False, 0.0))
    out = pipeline.cmd_run(
        tiers={"C"}, full=False, profile_name="m3pro-local-max", label="fd", out_root=tmp_path, max_legs=1
    )
    assert out["ran"] is False
    assert any(g["name"] == "free_disk" and not g["ok"] for g in out["gates"])


def test_capped_cache_drops_classes_honestly(tmp_path):
    from dataclasses import replace

    from mop.studio import controls
    from mop.studio.profiles import M3PRO_LOCAL_MAX

    controls.generate_controls(
        tmp_path / "c", families=["class_incremental"], clips_per_class=3, frames=4, h=8, w=8, seed=0
    )
    tiny = replace(M3PRO_LOCAL_MAX, max_cache_clips=2)  # force a cap below the 4 classes
    out = pipeline._build_tiny_cache(tmp_path / "c", tmp_path / "cache", "vjepa2_vitl_fpc64_256", tiny, 0)
    assert out["ok"] is True  # validates clean despite the cap
    assert out["n_classes_present"] < out["n_classes_declared"]
    assert out["classes_dropped"]  # the dropped classes are reported, not hidden
    lm = json.loads((Path(out["store_dir"]) / "label_map.json").read_text())
    assert len(lm) == out["n_classes_present"]


def test_acquire_defaults_budget_from_plan(tmp_path):
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=5, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"
    acq = pipeline.cmd_acquire(plan_path, execute=False)  # no budget_gb -> from plan (5.0)
    assert acq["manifest"]["budget_gb"] == 5.0


def test_local_max_smoke(tmp_path):
    _skip_if_disk_below_floor()
    s = pipeline.cmd_local_max(
        download_gb=10, time_min=90, cache_clips=16, seed=0, label="lm", out_root=tmp_path
    )
    assert s["overall"] == "pass", [st for st in s["stages"] if st["status"] == "fail"]
    assert s["effective"]["cache_clips"] <= 128
    assert s["effective"]["time_min"] <= 90
    assert "generate_controls" in s["real_vs_mocked"]
    cache_stage = next(st for st in s["stages"] if st["stage"] == "build_cache")
    assert cache_stage["status"] == "pass"
    assert (tmp_path / "lm" / "report.md").exists()
