"""The one Studio pipeline surface end to end: plan -> acquire(dry) -> validate -> cache(dry) ->
run(gated) -> report, plus the gated conveyor stopping on a failed gate, and a bounded local-max
smoke. Heavy acquisition stays dry-run; the conveyor refuses a disallowed tier."""

import json
from pathlib import Path

import pytest

from devsys.provenance import RESULT_TAGS
from devsys.studio import pipeline


def _skip_if_disk_below_floor():
    # The execute-path stages honor the profile's free-disk kill switch (refuse below the floor,
    # by design). On a machine that is genuinely below the floor those stages can never run, so
    # the tests that assert a REAL cache build skip cleanly instead of failing on the refusal.
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
    # latest pointer resolves back to this run
    resolved = pipeline._resolve_latest(tmp_path / "latest" / "plan.json")
    assert resolved.exists()
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["provenance"]["result_tag"] in RESULT_TAGS
    assert plan["totals"]["selected_sources"] >= 1


def test_acquire_dry_run_then_validate_then_cache_estimate(tmp_path):
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=10, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"

    # acquire dry-run: synthetic_controls is a 'generate' source but dry-run writes nothing
    acq = pipeline.cmd_acquire(plan_path, execute=False)
    assert acq["manifest"]["mode"] == "dry-run"
    assert acq["manifest"]["totals"]["bytes_spent"] == 0

    # validate: nothing acquired yet -> reported pending, never a crash
    val = pipeline.cmd_validate(plan_path)
    assert all(r["validation"].get("ok") in (None, True, False) for r in val["results"])

    # cache dry-run: estimates only, no store built
    cache = pipeline.cmd_cache(plan_path, execute=False)
    assert any("estimate" in r for r in cache["results"])
    assert all(r.get("cache", {}).get("ok") in (None,) for r in cache["results"] if "cache" in r)


def test_acquire_execute_generates_controls_then_caches(tmp_path):
    # execute on the m3pro plan actually GENERATES the synthetic-controls corpus (safe, on-device),
    # then cache --execute builds and validates a tiny real cache from it.
    _skip_if_disk_below_floor()
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=10, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"
    acq = pipeline.cmd_acquire(plan_path, execute=True, accept_license=False)
    by = {s["slug"]: s for s in acq["manifest"]["sources"]}
    assert by["synthetic_controls"]["status"] == "complete"
    assert by["synthetic_controls"]["bytes"] > 0
    # remote sources without credentials stay blocked, never crash
    assert by.get("epic_kitchens_subset", {}).get("status") in ("blocked", "skipped-budget", "needs-license")

    cache = pipeline.cmd_cache(plan_path, execute=True)
    syn = next(r for r in cache["results"] if r["slug"] == "synthetic_controls")
    assert syn["cache"]["ok"] is True
    assert syn["cache"]["count"] > 0
    assert syn["cache"]["backend"] == "frozen_random"


def test_gated_conveyor_stops_on_disallowed_tier(tmp_path):
    # m3pro disallows Tier E: the conveyor must STOP at the tier gate, not run
    out = pipeline.cmd_run(
        gated=True, tiers={"E"}, full=False, profile_name="m3pro-local-max", label="t", out_root=tmp_path
    )
    assert out["ran"] is False
    assert any(g["name"] == "tier_allowed" and not g["ok"] for g in out["gates"])
    assert "reason" in out


def test_gated_conveyor_stops_full_over_run_cap(tmp_path):
    # m3pro caps run-count; a FULL campaign expands past it and must be refused
    out = pipeline.cmd_run(
        gated=True, tiers={"C"}, full=True, profile_name="m3pro-local-max", label="t2", out_root=tmp_path
    )
    assert out["ran"] is False
    assert any(g["name"] == "run_count_cap" and not g["ok"] for g in out["gates"])


def test_gated_conveyor_runs_bounded_toy(tmp_path, monkeypatch):
    # a bounded toy Tier C run passes the gates and actually runs one leg. The free-disk gate is
    # forced open (the toy leg writes only to tmp_path) so a host genuinely below the floor still
    # exercises the run path; the gate-stops behavior is covered by
    # test_gated_conveyor_stops_on_low_free_disk, which forces it shut the same way.
    from devsys.studio.profiles import Profile

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
    # the real validation success path: generated control families validate ok=True
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=10, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"
    pipeline.cmd_acquire(plan_path, execute=True)
    val = pipeline.cmd_validate(plan_path)
    syn = next(r for r in val["results"] if r["slug"] == "synthetic_controls")
    assert syn["validation"]["ok"] is True
    assert all(f["ok"] for f in syn["validation"]["families"])


def test_gated_conveyor_stops_on_invalid_registry(tmp_path, monkeypatch):
    # registry_valid is a kill switch: a non-empty problem list must STOP the run
    monkeypatch.setattr(pipeline.registry, "validate_registry", lambda *a, **k: ["injected problem"])
    out = pipeline.cmd_run(
        tiers={"C"}, full=False, profile_name="m3pro-local-max", label="rg", out_root=tmp_path, max_legs=1
    )
    assert out["ran"] is False
    assert any(g["name"] == "registry_valid" and not g["ok"] for g in out["gates"])


def test_gated_conveyor_stops_on_low_free_disk(tmp_path, monkeypatch):
    # free_disk is a kill switch: below the floor must STOP the run
    from devsys.studio.profiles import Profile

    monkeypatch.setattr(Profile, "free_disk_ok", lambda self, root=None: (False, 0.0))
    out = pipeline.cmd_run(
        tiers={"C"}, full=False, profile_name="m3pro-local-max", label="fd", out_root=tmp_path, max_legs=1
    )
    assert out["ran"] is False
    assert any(g["name"] == "free_disk" and not g["ok"] for g in out["gates"])


def test_capped_cache_drops_classes_honestly(tmp_path):
    # the honest-coverage invariant: a clip cap below the class count must NOT leave label_map
    # claiming dropped classes; the cache validates clean and reports the coverage gap.
    from dataclasses import replace

    from devsys.studio import controls
    from devsys.studio.profiles import M3PRO_LOCAL_MAX

    controls.generate_controls(
        tmp_path / "c", families=["class_incremental"], clips_per_class=3, frames=4, h=8, w=8, seed=0
    )
    tiny = replace(M3PRO_LOCAL_MAX, max_cache_clips=2)  # force a cap below the 4 classes
    out = pipeline._build_tiny_cache(tmp_path / "c", tmp_path / "cache", "vjepa2_vitl_fpc64_256", tiny, 0)
    assert out["ok"] is True  # validates clean despite the cap
    assert out["n_classes_present"] < out["n_classes_declared"]
    assert out["classes_dropped"]  # the dropped classes are reported, not hidden
    # label_map.json persists ONLY the present classes
    lm = json.loads((Path(out["store_dir"]) / "label_map.json").read_text())
    assert len(lm) == out["n_classes_present"]


def test_acquire_defaults_budget_from_plan(tmp_path):
    # acquire honors the plan's budget when --budget-gb is omitted (no silent larger envelope)
    out = pipeline.cmd_plan("m3pro-local-max", budget_gb=5, label="t", out_root=tmp_path)
    plan_path = Path(out["run_dir"]) / "plan.json"
    acq = pipeline.cmd_acquire(plan_path, execute=False)  # no budget_gb -> from plan (5.0)
    assert acq["manifest"]["budget_gb"] == 5.0


def test_local_max_smoke(tmp_path):
    # the current-device lane, bounded tiny; every stage passes and kill switches are recorded
    _skip_if_disk_below_floor()
    s = pipeline.cmd_local_max(
        download_gb=10, time_min=90, cache_clips=16, seed=0, label="lm", out_root=tmp_path
    )
    assert s["overall"] == "pass", [st for st in s["stages"] if st["status"] == "fail"]
    # requested knobs clamped to the m3pro envelope
    assert s["effective"]["cache_clips"] <= 128
    assert s["effective"]["time_min"] <= 90
    # honest real/mocked tagging present and a built cache reported
    assert "generate_controls" in s["real_vs_mocked"]
    cache_stage = next(st for st in s["stages"] if st["stage"] == "build_cache")
    assert cache_stage["status"] == "pass"
    assert (tmp_path / "lm" / "report.md").exists()
