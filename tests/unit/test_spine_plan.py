import json

import scripts.studio.__main__ as studio_cli

from mop.studio.long_run import load_plan
from mop.studio.spine_plan import (
    StudioSpineConfig,
    build_studio_spine_plan,
    build_studio_spine_status,
    validate_studio_spine_plan,
    write_spine_wave0_plan,
    write_studio_spine_plan,
)


def test_studio_spine_plan_orders_wave0_dr1_pr9_dense_atlas():
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    assert plan["schema"] == "mop-studio-spine-plan/v1"
    assert plan["profile"]["name"] == "studio-m1ultra"
    assert validate_studio_spine_plan(plan) == []
    ids = [step["id"] for step in plan["steps"]]
    assert ids.index("dr1_source_card_validate") < ids.index("dr1_source_intake")
    assert ids.index("dr1_source_intake") < ids.index("dr1_schedule_build")
    assert ids.index("dr1_artifact_bundle") < ids.index("pr9_run")
    assert ids.index("pr9_run") < ids.index("pr9_verdict_ledger")
    assert ids.index("pr9_verdict_ledger") < ids.index("process_c_license_gate")
    assert ids.index("process_c_license_gate") < ids.index("pr9_artifact_bundle")
    assert ids.index("pr9_artifact_bundle") < ids.index("dense_cache_plan")
    assert ids.index("dense_atlas_cache_gate") < ids.index("atlas_run")
    assert ids.index("atlas_run") < ids.index("atlas_verdict_ledger")
    assert ids.index("atlas_verdict_ledger") < ids.index("atlas_artifact_bundle")
    assert ids.index("atlas_artifact_bundle") < ids.index("studio_scorecard")
    assert ids.index("studio_scorecard") < ids.index("spine_status_receipt")
    assert ids.index("spine_status_receipt") < ids.index("studio_objective_audit")
    assert ids.index("studio_objective_audit") < ids.index("spine_artifact_bundle")
    assert ids.index("spine_status_receipt") < ids.index("spine_artifact_bundle")
    atlas_cmd = next(step["cmd"] for step in plan["steps"] if step["id"] == "atlas_run")
    assert "--allow-partial" not in atlas_cmd


def test_studio_spine_accepts_any_registered_studio_resource_envelope():
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video", profile_name="studio-1tb"))
    assert plan["profile"]["name"] == "studio-1tb"
    assert validate_studio_spine_plan(plan) == []


def test_studio_spine_plan_carries_verifier_and_bundle_receipts():
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    card_validation = next(step for step in plan["steps"] if step["id"] == "dr1_source_card_validate")
    assert "runs/studio_dr1/dr1_source_card_validation.json" in card_validation["expected_receipts"]
    source_intake = next(step for step in plan["steps"] if step["id"] == "dr1_source_intake")
    assert "runs/studio_dr1/dr1_source_intake.json" in source_intake["expected_receipts"]
    dr1_run = next(step for step in plan["steps"] if step["id"] == "dr1_run")
    assert "data/cache/vjepa2_vitl_comp_video/dr1_verification.json" in dr1_run["expected_receipts"]
    pr9_ledger = next(step for step in plan["steps"] if step["id"] == "pr9_verdict_ledger")
    assert "runs/mot/pr9_verdict_ledger.json" in pr9_ledger["expected_receipts"]
    process_c_gate = next(step for step in plan["steps"] if step["id"] == "process_c_license_gate")
    assert "runs/mot/process_c_license_gate.json" in process_c_gate["expected_receipts"]
    dense_gate = next(step for step in plan["steps"] if step["id"] == "dense_atlas_cache_gate")
    assert "runs/mot/dense_atlas_cache_gate.json" in dense_gate["expected_receipts"]
    assert (
        "data/cache/vjepa21_vitl_dense8192_randominit/cache_manifest.json" in dense_gate["expected_receipts"]
    )
    atlas_ledger = next(step for step in plan["steps"] if step["id"] == "atlas_verdict_ledger")
    assert "runs/mot/atlas_verdict_ledger.json" in atlas_ledger["expected_receipts"]
    objective_audit = next(step for step in plan["steps"] if step["id"] == "studio_objective_audit")
    assert "runs/studio_objective_audit.json" in objective_audit["expected_receipts"]
    assert "--allow-not-ready" in objective_audit["cmd"]
    studio_scorecard = next(step for step in plan["steps"] if step["id"] == "studio_scorecard")
    assert "--allow-incomplete" in studio_scorecard["cmd"]
    assert "proof/ARTIFACT_INDEX/dr1.json" in plan["expected_receipts"]
    assert "proof/ARTIFACT_INDEX/pr9.json" in plan["expected_receipts"]
    assert "proof/ARTIFACT_INDEX/atlas.json" in plan["expected_receipts"]


def test_write_spine_plan_and_wave0_subplan_round_trip(tmp_path):
    plan_path = tmp_path / "spine.json"
    wave0_path = tmp_path / "wave0.json"
    write_spine_wave0_plan(wave0_path)
    assert [job.job_id for job in load_plan(wave0_path)][0] == "transfer_check"
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video", spine_dir=tmp_path))
    write_studio_spine_plan(plan, plan_path)
    loaded = json.loads(plan_path.read_text())
    assert loaded["schema"] == "mop-studio-spine-plan/v1"


def test_spine_plan_cli_writes_plan_and_wave0_subplan(tmp_path):
    plan_path = tmp_path / "spine_plan.json"
    wave0_path = tmp_path / "wave0_daemon_plan.json"
    rc = studio_cli.main(
        [
            "spine-plan",
            "--source",
            "/data/comp_video",
            "--out",
            str(plan_path),
            "--wave0-plan-out",
            str(wave0_path),
        ]
    )
    assert rc == 0
    plan = json.loads(plan_path.read_text())
    assert plan["subplans"]["wave0_daemon_plan"] == str(wave0_path)
    assert wave0_path.exists()
    assert load_plan(wave0_path)


def test_spine_status_reports_next_command_from_missing_receipts(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "runs" / "studio_spine").mkdir(parents=True)
    (root / "runs" / "studio_spine" / "wave0_daemon_plan.json").write_text(
        json.dumps({"schema": "mop-long-run-daemon/v1", "jobs": [{"id": "x", "cmd": ["ok"]}]})
    )
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    status = build_studio_spine_status(plan, repo_root=root)
    assert status["summary"]["complete"] == 1
    assert status["next_step"]["id"] == "wave0_run"
    assert "runs/studio_wave0/daemon_state.json" in status["next_step"]["missing_receipts"]


def test_spine_status_blocks_on_dense_schedule_wall(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    dense_step = next(step for step in plan["steps"] if step["id"] == "dense_cache_plan")
    for receipt in dense_step["expected_receipts"]:
        path = root / receipt
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ok_to_launch": True}
        if receipt.endswith("dense_encode_schedule.json"):
            payload = {"ok_to_launch": False, "blocked_reasons": ["dense encoder unavailable"]}
        path.write_text(json.dumps(payload))
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "dense_cache_plan")
    assert step["status"] == "blocked"
    assert "dense encoder unavailable" in step["signals"][0]["blocked_reasons"]


def test_spine_status_blocks_bad_dense_atlas_gate(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "runs" / "mot" / "dense_atlas_cache_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-dense-atlas-cache-gate/v1",
                "all_ok": False,
                "problems": ["randominit: cache_manifest.json missing"],
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "dense_atlas_cache_gate")
    assert step["status"] == "blocked"
    assert "randominit" in step["signals"][0]["problems"][0]


def test_spine_status_blocks_bad_atlas_verdict_ledger(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "runs" / "mot" / "atlas_verdict_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-atlas-verdict-ledger/v1",
                "all_ok": False,
                "status": "partial_non_scoring",
                "problems": ["missing registered columns"],
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "atlas_verdict_ledger")
    assert step["status"] == "blocked"
    assert "missing registered columns" in step["signals"][0]["problems"][0]


def test_spine_status_fails_bad_artifact_bundle(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "proof" / "ARTIFACT_INDEX" / "pr9.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-artifact-bundle/v1",
                "all_ok": False,
                "problems": ["artifact is not durable"],
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "pr9_artifact_bundle")
    assert step["status"] == "failed"
    assert "artifact is not durable" in step["signals"][0]["problems"]


def test_spine_status_blocks_bad_dr1_source_intake(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "runs" / "studio_dr1" / "dr1_source_intake.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-dr1-source-intake/v1",
                "all_ok": False,
                "problems": ["missing source card"],
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "dr1_source_intake")
    assert step["status"] == "blocked"
    assert "missing source card" in step["signals"][0]["problems"]


def test_spine_status_blocks_bad_dr1_source_card_validation(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "runs" / "studio_dr1" / "dr1_source_card_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-dr1-source-card-validation/v1",
                "all_ok": False,
                "problems": ["source card field 'license' is empty or unknown"],
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "dr1_source_card_validate")
    assert step["status"] == "blocked"
    assert "license" in step["signals"][0]["problems"][0]


def test_spine_status_fails_non_scoring_pr9_verdict(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "runs" / "mot" / "pr9_verdict_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-pr9-verdict-ledger/v1",
                "all_ok": False,
                "status": "non_scoring",
                "problems": ["wrong cache"],
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "pr9_verdict_ledger")
    assert step["status"] == "failed"
    assert "wrong cache" in step["signals"][0]["problems"]


def test_spine_status_blocks_not_ready_objective_audit(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "runs" / "studio_objective_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-studio-objective-audit/v1",
                "studio_10_ready": False,
                "summary": {"points_earned": 2.768, "points_possible": 8.0},
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "studio_objective_audit")
    assert step["status"] == "blocked"
    assert step["signals"][0]["summary"]["points_possible"] == 8.0


def test_spine_status_blocks_undecidable_process_c_gate(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video"))
    path = root / "runs" / "mot" / "process_c_license_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mop-process-c-license-gate/v1",
                "all_ok": False,
                "status": "undecidable",
                "problems": ["no decisive PR9 or DR1 receipt"],
            }
        )
    )
    status = build_studio_spine_status(plan, repo_root=root)
    step = next(step for step in status["steps"] if step["id"] == "process_c_license_gate")
    assert step["status"] == "blocked"
    assert "no decisive" in step["signals"][0]["problems"][0]


def test_spine_status_cli_writes_status_receipt(tmp_path):
    plan_path = tmp_path / "spine_plan.json"
    status_path = tmp_path / "spine_status.json"
    plan = build_studio_spine_plan(StudioSpineConfig(source="/data/comp_video", spine_dir=tmp_path))
    write_studio_spine_plan(plan, plan_path)
    rc = studio_cli.main(
        ["spine-plan", "--status", "--plan", str(plan_path), "--status-out", str(status_path)]
    )
    assert rc == 0
    status = json.loads(status_path.read_text())
    assert status["schema"] == "mop-studio-spine-status/v1"
    assert status["next_step"]["id"] == "wave0_validate_plan"
