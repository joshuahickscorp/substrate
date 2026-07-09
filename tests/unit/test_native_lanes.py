import json

from mop.studio.long_run import load_plan
from mop.studio.native_lanes import build_native_lane_manifest, write_native_daemon_plan


def _by_id(manifest):
    return {lane["id"]: lane for lane in manifest["lanes"]}


def test_native_lane_manifest_keeps_heavy_lanes_blocked_by_default():
    manifest = build_native_lane_manifest(profile_name="studio-m1ultra")
    lanes = _by_id(manifest)
    assert lanes["hosted_corpora_plan"]["status"] == "ready"
    assert lanes["dr13_predictor_fidelity_real"]["status"] == "blocked"
    assert "include-heavy" in lanes["dr13_predictor_fidelity_real"]["blocked_reason"]


def test_native_lane_manifest_requires_inputs_for_heavy_lanes():
    manifest = build_native_lane_manifest(profile_name="studio-m1ultra", include_heavy=True)
    lanes = _by_id(manifest)
    assert lanes["dr13_predictor_fidelity_real"]["status"] == "blocked"
    assert "clip_dir" in lanes["dr13_predictor_fidelity_real"]["blocked_reason"]
    assert lanes["hosted_corpora_acquire"]["status"] == "blocked"
    assert "plan_path" in lanes["hosted_corpora_acquire"]["blocked_reason"]
    assert lanes["process_c_dense_token_decision"]["status"] == "blocked"
    assert "pr9_verdict" in lanes["process_c_dense_token_decision"]["blocked_reason"]


def test_native_lane_manifest_materializes_concrete_commands():
    manifest = build_native_lane_manifest(
        profile_name="studio-m1ultra",
        include_heavy=True,
        lane_ids=["dr13_predictor_fidelity_real", "pr9_long_stream", "process_c_dense_token_decision"],
        inputs={"clip_dir": "/tmp/real_clips", "dr1_cache": "data/cache/dr1"},
    )
    lanes = _by_id(manifest)
    assert lanes["dr13_predictor_fidelity_real"]["status"] == "ready"
    assert "/tmp/real_clips" in lanes["dr13_predictor_fidelity_real"]["command"]
    assert lanes["pr9_long_stream"]["command"][:3] == [
        "python",
        "scripts/studio/pr9_continual_backprop.py",
        "--cache",
    ]
    assert lanes["process_c_dense_token_decision"]["status"] == "blocked"


def test_native_lane_manifest_materializes_process_c_license_gate():
    manifest = build_native_lane_manifest(
        profile_name="studio-m1ultra",
        lane_ids=["process_c_dense_token_decision"],
        inputs={
            "pr9_verdict": "runs/mot/pr9_verdict_ledger.json",
            "dr1_verification": "data/cache/vjepa2_vitl_comp_video/dr1_verification.json",
        },
    )
    lane = manifest["lanes"][0]
    assert lane["status"] == "ready"
    assert lane["command"][:3] == ["python", "scripts/studio/process_c_license_gate.py", "--pr9-verdict"]
    assert "runs/mot/process_c_license_gate.json" in lane["command"]


def test_native_daemon_plan_contains_ready_jobs_and_blocked_receipts(tmp_path):
    manifest = build_native_lane_manifest(profile_name="studio-m1ultra")
    out = tmp_path / "native_plan.json"
    plan = write_native_daemon_plan(manifest, out)
    assert [job["id"] for job in plan["jobs"]] == ["hosted_corpora_plan"]
    assert any(lane["id"] == "process_c_dense_token_decision" for lane in plan["blocked_lanes"])
    jobs = load_plan(out)
    assert [job.job_id for job in jobs] == ["hosted_corpora_plan"]
    assert json.loads(out.read_text())["native_schema"] == "mop-studio-native-lanes/v1"
