import hashlib
import json
from dataclasses import replace

import pytest
import yaml

from mop.config import REPO_ROOT
from mop.studio.local_throttle import (
    DECISION_SCHEMA,
    TaskDeclaration,
    ThrottleRefused,
    active_lanes,
    aggregate_admission,
    checkpoint_snapshot,
    evaluate_task,
    hysteresis_transition,
    load_policy,
)


def _snapshot(**overrides):
    payload = {
        "cpu": {
            "available": True,
            "logical_cpus": 12,
            "load_1m_per_logical_cpu": 0.20,
            "utilization_fraction": 0.20,
        },
        "memory": {
            "available": True,
            "total_bytes": int(19.3e9),
            "available_bytes": int(14e9),
            "available_percent": 70.0,
            "pressure": {"available": True, "free_percent": 68.0},
        },
        "swap": {"available": True, "used_gb": 1.0},
        "disk": {"available": True, "free_gb": 80.0},
        "processes": {
            "available": True,
            "foreground_resource_processes": [],
            "unmanaged_known_heavy": [],
        },
        "mps": {
            "telemetry_available": True,
            "available": True,
            "declared_headroom_bytes": int(14e9),
            "scope": "test",
        },
        "thermal": {"available": True, "status": "normal"},
        "power": {"available": True, "source": "AC Power", "on_ac": True},
        "missing_required_telemetry": [],
        "all_required_available": True,
    }
    payload.update(overrides)
    return payload


def _active(task, run_id="active"):
    return {
        "run_id": run_id,
        "lane": task.lane,
        "accelerator": task.accelerator,
        "cpu_cores": task.cpu_cores,
        "estimated_unified_memory_gb": task.estimated_unified_memory_gb,
        "estimated_mps_gb": task.estimated_mps_gb,
        "forecast_write_gb": task.forecast_write_gb,
        "atomic_write_gb": task.atomic_write_gb,
    }


def test_policy_pins_five_hour_envelope_and_exact_p4_resume_command():
    policy = load_policy()
    cpu_task = policy.task("p4_resume_cpu")
    mps_task = policy.task("p4_resume_mps")
    assert policy.limits["hard_wall_minutes"] == 300
    assert policy.limits["disk_floor_gb"] == 40.0
    assert cpu_task.wall_minutes == 300
    assert cpu_task.restart_exit_codes == (2,)
    assert cpu_task.command == (
        ".venv/bin/python",
        "scripts/p4_capability_density.py",
        "--profile",
        "p4screen",
        "--device",
        "cpu",
        "--run-dir",
        "runs/p4_screen/p4screen",
        "--out",
        "proof/P4_CAPABILITY_DENSITY_SCREEN.json",
    )
    assert mps_task.command == (
        ".venv/bin/python",
        "scripts/p4_capability_density.py",
        "--profile",
        "p4screen",
        "--device",
        "mps",
        "--run-dir",
        "runs/p4_screen/p4screen_mps_clean",
        "--out",
        "proof/P4_CAPABILITY_DENSITY_SCREEN_MPS_CLEAN.json",
    )
    p4 = yaml.safe_load((REPO_ROOT / "configs/experiment/mop_p4_capability_density_screen.yaml").read_text())
    assert p4["profiles"]["p4screen"]["wall_budget_seconds"] == 10800.0


def test_policy_pins_cpu_p5_order_and_exact_commands():
    policy = load_policy()
    assert policy.execution_order["p5_cpu"] == (
        "p5smoke_cpu",
        "p5_traingrid_memory_probe_cpu",
        "p5pilot_cpu",
    )
    assert policy.task("p5smoke_cpu").command == (
        ".venv/bin/python",
        "scripts/p5_context_capability.py",
        "--profile",
        "p5smoke",
        "--device",
        "cpu",
        "--run-dir",
        "runs/p5_context/p5smoke",
        "--out",
        "proof/P5_CONTEXT_CAPABILITY_SMOKE.json",
    )
    assert policy.task("p5_traingrid_memory_probe_cpu").command == (
        ".venv/bin/python",
        "scripts/p5_traingrid_memory_probe.py",
        "--out",
        "proof/P5_TRAINGRID_MEMORY_TRACE.json",
        "--repeats",
        "3",
        "--seed",
        "0",
    )
    assert policy.task("p5pilot_cpu").command == (
        ".venv/bin/python",
        "scripts/p5_context_capability.py",
        "--profile",
        "p5pilot",
        "--device",
        "cpu",
        "--run-dir",
        "runs/p5_context/p5pilot",
        "--out",
        "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
    )


def test_policy_pins_p6_progressive_order_dependencies_and_exact_commands():
    policy = load_policy()
    order = policy.execution_order["p6_cpu"]
    assert order == (
        "p6_10k_resource_probe_cpu",
        "p6_10k_replication_cpu",
        "p6_100k_replication_cpu",
        "p6_1m_replication_cpu",
    )
    assert policy.task(order[0]).depends_on == ()
    assert policy.task(order[1]).depends_on == (order[0],)
    assert policy.task(order[2]).depends_on == (order[1],)
    assert policy.task(order[3]).depends_on == (order[2],)
    probe = policy.task(order[0])
    assert probe.resource_probe is True
    assert probe.requires_empty_lanes is True
    assert probe.estimated_unified_memory_gb is None
    assert probe.command == (
        ".venv/bin/python",
        "scripts/continual_million_event_rung.py",
        "--config",
        "configs/experiment/continual_million_event_rungs.yaml",
        "--rung",
        "10000",
        "--work-root",
        "runs/continual_million_event/rung_010000_probe",
        "--out",
        "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
        "--resource-probe",
        "--seed-count",
        "1",
        "--schedules",
        "abrupt",
        "--arms",
        "replay",
    )
    assert policy.task(order[3]).command[-5:] == (
        "1000000",
        "--work-root",
        "runs/continual_million_event/rung_1000000",
        "--out",
        "proof/P6_CONTINUAL_1M.json",
    )


def _write_p6_receipt(path, *, rung, mode="replication", replication=True, rss=200_000_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mop-continual-progressive-rung/v1",
        "mode": mode,
        "rung": rung,
        "all_mechanics_ok": True,
        "replication_execution_complete": replication,
        "identity": {
            "config_sha256": "8452b7a3d7b99a42f48576cf1bddc758079e4b4f09c24214cef6e75e7ca6ec59",
            "runner_sha256": "043131b19fe0d7447ecb347a58c46132add17f6a6a4bc3e5c83ed235bd420fdc",
            "source_preflight_file_sha256": (
                "fa1f65a6e839f0e4ac8c310da02f918a5666f70221f012dfb407eb4335d34e03"
            ),
        },
        "resource_measurement": {
            "max_rss_bytes": rss,
            "measured_after_complete": True,
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(payload))


def test_p6_probe_is_exclusive_and_never_overlaps_p4():
    policy = load_policy()
    probe = policy.task("p6_10k_resource_probe_cpu")
    p4 = policy.task("p4_resume_cpu")
    decision = evaluate_task(probe, _snapshot(), policy, active=[_active(p4)])
    assert decision["allowed"] is False
    exclusive = next(gate for gate in decision["gates"] if gate["name"] == "exclusive_lane")
    assert exclusive["ok"] is False


def test_p6_full_10k_fails_closed_until_probe_receipt_then_uses_measured_rss(tmp_path):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    blocked = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert blocked["allowed"] is False
    failed = {gate["name"] for gate in blocked["gates"] if not gate["ok"]}
    assert {"receipt_prerequisites", "resource_measurement"} <= failed

    _write_p6_receipt(
        tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
        rung=10_000,
        mode="resource-probe",
        replication=False,
    )
    admitted = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert admitted["allowed"] is True
    resource_gate = next(gate for gate in admitted["gates"] if gate["name"] == "resource_measurement")
    assert resource_gate["observed"]["max_rss_bytes"] == 200_000_000
    assert resource_gate["observed"]["effective_unified_memory_gb"] == pytest.approx(0.25)

    receipt_path = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    tampered = json.loads(receipt_path.read_text())
    tampered["resource_measurement"]["max_rss_bytes"] = 1
    receipt_path.write_text(json.dumps(tampered))
    drifted = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert drifted["allowed"] is False
    assert any(gate["name"] == "resource_measurement" and not gate["ok"] for gate in drifted["gates"])


def test_p6_one_million_disk_projection_fails_before_crossing_floor(tmp_path):
    policy = load_policy()
    task = policy.task("p6_1m_replication_cpu")
    _write_p6_receipt(tmp_path / "proof/P6_CONTINUAL_100K.json", rung=100_000)
    decision = evaluate_task(
        task,
        _snapshot(disk={"available": True, "free_gb": 41.0}),
        policy,
        evidence_root=tmp_path,
    )
    assert decision["allowed"] is False
    disk_gate = next(gate for gate in decision["gates"] if gate["name"] == "forecasted_disk")
    assert disk_gate["critical"] is True
    assert disk_gate["observed"]["projected_free_gb"] < 40.0


def test_p6_checkpoint_snapshot_covers_exact_resume_files_and_ignores_tmp(tmp_path):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    paths = (
        "runs/continual_million_event/rung_010000/streams/seed_20260710/abrupt/chunk_000000.bin",
        "runs/continual_million_event/rung_010000/streams/seed_20260710/abrupt/manifest.json",
        "runs/continual_million_event/rung_010000/checkpoints/seed_20260710/abrupt/replay.json",
        "runs/continual_million_event/rung_010000/progress.json",
        "proof/P6_CONTINUAL_10K.json",
    )
    for value in paths:
        path = tmp_path / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    temporary = tmp_path / (
        "runs/continual_million_event/rung_010000/checkpoints/seed_20260710/abrupt/replay.json.tmp"
    )
    temporary.write_text("partial")
    snapshot = checkpoint_snapshot(task, tmp_path)
    assert {row["path"] for row in snapshot["files"]} == set(paths)
    assert all(not row["path"].endswith(".tmp") for row in snapshot["files"])


def test_p6_policy_refuses_order_and_evidence_derived_forecast_drift(tmp_path):
    source = yaml.safe_load((REPO_ROOT / "configs/local_execution_throttle.yaml").read_text())
    wrong_order = json.loads(json.dumps(source))
    wrong_order["execution_order"]["p6_cpu"][0:2] = reversed(wrong_order["execution_order"]["p6_cpu"][0:2])
    order_path = tmp_path / "wrong-order.yaml"
    order_path.write_text(yaml.safe_dump(wrong_order, sort_keys=False))
    with pytest.raises(ThrottleRefused, match="execution_order.p6_cpu"):
        load_policy(order_path)

    wrong_forecast = json.loads(json.dumps(source))
    wrong_forecast["tasks"]["p6_1m_replication_cpu"]["forecast_write_gb"] += 0.001
    forecast_path = tmp_path / "wrong-forecast.yaml"
    forecast_path.write_text(yaml.safe_dump(wrong_forecast, sort_keys=False))
    with pytest.raises(ThrottleRefused, match="not derived from the 384-event receipt"):
        load_policy(forecast_path)


def test_first_mps_heavy_lane_is_allowed_with_measured_headroom():
    policy = load_policy()
    decision = evaluate_task(policy.task("p4_resume_mps"), _snapshot(), policy)
    assert decision["schema"] == DECISION_SCHEMA
    assert decision["allowed"] is True
    assert decision["disk_forecast"]["projected_free_gb"] >= 40.0


def test_second_heavy_lane_is_always_denied():
    policy = load_policy()
    task = policy.task("p4_resume_mps")
    decision = evaluate_task(task, _snapshot(), policy, active=[_active(task)])
    assert decision["allowed"] is False
    failed = {gate["name"] for gate in decision["gates"] if not gate["ok"]}
    assert {"one_heavy", "second_lane_kind", "single_mps_owner"} <= failed


def test_second_light_lane_is_allowed_only_under_strict_headroom():
    policy = load_policy()
    heavy = policy.task("p4_resume_mps")
    light = policy.task("docs_verification")
    decision = evaluate_task(light, _snapshot(), policy, active=[_active(heavy)])
    assert decision["threshold_tier"] == "second_lane"
    assert decision["allowed"] is True


def test_blender_presence_blocks_a_second_lane_but_not_the_only_experiment_lane():
    policy = load_policy()
    heavy = policy.task("p4_resume_mps")
    light = policy.task("docs_verification")
    processes = {
        "available": True,
        "foreground_resource_processes": [{"pid": 77, "name": "Blender"}],
        "unmanaged_known_heavy": [],
    }
    snapshot = _snapshot(processes=processes)
    assert evaluate_task(heavy, snapshot, policy)["allowed"] is True
    decision = evaluate_task(light, snapshot, policy, active=[_active(heavy)])
    assert decision["allowed"] is False
    assert any(gate["name"] == "foreground_second_lane" and not gate["ok"] for gate in decision["gates"])


def test_missing_telemetry_fails_closed():
    policy = load_policy()
    snapshot = _snapshot(missing_required_telemetry=["thermal"], all_required_available=False)
    decision = evaluate_task(policy.task("docs_verification"), snapshot, policy)
    assert decision["allowed"] is False
    assert decision["gates"][0]["name"] == "required_telemetry"


def test_mps_task_fails_closed_without_working_set_telemetry():
    policy = load_policy()
    snapshot = _snapshot(mps={"available": True, "telemetry_available": False, "scope": "missing"})
    decision = evaluate_task(policy.task("p4_resume_mps"), snapshot, policy)
    assert decision["allowed"] is False
    assert any(gate["name"] == "mps_telemetry" and not gate["ok"] for gate in decision["gates"])


def test_runtime_memory_gate_does_not_reserve_the_already_running_peak_twice():
    policy = load_policy()
    memory = {
        "available": True,
        "total_bytes": int(19.3e9),
        "available_bytes": int(3e9),
        "available_percent": 25.0,
        "pressure": {"available": True, "free_percent": 60.0},
    }
    snapshot = _snapshot(memory=memory)
    task = policy.task("p4_resume_mps")
    assert evaluate_task(task, snapshot, policy)["allowed"] is False
    assert evaluate_task(task, snapshot, policy, task_already_active=True)["allowed"] is True


def test_owned_cpu_saturation_is_admission_only_not_a_runtime_self_pause():
    policy = load_policy()
    task = policy.task("p4_resume_cpu")
    cpu = {
        "available": True,
        "logical_cpus": 12,
        "load_1m_per_logical_cpu": 1.25,
        "utilization_fraction": 1.0,
    }
    snapshot = _snapshot(cpu=cpu)
    admission = evaluate_task(task, snapshot, policy)
    assert admission["allowed"] is False
    assert {gate["name"] for gate in admission["gates"] if not gate["ok"]} == {
        "cpu_load",
        "cpu_utilization",
    }
    runtime = evaluate_task(task, snapshot, policy, task_already_active=True)
    assert runtime["allowed"] is True
    for name in ("cpu_load", "cpu_utilization"):
        gate = next(value for value in runtime["gates"] if value["name"] == name)
        assert gate["ok"] is True
        assert gate["limit"] == "admission-only"


def test_forecasted_writes_preserve_the_40gb_floor_and_fail_critical():
    policy = load_policy()
    snapshot = _snapshot(disk={"available": True, "free_gb": 45.0})
    decision = evaluate_task(policy.task("p4_resume_mps"), snapshot, policy)
    assert decision["allowed"] is False
    assert decision["critical"] is True
    disk_gate = next(gate for gate in decision["gates"] if gate["name"] == "forecasted_disk")
    assert disk_gate["observed"]["projected_free_gb"] < 40.0


def test_admission_and_runtime_hysteresis_require_consecutive_samples():
    policy = load_policy()
    allowed = {"allowed": True, "critical": False}
    denied = {"allowed": False, "critical": False}
    assert aggregate_admission([allowed, allowed], 3)["allowed"] is False
    assert aggregate_admission([denied, allowed, allowed, allowed], 3)["allowed"] is True
    first = hysteresis_transition(
        "running",
        denied,
        good_count=0,
        bad_count=0,
        last_transition_monotonic=0.0,
        now_monotonic=10.0,
        policy=policy,
    )
    assert first["action"] == "none"
    second = hysteresis_transition(
        "running",
        denied,
        good_count=first["good_count"],
        bad_count=first["bad_count"],
        last_transition_monotonic=0.0,
        now_monotonic=20.0,
        policy=policy,
    )
    assert second["action"] == "pause"
    state = second
    for now in (30.0, 40.0, 90.0):
        state = hysteresis_transition(
            "paused",
            allowed,
            good_count=state["good_count"],
            bad_count=state["bad_count"],
            last_transition_monotonic=20.0,
            now_monotonic=now,
            policy=policy,
        )
    assert state["action"] == "resume"


def test_checkpoint_snapshot_hashes_final_publication_and_ignores_tmp(tmp_path):
    task = TaskDeclaration(
        task_id="fixture",
        lane="light",
        accelerator="none",
        cpu_cores=1,
        estimated_unified_memory_gb=0.1,
        estimated_mps_gb=0.0,
        resource_basis="fixture",
        forecast_write_gb=0.1,
        atomic_write_gb=0.1,
        wall_minutes=1,
        pause_safe=True,
        atomic_checkpoints=True,
        checkpoint_globs=("run/*",),
        restart_exit_codes=(),
        command=("true",),
    )
    (tmp_path / "run").mkdir()
    (tmp_path / "run/checkpoint.pt").write_bytes(b"complete")
    (tmp_path / "run/checkpoint.pt.tmp").write_bytes(b"partial")
    first = checkpoint_snapshot(task, tmp_path)
    assert [row["path"] for row in first["files"]] == ["run/checkpoint.pt"]
    assert first["file_count"] == 1
    second = checkpoint_snapshot(replace(task), tmp_path)
    assert first["aggregate_sha256"] == second["aggregate_sha256"]


def test_decision_payload_is_json_serializable():
    policy = load_policy()
    decision = evaluate_task(policy.task("p4_resume_mps"), _snapshot(), policy)
    assert json.loads(json.dumps(decision))["allowed"] is True


def test_list_cli_serializes_slotted_task_declarations(capsys):
    from scripts.local_execution_throttle import main

    assert main(["list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tasks"]["p4_resume_cpu"]["task_id"] == "p4_resume_cpu"
    assert payload["tasks"]["p4_resume_cpu"]["command"][1] == ("scripts/p4_capability_density.py")


def test_dry_receipt_binds_policy_and_implementation(monkeypatch):
    from mop.studio import local_throttle

    policy = load_policy()
    monkeypatch.setattr(local_throttle, "collect_host_telemetry", lambda *_args, **_kwargs: _snapshot())
    receipt = local_throttle.dry_run_decision(
        policy.task("p4_resume_cpu"), policy, samples=3, interval_seconds=0
    )
    assert receipt["policy"]["sha256"] == policy.sha256
    assert receipt["implementation"]["path"] == "src/mop/studio/local_throttle.py"
    assert len(receipt["implementation"]["sha256"]) == 64


def test_corrupt_active_registry_fails_closed(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "active.json").write_text("not json")
    with pytest.raises(ThrottleRefused, match="registry"):
        active_lanes(tmp_path)
