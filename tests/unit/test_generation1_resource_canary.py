from __future__ import annotations

import copy
import io
import json
import os
import signal
from pathlib import Path

import pytest

from mop.studies import generation1_cognitive_corpus as corpus
from mop.studies import generation1_resource_canary as canary


def _telemetry() -> dict[str, object]:
    return {
        "schema": "mop-local-execution-telemetry/v1",
        "created_at": "2026-07-13T00:00:00+00:00",
        "cpu": {
            "available": True,
            "logical_cpus": 28,
            "load_1m": 2.8,
            "load_5m": 2.0,
            "load_15m": 1.0,
            "load_1m_per_logical_cpu": 0.1,
            "utilization_fraction": 0.12,
        },
        "memory": {
            "available": True,
            "total_bytes": 103_000_000_000,
            "available_bytes": 80_000_000_000,
            "available_percent": 78.0,
            "pressure": {"available": True, "free_percent": 92.0},
        },
        "swap": {
            "available": True,
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "used_gb": 0.0,
            "percent": 0.0,
        },
        "disk": {"available": True, "free_bytes": 500_000_000_000, "free_gb": 500.0},
        "processes": {
            "available": True,
            "foreground_resource_processes": [],
            "unmanaged_known_heavy": [],
        },
        "thermal": {"available": True, "status": "normal"},
        "power": {"available": True, "source": "AC Power", "on_ac": True},
        "missing_required_telemetry": [],
        "all_required_available": True,
    }


def _fast(
    *,
    memory: int,
    memory_percent: float,
    swap: int,
    swap_percent: float,
    cpu: float,
    load: float,
) -> dict[str, object]:
    return {
        "memory_available_bytes": memory,
        "memory_available_percent": memory_percent,
        "swap_used_bytes": swap,
        "swap_used_percent": swap_percent,
        "cpu_utilization_fraction": cpu,
        "load_1m": load,
        "load_1m_per_logical_cpu": load / 28.0,
    }


def test_frozen_batch_is_the_aligned_sorted_slice_containing_ex9() -> None:
    config = json.loads(corpus.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    eligible = corpus.eligible_experiment_ids(config)
    anchor_index = eligible.index(canary.CANARY_ANCHOR)
    start = anchor_index // canary.CANARY_BATCH_SIZE * canary.CANARY_BATCH_SIZE

    assert canary.derive_exact_batch(config) == canary.CANARY_BATCH
    assert tuple(eligible[start : start + 16]) == canary.CANARY_BATCH
    assert len(canary.CANARY_BATCH) == 16
    assert canary.CANARY_BATCH.index("ex9_slot_attention") == 13


def test_batch_derivation_fails_closed_on_registry_scope_drift() -> None:
    config = json.loads(corpus.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["experiment_scope"]["excluded_ids"] = ["ex14_memory_bakeoff"]

    with pytest.raises(canary.CanaryRefused, match="batch drifted"):
        canary.derive_exact_batch(config)


def test_source_snapshot_binds_canary_and_exact_worker_authorities() -> None:
    config = corpus.load_config(corpus.DEFAULT_CONFIG)
    snapshot, _ = canary.source_snapshot(
        corpus.DEFAULT_CONFIG,
        config,
        canary.CANARY_BATCH,
        int(config["seeds"][0]),
    )
    paths = {row["path"] for row in snapshot["files"]}

    assert "scripts/generation1_resource_canary.py" in paths
    assert "src/mop/studies/generation1_resource_canary.py" in paths
    assert "scripts/generation1_cognitive_corpus.py" in paths
    assert "src/mop/studies/generation1_cognitive_corpus.py" in paths
    assert "src/mop/studio/local_throttle.py" in paths
    assert "configs/local_execution_throttle.yaml" in paths


def test_admission_requires_empty_host_and_conservative_resource_headroom() -> None:
    safe = canary.evaluate_admission(_telemetry(), [])
    assert safe["safe"] is True
    assert all(row["ok"] for row in safe["gates"])
    gate_names = [row["name"] for row in safe["gates"]]
    assert len(gate_names) == len(set(gate_names))
    assert gate_names.count("thermal") == 1

    unsafe_telemetry = copy.deepcopy(_telemetry())
    unsafe_telemetry["memory"]["pressure"]["free_percent"] = 30.0
    unsafe = canary.evaluate_admission(unsafe_telemetry, [{"run_id": "other"}])
    assert unsafe["safe"] is False
    assert {"active_lanes", "memory_pressure_free_percent"}.issubset(unsafe["problems"])


def test_runtime_gate_ignores_expected_cpu_saturation_but_catches_memory_risk() -> None:
    telemetry = _telemetry()
    telemetry["cpu"]["utilization_fraction"] = 1.0
    telemetry["cpu"]["load_1m_per_logical_cpu"] = 1.0
    assert canary.runtime_safety_problems(telemetry, []) == []

    telemetry["swap"]["used_gb"] = 5.0
    telemetry["memory"]["available_bytes"] = 20_000_000_000
    problems = canary.runtime_safety_problems(telemetry, [])
    assert "swap crossed the runtime abort ceiling" in problems
    assert "available unified memory crossed the runtime abort floor" in problems


def test_measurement_fold_tracks_required_extrema_and_process_tree_peak() -> None:
    summary = canary.empty_measurements()
    first_probe = _telemetry()
    second_probe = copy.deepcopy(first_probe)
    second_probe["memory"]["pressure"]["free_percent"] = 81.0
    second_probe["thermal"]["status"] = "normal"

    canary.record_measurement(
        summary,
        fast_host=_fast(
            memory=80_000_000_000,
            memory_percent=78.0,
            swap=0,
            swap_percent=0.0,
            cpu=0.5,
            load=8.0,
        ),
        aggregate_process_tree_rss_bytes=7_000_000_000,
        worker_process_trees_rss_bytes=6_000_000_000,
        full_probe=first_probe,
    )
    canary.record_measurement(
        summary,
        fast_host=_fast(
            memory=70_000_000_000,
            memory_percent=68.0,
            swap=500_000_000,
            swap_percent=1.0,
            cpu=0.9,
            load=16.0,
        ),
        aggregate_process_tree_rss_bytes=11_000_000_000,
        worker_process_trees_rss_bytes=10_000_000_000,
        full_probe=second_probe,
    )

    assert summary["sample_count"] == 2
    assert summary["host_probe_count"] == 2
    assert summary["aggregate_process_tree_peak_rss_bytes"] == 11_000_000_000
    assert summary["worker_process_trees_peak_rss_bytes"] == 10_000_000_000
    assert summary["minimum_memory_available_bytes"] == 70_000_000_000
    assert summary["minimum_memory_pressure_free_percent"] == 81.0
    assert summary["maximum_swap_used_bytes"] == 500_000_000
    assert summary["maximum_cpu_utilization_fraction"] == 0.9
    assert summary["maximum_load_1m"] == 16.0
    assert summary["thermal_statuses"] == ["normal"]
    assert summary["power_sources"] == ["AC Power"]


def test_recommendation_never_extrapolates_beyond_successful_measured_16() -> None:
    measurements = canary.empty_measurements()
    measurements["aggregate_process_tree_peak_rss_bytes"] = 10_000_000_000
    workers = [
        {"experiment_id": experiment_id, "outcome": "ok", "peak_rss_bytes": 800_000_000}
        for experiment_id in canary.CANARY_BATCH
    ]

    recommendation = canary.recommend_resources(measurements, workers, source_stable=True)
    assert recommendation["eligible"] is True
    assert recommendation["recommended_max_workers"] == 16
    assert recommendation["recommended_estimated_unified_memory_gb"] == 16.0
    assert "no extrapolation" in recommendation["scaling_boundary"]

    failed = copy.deepcopy(workers)
    failed[0]["outcome"] = "failed"
    refused = canary.recommend_resources(measurements, failed, source_stable=True)
    assert refused["eligible"] is False
    assert refused["recommended_max_workers"] is None
    assert refused["recommended_estimated_unified_memory_gb"] is None


def test_worker_command_uses_exact_corpus_worker_entrypoint_and_isolated_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs/generation1/resource_canary/test/workers/ex9/attempt_001"
    config = tmp_path / "config.json"
    command = canary.worker_command(
        "ex9_slot_attention",
        outer_seed=20260801,
        run_dir=run_dir,
        result_tag="generation1-test",
        config_path=config,
        python="python",
        worker_script=Path("scripts/generation1_cognitive_corpus.py"),
    )

    assert command == [
        "python",
        "scripts/generation1_cognitive_corpus.py",
        "worker",
        "--experiment",
        "ex9_slot_attention",
        "--seed",
        "20260801",
        "--run-dir",
        str(run_dir),
        "--result-tag",
        "generation1-test",
        "--config",
        str(config),
    ]
    with pytest.raises(canary.CanaryRefused, match="outside the frozen"):
        canary.worker_command(
            "a1_affordance_decode",
            outer_seed=1,
            run_dir=run_dir,
            result_tag="x",
            config_path=config,
        )


def test_canary_writes_the_same_sealed_outer_attempt_receipt_as_corpus_supervisor(
    tmp_path: Path,
) -> None:
    experiment_id = canary.CANARY_BATCH[0]
    authority = {
        "evidence_class": "inferential",
        "seed_mode": "varied",
        "seed_authority": {"authority_sha256": "seed-authority"},
        "experiment_config": {"path": "config.yaml", "sha256": "config"},
        "implementation_authorities": [{"path": "worker.py", "sha256": "worker"}],
    }
    run_dir = tmp_path / "attempt_001"
    run_dir.mkdir()
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.write_text("worker output", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    class Finished:
        pid = 12345

        @staticmethod
        def poll() -> int:
            return 0

    handle = canary.WorkerHandle(
        experiment_id=experiment_id,
        process=Finished(),
        create_time=10.0,
        command=["python", "worker.py"],
        run_dir=run_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=io.BytesIO(),
        stderr_handle=io.BytesIO(),
        started_monotonic=0.0,
    )
    plan = canary.CanaryPlan(
        config_path=tmp_path / "campaign.json",
        config={"result_tag": "test"},
        outer_seed=20260801,
        batch=canary.CANARY_BATCH,
        cell_authorities={experiment_id: authority},
        source_snapshot={},
        policy=None,
        preflight={},
    )
    report = {
        "resolved_config": {"path": "resolved.yaml", "sha256": "resolved"},
        "manifest": {"path": "manifest.json", "sha256": "manifest"},
    }

    binding = canary._write_attempt_receipt(
        plan,
        handle,
        returncode=0,
        seconds=1.25,
        stdout_tail="worker output",
        stderr_tail="",
        report=report,
    )
    receipt = json.loads((run_dir / "attempt_receipt.json").read_text(encoding="utf-8"))

    assert binding is not None
    assert binding["self_seal_valid"] is True
    assert receipt["schema"] == corpus.ATTEMPT_SCHEMA
    assert corpus._valid_seal(receipt, "attempt_sha256") is True
    assert receipt["seed_authority"] == authority["seed_authority"]
    assert receipt["manifest"] == report["manifest"]
    assert receipt["worker_report"] == report


def test_owned_signal_refuses_pid_identity_drift_without_calling_killpg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ["python", "worker.py"]

    class Spawned:
        pid = 12345

        @staticmethod
        def poll() -> None:
            return None

    class ReusedPid:
        @staticmethod
        def create_time() -> float:
            return 20.0

        @staticmethod
        def ppid() -> int:
            return os.getpid()

        @staticmethod
        def cmdline() -> list[str]:
            return command

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(canary.psutil, "Process", lambda _pid: ReusedPid())
    monkeypatch.setattr(canary.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    handle = canary.WorkerHandle(
        experiment_id="ex9_slot_attention",
        process=Spawned(),
        create_time=10.0,
        command=command,
        run_dir=tmp_path / "run",
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        stdout_handle=io.BytesIO(),
        stderr_handle=io.BytesIO(),
        started_monotonic=0.0,
    )

    assert canary.signal_owned_process_group(handle, signal.SIGTERM) is False
    assert killed == []


def test_receipt_seal_detects_mutation() -> None:
    receipt = canary._sealed(
        {
            "schema": canary.CANARY_SCHEMA,
            "complete": True,
            "scientific_promotion": False,
        },
        "receipt_sha256",
    )
    assert canary.valid_seal(receipt)

    receipt["complete"] = False
    assert canary.valid_seal(receipt) is False


def test_success_only_atomic_proof_publication_preserves_exact_receipt_bytes(
    tmp_path: Path,
) -> None:
    receipt = canary._sealed(
        {
            "schema": canary.CANARY_SCHEMA,
            "complete": True,
            "scientific_promotion": False,
        },
        "receipt_sha256",
    )
    run_receipt = tmp_path / "runs/generation1/resource_canary/test/resource_canary.json"
    corpus._atomic_json(run_receipt, receipt)
    proof = tmp_path / "proof/GENERATION1_RESOURCE_CANARY.json"

    binding = canary.publish_canonical_proof(
        run_receipt,
        receipt,
        proof,
        repo_root=tmp_path,
    )

    assert binding["byte_equal_to_run_receipt"] is True
    assert binding["receipt_sha256"] == receipt["receipt_sha256"]
    assert proof.read_bytes() == run_receipt.read_bytes()
    with pytest.raises(canary.CanaryRefused, match="canonical path"):
        canary.publish_canonical_proof(
            run_receipt,
            receipt,
            tmp_path / "proof/wrong.json",
            repo_root=tmp_path,
        )
    bad = dict(receipt)
    bad["complete"] = False
    with pytest.raises(canary.CanaryRefused, match="complete self-sealed"):
        canary.publish_canonical_proof(run_receipt, bad, proof, repo_root=tmp_path)
