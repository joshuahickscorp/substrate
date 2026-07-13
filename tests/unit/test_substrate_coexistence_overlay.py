from __future__ import annotations

import hashlib

from mop.config import REPO_ROOT
from mop.studio.local_throttle import EXTERNAL_COEXISTENCE_PROFILE, load_policy
from mop.studio.policy_overlay import load_task_overlay, render_policy_overlay

BASELINE = REPO_ROOT / "configs/campaign/local_execution_throttle_policy_baseline_v1.yaml"
HISTORICAL_OVERLAY = REPO_ROOT / "configs/campaign/substrate_task_overlay.yaml"
COEXISTENCE_OVERLAY = REPO_ROOT / "configs/campaign/substrate_coexistence_task_overlay.yaml"
LIVE_POLICY = REPO_ROOT / "configs/local_execution_throttle.yaml"
TASKS = (
    "edcm1_official_cpu",
    "edcm1_verify_cpu",
    "escs_x0_official_cpu",
    "escs_x0_verify_cpu",
)
PREFIX = (
    "/usr/sbin/taskpolicy",
    "-b",
    "-d",
    "throttle",
    "-c",
    "background",
    "-m",
    "4096",
    "-P",
    "kill",
    "/usr/bin/env",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "VECLIB_MAXIMUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
)


def test_coexistence_overlay_exactly_reproduces_the_live_policy() -> None:
    overlay = load_task_overlay(COEXISTENCE_OVERLAY, repository_root=REPO_ROOT)
    rendered, preview = render_policy_overlay(BASELINE, overlay)

    assert rendered.encode() == LIVE_POLICY.read_bytes()
    assert preview["new_policy_sha256"] == hashlib.sha256(LIVE_POLICY.read_bytes()).hexdigest()
    assert hashlib.sha256(HISTORICAL_OVERLAY.read_bytes()).hexdigest() == (
        "023852c5d044c2da6275d55d4bcaa186ecc1bda98907ac6bb613299a1fb255b7"
    )


def test_coexistence_tasks_are_exactly_bounded_and_seed_producers_yield() -> None:
    overlay = load_task_overlay(COEXISTENCE_OVERLAY, repository_root=REPO_ROOT)
    policy = load_policy(LIVE_POLICY)

    for task_id in TASKS:
        declaration = overlay.tasks[task_id]
        task = policy.task(task_id)
        assert declaration["external_coexistence"] == EXTERNAL_COEXISTENCE_PROFILE
        assert task.command[: len(PREFIX)] == PREFIX
        assert task.lane == "cpu"
        assert task.accelerator == "none"
        assert task.cpu_cores == 1
        assert task.estimated_unified_memory_gb == 4096 * 1024 * 1024 / 1e9
        assert task.requires_empty_lanes is True
        assert task.resource_probe is False

    assert overlay.tasks["edcm1_official_cpu"]["max_invocations_per_run"] == 1
    assert overlay.tasks["escs_x0_official_cpu"]["max_invocations_per_run"] == 1
    assert "max_invocations_per_run" not in overlay.tasks["edcm1_verify_cpu"]
    assert "max_invocations_per_run" not in overlay.tasks["escs_x0_verify_cpu"]
