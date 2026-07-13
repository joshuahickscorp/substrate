from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from mop.config import REPO_ROOT
from mop.studio.local_throttle import load_policy
from mop.studio.policy_overlay import (
    POLICY_MIGRATION_PREVIEW_SCHEMA,
    load_task_overlay,
    render_policy_overlay,
)

OVERLAY_PATH = REPO_ROOT / "configs/campaign/substrate_task_overlay.yaml"
POLICY_PATH = REPO_ROOT / "configs/local_execution_throttle.yaml"
BASELINE_POLICY_PATH = REPO_ROOT / "configs/campaign/local_execution_throttle_policy_baseline_v1.yaml"


def test_overlay_is_additive_and_renders_a_valid_throttle_policy(tmp_path: Path) -> None:
    overlay = load_task_overlay(OVERLAY_PATH, repository_root=REPO_ROOT)
    rendered, preview = render_policy_overlay(BASELINE_POLICY_PATH, overlay)
    rendered_path = tmp_path / "policy.yaml"
    rendered_path.write_text(rendered)
    policy = load_policy(rendered_path)

    assert preview["schema"] == POLICY_MIGRATION_PREVIEW_SCHEMA
    assert preview["scientific_promotion"] is False
    baseline = json.loads((REPO_ROOT / "proof/LOCAL_THROTTLE_POLICY_BASELINE_V1.json").read_text())
    assert preview["safety_contract_sha256"] == baseline["safety_contract_sha256"]
    assert {
        "tools/condense/audit_ladder.py",
        "tools/condense/doctor.py",
        "vendor/strand-quant/target/release/quantize-model",
        "bin/hf download",
        "bin/hf cache verify",
        "computexchange/agent/target/release",
    } <= set(preview["added_known_heavy_markers"])
    assert preview["added_task_ids"] == [
        "edcm1_official_cpu",
        "edcm1_verify_cpu",
        "escs_x0_official_cpu",
        "escs_x0_verify_cpu",
    ]
    assert policy.execution_order["escs_substrate_cpu"] == (
        "p5verify_cpu",
        "edcm1_official_cpu",
        "edcm1_verify_cpu",
        "escs_x0_official_cpu",
        "escs_x0_verify_cpu",
    )
    for task_id in preview["added_task_ids"]:
        task = policy.task(task_id)
        assert task.requires_empty_lanes is True
        assert task.resource_probe is True
        assert task.estimated_unified_memory_gb is None
        assert task.cpu_cores == 1

    assert set(preview["added_known_heavy_markers"]) <= set(policy.monitor["known_heavy_markers"])


def test_overlay_pins_exact_output_and_implementation_authority_arguments() -> None:
    overlay = load_task_overlay(OVERLAY_PATH, repository_root=REPO_ROOT)
    commands = {task_id: tuple(task["command"]) for task_id, task in overlay.tasks.items()}

    assert commands["edcm1_official_cpu"][commands["edcm1_official_cpu"].index("--out") + 1] == (
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json"
    )
    assert commands["edcm1_verify_cpu"][commands["edcm1_verify_cpu"].index("--verification-out") + 1] == (
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json"
    )
    assert commands["escs_x0_official_cpu"][commands["escs_x0_official_cpu"].index("--output") + 1] == (
        "proof/ESCS_X0_EVENT_FORMATION.json"
    )
    assert commands["escs_x0_verify_cpu"][commands["escs_x0_verify_cpu"].index("--verification-out") + 1] == (
        "proof/ESCS_X0_EVENT_FORMATION.verification.json"
    )
    assert (
        "5c64fbb99788b43c51abb66c3d77e68204326183916cecc73a93d3be28adeff8" in commands["edcm1_official_cpu"]
    )
    assert (
        "3ada805ce564141694927de541121e1f57137533df987352302cca4409f36433" in commands["escs_x0_official_cpu"]
    )


def test_new_producers_require_the_exact_quiescent_substrate_preflight() -> None:
    overlay = load_task_overlay(OVERLAY_PATH, repository_root=REPO_ROOT)
    report = json.loads((REPO_ROOT / "proof/ESCS_SUBSTRATE_PREFLIGHT.json").read_text())

    for task_id in ("edcm1_official_cpu", "escs_x0_official_cpu"):
        prerequisites = overlay.tasks[task_id]["prerequisites"]
        matches = [
            requirement
            for requirement in prerequisites
            if requirement["path"] == "proof/ESCS_SUBSTRATE_PREFLIGHT.json"
        ]
        assert len(matches) == 1
        requirement = matches[0]
        assert requirement["schema"] == "mop-escs-substrate-preflight-report/v1"
        assert all(report[key] == value for key, value in requirement["fields"].items())
        assert requirement["fields"]["scaffold_ready"] is True
        assert requirement["fields"]["activation_ready"] is False
        assert requirement["fields"]["scientific_promotion_allowed"] is False


def test_overlay_refuses_baseline_drift_and_task_collision(tmp_path: Path) -> None:
    overlay = load_task_overlay(OVERLAY_PATH, repository_root=REPO_ROOT)
    changed_policy = tmp_path / "changed-policy.yaml"
    changed_policy.write_bytes(BASELINE_POLICY_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="reviewed baseline"):
        render_policy_overlay(changed_policy, overlay)

    raw = yaml.safe_load(OVERLAY_PATH.read_text())
    raw["tasks"]["p5verify_cpu"] = copy.deepcopy(raw["tasks"]["edcm1_verify_cpu"])
    raw["execution_order"]["escs_substrate_cpu"].append("p5verify_cpu_duplicate")
    changed_overlay = tmp_path / "overlay.yaml"
    changed_overlay.write_text(yaml.safe_dump(raw, sort_keys=False))
    collision = load_task_overlay(changed_overlay, repository_root=REPO_ROOT)
    with pytest.raises(ValueError, match="not additive"):
        render_policy_overlay(BASELINE_POLICY_PATH, collision)
