from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from substrate import tangible_next as next


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _r2_authority(schema: str, **payload: object) -> dict:
    body = {"schema": schema, "activation": False, "external_activation": False, **payload}
    body["sha256"] = next.digest(body)
    return body


def _install_r2(root: Path, *, complete: bool) -> None:
    evidence = root / next.R2_EVIDENCE_RELATIVE
    _write(
        evidence / "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json",
        _r2_authority(
            "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT",
            scientific_status="complete" if complete else "not_run",
            actual_wall_hours=24 if complete else 0,
            continuity_passing=complete,
        ),
    )
    _write(
        evidence / "SUBSTRATE_SANDBOX_INDEPENDENT_VERIFICATION.json",
        _r2_authority(
            "SUBSTRATE_SANDBOX_INDEPENDENT_VERIFICATION",
            scientific_status="pass" if complete else "fail",
            independently_verified=complete,
        ),
    )
    _write(
        evidence / "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json",
        _r2_authority(
            "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION",
            outcome="B" if complete else "C",
            H_T12={"status": "tested" if complete else "not_tested"},
        ),
    )
    _write(
        evidence / "SUBSTRATE_SANDBOX_FINAL_STATE.json",
        _r2_authority(
            "SUBSTRATE_SANDBOX_FINAL_STATE",
            outcome="B" if complete else "C",
            longitudinal_hours=24 if complete else 0,
        ),
    )
    _write(
        evidence / "SUBSTRATE_SANDBOX_CLEAN_CLONE.json",
        _r2_authority("SUBSTRATE_SANDBOX_CLEAN_CLONE", all_pass=complete),
    )


def _approved_drafts(root: Path, review: dict) -> None:
    paths = next._paths(root)
    design = next._read_json(paths["design"])
    design["status"] = "approved_after_r2_review"
    design["approval"] = {
        "state": "approved_after_r2_review",
        "r2_review_sha256": review["sha256"],
        "reviewer": "scientific-custodian",
        "approved_at": "2026-07-31T20:00:00+00:00",
    }
    design["custody"]["seed_commitment_sha256"] = hashlib.sha256(b"custodian-secret").hexdigest()
    design["custody"]["isolation_attested"] = True
    design["stimulus_bank"]["selection_state"] = "selected_and_pinned"
    design["stimulus_bank"]["selected_card_ids"] = ["stsc_r2_recomposition"]
    design["adapters"] = {
        "candidate_command": "substrate.next_candidate:run@v1",
        "control_command": "substrate.next_control:run@v1",
        "evaluator_command": "substrate.next_evaluator:run@v1",
        "contract": "docs/archive/staging/tangible_sandbox/NEXT_LAUNCH_RUNBOOK.md#adapter-contract",
    }
    _write(paths["design"], design)
    data = next._read_json(paths["data"])
    for card in data["cards"]:
        if card["id"] == "stsc_r2_recomposition":
            card.update(
                {
                    "state": "accepted",
                    "selected": True,
                    "license_or_terms_reviewed": True,
                    "hashes_pinned": True,
                    "candidate_control_parity": True,
                    "evaluator_only_split": True,
                    "additional_bytes": 0,
                }
            )
    _write(paths["data"], data)
    calibration = next._read_json(paths["calibration"])
    calibration.update({"repetitions": 2, "unit_count": 4, "hash_rounds": 1, "receipt_bytes": 64, "max_slowdown_ratio": 100})
    _write(paths["calibration"], calibration)


def test_live_r2_blocks_review_and_deterministic_state(tmp_path: Path) -> None:
    next.bootstrap(tmp_path)
    live = tmp_path / next.R2_LIVE_STATE_RELATIVE
    _write(live, {"complete": False, "elapsed_seconds": 10, "target_seconds": 86_400, "events_emitted": [0]})

    assert next.status(tmp_path)["deterministic_next_state"] == "await_r2_review"
    with pytest.raises(next.Refused, match="still live"):
        next.review_r2(tmp_path)


def test_stale_r2_evidence_is_recorded_as_invalid_not_completion(tmp_path: Path) -> None:
    next.bootstrap(tmp_path)
    _install_r2(tmp_path, complete=False)

    review = next.review_r2(tmp_path)

    assert review["valid"] is False
    assert review["next_state"] == "repair_diagnosis"
    assert next.status(tmp_path)["deterministic_next_state"] == "repair_diagnosis"


def test_review_to_sealed_preflight_and_custody_handoff(tmp_path: Path) -> None:
    next.bootstrap(tmp_path)
    _install_r2(tmp_path, complete=True)
    review = next.review_r2(tmp_path)
    assert review["valid"] is True

    _approved_drafts(tmp_path, review)
    sealed = next.seal_design(tmp_path)
    assert sealed["scientific_status"] == "sealed_before_calibration"
    calibration = next.run_calibration(tmp_path)
    assert calibration["admitted"] is True
    assert calibration["checks"]["receipt_invariant_across_widths"] is True
    assert calibration["checks"]["distinct_run_roots"] is True

    admission = next.preflight(tmp_path)
    assert admission["admitted"] is True
    handoff = next.prepare(tmp_path)
    evaluator = tmp_path / handoff["roots"]["evaluator_only"]
    assert handoff["prepared_not_launched"] is True
    assert stat.S_IMODE(evaluator.stat().st_mode) == 0o700

    builder = tmp_path / handoff["roots"]["builder_visible"]
    task_rows = []
    for hour, event, _ in sealed["design"]["duration"]["schedule"]:
        task_file = builder / "tasks" / f"task-{hour:02d}.json"
        _write(task_file, {"task_id": f"task-{hour:02d}", "activation": False})
        task_rows.append(
            {
                "scheduled_hour": hour,
                "event": event,
                "task_id": f"task-{hour:02d}",
                "builder_task": str(task_file.relative_to(builder)),
            }
        )
    task_manifest = builder / "TASK_MANIFEST.json"
    _write(
        task_manifest,
        {
            "schema": "SUBSTRATE_TANGIBLE_BLIND_SHADOW_TASK_MANIFEST/v1",
            "run_id": handoff["run_id"],
            "sealed_design_sha256": sealed["sha256"],
            "tasks": task_rows,
            "activation": False,
        },
    )
    answer_manifest = evaluator / "ANSWER_MANIFEST.json"
    _write(
        answer_manifest,
        {
            "schema": "SUBSTRATE_TANGIBLE_BLIND_SHADOW_ANSWER_MANIFEST/v1",
            "run_id": handoff["run_id"],
            "sealed_design_sha256": sealed["sha256"],
            "answers": {row["task_id"]: "custodian-only" for row in task_rows},
            "activation": False,
        },
    )
    seed = tmp_path.parent / f"{tmp_path.name}-custodian-seed"
    seed.write_bytes(b"custodian-secret")
    custody = next.seal_custody(
        tmp_path,
        handoff_path=tmp_path / "runs/substrate/tangible_next_launch/blind-shadow" / handoff["run_id"] / "CUSTODY_HANDOFF.json",
        task_manifest_path=task_manifest,
        answer_manifest_path=answer_manifest,
        seed_file=seed,
    )
    assert custody["scientific_status"] == "sealed_before_detached_launch"
    job = next._launchd_job("org.substrate.test", Path("/tmp/manifest.json"), tmp_path, Path("/tmp/out"), Path("/tmp/err"))
    assert job["KeepAlive"] is False
    assert "supervised-run" in job["ProgramArguments"]


def test_adapter_contract_accepts_matching_receipt_and_rejects_wrong_task(tmp_path: Path) -> None:
    next.bootstrap(tmp_path)
    request_path = tmp_path / "request.json"
    receipt_path = tmp_path / "receipt.json"
    request = {
        "schema": "SUBSTRATE_TANGIBLE_ADAPTER_REQUEST/v1",
        "role": "candidate",
        "run_id": "shadow-test",
        "task_id": "task-1",
        "input_manifest_sha256": "b" * 64,
        "builder_visible_task": "builder_visible/tasks/task-1.json",
        "receipt_path": "candidate_workspace/task-1-receipt.json",
        "activation": False,
    }
    receipt = {
        "schema": "SUBSTRATE_TANGIBLE_ADAPTER_RECEIPT/v1",
        "role": "candidate",
        "run_id": "shadow-test",
        "task_id": "task-1",
        "input_manifest_sha256": "b" * 64,
        "output_artifacts": [],
        "elapsed_seconds": 0.1,
        "resource_usage": {"cpu_seconds": 0.01},
        "activation": False,
    }
    _write(request_path, request)
    _write(receipt_path, receipt)
    assert next.validate_receipt(tmp_path, request_path, receipt_path)["all_pass"] is True

    receipt["task_id"] = "wrong-task"
    _write(receipt_path, receipt)
    assert next.validate_receipt(tmp_path, request_path, receipt_path)["all_pass"] is False
