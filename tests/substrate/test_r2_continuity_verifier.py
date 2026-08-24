"""Tests for the non-destructive R2 continuity handoff verifier."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate import r2_continuity_verifier as verifier
from substrate import sandbox_config as config


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(schema: str, payload: dict, *, status: str = "complete") -> dict:
    body = {
        "schema": schema,
        "program": verifier.PROGRAM,
        "scientific_status": status,
        **payload,
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
    }
    body["sha256"] = verifier.digest(body)
    return body


def _install_completed_lane(root: Path) -> dict[str, Path]:
    runs = root / verifier.RUNS
    evidence = root / verifier.EVIDENCE
    run_id = "r2-fixture"
    repair_path = evidence / "SUBSTRATE_SANDBOX_TRACE_RECEIPT_REPAIR_LAUNCH_SEAL.json"
    repair = _sealed("SUBSTRATE_SANDBOX_TRACE_RECEIPT_REPAIR_SEAL", {"fixed": True})
    _write(repair_path, repair)
    trace_path = runs / "longitudinal/trace.jsonl"
    rows = []
    for hour, event, activity in config.LONGITUDINAL_SCHEDULE:
        checkpoint = runs / f"longitudinal/workspace/checkpoints/checkpoint-{hour:02d}.json"
        _write(checkpoint, {"hour": hour, "activation": False})
        receipt = {
            "checkpoint": str(checkpoint.relative_to(root)),
            "restart": event.startswith("restart_"),
            "model_replacement": event == "model_replacement",
            "tool_or_body_change": event == "restart_2_tool_body_change",
            "sensor_interruption": {"fallback": "telemetry"} if event == "sensor_interruption" else None,
            "human_correction": event.startswith("human_correction"),
            "return_to_old_work": activity.startswith("return_old_work"),
            "new_task_requires_earlier_history": "requires_earlier_history" in activity,
            "history_required": True,
        }
        if receipt["human_correction"]:
            receipt["correction_receipt"] = {"directive": "activation: false"}
        if receipt["tool_or_body_change"]:
            receipt["tool_body_change"] = {"receipt": {"operation": "video_frame_decode"}}
        rows.append(
            {
                "scheduled_hour": hour,
                "event": event,
                "activity": activity,
                "work_receipt": receipt,
                "activation": False,
            }
        )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    counts = {
        "process_restarts": sum(row["work_receipt"]["restart"] for row in rows),
        "checkpoints": len(rows),
        "model_replacements": sum(row["work_receipt"]["model_replacement"] for row in rows),
        "tool_or_body_changes": sum(row["work_receipt"]["tool_or_body_change"] for row in rows),
        "sensor_interruptions": sum(bool(row["work_receipt"]["sensor_interruption"]) for row in rows),
        "human_corrections": sum(row["work_receipt"]["human_correction"] for row in rows),
        "returns_to_old_work": sum(row["work_receipt"]["return_to_old_work"] for row in rows),
        "new_tasks_requiring_earlier_history": sum(row["work_receipt"]["new_task_requires_earlier_history"] for row in rows),
    }
    manifest_path = runs / f"longitudinal-supervision/{run_id}/manifest.json"
    manifest = _sealed("SUBSTRATE_SANDBOX_LONGITUDINAL_SUPERVISION_MANIFEST", {"run_id": run_id})
    _write(manifest_path, manifest)
    supervisor_result_path = manifest_path.parent / "supervisor-result.json"
    supervisor_result = _sealed(
        "SUBSTRATE_SANDBOX_LONGITUDINAL_SUPERVISION_RESULT",
        {"run_id": run_id, "status": "worker_complete", "worker_returncode": 0},
        status="worker_complete",
    )
    _write(supervisor_result_path, supervisor_result)
    state_path = runs / "longitudinal/state.json"
    _write(
        state_path,
        {
            "complete": True,
            "events_emitted": [hour for hour, _, _ in config.LONGITUDINAL_SCHEDULE],
            "checkpoint_count": len(rows),
            "supervision": {"run_id": run_id},
            "activation": False,
        },
    )
    result_path = evidence / "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json"
    result = _sealed(
        "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT",
        {
            "actual_wall_hours": 24.0001,
            "actual_elapsed_seconds": 86400.4,
            "continuity_passing": True,
            **counts,
            "trace": str(trace_path.relative_to(root)),
            "trace_sha256": verifier.file_digest(trace_path),
            "continuity_repair_seal": str(repair_path.relative_to(root)),
            "continuity_repair_seal_sha256": repair["sha256"],
            "supervision": {
                "run_id": run_id,
                "manifest": str(manifest_path.relative_to(root)),
                "manifest_sha256": manifest["sha256"],
            },
        },
    )
    _write(result_path, result)
    final_state_path = evidence / "SUBSTRATE_SANDBOX_FINAL_STATE.json"
    historical = _sealed(
        "SUBSTRATE_SANDBOX_FINAL_STATE",
        {"outcome": "C", "classification": "terminal_tangible_sandbox_null", "longitudinal_hours": 0},
        status="terminal_evidence_prepared",
    )
    _write(final_state_path, historical)
    return {
        "result": result_path,
        "trace": trace_path,
        "final_state": final_state_path,
        "output": evidence / "SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION.json",
    }


def test_scope_limited_verifier_preserves_historical_outcome_c(tmp_path: Path) -> None:
    paths = _install_completed_lane(tmp_path)

    receipt = verifier.verify(tmp_path, paths["output"])

    assert receipt["scientific_status"] == "pass"
    assert receipt["independently_verified"] is True
    assert receipt["historical_final_state"]["outcome"] == "C"
    assert receipt["historical_final_state"]["longitudinal_hours"] == 0
    assert receipt["historical_final_state"]["superseded"] is False
    assert verifier._read_json(paths["output"], require_digest=True)["sha256"] == receipt["sha256"]
    assert verifier._read_json(paths["final_state"], require_digest=True)["outcome"] == "C"


def test_verifier_refuses_trace_drift_and_never_overwrites_a_receipt(tmp_path: Path) -> None:
    paths = _install_completed_lane(tmp_path)
    rows = paths["trace"].read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0])
    changed["event"] = "forged"
    rows[0] = json.dumps(changed, sort_keys=True)
    paths["trace"].write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(verifier.Refused, match="digest-drifted"):
        verifier.verify(tmp_path, paths["output"])

    _install_completed_lane(tmp_path / "fresh")
    fresh_output = tmp_path / "fresh" / verifier.EVIDENCE / "SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION.json"
    verifier.verify(tmp_path / "fresh", fresh_output)
    with pytest.raises(verifier.Refused, match="refusing to overwrite"):
        verifier.verify(tmp_path / "fresh", fresh_output)
