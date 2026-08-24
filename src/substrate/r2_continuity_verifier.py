"""Independent, scope-limited verifier for the completed R2 continuity lane.

The historic R2 terminal state is an Outcome-C record.  A later repaired,
launchd-owned 24-hour continuity lane must not overwrite or relabel that
record.  This verifier therefore binds the historical terminal document as
immutable context and independently checks only the later continuity lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from substrate import sandbox_config as config

PROGRAM = "substrate-tangible-sandbox-r2"
EVIDENCE = Path("evidence/substrate/tangible_sandbox")
RUNS = Path("runs/substrate/tangible_sandbox")


class Refused(RuntimeError):
    """The R2 continuity handoff has incomplete, changed, or invalid evidence."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _contains_true_activation(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (key.casefold() in {"activation", "external_activation"} and child is not False)
            or _contains_true_activation(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_true_activation(child) for child in value)
    return False


def _read_json(path: Path, *, require_digest: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} must contain a JSON object")
    if _contains_true_activation(value):
        raise Refused(f"{path} enables activation")
    claimed = value.get("sha256")
    if require_digest and not isinstance(claimed, str):
        raise Refused(f"{path} is missing a self-digest")
    if isinstance(claimed, str):
        unsigned = dict(value)
        unsigned.pop("sha256")
        if digest(unsigned) != claimed:
            raise Refused(f"{path} self-digest mismatch")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Refused(f"refusing to overwrite {path}")
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _sealed(payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema": "SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION/v1",
        "program": PROGRAM,
        "scientific_status": "pass",
        "independently_verified": True,
        "verification_scope": (
            "R2 24-hour continuity lane only; this receipt does not overwrite, "
            "reclassify, or pool the historical terminal Outcome-C evidence."
        ),
        **payload,
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
    }
    body["sha256"] = digest(body)
    return body


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _relative(root: Path, path: Path) -> str:
    if not _inside(root, path):
        raise Refused(f"path escapes repository root: {path}")
    return str(path.resolve().relative_to(root.resolve()))


def _resolve_relative(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise Refused(f"{label} must be a non-empty root-relative path")
    path = (root / raw).resolve()
    if not _inside(root, path):
        raise Refused(f"{label} escapes repository root")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read trace {path}: {error}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise Refused("trace must contain non-empty JSON-object rows")
    return rows


def _sealed_ref(root: Path, path: Path, document: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "file_sha256": file_digest(path),
        "sha256": document["sha256"],
        "schema": document.get("schema"),
    }


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row["work_receipt"].get(key)) for row in rows)


def _check_trace(root: Path, result: dict[str, Any], state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_path = _resolve_relative(root, result.get("trace"), label="longitudinal result.trace")
    if not trace_path.is_file() or result.get("trace_sha256") != file_digest(trace_path):
        raise Refused("longitudinal trace is missing or digest-drifted")
    rows = _read_jsonl(trace_path)
    expected_schedule = [(hour, event, activity) for hour, event, activity in config.LONGITUDINAL_SCHEDULE]
    observed_schedule = [
        (row.get("scheduled_hour"), row.get("event"), row.get("activity"))
        for row in rows
    ]
    if observed_schedule != expected_schedule:
        raise Refused("trace schedule differs from the sealed R2 continuity schedule")
    if state.get("complete") is not True or state.get("events_emitted") != [hour for hour, _, _ in expected_schedule]:
        raise Refused("terminal R2 state is not complete with the exact scheduled events")
    if state.get("checkpoint_count") != len(expected_schedule):
        raise Refused("terminal R2 state checkpoint count does not match the schedule")
    for row in rows:
        receipt = row.get("work_receipt")
        if not isinstance(receipt, dict):
            raise Refused("trace row lacks a work receipt")
        checkpoint = _resolve_relative(root, receipt.get("checkpoint"), label="trace checkpoint")
        if not checkpoint.is_file():
            raise Refused(f"trace checkpoint is missing: {checkpoint}")
        if row.get("activation") is not False or _contains_true_activation(receipt):
            raise Refused("trace contains an activation field that is not false")
    counts = {
        "process_restarts": _count(rows, "restart"),
        "checkpoints": len(rows),
        "model_replacements": _count(rows, "model_replacement"),
        "tool_or_body_changes": _count(rows, "tool_or_body_change"),
        "sensor_interruptions": _count(rows, "sensor_interruption"),
        "human_corrections": _count(rows, "human_correction"),
        "returns_to_old_work": _count(rows, "return_to_old_work"),
        "new_tasks_requiring_earlier_history": _count(rows, "new_task_requires_earlier_history"),
    }
    if any(result.get(name) != value for name, value in counts.items()):
        raise Refused("longitudinal result counts do not recompute from the trace")
    if any(counts[name] < minimum for name, minimum in config.LONGITUDINAL_MINIMUMS.items()):
        raise Refused("trace misses an R2 continuity minimum")
    corrections = [row["work_receipt"] for row in rows if row["work_receipt"].get("human_correction")]
    sensors = [row["work_receipt"] for row in rows if row["work_receipt"].get("sensor_interruption")]
    tools = [row["work_receipt"] for row in rows if row["work_receipt"].get("tool_or_body_change")]
    if not all(isinstance(row.get("correction_receipt"), dict) for row in corrections):
        raise Refused("human correction receipt was lost from the compact trace")
    if not all(isinstance(row.get("sensor_interruption"), dict) for row in sensors):
        raise Refused("sensor interruption receipt was reduced to a boolean or missing")
    if not all(row.get("tool_body_change", {}).get("receipt", {}).get("operation") == "video_frame_decode" for row in tools):
        raise Refused("tool/body-change trace lacks the required video decoder receipt")
    history_rows = [
        row
        for row in rows
        if row["work_receipt"].get("return_to_old_work")
        or row["work_receipt"].get("new_task_requires_earlier_history")
    ]
    if not all(row["work_receipt"].get("history_required") is True for row in history_rows):
        raise Refused("history-dependent trace work did not bind to prior state")
    return rows, {"trace": trace_path, "counts": counts, "scheduled_hours": [hour for hour, _, _ in expected_schedule]}


def verify(root: Path, output_path: Path) -> dict[str, Any]:
    """Recompute a completed R2 continuity lane and write a new handoff receipt."""
    evidence = root / EVIDENCE
    runs = root / RUNS
    result_path = evidence / "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json"
    final_state_path = evidence / "SUBSTRATE_SANDBOX_FINAL_STATE.json"
    state_path = runs / "longitudinal/state.json"
    result = _read_json(result_path, require_digest=True)
    historical_final_state = _read_json(final_state_path, require_digest=True)
    state = _read_json(state_path)
    if result.get("schema") != "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT" or result.get("program") != PROGRAM:
        raise Refused("unexpected R2 longitudinal-result schema or program")
    if result.get("scientific_status") != "complete" or result.get("continuity_passing") is not True:
        raise Refused("R2 longitudinal result is not terminally complete and passing")
    if float(result.get("actual_wall_hours", 0)) < 24 or float(result.get("actual_elapsed_seconds", 0)) < 24 * 3600:
        raise Refused("R2 continuity lane did not complete a full 24 hours")
    if historical_final_state.get("schema") != "SUBSTRATE_SANDBOX_FINAL_STATE" or historical_final_state.get("program") != PROGRAM:
        raise Refused("unexpected historical R2 final-state schema or program")
    rows, trace = _check_trace(root, result, state)
    repair_path = _resolve_relative(root, result.get("continuity_repair_seal"), label="continuity_repair_seal")
    repair = _read_json(repair_path, require_digest=True)
    if repair.get("sha256") != result.get("continuity_repair_seal_sha256"):
        raise Refused("continuity repair seal does not match the terminal result")
    supervision = result.get("supervision")
    if not isinstance(supervision, dict):
        raise Refused("longitudinal result lacks supervision binding")
    manifest_path = _resolve_relative(root, supervision.get("manifest"), label="supervision manifest")
    manifest = _read_json(manifest_path, require_digest=True)
    if manifest.get("sha256") != supervision.get("manifest_sha256") or manifest.get("run_id") != supervision.get("run_id"):
        raise Refused("supervision manifest does not match longitudinal result")
    supervisor_result_path = manifest_path.parent / "supervisor-result.json"
    supervisor_result = _read_json(supervisor_result_path, require_digest=True)
    if (
        supervisor_result.get("run_id") != supervision.get("run_id")
        or supervisor_result.get("status") != "worker_complete"
        or supervisor_result.get("worker_returncode") != 0
    ):
        raise Refused("R2 launchd supervisor did not record a clean worker completion")
    if state.get("supervision", {}).get("run_id") != supervision.get("run_id"):
        raise Refused("terminal state is not bound to the supervised R2 run")
    checks = {
        "result_complete": True,
        "full_24_hours": True,
        "trace_schedule_exact": True,
        "trace_digest_matches": True,
        "counts_recomputed": True,
        "minimums_met": True,
        "trace_level_intervention_receipts": True,
        "checkpoint_files_present": True,
        "state_complete": True,
        "repair_seal_bound": True,
        "supervision_manifest_bound": True,
        "supervised_worker_exit_zero": True,
        "historical_final_state_preserved": True,
    }
    document = _sealed(
        {
            "result": _sealed_ref(root, result_path, result),
            "trace": {"path": _relative(root, trace["trace"]), "file_sha256": file_digest(trace["trace"]), "rows": len(rows)},
            "terminal_state": {"path": _relative(root, state_path), "file_sha256": file_digest(state_path), "complete": state["complete"]},
            "repair_seal": _sealed_ref(root, repair_path, repair),
            "supervision_manifest": _sealed_ref(root, manifest_path, manifest),
            "supervision_result": _sealed_ref(root, supervisor_result_path, supervisor_result),
            "historical_final_state": {
                **_sealed_ref(root, final_state_path, historical_final_state),
                "outcome": historical_final_state.get("outcome"),
                "classification": historical_final_state.get("classification"),
                "longitudinal_hours": historical_final_state.get("longitudinal_hours"),
                "superseded": False,
            },
            "recomputed": {
                "counts": trace["counts"],
                "scheduled_hours": trace["scheduled_hours"],
                "actual_wall_hours": result["actual_wall_hours"],
                "actual_elapsed_seconds": result["actual_elapsed_seconds"],
                "run_id": supervision["run_id"],
            },
            "checks": checks,
            "verifier_source_sha256": file_digest(Path(__file__)),
        }
    )
    _write_json(output_path, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the R2 continuity handoff without rewriting historical R2 evidence")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not _inside(root, output):
        print(json.dumps({"refused": "output path must stay inside repository root", "activation": False}, sort_keys=True))
        return 2
    try:
        result = verify(root, output)
    except Refused as error:
        print(json.dumps({"refused": str(error), "activation": False}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
