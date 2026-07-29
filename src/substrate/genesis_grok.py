"""Grok swarm ledger for Substrate Cognitive Material Genesis.

Every credited role is one independent on-device Grok invocation. The wrapper
artifacts are validated before a role may be credited: the executed prompt must
equal the supplied contract byte for byte, the run must have terminated
normally under a read-only sandbox for review rounds, and the emitted review
object must carry the assigned role and round, a feasibility grade, at least
one blocker and at least one falsifier.

Grok opinions are proposals. Evidence decides. Nothing recorded here may set
activation true or relax a threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_io as io

LEDGER = "SUBSTRATE_GENESIS_GROK_INVOCATION_LEDGER.json"
REVIEW = "SUBSTRATE_GENESIS_GROK_REVIEW.json"

REVIEW_KEYS = (
    "role",
    "round",
    "feasibility_out_of_20",
    "summary",
    "blockers",
    "falsification_evidence",
    "concrete_tests",
    "recommendations",
    "activation",
)

BLOCKER_KEYS = ("id", "severity", "claim", "evidence", "concrete_test")
FALSIFIER_KEYS = ("hypothesis", "cheapest_falsifier", "expected_result_if_claim_is_false")
TEST_KEYS = ("name", "asserts", "fails_when")
SEVERITIES = ("blocking", "major", "minor")

READ_ONLY_ROUNDS = frozenset(C.REVIEW_ROUNDS)


def _terminal_remainder_is_empty(remainder: str) -> bool:
    """The review object may be closed by a markdown fence and nothing else."""
    return remainder.strip().strip("`") == ""


def _single_review_object(text: str) -> tuple[dict[str, Any], str]:
    """Extract the single terminal review object from a Grok text stream."""
    decoder = json.JSONDecoder()
    found: list[tuple[dict[str, Any], str]] = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not {"role", "round", "feasibility_out_of_20"} <= set(value):
            continue
        if not _terminal_remainder_is_empty(text[end:]):
            continue
        found.append((value, text[:offset]))
    if len(found) != 1:
        raise io.Refused(f"expected exactly one terminal Grok review object, found {len(found)}")
    return found[0]


def _require_rows(value: Any, keys: tuple[str, ...], label: str, *, minimum: int) -> None:
    if not isinstance(value, list) or len(value) < minimum:
        raise io.Refused(f"Grok review {label} must be a list with at least {minimum} entries")
    for row in value:
        if not isinstance(row, dict):
            raise io.Refused(f"Grok review {label} entry is not an object")
        missing = [key for key in keys if key not in row]
        if missing:
            raise io.Refused(f"Grok review {label} entry missing {', '.join(missing)}")
        for key in keys:
            if not isinstance(row[key], str) or not row[key].strip():
                raise io.Refused(f"Grok review {label} entry has an empty {key}")


def validate_review(output: Any, *, expected_role: str, expected_round: str) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise io.Refused("Grok review output is not an object")
    missing = [key for key in REVIEW_KEYS if key not in output]
    if missing:
        raise io.Refused(f"Grok review missing required keys: {', '.join(missing)}")
    if output["role"] != expected_role:
        raise io.Refused(f"Grok review role {output['role']!r} does not match assigned {expected_role!r}")
    if output["round"] != expected_round:
        raise io.Refused(f"Grok review round {output['round']!r} does not match assigned {expected_round!r}")
    if expected_role not in C.REVIEW_CELLS:
        raise io.Refused(f"unknown genesis review cell {expected_role!r}")
    if expected_round not in READ_ONLY_ROUNDS:
        raise io.Refused(f"unknown genesis review round {expected_round!r}")
    grade = output["feasibility_out_of_20"]
    if not isinstance(grade, int) or isinstance(grade, bool) or not 0 <= grade <= 20:
        raise io.Refused("feasibility_out_of_20 must be an integer between 0 and 20")
    if not isinstance(output["summary"], str) or not output["summary"].strip():
        raise io.Refused("Grok review summary is empty")
    _require_rows(output["blockers"], BLOCKER_KEYS, "blockers", minimum=1)
    _require_rows(output["falsification_evidence"], FALSIFIER_KEYS, "falsification_evidence", minimum=1)
    _require_rows(output["concrete_tests"], TEST_KEYS, "concrete_tests", minimum=1)
    for blocker in output["blockers"]:
        if blocker["severity"] not in SEVERITIES:
            raise io.Refused(f"unknown blocker severity {blocker['severity']!r}")
    recommendations = output["recommendations"]
    if not isinstance(recommendations, list) or not recommendations:
        raise io.Refused("Grok review recommendations must be a non-empty list")
    if any(not isinstance(entry, str) or not entry.strip() for entry in recommendations):
        raise io.Refused("Grok review recommendations must be non-empty strings")
    if output["activation"] is not False:
        raise io.Refused("Grok review set activation true")
    if io.contains_true_activation(output):
        raise io.Refused("Grok review contains a true activation key")
    return output


def invocation_record(
    task_directory: Path,
    contract_path: Path,
    *,
    expected_role: str,
    expected_round: str,
    expected_repository: Path | None = None,
) -> dict[str, Any]:
    """Validate one on-device Grok invocation and build its ledger row."""
    task_directory = task_directory.resolve()
    contract_path = contract_path.resolve()
    required = {
        "metadata": task_directory / "metadata.json",
        "output": task_directory / "grok-output.json",
        "status": task_directory / "status",
        "exit_code": task_directory / "exit_code",
        "task": task_directory / "task.md",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise io.Refused(f"Grok task is missing artifacts: {', '.join(missing)}")
    metadata = json.loads(required["metadata"].read_text())
    envelope = json.loads(required["output"].read_text())
    prompt = required["task"].read_text()
    if prompt != contract_path.read_text():
        raise io.Refused("executed Grok task does not match the supplied contract byte for byte")
    model_usage = envelope.get("modelUsage")
    repository = (expected_repository or io.ROOT).resolve()
    checks = {
        "completed_status": required["status"].read_text().strip() == "done",
        "zero_exit": required["exit_code"].read_text().strip() == "0",
        "read_only_sandbox": metadata.get("sandbox") == "read-only",
        "repository": Path(str(metadata.get("repo", ""))).resolve() == repository,
        "session_identity": metadata.get("session_id") == envelope.get("sessionId"),
        "normal_stop": envelope.get("stopReason") == "EndTurn",
        "single_model": isinstance(model_usage, dict) and len(model_usage) == 1,
        "wrapper_model": metadata.get("model") == "grok-4.5",
        "turns_observed": isinstance(envelope.get("num_turns"), int) and envelope["num_turns"] > 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise io.Refused(f"Grok task validation failed: {', '.join(failed)}")
    output, prefix = _single_review_object(str(envelope.get("text", "")))
    validate_review(output, expected_role=expected_role, expected_round=expected_round)
    return {
        "invocation_id": str(metadata["task_id"]),
        "role": expected_role,
        "round": expected_round,
        "prompt_digest": io.digest(prompt),
        "contract_digest": io.file_digest(contract_path),
        "model_identity": next(iter(model_usage)),
        "wrapper_model": metadata["model"],
        "observed_at": metadata.get("started_at"),
        "execution_mode": metadata.get("mode"),
        "sandbox": metadata.get("sandbox"),
        "transport": {
            "source": "on_device_grok_cli",
            "session_id": envelope["sessionId"],
            "request_id": envelope.get("requestId"),
            "stop_reason": envelope["stopReason"],
            "num_turns": envelope["num_turns"],
            "usage": envelope.get("usage"),
            "non_json_prefix_digest": io.digest(prefix) if prefix.strip() else None,
        },
        "output": output,
        "output_digest": io.digest(output),
        "feasibility_out_of_20": output["feasibility_out_of_20"],
        "blocking_defect_count": sum(1 for row in output["blockers"] if row["severity"] == "blocking"),
        "dispositions": [],
        "activation": False,
    }


def load_ledger() -> dict[str, Any]:
    existing = io.read_optional(LEDGER)
    if existing is not None:
        return existing
    return io.authority(
        "substrate-genesis-grok-ledger/v1",
        {"invocations": [], "distinct_role_count": 0, "rounds_observed": []},
    )


def record(row: dict[str, Any]) -> dict[str, Any]:
    """Append one validated invocation, refusing duplicate role credit."""
    ledger = load_ledger()
    invocations = list(ledger.get("invocations", []))
    for existing in invocations:
        if existing["invocation_id"] == row["invocation_id"]:
            raise io.Refused(f"invocation {row['invocation_id']} is already recorded")
        if existing["role"] == row["role"] and existing["round"] == row["round"]:
            raise io.Refused(f"role {row['role']!r} already credited in round {row['round']!r}")
    invocations.append(row)
    invocations.sort(key=lambda entry: (entry["round"], entry["role"]))
    roles = sorted({entry["role"] for entry in invocations})
    rounds = sorted({entry["round"] for entry in invocations})
    document = io.authority(
        "substrate-genesis-grok-ledger/v1",
        {
            "invocations": invocations,
            "distinct_role_count": len(roles),
            "distinct_roles": roles,
            "rounds_observed": rounds,
            "minimum_roles": C.GROK_MINIMUM_ROLES,
            "preferred_roles": C.GROK_PREFERRED_ROLES,
            "minimum_met": len(roles) >= C.GROK_MINIMUM_ROLES,
            "preferred_met": len(roles) >= C.GROK_PREFERRED_ROLES,
        },
    )
    io.write_json(io.EVIDENCE / LEDGER, document)
    return document


def summary() -> dict[str, Any]:
    ledger = load_ledger()
    invocations = ledger.get("invocations", [])
    grades = [entry["feasibility_out_of_20"] for entry in invocations]
    blocking = [
        {"role": entry["role"], "round": entry["round"], **blocker}
        for entry in invocations
        for blocker in entry["output"]["blockers"]
        if blocker["severity"] == "blocking"
    ]
    return {
        "invocation_count": len(invocations),
        "distinct_role_count": ledger.get("distinct_role_count", 0),
        "rounds_observed": ledger.get("rounds_observed", []),
        "minimum_met": len(set(entry["role"] for entry in invocations)) >= C.GROK_MINIMUM_ROLES,
        "preferred_met": len(set(entry["role"] for entry in invocations)) >= C.GROK_PREFERRED_ROLES,
        "feasibility_min": min(grades) if grades else None,
        "feasibility_max": max(grades) if grades else None,
        "feasibility_median": sorted(grades)[len(grades) // 2] if grades else None,
        "blocking_defect_count": len(blocking),
        "blocking_defects": blocking,
        "activation": False,
    }
