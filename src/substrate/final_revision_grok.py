"""Distinct, content-addressable Grok review prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from substrate import final_revision_config as C
from substrate import final_revision_io as io

PREFREEZE_ROUND_ASSIGNMENT = {
    **{role: "blind_independent_review" for role in C.REVIEW_CELLS[0:8]},
    **{role: "cross_examination" for role in C.REVIEW_CELLS[8:12]},
    **{role: "architecture_proposals" for role in C.REVIEW_CELLS[12:16]},
    **{role: "test_and_baseline_proposals" for role in C.REVIEW_CELLS[16:20]},
    **{role: "code_and_implementation_review" for role in C.REVIEW_CELLS[20:24]},
    **{role: "post_pilot_review" for role in C.REVIEW_CELLS[24:32]},
}


def _single_review_object(text: str) -> tuple[dict[str, Any], str]:
    """Extract one final review object from Grok Build's redacted text stream."""
    decoder = json.JSONDecoder()
    candidates: list[tuple[dict[str, Any], str]] = []
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not {"role", "round", "facets"} <= set(value):
            continue
        if text[end:].strip():
            raise io.Refused("Grok Build review has a non-whitespace payload after the final JSON object")
        candidates.append((value, text[:offset]))
    if len(candidates) != 1:
        raise io.Refused(f"expected exactly one final Grok review object, found {len(candidates)}")
    return candidates[0]


def grok_build_record(
    task_directory: Path,
    contract_path: Path,
    *,
    expected_repository: Path | None = None,
) -> dict[str, Any]:
    """Validate redacted on-device Grok Build artifacts and construct a ledger row."""
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
        raise io.Refused(f"Grok Build task is missing redacted artifacts: {', '.join(missing)}")
    metadata = json.loads(required["metadata"].read_text())
    envelope = json.loads(required["output"].read_text())
    prompt = required["task"].read_text()
    if prompt != contract_path.read_text():
        raise io.Refused("executed Grok Build task does not exactly match the supplied contract")
    model_usage = envelope.get("modelUsage")
    repository = (expected_repository or io.ROOT).resolve()
    checks = {
        "completed_status": required["status"].read_text().strip() == "done",
        "zero_exit": required["exit_code"].read_text().strip() == "0",
        "audit_mode": metadata.get("mode") == "audit",
        "read_only_sandbox": metadata.get("sandbox") == "read-only",
        "repository": Path(str(metadata.get("repo", ""))).resolve() == repository,
        "session_identity": metadata.get("session_id") == envelope.get("sessionId"),
        "normal_stop": envelope.get("stopReason") == "EndTurn",
        "model_usage": isinstance(model_usage, dict) and len(model_usage) == 1,
        "wrapper_model": metadata.get("model") == "grok-4.5",
        "turns_observed": isinstance(envelope.get("num_turns"), int) and envelope["num_turns"] > 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise io.Refused(f"Grok Build task validation failed: {', '.join(failed)}")
    output, prefix = _single_review_object(str(envelope.get("text", "")))
    model_identity = next(iter(model_usage))
    evidence_match = re.search(r"PUBLIC EVIDENCE COMMIT:\s*([0-9a-f]{40})", prompt)
    if evidence_match is None:
        raise io.Refused("Grok Build contract does not identify a full public evidence commit")
    record = {
        "invocation_id": str(metadata["task_id"]),
        "role": output.get("role"),
        "round": output.get("round"),
        "prompt": prompt,
        "prompt_digest": io.digest(prompt),
        "model_identity": model_identity,
        "wrapper_model": metadata["model"],
        "observed_at": metadata.get("started_at"),
        "inputs": {
            "evidence_commit": evidence_match.group(1),
            "contract_digest": io.file_digest(contract_path),
            "task_prompt_digest": io.file_digest(required["task"]),
            "execution_mode": metadata["mode"],
            "sandbox": metadata["sandbox"],
        },
        "transport": {
            "source": "on_device_grok_build_cli",
            "redacted_artifacts_only": True,
            "session_id": envelope["sessionId"],
            "request_id": envelope.get("requestId"),
            "stop_reason": envelope["stopReason"],
            "non_json_prefix_present": bool(prefix),
            "non_json_prefix_digest": io.digest(prefix) if prefix else None,
            "trailing_payload_present": False,
            "num_turns": envelope["num_turns"],
            "usage": envelope.get("usage"),
            "model_usage": model_usage,
        },
        "output_received": True,
        "output": output,
        "output_digest": io.digest(output),
        "resolved_blocking_defects": [],
        "resolutions": [],
        "adopted_revisions": [],
        "rejected_revisions": [],
        "activation": False,
    }
    return record


def grok_build_rejected_record(
    task_directory: Path,
    contract_path: Path,
    *,
    expected_repository: Path | None = None,
) -> dict[str, Any]:
    """Record a real completed invocation whose review framing is not creditable."""
    task_directory = task_directory.resolve()
    contract_path = contract_path.resolve()
    metadata_path = task_directory / "metadata.json"
    output_path = task_directory / "grok-output.json"
    task_path = task_directory / "task.md"
    required = [metadata_path, output_path, task_path, task_directory / "status", task_directory / "exit_code"]
    if not all(path.is_file() for path in required):
        raise io.Refused("rejected Grok Build attempt is missing required redacted artifacts")
    metadata = json.loads(metadata_path.read_text())
    envelope = json.loads(output_path.read_text())
    prompt = task_path.read_text()
    if prompt != contract_path.read_text():
        raise io.Refused("rejected Grok Build task does not exactly match the supplied contract")
    model_usage = envelope.get("modelUsage")
    repository = (expected_repository or io.ROOT).resolve()
    checks = {
        "completed_status": (task_directory / "status").read_text().strip() == "done",
        "zero_exit": (task_directory / "exit_code").read_text().strip() == "0",
        "audit_mode": metadata.get("mode") == "audit",
        "read_only_sandbox": metadata.get("sandbox") == "read-only",
        "repository": Path(str(metadata.get("repo", ""))).resolve() == repository,
        "session_identity": metadata.get("session_id") == envelope.get("sessionId"),
        "normal_stop": envelope.get("stopReason") == "EndTurn",
        "model_usage": isinstance(model_usage, dict) and len(model_usage) == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise io.Refused(f"rejected Grok Build task validation failed: {', '.join(failed)}")
    try:
        _single_review_object(str(envelope.get("text", "")))
    except io.Refused as exc:
        rejection_reason = str(exc)
    else:
        raise io.Refused("Grok Build attempt is schema-extractable and must use the credited importer")
    role_match = re.search(r"ROLE:\s*([a-z0-9_]+)", prompt) or re.search(r"role\s+`([a-z0-9_]+)`", prompt)
    round_match = re.search(r"ROUND:\s*([a-z0-9_]+)", prompt) or re.search(r"round\s+`([a-z0-9_]+)`", prompt)
    evidence_match = re.search(r"PUBLIC EVIDENCE COMMIT:\s*([0-9a-f]{40})", prompt)
    if role_match is None or round_match is None or evidence_match is None:
        raise io.Refused("rejected Grok Build contract lacks role, round, or evidence commit")
    return {
        "invocation_id": str(metadata["task_id"]),
        "role": role_match.group(1),
        "round": round_match.group(1),
        "prompt": prompt,
        "prompt_digest": io.digest(prompt),
        "model_identity": next(iter(model_usage)),
        "wrapper_model": metadata.get("model"),
        "observed_at": metadata.get("started_at"),
        "inputs": {
            "evidence_commit": evidence_match.group(1),
            "contract_digest": io.file_digest(contract_path),
            "task_prompt_digest": io.file_digest(task_path),
            "execution_mode": metadata["mode"],
            "sandbox": metadata["sandbox"],
        },
        "transport": {
            "source": "on_device_grok_build_cli",
            "redacted_artifacts_only": True,
            "session_id": envelope["sessionId"],
            "request_id": envelope.get("requestId"),
            "stop_reason": envelope["stopReason"],
            "redacted_text_digest": io.digest(str(envelope.get("text", ""))),
            "num_turns": envelope.get("num_turns"),
            "usage": envelope.get("usage"),
            "model_usage": model_usage,
        },
        "output_received": True,
        "output": None,
        "output_digest": None,
        "credited": False,
        "rejection_reason": rejection_reason,
        "activation": False,
    }


def prompt_for(role: str, round_identity: str, *, evidence_commit: str, evidence_scope: str) -> str:
    if role not in C.REVIEW_CELLS:
        raise io.Refused(f"unknown Grok review role {role!r}")
    if round_identity not in C.REVIEW_ROUNDS:
        raise io.Refused(f"unknown Grok review round {round_identity!r}")
    facet_lines = "\n".join(f"{index}. {name}" for index, name in enumerate(C.FACETS, start=1))
    special = {
        "closure_null_defender": "Argue the strongest possible case that S2 invalidates an architectural Nous claim.",
        "closure_null_challenger": (
            "Test task compilation, capacity, resource mismatch, answer encoding, saturation, implementation access, and metric design. "
            "You may conclude S2 is fully valid."
        ),
        "architecture_proposals": (
            "Propose a Candidate-H substrate-native architecture with mechanism, state representation, executable bounded prototype, "
            "failure modes, resource costs, and a discriminating ablation. Do not merely rename an existing candidate."
        ),
        "code_and_implementation_review": (
            "Inspect correctness, leakage, architecture, simplicity, performance, security, checkpoint coverage, and claim alignment."
        ),
        "post_pilot_review": "Audit raw decision receipts, controls, oracle headroom, complexity, and every null before recommending selection.",
        "publication_and_claim_boundary_review": "Audit every public sentence against frozen effects and forbid unqualified Nous.",
    }
    special_instruction = special.get(role, special.get(round_identity, "Search for the strongest role-specific falsification."))
    return f"""You are an independent Grok review cell for the Substrate final revision.

ROLE: {role}
ROUND: {round_identity}
PUBLIC EVIDENCE COMMIT: {evidence_commit}
EVIDENCE SCOPE: {evidence_scope}

Do not assume any project claim, receipt, test, benchmark, or narrative is true.
Do not invent repository or execution access. State access limitations exactly.
Historical result `terminal_closed_null` is immutable: candidate minus S2 was
0.0 with 95% CI [0,0]; a tie and any favorable effect below SESOI 0.05 are
nulls. Grok opinion is not a primary endpoint.

ROLE-SPECIFIC MANDATE:
{special_instruction}

Grade these 20 facets independently with binary 0 or 1:
{facet_lines}

Return STRICT JSON only. Required top-level fields:
- role (must equal {role!r})
- round (must equal {round_identity!r})
- evidence_scope
- access_limitations
- assumptions_prohibited
- facets: exactly 20 objects with facet_number 1..20, name,
  score_binary 0|1, discussion_credit 0|0.5|1, rationale
- total_binary_out_of_20
- confidence: low|medium|high
- blocking_defects: array
- nonblocking_concerns: array
- strongest_evidence
- strongest_falsification_evidence
- falsification_tests: nonempty array
- concrete_revisions: array
- recommended_terminal_classification
- minority_or_uncertain_points: array
- candidate_h_proposal: object or null

Every proposed revision must be testable. Preserve minority objections. Do not
turn architecture presence, long prompts, transcript replay, model calls,
digests, modality labels, or reviewer votes into cognitive evidence."""


def prefreeze_manifest(evidence_commit: str) -> list[dict[str, Any]]:
    rows = []
    for role in C.REVIEW_CELLS:
        round_identity = PREFREEZE_ROUND_ASSIGNMENT[role]
        scope = {
            "blind_independent_review": "historical authorities, research ledger, and preflight evidence",
            "cross_examination": "conflicting historical and architectural interpretations",
            "architecture_proposals": "candidate contract, tournament prototypes, and Candidate-H design",
            "test_and_baseline_proposals": "new generator, strongest baselines, fairness, and mutation plan",
            "code_and_implementation_review": "final_revision source, focused tests, checkpoint and activation logic",
            "post_pilot_review": "moderate-pilot raw receipts, decision implementations, headroom, nulls, and resources",
        }[round_identity]
        prompt = prompt_for(role, round_identity, evidence_commit=evidence_commit, evidence_scope=scope)
        rows.append(
            {
                "role": role,
                "round": round_identity,
                "prompt": prompt,
                "prompt_digest": io.digest(prompt),
                "evidence_commit": evidence_commit,
                "evidence_scope": scope,
                "activation": False,
            }
        )
    return rows
