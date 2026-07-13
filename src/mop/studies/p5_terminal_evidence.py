"""Fail-closed join for the terminal governed P5 evidence chain.

The potential atlas and extended-compute matrix describe the latest durable P5
state, not an earlier host-admission attempt.  This module validates every
completed governor leg against the current policy's reviewed legacy-baseline
compatibility contract, then joins the smoke, pilot, fresh challenge, and
independent verifier artifacts by exact hashes and terminal semantics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mop.studio.local_throttle import (
    _completion_receipt_problems,
    _task_output_path,
    load_policy,
)

P5_SMOKE_RECEIPT_PATH = "proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json"
P5_PILOT_RECEIPT_PATH = "proof/LOCAL_THROTTLE_P5_PILOT_RUN_LEG6.json"
P5_FRESH_RECEIPT_PATH = "proof/LOCAL_THROTTLE_P5_FRESH_CHALLENGE_RUN_LEG2.json"
P5_VERIFIER_RECEIPT_PATH = (
    "runs/local_throttle/"
    "mac-studio-substrate-policy-transition-v1-p5verify_cpu-20260712T135308Z-leg01/"
    "run_receipt.json"
)

P5_SMOKE_ARTIFACT_PATH = "proof/P5_CONTEXT_CAPABILITY_SMOKE.json"
P5_PILOT_ARTIFACT_PATH = "proof/P5_CONTEXT_CAPABILITY_PILOT.json"
P5_FRESH_ARTIFACT_PATH = "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json"
P5_VERIFICATION_PATH = "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"

P5_TERMINAL_EVIDENCE_PATHS = (
    P5_SMOKE_RECEIPT_PATH,
    P5_SMOKE_ARTIFACT_PATH,
    P5_PILOT_RECEIPT_PATH,
    P5_PILOT_ARTIFACT_PATH,
    P5_FRESH_RECEIPT_PATH,
    P5_FRESH_ARTIFACT_PATH,
    P5_VERIFIER_RECEIPT_PATH,
    P5_VERIFICATION_PATH,
)

P5_VERIFICATION_SCHEMA = "mop-p5-context-independent-verifier/v1"
P5_VERIFICATION_FIELDS = {
    "verification_complete": True,
    "all_ok": True,
    "prerequisite_ready": True,
    "problems": [],
    "all_controls_passed": True,
    "all_mutations_rejected": True,
    "controls.seed_arm_checkpoint_artifacts_exactly_joined": True,
    "controls.nonterminal_outcome_has_off_ceiling_multiunit_support": True,
    "controls.threshold_tie_is_null": True,
    "independence.checkpoint_files_opened_with_weights_only": True,
    "independence.checkpoint_model_and_target_state_hashes_recomputed": True,
    "independence.heldout_metrics_reexecuted_from_checkpoint": False,
    "outcome_contract.tie_is_null": True,
    "promotion.confirmatory_promotable": False,
    "scientific_promotion": False,
}

_P5_TERMINAL_VERIFICATION_FIELDS = {
    **P5_VERIFICATION_FIELDS,
    "classification": "null",
    "outcome": "null",
    "primary_outcome": "favorable-programmatic-only",
    "terminal_null": False,
    "verified_patterns": [],
    "independence.fresh_training_seeds": [5101, 5102, 5103],
    "independence.fresh_seeds_disjoint_from_primary": True,
    "outcome_contract.programmatic_only": True,
    "outcome_contract.scientific_capability_claim": False,
    "promotion.scientific_capability_claim": False,
}

_GOVERNED_STAGES = {
    "smoke": {
        "path": P5_SMOKE_RECEIPT_PATH,
        "task_id": "p5smoke_cpu",
        "run_id": "p5smoke_20260711_leg4",
        "artifact": P5_SMOKE_ARTIFACT_PATH,
    },
    "pilot": {
        "path": P5_PILOT_RECEIPT_PATH,
        "task_id": "p5pilot_cpu",
        "run_id": "p5pilot_20260712_leg6",
        "artifact": P5_PILOT_ARTIFACT_PATH,
    },
    "fresh_challenge": {
        "path": P5_FRESH_RECEIPT_PATH,
        "task_id": "p5fresh_challenge_cpu",
        "run_id": "p5fresh_challenge_20260712_leg2",
        "artifact": P5_FRESH_ARTIFACT_PATH,
    },
    "verifier": {
        "path": P5_VERIFIER_RECEIPT_PATH,
        "task_id": "p5verify_cpu",
        "run_id": ("mac-studio-substrate-policy-transition-v1-p5verify_cpu-20260712T135308Z-leg01"),
        "artifact": P5_VERIFICATION_PATH,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _field(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _load_documents(
    repo_root: Path,
    documents: Mapping[str, dict[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    loaded: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    provided = documents or {}
    for relative in P5_TERMINAL_EVIDENCE_PATHS:
        candidate = provided.get(relative)
        if candidate is not None:
            if not isinstance(candidate, dict):
                problems.append(f"{relative} is not an object")
            else:
                loaded[relative] = candidate
            continue
        try:
            payload = json.loads((repo_root / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{relative} is unreadable: {type(exc).__name__}")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{relative} is not an object")
            continue
        loaded[relative] = payload
    return loaded, problems


def _self_seal_problems(payload: Mapping[str, Any], label: str) -> list[str]:
    core = dict(payload)
    declared = core.pop("payload_sha256", None)
    if not isinstance(declared, str) or len(declared) != 64:
        return [f"{label} payload self-seal is absent"]
    try:
        observed = _canonical_sha256(core)
    except (TypeError, ValueError):
        return [f"{label} payload is not canonically serializable"]
    return [] if observed == declared else [f"{label} payload self-seal drifted"]


def _source_binding_problems(
    payload: Mapping[str, Any],
    label: str,
    repo_root: Path,
) -> list[str]:
    bindings = payload.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        return [f"{label} source bindings are absent"]
    problems: list[str] = []
    seen: set[str] = set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            problems.append(f"{label} source binding {index} is malformed")
            continue
        relative = binding.get("path")
        expected = binding.get("file_sha256")
        if not isinstance(relative, str) or relative in seen:
            problems.append(f"{label} source binding {index} path is invalid or duplicated")
            continue
        seen.add(relative)
        path = (repo_root / relative).resolve()
        if (
            not path.is_relative_to(repo_root.resolve())
            or not path.is_file()
            or not isinstance(expected, str)
            or _sha256(path) != expected
        ):
            problems.append(f"{label} source binding {relative} is not current")
    return problems


def _governor_problems(
    documents: Mapping[str, dict[str, Any]],
    repo_root: Path,
) -> list[str]:
    problems: list[str] = []
    try:
        policy = load_policy(repo_root / "configs/local_execution_throttle.yaml")
    except Exception as exc:  # fail closed across policy parser/refusal types
        return [f"current local-throttle policy is unavailable: {type(exc).__name__}: {exc}"]
    for stage, spec in _GOVERNED_STAGES.items():
        receipt = documents.get(str(spec["path"]))
        if not isinstance(receipt, dict):
            problems.append(f"P5 {stage} governor receipt is unavailable")
            continue
        if receipt.get("run_id") != spec["run_id"]:
            problems.append(f"P5 {stage} governor run identity drifted")
        try:
            task = policy.tasks[str(spec["task_id"])]
            output_path = _task_output_path(task)
            if output_path != spec["artifact"]:
                problems.append(f"P5 {stage} live task output authority drifted")
                continue
            stage_problems = _completion_receipt_problems(
                receipt,
                receipt_path=(repo_root / str(spec["path"])).resolve(),
                task=task,
                output_path=output_path,
                policy=policy,
                evidence_root=repo_root.resolve(),
            )
        except Exception as exc:  # fail closed across governor compatibility checks
            problems.append(f"P5 {stage} governor completion validation failed: {type(exc).__name__}: {exc}")
            continue
        problems.extend(f"P5 {stage} governor: {problem}" for problem in stage_problems)
    return problems


def _screen_problems(
    payload: Mapping[str, Any],
    *,
    label: str,
    profile: str,
    seeds: list[int],
    fresh_challenge_required: bool,
    repo_root: Path,
) -> list[str]:
    problems = _self_seal_problems(payload, label)
    expected = {
        "schema": "mop-p5-context-screen/v1",
        "profile": profile,
        "seeds": seeds,
        "complete": True,
        "all_ok": True,
        "resumable": False,
        "execution_status": "complete",
        "problems": [],
        "terminal_scientific_stop": False,
        "terminal_stop_reason": None,
        "trainability_gate_failed": False,
        "fresh_challenge_required": fresh_challenge_required,
        "promotion.confirmatory_promotable": False,
        "promotion.refused_by_construction": True,
    }
    for dotted, value in expected.items():
        if _field(payload, dotted) != value:
            problems.append(f"{label} field {dotted} drifted")
    problems.extend(_source_binding_problems(payload, label, repo_root))
    return problems


def p5_terminal_evidence(
    repo_root: Path,
    *,
    documents: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return problems and a compact terminal P5 summary.

    Any missing leg, historical-governor compatibility failure, self-seal
    mismatch, stale source binding, cross-artifact hash mismatch, or semantic
    promotion drift withholds the summary.
    """

    repo_root = repo_root.resolve()
    loaded, problems = _load_documents(repo_root, documents)
    if problems:
        return problems, {}
    problems.extend(_governor_problems(loaded, repo_root))

    smoke = loaded[P5_SMOKE_ARTIFACT_PATH]
    pilot = loaded[P5_PILOT_ARTIFACT_PATH]
    fresh = loaded[P5_FRESH_ARTIFACT_PATH]
    verifier = loaded[P5_VERIFICATION_PATH]
    problems.extend(
        _screen_problems(
            smoke,
            label="P5 smoke artifact",
            profile="p5smoke",
            seeds=[0],
            fresh_challenge_required=False,
            repo_root=repo_root,
        )
    )
    problems.extend(
        _screen_problems(
            pilot,
            label="P5 pilot artifact",
            profile="p5pilot",
            seeds=[0, 1, 2, 3, 4],
            fresh_challenge_required=True,
            repo_root=repo_root,
        )
    )

    problems.extend(_self_seal_problems(fresh, "P5 fresh challenge artifact"))
    fresh_expected = {
        "schema": "mop-p5-context-fresh-training-challenge/v1",
        "complete": True,
        "all_ok": True,
        "resumable": False,
        "problems": [],
        "fresh_training_seeds": [5101, 5102, 5103],
        "fresh_seeds_disjoint_from_primary": True,
        "verification_ready": True,
        "promotion.confirmatory_promotable": False,
        "promotion.scientific_capability_claim": False,
        "scientific_promotion": False,
    }
    for dotted, value in fresh_expected.items():
        if _field(fresh, dotted) != value:
            problems.append(f"P5 fresh challenge field {dotted} drifted")
    patterns = fresh.get("patterns")
    expected_pattern_ids = {
        "f64-exact-minus-recurrent",
        "f64-exact-minus-hierarchical_pooled",
        "f32-exact-minus-recurrent",
        "f32-exact-minus-hierarchical_pooled",
    }
    if (
        not isinstance(patterns, list)
        or len(patterns) != 4
        or {row.get("id") for row in patterns if isinstance(row, dict)} != expected_pattern_ids
    ):
        problems.append("P5 fresh challenge pattern inventory drifted")
    problems.extend(_source_binding_problems(fresh, "P5 fresh challenge artifact", repo_root))

    problems.extend(_self_seal_problems(verifier, "P5 verifier artifact"))
    if verifier.get("schema") != P5_VERIFICATION_SCHEMA:
        problems.append("P5 verifier schema drifted")
    for dotted, value in _P5_TERMINAL_VERIFICATION_FIELDS.items():
        if _field(verifier, dotted) != value:
            problems.append(f"P5 verifier field {dotted} drifted")
    mutations = verifier.get("mutation_tests")
    if not isinstance(mutations, list) or len(mutations) != 23:
        problems.append("P5 verifier mutation inventory drifted")
    problems.extend(_source_binding_problems(verifier, "P5 verifier artifact", repo_root))

    pilot_sha = _sha256(repo_root / P5_PILOT_ARTIFACT_PATH)
    fresh_sha = _sha256(repo_root / P5_FRESH_ARTIFACT_PATH)
    expected_primary = {
        "path": P5_PILOT_ARTIFACT_PATH,
        "sha256": pilot_sha,
        "payload_sha256": pilot.get("payload_sha256"),
    }
    if fresh.get("primary_receipt") != expected_primary:
        problems.append("P5 fresh challenge does not exactly bind the pilot artifact")
    if verifier.get("primary_receipt") != expected_primary:
        problems.append("P5 verifier does not exactly bind the pilot artifact")
    verifier_fresh = verifier.get("fresh_challenge")
    if not isinstance(verifier_fresh, dict) or any(
        verifier_fresh.get(key) != expected
        for key, expected in {
            "path": P5_FRESH_ARTIFACT_PATH,
            "sha256": fresh_sha,
            "payload_sha256": fresh.get("payload_sha256"),
        }.items()
    ):
        problems.append("P5 verifier does not exactly bind the fresh challenge artifact")

    if problems:
        return list(dict.fromkeys(problems)), {}

    receipts = {
        stage: {
            "path": str(spec["path"]),
            "run_id": str(spec["run_id"]),
            "sha256": _sha256(repo_root / str(spec["path"])),
        }
        for stage, spec in _GOVERNED_STAGES.items()
    }
    artifacts = {
        "smoke": {
            "path": P5_SMOKE_ARTIFACT_PATH,
            "sha256": _sha256(repo_root / P5_SMOKE_ARTIFACT_PATH),
        },
        "pilot": {"path": P5_PILOT_ARTIFACT_PATH, "sha256": pilot_sha},
        "fresh_challenge": {"path": P5_FRESH_ARTIFACT_PATH, "sha256": fresh_sha},
        "verifier": {
            "path": P5_VERIFICATION_PATH,
            "sha256": _sha256(repo_root / P5_VERIFICATION_PATH),
        },
    }
    return [], {
        "state": "governed-terminal-null",
        "outcome": "null",
        "classification": "null",
        "primary_outcome": "favorable-programmatic-only",
        "primary_seed_count": 5,
        "fresh_training_seeds": [5101, 5102, 5103],
        "fresh_pattern_count": 4,
        "verified_pattern_count": 0,
        "mutation_count": 23,
        "prerequisite_ready": True,
        "programmatic_only": True,
        "scientific_promotion": False,
        "confirmatory_promotable": False,
        "claim_scope": verifier.get("claim_scope"),
        "governor_receipts": receipts,
        "artifacts": artifacts,
    }


def require_p5_terminal_evidence(
    repo_root: Path,
    *,
    documents: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    problems, summary = p5_terminal_evidence(repo_root, documents=documents)
    if problems:
        raise ValueError("invalid terminal P5 evidence: " + ", ".join(problems))
    return summary
