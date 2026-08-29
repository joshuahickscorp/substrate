"""Post-R2 blind-shadow launch scaffold.

This module is intentionally separate from the sealed R2 execution sources.
It creates and validates the next-launch control plane, but it never alters,
starts, or reclassifies the active R2 continuity lane.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROGRAM = "substrate-tangible-next-launch"
CONTROL_RELATIVE = Path("docs/plans/substrate/tangible_next_launch")
R2_EVIDENCE_RELATIVE = Path("evidence/substrate/tangible_sandbox")
R2_LIVE_STATE_RELATIVE = Path("runs/substrate/tangible_sandbox/longitudinal/state.json")
RUNS_RELATIVE = Path("runs/substrate/tangible_next_launch")
EVIDENCE_RELATIVE = Path("evidence/substrate/tangible_next_launch")
RUNBOOK_RELATIVE = Path("docs/archive/staging/tangible_sandbox/NEXT_LAUNCH_RUNBOOK.md")
GIB = 1024**3


class Refused(RuntimeError):
    """A next-launch gate was not satisfied."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


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
    if isinstance(value, (list, tuple)):
        return any(_contains_true_activation(child) for child in value)
    return False


def _read_json(path: Path, *, require_digest: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} must contain a JSON object")
    if _contains_true_activation(value):
        raise Refused(f"{path} enables activation")
    claimed = value.get("sha256")
    if require_digest and not isinstance(claimed, str):
        raise Refused(f"{path} is missing its authority digest")
    if isinstance(claimed, str):
        unsigned = dict(value)
        unsigned.pop("sha256")
        if digest(unsigned) != claimed:
            raise Refused(f"{path} authority digest mismatch")
    return value


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != payload and not overwrite:
        raise Refused(f"refusing to overwrite {path}")
    if path.exists() and path.read_bytes() == payload:
        return path
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _authority(schema: str, payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    body = {
        "schema": schema,
        "program": PROGRAM,
        "scientific_status": status,
        **payload,
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
    }
    body.pop("sha256", None)
    body["sha256"] = digest(body)
    return body


def _paths(root: Path) -> dict[str, Path]:
    control = root / CONTROL_RELATIVE
    return {
        "control": control,
        "design": control / "BLIND_SHADOW_DESIGN.draft.json",
        "data": control / "DATASET_ADOPTION_REGISTER.draft.json",
        "calibration": control / "RESOURCE_CALIBRATION_SPEC.draft.json",
        "policy": control / "PIVOT_POLICY.sealed.json",
        "adapter_contract": control / "ADAPTER_CONTRACT.sealed.json",
        "review": control / "POST_R2_REVIEW.json",
        "sealed_design": control / "BLIND_SHADOW_DESIGN.sealed.json",
        "preflight": control / "BLIND_SHADOW_PREFLIGHT.json",
        "calibration_result": control / "RESOURCE_CALIBRATION_RESULT.json",
        "runbook": root / RUNBOOK_RELATIVE,
    }


def _default_design() -> dict[str, Any]:
    return {
        "schema": "SUBSTRATE_TANGIBLE_BLIND_SHADOW_DESIGN_DRAFT/v1",
        "status": "draft_waiting_for_r2_review",
        "identity": {
            "experiment_id": "tangible-blind-shadow-r1",
            "parent": "substrate-tangible-sandbox-r2",
            "purpose": "independent 24-hour blinded continuity replication on a new stimulus composition",
        },
        "hypothesis": {
            "question": (
                "Does the precommitted candidate preserve and use project state across a "
                "blinded tangible work timeline beyond the matched project-state control?"
            ),
            "primary_comparison": "L1_full_minus_project_state_database",
            "sesoi": 0.05,
            "tie_is_null": True,
            "wrong_direction_is_failure": True,
        },
        "duration": {
            "hours": 24,
            "one_longitudinal_writer": True,
            "schedule": [
                [0, "intake", "new_project_intake"],
                [3, "multimodal_review", "inspect_new_stimulus_family"],
                [6, "restart_1", "return_to_old_work"],
                [9, "correction_1", "new_task_requires_earlier_history"],
                [12, "model_replacement", "rebuild_from_checkpoint"],
                [15, "sensor_interruption", "degraded_sensor_recovery"],
                [18, "tool_body_change", "return_to_old_work"],
                [21, "correction_2", "new_task_requires_earlier_history"],
                [24, "final_checkpoint", "sealed_scoring_handoff"],
            ],
        },
        "arms": {
            "policy": "R2 parity unless a separately approved causal change is stated below",
            "required": [
                "L1_full", "L1_no_development", "fresh_model", "full_transcript_replay",
                "summary_replay", "strong_retrieval", "conventional_memory_agent",
                "project_state_database", "stateless_router", "direct_strongest_model",
                "best_of_n_direct_model", "S2", "oracle",
            ],
            "approved_causal_change": None,
        },
        "custody": {
            "custodian_role": "separate custodian process/account",
            "seed_commitment_sha256": "REPLACE_WITH_CUSTODIAN_COMMITMENT",
            "seed_plaintext_location": "outside_repository_and_candidate_access",
            "builder_visible_root": "builder_visible",
            "candidate_root": "candidate_workspace",
            "evaluator_only_root": "evaluator_only",
            "publication_safe_root": "publication_safe",
            "candidate_can_read_evaluator_only": False,
            "isolation_attested": False,
            "scoring_key_release": "only_after_candidate_trace_sha256_is_sealed",
        },
        "stimulus_bank": {
            "selection_state": "pending_post_r2_review",
            "selected_card_ids": [],
            "independent_units": "held-out tangible work histories",
            "novelty_requirement": "new composition and task identities; no reuse of R2 answer mappings",
            "required_modalities": ["document", "image_or_video", "audio_or_telemetry", "structured_project_state"],
        },
        "adapters": {
            "candidate_command": "REPLACE_WITH_VERSIONED_CANDIDATE_ENTRYPOINT",
            "control_command": "REPLACE_WITH_VERSIONED_MATCHED_CONTROL_ENTRYPOINT",
            "evaluator_command": "REPLACE_WITH_INDEPENDENT_SCORER_ENTRYPOINT",
            "contract": f"{RUNBOOK_RELATIVE.as_posix()}#adapter-contract",
        },
        "storage": {
            "same_filesystem_static_project_is_double_counted": False,
            "minimum_floor_fraction": 0.20,
            "minimum_floor_gib": 50,
            "estimated_own_run_growth_mib": 16,
            "peak_transient_mib": 128,
            "post_run_clean_clone_gib": 2.5,
            "user_reserve_gib": 0,
            "pending_acquisition_bytes": 0,
        },
        "approval": {
            "state": "draft",
            "r2_review_sha256": "REPLACE_AFTER_R2_REVIEW",
            "reviewer": "REPLACE_WITH_REVIEWER_ID",
            "approved_at": "REPLACE_WITH_ISO8601_TIMESTAMP",
        },
        "activation": False,
        "external_activation": False,
    }


def _default_data_register() -> dict[str, Any]:
    return {
        "schema": "SUBSTRATE_TANGIBLE_DATASET_ADOPTION_REGISTER_DRAFT/v1",
        "status": "metadata_only_no_downloads_authorized",
        "cards": [
            {
                "id": "stsc_r2_recomposition",
                "state": "candidate",
                "selected": False,
                "source": "existing STSC-1 material, new manifest-pinned composition only",
                "new_stimulus_role": "new tangible history composition without a new bulk download",
                "license_or_terms_reviewed": False,
                "hashes_pinned": False,
                "candidate_control_parity": False,
                "evaluator_only_split": False,
                "additional_bytes": 0,
            },
            {
                "id": "long_session_memory_holdout",
                "state": "candidate",
                "selected": False,
                "source": "locally available long-session benchmark metadata; exact source/version to be selected",
                "new_stimulus_role": "multi-session held-out future work",
                "license_or_terms_reviewed": False,
                "hashes_pinned": False,
                "candidate_control_parity": False,
                "evaluator_only_split": False,
                "additional_bytes": "REPLACE_WITH_MEASURED_BYTES",
            },
            {
                "id": "multimodal_incident_extension",
                "state": "candidate",
                "selected": False,
                "source": "license-cleared real project artifacts",
                "new_stimulus_role": "durable cross-modal disturbance and recovery",
                "license_or_terms_reviewed": False,
                "hashes_pinned": False,
                "candidate_control_parity": False,
                "evaluator_only_split": False,
                "additional_bytes": "REPLACE_WITH_MEASURED_BYTES",
            },
            {
                "id": "interactive_replayable_environment",
                "state": "candidate",
                "selected": False,
                "source": "admitted environment with frozen state/actions/scoring",
                "new_stimulus_role": "tool-and-body continuity under replayable dynamics",
                "license_or_terms_reviewed": False,
                "hashes_pinned": False,
                "candidate_control_parity": False,
                "evaluator_only_split": False,
                "additional_bytes": "REPLACE_WITH_MEASURED_BYTES",
            },
        ],
        "adoption_rule": "No source is accepted or downloaded until its selected card has true rights, hash, parity, and evaluator-only fields.",
        "activation": False,
    }


def _default_calibration() -> dict[str, Any]:
    return {
        "schema": "SUBSTRATE_TANGIBLE_RESOURCE_CALIBRATION_SPEC_DRAFT/v1",
        "status": "pending_after_r2",
        "widths": [1, 2, 4],
        "repetitions": 3,
        "unit_count": 8,
        "hash_rounds": 96,
        "receipt_bytes": 262144,
        "max_slowdown_ratio": 1.35,
        "requirements": {
            "distinct_run_roots": True,
            "no_shared_writable_evaluator_or_data_root": True,
            "receipt_invariance": True,
            "record_external_disk_drift": True,
            "record_cpu_memory_io": True,
        },
        "activation": False,
    }


def _default_policy() -> dict[str, Any]:
    return _authority(
        "SUBSTRATE_TANGIBLE_NEXT_PIVOT_POLICY/v1",
        {
            "policy_version": "1.0.0",
            "rules": [
                {
                    "id": "r2_live_or_unreviewed",
                    "when": "r2_review_missing_or_active",
                    "state": "await_r2_review",
                    "action": "read_only_monitoring_only",
                },
                {
                    "id": "r2_invalid",
                    "when": "r2_review_valid_is_false",
                    "state": "repair_diagnosis",
                    "action": "preserve_and_diagnose_no_new_science",
                },
                {
                    "id": "shadow_draft",
                    "when": "r2_valid_and_design_unsealed",
                    "state": "blind_shadow_design_review",
                    "action": "complete_draft_and_custody_review",
                },
                {
                    "id": "calibration",
                    "when": "sealed_design_and_calibration_missing",
                    "state": "resource_calibration",
                    "action": "run_independent_capsule_calibration",
                },
                {
                    "id": "preflight",
                    "when": "calibration_admitted_and_preflight_missing",
                    "state": "shadow_preflight",
                    "action": "compute_fresh_storage_and_isolation_gate",
                },
                {
                    "id": "prepared",
                    "when": "preflight_admitted",
                    "state": "prepare_custody_handoff",
                    "action": "create_new_roots_without_starting_candidate",
                },
                {
                    "id": "safe_hold",
                    "when": "any_guard_fails",
                    "state": "safe_hold",
                    "action": "preserve_receipts_alert_and_do_not_restart_with_changed_protocol",
                },
            ],
            "automatic_side_effect_boundary": (
                "State selection may be automatic. Dataset acquisition, rights acceptance, "
                "custody-key creation, protocol changes, and scientific launch require a "
                "sealed approved manifest."
            ),
        },
        status="sealed",
    )


def _default_adapter_contract() -> dict[str, Any]:
    return _authority(
        "SUBSTRATE_TANGIBLE_ADAPTER_CONTRACT/v1",
        {
            "version": "1.0.0",
            "roles": ["candidate", "matched_control", "independent_evaluator"],
            "request_required": [
                "schema",
                "role",
                "run_id",
                "task_id",
                "input_manifest_sha256",
                "builder_visible_task",
                "receipt_path",
                "activation",
            ],
            "receipt_required": [
                "schema",
                "role",
                "run_id",
                "task_id",
                "input_manifest_sha256",
                "output_artifacts",
                "elapsed_seconds",
                "resource_usage",
                "activation",
            ],
            "request_schema": "SUBSTRATE_TANGIBLE_ADAPTER_REQUEST/v1",
            "receipt_schema": "SUBSTRATE_TANGIBLE_ADAPTER_RECEIPT/v1",
            "evaluator_release_rule": "evaluator answer mapping is released only after candidate trace digest is sealed",
            "forbidden_candidate_input": "evaluator_only root, answer mapping, or uncommitted seed plaintext",
        },
        status="sealed",
    )


def _runbook() -> str:
    return """# Tangible Sandbox next-launch runbook

This control plane is separate from the running R2 source identity. It makes a
blinded 24-hour shadow the default next scientific unit; it does not edit R2 or
turn an incomplete R2 trace into evidence.

## Handoff sequence

```bash
python -m substrate.tangible_next status
python -m substrate.tangible_next review-r2
# Review and complete the three draft JSON files under docs/plans/substrate/tangible_next_launch/
python -m substrate.tangible_next seal-design
python -m substrate.tangible_next run-calibration
python -m substrate.tangible_next preflight
python -m substrate.tangible_next prepare
# Custodian materializes task/answer manifests, then:
python -m substrate.tangible_next seal-custody --handoff RUN/CUSTODY_HANDOFF.json \\
  --task-manifest RUN/builder_visible/TASK_MANIFEST.json \\
  --answer-manifest RUN/evaluator_only/ANSWER_MANIFEST.json \\
  --seed-file /outside/repository/custodian-seed
python -m substrate.tangible_next launch --handoff RUN/CUSTODY_HANDOFF.json
```

`review-r2` deliberately rejects the historic `not_run` evidence and any live
or incomplete longitudinal state. `seal-design` rejects placeholders, missing
custody commitment, unfrozen data cards, or an unapproved causal change.
`seal-custody` verifies that task identities match the sealed 24-hour schedule,
the seed matches its sealed commitment, and answer material stays evaluator
only. `launch` creates a one-shot launchd job; its worker locks the complete
candidate/control trace before it invokes the evaluator.

## Adapter contract

The final short scaffolding session binds three versioned commands in the
sealed design: candidate, matched control, and independent evaluator. Each
must accept a JSON request path and write exactly one JSON receipt path. A
receipt must contain: `task_id`, `run_id`, `input_manifest_sha256`,
`output_artifacts`, `elapsed_seconds`, `resource_usage`, and
`activation:false`. The evaluator command is not released the answer mapping
until the candidate trace digest has been sealed by the custodian.

Use `python -m substrate.tangible_next validate-receipt REQUEST RECEIPT` to
check the contract before the command is admitted. The contract itself is
sealed in `docs/plans/substrate/tangible_next_launch/ADAPTER_CONTRACT.sealed.json`.

The generic control plane intentionally does not invent these scientific
adapters. Their exact behavior depends on the R2 result and the newly selected
stimulus bank; binding a placeholder would look launch-ready while invalidating
the blind comparison.

## What may run in parallel

Only the synthetic resource-calibration capsules and independent preparation
or verification work may run concurrently after their admission check. The
24-hour continuity timeline remains one dedicated writer. Every parallel
capsule receives its own root and may not share writable evaluator or data
state.
"""


def bootstrap(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    paths = _paths(root)
    entries = {
        paths["design"]: _default_design(),
        paths["data"]: _default_data_register(),
        paths["calibration"]: _default_calibration(),
    }
    written = []
    for path, value in entries.items():
        _write_json(path, value, overwrite=overwrite)
        written.append(str(path.relative_to(root)))
    _write_json(paths["policy"], _default_policy())
    written.append(str(paths["policy"].relative_to(root)))
    _write_json(paths["adapter_contract"], _default_adapter_contract())
    written.append(str(paths["adapter_contract"].relative_to(root)))
    paths["runbook"].parent.mkdir(parents=True, exist_ok=True)
    if paths["runbook"].exists() and paths["runbook"].read_text(encoding="utf-8") != _runbook() and not overwrite:
        raise Refused(f"refusing to overwrite {paths['runbook']}")
    paths["runbook"].write_text(_runbook(), encoding="utf-8")
    return {"created_or_verified": written + [str(paths["runbook"].relative_to(root))], "activation": False}


def _r2_live(root: Path) -> dict[str, Any]:
    path = root / R2_LIVE_STATE_RELATIVE
    if not path.is_file():
        return {"present": False, "complete": None}
    state = _read_json(path)
    return {
        "present": True,
        "complete": state.get("complete") is True,
        "elapsed_seconds": state.get("elapsed_seconds"),
        "target_seconds": state.get("target_seconds"),
        "events_emitted": state.get("events_emitted", []),
        "heartbeat_at": state.get("heartbeat_at"),
    }


def _r2_documents(root: Path) -> dict[str, dict[str, Any]]:
    evidence = root / R2_EVIDENCE_RELATIVE
    names = {
        "longitudinal": "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json",
        "verification": "SUBSTRATE_SANDBOX_INDEPENDENT_VERIFICATION.json",
        "classification": "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json",
        "final_state": "SUBSTRATE_SANDBOX_FINAL_STATE.json",
        "clean_clone": "SUBSTRATE_SANDBOX_CLEAN_CLONE.json",
    }
    documents: dict[str, dict[str, Any]] = {}
    for name, filename in names.items():
        path = evidence / filename
        if not path.is_file():
            raise Refused(f"R2 evidence missing {filename}")
        documents[name] = _read_json(path, require_digest=True)
    return documents


def review_r2(root: Path) -> dict[str, Any]:
    live = _r2_live(root)
    if live["present"] and not live["complete"]:
        raise Refused("R2 is still live or incomplete; post-R2 review is blocked")
    documents = _r2_documents(root)
    longitudinal = documents["longitudinal"]
    verification = documents["verification"]
    classification = documents["classification"]
    final_state = documents["final_state"]
    clean_clone = documents["clean_clone"]
    checks = {
        "longitudinal_completed": longitudinal.get("scientific_status") == "complete",
        "actual_wall_at_least_24h": float(longitudinal.get("actual_wall_hours", 0)) >= 24,
        "continuity_passing": longitudinal.get("continuity_passing") is True,
        "independent_verification_passed": verification.get("scientific_status") == "pass" and verification.get("independently_verified") is True,
        "terminal_classification_is_measured": classification.get("outcome") == "B" and classification.get("H_T12", {}).get("status") == "tested",
        "final_state_agrees": final_state.get("outcome") == "B" and float(final_state.get("longitudinal_hours", 0)) >= 24,
        "clean_clone_passed": clean_clone.get("all_pass") is True,
        "activation_false": not any(_contains_true_activation(value) for value in documents.values()),
    }
    review = _authority(
        "SUBSTRATE_TANGIBLE_POST_R2_REVIEW/v1",
        {
            "r2_live_state": live,
            "r2_document_digests": {name: value["sha256"] for name, value in documents.items()},
            "checks": checks,
            "valid": all(checks.values()),
            "next_state": "blind_shadow_design_review" if all(checks.values()) else "repair_diagnosis",
            "note": (
                "R2 effect direction does not select a new result-dependent protocol; this "
                "review checks completion, integrity, and the fixed evidence boundary only."
            ),
        },
        status="pass" if all(checks.values()) else "fail",
    )
    _write_json(_paths(root)["review"], review, overwrite=True)
    return review


def _no_placeholders(value: Any) -> bool:
    if isinstance(value, str):
        return "REPLACE_" not in value and "pending_" not in value
    if isinstance(value, dict):
        return all(_no_placeholders(child) for child in value.values())
    if isinstance(value, list):
        return all(_no_placeholders(child) for child in value)
    return True


def _selected_cards(design: dict[str, Any], data: dict[str, Any]) -> list[dict[str, Any]]:
    selected_ids = design.get("stimulus_bank", {}).get("selected_card_ids")
    if not isinstance(selected_ids, list) or not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise Refused("design must select one or more unique dataset-adoption card IDs")
    cards = {card.get("id"): card for card in data.get("cards", []) if isinstance(card, dict)}
    missing = [card_id for card_id in selected_ids if card_id not in cards]
    if missing:
        raise Refused(f"selected dataset-adoption cards are missing: {missing}")
    selected = [cards[card_id] for card_id in selected_ids]
    if any(card.get("selected") is not True for card in selected):
        raise Refused("each design-selected dataset card must also be selected in the register")
    return selected


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _adopted(card: dict[str, Any]) -> bool:
    return (
        card.get("state") == "accepted"
        and card.get("license_or_terms_reviewed") is True
        and card.get("hashes_pinned") is True
        and card.get("candidate_control_parity") is True
        and card.get("evaluator_only_split") is True
        and isinstance(card.get("additional_bytes"), int)
        and card["additional_bytes"] >= 0
    )


def seal_design(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    if _r2_live(root)["present"] and not _r2_live(root)["complete"]:
        raise Refused("cannot seal next design while R2 is live")
    review = _read_json(paths["review"], require_digest=True)
    design = _read_json(paths["design"])
    data = _read_json(paths["data"])
    policy = _read_json(paths["policy"], require_digest=True)
    adapter_contract = _read_json(paths["adapter_contract"], require_digest=True)
    selected_cards = _selected_cards(design, data)
    pending_acquisition = sum(card["additional_bytes"] for card in selected_cards)
    checks = {
        "r2_review_valid": review.get("valid") is True,
        "design_is_approved": design.get("approval", {}).get("state") == "approved_after_r2_review",
        "review_digest_bound": design.get("approval", {}).get("r2_review_sha256") == review.get("sha256"),
        "no_placeholders": _no_placeholders(design),
        "custody_isolation_attested": design.get("custody", {}).get("isolation_attested") is True,
        "new_stimulus_composition": design.get("stimulus_bank", {}).get("selection_state") == "selected_and_pinned",
        "selected_data_cards_accepted": all(_adopted(card) for card in selected_cards),
        "pending_acquisition_matches_selected_cards": design.get("storage", {}).get("pending_acquisition_bytes") == pending_acquisition,
        "single_longitudinal_writer": design.get("duration", {}).get("one_longitudinal_writer") is True,
        "pivot_policy_is_sealed": policy.get("scientific_status") == "sealed",
        "adapter_contract_is_sealed": adapter_contract.get("scientific_status") == "sealed",
        "activation_false": not _contains_true_activation(design) and not _contains_true_activation(data),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise Refused(f"cannot seal blind-shadow design: {failed}")
    sealed = _authority(
        "SUBSTRATE_TANGIBLE_BLIND_SHADOW_DESIGN/v1",
        {
            "design": design,
            "design_draft_sha256": digest(design),
            "dataset_register": data,
            "dataset_register_sha256": digest(data),
            "selected_dataset_cards": selected_cards,
            "pivot_policy_sha256": policy["sha256"],
            "adapter_contract_sha256": adapter_contract["sha256"],
            "post_r2_review_sha256": review["sha256"],
            "checks": checks,
        },
        status="sealed_before_calibration",
    )
    _write_json(paths["sealed_design"], sealed)
    return sealed


def _calibration_worker(payload: tuple[str, tuple[int, ...], int, int, int]) -> dict[str, Any]:
    directory_text, units, repetition_seed, hash_rounds, receipt_bytes = payload
    directory = Path(directory_text)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipts = []
    for unit in units:
        block = hashlib.sha256(f"{repetition_seed}:{unit}".encode()).digest() * 4096
        aggregate = b""
        for index in range(hash_rounds):
            aggregate = hashlib.sha256(aggregate + block + index.to_bytes(4, "big")).digest()
        receipt = (aggregate * ((receipt_bytes + len(aggregate) - 1) // len(aggregate)))[:receipt_bytes]
        receipt_path = directory / f"receipt-{unit:02d}.bin"
        receipt_path.write_bytes(receipt)
        receipts.append(
            {
                "unit": unit,
                "receipt_sha256": file_digest(receipt_path),
                "receipt_bytes": receipt_path.stat().st_size,
            }
        )
    return {
        "workspace": str(directory),
        "receipts": receipts,
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "pid": os.getpid(),
    }


def run_calibration(root: Path) -> dict[str, Any]:
    if _r2_live(root)["present"] and not _r2_live(root)["complete"]:
        raise Refused("resource calibration is blocked while R2 is live")
    paths = _paths(root)
    sealed = _read_json(paths["sealed_design"], require_digest=True)
    policy = _read_json(paths["policy"], require_digest=True)
    adapter_contract = _read_json(paths["adapter_contract"], require_digest=True)
    spec = _read_json(paths["calibration"])
    if sealed.get("pivot_policy_sha256") != policy.get("sha256"):
        raise Refused("sealed design is not bound to the current immutable pivot policy")
    if sealed.get("adapter_contract_sha256") != adapter_contract.get("sha256"):
        raise Refused("sealed design is not bound to the current immutable adapter contract")
    widths = spec.get("widths")
    if widths != [1, 2, 4] or not all(isinstance(width, int) and width > 0 for width in widths):
        raise Refused("calibration widths must be the deterministic [1, 2, 4] ladder")
    if not isinstance(spec.get("repetitions"), int) or spec["repetitions"] < 2:
        raise Refused("calibration requires at least two repetitions")
    if not isinstance(spec.get("unit_count"), int) or spec["unit_count"] < max(widths):
        raise Refused("calibration unit_count must cover every requested worker width")
    calibration_root = root / RUNS_RELATIVE / "resource-calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    before_free = shutil.disk_usage(root).free
    rows: dict[str, list[dict[str, Any]]] = {}
    for width in widths:
        samples = []
        for repetition in range(spec["repetitions"]):
            cell = calibration_root / f"w{width}-r{repetition}-{uuid.uuid4().hex}"
            cell.mkdir(parents=True, exist_ok=False)
            started = time.monotonic()
            groups = [tuple(unit for unit in range(spec["unit_count"]) if unit % width == worker) for worker in range(width)]
            payloads = [
                (str(cell / f"capsule-{index}"), groups[index], 10_000 + repetition, spec["hash_rounds"], spec["receipt_bytes"])
                for index in range(width)
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=width) as pool:
                receipts = list(pool.map(_calibration_worker, payloads))
            flattened = [receipt for row in receipts for receipt in row["receipts"]]
            receipt_set_digest = digest(sorted((row["unit"], row["receipt_sha256"]) for row in flattened))
            samples.append({
                "wall_seconds": round(time.monotonic() - started, 6),
                "receipts": receipts,
                "receipt_count": len(flattened),
                "receipt_set_sha256": receipt_set_digest,
                "distinct_roots": len({row["workspace"] for row in receipts}) == width,
            })
        rows[str(width)] = samples
    base = sorted(row["wall_seconds"] for row in rows["1"])[len(rows["1"]) // 2]
    medians = {
        width: sorted(row["wall_seconds"] for row in samples)[len(samples) // 2]
        for width, samples in rows.items()
    }
    slowdowns = {width: round(value / base, 6) for width, value in medians.items()}
    invariants = {
        repetition: {
            sample["receipt_set_sha256"]
            for samples in rows.values()
            for sample in (samples[repetition],)
        }
        for repetition in range(spec["repetitions"])
    }
    checks = {
        "all_capsules_returned": all(
            sample["receipt_count"] == spec["unit_count"]
            for samples in rows.values()
            for sample in samples
        ),
        "receipt_sizes_match": all(
            receipt["receipt_bytes"] == spec["receipt_bytes"]
            for samples in rows.values()
            for sample in samples
            for capsule in sample["receipts"]
            for receipt in capsule["receipts"]
        ),
        "receipt_invariant_across_widths": all(len(values) == 1 for values in invariants.values()),
        "distinct_run_roots": all(sample["distinct_roots"] for samples in rows.values() for sample in samples),
        "slowdown_within_limit": all(value <= float(spec["max_slowdown_ratio"]) for value in slowdowns.values()),
        "sealed_design_bound": sealed.get("scientific_status") == "sealed_before_calibration",
    }
    result = _authority(
        "SUBSTRATE_TANGIBLE_RESOURCE_CALIBRATION_RESULT/v1",
        {
            "spec_sha256": digest(spec),
            "sealed_design_sha256": sealed["sha256"],
            "pivot_policy_sha256": policy["sha256"],
            "adapter_contract_sha256": adapter_contract["sha256"],
            "host_logical_cores": os.cpu_count(),
            "free_bytes_before": before_free,
            "free_bytes_after": shutil.disk_usage(root).free,
            "samples": rows,
            "median_wall_seconds": medians,
            "slowdown_ratio": slowdowns,
            "checks": checks,
            "admitted": all(checks.values()),
            "concurrency_scope": "independent synthetic capsules only; no longitudinal shared-state worker was parallelized",
        },
        status="admitted" if all(checks.values()) else "refused",
    )
    _write_json(paths["calibration_result"], result, overwrite=True)
    return result


def preflight(root: Path) -> dict[str, Any]:
    if _r2_live(root)["present"] and not _r2_live(root)["complete"]:
        raise Refused("blind-shadow preflight is blocked while R2 is live")
    paths = _paths(root)
    sealed = _read_json(paths["sealed_design"], require_digest=True)
    calibration = _read_json(paths["calibration_result"], require_digest=True)
    policy = _read_json(paths["policy"], require_digest=True)
    adapter_contract = _read_json(paths["adapter_contract"], require_digest=True)
    design = sealed["design"]
    storage = design["storage"]
    usage = shutil.disk_usage(root)
    protected = max(int(usage.total * float(storage["minimum_floor_fraction"])), int(float(storage["minimum_floor_gib"]) * GIB))
    required = (
        protected
        + int(float(storage["estimated_own_run_growth_mib"]) * 1024**2)
        + int(float(storage["peak_transient_mib"]) * 1024**2)
        + int(float(storage["post_run_clean_clone_gib"]) * GIB)
        + int(float(storage["user_reserve_gib"]) * GIB)
        + int(storage["pending_acquisition_bytes"])
    )
    checks = {
        "calibration_admitted": calibration.get("admitted") is True,
        "design_bound_to_calibration": calibration.get("sealed_design_sha256") == sealed["sha256"],
        "policy_bound_to_design": sealed.get("pivot_policy_sha256") == policy.get("sha256"),
        "policy_bound_to_calibration": calibration.get("pivot_policy_sha256") == policy.get("sha256"),
        "adapter_contract_bound_to_design": sealed.get("adapter_contract_sha256") == adapter_contract.get("sha256"),
        "adapter_contract_bound_to_calibration": calibration.get("adapter_contract_sha256") == adapter_contract.get("sha256"),
        "free_space_sufficient": usage.free >= required,
        "static_project_not_double_counted": storage.get("same_filesystem_static_project_is_double_counted") is False,
        "single_writer": design.get("duration", {}).get("one_longitudinal_writer") is True,
        "candidate_evaluator_isolation_attested": design.get("custody", {}).get("isolation_attested") is True,
    }
    result = _authority(
        "SUBSTRATE_TANGIBLE_BLIND_SHADOW_PREFLIGHT/v1",
        {
            "sealed_design_sha256": sealed["sha256"],
            "calibration_sha256": calibration["sha256"],
            "pivot_policy_sha256": policy["sha256"],
            "adapter_contract_sha256": adapter_contract["sha256"],
            "volume": {"total_bytes": usage.total, "free_bytes": usage.free},
            "storage": {"protected_floor_bytes": protected, "required_free_bytes": required, "headroom_bytes": usage.free - required},
            "checks": checks,
            "admitted": all(checks.values()),
        },
        status="admitted" if all(checks.values()) else "refused",
    )
    _write_json(paths["preflight"], result, overwrite=True)
    return result


def prepare(root: Path) -> dict[str, Any]:
    if _r2_live(root)["present"] and not _r2_live(root)["complete"]:
        raise Refused("custody preparation is blocked while R2 is live")
    paths = _paths(root)
    sealed = _read_json(paths["sealed_design"], require_digest=True)
    preflight_result = _read_json(paths["preflight"], require_digest=True)
    if preflight_result.get("admitted") is not True or preflight_result.get("sealed_design_sha256") != sealed.get("sha256"):
        raise Refused("custody preparation requires an admitted preflight bound to the sealed design")
    run_id = f"shadow-{uuid.uuid4().hex}"
    run_root = root / RUNS_RELATIVE / "blind-shadow" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    roots = {name: run_root / name for name in ("builder_visible", "candidate_workspace", "evaluator_only", "publication_safe", "logs")}
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=False)
    roots["evaluator_only"].chmod(0o700)
    schedule = sealed["design"]["duration"]["schedule"]
    task_template = {
        "schema": "SUBSTRATE_TANGIBLE_BLIND_SHADOW_TASK_MANIFEST/v1",
        "run_id": run_id,
        "sealed_design_sha256": sealed["sha256"],
        "tasks": [
            {
                "scheduled_hour": hour,
                "event": event,
                "task_id": f"REPLACE_TASK_ID_{hour:02d}",
                "builder_task": f"tasks/REPLACE_TASK_FILE_{hour:02d}.json",
            }
            for hour, event, _ in schedule
        ],
        "activation": False,
    }
    answer_template = {
        "schema": "SUBSTRATE_TANGIBLE_BLIND_SHADOW_ANSWER_MANIFEST/v1",
        "run_id": run_id,
        "sealed_design_sha256": sealed["sha256"],
        "answers": "REPLACE_WITH_CUSTODIAN_ONLY_ANSWER_MAPPING",
        "activation": False,
    }
    _write_json(roots["builder_visible"] / "TASK_MANIFEST.template.json", task_template)
    _write_json(roots["evaluator_only"] / "ANSWER_MANIFEST.template.json", answer_template)
    handoff = _authority(
        "SUBSTRATE_TANGIBLE_CUSTODY_HANDOFF/v1",
        {
            "run_id": run_id,
            "sealed_design_sha256": sealed["sha256"],
            "preflight_sha256": preflight_result["sha256"],
            "adapter_contract_sha256": sealed["adapter_contract_sha256"],
            "roots": {name: str(path.relative_to(root)) for name, path in roots.items()},
            "filesystem_mode": "evaluator_only=0700",
            "required_before_candidate_launch": [
                "custodian writes committed seed and independent answer mapping outside candidate access",
                "custodian materializes manifest-pinned builder-visible tasks",
                "candidate/control/evaluator adapter commands are version-pinned in the sealed design",
                "candidate process isolation is independently checked",
            ],
            "prepared_not_launched": True,
        },
        status="prepared_waiting_for_custody",
    )
    _write_json(run_root / "CUSTODY_HANDOFF.json", handoff)
    return handoff


def _load_handoff(root: Path, handoff_path: Path) -> tuple[Path, dict[str, Any]]:
    handoff_path = handoff_path.expanduser().resolve()
    runs_root = (root / RUNS_RELATIVE / "blind-shadow").resolve()
    if not _inside(handoff_path, runs_root):
        raise Refused("custody handoff must be inside the next-launch blind-shadow run root")
    handoff = _read_json(handoff_path, require_digest=True)
    if handoff.get("schema") != "SUBSTRATE_TANGIBLE_CUSTODY_HANDOFF/v1":
        raise Refused("invalid custody handoff schema")
    if handoff.get("scientific_status") != "prepared_waiting_for_custody":
        raise Refused("custody handoff is not in a launchable preparation state")
    return handoff_path, handoff


def _resolved_roots(root: Path, handoff: dict[str, Any]) -> dict[str, Path]:
    roots = {name: (root / relative).resolve() for name, relative in handoff.get("roots", {}).items()}
    required = {"builder_visible", "candidate_workspace", "evaluator_only", "publication_safe", "logs"}
    if set(roots) != required or not all(path.is_dir() for path in roots.values()):
        raise Refused("custody handoff roots are incomplete")
    run_root = roots["builder_visible"].parent
    if not all(_inside(path, run_root) for path in roots.values()):
        raise Refused("custody handoff root escapes its run directory")
    return roots


def _task_manifest(root: Path, handoff: dict[str, Any], task_manifest_path: Path) -> dict[str, Any]:
    roots = _resolved_roots(root, handoff)
    task_manifest_path = task_manifest_path.expanduser().resolve()
    if not _inside(task_manifest_path, roots["builder_visible"]):
        raise Refused("task manifest must live in the builder-visible root")
    manifest = _read_json(task_manifest_path)
    if manifest.get("schema") != "SUBSTRATE_TANGIBLE_BLIND_SHADOW_TASK_MANIFEST/v1":
        raise Refused("invalid blind-shadow task manifest schema")
    if manifest.get("run_id") != handoff.get("run_id") or manifest.get("sealed_design_sha256") != handoff.get("sealed_design_sha256"):
        raise Refused("task manifest is not bound to this custody handoff")
    sealed = _read_json(_paths(root)["sealed_design"], require_digest=True)
    schedule = [(row[0], row[1]) for row in sealed["design"]["duration"]["schedule"]]
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or [(row.get("scheduled_hour"), row.get("event")) for row in tasks] != schedule:
        raise Refused("task manifest schedule does not exactly match the sealed design")
    task_ids = [row.get("task_id") for row in tasks]
    if any(not isinstance(task_id, str) or "REPLACE_" in task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise Refused("task manifest has unresolved or duplicate task IDs")
    for row in tasks:
        relative = row.get("builder_task")
        if not isinstance(relative, str) or "REPLACE_" in relative:
            raise Refused("task manifest has an unresolved builder-visible task path")
        task_path = (roots["builder_visible"] / relative).resolve()
        if not _inside(task_path, roots["builder_visible"]) or not task_path.is_file():
            raise Refused(f"builder-visible task is unavailable: {relative}")
    return manifest


def seal_custody(
    root: Path,
    *,
    handoff_path: Path,
    task_manifest_path: Path,
    answer_manifest_path: Path,
    seed_file: Path,
) -> dict[str, Any]:
    """Bind custodian-owned task, answer, and seed commitments before launch."""

    if _r2_live(root)["present"] and not _r2_live(root)["complete"]:
        raise Refused("custody sealing is blocked while R2 is live")
    _, handoff = _load_handoff(root, handoff_path)
    roots = _resolved_roots(root, handoff)
    task_manifest = _task_manifest(root, handoff, task_manifest_path)
    answer_manifest_path = answer_manifest_path.expanduser().resolve()
    seed_file = seed_file.expanduser().resolve()
    if not _inside(answer_manifest_path, roots["evaluator_only"]):
        raise Refused("answer manifest must stay inside the evaluator-only root")
    if not seed_file.is_file() or _inside(seed_file, root):
        raise Refused("custodian seed file must exist outside the repository")
    answers = _read_json(answer_manifest_path)
    if answers.get("schema") != "SUBSTRATE_TANGIBLE_BLIND_SHADOW_ANSWER_MANIFEST/v1":
        raise Refused("invalid blind-shadow answer manifest schema")
    if answers.get("run_id") != handoff["run_id"] or answers.get("sealed_design_sha256") != handoff["sealed_design_sha256"]:
        raise Refused("answer manifest is not bound to this custody handoff")
    if "REPLACE_" in canonical_bytes(answers).decode("utf-8"):
        raise Refused("answer manifest still contains a template placeholder")
    design = _read_json(_paths(root)["sealed_design"], require_digest=True)
    commitment = design["design"]["custody"]["seed_commitment_sha256"]
    if file_digest(seed_file) != commitment:
        raise Refused("custodian seed does not match the sealed design commitment")
    output = roots["publication_safe"] / "CUSTODY_SEAL.json"
    custody = _authority(
        "SUBSTRATE_TANGIBLE_CUSTODY_SEAL/v1",
        {
            "run_id": handoff["run_id"],
            "custody_handoff_sha256": handoff["sha256"],
            "sealed_design_sha256": handoff["sealed_design_sha256"],
            "task_manifest": str(task_manifest_path.relative_to(root)),
            "task_manifest_sha256": file_digest(task_manifest_path),
            "answer_manifest": str(answer_manifest_path.relative_to(root)),
            "answer_manifest_sha256": file_digest(answer_manifest_path),
            "seed_commitment_sha256": commitment,
            "task_ids": [row["task_id"] for row in task_manifest["tasks"]],
            "candidate_evaluator_isolation_attested": design["design"]["custody"]["isolation_attested"] is True,
            "answer_plaintext_in_candidate_trace": False,
        },
        status="sealed_before_detached_launch",
    )
    _write_json(output, custody)
    return custody


def _load_custody(root: Path, handoff: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    roots = _resolved_roots(root, handoff)
    custody_path = roots["publication_safe"] / "CUSTODY_SEAL.json"
    custody = _read_json(custody_path, require_digest=True)
    if (
        custody.get("schema") != "SUBSTRATE_TANGIBLE_CUSTODY_SEAL/v1"
        or custody.get("scientific_status") != "sealed_before_detached_launch"
        or custody.get("custody_handoff_sha256") != handoff["sha256"]
    ):
        raise Refused("custody seal is missing or does not bind the prepared handoff")
    return roots, custody


def _launchd_job(label: str, manifest_path: Path, root: Path, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    return {
        "Label": label,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "substrate.tangible_next",
            "--root",
            str(root),
            "supervised-run",
            "--supervision-manifest",
            str(manifest_path),
        ],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": {"SUBSTRATE_TANGIBLE_SUPERVISOR": "launchd"},
        "KeepAlive": False,
        "RunAtLoad": False,
        "ProcessType": "Adaptive",
        "ThrottleInterval": 60,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "AbandonProcessGroup": False,
    }


def launch(root: Path, *, handoff_path: Path) -> dict[str, Any]:
    """Detach an admitted blind shadow through a one-shot user launchd agent."""

    if _r2_live(root)["present"] and not _r2_live(root)["complete"]:
        raise Refused("detached blind-shadow launch is blocked while R2 is live")
    handoff_path, handoff = _load_handoff(root, handoff_path)
    roots, custody = _load_custody(root, handoff)
    preflight_result = _read_json(_paths(root)["preflight"], require_digest=True)
    sealed = _read_json(_paths(root)["sealed_design"], require_digest=True)
    if preflight_result.get("admitted") is not True or preflight_result.get("sealed_design_sha256") != sealed["sha256"]:
        raise Refused("detached launch requires an admitted current preflight")
    if custody.get("sealed_design_sha256") != sealed["sha256"]:
        raise Refused("custody seal and sealed design disagree")
    adapters = sealed["design"]["adapters"]
    adapter_names = ("candidate_command", "control_command", "evaluator_command")
    if not all(
        isinstance(adapters.get(name), str)
        and adapters[name].strip()
        and "REPLACE_" not in adapters[name]
        for name in adapter_names
    ):
        raise Refused("detached launch requires three version-pinned adapter commands")
    run_root = handoff_path.parent
    supervision_root = run_root / "supervision"
    supervision_root.mkdir(parents=True, exist_ok=False)
    label = f"org.substrate.tangible-shadow.{handoff['run_id']}"
    manifest = _authority(
        "SUBSTRATE_TANGIBLE_BLIND_SHADOW_SUPERVISION_MANIFEST/v1",
        {
            "run_id": handoff["run_id"],
            "launchd_label": label,
            "handoff": str(handoff_path.relative_to(root)),
            "handoff_sha256": handoff["sha256"],
            "custody_seal_sha256": custody["sha256"],
            "sealed_design_sha256": sealed["sha256"],
            "preflight_sha256": preflight_result["sha256"],
            "worker_source_sha256": file_digest(Path(__file__)),
            "process_exit_is_not_completion": True,
        },
        status="sealed_before_detached_launch",
    )
    manifest_path = supervision_root / "manifest.json"
    _write_json(manifest_path, manifest)
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = roots["logs"] / "launchd.stdout.log"
    stderr_path = roots["logs"] / "launchd.stderr.log"
    with plist_path.open("wb") as handle:
        plistlib.dump(_launchd_job(label, manifest_path, root, stdout_path, stderr_path), handle, sort_keys=True)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, label], capture_output=True, text=True, check=False)
    boot = subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)], capture_output=True, text=True, check=False)
    if boot.returncode:
        raise Refused(boot.stderr.strip() or "launchctl bootstrap failed")
    start = subprocess.run(["launchctl", "kickstart", "-k", f"{domain}/{label}"], capture_output=True, text=True, check=False)
    if start.returncode:
        raise Refused(start.stderr.strip() or "launchctl kickstart failed")
    return {
        "run_id": handoff["run_id"],
        "launchd_label": label,
        "supervision_manifest": str(manifest_path.relative_to(root)),
        "plist": str(plist_path),
        "detached": True,
        "activation": False,
    }


def _load_supervision(root: Path, manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    if not _inside(manifest_path, root / RUNS_RELATIVE / "blind-shadow"):
        raise Refused("supervision manifest must be inside a next-launch blind-shadow run root")
    manifest = _read_json(manifest_path, require_digest=True)
    if manifest.get("schema") != "SUBSTRATE_TANGIBLE_BLIND_SHADOW_SUPERVISION_MANIFEST/v1":
        raise Refused("invalid blind-shadow supervision manifest")
    if manifest.get("worker_source_sha256") != file_digest(Path(__file__)):
        raise Refused("blind-shadow worker source drifted after detached launch")
    return manifest_path, manifest


def _invoke_adapter(
    root: Path,
    *,
    command: str,
    request: dict[str, Any],
    request_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    _write_json(request_path, request)
    arguments = shlex.split(command)
    if not arguments:
        raise Refused("adapter command is empty")
    completed = subprocess.run(
        [*arguments, str(request_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15 * 60,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip() or "adapter exited nonzero"
        raise Refused(f"{request['role']} adapter failed: {message}")
    if not receipt_path.is_file():
        raise Refused(f"{request['role']} adapter did not write its receipt")
    check = validate_receipt(root, request_path, receipt_path)
    if not check["all_pass"]:
        failed = [name for name, passed in check["checks"].items() if not passed]
        raise Refused(f"{request['role']} adapter receipt failed contract checks: {failed}")
    return _read_json(receipt_path)


def supervised_run(root: Path, *, supervision_manifest: Path) -> dict[str, Any]:
    """Run the sealed 24-hour shadow under launchd ownership.

    The only parallel work admitted by this control plane is the earlier
    independent resource calibration. This worker serializes the longitudinal
    candidate/control checkpoints and calls the evaluator only after locking
    the candidate trace.
    """

    if os.environ.get("SUBSTRATE_TANGIBLE_SUPERVISOR") != "launchd":
        raise Refused("blind-shadow worker must be owned by its one-shot launchd agent")
    manifest_path, manifest = _load_supervision(root, supervision_manifest)
    handoff_path = (root / manifest["handoff"]).resolve()
    _, handoff = _load_handoff(root, handoff_path)
    if handoff["sha256"] != manifest.get("handoff_sha256"):
        raise Refused("supervision manifest is not bound to the custody handoff")
    roots, custody = _load_custody(root, handoff)
    if custody["sha256"] != manifest.get("custody_seal_sha256"):
        raise Refused("supervision manifest is not bound to the custody seal")
    sealed = _read_json(_paths(root)["sealed_design"], require_digest=True)
    preflight_result = _read_json(_paths(root)["preflight"], require_digest=True)
    if sealed["sha256"] != manifest.get("sealed_design_sha256") or preflight_result["sha256"] != manifest.get("preflight_sha256"):
        raise Refused("supervision manifest has stale design or preflight bindings")
    if preflight_result.get("admitted") is not True:
        raise Refused("blind-shadow preflight is not admitted")
    task_manifest_path = (root / custody["task_manifest"]).resolve()
    answer_manifest_path = (root / custody["answer_manifest"]).resolve()
    if file_digest(task_manifest_path) != custody["task_manifest_sha256"] or file_digest(answer_manifest_path) != custody["answer_manifest_sha256"]:
        raise Refused("custody materials drifted after their seal")
    task_manifest = _task_manifest(root, handoff, task_manifest_path)
    run_root = manifest_path.parents[1]
    worker_root = run_root / "worker"
    if worker_root.exists():
        raise Refused("blind-shadow worker root already exists")
    worker_root.mkdir(parents=True, exist_ok=False)
    trace_path = worker_root / "candidate-control-trace.jsonl"
    state_path = worker_root / "state.json"
    duration = int(sealed["design"]["duration"]["hours"]) * 3600
    if duration != 24 * 3600 or sealed["design"]["duration"].get("one_longitudinal_writer") is not True:
        raise Refused("blind-shadow supervision requires the sealed single-writer 24-hour duration")
    task_by_hour = {row["scheduled_hour"]: row for row in task_manifest["tasks"]}
    schedule = sealed["design"]["duration"]["schedule"]
    started_wall = time.time()
    started_mono = time.monotonic()
    emitted: list[int] = []
    last_report_bucket = -1
    adapters = sealed["design"]["adapters"]
    try:
        with trace_path.open("a", encoding="utf-8") as trace:
            while True:
                elapsed = time.monotonic() - started_mono
                for hour, event, activity in schedule:
                    if hour not in emitted and elapsed >= hour * 3600:
                        task = task_by_hour[hour]
                        task_path = (roots["builder_visible"] / task["builder_task"]).resolve()
                        receipt_records: dict[str, dict[str, Any]] = {}
                        for role, command, workspace in (
                            ("candidate", adapters["candidate_command"], roots["candidate_workspace"]),
                            ("matched_control", adapters["control_command"], roots["candidate_workspace"] / "matched_control"),
                        ):
                            workspace.mkdir(parents=True, exist_ok=True)
                            request_path = workspace / "requests" / f"{hour:02d}-{role}.json"
                            receipt_path = workspace / "receipts" / f"{hour:02d}-{role}.json"
                            request = {
                                "schema": "SUBSTRATE_TANGIBLE_ADAPTER_REQUEST/v1",
                                "role": role,
                                "run_id": handoff["run_id"],
                                "task_id": task["task_id"],
                                "input_manifest_sha256": custody["task_manifest_sha256"],
                                "builder_visible_task": str(task_path.relative_to(root)),
                                "receipt_path": str(receipt_path.relative_to(root)),
                                "scheduled_hour": hour,
                                "event": event,
                                "activity": activity,
                                "activation": False,
                            }
                            receipt_records[role] = _invoke_adapter(
                                root,
                                command=command,
                                request=request,
                                request_path=request_path,
                                receipt_path=receipt_path,
                            )
                        row = {
                            "scheduled_hour": hour,
                            "event": event,
                            "activity": activity,
                            "task_id": task["task_id"],
                            "candidate_receipt_sha256": file_digest(
                                roots["candidate_workspace"] / "receipts" / f"{hour:02d}-candidate.json"
                            ),
                            "control_receipt_sha256": file_digest(
                                roots["candidate_workspace"]
                                / "matched_control"
                                / "receipts"
                                / f"{hour:02d}-matched_control.json"
                            ),
                            "candidate_elapsed_seconds": receipt_records["candidate"]["elapsed_seconds"],
                            "control_elapsed_seconds": receipt_records["matched_control"]["elapsed_seconds"],
                            "wall_time": time.time(),
                            "activation": False,
                        }
                        trace.write(json.dumps(row, sort_keys=True) + "\n")
                        trace.flush()
                        emitted.append(hour)
                disk_free = shutil.disk_usage(root).free
                if disk_free < int(preflight_result["storage"]["required_free_bytes"]):
                    raise Refused("blind-shadow worker crossed the sealed disk guard")
                report_bucket = int(elapsed // (30 * 60))
                if report_bucket != last_report_bucket:
                    last_report_bucket = report_bucket
                _write_json(
                    state_path,
                    {
                        "schema": "SUBSTRATE_TANGIBLE_BLIND_SHADOW_STATE/v1",
                        "run_id": handoff["run_id"],
                        "elapsed_seconds": round(elapsed, 3),
                        "target_seconds": duration,
                        "percent_complete": round(100 * elapsed / duration, 3),
                        "events_emitted": emitted,
                        "worker_pid": os.getpid(),
                        "cpu_time_seconds": round(sum(os.times()[:2]), 3),
                        "disk_free_bytes": disk_free,
                        "disk_guard_bytes": preflight_result["storage"]["required_free_bytes"],
                        "last_30_minute_report_bucket": last_report_bucket,
                        "heartbeat_at_epoch": time.time(),
                        "complete": elapsed >= duration,
                        "activation": False,
                    },
                    overwrite=True,
                )
                if elapsed >= duration:
                    break
                time.sleep(min(60, duration - elapsed))
        if emitted != [row[0] for row in schedule]:
            raise Refused("blind-shadow worker did not emit every sealed checkpoint")
        trace_lock = _authority(
            "SUBSTRATE_TANGIBLE_CANDIDATE_TRACE_LOCK/v1",
            {
                "run_id": handoff["run_id"],
                "trace": str(trace_path.relative_to(root)),
                "trace_sha256": file_digest(trace_path),
                "candidate_trace_locked_before_evaluation": True,
            },
            status="sealed_before_evaluation",
        )
        trace_lock_path = roots["publication_safe"] / "CANDIDATE_TRACE_LOCK.json"
        _write_json(trace_lock_path, trace_lock)
        evaluator_request_path = roots["evaluator_only"] / "EVALUATOR_REQUEST.json"
        evaluator_receipt_path = roots["evaluator_only"] / "EVALUATOR_RECEIPT.json"
        evaluator_request = {
            "schema": "SUBSTRATE_TANGIBLE_ADAPTER_REQUEST/v1",
            "role": "independent_evaluator",
            "run_id": handoff["run_id"],
            "task_id": "blind-shadow-terminal-evaluation",
            "input_manifest_sha256": custody["task_manifest_sha256"],
            "builder_visible_task": str(trace_path.relative_to(root)),
            "receipt_path": str(evaluator_receipt_path.relative_to(root)),
            "candidate_trace_sha256": trace_lock["trace_sha256"],
            "answer_manifest": str(answer_manifest_path.relative_to(root)),
            "activation": False,
        }
        evaluator_receipt = _invoke_adapter(
            root,
            command=adapters["evaluator_command"],
            request=evaluator_request,
            request_path=evaluator_request_path,
            receipt_path=evaluator_receipt_path,
        )
        result = _authority(
            "SUBSTRATE_TANGIBLE_BLIND_SHADOW_RESULT/v1",
            {
                "run_id": handoff["run_id"],
                "actual_wall_hours": round((time.time() - started_wall) / 3600, 6),
                "trace_lock_sha256": trace_lock["sha256"],
                "evaluator_receipt_sha256": file_digest(evaluator_receipt_path),
                "evaluator_elapsed_seconds": evaluator_receipt["elapsed_seconds"],
                "checkpoints": len(emitted),
                "continuity_passing": True,
            },
            status="complete_waiting_for_independent_scoring_interpretation",
        )
        _write_json(roots["publication_safe"] / "BLIND_SHADOW_RESULT.json", result)
        return result
    except Exception as error:
        _write_json(
            worker_root / "FAILURE.json",
            _authority(
                "SUBSTRATE_TANGIBLE_BLIND_SHADOW_FAILURE/v1",
                {
                    "run_id": handoff["run_id"],
                    "reason": str(error),
                    "events_emitted": emitted,
                    "completed": False,
                },
                status="invalid_or_interrupted",
            ),
        )
        raise


def status(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    live = _r2_live(root)
    present = {name: path.is_file() for name, path in paths.items() if name != "control"}
    state = "await_r2_review"
    policy_sha256 = None
    adapter_contract_sha256 = None
    try:
        policy_sha256 = _read_json(paths["policy"], require_digest=True).get("sha256")
    except Refused:
        state = "safe_hold"
    try:
        adapter_contract_sha256 = _read_json(paths["adapter_contract"], require_digest=True).get("sha256")
    except Refused:
        state = "safe_hold"
    review_valid = False
    if present["review"]:
        try:
            review_valid = _read_json(paths["review"], require_digest=True).get("valid") is True
            if not review_valid:
                state = "repair_diagnosis"
        except Refused:
            state = "safe_hold"
    if review_valid and not present["sealed_design"]:
        state = "blind_shadow_design_review"
    if present["sealed_design"] and not present["calibration_result"]:
        state = "resource_calibration"
    if present["calibration_result"] and not present["preflight"]:
        state = "shadow_preflight"
    if present["preflight"]:
        try:
            preflight_result = _read_json(paths["preflight"], require_digest=True)
            state = "prepare_custody_handoff" if preflight_result.get("admitted") else "safe_hold"
        except Refused:
            state = "safe_hold"
    if live["present"] and not live["complete"]:
        state = "await_r2_review"
    return {
        "schema": "SUBSTRATE_TANGIBLE_NEXT_LAUNCH_STATUS/v1",
        "r2_live": live,
        "artifacts_present": present,
        "pivot_policy_sha256": policy_sha256,
        "adapter_contract_sha256": adapter_contract_sha256,
        "deterministic_next_state": state,
        "activation": False,
    }


def validate_receipt(root: Path, request_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Validate a candidate/control/evaluator adapter receipt before admission.

    This is deliberately a contract check, not a scorer. The independent
    evaluator remains responsible for the scientific verdict.
    """

    contract = _read_json(_paths(root)["adapter_contract"], require_digest=True)
    request = _read_json(request_path)
    receipt = _read_json(receipt_path)
    required_request = contract["request_required"]
    required_receipt = contract["receipt_required"]
    checks = {
        "request_schema": request.get("schema") == contract["request_schema"],
        "receipt_schema": receipt.get("schema") == contract["receipt_schema"],
        "request_required_fields": all(field in request for field in required_request),
        "receipt_required_fields": all(field in receipt for field in required_receipt),
        "role_matches": receipt.get("role") == request.get("role") and receipt.get("role") in contract["roles"],
        "run_matches": receipt.get("run_id") == request.get("run_id"),
        "task_matches": receipt.get("task_id") == request.get("task_id"),
        "manifest_matches": receipt.get("input_manifest_sha256") == request.get("input_manifest_sha256"),
        "nonnegative_elapsed": isinstance(receipt.get("elapsed_seconds"), (int, float)) and receipt.get("elapsed_seconds") >= 0,
        "artifact_list": isinstance(receipt.get("output_artifacts"), list),
        "resource_usage_object": isinstance(receipt.get("resource_usage"), dict),
        "activation_false": not _contains_true_activation(request) and not _contains_true_activation(receipt),
    }
    return {
        "schema": "SUBSTRATE_TANGIBLE_ADAPTER_CONTRACT_CHECK/v1",
        "adapter_contract_sha256": contract["sha256"],
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-R2 tangible blind-shadow scaffold")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap_parser = commands.add_parser("bootstrap", help="create editable launch-control templates")
    bootstrap_parser.add_argument("--overwrite", action="store_true")
    for name, help_text in (
        ("status", "show deterministic next state"),
        ("review-r2", "validate completed R2 evidence before any next-run work"),
        ("seal-design", "seal approved blind-shadow draft and data register"),
        ("run-calibration", "measure independent one/two/four-capsule resource headroom"),
        ("preflight", "recompute storage and isolation admission"),
        ("prepare", "create custody-separated roots without launching a candidate"),
    ):
        commands.add_parser(name, help=help_text)
    custody_parser = commands.add_parser("seal-custody", help="bind custodian-owned tasks, answers, and seed commitment")
    custody_parser.add_argument("--handoff", type=Path, required=True)
    custody_parser.add_argument("--task-manifest", type=Path, required=True)
    custody_parser.add_argument("--answer-manifest", type=Path, required=True)
    custody_parser.add_argument("--seed-file", type=Path, required=True)
    launch_parser = commands.add_parser("launch", help="detach an admitted shadow through launchd")
    launch_parser.add_argument("--handoff", type=Path, required=True)
    worker_parser = commands.add_parser("supervised-run", help="launchd-owned single-writer shadow worker")
    worker_parser.add_argument("--supervision-manifest", type=Path, required=True)
    receipt_parser = commands.add_parser("validate-receipt", help="validate a bound adapter receipt against the sealed contract")
    receipt_parser.add_argument("request", type=Path)
    receipt_parser.add_argument("receipt", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.root.expanduser().resolve()
    try:
        if arguments.command == "bootstrap":
            result = bootstrap(root, overwrite=arguments.overwrite)
        elif arguments.command == "status":
            result = status(root)
        elif arguments.command == "review-r2":
            result = review_r2(root)
        elif arguments.command == "seal-design":
            result = seal_design(root)
        elif arguments.command == "run-calibration":
            result = run_calibration(root)
        elif arguments.command == "preflight":
            result = preflight(root)
        elif arguments.command == "seal-custody":
            result = seal_custody(
                root,
                handoff_path=arguments.handoff,
                task_manifest_path=arguments.task_manifest,
                answer_manifest_path=arguments.answer_manifest,
                seed_file=arguments.seed_file,
            )
        elif arguments.command == "launch":
            result = launch(root, handoff_path=arguments.handoff)
        elif arguments.command == "supervised-run":
            result = supervised_run(root, supervision_manifest=arguments.supervision_manifest)
        elif arguments.command == "validate-receipt":
            result = validate_receipt(root, arguments.request.expanduser().resolve(), arguments.receipt.expanduser().resolve())
        else:
            result = prepare(root)
    except Refused as error:
        _print({"refused": str(error), "activation": False})
        return 2
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
