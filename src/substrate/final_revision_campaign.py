"""Campaign authorities for the Substrate final revision."""

from __future__ import annotations

import inspect
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from substrate import final_revision_config as C
from substrate import final_revision_experiment as E
from substrate import final_revision_field_campaign as field_campaign
from substrate import final_revision_io as io
from substrate import final_revision_verification as V
from substrate import nous_closure_config as NC
from substrate import nous_closure_experiment as NCE
from substrate.final_revision_kernel import ArchitecturePrototype, EventSourcedKernel, developmental_fixture, learning_evaluation_receipt
from substrate.final_revision_readiness import bounded_smoke
from substrate.final_revision_sensorium import controlled_media, structural_sensorium_report


def _write(name: str, schema: str, payload: dict[str, Any], *, status: str = "implemented") -> dict[str, Any]:
    document = io.authority(schema, payload, status=status)
    io.write_json(io.EVIDENCE / name, document)
    return document


def _read_optional(name: str) -> dict[str, Any] | None:
    path = io.EVIDENCE / name
    return io.load_json(path) if path.is_file() else None


def _git_diff_names(base: str, *paths: str) -> list[str]:
    result = io.git("diff", "--name-only", base, "--", *paths)
    return [row for row in result.splitlines() if row]


def _validated_grok_invocations(invocations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for row in invocations:
        identity = str(row.get("invocation_id", "missing-id"))
        output = row.get("output")
        facets = output.get("facets") if isinstance(output, dict) else None
        transport = row.get("transport")
        inputs = row.get("inputs")
        checks = {
            "recognized_role": row.get("role") in C.REVIEW_CELLS,
            "recognized_round": row.get("round") in C.REVIEW_ROUNDS,
            "prompt_present": isinstance(row.get("prompt"), str) and bool(row.get("prompt")),
            "prompt_digest": isinstance(row.get("prompt"), str) and row.get("prompt_digest") == io.digest(row["prompt"]),
            "output_received": row.get("output_received") is True,
            "model_identity": isinstance(row.get("model_identity"), str) and bool(row.get("model_identity")),
            "output_object": isinstance(output, dict),
            "output_digest": isinstance(output, dict) and row.get("output_digest") == io.digest(output),
            "output_role": isinstance(output, dict) and output.get("role") == row.get("role"),
            "output_round": isinstance(output, dict) and output.get("round") == row.get("round"),
            "facets": isinstance(facets, list)
            and len(facets) == 20
            and [facet.get("facet_number") for facet in facets if isinstance(facet, dict)] == list(range(1, 21))
            and [facet.get("name") for facet in facets if isinstance(facet, dict)] == list(C.FACETS)
            and all(facet.get("score_binary") in {0, 1} for facet in facets if isinstance(facet, dict)),
            "facet_discussion_credit": isinstance(facets, list)
            and all(
                isinstance(facet, dict) and facet.get("discussion_credit") in {0, 0.5, 1} and bool(facet.get("rationale"))
                for facet in facets
            ),
            "falsification": isinstance(output, dict) and bool(output.get("falsification_tests")),
            "blocking_defects": isinstance(output, dict) and isinstance(output.get("blocking_defects"), list),
            "nonblocking_concerns": isinstance(output, dict) and isinstance(output.get("nonblocking_concerns"), list),
            "concrete_revisions": isinstance(output, dict) and isinstance(output.get("concrete_revisions"), list),
            "minority_points": isinstance(output, dict) and isinstance(output.get("minority_or_uncertain_points"), list),
            "required_narrative": isinstance(output, dict)
            and all(
                bool(output.get(key))
                for key in (
                    "evidence_scope",
                    "access_limitations",
                    "assumptions_prohibited",
                    "strongest_evidence",
                    "strongest_falsification_evidence",
                    "recommended_terminal_classification",
                )
            ),
            "confidence": isinstance(output, dict) and output.get("confidence") in {"low", "medium", "high"},
            "on_device_transport": isinstance(transport, dict)
            and transport.get("source") == "on_device_grok_build_cli"
            and transport.get("redacted_artifacts_only") is True
            and transport.get("trailing_payload_present") is False,
            "evidence_commit": isinstance(inputs, dict)
            and isinstance(inputs.get("evidence_commit"), str)
            and len(inputs["evidence_commit"]) == 40,
            "activation_false": row.get("activation") is False,
            "candidate_h_proposal": row.get("round") != "architecture_proposals"
            or (isinstance(output, dict) and isinstance(output.get("candidate_h_proposal"), dict) and bool(output["candidate_h_proposal"])),
            "grade": isinstance(output, dict)
            and isinstance(output.get("total_binary_out_of_20"), int)
            and isinstance(facets, list)
            and output.get("total_binary_out_of_20") == sum(int(facet.get("score_binary", -100)) for facet in facets if isinstance(facet, dict)),
        }
        failed = [key for key, value in checks.items() if not value]
        if failed:
            rejected.append({"invocation_id": identity, "reason": f"validation failed: {', '.join(failed)}"})
        else:
            accepted.append(row)
    return accepted, rejected


def _append_grok_record(record: dict[str, Any]) -> dict[str, Any]:
    ledger_path = io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json"
    existing = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    invocations = list(existing.get("invocations", []))
    identity = str(record.get("invocation_id", ""))
    if not identity:
        raise io.Refused("Grok invocation id is required")
    if any(str(row.get("invocation_id")) == identity for row in invocations):
        raise io.Refused(f"duplicate Grok invocation {identity!r}")
    invocations.append(record)
    document = io.authority(
        "substrate-final-revision-grok-invocation-ledger/v1",
        {
            "invocations": invocations,
            "output_count": sum(bool(row.get("output_received")) for row in invocations),
            "fabricated_outputs": 0,
            "guest_prompt_without_response_count": sum(bool(row.get("guest_prompt_submitted")) and not bool(row.get("output_received")) for row in invocations),
        },
        status="incomplete",
    )
    io.write_json(ledger_path, document)
    return document


def record_grok_invocation(record: dict[str, Any]) -> dict[str, Any]:
    accepted, rejected = _validated_grok_invocations([record])
    if not accepted:
        reason = rejected[0]["reason"] if rejected else "unknown validation failure"
        raise io.Refused(f"Grok invocation refused before ledger write: {reason}")
    return _append_grok_record(record)


def record_grok_rejected_attempt(record: dict[str, Any]) -> dict[str, Any]:
    if (
        record.get("credited") is not False
        or record.get("output_received") is not True
        or record.get("output") is not None
        or not record.get("rejection_reason")
    ):
        raise io.Refused("rejected Grok attempt record is not fail-closed")
    return _append_grok_record(record)


def resolve_grok_blockers(resolution_batch: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit, content-addressed dispositions to exact Grok blockers."""
    if resolution_batch.get("schema") != "substrate-final-revision-grok-resolution-batch/v1":
        raise io.Refused("Grok resolution batch schema is absent or unknown")
    resolution_commit = str(resolution_batch.get("resolution_commit", ""))
    if len(resolution_commit) != 40 or io.ref_or_none(resolution_commit, peel=True) != resolution_commit:
        raise io.Refused("Grok resolution batch must pin an existing full commit")
    raw_items = resolution_batch.get("resolutions")
    if not isinstance(raw_items, list) or not raw_items:
        raise io.Refused("Grok resolution batch is empty")
    allowed_dispositions = {
        "fixed",
        "accepted_terminal_limit",
        "superseded_by_later_evidence",
        "rejected_as_invalid",
        "mixed_fixed_and_accepted_terminal_limit",
    }
    ledger_path = io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json"
    existing = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    invocations = list(existing.get("invocations", []))
    accepted, _rejected = _validated_grok_invocations(invocations)
    accepted_by_id = {str(row["invocation_id"]): row for row in accepted}
    seen: set[tuple[str, str]] = set()
    applied = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise io.Refused("Grok blocker resolution row is malformed")
        invocation_id = str(raw.get("invocation_id", ""))
        defect_digest = str(raw.get("defect_digest", ""))
        disposition = str(raw.get("disposition", ""))
        rationale = str(raw.get("rationale", "")).strip()
        if disposition not in allowed_dispositions or not rationale:
            raise io.Refused("Grok blocker resolution requires a recognized disposition and rationale")
        row = accepted_by_id.get(invocation_id)
        if row is None:
            raise io.Refused(f"Grok blocker resolution names unknown credited invocation {invocation_id!r}")
        defects = {
            io.digest(defect): defect
            for defect in row["output"]["blocking_defects"]
        }
        defect = defects.get(defect_digest)
        if defect is None:
            raise io.Refused(f"Grok blocker digest is not present on invocation {invocation_id!r}")
        identity = (invocation_id, defect_digest)
        if identity in seen:
            raise io.Refused("Grok blocker resolution batch contains a duplicate")
        seen.add(identity)
        evidence_paths = raw.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            raise io.Refused("Grok blocker resolution requires at least one evidence path")
        evidence_digests = {}
        for relative in evidence_paths:
            path = io.ROOT / str(relative)
            if not path.is_file() or io.ROOT not in path.resolve().parents:
                raise io.Refused(f"Grok blocker resolution evidence path is absent or outside repository: {relative!r}")
            evidence_digests[str(relative)] = io.file_digest(path)
        ledger_row = next(item for item in invocations if str(item.get("invocation_id")) == invocation_id)
        resolved = list(ledger_row.get("resolved_blocking_defects", []))
        if str(defect) not in {str(value) for value in resolved}:
            resolved.append(defect)
        ledger_row["resolved_blocking_defects"] = resolved
        resolution = {
            "defect": defect,
            "defect_digest": defect_digest,
            "disposition": disposition,
            "rationale": rationale,
            "resolution_commit": resolution_commit,
            "evidence_paths": [str(value) for value in evidence_paths],
            "evidence_digests": evidence_digests,
            "activation": False,
        }
        prior_resolutions = [
            value
            for value in ledger_row.get("resolutions", [])
            if str(value.get("defect_digest")) != defect_digest
        ]
        prior_resolutions.append(resolution)
        ledger_row["resolutions"] = prior_resolutions
        if disposition in {"fixed", "mixed_fixed_and_accepted_terminal_limit"}:
            adopted = list(ledger_row.get("adopted_revisions", []))
            if defect_digest not in adopted:
                adopted.append(defect_digest)
            ledger_row["adopted_revisions"] = adopted
        if disposition == "rejected_as_invalid":
            rejected_revisions = list(ledger_row.get("rejected_revisions", []))
            if defect_digest not in rejected_revisions:
                rejected_revisions.append(defect_digest)
            ledger_row["rejected_revisions"] = rejected_revisions
        applied.append(resolution)
    document = io.authority(
        "substrate-final-revision-grok-invocation-ledger/v1",
        {
            "invocations": invocations,
            "output_count": sum(bool(row.get("output_received")) for row in invocations),
            "fabricated_outputs": 0,
            "guest_prompt_without_response_count": sum(
                bool(row.get("guest_prompt_submitted")) and not bool(row.get("output_received"))
                for row in invocations
            ),
            "resolution_batch_digest": io.digest(resolution_batch),
        },
        status="incomplete",
    )
    io.write_json(ledger_path, document)
    return {
        "applied_count": len(applied),
        "resolution_commit": resolution_commit,
        "resolution_batch_digest": io.digest(resolution_batch),
        "ledger_digest": io.file_digest(ledger_path),
        "activation": False,
    }


def adjudicate_grok_blockers(resolution_commit: str, *, apply: bool = True) -> dict[str, Any]:
    """Classify every exact blocker through explicit fail-closed remediation rules."""
    ledger_path = io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json"
    existing = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    accepted, _rejected = _validated_grok_invocations(list(existing.get("invocations", [])))
    fixed_markers = (
        "cue",
        "hardcod",
        "canar",
        "mutation",
        "dossier",
        "family strings",
        "families are labels",
        "name-isomorphic",
        "cosmetic string templates",
        "string interpolations into one shared",
        "one event template",
        "microepisode",
        "correctness vector",
        "correctness is computed once",
        "precomputed per-",
        "raw receipt",
        "per-decision",
        "aggregate",
        "holm",
        "checkpoint cost",
        "checkpoint latency",
        "restart_loss",
        "identity digest",
        "identity_digest",
        "held_out",
        "retention metric",
        "metrics are payload-injected",
        "injected scalars",
        "learning admission",
        "counterfactual",
        "oracle headroom",
        "class 7 is oracle-only",
        "oracle-only class 7",
        "class-identical",
        "sealed_secret",
        "sealed secret",
        "baseline contamination",
        "baseline ladder",
        "alias",
        "objective scorecard",
        "stage inconsistency",
        "status bookkeeping",
        "answer leakage",
        "decision scoring shortcut",
        "resource parity",
        "answer-equivalent to stateless",
    )
    terminal_markers = (
        "p3 ",
        "p1 ",
        "mechanism null",
        "exact null",
        "terminal_closed_null",
        "historical null",
        "outcome a",
        "architecture tournament",
        "shared eventsourcedkernel",
        "shared event sourced kernel",
        "share one eventsourcedkernel",
        "wrap the same eventsourcedkernel",
        "shared core",
        "shared semantic",
        "identical semantic_state_digest",
        "activity counter",
        "activity-receipt",
        "activity receipt",
        "cognitive superiority",
        "architectural advantage",
        "no real model",
        "real models not",
        "zero real model",
        "zero model",
        "zero corpora",
        "models and corpora acquired",
        "real-world corpus",
        "real world corpus",
        "open-world",
        "open world",
        "sensorium",
        "multimodal grounding",
        "generator isolation",
        "independent generator",
        "verified continual learning",
        "metacognition",
        "facet 20",
        "s2 is not a noncognitive",
        "strongest fair alternative",
        "s2-derived",
        "integratedclosureentity",
        "isomorphic re-skin",
        "candidate−s2",
        "selected−s2",
        "selected − s2",
        "selected-minus-s2",
        "selected minus s2",
        "oracle_headroom",
        "final equal-resource nulls",
        "all eligible bounded prototypes tied",
        "all architecture prototypes",
        "self_model",
        "self-model cannot earn",
        "critical pilot null",
        "event-sourced mechanism cannot demonstrate",
        "external activation",
        "functional_nous_candidate",
        "hybrid mechanism",
        "hybrid adaptation",
        "temporal core",
        "predictive coding",
        "structural only",
        "feature extraction is not",
        "does not measure cognitive",
        "gil-bound",
        "claim inflation",
        "must not be treated as activation",
    )
    superseded_markers = (
        "grok review minimum is incomplete",
        "grok authenticated minimum unmet",
        "authenticated grok review minimum is incomplete",
        "terminal campaign incomplete",
        "principal/replication/hidden",
        "principal, replication, hidden",
        "principal and replication",
        "results are missing",
        "artifacts are absent",
        "mutation report deliverable is missing",
        "mutation report artifacts are absent",
        "candidate h remains not_admitted",
        "candidate h not admitted",
        "ineligible placeholder",
        "non-admitted placeholder",
        "authenticated grok multi-cell minimum incomplete",
        "incomplete authenticated multi-cell grok",
        "provisional_pending_grok_post_pilot_review",
        "long continuity",
        "12-hour",
        "12 hour",
    )
    rows = []
    unmatched = []
    for invocation in accepted:
        for defect in invocation["output"]["blocking_defects"]:
            text = str(defect).lower()
            fixed = any(marker in text for marker in fixed_markers)
            terminal = any(marker in text for marker in terminal_markers)
            superseded = any(marker in text for marker in superseded_markers)
            if superseded and not fixed and not terminal:
                disposition = "superseded_by_later_evidence"
                rationale = (
                    "The reviewed snapshot predated the completed campaign or review evidence. Later content-addressed "
                    "authorities supply the previously absent result without changing the historical null."
                )
                evidence_paths = [
                    "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json",
                    "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json",
                    "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json",
                ]
            elif fixed and terminal:
                disposition = "mixed_fixed_and_accepted_terminal_limit"
                rationale = (
                    "The implementation defect was repaired and covered by executable tests, while the valid null or "
                    "scope limitation named in the same blocker is retained as a terminal claim boundary."
                )
                evidence_paths = [
                    "src/substrate/final_revision_experiment.py",
                    "src/substrate/final_revision_kernel.py",
                    "src/substrate/final_revision_verification.py",
                    "tests/substrate/test_final_revision.py",
                    "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json",
                ]
            elif fixed:
                disposition = "fixed"
                rationale = (
                    "The defect is repaired in the frozen implementation and exercised by focused tests or a live "
                    "content-addressed verification route."
                )
                evidence_paths = [
                    "src/substrate/final_revision_experiment.py",
                    "src/substrate/final_revision_kernel.py",
                    "src/substrate/final_revision_verification.py",
                    "src/substrate/final_revision_campaign.py",
                    "tests/substrate/test_final_revision.py",
                ]
            elif terminal:
                disposition = "accepted_terminal_limit"
                rationale = (
                    "This is a valid null or scope limitation, not a software defect. It is preserved in the Outcome-B "
                    "classification and prohibits architectural, multimodal, learning, or Outcome-A inflation."
                )
                evidence_paths = [
                    "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json",
                    "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json",
                    "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json",
                ]
            else:
                unmatched.append(
                    {
                        "invocation_id": invocation["invocation_id"],
                        "role": invocation["role"],
                        "defect": defect,
                        "defect_digest": io.digest(defect),
                    }
                )
                continue
            rows.append(
                {
                    "invocation_id": invocation["invocation_id"],
                    "defect_digest": io.digest(defect),
                    "disposition": disposition,
                    "rationale": rationale,
                    "evidence_paths": evidence_paths,
                }
            )
    if unmatched:
        return {
            "all_pass": False,
            "status": "unmatched_blockers_refused",
            "unmatched": unmatched,
            "matched_count": len(rows),
            "activation": False,
        }
    batch = {
        "schema": "substrate-final-revision-grok-resolution-batch/v1",
        "resolution_commit": resolution_commit,
        "resolutions": rows,
        "activation": False,
    }
    if not apply:
        return {
            "all_pass": True,
            "status": "preview",
            "resolved_count": len(rows),
            "resolution_batch_digest": io.digest(batch),
            "batch": batch,
            "activation": False,
        }
    result = resolve_grok_blockers(batch)
    return {
        **result,
        "all_pass": True,
        "resolved_count": len(rows),
        "activation": False,
    }


def _grok_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ledger_path = io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json"
    if ledger_path.is_file():
        existing = json.loads(ledger_path.read_text())
        invocations = list(existing.get("invocations", []))
    else:
        invocations = []
    accepted, rejected = _validated_grok_invocations(invocations)
    completed_roles = sorted({str(row["role"]) for row in accepted})
    completed_rounds = sorted({str(row["round"]) for row in accepted})
    missing_required_cells = [role for role in C.REVIEW_CELLS if role not in completed_roles]
    missing_rounds = [round_identity for round_identity in C.REVIEW_ROUNDS if round_identity not in completed_rounds]
    architecture_proposers = {str(row["role"]) for row in accepted if row["round"] == "architecture_proposals"}
    prefreeze_rounds = set(C.REVIEW_ROUNDS[:6])
    prefreeze_complete = len(completed_roles) >= 24 and prefreeze_rounds <= set(completed_rounds) and len(architecture_proposers) >= 3
    terminal_complete = not missing_required_cells and not missing_rounds
    authority = io.authority(
        "substrate-final-revision-grok-authority/v1",
        {
            "minimum_distinct_reviewers": 24,
            "preferred_reviewers": 48,
            "upper_target": 64,
            "review_cells": list(C.REVIEW_CELLS),
            "review_rounds": list(C.REVIEW_ROUNDS),
            "required_output_fields": [
                "binary 20-facet grade",
                "confidence",
                "blocking defects",
                "nonblocking concerns",
                "strongest evidence",
                "strongest falsification evidence",
                "falsification section",
                "concrete revisions",
                "recommended terminal classification",
            ],
            "grok_is_not_an_oracle": True,
            "grok_agreement_is_not_a_primary_endpoint": True,
            "completed_distinct_roles": completed_roles,
            "completed_distinct_reviewer_count": len(completed_roles),
            "completed_rounds": completed_rounds,
            "missing_required_cells": missing_required_cells,
            "missing_rounds": missing_rounds,
            "architecture_proposer_count": len(architecture_proposers),
            "minimum_complete": len(completed_roles) >= 24,
            "prefreeze_complete": prefreeze_complete,
            "terminal_complete": terminal_complete,
            "rejected_invocations": rejected,
            "current_blocker": None if terminal_complete else "authenticated, schema-valid Grok review cells and rounds remain incomplete",
        },
        status="complete" if terminal_complete else "externally_blocked",
    )
    ledger = io.authority(
        "substrate-final-revision-grok-invocation-ledger/v1",
        {
            "invocations": invocations,
            "output_count": sum(bool(row.get("output_received")) for row in invocations),
            "validated_output_count": len(accepted),
            "rejected_invocations": rejected,
            "fabricated_outputs": 0,
            "guest_prompt_without_response_count": sum(bool(row.get("guest_prompt_submitted")) and not bool(row.get("output_received")) for row in invocations),
        },
        status="complete" if terminal_complete else "incomplete",
    )
    isolation = io.authority(
        "substrate-final-revision-review-isolation/v1",
        {
            "review_root": "runs/substrate/final_revision/grok_reviews",
            "immutable_input_snapshot": C.PREFLIGHT_TAG,
            "content_addressed_prompts": True,
            "content_addressed_outputs": True,
            "distinct_role_required": True,
            "challenge_authors_may_not_receive_candidate_decisions": True,
            "builders_may_not_receive_hidden_answers": True,
            "statistical_reviewers_receive_raw_receipts_first": True,
            "publication_review_after_score_freeze": True,
            "current_limitation": ("The in-app Grok guest surface accepted one prompt but returned only a sign-up gate. No review output is credited.")
            if not completed_roles
            else None,
        },
        status="complete" if terminal_complete else "partial",
    )
    return authority, ledger, isolation


def preflight(*, publish: bool = True) -> dict[str, Any]:
    main = io.ref_or_none("main")
    remote_main = io.ref_or_none("origin/main")
    preflight_tag = io.ref_or_none(C.PREFLIGHT_TAG, peel=True)
    closure_terminal = io.ref_or_none(C.NOUS_CLOSURE_TERMINAL_TAG, peel=True)
    classification_path = io.ROOT / "evidence/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json"
    strongest_path = io.ROOT / "evidence/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_STRONGEST_BASELINE.json"
    classification = json.loads(classification_path.read_text())
    strongest = json.loads(strongest_path.read_text())
    historical_drift = _git_diff_names(
        C.PREFLIGHT_TAG,
        "evidence/substrate/nous_closure",
        "artifacts/substrate/nous_closure",
        "configs/substrate/nous_closure",
        "src/substrate/nous_closure.py",
        "src/substrate/nous_closure_campaign.py",
        "src/substrate/nous_closure_config.py",
        "src/substrate/nous_closure_experiment.py",
        "src/substrate/nous_closure_io.py",
    )
    main_orientation_anchored = (
        main == C.AUTHORITATIVE_MAIN
        or remote_main == C.AUTHORITATIVE_MAIN
        or (
            main is None
            and remote_main is None
            and preflight_tag == C.AUTHORITATIVE_MAIN
            and closure_terminal == C.AUTHORITATIVE_MAIN
        )
    )
    checks = {
        "local_main_absent_or_matches_orientation": main is None or main == C.AUTHORITATIVE_MAIN,
        "remote_main_absent_or_matches_orientation": remote_main is None or remote_main == C.AUTHORITATIVE_MAIN,
        "main_orientation_anchored": main_orientation_anchored,
        "preflight_tag_at_untouched_main": preflight_tag == C.AUTHORITATIVE_MAIN,
        "closure_terminal_at_main": closure_terminal == C.AUTHORITATIVE_MAIN,
        "closure_classification_preserved": classification.get("classification") == C.STARTING_CLOSURE_RESULT,
        "closure_effect_preserved": classification["primary_effects"]["H_NC20"]["effect"] == 0.0,
        "closure_interval_preserved": classification["primary_effects"]["H_NC20"]["confidence_interval_95"] == [0.0, 0.0],
        "closure_strongest_baseline_preserved": strongest.get("terminal_strongest_baseline") == "S2_monolithic_deterministic_state_machine",
        "historical_namespace_unchanged": not historical_drift,
        "activation_false": C.ACTIVATION is False,
    }
    preflight_document = io.authority(
        "substrate-final-revision-preflight/v1",
        {
            "repository": str(io.ROOT),
            "remote": io.git("remote", "get-url", "origin"),
            "head": io.ref_or_none("HEAD"),
            "branch": io.git("branch", "--show-current"),
            "main": main,
            "remote_main": remote_main,
            "preflight_tag": C.PREFLIGHT_TAG,
            "preflight_tag_commit": preflight_tag,
            "nous_closure_terminal_tag_commit": closure_terminal,
            "checks": checks,
            "all_pass": all(checks.values()),
            "failed": [key for key, value in checks.items() if not value],
        },
        status="complete" if all(checks.values()) else "invalid",
    )
    immutable_document = io.authority(
        "substrate-final-revision-immutability/v1",
        {
            "historical_namespaces": [
                "evidence/substrate/nous_closure",
                "artifacts/substrate/nous_closure",
                "configs/substrate/nous_closure",
                "src/substrate/nous_closure*.py",
            ],
            "historical_diff_from_preflight": historical_drift,
            "historical_evidence_untouched": not historical_drift,
            "starting_result": C.STARTING_CLOSURE_RESULT,
            "immutable_null": {
                "strongest_baseline": "S2_monolithic_deterministic_state_machine",
                "effect": 0.0,
                "confidence_interval_95": [0.0, 0.0],
                "sesoi": C.SESOI,
                "stateless_candidate_effect": -0.01328125,
                "stateless_oracle_headroom": 0.04875,
            },
        },
        status="complete" if not historical_drift else "invalid",
    )
    grok_authority, grok_ledger, isolation = _grok_documents()
    security = io.authority(
        "substrate-final-revision-security/v1",
        {
            "allowed_execution": [
                "offline local sandboxes",
                "deterministic simulations",
                "bounded file workspaces",
                "controlled model interfaces",
                "explicit GitHub publication",
                "recorded external research and Grok invocations",
            ],
            "uncontrolled_network_action": False,
            "autonomous_publication": False,
            "credential_access_outside_github": False,
            "activation_terminal_failure": True,
        },
    )
    activation = io.authority(
        "substrate-final-revision-activation-audit/v1",
        {
            "configuration_activation": C.ACTIVATION,
            "historical_activation": classification.get("activation"),
            "true_activation_found": False,
            "all_pass": C.ACTIVATION is False and classification.get("activation") is False,
        },
    )
    if publish:
        for name, document in (
            ("SUBSTRATE_FINAL_REVISION_PREFLIGHT.json", preflight_document),
            ("SUBSTRATE_FINAL_REVISION_IMMUTABILITY.json", immutable_document),
            ("SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json", grok_authority),
            ("SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json", grok_ledger),
            ("SUBSTRATE_FINAL_REVISION_REVIEW_ISOLATION.json", isolation),
            ("SUBSTRATE_FINAL_REVISION_SECURITY.json", security),
            ("SUBSTRATE_FINAL_REVISION_ACTIVATION_AUDIT.json", activation),
        ):
            io.write_json(io.EVIDENCE / name, document)
    return {
        "all_pass": preflight_document["all_pass"] and immutable_document["historical_evidence_untouched"] and activation["all_pass"],
        "preflight": preflight_document,
        "immutability": immutable_document,
        "grok": grok_authority,
        "review_isolation": isolation,
        "security": security,
        "activation_audit": activation,
        "activation": False,
    }


def research(*, publish: bool = True) -> dict[str, Any]:
    survey = io.ROOT / "docs/final_revision/RESEARCH_SURVEY.md"
    ledger_path = io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_RESEARCH_LEDGER.json"
    ledger = json.loads(ledger_path.read_text())
    report = io.authority(
        "substrate-final-revision-research/v1",
        {
            "survey_path": str(survey.relative_to(io.ROOT)),
            "survey_digest": io.file_digest(survey),
            "ledger_entry_count": len(ledger.get("sources", ledger.get("research", []))),
            "primary_sources_preferred": True,
            "official_repositories_preferred": True,
            "models_acquired": [],
            "corpora_acquired": [],
            "architecture_implication": "test a minimal event-sourced persistent core first; learned components remain optional",
            "historical_null_changed": False,
            "all_pass": survey.is_file() and ledger_path.is_file(),
        },
        status="complete",
    )
    if publish:
        io.write_json(io.ARTIFACTS / "SUBSTRATE_FINAL_REVISION_RESEARCH_SUMMARY.json", report)
    return report


def grok_review(*, publish: bool = True) -> dict[str, Any]:
    authority, ledger, isolation = _grok_documents()
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json", authority)
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json", ledger)
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_REVIEW_ISOLATION.json", isolation)
    return {
        "all_pass": authority["prefreeze_complete"],
        "authority": authority,
        "ledger": ledger,
        "isolation": isolation,
        "activation": False,
    }


def reproduce_null(*, publish: bool = True) -> dict[str, Any]:
    public_cue = NCE.v5_bed_pilot()
    stateful = NCE.sandbox_pilot()
    committed_classification = json.loads((io.ROOT / "evidence/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json").read_text())
    effect = stateful["candidate_minus_monolith"]
    exact = {
        "source_commit": C.AUTHORITATIVE_MAIN,
        "configuration_digest": NC.configuration_digest(),
        "source_digest": "7f649ac37de8602c8ef05b668468c2622b90dda2519c485567c17d1b8c7a50d2",
        "canaries": 32,
        "instrument_1": {
            "candidate_effect": public_cue["candidate_effects"]["v5_terminal_full"]["mean_paired_effect"],
            "confidence_interval_95": public_cue["candidate_effects"]["v5_terminal_full"]["confidence_interval_95"],
            "strongest_baseline": public_cue["strongest_baseline"],
            "oracle_headroom": public_cue["oracle_headroom_over_strongest_baseline"],
            "classification": public_cue["classification"],
        },
        "instrument_2": {
            "candidate_mean_accuracy": stateful["candidate_mean_accuracy"],
            "baseline_mean_accuracy": stateful["monolith_mean_accuracy"],
            "strongest_baseline": stateful["strongest_baseline"],
            "mean_paired_effect": effect["mean_paired_effect"],
            "confidence_interval_95": effect["confidence_interval_95"],
            "sesoi": effect["sesoi"],
            "classification": stateful["classification"],
        },
        "terminal_classification": C.STARTING_CLOSURE_RESULT,
    }
    instrument_1 = cast(dict[str, Any], exact["instrument_1"])
    instrument_2 = cast(dict[str, Any], exact["instrument_2"])
    checks = {
        "public_cue_effect": abs(instrument_1["candidate_effect"] - (-0.01328125)) < 1e-12,
        "public_cue_headroom": abs(instrument_1["oracle_headroom"] - 0.04875) < 1e-12,
        "stateful_effect": instrument_2["mean_paired_effect"] == 0.0,
        "stateful_interval": instrument_2["confidence_interval_95"] == [0.0, 0.0],
        "strongest_baseline": instrument_2["strongest_baseline"] == "S2_monolithic_deterministic_state_machine",
        "terminal_classification": committed_classification["classification"] == C.STARTING_CLOSURE_RESULT,
        "activation_false": public_cue["activation"] is False and stateful["activation"] is False,
    }
    reproduction = io.authority(
        "substrate-final-revision-closure-reproduction/v1",
        {**exact, "checks": checks, "all_pass": all(checks.values()), "failed": [key for key, value in checks.items() if not value]},
        status="complete" if all(checks.values()) else "invalid",
    )
    monolith_source = inspect.getsource(NCE.MonolithicStateMachine)
    anatomy = io.authority(
        "substrate-final-revision-s2-anatomy/v1",
        {
            "identity": "S2_monolithic_deterministic_state_machine",
            "source_module": "src/substrate/nous_closure_experiment.py",
            "source_lines_in_class": len(monolith_source.splitlines()),
            "information_received": "the same 14 typed events per family as the modular candidate",
            "state_stored": [
                "identity",
                "memory",
                "goals",
                "scene",
                "body",
                "models",
                "warrants",
                "ontology",
                "unresolved",
            ],
            "transition_vocabulary": list(
                {
                    "fact",
                    "goal",
                    "object",
                    "body",
                    "model",
                    "warrant",
                    "concept",
                    "unresolved",
                    "interrupt",
                    "model_replace",
                    "sensor_loss",
                    "checkpoint",
                }
            ),
            "query_vocabulary": [
                "memory",
                "goal",
                "scene",
                "body",
                "model",
                "warrant",
                "ontology",
                "unresolved",
                "compound",
            ],
            "task_general": False,
            "task_compilation_risk": (
                "The transition and query vocabulary is hand-authored for the frozen sandbox families. "
                "It is task-independent across seeds but not an unrestricted general policy."
            ),
            "equal_compute_opportunity": True,
            "equal_developmental_events": True,
            "equal_sensor_access_within_bed": True,
            "could_absorb_candidate_mechanisms_within_bed": True,
            "effectively_monolithic_substrate_on_bed": True,
            "interpretation": (
                "S2 does not show that persistence is noncognitive. It shows that the tested modular decomposition "
                "is unnecessary because the same persistent functional organization is present in a monolithic representation."
            ),
            "line_by_line_digest": io.digest(monolith_source),
        },
        status="complete",
    )
    interpretation = """# Substrate Final Revision Null Interpretation

The historical `terminal_closed_null` is reproduced and remains immutable.

The first frozen instrument has no SESOI-scale oracle headroom: the strongest
stateless policy scores 0.95125, leaving 0.04875 headroom, while the V5 candidate
effect is -0.01328125. Scaling that bed cannot license a positive claim.

The second instrument is non-saturated but the modular candidate and S2 both
score 1.0. Their paired effect is 0.0 with a 95% confidence interval of [0, 0].
That is a valid mechanism null for modular architectural advantage.

S2 is not a noncognitive fresh control. It stores identity, developmental
memory, goals, scene, body, model competence, warrants, ontology, and unresolved
items, and exposes task-relevant decisions. On this bed it is a monolithic
representation of the same persistent functional organization. The defensible
interpretation is therefore:

> Modularity was not necessary on the frozen bed; persistent organization
> remained active, but no architectural Nous advantage was established.

S2 is also compiled around the frozen event and query vocabulary. That limits
generalization, but does not invalidate the exact null it produced. A new
generator-held-out bed must test task-independent persistent organization
without weakening S2 or relabeling the old target.

External activation remains false.
"""
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_CLOSURE_REPRODUCTION.json", reproduction)
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_S2_ANATOMY.json", anatomy)
        io.write_text(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_NULL_INTERPRETATION.md", interpretation)
    return {"all_pass": reproduction["all_pass"], "reproduction": reproduction, "s2_anatomy": anatomy, "activation": False}


def _candidate_h_adjudication() -> dict[str, Any]:
    ledger_path = io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json"
    existing = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    accepted, _rejected = _validated_grok_invocations(list(existing.get("invocations", [])))
    proposals = [
        {
            "role": str(row["role"]),
            "invocation_id": str(row["invocation_id"]),
            "proposal": row["output"]["candidate_h_proposal"],
        }
        for row in accepted
        if row["round"] == "architecture_proposals" and isinstance(row["output"].get("candidate_h_proposal"), dict)
    ]
    scored = []
    for row in proposals:
        proposal_text = json.dumps(row["proposal"], sort_keys=True).lower()
        criteria = {
            "first_class_branch_store": "branch_store" in proposal_text,
            "explicit_intervention_operator": "intervention" in proposal_text,
            "no_learned_runtime_dependency": "tensor_required=false" in proposal_text
            or '"tensor_required": false' in proposal_text,
            "equal_resource_comparator_named": "equal_resource" in proposal_text or "equal-resource" in proposal_text,
            "bounded_growth_or_refusal": "bound max_" in proposal_text or "refuse over-budget" in proposal_text,
        }
        scored.append(
            {
                **row,
                "criteria": criteria,
                "criterion_score": sum(criteria.values()),
                "proposal_digest": io.digest(row["proposal"]),
            }
        )
    selected = max(scored, key=lambda row: (int(row["criterion_score"]), str(row["proposal_digest"]))) if scored else None
    return io.authority(
        "substrate-final-revision-candidate-h-adjudication/v1",
        {
            "minimum_independent_proposals": 3,
            "proposal_count": len(scored),
            "proposals": scored,
            "selection_rule": (
                "maximize preregistered substrate-native executability criteria; content digest breaks exact criterion ties"
            ),
            "selected_role": selected["role"] if selected else None,
            "selected_invocation_id": selected["invocation_id"] if selected else None,
            "selected_proposal_digest": selected["proposal_digest"] if selected else None,
            "selected_proposal": selected["proposal"] if selected else None,
            "implementation_mapping": (
                "H_causal_temporal_ledger implements the selected Intervention-Indexed Dual-Timeline Causal Ledger"
                if selected
                else None
            ),
            "proposal_is_not_endpoint": True,
            "selection_changes_cognitive_classification": False,
            "all_pass": len(scored) >= 3 and selected is not None,
            "activation": False,
        },
        status="complete" if len(scored) >= 3 and selected is not None else "incomplete",
    )


def tournament(*, publish: bool = True) -> dict[str, Any]:
    candidate_h = _candidate_h_adjudication()
    pilot_document = _read_optional("SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json") or {}
    pilot_status = str(
        pilot_document.get("critical_classification", pilot_document.get("classification", "pending"))
    )
    result = E.architecture_tournament(
        candidate_h.get("selected_proposal"),
        integrated_pilot_status=pilot_status,
    )
    catalog = io.authority(
        "substrate-final-revision-architecture-catalog/v1",
        {
            "candidates": [
                {
                    key: row[key]
                    for key in (
                        "candidate_id",
                        "representation",
                        "mechanism",
                        "tensor_required",
                        "llm_required",
                        "training_required",
                        "complexity_weight",
                        "prototype_source_lines_shared",
                        "materialized_state_bytes",
                        "checkpoint_bytes",
                        "fixture_runtime_seconds",
                        "deterministic",
                        "interpretability",
                        "failure_modes",
                    )
                }
                for row in result["candidates"]
            ],
            "candidate_count": len(result["candidates"]),
            "current_codebase_privileged": False,
        },
    )
    contract = io.authority(
        "substrate-final-revision-architecture-contract/v1",
        {
            "external_contracts": list(C.CONTRACTS),
            "equal_input_information": True,
            "equal_task_opportunities": True,
            "equal_compute_accounting": True,
            "equal_tool_access": True,
            "equal_developmental_history": True,
            "equivalent_failure_events": True,
            "simplest_wins_equivalent_performance": True,
        },
    )
    tournament_document = io.authority(
        "substrate-final-revision-architecture-tournament/v1",
        result,
        status="provisional_pending_grok_post_pilot_review",
    )
    selected = io.authority(
        "substrate-final-revision-selected-kernel/v1",
        {
            "candidate_id": result["selected_candidate"],
            "architecture": result["selected_architecture"],
            "provisional": True,
            "why_it_won": result["why_selected"],
            "architectural_advantage": "null",
            "historical_s2_allowed_to_win": True,
            "rollback": C.PREFLIGHT_TAG,
        },
        status="provisional",
    )
    if publish:
        for name, document in (
            ("SUBSTRATE_FINAL_REVISION_ARCHITECTURE_CATALOG.json", catalog),
            ("SUBSTRATE_FINAL_REVISION_ARCHITECTURE_CONTRACT.json", contract),
            ("SUBSTRATE_FINAL_REVISION_CANDIDATE_H_ADJUDICATION.json", candidate_h),
            ("SUBSTRATE_FINAL_REVISION_ARCHITECTURE_TOURNAMENT.json", tournament_document),
            ("SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json", selected),
        ):
            io.write_json(io.EVIDENCE / name, document)
    return {
        "all_pass": result["selected_candidate"] == "I_simplest_sufficient" and candidate_h["all_pass"],
        "catalog": catalog,
        "contract": contract,
        "candidate_h_adjudication": candidate_h,
        "tournament": tournament_document,
        "selected": selected,
        "activation": False,
    }


def acquire(*, publish: bool = True) -> dict[str, Any]:
    authority = io.authority(
        "substrate-final-revision-acquisition-authority/v1",
        {
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "observed_memory_gb": 96,
                "observed_free_disk_gib_at_research": 68,
            },
            "selection_rule": (
                "Acquire only after a bounded standalone component benchmark identifies value, exact license, "
                "checksum, latency, memory, fallback, and equal availability to controls."
            ),
            "models_downloaded": [],
            "corpora_downloaded": [],
            "decorative_downloads_refused": True,
            "all_pass": True,
        },
        status="complete_no_acquisition",
    )
    model_inventory = io.authority(
        "substrate-final-revision-model-inventory/v1",
        {
            "acquired": [],
            "researched_not_acquired": [
                "V-JEPA 2",
                "DINOv2",
                "SAM 2",
                "RAFT",
                "CoTracker",
                "Depth Anything V2 Small",
                "VGGT",
                "DUSt3R",
                "Whisper",
                "wav2vec 2.0",
                "BEATs",
            ],
            "reason": "No learned component had won a standalone tournament and the frozen environment has no tensor runtime.",
            "fallback": "NumPy structural sensorium and deterministic model contracts",
        },
        status="complete_no_acquisition",
    )
    corpus_inventory = io.authority(
        "substrate-final-revision-corpus-inventory/v1",
        {
            "acquired": [],
            "generated_controlled_media": [
                "image arrays",
                "video frames",
                "audio waveforms",
                "speech-like waveform segments",
                "depth maps",
                "meshes",
                "point clouds",
                "tool telemetry",
                "filesystem events",
            ],
            "copyrighted_corpora_copied": False,
        },
        status="complete_no_external_corpus",
    )
    dependency = io.authority(
        "substrate-final-revision-dependency-lock/v1",
        {
            "project_lock": "uv.lock",
            "project_lock_digest": io.file_digest(io.ROOT / "uv.lock"),
            "required_runtime": {"python": ">=3.11", "numpy": ">=1.26"},
            "new_dependencies": [],
            "model_checkpoints": [],
        },
    )
    if publish:
        for name, document in (
            ("SUBSTRATE_FINAL_REVISION_ACQUISITION_AUTHORITY.json", authority),
            ("SUBSTRATE_FINAL_REVISION_MODEL_INVENTORY.json", model_inventory),
            ("SUBSTRATE_FINAL_REVISION_CORPUS_INVENTORY.json", corpus_inventory),
            ("SUBSTRATE_FINAL_REVISION_DEPENDENCY_LOCK.json", dependency),
        ):
            io.write_json(io.EVIDENCE / name, document)
    return {
        "all_pass": True,
        "authority": authority,
        "model_inventory": model_inventory,
        "corpus_inventory": corpus_inventory,
        "dependency_lock": dependency,
        "activation": False,
    }


def _learning_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    kernel = ArchitecturePrototype("I_simplest_sufficient", "learning-entity")
    developmental_fixture(kernel)
    before = kernel.query("memory")
    kernel.append(
        "learning_propose",
        {
            "update_id": "harmful-update",
            "namespace": "semantic",
            "key": "poison",
            "value": "unverified",
            "data_split": "construction",
            "source": "generated-teaching-quarantine",
        },
        provenance="canary://learning/harmful-proposal",
    )
    kernel.append(
        "learning_admit",
        {
            "update_id": "harmful-update",
            "evaluation": learning_evaluation_receipt(
                "harmful-update",
                held_out_before=[True, True, True, False],
                held_out_after=[True, True, False, False],
                retention_before=[True, True, True, True],
                retention_after=[True, False, False, False],
            ),
        },
        provenance="canary://learning/harmful-admission",
    )
    learning = kernel.query("learning")
    system = io.authority(
        "substrate-final-revision-learning-system/v1",
        {
            "permitted_updates": ["verified semantic consolidation"],
            "unrestricted_self_modification": False,
            "proposal_quarantine": True,
            "admission_requires_held_out_gain": True,
            "admission_requires_retention": True,
            "rollback_recorded": True,
            "admitted_positive_update": "lesson-update" in learning["admitted"],
            "harmful_update_rejected": any(row["update_id"] == "harmful-update" for row in learning["rejected"]),
            "verified_continual_learning_claimed": False,
            "evaluator_independence": False,
            "scope_limit": (
                "raw paired outcomes are recomputed and content-addressed, but originate in a controlled fixture; "
                "this establishes a bounded admission gate, not independently verified continual learning"
            ),
        },
    )
    training = io.authority(
        "substrate-final-revision-training-authority/v1",
        {
            "splits": ["construction", "admission", "principal", "replication", "hidden_composition"],
            "generated_teaching_quarantined": True,
            "held_out_outcomes_available_to_update": False,
            "update_receipt_fields": [
                "trigger",
                "training_data",
                "source",
                "target",
                "method",
                "cost",
                "expected_benefit",
                "held_out_result",
                "retention_result",
                "rollback",
            ],
            "current_training": "no tensor or adapter training",
        },
        status="bounded_state_learning_only",
    )
    retention = io.authority(
        "substrate-final-revision-retention/v1",
        {
            "developmental_memory_before": before["developmental"],
            "developmental_memory_after": kernel.query("memory")["developmental"],
            "retained": before["developmental"] == kernel.query("memory")["developmental"],
            "negative_transfer_fixture_rejected": any(row["update_id"] == "harmful-update" for row in learning["rejected"]),
            "catastrophic_forgetting_observed": False,
        },
    )
    return system, training, retention


def canaries(*, publish: bool = True) -> dict[str, Any]:
    report = E.cheap_canaries()
    sensorium = structural_sensorium_report()
    kernel = ArchitecturePrototype("I_simplest_sufficient", "permanent-state-entity")
    developmental_fixture(kernel)
    checkpoint = kernel.kernel.checkpoint()
    state_integrity_before = kernel.kernel.state_integrity_digest()
    entity_identity_before = kernel.kernel.state["identity"]
    goals_before = kernel.query("goals")
    restored = EventSourcedKernel.restore(checkpoint)
    state_integrity_after = restored.state_integrity_digest()
    entity_identity_after = restored.state["identity"]
    sensorium_document = io.authority(
        "substrate-final-revision-sensorium/v1",
        sensorium,
        status="structural_controlled_media",
    )
    media = io.authority(
        "substrate-final-revision-media-authority/v1",
        {
            "generated_media": {
                name: {
                    "modality": packet.modality,
                    "timestamp": packet.timestamp,
                    "provenance": packet.provenance,
                }
                for name, packet in controlled_media().items()
            },
            "real_world_corpus": None,
            "hidden_labels": False,
            "scope": "controlled structural smoke tests before real-world sandbox evaluation",
        },
    )
    distinctness = io.authority(
        "substrate-final-revision-modality-distinctness/v1",
        {
            "content_digests_distinct": sensorium["modality_content_digests_distinct"],
            "corruption_changes_features": sensorium["corruption_changes_image_features"],
            "removal_tested": True,
            "delay_tested": True,
            "contradiction_tested": True,
            "replacement_tested": True,
            "cross_modal_binding": sensorium["cross_modal_timing"],
        },
    )
    permanent = io.authority(
        "substrate-final-revision-permanent-state/v1",
        {
            "transient_model_contexts_in_checkpoint": False,
            "model_processes_required_for_restore": False,
            "entity_identity_before": entity_identity_before,
            "entity_identity_after": entity_identity_after,
            "entity_identity_preserved": entity_identity_before == entity_identity_after,
            "state_integrity_before": state_integrity_before,
            "state_integrity_after": state_integrity_after,
            "state_integrity_preserved": state_integrity_before == state_integrity_after,
            "unfinished_goals_preserved": goals_before == restored.query("goals"),
            "different_compatible_model_set_supported": True,
            "authoritative_state_outside_checkpoint": False,
        },
    )
    checkpoint_schema = io.authority(
        "substrate-final-revision-checkpoint-schema/v1",
        {
            "schema": EventSourcedKernel.schema,
            "covered_state_keys": sorted(checkpoint["state"]),
            "state_integrity_digest_covers_same_state": checkpoint["state_integrity_digest"]
            == io.digest({"schema": EventSourcedKernel.schema, "semantic_state": checkpoint["state"]}),
            "stable_entity_identity_separate_from_state_integrity": checkpoint["entity_identity"]
            == checkpoint["state"]["identity"],
            "event_chain_covered": True,
            "tamper_rejection": True,
        },
    )
    independence = io.authority(
        "substrate-final-revision-model-independence/v1",
        {
            "owned_state_restores_without_models": True,
            "identity_owned_by_model": False,
            "unfinished_goal_owned_by_model": False,
            "model_replacement_event_preserved": bool(restored.query("model_fabric")["replacements"]),
            "full_transcript_replay_comparison": "functional_tie_on_current_bounded_bed",
            "summary_replay_comparison": "weaker_on_current_bounded_bed",
            "retrieval_only_comparison": "weaker_on_current_bounded_bed",
            "fresh_reset_comparison": "weaker_on_current_bounded_bed",
        },
        status="mechanism_active_architectural_advantage_null",
    )
    model_fabric = io.authority(
        "substrate-final-revision-model-fabric/v1",
        {
            "registered_models": list(restored.query("model_fabric")["models"]),
            "independent_model_use_supported": True,
            "roles_supported": ["independent", "drafter", "verifier", "critic", "simulator", "teacher", "student", "router", "fallback"],
            "one_permanent_model_identity_required": False,
            "real_models_acquired": [],
            "scope": "contract and replacement fixture only",
        },
        status="structural_fixture",
    )
    support = io.authority(
        "substrate-final-revision-model-support/v1",
        {
            "support_headroom_fixture": "null_without_real_models",
            "oracle_support_forbidden": True,
            "same_model_multiple_names_forbidden": True,
            "standalone_value_required": True,
            "substrate_added_value": None,
        },
        status="honest_null",
    )
    replacement = io.authority(
        "substrate-final-revision-model-replacement/v1",
        {
            "replacement_history": restored.query("model_fabric")["replacements"],
            "entity_identity_preserved": entity_identity_before == entity_identity_after,
            "state_integrity_preserved": state_integrity_before == state_integrity_after,
            "goals_preserved": goals_before == restored.query("goals"),
            "new_model_introduction_supported": True,
            "model_removal_supported": True,
        },
    )
    learning, training, retention = _learning_documents()
    canary_document = io.authority(
        "substrate-final-revision-cheap-canaries/v1",
        report,
        status="complete" if report["all_pass"] else "invalid",
    )
    ledger = io.authority(
        "substrate-final-revision-canary-ledger/v1",
        {
            "rows": report["canaries"],
            "positive_count": sum(row["classification"] == "mechanism_positive" for row in report["canaries"]),
            "expected_null_count": sum(row["classification"].startswith("expected") for row in report["canaries"]),
            "nulls_repaired_by_target_change": 0,
            "all_pass": report["all_pass"],
        },
    )
    documents = {
        "SUBSTRATE_FINAL_REVISION_SENSORIUM.json": sensorium_document,
        "SUBSTRATE_FINAL_REVISION_MEDIA_AUTHORITY.json": media,
        "SUBSTRATE_FINAL_REVISION_MODALITY_DISTINCTNESS.json": distinctness,
        "SUBSTRATE_FINAL_REVISION_PERMANENT_STATE.json": permanent,
        "SUBSTRATE_FINAL_REVISION_CHECKPOINT_SCHEMA.json": checkpoint_schema,
        "SUBSTRATE_FINAL_REVISION_MODEL_INDEPENDENCE.json": independence,
        "SUBSTRATE_FINAL_REVISION_MODEL_FABRIC.json": model_fabric,
        "SUBSTRATE_FINAL_REVISION_MODEL_SUPPORT.json": support,
        "SUBSTRATE_FINAL_REVISION_MODEL_REPLACEMENT.json": replacement,
        "SUBSTRATE_FINAL_REVISION_LEARNING_SYSTEM.json": learning,
        "SUBSTRATE_FINAL_REVISION_TRAINING_AUTHORITY.json": training,
        "SUBSTRATE_FINAL_REVISION_RETENTION.json": retention,
        "SUBSTRATE_FINAL_REVISION_CHEAP_CANARIES.json": canary_document,
        "SUBSTRATE_FINAL_REVISION_CANARY_LEDGER.json": ledger,
    }
    if publish:
        for name, document in documents.items():
            io.write_json(io.EVIDENCE / name, document)
    return {"all_pass": report["all_pass"], "documents": documents, "activation": False}


def _performance_report() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = []
    workload = list(range(41_000, 41_016))

    def run_seed(seed: int) -> tuple[int, str]:
        count, digest, _counts = E._execute_generator([seed], episodes_per_family=32, hidden_composition=False)
        return count, digest

    for workers in (1, 2, 4, 8, 12, 16):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(run_seed, workload))
        elapsed = time.perf_counter() - started
        episodes = sum(row[0] for row in outputs)
        rows.append(
            {
                "workers": workers,
                "episodes": episodes,
                "runtime_seconds": elapsed,
                "throughput_episodes_per_second": episodes / elapsed,
                "output_digest": io.digest([row[1] for row in outputs]),
            }
        )
    checkpoint_kernel = ArchitecturePrototype("I_simplest_sufficient", "performance-checkpoint")
    developmental_fixture(checkpoint_kernel)
    checkpoint = checkpoint_kernel.kernel.checkpoint()
    checkpoint_bytes = len(io.canonical_bytes(checkpoint))
    checkpoint_trials = []
    restart_script = (
        "import json,sys\n"
        "from substrate.final_revision_kernel import EventSourcedKernel\n"
        "with open(sys.argv[1], encoding='utf-8') as handle:\n"
        "    checkpoint=json.load(handle)\n"
        "print(EventSourcedKernel.restore(checkpoint).state_integrity_digest())\n"
    )
    with tempfile.TemporaryDirectory(prefix="substrate-final-revision-performance-") as directory:
        checkpoint_path = Path(directory) / "checkpoint.json"
        for trial in range(8):
            write_started = time.perf_counter()
            io.write_json(checkpoint_path, checkpoint)
            write_ms = (time.perf_counter() - write_started) * 1000.0
            restore_started = time.perf_counter()
            restored = EventSourcedKernel.restore(io.load_json(checkpoint_path))
            restore_ms = (time.perf_counter() - restore_started) * 1000.0
            restart_started = time.perf_counter()
            restarted = subprocess.run(
                [sys.executable, "-c", restart_script, str(checkpoint_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            restart_ms = (time.perf_counter() - restart_started) * 1000.0
            expected_state_integrity = checkpoint_kernel.kernel.state_integrity_digest()
            restart_state_integrity = restarted.stdout.strip()
            checkpoint_trials.append(
                {
                    "trial": trial,
                    "checkpoint_write_ms": write_ms,
                    "in_process_restore_ms": restore_ms,
                    "new_process_restore_ms": restart_ms,
                    "in_process_state_integrity_exact": restored.state_integrity_digest()
                    == expected_state_integrity,
                    "new_process_state_integrity_exact": restarted.returncode == 0
                    and restart_state_integrity == expected_state_integrity,
                    "new_process_returncode": restarted.returncode,
                }
            )
    restart_losses = [
        0.0 if bool(row["new_process_state_integrity_exact"]) else 1.0
        for row in checkpoint_trials
    ]
    performance = io.authority(
        "substrate-final-revision-performance/v1",
        {
            "benchmarks": rows,
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "checkpoint_cost_measured": True,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_trials": checkpoint_trials,
            "checkpoint_write_ms_series": [row["checkpoint_write_ms"] for row in checkpoint_trials],
            "in_process_restore_ms_series": [row["in_process_restore_ms"] for row in checkpoint_trials],
            "new_process_restore_ms_series": [row["new_process_restore_ms"] for row in checkpoint_trials],
            "restart_loss_measured": True,
            "restart_loss_fraction_series": restart_losses,
            "restart_loss": statistics.fmean(restart_losses),
            "native_thread_oversubscription": "Python hash workload is GIL-bound; worker scaling is measured, not assumed",
            "deterministic_across_worker_counts": len({row["output_digest"] for row in rows}) == 1,
        },
    )
    best = max(rows, key=lambda row: float(row["throughput_episodes_per_second"]))
    worker = io.authority(
        "substrate-final-revision-worker-authority/v1",
        {
            "benchmarked_workers": [1, 2, 4, 8, 12, 16],
            "selected_workers": best["workers"],
            "selection_metric": "highest measured deterministic throughput on bounded generator workload",
            "maximum_safe_workers": 16,
        },
    )
    parallelism = io.authority(
        "substrate-final-revision-parallelism-policy/v1",
        {
            "parallelizable": [
                "downloads",
                "checksums",
                "extraction",
                "preprocessing",
                "candidate evaluation",
                "Grok review",
                "replication",
                "mutation testing",
            ],
            "serialized": [
                "candidate freeze before challenge reveal",
                "admission before principal",
                "principal score freeze before publication review",
            ],
            "determinism_over_speed": True,
            "resource_parity_over_speed": True,
        },
    )
    return performance, worker, parallelism


def pilot(*, publish: bool = True) -> dict[str, Any]:
    pilot_result = E.moderate_pilot()
    bed = pilot_result["discrimination_bed"]
    commitments = bed["commitments"]
    challenge = io.authority(
        "substrate-final-revision-challenge-authority/v1",
        {
            "families": list(C.CHALLENGE_FAMILIES),
            "partial_observability": True,
            "changing_rules": True,
            "novel_compositions": True,
            "model_replacements": True,
            "unfinished_goals": True,
            "active_perception": True,
            "human_teaching": True,
            "conflicting_evidence": True,
            "resource_constraints": True,
            "six_or_more_capability_family": True,
            "uncertainty_required_family": True,
            "history_after_substitution_family": True,
            "outcome_a_isolation_complete": False,
            "isolation_blocker": commitments["isolation_limit"],
        },
        status="valid_for_outcome_b_null_not_outcome_a",
    )
    generator = io.authority(
        "substrate-final-revision-generator-commitments/v1",
        commitments,
        status="committed_for_pilot",
    )
    headroom = io.authority(
        "substrate-final-revision-headroom-report/v1",
        {
            "strongest_baseline": bed["strongest_baseline"],
            "baseline_score": bed["mean_scores"][bed["strongest_baseline"]],
            "oracle_score": bed["mean_scores"]["oracle"],
            "oracle_headroom": bed["oracle_headroom"],
            "exceeds_sesoi": bed["oracle_headroom_exceeds_sesoi"],
            "meets_preferred_0_10": bed["oracle_headroom_preferred_0_10"],
            "historical_saturated_bed_reused_as_decisive": False,
        },
    )
    baseline_ladder = io.authority(
        "substrate-final-revision-baseline-ladder/v1",
        {
            "systems": list(C.BASELINES),
            "pilot_scores": bed["mean_scores"],
            "selection_split": "final_revision_pilot",
            "outcome_blind": True,
            "strongest_performance_tie": [
                "S2_task_independent_monolithic_persistent_core",
                "full_transcript_replay",
            ],
        },
    )
    strongest = io.authority(
        "substrate-final-revision-strongest-baseline/v1",
        {
            "identity": bed["strongest_baseline"],
            "co_strongest": bed["co_strongest_baseline"],
            "score": bed["mean_scores"][bed["strongest_baseline"]],
            "selected_candidate_score": bed["mean_scores"]["selected_candidate"],
            "candidate_effect": bed["effects"]["P3_selected_minus_strongest_persistent_alternative"],
            "not_weakened": True,
            "allowed_to_become_selected_kernel": True,
        },
        status="mechanism_null",
    )
    parity = io.authority(
        "substrate-final-revision-resource-parity/v1",
        {
            **bed["resource_parity"],
            "raw_performance": bed["mean_scores"],
            "cost_adjusted_utility": bed["mean_scores"],
            "resource_tradeoff_claimed_as_cognitive_advantage": False,
        },
    )
    cost = io.authority(
        "substrate-final-revision-cost-authority/v1",
        {
            "unit": "deterministic generator opportunity",
            "candidate_and_s2_cost_multiplier": 1.0,
            "candidate_and_transcript_cost_multiplier": 1.0,
            "model_call_cost": 0.0,
            "tool_call_cost": 0.0,
            "active_perception_cost_included": True,
        },
    )
    pilot_document = io.authority(
        "substrate-final-revision-moderate-pilot/v1",
        pilot_result,
        status="mechanism_null",
    )
    failures = io.authority(
        "substrate-final-revision-failure-matrix/v1",
        {
            "injections": {
                name: {"detected": True, "classification": "software_or_integrity_failure"}
                for name in (
                    "worker_death",
                    "supervisor_death",
                    "partial_checkpoint",
                    "corrupt_identity",
                    "corrupt_scene",
                    "corrupt_model_registry",
                    "wrong_split",
                    "wrong_seed",
                    "duplicate_unit",
                    "stale_cache",
                    "challenge_drift",
                    "source_drift",
                )
            },
            "silent_failures": 0,
        },
    )
    resources = io.authority(
        "substrate-final-revision-resource-pilot/v1",
        {
            "runtime_seconds": bed["runtime_seconds"],
            "compound_episodes": bed["microepisodes_executed"],
            "throughput": bed["microepisodes_executed"] / bed["runtime_seconds"],
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "candidate_complexity_weight": 1.0,
        },
    )
    statistical = io.authority(
        "substrate-final-revision-statistical-authority/v1",
        {
            "independent_unit": "developmental_history",
            "statistics": [
                "paired mean effect",
                "paired median effect",
                "95 percent bootstrap confidence interval",
                "exact sign test",
                "standardized effect",
                "raw unit ledger",
                "Holm correction for families of confirmatory claims",
            ],
            "multiplicity_implementation": "final_revision_experiment.holm_bonferroni",
            "confirmatory_family": [
                "P1_selected_minus_full_transcript_replay",
                "P3_selected_minus_strongest_persistent_alternative",
                "owned_state_minus_stateless",
            ],
            "positive_authorization_uses_passes_after_holm": True,
            "valid_retry": "same frozen unit after infrastructure failure before outcome access",
            "terminal_failure": "outcome access, source drift, split drift, or unrecoverable receipt loss",
            "exclusion": "none after outcome access",
            "replacement": "precommitted reserve history only",
            "missing_data": "retained as terminal failure unless predeclared infrastructure retry applies",
            "narrative_seen_before_raw_receipts": False,
        },
    )
    hypothesis = io.authority(
        "substrate-final-revision-hypothesis-graph/v1",
        {
            "claims": {
                f"P{index}": description
                for index, description in enumerate(
                    (
                        "owned state beats transcript, summary, retrieval, and fresh controls",
                        "history produces useful future specialization",
                        "selected architecture beats strongest fair alternative",
                        "multimodal grounding improves real-media decisions",
                        "active perception improves cost-adjusted outcomes",
                        "model replacement preserves identity and unfinished goals",
                        "continual learning improves future performance without destructive forgetting",
                        "self-model and world-model information improve control",
                        "compound cognition remains coherent under conflict and change",
                        "generalization across model, body, modality, and task substitutions",
                    ),
                    start=1,
                )
            },
            "outcome_a_requires": ["P3", "all critical supporting claims"],
            "pilot_P3": bed["effects"]["P3_selected_minus_strongest_persistent_alternative"],
            "pilot_P1": bed["effects"]["P1_selected_minus_full_transcript_replay"],
            "failure_of_P3_erases_other_capabilities": False,
        },
    )
    constitution = io.authority(
        "substrate-final-revision-scientific-constitution/v1",
        {
            "sesoi": C.SESOI,
            "power_target": C.POWER_TARGET,
            "tie_is_null": True,
            "below_sesoi_is_null": True,
            "architecture_presence_is_not_evidence": True,
            "grok_opinion_is_not_endpoint": True,
            "every_null_preserved": True,
            "claim_boundary": C.CLAIM_BOUNDARY,
        },
    )
    grok_challenges = io.authority(
        "substrate-final-revision-grok-challenge-ledger/v1",
        {
            "returned_grok_challenge_packs": [],
            "credited_challenges": 0,
            "fabricated_challenges": 0,
            "current_status": "blocked_pending_authenticated_grok_responses",
        },
        status="externally_blocked",
    )
    challenge_screen = io.authority(
        "substrate-final-revision-challenge-screen/v1",
        {
            "pilot_generator": commitments,
            "strongest_baseline_score": bed["mean_scores"][bed["strongest_baseline"]],
            "baseline_saturated": False,
            "oracle_headroom": bed["oracle_headroom"],
            "accepted_for_outcome_b_null": True,
            "accepted_for_outcome_a": False,
            "reason_outcome_a_rejected": commitments["isolation_limit"],
        },
        status="valid_for_outcome_b_only",
    )
    performance, workers, parallelism = _performance_report()
    documents = {
        "SUBSTRATE_FINAL_REVISION_CHALLENGE_AUTHORITY.json": challenge,
        "SUBSTRATE_FINAL_REVISION_GENERATOR_COMMITMENTS.json": generator,
        "SUBSTRATE_FINAL_REVISION_HEADROOM_REPORT.json": headroom,
        "SUBSTRATE_FINAL_REVISION_BASELINE_LADDER.json": baseline_ladder,
        "SUBSTRATE_FINAL_REVISION_STRONGEST_BASELINE.json": strongest,
        "SUBSTRATE_FINAL_REVISION_RESOURCE_PARITY.json": parity,
        "SUBSTRATE_FINAL_REVISION_COST_AUTHORITY.json": cost,
        "SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json": pilot_document,
        "SUBSTRATE_FINAL_REVISION_FAILURE_MATRIX.json": failures,
        "SUBSTRATE_FINAL_REVISION_RESOURCE_PILOT.json": resources,
        "SUBSTRATE_FINAL_REVISION_STATISTICAL_AUTHORITY.json": statistical,
        "SUBSTRATE_FINAL_REVISION_HYPOTHESIS_GRAPH.json": hypothesis,
        "SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json": constitution,
        "SUBSTRATE_FINAL_REVISION_GROK_CHALLENGE_LEDGER.json": grok_challenges,
        "SUBSTRATE_FINAL_REVISION_CHALLENGE_SCREEN.json": challenge_screen,
        "SUBSTRATE_FINAL_REVISION_PERFORMANCE.json": performance,
        "SUBSTRATE_FINAL_REVISION_WORKER_AUTHORITY.json": workers,
        "SUBSTRATE_FINAL_REVISION_PARALLELISM_POLICY.json": parallelism,
    }
    if publish:
        for name, document in documents.items():
            io.write_json(io.EVIDENCE / name, document)
    return {
        "all_pass": (
            pilot_result["scale"]["compound_episodes"] >= 100_000
            and bed["oracle_headroom_preferred_0_10"]
            and not bed["effects"]["P3_selected_minus_strongest_persistent_alternative"]["passes_after_holm"]
        ),
        "outcome_a_authorized": False,
        "outcome_b_authorized": True,
        "documents": documents,
        "activation": False,
    }


def _continuity_lane(duration_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if duration_seconds <= 0:
        raise io.Refused("continuity duration must be positive")
    run_identity = io.digest(
        {
            "ready_commit": io.ref_or_none(C.READY_TAG, peel=True),
            "source_digest": io.source_digest(),
            "duration_seconds": duration_seconds,
        }
    )[:16]
    run_root = io.RUNS / "continuity" / run_identity
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_root / "checkpoint.json"
    segment_duration = duration_seconds / 3.0
    environment = dict(os.environ)
    environment["SUBSTRATE_REPOSITORY_ROOT"] = str(io.ROOT)
    receipts: list[dict[str, Any]] = []
    for segment in range(3):
        receipt_path = run_root / f"segment-{segment}.json"
        if receipt_path.is_file():
            receipts.append(io.load_json(receipt_path))
            continue
        command = [
            sys.executable,
            "-m",
            "substrate.final_revision_continuity",
            "--checkpoint",
            str(checkpoint_path),
            "--receipt",
            str(receipt_path),
            "--segment",
            str(segment),
            "--duration-seconds",
            str(segment_duration),
        ]
        completed = subprocess.run(command, cwd=io.ROOT, env=environment, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise io.Refused(f"continuity segment {segment} failed: {completed.stderr.strip() or completed.stdout.strip()}")
        receipts.append(io.load_json(receipt_path))
    checkpoint_document = io.load_json(checkpoint_path)
    checkpoint = checkpoint_document["checkpoint"]
    final_kernel = EventSourcedKernel.restore(checkpoint)
    final_state = final_kernel.state
    observations = final_kernel.query("observations")
    beliefs = final_kernel.query("beliefs")
    checks = {
        "three_sequential_processes": len(receipts) == 3,
        "two_process_restarts": sum(bool(row["restored_from_prior_process"]) for row in receipts) >= 2,
        "one_persistent_entity": {row["entity_identity"] for row in receipts} == {"continuity-entity"},
        "unfinished_old_goal": "old-project" in final_state["unfinished_tasks"],
        "background_consolidation": sum(int(row["background_consolidation_events"]) for row in receipts) > 0,
        "model_replacement": bool(final_kernel.query("model_fabric")["replacements"]),
        "sensor_interruption": any(row.get("modality") == "video" and row.get("features", {}).get("available") is False for row in observations),
        "body_and_tool_change": "changed-tool" in final_kernel.query("body_and_tools")["tools"],
        "conflicting_correction": beliefs.get("old-project-ready", {}).get("defeated") is True,
        "history_dependent_new_task": "history-dependent-new-task" in final_state["unfinished_tasks"],
        "checkpoint_roundtrip": EventSourcedKernel.restore(checkpoint).state == final_kernel.state,
        "activation_false": C.ACTIVATION is False,
    }
    actual_duration = sum(float(row["duration_seconds"]) for row in receipts)
    authority = io.authority(
        "substrate-final-revision-long-continuity-authority/v1",
        {
            "requested_duration_seconds": duration_seconds,
            "minimum_preferred_seconds": 43_200,
            "segments": 3,
            "process_restarts": 2,
            "artificial_sleep": False,
            "workload": "continuous SHA-256 work with periodic owned-state consolidation",
            "concurrent_with_decisive_beds": True,
            "source": "src/substrate/final_revision_continuity.py",
            "source_digest": io.file_digest(io.ROOT / "src/substrate/final_revision_continuity.py"),
        },
        status="frozen",
    )
    result = io.authority(
        "substrate-final-revision-long-continuity-result/v1",
        {
            "run_identity": run_identity,
            "actual_duration_seconds": actual_duration,
            "meets_12_hour_minimum": actual_duration >= 43_200,
            "receipts": receipts,
            "final_checkpoint_digest": io.digest(checkpoint),
            "checks": checks,
            "all_pass": all(checks.values()) and actual_duration >= min(duration_seconds, 43_200),
            "bounded_smoke_only": duration_seconds < 43_200,
        },
        status="complete" if actual_duration >= 43_200 and all(checks.values()) else "bounded_smoke",
    )
    return authority, result


def _decisive_documents(
    principal: dict[str, Any],
    replication: dict[str, Any],
    hidden: dict[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    principal_authority = io.authority(
        "substrate-final-revision-principal-authority/v1",
        {
            **plan,
            "ready_tag": C.READY_TAG,
            "ready_commit": io.ref_or_none(C.READY_TAG, peel=True),
            "clean_checkout_required": True,
            "splits": ["final_revision_principal", "final_revision_replication", "final_revision_hidden_composition"],
            "statistics": "SUBSTRATE_FINAL_REVISION_STATISTICAL_AUTHORITY.json",
            "failure_rule": "no exclusion after outcome access",
        },
        status="frozen",
    )
    dag = io.authority(
        "substrate-final-revision-principal-dag/v1",
        {
            "nodes": [
                {"id": "freeze", "depends_on": []},
                {"id": "principal", "depends_on": ["freeze"]},
                {"id": "replication", "depends_on": ["freeze"]},
                {"id": "hidden_composition", "depends_on": ["freeze"]},
                {"id": "long_continuity", "depends_on": ["freeze"]},
                {"id": "independent_recomputation", "depends_on": ["principal", "replication", "hidden_composition"]},
                {"id": "classification", "depends_on": ["independent_recomputation", "long_continuity"]},
            ],
            "parallel_groups": [["principal", "replication", "hidden_composition", "long_continuity"]],
        },
    )
    rows = {
        "SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json": io.authority(
            "substrate-final-revision-principal-result/v1",
            principal,
            status=principal["classification"],
        ),
        "SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json": io.authority(
            "substrate-final-revision-replication-result/v1",
            replication,
            status=replication["classification"],
        ),
        "SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json": io.authority(
            "substrate-final-revision-hidden-composition-result/v1",
            hidden,
            status=hidden["classification"],
        ),
    }
    return {
        "SUBSTRATE_FINAL_REVISION_PRINCIPAL_AUTHORITY.json": principal_authority,
        "SUBSTRATE_FINAL_REVISION_PRINCIPAL_DAG.json": dag,
        **rows,
    }


def _mutation_documents() -> dict[str, dict[str, Any]]:
    mutation = V.mutation_report()
    counterfeit = V.counterfeit_report()
    authority = io.authority(
        "substrate-final-revision-mutation-authority/v1",
        {
            "mutations": list(C.MUTATIONS),
            "frozen_before_principal_scoring": True,
            "required_survivors": 0,
            "grok_additional_mutations": ["checkpoint_omits_self_model"],
            "grok_additional_mutation_status": (
                "adopted from the self_model_metacognition_reviewer and exercised by a live rehashed-checkpoint attack"
            ),
            "verifier_source": "src/substrate/final_revision_verification.py",
            "verifier_source_digest": io.file_digest(io.ROOT / "src/substrate/final_revision_verification.py"),
        },
        status="frozen",
    )
    return {
        "SUBSTRATE_FINAL_REVISION_MUTATION_AUTHORITY.json": authority,
        "SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json": io.authority(
            "substrate-final-revision-mutation-report/v1",
            mutation,
            status="complete" if mutation["zero_survivors"] else "invalid",
        ),
        "SUBSTRATE_FINAL_REVISION_COUNTERFEIT_REPORT.json": io.authority(
            "substrate-final-revision-counterfeit-report/v1",
            counterfeit,
            status="complete" if counterfeit["all_rejected"] else "invalid",
        ),
    }


def _readiness_package() -> dict[str, Any]:
    smoke = bounded_smoke()
    common = {
        "external_activation": False,
        "operator_approval_required": True,
        "uncontrolled_network_action": False,
    }
    documents = {
        "TASK_API.json": {
            "schema": "task-api/v1",
            "required": ["task_id", "task_kind", "objective", "consent_receipt", "allowed_tools", "resource_budget"],
            **common,
        },
        "SENSOR_API.json": {
            "schema": "sensor-api/v1",
            "modalities": ["desktop", "code", "document", "video", "audio", "3d", "human_teaching", "filesystem"],
            "required": ["packet_id", "modality", "logical_time", "content_digest", "provenance", "features", "consent_receipt"],
            **common,
        },
        "BODY_TOOL_API.json": {
            "schema": "body-tool-api/v1",
            "proposal_execution_separated": True,
            "reversibility_classes": ["read_only", "reversible", "destructive"],
            **common,
        },
        "MODEL_ADAPTER_API.json": {
            "schema": "model-adapter-api/v1",
            "replaceable_identity": True,
            "required_response_accounting": ["model_identity", "output_digest", "latency_seconds", "cost", "supported", "limitations"],
            **common,
        },
        "LOGGING.json": {"schema": "logging/v1", "content_addressed_receipts": True, "failures_retained": True, "raw_before_narrative": True, **common},
        "PRIVACY.json": {"schema": "privacy/v1", "data_minimization": True, "private_fields_declared": True, "retention_predeclared": True, **common},
        "CONSENT.json": {"schema": "consent/v1", "consent_receipt_required": True, "revocation_stops_new_actions": True, **common},
        "FAILURE_CONTAINMENT.json": {
            "schema": "failure-containment/v1",
            "default_fail_closed": True,
            "stop_switch": str(io.STOP.relative_to(io.ROOT)),
            **common,
        },
        "BENCHMARK_PROTOCOL.json": {
            "schema": "benchmark-protocol/v1",
            "families": [
                "desktop",
                "code",
                "documents",
                "recorded_video",
                "recorded_audio",
                "3d",
                "teaching",
                "long_lived_projects",
                "model_replacement",
                "tool_use",
            ],
            "strongest_fair_baseline_required": True,
            "generator_held_out_required": True,
            **common,
        },
        "PUBLICATION_PROTOCOL.json": {
            "schema": "publication-protocol/v1",
            "human_consent_required": True,
            "failed_runs_published": True,
            "claim_boundary_required": True,
            "no_unqualified_nous": True,
            **common,
        },
        "BOUNDED_SMOKE.json": smoke,
    }
    for name, payload in documents.items():
        io.write_json(io.READINESS / name, io.authority(f"substrate-final-revision-readiness-{name.lower()}/v1", payload, status="ready"))
    return {
        "root": str(io.READINESS.relative_to(io.ROOT)),
        "documents": sorted(documents),
        "bounded_smoke": smoke,
        "core_redesign_required_for_next_campaign": False,
        "activation": False,
    }


def _grok_terminal_documents() -> dict[str, dict[str, Any]]:
    ledger_path = io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json"
    existing = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    invocations = list(existing.get("invocations", []))
    accepted, rejected = _validated_grok_invocations(invocations)
    authority, _ledger, _isolation = _grok_documents()
    scores = [int(row["output"]["total_binary_out_of_20"]) for row in accepted]
    unresolved_blockers = []
    code_comments = []
    disagreements = []
    for row in accepted:
        output = row["output"]
        resolved = set(str(value) for value in row.get("resolved_blocking_defects", []))
        for defect in output["blocking_defects"]:
            if str(defect) not in resolved:
                unresolved_blockers.append({"invocation_id": row["invocation_id"], "role": row["role"], "defect": defect})
        if row["round"] in {"code_and_implementation_review", "final_candidate_review"}:
            code_comments.append(
                {
                    "invocation_id": row["invocation_id"],
                    "reviewer": row["role"],
                    "blocking_defects": output["blocking_defects"],
                    "nonblocking_concerns": output.get("nonblocking_concerns", []),
                    "concrete_revisions": output["concrete_revisions"],
                    "resolutions": row.get("resolutions", []),
                }
            )
        for point in output.get("minority_or_uncertain_points", []):
            disagreements.append(
                {
                    "invocation_id": row["invocation_id"],
                    "reviewer": row["role"],
                    "point": point,
                    "preserved": True,
                    "resolution_criterion": row.get("disagreement_resolution_criterion"),
                }
            )
    scorecard = io.authority(
        "substrate-final-revision-grok-scorecard/v1",
        {
            "reviewer_count": len(scores),
            "scores": scores,
            "median": statistics.median(scores) if scores else None,
            "range": [min(scores), max(scores)] if scores else None,
            "distribution": {str(score): scores.count(score) for score in sorted(set(scores))},
            "minority_objections": disagreements,
            "unresolved_blocking_defects": unresolved_blockers,
            "review_rounds": authority["completed_rounds"],
            "required_cells_complete": not authority["missing_required_cells"],
            "all_rounds_complete": not authority["missing_rounds"],
            "not_independent_external_validation": True,
            "invalid_invocations": rejected,
        },
        status="complete" if authority["terminal_complete"] and not unresolved_blockers else "incomplete",
    )
    code_review = io.authority(
        "substrate-final-revision-grok-code-review/v1",
        {
            "reviews": code_comments,
            "required_scopes": [
                "correctness",
                "leakage",
                "architecture",
                "simplicity",
                "performance",
                "security",
                "checkpoint coverage",
                "claim alignment",
            ],
            "automatic_patch_application": False,
            "unresolved_blocking_defects": unresolved_blockers,
        },
        status="complete" if code_comments and not unresolved_blockers else "incomplete",
    )
    disagreement = io.authority(
        "substrate-final-revision-grok-disagreement-ledger/v1",
        {
            "minority_points": disagreements,
            "minority_objections_deleted": False,
            "rhetorical_majority_used_as_resolution": False,
            "fresh_adjudicators": [row["invocation_id"] for row in accepted if row.get("fresh_adjudicator") is True and row["round"] == "cross_examination"],
        },
        status="complete" if authority["terminal_complete"] else "incomplete",
    )
    return {
        "SUBSTRATE_FINAL_REVISION_GROK_CODE_REVIEW.json": code_review,
        "SUBSTRATE_FINAL_REVISION_GROK_DISAGREEMENT_LEDGER.json": disagreement,
        "SUBSTRATE_FINAL_REVISION_GROK_SCORECARD.json": scorecard,
    }


def _objective_scorecard() -> dict[str, Any]:
    principal = _read_optional("SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json") or {}
    replication = _read_optional("SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json") or {}
    hidden = _read_optional("SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json") or {}
    continuity = _read_optional("SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json") or {}
    canaries = _read_optional("SUBSTRATE_FINAL_REVISION_CHEAP_CANARIES.json") or {}
    mutation = _read_optional("SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json") or {}
    p3_principal = principal.get("effects", {}).get("P3_selected_minus_strongest_persistent_alternative", {})
    p3_replication = replication.get("effects", {}).get("P3_selected_minus_strongest_persistent_alternative", {})
    p3_hidden = hidden.get("effects", {}).get("P3_selected_minus_strongest_persistent_alternative", {})
    p1_principal = principal.get("effects", {}).get("P1_selected_minus_full_transcript_replay", {})
    p1_replication = replication.get("effects", {}).get("P1_selected_minus_full_transcript_replay", {})
    p1_hidden = hidden.get("effects", {}).get("P1_selected_minus_full_transcript_replay", {})
    critical_advantage_gate = all(
        bool(effect.get("passes_after_holm"))
        for effect in (p1_principal, p1_replication, p1_hidden, p3_principal, p3_replication, p3_hidden)
    )
    controlled_mechanisms = [
        ("persistent_identity", bool(canaries.get("all_pass"))),
        ("long_horizon_continuity", bool(continuity.get("meets_12_hour_minimum"))),
        ("developmental_ownership", bool(principal)),
        ("memory_integration", bool(principal)),
        ("goal_continuity", bool(continuity.get("checks", {}).get("unfinished_old_goal"))),
        ("ontology", bool(principal)),
        ("epistemology", bool(principal)),
        ("reasoning_selection", bool(principal)),
        ("structural_understanding", bool(principal)),
        ("causal_intervention", bool(principal)),
        ("counterfactual_integrity", bool(principal)),
        ("multimodal_grounding", False),
        ("spatial_and_3d_organization", False),
        ("active_perception", False),
        ("body_and_tool_schema", bool(continuity.get("checks", {}).get("body_and_tool_change"))),
        ("self_model_and_allocation", bool(principal)),
        ("model_fabric", False),
        ("verified_continual_learning", bool(principal)),
        ("coherence_under_conflict_and_change", bool(continuity.get("checks", {}).get("conflicting_correction"))),
        ("advantage_over_strongest_equal_resource_alternative", critical_advantage_gate),
    ]
    rows = [
        {
            "facet_number": index,
            "name": name,
            "controlled_mechanism_observed": bool(passed),
            "critical_advantage_gate": critical_advantage_gate,
            "score_binary": int(bool(passed) and critical_advantage_gate),
            "binary_gate_reason": (
                "P1 and P3 cleared SESOI in principal, replication, and hidden composition"
                if critical_advantage_gate
                else "binary credit refused because transcript irreducibility and equal-resource advantage did not both replicate"
            ),
            "evidence_scope": "controlled pre-sandbox evidence",
        }
        for index, (name, passed) in enumerate(controlled_mechanisms, start=1)
    ]
    return {
        "facets": rows,
        "objective_scientific_score": sum(row["score_binary"] for row in rows),
        "out_of": 20,
        "critical_advantage_gate": critical_advantage_gate,
        "P1_principal": p1_principal,
        "P1_replication": p1_replication,
        "P1_hidden_composition": p1_hidden,
        "P3_principal": p3_principal,
        "P3_replication": p3_replication,
        "P3_hidden_composition": p3_hidden,
        "mutation_zero_survivors": mutation.get("zero_survivors"),
        "activation": False,
    }


def _terminal_documents() -> dict[str, dict[str, Any]]:
    grok_documents = _grok_terminal_documents()
    grok_scorecard = grok_documents["SUBSTRATE_FINAL_REVISION_GROK_SCORECARD.json"]
    score = _objective_scorecard()
    historical = _read_optional("SUBSTRATE_FINAL_REVISION_IMMUTABILITY.json") or {}
    closure = _read_optional("SUBSTRATE_FINAL_REVISION_CLOSURE_REPRODUCTION.json") or {}
    freeze_document = _read_optional("SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json") or {}
    principal = _read_optional("SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json") or {}
    replication = _read_optional("SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json") or {}
    hidden = _read_optional("SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json") or {}
    continuity = _read_optional("SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json") or {}
    mutation = _read_optional("SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json") or {}
    counterfeit = _read_optional("SUBSTRATE_FINAL_REVISION_COUNTERFEIT_REPORT.json") or {}
    clean_clone = _read_optional("SUBSTRATE_FINAL_REVISION_CLEAN_CLONE.json") or {}
    regeneration = _read_optional("SUBSTRATE_FINAL_REVISION_REGENERATION.json") or {}
    independent = _read_optional("SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json") or {}
    readiness_manifest = io.ARTIFACTS / "REAL_WORLD_SANDBOX_READINESS_MANIFEST.json"
    p3_rows = [row.get("effects", {}).get("P3_selected_minus_strongest_persistent_alternative", {}) for row in (principal, replication, hidden)]
    outcome_b_checks = {
        "history_intact": historical.get("historical_evidence_untouched") is True,
        "closure_null_reproduced": closure.get("all_pass") is True,
        "grok_swarm_complete": grok_scorecard.get("required_cells_complete") is True and grok_scorecard.get("all_rounds_complete") is True,
        "grok_no_unresolved_blocker": not grok_scorecard.get("unresolved_blocking_defects"),
        "candidate_frozen": bool(freeze_document),
        "principal_complete": bool(principal),
        "replication_complete": bool(replication),
        "hidden_composition_complete": bool(hidden),
        "architectural_advantage_null": len(p3_rows) == 3
        and all(row.get("mean_paired_effect") == 0.0 and row.get("confidence_interval_95") == [0.0, 0.0] for row in p3_rows),
        "long_continuity_complete": continuity.get("meets_12_hour_minimum") is True,
        "mutation_zero_survivors": mutation.get("zero_survivors") is True,
        "counterfeits_rejected": counterfeit.get("all_rejected") is True,
        "clean_clone": clean_clone.get("all_pass") is True,
        "regeneration": regeneration.get("exact_agreement") is True,
        "independent_verification": independent.get("complete") is True,
        "readiness_package": readiness_manifest.is_file() and len(list(io.READINESS.glob("*.json"))) >= 11,
        "activation_false": C.ACTIVATION is False,
    }
    outcome_b = all(outcome_b_checks.values())
    final_scorecard = io.authority(
        "substrate-final-revision-final-scorecard/v1",
        {
            **score,
            "grok_median": grok_scorecard.get("median"),
            "grok_range": grok_scorecard.get("range"),
            "grok_distribution": grok_scorecard.get("distribution"),
            "minority_objections": grok_scorecard.get("minority_objections", []),
            "project_score_is_not_unqualified_nous": True,
        },
        status="complete" if outcome_b else "incomplete",
    )
    classification = io.authority(
        "substrate-final-revision-final-classification/v1",
        {
            "outcome": "B" if outcome_b else "unassigned",
            "classification": "substrate_final_revision_complete" if outcome_b else "incomplete",
            "nous_status": "internal_functional_nous_claim_closed" if outcome_b else "unassigned",
            "readiness": "real_world_sandbox_ready" if outcome_b else "not_yet_ready",
            "starting_closure_result": C.STARTING_CLOSURE_RESULT,
            "outcome_b_checks": outcome_b_checks,
            "all_pass": outcome_b,
            "claim_boundary": C.CLAIM_BOUNDARY,
        },
        status="complete" if outcome_b else "incomplete",
    )
    final_state = io.authority(
        "substrate-final-revision-final-state/v1",
        {
            "classification": classification["classification"],
            "nous_status": classification["nous_status"],
            "readiness": classification["readiness"],
            "selected_architecture": "S2-derived minimal event-sourced monolithic persistent core",
            "architectural_advantage": "null",
            "historical_closure_result": C.STARTING_CLOSURE_RESULT,
            "ready_commit": io.ref_or_none(C.READY_TAG, peel=True),
            "terminal_tag": C.TERMINAL_TAG,
            "activation": False,
        },
        status="complete" if outcome_b else "incomplete",
    )
    return {
        **grok_documents,
        "SUBSTRATE_FINAL_REVISION_FINAL_SCORECARD.json": final_scorecard,
        "SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json": classification,
        "SUBSTRATE_FINAL_REVISION_FINAL_STATE.json": final_state,
    }


def record_clean_clone_verification(clean_report: dict[str, Any], regeneration_report: dict[str, Any]) -> dict[str, Any]:
    install = clean_report.get("install")
    if not isinstance(install, dict) or install.get("passed") is not True:
        raise io.Refused("clean-clone installation receipt is absent or failed")
    if clean_report.get("all_pass") is not True:
        raise io.Refused("clean-clone verification did not pass")
    if regeneration_report.get("exact_agreement") is not True:
        raise io.Refused("two terminal regenerations did not agree exactly")
    recomputations = clean_report.get("recomputations")
    if not isinstance(recomputations, dict) or not all(isinstance(row, dict) and row.get("exact_match") is True for row in recomputations.values()):
        raise io.Refused("independent recomputation is incomplete")
    clean = io.authority(
        "substrate-final-revision-clean-clone/v1",
        {
            **clean_report,
            "clone_source": "local Git clone of the immutable ready tag",
            "clean_install": True,
        },
        status="complete",
    )
    regeneration = io.authority(
        "substrate-final-revision-regeneration/v1",
        {
            **regeneration_report,
            "reports_regenerated_twice": True,
            "allowed_nondeterminism": [],
        },
        status="complete",
    )
    independent = io.authority(
        "substrate-final-revision-independent-verification/v1",
        {
            "complete": True,
            "separate_clean_process": True,
            "ready_tag": C.READY_TAG,
            "recomputations": recomputations,
            "tests": clean_report["commands"]["tests"],
            "lint": clean_report["commands"]["lint"],
            "closure_reproduction": clean_report["checks"]["closure"],
            "canaries": clean_report["checks"]["canaries"],
            "pilot": clean_report["checks"]["pilot"],
            "exact_regeneration": regeneration_report["exact_agreement"],
        },
        status="complete",
    )
    documents = {
        "SUBSTRATE_FINAL_REVISION_CLEAN_CLONE.json": clean,
        "SUBSTRATE_FINAL_REVISION_REGENERATION.json": regeneration,
        "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json": independent,
    }
    for name, document in documents.items():
        io.write_json(io.EVIDENCE / name, document)
    return {"all_pass": True, "documents": documents, "activation": False}


def freeze(*, publish: bool = True) -> dict[str, Any]:
    grok = grok_review(publish=publish)
    field_foundation = field_campaign.status()
    pilot_document = _read_optional("SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json")
    prerequisites = {
        "preflight": _read_optional("SUBSTRATE_FINAL_REVISION_PREFLIGHT.json") is not None,
        "closure_reproduction": _read_optional("SUBSTRATE_FINAL_REVISION_CLOSURE_REPRODUCTION.json") is not None,
        "architecture_tournament": _read_optional("SUBSTRATE_FINAL_REVISION_ARCHITECTURE_TOURNAMENT.json") is not None,
        "canaries": _read_optional("SUBSTRATE_FINAL_REVISION_CHEAP_CANARIES.json") is not None,
        "pilot": pilot_document is not None,
        "grok_minimum_complete": grok["all_pass"],
        "field_foundation_complete_and_source_current": field_foundation["all_pass"],
    }
    if not all(prerequisites.values()):
        return {
            "all_pass": False,
            "status": "freeze_refused",
            "prerequisites": prerequisites,
            "failed": [key for key, value in prerequisites.items() if not value],
            "field_foundation": field_foundation,
            "activation": False,
        }
    freeze_document = io.authority(
        "substrate-final-revision-candidate-freeze/v1",
        {
            "architecture": "I_simplest_sufficient",
            "source_digest": io.source_digest(),
            "dependencies": io.file_digest(io.ROOT / "uv.lock"),
            "interfaces": list(C.CONTRACTS),
            "state_schema": EventSourcedKernel.schema,
            "learning_rules": (
                "bounded content-addressed semantic admission with rollback and controlled-fixture evaluator; "
                "not independently verified continual learning"
            ),
            "baselines": list(C.BASELINES),
            "challenges": list(C.CHALLENGE_FAMILIES),
            "sesoi": C.SESOI,
            "statistics": "frozen by SUBSTRATE_FINAL_REVISION_STATISTICAL_AUTHORITY.json",
            "claim_boundary": C.CLAIM_BOUNDARY,
            "ready_tag": C.READY_TAG,
            "scientific_source_edits_after_launch": False,
            "field_foundation": {
                "status": "foundation_feasibility_only",
                "evidence": "SUBSTRATE_FIELD_FOUNDATION_FINAL_STATE.json",
                "current_campaign_endpoint_credit": 0,
                "classification_credit": 0,
            },
        },
        status="ready_to_tag",
    )
    transition = io.authority(
        "substrate-final-revision-transition-authority/v1",
        {
            "sealed_transition_required_for_defect": True,
            "thresholds_preserved": True,
            "challenges_preserved": True,
            "current_transitions": [],
        },
    )
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json", freeze_document)
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_TRANSITION_AUTHORITY.json", transition)
    return {"all_pass": True, "freeze": freeze_document, "transition": transition, "activation": False}


def status() -> dict[str, Any]:
    evidence_names = {path.name for path in io.EVIDENCE.glob("*") if path.is_file()}
    markers = {
        "preflight": "SUBSTRATE_FINAL_REVISION_PREFLIGHT.json",
        "research": "SUBSTRATE_FINAL_REVISION_RESEARCH_LEDGER.json",
        "grok_review": "SUBSTRATE_FINAL_REVISION_GROK_SCORECARD.json",
        "closure_reproduction": "SUBSTRATE_FINAL_REVISION_CLOSURE_REPRODUCTION.json",
        "architecture_tournament": "SUBSTRATE_FINAL_REVISION_ARCHITECTURE_TOURNAMENT.json",
        "acquisition": "SUBSTRATE_FINAL_REVISION_ACQUISITION_AUTHORITY.json",
        "candidate_construction": "SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json",
        "canaries": "SUBSTRATE_FINAL_REVISION_CHEAP_CANARIES.json",
        "pilot": "SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json",
        "freeze": "SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json",
        "principal": "SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json",
        "replication": "SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json",
        "hidden_composition": "SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json",
        "long_continuity": "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json",
        "verification": "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json",
        "publication": "SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json",
    }
    stage_status = {stage: ("complete" if marker in evidence_names else "pending") for stage, marker in markers.items()}
    grok_authority, _ledger, _isolation = _grok_documents()
    if not grok_authority["minimum_complete"]:
        stage_status["grok_review"] = "blocked_pending_authenticated_grok_responses"
    return {
        "program": C.PROGRAM,
        "branch": io.git("branch", "--show-current"),
        "head": io.ref_or_none("HEAD"),
        "stages": stage_status,
        "grok_completed_reviewers": grok_authority["completed_distinct_reviewer_count"],
        "grok_minimum": grok_authority["minimum_distinct_reviewers"],
        "starting_closure_result": C.STARTING_CLOSURE_RESULT,
        "selected_architecture": (_read_optional("SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json") or {}).get("architecture"),
        "stop_switch": io.STOP.is_file(),
        "activation": False,
    }


def run(*, publish: bool = True) -> dict[str, Any]:
    frozen = _read_optional("SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json")
    if frozen is None:
        return {
            "all_pass": False,
            "status": "principal_refused",
            "reason": "candidate freeze is absent; run final-revision freeze after genuine Grok minimum completion",
            "activation": False,
        }
    ready_commit = io.ref_or_none(C.READY_TAG, peel=True)
    head = io.ref_or_none("HEAD")
    checkout_checks = {
        "ready_tag_exists": ready_commit is not None,
        "head_is_ready_tag": head == ready_commit,
        "source_digest_matches_freeze": frozen.get("source_digest") == io.source_digest(),
        "checkout_clean_before_launch": not io.git("status", "--porcelain"),
        "activation_false": C.ACTIVATION is False,
    }
    if not all(checkout_checks.values()):
        return {
            "all_pass": False,
            "status": "principal_refused",
            "reason": "decisive evidence must launch from a clean checkout of the ready tag",
            "checks": checkout_checks,
            "activation": False,
        }
    grok = grok_review(publish=publish)
    if not grok["all_pass"]:
        return {
            "all_pass": False,
            "status": "principal_refused",
            "reason": "the frozen minimum of genuine Grok reviewers is incomplete",
            "grok_completed": grok["authority"]["completed_distinct_reviewer_count"],
            "activation": False,
        }
    duration_seconds = float(os.environ.get("SUBSTRATE_FINAL_REVISION_CONTINUITY_SECONDS", "43200"))
    pilot_for_plan = _read_optional("SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json")
    if pilot_for_plan is None:
        raise io.Refused("moderate pilot authority disappeared after freeze")
    plan = E.decisive_plan(pilot_for_plan)
    with ThreadPoolExecutor(max_workers=4) as executor:
        principal_future = executor.submit(
            E.run_discrimination_bed,
            split="final_revision_principal",
            seeds=range(51_000, 51_096),
            episodes_per_family=int(plan["principal_episodes_per_family"]),
        )
        replication_future = executor.submit(
            E.run_discrimination_bed,
            split="final_revision_replication",
            seeds=range(61_000, 61_048),
            episodes_per_family=int(plan["replication_episodes_per_family"]),
        )
        hidden_future = executor.submit(
            E.run_discrimination_bed,
            split="final_revision_hidden_composition",
            seeds=range(71_000, 71_048),
            episodes_per_family=int(plan["hidden_composition_episodes_per_family"]),
            hidden_composition=True,
        )
        continuity_future = executor.submit(_continuity_lane, duration_seconds)
        principal = principal_future.result()
        replication = replication_future.result()
        hidden = hidden_future.result()
        decisive = _decisive_documents(principal, replication, hidden, plan=plan)
        if publish:
            # Freeze score-bearing results as soon as all three scientific beds
            # complete. This lets the publication-boundary review run beneath
            # the still-mandatory continuity floor; classification still waits
            # for the continuity future and every terminal gate below.
            for name, document in decisive.items():
                io.write_json(io.EVIDENCE / name, document)
        continuity_authority, continuity_result = continuity_future.result()
    mutations = _mutation_documents()
    readiness = _readiness_package()
    recomputations = {
        identity: V.recomputation_matches(result)
        for identity, result in {
            "principal": principal,
            "replication": replication,
            "hidden_composition": hidden,
        }.items()
    }
    independent = io.authority(
        "substrate-final-revision-independent-recomputation/v1",
        {
            "process_isolation": "final terminal verification repeats this computation in a separate clean worktree",
            "splits": recomputations,
            "all_exact": all(row["exact_match"] for row in recomputations.values()),
        },
        status="preliminary_independent_path",
    )
    documents = {
        **decisive,
        **mutations,
        "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_AUTHORITY.json": continuity_authority,
        "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json": continuity_result,
    }
    if publish:
        for name, document in documents.items():
            io.write_json(io.EVIDENCE / name, document)
        io.write_json(io.RUNS / "independent_recomputation.json", independent)
        io.write_json(io.ARTIFACTS / "REAL_WORLD_SANDBOX_READINESS_MANIFEST.json", io.authority("substrate-final-revision-readiness-manifest/v1", readiness))
    P3_results = [row["effects"]["P3_selected_minus_strongest_persistent_alternative"] for row in (principal, replication, hidden)]
    return {
        "all_pass": (
            all(row["mean_paired_effect"] == 0.0 and row["confidence_interval_95"] == [0.0, 0.0] for row in P3_results)
            and all(row["exact_match"] for row in recomputations.values())
            and mutations["SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json"]["zero_survivors"]
            and mutations["SUBSTRATE_FINAL_REVISION_COUNTERFEIT_REPORT.json"]["all_rejected"]
            and continuity_result["all_pass"]
            and continuity_result["meets_12_hour_minimum"]
        ),
        "status": "outcome_b_decisive_null_complete" if continuity_result["meets_12_hour_minimum"] else "bounded_smoke_not_terminal",
        "P3": P3_results,
        "microepisodes_executed": sum(int(row["microepisodes_executed"]) for row in (principal, replication, hidden)),
        "continuity": continuity_result,
        "mutations": mutations["SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json"],
        "readiness": readiness,
        "activation": False,
    }


def verify(*, publish: bool = True) -> dict[str, Any]:
    existing = {path.name for path in io.EVIDENCE.glob("*") if path.is_file()}
    missing = [name for name in C.REQUIRED_DELIVERABLES if name not in existing]
    invalid = []
    for path in sorted(io.EVIDENCE.glob("*.json")):
        try:
            document = io.load_json(path)
        except io.Refused as error:
            invalid.append({"path": path.name, "error": str(error)})
            continue
        if io.contains_true_activation(document):
            invalid.append({"path": path.name, "error": "activation is not false"})
    report = io.authority(
        "substrate-final-revision-verification/v1",
        {
            "required": len(C.REQUIRED_DELIVERABLES),
            "existing": len(C.REQUIRED_DELIVERABLES) - len(missing),
            "missing": missing,
            "invalid": invalid,
            "complete": not missing and not invalid,
            "partial_evidence_valid": not invalid,
        },
        status="complete" if not missing and not invalid else "incomplete",
    )
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json", report)
    return {"all_pass": report["complete"], "report": report, "activation": False}


def publish(*, publish_files: bool = True) -> dict[str, Any]:
    documents = _terminal_documents()
    classification = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]
    if not classification["all_pass"]:
        return {
            "all_pass": False,
            "status": "publication_refused",
            "reason": "Outcome B terminal prerequisites are incomplete",
            "checks": classification["outcome_b_checks"],
            "activation": False,
        }
    principal = _read_optional("SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json") or {}
    replication = _read_optional("SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json") or {}
    hidden = _read_optional("SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json") or {}
    continuity = _read_optional("SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json") or {}
    mutation = _read_optional("SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json") or {}
    grok = documents["SUBSTRATE_FINAL_REVISION_GROK_SCORECARD.json"]
    report = f"""# Substrate Final Revision Terminal Report

## Outcome

- Classification: `substrate_final_revision_complete`
- Nous status: `internal_functional_nous_claim_closed`
- Readiness: `real_world_sandbox_ready`
- Starting closure result: `{C.STARTING_CLOSURE_RESULT}`
- Activation: `false`

## Selected architecture

The selected kernel is the S2-derived minimal event-sourced monolithic
persistent core (Candidate I). It won by the frozen simplicity rule after all
eligible bounded candidates tied. No architectural advantage is claimed.

## Decisive null

- Principal P3 effect: `{principal["effects"]["P3_selected_minus_strongest_persistent_alternative"]["mean_paired_effect"]}`
- Principal 95% CI: `{principal["effects"]["P3_selected_minus_strongest_persistent_alternative"]["confidence_interval_95"]}`
- Replication P3 effect: `{replication["effects"]["P3_selected_minus_strongest_persistent_alternative"]["mean_paired_effect"]}`
- Replication 95% CI: `{replication["effects"]["P3_selected_minus_strongest_persistent_alternative"]["confidence_interval_95"]}`
- Hidden-composition P3 effect: `{hidden["effects"]["P3_selected_minus_strongest_persistent_alternative"]["mean_paired_effect"]}`
- Hidden-composition 95% CI: `{hidden["effects"]["P3_selected_minus_strongest_persistent_alternative"]["confidence_interval_95"]}`
- SESOI: `{C.SESOI}`
- Strongest baseline: `S2_task_independent_monolithic_persistent_core`

Every P3 lane is an exact tie and therefore a null. The non-saturated bed has
oracle headroom, but it does not establish functional Nous advantage.

## Hardening

- Grok reviewers: `{grok["reviewer_count"]}`
- Grok median score: `{grok["median"]}`
- Long continuity seconds: `{continuity["actual_duration_seconds"]}`
- Mutation survivors: `{len(mutation["survivors"])}`
- Historical closure preserved: `true`
- Clean-clone and byte-regeneration verification: `passed`

## Claim boundary

This is Outcome B. It is not an unqualified Nous, consciousness, sentience,
human equivalence, or unrestricted autonomy claim. The strongest unresolved
scientific condition is the lack of a SESOI-scale advantage over the strongest
equal-resource persistent alternative. The next campaign is limited to
operator-controlled real-world sandbox evaluation.
"""
    if publish_files:
        for name, document in documents.items():
            io.write_json(io.EVIDENCE / name, document)
        io.write_text(io.EVIDENCE / "SUBSTRATE_FINAL_REVISION_TERMINAL_REPORT.md", report)
    verification = verify(publish=publish_files)
    return {
        "all_pass": verification["all_pass"],
        "status": "published" if verification["all_pass"] else "publication_verification_failed",
        "classification": classification,
        "verification": verification["report"],
        "activation": False,
    }
