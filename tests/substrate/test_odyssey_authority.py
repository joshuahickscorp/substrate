"""Focused tests for the fail-closed Odyssey authority control plane."""

from __future__ import annotations

import json
import shutil
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from substrate import odyssey_authority as authority
from substrate import odyssey_manifest_materializer as materializer
from substrate import odyssey_task_bank as task_bank
from substrate import odyssey_transition as transition
from tests.substrate.librispeech_audio_fixture import install_librispeech_audio_fixture

Usage = namedtuple("Usage", "total used free")


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write(path: Path, value: dict) -> None:
    authority._write_json(path, value)


def _sealed(schema: str, payload: dict, *, status: str = "pass") -> dict:
    return authority._sealed(schema, payload, status=status)


def _fixture_root(tmp_path: Path) -> Path:
    repository = Path(__file__).parents[2]
    plan = repository / "docs/plans/substrate/tangible_next_launch"
    for filename in (
        "ODYSSEY_7D.hardened.draft.json",
        "R2_TO_ODYSSEY_AUTOPIVOT_POLICY.sealed.json",
        "ODYSSEY_TASK_BANK_MANIFEST.draft.json",
        "RESOURCE_CALIBRATION_SPEC.draft.json",
        "ODYSSEY_SHARED_STORAGE_RESERVE.draft.json",
        "ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json",
        "ODYSSEY_SOURCE_SELECTION.template.json",
        "ODYSSEY_PUBLIC_MODEL_CANARY.template.json",
        "ODYSSEY_HUMAN_EVIDENCE_PACK.template.json",
        "frontiers/FRONTIER_BUILD_INDEX.json",
    ):
        _copy(plan / filename, tmp_path / "docs/plans/substrate/tangible_next_launch" / filename)
    _copy(
        repository / "ops/operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json",
        tmp_path / "ops/operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json",
    )
    for filename in (
        "odyssey_transition.py",
        "odyssey7d.py",
        "odyssey_task_bank.py",
        "odyssey_manifest_materializer.py",
        "odyssey_machine_subjects.py",
        "odyssey_rehearsal.py",
        "odyssey_arms.py",
        "odyssey_authority.py",
        "odyssey_model_canary.py",
        "odyssey_clean_clone.py",
        "odyssey_mutations.py",
        "odyssey_detachment.py",
        "odyssey_telegram_probe.py",
        "r2_continuity_verifier.py",
        "r2_provenance_verifier.py",
    ):
        _copy(repository / "src/substrate" / filename, tmp_path / "src/substrate" / filename)
    _copy(
        repository / "ops/tools/odyssey7d_telegram_notifier.py",
        tmp_path / "ops/tools/odyssey7d_telegram_notifier.py",
    )
    # G06-DC binds the historical width-calibration diagnostic; the fixture root
    # must carry the same sealed prior so the launch subject can verify it.
    _copy(
        repository / "evidence/substrate/odyssey/ODYSSEY_ARM_PROTOCOL_V2_WIDTH_CALIBRATION.json",
        tmp_path / "evidence/substrate/odyssey/ODYSSEY_ARM_PROTOCOL_V2_WIDTH_CALIBRATION.json",
    )
    worker = tmp_path / "src/substrate/odyssey_worker.py"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text("# fixture worker\n", encoding="utf-8")
    implementation = {
        **transition.implementation_inputs(tmp_path),
        "odyssey_worker": worker,
        "odyssey_authority": tmp_path / "src/substrate/odyssey_authority.py",
    }
    body = {
        "schema": "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1",
        "program": "substrate-odyssey-r2-handoff-v1",
        "activation": False,
        "scientific_status": "frozen_waiting_for_verified_r2",
        "input_sha256": {name: authority.file_digest(path) for name, path in transition.build_inputs(tmp_path).items()},
        "implementation_sha256": {
            name: transition.canonical_source_digest(path) for name, path in implementation.items()
        },
        "r2_requirements": {},
        "transition": {},
    }
    body["sha256"] = authority.digest(body)
    _write(tmp_path / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", body)
    return tmp_path


def _machine_binding(frozen: dict[str, Any]) -> dict[str, Any]:
    return {
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": "fixture-head",
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
    }


def _receipt_ref(root: Path, name: str, payload: dict[str, Any] | None = None) -> dict[str, str]:
    path = root / "evidence" / f"{name}.json"
    _write(path, _sealed("SUBSTRATE_ODYSSEY_TEST_OBSERVATION/v1", {"name": name, **(payload or {})}))
    return {"path": str(path.relative_to(root)), "sha256": authority.file_digest(path)}


def _self_digested(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("sha256", None)
    body["sha256"] = authority.digest(body)
    return body


def _g06_adapter_receipt_ref(
    root: Path,
    *,
    name: str,
    role: str,
    frontier: str,
    task: dict[str, Any],
    manifest_sha256: str,
    authority_sha256: str,
    run_id: str,
    model: str,
    adapter_sha256: str,
) -> dict[str, str]:
    """Materialize a real-shaped arm receipt plus its content-addressed output."""
    request_sha256 = authority.digest({"fixture": name, "kind": "request"})
    usage = {
        "prompt_eval_count": 1,
        "eval_count": 1,
        "total_duration_ns": 1,
        "load_duration_ns": 0,
        "eval_duration_ns": 1,
    }
    output_path = root / "calibration" / name / "outputs" / f"{role}.json"
    output = _self_digested(
        {
            "schema": "SUBSTRATE_ODYSSEY_ARM_OUTPUT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "frontier": frontier,
            "role": role,
            "cycle": 0,
            "phase": "retrieval",
            "task_id": task["task_id"],
            "request_sha256": request_sha256,
            "candidate_manifest_sha256": manifest_sha256,
            "adapter_sha256": adapter_sha256,
            "model": model,
            "prompt_sha256": authority.digest({"fixture": name, "kind": "prompt"}),
            "response": {"fixture_response": role},
            "resource_usage": usage,
        }
    )
    _write(output_path, output)
    output_ref = {"path": str(output_path.relative_to(root)), "sha256": authority.file_digest(output_path)}
    receipt_path = root / "evidence" / f"{name}-{role}-adapter-receipt.json"
    receipt = _self_digested(
        {
            "schema": "SUBSTRATE_ODYSSEY_ADAPTER_RECEIPT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "frontier": frontier,
            "role": role,
            "cycle": 0,
            "phase": "retrieval",
            "task_id": task["task_id"],
            "candidate_manifest_sha256": manifest_sha256,
            "request_sha256": request_sha256,
            "elapsed_seconds": 1.0,
            "adapter_sha256": adapter_sha256,
            "model": {"id": model, "endpoint": "http://127.0.0.1:11434"},
            "output_artifacts": [output_ref],
            "response_sha256": output["sha256"],
            "state_before_sha256": authority.digest({"fixture": name, "role": role, "state": "before"}),
            "state_after_sha256": authority.digest({"fixture": name, "role": role, "state": "after"}),
            "state_change": {
                "mode": "flat_exact_associative_monolith" if role == "candidate" else "append_only_history_retrieval"
            },
            "resource_usage": usage,
        }
    )
    _write(receipt_path, receipt)
    return {"path": str(receipt_path.relative_to(root)), "sha256": authority.file_digest(receipt_path)}


def _g06_phase_boundary_ref(
    root: Path,
    *,
    name: str,
    authority_sha256: str,
    run_id: str,
    cells: list[dict[str, Any]],
) -> dict[str, str]:
    """Materialize the same one-phase trace/checkpoint/state chain G06 records."""
    boundary_root = root / "calibration" / name / "phase-boundary"
    trace_path = boundary_root / "EVENTS.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    chain = ""
    trace_rows: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda item: str(item["id"])):
        event = {
            "schema": "SUBSTRATE_ODYSSEY_PAIRED_EVENT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "frontier": cell["id"],
            "cycle": 0,
            "phase": "retrieval",
            "task_id": cell["task_binding"]["task_id"],
            "candidate_receipt_sha256": cell["candidate_receipt"]["sha256"],
            "control_receipt_sha256": cell["control_receipt"]["sha256"],
            "candidate_elapsed_seconds": 1.0,
            "control_elapsed_seconds": 1.0,
            "source_bundle_guard_calls": 2,
            "previous_event_sha256": chain,
        }
        event["event_sha256"] = authority.digest(event)
        chain = event["event_sha256"]
        trace_rows.append(event)
    trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows), encoding="utf-8")
    checkpoint_path = boundary_root / "checkpoints" / "delta-001.json"
    checkpoint = _self_digested(
        {
            "schema": "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "kind": "delta",
            "cycle": 0,
            "completed_phase_count": 1,
            "completed_paired_events": len(cells),
            "event_chain_sha256": chain,
            "parent_checkpoint_sha256": "",
        }
    )
    _write(checkpoint_path, checkpoint)
    state_path = boundary_root / "STATE.json"
    state = _self_digested(
        {
            "schema": "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "completed_phase_count": 1,
            "total_phase_count": 1,
            "completed_paired_events": len(cells),
            "event_chain_sha256": chain,
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_count": 1,
            "complete": False,
            "elapsed_seconds": 1.0,
            "broker_hold_seconds": 0.0,
        }
    )
    _write(state_path, state)
    return {"path": str(state_path.relative_to(root)), "sha256": authority.file_digest(state_path)}


def _isolation_observation_ref(
    root: Path,
    name: str,
    *,
    kind: str,
    frozen: dict[str, Any],
    roots: dict[str, str],
    principals: dict[str, dict[str, Any]],
    mounts: dict[str, Any],
) -> dict[str, str]:
    expectation = authority.ISOLATION_OBSERVATION_EXPECTATIONS[kind]
    actor_role = expectation["actor_role"]
    target_field = expectation["target_root_field"]
    assert isinstance(actor_role, str)
    path = root / "evidence" / f"{name}.json"
    denied = expectation["access_result"] == "denied"
    _write(
        path,
        _sealed(
            authority.ISOLATION_OBSERVATION_SCHEMA,
            {
                "frozen_build_sha256": frozen["sha256"],
                "observation_kind": kind,
                "observed_at": "2026-08-02T00:00:00Z",
                "command_argv": ["fixture-isolation-check", kind],
                "actor_role": actor_role,
                "actor_id": principals[actor_role]["id"],
                "actor_uid": principals[actor_role]["uid"],
                "attempt": {
                    "operation": expectation["operation"],
                    "target_root": roots[target_field] if isinstance(target_field, str) else None,
                },
                "access_result": expectation["access_result"],
                "assertion_passed": True,
                "process_exit_code": 1 if denied else 0,
                "attempted": True,
                "errno_name": "EACCES" if denied else None,
                "errno": 13 if denied else None,
                "topology": {"roots": roots, "principals": principals, "mounts": mounts},
            },
            status="observed",
        ),
    )
    return {"path": str(path.relative_to(root)), "sha256": authority.file_digest(path)}


def _checkpoint_ref(root: Path, name: str, *, kind: str, parent: str, event_chain: str) -> tuple[dict[str, str], dict[str, Any]]:
    path = root / "evidence" / f"{name}.json"
    document = _sealed(
        "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
        {
            "kind": kind,
            "parent_checkpoint_sha256": parent,
            "event_chain_sha256": event_chain,
            "cycle": 1,
            "completed_phase_count": 4,
        },
    )
    _write(path, document)
    return {"path": str(path.relative_to(root)), "sha256": authority.file_digest(path)}, document


def _candidate_manifest_rows(root: Path, frozen: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frontier in authority.FRONTIER_IDS:
        source_path = root / "inputs" / f"{frontier}-corpus.json"
        _write(source_path, _sealed("SUBSTRATE_ODYSSEY_TEST_SOURCE/v1", {"frontier": frontier, "role": "fixture_corpus"}))
        assets = [
            {
                "path": str(source_path.relative_to(root)),
                "sha256": authority.file_digest(source_path),
                "role": "fixture_corpus",
                "read_only": True,
            }
        ]
        source_bundle = {
            "selection_sha256": authority.digest({"frontier": frontier, "assets": assets}),
            "assets": assets,
        }
        secret = f"fixture-custodian-seed-{frontier}"
        seed_commitment = task_bank._digest({"seed": secret})
        manifest, _evaluator = task_bank.materialize(seed_commitment, secret, frontier, authority.FULL_TASKS_PER_FRONTIER)
        manifest["source_bundle"] = source_bundle
        manifest.pop("sha256")
        manifest["sha256"] = authority.digest(manifest)
        path = root / "manifests" / f"{frontier}.candidate.json"
        _write(path, manifest)
        task_ids = [task["task_id"] for task in manifest["tasks"]]
        rows.append(
            {
                "id": frontier,
                "path": str(path.relative_to(root)),
                "file_sha256": authority.file_digest(path),
                "schema": manifest["schema"],
                "frontier": frontier,
                "seed_commitment": manifest["seed_commitment"],
                "task_count": len(manifest["tasks"]),
                "task_ids_sha256": authority.digest({"task_ids": task_ids}),
                "source_bundle_sha256": authority.digest(source_bundle),
                "source_selection_sha256": source_bundle["selection_sha256"],
            }
        )
    return rows


def _g06_payload(root: Path, frozen: dict[str, Any], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "receipt_invariant": True,
        "no_memory_threshold_breach": True,
        "no_critical_pressure": True,
        "no_unexpected_swap_or_pageout_increase": True,
        "io_latency_within_sealed_limit": True,
        "slowdown_within_sealed_limit": True,
        "distinct_run_roots": True,
        "no_shared_writable_evaluator_or_data_root": True,
        "record_cpu_memory_io": True,
        "strict_dispatch_deadline_met": True,
        "production_paired_adapters_complete": True,
        "source_bundle_revalidation_complete": True,
        "parent_global_dwell_complete": True,
    }
    g02_path = root / "receipts/G02.subject.json"
    g03_path = root / "receipts/G03.subject.json"
    g02 = authority._read_json(g02_path, require_digest=True)
    base_model = g02["base_model"]
    adapter_sha256 = g02["candidate"]["adapter_sha256"]
    manifest_by_frontier = {row["id"]: row for row in manifest_rows}
    manifest_bindings = [
        {
            "id": frontier,
            "path": manifest_by_frontier[frontier]["path"],
            "sha256": manifest_by_frontier[frontier]["file_sha256"],
        }
        for frontier in authority.FRONTIER_IDS
    ]
    retrieval_tasks = {
        frontier: authority._read_json(root / manifest_by_frontier[frontier]["path"], require_digest=True)["tasks"][0]
        for frontier in authority.FRONTIER_IDS
    }
    phase_harness = {
        "schema": "SUBSTRATE_ODYSSEY_G06_REAL_PHASE_HARNESS/v1",
        "measurement_basis": "active_paired_dispatch_wall_with_deadline_guard",
        "full_phase_seconds": 1800,
        "strict_dispatch_budget_seconds": 150,
        "scale_factor": 12,
        "phase_boundary_guard_interval_seconds": 30,
        "paired_adapter_dispatches_per_cell": 2,
        "source_bundle_pre_dispatch_revalidation": True,
        "scheduling_mode": "initial_release_only;per_frontier_candidate_then_control;no_global_role_barrier;parent_global_dwell",
        "worker_sha256": frozen["implementation_sha256"]["odyssey_worker"],
        "adapter_sha256": adapter_sha256,
        "model": base_model["id"],
        "max_output_tokens": 64,
        "g03_manifest_bindings": manifest_bindings,
    }
    phase_harness["dispatch_contract_sha256"] = authority.digest(phase_harness)
    phase_harness.update(
        {
            "g02_subject": {"path": str(g02_path.relative_to(root)), "sha256": authority.file_digest(g02_path)},
            "g03_subject": {"path": str(g03_path.relative_to(root)), "sha256": authority.file_digest(g03_path)},
            "minimum_width_eight_scheduled_seconds": 450,
        }
    )
    observations = []
    for width in authority.CALIBRATION_WIDTHS:
        for repetition in range(1, authority.CALIBRATION_REPETITIONS + 1):
            phase_authority_sha256 = authority.digest(
                {
                    "schema": "SUBSTRATE_ODYSSEY_G06_CALIBRATION_AUTHORITY/v1",
                    "dispatch_contract_sha256": phase_harness["dispatch_contract_sha256"],
                    "width": width,
                    "repetition": repetition,
                }
            )
            phase_run_id = f"g06-{phase_harness['dispatch_contract_sha256'][:16]}-{width}x-{repetition}"
            cells = []
            refs = []
            for frontier in authority.FRONTIER_IDS[:width]:
                prefix = f"calibration/{width}/{repetition}/{frontier}"
                task = retrieval_tasks[frontier]
                cells.append(
                    {
                        "id": frontier,
                        "candidate_root": f"{prefix}/candidate",
                        "control_root": f"{prefix}/control",
                        "candidate_event_ledger": f"{prefix}/candidate-events.jsonl",
                        "control_event_ledger": f"{prefix}/control-events.jsonl",
                        "candidate_checkpoint_root": f"{prefix}/candidate-checkpoints",
                        "control_checkpoint_root": f"{prefix}/control-checkpoints",
                        "candidate_mutable_state_root": f"{prefix}/candidate-state",
                        "control_mutable_state_root": f"{prefix}/control-state",
                        "candidate_model_context_root": f"{prefix}/candidate-context",
                        "control_model_context_root": f"{prefix}/control-context",
                        "resource_parity": {
                            "candidate": {
                                "allowed_observations": ["fixture-observation"],
                                "models": [base_model["id"]],
                                "tools": ["fixture-tool@pin"],
                                "token_budget": 64,
                                "compute_ceiling": 100,
                                "storage_ceiling": 100,
                                "wall_time_seconds": 1800,
                            },
                            "control": {
                                "allowed_observations": ["fixture-observation"],
                                "models": [base_model["id"]],
                                "tools": ["fixture-tool@pin"],
                                "token_budget": 64,
                                "compute_ceiling": 100,
                                "storage_ceiling": 100,
                                "wall_time_seconds": 1800,
                            },
                        },
                        "task_binding": {
                            "manifest_path": manifest_by_frontier[frontier]["path"],
                            "manifest_sha256": manifest_by_frontier[frontier]["file_sha256"],
                            "task_index": 0,
                            "task_id": task["task_id"],
                            "task_sha256": authority.digest(task),
                        },
                        "model_call_count": 2,
                        "source_bundle_guard_calls": 2,
                        "active_work_seconds": 1.0,
                        "deadline_met": True,
                    }
                )
                cell = cells[-1]
                candidate_receipt = _g06_adapter_receipt_ref(
                    root,
                    name=f"g06-{width}-{repetition}-{frontier}",
                    role="candidate",
                    frontier=frontier,
                    task=task,
                    manifest_sha256=manifest_by_frontier[frontier]["file_sha256"],
                    authority_sha256=phase_authority_sha256,
                    run_id=phase_run_id,
                    model=base_model["id"],
                    adapter_sha256=adapter_sha256,
                )
                control_receipt = _g06_adapter_receipt_ref(
                    root,
                    name=f"g06-{width}-{repetition}-{frontier}",
                    role="control",
                    frontier=frontier,
                    task=task,
                    manifest_sha256=manifest_by_frontier[frontier]["file_sha256"],
                    authority_sha256=phase_authority_sha256,
                    run_id=phase_run_id,
                    model=base_model["id"],
                    adapter_sha256=adapter_sha256,
                )
                cell["candidate_receipt"] = candidate_receipt
                cell["control_receipt"] = control_receipt
                refs.extend((candidate_receipt, control_receipt))
            boundary_receipt = _g06_phase_boundary_ref(
                root,
                name=f"g06-{width}-{repetition}",
                authority_sha256=phase_authority_sha256,
                run_id=phase_run_id,
                cells=cells,
            )
            refs.append(boundary_receipt)
            observations.append(
                {
                    "width": width,
                    "repetition": repetition,
                    "cells": cells,
                    "metrics": {
                        "aggregate_throughput": float(width),
                        "per_cell_slowdown_ratio": 1.0,
                        "resident_memory_bytes": width * 1024,
                        "swap_pageout_delta_bytes": 0,
                        "disk_latency_ms": 1.0,
                        "checkpoint_latency_ms": 1.0,
                        "model_latency_ms": 1.0,
                        "cpu_time_seconds": 1.0,
                        "io_bytes": width * 1024,
                        "thermal_pressure": "nominal",
                        "critical_pressure": False,
                        "strict_dispatch_budget_seconds": 150,
                        "scheduled_phase_seconds": 150,
                        "global_dwell_seconds": 1.0,
                        "parent_guard_samples": 1,
                        "paired_adapter_dispatches": 2 * width,
                        "phase_boundary_receipt": boundary_receipt,
                        "observation_wall_seconds": 150.0,
                        "active_dispatch_wall_seconds": 1.0,
                        "raw_active_dispatch_slowdown_ratio": 1.0,
                        "e2e_slowdown_ratio": 1.0,
                        "width1_baseline_seconds": 1.0,
                        "slowdown_basis": "active_paired_dispatch_wall_with_deadline_guard",
                    },
                    "checks": dict(required_checks),
                    "receipt_refs": refs,
                }
            )
    return {
        **_machine_binding(frozen),
        "admitted_width": 8,
        "full_program_requires_width": 8,
        "calibration_widths": list(authority.CALIBRATION_WIDTHS),
        "repetitions_per_width": authority.CALIBRATION_REPETITIONS,
        "checks": required_checks,
        "observations": observations,
        "phase_harness": phase_harness,
        "width_eight_scheduled_seconds": 450.0,
        "all_pass": True,
    }


def _g06_dc_payload(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    """Build a genuinely valid G06-DC deadline-capacity subject for fixture roots.

    Reuses the dedicated G06-DC passing subject so this fixture stays aligned with
    the real gate contract (tool-bearing ladder, deadline headroom, preserved
    4.39x width-8 slowdown, prior width-calibration binding). Envelope keys are
    stripped because ``_prepared_inputs`` re-seals with the launch schema; the
    frozen machine binding is kept so ``_require_frozen_subject_binding`` still
    sees the fixture build.
    """
    from tests.substrate.test_odyssey_g06_dc import _passing_subject

    prior_path = root / "evidence/substrate/odyssey/ODYSSEY_ARM_PROTOCOL_V2_WIDTH_CALIBRATION.json"
    if not prior_path.is_file():
        raise AssertionError("fixture root lacks prior G06 width-calibration evidence for G06-DC")
    subject = _passing_subject(root, frozen)
    envelope = {
        "schema",
        "program",
        "status",
        "activation",
        "external_activation",
        "unqualified_nous",
        "sha256",
    }
    payload = {key: value for key, value in subject.items() if key not in envelope}
    # Re-assert the fixture machine binding in case the dedicated builder ever
    # drifts away from the shared authority-test frozen maps.
    payload.update(_machine_binding(frozen))
    return payload


def _g07_payload(root: Path, frozen: dict[str, Any], storage: dict[str, int]) -> dict[str, Any]:
    observations = []
    base_growth, growth_remainder = divmod(storage["p95_private_growth_bytes"], len(authority.FRONTIER_IDS))
    for offset, frontier in enumerate(authority.FRONTIER_IDS):
        observations.append(
            {
                "id": frontier,
                "candidate_root": f"storage/{frontier}/candidate",
                "control_root": f"storage/{frontier}/control",
                "candidate_checkpoint_root": f"storage/{frontier}/candidate-checkpoints",
                "control_checkpoint_root": f"storage/{frontier}/control-checkpoints",
                "candidate_mutable_state_root": f"storage/{frontier}/candidate-state",
                "control_mutable_state_root": f"storage/{frontier}/control-state",
                "candidate_model_context_root": f"storage/{frontier}/candidate-context",
                "control_model_context_root": f"storage/{frontier}/control-context",
                "event_count": 4,
                "checkpoint_count": 1,
                "log_bytes": 100 + offset,
                "model_call_ledger_bytes": 100 + offset,
                "media_access_count": 1,
                "daily_compaction": True,
                "restart_count": 1,
                "restore_count": 1,
                "durable_growth_bytes": base_growth + (1 if offset < growth_remainder else 0),
                "largest_transient_bytes": storage["largest_transient_bytes"] - (len(authority.FRONTIER_IDS) - 1 - offset),
                "receipt_refs": [_receipt_ref(root, f"g07-{frontier}", {"frontier": frontier})],
            }
        )
    checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "eight_cells_exercised": True,
        "event_rate_reproduced": True,
        "checkpoint_rate_reproduced": True,
        "log_rate_reproduced": True,
        "model_call_ledger_rate_reproduced": True,
        "media_access_reproduced": True,
        "daily_compaction_reproduced": True,
        "restart_reproduced": True,
        "restore_reproduced": True,
        "private_roots_distinct": True,
        "measurements_nonzero": True,
        "full_width_concurrent_transient_bound": True,
        "formula_bound": True,
    }
    design = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_7D.hardened.draft.json")
    return {
        **_machine_binding(frozen),
        "cells": len(authority.FRONTIER_IDS),
        "reproduced_operations": list(authority.STORAGE_REHEARSAL_OPERATIONS),
        "formula": design["storage"]["launch_formula"],
        "cell_observations": observations,
        "observed_total_private_growth_bytes": sum(row["durable_growth_bytes"] for row in observations),
        "concurrent_transient_slots": len(authority.FRONTIER_IDS),
        **storage,
        "private_write_cap_bytes": storage["p95_private_growth_bytes"],
        "observed_free_before_bytes": 400 * 1024**3,
        "observed_free_after_bytes": 399 * 1024**3,
        "minimum_free_bytes_observed": 398 * 1024**3,
        "checks": checks,
        "all_pass": True,
    }


def _g08_payload(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    cases = (
        ("below_normal_admission", 74.9, False, "admit_or_resume"),
        ("normal_admission_boundary", 75.0, False, "deny_new_work"),
        ("p2_checkpoint_boundary", 80.0, False, "checkpoint_reduce_p2"),
        ("p1_pause_boundary", 82.0, False, "pause_p1_checkpoint_p2"),
        ("global_hold_boundary", 85.0, False, "safe_hold_non_p0"),
        ("critical_pressure_override", 74.0, True, "safe_hold_non_p0"),
    )
    observations = []
    for case, resident, critical, decision in cases:
        pools = {"host": resident - 22.0, "vm": 10.0, "container": 5.0, "model_service": 4.0, "broker": 1.0}
        observations.append(
            {
                "case": case,
                "resident_gib": resident,
                "critical_pressure": critical,
                "decision": decision,
                "memory_pools_gib": pools,
                "accounted_total_gib": resident,
                "lane_resident_gib": {"A": 4.0, "B": 3.0},
                "receipt_refs": [_receipt_ref(root, f"g08-{case}", {"case": case})],
            }
        )
    checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "threshold_table_bound": True,
        "all_required_pools_observed": True,
        "sampling_cadence_bound": True,
        "critical_pressure_override": True,
        "decision_receipts_bound": True,
        "no_semantic_decision": True,
    }
    return {
        **_machine_binding(frozen),
        "resident_cap_gib": 85,
        "normal_admission_ceiling_gib": 75,
        "p2_checkpoint_threshold_gib": 80,
        "p1_pause_threshold_gib": 82,
        "global_hold_threshold_gib": 85,
        "measurement_interval_seconds": 30,
        "accounting_uncertainty_gib": 2,
        "broker_source_sha256": frozen["implementation_sha256"]["odyssey_worker"],
        "observations": observations,
        "checks": checks,
        "all_pass": True,
    }


def _recovery_arm(root: Path, frontier: str, role: str) -> dict[str, Any]:
    full_ref, full = _checkpoint_ref(root, f"g09-{frontier}-{role}-full", kind="full", parent="", event_chain="b" * 64)
    delta_ref, delta = _checkpoint_ref(
        root,
        f"g09-{frontier}-{role}-delta",
        kind="delta",
        parent=full["sha256"],
        event_chain="c" * 64,
    )
    return {
        "pre_interrupt_state_sha256": "d" * 64,
        "restored_state_sha256": "d" * 64,
        "full_checkpoint": full_ref,
        "delta_checkpoints": [delta_ref],
        "event_trace": _receipt_ref(root, f"g09-{frontier}-{role}-trace", {"event_chain_sha256": delta["event_chain_sha256"]}),
        "restart_receipt": _receipt_ref(root, f"g09-{frontier}-{role}-restart", {"recovered": True, "interactive_shell_independent": True}),
        "writer_lock": f"durability/{frontier}/{role}.lock",
        "single_writer_receipt": _receipt_ref(root, f"g09-{frontier}-{role}-writer", {"single_writer": True}),
        "recovery_downtime_seconds": 10,
        "resumed_at_sealed_boundary": True,
    }


def _g09_payload(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "checkpoint_round_trip": True,
        "delta_plus_full_restore": True,
        "process_restart": True,
        "model_replacement": True,
        "tool_or_body_change": True,
        "sensor_or_source_interruption": True,
        "single_writer": True,
        "interactive_shell_independent": True,
        "recovery_limits_bound": True,
        "event_chain_valid": True,
    }
    rehearsals = [
        {
            "frontier": frontier,
            "arms": {"candidate": _recovery_arm(root, frontier, "candidate"), "control": _recovery_arm(root, frontier, "control")},
            "unplanned_interruptions": 1,
            "max_single_unplanned_downtime_seconds": 10,
            "cumulative_unplanned_downtime_seconds": 10,
        }
        for frontier in authority.FRONTIER_IDS
    ]
    return {
        **_machine_binding(frozen),
        "checkpoint_policy": {"delta_interval_seconds": 7200, "full_interval_seconds": 43200},
        "rehearsals": rehearsals,
        "scheduled_disturbance_receipts": {
            name: _receipt_ref(root, f"g09-{name}", {"recovered": True})
            for name in ("process_restart", "model_replacement", "tool_or_body_change", "sensor_or_source_interruption")
        },
        "checks": checks,
        "all_pass": True,
    }


def _g12_payload(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for mutation_id in sorted(authority.REQUIRED_MUTATION_IDS):
        rows.append(
            {
                "id": mutation_id,
                "target": f"fixture/{mutation_id}",
                "injected": True,
                "detected": True,
                "survived": False,
                "clean_case_passed": True,
                "clean_receipt": _receipt_ref(root, f"g12-{mutation_id}-clean", {"mutation": mutation_id, "rejected": False}),
                "mutant_receipt": _receipt_ref(root, f"g12-{mutation_id}-mutant", {"mutation": mutation_id, "rejected": True}),
            }
        )
    checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "runtime_mutants_injected": True,
        "runtime_mutants_detected": True,
        "clean_baselines_accepted": True,
        "guard_coverage_complete": True,
        "no_pending_mutations": True,
    }
    return {
        **_machine_binding(frozen),
        "mutations": rows,
        "declared_mutation_count": len(rows),
        "injected_count": len(rows),
        "detected_count": len(rows),
        "pending_count": 0,
        "survivor_count": 0,
        "survivors": [],
        "uncovered": [],
        "undeclared": [],
        "checks": checks,
        "all_pass": True,
    }


def _human_binding(frozen: dict[str, Any]) -> dict[str, Any]:
    """Binding for converted machine subjects (status=pass on the machine path)."""
    return {
        "status": "pass",
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": "fixture-head",
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
        "unqualified_nous": False,
    }


def _fixture_digest(*parts: str) -> str:
    return authority.digest({"fixture": list(parts)})


def _fixture_base_model(name: str, runtime_sha256: str) -> dict[str, str]:
    return {
        "id": name,
        "revision": f"ollama:{_fixture_digest(name, 'weight')[:16]}",
        "weight_sha256": _fixture_digest(name, "weight"),
        "tokenizer_sha256": _fixture_digest(name, "tokenizer"),
        "runtime_sha256": runtime_sha256,
        "quantization": "Q4_K_M",
    }


def _fixture_selected_base(frozen: dict[str, Any]) -> dict[str, str]:
    return _fixture_base_model("qwen3:30b", _fixture_digest("public-canary", "runtime"))


def _public_model_canary_ref(root: Path, frozen: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    template = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_PUBLIC_MODEL_CANARY.template.json")
    runtime_sha256 = _fixture_digest("public-canary", "runtime")
    scores = {"gpt-oss:20b": 14, "qwen3:30b": 16, "deepseek-r1:32b": 15}
    service_peaks = {"gpt-oss:20b": 14 * 1024**3, "qwen3:30b": 19 * 1024**3, "deepseek-r1:32b": 20 * 1024**3}
    candidates = []
    for alias in template["candidate_aliases"]:
        passed = scores[alias]
        case_results = [
            {
                "id": case["id"],
                "response_sha256": _fixture_digest(alias, case["id"], "response"),
                "answer": f"fixture-{case['id']}",
                "passed": index < passed,
                "latency_ms": float(100 + index),
            }
            for index, case in enumerate(template["case_set"])
        ]
        candidates.append(
            {
                "base_model": _fixture_base_model(alias, runtime_sha256),
                "model_size_bytes": service_peaks[alias],
                "service_peak_bytes": service_peaks[alias],
                "swap_pageout_delta_bytes": 0,
                "width_eight": {"requests": 8, "completed": 8, "all_responses_valid": True},
                "canary": {
                    "total": len(case_results),
                    "passed": passed,
                    "median_latency_ms": 107.5,
                    "case_results": case_results,
                },
                "errors": [],
                "eligible": True,
            }
        )
    selected = _fixture_selected_base(frozen)
    document = _sealed(
        authority.PUBLIC_MODEL_CANARY_SCHEMA,
        {
            "scientific_evidence": False,
            "evidence_scope": "frozen_public_model_selection_canaries_only",
            "completed_at": "2026-08-03T00:00:00+00:00",
            "frozen_build_sha256": frozen["sha256"],
            "canary_template_sha256": frozen["input_sha256"]["public_model_canary_template"],
            "runtime": {"id": "ollama", "version": "fixture", "sha256": runtime_sha256},
            "model_service_cap_bytes": 24 * 1024**3,
            "required_concurrent_clients": 8,
            "selection_rule": template["selection_rule"],
            "hidden_seed_commitments_materialized": False,
            "candidates": candidates,
            "selected_base_model": selected,
            "checks": {name: True for name in authority.PUBLIC_MODEL_CANARY_CHECKS},
            "all_pass": True,
            "non_claims": ["fixture technical screen only"],
        },
    )
    path = root / "evidence/public-model-canary.json"
    _write(path, document)
    return {"path": str(path.relative_to(root)), "sha256": authority.file_digest(path)}, selected


def _arm_pin(root: Path, name: str, base_model: dict[str, str]) -> dict[str, str]:
    return {
        "id": f"{name}-id",
        "revision": base_model["revision"],
        "artifact_sha256": base_model["weight_sha256"],
        "adapter_sha256": transition.canonical_source_digest(root / "src/substrate/odyssey_arms.py"),
    }


def _roots() -> dict[str, str]:
    return {
        "builder_visible_root": "builder-visible",
        "candidate_visible_root": "candidate-visible",
        "evaluator_only_root": "evaluator-only",
        "publication_safe_root": "publication-safe",
    }


def _subject_payload(
    root: Path,
    gate_id: str,
    frozen: dict[str, Any],
    storage: dict[str, int],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen_sha256 = frozen["sha256"]
    if gate_id == "G01":
        return {"state": "odyssey_preflight_authorized", "preflight_authorized": True, "frozen_build_sha256": frozen_sha256}
    if gate_id == "G02":
        public_model_canary, base_model = _public_model_canary_ref(root, frozen)
        candidate = _arm_pin(root, "candidate", base_model)
        return {
            **_human_binding(frozen),
            "selection_id": "fixture-pre-outcome-selection",
            "public_model_canary": public_model_canary,
            "base_model": base_model,
            "candidate": {**candidate, "treatment_id": "fixture-developmental-intervention"},
            "controls_by_frontier": {
                frontier: _arm_pin(root, f"control-{frontier}", base_model) for frontier in authority.FRONTIER_IDS
            },
            "parity_by_frontier": {
                frontier: {field: True for field in authority.PARITY_FIELDS} for frontier in authority.FRONTIER_IDS
            },
            "selection_checks": {
                "pre_outcome_selection": True,
                "public_canary_receipt_reviewed": True,
                "one_shared_base_body_verified": True,
                "candidate_pin_complete": True,
                "control_pins_complete": True,
                "candidate_control_difference_declared": True,
                "parity_reviewed": True,
            },
        }
    if gate_id == "G03":
        return {
            **_machine_binding(frozen),
            "task_bank_generator_sha256": frozen["implementation_sha256"]["task_bank_generator"],
            "frontier_contract_sha256": frozen["input_sha256"]["frontier_contract"],
            "task_bank_sha256": frozen["input_sha256"]["task_bank"],
            "rendered_build_index_sha256": frozen["input_sha256"]["rendered_build_index"],
            "source_selection_sha256": authority.digest({"fixture": "source-selection"}),
            "manifest_count": len(manifest_rows),
            "manifests": manifest_rows,
            "checks": {
                "frozen_build_bound": True,
                "source_maps_bound": True,
                "candidate_manifests_structurally_safe": True,
                "frontier_set_exact": True,
                "scheduled_task_count_exact": True,
                "source_bundle_bound": True,
            },
            "all_pass": True,
        }
    if gate_id == "G04":
        manifest_by_frontier = {row["id"]: row for row in manifest_rows}
        ordered_commitment_rows = []
        frontiers = []
        for frontier in authority.FRONTIER_IDS:
            row = {
                "id": frontier,
                "task_seed_commitment_sha256": _fixture_digest("G04", frontier, "task-seed"),
                "answer_commitment_sha256": _fixture_digest("G04", frontier, "answer"),
                "scorer_commitment_sha256": _fixture_digest("G04", frontier, "scorer"),
                "candidate_manifest_sha256": manifest_by_frontier[frontier]["file_sha256"],
                "candidate_can_read_evaluator_only": False,
                "trace_lock_required": True,
                "daily_scores_hidden": True,
            }
            frontiers.append(row)
            ordered_commitment_rows.append(
                {
                    "id": frontier,
                    "task_seed_commitment_sha256": row["task_seed_commitment_sha256"],
                    "answer_commitment_sha256": row["answer_commitment_sha256"],
                    "scorer_commitment_sha256": row["scorer_commitment_sha256"],
                    "candidate_manifest_sha256": row["candidate_manifest_sha256"],
                }
            )
        return {
            **_human_binding(frozen),
            "answers_evaluator_only": True,
            "trace_lock_before_answer_reveal": True,
            "daily_scores_hidden": True,
            "custody_independence": authority.G04_CUSTODY_INDEPENDENCE,
            "custody_limitations": [authority.G04_CUSTODY_LIMITATION_STATEMENT],
            "roots": _roots(),
            "frontiers": frontiers,
            "pre_launch_commitment_seal": {
                "sealed_before_launch": True,
                "commitment_set_sha256": authority.digest({"frontiers": ordered_commitment_rows}),
                "frontiers_commitment_chain_sha256": authority.digest(
                    {
                        "algorithm": "sha256_canonical_json",
                        "ordered_frontier_commitments": ordered_commitment_rows,
                    }
                ),
            },
            "day7_reveal": {
                "gated_on_trace_lock": True,
                "trace_lock_recipe": dict(authority.G04_TRACE_LOCK_RECIPE),
                "trace_lock_recipe_sha256": authority.digest(authority.G04_TRACE_LOCK_RECIPE),
                "release_after_candidate_and_control_trace_lock": True,
            },
            "custody_checks": {name: True for name in sorted(authority.G04_CUSTODY_CHECKS)},
        }
    if gate_id == "G05":
        base_model = _fixture_selected_base(frozen)
        candidate = _arm_pin(root, "candidate", base_model)
        controls = {frontier: _arm_pin(root, f"control-{frontier}", base_model) for frontier in authority.FRONTIER_IDS}
        return {
            **_human_binding(frozen),
            "panel_id": "fixture-model-tool-panel",
            "models": [candidate, *[controls[frontier] for frontier in authority.FRONTIER_IDS]],
            "tools": [
                {
                    "id": "fixture-tool",
                    "version": "fixture-tool-v1",
                    "artifact_sha256": _fixture_digest("G05", "tool"),
                }
            ],
            "gateway": {
                "id": "fixture-stateless-gateway",
                "revision": "fixture-gateway-v1",
                "artifact_sha256": _fixture_digest("G05", "gateway"),
                "stateless": True,
            },
            "frontier_assignments": {
                frontier: {
                    "candidate_model_id": candidate["id"],
                    "control_model_id": controls[frontier]["id"],
                    "candidate_tool_ids": ["fixture-tool"],
                    "control_tool_ids": ["fixture-tool"],
                }
                for frontier in authority.FRONTIER_IDS
            },
            "panel_checks": {
                "model_pins_complete": True,
                "tool_pins_complete": True,
                "stateless_gateway_pinned": True,
                "frontier_assignments_complete": True,
                "candidate_control_tool_parity": True,
            },
        }
    if gate_id == "G06":
        return _g06_payload(root, frozen, manifest_rows)
    if gate_id == "G06-DC":
        return _g06_dc_payload(root, frozen)
    if gate_id == "G07":
        return _g07_payload(root, frozen, storage)
    if gate_id == "G08":
        return _g08_payload(root, frozen)
    if gate_id == "G09":
        return _g09_payload(root, frozen)
    if gate_id in {"G13", "G14"}:
        return {"all_pass": True}
    if gate_id == "G10":
        roots = _roots()
        principals = {
            "candidate": {"id": "candidate-fixture", "uid": 501},
            "evaluator": {"id": "evaluator-fixture", "uid": 502},
            "builder": {"id": "builder-fixture", "uid": 503},
        }
        mounts: dict[str, Any] = {}
        return {
            **_human_binding(frozen),
            "isolation_mode": "separate_uid",
            "candidate_can_read_evaluator_only": False,
            "candidate_can_write_evaluator_only": False,
            "evaluator_can_write_candidate_private_state": False,
            "builder_can_read_evaluator_only": False,
            "roots": roots,
            "principals": principals,
            "mounts": mounts,
            "isolation_receipts": {
                name: _isolation_observation_ref(
                    root,
                    f"G10-{name}",
                    kind=name,
                    frozen=frozen,
                    roots=roots,
                    principals=principals,
                    mounts=mounts,
                )
                for name in authority.ISOLATION_OBSERVATION_EXPECTATIONS
            },
            "isolation_checks": {
                "candidate_evaluator_read_denied": True,
                "candidate_evaluator_write_denied": True,
                "evaluator_candidate_private_write_denied": True,
                "builder_evaluator_read_denied": True,
                "topology_observed": True,
                "no_shared_mutable_roots": True,
            },
        }
    if gate_id == "G11":
        design = authority._frozen_design(root, frozen)
        statistics = design["statistics"]
        independent = design["independent_units"]
        return {
            **_human_binding(frozen),
            "statistics_authority_id": "fixture-statistics-authority",
            "score_weights_frozen": True,
            "score_weights": {dimension: 0.25 for dimension in authority.SCORE_DIMENSIONS},
            "rubric_sha256": {dimension: _fixture_digest("G11", dimension, "rubric") for dimension in authority.SCORE_DIMENSIONS},
            "analysis_plan_sha256": _fixture_digest("G11", "analysis-plan"),
            "primary_unit": statistics["primary_unit"],
            "independent_unit_count": independent["count"],
            "repeated_observations_are_independent_replicates": False,
            "sesoi": statistics["sesoi"],
            "primary_methods": statistics["primary_methods"],
            "secondary_event_model": statistics["secondary_event_model"],
            "outcome_a_requires_all_eight_valid": statistics["outcome_a_requires_all_eight_valid"],
            "analysis_checks": {
                "score_weights_sum_to_one": True,
                "rubrics_pinned": True,
                "primary_unit_matches_design": True,
                "pseudoreplication_guard": True,
                "primary_methods_frozen": True,
                "outcome_rule_frozen": True,
            },
        }
    if gate_id == "G12":
        return _g12_payload(root, frozen)
    if gate_id == "G15":
        return {"frozen_build_sha256": frozen_sha256, "source_digest": "b" * 64, "protocol_digest": "c" * 64}
    raise AssertionError(gate_id)


def _prepared_inputs(root: Path, *, model_reserve: int = 0) -> tuple[Path, dict]:
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    p95, transient, terminal = 8 * 1024**2, 2 * 1024**2, 1 * 1024**2
    runtime_required = (
        authority.BASE_PROTECTED_FLOOR_BYTES
        + model_reserve
        + len(authority.FRONTIER_IDS) * transient
        + terminal
    )
    storage = {
        "base_protected_floor_bytes": authority.BASE_PROTECTED_FLOOR_BYTES,
        "runtime_required_free_bytes": runtime_required,
        "p95_private_growth_bytes": p95,
        "largest_transient_bytes": transient,
        "terminal_allowance_bytes": terminal,
        "explicit_model_reserve_bytes": model_reserve,
        "measured_required_free_bytes": runtime_required + p95,
    }
    manifest_rows = _candidate_manifest_rows(root, frozen)
    gate_refs: dict[str, dict[str, str]] = {}
    for gate_id, spec in authority.GATE_SPECS.items():
        payload = _subject_payload(root, gate_id, frozen, storage, manifest_rows)
        if gate_id == "G13":
            payload.update(
                {
                    "checks": {
                        "exact_commit_checkout": True,
                        "scoped_tests": True,
                        "ruff_check": True,
                        "frozen_build_regeneration": True,
                        "source_map_match": True,
                    },
                    "source_commit": "fixture-head",
                    "frozen_build_sha256": frozen["sha256"],
                    "regenerated_frozen_build_sha256": frozen["sha256"],
                    "implementation_sha256": frozen["implementation_sha256"],
                    "input_sha256": frozen["input_sha256"],
                }
            )
        if gate_id == "G14":
            payload.update(
                {
                    "checks": {
                        "frozen_build_bound": True,
                        "notifier_source_bound": True,
                        "telegram_api_acknowledged": True,
                        "probe_message_id_valid": True,
                    },
                    "source_commit": "fixture-head",
                    "frozen_build_sha256": frozen["sha256"],
                    "notifier_source_sha256": frozen["implementation_sha256"]["telegram_notifier"],
                    "delivery": {"message_id": 1, "acknowledged": True},
                }
            )
        if gate_id == "G15":
            payload.update(
                {
                    "all_pass": True,
                    "source_digest": authority.source_digest_for_frozen(frozen),
                    "protocol_digest": authority.protocol_digest_for_frozen(frozen),
                }
            )
        subject = _sealed(spec["subject_schema"], payload)
        subject_path = root / "receipts" / f"{gate_id}.subject.json"
        _write(subject_path, subject)
        gate = _sealed(
            "SUBSTRATE_ODYSSEY_GATE_EVIDENCE/v1",
            {
                "gate_id": gate_id,
                "gate_name": spec["name"],
                "evidence_kind": spec["kind"],
                "frozen_build_sha256": frozen["sha256"],
                "subject": {
                    "path": str(subject_path.relative_to(root)),
                    "file_sha256": authority.file_digest(subject_path),
                    "schema": spec["subject_schema"],
                },
                "checks": {"fixture_check": True},
                "human_attestation": (
                    {"actor": "custodian-fixture", "attested_at": "2026-08-02T00:00:00Z", "statement": f"{gate_id} explicitly reviewed"}
                    if spec["kind"] == "human_attested"
                    else None
                ),
            },
        )
        gate_path = root / "receipts" / f"{gate_id}.gate.json"
        _write(gate_path, gate)
        gate_refs[gate_id] = {"path": str(gate_path.relative_to(root)), "file_sha256": authority.file_digest(gate_path)}
    # Historical G06 is no longer a launch gate, but its validator and 1.35
    # simultaneity limit remain reachable. Emit a fixture subject so preserved-
    # validator tests can still exercise real arm receipts and resource parity.
    historical_g06 = _sealed("SUBSTRATE_ODYSSEY_WIDTH_CALIBRATION/v1", _g06_payload(root, frozen, manifest_rows))
    _write(root / "receipts/G06.subject.json", historical_g06)
    worker_frontiers = [
        {
            "id": row["id"],
            "candidate_manifest": row["path"],
            "candidate_manifest_sha256": row["file_sha256"],
            # The operator-input row must carry the whole G03 identity, not
            # merely a path to something that happens to look like a task bank.
            "candidate_manifest_binding": dict(row),
            "candidate_command": ["/usr/bin/true"],
            "control_command": ["/usr/bin/true"],
        }
        for row in manifest_rows
    ]
    worker = root / "src/substrate/odyssey_worker.py"
    draft = {
        "schema": "SUBSTRATE_ODYSSEY_OPERATOR_INPUTS/v1",
        "program": authority.PROGRAM,
        "input_status": "ready_for_preflight",
        "run_id": "odyssey-fixture-001",
        "operator_approval": {
            "actor": "operator-fixture",
            "attested_at": "2026-08-02T00:00:00Z",
            "scope": "Seal only the exact reviewed inputs and passing gate receipts.",
        },
        "frozen_build_sha256": frozen["sha256"],
        "gate_evidence": gate_refs,
        "storage_admission": {
            "p95_private_growth_bytes": p95,
            "largest_transient_bytes": transient,
            "terminal_allowance_bytes": terminal,
            "explicit_model_reserve_bytes": model_reserve,
            "private_write_cap_bytes": p95,
        },
        "worker": {
            "argv": ["/usr/bin/true"],
            "source_files": [{"path": "src/substrate/odyssey_worker.py", "sha256": authority.file_digest(worker)}],
            "run_root": "runs/substrate/odyssey7d/v1/odyssey-fixture-001",
            "frontiers": worker_frontiers,
            "phase_names": list(authority.PHASE_NAMES),
            "phase_seconds": 1800,
            "microcycles_per_frontier": 84,
            "max_parallel_frontiers": len(authority.FRONTIER_IDS),
            "checkpoint": {"delta_interval_seconds": 7200, "full_interval_seconds": 43200},
            "storage": {"candidate_root": "candidate", "control_root": "control", "private_blob_root": "private-blobs"},
        },
        "activation": False,
        "external_activation": False,
    }
    path = root / "authority-inputs.draft.json"
    _write(path, draft)
    return path, draft


@pytest.fixture
def large_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority.shutil, "disk_usage", lambda _path: Usage(500 * 1024**3, 100 * 1024**3, 400 * 1024**3))


@pytest.fixture(autouse=True)
def fixture_git_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority, "_git_head", lambda _root: "fixture-head")


def test_sealed_authority_requires_exact_15_gate_receipts_and_worker_binding(tmp_path: Path, large_volume: None) -> None:
    root = _fixture_root(tmp_path)
    draft, _ = _prepared_inputs(root)
    inputs = root / "authority-inputs.sealed.json"
    authority.seal_inputs(root, draft, inputs)
    preflight = root / "preflight.json"
    result = authority.preflight(root, inputs, preflight)
    assert result["all_gates_pass"] is True
    assert result["preflight_admitted"] is True
    assert result["storage"]["launch_required_free_bytes"] == result["storage"]["measured_required_free_bytes"]
    assert result["storage"]["required_free_bytes"] == result["storage"]["runtime_required_free_bytes"]
    sealed_path = root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json"
    sealed = authority.seal(root, inputs, preflight, sealed_path)
    assert sealed["seal"]["status"] == "sealed"
    assert sealed["program"]["launch_allowed"] is True
    assert sealed["worker_source_sha256"] == authority.file_digest(root / "src/substrate/odyssey_worker.py")
    assert sealed["worker"]["phase_names"] == list(authority.PHASE_NAMES)
    assert sealed["worker"]["max_parallel_frontiers"] == len(authority.FRONTIER_IDS)
    assert authority.verify(root, sealed_path)["all_pass"] is True


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("phase_names", "four-phase Odyssey schedule"),
        ("max_parallel_frontiers", "width-eight admission"),
    ),
)
def test_worker_binding_rejects_a_full_contract_it_cannot_execute(
    tmp_path: Path, large_volume: None, field: str, expected: str
) -> None:
    """Do not seal inputs that the full worker will inevitably reject."""
    root = _fixture_root(tmp_path)
    draft_path, draft = _prepared_inputs(root)
    draft["worker"].pop(field)
    authority._write_json(draft_path, draft, overwrite=True)

    with pytest.raises(authority.Refused, match=expected):
        authority.seal_inputs(root, draft_path, root / "authority-inputs.sealed.json")


def test_machine_gate_rejects_human_attestation_relabel(tmp_path: Path, large_volume: None) -> None:
    """Converted gates are machine_verified and may not carry human attestation."""
    root = _fixture_root(tmp_path)
    draft_path, draft = _prepared_inputs(root)
    gate_path = root / "receipts/G02.gate.json"
    gate = authority._read_json(gate_path, require_digest=True)
    gate["human_attestation"] = {
        "actor": "should-not-appear",
        "attested_at": "2026-08-02T00:00:00Z",
        "statement": "illegal human wrapper on machine gate",
    }
    gate.pop("sha256")
    gate["sha256"] = authority.digest(gate)
    gate_path.unlink()
    _write(gate_path, gate)
    draft["gate_evidence"]["G02"]["file_sha256"] = authority.file_digest(gate_path)
    draft_path.unlink()
    _write(draft_path, draft)
    with pytest.raises(authority.Refused, match="machine-verified and may not be relabeled"):
        authority.seal_inputs(root, draft_path, root / "authority-inputs.sealed.json")


def test_human_subject_validators_reject_shallow_or_incompatible_claims(tmp_path: Path) -> None:
    """Converted machine subjects keep closed shapes and refuse shallow claims."""
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)

    def subject(gate_id: str) -> dict[str, Any]:
        return authority._read_json(root / "receipts" / f"{gate_id}.subject.json", require_digest=True)

    g02 = json.loads(json.dumps(subject("G02")))
    g02["candidate"].pop("adapter_sha256")
    with pytest.raises(authority.Refused, match="G02 candidate has the wrong fields"):
        authority._gate_specific_checks(root, "G02", g02, frozen)

    g02 = json.loads(json.dumps(subject("G02")))
    g02["base_model"]["weight_sha256"] = "0" * 64
    with pytest.raises(authority.Refused, match="does not exactly match the reviewed public-canary selection"):
        authority._gate_specific_checks(root, "G02", g02, frozen)

    g04 = json.loads(json.dumps(subject("G04")))
    g04.pop("custody_limitations")
    with pytest.raises(authority.Refused, match="custody_limitations"):
        authority._gate_specific_checks(root, "G04", g04, frozen)

    g04 = json.loads(json.dumps(subject("G04")))
    g04.pop("custody_independence")
    with pytest.raises(authority.Refused, match="custody_independence|wrong fields"):
        authority._gate_specific_checks(root, "G04", g04, frozen)

    g04 = json.loads(json.dumps(subject("G04")))
    g04["frontiers"][1]["answer_commitment_sha256"] = g04["frontiers"][0]["answer_commitment_sha256"]
    with pytest.raises(authority.Refused, match="must be distinct"):
        authority._gate_specific_checks(root, "G04", g04, frozen)

    g04 = json.loads(json.dumps(subject("G04")))
    g04["day7_reveal"]["gated_on_trace_lock"] = False
    with pytest.raises(authority.Refused, match="not chained to trace lock"):
        authority._gate_specific_checks(root, "G04", g04, frozen)

    g05 = json.loads(json.dumps(subject("G05")))
    g05["frontier_assignments"]["A"]["control_tool_ids"] = []
    with pytest.raises(authority.Refused, match="tool assignment is not parity-preserving"):
        authority._gate_specific_checks(root, "G05", g05, frozen)

    g10 = json.loads(json.dumps(subject("G10")))
    g10["principals"]["evaluator"]["uid"] = g10["principals"]["candidate"]["uid"]
    with pytest.raises(authority.Refused, match="evaluator UID distinct from candidate and builder"):
        authority._gate_specific_checks(root, "G10", g10, frozen)

    g10 = json.loads(json.dumps(subject("G10")))
    g10["isolation_receipts"].pop("builder_evaluator_read_denied")
    with pytest.raises(authority.Refused, match="G10 isolation_receipts has the wrong fields"):
        authority._gate_specific_checks(root, "G10", g10, frozen)

    g10 = json.loads(json.dumps(subject("G10")))
    receipt_ref = g10["isolation_receipts"]["builder_evaluator_read_denied"]
    receipt_path = root / receipt_ref["path"]
    receipt = authority._read_json(receipt_path, require_digest=True)
    receipt["schema"] = "SUBSTRATE_ODYSSEY_TEST_OBSERVATION/v1"
    receipt.pop("sha256")
    receipt["sha256"] = authority.digest(receipt)
    receipt_path.unlink()
    _write(receipt_path, receipt)
    receipt_ref["sha256"] = authority.file_digest(receipt_path)
    with pytest.raises(authority.Refused, match="does not identify the exact inactive G10 observation"):
        authority._gate_specific_checks(root, "G10", g10, frozen)

    g10 = json.loads(json.dumps(subject("G10")))
    receipt_ref = g10["isolation_receipts"]["candidate_evaluator_read_denied"]
    receipt_path = root / receipt_ref["path"]
    receipt = authority._read_json(receipt_path, require_digest=True)
    receipt["attempted"] = False
    receipt.pop("sha256")
    receipt["sha256"] = authority.digest(receipt)
    receipt_path.unlink()
    _write(receipt_path, receipt)
    receipt_ref["sha256"] = authority.file_digest(receipt_path)
    with pytest.raises(authority.Refused, match="denial was not attempted"):
        authority._gate_specific_checks(root, "G10", g10, frozen)

    g11 = json.loads(json.dumps(subject("G11")))
    g11["score_weights"]["task_utility"] = 0.0
    with pytest.raises(authority.Refused, match="finite positive number"):
        authority._gate_specific_checks(root, "G11", g11, frozen)


def test_human_gate_cross_bindings_reject_subjects_from_different_selections(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    subjects = {
        gate_id: authority._read_json(root / "receipts" / f"{gate_id}.subject.json", require_digest=True)
        for gate_id in ("G02", "G03", "G04", "G05", "G10", "G11")
    }
    subjects["G04"]["frontiers"][0]["candidate_manifest_sha256"] = "0" * 64
    with pytest.raises(authority.Refused, match="not bound to the validated G03 manifest"):
        authority._validate_human_cross_gate_bindings(subjects)

    subjects = {
        gate_id: authority._read_json(root / "receipts" / f"{gate_id}.subject.json", require_digest=True)
        for gate_id in ("G02", "G03", "G04", "G05", "G10", "G11")
    }
    subjects["G10"]["roots"]["candidate_visible_root"] = "different-candidate-root"
    with pytest.raises(authority.Refused, match="exact G04 custody topology"):
        authority._validate_human_cross_gate_bindings(subjects)


def test_human_evidence_pack_is_fill_only_and_current_frozen_validation_is_read_only(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    frozen_path = root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json"
    before = frozen_path.read_bytes()
    frozen = authority.validate_current_frozen_build(root)
    assert frozen["sha256"]
    assert frozen_path.read_bytes() == before

    pack = authority.human_evidence_pack(root)
    assert pack["status"] == "template_unsealed"
    assert pack["activation"] is False
    assert pack["never_a_gate_receipt"] is True
    assert pack["never_an_attestation"] is True
    assert pack["custody_independence"] == "single_operator"
    assert pack["custody_limitations"]
    assert set(pack["subjects"]) == {"G02", "G04", "G05", "G10", "G11"}
    assert pack["subjects"]["G02"]["status"] == "generate_via_odyssey_machine_subjects"
    assert pack["gate_wrappers"] == {}
    assert pack["subjects"]["G04"]["custody_independence"] == "single_operator"
    assert set(pack["supporting_observation_templates"]["G10"]) == set(authority.ISOLATION_OBSERVATION_EXPECTATIONS)

    output = root / "templates" / "human-evidence-pack.json"
    assert authority.main(["human-evidence-pack", "--root", str(root), "--out", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["never_a_gate_receipt"] is True
    assert written["gate_wrappers"] == {}
    assert written["custody_independence"] == "single_operator"


def test_clean_clone_gate_requires_current_commit_and_exact_frozen_maps(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    subject = authority._read_json(root / "receipts/G13.subject.json", require_digest=True)
    subject["source_commit"] = "stale-fixture-head"
    subject.pop("sha256")
    subject["sha256"] = authority.digest(subject)
    with pytest.raises(authority.Refused, match="current git HEAD"):
        authority._gate_specific_checks(root, "G13", subject, frozen)

    subject["source_commit"] = "fixture-head"
    subject["implementation_sha256"] = {}
    subject.pop("sha256")
    subject["sha256"] = authority.digest(subject)
    with pytest.raises(authority.Refused, match="implementation source map drifted"):
        authority._gate_specific_checks(root, "G13", subject, frozen)


def test_machine_gate_sealer_only_admits_an_exact_machine_receipt(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    subject = root / "receipts/G13.subject.json"
    output = root / "receipts/G13.gate.sealed.json"

    gate = authority.seal_machine_gate(root, "G13", subject, output)

    assert gate["status"] == "pass"
    assert gate["gate_id"] == "G13"
    assert authority._read_json(output, require_digest=True)["sha256"] == gate["sha256"]
    # Converted G02 is machine_verified and seals from its own subject.
    g02_gate = authority.seal_machine_gate(
        root, "G02", root / "receipts/G02.subject.json", root / "receipts/G02.gate.sealed.json"
    )
    assert g02_gate["gate_id"] == "G02"
    assert g02_gate["human_attestation"] is None
    with pytest.raises(authority.Refused, match="machine-verified|subject schema"):
        authority.seal_machine_gate(root, "G02", subject, root / "receipts/G02.wrong.sealed.json")


def test_protocol_digest_receipt_is_derived_from_the_exact_frozen_build(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    output = root / "receipts/G15.protocol-digests.json"

    subject = authority.emit_protocol_digests(root, output)

    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    assert subject["source_digest"] == authority.source_digest_for_frozen(frozen)
    assert subject["protocol_digest"] == authority.protocol_digest_for_frozen(frozen)
    subject["protocol_digest"] = "0" * 64
    subject.pop("sha256")
    subject["sha256"] = authority.digest(subject)
    with pytest.raises(authority.Refused, match="protocol digest does not match"):
        authority._gate_specific_checks(root, "G15", subject, frozen)


def test_machine_gate_inventory_includes_only_valid_current_receipts(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    authority.seal_machine_gate(
        root,
        "G13",
        root / "receipts/G13.subject.json",
        root / authority.MACHINE_GATE_EVIDENCE / "G13.valid.json",
    )

    assert authority.machine_gate_ids(root) == frozenset({"G13"})


def test_storage_is_live_and_reserve_is_not_silently_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fixture_root(tmp_path)
    model_reserve = 120_000_000_000
    draft, _ = _prepared_inputs(root, model_reserve=model_reserve)
    # Enough for the base safety floor, but not for a declared active model
    # reservation.  The preflight must be a clean refusal, not a hidden cap cut.
    minimum = authority.BASE_PROTECTED_FLOOR_BYTES + 32 * 1024**2
    monkeypatch.setattr(authority.shutil, "disk_usage", lambda _path: Usage(500 * 1024**3, 0, minimum))
    inputs = root / "authority-inputs.sealed.json"
    authority.seal_inputs(root, draft, inputs)
    receipt = authority.preflight(root, inputs, root / "preflight.json")
    assert receipt["preflight_admitted"] is False
    assert receipt["storage"]["free_bytes_now"] == minimum
    assert receipt["storage"]["measured_required_free_bytes"] > minimum
    with pytest.raises(authority.Refused, match="live storage"):
        authority.seal(root, inputs, root / "preflight.json", root / "authority.json")


def test_seal_rechecks_subjects_after_preflight(tmp_path: Path, large_volume: None) -> None:
    root = _fixture_root(tmp_path)
    draft, _ = _prepared_inputs(root)
    inputs = root / "authority-inputs.sealed.json"
    authority.seal_inputs(root, draft, inputs)
    preflight = root / "preflight.json"
    authority.preflight(root, inputs, preflight)
    subject_path = root / "receipts/G12.subject.json"
    subject = authority._read_json(subject_path, require_digest=True)
    subject["survivors"] = ["escaped-mutation"]
    subject.pop("sha256")
    subject["sha256"] = authority.digest(subject)
    subject_path.unlink()
    _write(subject_path, subject)
    with pytest.raises(authority.Refused, match="subject file is missing or drifted"):
        authority.seal(root, inputs, preflight, root / "authority.json")


def test_machine_gate_validators_reject_shallow_or_spoofed_workloads(tmp_path: Path) -> None:
    """Each strengthened gate must reject more than a forged ``all_pass`` flag."""
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)

    def subject(gate_id: str) -> dict[str, Any]:
        return authority._read_json(root / "receipts" / f"{gate_id}.subject.json", require_digest=True)

    g03 = json.loads(json.dumps(subject("G03")))
    g03["manifests"][0]["task_count"] = 1
    with pytest.raises(authority.Refused, match="wrong task count"):
        authority._gate_specific_checks(root, "G03", g03, frozen)

    # Launch gate: G06-DC must reject a synthetic workload pretending to pass.
    g06_dc = json.loads(json.dumps(subject("G06-DC")))
    g06_dc["synthetic_workload"] = True
    g06_dc["workload_class"] = "synthetic"
    with pytest.raises(authority.Refused, match="synthetic|tool-bearing|workload"):
        authority._gate_specific_checks(root, "G06-DC", g06_dc, frozen)

    # Preserved historical G06 validator still rejects resource-parity spoofs.
    g06 = json.loads(json.dumps(subject("G06")))
    g06["observations"][0]["cells"][0]["resource_parity"]["control"]["compute_ceiling"] = 99
    with pytest.raises(authority.Refused, match="resource parity"):
        authority._gate_specific_checks(root, "G06", g06, frozen)

    g07 = json.loads(json.dumps(subject("G07")))
    g07["p95_private_growth_bytes"] = 1
    with pytest.raises(authority.Refused, match="growth/transient summary"):
        authority._gate_specific_checks(root, "G07", g07, frozen)

    g07 = json.loads(json.dumps(subject("G07")))
    g07["private_write_cap_bytes"] = (
        g07["observed_free_before_bytes"] - g07["runtime_required_free_bytes"] + 1
    )
    with pytest.raises(authority.Refused, match="live dynamic capacity"):
        authority._gate_specific_checks(root, "G07", g07, frozen)

    g08 = json.loads(json.dumps(subject("G08")))
    g08["observations"][3]["decision"] = "admit_or_resume"
    with pytest.raises(authority.Refused, match="p1_pause_boundary"):
        authority._gate_specific_checks(root, "G08", g08, frozen)

    g08 = json.loads(json.dumps(subject("G08")))
    # The renderer has a diagnostic mirror, but it is not the executable
    # admission path.  A canary bound only to it must not pass after a worker
    # broker change.
    g08["broker_source_sha256"] = frozen["implementation_sha256"]["frontier_renderer"]
    with pytest.raises(authority.Refused, match="broker source"):
        authority._gate_specific_checks(root, "G08", g08, frozen)

    g09 = json.loads(json.dumps(subject("G09")))
    g09["rehearsals"][0]["arms"]["control"]["writer_lock"] = g09["rehearsals"][0]["arms"]["candidate"]["writer_lock"]
    with pytest.raises(authority.Refused, match="writer lock"):
        authority._gate_specific_checks(root, "G09", g09, frozen)

    g12 = json.loads(json.dumps(subject("G12")))
    g12["mutations"] = g12["mutations"][:1]
    with pytest.raises(authority.Refused, match="lacks required live attack coverage"):
        authority._gate_specific_checks(root, "G12", g12, frozen)


def test_historical_g06_validator_rejects_shallow_or_spoofed_width_subjects(tmp_path: Path) -> None:
    """Preserved ``_validate_g06`` still refuses generic or forged width subjects.

    G06 is no longer a launch gate (G06-DC holds that slot), but its validator
    and 1.35 simultaneity limit remain so the historical receipt stays honest.
    """
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    subject = authority._read_json(root / "receipts/G06.subject.json", require_digest=True)

    wrong_model = json.loads(json.dumps(subject))
    for arm in ("candidate", "control"):
        wrong_model["observations"][0]["cells"][0]["resource_parity"][arm]["models"] = ["wrong-body"]
    with pytest.raises(authority.Refused, match="selected G02 base model"):
        authority._gate_specific_checks(root, "G06", wrong_model, frozen)

    nonfinite = json.loads(json.dumps(subject))
    nonfinite["observations"][0]["metrics"]["per_cell_slowdown_ratio"] = float("nan")
    with pytest.raises(authority.Refused, match="must be a number"):
        authority._gate_specific_checks(root, "G06", nonfinite, frozen)

    receipt_root = _fixture_root(tmp_path / "receipt")
    _prepared_inputs(receipt_root)
    receipt_frozen = authority._read_json(
        receipt_root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    forged = authority._read_json(receipt_root / "receipts/G06.subject.json", require_digest=True)
    receipt_ref = forged["observations"][0]["cells"][0]["candidate_receipt"]
    receipt_path = receipt_root / receipt_ref["path"]
    receipt = authority._read_json(receipt_path, require_digest=True)
    receipt["role"] = "control"
    receipt.pop("sha256")
    receipt["sha256"] = authority.digest(receipt)
    receipt_path.unlink()
    _write(receipt_path, receipt)
    receipt_ref["sha256"] = authority.file_digest(receipt_path)
    with pytest.raises(authority.Refused, match="real G06 candidate dispatch"):
        authority._gate_specific_checks(receipt_root, "G06", forged, receipt_frozen)

    boundary_root = _fixture_root(tmp_path / "boundary")
    _prepared_inputs(boundary_root)
    boundary_frozen = authority._read_json(
        boundary_root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    broken_boundary = authority._read_json(boundary_root / "receipts/G06.subject.json", require_digest=True)
    boundary_ref = broken_boundary["observations"][0]["metrics"]["phase_boundary_receipt"]
    trace_path = boundary_root / boundary_ref["path"]
    trace_path = trace_path.parent / "EVENTS.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(authority.Refused, match="trace"):
        authority._gate_specific_checks(boundary_root, "G06", broken_boundary, boundary_frozen)


def test_historical_g06_receipt_remains_verifiable_through_preserved_validator(tmp_path: Path) -> None:
    """A structurally complete historical G06 subject still passes ``_validate_g06``."""
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    subject = authority._read_json(root / "receipts/G06.subject.json", require_digest=True)

    assert "G06" not in authority.GATE_SPECS
    assert "G06-DC" in authority.GATE_SPECS
    # Direct preserved-validator path (not only the dispatch table).
    authority._validate_g06(root, subject, frozen)
    authority._gate_specific_checks(root, "G06", subject, frozen)


def test_g09_structurally_valid_recovery_subject_passes_once_per_arm(tmp_path: Path) -> None:
    """A genuine candidate/control rehearsal remains admissible as-is."""
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    g09 = authority._read_json(root / "receipts/G09.subject.json", require_digest=True)

    authority._gate_specific_checks(root, "G09", g09, frozen)


def test_worker_frontier_cannot_substitute_another_valid_g03_manifest(tmp_path: Path, large_volume: None) -> None:
    """A valid B manifest cannot be silently substituted into worker A."""
    root = _fixture_root(tmp_path)
    draft_path, draft = _prepared_inputs(root)
    g03 = authority._read_json(root / "receipts/G03.subject.json", require_digest=True)
    substituted = dict(g03["manifests"][1])
    worker_a = draft["worker"]["frontiers"][0]
    worker_a["candidate_manifest"] = substituted["path"]
    worker_a["candidate_manifest_sha256"] = substituted["file_sha256"]
    worker_a["candidate_manifest_binding"] = substituted
    authority._write_json(draft_path, draft, overwrite=True)

    with pytest.raises(authority.Refused, match="does not exactly equal validated G03"):
        authority.seal_inputs(root, draft_path, root / "authority-inputs.sealed.json")


def test_g03_authority_accepts_the_source_bound_materializer_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real materializer's candidate-only receipt matches the gate contract."""
    install_librispeech_audio_fixture(monkeypatch)
    root = _fixture_root(tmp_path)
    frozen = authority._read_json(root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    frontiers = []
    for frontier in authority.FRONTIER_IDS:
        asset = root / "inputs" / "assets" / f"{frontier}.json"
        rights = root / "inputs" / "rights" / f"{frontier}.json"
        _write(asset, _sealed("SUBSTRATE_ODYSSEY_TEST_SOURCE/v1", {"frontier": frontier}))
        _write(rights, _sealed("SUBSTRATE_ODYSSEY_TEST_RIGHTS/v1", {"frontier": frontier}))
        frontiers.append(
            {
                "id": frontier,
                "assets": [
                    {
                        "path": str(asset.relative_to(root)),
                        "sha256": authority.file_digest(asset),
                        "role": "fixture_corpus",
                        "rights_reference": str(rights.relative_to(root)),
                    }
                ],
            }
        )
    selection = {
        "schema": "SUBSTRATE_ODYSSEY_SOURCE_SELECTION/v1",
        "program": authority.PROGRAM,
        "status": "sealed",
        "frontiers": frontiers,
        "activation": False,
    }
    selection["sha256"] = authority.digest(selection)
    subject, _artifacts = materializer.build_manifest_set(
        root,
        selection=selection,
        seed_bytes=b"fixture-custodian-seed",
        seed_provenance="operator_supplied",
        candidate_root=root / "candidate-visible",
        evaluator_root=root / "evaluator-only",
        frozen=frozen,
        source_commit="fixture-head",
    )

    authority._gate_specific_checks(root, "G03", subject, frozen)
