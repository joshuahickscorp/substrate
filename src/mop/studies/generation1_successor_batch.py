"""Fail-closed Generation 1 successor-study batch authority.

This module does not execute or license a mechanism experiment.  It binds the
independently verified C2 result to four separately gated preregistration
drafts so their implementation can be prepared concurrently without leaking
labels, seeds, or downstream conclusions across study boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT

BATCH_SCHEMA = "mop-generation1-successor-batch/v1"
READINESS_SCHEMA = "mop-generation1-successor-batch-readiness/v1"
BATCH_ID = "generation1-c3-successor-mechanisms-v1"
CLAIM_SCOPE = "generated-c2-successor-scaffolding-only"
STATUS = "draft_unexecuted"

C2_RESULT_PATH = "proof/GENERATION1_CONTEXT_ROUTING.json"
C2_VERIFICATION_PATH = "proof/GENERATION1_CONTEXT_ROUTING.verification.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_self_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = canonical_sha256({key: item for key, item in value.items() if key != field})
    if value.get(field) != expected:
        raise ValueError(f"{label} self-seal is invalid")


def _seed_partition(name: str, start: int, count: int, source: str) -> dict[str, Any]:
    return {"name": name, "start": start, "count": count, "source": source}


def _study(
    *,
    study_id: str,
    canonical_epoch: str,
    question: str,
    dependencies: list[str],
    partitions: list[dict[str, Any]],
    visible_inputs: list[str],
    forbidden_inputs: list[str],
    controls: list[str],
    criteria: dict[str, Any],
    authorities: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "canonical_epoch": canonical_epoch,
        "question": question,
        "dependencies": dependencies,
        "seed_partitions": partitions,
        "visible_inputs": visible_inputs,
        "forbidden_inputs": forbidden_inputs,
        "controls": controls,
        "criteria": criteria,
        "authorities": authorities,
        "implementation_status": "mechanics_and_pilot_adapter_present_confirmatory_adapter_pending",
        "preregistration_status": STATUS,
        "execution_authorized": False,
        "runnable_now": False,
        "blockers": blockers,
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def _studies() -> list[dict[str, Any]]:
    common_blockers = [
        "operator review and freeze of this preregistration draft",
        "confirmatory producer and separately authored independent verifier are not implemented",
    ]
    c3 = _study(
        study_id="G1-C3-D1-LEARNED-DISPATCH",
        canonical_epoch="G1-D1",
        question="Can label-free value-of-computation dispatch exploit the C2 niches?",
        dependencies=["G1-C1", "G1-C2"],
        partitions=[
            _seed_partition("training", 20261001, 512, "frozen_c2_evidence"),
            _seed_partition("calibration", 20261513, 128, "frozen_c2_evidence"),
            _seed_partition("producer_heldout", 20270001, 512, "fresh_generated"),
            _seed_partition("independent_verifier", 20271001, 512, "fresh_generated"),
        ],
        visible_inputs=["latent_vector", "difficulty_index"],
        forbidden_inputs=[
            "context_id",
            "truth",
            "actor_predictions",
            "oracle_actor_id",
            "heldout_route_labels",
        ],
        controls=[
            "global_static_actor",
            "per_difficulty_static_actor",
            "deterministic_random_actor",
            "fixed_c2_context_route_nonpromotable",
            "all_five_actors_fully_charged",
            "equal_compute_homogeneous",
            "per_example_oracle_nonpromotable",
        ],
        criteria={
            "primary_endpoint": "heldout_accuracy_after_full_router_and_actor_cost",
            "minimum_mean_advantage_over_each_static_control": 0.01,
            "comparison_interval_lower_bound_must_exceed": 0.0,
            "minimum_favorable_seed_fraction": 0.75,
            "maximum_gap_below_fixed_c2_context_route": 0.02,
            "minimum_work_saving_vs_all_five_actors": 0.70,
            "all_conditions_required": True,
        },
        authorities=[
            "src/mop/mechanisms/niche_dispatch_scaffold.py",
            "src/mop/mechanisms/niche_dispatch_impl.py",
            "src/mop/mechanisms/niche_dispatch_bed.py",
            "src/mop/studies/generation1_context_routing.py",
        ],
        blockers=common_blockers,
    )
    c4 = _study(
        study_id="G1-C3-V1-SELECTIVE-VERIFICATION",
        canonical_epoch="G1-V1",
        question="When does a verifier add enough correction value to repay its charged cost?",
        dependencies=["G1-C1"],
        partitions=[
            _seed_partition("producer_heldout", 20272001, 256, "fresh_generated"),
            _seed_partition("independent_verifier", 20273001, 256, "fresh_generated"),
        ],
        visible_inputs=["primary_prediction", "primary_confidence", "actor_disagreement", "difficulty_index"],
        forbidden_inputs=["truth", "oracle_correction_value", "verifier_future_output"],
        controls=[
            "verify_none",
            "verify_all",
            "random_rate_matched",
            "disagreement_triggered",
            "calibrated_risk_triggered",
            "oracle_value_of_verification_nonpromotable",
        ],
        criteria={
            "primary_endpoint": "corrected_decision_utility_minus_fully_charged_verification_cost",
            "verification_budget_fractions": [0.1, 0.25, 0.5],
            "comparison_interval_lower_bound_must_exceed": 0.0,
            "must_beat": ["verify_none", "verify_all", "random_rate_matched"],
            "minimum_favorable_seed_fraction": 0.75,
            "all_conditions_required": True,
        },
        authorities=[
            "src/mop/mechanisms/messaging_repair_scaffold.py",
            "src/mop/mechanisms/messaging_repair_impl.py",
            "src/mop/mechanisms/messaging_repair_bed.py",
        ],
        blockers=common_blockers,
    )
    c5 = _study(
        study_id="G1-C3-M1-BOUNDED-MESSAGING",
        canonical_epoch="G1-M1",
        question="Can bounded typed messages create causal cooperation between disjoint actors?",
        dependencies=["G1-C2"],
        partitions=[
            _seed_partition("producer_heldout", 20274001, 256, "fresh_generated"),
            _seed_partition("independent_verifier", 20275001, 256, "fresh_generated"),
        ],
        visible_inputs=["local_actor_observation", "typed_bounded_message", "difficulty_index"],
        forbidden_inputs=["other_actor_private_observation", "truth", "oracle_message"],
        controls=[
            "no_message",
            "broadcast_all",
            "random_route_rate_matched",
            "message_shuffle",
            "wrong_message",
            "message_delay",
            "link_lesion",
            "link_restoration",
        ],
        criteria={
            "primary_endpoint": "causal_decision_utility_per_message_byte_after_full_cost",
            "message_byte_budgets": [0, 8, 32, 128, 512],
            "comparison_interval_lower_bound_must_exceed": 0.0,
            "lesion_must_remove_benefit": True,
            "restoration_must_restore_benefit": True,
            "minimum_favorable_seed_fraction": 0.75,
            "all_conditions_required": True,
        },
        authorities=[
            "src/mop/mechanisms/messaging_repair_scaffold.py",
            "src/mop/mechanisms/messaging_repair_impl.py",
            "src/mop/mechanisms/messaging_repair_bed.py",
        ],
        blockers=common_blockers,
    )
    c6 = _study(
        study_id="G1-C3-G1-CONSTRUCTION-SEARCH",
        canonical_epoch="G1-G1",
        question="Can finite charged construction search form a better shadow coalition?",
        dependencies=["G1-C1", "G1-C2"],
        partitions=[
            _seed_partition("producer_heldout", 20276001, 256, "fresh_generated"),
            _seed_partition("independent_verifier", 20277001, 256, "fresh_generated"),
        ],
        visible_inputs=["sealed_actor_competence", "finite_grammar", "charged_candidate_receipts"],
        forbidden_inputs=["heldout_truth", "unsearched_oracle_topology", "unmetered_compute"],
        controls=[
            "fixed_genotype_no_search",
            "same_final_genotype_without_search",
            "random_grammar_rate_matched",
            "greedy_search",
            "spare_capacity",
            "restart",
            "rollback",
            "no_mutation",
        ],
        criteria={
            "primary_endpoint": "heldout_quality_minus_full_search_and_lifecycle_cost",
            "finite_grammar_required": True,
            "pareto_nondominance_required": True,
            "comparison_interval_lower_bound_must_exceed": 0.0,
            "minimum_favorable_seed_fraction": 0.75,
            "topology_activation_remains_prohibited": True,
            "all_conditions_required": True,
        },
        authorities=[
            "src/mop/mechanisms/construction_search_scaffold.py",
            "src/mop/mechanisms/construction_search_impl.py",
            "src/mop/mechanisms/construction_search_bed.py",
        ],
        blockers=common_blockers,
    )
    return [c3, c4, c5, c6]


def build_batch(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    result_path = repo_root / C2_RESULT_PATH
    verification_path = repo_root / C2_VERIFICATION_PATH
    result = _read_object(result_path, "C2 result")
    verification = _read_object(verification_path, "C2 verification")
    prerequisite = {
        "result_path": C2_RESULT_PATH,
        "result_file_sha256": sha256_file(result_path),
        "result_sha256": result.get("result_sha256"),
        "verification_path": C2_VERIFICATION_PATH,
        "verification_file_sha256": sha256_file(verification_path),
        "verification_sha256": verification.get("verification_sha256"),
        "c2_complete": result.get("complete"),
        "c2_ready_to_preregister_learned_dispatch": result.get("decision", {}).get(
            "ready_to_preregister_g1_c3_learned_dispatch"
        ),
        "c2_training_authorized": result.get("decision", {}).get("ready_to_train_dispatcher"),
        "independent_verification_complete": verification.get("verification_complete"),
        "all_cells_reproduced": verification.get("dataset_reproduction", {}).get(
            "all_dataset_and_metric_reproductions_passed"
        ),
        "fresh_actor_canary_passed": verification.get("fresh_actor_canary", {}).get("passed"),
        "all_mutations_rejected": verification.get("mutation_suite", {}).get("all_rejected"),
    }
    core = {
        "schema": BATCH_SCHEMA,
        "batch_id": BATCH_ID,
        "claim_scope": CLAIM_SCOPE,
        "status": STATUS,
        "prerequisite": prerequisite,
        "dependency_policy": {
            "preparation_may_run_in_parallel": True,
            "producer_execution_requires_frozen_study_preregistration": True,
            "independent_verifier_must_use_disjoint_fresh_seeds": True,
            "downstream_results_may_not_satisfy_upstream_dependencies": True,
            "failed_or_null_study_does_not_cancel_independent_siblings": True,
        },
        "execution_strategy": {
            "campaign_name": "G1-C3 successor mechanisms",
            "lane_order": ["G1-D1", "G1-V1", "G1-M1", "G1-G1"],
            "study_scheduling": "sequential_cpu_saturation",
            "within_study_parallelism": True,
            "producer_then_independent_verifier": True,
            "final_synthesis_requires_all_completed_or_null_safe_receipts": True,
            "reason": (
                "All four studies are CPU-bound. One adaptive pool minimizes aggregate wall time "
                "and memory pressure while preserving independent seed and verifier boundaries."
            ),
            "estimated_compute_wall_hours_after_implementation": {
                "host_mostly_idle": {"lower": 3.0, "upper": 6.0},
                "hawking_mostly_active": {"lower": 5.0, "upper": 10.0},
                "confidence": "low_until_32_cell_canaries_complete",
            },
        },
        "shared_authority": {
            "actor_inventory": ["reactive_linear", "mlp", "knn", "prototype", "recurrent_refiner"],
            "difficulty_separations": [0.06, 0.08, 0.1, 0.12, 0.16],
            "dataset": {"n_train": 4000, "n_test": 1200, "n_classes": 10, "dim": 1024},
            "producer_and_verifier_code_must_be_independent": True,
            "adaptive_worker_targets": {"host_idle": 25, "hawking_active": 6},
            "all_compute_and_lifecycle_costs_charged": True,
        },
        "studies": _studies(),
        "execution_authorized": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "batch_sha256": canonical_sha256(core)}


def _all_ranges(batch: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for study in batch.get("studies", []):
        for partition in study.get("seed_partitions", []):
            start = partition.get("start")
            count = partition.get("count")
            if isinstance(start, bool) or not isinstance(start, int) or start <= 0:
                raise ValueError("seed partition start must be a positive integer")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("seed partition count must be a positive integer")
            ranges.append((start, start + count, f"{study.get('study_id')}:{partition.get('name')}"))
    return ranges


def validate_batch(batch: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    if batch.get("schema") != BATCH_SCHEMA or batch.get("batch_id") != BATCH_ID:
        raise ValueError("successor batch identity drifted")
    if batch.get("claim_scope") != CLAIM_SCOPE or batch.get("status") != STATUS:
        raise ValueError("successor batch scope or status escaped")
    _validate_self_seal(batch, "batch_sha256", "successor batch")
    if any(
        batch.get(field) is not False
        for field in (
            "execution_authorized",
            "activation_allowed",
            "scientific_promotion",
        )
    ):
        raise ValueError("successor batch execution, activation, or promotion escaped")

    result_path = repo_root / C2_RESULT_PATH
    verification_path = repo_root / C2_VERIFICATION_PATH
    result = _read_object(result_path, "C2 result")
    verification = _read_object(verification_path, "C2 verification")
    _validate_self_seal(result, "result_sha256", "C2 result")
    _validate_self_seal(verification, "verification_sha256", "C2 verification")
    prerequisite = batch.get("prerequisite", {})
    required_bindings = {
        "result_file_sha256": sha256_file(result_path),
        "result_sha256": result.get("result_sha256"),
        "verification_file_sha256": sha256_file(verification_path),
        "verification_sha256": verification.get("verification_sha256"),
    }
    if any(prerequisite.get(key) != value for key, value in required_bindings.items()):
        raise ValueError("successor batch C2 proof binding drifted")
    if not all(
        prerequisite.get(field) is True
        for field in (
            "c2_complete",
            "c2_ready_to_preregister_learned_dispatch",
            "independent_verification_complete",
            "all_cells_reproduced",
            "fresh_actor_canary_passed",
            "all_mutations_rejected",
        )
    ):
        raise ValueError("successor batch lacks a clean independently verified C2 prerequisite")
    if prerequisite.get("c2_training_authorized") is not False:
        raise ValueError("C2 did not authorize dispatcher training")

    studies = batch.get("studies")
    if not isinstance(studies, list) or len(studies) != 4:
        raise ValueError("successor batch must contain exactly four studies")
    expected = {
        "G1-C3-D1-LEARNED-DISPATCH": ("G1-D1", {"G1-C1", "G1-C2"}),
        "G1-C3-V1-SELECTIVE-VERIFICATION": ("G1-V1", {"G1-C1"}),
        "G1-C3-M1-BOUNDED-MESSAGING": ("G1-M1", {"G1-C2"}),
        "G1-C3-G1-CONSTRUCTION-SEARCH": ("G1-G1", {"G1-C1", "G1-C2"}),
    }
    if {study.get("study_id") for study in studies} != set(expected):
        raise ValueError("successor batch study inventory drifted")
    for study in studies:
        canonical_epoch, dependencies = expected[str(study.get("study_id"))]
        if (
            study.get("canonical_epoch") != canonical_epoch
            or set(study.get("dependencies", [])) != dependencies
        ):
            raise ValueError(f"{study.get('study_id')} dependency or epoch drifted")
        if any(
            study.get(field) is not False
            for field in (
                "execution_authorized",
                "runnable_now",
                "activation_allowed",
                "scientific_promotion",
            )
        ):
            raise ValueError(f"{study.get('study_id')} escaped a fail-closed gate")
        if study.get("preregistration_status") != STATUS or not study.get("blockers"):
            raise ValueError(f"{study.get('study_id')} preregistration state drifted")
        if set(study.get("visible_inputs", [])) & set(study.get("forbidden_inputs", [])):
            raise ValueError(f"{study.get('study_id')} exposes a forbidden input")
        for authority in study.get("authorities", []):
            source = repo_root / authority
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"missing regular-file study authority: {authority}")

    c3 = next(study for study in studies if study["canonical_epoch"] == "G1-D1")
    if set(c3["visible_inputs"]) != {"latent_vector", "difficulty_index"}:
        raise ValueError("C3/D1 visible input boundary drifted")
    required_forbidden = {"context_id", "truth", "actor_predictions", "oracle_actor_id"}
    if not required_forbidden <= set(c3["forbidden_inputs"]):
        raise ValueError("C3/D1 leakage guard drifted")

    ranges = sorted(_all_ranges(batch))
    for prior, current in zip(ranges, ranges[1:], strict=False):
        if current[0] < prior[1]:
            raise ValueError(f"seed partitions overlap: {prior[2]} and {current[2]}")
    c2_start = int(result["config"]["seed_start"])
    c2_end = c2_start + int(result["config"]["seed_count"])
    for start, end, label in ranges:
        source = next(
            partition["source"]
            for study in studies
            for partition in study["seed_partitions"]
            if f"{study['study_id']}:{partition['name']}" == label
        )
        overlaps_c2 = start < c2_end and c2_start < end
        if source == "frozen_c2_evidence" and not (c2_start <= start and end <= c2_end):
            raise ValueError(f"{label} is not contained in frozen C2 evidence")
        if source == "fresh_generated" and overlaps_c2:
            raise ValueError(f"{label} reuses C2 evidence despite being declared fresh")


def build_readiness(batch: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    validate_batch(batch, repo_root=repo_root)
    studies = list(batch["studies"])
    core = {
        "schema": READINESS_SCHEMA,
        "batch_id": BATCH_ID,
        "batch_sha256": batch["batch_sha256"],
        "claim_scope": CLAIM_SCOPE,
        "c2_complete_and_independently_verified": True,
        "study_count": len(studies),
        "study_ids": [study["study_id"] for study in studies],
        "mechanics_scaffolds_validated": True,
        "parallel_preparation_ready": True,
        "execution_ready": False,
        "runnable_study_ids": [],
        "blockers": {study["study_id"]: study["blockers"] for study in studies},
        "interpretation_limit": (
            "This receipt confirms a C2-bound, leakage-checked, disjoint-seed batch draft and "
            "existing mechanics scaffolds only. It does not preregister, execute, activate, or "
            "scientifically promote any successor mechanism."
        ),
        "execution_authorized": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "readiness_sha256": canonical_sha256(core)}


def validate_readiness(readiness: Mapping[str, Any], batch: Mapping[str, Any]) -> None:
    if readiness.get("schema") != READINESS_SCHEMA or readiness.get("batch_id") != BATCH_ID:
        raise ValueError("successor readiness identity drifted")
    _validate_self_seal(readiness, "readiness_sha256", "successor readiness")
    if readiness.get("batch_sha256") != batch.get("batch_sha256"):
        raise ValueError("successor readiness batch binding drifted")
    if readiness.get("execution_ready") is not False or readiness.get("runnable_study_ids") != []:
        raise ValueError("successor readiness falsely authorizes execution")
    if any(
        readiness.get(field) is not False
        for field in (
            "execution_authorized",
            "activation_allowed",
            "scientific_promotion",
        )
    ):
        raise ValueError("successor readiness execution, activation, or promotion escaped")
