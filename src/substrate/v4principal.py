"""Frozen principal, replication, and open-world execution for Substrate v4."""

from __future__ import annotations

import hashlib
import json
import resource
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from substrate import v4config as C
from substrate import v4fabric as F
from substrate import v4io as io
from substrate.evidence import canonical_current_path
from substrate.runtime import StructuralSubstrate

READY_TAG = "substrate-v4-structural-ready"
PRINCIPAL = io.RUNS / "principal"
UNITS = PRINCIPAL / "units"
CHECKPOINTS = PRINCIPAL / "checkpoints"
MANIFEST = io.CONFIGS / "principal_manifest.json"
EPISODES_PER_PHASE = 5
FROZEN_INPUTS = (
    "configs/substrate/v4/frozen_configuration.json",
    "configs/substrate/v4/split_manifest.json",
    "evidence/substrate/v4/SUBSTRATE_V4_ADMISSION.json",
)


def _input_digests() -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            relative,
            hashlib.sha256(current.read_bytes()).hexdigest() if current.is_file() else "missing",
        )
        for relative in FROZEN_INPUTS
        for current in (canonical_current_path(io.ROOT, relative),)
    )


@dataclass(frozen=True)
class WorkUnit:
    identity: str
    hypothesis: str
    arm: str
    history_seed: int
    split: str
    latent_model_family: str
    surface_representation_family: str
    developmental_phases: tuple[str, ...]
    body: str
    shard: int
    dependencies: tuple[str, ...]
    inputs: tuple[str, ...]
    input_digests: tuple[tuple[str, str], ...]
    outputs: tuple[str, ...]
    resource_class: str
    timeout: int
    retry_rule: str
    checkpoint: str
    artifact_family: str
    claim_ceiling: str


def _unit(seed: int, arm: str, split: str, shard: int, body: str = "general") -> WorkUnit:
    identity = f"{split}-{seed}-{arm}-{body}-shard{shard}"
    orientation = tuple(F.ORIENTATIONS)[(seed + shard + (1 if split == "open_world_review" else 0)) % len(F.ORIENTATIONS)]
    return WorkUnit(
        identity=identity,
        hypothesis=",".join(C.HYPOTHESES),
        arm=arm,
        history_seed=seed,
        split=split,
        latent_model_family=(f"held_out_asymmetric_causal_tree_{orientation}" if split == "open_world_review" else f"asymmetric_causal_tree_{orientation}"),
        surface_representation_family=C.REPRESENTATIONS[(seed + shard) % len(C.REPRESENTATIONS)],
        developmental_phases=C.PHASES,
        body=body,
        shard=shard,
        dependencies=(),
        inputs=FROZEN_INPUTS,
        input_digests=_input_digests(),
        outputs=(
            f"runs/substrate/v4/principal/units/{identity}.json",
            f"runs/substrate/v4/principal/checkpoints/{identity}.json",
        ),
        resource_class="small_cpu",
        timeout=300,
        retry_rule=C.STATISTICS["retry_rule"],
        checkpoint="exact interruption restore and compact terminal state identity",
        artifact_family=f"{split}:{identity}",
        claim_ceiling=C.CLAIM_BOUNDARY["maximum"],
    )


def work_units() -> tuple[WorkUnit, ...]:
    units = []
    for seed in C.SPLITS["principal"]:
        for arm in C.CORE_ARMS:
            for shard in range(3):
                units.append(_unit(seed, arm, "principal", shard))
    for seed in C.SPLITS["replication"]:
        for arm in (
            "full_v4",
            "semantic_retrieval_control",
            "no_counterfactual",
            "no_alignment",
            "simple_structural_inquiry",
            "more_compute",
        ):
            units.append(_unit(seed, arm, "replication", 0, body="compact"))
    for seed in C.SPLITS["open_world_review"]:
        for arm in ("full_v4", "no_alignment", "more_compute", "transcript_replay"):
            units.append(_unit(seed, arm, "open_world_review", 0, body="tool_dominant"))
    return tuple(units)


PHASE_FAMILY = {
    "phase_0_cold_baseline": "dynamic_transition_systems",
    "phase_1_observational_structure_acquisition": "dynamic_transition_systems",
    "phase_2_competing_structural_hypotheses": "causal_systems",
    "phase_3_discriminating_inquiry": "structural_scientific_inquiry",
    "phase_4_causal_intervention": "causal_systems",
    "phase_5_model_revision": "ontology_structure_conflict",
    "phase_6_first_cross_representation_encounter": "cross_representation_isomorphisms",
    "phase_7_counterfactual_challenge": "counterfactual_planning",
    "phase_8_explanation_and_falsifier": "mechanism_diagnosis",
    "phase_9_conflicting_structural_history": "causal_systems",
    "phase_10_interruption_and_exact_restoration": "integrated_interrupted_development",
    "phase_11_body_or_tool_change": "integrated_interrupted_development",
    "phase_12_return_to_prior_structural_domain": "causal_systems",
    "phase_13_negative_alignment_trap": "cross_representation_isomorphisms",
    "phase_14_useful_history_specialization": "cross_representation_isomorphisms",
    "phase_15_generator_held_out_open_world": "integrated_interrupted_development",
    "phase_16_terminal_integrated_evaluation": "integrated_interrupted_development",
}


def _original_variant(seed: int, split: str) -> str:
    offset = 1 if split == "open_world_review" else 0
    return tuple(F.ORIENTATIONS)[(seed + offset) % len(F.ORIENTATIONS)]


def _revision_variant(seed: int, split: str) -> str:
    original = _original_variant(seed, split)
    return "C" if original != "C" else "D"


def _phase_task(unit: WorkUnit, phase_index: int, local_index: int):
    phase = C.PHASES[phase_index]
    family = PHASE_FAMILY[phase]
    original = _original_variant(unit.history_seed, unit.split)
    revised = _revision_variant(unit.history_seed, unit.split)
    representation = None
    include_training = True
    variant = original
    if phase_index == 0:
        include_training = False
    elif phase_index == 5:
        # The frozen generator converts the original orientation into its declared
        # C-or-D revision when the phase name contains ``model_revision``.
        variant = original
    elif phase_index in {6, 7, 8}:
        variant = revised
        include_training = family != "cross_representation_isomorphisms"
    elif phase_index in {9, 10, 11}:
        variant = "B"
    elif phase_index == 12:
        variant = original
    elif phase_index == 13:
        variant = "D" if original != "D" else "C"
    elif phase_index == 14:
        variant = original
        include_training = False
    elif phase_index >= 15:
        variant = "D" if unit.split == "open_world_review" else original
    if family == "cross_representation_isomorphisms":
        representation_offset = 3
        if phase_index == 13:
            representation_offset = 4
        elif phase_index == 14:
            representation_offset = 5
        representation = C.REPRESENTATIONS[(unit.history_seed + representation_offset + unit.shard) % len(C.REPRESENTATIONS)]
    index = unit.shard * 100_000 + phase_index * 100 + local_index
    return F.generate_task(
        unit.history_seed,
        family,
        index,
        unit.split,
        phase=phase,
        representation=representation,
        include_training=include_training,
        history_variant=variant,
    )


def _history_probe(seed: int, split: str) -> dict:
    source_representation = C.REPRESENTATIONS[seed % len(C.REPRESENTATIONS)]
    target_representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)]
    entities = {
        "A": StructuralSubstrate(entity_id=f"history:{seed}:A"),
        "B": StructuralSubstrate(entity_id=f"history:{seed}:B"),
        "replica": StructuralSubstrate(entity_id=f"history:{seed}:A"),
        "mixed": StructuralSubstrate(entity_id=f"history:{seed}:mixed"),
    }
    task_a = F.generate_task(
        seed,
        "causal_systems",
        900_000,
        split,
        phase="history_A",
        representation=source_representation,
        include_training=True,
        history_variant="A",
    )
    task_b = F.generate_task(
        seed,
        "causal_systems",
        900_001,
        split,
        phase="history_B",
        representation=source_representation,
        include_training=True,
        history_variant="B",
    )
    for name in ("A", "replica", "mixed"):
        entities[name].step_structural(task_a)
    for name in ("B", "mixed"):
        entities[name].step_structural(task_b)
    probes = {}
    for label, index in (("A", 900_010), ("B", 900_011)):
        probes[label] = F.generate_task(
            seed,
            "cross_representation_isomorphisms",
            index,
            split,
            phase=f"history_probe_{label}",
            representation=target_representation,
            include_training=False,
            history_variant=label,
        )
    a_own = entities["A"].step_structural(probes["A"], learn=False)
    entities["replica"].step_structural(probes["A"], learn=False)
    a_wrong = entities["B"].step_structural(probes["A"], learn=False)
    b_own = entities["B"].step_structural(probes["B"], learn=False)
    b_wrong = entities["A"].step_structural(probes["B"], learn=False)
    entities["replica"].step_structural(probes["B"], learn=False)
    a_models = entities["A"].checkpoint()["extension"]["structural_world"]["models"]
    replica_models = entities["replica"].checkpoint()["extension"]["structural_world"]["models"]
    effects = [
        float(a_own["outcome"]["correct"]) - float(a_wrong["outcome"]["correct"]),
        float(b_own["outcome"]["correct"]) - float(b_wrong["outcome"]["correct"]),
    ]
    return {
        "identical_histories_equivalent": a_models == replica_models,
        "different_histories_diverge": sorted(entities["A"].structural_world.models) != sorted(entities["B"].structural_world.models),
        "matched_A": float(a_own["outcome"]["correct"]),
        "wrong_A": float(a_wrong["outcome"]["correct"]),
        "matched_B": float(b_own["outcome"]["correct"]),
        "wrong_B": float(b_wrong["outcome"]["correct"]),
        "mean_specialization_margin": statistics.fmean(effects),
        "wrong_history_clean": all(effect >= 0 for effect in effects),
        "mixed_alternatives_preserved": len(entities["mixed"].structural_world.models) >= 2,
        "activation": False,
    }


def _compact_cycle(row: dict) -> dict:
    return {
        "identity": row["identity"],
        "family": row["family"],
        "phase": row["phase"],
        "representation": row["representation"],
        "query_kind": row["query_kind"],
        "decision": row["decision"],
        "target": row["outcome"]["target"],
        "correct": bool(row["outcome"]["correct"]),
        "revealed_after_commitment": row["outcome"]["revealed_after_commitment"],
        "model": row["structural_execution"].get("model"),
        "model_status": row["structural_execution"].get("model_status"),
        "operation": row["structural_execution"].get("operation"),
        "causally_active": bool(row["structural_execution"].get("causally_active", False)),
        "compute": float(row["compute"]),
        "self_prediction_step": row["self_prediction_step"],
        "outcome_step": row["outcome_step"],
        "body": row["body"],
        "activation": False,
    }


def execute_unit(unit: WorkUnit) -> dict:
    started = time.perf_counter()
    entity = StructuralSubstrate(unit.arm, entity_id=f"v4:{unit.history_seed}:{unit.arm}", body=unit.body)
    cycles = []
    phases = []
    checkpoint_exact = True
    body_continuity = True
    for phase_index, phase in enumerate(C.PHASES):
        if unit.arm == "fresh_reset" and phase_index:
            entity = StructuralSubstrate(unit.arm, entity_id=f"v4:{unit.history_seed}:{unit.arm}", body=unit.body)
        phase_rows = []
        for local_index in range(EPISODES_PER_PHASE):
            row = entity.step_structural(
                _phase_task(unit, phase_index, local_index),
                learn=unit.arm != "transcript_replay",
            )
            compact = _compact_cycle(row)
            cycles.append(compact)
            phase_rows.append(compact)
        if phase == "phase_10_interruption_and_exact_restoration":
            checkpoint = entity.checkpoint()
            restored = StructuralSubstrate(unit.arm).restore(checkpoint)
            checkpoint_exact = checkpoint_exact and restored.checkpoint()["identity"] == checkpoint["identity"]
            entity = restored
        if phase == "phase_11_body_or_tool_change":
            transition = entity.change_body(
                "tool_dominant" if unit.body == "general" else unit.body,
                ["sandbox_simulation", "structural_inspector", "counterfactual_runner"],
            )
            body_continuity = body_continuity and transition["owned_identity_preserved"]
        phases.append(
            {
                "phase": phase,
                "family": PHASE_FAMILY[phase],
                "episodes": len(phase_rows),
                "accuracy": statistics.fmean(float(row["correct"]) for row in phase_rows),
                "utility": statistics.fmean(float(row["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in phase_rows),
                "causally_active_rate": statistics.fmean(float(row["causally_active"]) for row in phase_rows),
            }
        )
    checkpoint = entity.checkpoint()
    restored = StructuralSubstrate(unit.arm).restore(checkpoint)
    checkpoint_exact = checkpoint_exact and restored.checkpoint()["identity"] == checkpoint["identity"]
    structural = entity.structural_world
    history = (
        _history_probe(unit.history_seed, unit.split) if unit.arm == "full_v4" and unit.shard == 0 and unit.split in {"principal", "replication"} else None
    )
    summary = {
        "episodes": len(cycles),
        "accuracy": statistics.fmean(float(row["correct"]) for row in cycles),
        "utility": statistics.fmean(float(row["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in cycles),
        "compute": sum(row["compute"] for row in cycles),
        "causally_active_rate": statistics.fmean(float(row["causally_active"]) for row in cycles),
        "models": len(structural.models),
        "alternatives": sum(len(model.alternatives) for model in structural.models.values()),
        "revisions": len(structural.revisions),
        "interventions": len(structural.interventions),
        "counterfactuals": len(structural.counterfactuals),
        "mappings": len(structural.mappings),
        "inquiries": len(structural.inquiries),
        "checkpoint_exact": checkpoint_exact,
        "body_continuity": body_continuity,
        "owned_identity": entity.entity_id,
        "state_identity": checkpoint["identity"],
        "structural_state_digest": io.sha_obj(checkpoint["extension"]["structural_world"]),
        "history_specialization": history,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "activation": False,
    }
    receipt = {
        "schema": "substrate-v4-principal-unit/v1",
        "unit": json.loads(json.dumps(asdict(unit))),
        "cycles": cycles,
        "phases": phases,
        "summary": summary,
        "activation": False,
    }
    receipt["receipt_identity"] = io.sha_obj(receipt)
    return receipt


def checkpoint_receipt(receipt: dict) -> dict:
    summary = receipt["summary"]
    return {
        "schema": "substrate-v4-compact-checkpoint/v1",
        "unit_identity": receipt["unit"]["identity"],
        "state_identity": summary["state_identity"],
        "structural_state_digest": summary["structural_state_digest"],
        "checkpoint_exact": summary["checkpoint_exact"],
        "body_continuity": summary["body_continuity"],
        "activation": False,
    }


def validate_receipt(receipt: dict, unit: WorkUnit) -> bool:
    expected = io.sha_obj({key: value for key, value in receipt.items() if key not in {"receipt_identity", "program"}})
    expected_unit = json.loads(json.dumps(asdict(unit)))
    return (
        receipt.get("receipt_identity") == expected
        and receipt.get("activation") is False
        and receipt.get("unit") == expected_unit
        and len(receipt.get("cycles", [])) == len(C.PHASES) * EPISODES_PER_PHASE
        and receipt["summary"]["checkpoint_exact"]
    )


def _write_unit(receipt: dict) -> None:
    identity = receipt["unit"]["identity"]
    io.run_json(f"principal/units/{identity}.json", receipt)
    io.run_json(f"principal/checkpoints/{identity}.json", checkpoint_receipt(receipt))


def _execute_and_write(unit: WorkUnit) -> dict:
    receipt = execute_unit(unit)
    _write_unit(receipt)
    return {
        "identity": unit.identity,
        "valid": validate_receipt(receipt, unit),
        "runtime_seconds": receipt["summary"]["runtime_seconds"],
    }


def prepare() -> dict:
    units = work_units()
    total_episodes = len(units) * len(C.PHASES) * EPISODES_PER_PHASE
    manifest = {
        "schema": "substrate-v4-principal-manifest/v1",
        "units": [asdict(unit) for unit in units],
        "unit_count": len(units),
        "episodes_per_unit": len(C.PHASES) * EPISODES_PER_PHASE,
        "total_episodes": total_episodes,
        "source_digest": io.source_digest(),
        "configuration_digest": C.configuration()["configuration_digest"],
        "content_addressed_inputs": dict(_input_digests()),
        "activation": False,
    }
    principal = {
        "schema": "substrate-v4-principal-authority/v1",
        "minimum_histories": 48,
        "principal_histories": len(C.SPLITS["principal"]),
        "latent_model_families": sorted({unit.latent_model_family for unit in units}),
        "representations": list(C.REPRESENTATIONS),
        "phases": list(C.PHASES),
        "arms": list(C.CORE_ARMS),
        "principal_shards": 3,
        "unit_count": len(units),
        "total_episodes": total_episodes,
        "input_digests_complete": all(digest != "missing" for _, digest in _input_digests()),
        "bounds": {"units": [1500, 6000], "episodes": [150_000, 600_000]},
        "ready_tag": READY_TAG,
        "activation": False,
    }
    dag = {
        "schema": "substrate-v4-principal-dag/v1",
        "nodes": [asdict(unit) for unit in units],
        "edges": [],
        "acyclic": True,
        "complete": True,
        "activation": False,
    }
    resource_plan = {
        "schema": "substrate-v4-resource-plan/v1",
        "selected_workers": 4,
        "incremental_atomic_receipts": True,
        "exact_retry_limit": 1,
        "hawking_observation_only": True,
        "estimated_units": len(units),
        "estimated_episodes": total_episodes,
        "activation": False,
    }
    io.config_json("principal_manifest.json", manifest)
    io.seal("SUBSTRATE_V4_PRINCIPAL_AUTHORITY.json", principal)
    io.seal("SUBSTRATE_V4_PRINCIPAL_DAG.json", dag)
    io.seal("SUBSTRATE_V4_RESOURCE_PLAN.json", resource_plan)
    return {"manifest": manifest, "principal": principal, "dag": dag, "resource_plan": resource_plan}


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=io.ROOT, text=True).strip()


def source_authority() -> dict:
    head = _git("rev-parse", "HEAD")
    ready = _git("rev-parse", f"{READY_TAG}^{{}}")
    admission = io.load("SUBSTRATE_V4_ADMISSION.json")
    manifest = json.loads(MANIFEST.read_text())
    actual_inputs = dict(_input_digests())
    frozen_inputs = manifest.get("content_addressed_inputs", {})
    checks = {
        "head_is_ready_tag": head == ready,
        "worktree_clean": not _git("status", "--porcelain=v1"),
        "admission_passed": admission["admitted"],
        "principal_launch_authorized": admission["principal_launch_authorized"],
        "worker_authority_is_four": io.load("SUBSTRATE_V4_WORKER_AUTHORITY.json")["selected_workers"] == 4,
        "source_digest_exact": manifest.get("source_digest") == io.source_digest(),
        "configuration_digest_exact": manifest.get("configuration_digest") == C.configuration()["configuration_digest"],
        "content_addressed_inputs_exact": frozen_inputs == actual_inputs and all(value != "missing" for value in actual_inputs.values()),
        "activation_false": True,
    }
    document = {
        "schema": "substrate-v4-principal-source-authority/v1",
        "head": head,
        "ready_tag": READY_TAG,
        "ready_commit": ready,
        "source_digest": io.source_digest(),
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }
    io.seal("SUBSTRATE_V4_PRINCIPAL_SOURCE_AUTHORITY.json", document)
    return document


def _existing_valid(unit: WorkUnit) -> bool:
    path = UNITS / f"{unit.identity}.json"
    if not path.is_file():
        return False
    try:
        return validate_receipt(json.loads(path.read_text()), unit)
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def status() -> dict:
    units = work_units()
    expected_by_split = {split: sum(unit.split == split for unit in units) for split in ("principal", "replication", "open_world_review")}
    valid_by_split = {split: sum(unit.split == split and _existing_valid(unit) for unit in units) for split in expected_by_split}
    valid = sum(valid_by_split.values())
    root_complete = (io.EVIDENCE / "SUBSTRATE_V4_V3_ROOT_CAUSE_AUDIT.json").is_file()
    canary_path = io.EVIDENCE / "SUBSTRATE_V4_CHEAP_CANARIES.json"
    try:
        canary_complete = canary_path.is_file() and io.load(canary_path.name).get("all_pass") is True
    except (io.Refused, json.JSONDecodeError):
        canary_complete = False
    admission_path = io.EVIDENCE / "SUBSTRATE_V4_ADMISSION.json"
    try:
        pilot_complete = admission_path.is_file() and io.load(admission_path.name).get("admitted") is True
    except (io.Refused, json.JSONDecodeError):
        pilot_complete = False
    terminal_complete = (io.EVIDENCE / "SUBSTRATE_V4_FINAL_STATE.json").is_file()
    if terminal_complete:
        current_stage = "terminal_classification"
    elif valid_by_split["open_world_review"] == expected_by_split["open_world_review"] and valid == len(units):
        current_stage = "verification"
    elif valid_by_split["replication"] == expected_by_split["replication"] and valid_by_split["principal"] == expected_by_split["principal"]:
        current_stage = "open_world_review"
    elif valid_by_split["principal"] == expected_by_split["principal"]:
        current_stage = "replication"
    elif pilot_complete:
        current_stage = "principal_development"
    elif canary_complete:
        current_stage = "moderate_pilot"
    elif canary_path.is_file():
        current_stage = "cheap_admission"
    elif root_complete:
        current_stage = "mechanism_construction"
    else:
        current_stage = "root_cause_audit"
    ordered_stages = (
        "root_cause_audit",
        "mechanism_construction",
        "cheap_admission",
        "moderate_pilot",
        "principal_development",
        "replication",
        "open_world_review",
        "verification",
        "terminal_classification",
    )
    current_index = ordered_stages.index(current_stage)
    return {
        "schema": "substrate-v4-principal-status/v1",
        "current_stage": current_stage,
        "stages": {
            stage: "current" if index == current_index else ("completed" if index < current_index else "pending") for index, stage in enumerate(ordered_stages)
        },
        "expected": len(units),
        "valid": valid,
        "remaining": len(units) - valid,
        "splits": {
            split: {
                "expected": expected_by_split[split],
                "valid": valid_by_split[split],
                "remaining": expected_by_split[split] - valid_by_split[split],
            }
            for split in expected_by_split
        },
        "complete": valid == len(units),
        "activation": False,
    }


def run(*, workers: int = 4) -> dict:
    authority = source_authority()
    if not authority["all_pass"]:
        raise io.Refused("principal source authority failed")
    if workers != 4:
        raise io.Refused("worker count is frozen at four")
    units = [unit for unit in work_units() if not _existing_valid(unit)]
    failures = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_execute_and_write, unit): unit for unit in units}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                result = future.result(timeout=unit.timeout)
                if not result["valid"]:
                    failures.append({"identity": unit.identity, "error": "receipt validation failed"})
            except Exception as error:
                failures.append({"identity": unit.identity, "error": f"{type(error).__name__}: {error}"})
            if io.STOP.exists():
                for pending in futures:
                    pending.cancel()
                raise io.Refused("operator stop requested")
    final = status()
    report = {
        "schema": "substrate-v4-principal-run/v1",
        "launched_units": len(units),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "status": final,
        "all_pass": final["complete"] and not failures,
        "activation": False,
    }
    io.run_json("principal/run.json", report)
    if not report["all_pass"]:
        raise io.Refused(f"principal execution failed for {len(failures)} units")
    return report
