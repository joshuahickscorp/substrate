"""Frozen principal, replication, and generator held out execution for Substrate v3."""

from __future__ import annotations

import json
import resource
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from substrate import v2config, v2fabric, v2state
from substrate import v3config as C
from substrate import v3fabric as F
from substrate import v3io as io
from substrate import v3state as S

READY_TAG = "substrate-v3-nous-ready"
PRINCIPAL = io.RUNS / "principal"
UNITS = PRINCIPAL / "units"
CHECKPOINTS = PRINCIPAL / "checkpoints"
MANIFEST = io.CONFIGS / "principal_manifest.json"


@dataclass(frozen=True)
class WorkUnit:
    identity: str
    hypothesis: str
    arm: str
    history_seed: int
    split: str
    workload_families: tuple[str, ...]
    developmental_phases: tuple[str, ...]
    body: str
    shard: int
    dependencies: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    resource_class: str
    timeout: int
    retry_rule: str
    checkpoint: str
    artifact_family: str
    claim_ceiling: str


def _unit(seed: int, arm: str, split: str, shard: int, body: str = "general") -> WorkUnit:
    identity = f"{split}-{seed}-{arm}-{body}-shard{shard}"
    return WorkUnit(
        identity=identity,
        hypothesis=",".join(C.HYPOTHESES),
        arm=arm,
        history_seed=seed,
        split=split,
        workload_families=tuple(C.WORKLOADS),
        developmental_phases=C.PHASES,
        body=body,
        shard=shard,
        dependencies=(),
        inputs=(
            "configs/substrate/v3/frozen_configuration.json",
            "configs/substrate/v3/split_manifest.json",
            "evidence/substrate/v3/SUBSTRATE_V3_ADMISSION.json",
        ),
        outputs=(
            f"runs/substrate/v3/principal/units/{identity}.json",
            f"runs/substrate/v3/principal/checkpoints/{identity}.json",
        ),
        resource_class="small_cpu",
        timeout=300,
        retry_rule=C.STATISTICS["retry_rule"],
        checkpoint="every phase and terminal semantic state",
        artifact_family=f"principal:{identity}",
        claim_ceiling=C.CLAIM_BOUNDARY["maximum"],
    )


def work_units() -> tuple[WorkUnit, ...]:
    units = []
    for seed in C.SPLITS["principal"]:
        for arm in C.CORE_ARMS:
            for shard in range(2):
                units.append(_unit(seed, arm, "principal", shard))
    for seed in C.SPLITS["replication"]:
        for arm in ("full_v3", "v2_developmental_control", "fixed_reasoning", "simple_inquiry"):
            units.append(_unit(seed, arm, "replication", 0, body="compact"))
    for seed in C.SPLITS["open_world_review"]:
        for arm in ("full_v3", "no_understanding_structure"):
            units.append(_unit(seed, arm, "open_world_review", 0, body="tool_dominant"))
    return tuple(units)


PHASE_FAMILY = {
    "phase_0_cold_baseline": "reasoning_method_selection",
    "phase_1_ontology_acquisition": "ontology_garden",
    "phase_2_semantic_procedural_development": "epistemic_laboratory",
    "phase_3_conflicting_evidence_defeaters": "epistemic_laboratory",
    "phase_4_ontology_repair": "ontology_garden",
    "phase_5_cross_representation_transfer": "cross_representation_systems",
    "phase_6_causal_intervention": "causal_micro_worlds",
    "phase_7_counterfactual": "causal_micro_worlds",
    "phase_8_reasoning_method_switching": "reasoning_method_selection",
    "phase_9_inquiry_under_cost": "scientific_inquiry",
    "phase_10_interruption_checkpoint": "epistemic_laboratory",
    "phase_11_body_tool_change": "cross_representation_systems",
    "phase_12_return_prior_domains": "ontology_garden",
    "phase_13_misleading_analogy": "adversarial_ambiguity",
    "phase_14_open_world_structures": "cross_representation_systems",
    "phase_15_terminal_integrated_evaluation": "scientific_inquiry",
}


def _authorized_v2_seed(v3_seed: int) -> int:
    """Map a frozen v3 history onto the immutable v2 generator's authorized seeds."""
    authorized = tuple(seed for split in v2config.SPLITS.values() for seed in split)
    if v3_seed in C.SPLITS["principal"]:
        index = C.SPLITS["principal"].index(v3_seed)
    elif v3_seed in C.SPLITS["replication"]:
        index = len(C.SPLITS["principal"]) + C.SPLITS["replication"].index(v3_seed)
    else:
        raise ValueError(f"v3 history seed {v3_seed} has no v2 preservation authorization")
    return authorized[index % len(authorized)]


def _v2_preservation(seed: int) -> dict:
    v2_seed = _authorized_v2_seed(seed)
    namespace = f"v3_preservation_{seed}"
    entity = v2state.DevelopmentalEntity("full_v2", entity_id=f"v3-v2-preservation:{seed}")
    for index in range(16):
        entity.experience(v2fabric.generate_task(v2_seed, "A", index, namespace), allow_verification=False)
    before = [
        entity.experience(v2fabric.generate_task(v2_seed, "A", 100 + index, namespace), allow_verification=False)
        for index in range(8)
    ]
    for index in range(16):
        entity.experience(v2fabric.generate_task(v2_seed, "B", index, namespace), allow_verification=False)
    after = [
        entity.experience(v2fabric.generate_task(v2_seed, "A", 200 + index, namespace), allow_verification=False)
        for index in range(8)
    ]
    transfer = [
        entity.experience(v2fabric.generate_task(v2_seed, "B", 300 + index, namespace), allow_verification=False)
        for index in range(8)
    ]
    fresh = v2state.DevelopmentalEntity("fresh_control", entity_id=f"v3-v2-fresh:{seed}")
    fresh_rows = [
        fresh.experience(v2fabric.generate_task(v2_seed, "B", 300 + index, namespace), allow_verification=False)
        for index in range(8)
    ]
    checkpoint = entity.checkpoint()
    restored = v2state.DevelopmentalEntity.restore(checkpoint)
    before_accuracy = statistics.fmean(float(row.outcome["correct"]) for row in before)
    after_accuracy = statistics.fmean(float(row.outcome["correct"]) for row in after)
    transfer_accuracy = statistics.fmean(float(row.outcome["correct"]) for row in transfer)
    fresh_accuracy = statistics.fmean(float(row.outcome["correct"]) for row in fresh_rows)
    return {
        "v3_history_seed": seed,
        "authorized_v2_generator_seed": v2_seed,
        "namespace": namespace,
        "retention_change": after_accuracy - before_accuracy,
        "transfer_margin": transfer_accuracy - fresh_accuracy,
        "identity_exact": restored.identity_hash() == checkpoint["identity"],
        "body_continuity": True,
        "activation": False,
    }


def _divergence_probe(seed: int) -> dict:
    ontology_history = S.IntegratedEntity("full_v3", entity_id=f"divergence:{seed}")
    epistemic_history = S.IntegratedEntity("full_v3", entity_id=f"divergence:{seed}")
    ontology_replica = S.IntegratedEntity("full_v3", entity_id=f"divergence:{seed}")
    for index in range(32):
        ontology_task = F.generate_task(seed, "ontology_garden", index, "principal", phase="divergence_history")
        epistemic_task = F.generate_task(seed, "epistemic_laboratory", index, "principal", phase="divergence_history")
        ontology_history.experience(ontology_task)
        ontology_replica.experience(ontology_task)
        epistemic_history.experience(epistemic_task)
    identical = ontology_history.identity_hash() == ontology_replica.identity_hash()
    different = ontology_history.identity_hash() != epistemic_history.identity_hash()
    ontology_tasks = [F.generate_task(seed, "ontology_garden", 100 + index, "principal", phase="divergence_probe") for index in range(16)]
    epistemic_tasks = [F.generate_task(seed, "epistemic_laboratory", 100 + index, "principal", phase="divergence_probe") for index in range(16)]
    ontology_own = statistics.fmean(float(ontology_history.experience(task, learn=False)["outcome"]["correct"]) for task in ontology_tasks)
    ontology_wrong = statistics.fmean(float(epistemic_history.experience(task, learn=False)["outcome"]["correct"]) for task in ontology_tasks)
    epistemic_own = statistics.fmean(float(epistemic_history.experience(task, learn=False)["outcome"]["correct"]) for task in epistemic_tasks)
    epistemic_wrong = statistics.fmean(float(ontology_history.experience(task, learn=False)["outcome"]["correct"]) for task in epistemic_tasks)
    return {
        "identical_histories_equivalent": identical,
        "different_histories_diverge": different,
        "ontology_specialization_margin": ontology_own - ontology_wrong,
        "epistemic_specialization_margin": epistemic_own - epistemic_wrong,
        "mean_specialization_margin": statistics.fmean((ontology_own - ontology_wrong, epistemic_own - epistemic_wrong)),
        "wrong_history_clean": ontology_wrong <= ontology_own and epistemic_wrong <= epistemic_own,
        "activation": False,
    }


def execute_unit(unit: WorkUnit) -> dict:
    start = time.perf_counter()
    entity = S.IntegratedEntity(unit.arm, entity_id=f"v3:{unit.history_seed}:{unit.arm}", body=unit.body)
    cycles = []
    phases = []
    checkpoint_exact = True
    body_continuity = True
    for phase_index, phase in enumerate(C.PHASES):
        if unit.arm == "fresh_reset" and phase_index:
            entity = S.IntegratedEntity(unit.arm, entity_id=f"v3:{unit.history_seed}:{unit.arm}", body=unit.body)
        family = PHASE_FAMILY[phase]
        phase_rows = []
        for local_index in range(8):
            index = unit.shard * 10_000 + phase_index * 100 + local_index
            task = F.generate_task(unit.history_seed, family, index, unit.split, phase=phase)
            row = entity.experience(task, learn=unit.arm != "transcript_replay")
            phase_rows.append(row)
            cycles.append(row)
        if phase == "phase_10_interruption_checkpoint":
            checkpoint = entity.checkpoint()
            entity = S.IntegratedEntity.restore(checkpoint)
            checkpoint_exact = checkpoint_exact and entity.identity_hash() == checkpoint["identity_hash"]
        if phase == "phase_11_body_tool_change":
            transition = entity.change_body(
                "tool_dominant" if unit.body == "general" else unit.body,
                ["deterministic_compare", "sandbox_simulation", "graph_inspector"],
            )
            body_continuity = body_continuity and transition["owned_identity_preserved"]
        phases.append(
            {
                "phase": phase,
                "family": family,
                "episodes": len(phase_rows),
                "accuracy": statistics.fmean(float(row["outcome"]["correct"]) for row in phase_rows),
                "utility": statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in phase_rows),
                "compute": sum(row["compute"] for row in phase_rows),
            }
        )
    checkpoint = entity.checkpoint()
    restored = S.IntegratedEntity.restore(checkpoint)
    checkpoint_exact = checkpoint_exact and restored.identity_hash() == checkpoint["identity_hash"]
    v2_preservation = _v2_preservation(unit.history_seed) if unit.arm == "full_v3" and unit.shard == 0 and unit.split in {"principal", "replication"} else None
    divergence = _divergence_probe(unit.history_seed) if unit.arm == "full_v3" and unit.shard == 0 and unit.split in {"principal", "replication"} else None
    receipt = {
        "schema": "substrate-v3-principal-unit/v1",
        "unit": asdict(unit),
        "cycles": cycles,
        "phases": phases,
        "summary": {
            "episodes": len(cycles),
            "accuracy": statistics.fmean(float(row["outcome"]["correct"]) for row in cycles),
            "utility": statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in cycles),
            "compute": sum(row["compute"] for row in cycles),
            "ontology_revisions": len(entity.ontology_receipts),
            "belief_revisions": len(entity.epistemology.defeater_receipts),
            "knowledge_admissions": sum(row["admitted"] for row in entity.epistemology.knowledge_admissions),
            "defeaters_processed": len(entity.epistemology.defeater_receipts),
            "reasoning_operations": len(entity.reasoning_receipts),
            "inquiry_actions": sum(row["family"] == "scientific_inquiry" and row["decision"] == "inquire" for row in cycles),
            "semantic_records": len(entity.semantic),
            "procedures": len(entity.procedures),
            "checkpoint_exact": checkpoint_exact,
            "body_continuity": body_continuity,
            "owned_identity": entity.entity_id,
            "semantic_identity": checkpoint["identity_hash"],
            "v2_preservation": v2_preservation,
            "divergence": divergence,
            "runtime_seconds": time.perf_counter() - start,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
            "activation": False,
        },
        "checkpoint": checkpoint,
        "activation": False,
    }
    receipt["receipt_identity"] = io.sha_obj({key: value for key, value in receipt.items() if key != "receipt_identity"})
    return receipt


def validate_receipt(receipt: dict, unit: WorkUnit) -> bool:
    expected = io.sha_obj({key: value for key, value in receipt.items() if key not in {"receipt_identity", "program"}})
    return (
        receipt.get("receipt_identity") == expected
        and receipt.get("activation") is False
        and receipt.get("unit", {}).get("identity") == unit.identity
        and receipt.get("summary", {}).get("episodes") == 128
        and receipt.get("summary", {}).get("checkpoint_exact") is True
        and receipt.get("summary", {}).get("body_continuity") is True
    )


def manifest() -> dict:
    units = work_units()
    identities = [unit.identity for unit in units]
    return {
        "schema": "substrate-v3-principal-manifest/v1",
        "ready_tag": READY_TAG,
        "source_digest": io.source_digest(),
        "configuration_digest": C.configuration()["configuration_digest"],
        "units": [asdict(unit) for unit in units],
        "unit_count": len(units),
        "independent_principal_histories": len(C.SPLITS["principal"]),
        "principal_units": sum(unit.split == "principal" for unit in units),
        "replication_units": sum(unit.split == "replication" for unit in units),
        "open_world_units": sum(unit.split == "open_world_review" for unit in units),
        "expected_episodes": len(units) * 128,
        "unique_identities": len(set(identities)) == len(identities),
        "activation": False,
    }


def freeze() -> dict:
    document = manifest()
    selected_workers = io.load("SUBSTRATE_V3_WORKER_AUTHORITY.json")["selected_workers"]
    authority = {
        "schema": "substrate-v3-principal-authority/v1",
        "ready_tag": READY_TAG,
        "source_digest": document["source_digest"],
        "configuration_digest": document["configuration_digest"],
        "admission_sha256": io.load("SUBSTRATE_V3_ADMISSION.json")["sha256"],
        "primary_hypotheses": C.HYPOTHESES,
        "primary_unit": C.STATISTICS["independent_unit"],
        "selected_workers": selected_workers,
        "worker_publication": "unit local staging only",
        "supervisor_publication": "validate then atomically publish unit and checkpoint",
        "live_source_edit": "forbidden",
        "activation": False,
    }
    resource_plan = {
        "schema": "substrate-v3-resource-plan/v1",
        "workers": selected_workers,
        "native_threads_per_worker": 1,
        "expected_units": document["unit_count"],
        "expected_episodes": document["expected_episodes"],
        "estimated_peak_rss_mib": selected_workers * 256,
        "minimum_free_disk_gib": 25,
        "mps": False,
        "hawking": "observation only",
        "activation": False,
    }
    io.config_json("principal_manifest.json", document)
    io.seal("SUBSTRATE_V3_PRINCIPAL_AUTHORITY.json", authority)
    io.seal("SUBSTRATE_V3_PRINCIPAL_DAG.json", document)
    io.seal("SUBSTRATE_V3_RESOURCE_PLAN.json", resource_plan)
    return {"manifest": document, "authority": authority, "resources": resource_plan}


def _load_manifest() -> dict:
    document = json.loads(MANIFEST.read_text())
    expected = io.sha_obj({key: value for key, value in document.items() if key != "sha256"})
    if document.get("sha256") != expected or document.get("activation") is not False:
        raise io.Refused("principal manifest seal invalid")
    return document


def _commit_line(value: str) -> bool:
    return len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value.lower())


def _parse_source_refs(stdout: str, returncode: int) -> tuple[str, str | None]:
    lines = [line.strip() for line in stdout.splitlines()]
    head = lines[0] if lines and _commit_line(lines[0]) else ""
    ready_commit = (
        lines[1]
        if returncode == 0 and len(lines) == 2 and _commit_line(lines[1])
        else None
    )
    return head, ready_commit


def _source_ready() -> dict:
    source_refs = subprocess.run(
        ["git", "rev-parse", "HEAD", f"{READY_TAG}^{{}}"],
        cwd=io.ROOT,
        capture_output=True,
        text=True,
    )
    head, ready_commit = _parse_source_refs(source_refs.stdout, source_refs.returncode)
    if not head:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=io.ROOT, text=True).strip()
    ready_tag_exists = ready_commit is not None
    if not MANIFEST.is_file():
        return {
            "ready_tag_exists": ready_tag_exists,
            "ready_commit": ready_commit,
            "head": head,
            "head_matches_ready": False,
            "source_digest_matches": False,
            "configuration_digest_matches": False,
            "transition_matches": False,
            "reason": "principal manifest has not been frozen",
        }
    manifest_document = _load_manifest()
    transition_path = io.EVIDENCE / "SUBSTRATE_V3_IMPLEMENTATION_TRANSITION.json"
    transition = io.load(transition_path.name) if transition_path.is_file() else None
    transition_matches = bool(
        transition
        and transition.get("classification") == "implementation_defect"
        and transition.get("old_source_digest") == manifest_document["source_digest"]
        and transition.get("new_source_digest") == io.source_digest()
        and transition.get("configuration_digest") == manifest_document["configuration_digest"]
        and transition.get("scientific_configuration_changed") is False
        and transition.get("thresholds_splits_seeds_changed") is False
    )
    head_matches_ready = ready_tag_exists and ready_commit == head
    return {
        "ready_tag_exists": ready_tag_exists,
        "ready_commit": ready_commit,
        "head": head,
        "head_matches_ready": head_matches_ready or transition_matches,
        "head_is_exact_ready_commit": head_matches_ready,
        "source_digest_matches": io.source_digest() == manifest_document["source_digest"],
        "configuration_digest_matches": C.configuration()["configuration_digest"] == manifest_document["configuration_digest"],
        "transition_matches": transition_matches,
        "transition_sha256": transition.get("sha256") if transition else None,
    }


def seal_implementation_transition(affected_units: list[str]) -> dict:
    """Authorize one source-only repair while retaining all unaffected receipts."""
    frozen = _load_manifest()
    known = {unit.identity for unit in work_units()}
    affected = sorted(set(affected_units))
    if not affected or any(identity not in known for identity in affected):
        raise io.Refused("implementation transition must name known affected units")
    current = status()
    if any(identity in current["valid"] for identity in affected):
        raise io.Refused("affected transition units already have valid published receipts")
    if C.configuration()["configuration_digest"] != frozen["configuration_digest"]:
        raise io.Refused("scientific configuration changed during implementation transition")
    document = {
        "schema": "substrate-v3-implementation-transition/v1",
        "classification": "implementation_defect",
        "defect": "v3 histories 1024 through 1047 were passed directly to an immutable v2 generator restricted to preregistered v2 seeds",
        "repair": "deterministically map frozen v3 histories onto authorized immutable v2 generator seeds with a v3-history-specific namespace",
        "old_source_digest": frozen["source_digest"],
        "new_source_digest": io.source_digest(),
        "configuration_digest": frozen["configuration_digest"],
        "scientific_configuration_changed": False,
        "thresholds_splits_seeds_changed": False,
        "hypotheses_changed": False,
        "affected_units": affected,
        "invalidated_units": affected,
        "retained_valid_units": current["complete"],
        "regression": "test_v2_preservation_maps_frozen_v3_histories_to_authorized_seeds",
        "ready_tag_moved": False,
        "ready_tag": READY_TAG,
        "activation": False,
    }
    io.seal("SUBSTRATE_V3_IMPLEMENTATION_TRANSITION.json", document)
    return io.load("SUBSTRATE_V3_IMPLEMENTATION_TRANSITION.json")


def status() -> dict:
    if not MANIFEST.is_file():
        if (io.EVIDENCE / "SUBSTRATE_V3_MODERATE_PILOT.json").is_file():
            stage = "moderate pilot"
        elif (io.EVIDENCE / "SUBSTRATE_V3_CHEAP_CANARIES.json").is_file():
            stage = "cheap admission"
        elif (io.EVIDENCE / "SUBSTRATE_V3_CONSTITUTIONAL_RETROSPECTIVE.json").is_file():
            stage = "constitutional retrospective"
        else:
            stage = "mechanism construction"
        return {
            "stage": stage,
            "expected": 0,
            "complete": 0,
            "remaining": 0,
            "invalid": [],
            "valid": [],
            "source": _source_ready(),
            "stopped": io.STOP.is_file(),
            "activation": False,
        }
    expected = {unit.identity: unit for unit in work_units()}
    valid = []
    invalid = []
    receipt_files = io.regular_file_names(UNITS)
    for identity, unit in expected.items():
        filename = f"{identity}.json"
        if filename not in receipt_files:
            continue
        path = UNITS / filename
        try:
            document = json.loads(path.read_text())
            if validate_receipt(document, unit):
                valid.append(identity)
            else:
                invalid.append(identity)
        except (OSError, json.JSONDecodeError):
            invalid.append(identity)
    complete = len(valid)
    return {
        "stage": "principal development" if complete < len(expected) else "principal complete",
        "expected": len(expected),
        "complete": complete,
        "remaining": len(expected) - complete,
        "invalid": sorted(invalid),
        "valid": sorted(valid),
        "source": _source_ready(),
        "stopped": io.STOP.is_file(),
        "activation": False,
    }


def run() -> dict:
    admission = io.load("SUBSTRATE_V3_ADMISSION.json")
    if not admission.get("principal_execution_licensed"):
        raise io.Refused("principal execution is not licensed by admission")
    source = _source_ready()
    if not all(
        (
            source["ready_tag_exists"],
            source["head_matches_ready"],
            source["source_digest_matches"] or source["transition_matches"],
            source["configuration_digest_matches"],
        )
    ):
        raise io.Refused(f"ready source mismatch: {source}")
    if io.STOP.is_file():
        raise io.Refused("v3 operator stop switch is present")
    units = {unit.identity: unit for unit in work_units()}
    receipt_files = io.regular_file_names(UNITS)
    pending = [unit for unit in units.values() if f"{unit.identity}.json" not in receipt_files]
    workers = int(io.load("SUBSTRATE_V3_WORKER_AUTHORITY.json")["selected_workers"])
    start = time.perf_counter()
    retries = {}
    failures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute_unit, unit): unit for unit in pending}
        for future in as_completed(futures):
            unit = futures[future]
            if io.STOP.is_file():
                for active in futures:
                    active.cancel()
                break
            try:
                receipt = future.result(timeout=unit.timeout)
            except Exception as exc:
                retries[unit.identity] = retries.get(unit.identity, 0) + 1
                if retries[unit.identity] <= 1:
                    try:
                        receipt = execute_unit(unit)
                    except Exception as retry_exc:
                        failures[unit.identity] = f"{retry_exc.__class__.__name__}: {retry_exc}"
                        continue
                else:
                    failures[unit.identity] = f"{exc.__class__.__name__}: {exc}"
                    continue
            if not validate_receipt(receipt, unit):
                failures[unit.identity] = "invalid deterministic receipt"
                continue
            checkpoint = receipt["checkpoint"]
            io.run_json(f"principal/checkpoints/{unit.identity}.json", checkpoint)
            io.run_json(f"principal/units/{unit.identity}.json", receipt)
    current = status()
    launch = {
        "schema": "substrate-v3-principal-launch/v1",
        "workers": workers,
        "pending_at_start": len(pending),
        "retries": retries,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - start,
        "status": current,
        "activation": False,
    }
    io.run_json("principal/launch.json", launch)
    return launch
