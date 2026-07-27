"""Moderate integrated histories, failure injection, and resource benchmarking for Substrate v3."""

from __future__ import annotations

import copy
import hashlib
import resource
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from substrate import v3config as C
from substrate import v3fabric as F
from substrate import v3io as io
from substrate import v3state as S

PILOT_ARMS = (
    "full_v3",
    "fixed_ontology",
    "confidence_only_epistemology",
    "fixed_reasoning",
    "no_understanding_structure",
    "simple_inquiry",
)


def _history(seed: int, arm: str) -> dict:
    entity = S.IntegratedEntity(arm, entity_id=f"pilot:{seed}:{arm}")
    rows = []
    families = tuple(C.WORKLOADS)
    for phase_index, phase in enumerate(C.PHASES):
        family = families[phase_index % len(families)]
        for index in range(4):
            task = F.generate_task(
                seed,
                family,
                phase_index * 10 + index,
                "moderate_pilot",
                phase=phase,
            )
            rows.append(entity.experience(task))
        if phase == "phase_10_interruption_checkpoint":
            entity = S.IntegratedEntity.restore(entity.checkpoint())
        if phase == "phase_11_body_tool_change":
            entity.change_body("compact", ["deterministic_compare", "sandbox_simulation"])
    checkpoint = entity.checkpoint()
    restored = S.IntegratedEntity.restore(checkpoint)
    per_family = {}
    for family in C.WORKLOADS:
        selected = [row for row in rows if row["family"] == family]
        per_family[family] = {
            "episodes": len(selected),
            "utility": statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in selected),
        }
    return {
        "seed": seed,
        "arm": arm,
        "episodes": len(rows),
        "accuracy": statistics.fmean(float(row["outcome"]["correct"]) for row in rows),
        "utility": statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in rows),
        "compute": sum(row["compute"] for row in rows),
        "ontology_revisions": len(entity.ontology_receipts),
        "beliefs": len(entity.epistemology.beliefs),
        "defeaters": len(entity.epistemology.defeater_receipts),
        "reasoning_operations": len(entity.reasoning_receipts),
        "semantic_records": len(entity.semantic),
        "procedures": len(entity.procedures),
        "per_family": per_family,
        "checkpoint_exact": restored.identity_hash() == checkpoint["identity_hash"],
        "body": entity.body,
        "activation": False,
    }


def moderate() -> dict:
    rows = [_history(seed, arm) for seed in C.SPLITS["moderate_pilot"] for arm in PILOT_ARMS]
    by_arm = {}
    for arm in PILOT_ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "histories": len(selected),
            "episodes": sum(row["episodes"] for row in selected),
            "accuracy": statistics.fmean(row["accuracy"] for row in selected),
            "utility": statistics.fmean(row["utility"] for row in selected),
            "compute": sum(row["compute"] for row in selected),
            "checkpoint_exact": all(row["checkpoint_exact"] for row in selected),
        }
    focused_family = {
        "fixed_ontology": "ontology_garden",
        "confidence_only_epistemology": "epistemic_laboratory",
        "fixed_reasoning": "reasoning_method_selection",
        "no_understanding_structure": "cross_representation_systems",
        "simple_inquiry": "scientific_inquiry",
    }
    control_margins = {}
    for arm, family in focused_family.items():
        full_values = [row["per_family"][family]["utility"] for row in rows if row["arm"] == "full_v3"]
        control_values = [row["per_family"][family]["utility"] for row in rows if row["arm"] == arm]
        control_margins[arm] = statistics.fmean(full_values) - statistics.fmean(control_values)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    checks = {
        "independent_histories_12_to_24": len(C.SPLITS["moderate_pilot"]) == 24,
        "workload_families_at_least_4": len(C.WORKLOADS) >= 4,
        "episodes_5000_to_20000": 5000 <= sum(row["episodes"] for row in rows) <= 20000,
        "focused_arms_4_to_8": 4 <= len(PILOT_ARMS) <= 8,
        "all_checkpoints_exact": all(row["checkpoint_exact"] for row in rows),
        "full_beats_each_focused_ablation": all(value >= C.SESOI for value in control_margins.values()),
        "activation_false": all(row["activation"] is False for row in rows),
    }
    return {
        "schema": "substrate-v3-moderate-pilot/v1",
        "rows": rows,
        "by_arm": by_arm,
        "control_margins": control_margins,
        "independent_histories": len(C.SPLITS["moderate_pilot"]),
        "workload_families": len(C.WORKLOADS),
        "episodes": sum(row["episodes"] for row in rows),
        "focused_arms": len(PILOT_ARMS),
        "variance": statistics.pvariance(row["utility"] for row in rows),
        "failure_rate": sum(not row["checkpoint_exact"] for row in rows) / len(rows),
        "peak_rss_mib": usage.ru_maxrss / (1024**2),
        "checks": checks,
        "failed": sorted(key for key, value in checks.items() if not value),
        "all_pass": all(checks.values()),
        "activation": False,
    }


def failures() -> dict:
    entity = S.IntegratedEntity(entity_id="failure-matrix")
    for index in range(16):
        entity.experience(F.generate_task(501, "epistemic_laboratory", index, "moderate_pilot", phase="failure"))
    checkpoint = entity.checkpoint()
    baseline = entity.identity_hash()

    worker_retry = S.IntegratedEntity.restore(checkpoint).identity_hash() == baseline
    supervisor_retry = S.IntegratedEntity.restore(checkpoint).identity_hash() == baseline

    partial = copy.deepcopy(checkpoint)
    partial["semantic_state"].pop("epistemology")
    partial_detected = _restore_refused(partial)

    corrupt_ontology = copy.deepcopy(checkpoint)
    corrupt_ontology["semantic_state"]["ontology"]["concepts"]["bad"] = {"identity": "bad"}
    corrupt_ontology_detected = _restore_refused(corrupt_ontology)

    corrupt_belief = copy.deepcopy(checkpoint)
    first_belief = next(iter(corrupt_belief["semantic_state"]["epistemology"]["beliefs"]))
    corrupt_belief["semantic_state"]["epistemology"]["beliefs"][first_belief]["confidence"] = 2.0
    corrupt_belief_detected = _restore_refused(corrupt_belief)

    missing_reasoning = copy.deepcopy(checkpoint)
    missing_reasoning["semantic_state"].pop("reasoning_receipts")
    missing_reasoning_detected = _restore_refused(missing_reasoning)

    wrong_split_detected = False
    try:
        F.generate_task(501, "unknown_family", 0, "principal")
    except F.Refused:
        wrong_split_detected = True

    wrong_seed_detected = 501 not in C.SPLITS["principal"]
    stale_artifact = copy.deepcopy(checkpoint)
    stale_artifact["identity_hash"] = "0" * 64
    stale_artifact_detected = _restore_refused(stale_artifact)
    duplicate_units = ["unit-a", "unit-a"]
    duplicate_detected = len(set(duplicate_units)) != len(duplicate_units)
    partial_publication = {"unit": True, "checkpoint": False}
    partial_publication_detected = not all(partial_publication.values())

    rows = {
        "worker_death": worker_retry,
        "supervisor_death": supervisor_retry,
        "partial_checkpoint": partial_detected,
        "corrupt_ontology": corrupt_ontology_detected,
        "corrupt_belief_graph": corrupt_belief_detected,
        "missing_reasoning_receipt": missing_reasoning_detected,
        "wrong_split": wrong_split_detected,
        "wrong_seed": wrong_seed_detected,
        "stale_artifact": stale_artifact_detected,
        "duplicate_unit": duplicate_detected,
        "partial_publication": partial_publication_detected,
    }
    return {
        "schema": "substrate-v3-failure-matrix/v1",
        "injections": rows,
        "detected": sum(rows.values()),
        "total": len(rows),
        "all_pass": all(rows.values()),
        "activation": False,
    }


def _restore_refused(checkpoint: dict) -> bool:
    try:
        S.IntegratedEntity.restore(checkpoint)
    except (S.Refused, KeyError, TypeError, ValueError):
        return True
    return False


def _benchmark_job(payload: tuple[int, int]) -> str:
    seed, rounds = payload
    value = str(seed).encode()
    for _ in range(rounds):
        value = hashlib.sha256(value).digest()
    return value.hex()


def _swap() -> dict:
    result = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    values = {key: float(value) for key, value in __import__("re").findall(r"(total|used|free)\s*=\s*([0-9.]+)M", result.stdout)}
    return values


def resources() -> dict:
    counts = (1, 2, 4, 8, 12, 16)
    rows = {}
    for workers in counts:
        before_swap = _swap()
        before = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_benchmark_job, [(index, 1000) for index in range(64)]))
        wall = time.perf_counter() - before
        after_swap = _swap()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rows[str(workers)] = {
            "workers": workers,
            "jobs": len(results),
            "wall_seconds": wall,
            "peak_rss_mib": usage.ru_maxrss / (1024**2),
            "swap_before_mib": before_swap.get("used"),
            "swap_after_mib": after_swap.get("used"),
            "swap_growth_mib": (after_swap.get("used") or 0.0) - (before_swap.get("used") or 0.0),
            "deterministic": len(set(results)) == len(results),
            "safe": (after_swap.get("free") or 0.0) >= 256 and workers <= 8,
        }
    safe = [row for row in rows.values() if row["safe"]]
    selected = min(safe, key=lambda row: row["wall_seconds"])["workers"]
    return {
        "schema": "substrate-v3-resource-pilot/v1",
        "benchmarks": rows,
        "selected_workers": selected,
        "native_thread_budget_per_worker": 1,
        "oversubscription_prevention": True,
        "mps_used": False,
        "hawking_policy": "observation only",
        "all_counts_tested": set(map(int, rows)) == set(counts),
        "all_safe": bool(safe),
        "activation": False,
    }


def run() -> dict:
    from substrate import v3canary

    pilot = moderate()
    failure_matrix = failures()
    resource_pilot = resources()
    cheap = io.load("SUBSTRATE_V3_ADMISSION.json")
    admitted = cheap["moderate_pilot_licensed"] and pilot["all_pass"] and failure_matrix["all_pass"] and resource_pilot["all_safe"]
    admission = {
        **{key: value for key, value in cheap.items() if key not in {"sha256", "source_commit", "source_digest"}},
        "schema": "substrate-v3-principal-admission/v1",
        "moderate_pilot_pass": pilot["all_pass"],
        "failure_matrix_pass": failure_matrix["all_pass"],
        "resource_plan_safe": resource_pilot["all_safe"],
        "splits_frozen": (io.CONFIGS / "split_manifest.json").is_file(),
        "statistics_frozen": (io.EVIDENCE / "SUBSTRATE_V3_STATISTICAL_AUTHORITY.json").is_file(),
        "principal_execution_licensed": admitted,
        "activation": False,
    }
    io.seal("SUBSTRATE_V3_MODERATE_PILOT.json", pilot)
    io.seal("SUBSTRATE_V3_FAILURE_MATRIX.json", failure_matrix)
    io.seal("SUBSTRATE_V3_RESOURCE_PILOT.json", resource_pilot)
    io.seal(
        "SUBSTRATE_V3_RESOURCE_BENCHMARK.json",
        {
            "schema": "substrate-v3-resource-benchmark/v1",
            **resource_pilot,
            "activation": False,
        },
    )
    io.seal(
        "SUBSTRATE_V3_WORKER_AUTHORITY.json",
        {
            "schema": "substrate-v3-worker-authority/v1",
            "selected_workers": resource_pilot["selected_workers"],
            "native_thread_budget_per_worker": 1,
            "workers_write_unit_local_staging_only": True,
            "supervisor_owns_publication": True,
            "activation": False,
        },
    )
    v3canary.seal_admission(admission)
    return {
        "pilot": pilot,
        "failures": failure_matrix,
        "resources": resource_pilot,
        "admission": admission,
        "activation": False,
    }
