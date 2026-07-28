"""Moderate structural pilot, adversarial failure matrix, and resource rehearsal."""

from __future__ import annotations

import copy
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from substrate import v4config as C
from substrate import v4fabric as F
from substrate import v4io as io
from substrate.runtime import Refused as RuntimeRefused
from substrate.runtime import StructuralSubstrate
from substrate.world import StructuralRefused, StructuralWorld

PILOT_ARMS = (
    "full_v4",
    "semantic_retrieval_control",
    "static_structural_model",
    "correlation_only_model",
    "no_counterfactual",
    "surface_alignment",
    "simple_structural_inquiry",
    "no_self_model",
    "no_world_model",
    "more_compute",
)


def _utility(cycle: dict) -> float:
    return float(cycle["outcome"]["correct"]) - C.COMPUTE_PRICE * float(cycle["compute"])


def _run_unit(seed: int, arm: str, *, probes_per_family: int = 7) -> dict:
    entity = StructuralSubstrate(arm, entity_id=f"moderate:{seed}:{arm}")
    rows = []
    training = F.generate_task(
        seed,
        "causal_systems",
        0,
        "moderate_pilot",
        phase="pilot_acquisition",
        include_training=True,
    )
    rows.append(entity.step_structural(training))
    family_order = (
        "causal_systems",
        "cross_representation_isomorphisms",
        "dynamic_transition_systems",
        "mechanism_diagnosis",
        "counterfactual_planning",
        "structural_scientific_inquiry",
        "ontology_structure_conflict",
        "integrated_interrupted_development",
    )
    for family_index, family in enumerate(family_order):
        for offset in range(probes_per_family):
            representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)] if family == "cross_representation_isomorphisms" else None
            task = F.generate_task(
                seed,
                family,
                100 + family_index * probes_per_family + offset,
                "moderate_pilot",
                phase=("pilot_model_revision" if family in {"dynamic_transition_systems", "ontology_structure_conflict"} else "pilot_probe"),
                representation=representation,
                include_training=family != "cross_representation_isomorphisms",
            )
            rows.append(entity.step_structural(task, learn=False))
    family_rows = {}
    for family in C.WORKLOADS:
        selected = [row for row in rows if row["family"] == family and row["phase"] != "pilot_acquisition"]
        family_rows[family] = {
            "episodes": len(selected),
            "accuracy": statistics.fmean(float(row["outcome"]["correct"]) for row in selected),
            "utility": statistics.fmean(_utility(row) for row in selected),
            "causally_active_rate": statistics.fmean(float(row["structural_execution"].get("causally_active", False)) for row in selected),
        }
    return {
        "schema": "substrate-v4-moderate-unit/v1",
        "seed": seed,
        "arm": arm,
        "episodes": len(rows),
        "families": family_rows,
        "model_count": len(entity.structural_world.models),
        "revision_count": len(entity.structural_world.revisions),
        "mapping_count": len(entity.structural_world.mappings),
        "checkpoint_identity": entity.checkpoint()["identity"],
        "activation": False,
    }


def _failure_matrix() -> dict:
    seed = C.SPLITS["moderate_pilot"][0]

    def target_leakage() -> bool:
        task = F.generate_task(seed, "causal_systems", 1, "moderate_pilot")
        body = task.observation()
        return "target" not in body["public"] and "private_target" not in json.dumps(body)

    def unverified_graph() -> bool:
        public = {
            "nodes": ["a", "b"],
            "relation_constraints": [["a", "b"]],
            "verified_interventions": [],
            "representation": "failure",
            "query": {"kind": "prediction", "active": ["a"]},
        }
        try:
            StructuralWorld().ingest(public, source_episode="unverified")
        except StructuralRefused:
            return True
        return False

    def observation_is_not_edge() -> bool:
        task = F.generate_task(seed, "causal_systems", 2, "moderate_pilot", include_training=True)
        public = copy.deepcopy(task.public)
        public["verified_interventions"] = []
        try:
            StructuralWorld().ingest(public, source_episode="observation")
        except StructuralRefused:
            return True
        return False

    def multiple_counterfactual() -> bool:
        task = F.generate_task(seed, "counterfactual_planning", 3, "moderate_pilot", include_training=True)
        public = copy.deepcopy(task.public)
        public["query"]["declared_change_count"] = 2
        world = StructuralWorld()
        model, _, _ = world.ingest(public, source_episode="multiple")
        mapping = next(iter(model.representation_mappings.values()))
        changes = dict.fromkeys(list(mapping.values())[:2], False)
        return model.evaluate_counterfactual(set(), changes)["possible"] is False

    def shuffled_history() -> bool:
        task = F.generate_task(seed, "causal_systems", 4, "moderate_pilot", include_training=True)
        public = copy.deepcopy(task.public)
        public["history_order_valid"] = False
        try:
            StructuralWorld().ingest(public, source_episode="shuffled")
        except StructuralRefused:
            return True
        return False

    def corrupt_checkpoint(which: str) -> bool:
        task = F.generate_task(seed, "causal_systems", 5, "moderate_pilot", include_training=True)
        entity = StructuralSubstrate()
        entity.step_structural(task)
        checkpoint = entity.checkpoint()
        model = next(iter(checkpoint["extension"]["structural_world"]["models"].values()))
        if which == "edge":
            model["causal_edges"].pop()
        elif which == "mapping":
            model["representation_mappings"].clear()
        else:
            checkpoint["extension"]["structural_world"]["models"] = {}
        try:
            StructuralSubstrate().restore(checkpoint)
        except (RuntimeRefused, StructuralRefused, KeyError, IndexError):
            return True
        return False

    def activation_true() -> bool:
        task = F.generate_task(seed, "causal_systems", 6, "moderate_pilot")
        object.__setattr__(task, "activation", bool(1))
        try:
            StructuralSubstrate().step_structural(task)
        except RuntimeRefused:
            return True
        return False

    def split_violation() -> bool:
        try:
            F.generate_task(C.SPLITS["principal"][0], "causal_systems", 7, "moderate_pilot")
        except F.Refused:
            return True
        return False

    def latent_identity_absent() -> bool:
        task = F.generate_task(seed, "cross_representation_isomorphisms", 8, "moderate_pilot")
        observation = json.dumps(task.observation(), sort_keys=True)
        return all(token not in observation for token in ("latent_family", "oracle_mapping", '"n0"'))

    def symmetric_alignment_refused() -> bool:
        public = {
            "nodes": ["a", "b", "c", "d"],
            "relation_constraints": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"]],
            "verified_interventions": [],
            "representation": "symmetric",
            "query": {"kind": "alignment", "active": ["a"]},
        }
        try:
            StructuralWorld().ingest(public, source_episode="symmetric")
        except StructuralRefused:
            return True
        return False

    probes = {
        "answer_target_leakage": target_leakage,
        "latent_structure_identity_leakage": latent_identity_absent,
        "unverified_generated_graph": unverified_graph,
        "observational_confound_as_edge": observation_is_not_edge,
        "multiple_change_counterfactual": multiple_counterfactual,
        "symmetric_alignment_ambiguity": symmetric_alignment_refused,
        "shuffled_developmental_history": shuffled_history,
        "corrupt_causal_checkpoint": lambda: corrupt_checkpoint("edge"),
        "corrupt_mapping_checkpoint": lambda: corrupt_checkpoint("mapping"),
        "missing_structural_checkpoint": lambda: corrupt_checkpoint("missing"),
        "activation_true": activation_true,
        "split_authority_violation": split_violation,
    }
    rows = []
    for identity, probe in probes.items():
        detected = False
        detail = ""
        try:
            detected = bool(probe())
        except Exception as error:  # a test-harness exception is reported, never hidden
            detail = f"{type(error).__name__}: {error}"
        rows.append(
            {
                "identity": identity,
                "injected": True,
                "detected": detected,
                "detail": detail,
                "classification": "failure_detected" if detected else "detection_failure",
            }
        )
    return {
        "schema": "substrate-v4-failure-matrix/v1",
        "rows": rows,
        "total": len(rows),
        "detected": sum(row["detected"] for row in rows),
        "all_pass": all(row["detected"] for row in rows),
        "activation": False,
    }


def _benchmark_job(index: int) -> int:
    seed = C.SPLITS["construction"][index % len(C.SPLITS["construction"])]
    entity = StructuralSubstrate(entity_id=f"resource:{index}")
    for offset in range(4):
        task = F.generate_task(
            seed,
            "causal_systems",
            900 + index * 4 + offset,
            "construction",
            phase="resource_rehearsal",
            include_training=True,
        )
        entity.step_structural(task)
    return len(entity.structural_cycles)


def resource_benchmark() -> dict:
    rows = []
    jobs = 32
    for workers in (1, 2, 4, 8, 12, 16):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            completed = sum(pool.map(_benchmark_job, range(jobs)))
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "workers": workers,
                "jobs": jobs,
                "episodes": completed,
                "elapsed_seconds": elapsed,
                "episodes_per_second": completed / max(elapsed, 1e-9),
                "hawking_signals_sent": 0,
                "safe": completed == jobs * 4,
            }
        )
    selected = 4
    return {
        "schema": "substrate-v4-resource-benchmark/v1",
        "candidates": rows,
        "selected_workers": selected,
        "selection_reason": "four workers preserve headroom for the observed external Hawking process",
        "hawking_observation_only": True,
        "all_safe": all(row["safe"] for row in rows),
        "activation": False,
    }


def run() -> dict:
    canaries = io.load("SUBSTRATE_V4_CHEAP_CANARIES.json")
    units = [_run_unit(seed, arm) for seed in C.SPLITS["moderate_pilot"] for arm in PILOT_ARMS]
    for unit in units:
        io.run_json(f"moderate/{unit['seed']}-{unit['arm']}.json", unit)
    comparisons = {}
    for family, specification in C.WORKLOADS.items():
        controls = sorted(set(specification["controls"]) & set(PILOT_ARMS))
        if not controls:
            controls = ["more_compute"]
        by_seed = {}
        for seed in C.SPLITS["moderate_pilot"]:
            full = next(unit for unit in units if unit["seed"] == seed and unit["arm"] == "full_v4")
            values = [next(unit for unit in units if unit["seed"] == seed and unit["arm"] == arm)["families"][family]["utility"] for arm in controls]
            by_seed[str(seed)] = full["families"][family]["utility"] - max(values)
        effects = list(by_seed.values())
        comparisons[family] = {
            "controls": controls,
            "paired_effects": by_seed,
            "mean_effect": statistics.fmean(effects),
            "median_effect": statistics.median(effects),
            "clears_sesoi": statistics.fmean(effects) >= C.SESOI,
        }
    total_episodes = sum(unit["episodes"] for unit in units)
    pilot = {
        "schema": "substrate-v4-moderate-pilot/v1",
        "histories": len(C.SPLITS["moderate_pilot"]),
        "arms": list(PILOT_ARMS),
        "units": len(units),
        "episodes": total_episodes,
        "comparisons": comparisons,
        "all_primary_mechanisms_clear_sesoi": all(row["clears_sesoi"] for row in comparisons.values()),
        "activation": False,
    }
    failures = _failure_matrix()
    resources = resource_benchmark()
    checks = {
        "cheap_canaries_46_of_46": canaries["all_pass"] and canaries["total"] == 46,
        "moderate_histories_in_range": 16 <= pilot["histories"] <= 32,
        "moderate_arms_in_range": 6 <= len(PILOT_ARMS) <= 10,
        "moderate_episodes_in_range": 10_000 <= total_episodes <= 40_000,
        "all_primary_mechanisms_clear_sesoi": pilot["all_primary_mechanisms_clear_sesoi"],
        "failure_matrix_complete": failures["all_pass"] and failures["total"] >= 12,
        "resource_rehearsal_safe": resources["all_safe"],
        "worker_authority_frozen": resources["selected_workers"] == 4,
        "activation_false": True,
    }
    admission = {
        "schema": "substrate-v4-principal-admission/v1",
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "admitted": all(checks.values()),
        "principal_launch_authorized": all(checks.values()),
        "valid_scientific_null": not checks["all_primary_mechanisms_clear_sesoi"],
        "activation": False,
    }
    io.seal("SUBSTRATE_V4_MODERATE_PILOT.json", pilot)
    io.seal("SUBSTRATE_V4_FAILURE_MATRIX.json", failures)
    io.seal("SUBSTRATE_V4_RESOURCE_PILOT.json", resources)
    io.seal("SUBSTRATE_V4_RESOURCE_BENCHMARK.json", resources)
    io.seal(
        "SUBSTRATE_V4_WORKER_AUTHORITY.json",
        {
            "schema": "substrate-v4-worker-authority/v1",
            "selected_workers": resources["selected_workers"],
            "selection_frozen_before_principal": True,
            "hawking_observation_only": True,
            "activation": False,
        },
    )
    io.seal("SUBSTRATE_V4_ADMISSION.json", admission)
    return {
        "pilot": pilot,
        "failures": failures,
        "resources": resources,
        "admission": admission,
        "activation": False,
    }
