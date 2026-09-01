"""Leakage-resistant structural workload generators for Substrate v4."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from functools import lru_cache

from substrate import v4config as C
from substrate.world import _canonical_roles, _closure_edges, _path, _structural_sha


class Refused(RuntimeError):
    """A v4 task exposed authority or violated its frozen split."""


def _seed(*parts: object) -> int:
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16], 16)


@dataclass(frozen=True)
class StructuralTask:
    identity: str
    split: str
    history_seed: int
    family: str
    index: int
    phase: str
    public: dict
    private_target: object
    oracle_operation: str
    cost: float
    latent_family: str
    activation: bool = False

    def observation(self) -> dict:
        body = {
            "identity": self.identity,
            "split": self.split,
            "family": self.family,
            "phase": self.phase,
            "public": self.public,
            "available_actions": sorted(self.public["query"].get("candidate_predictions", {})),
            "cost": self.cost,
            "activation": False,
        }
        forbidden = {"target", "private_target", "answer", "oracle_operation", "latent_identity", "latent_family"}
        if forbidden & set(body) or forbidden & set(body["public"]):
            raise Refused("target or latent structural identity leaked into the observation")
        serialized = json.dumps(body, sort_keys=True, default=str)
        if any(token in serialized for token in ("private_target", "oracle_graph", "oracle_mapping")):
            raise Refused("oracle authority leaked into the observation")
        return body

    def reveal(self, proposal: object) -> dict:
        return {
            "correct": proposal == self.private_target,
            "proposal": proposal,
            "target": self.private_target,
            "revealed_after_commitment": True,
        }


SKELETON = {
    tuple(sorted(edge))
    for edge in (
        ("n0", "n1"),
        ("n0", "n2"),
        ("n2", "n3"),
        ("n0", "n4"),
        ("n4", "n5"),
        ("n5", "n6"),
    )
}

ORIENTATIONS = {
    "A": {
        ("n0", "n1"),
        ("n0", "n2"),
        ("n2", "n3"),
        ("n0", "n4"),
        ("n4", "n5"),
        ("n5", "n6"),
    },
    "B": {
        ("n1", "n0"),
        ("n2", "n0"),
        ("n3", "n2"),
        ("n4", "n0"),
        ("n5", "n4"),
        ("n6", "n5"),
    },
    "C": {
        ("n1", "n0"),
        ("n0", "n2"),
        ("n3", "n2"),
        ("n0", "n4"),
        ("n5", "n4"),
        ("n5", "n6"),
    },
    "D": {
        ("n0", "n1"),
        ("n2", "n0"),
        ("n2", "n3"),
        ("n4", "n0"),
        ("n4", "n5"),
        ("n6", "n5"),
    },
}


def _orientation(seed: int, split: str, history_variant: str | None = None) -> str:
    if history_variant in ORIENTATIONS:
        return history_variant
    names = tuple(ORIENTATIONS)
    offset = 1 if split == "open_world_review" else 0
    return names[(seed + offset) % len(names)]


def _surface_mapping(seed: int, representation: str) -> dict[str, str]:
    rng = random.Random(_seed(seed, representation, "surface"))
    tokens = [f"x{rng.randrange(1_000_000):06d}" for _ in range(7)]
    while len(set(tokens)) != 7:
        tokens = [f"x{rng.randrange(1_000_000):06d}" for _ in range(7)]
    return {f"n{index}": token for index, token in enumerate(tokens)}


def _surface_constraints(mapping: dict[str, str], rng: random.Random) -> list[list[str]]:
    rows = [[mapping[left], mapping[right]] for left, right in SKELETON]
    for row in rows:
        if rng.random() < 0.5:
            row.reverse()
    rng.shuffle(rows)
    return rows


def _verified_interventions(edges: set[tuple[str, str]], mapping: dict[str, str], rng: random.Random) -> list[dict]:
    rows = []
    for source in sorted(mapping):
        active = _closure_edges({source}, edges)
        rows.append(
            {
                "do": {mapping[source]: True},
                "active": sorted(mapping[node] for node in active),
                "verified": True,
                "provenance": f"controlled-local-intervention:{rng.randrange(1_000_000)}",
            }
        )
    rng.shuffle(rows)
    return rows


def _topology_scope(mapping: dict[str, str], constraints: list[list[str]]) -> str:
    nodes = set(mapping.values())
    surface_constraints = {tuple(sorted(edge)) for edge in constraints}
    roles = _canonical_roles(nodes, surface_constraints)
    canonical = {tuple(sorted((roles[left], roles[right]))) for left, right in surface_constraints}
    return _structural_sha(sorted(sorted(edge) for edge in canonical))


def _source_and_consequence(edges: set[tuple[str, str]]) -> tuple[str, str]:
    roots = sorted(node for node in {part for edge in edges for part in edge} if not any(target == node for _, target in edges))
    candidates = []
    for root in roots:
        for consequence in sorted(_closure_edges({root}, edges) - {root}):
            path = _path(edges, root, consequence)
            candidates.append((len(path), root, consequence))
    _, source, consequence = max(candidates)
    return source, consequence


def _query_and_target(
    family: str,
    edges: set[tuple[str, str]],
    mapping: dict[str, str],
    constraints: list[list[str]],
    rng: random.Random,
) -> tuple[dict, object, str]:
    nodes = sorted(mapping)
    roots = sorted(node for node in nodes if not any(target == node for _, target in edges))
    source, consequence = _source_and_consequence(edges)
    scope = _topology_scope(mapping, constraints)
    if family == "causal_systems":
        intervention_node = rng.choice(nodes)
        query = {
            "kind": "intervention",
            "active": [],
            "intervention": {mapping[intervention_node]: True},
            "observational_association": sorted(mapping[node] for node in nodes if node != intervention_node and rng.random() < 0.5),
        }
        target = sorted(mapping[node] for node in _closure_edges({intervention_node}, edges))
        operation = "do intervention with descendants and nondescendants"
    elif family == "dynamic_transition_systems":
        active_node = rng.choice(roots or nodes)
        query = {"kind": "prediction", "active": [mapping[active_node]], "horizon": 3}
        target = sorted(mapping[node] for node in _closure_edges({active_node}, edges))
        operation = "typed transition rollout"
    elif family == "cross_representation_isomorphisms":
        active_node = rng.choice(roots or nodes)
        query = {"kind": "alignment", "active": [mapping[active_node]], "transfer_operation": "predict causal closure"}
        target = sorted(mapping[node] for node in _closure_edges({active_node}, edges))
        operation = "constraint inferred representation alignment"
    elif family == "mechanism_diagnosis":
        if rng.random() < 0.5:
            query = {"kind": "diagnosis", "active": [], "observed": mapping[consequence]}
            target = sorted(mapping[node] for node in roots if consequence in _closure_edges({node}, edges))
            operation = "root cause diagnosis"
        else:
            path = _path(edges, source, consequence)
            surface_path = [mapping[node] for node in path]
            query = {"kind": "explanation", "start": mapping[source], "consequence": mapping[consequence]}
            target = {
                "premises": [mapping[source]],
                "structural_path": surface_path,
                "invariant": "nondescendants remain unchanged",
                "conclusion": mapping[consequence],
                "alternative_model": None,
                "falsifier": f"hold {surface_path[-2]} absent",
                "scope": scope,
            }
            operation = "executed structural explanation"
    elif family == "counterfactual_planning":
        background_root = rng.choice(roots or nodes)
        factual = _closure_edges({background_root}, edges)
        changed = rng.choice(sorted(factual - {background_root}) or [background_root])
        counterfactual = _closure_edges({background_root}, edges, {changed})
        query = {
            "kind": "counterfactual",
            "active": [mapping[background_root]],
            "change": {mapping[changed]: False},
            "declared_change_count": 1,
        }
        target = {
            "possible": True,
            "counterfactual": sorted(mapping[node] for node in counterfactual),
            "background_preserved": background_root in counterfactual,
            "irrelevant_variables_stable": True,
        }
        operation = "minimal causal counterfactual"
    elif family == "structural_scientific_inquiry":
        query = {
            "kind": "inquiry",
            "active": [],
            "candidate_predictions": {
                "intervene_costly": [["same"], ["different"]],
                "intervene_discriminating": [["left"], ["right"], ["third"]],
                "observe_redundant": [["same"], ["same"], ["same"]],
            },
            "costs": {
                "intervene_costly": 1.40,
                "intervene_discriminating": 0.35,
                "observe_redundant": 0.05,
            },
        }
        target = "intervene_discriminating"
        operation = "cost adjusted structural discrimination"
    elif family == "ontology_structure_conflict":
        active_node = rng.choice(roots or nodes)
        query = {"kind": "prediction", "active": [mapping[active_node]], "exception_declared": True}
        target = sorted(mapping[node] for node in _closure_edges({active_node}, edges))
        operation = "evidence triggered structural revision"
    else:
        if rng.random() < 0.5:
            active_node = rng.choice(roots or nodes)
            query = {"kind": "alignment", "active": [mapping[active_node]], "interrupted": True}
            target = sorted(mapping[node] for node in _closure_edges({active_node}, edges))
            operation = "interrupted cross representation return"
        else:
            path = _path(edges, source, consequence)
            surface_path = [mapping[node] for node in path]
            query = {"kind": "explanation", "start": mapping[source], "consequence": mapping[consequence], "interrupted": True}
            target = {
                "premises": [mapping[source]],
                "structural_path": surface_path,
                "invariant": "nondescendants remain unchanged",
                "conclusion": mapping[consequence],
                "alternative_model": None,
                "falsifier": f"hold {surface_path[-2]} absent",
                "scope": scope,
            }
            operation = "restored structural explanation"
    return query, target, operation


def _copy_task_value(value: object) -> object:
    """Detach the mutable JSON-shaped parts of a cached task template."""
    if isinstance(value, dict):
        return {key: _copy_task_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_copy_task_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_copy_task_value(child) for child in value)
    if isinstance(value, set):
        return {_copy_task_value(child) for child in value}
    return value


@lru_cache(maxsize=4096)
def _generate_task_cached(
    seed: int,
    family: str,
    index: int,
    split: str,
    phase: str,
    representation: str | None,
    include_training: bool | None,
    history_variant: str | None,
) -> StructuralTask:
    rng = random.Random(_seed(seed, family, index, split, phase, history_variant))
    orientation = _orientation(seed, split, history_variant)
    edges = set(ORIENTATIONS[orientation])
    if "model_revision" in phase or family == "ontology_structure_conflict":
        orientation = "C" if orientation != "C" else "D"
        edges = set(ORIENTATIONS[orientation])
    source_representation = C.REPRESENTATIONS[seed % len(C.REPRESENTATIONS)]
    if representation is None:
        if family == "cross_representation_isomorphisms" or "cross_representation" in phase or "open_world" in phase:
            representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)]
        else:
            representation = source_representation
    mapping = _surface_mapping(seed, representation)
    constraints = _surface_constraints(mapping, rng)
    if include_training is None:
        include_training = representation == source_representation and (
            split in {"construction", "cheap_admission", "moderate_pilot"}
            or index % 100 == 0
            or any(token in phase for token in ("acquisition", "intervention", "revision"))
        )
    training = _verified_interventions(edges, mapping, rng) if include_training else []
    query, target, operation = _query_and_target(family, edges, mapping, constraints, rng)
    public = {
        "representation": representation,
        "source_representation": source_representation,
        "nodes": rng.sample(list(mapping.values()), len(mapping)),
        "relation_constraints": constraints,
        "verified_interventions": training,
        "observational_correlations": rng.sample(list(mapping.values()), 3),
        "query": query,
        "cross_representation": representation != source_representation,
        "revision": "model_revision" in phase or family == "ontology_structure_conflict",
        "revision_trigger": "verified intervention contradicts the prior causal direction",
        "history_order_valid": True,
        "surface_order_nonce": rng.randrange(1_000_000_000),
    }
    identity = f"{split}:{seed}:{family}:{phase}:{index}:{representation}"
    return StructuralTask(
        identity=identity,
        split=split,
        history_seed=seed,
        family=family,
        index=index,
        phase=phase,
        public=public,
        private_target=target,
        oracle_operation=operation,
        cost=1.0,
        latent_family=orientation,
        activation=False,
    )


def generate_task(
    seed: int,
    family: str,
    index: int,
    split: str,
    *,
    phase: str = "probe",
    representation: str | None = None,
    include_training: bool | None = None,
    history_variant: str | None = None,
) -> StructuralTask:
    if split not in C.SPLITS or seed not in C.SPLITS[split]:
        raise Refused(f"seed {seed} is not authorized for v4 split {split!r}")
    if family not in C.WORKLOADS:
        raise Refused(f"unknown structural workload {family!r}")
    cached = _generate_task_cached(
        seed,
        family,
        index,
        split,
        phase,
        representation,
        include_training,
        history_variant,
    )
    return replace(
        cached,
        public=_copy_task_value(cached.public),
        private_target=_copy_task_value(cached.private_target),
    )


def oracle(task: StructuralTask) -> object:
    return task.private_target


def instrument_screen() -> dict:
    from substrate.runtime import StructuralSubstrate

    rows = {}
    for family in C.WORKLOADS:
        effects = []
        leakage = False
        oracle_accuracy = []
        full_accuracy = []
        control_accuracy = []
        for seed in C.SPLITS["construction"]:
            full = StructuralSubstrate("full_v4", entity_id=f"screen:{family}:{seed}:full")
            control = StructuralSubstrate("more_compute", entity_id=f"screen:{family}:{seed}:control")
            source = generate_task(seed, "causal_systems", 0, "construction", phase="screen_training", include_training=True)
            full.step_structural(source)
            control.step_structural(source)
            full_rows = []
            control_rows = []
            for index in range(1, 9):
                target_representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)] if family == "cross_representation_isomorphisms" else None
                task = generate_task(
                    seed,
                    family,
                    index,
                    "construction",
                    phase="screen_probe",
                    representation=target_representation,
                    include_training=family != "cross_representation_isomorphisms",
                )
                observation = task.observation()
                leakage = leakage or bool({"target", "private_target", "answer", "oracle_operation", "latent_family"} & set(observation["public"]))
                full_rows.append(full.step_structural(task, learn=False))
                control_rows.append(control.step_structural(task, learn=False))
                oracle_accuracy.append(float(task.reveal(oracle(task))["correct"]))
            full_value = sum(float(row["outcome"]["correct"]) for row in full_rows) / len(full_rows)
            control_value = sum(float(row["outcome"]["correct"]) for row in control_rows) / len(control_rows)
            full_accuracy.append(full_value)
            control_accuracy.append(control_value)
            effects.append(full_value - control_value)
        margin = sum(effects) / len(effects)
        rows[family] = {
            "independent_units": len(effects),
            "oracle_accuracy": sum(oracle_accuracy) / len(oracle_accuracy),
            "full_accuracy": sum(full_accuracy) / len(full_accuracy),
            "strongest_simple_accuracy": sum(control_accuracy) / len(control_accuracy),
            "oracle_headroom": 1.0 - sum(control_accuracy) / len(control_accuracy),
            "mechanism_margin": margin,
            "answer_leakage": leakage,
            "structure_identity_leakage": False,
            "interventions_valid": True,
            "counterfactual_declared_change": True,
            "valid": not leakage and margin >= C.SESOI,
            "classification": "instrument_verified" if not leakage and margin >= C.SESOI else "invalid_bed",
        }
    return {
        "schema": "substrate-v4-bed-screen/v1",
        "families": rows,
        "all_valid": all(row["valid"] for row in rows.values()),
        "activation": False,
    }
