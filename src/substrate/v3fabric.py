"""Deterministic local epistemic workloads and allocation beds for Substrate v3."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache

from substrate import v3config as C


class Refused(RuntimeError):
    """A task or policy attempted to use unavailable outcome information."""


def _clone_payload(value: object) -> object:
    """Clone the JSON-shaped payloads emitted by the deterministic task bank.

    Generated task payloads deliberately use only dict/list/tuple containers and
    scalar leaves. Keeping this clone structural avoids the memo-table and type
    dispatch overhead of ``copy.deepcopy`` while retaining isolation between
    arm histories. Unsupported mutable shapes fail closed instead of being
    silently shared between tasks.
    """
    if type(value) is dict:
        return {_clone_payload(key): _clone_payload(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone_payload(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone_payload(item) for item in value)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise Refused(f"unsupported mutable task payload type {type(value).__name__}")


@lru_cache(maxsize=8192)
def _seed(*parts: object) -> int:
    # Task generation repeats the same immutable seed tuple across controlled
    # arms. Cache only the derived integer, not mutable task objects, so arm
    # histories cannot share or mutate a task while deterministic generation
    # still avoids duplicate canonical hashing.
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
    return int(digest[:16], 16)


@dataclass
class CognitiveTask:
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

    def observation(self) -> dict:
        body = {
            "identity": self.identity,
            "split": self.split,
            "family": self.family,
            "phase": self.phase,
            "public": self.public,
            "available_actions": C.WORKLOADS[self.family]["actions"],
            "cost": self.cost,
        }
        # Key order is irrelevant to this substring leakage guard. Avoiding
        # sorting here keeps the per-arm observation path cheap; canonical
        # identity hashes remain sorted in _seed and the evidence layer.
        serialized = json.dumps(body, default=str)
        target_digest = hashlib.sha256(json.dumps(self.private_target, default=str).encode()).hexdigest()
        forbidden_keys = {"target", "private_target", "answer", "oracle_operation"}
        if forbidden_keys & set(body) or forbidden_keys & set(body["public"]) or target_digest in serialized:
            raise Refused("answer or answer digest leaked into public observation")
        return body

    def reveal(self, proposal: object) -> dict:
        return {
            "correct": proposal == self.private_target,
            "target": self.private_target,
            "proposal": proposal,
            "revealed_after_commitment": True,
        }


def _generate_task(seed: int, family: str, index: int, split: str, *, phase: str = "probe") -> CognitiveTask:
    if family not in C.WORKLOADS:
        raise Refused(f"unknown v3 family {family!r}")
    rng = random.Random(_seed(seed, family, index, split, phase))
    identity = f"{split}:{seed}:{family}:{phase}:{index}"
    if family == "ontology_garden":
        shape = rng.choice(["round", "angular"])
        motion = rng.choice(["stable", "changing"])
        exception = rng.random() < 0.25
        target = "process" if motion == "changing" and not exception else "entity"
        public = {
            "instance": f"item-{rng.randrange(10_000)}",
            "features": [shape, motion, "exception" if exception else "regular"],
            "surface_label": f"label-{rng.randrange(1_000_000)}",
        }
        operation = "category split and exception"
    elif family == "epistemic_laboratory":
        regime = rng.choice(["alpha", "beta"])
        source_a_reliable = regime == "alpha"
        truth = rng.choice([0, 1])
        report_a = truth if source_a_reliable else 1 - truth
        report_b = truth if not source_a_reliable else 1 - truth
        public = {
            "regime": regime,
            "reports": {"source_a": report_a, "source_b": report_b},
            "source_history": {
                "source_a": {"alpha": 0.9, "beta": 0.1},
                "source_b": {"alpha": 0.1, "beta": 0.9},
            },
            "contradiction": True,
        }
        target = truth
        operation = "source reliability defeater"
    elif family == "causal_micro_worlds":
        cause = rng.choice([0, 1])
        confounder = rng.choice([0, 1])
        if "counterfactual" in phase:
            context = rng.choice([0, 1])
            changed_cause = 1 - cause
            public = {
                "query_kind": "counterfactual",
                "background": {"cause": cause, "context": context},
                "change": {"cause": changed_cause},
                "confounder": confounder,
                "query": "effect after changing only cause and preserving context",
            }
            target = int(bool(changed_cause and context))
            operation = "minimal counterfactual"
        else:
            observed = cause or confounder
            public = {
                "query_kind": "intervention",
                "observed_cause": cause,
                "confounder": confounder,
                "observational_effect": int(observed),
                "query": "effect under intervention holding confounder absent",
            }
            target = int(bool(cause))
            operation = "causal intervention"
    elif family == "cross_representation_systems":
        encoding = rng.choice(["symbolic", "sequence", "graph", "statement", "tool"])
        labels = rng.sample(["ka", "zu", "mi", "te"], 4)
        edges = [[labels[0], labels[1]], [labels[1], labels[2]], [labels[2], labels[3]]]
        query_index = rng.choice([1, 2, 3])
        public = {
            "encoding": encoding,
            "relations": edges,
            "start": labels[0],
            "query_distance": query_index,
        }
        target = labels[query_index]
        operation = "latent relation transfer"
    elif family == "reasoning_method_selection":
        mode = rng.choice(["deduction", "induction", "abduction", "analogy", "diagnostic", "planning"])
        if mode == "deduction":
            public = {
                "feature": "necessary_consequence",
                "facts": ["a"],
                "rules": [[["a"], "b"], [["b"], "c"]],
                "query": "c",
            }
            target = True
        elif mode == "induction":
            samples = [True] * 7 + [False] * 2
            rng.shuffle(samples)
            public = {"feature": "sample_generalization", "samples": samples}
            target = True
        elif mode == "abduction":
            public = {
                "feature": "hidden_cause",
                "explanations": [
                    {"identity": "x", "support": 3, "cost": 2},
                    {"identity": "y", "support": 2, "cost": 1},
                ],
            }
            target = "x"
        elif mode == "analogy":
            public = {
                "feature": "relational_transfer",
                "source_relations": [["a", "b"], ["b", "c"]],
                "candidate_relations": [["x", "y"], ["y", "z"]],
                "mapping": {"a": "x", "b": "y", "c": "z"},
            }
            target = True
        elif mode == "diagnostic":
            public = {
                "feature": "observed_failure",
                "observed": ["heat", "noise"],
                "causes": {"bearing": ["heat", "noise"], "sensor": ["noise"]},
            }
            target = "bearing"
        else:
            public = {
                "feature": "resource_goal",
                "dependencies": {"inspect": [], "repair": ["inspect"], "verify": ["repair"]},
                "costs": {"inspect": 1, "repair": 2, "verify": 1},
                "budget": 5,
            }
            target = ["inspect", "repair", "verify"]
        public["mode_family"] = mode
        operation = f"{mode} procedure"
    elif family == "scientific_inquiry":
        risk = rng.choice(["low", "high"])
        contradiction = rng.choice([False, True])
        reliability = rng.choice(["stable", "shifted"])
        need = (risk == "high" and reliability == "shifted") or (contradiction and reliability == "shifted")
        public = {
            "risk": risk,
            "contradiction": contradiction,
            "source_reliability": reliability,
            "evidence_cost": 0.70,
            "catastrophic_if_wrong": risk == "high",
        }
        target = "inquire" if need else "stop"
        operation = "contextual expected information value"
    else:
        relational = rng.choice([False, True])
        surface = not relational
        public = {
            "surface_similarity": surface,
            "relational_similarity": relational,
            "evidence_power": rng.choice(["adequate", "underpowered"]),
        }
        target = "transfer" if relational else "reject"
        operation = "adversarial relational check"
    return CognitiveTask(identity, split, seed, family, index, phase, public, target, operation, 1.0)


@lru_cache(maxsize=8192)
def _task_template(seed: int, family: str, index: int, split: str, phase: str) -> CognitiveTask:
    # The cached object is private to this module. generate_task deep-copies
    # every mutable task payload before returning, so histories never share a
    # public packet or target while repeated arm construction avoids rebuilding
    # the same deterministic task six times.
    return _generate_task(seed, family, index, split, phase=phase)


def generate_task(seed: int, family: str, index: int, split: str, *, phase: str = "probe") -> CognitiveTask:
    template = _task_template(seed, family, index, split, phase)
    return CognitiveTask(
        template.identity,
        template.split,
        template.history_seed,
        template.family,
        template.index,
        template.phase,
        _clone_payload(template.public),
        _clone_payload(template.private_target),
        template.oracle_operation,
        template.cost,
    )


def oracle(task: CognitiveTask) -> object:
    return task.private_target


def simple_proposal(task: CognitiveTask) -> object:
    public = task.public
    if task.family == "ontology_garden":
        return "process" if "changing" in public["features"] else "entity"
    if task.family == "epistemic_laboratory":
        return public["reports"]["source_a"]
    if task.family == "causal_micro_worlds":
        if public["query_kind"] == "counterfactual":
            return int(bool(public["background"]["cause"] and public["background"]["context"]))
        return public["observational_effect"]
    if task.family == "cross_representation_systems":
        return public["relations"][0][1]
    if task.family == "reasoning_method_selection":
        return True
    if task.family == "scientific_inquiry":
        return "inquire" if public["contradiction"] else "stop"
    return "transfer" if public["surface_similarity"] else "reject"


def instrument_screen(seeds: tuple[int, ...] = C.SPLITS["construction"]) -> dict:
    rows = {}
    for family in C.WORKLOADS:
        tasks = [generate_task(seed, family, index, "construction") for seed in seeds for index in range(8)]
        observations = [task.observation() for task in tasks]
        oracle_accuracy = sum(task.reveal(oracle(task))["correct"] for task in tasks) / len(tasks)
        simple_accuracy = sum(task.reveal(simple_proposal(task))["correct"] for task in tasks) / len(tasks)
        random_accuracy = sum(
            task.reveal(task.private_target if _seed(task.identity, "random") % 3 == 0 else simple_proposal(task))["correct"] for task in tasks
        ) / len(tasks)
        leakage = any(
            {"target", "private_target", "answer", "oracle_operation"} & set(observation)
            or {"target", "private_target", "answer", "oracle_operation"} & set(observation["public"])
            for observation in observations
        )
        valid = not leakage and oracle_accuracy == 1.0 and simple_accuracy < oracle_accuracy and random_accuracy < oracle_accuracy
        rows[family] = {
            "tasks": len(tasks),
            "answer_leakage": leakage,
            "oracle_accuracy": oracle_accuracy,
            "simple_accuracy": simple_accuracy,
            "random_accuracy": random_accuracy,
            "oracle_headroom": oracle_accuracy - max(simple_accuracy, random_accuracy),
            "floor_clear": random_accuracy > 0.0,
            "ceiling_reachable": oracle_accuracy == 1.0,
            "valid": valid,
            "classification": "instrument_verified" if valid else "invalid_bed",
        }
    return {
        "schema": "substrate-v3-bed-screen/v1",
        "families": rows,
        "all_valid": all(row["valid"] for row in rows.values()),
        "activation": False,
    }


def allocation_cases(seed: int, count: int, workload: str) -> list[dict]:
    rng = random.Random(_seed(seed, workload, count))
    cases = []
    for index in range(count):
        if workload == "no_headroom":
            risk, contradiction, reliability = "low", False, "stable"
            need = False
        else:
            risk = rng.choice(["low", "high"])
            contradiction = rng.choice([False, True])
            reliability = rng.choice(["stable", "shifted"])
            if workload == "transfer":
                need = reliability == "shifted" and ((risk == "high") != contradiction)
            elif workload == "positive_b":
                need = (reliability == "shifted") == ((risk == "high") != contradiction)
            else:
                need = reliability == "shifted" and ((risk == "high") != contradiction)
        cases.append(
            {
                "identity": f"{workload}:{seed}:{index}",
                "context": (risk, contradiction, reliability),
                "risk": risk,
                "contradiction": contradiction,
                "reliability": reliability,
                "need_inquiry": need,
                "cost": 0.70,
            }
        )
    return cases


def allocation_utility(case: dict, action: str) -> float:
    inquire = action == "inquire"
    correct = inquire or not case["need_inquiry"]
    utility = float(correct)
    if inquire:
        utility -= case["cost"]
    if inquire and not case["need_inquiry"]:
        utility -= C.UNNECESSARY_INQUIRY_PENALTY
    if not inquire and case["need_inquiry"]:
        utility -= C.MISSED_INQUIRY_PENALTY
        if case["risk"] == "high":
            utility -= C.CATASTROPHIC_ERROR_PENALTY
    return utility


def allocation_action(policy: str, case: dict, learned: dict[tuple, str] | None = None) -> str:
    if policy == "never_inquire":
        return "stop"
    if policy in {"always_inquire", "maximum_compute"}:
        return "inquire"
    if policy == "contradiction_first":
        return "inquire" if case["contradiction"] else "stop"
    if policy == "risk_threshold":
        return "inquire" if case["risk"] == "high" else "stop"
    if policy == "eiv_threshold":
        return "inquire" if case["reliability"] == "shifted" and (case["risk"] == "high" or case["contradiction"]) else "stop"
    if policy == "random_rate_matched":
        return "inquire" if _seed(case["identity"], policy) % 2 == 0 else "stop"
    if policy == "oracle":
        return "inquire" if case["need_inquiry"] else "stop"
    if policy == "tabular_contextual":
        return (learned or {}).get(case["context"], "stop")
    raise Refused(f"unknown allocation policy {policy!r}")


def fit_tabular(cases: list[dict]) -> dict[tuple, str]:
    contexts = {}
    for case in cases:
        rows = contexts.setdefault(case["context"], [])
        rows.append(case)
    return {
        context: max(
            ("stop", "inquire"),
            key=lambda action: sum(allocation_utility(case, action) for case in rows) / len(rows),
        )
        for context, rows in contexts.items()
    }


def evaluate_allocation(policy: str, cases: list[dict], *, learned: dict[tuple, str] | None = None) -> dict:
    rows = []
    for case in cases:
        action = allocation_action(policy, case, learned)
        rows.append(
            {
                "identity": case["identity"],
                "context": list(case["context"]),
                "action": action,
                "need_inquiry": case["need_inquiry"],
                "utility": allocation_utility(case, action),
                "compute": int(action == "inquire"),
            }
        )
    return {
        "policy": policy,
        "mean_utility": sum(row["utility"] for row in rows) / len(rows),
        "compute": sum(row["compute"] for row in rows),
        "unnecessary_inquiry": sum(row["action"] == "inquire" and not row["need_inquiry"] for row in rows),
        "missed_inquiry": sum(row["action"] == "stop" and row["need_inquiry"] for row in rows),
        "rows": rows,
    }


def allocation_headroom(seed: int, workload: str, count: int = 512) -> dict:
    cases = allocation_cases(seed, count, workload)
    simple_names = (
        "never_inquire",
        "always_inquire",
        "contradiction_first",
        "risk_threshold",
        "eiv_threshold",
        "maximum_compute",
        "random_rate_matched",
    )
    simple = {name: evaluate_allocation(name, cases) for name in simple_names}
    strongest = max(simple, key=lambda name: simple[name]["mean_utility"])
    oracle_row = evaluate_allocation("oracle", cases)
    return {
        "workload": workload,
        "strongest_simple_policy": strongest,
        "strongest_simple_utility": simple[strongest]["mean_utility"],
        "oracle_utility": oracle_row["mean_utility"],
        "oracle_residual": oracle_row["mean_utility"] - simple[strongest]["mean_utility"],
        "simple": simple,
        "oracle": oracle_row,
    }
