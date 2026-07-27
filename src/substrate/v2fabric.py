"""Deterministic delayed developmental task fabric.

Observations contain available alternatives and preoutcome context.  Targets are private fields returned
only by ``reveal`` after a proposal is committed.  All tools are pure functions and external activation is
absent.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from substrate import v2config as C
from substrate import v2io as io

OPERATIONS = (
    "boundary_route",
    "reverse_boundary_route",
    "always_first",
    "always_second",
    "risk_route",
    "reverse_risk_route",
)


@dataclass(frozen=True)
class Task:
    identity: str
    seed: int
    split: str
    domain: str
    index: int
    observation: dict
    alternatives: tuple[str, ...]
    private_target: str
    required_operation: str
    task_signature: str
    phase: str

    def public(self) -> dict:
        return {
            "identity": self.identity,
            "domain": self.domain,
            "observation": self.observation,
            "alternatives": list(self.alternatives),
            "task_signature": self.task_signature,
            "phase": self.phase,
        }

    def reveal(self, proposal: str) -> dict:
        receipt = {
            "task": self.identity,
            "proposal": proposal,
            "target": self.private_target,
            "correct": proposal == self.private_target,
            "required_operation": self.required_operation,
            "revealed_after_commitment": True,
            "activation": False,
        }
        receipt["verification_digest"] = io.sha_obj(receipt)
        return receipt


def split_for_seed(seed: int) -> str:
    matches = [name for name, values in C.SPLITS.items() if seed in values]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} belongs to {len(matches)} splits")
    return matches[0]


def execute(operation: str, observation: dict, alternatives: tuple[str, ...]) -> str:
    def choice(index: int) -> str:
        # The scored action is not one of the visible alternatives.  The observation exposes candidate
        # surfaces, while the delayed target is an internal selection action revealed only after commit.
        if index >= len(alternatives):
            raise ValueError("operation selected an unavailable alternative")
        return f"select_position_{index}"

    if operation == "boundary_route":
        condition = bool(observation.get("boundary", observation.get("exception", False)))
        return choice(0 if condition else 1)
    if operation == "reverse_boundary_route":
        condition = bool(observation.get("boundary", observation.get("exception", False)))
        return choice(1 if condition else 0)
    if operation == "always_first":
        return choice(0)
    if operation == "always_second":
        return choice(1)
    if operation == "risk_route":
        risk = float(observation.get("risk", observation.get("failure_risk", 0.0)))
        limitation = bool(observation.get("contradiction", observation.get("known_limitation", False)))
        return choice(1 if risk >= 0.55 or limitation else 0)
    if operation == "reverse_risk_route":
        risk = float(observation.get("risk", observation.get("failure_risk", 0.0)))
        limitation = bool(observation.get("contradiction", observation.get("known_limitation", False)))
        return choice(0 if risk >= 0.55 or limitation else 1)
    raise ValueError(f"unknown operation {operation!r}")


def _opaque(prefix: str, seed: int, index: int, count: int) -> tuple[str, ...]:
    values = []
    for position in range(count):
        digest = hashlib.sha256(f"{prefix}:{seed}:{index}:{position}".encode()).hexdigest()[:6]
        values.append(f"{prefix}{digest}")
    return tuple(values)


def generate_task(seed: int, domain: str, index: int, phase: str = "development") -> Task:
    split = split_for_seed(seed)
    if domain not in C.DOMAIN_CATALOG:
        raise ValueError(f"unknown domain {domain!r}")
    rng = random.Random(f"substrate-v2:{seed}:{domain}:{index}:{phase}")
    identity = f"{split}:{seed}:{domain}:{phase}:{index}"
    if domain == "A":
        alternatives = _opaque("tok", seed, index, 3)
        observation = {
            "tokens": list(alternatives),
            "boundary": rng.random() < 0.5,
            "context": f"seq{rng.randrange(7)}",
            "distractor": rng.randrange(1000),
        }
    elif domain == "B":
        alternatives = _opaque("glyph", seed + 17, index, 3)
        observation = {
            "glyphs": list(alternatives),
            "exception": rng.random() < 0.5,
            "rule_context": f"rule{rng.randrange(7)}",
            "distractor": rng.choice(("circle", "square", "triangle")),
        }
    elif domain == "C":
        alternatives = _opaque("source", seed + 29, index, 2)
        observation = {
            "sources": list(alternatives),
            "risk": round(rng.random(), 6),
            "contradiction": rng.random() < 0.25,
            "budget": 1.0,
            "distractor": rng.randrange(1000),
        }
    else:
        alternatives = _opaque("tool", seed + 43, index, 2)
        observation = {
            "tools": list(alternatives),
            "failure_risk": round(rng.random(), 6),
            "known_limitation": rng.random() < 0.25,
            "budget": 1.0,
            "distractor": rng.choice(("lookup", "transform", "compare")),
        }
    spec = C.DOMAIN_CATALOG[domain]
    target = execute(spec["required_operation"], observation, alternatives)
    return Task(
        identity=identity,
        seed=seed,
        split=split,
        domain=domain,
        index=index,
        observation=observation,
        alternatives=alternatives,
        private_target=target,
        required_operation=spec["required_operation"],
        task_signature=spec["task_signature"],
        phase=phase,
    )


def leakage(task: Task) -> dict:
    serialized = json.dumps(task.public(), sort_keys=True)
    target_digest = hashlib.sha256(task.private_target.encode()).hexdigest()
    return {
        "target_key_absent": "private_target" not in serialized and '"target"' not in serialized,
        "target_value_absent": task.private_target not in serialized,
        "target_digest_absent": target_digest not in serialized,
        "passes": (
            "private_target" not in serialized
            and '"target"' not in serialized
            and task.private_target not in serialized
            and target_digest not in serialized
        ),
    }


def screen(seeds: tuple[int, ...] | None = None, per_domain: int = 64) -> dict:
    seeds = seeds or C.SPLITS["development"]
    rows = {}
    all_ids: set[str] = set()
    collisions = []
    for domain in C.DOMAIN_CATALOG:
        tasks = [generate_task(seed, domain, index, "screen") for seed in seeds for index in range(per_domain)]
        for task in tasks:
            if task.identity in all_ids:
                collisions.append(task.identity)
            all_ids.add(task.identity)
        oracle = sum(execute(task.required_operation, task.observation, task.alternatives) == task.private_target for task in tasks) / len(tasks)
        first = sum(execute("always_first", task.observation, task.alternatives) == task.private_target for task in tasks) / len(tasks)
        second = sum(execute("always_second", task.observation, task.alternatives) == task.private_target for task in tasks) / len(tasks)
        random_expected = 1.0 / len(tasks[0].alternatives)
        headroom = oracle - max(first, second, random_expected)
        leaks = [task.identity for task in tasks if not leakage(task)["passes"]]
        rows[domain] = {
            "n_tasks": len(tasks),
            "oracle_accuracy": oracle,
            "simple_first_accuracy": first,
            "simple_second_accuracy": second,
            "random_expected_accuracy": random_expected,
            "best_nonoracle_accuracy": max(first, second, random_expected),
            "oracle_headroom": headroom,
            "floor_clear": max(first, second) > 0.0,
            "not_saturated": max(first, second) < 0.95,
            "headroom_above_sesoi": headroom > C.SESOI,
            "answer_leakage_failures": leaks,
            "valid": not leaks and max(first, second) < 0.95 and headroom > C.SESOI,
        }
    split_sets = {name: set(values) for name, values in C.SPLITS.items()}
    split_pairs = {
        f"{left}:{right}": sorted(split_sets[left] & split_sets[right])
        for i, left in enumerate(split_sets)
        for right in tuple(split_sets)[i + 1 :]
    }
    return {
        "schema": "substrate-v2-bed-screen/v1",
        "domains": rows,
        "task_identity_collisions": collisions,
        "split_seed_overlap": split_pairs,
        "all_valid": all(row["valid"] for row in rows.values())
        and not collisions
        and not any(split_pairs.values()),
        "activation": False,
    }
