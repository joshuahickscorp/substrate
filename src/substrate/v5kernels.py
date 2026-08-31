"""Bounded, executable kernel candidates for Substrate v5.

These candidates deliberately share a small test contract.  The comparison is
not a claim that a micro-benchmark proves general cognition; it is the
construction-stage authority for choosing the v5 state kernel used by later
sensorium experiments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any


class Refused(RuntimeError):
    """A kernel event or checkpoint violated the bounded candidate contract."""


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KernelEvent:
    sequence: int
    modality: str
    kind: str
    subject: str
    value: Any
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise Refused("event sequence must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise Refused("event confidence must be in [0, 1]")


class Kernel:
    """Common persistence, replacement, and checkpoint contract."""

    name = "base"
    architecture = "base"
    explicit = False
    latent = False
    event_sourced = False
    actors = False

    def __init__(self, identity: str = "substrate-v5") -> None:
        self.identity = identity
        self.model_identity = "model-alpha"
        self.last_sequence = -1
        self.objects: dict[str, dict[str, Any]] = {}
        self.goals: dict[str, str] = {}
        self.events: list[dict[str, Any]] = []
        self.latent_state = [0.0, 0.0, 0.0, 0.0]
        self.messages: dict[str, list[dict[str, Any]]] = {}

    def _validate(self, event: KernelEvent) -> None:
        if event.sequence <= self.last_sequence:
            raise Refused("event sequence must increase monotonically")
        self.last_sequence = event.sequence

    def _latent_update(self, event: KernelEvent) -> None:
        modality_digest = hashlib.sha256(event.modality.encode()).digest()
        index = int.from_bytes(modality_digest[:2], "big")
        index %= len(self.latent_state)
        signal_digest = hashlib.sha256(f"{event.kind}:{event.subject}".encode()).digest()
        signal = (
            int.from_bytes(signal_digest[:4], "big")
            / 0xFFFFFFFF
        )
        self.latent_state[index] = 0.7 * self.latent_state[index] + 0.3 * signal

    def apply(self, event: KernelEvent) -> None:
        raise NotImplementedError

    def replace_model(self, new_identity: str) -> None:
        if not new_identity:
            raise Refused("replacement model identity is required")
        self.model_identity = new_identity

    def checkpoint(self) -> dict[str, Any]:
        body = {
            "kernel": self.name,
            "identity": self.identity,
            "model_identity": self.model_identity,
            "last_sequence": self.last_sequence,
            "objects": self.objects,
            "goals": self.goals,
            "events": self.events,
            "latent_state": self.latent_state,
            "messages": self.messages,
            "activation": False,
        }
        return {"body": copy.deepcopy(body), "sha256": _digest(body)}

    def restore(self, checkpoint: dict[str, Any]) -> Kernel:
        body = checkpoint.get("body")
        if not isinstance(body, dict) or checkpoint.get("sha256") != _digest(body):
            raise Refused("corrupted kernel checkpoint")
        if body.get("kernel") != self.name or body.get("activation") is not False:
            raise Refused("checkpoint kernel or activation mismatch")
        self.identity = str(body["identity"])
        self.model_identity = str(body["model_identity"])
        self.last_sequence = int(body["last_sequence"])
        self.objects = copy.deepcopy(body["objects"])
        self.goals = copy.deepcopy(body["goals"])
        self.events = copy.deepcopy(body["events"])
        self.latent_state = [float(value) for value in body["latent_state"]]
        self.messages = copy.deepcopy(body["messages"])
        return self


class ExtendedV4Kernel(Kernel):
    name = "candidate_a_extended_v4"
    architecture = "extended_explicit_runtime"
    explicit = True

    def apply(self, event: KernelEvent) -> None:
        self._validate(event)
        if event.kind == "observation":
            self.objects[event.subject] = {
                "visible": bool(event.value),
                "last_sequence": event.sequence,
                "source_modality": event.modality,
            }
        elif event.kind == "goal":
            self.goals[event.subject] = str(event.value)


class EventSourcedGraphKernel(Kernel):
    name = "candidate_b_event_sourced_graph"
    architecture = "event_sourced_cognitive_graph"
    explicit = True
    event_sourced = True

    def apply(self, event: KernelEvent) -> None:
        self._validate(event)
        row = {
            "sequence": event.sequence,
            "modality": event.modality,
            "kind": event.kind,
            "subject": event.subject,
            "value": event.value,
            "confidence": event.confidence,
        }
        self.events.append(row)
        if event.kind == "observation":
            prior = self.objects.get(event.subject, {})
            self.objects[event.subject] = {
                **prior,
                "visible": bool(event.value),
                "last_sequence": event.sequence,
                "source_modality": event.modality,
                "evidence_sequences": [
                    *prior.get("evidence_sequences", []),
                    event.sequence,
                ],
            }
        elif event.kind == "goal":
            self.goals[event.subject] = str(event.value)


class RecurrentLatentKernel(Kernel):
    name = "candidate_c_recurrent_latent"
    architecture = "bounded_recurrent_latent_state"
    latent = True

    def apply(self, event: KernelEvent) -> None:
        self._validate(event)
        self._latent_update(event)
        if event.kind == "goal":
            self.goals[event.subject] = str(event.value)
        if event.kind == "observation" and bool(event.value):
            self.objects[event.subject] = {
                "visible": True,
                "last_sequence": event.sequence,
            }
        elif event.kind == "observation":
            self.objects.pop(event.subject, None)


class HybridKernel(EventSourcedGraphKernel):
    name = "candidate_d_hybrid_explicit_latent"
    architecture = "event_sourced_explicit_latent_substrate"
    latent = True

    def apply(self, event: KernelEvent) -> None:
        super().apply(event)
        self._latent_update(event)
        if event.kind == "correction" and event.subject in self.objects:
            self.objects[event.subject]["corrected_value"] = event.value
            self.objects[event.subject]["correction_sequence"] = event.sequence


class ActorKernel(EventSourcedGraphKernel):
    name = "candidate_e_actor_cells"
    architecture = "typed_actor_nervous_system"
    actors = True

    def apply(self, event: KernelEvent) -> None:
        super().apply(event)
        organ = {
            "image": "vision",
            "video": "vision",
            "motion": "motion",
            "audio": "audio",
            "speech": "language",
            "depth": "spatial",
            "body": "body",
        }.get(event.modality, "workspace")
        self.messages.setdefault(organ, []).append(
            {
                "sequence": event.sequence,
                "kind": event.kind,
                "subject": event.subject,
            }
        )


CANDIDATES = (
    ExtendedV4Kernel,
    EventSourcedGraphKernel,
    RecurrentLatentKernel,
    HybridKernel,
    ActorKernel,
)


def fixture() -> tuple[KernelEvent, ...]:
    """A shared positive construction fixture with no hidden answer field."""

    return (
        KernelEvent(0, "text", "goal", "unfinished-inspection", "active"),
        KernelEvent(1, "image", "observation", "object-red", True),
        KernelEvent(2, "motion", "observation", "object-red", True),
        KernelEvent(3, "video", "observation", "object-red", False),
        KernelEvent(4, "audio", "observation", "source-bell", True),
        KernelEvent(5, "depth", "observation", "surface-table", True),
        KernelEvent(6, "body", "observation", "camera-pose", True),
        KernelEvent(7, "speech", "correction", "object-red", "object-crimson"),
    )


def evaluate(kernel_type: type[Kernel], *, iterations: int = 64) -> dict[str, Any]:
    if iterations < 1:
        raise Refused("kernel benchmark requires a positive iteration count")
    start = time.perf_counter_ns()
    last: Kernel | None = None
    for index in range(iterations):
        kernel = kernel_type(identity=f"entity-{index}")
        for event in fixture():
            kernel.apply(event)
        kernel.replace_model("model-beta")
        checkpoint = kernel.checkpoint()
        restored = kernel_type().restore(checkpoint)
        if checkpoint["body"]["identity"] != restored.identity:
            raise Refused("kernel replacement changed entity identity")
        last = restored
    elapsed = max(1, time.perf_counter_ns() - start)
    assert last is not None
    object_row = last.objects.get("object-red", {})
    checks = {
        "identity_persistence": last.identity == f"entity-{iterations - 1}",
        "unfinished_goal": last.goals.get("unfinished-inspection") == "active",
        "object_permanence": "object-red" in last.objects,
        "occlusion_state": object_row.get("visible") is False,
        "model_replacement": last.model_identity == "model-beta",
        "checkpoint_restore": last.checkpoint()["body"]["last_sequence"] == 7,
        "multimodal_coverage": len({event.modality for event in fixture()}) >= 6,
        "explicit_provenance": bool(object_row.get("evidence_sequences")),
        "latent_transition": any(abs(value) > 0.0 for value in last.latent_state),
        "explicit_latent_sync": bool(object_row.get("correction_sequence"))
        and any(abs(value) > 0.0 for value in last.latent_state),
        "typed_actor_messages": bool(last.messages),
    }
    weights = {
        "identity_persistence": 1.0,
        "unfinished_goal": 1.0,
        "object_permanence": 1.0,
        "occlusion_state": 1.0,
        "model_replacement": 1.0,
        "checkpoint_restore": 1.0,
        "multimodal_coverage": 1.0,
        "explicit_provenance": 1.5,
        "latent_transition": 1.0,
        "explicit_latent_sync": 2.0,
        "typed_actor_messages": 0.25,
    }
    mechanism_utility = sum(weights[name] for name, passed in checks.items() if passed)
    maximum = sum(weights.values())
    events = iterations * len(fixture())
    return {
        "candidate": kernel_type.name,
        "architecture": kernel_type.architecture,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "mechanism_utility": mechanism_utility / maximum,
        "events": events,
        "elapsed_ns": elapsed,
        "events_per_second": events / (elapsed / 1_000_000_000),
        "checkpoint_bytes": len(
            json.dumps(last.checkpoint(), sort_keys=True, separators=(",", ":"))
        ),
        "activation": False,
    }


def benchmark(*, iterations: int = 64) -> dict[str, Any]:
    rows = [evaluate(candidate, iterations=iterations) for candidate in CANDIDATES]
    fastest = max(float(row["events_per_second"]) for row in rows)
    smallest = min(int(row["checkpoint_bytes"]) for row in rows)
    for row in rows:
        throughput = float(row["events_per_second"]) / fastest
        compactness = smallest / int(row["checkpoint_bytes"])
        row["integrated_utility"] = (
            0.86 * float(row["mechanism_utility"])
            + 0.08 * math.sqrt(throughput)
            + 0.06 * compactness
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["integrated_utility"]),
            float(row["mechanism_utility"]),
            -int(row["checkpoint_bytes"]),
            str(row["candidate"]),
        ),
    )
    return {
        "schema": "substrate-v5-kernel-benchmark/v1",
        "shared_fixture_digest": _digest(
            [event.__dict__ for event in fixture()]
        ),
        "iterations": iterations,
        "candidates": rows,
        "selected": selected["candidate"],
        "selection_rule": (
            "maximum integrated mechanism/auditability/throughput/compactness utility; "
            "throughput alone cannot select a kernel"
        ),
        "activation": False,
    }
