"""Typed, activation-safe interfaces for the next real-world sandbox campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from substrate import final_revision_io as io


@dataclass(frozen=True)
class TaskRequest:
    task_id: str
    task_kind: str
    objective: str
    consent_receipt: str
    allowed_tools: tuple[str, ...]
    resource_budget: dict[str, float]
    private_fields: tuple[str, ...] = ()
    activation: bool = False

    def validate(self) -> None:
        if self.activation is not False:
            raise io.Refused("external activation must remain false")
        if not self.task_id or not self.objective or not self.consent_receipt:
            raise io.Refused("task identity, objective, and consent receipt are required")
        if any(value < 0 for value in self.resource_budget.values()):
            raise io.Refused("resource budgets must be nonnegative")


@dataclass(frozen=True)
class SensorPacket:
    packet_id: str
    modality: str
    logical_time: int
    content_digest: str
    provenance: str
    features: dict[str, Any]
    consent_receipt: str

    def validate(self) -> None:
        if not all((self.packet_id, self.modality, self.content_digest, self.provenance, self.consent_receipt)):
            raise io.Refused("sensor packet omits a required provenance or consent field")
        if self.logical_time < 0:
            raise io.Refused("sensor logical time must be nonnegative")


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    task_id: str
    tool: str
    arguments: dict[str, Any]
    expected_effect: str
    reversibility: str
    requires_operator_approval: bool = True
    external_execution_authorized: bool = False

    def validate(self) -> None:
        if self.external_execution_authorized is not False:
            raise io.Refused("this package does not authorize external execution")
        if not self.requires_operator_approval:
            raise io.Refused("real-world actions require an operator approval gate")
        if self.reversibility not in {"read_only", "reversible", "destructive"}:
            raise io.Refused("unknown reversibility class")


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    role: str
    input_digest: str
    context: dict[str, Any]
    output_schema: dict[str, Any]
    cost_limit: float


@dataclass(frozen=True)
class ModelResponse:
    request_id: str
    model_identity: str
    output: dict[str, Any]
    output_digest: str
    latency_seconds: float
    cost: float
    supported: bool
    limitations: tuple[str, ...] = ()


@runtime_checkable
class ModelAdapter(Protocol):
    identity: str

    def infer(self, request: ModelRequest) -> ModelResponse: ...

    def health(self) -> dict[str, Any]: ...


@dataclass
class SandboxReceipt:
    task_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    external_actions_executed: int = 0
    activation: bool = False

    def append(self, event: dict[str, Any]) -> None:
        if io.contains_true_activation(event):
            raise io.Refused("receipt event attempts to enable activation")
        self.events.append(dict(event))

    def document(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "events": list(self.events),
            "failures": list(self.failures),
            "external_actions_executed": self.external_actions_executed,
            "activation": self.activation,
            "sha256": io.digest(
                {
                    "task_id": self.task_id,
                    "events": self.events,
                    "failures": self.failures,
                    "external_actions_executed": self.external_actions_executed,
                    "activation": self.activation,
                }
            ),
        }


def bounded_smoke() -> dict[str, Any]:
    task = TaskRequest(
        task_id="readiness-smoke",
        task_kind="offline_document",
        objective="validate interfaces without executing a real-world action",
        consent_receipt=io.digest("operator-consent-fixture"),
        allowed_tools=("read_only_fixture",),
        resource_budget={"seconds": 1.0, "model_calls": 0.0},
    )
    sensor = SensorPacket(
        packet_id="packet-1",
        modality="filesystem",
        logical_time=1,
        content_digest=io.digest(b"fixture"),
        provenance="readiness://controlled-fixture",
        features={"exists": True, "bytes": 7},
        consent_receipt=task.consent_receipt,
    )
    action = ActionProposal(
        action_id="proposal-1",
        task_id=task.task_id,
        tool="read_only_fixture",
        arguments={"path": "controlled-fixture"},
        expected_effect="read a bounded fixture",
        reversibility="read_only",
    )
    task.validate()
    sensor.validate()
    action.validate()
    receipt = SandboxReceipt(task_id=task.task_id)
    receipt.append({"kind": "sensor", "packet": asdict(sensor), "activation": False})
    receipt.append({"kind": "proposal", "proposal": asdict(action), "executed": False, "activation": False})
    return {
        "task": asdict(task),
        "sensor": asdict(sensor),
        "action_proposal": asdict(action),
        "receipt": receipt.document(),
        "external_action_executed": False,
        "all_pass": True,
        "activation": False,
    }
