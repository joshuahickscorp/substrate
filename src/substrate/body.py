"""The model body interface.

Section 13 wants the same Substrate to inhabit different bodies, which only means anything if the contract
is explicit enough that a body can fail it. Nine message kinds, six integration modes, and a conformance
check that names exactly which kinds a candidate body does not implement.

The point of running the same Substrate against a compact specialist, a larger general model and a tool
dominant system is to find out how much of the intelligence has to live in weights rather than in state,
memory, tools and cognitive organization. That comparison is not run here. What is here is the contract it
would need, and the conformance report says plainly that no body has been attached.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

MESSAGE_KINDS = (
    "inference",
    "hidden_state",
    "selected_activations",
    "tool_request",
    "memory_request",
    "verification_request",
    "adaptation_proposal",
    "resource_report",
    "checkpoint",
)

INTEGRATION_MODES = (
    "external_cognitive_shell",
    "sidecar_temporal_core",
    "adapter_layer",
    "internal_recurrent_layer",
    "multi_model_orchestrator",
    "end_to_end_owned_model",
)

TEST_BODIES = ("compact_specialist", "larger_general_model", "tool_dominant_system")

# what each message kind must carry for the message to mean anything
REQUIRED_FIELDS = {
    "inference": ("input", "output", "seed"),
    "hidden_state": ("layer", "shape", "provenance"),
    "selected_activations": ("selector", "shape", "provenance"),
    "tool_request": ("tool", "arguments", "cost"),
    "memory_request": ("store", "query", "permitted_regions"),
    "verification_request": ("claim", "method"),
    "adaptation_proposal": tuple(("information_used", "affected_state", "reversibility", "cost", "risk", "verification", "rollback")),
    "resource_report": ("wall_seconds", "peak_memory", "budget_remaining"),
    "checkpoint": ("identity", "sha256"),
}


class Refused(RuntimeError):
    """A body message the contract does not accept."""


@dataclass
class BodyContract:
    name: str
    mode: str
    implements: tuple[str, ...]

    def violations(self) -> list[str]:
        v = []
        if self.mode not in INTEGRATION_MODES:
            v.append(f"{self.name}: unknown integration mode {self.mode!r}")
        for kind in self.implements:
            if kind not in MESSAGE_KINDS:
                v.append(f"{self.name}: {kind!r} is not a declared message kind")
        return v

    def missing(self) -> list[str]:
        return [k for k in MESSAGE_KINDS if k not in self.implements]


def _absent(value) -> bool:
    """Membership testing against a tuple calls __eq__, which a numpy array answers elementwise and then
    refuses to reduce to one truth value. A body returning a prediction array made the validator raise
    instead of validating. Absent means None or the empty string, and nothing else."""
    return value is None or (isinstance(value, str) and not value)


def validate_message(kind: str, message: dict) -> list[str]:
    if kind not in MESSAGE_KINDS:
        raise Refused(f"unknown message kind {kind!r}")
    return [f"{kind}: {field} not supplied" for field in REQUIRED_FIELDS[kind] if _absent(message.get(field))]


def conformance(contract: BodyContract) -> dict:
    missing = contract.missing()
    return {
        "body": contract.name,
        "mode": contract.mode,
        "implements": sorted(contract.implements),
        "missing": missing,
        "declaration_violations": contract.violations(),
        "conforms": not missing and not contract.violations(),
        "partial": bool(missing) and not contract.violations(),
    }


def declaration(bodies: list[BodyContract] | None = None) -> dict:
    bodies = bodies or []
    reports = [conformance(b) for b in bodies]
    return {
        "schema": "substrate-model-body-interface/v1",
        "message_kinds": list(MESSAGE_KINDS),
        "required_fields": {k: list(v) for k, v in REQUIRED_FIELDS.items()},
        "integration_modes": list(INTEGRATION_MODES),
        "bodies_to_test": list(TEST_BODIES),
        "why_three_bodies": (
            "running the same Substrate against a compact specialist, a larger general "
            "model and a tool dominant system measures how much intelligence must live "
            "in weights rather than in state, memory, tools and organization"
        ),
        "attached_bodies": reports,
        "any_body_attached": bool(reports),
        "honest_state": (
            "no model body is attached. The contract exists and is testable; the comparison it would support has not been run and no result is claimed for it"
        ),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    # SUBSTRATE_MODEL_BODY_INTERFACE.json is owned by substrate.bodies, which attaches three real
    # bodies and measures the ablation ladder. This module declares the contract they conform to.
    doc = declaration()
    print(
        json.dumps(
            {
                "contract_only": True,
                "producer": "substrate.bodies",
                "message_kinds": len(doc["message_kinds"]),
                "any_body_attached": doc["any_body_attached"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
