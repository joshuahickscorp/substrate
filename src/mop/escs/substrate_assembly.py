
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from mop.substrate.events import canonical_sha256

from .perspective_registry import (
    EffectBoundary,
    IntegrationDisposition,
    PerspectiveCandidate,
    PerspectiveCandidateRegistry,
    TriggerAuthority,
)

SUBSTRATE_ASSEMBLY_SCHEMA = "mop-escs-substrate-assembly/v1"


class SlotMode(StrEnum):
    INFRASTRUCTURE_INERT = "infrastructure-inert"
    FEATURE_CANDIDATE_INERT = "feature-candidate-inert"
    CONTROL_INERT = "control-inert"
    SANDBOX_STUB_INERT = "sandbox-stub-inert"
    EXCLUDED = "excluded"


_MODE_BY_DISPOSITION = {
    IntegrationDisposition.INFRASTRUCTURE: SlotMode.INFRASTRUCTURE_INERT,
    IntegrationDisposition.FEATURE_FLAGGED: SlotMode.FEATURE_CANDIDATE_INERT,
    IntegrationDisposition.CONTROL_ONLY: SlotMode.CONTROL_INERT,
    IntegrationDisposition.SANDBOX_STUB: SlotMode.SANDBOX_STUB_INERT,
    IntegrationDisposition.EXCLUDED: SlotMode.EXCLUDED,
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class PerspectiveSlot:
    candidate_id: str
    candidate_sha256: str
    facet: str
    mode: SlotMode
    interface: str
    effect_boundary: EffectBoundary
    trigger_authority: TriggerAuthority
    required_guards: tuple[str, ...]
    activation_enabled: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.facet or not self.interface:
            raise ValueError("perspective slot identifiers must be nonempty")
        _digest(self.candidate_sha256, "candidate_sha256")
        if not isinstance(self.mode, SlotMode):
            raise ValueError("slot mode must be a SlotMode")
        if not isinstance(self.effect_boundary, EffectBoundary):
            raise ValueError("slot effect boundary must be typed")
        if not isinstance(self.trigger_authority, TriggerAuthority):
            raise ValueError("slot trigger authority must be typed")
        if self.required_guards != tuple(sorted(self.required_guards)):
            raise ValueError("slot guards must be canonically sorted")
        if len(set(self.required_guards)) != len(self.required_guards):
            raise ValueError("slot guards must be unique")
        if self.activation_enabled is not False:
            raise ValueError("Gate-A assembly slots must be disabled")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("an assembly slot cannot grant scientific promotion")
        if self.mode is SlotMode.EXCLUDED and self.effect_boundary is not EffectBoundary.NONE:
            raise ValueError("excluded slots cannot retain an effect boundary")

    @classmethod
    def from_candidate(cls, candidate: PerspectiveCandidate) -> Self:
        return cls(
            candidate_id=candidate.candidate_id,
            candidate_sha256=canonical_sha256(candidate.payload()),
            facet=candidate.facet.value,
            mode=_MODE_BY_DISPOSITION[candidate.integration_disposition],
            interface=candidate.interface.value,
            effect_boundary=candidate.effect_boundary,
            trigger_authority=candidate.trigger_authority,
            required_guards=tuple(sorted(value.value for value in candidate.required_guards)),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "candidate_id",
                "candidate_sha256",
                "facet",
                "mode",
                "interface",
                "effect_boundary",
                "trigger_authority",
                "required_guards",
                "activation_enabled",
                "scientific_promotion_allowed",
            },
            "PerspectiveSlot",
        )
        guards = payload["required_guards"]
        if not isinstance(guards, list) or any(not isinstance(value, str) for value in guards):
            raise ValueError("slot required_guards must be a string list")
        return cls(
            candidate_id=payload["candidate_id"],
            candidate_sha256=payload["candidate_sha256"],
            facet=payload["facet"],
            mode=SlotMode(payload["mode"]),
            interface=payload["interface"],
            effect_boundary=EffectBoundary(payload["effect_boundary"]),
            trigger_authority=TriggerAuthority(payload["trigger_authority"]),
            required_guards=tuple(guards),
            activation_enabled=payload["activation_enabled"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "facet": self.facet,
            "mode": self.mode.value,
            "interface": self.interface,
            "effect_boundary": self.effect_boundary.value,
            "trigger_authority": self.trigger_authority.value,
            "required_guards": list(self.required_guards),
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }


@dataclass(frozen=True, slots=True)
class SubstrateAssembly:
    assembly_id: str
    candidate_registry_sha256: str
    slots: tuple[PerspectiveSlot, ...]
    default_quiescent: bool
    scientific_promotion_allowed: bool
    assembly_sha256: str
    schema: str = SUBSTRATE_ASSEMBLY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUBSTRATE_ASSEMBLY_SCHEMA:
            raise ValueError(f"unsupported substrate assembly schema {self.schema!r}")
        if not self.assembly_id.strip():
            raise ValueError("assembly_id must be nonempty")
        _digest(self.candidate_registry_sha256, "candidate_registry_sha256")
        ids = tuple(slot.candidate_id for slot in self.slots)
        if not self.slots or ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ValueError("assembly slots must be nonempty, unique, and canonically sorted")
        if self.default_quiescent is not True:
            raise ValueError("the Gate-A assembly must default to quiescence")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("the assembly cannot grant scientific promotion")
        _digest(self.assembly_sha256, "assembly_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.assembly_sha256:
            raise ValueError("substrate assembly self-hash mismatch")

    @classmethod
    def create(
        cls,
        registry: PerspectiveCandidateRegistry,
        *,
        assembly_id: str = "escs_all_perspectives_gate_a_v1",
    ) -> Self:
        slots = tuple(
            sorted(
                (PerspectiveSlot.from_candidate(candidate) for candidate in registry.candidates),
                key=lambda slot: slot.candidate_id,
            )
        )
        core = {
            "schema": SUBSTRATE_ASSEMBLY_SCHEMA,
            "assembly_id": assembly_id,
            "candidate_registry_sha256": registry.sha256,
            "slots": [slot.payload() for slot in slots],
            "default_quiescent": True,
            "scientific_promotion_allowed": False,
        }
        return cls(
            assembly_id=assembly_id,
            candidate_registry_sha256=registry.sha256,
            slots=slots,
            default_quiescent=True,
            scientific_promotion_allowed=False,
            assembly_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "schema",
                "assembly_id",
                "candidate_registry_sha256",
                "slots",
                "default_quiescent",
                "scientific_promotion_allowed",
                "assembly_sha256",
            },
            "SubstrateAssembly",
        )
        slots = payload["slots"]
        if not isinstance(slots, list):
            raise ValueError("assembly slots must be a list")
        return cls(
            schema=payload["schema"],
            assembly_id=payload["assembly_id"],
            candidate_registry_sha256=payload["candidate_registry_sha256"],
            slots=tuple(PerspectiveSlot.from_payload(row) for row in slots),
            default_quiescent=payload["default_quiescent"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            assembly_sha256=payload["assembly_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "assembly_id": self.assembly_id,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "slots": [slot.payload() for slot in self.slots],
            "default_quiescent": self.default_quiescent,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["assembly_sha256"] = self.assembly_sha256
        return result

    def validate_registry(self, registry: PerspectiveCandidateRegistry) -> tuple[str, ...]:
        problems: list[str] = []
        expected = type(self).create(registry, assembly_id=self.assembly_id)
        if self.candidate_registry_sha256 != registry.sha256:
            problems.append("candidate-registry-authority-mismatch")
        if self.slots != expected.slots:
            problems.append("candidate-slot-projection-mismatch")
        return tuple(problems)


def load_substrate_assembly(path: str | Path) -> SubstrateAssembly:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("substrate assembly artifact must be a JSON object")
    return SubstrateAssembly.from_payload(payload)
