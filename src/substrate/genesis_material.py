"""The cognitive material interface every genesis arm implements.

The single most important property of this module is that a material can never
read a held-out label. A material sees observations and probes; it proposes
durable changes; an evaluator that lives outside the material returns a scalar
verdict. Expected answers are held by the evaluator and are never handed to,
stored on, or serialized by the material.

The second property is that every arm is a separate implementation. A material
declares an exclusive mechanism identifier and the distinctness gate refuses a
tournament in which two arms produce the same durable-state trace under an
identical probe sequence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from substrate import genesis_config as C

# --------------------------------------------------------------------------
# Values carried between the world, the material and the evaluator
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """One sensory or teaching event delivered to every arm identically."""

    index: int
    channel: str
    payload: tuple[int, ...]
    elapsed_ms: int = 0
    teaching: bool = False
    modality: str = "symbolic"

    def digest(self) -> str:
        return _digest((self.index, self.channel, self.payload, self.elapsed_ms, self.teaching, self.modality))


@dataclass(frozen=True, slots=True)
class Probe:
    """A question. It never carries its own answer."""

    index: int
    family: str
    channel: str
    probe: tuple[int, ...]
    arity: int

    def digest(self) -> str:
        return _digest((self.index, self.family, self.channel, self.probe, self.arity))


@dataclass(frozen=True, slots=True)
class Answer:
    """A material's response. ``confidence`` is a low-bit calibrated integer."""

    probe_index: int
    value: tuple[int, ...]
    confidence: int = 0
    abstained: bool = False


@dataclass(frozen=True, slots=True)
class Proposal:
    """A candidate durable change. Nothing here is committed until verified."""

    proposal_id: str
    kind: str
    target: str
    delta: tuple[int, ...] = ()
    precision_request: str | None = None
    topology_operation: str | None = None
    trigger: str = ""
    expected_value: float = 0.0
    cost_bytes: int = 0


@dataclass(frozen=True, slots=True)
class Verdict:
    """The only evaluation signal a material ever receives.

    ``improvement`` and ``retention`` are scalars computed by the evaluator from
    sealed cases the material cannot see. ``admitted`` is the evaluator's
    decision. No expected value, case identity or label is carried here.
    """

    proposal_id: str
    admitted: bool
    improvement: float
    retention: float


@dataclass(frozen=True, slots=True)
class Receipt:
    """An append-only record of one committed or rolled back durable change."""

    proposal_id: str
    kind: str
    target: str
    committed: bool
    improvement: float
    retention: float
    durable_state_digest_before: str
    durable_state_digest_after: str
    cost_bytes: int
    mechanism: str


class ResourceExhausted(RuntimeError):
    """An arm tried to spend more than its equal share of a budget."""


@dataclass
class ResourceLedger:
    """Measured cost. Every arm receives an identical ledger for a unit.

    Budgets are enforced, not advisory: exceeding one raises rather than
    silently letting an arm buy an advantage.
    """

    envelope: str
    operation_budget: int
    durable_write_budget: int
    byte_budget: int | None
    operations: int = 0
    durable_writes: int = 0
    resident_bytes: int = 0
    peak_resident_bytes: int = 0
    retrievals: int = 0
    precision_bits: int = 0
    checkpoints: int = 0
    restores: int = 0

    def spend(self, operations: int = 1, *, retrievals: int = 0, precision_bits: int = 0) -> None:
        self.operations += operations
        self.retrievals += retrievals
        self.precision_bits += precision_bits
        if self.operations > self.operation_budget:
            raise ResourceExhausted(f"operation budget {self.operation_budget} exceeded in envelope {self.envelope}")

    def durable_write(self) -> None:
        self.durable_writes += 1
        if self.durable_writes > self.durable_write_budget:
            raise ResourceExhausted(f"durable write budget {self.durable_write_budget} exceeded")

    def resize(self, resident_bytes: int) -> None:
        self.resident_bytes = resident_bytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, resident_bytes)
        if self.byte_budget is not None and resident_bytes > self.byte_budget:
            raise ResourceExhausted(f"resident byte budget {self.byte_budget} exceeded in envelope {self.envelope}")

    def measurement(self) -> dict[str, int]:
        return {
            "compute": self.operations,
            "plasticity": self.durable_writes,
            "memory": self.peak_resident_bytes,
            "persistence": self.checkpoints + self.restores,
            "retrieval_cost": self.retrievals,
            "precision_bits": self.precision_bits,
        }


@dataclass
class Opportunity:
    """The equal-opportunity vector handed to every arm for one unit.

    ``deprived`` names the single opportunity a control is defined by lacking.
    An arm with an empty ``deprived`` tuple has received everything.
    """

    observation_digest: str
    sensor_channels: tuple[str, ...]
    teaching_digest: str
    ledger: ResourceLedger
    plasticity_enabled: bool = True
    persistence_enabled: bool = True
    deprived: tuple[str, ...] = ()

    def vector(self) -> dict[str, Any]:
        return {
            "information": self.observation_digest,
            "sensors": tuple(self.sensor_channels),
            "teaching": self.teaching_digest,
            "compute": self.ledger.operation_budget,
            "plasticity": self.ledger.durable_write_budget if self.plasticity_enabled else 0,
            "persistence": self.persistence_enabled,
            "memory": self.ledger.byte_budget,
        }


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


@runtime_checkable
class CognitiveMaterial(Protocol):
    """What every candidate, control and baseline must implement."""

    name: str
    mechanism: str

    def opportunity(self) -> Opportunity: ...

    def observe(self, observation: Observation) -> None:
        """Update active state only. This must never write durable state."""

    def answer(self, probe: Probe) -> Answer:
        """Answer a probe. Read only: no durable write, no proposal admission."""

    def propose(self) -> Sequence[Proposal]:
        """Offer durable changes justified by observation, never by evaluation."""

    def apply(self, verdicts: Sequence[Verdict]) -> Sequence[Receipt]:
        """Commit admitted proposals, roll back the rest, and emit receipts."""

    def durable_state_digest(self) -> str: ...

    def active_state_digest(self) -> str: ...

    def checkpoint(self) -> dict[str, Any]: ...

    def restore(self, checkpoint: Mapping[str, Any]) -> None: ...

    def freeze_mechanism(self, mechanism: str) -> None:
        """Disable a mechanism this arm does not exclusively own."""

    def cost(self) -> dict[str, int]: ...


# --------------------------------------------------------------------------
# Shared plumbing.  Concrete materials subclass this; the distinct mechanism
# lives entirely in ``_transition``, ``_propose`` and ``_commit``.
# --------------------------------------------------------------------------


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class MaterialBase:
    """Bookkeeping every material shares, and nothing that decides a result."""

    name: str
    mechanism: str
    _opportunity: Opportunity
    frozen_mechanisms: set[str] = field(default_factory=set)
    receipts: list[Receipt] = field(default_factory=list)
    pending: dict[str, Proposal] = field(default_factory=dict)
    observations_seen: int = 0
    elapsed_ms: int = 0

    # -- required by the protocol -----------------------------------------

    def opportunity(self) -> Opportunity:
        return self._opportunity

    def cost(self) -> dict[str, int]:
        return self._opportunity.ledger.measurement()

    def freeze_mechanism(self, mechanism: str) -> None:
        owned = C.EXCLUSIVE_MECHANISMS.get(self.name, ())
        if mechanism in owned:
            raise ValueError(f"{self.name} may not freeze the mechanism it owns: {mechanism}")
        self.frozen_mechanisms.add(mechanism)

    def durable_state_digest(self) -> str:
        return _digest(self._durable_state())

    def active_state_digest(self) -> str:
        return _digest(self._active_state())

    def checkpoint(self) -> dict[str, Any]:
        self._opportunity.ledger.checkpoints += 1
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "durable": self._durable_state(),
            "active": self._active_state(),
            "observations_seen": self.observations_seen,
            "elapsed_ms": self.elapsed_ms,
            "frozen_mechanisms": sorted(self.frozen_mechanisms),
            "receipt_digest": _digest([_receipt_row(row) for row in self.receipts]),
            "activation": False,
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("mechanism") != self.mechanism:
            raise ValueError("checkpoint mechanism does not match this material")
        self._opportunity.ledger.restores += 1
        self._restore_durable(checkpoint["durable"])
        self._restore_active(checkpoint["active"])
        self.observations_seen = int(checkpoint["observations_seen"])
        self.elapsed_ms = int(checkpoint["elapsed_ms"])
        self.frozen_mechanisms = set(checkpoint.get("frozen_mechanisms", ()))

    # -- the sealed plasticity cycle --------------------------------------

    def observe(self, observation: Observation) -> None:
        before = self.durable_state_digest()
        self._opportunity.ledger.spend(1)
        self.observations_seen += 1
        self.elapsed_ms += max(0, observation.elapsed_ms)
        self._transition(observation)
        if self.durable_state_digest() != before:
            raise RuntimeError(f"{self.name} wrote durable state during observation")

    def answer(self, probe: Probe) -> Answer:
        before = self.durable_state_digest()
        self._opportunity.ledger.spend(1, retrievals=1)
        result = self._answer(probe)
        if self.durable_state_digest() != before:
            raise RuntimeError(f"{self.name} wrote durable state while answering")
        return result

    def propose(self) -> Sequence[Proposal]:
        if not self._opportunity.plasticity_enabled:
            return ()
        proposals = tuple(self._propose())
        self.pending = {proposal.proposal_id: proposal for proposal in proposals}
        if len(self.pending) != len(proposals):
            raise RuntimeError(f"{self.name} emitted duplicate proposal identifiers")
        return proposals

    def apply(self, verdicts: Sequence[Verdict]) -> Sequence[Receipt]:
        emitted: list[Receipt] = []
        for verdict in verdicts:
            proposal = self.pending.get(verdict.proposal_id)
            if proposal is None:
                raise RuntimeError(f"{self.name} received a verdict for an unknown proposal")
            before = self.durable_state_digest()
            committed = False
            if verdict.admitted and self._opportunity.plasticity_enabled:
                self._opportunity.ledger.durable_write()
                self._commit(proposal)
                committed = True
            after = self.durable_state_digest()
            if not committed and after != before:
                raise RuntimeError(f"{self.name} changed durable state on a refused proposal")
            receipt = Receipt(
                proposal_id=proposal.proposal_id,
                kind=proposal.kind,
                target=proposal.target,
                committed=committed,
                improvement=verdict.improvement,
                retention=verdict.retention,
                durable_state_digest_before=before,
                durable_state_digest_after=after,
                cost_bytes=proposal.cost_bytes,
                mechanism=self.mechanism,
            )
            self.receipts.append(receipt)
            emitted.append(receipt)
            # Only the handled proposals are retired, so a caller may verify
            # proposals one at a time: tentatively commit, measure, then keep
            # or roll back. Clearing everything here would force a batch
            # decision and make per-proposal improvement unattributable.
            self.pending.pop(verdict.proposal_id, None)
        return tuple(emitted)

    def rollback(self, receipt: Receipt) -> None:
        """Undo one committed change. P2 requires the benefit to disappear."""
        if not receipt.committed:
            return
        self._rollback(receipt)
        if self.durable_state_digest() != receipt.durable_state_digest_before:
            raise RuntimeError(f"{self.name} rollback did not restore the prior durable state")

    # -- hooks a concrete material must implement -------------------------

    def _transition(self, observation: Observation) -> None:
        raise NotImplementedError

    def _answer(self, probe: Probe) -> Answer:
        raise NotImplementedError

    def _propose(self) -> Iterable[Proposal]:
        raise NotImplementedError

    def _commit(self, proposal: Proposal) -> None:
        raise NotImplementedError

    def _rollback(self, receipt: Receipt) -> None:
        raise NotImplementedError

    def _durable_state(self) -> Any:
        raise NotImplementedError

    def _active_state(self) -> Any:
        raise NotImplementedError

    def _restore_durable(self, state: Any) -> None:
        raise NotImplementedError

    def _restore_active(self, state: Any) -> None:
        raise NotImplementedError


def _receipt_row(receipt: Receipt) -> tuple[Any, ...]:
    return (
        receipt.proposal_id,
        receipt.kind,
        receipt.target,
        receipt.committed,
        round(receipt.improvement, 9),
        round(receipt.retention, 9),
        receipt.durable_state_digest_before,
        receipt.durable_state_digest_after,
        receipt.cost_bytes,
        receipt.mechanism,
    )


# --------------------------------------------------------------------------
# Registry and the distinctness gate
# --------------------------------------------------------------------------

_REGISTRY: dict[str, Any] = {}


def register(name: str, factory: Any) -> Any:
    if name in _REGISTRY:
        raise ValueError(f"material {name!r} is already registered")
    _REGISTRY[name] = factory
    return factory


def build(name: str, opportunity: Opportunity, **options: Any) -> CognitiveMaterial:
    if name not in _REGISTRY:
        raise KeyError(f"no material registered as {name!r}")
    return _REGISTRY[name](opportunity, **options)


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def equal_opportunity(
    *,
    envelope: str,
    observations: Sequence[Observation],
    sensor_channels: Sequence[str],
    operation_budget: int,
    durable_write_budget: int,
    deprived: Sequence[str] = (),
) -> Opportunity:
    """Build one identical opportunity vector, then subtract exactly one thing."""
    deprived = tuple(deprived)
    unknown = [item for item in deprived if item not in C.PARITY_CHANNELS + ("correct_history", "history_order", "verified_growth", "development")]
    if unknown:
        raise ValueError(f"unknown deprivation {unknown}")
    teaching = tuple(observation.digest() for observation in observations if observation.teaching)
    ledger = ResourceLedger(
        envelope=envelope,
        operation_budget=operation_budget,
        durable_write_budget=0 if "plasticity" in deprived else durable_write_budget,
        byte_budget=C.ENVELOPE_BYTES[envelope],
    )
    return Opportunity(
        observation_digest=_digest([observation.digest() for observation in observations]),
        sensor_channels=tuple(sensor_channels),
        teaching_digest=_digest(teaching),
        ledger=ledger,
        plasticity_enabled="plasticity" not in deprived,
        persistence_enabled="persistence" not in deprived,
        deprived=deprived,
    )


def distinctness_report(
    materials: Sequence[CognitiveMaterial],
    observations: Sequence[Observation],
    probes: Sequence[Probe],
) -> dict[str, Any]:
    """Refuse a tournament in which two arms are the same material renamed.

    Every arm receives the identical probe sequence and an evaluator that
    admits every proposal, so the only thing that can differ is the mechanism.
    """
    traces: dict[str, dict[str, Any]] = {}
    for material in materials:
        for observation in observations:
            material.observe(observation)
        for probe in probes:
            material.answer(probe)
        proposals = material.propose()
        verdicts = [Verdict(proposal.proposal_id, True, 1.0, 1.0) for proposal in proposals]
        material.apply(verdicts)
        traces[material.name] = {
            "mechanism": material.mechanism,
            "durable_state_digest": material.durable_state_digest(),
            "proposal_kinds": sorted({proposal.kind for proposal in proposals}),
        }
    mechanisms = [row["mechanism"] for row in traces.values()]
    digests = [row["durable_state_digest"] for row in traces.values()]
    collisions = sorted(
        {
            tuple(sorted((left, right)))
            for left, left_row in traces.items()
            for right, right_row in traces.items()
            if left != right and left_row["durable_state_digest"] == right_row["durable_state_digest"]
        }
    )
    checks = {
        "distinct_mechanism_identifiers": len(set(mechanisms)) == len(mechanisms),
        "distinct_durable_state_under_identical_probe": len(set(digests)) == len(digests),
        "no_colliding_pair": not collisions,
    }
    return {
        "traces": traces,
        "collisions": [list(pair) for pair in collisions],
        "checks": checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


__all__ = [
    "Answer",
    "CognitiveMaterial",
    "MaterialBase",
    "Observation",
    "Opportunity",
    "Probe",
    "Proposal",
    "Receipt",
    "ResourceExhausted",
    "ResourceLedger",
    "Verdict",
    "build",
    "distinctness_report",
    "equal_opportunity",
    "register",
    "registered",
    "replace",
]
