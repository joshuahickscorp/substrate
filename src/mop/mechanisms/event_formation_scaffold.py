from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

EVENT_FORMATION_SCHEMA = "mop-event-formation/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_CONTROLS: tuple[str, ...] = (
    "wrong-time",  # the bound event fired outside its temporal window
    "wrong-event",  # a decoy event stood in for the target event
    "appearance-only",  # only surface appearance, no relational or temporal structure
    "stateless-delayed-trigger",  # a stateless delayed-scalar trigger, the X0 null mechanism
)

UNTRAINED_CONTROLS: tuple[str, ...] = ("appearance-only", "stateless-delayed-trigger")

REQUIRED_REPLICATIONS = 3

SCIENTIFIC_CAPABILITY_CLAIM = False

_EPISODE_RELATIONS: tuple[str, ...] = ("supports", "contains", "precedes", "opposes")


class EventFormationRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise EventFormationRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise EventFormationRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_finite(value: float, label: str) -> None:
    if value != value or value in (float("inf"), float("-inf")):
        raise EventFormationRefusal(f"{label} must be a finite number")


def assert_control_ledger(controls: Sequence[str]) -> None:

    if tuple(controls) != REQUIRED_CONTROLS:
        raise EventFormationRefusal("control ledger has membership or order drift vs the required controls")


@dataclass(frozen=True, slots=True)
class ControlLedgerContract:
    controls: tuple[str, ...] = REQUIRED_CONTROLS
    claim_scope: str = CLAIM_SCOPE
    schema: str = EVENT_FORMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMATION_SCHEMA:
            raise EventFormationRefusal(f"unsupported control ledger schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise EventFormationRefusal("control ledger claim scope cannot be widened")
        assert_control_ledger(self.controls)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "controls": list(self.controls),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class RelationalEventContract:
    event_id: str
    relation: str
    entity_refs: tuple[str, ...]
    scalar_only: bool = False
    claim_scope: str = CLAIM_SCOPE
    schema: str = EVENT_FORMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMATION_SCHEMA:
            raise EventFormationRefusal(f"unsupported relational event schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise EventFormationRefusal("relational event claim scope cannot be widened")
        if self.scalar_only:
            raise EventFormationRefusal("a relational event cannot be a single scalar")
        _require_id(self.event_id, "RelationalEventContract.event_id")
        _require_id(self.relation, "RelationalEventContract.relation")
        if len(self.entity_refs) < 2:
            raise EventFormationRefusal("a relational event needs at least two entities")
        if len(set(self.entity_refs)) != len(self.entity_refs):
            raise EventFormationRefusal("relational event entities must be unique")
        for ref in self.entity_refs:
            _require_id(ref, "RelationalEventContract.entity_ref")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "event_id": self.event_id,
            "relation": self.relation,
            "entity_refs": list(self.entity_refs),
            "scalar_only": self.scalar_only,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class TemporalEventBindingContract:
    event_id: str
    clock_id: str
    window_start_tick: int
    window_end_tick: int
    wrong_time_tick: int
    wrong_event_id: str
    claim_scope: str = CLAIM_SCOPE
    schema: str = EVENT_FORMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMATION_SCHEMA:
            raise EventFormationRefusal(f"unsupported temporal binding schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise EventFormationRefusal("temporal binding claim scope cannot be widened")
        _require_id(self.event_id, "TemporalEventBindingContract.event_id")
        _require_id(self.clock_id, "TemporalEventBindingContract.clock_id")
        _require_id(self.wrong_event_id, "TemporalEventBindingContract.wrong_event_id")
        if self.window_end_tick <= self.window_start_tick:
            raise EventFormationRefusal("binding window must end strictly after its start")
        if self.window_start_tick <= self.wrong_time_tick <= self.window_end_tick:
            raise EventFormationRefusal("wrong-time control must sit outside the binding window")
        if self.wrong_event_id == self.event_id:
            raise EventFormationRefusal("wrong-event control must be a different event")

    @property
    def window_ticks(self) -> int:
        return self.window_end_tick - self.window_start_tick

    @property
    def controls(self) -> tuple[str, ...]:
        return ("wrong-time", "wrong-event")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "event_id": self.event_id,
            "clock_id": self.clock_id,
            "window_start_tick": self.window_start_tick,
            "window_end_tick": self.window_end_tick,
            "wrong_time_tick": self.wrong_time_tick,
            "wrong_event_id": self.wrong_event_id,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class OracleHeadroomContract:
    oracle_id: str
    oracle_utility: float
    baseline_utility: float
    full_charged_compute: float
    oracle_charged_compute: float
    headroom_min: float
    measured: bool
    claim_scope: str = CLAIM_SCOPE
    schema: str = EVENT_FORMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMATION_SCHEMA:
            raise EventFormationRefusal(f"unsupported oracle headroom schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise EventFormationRefusal("oracle headroom claim scope cannot be widened")
        _require_id(self.oracle_id, "OracleHeadroomContract.oracle_id")
        for name, value in (
            ("oracle_utility", self.oracle_utility),
            ("baseline_utility", self.baseline_utility),
            ("full_charged_compute", self.full_charged_compute),
            ("oracle_charged_compute", self.oracle_charged_compute),
            ("headroom_min", self.headroom_min),
        ):
            _require_finite(value, f"OracleHeadroomContract.{name}")
        if self.full_charged_compute <= 0.0:
            raise EventFormationRefusal("full charged compute must be positive")
        if not self.measured:
            raise EventFormationRefusal("oracle headroom must be measured, not assumed")
        if self.utility_headroom() < self.headroom_min:
            raise EventFormationRefusal("oracle shows no real headroom over the baseline")

    def utility_headroom(self) -> float:
        return self.oracle_utility - self.baseline_utility

    def oracle_savings_fraction(self) -> float:
        return 1.0 - self.oracle_charged_compute / self.full_charged_compute

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "oracle_id": self.oracle_id,
            "oracle_utility": self.oracle_utility,
            "baseline_utility": self.baseline_utility,
            "full_charged_compute": self.full_charged_compute,
            "oracle_charged_compute": self.oracle_charged_compute,
            "headroom_min": self.headroom_min,
            "measured": self.measured,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    relational_ops: int
    temporal_ops: int
    trigger_evals: int
    memory_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("relational_ops", self.relational_ops),
            ("temporal_ops", self.temporal_ops),
            ("trigger_evals", self.trigger_evals),
            ("memory_bytes", self.memory_bytes),
        ):
            if value <= 0:
                raise EventFormationRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "relational_ops": self.relational_ops,
            "temporal_ops": self.temporal_ops,
            "trigger_evals": self.trigger_evals,
            "memory_bytes": self.memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class EventUtilityVerdict:
    candidate_id: str
    oracle: OracleHeadroomContract
    utility_candidate: float
    utility_floor: float
    utility_by_control: Mapping[str, float]
    charged_compute_candidate: float
    charged_compute_by_control: Mapping[str, float]
    budget: MatchedBudget
    matched_cost_required: bool = True
    claim_scope: str = CLAIM_SCOPE
    schema: str = EVENT_FORMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMATION_SCHEMA:
            raise EventFormationRefusal(f"unsupported event utility schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise EventFormationRefusal("event utility claim scope cannot be widened")
        _require_id(self.candidate_id, "EventUtilityVerdict.candidate_id")
        if not self.matched_cost_required:
            raise EventFormationRefusal(
                "event utility must require matched charged-compute cost before any claim"
            )
        if set(self.utility_by_control) != set(UNTRAINED_CONTROLS):
            raise EventFormationRefusal("utility by control must cover exactly the untrained controls")
        if set(self.charged_compute_by_control) != set(UNTRAINED_CONTROLS):
            raise EventFormationRefusal(
                "charged compute by control must cover exactly the untrained controls"
            )
        _require_finite(self.utility_candidate, "EventUtilityVerdict.utility_candidate")
        _require_finite(self.utility_floor, "EventUtilityVerdict.utility_floor")
        _require_finite(self.charged_compute_candidate, "EventUtilityVerdict.charged_compute_candidate")
        for control in UNTRAINED_CONTROLS:
            _require_finite(self.utility_by_control[control], f"utility_by_control[{control}]")
            _require_finite(
                self.charged_compute_by_control[control], f"charged_compute_by_control[{control}]"
            )

    def utility_preserved(self) -> bool:
        return self.utility_candidate >= self.utility_floor

    def beats_untrained_utility(self) -> bool:
        return all(
            self.utility_candidate > self.utility_by_control[control] for control in UNTRAINED_CONTROLS
        )

    def compute_cut_vs_both_untrained(self) -> bool:
        return all(
            self.charged_compute_candidate < self.charged_compute_by_control[control]
            for control in UNTRAINED_CONTROLS
        )

    def claims_useful_event(self) -> bool:
        return (
            self.utility_preserved()
            and self.beats_untrained_utility()
            and self.compute_cut_vs_both_untrained()
        )

    def refutes_x0_strong_null(self) -> bool:
        return self.claims_useful_event()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "candidate_id": self.candidate_id,
            "oracle": self.oracle.payload(),
            "utility_candidate": self.utility_candidate,
            "utility_floor": self.utility_floor,
            "utility_by_control": dict(self.utility_by_control),
            "charged_compute_candidate": self.charged_compute_candidate,
            "charged_compute_by_control": dict(self.charged_compute_by_control),
            "budget": self.budget.payload(),
            "matched_cost_required": self.matched_cost_required,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ActivationReceipt:
    license_id: str
    verdict_digest: str
    claims_useful_event: bool
    independent_replications: int
    claim_scope: str = CLAIM_SCOPE
    schema: str = EVENT_FORMATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMATION_SCHEMA:
            raise EventFormationRefusal(f"unsupported activation receipt schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise EventFormationRefusal("activation receipt claim scope cannot be widened")
        _require_id(self.license_id, "ActivationReceipt.license_id")
        _require_sha256(self.verdict_digest, "ActivationReceipt.verdict_digest")
        if self.independent_replications < 0:
            raise EventFormationRefusal("independent replications must be nonnegative")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "license_id": self.license_id,
            "verdict_digest": self.verdict_digest,
            "claims_useful_event": self.claims_useful_event,
            "independent_replications": self.independent_replications,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class EventFormationActivationGate:
    activation_permitted: bool = False
    required_replications: int = REQUIRED_REPLICATIONS

    def __post_init__(self) -> None:
        if self.activation_permitted:
            raise EventFormationRefusal("event formation activation cannot be self-permitted by local code")
        if self.required_replications < 1:
            raise EventFormationRefusal("required replications must be at least one")

    def authorize(self, receipt: ActivationReceipt | None = None) -> ActivationReceipt:

        if receipt is None:
            raise EventFormationRefusal(
                "event formation is not activated; supply a valid activation receipt from an "
                "external authority with its own randomization and held-out evaluation"
            )
        if not receipt.claims_useful_event:
            raise EventFormationRefusal("receipt does not claim a useful event; activation stays closed")
        if receipt.independent_replications < self.required_replications:
            raise EventFormationRefusal("receipt carries fewer than the required independent replications")
        return receipt

    def authorize_local(self) -> ActivationReceipt:

        raise EventFormationRefusal("local activation of event formation is never permitted")

    def payload(self) -> dict[str, Any]:
        return {
            "activation_permitted": self.activation_permitted,
            "required_replications": self.required_replications,
        }


def default_matched_budget() -> MatchedBudget:

    return MatchedBudget(relational_ops=8, temporal_ops=4, trigger_evals=4, memory_bytes=1024)


def _reference_oracle() -> OracleHeadroomContract:
    return OracleHeadroomContract(
        oracle_id="oracle.reference",
        oracle_utility=1.0,
        baseline_utility=0.3,
        full_charged_compute=1.0,
        oracle_charged_compute=0.27,
        headroom_min=0.2,
        measured=True,
    )


def build_x0_strong_null_verdict() -> EventUtilityVerdict:

    return EventUtilityVerdict(
        candidate_id="candidate.stateless-delayed-trigger",
        oracle=_reference_oracle(),
        utility_candidate=0.2,
        utility_floor=0.8,
        utility_by_control={"appearance-only": 0.5, "stateless-delayed-trigger": 0.45},
        charged_compute_candidate=0.1,
        charged_compute_by_control={"appearance-only": 0.8, "stateless-delayed-trigger": 0.45},
        budget=default_matched_budget(),
    )


def build_hypothetical_useful_verdict() -> EventUtilityVerdict:

    return EventUtilityVerdict(
        candidate_id="candidate.hypothetical",
        oracle=_reference_oracle(),
        utility_candidate=0.9,
        utility_floor=0.8,
        utility_by_control={"appearance-only": 0.4, "stateless-delayed-trigger": 0.35},
        charged_compute_candidate=0.3,
        charged_compute_by_control={"appearance-only": 0.8, "stateless-delayed-trigger": 0.45},
        budget=default_matched_budget(),
    )


def mint_receipt(
    verdict: EventUtilityVerdict,
    *,
    license_id: str,
    independent_replications: int,
) -> ActivationReceipt:

    return ActivationReceipt(
        license_id=license_id,
        verdict_digest=verdict.digest(),
        claims_useful_event=verdict.claims_useful_event(),
        independent_replications=independent_replications,
    )


def synthesize_relational_episode(seed: int, *, num_entities: int = 3) -> RelationalEventContract:

    if seed < 0:
        raise EventFormationRefusal("episode seed must be nonnegative")
    if num_entities < 2:
        raise EventFormationRefusal("a relational episode needs at least two entities")
    digest = hashlib.sha256(str(seed).encode("ascii")).digest()
    relation = _EPISODE_RELATIONS[digest[0] % len(_EPISODE_RELATIONS)]
    entity_refs = tuple(f"entity.{seed}.{index}" for index in range(num_entities))
    return RelationalEventContract(
        event_id=f"event.episode.{seed}",
        relation=relation,
        entity_refs=entity_refs,
    )
