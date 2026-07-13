"""Scaffold spine for the causal-intervention and simulation-for-action mechanism cluster.

Epoch G1-A1/S1/U1/N1. This module raises the SCAFFOLDING axis only. It supplies machine-checkable
contracts that ENCODE the scientific bar a mechanism must clear before any capability may be claimed:
a do-operator arm that must beat an observational-only control, a rollout policy that must beat a
random-action control at matched compute, a reliability metric that must beat an overconfident and a
temperature-one control, and a curiosity signal that may target reducible novelty only and refuses
irreducible-noise seeking. It builds the harness, not the result.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim. Nothing here establishes planning value, calibrated uncertainty, or useful
curiosity. The comparison helpers are seeded toy arithmetic over hashes; the activation gate is off
by default and refuses until an earned confirmation receipt is supplied by an external audit.

The named prior nulls are encoded as fail-closed conditions: the P7 planning null forces the
simulation-for-action bar, and the intervention, uncertainty, and curiosity nulls each force their
own contract to refuse unless the declared arm beats its matched control by the declared margin.

House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..substrate.events import canonical_sha256

INTERVENTION_SIM_SCHEMA = "mop-intervention-simulation/v1"
# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InterventionSimulationRefusal(ValueError):
    """Raised whenever a declaration is missing, malformed, or outside its declared scope."""


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise InterventionSimulationRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise InterventionSimulationRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_scope(value: str, label: str) -> None:
    if value != CLAIM_SCOPE:
        raise InterventionSimulationRefusal(f"{label} claim scope cannot be widened")


def _require_schema(value: str) -> None:
    if value != INTERVENTION_SIM_SCHEMA:
        raise InterventionSimulationRefusal(f"unsupported intervention-simulation schema {value!r}")


# ---------------------------------------------------------------------------
# Named prior nulls. Each contract binds to exactly one; a mismatched null is refused so a
# contract can never be re-pointed at a weaker null that its arm would trivially clear.
# ---------------------------------------------------------------------------

INTERVENTION_NULL = "observational-confound-null"
PLANNING_NULL = "p7-planning-null"
UNCERTAINTY_NULL = "temperature-one-uncertainty-null"
NOVELTY_NULL = "irreducible-noise-seeking-null"

PRIOR_NULLS: tuple[str, ...] = (
    INTERVENTION_NULL,
    PLANNING_NULL,
    UNCERTAINTY_NULL,
    NOVELTY_NULL,
)


# ---------------------------------------------------------------------------
# Canonical control sets. Membership and order are load-bearing for the registry digest, so the
# completeness check can detect a control being dropped, added, or reordered.
# ---------------------------------------------------------------------------

INTERVENTION_CONTROLS: tuple[str, ...] = ("observational-only", "backdoor-adjusted")
SIMULATION_CONTROLS: tuple[str, ...] = ("random-action", "zero-step-greedy", "replay-only")
UNCERTAINTY_CONTROLS: tuple[str, ...] = ("overconfident", "temperature-one")
NOVELTY_CONTROLS: tuple[str, ...] = ("random-curiosity", "count-based")

# The registry the completeness check pins. Order across families is also load-bearing.
CONTROL_REGISTRY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intervention", INTERVENTION_CONTROLS),
    ("simulation-for-action", SIMULATION_CONTROLS),
    ("calibrated-uncertainty", UNCERTAINTY_CONTROLS),
    ("reducible-novelty", NOVELTY_CONTROLS),
)


def assert_control_registry_intact() -> None:
    """Fail closed if any declared control family is empty, non-unique, or duplicated by name."""

    family_names = [name for name, _ in CONTROL_REGISTRY]
    if len(set(family_names)) != len(family_names):
        raise InterventionSimulationRefusal("control family names must be unique")
    for name, controls in CONTROL_REGISTRY:
        if not controls:
            raise InterventionSimulationRefusal(f"control family {name!r} declares no controls")
        if len(set(controls)) != len(controls):
            raise InterventionSimulationRefusal(f"control family {name!r} has duplicate control arms")


def control_registry_digest() -> str:
    """Stable digest over the entire control registry, membership and order included."""

    assert_control_registry_intact()
    return canonical_sha256({"registry": [[name, list(arms)] for name, arms in CONTROL_REGISTRY]})


# ---------------------------------------------------------------------------
# Matched cost discipline. Every contract implies a comparison, so every contract carries a matched
# budget that must be non-vacuous, and each refuses unless matched cost is explicitly required.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    """The matched full-system budget an arm and its control must share before comparison."""

    params: int
    flops: int
    memory_bytes: int
    rollout_steps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("memory_bytes", self.memory_bytes),
            ("rollout_steps", self.rollout_steps),
        ):
            if value <= 0:
                raise InterventionSimulationRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "rollout_steps": self.rollout_steps,
        }


def _require_matched_cost(matched_cost_required: bool, label: str) -> None:
    if not matched_cost_required:
        raise InterventionSimulationRefusal(f"{label} must require matched full-system cost")


# ---------------------------------------------------------------------------
# Deterministic, seeded comparison mechanics. These are toy hash-derived scores, not measurements
# on natural data. They exist so the fail-closed control-win rule can be exercised reproducibly.
# ---------------------------------------------------------------------------


def deterministic_unit_score(*, seed: int, label: str) -> float:
    """A reproducible score in [0, 1) from hashing; no wall-clock, no unseeded randomness.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    if seed < 0:
        raise InterventionSimulationRefusal("score seed must be nonnegative")
    _require_id(label, "deterministic_unit_score label")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:12], 16) / float(1 << 48)


@dataclass(frozen=True, slots=True)
class ControlWinOutcome:
    """The result of a single matched arm-vs-control comparison. Carries no capability claim."""

    arm_id: str
    control_id: str
    arm_score: float
    control_score: float
    margin_required: float

    def __post_init__(self) -> None:
        _require_id(self.arm_id, "ControlWinOutcome.arm_id")
        _require_id(self.control_id, "ControlWinOutcome.control_id")
        if self.arm_id == self.control_id:
            raise InterventionSimulationRefusal("an arm cannot be compared against itself")
        if self.margin_required <= 0.0:
            raise InterventionSimulationRefusal("a non-vacuous comparison needs a positive margin")

    @property
    def beats_control(self) -> bool:
        return (self.arm_score - self.control_score) >= self.margin_required

    def payload(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "control_id": self.control_id,
            "arm_score": self.arm_score,
            "control_score": self.control_score,
            "margin_required": self.margin_required,
            "beats_control": self.beats_control,
        }


def evaluate_control_win(
    *, arm_id: str, control_id: str, seed: int, margin_required: float
) -> ControlWinOutcome:
    """Deterministically score an arm and its control from a shared seed and compare them."""

    return ControlWinOutcome(
        arm_id=arm_id,
        control_id=control_id,
        arm_score=deterministic_unit_score(seed=seed, label=arm_id),
        control_score=deterministic_unit_score(seed=seed, label=control_id),
        margin_required=margin_required,
    )


def require_control_win(outcome: ControlWinOutcome, *, null_name: str) -> None:
    """Encode the named null as code: refuse unless the arm beats its matched control.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    if null_name not in PRIOR_NULLS:
        raise InterventionSimulationRefusal(f"unknown prior null {null_name!r}")
    if not outcome.beats_control:
        raise InterventionSimulationRefusal(
            f"{null_name} not rejected: arm {outcome.arm_id!r} does not beat control "
            f"{outcome.control_id!r} by the declared margin"
        )


# ---------------------------------------------------------------------------
# Contract A1. Causal intervention: a do-operator arm vs an observational-only control.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InterventionContract:
    """Declares a do-operator arm and the observational control it must beat under the null."""

    do_operator_arm: str
    observational_control: str
    controls: tuple[str, ...]
    prior_null: str
    matched: MatchedBudget
    matched_cost_required: bool
    margin_required: float
    schema: str = INTERVENTION_SIM_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_schema(self.schema)
        _require_scope(self.claim_scope, "intervention")
        _require_id(self.do_operator_arm, "InterventionContract.do_operator_arm")
        _require_id(self.observational_control, "InterventionContract.observational_control")
        if tuple(self.controls) != INTERVENTION_CONTROLS:
            raise InterventionSimulationRefusal("intervention controls or order drift")
        if self.observational_control not in self.controls:
            raise InterventionSimulationRefusal("the observational control must be a declared control arm")
        if self.do_operator_arm == self.observational_control:
            raise InterventionSimulationRefusal("the do-operator arm cannot be the observational control")
        if self.prior_null != INTERVENTION_NULL:
            raise InterventionSimulationRefusal("intervention contract is bound to the confound null")
        _require_matched_cost(self.matched_cost_required, "intervention")
        if self.margin_required <= 0.0:
            raise InterventionSimulationRefusal("intervention margin must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "do_operator_arm": self.do_operator_arm,
            "observational_control": self.observational_control,
            "controls": list(self.controls),
            "prior_null": self.prior_null,
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
            "margin_required": self.margin_required,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    def evaluate(self, *, seed: int) -> ControlWinOutcome:
        return evaluate_control_win(
            arm_id=self.do_operator_arm,
            control_id=self.observational_control,
            seed=seed,
            margin_required=self.margin_required,
        )


# ---------------------------------------------------------------------------
# Contract S1. Simulation for action: a rollout policy that must beat a random-action control at
# matched compute. Bound to the P7 planning null.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimulationForActionContract:
    """Declares a rollout policy whose value must beat a random-action control at matched compute."""

    policy_id: str
    controls: tuple[str, ...]
    rollout_horizon: int
    value_margin_required: float
    prior_null: str
    matched: MatchedBudget
    matched_cost_required: bool
    schema: str = INTERVENTION_SIM_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_schema(self.schema)
        _require_scope(self.claim_scope, "simulation-for-action")
        _require_id(self.policy_id, "SimulationForActionContract.policy_id")
        if tuple(self.controls) != SIMULATION_CONTROLS:
            raise InterventionSimulationRefusal("simulation controls or order drift")
        if "random-action" not in self.controls:
            raise InterventionSimulationRefusal("simulation-for-action must name a random-action control")
        if self.rollout_horizon < 1:
            raise InterventionSimulationRefusal("rollout horizon must be at least one step")
        if self.value_margin_required <= 0.0:
            raise InterventionSimulationRefusal("value margin must be positive")
        if self.prior_null != PLANNING_NULL:
            raise InterventionSimulationRefusal("simulation-for-action is forced by the P7 planning null")
        _require_matched_cost(self.matched_cost_required, "simulation-for-action")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "policy_id": self.policy_id,
            "controls": list(self.controls),
            "rollout_horizon": self.rollout_horizon,
            "value_margin_required": self.value_margin_required,
            "prior_null": self.prior_null,
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    def evaluate(self, *, seed: int) -> ControlWinOutcome:
        return evaluate_control_win(
            arm_id=self.policy_id,
            control_id="random-action",
            seed=seed,
            margin_required=self.value_margin_required,
        )


# ---------------------------------------------------------------------------
# Contract U1. Calibrated uncertainty: a reliability metric that must beat an overconfident and a
# temperature-one control. Bound to the temperature-one uncertainty null.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibratedUncertaintyContract:
    """Declares a reliability metric and the two miscalibrated controls it must beat."""

    reliability_metric: str
    controls: tuple[str, ...]
    calibration_bins: int
    max_calibration_error: float
    reliability_margin_required: float
    prior_null: str
    matched: MatchedBudget
    matched_cost_required: bool
    schema: str = INTERVENTION_SIM_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_schema(self.schema)
        _require_scope(self.claim_scope, "calibrated-uncertainty")
        _require_id(self.reliability_metric, "CalibratedUncertaintyContract.reliability_metric")
        if tuple(self.controls) != UNCERTAINTY_CONTROLS:
            raise InterventionSimulationRefusal("uncertainty controls or order drift")
        if self.calibration_bins < 2:
            raise InterventionSimulationRefusal("a reliability curve needs at least two calibration bins")
        if not 0.0 < self.max_calibration_error < 1.0:
            raise InterventionSimulationRefusal("max calibration error must be in (0, 1)")
        if self.reliability_margin_required <= 0.0:
            raise InterventionSimulationRefusal("reliability margin must be positive")
        if self.prior_null != UNCERTAINTY_NULL:
            raise InterventionSimulationRefusal("uncertainty contract is bound to the temperature-one null")
        _require_matched_cost(self.matched_cost_required, "calibrated-uncertainty")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "reliability_metric": self.reliability_metric,
            "controls": list(self.controls),
            "calibration_bins": self.calibration_bins,
            "max_calibration_error": self.max_calibration_error,
            "reliability_margin_required": self.reliability_margin_required,
            "prior_null": self.prior_null,
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    def evaluate(self, *, seed: int, against: str = "temperature-one") -> ControlWinOutcome:
        if against not in self.controls:
            raise InterventionSimulationRefusal(f"{against!r} is not a declared uncertainty control")
        return evaluate_control_win(
            arm_id=self.reliability_metric,
            control_id=against,
            seed=seed,
            margin_required=self.reliability_margin_required,
        )


# ---------------------------------------------------------------------------
# Contract N1. Reducible-novelty curiosity: the signal may target reducible novelty only and REFUSES
# irreducible-noise seeking; it must beat random-curiosity and count-based controls.
# ---------------------------------------------------------------------------

ALLOWED_NOVELTY_TARGET = "reducible"
FORBIDDEN_NOVELTY_TARGETS: tuple[str, ...] = (
    "irreducible",
    "aleatoric-noise",
    "white-noise",
    "unpredictable-observation",
)


@dataclass(frozen=True, slots=True)
class ReducibleNoveltyContract:
    """Declares a curiosity signal restricted to reducible novelty, with a noise-seeking refusal."""

    curiosity_signal: str
    novelty_target: str
    reducibility_metric: str
    controls: tuple[str, ...]
    curiosity_margin_required: float
    prior_null: str
    matched: MatchedBudget
    matched_cost_required: bool
    schema: str = INTERVENTION_SIM_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_schema(self.schema)
        _require_scope(self.claim_scope, "reducible-novelty")
        _require_id(self.curiosity_signal, "ReducibleNoveltyContract.curiosity_signal")
        _require_id(self.reducibility_metric, "ReducibleNoveltyContract.reducibility_metric")
        if self.novelty_target in FORBIDDEN_NOVELTY_TARGETS:
            raise InterventionSimulationRefusal(
                "irreducible-noise seeking refused: curiosity may target reducible novelty only"
            )
        if self.novelty_target != ALLOWED_NOVELTY_TARGET:
            raise InterventionSimulationRefusal(
                f"unsupported novelty target {self.novelty_target!r}; "
                f"only {ALLOWED_NOVELTY_TARGET!r} is admissible"
            )
        if tuple(self.controls) != NOVELTY_CONTROLS:
            raise InterventionSimulationRefusal("novelty controls or order drift")
        if self.curiosity_margin_required <= 0.0:
            raise InterventionSimulationRefusal("curiosity margin must be positive")
        if self.prior_null != NOVELTY_NULL:
            raise InterventionSimulationRefusal("novelty contract is bound to the noise-seeking null")
        _require_matched_cost(self.matched_cost_required, "reducible-novelty")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "curiosity_signal": self.curiosity_signal,
            "novelty_target": self.novelty_target,
            "reducibility_metric": self.reducibility_metric,
            "controls": list(self.controls),
            "curiosity_margin_required": self.curiosity_margin_required,
            "prior_null": self.prior_null,
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    def evaluate(self, *, seed: int, against: str = "count-based") -> ControlWinOutcome:
        if against not in self.controls:
            raise InterventionSimulationRefusal(f"{against!r} is not a declared novelty control")
        return evaluate_control_win(
            arm_id=self.curiosity_signal,
            control_id=against,
            seed=seed,
            margin_required=self.curiosity_margin_required,
        )


# ---------------------------------------------------------------------------
# Activation gate. Off by default. Any local attempt to authorize deployment of these mechanisms
# raises unless an earned confirmation receipt from an external audit is supplied and matches the
# preregistered digest. This encodes "activation not earned yet".
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeploymentActivationGate:
    """A fail-closed gate. Activation is off until an external audit receipt is presented."""

    preregistration_digest: str
    activation_requested: bool = False
    confirmation_receipt: str | None = None
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_sha256(self.preregistration_digest, "DeploymentActivationGate.preregistration_digest")
        _require_scope(self.claim_scope, "activation gate")
        if self.confirmation_receipt is not None:
            _require_sha256(self.confirmation_receipt, "DeploymentActivationGate.confirmation_receipt")

    def authorize(self) -> None:
        """Fail closed unless activation was explicitly requested with a matching audit receipt."""

        if not self.activation_requested:
            raise InterventionSimulationRefusal(
                "deployment activation is off by default; no capability has been earned"
            )
        if self.confirmation_receipt is None:
            raise InterventionSimulationRefusal(
                "activation requires an external audit confirmation receipt; none supplied"
            )
        if self.confirmation_receipt != self.preregistration_digest:
            raise InterventionSimulationRefusal(
                "confirmation receipt does not match the preregistered digest; activation refused"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "preregistration_digest": self.preregistration_digest,
            "activation_requested": self.activation_requested,
            "confirmation_receipt": self.confirmation_receipt,
            "claim_scope": self.claim_scope,
        }


# ---------------------------------------------------------------------------
# Epoch aggregate. Composes all four contracts and the gate into one digest-bound scaffold.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpochScaffold:
    """The complete G1-A1/S1/U1/N1 scaffold with a stable digest over all four contracts."""

    intervention: InterventionContract
    simulation: SimulationForActionContract
    uncertainty: CalibratedUncertaintyContract
    novelty: ReducibleNoveltyContract
    gate: DeploymentActivationGate
    schema: str = field(default=INTERVENTION_SIM_SCHEMA)
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_schema(self.schema)
        _require_scope(self.claim_scope, "epoch scaffold")
        assert_control_registry_intact()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "control_registry_digest": control_registry_digest(),
            "intervention": self.intervention.payload(),
            "simulation": self.simulation.payload(),
            "uncertainty": self.uncertainty.payload(),
            "novelty": self.novelty.payload(),
            "gate": self.gate.payload(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Deterministic default builders (used by tests and later wiring).
# ---------------------------------------------------------------------------


def _default_budget() -> MatchedBudget:
    return MatchedBudget(params=256, flops=8192, memory_bytes=16384, rollout_steps=16)


def default_intervention_contract() -> InterventionContract:
    return InterventionContract(
        do_operator_arm="do-operator-arm",
        observational_control="observational-only",
        controls=INTERVENTION_CONTROLS,
        prior_null=INTERVENTION_NULL,
        matched=_default_budget(),
        matched_cost_required=True,
        margin_required=0.02,
    )


def default_simulation_contract() -> SimulationForActionContract:
    return SimulationForActionContract(
        policy_id="rollout-value-policy",
        controls=SIMULATION_CONTROLS,
        rollout_horizon=8,
        value_margin_required=0.05,
        prior_null=PLANNING_NULL,
        matched=_default_budget(),
        matched_cost_required=True,
    )


def default_uncertainty_contract() -> CalibratedUncertaintyContract:
    return CalibratedUncertaintyContract(
        reliability_metric="expected-calibration-error",
        controls=UNCERTAINTY_CONTROLS,
        calibration_bins=10,
        max_calibration_error=0.1,
        reliability_margin_required=0.02,
        prior_null=UNCERTAINTY_NULL,
        matched=_default_budget(),
        matched_cost_required=True,
    )


def default_novelty_contract() -> ReducibleNoveltyContract:
    return ReducibleNoveltyContract(
        curiosity_signal="information-gain-signal",
        novelty_target=ALLOWED_NOVELTY_TARGET,
        reducibility_metric="learning-progress",
        controls=NOVELTY_CONTROLS,
        curiosity_margin_required=0.03,
        prior_null=NOVELTY_NULL,
        matched=_default_budget(),
        matched_cost_required=True,
    )


def default_activation_gate() -> DeploymentActivationGate:
    return DeploymentActivationGate(preregistration_digest="0" * 64)


def build_epoch_scaffold() -> EpochScaffold:
    """Return a fully valid, off-by-default epoch scaffold; deterministic and side-effect free."""

    return EpochScaffold(
        intervention=default_intervention_contract(),
        simulation=default_simulation_contract(),
        uncertainty=default_uncertainty_contract(),
        novelty=default_novelty_contract(),
        gate=default_activation_gate(),
    )


# ---------------------------------------------------------------------------
# Capability posture and coverage record.
# ---------------------------------------------------------------------------

SCIENTIFIC_CAPABILITY_CLAIM = False


def coverage() -> dict[str, list[str]]:
    """Static record of the epoch sub-questions this scaffold encodes (readiness only)."""

    return {
        "A1-causal-intervention": [
            "InterventionContract pins a do-operator arm against an observational-only control",
            "the observational-confound null is refused unless the do arm beats the control by margin",
            "matched full-system cost is required before the intervention comparison is admissible",
        ],
        "S1-simulation-for-action": [
            "SimulationForActionContract requires rollout value to beat a random-action control",
            "the comparison runs at matched compute via a non-vacuous MatchedBudget",
            "the P7 planning null forces the bar and is refused unless the policy wins",
        ],
        "U1-calibrated-uncertainty": [
            "CalibratedUncertaintyContract declares a reliability metric over at least two bins",
            "overconfident and temperature-one controls must both be beaten by the reliability margin",
            "the temperature-one uncertainty null is bound and cannot be swapped for a weaker null",
        ],
        "N1-reducible-novelty": [
            "ReducibleNoveltyContract admits the reducible novelty target only",
            "irreducible-noise seeking is refused at construction, encoding the noise-seeking null",
            "random-curiosity and count-based controls must be beaten by the curiosity margin",
        ],
    }
