
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..substrate.events import canonical_sha256

INTERVENTION_SIM_SCHEMA = "mop-intervention-simulation/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InterventionSimulationRefusal(ValueError):
    pass


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


INTERVENTION_CONTROLS: tuple[str, ...] = ("observational-only", "backdoor-adjusted")
SIMULATION_CONTROLS: tuple[str, ...] = ("random-action", "zero-step-greedy", "replay-only")
UNCERTAINTY_CONTROLS: tuple[str, ...] = ("overconfident", "temperature-one")
NOVELTY_CONTROLS: tuple[str, ...] = ("random-curiosity", "count-based")

CONTROL_REGISTRY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intervention", INTERVENTION_CONTROLS),
    ("simulation-for-action", SIMULATION_CONTROLS),
    ("calibrated-uncertainty", UNCERTAINTY_CONTROLS),
    ("reducible-novelty", NOVELTY_CONTROLS),
)


def assert_control_registry_intact() -> None:

    family_names = [name for name, _ in CONTROL_REGISTRY]
    if len(set(family_names)) != len(family_names):
        raise InterventionSimulationRefusal("control family names must be unique")
    for name, controls in CONTROL_REGISTRY:
        if not controls:
            raise InterventionSimulationRefusal(f"control family {name!r} declares no controls")
        if len(set(controls)) != len(controls):
            raise InterventionSimulationRefusal(f"control family {name!r} has duplicate control arms")


def control_registry_digest() -> str:

    assert_control_registry_intact()
    return canonical_sha256({"registry": [[name, list(arms)] for name, arms in CONTROL_REGISTRY]})


@dataclass(frozen=True, slots=True)
class MatchedBudget:

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


def deterministic_unit_score(*, seed: int, label: str) -> float:

    if seed < 0:
        raise InterventionSimulationRefusal("score seed must be nonnegative")
    _require_id(label, "deterministic_unit_score label")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:12], 16) / float(1 << 48)


@dataclass(frozen=True, slots=True)
class ControlWinOutcome:

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

    return ControlWinOutcome(
        arm_id=arm_id,
        control_id=control_id,
        arm_score=deterministic_unit_score(seed=seed, label=arm_id),
        control_score=deterministic_unit_score(seed=seed, label=control_id),
        margin_required=margin_required,
    )


def require_control_win(outcome: ControlWinOutcome, *, null_name: str) -> None:

    if null_name not in PRIOR_NULLS:
        raise InterventionSimulationRefusal(f"unknown prior null {null_name!r}")
    if not outcome.beats_control:
        raise InterventionSimulationRefusal(
            f"{null_name} not rejected: arm {outcome.arm_id!r} does not beat control "
            f"{outcome.control_id!r} by the declared margin"
        )


@dataclass(frozen=True, slots=True)
class InterventionContract:

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


@dataclass(frozen=True, slots=True)
class SimulationForActionContract:

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


@dataclass(frozen=True, slots=True)
class CalibratedUncertaintyContract:

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


ALLOWED_NOVELTY_TARGET = "reducible"
FORBIDDEN_NOVELTY_TARGETS: tuple[str, ...] = (
    "irreducible",
    "aleatoric-noise",
    "white-noise",
    "unpredictable-observation",
)


@dataclass(frozen=True, slots=True)
class ReducibleNoveltyContract:

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


@dataclass(frozen=True, slots=True)
class DeploymentActivationGate:

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


@dataclass(frozen=True, slots=True)
class EpochScaffold:

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

    return EpochScaffold(
        intervention=default_intervention_contract(),
        simulation=default_simulation_contract(),
        uncertainty=default_uncertainty_contract(),
        novelty=default_novelty_contract(),
        gate=default_activation_gate(),
    )


SCIENTIFIC_CAPABILITY_CLAIM = False


def coverage() -> dict[str, list[str]]:

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
