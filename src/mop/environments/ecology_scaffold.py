"""Interactive ecology scaffold contracts for facets RA5, RA6, PA7, PA8, and PA9.

Five machine-checkable contract families raise the scaffolding axis of the interactive ecology
cluster without running any experiment:

1. Bounded world generation (RA5): hidden dynamics, tools, damage, and sensor cost declarations
   plus a declared world-resampling rule that separates adaptation from map memorization.
2. Active perception (RA6, rows f22 and f28): costed choices among view, time, audio, and
   abstention with declared acquisition controls and a pre-cost sensor value forecast declaration.
3. Autotelic and quality-diverse ecology (PA7, rows f50, f51, f52): a goal archive, a
   learning-progress curriculum band, noisy-TV and reward-hacking guards written as refusal rules,
   and plateau, collapse, archive-bloat, and unsafe-goal stop rules.
4. Simulated partners (PA8, rows f53, f54, f55, f56, f58): private observations, held-out
   policies, declared joint-attention, repair, selective-imitation, teaching-versus-equal-
   information, and cumulative-convention experiments, and a partner-policy pattern-matching
   control declaration.
5. Communication grounding (PA9, row f57): messages bound to events, actions, consequences,
   uncertainty, and repair, with random, fixed, direct-state, and equal-bandwidth controls.

Everything is deterministic and content-addressed through the Wave E0 canonical JSON identity in
``mop.substrate.events``. Fixtures reuse the persistent grid world specification instead of adding
a second environment primitive. Every validator fails closed: malformed or missing declarations
raise at construction time, and refusal rules are executable code, never comments.

Claim scope: deterministic programmatic mechanics only; no capability claim. Passing these
contracts licenses scaffold readiness, not embodiment, open-endedness, social cognition, or
grounded communication on natural data.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..devel.north_star import assert_no_sentience_claims
from ..experiments.expansion_harness import CLAIM_SCOPE
from ..substrate.events import canonical_bytes, canonical_sha256
from .persistent_grid import WorldSpec, make_world_spec

ECOLOGY_SCHEMA = "mop-interactive-ecology-scaffold/v1"

ADAPTATION_CONTROLS = ("reactive", "scripted", "model-free", "oracle-state", "replay-only")
SENSING_ACTIONS = ("view", "time", "audio", "abstain")
ACQUISITION_CONTROLS = (
    "random-acquisition",
    "uncertainty-acquisition",
    "saliency-acquisition",
    "full-observation",
)
CURRICULUM_CONTROLS = ("random-task", "fixed-order", "easiest-first", "hardest-first")
ECOLOGY_GUARDS = ("noisy-tv", "reward-hacking")
ECOLOGY_STOP_RULES = ("plateau", "collapse", "archive-bloat", "unsafe-goal")
PARTNER_EXPERIMENTS = (
    "joint-attention",
    "communicative-repair",
    "selective-imitation",
    "teaching-vs-equal-information",
    "cumulative-convention",
)
REQUIRED_EXPERIMENT_CONTROLS = {
    "joint-attention": "cue-blind-partner",
    "communicative-repair": "no-repair-channel",
    "selective-imitation": "indiscriminate-imitation",
    "teaching-vs-equal-information": "equal-information",
    "cumulative-convention": "generation-reset",
}
PARTNER_POLICY_CONTROL = "partner-policy-pattern-matching"
MESSAGE_BINDINGS = ("event", "action", "consequence", "uncertainty", "repair")
CHANNEL_CONTROLS = ("random-message", "fixed-message", "direct-state", "equal-bandwidth")
CODE_SCORE_AXES = (
    "usefulness",
    "stability",
    "composition",
    "interpretability",
    "bandwidth",
    "deception",
)
TRANSFER_REQUIREMENTS = ("cross-seed", "cross-partner", "cross-generation", "cross-referent")
REFUSAL_RULES = (
    "noisy-tv",
    "reward-hacking",
    "unsafe-goal",
    "archive-bloat",
    "identity-collision",
)


class EcologyRefusal(ValueError):
    """A named, fail-closed refusal. The rule name is code-checked against REFUSAL_RULES."""

    def __init__(self, rule: str, message: str) -> None:
        if rule not in REFUSAL_RULES:
            raise ValueError(f"unknown refusal rule {rule!r}")
        super().__init__(f"{rule}: {message}")
        self.rule = rule


def _stable_int(seed: int, label: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.sha256(canonical_bytes({"seed": seed, "label": label})).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _derive_world_seed(family_seed: int, role: str, index: int) -> int:
    digest = hashlib.sha256(
        canonical_bytes({"family_seed": family_seed, "role": role, "index": index})
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _require_clean_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must not be empty")
    if "\u2014" in value or "\u2013" in value:
        raise ValueError(f"{label} must not contain em or en dashes")
    assert_no_sentience_claims(value, where=label)


def _require_ref(value: str, namespace: str) -> None:
    prefix = f"{namespace}:"
    body = value[len(prefix) :]
    if (
        not value.startswith(prefix)
        or not body
        or not all(ch.islower() or ch.isdigit() or ch in "._:/-" for ch in body)
    ):
        raise ValueError(f"reference {value!r} must start with {prefix!r} and use stable characters")


def _require_claim_scope(value: str) -> None:
    if value != CLAIM_SCOPE:
        raise ValueError("ecology claim scope cannot be widened or dropped")


# ---------------------------------------------------------------------------
# (a) Bounded world generation contract (RA5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorldResamplingRule:
    """Declared rule separating adaptation from map or state memorization.

    Held-out worlds are freshly derived layouts on the declared resample axis, so a policy that
    memorizes one map cannot transfer by lookup. The memorization control names the arm that is
    allowed to memorize (replay-only) and must therefore fail on held-out worlds.
    """

    train_worlds: int
    held_out_worlds: int
    resample_axis: str
    memorization_control: str = "replay-only"

    def __post_init__(self) -> None:
        if self.train_worlds < 2:
            raise ValueError("at least two training worlds are required")
        if self.held_out_worlds < 1:
            raise ValueError("at least one held-out world is required")
        _require_clean_text(self.resample_axis, "resample_axis")
        if self.memorization_control not in ADAPTATION_CONTROLS:
            raise ValueError("memorization control must be a declared adaptation control")

    def world_seeds(self, family_seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if family_seed < 0:
            raise ValueError("family seed must be nonnegative")
        train = tuple(_derive_world_seed(family_seed, "train", i) for i in range(self.train_worlds))
        held = tuple(_derive_world_seed(family_seed, "held-out", i) for i in range(self.held_out_worlds))
        combined = train + held
        if len(set(combined)) != len(combined):
            raise EcologyRefusal(
                "identity-collision",
                "world seed collision; resampling cannot separate adaptation from memorization",
            )
        return train, held

    def payload(self) -> dict[str, Any]:
        return {
            "train_worlds": self.train_worlds,
            "held_out_worlds": self.held_out_worlds,
            "resample_axis": self.resample_axis,
            "memorization_control": self.memorization_control,
        }


@dataclass(frozen=True, slots=True)
class HiddenDynamicsDeclaration:
    """Declares which world parameters stay hidden from the observation channel."""

    hidden_parameters: tuple[str, ...]
    tool_count: int
    damage_states: int
    sensor_cost_per_probe: int
    observable_leak_forbidden: bool

    def __post_init__(self) -> None:
        if not self.hidden_parameters:
            raise ValueError("at least one hidden parameter must be declared")
        if len(set(self.hidden_parameters)) != len(self.hidden_parameters):
            raise ValueError("hidden parameter names must be unique")
        for name in self.hidden_parameters:
            _require_clean_text(name, "hidden parameter name")
        if self.tool_count < 1 or self.damage_states < 2 or self.sensor_cost_per_probe < 1:
            raise ValueError("tools, damage states, and sensor cost must be declared and positive")
        if not self.observable_leak_forbidden:
            raise ValueError("hidden dynamics may never leak into the observation channel")

    def payload(self) -> dict[str, Any]:
        return {
            "hidden_parameters": list(self.hidden_parameters),
            "tool_count": self.tool_count,
            "damage_states": self.damage_states,
            "sensor_cost_per_probe": self.sensor_cost_per_probe,
            "observable_leak_forbidden": self.observable_leak_forbidden,
        }


@dataclass(frozen=True, slots=True)
class BoundedWorldContract:
    """Bounded world generation contract for RA5.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    family_seed: int
    resampling: WorldResamplingRule
    hidden: HiddenDynamicsDeclaration
    adaptation_controls: tuple[str, ...] = ADAPTATION_CONTROLS
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.family_seed < 0:
            raise ValueError("family seed must be nonnegative")
        if tuple(self.adaptation_controls) != ADAPTATION_CONTROLS:
            raise ValueError("adaptation control set or order drift")
        _require_claim_scope(self.claim_scope)
        self.resampling.world_seeds(self.family_seed)

    def world_fixture(self) -> dict[str, Any]:
        """Instantiate deterministic persistent-grid worlds for every declared seed."""

        train_seeds, held_seeds = self.resampling.world_seeds(self.family_seed)

        def build(seed: int) -> WorldSpec:
            return make_world_spec(seed=seed, grid_size=5 + seed % 3)

        train_specs = [build(seed) for seed in train_seeds]
        held_specs = [build(seed) for seed in held_seeds]
        hashes = [spec.sha256 for spec in train_specs + held_specs]
        if len(set(hashes)) != len(hashes):
            raise EcologyRefusal("identity-collision", "world family produced duplicate world hashes")
        return {
            "family_seed": self.family_seed,
            "train_world_sha256": [spec.sha256 for spec in train_specs],
            "held_out_world_sha256": [spec.sha256 for spec in held_specs],
            "claim_scope": self.claim_scope,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "family_seed": self.family_seed,
            "resampling": self.resampling.payload(),
            "hidden": self.hidden.payload(),
            "adaptation_controls": list(self.adaptation_controls),
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# (b) Active perception contract (RA6, rows f22 and f28)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensingCostDeclaration:
    """One sensing action with declared sensing, motion, latency, and compute charges."""

    action: str
    sensing_units: int
    motion_units: int
    latency_ticks: int
    compute_units: int

    def __post_init__(self) -> None:
        if self.action not in SENSING_ACTIONS:
            raise ValueError(f"unsupported sensing action {self.action!r}")
        charges = (self.sensing_units, self.motion_units, self.latency_ticks, self.compute_units)
        if any(value < 0 for value in charges):
            raise ValueError("sensing charges must be nonnegative")
        if self.action == "abstain":
            if self.sensing_units != 0 or self.motion_units != 0:
                raise ValueError("abstention cannot charge sensing or motion")
        elif sum(charges) == 0:
            raise ValueError("non-abstention sensing actions must charge at least one unit")

    def payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "sensing_units": self.sensing_units,
            "motion_units": self.motion_units,
            "latency_ticks": self.latency_ticks,
            "compute_units": self.compute_units,
        }


@dataclass(frozen=True, slots=True)
class ValueForecastDeclaration:
    """Pre-cost sensor value forecast declaration for row f28.

    The forecast must be committed before the sensing charge is paid; a post-hoc value control is
    mandatory so hindsight scoring can never impersonate a forecast.
    """

    forecast_metric: str
    horizon_steps: int
    pre_cost_commitment: bool
    control: str

    def __post_init__(self) -> None:
        if self.forecast_metric != "expected-information-gain":
            raise ValueError("sensor value forecast metric drift")
        if self.horizon_steps < 1:
            raise ValueError("forecast horizon must be positive")
        if not self.pre_cost_commitment:
            raise ValueError("value forecasts must be committed before the sensing cost is paid")
        if self.control != "post-hoc-value":
            raise ValueError("value forecast requires the post-hoc-value control")

    def payload(self) -> dict[str, Any]:
        return {
            "forecast_metric": self.forecast_metric,
            "horizon_steps": self.horizon_steps,
            "pre_cost_commitment": self.pre_cost_commitment,
            "control": self.control,
        }


@dataclass(frozen=True, slots=True)
class ActivePerceptionContract:
    """Costed acquisition contract for RA6 and row f22.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    costs: tuple[SensingCostDeclaration, ...]
    value_forecast: ValueForecastDeclaration
    controls: tuple[str, ...] = ACQUISITION_CONTROLS
    primary_arm: str = "information-gain"
    require_untouched_downstream_gain: bool = True
    require_cross_world_transfer: bool = True
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if tuple(row.action for row in self.costs) != SENSING_ACTIONS:
            raise ValueError("sensing cost declarations must cover view, time, audio, abstain in order")
        if tuple(self.controls) != ACQUISITION_CONTROLS:
            raise ValueError("acquisition control set or order drift")
        if self.primary_arm != "information-gain":
            raise ValueError("primary acquisition arm drift")
        if not self.require_untouched_downstream_gain or not self.require_cross_world_transfer:
            raise ValueError("downstream gain and cross-world transfer requirements cannot be waived")
        _require_claim_scope(self.claim_scope)

    def payload(self) -> dict[str, Any]:
        return {
            "costs": [row.payload() for row in self.costs],
            "value_forecast": self.value_forecast.payload(),
            "controls": list(self.controls),
            "primary_arm": self.primary_arm,
            "require_untouched_downstream_gain": self.require_untouched_downstream_gain,
            "require_cross_world_transfer": self.require_cross_world_transfer,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# (c) Autotelic and quality-diverse ecology contract (PA7, rows f50, f51, f52)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalRecord:
    """One candidate goal with the fields the refusal rules inspect."""

    goal_ref: str
    descriptor: tuple[int, ...]
    quality: float
    reward_source: str
    target_kind: str
    unsafe: bool

    def __post_init__(self) -> None:
        _require_ref(self.goal_ref, "goal")
        if not self.descriptor:
            raise ValueError("goal descriptor must not be empty")
        if any(value < 0 for value in self.descriptor):
            raise ValueError("goal descriptor cells must be nonnegative")
        if not math.isfinite(self.quality) or self.quality < 0.0:
            raise ValueError("goal quality must be finite and nonnegative")
        _require_clean_text(self.reward_source, "reward_source")
        _require_clean_text(self.target_kind, "target_kind")

    def payload(self) -> dict[str, Any]:
        return {
            "goal_ref": self.goal_ref,
            "descriptor": list(self.descriptor),
            "quality": self.quality,
            "reward_source": self.reward_source,
            "target_kind": self.target_kind,
            "unsafe": self.unsafe,
        }


def guard_noisy_tv(goal: GoalRecord) -> None:
    """Refuse goals that target a declared irreducible noise source."""

    if goal.target_kind == "noise-source":
        raise EcologyRefusal("noisy-tv", f"goal {goal.goal_ref} targets an irreducible noise source")


def guard_reward_hacking(goal: GoalRecord) -> None:
    """Refuse goals whose reward is not bound to an environment consequence."""

    if goal.reward_source != "environment-consequence":
        raise EcologyRefusal(
            "reward-hacking",
            f"goal {goal.goal_ref} reward source {goal.reward_source!r} is not consequence-bound",
        )


def guard_unsafe_goal(goal: GoalRecord) -> None:
    """Refuse goals carrying the declared unsafe flag."""

    if goal.unsafe:
        raise EcologyRefusal("unsafe-goal", f"goal {goal.goal_ref} is flagged unsafe")


class GoalArchive:
    """Quality-diverse goal archive keyed by descriptor cell.

    Admission runs every guard first, then either fills a new cell, improves an existing cell, or
    refuses with archive-bloat when a new cell would exceed the declared capacity.
    """

    def __init__(self, *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("archive capacity must be positive")
        self.capacity = capacity
        self._cells: dict[tuple[int, ...], GoalRecord] = {}

    def admit(self, goal: GoalRecord) -> bool:
        guard_noisy_tv(goal)
        guard_reward_hacking(goal)
        guard_unsafe_goal(goal)
        existing = self._cells.get(goal.descriptor)
        if existing is None:
            if len(self._cells) >= self.capacity:
                raise EcologyRefusal(
                    "archive-bloat",
                    f"admitting {goal.goal_ref} would exceed the declared capacity {self.capacity}",
                )
            self._cells[goal.descriptor] = goal
            return True
        if goal.quality > existing.quality:
            self._cells[goal.descriptor] = goal
            return True
        return False

    def distinct_cells(self) -> int:
        return len(self._cells)

    def payload(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "cells": [self._cells[key].payload() for key in sorted(self._cells)],
        }


@dataclass(frozen=True, slots=True)
class CurriculumDeclaration:
    """Learning-progress goldilocks band for row f50.

    Tasks are selectable only when their measured pass rate sits strictly inside the declared
    band, so both trivially easy and currently impossible tasks are excluded by construction.
    """

    selection_signal: str
    too_hard_pass_rate: float
    too_easy_pass_rate: float
    controls: tuple[str, ...] = CURRICULUM_CONTROLS

    def __post_init__(self) -> None:
        if self.selection_signal != "learning-progress":
            raise ValueError("curriculum selection signal drift")
        if not (0.0 <= self.too_hard_pass_rate < self.too_easy_pass_rate <= 1.0):
            raise ValueError("curriculum band bounds must satisfy 0 <= hard < easy <= 1")
        if tuple(self.controls) != CURRICULUM_CONTROLS:
            raise ValueError("curriculum control set or order drift")

    def in_band(self, pass_rate: float) -> bool:
        if not math.isfinite(pass_rate) or not (0.0 <= pass_rate <= 1.0):
            raise ValueError("pass rate must be a finite value in [0, 1]")
        return self.too_hard_pass_rate < pass_rate < self.too_easy_pass_rate

    def payload(self) -> dict[str, Any]:
        return {
            "selection_signal": self.selection_signal,
            "too_hard_pass_rate": self.too_hard_pass_rate,
            "too_easy_pass_rate": self.too_easy_pass_rate,
            "controls": list(self.controls),
        }


@dataclass(frozen=True, slots=True)
class StopRuleConfig:
    """Thresholds for the four declared ecology stop rules."""

    plateau_window: int
    min_new_cells_per_window: int
    diversity_floor: int
    bloat_limit: int

    def __post_init__(self) -> None:
        if self.plateau_window < 2:
            raise ValueError("plateau window must cover at least two steps")
        if self.min_new_cells_per_window < 1:
            raise ValueError("plateau rule requires at least one new cell per window")
        if self.diversity_floor < 1 or self.bloat_limit < 1:
            raise ValueError("diversity floor and bloat limit must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "plateau_window": self.plateau_window,
            "min_new_cells_per_window": self.min_new_cells_per_window,
            "diversity_floor": self.diversity_floor,
            "bloat_limit": self.bloat_limit,
        }


_HISTORY_KEYS = ("new_cells", "distinct_cells", "archive_size", "unsafe_flag")


def evaluate_stop_rules(config: StopRuleConfig, history: Sequence[dict[str, Any]]) -> str | None:
    """Return the first triggered stop rule name, or None. Malformed history raises."""

    if not history:
        raise ValueError("stop rule evaluation requires at least one history row")
    for row in history:
        if set(row) != set(_HISTORY_KEYS):
            raise ValueError("history rows must carry exactly the declared keys")
        if any(not isinstance(row[key], int) or row[key] < 0 for key in _HISTORY_KEYS[:3]):
            raise ValueError("history counts must be nonnegative integers")
        if not isinstance(row["unsafe_flag"], bool):
            raise ValueError("history unsafe flag must be boolean")
    if any(row["unsafe_flag"] for row in history):
        return "unsafe-goal"
    if history[-1]["archive_size"] > config.bloat_limit:
        return "archive-bloat"
    if len(history) >= config.plateau_window:
        if history[-1]["distinct_cells"] < config.diversity_floor:
            return "collapse"
        window = history[-config.plateau_window :]
        if sum(row["new_cells"] for row in window) < config.min_new_cells_per_window:
            return "plateau"
    return None


@dataclass(frozen=True, slots=True)
class AutotelicEcologyContract:
    """Autotelic goal ecology contract for PA7 and rows f50, f51, f52.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    archive_capacity: int
    stop: StopRuleConfig
    curriculum: CurriculumDeclaration
    guards: tuple[str, ...] = ECOLOGY_GUARDS
    stop_rules: tuple[str, ...] = ECOLOGY_STOP_RULES
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.archive_capacity < 1:
            raise ValueError("archive capacity must be positive")
        if self.stop.bloat_limit < self.archive_capacity:
            raise ValueError("bloat limit cannot be below the archive capacity")
        if tuple(self.guards) != ECOLOGY_GUARDS:
            raise ValueError("ecology guard set or order drift")
        if tuple(self.stop_rules) != ECOLOGY_STOP_RULES:
            raise ValueError("ecology stop rule set or order drift")
        _require_claim_scope(self.claim_scope)

    def payload(self) -> dict[str, Any]:
        return {
            "archive_capacity": self.archive_capacity,
            "stop": self.stop.payload(),
            "curriculum": self.curriculum.payload(),
            "guards": list(self.guards),
            "stop_rules": list(self.stop_rules),
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def run_goal_babbling_fixture(
    contract: AutotelicEcologyContract, *, seed: int, steps: int = 24
) -> dict[str, Any]:
    """Deterministic safe-play fixture for row f51.

    Generates seeded candidate goals, routes each through the guarded archive, records every
    refusal by rule name, and evaluates the stop rules after every step. This exercises the
    scaffold mechanics only; it measures nothing about learned behavior.
    """

    if seed < 0:
        raise ValueError("babbling seed must be nonnegative")
    if steps < contract.stop.plateau_window:
        raise ValueError("babbling must run at least one full plateau window")
    archive = GoalArchive(capacity=contract.archive_capacity)
    refusal_counts = dict.fromkeys(REFUSAL_RULES, 0)
    history: list[dict[str, Any]] = []
    stop_reason: str | None = None
    executed_steps = 0
    for step in range(steps):
        draw = _stable_int(seed, f"goal-kind-{step}", 8)
        goal = GoalRecord(
            goal_ref=f"goal:babble-{seed}-{step}",
            descriptor=(
                _stable_int(seed, f"cell-x-{step}", 3),
                _stable_int(seed, f"cell-y-{step}", 3),
            ),
            quality=_stable_int(seed, f"quality-{step}", 100) / 100.0,
            reward_source="self-scored" if draw == 1 else "environment-consequence",
            target_kind="noise-source" if draw == 0 else "environment-entity",
            unsafe=False,
        )
        before = archive.distinct_cells()
        try:
            archive.admit(goal)
        except EcologyRefusal as refusal:
            refusal_counts[refusal.rule] += 1
        history.append(
            {
                "new_cells": archive.distinct_cells() - before,
                "distinct_cells": archive.distinct_cells(),
                "archive_size": archive.distinct_cells(),
                "unsafe_flag": False,
            }
        )
        executed_steps = step + 1
        stop_reason = evaluate_stop_rules(contract.stop, history)
        if stop_reason is not None:
            break
    trace = {
        "schema": "mop-goal-babbling-trace/v1",
        "seed": seed,
        "steps_requested": steps,
        "steps_executed": executed_steps,
        "refusal_counts": refusal_counts,
        "archive": archive.payload(),
        "stop_reason": stop_reason,
        "contract_sha256": contract.sha256,
        "claim_scope": CLAIM_SCOPE,
    }
    trace["trace_sha256"] = canonical_sha256(trace)
    return trace


# ---------------------------------------------------------------------------
# (d) Simulated partner contract (PA8, rows f53, f54, f55, f56, f58)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartnerSpec:
    """One simulated partner with private observations and a held-out flag."""

    partner_ref: str
    policy_ref: str
    private_observation_keys: tuple[str, ...]
    held_out: bool

    def __post_init__(self) -> None:
        _require_ref(self.partner_ref, "partner")
        _require_ref(self.policy_ref, "policy")
        if not self.private_observation_keys:
            raise ValueError("partners must declare at least one private observation key")
        if len(set(self.private_observation_keys)) != len(self.private_observation_keys):
            raise ValueError("private observation keys must be unique")

    def payload(self) -> dict[str, Any]:
        return {
            "partner_ref": self.partner_ref,
            "policy_ref": self.policy_ref,
            "private_observation_keys": list(self.private_observation_keys),
            "held_out": self.held_out,
        }


@dataclass(frozen=True, slots=True)
class PartnerExperimentDeclaration:
    """One declared social experiment with its metric, null, and mandatory control."""

    name: str
    metric: str
    null_statement: str
    control: str

    def __post_init__(self) -> None:
        if self.name not in PARTNER_EXPERIMENTS:
            raise ValueError(f"unsupported partner experiment {self.name!r}")
        _require_clean_text(self.metric, "experiment metric")
        _require_clean_text(self.null_statement, "experiment null statement")
        required = REQUIRED_EXPERIMENT_CONTROLS[self.name]
        if self.control != required:
            raise ValueError(f"experiment {self.name} requires the {required} control")

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "null_statement": self.null_statement,
            "control": self.control,
        }


@dataclass(frozen=True, slots=True)
class SimulatedPartnerContract:
    """Simulated partner contract for PA8.

    The pattern-matching control pins the honest alternative reading: apparent partner modeling
    may be regularity matching over the partner policy family, so the control arm is mandatory.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    partners: tuple[PartnerSpec, ...]
    learner_visible_keys: tuple[str, ...]
    experiments: tuple[PartnerExperimentDeclaration, ...]
    partner_model_control: str = PARTNER_POLICY_CONTROL
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if len(self.partners) < 2:
            raise ValueError("at least two simulated partners are required")
        refs = [row.partner_ref for row in self.partners]
        policies = [row.policy_ref for row in self.partners]
        if len(set(refs)) != len(refs) or len(set(policies)) != len(policies):
            raise ValueError("partner and policy references must be unique")
        held = [row.held_out for row in self.partners]
        if not any(held) or all(held):
            raise ValueError("partners must include both training and held-out policies")
        if not self.learner_visible_keys or len(set(self.learner_visible_keys)) != len(
            self.learner_visible_keys
        ):
            raise ValueError("learner visible keys must be declared and unique")
        visible = set(self.learner_visible_keys)
        for row in self.partners:
            if visible & set(row.private_observation_keys):
                raise ValueError(f"partner {row.partner_ref} private observations leak to the learner")
        names = [row.name for row in self.experiments]
        if sorted(names) != sorted(PARTNER_EXPERIMENTS) or len(set(names)) != len(names):
            raise ValueError("partner experiment declarations must cover each experiment exactly once")
        if self.partner_model_control != PARTNER_POLICY_CONTROL:
            raise ValueError("partner-policy pattern-matching control cannot be dropped")
        _require_claim_scope(self.claim_scope)

    def payload(self) -> dict[str, Any]:
        return {
            "partners": [row.payload() for row in self.partners],
            "learner_visible_keys": list(self.learner_visible_keys),
            "experiments": [row.payload() for row in self.experiments],
            "partner_model_control": self.partner_model_control,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# (e) Communication grounding contract (PA9, row f57)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MessageBindingDeclaration:
    """One message bound to a referent whose namespace must match the binding kind."""

    message_ref: str
    binding: str
    referent_ref: str

    def __post_init__(self) -> None:
        _require_ref(self.message_ref, "message")
        if self.binding not in MESSAGE_BINDINGS:
            raise ValueError(f"unsupported message binding {self.binding!r}")
        _require_ref(self.referent_ref, self.binding)

    def payload(self) -> dict[str, Any]:
        return {
            "message_ref": self.message_ref,
            "binding": self.binding,
            "referent_ref": self.referent_ref,
        }


@dataclass(frozen=True, slots=True)
class CommunicationGroundingContract:
    """Communication grounding contract for PA9 and row f57.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    bindings: tuple[MessageBindingDeclaration, ...]
    channel_bits: int
    equal_bandwidth_bits: int
    controls: tuple[str, ...] = CHANNEL_CONTROLS
    transfer_requirements: tuple[str, ...] = TRANSFER_REQUIREMENTS
    score_axes: tuple[str, ...] = CODE_SCORE_AXES
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        refs = [row.message_ref for row in self.bindings]
        if len(set(refs)) != len(refs):
            raise ValueError("message references must be unique")
        covered = {row.binding for row in self.bindings}
        if covered != set(MESSAGE_BINDINGS):
            raise ValueError("messages must bind every declared referent kind at least once")
        if self.channel_bits < 1:
            raise ValueError("channel budget must be positive")
        if self.equal_bandwidth_bits != self.channel_bits:
            raise ValueError("the equal-bandwidth control must share the exact channel budget")
        if tuple(self.controls) != CHANNEL_CONTROLS:
            raise ValueError("communication control set or order drift")
        if tuple(self.transfer_requirements) != TRANSFER_REQUIREMENTS:
            raise ValueError("transfer requirement set or order drift")
        if tuple(self.score_axes) != CODE_SCORE_AXES:
            raise ValueError("code scoring axes set or order drift")
        _require_claim_scope(self.claim_scope)

    def payload(self) -> dict[str, Any]:
        return {
            "bindings": [row.payload() for row in self.bindings],
            "channel_bits": self.channel_bits,
            "equal_bandwidth_bits": self.equal_bandwidth_bits,
            "controls": list(self.controls),
            "transfer_requirements": list(self.transfer_requirements),
            "score_axes": list(self.score_axes),
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Bundle and deterministic fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EcologyScaffoldBundle:
    """All five interactive ecology contracts under one content-addressed identity.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    seed: int
    world: BoundedWorldContract
    perception: ActivePerceptionContract
    ecology: AutotelicEcologyContract
    partner: SimulatedPartnerContract
    communication: CommunicationGroundingContract
    schema: str = field(default=ECOLOGY_SCHEMA)

    def __post_init__(self) -> None:
        if self.schema != ECOLOGY_SCHEMA:
            raise ValueError(f"unsupported ecology scaffold schema {self.schema!r}")
        if self.seed < 0:
            raise ValueError("bundle seed must be nonnegative")
        if self.world.family_seed != self.seed:
            raise ValueError("world family seed must match the bundle seed")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "world": self.world.payload(),
            "world_sha256": self.world.sha256,
            "perception": self.perception.payload(),
            "perception_sha256": self.perception.sha256,
            "ecology": self.ecology.payload(),
            "ecology_sha256": self.ecology.sha256,
            "partner": self.partner.payload(),
            "partner_sha256": self.partner.sha256,
            "communication": self.communication.payload(),
            "communication_sha256": self.communication.sha256,
            "claim_scope": CLAIM_SCOPE,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def make_ecology_fixture(seed: int) -> EcologyScaffoldBundle:
    """Build the deterministic toy fixture bundle for one nonnegative seed."""

    if seed < 0:
        raise ValueError("fixture seed must be nonnegative")
    prefix = canonical_sha256({"seed": seed, "fixture": "interactive-ecology"})[:12]

    world = BoundedWorldContract(
        family_seed=seed,
        resampling=WorldResamplingRule(
            train_worlds=3,
            held_out_worlds=2,
            resample_axis="layout-and-goal-distribution",
        ),
        hidden=HiddenDynamicsDeclaration(
            hidden_parameters=("tool-effect", "damage-rate", "sensor-cost"),
            tool_count=2,
            damage_states=3,
            sensor_cost_per_probe=1,
            observable_leak_forbidden=True,
        ),
    )
    perception = ActivePerceptionContract(
        costs=(
            SensingCostDeclaration("view", 2, 1, 1, 1),
            SensingCostDeclaration("time", 1, 0, 2 + _stable_int(seed, "time-latency", 2), 1),
            SensingCostDeclaration("audio", 1, 0, 1, 2),
            SensingCostDeclaration("abstain", 0, 0, 0, 0),
        ),
        value_forecast=ValueForecastDeclaration(
            forecast_metric="expected-information-gain",
            horizon_steps=1 + _stable_int(seed, "forecast-horizon", 3),
            pre_cost_commitment=True,
            control="post-hoc-value",
        ),
    )
    capacity = 6 + _stable_int(seed, "archive-capacity", 3)
    ecology = AutotelicEcologyContract(
        archive_capacity=capacity,
        stop=StopRuleConfig(
            plateau_window=4,
            min_new_cells_per_window=1,
            diversity_floor=2,
            bloat_limit=capacity,
        ),
        curriculum=CurriculumDeclaration(
            selection_signal="learning-progress",
            too_hard_pass_rate=0.2,
            too_easy_pass_rate=0.8,
        ),
    )
    partner = SimulatedPartnerContract(
        partners=(
            PartnerSpec(
                partner_ref=f"partner:train-a-{prefix}",
                policy_ref=f"policy:train-a-{prefix}",
                private_observation_keys=("private-cue-a",),
                held_out=False,
            ),
            PartnerSpec(
                partner_ref=f"partner:train-b-{prefix}",
                policy_ref=f"policy:train-b-{prefix}",
                private_observation_keys=("private-cue-b",),
                held_out=False,
            ),
            PartnerSpec(
                partner_ref=f"partner:held-out-{prefix}",
                policy_ref=f"policy:held-out-{prefix}",
                private_observation_keys=("private-cue-h",),
                held_out=True,
            ),
        ),
        learner_visible_keys=("public-state", "public-message"),
        experiments=(
            PartnerExperimentDeclaration(
                name="joint-attention",
                metric="joint-referent-agreement",
                null_statement="agreement never exceeds the cue-blind partner control",
                control="cue-blind-partner",
            ),
            PartnerExperimentDeclaration(
                name="communicative-repair",
                metric="post-repair-task-recovery",
                null_statement="recovery never exceeds the no-repair-channel control",
                control="no-repair-channel",
            ),
            PartnerExperimentDeclaration(
                name="selective-imitation",
                metric="copied-step-usefulness",
                null_statement="usefulness never exceeds the indiscriminate imitation control",
                control="indiscriminate-imitation",
            ),
            PartnerExperimentDeclaration(
                name="teaching-vs-equal-information",
                metric="learner-gain-per-message",
                null_statement="teaching gain never exceeds the equal-information control",
                control="equal-information",
            ),
            PartnerExperimentDeclaration(
                name="cumulative-convention",
                metric="cross-generation-convention-retention",
                null_statement="retention never exceeds the generation-reset control",
                control="generation-reset",
            ),
        ),
    )
    communication = CommunicationGroundingContract(
        bindings=(
            MessageBindingDeclaration(f"message:event-{prefix}", "event", f"event:eco-{prefix}"),
            MessageBindingDeclaration(f"message:action-{prefix}", "action", f"action:eco-{prefix}"),
            MessageBindingDeclaration(
                f"message:consequence-{prefix}", "consequence", f"consequence:eco-{prefix}"
            ),
            MessageBindingDeclaration(
                f"message:uncertainty-{prefix}", "uncertainty", f"uncertainty:eco-{prefix}"
            ),
            MessageBindingDeclaration(f"message:repair-{prefix}", "repair", f"repair:eco-{prefix}"),
        ),
        channel_bits=8,
        equal_bandwidth_bits=8,
    )
    return EcologyScaffoldBundle(
        seed=seed,
        world=world,
        perception=perception,
        ecology=ecology,
        partner=partner,
        communication=communication,
    )


def verify_ecology_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the fixture from its seed and compare canonical bytes. Fail closed on drift."""

    checks: dict[str, bool] = {}
    errors: list[str] = []
    checks["schema"] = payload.get("schema") == ECOLOGY_SCHEMA
    seed = payload.get("seed")
    checks["seed"] = isinstance(seed, int) and seed >= 0
    if not checks["schema"] or not checks["seed"]:
        return {"verified": False, "checks": checks, "errors": ["schema or seed malformed"]}
    assert isinstance(seed, int)
    rebuilt = make_ecology_fixture(seed).payload()
    checks["exact_rederivation"] = canonical_bytes(rebuilt) == canonical_bytes(payload)
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {"verified": all(checks.values()), "checks": checks, "errors": errors}
