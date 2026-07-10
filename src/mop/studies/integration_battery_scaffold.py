"""Contract scaffold for the workspace, self-model, and integration-theory cluster.

Scaffolding only: machine-checkable contracts, deterministic seeded fixture generators, declared
controls and nulls, and fail-closed refusal rules for facet PA6 (competing integration-theory
operationalization) with scaffold support for PA4 (limited-capacity broadcast) and PA5 (operational
self-model and metacognition). Claim scope everywhere: deterministic programmatic mechanics only;
no capability claim. Nothing here runs an experiment, loads weights, touches the network, or reads
the clock. The module raises on missing or malformed declarations rather than defaulting them.

Five contract families:

1. Theory battery (proposed rows around PA6): each registered theory declares DIVERGENT bounded
   predictions over the same operations at five levels (activation, behavior, lesion, restoration,
   construct-validity), plus precommitted disconfirming patterns and neighboring dissociations.
   A hard code rule keeps nonfunctional and moral-status interpretation out of the functional
   score: the scorer refuses any theory entry whose free text trips the north_star sentience rail
   and refuses any observation annotated with interpretation vocabulary.
2. Operational self-model contracts (f31 hardware and body model, f32 tool incorporation,
   f33 internal telemetry prediction, f34 homeostatic resource control): declared prediction
   targets over the telemetry vocabulary of studio/local_throttle.collect_host_telemetry. The
   field names are mirrored by NAME only; the heavy throttle machinery is never imported here.
3. Self-report grounding contract (f35): declared report fields over the same telemetry
   vocabulary, reusing the report_grounding metric names from diagnostics/operational_awareness.
4. Limited-broadcast necessity and sufficiency contracts (f36, f37): capacity-limited broadcast
   arm against an unrestricted-bus control plus mode-specific lesion/restoration or matched
   dense-state/depth/routing controls at a declared matched FLOP budget.
5. Metacognitive-efficiency contract (f38): per-component OA baselines and monitor cost budgets,
   reusing the OA component names from diagnostics/operational_awareness, with a code rule that
   refuses any composite score spanning more than one OA component.

House style: no em or en dashes; engineering vocabulary only. All free text in every contract is
gated by the north_star sentience rail at construction time and again inside the scorer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from ..devel.north_star import scan_text
from ..diagnostics.operational_awareness import OA_COMPONENTS
from ..experiments.expansion_harness import CLAIM_SCOPE
from ..substrate.events import canonical_sha256

THEORY_BATTERY_SCHEMA = "mop-integration-theory-battery/v1"
FUNCTIONAL_SCORE_SCHEMA = "mop-integration-functional-score/v1"
SELF_MODEL_SCHEMA = "mop-operational-self-model/v1"
REPORT_GROUNDING_SCHEMA = "mop-self-report-grounding/v1"
BROADCAST_CONTRACT_SCHEMA = "mop-limited-broadcast-contract/v1"
METACOG_EFFICIENCY_SCHEMA = "mop-metacognitive-efficiency/v1"

OPERATION_LEVELS = ("activation", "behavior", "lesion", "restoration", "construct-validity")
PREDICTION_DIRECTIONS = ("increase", "decrease", "no-change")

# Tokens that mark nonfunctional or moral-status interpretation. Any metric name, annotation key,
# or annotation value carrying one of these is refused by code, so the functional score can never
# absorb an interpretation claim. "experien" covers experience/experiential without colliding with
# "experiment"; "sentien" and "conscious" duplicate the rail on purpose (defense in depth).
INTERPRETATION_EXCLUDED_TOKENS = (
    "moral",
    "welfare",
    "phenomenal",
    "experien",
    "sentien",
    "conscious",
    "qualia",
    "suffer",
    "valence",
    "feel",
)

# Telemetry vocabulary mirrored BY NAME from studio/local_throttle.collect_host_telemetry (and its
# _mps/_thermal/_power/_processes probes). The throttle module is deliberately not imported: this
# scaffold only needs the field names as declared prediction targets.
TELEMETRY_FIELDS: Mapping[str, tuple[str, ...]] = {
    "cpu": (
        "logical_cpus",
        "load_1m",
        "load_5m",
        "load_15m",
        "load_1m_per_logical_cpu",
        "utilization_fraction",
    ),
    "memory": ("total_bytes", "available_bytes", "available_percent"),
    "swap": ("total_bytes", "used_bytes", "free_bytes", "used_gb", "percent"),
    "disk": ("total_bytes", "free_bytes", "free_gb"),
    "processes": ("foreground_resource_processes", "unmanaged_known_heavy", "inaccessible_processes"),
    "mps": (
        "current_allocated_bytes",
        "driver_allocated_bytes",
        "recommended_working_set_bytes",
        "declared_headroom_bytes",
    ),
    "thermal": ("status",),
    "power": ("on_ac", "battery_percent"),
}

# The numeric subset: forecast targets and homeostatic setpoints must be numbers, not lists or
# categorical strings.
TELEMETRY_NUMERIC_FIELDS: Mapping[str, tuple[str, ...]] = {
    "cpu": TELEMETRY_FIELDS["cpu"],
    "memory": TELEMETRY_FIELDS["memory"],
    "swap": TELEMETRY_FIELDS["swap"],
    "disk": TELEMETRY_FIELDS["disk"],
    "processes": ("inaccessible_processes",),
    "mps": TELEMETRY_FIELDS["mps"],
    "power": ("battery_percent",),
}

SELF_MODEL_KINDS = (
    "hardware-body",
    "tool-incorporation",
    "telemetry-prediction",
    "homeostatic-control",
)
TARGET_PHASES = ("standing", "pre-tool", "post-tool")

# Required control sets are FIXED tuples (order included), mirroring the Wave E0 harness rule that
# control drift fails closed instead of silently shrinking.
REQUIRED_SELF_MODEL_CONTROLS: Mapping[str, tuple[str, ...]] = {
    "hardware-body": ("boundary-shuffled", "wrong-channel", "constant-report"),
    "tool-incorporation": ("tool-detached", "wrong-tool", "phase-shuffled"),
    "telemetry-prediction": ("persistence-baseline", "rolling-mean-baseline", "shuffled-trace"),
    "homeostatic-control": ("matched-random-actuation", "no-actuation", "setpoint-shuffled"),
}

# Actuator vocabulary mirrored by name from the throttle governor's admission, lane, checkpoint,
# and cache levers. Declaring an actuator here claims nothing about its effect.
HOMEOSTATIC_ACTUATORS = (
    "defer-admission",
    "downshift-lane",
    "checkpoint-and-stop",
    "release-cache-bytes",
)

REPORT_GROUNDING_METRICS = ("grounded_fraction", "shared_fields")
REQUIRED_REPORT_GROUNDING_CONTROLS = (
    "shuffled-report",
    "wrong-trace",
    "stale-trace",
    "intervened-state-selective-change",
)

BROADCAST_MODES = ("necessity", "sufficiency")
REQUIRED_BROADCAST_CONTROLS: Mapping[str, tuple[str, ...]] = {
    "necessity": (
        "unrestricted-bus",
        "lesion-broadcast",
        "delay-broadcast",
        "restore-broadcast",
        "message-shuffled",
    ),
    "sufficiency": (
        "unrestricted-bus",
        "matched-dense-state",
        "feed-forward-depth",
        "independent-specialists",
        "equal-flop-routing",
    ),
}

METACOG_EFFICIENCY_METRIC = "benefit_per_monitor_flop"
REQUIRED_METACOG_CONTROLS = ("matched-cost-null", "no-monitor", "random-monitor-matched-rate")

_ID_RE = re.compile(r"^[a-z][a-z0-9_.:-]*$")


def _require_id(value: str, label: str) -> str:
    if _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable lowercase identifier, got {value!r}")
    return value


def _clean_text(text: str, where: str) -> str:
    """Fail closed on empty text, em or en dashes, and sentience-rail hits."""
    if not text.strip():
        raise ValueError(f"{where} must not be empty")
    if "\u2014" in text or "\u2013" in text:
        raise ValueError(f"{where} must not contain em or en dashes")
    hits = scan_text(text)
    if hits:
        joined = "; ".join(repr(h["match"]) for h in hits[:3])
        raise ValueError(f"{where} trips the sentience rail: {joined}")
    return text


def refuse_interpretation_tokens(values: Iterable[str], where: str) -> None:
    """Hard code rule: nonfunctional and moral-status vocabulary never enters the functional score."""
    for value in values:
        lowered = value.lower()
        for token in INTERPRETATION_EXCLUDED_TOKENS:
            if token in lowered:
                raise ValueError(
                    f"{where} carries interpretation vocabulary ({token!r} in {value!r}); "
                    "the functional score refuses nonfunctional and moral-status content"
                )


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------------------------
# 1. Competing integration-theory battery
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundedPrediction:
    """One theory's bounded, directional prediction for one operation at one level."""

    theory_id: str
    operation_id: str
    level: str
    metric: str
    direction: str
    lower_bound: float
    upper_bound: float
    rationale: str

    def __post_init__(self) -> None:
        _require_id(self.theory_id, "prediction theory_id")
        _require_id(self.operation_id, "prediction operation_id")
        _require_id(self.metric, "prediction metric")
        refuse_interpretation_tokens((self.metric,), "prediction metric")
        if self.level not in OPERATION_LEVELS:
            raise ValueError(f"prediction level {self.level!r} not in {OPERATION_LEVELS}")
        if self.direction not in PREDICTION_DIRECTIONS:
            raise ValueError(f"prediction direction {self.direction!r} not in {PREDICTION_DIRECTIONS}")
        _require_finite(self.lower_bound, "prediction lower_bound")
        _require_finite(self.upper_bound, "prediction upper_bound")
        if self.lower_bound > self.upper_bound:
            raise ValueError("prediction bounds are inverted; a bounded prediction needs lower <= upper")
        _clean_text(self.rationale, "prediction rationale")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.operation_id, self.level, self.metric)

    def payload(self) -> dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "operation_id": self.operation_id,
            "level": self.level,
            "metric": self.metric,
            "direction": self.direction,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class DisconfirmingPattern:
    """A precommitted observation pattern that counts AGAINST the theory if realized."""

    theory_id: str
    operation_id: str
    level: str
    metric: str
    disconfirming_direction: str
    note: str

    def __post_init__(self) -> None:
        _require_id(self.theory_id, "disconfirmer theory_id")
        _require_id(self.operation_id, "disconfirmer operation_id")
        _require_id(self.metric, "disconfirmer metric")
        if self.level not in OPERATION_LEVELS:
            raise ValueError(f"disconfirmer level {self.level!r} not in {OPERATION_LEVELS}")
        if self.disconfirming_direction not in PREDICTION_DIRECTIONS:
            raise ValueError(
                f"disconfirming direction {self.disconfirming_direction!r} not in {PREDICTION_DIRECTIONS}"
            )
        _clean_text(self.note, "disconfirmer note")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.operation_id, self.level, self.metric)

    def payload(self) -> dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "operation_id": self.operation_id,
            "level": self.level,
            "metric": self.metric,
            "disconfirming_direction": self.disconfirming_direction,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class NeighboringDissociation:
    """A declared neighboring pair: the moving operation shifts while the unaffected one holds."""

    theory_id: str
    moving_operation_id: str
    unaffected_operation_id: str
    level: str
    metric: str
    note: str

    def __post_init__(self) -> None:
        _require_id(self.theory_id, "dissociation theory_id")
        _require_id(self.moving_operation_id, "dissociation moving_operation_id")
        _require_id(self.unaffected_operation_id, "dissociation unaffected_operation_id")
        _require_id(self.metric, "dissociation metric")
        if self.moving_operation_id == self.unaffected_operation_id:
            raise ValueError("a dissociation needs two distinct neighboring operations")
        if self.level not in OPERATION_LEVELS:
            raise ValueError(f"dissociation level {self.level!r} not in {OPERATION_LEVELS}")
        _clean_text(self.note, "dissociation note")

    def payload(self) -> dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "moving_operation_id": self.moving_operation_id,
            "unaffected_operation_id": self.unaffected_operation_id,
            "level": self.level,
            "metric": self.metric,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class TheoryEntry:
    """One registered theory: predictions at all five levels plus its own falsification surface."""

    id: str
    name: str
    predictions: tuple[BoundedPrediction, ...]
    disconfirmers: tuple[DisconfirmingPattern, ...]
    dissociations: tuple[NeighboringDissociation, ...]

    def __post_init__(self) -> None:
        _require_id(self.id, "theory id")
        _clean_text(self.name, "theory name")
        if not self.predictions:
            raise ValueError(f"theory {self.id} declares no predictions")
        if not self.disconfirmers:
            raise ValueError(f"theory {self.id} declares no precommitted disconfirming pattern")
        if not self.dissociations:
            raise ValueError(f"theory {self.id} declares no neighboring dissociation")
        for row in self.predictions + self.disconfirmers + self.dissociations:
            if row.theory_id != self.id:
                raise ValueError(f"theory {self.id} contains a row bound to {row.theory_id!r}")
        covered = {row.level for row in self.predictions}
        missing = [level for level in OPERATION_LEVELS if level not in covered]
        if missing:
            raise ValueError(f"theory {self.id} misses prediction levels {missing}")
        keys = [row.key for row in self.predictions]
        if len(set(keys)) != len(keys):
            raise ValueError(f"theory {self.id} declares duplicate prediction keys")

    def free_text(self) -> list[str]:
        out = [self.name]
        out += [row.rationale for row in self.predictions]
        out += [row.note for row in self.disconfirmers]
        out += [row.note for row in self.dissociations]
        return out

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "predictions": [row.payload() for row in self.predictions],
            "disconfirmers": [row.payload() for row in self.disconfirmers],
            "dissociations": [row.payload() for row in self.dissociations],
        }


def _pair_diverges(a: TheoryEntry, b: TheoryEntry) -> bool:
    """Two theories diverge when a shared key predicts different directions or disjoint bounds."""
    index = {row.key: row for row in a.predictions}
    for row in b.predictions:
        other = index.get(row.key)
        if other is None:
            continue
        if other.direction != row.direction:
            return True
        if other.lower_bound > row.upper_bound or row.lower_bound > other.upper_bound:
            return True
    return False


@dataclass(frozen=True, slots=True)
class TheoryBatteryContract:
    """The battery: at least two theories with divergent bounded predictions over shared operations."""

    theories: tuple[TheoryEntry, ...]
    operations: tuple[str, ...]
    schema: str = THEORY_BATTERY_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != THEORY_BATTERY_SCHEMA:
            raise ValueError(f"unsupported theory battery schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("theory battery claim scope cannot be widened")
        if len(self.theories) < 2:
            raise ValueError("a competing battery needs at least two theories")
        if len({row.id for row in self.theories}) != len(self.theories):
            raise ValueError("theory ids must be unique")
        if not self.operations:
            raise ValueError("the battery declares no shared operations")
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("shared operation ids must be unique")
        for operation in self.operations:
            _require_id(operation, "shared operation id")
        allowed = set(self.operations)
        for theory in self.theories:
            for row in theory.predictions:
                if row.operation_id not in allowed:
                    raise ValueError(
                        f"theory {theory.id} predicts over undeclared operation {row.operation_id!r}"
                    )
            for pattern in theory.disconfirmers:
                if pattern.operation_id not in allowed:
                    raise ValueError(
                        f"theory {theory.id} disconfirmer uses undeclared operation {pattern.operation_id!r}"
                    )
            for dis in theory.dissociations:
                if dis.moving_operation_id not in allowed or dis.unaffected_operation_id not in allowed:
                    raise ValueError(f"theory {theory.id} dissociation uses an undeclared operation")
        for i, a in enumerate(self.theories):
            for b in self.theories[i + 1 :]:
                if not _pair_diverges(a, b):
                    raise ValueError(
                        f"theories {a.id} and {b.id} share no divergent bounded prediction; "
                        "a battery that cannot discriminate is refused"
                    )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "operations": list(self.operations),
            "theories": [row.payload() for row in self.theories],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class Observation:
    """One realized functional outcome for one (operation, level, metric) key."""

    operation_id: str
    level: str
    metric: str
    direction: str
    value: float
    annotations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.operation_id, "observation operation_id")
        _require_id(self.metric, "observation metric")
        if self.level not in OPERATION_LEVELS:
            raise ValueError(f"observation level {self.level!r} not in {OPERATION_LEVELS}")
        if self.direction not in PREDICTION_DIRECTIONS:
            raise ValueError(f"observation direction {self.direction!r} not in {PREDICTION_DIRECTIONS}")
        _require_finite(self.value, "observation value")
        flat: list[str] = []
        for pair in self.annotations:
            if len(pair) != 2:
                raise ValueError("observation annotations must be (key, value) string pairs")
            flat.extend(pair)
        refuse_interpretation_tokens(flat, "observation annotations")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.operation_id, self.level, self.metric)

    def payload(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "level": self.level,
            "metric": self.metric,
            "direction": self.direction,
            "value": self.value,
            "annotations": [list(pair) for pair in self.annotations],
        }


def score_functional_outcomes(
    contract: TheoryBatteryContract,
    observations: Sequence[Observation],
) -> dict[str, Any]:
    """Score each theory's bounded predictions against realized functional outcomes.

    Deterministic programmatic mechanics only; no capability claim. The score is purely
    functional: the scorer re-scans every theory's free text through the sentience rail and
    refuses interpretation vocabulary anywhere in observations or output keys, so moral-status
    and nonfunctional interpretation stay outside the number by construction.
    """
    if not observations:
        raise ValueError("the functional scorer needs at least one observation")
    index: dict[tuple[str, str, str], Observation] = {}
    for obs in observations:
        if obs.key in index:
            raise ValueError(f"duplicate observation for key {obs.key}")
        index[obs.key] = obs
    per_theory: dict[str, dict[str, Any]] = {}
    for theory in contract.theories:
        for text in theory.free_text():
            _clean_text(text, f"theory {theory.id} free text (scorer gate)")
        scored = [row for row in theory.predictions if row.key in index]
        if not scored:
            raise ValueError(f"theory {theory.id} has no scorable prediction under these observations")
        matches = 0
        for row in scored:
            obs = index[row.key]
            if obs.direction == row.direction and row.lower_bound <= obs.value <= row.upper_bound:
                matches += 1
        disconfirmed = [
            pattern.key
            for pattern in theory.disconfirmers
            if pattern.key in index and index[pattern.key].direction == pattern.disconfirming_direction
        ]
        dis_checked = 0
        dis_passed = 0
        for dis in theory.dissociations:
            moving = index.get((dis.moving_operation_id, dis.level, dis.metric))
            unaffected = index.get((dis.unaffected_operation_id, dis.level, dis.metric))
            if moving is None or unaffected is None:
                continue
            dis_checked += 1
            if moving.direction != "no-change" and unaffected.direction == "no-change":
                dis_passed += 1
        per_theory[theory.id] = {
            "n_predictions": len(theory.predictions),
            "n_scored": len(scored),
            "functional_match_fraction": matches / len(scored),
            "disconfirmed": bool(disconfirmed),
            "disconfirmed_keys": [list(key) for key in disconfirmed],
            "dissociations_checked": dis_checked,
            "dissociations_passed": dis_passed,
        }
    result = {
        "schema": FUNCTIONAL_SCORE_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "battery_sha256": contract.sha256,
        "n_observations": len(index),
        "interpretation_excluded_tokens": list(INTERPRETATION_EXCLUDED_TOKENS),
        "per_theory": per_theory,
    }
    refuse_interpretation_tokens(per_theory.keys(), "functional score theory keys")
    return result


# ---------------------------------------------------------------------------------------------
# 2. Operational self-model contracts (f31-f34)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionTarget:
    """One declared prediction target over the mirrored throttle telemetry vocabulary."""

    channel: str
    field_name: str
    horizon_steps: int
    tolerance: float
    phase: str = "standing"

    def __post_init__(self) -> None:
        fields = TELEMETRY_FIELDS.get(self.channel)
        if fields is None:
            raise ValueError(
                f"unknown telemetry channel {self.channel!r}; allowed {sorted(TELEMETRY_FIELDS)}"
            )
        if self.field_name not in fields:
            raise ValueError(f"unknown telemetry field {self.channel}.{self.field_name}; allowed {fields}")
        if self.horizon_steps < 1:
            raise ValueError("prediction horizon must be at least one step")
        _require_finite(self.tolerance, "prediction tolerance")
        if self.tolerance <= 0:
            raise ValueError("prediction tolerance must be positive")
        if self.phase not in TARGET_PHASES:
            raise ValueError(f"target phase {self.phase!r} not in {TARGET_PHASES}")

    @property
    def is_numeric(self) -> bool:
        return self.field_name in TELEMETRY_NUMERIC_FIELDS.get(self.channel, ())

    def payload(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "field": self.field_name,
            "horizon_steps": self.horizon_steps,
            "tolerance": self.tolerance,
            "phase": self.phase,
        }


@dataclass(frozen=True, slots=True)
class HomeostaticSetpoint:
    """A declared numeric operating band with a named actuator from the throttle vocabulary."""

    channel: str
    field_name: str
    lower: float
    upper: float
    actuator: str

    def __post_init__(self) -> None:
        numeric = TELEMETRY_NUMERIC_FIELDS.get(self.channel, ())
        if self.field_name not in numeric:
            raise ValueError(
                f"setpoint field {self.channel}.{self.field_name} is not a numeric telemetry field"
            )
        _require_finite(self.lower, "setpoint lower")
        _require_finite(self.upper, "setpoint upper")
        if self.lower >= self.upper:
            raise ValueError("setpoint band needs lower < upper")
        if self.actuator not in HOMEOSTATIC_ACTUATORS:
            raise ValueError(f"unknown actuator {self.actuator!r}; allowed {HOMEOSTATIC_ACTUATORS}")

    def payload(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "field": self.field_name,
            "lower": self.lower,
            "upper": self.upper,
            "actuator": self.actuator,
        }


@dataclass(frozen=True, slots=True)
class SelfModelContract:
    """One operational self-model contract (f31-f34) over the mirrored telemetry vocabulary.

    Deterministic programmatic mechanics only; no capability claim. "Self-model" here means a
    declared set of prediction targets about the host process and hardware, nothing more.
    """

    kind: str
    targets: tuple[PredictionTarget, ...]
    controls: tuple[str, ...]
    seed: int
    tool_id: str = ""
    setpoints: tuple[HomeostaticSetpoint, ...] = ()
    schema: str = SELF_MODEL_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != SELF_MODEL_SCHEMA:
            raise ValueError(f"unsupported self-model schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("self-model claim scope cannot be widened")
        if self.kind not in SELF_MODEL_KINDS:
            raise ValueError(f"unknown self-model kind {self.kind!r}; allowed {SELF_MODEL_KINDS}")
        if self.seed < 0:
            raise ValueError("self-model seed must be nonnegative")
        if not self.targets:
            raise ValueError(f"self-model contract {self.kind} declares no prediction targets")
        signature = [(row.channel, row.field_name, row.horizon_steps, row.phase) for row in self.targets]
        if len(set(signature)) != len(signature):
            raise ValueError("self-model prediction targets must be unique")
        if tuple(self.controls) != REQUIRED_SELF_MODEL_CONTROLS[self.kind]:
            raise ValueError(
                f"self-model control drift for {self.kind}: "
                f"required {REQUIRED_SELF_MODEL_CONTROLS[self.kind]}"
            )
        getattr(self, f"_check_{self.kind.replace('-', '_')}")()

    def _check_hardware_body(self) -> None:
        if self.tool_id or self.setpoints:
            raise ValueError("hardware-body contracts declare no tool and no setpoints")
        if any(row.phase != "standing" for row in self.targets):
            raise ValueError("hardware-body targets must use the standing phase")
        if len({row.channel for row in self.targets}) < 2:
            raise ValueError("a hardware and body model must span at least two telemetry channels")

    def _check_tool_incorporation(self) -> None:
        if self.setpoints:
            raise ValueError("tool-incorporation contracts declare no setpoints")
        _require_id(self.tool_id, "tool id")
        phases = {row.phase for row in self.targets}
        if not phases <= {"pre-tool", "post-tool"}:
            raise ValueError("tool-incorporation targets must be pre-tool or post-tool")
        pre = {(row.channel, row.field_name) for row in self.targets if row.phase == "pre-tool"}
        post = {(row.channel, row.field_name) for row in self.targets if row.phase == "post-tool"}
        if not pre or pre != post:
            raise ValueError("tool-incorporation needs matched pre-tool and post-tool targets per field")

    def _check_telemetry_prediction(self) -> None:
        if self.tool_id or self.setpoints:
            raise ValueError("telemetry-prediction contracts declare no tool and no setpoints")
        if any(row.phase != "standing" for row in self.targets):
            raise ValueError("telemetry-prediction targets must use the standing phase")
        if any(not row.is_numeric for row in self.targets):
            raise ValueError("telemetry-prediction targets must be numeric telemetry fields")
        if all(row.horizon_steps < 2 for row in self.targets):
            raise ValueError("telemetry-prediction needs at least one horizon of two or more steps")

    def _check_homeostatic_control(self) -> None:
        if self.tool_id:
            raise ValueError("homeostatic-control contracts declare no tool")
        if not self.setpoints:
            raise ValueError("homeostatic-control needs at least one declared setpoint")
        bands = [(row.channel, row.field_name) for row in self.setpoints]
        if len(set(bands)) != len(bands):
            raise ValueError("homeostatic setpoints must be unique per field")
        if any(row.phase != "standing" for row in self.targets):
            raise ValueError("homeostatic-control targets must use the standing phase")
        if any(not row.is_numeric for row in self.targets):
            raise ValueError("homeostatic-control targets must be numeric telemetry fields")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "kind": self.kind,
            "seed": self.seed,
            "tool_id": self.tool_id,
            "targets": [row.payload() for row in self.targets],
            "setpoints": [row.payload() for row in self.setpoints],
            "controls": list(self.controls),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------------------------
# 3. Self-report grounding contract (f35)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelfReportGroundingContract:
    """Declared report fields checked against the run trace (reuses OA8 report_grounding names).

    Deterministic programmatic mechanics only; no capability claim. Grounding means field-level
    agreement between a rendered report and traced telemetry, nothing about inner life.
    """

    report_fields: tuple[str, ...]
    controls: tuple[str, ...]
    seed: int
    metric_names: tuple[str, ...] = REPORT_GROUNDING_METRICS
    schema: str = REPORT_GROUNDING_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != REPORT_GROUNDING_SCHEMA:
            raise ValueError(f"unsupported report-grounding schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("report-grounding claim scope cannot be widened")
        if self.seed < 0:
            raise ValueError("report-grounding seed must be nonnegative")
        if tuple(self.metric_names) != REPORT_GROUNDING_METRICS:
            raise ValueError(f"report-grounding metric names must be {REPORT_GROUNDING_METRICS}")
        if tuple(self.controls) != REQUIRED_REPORT_GROUNDING_CONTROLS:
            raise ValueError(f"report-grounding controls must be {REQUIRED_REPORT_GROUNDING_CONTROLS}")
        if not self.report_fields:
            raise ValueError("report-grounding declares no report fields")
        if len(set(self.report_fields)) != len(self.report_fields):
            raise ValueError("report fields must be unique")
        for dotted in self.report_fields:
            channel, _, field_name = dotted.partition(".")
            if field_name not in TELEMETRY_FIELDS.get(channel, ()):
                raise ValueError(f"report field {dotted!r} is not in the telemetry vocabulary")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "seed": self.seed,
            "report_fields": list(self.report_fields),
            "metric_names": list(self.metric_names),
            "controls": list(self.controls),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------------------------
# 4. Limited-broadcast necessity and sufficiency contracts (f36, f37)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BroadcastExperimentContract:
    """Capacity-limited broadcast against an unrestricted bus at a matched FLOP budget.

    Deterministic programmatic mechanics only; no capability claim. Necessity mode declares
    lesion, delay, restoration, and shuffle operations on the broadcast link; sufficiency mode
    declares matched dense-state, depth, specialist, and equal-FLOP routing comparators.
    """

    mode: str
    capacity_slots: int
    bus_slots: int
    n_consumers: int
    recurrence_steps: int
    matched_flop_budget: int
    controls: tuple[str, ...]
    seed: int
    schema: str = BROADCAST_CONTRACT_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != BROADCAST_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported broadcast contract schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("broadcast claim scope cannot be widened")
        if self.mode not in BROADCAST_MODES:
            raise ValueError(f"broadcast mode {self.mode!r} not in {BROADCAST_MODES}")
        if self.capacity_slots < 1:
            raise ValueError("broadcast capacity must be at least one slot")
        if self.bus_slots <= self.capacity_slots:
            raise ValueError(
                "the unrestricted bus must have strictly more slots than the limited broadcast; "
                "otherwise the capacity limit is vacuous"
            )
        if self.n_consumers < 2:
            raise ValueError("broadcast needs at least two separated consumers")
        if self.recurrence_steps < 1:
            raise ValueError("broadcast needs at least one recurrence step")
        if self.matched_flop_budget < 1:
            raise ValueError("the matched FLOP budget must be positive and identical across arms")
        if self.seed < 0:
            raise ValueError("broadcast seed must be nonnegative")
        if tuple(self.controls) != REQUIRED_BROADCAST_CONTROLS[self.mode]:
            raise ValueError(
                f"broadcast control drift for {self.mode}: required {REQUIRED_BROADCAST_CONTROLS[self.mode]}"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "mode": self.mode,
            "capacity_slots": self.capacity_slots,
            "bus_slots": self.bus_slots,
            "n_consumers": self.n_consumers,
            "recurrence_steps": self.recurrence_steps,
            "matched_flop_budget": self.matched_flop_budget,
            "controls": list(self.controls),
            "seed": self.seed,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------------------------
# 5. Metacognitive-efficiency contract (f38)
# ---------------------------------------------------------------------------------------------


def refuse_composite_metric(metric_name: str, component_span: Sequence[str]) -> None:
    """Fail closed on any request to aggregate OA components into one number.

    Mirrors the oa_suite doctrine as code: a metric may describe exactly one OA component. Any
    span over two or more components is refused before it can become a headline score.
    """
    _require_id(metric_name, "metacognitive metric name")
    span = set(component_span)
    unknown = sorted(span - set(OA_COMPONENTS))
    if unknown:
        raise ValueError(f"unknown OA components {unknown}; allowed {OA_COMPONENTS}")
    if len(span) != 1:
        raise ValueError(
            f"metric {metric_name!r} spans {sorted(span)}; composite scores over multiple "
            "OA components are refused"
        )


@dataclass(frozen=True, slots=True)
class MetacognitiveEfficiencyContract:
    """Per-component monitoring benefit against declared monitor cost budgets (f38).

    Deterministic programmatic mechanics only; no capability claim. Reuses the OA component
    names from diagnostics/operational_awareness; every component carries its own named
    baseline and the efficiency metric is benefit per monitor FLOP, never a composite.
    """

    components: tuple[str, ...]
    baselines: tuple[tuple[str, str], ...]
    monitor_flop_budget: int
    monitor_seconds_budget: float
    controls: tuple[str, ...]
    seed: int
    efficiency_metric: str = METACOG_EFFICIENCY_METRIC
    schema: str = METACOG_EFFICIENCY_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != METACOG_EFFICIENCY_SCHEMA:
            raise ValueError(f"unsupported metacognitive-efficiency schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("metacognitive-efficiency claim scope cannot be widened")
        if self.efficiency_metric != METACOG_EFFICIENCY_METRIC:
            raise ValueError(f"the efficiency metric is fixed to {METACOG_EFFICIENCY_METRIC!r}")
        if not self.components:
            raise ValueError("metacognitive-efficiency declares no OA components")
        if len(set(self.components)) != len(self.components):
            raise ValueError("OA components must be unique")
        unknown = sorted(set(self.components) - set(OA_COMPONENTS))
        if unknown:
            raise ValueError(f"unknown OA components {unknown}; allowed {OA_COMPONENTS}")
        baseline_map = dict(self.baselines)
        if len(baseline_map) != len(self.baselines):
            raise ValueError("duplicate baseline declarations")
        missing = sorted(set(self.components) - set(baseline_map))
        if missing:
            raise ValueError(f"OA components without a named baseline: {missing}")
        extra = sorted(set(baseline_map) - set(self.components))
        if extra:
            raise ValueError(f"baselines for undeclared components: {extra}")
        for component, baseline in self.baselines:
            _require_id(baseline, f"baseline for {component}")
            if baseline == "none":
                raise ValueError(f"component {component} needs a real baseline, not 'none'")
            refuse_composite_metric(self.efficiency_metric, (component,))
        if self.monitor_flop_budget < 1:
            raise ValueError("monitor FLOP budget must be positive")
        _require_finite(self.monitor_seconds_budget, "monitor seconds budget")
        if self.monitor_seconds_budget <= 0:
            raise ValueError("monitor seconds budget must be positive")
        if tuple(self.controls) != REQUIRED_METACOG_CONTROLS:
            raise ValueError(f"metacognitive-efficiency controls must be {REQUIRED_METACOG_CONTROLS}")
        if self.seed < 0:
            raise ValueError("metacognitive-efficiency seed must be nonnegative")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "components": list(self.components),
            "baselines": [list(pair) for pair in self.baselines],
            "monitor_flop_budget": self.monitor_flop_budget,
            "monitor_seconds_budget": self.monitor_seconds_budget,
            "efficiency_metric": self.efficiency_metric,
            "controls": list(self.controls),
            "seed": self.seed,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------------------------
# Deterministic seeded fixture generators (no clock, no network, no weights)
# ---------------------------------------------------------------------------------------------

_FIXTURE_OPERATIONS = (
    "op:workspace-report",
    "op:distractor-recall",
    "op:cross-consumer-transfer",
)
_FIXTURE_METRIC = "transfer_accuracy"


def _fixture_theory(
    theory_id: str,
    name: str,
    lesion_direction: str,
    restoration_direction: str,
) -> TheoryEntry:
    predictions = []
    level_directions = {
        "activation": "increase",
        "behavior": "increase",
        "lesion": lesion_direction,
        "restoration": restoration_direction,
        "construct-validity": "increase",
    }
    for level, direction in level_directions.items():
        predictions.append(
            BoundedPrediction(
                theory_id=theory_id,
                operation_id="op:cross-consumer-transfer",
                level=level,
                metric=_FIXTURE_METRIC,
                direction=direction,
                lower_bound=0.0,
                upper_bound=1.0,
                rationale=f"bounded {level} prediction over the shared transfer operation",
            )
        )
    disconfirmers = (
        DisconfirmingPattern(
            theory_id=theory_id,
            operation_id="op:cross-consumer-transfer",
            level="lesion",
            metric=_FIXTURE_METRIC,
            disconfirming_direction="no-change" if lesion_direction == "decrease" else "decrease",
            note="the opposite lesion outcome counts against this theory by precommitment",
        ),
    )
    dissociations = (
        NeighboringDissociation(
            theory_id=theory_id,
            moving_operation_id="op:cross-consumer-transfer",
            unaffected_operation_id="op:distractor-recall",
            level="lesion",
            metric=_FIXTURE_METRIC,
            note="the lesion must move transfer while leaving distractor recall flat",
        ),
    )
    return TheoryEntry(
        id=theory_id,
        name=name,
        predictions=tuple(predictions),
        disconfirmers=disconfirmers,
        dissociations=dissociations,
    )


def make_theory_battery_fixture(seed: int = 0) -> tuple[TheoryBatteryContract, tuple[Observation, ...]]:
    """Build a two-theory divergent battery plus deterministic observations for the scorer."""
    if seed < 0:
        raise ValueError("fixture seed must be nonnegative")
    contract = TheoryBatteryContract(
        theories=(
            _fixture_theory(
                "theory:capacity-broadcast",
                "capacity-limited broadcast account (fixture stand-in)",
                lesion_direction="decrease",
                restoration_direction="increase",
            ),
            _fixture_theory(
                "theory:dense-integration",
                "dense shared-state account (fixture stand-in)",
                lesion_direction="no-change",
                restoration_direction="no-change",
            ),
        ),
        operations=_FIXTURE_OPERATIONS,
    )
    rng = Random(seed)
    observations = []
    for level in OPERATION_LEVELS:
        direction = "decrease" if level == "lesion" else "increase"
        observations.append(
            Observation(
                operation_id="op:cross-consumer-transfer",
                level=level,
                metric=_FIXTURE_METRIC,
                direction=direction,
                value=round(rng.uniform(0.0, 1.0), 6),
            )
        )
    observations.append(
        Observation(
            operation_id="op:distractor-recall",
            level="lesion",
            metric=_FIXTURE_METRIC,
            direction="no-change",
            value=round(rng.uniform(0.0, 1.0), 6),
        )
    )
    return contract, tuple(observations)


def make_telemetry_trace_fixture(seed: int = 0, steps: int = 8) -> tuple[dict[str, Any], ...]:
    """Deterministic synthetic telemetry snapshots over the numeric mirrored vocabulary."""
    if seed < 0:
        raise ValueError("fixture seed must be nonnegative")
    if steps < 1:
        raise ValueError("a telemetry trace needs at least one step")
    rng = Random(seed)
    state: dict[tuple[str, str], float] = {}
    for channel, fields in sorted(TELEMETRY_NUMERIC_FIELDS.items()):
        for field_name in fields:
            state[(channel, field_name)] = round(rng.uniform(0.0, 100.0), 6)
    trace: list[dict[str, Any]] = []
    for step in range(steps):
        snapshot: dict[str, Any] = {"step": step}
        for (channel, field_name), value in sorted(state.items()):
            drifted = max(0.0, value + rng.uniform(-1.0, 1.0))
            state[(channel, field_name)] = round(drifted, 6)
            snapshot.setdefault(channel, {})[field_name] = state[(channel, field_name)]
        trace.append(snapshot)
    return tuple(trace)


def make_hardware_body_contract(seed: int = 0) -> SelfModelContract:
    return SelfModelContract(
        kind="hardware-body",
        targets=(
            PredictionTarget("cpu", "utilization_fraction", horizon_steps=1, tolerance=0.05),
            PredictionTarget("memory", "available_percent", horizon_steps=1, tolerance=2.0),
            PredictionTarget("thermal", "status", horizon_steps=1, tolerance=1.0),
            PredictionTarget("processes", "inaccessible_processes", horizon_steps=1, tolerance=1.0),
        ),
        controls=REQUIRED_SELF_MODEL_CONTROLS["hardware-body"],
        seed=seed,
    )


def make_tool_incorporation_contract(seed: int = 0) -> SelfModelContract:
    return SelfModelContract(
        kind="tool-incorporation",
        targets=(
            PredictionTarget("mps", "driver_allocated_bytes", 1, 1.0e8, phase="pre-tool"),
            PredictionTarget("mps", "driver_allocated_bytes", 1, 1.0e8, phase="post-tool"),
            PredictionTarget("disk", "free_gb", 1, 1.0, phase="pre-tool"),
            PredictionTarget("disk", "free_gb", 1, 1.0, phase="post-tool"),
        ),
        controls=REQUIRED_SELF_MODEL_CONTROLS["tool-incorporation"],
        seed=seed,
        tool_id="tool:dense-cache-writer",
    )


def make_telemetry_prediction_contract(seed: int = 0) -> SelfModelContract:
    return SelfModelContract(
        kind="telemetry-prediction",
        targets=(
            PredictionTarget("cpu", "load_1m", horizon_steps=4, tolerance=0.5),
            PredictionTarget("memory", "available_bytes", horizon_steps=2, tolerance=5.0e8),
            PredictionTarget("swap", "used_gb", horizon_steps=2, tolerance=0.5),
        ),
        controls=REQUIRED_SELF_MODEL_CONTROLS["telemetry-prediction"],
        seed=seed,
    )


def make_homeostatic_control_contract(seed: int = 0) -> SelfModelContract:
    return SelfModelContract(
        kind="homeostatic-control",
        targets=(
            PredictionTarget("memory", "available_percent", horizon_steps=1, tolerance=2.0),
            PredictionTarget("swap", "used_gb", horizon_steps=1, tolerance=0.5),
        ),
        controls=REQUIRED_SELF_MODEL_CONTROLS["homeostatic-control"],
        seed=seed,
        setpoints=(
            HomeostaticSetpoint("memory", "available_percent", 20.0, 100.0, "defer-admission"),
            HomeostaticSetpoint("swap", "used_gb", 0.0, 4.0, "checkpoint-and-stop"),
            HomeostaticSetpoint("disk", "free_gb", 30.0, 4000.0, "release-cache-bytes"),
        ),
    )


def make_self_report_grounding_contract(seed: int = 0) -> SelfReportGroundingContract:
    return SelfReportGroundingContract(
        report_fields=(
            "cpu.utilization_fraction",
            "memory.available_percent",
            "swap.used_gb",
            "disk.free_gb",
            "thermal.status",
        ),
        controls=REQUIRED_REPORT_GROUNDING_CONTROLS,
        seed=seed,
    )


def make_broadcast_contract(mode: str, seed: int = 0) -> BroadcastExperimentContract:
    if mode not in BROADCAST_MODES:
        raise ValueError(f"broadcast mode {mode!r} not in {BROADCAST_MODES}")
    return BroadcastExperimentContract(
        mode=mode,
        capacity_slots=1,
        bus_slots=8,
        n_consumers=4,
        recurrence_steps=6,
        matched_flop_budget=1_000_000,
        controls=REQUIRED_BROADCAST_CONTROLS[mode],
        seed=seed,
    )


def make_metacognitive_efficiency_contract(seed: int = 0) -> MetacognitiveEfficiencyContract:
    return MetacognitiveEfficiencyContract(
        components=("oa2_calibration", "oa5_compute_value", "oa6_crisis_detection"),
        baselines=(
            ("oa2_calibration", "fixed-confidence-threshold"),
            ("oa5_compute_value", "always-continue"),
            ("oa6_crisis_detection", "raw-error-signal"),
        ),
        monitor_flop_budget=1_000_000,
        monitor_seconds_budget=10.0,
        controls=REQUIRED_METACOG_CONTROLS,
        seed=seed,
    )


def scaffold_manifest() -> dict[str, Any]:
    """One deterministic receipt naming every contract family this scaffold declares."""
    manifest = {
        "schema": "mop-integration-battery-scaffold-manifest/v1",
        "claim_scope": CLAIM_SCOPE,
        "contract_schemas": [
            THEORY_BATTERY_SCHEMA,
            FUNCTIONAL_SCORE_SCHEMA,
            SELF_MODEL_SCHEMA,
            REPORT_GROUNDING_SCHEMA,
            BROADCAST_CONTRACT_SCHEMA,
            METACOG_EFFICIENCY_SCHEMA,
        ],
        "operation_levels": list(OPERATION_LEVELS),
        "self_model_kinds": list(SELF_MODEL_KINDS),
        "broadcast_modes": list(BROADCAST_MODES),
        "oa_components": list(OA_COMPONENTS),
        "telemetry_channels": sorted(TELEMETRY_FIELDS),
        "interpretation_excluded_tokens": list(INTERPRETATION_EXCLUDED_TOKENS),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest
