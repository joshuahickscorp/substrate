"""Scaffold spine for the bio-morphogenic material digital-twin cluster (facets BM1 to BM4).

This module raises the SCAFFOLDING axis only. It supplies machine-checkable contracts, three tiny
seeded material priors, declared conventional-control families, damage and repair twin contracts,
and a bench handoff protocol whose physical work is quarantined behind a gate that local code
refuses to pass. It builds the harness, not the result.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability claim.
Nothing here establishes native-dynamics value, biological equivalence, or device behavior. The
material priors are toy numpy maps. The control families are declarations, not implementations. The
bench protocol is a form to be executed by an external specimen lab, never by this process.

House style: no em or en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..devel.north_star import assert_no_sentience_claims
from ..experiments.expansion_harness import CLAIM_SCOPE
from ..substrate.events import canonical_sha256

MATERIAL_TWIN_SCHEMA = "mop-material-twin/v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

Vector = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Section A. Common digital-twin protocol contract.
# excitation / time / readout / adaptation / drift / damage / repair / energy / lineage.
# ---------------------------------------------------------------------------

# The exact method surface any simulated material or bio-motif substrate must expose. Ordering is
# load-bearing for the interface digest, so the validator can detect surface drift.
REQUIRED_TWIN_METHODS: tuple[str, ...] = (
    "reset",  # deterministic reseed to a known state
    "excite",  # excitation: inject an input drive
    "evolve",  # time: advance the internal dynamics
    "readout",  # readout: emit the observable feature vector
    "adapt",  # adaptation: fit the linear readout to a target (no deep training)
    "apply_drift",  # drift: age the substrate parameters
    "apply_damage",  # damage: apply a selective lesion
    "repair",  # repair: restore under a declared repair policy
    "energy",  # energy: cumulative activity cost proxy
    "lineage",  # lineage: ordered, hash-linked provenance of every operation
)


@runtime_checkable
class MaterialTwin(Protocol):
    """Structural protocol for a material or bio-motif digital twin.

    Claim scope: deterministic programmatic mechanics only; no capability claim. Implementing this
    protocol asserts nothing about physical realizability or advantage over digital controls.
    """

    def reset(self, seed: int) -> None: ...
    def excite(self, drive: Vector) -> None: ...
    def evolve(self, steps: int) -> None: ...
    def readout(self) -> Vector: ...
    def adapt(self, targets: Vector) -> None: ...
    def apply_drift(self, epochs: int) -> None: ...
    def apply_damage(self, lesion: LesionSpec) -> None: ...
    def repair(self, policy: str) -> None: ...
    def energy(self) -> float: ...
    def lineage(self) -> Lineage: ...


@dataclass(frozen=True, slots=True)
class Lineage:
    """Ordered, hash-linked provenance of the operations applied to one twin instance."""

    schema: str
    ops: tuple[str, ...]
    head_digest: str

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported lineage schema {self.schema!r}")
        if _SHA256_RE.fullmatch(self.head_digest) is None:
            raise ValueError("lineage head digest must be a lowercase SHA-256 digest")
        expected = canonical_sha256(list(self.ops))
        if expected != self.head_digest:
            raise ValueError("lineage head digest does not match its op sequence")

    def payload(self) -> dict[str, Any]:
        return {"schema": self.schema, "ops": list(self.ops), "head_digest": self.head_digest}


@dataclass(frozen=True, slots=True)
class TwinInterfaceContract:
    """Declares the required twin method surface and validates candidate implementations."""

    schema: str = MATERIAL_TWIN_SCHEMA
    required_methods: tuple[str, ...] = REQUIRED_TWIN_METHODS
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported twin interface schema {self.schema!r}")
        if tuple(self.required_methods) != REQUIRED_TWIN_METHODS:
            raise ValueError("twin interface method surface or order drift")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("twin interface claim scope cannot be widened")

    def interface_digest(self) -> str:
        return canonical_sha256({"schema": self.schema, "methods": list(self.required_methods)})

    def validate(self, candidate: object) -> None:
        """Fail closed if the candidate omits or shadows any required twin method."""

        for name in self.required_methods:
            member = getattr(candidate, name, None)
            if member is None or not callable(member):
                raise ValueError(f"twin candidate is missing callable method {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "required_methods": list(self.required_methods),
            "interface_digest": self.interface_digest(),
            "claim_scope": self.claim_scope,
        }


def validate_twin_interface(candidate: object) -> None:
    """Module-level fail-closed check that a candidate satisfies the twin interface contract."""

    TwinInterfaceContract().validate(candidate)


# ---------------------------------------------------------------------------
# Section A helpers: lesion spec is shared by the protocol and the damage/repair contract.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LesionSpec:
    """A deterministic, selective lesion over the internal units of a twin."""

    unit_indices: tuple[int, ...]
    fraction: float
    seed: int
    selective: bool

    def __post_init__(self) -> None:
        if not self.unit_indices:
            raise ValueError("lesion must name at least one unit index")
        if any(i < 0 for i in self.unit_indices):
            raise ValueError("lesion unit indices must be nonnegative")
        if len(set(self.unit_indices)) != len(self.unit_indices):
            raise ValueError("lesion unit indices must be unique")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("lesion fraction must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("lesion seed must be nonnegative")

    def payload(self) -> dict[str, Any]:
        return {
            "unit_indices": list(self.unit_indices),
            "fraction": self.fraction,
            "seed": self.seed,
            "selective": self.selective,
        }


# ---------------------------------------------------------------------------
# Section B. Three tiny, seeded material priors implementing the protocol.
# Pure numpy math; no network, no model weights, no heavy compute.
# ---------------------------------------------------------------------------

REPAIR_POLICIES: tuple[str, ...] = ("fixed-final", "restart", "spare", "random", "full-retraining")


class _ToyMaterialPrior:
    """Shared bookkeeping for the toy priors: energy, lineage, damage, and repair state.

    Claim scope: deterministic programmatic mechanics only; no capability claim. Subclasses supply
    only the excitation, evolution, and readout math. All priors are tiny and CPU-only.
    """

    kind: str = "abstract"

    def __init__(self, units: int = 12, seed: int = 0) -> None:
        if units < 2:
            raise ValueError("toy prior needs at least two units")
        self.units = units
        self._seed = seed
        self._energy = 0.0
        self._ops: list[str] = []
        self._readout_weights: Vector | None = None
        self._pristine_state: Vector | None = None
        self._damage_mask: Vector = np.ones(units, dtype=np.float64)
        self._drift_epochs = 0
        self.reset(seed)

    # -- protocol surface -------------------------------------------------
    def reset(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.state: Vector = np.zeros(self.units, dtype=np.float64)
        self._build_parameters()
        self._pristine_state = self.state.copy()
        self._damage_mask = np.ones(self.units, dtype=np.float64)
        self._drift_epochs = 0
        self._energy = 0.0
        self._ops = [f"reset:seed={seed}"]

    def excite(self, drive: Vector) -> None:
        drive = np.asarray(drive, dtype=np.float64)
        if drive.shape != (self.units,):
            raise ValueError(f"drive must have shape ({self.units},)")
        self._pending_drive = drive.copy()
        self._ops.append("excite")

    def evolve(self, steps: int) -> None:
        if steps < 1:
            raise ValueError("evolve needs at least one step")
        drive = getattr(self, "_pending_drive", np.zeros(self.units, dtype=np.float64))
        for _ in range(steps):
            self.state = self._step(self.state, drive) * self._damage_mask
            self._energy += float(np.sum(self.state * self.state))
        self._ops.append(f"evolve:{steps}")

    def readout(self) -> Vector:
        feats = self.state.copy()
        if self._readout_weights is not None:
            feats = self._readout_weights @ feats
        self._ops.append("readout")
        return feats

    def adapt(self, targets: Vector) -> None:
        # Linear ridge readout fit only. No deep training; the substrate parameters are untouched.
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 1:
            raise ValueError("adapt targets must be a 1D vector")
        state = self.state
        denom = float(state @ state) + 1e-3
        self._readout_weights = (np.outer(targets, state) / denom).astype(np.float64)
        self._ops.append("adapt")

    def apply_drift(self, epochs: int) -> None:
        if epochs < 1:
            raise ValueError("apply_drift needs at least one epoch")
        self._drift_epochs += epochs
        self._apply_drift(epochs)
        self._ops.append(f"drift:{epochs}")

    def apply_damage(self, lesion: LesionSpec) -> None:
        realized_fraction = len(lesion.unit_indices) / self.units
        if not np.isclose(realized_fraction, lesion.fraction, rtol=0.0, atol=1e-12):
            raise ValueError("lesion fraction must equal the fraction of uniquely selected twin units")
        for i in lesion.unit_indices:
            if i >= self.units:
                raise ValueError("lesion unit index out of range for this prior")
            self._damage_mask[i] = 0.0
            self.state[i] = 0.0
        self._ops.append(f"damage:{','.join(str(i) for i in lesion.unit_indices)}")

    def repair(self, policy: str) -> None:
        if policy not in REPAIR_POLICIES:
            raise ValueError(f"unsupported repair policy {policy!r}")
        if policy == "fixed-final":
            pass  # keep the damaged mask; the fixed-final null does nothing
        elif policy == "restart":
            self.reset(self._seed)
        elif policy in {"spare", "random", "full-retraining"}:
            self._damage_mask = np.ones(self.units, dtype=np.float64)
            if policy == "random":
                self.state = self._rng.standard_normal(self.units) * 0.1
            elif policy == "spare" and self._pristine_state is not None:
                self.state = self._pristine_state.copy()
        self._ops.append(f"repair:{policy}")

    def energy(self) -> float:
        return self._energy

    def lineage(self) -> Lineage:
        ops = tuple(self._ops)
        return Lineage(
            schema=MATERIAL_TWIN_SCHEMA,
            ops=ops,
            head_digest=canonical_sha256(list(ops)),
        )

    # -- subclass hooks ---------------------------------------------------
    def _build_parameters(self) -> None:
        raise NotImplementedError

    def _step(self, state: Vector, drive: Vector) -> Vector:
        raise NotImplementedError

    def _apply_drift(self, epochs: int) -> None:
        raise NotImplementedError


class LeakyEchoStateReservoir(_ToyMaterialPrior):
    """A tiny leaky echo-state reservoir. Physical-reservoir prior (facet BM3 and BM4)."""

    kind = "leaky-echo-state-reservoir"

    def __init__(self, units: int = 12, seed: int = 0, leak: float = 0.3, radius: float = 0.9) -> None:
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")
        if not 0.0 < radius < 1.0:
            raise ValueError("spectral radius must be in (0, 1)")
        self.leak = leak
        self.radius = radius
        super().__init__(units=units, seed=seed)

    def _build_parameters(self) -> None:
        w = self._rng.standard_normal((self.units, self.units))
        spectral = float(np.max(np.abs(np.linalg.eigvals(w))))
        self._w = w * (self.radius / spectral) if spectral > 0 else w
        self._win = self._rng.standard_normal((self.units, self.units)) * 0.1

    def _step(self, state: Vector, drive: Vector) -> Vector:
        pre = self._w @ state + self._win @ drive
        return (1.0 - self.leak) * state + self.leak * np.tanh(pre)

    def _apply_drift(self, epochs: int) -> None:
        self._w = self._w * (0.999**epochs)


class DecayingConductanceMap(_ToyMaterialPrior):
    """A decaying-conductance memristive-like map. Memristive prior (facet BM3 and BM4)."""

    kind = "decaying-conductance-map"

    def __init__(self, units: int = 12, seed: int = 0, decay: float = 0.8, gain: float = 0.5) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        self.decay = decay
        self.gain = gain
        super().__init__(units=units, seed=seed)

    def _build_parameters(self) -> None:
        self._pulse = 0.5 + 0.5 * self._rng.random(self.units)

    def _step(self, state: Vector, drive: Vector) -> Vector:
        conductance = self.decay * state + self.gain * self._pulse * np.abs(drive)
        return np.clip(conductance, 0.0, 1.0)

    def _apply_drift(self, epochs: int) -> None:
        self._pulse = self._pulse * (0.99**epochs)


class OscillatorBank(_ToyMaterialPrior):
    """A bank of weakly coupled phase oscillators. Oscillatory prior (facet BM3 and BM4)."""

    kind = "oscillator-bank"

    def __init__(self, units: int = 12, seed: int = 0, dt: float = 0.1, coupling: float = 0.05) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if coupling < 0.0:
            raise ValueError("coupling must be nonnegative")
        self.dt = dt
        self.coupling = coupling
        super().__init__(units=units, seed=seed)

    def _build_parameters(self) -> None:
        self._omega = 0.5 + self._rng.random(self.units)
        self.state = self._rng.random(self.units) * 2.0 * np.pi

    def _step(self, state: Vector, drive: Vector) -> Vector:
        mean_field = float(np.mean(np.sin(state)))
        phase = state + self.dt * (self._omega + self.coupling * mean_field + 0.1 * drive)
        return np.mod(phase, 2.0 * np.pi)

    def _apply_drift(self, epochs: int) -> None:
        self._omega = self._omega * (1.0 - 0.001 * epochs)


TOY_PRIORS: tuple[type[_ToyMaterialPrior], ...] = (
    LeakyEchoStateReservoir,
    DecayingConductanceMap,
    OscillatorBank,
)


# ---------------------------------------------------------------------------
# Section C. Matched conventional-control declarations (declared, not implemented).
# ---------------------------------------------------------------------------

CONTROL_FAMILIES: frozenset[str] = frozenset(
    {
        "tuned-rnn",
        "state-space-model",
        "random-reservoir",
        "fourier-features",
        "associative-memory",
    }
)


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    """The matched full-system budget a control family must be tuned to before comparison."""

    params: int
    flops: int
    memory_bytes: int
    update_steps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("memory_bytes", self.memory_bytes),
            ("update_steps", self.update_steps),
        ):
            if value <= 0:
                raise ValueError(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "update_steps": self.update_steps,
        }


@dataclass(frozen=True, slots=True)
class ControlFamilyDeclaration:
    """A declared conventional-control family. Not an implementation; a comparison commitment."""

    id: str
    family: str
    rationale: str
    matched: MatchedBudget
    tuned: bool
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if _ID_RE.fullmatch(self.id) is None:
            raise ValueError("control id must use stable lowercase characters")
        if self.family not in CONTROL_FAMILIES:
            raise ValueError(f"unsupported control family {self.family!r}")
        if not self.rationale.strip():
            raise ValueError("control rationale must not be empty")
        assert_no_sentience_claims(self.rationale, where="material twin control rationale")
        if not self.tuned:
            raise ValueError("control family must be declared tuned to be a fair comparator")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("control claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "rationale": self.rationale,
            "matched": self.matched.payload(),
            "tuned": self.tuned,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class ControlDeclarationSet:
    """A set of control-family declarations that must cover every declared conventional family."""

    schema: str
    declarations: tuple[ControlFamilyDeclaration, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported control set schema {self.schema!r}")
        ids = [d.id for d in self.declarations]
        if len(set(ids)) != len(ids):
            raise ValueError("control declaration ids must be unique")
        covered = {d.family for d in self.declarations}
        if covered != CONTROL_FAMILIES:
            missing = CONTROL_FAMILIES - covered
            raise ValueError(f"control set does not cover every conventional family; missing {missing}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("control set claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "declarations": [d.payload() for d in self.declarations],
            "claim_scope": self.claim_scope,
        }


# ---------------------------------------------------------------------------
# Section C helper. Native-dynamics value contract (facet BM1 and BM2 closure test).
# ---------------------------------------------------------------------------

RETAIN_ONLY_WINS: frozenset[str] = frozenset({"cross-task-future-learnability", "capability-density"})


@dataclass(frozen=True, slots=True)
class NativeDynamicsValueContract:
    """Declares the retain-only rule: keep a material prior only for a replicated matched-cost win."""

    schema: str
    controls: tuple[str, ...]
    matched_cost_required: bool
    retain_only: tuple[str, ...]
    replication_min: int
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported native-dynamics schema {self.schema!r}")
        if not set(self.controls) <= CONTROL_FAMILIES:
            raise ValueError("native-dynamics controls must be drawn from the declared families")
        if not self.controls:
            raise ValueError("native-dynamics value needs at least one declared control")
        if not self.matched_cost_required:
            raise ValueError("native-dynamics value must require matched full-system cost")
        if not set(self.retain_only) <= RETAIN_ONLY_WINS or not self.retain_only:
            raise ValueError("retain-only wins must be drawn from the allowed win vocabulary")
        if self.replication_min < 2:
            raise ValueError("native-dynamics value requires at least two independent replications")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("native-dynamics claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controls": list(self.controls),
            "matched_cost_required": self.matched_cost_required,
            "retain_only": list(self.retain_only),
            "replication_min": self.replication_min,
            "claim_scope": self.claim_scope,
        }


# ---------------------------------------------------------------------------
# Section D. Damage / repair twin contracts (facet BM2 and BM4).
# ---------------------------------------------------------------------------

REPAIR_CONTROLS: tuple[str, ...] = ("fixed-final", "restart", "spare", "random", "full-retraining")
DAMAGE_REPAIR_METRICS: tuple[str, ...] = (
    "repair_time",
    "restored_function",
    "future_plasticity",
    "repair_energy",
    "topology_change",
)


@dataclass(frozen=True, slots=True)
class DamageRepairContract:
    """Damage and repair twin contract with fail-closed control and metric coverage."""

    schema: str
    controls: tuple[str, ...]
    metrics: tuple[str, ...]
    selective_lesion_required: bool
    restoration_under_novelty_required: bool
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported damage/repair schema {self.schema!r}")
        if tuple(self.controls) != REPAIR_CONTROLS:
            raise ValueError("damage/repair controls or order drift")
        if tuple(self.metrics) != DAMAGE_REPAIR_METRICS:
            raise ValueError("damage/repair metric surface or order drift")
        if not self.selective_lesion_required:
            raise ValueError("damage/repair must require a selective lesion, not a diffuse one")
        if not self.restoration_under_novelty_required:
            raise ValueError("damage/repair must require restoration under novel inputs")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("damage/repair claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controls": list(self.controls),
            "metrics": list(self.metrics),
            "selective_lesion_required": self.selective_lesion_required,
            "restoration_under_novelty_required": self.restoration_under_novelty_required,
            "claim_scope": self.claim_scope,
        }


def selective_lesion_delta(twin: _ToyMaterialPrior, lesion: LesionSpec) -> dict[str, float]:
    """Deterministic mechanics probe: readout-norm change under a selective lesion.

    Claim scope: deterministic programmatic mechanics only; no capability claim. This measures a
    programmatic delta on a toy prior. It is not evidence of biological repair or advantage.
    """

    if not lesion.selective:
        raise ValueError("selective_lesion_delta requires a selective lesion")
    before = float(np.linalg.norm(twin.readout()))
    twin.apply_damage(lesion)
    twin.evolve(1)
    after = float(np.linalg.norm(twin.readout()))
    return {"readout_norm_before": before, "readout_norm_after": after, "delta": after - before}


# ---------------------------------------------------------------------------
# Section D helper. Drift and aging adaptation contract (facet BM3, proposed row f63).
# ---------------------------------------------------------------------------

DRIFT_CONTROLS: tuple[str, ...] = ("no-adapt", "oracle-reset", "full-retraining")
DRIFT_METRICS: tuple[str, ...] = (
    "drift_rate",
    "adaptation_lag",
    "retained_function",
    "adaptation_energy",
)


@dataclass(frozen=True, slots=True)
class DriftAdaptationContract:
    """Declares how a twin's aging drift is tracked and what adaptation must beat."""

    schema: str
    drift_model: str
    adaptation_policy: str
    controls: tuple[str, ...]
    metrics: tuple[str, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported drift schema {self.schema!r}")
        if not self.drift_model.strip() or not self.adaptation_policy.strip():
            raise ValueError("drift model and adaptation policy must be declared")
        if tuple(self.controls) != DRIFT_CONTROLS:
            raise ValueError("drift controls or order drift")
        if tuple(self.metrics) != DRIFT_METRICS:
            raise ValueError("drift metric surface or order drift")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("drift claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "drift_model": self.drift_model,
            "adaptation_policy": self.adaptation_policy,
            "controls": list(self.controls),
            "metrics": list(self.metrics),
            "claim_scope": self.claim_scope,
        }


# ---------------------------------------------------------------------------
# Section D helper. Cross-substrate form portability contract (facet BM3, proposed row f66).
# This one is simulation-portable and can run locally; no physical gate.
# ---------------------------------------------------------------------------

PORTABILITY_CONTROLS: tuple[str, ...] = ("identity", "random-remap", "full-retrain")


@dataclass(frozen=True, slots=True)
class CrossSubstratePortabilityContract:
    """Declares how a fitted readout is carried from one simulated form to another."""

    schema: str
    source_kind: str
    target_kind: str
    portability_metric: str
    controls: tuple[str, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported portability schema {self.schema!r}")
        if self.source_kind == self.target_kind:
            raise ValueError("cross-substrate portability needs two distinct forms")
        if not self.portability_metric.strip():
            raise ValueError("portability metric must be declared")
        if tuple(self.controls) != PORTABILITY_CONTROLS:
            raise ValueError("portability controls or order drift")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("portability claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_kind": self.source_kind,
            "target_kind": self.target_kind,
            "portability_metric": self.portability_metric,
            "controls": list(self.controls),
            "claim_scope": self.claim_scope,
        }


# ---------------------------------------------------------------------------
# Section E. Bench handoff protocol contract. Physical work is quarantined behind a gate that
# local code refuses to pass. Specimen-to-specimen transfer (proposed row f65) rides the same gate.
# ---------------------------------------------------------------------------


class ExternalSpecimenRefusal(RuntimeError):
    """Raised when local code attempts to perform work reserved for an external specimen lab."""


@dataclass(frozen=True, slots=True)
class ExternalSpecimenGate:
    """A fail-closed gate. Any local attempt to authorize physical specimen work raises.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The gate exists so
    that simulation code can never quietly stand in for a fabricated device or a living specimen.
    """

    physical_work_required: bool = True
    local_execution_permitted: bool = False

    def __post_init__(self) -> None:
        if not self.physical_work_required:
            raise ValueError("the external specimen gate only guards physical work")
        if self.local_execution_permitted:
            raise ValueError("local execution of physical specimen work is never permitted")

    def authorize_local(self) -> None:
        raise ExternalSpecimenRefusal(
            "physical specimen work is quarantined; local code cannot pass this gate. "
            "Route to an external specimen lab with independent randomization and multiple specimens."
        )

    def payload(self) -> dict[str, Any]:
        return {
            "physical_work_required": self.physical_work_required,
            "local_execution_permitted": self.local_execution_permitted,
        }


_REQUIRED_HANDOFF_FIELDS: tuple[str, ...] = (
    "specimen",
    "randomization",
    "environment",
    "safety",
    "stop",
    "portability",
)


@dataclass(frozen=True, slots=True)
class BenchHandoffContract:
    """A bench-ready handoff form whose physical execution is refused locally.

    Every required field must be a non-empty declaration. Calling ``pass_locally`` always raises;
    the only honest local operation is to emit the form for an external specimen lab.
    """

    schema: str
    fields: Mapping[str, str]
    gate: ExternalSpecimenGate = field(default_factory=ExternalSpecimenGate)
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported bench handoff schema {self.schema!r}")
        missing = [k for k in _REQUIRED_HANDOFF_FIELDS if not str(self.fields.get(k, "")).strip()]
        if missing:
            raise ValueError(f"bench handoff form is missing declarations for {missing}")
        extra = set(self.fields) - set(_REQUIRED_HANDOFF_FIELDS)
        if extra:
            raise ValueError(f"bench handoff form has unknown fields {sorted(extra)}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("bench handoff claim scope cannot be widened")

    def pass_locally(self) -> None:
        """Fail closed: local code may never execute the physical specimen protocol."""

        self.gate.authorize_local()

    def emit_form(self) -> dict[str, Any]:
        """The only sanctioned local operation: return the declared handoff form."""

        return {
            "schema": self.schema,
            "fields": {k: self.fields[k] for k in _REQUIRED_HANDOFF_FIELDS},
            "gate": self.gate.payload(),
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class SpecimenTransferContract:
    """Specimen-to-specimen transfer contract; physical transfer rides the external gate."""

    schema: str
    source_specimen: str
    target_specimen: str
    randomization: str
    transfer_metric: str
    controls: tuple[str, ...]
    gate: ExternalSpecimenGate = field(default_factory=ExternalSpecimenGate)
    claim_scope: str = CLAIM_SCOPE

    _CONTROLS: tuple[str, ...] = field(
        default=("within-specimen", "random-specimen", "full-refit"),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.schema != MATERIAL_TWIN_SCHEMA:
            raise ValueError(f"unsupported specimen transfer schema {self.schema!r}")
        if self.source_specimen == self.target_specimen:
            raise ValueError("specimen transfer needs two distinct specimens")
        for name, value in (
            ("randomization", self.randomization),
            ("transfer_metric", self.transfer_metric),
        ):
            if not value.strip():
                raise ValueError(f"specimen transfer {name} must be declared")
        if tuple(self.controls) != self._CONTROLS:
            raise ValueError("specimen transfer controls or order drift")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("specimen transfer claim scope cannot be widened")

    def pass_locally(self) -> None:
        """Fail closed: physical specimen-to-specimen transfer cannot run in this process."""

        self.gate.authorize_local()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_specimen": self.source_specimen,
            "target_specimen": self.target_specimen,
            "randomization": self.randomization,
            "transfer_metric": self.transfer_metric,
            "controls": list(self.controls),
            "gate": self.gate.payload(),
            "claim_scope": self.claim_scope,
        }


# ---------------------------------------------------------------------------
# Convenience builders (deterministic, used by tests and later wiring).
# ---------------------------------------------------------------------------


def build_default_control_set() -> ControlDeclarationSet:
    """Return a fully covering, non-vacuous set of declared conventional-control families."""

    unit_budget = MatchedBudget(params=144, flops=4096, memory_bytes=8192, update_steps=200)
    declarations = tuple(
        ControlFamilyDeclaration(
            id=f"ctrl.{family.replace('-', '_')}",
            family=family,
            rationale=f"tuned {family} matched to the toy prior full-system budget",
            matched=unit_budget,
            tuned=True,
        )
        for family in sorted(CONTROL_FAMILIES)
    )
    return ControlDeclarationSet(schema=MATERIAL_TWIN_SCHEMA, declarations=declarations)


def default_damage_repair_contract() -> DamageRepairContract:
    return DamageRepairContract(
        schema=MATERIAL_TWIN_SCHEMA,
        controls=REPAIR_CONTROLS,
        metrics=DAMAGE_REPAIR_METRICS,
        selective_lesion_required=True,
        restoration_under_novelty_required=True,
    )


def default_bench_handoff() -> BenchHandoffContract:
    return BenchHandoffContract(
        schema=MATERIAL_TWIN_SCHEMA,
        fields={
            "specimen": "declared external specimen family; count and provenance set by the lab",
            "randomization": "independent per-specimen randomization sequence, seed held by the lab",
            "environment": "declared temperature, humidity, and power envelope with logging",
            "safety": "declared electrical and material safety limits and interlocks",
            "stop": "declared preregistered stop rule on specimen count and effect estimate",
            "portability": "declared readout portability check across specimens and forms",
        },
    )


def facet_coverage_map() -> dict[str, Sequence[str]]:
    """Static record of which BM facet local-to-10 bullets this scaffold supports (readiness only)."""

    return {
        "BM1": (
            "map each motif to function/falsifier/control/cost via NativeDynamicsValueContract",
            "retain-only rule for cross-task future-learnability or density wins",
        ),
        "BM2": (
            "damage-repair twin contract with fixed-final/restart/spare/random/full-retraining controls",
            "selective lesion and restoration-under-novelty required by contract",
        ),
        "BM3": (
            "shared excitation/evolution/readout/update protocol via TwinInterfaceContract",
            "reservoir/oscillator/memristive priors plus declared RNN/SSM/random/Fourier/assoc controls",
            "drift and cross-substrate portability contracts",
        ),
        "BM4": (
            "common excitation/time/readout/adaptation/drift/damage/repair/energy/lineage twin",
            "three material priors and declared conventional controls",
            "bench-ready handoff form gated behind an external-specimen refusal",
        ),
    }
