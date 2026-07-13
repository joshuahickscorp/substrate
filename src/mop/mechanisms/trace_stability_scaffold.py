"""Scaffold spine for epoch G1-C0: cross-seed, cross-session cognitive-trace stability.

This module raises the SCAFFOLDING axis only. It supplies machine-checkable contracts and
deterministic mechanics that ENCODE the bar a claimed cognitive trace must clear BEFORE any
mechanism is licensed to build on it. It contains no measurement of a real system, loads no
model weights, and touches no network or natural data. A green contract here means the stability
declaration is complete and self-consistent, never that any trace is real.

Named prior null (G1-C0): earlier few-seed studies over-claimed. A trace is not real until it
survives seed variation across at least min_seeds seeds with every declared control dead. This
module fails closed on fewer than min_seeds seeds, on any control that reproduces the trace, and
on an undeclared stability metric. The verdict "stable" is unreachable unless the null is beaten.

House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

TRACE_STABILITY_SCHEMA = "mop-trace-stability/v1"

# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

SCIENTIFIC_CAPABILITY_CLAIM = False
scientific_capability_claim = False

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The minimum number of distinct seeds a trace must survive. Set to reject the few-seed regime
# the G1-C0 null indicts. Any contract that declares fewer fails closed.
MIN_SEEDS = 8

# The only stability metrics this epoch admits. An undeclared or off-vocabulary metric is refused.
STABILITY_METRICS: tuple[str, ...] = ("rank-correlation", "sign-agreement", "effect-direction")

# The mandatory control set. Membership and order are load-bearing so the completeness check can
# detect drift. Every control must fail to reproduce the trace for a "stable" verdict.
REQUIRED_CONTROLS: tuple[str, ...] = (
    "single-seed",  # collapse to one seed: the exact few-seed regime the null indicts
    "shuffled-seed",  # break the seed-to-measurement binding
    "permuted-trace",  # permute the trace values before agreement is scored
    "label-shuffled",  # destroy the identity-to-value alignment
)

VERDICT_STABLE = "stable"
VERDICT_NULL = "null"


class TraceStabilityRefusal(ValueError):
    """Raised whenever a stability declaration is missing, malformed, or outside its scope."""


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise TraceStabilityRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise TraceStabilityRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, label: str) -> None:
    if not value.strip():
        raise TraceStabilityRefusal(f"{label} must not be empty")


def assert_controls_complete(controls: Sequence[str]) -> None:
    """Fail closed if the declared control set drifts in membership or order."""

    if tuple(controls) != REQUIRED_CONTROLS:
        raise TraceStabilityRefusal("stability control set membership or order drift")


# ---------------------------------------------------------------------------
# Section A. Identity-bound per-seed measurement record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """One per-seed, identity-bound measurement of a candidate cognitive trace.

    A record declares that a trace with this content digest yielded this effect under this seed and
    session. Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    trace_id: str
    seed: int
    session_id: str
    effect: float
    trace_sha256: str
    ranking: tuple[int, ...] = ()
    claim_scope: str = CLAIM_SCOPE
    schema: str = TRACE_STABILITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_STABILITY_SCHEMA:
            raise TraceStabilityRefusal(f"unsupported trace record schema {self.schema!r}")
        _require_id(self.trace_id, "TraceRecord.trace_id")
        _require_nonempty(self.session_id, "TraceRecord.session_id")
        _require_sha256(self.trace_sha256, "TraceRecord.trace_sha256")
        if self.seed < 0:
            raise TraceStabilityRefusal("TraceRecord.seed must be nonnegative")
        if self.effect != self.effect:  # reject NaN without importing math
            raise TraceStabilityRefusal("TraceRecord.effect must be a finite number")
        if self.ranking and sorted(self.ranking) != list(range(len(self.ranking))):
            raise TraceStabilityRefusal("TraceRecord.ranking must be a permutation of range(n)")
        if self.claim_scope != CLAIM_SCOPE:
            raise TraceStabilityRefusal("trace record claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trace_id": self.trace_id,
            "seed": self.seed,
            "session_id": self.session_id,
            "effect": self.effect,
            "trace_sha256": self.trace_sha256,
            "ranking": list(self.ranking),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _spearman(rank_a: tuple[int, ...], rank_b: tuple[int, ...]) -> float:
    """Exact Spearman rank correlation between two rankings of equal length."""

    n = len(rank_a)
    if n < 2 or len(rank_b) != n:
        raise TraceStabilityRefusal("spearman correlation needs two rankings of equal length >= 2")
    d_squared = sum((a - b) * (a - b) for a, b in zip(rank_a, rank_b, strict=True))
    return 1.0 - (6.0 * d_squared) / (n * (n * n - 1))


def cross_seed_agreement(records: Sequence[TraceRecord], metric: str) -> float:
    """Deterministic cross-seed agreement in [0, 1] under the declared metric.

    Fails closed on an undeclared metric, on fewer than two records, or on records that do not
    all bind to one trace identity. Claim scope: deterministic programmatic mechanics only.
    """

    if metric not in STABILITY_METRICS:
        raise TraceStabilityRefusal(f"undeclared stability metric {metric!r}")
    if len(records) < 2:
        raise TraceStabilityRefusal("cross-seed agreement needs at least two records")
    if len({row.trace_id for row in records}) != 1:
        raise TraceStabilityRefusal("agreement is identity-bound; records name more than one trace")

    if metric in ("sign-agreement", "effect-direction"):
        signs = [_sign(row.effect) for row in records]
        positives = signs.count(1)
        negatives = signs.count(-1)
        zeros = signs.count(0)
        if metric == "effect-direction":
            # Zeros carry no direction and count as disagreement against the dominant direction.
            dominant = max(positives, negatives)
        else:
            # Sign agreement admits a zero majority as its own consistent class.
            dominant = max(positives, negatives, zeros)
        return dominant / len(records)

    rankings = [row.ranking for row in records]
    if any(len(rank) == 0 for rank in rankings):
        raise TraceStabilityRefusal("rank-correlation requires every record to declare a ranking")
    if len({len(rank) for rank in rankings}) != 1:
        raise TraceStabilityRefusal("rank-correlation requires rankings of equal length")
    rhos: list[float] = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            rhos.append(_spearman(rankings[i], rankings[j]))
    mean_rho = sum(rhos) / len(rhos)
    # Map correlation in [-1, 1] onto an agreement fraction in [0, 1].
    return (mean_rho + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Section B. Matched measurement budget: the control comparison must be non-vacuous.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedMeasurementBudget:
    """The matched measurement budget every control arm must be run under before comparison."""

    seeds: int
    sessions: int
    samples_per_seed: int
    compute_units: int

    def __post_init__(self) -> None:
        for name, value in (
            ("seeds", self.seeds),
            ("sessions", self.sessions),
            ("samples_per_seed", self.samples_per_seed),
            ("compute_units", self.compute_units),
        ):
            if value <= 0:
                raise TraceStabilityRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "seeds": self.seeds,
            "sessions": self.sessions,
            "samples_per_seed": self.samples_per_seed,
            "compute_units": self.compute_units,
        }


# ---------------------------------------------------------------------------
# Section C. Stability contract: the preregistered bar.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceStabilityContract:
    """Preregisters the bar a candidate trace must clear before a mechanism is licensed.

    A valid contract fixes min_seeds, the metric, the agreement threshold, the mandatory control
    set, and the matched budget; it fixes the target, not any result. Claim scope: deterministic
    programmatic mechanics only; no capability claim.
    """

    schema: str
    trace_id: str
    stability_metric: str
    min_seeds: int
    agreement_threshold: float
    controls: tuple[str, ...]
    matched_cost_required: bool
    matched: MatchedMeasurementBudget
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != TRACE_STABILITY_SCHEMA:
            raise TraceStabilityRefusal(f"unsupported stability contract schema {self.schema!r}")
        _require_id(self.trace_id, "TraceStabilityContract.trace_id")
        if self.stability_metric not in STABILITY_METRICS:
            raise TraceStabilityRefusal("stability metric is undeclared or off vocabulary")
        if self.min_seeds < MIN_SEEDS:
            raise TraceStabilityRefusal(f"stability requires at least {MIN_SEEDS} seeds")
        if not 0.0 < self.agreement_threshold <= 1.0:
            raise TraceStabilityRefusal("agreement threshold must be in (0, 1]")
        assert_controls_complete(self.controls)
        if not self.matched_cost_required:
            raise TraceStabilityRefusal("stability comparison must require matched measurement cost")
        if self.matched.seeds < self.min_seeds:
            raise TraceStabilityRefusal("matched budget must fund at least min_seeds seeds")
        if self.claim_scope != CLAIM_SCOPE:
            raise TraceStabilityRefusal("stability contract claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trace_id": self.trace_id,
            "stability_metric": self.stability_metric,
            "min_seeds": self.min_seeds,
            "agreement_threshold": self.agreement_threshold,
            "controls": list(self.controls),
            "matched_cost_required": self.matched_cost_required,
            "matched": self.matched.payload(),
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section D. Control outcomes and the fail-closed verdict.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    """One control arm's declared reproduction result under the matched budget.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    control: str
    agreement: float
    reproduced: bool

    def __post_init__(self) -> None:
        if self.control not in REQUIRED_CONTROLS:
            raise TraceStabilityRefusal(f"unsupported control arm {self.control!r}")
        if not 0.0 <= self.agreement <= 1.0:
            raise TraceStabilityRefusal("control agreement must lie in [0, 1]")

    def payload(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "agreement": self.agreement,
            "reproduced": self.reproduced,
        }


@dataclass(frozen=True, slots=True)
class StabilityVerdict:
    """Binds a contract, its per-seed records, and every control outcome into one fail-closed verdict.

    The verdict is "stable" ONLY IF cross-seed agreement meets the declared threshold AND every
    declared control fails to reproduce the trace; otherwise it is "null". Construction fails closed
    on fewer than min_seeds distinct seeds, on foreign record identity, on control coverage drift, or
    on a reproduced flag that disagrees with the threshold. Claim scope: deterministic programmatic
    mechanics only; no capability claim.
    """

    schema: str
    contract: TraceStabilityContract
    records: tuple[TraceRecord, ...]
    control_outcomes: tuple[ControlOutcome, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != TRACE_STABILITY_SCHEMA:
            raise TraceStabilityRefusal(f"unsupported stability verdict schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise TraceStabilityRefusal("stability verdict claim scope cannot be widened")
        if any(row.trace_id != self.contract.trace_id for row in self.records):
            raise TraceStabilityRefusal("every record must bind to the contract trace identity")
        distinct_seeds = {row.seed for row in self.records}
        if len(distinct_seeds) < self.contract.min_seeds:
            raise TraceStabilityRefusal(
                f"stability needs at least {self.contract.min_seeds} distinct seeds; "
                f"got {len(distinct_seeds)}"
            )
        if tuple(row.control for row in self.control_outcomes) != REQUIRED_CONTROLS:
            raise TraceStabilityRefusal("control outcome coverage is incomplete or out of order")
        threshold = self.contract.agreement_threshold
        for outcome in self.control_outcomes:
            declared = outcome.reproduced
            implied = outcome.agreement >= threshold
            if declared != implied:
                raise TraceStabilityRefusal(
                    f"control {outcome.control} reproduced flag disagrees with its agreement vs threshold"
                )

    def primary_agreement(self) -> float:
        """Cross-seed agreement of the primary records under the declared metric."""

        return cross_seed_agreement(self.records, self.contract.stability_metric)

    def controls_dead(self) -> bool:
        """True only if no declared control reproduces the trace."""

        return not any(outcome.reproduced for outcome in self.control_outcomes)

    def decide(self) -> str:
        """Return the fail-closed verdict label. "stable" is unreachable unless the null is beaten."""

        agreement_met = self.primary_agreement() >= self.contract.agreement_threshold
        if agreement_met and self.controls_dead():
            return VERDICT_STABLE
        return VERDICT_NULL

    def assert_stable(self) -> None:
        """Fail closed unless the verdict is genuinely stable; used before any license is issued."""

        if self.decide() != VERDICT_STABLE:
            raise TraceStabilityRefusal(
                "trace is not stable: agreement below threshold or a control reproduced it"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract": self.contract.payload(),
            "records": [row.payload() for row in self.records],
            "control_outcomes": [row.payload() for row in self.control_outcomes],
            "verdict": self.decide(),
            "primary_agreement": self.primary_agreement(),
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section E. Activation gate: mechanism licensing is not earned until stability is proven.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LicenseReceipt:
    """A confirmation receipt that a stable verdict was reached and independently replicated."""

    verdict_sha256: str
    verdict_label: str
    independent_confirmations: int
    replication_min: int
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_sha256(self.verdict_sha256, "LicenseReceipt.verdict_sha256")
        if self.verdict_label not in (VERDICT_STABLE, VERDICT_NULL):
            raise TraceStabilityRefusal(f"unsupported verdict label {self.verdict_label!r}")
        if self.replication_min < 2:
            raise TraceStabilityRefusal("a license needs at least two independent replications")
        if self.independent_confirmations < 0:
            raise TraceStabilityRefusal("independent confirmations cannot be negative")
        if self.claim_scope != CLAIM_SCOPE:
            raise TraceStabilityRefusal("license receipt claim scope cannot be widened")

    def assert_grants_license(self) -> None:
        if self.verdict_label != VERDICT_STABLE:
            raise TraceStabilityRefusal("a null verdict cannot license a mechanism")
        if self.independent_confirmations < self.replication_min:
            raise TraceStabilityRefusal("stability was not independently replicated enough to license")

    def payload(self) -> dict[str, Any]:
        return {
            "verdict_sha256": self.verdict_sha256,
            "verdict_label": self.verdict_label,
            "independent_confirmations": self.independent_confirmations,
            "replication_min": self.replication_min,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class MechanismLicenseGate:
    """A fail-closed gate. Off by default; licensing a mechanism on a trace requires a valid receipt.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The gate exists so
    that no mechanism can quietly be built on a trace that has not survived seed variation with its
    controls dead. Calling authorize with no receipt always raises.
    """

    stability_required: bool = True
    license_granted: bool = False
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if not self.stability_required:
            raise TraceStabilityRefusal("the license gate only guards mechanisms that require stability")
        if self.license_granted:
            raise TraceStabilityRefusal("mechanism licensing is never pre-granted; stability is unproven")
        if self.claim_scope != CLAIM_SCOPE:
            raise TraceStabilityRefusal("license gate claim scope cannot be widened")

    def authorize(self, receipt: LicenseReceipt | None = None) -> None:
        """Fail closed unless a valid stable, replicated receipt is supplied."""

        if receipt is None:
            raise TraceStabilityRefusal(
                "mechanism licensing refused: no stability receipt supplied; activation is not earned"
            )
        receipt.assert_grants_license()

    def payload(self) -> dict[str, Any]:
        return {
            "stability_required": self.stability_required,
            "license_granted": self.license_granted,
            "claim_scope": self.claim_scope,
        }


# ---------------------------------------------------------------------------
# Section F. Deterministic synthesis and convenience builders.
# ---------------------------------------------------------------------------


def _deterministic_effect(
    trace_id: str, seed: int, session_id: str, base_effect: float, jitter: float
) -> float:
    """A seeded, reproducible effect near base_effect; no wall-clock, no OS entropy."""

    digest = canonical_sha256({"trace_id": trace_id, "seed": seed, "session_id": session_id})
    fraction = int(digest[:8], 16) / 0xFFFFFFFF  # in [0, 1]
    return round(base_effect + (fraction - 0.5) * 2.0 * jitter, 9)


def synthesize_trace_records(
    *,
    trace_id: str,
    seeds: Sequence[int],
    session_id: str = "session-0",
    base_effect: float = 0.6,
    jitter: float = 0.05,
) -> tuple[TraceRecord, ...]:
    """Build a deterministic, reproducible set of per-seed records for one trace identity.

    This is a test vector, not evidence. The same arguments always yield byte-identical records.
    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    _require_id(trace_id, "synthesize_trace_records.trace_id")
    if len(seeds) < 2:
        raise TraceStabilityRefusal("synthesis needs at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise TraceStabilityRefusal("synthesis seeds must be distinct")
    if jitter < 0.0:
        raise TraceStabilityRefusal("synthesis jitter must be nonnegative")
    trace_sha256 = canonical_sha256({"trace_id": trace_id, "session_id": session_id})
    records: list[TraceRecord] = []
    for seed in seeds:
        effect = _deterministic_effect(trace_id, seed, session_id, base_effect, jitter)
        records.append(
            TraceRecord(
                trace_id=trace_id,
                seed=seed,
                session_id=session_id,
                effect=effect,
                trace_sha256=trace_sha256,
            )
        )
    return tuple(records)


def build_default_contract(trace_id: str = "trace.candidate") -> TraceStabilityContract:
    """Return a non-vacuous default stability contract at the epoch minimum."""

    return TraceStabilityContract(
        schema=TRACE_STABILITY_SCHEMA,
        trace_id=trace_id,
        stability_metric="sign-agreement",
        min_seeds=MIN_SEEDS,
        agreement_threshold=0.9,
        controls=REQUIRED_CONTROLS,
        matched_cost_required=True,
        matched=MatchedMeasurementBudget(
            seeds=MIN_SEEDS, sessions=2, samples_per_seed=256, compute_units=1024
        ),
    )


def dead_control_outcomes(agreement: float = 0.5) -> tuple[ControlOutcome, ...]:
    """Return the mandatory control set with every arm failing to reproduce the trace."""

    if agreement >= 0.9:
        raise TraceStabilityRefusal("dead controls must fall below a plausible stability threshold")
    return tuple(
        ControlOutcome(control=control, agreement=agreement, reproduced=False)
        for control in REQUIRED_CONTROLS
    )


def coverage() -> dict[str, Sequence[str]]:
    """Static record of which G1-C0 sub-questions this scaffold supports (readiness only)."""

    return {
        "seed-stability": (
            "TraceStabilityContract refuses fewer than MIN_SEEDS seeds",
            "StabilityVerdict refuses fewer than min_seeds distinct seeds among the records",
        ),
        "session-stability": (
            "TraceRecord binds every measurement to a named session id",
            "synthesize_trace_records reproduces records byte-identically per session",
        ),
        "control-death": (
            "REQUIRED_CONTROLS pins single-seed, shuffled-seed, permuted-trace, and label-shuffled",
            "StabilityVerdict.decide returns stable only when no control reproduces the trace",
        ),
        "license-gate": (
            "MechanismLicenseGate is off by default and refuses authorization without a receipt",
            "LicenseReceipt grants a license only for a stable, independently replicated verdict",
        ),
    }
