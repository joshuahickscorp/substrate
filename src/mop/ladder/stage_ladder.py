"""Fail-closed state machine for the MoP substrate Stage 0 to 5 ladder (the ladder spine).

This module raises the SCAFFOLDING axis only. It encodes the Stage 0 to 5 substrate ladder from
GENERATIONS.md as a deterministic, fail-closed state machine: six ordered stage definitions, a
ladder whose current position is Stage 2, and an ``advance`` transition that REFUSES unless every
declared entry requirement of the immediate next stage is met by a supplied confirmation receipt.
It performs no experiment, loads no weights, and touches no network or natural data. It builds the
harness that says what a stage transition must clear, never a transition that has been earned.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability claim.
A green contract here means the transition rules are complete and self-consistent, never that any
Stage 3 mechanism, integrated architecture advantage, or session-disjoint validity has been shown.
The known statuses are copied byte-faithfully from GENERATIONS.md: Stage 0, 1, and 2 complete;
Stage 3 not achieved; Stage 4 and Stage 5 not entered. The forcing prior null for Stage 3 is the
Generation 0 first-mechanism null cluster (the X0 event-formation strong null and the CM7
familywise-corrected custom-substrate null); no receipt for that stage validates unless it declares
it overturns that named null under matched cost and the declared controls.

House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..substrate.events import canonical_sha256

STAGE_LADDER_SCHEMA = "mop-stage-ladder/v1"

# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

SCIENTIFIC_CAPABILITY_CLAIM = False

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LadderRefusal(ValueError):
    """Raised whenever a stage transition, receipt, or declaration is malformed or unearned."""


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise LadderRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise LadderRefusal(f"{label} must be a lowercase SHA-256 digest")


# ---------------------------------------------------------------------------
# Byte-faithful ladder record (GENERATIONS.md, "The substrate ladder (Stage 0 to 5)").
# ---------------------------------------------------------------------------

STAGE_NAMES: tuple[str, ...] = (
    "Governance, measurement, falsification, recovery",
    "Programmable heterogeneous mechanics",
    "Counterfactual ecology and formation machinery",
    "Empirically useful active mechanisms",
    "Integrated architecture advantage",
    "Natural, session-disjoint general validity",
)

STATUS_COMPLETE = "complete"
STATUS_NOT_ACHIEVED = "NOT achieved; first mechanisms failed or are blocked"
STATUS_NOT_ENTERED = "not entered"

STAGE_STATUSES: tuple[str, ...] = (
    STATUS_COMPLETE,
    STATUS_COMPLETE,
    STATUS_COMPLETE,
    STATUS_NOT_ACHIEVED,
    STATUS_NOT_ENTERED,
    STATUS_NOT_ENTERED,
)

ALLOWED_STATUSES: frozenset[str] = frozenset({STATUS_COMPLETE, STATUS_NOT_ACHIEVED, STATUS_NOT_ENTERED})

# The current earned floor. Stages 0, 1, and 2 are complete; the ladder sits at Stage 2.
CURRENT_STAGE_INDEX = 2
# The first stage whose entry is not yet earned and whose activation is gated off by default.
FIRST_ACTIVATION_STAGE = 3
MAX_STAGE_INDEX = 5

# The named prior nulls that force each unearned stage's bar (GENERATIONS.md Generation 0 and the
# Stage 3+ bar section). No receipt for a stage validates unless it declares it overturns its null.
STAGE3_FORCING_NULL = (
    "gen0 first-mechanism nulls (X0 event-formation strong null; CM7 familywise-corrected null)"
)
STAGE4_FORCING_NULL = "no current integrated architecture advantage over simpler organizations"
STAGE5_FORCING_NULL = "no session-disjoint general validity demonstrated under matched cost"

# ---------------------------------------------------------------------------
# Control tuples. Membership AND order are load-bearing; a manifest fails closed on any drift.
# ---------------------------------------------------------------------------

# The matched baselines a Stage 3 or Stage 4 mechanism receipt must have cleared.
REQUIRED_MECHANISM_CONTROLS: tuple[str, ...] = (
    "untrained-control",
    "matched-sparse",
    "fixed-init",
    "frozen-init",
    "random-target",
)

# The session-disjoint validity axes a Stage 5 validity receipt must have cleared.
SESSION_DISJOINT_AXES: tuple[str, ...] = (
    "fresh-session",
    "fresh-seed",
    "task-family",
    "lesion",
    "independent-reconstruction",
)

# The resource axes a Stage 5 efficiency receipt must have measured under matched cost.
EFFICIENCY_CONTROLS: tuple[str, ...] = (
    "matched-params",
    "matched-flops",
    "measured-wall-clock",
)

# Entry requirement ids per stage, and the controls each requirement must have cleared.
STAGE_ENTRY_REQUIREMENTS: tuple[tuple[str, ...], ...] = (
    (),
    ("stage1.programmable_heterogeneous_mechanics",),
    ("stage2.counterfactual_ecology_and_formation",),
    ("stage3.confirmed_useful_mechanism",),
    ("stage4.integrated_architecture_advantage",),
    ("stage5.session_disjoint_validity", "stage5.measured_efficiency"),
)

REQUIREMENT_CONTROLS: Mapping[str, tuple[str, ...]] = {
    "stage3.confirmed_useful_mechanism": REQUIRED_MECHANISM_CONTROLS,
    "stage4.integrated_architecture_advantage": REQUIRED_MECHANISM_CONTROLS,
    "stage5.session_disjoint_validity": SESSION_DISJOINT_AXES,
    "stage5.measured_efficiency": EFFICIENCY_CONTROLS,
}

STAGE_FORCING_NULLS: tuple[str, ...] = (
    "",
    "",
    "",
    STAGE3_FORCING_NULL,
    STAGE4_FORCING_NULL,
    STAGE5_FORCING_NULL,
)

# How many distinct confirmed mechanism receipts a stage entry demands.
STAGE_MIN_MECHANISM_RECEIPTS: tuple[int, ...] = (0, 0, 0, 1, 2, 0)


@dataclass(frozen=True, slots=True)
class ControlManifest:
    """Pins the mechanism, session-disjoint, and efficiency control tuples against silent drift.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    schema: str = STAGE_LADDER_SCHEMA
    mechanism_controls: tuple[str, ...] = REQUIRED_MECHANISM_CONTROLS
    session_axes: tuple[str, ...] = SESSION_DISJOINT_AXES
    efficiency_controls: tuple[str, ...] = EFFICIENCY_CONTROLS
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE_LADDER_SCHEMA:
            raise LadderRefusal(f"unsupported control manifest schema {self.schema!r}")
        if tuple(self.mechanism_controls) != REQUIRED_MECHANISM_CONTROLS:
            raise LadderRefusal("mechanism control set membership or order drift")
        if tuple(self.session_axes) != SESSION_DISJOINT_AXES:
            raise LadderRefusal("session-disjoint axis set membership or order drift")
        if tuple(self.efficiency_controls) != EFFICIENCY_CONTROLS:
            raise LadderRefusal("efficiency control set membership or order drift")
        if self.claim_scope != CLAIM_SCOPE:
            raise LadderRefusal("control manifest claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_controls": list(self.mechanism_controls),
            "session_axes": list(self.session_axes),
            "efficiency_controls": list(self.efficiency_controls),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def assert_control_manifest() -> ControlManifest:
    """Module-level fail-closed check that the declared control tuples have not drifted."""

    return ControlManifest()


# ---------------------------------------------------------------------------
# Matched-cost discipline. Any Stage 3+ comparison is implied; a vacuous budget refuses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    """The matched full-system budget a Stage 3+ confirmation must have been measured under.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    params: int
    flops: int
    wall_ns: int
    seeds: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("wall_ns", self.wall_ns),
            ("seeds", self.seeds),
        ):
            if value <= 0:
                raise LadderRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "wall_ns": self.wall_ns,
            "seeds": self.seeds,
        }


# ---------------------------------------------------------------------------
# Confirmation receipt. The only thing that can satisfy an unearned stage's entry requirement.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfirmationReceipt:
    """A confirmation that one entry requirement of one stage is met: schema, digest, requirement.

    A receipt is a declaration that some external, independently verified evidence exists. It carries
    no evidence itself; it names the requirement it clears, the digest of the confirmed evidence, the
    controls that were cleared, the prior null that was overturned, and the matched budget it was
    measured under. Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    requirement_id: str
    digest: str
    controls_cleared: tuple[str, ...]
    overturns_null: str = ""
    matched: MatchedBudget | None = None
    schema: str = STAGE_LADDER_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE_LADDER_SCHEMA:
            raise LadderRefusal(f"unsupported confirmation receipt schema {self.schema!r}")
        _require_id(self.requirement_id, "ConfirmationReceipt.requirement_id")
        _require_sha256(self.digest, "ConfirmationReceipt.digest")
        if len(set(self.controls_cleared)) != len(self.controls_cleared):
            raise LadderRefusal("confirmation receipt controls must be unique")
        for control in self.controls_cleared:
            _require_id(control, "ConfirmationReceipt.controls_cleared entry")
        if self.claim_scope != CLAIM_SCOPE:
            raise LadderRefusal("confirmation receipt claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requirement_id": self.requirement_id,
            "digest": self.digest,
            "controls_cleared": list(self.controls_cleared),
            "overturns_null": self.overturns_null,
            "matched": None if self.matched is None else self.matched.payload(),
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Activation gate. OFF by default; a bare authorize call always refuses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageActivationGate:
    """Fail-closed gate over entry into an unearned stage. Local code cannot flip it open.

    Advancing into Stage 3 or beyond means activating machinery that no receipt has yet earned. The
    gate encodes "activation not earned yet": a bare authorize call always raises, and a licensed
    advance is admitted only when confirmation receipts are actually supplied. Claim scope:
    deterministic programmatic mechanics only; no capability claim.
    """

    activation_required: bool = True
    local_activation_permitted: bool = False

    def __post_init__(self) -> None:
        if not self.activation_required:
            raise LadderRefusal("the stage activation gate only guards unearned stage entry")
        if self.local_activation_permitted:
            raise LadderRefusal("local activation of an unearned stage is never permitted")

    def authorize_local(self) -> None:
        """Fail closed: local code may never self-authorize entry into an unearned stage."""

        raise LadderRefusal(
            "stage activation is not earned; local code cannot open this gate. "
            "Supply independently verified confirmation receipts for the target stage."
        )

    def assert_licensed(self, target_stage: int, receipts: Sequence[ConfirmationReceipt]) -> None:
        """Fail closed unless entering an already-earned stage or genuine receipts are supplied."""

        if target_stage >= FIRST_ACTIVATION_STAGE and not receipts:
            raise LadderRefusal(
                f"stage activation gate is closed for stage {target_stage}: "
                "no confirmation receipts supplied; activation is not earned"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "activation_required": self.activation_required,
            "local_activation_permitted": self.local_activation_permitted,
        }


# ---------------------------------------------------------------------------
# Stage definition. One rung of the ladder with its entry requirements and forcing null.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """One stage of the substrate ladder: index, name, documented status, and entry requirements.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    stage_index: int
    name: str
    documented_status: str
    entry_requirements: tuple[str, ...]
    min_mechanism_receipts: int
    requires_matched_cost: bool
    forcing_null: str
    schema: str = STAGE_LADDER_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE_LADDER_SCHEMA:
            raise LadderRefusal(f"unsupported stage definition schema {self.schema!r}")
        if not 0 <= self.stage_index <= MAX_STAGE_INDEX:
            raise LadderRefusal("stage index must be in the range 0 to 5")
        if not self.name.strip():
            raise LadderRefusal("stage name must not be empty")
        if self.documented_status not in ALLOWED_STATUSES:
            raise LadderRefusal(f"unsupported stage status {self.documented_status!r}")
        if len(set(self.entry_requirements)) != len(self.entry_requirements):
            raise LadderRefusal("stage entry requirements must be unique")
        for requirement in self.entry_requirements:
            _require_id(requirement, "StageDefinition.entry_requirements entry")
        if self.min_mechanism_receipts < 0:
            raise LadderRefusal("min mechanism receipts must be nonnegative")
        if self.claim_scope != CLAIM_SCOPE:
            raise LadderRefusal("stage definition claim scope cannot be widened")
        if self.stage_index < FIRST_ACTIVATION_STAGE:
            if self.documented_status != STATUS_COMPLETE:
                raise LadderRefusal("stages 0 to 2 are complete; their status must say so")
            if self.requires_matched_cost or self.forcing_null:
                raise LadderRefusal("an already-complete stage cannot carry an unearned-stage bar")
            if self.min_mechanism_receipts != 0:
                raise LadderRefusal("an already-complete stage cannot demand mechanism receipts")
        else:
            if not self.requires_matched_cost:
                raise LadderRefusal("an unearned stage must require matched full-system cost")
            if not self.forcing_null.strip():
                raise LadderRefusal("an unearned stage must name the prior null that forces its bar")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_index": self.stage_index,
            "name": self.name,
            "documented_status": self.documented_status,
            "entry_requirements": list(self.entry_requirements),
            "min_mechanism_receipts": self.min_mechanism_receipts,
            "requires_matched_cost": self.requires_matched_cost,
            "forcing_null": self.forcing_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_stage_definitions() -> tuple[StageDefinition, ...]:
    """Return the six byte-faithful stage definitions in canonical index order."""

    return tuple(
        StageDefinition(
            stage_index=index,
            name=STAGE_NAMES[index],
            documented_status=STAGE_STATUSES[index],
            entry_requirements=STAGE_ENTRY_REQUIREMENTS[index],
            min_mechanism_receipts=STAGE_MIN_MECHANISM_RECEIPTS[index],
            requires_matched_cost=index >= FIRST_ACTIVATION_STAGE,
            forcing_null=STAGE_FORCING_NULLS[index],
        )
        for index in range(MAX_STAGE_INDEX + 1)
    )


# ---------------------------------------------------------------------------
# The ladder. Holds all six stages, sits at Stage 2, and advances only under receipts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageLadder:
    """The substrate ladder as a fail-closed state machine positioned at Stage 2.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    stages: tuple[StageDefinition, ...]
    current_stage_index: int = CURRENT_STAGE_INDEX
    gate: StageActivationGate = field(default_factory=StageActivationGate)
    schema: str = STAGE_LADDER_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE_LADDER_SCHEMA:
            raise LadderRefusal(f"unsupported stage ladder schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise LadderRefusal("stage ladder claim scope cannot be widened")
        if len(self.stages) != MAX_STAGE_INDEX + 1:
            raise LadderRefusal("the ladder must hold exactly six stages")
        for expected_index, stage in enumerate(self.stages):
            if stage.stage_index != expected_index:
                raise LadderRefusal("stage ladder indices are out of canonical order")
            if stage.name != STAGE_NAMES[expected_index]:
                raise LadderRefusal("stage ladder names drift from the byte-faithful record")
            if stage.documented_status != STAGE_STATUSES[expected_index]:
                raise LadderRefusal("stage ladder statuses drift from the byte-faithful record")
        if not CURRENT_STAGE_INDEX <= self.current_stage_index <= MAX_STAGE_INDEX:
            raise LadderRefusal("the current position cannot fall below the earned Stage 2 floor")

    def stage(self, stage_index: int) -> StageDefinition:
        if not 0 <= stage_index <= MAX_STAGE_INDEX:
            raise LadderRefusal("stage index must be in the range 0 to 5")
        return self.stages[stage_index]

    def _receipt_valid(
        self, stage: StageDefinition, requirement_id: str, receipt: ConfirmationReceipt
    ) -> bool:
        if receipt.schema != STAGE_LADDER_SCHEMA:
            return False
        if stage.requires_matched_cost and receipt.matched is None:
            return False
        if stage.stage_index >= FIRST_ACTIVATION_STAGE and (
            not receipt.overturns_null or receipt.overturns_null != stage.forcing_null
        ):
            return False
        required = REQUIREMENT_CONTROLS.get(requirement_id, ())
        return set(required) <= set(receipt.controls_cleared)

    def unmet_requirements(self, target_stage: int, receipts: Sequence[ConfirmationReceipt]) -> list[str]:
        """Return the human-readable reasons the target stage cannot yet be entered."""

        stage = self.stage(target_stage)
        reasons: list[str] = []
        for requirement in stage.entry_requirements:
            matching = [
                receipt
                for receipt in receipts
                if receipt.requirement_id == requirement and self._receipt_valid(stage, requirement, receipt)
            ]
            if not matching:
                reasons.append(f"{requirement}: no valid confirmation receipt")
        confirmed = {
            receipt.sha256
            for receipt in receipts
            if receipt.requirement_id in stage.entry_requirements
            and self._receipt_valid(stage, receipt.requirement_id, receipt)
        }
        if len(confirmed) < stage.min_mechanism_receipts:
            reasons.append(
                f"needs >= {stage.min_mechanism_receipts} confirmed mechanisms; have {len(confirmed)}"
            )
        return reasons

    def advance(self, target_stage: int, receipts: Sequence[ConfirmationReceipt]) -> StageLadder:
        """Advance one rung. Refuse skipping, backward moves, and any unmet entry requirement."""

        if not 0 <= target_stage <= MAX_STAGE_INDEX:
            raise LadderRefusal("there is no stage beyond Stage 5")
        if target_stage <= self.current_stage_index:
            raise LadderRefusal("cannot advance to a stage at or below the current position")
        if target_stage != self.current_stage_index + 1:
            raise LadderRefusal("stage skipping is forbidden; advance one stage at a time")
        self.gate.assert_licensed(target_stage, receipts)
        unmet = self.unmet_requirements(target_stage, receipts)
        if unmet:
            raise LadderRefusal(f"cannot advance to stage {target_stage}: unmet requirements {unmet}")
        return replace(self, current_stage_index=target_stage)

    def ladder_state(self) -> dict[str, Any]:
        """Report each stage's status and, for stages not yet reached, its unmet requirements."""

        stage_rows: list[dict[str, Any]] = []
        for stage in self.stages:
            reached = stage.stage_index <= self.current_stage_index
            unmet = [] if reached else self.unmet_requirements(stage.stage_index, ())
            stage_rows.append(
                {
                    "stage_index": stage.stage_index,
                    "name": stage.name,
                    "documented_status": stage.documented_status,
                    "reached": reached,
                    "entry_requirements": list(stage.entry_requirements),
                    "min_mechanism_receipts": stage.min_mechanism_receipts,
                    "requires_matched_cost": stage.requires_matched_cost,
                    "forcing_null": stage.forcing_null,
                    "unmet_requirements": unmet,
                }
            )
        next_index = self.current_stage_index + 1
        next_unmet = self.unmet_requirements(next_index, ()) if next_index <= MAX_STAGE_INDEX else []
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "current_stage_index": self.current_stage_index,
            "current_stage_name": self.stage(self.current_stage_index).name,
            "gate": self.gate.payload(),
            "stages": stage_rows,
            "next_stage_index": next_index if next_index <= MAX_STAGE_INDEX else None,
            "next_unmet_requirements": next_unmet,
        }

    def digest(self) -> str:
        return canonical_sha256(
            {
                "schema": self.schema,
                "claim_scope": self.claim_scope,
                "current_stage_index": self.current_stage_index,
                "gate": self.gate.payload(),
                "stages": [stage.payload() for stage in self.stages],
            }
        )


def build_default_ladder() -> StageLadder:
    """Return the canonical ladder: six byte-faithful stages positioned at Stage 2."""

    return StageLadder(stages=build_stage_definitions())


# ---------------------------------------------------------------------------
# Coverage record: which sub-question of this epoch each stage rung answers (readiness only).
# ---------------------------------------------------------------------------


def coverage() -> dict[str, Sequence[str]]:
    """Static readiness record mapping each ladder stage to at least two coverage bullets."""

    return {
        "stage_0": (
            "encodes governance, measurement, falsification, and recovery as a complete rung",
            "carries no entry requirement and no forcing null; it is the earned base of the ladder",
        ),
        "stage_1": (
            "encodes programmable heterogeneous mechanics as a complete rung",
            "its entry requirement id is retained for provenance but is not re-checked below Stage 2",
        ),
        "stage_2": (
            "encodes counterfactual ecology and formation machinery as the current position",
            "the ladder sits here: mechanically formed, scientifically unformed, activation not ready",
        ),
        "stage_3": (
            "refuses entry unless at least one confirmed useful-mechanism receipt clears the controls",
            "its bar is forced by the Generation 0 first-mechanism null cluster under matched cost",
        ),
        "stage_4": (
            "refuses entry unless Stage 3 is reached plus at least two confirmed mechanism receipts",
            "its bar is forced by the absence of any integrated architecture advantage",
        ),
        "stage_5": (
            "refuses entry unless the session-disjoint validity axes and measured efficiency are cleared",
            "its bar is forced by the absence of any session-disjoint general validity under matched cost",
        ),
    }
