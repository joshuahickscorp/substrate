"""Scaffold spine for reproducible niches, real complementarity, and value-of-compute dispatch.

This module raises the SCAFFOLDING axis only for epoch G1 sub-questions C1, C2, and D1. It supplies
machine-checkable contracts and deterministic mechanics that ENCODE the bar a niche-dispatch
mechanism must clear before it may claim any value. It builds the harness, not the result.

C1 asks whether perspectives occupy reproducible, disjoint context niches. C2 asks whether those
niches are genuinely complementary, that is, whether each retained perspective owns at least one
uniquely positive niche and no retained perspective is net-harmful. D1 asks whether dispatching
compute across niches beats every matched control at matched cost.

The named prior null this module encodes is the EDCM invalid-bed null: if the evidence bed is uneven
in niche size, contains a net-harmful perspective, or has no uniquely positive niche, then the
forced assumption of universal complementarity is the wrong assumption and every downstream
complementarity or dispatch claim must fail closed. Nothing here asserts that any perspective
actually is complementary or that dispatch actually carries value. The synthetic beds are toy,
seeded partitions. The controls are declarations, not runs. Real activation is quarantined behind a
gate that local code refuses to pass without a valid confirmation receipt.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

NICHE_DISPATCH_SCHEMA = "mop-niche-dispatch/v1"

# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The matched-control family a dispatch claim must beat. Ordering is load-bearing for the interface
# digest and for the completeness check, so a membership or order drift is refused.
DISPATCH_CONTROLS: tuple[str, ...] = (
    "all-perspectives",  # always run every perspective; no dispatch saving
    "random-dispatch",  # dispatch to a perspective chosen at random per context
    "single-best",  # always run the single globally best perspective
    "majority-vote",  # run all perspectives and take a majority vote
)

# The scopes a confirmation receipt may authorize. Activation stays off until a receipt is supplied.
ACTIVATION_SCOPES: tuple[str, ...] = ("shadow-eval", "gated-pilot", "full-dispatch")


class NicheDispatchRefusal(ValueError):
    """Raised whenever a declaration is missing, malformed, drifted, or outside its declared scope."""


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise NicheDispatchRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise NicheDispatchRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_seed(value: int, label: str) -> None:
    if value < 0:
        raise NicheDispatchRefusal(f"{label} must be a nonnegative seed")


def _require_cells(cells: Sequence[str], label: str) -> tuple[str, ...]:
    if not cells:
        raise NicheDispatchRefusal(f"{label} must name at least one context cell")
    for cell in cells:
        _require_id(cell, f"{label} cell")
    if len(set(cells)) != len(cells):
        raise NicheDispatchRefusal(f"{label} cells must be unique")
    return tuple(sorted(cells))


# ---------------------------------------------------------------------------
# Section A. Context partition and single-perspective niche declaration (C1 substrate).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContextPartition:
    """A closed universe of disjoint context cells that niches are claimed over.

    Claim scope: deterministic programmatic mechanics only; no capability claim. A partition is a
    bookkeeping universe, never a statement that any partitioning of natural context is meaningful.
    """

    partition_id: str
    cells: tuple[str, ...]
    claim_scope: str = CLAIM_SCOPE
    schema: str = NICHE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NICHE_DISPATCH_SCHEMA:
            raise NicheDispatchRefusal(f"unsupported partition schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise NicheDispatchRefusal("context partition claim scope cannot be widened")
        _require_id(self.partition_id, "ContextPartition.partition_id")
        canonical = _require_cells(self.cells, "ContextPartition")
        if canonical != self.cells:
            raise NicheDispatchRefusal("partition cells must be sorted and unique")
        if len(self.cells) < 2:
            raise NicheDispatchRefusal("a partition needs at least two context cells to be a partition")

    def contains(self, cells: Sequence[str]) -> bool:
        return set(cells) <= set(self.cells)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "partition_id": self.partition_id,
            "cells": list(self.cells),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class NicheDeclaration:
    """A perspective plus the context cells where it is claimed useful, for one seed.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    perspective_id: str
    seed: int
    cells: tuple[str, ...]
    claim_scope: str = CLAIM_SCOPE
    schema: str = NICHE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NICHE_DISPATCH_SCHEMA:
            raise NicheDispatchRefusal(f"unsupported niche declaration schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise NicheDispatchRefusal("niche declaration claim scope cannot be widened")
        _require_id(self.perspective_id, "NicheDeclaration.perspective_id")
        _require_seed(self.seed, "NicheDeclaration.seed")
        canonical = _require_cells(self.cells, "NicheDeclaration")
        if canonical != self.cells:
            raise NicheDispatchRefusal("niche cells must be sorted and unique")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "perspective_id": self.perspective_id,
            "seed": self.seed,
            "cells": list(self.cells),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section B. Reproduction across seeds and the disjoint-niche contract (C1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReproducedNiche:
    """One perspective's niche declared across at least two seeds; reproduced only if it is stable.

    A niche reproduces only when every seed yields the identical cell set. Claim scope:
    deterministic programmatic mechanics only; no capability claim.
    """

    perspective_id: str
    seed_cells: Mapping[int, tuple[str, ...]]
    claim_scope: str = CLAIM_SCOPE
    schema: str = NICHE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NICHE_DISPATCH_SCHEMA:
            raise NicheDispatchRefusal(f"unsupported reproduced niche schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise NicheDispatchRefusal("reproduced niche claim scope cannot be widened")
        _require_id(self.perspective_id, "ReproducedNiche.perspective_id")
        if len(self.seed_cells) < 2:
            raise NicheDispatchRefusal("reproduction requires at least two independent seeds")
        normalized: dict[int, tuple[str, ...]] = {}
        for seed, cells in self.seed_cells.items():
            _require_seed(seed, "ReproducedNiche.seed")
            normalized[seed] = _require_cells(cells, f"ReproducedNiche.seed[{seed}]")
        distinct = {frozenset(cells) for cells in normalized.values()}
        if len(distinct) != 1:
            raise NicheDispatchRefusal(
                f"niche for {self.perspective_id!r} does not reproduce; seeds disagree on its cells"
            )

    @property
    def cells(self) -> tuple[str, ...]:
        any_seed = min(self.seed_cells)
        return tuple(sorted(self.seed_cells[any_seed]))

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.seed_cells))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "perspective_id": self.perspective_id,
            "seed_cells": {str(seed): sorted(self.seed_cells[seed]) for seed in self.seeds},
            "cells": list(self.cells),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class DisjointNicheContract:
    """C1 contract: every perspective niche reproduces across seeds and no two niches overlap.

    Claim scope: deterministic programmatic mechanics only; no capability claim. A green contract
    means the niches are reproducible and disjoint by construction, never that the partition carves
    natural context at a real joint.
    """

    partition: ContextPartition
    niches: tuple[ReproducedNiche, ...]
    claim_scope: str = CLAIM_SCOPE
    schema: str = NICHE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NICHE_DISPATCH_SCHEMA:
            raise NicheDispatchRefusal(f"unsupported disjoint niche schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise NicheDispatchRefusal("disjoint niche claim scope cannot be widened")
        if len(self.niches) < 2:
            raise NicheDispatchRefusal("a disjoint-niche bed needs at least two perspective niches")
        ids = [niche.perspective_id for niche in self.niches]
        if len(set(ids)) != len(ids):
            raise NicheDispatchRefusal("perspective ids in a disjoint-niche bed must be unique")
        seed_signatures = {niche.seeds for niche in self.niches}
        if len(seed_signatures) != 1:
            raise NicheDispatchRefusal("every niche must be reproduced over the same seed set")
        seen: set[str] = set()
        for niche in self.niches:
            if not self.partition.contains(niche.cells):
                raise NicheDispatchRefusal(
                    f"niche for {niche.perspective_id!r} names cells outside the declared partition"
                )
            overlap = seen & set(niche.cells)
            if overlap:
                raise NicheDispatchRefusal(
                    f"niches overlap on cells {sorted(overlap)}; niches must be disjoint"
                )
            seen |= set(niche.cells)

    @property
    def perspective_ids(self) -> tuple[str, ...]:
        return tuple(sorted(niche.perspective_id for niche in self.niches))

    def cells_owned_by(self, perspective_id: str) -> tuple[str, ...]:
        for niche in self.niches:
            if niche.perspective_id == perspective_id:
                return niche.cells
        raise NicheDispatchRefusal(f"no niche declared for perspective {perspective_id!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "partition": self.partition.payload(),
            "niches": [niche.payload() for niche in self.niches],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section C. Complementarity screening (C2) and the EDCM invalid-bed null.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerspectiveAssessment:
    """A perspective's measured net effect and the cells where it is uniquely positive.

    net_effect is a signed held-out delta vs the no-perspective baseline. A uniquely positive cell
    is one this perspective helps on where no retained rival also helps. Claim scope: deterministic
    programmatic mechanics only; no capability claim.
    """

    perspective_id: str
    net_effect: float
    unique_positive_cells: tuple[str, ...]
    claim_scope: str = CLAIM_SCOPE
    schema: str = NICHE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NICHE_DISPATCH_SCHEMA:
            raise NicheDispatchRefusal(f"unsupported assessment schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise NicheDispatchRefusal("assessment claim scope cannot be widened")
        _require_id(self.perspective_id, "PerspectiveAssessment.perspective_id")
        if self.net_effect != self.net_effect:  # NaN guard
            raise NicheDispatchRefusal("net effect must be a finite number")
        if self.net_effect in (float("inf"), float("-inf")):
            raise NicheDispatchRefusal("net effect must be a finite number")
        for cell in self.unique_positive_cells:
            _require_id(cell, "PerspectiveAssessment cell")
        if len(set(self.unique_positive_cells)) != len(self.unique_positive_cells):
            raise NicheDispatchRefusal("unique positive cells must be unique")

    @property
    def net_harmful(self) -> bool:
        return self.net_effect < 0.0

    @property
    def uniquely_positive(self) -> bool:
        return self.net_effect > 0.0 and len(self.unique_positive_cells) >= 1

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "perspective_id": self.perspective_id,
            "net_effect": self.net_effect,
            "unique_positive_cells": sorted(self.unique_positive_cells),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def assert_valid_edcm_bed(
    assessments: Sequence[PerspectiveAssessment],
    *,
    evenness_tolerance: float,
) -> None:
    """Encode the EDCM invalid-bed null: refuse the forced universal-complementarity assumption.

    The bed is invalid, and forced universal complementarity is the wrong assumption, when the bed
    is uneven in niche size, contains any net-harmful perspective, or has no uniquely positive
    perspective. Any of these fails closed. Claim scope: deterministic programmatic mechanics only;
    no capability claim.
    """

    if len(assessments) < 2:
        raise NicheDispatchRefusal("an EDCM bed needs at least two perspectives")
    if evenness_tolerance < 0.0:
        raise NicheDispatchRefusal("evenness tolerance must be nonnegative")
    ids = [a.perspective_id for a in assessments]
    if len(set(ids)) != len(ids):
        raise NicheDispatchRefusal("EDCM bed perspective ids must be unique")
    sizes = [len(a.unique_positive_cells) for a in assessments]
    if max(sizes) - min(sizes) > evenness_tolerance:
        raise NicheDispatchRefusal(
            "EDCM bed is uneven in niche size; forced universal complementarity is the wrong assumption"
        )
    if any(a.net_harmful for a in assessments):
        raise NicheDispatchRefusal(
            "EDCM bed contains a net-harmful perspective; universal complementarity cannot be assumed"
        )
    if not any(a.uniquely_positive for a in assessments):
        raise NicheDispatchRefusal(
            "EDCM bed has no uniquely positive perspective; complementarity is unsupported"
        )


@dataclass(frozen=True, slots=True)
class ComplementarityContract:
    """C2 contract: screen perspectives, keep only those with a uniquely positive niche.

    This contract refuses the universal-complementarity assumption outright: it never assumes every
    perspective helps. It retains a perspective only when that perspective is net-positive and owns
    at least one uniquely positive cell, and it refuses any perspective declared retained that is
    net-harmful. Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    assessments: tuple[PerspectiveAssessment, ...]
    evenness_tolerance: float
    assume_universal_complementarity: bool = False
    claim_scope: str = CLAIM_SCOPE
    schema: str = NICHE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NICHE_DISPATCH_SCHEMA:
            raise NicheDispatchRefusal(f"unsupported complementarity schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise NicheDispatchRefusal("complementarity claim scope cannot be widened")
        if self.assume_universal_complementarity:
            raise NicheDispatchRefusal(
                "complementarity contract refuses the universal-complementarity assumption"
            )
        # The EDCM null gates the whole contract: an invalid bed fails closed here.
        assert_valid_edcm_bed(self.assessments, evenness_tolerance=self.evenness_tolerance)

    @property
    def retained(self) -> tuple[str, ...]:
        """The perspectives that earn retention: net-positive and uniquely positive."""

        return tuple(sorted(a.perspective_id for a in self.assessments if a.uniquely_positive))

    @property
    def refused(self) -> tuple[str, ...]:
        """The perspectives screened out: not uniquely positive under this bed."""

        return tuple(sorted(a.perspective_id for a in self.assessments if not a.uniquely_positive))

    def assert_retention_honest(self, claimed_retained: Sequence[str]) -> None:
        """Fail closed if a caller tries to retain a perspective that did not earn it."""

        claimed = set(claimed_retained)
        earned = set(self.retained)
        harmful = {a.perspective_id for a in self.assessments if a.net_harmful}
        if claimed & harmful:
            raise NicheDispatchRefusal(f"cannot retain net-harmful perspectives {sorted(claimed & harmful)}")
        if not claimed <= earned:
            raise NicheDispatchRefusal(
                f"cannot retain perspectives without a uniquely positive niche {sorted(claimed - earned)}"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "evenness_tolerance": self.evenness_tolerance,
            "assume_universal_complementarity": self.assume_universal_complementarity,
            "assessments": [a.payload() for a in self.assessments],
            "retained": list(self.retained),
            "refused": list(self.refused),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section D. Matched compute and the value-of-compute dispatch contract (D1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedBudget:
    """The matched full-system compute budget every dispatch arm must be held to before comparison."""

    params: int
    flops: int
    wall_ticks: int
    dispatch_calls: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("wall_ticks", self.wall_ticks),
            ("dispatch_calls", self.dispatch_calls),
        ):
            if value <= 0:
                raise NicheDispatchRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "wall_ticks": self.wall_ticks,
            "dispatch_calls": self.dispatch_calls,
        }


@dataclass(frozen=True, slots=True)
class DispatchValueContract:
    """D1 contract: dispatch may claim value only if it beats every matched control at matched cost.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The controls tuple
    is fixed and its order is load-bearing. A losing or tied dispatch does not raise; it simply
    yields no value claim, which is the fail-closed default.
    """

    controls: tuple[str, ...]
    matched: MatchedBudget
    matched_cost_required: bool
    replication_min: int
    metric_name: str
    claim_scope: str = CLAIM_SCOPE
    schema: str = NICHE_DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NICHE_DISPATCH_SCHEMA:
            raise NicheDispatchRefusal(f"unsupported dispatch value schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise NicheDispatchRefusal("dispatch value claim scope cannot be widened")
        if tuple(self.controls) != DISPATCH_CONTROLS:
            raise NicheDispatchRefusal("dispatch controls are incomplete or out of canonical order")
        if not self.matched_cost_required:
            raise NicheDispatchRefusal("dispatch value must require matched full-system cost")
        if self.replication_min < 2:
            raise NicheDispatchRefusal("dispatch value requires at least two independent replications")
        _require_id(self.metric_name, "DispatchValueContract.metric_name")

    def may_claim_value(
        self,
        dispatch_score: float,
        control_scores: Mapping[str, float],
    ) -> bool:
        """Return True only if dispatch strictly beats every matched control on the declared metric.

        Fails closed on an incomplete or non-finite control set. A tie or loss returns False.
        """

        if set(control_scores) != set(DISPATCH_CONTROLS):
            raise NicheDispatchRefusal("control scores must cover exactly the declared control family")
        for name, value in control_scores.items():
            if value != value or value in (float("inf"), float("-inf")):
                raise NicheDispatchRefusal(f"control score for {name!r} must be a finite number")
        if dispatch_score != dispatch_score or dispatch_score in (float("inf"), float("-inf")):
            raise NicheDispatchRefusal("dispatch score must be a finite number")
        return all(dispatch_score > value for value in control_scores.values())

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "controls": list(self.controls),
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
            "replication_min": self.replication_min,
            "metric_name": self.metric_name,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section E. Activation gate. Real dispatch activation is quarantined behind a receipt this
# process cannot mint, encoding that activation is not earned yet.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfirmationReceipt:
    """A confirmation receipt that a downstream authority issues to license dispatch activation.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The receipt is a
    declaration; possessing one does not create evidence, it only records that an authority signed.
    """

    issuer: str
    license_sha256: str
    scope: str

    def __post_init__(self) -> None:
        _require_id(self.issuer, "ConfirmationReceipt.issuer")
        _require_sha256(self.license_sha256, "ConfirmationReceipt.license_sha256")
        if self.scope not in ACTIVATION_SCOPES:
            raise NicheDispatchRefusal(f"unsupported activation scope {self.scope!r}")

    def payload(self) -> dict[str, Any]:
        return {"issuer": self.issuer, "license_sha256": self.license_sha256, "scope": self.scope}


@dataclass(frozen=True, slots=True)
class DispatchActivationGate:
    """A fail-closed gate. Local code cannot activate real dispatch without a valid receipt.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The gate exists so
    that scaffold code can never quietly stand in for an earned, externally confirmed deployment.
    """

    activation_required: bool = True
    local_activation_permitted: bool = False

    def __post_init__(self) -> None:
        if not self.activation_required:
            raise NicheDispatchRefusal("the activation gate only guards work that requires activation")
        if self.local_activation_permitted:
            raise NicheDispatchRefusal("local activation of dispatch is never permitted by default")

    def authorize(self, receipt: ConfirmationReceipt | None = None) -> ConfirmationReceipt:
        """Fail closed unless a valid confirmation receipt is supplied by an external authority."""

        if receipt is None:
            raise NicheDispatchRefusal(
                "dispatch activation is not earned yet; supply a valid confirmation receipt from an "
                "external authority with its own randomization and held-out evaluation"
            )
        if receipt.scope not in ACTIVATION_SCOPES:
            raise NicheDispatchRefusal(f"receipt scope {receipt.scope!r} is not an activation scope")
        return receipt

    def payload(self) -> dict[str, Any]:
        return {
            "activation_required": self.activation_required,
            "local_activation_permitted": self.local_activation_permitted,
        }


# ---------------------------------------------------------------------------
# Section F. Deterministic synthetic bed builders (seeded, reproducible; used by tests and wiring).
# ---------------------------------------------------------------------------


def _cell_id(index: int) -> str:
    return f"cell.{index:03d}"


def _perspective_id(index: int) -> str:
    return f"perspective.{index:02d}"


def synthesize_partition(*, n_perspectives: int, cells_each: int) -> ContextPartition:
    """Build a deterministic partition with disjoint cell blocks, one block per perspective."""

    if n_perspectives < 2:
        raise NicheDispatchRefusal("a synthetic bed needs at least two perspectives")
    if cells_each < 1:
        raise NicheDispatchRefusal("each perspective needs at least one cell")
    total = n_perspectives * cells_each
    cells = tuple(_cell_id(i) for i in range(total))
    return ContextPartition(partition_id="synthetic.partition", cells=cells)


def synthesize_disjoint_bed(
    *,
    seeds: Sequence[int],
    n_perspectives: int = 3,
    cells_each: int = 2,
) -> DisjointNicheContract:
    """Build a reproducible, disjoint bed. Cells depend on perspective index only, so the niche is
    identical across seeds by construction and therefore reproduces. Claim scope: deterministic
    programmatic mechanics only; no capability claim.
    """

    if len(seeds) < 2:
        raise NicheDispatchRefusal("reproduction requires at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise NicheDispatchRefusal("synthetic bed seeds must be unique")
    partition = synthesize_partition(n_perspectives=n_perspectives, cells_each=cells_each)
    niches: list[ReproducedNiche] = []
    for p in range(n_perspectives):
        block = tuple(_cell_id(p * cells_each + k) for k in range(cells_each))
        seed_cells = {int(seed): block for seed in seeds}
        niches.append(ReproducedNiche(perspective_id=_perspective_id(p), seed_cells=seed_cells))
    return DisjointNicheContract(partition=partition, niches=tuple(niches))


def synthesize_valid_assessments(
    *,
    seed: int,
    n_perspectives: int = 3,
    cells_each: int = 2,
) -> tuple[PerspectiveAssessment, ...]:
    """Build a deterministic set of even, net-positive, uniquely positive assessments (a valid bed).

    Effects are drawn from a seeded hash so they are reproducible without any wall-clock or OS
    entropy. Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    _require_seed(seed, "synthesize_valid_assessments.seed")
    assessments: list[PerspectiveAssessment] = []
    for p in range(n_perspectives):
        digest = hashlib.sha256(f"{seed}:{p}".encode("ascii")).digest()
        magnitude = 1.0 + (digest[0] % 16) / 16.0  # strictly positive, deterministic
        cells = tuple(_cell_id(p * cells_each + k) for k in range(cells_each))
        assessments.append(
            PerspectiveAssessment(
                perspective_id=_perspective_id(p),
                net_effect=round(magnitude, 6),
                unique_positive_cells=cells,
            )
        )
    return tuple(assessments)


def build_default_dispatch_value_contract() -> DispatchValueContract:
    """Return a canonical, non-vacuous D1 dispatch value contract."""

    return DispatchValueContract(
        controls=DISPATCH_CONTROLS,
        matched=MatchedBudget(params=256, flops=8192, wall_ticks=64, dispatch_calls=32),
        matched_cost_required=True,
        replication_min=3,
        metric_name="held_out_task_score",
    )


# ---------------------------------------------------------------------------
# Coverage record and capability declaration.
# ---------------------------------------------------------------------------

SCIENTIFIC_CAPABILITY_CLAIM = False


def coverage() -> dict[str, Sequence[str]]:
    """Static record of which epoch G1 sub-questions this scaffold arms (readiness only)."""

    return {
        "C1": (
            "reproducible niches via ReproducedNiche, which refuses unless seeds agree on the cells",
            "disjoint niches via DisjointNicheContract, which refuses any cross-perspective overlap",
        ),
        "C2": (
            "complementarity screening that retains only net-positive, uniquely positive perspectives",
            "explicit refusal of the universal-complementarity assumption and net-harmful perspectives",
            "EDCM invalid-bed null encoded as a fail-closed check on evenness, harm, and uniqueness",
        ),
        "D1": (
            "matched compute via MatchedBudget and the all-perspectives/random/single-best/majority controls",
            "dispatch may claim value only if it strictly beats every matched control at matched cost",
            "real activation quarantined behind a receipt gate that is off by default",
        ),
    }
