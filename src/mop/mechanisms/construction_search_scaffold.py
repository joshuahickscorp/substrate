
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

CONSTRUCTION_SEARCH_SCHEMA = "mop-construction-search/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

SCIENTIFIC_CAPABILITY_CLAIM = False

PRIOR_NULL = (
    "G0 formation mechanics existed without demonstrated efficacy; charged-cost net improvement "
    "over every declared control must be strictly positive before search value may be claimed"
)

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MASK64 = (1 << 64) - 1
_LCG_MULT = 6364136223846793005
_LCG_ADD = 1442695040888963407

REQUIRED_CONTROLS: tuple[str, ...] = (
    "no-search",  # the G0 formation default coalition; no search is spent
    "random-construction",  # random subsets sampled under the same budget
    "greedy-only",  # single greedy pass without restarts
)
ORACLE_HEADROOM_CONTROL: str = "oracle-headroom"
ALL_CONTROLS: tuple[str, ...] = REQUIRED_CONTROLS + (ORACLE_HEADROOM_CONTROL,)

VERDICT_METRICS: tuple[str, ...] = (
    "gross_improvement",
    "charged_search_cost",
    "net_improvement",
    "oracle_headroom_gap",
)

FORMATION_DEFAULT: tuple[int, ...] = (0, 1)


class ConstructionSearchRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise ConstructionSearchRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ConstructionSearchRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_positive(value: int, label: str) -> None:
    if value <= 0:
        raise ConstructionSearchRefusal(f"{label} must be positive (non-vacuous)")




@dataclass(frozen=True, slots=True)
class SearchBudget:

    candidate_evaluations: int
    objective_queries: int
    wall_proxy_units: int
    memory_bytes: int
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_positive(self.candidate_evaluations, "candidate_evaluations")
        _require_positive(self.objective_queries, "objective_queries")
        _require_positive(self.wall_proxy_units, "wall_proxy_units")
        _require_positive(self.memory_bytes, "memory_bytes")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("search budget claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_evaluations": self.candidate_evaluations,
            "objective_queries": self.objective_queries,
            "wall_proxy_units": self.wall_proxy_units,
            "memory_bytes": self.memory_bytes,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())




@dataclass(frozen=True, slots=True)
class SealedObjective:

    objective_id: str
    task_ids: tuple[str, ...]
    num_members: int
    size_penalty: float
    objective_sha256: str
    sealed_before_search: bool = True
    schema: str = CONSTRUCTION_SEARCH_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CONSTRUCTION_SEARCH_SCHEMA:
            raise ConstructionSearchRefusal(f"unsupported objective schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("objective claim scope cannot be widened")
        _require_id(self.objective_id, "objective_id")
        if len(self.task_ids) < 2:
            raise ConstructionSearchRefusal("a multi-task objective needs at least two tasks")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ConstructionSearchRefusal("objective task ids must be unique")
        for task_id in self.task_ids:
            _require_id(task_id, "task_id")
        if self.num_members < 2:
            raise ConstructionSearchRefusal("objective needs at least two candidate members")
        if not self.size_penalty > 0.0:
            raise ConstructionSearchRefusal("objective size penalty must be positive")
        if not self.sealed_before_search:
            raise ConstructionSearchRefusal("objective must be sealed before search begins")
        _require_sha256(self.objective_sha256, "objective_sha256")
        if self.objective_sha256 != self._core_digest():
            raise ConstructionSearchRefusal("objective seal digest does not match its core spec")

    def _core_digest(self) -> str:
        return canonical_sha256(
            {
                "objective_id": self.objective_id,
                "task_ids": list(self.task_ids),
                "num_members": self.num_members,
                "size_penalty": self.size_penalty,
            }
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "objective_id": self.objective_id,
            "task_ids": list(self.task_ids),
            "num_members": self.num_members,
            "size_penalty": self.size_penalty,
            "objective_sha256": self.objective_sha256,
            "sealed_before_search": self.sealed_before_search,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def seal_objective(
    *, objective_id: str, task_ids: Sequence[str], num_members: int, size_penalty: float
) -> SealedObjective:

    core = canonical_sha256(
        {
            "objective_id": objective_id,
            "task_ids": list(task_ids),
            "num_members": num_members,
            "size_penalty": size_penalty,
        }
    )
    return SealedObjective(
        objective_id=objective_id,
        task_ids=tuple(task_ids),
        num_members=num_members,
        size_penalty=size_penalty,
        objective_sha256=core,
    )




@dataclass(frozen=True, slots=True)
class ConstructionControl:

    id: str
    family: str
    rationale: str
    is_oracle_headroom: bool
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_id(self.id, "control id")
        if self.family not in ALL_CONTROLS:
            raise ConstructionSearchRefusal(f"unsupported control family {self.family!r}")
        if not self.rationale.strip():
            raise ConstructionSearchRefusal("control rationale must not be empty")
        expected_headroom = self.family == ORACLE_HEADROOM_CONTROL
        if self.is_oracle_headroom != expected_headroom:
            raise ConstructionSearchRefusal("oracle-headroom flag disagrees with the control family")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("control claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "rationale": self.rationale,
            "is_oracle_headroom": self.is_oracle_headroom,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class ConstructionControlSet:

    schema: str
    controls: tuple[ConstructionControl, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CONSTRUCTION_SEARCH_SCHEMA:
            raise ConstructionSearchRefusal(f"unsupported control set schema {self.schema!r}")
        families = tuple(control.family for control in self.controls)
        if families != ALL_CONTROLS:
            raise ConstructionSearchRefusal("control set membership or order drift")
        ids = [control.id for control in self.controls]
        if len(set(ids)) != len(ids):
            raise ConstructionSearchRefusal("control ids must be unique")
        headroom = [control for control in self.controls if control.is_oracle_headroom]
        if len(headroom) != 1:
            raise ConstructionSearchRefusal("control set needs exactly one oracle-headroom reference")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("control set claim scope cannot be widened")

    def cheap_control_families(self) -> tuple[str, ...]:
        return tuple(control.family for control in self.controls if not control.is_oracle_headroom)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controls": [control.payload() for control in self.controls],
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_default_control_set() -> ConstructionControlSet:

    rationale = {
        "no-search": "reuse the G0 formation default coalition; spends no search",
        "random-construction": "sample random subsets under the identical matched budget",
        "greedy-only": "one greedy add pass with no restarts under the matched budget",
        "oracle-headroom": "exhaustive best over all subsets; the uncharged ceiling, never a claim",
    }
    controls = tuple(
        ConstructionControl(
            id=f"ctrl.{family.replace('-', '_')}",
            family=family,
            rationale=rationale[family],
            is_oracle_headroom=(family == ORACLE_HEADROOM_CONTROL),
        )
        for family in ALL_CONTROLS
    )
    return ConstructionControlSet(schema=CONSTRUCTION_SEARCH_SCHEMA, controls=controls)




@dataclass(frozen=True, slots=True)
class ConstructionSearchContract:

    objective: SealedObjective
    budget: SearchBudget
    controls: ConstructionControlSet
    search_budget_charged: bool = True
    matched_cost_required: bool = True
    schema: str = CONSTRUCTION_SEARCH_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CONSTRUCTION_SEARCH_SCHEMA:
            raise ConstructionSearchRefusal(f"unsupported contract schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("contract claim scope cannot be widened")
        if not self.matched_cost_required:
            raise ConstructionSearchRefusal("construction search must require matched full-system cost")
        if not self.search_budget_charged:
            raise ConstructionSearchRefusal(
                "construction search must charge the search cost against the budget"
            )
        if self.budget.candidate_evaluations < self.objective.num_members:
            raise ConstructionSearchRefusal(
                "the matched budget must permit at least one evaluation per candidate member"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "objective": self.objective.payload(),
            "budget": self.budget.payload(),
            "controls": self.controls.payload(),
            "search_budget_charged": self.search_budget_charged,
            "matched_cost_required": self.matched_cost_required,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_default_contract(
    *, num_members: int = 8, num_tasks: int = 3, size_penalty: float = 0.03
) -> ConstructionSearchContract:

    objective = seal_objective(
        objective_id="obj.shadow_coalition",
        task_ids=tuple(f"task.{i}" for i in range(num_tasks)),
        num_members=num_members,
        size_penalty=size_penalty,
    )
    budget = SearchBudget(
        candidate_evaluations=8 * num_members,
        objective_queries=8 * num_members,
        wall_proxy_units=64,
        memory_bytes=8192,
    )
    return ConstructionSearchContract(
        objective=objective, budget=budget, controls=build_default_control_set()
    )




@dataclass(frozen=True, slots=True)
class SearchValueVerdict:

    gross_improvement: float
    charged_search_cost: float
    oracle_headroom_gap: float
    claims_improvement: bool
    matched_cost_charged: bool = True
    schema: str = CONSTRUCTION_SEARCH_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CONSTRUCTION_SEARCH_SCHEMA:
            raise ConstructionSearchRefusal(f"unsupported verdict schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("verdict claim scope cannot be widened")
        if not self.matched_cost_charged:
            raise ConstructionSearchRefusal(
                "a search-value verdict is refused unless the matched search cost is charged"
            )
        if self.charged_search_cost < 0.0:
            raise ConstructionSearchRefusal("charged search cost cannot be negative")
        if self.oracle_headroom_gap < 0.0:
            raise ConstructionSearchRefusal("oracle headroom gap cannot be negative")
        if self.claims_improvement and not self.net_improvement > 0.0:
            raise ConstructionSearchRefusal(
                "prior null holds: improvement may be claimed only when charged-cost net is positive"
            )

    @property
    def net_improvement(self) -> float:
        return self.gross_improvement - self.charged_search_cost

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "gross_improvement": self.gross_improvement,
            "charged_search_cost": self.charged_search_cost,
            "net_improvement": self.net_improvement,
            "oracle_headroom_gap": self.oracle_headroom_gap,
            "claims_improvement": self.claims_improvement,
            "matched_cost_charged": self.matched_cost_charged,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())




class ConstructionSearchActivationRefusal(ConstructionSearchRefusal):
    pass


@dataclass(frozen=True, slots=True)
class ConstructionSearchActivationGate:

    activated: bool = False
    confirmation_receipt_sha256: str = ""
    license_id: str = ""
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("activation gate claim scope cannot be widened")
        if self.activated:
            _require_sha256(self.confirmation_receipt_sha256, "confirmation_receipt_sha256")
            _require_id(self.license_id, "license_id")
        else:
            if self.confirmation_receipt_sha256 or self.license_id:
                raise ConstructionSearchRefusal("an inactive gate must carry neither a receipt nor a license")

    def authorize_claim(self) -> None:

        if not self.activated:
            raise ConstructionSearchActivationRefusal(
                "construction-search value claims are not activated; supply a confirmation receipt "
                "and license from an independent replication before any claim can be authorized"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "activated": self.activated,
            "confirmation_receipt_sha256": self.confirmation_receipt_sha256,
            "license_id": self.license_id,
            "claim_scope": self.claim_scope,
        }



_MAX_ORACLE_MEMBERS = 16


class _DeterministicStream:

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ConstructionSearchRefusal("stream seed must be nonnegative")
        self._state = (seed * _LCG_MULT + _LCG_ADD) & _MASK64

    def next_float(self) -> float:
        self._state = (self._state * _LCG_MULT + _LCG_ADD) & _MASK64
        return ((self._state >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def _affinity_matrix(seed: int, num_members: int, num_tasks: int) -> tuple[tuple[float, ...], ...]:

    stream = _DeterministicStream(seed)
    return tuple(tuple(stream.next_float() for _ in range(num_tasks)) for _ in range(num_members))


def _score_coalition(
    matrix: tuple[tuple[float, ...], ...],
    members: Iterable[int],
    size_penalty: float,
    num_tasks: int,
) -> float:

    member_list = list(members)
    if not member_list:
        return 0.0
    total = 0.0
    for task in range(num_tasks):
        total += max(matrix[m][task] for m in member_list)
    return total / num_tasks - size_penalty * len(member_list)


@dataclass(frozen=True, slots=True)
class SearchTrace:

    seed: int
    num_members: int
    num_tasks: int
    size_penalty: float
    per_eval_cost: float
    scores: dict[str, float]
    evaluations: dict[str, int]
    schema: str = CONSTRUCTION_SEARCH_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CONSTRUCTION_SEARCH_SCHEMA:
            raise ConstructionSearchRefusal(f"unsupported trace schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchRefusal("trace claim scope cannot be widened")
        if self.per_eval_cost < 0.0:
            raise ConstructionSearchRefusal("per-evaluation cost cannot be negative")
        expected = set(REQUIRED_CONTROLS) | {ORACLE_HEADROOM_CONTROL, "construction-search"}
        if set(self.scores) != expected:
            raise ConstructionSearchRefusal("trace scores must cover every arm exactly")
        if set(self.evaluations) != expected:
            raise ConstructionSearchRefusal("trace evaluation counts must cover every arm exactly")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "num_members": self.num_members,
            "num_tasks": self.num_tasks,
            "size_penalty": self.size_penalty,
            "per_eval_cost": self.per_eval_cost,
            "scores": {k: self.scores[k] for k in sorted(self.scores)},
            "evaluations": {k: self.evaluations[k] for k in sorted(self.evaluations)},
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _random_construction(
    matrix: tuple[tuple[float, ...], ...],
    num_members: int,
    num_tasks: int,
    size_penalty: float,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    stream = _DeterministicStream(seed ^ 0x9E3779B97F4A7C15)
    best = 0.0
    evals = 0
    while evals < budget:
        members = [m for m in range(num_members) if stream.next_float() < 0.5]
        best = max(best, _score_coalition(matrix, members, size_penalty, num_tasks))
        evals += 1
    return best, evals


def _greedy_only(
    matrix: tuple[tuple[float, ...], ...],
    num_members: int,
    num_tasks: int,
    size_penalty: float,
    budget: int,
) -> tuple[float, int]:
    members: list[int] = []
    current = 0.0
    evals = 0
    improved = True
    while improved and evals < budget:
        improved = False
        best_gain_member = -1
        best_gain_score = current
        for m in range(num_members):
            if m in members or evals >= budget:
                continue
            evals += 1
            trial = _score_coalition(matrix, members + [m], size_penalty, num_tasks)
            if trial > best_gain_score:
                best_gain_score = trial
                best_gain_member = m
        if best_gain_member >= 0:
            members.append(best_gain_member)
            current = best_gain_score
            improved = True
    return current, evals


def _construction_search(
    matrix: tuple[tuple[float, ...], ...],
    num_members: int,
    num_tasks: int,
    size_penalty: float,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    stream = _DeterministicStream(seed ^ 0xD1B54A32D192ED03)
    best = 0.0
    evals = 0
    while evals < budget:
        members = {m for m in range(num_members) if stream.next_float() < 0.5}
        score = _score_coalition(matrix, members, size_penalty, num_tasks)
        evals += 1
        improved = True
        while improved and evals < budget:
            improved = False
            for m in range(num_members):
                if evals >= budget:
                    break
                trial = set(members)
                trial.symmetric_difference_update({m})
                trial_score = _score_coalition(matrix, trial, size_penalty, num_tasks)
                evals += 1
                if trial_score > score:
                    members = trial
                    score = trial_score
                    improved = True
        best = max(best, score)
    return best, evals


def _oracle_headroom(
    matrix: tuple[tuple[float, ...], ...],
    num_members: int,
    num_tasks: int,
    size_penalty: float,
) -> float:
    if num_members > _MAX_ORACLE_MEMBERS:
        raise ConstructionSearchRefusal("oracle headroom is only defined for small member counts")
    best = 0.0
    for mask in range(1 << num_members):
        members = [i for i in range(num_members) if (mask >> i) & 1]
        best = max(best, _score_coalition(matrix, members, size_penalty, num_tasks))
    return best


def run_construction_search(
    *,
    seed: int,
    num_members: int = 8,
    num_tasks: int = 3,
    size_penalty: float = 0.03,
    budget: int | None = None,
    per_eval_cost: float = 0.0,
) -> SearchTrace:

    if num_members < 2:
        raise ConstructionSearchRefusal("run needs at least two candidate members")
    if num_tasks < 2:
        raise ConstructionSearchRefusal("run needs at least two tasks")
    effective_budget = 8 * num_members if budget is None else budget
    _require_positive(effective_budget, "budget")
    matrix = _affinity_matrix(seed, num_members, num_tasks)

    no_search_score = _score_coalition(matrix, FORMATION_DEFAULT, size_penalty, num_tasks)
    random_score, random_evals = _random_construction(
        matrix, num_members, num_tasks, size_penalty, effective_budget, seed
    )
    greedy_score, greedy_evals = _greedy_only(matrix, num_members, num_tasks, size_penalty, effective_budget)
    search_score, search_evals = _construction_search(
        matrix, num_members, num_tasks, size_penalty, effective_budget, seed
    )
    oracle_score = _oracle_headroom(matrix, num_members, num_tasks, size_penalty)

    return SearchTrace(
        seed=seed,
        num_members=num_members,
        num_tasks=num_tasks,
        size_penalty=size_penalty,
        per_eval_cost=per_eval_cost,
        scores={
            "no-search": no_search_score,
            "random-construction": random_score,
            "greedy-only": greedy_score,
            "construction-search": search_score,
            ORACLE_HEADROOM_CONTROL: oracle_score,
        },
        evaluations={
            "no-search": 1,
            "random-construction": random_evals,
            "greedy-only": greedy_evals,
            "construction-search": search_evals,
            ORACLE_HEADROOM_CONTROL: 1 << num_members,
        },
    )


def verdict_from_trace(trace: SearchTrace, *, matched_cost_charged: bool = True) -> SearchValueVerdict:

    cheap_best = max(trace.scores[family] for family in REQUIRED_CONTROLS)
    search_score = trace.scores["construction-search"]
    gross = search_score - cheap_best
    charged = trace.per_eval_cost * trace.evaluations["construction-search"]
    gap = trace.scores[ORACLE_HEADROOM_CONTROL] - search_score
    net = gross - charged
    return SearchValueVerdict(
        gross_improvement=gross,
        charged_search_cost=charged,
        oracle_headroom_gap=max(0.0, gap),
        claims_improvement=net > 0.0,
        matched_cost_charged=matched_cost_charged,
    )


# Section H. Coverage record for the epoch sub-questions (readiness only).


def coverage() -> dict[str, Sequence[str]]:

    return {
        "is-the-objective-sealed-before-search": (
            "SealedObjective binds its core spec with a digest and refuses a moving target",
            "seal_objective recomputes the digest so a rewritten objective fails closed",
        ),
        "does-search-beat-the-cheap-controls": (
            "ConstructionControlSet requires no-search, random-construction, and greedy-only in order",
            "verdict_from_trace takes the best cheap control as the bar to clear",
        ),
        "does-it-survive-charging-the-search-cost": (
            "SearchValueVerdict refuses an improvement claim unless charged-cost net is positive",
            "SearchBudget fixes a non-vacuous matched budget every arm must share",
            "PRIOR_NULL is encoded so a zero or negative net leaves the null standing",
        ),
        "how-far-below-oracle-headroom-does-it-sit": (
            "the oracle-headroom control gives an exhaustive, uncharged ceiling reference",
            "SearchValueVerdict reports a nonnegative oracle_headroom_gap for every run",
        ),
        "is-a-value-claim-earned-yet": (
            "ConstructionSearchActivationGate is off by default and refuses local authorization",
            "activation requires an independent confirmation receipt and license id",
        ),
    }
