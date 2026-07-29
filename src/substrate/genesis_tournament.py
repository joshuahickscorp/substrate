"""Stage 5. Every surviving arm over many histories and many families.

Two properties this module is responsible for, beyond running the arms:

Ranking is by verified developmental utility, not by raw capacity. An arm that
answers well without having developed anything is a record store, and the
record-store null is scored alongside every candidate precisely so that
capacity alone is visible and cannot be mistaken for development.

Equal budgets are not equal spend. An arm may legitimately be more efficient
inside its budget, but a win bought by spending several times more compute than
the comparator is a compute artefact until proven otherwise. The tournament
measures utilisation, and when the winner outspends its comparator beyond
tolerance it reports the win as requiring a compute-matched rerun rather than
quietly accepting it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_harness as H
from substrate import genesis_material as M
from substrate import genesis_statistics as S


class TournamentRefused(RuntimeError):
    """A tournament condition that would invalidate the ranking was violated."""


@dataclass(frozen=True, slots=True)
class Unit:
    """One developmental history plus its probes and its scoring functions."""

    history_id: int
    family: str
    observations: tuple[M.Observation, ...]
    alternative_observations: tuple[M.Observation, ...]
    probes: H.ProbeSplit
    judge: H.Judge


UnitProvider = Callable[[str, int], Unit]


@dataclass(frozen=True, slots=True)
class ArmSummary:
    arm: str
    is_candidate: bool
    mean_score: float
    histories: int
    mean_committed_rewrites: float
    mean_refused_rewrites: float
    mean_compute: float
    mean_peak_bytes: float
    exhausted_count: int
    developmental_utility: float
    complexity_weight: float


def _developmental_utility(score: float, record_store_score: float) -> float:
    """Score above what a pure record store achieves on the same instances."""
    return score - record_store_score


def run(
    *,
    arms: Sequence[str],
    families: Sequence[str],
    histories: Sequence[int],
    provider: UnitProvider,
    envelope: str = "1GB",
    operation_budget: int = 2_000_000,
    durable_write_budget: int = 4_096,
    record_store: str = "record_store_null",
    enforce_scale: bool = True,
) -> dict[str, Any]:
    """Run every arm over every family and history under identical conditions.

    ``enforce_scale`` is only lowered for diagnostic runs, which never feed a
    classification. The published tournament always enforces it.
    """
    if len(histories) < C.TOURNAMENT_MINIMUM_HISTORIES:
        raise TournamentRefused(f"the tournament needs at least {C.TOURNAMENT_MINIMUM_HISTORIES} developmental histories")
    if len(histories) > C.TOURNAMENT_MAXIMUM_HISTORIES:
        raise TournamentRefused(f"the tournament is capped at {C.TOURNAMENT_MAXIMUM_HISTORIES} developmental histories")
    if len(families) < C.TOURNAMENT_MINIMUM_FAMILIES:
        raise TournamentRefused(f"the tournament needs at least {C.TOURNAMENT_MINIMUM_FAMILIES} challenge families")
    if record_store not in arms:
        raise TournamentRefused("the record-store null must run alongside every candidate")

    registered = set(M.registered())
    missing = [arm for arm in arms if arm not in registered]
    if missing:
        raise TournamentRefused(f"unregistered arms: {missing}")

    factories: dict[str, Callable[[M.Opportunity], M.CognitiveMaterial]] = {
        arm: (lambda opportunity, name=arm: M.build(name, opportunity)) for arm in arms
    }

    rows: list[dict[str, Any]] = []
    episodes = 0
    for family in families:
        for history_id in histories:
            unit = provider(family, history_id)
            result = H.run_history(
                history_id=history_id,
                family=family,
                arms=factories,
                observations=unit.observations,
                alternative_observations=unit.alternative_observations,
                probes=unit.probes,
                judge=unit.judge,
                envelope=envelope,
                operation_budget=operation_budget,
                durable_write_budget=durable_write_budget,
            )
            probe_count = len(unit.probes.development) + len(unit.probes.retention) + len(unit.probes.scoring)
            episodes += probe_count * len(arms)
            for arm, arm_run in result["runs"].items():
                rows.append(
                    {
                        "family": family,
                        "history_id": history_id,
                        "arm": arm,
                        "score": arm_run.score,
                        "retention_score": arm_run.retention_score,
                        "development_score": arm_run.development_score,
                        "committed": sum(1 for receipt in arm_run.receipts if receipt.committed),
                        "refused": sum(1 for receipt in arm_run.receipts if not receipt.committed),
                        "compute": arm_run.cost.get("compute", 0),
                        "peak_bytes": arm_run.peak_resident_bytes,
                        "exhausted": arm_run.exhausted,
                        "mechanism": arm_run.mechanism,
                        "stream_transform": arm_run.stream_transform,
                    }
                )

    in_range = C.TOURNAMENT_MINIMUM_EPISODES <= episodes <= C.TOURNAMENT_MAXIMUM_EPISODES
    if enforce_scale and not in_range:
        raise TournamentRefused(
            f"episode count {episodes} outside the frozen range "
            f"[{C.TOURNAMENT_MINIMUM_EPISODES}, {C.TOURNAMENT_MAXIMUM_EPISODES}]"
        )

    return {
        "rows": rows,
        "scale_enforced": enforce_scale,
        "episodes_in_frozen_range": in_range,
        "arms": list(arms),
        "families": list(families),
        "histories": list(histories),
        "envelope": envelope,
        "episodes": episodes,
        "operation_budget": operation_budget,
        "durable_write_budget": durable_write_budget,
        "record_store": record_store,
        "activation": False,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(result: Mapping[str, Any]) -> dict[str, Any]:
    """Rank arms by developmental utility over the record-store floor."""
    rows = result["rows"]
    record_store = result["record_store"]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    floor_by_cell = {
        (row["family"], row["history_id"]): row["score"] for row in rows if row["arm"] == record_store
    }

    summaries: dict[str, ArmSummary] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        utilities = [
            _developmental_utility(row["score"], floor_by_cell[(row["family"], row["history_id"])]) for row in arm_rows
        ]
        canonical = C.S2_ALIASES.get(arm, arm)
        summaries[arm] = ArmSummary(
            arm=arm,
            is_candidate=canonical in C.CANDIDATES,
            mean_score=_mean([row["score"] for row in arm_rows]),
            histories=len({row["history_id"] for row in arm_rows}),
            mean_committed_rewrites=_mean([row["committed"] for row in arm_rows]),
            mean_refused_rewrites=_mean([row["refused"] for row in arm_rows]),
            mean_compute=_mean([row["compute"] for row in arm_rows]),
            mean_peak_bytes=_mean([row["peak_bytes"] for row in arm_rows]),
            exhausted_count=sum(1 for row in arm_rows if row["exhausted"]),
            developmental_utility=_mean(utilities),
            complexity_weight=float(C.CANDIDATES.get(canonical, {}).get("complexity_weight", 0.0)),
        )

    candidates = {arm: row for arm, row in summaries.items() if row.is_candidate}
    if not candidates:
        raise TournamentRefused("no candidate arm ran in the tournament")

    # Rank on developmental utility. Ties break toward the simpler material,
    # because a more complex arm that only matches a simpler one has not earned
    # its complexity.
    ranked = sorted(
        candidates.values(),
        key=lambda row: (-round(row.developmental_utility, 9), row.complexity_weight, row.arm),
    )
    selected = ranked[0]

    return {
        "summaries": {arm: asdict(row) for arm, row in summaries.items()},
        "ranking": [row.arm for row in ranked],
        "selected_candidate": selected.arm,
        "selected_developmental_utility": selected.developmental_utility,
        "record_store_mean_score": summaries[record_store].mean_score,
        "activation": False,
    }


def compute_utilisation(result: Mapping[str, Any], summary: Mapping[str, Any], *, candidate: str, comparator: str) -> dict[str, Any]:
    """Is the win explained by the winner simply spending more compute?

    Equal budgets do not make equal spend. Efficiency inside a shared budget is
    a legitimate advantage; outspending the comparator several times over is
    not, until a compute-matched rerun says otherwise.
    """
    summaries = summary["summaries"]
    candidate_compute = summaries[candidate]["mean_compute"]
    comparator_compute = summaries[comparator]["mean_compute"]
    ratio = candidate_compute / comparator_compute if comparator_compute else float("inf")
    within_tolerance = abs(candidate_compute - comparator_compute) <= C.PARITY_RELATIVE_TOLERANCE * max(
        candidate_compute, comparator_compute, 1.0
    )
    return {
        "candidate": candidate,
        "comparator": comparator,
        "candidate_mean_compute": candidate_compute,
        "comparator_mean_compute": comparator_compute,
        "ratio": ratio,
        "within_tolerance": within_tolerance,
        "compute_matched_rerun_required": ratio > 1.0 + C.PARITY_RELATIVE_TOLERANCE,
        "budget": result["operation_budget"],
        "activation": False,
    }


def history_scores(result: Mapping[str, Any], *, family: str | None = None) -> list[S.HistoryScore]:
    """One scalar per arm per history, which is the only analysis input.

    Scores are averaged over families within a history, because the history is
    the independent unit and its families are repeated measures on it.
    """
    accumulated: dict[tuple[int, str], list[float]] = {}
    for row in result["rows"]:
        if family is not None and row["family"] != family:
            continue
        accumulated.setdefault((row["history_id"], row["arm"]), []).append(row["score"])
    return [
        S.HistoryScore(history_id, arm, _mean(values))
        for (history_id, arm), values in sorted(accumulated.items())
    ]


def demo() -> None:
    """Runnable self-check on synthetic units with a known answer."""
    import substrate.genesis_controls  # noqa: F401
    import substrate.genesis_k_advanced  # noqa: F401
    import substrate.genesis_k_basic  # noqa: F401
    import substrate.genesis_k_structural  # noqa: F401

    families = [f"family_{index}" for index in range(12)]
    histories = list(range(32))

    def provider(family: str, history_id: int) -> Unit:
        observations = tuple(
            M.Observation(index, f"c{index % 3}", (index % 3 - 1, (index * 2) % 3 - 1), elapsed_ms=5, teaching=index % 7 == 0)
            for index in range(12)
        )
        alternative = tuple(
            M.Observation(index, f"c{index % 3}", ((index + 1) % 3 - 1, (index * 5) % 3 - 1), elapsed_ms=5)
            for index in range(12)
        )
        probes = H.split_probes([M.Probe(index, family, f"c{index % 3}", (index % 3 - 1,), 2) for index in range(6)])
        constant = 0.5
        judge = H.Judge(
            score_development=lambda answers: constant,
            score_retention=lambda answers: constant,
            score_scoring=lambda answers: constant,
        )
        return Unit(history_id, family, observations, alternative, probes, judge)

    arms = ["K1_monolithic_plastic_field", C.CANONICAL_S2_ID, "record_store_null"]
    result = run(arms=arms, families=families, histories=histories, provider=provider, operation_budget=500_000, enforce_scale=False)
    assert result["episodes"] == 12 * 32 * 6 * 3, result["episodes"]
    assert not result["episodes_in_frozen_range"], "the diagnostic run must not masquerade as a full tournament"

    try:
        run(arms=arms, families=families, histories=histories, provider=provider, operation_budget=500_000)
    except TournamentRefused:
        pass
    else:  # pragma: no cover
        raise AssertionError("the episode scale floor was not enforced")

    summary = summarise(result)
    assert summary["selected_candidate"] == "K1_monolithic_plastic_field", summary["selected_candidate"]
    # Every arm scored the same constant, so developmental utility is zero: the
    # tournament must not manufacture a winner out of a flat field.
    assert abs(summary["selected_developmental_utility"]) < 1e-12, summary

    scores = history_scores(result)
    analysis = S.decisive_analysis(scores, candidate="K1_monolithic_plastic_field", comparator=C.CANONICAL_S2_ID)
    assert analysis["effect"] == 0.0 and not analysis["primary_gate_pass"], analysis

    utilisation = compute_utilisation(result, summary, candidate="K1_monolithic_plastic_field", comparator=C.CANONICAL_S2_ID)
    assert utilisation["ratio"] > 1.0, utilisation
    assert utilisation["compute_matched_rerun_required"], utilisation

    try:
        run(arms=arms, families=families[:4], histories=histories, provider=provider)
    except TournamentRefused:
        pass
    else:  # pragma: no cover
        raise AssertionError("the family minimum was not enforced")

    try:
        run(arms=["K1_monolithic_plastic_field"], families=families, histories=histories, provider=provider)
    except TournamentRefused:
        pass
    else:  # pragma: no cover
        raise AssertionError("the record-store floor was not required")

    print(f"genesis tournament self-check passed: {result['episodes']} episodes, flat field produced no winner")


if __name__ == "__main__":
    demo()
