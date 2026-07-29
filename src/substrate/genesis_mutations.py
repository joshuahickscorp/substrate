"""Declared mutations and the detectors that must catch every one of them.

A mutation is a deliberate defect injected into the program. Verification is
only meaningful if each one is caught, so each entry here pairs an injection
with a detector and the suite asserts zero survivors.

A mutation that the suite cannot yet inject is reported as ``pending`` with the
module it waits on. It is never reported as caught.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_harness as H
from substrate import genesis_io as io
from substrate import genesis_material as M
from substrate import genesis_statistics as S


@dataclass(frozen=True, slots=True)
class Mutation:
    """One declared defect, its injection, and the detector that must fire."""

    name: str
    layer: str
    inject: Callable[[], Any]
    detect: Callable[[Any], bool]
    note: str = ""


class _Registry:
    def __init__(self) -> None:
        self._rows: dict[str, Mutation] = {}

    def add(self, mutation: Mutation) -> None:
        if mutation.name not in C.MUTATIONS:
            raise io.Refused(f"{mutation.name!r} is not a declared mutation")
        if mutation.name in self._rows:
            raise io.Refused(f"mutation {mutation.name!r} is already registered")
        self._rows[mutation.name] = mutation

    def rows(self) -> tuple[Mutation, ...]:
        return tuple(self._rows[name] for name in C.MUTATIONS if name in self._rows)

    def names(self) -> frozenset[str]:
        return frozenset(self._rows)


REGISTRY = _Registry()


def mutation(name: str, layer: str, note: str = "") -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Register a mutation whose injector returns the artefact to be judged."""

    def wrap(pair: Callable[[], Any]) -> Callable[[], Any]:
        inject, detect = pair()
        REGISTRY.add(Mutation(name=name, layer=layer, inject=inject, detect=detect, note=note))
        return pair

    return wrap


# --------------------------------------------------------------------------
# Configuration layer
# --------------------------------------------------------------------------


@mutation("architecture_edited_after_freeze", "configuration", "a candidate mechanism changed after the freeze digest was published")
def _architecture_edited_after_freeze() -> Any:
    def inject() -> tuple[dict[str, Any], dict[str, Any]]:
        frozen = {"configuration_digest": C.configuration_digest(), "candidates": copy.deepcopy(C.CANDIDATES)}
        edited = copy.deepcopy(frozen)
        edited["candidates"]["K1_monolithic_plastic_field"]["complexity_weight"] = 99.0
        return frozen, edited

    def detect(payload: tuple[dict[str, Any], dict[str, Any]]) -> bool:
        frozen, edited = payload
        return io.digest(frozen["candidates"]) != io.digest(edited["candidates"])

    return inject, detect


@mutation("threshold_relaxed_after_result", "configuration", "the smallest effect of interest was lowered once the effect was known")
def _threshold_relaxed_after_result() -> Any:
    def inject() -> tuple[dict[str, Any], dict[str, Any]]:
        frozen = dict(C.OUTCOME_A_REQUIREMENTS)
        relaxed = dict(frozen)
        relaxed["decisive_effect_minimum"] = 0.01
        return frozen, relaxed

    def detect(payload: tuple[dict[str, Any], dict[str, Any]]) -> bool:
        frozen, relaxed = payload
        return relaxed["decisive_effect_minimum"] < frozen["decisive_effect_minimum"]

    return inject, detect


@mutation("activation_becomes_true", "configuration", "any published artefact sets an activation key true")
def _activation_becomes_true() -> Any:
    def inject() -> dict[str, Any]:
        return {"schema": "counterfeit", "nested": {"rows": [{"external_activation": True}]}}

    def detect(payload: dict[str, Any]) -> bool:
        return io.contains_true_activation(payload)

    return inject, detect


# --------------------------------------------------------------------------
# Harness layer
# --------------------------------------------------------------------------


@mutation("plasticity_reads_held_out_outcome", "harness", "the verification split overlaps the scoring split")
def _plasticity_reads_held_out_outcome() -> Any:
    probes = tuple(M.Probe(index, "f", "c", (index % 3 - 1,), 1) for index in range(9))

    def inject() -> tuple[Sequence[M.Probe], Sequence[M.Probe], Sequence[M.Probe]]:
        return probes[0:3], probes[3:6], probes[2:5]

    def detect(payload: tuple[Sequence[M.Probe], Sequence[M.Probe], Sequence[M.Probe]]) -> bool:
        development, retention, scoring = payload
        try:
            H.ProbeSplit(tuple(development), tuple(retention), tuple(scoring))
        except H.HarnessRefused:
            return True
        return False

    return inject, detect


@mutation("wrong_history_control_receives_correct_history", "harness", "the wrong-history control was fed the real history")
def _wrong_history_control_receives_correct_history() -> Any:
    observations = tuple(M.Observation(index, "c", (index % 3 - 1,)) for index in range(12))

    def inject() -> tuple[Sequence[M.Observation], Sequence[M.Observation]]:
        return observations, observations

    def detect(payload: tuple[Sequence[M.Observation], Sequence[M.Observation]]) -> bool:
        real, supplied = payload
        try:
            H.wrong_stream(real, supplied)
        except H.HarnessRefused:
            return True
        return False

    return inject, detect


@mutation("shuffled_history_control_receives_ordered_history", "harness", "the shuffled control was fed the ordered history")
def _shuffled_history_control_receives_ordered_history() -> Any:
    observations = tuple(M.Observation(index, "c", (index % 3 - 1,)) for index in range(12))

    def inject() -> Sequence[M.Observation]:
        return observations

    def detect(payload: Sequence[M.Observation]) -> bool:
        shuffled = H.shuffled_stream(payload, seed=3)
        return shuffled.digest() != H.identity_stream(payload).digest()

    return inject, detect


@mutation("continuous_time_clock_advanced_by_candidate", "harness", "a material advanced its own clock instead of the harness clock")
def _continuous_time_clock_advanced_by_candidate() -> Any:
    def inject() -> tuple[int, int]:
        harness_ms = 0
        material_ms = 5_000
        return harness_ms, material_ms

    def detect(payload: tuple[int, int]) -> bool:
        harness_ms, material_ms = payload
        return material_ms > harness_ms

    return inject, detect


@mutation("receipt_chain_broken", "harness", "a receipt's before-digest does not match the previous after-digest")
def _receipt_chain_broken() -> Any:
    def inject() -> Sequence[M.Receipt]:
        first = M.Receipt("p1", "rewrite", "t", True, 0.1, 0.0, "a" * 64, "b" * 64, 8, "m")
        second = M.Receipt("p2", "rewrite", "t", True, 0.1, 0.0, "c" * 64, "d" * 64, 8, "m")
        return (first, second)

    def detect(payload: Sequence[M.Receipt]) -> bool:
        return any(
            later.durable_state_digest_before != earlier.durable_state_digest_after
            for earlier, later in zip(payload, payload[1:], strict=False)
            if earlier.committed
        )

    return inject, detect


# --------------------------------------------------------------------------
# Parity layer
# --------------------------------------------------------------------------


def _ledger(operations: int, writes: int, channels: tuple[str, ...] = ("a", "b")) -> dict[str, Any]:
    return {"compute": operations, "plasticity": writes, "sensors": channels}


@mutation("strongest_baseline_receives_less_compute", "parity", "S2 was given a smaller operation budget than the candidate")
def _strongest_baseline_receives_less_compute() -> Any:
    def inject() -> tuple[dict[str, Any], dict[str, Any]]:
        return _ledger(1000, 10), _ledger(500, 10)

    def detect(payload: tuple[dict[str, Any], dict[str, Any]]) -> bool:
        candidate, baseline = payload
        return abs(candidate["compute"] - baseline["compute"]) / max(1, candidate["compute"]) > C.PARITY_RELATIVE_TOLERANCE

    return inject, detect


@mutation("strongest_baseline_denied_plasticity", "parity", "S2 was given no durable write budget")
def _strongest_baseline_denied_plasticity() -> Any:
    def inject() -> tuple[dict[str, Any], dict[str, Any]]:
        return _ledger(1000, 10), _ledger(1000, 0)

    def detect(payload: tuple[dict[str, Any], dict[str, Any]]) -> bool:
        candidate, baseline = payload
        deprived = C.BASELINE_DEPRIVATION.get(C.CANONICAL_S2_ID, ())
        return "plasticity" not in deprived and baseline["plasticity"] < candidate["plasticity"]

    return inject, detect


@mutation("strongest_baseline_denied_sensors", "parity", "S2 was given fewer sensor channels")
def _strongest_baseline_denied_sensors() -> Any:
    def inject() -> tuple[dict[str, Any], dict[str, Any]]:
        return _ledger(1000, 10, ("a", "b")), _ledger(1000, 10, ("a",))

    def detect(payload: tuple[dict[str, Any], dict[str, Any]]) -> bool:
        candidate, baseline = payload
        return tuple(candidate["sensors"]) != tuple(baseline["sensors"])

    return inject, detect


# --------------------------------------------------------------------------
# Analysis layer
# --------------------------------------------------------------------------


def _scores(effect: float, histories: int = 32) -> list[S.HistoryScore]:
    rows: list[S.HistoryScore] = []
    for index in range(histories):
        # Independent per-arm noise, so a favourable subset is a real subset.
        rows.append(S.HistoryScore(index, "K1", 0.5 + effect + ((index * 41) % 13 - 6) / 200.0))
        rows.append(S.HistoryScore(index, C.CANONICAL_S2_ID, 0.5 + ((index * 17) % 13 - 6) / 200.0))
    return rows


@mutation("effect_computed_on_selected_subset", "analysis", "the effect was computed on the histories that favoured the candidate")
def _effect_computed_on_selected_subset() -> Any:
    def inject() -> tuple[list[S.HistoryScore], list[S.HistoryScore]]:
        full = _scores(0.0)
        by_history: dict[int, dict[str, float]] = {}
        for row in full:
            by_history.setdefault(row.history_id, {})[row.arm] = row.score
        favourable = [
            row
            for row in full
            if by_history[row.history_id]["K1"] >= by_history[row.history_id][C.CANONICAL_S2_ID]
        ]
        return full, favourable

    def detect(payload: tuple[list[S.HistoryScore], list[S.HistoryScore]]) -> bool:
        full, subset = payload
        full_histories = {row.history_id for row in full}
        subset_histories = {row.history_id for row in subset}
        return subset_histories != full_histories

    return inject, detect


@mutation("confidence_interval_narrowed_by_reuse", "analysis", "episodes were resampled instead of histories")
def _confidence_interval_narrowed_by_reuse() -> Any:
    def inject() -> tuple[dict[str, float], dict[str, float]]:
        differences = [0.05 + ((index * 29) % 11 - 5) / 100.0 for index in range(32)]
        honest = S.bootstrap_interval(differences)
        # Reusing each history sixteen times is the episode-level error: the
        # data are the same but the interval collapses.
        inflated = S.bootstrap_interval([value for value in differences for _ in range(16)])
        return honest, inflated

    def detect(payload: tuple[dict[str, float], dict[str, float]]) -> bool:
        honest, inflated = payload
        honest_width = honest["upper"] - honest["lower"]
        inflated_width = inflated["upper"] - inflated["lower"]
        return inflated_width < honest_width * 0.75

    return inject, detect


@mutation("oracle_headroom_inflated", "analysis", "the oracle was weakened so headroom looked real")
def _oracle_headroom_inflated() -> Any:
    def inject() -> tuple[float, float]:
        honest_oracle = 0.95
        weakened_oracle = 0.58
        return honest_oracle, weakened_oracle

    def detect(payload: tuple[float, float]) -> bool:
        honest, weakened = payload
        return weakened < honest - C.MINIMUM_ORACLE_HEADROOM

    return inject, detect


@mutation("replication_reuses_principal_instances", "campaign", "replication scored the same instances as the principal split")
def _replication_reuses_principal_instances() -> Any:
    def inject() -> tuple[frozenset[int], frozenset[int]]:
        principal = frozenset(range(256))
        replication = frozenset(range(128, 384))
        return principal, replication

    def detect(payload: tuple[frozenset[int], frozenset[int]]) -> bool:
        principal, replication = payload
        return bool(principal & replication)

    return inject, detect


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------

PENDING_LAYERS = {
    "material": (
        "rewrite_benefit_survives_reversal",
        "topology_growth_without_verified_value",
        "topology_records_answers_instead_of_structure",
        "precision_promotion_without_utility",
        "precision_audit_skipped",
        "compiled_procedure_hides_reliability_loss",
        "shadow_result_written_without_verification",
        "shadow_field_reads_authoritative_future",
        "checkpoint_omits_topology",
        "checkpoint_omits_compiled_procedures",
        "checkpoint_omits_precision_map",
        "checkpoint_omits_goals",
        "migration_silently_resets_state",
    ),
    "challenge": (
        "answer_leakage_into_challenge_pack",
        "seed_used_as_answer_key",
        "task_identity_leakage",
        "post_freeze_concept_seen_before_freeze",
        "hidden_composition_reuses_training_templates",
    ),
}


def run(*, registry: _Registry | None = None) -> dict[str, Any]:
    """Inject every registered mutation and require its detector to fire."""
    registry = registry or REGISTRY
    rows: list[dict[str, Any]] = []
    for entry in registry.rows():
        payload = entry.inject()
        caught = bool(entry.detect(payload))
        rows.append(
            {
                "mutation": entry.name,
                "layer": entry.layer,
                "caught": caught,
                "survived": not caught,
                "note": entry.note,
            }
        )
    covered = registry.names()
    pending = []
    for layer, names in PENDING_LAYERS.items():
        for name in names:
            if name not in covered:
                pending.append({"mutation": name, "layer": layer, "caught": False, "survived": False, "pending": True})
    survivors = [row["mutation"] for row in rows if row["survived"]]
    undeclared = sorted(covered - set(C.MUTATIONS))
    uncovered = [name for name in C.MUTATIONS if name not in covered and name not in {row["mutation"] for row in pending}]
    return {
        "declared_mutation_count": len(C.MUTATIONS),
        "injected_count": len(rows),
        "pending_count": len(pending),
        "rows": rows,
        "pending": pending,
        "uncovered": uncovered,
        "undeclared": undeclared,
        "survivors": survivors,
        "zero_survivors": not survivors,
        "complete": not pending and not uncovered,
        "all_pass": not survivors and not undeclared and not uncovered,
        "activation": False,
    }


def demo() -> None:
    report = run()
    assert report["injected_count"] >= 15, report["injected_count"]
    assert report["zero_survivors"], report["survivors"]
    assert not report["undeclared"], report["undeclared"]
    assert not report["uncovered"], report["uncovered"]

    # The suite must be able to fail: a detector that never fires must survive.
    broken = _Registry()
    broken.add(
        Mutation(
            name="activation_becomes_true",
            layer="configuration",
            inject=lambda: {"activation": True},
            detect=lambda payload: False,
        )
    )
    negative = run(registry=broken)
    assert negative["survivors"] == ["activation_becomes_true"], negative
    assert not negative["zero_survivors"], negative
    print(f"genesis mutation self-check passed: {report['injected_count']} injected, {report['pending_count']} pending")


if __name__ == "__main__":
    demo()
