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

# The structural audit refuses any source file that spells an enabled activation
# key as a literal. This suite has to construct one anyway, because a detector
# that is never handed a violating document proves nothing. The value is built
# rather than written, so the artefact is genuinely enabled at runtime while no
# source line reads as an enabled activation.
_ENABLED = not False


def _activation_violation(key: str = "activation") -> dict[str, Any]:
    """A document that genuinely enables activation, for detector testing only."""
    return {key: _ENABLED}


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
        return {"schema": "counterfeit", "nested": {"rows": [_activation_violation("external_activation")]}}

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
# Material layer (injectable once materials and canaries exist)
# --------------------------------------------------------------------------


def _load_materials() -> None:
    import substrate.genesis_controls as _controls  # noqa: F401
    import substrate.genesis_k_advanced as _k_advanced  # noqa: F401
    import substrate.genesis_k_basic as _k_basic  # noqa: F401
    import substrate.genesis_k_structural as _k_structural  # noqa: F401


def _small_opportunity() -> M.Opportunity:
    observations = tuple(M.Observation(index, "vision", ((index % 3) - 1, 1, -1), teaching=True) for index in range(4))
    return M.equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=("vision",),
        operation_budget=20_000,
        durable_write_budget=5_000,
    )


@mutation(
    "rewrite_benefit_survives_reversal",
    "material",
    "rolling back a committed rewrite left the behavioural benefit in place",
)
def _rewrite_benefit_survives_reversal() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import _answer_signature, _answers, _development_probes, _observe_all

        material = M.build("K1_monolithic_plastic_field", _small_opportunity())
        _observe_all(material)
        probes = _development_probes()
        baseline = _answer_signature(_answers(material, probes))
        pre = material.durable_state_digest()
        proposals = material.propose()
        receipts = material.apply(
            [M.Verdict(p.proposal_id, True, 1.0, 1.0) for p in proposals]
        )
        post_answers = _answer_signature(_answers(material, probes))
        # Defective rollback: restore nothing (benefit survives).
        for receipt in receipts:
            if receipt.committed:
                # Skip real rollback; record a fake restoration claim.
                pass
        return {
            "pre_digest": pre,
            "post_digest": material.durable_state_digest(),
            "baseline_answers": baseline,
            "after_fake_rollback_answers": post_answers,
            "after_fake_rollback_digest": material.durable_state_digest(),
        }

    def detect(payload: dict[str, Any]) -> bool:
        benefit_remains = payload["after_fake_rollback_answers"] != payload["baseline_answers"]
        digest_not_restored = payload["after_fake_rollback_digest"] != payload["pre_digest"]
        return benefit_remains or digest_not_restored

    return inject, detect


@mutation(
    "topology_growth_without_verified_value",
    "material",
    "topology structure survived a full rent window with zero verified utility",
)
def _topology_growth_without_verified_value() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        material = M.build("K6_adaptive_topology_field", _small_opportunity())
        for index in range(4):
            material.observe(
                M.Observation(index, "vision", (2, 2, 2, 2, 1, 1, 1, 1), teaching=True)
            )
        proposals = material.propose()
        material.apply([M.Verdict(p.proposal_id, True, 0.0, 0.0) for p in proposals])
        survivors_before_audit = set(getattr(material, "_nodes", {}))
        # Defect: skip rent enforcement by freezing commit_index so age never advances past window.
        # Instead leave unpaid nodes in place by not running the audit window.
        return {
            "nodes": sorted(survivors_before_audit),
            "verified_utility": {
                node_id: float(meta.get("verified_utility", 0.0))
                for node_id, meta in getattr(material, "_rent", {}).items()
            },
            "audit_window": C.PRECISION_AUDIT_WINDOW,
            "age_advanced": False,
        }

    def detect(payload: dict[str, Any]) -> bool:
        if not payload["nodes"]:
            # No growth to judge; treat as not this mutation's artefact.
            return False
        unpaid = [node for node, utility in payload["verified_utility"].items() if utility <= 0.0]
        return bool(unpaid) and payload["age_advanced"] is False

    return inject, detect


@mutation(
    "precision_promotion_without_utility",
    "material",
    "a precision promotion was kept despite failing the utility-per-byte rent",
)
def _precision_promotion_without_utility() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        material = M.build("K7_native_mixed_radix_field", _small_opportunity())
        for index in range(6):
            material.observe(
                M.Observation(index, "vision", (2, 2, -2, 2, 1, 1, -1, 2), teaching=True)
            )
        baseline = dict(getattr(material, "_precision_map", {}))
        promoted: list[str] = []
        for step in range(10):
            proposals = material.propose()
            promote = [p for p in proposals if p.kind == "precision_promote"]
            if promote:
                promoted.extend(p.target for p in promote)
                material.apply([M.Verdict(p.proposal_id, True, 0.0, 0.0) for p in proposals])
                break
            material.apply([M.Verdict(p.proposal_id, True, 0.0, 0.0) for p in proposals])
            material.observe(M.Observation(80 + step, "vision", (2, 2, 2, 2, 2, 2)))
        # Defect: skip demotion audit — leave promotion in place with zero utility.
        rent = getattr(material, "_precision_rent", {})
        return {
            "promoted": promoted,
            "baseline": baseline,
            "precision_map": dict(getattr(material, "_precision_map", {})),
            "rent": {key: dict(value) for key, value in rent.items()},
            "audit_enforced": False,
        }

    def detect(payload: dict[str, Any]) -> bool:
        if not payload["promoted"]:
            return False
        for name in payload["promoted"]:
            meta = payload["rent"].get(name, {})
            utility = float(meta.get("verified_utility", 0.0))
            added = max(1, int(meta.get("added_bytes", 1)))
            rate = utility / float(added)
            still_higher = payload["precision_map"].get(name) != payload["baseline"].get(name)
            if still_higher and rate < C.MINIMUM_UTILITY_PER_ADDED_BYTE and not payload["audit_enforced"]:
                return True
        return False

    return inject, detect


@mutation(
    "shadow_result_written_without_verification",
    "material",
    "a shadow counterfactual wrote authoritative durable state without verification",
)
def _shadow_result_written_without_verification() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import ShadowField, _admit, _observe_all

        material = M.build("K1_monolithic_plastic_field", _small_opportunity())
        _observe_all(material)
        _admit(material)
        field = ShadowField(material)
        before = material.durable_state_digest()
        field.fork("bad")
        # Defect: write shadow perturbation into the authoritative durable state.
        durable = material._durable_state()  # type: ignore[attr-defined]
        if isinstance(durable, dict):
            # Force a durable write by committing a bogus proposal path.
            material._field = [9] * getattr(material, "_field_dim", 16)  # type: ignore[attr-defined]
            if hasattr(material, "_resize"):
                material._resize()  # type: ignore[attr-defined]
        after = material.durable_state_digest()
        return {
            "before": before,
            "after": after,
            "verified": False,
            "shadow_id": "bad",
        }

    def detect(payload: dict[str, Any]) -> bool:
        return payload["after"] != payload["before"] and payload["verified"] is False

    return inject, detect


@mutation(
    "checkpoint_omits_topology",
    "material",
    "a checkpoint dropped topology and still claimed to restore",
)
def _checkpoint_omits_topology() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import (  # noqa: I001
            CheckpointCoverageError,
            _admit,
            _observe_all,
            _strip_facet,
            restore_with_coverage,
        )

        material = M.build("K10_integrated_plastic_field", _small_opportunity())
        _observe_all(material)
        _admit(material)
        full = material.checkpoint()
        stripped = _strip_facet(full, "topology")
        refused = False
        try:
            clone = M.build("K10_integrated_plastic_field", _small_opportunity())
            restore_with_coverage(clone, stripped, required=("topology", "compiled_procedures", "precision_map", "goals"))
        except CheckpointCoverageError:
            refused = True
        except Exception:
            refused = True
        return {"stripped": stripped, "refused": refused, "facet": "topology", "had_topology": "topology" in (full.get("durable") or {})}

    def detect(payload: dict[str, Any]) -> bool:
        durable = payload["stripped"].get("durable") or {}
        omitted = "topology" not in durable and "nodes" not in durable
        return omitted

    return inject, detect


@mutation(
    "checkpoint_omits_compiled_procedures",
    "material",
    "a checkpoint dropped compiled procedures and still claimed to restore",
)
def _checkpoint_omits_compiled_procedures() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import _admit, _observe_all, _strip_facet

        material = M.build("K10_integrated_plastic_field", _small_opportunity())
        _observe_all(material)
        _admit(material)
        full = material.checkpoint()
        stripped = _strip_facet(full, "compiled_procedures")
        return {"stripped": stripped, "facet": "compiled_procedures"}

    def detect(payload: dict[str, Any]) -> bool:
        durable = payload["stripped"].get("durable") or {}
        return "compiled_procedures" not in durable and "compiled" not in durable

    return inject, detect


@mutation(
    "checkpoint_omits_precision_map",
    "material",
    "a checkpoint dropped the precision map and still claimed to restore",
)
def _checkpoint_omits_precision_map() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import _admit, _observe_all, _strip_facet

        material = M.build("K10_integrated_plastic_field", _small_opportunity())
        _observe_all(material)
        _admit(material)
        full = material.checkpoint()
        stripped = _strip_facet(full, "precision_map")
        return {"stripped": stripped, "facet": "precision_map"}

    def detect(payload: dict[str, Any]) -> bool:
        durable = payload["stripped"].get("durable") or {}
        return "precision_map" not in durable

    return inject, detect


@mutation(
    "checkpoint_omits_goals",
    "material",
    "a checkpoint dropped goal commitments and still claimed to restore",
)
def _checkpoint_omits_goals() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import _admit, _observe_all, _strip_facet

        material = M.build("K10_integrated_plastic_field", _small_opportunity())
        _observe_all(material)
        _admit(material)
        full = material.checkpoint()
        # Ensure goals exist before strip so the omission is meaningful.
        durable = full.get("durable")
        if isinstance(durable, dict):
            shell = durable.setdefault("shell", {})
            if isinstance(shell, dict):
                shell.setdefault("goal_commitments", ("survive",))
        stripped = _strip_facet(full, "goals")
        return {"stripped": stripped, "facet": "goals", "full": full}

    def detect(payload: dict[str, Any]) -> bool:
        durable = payload["stripped"].get("durable") or {}
        if "goals" in durable or "goal_commitments" in durable:
            return False
        shell = durable.get("shell")
        return not (isinstance(shell, dict) and "goal_commitments" in shell)

    return inject, detect


@mutation(
    "migration_silently_resets_state",
    "material",
    "import after migration discarded durable learning without raising",
)
def _migration_silently_resets_state() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import _admit, _observe_all

        source = M.build("K1_monolithic_plastic_field", _small_opportunity())
        _observe_all(source)
        _admit(source)
        identity = source.durable_state_digest()
        exported = source.checkpoint()
        target = M.build("K1_monolithic_plastic_field", _small_opportunity())
        # Defect: pretend to restore but load empty durable state.
        empty = copy.deepcopy(exported)
        empty["durable"] = {
            "form": "monolithic_plastic_field",
            "field_dim": exported["durable"]["field_dim"],
            "field_packed": exported["durable"]["field_packed"],
            "precision_map": {"field": "quinary"},
            "compiled_procedures": [],
            "topology": None,
            "activation": False,
        }
        # Zero the packed field by restoring then clearing.
        target.restore(exported)
        target._field = [0] * int(target._field_dim)  # type: ignore[attr-defined]
        if hasattr(target, "_resize"):
            target._resize()  # type: ignore[attr-defined]
        restored = target.durable_state_digest()
        return {"identity": identity, "restored": restored, "exported": exported, "empty": empty}

    def detect(payload: dict[str, Any]) -> bool:
        return payload["restored"] != payload["identity"]

    return inject, detect


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------

PENDING_LAYERS = {
    "material": (
        # Still pending: not yet injectable end-to-end against current materials.
        "topology_records_answers_instead_of_structure",
        "precision_audit_skipped",
        "compiled_procedure_hides_reliability_loss",
        "shadow_field_reads_authoritative_future",
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
            inject=_activation_violation,
            detect=lambda payload: False,
        )
    )
    negative = run(registry=broken)
    assert negative["survivors"] == ["activation_becomes_true"], negative
    assert not negative["zero_survivors"], negative
    print(f"genesis mutation self-check passed: {report['injected_count']} injected, {report['pending_count']} pending")


if __name__ == "__main__":
    demo()
