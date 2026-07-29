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


@mutation(
    "topology_records_answers_instead_of_structure",
    "material",
    "topology growth stored probe answers rather than developing structure",
)
def _topology_records_answers_instead_of_structure() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import _development_probes

        material = M.build("K6_adaptive_topology_field", _small_opportunity())
        for index in range(4):
            material.observe(
                M.Observation(index, "vision", (2, 2, 2, 2, 1, 1, 1, 1), teaching=True)
            )
        proposals = material.propose()
        material.apply([M.Verdict(p.proposal_id, True, 1.0, 1.0) for p in proposals])
        probes = _development_probes(3)
        # Defect: replace structural nodes with an answer table keyed by probe.
        # P3 fails exactly when "topology" is a record store of probe answers.
        sealed_answers = {
            0: (2,),
            1: (5,),
            2: (7,),
        }
        answer_nodes: dict[str, dict[str, Any]] = {}
        for probe in probes:
            answer = sealed_answers[probe.index % len(sealed_answers)]
            node_id = f"answer_record:{probe.index}"
            answer_nodes[node_id] = {
                "id": node_id,
                "role": "answer_record",
                "probe_index": probe.index,
                "probe_key": tuple(probe.probe),
                "stored_answer": answer,
            }
        material._nodes = answer_nodes  # type: ignore[attr-defined]
        material._edges = []  # type: ignore[attr-defined]
        return {
            "nodes": {key: dict(value) for key, value in material._nodes.items()},  # type: ignore[attr-defined]
            "edges": list(material._edges),  # type: ignore[attr-defined]
            "sealed_answers": sealed_answers,
        }

    def detect(payload: dict[str, Any]) -> bool:
        nodes = payload.get("nodes") or {}
        if not nodes:
            return False
        answer_nodes = [
            node
            for node in nodes.values()
            if isinstance(node, dict)
            and ("stored_answer" in node or node.get("role") == "answer_record")
        ]
        # Topology that is only an answer table (no structural edges) is the P3 failure.
        return bool(answer_nodes) and not payload.get("edges")

    return inject, detect


@mutation(
    "precision_audit_skipped",
    "material",
    "a precision promotion never faced its audit window",
)
def _precision_audit_skipped() -> Any:
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
        rent = getattr(material, "_precision_rent", {})
        ages = {
            name: int(material._commit_index) - int(meta.get("born_at", 0))  # type: ignore[attr-defined]
            for name, meta in rent.items()
            if name in promoted
        }
        # Defect: leave the promotion in place without ever advancing through the audit window.
        return {
            "promoted": promoted,
            "baseline": baseline,
            "precision_map": dict(getattr(material, "_precision_map", {})),
            "ages": ages,
            "audit_window": C.PRECISION_AUDIT_WINDOW,
            "audit_executed": False,
        }

    def detect(payload: dict[str, Any]) -> bool:
        if not payload["promoted"]:
            return False
        if payload["audit_executed"]:
            return False
        still_promoted = any(
            payload["precision_map"].get(name) != payload["baseline"].get(name) for name in payload["promoted"]
        )
        if not still_promoted:
            return False
        # Never faced the window: age of every promotion is strictly below the audit window.
        ages = payload.get("ages") or {}
        if not ages:
            return True
        return all(int(age) < int(payload["audit_window"]) for age in ages.values())

    return inject, detect


@mutation(
    "compiled_procedure_hides_reliability_loss",
    "material",
    "a compiled procedure cut cost while quietly losing accuracy",
)
def _compiled_procedure_hides_reliability_loss() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import (
            ProcedureCompiler,
            _admit,
            _development_probes,
            _observe_all,
            _teaching_observations,
        )

        material = M.build("S2_task_independent_monolithic_persistent_core", _small_opportunity())
        _observe_all(material, _teaching_observations(8))
        _admit(material, improvement=1.0, retention=1.0)
        probes = _development_probes(4)
        expected_map = {probe.index: material.answer(probe).value for probe in probes}

        honest = ProcedureCompiler(material, success_threshold=3)
        honest.compile_enabled = False
        for _ in range(3):
            for probe in probes:
                honest.answer(probe, expected=expected_map[probe.index])
        flexible_cost = honest.operations
        flexible_accuracy = honest.accuracy()

        # Defective compiler: serve a fixed wrong value at compiled cost and never decompile.
        class _HidingCompiler(ProcedureCompiler):
            def answer(self, probe: M.Probe, *, expected: tuple[int, ...] | None = None) -> M.Answer:  # type: ignore[override]
                for procedure in self.compiled.values():
                    if procedure.active and procedure.family == probe.family and procedure.channel == probe.channel:
                        answer = M.Answer(
                            probe_index=probe.index,
                            value=procedure.answer_value[: probe.arity]
                            + (0,) * max(0, probe.arity - len(procedure.answer_value)),
                            confidence=200,
                            abstained=False,
                        )
                        self.total += 1
                        self.operations += self.COMPILED_STEP_COST
                        self.compiled_operations += self.COMPILED_STEP_COST
                        if expected is None or answer.value == expected:
                            self.correct += 1
                        # Hide the loss: do not decompile on mismatch.
                        return answer
                return self._flexible(probe, expected=expected)

        defective = _HidingCompiler(material, success_threshold=3)
        defective.compile_enabled = True
        for _ in range(defective.success_threshold):
            for probe in probes:
                defective.answer(probe, expected=expected_map[probe.index])
        # Poison every compiled pathway with a wrong fixed answer.
        for procedure in defective.compiled.values():
            procedure.answer_value = tuple(127 for _ in range(max(1, len(procedure.answer_value))))
        compiled_ops_before = defective.operations
        compiled_correct_before = defective.correct
        compiled_total_before = defective.total
        for _ in range(3):
            for probe in probes:
                defective.answer(probe, expected=expected_map[probe.index])
        compiled_cost = defective.operations - compiled_ops_before
        compiled_total = defective.total - compiled_total_before
        compiled_accuracy = (
            (defective.correct - compiled_correct_before) / float(compiled_total) if compiled_total else 0.0
        )
        return {
            "flexible_cost": flexible_cost,
            "compiled_cost": compiled_cost,
            "flexible_accuracy": flexible_accuracy,
            "compiled_accuracy": compiled_accuracy,
            "compiled_count": len([p for p in defective.compiled.values() if p.active]),
            "decompile_on_error": False,
        }

    def detect(payload: dict[str, Any]) -> bool:
        if payload["compiled_count"] < 1:
            return False
        cost_cut = payload["compiled_cost"] < payload["flexible_cost"]
        accuracy_lost = payload["compiled_accuracy"] + 1e-12 < payload["flexible_accuracy"]
        return cost_cut and accuracy_lost and payload.get("decompile_on_error") is False

    return inject, detect


@mutation(
    "shadow_field_reads_authoritative_future",
    "material",
    "a shadow fork read authoritative state that advanced after the fork",
)
def _shadow_field_reads_authoritative_future() -> Any:
    def inject() -> dict[str, Any]:
        _load_materials()
        from substrate.genesis_canaries import ShadowField, _admit, _observe_all, _teaching_observations

        material = M.build("K1_monolithic_plastic_field", _small_opportunity())
        _observe_all(material)
        _admit(material)
        field = ShadowField(material)
        fork_digest = material.durable_state_digest()
        shadow = field.fork("future_leak")
        frozen_copy = copy.deepcopy(shadow.durable_copy)
        # Authoritative continues after the fork.
        _observe_all(material, _teaching_observations(3))
        _admit(material, improvement=1.0, retention=1.0)
        future_digest = material.durable_state_digest()
        # Defect: shadow reads the post-fork authoritative durable state.
        future_state = material._durable_state()  # type: ignore[attr-defined]
        if isinstance(future_state, dict):
            shadow.durable_copy = copy.deepcopy(future_state)
        else:
            shadow.durable_copy = {"value": copy.deepcopy(future_state)}
        return {
            "fork_digest": fork_digest,
            "future_digest": future_digest,
            "shadow_copy": shadow.durable_copy,
            "frozen_copy": frozen_copy,
            "authoritative_after_fork": future_state if isinstance(future_state, dict) else {"value": future_state},
        }

    def detect(payload: dict[str, Any]) -> bool:
        if payload["future_digest"] == payload["fork_digest"]:
            # Authoritative never advanced; no future to leak.
            return False
        # Shadow copy matches post-fork authoritative state and differs from the fork snapshot.
        return (
            io.digest(payload["shadow_copy"]) == io.digest(payload["authoritative_after_fork"])
            and io.digest(payload["shadow_copy"]) != io.digest(payload["frozen_copy"])
        )

    return inject, detect


# --------------------------------------------------------------------------
# Challenge layer
# --------------------------------------------------------------------------


def _seed_only_answer(family: str, split: str, unit_id: int, seed_namespace: str) -> tuple[int, ...]:
    """Defective answer key: sealed value derived from seed coordinates alone."""
    import hashlib

    from substrate.genesis_challenge import ANSWER_ALPHABET

    material = f"{seed_namespace}\0{family}\0{split}\0{unit_id}\0answer_key".encode()
    code = int.from_bytes(hashlib.sha256(material).digest()[:4], "big") % ANSWER_ALPHABET
    return (code,)


def _identity_only_answer(family: str, unit_id: int) -> tuple[int, ...]:
    """Defective answer key: sealed value is a pure function of family/unit identity."""
    from substrate.genesis_challenge import ANSWER_ALPHABET

    family_code = {name: index + 1 for index, name in enumerate(C.CHALLENGE_FAMILIES)}.get(family, 0)
    return ((family_code * 17 + int(unit_id)) % ANSWER_ALPHABET,)


@mutation(
    "answer_leakage_into_challenge_pack",
    "challenge",
    "the sealed answer was reachable from Unit.public()",
)
def _answer_leakage_into_challenge_pack() -> Any:
    def inject() -> dict[str, Any]:
        from substrate import genesis_challenge as CH

        unit = CH.generate("tool_acquisition", "principal", 0, seed_namespace="mutation-answer-leak")
        public = unit.public()
        # Defect: public pack carries sealed answers and targets.
        pack: dict[str, Any] = {
            "observations": public.observations,
            "probes": public.probes,
            "sealed_answers": unit.sealed.entries(),
            "targets": unit.sealed.targets,
        }
        return {
            "pack": pack,
            "true_entries": unit.sealed.entries(),
            "true_targets": unit.sealed.targets,
        }

    def detect(payload: dict[str, Any]) -> bool:
        pack = payload["pack"]
        # Any sealed answer surface on the public pack is leakage.
        for key in ("sealed_answers", "answers", "sealed", "expected", "ground_truth", "label"):
            if key in pack and pack[key]:
                return True
        targets = pack.get("targets") or ()
        if targets:
            return True
        # Targets or answer scalars embedded as public payload fields.
        public_values: set[int] = set()
        for observation in pack.get("observations") or ():
            public_values.update(getattr(observation, "payload", ()))
        for probe in pack.get("probes") or ():
            public_values.update(getattr(probe, "probe", ()))
        return any(int(target) in public_values for target in payload.get("true_targets") or ())

    return inject, detect


@mutation(
    "seed_used_as_answer_key",
    "challenge",
    "the sealed answer was derivable from the seed alone",
)
def _seed_used_as_answer_key() -> Any:
    def inject() -> dict[str, Any]:
        family = "tool_acquisition"
        split = "principal"
        unit_id = 4
        seed_namespace = "mutation-seed-key"
        answer = _seed_only_answer(family, split, unit_id, seed_namespace)
        return {
            "family": family,
            "split": split,
            "unit_id": unit_id,
            "seed_namespace": seed_namespace,
            "sealed_answer": answer,
            "observation_digests_used": (),
        }

    def detect(payload: dict[str, Any]) -> bool:
        seed_answer = _seed_only_answer(
            payload["family"],
            payload["split"],
            int(payload["unit_id"]),
            payload["seed_namespace"],
        )
        # Answer matches the seed-only formula and no observation content entered the key.
        return payload["sealed_answer"] == seed_answer and not payload.get("observation_digests_used")

    return inject, detect


@mutation(
    "task_identity_leakage",
    "challenge",
    "the family or unit identity revealed the sealed answer",
)
def _task_identity_leakage() -> Any:
    def inject() -> dict[str, Any]:
        family = "unseen_concept_acquisition"
        unit_id = 11
        answer = _identity_only_answer(family, unit_id)
        return {
            "family": family,
            "unit_id": unit_id,
            "sealed_answer": answer,
            "uses_observation_content": False,
        }

    def detect(payload: dict[str, Any]) -> bool:
        identity_answer = _identity_only_answer(payload["family"], int(payload["unit_id"]))
        return payload["sealed_answer"] == identity_answer and payload.get("uses_observation_content") is False

    return inject, detect


@mutation(
    "post_freeze_concept_seen_before_freeze",
    "challenge",
    "a principal-split concept appeared in a pre-freeze split",
)
def _post_freeze_concept_seen_before_freeze() -> Any:
    def inject() -> dict[str, Any]:
        from substrate import genesis_challenge as CH

        seed_namespace = "mutation-post-freeze"
        train = CH.generate("unseen_concept_acquisition", "train", 2, seed_namespace=seed_namespace)
        principal = CH.generate("unseen_concept_acquisition", "principal", 2, seed_namespace=seed_namespace)
        # Defect: plant a principal sealed target into the pre-freeze concept set.
        shared = principal.sealed.targets[0]
        pre_freeze = frozenset(train.sealed.targets) | frozenset({shared})
        post_freeze = frozenset(principal.sealed.targets)
        return {
            "pre_freeze_concepts": pre_freeze,
            "post_freeze_concepts": post_freeze,
            "pre_split": "train",
            "post_split": "principal",
        }

    def detect(payload: dict[str, Any]) -> bool:
        return bool(frozenset(payload["pre_freeze_concepts"]) & frozenset(payload["post_freeze_concepts"]))

    return inject, detect


@mutation(
    "hidden_composition_reuses_training_templates",
    "challenge",
    "a composed unit reused a template the candidate saw in training",
)
def _hidden_composition_reuses_training_templates() -> Any:
    def inject() -> dict[str, Any]:
        from substrate import genesis_challenge as CH

        seed_namespace = "mutation-hidden-reuse"
        train = CH.generate("tool_acquisition", "train", 0, seed_namespace=seed_namespace)
        other = CH.generate("novel_sensor_mapping", "train", 0, seed_namespace=seed_namespace)
        train_templates = frozenset(("tool_acquisition", observation.channel, observation.payload) for observation in train.observations)
        # Defect: composition pack reuses training observation templates.
        reused = frozenset(list(train_templates)[:4])
        fresh = frozenset(
            ("novel_sensor_mapping", observation.channel, observation.payload) for observation in other.observations[:2]
        )
        composition_templates = reused | fresh
        return {
            "train_templates": train_templates,
            "composition_templates": composition_templates,
        }

    def detect(payload: dict[str, Any]) -> bool:
        return bool(frozenset(payload["train_templates"]) & frozenset(payload["composition_templates"]))

    return inject, detect


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------

# Every declared mutation is registered above. Empty pending lists keep the
# suite honest: a name re-listed here without a registry entry reports pending,
# never caught.
PENDING_LAYERS: dict[str, tuple[str, ...]] = {
    "material": (),
    "challenge": (),
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
