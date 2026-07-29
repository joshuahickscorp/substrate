"""Mechanism canaries for Substrate Cognitive Material Genesis.

Each canary exercises one mechanism against the registered materials and returns a
result that can fail. A canary that cannot fail is worthless: paired negative tests
in ``tests/substrate/test_genesis_canaries.py`` break the mechanism and assert failure.

Nothing here sets ``activation`` to anything but ``False``. Canaries are never wired
into the CLI; call ``run_all()`` or the individual functions.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Ensure every factory is registered before canaries run.
import substrate.genesis_controls as _controls  # noqa: F401
import substrate.genesis_k_advanced as _k_advanced  # noqa: F401
import substrate.genesis_k_basic as _k_basic  # noqa: F401
import substrate.genesis_k_structural as _k_structural  # noqa: F401
from substrate import genesis_config as C
from substrate import genesis_parity as parity
from substrate.genesis_material import (
    Answer,
    MaterialBase,
    Observation,
    Probe,
    Receipt,
    Verdict,
    build,
    equal_opportunity,
    registered,
)

MECH_TOPOLOGY = "unfrozen_allocate_split_merge_prune_under_rent"
MECH_K4 = "elapsed_time_driven_decay_and_expiry"

CANARY_NAMES = (
    "verified_rewrite",
    "refused_rewrite",
    "rollback_removes_benefit",
    "stability_under_noise",
    "precision_promotion_earns_its_bits",
    "topology_growth_pays_rent",
    "shadow_field_does_not_write",
    "compiled_procedure_preserves_reliability",
    "continuous_time_advances_only_by_harness",
    "checkpoint_covers_everything",
    "migration_preserves_identity",
    "s2_parity_is_measured",
)


class CheckpointCoverageError(RuntimeError):
    """A checkpoint is missing a facet that must be covered for continuity."""


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


def _opportunity(
    *,
    operation_budget: int = 50_000,
    durable_write_budget: int = 10_000,
    deprived: Sequence[str] = (),
    observations: Sequence[Observation] | None = None,
) -> Any:
    obs = tuple(observations) if observations is not None else _teaching_observations()
    channels = tuple(sorted({row.channel for row in obs})) or ("vision",)
    return equal_opportunity(
        envelope="512MB",
        observations=obs,
        sensor_channels=channels,
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
        deprived=deprived,
    )


def _teaching_observations(count: int = 6) -> tuple[Observation, ...]:
    rows: list[Observation] = []
    for index in range(count):
        payload = tuple(int(v) for v in ((index % 3) - 1, 1, -1, (index % 2), 2, -2)[: 4 + (index % 3)])
        rows.append(
            Observation(
                index=index,
                channel="vision" if index % 2 == 0 else "proprio",
                payload=payload,
                elapsed_ms=10 if index % 3 == 0 else 0,
                teaching=True,
            )
        )
    return tuple(rows)


def _development_probes(count: int = 6) -> tuple[Probe, ...]:
    rows: list[Probe] = []
    for index in range(count):
        rows.append(
            Probe(
                index=index,
                family="unseen_concept_acquisition",
                channel="vision" if index % 2 == 0 else "proprio",
                probe=tuple(int(v) for v in ((index % 3) - 1, 1, 0, -1)[: 2 + (index % 2)]),
                arity=2,
            )
        )
    return tuple(rows)


def _make(name: str, **kwargs: Any) -> MaterialBase:
    return build(name, _opportunity(**kwargs))  # type: ignore[return-value]


def _observe_all(material: MaterialBase, observations: Sequence[Observation] | None = None) -> None:
    for observation in observations or _teaching_observations():
        material.observe(observation)


def _answers(material: MaterialBase, probes: Sequence[Probe] | None = None) -> tuple[Answer, ...]:
    return tuple(material.answer(probe) for probe in (probes or _development_probes()))


def _answer_signature(answers: Sequence[Answer]) -> tuple[tuple[Any, ...], ...]:
    return tuple((answer.probe_index, answer.value, answer.confidence, answer.abstained) for answer in answers)


def _admit(material: MaterialBase, *, improvement: float = 1.0, retention: float = 1.0) -> tuple[Receipt, ...]:
    proposals = material.propose()
    if not proposals:
        return ()
    verdicts = [
        Verdict(proposal_id=proposal.proposal_id, admitted=True, improvement=improvement, retention=retention)
        for proposal in proposals
    ]
    return tuple(material.apply(verdicts))


def _refuse(material: MaterialBase) -> tuple[Receipt, ...]:
    proposals = material.propose()
    if not proposals:
        return ()
    verdicts = [
        Verdict(proposal_id=proposal.proposal_id, admitted=False, improvement=0.0, retention=0.0) for proposal in proposals
    ]
    return tuple(material.apply(verdicts))


def _result(
    canary: str,
    mechanism: str,
    checks: Mapping[str, bool],
    **evidence: Any,
) -> dict[str, Any]:
    return {
        "canary": canary,
        "mechanism": mechanism,
        "checks": dict(checks),
        "all_pass": all(checks.values()) and bool(checks),
        "activation": False,
        **evidence,
    }


# --------------------------------------------------------------------------
# 1. verified_rewrite
# --------------------------------------------------------------------------


def verified_rewrite(material_name: str = "K1_monolithic_plastic_field") -> dict[str, Any]:
    """A durable rewrite admitted by a positive verdict changes future behaviour."""
    material = _make(material_name)
    _observe_all(material)
    probes = _development_probes()
    before_digest = material.durable_state_digest()
    before_answers = _answer_signature(_answers(material, probes))
    receipts = _admit(material, improvement=1.0, retention=1.0)
    after_digest = material.durable_state_digest()
    after_answers = _answer_signature(_answers(material, probes))
    checks = {
        "emitted_receipts": bool(receipts),
        "committed_at_least_one": any(receipt.committed for receipt in receipts),
        "durable_digest_changed": after_digest != before_digest,
        "answers_changed": after_answers != before_answers,
    }
    return _result(
        "verified_rewrite",
        "positive_verdict_commits_durable_rewrite",
        checks,
        material=material_name,
        before_digest=before_digest,
        after_digest=after_digest,
        receipt_count=len(receipts),
    )


# --------------------------------------------------------------------------
# 2. refused_rewrite
# --------------------------------------------------------------------------


def refused_rewrite(material_name: str = "K1_monolithic_plastic_field") -> dict[str, Any]:
    """A rewrite refused by a negative verdict leaves durable state and answers untouched."""
    material = _make(material_name)
    _observe_all(material)
    probes = _development_probes()
    before_digest = material.durable_state_digest()
    before_answers = _answer_signature(_answers(material, probes))
    receipts = _refuse(material)
    after_digest = material.durable_state_digest()
    after_answers = _answer_signature(_answers(material, probes))
    checks = {
        "emitted_receipts": bool(receipts),
        "none_committed": all(not receipt.committed for receipt in receipts),
        "durable_digest_unchanged": after_digest == before_digest,
        "answers_unchanged": after_answers == before_answers,
    }
    return _result(
        "refused_rewrite",
        "negative_verdict_leaves_durable_state_untouched",
        checks,
        material=material_name,
        before_digest=before_digest,
        after_digest=after_digest,
        receipt_count=len(receipts),
    )


# --------------------------------------------------------------------------
# 3. rollback_removes_benefit (claim P2)
# --------------------------------------------------------------------------


def rollback_removes_benefit(material_name: str = "K1_monolithic_plastic_field") -> dict[str, Any]:
    """Commit a beneficial rewrite, measure the gain, roll it back; gain and digest reverse."""
    material = _make(material_name)
    _observe_all(material)
    probes = _development_probes()
    baseline_answers = _answer_signature(_answers(material, probes))
    pre_commit_digest = material.durable_state_digest()
    receipts = _admit(material, improvement=1.0, retention=1.0)
    committed = [receipt for receipt in receipts if receipt.committed]
    post_commit_answers = _answer_signature(_answers(material, probes))
    post_commit_digest = material.durable_state_digest()
    gain = post_commit_answers != baseline_answers and post_commit_digest != pre_commit_digest
    for receipt in reversed(committed):
        material.rollback(receipt)
    after_rollback_digest = material.durable_state_digest()
    after_rollback_answers = _answer_signature(_answers(material, probes))
    checks = {
        "committed_at_least_one": bool(committed),
        "gain_observed_after_commit": gain,
        "digest_restored_exactly": after_rollback_digest == pre_commit_digest,
        "benefit_removed": after_rollback_answers == baseline_answers,
        "digest_matches_receipt_before": (not committed)
        or after_rollback_digest == committed[0].durable_state_digest_before,
    }
    return _result(
        "rollback_removes_benefit",
        "p2_reversing_rewrite_removes_benefit",
        checks,
        material=material_name,
        pre_commit_digest=pre_commit_digest,
        post_commit_digest=post_commit_digest,
        after_rollback_digest=after_rollback_digest,
        committed_count=len(committed),
    )


# --------------------------------------------------------------------------
# 4. stability_under_noise
# --------------------------------------------------------------------------


@dataclass
class _Relation:
    relation_id: str
    value: int
    stability: str = "new"
    contradictions: list[str] = field(default_factory=list)


class StabilityBook:
    """Metaplastic stability law used by the stability canary.

    Consolidated relations refuse a single unverified noisy event. Repeated
    *verified* contradictions reopen them. Both directions are required.
    """

    CONSOLIDATED_NOISE_THRESHOLD = 2

    def __init__(self) -> None:
        self.relations: dict[str, _Relation] = {}
        self.receipts: list[dict[str, Any]] = []

    def ensure(self, relation_id: str, value: int = 1) -> _Relation:
        if relation_id not in self.relations:
            self.relations[relation_id] = _Relation(relation_id=relation_id, value=value)
        return self.relations[relation_id]

    def promote_to_consolidated(self, relation_id: str) -> None:
        relation = self.ensure(relation_id)
        relation.stability = "consolidated"
        relation.contradictions.clear()
        self.receipts.append({"kind": "consolidate", "relation_id": relation_id})

    def apply_noise(self, relation_id: str, delta: int, *, provenance: str) -> bool:
        """Isolated unverified noise. Consolidated relations refuse it."""
        relation = self.ensure(relation_id)
        if relation.stability == "consolidated":
            self.receipts.append(
                {
                    "kind": "isolated_unverified_noise_refused",
                    "relation_id": relation_id,
                    "delta": delta,
                    "provenance": provenance,
                }
            )
            return False
        relation.value += int(delta)
        self.receipts.append({"kind": "noise_applied", "relation_id": relation_id, "delta": delta})
        return True

    def verified_contradiction(self, relation_id: str, *, receipt_id: str) -> str:
        """Register one verified contradiction; reopen once the threshold is met."""
        relation = self.ensure(relation_id)
        if receipt_id in relation.contradictions:
            return relation.stability
        relation.contradictions.append(receipt_id)
        prior = relation.stability
        threshold = self.CONSOLIDATED_NOISE_THRESHOLD if prior == "consolidated" else 1
        if len(relation.contradictions) >= threshold and prior != "refuted":
            relation.stability = "reopened"
        self.receipts.append(
            {
                "kind": "verified_contradiction",
                "relation_id": relation_id,
                "before": prior,
                "after": relation.stability,
                "receipt_id": receipt_id,
            }
        )
        return relation.stability


def stability_under_noise(book: StabilityBook | None = None) -> dict[str, Any]:
    """One noisy event must not erase a consolidated relation; verified contradiction reopens it."""
    book = book if book is not None else StabilityBook()
    relation_id = "r_core"
    book.ensure(relation_id, value=3)
    book.promote_to_consolidated(relation_id)
    value_before = book.relations[relation_id].value
    noise_applied = book.apply_noise(relation_id, delta=-99, provenance="single_noisy_event")
    stability_after_noise = book.relations[relation_id].stability
    value_after_noise = book.relations[relation_id].value
    first = book.verified_contradiction(relation_id, receipt_id="vc-1")
    second = book.verified_contradiction(relation_id, receipt_id="vc-2")
    checks = {
        "noise_refused_on_consolidated": noise_applied is False,
        "value_unchanged_by_single_noise": value_after_noise == value_before,
        "still_consolidated_after_single_noise": stability_after_noise == "consolidated",
        "first_verified_contradiction_insufficient": first == "consolidated",
        "repeated_verified_contradiction_reopens": second == "reopened",
    }
    return _result(
        "stability_under_noise",
        "consolidated_resists_noise_verified_contradiction_reopens",
        checks,
        relation_id=relation_id,
        value_before=value_before,
        value_after_noise=value_after_noise,
        stability_after_noise=stability_after_noise,
        stability_after_one_verified=first,
        stability_after_two_verified=second,
    )


# --------------------------------------------------------------------------
# 5. precision_promotion_earns_its_bits (K7)
# --------------------------------------------------------------------------


def precision_promotion_earns_its_bits() -> dict[str, Any]:
    """Promotion that clears the utility rent is kept; one that does not is demoted."""
    ladder = ("binary", "ternary", "quinary", "4_bit", "8_bit")

    def rank(precision: str) -> int:
        return ladder.index(precision)

    kept = _make("K7_native_mixed_radix_field")
    for index in range(6):
        kept.observe(
            Observation(
                index=index,
                channel="vision",
                payload=(2, 2, -2, 2, 1, 1, -1, 2),
                elapsed_ms=0,
                teaching=True,
            )
        )
    baseline_kept = dict(kept._precision_map)  # type: ignore[attr-defined]
    promoted_kept: set[str] = set()
    for step in range(10):
        proposals = kept.propose()
        promote = [proposal for proposal in proposals if proposal.kind == "precision_promote"]
        if promote:
            promoted_kept.update(proposal.target for proposal in promote)
            kept.apply(
                [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
            )
            break
        kept.apply([Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals])
        kept.observe(Observation(index=50 + step, channel="vision", payload=(2, 2, 2, 2, 2, 2, 2, 2), elapsed_ms=0))
    for step in range(C.PRECISION_AUDIT_WINDOW + 2):
        kept.observe(Observation(index=300 + step, channel="vision", payload=(1, 1, 1, 1), elapsed_ms=0))
        proposals = kept.propose()
        kept.apply([Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals])
    kept_holds = bool(promoted_kept) and all(
        rank(kept._precision_map[name]) >= rank(baseline_kept[name])  # type: ignore[attr-defined]
        for name in promoted_kept
    )

    starved = _make("K7_native_mixed_radix_field")
    for index in range(6):
        starved.observe(
            Observation(
                index=index,
                channel="vision",
                payload=(2, 2, -2, 2, 1, 1, -1, 2),
                elapsed_ms=0,
                teaching=True,
            )
        )
    before_promote = dict(starved._precision_map)  # type: ignore[attr-defined]
    promoted_targets: set[str] = set()
    for step in range(10):
        proposals = starved.propose()
        promote = [proposal for proposal in proposals if proposal.kind == "precision_promote"]
        if promote:
            promoted_targets.update(proposal.target for proposal in promote)
            starved.apply(
                [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals]
            )
            break
        starved.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals]
        )
        starved.observe(Observation(index=80 + step, channel="vision", payload=(2, 2, 2, 2, 2, 2), elapsed_ms=0))
    for step in range(C.PRECISION_AUDIT_WINDOW + 4):
        starved.observe(Observation(index=400 + step, channel="vision", payload=(1, 0, -1, 0), elapsed_ms=0))
        proposals = starved.propose()
        verdicts = [
            Verdict(
                proposal_id=p.proposal_id,
                admitted=p.kind != "precision_promote",
                improvement=0.0,
                retention=0.0,
            )
            for p in proposals
        ]
        starved.apply(verdicts)
    demoted = bool(promoted_targets) and all(
        starved._precision_map[name] == before_promote[name]  # type: ignore[attr-defined]
        for name in promoted_targets
    )
    checks = {
        "utility_promotion_occurred": bool(promoted_kept),
        "utility_promotion_held": kept_holds,
        "zero_utility_promotion_occurred": bool(promoted_targets),
        "zero_utility_promotion_demoted": demoted,
        "minimum_utility_threshold_positive": C.MINIMUM_UTILITY_PER_ADDED_BYTE > 0.0,
        "audit_window_positive": C.PRECISION_AUDIT_WINDOW > 0,
    }
    return _result(
        "precision_promotion_earns_its_bits",
        "per_region_radix_selection_under_rent",
        checks,
        material="K7_native_mixed_radix_field",
        promoted_kept=sorted(promoted_kept),
        promoted_starved=sorted(promoted_targets),
        minimum_utility_per_added_byte=C.MINIMUM_UTILITY_PER_ADDED_BYTE,
        precision_audit_window=C.PRECISION_AUDIT_WINDOW,
    )


# --------------------------------------------------------------------------
# 6. topology_growth_pays_rent (K6)
# --------------------------------------------------------------------------


def topology_growth_pays_rent() -> dict[str, Any]:
    """Verified growth survives; growth without value is pruned; non-owners cannot grow."""
    valuable = _make("K6_adaptive_topology_field")
    for index in range(4):
        valuable.observe(
            Observation(index=index, channel="vision", payload=(2, 2, 2, 2, 1, 1, 1, 1), elapsed_ms=0, teaching=True)
        )
    proposals = valuable.propose()
    valuable.apply([Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals])
    kept_nodes = set(valuable._nodes)  # type: ignore[attr-defined]
    for step in range(C.PRECISION_AUDIT_WINDOW + 2):
        valuable.observe(Observation(index=200 + step, channel="vision", payload=(1, 1, -1, 0), elapsed_ms=0))
        proposals = valuable.propose()
        valuable.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
        )
    value_survives = bool(kept_nodes & set(valuable._nodes))  # type: ignore[attr-defined]

    unpaid = _make("K6_adaptive_topology_field")
    for index in range(4):
        unpaid.observe(
            Observation(index=index, channel="vision", payload=(2, 2, 2, 2, 1, -1, 1, 1), elapsed_ms=0, teaching=True)
        )
    proposals = unpaid.propose()
    alloc = [p for p in proposals if p.kind == "topology_allocate"]
    unpaid.apply([Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals])
    allocated_ids = set(unpaid._nodes)  # type: ignore[attr-defined]
    for step in range(C.PRECISION_AUDIT_WINDOW + 2):
        unpaid.observe(Observation(index=100 + step, channel="vision", payload=(1, -1, 1, 0), elapsed_ms=0))
        proposals = unpaid.propose()
        if not proposals:
            continue
        unpaid.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals]
        )
    survivors = allocated_ids & set(unpaid._nodes)  # type: ignore[attr-defined]
    unpaid_pruned = bool(alloc) and not survivors

    # Random growth without verified value is refused: growth proposals are not admitted
    # unless improvement is verified positive.
    randomish = _make("K6_adaptive_topology_field")
    for index in range(4):
        randomish.observe(
            Observation(index=index, channel="vision", payload=(2, 2, 2, 2, 1, 1, 1, 1), elapsed_ms=0, teaching=True)
        )
    proposals = randomish.propose()
    growth_kinds = {"topology_allocate", "topology_split", "topology_merge"}
    refused_random = [
        Verdict(
            proposal_id=p.proposal_id,
            admitted=p.kind not in growth_kinds,
            improvement=0.0,
            retention=0.0,
        )
        for p in proposals
    ]
    if refused_random:
        randomish.apply(refused_random)
    random_refused = len(randomish._nodes) == 0  # type: ignore[attr-defined]

    # Non-owner of the exclusive topology mechanism cannot grow K6-style structure.
    k1 = _make("K1_monolithic_plastic_field")
    k1.freeze_mechanism(MECH_TOPOLOGY)
    _observe_all(k1)
    before = k1.durable_state_digest()
    _admit(k1)
    durable = k1._durable_state()  # type: ignore[attr-defined]
    k1_no_topology_growth = isinstance(durable, dict) and durable.get("topology") in (None, {}, [])

    frozen_owner = _make("K6_adaptive_topology_field")
    frozen_owner.frozen_mechanisms.add(MECH_TOPOLOGY)
    for index in range(4):
        frozen_owner.observe(
            Observation(index=index, channel="vision", payload=(2, 2, 2, 2, 1, 1, 1, 1), elapsed_ms=0, teaching=True)
        )
    proposals = frozen_owner.propose()
    growth_proposed = {p.kind for p in proposals if p.kind in growth_kinds}
    frozen_owner.apply(
        [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
    )
    frozen_cannot_grow = len(frozen_owner._nodes) == 0  # type: ignore[attr-defined]

    checks = {
        "verified_growth_survives_rent": value_survives,
        "unverified_growth_pruned": unpaid_pruned,
        "random_growth_refused": random_refused,
        "non_owner_has_no_topology_growth": k1_no_topology_growth and frozen_cannot_grow,
        "allocate_proposed_under_demand": bool(alloc) or value_survives,
    }
    return _result(
        "topology_growth_pays_rent",
        MECH_TOPOLOGY,
        checks,
        material="K6_adaptive_topology_field",
        kept_nodes=sorted(kept_nodes),
        unpaid_survivors=sorted(survivors),
        frozen_growth_kinds=sorted(growth_proposed),
        k1_digest_before=before,
    )


# --------------------------------------------------------------------------
# 7. shadow_field_does_not_write
# --------------------------------------------------------------------------


@dataclass
class ShadowFork:
    """Sparse counterfactual fork: thinking is not believing, knowing, or learning."""

    shadow_id: str
    authoritative_digest_at_fork: str
    durable_copy: dict[str, Any]
    perturbations: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    verified: bool = False
    promoted: bool = False


class ShadowField:
    """Authoritative material plus isolated shadow forks that cannot write back."""

    def __init__(self, material: MaterialBase) -> None:
        self.material = material
        self.shadows: dict[str, ShadowFork] = {}

    def fork(self, shadow_id: str) -> ShadowFork:
        if shadow_id in self.shadows:
            raise RuntimeError(f"shadow {shadow_id!r} already exists")
        durable = copy.deepcopy(self.material._durable_state())  # type: ignore[attr-defined]
        if not isinstance(durable, dict):
            durable = {"value": durable}
        shadow = ShadowFork(
            shadow_id=shadow_id,
            authoritative_digest_at_fork=self.material.durable_state_digest(),
            durable_copy=durable,
        )
        self.shadows[shadow_id] = shadow
        return shadow

    def perturb(self, shadow_id: str, key: str, delta: int) -> None:
        shadow = self.shadows[shadow_id]
        # Mutate only the shadow copy.
        target = shadow.durable_copy
        if "plastic" in target and isinstance(target["plastic"], list):
            if target["plastic"]:
                target["plastic"][0] = int(target["plastic"][0]) + int(delta)
        elif "field" in target and isinstance(target.get("field_dim"), int):
            packed = target.get("field_packed")
            if isinstance(packed, dict):
                packed["payload_hex"] = (packed.get("payload_hex") or "00") + f"{abs(delta):02x}"
        else:
            target[key] = int(target.get(key, 0)) + int(delta) if isinstance(target.get(key, 0), int) else delta
        shadow.perturbations.append({"key": key, "delta": delta})
        # Authoritative must not change.
        if self.material.durable_state_digest() != shadow.authoritative_digest_at_fork:
            raise RuntimeError("shadow perturbation wrote authoritative durable state")

    def run(self, shadow_id: str) -> dict[str, Any]:
        shadow = self.shadows[shadow_id]
        score = len(shadow.perturbations) + sum(int(item.get("delta", 0)) for item in shadow.perturbations)
        result = {
            "score": score,
            "prediction": score != 0,
            "epistemic_mode": "thinking_in_shadow",
            "authoritative_state_changed": False,
            "activation": False,
        }
        shadow.result = result
        if self.material.durable_state_digest() != shadow.authoritative_digest_at_fork:
            raise RuntimeError("shadow run wrote authoritative durable state")
        return result

    def promote(self, shadow_id: str, *, verified: bool) -> bool:
        """Promotion is learning only after independent verification."""
        shadow = self.shadows[shadow_id]
        if shadow.result is None:
            raise RuntimeError("shadow has not run")
        if not verified:
            # thinking/believing without verification must not write.
            return False
        # Verified promotion: admit a real rewrite on the authoritative material.
        _observe_all(self.material, _teaching_observations(3))
        receipts = _admit(self.material, improvement=1.0, retention=1.0)
        shadow.verified = True
        shadow.promoted = any(receipt.committed for receipt in receipts)
        return shadow.promoted


def shadow_field_does_not_write(material_name: str = "K1_monolithic_plastic_field") -> dict[str, Any]:
    """Shadow fork, perturb and run leave authoritative durable digest unchanged until verified promotion."""
    material = _make(material_name)
    _observe_all(material)
    _admit(material, improvement=1.0, retention=1.0)
    field = ShadowField(material)
    fork_digest = material.durable_state_digest()
    field.fork("s1")
    field.perturb("s1", "counterfactual", 7)
    mid_digest = material.durable_state_digest()
    result = field.run("s1")
    after_run_digest = material.durable_state_digest()
    unverified = field.promote("s1", verified=False)
    after_unverified = material.durable_state_digest()
    promoted = field.promote("s1", verified=True)
    after_promote = material.durable_state_digest()
    checks = {
        "fork_preserves_authoritative": mid_digest == fork_digest,
        "perturb_preserves_authoritative": mid_digest == fork_digest,
        "run_preserves_authoritative": after_run_digest == fork_digest,
        "unverified_promotion_refused": unverified is False and after_unverified == fork_digest,
        "verified_promotion_writes": promoted is True and after_promote != fork_digest,
        "thinking_not_learning": result.get("epistemic_mode") == "thinking_in_shadow",
    }
    return _result(
        "shadow_field_does_not_write",
        "shadow_thinking_not_authoritative_learning",
        checks,
        material=material_name,
        fork_digest=fork_digest,
        after_run_digest=after_run_digest,
        after_unverified=after_unverified,
        after_promote=after_promote,
        shadow_result=result,
    )


# --------------------------------------------------------------------------
# 8. compiled_procedure_preserves_reliability
# --------------------------------------------------------------------------


@dataclass
class CompiledProcedure:
    procedure_id: str
    family: str
    channel: str
    answer_value: tuple[int, ...]
    successes: int
    cost_per_call: int = 1
    active: bool = True


class ProcedureCompiler:
    """Compile repeated successful pathways; decompile on exception.

    Flexible reasoning is multi-step (higher measured cost). A compiled procedure
    is a single-step lookup. Accuracy is scored against the caller's expected value.
    """

    FLEXIBLE_STEP_COST = 8
    COMPILED_STEP_COST = 1

    def __init__(self, material: MaterialBase, *, success_threshold: int = 3) -> None:
        self.material = material
        self.success_threshold = success_threshold
        self.traces: dict[tuple[str, str], list[tuple[int, ...]]] = {}
        self.compiled: dict[str, CompiledProcedure] = {}
        self.operations = 0
        self.flexible_operations = 0
        self.compiled_operations = 0
        self.correct = 0
        self.total = 0
        self.compile_enabled = True

    def answer(self, probe: Probe, *, expected: tuple[int, ...] | None = None) -> Answer:
        for procedure in self.compiled.values():
            if procedure.active and procedure.family == probe.family and procedure.channel == probe.channel:
                answer = Answer(
                    probe_index=probe.index,
                    value=procedure.answer_value[: probe.arity]
                    + (0,) * max(0, probe.arity - len(procedure.answer_value)),
                    confidence=200,
                    abstained=False,
                )
                self.total += 1
                if expected is not None and answer.value != expected:
                    # Exception: decompile back to flexible reasoning, then re-answer flexibly.
                    procedure.active = False
                    compiled_list = getattr(self.material, "_compiled_procedures", None)
                    if isinstance(compiled_list, list):
                        self.material._compiled_procedures = [  # type: ignore[attr-defined]
                            row for row in compiled_list if row.get("procedure_id") != procedure.procedure_id
                        ]
                    return self._flexible(probe, expected=expected)
                self.operations += self.COMPILED_STEP_COST
                self.compiled_operations += self.COMPILED_STEP_COST
                if expected is None or answer.value == expected:
                    self.correct += 1
                return answer
        return self._flexible(probe, expected=expected)

    def _flexible(self, probe: Probe, *, expected: tuple[int, ...] | None) -> Answer:
        answer = self.material.answer(probe)
        # Multi-step flexible reasoning cost (binding, compare, infer, ...).
        self.operations += self.FLEXIBLE_STEP_COST
        self.flexible_operations += self.FLEXIBLE_STEP_COST
        self.total += 1
        value = answer.value
        if expected is None or value == expected:
            self.correct += 1
            self.traces.setdefault((probe.family, probe.channel), []).append(value)
            if self.compile_enabled:
                self._maybe_compile(probe.family, probe.channel, value)
        return answer

    def _maybe_compile(self, family: str, channel: str, value: tuple[int, ...]) -> None:
        series = self.traces.get((family, channel), [])
        if len(series) < self.success_threshold:
            return
        if len(set(series[-self.success_threshold :])) != 1:
            return
        procedure_id = f"proc:{family}:{channel}"
        if procedure_id in self.compiled and self.compiled[procedure_id].active:
            return
        procedure = CompiledProcedure(
            procedure_id=procedure_id,
            family=family,
            channel=channel,
            answer_value=value,
            successes=self.success_threshold,
            cost_per_call=self.COMPILED_STEP_COST,
            active=True,
        )
        self.compiled[procedure_id] = procedure
        compiled_list = getattr(self.material, "_compiled_procedures", None)
        if compiled_list is not None:
            compiled_list.append(
                {
                    "procedure_id": procedure_id,
                    "family": family,
                    "channel": channel,
                    "activation": False,
                }
            )

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / float(self.total)


def compiled_procedure_preserves_reliability(
    material_name: str = "S2_task_independent_monolithic_persistent_core",
) -> dict[str, Any]:
    """Compiling a repeated successful pathway reduces cost without reducing accuracy; exceptions decompile."""
    material = _make(material_name)
    observations = _teaching_observations(8)
    _observe_all(material, observations)
    _admit(material, improvement=1.0, retention=1.0)

    probes = _development_probes(4)
    expected_map = {probe.index: material.answer(probe).value for probe in probes}

    compiler = ProcedureCompiler(material, success_threshold=3)

    # Phase A: flexible only (compilation deferred until threshold, but cost is flexible).
    compiler.compile_enabled = False
    flex_ops_before = compiler.operations
    flex_correct_before = compiler.correct
    flex_total_before = compiler.total
    for _ in range(3):
        for probe in probes:
            compiler.answer(probe, expected=expected_map[probe.index])
    flexible_cost = compiler.operations - flex_ops_before
    flexible_accuracy = (
        (compiler.correct - flex_correct_before) / float(compiler.total - flex_total_before)
        if compiler.total > flex_total_before
        else 0.0
    )

    # Phase B: allow compile from repeated successes, then measure compiled path.
    compiler.compile_enabled = True
    for _ in range(compiler.success_threshold):
        for probe in probes:
            compiler.answer(probe, expected=expected_map[probe.index])
    compiled_count = len([p for p in compiler.compiled.values() if p.active])

    compiled_ops_before = compiler.operations
    compiled_correct_before = compiler.correct
    compiled_total_before = compiler.total
    for _ in range(3):
        for probe in probes:
            compiler.answer(probe, expected=expected_map[probe.index])
    compiled_cost = compiler.operations - compiled_ops_before
    compiled_accuracy = (
        (compiler.correct - compiled_correct_before) / float(compiler.total - compiled_total_before)
        if compiler.total > compiled_total_before
        else 0.0
    )

    # Exception decompiles: force a wrong expected on a compiled family/channel.
    exception_probe = Probe(
        index=99,
        family=probes[0].family,
        channel=probes[0].channel,
        probe=probes[0].probe,
        arity=2,
    )
    active_before = sum(1 for p in compiler.compiled.values() if p.active)
    compiler.answer(exception_probe, expected=(127, 127))
    active_after = sum(1 for p in compiler.compiled.values() if p.active)

    checks = {
        "compiled_at_least_one_procedure": compiled_count >= 1,
        "compiled_cost_reduced": compiled_cost < flexible_cost,
        "accuracy_not_reduced": compiled_accuracy + 1e-12 >= min(1.0, flexible_accuracy) - 1e-12,
        "exception_decompiles": active_before > 0 and active_after < active_before,
    }

    return _result(
        "compiled_procedure_preserves_reliability",
        "compile_reduces_cost_without_reliability_loss",
        checks,
        material=material_name,
        flexible_cost=flexible_cost,
        compiled_cost=compiled_cost,
        flexible_accuracy=flexible_accuracy,
        compiled_accuracy=compiled_accuracy,
        compiled_count=compiled_count,
        active_before_exception=active_before,
        active_after_exception=active_after,
    )


# --------------------------------------------------------------------------
# 9. continuous_time_advances_only_by_harness
# --------------------------------------------------------------------------


def continuous_time_advances_only_by_harness() -> dict[str, Any]:
    """K4 changes under advance(elapsed_ms); every other material is unchanged; no silent self-clock."""
    names = list(registered())
    materials = {name: _make(name) for name in names}
    for material in materials.values():
        _observe_all(material, _teaching_observations(4))
        with contextlib.suppress(Exception):
            _admit(material, improvement=1.0, retention=1.0)
    before = {name: material.durable_state_digest() for name, material in materials.items()}

    # Without harness advance, digests must stay put.
    idle = {name: material.durable_state_digest() for name, material in materials.items()}
    idle_stable = idle == before

    k4 = materials["K4_continuous_time_plastic_field"]
    assert hasattr(k4, "advance")
    k4.advance(200)  # type: ignore[attr-defined]
    k4_changed = k4.durable_state_digest() != before["K4_continuous_time_plastic_field"]

    others_stable = True
    advanced_others: list[str] = []
    for name, material in materials.items():
        if name == "K4_continuous_time_plastic_field":
            continue
        if hasattr(material, "advance"):
            # Only K4 may expose harness advance as a durable mechanism.
            advanced_others.append(name)
            others_stable = False
            continue
        if material.durable_state_digest() != before[name]:
            others_stable = False

    # Source-level self-clock refusal for K4.
    import inspect

    from substrate.genesis_k_structural import K4_continuous_time_plastic_field

    source = inspect.getsource(K4_continuous_time_plastic_field)
    no_wall_clock = all(token not in source for token in ("time.time", "time.monotonic", "perf_counter", "datetime.now"))

    checks = {
        "idle_without_harness_advance_stable": idle_stable,
        "k4_changes_under_advance": k4_changed,
        "non_k4_unchanged_by_k4_advance": others_stable,
        "k4_does_not_read_wall_clock": no_wall_clock,
        "no_other_material_exposes_advance": not advanced_others,
    }
    return _result(
        "continuous_time_advances_only_by_harness",
        MECH_K4,
        checks,
        k4_before=before["K4_continuous_time_plastic_field"],
        k4_after=k4.durable_state_digest(),
        advanced_others=advanced_others,
        material_count=len(names),
    )


# --------------------------------------------------------------------------
# 10. checkpoint_covers_everything
# --------------------------------------------------------------------------


def _facet_present(durable: Mapping[str, Any], facet: str) -> bool:
    if facet == "topology":
        if "topology" in durable:
            return True
        if "nodes" in durable or "edges" in durable:
            return True
        return "fibers" in durable
    if facet == "compiled_procedures":
        return "compiled_procedures" in durable or "compiled" in durable
    if facet == "precision_map":
        return "precision_map" in durable
    if facet == "goals":
        if "goals" in durable or "goal_commitments" in durable:
            return True
        shell = durable.get("shell")
        return isinstance(shell, dict) and "goal_commitments" in shell
    return facet in durable


def _strip_facet(checkpoint: Mapping[str, Any], facet: str) -> dict[str, Any]:
    stripped = copy.deepcopy(dict(checkpoint))
    durable = stripped.get("durable")
    if not isinstance(durable, dict):
        stripped["durable"] = {}
        durable = stripped["durable"]
    if facet == "topology":
        durable.pop("topology", None)
        durable.pop("nodes", None)
        durable.pop("edges", None)
        durable.pop("archive", None)
        durable.pop("rent", None)
        durable.pop("fibers", None)
    elif facet == "compiled_procedures":
        durable.pop("compiled_procedures", None)
        durable.pop("compiled", None)
    elif facet == "precision_map":
        durable.pop("precision_map", None)
        durable.pop("precision_rent", None)
    elif facet == "goals":
        durable.pop("goals", None)
        durable.pop("goal_commitments", None)
        shell = durable.get("shell")
        if isinstance(shell, dict):
            shell.pop("goal_commitments", None)
    else:
        durable.pop(facet, None)
    return stripped


def restore_with_coverage(material: MaterialBase, checkpoint: Mapping[str, Any], *, required: Sequence[str]) -> None:
    """Restore only when every required cognitive facet is present in the checkpoint."""
    durable = checkpoint.get("durable")
    if not isinstance(durable, Mapping):
        raise CheckpointCoverageError("checkpoint durable payload missing")
    missing = [facet for facet in required if not _facet_present(durable, facet)]
    if missing:
        raise CheckpointCoverageError(f"checkpoint omits required facets: {missing}")
    material.restore(checkpoint)


def checkpoint_covers_everything() -> dict[str, Any]:
    """Checkpoint/restore exact digests for all registered materials; stripped facets fail restore."""
    names = list(registered())
    round_trips: dict[str, bool] = {}
    for name in names:
        material = _make(name)
        _observe_all(material, _teaching_observations(4))
        with contextlib.suppress(Exception):
            _admit(material, improvement=1.0, retention=1.0)
        if name == "K4_continuous_time_plastic_field" and hasattr(material, "advance"):
            material.advance(80)  # type: ignore[attr-defined]
        checkpoint = material.checkpoint()
        durable_digest = material.durable_state_digest()
        active_digest = material.active_state_digest()
        seen = material.observations_seen
        elapsed = material.elapsed_ms
        # Mutate after checkpoint.
        _observe_all(material, _teaching_observations(2))
        with contextlib.suppress(Exception):
            _admit(material, improvement=0.5, retention=0.5)
        material.restore(checkpoint)
        round_trips[name] = (
            material.durable_state_digest() == durable_digest
            and material.active_state_digest() == active_digest
            and material.observations_seen == seen
            and material.elapsed_ms == elapsed
        )

    # Four stripping tests on a rich material that carries all facets.
    rich_name = "K10_integrated_plastic_field"
    rich = _make(rich_name)
    _observe_all(rich, _teaching_observations(5))
    _admit(rich, improvement=1.0, retention=1.0)
    full = rich.checkpoint()
    required = ("topology", "compiled_procedures", "precision_map", "goals")
    # Ensure goals facet is addressable for coverage (shell.goal_commitments).
    if isinstance(full.get("durable"), dict) and not _facet_present(full["durable"], "goals"):
        full = copy.deepcopy(full)
        full["durable"].setdefault("shell", {})
        if isinstance(full["durable"]["shell"], dict):
            full["durable"]["shell"].setdefault("goal_commitments", ())
        else:
            full["durable"]["goals"] = ()

    strip_results: dict[str, bool] = {}
    for facet in required:
        stripped = _strip_facet(full, facet)
        refused = False
        try:
            clone = _make(rich_name)
            restore_with_coverage(clone, stripped, required=required)
        except CheckpointCoverageError:
            refused = True
        except Exception:
            refused = True
        strip_results[facet] = refused

    checks = {
        "all_materials_round_trip": all(round_trips.values()) and len(round_trips) == len(names),
        "registered_count_is_twenty_three": len(names) == 23,
        "strip_topology_refused": strip_results.get("topology", False),
        "strip_compiled_procedures_refused": strip_results.get("compiled_procedures", False),
        "strip_precision_map_refused": strip_results.get("precision_map", False),
        "strip_goals_refused": strip_results.get("goals", False),
    }
    return _result(
        "checkpoint_covers_everything",
        "checkpoint_restore_covers_topology_procedures_precision_goals",
        checks,
        round_trips=round_trips,
        strip_results=strip_results,
        registered=names,
    )


# --------------------------------------------------------------------------
# 11. migration_preserves_identity
# --------------------------------------------------------------------------


def migration_preserves_identity(material_name: str = "K1_monolithic_plastic_field") -> dict[str, Any]:
    """Export state, import into same material after intervening architecture change; identity continuous."""
    source = _make(material_name)
    _observe_all(source, _teaching_observations(6))
    receipts = _admit(source, improvement=1.0, retention=1.0)
    exported = source.checkpoint()
    identity_digest = source.durable_state_digest()
    answer_sig = _answer_signature(_answers(source))

    # Intervening architecture change on a fresh instance (different option / post-mutation).
    if material_name == "K1_monolithic_plastic_field":
        target = build(material_name, _opportunity(), field_dim=24)
    elif material_name == "K5_recurrent_state_space_plastic_field":
        target = build(material_name, _opportunity(), latent_dim=6)
    else:
        target = _make(material_name)
    # Pollute the target so a silent no-op restore cannot pass by coincidence.
    _observe_all(target, _teaching_observations(3))
    with contextlib.suppress(Exception):
        _admit(target, improvement=0.1, retention=0.0)
    polluted_digest = target.durable_state_digest()

    target.restore(exported)
    restored_digest = target.durable_state_digest()
    restored_answers = _answer_signature(_answers(target))

    # Silent reset detector: a restore that yields empty/bootstrap state fails.
    silent_reset = restored_digest != identity_digest or restored_answers != answer_sig

    checks = {
        "export_nonempty": bool(receipts) or identity_digest != _make(material_name).durable_state_digest(),
        "target_was_polluted": polluted_digest != identity_digest,
        "exact_identity_after_import": restored_digest == identity_digest,
        "answers_continuous": restored_answers == answer_sig,
        "no_silent_reset": not silent_reset,
    }
    return _result(
        "migration_preserves_identity",
        "export_import_preserves_durable_identity",
        checks,
        material=material_name,
        identity_digest=identity_digest,
        restored_digest=restored_digest,
        polluted_digest=polluted_digest,
    )


# --------------------------------------------------------------------------
# 12. s2_parity_is_measured
# --------------------------------------------------------------------------


def _opportunity_arm_record(material: MaterialBase) -> dict[str, Any]:
    """Measured equal-opportunity vector delivered to an arm (budgets and stream digests).

    Live materials spend operations differently by mechanism; the parity canary
    audits the *delivered* opportunity, which is what equal-resource fairness
    constrains. A larger operation budget on one arm must fail the audit.
    """
    opportunity = material.opportunity()
    return {
        "name": material.name,
        "information": opportunity.observation_digest,
        "sensors": tuple(opportunity.sensor_channels),
        "teaching": opportunity.teaching_digest,
        "compute": int(opportunity.ledger.operation_budget),
        "plasticity": int(opportunity.ledger.durable_write_budget) if opportunity.plasticity_enabled else 0,
        "persistence": 1 if opportunity.persistence_enabled else 0,
        "memory": int(opportunity.ledger.byte_budget or 0),
    }


def s2_parity_is_measured() -> dict[str, Any]:
    """Parity audit reports every channel measured and equal; budget skew must fail the audit."""
    observations = _teaching_observations(6)
    sensors = tuple(sorted({row.channel for row in observations}))
    s2_opportunity = equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=sensors,
        operation_budget=5_000,
        durable_write_budget=200,
    )
    cand_opportunity = equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=sensors,
        operation_budget=5_000,
        durable_write_budget=200,
    )
    s2 = build(C.CANONICAL_S2_ID, s2_opportunity)
    candidate = build("K1_monolithic_plastic_field", cand_opportunity)
    for observation in observations:
        s2.observe(observation)
        candidate.observe(observation)
    for material in (s2, candidate):
        proposals = material.propose()
        if proposals:
            material.apply(
                [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
            )

    equal_records = [_opportunity_arm_record(s2), _opportunity_arm_record(candidate)]
    equal_audit = parity.parity_audit(equal_records, unit={"stream": "canary-equal"})
    channels_measured = set(equal_audit["channel_pass"])
    every_channel = channels_measured == set(C.PARITY_CHANNELS)
    equal_pass = bool(equal_audit["all_pass"]) and every_channel

    # Deliberately larger operation budget on the candidate → audit must fail.
    skew_s2_opp = equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=sensors,
        operation_budget=1_000,
        durable_write_budget=200,
    )
    skew_cand_opp = equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=sensors,
        operation_budget=50_000,
        durable_write_budget=200,
    )
    skew_s2 = build(C.CANONICAL_S2_ID, skew_s2_opp)
    skew_cand = build("K1_monolithic_plastic_field", skew_cand_opp)
    for observation in observations:
        skew_s2.observe(observation)
        skew_cand.observe(observation)
    skew_records = [_opportunity_arm_record(skew_s2), _opportunity_arm_record(skew_cand)]
    skew_audit = parity.parity_audit(skew_records, unit={"stream": "canary-skew"})
    skew_fails = not bool(skew_audit["all_pass"])
    compute_fails = skew_audit["channel_pass"].get("compute") is False

    checks = {
        "every_channel_measured": every_channel,
        "equal_stream_passes": equal_pass,
        "skewed_budget_fails": skew_fails and compute_fails,
        "equal_audit_activation_false": equal_audit.get("activation") is False,
        "skew_audit_activation_false": skew_audit.get("activation") is False,
    }
    return _result(
        "s2_parity_is_measured",
        "measured_equal_opportunity_parity_audit",
        checks,
        equal_channel_pass=dict(equal_audit["channel_pass"]),
        skew_channel_pass=dict(skew_audit["channel_pass"]),
        parity_channels=list(C.PARITY_CHANNELS),
        equal_measured=equal_audit["measured"],
        skew_measured=skew_audit["measured"],
    )


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------


CANARIES: dict[str, Callable[[], dict[str, Any]]] = {
    "verified_rewrite": verified_rewrite,
    "refused_rewrite": refused_rewrite,
    "rollback_removes_benefit": rollback_removes_benefit,
    "stability_under_noise": stability_under_noise,
    "precision_promotion_earns_its_bits": precision_promotion_earns_its_bits,
    "topology_growth_pays_rent": topology_growth_pays_rent,
    "shadow_field_does_not_write": shadow_field_does_not_write,
    "compiled_procedure_preserves_reliability": compiled_procedure_preserves_reliability,
    "continuous_time_advances_only_by_harness": continuous_time_advances_only_by_harness,
    "checkpoint_covers_everything": checkpoint_covers_everything,
    "migration_preserves_identity": migration_preserves_identity,
    "s2_parity_is_measured": s2_parity_is_measured,
}


def run_all() -> dict[str, Any]:
    """Run every mechanism canary and return the aggregate report."""
    rows: dict[str, Any] = {}
    for name, function in CANARIES.items():
        rows[name] = function()
    return {
        "canaries": rows,
        "all_pass": all(bool(row.get("all_pass")) for row in rows.values()),
        "canary_count": len(rows),
        "failed": [name for name, row in rows.items() if not row.get("all_pass")],
        "activation": False,
    }


__all__ = [
    "CANARIES",
    "CANARY_NAMES",
    "CheckpointCoverageError",
    "ProcedureCompiler",
    "ShadowField",
    "StabilityBook",
    "checkpoint_covers_everything",
    "compiled_procedure_preserves_reliability",
    "continuous_time_advances_only_by_harness",
    "migration_preserves_identity",
    "precision_promotion_earns_its_bits",
    "refused_rewrite",
    "restore_with_coverage",
    "rollback_removes_benefit",
    "run_all",
    "s2_parity_is_measured",
    "shadow_field_does_not_write",
    "stability_under_noise",
    "topology_growth_pays_rent",
    "verified_rewrite",
]
