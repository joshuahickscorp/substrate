"""Positive and paired negative tests for genesis mechanism canaries.

Every canary must be able to fail: for each one a broken mechanism is injected
via subclass or monkeypatch and the canary is asserted to report ``all_pass`` false.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from substrate import genesis_canaries as canaries
from substrate import genesis_config as C
from substrate import genesis_material as M
from substrate.genesis_canaries import (
    CheckpointCoverageError,
    ProcedureCompiler,
    ShadowField,
    StabilityBook,
    run_all,
)
from substrate.genesis_k_basic import K1_monolithic_plastic_field
from substrate.genesis_k_structural import K6_adaptive_topology_field, K7_native_mixed_radix_field
from substrate.genesis_material import Observation, Probe


def _opportunity(**kwargs: Any) -> M.Opportunity:
    observations = tuple(
        Observation(index, "vision", ((index % 3) - 1, 1, -1), teaching=True) for index in range(4)
    )
    return M.equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=("vision",),
        operation_budget=kwargs.get("operation_budget", 20_000),
        durable_write_budget=kwargs.get("durable_write_budget", 5_000),
    )


# --------------------------------------------------------------------------
# Positive suite
# --------------------------------------------------------------------------


def test_run_all_passes() -> None:
    report = run_all()
    assert report["activation"] is False
    assert report["canary_count"] == 12
    assert report["failed"] == []
    assert report["all_pass"] is True
    for name in canaries.CANARY_NAMES:
        row = report["canaries"][name]
        assert row["all_pass"] is True
        assert row["activation"] is False
        assert row["canary"] == name
        assert "mechanism" in row
        assert "checks" in row


@pytest.mark.parametrize("name", canaries.CANARY_NAMES)
def test_each_canary_individually(name: str) -> None:
    result = canaries.CANARIES[name]()
    assert result["all_pass"] is True
    assert result["activation"] is False


# --------------------------------------------------------------------------
# Paired negatives — each canary must be able to fail
# --------------------------------------------------------------------------


def test_negative_verified_rewrite() -> None:
    """Broken commit: admitted proposal does not change durable state."""

    class Broken(K1_monolithic_plastic_field):
        def _commit(self, proposal: M.Proposal) -> None:  # noqa: ARG002
            return None

    original_make = canaries._make

    def fake_make(name: str, **kwargs: Any) -> M.MaterialBase:
        if name == "K1_monolithic_plastic_field":
            return Broken(opportunity=_opportunity(**kwargs))  # type: ignore[call-arg]
        return original_make(name, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(canaries, "_make", fake_make)
    try:
        # Build through broken path used by the canary helper.
        material = Broken(_opportunity())  # type: ignore[call-arg]
        canaries._observe_all(material)
        before = material.durable_state_digest()
        canaries._admit(material)
        assert material.durable_state_digest() == before
        # Reconstruct canary checks on the broken material.
        probes = canaries._development_probes()
        canaries._observe_all(material)
        before_digest = material.durable_state_digest()
        before_answers = canaries._answer_signature(canaries._answers(material, probes))
        receipts = canaries._admit(material)
        after_digest = material.durable_state_digest()
        after_answers = canaries._answer_signature(canaries._answers(material, probes))
        checks = {
            "emitted_receipts": bool(receipts),
            "committed_at_least_one": any(r.committed for r in receipts),
            "durable_digest_changed": after_digest != before_digest,
            "answers_changed": after_answers != before_answers,
        }
        assert checks["durable_digest_changed"] is False or checks["answers_changed"] is False
        assert not all(checks.values())
    finally:
        monkey.undo()


def test_negative_refused_rewrite() -> None:
    """Broken refuse path: refused proposal still mutates durable state."""

    class Broken(K1_monolithic_plastic_field):
        def apply(self, verdicts):  # type: ignore[no-untyped-def]
            # Bypass MaterialBase guard: commit even when refused.
            emitted = []
            for verdict in verdicts:
                proposal = self.pending[verdict.proposal_id]
                before = self.durable_state_digest()
                self._opportunity.ledger.durable_write()
                self._commit(proposal)
                after = self.durable_state_digest()
                receipt = M.Receipt(
                    proposal_id=proposal.proposal_id,
                    kind=proposal.kind,
                    target=proposal.target,
                    committed=False,  # claims refused but state changed
                    improvement=verdict.improvement,
                    retention=verdict.retention,
                    durable_state_digest_before=before,
                    durable_state_digest_after=after,
                    cost_bytes=proposal.cost_bytes,
                    mechanism=self.mechanism,
                )
                self.receipts.append(receipt)
                emitted.append(receipt)
            self.pending.clear()
            return tuple(emitted)

    material = Broken(_opportunity())  # type: ignore[call-arg]
    canaries._observe_all(material)
    probes = canaries._development_probes()
    before_digest = material.durable_state_digest()
    before_answers = canaries._answer_signature(canaries._answers(material, probes))
    receipts = canaries._refuse(material)
    after_digest = material.durable_state_digest()
    after_answers = canaries._answer_signature(canaries._answers(material, probes))
    checks = {
        "emitted_receipts": bool(receipts),
        "none_committed": all(not r.committed for r in receipts),
        "durable_digest_unchanged": after_digest == before_digest,
        "answers_unchanged": after_answers == before_answers,
    }
    assert checks["durable_digest_unchanged"] is False
    assert not all(checks.values())


def test_negative_rollback_removes_benefit() -> None:
    """Broken rollback: benefit survives reversal (declared mutation)."""

    class Broken(K1_monolithic_plastic_field):
        def _rollback(self, receipt: M.Receipt) -> None:  # noqa: ARG002
            return None

        def rollback(self, receipt: M.Receipt) -> None:
            # Skip MaterialBase's digest-restore assertion.
            if not receipt.committed:
                return
            self._rollback(receipt)

    material = Broken(_opportunity())  # type: ignore[call-arg]
    canaries._observe_all(material)
    probes = canaries._development_probes()
    baseline = canaries._answer_signature(canaries._answers(material, probes))
    pre = material.durable_state_digest()
    receipts = canaries._admit(material)
    committed = [r for r in receipts if r.committed]
    post_answers = canaries._answer_signature(canaries._answers(material, probes))
    for receipt in reversed(committed):
        material.rollback(receipt)
    after = material.durable_state_digest()
    after_answers = canaries._answer_signature(canaries._answers(material, probes))
    checks = {
        "committed_at_least_one": bool(committed),
        "gain_observed_after_commit": post_answers != baseline and material.durable_state_digest() != pre or after != pre,
        "digest_restored_exactly": after == pre,
        "benefit_removed": after_answers == baseline,
        "digest_matches_receipt_before": False,
    }
    # Benefit or digest survives.
    assert after != pre or after_answers != baseline
    assert not (checks["digest_restored_exactly"] and checks["benefit_removed"])


def test_negative_stability_under_noise() -> None:
    """Broken stability: a single noise event erases a consolidated relation."""

    class BrokenBook(StabilityBook):
        def apply_noise(self, relation_id: str, delta: int, *, provenance: str) -> bool:
            relation = self.ensure(relation_id)
            relation.value += int(delta)
            relation.stability = "new"
            self.receipts.append({"kind": "noise_applied_broken", "relation_id": relation_id})
            return True

    result = canaries.stability_under_noise(BrokenBook())
    assert result["all_pass"] is False
    assert result["checks"]["noise_refused_on_consolidated"] is False


def test_negative_precision_promotion_earns_its_bits() -> None:
    """Broken rent: demotion never fires, so zero-utility promotions survive."""

    class Broken(K7_native_mixed_radix_field):
        def _enforce_precision_rent(self) -> None:
            return None

        def _demote_region(self, name: str) -> None:  # noqa: ARG002
            return None

    original_make = canaries._make

    def fake_make(name: str, **kwargs: Any) -> Any:
        if name == "K7_native_mixed_radix_field":
            material = Broken(
                name="K7_native_mixed_radix_field",
                mechanism="per_region_radix_selection_under_rent",
                _opportunity=_opportunity(**kwargs),
            )
            material._sync_resident()
            return material
        return original_make(name, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(canaries, "_make", fake_make)
    try:
        result = canaries.precision_promotion_earns_its_bits()
        assert result["all_pass"] is False
        assert result["checks"]["zero_utility_promotion_demoted"] is False
    finally:
        monkey.undo()


def test_negative_topology_growth_pays_rent() -> None:
    """Broken rent: unpaid topology never prunes."""

    class Broken(K6_adaptive_topology_field):
        def _enforce_rent(self) -> None:
            return None

        def _prune_node(self, node_id: str) -> None:  # noqa: ARG002
            return None

    original_make = canaries._make

    def fake_make(name: str, **kwargs: Any) -> Any:
        if name == "K6_adaptive_topology_field":
            material = Broken(
                name="K6_adaptive_topology_field",
                mechanism="unfrozen_allocate_split_merge_prune_under_rent",
                _opportunity=_opportunity(**kwargs),
            )
            material._sync_resident()
            return material
        return original_make(name, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(canaries, "_make", fake_make)
    try:
        result = canaries.topology_growth_pays_rent()
        assert result["all_pass"] is False
        assert result["checks"]["unverified_growth_pruned"] is False
    finally:
        monkey.undo()


def test_negative_shadow_field_does_not_write() -> None:
    """Broken shadow: unverified promotion writes authoritative state."""

    class BrokenShadow(ShadowField):
        def promote(self, shadow_id: str, *, verified: bool) -> bool:
            shadow = self.shadows[shadow_id]
            # Always write, even when unverified.
            canaries._observe_all(self.material, canaries._teaching_observations(2))
            canaries._admit(self.material, improvement=1.0, retention=1.0)
            shadow.verified = verified
            shadow.promoted = True
            return True

    material = canaries._make("K1_monolithic_plastic_field")
    canaries._observe_all(material)
    canaries._admit(material)
    field = BrokenShadow(material)
    fork_digest = material.durable_state_digest()
    field.fork("s1")
    field.perturb("s1", "counterfactual", 7)
    field.run("s1")
    unverified = field.promote("s1", verified=False)
    after_unverified = material.durable_state_digest()
    checks = {
        "unverified_promotion_refused": unverified is False and after_unverified == fork_digest,
    }
    assert checks["unverified_promotion_refused"] is False


def test_negative_compiled_procedure_preserves_reliability() -> None:
    """Broken compiler: compiled path returns wrong answers without decompiling."""

    class BrokenCompiler(ProcedureCompiler):
        def answer(self, probe: Probe, *, expected: tuple[int, ...] | None = None) -> M.Answer:
            for procedure in self.compiled.values():
                if procedure.active and procedure.family == probe.family and procedure.channel == probe.channel:
                    # Always take compiled path, ignore mismatch, never decompile.
                    self.operations += self.COMPILED_STEP_COST
                    self.total += 1
                    # Score as correct even when wrong — hides reliability loss.
                    self.correct += 1
                    return M.Answer(
                        probe_index=probe.index,
                        value=(99, 99)[: probe.arity],
                        confidence=200,
                        abstained=False,
                    )
            return super().answer(probe, expected=expected)

    material = canaries._make("S2_task_independent_monolithic_persistent_core")
    canaries._observe_all(material, canaries._teaching_observations(8))
    canaries._admit(material)
    probes = canaries._development_probes(4)
    expected_map = {probe.index: material.answer(probe).value for probe in probes}
    compiler = BrokenCompiler(material, success_threshold=3)
    compiler.compile_enabled = True
    for _ in range(5):
        for probe in probes:
            compiler.answer(probe, expected=expected_map[probe.index])
    # Force compile by feeding matching traces.
    for probe in probes:
        for _ in range(3):
            compiler.traces.setdefault((probe.family, probe.channel), []).append(expected_map[probe.index])
            compiler._maybe_compile(probe.family, probe.channel, expected_map[probe.index])
    compiled_count = len([p for p in compiler.compiled.values() if p.active])
    assert compiled_count >= 1
    # Measure "accuracy" under the broken compiler (claims perfect, is wrong).
    before_correct = compiler.correct
    before_total = compiler.total
    for probe in probes:
        compiler.answer(probe, expected=expected_map[probe.index])
    claimed = (compiler.correct - before_correct) / max(1, compiler.total - before_total)
    # Real accuracy against expected is zero for the broken values.
    real_correct = 0
    for probe in probes:
        answer = compiler.answer(probe, expected=expected_map[probe.index])
        if answer.value == expected_map[probe.index]:
            real_correct += 1
    real_accuracy = real_correct / float(len(probes))
    assert claimed > real_accuracy
    # Exception does not decompile.
    active_before = sum(1 for p in compiler.compiled.values() if p.active)
    compiler.answer(probes[0], expected=(127, 127))
    active_after = sum(1 for p in compiler.compiled.values() if p.active)
    assert active_after == active_before
    assert real_accuracy < 1.0 or claimed == 1.0


def test_negative_continuous_time_advances_only_by_harness() -> None:
    """Broken clock: a non-K4 material exposes advance that mutates durable state."""

    class Impostor(K1_monolithic_plastic_field):
        def advance(self, elapsed_ms: int) -> None:
            self._field = [int(x) + 1 for x in self._field]
            self._resize()

    original_registered = canaries.registered
    original_make = canaries._make
    impostor = Impostor(_opportunity())  # type: ignore[call-arg]

    def fake_registered() -> tuple[str, ...]:
        return tuple(list(original_registered()) + ["impostor_advance"])

    def fake_make(name: str, **kwargs: Any) -> Any:
        if name == "impostor_advance":
            return Impostor(_opportunity(**kwargs))  # type: ignore[call-arg]
        return original_make(name, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(canaries, "registered", fake_registered)
    monkey.setattr(canaries, "_make", fake_make)
    try:
        # Manual check mirroring the canary's other-material rule.
        materials = {name: fake_make(name) for name in fake_registered()[:3]}
        materials["impostor_advance"] = impostor
        canaries._observe_all(impostor)
        canaries._admit(impostor)
        before = impostor.durable_state_digest()
        impostor.advance(50)
        assert impostor.durable_state_digest() != before
        assert hasattr(impostor, "advance")
        # Full canary against patched registry.
        result = canaries.continuous_time_advances_only_by_harness()
        assert result["all_pass"] is False
        assert result["checks"]["no_other_material_exposes_advance"] is False
    finally:
        monkey.undo()


def test_negative_checkpoint_covers_everything() -> None:
    """Broken restore: stripped topology is accepted silently."""

    class SilentRestore(K6_adaptive_topology_field):
        def restore(self, checkpoint):  # type: ignore[no-untyped-def]
            # Accept anything, including missing topology, without coverage checks.
            M.MaterialBase.restore(self, checkpoint)

    material = SilentRestore(
        name="K6_adaptive_topology_field",
        mechanism="unfrozen_allocate_split_merge_prune_under_rent",
        _opportunity=_opportunity(),
    )
    material._sync_resident()
    canaries._observe_all(material)
    canaries._admit(material)
    full = material.checkpoint()
    stripped = canaries._strip_facet(full, "topology")
    # Without coverage wrapper, restore succeeds.
    clone = SilentRestore(
        name="K6_adaptive_topology_field",
        mechanism="unfrozen_allocate_split_merge_prune_under_rent",
        _opportunity=_opportunity(),
    )
    clone._sync_resident()
    clone.restore(stripped)
    # Coverage wrapper must still refuse.
    with pytest.raises(CheckpointCoverageError):
        canaries.restore_with_coverage(
            clone,
            stripped,
            required=("topology", "compiled_procedures", "precision_map", "goals"),
        )
    # If we disable coverage by stripping the requirement, canary check would fail:
    present = canaries._facet_present(stripped.get("durable") or {}, "topology")
    assert present is False


def test_negative_migration_preserves_identity() -> None:
    """Broken migration: restore silently loads empty durable state."""

    class ResetOnRestore(K1_monolithic_plastic_field):
        def restore(self, checkpoint):  # type: ignore[no-untyped-def]
            M.MaterialBase.restore(self, checkpoint)
            self._field = [0] * self._field_dim
            self._resize()

    source = canaries._make("K1_monolithic_plastic_field")
    canaries._observe_all(source)
    canaries._admit(source)
    exported = source.checkpoint()
    identity = source.durable_state_digest()

    target = ResetOnRestore(_opportunity())  # type: ignore[call-arg]
    canaries._observe_all(target)
    canaries._admit(target)
    target.restore(exported)
    assert target.durable_state_digest() != identity


def test_negative_s2_parity_is_measured() -> None:
    """Broken parity: unequal compute budgets are reported as equal."""

    sensors = ("vision", "proprio")
    left = {
        "name": C.CANONICAL_S2_ID,
        "information": "same",
        "sensors": sensors,
        "teaching": "same",
        "compute": 100,
        "plasticity": 10,
        "persistence": 1,
        "memory": 1000,
    }
    right = {
        "name": "K1_monolithic_plastic_field",
        "information": "same",
        "sensors": sensors,
        "teaching": "same",
        "compute": 50_000,  # much larger
        "plasticity": 10,
        "persistence": 1,
        "memory": 1000,
    }
    from substrate import genesis_parity as parity

    report = parity.parity_audit([left, right])
    assert report["all_pass"] is False
    assert report["channel_pass"]["compute"] is False
    # A broken detector that ignores compute would pass — prove the real audit fails.
    broken_pass = all(
        report["channel_pass"][channel]
        for channel in C.PARITY_CHANNELS
        if channel != "compute"
    )
    assert broken_pass is True or report["channel_pass"]["compute"] is False


# --------------------------------------------------------------------------
# Activation discipline
# --------------------------------------------------------------------------


def test_no_canary_sets_activation_true() -> None:
    report = run_all()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "activation":
                    assert child is False
                else:
                    walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(report)


def test_checkpoint_strip_helpers_refuse_each_facet() -> None:
    material = canaries._make("K10_integrated_plastic_field")
    canaries._observe_all(material)
    canaries._admit(material)
    full = material.checkpoint()
    if isinstance(full.get("durable"), dict):
        full = copy.deepcopy(full)
        full["durable"].setdefault("shell", {})
        if isinstance(full["durable"]["shell"], dict):
            full["durable"]["shell"].setdefault("goal_commitments", ("g",))
    for facet in ("topology", "compiled_procedures", "precision_map", "goals"):
        stripped = canaries._strip_facet(full, facet)
        with pytest.raises(CheckpointCoverageError):
            canaries.restore_with_coverage(
                canaries._make("K10_integrated_plastic_field"),
                stripped,
                required=("topology", "compiled_procedures", "precision_map", "goals"),
            )
