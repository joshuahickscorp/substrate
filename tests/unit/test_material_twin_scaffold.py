
from __future__ import annotations

import numpy as np
import pytest

from mop.studies.material_twin_scaffold import (
    CONTROL_FAMILIES,
    REPAIR_CONTROLS,
    REPAIR_POLICIES,
    BenchHandoffContract,
    ControlDeclarationSet,
    ControlFamilyDeclaration,
    CrossSubstratePortabilityContract,
    DamageRepairContract,
    DecayingConductanceMap,
    DriftAdaptationContract,
    ExternalSpecimenGate,
    ExternalSpecimenRefusal,
    LeakyEchoStateReservoir,
    LesionSpec,
    Lineage,
    MatchedBudget,
    MaterialTwin,
    NativeDynamicsValueContract,
    OscillatorBank,
    SpecimenTransferContract,
    TwinInterfaceContract,
    build_default_control_set,
    default_bench_handoff,
    default_damage_repair_contract,
    facet_coverage_map,
    selective_lesion_delta,
    validate_twin_interface,
)

MATERIAL_TWIN_SCHEMA = "mop-material-twin/v1"
ALL_PRIORS = (LeakyEchoStateReservoir, DecayingConductanceMap, OscillatorBank)


def test_interface_contract_digest_is_stable() -> None:
    contract = TwinInterfaceContract()
    assert contract.interface_digest() == TwinInterfaceContract().interface_digest()
    assert len(contract.interface_digest()) == 64


def test_interface_rejects_surface_drift() -> None:
    with pytest.raises(ValueError):
        TwinInterfaceContract(required_methods=("reset", "excite"))


def test_interface_rejects_widened_claim_scope() -> None:
    with pytest.raises(ValueError):
        TwinInterfaceContract(claim_scope="a capability was demonstrated")


@pytest.mark.parametrize("prior_cls", ALL_PRIORS)
def test_priors_satisfy_the_interface(prior_cls: type) -> None:
    twin = prior_cls(units=8, seed=1)
    validate_twin_interface(twin)
    assert isinstance(twin, MaterialTwin)


def test_validate_twin_interface_fails_closed_on_partial_object() -> None:
    class Partial:
        def reset(self, seed: int) -> None: ...

    with pytest.raises(ValueError, match="missing callable method"):
        validate_twin_interface(Partial())


@pytest.mark.parametrize("prior_cls", ALL_PRIORS)
def test_prior_is_deterministic_under_seed(prior_cls: type) -> None:
    def run() -> np.ndarray:
        twin = prior_cls(units=10, seed=7)
        twin.excite(np.ones(10) * 0.2)
        twin.evolve(5)
        return twin.readout()

    first = run()
    second = run()
    assert np.allclose(first, second)


@pytest.mark.parametrize("prior_cls", ALL_PRIORS)
def test_prior_energy_and_lineage_grow(prior_cls: type) -> None:
    twin = prior_cls(units=8, seed=3)
    twin.excite(np.ones(8) * 0.1)
    twin.evolve(4)
    assert twin.energy() >= 0.0
    lineage = twin.lineage()
    assert isinstance(lineage, Lineage)
    assert lineage.ops[0].startswith("reset")
    assert any(op.startswith("evolve") for op in lineage.ops)


@pytest.mark.parametrize("prior_cls", ALL_PRIORS)
def test_prior_excite_rejects_wrong_shape(prior_cls: type) -> None:
    twin = prior_cls(units=8, seed=0)
    with pytest.raises(ValueError):
        twin.excite(np.ones(3))


@pytest.mark.parametrize("prior_cls", ALL_PRIORS)
def test_prior_adapt_fits_linear_readout(prior_cls: type) -> None:
    twin = prior_cls(units=6, seed=2)
    twin.excite(np.ones(6) * 0.3)
    twin.evolve(3)
    twin.adapt(np.array([1.0, -1.0]))
    out = twin.readout()
    assert out.shape == (2,)


def test_reservoir_rejects_bad_hyperparameters() -> None:
    with pytest.raises(ValueError):
        LeakyEchoStateReservoir(units=8, leak=0.0)
    with pytest.raises(ValueError):
        LeakyEchoStateReservoir(units=8, radius=1.0)


def test_prior_rejects_tiny_unit_count() -> None:
    with pytest.raises(ValueError):
        OscillatorBank(units=1)


def test_lineage_digest_detects_tampering() -> None:
    with pytest.raises(ValueError):
        Lineage(schema=MATERIAL_TWIN_SCHEMA, ops=("reset:seed=0",), head_digest="0" * 64)


def test_default_control_set_covers_every_family() -> None:
    control_set = build_default_control_set()
    covered = {d.family for d in control_set.declarations}
    assert covered == CONTROL_FAMILIES


def test_control_set_fails_closed_on_incomplete_coverage() -> None:
    budget = MatchedBudget(params=1, flops=1, memory_bytes=1, update_steps=1)
    only_one = ControlFamilyDeclaration(
        id="ctrl.tuned_rnn",
        family="tuned-rnn",
        rationale="r",
        matched=budget,
        tuned=True,
    )
    with pytest.raises(ValueError, match="does not cover"):
        ControlDeclarationSet(schema=MATERIAL_TWIN_SCHEMA, declarations=(only_one,))


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(ValueError):
        MatchedBudget(params=0, flops=1, memory_bytes=1, update_steps=1)


def test_control_must_be_declared_tuned() -> None:
    budget = MatchedBudget(params=1, flops=1, memory_bytes=1, update_steps=1)
    with pytest.raises(ValueError, match="tuned"):
        ControlFamilyDeclaration(
            id="ctrl.ssm",
            family="state-space-model",
            rationale="r",
            matched=budget,
            tuned=False,
        )


def test_control_rejects_unknown_family() -> None:
    budget = MatchedBudget(params=1, flops=1, memory_bytes=1, update_steps=1)
    with pytest.raises(ValueError, match="unsupported control family"):
        ControlFamilyDeclaration(
            id="ctrl.x",
            family="quantum-annealer",
            rationale="r",
            matched=budget,
            tuned=True,
        )


def test_native_dynamics_contract_valid() -> None:
    contract = NativeDynamicsValueContract(
        schema=MATERIAL_TWIN_SCHEMA,
        controls=("tuned-rnn", "random-reservoir"),
        matched_cost_required=True,
        retain_only=("cross-task-future-learnability",),
        replication_min=3,
    )
    assert contract.payload()["replication_min"] == 3


def test_native_dynamics_requires_matched_cost() -> None:
    with pytest.raises(ValueError, match="matched full-system cost"):
        NativeDynamicsValueContract(
            schema=MATERIAL_TWIN_SCHEMA,
            controls=("tuned-rnn",),
            matched_cost_required=False,
            retain_only=("capability-density",),
            replication_min=2,
        )


def test_native_dynamics_rejects_single_replication() -> None:
    with pytest.raises(ValueError, match="two independent replications"):
        NativeDynamicsValueContract(
            schema=MATERIAL_TWIN_SCHEMA,
            controls=("tuned-rnn",),
            matched_cost_required=True,
            retain_only=("capability-density",),
            replication_min=1,
        )


def test_default_damage_repair_contract_valid() -> None:
    contract = default_damage_repair_contract()
    assert contract.controls == REPAIR_CONTROLS
    assert contract.selective_lesion_required


def test_damage_repair_rejects_control_drift() -> None:
    with pytest.raises(ValueError, match="controls or order drift"):
        DamageRepairContract(
            schema=MATERIAL_TWIN_SCHEMA,
            controls=("restart", "fixed-final"),
            metrics=(
                "repair_time",
                "restored_function",
                "future_plasticity",
                "repair_energy",
                "topology_change",
            ),
            selective_lesion_required=True,
            restoration_under_novelty_required=True,
        )


def test_damage_repair_requires_selective_lesion() -> None:
    with pytest.raises(ValueError, match="selective lesion"):
        DamageRepairContract(
            schema=MATERIAL_TWIN_SCHEMA,
            controls=REPAIR_CONTROLS,
            metrics=(
                "repair_time",
                "restored_function",
                "future_plasticity",
                "repair_energy",
                "topology_change",
            ),
            selective_lesion_required=False,
            restoration_under_novelty_required=True,
        )


def test_lesion_spec_validation() -> None:
    with pytest.raises(ValueError):
        LesionSpec(unit_indices=(), fraction=0.5, seed=0, selective=True)
    with pytest.raises(ValueError):
        LesionSpec(unit_indices=(0, 0), fraction=0.5, seed=0, selective=True)
    with pytest.raises(ValueError):
        LesionSpec(unit_indices=(0,), fraction=1.5, seed=0, selective=True)


@pytest.mark.parametrize("prior_cls", ALL_PRIORS)
def test_selective_lesion_delta_is_deterministic(prior_cls: type) -> None:
    lesion = LesionSpec(unit_indices=(0, 1), fraction=0.25, seed=0, selective=True)

    def run() -> dict[str, float]:
        twin = prior_cls(units=8, seed=5)
        twin.excite(np.ones(8) * 0.4)
        twin.evolve(3)
        return selective_lesion_delta(twin, lesion)

    assert run() == run()


def test_selective_lesion_delta_refuses_diffuse_lesion() -> None:
    twin = LeakyEchoStateReservoir(units=8, seed=0)
    diffuse = LesionSpec(unit_indices=(0,), fraction=0.5, seed=0, selective=False)
    with pytest.raises(ValueError, match="selective lesion"):
        selective_lesion_delta(twin, diffuse)


def test_damage_rejects_fraction_that_does_not_match_selected_units() -> None:
    twin = LeakyEchoStateReservoir(units=8, seed=0)
    inconsistent = LesionSpec(unit_indices=(0,), fraction=0.5, seed=0, selective=True)
    with pytest.raises(ValueError, match="fraction"):
        twin.apply_damage(inconsistent)


@pytest.mark.parametrize("policy", REPAIR_POLICIES)
def test_repair_policies_run(policy: str) -> None:
    twin = LeakyEchoStateReservoir(units=8, seed=1)
    twin.excite(np.ones(8) * 0.2)
    twin.evolve(2)
    twin.apply_damage(LesionSpec(unit_indices=(0, 2), fraction=0.25, seed=0, selective=True))
    twin.repair(policy)
    assert any(op == f"repair:{policy}" for op in twin.lineage().ops)


def test_repair_rejects_unknown_policy() -> None:
    twin = OscillatorBank(units=6, seed=0)
    with pytest.raises(ValueError, match="unsupported repair policy"):
        twin.repair("teleport")


def test_drift_contract_valid_and_fails_closed() -> None:
    ok = DriftAdaptationContract(
        schema=MATERIAL_TWIN_SCHEMA,
        drift_model="multiplicative parameter decay",
        adaptation_policy="periodic readout refit",
        controls=("no-adapt", "oracle-reset", "full-retraining"),
        metrics=("drift_rate", "adaptation_lag", "retained_function", "adaptation_energy"),
    )
    assert ok.payload()["drift_model"]
    with pytest.raises(ValueError):
        DriftAdaptationContract(
            schema=MATERIAL_TWIN_SCHEMA,
            drift_model="",
            adaptation_policy="x",
            controls=("no-adapt", "oracle-reset", "full-retraining"),
            metrics=("drift_rate", "adaptation_lag", "retained_function", "adaptation_energy"),
        )


def test_apply_drift_records_lineage() -> None:
    twin = DecayingConductanceMap(units=6, seed=0)
    twin.apply_drift(4)
    assert any(op == "drift:4" for op in twin.lineage().ops)


def test_portability_contract_needs_distinct_forms() -> None:
    with pytest.raises(ValueError, match="two distinct forms"):
        CrossSubstratePortabilityContract(
            schema=MATERIAL_TWIN_SCHEMA,
            source_kind="reservoir",
            target_kind="reservoir",
            portability_metric="readout refit cost",
            controls=("identity", "random-remap", "full-retrain"),
        )


def test_portability_contract_valid() -> None:
    contract = CrossSubstratePortabilityContract(
        schema=MATERIAL_TWIN_SCHEMA,
        source_kind="leaky-echo-state-reservoir",
        target_kind="oscillator-bank",
        portability_metric="readout refit cost under held form",
        controls=("identity", "random-remap", "full-retrain"),
    )
    assert contract.payload()["source_kind"] != contract.payload()["target_kind"]


def test_external_gate_refuses_local_execution() -> None:
    gate = ExternalSpecimenGate()
    with pytest.raises(ExternalSpecimenRefusal):
        gate.authorize_local()


def test_external_gate_rejects_permissive_construction() -> None:
    with pytest.raises(ValueError):
        ExternalSpecimenGate(local_execution_permitted=True)


def test_bench_handoff_emits_form_but_refuses_local_pass() -> None:
    contract = default_bench_handoff()
    form = contract.emit_form()
    assert set(form["fields"]) == {
        "specimen",
        "randomization",
        "environment",
        "safety",
        "stop",
        "portability",
    }
    with pytest.raises(ExternalSpecimenRefusal):
        contract.pass_locally()


def test_bench_handoff_fails_closed_on_missing_field() -> None:
    with pytest.raises(ValueError, match="missing declarations"):
        BenchHandoffContract(
            schema=MATERIAL_TWIN_SCHEMA,
            fields={
                "specimen": "x",
                "randomization": "x",
                "environment": "x",
                "safety": "x",
                "stop": "x",
                "portability": "   ",
            },
        )


def test_bench_handoff_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        BenchHandoffContract(
            schema=MATERIAL_TWIN_SCHEMA,
            fields={
                "specimen": "x",
                "randomization": "x",
                "environment": "x",
                "safety": "x",
                "stop": "x",
                "portability": "x",
                "rogue": "x",
            },
        )


def test_specimen_transfer_refuses_local_pass() -> None:
    contract = SpecimenTransferContract(
        schema=MATERIAL_TWIN_SCHEMA,
        source_specimen="specimen-a",
        target_specimen="specimen-b",
        randomization="independent per-specimen seeds held by the lab",
        transfer_metric="readout re-fit cost across specimens",
        controls=("within-specimen", "random-specimen", "full-refit"),
    )
    with pytest.raises(ExternalSpecimenRefusal):
        contract.pass_locally()


def test_specimen_transfer_needs_distinct_specimens() -> None:
    with pytest.raises(ValueError, match="two distinct specimens"):
        SpecimenTransferContract(
            schema=MATERIAL_TWIN_SCHEMA,
            source_specimen="specimen-a",
            target_specimen="specimen-a",
            randomization="x",
            transfer_metric="x",
            controls=("within-specimen", "random-specimen", "full-refit"),
        )


def test_facet_coverage_map_lists_all_bm_facets() -> None:
    coverage = facet_coverage_map()
    assert set(coverage) == {"BM1", "BM2", "BM3", "BM4"}
    for bullets in coverage.values():
        assert len(bullets) >= 2
