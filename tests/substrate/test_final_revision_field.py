from __future__ import annotations

import copy

import pytest

from substrate import final_revision_field as field
from substrate import final_revision_field_campaign as field_campaign
from substrate import final_revision_io as io


def test_field_contracts_and_exact_shell_are_complete() -> None:
    assert field.FIELD_SYMBOLS == ("Theta", "P_t", "G_t", "Z_t", "E_t", "A", "C_t", "M_t")
    assert len(field.TIMESCALES) == 6
    assert len(field.SKELETONS) == 8
    assert len(field.FIELD_SYMBOLS) == len(set(field.FIELD_SYMBOLS))
    assert len(field.COMMON_FIELD_CONTRACTS) == len(set(field.COMMON_FIELD_CONTRACTS))
    assert set(field.FIELD_SYMBOLS) == set(field.foundation_state_schema()["field_symbols"])
    candidate = field.EndogenousPlasticField("entity")
    assert set(candidate.document()) >= {*field.FIELD_SYMBOLS, "activation"}
    assert not candidate.document()["E_t"]["permissions"]["external_activation"]
    assert not candidate.document()["E_t"]["permissions"]["field_may_rewrite_shell"]
    assert candidate.document()["E_t"]["claim_boundaries"]["foundation_only"] is not False
    shell_snapshot = candidate.exact
    shell_snapshot["permissions"]["external_activation"] = True
    assert not candidate.exact["permissions"]["external_activation"]
    candidate.add_relation("snapshot", 0, precision="ternary", provenance="test://snapshot")
    plastic_snapshot = candidate.plastic
    plastic_snapshot["snapshot"].value = 1
    assert candidate.plastic["snapshot"].value == 0


def test_native_packing_is_exact_and_quinary_triplets_use_seven_bits() -> None:
    for alphabet in field.PRECISION_ALPHABETS.values():
        values = [alphabet[(index * 3) % len(alphabet)] for index in range(257)]
        packed = field.pack_radix(values, alphabet, group_size=3)
        assert field.unpack_radix(packed) == values
    for left in field.PRECISION_ALPHABETS["quinary"]:
        for middle in field.PRECISION_ALPHABETS["quinary"]:
            for right in field.PRECISION_ALPHABETS["quinary"]:
                packed = field.pack_radix([left, middle, right], field.PRECISION_ALPHABETS["quinary"], group_size=3)
                assert packed.bit_length == 7
                assert field.unpack_radix(packed) == [left, middle, right]
    benchmark = field.packing_benchmark(repetitions=2)
    assert benchmark["adaptive_mixed_radix"]["exact_round_trip"]
    assert benchmark["native_projected_update"]["after"] == 1
    assert benchmark["native_low_bit_training"]["accuracy"] == 1.0
    assert not benchmark["native_low_bit_training"]["full_precision_master_weights"]
    assert len(benchmark["learned_codebook"]["centroids"]) == 8
    assert benchmark["ternary_plus_sparse_outliers"]["exact_outlier_count"] > 0


def test_durable_rewrite_requires_independent_verification_and_rolls_back() -> None:
    candidate = field.EndogenousPlasticField("plastic")
    candidate.add_relation("cause", -1, precision="ternary", provenance="test://prior")
    candidate.add_relation("retention", 1, precision="ternary", provenance="test://retention")
    candidate.plasticity_propose("p", "cause", 2, source="model", source_kind="thought", evidence=[])
    with pytest.raises(io.Refused, match="unverified"):
        candidate.plasticity_commit("p")
    self_batch = field.SealedEvaluationBatch.create(
        "self-eval",
        "model",
        [
            field.EvaluationCase("held", ("cause",), True, "held_out"),
            field.EvaluationCase("retain", ("retention",), True, "retention"),
        ],
        authority="test://self-eval",
    )
    candidate.register_evaluation_batch(self_batch)
    with pytest.raises(io.Refused, match="independently"):
        candidate.plasticity_verify("p", evaluation_batch_id="self-eval")
    independent_batch = field.SealedEvaluationBatch.create(
        "independent-eval",
        "independent",
        [
            field.EvaluationCase("held", ("cause",), True, "held_out"),
            field.EvaluationCase("retain", ("retention",), True, "retention"),
        ],
        authority="test://independent-eval",
    )
    candidate.register_evaluation_batch(independent_batch)
    verification = candidate.plasticity_verify("p", evaluation_batch_id="independent-eval")
    assert verification["passed"]
    candidate.plasticity_commit("p")
    assert candidate.predict(["cause"])
    candidate.plasticity_rollback("p")
    assert not candidate.predict(["cause"])


def test_metaplasticity_avoids_isolated_noise_and_remains_revisable() -> None:
    candidate = field.EndogenousPlasticField("meta")
    candidate.add_relation("fact", 1, precision="ternary", stability="consolidated", provenance="test://prior")
    candidate.add_relation("retention", 1, precision="ternary", provenance="test://retention")
    assert not candidate.apply_noise("fact", -1, provenance="test://noise-one")
    assert candidate.plastic["fact"].stability == "consolidated"
    for index in range(2):
        batch_id = f"contradiction-{index}"
        batch = field.SealedEvaluationBatch.create(
            batch_id,
            f"independent-{index}",
            [
                field.EvaluationCase(f"contradiction-{index}", ("fact",), False, "held_out"),
                field.EvaluationCase(f"retain-{index}", ("retention",), True, "retention"),
            ],
            authority=f"test://{batch_id}",
        )
        candidate.register_evaluation_batch(batch)
        contradiction = candidate.verify_contradiction("fact", evaluation_batch_id=batch_id)
        candidate.metaplasticity_destabilize("fact", contradiction=contradiction["digest"])
    assert candidate.plastic["fact"].stability == "reopened"
    competence_batch = field.SealedEvaluationBatch.create(
        "competence",
        "independent-competence",
        [
            field.EvaluationCase("fact-correct", ("fact",), True, "held_out"),
            field.EvaluationCase("retain-correct", ("retention",), True, "retention"),
        ],
        authority="test://competence",
    )
    candidate.register_evaluation_batch(competence_batch)
    competence = candidate.verify_relation_competence("fact", evaluation_batch_id="competence")
    candidate.metaplasticity_reconsolidate("fact", verification=competence["digest"])
    assert candidate.plastic["fact"].stability == "consolidated"
    assert not candidate.plastic["fact"].contradictions


def test_precision_bits_must_earn_promotion_and_demotion_preserves_utility() -> None:
    candidate = field.EndogenousPlasticField("precision")
    candidate.add_relation("r", 1, precision="binary", provenance="test://prior")
    precision_batch = field.PrecisionEvaluationBatch.create(
        "precision-limited",
        "independent-precision",
        [0.0, 2.0],
        tolerance=0.0,
        authority="test://precision-limited",
    )
    candidate.register_precision_evaluation_batch(precision_batch)
    refused = candidate.precision_request(
        "r",
        "quinary",
        evaluation_batch_id="precision-limited",
    )
    assert refused["passed"]
    no_gain_batch = field.PrecisionEvaluationBatch.create(
        "no-gain",
        "independent-no-gain",
        [-1.0, 1.0],
        tolerance=0.0,
        authority="test://no-gain",
    )
    candidate.register_precision_evaluation_batch(no_gain_batch)
    no_gain = candidate.precision_request("r", "quinary", evaluation_batch_id="no-gain")
    with pytest.raises(io.Refused, match="earned"):
        candidate.precision_promote(no_gain)
    earned = candidate.precision_request(
        "r",
        "quinary",
        evaluation_batch_id="precision-limited",
    )
    candidate.precision_promote(earned)
    assert candidate.plastic["r"].precision == "quinary"
    lossy_batch = field.PrecisionEvaluationBatch.create(
        "lossy-demotion",
        "independent-lossy",
        [2.0],
        tolerance=0.0,
        authority="test://lossy",
    )
    candidate.register_precision_evaluation_batch(lossy_batch)
    with pytest.raises(io.Refused, match="loses"):
        candidate.precision_demote("r", "ternary", evaluation_batch_id="lossy-demotion")
    safe_batch = field.PrecisionEvaluationBatch.create(
        "safe-demotion",
        "independent-safe",
        [-1.0, 0.0, 1.0],
        tolerance=0.0,
        authority="test://safe",
    )
    candidate.register_precision_evaluation_batch(safe_batch)
    candidate.precision_demote("r", "ternary", evaluation_batch_id="safe-demotion")
    assert candidate.precision_audit()["all_valid"]


def test_topology_growth_pays_rent_and_is_reversible() -> None:
    candidate = field.EndogenousPlasticField("topology", skeleton="K6_adaptive_topology_field")
    reject_batch = field.TopologyEvaluationBatch.create(
        "reject-random",
        "independent-random",
        [field.TopologyQuery("random-absent", "node_absent", "random", None, True)],
        improvement_required=True,
        authority="test://reject-random",
    )
    candidate.register_topology_evaluation_batch(reject_batch)
    with pytest.raises(io.Refused, match="sealed held-out"):
        candidate.topology_allocate(
            field.CognitiveCell("random", "concept", provenance_pointer="test://random"),
            trigger="random",
            evaluation_batch_id="reject-random",
        )
    allocate_batch = field.TopologyEvaluationBatch.create(
        "allocate-useful",
        "independent-allocation",
        [field.TopologyQuery("useful-present", "node_present", "useful", None, True)],
        improvement_required=True,
        authority="test://allocate-useful",
    )
    candidate.register_topology_evaluation_batch(allocate_batch)
    allocation = candidate.topology_allocate(
        field.CognitiveCell("useful", "concept", provenance_pointer="test://useful"),
        trigger="verified exception",
        evaluation_batch_id="allocate-useful",
    )
    prune_batch = field.TopologyEvaluationBatch.create(
        "prune-useful",
        "independent-prune",
        [field.TopologyQuery("useful-absent", "node_absent", "useful", None, True)],
        improvement_required=True,
        authority="test://prune-useful",
    )
    candidate.register_topology_evaluation_batch(prune_batch)
    candidate.topology_prune("useful", evaluation_batch_id="prune-useful")
    assert "useful" not in candidate.cells
    restore_batch = field.TopologyEvaluationBatch.create(
        "restore-useful",
        "independent-restore",
        [field.TopologyQuery("useful-present-again", "node_present", "useful", None, True)],
        improvement_required=True,
        authority="test://restore-useful",
    )
    candidate.register_topology_evaluation_batch(restore_batch)
    candidate.topology_restore("useful", evaluation_batch_id="restore-useful")
    assert "useful" in candidate.cells
    candidate.topology_rollback(allocation["digest"])
    assert "useful" not in candidate.cells


def test_shadow_isolation_compiler_reopen_and_elapsed_goal() -> None:
    candidate = field.EndogenousPlasticField("combined")
    candidate.add_relation("r", -1, precision="ternary", provenance="test://prior")
    candidate.shadow_fork("shadow", relevant_relations=["r"], relevant_cells=[])
    candidate.shadow_perturb("shadow", "r", 2)
    assert candidate.shadow_run("shadow", query_relations=["r"])["prediction"]
    assert candidate.plastic["r"].value == -1
    candidate.add_relation("retention", 1, precision="ternary", provenance="test://retention")
    shadow_batch = field.SealedEvaluationBatch.create(
        "shadow-promotion",
        "independent",
        [
            field.EvaluationCase("shadow-held", ("r",), True, "held_out"),
            field.EvaluationCase("shadow-retain", ("retention",), True, "retention"),
        ],
        authority="test://shadow-promotion",
    )
    candidate.register_evaluation_batch(shadow_batch)
    candidate.shadow_promote("shadow", evaluation_batch_id="shadow-promotion")
    assert candidate.plastic["r"].value == 1

    candidate.procedure_observe_trace("trace", "OBSERVE", "x")
    candidate.procedure_observe_trace("trace", "INFER")
    candidate.procedure_observe_trace("trace", "COMPARE", "target")
    candidate.procedure_observe_trace("trace", "INFER")
    candidate.procedure_propose(
        "p",
        "trace",
        proposer="trace-learner",
        inputs=["x", "target"],
        assumptions=[],
        scope="test",
        branch_conditions=[],
        failure_conditions=[],
    )
    procedure_batch = field.ProcedureEvaluationBatch.create(
        "procedure-held-out",
        "held-out-evaluator",
        [
            field.ProcedureCase.create("equal", {"x": True, "target": True}, True),
            field.ProcedureCase.create("unequal", {"x": True, "target": False}, False),
        ],
        authority="test://procedure-held-out",
    )
    candidate.register_procedure_evaluation_batch(procedure_batch)
    candidate.procedure_verify("p", evaluation_batch_id="procedure-held-out")
    candidate.procedure_compile(
        "p",
        "trace",
        inputs=["x", "target"],
        assumptions=[],
        scope="test",
        branch_conditions=[],
        failure_conditions=[],
        verification_method="held-out",
        provenance="test://trace",
    )
    assert candidate.procedure_execute("p", {"x": True, "target": True})["result"] is True
    candidate.procedure_monitor("p", passed=False, exception="novel")
    assert candidate.compiled["p"]["reopen_flexible_reasoning"]

    candidate.create_goal("g", "unfinished", provenance="test://goal")
    candidate.schedule_event("GoalDeadline", 2.0, {"goal_id": "g"})
    assert candidate.elapsed_time(3.0)
    assert candidate.exact["goal_commitments"]["g"]["status"] == "unfinished"
    assert candidate.exact["goal_commitments"]["g"]["overdue"]


def test_checkpoint_corruption_fails_closed_and_neutral_migration_is_not_identity_transfer() -> None:
    candidate = field.EndogenousPlasticField("migration", skeleton="K2_graph_plastic_field")
    candidate.add_relation("known", 1, precision="ternary", stability="consolidated", provenance="test://known")
    checkpoint = candidate.checkpoint()
    restored = field.EndogenousPlasticField.restore(checkpoint)
    assert restored.plastic["known"].value == 1
    corrupted = copy.deepcopy(checkpoint)
    corrupted["state"]["P_t"]["known"]["value"] = -1
    with pytest.raises(io.Refused, match="seal mismatch|state digest mismatch"):
        field.EndogenousPlasticField.restore(corrupted)

    neutral = candidate.cognitive_state_export()
    migrated = field.EndogenousPlasticField.cognitive_state_import(
        neutral,
        target_skeleton="K5_recurrent_state_space_plastic_field",
        resource_envelope="512_MB",
    )
    comparison = field.EndogenousPlasticField.cognitive_state_compare(neutral, migrated.cognitive_state_export())
    assert comparison["portable_fields_equal"]
    assert not comparison["semantic_fields_equal"]
    assert not comparison["identity_exact"]
    assert not comparison["identity_transfer_claimed"]
    assert not neutral["semantic_continuity_tested"]
    assert not neutral["identity_transfer_claimed"]
    migration = field.EndogenousPlasticField.cognitive_state_migrate(
        candidate,
        target_skeleton="K5_recurrent_state_space_plastic_field",
        resource_envelope="512_MB",
    )
    rolled_back = field.EndogenousPlasticField.cognitive_state_rollback(migration)
    assert rolled_back.skeleton == candidate.skeleton
    assert rolled_back.plastic["known"].value == candidate.plastic["known"].value


def test_all_28_canaries_pass_but_receive_zero_classification_credit() -> None:
    report = field.run_foundation_canaries()
    assert report["canary_count"] == 28
    assert report["all_present"]
    assert report["all_pass"], [row["canary_id"] for row in report["rows"] if not row["passed"]]
    assert report["classification_credit"] == 0
    assert report["current_final_revision_endpoint_credit"] == 0
    assert all(not row["activation"] for row in report["rows"])


def test_skeletons_resource_frontier_and_s2_parity_are_raw_and_bounded() -> None:
    skeletons = field.skeleton_activity_report()
    assert len(skeletons["rows"]) == 8
    assert skeletons["all_runnable"]
    assert skeletons["mechanically_distinct_transition_count"] == 8
    assert all(not row["principal_quality_claimed"] for row in skeletons["rows"])
    frontier = field.capability_density_frontier()
    assert len(frontier["raw_rows"]) == 2 * 6 * 4
    assert frontier["s2_equal_resource_opportunity_verified"]
    assert not frontier["s2_scientific_parity_claimed"]
    assert frontier["full_raw_performance_reported"]
    receipt = field.raw_metric_receipt(frontier["raw_rows"])
    assert field.verify_raw_metric_receipt(receipt)
    receipt["utility_mean"] = 1.0
    assert not field.verify_raw_metric_receipt(receipt)
    assert field.attractor_microtests()["all_pass"]


def test_field_grok_contracts_are_distinct_local_read_only_and_nonclaiming() -> None:
    commit = "a" * 40
    prompts = [field_campaign.field_grok_prompt(role, evidence_commit=commit) for role in field_campaign.FIELD_GROK_ROLES]
    assert len(prompts) == 20
    assert len(set(prompts)) == 20
    assert field_campaign.REQUIRED_GROK_ROLE_COUNT == 16
    assert all("read-only" in prompt for prompt in prompts)
    assert all("use Grok web" in prompt for prompt in prompts)
    assert all("current_campaign_endpoint_credit (must be 0)" in prompt for prompt in prompts)
    assert all("classification_credit (must be 0)" in prompt for prompt in prompts)


def test_concept_micro_worlds_do_not_consume_future_principal_instances() -> None:
    worlds = field.concept_micro_worlds()
    assert len(worlds["templates"]) == 7
    assert not worlds["future_principal_instances_consumed"]
    assert all(not row["future_principal_instances_consumed"] for row in worlds["templates"])
    assert all("transfer" in row["developmental_stages"] for row in worlds["templates"])
