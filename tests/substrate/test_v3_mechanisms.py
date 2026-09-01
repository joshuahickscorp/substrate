from __future__ import annotations

import copy

import pytest

from substrate import epistemology as E
from substrate import metacog as M
from substrate import ontology as O
from substrate import v3fabric as F
from substrate import v3state as S
from substrate.world import StructuralUnderstanding


def test_active_ontology_admits_gain_and_rolls_back_null():
    ontology = O.ActiveOntology()
    ontology.observe("a", {"shared", "left"})
    ontology.observe("b", {"shared", "right"})
    revision = ontology.form_category("shared", {"a", "b"}, evidence=("receipt",), predicted_benefit=0.2)
    ontology.complete_revision(revision, held_out_benefit=0.2)
    assert revision.admitted
    rejected = ontology.map_representation("surface", "shared", evidence=("receipt",), predicted_benefit=0.1)
    ontology.complete_revision(rejected, held_out_benefit=0.0)
    assert rejected.rolled_back
    assert "surface" not in ontology.cross_representation_maps


def test_epistemic_knowledge_requires_warrant_not_confidence():
    ledger = E.EpistemicLedger()
    ledger.add(
        E.EpistemicBelief(
            "claim",
            "generated",
            "simulated",
            "simulation",
            "simulated",
            ("generated",),
            confidence=0.99,
            domain_scope=("simulation",),
            held_out_utility=1.0,
        )
    )
    receipt = ledger.admit_knowledge("claim", independently_verified=False)
    assert not receipt["admitted"]
    assert "method does not warrant knowledge" in receipt["reasons"]


def test_reasoning_selection_is_preoutcome_and_relational():
    portfolio = M.ReasoningPortfolio()
    trace = portfolio.select_and_run(
        {
            "features": {"relational_transfer"},
            "source_relations": [["a", "b"]],
            "candidate_relations": [["x", "y"]],
            "mapping": {"a": "x", "b": "y"},
        }
    )
    assert trace.mode == "analogy"
    assert trace.conclusion is True
    assert trace.selected_from_preoutcome_features


def test_structural_understanding_explains_and_preserves_counterfactual_background():
    model = StructuralUnderstanding({("a", "b"), ("b", "c")})
    explanation = model.explain("a", "c")
    counterfactual = model.counterfactual({"a"}, {"b": False})
    assert explanation["derived"]
    assert explanation["falsifier"]
    assert counterfactual["background_preserved"]
    assert "c" not in counterfactual["counterfactual"]


def test_task_observation_excludes_explicit_answer_fields():
    task = F.generate_task(1, "cross_representation_systems", 1, "construction")
    observation = task.observation()
    assert not {"target", "private_target", "answer", "oracle_operation"} & set(observation)
    assert not {"target", "private_target", "answer", "oracle_operation"} & set(observation["public"])
    assert "surface_dictionary" not in observation["public"]


def test_cached_task_generation_returns_isolated_mutable_payloads():
    first = F.generate_task(1, "cross_representation_systems", 1, "construction")
    second = F.generate_task(1, "cross_representation_systems", 1, "construction")
    assert first is not second
    first.public["relations"][0][0] = "mutated"
    first.public["relations"].append(["leak", "shared-state"])
    first.public["mutated"] = True
    assert "mutated" not in second.public
    assert second.public["relations"][0][0] != "mutated"
    assert ["leak", "shared-state"] not in second.public["relations"]
    assert first.private_target == second.private_target


def test_positive_and_negative_allocation_headroom():
    positive = F.allocation_headroom(1, "positive_a", 4096)
    negative = F.allocation_headroom(1, "no_headroom", 4096)
    assert positive["oracle_residual"] > 0.08
    assert negative["oracle_residual"] == 0.0


def test_integrated_checkpoint_exact_and_corruption_refused():
    entity = S.IntegratedEntity()
    for family in (
        "ontology_garden",
        "epistemic_laboratory",
        "reasoning_method_selection",
        "scientific_inquiry",
    ):
        for index in range(8):
            entity.experience(F.generate_task(101, family, index, "cheap_admission"))
    checkpoint = entity.checkpoint()
    assert S._semantic_state_hash(checkpoint["semantic_state"]) == S.sha_obj(checkpoint["semantic_state"])
    restored = S.IntegratedEntity.restore(checkpoint)
    assert restored.identity_hash() == checkpoint["identity_hash"]
    corrupted = copy.deepcopy(checkpoint)
    corrupted["semantic_state"]["step"] += 1
    with pytest.raises(S.Refused):
        S.IntegratedEntity.restore(corrupted)


def test_semantic_state_hash_preserves_circular_refusal():
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="Circular reference"):
        S._semantic_state_hash(cyclic)


def test_semantic_state_hash_matches_canonical_bytes_for_unicode_and_scalars():
    value = {"unicode": "naïve café", "tuple": (1, False, None), "float": 0.125}
    assert S._semantic_state_hash(value) == S.sha_obj(value)


def test_activation_is_false():
    assert F.generate_task(1, "ontology_garden", 0, "construction").observation()
    assert S.IntegratedEntity().activation is False
