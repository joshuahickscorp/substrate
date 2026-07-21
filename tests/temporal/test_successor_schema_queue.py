from __future__ import annotations

import torch

from mop.temporal.runs import e3, hybrid, successors


REQUIRED = {
    "mean", "lower_95_cb", "upper_95_cb", "per_seed_effects", "per_unit_effects",
    "group_mean", "group_lower_95_cb", "group_upper_95_cb", "group_heterogeneity",
    "bed_specific_effects", "cost_adjusted_effect_per_million_parameter_updates",
    "cost_denominator", "component_floor_status",
}


def test_successor_effect_helpers_emit_complete_contract():
    seeds = [0.10, 0.12, 0.11]
    units = {"u1": 0.10, "u2": 0.12, "u3": 0.11}
    effect = e3._effect_summary(seeds, units, e3.e2.PREREG, 1000,
                                {"name": "floor", "all_pass": True}, {"b": {"mean": 0.11}})
    assert REQUIRED <= set(effect)


def test_matched_noise_is_seeded_reconstructable_and_magnitude_matched():
    learned = torch.tensor([1.0, -2.0, 3.0])
    left = hybrid._noise_matched_to(learned, 1234)
    right = hybrid._noise_matched_to(learned, 1234)
    assert hybrid._tensor_sha(left) == hybrid._tensor_sha(right)
    assert abs(float(left.norm()) - float(learned.norm())) < 1e-6


def test_queue_is_exact_declared_five_and_third_voi_uses_both_effects(monkeypatch):
    artifacts = {
        "MOP_E2_PRINCIPAL_RESULT.json": {"principal_beds": [], "hypothesis_fold": {"hypotheses": {}}},
        "MOP_OWNED_TEMPORAL_CORE_V1.json": {"selected": True, "core": {"owned_parameters": 100}},
        "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json": {"selected": ["harth_stream"]},
        "MOP_THIRD_TEMPORAL_BED_RESULT.json": {"classification": "replicated", "effects": {
            "torch_gru_vs_full_history": {"group_lower_95_cb": 0.12},
            "explicit_mgu_vs_full_history": {"group_lower_95_cb": 0.08},
        }},
    }
    monkeypatch.setattr(successors.io, "exists", lambda name: name in artifacts)
    monkeypatch.setattr(successors.io, "load", lambda name: artifacts[name])
    gates = successors.gates()
    assert {gate["candidate_id"] for gate in gates.values()} == {"E3", "E5", "E6", "E7", "E8"}
    assert gates["third_bed_replication"]["ranking"]["value_of_information"] == 0.08
    assert not gates["minimal_core_cross_domain_transfer"]["opens"]
    assert gates["minimal_core_cross_domain_transfer"]["no_invented_premise"]
    assert not successors._ranking(0.1, 100, [{"artifact": "a", "field": "/x", "value": None}])["complete"]
