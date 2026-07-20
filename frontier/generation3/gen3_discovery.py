"""Generation 3 discovery authority: five candidate theses derived from the Gen2 null map, a 10-criterion
ranking, selection of at most two, and complete 8-gate precompute designs for the selected pair. No principal
Gen3 compute is launched here. House style: no dashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

W = Path("/Users/scammermike/Downloads/mop-scientific-frontier")
OUT = W / "frontier/generation3"
OUT.mkdir(parents=True, exist_ok=True)


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def thesis(prior_failure, new_premise, why_old_could_not, baseline, oracle_headroom, units, cheap_falsification):
    return {"prior_failure_addressed": prior_failure, "new_causal_premise": new_premise,
            "why_old_mechanism_could_not_express_it": why_old_could_not, "established_baseline_to_beat": baseline,
            "oracle_headroom_required": oracle_headroom, "independent_units_required": units,
            "cheapest_falsification": cheap_falsification}


CANDIDATES = {
    "C1_P1R_priority": thesis(
        "P1R external replication null: per-item replay-value prediction did not yield a competitive replay policy",
        "replay value is informative for HOW OFTEN to replay an item (a soft sampling priority over a representative buffer), not for WHICH items to KEEP; the keep decision should stay class-balanced (GDumb-like) while value modulates sampling",
        "the Gen2 P1R was a buffer-admission/eviction filter, so value biased the buffer toward hard/atypical exemplars and destabilized the shared head; it could not separate the keep decision from the replay-frequency decision",
        "GDumb (best established), reservoir, uniform, loss-sampling, class-balanced sampling",
        "an oracle sampling priority over a fixed GDumb buffer must beat uniform GDumb sampling by SESOI",
        "class-incremental tasks (EMNIST-balanced), several seeds",
        "measure whether ANY sampling priority (loss/oracle) over the fixed GDumb buffer beats uniform GDumb sampling; if not, no headroom"),
    "C2_V1_capable_family": thesis(
        "V1 verification was architecture_dependent: real per-item value but only 1 of 3 capable estimators decoded it",
        "verification value is a downstream-decision quantity decodable by a sufficiently expressive estimator; the Gen2 architecture-dependence was an estimator-capacity artifact, not an absence of value",
        "Gen2 admission judged architecture agreement over a fixed small capable set (knn, kernel_ridge, rff_ridge); it could not distinguish a capacity limit from a genuine architecture-dependence",
        "raw uncertainty (U1 frozen negative), entropy, margin, loss; the always/never/random verify controls",
        "an expanded capable estimator family (add an MLP and higher-capacity RFF) must robustly decode the value in >=2 members beyond the best simple control",
        "CIFAR-10 distribution regimes",
        "re-evaluate the existing V1 bed with the expanded family; if <2 members clear the group rule, no headroom"),
    "C3_S1_model_error_aware": thesis(
        "S1 simulation was worse than random: learned-model rollout errors compounded over the horizon",
        "simulate only as deep as the estimated rollout error permits, and trust the simulated value in proportion to a calibrated model-error estimate; depth and trust are conditioned on estimated error",
        "Gen2 S1 used a fixed-depth learned rollout with no error awareness, so compounding error dominated",
        "reactive policy, one-step planner, direct predictor, matched-compute planner, and an oracle-model planner",
        "an oracle-model planner must beat reactive by SESOI (headroom exists), and error-aware depth must recover a fraction of that gap",
        "distinct dynamical systems (gymnasium classic-control plus perturbations)",
        "check that an oracle-model planner beats reactive; if not even a perfect model helps, no headroom (needs a new env harness with a calibrated error estimator, so gate 5 is not yet cheaply runnable)"),
    "C4_M1_causal_message": thesis(
        "M1 messaging fired on predictable-but-noncausal items: message predictability is not causal message value",
        "estimate the INTERVENTION-level value of a message (the causal effect of delivering vs withholding it on the recipient decision), not its predictability",
        "Gen2 M1 predicted per-item message benefit from the sender features but conflated predictability with causal value",
        "no-message, concatenated features, and a centralized matched-capacity model",
        "an oracle causal-message value (measured by counterfactual delivery) must beat a centralized matched-capacity control",
        "genuine multi-view or multi-agent tasks with disjoint views",
        "construct a counterfactual-delivery oracle; if it does not beat the centralized control, no headroom (needs a genuine causal multi-view bed)"),
    "C5_E1_relational_roles": thesis(
        "E1 event boundaries did not beat simple change detectors: relational structure was not captured",
        "model temporal ROLE changes and relations between entities (not local novelty); a boundary marks a change in the relational configuration, not a magnitude change",
        "Gen2 E1 used a relational transition surprise that still reduced to a per-frame novelty signal",
        "fixed windows, novelty, prediction-error, and change-point detectors",
        "an oracle relational-boundary set must improve downstream value beyond the best change detector by SESOI on a task where relations, not magnitudes, define events",
        "sessions with genuine relational structure (multi-entity streams)",
        "build a multi-entity stream where change detectors provably fail; if oracle relational boundaries do not beat them, no headroom (needs a new relational bed)"),
}

# 10-criterion ranking (1-5; falsification-cost and relabeling-risk are scored so higher = better)
SCORES = {
    "C1_P1R_priority": {"novel_causal_premise": 3, "distance_from_predecessor": 3, "oracle_headroom_est": 3,
                        "data_quality": 5, "independent_unit_availability": 4, "established_control_strength": 5,
                        "falsification_cost_inv": 5, "implementation_independence": 3, "potential_downstream_value": 4,
                        "relabeling_risk_inv": 2},
    "C2_V1_capable_family": {"novel_causal_premise": 2, "distance_from_predecessor": 2, "oracle_headroom_est": 3,
                             "data_quality": 5, "independent_unit_availability": 4, "established_control_strength": 4,
                             "falsification_cost_inv": 5, "implementation_independence": 3, "potential_downstream_value": 3,
                             "relabeling_risk_inv": 2},
    "C3_S1_model_error_aware": {"novel_causal_premise": 5, "distance_from_predecessor": 5, "oracle_headroom_est": 2,
                                "data_quality": 2, "independent_unit_availability": 3, "established_control_strength": 4,
                                "falsification_cost_inv": 3, "implementation_independence": 3, "potential_downstream_value": 3,
                                "relabeling_risk_inv": 4},
    "C4_M1_causal_message": {"novel_causal_premise": 4, "distance_from_predecessor": 4, "oracle_headroom_est": 2,
                             "data_quality": 2, "independent_unit_availability": 2, "established_control_strength": 3,
                             "falsification_cost_inv": 2, "implementation_independence": 3, "potential_downstream_value": 3,
                             "relabeling_risk_inv": 3},
    "C5_E1_relational_roles": {"novel_causal_premise": 4, "distance_from_predecessor": 4, "oracle_headroom_est": 2,
                               "data_quality": 2, "independent_unit_availability": 3, "established_control_strength": 3,
                               "falsification_cost_inv": 2, "implementation_independence": 3, "potential_downstream_value": 3,
                               "relabeling_risk_inv": 4},
}


def eight_gates(cid, precompute):
    pc = precompute.get(cid, {})
    designs = {
        "C1_P1R_priority": {
            "1_causal_hypothesis": "Using predicted replay value as a soft sampling priority over a fixed class-balanced (GDumb) buffer improves final retention beyond uniform GDumb sampling under matched memory and compute.",
            "2_null": "Value-weighted sampling over the GDumb buffer does not beat uniform GDumb sampling (a tie is a null).",
            "3_strongest_established_method": "GDumb (uniform sampling over a class-balanced buffer); also reservoir, uniform, loss-sampling.",
            "4_oracle": "An oracle sampling priority that weights each buffer item by its true retention benefit (measured reduction in end-of-stream forgetting).",
            "5_residual_oracle_headroom": pc.get("gate5_verdict", "pending") + f" (oracle-priority minus uniform GDumb = {pc.get('priority_headroom_over_gdumb_uniform')})",
            "6_controls": "positive control: oracle priority should beat uniform; negative control: random priority must not beat uniform; shuffled-priority must not beat uniform.",
            "7_power_analysis": "with per-task retention units and 5 seeds, min detectable effect approx 0.05 at SESOI 0.05; scale seeds until the CI half-width < SESOI.",
            "8_independent_units": "class-incremental tasks (retention), across seeds; report per-task seed-averaged effects.",
            "9_cheapest_falsification": CANDIDATES[cid]["cheapest_falsification"],
        },
        "C2_V1_capable_family": {
            "1_causal_hypothesis": "Verification value is decodable by a sufficiently expressive estimator family; an expanded capable family robustly decodes it beyond the best simple uncertainty control.",
            "2_null": "No expanded-family gain: fewer than two capable estimators clear the group rule (a tie is a null).",
            "3_strongest_established_method": "raw uncertainty / entropy / margin / loss (the U1 frozen negative) and always/never/random verify.",
            "4_oracle": "the measured verification-decision gain (base wrong and verifier correct) per item.",
            "5_residual_oracle_headroom": pc.get("gate5_verdict", "pending") + f" ({pc.get('n_passing_of_family')} of family clear the group rule; per-estimator lcb {pc.get('per_estimator_incremental_lcb')})",
            "6_controls": "positive: kernel_ridge (the Gen2 passer) must still pass; negative: shuffled-target and rate-matched-random must fail.",
            "7_power_analysis": "6 regimes as units; min detectable effect approx 0.06; add regimes or seeds if the family lcb straddles SESOI.",
            "8_independent_units": "CIFAR-10 distribution regimes (group-disjoint).",
            "9_cheapest_falsification": CANDIDATES[cid]["cheapest_falsification"],
        },
    }
    return designs.get(cid)


def main():
    pc_path = OUT / "MOP_GENERATION3_PRECOMPUTE.json"
    precompute = json.loads(pc_path.read_text()) if pc_path.exists() else {}
    ranked = sorted(SCORES, key=lambda c: -sum(SCORES[c].values()))
    totals = {c: sum(SCORES[c].values()) for c in SCORES}
    # selection is CONDITIONED on the measured gate-5 headroom, not asserted: a candidate is selected for
    # implementation only if its cheap precompute shows headroom; a no_headroom verdict falsifies it at the gate.
    gate5 = {c: (precompute.get(c, {}).get("gate5_verdict", "pending")) for c in ["C1_P1R_priority", "C2_V1_capable_family"]}
    selected = [c for c in ["C1_P1R_priority", "C2_V1_capable_family"] if gate5.get(c) == "headroom_present"]
    falsified_at_gate = [c for c in ["C1_P1R_priority", "C2_V1_capable_family"] if gate5.get(c) == "no_headroom"]
    authority = {
        "schema": "mop-generation3-discovery-authority/v1",
        "principle": "Generation 3 begins with competing scientific premises derived from the null map, not a large run.",
        "candidates": CANDIDATES,
        "ranking_criteria": list(next(iter(SCORES.values())).keys()),
        "ranking_scores": SCORES, "ranking_totals": totals, "ranked_order": ranked,
        "gate5_verdicts": gate5,
        "selected_for_implementation": selected,
        "falsified_at_precompute_gate": falsified_at_gate,
        "selection_rationale": ("C1 and C2 were the two candidates with cheaply measurable gate-5 headroom on "
                                "existing beds. Selection is CONDITIONED on that measurement: only a candidate whose "
                                "cheap precompute shows headroom is selected for implementation. A no_headroom "
                                "verdict falsifies the candidate at the gate, exactly as intended. C3 (model-error-"
                                "aware simulation) is the highest-novelty candidate but its gate-5 needs a new gym-"
                                "with-error harness, so it is deferred as the immediate next thesis to build."),
        "deferred_but_ranked": ["C3_S1_model_error_aware (highest novelty; build the error-aware gym harness next)",
                                "C4_M1_causal_message", "C5_E1_relational_roles"],
        "precompute_designs": {c: eight_gates(c, precompute) for c in ["C1_P1R_priority", "C2_V1_capable_family"]},
        "precompute_results": precompute,
        "no_principal_run_launched": True,
        "mandatory_admission_clauses": "every candidate must additionally satisfy the null-derived constraints (MOP_NULL_DERIVED_CONSTRAINTS.json).",
    }
    authority["sha256"] = sha(authority)
    (W / "frontier/MOP_GENERATION3_DISCOVERY_AUTHORITY.json").write_text(json.dumps(authority, indent=2))
    print("Gen3 authority sealed. ranked:", ranked)
    print("selected:", selected, "| totals:", {c: totals[c] for c in ranked})
    for c in selected:
        g5 = authority["precompute_designs"][c]["5_residual_oracle_headroom"]
        print(f"  {c} gate5: {g5}")


if __name__ == "__main__":
    main()
