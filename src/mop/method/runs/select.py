"""Experiment selection by measured information value, over a hypothesis graph rather than a chain.

The scores below are judgements, and they are written down so that a wrong one is visibly wrong. Two of them
are not judgements: the oracle headroom entry for the self supervised candidate is zero because nobody has
measured it, and the closed premise risk for the cross modality candidates is high because five programs
have now returned the same null on that premise.

House style: no dashes.
"""

from __future__ import annotations

import json

from mop.method import hypothesis, io, voi

REAUDIT = "MOP_FAST_STATE_REAUDIT.json"

HYPOTHESES = [
    {
        "id": "H_fast_state",
        "premise": "a fast recurrent state is the component that carries the useful temporal capability",
        "predecessor": None,
        "support": [
            "inherited timescale ablation: fast state is the only timescale with a positive contribution, +0.063"
        ],
        "contradictions": [
            "cross domain transfer of the shared fast core is null in both directions",
            "the within domain battery is null, but only on beds that do not require dynamics",
        ],
        "required_bed": "a bed sealed temporal_headroom_present",
        "required_headroom": "residual headroom over the strongest order free and readout matched control",
        "strongest_baseline": "matched capacity conventional recurrent model",
        "cheapest_falsifier": "a core by readout factorial on a valid temporal bed",
        "dependent_hypotheses": ["H_readout_capacity", "H_shared_core_capacity"],
        "requires_premise_of": [],
        "state": "instrument_pending",
    },
    {
        "id": "H_readout_capacity",
        "premise": "the measured advantage of the owned substrate comes from readout capacity, not from state",
        "predecessor": "H_fast_state",
        "support": ["never separated: every prior arm changed core and readout together"],
        "contradictions": [],
        "required_bed": "a bed sealed temporal_headroom_present",
        "required_headroom": "any measurable difference between the two factors",
        "strongest_baseline": "simple core with strong readout",
        "cheapest_falsifier": "the same factorial, read along the readout axis",
        "dependent_hypotheses": [],
        "requires_premise_of": [],
        "state": "unopened",
    },
    {
        "id": "H_shared_core_capacity",
        "premise": "shared core capacity moves the acquisition retention frontier rather than scaling both",
        "predecessor": "H_fast_state",
        "support": [],
        "contradictions": ["matched capacity conventional baselines were never beaten"],
        "required_bed": "a bed sealed temporal_headroom_present with two contexts",
        "required_headroom": "a frontier that moves, not two curves that scale together",
        "strongest_baseline": "matched capacity separate per context models",
        "cheapest_falsifier": "three capacities against matched separate models",
        "dependent_hypotheses": [],
        "requires_premise_of": [],
        "state": "unopened",
    },
    {
        "id": "H_domain_specific_representation",
        "premise": "no transferable temporal structure exists because representations are domain specific",
        "predecessor": None,
        "support": ["cross domain null in both directions", "secondary matrix null on two strongly temporal beds"],
        "contradictions": [],
        "required_bed": "two beds from different modalities",
        "required_headroom": "an oracle that shows any transferable component at all",
        "strongest_baseline": "fresh independent per domain",
        "cheapest_falsifier": "causal parameter group interventions on the shared groups",
        "dependent_hypotheses": [],
        "requires_premise_of": [],
        "state": "supported",
    },
    {
        "id": "H_interference",
        "premise": "shared updates interfere destructively and that interference is the binding constraint",
        "predecessor": None,
        "support": ["the interference map attributes forgetting to the shared groups"],
        "contradictions": ["simple fixed partitioning already removes it and does not produce a gain"],
        "required_bed": "a bed with two contexts and a return phase",
        "required_headroom": "a partition oracle above SESOI",
        "strongest_baseline": "simple fixed partition",
        "cheapest_falsifier": "adaptation locus factorial: state only, head only, adapter only, core, full",
        "dependent_hypotheses": [],
        "requires_premise_of": [],
        "state": "mixed",
    },
    {
        "id": "H_bed_insufficiency",
        "premise": "the nulls are explained by beds that never required the capability under test",
        "predecessor": None,
        "support": [
            "har and speech carry the sealed verdict invalid_no_temporal_headroom",
            "the principal matrix and the within domain battery ran on exactly those beds",
        ],
        "contradictions": ["the secondary matrix is null on two beds sealed temporal_headroom_present"],
        "required_bed": "a bed sealed temporal_headroom_present",
        "required_headroom": "not applicable: this hypothesis is about the bed, not the mechanism",
        "strongest_baseline": "the same arms on a valid bed",
        "cheapest_falsifier": "rerun the unmeasured within domain cell on har_stream and speech_stream",
        "dependent_hypotheses": ["H_fast_state"],
        "requires_premise_of": [],
        "state": "admitted",
    },
]

CANDIDATES = [
    {
        "id": "E1",
        "title": "fast core versus readout, factorial, on the two sealed valid temporal beds",
        "question": (
            "does the within domain capability come from the recurrent core, from the readout, or from their "
            "interaction, on a bed that actually requires temporal order"
        ),
        "hypotheses_separated": ["H_fast_state", "H_readout_capacity", "H_bed_insufficiency"],
        "scores": {
            "expected_information_gain": 0.95,
            "probability_of_changing_the_substrate_decision": 0.8,
            "compute_cost": 0.35,
            "duration_cost": 0.3,
            "instrumentation_risk": 0.2,
            "baseline_uncertainty": 0.2,
            "oracle_headroom": 0.7,
            "independent_unit_quality": 0.9,
            "reusability_of_implementation": 0.9,
            "reusability_of_data": 1.0,
            "discriminates_competing_hypotheses": 0.95,
            "risk_of_repeating_a_closed_premise": 0.05,
        },
        "justification": {
            "expected_information_gain": "the prior program named this exact question unresolved and never separated the two factors",
            "risk_of_repeating_a_closed_premise": "the closed premise is cross modality transfer; this is within domain on a different bed",
            "oracle_headroom": "the stream beds are sealed temporal_headroom_present, so an order free reader provably cannot solve them",
            "compute_cost": "four cells plus controls on beds the forge already ran at this size",
        },
    },
    {
        "id": "E2",
        "title": "shared core capacity scaling against matched separate models",
        "question": "does more shared core capacity move the acquisition retention frontier or scale both",
        "hypotheses_separated": ["H_shared_core_capacity", "H_interference"],
        "scores": {
            "expected_information_gain": 0.6,
            "probability_of_changing_the_substrate_decision": 0.45,
            "compute_cost": 0.8,
            "duration_cost": 0.75,
            "instrumentation_risk": 0.25,
            "baseline_uncertainty": 0.3,
            "oracle_headroom": 0.4,
            "independent_unit_quality": 0.9,
            "reusability_of_implementation": 0.8,
            "reusability_of_data": 1.0,
            "discriminates_competing_hypotheses": 0.6,
            "risk_of_repeating_a_closed_premise": 0.3,
        },
        "justification": {
            "compute_cost": "three capacities times two beds times two contexts times eight seeds, the largest of the five",
            "oracle_headroom": "unmeasured on the frontier framing; the matched capacity baseline was never beaten at any size tried",
        },
    },
    {
        "id": "E3",
        "title": "domain local versus shared representation by causal parameter group intervention",
        "question": "which representation components carry transferable temporal structure",
        "hypotheses_separated": ["H_domain_specific_representation", "H_interference"],
        "scores": {
            "expected_information_gain": 0.5,
            "probability_of_changing_the_substrate_decision": 0.25,
            "compute_cost": 0.5,
            "duration_cost": 0.45,
            "instrumentation_risk": 0.35,
            "baseline_uncertainty": 0.3,
            "oracle_headroom": 0.2,
            "independent_unit_quality": 0.9,
            "reusability_of_implementation": 0.85,
            "reusability_of_data": 1.0,
            "discriminates_competing_hypotheses": 0.5,
            "risk_of_repeating_a_closed_premise": 0.55,
        },
        "justification": {
            "risk_of_repeating_a_closed_premise": "this is the cross modality transfer premise again, now null on five programs including two strongly temporal beds",
            "oracle_headroom": "the transfer oracle is already measured near zero",
        },
    },
    {
        "id": "E4",
        "title": "adaptation locus: state only, head only, adapter only, core, full",
        "question": "can useful fast adaptation occur in owned state without changing shared slow parameters",
        "hypotheses_separated": ["H_fast_state", "H_interference", "H_domain_specific_representation"],
        "scores": {
            "expected_information_gain": 0.8,
            "probability_of_changing_the_substrate_decision": 0.6,
            "compute_cost": 0.2,
            "duration_cost": 0.2,
            "instrumentation_risk": 0.3,
            "baseline_uncertainty": 0.25,
            "oracle_headroom": 0.6,
            "independent_unit_quality": 0.9,
            "reusability_of_implementation": 0.9,
            "reusability_of_data": 1.0,
            "discriminates_competing_hypotheses": 0.8,
            "risk_of_repeating_a_closed_premise": 0.15,
        },
        "justification": {
            "compute_cost": "adaptation is a short phase on top of one shared pretrained checkpoint per seed, the cheapest of the five",
            "oracle_headroom": "the update partition oracle is positive with a positive lower bound, 0.0052 and 0.0129 by direction, though below SESOI",
            "discriminates_competing_hypotheses": "the five loci map one to one onto three competing explanations of the nulls",
        },
    },
    {
        "id": "E5",
        "title": "self supervised temporal state pretraining",
        "question": "does self supervised temporal training improve adaptation after controlling for compute and readout",
        "hypotheses_separated": ["H_fast_state"],
        "scores": {
            "expected_information_gain": 0.55,
            "probability_of_changing_the_substrate_decision": 0.35,
            "compute_cost": 0.9,
            "duration_cost": 0.85,
            "instrumentation_risk": 0.6,
            "baseline_uncertainty": 0.4,
            "oracle_headroom": 0.0,
            "independent_unit_quality": 0.9,
            "reusability_of_implementation": 0.4,
            "reusability_of_data": 1.0,
            "discriminates_competing_hypotheses": 0.4,
            "risk_of_repeating_a_closed_premise": 0.2,
        },
        "justification": {
            "oracle_headroom": "zero because it is unmeasured. The candidate declares itself open only when residual headroom exists, and no such measurement exists, so the queue refuses it rather than guessing",
        },
    },
]


def main():
    ra = io.load(REAUDIT)
    q = voi.queue(CANDIDATES, select=2)
    q["derived_from_reaudit"] = {
        "unmeasured_cell": ra["corrected_claim_ceilings"]["within_domain_battery"]["opens"],
        "why_it_raises_E1": "E1 measures that cell and separates the two factors in the same design",
    }
    io.seal("MOP_EXPERIMENT_VALUE_QUEUE.json", q)

    v = hypothesis.validate(HYPOTHESES)
    if v:
        raise SystemExit(f"hypothesis graph invalid: {v}")
    graph_doc = {
        "schema": "mop-substrate-hypothesis-graph/v1",
        "states": list(hypothesis.STATES),
        "hypotheses": HYPOTHESES,
        "summary": hypothesis.summarize(HYPOTHESES),
        "rule": "a null closes only the descendants that require the failed premise; independent hypotheses continue",
    }
    io.seal("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json", graph_doc)

    sel = [c for c in q["selected_detail"]]
    io.seal(
        "MOP_SELECTED_EXPERIMENTS.json",
        {
            "schema": "mop-selected-experiments/v1",
            "selected": [s["id"] for s in sel],
            "detail": sel,
            "pair_covers_hypotheses": q["hypotheses_covered_by_selection"],
            "pair_rationale": (
                "E1 splits the core from the readout on a bed that requires order, which is the only way to "
                "tell the fast state hypothesis from the readout capacity hypothesis. E4 splits the locus of "
                "adaptation, which is the only way to tell interference from domain specific representation "
                "while holding the core fixed. Neither can produce the same verdict under both explanations."
            ),
            "predictions_required_before_execution": True,
        },
    )

    rows = "\n".join(
        f"| {c['id']} | {c['title']} | {c['decision_information']} | {c['cost_index']} | {c['priority']} | {c['status']} |"
        for c in q["candidates"]
    )
    hrows = "\n".join(f"| {h['id']} | {h['state']} | {h['cheapest_falsifier']} |" for h in HYPOTHESES)
    io.seal_md(
        "MOP_EXPERIMENT_VALUE_QUEUE.md",
        f"""# Experiment value queue

{q["formula"]}

| id | experiment | decision information | cost index | priority | status |
|---|---|---|---|---|---|
{rows}

Selected: {", ".join(q["selected"])}. Refused: {", ".join(q["refused"]) or "none"}.

{q["caveat"]}

## Why these two

{json.loads(json.dumps(io.load("MOP_SELECTED_EXPERIMENTS.json")))["pair_rationale"] if io.exists("MOP_SELECTED_EXPERIMENTS.json") else ""}
""",
    )
    io.seal_md(
        "MOP_SUBSTRATE_HYPOTHESIS_GRAPH.md",
        f"""# Substrate hypothesis graph

| hypothesis | state | cheapest falsifier |
|---|---|---|
{hrows}

Open: {", ".join(hypothesis.summarize(HYPOTHESES)["open"])}.

{graph_doc["rule"]}
""",
    )
    print(f"selected {q['selected']} | refused {q['refused']} | open {hypothesis.summarize(HYPOTHESES)['open']}", flush=True)
    print("SELECT_DONE", flush=True)


if __name__ == "__main__":
    main()
