"""The historical defect ledger and the defect veto rule.

This module is the single source of truth for what went wrong before. The acceptance gate injects every
entry here as a mutation, the regression tests import it, and the self improving loop appends to it. A
defect class that is not in this list is a defect class the method does not yet catch.

The veto rule: a reproduced defect outranks any number of reviewer votes. Consensus is evidence, not proof.

House style: no dashes.
"""

from __future__ import annotations

SUBSTANTIATION_FIELDS = (
    "path",
    "condition",
    "reproduction",
    "expected",
    "actual",
    "consequence",
)

# id, title, declared, actual, rule, detector, catchable_before_principal
LEDGER = [
    {
        "id": "D1",
        "title": "order free control consumed temporal order",
        "declared": "order free temporal control",
        "actual": "the control contained a Conv1d with kernel 5 and therefore still consumed temporal order",
        "consequence": "the inherited temporal headroom interpretation was invalid",
        "rule": "a control must be proven to remove the information or capability it claims to remove",
        "detector": "mop.method.controls.order_free",
        "mutation": "temporal_conv_in_order_free_control",
        "stage_caught": "control_semantic_proof",
    },
    {
        "id": "D2",
        "title": "replay buffer stopped admitting items",
        "declared": "continual replay",
        "actual": "the buffer stopped admitting items after filling, so lstm and lstm_gdumb resolved to one policy",
        "consequence": "two named arms were the same experiment run twice",
        "rule": "an experimental mechanism must be proven causally active, not merely instantiated or named",
        "detector": "mop.method.controls.replay_active",
        "mutation": "inactive_replay",
        "stage_caught": "mechanism_activity_proof",
    },
    {
        "id": "D3",
        "title": "within domain runs never crossed a context boundary",
        "declared": "continual learning across contexts",
        "actual": "the within domain runs never crossed a context boundary, so replay had nothing to replay",
        "consequence": "a continual learning null was measured on a task that was not continual",
        "rule": "a task being sequentialized does not make it continual",
        "detector": "mop.method.controls.replay_active",
        "mutation": "buffer_that_stops_replacing",
        "stage_caught": "mechanism_activity_proof",
    },
    {
        "id": "D4",
        "title": "arm aliasing",
        "declared": "separate experimental arms",
        "actual": "multiple arms shared implementations, defaults or behaviour",
        "consequence": "comparisons between identical things were reported as comparisons between policies",
        "rule": "every principal arm requires machine verifiable implementation, state transition, parameter update, resource use and output distinctness",
        "detector": "mop.method.arms.distinctness",
        "mutation": "aliased_lstm_and_lstm_gdumb",
        "stage_caught": "arm_distinctness_proof",
    },
    {
        "id": "D5",
        "title": "causal variable with no implementation path",
        "declared": "memory_state and H.norm causal effects",
        "actual": "one had no causal implementation path and one was a phantom alias",
        "consequence": "a reported cause could not have caused anything",
        "rule": "every reported causal variable must bind to a real implementation path and a measurable intervention",
        "detector": "mop.method.graph.validate",
        "mutation": "phantom_parameter_group",
        "stage_caught": "causal_graph_validation",
    },
    {
        "id": "D6",
        "title": "analytic quantity reported as measured",
        "declared": "measured zero forgetting for domain local groups",
        "actual": "the zero was true by construction from parameter partitioning and was never measured",
        "consequence": "a structural guarantee was presented as an empirical finding",
        "rule": "reports must distinguish measured, recomputed, derived, analytic, assumed and structurally guaranteed quantities",
        "detector": "mop.method.contracts.Quantity",
        "mutation": "analytic_value_marked_measured",
        "stage_caught": "causal_graph_validation",
    },
    {
        "id": "D7",
        "title": "report read a nonexistent key",
        "declared": "an answer to Q13",
        "actual": "the code read a key that did not exist and returned None",
        "consequence": "a terminal question was answered with nothing and looked answered",
        "rule": "every report value requires schema validation and evidence path resolution",
        "detector": "mop.method.report.resolve",
        "mutation": "missing_report_key",
        "stage_caught": "report_integrity",
    },
    {
        "id": "D8",
        "title": "baseline identity mismatch",
        "declared": "effect versus LSTM plus GDumb",
        "actual": "the effect was computed against a different baseline",
        "consequence": "the sentence and the number disagreed about what was compared",
        "rule": "baseline identities must be explicit, immutable and verified by name, implementation, configuration and receipt",
        "detector": "mop.method.baseline.comparison",
        "mutation": "wrong_baseline_comparison",
        "stage_caught": "report_integrity",
    },
    {
        "id": "D9",
        "title": "verdict softening in prose",
        "declared": "a summary interpretation",
        "actual": "the summary softened the sealed invalid_no_temporal_headroom verdict to the word marginal",
        "consequence": "the human readable layer contradicted the machine classification it summarized",
        "rule": "human readable synthesis may never broaden, soften or substitute for the sealed machine classification",
        "detector": "mop.method.report.wording_check",
        "mutation": "softened_verdict_wording",
        "stage_caught": "report_integrity",
    },
    {
        "id": "D10",
        "title": "adversarial panel refuted genuine defects",
        "declared": "the panel refuted all attacks",
        "actual": "several refuted attacks were genuine reproducible defects",
        "consequence": "consensus overrode reproduction",
        "rule": "concrete reproducible defects have veto authority over reviewer votes",
        "detector": "mop.method.defects.adjudicate",
        "mutation": "reviewer_consensus_overrides_reproduction",
        "stage_caught": "adjudication",
    },
    {
        "id": "D11",
        "title": "coverage shortfall left implicit",
        "declared": "test coverage targets of 92 and 82 percent",
        "actual": "statement coverage 68.9 percent and branch coverage 56.0 percent",
        "consequence": "the gap between claimed and actual verification was not visible in the authority",
        "rule": "coverage misses must remain visible and test scopes may not be narrowed to claim compliance",
        "detector": "mop.method.gate.coverage_gate",
        "mutation": "narrowed_coverage_scope",
        "stage_caught": "acceptance",
    },
    {
        "id": "D12",
        "title": "ignored treatment flag",
        "declared": "a treatment arm controlled by a configuration flag",
        "actual": "the flag was read into a configuration object and never reached the implementation",
        "consequence": "treatment and control ran the same code",
        "rule": "every load bearing configuration field must be proven to change a runtime trace",
        "detector": "mop.method.arms.config_sensitivity",
        "mutation": "ignored_treatment_flag",
        "stage_caught": "arm_distinctness_proof",
    },
    {
        "id": "D13",
        "title": "future information reached a decision time mechanism",
        "declared": "a decision made from information available at the time",
        "actual": "a statistic computed over the whole sequence entered a per step decision",
        "consequence": "the mechanism could not be deployed and its effect was not attributable",
        "rule": "no future information may enter a decision time mechanism",
        "detector": "mop.method.graph.validate",
        "mutation": "future_information_leakage",
        "stage_caught": "causal_graph_validation",
    },
    {
        "id": "D14",
        "title": "headroom authority from two seeds",
        "declared": "measured oracle headroom",
        "actual": "the headroom rested on two seeds, inside its own noise",
        "consequence": "mechanism development was licensed by an estimate that could not support it",
        "rule": "no two seed headroom authority and no architecture development without stable residual headroom",
        "detector": "mop.method.contracts.OracleContract",
        "mutation": "two_seed_false_headroom",
        "stage_caught": "oracle_headroom",
    },
    {
        "id": "D15",
        "title": "unconverged baseline produced a verdict",
        "declared": "a comparison against a strong baseline",
        "actual": "the baseline had not plateaued when the comparison was taken",
        "consequence": "the treatment was flattered by the baseline's remaining training headroom",
        "rule": "a comparison against an unconverged baseline is provisional and cannot be terminal",
        "detector": "mop.method.baseline.comparison",
        "mutation": "unconverged_baseline",
        "stage_caught": "baseline_convergence",
    },
    {
        "id": "D16",
        "title": "a context split that crosses no boundary",
        "declared": "two contexts with an adaptation phase between them",
        "actual": "two random unit groups from one corpus, so adapting to the second improved the first as well",
        "consequence": "an adaptation experiment with no stability plasticity tradeoff to measure, and an "
                       "oracle headroom below the smallest effect the design declared interesting",
        "rule": "a context boundary must be proven by measurement: the new context must be measurably harder "
                "before adaptation, and adapting to it must cost something on the old one",
        "detector": "mop.method.bed.context_boundary",
        "mutation": "context_split_that_crosses_no_boundary",
        "stage_caught": "bed_validity",
        "discovered_by": "mop-experimental-method-reformation-v1, E4 scout on speech_stream",
        "discovered_in_this_program": True,
    },
    {
        "id": "D17",
        "title": "a brittle plateau criterion reported a flat curve as still improving",
        "declared": "a convergence receipt",
        "actual": "on a noisy single seed validation curve the argmax lands late by chance, so a curve that "
                  "moved 0.016 across a tenfold budget range was classified unconverged",
        "consequence": "a comparison whose baselines were converged in substance would have been sealed as "
                       "provisional, which understates the evidence exactly as badly as overstating it",
        "rule": "a convergence criterion must answer whether training headroom remains, and when two "
                "criteria disagree both are reported and the one used is named",
        "detector": "mop.method.baseline.plateau",
        "mutation": "brittle_plateau_criterion",
        "stage_caught": "baseline_convergence",
        "discovered_by": "mop-experimental-method-reformation-v1, E1 scout convergence curves",
        "discovered_in_this_program": True,
    },
    {
        "id": "D18",
        "title": "a label permutation control scored against zero difference instead of the majority class rate",
        "declared": "with permuted labels no arm may separate from another",
        "actual": "two architectures with nothing to learn degenerate differently. The pooled control "
                  "collapsed onto one class and scored below the majority class rate while the recurrent core "
                  "spread its predictions and scored at it, leaving a 0.047 difference that is not signal",
        "consequence": "a sound positive would have been downgraded to provisional by an artifact of how the "
                       "control degenerates",
        "rule": "a label permutation control is scored against the majority class rate, not against a zero "
                "difference between arms",
        "detector": "mop.method.runs.mutations.e1_mutations",
        "mutation": "label_permutation_scored_against_zero",
        "stage_caught": "mutation_attacks",
        "discovered_by": "mop-experimental-method-reformation-v1, E1 positive mutation suite on har_stream",
        "discovered_in_this_program": True,
    },
]

BY_ID = {d["id"]: d for d in LEDGER}
MUTATIONS = [d["mutation"] for d in LEDGER]


# ---------------------------------------------------------------- the veto rule


def substantiated(report: dict) -> bool:
    return all(report.get(f) for f in SUBSTANTIATION_FIELDS)


def adjudicate(attack: dict, votes: list[dict], reproduction: dict | None) -> dict:
    """Decide the status of one adversarial attack.

    reproduction, when supplied, is the reconciliation role's rerun: {"reproduced": bool, "evidence": ...}.
    A reproduced defect is confirmed no matter how the panel voted. Votes only matter when nothing was
    reproduced, and even then they produce unresolved rather than refuted unless a reproduction attempt ran.
    """
    refute = sum(1 for v in votes if v.get("verdict") == "refuted")
    confirm = sum(1 for v in votes if v.get("verdict") == "confirmed")
    if reproduction and reproduction.get("reproduced"):
        if not substantiated(attack):
            return {
                "status": "defect_confirmed",
                "authority": "reproduction",
                "note": "reproduced despite an incomplete substantiation record",
                "votes": {"confirmed": confirm, "refuted": refute},
            }
        return {
            "status": "defect_confirmed",
            "authority": "reproduction",
            "votes": {"confirmed": confirm, "refuted": refute},
            "vote_overridden": refute > confirm,
        }
    if reproduction and reproduction.get("reproduced") is False:
        return {
            "status": "refuted_by_reproduction",
            "authority": "reproduction",
            "votes": {"confirmed": confirm, "refuted": refute},
        }
    return {
        "status": "unresolved_no_reproduction_attempt",
        "authority": "none",
        "votes": {"confirmed": confirm, "refuted": refute},
        "note": "votes alone cannot refute an attack",
    }


def required_followups(defect_id: str) -> list[str]:
    return [
        "freeze the original result",
        "add a permanent regression test",
        "open a bounded repair authority",
        "produce the repaired result",
        "write the consequence analysis",
        "revalidate every dependent artifact",
    ]
