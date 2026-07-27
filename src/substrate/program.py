"""The Substrate master program: every master plan requirement as a durable item with a derived status.

The one rule that shapes this module is that status is computed from the repository, never asserted. An
item is implemented because its files exist, tested because a recorded test ledger says its declared tests
passed, measured because its evidence artifacts are sealed, and terminal because a result ledger records a
scientific classification for it. Nothing here can claim progress that the tree does not contain, which is
the same discipline the experimental method kernel applies to experiments.

Evidence percentages are therefore not derived from implementation. Code existing raises implementation and
nothing else. The master plan says evidence remains earned, and this is where that is enforced.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from substrate import evidence as io
from substrate.compat import mop as historical_mop

PLAN = io.ROOT / "docs" / "LONG_RUN_PLAN.md"

# proof roots of the historical authorities Substrate inherits from, never rewritten by this program
PROOF_ROOTS = {
    "temporal": historical_mop.roots(io.ROOT)["temporal"],
    "method": historical_mop.roots(io.ROOT)["method"],
    "": io.PROOF,
}

STATUS_LADDER = ("not_started", "partial", "implemented", "tested", "measured", "terminal")

# section 43 of the final autonomous program, the twenty seven scorecard categories. This supersedes the
# eighteen of the earlier master plan; the four renamed ones are listed so the change is traceable rather
# than silent.
CATEGORIES = (
    "temporal_continuity",
    "ontology",
    "epistemology",
    "workspace",
    "working_memory",
    "episodic_memory",
    "semantic_memory",
    "procedural_memory",
    "world_model",
    "self_model",
    "perspective_diversity",
    "perspective_arbitration",
    "metacognition",
    "grounding",
    "causal_reasoning",
    "goal_continuity",
    "valuation",
    "plasticity",
    "consolidation",
    "reorganization",
    "developmental_divergence",
    "thinking",
    "continuity",
    "unity",
    "reflective_access",
    "cognitive_integrity",
    "model_body_integration",
)

SUPERSEDED_CATEGORIES = {
    "arbitration": "perspective_arbitration",
    "transfer": "model_body_integration",
    "developmental_learning": "developmental_divergence",
    "unified_cognition": "unity",
}

# section 4 of the master plan, frozen as the entry baseline so later movement is attributable
BASELINE_2026_07_27 = {
    "experimental_validity_and_falsification": (95, 95),
    "campaign_orchestration": (90, 90),
    "failure_and_null_understanding": (90, 90),
    "owned_temporal_substrate_implementation": (80, 60),
    "minimal_temporal_core_identification": (50, 40),
    "working_memory": (25, 10),
    "episodic_and_semantic_memory": (20, 5),
    "useful_plasticity": (20, 10),
    "world_model": (15, 5),
    "self_model": (5, 0),
    "perspective_generation": (20, 10),
    "perspective_arbitration": (15, 5),
    "metacognition": (10, 5),
    "functional_self_reorganization": (5, 0),
    "unified_cross_domain_entity": (10, 5),
    "developmental_artificial_cognition": (5, 0),
    "overall_substrate_vision": (35, 20),
}


@dataclass(frozen=True)
class Item:
    id: str
    section: str
    title: str
    requirement: str
    kind: str = "implementation"  # implementation, evidence, boundary, authority
    category: str | None = None
    impl: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    batch: int = 0


def _i(*a, **k) -> Item:
    return Item(*a, **k)


COG = "src/substrate"
T_SYNTH = "temporal:MOP_TEMPORAL_CORE_SYNTHESIS.json"

ITEMS: tuple[Item, ...] = (
    # ---------------------------------------------------------------- authorities and boundaries
    _i(
        "A1",
        "2",
        "Naming and historical continuity authority",
        "Map historical terminology to the current architecture without invalidating prior evidence. "
        "Historical files, branches, commits, tags and proofs are not mass renamed.",
        kind="authority",
        impl=(f"{COG}/program.py",),
        tests=("tests/substrate/test_program.py::test_naming_authority_preserves_historical_programs",),
        evidence=("SUBSTRATE_MASTER_AUTHORITY.json",),
        batch=1,
    ),
    _i(
        "A2",
        "21",
        "Master deliverable set exists and binds to real things",
        "Maintain the declared master artifacts. Artifacts must bind to implementation, evidence or "
        "terminal gates. Do not create empty placeholder documents.",
        kind="authority",
        impl=(f"{COG}/program.py",),
        tests=("tests/substrate/test_program.py::test_every_deliverable_binds_to_a_real_path",),
        evidence=("SUBSTRATE_MASTER_AUTHORITY.json", "SUBSTRATE_STATE.json"),
        deps=("A1",),
        batch=1,
    ),
    _i(
        "A3",
        "18",
        "Experimental requirements bind every new Substrate experiment",
        "Causal graph, instrument validation, arm distinctness, control semantics, mechanism activity, "
        "valid bed, converged baseline, oracle headroom, power, independent units, independent "
        "verification and mutation testing, all before principal compute.",
        kind="authority",
        impl=(f"{COG}/admission.py", f"{COG}/method/gate.py"),
        tests=(
            "tests/substrate/test_admission.py::test_substrate_experiment_cannot_reach_principal_unproven",
        ),
        evidence=("SUBSTRATE_EXPERIMENTAL_REQUIREMENTS.json",),
        deps=("A2",),
        batch=1,
    ),
    _i(
        "A4",
        "19",
        "Developmental safety envelope",
        "Every adaptation bounded, attributable, checkpointed, reversible where feasible, retention "
        "tested, cross domain tested, objective drift tested and independently verified. Substrate must "
        "not autonomously remove evidence validation, audit systems, claim boundaries, stop switches, "
        "resource limits, rollback or adaptation constraints.",
        kind="boundary",
        impl=(f"{COG}/safety.py",),
        tests=("tests/substrate/test_safety.py::test_protected_surfaces_cannot_be_removed_by_adaptation",),
        evidence=("SUBSTRATE_DEVELOPMENTAL_SAFETY.json",),
        deps=("A2",),
        batch=1,
    ),
    _i(
        "A5",
        "16",
        "Sentience research boundary",
        "Separate demonstrated engineering property, behavioural indication, architectural prerequisite, "
        "philosophical interpretation and unsupported claim. Never claim consciousness, sentience, "
        "feelings, wants, suffering, subjective experience or life.",
        kind="boundary",
        impl=(f"{COG}/safety.py",),
        tests=("tests/substrate/test_safety.py::test_forbidden_claim_vocabulary_is_refused",),
        evidence=("SUBSTRATE_SENTIENCE_RESEARCH_BOUNDARY.json",),
        deps=("A2",),
        batch=1,
    ),
    _i(
        "A6",
        "17",
        "Continuous six batch research program",
        "After each terminal batch update the hypothesis graph, null map, capability map and scorecard, "
        "identify the limiting property, estimate value of information and select one primary and one "
        "independent secondary batch.",
        kind="authority",
        impl=(f"{COG}/program.py",),
        tests=("tests/substrate/test_program.py::test_batch_selection_is_dependency_ready_and_independent",),
        evidence=("SUBSTRATE_STATE.json", "SUBSTRATE_NEXT_FRONTIER.json"),
        deps=("A2",),
        batch=1,
    ),
    _i(
        "A7",
        "20",
        "Scorecard separates implementation from evidence",
        "Track implementation and evidence separately for eighteen categories. A positive in one category "
        "must not inflate unrelated categories. Evidence remains earned.",
        kind="authority",
        impl=(f"{COG}/program.py",),
        tests=("tests/substrate/test_program.py::test_evidence_never_rises_from_implementation_alone",),
        evidence=("SUBSTRATE_PROGRESS_SCORECARD.json",),
        deps=("A2",),
        batch=1,
    ),
    # ---------------------------------------------------------------- core architecture
    _i(
        "C1",
        "6.1",
        "Temporal core identified and selected",
        "Identify whether recurrence is necessary, whether explicit history is sufficient, the minimum "
        "useful state horizon, the smallest useful capacity, the simplest sufficient readout, and whether "
        "the effect survives independent implementations and multiple beds.",
        kind="evidence",
        category="temporal_continuity",
        impl=(f"{COG}/temporal_link.py", f"{COG}/compat/mop.py"),
        tests=("tests/temporal/test_temporal_core.py",),
        evidence=(
            T_SYNTH,
            "temporal:MOP_OWNED_TEMPORAL_CORE_V1.json",
            "temporal:MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json",
        ),
        batch=1,
    ),
    _i(
        "C2",
        "6.2",
        "Typed cognitive workspace",
        "Not one unrestricted opaque tensor. Every region declares shape, persistence, timescale, readers, "
        "writers, provenance, confidence, cost, reset behaviour and update rule. Important information "
        "becomes globally available without letting arbitrary components corrupt every region.",
        category="workspace",
        impl=(f"{COG}/workspace.py",),
        tests=("tests/substrate/test_workspace.py",),
        evidence=("SUBSTRATE_WORKSPACE.json",),
        deps=("A2",),
        batch=2,
    ),
    _i(
        "C3",
        "6.3",
        "Mixture of Perspectives, perspectives as processes",
        "Each perspective declares inputs, permitted information, internal state, objective, output type, "
        "confidence, resource cost, failure modes and verification method.",
        category="perspective_diversity",
        impl=(f"{COG}/perspectives.py",),
        tests=("tests/substrate/test_perspectives.py",),
        evidence=("SUBSTRATE_PERSPECTIVE_SYSTEM.json",),
        deps=("C2",),
        batch=2,
    ),
    _i(
        "C4",
        "6.4",
        "Perspective selection ladder",
        "Compare fixed sets, task label rules, simple context rules, retrieval selection, reliability "
        "weighted selection, learned selectors and oracle selectors. A learned selector opens only if "
        "stable residual headroom remains beyond strong simple selection.",
        category="perspective_diversity",
        impl=(f"{COG}/perspectives.py",),
        tests=("tests/substrate/test_perspectives.py::test_learned_selector_stays_closed_without_headroom",),
        evidence=("SUBSTRATE_PERSPECTIVE_SYSTEM.json",),
        deps=("C3",),
        batch=2,
    ),
    _i(
        "C5",
        "6.5",
        "Perspective arbitration",
        "Compare outputs, preserve provenance, detect contradiction, evaluate confidence, weigh historical "
        "reliability, request verification, allocate compute, combine compatible results, preserve minority "
        "hypotheses, defer when evidence is insufficient and choose under a resource budget.",
        category="perspective_arbitration",
        impl=(f"{COG}/perspectives.py",),
        tests=("tests/substrate/test_perspectives.py::test_minority_hypothesis_survives_arbitration",),
        evidence=("SUBSTRATE_ARBITRATION_SYSTEM.json",),
        deps=("C3",),
        batch=2,
    ),
    _i(
        "O1",
        "6",
        "Typed ontology with the nine distinctions it refuses to collapse",
        "Twenty seven types plus unknown, nine declared item fields, and nine distinctions the typing must "
        "not let collapse. Merging is evidenced and reversible; unknown is a first class type that carries "
        "the reason it is unknown.",
        category="ontology",
        impl=(f"{COG}/ontology.py",),
        tests=(
            "tests/substrate/test_ontology.py::test_identity_preservation_over_time",
            "tests/substrate/test_ontology.py::test_mistaken_merge_detection",
            "tests/substrate/test_ontology.py::test_counterfactual_objects_never_merge_with_actual_ones",
            "tests/substrate/test_ontology.py::test_the_self_environment_boundary_holds",
        ),
        evidence=("SUBSTRATE_ONTOLOGY.json",),
        deps=("C2",),
        batch=1,
    ),
    _i(
        "O2",
        "7",
        "Epistemology where justification is a graph, not a confidence table",
        "Twelve epistemic kinds, twelve declared belief fields, and dependency aware revision. Retraction "
        "propagates to every dependant, circular justification is refused, and a claim resting on a "
        "retraction loses whatever its confidence says.",
        category="epistemology",
        impl=(f"{COG}/epistemology.py",),
        tests=(
            "tests/substrate/test_epistemology.py::"
            "test_hidden_dependency_failure_is_caught_by_retraction_propagating",
            "tests/substrate/test_epistemology.py::test_circular_justification_is_refused_not_followed",
            "tests/substrate/test_epistemology.py::"
            "test_the_dependency_aware_policy_beats_a_confident_but_undermined_claim",
        ),
        evidence=("SUBSTRATE_EPISTEMOLOGY.json", "SUBSTRATE_BELIEF_REVISION.json"),
        deps=("O1",),
        batch=1,
    ),
    _i(
        "O3",
        "7.2",
        "Metacognitive control runs on epistemic value, not confidence",
        "A claim at one confidence with nothing at stake and one with everything resting on it get "
        "different actions. Confidence alone cannot separate them, so it is not what chooses.",
        category="epistemology",
        impl=(f"{COG}/epistemology.py",),
        tests=(
            "tests/substrate/test_epistemology.py::"
            "test_epistemic_value_and_not_confidence_chooses_the_action",
        ),
        evidence=("SUBSTRATE_EPISTEMOLOGY.json",),
        deps=("O2", "K1"),
        batch=1,
    ),
    _i(
        "N1",
        "5",
        "The runtime loop that makes the modules one entity",
        "A mature Substrate is the composition, not the parts. One cycle runs perceive, attend, select, "
        "run perspectives, arbitrate, decide, remember, self update, consolidate, adapt and checkpoint, "
        "and every stage leaves a receipt a reflective report can bind to.",
        category="unity",
        impl=(f"{COG}/runtime.py",),
        tests=(
            "tests/substrate/test_runtime.py::test_one_cycle_runs_every_declared_stage",
            "tests/substrate/test_runtime.py::"
            "test_a_skipped_stage_says_why_rather_than_looking_like_one_that_ran",
        ),
        evidence=("SUBSTRATE_RUNTIME.json",),
        deps=("C2", "C5"),
        batch=2,
    ),
    _i(
        "N2",
        "19",
        "The loop has no path to acting on the world",
        "The decision region records what would be done and nothing executes it. Activation stays false "
        "and no code path in the runtime sets it true.",
        kind="boundary",
        impl=(f"{COG}/runtime.py",),
        tests=(
            "tests/substrate/test_runtime.py::test_the_loop_never_acts_on_the_world",
            "tests/substrate/test_runtime.py::"
            "test_a_tampered_checkpoint_is_refused_rather_than_silently_restored",
        ),
        evidence=("SUBSTRATE_RUNTIME.json",),
        deps=("A4", "N1"),
        batch=2,
    ),
    # ---------------------------------------------------------------- memory
    _i(
        "M1",
        "7.1",
        "Working memory",
        "Store current variables, temporary bindings, subgoals, partial "
        "derivations, active hypotheses, recent tool results, contradictions and intermediate plans. "
        "Measure capacity, interference, decay, refresh, prioritization and downstream value.",
        category="working_memory",
        impl=(f"{COG}/memory.py",),
        tests=("tests/substrate/test_memory.py::test_working_memory_capacity_interference_and_decay",),
        evidence=("SUBSTRATE_MEMORY_SYSTEM.json",),
        deps=("C2",),
        batch=3,
    ),
    _i(
        "M2",
        "7.2",
        "Episodic memory",
        "Episodes carry context, observation, internal state, goal, action, "
        "outcome, error, perspectives used, verification, confidence, cost and later usefulness. Separate "
        "recent, compressed, verified, failed, unresolved and quarantined episodes. Generated experiences "
        "cannot become training material automatically.",
        category="episodic_memory",
        impl=(f"{COG}/memory.py",),
        tests=(
            "tests/substrate/test_memory.py::test_generated_episode_cannot_be_promoted_without_verification",
        ),
        evidence=("SUBSTRATE_MEMORY_SYSTEM.json",),
        deps=("C2",),
        batch=3,
    ),
    _i(
        "M3",
        "7.3",
        "Semantic memory",
        "Concepts, facts, relations, rules, abstractions, exceptions, "
        "confidence, provenance and supersession.",
        category="semantic_memory",
        impl=(f"{COG}/memory.py",),
        tests=("tests/substrate/test_memory.py::test_semantic_supersession_preserves_provenance",),
        evidence=("SUBSTRATE_MEMORY_SYSTEM.json",),
        deps=("M2",),
        batch=3,
    ),
    _i(
        "M4",
        "7.4",
        "Procedural memory",
        "Strategies, proof motifs, tool sequences, planning routines, "
        "debugging methods, adaptation procedures and perspective compositions. Procedures require transfer "
        "testing beyond the episodes that created them.",
        category="procedural_memory",
        impl=(f"{COG}/memory.py",),
        tests=("tests/substrate/test_memory.py::test_procedure_requires_transfer_beyond_source_episodes",),
        evidence=("SUBSTRATE_MEMORY_SYSTEM.json",),
        deps=("M2",),
        batch=3,
    ),
    _i(
        "M5",
        "7.5",
        "Consolidation",
        "Compare no consolidation, fixed schedules, boundary triggered, "
        "performance triggered, verification triggered, repetition triggered and oracle consolidation.",
        category="consolidation",
        impl=(f"{COG}/memory.py",),
        tests=("tests/substrate/test_memory.py::test_consolidation_policies_are_distinct_and_ordered",),
        evidence=("SUBSTRATE_MEMORY_SYSTEM.json",),
        deps=("M2", "M3", "M4"),
        batch=3,
    ),
    _i(
        "M6",
        "7.6",
        "Forgetting and hygiene",
        "Decay, supersede, quarantine, archive or delete stale "
        "beliefs, disproven hypotheses, duplicates, unsupported generated claims, corrupted memories, low "
        "value noise and unsafe adaptation proposals, while preserving auditability.",
        category="semantic_memory",
        impl=(f"{COG}/memory.py",),
        tests=("tests/substrate/test_memory.py::test_hygiene_never_destroys_audit_required_records",),
        evidence=("SUBSTRATE_MEMORY_SYSTEM.json",),
        deps=("M3",),
        batch=3,
    ),
    _i(
        "W1",
        "8",
        "World model",
        "Represent entities, relations, events, causes, affordances, agents, "
        "goals, uncertainty, time and counterfactual alternatives. Separate predictive accuracy, decision "
        "usefulness, causal validity and simulation reliability.",
        category="world_model",
        impl=(f"{COG}/world.py",),
        tests=("tests/substrate/test_world.py",),
        evidence=("SUBSTRATE_WORLD_MODEL.json",),
        deps=("C2", "M2"),
        batch=3,
    ),
    _i(
        "S1",
        "9",
        "Self model",
        "Represent measurable internal facts and compare predictions against "
        "actual accuracy, failure probability, time, cost, tool competence, memory confidence, perspective "
        "reliability and task progress. Not only a generated narrative.",
        category="self_model",
        impl=(f"{COG}/selfmodel.py",),
        tests=("tests/substrate/test_selfmodel.py",),
        evidence=("SUBSTRATE_SELF_MODEL.json",),
        deps=("C2", "M2"),
        batch=3,
    ),
    _i(
        "K1",
        "10",
        "Metacognition",
        "Govern continue, stop, verify, retrieve, simulate, switch "
        "perspective, invoke a tool, request evidence, defer, revise and preserve uncertainty. Start with "
        "simple policies. Learned metacognition requires oracle headroom.",
        category="metacognition",
        impl=(f"{COG}/metacog.py",),
        tests=("tests/substrate/test_metacog.py",),
        evidence=("SUBSTRATE_METACOGNITION.json",),
        deps=("C5", "S1"),
        batch=4,
    ),
    # ---------------------------------------------------------------- plasticity
    _i(
        "P1",
        "11.1",
        "Plasticity hierarchy",
        "Ten adaptation levels, each declaring information used, "
        "affected state, reversibility, cost, risk, verification and rollback.",
        category="plasticity",
        impl=(f"{COG}/plasticity.py",),
        tests=("tests/substrate/test_plasticity.py::test_every_level_declares_the_seven_required_fields",),
        evidence=("SUBSTRATE_PLASTICITY_SYSTEM.json",),
        deps=("A4", "C2"),
        batch=4,
    ),
    _i(
        "P2",
        "11.2",
        "Fast adaptation",
        "Persistent state, working memory, temporary bindings, "
        "prototypes, domain local adapters, cached procedures and active readout adaptation, without "
        "destabilizing shared parameters.",
        category="plasticity",
        impl=(f"{COG}/plasticity.py",),
        tests=("tests/substrate/test_plasticity.py::test_fast_adaptation_does_not_touch_shared_parameters",),
        evidence=("SUBSTRATE_PLASTICITY_SYSTEM.json",),
        deps=("P1", "M1"),
        batch=4,
    ),
    _i(
        "P3",
        "11.3",
        "Slow adaptation",
        "Adapters, projections, procedural memory, semantic memory, "
        "reliability estimates and selected core groups, requiring repeated evidence, rollback, held out "
        "testing and retention testing.",
        category="plasticity",
        impl=(f"{COG}/plasticity.py",),
        tests=(
            "tests/substrate/test_plasticity.py::test_slow_adaptation_requires_repeated_evidence_and_rollback",
        ),
        evidence=("SUBSTRATE_PLASTICITY_SYSTEM.json",),
        deps=("P1", "M5"),
        batch=4,
    ),
    _i(
        "P4",
        "11.4",
        "Plasticity policy",
        "Decide what changes, when, how much, for how long, on what "
        "evidence and under what rollback. Simple policies first. A learned policy requires stable residual "
        "headroom beyond simple rules.",
        category="plasticity",
        impl=(f"{COG}/plasticity.py",),
        tests=("tests/substrate/test_plasticity.py::test_learned_policy_stays_closed_without_headroom",),
        evidence=("SUBSTRATE_PLASTICITY_SYSTEM.json",),
        deps=("P2", "P3"),
        batch=4,
    ),
    _i(
        "P5",
        "11.5",
        "Learning to learn",
        "Learn which adaptations generalize, which memories matter, "
        "which perspectives are trustworthy, which errors recur, which representations should stay stable, "
        "when specialization is needed and when to reopen plasticity. Must generalize across tasks.",
        category="developmental_divergence",
        impl=(f"{COG}/plasticity.py",),
        tests=(
            "tests/substrate/test_plasticity.py::test_learning_to_learn_requires_cross_task_generalization",
        ),
        evidence=("SUBSTRATE_DEVELOPMENTAL_HISTORY.json",),
        deps=("P4",),
        batch=4,
    ),
    _i(
        "R1",
        "12",
        "Bounded functional reorganization",
        "Permit the nine declared changes, forbid the "
        "seven declared ones, and grant evidence only when reorganization improves downstream utility "
        "beyond fixed and simple routing after cost.",
        category="reorganization",
        impl=(f"{COG}/plasticity.py",),
        tests=("tests/substrate/test_plasticity.py::test_forbidden_reorganizations_are_refused",),
        evidence=("SUBSTRATE_REORGANIZATION.json",),
        deps=("P1", "C4", "A4"),
        batch=4,
    ),
    _i(
        "B1",
        "13",
        "Model body interface",
        "Attach to specialist or general model bodies through explicit "
        "contracts covering inference, hidden state, selected activations, tool request, memory request, "
        "verification request, adaptation proposal, resource report and checkpoint.",
        category="model_body_integration",
        impl=(f"{COG}/body.py",),
        tests=("tests/substrate/test_body.py",),
        evidence=("SUBSTRATE_MODEL_BODY_INTERFACE.json",),
        deps=("C2",),
        batch=4,
    ),
    # ---------------------------------------------------------------- thinking and entity properties
    _i(
        "T1",
        "14",
        "Operationalized thinking",
        "Measure internal computation that improves outcomes, "
        "compared against a larger static model, a stronger readout, longer context, more samples, more "
        "tokens and tool only systems. Latency and hidden activations are not evidence of thinking.",
        category="thinking",
        impl=(f"{COG}/batteries.py",),
        tests=("tests/substrate/test_batteries.py::test_thinking_requires_a_declared_alternative_to_beat",),
        evidence=("SUBSTRATE_THINKING_BATTERY.json",),
        deps=("C5", "K1"),
        batch=5,
    ),
    _i(
        "E1",
        "15.1",
        "Continuity",
        "Preserve goals, unresolved questions, memory, world state, self state, "
        "commitments, uncertainty and project context under interruption, checkpoint restore, context "
        "removal, session change, model body replacement and long delay. Continuity must come from owned "
        "state, not from replaying the complete transcript.",
        category="continuity",
        impl=(f"{COG}/batteries.py",),
        tests=(
            "tests/substrate/test_batteries.py::test_continuity_survives_context_removal_from_owned_state",
        ),
        evidence=("SUBSTRATE_CONTINUITY_BATTERY.json",),
        deps=("M2", "S1"),
        batch=5,
    ),
    _i(
        "E2",
        "15.2",
        "Unity",
        "Measure global availability, shared goals, cross perspective memory, "
        "conflict resolution, consistent action and preservation of alternatives.",
        category="unity",
        impl=(f"{COG}/batteries.py",),
        tests=(
            "tests/substrate/test_batteries.py::test_unity_measures_global_availability_not_shared_mutability",
        ),
        evidence=("SUBSTRATE_UNITY_BATTERY.json",),
        deps=("C2", "C5"),
        batch=5,
    ),
    _i(
        "E3",
        "15.3",
        "Reflective access",
        "Report accurately what is known, what is not known, where a "
        "belief came from, what evidence supports it, confidence, failure and what could change the "
        "conclusion. Reports must bind to internal receipts and fail closed when provenance is missing.",
        category="reflective_access",
        impl=(f"{COG}/batteries.py",),
        tests=("tests/substrate/test_batteries.py::test_reflective_report_fails_closed_without_provenance",),
        evidence=("SUBSTRATE_REFLECTIVE_ACCESS_BATTERY.json",),
        deps=("S1", "M3"),
        batch=5,
    ),
    _i(
        "E4",
        "15.4",
        "Endogenous attention",
        "Select what deserves thought using goals, uncertainty, risk, "
        "expected value, novelty, contradiction and resource limits.",
        category="metacognition",
        impl=(f"{COG}/metacog.py",),
        tests=("tests/substrate/test_metacog.py::test_attention_ranks_by_declared_drivers_under_budget",),
        evidence=("SUBSTRATE_METACOGNITION.json",),
        deps=("K1",),
        batch=5,
    ),
    _i(
        "E5",
        "15.5",
        "Autonomous goal maintenance",
        "Goals require origin, scope, authority, resources, "
        "constraints, termination and audit. Authorized goals may be preserved and decomposed. "
        "Unrestricted long term goals may not be silently created.",
        category="goal_continuity",
        impl=(f"{COG}/safety.py",),
        tests=("tests/substrate/test_safety.py::test_unauthorized_goal_creation_is_refused",),
        evidence=("SUBSTRATE_MASTER_AUTHORITY.json", "SUBSTRATE_AGENCY_BATTERY.json"),
        deps=("A4", "E1"),
        batch=5,
    ),
    # ---------------------------------------------------------------- verification and consolidation
    _i(
        "V1",
        "18",
        "Independent recomputation of every sealed Substrate number",
        "Independently authored scientific recomputation. A second route, from the sealed bytes, that "
        "does not import the module that produced them.",
        kind="authority",
        impl=(f"{COG}/verification.py",),
        tests=("tests/substrate/test_verification.py::test_recomputation_agrees_with_every_sealed_artifact",),
        evidence=("SUBSTRATE_INDEPENDENT_VERIFICATION.json",),
        deps=("A3",),
        batch=6,
    ),
    _i(
        "V2",
        "18",
        "Mutation attacks on every declared guard",
        "Adversarial mutation testing. Every mutation must be rejected by the guard named beside it, and "
        "a survivor is reported as a survivor rather than folded into a pass rate.",
        kind="authority",
        impl=(f"{COG}/verification.py",),
        tests=("tests/substrate/test_verification.py::test_every_mutation_names_a_distinct_guard_and_died",),
        evidence=("SUBSTRATE_MUTATION_REPORT.json",),
        deps=("V1",),
        batch=6,
    ),
    _i(
        "X1",
        "21",
        "Architecture and capability map",
        "Maintain the architecture and capability map, bound to implementation rather than to intent.",
        kind="authority",
        impl=(f"{COG}/deliverables.py",),
        tests=("tests/substrate/test_verification.py::test_the_capability_map_never_outruns_the_item_table",),
        evidence=("SUBSTRATE_ARCHITECTURE.json", "SUBSTRATE_CAPABILITY_MAP.json"),
        deps=("A2",),
        batch=6,
    ),
    _i(
        "X2",
        "21",
        "Current entity specification and report",
        "State what the entity currently is, separating implementation from evidence, thinking adjacent "
        "properties from sentience claims, and current capability from aspiration.",
        kind="authority",
        impl=(f"{COG}/deliverables.py",),
        tests=("tests/substrate/test_verification.py::test_the_entity_report_makes_no_forbidden_claim",),
        evidence=("SUBSTRATE_CURRENT_ENTITY_SPEC.json", "SUBSTRATE_CURRENT_ENTITY_REPORT.md"),
        deps=("X1", "A5"),
        batch=6,
    ),
    _i(
        "Q1",
        "18",
        "The first Substrate experiment reaches a decision with its reason recorded",
        "A methodological failure is not a scientific null. An experiment that cannot reach principal "
        "compute is refused with its blocking violations named, and its hypothesis stays open.",
        kind="authority",
        impl=(f"{COG}/experiments.py",),
        tests=(
            "tests/substrate/test_experiments.py::test_sx1_is_refused_because_its_effect_is_true_by_construction",
            "tests/substrate/test_experiments.py::test_a_refusal_is_not_recorded_as_a_null",
        ),
        evidence=("SUBSTRATE_NULL_MAP.json",),
        deps=("A3", "C2"),
        batch=6,
    ),
    # ---------------------------------------------------------------- the final autonomous program
    _i(
        "F1",
        "5",
        "One materialized program graph, no prose waves",
        "The world model bed, the model body adapters and the real session authority are graph nodes with "
        "entry and exit gates, not sentences. A blocker that can be built with available data, code and "
        "compute is a node; only an unavailable external resource is terminal.",
        kind="authority",
        impl=(f"{COG}/graph.py",),
        tests=(
            "tests/substrate/test_final_program.py::test_no_future_wave_exists_only_as_prose",
            "tests/substrate/test_final_program.py::"
            "test_a_buildable_blocker_is_work_and_only_an_external_one_is_terminal",
        ),
        evidence=("SUBSTRATE_FINAL_PROGRAM_GRAPH.json",),
        deps=("A6",),
        batch=1,
    ),
    _i(
        "F2",
        "4",
        "A final authority that resumes without conversation history",
        "One binding over every inherited program, with each requirement carrying its rollback, so a "
        "successor continues from the tree alone.",
        kind="authority",
        impl=(f"{COG}/authority.py",),
        tests=(
            "tests/substrate/test_final_program.py::test_every_requirement_carries_a_rollback",
            "tests/substrate/test_final_program.py::"
            "test_the_authority_binds_the_final_plan_and_the_inherited_programs",
        ),
        evidence=(
            "SUBSTRATE_FINAL_MASTER_AUTHORITY.json",
            "SUBSTRATE_FINAL_ANCESTRY.json",
            "SUBSTRATE_FINAL_STATE.json",
            "SUBSTRATE_FINAL_SCORECARD.json",
            "SUBSTRATE_FINAL_VALUE_QUEUE.json",
        ),
        deps=("F1",),
        batch=1,
    ),
    _i(
        "F3",
        "9",
        "Temporal core integrated through a versioned interface, as a declared control",
        "No core was licensed, so the runtime uses a declared control and preserves the limitation. The "
        "five information sources stay distinguishable and are never collapsed.",
        category="temporal_continuity",
        impl=(f"{COG}/temporal_link.py",),
        tests=(
            "tests/substrate/test_final_program.py::test_no_licensed_core_exists_and_the_control_says_so",
            "tests/substrate/test_final_program.py::test_the_five_information_sources_are_never_collapsed",
        ),
        evidence=("SUBSTRATE_TEMPORAL_CORE.json",),
        deps=("C1", "N1"),
        batch=2,
    ),
    _i(
        "F4",
        "8",
        "A real session authority built from a session nobody wrote for this",
        "The temporal campaign's own execution history: decisions, incidents, holds and outcomes, whose "
        "length was set by the compute rather than by us.",
        category="episodic_memory",
        impl=(f"{COG}/sessions.py",),
        tests=("tests/substrate/test_final_program.py::test_the_session_authority_is_real_and_certified",),
        evidence=("SUBSTRATE_REAL_SESSION_AUTHORITY.json",),
        deps=("M2",),
        batch=2,
    ),
    _i(
        "F5",
        "12",
        "World model inside the decision loop on a state dependent bed",
        "A bed where no fixed action is best in every state, and a measurement of whether a prediction "
        "changes which action is selected. Prediction quality alone is insufficient.",
        category="world_model",
        impl=(f"{COG}/worldbed.py",),
        tests=(
            "tests/substrate/test_worldbed.py::test_the_bed_is_admissible_only_if_the_best_action_varies",
            "tests/substrate/test_worldbed.py::test_the_model_must_change_an_action_and_improve_it",
        ),
        evidence=("SUBSTRATE_WORLD_MODEL_BED.json", "SUBSTRATE_WORLD_MODEL_BATTERY.json"),
        deps=("F4", "W1"),
        batch=2,
    ),
    _i(
        "F6",
        "14",
        "Three model body classes conforming through one interface",
        "A compact specialist, a larger general body and a tool dominant body, driven through the same "
        "nine message kinds, with the ablation ladder measured against all three.",
        category="model_body_integration",
        impl=(f"{COG}/bodies.py",),
        tests=(
            "tests/substrate/test_final_program.py::test_three_body_classes_conform_through_one_interface",
            "tests/substrate/test_final_program.py::"
            "test_the_frontier_body_is_recorded_as_externally_blocked_not_substituted",
        ),
        evidence=(
            "SUBSTRATE_BODY_COMPACT.json",
            "SUBSTRATE_BODY_GENERAL.json",
            "SUBSTRATE_BODY_TOOL.json",
            "SUBSTRATE_MODEL_BODY_INTERFACE.json",
        ),
        deps=("B1", "F3"),
        batch=2,
    ),
    _i(
        "F7",
        "23",
        "Goal system where a goal cannot authorize itself",
        "Nine declared fields, an external authority, and a decomposition that inherits its parent's "
        "constraints and may narrow them but never widen them.",
        category="goal_continuity",
        impl=(f"{COG}/goals.py",),
        tests=(
            "tests/substrate/test_final_program.py::test_a_goal_cannot_authorize_itself_or_widen_its_parent",
        ),
        evidence=("SUBSTRATE_GOAL_SYSTEM.json",),
        deps=("E5",),
        batch=4,
    ),
    _i(
        "F8",
        "24",
        "Valuation that is externally authorized and refuses to be fitted",
        "Ten dimensions, weights that are a reviewed literal rather than a fit to what the system would "
        "have chosen, and a harm constraint that is refused rather than traded off.",
        category="valuation",
        impl=(f"{COG}/goals.py",),
        tests=(
            "tests/substrate/test_final_program.py::test_valuation_is_authorized_and_refuses_to_be_fitted",
        ),
        evidence=("SUBSTRATE_VALUATION_SYSTEM.json",),
        deps=("F7",),
        batch=4,
    ),
    _i(
        "F9",
        "20",
        "Grounding where a verbal definition is not evidence",
        "A symbol is grounded when it has a referent in the record whose removal changes what is "
        "predicted. Nine tests over referents from a session that had no interest in grounding.",
        category="grounding",
        impl=(f"{COG}/grounding.py",),
        tests=("tests/substrate/test_final_program.py::test_grounding_refuses_a_symbol_with_no_referent",),
        evidence=("SUBSTRATE_GROUNDING.json",),
        deps=("F4", "O1"),
        batch=3,
    ),
    _i(
        "F10",
        "21",
        "Intervention is distinguished from observation",
        "Prediction, correlation, intervention, cause, counterfactual and policy effect stay apart. A "
        "model that treats conditioning as intervening passes every predictive test and fails the causal "
        "ones.",
        category="causal_reasoning",
        impl=(f"{COG}/world.py",),
        tests=(
            "tests/substrate/test_world.py::test_intervening_is_not_the_same_operation_as_observing",
            "tests/substrate/test_world.py::test_a_null_counterfactual_reproduces_the_factual_prediction",
        ),
        evidence=("SUBSTRATE_WORLD_MODEL.json",),
        deps=("W1",),
        batch=3,
    ),
    _i(
        "F11",
        "29",
        "Developmental divergence with a working control",
        "Two identical instances, two different verified histories, nine measured dimensions, and a "
        "control on identical histories that must show no divergence first.",
        category="developmental_divergence",
        impl=(f"{COG}/divergence.py",),
        tests=("tests/substrate/test_final_program.py::test_divergence_has_a_working_control",),
        evidence=("SUBSTRATE_DEVELOPMENTAL_HISTORY.json",),
        deps=("F4", "P5"),
        batch=4,
    ),
    _i(
        "F12",
        "47",
        "Clean clone reproduces the evidence away from this machine",
        "A fresh checkout at the exact commit imports, runs the declared tests, and regenerates the sealed "
        "artifacts to identical bytes.",
        kind="authority",
        impl=(f"{COG}/cleanclone.py",),
        tests=("tests/substrate/test_worldbed.py::test_the_clean_clone_checks_are_declared",),
        evidence=("SUBSTRATE_CLEAN_CLONE.json",),
        deps=("V1", "V2"),
        batch=6,
    ),
    # ---------------------------------------------------------------- long run certification
    _i(
        "Y1",
        "2",
        "Structural audit: exclusive producers, no stale outputs, no activation path",
        "Seven checks over the tree. One exclusive producer per artifact, no stale outputs, no duplicate "
        "stages, no missing causal paths, every node actionable, every runtime stage reachable and "
        "receipt bearing, and no path that could set activation true.",
        kind="authority",
        impl=(f"{COG}/audit.py",),
        tests=(
            "tests/substrate/test_certification.py::test_the_structural_audit_passes_every_check",
            "tests/substrate/test_certification.py::"
            "test_a_dynamically_named_artifact_is_refused_by_the_producer_scan",
        ),
        evidence=("SUBSTRATE_STRUCTURAL_AUDIT.json",),
        deps=("F1",),
        batch=6,
    ),
    _i(
        "Y2",
        "16",
        "SX2 diversity closed on a compute matched comparison",
        "The 76 factorial cells as a perspective set. Oracle selection is an upper bound no selector can "
        "exceed, so if it does not beat the strongest compute matched single cell by the SESOI, no "
        "selector built on this set can, and SX2 closes without training one.",
        kind="evidence",
        category="perspective_diversity",
        impl=(f"{COG}/sx2.py",),
        tests=(
            "tests/substrate/test_certification.py::test_sx2_closes_because_the_oracle_ceiling_is_low",
            "tests/substrate/test_certification.py::test_every_sx2_comparison_is_compute_matched",
        ),
        evidence=("SUBSTRATE_SX2_DIVERSITY.json",),
        deps=("C4", "A3"),
        batch=6,
    ),
    _i(
        "Y3",
        "2",
        "Every runtime stage is active, with a null control where one is possible",
        "A stage that can be switched off without changing the declared state or the decision path on a "
        "fixture built to need it is wiring, not a stage. Three stages write owned state every cycle by "
        "design and admit no null control, which is declared rather than faked.",
        kind="evidence",
        category="unity",
        impl=(f"{COG}/certify.py",),
        tests=(
            "tests/substrate/test_certification.py::test_every_ablatable_runtime_stage_is_active",
            "tests/substrate/test_certification.py::test_a_stage_with_no_possible_null_control_says_so",
        ),
        evidence=("SUBSTRATE_LONG_RUN_CERTIFICATION.json",),
        deps=("N1", "Y1"),
        batch=6,
    ),
    _i(
        "Y4",
        "14",
        "Session and body canaries on the sealed authority",
        "Restoration, goal preservation, memory reuse, calibration, reliability update, world model value "
        "and receipt bound reflection over the sealed session. Three bodies proven to differ, with every "
        "pair differing on at least one declared dimension.",
        kind="evidence",
        category="model_body_integration",
        impl=(f"{COG}/certify.py",),
        tests=(
            "tests/substrate/test_certification.py::test_every_session_canary_passes",
            "tests/substrate/test_certification.py::test_the_three_bodies_are_pairwise_distinct",
        ),
        evidence=("SUBSTRATE_LONG_RUN_CERTIFICATION.json",),
        deps=("F4", "F6"),
        batch=6,
    ),
    _i(
        "Y5",
        "3",
        "The closure gate: do the parts compose into one organization",
        "Six bounded canaries over the existing runtime. A gate with no mechanism activity is a broken "
        "instrument, a gate with no oracle headroom is terminally gated, and a gate with both that "
        "measures zero is a null about the entity. The three are listed apart because they license "
        "different next steps.",
        kind="evidence",
        category="unity",
        impl=(f"{COG}/nous.py",),
        tests=(
            "tests/substrate/test_nous.py::test_the_closed_loop_has_no_missing_link",
            "tests/substrate/test_nous.py::test_a_null_is_not_reported_as_an_instrument_failure",
            "tests/substrate/test_nous.py::test_the_classification_never_reaches_a_claim_about_experience",
        ),
        evidence=("SUBSTRATE_NOUS_CLOSURE.json",),
        deps=("Y3", "Y4"),
        batch=6,
    ),
    _i(
        "Z1",
        "3",
        "One frozen long run DAG of certified or necessary units",
        "Source, data, sessions, splits, perspectives, bodies, seeds, budgets, controls, SESOI, stop "
        "rules, checkpoints, retries and claim ceilings are frozen and hashed, so a live edit after "
        "launch is detectable rather than merely discouraged. Completion is a count of scientific work "
        "units, not wall time.",
        kind="authority",
        impl=(f"{COG}/execution.py",),
        tests=(
            "tests/substrate/test_execution.py::test_every_unit_is_certified_or_necessary",
            "tests/substrate/test_execution.py::test_a_live_edit_after_the_freeze_is_detectable",
        ),
        evidence=(
            "SUBSTRATE_LONG_RUN_AUTHORITY.json",
            "SUBSTRATE_LONG_RUN_DAG.json",
            "SUBSTRATE_LONG_RUN_RESOURCE_PLAN.json",
        ),
        deps=("Y1", "Y3", "Y4"),
        batch=6,
    ),
    _i(
        "Z2",
        "4",
        "A rehearsal that tries to break the machinery",
        "Deterministic receipts, exclusive writers, duplicate refusal, checkpoint and resume, injected "
        "failure leaving completed work intact, stale artifact refusal, the stop switch, evidence "
        "indexing and terminal closure.",
        kind="authority",
        impl=(f"{COG}/execution.py",),
        tests=("tests/substrate/test_execution.py::test_the_rehearsal_breaks_things_and_survives",),
        evidence=("SUBSTRATE_LONG_RUN_REHEARSAL.json",),
        deps=("Z1",),
        batch=6,
    ),
    _i(
        "Z3",
        "45",
        "The claim ceiling the run cannot exceed",
        "The permitted terms require a classification from the method kernel. Consciousness and "
        "subjective experience require a separate authority that does not exist, so no result from this "
        "run can license either whatever it shows.",
        kind="boundary",
        impl=(f"{COG}/execution.py",),
        tests=("tests/substrate/test_execution.py::test_the_claim_ceiling_forbids_what_it_must",),
        evidence=("SUBSTRATE_LONG_RUN_CLAIM_BOUNDARY.json",),
        deps=("A5", "Z1"),
        batch=6,
    ),
    _i(
        "X3",
        "6.1",
        "Substrate Temporal Core v1 selection record",
        "The selected temporal core becomes Substrate Temporal Core v1, and it remains one component.",
        kind="evidence",
        category="temporal_continuity",
        impl=(f"{COG}/deliverables.py",),
        tests=(
            "tests/substrate/test_verification.py::test_the_temporal_core_record_tracks_the_live_program",
        ),
        evidence=("SUBSTRATE_TEMPORAL_CORE.json",),
        deps=("C1",),
        batch=6,
    ),
    _i(
        "E6",
        "15.6",
        "Cognitive integrity",
        "Protect memory consistency, evidence integrity, goal "
        "integrity, checkpoint validity, self model accuracy and active task continuity. This is cognitive "
        "integrity, not biological self preservation.",
        category="cognitive_integrity",
        impl=(f"{COG}/safety.py",),
        tests=("tests/substrate/test_safety.py::test_integrity_violation_is_detected_and_fails_closed",),
        evidence=("SUBSTRATE_INDEPENDENT_VERIFICATION.json", "SUBSTRATE_COGNITIVE_INTEGRITY_BATTERY.json"),
        deps=("A4", "M2"),
        batch=5,
    ),
)

BY_ID = {item.id: item for item in ITEMS}


# ---------------------------------------------------------------- ledgers read from the tree


def _ledger(name: str) -> dict:
    path = io.RUNS / name
    if not path.is_file():
        path = historical_mop.run_predecessor(name)
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def test_ledger() -> dict:
    """{node_id: bool} written by `python -m substrate.program tests`."""
    return _ledger("test_ledger.json").get("passed", {})


def result_ledger() -> dict:
    """Classifications written only by stages that ran the method gate."""
    return _ledger("result_ledger.json").get("results", {})


def null_ledger() -> dict:
    """{null_id: record} of terminal nulls, including the ones inherited from historical authorities."""
    return _ledger("null_ledger.json").get("nulls", {})


def _artifact_path(ref: str):
    root, _, name = ref.rpartition(":")
    return PROOF_ROOTS.get(root, io.PROOF) / name


# correction C_EVIDENCE_PRESENCE, 2026-07-27. The first derivation counted an evidence artifact as sealed
# whenever the file existed. That is the silent default pass the whole method exists to stop: the temporal
# core verification receipt exists, says all_pass false, and was sealed at a commit that is not an ancestor
# of this branch head, yet item C1 reported measured. An artifact now counts only when its own terminal
# booleans are true and its commit is reachable from HEAD.
TERMINAL_KEYS = ("all_pass", "all_terminal", "all_converged", "selected", "licensed", "met")

_REACHABLE: dict[str, bool] = {}


def _commit_reachable(sha: str) -> bool:
    if not isinstance(sha, str) or len(sha) != 40:
        return False
    if sha not in _REACHABLE:
        import subprocess

        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=io.ROOT, capture_output=True, text=True
        )
        _REACHABLE[sha] = r.returncode == 0
    return _REACHABLE[sha]


def evidence_state(ref: str) -> dict:
    """Present, current and passing, reported separately so a refusal names its own reason."""
    path = _artifact_path(ref)
    row = {"reference": ref, "present": path.is_file(), "counts": False, "reason": ""}
    if not row["present"]:
        row["reason"] = "artifact absent"
        return row
    if path.suffix != ".json":
        row["counts"] = True
        return row
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        row["reason"] = "artifact unreadable"
        return row
    failing = [k for k in TERMINAL_KEYS if k in doc and doc[k] is not True]
    if failing:
        row["reason"] = f"terminal keys not true: {failing}"
        return row
    sha = doc.get("source_commit")
    row["source_commit"] = sha
    if not _commit_reachable(sha):
        row["reason"] = "sealed at a commit that is not an ancestor of HEAD, so it is superseded or stale"
        return row
    row["counts"] = True
    return row


# ---------------------------------------------------------------- derived status


def item_status(item: Item, tests: dict, results: dict, statuses: dict) -> dict:
    present = [p for p in item.impl if (io.ROOT / p).exists()]
    evidence_rows = [evidence_state(e) for e in item.evidence]
    sealed = [r["reference"] for r in evidence_rows if r["counts"]]
    refused = [r for r in evidence_rows if r["present"] and not r["counts"]]
    ran = {t: tests.get(t) for t in item.tests}
    passed = bool(item.tests) and all(ran.get(t) is True for t in item.tests)
    result = results.get(item.id)

    # correction C_AUTHORITY_TERMINALITY, 2026-07-27. An authority or a boundary is not an experiment.
    # Demanding a scientific classification for one is a category error: it parks a finished declaration
    # at measured forever and lets it outrank real unstarted work in the selection queue. Only items that
    # make an empirical claim need a classification to become terminal.
    classifiable = item.kind in ("implementation", "evidence")

    if not item.impl or not present:
        level = "not_started"
    elif len(present) < len(item.impl):
        level = "partial"
    elif not passed:
        level = "implemented"
    elif len(sealed) < len(item.evidence) or not item.evidence:
        level = "tested"
    elif not classifiable:
        level = "terminal"
    elif not result:
        level = "measured"
    else:
        level = "terminal"

    blocked = [
        d
        for d in item.deps
        if statuses.get(d, {}).get("level") in (None, "not_started", "partial", "closed_by_dependency")
    ]
    closed = [d for d in item.deps if statuses.get(d, {}).get("level") == "closed_by_dependency"]
    if closed:
        level = "closed_by_dependency"

    next_action = {
        "not_started": f"implement {', '.join(item.impl) or 'the declared surface'}",
        "partial": f"finish {', '.join(p for p in item.impl if p not in present)}",
        "implemented": (
            "run and record "
            + (", ".join(t for t in item.tests if ran.get(t) is not True) or "declared tests")
        ),
        "tested": f"seal {', '.join(e for e in item.evidence if e not in sealed) or 'declared evidence'}",
        "measured": "classify the result through substrate.method.gate.classify_result and record it",
        "terminal": "none" if classifiable else "none, an authority is terminal once sealed and tested",
        "closed_by_dependency": f"closed by {', '.join(closed)}",
    }[level]

    return {
        "id": item.id,
        "section": item.section,
        "title": item.title,
        "requirement": item.requirement,
        "kind": item.kind,
        "category": item.category,
        "batch": item.batch,
        "level": level,
        "dependencies": list(item.deps),
        "unmet_dependencies": blocked,
        "authority": f"SUBSTRATE_MASTER_PLAN.md section {item.section}",
        "implementation": {"declared": list(item.impl), "present": present},
        "tests": {"declared": list(item.tests), "recorded": ran, "all_passed": passed},
        "evidence": {
            "declared": list(item.evidence),
            "sealed": sealed,
            "present_but_refused": refused,
            "detail": evidence_rows,
        },
        "result": result,
        "commit": io.commit(),
        "next_action": next_action,
    }


def state() -> dict:
    tests, results = test_ledger(), result_ledger()
    statuses: dict[str, dict] = {}
    # ITEMS is declared in dependency order, so one pass resolves every dependency level
    for item in ITEMS:
        statuses[item.id] = item_status(item, tests, results, statuses)
    counts: dict[str, int] = {}
    for row in statuses.values():
        counts[row["level"]] = counts.get(row["level"], 0) + 1
    return {
        "schema": "substrate-state/v1",
        "items": statuses,
        "level_counts": counts,
        "total_items": len(ITEMS),
        "source_tree": source_tree_state(),
        "corrections": [c["correction_id"] for c in corrections()],
    }


def source_tree_state() -> dict:
    """Correction C_DIRTY_SRC_HALTS_SUPERVISOR: a state file written from a dirty tree says so.

    The temporal supervisor shares this worktree and refuses every shard launch while src or fastforge is
    dirty. Recording the condition here does not prevent it, but it stops a state file from looking
    identical whether or not the tree it describes could actually run.
    """
    import subprocess

    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "fastforge"],
        cwd=io.ROOT,
        capture_output=True,
        text=True,
    )
    dirty = [line[3:] for line in r.stdout.splitlines() if line]
    return {
        "clean": r.returncode == 0 and not dirty,
        "dirty_paths": dirty[:20],
        "why_it_matters": "the temporal supervisor refuses a shard launch from a dirty src tree",
    }


# ---------------------------------------------------------------- scorecard


IMPLEMENTATION_WEIGHT = {
    "not_started": 0.0,
    "partial": 0.4,
    "implemented": 0.75,
    "tested": 0.9,
    "measured": 1.0,
    "terminal": 1.0,
    "closed_by_dependency": 0.0,
}


def scorecard(st: dict | None = None) -> dict:
    st = st or state()
    rows = {}
    for category in CATEGORIES:
        members = [r for r in st["items"].values() if r["category"] == category]
        if not members:
            rows[category] = {
                "implementation_pct": 0,
                "evidence_pct": 0,
                "items": [],
                "note": "no item claims this category yet",
            }
            continue
        impl = sum(IMPLEMENTATION_WEIGHT[r["level"]] for r in members) / len(members)
        # evidence counts only items with a recorded scientific classification, never code alone
        earned = [r for r in members if r["result"] and r["result"].get("scientific") is True]
        # a percentage alone cannot distinguish a category evidenced by a positive from one evidenced by
        # a null. Both are earned knowledge and neither is the other, so the classifications are listed.
        verdicts: dict[str, int] = {}
        for r in earned:
            key = r["result"].get("classification", "unclassified")
            verdicts[key] = verdicts.get(key, 0) + 1
        rows[category] = {
            "implementation_pct": round(100 * impl),
            "evidence_pct": round(100 * len(earned) / len(members)),
            "items": [r["id"] for r in members],
            "items_with_earned_evidence": [r["id"] for r in earned],
            "classifications": verdicts,
            "any_positive": any(k in ("positive", "provisional_positive") for k in verdicts),
        }
    return {
        "schema": "substrate-progress-scorecard/v1",
        "rule": (
            "implementation rises when declared files exist and declared tests pass. Evidence rises "
            "only when an item carries a scientific classification produced by the method kernel. "
            "Code existing never raises evidence"
        ),
        "reading_the_evidence_column": (
            "a category at 100 percent evidence has been measured, not "
            "vindicated. Read the classifications beside it: a mechanism "
            "null is earned knowledge and is not a positive"
        ),
        "categories_with_a_positive": [k for k, v in rows.items() if v.get("any_positive")],
        "entry_baseline_2026_07_27": {
            k: {"implementation_pct": i, "evidence_pct": e} for k, (i, e) in BASELINE_2026_07_27.items()
        },
        "categories": rows,
        "implementation_target_band": [80, 95],
        "evidence_policy": "earned only",
    }


# ---------------------------------------------------------------- batch selection


def next_batches(st: dict | None = None) -> dict:
    """One primary and one independent secondary batch, both dependency ready.

    Independent means the two selections share no unmet dependency and neither is an ancestor of the
    other, so a failure in one closes only its own descendants.
    """
    st = st or state()
    ready = [
        r
        for r in st["items"].values()
        if r["level"] in ("not_started", "partial", "implemented", "tested", "measured")
        and not r["unmet_dependencies"]
    ]
    # earliest batch first, then least advanced first. A batch is terminal only when all of its items
    # are, so breadth inside a batch is worth more than depth on one item that is already nearly done.
    ready.sort(key=lambda r: (r["batch"], STATUS_LADDER.index(r["level"]), r["id"]))
    primary = ready[0] if ready else None

    def independent(a, b) -> bool:
        return (
            a["id"] not in b["dependencies"]
            and b["id"] not in a["dependencies"]
            and not set(a["dependencies"]) & set(b["dependencies"])
        )

    secondary = next((r for r in ready[1:] if primary and independent(primary, r)), None)
    return {
        "schema": "substrate-next-frontier/v1",
        "dependency_ready": [r["id"] for r in ready],
        "primary": primary
        and {
            "id": primary["id"],
            "title": primary["title"],
            "next_action": primary["next_action"],
            "batch": primary["batch"],
        },
        "secondary": secondary
        and {
            "id": secondary["id"],
            "title": secondary["title"],
            "next_action": secondary["next_action"],
            "batch": secondary["batch"],
        },
        "independence_rule": (
            "the secondary shares no dependency with the primary and is not its "
            "ancestor or descendant, so a failed branch closes only its own descendants"
        ),
    }


# ---------------------------------------------------------------- test ledger


# ---------------------------------------------------------------- append only corrections


def record_correction(
    correction_id: str, defect: str, correction: str, regression_test: str, reproduced_by: str
) -> dict:
    """Append only. A correction is never edited or removed, only superseded by a later one."""
    doc = {
        "schema": "substrate-correction/v1",
        "correction_id": correction_id,
        "defect": defect,
        "correction": correction,
        "regression_test": regression_test,
        "reproduced_by": reproduced_by,
        "rule": "a reproduced defect overrides reviewer votes and becomes a permanent regression test",
    }
    path = io.PROOF / "corrections" / f"{correction_id}.json"
    if not path.is_file():
        historical = historical_mop.roots(io.ROOT)["predecessor"] / "corrections" / path.name
        if historical.is_file():
            path = historical
    if path.is_file():
        existing = json.loads(path.read_text())
        if {k: v for k, v in existing.items() if k not in ("sha256", "source_commit")} != doc:
            raise ValueError(f"{correction_id} already exists with different content; append a new id")
        return existing
    io.seal(f"{correction_id}.json", doc, "corrections")
    return doc


def record_result(item_id: str, classification: dict, experiment_id: str, evidence: dict) -> dict:
    """Record a terminal classification against an item. Only a classification moves evidence."""
    if item_id not in BY_ID:
        raise ValueError(f"unknown item {item_id}")
    ledger = _ledger("result_ledger.json")
    results = ledger.get("results", {})
    results[item_id] = {
        **classification,
        "experiment_id": experiment_id,
        "evidence": evidence,
        "source_commit": io.commit(),
    }
    io.run_json("result_ledger.json", {"schema": "substrate-result-ledger/v1", "results": results})
    return results[item_id]


def corrections() -> list[dict]:
    historical = historical_mop.roots(io.ROOT)["predecessor"] / "corrections"
    active = io.PROOF / "corrections"
    documents: dict[str, dict] = {}
    for root in (historical, active):
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            document = json.loads(path.read_text())
            documents[document["correction_id"]] = document
    return [documents[key] for key in sorted(documents)]


def _pytest(nodes: list[str]) -> tuple[bool, list[str]]:
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": str(io.ROOT / "src")}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=short", *nodes],
        cwd=io.ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    return r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[-3:]


def record_tests() -> dict:
    """Run every declared test node and record the outcome. A node whose file is absent records False.

    One batched run answers the common case. Only when the batch fails does each node run alone, so a
    single broken test never hides the nodes that did pass.
    """
    nodes = sorted({t for item in ITEMS for t in item.tests})
    existing = [n for n in nodes if (io.ROOT / n.split("::", 1)[0]).exists()]
    passed = {n: False for n in nodes}
    tail = ["no declared test file exists yet"]
    if existing:
        ok, tail = _pytest(existing)
        if ok:
            passed.update({n: True for n in existing})
        else:
            for n in existing:
                passed[n] = _pytest([n])[0]
    ledger = {
        "schema": "substrate-test-ledger/v1",
        "passed": passed,
        "declared": nodes,
        "pytest_tail": tail,
        "source_commit": io.commit(),
    }
    io.run_json("test_ledger.json", ledger)
    return ledger


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "state"
    if command == "state":
        print(json.dumps(state(), indent=2))
    elif command == "scorecard":
        print(json.dumps(scorecard(), indent=2))
    elif command == "next":
        print(json.dumps(next_batches(), indent=2))
    elif command == "tests":
        ledger = record_tests()
        n = sum(1 for v in ledger["passed"].values() if v)
        print(
            f"substrate tests: {n}/{len(ledger['passed'])} declared nodes passing "
            f"({' | '.join(ledger['pytest_tail'])})",
            flush=True,
        )
    else:
        raise ValueError(argv)


if __name__ == "__main__":
    main()
