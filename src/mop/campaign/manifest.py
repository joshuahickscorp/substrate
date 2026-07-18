"""The broad pre-substrate discovery campaign manifest.

Assembles one durable ``CampaignSpec`` DAG: operational invariance, real runnable science question families
across Waves A through J, cross-question analysis (mechanism diagnosis, negative-space synthesis, mechanism
cards, the readiness gate), precommitted decision-branch reproductions, and contracted external-input
families whose only blocker is named data or an external authority. It also declares the live General Run
and horizon successor chain as external resource consumers and authority boundaries, adopted not disturbed.

Adding a science family is one row in ``_SCIENCE`` (a real runner) or ``_EXTERNAL`` (a contract plus its
exact blocker). Nothing here mutates a sealed artifact.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from .specs import (
    BedSpec,
    CampaignSpec,
    Coverage,
    DecisionRule,
    Dependency,
    DependencyKind,
    ExternalDependency,
    NodeKind,
    NodeSpec,
    ReproductionSpec,
    ResearchQuestionSpec,
    ResourceClass,
    ResourceRequest,
)

CAMPAIGN_ID = "mop_pre_substrate_expansion_v1"

_HASH = ResourceRequest(
    resource_class=ResourceClass.CPU_HASH_HEAVY, cpu_slots=1, mem_gb=1.0, est_seconds=15.0
)
_LIGHT = ResourceRequest(resource_class=ResourceClass.CPU_LIGHT, cpu_slots=1, mem_gb=0.75, est_seconds=8.0)
_EXCLUSIVE_SWEEP = ResourceRequest(
    resource_class=ResourceClass.EXCLUSIVE, cpu_slots=8, mem_gb=2.0, exclusive=True, est_seconds=30.0
)


# (node_id, entrypoint, title, form, phenomenon, mechanism_family, unit_class, evidence_level)
_SCIENCE: list[tuple[str, str, str, str, str, str, str, str]] = [
    (
        "wave_a_identity",
        "mop.campaign.nodes.wave_a_identity:wave_a_identity_runner",
        "Wave A: event-referent identity vs shuffled lineage",
        "events",
        "event_formation_identity",
        "referent_binder",
        "synthetic_session",
        "M1",
    ),
    (
        "wave_b_audio_onset",
        "mop.campaign.nodes.wave_b_audio_onset:wave_b_audio_onset_runner",
        "Wave B: native-audio onset (spectral flux) vs energy threshold",
        "native_audio",
        "temporal_boundary",
        "onset_detector",
        "synthetic_waveform",
        "M1",
    ),
    (
        "wave_c_object_state",
        "mop.campaign.nodes.wave_c_object_state:wave_c_object_state_runner",
        "Wave C: object-centric vs pooled state through occlusion",
        "vision_synthetic",
        "persistence_structured_state",
        "structured_state",
        "synthetic_scene",
        "M1",
    ),
    (
        "wave_d_world_model",
        "mop.campaign.nodes.wave_d_world_model:wave_d_world_model_runner",
        "Wave D: action-conditioned transition vs action-shuffled",
        "action",
        "prediction_planning",
        "world_model",
        "synthetic_gridworld",
        "M1",
    ),
    (
        "wave_e_memory",
        "mop.campaign.nodes.wave_e_memory:wave_e_memory_runner",
        "Wave E: replay retention-vs-future-learning frontier",
        "memory_episode",
        "continual_learning",
        "replay",
        "synthetic_stream",
        "M1",
    ),
    (
        "wave_f_plasticity",
        "mop.campaign.nodes.wave_f_plasticity:wave_f_plasticity_runner",
        "Wave F: continual-reset plasticity vs SGD and frozen-plus-shell",
        "memory_episode",
        "plasticity",
        "plasticity_controller",
        "synthetic_stream",
        "M1",
    ),
    (
        "wave_h_routing",
        "mop.campaign.nodes.wave_h_routing:wave_h_routing_runner",
        "Wave H: competence-conditioned routing vs best-single/homogeneous/random",
        "perspective",
        "routing_value_of_computation",
        "routing",
        "synthetic_task",
        "M1",
    ),
]

_ANALYSIS: list[tuple[str, str, str, str, str]] = [
    (
        "analysis_diagnosis",
        "mop.campaign.nodes.analysis:mechanism_diagnosis_runner",
        "Cross-question value-of-computation mechanism diagnosis",
        "value_of_computation",
        "sealed_result",
    ),
    (
        "analysis_negative_space",
        "mop.campaign.nodes.analysis:negative_space_runner",
        "Negative-space synthesis of nulls into failure families",
        "negative_space",
        "sealed_null",
    ),
]

# external families: (node_id, title, form, phenomenon, blocked_reason)
_EXTERNAL: list[tuple[str, str, str, str, str]] = [
    (
        "ext_ava_audiovisual",
        "Synchronized audiovisual same-event vs wrong-event",
        "audiovisual",
        "cross_form_conditional_information",
        "needs AVA-AudioSet clips with original clocks and rights",
    ),
    (
        "ext_ego4d",
        "Egocentric object/action persistence",
        "vision",
        "persistence",
        "needs Ego4D license and referent/track annotations",
    ),
    (
        "ext_something_something",
        "Temporal action-boundary formation on natural video",
        "vision",
        "temporal_boundary",
        "needs Something-Something-v2 access",
    ),
    (
        "ext_epic_kitchens",
        "Object-interaction relation state",
        "vision",
        "relation_role_state",
        "needs EPIC-KITCHENS with interaction labels",
    ),
    (
        "ext_musdb_sources",
        "Multi-source audio separation and counting on real mixtures",
        "native_audio",
        "source_object_tracking",
        "needs MUSDB18 or equivalent rights-clean stems",
    ),
    (
        "ext_code_execution",
        "Program-execution state prediction with an exact verifier",
        "symbolic",
        "predictive_state",
        "needs a code-trace corpus with exact execution oracle",
    ),
    (
        "ext_equation_transform",
        "Equation transformation and algebraic identity",
        "symbolic",
        "abstraction_factor_preservation",
        "needs a symbolic equation corpus and verifier",
    ),
    (
        "ext_graph_reasoning",
        "Graph transfer and role/filler exchange",
        "graph",
        "compositional_binding",
        "needs a controlled graph-reasoning generator with held-out splits",
    ),
    (
        "ext_partner_interaction",
        "Joint reference and communicative repair with held-out partners",
        "messages",
        "communication_joint_reference",
        "needs partner-interaction episodes or a partner simulator",
    ),
    (
        "ext_multi_agent_comm",
        "Communication protocol transfer to unseen partners",
        "messages",
        "teaching_transfer",
        "needs a multi-agent environment and partner population",
    ),
    (
        "ext_robot_trajectories",
        "Action-trajectory intervention ranking",
        "action",
        "causal_intervention",
        "needs robot or embodied trajectories with repeated initial states",
    ),
    (
        "ext_mot_tracking",
        "Multi-object tracking identity through split and merge",
        "vision",
        "split_merge_identity",
        "needs a MOT dataset with identity tracks",
    ),
    (
        "ext_telemetry_failure",
        "Operational failure prediction and recovery",
        "host_telemetry",
        "operational_self_monitoring",
        "needs bounded telemetry and failure episodes",
    ),
    (
        "ext_counterfactual_env",
        "Counterfactual branch consequence",
        "counterfactual",
        "counterfactual_consequence",
        "needs a controlled environment with exact interventions",
    ),
    (
        "ext_proprioceptive",
        "Proprioceptive embodied state in a controlled environment",
        "proprioception",
        "active_sensing",
        "needs an embodied simulator with proprioceptive channels",
    ),
    (
        "ext_teaching_culture",
        "Cultural accumulation across generations vs reset control",
        "messages",
        "teaching_cultural_accumulation",
        "needs a multi-generation partner-teaching environment",
    ),
]


def _science_nodes() -> list[NodeSpec]:
    out: list[NodeSpec] = []
    for node_id, ep, title, form, phen, mech, unit, level in _SCIENCE:
        cov = Coverage(
            form_family=form, phenomenon=phen, mechanism_family=mech, unit_class=unit, evidence_level=level
        )
        rules: tuple[DecisionRule, ...] = ()
        if node_id == "wave_h_routing":
            rules = (
                DecisionRule(
                    "survives",
                    {"field": "verdict", "equals": "survives"},
                    ("wave_h_routing_repro",),
                    "if competence routing survives, reproduce on fresh independent tasks",
                ),
            )
        elif node_id == "wave_e_memory":
            rules = (
                DecisionRule(
                    "survives",
                    {"field": "verdict", "equals": "survives"},
                    ("wave_e_memory_repro",),
                    "if replay improves the frontier, reproduce on a second drift regime",
                ),
            )
        out.append(
            ResearchQuestionSpec(
                node_id=node_id,
                title=title,
                entrypoint=ep,
                resources=_HASH,
                coverage=cov,
                decision_rules=rules,
                priority=50,
            )
        )
    # precommitted decision-branch reproductions (real runnable, gated behind a surviving parent)
    out.append(
        ReproductionSpec(
            node_id="wave_h_routing_repro",
            title="Wave H reproduction on fresh independent tasks",
            entrypoint="mop.campaign.nodes.wave_h_routing:wave_h_routing_runner",
            params={"fresh": True},
            resources=_HASH,
            authorities=("branch:wave_h_routing:survives",),
            coverage=Coverage(
                form_family="perspective",
                phenomenon="routing_value_of_computation",
                mechanism_family="routing",
                unit_class="synthetic_task",
                evidence_level="M3",
            ),
            priority=60,
        )
    )
    out.append(
        ReproductionSpec(
            node_id="wave_e_memory_repro",
            title="Wave E reproduction on a second drift regime",
            entrypoint="mop.campaign.nodes.wave_e_memory:wave_e_memory_runner",
            params={"fresh": True},
            resources=_HASH,
            authorities=("branch:wave_e_memory:survives",),
            coverage=Coverage(
                form_family="memory_episode",
                phenomenon="continual_learning",
                mechanism_family="replay",
                unit_class="synthetic_stream",
                evidence_level="M3",
            ),
            priority=60,
        )
    )
    return out


def build_campaign() -> CampaignSpec:
    nodes: list[NodeSpec] = []

    # 1. operational: receipt-invariance and throughput optimum (exclusive so it does not overlap).
    nodes.append(
        BedSpec(
            node_id="op_invariance",
            title="Receipt-invariance and throughput sweep across worker widths",
            entrypoint="mop.campaign.invariance:invariance_node_runner",
            params={"n_items": 4000, "widths": [1, 2, 4, 8]},
            resources=_EXCLUSIVE_SWEEP,
            coverage=Coverage(
                form_family="none",
                phenomenon="operational_self_monitoring",
                mechanism_family="resource_allocator",
                unit_class="worker_width",
                evidence_level="M0",
            ),
            priority=10,
        )
    )

    # 2. science question families (Waves A-J) and their precommitted reproductions.
    science = _science_nodes()
    nodes.extend(science)
    science_ids = [n.node_id for n in science if n.node_id.startswith("wave_")]

    # 3. analysis nodes.
    for node_id, ep, title, mech, unit in _ANALYSIS:
        nodes.append(
            NodeSpec(
                node_id=node_id,
                kind=NodeKind.ANALYSIS,
                title=title,
                entrypoint=ep,
                resources=_LIGHT,
                coverage=Coverage(
                    form_family="cross",
                    phenomenon="failure_synthesis",
                    mechanism_family=mech,
                    unit_class=unit,
                    evidence_level="M0",
                ),
                priority=80,
            )
        )
    # mechanism cards summarize the whole record: depend on the science frontier and the analyses.
    nodes.append(
        NodeSpec(
            node_id="analysis_mechanism_cards",
            kind=NodeKind.SYNTHESIS,
            title="Generate mechanism cards from sealed results (M0-M7 levels)",
            entrypoint="mop.campaign.nodes.analysis:mechanism_cards_runner",
            resources=_LIGHT,
            dependencies=tuple(Dependency(sid, DependencyKind.COMPLETION) for sid in science_ids)
            + (Dependency("analysis_diagnosis"), Dependency("analysis_negative_space")),
            coverage=Coverage(
                form_family="cross",
                phenomenon="mechanism_cataloguing",
                mechanism_family="cards",
                unit_class="sealed_result",
                evidence_level="M0",
            ),
            priority=90,
        )
    )
    nodes.append(
        NodeSpec(
            node_id="analysis_readiness",
            kind=NodeKind.READINESS,
            title="Executable Stage-3 substrate readiness gate over the twelve evidence gates",
            entrypoint="mop.campaign.nodes.analysis:readiness_gate_runner",
            resources=_LIGHT,
            dependencies=(Dependency("analysis_mechanism_cards"),),
            coverage=Coverage(
                form_family="cross",
                phenomenon="substrate_readiness",
                mechanism_family="readiness",
                unit_class="sealed_card",
                evidence_level="M0",
            ),
            priority=95,
        )
    )

    # 4. contracted external-input families (blocked on named data/authority; durably represented).
    for node_id, title, form, phen, reason in _EXTERNAL:
        nodes.append(
            BedSpec(
                node_id=node_id,
                title=title,
                entrypoint="mop.campaign.nodes.external:contracted",
                blocked_reason=reason,
                coverage=Coverage(
                    form_family=form,
                    phenomenon=phen,
                    mechanism_family="external_intake",
                    unit_class="external",
                    evidence_level="M0",
                ),
                priority=200,
            )
        )

    external_deps = (
        ExternalDependency(
            name="general-run-adopter",
            kind="live_process",
            match_label="general-run:adopter",
            est_cpu_workers=1,
            note="live General Run; adopted as an external resource consumer",
        ),
        ExternalDependency(
            name="horizon-v2-supervisor",
            kind="live_process",
            match_label="mop-supervisor:generation1-successor-horizon-v2",
            est_cpu_workers=8,
            note="live successor horizon chain; observed, never signaled",
        ),
    )
    return CampaignSpec(
        campaign_id=CAMPAIGN_ID,
        title="MOP pre-substrate discovery campaign",
        nodes=tuple(nodes),
        external_dependencies=external_deps,
        coverage_targets={
            "form_families": 6,
            "phenomena": 10,
            "local_question_families": 24,
            "external_families": 16,
        },
    )
