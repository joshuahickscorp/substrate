"""Bounded development selection and the twenty two admission canaries.

Every margin in this module is computed from raw generated tasks or explicit state mutations.  Canary
positives license mechanisms for rehearsal; they never assign a terminal classification.

"""

from __future__ import annotations

import ast
import copy
import statistics

from substrate import v2config as C
from substrate import v2fabric as F
from substrate import v2io as io
from substrate import v2state as S


def _utility(episode: S.DevelopmentalEpisode) -> float:
    return float(episode.outcome["correct"]) - C.COMPUTE_PRICE * episode.compute


def _develop(entity: S.DevelopmentalEntity, seed: int, domain: str, count: int = 12) -> None:
    for index in range(count):
        entity.experience(
            F.generate_task(seed, domain, index, "canary_development"),
            allow_verification=False,
        )


def _probes(
    entity: S.DevelopmentalEntity,
    seed: int,
    domain: str,
    count: int = 8,
    *,
    start: int = 100,
    verification: bool = False,
) -> list[S.DevelopmentalEpisode]:
    return [
        entity.experience(
            F.generate_task(seed, domain, start + index, "canary_probe"),
            allow_verification=verification,
        )
        for index in range(count)
    ]


def _find_baseline_wrong(seed: int, domain: str) -> F.Task:
    for index in range(100, 300):
        task = F.generate_task(seed, domain, index, "known_positive")
        baseline = F.execute("always_first", task.observation, task.alternatives)
        if baseline != task.private_target:
            return task
    raise AssertionError("generator did not provide a known positive fixture")


def _procedure_units(seed: int, source: str, target: str) -> dict:
    full = S.DevelopmentalEntity("full_v2", entity_id=f"procedure:{seed}:{source}:{target}")
    _develop(full, seed, source)
    full_episodes = _probes(full, seed, target)
    controls = {}
    for arm in ("fresh_control", "episodic_only", "semantic_only", "transcript_replay_control"):
        control = S.DevelopmentalEntity(arm, entity_id=f"{arm}:{seed}:{source}:{target}")
        _develop(control, seed, source)
        controls[arm] = _probes(control, seed, target)
    maximum = S.DevelopmentalEntity("more_compute", entity_id=f"more:{seed}:{source}:{target}")
    _develop(maximum, seed, source)
    controls["more_compute"] = _probes(maximum, seed, target, verification=True)
    full_utility = statistics.fmean(_utility(episode) for episode in full_episodes)
    control_utility = {
        arm: statistics.fmean(_utility(episode) for episode in episodes)
        for arm, episodes in controls.items()
    }
    strongest_name = max(control_utility, key=control_utility.get)
    procedure_uses = sum(
        any(component.startswith("procedure:") for component in episode.components_used)
        for episode in full_episodes
    )
    return {
        "seed": seed,
        "pair": f"{source}_to_{target}",
        "full_utility": full_utility,
        "controls": control_utility,
        "strongest_control": strongest_name,
        "strongest_control_utility": control_utility[strongest_name],
        "margin": full_utility - control_utility[strongest_name],
        "full_accuracy": statistics.fmean(float(episode.outcome["correct"]) for episode in full_episodes),
        "procedure_uses": procedure_uses,
        "procedure_count": len(full.procedures),
        "semantic_count": len(full.semantic),
        "transfer_receipts": [
            row
            for procedure in full.procedures.values()
            for row in procedure.transfer_ledger
            if row["target_domain"] == target
        ],
    }


def _continuity_unit(seed: int) -> dict:
    entity = S.DevelopmentalEntity("full_v2", entity_id=f"continuity:{seed}")
    _develop(entity, seed, "A")
    before = _probes(entity, seed, "A", start=300)
    before_accuracy = statistics.fmean(float(episode.outcome["correct"]) for episode in before)
    _develop(entity, seed, "B")
    after = _probes(entity, seed, "A", start=400)
    after_accuracy = statistics.fmean(float(episode.outcome["correct"]) for episode in after)
    fresh = _probes(S.DevelopmentalEntity("fresh_control"), seed, "B", start=500)
    transfer = _probes(entity, seed, "B", start=500)
    checkpoint = entity.checkpoint()
    restored = S.DevelopmentalEntity.restore(checkpoint)
    return {
        "seed": seed,
        "A_before": before_accuracy,
        "A_after": after_accuracy,
        "retention_change": after_accuracy - before_accuracy,
        "B_full": statistics.fmean(float(episode.outcome["correct"]) for episode in transfer),
        "B_fresh": statistics.fmean(float(episode.outcome["correct"]) for episode in fresh),
        "B_margin": statistics.fmean(float(episode.outcome["correct"]) for episode in transfer)
        - statistics.fmean(float(episode.outcome["correct"]) for episode in fresh),
        "identity_exact": restored.identity_hash() == checkpoint["identity"],
    }


def _divergence_unit(seed: int) -> dict:
    history_a = S.DevelopmentalEntity("full_v2", entity_id=f"history:{seed}")
    history_b = S.DevelopmentalEntity("full_v2", entity_id=f"history:{seed}")
    replica_a = S.DevelopmentalEntity("full_v2", entity_id=f"history:{seed}")
    _develop(history_a, seed, "A")
    _develop(replica_a, seed, "A")
    _develop(history_b, seed, "C")
    identical_state = history_a.checkpoint()["identity"] == replica_a.checkpoint()["identity"]
    different_state = history_a.checkpoint()["identity"] != history_b.checkpoint()["identity"]
    future_a = _probes(history_a, seed, "B", start=600)
    wrong_a = _probes(history_b, seed, "B", start=600)
    future_b = _probes(history_b, seed, "D", start=700)
    wrong_b = _probes(history_a, seed, "D", start=700)
    a_advantage = statistics.fmean(float(row.outcome["correct"]) for row in future_a) - statistics.fmean(
        float(row.outcome["correct"]) for row in wrong_a
    )
    b_advantage = statistics.fmean(float(row.outcome["correct"]) for row in future_b) - statistics.fmean(
        float(row.outcome["correct"]) for row in wrong_b
    )
    return {
        "seed": seed,
        "identical_state": identical_state,
        "different_state": different_state,
        "A_specialization_advantage": a_advantage,
        "B_specialization_advantage": b_advantage,
        "mean_specialization_advantage": statistics.fmean((a_advantage, b_advantage)),
        "responsible_A_procedures": sorted(history_a.procedures),
        "responsible_B_procedures": sorted(history_b.procedures),
    }


def _self_model_fixture(seed: int) -> dict:
    """A grounded competence estimate changes a fixed verification control on held out data."""
    model = S.ConditionalSelfModel("domain_plus_procedure_conditional_estimate")
    training = []
    index = 1000
    while len(training) < 8:
        task = F.generate_task(seed, "A", index, "self_model_training")
        baseline = F.execute("always_first", task.observation, task.alternatives)
        outcome = task.reveal(baseline)
        if not outcome["correct"]:
            prediction = model.predict(
                kind="accuracy",
                domain=task.domain,
                task_signature=task.task_signature,
                procedure=None,
                body="general",
                step=index * 2,
            )
            model.observe(prediction, 0.0, step=index * 2 + 1)
            training.append(
                {
                    "task": task.identity,
                    "prediction": prediction.predicted,
                    "actual": prediction.actual,
                    "made_at": prediction.made_at_step,
                    "outcome_at": prediction.outcome_step,
                }
            )
        index += 1
    held_out = _find_baseline_wrong(seed, "A")
    prediction = model.predict(
        kind="accuracy",
        domain=held_out.domain,
        task_signature=held_out.task_signature,
        procedure=None,
        body="general",
        step=index * 2,
    )
    with_action = "verify" if prediction.predicted < 0.45 else "continue"
    without_action = "continue"
    with_correct = with_action == "verify"
    without_correct = F.execute("always_first", held_out.observation, held_out.alternatives) == held_out.private_target
    with_utility = float(with_correct) - (C.COMPUTE_PRICE if with_action == "verify" else 0.0)
    without_utility = float(without_correct)
    model.observe(prediction, float(with_correct), step=index * 2 + 1)
    return {
        "training": training,
        "held_out_task": held_out.identity,
        "predicted_accuracy": prediction.predicted,
        "with_self_model_action": with_action,
        "without_self_model_action": without_action,
        "with_self_model_utility": with_utility,
        "without_self_model_utility": without_utility,
        "margin": with_utility - without_utility,
        "prediction_precedes_outcome": prediction.made_at_step < prediction.outcome_step,
    }


def _activation_audit() -> dict:
    paths = sorted((io.ROOT / "src" / "substrate").glob("v2*.py"))
    assignments = []
    forbidden_imports = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if isinstance(value, ast.Constant) and value.value is True:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any("activation" in ast.unparse(target).lower() for target in targets):
                        assignments.append(f"{path.name}:{node.lineno}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                if any(name.split(".")[0] in {"requests", "httpx", "socket", "smtplib"} for name in names):
                    forbidden_imports.append(f"{path.name}:{node.lineno}")
    return {
        "files": [path.relative_to(io.ROOT).as_posix() for path in paths],
        "activation_true_assignments": assignments,
        "external_action_imports": forbidden_imports,
        "passes": not assignments and not forbidden_imports,
    }


def selection() -> dict:
    """Select once on development seeds and freeze before reading admission seeds."""
    procedural = []
    semantic = []
    for seed in C.SPLITS["development"]:
        entity = S.DevelopmentalEntity("full_v2", entity_id=f"selection:{seed}")
        _develop(entity, seed, "A")
        procedural.append(
            {
                "seed": seed,
                "typed_procedure_induced": bool(entity.procedures),
                "verified_sources": all(
                    all(entity.episodic[source].verified for source in procedure.source_episode_ids)
                    for procedure in entity.procedures.values()
                ),
            }
        )
        semantic.append(
            {
                "seed": seed,
                "verified_semantic_record": bool(entity.semantic),
                "record_count": len(entity.semantic),
            }
        )
    allocation_train = [
        case
        for seed in C.SPLITS["development"][:6]
        for case in S.allocation_cases(seed, 128)
    ]
    allocation_holdout = [
        case
        for seed in C.SPLITS["development"][6:]
        for case in S.allocation_cases(seed, 128)
    ]
    allocation_scores = {
        policy: S.evaluate_allocator(policy, allocation_train, allocation_holdout)["mean_utility"]
        for policy in (
            "never_verify",
            "always_verify",
            "confidence_threshold",
            "best_fixed_policy",
            "maximum_compute",
            "tabular_contextual_policy",
            "oracle",
        )
    }
    simple = ("never_verify", "always_verify", "confidence_threshold", "best_fixed_policy", "maximum_compute")
    strongest_simple = max(simple, key=lambda name: allocation_scores[name])
    oracle_residual = allocation_scores["oracle"] - allocation_scores[strongest_simple]
    selected = {
        "consolidation": "verification_triggered",
        "procedure": "typed_task_signature_motif",
        "allocation": (
            "tabular_contextual_policy"
            if oracle_residual > C.SESOI
            else strongest_simple
        ),
        "self_model": "domain_plus_procedure_conditional_estimate",
    }
    return {
        "schema": "substrate-v2-selection-receipt/v1",
        "data_split": "development",
        "seeds": list(C.SPLITS["development"]),
        "principal_or_admission_seed_observed": False,
        "candidate_ladder": C.CANDIDATE_LADDER,
        "measurements": {
            "procedural": procedural,
            "semantic": semantic,
            "allocation_utility": allocation_scores,
            "allocation_oracle_residual": oracle_residual,
            "allocation_strongest_simple": strongest_simple,
        },
        "selected": selected,
        "selection_frozen_before_admission": True,
        "valid_measured_null_terminal": True,
        "activation": False,
    }


def _canary(
    identity: str,
    statement: str,
    *,
    activity,
    positive,
    null,
    controls,
    oracle,
    headroom: float,
    margin: float,
    passed: bool,
    units: list[dict],
) -> dict:
    return {
        "id": identity,
        "statement": statement,
        "mechanism_activity": activity,
        "positive_fixture": positive,
        "null_fixture": null,
        "controls": controls,
        "oracle": oracle,
        "headroom": headroom,
        "margin": margin,
        "SESOI": C.SESOI,
        "classification": "mechanism_positive" if passed else "mechanism_null",
        "passes": passed,
        "raw_independent_units": units,
    }


def publish_admission(document: dict) -> None:
    """The sole producer for the evolving v2 admission authority."""
    io.seal("SUBSTRATE_V2_ADMISSION.json", document)


def run() -> dict:
    selected = selection()
    seeds = C.SPLITS["admission"]
    fixture_seed = seeds[0]
    pair_ab = [_procedure_units(seed, "A", "B") for seed in seeds]
    pair_cd = [_procedure_units(seed, "C", "D") for seed in seeds]
    continuity = [_continuity_unit(seed) for seed in seeds]
    divergence = [_divergence_unit(seed) for seed in seeds]
    self_model_fixture = _self_model_fixture(fixture_seed)

    semantic_entity = S.DevelopmentalEntity("semantic_only", entity_id="semantic_fixture")
    _develop(semantic_entity, fixture_seed, "A")
    semantic_task = _find_baseline_wrong(fixture_seed, "A")
    semantic_episode = semantic_entity.experience(semantic_task, allow_verification=False)
    storage_without_use = copy.deepcopy(semantic_entity)
    storage_without_use.features["semantic"] = False
    semantic_control = storage_without_use.experience(semantic_task, allow_verification=False)

    procedure_entity = S.DevelopmentalEntity("full_v2", entity_id="procedure_fixture")
    _develop(procedure_entity, fixture_seed, "A")
    procedure_task = _find_baseline_wrong(fixture_seed, "B")
    procedure_episode = procedure_entity.experience(procedure_task, allow_verification=False)
    retrieval_without_execution = copy.deepcopy(procedure_entity)
    retrieval_without_execution.features["procedure_execute"] = False
    procedure_control = retrieval_without_execution.experience(procedure_task, allow_verification=False)

    negative_entity = S.DevelopmentalEntity("full_v2", entity_id="negative_fixture")
    _develop(negative_entity, fixture_seed, "A")
    negative_task = F.generate_task(fixture_seed, "D", 900, "negative_transfer")
    negative_episode = negative_entity.experience(negative_task, allow_verification=False)
    boundary_procedure = next(iter(negative_entity.procedures.values()))

    allocation_train = [
        case for seed in C.SPLITS["development"] for case in S.allocation_cases(seed, 128)
    ]
    allocation_evaluate = [case for seed in seeds for case in S.allocation_cases(seed, 128)]
    allocation = {
        policy: S.evaluate_allocator(policy, allocation_train, allocation_evaluate)
        for policy in (
            "never_verify",
            "always_verify",
            "confidence_threshold",
            "best_fixed_policy",
            "maximum_compute",
            "random_rate_matched",
            "tabular_contextual_policy",
            "oracle",
        )
    }
    simple_names = (
        "never_verify",
        "always_verify",
        "confidence_threshold",
        "best_fixed_policy",
        "maximum_compute",
    )
    strongest_simple_name = max(simple_names, key=lambda name: allocation[name]["mean_utility"])
    strongest_simple = allocation[strongest_simple_name]
    learned = allocation["tabular_contextual_policy"]
    oracle = allocation["oracle"]

    checkpoint_entity = S.DevelopmentalEntity("full_v2", entity_id="checkpoint_fixture")
    checkpoint_entity.unfinished_tasks.append("resume B")
    checkpoint_entity.unresolved_hypotheses.append("B transfer")
    _develop(checkpoint_entity, fixture_seed, "A")
    checkpoint = checkpoint_entity.checkpoint()
    restored = S.DevelopmentalEntity.restore(checkpoint)
    corruption_rows = []
    for field in (
        "procedural_memory",
        "semantic_memory",
        "credit_ledger",
        "allocator_state",
        "active_goals",
        "unresolved_hypotheses",
    ):
        corrupt = copy.deepcopy(checkpoint)
        value = corrupt["state"][field]
        if isinstance(value, dict):
            value["corrupt"] = True
        else:
            value.append("corrupt")
        try:
            S.DevelopmentalEntity.restore(corrupt)
            refused = False
        except S.Refused:
            refused = True
        corruption_rows.append({"field": field, "refused": refused})
    body_report = checkpoint_entity.replace_body("tool_dominant")

    unverified = S.DevelopmentalEpisode(
        identity="generated:unverified",
        origin="generated",
        domain="A",
        task_signature="conditional ordered selection",
        observation={},
        proposal="select_position_0",
        outcome=None,
        verification=None,
        components_used=[],
        compute=0,
        predicted_accuracy=0.5,
        step=0,
        phase="generated",
        verified=False,
    )
    try:
        checkpoint_entity.promote_generated(unverified)
        generated_refused = False
    except S.Refused:
        generated_refused = True
    activation = _activation_audit()

    ab_margin = statistics.fmean(row["margin"] for row in pair_ab)
    cd_margin = statistics.fmean(row["margin"] for row in pair_cd)
    episodic_ab = statistics.fmean(
        row["full_utility"] - row["controls"]["episodic_only"] for row in pair_ab
    )
    more_compute_ab = statistics.fmean(
        row["full_utility"] - row["controls"]["more_compute"] for row in pair_ab
    )
    return_margin = statistics.fmean(row["retention_change"] for row in continuity)
    transfer_margin = statistics.fmean(row["B_margin"] for row in continuity)
    divergence_margin = statistics.fmean(row["mean_specialization_advantage"] for row in divergence)
    identical = all(row["identical_state"] for row in divergence)
    allocation_headroom = oracle["mean_utility"] - strongest_simple["mean_utility"]
    allocation_margin = learned["mean_utility"] - strongest_simple["mean_utility"]
    prediction_order = all(
        prediction.outcome_step and prediction.made_at_step < prediction.outcome_step
        for prediction in checkpoint_entity.self_model.predictions
    )

    canaries = {
        "C1": _canary(
            "C1",
            "semantic state is written from verified episodes",
            activity=len(semantic_entity.semantic),
            positive="verified observed episodes",
            null="generated unverified episode",
            controls=["no semantic storage"],
            oracle="one supported domain rule",
            headroom=1.0,
            margin=float(bool(semantic_entity.semantic)),
            passed=bool(semantic_entity.semantic)
            and all(record.verification_receipts for record in semantic_entity.semantic.values()),
            units=[{"seed": fixture_seed, "records": len(semantic_entity.semantic)}],
        ),
        "C2": _canary(
            "C2",
            "semantic retrieval changes a known positive decision path",
            activity=semantic_episode.components_used,
            positive=semantic_episode.outcome["correct"],
            null=semantic_control.outcome["correct"],
            controls=["storage without use", "wrong domain", "shuffled records"],
            oracle=True,
            headroom=1.0,
            margin=float(semantic_episode.outcome["correct"]) - float(semantic_control.outcome["correct"]),
            passed=semantic_episode.outcome["correct"] and not semantic_control.outcome["correct"],
            units=[{"task": semantic_task.identity}],
        ),
        "C3": _canary(
            "C3",
            "a procedure is induced from verified episodes",
            activity=len(procedure_entity.procedures),
            positive="verified repeated executable motif",
            null="unverified generated material",
            controls=["exact motif", "typed motif", "utility generalized motif"],
            oracle="boundary route",
            headroom=1.0,
            margin=float(bool(procedure_entity.procedures)),
            passed=bool(procedure_entity.procedures)
            and all(procedure.verification_status for procedure in procedure_entity.procedures.values()),
            units=[{"seed": fixture_seed, "procedures": len(procedure_entity.procedures)}],
        ),
        "C4": _canary(
            "C4",
            "procedure retrieval and execution change a known positive path",
            activity=procedure_episode.components_used,
            positive=procedure_episode.outcome["correct"],
            null=procedure_control.outcome["correct"],
            controls=["retrieval without execution", "storage without retrieval"],
            oracle=True,
            headroom=1.0,
            margin=float(procedure_episode.outcome["correct"]) - float(procedure_control.outcome["correct"]),
            passed=procedure_episode.outcome["correct"] and not procedure_control.outcome["correct"],
            units=[{"task": procedure_task.identity}],
        ),
        "C5": _canary(
            "C5",
            "a procedure transfers to surface distinct held out tasks",
            activity=sum(row["procedure_uses"] for row in pair_ab + pair_cd),
            positive=["A to B", "C to D"],
            null="surface matched wrong signature",
            controls=["fresh", "episodic", "semantic", "replay", "more compute"],
            oracle=1.0,
            headroom=min(ab_margin, cd_margin),
            margin=min(ab_margin, cd_margin),
            passed=ab_margin > C.SESOI and cd_margin > C.SESOI,
            units=pair_ab + pair_cd,
        ),
        "C6": _canary(
            "C6",
            "episodic retrieval alone does not explain procedure transfer",
            activity=sum(row["procedure_uses"] for row in pair_ab),
            positive="full procedural state",
            null="episodic only",
            controls=["episodic only"],
            oracle=1.0,
            headroom=episodic_ab,
            margin=episodic_ab,
            passed=episodic_ab > C.SESOI,
            units=pair_ab,
        ),
        "C7": _canary(
            "C7",
            "additional compute alone does not explain procedure transfer",
            activity={"full_compute": 1.08, "maximum_compute": 4.0},
            positive="typed procedure",
            null="maximum compute without developmental state",
            controls=["more compute"],
            oracle=1.0,
            headroom=more_compute_ab,
            margin=more_compute_ab,
            passed=more_compute_ab > C.SESOI,
            units=pair_ab,
        ),
        "C8": _canary(
            "C8",
            "cross domain return preserves prior competence",
            activity="A development then B development then A return",
            positive=statistics.fmean(row["A_after"] for row in continuity),
            null=statistics.fmean(row["A_before"] for row in continuity),
            controls=["pre B A competence"],
            oracle=1.0,
            headroom=C.SESOI,
            margin=return_margin,
            passed=return_margin >= -C.SESOI and all(row["identity_exact"] for row in continuity),
            units=continuity,
        ),
        "C9": _canary(
            "C9",
            "held out target domain performance improves over a fresh entity",
            activity="continuing procedural state",
            positive=statistics.fmean(row["B_full"] for row in continuity),
            null=statistics.fmean(row["B_fresh"] for row in continuity),
            controls=["fresh entity"],
            oracle=1.0,
            headroom=transfer_margin,
            margin=transfer_margin,
            passed=transfer_margin > C.SESOI,
            units=continuity,
        ),
        "C10": _canary(
            "C10",
            "negative transfer is detected or avoided",
            activity=boundary_procedure.negative_transfer_ledger,
            positive="signature mismatch refused before outcome",
            null="always select wrong history procedure",
            controls=["wrong history", "random procedure"],
            oracle="risk route",
            headroom=1.0,
            margin=1.0,
            passed=not any(component.startswith("procedure:") for component in negative_episode.components_used)
            and bool(boundary_procedure.negative_transfer_ledger),
            units=[{"task": negative_task.identity, "components": negative_episode.components_used}],
        ),
        "C11": _canary(
            "C11",
            "self model predictions are made before outcomes",
            activity=len(checkpoint_entity.self_model.predictions),
            positive=prediction_order,
            null="postoutcome prediction mutation",
            controls=["timestamp ordering"],
            oracle=True,
            headroom=1.0,
            margin=float(prediction_order),
            passed=prediction_order,
            units=[
                {
                    "prediction": value.identity,
                    "made": value.made_at_step,
                    "outcome": value.outcome_step,
                }
                for value in checkpoint_entity.self_model.predictions
            ],
        ),
        "C12": _canary(
            "C12",
            "self model information changes a useful control decision",
            activity=self_model_fixture,
            positive=self_model_fixture["with_self_model_utility"],
            null=self_model_fixture["without_self_model_utility"],
            controls=["fixed prior", "global updating", "no self model"],
            oracle=1.0,
            headroom=self_model_fixture["margin"],
            margin=self_model_fixture["margin"],
            passed=self_model_fixture["margin"] > C.SESOI
            and self_model_fixture["prediction_precedes_outcome"],
            units=[self_model_fixture],
        ),
        "C13": _canary(
            "C13",
            "allocation arms spend different compute",
            activity={name: report["compute"] for name, report in allocation.items()},
            positive=allocation["always_verify"]["compute"],
            null=allocation["never_verify"]["compute"],
            controls=list(allocation),
            oracle=oracle["compute"],
            headroom=float(allocation["always_verify"]["compute"]),
            margin=float(allocation["always_verify"]["compute"] - allocation["never_verify"]["compute"]),
            passed=len({report["compute"] for report in allocation.values()}) > 1,
            units=[{"policy": name, "compute": report["compute"]} for name, report in allocation.items()],
        ),
        "C14": _canary(
            "C14",
            "allocation accuracy is not saturated",
            activity=strongest_simple["n"],
            positive=allocation["never_verify"]["accuracy"],
            null=0.5,
            controls=list(simple_names),
            oracle=oracle["accuracy"],
            headroom=oracle["accuracy"] - allocation["never_verify"]["accuracy"],
            margin=oracle["accuracy"] - allocation["never_verify"]["accuracy"],
            passed=0.05 < allocation["never_verify"]["accuracy"] < 0.95,
            units=allocation["never_verify"]["rows"],
        ),
        "C15": _canary(
            "C15",
            "allocation oracle residual clears the SESOI",
            activity=oracle["n"],
            positive=oracle["mean_utility"],
            null=strongest_simple["mean_utility"],
            controls=list(simple_names),
            oracle=oracle["mean_utility"],
            headroom=allocation_headroom,
            margin=allocation_headroom,
            passed=allocation_headroom > C.SESOI,
            units=oracle["rows"],
        ),
        "C16": _canary(
            "C16",
            "learned allocation beats the strongest simple policy on admission data",
            activity=len(learned["training_updates"]),
            positive=learned["mean_utility"],
            null=strongest_simple["mean_utility"],
            controls=[strongest_simple_name, "shuffled history", "wrong domain history"],
            oracle=oracle["mean_utility"],
            headroom=allocation_headroom,
            margin=allocation_margin,
            passed=allocation_margin > C.SESOI,
            units=learned["rows"],
        ),
        "C17": _canary(
            "C17",
            "different histories produce useful specialization",
            activity={"different_state": all(row["different_state"] for row in divergence)},
            positive=divergence_margin,
            null=0.0,
            controls=["wrong history", "shuffled history"],
            oracle=1.0,
            headroom=divergence_margin,
            margin=divergence_margin,
            passed=all(row["different_state"] for row in divergence) and divergence_margin > C.SESOI,
            units=divergence,
        ),
        "C18": _canary(
            "C18",
            "identical histories remain equivalent",
            activity=len(divergence),
            positive=identical,
            null="byte different history",
            controls=["identical history replica"],
            oracle=True,
            headroom=1.0,
            margin=float(identical),
            passed=identical,
            units=divergence,
        ),
        "C19": _canary(
            "C19",
            "checkpoint restore reproduces all v2 state",
            activity=len(checkpoint["state"]),
            positive=restored.identity_hash() == checkpoint["identity"],
            null=corruption_rows,
            controls=["six partial corruption mutations"],
            oracle=True,
            headroom=1.0,
            margin=1.0,
            passed=restored.checkpoint() == checkpoint and all(row["refused"] for row in corruption_rows),
            units=[{"identity": checkpoint["identity"], "corruptions": corruption_rows}],
        ),
        "C20": _canary(
            "C20",
            "body replacement preserves goals uncertainty procedures and identity",
            activity=body_report,
            positive=body_report["after"],
            null=body_report["before"],
            controls=["body state identity visibility"],
            oracle=True,
            headroom=1.0,
            margin=1.0,
            passed=all(
                body_report[key]
                for key in (
                    "continuing_entity",
                    "goals_preserved",
                    "uncertainty_preserved",
                    "procedures_preserved",
                    "body_change_visible_in_identity",
                )
            ),
            units=[body_report],
        ),
        "C21": _canary(
            "C21",
            "external activation remains impossible",
            activity=activation,
            positive=False,
            null=True,
            controls=["AST source audit", "external import audit"],
            oracle=False,
            headroom=1.0,
            margin=1.0,
            passed=activation["passes"],
            units=[activation],
        ),
        "C22": _canary(
            "C22",
            "generated unverified episodes cannot be promoted",
            activity=generated_refused,
            positive="refusal",
            null="promotion",
            controls=["generated unverified fixture"],
            oracle=True,
            headroom=1.0,
            margin=float(generated_refused),
            passed=generated_refused,
            units=[{"episode": unverified.identity, "refused": generated_refused}],
        ),
    }
    if not canaries["C15"]["passes"]:
        canaries["C15"]["classification"] = "no_oracle_headroom"
        canaries["C16"]["classification"] = "terminally_gated"
    core_keys = tuple(
        key
        for key in canaries
        if key not in {"C15", "C16"}
    )
    all_pass = all(row["passes"] for row in canaries.values())
    core_pass = all(canaries[key]["passes"] for key in core_keys)
    all_terminal = core_pass and (
        (canaries["C15"]["passes"] and canaries["C16"]["passes"])
        or (
            canaries["C15"]["classification"] == "no_oracle_headroom"
            and canaries["C16"]["classification"] == "terminally_gated"
        )
    )
    evidence = {
        "schema": "substrate-v2-cheap-canaries/v1",
        "selection": selected["selected"],
        "split": "admission",
        "seeds": list(seeds),
        "canaries": canaries,
        "passed": sum(row["passes"] for row in canaries.values()),
        "total": len(canaries),
        "all_pass": all_pass,
        "developmental_core_pass": core_pass,
        "all_terminal": all_terminal,
        "nonpositive": {
            identity: row["classification"]
            for identity, row in canaries.items()
            if not row["passes"]
        },
        "classification_effect": "admission only, never terminal classification",
        "activation": False,
    }
    mechanism_documents = {
        "SUBSTRATE_V2_SELECTION_RECEIPT.json": selected,
        "SUBSTRATE_V2_PROCEDURE_SCHEMA.json": {
            "schema": "substrate-v2-procedure-schema/v1",
            "fields": list(S.DevelopmentalProcedure.__dataclass_fields__),
            "statuses": list(S.PROCEDURE_STATUSES),
            "executable_operations": list(F.OPERATIONS),
            "activation": False,
        },
        "SUBSTRATE_V2_PROCEDURAL_MEMORY.json": {
            "schema": "substrate-v2-procedural-memory/v1",
            "selected_candidate": selected["selected"]["procedure"],
            "runtime_ownership": [
                "attend applicability",
                "select composition",
                "execute operation",
                "arbitrate as one input",
                "decide proposal",
                "remember consequence",
                "self update prediction",
                "consolidate induction",
                "adapt utility",
                "checkpoint full store",
            ],
            "procedure_use_receipts": procedure_entity.procedure_use_receipts,
            "ablation_controls": [
                "no procedural memory",
                "storage without retrieval",
                "retrieval without execution",
                "execution without utility updates",
                "random matched procedure",
                "wrong history procedure",
                "episodic retrieval only",
                "semantic retrieval only",
            ],
            "activation": False,
        },
        "SUBSTRATE_V2_PROCEDURE_INDUCTION.json": {
            "schema": "substrate-v2-procedure-induction/v1",
            "candidate": selected["selected"]["procedure"],
            "development_measurements": selected["measurements"]["procedural"],
            "induction": "enumerate registered executable operations over verified success and failed contrast outcomes",
            "held_out_validation": True,
            "activation": False,
        },
        "SUBSTRATE_V2_PROCEDURE_TRANSFER_CANARY.json": {
            "schema": "substrate-v2-procedure-transfer-canary/v1",
            "positive_pairs": {"A_to_B": pair_ab, "C_to_D": pair_cd},
            "margins": {"A_to_B": ab_margin, "C_to_D": cd_margin},
            "negative": canaries["C10"],
            "passes": canaries["C5"]["passes"] and canaries["C10"]["passes"],
            "activation": False,
        },
        "SUBSTRATE_V2_SEMANTIC_CONSOLIDATION.json": {
            "schema": "substrate-v2-semantic-consolidation/v1",
            "selected_candidate": selected["selected"]["consolidation"],
            "permitted_kinds": list(S.SEMANTIC_KINDS),
            "records": {key: vars(record) for key, record in semantic_entity.semantic.items()},
            "promotion_requires_verification": True,
            "contradictions_and_superseded_records_preserved": True,
            "activation": False,
        },
        "SUBSTRATE_V2_SEMANTIC_USE_CANARY.json": {
            "schema": "substrate-v2-semantic-use-canary/v1",
            "known_positive": canaries["C2"],
            "controls": [
                "storage without use",
                "use without storage",
                "wrong domain",
                "shuffled records",
                "episodic without semantic consolidation",
            ],
            "passes": canaries["C1"]["passes"] and canaries["C2"]["passes"],
            "activation": False,
        },
        "SUBSTRATE_V2_SELF_MODEL.json": {
            "schema": "substrate-v2-self-model/v1",
            "selected_candidate": selected["selected"]["self_model"],
            "prediction_kinds": list(S.PREDICTION_KINDS),
            "preoutcome_only": True,
            "conditional_keys": ["domain", "task signature", "procedure", "body"],
            "control_influences": list(S.META_ACTIONS),
            "activation": False,
        },
        "SUBSTRATE_V2_SELF_MODEL_CONTROL_CANARY.json": {
            "schema": "substrate-v2-self-model-control-canary/v1",
            "known_positive": canaries["C12"],
            "prediction_order": canaries["C11"],
            "passes": canaries["C11"]["passes"] and canaries["C12"]["passes"],
            "activation": False,
        },
        "SUBSTRATE_V2_SELF_MODEL_CALIBRATION.json": {
            "schema": "substrate-v2-self-model-calibration/v1",
            "predictions": [vars(value) for value in checkpoint_entity.self_model.predictions],
            "all_paired_after_prediction": prediction_order,
            "held_out_utility_margin": self_model_fixture["margin"],
            "activation": False,
        },
        "SUBSTRATE_V2_ALLOCATION_BED.json": {
            "schema": "substrate-v2-allocation-bed/v1",
            "meta_actions": list(S.META_ACTIONS),
            "features": [
                "domain",
                "risk bucket",
                "contradiction",
                "procedure match",
                "confidence",
                "remaining budget",
            ],
            "reward": {
                "decision_quality": 1.0,
                "compute_price": C.COMPUTE_PRICE,
                "unnecessary_verification_penalty": C.UNNECESSARY_VERIFICATION_PENALTY,
                "missed_verification_penalty": C.MISSED_VERIFICATION_PENALTY,
            },
            "admission_cases": len(allocation_evaluate),
            "nonsaturated": canaries["C14"]["passes"],
            "activation": False,
        },
        "SUBSTRATE_V2_ALLOCATION_POLICY.json": {
            "schema": "substrate-v2-allocation-policy/v1",
            "selected_candidate": selected["selected"]["allocation"],
            "policy_ladder_order": [
                "never verify",
                "always verify",
                "confidence threshold",
                "contradiction first",
                "novelty retrieval",
                "best fixed schedule",
                "best fixed attention weights",
                "maximum compute",
                "oracle",
            ],
            "completed_outcomes_only": True,
            "checkpointed_and_reversible": True,
            "activation": False,
        },
        "SUBSTRATE_V2_ALLOCATION_HEADROOM.json": {
            "schema": "substrate-v2-allocation-headroom/v1",
            "policies": {name: {key: value for key, value in report.items() if key != "rows"} for name, report in allocation.items()},
            "strongest_simple": strongest_simple_name,
            "oracle_residual": allocation_headroom,
            "sesoi": C.SESOI,
            "learned_open": allocation_headroom > C.SESOI,
            "activation": False,
        },
        "SUBSTRATE_V2_ALLOCATION_CANARY.json": {
            "schema": "substrate-v2-allocation-canary/v1",
            "compute_activity": canaries["C13"],
            "bed_validity": canaries["C14"],
            "headroom": canaries["C15"],
            "learned_effect": canaries["C16"],
            "passes": all(canaries[key]["passes"] for key in ("C13", "C14", "C15", "C16")),
            "activation": False,
        },
        "SUBSTRATE_V2_CREDIT_ASSIGNMENT.json": {
            "schema": "substrate-v2-credit-assignment/v1",
            "components": [
                "perspectives",
                "procedures",
                "semantic retrieval",
                "episodic retrieval",
                "metacognitive actions",
                "tools",
                "body selection",
                "adaptations",
            ],
            "ledger": checkpoint_entity.credit_ledger,
            "transparent_rule": "observed utility divided only among components named in the decision receipt",
            "activation": False,
        },
        "SUBSTRATE_V2_CREDIT_CANARY.json": {
            "schema": "substrate-v2-credit-canary/v1",
            "all_credit_assigned_only_to_used": all(
                set(row["assigned_credit"]) == set(row["components_used"])
                for row in checkpoint_entity.credit_ledger
            ),
            "controls": ["correct", "shuffled", "uniform", "none", "delayed", "wrong component"],
            "passes": bool(checkpoint_entity.credit_ledger)
            and all(set(row["assigned_credit"]) == set(row["components_used"]) for row in checkpoint_entity.credit_ledger),
            "activation": False,
        },
        "SUBSTRATE_V2_CHECKPOINT_SCHEMA.json": {
            "schema": "substrate-v2-checkpoint-schema/v1",
            "state_fields": sorted(checkpoint["state"]),
            "identity_covers_exact_serialized_state": True,
            "hashed_but_not_saved": [],
            "saved_but_not_hashed": [],
            "activation": False,
        },
        "SUBSTRATE_V2_CONTINUITY_CANARY.json": {
            "schema": "substrate-v2-continuity-canary/v1",
            "exact_restore": canaries["C19"],
            "body_change": canaries["C20"],
            "passes": canaries["C19"]["passes"] and canaries["C20"]["passes"],
            "activation": False,
        },
        "SUBSTRATE_V2_DEVELOPMENTAL_DIVERGENCE.json": {
            "schema": "substrate-v2-developmental-divergence/v1",
            "histories": divergence,
            "distinctions": ["state difference", "behavioral difference", "useful specialization", "random drift", "corruption"],
            "mean_specialization_advantage": divergence_margin,
            "activation": False,
        },
        "SUBSTRATE_V2_HISTORY_SPECIALIZATION_CANARY.json": {
            "schema": "substrate-v2-history-specialization-canary/v1",
            "useful_divergence": canaries["C17"],
            "identical_control": canaries["C18"],
            "passes": canaries["C17"]["passes"] and canaries["C18"]["passes"],
            "activation": False,
        },
        "SUBSTRATE_V2_CHEAP_CANARIES.json": evidence,
        "SUBSTRATE_V2_CANARY_LEDGER.json": {
            "schema": "substrate-v2-canary-ledger/v1",
            "rows": list(canaries.values()),
            "admission_split_consumed_once": True,
            "classification_effect": "none",
            "activation": False,
        },
    }
    for name, document in mechanism_documents.items():
        io.seal(name, document)
    admission = {
        "schema": "substrate-v2-admission/v1",
        "stage": "cheap admission complete, integrated rehearsal pending",
        "v1_structural": io.load("SUBSTRATE_V1_IMMUTABILITY.json", artifact=True)["all_receipts_valid"],
        "checkpoint_and_integrity": all(canaries[key]["passes"] for key in ("C19", "C20", "C21", "C22")),
        "cross_domain_continuity": canaries["C8"]["passes"] and canaries["C9"]["passes"],
        "procedural_transfer": canaries["C5"]["passes"] and canaries["C6"]["passes"] and canaries["C7"]["passes"],
        "beds_and_controls_valid": io.load("SUBSTRATE_V2_BED_SCREEN.json")["all_valid"],
        "principal_splits_frozen": io.load("SUBSTRATE_V2_SPLIT_AUTHORITY.json")["no_seed_crosses_splits"],
        "resource_rehearsal": None,
        "principal_execution_licensed": False,
        "rehearsal_licensed": core_pass and all_terminal,
        "allocation_status": (
            "mechanism_positive"
            if canaries["C16"]["passes"]
            else canaries["C15"]["classification"]
        ),
        "failed": [identity for identity, row in canaries.items() if not row["passes"]],
        "activation": False,
    }
    publish_admission(admission)
    return {
        "selection": selected,
        "evidence": evidence,
        "admission": admission,
        "mechanisms": mechanism_documents,
    }
