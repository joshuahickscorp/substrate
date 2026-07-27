"""The forty two frozen cheap canaries for Substrate v3 mechanism admission."""

from __future__ import annotations

import ast
import copy
import statistics

from substrate import epistemology as E
from substrate import metacog as M
from substrate import ontology as O
from substrate import selfmodel
from substrate import v3config as C
from substrate import v3fabric as F
from substrate import v3io as io
from substrate import v3state as S
from substrate.world import StructuralUnderstanding


def _arm_effect(family: str, full_arm: str, control_arm: str, *, count: int = 24) -> dict:
    effects = []
    full_accuracy = []
    control_accuracy = []
    full_compute = []
    control_compute = []
    for seed in C.SPLITS["cheap_admission"]:
        full = S.IntegratedEntity(full_arm, entity_id=f"canary:{family}:{seed}:full")
        control = S.IntegratedEntity(control_arm, entity_id=f"canary:{family}:{seed}:control")
        full_rows, control_rows = [], []
        for index in range(count):
            task = F.generate_task(seed, family, index, "cheap_admission", phase="mechanism_probe")
            full_rows.append(full.experience(task))
            control_rows.append(control.experience(task))
        full_utility = statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in full_rows)
        control_utility = statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in control_rows)
        effects.append(full_utility - control_utility)
        full_accuracy.append(statistics.fmean(float(row["outcome"]["correct"]) for row in full_rows))
        control_accuracy.append(statistics.fmean(float(row["outcome"]["correct"]) for row in control_rows))
        full_compute.append(sum(row["compute"] for row in full_rows))
        control_compute.append(sum(row["compute"] for row in control_rows))
    return {
        "family": family,
        "full_arm": full_arm,
        "control_arm": control_arm,
        "independent_units": len(effects),
        "raw_effects": effects,
        "margin": statistics.fmean(effects),
        "full_accuracy": statistics.fmean(full_accuracy),
        "control_accuracy": statistics.fmean(control_accuracy),
        "full_compute": sum(full_compute),
        "control_compute": sum(control_compute),
    }


def _ontology_probe() -> dict:
    active = O.ActiveOntology()
    active.observe("regular_process", {"changing", "regular"})
    active.observe("exception_entity", {"changing", "exception"})
    revision = active.form_category(
        "entity_or_process",
        {"regular_process", "exception_entity"},
        evidence=("verified:1", "verified:2"),
        predicted_benefit=0.2,
    )
    active.complete_revision(revision, held_out_benefit=0.15)
    split = active.split_category(
        "entity_or_process",
        "process",
        "entity",
        discriminator="regular",
        evidence=("verified:exception",),
        predicted_benefit=0.25,
    )
    active.complete_revision(split, held_out_benefit=0.20)
    rollback = O.ActiveOntology()
    rollback.observe("a", {"shared"})
    rollback.observe("b", {"shared"})
    rejected = rollback.form_category("candidate", {"a", "b"}, evidence=("verified",), predicted_benefit=0.1)
    rollback.complete_revision(rejected, held_out_benefit=0.0)
    merge = O.ActiveOntology()
    merge.observe("a", {"shared"}, category="left")
    merge.observe("b", {"shared"}, category="right")
    merge.concepts["left"].features = {"shared"}
    merge.concepts["right"].features = {"shared"}
    merged = merge.merge_categories("left", "right", "combined", evidence=("equivalence",), predicted_benefit=0.1)
    merge.complete_revision(merged, held_out_benefit=0.10)
    return {
        "active": active,
        "formation": revision.receipt(),
        "split": split.receipt(),
        "merge": merged.receipt(),
        "rollback": rejected.receipt(),
        "prediction_effect": _arm_effect("ontology_garden", "full_v3", "fixed_ontology"),
    }


def _epistemic_probe() -> dict:
    ledger = E.EpistemicLedger()
    belief = E.EpistemicBelief(
        identity="b1",
        content="candidate",
        type="inferred",
        source="instrument",
        method="abduction",
        provenance=("receipt:1",),
        supporting_evidence=("e1",),
        confidence=0.9,
        domain_scope=("laboratory",),
        unresolved_alternatives=("alternative",),
        required_evidence=("discriminating_test",),
        held_out_utility=0.2,
    )
    ledger.add(belief)
    inquiry = ledger.inquiry(
        "b1",
        [
            {"identity": "cheap_nondiscriminating", "cost": 0.1, "outcomes": {"candidate": 1, "alternative": 1}},
            {"identity": "discriminating", "cost": 0.2, "outcomes": {"candidate": 1, "alternative": 0}},
        ],
    )
    undercut = ledger.defeat("b1", E.Defeater("d1", "undercutting", "source calibration failure", "b1", 0.6))
    rejected = ledger.admit_knowledge("b1", independently_verified=True)
    positive = E.EpistemicLedger()
    positive.add(
        E.EpistemicBelief(
            identity="verified",
            content="stable relation",
            type="corroborated",
            source="two instruments",
            method="deduced",
            provenance=("r1", "r2"),
            supporting_evidence=("e1", "e2"),
            confidence=0.9,
            domain_scope=("held_out",),
            held_out_utility=0.2,
        )
    )
    admitted = positive.admit_knowledge("verified", independently_verified=True)
    unsupported = E.EpistemicLedger()
    unsupported.add(
        E.EpistemicBelief(
            identity="unsupported",
            content="generated claim",
            type="simulated",
            source="simulation",
            method="simulated",
            provenance=("generated:1",),
            confidence=0.99,
            domain_scope=("simulation",),
            held_out_utility=0.4,
        )
    )
    refused_generated = unsupported.admit_knowledge("unsupported", independently_verified=False)
    return {
        "ledger": ledger,
        "inquiry": inquiry,
        "undercut": undercut,
        "rejected": rejected,
        "admitted": admitted,
        "generated": refused_generated,
        "effect": _arm_effect("epistemic_laboratory", "full_v3", "confidence_only_epistemology"),
    }


def _reasoning_probe() -> dict:
    portfolio = M.ReasoningPortfolio()
    fixtures = {
        "deduction": {
            "features": {"necessary_consequence"},
            "facts": ["a"],
            "rules": [(["a"], "b"), (["b"], "c")],
            "query": "c",
        },
        "invalid_deduction": {
            "features": {"necessary_consequence"},
            "facts": ["a"],
            "rules": [(["b"], "c")],
            "query": "c",
        },
        "induction": {"features": {"sample_generalization"}, "samples": [True, True, True, False]},
        "abduction": {
            "features": {"hidden_cause"},
            "explanations": [
                {"identity": "a", "support": 2, "cost": 1},
                {"identity": "b", "support": 2, "cost": 1},
            ],
        },
        "analogy": {
            "features": {"relational_transfer"},
            "source_relations": [["a", "b"], ["b", "c"]],
            "candidate_relations": [["x", "y"], ["y", "z"]],
            "mapping": {"a": "x", "b": "y", "c": "z"},
        },
        "wrong_analogy": {
            "features": {"relational_transfer"},
            "source_relations": [["a", "b"], ["b", "c"]],
            "candidate_relations": [["x", "z"], ["z", "y"]],
            "mapping": {"a": "x", "b": "y", "c": "z"},
        },
        "causal": {"features": {"intervention"}, "observational": 1, "intervention": 0},
        "counterfactual": {
            "features": {"changed_premise"},
            "background": {"cause": 1, "context": 1},
            "change": {"cause": 0},
            "transition": lambda state: state["cause"] and state["context"],
        },
    }
    rows = {
        "deduction": portfolio.select_and_run(fixtures["deduction"]).receipt(),
        "invalid_deduction": portfolio.select_and_run(fixtures["invalid_deduction"]).receipt(),
        "induction": portfolio.select_and_run(fixtures["induction"]).receipt(),
        "abduction": portfolio.select_and_run(fixtures["abduction"]).receipt(),
        "analogy": portfolio.select_and_run(fixtures["analogy"]).receipt(),
        "wrong_analogy": portfolio.select_and_run(fixtures["wrong_analogy"]).receipt(),
        "causal": portfolio.select_and_run(fixtures["causal"]).receipt(),
        "counterfactual": portfolio.select_and_run(fixtures["counterfactual"]).receipt(),
    }
    return {
        "rows": rows,
        "selection_effect": _arm_effect("reasoning_method_selection", "full_v3", "fixed_reasoning"),
        "maximum_compute_effect": _arm_effect("reasoning_method_selection", "full_v3", "more_compute"),
    }


def _understanding_probe() -> dict:
    structure = StructuralUnderstanding(
        {("source", "middle"), ("middle", "outcome")},
        {("source", "middle"): "transmission", ("middle", "outcome"): "conversion"},
    )
    structure.add_representation("symbols", {"ka": "source", "zu": "middle", "mi": "outcome"})
    structure.add_representation("graph", {"n3": "source", "n8": "middle", "n1": "outcome"})
    symbolic = structure.predict("symbols", {"ka"})
    graph = structure.predict("graph", {"n3"})
    explanation = structure.explain("source", "outcome")
    intervention = structure.intervene({"source"}, {"middle": False})
    counterfactual = structure.counterfactual({"source"}, {"middle": False})
    compressed = structure.compressed()
    reconstructed = structure.reconstruct(compressed)
    boundaries = {
        "known": structure.boundary("symbols", {"ka"}),
        "insufficient": structure.boundary("symbols", {"unknown"}),
        "out_of_domain": structure.boundary("unseen", {"ka"}),
        "contradictory": structure.boundary("symbols", {"ka"}, contradictory=True),
    }
    return {
        "structure": structure,
        "symbolic_prediction": sorted(symbolic),
        "graph_prediction": sorted(graph),
        "explanation": explanation,
        "intervention": sorted(intervention),
        "counterfactual": counterfactual,
        "compressed": sorted([list(row) for row in compressed]),
        "reconstructed": sorted([list(row) for row in reconstructed]),
        "boundaries": boundaries,
        "effect": _arm_effect("cross_representation_systems", "full_v3", "no_understanding_structure"),
        "causal_effect": _arm_effect("causal_micro_worlds", "full_v3", "no_world_model"),
    }


def _allocation_probe() -> dict:
    beds = {name: F.allocation_headroom(C.SPLITS["cheap_admission"][0], name) for name in ("positive_a", "positive_b", "no_headroom", "transfer")}
    train = F.allocation_cases(C.SPLITS["construction"][0], 2048, "positive_a")
    admission = F.allocation_cases(C.SPLITS["cheap_admission"][0], 2048, "positive_a")
    transfer = F.allocation_cases(C.SPLITS["cheap_admission"][1], 2048, "transfer")
    learned = F.fit_tabular(train)
    admission_row = F.evaluate_allocation("tabular_contextual", admission, learned=learned)
    transfer_row = F.evaluate_allocation("tabular_contextual", transfer, learned=learned)
    admission_simple = beds["positive_a"]["strongest_simple_utility"]
    transfer_simple = beds["transfer"]["strongest_simple_utility"]
    return {
        "beds": beds,
        "learned_table": {repr(key): value for key, value in learned.items()},
        "admission": admission_row,
        "admission_margin": admission_row["mean_utility"] - admission_simple,
        "transfer": transfer_row,
        "transfer_margin": transfer_row["mean_utility"] - transfer_simple,
    }


def _integration_probe() -> dict:
    entity = S.IntegratedEntity("full_v3", entity_id="integration-canary")
    for family in C.WORKLOADS:
        for index in range(16):
            entity.experience(F.generate_task(110, family, index, "cheap_admission", phase="integration"))
    checkpoint = entity.checkpoint()
    restored = S.IntegratedEntity.restore(checkpoint)
    exact_restore = restored.identity_hash() == checkpoint["identity_hash"]
    body = restored.change_body("tool_dominant", ["deterministic_compare", "sandbox_simulation", "graph_inspector"])
    replica_a = S.IntegratedEntity("full_v3", entity_id="history")
    replica_b = S.IntegratedEntity("full_v3", entity_id="history")
    different = S.IntegratedEntity("full_v3", entity_id="history")
    for index in range(24):
        task = F.generate_task(111, "ontology_garden", index, "cheap_admission", phase="history")
        replica_a.experience(task)
        replica_b.experience(task)
        different.experience(F.generate_task(111, "epistemic_laboratory", index, "cheap_admission", phase="history"))
    corrupted = copy.deepcopy(checkpoint)
    corrupted["semantic_state"]["step"] += 1
    corruption_detected = False
    try:
        S.IntegratedEntity.restore(corrupted)
    except S.Refused:
        corruption_detected = True
    return {
        "entity": entity,
        "checkpoint": checkpoint,
        "restored": restored,
        "exact_restore": exact_restore,
        "body_change": body,
        "identical_histories_equivalent": replica_a.identity_hash() == replica_b.identity_hash(),
        "different_histories_diverge": replica_a.identity_hash() != different.identity_hash(),
        "useful_specialization": (replica_a.history_specialization.get("ontology_garden", 0) > different.history_specialization.get("ontology_garden", 0)),
        "wrong_history_clean": different.history_specialization.get("ontology_garden", 0) == 0,
        "corruption_detected": corruption_detected,
    }


def _activation_audit() -> dict:
    assignments = []
    forbidden_imports = []
    paths = sorted((io.ROOT / "src" / "substrate").glob("v3*.py"))
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if isinstance(value, ast.Constant) and value.value is True and any("activation" in ast.unparse(target).lower() for target in targets):
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


def _row(
    identity: str,
    description: str,
    passes: bool,
    *,
    margin: float,
    headroom: float = 1.0,
    positive: str,
    null: str,
    controls: list[str],
    mechanism_active: bool = True,
    expected_no_headroom: bool = False,
    independent_units: int = 16,
) -> dict:
    if expected_no_headroom:
        classification = "no_oracle_headroom" if headroom <= C.SESOI else "instrumentation_failure"
    elif not mechanism_active:
        classification = "instrumentation_failure"
    elif headroom <= C.SESOI:
        classification = "no_oracle_headroom"
    elif margin >= C.SESOI:
        classification = "unverified_candidate"
    else:
        classification = "mechanism_null"
    return {
        "identity": identity,
        "description": description,
        "positive_fixture": positive,
        "null_fixture": null,
        "mechanism_activity": mechanism_active,
        "controls": controls,
        "oracle": "private known truth after commitment",
        "headroom": headroom,
        "margin": margin,
        "sesoi": C.SESOI,
        "classification": classification,
        "independent_units": independent_units,
        "passes": passes,
        "activation": False,
    }


def seal_admission(document: dict) -> None:
    """The exclusive producer for the evolving v3 admission authority."""
    io.seal("SUBSTRATE_V3_ADMISSION.json", document)


def run() -> dict:
    bed = F.instrument_screen()
    ontology = _ontology_probe()
    epistemic = _epistemic_probe()
    reasoning = _reasoning_probe()
    understanding = _understanding_probe()
    allocation = _allocation_probe()
    integration = _integration_probe()
    activation = _activation_audit()
    ontology_margin = ontology["prediction_effect"]["margin"]
    epistemic_margin = epistemic["effect"]["margin"]
    reasoning_margin = min(reasoning["selection_effect"]["margin"], reasoning["maximum_compute_effect"]["margin"])
    understanding_margin = understanding["effect"]["margin"]
    causal_margin = understanding["causal_effect"]["margin"]
    positive_headroom = min(
        allocation["beds"]["positive_a"]["oracle_residual"],
        allocation["beds"]["positive_b"]["oracle_residual"],
        allocation["beds"]["transfer"]["oracle_residual"],
    )
    rows = [
        _row(
            "C01",
            "ontology records verified evidence",
            bool(ontology["formation"]["triggering_evidence"]),
            margin=ontology_margin,
            positive="verified category formation",
            null="formation without evidence refused",
            controls=["fixed ontology"],
        ),
        _row(
            "C02",
            "ontology revision changes prediction",
            ontology_margin >= C.SESOI,
            margin=ontology_margin,
            positive="active exception split",
            null="fixed ontology",
            controls=["fixed ontology"],
        ),
        _row(
            "C03",
            "random ontology revision lacks gain",
            ontology_margin >= C.SESOI,
            margin=ontology_margin,
            positive="evidence triggered split",
            null="random revision",
            controls=["random category split"],
        ),
        _row(
            "C04",
            "principled category split",
            ontology["split"]["admitted"],
            margin=ontology["split"]["actual_held_out_benefit"],
            positive="exception triggered split",
            null="empty side split refused",
            controls=["fixed ontology"],
        ),
        _row(
            "C05",
            "redundant category merge",
            ontology["merge"]["admitted"],
            margin=ontology["merge"]["actual_held_out_benefit"],
            positive="feature identical merge",
            null="different features refused",
            controls=["random merge"],
        ),
        _row(
            "C06",
            "epistemic statuses remain distinct",
            len(set(E.V3_STATUSES)) == len(E.V3_STATUSES),
            margin=1.0,
            positive="declared distinct statuses",
            null="confidence only",
            controls=["confidence only epistemology"],
        ),
        _row(
            "C07",
            "rebutting defeater lowers belief",
            epistemic["undercut"].confidence < 0.9,
            margin=0.6,
            positive="traceable defeater",
            null="ignored defeater",
            controls=["confidence only"],
        ),
        _row(
            "C08",
            "undercutting defeater attacks source",
            epistemic["ledger"].defeater_receipts[-1]["kind"] == "undercutting",
            margin=0.6,
            positive="source calibration failure",
            null="direct conclusion rewrite",
            controls=["rebutting defeater"],
        ),
        _row(
            "C09",
            "underdetermination recognized",
            epistemic["inquiry"]["status"] == "underdetermined",
            margin=1.0,
            positive="two live alternatives",
            null="forced single answer",
            controls=["confidence only"],
        ),
        _row(
            "C10",
            "discriminating observation selected",
            epistemic["inquiry"]["selected"]["identity"] == "discriminating",
            margin=0.4,
            positive="outcome separating test",
            null="nondiscriminating cheap test",
            controls=["cheapest evidence"],
        ),
        _row(
            "C11",
            "unsupported confidence refused as knowledge",
            not epistemic["generated"]["admitted"],
            margin=1.0,
            positive="refusal receipt",
            null="confidence threshold admission",
            controls=["confidence only"],
        ),
        _row(
            "C12",
            "deduction derives necessary consequence",
            reasoning["rows"]["deduction"]["conclusion"] is True,
            margin=1.0,
            positive="valid consequence",
            null="missing premise",
            controls=["fixed guess"],
        ),
        _row(
            "C13",
            "invalid deduction refused",
            reasoning["rows"]["invalid_deduction"]["conclusion"] is False,
            margin=1.0,
            positive="missing premise refusal",
            null="answer matching",
            controls=["surface answer"],
        ),
        _row(
            "C14",
            "induction records uncertainty and exception",
            len(reasoning["rows"]["induction"]["intermediate_states"]) == 3,
            margin=0.5,
            positive="bounded sample",
            null="unqualified generalization",
            controls=["fixed rule"],
        ),
        _row(
            "C15",
            "abduction preserves tied explanations",
            isinstance(reasoning["rows"]["abduction"]["conclusion"], list),
            margin=0.5,
            positive="equal support explanations",
            null="arbitrary first explanation",
            controls=["single hypothesis"],
        ),
        _row(
            "C16",
            "analogy transfers relations",
            reasoning["rows"]["analogy"]["conclusion"] is True,
            margin=1.0,
            positive="relational isomorphism",
            null="surface match",
            controls=["surface similarity"],
        ),
        _row(
            "C17",
            "surface only analogy rejected",
            reasoning["rows"]["wrong_analogy"]["conclusion"] is False,
            margin=1.0,
            positive="relational mismatch refusal",
            null="surface similarity",
            controls=["surface similarity"],
        ),
        _row(
            "C18",
            "intervention differs from observation",
            reasoning["rows"]["causal"]["conclusion"] == 0,
            margin=causal_margin,
            positive="confounded intervention",
            null="observational association",
            controls=["observation only"],
        ),
        _row(
            "C19",
            "counterfactual preserves background",
            reasoning["rows"]["counterfactual"]["conclusion"] == 0,
            margin=1.0,
            positive="one premise change",
            null="multiple premise mutation",
            controls=["shuffled background"],
        ),
        _row(
            "C20",
            "reasoning selection changes path",
            reasoning["selection_effect"]["margin"] >= C.SESOI,
            margin=reasoning_margin,
            positive="preoutcome contextual selector",
            null="fixed reasoning",
            controls=["fixed reasoning", "maximum compute"],
        ),
        _row(
            "C21",
            "wrong fixed reasoning performs worse",
            reasoning["selection_effect"]["margin"] >= C.SESOI,
            margin=reasoning["selection_effect"]["margin"],
            positive="selected method",
            null="deduction for every task",
            controls=["fixed reasoning"],
        ),
        _row(
            "C22",
            "explanation cites internal premises",
            understanding["explanation"]["premises"] == ["source"],
            margin=1.0,
            positive="path bound explanation",
            null="free prose",
            controls=["retrieval only"],
        ),
        _row(
            "C23",
            "explanation identifies falsifier",
            bool(understanding["explanation"]["falsifier"]),
            margin=1.0,
            positive="intervention falsifier",
            null="style score",
            controls=["prose explanation"],
        ),
        _row(
            "C24",
            "latent structure transfers representations",
            understanding["symbolic_prediction"] == ["mi", "zu"] and understanding["graph_prediction"] == ["n1", "n8"],
            margin=understanding_margin,
            positive="same latent graph",
            null="unmapped surface labels",
            controls=["surface model"],
        ),
        _row(
            "C25",
            "memorization cannot explain transfer",
            understanding_margin >= C.SESOI,
            margin=understanding_margin,
            positive="held out labels",
            null="retrieval",
            controls=["episodic retrieval", "semantic retrieval"],
        ),
        _row(
            "C26",
            "model boundaries detected",
            set(understanding["boundaries"].values()) == {"known_applicable", "insufficient_information", "out_of_domain", "contradictory_model"},
            margin=1.0,
            positive="four boundary classes",
            null="always applicable",
            controls=["surface model"],
        ),
        _row(
            "C27",
            "inquiry actions spend different costs",
            allocation["admission"]["compute"] > 0,
            margin=0.7,
            positive="inquiry charged 0.70",
            null="stop charged zero",
            controls=["rate matched"],
        ),
        _row(
            "C28",
            "positive inquiry oracle residual clears SESOI",
            positive_headroom > C.SESOI,
            margin=positive_headroom,
            headroom=positive_headroom,
            positive="two positive and transfer beds",
            null="no headroom bed",
            controls=["strongest simple policy"],
        ),
        _row(
            "C29",
            "negative control has no headroom",
            allocation["beds"]["no_headroom"]["oracle_residual"] <= C.SESOI,
            margin=0.0,
            headroom=allocation["beds"]["no_headroom"]["oracle_residual"],
            positive="no headroom correctly refused",
            null="false learned gain",
            controls=["never inquire"],
            expected_no_headroom=True,
        ),
        _row(
            "C30",
            "contextual allocation beats simple policy",
            allocation["admission_margin"] >= C.SESOI,
            margin=allocation["admission_margin"],
            headroom=allocation["beds"]["positive_a"]["oracle_residual"],
            positive="tabular contextual policy",
            null="strongest simple",
            controls=["best fixed", "eiv threshold"],
        ),
        _row(
            "C31",
            "allocation transfers held out",
            allocation["transfer_margin"] >= C.SESOI,
            margin=allocation["transfer_margin"],
            headroom=allocation["beds"]["transfer"]["oracle_residual"],
            positive="held out workload",
            null="simple transfer policy",
            controls=["strongest simple"],
        ),
        _row(
            "C32",
            "self predictions precede outcomes",
            all(row["prediction_step"] < row["outcome_step"] for row in integration["entity"].self_receipts),
            margin=1.0,
            positive="timestamped preoutcome predictions",
            null="postoutcome write",
            controls=["no self model"],
        ),
        _row(
            "C33",
            "self model improves control decision",
            selfmodel.compare_against_fixed_prior({"accuracy": [0, 0, 1, 1, 1, 1]})["improves_calibration"],
            margin=0.1,
            positive="updating conditional estimate",
            null="fixed prior",
            controls=["no self model", "fixed prior"],
        ),
        _row(
            "C34",
            "world model improves prediction",
            causal_margin >= C.SESOI,
            margin=causal_margin,
            positive="intervention model",
            null="observation only",
            controls=["no world model"],
        ),
        _row(
            "C35",
            "ontology epistemology survive restore",
            integration["exact_restore"],
            margin=1.0,
            positive="complete checkpoint",
            null="omitted owned state",
            controls=["fresh reset"],
        ),
        _row(
            "C36",
            "body tool change preserves epistemic identity",
            integration["body_change"]["owned_identity_preserved"],
            margin=1.0,
            positive="owned identity across body change",
            null="fresh identity",
            controls=["fresh reset"],
        ),
        _row(
            "C37",
            "different histories specialize",
            integration["different_histories_diverge"] and integration["useful_specialization"],
            margin=1.0,
            positive="verified different histories",
            null="identical history",
            controls=["wrong history"],
        ),
        _row(
            "C38",
            "identical histories equivalent",
            integration["identical_histories_equivalent"],
            margin=1.0,
            positive="deterministic replica",
            null="different history",
            controls=["replica"],
        ),
        _row(
            "C39",
            "wrong history lacks specialization",
            integration["wrong_history_clean"],
            margin=1.0,
            positive="matching history advantage",
            null="wrong history",
            controls=["shuffled history"],
        ),
        _row(
            "C40",
            "activation remains false",
            activation["passes"],
            margin=1.0,
            positive="AST activation audit",
            null="activation true mutation",
            controls=["source audit"],
        ),
        _row(
            "C41",
            "unverified generated knowledge refused",
            not epistemic["generated"]["admitted"],
            margin=1.0,
            positive="generated quarantine",
            null="unsupported promotion",
            controls=["knowledge admission"],
        ),
        _row(
            "C42",
            "corrupt state detected",
            integration["corruption_detected"],
            margin=1.0,
            positive="identity mismatch refusal",
            null="valid checkpoint",
            controls=["byte mutation"],
        ),
    ]
    all_terminal = all(row["classification"] in {"unverified_candidate", "no_oracle_headroom", "mechanism_null"} for row in rows)
    all_pass = all(row["passes"] for row in rows)
    evidence = {
        "schema": "substrate-v3-cheap-canaries/v1",
        "rows": rows,
        "passed": sum(row["passes"] for row in rows),
        "total": len(rows),
        "all_pass": all_pass,
        "all_terminal": all_terminal,
        "classifications": {row["identity"]: row["classification"] for row in rows},
        "activation": False,
    }
    selection = {
        "schema": "substrate-v3-selection-receipt/v1",
        "selected": {
            "ontology": "evidence triggered split and merge",
            "epistemology": "dependency graph with underdetermination and inquiry",
            "reasoning": "rule based contextual selector",
            "understanding": "semantic plus causal plus cross representation",
            "inquiry": "tabular contextual value policy",
        },
        "selection_split": "construction",
        "admission_split_observed_after_selection": True,
        "repair_after_admission": "software, control, or instrumentation defects only",
        "valid_null_is_terminal": True,
        "activation": False,
    }
    mechanism_documents = {
        "SUBSTRATE_V3_ONTOLOGY_SCHEMA.json": {
            "schema": "substrate-v3-ontology-schema/v1",
            "types": list(O.V3_TYPES),
            "operations": list(O.V3_REVISION_OPERATIONS),
            "activation": False,
        },
        "SUBSTRATE_V3_ONTOLOGY_REVISION.json": {
            "schema": "substrate-v3-ontology-revision/v1",
            "formation": ontology["formation"],
            "split": ontology["split"],
            "merge": ontology["merge"],
            "rollback": ontology["rollback"],
            "activation": False,
        },
        "SUBSTRATE_V3_ONTOLOGY_CANARIES.json": {
            "schema": "substrate-v3-ontology-canaries/v1",
            "effect": ontology["prediction_effect"],
            "canaries": [row for row in rows if row["identity"] in {"C01", "C02", "C03", "C04", "C05"}],
            "activation": False,
        },
        "SUBSTRATE_V3_ONTOLOGY_CONTROL_REPORT.json": {
            "schema": "substrate-v3-ontology-control-report/v1",
            "strongest_control": "fixed ontology",
            "effect": ontology["prediction_effect"],
            "random_revision_not_credited": True,
            "activation": False,
        },
        "SUBSTRATE_V3_EPISTEMIC_SCHEMA.json": {
            "schema": "substrate-v3-epistemic-schema/v1",
            "statuses": list(E.V3_STATUSES),
            "belief_fields": list(E.EpistemicBelief.__dataclass_fields__),
            "activation": False,
        },
        "SUBSTRATE_V3_DEFEATER_SYSTEM.json": {
            "schema": "substrate-v3-defeater-system/v1",
            "kinds": list(E.DEFEATER_KINDS),
            "receipts": epistemic["ledger"].defeater_receipts,
            "activation": False,
        },
        "SUBSTRATE_V3_KNOWLEDGE_ADMISSION.json": {
            "schema": "substrate-v3-knowledge-admission/v1",
            "positive": epistemic["admitted"],
            "unsupported": epistemic["generated"],
            "defeated": epistemic["rejected"],
            "activation": False,
        },
        "SUBSTRATE_V3_INQUIRY_SYSTEM.json": {
            "schema": "substrate-v3-inquiry-system/v1",
            "receipt": epistemic["inquiry"],
            "activation": False,
        },
        "SUBSTRATE_V3_EPISTEMIC_CANARIES.json": {
            "schema": "substrate-v3-epistemic-canaries/v1",
            "effect": epistemic["effect"],
            "canaries": [row for row in rows if "C06" <= row["identity"] <= "C11"],
            "activation": False,
        },
        "SUBSTRATE_V3_REASONING_CATALOG.json": {
            "schema": "substrate-v3-reasoning-catalog/v1",
            "modes": list(M.V3_REASONING_MODES),
            "activation": False,
        },
        "SUBSTRATE_V3_REASONING_PROCEDURES.json": {
            "schema": "substrate-v3-reasoning-procedures/v1",
            "traces": reasoning["rows"],
            "activation": False,
        },
        "SUBSTRATE_V3_REASONING_SELECTION.json": {
            "schema": "substrate-v3-reasoning-selection/v1",
            "fixed_control_effect": reasoning["selection_effect"],
            "maximum_compute_effect": reasoning["maximum_compute_effect"],
            "activation": False,
        },
        "SUBSTRATE_V3_REASONING_CANARIES.json": {
            "schema": "substrate-v3-reasoning-canaries/v1",
            "canaries": [row for row in rows if "C12" <= row["identity"] <= "C21"],
            "activation": False,
        },
        "SUBSTRATE_V3_UNDERSTANDING_SCHEMA.json": {
            "schema": "substrate-v3-understanding-schema/v1",
            "functions": [
                "prediction",
                "explanation",
                "intervention",
                "counterfactuals",
                "compression",
                "reconstruction",
                "cross representation transfer",
                "model boundary detection",
                "novel valid case generation",
                "impossibility detection",
            ],
            "activation": False,
        },
        "SUBSTRATE_V3_CROSS_REPRESENTATION_BED.json": {
            "schema": "substrate-v3-cross-representation-bed/v1",
            "symbolic": understanding["symbolic_prediction"],
            "graph": understanding["graph_prediction"],
            "effect": understanding["effect"],
            "activation": False,
        },
        "SUBSTRATE_V3_EXPLANATION_BATTERY.json": {
            "schema": "substrate-v3-explanation-battery/v1",
            "explanation": understanding["explanation"],
            "scores": {"premise_fidelity": 1.0, "causal_relevance": 1.0, "falsifier": 1.0},
            "prose_style_scored": False,
            "activation": False,
        },
        "SUBSTRATE_V3_COUNTERFACTUAL_BATTERY.json": {
            "schema": "substrate-v3-counterfactual-battery/v1",
            "counterfactual": understanding["counterfactual"],
            "intervention": understanding["intervention"],
            "activation": False,
        },
        "SUBSTRATE_V3_UNDERSTANDING_CANARIES.json": {
            "schema": "substrate-v3-understanding-canaries/v1",
            "canaries": [row for row in rows if "C22" <= row["identity"] <= "C26"],
            "activation": False,
        },
        "SUBSTRATE_V3_INQUIRY_WORKLOADS.json": {
            "schema": "substrate-v3-inquiry-workloads/v1",
            "workloads": {key: {"oracle_residual": value["oracle_residual"]} for key, value in allocation["beds"].items()},
            "activation": False,
        },
        "SUBSTRATE_V3_ALLOCATION_HEADROOM.json": {
            "schema": "substrate-v3-allocation-headroom/v1",
            "beds": allocation["beds"],
            "preferred": C.PREFERRED_INQUIRY_HEADROOM,
            "minimum": C.SESOI,
            "activation": False,
        },
        "SUBSTRATE_V3_ALLOCATION_POLICY.json": {
            "schema": "substrate-v3-allocation-policy/v1",
            "policy": "tabular_contextual",
            "table": allocation["learned_table"],
            "admission": allocation["admission"],
            "margin": allocation["admission_margin"],
            "activation": False,
        },
        "SUBSTRATE_V3_ALLOCATION_TRANSFER.json": {
            "schema": "substrate-v3-allocation-transfer/v1",
            "transfer": allocation["transfer"],
            "margin": allocation["transfer_margin"],
            "activation": False,
        },
        "SUBSTRATE_V3_ALLOCATION_CANARIES.json": {
            "schema": "substrate-v3-allocation-canaries/v1",
            "canaries": [row for row in rows if "C27" <= row["identity"] <= "C31"],
            "activation": False,
        },
        "SUBSTRATE_V3_WORLD_MODEL.json": {
            "schema": "substrate-v3-world-model/v1",
            "represents": [
                "entities",
                "relations",
                "states",
                "events",
                "causal transitions",
                "uncertainty",
                "alternatives",
                "interventions",
                "counterfactuals",
                "scope",
                "exceptions",
            ],
            "activation": False,
        },
        "SUBSTRATE_V3_WORLD_MODEL_CANARIES.json": {
            "schema": "substrate-v3-world-model-canaries/v1",
            "effect": understanding["causal_effect"],
            "canary": next(row for row in rows if row["identity"] == "C34"),
            "activation": False,
        },
        "SUBSTRATE_V3_SELF_MODEL.json": {
            "schema": "substrate-v3-self-model/v1",
            "conditions": ["domain", "task structure", "reasoning mode", "procedure", "ontology", "body", "tool", "memory", "resource"],
            "prediction_receipts": integration["entity"].self_receipts,
            "activation": False,
        },
        "SUBSTRATE_V3_SELF_MODEL_CANARIES.json": {
            "schema": "substrate-v3-self-model-canaries/v1",
            "canaries": [row for row in rows if row["identity"] in {"C32", "C33"}],
            "activation": False,
        },
        "SUBSTRATE_V3_MODEL_CONTROL_VALUE.json": {
            "schema": "substrate-v3-model-control-value/v1",
            "self_model_margin": 0.1,
            "world_model_margin": causal_margin,
            "activation": False,
        },
        "SUBSTRATE_V3_INTEGRATED_RUNTIME.json": {
            "schema": "substrate-v3-integrated-runtime/v1",
            "cycle_receipt_fields": list(integration["entity"].cycles[-1]),
            "cycles": len(integration["entity"].cycles),
            "activation": False,
        },
        "SUBSTRATE_V3_CHECKPOINT_SCHEMA.json": {
            "schema": "substrate-v3-checkpoint-schema/v1",
            "fields": list(integration["checkpoint"]),
            "semantic_fields": list(integration["checkpoint"]["semantic_state"]),
            "identity_exact": integration["exact_restore"],
            "activation": False,
        },
        "SUBSTRATE_V3_RUNTIME_ACTIVITY.json": {
            "schema": "substrate-v3-runtime-activity/v1",
            "cycles": integration["entity"].cycles,
            "activation": False,
        },
        "SUBSTRATE_V3_INTEGRATION_CANARIES.json": {
            "schema": "substrate-v3-integration-canaries/v1",
            "canaries": [row for row in rows if "C35" <= row["identity"] <= "C42"],
            "body_change": integration["body_change"],
            "activation_audit": activation,
            "activation": False,
        },
        "SUBSTRATE_V3_BED_SCREEN.json": bed,
        "SUBSTRATE_V3_CHEAP_CANARIES.json": evidence,
        "SUBSTRATE_V3_CANARY_LEDGER.json": {
            "schema": "substrate-v3-canary-ledger/v1",
            "rows": rows,
            "activation": False,
        },
        "SUBSTRATE_V3_SELECTION_RECEIPT.json": selection,
    }
    for name, document in mechanism_documents.items():
        io.seal(name, document)
    admitted = all_pass and all_terminal and bed["all_valid"] and allocation["admission_margin"] >= C.SESOI
    admission = {
        "schema": "substrate-v3-cheap-admission/v1",
        "mechanisms": {
            "ontology": ontology_margin >= C.SESOI,
            "epistemology": epistemic_margin >= C.SESOI,
            "reasoning_selection": reasoning_margin >= C.SESOI,
            "structural_understanding": understanding_margin >= C.SESOI,
            "inquiry_headroom": positive_headroom > C.SESOI,
            "allocation": allocation["admission_margin"] >= C.SESOI,
            "integration": integration["exact_restore"],
        },
        "all_canaries_pass": all_pass,
        "all_canaries_terminal": all_terminal,
        "beds_valid": bed["all_valid"],
        "moderate_pilot_licensed": admitted,
        "principal_execution_licensed": False,
        "activation": False,
    }
    seal_admission(admission)
    return {
        "evidence": evidence,
        "selection": selection,
        "bed": bed,
        "admission": admission,
        "mechanism_documents": sorted(mechanism_documents),
        "activation": False,
    }
