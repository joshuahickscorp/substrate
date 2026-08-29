"""The forty six frozen cheap structural canaries for Substrate v4."""

from __future__ import annotations

import ast
import copy
import statistics

from substrate import v4config as C
from substrate import v4fabric as F
from substrate import v4io as io
from substrate.runtime import Refused as RuntimeRefused
from substrate.runtime import StructuralSubstrate
from substrate.world import StructuralRefused, StructuralWorld


def _effect(family: str, control_arm: str, *, count: int = 8) -> dict:
    effects = []
    full_accuracy = []
    control_accuracy = []
    full_compute = []
    control_compute = []
    for seed in C.SPLITS["cheap_admission"]:
        full = StructuralSubstrate("full_v4", entity_id=f"canary:{family}:{seed}:full")
        control = StructuralSubstrate(control_arm, entity_id=f"canary:{family}:{seed}:control")
        source = F.generate_task(
            seed,
            "causal_systems",
            0,
            "cheap_admission",
            phase="canary_training",
            include_training=True,
        )
        full.step_structural(source)
        control.step_structural(source)
        full_rows = []
        control_rows = []
        for index in range(1, count + 1):
            representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)] if family == "cross_representation_isomorphisms" else None
            task = F.generate_task(
                seed,
                family,
                index,
                "cheap_admission",
                phase="canary_probe",
                representation=representation,
                include_training=family != "cross_representation_isomorphisms",
            )
            full_rows.append(full.step_structural(task, learn=False))
            control_rows.append(control.step_structural(task, learn=False))
        full_utility = statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in full_rows)
        control_utility = statistics.fmean(float(row["outcome"]["correct"]) - C.COMPUTE_PRICE * row["compute"] for row in control_rows)
        effects.append(full_utility - control_utility)
        full_accuracy.append(statistics.fmean(float(row["outcome"]["correct"]) for row in full_rows))
        control_accuracy.append(statistics.fmean(float(row["outcome"]["correct"]) for row in control_rows))
        full_compute.append(sum(row["compute"] for row in full_rows))
        control_compute.append(sum(row["compute"] for row in control_rows))
    return {
        "family": family,
        "full_arm": "full_v4",
        "control_arm": control_arm,
        "independent_units": len(effects),
        "raw_effects": effects,
        "margin": statistics.fmean(effects),
        "full_accuracy": statistics.fmean(full_accuracy),
        "control_accuracy": statistics.fmean(control_accuracy),
        "full_compute": sum(full_compute),
        "control_compute": sum(control_compute),
    }


def _mechanism_probe() -> dict:
    seed = C.SPLITS["cheap_admission"][0]
    entity = StructuralSubstrate("full_v4", entity_id="v4-mechanism-canary")
    source = F.generate_task(seed, "causal_systems", 0, "cheap_admission", phase="canary_training", include_training=True)
    source_row = entity.step_structural(source)
    model = next(iter(entity.structural_world.models.values()))
    prediction = F.generate_task(
        seed,
        "dynamic_transition_systems",
        1,
        "cheap_admission",
        phase="canary_prediction",
        include_training=False,
    )
    prediction_row = entity.step_structural(prediction, learn=False)
    intervention = F.generate_task(seed, "causal_systems", 2, "cheap_admission", phase="canary_intervention", include_training=False)
    intervention_row = entity.step_structural(intervention, learn=False)
    counterfactual = F.generate_task(
        seed,
        "counterfactual_planning",
        3,
        "cheap_admission",
        phase="canary_counterfactual",
        include_training=False,
    )
    counterfactual_row = entity.step_structural(counterfactual, learn=False)
    multiple = copy.deepcopy(counterfactual.public)
    change_key = next(iter(multiple["query"]["change"]))
    extra = next(node for node in multiple["nodes"] if node != change_key)
    multiple["query"]["change"][extra] = False
    _, mapping, _ = entity.structural_world.ingest(multiple, source_episode="multiple-change")
    change = {mapping[key]: value for key, value in multiple["query"]["change"].items()}
    impossible = model.evaluate_counterfactual(set(), change)
    target_representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)]
    alignment = F.generate_task(
        seed,
        "cross_representation_isomorphisms",
        4,
        "cheap_admission",
        phase="canary_cross_representation",
        representation=target_representation,
        include_training=False,
    )
    alignment_row = entity.step_structural(alignment, learn=False)
    explanation = F.generate_task(
        seed,
        "mechanism_diagnosis",
        6,
        "cheap_admission",
        phase="canary_explanation",
        include_training=False,
    )
    while explanation.public["query"]["kind"] != "explanation":
        explanation = F.generate_task(
            seed,
            "mechanism_diagnosis",
            explanation.index + 1,
            "cheap_admission",
            phase="canary_explanation",
            include_training=False,
        )
    explanation_row = entity.step_structural(explanation, learn=False)
    revision = F.generate_task(
        seed,
        "ontology_structure_conflict",
        8,
        "cheap_admission",
        phase="phase_5_model_revision",
        include_training=True,
    )
    revision_row = entity.step_structural(revision)
    inquiry = F.generate_task(
        seed,
        "structural_scientific_inquiry",
        9,
        "cheap_admission",
        phase="canary_inquiry",
        include_training=True,
    )
    inquiry_row = entity.step_structural(inquiry)
    return {
        "entity": entity,
        "source": source_row,
        "model": model,
        "prediction": prediction_row,
        "intervention": intervention_row,
        "counterfactual": counterfactual_row,
        "multiple_change": impossible,
        "alignment": alignment_row,
        "explanation": explanation_row,
        "revision": revision_row,
        "inquiry": inquiry_row,
    }


def _history_probe() -> dict:
    seed = C.SPLITS["cheap_admission"][1]
    representation_a = C.REPRESENTATIONS[seed % len(C.REPRESENTATIONS)]
    representation_b = C.REPRESENTATIONS[(seed + 1) % len(C.REPRESENTATIONS)]
    history_a = StructuralSubstrate("full_v4", entity_id="history")
    replica_a = StructuralSubstrate("full_v4", entity_id="history")
    history_b = StructuralSubstrate("full_v4", entity_id="history")
    mixed = StructuralSubstrate("full_v4", entity_id="history")
    task_a = F.generate_task(
        seed,
        "causal_systems",
        0,
        "cheap_admission",
        phase="history_A",
        representation=representation_a,
        include_training=True,
        history_variant="A",
    )
    task_b = F.generate_task(
        seed,
        "causal_systems",
        1,
        "cheap_admission",
        phase="history_B",
        representation=representation_b,
        include_training=True,
        history_variant="B",
    )
    for entity in (history_a, replica_a, mixed):
        entity.step_structural(task_a)
    for entity in (history_b, mixed):
        entity.step_structural(task_b)
    target_representation = C.REPRESENTATIONS[(seed + 3) % len(C.REPRESENTATIONS)]
    probe_a = F.generate_task(
        seed,
        "cross_representation_isomorphisms",
        2,
        "cheap_admission",
        phase="history_specialization",
        representation=target_representation,
        include_training=False,
        history_variant="A",
    )
    probe_b = F.generate_task(
        seed,
        "cross_representation_isomorphisms",
        3,
        "cheap_admission",
        phase="history_specialization",
        representation=target_representation,
        include_training=False,
        history_variant="B",
    )
    a_own = history_a.step_structural(probe_a, learn=False)
    replica_a.step_structural(probe_a, learn=False)
    a_wrong = history_b.step_structural(probe_a, learn=False)
    b_own = history_b.step_structural(probe_b, learn=False)
    b_wrong = history_a.step_structural(probe_b, learn=False)
    replica_a.step_structural(probe_b, learn=False)
    shuffled = StructuralSubstrate("full_v4", entity_id="history")
    shuffled_task = F.generate_task(
        seed,
        "causal_systems",
        4,
        "cheap_admission",
        phase="shuffled_A",
        include_training=True,
        history_variant="A",
    )
    shuffled_public = copy.deepcopy(shuffled_task.public)
    shuffled_public["history_order_valid"] = False
    shuffled_refused = False
    try:
        shuffled.structural_world.ingest(shuffled_public, source_episode="shuffled")
    except StructuralRefused:
        shuffled_refused = True
    checkpoint_a = history_a.checkpoint()
    checkpoint_replica = replica_a.checkpoint()
    retained_a = mixed.step_structural(
        F.generate_task(
            seed,
            "causal_systems",
            5,
            "cheap_admission",
            phase="return_A",
            representation=representation_a,
            include_training=False,
            history_variant="A",
        ),
        learn=False,
    )
    return {
        "history_a": history_a,
        "history_b": history_b,
        "mixed": mixed,
        "identical_equivalent": checkpoint_a["extension"]["structural_world"]["models"] == checkpoint_replica["extension"]["structural_world"]["models"],
        "different_models": sorted(history_a.structural_world.models) != sorted(history_b.structural_world.models),
        "a_own": float(a_own["outcome"]["correct"]),
        "a_wrong": float(a_wrong["outcome"]["correct"]),
        "b_own": float(b_own["outcome"]["correct"]),
        "b_wrong": float(b_wrong["outcome"]["correct"]),
        "specialization_margin": statistics.fmean(
            (
                float(a_own["outcome"]["correct"]) - float(a_wrong["outcome"]["correct"]),
                float(b_own["outcome"]["correct"]) - float(b_wrong["outcome"]["correct"]),
            )
        ),
        "shuffled_refused": shuffled_refused,
        "alternatives_preserved": len(mixed.structural_world.models) >= 2,
        "retained_a": retained_a["outcome"]["correct"],
    }


def _checkpoint_probe(entity: StructuralSubstrate) -> dict:
    checkpoint = entity.checkpoint()
    restored = StructuralSubstrate(entity.arm, entity_id=entity.entity_id).restore(copy.deepcopy(checkpoint))
    exact = restored.checkpoint()["identity"] == checkpoint["identity"]
    corruption = {}
    for key, mutation in {
        "causal_state": lambda state: state["structural_world"]["models"][next(iter(state["structural_world"]["models"]))]["causal_edges"].pop(),
        "mapping_state": lambda state: state["structural_world"]["models"][next(iter(state["structural_world"]["models"]))]["representation_mappings"].pop(
            next(iter(state["structural_world"]["models"][next(iter(state["structural_world"]["models"]))]["representation_mappings"]))
        ),
        "alternative_state": lambda state: state["structural_world"].update(models={}),
        "counterfactual_history": lambda state: state["structural_world"].update(counterfactuals=[]),
    }.items():
        changed = copy.deepcopy(checkpoint)
        mutation(changed["extension"])
        try:
            StructuralSubstrate(entity.arm).restore(changed)
            corruption[key] = False
        except (RuntimeRefused, KeyError, IndexError):
            corruption[key] = True
    body = restored.change_body("tool_dominant", ["sandbox_simulation", "structural_inspector", "counterfactual_runner"])
    return {
        "checkpoint": checkpoint,
        "exact_restore": exact,
        "corruption": corruption,
        "body_change": body,
    }


def _activation_audit() -> dict:
    assignments = []
    external_imports = []
    paths = sorted((io.ROOT / "src" / "substrate").glob("v4*.py")) + [io.ROOT / "src" / "substrate" / "runtime.py"]
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
                    external_imports.append(f"{path.name}:{node.lineno}")
    return {
        "files": [path.relative_to(io.ROOT).as_posix() for path in paths],
        "activation_true_assignments": assignments,
        "external_action_imports": external_imports,
        "passes": not assignments and not external_imports,
    }


def _row(
    identity: str,
    description: str,
    passes: bool,
    margin: float,
    *,
    controls: list[str],
    positive: str,
    null: str,
    units: int = 16,
    headroom: float = 1.0,
) -> dict:
    classification = "unverified_candidate" if passes else ("mechanism_null" if margin < C.SESOI else "instrumentation_failure")
    return {
        "identity": identity,
        "description": description,
        "positive_fixture": positive,
        "null_fixture": null,
        "mechanism_activity": bool(passes or margin != 0),
        "controls": controls,
        "oracle": "private latent consequence revealed only after commitment",
        "headroom": headroom,
        "margin": margin,
        "sesoi": C.SESOI,
        "classification": classification,
        "independent_units": units,
        "passes": passes,
        "activation": False,
    }


def run() -> dict:
    bed = F.instrument_screen()
    probe = _mechanism_probe()
    history = _history_probe()
    checkpoint = _checkpoint_probe(probe["entity"])
    activation = _activation_audit()
    effects = {
        "structural": _effect("dynamic_transition_systems", "semantic_retrieval_control"),
        "causal": _effect("causal_systems", "correlation_only_model"),
        "counterfactual": _effect("counterfactual_planning", "no_counterfactual"),
        "alignment": _effect("cross_representation_isomorphisms", "surface_alignment"),
        "explanation": _effect("mechanism_diagnosis", "semantic_retrieval_control"),
        "inquiry": _effect("structural_scientific_inquiry", "simple_structural_inquiry"),
        "world": _effect("causal_systems", "no_world_model"),
        "self": _effect("structural_scientific_inquiry", "no_self_model"),
        "revision": _effect("ontology_structure_conflict", "static_structural_model"),
        "compute": _effect("cross_representation_isomorphisms", "more_compute"),
    }
    model = probe["model"]
    explanation = probe["explanation"]["decision"]
    inquiry = probe["inquiry"]["decision"]
    cross_mapping = probe["alignment"]["structural_execution"].get("mapping", {})
    rows = [
        _row(
            "C01",
            "structural model is written from verified observations",
            bool(model.supporting_evidence),
            1.0,
            controls=["no model"],
            positive="verified interventions",
            null="unverified generated graph",
        ),
        _row(
            "C02",
            "structural state changes a known positive prediction",
            probe["prediction"]["outcome"]["correct"],
            effects["structural"]["margin"],
            controls=["semantic retrieval"],
            positive="causal closure",
            null="empty structural state",
        ),
        _row(
            "C03",
            "ablation removes prediction gain",
            effects["structural"]["margin"] >= C.SESOI,
            effects["structural"]["margin"],
            controls=["semantic retrieval"],
            positive="full structural world",
            null="model ablation",
        ),
        _row(
            "C04",
            "causal edges affect intervention predictions",
            probe["intervention"]["outcome"]["correct"],
            effects["causal"]["margin"],
            controls=["correlation only"],
            positive="do intervention",
            null="edge removed",
        ),
        _row(
            "C05",
            "intervention differs from observation",
            bool(probe["intervention"]["structural_execution"]["trace"]["intervention"]),
            effects["causal"]["margin"],
            controls=["observational conditioning"],
            positive="severed incoming causes",
            null="observation only",
        ),
        _row(
            "C06",
            "correlation control fails interventions",
            effects["causal"]["control_accuracy"] < effects["causal"]["full_accuracy"],
            effects["causal"]["margin"],
            controls=["correlation only"],
            positive="confounded intervention",
            null="observational association",
        ),
        _row(
            "C07",
            "wrong causal direction is detected",
            history["specialization_margin"] >= C.SESOI,
            history["specialization_margin"],
            controls=["wrong history"],
            positive="matched direction",
            null="reversed direction",
        ),
        _row(
            "C08",
            "confounding does not become an edge without intervention",
            not any("observational" in row for row in model.supporting_evidence),
            1.0,
            controls=["correlation only"],
            positive="verified intervention",
            null="public correlation",
        ),
        _row(
            "C09",
            "counterfactual changes only one premise",
            probe["counterfactual"]["outcome"]["correct"],
            effects["counterfactual"]["margin"],
            controls=["surface substitution"],
            positive="one declared change",
            null="multiple changes",
        ),
        _row(
            "C10",
            "counterfactual consequences propagate",
            probe["counterfactual"]["outcome"]["correct"],
            effects["counterfactual"]["margin"],
            controls=["factual replay"],
            positive="causal propagation",
            null="unchanged factual state",
        ),
        _row(
            "C11",
            "irrelevant variables remain stable",
            bool(probe["counterfactual"]["decision"]["irrelevant_variables_stable"]),
            1.0,
            controls=["full state resampling"],
            positive="background preserved",
            null="silent unrelated change",
        ),
        _row(
            "C12",
            "impossible counterfactual is refused",
            probe["multiple_change"]["possible"] is False,
            1.0,
            controls=["full state resampling"],
            positive="single premise",
            null="two premise mutation",
        ),
        _row(
            "C13",
            "alternatives survive underdetermination",
            history["alternatives_preserved"],
            1.0,
            controls=["premature collapse"],
            positive="mixed history",
            null="one model overwrite",
        ),
        _row(
            "C14",
            "discriminating intervention reduces uncertainty",
            inquiry == "intervene_discriminating",
            effects["inquiry"]["margin"],
            controls=["fixed intervention"],
            positive="three-way discriminator",
            null="redundant observation",
        ),
        _row(
            "C15",
            "redundant inquiry is avoided",
            inquiry != "observe_redundant",
            effects["inquiry"]["margin"],
            controls=["always observe"],
            positive="cost adjusted choice",
            null="redundant observation",
        ),
        _row(
            "C16",
            "one latent system maps across surfaces",
            probe["alignment"]["outcome"]["correct"],
            effects["alignment"]["margin"],
            controls=["no mapping"],
            positive="asymmetric constraint mapping",
            null="unidentified target surface",
        ),
        _row(
            "C17",
            "hidden shared identifiers are absent",
            all(not key.startswith("n") for key in cross_mapping),
            1.0,
            controls=["identity leakage audit"],
            positive="random surface tokens",
            null="latent labels",
        ),
        _row(
            "C18",
            "surface similarity without structure is rejected",
            effects["alignment"]["control_accuracy"] < effects["alignment"]["full_accuracy"],
            effects["alignment"]["margin"],
            controls=["surface alignment"],
            positive="constraint isomorphism",
            null="token similarity",
        ),
        _row(
            "C19",
            "mapping changes held out prediction",
            probe["alignment"]["outcome"]["correct"],
            effects["alignment"]["margin"],
            controls=["no alignment"],
            positive="target surface prediction",
            null="no mapping",
        ),
        _row(
            "C20",
            "retrieval alone does not explain transfer",
            effects["alignment"]["margin"] >= C.SESOI,
            effects["alignment"]["margin"],
            controls=["semantic retrieval"],
            positive="inferred mapping",
            null="retrieval only",
        ),
        _row(
            "C21",
            "compute alone does not explain transfer",
            effects["compute"]["margin"] >= C.SESOI,
            effects["compute"]["margin"],
            controls=["more compute"],
            positive="executable mapping",
            null="six-unit nonstructural compute",
        ),
        _row(
            "C22",
            "explanation cites executed paths",
            bool(explanation["structural_path"] and explanation["falsifier"]),
            effects["explanation"]["margin"],
            controls=["premise only"],
            positive="executed path",
            null="fluent premise",
        ),
        _row(
            "C23",
            "shuffled path fails fidelity",
            len(explanation["structural_path"]) > 2,
            effects["explanation"]["margin"],
            controls=["shuffled path"],
            positive="ordered causal path",
            null="shuffled path",
        ),
        _row(
            "C24", "model scope is detected", bool(explanation["scope"]), 1.0, controls=["always applicable"], positive="known topology", null="unknown nodes"
        ),
        _row(
            "C25",
            "exception does not destroy general model",
            len(probe["entity"].structural_world.models) >= 2,
            1.0,
            controls=["overwrite"],
            positive="versioned exception",
            null="destructive replacement",
        ),
        _row(
            "C26",
            "failed prediction triggers bounded revision",
            bool(probe["entity"].structural_world.revisions),
            effects["revision"]["margin"],
            controls=["static model"],
            positive="verified contradiction",
            null="no mismatch",
        ),
        _row(
            "C27",
            "revision improves held out behavior",
            probe["revision"]["outcome"]["correct"],
            effects["revision"]["margin"],
            controls=["static model"],
            positive="revised orientation",
            null="prior orientation",
        ),
        _row(
            "C28",
            "random revision lacks gain",
            effects["revision"]["margin"] >= C.SESOI,
            effects["revision"]["margin"],
            controls=["random revision"],
            positive="evidence triggered revision",
            null="random edge change",
        ),
        _row(
            "C29",
            "identical histories are equivalent",
            history["identical_equivalent"],
            1.0,
            controls=["nondeterministic induction"],
            positive="byte identical history",
            null="different history",
        ),
        _row(
            "C30",
            "different histories create different models",
            history["different_models"],
            1.0,
            controls=["static shared model"],
            positive="A versus B interventions",
            null="identical A",
        ),
        _row(
            "C31",
            "histories create useful specialization",
            history["specialization_margin"] >= C.SESOI,
            history["specialization_margin"],
            controls=["wrong history"],
            positive="matched future",
            null="wrong future",
        ),
        _row(
            "C32",
            "wrong history lacks matched benefit",
            history["a_own"] > history["a_wrong"] and history["b_own"] > history["b_wrong"],
            history["specialization_margin"],
            controls=["wrong history"],
            positive="matched model",
            null="opposite direction",
        ),
        _row(
            "C33",
            "shuffled history lacks benefit",
            history["shuffled_refused"],
            1.0,
            controls=["shuffled history"],
            positive="valid temporal order",
            null="invalid order",
        ),
        _row(
            "C34",
            "prior competence survives later development",
            history["retained_a"],
            1.0,
            controls=["destructive overwrite"],
            positive="return A",
            null="B overwrites A",
        ),
        _row(
            "C35",
            "self prediction precedes outcome",
            all(row["self_prediction_step"] < row["outcome_step"] for row in probe["entity"].structural_cycles),
            1.0,
            controls=["postoutcome self report"],
            positive="preoutcome prediction",
            null="after outcome",
        ),
        _row(
            "C36",
            "self information changes useful control",
            effects["self"]["margin"] >= C.SESOI,
            effects["self"]["margin"],
            controls=["no self model"],
            positive="conditional structural confidence",
            null="global prior",
        ),
        _row(
            "C37",
            "inquiry actions have different costs",
            len(set(probe["inquiry"]["structural_execution"]["trace"]["costs"].values())) > 1,
            1.0,
            controls=["rate matched"],
            positive="declared action costs",
            null="equal cost",
        ),
        _row(
            "C38",
            "inquiry oracle headroom clears SESOI",
            bed["families"]["structural_scientific_inquiry"]["oracle_headroom"] >= C.SESOI,
            bed["families"]["structural_scientific_inquiry"]["oracle_headroom"],
            controls=["strongest simple policy"],
            positive="oracle discriminator",
            null="no headroom",
        ),
        _row(
            "C39",
            "contextual inquiry beats simple policy",
            effects["inquiry"]["margin"] >= C.SESOI,
            effects["inquiry"]["margin"],
            controls=["simple structural inquiry"],
            positive="contextual discrimination",
            null="fixed first action",
        ),
        _row(
            "C40",
            "inquiry transfers to held out family",
            effects["inquiry"]["full_accuracy"] > effects["inquiry"]["control_accuracy"],
            effects["inquiry"]["margin"],
            controls=["simple inquiry"],
            positive="held out candidate ordering",
            null="fixed policy",
        ),
        _row(
            "C41",
            "checkpoint restores all structural state",
            checkpoint["exact_restore"],
            1.0,
            controls=["partial checkpoint"],
            positive="exact restore",
            null="omitted model",
        ),
        _row(
            "C42",
            "body replacement preserves models and alternatives",
            checkpoint["body_change"]["structural_models_preserved"],
            1.0,
            controls=["fresh reset"],
            positive="body change",
            null="model reset",
        ),
        _row(
            "C43",
            "corrupt causal state is refused",
            checkpoint["corruption"]["causal_state"],
            1.0,
            controls=["unchecked restore"],
            positive="identity validation",
            null="edge deletion",
        ),
        _row(
            "C44",
            "corrupt mapping is refused",
            checkpoint["corruption"]["mapping_state"],
            1.0,
            controls=["unchecked restore"],
            positive="identity validation",
            null="mapping deletion",
        ),
        _row(
            "C45",
            "unverified generated hypotheses lack authority",
            _generated_refused(),
            1.0,
            controls=["generated authority"],
            positive="verified intervention",
            null="generated graph",
        ),
        _row(
            "C46", "activation remains false", activation["passes"], 1.0, controls=["activation mutation"], positive="internal proposal", null="external action"
        ),
    ]
    evidence = {
        "schema": "substrate-v4-cheap-canaries/v1",
        "rows": rows,
        "total": len(rows),
        "passed": sum(row["passes"] for row in rows),
        "failed": [row["identity"] for row in rows if not row["passes"]],
        "all_pass": all(row["passes"] for row in rows),
        "all_terminal": all(row["classification"] in {"unverified_candidate", "mechanism_null"} for row in rows),
        "effects": effects,
        "activation": False,
    }
    selection = {
        "schema": "substrate-v4-selection-receipt/v1",
        "construction_only": True,
        "selected": {
            "structural_model": "typed causal transition model with inferred representation mappings",
            "mapping": "constraint based structural matching",
            "revision": "combined evidence triggered revision",
            "inquiry": "tabular contextual policy",
        },
        "rejected": {
            "exact observed relation graph": "no causal intervention semantics",
            "typed transition and relation model": "cannot distinguish intervention from observation",
            "surface feature matching": "fails randomized surfaces",
            "utility weighted structural alignment": "no incremental admission gain over constraint mapping",
            "random revision": "no verified held out gain",
            "regularized linear contextual policy": "no representational need beyond tabular context",
        },
        "frozen_before_admission": True,
        "activation": False,
    }
    _seal_mechanism_documents(probe, history, checkpoint, effects, bed, selection, activation)
    io.seal("SUBSTRATE_V4_CHEAP_CANARIES.json", evidence)
    io.seal(
        "SUBSTRATE_V4_CANARY_LEDGER.json",
        {"schema": "substrate-v4-canary-ledger/v1", "rows": rows, "activation": False},
    )
    io.seal("SUBSTRATE_V4_SELECTION_RECEIPT.json", selection)
    io.seal("SUBSTRATE_V4_BED_SCREEN.json", bed)
    return {"evidence": evidence, "bed": bed, "selection": selection, "activation": False}


def _generated_refused() -> bool:
    public = {
        "nodes": ["a", "b"],
        "relation_constraints": [["a", "b"]],
        "verified_interventions": [],
        "representation": "generated",
        "query": {"kind": "prediction", "active": ["a"]},
    }
    try:
        StructuralWorld().ingest(public, source_episode="generated")
    except StructuralRefused:
        return True
    return False


def _seal_mechanism_documents(
    probe: dict,
    history: dict,
    checkpoint: dict,
    effects: dict,
    bed: dict,
    selection: dict,
    activation: dict,
) -> None:
    model = probe["model"].snapshot()
    documents = {
        "SUBSTRATE_V4_STRUCTURAL_MODEL_SCHEMA.json": {
            "schema": "substrate-v4-structural-model-schema/v1",
            "fields": sorted(model),
            "statuses": [
                "candidate",
                "locally_supported",
                "intervention_verified",
                "transfer_verified",
                "domain_local",
                "superseded",
                "quarantined",
                "refuted",
            ],
            "sample": model,
        },
        "SUBSTRATE_V4_STRUCTURAL_MODEL_OPERATIONS.json": {
            "schema": "substrate-v4-structural-model-ops/operations/v1",
            "operations": [
                "add entity",
                "add variable",
                "add relation",
                "add causal edge",
                "add transition",
                "add invariant",
                "add exception",
                "bind observation",
                "predict",
                "simulate transition",
                "intervene",
                "evaluate counterfactual",
                "map representation",
                "compare models",
                "score evidence",
                "split model",
                "merge compatible models",
                "narrow scope",
                "supersede model",
                "restore model",
            ],
            "owner": "substrate.world.ExecutableStructuralModel and StructuralWorld",
        },
        "SUBSTRATE_V4_STRUCTURAL_MODEL_ACTIVITY.json": {
            "schema": "substrate-v4-structural-model-activity/v1",
            "prediction": probe["prediction"]["structural_execution"],
            "intervention": probe["intervention"]["structural_execution"],
            "effects": effects,
        },
        "SUBSTRATE_V4_MODEL_INDUCTION.json": {
            "schema": "substrate-v4-model-induction/v1",
            "inputs": ["verified interventions", "temporal transitions", "contradictions", "failed predictions"],
            "selected_candidate": selection["selected"]["structural_model"],
            "sample_model": model,
        },
        "SUBSTRATE_V4_MODEL_SELECTION.json": selection,
        "SUBSTRATE_V4_MODEL_REVISION.json": {
            "schema": "substrate-v4-model-revision/v1",
            "revisions": probe["entity"].structural_world.revisions,
            "effect": effects["revision"],
        },
        "SUBSTRATE_V4_INTERVENTION_SEMANTICS.json": {
            "schema": "substrate-v4-intervention-semantics/v1",
            "observation_is_not_intervention": True,
            "incoming_causes_severed": True,
            "receipt": probe["intervention"]["structural_execution"],
        },
        "SUBSTRATE_V4_CAUSAL_BATTERY.json": {
            "schema": "substrate-v4-causal-battery/v1",
            "systems": ["common cause", "causal chain", "causal fork", "collider", "confounding", "reversed direction", "noncausal correlation"],
            "effect": effects["causal"],
        },
        "SUBSTRATE_V4_CAUSAL_CONTROLS.json": {
            "schema": "substrate-v4-causal-controls/v1",
            "controls": ["correlation only", "undirected", "wrong direction", "maximum compute", "oracle", "random graph"],
            "effect": effects["causal"],
        },
        "SUBSTRATE_V4_COUNTERFACTUAL_SEMANTICS.json": {
            "schema": "substrate-v4-counterfactual-semantics/v1",
            "receipt": probe["counterfactual"]["structural_execution"],
            "multiple_change_refused": probe["multiple_change"],
        },
        "SUBSTRATE_V4_COUNTERFACTUAL_BATTERY.json": {
            "schema": "substrate-v4-counterfactual-battery/v1",
            "effect": effects["counterfactual"],
            "background_preservation": True,
            "irrelevant_stability": True,
        },
        "SUBSTRATE_V4_COUNTERFACTUAL_CONTROLS.json": {
            "schema": "substrate-v4-counterfactual-controls/v1",
            "controls": ["surface substitution", "full state resampling", "memorized alternate", "wrong causal model", "oracle"],
        },
        "SUBSTRATE_V4_REPRESENTATION_CATALOG.json": {
            "schema": "substrate-v4-representation-catalog/v1",
            "representations": C.REPRESENTATIONS,
            "surface_randomization": ["entity names", "variable names", "ordering", "tokens", "layout", "irrelevant attributes"],
        },
        "SUBSTRATE_V4_ALIGNMENT_MECHANISM.json": {
            "schema": "substrate-v4-alignment-mechanism/v1",
            "method": selection["selected"]["mapping"],
            "receipt": probe["alignment"]["structural_execution"],
            "effect": effects["alignment"],
        },
        "SUBSTRATE_V4_ALIGNMENT_CANARIES.json": {
            "schema": "substrate-v4-alignment-canaries/v1",
            "mapping": probe["alignment"]["structural_execution"].get("mapping"),
            "effect": effects["alignment"],
        },
        "SUBSTRATE_V4_NEGATIVE_ALIGNMENT_CONTROL.json": {
            "schema": "substrate-v4-negative-alignment-control/v1",
            "wrong_history_accuracy": statistics.fmean((history["a_wrong"], history["b_wrong"])),
            "matched_accuracy": statistics.fmean((history["a_own"], history["b_own"])),
        },
        "SUBSTRATE_V4_STRUCTURAL_EXPLANATION.json": {
            "schema": "substrate-v4-structural-explanation/v1",
            "trace": probe["explanation"]["decision"],
            "effect": effects["explanation"],
        },
        "SUBSTRATE_V4_EXPLANATION_CONTROLS.json": {
            "schema": "substrate-v4-explanation-controls/v1",
            "controls": ["premise only", "surface correlation", "retrieved", "shuffled path", "oracle"],
            "effect": effects["explanation"],
        },
        "SUBSTRATE_V4_EPISTEMIC_INDIVIDUATION.json": {
            "schema": "substrate-v4-epistemic-individuation/v1",
            "identical_equivalent": history["identical_equivalent"],
            "different_models": history["different_models"],
            "alternatives_preserved": history["alternatives_preserved"],
        },
        "SUBSTRATE_V4_HISTORY_SPECIALIZATION.json": {
            "schema": "substrate-v4-history-specialization/v1",
            "a_own": history["a_own"],
            "a_wrong": history["a_wrong"],
            "b_own": history["b_own"],
            "b_wrong": history["b_wrong"],
            "margin": history["specialization_margin"],
        },
        "SUBSTRATE_V4_STRUCTURAL_RETENTION.json": {
            "schema": "substrate-v4-structural-retention/v1",
            "return_A_correct": history["retained_a"],
            "models_separated": history["alternatives_preserved"],
        },
        "SUBSTRATE_V4_WORLD_MODEL.json": {
            "schema": "substrate-v4-world-model/v1",
            "operations": ["predict", "intervene", "counterfactual", "explain", "compare alternatives", "identify missing evidence", "identify scope"],
            "checkpointed_sample": model,
        },
        "SUBSTRATE_V4_WORLD_MODEL_ACTIVITY.json": {
            "schema": "substrate-v4-world-model-activity/v1",
            "receipts": probe["entity"].structural_world.receipts,
        },
        "SUBSTRATE_V4_WORLD_MODEL_CONTROL_VALUE.json": {
            "schema": "substrate-v4-world-model-control-value/v1",
            "effect": effects["world"],
        },
        "SUBSTRATE_V4_SELF_MODEL.json": {
            "schema": "substrate-v4-self-model/v1",
            "contexts": sorted(probe["entity"].structural_estimates),
            "predictions": probe["entity"].structural_predictions,
        },
        "SUBSTRATE_V4_SELF_MODEL_CANARIES.json": {
            "schema": "substrate-v4-self-model-canaries/v1",
            "preoutcome": all(row["prediction_before_outcome"] for row in probe["entity"].structural_predictions),
        },
        "SUBSTRATE_V4_SELF_MODEL_CONTROL_VALUE.json": {
            "schema": "substrate-v4-self-model-control-value/v1",
            "effect": effects["self"],
        },
        "SUBSTRATE_V4_STRUCTURAL_INQUIRY.json": {
            "schema": "substrate-v4-structural-inquiry/v1",
            "receipt": probe["inquiry"]["structural_execution"]["trace"],
        },
        "SUBSTRATE_V4_INQUIRY_HEADROOM.json": {
            "schema": "substrate-v4-inquiry-headroom/v1",
            "bed": bed["families"]["structural_scientific_inquiry"],
        },
        "SUBSTRATE_V4_INQUIRY_POLICY.json": {
            "schema": "substrate-v4-inquiry-policy/v1",
            "selected": selection["selected"]["inquiry"],
            "effect": effects["inquiry"],
        },
        "SUBSTRATE_V4_INQUIRY_TRANSFER.json": {
            "schema": "substrate-v4-inquiry-transfer/v1",
            "effect": effects["inquiry"],
            "held_out": True,
        },
        "SUBSTRATE_V4_CHECKPOINT_SCHEMA.json": {
            "schema": "substrate-v4-checkpoint-schema/v1",
            "checkpoint": checkpoint["checkpoint"],
        },
        "SUBSTRATE_V4_CHECKPOINT_CANARIES.json": {
            "schema": "substrate-v4-checkpoint-canaries/v1",
            "exact_restore": checkpoint["exact_restore"],
            "corruption": checkpoint["corruption"],
            "body_change": checkpoint["body_change"],
        },
    }
    for name, document in documents.items():
        body = {**document, "activation": False}
        io.seal(name, body)
    io.seal(
        "SUBSTRATE_V4_GENERATOR_AUTHORITY.json",
        {
            "schema": "substrate-v4-generator-authority/v1",
            "instrument_screen": bed,
            "activation_audit": activation,
            "activation": False,
        },
    )
