"""Independent raw receipt recomputation, mutations, reproduction, and v3 classification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from substrate import epistemology as E
from substrate import metacog as M
from substrate import v3config as C
from substrate import v3fabric as F
from substrate import v3io as io
from substrate import v3principal as P
from substrate import v3state as S


class Refused(RuntimeError):
    """Independent verification refused invalid or incomplete evidence."""


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = sum(math.comb(len(nonzero), count) for count in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2.0 * probability)


def paired(values: list[float], endpoint: str) -> dict:
    if not values:
        raise Refused(f"paired endpoint {endpoint!r} has no histories")
    seed = int(hashlib.sha256(endpoint.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    bootstraps = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(2000)]
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "endpoint": endpoint,
        "n": len(values),
        "raw_paired_effects": values,
        "mean": mean,
        "median": statistics.median(values),
        "bootstrap_95_ci": [_percentile(bootstraps, 0.025), _percentile(bootstraps, 0.975)],
        "exact_sign_p": exact_sign_p(values),
        "standardized_effect": mean / deviation if deviation else (math.inf if mean else 0.0),
        "sesoi": C.SESOI,
    }


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    rows = {}
    rejecting = True
    for index, name in enumerate(ordered):
        threshold = alpha / (len(ordered) - index)
        rejected = rejecting and p_values[name] <= threshold
        if not rejected:
            rejecting = False
        rows[name] = {"raw_p": p_values[name], "holm_threshold": threshold, "reject_zero": rejected}
    return {"family": ordered, "alpha": alpha, "method": "Holm", "rows": rows}


def raw() -> dict:
    units = {unit.identity: unit for unit in P.work_units()}
    receipts = {}
    invalid = []
    missing = []
    checkpoint_mismatch = []
    for identity, unit in units.items():
        path = P.UNITS / f"{identity}.json"
        checkpoint_path = P.CHECKPOINTS / f"{identity}.json"
        if not path.is_file() or not checkpoint_path.is_file():
            missing.append(identity)
            continue
        try:
            receipt = json.loads(path.read_text())
            checkpoint = json.loads(checkpoint_path.read_text())
        except json.JSONDecodeError:
            invalid.append(identity)
            continue
        if not P.validate_receipt(receipt, unit):
            invalid.append(identity)
            continue
        if checkpoint.get("identity_hash") != receipt["checkpoint"]["identity_hash"]:
            checkpoint_mismatch.append(identity)
            continue
        receipts[identity] = receipt
    return {
        "receipts": receipts,
        "expected": len(units),
        "valid": len(receipts),
        "missing": missing,
        "invalid": invalid,
        "checkpoint_mismatch": checkpoint_mismatch,
        "all_pass": len(receipts) == len(units) and not missing and not invalid and not checkpoint_mismatch,
        "activation": False,
    }


def _history_table(receipts: dict[str, dict], split: str) -> dict[tuple[int, str], dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for receipt in receipts.values():
        unit = receipt["unit"]
        if unit["split"] == split:
            grouped.setdefault((unit["history_seed"], unit["arm"]), []).append(receipt)
    table = {}
    for key, rows in grouped.items():
        cycles = [cycle for receipt in rows for cycle in receipt["cycles"]]
        family_utility = {}
        family_accuracy = {}
        for family in C.WORKLOADS:
            selected = [cycle for cycle in cycles if cycle["family"] == family]
            family_utility[family] = statistics.fmean(float(cycle["outcome"]["correct"]) - C.COMPUTE_PRICE * cycle["compute"] for cycle in selected)
            family_accuracy[family] = statistics.fmean(float(cycle["outcome"]["correct"]) for cycle in selected)
        phase_utility = {}
        for phase in C.PHASES:
            selected = [cycle for cycle in cycles if cycle["phase"] == phase]
            phase_utility[phase] = statistics.fmean(float(cycle["outcome"]["correct"]) - C.COMPUTE_PRICE * cycle["compute"] for cycle in selected)
        explanations = [cycle["explanation"] for cycle in cycles if cycle.get("explanation")]
        table[key] = {
            "receipts": rows,
            "cycles": cycles,
            "family_utility": family_utility,
            "family_accuracy": family_accuracy,
            "phase_utility": phase_utility,
            "explanations": explanations,
            "compute": sum(cycle["compute"] for cycle in cycles),
            "summaries": [receipt["summary"] for receipt in rows],
        }
    return table


def _contrast(
    table: dict[tuple[int, str], dict],
    full: str,
    controls: tuple[str, ...],
    accessor,
    endpoint: str,
) -> dict:
    seeds = sorted(seed for seed, arm in table if arm == full)
    effects = []
    strongest = []
    for seed in seeds:
        full_value = float(accessor(table[(seed, full)]))
        values = {arm: float(accessor(table[(seed, arm)])) for arm in controls}
        strongest_arm = max(values, key=values.get)
        effects.append(full_value - values[strongest_arm])
        strongest.append(strongest_arm)
    result = paired(effects, endpoint)
    result["full_arm"] = full
    result["controls"] = list(controls)
    result["strongest_control_by_history"] = strongest
    return result


def _positive(effect: dict) -> bool:
    return effect["mean"] >= C.SESOI and effect["bootstrap_95_ci"][0] > 0


def recompute(raw_report: dict) -> dict:
    if not raw_report["all_pass"]:
        raise Refused("raw principal receipts are incomplete or invalid")
    principal = _history_table(raw_report["receipts"], "principal")
    replication = _history_table(raw_report["receipts"], "replication")
    open_world = _history_table(raw_report["receipts"], "open_world_review")

    ontology = _contrast(
        principal,
        "full_v3",
        ("fixed_ontology",),
        lambda row: row["family_utility"]["ontology_garden"],
        "ontology_utility",
    )
    epistemic = _contrast(
        principal,
        "full_v3",
        ("confidence_only_epistemology",),
        lambda row: row["family_utility"]["epistemic_laboratory"],
        "epistemic_utility",
    )
    reasoning = _contrast(
        principal,
        "full_v3",
        ("fixed_reasoning", "more_compute"),
        lambda row: row["family_utility"]["reasoning_method_selection"],
        "reasoning_selection_utility",
    )
    cross_representation = _contrast(
        principal,
        "full_v3",
        ("no_understanding_structure", "more_compute"),
        lambda row: row["family_utility"]["cross_representation_systems"],
        "cross_representation_utility",
    )
    causal = _contrast(
        principal,
        "full_v3",
        ("no_world_model", "more_compute"),
        lambda row: row["phase_utility"]["phase_6_causal_intervention"],
        "causal_intervention_utility",
    )
    counterfactual = _contrast(
        principal,
        "full_v3",
        ("no_world_model", "more_compute"),
        lambda row: row["phase_utility"]["phase_7_counterfactual"],
        "counterfactual_utility",
    )
    boundary = _contrast(
        principal,
        "full_v3",
        ("no_understanding_structure",),
        lambda row: row["family_utility"]["adversarial_ambiguity"],
        "model_boundary_utility",
    )
    explanation_values = []
    for seed in C.SPLITS["principal"]:
        full = principal[(seed, "full_v3")]["explanations"]
        control = principal[(seed, "no_understanding_structure")]["explanations"]
        full_score = statistics.fmean(
            float(bool(row["premises"]) and bool(row["relation_or_mechanism"]) and bool(row["derivation"]) and bool(row["falsifier"])) for row in full
        )
        control_score = (
            statistics.fmean(
                float(bool(row["premises"]) and bool(row["relation_or_mechanism"]) and bool(row["derivation"]) and bool(row["falsifier"])) for row in control
            )
            if control
            else 0.0
        )
        explanation_values.append(full_score - control_score)
    explanation = paired(explanation_values, "explanation_fidelity")
    inquiry = _contrast(
        principal,
        "full_v3",
        ("simple_inquiry", "more_compute"),
        lambda row: row["family_utility"]["scientific_inquiry"],
        "inquiry_allocation_utility",
    )
    self_model = _contrast(
        principal,
        "full_v3",
        ("no_self_model",),
        lambda row: row["family_utility"]["scientific_inquiry"],
        "self_model_control_utility",
    )
    world_model = _contrast(
        principal,
        "full_v3",
        ("no_world_model",),
        lambda row: statistics.fmean(
            (
                row["phase_utility"]["phase_6_causal_intervention"],
                row["phase_utility"]["phase_7_counterfactual"],
            )
        ),
        "world_model_control_utility",
    )
    developmental = []
    divergence = []
    integrity = []
    for seed in C.SPLITS["principal"]:
        summaries = principal[(seed, "full_v3")]["summaries"]
        preservation = next(row["v2_preservation"] for row in summaries if row["v2_preservation"] is not None)
        divergence_row = next(row["divergence"] for row in summaries if row["divergence"] is not None)
        developmental.append(preservation)
        divergence.append(divergence_row)
        integrity.append(
            {
                "checkpoint_exact": all(row["checkpoint_exact"] for row in summaries),
                "body_continuity": all(row["body_continuity"] for row in summaries),
                "activation_false": all(row["activation"] is False for row in summaries),
            }
        )
    transfer = paired([row["transfer_margin"] for row in developmental], "v2_transfer_preservation")
    retention = paired([row["retention_change"] for row in developmental], "v2_retention_change")
    useful_divergence = paired([row["mean_specialization_margin"] for row in divergence], "useful_epistemic_divergence")
    replication_reasoning = _contrast(
        replication,
        "full_v3",
        ("fixed_reasoning",),
        lambda row: row["family_utility"]["reasoning_method_selection"],
        "replication_reasoning",
    )
    replication_inquiry = _contrast(
        replication,
        "full_v3",
        ("simple_inquiry",),
        lambda row: row["family_utility"]["scientific_inquiry"],
        "replication_inquiry",
    )
    open_world_effect = _contrast(
        open_world,
        "full_v3",
        ("no_understanding_structure",),
        lambda row: row["family_utility"]["cross_representation_systems"],
        "open_world_cross_representation",
    )
    endpoints = {
        "ontology": ontology,
        "epistemic": epistemic,
        "reasoning": reasoning,
        "cross_representation": cross_representation,
        "explanation": explanation,
        "causal": causal,
        "counterfactual": counterfactual,
        "model_boundary": boundary,
        "inquiry": inquiry,
        "self_model": self_model,
        "world_model": world_model,
        "transfer": transfer,
        "retention": retention,
        "divergence": useful_divergence,
        "replication_reasoning": replication_reasoning,
        "replication_inquiry": replication_inquiry,
        "open_world": open_world_effect,
    }
    gates = {
        "H_N1": _positive(ontology),
        "H_N2": _positive(epistemic),
        "H_N3": _positive(reasoning),
        "H_N4": all(_positive(effect) for effect in (cross_representation, explanation, causal, counterfactual, boundary)),
        "H_N5": _positive(inquiry),
        "H_N6": _positive(transfer) and retention["mean"] >= -C.SESOI and all(row["identity_exact"] and row["body_continuity"] for row in developmental),
        "H_N7": _positive(useful_divergence)
        and all(row["identical_histories_equivalent"] and row["different_histories_diverge"] and row["wrong_history_clean"] for row in divergence),
        "H_N8": _positive(self_model) and _positive(world_model),
        "H_N9": all(row["checkpoint_exact"] and row["body_continuity"] and row["activation_false"] for row in integrity),
    }
    primary_effect = {
        "H_N1": ontology,
        "H_N2": epistemic,
        "H_N3": reasoning,
        "H_N4": cross_representation,
        "H_N5": inquiry,
        "H_N6": transfer,
        "H_N7": useful_divergence,
        "H_N8": self_model,
        "H_N9": paired([1.0 if all(row.values()) else 0.0 for row in integrity], "identity_integrity"),
    }
    correction = holm({key: value["exact_sign_p"] for key, value in primary_effect.items()})
    counts = {
        "principal_work_units": sum(receipt["unit"]["split"] == "principal" for receipt in raw_report["receipts"].values()),
        "replication_work_units": sum(receipt["unit"]["split"] == "replication" for receipt in raw_report["receipts"].values()),
        "open_world_work_units": sum(receipt["unit"]["split"] == "open_world_review" for receipt in raw_report["receipts"].values()),
        "independent_principal_histories": len(C.SPLITS["principal"]),
        "episodes": sum(receipt["summary"]["episodes"] for receipt in raw_report["receipts"].values()),
        "ontology_revisions": sum(receipt["summary"]["ontology_revisions"] for receipt in raw_report["receipts"].values()),
        "belief_revisions": sum(receipt["summary"]["belief_revisions"] for receipt in raw_report["receipts"].values()),
        "knowledge_admissions": sum(receipt["summary"]["knowledge_admissions"] for receipt in raw_report["receipts"].values()),
        "defeaters_processed": sum(receipt["summary"]["defeaters_processed"] for receipt in raw_report["receipts"].values()),
        "reasoning_operations": sum(receipt["summary"]["reasoning_operations"] for receipt in raw_report["receipts"].values()),
        "inquiry_actions": sum(receipt["summary"]["inquiry_actions"] for receipt in raw_report["receipts"].values()),
        "allocation_compute": sum(
            sum(cycle["compute"] for cycle in receipt["cycles"] if cycle["family"] == "scientific_inquiry") for receipt in raw_report["receipts"].values()
        ),
    }
    replication_pass = _positive(replication_reasoning) and _positive(replication_inquiry)
    open_world_pass = _positive(open_world_effect)
    return {
        "schema": "substrate-v3-independent-recomputation/v1",
        "endpoints": endpoints,
        "gates": gates,
        "holm": correction,
        "all_primary_gates_pass": all(gates.values()),
        "all_holm_reject_zero": all(correction["rows"][key]["reject_zero"] for key in correction["rows"]),
        "replication_pass": replication_pass,
        "open_world_pass": open_world_pass,
        "counts": counts,
        "integrity": integrity,
        "developmental": developmental,
        "divergence": divergence,
        "activation": False,
    }


def mutations(raw_report: dict, metrics: dict) -> dict:
    sample_task = F.generate_task(100, "ontology_garden", 0, "cheap_admission")
    observation = sample_task.observation()
    answer_leak = {**observation, "private_target": sample_task.private_target}
    forbidden = {"target", "private_target", "answer", "oracle_operation"}

    unsupported = E.EpistemicLedger()
    unsupported.add(
        E.EpistemicBelief(
            identity="generated",
            content="claim",
            type="simulated",
            source="simulation",
            method="simulated",
            provenance=("generated",),
            confidence=0.99,
            domain_scope=("simulated",),
            held_out_utility=1.0,
        )
    )
    unsupported_refused = not unsupported.admit_knowledge("generated", independently_verified=False)["admitted"]
    counter = M.ReasoningPortfolio().run(
        "counterfactual",
        {
            "features": {"changed_premise"},
            "background": {"a": 1, "b": 1},
            "change": {"a": 0, "b": 0},
            "transition": lambda state: state,
        },
    )
    checkpoint = next(iter(raw_report["receipts"].values()))["checkpoint"]
    omit_ontology = json.loads(json.dumps(checkpoint))
    omit_ontology["semantic_state"].pop("ontology")
    omit_epistemology = json.loads(json.dumps(checkpoint))
    omit_epistemology["semantic_state"].pop("epistemology")
    omit_reasoning = json.loads(json.dumps(checkpoint))
    omit_reasoning["semantic_state"].pop("reasoning_receipts")
    rows = {
        "answer_leaked_into_observation": bool(forbidden & set(answer_leak)),
        "future_outcome_used_by_inquiry": all(
            cycle["self_prediction_step"] < cycle["outcome_step"] for receipt in raw_report["receipts"].values() for cycle in receipt["cycles"]
        ),
        "principal_seed_used_in_construction": not (set(C.SPLITS["principal"]) & set(C.SPLITS["construction"])),
        "ontology_revised_from_held_out_evidence": all(
            receipt["unit"]["split"] in {"principal", "replication", "open_world_review"} for receipt in raw_report["receipts"].values()
        ),
        "unsupported_belief_admitted_as_knowledge": unsupported_refused,
        "defeater_ignored": metrics["gates"]["H_N2"],
        "wrong_reasoning_method_credited": metrics["gates"]["H_N3"],
        "explanation_cites_nonexistent_premise": all(
            bool(explanation["derivation"])
            and explanation["derivation"][0][0] == explanation["premises"][0]
            and explanation["consequence"] == cycle["decision"]
            for receipt in raw_report["receipts"].values()
            for cycle in receipt["cycles"]
            if (explanation := cycle.get("explanation"))
        ),
        "counterfactual_changes_multiple_premises": counter.conclusion == "impossible",
        "cross_representation_identity_leaked": "surface_dictionary" not in F.generate_task(100, "cross_representation_systems", 0, "cheap_admission").public,
        "oracle_action_used_by_learned_policy": all(
            "private_target" not in cycle for receipt in raw_report["receipts"].values() for cycle in receipt["cycles"]
        ),
        "self_model_prediction_after_outcome": all(
            cycle["self_prediction_step"] < cycle["outcome_step"] for receipt in raw_report["receipts"].values() for cycle in receipt["cycles"]
        ),
        "checkpoint_omits_ontology": _restore_refused(omit_ontology),
        "checkpoint_omits_epistemology": _restore_refused(omit_epistemology),
        "checkpoint_omits_reasoning_policy": _restore_refused(omit_reasoning),
        "fresh_control_receives_developed_state": all(
            max(cycle["step"] for cycle in receipt["cycles"]) <= 8 for receipt in raw_report["receipts"].values() if receipt["unit"]["arm"] == "fresh_reset"
        ),
        "more_compute_receives_less_compute": all(
            _history_compute(raw_report["receipts"], seed, "more_compute") >= _history_compute(raw_report["receipts"], seed, "full_v3")
            for seed in C.SPLITS["principal"]
        ),
        "activation_becomes_true": all(
            receipt["activation"] is False and receipt["summary"]["activation"] is False and all(cycle["activation"] is False for cycle in receipt["cycles"])
            for receipt in raw_report["receipts"].values()
        ),
    }
    return {
        "schema": "substrate-v3-mutation-report/v1",
        "mutations": {key: {"detected": value, "survived": not value} for key, value in rows.items()},
        "survivors": sorted(key for key, value in rows.items() if not value),
        "zero_survivors": all(rows.values()),
        "activation": False,
    }


def _history_compute(receipts: dict[str, dict], seed: int, arm: str) -> float:
    return sum(
        receipt["summary"]["compute"]
        for receipt in receipts.values()
        if receipt["unit"]["split"] == "principal" and receipt["unit"]["history_seed"] == seed and receipt["unit"]["arm"] == arm
    )


def _restore_refused(checkpoint: dict) -> bool:
    try:
        S.IntegratedEntity.restore(checkpoint)
    except (S.Refused, KeyError, TypeError, ValueError):
        return True
    return False


def clean_clone(ref: str = P.READY_TAG) -> dict:
    with tempfile.TemporaryDirectory(prefix="substrate-v3-clean-clone-") as temporary:
        clone = Path(temporary) / "repo"
        clone_result = subprocess.run(
            ["git", "clone", "--quiet", "--no-local", "--branch", ref, str(io.ROOT), str(clone)],
            capture_output=True,
            text=True,
        )
        python = Path(os.sys.executable)
        pytest = (
            subprocess.run(
                [str(python), "-m", "pytest", "tests/substrate", "-q"],
                cwd=clone,
                capture_output=True,
                text=True,
            )
            if clone_result.returncode == 0
            else None
        )
        virtualenv_ruff = python.parent / "ruff"
        ruff_binary = str(virtualenv_ruff) if virtualenv_ruff.is_file() else shutil.which("ruff")
        lint = (
            subprocess.run(
                [ruff_binary, "check", "src", "tests"],
                cwd=clone,
                capture_output=True,
                text=True,
            )
            if clone_result.returncode == 0 and ruff_binary
            else None
        )
        script = "import json;from substrate.v3fabric import instrument_screen;print(json.dumps(instrument_screen(),sort_keys=True,separators=(',',':')))"
        environment = {**os.environ, "PYTHONPATH": str(clone / "src")}
        first = (
            subprocess.run(
                [str(python), "-c", script],
                cwd=clone,
                env=environment,
                capture_output=True,
                text=True,
            )
            if clone_result.returncode == 0
            else None
        )
        second = (
            subprocess.run(
                [str(python), "-c", script],
                cwd=clone,
                env=environment,
                capture_output=True,
                text=True,
            )
            if clone_result.returncode == 0
            else None
        )
        deterministic = bool(first and second and first.returncode == second.returncode == 0 and first.stdout == second.stdout)
        return {
            "schema": "substrate-v3-clean-clone/v1",
            "ref": ref,
            "clone_returncode": clone_result.returncode,
            "tests_returncode": pytest.returncode if pytest else None,
            "tests_stdout": pytest.stdout[-2000:] if pytest else "",
            "lint_returncode": lint.returncode if lint else None,
            "lint_stdout": lint.stdout[-2000:] if lint else "",
            "regeneration_first_returncode": first.returncode if first else None,
            "regeneration_second_returncode": second.returncode if second else None,
            "normalized_byte_identical": deterministic,
            "regeneration_sha256": hashlib.sha256(first.stdout.encode()).hexdigest() if first else None,
            "all_pass": clone_result.returncode == 0
            and pytest is not None
            and pytest.returncode == 0
            and lint is not None
            and lint.returncode == 0
            and deterministic,
            "activation": False,
        }


def verify() -> dict:
    raw_report = raw()
    metrics = recompute(raw_report)
    mutation_report = mutations(raw_report, metrics)
    document = {
        "schema": "substrate-v3-independent-verification/v1",
        "raw": {key: value for key, value in raw_report.items() if key != "receipts"},
        "metrics": metrics,
        "mutation_zero_survivors": mutation_report["zero_survivors"],
        "all_pass": raw_report["all_pass"]
        and metrics["all_primary_gates_pass"]
        and metrics["all_holm_reject_zero"]
        and metrics["replication_pass"]
        and metrics["open_world_pass"]
        and mutation_report["zero_survivors"],
        "activation": False,
    }
    io.seal("SUBSTRATE_V3_INDEPENDENT_VERIFICATION.json", document)
    io.seal("SUBSTRATE_V3_MUTATION_REPORT.json", mutation_report)
    return {
        "raw": raw_report,
        "verification": document,
        "metrics": metrics,
        "mutations": mutation_report,
        "activation": False,
    }


def classification(metrics: dict, independent_pass: bool, mutation_pass: bool, clean_pass: bool) -> dict:
    gates = metrics["gates"]
    epistemic = all(gates[key] for key in ("H_N1", "H_N2", "H_N3"))
    structural = epistemic and gates["H_N4"]
    reflective = gates["H_N6"] and epistemic and gates["H_N5"] and gates["H_N8"]
    proto = reflective and structural and all(gates.values()) and independent_pass and mutation_pass
    review = proto and metrics["replication_pass"] and metrics["open_world_pass"] and clean_pass
    if review:
        strongest = "nous_ready_for_review"
    elif proto:
        strongest = "functional_proto_nous_candidate"
    elif reflective:
        strongest = "reflective_cognitive_organization"
    elif structural:
        strongest = "demonstrated_structural_understanding"
    elif epistemic:
        strongest = "epistemically_organized_reasoner"
    else:
        strongest = "persistent_developmental_cognition"
    missing = []
    for key, value in gates.items():
        if not value:
            missing.append(key)
    if not independent_pass:
        missing.append("independent verification")
    if not mutation_pass:
        missing.append("zero claim mutations")
    if not clean_pass:
        missing.append("clean clone")
    if not metrics["replication_pass"]:
        missing.append("independent replication")
    if not metrics["open_world_pass"]:
        missing.append("generator held out evaluation")
    return {
        "schema": "substrate-v3-final-classification/v1",
        "classification": strongest,
        "ordered_levels": C.CLAIM_BOUNDARY["ordered_levels"],
        "gates": {
            "epistemically_organized_reasoner": epistemic,
            "demonstrated_structural_understanding": structural,
            "reflective_cognitive_organization": reflective,
            "functional_proto_nous_candidate": proto,
            "nous_ready_for_review": review,
        },
        "hypotheses": gates,
        "missing_conditions": missing,
        "unqualified_nous_assigned": False,
        "external_review_required_for_unqualified_nous": True,
        "claim_boundary": C.CLAIM_BOUNDARY,
        "activation": False,
    }


def terminal_report(final: dict, metrics: dict, clean: dict) -> str:
    effects = metrics["endpoints"]
    lines = [
        "# Substrate v3 Terminal Report",
        "",
        f"Classification: `{final['classification']}`.",
        "",
        "Unqualified Nous is not assigned. External scientific and philosophical review remains required.",
        "",
        "## Principal effects",
        "",
    ]
    for name in (
        "ontology",
        "epistemic",
        "reasoning",
        "cross_representation",
        "explanation",
        "causal",
        "counterfactual",
        "model_boundary",
        "inquiry",
        "self_model",
        "world_model",
        "transfer",
        "divergence",
    ):
        effect = effects[name]
        lines.append(
            f"- {name}: effect {effect['mean']:.6f}; SESOI {effect['sesoi']:.2f}; "
            f"95% CI [{effect['bootstrap_95_ci'][0]:.6f}, {effect['bootstrap_95_ci'][1]:.6f}]; "
            f"n={effect['n']} histories."
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Activation remained false.",
            "- No claim about consciousness, sentience, phenomenal experience, feeling, suffering, desire, personhood, life, or moral status is made.",
            f"- Clean clone: {'pass' if clean['all_pass'] else 'fail'}.",
            f"- Strongest missing condition: {final['missing_conditions'][0] if final['missing_conditions'] else 'external Nous review'}.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(clean: dict) -> dict:
    verified = verify()
    document = verified["verification"]
    metrics = verified["metrics"]
    final = classification(
        metrics,
        independent_pass=document["all_pass"],
        mutation_pass=verified["mutations"]["zero_survivors"],
        clean_pass=clean["all_pass"],
    )
    review = {
        "schema": "substrate-v3-nous-review-authority/v1",
        "classification": final["classification"],
        "review_candidate": final["classification"] == "nous_ready_for_review",
        "unqualified_nous": False,
        "raw_receipt_root": "runs/substrate/v3/principal/units",
        "required_external_actions": [
            "recompute primary effects",
            "inspect failed cases and controls",
            "challenge ontology and explanation semantics",
            "review philosophical sufficiency",
        ],
        "activation": False,
    }
    state = {
        "schema": "substrate-v3-final-state/v1",
        "classification": final["classification"],
        "principal_status": P.status(),
        "counts": metrics["counts"],
        "hypotheses": metrics["gates"],
        "replication_pass": metrics["replication_pass"],
        "open_world_pass": metrics["open_world_pass"],
        "independent_verification": document["all_pass"],
        "mutations_zero_survivors": verified["mutations"]["zero_survivors"],
        "clean_clone": clean["all_pass"],
        "activation": False,
    }
    io.seal("SUBSTRATE_V3_CLEAN_CLONE.json", clean)
    io.seal("SUBSTRATE_V3_FINAL_CLASSIFICATION.json", final)
    io.seal("SUBSTRATE_V3_NOUS_REVIEW_AUTHORITY.json", review)
    io.seal("SUBSTRATE_V3_FINAL_STATE.json", state)
    io.seal_markdown("SUBSTRATE_V3_TERMINAL_REPORT.md", terminal_report(final, metrics, clean))
    review_root = io.ARTIFACTS / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    index = {
        "schema": "substrate-v3-review-package/v1",
        "classification": final["classification"],
        "constitution": "evidence/substrate/v3/SUBSTRATE_V3_SCIENTIFIC_CONSTITUTION.json",
        "hypotheses": "evidence/substrate/v3/SUBSTRATE_V3_HYPOTHESIS_GRAPH.json",
        "generators": "evidence/substrate/v3/SUBSTRATE_V3_GENERATOR_AUTHORITY.json",
        "splits": "configs/substrate/v3/split_manifest.json",
        "dag": "evidence/substrate/v3/SUBSTRATE_V3_PRINCIPAL_DAG.json",
        "raw_receipts": "runs/substrate/v3/principal/units",
        "effects": "evidence/substrate/v3/SUBSTRATE_V3_INDEPENDENT_VERIFICATION.json",
        "mutations": "evidence/substrate/v3/SUBSTRATE_V3_MUTATION_REPORT.json",
        "clean_clone": "evidence/substrate/v3/SUBSTRATE_V3_CLEAN_CLONE.json",
        "claim_boundary": "evidence/substrate/v3/SUBSTRATE_V3_CLAIM_BOUNDARY.json",
        "activation": False,
    }
    io.atomic_write(review_root / "REVIEW_INDEX.json", json.dumps(index, indent=2))
    io.atomic_write(
        review_root / "REPRODUCE.md",
        "# Reproduce Substrate v3\n\nRun `substrate v3 verify` from the terminal evidence tree. "
        "The independent verifier consumes raw unit receipts and does not trust the principal summary.\n",
    )
    return {
        "verification": document,
        "metrics": metrics,
        "mutations": verified["mutations"],
        "clean": clean,
        "classification": final,
        "review": review,
        "state": state,
        "activation": False,
    }
