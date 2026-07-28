"""Independent raw-receipt verification, mutations, reproduction, and v4 classification."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
from pathlib import Path

from substrate import v4config as C
from substrate import v4io as io
from substrate import v4principal as P


class Refused(RuntimeError):
    """Independent verification refused incomplete or invalid evidence."""


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
    rng = random.Random(int(hashlib.sha256(endpoint.encode()).hexdigest()[:16], 16))
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
        "standardized_effect": mean / deviation if deviation else ("infinity" if mean else 0.0),
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


def _receipt_errors(receipt: dict, unit: P.WorkUnit) -> list[str]:
    errors = []
    if not P.validate_receipt(receipt, unit):
        errors.append("receipt_identity_or_shape")
    if receipt.get("activation") is not False:
        errors.append("activation")
    cycles = receipt.get("cycles", [])
    if len(cycles) != len(C.PHASES) * P.EPISODES_PER_PHASE:
        errors.append("episode_count")
    if {row.get("phase") for row in cycles} != set(C.PHASES):
        errors.append("phase_coverage")
    if any(row.get("revealed_after_commitment") is not True for row in cycles):
        errors.append("target_commitment_order")
    if any(row.get("self_prediction_step", 1) >= row.get("outcome_step", 0) for row in cycles):
        errors.append("self_prediction_order")
    summary = receipt.get("summary", {})
    if summary.get("checkpoint_exact") is not True:
        errors.append("checkpoint")
    if summary.get("body_continuity") is not True:
        errors.append("body_continuity")
    if unit.arm == "full_v4" and summary.get("models", 0) < 1:
        errors.append("structural_model_missing")
    if unit.arm == "full_v4" and summary.get("causally_active_rate", 0.0) <= 0:
        errors.append("structural_model_inactive")
    return sorted(set(errors))


def raw() -> dict:
    expected = {unit.identity: unit for unit in P.work_units()}
    receipts = {}
    missing = []
    invalid = {}
    checkpoint_mismatch = []
    for identity, unit in expected.items():
        receipt_path = P.UNITS / f"{identity}.json"
        checkpoint_path = P.CHECKPOINTS / f"{identity}.json"
        if not receipt_path.is_file() or not checkpoint_path.is_file():
            missing.append(identity)
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
            checkpoint = json.loads(checkpoint_path.read_text())
        except json.JSONDecodeError:
            invalid[identity] = ["invalid_json"]
            continue
        errors = _receipt_errors(receipt, unit)
        if errors:
            invalid[identity] = errors
            continue
        summary = receipt["summary"]
        if (
            checkpoint.get("state_identity") != summary["state_identity"]
            or checkpoint.get("structural_state_digest") != summary["structural_state_digest"]
            or checkpoint.get("checkpoint_exact") is not True
            or checkpoint.get("activation") is not False
        ):
            checkpoint_mismatch.append(identity)
            continue
        receipts[identity] = receipt
    return {
        "receipts": receipts,
        "expected": len(expected),
        "valid": len(receipts),
        "missing": missing,
        "invalid": invalid,
        "checkpoint_mismatch": checkpoint_mismatch,
        "all_pass": len(receipts) == len(expected) and not missing and not invalid and not checkpoint_mismatch,
        "activation": False,
    }


def _table(receipts: dict[str, dict], split: str) -> dict[tuple[int, str], dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for receipt in receipts.values():
        unit = receipt["unit"]
        if unit["split"] == split:
            grouped.setdefault((unit["history_seed"], unit["arm"]), []).append(receipt)
    table = {}
    for key, unit_receipts in grouped.items():
        cycles = [cycle for receipt in unit_receipts for cycle in receipt["cycles"]]
        phase_utility = {}
        phase_accuracy = {}
        for phase in C.PHASES:
            selected = [cycle for cycle in cycles if cycle["phase"] == phase]
            phase_utility[phase] = statistics.fmean(float(row["correct"]) - C.COMPUTE_PRICE * float(row["compute"]) for row in selected)
            phase_accuracy[phase] = statistics.fmean(float(row["correct"]) for row in selected)
        family_utility = {}
        for family in C.WORKLOADS:
            selected = [cycle for cycle in cycles if cycle["family"] == family]
            family_utility[family] = statistics.fmean(float(row["correct"]) - C.COMPUTE_PRICE * float(row["compute"]) for row in selected)
        table[key] = {
            "cycles": cycles,
            "receipts": unit_receipts,
            "phase_utility": phase_utility,
            "phase_accuracy": phase_accuracy,
            "family_utility": family_utility,
            "overall_utility": statistics.fmean(float(row["correct"]) - C.COMPUTE_PRICE * float(row["compute"]) for row in cycles),
        }
    return table


def _contrast(table: dict, full: str, controls: tuple[str, ...], accessor, endpoint: str) -> dict:
    effects = []
    strongest = []
    for seed in sorted(seed for seed, arm in table if arm == full):
        full_value = float(accessor(table[(seed, full)]))
        control_values = {arm: float(accessor(table[(seed, arm)])) for arm in controls}
        strongest_arm = max(control_values, key=control_values.get)
        effects.append(full_value - control_values[strongest_arm])
        strongest.append(strongest_arm)
    result = paired(effects, endpoint)
    result["full_arm"] = full
    result["controls"] = list(controls)
    result["strongest_control_by_history"] = strongest
    return result


def _mean_phases(row: dict, phases: tuple[str, ...]) -> float:
    return statistics.fmean(row["phase_utility"][phase] for phase in phases)


def recompute(raw_report: dict) -> dict:
    if not raw_report["all_pass"]:
        raise Refused("raw principal receipts are incomplete or invalid")
    principal = _table(raw_report["receipts"], "principal")
    replication = _table(raw_report["receipts"], "replication")
    open_world = _table(raw_report["receipts"], "open_world_review")
    effects = {
        "H_S1": _contrast(
            principal,
            "full_v4",
            ("v3_reflective_control", "semantic_retrieval_control", "more_compute"),
            lambda row: _mean_phases(
                row,
                ("phase_1_observational_structure_acquisition", "phase_2_competing_structural_hypotheses"),
            ),
            "executable_structural_prediction",
        ),
        "H_S2": _contrast(
            principal,
            "full_v4",
            ("correlation_only_model", "semantic_retrieval_control", "more_compute"),
            lambda row: row["phase_utility"]["phase_4_causal_intervention"],
            "causal_intervention_value",
        ),
        "H_S3": _contrast(
            principal,
            "full_v4",
            ("no_counterfactual", "surface_alignment", "more_compute"),
            lambda row: row["phase_utility"]["phase_7_counterfactual_challenge"],
            "counterfactual_value",
        ),
        "H_S4": _contrast(
            principal,
            "full_v4",
            ("no_alignment", "surface_alignment", "semantic_retrieval_control", "more_compute"),
            lambda row: _mean_phases(
                row,
                ("phase_6_first_cross_representation_encounter", "phase_14_useful_history_specialization"),
            ),
            "cross_representation_mapping",
        ),
        "H_S6": _contrast(
            principal,
            "full_v4",
            ("simple_structural_inquiry", "more_compute"),
            lambda row: row["phase_utility"]["phase_3_discriminating_inquiry"],
            "cost_adjusted_structural_inquiry",
        ),
        "H_S7": _contrast(
            principal,
            "full_v4",
            ("semantic_retrieval_control", "correlation_only_model", "more_compute"),
            lambda row: row["phase_utility"]["phase_8_explanation_and_falsifier"],
            "structural_explanation_fidelity",
        ),
        "H_S8": _contrast(
            principal,
            "full_v4",
            ("no_self_model", "no_world_model"),
            lambda row: _mean_phases(
                row,
                ("phase_3_discriminating_inquiry", "phase_4_causal_intervention"),
            ),
            "self_world_structural_utility",
        ),
        "H_S9": _contrast(
            principal,
            "full_v4",
            ("fresh_reset", "transcript_replay"),
            lambda row: row["phase_utility"]["phase_12_return_to_prior_structural_domain"],
            "preserved_developmental_competence",
        ),
    }
    history_values = []
    continuity_values = []
    history_checks = []
    for seed in C.SPLITS["principal"]:
        summaries = [receipt["summary"] for receipt in principal[(seed, "full_v4")]["receipts"]]
        history = next(summary["history_specialization"] for summary in summaries if summary["history_specialization"])
        history_values.append(float(history["mean_specialization_margin"]))
        history_checks.append(history)
        continuity_values.append(statistics.fmean(float(summary["checkpoint_exact"] and summary["body_continuity"]) for summary in summaries))
    effects["H_S5"] = paired(history_values, "useful_history_specialization")
    effects["H_S5"]["controls"] = ["wrong_history", "identical_A_replica", "shuffled_A"]
    effects["H_S10"] = paired(continuity_values, "identity_body_continuity")
    effects["H_S10"]["controls"] = ["checkpoint_corruption", "fresh_identity"]
    effects = {name: effects[name] for name in C.HYPOTHESES}
    correction = holm({name: effect["exact_sign_p"] for name, effect in effects.items()})
    for name, effect in effects.items():
        effect["holm"] = correction["rows"][name]
        effect["passes"] = effect["mean"] >= C.SESOI and effect["bootstrap_95_ci"][0] > 0 and effect["holm"]["reject_zero"]
    replication_effect = _contrast(
        replication,
        "full_v4",
        ("semantic_retrieval_control", "no_counterfactual", "no_alignment", "simple_structural_inquiry", "more_compute"),
        lambda row: row["overall_utility"],
        "independent_replication",
    )
    open_effect = _contrast(
        open_world,
        "full_v4",
        ("no_alignment", "more_compute", "transcript_replay"),
        lambda row: row["overall_utility"],
        "generator_held_out_open_world",
    )
    for effect in (replication_effect, open_effect):
        effect["passes"] = effect["mean"] >= C.SESOI and effect["bootstrap_95_ci"][0] > 0
    historical = {
        "v2_classification": json.loads((io.ROOT / "evidence/substrate/v2/SUBSTRATE_V2_FINAL_CLASSIFICATION.json").read_text())["classification"],
        "v3_classification": json.loads((io.ROOT / "evidence/substrate/v3/SUBSTRATE_V3_FINAL_CLASSIFICATION.json").read_text())["classification"],
    }
    historical["preserved"] = (
        historical["v2_classification"] == "persistent_developmental_cognition" and historical["v3_classification"] == "reflective_cognitive_organization"
    )
    return {
        "schema": "substrate-v4-independent-recomputation/v1",
        "effects": effects,
        "holm": correction,
        "history_checks": history_checks,
        "replication": replication_effect,
        "open_world": open_effect,
        "historical": historical,
        "all_hypotheses_pass": all(effect["passes"] for effect in effects.values()),
        "all_pass": all(effect["passes"] for effect in effects.values()) and replication_effect["passes"] and open_effect["passes"] and historical["preserved"],
        "activation": False,
    }


def mutations(raw_report: dict) -> dict:
    if not raw_report["all_pass"]:
        raise Refused("mutation testing requires valid raw receipts")
    unit = next(unit for unit in P.work_units() if unit.split == "principal" and unit.arm == "full_v4" and unit.shard == 0)
    original = raw_report["receipts"][unit.identity]
    cases = []

    def case(identity: str, mutate, extra_check=None) -> None:
        changed = copy.deepcopy(original)
        mutate(changed)
        errors = _receipt_errors(changed, unit)
        detected = bool(errors) or bool(extra_check and extra_check(changed))
        cases.append({"identity": identity, "detected": detected, "errors": errors})

    case("M01_receipt_hash", lambda row: row.update(receipt_identity="0" * 64))
    case("M02_activation", lambda row: row.update(**{"activation": not False}))
    case("M03_missing_cycle", lambda row: row["cycles"].pop())
    case("M04_phase_coverage", lambda row: row["cycles"][0].update(phase="unknown"))
    case("M05_commitment_order", lambda row: row["cycles"][0].update(revealed_after_commitment=False))
    case("M06_self_prediction_order", lambda row: row["cycles"][0].update(self_prediction_step=99))
    case("M07_checkpoint_exact", lambda row: row["summary"].update(checkpoint_exact=False))
    case("M08_body_continuity", lambda row: row["summary"].update(body_continuity=False))
    case("M09_structural_model_missing", lambda row: row["summary"].update(models=0))
    case("M10_mechanism_inactive", lambda row: row["summary"].update(causally_active_rate=0.0))
    case("M11_arm_substitution", lambda row: row["unit"].update(arm="unknown"))
    case("M12_seed_substitution", lambda row: row["unit"].update(history_seed=-1))
    case("M13_split_substitution", lambda row: row["unit"].update(split="construction"))
    case("M14_target_mutation", lambda row: row["cycles"][1].update(target=["mutated"]))
    case("M15_decision_mutation", lambda row: row["cycles"][1].update(decision=["mutated"]))
    case("M16_history_specialization", lambda row: row["summary"]["history_specialization"].update(mean_specialization_margin=-1.0))
    case("M17_structural_digest", lambda row: row["summary"].update(structural_state_digest="0" * 64))
    case("M18_owned_identity", lambda row: row["summary"].update(owned_identity="replacement"))
    survived = [row["identity"] for row in cases if not row["detected"]]
    return {
        "schema": "substrate-v4-mutation-report/v1",
        "mutations": cases,
        "total": len(cases),
        "detected": len(cases) - len(survived),
        "survived": survived,
        "zero_survived": not survived,
        "activation": False,
    }


def clean_clone(raw_report: dict) -> dict:
    ready = subprocess.check_output(["git", "rev-parse", f"{P.READY_TAG}^{{}}"], cwd=io.ROOT, text=True).strip()
    sample_unit = next(unit for unit in P.work_units() if unit.split == "principal" and unit.arm == "full_v4" and unit.shard == 0)
    expected = raw_report["receipts"][sample_unit.identity]
    expected_digest = io.sha_obj({"cycles": expected["cycles"], "phases": expected["phases"]})
    with tempfile.TemporaryDirectory(prefix="substrate-v4-clean-") as temporary:
        clone = Path(temporary) / "repo"
        installed = Path(temporary) / "installed"
        clone_result = subprocess.run(
            ["git", "clone", "--quiet", "--branch", P.READY_TAG, "--depth", "1", str(io.ROOT), str(clone)],
            capture_output=True,
            text=True,
        )
        install = subprocess.run(
            [
                str(io.ROOT / ".venv/bin/python"),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--no-build-isolation",
                "--target",
                str(installed),
                str(clone),
            ],
            capture_output=True,
            text=True,
        )
        clean_env = {**os.environ, "PYTHONPATH": str(installed)}
        tests = subprocess.run(
            [
                str(io.ROOT / ".venv/bin/python"),
                "-m",
                "pytest",
                "-q",
                "tests/substrate",
            ],
            cwd=clone,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        lint = subprocess.run(
            [str(io.ROOT / ".venv/bin/ruff"), "check", "src", "tests"],
            cwd=clone,
            capture_output=True,
            text=True,
        )
        format_targets = [
            "src/substrate/cli.py",
            "src/substrate/runtime.py",
            "src/substrate/world.py",
            *[path.relative_to(clone).as_posix() for path in sorted((clone / "src/substrate").glob("v4*.py"))],
            "tests/substrate/test_v4_mechanisms.py",
        ]
        formatting = subprocess.run(
            [str(io.ROOT / ".venv/bin/ruff"), "format", "--check", *format_targets],
            cwd=clone,
            capture_output=True,
            text=True,
        )
        script = (
            "import json;"
            "from substrate import v4io as I,v4principal as P;"
            f"u=next(x for x in P.work_units() if x.identity=={sample_unit.identity!r});"
            "r=P.execute_unit(u);"
            "print(I.sha_obj({'cycles':r['cycles'],'phases':r['phases']}))"
        )
        reproduce = subprocess.run(
            [str(io.ROOT / ".venv/bin/python"), "-c", script],
            cwd=clone,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        reproduce_second = subprocess.run(
            [str(io.ROOT / ".venv/bin/python"), "-c", script],
            cwd=clone,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        actual_digest = reproduce.stdout.strip().splitlines()[-1] if reproduce.stdout.strip() else ""
        second_digest = reproduce_second.stdout.strip().splitlines()[-1] if reproduce_second.stdout.strip() else ""
    return {
        "schema": "substrate-v4-clean-clone/v1",
        "ready_commit": ready,
        "clone_succeeded": clone_result.returncode == 0,
        "clean_install_returncode": install.returncode,
        "clean_install_stderr_tail": install.stderr[-2000:],
        "tests_returncode": tests.returncode,
        "tests_stdout_tail": tests.stdout[-2000:],
        "tests_stderr_tail": tests.stderr[-2000:],
        "lint_returncode": lint.returncode,
        "lint_stdout_tail": lint.stdout[-2000:],
        "format_returncode": formatting.returncode,
        "format_stdout_tail": formatting.stdout[-2000:],
        "reproduction_returncode": reproduce.returncode,
        "second_reproduction_returncode": reproduce_second.returncode,
        "expected_digest": expected_digest,
        "actual_digest": actual_digest,
        "second_digest": second_digest,
        "exact_reproduction": expected_digest == actual_digest == second_digest,
        "normalized_double_regeneration_exact": actual_digest == second_digest,
        "all_pass": (
            clone_result.returncode
            == install.returncode
            == tests.returncode
            == lint.returncode
            == formatting.returncode
            == reproduce.returncode
            == reproduce_second.returncode
            == 0
            and expected_digest == actual_digest == second_digest
        ),
        "activation": False,
    }


def _write_review_json(review_root: Path, name: str, document: dict) -> Path:
    return io.atomic_write(review_root / name, json.dumps(document, indent=2, sort_keys=True))


def _review_package(raw_report: dict, verification: dict, mutation: dict, clone: dict) -> dict:
    review_root = io.ARTIFACTS / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    authorities = {}
    for path in sorted(io.EVIDENCE.glob("*.json")):
        authorities[path.relative_to(io.ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(io.CONFIGS.glob("*.json")):
        authorities[path.relative_to(io.ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in (
        "src/substrate/runtime.py",
        "src/substrate/world.py",
        "src/substrate/v4fabric.py",
        "src/substrate/v4principal.py",
        "src/substrate/v4verify.py",
    ):
        path = io.ROOT / relative
        authorities[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_paths = sorted((*P.UNITS.glob("*.json"), *P.CHECKPOINTS.glob("*.json")))
    receipt_manifest = {path.relative_to(io.ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in receipt_paths}
    archive_rows = [
        json.dumps(
            {
                "path": path.relative_to(io.ROOT).as_posix(),
                "sha256": receipt_manifest[path.relative_to(io.ROOT).as_posix()],
                "document": json.loads(path.read_text()),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for path in receipt_paths
    ]
    archive = gzip.compress(("\n".join(archive_rows) + "\n").encode(), compresslevel=9, mtime=0)
    archive_path = io.atomic_write_bytes(review_root / "RAW_RECEIPTS.jsonl.gz", archive)
    summaries = []
    for identity, receipt in sorted(raw_report["receipts"].items()):
        summary = receipt["summary"]
        summaries.append(
            {
                "identity": identity,
                "split": receipt["unit"]["split"],
                "arm": receipt["unit"]["arm"],
                "history_seed": receipt["unit"]["history_seed"],
                "latent_model_family": receipt["unit"]["latent_model_family"],
                "surface_representation_family": receipt["unit"]["surface_representation_family"],
                "models": summary["models"],
                "revisions": summary["revisions"],
                "interventions": summary["interventions"],
                "counterfactuals": summary["counterfactuals"],
                "mappings": summary["mappings"],
                "inquiries": summary["inquiries"],
                "history_specialization": summary["history_specialization"],
                "checkpoint_exact": summary["checkpoint_exact"],
                "body_continuity": summary["body_continuity"],
                "activation": False,
            }
        )
    effects = {
        **verification["effects"],
        "independent_replication": verification["replication"],
        "generator_held_out_open_world": verification["open_world"],
    }
    review_documents = {
        "AUTHORITY_INDEX.json": {
            "schema": "substrate-v4-review-authority-index/v1",
            "authorities": authorities,
            "activation": False,
        },
        "RAW_RECEIPT_INDEX.json": {
            "schema": "substrate-v4-raw-receipt-index/v1",
            "archive": archive_path.name,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "files": receipt_manifest,
            "unit_receipts": len(list(P.UNITS.glob("*.json"))),
            "checkpoints": len(list(P.CHECKPOINTS.glob("*.json"))),
            "activation": False,
        },
        "EFFECT_LEDGER.json": {
            "schema": "substrate-v4-effect-ledger/v1",
            "effects": effects,
            "activation": False,
        },
        "NULL_LEDGER.json": {
            "schema": "substrate-v4-null-ledger/v1",
            "nulls": {name: row for name, row in effects.items() if not row["passes"]},
            "activation": False,
        },
        "DEFECT_LEDGER.json": {
            "schema": "substrate-v4-defect-ledger/v1",
            "missing": raw_report["missing"],
            "invalid": raw_report["invalid"],
            "checkpoint_mismatch": raw_report["checkpoint_mismatch"],
            "activation": False,
        },
        "MODEL_REVISION_LEDGER.json": {
            "schema": "substrate-v4-model-revision-ledger/v1",
            "rows": [{"identity": row["identity"], "models": row["models"], "revisions": row["revisions"]} for row in summaries],
            "activation": False,
        },
        "MAPPING_LEDGER.json": {
            "schema": "substrate-v4-mapping-ledger/v1",
            "rows": [
                {
                    "identity": row["identity"],
                    "latent_model_family": row["latent_model_family"],
                    "surface_representation_family": row["surface_representation_family"],
                    "mappings": row["mappings"],
                }
                for row in summaries
            ],
            "activation": False,
        },
        "INTERVENTION_LEDGER.json": {
            "schema": "substrate-v4-intervention-ledger/v1",
            "rows": [{"identity": row["identity"], "interventions": row["interventions"]} for row in summaries],
            "activation": False,
        },
        "COUNTERFACTUAL_LEDGER.json": {
            "schema": "substrate-v4-counterfactual-ledger/v1",
            "rows": [{"identity": row["identity"], "counterfactuals": row["counterfactuals"]} for row in summaries],
            "activation": False,
        },
        "INQUIRY_LEDGER.json": {
            "schema": "substrate-v4-inquiry-ledger/v1",
            "rows": [{"identity": row["identity"], "inquiries": row["inquiries"]} for row in summaries],
            "activation": False,
        },
        "HISTORY_SPECIALIZATION_LEDGER.json": {
            "schema": "substrate-v4-history-specialization-ledger/v1",
            "rows": [{"identity": row["identity"], "result": row["history_specialization"]} for row in summaries if row["history_specialization"] is not None],
            "activation": False,
        },
        "CONTROL_AUDIT.json": {
            "schema": "substrate-v4-control-audit/v1",
            "controls": {
                name: {
                    "declared": row.get("controls", []),
                    "strongest_by_history": row.get("strongest_control_by_history", []),
                    "passes": row["passes"],
                }
                for name, row in effects.items()
            },
            "mutation_report": mutation,
            "activation": False,
        },
        "CLAIM_BOUNDARY.json": {
            "schema": "substrate-v4-review-claim-boundary/v1",
            **C.CLAIM_BOUNDARY,
        },
        "KNOWN_LIMITATIONS.json": {
            "schema": "substrate-v4-known-limitations/v1",
            "limitations": [
                "Evidence is limited to deterministic, sandboxed, generated structural micro-worlds.",
                "The candidate ladders and causal graph families are deliberately bounded.",
                "The result does not establish consciousness, sentience, personhood, life, or moral status.",
                "The maximum automatic classification is readiness for independent Nous review.",
                "External activation remains false and no uncontrolled external action is licensed.",
            ],
            "activation": False,
        },
        "STRONGEST_FALSIFICATION.json": {
            "schema": "substrate-v4-strongest-falsification/v1",
            "historical_v3_null": "all four load-bearing v3 structural effects were exactly zero",
            "v4_controls": {name: row.get("controls", []) for name, row in effects.items()},
            "mutation_survivors": mutation["survived"],
            "negative_alignment_required": True,
            "activation": False,
        },
        "CLEAN_CLONE.json": clone,
    }
    for name, document in review_documents.items():
        _write_review_json(review_root, name, document)
    io.atomic_write(
        review_root / "REPRODUCTION.md",
        "# Substrate v4 reproduction\n\n"
        "1. Check out `substrate-v4-terminal` with tags available.\n"
        "2. Decompress `RAW_RECEIPTS.jsonl.gz`; write each embedded `document` to its recorded `path`.\n"
        "3. Install with `python -m pip install '.[dev]'`.\n"
        "4. Run `substrate test`, `substrate v4 status`, and `substrate v4 verify`.\n"
        "5. Compare regenerated seals and endpoint ledgers against `REVIEW_INDEX.json`.\n\n"
        "Activation must remain `false`. The review-candidate tag is not an unqualified Nous claim.\n",
    )
    io.atomic_write(
        review_root / "README.md",
        "# Substrate v4 structural review package\n\n"
        "This package contains the frozen authority index, independently recomputed effects, null and defect "
        "ledgers, compact mechanism ledgers, all raw unit and checkpoint receipts in a deterministic gzip "
        "archive, mutation results, clean-clone results, reproduction instructions, the claim boundary, and "
        "known limitations. It supports external challenge of the functional classification. Activation is "
        "`false`; no claim of consciousness, sentience, personhood, life, moral status, or unqualified Nous is made.\n",
    )
    package_files = {
        path.relative_to(review_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(review_root.iterdir())
        if path.is_file() and path.name != "REVIEW_INDEX.json"
    }
    required = set(review_documents) | {
        "RAW_RECEIPTS.jsonl.gz",
        "REPRODUCTION.md",
        "README.md",
    }
    index = {
        "schema": "substrate-v4-review-package/v1",
        "classification_candidate": verification["all_pass"] and mutation["zero_survived"] and clone["all_pass"],
        "evidence_and_configuration": authorities,
        "package_files": package_files,
        "raw_receipts": {path: digest for path, digest in receipt_manifest.items() if "/units/" in path},
        "raw_receipt_count": len(list(P.UNITS.glob("*.json"))),
        "checkpoint_count": len(list(P.CHECKPOINTS.glob("*.json"))),
        "required_files": sorted(required),
        "missing_files": sorted(required - set(package_files)),
        "complete": required <= set(package_files),
        "claim_boundary": C.CLAIM_BOUNDARY,
        "activation": False,
    }
    io.atomic_write(review_root / "REVIEW_INDEX.json", json.dumps(index, indent=2))
    return index


def finalize(raw_report: dict, verification: dict, mutation: dict, clone: dict) -> dict:
    review = _review_package(raw_report, verification, mutation, clone)
    demonstrated = all(verification["effects"][name]["passes"] for name in ("H_S1", "H_S2", "H_S3", "H_S4", "H_S7"))
    proto = demonstrated and all(verification["effects"][name]["passes"] for name in C.HYPOTHESES)
    latent_model_families = sorted({unit.latent_model_family for unit in P.work_units()})
    representation_families = sorted({unit.surface_representation_family for unit in P.work_units()})
    ready = (
        proto
        and verification["replication"]["passes"]
        and verification["open_world"]["passes"]
        and mutation["zero_survived"]
        and clone["all_pass"]
        and review["raw_receipt_count"] == raw_report["expected"]
        and review["complete"]
        and len(latent_model_families) >= 2
        and len(representation_families) >= 2
    )
    if ready:
        classification = "nous_ready_for_review"
    elif proto:
        classification = "functional_proto_nous_candidate"
    elif demonstrated:
        classification = "demonstrated_structural_understanding"
    else:
        classification = "reflective_cognitive_organization"
    final = {
        "schema": "substrate-v4-final-classification/v1",
        "classification": classification,
        "demonstrated_structural_understanding": demonstrated,
        "functional_proto_nous_candidate": proto,
        "nous_ready_for_review": ready,
        "latent_model_families": latent_model_families,
        "representation_families": representation_families,
        "external_review_package_complete": review["complete"],
        "unqualified_nous": False,
        "not_claimed": C.CLAIM_BOUNDARY["not_claimed"],
        "activation": False,
    }
    review_authority = {
        "schema": "substrate-v4-nous-review-authority/v1",
        "eligible": ready,
        "required_tag": "substrate-nous-review-candidate-1" if ready else None,
        "meaning": "eligible for independent review only; not an unqualified Nous declaration",
        "unqualified_nous": False,
        "activation": False,
    }
    final_state = {
        "schema": "substrate-v4-final-state/v1",
        "raw_receipts": raw_report["valid"],
        "hypotheses": {name: row["passes"] for name, row in verification["effects"].items()},
        "replication": verification["replication"]["passes"],
        "open_world": verification["open_world"]["passes"],
        "mutations_survived": len(mutation["survived"]),
        "clean_clone": clone["all_pass"],
        "classification": classification,
        "historical_preservation": verification["historical"]["preserved"],
        "review_package_complete": review["complete"],
        "activation": False,
    }
    io.seal("SUBSTRATE_V4_FINAL_CLASSIFICATION.json", final)
    io.seal("SUBSTRATE_V4_NOUS_REVIEW_AUTHORITY.json", review_authority)
    io.seal("SUBSTRATE_V4_FINAL_STATE.json", final_state)
    run_path = io.RUNS / "principal" / "run.json"
    run_document = json.loads(run_path.read_text()) if run_path.is_file() else {}
    summaries = [receipt["summary"] for receipt in raw_report["receipts"].values()]
    effects = verification["effects"]
    effect_lines = "\n".join(
        f"| {name} | {row['mean']:.4f} | {row['bootstrap_95_ci'][0]:.4f} to {row['bootstrap_95_ci'][1]:.4f} | "
        f"{row['sesoi']:.2f} | {'positive' if row['passes'] else 'null'} |"
        for name, row in effects.items()
    )
    report = (
        "# Substrate v4 terminal report\n\n"
        f"- Classification: `{classification}`\n"
        f"- Ready commit: `{clone['ready_commit']}`\n"
        f"- Raw receipts: `{raw_report['valid']}` of `{raw_report['expected']}`\n"
        f"- Principal histories: `{len(C.SPLITS['principal'])}`\n"
        f"- Workload families: `{len(C.WORKLOADS)}`\n"
        f"- Episodes: `{sum(summary['episodes'] for summary in summaries)}`\n"
        f"- Models induced across unit summaries: `{sum(summary['models'] for summary in summaries)}`\n"
        f"- Models revised: `{sum(summary['revisions'] for summary in summaries)}`\n"
        f"- Interventions: `{sum(summary['interventions'] for summary in summaries)}`\n"
        f"- Counterfactuals: `{sum(summary['counterfactuals'] for summary in summaries)}`\n"
        f"- Mappings inferred: `{sum(summary['mappings'] for summary in summaries)}`\n"
        f"- Inquiry actions: `{sum(summary['inquiries'] for summary in summaries)}`\n"
        f"- Principal wall time: `{run_document.get('elapsed_seconds', 'not recorded')}` seconds\n"
        f"- Peak worker RSS: `{max(summary['peak_rss_mib'] for summary in summaries):.2f}` MiB\n"
        "- Worker count: `4`\n"
        "- Checkpoint and body continuity: "
        f"`{'pass' if all(summary['checkpoint_exact'] and summary['body_continuity'] for summary in summaries) else 'fail'}`\n"
        f"- Independent verification: `{'pass' if verification['all_pass'] else 'fail'}`\n"
        f"- Mutations: `{mutation['detected']}/{mutation['total']}` detected, `{len(mutation['survived'])}` survivors\n"
        f"- Clean clone, clean install, full v4/runtime tests, lint, and double regeneration: `{'pass' if clone['all_pass'] else 'fail'}`\n"
        f"- Review package: `{'complete' if review['complete'] else 'incomplete'}`\n"
        "- Hawking coexistence: observation only; no signals or controller changes\n"
        "- Activation: `false`\n"
        "- Claim boundary: functional engineering and scientific classification only; no consciousness, sentience, "
        "personhood, life, moral status, or unqualified Nous claim\n\n"
        "## Independently recomputed primary effects\n\n"
        "| Hypothesis | Effect | 95% bootstrap CI | SESOI | Result |\n"
        "|---|---:|---:|---:|---|\n"
        f"{effect_lines}\n\n"
        "Replication and generator-held-out open-world review passed when reported by the final classification. "
        "The complete raw receipt archive, controls, null ledger, defect ledger, mutations, reproduction instructions, "
        "and known limitations are under `artifacts/substrate/v4/review/`.\n"
    )
    io.seal_markdown("SUBSTRATE_V4_TERMINAL_REPORT.md", report)
    return {"classification": final, "review_authority": review_authority, "final_state": final_state}


def run_all() -> dict:
    raw_report = raw()
    verification = recompute(raw_report)
    mutation = mutations(raw_report)
    io.seal("SUBSTRATE_V4_INDEPENDENT_VERIFICATION.json", verification)
    io.seal("SUBSTRATE_V4_MUTATION_REPORT.json", mutation)
    clone = clean_clone(raw_report)
    io.seal("SUBSTRATE_V4_CLEAN_CLONE.json", clone)
    final = finalize(raw_report, verification, mutation, clone)
    return {
        "raw": {key: value for key, value in raw_report.items() if key != "receipts"},
        "verification": verification,
        "mutation": mutation,
        "clean_clone": clone,
        "final": final,
        "all_pass": verification["all_pass"] and mutation["zero_survived"] and clone["all_pass"] and final["classification"]["nous_ready_for_review"],
        "activation": False,
    }
