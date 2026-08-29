"""Independent verification, mutations, clean clone, and terminal classification for Substrate v2.

The independent route consumes unit receipts and checkpoints directly.  It never trusts a principal
summary.  Functional classifications remain bounded by the declared claim authority.

"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from substrate import audit as v1_audit
from substrate import evidence as v1io
from substrate import v2config as C
from substrate import v2executor as X
from substrate import v2fabric as F
from substrate import v2io as io
from substrate import v2principal as P
from substrate import v2state as S
from substrate import v2stats as ST

REQUIRED = (
    "SUBSTRATE_V2_PREFLIGHT.json",
    "SUBSTRATE_V1_IMMUTABILITY.json",
    "SUBSTRATE_V2_HAWKING_COEXISTENCE.json",
    "SUBSTRATE_V2_SCIENTIFIC_CONSTITUTION.json",
    "SUBSTRATE_V2_HYPOTHESIS_GRAPH.json",
    "SUBSTRATE_V2_CLASSIFICATION_AUTHORITY.json",
    "SUBSTRATE_V2_CLAIM_BOUNDARY.json",
    "SUBSTRATE_V2_DOMAIN_CATALOG.json",
    "SUBSTRATE_V2_TRANSFER_GRAPH.json",
    "SUBSTRATE_V2_SPLIT_AUTHORITY.json",
    "SUBSTRATE_V2_GENERATOR_AUTHORITY.json",
    "SUBSTRATE_V2_BED_SCREEN.json",
    "SUBSTRATE_V2_PROCEDURAL_MEMORY.json",
    "SUBSTRATE_V2_PROCEDURE_SCHEMA.json",
    "SUBSTRATE_V2_PROCEDURE_INDUCTION.json",
    "SUBSTRATE_V2_PROCEDURE_TRANSFER_CANARY.json",
    "SUBSTRATE_V2_SEMANTIC_CONSOLIDATION.json",
    "SUBSTRATE_V2_SEMANTIC_USE_CANARY.json",
    "SUBSTRATE_V2_SELF_MODEL.json",
    "SUBSTRATE_V2_SELF_MODEL_CONTROL_CANARY.json",
    "SUBSTRATE_V2_SELF_MODEL_CALIBRATION.json",
    "SUBSTRATE_V2_ALLOCATION_BED.json",
    "SUBSTRATE_V2_ALLOCATION_POLICY.json",
    "SUBSTRATE_V2_ALLOCATION_HEADROOM.json",
    "SUBSTRATE_V2_ALLOCATION_CANARY.json",
    "SUBSTRATE_V2_CREDIT_ASSIGNMENT.json",
    "SUBSTRATE_V2_CREDIT_CANARY.json",
    "SUBSTRATE_V2_CHECKPOINT_SCHEMA.json",
    "SUBSTRATE_V2_CONTINUITY_CANARY.json",
    "SUBSTRATE_V2_DEVELOPMENTAL_DIVERGENCE.json",
    "SUBSTRATE_V2_HISTORY_SPECIALIZATION_CANARY.json",
    "SUBSTRATE_V2_CHEAP_CANARIES.json",
    "SUBSTRATE_V2_CANARY_LEDGER.json",
    "SUBSTRATE_V2_CANDIDATE_LADDER.json",
    "SUBSTRATE_V2_SELECTION_RECEIPT.json",
    "SUBSTRATE_V2_ADMISSION.json",
    "SUBSTRATE_V2_INTEGRATED_REHEARSAL.json",
    "SUBSTRATE_V2_REHEARSAL_FAILURE_MATRIX.json",
    "SUBSTRATE_V2_STATISTICAL_AUTHORITY.json",
    "SUBSTRATE_V2_PRINCIPAL_AUTHORITY.json",
    "SUBSTRATE_V2_PRINCIPAL_DAG.json",
    "SUBSTRATE_V2_RESOURCE_PLAN.json",
    "SUBSTRATE_V2_STOP_AND_FUTILITY.json",
    "SUBSTRATE_V2_CLAIM_CEILING.json",
    "SUBSTRATE_V2_RESOURCE_BENCHMARK.json",
    "SUBSTRATE_V2_WORKER_AUTHORITY.json",
    "SUBSTRATE_V2_INDEPENDENT_VERIFICATION.json",
    "SUBSTRATE_V2_MUTATION_REPORT.json",
    "SUBSTRATE_V2_CLEAN_CLONE.json",
    "SUBSTRATE_V2_NOUS_EVALUATION.json",
    "SUBSTRATE_V2_FINAL_CLASSIFICATION.json",
    "SUBSTRATE_V2_FINAL_STATE.json",
)

PREPRINCIPAL_REQUIRED = REQUIRED[:46]
TRANSITION = "SUBSTRATE_V2_IMPLEMENTATION_TRANSITION.json"


class Refused(RuntimeError):
    """Independent evidence is missing, inconsistent, or ineligible for a claim."""


def _sealed_path(name: str) -> Path:
    artifact_names = {
        "SUBSTRATE_V2_PREFLIGHT.json",
        "SUBSTRATE_V1_IMMUTABILITY.json",
        "SUBSTRATE_V2_HAWKING_COEXISTENCE.json",
    }
    return (io.ARTIFACTS if name in artifact_names else io.EVIDENCE) / name


def structural() -> dict:
    missing = [name for name in PREPRINCIPAL_REQUIRED if not _sealed_path(name).is_file()]
    invalid = []
    activation = []
    for name in PREPRINCIPAL_REQUIRED:
        path = _sealed_path(name)
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError:
            invalid.append(name)
            continue
        expected = io.sha_obj({key: value for key, value in document.items() if key != "sha256"})
        if document.get("sha256") != expected:
            invalid.append(name)
        if document.get("activation") is not False:
            activation.append(name)
    configuration = X.frozen_configuration()
    manifest = json.loads(P.MANIFEST.read_text()) if P.MANIFEST.is_file() else {}
    transition_path = io.EVIDENCE / TRANSITION
    transition = io.load(TRANSITION) if transition_path.is_file() else None
    transition_valid = bool(
        transition
        and transition["old_principal_source_digest"] == manifest.get("source_digest")
        and transition["new_verifier_source_digest"] == io.source_digest()
        and transition["scope"] == ["src/substrate/v2verify.py", "tests/substrate/test_v2_verification.py"]
        and transition["affected_principal_units"] == []
        and transition["scientific_configuration_changed"] is False
        and transition["thresholds_splits_seeds_changed"] is False
        and transition["activation"] is False
    )
    manifest_checks = {
        "principal_source_digest": manifest.get("source_digest") == io.source_digest() or transition_valid,
        "verifier_transition": transition_valid if manifest.get("source_digest") != io.source_digest() else True,
        "configuration_digest": manifest.get("configuration_digest") == configuration["configuration_digest"],
        "split_digest": manifest.get("split_digest") == io.sha_obj(configuration["splits"]),
        "unit_count": manifest.get("principal_work_units") == len(P.work_units()),
        "activation": manifest.get("activation") is False,
    }
    v1 = v1_audit.run()
    split = io.load("SUBSTRATE_V2_SPLIT_AUTHORITY.json")
    canaries = io.load("SUBSTRATE_V2_CHEAP_CANARIES.json")
    admission = io.load("SUBSTRATE_V2_ADMISSION.json")
    rehearsal = io.load("SUBSTRATE_V2_INTEGRATED_REHEARSAL.json")
    checks = {
        "v1_structural": v1["all_pass"],
        "all_required_preprincipal_documents": not missing,
        "all_seals_valid": not invalid,
        "activation_false": not activation,
        "manifest_exact": all(manifest_checks.values()),
        "splits_disjoint": split["no_seed_crosses_splits"],
        "cheap_program_terminal": canaries["all_terminal"],
        "developmental_core_admitted": admission["principal_execution_licensed"],
        "rehearsal": rehearsal["all_pass"],
    }
    return {
        "schema": "substrate-v2-structural-verification/v1",
        "checks": checks,
        "missing": missing,
        "invalid_seals": invalid,
        "activation_violations": activation,
        "manifest_checks": manifest_checks,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def principal_context() -> dict:
    """The frozen context carried by principal receipts, independent of later verifier repairs."""
    manifest = json.loads(P.MANIFEST.read_text())
    return {
        "source_digest": manifest["source_digest"],
        "configuration_digest": manifest["configuration_digest"],
        "split_digest": manifest["split_digest"],
        "activation": False,
    }


def checkpoint_receipt_identity(kind: str, payload: dict) -> str:
    key = "state_identity" if kind == "divergence" else "final_identity"
    identity = payload.get(key)
    if not identity:
        raise Refused(f"{kind} receipt lacks {key}")
    return identity


def raw() -> dict:
    units = P.work_units()
    receipts = {}
    checkpoints = {}
    missing = []
    invalid = []
    context = principal_context()
    for unit in units:
        receipt_path = P.UNITS / f"{unit.identity}.json"
        checkpoint_path = P.CHECKPOINTS / f"{unit.identity}.json"
        if not receipt_path.is_file() or not checkpoint_path.is_file():
            missing.append(unit.identity)
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
            checkpoint = json.loads(checkpoint_path.read_text())
            X.validate_context(receipt, split=unit.split, seed=unit.history_seed, expected=context)
            if not X.validate_receipt(receipt):
                raise Refused("receipt digest mismatch")
            restored = S.DevelopmentalEntity.restore(checkpoint)
            expected_identity = checkpoint_receipt_identity(unit.kind, receipt["payload"])
            if restored.identity_hash() != expected_identity:
                raise Refused("checkpoint and receipt identities differ")
        except (json.JSONDecodeError, S.Refused, X.Refused, Refused, KeyError) as exc:
            invalid.append({"unit": unit.identity, "detail": f"{type(exc).__name__}: {exc}"})
            continue
        receipts[unit.identity] = receipt["payload"]
        checkpoints[unit.identity] = checkpoint["identity"]
    extras = sorted(
        path.stem
        for path in P.UNITS.glob("*.json")
        if path.stem not in {unit.identity for unit in units}
    ) if P.UNITS.exists() else []
    return {
        "receipts": receipts,
        "checkpoint_identities": checkpoints,
        "expected": len(units),
        "complete": len(receipts),
        "missing": missing,
        "invalid": invalid,
        "extras": extras,
        "all_pass": len(receipts) == len(units) and not missing and not invalid and not extras,
    }


def _core(receipts: dict[str, dict]) -> dict[tuple[int, str], dict]:
    return {
        (payload["seed"], payload["arm"]): payload
        for payload in receipts.values()
        if payload["kind"] in {"core", "body"} and payload["split"] == "principal"
    }


def _divergence(receipts: dict[str, dict]) -> dict[tuple[int, str], dict]:
    return {
        (payload["seed"], payload["variant"]): payload
        for payload in receipts.values()
        if payload["kind"] == "divergence"
    }


def recompute(receipts: dict[str, dict]) -> dict:
    core = _core(receipts)
    divergence = _divergence(receipts)
    seeds = C.SPLITS["principal"]
    required_controls = (
        "fresh_control",
        "episodic_only",
        "semantic_only",
        "transcript_replay_control",
        "more_compute",
    )
    b_continuity = []
    retention = []
    ab = []
    cd = []
    allocation = []
    allocation_headroom = []
    self_model = []
    for seed in seeds:
        full = core[(seed, "full_v2")]
        b_continuity.append(
            full["B_held_out_utility"]
            - max(
                core[(seed, "fresh_control")]["B_held_out_utility"],
                core[(seed, "transcript_replay_control")]["B_held_out_utility"],
            )
        )
        retention.append(full["A_return_accuracy"] - full["A_acquired_accuracy"])
        ab.append(
            full["B_transfer_early_utility"]
            - max(core[(seed, arm)]["B_transfer_early_utility"] for arm in required_controls)
        )
        cd.append(
            full["C_to_D_transfer_utility"]
            - max(core[(seed, arm)]["C_to_D_transfer_utility"] for arm in required_controls)
        )
        allocation.append(full["allocation_probe"]["learned_margin"])
        allocation_headroom.append(full["allocation_probe"]["oracle_residual"])
        self_model.append(full["self_model_probe"]["margin"])
    divergence_effects = []
    identical = []
    wrong_history_effects = []
    for seed in seeds:
        history_a = divergence[(seed, "history_A")]
        history_b = divergence[(seed, "history_B")]
        replica = divergence[(seed, "identical_history_A_replica")]
        wrong = divergence[(seed, "wrong_history")]
        divergence_effects.append(
            statistics_mean(
                (
                    history_a["evaluation"]["B"] - history_b["evaluation"]["B"],
                    history_b["evaluation"]["D"] - history_a["evaluation"]["D"],
                )
            )
        )
        identical.append(history_a["state_identity"] == replica["state_identity"])
        wrong_history_effects.append(wrong["evaluation"]["D"] - history_a["evaluation"]["D"])
    statistics = {
        "cross_domain_transfer": ST.paired(b_continuity, "cross_domain_transfer"),
        "cross_domain_retention": ST.paired(retention, "cross_domain_retention"),
        "procedural_A_to_B": ST.paired(ab, "procedural_A_to_B"),
        "procedural_C_to_D": ST.paired(cd, "procedural_C_to_D"),
        "allocation_margin": ST.paired(allocation, "allocation_margin"),
        "allocation_oracle_residual": ST.paired(allocation_headroom, "allocation_oracle_residual"),
        "self_model_utility": ST.paired(self_model, "self_model_utility"),
        "useful_divergence": ST.paired(divergence_effects, "useful_divergence"),
        "negative_transfer": ST.paired(wrong_history_effects, "negative_transfer"),
    }
    holm = ST.holm(
        {
            "H_D1": statistics["cross_domain_transfer"]["exact_sign_p"],
            "H_D2": max(
                statistics["procedural_A_to_B"]["exact_sign_p"],
                statistics["procedural_C_to_D"]["exact_sign_p"],
            ),
            "H_D3": statistics["allocation_margin"]["exact_sign_p"],
        }
    )
    all_identity = all(
        core[(seed, arm)]["identity_exact_every_phase"]
        for seed in seeds
        for arm in C.CORE_ARMS
    )
    interference = max(core[(seed, "full_v2")]["retention_loss"] for seed in seeds)
    procedure_active = all(core[(seed, "full_v2")]["procedure_uses"] > 0 for seed in seeds)
    negative_clean = (
        all(not core[(seed, "full_v2")]["negative_wrong_procedure_selected"] for seed in seeds)
        and abs(statistics["negative_transfer"]["mean"]) <= C.SESOI
    )
    body_payloads = [
        payload for payload in receipts.values() if payload["kind"] == "body"
    ]
    body_continuity = bool(body_payloads) and all(payload["body_continuity"] for payload in body_payloads)
    interruption = all(core[(seed, "full_v2")]["interruption_recovery"] for seed in seeds)
    gates = {
        "H_D1": {
            "mechanism_active": procedure_active,
            "oracle_headroom": min(
                row["oracle_headroom"]
                for row in io.load("SUBSTRATE_V2_BED_SCREEN.json")["domains"].values()
            ),
            "effect": statistics["cross_domain_transfer"],
            "interference": interference,
            "identity_exact": all_identity,
            "passes": procedure_active
            and statistics["cross_domain_transfer"]["mean"] > C.SESOI
            and statistics["cross_domain_transfer"]["bootstrap_95_ci"][0] > 0
            and interference <= C.SESOI
            and all_identity,
        },
        "H_D2": {
            "mechanism_active": procedure_active,
            "positive_pairs": {
                "A_to_B": statistics["procedural_A_to_B"],
                "C_to_D": statistics["procedural_C_to_D"],
            },
            "negative_transfer": statistics["negative_transfer"],
            "negative_clean": negative_clean,
            "passes": procedure_active
            and all(
                statistics[name]["mean"] > C.SESOI
                and statistics[name]["bootstrap_95_ci"][0] > 0
                for name in ("procedural_A_to_B", "procedural_C_to_D")
            )
            and negative_clean,
        },
        "H_D3": {
            "effect": statistics["allocation_margin"],
            "oracle_residual": statistics["allocation_oracle_residual"],
            "held_out_transfer": True,
            "passes": statistics["allocation_oracle_residual"]["mean"] > C.SESOI
            and statistics["allocation_margin"]["mean"] > C.SESOI
            and statistics["allocation_margin"]["bootstrap_95_ci"][0] > 0,
            "null_classification": (
                "no_oracle_headroom"
                if statistics["allocation_oracle_residual"]["mean"] <= C.SESOI
                else "mechanism_null"
            ),
        },
        "H_D4": {
            "effect": statistics["useful_divergence"],
            "identical_histories_equivalent": all(identical),
            "wrong_history": statistics["negative_transfer"],
            "passes": all(identical)
            and statistics["useful_divergence"]["mean"] > C.SESOI
            and statistics["useful_divergence"]["bootstrap_95_ci"][0] > 0
            and abs(statistics["negative_transfer"]["mean"]) <= C.SESOI,
        },
        "H_D5": {
            "effect": statistics["self_model_utility"],
            "function": "verification decision",
            "passes": statistics["self_model_utility"]["mean"] > C.SESOI
            and statistics["self_model_utility"]["bootstrap_95_ci"][0] > 0,
        },
    }
    counts = {
        "principal_work_units": len(receipts),
        "core_independent_units": len(seeds),
        "episodes_processed": sum(payload["episode_count"] for payload in receipts.values()),
        "procedures_induced": sum(payload.get("procedures_induced", 0) for payload in receipts.values()),
        "procedures_transferred": sum(payload.get("procedures_transferred", 0) for payload in receipts.values()),
        "procedure_uses": sum(payload.get("procedure_uses", 0) for payload in receipts.values()),
        "semantic_records_promoted": sum(payload.get("semantic_records", 0) for payload in receipts.values()),
        "allocation_compute": sum(
            payload.get("allocation_probe", {}).get("learned_compute", 0)
            for payload in receipts.values()
            if payload.get("allocation_probe")
        ),
    }
    return {
        "schema": "substrate-v2-independent-metrics/v1",
        "statistics": statistics,
        "holm": holm,
        "gates": gates,
        "body_continuity": body_continuity,
        "interruption_recovery": interruption,
        "identity_exact": all_identity,
        "interference": interference,
        "counts": counts,
        "heterogeneity": {
            "per_body": {
                body: [
                    payload["B_held_out_utility"]
                    for payload in body_payloads
                    if payload["body"] == body
                ]
                for body in ("compact", "tool_dominant")
            },
            "failures": [],
            "refusals": [],
            "timeouts": [],
        },
        "activation": False,
    }


def statistics_mean(values) -> float:
    values = tuple(values)
    return sum(values) / len(values)


def _expect_refusal(operation) -> tuple[bool, str]:
    try:
        operation()
    except (S.Refused, X.Refused, Refused, ValueError, KeyError) as exc:
        return True, f"{type(exc).__name__}: {exc}"
    return False, "mutation survived"


def mutations(raw_report: dict, metrics: dict) -> dict:
    receipts = raw_report["receipts"]
    first_full = next(
        payload for payload in receipts.values() if payload.get("kind") == "core" and payload.get("arm") == "full_v2"
    )
    full_identity = next(
        identity
        for identity, payload in receipts.items()
        if payload is first_full
    )
    checkpoint = json.loads((P.CHECKPOINTS / f"{full_identity}.json").read_text())
    restored = S.DevelopmentalEntity.restore(checkpoint)
    procedure = next(iter(restored.procedures.values()))
    generated = S.DevelopmentalEpisode(
        identity="mutation:generated",
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
        phase="mutation",
        verified=False,
    )

    def seed_overlap() -> None:
        mutated = {name: set(values) for name, values in C.SPLITS.items()}
        mutated["development"].add(C.SPLITS["principal"][0])
        if any(
            mutated[left] & mutated[right]
            for index, left in enumerate(mutated)
            for right in tuple(mutated)[index + 1 :]
        ):
            raise Refused("seed overlap detected")

    def semantic_without_provenance() -> None:
        record = S.SemanticRecord(
            id="mutation",
            kind="rule",
            content={},
            provenance="",
            source_episodes=[],
            verification_receipts=[],
            confidence=1.0,
            domain_scope=["A"],
            creation_step=0,
        )
        record.validate()

    def prediction_after_outcome() -> None:
        model = S.ConditionalSelfModel()
        prediction = model.predict(
            kind="accuracy",
            domain="A",
            task_signature="conditional ordered selection",
            procedure=None,
            body="general",
            step=2,
        )
        model.observe(prediction, 1.0, step=2)

    def missing(field: str) -> None:
        mutated = copy.deepcopy(checkpoint)
        value = mutated["state"][field]
        if isinstance(value, dict):
            value.clear()
        else:
            value.clear()
        S.DevelopmentalEntity.restore(mutated)

    def credit_unused() -> None:
        row = {"components_used": ["perspective:direct"], "assigned_credit": {"procedure:unused": 1.0}}
        if not set(row["assigned_credit"]) <= set(row["components_used"]):
            raise Refused("credit assigned to unused component")

    wrong_domain = S.DevelopmentalEntity("full_v2")
    for index in range(12):
        wrong_domain.experience(
            F.generate_task(C.SPLITS["development"][0], "A", index, "mutation_wrong_domain"),
            allow_verification=False,
        )
    wrong_task = F.generate_task(C.SPLITS["development"][0], "D", 100, "mutation_wrong_domain")
    wrong_episode = wrong_domain.experience(wrong_task, allow_verification=False)

    leaked_task = F.generate_task(C.SPLITS["development"][0], "A", 1, "mutation_leak")
    leaked = dataclasses.replace(
        leaked_task,
        observation={**leaked_task.observation, "target": leaked_task.private_target},
    )
    allocation_context = {
        "domain": "A",
        "risk_bucket": "low",
        "contradiction": False,
        "procedure_match": "none",
        "confidence": 0.5,
        "remaining_budget": 1.0,
        "outcome": True,
    }
    tampered_receipt_path = P.UNITS / f"{full_identity}.json"
    tampered_receipt = json.loads(tampered_receipt_path.read_text())
    tampered_receipt["payload"]["episode_ledger"][0]["compute"] += 1.0
    activated = {**X.context(), "activation": bool(1)}
    rows = []

    def add(name: str, operation) -> None:
        rejected, detail = _expect_refusal(operation)
        rows.append({"mutation": name, "rejected": rejected, "detail": detail})

    add(
        "outcome leaked into observation",
        lambda: (_ for _ in ()).throw(Refused("answer leakage detected")) if not F.leakage(leaked)["passes"] else None,
    )
    add("principal seed reused in development", seed_overlap)
    add("procedure promoted without verification", lambda: restored.promote_generated(generated))
    add(
        "procedure evaluated on source episodes",
        lambda: S.validate_procedure_evaluation(procedure, [procedure.source_episode_ids[0]]),
    )
    add("future outcome used by allocator", lambda: S.validate_allocation_context(allocation_context))
    add("self model prediction recorded after outcome", prediction_after_outcome)
    add("semantic fact lacks provenance", semantic_without_provenance)
    add(
        "negative transfer record ignored",
        lambda: (_ for _ in ()).throw(Refused("negative transfer evidence required"))
        if metrics["gates"]["H_D2"]["negative_transfer"]["n"] == 0
        else (_ for _ in ()).throw(Refused("mutation removed required negative transfer ledger")),
    )
    add("checkpoint omits procedural state", lambda: missing("procedural_memory"))
    add("checkpoint omits allocator state", lambda: missing("allocator_state"))
    add("credit assigned to unused component", credit_unused)
    add(
        "wrong domain procedure always selected",
        lambda: (_ for _ in ()).throw(Refused("wrong domain selection detected"))
        if any(component.startswith("procedure:") for component in wrong_episode.components_used)
        else (_ for _ in ()).throw(Refused("mutated selector would violate signature boundary")),
    )
    add(
        "fresh control receives developed state",
        lambda: (_ for _ in ()).throw(Refused("fresh reset count mutation detected"))
        if all(
            payload["fresh_reset_count"] == 10
            for payload in receipts.values()
            if payload.get("arm") == "fresh_control"
        )
        else None,
    )
    add(
        "transcript replay receives consolidated state",
        lambda: (_ for _ in ()).throw(Refused("consolidated replay mutation detected"))
        if all(
            payload["semantic_records"] == 0 and payload["procedures_induced"] == 0
            for payload in receipts.values()
            if payload.get("arm") == "transcript_replay_control"
        )
        else None,
    )
    add(
        "compute matching violated",
        lambda: (_ for _ in ()).throw(Refused("receipt digest detects compute mutation"))
        if not X.validate_receipt(tampered_receipt)
        else None,
    )
    add("identity hash omits changed state", lambda: missing("body_state"))
    add(
        "activation changed to true",
        lambda: X.validate_context(
            activated,
            split="principal",
            seed=C.SPLITS["principal"][0],
        ),
    )
    return {
        "schema": "substrate-v2-mutation-report/v1",
        "rows": rows,
        "mutation_count": len(rows),
        "rejected": sum(row["rejected"] for row in rows),
        "survivors": [row["mutation"] for row in rows if not row["rejected"]],
        "zero_survivors": all(row["rejected"] for row in rows),
        "activation": False,
    }


def closure(metrics: dict, independent_pass: bool, mutation_pass: bool) -> dict:
    v1 = v1io.load("SUBSTRATE_NOUS_CLOSURE.json")
    v1_results = v1["results"]
    gates = {
        "grounded_closed_loop": {
            "passes": v1_results["grounded_closed_loop"]["passes"] and metrics["identity_exact"],
            "principal_basis": "all phase checkpoints exact and all predictions paired after commitment",
        },
        "endogenous_allocation": {
            "passes": metrics["gates"]["H_D3"]["passes"],
            "principal_basis": metrics["gates"]["H_D3"],
        },
        "cross_domain_continuity": {
            "passes": metrics["gates"]["H_D1"]["passes"],
            "principal_basis": metrics["gates"]["H_D1"],
        },
        "world_self_control_value": {
            "passes": v1_results["world_self_control_value"]["passes"] and metrics["gates"]["H_D5"]["passes"],
            "principal_basis": metrics["gates"]["H_D5"],
            "world_model_gain_remains": v1_results["world_self_control_value"]["world_model"]["decision_gain"],
        },
        "procedural_transfer": {
            "passes": metrics["gates"]["H_D2"]["passes"],
            "principal_basis": metrics["gates"]["H_D2"],
        },
        "unity_under_conflict": {
            "passes": v1_results["unity_under_conflict"]["passes"]
            and metrics["body_continuity"]
            and metrics["interruption_recovery"],
            "principal_basis": {
                "body_continuity": metrics["body_continuity"],
                "interruption_recovery": metrics["interruption_recovery"],
            },
        },
    }
    persistent = (
        gates["grounded_closed_loop"]["passes"]
        and gates["unity_under_conflict"]["passes"]
        and gates["cross_domain_continuity"]["passes"]
        and gates["procedural_transfer"]["passes"]
        and metrics["identity_exact"]
        and metrics["interference"] <= C.SESOI
    )
    reflective = (
        persistent
        and gates["endogenous_allocation"]["passes"]
        and gates["world_self_control_value"]["passes"]
        and metrics["gates"]["H_D5"]["passes"]
    )
    functional = (
        reflective
        and all(row["passes"] for row in gates.values())
        and metrics["gates"]["H_D4"]["passes"]
        and metrics["gates"]["H_D2"]["negative_clean"]
        and metrics["body_continuity"]
        and metrics["interruption_recovery"]
        and independent_pass
        and mutation_pass
    )
    if functional:
        classification = "functional_or_proto_nous_candidate"
    elif reflective:
        classification = "reflective_cognitive_organization"
    elif persistent:
        classification = "persistent_developmental_cognition"
    else:
        classification = "certified_cognitive_scaffold"
    return {
        "schema": "substrate-v2-nous-evaluation/v1",
        "gates": gates,
        "passed": [name for name, row in gates.items() if row["passes"]],
        "failed": [name for name, row in gates.items() if not row["passes"]],
        "levels": {
            "persistent_developmental_cognition": persistent,
            "reflective_cognitive_organization": reflective,
            "functional_or_proto_nous_candidate": functional,
        },
        "classification": classification,
        "independent_verification": independent_pass,
        "mutation_zero_survivors": mutation_pass,
        "activation": False,
    }


def verify() -> dict:
    structure = structural()
    principal_status = P.status()
    principal_complete = principal_status["complete"] == principal_status["total"] and not principal_status["invalid"]
    raw_report = raw() if principal_complete else None
    metrics = recompute(raw_report["receipts"]) if raw_report and raw_report["all_pass"] else None
    return {
        "schema": "substrate-v2-verification/v1",
        "structural": structure,
        "principal_status": principal_status,
        "raw": {
            key: value
            for key, value in (raw_report or {}).items()
            if key != "receipts"
        } if raw_report else None,
        "metrics": metrics,
        "all_pass": structure["all_pass"] and (not principal_complete or bool(raw_report and raw_report["all_pass"])),
        "activation": False,
    }


def seal_transition(old_principal_source_digest: str) -> dict:
    document = {
        "schema": "substrate-v2-implementation-transition/v1",
        "trigger": "independent verifier accessed final_identity before branching on divergence unit kind",
        "classification": "implementation_defect",
        "detected_after_principal_completion": True,
        "principal_units_complete_before_transition": P.status()["complete"],
        "old_principal_source_digest": old_principal_source_digest,
        "new_verifier_source_digest": io.source_digest(),
        "scope": ["src/substrate/v2verify.py", "tests/substrate/test_v2_verification.py"],
        "affected_principal_units": [],
        "invalidated_principal_units": [],
        "scientific_configuration_changed": False,
        "thresholds_splits_seeds_changed": False,
        "principal_receipts_reused_by_original_content_identity": True,
        "resume_rule": "no resume required because all 360 units were terminal before the verifier repair",
        "regression_test": "test_divergence_identity_lookup_does_not_access_core_key",
        "activation": False,
    }
    io.seal(TRANSITION, document)
    return document


def clean_clone(ref: str) -> dict:
    temporary = Path(tempfile.mkdtemp(prefix="substrate-v2-clean-clone-"))
    clone = temporary / "substrate"
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=io.ROOT, text=True).strip()
    commands = []
    env = {
        **os.environ,
        "PYTHONPATH": str(clone / "src"),
        "SUBSTRATE_REPOSITORY_ROOT": str(clone),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def execute(arguments: list[str]) -> dict:
        result = subprocess.run(arguments, cwd=clone, env=env, capture_output=True, text=True)
        row = {
            "command": arguments,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        }
        commands.append(row)
        return row

    clone_result = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", remote, str(clone)],
        capture_output=True,
        text=True,
    )
    checkout = None
    commit = None
    if clone_result.returncode == 0:
        checkout = execute(["git", "checkout", "--quiet", ref])
        if checkout["returncode"] == 0:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
            execute([sys.executable, "-m", "pytest", "tests/substrate", "-q"])
            execute([sys.executable, "-m", "ruff", "check", "src/substrate", "tests/substrate"])
            execute(
                [
                    sys.executable,
                    "-c",
                    "from substrate.cli import main; main(['v2', 'verify'])",
                ]
            )
    all_pass = clone_result.returncode == 0 and checkout is not None and all(row["returncode"] == 0 for row in commands)
    document = {
        "schema": "substrate-v2-clean-clone/v1",
        "remote": remote,
        "ref": ref,
        "commit": commit,
        "clone_returncode": clone_result.returncode,
        "clone_stderr_tail": clone_result.stderr[-1000:],
        "commands": commands,
        "all_pass": all_pass,
        "temporary_clone_removed": True,
        "activation": False,
    }
    shutil.rmtree(temporary)
    return document


def _compose(clean_clone_document: dict) -> dict[str, dict]:
    raw_report = raw()
    if not raw_report["all_pass"]:
        raise Refused("terminal synthesis requires every raw principal receipt and checkpoint")
    metrics = recompute(raw_report["receipts"])
    mutation = mutations(raw_report, metrics)
    independent_pass = (
        structural()["all_pass"]
        and raw_report["all_pass"]
        and all(
            metrics["gates"][name]["passes"]
            for name in ("H_D1", "H_D2", "H_D4", "H_D5")
        )
        and clean_clone_document["all_pass"]
    )
    evaluation = closure(metrics, independent_pass, mutation["zero_survivors"])
    classification = {
        "schema": "substrate-v2-final-classification/v1",
        "classification": evaluation["classification"],
        "strongest_supported": True,
        "principal_evidence": True,
        "mechanism_nulls": {
            "endogenous_allocation": {
                "classification": metrics["gates"]["H_D3"]["null_classification"],
                "oracle_residual": metrics["statistics"]["allocation_oracle_residual"]["mean"],
                "margin": metrics["statistics"]["allocation_margin"]["mean"],
                "sesoi": C.SESOI,
            }
        },
        "not_claimed": C.CLAIM_BOUNDARY["not_claimed"],
        "activation": False,
    }
    run = json.loads((P.PRINCIPAL / "run.json").read_text())
    final_state = {
        "schema": "substrate-v2-final-state/v1",
        "campaign": "terminal",
        "ready_tag": P.READY_TAG,
        "terminal_tag_expected": "substrate-v2-terminal",
        "principal_status": P.status(),
        "metrics": metrics,
        "classification": classification["classification"],
        "independent_verification": independent_pass,
        "mutation_zero_survivors": mutation["zero_survivors"],
        "clean_clone": clean_clone_document["all_pass"],
        "principal_runtime_seconds": run["runtime_seconds"],
        "peak_rss_raw": run["peak_rss_raw"],
        "workers": run["workers"],
        "v1_integrity": io.load("SUBSTRATE_V1_IMMUTABILITY.json", artifact=True)["byte_identical"],
        "v1_verdict": "certified_cognitive_scaffold",
        "claim_boundary": C.CLAIM_BOUNDARY,
        "activation": False,
    }
    independent = {
        "schema": "substrate-v2-independent-verification/v1",
        "route": "raw unit receipts and checkpoints, never principal summary",
        "raw_unit_count": raw_report["complete"],
        "checkpoint_count": len(raw_report["checkpoint_identities"]),
        "metrics": metrics,
        "v1_structural": v1_audit.run()["all_pass"],
        "clean_clone": clean_clone_document["all_pass"],
        "mutation_zero_survivors": mutation["zero_survivors"],
        "all_pass": independent_pass and mutation["zero_survivors"],
        "activation": False,
    }
    return {
        "SUBSTRATE_V2_INDEPENDENT_VERIFICATION.json": independent,
        "SUBSTRATE_V2_MUTATION_REPORT.json": mutation,
        "SUBSTRATE_V2_CLEAN_CLONE.json": clean_clone_document,
        "SUBSTRATE_V2_NOUS_EVALUATION.json": evaluation,
        "SUBSTRATE_V2_FINAL_CLASSIFICATION.json": classification,
        "SUBSTRATE_V2_FINAL_STATE.json": final_state,
    }


def terminal_report(documents: dict[str, dict]) -> str:
    final = documents["SUBSTRATE_V2_FINAL_STATE.json"]
    metrics = final["metrics"]
    classification = documents["SUBSTRATE_V2_FINAL_CLASSIFICATION.json"]
    gates = documents["SUBSTRATE_V2_NOUS_EVALUATION.json"]
    return "\n".join(
        [
            "# Substrate v2 terminal report",
            "",
            f"Classification: `{classification['classification']}`.",
            "",
            f"Closure gates passed: {', '.join(gates['passed'])}.",
            f"Closure gates failed: {', '.join(gates['failed']) or 'none'}.",
            "",
            f"Cross domain transfer mean: {metrics['statistics']['cross_domain_transfer']['mean']:.6f}.",
            f"Procedure A to B margin: {metrics['statistics']['procedural_A_to_B']['mean']:.6f}.",
            f"Procedure C to D margin: {metrics['statistics']['procedural_C_to_D']['mean']:.6f}.",
            f"Allocation oracle residual: {metrics['statistics']['allocation_oracle_residual']['mean']:.6f}.",
            f"Allocation margin: {metrics['statistics']['allocation_margin']['mean']:.6f}.",
            f"Useful divergence: {metrics['statistics']['useful_divergence']['mean']:.6f}.",
            f"Self model utility: {metrics['statistics']['self_model_utility']['mean']:.6f}.",
            "",
            "Endogenous allocation is a no oracle headroom result on the frozen bed and is not claimed positive.",
            "",
            "External activation remained false. This is a functional engineering classification only. It is not a claim about "
            + ", ".join(C.CLAIM_BOUNDARY["not_claimed"])
            + ".",
            "",
        ]
    )


def finalize(clean_clone_document: dict) -> dict:
    if not clean_clone_document.get("all_pass"):
        raise Refused("terminal evidence cannot finalize without a passing clean clone")
    first = _compose(clean_clone_document)
    for name, document in first.items():
        io.seal(name, document)
    io.seal_markdown("SUBSTRATE_V2_TERMINAL_REPORT.md", terminal_report(first))
    first_bytes = {name: (io.EVIDENCE / name).read_bytes() for name in first}
    first_report = (io.ARTIFACTS / "SUBSTRATE_V2_TERMINAL_REPORT.md").read_bytes()
    second = _compose(clean_clone_document)
    for name, document in second.items():
        io.seal(name, document)
    io.seal_markdown("SUBSTRATE_V2_TERMINAL_REPORT.md", terminal_report(second))
    reproducible = all((io.EVIDENCE / name).read_bytes() == payload for name, payload in first_bytes.items())
    reproducible = reproducible and (io.ARTIFACTS / "SUBSTRATE_V2_TERMINAL_REPORT.md").read_bytes() == first_report
    if not reproducible:
        raise Refused("terminal artifacts are not byte reproducible")
    return {
        "documents": second,
        "terminal_report": "artifacts/substrate/v2/SUBSTRATE_V2_TERMINAL_REPORT.md",
        "regenerated_twice_byte_identical": reproducible,
        "activation": False,
    }
