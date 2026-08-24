"""Hash-locked R2-to-Odyssey handoff controller.

The controller is intentionally incapable of selecting an arm, task, answer,
or result.  Its only automatic effect is to emit a durable receipt permitting
the already-frozen Odyssey preflight after a complete independently verified
R2 terminal record appears.  It never starts an Odyssey worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROGRAM = "substrate-odyssey-r2-handoff-v1"
PLAN = Path("plans/substrate/tangible_next_launch")
EVIDENCE = Path("evidence/substrate/tangible_sandbox")
RUNS = Path("runs/substrate/odyssey_transition")
OPERATOR_DECISION = Path("operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json")


class Refused(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} is not an object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def build_inputs(root: Path) -> dict[str, Path]:
    return {
        "hardened_design": root / PLAN / "ODYSSEY_7D.hardened.draft.json",
        "autopivot_policy": root / PLAN / "R2_TO_ODYSSEY_AUTOPIVOT_POLICY.sealed.json",
        "task_bank": root / PLAN / "ODYSSEY_TASK_BANK_MANIFEST.draft.json",
        # This is a governing execution input rather than a convenience note:
        # G06 may admit the full program only after this exact width ladder and
        # repetition count have been exercised.  Freeze it alongside the
        # hardened design so a later calibration cannot silently use the old
        # 1/2/4-only Tangible-Sandbox ladder.
        "resource_calibration": root / PLAN / "RESOURCE_CALIBRATION_SPEC.draft.json",
        "shared_storage": root / PLAN / "ODYSSEY_SHARED_STORAGE_RESERVE.draft.json",
        "frontier_contract": root / PLAN / "ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json",
        "rendered_build_index": root / PLAN / "frontiers/FRONTIER_BUILD_INDEX.json",
        # The source-selection template fixes the later custodian handoff
        # shape.  It is intentionally an unsealed template, never a selected
        # corpus or substitute for G02/G04/G10.
        "source_selection_template": root / PLAN / "ODYSSEY_SOURCE_SELECTION.template.json",
        # A public-only, pre-hidden-seed model screen.  It is not a model-arm
        # selection or gate receipt, but its exact candidate set and tie-break
        # rule must not drift after an operator starts staging model bodies.
        "public_model_canary_template": root / PLAN / "ODYSSEY_PUBLIC_MODEL_CANARY.template.json",
        # Human-selection/custody/analysis gates are never auto-sealed, but
        # their closed fill-only shape is a protocol input and must not drift.
        "human_evidence_pack_template": root / PLAN / "ODYSSEY_HUMAN_EVIDENCE_PACK.template.json",
        # The user-selected scientific policy is a non-attesting input.  It
        # records decisions without standing in for actual models, custodians,
        # rights, isolation observations, or gate evidence.
        "operator_decision": root / OPERATOR_DECISION,
    }


def implementation_inputs(root: Path) -> dict[str, Path]:
    return {
        "transition_controller": root / "src/substrate/odyssey_transition.py",
        "frontier_renderer": root / "src/substrate/odyssey7d.py",
        "task_bank_generator": root / "src/substrate/odyssey_task_bank.py",
        "manifest_materializer": root / "src/substrate/odyssey_manifest_materializer.py",
        "machine_subject_generator": root / "src/substrate/odyssey_machine_subjects.py",
        "rehearsal_runner": root / "src/substrate/odyssey_rehearsal.py",
        "odyssey_worker": root / "src/substrate/odyssey_worker.py",
        "odyssey_arms": root / "src/substrate/odyssey_arms.py",
        "odyssey_authority": root / "src/substrate/odyssey_authority.py",
        "public_model_canary": root / "src/substrate/odyssey_model_canary.py",
        "odyssey_clean_clone": root / "src/substrate/odyssey_clean_clone.py",
        "odyssey_mutations": root / "src/substrate/odyssey_mutations.py",
        "odyssey_detachment": root / "src/substrate/odyssey_detachment.py",
        "telegram_probe": root / "src/substrate/odyssey_telegram_probe.py",
        "telegram_notifier": root / "tools/odyssey7d_telegram_notifier.py",
        "r2_continuity_verifier": root / "src/substrate/r2_continuity_verifier.py",
        "r2_provenance_verifier": root / "src/substrate/r2_provenance_verifier.py",
    }


def _inactive(value: dict[str, Any]) -> bool:
    program = value.get("program")
    if not isinstance(program, dict):
        program = {}
    top_activation = value.get("activation", program.get("activation", False))
    return top_activation is False and value.get("external_activation", False) is False


def _validate_resource_calibration_spec(design: dict[str, Any], calibration: dict[str, Any]) -> None:
    """Keep the static G06 calibration contract aligned with the design.

    The frozen map catches later byte drift.  This semantic check catches the
    more dangerous case where both files are deliberately refreshed but the
    governing calibration ladder no longer admits the declared full program.
    """
    resources = design.get("resources")
    if not isinstance(resources, dict):
        raise Refused("hardened design lacks a resource policy")
    if calibration.get("schema") != "SUBSTRATE_ODYSSEY_RESOURCE_CALIBRATION_SPEC_DRAFT/v1":
        raise Refused("resource calibration has the wrong Odyssey schema")
    if calibration.get("activation") is not False:
        raise Refused("resource calibration must remain inactive")
    expected_widths = resources.get("widths_to_calibrate")
    expected_full_width = resources.get("full_program_requires_width")
    if calibration.get("widths") != expected_widths or expected_widths != [1, 2, 4, 6, 8]:
        raise Refused("resource calibration must retain the exact 1/2/4/6/8 width ladder")
    if calibration.get("full_program_requires_width") != expected_full_width or expected_full_width != 8:
        raise Refused("resource calibration must retain width eight for the full program")
    if calibration.get("repetitions") != resources.get("calibration_repetitions") or resources.get("calibration_repetitions") != 3:
        raise Refused("resource calibration must retain exactly three repetitions per width")
    if calibration.get("unit") != "complete_paired_frontier_cell" or calibration.get("unit_count") != 8:
        raise Refused("resource calibration must measure complete paired frontier cells A-H")
    if calibration.get("full_phase_seconds") != 1800:
        raise Refused("resource calibration must retain the full 30-minute Odyssey phase")
    if calibration.get("strict_dispatch_budget_seconds") != 150 or calibration.get("scale_factor") != 12:
        raise Refused("resource calibration must retain the strict 150-second dispatch budget at a 12x cadence")
    if calibration.get("full_phase_seconds") // calibration.get("scale_factor") != calibration.get("strict_dispatch_budget_seconds"):
        raise Refused("resource calibration phase scale must be integral")
    if calibration.get("phase_boundary_guard_interval_seconds") != 30:
        raise Refused("resource calibration must retain 30-second global-dwell guard checks")
    if calibration.get("paired_adapter_dispatches_per_cell") != 2:
        raise Refused("resource calibration must retain one candidate and one control dispatch per cell")
    if calibration.get("minimum_width_eight_scheduled_seconds") != 450:
        raise Refused("resource calibration must retain three 150-second width-eight scheduled observations")
    if calibration.get("measurement_basis") != "active_paired_dispatch_wall_with_deadline_guard":
        raise Refused("resource calibration must measure active paired-dispatch wall time under the sealed deadline")
    if (
        calibration.get("scheduling_mode")
        != "initial_release_only;per_frontier_candidate_then_control;no_global_role_barrier;parent_global_dwell"
    ):
        raise Refused("resource calibration must retain production paired-dispatch scheduling")
    requirements = calibration.get("requirements")
    required_checks = {
        "distinct_run_roots",
        "no_shared_writable_evaluator_or_data_root",
        "strict_dispatch_deadline",
        "production_paired_adapters",
        "source_bundle_pre_dispatch_revalidation",
        "parent_global_dwell",
        "receipt_invariance",
        "record_cpu_memory_io",
        "record_external_disk_drift",
    }
    if not isinstance(requirements, dict) or any(requirements.get(name) is not True for name in required_checks):
        raise Refused("resource calibration lacks a required G06 measurement invariant")


def _validate_source_selection_template(template: dict[str, Any]) -> None:
    """Ensure the later G03 handoff remains an inactive A--H draft only."""
    required = {"schema", "program", "status", "frontiers", "activation", "external_activation"}
    if set(template) != required:
        raise Refused("source-selection template has undeclared or missing fields")
    if template.get("schema") != "SUBSTRATE_ODYSSEY_SOURCE_SELECTION_DRAFT/v1":
        raise Refused("source-selection template has the wrong schema")
    if template.get("program") != "substrate-odyssey-7d-v1" or template.get("status") != "template_unsealed":
        raise Refused("source-selection template must remain an unsealed Odyssey draft")
    if template.get("activation") is not False or template.get("external_activation") is not False:
        raise Refused("source-selection template must remain inactive")
    rows = template.get("frontiers")
    if not isinstance(rows, list) or [row.get("id") if isinstance(row, dict) else None for row in rows] != list("ABCDEFGH"):
        raise Refused("source-selection template must retain ordered A-H frontier rows")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "assets"} or not isinstance(row["assets"], list) or len(row["assets"]) != 1:
            raise Refused("source-selection template frontier rows must expose one asset placeholder")
        asset = row["assets"][0]
        if not isinstance(asset, dict) or set(asset) != {"path", "sha256", "role", "rights_reference"}:
            raise Refused("source-selection template asset placeholders have the wrong shape")


def _validate_public_model_canary_template(design: dict[str, Any], template: dict[str, Any]) -> None:
    """Keep the technical base-model screen public, bounded, and inert.

    The canary is deliberately narrower than G02: it screens a fixed set of
    locally staged public bodies before any hidden seed or arm decision exists.
    It must remain unable to turn that screen into an experiment authority.
    """
    required = {
        "schema",
        "program",
        "status",
        "activation",
        "external_activation",
        "visibility",
        "neutral_organ_prompt",
        "reasoning_effort_policy",
        "conversation_policy",
        "max_output_tokens",
        "candidate_aliases",
        "model_service_cap_bytes",
        "required_concurrent_clients",
        "selection_rule",
        "hidden_seed_policy",
        "case_set",
    }
    if set(template) != required:
        raise Refused("public model-canary template has undeclared or missing fields")
    if (
        template.get("schema") != "SUBSTRATE_ODYSSEY_PUBLIC_MODEL_CANARY_TEMPLATE/v1"
        or template.get("program") != "substrate-odyssey-7d-v1"
        or template.get("status") != "template_unsealed"
        or template.get("activation") is not False
        or template.get("external_activation") is not False
        or template.get("visibility") != "public_only"
        or not isinstance(template.get("neutral_organ_prompt"), str)
        or not template["neutral_organ_prompt"].strip()
        or template.get("reasoning_effort_policy") != "fixed_default_no_frontier_override"
        or template.get("conversation_policy") != "fresh_request_context_per_case; keep_alive_only_for_weights; no_cross_case_state"
        or template.get("max_output_tokens") != 512
    ):
        raise Refused("public model-canary template must remain an inactive public-only draft")
    resources = design.get("resources")
    expected_cap = resources.get("shared_model_service_cap_gib") if isinstance(resources, dict) else None
    if not isinstance(expected_cap, int) or template.get("model_service_cap_bytes") != expected_cap * 1024**3:
        raise Refused("public model-canary template must retain the hardened shared-model service cap")
    if template.get("required_concurrent_clients") != 8:
        raise Refused("public model-canary template must retain eight concurrent clients")
    if template.get("selection_rule") != "highest_public_canary_score_then_lower_service_peak_then_lower_median_latency_then_lexical_weight_digest":
        raise Refused("public model-canary template has an unrecognized deterministic selection rule")
    aliases = template.get("candidate_aliases")
    if aliases != ["gpt-oss:20b", "qwen3:30b", "deepseek-r1:32b"]:
        raise Refused("public model-canary template must retain the reviewed candidate body set")
    cases = template.get("case_set")
    expected_ids = [f"{frontier}{ordinal}" for frontier in "ABCDEFGH" for ordinal in (1, 2)]
    if not isinstance(cases, list) or [row.get("id") if isinstance(row, dict) else None for row in cases] != expected_ids:
        raise Refused("public model-canary template must retain ordered A1-H2 public cases")
    for row in cases:
        if not isinstance(row, dict) or set(row) != {"id", "frontier", "seed", "prompt", "answer"}:
            raise Refused("public model-canary case has an unexpected shape")
        if row.get("frontier") != row.get("id", "")[0] or not isinstance(row.get("seed"), int) or row["seed"] < 1:
            raise Refused("public model-canary case identity is invalid")
        if not all(isinstance(row.get(name), str) and row[name].strip() for name in ("prompt", "answer")):
            raise Refused("public model-canary case must retain explicit public prompt and answer text")


def _validate_operator_decision(design: dict[str, Any], decision: dict[str, Any]) -> None:
    """Keep the recorded user policy frozen, inert, and aligned to the design."""
    if (
        decision.get("schema") != "SUBSTRATE_ODYSSEY_OPERATOR_DECISION/v1"
        or decision.get("program") != "substrate-odyssey-7d-v1"
        or decision.get("status") != "user_policy_recorded_not_a_gate_receipt"
        or decision.get("activation") is not False
        or decision.get("external_activation") is not False
    ):
        raise Refused("operator decision must remain an inactive non-gate policy record")
    model_selection = decision.get("model_selection")
    if not isinstance(model_selection, dict) or model_selection.get("selection_state") not in {
        "waiting_for_public_canary_inventory",
        "public_canary_complete_human_review_pending",
    }:
        raise Refused("operator decision must remain either pre-canary or pending human review")
    statistics = decision.get("statistics")
    design_statistics = design.get("statistics")
    if not isinstance(statistics, dict) or not isinstance(design_statistics, dict):
        raise Refused("operator decision and hardened design require statistics")
    if statistics.get("score_weights") != design_statistics.get("score_weights"):
        raise Refused("operator decision score weights must match the hardened design")
    custody = decision.get("custody_policy")
    blindness = design.get("blindness")
    if not isinstance(custody, dict) or not isinstance(blindness, dict):
        raise Refused("operator decision and hardened design require custody policy")
    expected_custody = {
        "lane_custody_identity_count": blindness.get("lane_custody_identities"),
        "independent_human_task_custodian_count": blindness.get("human_task_custodians"),
        "frontiers_per_human_task_custodian": blindness.get("frontiers_per_task_custodian"),
        "independent_day7_answer_signature_custodian_count": blindness.get("day7_answer_signatories"),
    }
    if any(custody.get(field) != expected for field, expected in expected_custody.items()):
        raise Refused("operator decision custody counts must match the hardened design")
    limits = decision.get("limits")
    storage = design.get("storage")
    if (
        not isinstance(limits, dict)
        or not isinstance(storage, dict)
        or limits.get("storage_policy")
        != "25 GiB device-free floor plus measured dynamic private-write capacity bounded by a 120 GiB maximum private-write cap"
        or storage.get("device_free_floor_gib") != 25
        or storage.get("private_write_cap_gib") != 120
    ):
        raise Refused("operator decision storage policy must match the dynamic 25 GiB device floor")


def freeze(root: Path) -> dict[str, Any]:
    inputs = build_inputs(root)
    parsed = {name: read_json(path) for name, path in inputs.items()}
    if not all(_inactive(value) for value in parsed.values()):
        raise Refused("cannot freeze an activating source")
    design = parsed["hardened_design"]
    contracts = parsed["frontier_contract"]
    index = parsed["rendered_build_index"]
    _validate_resource_calibration_spec(design, parsed["resource_calibration"])
    _validate_source_selection_template(parsed["source_selection_template"])
    _validate_public_model_canary_template(design, parsed["public_model_canary_template"])
    _validate_operator_decision(design, parsed["operator_decision"])
    if design.get("program", {}).get("launch_allowed") is not False:
        raise Refused("hardened design must remain launch-disabled")
    if [row.get("id") for row in contracts.get("frontiers", [])] != list("ABCDEFGH"):
        raise Refused("frozen contract must cover exactly A-H")
    for row in contracts["frontiers"]:
        required = {"id", "task_families", "evaluator_only_rule", "parity_rule", "source_policy"}
        if not required.issubset(row) or not row["task_families"]:
            raise Refused(f"incomplete task contract for {row.get('id')}")
    if index.get("authority_sha256") != file_digest(inputs["hardened_design"]):
        raise Refused("rendered build does not match hardened design")
    if index.get("task_contract_sha256") != file_digest(inputs["frontier_contract"]):
        raise Refused("rendered build does not match task contract")
    for relative, expected in index.get("artifacts", {}).items():
        artifact = root / relative
        if file_digest(artifact) != expected:
            raise Refused(f"rendered artifact drift: {relative}")
    body = {
        "schema": "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1",
        "program": PROGRAM,
        "activation": False,
        "scientific_status": "frozen_waiting_for_verified_r2",
        "purpose": "complete static Odyssey build; transition only after R2 verification",
        "input_sha256": {name: file_digest(path) for name, path in inputs.items()},
        "implementation_sha256": {name: file_digest(path) for name, path in implementation_inputs(root).items()},
        "r2_requirements": {
            "longitudinal_result": {"scientific_status": "complete", "actual_wall_hours": 24, "continuity_passing": True},
            "continuity_verification": {
                "schema": "SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION/v1",
                "scientific_status": "pass",
                "independently_verified": True,
            },
            "source_provenance": {
                "schema": "SUBSTRATE_SANDBOX_R2_PROVENANCE_VERIFICATION/v1",
                "scientific_status": "pass",
                "independently_verified": True,
            },
            "no_live_r2_supervisor": True,
        },
        "transition": {
            "automatic_action": "write_preflight_authorization_receipt_only",
            "never_automatic": [
                "candidate_selection",
                "control_selection",
                "task_or_answer_selection",
                "rights_acceptance",
                "custody_key_creation",
                "scientific_worker_launch",
            ],
            "next_state": "odyssey_preflight_authorized",
        },
    }
    body["sha256"] = digest(body)
    return read_json(write_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", body))


def _live_r2(root: Path) -> bool:
    supervision = root / "runs/substrate/tangible_sandbox/longitudinal-supervision"
    for path in supervision.glob("*/supervisor-state.json"):
        try:
            if read_json(path).get("status") == "worker_running":
                return True
        except Refused:
            continue
    return False


def _complete_r2(root: Path, frozen: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    if _live_r2(root):
        return False, {"reason": "live_r2_supervisor_present"}
    paths = {
        "longitudinal_result": root / EVIDENCE / "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json",
        "continuity_verification": root / EVIDENCE / "SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION.json",
        "source_provenance": root / EVIDENCE / "SUBSTRATE_SANDBOX_R2_PROVENANCE_VERIFICATION.json",
    }
    values: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        try:
            values[name] = read_json(path)
        except Refused:
            return False, {"reason": f"r2_{name}_missing_or_invalid", "path": str(path)}
    for name, value in values.items():
        unsigned = dict(value)
        claimed = unsigned.pop("sha256", None)
        if not isinstance(claimed, str) or claimed != digest(unsigned):
            return False, {"reason": f"r2_{name}_digest_invalid", "path": str(paths[name])}
    requirements = frozen["r2_requirements"]
    for name, expected in requirements.items():
        if name == "no_live_r2_supervisor":
            continue
        matches = True
        for key, value in expected.items():
            observed = values[name].get(key)
            if name == "longitudinal_result" and key == "actual_wall_hours":
                matches = isinstance(observed, (int, float)) and observed >= value
            else:
                matches = observed == value
            if not matches:
                break
        if not matches:
            return False, {"reason": f"r2_{name}_does_not_match", "path": str(paths[name])}
    return True, {name: file_digest(path) for name, path in paths.items()}


def transition(root: Path) -> dict[str, Any]:
    frozen_path = root / PLAN / "ODYSSEY_FROZEN_BUILD.json"
    frozen = read_json(frozen_path)
    unsigned = dict(frozen)
    claimed = unsigned.pop("sha256", None)
    if claimed != digest(unsigned):
        raise Refused("frozen build digest mismatch")
    for name, expected in frozen["input_sha256"].items():
        observed = file_digest(build_inputs(root)[name])
        if observed != expected:
            raise Refused(f"frozen input drift: {name}")
    for name, expected in frozen["implementation_sha256"].items():
        if file_digest(implementation_inputs(root)[name]) != expected:
            raise Refused(f"frozen implementation drift: {name}")
    complete, details = _complete_r2(root, frozen)
    base = {
        "schema": "SUBSTRATE_ODYSSEY_R2_TRANSITION_RECEIPT/v1",
        "program": PROGRAM,
        "activation": False,
        "frozen_build_sha256": claimed,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    if not complete:
        receipt = {**base, "state": "waiting_for_verified_r2", "details": details, "preflight_authorized": False}
        receipt["sha256"] = digest(receipt)
        return read_json(write_json(root / RUNS / "TRANSITION_STATE.json", receipt))
    receipt = {
        **base,
        "state": "odyssey_preflight_authorized",
        "details": details,
        "preflight_authorized": True,
        "automatic_action_completed": "receipt_only",
        "forbidden_actions_not_taken": frozen["transition"]["never_automatic"],
    }
    receipt["sha256"] = digest(receipt)
    final = root / RUNS / "R2_VERIFIED_ODYSSEY_PREFLIGHT_AUTHORIZATION.json"
    return read_json(write_json(final, receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "transition"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = freeze(args.root) if args.command == "freeze" else transition(args.root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
