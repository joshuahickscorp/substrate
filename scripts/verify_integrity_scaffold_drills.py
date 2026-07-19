#!/usr/bin/env python
"""Independently replay and attack the F59 and F60 integrity drill receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.substrate.events import atomic_write_json

SCHEMA = "mop-integrity-scaffold-run/v1"
VERIFIER_SCHEMA = "mop-integrity-scaffold-independent-verifier/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
LIFECYCLE_SCHEMA = "mop-lifecycle-journal/v1"
REWRITE_SCHEMA = "mop-transactional-rewrite-contract/v1"
STAGE_SCHEMA = "mop-rewrite-stage-receipt/v1"
REWRITE_STAGES = ("shadow", "canary", "rollback", "evaluator-conflict")
REWRITE_CONTROLS = (
    "same-principal-request",
    "missing-stage-receipt",
    "truncated-receipt-digest",
    "evaluator-disagreement",
    "promoter-as-evaluator",
    "forged-stage-artifact",
    "wellformed-request",
)
IMPLEMENTATION_PATHS = (
    "registry/experiments.yaml",
    "src/mop/substrate/events.py",
    "src/mop/substrate/lifecycle.py",
    "src/mop/falsification/verdict_gate.py",
    "src/mop/falsification/integrity_scaffold.py",
    "scripts/run_integrity_scaffold_drills.py",
    "scripts/verify_integrity_scaffold_drills.py",
)
FRESH_SEEDS = (101, 103, 107)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "value" in value
        and isinstance(value.get("sha256"), str)
        and _SHA_RE.fullmatch(value["sha256"]) is not None
        and _sha(value["value"]) == value["sha256"]
    )


def _verify_chain(journal: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(journal, dict) or journal.get("schema") != LIFECYCLE_SCHEMA:
        return ["lifecycle schema drift"]
    entries = journal.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["lifecycle entries missing"]
    previous: str | None = None
    memory_ref = journal.get("memory_ref")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"lifecycle entry {index} is not a mapping")
            continue
        if entry.get("sequence") != index or entry.get("revision") != index + 1:
            errors.append(f"lifecycle entry {index} sequence or revision drift")
        if entry.get("memory_ref") != memory_ref:
            errors.append(f"lifecycle entry {index} memory identity drift")
        if entry.get("previous_entry_sha256") != previous:
            errors.append(f"lifecycle entry {index} previous digest drift")
        content = entry.get("content")
        if content is not None and not _verify_frozen(content):
            errors.append(f"lifecycle entry {index} content digest drift")
        declared = entry.get("entry_sha256")
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if not isinstance(declared, str) or _sha(body) != declared:
            errors.append(f"lifecycle entry {index} digest drift")
        previous = declared if isinstance(declared, str) else None
    if journal.get("head_sha256") != previous:
        errors.append("lifecycle head digest drift")
    return errors


def _verify_poisoning_unit(unit: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(unit, dict):
        return ["poisoning unit missing"]
    journal = unit.get("journal")
    errors.extend(_verify_chain(journal))
    if not isinstance(journal, dict):
        return errors
    entries = journal.get("entries", [])
    operations = [entry.get("operation") for entry in entries]
    if operations != ["record", "revise", "poisoning", "rollback"]:
        errors.append("poisoning operation sequence drift")
        return errors
    if entries[3].get("target_revision") != 1:
        errors.append("poisoning rollback does not restore clean revision")
    clean = entries[0].get("content")
    injected = entries[1].get("content")
    if not _verify_frozen(clean) or not _verify_frozen(injected):
        errors.append("poisoning clean or injected content invalid")
    elif clean.get("sha256") == injected.get("sha256"):
        errors.append("poisoning injection is not distinguishable from clean content")
    expected_metrics = {
        "quarantine_closes_availability": entries[2].get("operation") == "poisoning",
        "rollback_restores_clean_content": entries[3].get("target_revision") == 1,
        "chain_verifies_after_drill": not _verify_chain(journal),
    }
    if unit.get("metrics") != expected_metrics:
        errors.append("poisoning metric drift")
    controls = unit.get("controls", {})
    if set(controls) != {
        "clean-journal",
        "quarantine-only",
        "rollback-recovery",
        "stale-memory",
        "exact-replay",
    } or not all(value is True for value in controls.values()):
        errors.append("poisoning control coverage or result drift")
    if unit.get("journal_sha256") != _sha(journal):
        errors.append("poisoning journal digest drift")
    if unit.get("all_metrics_pass") is not all(expected_metrics.values()):
        errors.append("poisoning metric aggregate drift")
    if unit.get("all_controls_pass") is not True:
        errors.append("poisoning control aggregate drift")
    return errors


def _verify_consolidation_unit(unit: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(unit, dict):
        return ["consolidation unit missing"]
    journal = unit.get("journal")
    errors.extend(_verify_chain(journal))
    if not isinstance(journal, dict):
        return errors
    entries = journal.get("entries", [])
    operations = [entry.get("operation") for entry in entries]
    if operations != ["record", "revise", "delete"]:
        errors.append("consolidation operation sequence drift")
        return errors
    tokens = unit.get("private_tokens")
    if not isinstance(tokens, list) or not tokens or any(not isinstance(row, str) for row in tokens):
        errors.append("consolidation private token declaration drift")
        return errors
    consolidated = entries[1].get("content")
    token_free = _verify_frozen(consolidated) and all(
        token not in json.dumps(consolidated.get("value"), sort_keys=True) for token in tokens
    )
    deleted = entries[2].get("operation") == "delete" and entries[2].get("content") is None
    expected_metrics = {
        "consolidated_content_token_free": token_free,
        "deletion_closes_availability": deleted,
        "chain_verifies_after_drill": not _verify_chain(journal),
    }
    if unit.get("metrics") != expected_metrics:
        errors.append("consolidation metric drift")
    controls = unit.get("controls", {})
    if set(controls) != {
        "clean-journal",
        "token-free-consolidation",
        "deletion-follow-up",
        "exact-replay",
    } or not all(value is True for value in controls.values()):
        errors.append("consolidation control coverage or result drift")
    if unit.get("journal_sha256") != _sha(journal):
        errors.append("consolidation journal digest drift")
    if unit.get("all_metrics_pass") is not all(expected_metrics.values()):
        errors.append("consolidation metric aggregate drift")
    if unit.get("all_controls_pass") is not True:
        errors.append("consolidation control aggregate drift")
    return errors


def _raw_rewrite_outcome(contract: dict[str, Any], request: Any) -> tuple[str, str | None]:
    if not isinstance(request, dict):
        return "refuse", "request missing"
    authorities = {row.get("role"): row.get("principal") for row in contract.get("authorities", [])}
    promotion = authorities.get("promotion")
    execution = authorities.get("execution")
    requested_by = request.get("requested_by")
    executed_by = request.get("executed_by")
    if requested_by != promotion or executed_by != execution or requested_by == executed_by:
        return "refuse", "authority confusion"
    digests = request.get("stage_receipts")
    artifacts = request.get("stage_artifacts")
    if not isinstance(digests, dict) or not isinstance(artifacts, dict):
        return "refuse", "stage mappings missing"
    if set(digests) != set(REWRITE_STAGES) or set(artifacts) != set(REWRITE_STAGES):
        return "refuse", "stage coverage drift"
    contract_sha = _sha(contract)
    for stage in REWRITE_STAGES:
        digest = digests.get(stage)
        artifact = artifacts.get(stage)
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            return "refuse", "stage digest malformed"
        if not isinstance(artifact, dict):
            return "refuse", "stage artifact missing"
        if (
            artifact.get("schema") != STAGE_SCHEMA
            or artifact.get("stage") != stage
            or artifact.get("contract_sha256") != contract_sha
            or artifact.get("status") != "pass"
            or _sha(artifact) != digest
        ):
            return "refuse", "stage artifact invalid"
    verdicts = request.get("evaluator_verdicts")
    if not isinstance(verdicts, list) or len(verdicts) < 2:
        return "refuse", "evaluator coverage drift"
    evaluators: list[str] = []
    for row in verdicts:
        if not isinstance(row, dict):
            return "refuse", "evaluator row invalid"
        evaluator = row.get("evaluator")
        if not isinstance(evaluator, str) or not evaluator or evaluator in {promotion, execution}:
            return "refuse", "evaluator authority confusion"
        evaluators.append(evaluator)
        if row.get("verdict") != "pass":
            return "refuse", "evaluator disagreement"
    if len(set(evaluators)) != len(evaluators):
        return "refuse", "duplicate evaluator"
    return "allow", None


def _verify_rewrite(f60: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(f60, dict):
        return ["f60 block missing"]
    contract = f60.get("contract")
    if not isinstance(contract, dict) or contract.get("schema") != REWRITE_SCHEMA:
        return ["f60 rewrite contract missing or malformed"]
    if f60.get("contract_sha256") != _sha(contract):
        errors.append("f60 contract digest drift")
    if tuple(f60.get("controls", ())) != REWRITE_CONTROLS:
        errors.append("f60 declared control order drift")
    units = f60.get("units", [])
    if not isinstance(units, list) or len(units) < 3:
        return [*errors, "f60 independent unit count drift"]
    seeds: set[int] = set()
    for unit in units:
        if not isinstance(unit, dict):
            errors.append("f60 unit is not a mapping")
            continue
        seed = unit.get("seed")
        if not isinstance(seed, int) or seed < 0 or seed in seeds:
            errors.append("f60 seed identity drift")
        else:
            seeds.add(seed)
        cases = unit.get("cases", [])
        if [row.get("control") for row in cases] != list(REWRITE_CONTROLS):
            errors.append(f"f60 control coverage drift at seed {seed}")
            continue
        allowed = 0
        refused = 0
        for row in cases:
            observed, _ = _raw_rewrite_outcome(contract, row.get("request"))
            expected = "allow" if row.get("control") == "wellformed-request" else "refuse"
            if observed != expected or row.get("observed") != observed:
                errors.append(f"f60 independent outcome drift for {row.get('control')} seed {seed}")
            if row.get("expected") != expected or row.get("expectation_met") is not (observed == expected):
                errors.append(f"f60 expectation drift for {row.get('control')} seed {seed}")
            if observed == "allow":
                allowed += 1
                decision = row.get("decision")
                if not isinstance(decision, dict) or decision.get("decision") != "allow":
                    errors.append(f"f60 allow decision missing at seed {seed}")
                elif decision.get("decision_sha256") != _sha(
                    {key: value for key, value in decision.items() if key != "decision_sha256"}
                ):
                    errors.append(f"f60 allow decision digest drift at seed {seed}")
            else:
                refused += 1
                if not isinstance(row.get("refusal"), str) or not row.get("refusal"):
                    errors.append(f"f60 refusal reason missing at seed {seed}")
        if unit.get("allowed_count") != allowed or unit.get("refused_count") != refused:
            errors.append(f"f60 case count drift at seed {seed}")
        if unit.get("all_controls_pass") is not True:
            errors.append(f"f60 control aggregate drift at seed {seed}")
    if f60.get("independent_unit_count") != len(units):
        errors.append("f60 independent unit aggregate drift")
    return errors


def _fresh_challenges() -> tuple[list[dict[str, Any]], list[str]]:
    from mop.falsification.integrity_scaffold import (
        REWRITE_STAGE_RECEIPT_SCHEMA,
        PromotionRefused,
        build_consolidation_drill_journal,
        build_poisoning_drill_journal,
        build_rewrite_drill_contract,
        enforce_promotion_refusal,
        verify_deletion_through_consolidation,
        verify_poisoning_resistance,
    )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for seed in FRESH_SEEDS:
        poison = build_poisoning_drill_journal(seed)
        consolidation, tokens = build_consolidation_drill_journal(seed + 1000)
        contract = build_rewrite_drill_contract()
        artifacts = {
            stage: {
                "schema": REWRITE_STAGE_RECEIPT_SCHEMA,
                "stage": stage,
                "contract_sha256": contract.sha256,
                "seed": seed,
                "status": "pass",
                "checks": ["fresh-seed-independent-challenge"],
            }
            for stage in REWRITE_STAGES
        }
        request: dict[str, Any] = {
            "requested_by": contract.authority("promotion").principal,
            "executed_by": contract.authority("execution").principal,
            "stage_receipts": {stage: _sha(body) for stage, body in artifacts.items()},
            "stage_artifacts": artifacts,
            "evaluator_verdicts": [
                {"evaluator": f"evaluator:challenge-{seed}-a", "verdict": "pass"},
                {"evaluator": f"evaluator:challenge-{seed}-b", "verdict": "pass"},
            ],
        }
        try:
            decision = enforce_promotion_refusal(contract, request)
            allow_ok = decision.get("decision") == "allow"
        except PromotionRefused:
            allow_ok = False
        attacked = copy.deepcopy(request)
        attacked["stage_artifacts"]["shadow"]["status"] = "fail"
        attack_refused = False
        try:
            enforce_promotion_refusal(contract, attacked)
        except PromotionRefused:
            attack_refused = True
        poison_ok = verify_poisoning_resistance(poison) == []
        consolidation_ok = verify_deletion_through_consolidation(consolidation, tokens) == []
        row = {
            "seed": seed,
            "poisoning_pass": poison_ok,
            "consolidation_pass": consolidation_ok,
            "wellformed_rewrite_allowed": allow_ok,
            "forged_artifact_refused": attack_refused,
            "all_pass": poison_ok and consolidation_ok and allow_ok and attack_refused,
        }
        rows.append(row)
        if not row["all_pass"]:
            errors.append(f"fresh-seed challenge failed at seed {seed}")
    return rows, errors


def _base_errors(receipt: dict[str, Any], *, check_live_files: bool) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if receipt.get("schema") != SCHEMA:
        errors.append("integrity receipt schema drift")
    if receipt.get("claim_scope") != CLAIM_SCOPE:
        errors.append("integrity claim scope drift")
    core = {
        key: value
        for key, value in receipt.items()
        if key not in {"core_payload_sha256", "independent_verifier", "payload_sha256"}
    }
    if receipt.get("core_payload_sha256") != _sha(core):
        errors.append("integrity core payload digest drift")
    implementation = receipt.get("implementation", [])
    if tuple(row.get("path") for row in implementation) != IMPLEMENTATION_PATHS:
        errors.append("integrity implementation path set drift")
    if check_live_files:
        for row in implementation:
            path = REPO_ROOT / str(row.get("path"))
            if not path.is_file() or _sha_file(path) != row.get("sha256"):
                errors.append(f"integrity live implementation drift at {row.get('path')}")
    f59 = receipt.get("f59")
    if not isinstance(f59, dict):
        errors.append("f59 block missing")
        f59 = {}
    poisoning_units = f59.get("poisoning_units", [])
    consolidation_units = f59.get("consolidation_units", [])
    if len(poisoning_units) < 3 or len(consolidation_units) < 3:
        errors.append("f59 independent unit floor not met")
    for unit in poisoning_units:
        errors.extend(
            f"f59 poison seed {unit.get('seed')}: {error}"
            for error in _verify_poisoning_unit(unit)
        )
    for unit in consolidation_units:
        errors.extend(
            f"f59 consolidation seed {unit.get('seed')}: {error}"
            for error in _verify_consolidation_unit(unit)
        )
    errors.extend(_verify_rewrite(receipt.get("f60")))
    calibration = receipt.get("difficulty_calibration", {})
    if (
        calibration.get("calibrated") is not True
        or calibration.get("ceilinged_tie") is not False
        or set(calibration.get("observed_outcomes", [])) != {"allow", "refuse"}
    ):
        errors.append("integrity difficulty calibration drift")
    expected_status = "mechanics-pass" if not errors else "mechanics-fail"
    if receipt.get("status") != expected_status:
        errors.append("integrity status drift")
    return errors, {
        "f59_unit_count": len(poisoning_units) + len(consolidation_units),
        "f60_unit_count": len(receipt.get("f60", {}).get("units", [])),
        "live_implementation_checked": check_live_files,
    }


def _mutation_tests(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    poison_chain = copy.deepcopy(receipt["f59"]["poisoning_units"][0])
    poison_chain["journal"]["entries"][1]["reason"] = "mutated"
    rows.append({"id": "poison-chain", "errors": _verify_poisoning_unit(poison_chain)})

    rollback = copy.deepcopy(receipt["f59"]["poisoning_units"][0])
    rollback["journal"]["entries"][3]["target_revision"] = 2
    rows.append({"id": "rollback-target", "errors": _verify_poisoning_unit(rollback)})

    token = copy.deepcopy(receipt["f59"]["consolidation_units"][0])
    leaked = token["private_tokens"][0]
    token["journal"]["entries"][1]["content"]["value"]["summary"] = leaked
    token["journal"]["entries"][1]["content"]["sha256"] = _sha(
        token["journal"]["entries"][1]["content"]["value"]
    )
    rows.append({"id": "consolidation-token", "errors": _verify_consolidation_unit(token)})

    deletion = copy.deepcopy(receipt["f59"]["consolidation_units"][0])
    deletion["journal"]["entries"][2]["operation"] = "revise"
    rows.append({"id": "deletion-drop", "errors": _verify_consolidation_unit(deletion)})

    forged = copy.deepcopy(receipt["f60"])
    wellformed = forged["units"][0]["cases"][-1]
    wellformed["request"]["stage_artifacts"]["shadow"]["status"] = "fail"
    rows.append({"id": "stage-artifact", "errors": _verify_rewrite(forged)})

    coverage = copy.deepcopy(receipt["f60"])
    coverage["units"][0]["cases"].pop()
    rows.append({"id": "rewrite-control", "errors": _verify_rewrite(coverage)})

    return [
        {"id": row["id"], "rejected": bool(row["errors"]), "observed_errors": row["errors"]}
        for row in rows
    ]


def verify_receipt(
    receipt: dict[str, Any], *, check_live_files: bool = True, run_mutations: bool = True
) -> dict[str, Any]:
    errors, checks = _base_errors(receipt, check_live_files=check_live_files)
    fresh_rows, fresh_errors = _fresh_challenges()
    errors.extend(fresh_errors)
    mutations: list[dict[str, Any]] = []
    mutation_runner_error: str | None = None
    if run_mutations:
        try:
            mutations = _mutation_tests(receipt)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            mutation_runner_error = f"integrity mutation battery could not execute: {exc}"
            errors.append(mutation_runner_error)
    all_mutations_rejected = bool(mutations) and all(row["rejected"] for row in mutations)
    if run_mutations and not all_mutations_rejected:
        errors.append("integrity semantic mutation rejection incomplete")
    return {
        "schema": VERIFIER_SCHEMA,
        "implementation": "independent raw JSON replay plus fresh-seed public-API attacks",
        "claim_scope": CLAIM_SCOPE,
        "core_payload_sha256": receipt.get("core_payload_sha256"),
        "checks": checks,
        "fresh_seed_challenges": fresh_rows,
        "fresh_seed_count": len(fresh_rows),
        "fresh_seeds_disjoint_from_primary": not (
            set(FRESH_SEEDS)
            & {
                *[row.get("seed") for row in receipt.get("f59", {}).get("poisoning_units", [])],
                *[row.get("seed") for row in receipt.get("f59", {}).get("consolidation_units", [])],
                *[row.get("seed") for row in receipt.get("f60", {}).get("units", [])],
            }
        ),
        "mutation_tests": mutations,
        "all_mutations_rejected": all_mutations_rejected,
        "mutation_runner_error": mutation_runner_error,
        "errors": errors,
        "independent": True,
        "adversarial": True,
        "verified": not errors,
    }


def verify_payload_sha256(receipt: dict[str, Any]) -> bool:
    digest = receipt.get("payload_sha256")
    return isinstance(digest, str) and digest == _sha(
        {key: value for key, value in receipt.items() if key != "payload_sha256"}
    )




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "receipt",
        nargs="?",
        default=str(REPO_ROOT / "proof" / "F59_F60_INTEGRITY_SCAFFOLD_RUN.json"),
    )
    parser.add_argument(
        "--report", default=str(REPO_ROOT / "proof" / "F59_F60_INTEGRITY_VERIFICATION.json")
    )
    parser.add_argument("--skip-live-files", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt_path = Path(args.receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = verify_receipt(
        receipt, check_live_files=not args.skip_live_files, run_mutations=True
    )
    report["payload_sha256_verified"] = verify_payload_sha256(receipt)
    report["receipt_payload_sha256"] = receipt.get("payload_sha256")
    report["receipt_file_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if args.report:
        atomic_write_json(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["verified"] and report["payload_sha256_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
