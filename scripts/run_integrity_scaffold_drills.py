#!/usr/bin/env python
"""Execute the F59 memory and F60 transactional rewrite scaffold drills."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.experiments.expansion_harness import CLAIM_SCOPE
from mop.falsification.integrity_scaffold import (
    REWRITE_STAGE_RECEIPT_SCHEMA,
    REWRITE_STAGES,
    PromotionRefused,
    build_consolidation_contract,
    build_consolidation_drill_journal,
    build_poisoning_contract,
    build_poisoning_drill_journal,
    build_rewrite_drill_contract,
    enforce_promotion_refusal,
    verify_deletion_through_consolidation,
    verify_poisoning_resistance,
)
from mop.substrate.events import EventRef, canonical_sha256
from mop.substrate.lifecycle import LifecycleJournal, MemoryRef

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "mop-integrity-scaffold-run/v1"
PRIMARY_REWRITE_SEEDS = (17, 37, 61)
IMPLEMENTATION_PATHS = (
    "registry/experiments.yaml",
    "src/mop/substrate/events.py",
    "src/mop/substrate/lifecycle.py",
    "src/mop/falsification/verdict_gate.py",
    "src/mop/falsification/integrity_scaffold.py",
    "scripts/run_integrity_scaffold_drills.py",
    "scripts/verify_integrity_scaffold_drills.py",
)
REWRITE_CONTROLS = (
    "same-principal-request",
    "missing-stage-receipt",
    "truncated-receipt-digest",
    "evaluator-disagreement",
    "promoter-as-evaluator",
    "forged-stage-artifact",
    "wellformed-request",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_journal(seed: int, label: str) -> LifecycleJournal:
    journal = LifecycleJournal(MemoryRef(f"memory:{label}-{seed}"))
    journal.record(EventRef(f"event:{label}-{seed}-record"), {"clean": True, "seed": seed})
    return journal


def _quarantine_only_journal(seed: int) -> LifecycleJournal:
    journal = LifecycleJournal(MemoryRef(f"memory:quarantine-only-{seed}"))
    journal.record(EventRef(f"event:quarantine-only-{seed}-record"), {"clean": True})
    journal.revise(EventRef(f"event:quarantine-only-{seed}-inject"), {"injected": True})
    journal.mark_poisoned(EventRef(f"event:quarantine-only-{seed}-mark"))
    return journal


def _poisoning_unit(seed: int) -> dict[str, Any]:
    journal = build_poisoning_drill_journal(seed)
    repeated = build_poisoning_drill_journal(seed)
    clean = _clean_journal(seed, "clean-control")
    quarantine_only = _quarantine_only_journal(seed)
    operations = [entry.operation.value for entry in journal.entries]
    poison_revision = operations.index("poisoning") + 1
    rollback_entry = next(entry for entry in journal.entries if entry.operation.value == "rollback")
    clean_state = journal.state_at(revision=int(rollback_entry.target_revision or 0))
    injected_state = journal.state_at(revision=2)
    recovered_state = journal.state_at()
    metrics = {
        "quarantine_closes_availability": not any(
            journal.state_at(revision=poison_revision).available_at(tick) for tick in (0, 1, 7)
        ),
        "rollback_restores_clean_content": (
            clean_state.content is not None
            and recovered_state.content is not None
            and clean_state.content.canonical == recovered_state.content.canonical
        ),
        "chain_verifies_after_drill": verify_poisoning_resistance(journal) == [],
    }
    controls = {
        "clean-journal": clean.verify() == [] and clean.state_at().available_at(0),
        "quarantine-only": bool(verify_poisoning_resistance(quarantine_only)),
        "rollback-recovery": metrics["rollback_restores_clean_content"],
        "stale-memory": (
            injected_state.content is not None
            and recovered_state.content is not None
            and injected_state.content.sha256 != recovered_state.content.sha256
        ),
        "exact-replay": journal.payload() == repeated.payload(),
    }
    return {
        "seed": seed,
        "journal": journal.payload(),
        "journal_sha256": journal.sha256,
        "private_tokens": [],
        "metrics": metrics,
        "controls": controls,
        "all_metrics_pass": all(metrics.values()),
        "all_controls_pass": all(controls.values()),
    }


def _consolidation_unit(seed: int) -> dict[str, Any]:
    journal, tokens = build_consolidation_drill_journal(seed)
    repeated, repeated_tokens = build_consolidation_drill_journal(seed)
    clean = _clean_journal(seed, "consolidation-clean-control")
    operations = [entry.operation.value for entry in journal.entries]
    revise_revision = operations.index("revise") + 1
    consolidated = journal.state_at(revision=revise_revision)
    final = journal.state_at()
    canonical = consolidated.content.canonical if consolidated.content is not None else ""
    metrics = {
        "consolidated_content_token_free": all(token not in canonical for token in tokens),
        "deletion_closes_availability": (
            final.deleted
            and not final.exists
            and not any(final.available_at(tick) for tick in (0, 1, 7))
        ),
        "chain_verifies_after_drill": verify_deletion_through_consolidation(journal, tokens) == [],
    }
    controls = {
        "clean-journal": clean.verify() == [] and clean.state_at().available_at(0),
        "token-free-consolidation": metrics["consolidated_content_token_free"],
        "deletion-follow-up": metrics["deletion_closes_availability"],
        "exact-replay": journal.payload() == repeated.payload() and tokens == repeated_tokens,
    }
    return {
        "seed": seed,
        "journal": journal.payload(),
        "journal_sha256": journal.sha256,
        "private_tokens": list(tokens),
        "metrics": metrics,
        "controls": controls,
        "all_metrics_pass": all(metrics.values()),
        "all_controls_pass": all(controls.values()),
    }


def _stage_artifacts(contract_sha256: str, seed: int) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "schema": REWRITE_STAGE_RECEIPT_SCHEMA,
            "stage": stage,
            "contract_sha256": contract_sha256,
            "seed": seed,
            "status": "pass",
            "checks": ["entry-criteria-met", "abort-criteria-clear", "receipt-complete"],
        }
        for stage in REWRITE_STAGES
    }


def _wellformed_request(seed: int) -> dict[str, Any]:
    contract = build_rewrite_drill_contract()
    artifacts = _stage_artifacts(contract.sha256, seed)
    return {
        "requested_by": contract.authority("promotion").principal,
        "executed_by": contract.authority("execution").principal,
        "stage_receipts": {
            stage: canonical_sha256(artifact) for stage, artifact in artifacts.items()
        },
        "stage_artifacts": artifacts,
        "evaluator_verdicts": [
            {"evaluator": f"evaluator:fresh-{seed}-a", "verdict": "pass"},
            {"evaluator": f"evaluator:fresh-{seed}-b", "verdict": "pass"},
        ],
    }


def _run_rewrite_case(
    control: str, contract: Any, request: dict[str, Any], expected: str
) -> dict[str, Any]:
    decision: dict[str, Any] | None = None
    refusal: str | None = None
    try:
        decision = enforce_promotion_refusal(contract, request)
        observed = "allow"
    except PromotionRefused as exc:
        observed = "refuse"
        refusal = str(exc)
    return {
        "control": control,
        "expected": expected,
        "observed": observed,
        "expectation_met": observed == expected,
        "request": request,
        "decision": decision,
        "refusal": refusal,
    }


def _rewrite_unit(seed: int) -> dict[str, Any]:
    contract = build_rewrite_drill_contract()
    base = _wellformed_request(seed)
    requests: dict[str, dict[str, Any]] = {}

    same = copy.deepcopy(base)
    same["requested_by"] = same["executed_by"]
    requests["same-principal-request"] = same

    missing = copy.deepcopy(base)
    missing["stage_receipts"].pop("rollback")
    missing["stage_artifacts"].pop("rollback")
    requests["missing-stage-receipt"] = missing

    truncated = copy.deepcopy(base)
    truncated["stage_receipts"]["canary"] = truncated["stage_receipts"]["canary"][:12]
    requests["truncated-receipt-digest"] = truncated

    conflict = copy.deepcopy(base)
    conflict["evaluator_verdicts"][1]["verdict"] = "fail"
    requests["evaluator-disagreement"] = conflict

    promoter_eval = copy.deepcopy(base)
    promoter_eval["evaluator_verdicts"][0]["evaluator"] = contract.authority(
        "promotion"
    ).principal
    requests["promoter-as-evaluator"] = promoter_eval

    forged = copy.deepcopy(base)
    forged["stage_artifacts"]["shadow"]["status"] = "fail"
    requests["forged-stage-artifact"] = forged
    requests["wellformed-request"] = copy.deepcopy(base)

    cases = [
        _run_rewrite_case(
            control,
            contract,
            requests[control],
            "allow" if control == "wellformed-request" else "refuse",
        )
        for control in REWRITE_CONTROLS
    ]
    return {
        "seed": seed,
        "cases": cases,
        "allowed_count": sum(row["observed"] == "allow" for row in cases),
        "refused_count": sum(row["observed"] == "refuse" for row in cases),
        "all_controls_pass": all(row["expectation_met"] for row in cases),
    }


def build_receipt() -> dict[str, Any]:
    poisoning_contract = build_poisoning_contract()
    consolidation_contract = build_consolidation_contract()
    rewrite_contract = build_rewrite_drill_contract()
    poisoning = [_poisoning_unit(seed) for seed in poisoning_contract.seeds]
    consolidation = [_consolidation_unit(seed) for seed in consolidation_contract.seeds]
    rewrite = [_rewrite_unit(seed) for seed in PRIMARY_REWRITE_SEEDS]
    f59_pass = all(
        row["all_metrics_pass"] and row["all_controls_pass"]
        for row in [*poisoning, *consolidation]
    )
    f60_pass = all(row["all_controls_pass"] for row in rewrite)
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": "deterministic programmatic mechanics",
        "status": "mechanics-pass" if f59_pass and f60_pass else "mechanics-fail",
        "implementation": [
            {"path": path, "sha256": _sha_file(REPO_ROOT / path)}
            for path in IMPLEMENTATION_PATHS
        ],
        "difficulty_calibration": {
            "bed": "binary fail-closed boundary",
            "positive_outcome": "wellformed requests and complete recovery drills pass",
            "negative_outcome": "single-fault controls refuse or expose the fault",
            "observed_outcomes": ["allow", "refuse"],
            "ceilinged_tie": False,
            "calibrated": f59_pass and f60_pass,
        },
        "f59": {
            "experiment_id": "f59_memory_poisoning_resistance",
            "null_hypothesis": poisoning_contract.null_hypothesis,
            "poisoning_contract": poisoning_contract.payload(),
            "poisoning_contract_sha256": poisoning_contract.sha256,
            "consolidation_contract": consolidation_contract.payload(),
            "consolidation_contract_sha256": consolidation_contract.sha256,
            "poisoning_units": poisoning,
            "consolidation_units": consolidation,
            "independent_unit_count": len(poisoning) + len(consolidation),
            "all_declared_controls_exercised": True,
            "result": "programmatic-mechanics-pass" if f59_pass else "null",
            "promotion": False,
        },
        "f60": {
            "experiment_id": "f60_transactional_self_rewrite",
            "null_hypothesis": (
                "promotion is allowed under an authority confusion, receipt gap, or evaluator conflict"
            ),
            "contract": rewrite_contract.payload(),
            "contract_sha256": rewrite_contract.sha256,
            "controls": list(REWRITE_CONTROLS),
            "units": rewrite,
            "independent_unit_count": len(rewrite),
            "all_declared_controls_exercised": True,
            "result": "programmatic-mechanics-pass" if f60_pass else "null",
            "promotion": False,
        },
        "scientific_scope": {
            "natural_memory_content_tested": False,
            "substrate_capability_claimed": False,
            "byte_level_history_erasure_claimed": False,
            "transactional_policy_effectiveness_claimed": False,
        },
    }
    core["core_payload_sha256"] = canonical_sha256(core)

    from scripts.verify_integrity_scaffold_drills import verify_receipt

    verification = verify_receipt(core, check_live_files=True, run_mutations=True)
    if verification["verified"] is not True:
        raise RuntimeError("independent integrity verification failed: " + "; ".join(verification["errors"]))
    receipt = {**core, "independent_verifier": verification}
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "proof" / "F59_F60_INTEGRITY_SCAFFOLD_RUN.json")
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = build_receipt()
    output = Path(args.out)
    _atomic_write(output, receipt)
    print(
        f"wrote {output}: status={receipt['status']}, "
        f"units={receipt['f59']['independent_unit_count'] + receipt['f60']['independent_unit_count']}, "
        f"payload={receipt['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
