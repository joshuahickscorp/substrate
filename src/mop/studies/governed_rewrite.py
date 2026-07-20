
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import platform
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..config import REPO_ROOT
from ..substrate.events import EventRef, FrozenJSON, canonical_bytes, canonical_sha256
from ..substrate.lifecycle import LifecycleJournal, MemoryRef
from .process_resources import PeakRSSMonitor

CONFIG_SCHEMA = "mop-governed-rewrite-config/v1"
STATE_SCHEMA = "mop-governed-rewrite-state/v1"
STATE_FILE_SCHEMA = "mop-governed-rewrite-state-file/v1"
PROPOSAL_SCHEMA = "mop-governed-rewrite-proposal/v1"
AUTHORITY_SCHEMA = "mop-governed-rewrite-authority/v1"
EVALUATION_SCHEMA = "mop-governed-rewrite-evaluation/v1"
CHECKPOINT_SCHEMA = "mop-governed-rewrite-checkpoint/v1"
TRANSACTION_SCHEMA = "mop-governed-rewrite-transaction/v1"
PREFLIGHT_SCHEMA = "mop-governed-rewrite-preflight/v1"

CLAIM_SCOPE = (
    "deterministic project-owned transaction mechanics over a structural fixture only; no natural-data, "
    "learned-rewrite, capability, cognition, sentience, security, or production-safety claim"
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "governed_rewrite_preflight.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "GOVERNED_REWRITE_PREFLIGHT.json"

IMPLEMENTATION_PATHS = (
    "configs/experiment/governed_rewrite_preflight.yaml",
    "src/mop/studies/governed_rewrite.py",
    "scripts/governed_rewrite_preflight.py",
    "tests/unit/test_governed_rewrite.py",
    "docs/GOVERNED_REWRITE_PREFLIGHT.md",
    "src/mop/studies/process_resources.py",
)

REUSED_PRIMITIVE_PATHS = (
    "src/mop/substrate/events.py",
    "src/mop/substrate/lifecycle.py",
)

AUDITED_UPSTREAM_PATHS = (
    "src/mop/experiments/form_rewrite_engine.py",
    "src/mop/experiments/f_form_substrate_missing.py",
    "src/mop/experiments/f_form_substrate.py",
    "configs/experiment/f8_plastic_substrate_rewrite.yaml",
    "configs/experiment/f20_substrate_crisis_test.yaml",
    "proof/FORM_SUBSTRATE/PREFLIGHT/f8_plastic_substrate_rewrite.json",
    "proof/FORM_SUBSTRATE/RECEIPTS/f20_substrate_crisis_test.json",
    "src/mop/provenance.py",
    "src/mop/studio/local_throttle.py",
)

SOURCE_IDENTITY_PATHS = (
    "src/mop/studies/governed_rewrite.py",
    "src/mop/studies/process_resources.py",
    *REUSED_PRIMITIVE_PATHS,
)

_FIXTURE_AUTHORITY_KEY = hashlib.sha256(b"mop-project-owned-governed-rewrite-fixture-authority-v1").digest()


class GovernedRewriteRefused(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    try:
        display = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    os.replace(temporary, path)
    return {
        "bytes": len(raw.encode("utf-8")),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "temporary_absent_after_replace": not temporary.exists(),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("governed rewrite config schema drift")
    if payload.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("governed rewrite claim scope drift")
    if not str(payload.get("null_hypothesis", "")).strip():
        raise ValueError("governed rewrite null hypothesis is required")

    authority = payload.get("authority", {})
    key_sha256 = hashlib.sha256(_FIXTURE_AUTHORITY_KEY).hexdigest()
    if authority.get("key_sha256") != key_sha256:
        raise ValueError("governed rewrite fixture authority root drift")
    if authority.get("key_id") != "mop-fixture-hmac-sha256-v1":
        raise ValueError("governed rewrite authority key id drift")
    if tuple(authority.get("proposal_paths", ())) != ("policy.adaptation_threshold",):
        raise ValueError("governed rewrite proposal scope drift")
    if tuple(authority.get("transaction_side_effect_paths", ())) != (
        "revision",
        "governance.consumed_authority_tokens",
    ):
        raise ValueError("governed rewrite transaction side-effect scope drift")
    issued = int(authority.get("issued_tick", -1))
    expires = int(authority.get("expires_tick", -1))
    execution = int(authority.get("execution_tick", -1))
    if not 0 <= issued <= execution <= expires or int(authority.get("max_uses", 0)) != 1:
        raise ValueError("governed rewrite authority lifetime drift")

    initial = payload.get("initial_state", {})
    if (
        int(initial.get("revision", 0)) != 1
        or int(initial.get("policy", {}).get("adaptation_threshold", -1)) != 3
        or int(initial.get("policy", {}).get("protected_threshold", -1)) != 3
        or initial.get("governance", {}).get("consumed_authority_tokens") != []
    ):
        raise ValueError("governed rewrite initial state drift")

    proposal = payload.get("proposal", {})
    rollback = payload.get("rollback_probe", {})
    if int(proposal.get("target_adaptation_threshold", -1)) != 2:
        raise ValueError("governed rewrite canonical proposal drift")
    if int(rollback.get("target_adaptation_threshold", -1)) != 0:
        raise ValueError("governed rewrite rollback probe drift")
    if proposal.get("token_id") == rollback.get("token_id"):
        raise ValueError("governed rewrite authority token ids must be distinct")

    evaluation = payload.get("evaluation", {})
    if evaluation.get("evaluator_id") != "mop-exact-threshold-evaluator-v1":
        raise ValueError("governed rewrite evaluator id drift")
    if evaluation.get("comparison") != "greater-than-or-equal":
        raise ValueError("governed rewrite evaluator comparison drift")
    seen: set[str] = set()
    for split in ("canary", "shadow"):
        cases = evaluation.get(f"{split}_cases", ())
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"governed rewrite {split} fixture is empty")
        for case in cases:
            case_id = str(case.get("case_id", ""))
            if not case_id or case_id in seen:
                raise ValueError("governed rewrite case ids must be nonempty and unique")
            seen.add(case_id)
            if case.get("route") not in {"adaptive", "protected"}:
                raise ValueError("governed rewrite case route drift")
            if int(case.get("target", -1)) not in {0, 1}:
                raise ValueError("governed rewrite case target must be binary")
            int(case.get("evidence"))

    envelope = payload.get("resource_envelope", {})
    if (
        envelope.get("device") != "cpu"
        or int(envelope.get("cpu_threads", 0)) != 1
        or float(envelope.get("maximum_wall_seconds", 0.0)) > 10.0
        or int(envelope.get("maximum_rss_bytes", 0)) > 1024**3
        or envelope.get("accelerator_required") is not False
        or envelope.get("model_weights_loaded") is not False
        or envelope.get("model_downloads_allowed") is not False
        or envelope.get("external_data_allowed") is not False
    ):
        raise ValueError("governed rewrite resource envelope drift")
    return payload


def _source_identity() -> dict[str, Any]:
    files = [_file_receipt(REPO_ROOT / path) for path in SOURCE_IDENTITY_PATHS]
    core = {"schema": "mop-governed-rewrite-source-identity/v1", "files": files}
    return {**core, "sha256": canonical_sha256(core)}


def _evaluator_spec(config: dict[str, Any]) -> dict[str, Any]:
    core = {
        "schema": "mop-governed-rewrite-evaluator/v1",
        "evaluator_id": config["evaluation"]["evaluator_id"],
        "comparison": config["evaluation"]["comparison"],
        "adaptive_rule": "evidence + calibration_offset compared with adaptation_threshold",
        "protected_rule": "evidence compared with protected_threshold",
        "case_source": "immutable config payload",
    }
    return {**core, "sha256": canonical_sha256(core)}


def build_memory_journal(config: dict[str, Any], *, poisoned: bool = False) -> LifecycleJournal:
    memory = config["memory"]
    journal = LifecycleJournal(MemoryRef(str(memory["ref"])))
    journal.record(
        EventRef("event:governed-rewrite/memory-record"),
        copy.deepcopy(memory["content"]),
        available=True,
        available_from_tick=int(memory["available_from_tick"]),
        available_until_tick=int(memory["available_until_tick"]),
        reason="bind clean calibration memory to the rewrite evaluator",
    )
    if poisoned:
        journal.mark_poisoned(
            EventRef("event:governed-rewrite/memory-poisoning"),
            reason="adversarial poisoning drill quarantines evaluator memory",
        )
    return journal


def _memory_binding(journal: LifecycleJournal) -> dict[str, Any]:
    state = journal.state_at()
    return {
        "memory_ref": str(journal.memory_ref),
        "journal_sha256": journal.sha256,
        "journal_head_sha256": journal.head_sha256,
        "revision": state.revision,
        "content_sha256": state.content.sha256 if state.content is not None else None,
    }


def _initial_state(
    config: dict[str, Any], journal: LifecycleJournal, evaluator: dict[str, Any]
) -> dict[str, Any]:
    initial = config["initial_state"]
    return {
        "schema": STATE_SCHEMA,
        "revision": int(initial["revision"]),
        "policy": {
            "adaptation_threshold": int(initial["policy"]["adaptation_threshold"]),
            "protected_threshold": int(initial["policy"]["protected_threshold"]),
        },
        "governance": {"consumed_authority_tokens": list(initial["governance"]["consumed_authority_tokens"])},
        "memory_binding": _memory_binding(journal),
        "evaluator_sha256": evaluator["sha256"],
    }


def _state_identity(state: dict[str, Any]) -> dict[str, Any]:
    frozen = FrozenJSON.from_value(state)
    return {"payload": frozen.value(), "sha256": frozen.sha256}


def _state_file_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": STATE_FILE_SCHEMA,
        "state": copy.deepcopy(state),
        "state_sha256": canonical_sha256(state),
    }


def _write_state(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    return _atomic_json(root / "state.json", _state_file_payload(state))


def _read_state(root: Path) -> dict[str, Any]:
    path = root / "state.json"
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernedRewriteRefused(f"committed state is unreadable: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != STATE_FILE_SCHEMA:
        raise GovernedRewriteRefused("committed state file schema drift")
    state = envelope.get("state")
    if not isinstance(state, dict) or envelope.get("state_sha256") != canonical_sha256(state):
        raise GovernedRewriteRefused("committed state digest drift")
    FrozenJSON.from_value(state)
    return state


def _evaluation_memory(journal: LifecycleJournal, tick: int, field: str) -> int:
    state = journal.state_at()
    if journal.verify():
        raise GovernedRewriteRefused("memory journal failed verification")
    if state.poisoned:
        raise GovernedRewriteRefused("memory-poisoned")
    if state.conflicted:
        raise GovernedRewriteRefused("memory-conflicted")
    if not state.available_at(tick) or state.content is None:
        raise GovernedRewriteRefused("memory-unavailable")
    value = state.content.value()
    if not isinstance(value, dict) or field not in value:
        raise GovernedRewriteRefused("memory-field-missing")
    return int(value[field])


def _fraction(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "decimal": round(numerator / denominator if denominator else 0.0, 8),
    }


def _evaluate_cases(
    state: dict[str, Any],
    cases: list[dict[str, Any]],
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
    *,
    tick: int,
) -> dict[str, Any]:
    if evaluator.get("schema") != "mop-governed-rewrite-evaluator/v1":
        raise GovernedRewriteRefused("evaluator-schema-drift")
    expected_evaluator_sha = canonical_sha256(
        {key: value for key, value in evaluator.items() if key != "sha256"}
    )
    if evaluator.get("sha256") != expected_evaluator_sha:
        raise GovernedRewriteRefused("evaluator-digest-drift")
    if state.get("evaluator_sha256") != evaluator["sha256"]:
        raise GovernedRewriteRefused("evaluator-tampered")
    binding = _memory_binding(journal)
    if state.get("memory_binding") != binding:
        raise GovernedRewriteRefused("memory-binding-drift")
    offset = _evaluation_memory(journal, tick, "calibration_offset")
    comparison = evaluator["comparison"]
    rows: list[dict[str, Any]] = []
    for case in cases:
        route = str(case["route"])
        evidence = int(case["evidence"])
        threshold = int(
            state["policy"]["adaptation_threshold" if route == "adaptive" else "protected_threshold"]
        )
        score = evidence + (offset if route == "adaptive" else 0)
        if comparison == "greater-than-or-equal":
            prediction = int(score >= threshold)
        elif comparison == "greater-than":
            prediction = int(score > threshold)
        else:
            raise GovernedRewriteRefused("evaluator-comparison-unsupported")
        target = int(case["target"])
        row = {
            "case_id": str(case["case_id"]),
            "route": route,
            "evidence": evidence,
            "calibration_offset_applied": offset if route == "adaptive" else 0,
            "threshold": threshold,
            "prediction": prediction,
            "target": target,
            "correct": prediction == target,
        }
        row["row_sha256"] = canonical_sha256(row)
        rows.append(row)
    correct = sum(bool(row["correct"]) for row in rows)
    route_metrics: dict[str, Any] = {}
    for route in ("adaptive", "protected"):
        selected = [row for row in rows if row["route"] == route]
        route_correct = sum(bool(row["correct"]) for row in selected)
        route_metrics[route] = {
            "correct": route_correct,
            "total": len(selected),
            "accuracy": _fraction(route_correct, len(selected)),
        }
    core = {
        "schema": EVALUATION_SCHEMA,
        "state_sha256": canonical_sha256(state),
        "evaluator_sha256": evaluator["sha256"],
        "memory_journal_sha256": journal.sha256,
        "tick": tick,
        "correct": correct,
        "total": len(rows),
        "capability": _fraction(correct, len(rows)),
        "by_route": route_metrics,
        "rows": rows,
    }
    return {**core, "sha256": canonical_sha256(core)}


def _gate_pair(before: dict[str, Any], after: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    before_by_id = {row["case_id"]: row for row in before["rows"]}
    after_by_id = {row["case_id"]: row for row in after["rows"]}
    if set(before_by_id) != set(after_by_id):
        raise GovernedRewriteRefused("evaluation case identity drift")
    regressions = [
        case_id
        for case_id in sorted(before_by_id)
        if before_by_id[case_id]["correct"] and not after_by_id[case_id]["correct"]
    ]
    protected_regressions = [
        case_id for case_id in regressions if before_by_id[case_id]["route"] == "protected"
    ]
    gain = int(after["correct"]) - int(before["correct"])
    checks = {
        "minimum_after_correct": int(after["correct"]) >= int(gate["minimum_after_correct"]),
        "minimum_capability_gain": gain >= int(gate["minimum_capability_gain"]),
        "maximum_regressions": len(regressions) <= int(gate["maximum_regressions"]),
        "maximum_protected_regressions": len(protected_regressions)
        <= int(gate["maximum_protected_regressions"]),
    }
    return {
        "before_correct": int(before["correct"]),
        "after_correct": int(after["correct"]),
        "capability_gain": gain,
        "regressions": regressions,
        "protected_regressions": protected_regressions,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _evaluation_pair(
    config: dict[str, Any],
    split: str,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
    tick: int,
) -> dict[str, Any]:
    cases = config["evaluation"][f"{split}_cases"]
    before = _evaluate_cases(before_state, cases, evaluator, journal, tick=tick)
    after = _evaluate_cases(after_state, cases, evaluator, journal, tick=tick)
    core = {
        "split": split,
        "before": before,
        "after": after,
        "gate": _gate_pair(before, after, config["evaluation"]["gates"][split]),
    }
    return {**core, "sha256": canonical_sha256(core)}


def build_proposal(
    config: dict[str, Any],
    base_state: dict[str, Any],
    source_identity: dict[str, Any],
    config_payload_sha256: str,
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
    *,
    rollback_probe: bool = False,
) -> dict[str, Any]:
    spec = config["rollback_probe" if rollback_probe else "proposal"]
    before = int(base_state["policy"]["adaptation_threshold"])
    after = int(spec["target_adaptation_threshold"])
    core = {
        "schema": PROPOSAL_SCHEMA,
        "proposal_id": str(spec["proposal_id"]),
        "reason": str(spec["reason"]),
        "base_state_sha256": canonical_sha256(base_state),
        "base_revision": int(base_state["revision"]),
        "source_identity_sha256": source_identity["sha256"],
        "config_payload_sha256": config_payload_sha256,
        "evaluator_sha256": evaluator["sha256"],
        "memory_journal_sha256": journal.sha256,
        "changes": [
            {
                "path": "policy.adaptation_threshold",
                "before": before,
                "after": after,
            }
        ],
    }
    return {**core, "proposal_sha256": canonical_sha256(core)}


def _proposal_paths(proposal: dict[str, Any]) -> list[str]:
    return [str(change.get("path")) for change in proposal.get("changes", ())]


def _token_signature(claims: dict[str, Any]) -> str:
    return hmac.new(_FIXTURE_AUTHORITY_KEY, canonical_bytes(claims), hashlib.sha256).hexdigest()


def issue_authority_token(
    config: dict[str, Any], proposal: dict[str, Any], *, rollback_probe: bool = False
) -> dict[str, Any]:
    authority = config["authority"]
    spec = config["rollback_probe" if rollback_probe else "proposal"]
    allowed_paths = [*authority["proposal_paths"], *authority["transaction_side_effect_paths"]]
    claims = {
        "schema": AUTHORITY_SCHEMA,
        "token_id": str(spec["token_id"]),
        "issuer": str(authority["issuer"]),
        "key_id": str(authority["key_id"]),
        "proposal_sha256": proposal["proposal_sha256"],
        "base_state_sha256": proposal["base_state_sha256"],
        "source_identity_sha256": proposal["source_identity_sha256"],
        "config_payload_sha256": proposal["config_payload_sha256"],
        "evaluator_sha256": proposal["evaluator_sha256"],
        "memory_journal_sha256": proposal["memory_journal_sha256"],
        "allowed_paths": allowed_paths,
        "issued_tick": int(authority["issued_tick"]),
        "expires_tick": int(authority["expires_tick"]),
        "max_uses": int(authority["max_uses"]),
        "nonce": canonical_sha256(
            {"token_id": spec["token_id"], "proposal_sha256": proposal["proposal_sha256"]}
        ),
    }
    return {"claims": claims, "signature": _token_signature(claims)}


def _validate_authority(
    config: dict[str, Any],
    proposal: dict[str, Any],
    token: dict[str, Any],
    state: dict[str, Any],
    source_identity: dict[str, Any],
    config_payload_sha256: str,
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
    tick: int,
) -> list[str]:
    problems: list[str] = []
    claims = token.get("claims")
    if not isinstance(claims, dict):
        return ["authority-claims-missing"]
    token_id = str(claims.get("token_id", ""))
    consumed = state.get("governance", {}).get("consumed_authority_tokens", [])
    if token_id in consumed:
        problems.append("authority-replayed")
    if claims.get("schema") != AUTHORITY_SCHEMA:
        problems.append("authority-schema-drift")
    if claims.get("key_id") != config["authority"]["key_id"]:
        problems.append("authority-key-drift")
    signature = token.get("signature")
    if not isinstance(signature, str) or not hmac.compare_digest(signature, _token_signature(claims)):
        problems.append("authority-forged")
    if int(claims.get("max_uses", 0)) != 1:
        problems.append("authority-use-limit-drift")
    issued = int(claims.get("issued_tick", -1))
    expires = int(claims.get("expires_tick", -1))
    if tick < issued:
        problems.append("authority-not-yet-valid")
    if tick > expires:
        problems.append("authority-expired")
    bindings = {
        "proposal_sha256": proposal.get("proposal_sha256"),
        "base_state_sha256": canonical_sha256(state),
        "source_identity_sha256": source_identity.get("sha256"),
        "config_payload_sha256": config_payload_sha256,
        "evaluator_sha256": evaluator.get("sha256"),
        "memory_journal_sha256": journal.sha256,
    }
    for key, expected in bindings.items():
        if claims.get(key) != expected:
            problems.append(f"authority-{key}-mismatch")
    expected_paths = [
        *config["authority"]["proposal_paths"],
        *config["authority"]["transaction_side_effect_paths"],
    ]
    if claims.get("allowed_paths") != expected_paths:
        problems.append("authority-scope-escalation")
    if _proposal_paths(proposal) != list(config["authority"]["proposal_paths"]):
        problems.append("proposal-outside-authority-scope")
    return sorted(set(problems))


def _apply_proposal(state: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    expected_sha = canonical_sha256(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    if proposal.get("proposal_sha256") != expected_sha:
        raise GovernedRewriteRefused("proposal-digest-drift")
    changes = proposal.get("changes")
    if not isinstance(changes, list) or len(changes) != 1:
        raise GovernedRewriteRefused("proposal-change-count-drift")
    change = changes[0]
    if change.get("path") != "policy.adaptation_threshold":
        raise GovernedRewriteRefused("proposal-path-outside-scope")
    if int(change.get("before", -1)) != int(state["policy"]["adaptation_threshold"]):
        raise GovernedRewriteRefused("proposal-before-value-conflict")
    candidate = copy.deepcopy(state)
    candidate["revision"] = int(state["revision"]) + 1
    candidate["policy"]["adaptation_threshold"] = int(change["after"])
    return candidate


def _append_audit(events: list[dict[str, Any]], phase: str, details: dict[str, Any]) -> None:
    core = {
        "sequence": len(events),
        "phase": phase,
        "details": copy.deepcopy(details),
        "previous_entry_sha256": events[-1]["entry_sha256"] if events else None,
    }
    events.append({**core, "entry_sha256": canonical_sha256(core)})


def _verify_audit(events: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    previous: str | None = None
    for index, entry in enumerate(events):
        core = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if entry.get("sequence") != index:
            problems.append(f"audit entry {index} sequence drift")
        if entry.get("previous_entry_sha256") != previous:
            problems.append(f"audit entry {index} predecessor drift")
        if entry.get("entry_sha256") != canonical_sha256(core):
            problems.append(f"audit entry {index} digest drift")
        previous = entry.get("entry_sha256")
    return problems


def _seal_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    core = {key: copy.deepcopy(value) for key, value in payload.items() if key != "checkpoint_sha256"}
    return {**core, "checkpoint_sha256": canonical_sha256(core)}


def _read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernedRewriteRefused(f"rewrite checkpoint unreadable: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise GovernedRewriteRefused("rewrite checkpoint schema drift")
    expected = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    if checkpoint.get("checkpoint_sha256") != expected:
        raise GovernedRewriteRefused("rewrite checkpoint digest drift")
    if _verify_audit(checkpoint.get("audit", [])):
        raise GovernedRewriteRefused("rewrite checkpoint audit chain drift")
    return checkpoint


def _attempt_result(
    *,
    status: str,
    reasons: list[str],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    events: list[dict[str, Any]],
    proposal: dict[str, Any],
    token: dict[str, Any],
    canary: dict[str, Any] | None = None,
    shadow: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    commit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_codes": reasons,
        "proposal_sha256": proposal.get("proposal_sha256"),
        "token_id": token.get("claims", {}).get("token_id"),
        "before_state_sha256": canonical_sha256(before_state),
        "after_state_sha256": canonical_sha256(after_state),
        "state_unchanged": canonical_sha256(before_state) == canonical_sha256(after_state),
        "canary": canary,
        "shadow": shadow,
        "resume": resume,
        "commit": commit,
        "audit": events,
        "audit_head_sha256": events[-1]["entry_sha256"] if events else None,
        "audit_chain_verified": not _verify_audit(events),
    }


def _complete_transaction(
    config: dict[str, Any],
    root: Path,
    before_state: dict[str, Any],
    candidate_state: dict[str, Any],
    proposal: dict[str, Any],
    token: dict[str, Any],
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
    events: list[dict[str, Any]],
    canary: dict[str, Any],
    *,
    tick: int,
    resume: dict[str, Any],
) -> dict[str, Any]:
    shadow = _evaluation_pair(config, "shadow", before_state, candidate_state, evaluator, journal, tick)
    _append_audit(
        events,
        "shadow-evaluated",
        {"evaluation_sha256": shadow["sha256"], "passed": shadow["gate"]["passed"]},
    )
    if not shadow["gate"]["passed"]:
        _append_audit(events, "rolled-back", {"reason": "shadow-regression-gate"})
        checkpoint_path = root / "transaction.checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        after = _read_state(root)
        return _attempt_result(
            status="rolled-back",
            reasons=["shadow-regression-gate"],
            before_state=before_state,
            after_state=after,
            events=events,
            proposal=proposal,
            token=token,
            canary=canary,
            shadow=shadow,
            resume=resume,
        )

    committed = copy.deepcopy(candidate_state)
    token_id = str(token["claims"]["token_id"])
    committed["governance"]["consumed_authority_tokens"].append(token_id)
    before_file_sha256 = _sha256_file(root / "state.json")
    write_receipt = _write_state(root, committed)
    reread = _read_state(root)
    atomic_ok = reread == committed and not (root / "state.json.tmp").exists()
    if not atomic_ok:
        raise GovernedRewriteRefused("atomic state commit verification failed")
    commit = {
        "method": "same-directory temporary plus os.replace",
        "before_file_sha256": before_file_sha256,
        "after_file_sha256": _sha256_file(root / "state.json"),
        "write_receipt": write_receipt,
        "committed_state_sha256": canonical_sha256(committed),
        "atomic_replace_verified": atomic_ok,
        "token_consumed_once": committed["governance"]["consumed_authority_tokens"].count(token_id) == 1,
    }
    _append_audit(
        events,
        "committed",
        {
            "state_sha256": commit["committed_state_sha256"],
            "token_id": token_id,
            "atomic_replace_verified": atomic_ok,
        },
    )
    checkpoint_path = root / "transaction.checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return _attempt_result(
        status="committed",
        reasons=[],
        before_state=before_state,
        after_state=committed,
        events=events,
        proposal=proposal,
        token=token,
        canary=canary,
        shadow=shadow,
        resume=resume,
        commit=commit,
    )


def _resume_transaction(
    config: dict[str, Any],
    root: Path,
    checkpoint: dict[str, Any],
    source_identity: dict[str, Any],
    config_payload_sha256: str,
    proposal: dict[str, Any],
    token: dict[str, Any],
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
    tick: int,
) -> dict[str, Any]:
    if checkpoint.get("phase") != "interrupted-after-canary":
        raise GovernedRewriteRefused("rewrite checkpoint phase is not resumable")
    exact_bindings = {
        "source_identity": source_identity,
        "config_payload_sha256": config_payload_sha256,
        "proposal": proposal,
        "authority_token": token,
        "evaluator": evaluator,
        "memory_journal": journal.payload(),
    }
    for key, expected in exact_bindings.items():
        if checkpoint.get(key) != expected:
            raise GovernedRewriteRefused(f"rewrite checkpoint {key} identity drift")
    before_state = checkpoint.get("before_state")
    candidate_state = checkpoint.get("candidate_state")
    canary = checkpoint.get("canary")
    if (
        not isinstance(before_state, dict)
        or not isinstance(candidate_state, dict)
        or not isinstance(canary, dict)
    ):
        raise GovernedRewriteRefused("rewrite checkpoint state or canary payload missing")
    current = _read_state(root)
    if current != before_state:
        raise GovernedRewriteRefused("rewrite checkpoint conflicts with committed state")
    authority_problems = _validate_authority(
        config,
        proposal,
        token,
        current,
        source_identity,
        config_payload_sha256,
        evaluator,
        journal,
        tick,
    )
    if authority_problems:
        raise GovernedRewriteRefused("rewrite resume authority failed: " + ",".join(authority_problems))
    expected_candidate = _apply_proposal(current, proposal)
    expected_canary = _evaluation_pair(
        config, "canary", current, expected_candidate, evaluator, journal, tick
    )
    if candidate_state != expected_candidate or canary != expected_canary:
        raise GovernedRewriteRefused("rewrite checkpoint deterministic replay drift")

    partial = root / "state.json.tmp"
    partial_expected = checkpoint.get("partial_write", {})
    partial_detected = partial.exists()
    partial_digest_ok = partial_detected and _sha256_file(partial) == partial_expected.get("sha256")
    if not partial_detected or not partial_digest_ok:
        raise GovernedRewriteRefused("rewrite partial-write recovery authority drift")
    partial.unlink()
    events = copy.deepcopy(checkpoint["audit"])
    _append_audit(
        events,
        "recovered",
        {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "partial_write_sha256": partial_expected["sha256"],
            "committed_base_reverified": True,
        },
    )
    resume = {
        "was_resumed": True,
        "interruption_phase": "after-canary-before-shadow",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "partial_write_detected": partial_detected,
        "partial_write_digest_verified": partial_digest_ok,
        "partial_write_removed": not partial.exists(),
        "base_state_reverified": current == before_state,
        "candidate_rebuilt_exactly": expected_candidate == candidate_state,
        "canary_rebuilt_exactly": expected_canary == canary,
    }
    return _complete_transaction(
        config,
        root,
        before_state,
        candidate_state,
        proposal,
        token,
        evaluator,
        journal,
        events,
        canary,
        tick=tick,
        resume=resume,
    )


def execute_transaction(
    config: dict[str, Any],
    root: Path,
    stored_state: dict[str, Any],
    source_identity: dict[str, Any],
    config_payload_sha256: str,
    proposal: dict[str, Any],
    token: dict[str, Any],
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
    *,
    tick: int,
    interrupt_after_canary: bool = False,
) -> dict[str, Any]:

    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    checkpoint_path = root / "transaction.checkpoint.json"
    if checkpoint_path.exists():
        return _resume_transaction(
            config,
            root,
            _read_checkpoint(checkpoint_path),
            source_identity,
            config_payload_sha256,
            proposal,
            token,
            evaluator,
            journal,
            tick,
        )
    if not state_path.exists():
        _write_state(root, stored_state)
    current = _read_state(root)
    events: list[dict[str, Any]] = []
    _append_audit(
        events,
        "proposal-received",
        {
            "proposal_sha256": proposal.get("proposal_sha256"),
            "current_state_sha256": canonical_sha256(current),
        },
    )

    claims = token.get("claims", {})
    token_id = str(claims.get("token_id", "")) if isinstance(claims, dict) else ""
    if token_id in current.get("governance", {}).get("consumed_authority_tokens", []):
        reasons = ["authority-replayed"]
        _append_audit(events, "refused", {"reason_codes": reasons})
        return _attempt_result(
            status="refused",
            reasons=reasons,
            before_state=current,
            after_state=_read_state(root),
            events=events,
            proposal=proposal,
            token=token,
        )
    if proposal.get("base_state_sha256") != canonical_sha256(current):
        reasons = ["state-conflict"]
        _append_audit(events, "conflict-detected", {"reason_codes": reasons})
        return _attempt_result(
            status="refused",
            reasons=reasons,
            before_state=current,
            after_state=_read_state(root),
            events=events,
            proposal=proposal,
            token=token,
        )

    reasons = _validate_authority(
        config,
        proposal,
        token,
        current,
        source_identity,
        config_payload_sha256,
        evaluator,
        journal,
        tick,
    )
    if journal.verify():
        reasons.append("memory-journal-invalid")
    memory_state = journal.state_at()
    if memory_state.poisoned:
        reasons.append("memory-poisoned")
    if memory_state.conflicted:
        reasons.append("memory-conflicted")
    if current.get("memory_binding") != _memory_binding(journal):
        reasons.append("memory-binding-drift")
    expected_evaluator = _evaluator_spec(config)
    if evaluator != expected_evaluator:
        reasons.append("evaluator-tampered")
    reasons = sorted(set(reasons))
    if reasons:
        _append_audit(events, "refused", {"reason_codes": reasons})
        return _attempt_result(
            status="refused",
            reasons=reasons,
            before_state=current,
            after_state=_read_state(root),
            events=events,
            proposal=proposal,
            token=token,
        )

    _append_audit(
        events,
        "authority-validated",
        {
            "token_id": token_id,
            "proposal_sha256": proposal["proposal_sha256"],
            "allowed_paths": token["claims"]["allowed_paths"],
        },
    )
    candidate = _apply_proposal(current, proposal)
    canary = _evaluation_pair(config, "canary", current, candidate, evaluator, journal, tick)
    _append_audit(
        events,
        "canary-evaluated",
        {"evaluation_sha256": canary["sha256"], "passed": canary["gate"]["passed"]},
    )
    if not canary["gate"]["passed"]:
        _append_audit(events, "rolled-back", {"reason": "canary-regression-gate"})
        after = _read_state(root)
        return _attempt_result(
            status="rolled-back",
            reasons=["canary-regression-gate"],
            before_state=current,
            after_state=after,
            events=events,
            proposal=proposal,
            token=token,
            canary=canary,
            resume={"was_resumed": False},
        )

    if interrupt_after_canary:
        candidate_envelope = _state_file_payload(candidate)
        full_raw = json.dumps(candidate_envelope, indent=2, sort_keys=True, allow_nan=False).encode()
        partial_raw = full_raw[: max(1, len(full_raw) // 2)]
        partial_receipt = {
            "bytes": len(partial_raw),
            "sha256": hashlib.sha256(partial_raw).hexdigest(),
            "valid_json": False,
            "committed_state_untouched": canonical_sha256(_read_state(root)) == canonical_sha256(current),
        }
        _append_audit(
            events,
            "interrupted",
            {
                "phase": "after-canary-before-shadow",
                "partial_write_sha256": partial_receipt["sha256"],
            },
        )
        checkpoint = _seal_checkpoint(
            {
                "schema": CHECKPOINT_SCHEMA,
                "phase": "interrupted-after-canary",
                "source_identity": source_identity,
                "config_payload_sha256": config_payload_sha256,
                "before_state": current,
                "candidate_state": candidate,
                "proposal": proposal,
                "authority_token": token,
                "evaluator": evaluator,
                "memory_journal": journal.payload(),
                "canary": canary,
                "partial_write": partial_receipt,
                "audit": events,
            }
        )
        _atomic_json(checkpoint_path, checkpoint)
        (root / "state.json.tmp").write_bytes(partial_raw)
        return _attempt_result(
            status="interrupted",
            reasons=["planned-interrupt-after-canary"],
            before_state=current,
            after_state=_read_state(root),
            events=events,
            proposal=proposal,
            token=token,
            canary=canary,
            resume={
                "was_resumed": False,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "partial_write": partial_receipt,
            },
        )

    return _complete_transaction(
        config,
        root,
        current,
        candidate,
        proposal,
        token,
        evaluator,
        journal,
        events,
        canary,
        tick=tick,
        resume={
            "was_resumed": False,
            "interruption_phase": None,
            "partial_write_detected": False,
        },
    )


def _full_transaction_record(
    attempt: dict[str, Any],
    config_payload_sha256: str,
    source_identity: dict[str, Any],
    before_state: dict[str, Any],
    proposal: dict[str, Any],
    token: dict[str, Any],
    evaluator: dict[str, Any],
    journal: LifecycleJournal,
) -> dict[str, Any]:
    return {
        "schema": TRANSACTION_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "config_payload_sha256": config_payload_sha256,
        "source_identity": source_identity,
        "before_state": before_state,
        "proposal": proposal,
        "authority_token": token,
        "evaluator": evaluator,
        "memory_journal": journal.payload(),
        **copy.deepcopy(attempt),
        "scientific_promotion_allowed": False,
    }


def _journal_payload_matches(journal: LifecycleJournal, payload: Any) -> bool:
    return isinstance(payload, dict) and payload == journal.payload()


def verify_transaction_record(
    record: dict[str, Any],
    config: dict[str, Any],
    source_identity: dict[str, Any],
    config_payload_sha256: str,
) -> dict[str, Any]:

    problems: list[str] = []
    if record.get("schema") != TRANSACTION_SCHEMA:
        problems.append("transaction schema drift")
    if record.get("claim_scope") != CLAIM_SCOPE:
        problems.append("transaction claim scope drift")
    if record.get("scientific_promotion_allowed") is not False:
        problems.append("transaction illegally permits scientific promotion")
    if record.get("config_payload_sha256") != config_payload_sha256:
        problems.append("transaction config identity drift")
    if record.get("source_identity") != source_identity:
        problems.append("transaction source identity drift")

    evaluator = _evaluator_spec(config)
    clean_journal = build_memory_journal(config)
    before_state = _initial_state(config, clean_journal, evaluator)
    proposal = build_proposal(
        config,
        before_state,
        source_identity,
        config_payload_sha256,
        evaluator,
        clean_journal,
    )
    token = issue_authority_token(config, proposal)
    if record.get("before_state") != before_state:
        problems.append("transaction before-state identity drift")
    if record.get("proposal") != proposal:
        problems.append("transaction proposal identity drift")
    if record.get("authority_token") != token:
        problems.append("transaction authority identity drift")
    if record.get("evaluator") != evaluator:
        problems.append("transaction evaluator identity drift")
    if not _journal_payload_matches(clean_journal, record.get("memory_journal")):
        problems.append("transaction memory journal identity drift")
    if _validate_authority(
        config,
        proposal,
        token,
        before_state,
        source_identity,
        config_payload_sha256,
        evaluator,
        clean_journal,
        int(config["authority"]["execution_tick"]),
    ):
        problems.append("independent authority reconstruction failed")

    candidate = _apply_proposal(before_state, proposal)
    canary = _evaluation_pair(
        config,
        "canary",
        before_state,
        candidate,
        evaluator,
        clean_journal,
        int(config["authority"]["execution_tick"]),
    )
    shadow = _evaluation_pair(
        config,
        "shadow",
        before_state,
        candidate,
        evaluator,
        clean_journal,
        int(config["authority"]["execution_tick"]),
    )
    if record.get("canary") != canary:
        problems.append("transaction canary reconstruction drift")
    if record.get("shadow") != shadow:
        problems.append("transaction shadow reconstruction drift")
    if not canary["gate"]["passed"] or not shadow["gate"]["passed"]:
        problems.append("independent capability or regression gate failed")

    committed = copy.deepcopy(candidate)
    committed["governance"]["consumed_authority_tokens"].append(token["claims"]["token_id"])
    if record.get("status") != "committed":
        problems.append("transaction is not committed")
    if record.get("reason_codes") != []:
        problems.append("committed transaction has refusal reasons")
    if record.get("after_state_sha256") != canonical_sha256(committed):
        problems.append("transaction committed-state digest drift")
    if record.get("state_unchanged") is not False:
        problems.append("committed transaction incorrectly reports unchanged state")
    commit = record.get("commit")
    if not isinstance(commit, dict):
        problems.append("transaction commit receipt missing")
    else:
        if commit.get("committed_state_sha256") != canonical_sha256(committed):
            problems.append("commit receipt state digest drift")
        if commit.get("atomic_replace_verified") is not True:
            problems.append("atomic replacement was not verified")
        if commit.get("token_consumed_once") is not True:
            problems.append("authority consumption was not single use")
        write = commit.get("write_receipt", {})
        if write.get("temporary_absent_after_replace") is not True:
            problems.append("atomic temporary survived commit")

    events = record.get("audit")
    if not isinstance(events, list):
        problems.append("transaction audit missing")
    else:
        problems.extend(_verify_audit(events))
        phases = [str(entry.get("phase")) for entry in events]
        required = {
            "proposal-received",
            "authority-validated",
            "canary-evaluated",
            "shadow-evaluated",
            "committed",
        }
        if not required.issubset(phases):
            problems.append("transaction audit omits required phases")
        by_phase = {str(entry.get("phase")): entry.get("details", {}) for entry in events}
        expected_details = {
            "proposal-received": {
                "proposal_sha256": proposal["proposal_sha256"],
                "current_state_sha256": canonical_sha256(before_state),
            },
            "authority-validated": {
                "token_id": token["claims"]["token_id"],
                "proposal_sha256": proposal["proposal_sha256"],
                "allowed_paths": token["claims"]["allowed_paths"],
            },
            "canary-evaluated": {
                "evaluation_sha256": canary["sha256"],
                "passed": True,
            },
            "shadow-evaluated": {
                "evaluation_sha256": shadow["sha256"],
                "passed": True,
            },
            "committed": {
                "state_sha256": canonical_sha256(committed),
                "token_id": token["claims"]["token_id"],
                "atomic_replace_verified": True,
            },
        }
        for phase, expected in expected_details.items():
            if by_phase.get(phase) != expected:
                problems.append(f"transaction audit {phase} semantics drift")
        if record.get("audit_head_sha256") != (events[-1]["entry_sha256"] if events else None):
            problems.append("transaction audit head drift")
        if record.get("audit_chain_verified") is not True:
            problems.append("transaction audit self-verification is false")

    resume = record.get("resume")
    if not isinstance(resume, dict) or resume.get("was_resumed") is not True:
        problems.append("canonical transaction was not resumed")
    else:
        for field in (
            "partial_write_detected",
            "partial_write_digest_verified",
            "partial_write_removed",
            "base_state_reverified",
            "candidate_rebuilt_exactly",
            "canary_rebuilt_exactly",
        ):
            if resume.get(field) is not True:
                problems.append(f"resume invariant {field} failed")
        if not _is_sha256(resume.get("checkpoint_sha256")):
            problems.append("resume checkpoint identity missing")
    return {"verified": not problems, "problems": problems}


def _reseal_audit(events: list[dict[str, Any]]) -> None:
    previous: str | None = None
    for index, entry in enumerate(events):
        entry["sequence"] = index
        entry["previous_entry_sha256"] = previous
        core = {key: value for key, value in entry.items() if key != "entry_sha256"}
        entry["entry_sha256"] = canonical_sha256(core)
        previous = entry["entry_sha256"]


def mutation_suite(
    record: dict[str, Any],
    config: dict[str, Any],
    source_identity: dict[str, Any],
    config_payload_sha256: str,
) -> dict[str, Any]:

    mutations: dict[str, Any] = {}

    def run(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(record)
        mutate(candidate)
        verdict = verify_transaction_record(candidate, config, source_identity, config_payload_sha256)
        mutations[name] = {
            "rejected": verdict["verified"] is False,
            "problems": verdict["problems"][:4],
        }

    run(
        "source_identity",
        lambda value: value["source_identity"].__setitem__("sha256", "0" * 64),
    )
    run(
        "base_state",
        lambda value: value["before_state"]["policy"].__setitem__("protected_threshold", 4),
    )
    run(
        "proposal_change",
        lambda value: value["proposal"]["changes"][0].__setitem__("after", 1),
    )
    run(
        "authority_signature",
        lambda value: value["authority_token"].__setitem__("signature", "f" * 64),
    )

    def escalate(value: dict[str, Any]) -> None:
        claims = value["authority_token"]["claims"]
        claims["allowed_paths"].append("policy.protected_threshold")
        value["authority_token"]["signature"] = _token_signature(claims)

    run("authority_scope_with_valid_signature", escalate)
    run(
        "evaluator_rule",
        lambda value: value["evaluator"].__setitem__("comparison", "greater-than"),
    )
    run(
        "memory_content",
        lambda value: value["memory_journal"]["entries"][0]["content"]["value"].__setitem__(
            "calibration_offset", 7
        ),
    )
    run(
        "canary_metric",
        lambda value: value["canary"]["after"].__setitem__("correct", 7),
    )
    run(
        "shadow_metric",
        lambda value: value["shadow"]["gate"].__setitem__("passed", False),
    )
    run(
        "committed_state",
        lambda value: value.__setitem__("after_state_sha256", "1" * 64),
    )

    def audit_mutation(value: dict[str, Any]) -> None:
        value["audit"][2]["details"]["passed"] = False
        _reseal_audit(value["audit"])
        value["audit_head_sha256"] = value["audit"][-1]["entry_sha256"]

    run("audit_semantics_with_valid_chain", audit_mutation)
    run(
        "atomic_commit_receipt",
        lambda value: value["commit"].__setitem__("atomic_replace_verified", False),
    )
    run(
        "resume_provenance",
        lambda value: value["resume"].__setitem__("candidate_rebuilt_exactly", False),
    )
    run(
        "claim_boundary",
        lambda value: value.__setitem__("scientific_promotion_allowed", True),
    )
    rejected = sum(bool(value["rejected"]) for value in mutations.values())
    return {
        "count": len(mutations),
        "rejected": rejected,
        "all_rejected": rejected == len(mutations),
        "mutations": mutations,
    }


def _attempt_summary(attempt: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": attempt["status"],
        "reason_codes": attempt["reason_codes"],
        "state_unchanged": attempt["state_unchanged"],
        "before_state_sha256": attempt["before_state_sha256"],
        "after_state_sha256": attempt["after_state_sha256"],
        "audit_chain_verified": attempt["audit_chain_verified"],
        "audit_head_sha256": attempt["audit_head_sha256"],
    }
    if attempt.get("canary") is not None:
        summary["canary_gate"] = copy.deepcopy(attempt["canary"]["gate"])
    return summary


def _run_drills(
    config: dict[str, Any],
    scratch: Path,
    source_identity: dict[str, Any],
    config_payload_sha256: str,
) -> dict[str, Any]:
    evaluator = _evaluator_spec(config)
    clean_journal = build_memory_journal(config)
    before_state = _initial_state(config, clean_journal, evaluator)
    proposal = build_proposal(
        config,
        before_state,
        source_identity,
        config_payload_sha256,
        evaluator,
        clean_journal,
    )
    token = issue_authority_token(config, proposal)
    tick = int(config["authority"]["execution_tick"])

    uninterrupted = execute_transaction(
        config,
        scratch / "uninterrupted",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        evaluator,
        clean_journal,
        tick=tick,
    )
    interrupted = execute_transaction(
        config,
        scratch / "interrupted",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        evaluator,
        clean_journal,
        tick=tick,
        interrupt_after_canary=True,
    )
    resumed = execute_transaction(
        config,
        scratch / "interrupted",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        evaluator,
        clean_journal,
        tick=tick,
    )
    canonical = _full_transaction_record(
        resumed,
        config_payload_sha256,
        source_identity,
        before_state,
        proposal,
        token,
        evaluator,
        clean_journal,
    )
    exact_resume = {
        "interruption_reached": interrupted["status"] == "interrupted",
        "resumed_committed": resumed["status"] == "committed",
        "uninterrupted_committed": uninterrupted["status"] == "committed",
        "committed_state_exact": resumed["after_state_sha256"] == uninterrupted["after_state_sha256"],
        "canary_exact": canonical_sha256(resumed["canary"]) == canonical_sha256(uninterrupted["canary"]),
        "shadow_exact": canonical_sha256(resumed["shadow"]) == canonical_sha256(uninterrupted["shadow"]),
        "partial_write_never_became_committed_state": interrupted["state_unchanged"] is True,
    }
    exact_resume["all_ok"] = all(exact_resume.values())

    forged = copy.deepcopy(token)
    forged["signature"] = ("0" if token["signature"][0] != "0" else "1") + token["signature"][1:]
    forged_attempt = execute_transaction(
        config,
        scratch / "forged",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        forged,
        evaluator,
        clean_journal,
        tick=tick,
    )
    expired_attempt = execute_transaction(
        config,
        scratch / "expired",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        evaluator,
        clean_journal,
        tick=int(config["authority"]["expires_tick"]) + 1,
    )
    conflicting_state = copy.deepcopy(before_state)
    conflicting_state["policy"]["protected_threshold"] = 4
    conflict_attempt = execute_transaction(
        config,
        scratch / "conflict",
        conflicting_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        evaluator,
        clean_journal,
        tick=tick,
    )
    tampered_evaluator = copy.deepcopy(evaluator)
    tampered_evaluator["comparison"] = "greater-than"
    tampered_evaluator["sha256"] = canonical_sha256(
        {key: value for key, value in tampered_evaluator.items() if key != "sha256"}
    )
    evaluator_attempt = execute_transaction(
        config,
        scratch / "evaluator",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        tampered_evaluator,
        clean_journal,
        tick=tick,
    )
    poisoned_journal = build_memory_journal(config, poisoned=True)
    poison_attempt = execute_transaction(
        config,
        scratch / "poison",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        evaluator,
        poisoned_journal,
        tick=tick,
    )
    rollback_proposal = build_proposal(
        config,
        before_state,
        source_identity,
        config_payload_sha256,
        evaluator,
        clean_journal,
        rollback_probe=True,
    )
    rollback_token = issue_authority_token(config, rollback_proposal, rollback_probe=True)
    rollback_attempt = execute_transaction(
        config,
        scratch / "rollback",
        before_state,
        source_identity,
        config_payload_sha256,
        rollback_proposal,
        rollback_token,
        evaluator,
        clean_journal,
        tick=tick,
    )
    replay_state = _read_state(scratch / "interrupted")
    replay_attempt = execute_transaction(
        config,
        scratch / "replay",
        replay_state,
        source_identity,
        config_payload_sha256,
        proposal,
        token,
        evaluator,
        clean_journal,
        tick=tick,
    )
    escalated = copy.deepcopy(token)
    escalated["claims"]["allowed_paths"].append("policy.protected_threshold")
    escalated["signature"] = _token_signature(escalated["claims"])
    scope_attempt = execute_transaction(
        config,
        scratch / "scope",
        before_state,
        source_identity,
        config_payload_sha256,
        proposal,
        escalated,
        evaluator,
        clean_journal,
        tick=tick,
    )

    attempts = {
        "forged_authority": _attempt_summary(forged_attempt),
        "expired_authority": _attempt_summary(expired_attempt),
        "replayed_authority": _attempt_summary(replay_attempt),
        "scope_escalation_with_valid_signature": _attempt_summary(scope_attempt),
        "conflicting_base_state": _attempt_summary(conflict_attempt),
        "tampered_evaluator": _attempt_summary(evaluator_attempt),
        "poisoned_memory": _attempt_summary(poison_attempt),
        "authorized_regressing_proposal": _attempt_summary(rollback_attempt),
    }
    independent = verify_transaction_record(canonical, config, source_identity, config_payload_sha256)
    mutations = mutation_suite(canonical, config, source_identity, config_payload_sha256)
    return {
        "initial_state_identity": _state_identity(before_state),
        "memory_journal": clean_journal.payload(),
        "evaluator": evaluator,
        "proposal": proposal,
        "authority_token": token,
        "canonical_transaction": canonical,
        "uninterrupted_reference": _attempt_summary(uninterrupted),
        "interrupted_reference": _attempt_summary(interrupted),
        "exact_resume": exact_resume,
        "adversarial_attempts": attempts,
        "independent_verifier": independent,
        "mutation_suite": mutations,
    }


def _audit_existing() -> dict[str, Any]:
    return {
        "files": [_file_receipt(REPO_ROOT / path) for path in AUDITED_UPSTREAM_PATHS],
        "already_owned": {
            "f8": (
                "content-hashed scientific rewrite packages, learned and control arms, matched estimated "
                "compute, refusal receipts, and a fixture-tainted nonpromotion boundary"
            ),
            "f20": (
                "prospective insufficiency scoring against raw-error, confidence, random, and aleatoric "
                "noise controls; its durable fixture result remains a null with zero measured avoided compute"
            ),
            "lifecycle": (
                "immutable append-only revision, conflict, poisoning, availability, deletion, and rollback "
                "mechanics with exact hash-linked replay"
            ),
            "provenance": "run and cache environment stamping with explicit evidence tags",
            "local_governor": (
                "resource admission, pause, stop, and atomic resume authority for long local processes"
            ),
        },
        "gap_closed": (
            "none of the audited surfaces issued scoped single-use authority for a state rewrite or joined "
            "proposal, immutable base identity, canary, shadow, commit, rollback, and crash recovery "
            "into one "
            "project-owned transaction"
        ),
        "nonduplication": [
            "FrozenJSON and canonical_sha256 provide immutable identity",
            "LifecycleJournal supplies availability and poisoned-memory quarantine semantics",
            "the new module adds only rewrite-specific authority, evaluation, transaction, and verification",
        ],
    }


def _deterministic_part(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key not in {"resource_observation", "deterministic_core_sha256"}
    }


def build_preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    started = time.perf_counter()
    with PeakRSSMonitor() as rss_monitor:
        source_identity = _source_identity()
        config_payload_sha256 = canonical_sha256(config)
        with tempfile.TemporaryDirectory(prefix="mop-governed-rewrite-") as scratch_name:
            drills = _run_drills(config, Path(scratch_name), source_identity, config_payload_sha256)
    elapsed = time.perf_counter() - started
    max_rss = rss_monitor.peak_rss_bytes
    envelope = config["resource_envelope"]
    attempts = drills["adversarial_attempts"]
    expected_attempts = {
        "forged_authority": ("refused", "authority-forged"),
        "expired_authority": ("refused", "authority-expired"),
        "replayed_authority": ("refused", "authority-replayed"),
        "scope_escalation_with_valid_signature": ("refused", "authority-scope-escalation"),
        "conflicting_base_state": ("refused", "state-conflict"),
        "tampered_evaluator": ("refused", "evaluator-tampered"),
        "poisoned_memory": ("refused", "memory-poisoned"),
        "authorized_regressing_proposal": ("rolled-back", "canary-regression-gate"),
    }
    adversarial_ok = all(
        attempts[name]["status"] == status
        and reason in attempts[name]["reason_codes"]
        and attempts[name]["state_unchanged"] is True
        and attempts[name]["audit_chain_verified"] is True
        for name, (status, reason) in expected_attempts.items()
    )
    canonical = drills["canonical_transaction"]
    checks = {
        "source_identity_immutable": _is_sha256(source_identity["sha256"]),
        "config_identity_immutable": _is_sha256(config_payload_sha256),
        "state_identity_immutable": _is_sha256(drills["initial_state_identity"]["sha256"]),
        "proposal_scoped_to_one_policy_path": _proposal_paths(drills["proposal"])
        == list(config["authority"]["proposal_paths"]),
        "single_use_authority_committed": canonical["commit"]["token_consumed_once"] is True,
        "canary_passed": canonical["canary"]["gate"]["passed"] is True,
        "shadow_passed": canonical["shadow"]["gate"]["passed"] is True,
        "before_after_capability_recorded": canonical["canary"]["gate"]["capability_gain"] == 1
        and canonical["shadow"]["gate"]["capability_gain"] == 3,
        "protected_regressions_zero": canonical["canary"]["gate"]["protected_regressions"] == []
        and canonical["shadow"]["gate"]["protected_regressions"] == [],
        "atomic_commit_verified": canonical["commit"]["atomic_replace_verified"] is True,
        "exact_interrupted_resume": drills["exact_resume"]["all_ok"] is True,
        "all_adversarial_attempts_fail_closed": adversarial_ok,
        "independent_verifier_passed": drills["independent_verifier"]["verified"] is True,
        "all_mutations_rejected": drills["mutation_suite"]["all_rejected"] is True,
        "minimum_adversarial_attempts": len(attempts)
        >= int(config["stop_contract"]["minimum_adversarial_attempts"]),
        "minimum_mutations": drills["mutation_suite"]["count"]
        >= int(config["stop_contract"]["minimum_mutations"]),
        "resource_wall_envelope": elapsed <= float(envelope["maximum_wall_seconds"]),
        "resource_rss_envelope": rss_monitor.peak_increment_bytes <= int(envelope["maximum_rss_bytes"]),
        "resource_rss_sampling_complete": rss_monitor.all_ok,
        "no_weights_downloads_or_external_data": envelope["model_weights_loaded"] is False
        and envelope["model_downloads_allowed"] is False
        and envelope["external_data_allowed"] is False,
        "mechanics_claim_only": canonical["scientific_promotion_allowed"] is False,
    }
    core: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "status": "mechanics-pass" if all(checks.values()) else "mechanics-fail",
        "null_hypothesis": config["null_hypothesis"],
        "audit": _audit_existing(),
        "config": {
            "path": _file_receipt(config_path)["path"],
            "sha256": _sha256_file(config_path),
            "payload_sha256": config_payload_sha256,
            "payload": config,
        },
        "source_identity": source_identity,
        "drills": drills,
        "checks": checks,
        "claim_boundary": {
            "mechanics_only": True,
            "natural_data": False,
            "learned_rewrite": False,
            "general_capability": False,
            "cognition_or_sentience": False,
            "production_security_or_safety": False,
            "scientific_promotion_allowed": False,
            "remaining_evidence_gate": (
                "an externally governed production authority root, independently specified evaluators, "
                "rights-clean natural workloads, preregistered protected capabilities, real candidate "
                "artifacts, multi-process durability tests, and replicated post-commit surveillance"
            ),
            "hardware_boundary": (
                "none for transaction mechanics; data, authority, evaluator independence, and external "
                "validity are the current blockers"
            ),
        },
        "implementation": [
            _file_receipt(REPO_ROOT / path) for path in (*IMPLEMENTATION_PATHS, *REUSED_PRIMITIVE_PATHS)
        ],
        "all_mechanics_ok": all(checks.values()),
    }
    core["deterministic_core_sha256"] = canonical_sha256(core)
    core["resource_observation"] = {
        "device": "cpu",
        "cpu_threads": 1,
        "wall_seconds": elapsed,
        "maximum_rss_bytes": max_rss,
        "phase_local_peak_rss_increment_bytes": rss_monitor.peak_increment_bytes,
        "rss_limit_scope": "phase-local sampled peak increment above phase-start RSS",
        "rss_measurement": rss_monitor.receipt(),
        "wall_limit_seconds": float(envelope["maximum_wall_seconds"]),
        "rss_limit_bytes": int(envelope["maximum_rss_bytes"]),
        "model_weights_loaded": False,
        "downloads_attempted": False,
        "external_data_loaded": False,
        "platform": platform.platform(),
    }
    return core


def verify_preflight_receipt(receipt: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    expected_self_hash = canonical_sha256(_deterministic_part(receipt))
    rebuilt = build_preflight(config_path)
    checks = {
        "schema": receipt.get("schema") == PREFLIGHT_SCHEMA,
        "self_hash": receipt.get("deterministic_core_sha256") == expected_self_hash,
        "exact_deterministic_rebuild": _deterministic_part(receipt) == _deterministic_part(rebuilt),
        "mechanics_pass": receipt.get("status") == "mechanics-pass"
        and receipt.get("all_mechanics_ok") is True,
        "scientific_promotion_blocked": receipt.get("claim_boundary", {}).get("scientific_promotion_allowed")
        is False,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "expected_deterministic_core_sha256": expected_self_hash,
        "rebuilt_deterministic_core_sha256": rebuilt["deterministic_core_sha256"],
    }


def write_preflight(config_path: Path = DEFAULT_CONFIG, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    receipt = build_preflight(config_path)
    _atomic_json(output_path, receipt)
    return receipt
