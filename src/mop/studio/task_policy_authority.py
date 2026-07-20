
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

TASK_POLICY_AUTHORITY_SCHEMA = "mop-local-throttle-task-policy-authority/v1"
POLICY_BASELINE_MANIFEST_SCHEMA = "mop-local-throttle-policy-baseline/v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build_policy_safety_contract(
    *,
    profile: Mapping[str, Any],
    limits: Mapping[str, Any],
    monitor: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:

    monitor_payload = dict(monitor)
    monitor_payload.pop("foreground_markers", None)
    monitor_payload.pop("known_heavy_markers", None)
    return {
        "profile": dict(profile),
        "limits": dict(limits),
        "monitor": monitor_payload,
        "thresholds": dict(thresholds),
    }


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from exc
    return value


def _strings(values: Sequence[str], label: str) -> list[str]:
    rows = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in rows):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} must not contain duplicates")
    return sorted(rows)


def build_task_policy_authority(
    *,
    policy_schema: str,
    policy_path: str,
    full_policy_sha256: str,
    profile_name: str,
    safety_contract: Mapping[str, Any],
    foreground_markers: Sequence[str],
    known_heavy_markers: Sequence[str],
    task_id: str,
    task_payload: Mapping[str, Any],
) -> dict[str, Any]:

    for value, label in (
        (policy_schema, "policy schema"),
        (policy_path, "policy path"),
        (profile_name, "profile name"),
        (task_id, "task id"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must not be empty")
    full_digest = _digest(full_policy_sha256, "full policy SHA-256")
    safety = dict(safety_contract)
    task = dict(task_payload)
    payload: dict[str, Any] = {
        "schema": TASK_POLICY_AUTHORITY_SCHEMA,
        "policy_schema": policy_schema,
        "policy_path": policy_path,
        "full_policy_sha256": full_digest,
        "profile_name": profile_name,
        "safety_contract": safety,
        "safety_contract_sha256": canonical_sha256(safety),
        "foreground_markers": _strings(foreground_markers, "foreground markers"),
        "known_heavy_markers": _strings(known_heavy_markers, "known-heavy markers"),
        "task_id": task_id,
        "task_sha256": canonical_sha256(task),
        "scientific_promotion": False,
    }
    payload["authority_sha256"] = canonical_sha256(payload)
    return payload


def task_policy_authority_problems(
    authority: object,
    *,
    policy_schema: str,
    policy_path: str,
    full_policy_sha256: str,
    profile_name: str,
    safety_contract: Mapping[str, Any],
    foreground_markers: Sequence[str],
    known_heavy_markers: Sequence[str],
    task_id: str,
    task_payload: Mapping[str, Any],
) -> list[str]:

    if not isinstance(authority, dict):
        return ["task-policy authority is not an object"]
    required = {
        "schema",
        "policy_schema",
        "policy_path",
        "full_policy_sha256",
        "profile_name",
        "safety_contract",
        "safety_contract_sha256",
        "foreground_markers",
        "known_heavy_markers",
        "task_id",
        "task_sha256",
        "scientific_promotion",
        "authority_sha256",
    }
    problems: list[str] = []
    if set(authority) != required:
        problems.append("task-policy authority schema fields drifted")
        return problems
    if authority["schema"] != TASK_POLICY_AUTHORITY_SCHEMA:
        problems.append("task-policy authority schema is unsupported")
    declared_seal = authority.get("authority_sha256")
    unsealed = dict(authority)
    unsealed.pop("authority_sha256", None)
    if declared_seal != canonical_sha256(unsealed):
        problems.append("task-policy authority self-seal mismatch")
    if authority.get("policy_schema") != policy_schema:
        problems.append("task-policy policy schema drifted")
    if authority.get("policy_path") != policy_path:
        problems.append("task-policy policy path drifted")
    try:
        _digest(full_policy_sha256, "current full policy SHA-256")
    except ValueError:
        problems.append("current full policy digest is invalid")
    if authority.get("profile_name") != profile_name:
        problems.append("task-policy profile drifted")
    current_safety = dict(safety_contract)
    if authority.get("safety_contract") != current_safety:
        problems.append("task-policy safety contract drifted")
    if authority.get("safety_contract_sha256") != canonical_sha256(current_safety):
        problems.append("task-policy safety digest drifted")
    if authority.get("task_id") != task_id:
        problems.append("task-policy task id drifted")
    if authority.get("task_sha256") != canonical_sha256(dict(task_payload)):
        problems.append("task-policy task declaration drifted")
    if authority.get("scientific_promotion") is not False:
        problems.append("task-policy authority cannot grant scientific promotion")

    for field, current_values in (
        ("foreground_markers", foreground_markers),
        ("known_heavy_markers", known_heavy_markers),
    ):
        historical = authority.get(field)
        if not isinstance(historical, list) or not all(isinstance(row, str) for row in historical):
            problems.append(f"task-policy {field} are invalid")
            continue
        if len(historical) != len(set(historical)) or historical != sorted(historical):
            problems.append(f"task-policy {field} are not canonical")
            continue
        if not set(historical) <= set(current_values):
            problems.append(f"task-policy {field} were removed")
    try:
        _digest(authority.get("full_policy_sha256"), "full policy SHA-256")
    except ValueError:
        problems.append("task-policy full policy digest is invalid")
    return problems


def build_policy_baseline_manifest(
    *,
    policy_schema: str,
    policy_path: str,
    full_policy_sha256: str,
    governor_implementation_path: str,
    governor_implementation_sha256: str,
    profile_name: str,
    safety_contract: Mapping[str, Any],
    foreground_markers: Sequence[str],
    known_heavy_markers: Sequence[str],
    task_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:

    _digest(governor_implementation_sha256, "governor implementation SHA-256")
    if not governor_implementation_path.strip():
        raise ValueError("governor implementation path must not be empty")
    if not task_payloads:
        raise ValueError("policy baseline requires at least one task")
    authorities = [
        build_task_policy_authority(
            policy_schema=policy_schema,
            policy_path=policy_path,
            full_policy_sha256=full_policy_sha256,
            profile_name=profile_name,
            safety_contract=safety_contract,
            foreground_markers=foreground_markers,
            known_heavy_markers=known_heavy_markers,
            task_id=task_id,
            task_payload=task_payloads[task_id],
        )
        for task_id in sorted(task_payloads)
    ]
    core: dict[str, Any] = {
        "schema": POLICY_BASELINE_MANIFEST_SCHEMA,
        "policy": {"path": policy_path, "sha256": _digest(full_policy_sha256, "policy SHA-256")},
        "governor_implementation": {
            "path": governor_implementation_path,
            "sha256": governor_implementation_sha256,
        },
        "profile_name": profile_name,
        "safety_contract_sha256": canonical_sha256(dict(safety_contract)),
        "task_authorities": authorities,
        "scientific_promotion": False,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def policy_baseline_manifest_problems(manifest: object) -> list[str]:

    if not isinstance(manifest, dict):
        return ["policy baseline manifest is not an object"]
    expected = {
        "schema",
        "policy",
        "governor_implementation",
        "profile_name",
        "safety_contract_sha256",
        "task_authorities",
        "scientific_promotion",
        "manifest_sha256",
    }
    if set(manifest) != expected:
        return ["policy baseline manifest fields drifted"]
    problems: list[str] = []
    if manifest.get("schema") != POLICY_BASELINE_MANIFEST_SCHEMA:
        problems.append("policy baseline schema is unsupported")
    core = dict(manifest)
    declared = core.pop("manifest_sha256", None)
    if declared != canonical_sha256(core):
        problems.append("policy baseline self-seal mismatch")
    for label in ("policy", "governor_implementation"):
        binding = manifest.get(label)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            problems.append(f"policy baseline {label} binding is invalid")
            continue
        try:
            _digest(binding.get("sha256"), f"{label} SHA-256")
        except ValueError:
            problems.append(f"policy baseline {label} digest is invalid")
    try:
        _digest(manifest.get("safety_contract_sha256"), "safety contract SHA-256")
    except ValueError:
        problems.append("policy baseline safety digest is invalid")
    authorities = manifest.get("task_authorities")
    if not isinstance(authorities, list) or not authorities:
        problems.append("policy baseline task authorities are missing")
    else:
        task_ids: list[str] = []
        for authority in authorities:
            if not isinstance(authority, dict):
                problems.append("policy baseline task authority is invalid")
                continue
            unsealed = dict(authority)
            authority_sha = unsealed.pop("authority_sha256", None)
            if authority_sha != canonical_sha256(unsealed):
                problems.append("policy baseline task authority self-seal mismatch")
            task_ids.append(str(authority.get("task_id", "")))
        if task_ids != sorted(task_ids) or len(task_ids) != len(set(task_ids)):
            problems.append("policy baseline task authorities are not unique and canonical")
    if manifest.get("scientific_promotion") is not False:
        problems.append("policy baseline cannot grant scientific promotion")
    return problems


def receipt_task_policy_authority_problems(
    *,
    declared_policy: object,
    declared_implementation: object,
    declared_task_id: str,
    declared_task_payload: Mapping[str, Any],
    embedded_authority: object,
    legacy_manifests: Sequence[object],
    current_policy_schema: str,
    current_policy_path: str,
    current_full_policy_sha256: str,
    current_profile_name: str,
    current_safety_contract: Mapping[str, Any],
    current_foreground_markers: Sequence[str],
    current_known_heavy_markers: Sequence[str],
    current_task_payload: Mapping[str, Any],
) -> list[str]:

    problems: list[str] = []
    if not isinstance(declared_policy, dict) or set(declared_policy) != {"path", "sha256"}:
        return ["receipt policy binding is invalid"]
    if declared_policy.get("path") != current_policy_path:
        problems.append("receipt policy path drifted")
    try:
        _digest(declared_policy.get("sha256"), "receipt policy SHA-256")
    except ValueError:
        problems.append("receipt policy digest is invalid")
    if not isinstance(declared_implementation, dict) or set(declared_implementation) != {
        "path",
        "sha256",
    }:
        return [*problems, "receipt implementation binding is invalid"]
    try:
        _digest(declared_implementation.get("sha256"), "receipt implementation SHA-256")
    except ValueError:
        problems.append("receipt implementation digest is invalid")

    selected: object = embedded_authority
    if embedded_authority is None:
        matches: list[dict[str, Any]] = []
        for candidate in legacy_manifests:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("policy") != declared_policy:
                continue
            if candidate.get("governor_implementation") != declared_implementation:
                continue
            matches.append(candidate)
        if len(matches) != 1:
            return [*problems, f"receipt maps to {len(matches)} reviewed legacy policy baselines"]
        manifest = matches[0]
        problems.extend(policy_baseline_manifest_problems(manifest))
        authorities = manifest.get("task_authorities")
        rows = (
            [row for row in authorities if isinstance(row, dict) and row.get("task_id") == declared_task_id]
            if isinstance(authorities, list)
            else []
        )
        if len(rows) != 1:
            return [*problems, f"receipt maps to {len(rows)} legacy task authorities"]
        selected = rows[0]
    if not isinstance(selected, dict):
        return [*problems, "receipt task-policy authority is missing"]
    if selected.get("full_policy_sha256") != declared_policy.get("sha256"):
        problems.append("receipt task-policy/full-policy binding drifted")
    if selected.get("policy_path") != declared_policy.get("path"):
        problems.append("receipt task-policy path binding drifted")
    if selected.get("task_id") != declared_task_id:
        problems.append("receipt task-policy task id drifted")
    if selected.get("task_sha256") != canonical_sha256(dict(declared_task_payload)):
        problems.append("receipt task-policy declared task binding drifted")
    problems.extend(
        task_policy_authority_problems(
            selected,
            policy_schema=current_policy_schema,
            policy_path=current_policy_path,
            full_policy_sha256=current_full_policy_sha256,
            profile_name=current_profile_name,
            safety_contract=current_safety_contract,
            foreground_markers=current_foreground_markers,
            known_heavy_markers=current_known_heavy_markers,
            task_id=declared_task_id,
            task_payload=current_task_payload,
        )
    )
    return sorted(set(problems))


__all__ = [
    "TASK_POLICY_AUTHORITY_SCHEMA",
    "POLICY_BASELINE_MANIFEST_SCHEMA",
    "build_policy_baseline_manifest",
    "build_policy_safety_contract",
    "build_task_policy_authority",
    "canonical_sha256",
    "policy_baseline_manifest_problems",
    "receipt_task_policy_authority_problems",
    "task_policy_authority_problems",
]
