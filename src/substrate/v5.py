"""The single Substrate v5 sensorium/model-fabric command family."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from substrate import v5authorities, v5campaign, v5io, v5pilot, v5principal
from substrate import v5config as C


class Refused(RuntimeError):
    """A v5 command failed a frozen safety or scientific gate."""


def _print(document: Any) -> None:
    print(json.dumps(document, indent=2, default=str))


def _tag_commit(tag: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{tag}^{{}}"],
        cwd=v5io.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _head_commit() -> str | None:
    return _tag_commit("HEAD")


def _load_admission() -> tuple[dict[str, Any], str | None]:
    admission_path = v5io.EVIDENCE / "SUBSTRATE_V5_ADMISSION.json"
    if not admission_path.is_file():
        return {}, None
    try:
        return dict(v5io.load_json(admission_path)), None
    except v5io.Refused as error:
        return {}, str(error)


def _current_authority_identities() -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for identity, relative in v5pilot.ADMISSION_AUTHORITY_PATHS.items():
        path = v5io.ROOT / relative
        try:
            document = dict(v5io.load_json(path))
        except (OSError, v5io.Refused) as error:
            errors.append(f"{identity}: {error}")
            continue
        rows[identity] = {
            "path": relative,
            "sha256": document.get("sha256"),
            "configuration_digest": document.get("configuration_digest"),
        }
    return rows, errors


def principal_gate() -> dict[str, Any]:
    ready_commit = _tag_commit(C.READY_TAG)
    admission_path = v5io.EVIDENCE / "SUBSTRATE_V5_ADMISSION.json"
    current_head = _head_commit()
    admission, admission_error = _load_admission()
    current_authorities, authority_errors = _current_authority_identities()
    admitted_bindings = admission.get("authority_bindings", {})

    def authority_matches(identity: str) -> bool:
        expected_path = v5pilot.ADMISSION_AUTHORITY_PATHS[identity]
        admitted = admitted_bindings.get(identity, {}) if isinstance(admitted_bindings, dict) else {}
        current = current_authorities.get(identity, {})
        return (
            isinstance(admitted, dict)
            and admitted.get("path") == expected_path
            and current.get("path") == expected_path
            and isinstance(admitted.get("sha256"), str)
            and len(admitted["sha256"]) == 64
            and admitted.get("sha256") == current.get("sha256")
        )

    cleanliness = v5campaign.worktree_cleanliness(v5campaign.PRINCIPAL_RUNTIME_ROOTS)
    checks = {
        "ready_tag_exists": ready_commit is not None,
        "admission_exists": bool(admission),
        "principal_launch_authorized": admission.get("principal_launch_authorized") is True,
        "activation_false": admission.get("activation") is False,
        "admission_source_commit_matches_ready_commit": (ready_commit is not None and admission.get("source_commit") == ready_commit),
        "current_head_matches_ready_commit": (ready_commit is not None and current_head == ready_commit),
        "source_digest_matches_ready_source": admission.get("source_digest") == v5io.source_digest(),
        "pilot_authority_digest_matches": authority_matches("pilot"),
        "failure_authority_digest_matches": authority_matches("failure"),
        "kernel_authority_digest_matches": authority_matches("kernel"),
        "configuration_identity_matches": (
            authority_matches("configuration")
            and admission.get("configuration_digest") == current_authorities.get("configuration", {}).get("configuration_digest")
        ),
        "model_identity_matches": (authority_matches("model") and admission.get("model_registry_digest") == current_authorities.get("model", {}).get("sha256")),
        "corpus_identity_matches": (
            authority_matches("corpus") and admission.get("corpus_catalog_digest") == current_authorities.get("corpus", {}).get("sha256")
        ),
        "worktree_clean_except_declared_runtime_roots": cleanliness["clean_except_allowed_roots"],
    }
    return {
        "ready_tag": C.READY_TAG,
        "ready_commit": ready_commit,
        "current_head": current_head,
        "admission": str(admission_path.relative_to(v5io.ROOT)),
        "admission_error": admission_error,
        "authority_errors": authority_errors,
        "current_authorities": current_authorities,
        "worktree": cleanliness,
        "checks": checks,
        "authorized": all(checks.values()),
        "activation": False,
    }


def status() -> dict[str, Any]:
    principal = v5principal.status()
    evidence = v5io.EVIDENCE
    stages = {
        "acquisition": ("completed" if (evidence / "SUBSTRATE_V5_ACQUISITION_LEDGER.json").is_file() else "pending"),
        "preprocessing": ("completed" if (evidence / "SUBSTRATE_V5_PREPROCESSING_AUTHORITY.json").is_file() else "pending"),
        "model_preparation": ("completed" if (evidence / "SUBSTRATE_V5_MODEL_HEALTH_REPORT.json").is_file() else "pending"),
        "kernel_comparison": ("completed" if (evidence / "SUBSTRATE_V5_KERNEL_SELECTION.json").is_file() else "pending"),
        "sensorium_construction": ("completed" if (evidence / "SUBSTRATE_V5_SENSORIUM_SCHEMA.json").is_file() else "pending"),
        "micro_canaries": ("completed" if (evidence / "SUBSTRATE_V5_CHEAP_CANARIES.json").is_file() else "pending"),
        "moderate_pilot": ("completed" if (evidence / "SUBSTRATE_V5_MODERATE_PILOT.json").is_file() else "pending"),
        "principal_campaign": ("completed" if principal["complete"] else ("running" if principal["present"] else "pending")),
        "replication": ("completed" if principal["splits"]["replication"]["present"] == principal["splits"]["replication"]["expected"] else "pending"),
        "open_world_review": (
            "completed" if principal["splits"]["open_world_review"]["present"] == principal["splits"]["open_world_review"]["expected"] else "pending"
        ),
        "independent_verification": ("completed" if (evidence / "SUBSTRATE_V5_INDEPENDENT_VERIFICATION.json").is_file() else "pending"),
        "terminal_publication": (
            "completed" if (evidence / "SUBSTRATE_V5_FINAL_STATE.json").is_file() and _tag_commit(C.TERMINAL_TAG) is not None else "pending"
        ),
    }
    current = next(
        (name for name, value in stages.items() if value != "completed"),
        "terminal_publication",
    )
    return {
        "schema": "substrate-v5-status/v1",
        "current_stage": current,
        "stages": stages,
        "principal": principal,
        "principal_gate": principal_gate(),
        "stop_switch": v5io.STOP.exists(),
        "activation": False,
    }


def _exit(document: dict[str, Any], passed: bool) -> None:
    _print(document)
    raise SystemExit(0 if passed else 1)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else "status"
    if command == "preflight":
        report = v5campaign.seal_preflight()
        _exit(
            {
                "all_pass": report["all_pass"],
                "failed": report["preflight"]["failed"],
                "sealed": report["sealed"],
                "activation": False,
            },
            report["all_pass"],
        )
    if command == "inventory":
        _print(v5campaign.inventory())
        return
    if command == "acquire":
        v5authorities.publish_construction()
        acquisition = v5io.load("SUBSTRATE_V5_ACQUISITION_LEDGER.json")
        _exit(
            {
                "network_downloads": acquisition["mechanism"]["network_downloads"],
                "bytes_downloaded": acquisition["mechanism"]["bytes_downloaded"],
                "external_objects_admitted": 0,
                "activation": False,
            },
            True,
        )
    if command == "build":
        frozen = v5campaign.freeze()
        constructed = v5authorities.publish_construction()
        principal = v5principal.prepare()
        _exit(
            {
                "frozen": frozen["sealed"],
                "construction_authorities": constructed["count"],
                "principal_units": principal["manifest"]["unit_count"],
                "activation": False,
            },
            True,
        )
    if command == "canaries":
        from substrate import v5canary

        report = v5canary.run(publish=True)
        evidence = report.get("evidence", report)
        _exit(evidence, bool(evidence["all_pass"]))
    if command == "pilot":
        from substrate import v5pilot

        report = v5pilot.run(publish=True)
        admission = report["admission"]
        _exit(report, bool(admission["principal_launch_authorized"]))
    if command == "rehearse":
        from substrate import v5pilot

        report = v5pilot.rehearse(publish=True)
        _exit(report, bool(report.get("all_pass")))
    if command == "run":
        gate = principal_gate()
        if not gate["authorized"]:
            raise Refused(f"principal launch gate failed: {gate['checks']}")
        report = v5principal.run()
        _exit(report, bool(report["all_terminal"]))
    if command == "status":
        _print(status())
        return
    if command == "stop":
        _print(
            {
                "stopped": True,
                "stop_switch": str(v5io.stop()),
                "activation": False,
            }
        )
        return
    if command == "resume":
        gate = principal_gate()
        if not gate["authorized"]:
            raise Refused(f"principal resume gate failed: {gate['checks']}")
        v5io.resume()
        report = v5principal.run()
        _exit(report, bool(report["all_terminal"]))
    if command == "verify":
        from substrate import v5verify

        report = v5verify.run_all(publish=True)
        _exit(report, bool(report["all_pass"]))
    raise SystemExit(f"unknown v5 command {command!r}")
