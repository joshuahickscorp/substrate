
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studio.generation1_supervisor import (
    canonical_sha256,
    sha256_file,
    write_immutable_json,
)


RUNS_ROOT = REPO_ROOT / "runs/generation1"
PROOF_ROOT = REPO_ROOT / "proof"
OUTPUT_ROOT = REPO_ROOT / "reports/subaccomplishment_emitter"
LOG = OUTPUT_ROOT / "subaccomp.log"
ERROR_LOG = OUTPUT_ROOT / "subaccomp.error.log"
PLIST = Path.home() / "Library/LaunchAgents/com.mop.generation1.subaccomp.plist"
LABEL = "com.mop.generation1.subaccomp"

MAX_JSON_BYTES = 32 * 1024 * 1024
POLL_SECONDS = 120

SUBACCOMP_SCHEMA = "mop-generation1-subaccomplishment-milestone/v1"
MILESTONE_KINDS = ("barrier", "reprofile", "gate", "absorption")
MILESTONE_FIELDS = frozenset(
    {
        "schema",
        "milestone_kind",
        "source_program_id",
        "source",
        "headline",
        "summary",
        "grid",
        "decision",
        "complete",
        "advisory",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "milestone_sha256",
    }
)
SOURCE_FIELDS = frozenset({"path", "file_sha256", "seal_field", "seal"})
FILENAME_PREFIX = "GENERATION1_SUBACCOMP_"

REPROFILE_SCHEMA = "mop-generation1-reprofile/v1"
CATEGORIZED_CLASSIFICATION_SCHEMA = "mop-generation1-successor-categorized-classification/v1"
CATEGORIZED_GATE_SCHEMA = "mop-generation1-successor-categorized-gate/v1"
CONSOLIDATED_FINAL_PROGRAM_ID = "generation1-consolidated-final-campaign-v1"
CONSOLIDATED_FINAL_RESULT_SCHEMA = "mop-generation1-consolidated-final-result/v1"
CONSOLIDATED_FINAL_RESULT_NAME = "GENERATION1_CONSOLIDATED_FINAL_RESULT.json"
ABSORPTION_RECEIPT_SCHEMA = "mop-generation1-consolidated-final-absorption-receipt/v1"

_SHA_RE = re.compile(r"[0-9a-f]{64}")

_PROGRAM_LABELS = {
    "generation1-successor-categorized-batch-wave-v1": "Categorized Batch Wave",
    "generation1-consolidated-final-campaign-v1": "Final Campaign",
}


class SubaccomplishmentRefused(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any] | None:

    raw = Path(path)
    try:
        if raw.is_symlink():
            return None
        source = raw.resolve(strict=True)
        if not source.is_file() or source.stat().st_size > MAX_JSON_BYTES:
            return None
        value = json.loads(source.read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _seal_ok(value: Mapping[str, Any], field: str) -> bool:
    core = {key: item for key, item in value.items() if key != field}
    return value.get(field) == canonical_sha256(core)


def _valid_seal_str(value: Any) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_or_zero(value: Any) -> int:
    return int(value) if _is_int(value) else 0


def _program_label(program_id: Any) -> str:
    identifier = str(program_id)
    if identifier in _PROGRAM_LABELS:
        return _PROGRAM_LABELS[identifier]
    simplified = re.sub(r"^generation1-", "", identifier)
    simplified = re.sub(r"-v\d+$", "", simplified)
    return simplified.replace("-", " ").title() or "MOP"


def _source_path_str(path: Path, root: Path | str) -> str:

    resolved = Path(path).resolve()
    repo = REPO_ROOT.resolve()
    if resolved.is_relative_to(repo):
        return str(resolved.relative_to(repo))
    base = Path(root).resolve()
    if resolved.is_relative_to(base):
        return str(resolved.relative_to(base))
    return resolved.name


def _milestone_core(
    *,
    kind: str,
    program_id: Any,
    path: Path,
    root: Path | str,
    seal_field: str,
    seal: Any,
    headline: str,
    summary: Sequence[str],
    grid: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema": SUBACCOMP_SCHEMA,
        "milestone_kind": kind,
        "source_program_id": str(program_id) if program_id is not None else None,
        "source": {
            "path": _source_path_str(path, root),
            "file_sha256": sha256_file(Path(path)),
            "seal_field": seal_field,
            "seal": str(seal),
        },
        "headline": headline,
        "summary": [str(line) for line in summary],
        "grid": dict(grid),
        "decision": dict(decision),
        "complete": True,
        "advisory": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "milestone_sha256": canonical_sha256(core)}


def _milestone_filename(kind: str, seal: str) -> str:
    if kind not in MILESTONE_KINDS:
        raise SubaccomplishmentRefused(f"unknown milestone kind: {kind}")
    if not _valid_seal_str(seal):
        raise SubaccomplishmentRefused("milestone source seal is not a SHA-256")
    return f"{FILENAME_PREFIX}{kind}_{seal[:16]}.json"


def _barrier_milestone(classification: Mapping[str, Any], path: Path, root: Path | str) -> dict[str, Any]:
    summary_block = classification.get("summary")
    summary_block = summary_block if isinstance(summary_block, Mapping) else {}
    executed = _int_or_zero(summary_block.get("executed_item_count"))
    skipped = _int_or_zero(summary_block.get("skipped_item_count"))
    epoch_id = classification.get("epoch_id")
    epoch_index = classification.get("epoch_index")
    epoch_ordinal = epoch_index + 1 if _is_int(epoch_index) else "?"
    program_id = classification.get("program_id")
    label = _program_label(program_id)
    grid = {"completed_cell_count": executed, "expected_cell_count": executed + skipped}
    headline = f"{label}: epoch barrier sealed ({epoch_id})"
    summary = [
        f"epoch {epoch_ordinal}: {epoch_id}",
        f"executed {executed}, skipped {skipped}",
        "serial barrier only; no confirmation claim",
    ]
    decision = {
        "verdict": "epoch_barrier_sealed",
        "scientific_confirmation": False,
        "next_action": "review",
    }
    return _milestone_core(
        kind="barrier",
        program_id=program_id,
        path=path,
        root=root,
        seal_field="classification_sha256",
        seal=classification.get("classification_sha256"),
        headline=headline,
        summary=summary,
        grid=grid,
        decision=decision,
    )


def _reprofile_milestone(reprofile: Mapping[str, Any], path: Path, root: Path | str) -> dict[str, Any]:
    recommendation = reprofile.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, Mapping) else {}
    workers = recommendation.get("recommended_workers")
    workers_text = str(workers) if _is_int(workers) else "n/a"
    constraint = recommendation.get("binding_constraint")
    provisional = reprofile.get("provisional_mechanisms")
    provisional_count = len(provisional) if isinstance(provisional, list) else 0
    seen = _int_or_zero(reprofile.get("receipts_seen"))
    receipt_skipped = _int_or_zero(reprofile.get("receipts_skipped"))
    grid = {"completed_seed_count": seen, "expected_seed_count": seen + receipt_skipped}
    workers_line = f"recommended workers: {workers_text}"
    if isinstance(constraint, str) and constraint:
        workers_line += f" ({constraint})"
    summary = [
        workers_line,
        f"provisional mechanisms: {provisional_count}",
        "advisory only; no evidence, seed, or threshold change",
    ]
    decision = {
        "verdict": "advisory_reprofile",
        "scientific_confirmation": False,
        "next_action": "review",
    }
    return _milestone_core(
        kind="reprofile",
        program_id=None,
        path=path,
        root=root,
        seal_field="reprofile_sha256",
        seal=reprofile.get("reprofile_sha256"),
        headline="Reprofile: advisory re-tune",
        summary=summary,
        grid=grid,
        decision=decision,
    )


def _gate_milestone(gate: Mapping[str, Any], path: Path, root: Path | str) -> dict[str, Any]:
    program_id = gate.get("program_id")
    gate_id = gate.get("gate_id")
    gate_index = gate.get("gate_index")
    gate_ordinal = gate_index if _is_int(gate_index) else "?"
    payload = gate.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    lanes = payload.get("mechanics_lanes")
    lane_count = len(lanes) if isinstance(lanes, list) else 0
    label = _program_label(program_id)
    grid = {"completed_cell_count": lane_count, "expected_cell_count": lane_count}
    headline = f"{label}: transition gate sealed ({gate_id})"
    summary = [
        f"gate {gate_ordinal}: {gate_id}",
        f"mechanics lanes admitted: {lane_count}",
        "serial admission barrier; no confirmation claim",
    ]
    decision = {
        "verdict": "transition_gate_sealed",
        "scientific_confirmation": False,
        "next_action": "review",
    }
    return _milestone_core(
        kind="gate",
        program_id=program_id,
        path=path,
        root=root,
        seal_field="gate_sha256",
        seal=gate.get("gate_sha256"),
        headline=headline,
        summary=summary,
        grid=grid,
        decision=decision,
    )


def _absorption_milestone(
    receipt: Mapping[str, Any],
    terminal: Mapping[str, Any],
    receipt_path: Path,
    root: Path | str,
) -> dict[str, Any]:
    program_id = receipt.get("absorbed_program_id")
    label = _program_label(program_id)
    result_seal = terminal.get("result_sha256")
    result_seal_text = result_seal[:16] if isinstance(result_seal, str) else "n/a"
    grid_src = terminal.get("grid")
    grid_src = grid_src if isinstance(grid_src, Mapping) else {}
    work_items = _int_or_zero(grid_src.get("work_item_count"))
    grid = {"completed_cell_count": work_items, "expected_cell_count": work_items}
    decision_src = terminal.get("decision")
    decision_src = decision_src if isinstance(decision_src, Mapping) else {}
    next_action = decision_src.get("next_action")
    next_action = next_action if isinstance(next_action, str) else "review"
    headline = f"Absorption complete: {label}"
    summary = [
        f"result sealed: {result_seal_text}",
        "conditional final campaign complete; scientific_confirmation=false",
        "observe-only absorption; no signal sent to the live run",
    ]
    decision = {
        "verdict": "absorption_complete",
        "scientific_confirmation": False,
        "next_action": next_action,
    }
    return _milestone_core(
        kind="absorption",
        program_id=program_id,
        path=receipt_path,
        root=root,
        seal_field="receipt_sha256",
        seal=receipt.get("receipt_sha256"),
        headline=headline,
        summary=summary,
        grid=grid,
        decision=decision,
    )


def _default_classification_validators() -> dict[str, Callable[..., None]]:
    validators: dict[str, Callable[..., None]] = {}
    try:
        from mop.studies import generation1_successor_categorized_batch_wave as cbw

        validators[CATEGORIZED_CLASSIFICATION_SCHEMA] = cbw.validate_classification
    except Exception:
        pass
    return validators


def _default_gate_validators() -> dict[str, Callable[..., None]]:
    validators: dict[str, Callable[..., None]] = {}
    try:
        from mop.studies import generation1_successor_categorized_batch_wave as cbw

        validators[CATEGORIZED_GATE_SCHEMA] = cbw.validate_gate
    except Exception:
        pass
    return validators


def _default_reprofile_validator() -> Callable[[Mapping[str, Any]], None]:
    from mop.studio.generation1_result_aware_reprofiler import validate_reprofile

    return validate_reprofile


def _default_result_validator() -> Callable[[Mapping[str, Any]], None]:
    from mop.studies.generation1_consolidated_final_campaign import validate_result

    return validate_result


def _validate_absorption_receipt(value: Mapping[str, Any]) -> None:

    if not isinstance(value, Mapping):
        raise SubaccomplishmentRefused("absorption receipt must be an object")
    if not _seal_ok(value, "receipt_sha256"):
        raise SubaccomplishmentRefused("absorption receipt self-seal mismatch")
    if value.get("schema") != ABSORPTION_RECEIPT_SCHEMA:
        raise SubaccomplishmentRefused("absorption receipt schema drifted")
    if value.get("absorbed_program_id") != CONSOLIDATED_FINAL_PROGRAM_ID:
        raise SubaccomplishmentRefused("absorption receipt names the wrong absorbed program")
    policy = value.get("policy")
    if not isinstance(policy, Mapping) or (
        policy.get("observe_only") is not True
        or policy.get("signals_allowed") is not False
        or policy.get("restart_disallowed") is not True
        or policy.get("append_only") is not True
    ):
        raise SubaccomplishmentRefused("absorption receipt policy is not observe-only")
    observed = value.get("observed_status")
    if not isinstance(observed, Mapping) or observed.get("state") != "complete":
        raise SubaccomplishmentRefused("absorption receipt observed status is not complete")
    absorbed = value.get("absorbed_result")
    if not isinstance(absorbed, Mapping) or absorbed.get("schema") != CONSOLIDATED_FINAL_RESULT_SCHEMA:
        raise SubaccomplishmentRefused("absorption receipt absorbed_result schema drifted")
    if not _valid_seal_str(absorbed.get("result_sha256")):
        raise SubaccomplishmentRefused("absorption receipt absorbed_result seal is invalid")


def scan_classifications(
    runs_root: Path | str = RUNS_ROOT,
    *,
    validators: Mapping[str, Callable[..., None]] | None = None,
) -> Iterator[dict[str, Any]]:

    resolved = validators if validators is not None else _default_classification_validators()
    base = Path(runs_root)
    if not base.exists():
        return
    for path in sorted(base.glob("*/classifications/*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        schema = payload.get("schema")
        validate = resolved.get(schema) if isinstance(schema, str) else None
        if validate is None:
            continue
        epoch_index = payload.get("epoch_index")
        if not _is_int(epoch_index):
            continue
        if not _valid_seal_str(payload.get("classification_sha256")):
            continue
        wave_root = path.parent.parent
        try:
            validate(payload, epoch_index, root=wave_root)
        except Exception:
            continue
        yield _barrier_milestone(payload, path, runs_root)


def scan_gates(
    runs_root: Path | str = RUNS_ROOT,
    *,
    validators: Mapping[str, Callable[..., None]] | None = None,
) -> Iterator[dict[str, Any]]:

    resolved = validators if validators is not None else _default_gate_validators()
    base = Path(runs_root)
    if not base.exists():
        return
    for path in sorted(base.glob("*/gates/*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        schema = payload.get("schema")
        validate = resolved.get(schema) if isinstance(schema, str) else None
        if validate is None:
            continue
        gate_index = payload.get("gate_index")
        if not _is_int(gate_index):
            continue
        if not _valid_seal_str(payload.get("gate_sha256")):
            continue
        wave_root = path.parent.parent
        try:
            validate(payload, gate_index, root=wave_root)
        except Exception:
            continue
        yield _gate_milestone(payload, path, runs_root)


def scan_reprofiles(
    runs_root: Path | str = RUNS_ROOT,
    proof_root: Path | str = PROOF_ROOT,
    *,
    validate: Callable[[Mapping[str, Any]], None] | None = None,
    roots: Sequence[Path | str] | None = None,
) -> Iterator[dict[str, Any]]:

    validate_fn = validate if validate is not None else _default_reprofile_validator()
    search_roots = roots if roots is not None else (Path(runs_root), Path(proof_root))
    seen: set[Path] = set()
    for base in search_roots:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for path in sorted(base_path.rglob("*.json")):
            if path.name.startswith("GENERATION1") or "reprofile" not in path.name.lower():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = _read_json(path)
            if payload is None or payload.get("schema") != REPROFILE_SCHEMA:
                continue
            if not _valid_seal_str(payload.get("reprofile_sha256")):
                continue
            try:
                validate_fn(payload)
            except Exception:
                continue
            yield _reprofile_milestone(payload, path, base_path)


def scan_absorptions(
    runs_root: Path | str = RUNS_ROOT,
    proof_root: Path | str = PROOF_ROOT,
    *,
    receipt_validator: Callable[[Mapping[str, Any]], None] | None = None,
    result_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> Iterator[dict[str, Any]]:

    receipt_check = receipt_validator if receipt_validator is not None else _validate_absorption_receipt
    result_check = result_validator if result_validator is not None else _default_result_validator()
    terminal_path = Path(proof_root) / CONSOLIDATED_FINAL_RESULT_NAME
    if not terminal_path.is_file():
        return
    terminal = _read_json(terminal_path)
    if terminal is None:
        return
    try:
        result_check(terminal)
    except Exception:
        return
    terminal_result_seal = terminal.get("result_sha256")
    terminal_file_sha = sha256_file(terminal_path)
    base = Path(runs_root)
    if not base.exists():
        return
    for path in sorted(base.rglob("absorptions/*/*.json")):
        payload = _read_json(path)
        if payload is None or payload.get("schema") != ABSORPTION_RECEIPT_SCHEMA:
            continue
        if not _valid_seal_str(payload.get("receipt_sha256")):
            continue
        try:
            receipt_check(payload)
        except Exception:
            continue
        absorbed = payload.get("absorbed_result")
        absorbed = absorbed if isinstance(absorbed, Mapping) else {}
        if absorbed.get("result_sha256") != terminal_result_seal:
            continue
        if absorbed.get("file_sha256") != terminal_file_sha:
            continue
        yield _absorption_milestone(payload, terminal, path, runs_root)


def collect_milestones(
    runs_root: Path | str = RUNS_ROOT,
    proof_root: Path | str = PROOF_ROOT,
    *,
    classification_validators: Mapping[str, Callable[..., None]] | None = None,
    gate_validators: Mapping[str, Callable[..., None]] | None = None,
    reprofile_validator: Callable[[Mapping[str, Any]], None] | None = None,
    reprofile_roots: Sequence[Path | str] | None = None,
    absorption_receipt_validator: Callable[[Mapping[str, Any]], None] | None = None,
    result_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:

    milestones: list[dict[str, Any]] = []
    milestones.extend(scan_classifications(runs_root, validators=classification_validators))
    milestones.extend(scan_gates(runs_root, validators=gate_validators))
    milestones.extend(
        scan_reprofiles(
            runs_root,
            proof_root,
            validate=reprofile_validator,
            roots=reprofile_roots,
        )
    )
    milestones.extend(
        scan_absorptions(
            runs_root,
            proof_root,
            receipt_validator=absorption_receipt_validator,
            result_validator=result_validator,
        )
    )
    published: dict[str, dict[str, Any]] = {}
    for milestone in milestones:
        try:
            validate_milestone(milestone)
        except SubaccomplishmentRefused:
            continue
        name = _milestone_filename(milestone["milestone_kind"], milestone["source"]["seal"])
        published.setdefault(name, milestone)
    return [published[name] for name in sorted(published)]


def validate_milestone(value: Mapping[str, Any]) -> None:

    if not isinstance(value, Mapping):
        raise SubaccomplishmentRefused("milestone must be an object")
    if set(value) != set(MILESTONE_FIELDS):
        missing = sorted(set(MILESTONE_FIELDS) - set(value))
        extra = sorted(set(value) - set(MILESTONE_FIELDS))
        raise SubaccomplishmentRefused(f"milestone fields drifted; missing={missing}, extra={extra}")
    if not _seal_ok(value, "milestone_sha256"):
        raise SubaccomplishmentRefused("milestone self-seal mismatch")
    if value.get("schema") != SUBACCOMP_SCHEMA:
        raise SubaccomplishmentRefused("milestone schema drifted")
    if value.get("milestone_kind") not in MILESTONE_KINDS:
        raise SubaccomplishmentRefused("milestone kind is unknown")
    if value.get("complete") is not True:
        raise SubaccomplishmentRefused("milestone must be complete")
    if value.get("advisory") is not True:
        raise SubaccomplishmentRefused("milestone must remain advisory")
    if value.get("problems") != []:
        raise SubaccomplishmentRefused("milestone must carry no problems")
    if value.get("activation_allowed") is not False:
        raise SubaccomplishmentRefused("milestone must keep activation_allowed false")
    if value.get("scientific_promotion") is not False:
        raise SubaccomplishmentRefused("milestone must keep scientific_promotion false")
    summary = value.get("summary")
    if not isinstance(summary, list) or not all(isinstance(line, str) for line in summary):
        raise SubaccomplishmentRefused("milestone summary must be a list of strings")
    program_id = value.get("source_program_id")
    if program_id is not None and not isinstance(program_id, str):
        raise SubaccomplishmentRefused("milestone source_program_id must be a string or null")
    if not isinstance(value.get("grid"), Mapping):
        raise SubaccomplishmentRefused("milestone grid must be an object")
    decision = value.get("decision")
    if not isinstance(decision, Mapping) or decision.get("scientific_confirmation") is not False:
        raise SubaccomplishmentRefused("milestone decision must keep scientific_confirmation false")
    source = value.get("source")
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_FIELDS):
        raise SubaccomplishmentRefused("milestone source block drifted")
    if not _valid_seal_str(source.get("seal")) or not _valid_seal_str(source.get("file_sha256")):
        raise SubaccomplishmentRefused("milestone source seal or file digest is invalid")
    if not isinstance(source.get("seal_field"), str) or not isinstance(source.get("path"), str):
        raise SubaccomplishmentRefused("milestone source path or seal_field is invalid")


def write_milestone(milestone: Mapping[str, Any], proof_root: Path | str = PROOF_ROOT) -> dict[str, Any]:

    validate_milestone(milestone)
    name = _milestone_filename(milestone["milestone_kind"], milestone["source"]["seal"])
    path = Path(proof_root) / name
    if path.exists():
        return {"status": "exists", "path": str(path), "name": name}
    write_immutable_json(path, dict(milestone))
    return {"status": "written", "path": str(path), "name": name}


def scan(
    runs_root: Path | str = RUNS_ROOT,
    proof_root: Path | str = PROOF_ROOT,
    *,
    now: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:

    proof_dir = Path(proof_root)
    proof_dir.mkdir(parents=True, exist_ok=True)
    milestones = collect_milestones(runs_root, proof_root, **overrides)
    written: list[str] = []
    existing: list[str] = []
    for milestone in milestones:
        outcome = write_milestone(milestone, proof_dir)
        (written if outcome["status"] == "written" else existing).append(outcome["name"])
    return {
        "scanned_at": now,
        "milestones": len(milestones),
        "written": sorted(written),
        "existing": sorted(existing),
        "written_count": len(written),
        "existing_count": len(existing),
    }


def build_launch_agent_plist() -> dict[str, Any]:

    return {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "mop.studio.generation1_subaccomplishment_emitter",
            "scan",
        ],
        "WorkingDirectory": str(REPO_ROOT),
        "StartInterval": POLL_SECONDS,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "StandardOutPath": str(LOG),
        "StandardErrorPath": str(ERROR_LOG),
        "EnvironmentVariables": {"PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])},
    }


def install_launch_agent(*, execute: bool = False) -> dict[str, Any]:

    document = build_launch_agent_plist()
    if execute:
        raise SubaccomplishmentRefused(
            "install execution is intentionally disabled; install the returned plist plan by hand"
        )
    return {
        "installed": False,
        "dry_run": True,
        "label": LABEL,
        "plist": str(PLIST),
        "interval_seconds": POLL_SECONDS,
        "program_arguments": document["ProgramArguments"],
        "document": document,
    }


def status(proof_root: Path | str = PROOF_ROOT) -> dict[str, Any]:

    proof_dir = Path(proof_root)
    if proof_dir.exists():
        files = sorted(path.name for path in proof_dir.glob(f"{FILENAME_PREFIX}*.json"))
    else:
        files = []
    return {
        "label": LABEL,
        "launch_agent_installed": PLIST.exists(),
        "interval_seconds": POLL_SECONDS,
        "published_milestones": len(files),
        "milestone_files": files,
        "proof_root": str(proof_dir),
    }


def _cmd_scan(args: argparse.Namespace) -> int:
    report = scan(runs_root=Path(args.runs_root), proof_root=Path(args.proof_root))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    payload = _read_json(Path(args.path))
    if payload is None:
        print(f"INVALID: unreadable or malformed JSON at {args.path}")
        return 1
    try:
        validate_milestone(payload)
    except SubaccomplishmentRefused as exc:
        print(f"INVALID: {exc}")
        return 1
    print("VALID")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(status(proof_root=Path(args.proof_root)), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mop.studio.generation1_subaccomplishment_emitter",
        description="Observe-only re-publisher of Generation 1 sub-accomplishment milestones.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="scan sealed milestones and append subaccomp proofs")
    scan_parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    scan_parser.add_argument("--proof-root", default=str(PROOF_ROOT))
    scan_parser.set_defaults(func=_cmd_scan)

    validate_parser = subparsers.add_parser("validate", help="fail-closed validate a subaccomp proof file")
    validate_parser.add_argument("--path", required=True)
    validate_parser.set_defaults(func=_cmd_validate)

    status_parser = subparsers.add_parser("status", help="print published-milestone status (read-only)")
    status_parser.add_argument("--proof-root", default=str(PROOF_ROOT))
    status_parser.set_defaults(func=_cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.func
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
