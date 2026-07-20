
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from .long_run import load_plan, write_plan_template
from .profiles import get_profile, is_studio_profile

SCHEMA = "mop-studio-spine-plan/v1"
STATUS_SCHEMA = "mop-studio-spine-status/v1"
DEFAULT_SPINE_DIR = Path("runs") / "studio_spine"
DEFAULT_DR1_DIR = Path("runs") / "studio_dr1"
DEFAULT_WAVE0_DIR = Path("runs") / "studio_wave0"
DEFAULT_DR1_CACHE = "vjepa2_vitl_comp_video"
DEFAULT_DENSE_CACHE = "vjepa21_vitl_dense8192_real"
DEFAULT_DENSE_RANDOMINIT_CACHE = "vjepa21_vitl_dense8192_randominit"


@dataclass(frozen=True)
class StudioSpineConfig:

    source: str
    profile_name: str = "studio-m1ultra"
    python: str = ".venv/bin/python"
    spine_dir: Path = DEFAULT_SPINE_DIR
    wave0_dir: Path = DEFAULT_WAVE0_DIR
    dr1_dir: Path = DEFAULT_DR1_DIR
    dr1_cache: str = DEFAULT_DR1_CACHE
    dense_cache: str = DEFAULT_DENSE_CACHE
    dense_randominit_cache: str = DEFAULT_DENSE_RANDOMINIT_CACHE
    dense_encoder: str = "vjepa21_vitl"
    dense_min_tokens: int = 8192
    dense_expected_dim: int = 1024
    source_card: Path | None = None
    planned_clips: int = 1000
    dense_planned_clips: int = 1000
    pr9_seeds: str = "0-9"
    atlas_seeds: str = "0-9"


def build_studio_spine_plan(config: StudioSpineConfig | str) -> dict[str, Any]:
    cfg = config if isinstance(config, StudioSpineConfig) else StudioSpineConfig(source=str(config))
    if not cfg.source:
        raise ValueError("source is required for the DR1 real bound-attribute video stage")
    profile = get_profile(cfg.profile_name)
    wave0_plan = cfg.spine_dir / "wave0_daemon_plan.json"
    dr1_source_card = cfg.source_card or cfg.dr1_dir / "dr1_source_card.json"
    dr1_source_card_validation = cfg.dr1_dir / "dr1_source_card_validation.json"
    dr1_source_intake = cfg.dr1_dir / "dr1_source_intake.json"
    dr1_schedule = cfg.dr1_dir / "dr1_schedule_plan.json"
    dr1_daemon = cfg.dr1_dir / "dr1_daemon_plan.json"
    steps = [
        _step(
            "wave0_validate_plan",
            "wave0",
            "gate",
            _studio_cmd(cfg.python, "daemon", "validate", "--plan", str(wave0_plan)),
            receipts=[str(wave0_plan)],
            note="validate the Wave 0 daemon plan before the Studio can execute it",
        ),
        _step(
            "wave0_run",
            "wave0",
            "subdaemon",
            _studio_cmd(
                cfg.python,
                "daemon",
                "run",
                "--plan",
                str(wave0_plan),
                "--out-dir",
                str(cfg.wave0_dir),
                "--profile",
                cfg.profile_name,
                "--execute",
            ),
            receipts=[
                str(cfg.wave0_dir / "daemon_state.json"),
                "runs/mot/encode_device.json",
                "runs/mot/encode_schedule.json",
            ],
            note="runs transfer, disk recovery, doctor, docs, acceptance, DR1 smoke, microbench, and report",
        ),
        _step(
            "wave0_artifact_bundle",
            "wave0",
            "artifact-bundle",
            _bundle_cmd(
                cfg.python,
                "wave0",
                "proof/ARTIFACT_INDEX/wave0.json",
                "proof/ARTIFACT_BUNDLES/wave0",
            ),
            receipts=["proof/ARTIFACT_INDEX/wave0.json"],
            note="preserve ignored Wave 0 receipts after the daemon finishes",
        ),
        _step(
            "dr1_source_card_validate",
            "dr1",
            "gate",
            [
                cfg.python,
                "scripts/studio/dr1_source_card.py",
                "validate",
                str(dr1_source_card),
                "--out",
                str(dr1_source_card_validation),
            ],
            receipts=[str(dr1_source_card_validation)],
            note="prove the DR1 source card is populated with natural-video provenance before source intake",
        ),
        _step(
            "dr1_source_intake",
            "dr1",
            "gate",
            [
                cfg.python,
                "scripts/studio/dr1_source_intake.py",
                "--source",
                cfg.source,
                "--source-card",
                str(dr1_source_card),
                "--out",
                str(dr1_source_intake),
            ],
            receipts=[str(dr1_source_intake)],
            note=(
                "prove DR1 source layout, caption coverage, natural-video provenance, license, "
                "and non-overlap before encode"
            ),
        ),
        _step(
            "dr1_schedule_build",
            "dr1",
            "plan",
            [
                cfg.python,
                "scripts/studio/dr1_schedule_plan.py",
                "--schedule",
                "runs/mot/encode_schedule.json",
                "--source",
                cfg.source,
                "--name",
                cfg.dr1_cache,
                "--source-intake",
                str(dr1_source_intake),
                "--out",
                str(dr1_schedule),
                "--daemon-out",
                str(dr1_daemon),
            ],
            receipts=[str(dr1_schedule), str(dr1_daemon)],
            note=(
                "materialize DR1 caption gate, encode shards, merge, A6 guard, and verifier from Wave 0 speed"
            ),
        ),
        _step(
            "dr1_validate_plan",
            "dr1",
            "gate",
            _studio_cmd(cfg.python, "daemon", "validate", "--plan", str(dr1_daemon)),
            receipts=[str(dr1_daemon)],
            note="static contract check before long DR1 encode starts",
        ),
        _step(
            "dr1_run",
            "dr1",
            "subdaemon",
            _studio_cmd(
                cfg.python,
                "daemon",
                "run",
                "--plan",
                str(dr1_daemon),
                "--out-dir",
                str(cfg.dr1_dir),
                "--profile",
                cfg.profile_name,
                "--execute",
            ),
            receipts=[
                str(cfg.dr1_dir / "daemon_state.json"),
                f"data/cache/{cfg.dr1_cache}/merge_manifest.json",
                f"data/cache/{cfg.dr1_cache}/perspective_matrix_receipt.json",
                f"data/cache/{cfg.dr1_cache}/a6_residual_guard.json",
                f"data/cache/{cfg.dr1_cache}/dr1_verification.json",
            ],
            note="DR1 positive evidence is impossible here until dr1_verify writes an independent pass",
        ),
        _step(
            "dr1_artifact_bundle",
            "dr1",
            "artifact-bundle",
            _bundle_cmd(
                cfg.python,
                "dr1",
                "proof/ARTIFACT_INDEX/dr1.json",
                "proof/ARTIFACT_BUNDLES/dr1",
            ),
            receipts=["proof/ARTIFACT_INDEX/dr1.json"],
            note="preserve DR1 sidecars and null card without copying dense arrays",
        ),
        _step(
            "pr9_run",
            "pr9",
            "long-run",
            [
                cfg.python,
                "scripts/studio/pr9_continual_backprop.py",
                "--cache",
                str(Path("data") / "cache" / cfg.dr1_cache),
                "--seeds",
                cfg.pr9_seeds,
                "--out",
                "runs/mot/pr9_continual_backprop.json",
            ],
            receipts=[
                "runs/mot/pr9_continual_backprop.json",
                "runs/mot/pr9_continual_backprop.json.state.json",
            ],
            note="certificate-guarded long stream; resume by rerunning the same command",
        ),
        _step(
            "pr9_verdict_ledger",
            "pr9",
            "report",
            [
                cfg.python,
                "scripts/studio/pr9_verdict_ledger.py",
                "--result",
                "runs/mot/pr9_continual_backprop.json",
                "--state",
                "runs/mot/pr9_continual_backprop.json.state.json",
                "--out",
                "runs/mot/pr9_verdict_ledger.json",
            ],
            receipts=["runs/mot/pr9_verdict_ledger.json"],
            note="classify the PR9 raw result/state into null, wall, or candidate-positive before bundling",
        ),
        _step(
            "process_c_license_gate",
            "pr9",
            "verdict-gate",
            [
                cfg.python,
                "scripts/studio/process_c_license_gate.py",
                "--pr9-verdict",
                "runs/mot/pr9_verdict_ledger.json",
                "--dr1-verification",
                str(Path("data") / "cache" / cfg.dr1_cache / "dr1_verification.json"),
                "--out",
                "runs/mot/process_c_license_gate.json",
            ],
            receipts=["runs/mot/process_c_license_gate.json"],
            note=(
                "decide whether Process C is launch-allowed from PR9 and DR1 receipts; "
                "an unlicensed decision is a complete wall, not permission to train"
            ),
        ),
        _step(
            "pr9_artifact_bundle",
            "pr9",
            "artifact-bundle",
            _bundle_cmd(
                cfg.python,
                "pr9",
                "proof/ARTIFACT_INDEX/pr9.json",
                "proof/ARTIFACT_BUNDLES/pr9",
            ),
            receipts=["proof/ARTIFACT_INDEX/pr9.json"],
            note=(
                "preserve PR9 result, run-state, verdict ledger, Process C license gate, and null "
                "cards before any ledger mutation"
            ),
        ),
        _step(
            "dense_cache_plan",
            "dense_atlas",
            "gate",
            [
                cfg.python,
                "scripts/mop_encode_autoselect.py",
                "--profile",
                cfg.profile_name,
                "--planned-clips",
                str(int(cfg.dense_planned_clips)),
                "--n-clips",
                "8",
                "--dense",
                "--encoder",
                cfg.dense_encoder,
                "--out",
                "runs/mot/dense_encode_device.json",
                "--schedule-out",
                "runs/mot/dense_encode_schedule.json",
            ],
            receipts=["runs/mot/dense_encode_device.json", "runs/mot/dense_encode_schedule.json"],
            note="size the dense cache honestly; V-JEPA 2.1 remains blocked until real weights exist",
        ),
        _step(
            "dense_atlas_cache_gate",
            "dense_atlas",
            "verdict-gate",
            [
                cfg.python,
                "scripts/studio/dense_atlas_gate.py",
                "--real-cache",
                str(Path("data") / "cache" / cfg.dense_cache),
                "--randominit-cache",
                str(Path("data") / "cache" / cfg.dense_randominit_cache),
                "--min-count",
                str(int(cfg.dense_planned_clips)),
                "--min-tokens",
                str(int(cfg.dense_min_tokens)),
                "--expected-dim",
                str(int(cfg.dense_expected_dim)),
                "--out",
                "runs/mot/dense_atlas_cache_gate.json",
            ],
            receipts=[
                "runs/mot/dense_atlas_cache_gate.json",
                str(Path("data") / "cache" / cfg.dense_cache / "cache_manifest.json"),
                str(Path("data") / "cache" / cfg.dense_randominit_cache / "cache_manifest.json"),
            ],
            note="the atlas cannot claim dense-token scope until real and random-init dense caches match",
        ),
        _step(
            "atlas_run",
            "dense_atlas",
            "verdict-gate",
            [
                cfg.python,
                "scripts/studio/atlas_multi_encoder_grid.py",
                "--seeds",
                cfg.atlas_seeds,
                "--abstraction-seeds",
                cfg.atlas_seeds,
                "--out",
                "runs/mot/atlas_multi_encoder_grid.json",
            ],
            receipts=["runs/mot/atlas_multi_encoder_grid.json"],
            note="full registered atlas run; no --allow-partial in the spine path",
        ),
        _step(
            "atlas_verdict_ledger",
            "dense_atlas",
            "report",
            [
                cfg.python,
                "scripts/studio/atlas_verdict_ledger.py",
                "--atlas",
                "runs/mot/atlas_multi_encoder_grid.json",
                "--dense-gate",
                "runs/mot/dense_atlas_cache_gate.json",
                "--out",
                "runs/mot/atlas_verdict_ledger.json",
            ],
            receipts=["runs/mot/atlas_verdict_ledger.json"],
            note="classify the raw atlas result against its null card before bundling or score movement",
        ),
        _step(
            "atlas_artifact_bundle",
            "dense_atlas",
            "artifact-bundle",
            _studio_cmd(
                cfg.python,
                "artifact-bundle",
                "--only-paths",
                "--path",
                "proof/NULL_CARDS/atlas_dense_multiencoder.md",
                "--path",
                "runs/mot/dense_encode_device.json",
                "--path",
                "runs/mot/dense_encode_schedule.json",
                "--path",
                "runs/mot/dense_atlas_cache_gate.json",
                "--path",
                str(Path("data") / "cache" / cfg.dense_cache / "cache_manifest.json"),
                "--path",
                str(Path("data") / "cache" / cfg.dense_randominit_cache / "cache_manifest.json"),
                "--path",
                "runs/mot/atlas_multi_encoder_grid.json",
                "--path",
                "runs/mot/atlas_verdict_ledger.json",
                "--copy-dir",
                "proof/ARTIFACT_BUNDLES/atlas",
                "--require-durable",
                "--out",
                "proof/ARTIFACT_INDEX/atlas.json",
            ),
            receipts=["proof/ARTIFACT_INDEX/atlas.json"],
            note="preserve dense/atlas receipts; large cache arrays remain represented by manifests",
        ),
        _step(
            "studio_scorecard",
            "finalize",
            "report",
            _studio_cmd(
                cfg.python,
                "scorecard",
                "--apply",
                "--out",
                "runs/studio_scorecard.json",
                "--allow-incomplete",
            ),
            receipts=["runs/studio_scorecard.json"],
            note=(
                "synthesize the receipt-backed Studio scorecard without stopping final preservation "
                "on blockers"
            ),
        ),
        _step(
            "spine_status_receipt",
            "finalize",
            "report",
            _studio_cmd(
                cfg.python,
                "spine-plan",
                "--status",
                "--plan",
                str(cfg.spine_dir / "spine_plan.json"),
                "--status-out",
                str(cfg.spine_dir / "spine_status.json"),
            ),
            receipts=[str(cfg.spine_dir / "spine_status.json")],
            note="write the receipt-aware spine status before the final artifact bundle",
        ),
        _step(
            "studio_objective_audit",
            "finalize",
            "report",
            _studio_cmd(
                cfg.python,
                "objective-audit",
                "--out",
                "runs/studio_objective_audit.json",
                "--spine-status",
                str(cfg.spine_dir / "spine_status.json"),
                "--scorecard",
                "runs/studio_scorecard.json",
                "--allow-not-ready",
            ),
            receipts=["runs/studio_objective_audit.json"],
            note="reevaluate the active Studio 10/10 objective from final receipts before bundling",
        ),
        _step(
            "spine_artifact_bundle",
            "finalize",
            "artifact-bundle",
            _bundle_cmd(
                cfg.python,
                "spine",
                "proof/ARTIFACT_INDEX/spine.json",
                "proof/ARTIFACT_BUNDLES/spine",
            ),
            receipts=["proof/ARTIFACT_INDEX/spine.json"],
            note="final index of the staged spine plans and wave artifact indexes",
        ),
    ]
    plan = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile.as_dict(),
        "source": cfg.source,
        "python": cfg.python,
        "planned_clips": int(cfg.planned_clips),
        "dense_planned_clips": int(cfg.dense_planned_clips),
        "cache_names": {"dr1": cfg.dr1_cache, "dense": cfg.dense_cache},
        "subplans": {
            "wave0_daemon_plan": str(wave0_plan),
            "dr1_schedule_plan": str(dr1_schedule),
            "dr1_daemon_plan": str(dr1_daemon),
            "dr1_source_intake": str(dr1_source_intake),
            "dr1_source_card": str(dr1_source_card),
        },
        "steps": steps,
        "expected_receipts": _dedupe_receipts(steps),
        "blocked_conditions": [
            "Wave 0 must run on the actual M1 Ultra profile before DR1 schedule generation is meaningful",
            (
                "DR1 source intake must prove source layout, captions, license/provenance, "
                "and non-overlap before encode"
            ),
            "DR1 must produce dr1_verification.json before PR9 or any DR1 positive claim can proceed",
            "Process C must stay unlaunched unless runs/mot/process_c_license_gate.json has "
            "launch_allowed true",
            "Dense-token atlas scope is blocked until data/cache/"
            f"{cfg.dense_cache} and {cfg.dense_randominit_cache} cache manifests validate as a dense pair "
            "and atlas runs without --allow-partial and writes a verdict ledger",
        ],
        "summary": {
            "steps": len(steps),
            "phases": _phase_summary(steps),
            "profile": profile.name,
        },
    }
    problems = validate_studio_spine_plan(plan)
    if problems:
        raise ValueError("; ".join(problems))
    return plan


def validate_studio_spine_plan(plan: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if plan.get("schema") != SCHEMA:
        problems.append(f"schema {plan.get('schema')!r} != {SCHEMA!r}")
    profile_name = str(plan.get("profile", {}).get("name", ""))
    if not is_studio_profile(profile_name):
        problems.append("profile must be a registered Studio resource envelope")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return [*problems, "steps must be a non-empty list"]
    seen_ids: set[str] = set()
    last_phase = -1
    phase_order = {"wave0": 0, "dr1": 1, "pr9": 2, "dense_atlas": 3, "finalize": 4}
    positions: dict[str, int] = {}
    for i, step in enumerate(steps):
        sid = str(step.get("id", ""))
        if not sid:
            problems.append(f"step at index {i} has no id")
        if sid in seen_ids:
            problems.append(f"duplicate step id {sid!r}")
        seen_ids.add(sid)
        positions[sid] = i
        phase = str(step.get("phase", ""))
        rank = phase_order.get(phase)
        if rank is None:
            problems.append(f"step {sid!r} has unknown phase {phase!r}")
        elif rank < last_phase:
            problems.append(f"step {sid!r} regresses phase order")
        elif rank is not None:
            last_phase = rank
        cmd = step.get("cmd")
        if not isinstance(cmd, list) or not all(isinstance(part, str) and part for part in cmd):
            problems.append(f"step {sid!r} needs a non-empty string cmd list")
        if sid == "atlas_run" and "--allow-partial" in (cmd or []):
            problems.append("atlas_run must not use --allow-partial")
    required = {
        "wave0_run",
        "wave0_artifact_bundle",
        "dr1_source_card_validate",
        "dr1_schedule_build",
        "dr1_source_intake",
        "dr1_validate_plan",
        "dr1_run",
        "dr1_artifact_bundle",
        "pr9_run",
        "pr9_verdict_ledger",
        "process_c_license_gate",
        "pr9_artifact_bundle",
        "dense_cache_plan",
        "dense_atlas_cache_gate",
        "atlas_run",
        "atlas_verdict_ledger",
        "atlas_artifact_bundle",
        "studio_scorecard",
        "spine_status_receipt",
        "studio_objective_audit",
        "spine_artifact_bundle",
    }
    missing = sorted(required.difference(seen_ids))
    if missing:
        problems.append(f"missing required spine step(s): {missing}")
    _must_precede(problems, positions, "dr1_source_card_validate", "dr1_source_intake")
    _must_precede(problems, positions, "dr1_source_intake", "dr1_schedule_build")
    _must_precede(problems, positions, "dr1_schedule_build", "dr1_validate_plan")
    _must_precede(problems, positions, "dr1_validate_plan", "dr1_run")
    _must_precede(problems, positions, "dr1_run", "dr1_artifact_bundle")
    _must_precede(problems, positions, "dr1_artifact_bundle", "pr9_run")
    _must_precede(problems, positions, "pr9_run", "pr9_verdict_ledger")
    _must_precede(problems, positions, "pr9_verdict_ledger", "process_c_license_gate")
    _must_precede(problems, positions, "process_c_license_gate", "pr9_artifact_bundle")
    _must_precede(problems, positions, "pr9_artifact_bundle", "dense_cache_plan")
    _must_precede(problems, positions, "dense_atlas_cache_gate", "atlas_run")
    _must_precede(problems, positions, "atlas_run", "atlas_verdict_ledger")
    _must_precede(problems, positions, "atlas_verdict_ledger", "atlas_artifact_bundle")
    _must_precede(problems, positions, "atlas_artifact_bundle", "studio_scorecard")
    _must_precede(problems, positions, "studio_scorecard", "spine_status_receipt")
    _must_precede(problems, positions, "spine_status_receipt", "studio_objective_audit")
    _must_precede(problems, positions, "atlas_artifact_bundle", "spine_artifact_bundle")
    _must_precede(problems, positions, "studio_objective_audit", "spine_artifact_bundle")
    return problems


def load_studio_spine_plan(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    problems = validate_studio_spine_plan(data)
    if problems:
        raise ValueError("; ".join(problems))
    return data


def build_studio_spine_status(
    plan: dict[str, Any] | Path | str,
    *,
    repo_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    loaded = load_studio_spine_plan(plan) if isinstance(plan, str | Path) else plan
    problems = validate_studio_spine_plan(loaded)
    if problems:
        raise ValueError("; ".join(problems))
    root = Path(repo_root)
    steps = [_step_status(step, root) for step in loaded["steps"]]
    next_step = next((step for step in steps if step["status"] != "complete"), None)
    summary = {
        "total": len(steps),
        "complete": sum(1 for step in steps if step["status"] == "complete"),
        "pending": sum(1 for step in steps if step["status"] == "pending"),
        "running": sum(1 for step in steps if step["status"] == "running"),
        "blocked": sum(1 for step in steps if step["status"] == "blocked"),
        "failed": sum(1 for step in steps if step["status"] == "failed"),
    }
    return {
        "schema": STATUS_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "plan_schema": loaded.get("schema"),
        "profile": loaded.get("profile", {}).get("name"),
        "repo_root": str(root),
        "all_complete": next_step is None,
        "summary": summary,
        "next_step": None if next_step is None else _next_step_record(next_step),
        "steps": steps,
    }


def write_studio_spine_status(status: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, default=str) + "\n")


def write_studio_spine_plan(plan: dict[str, Any], path: Path | str) -> None:
    problems = validate_studio_spine_plan(plan)
    if problems:
        raise ValueError("; ".join(problems))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, default=str) + "\n")


def write_spine_wave0_plan(path: Path | str) -> dict[str, Any]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plan = write_plan_template(path)
    load_plan(path)
    return plan


def _step(
    step_id: str,
    phase: str,
    kind: str,
    cmd: list[str],
    *,
    receipts: list[str],
    note: str,
) -> dict[str, Any]:
    return {
        "id": step_id,
        "phase": phase,
        "kind": kind,
        "cmd": cmd,
        "expected_receipts": receipts,
        "resume": "rerun this exact command; completed daemon jobs and PR9 legs skip finished work",
        "note": note,
    }


def _bundle_cmd(python: str, preset: str, out: str, copy_dir: str) -> list[str]:
    return _studio_cmd(
        python,
        "artifact-bundle",
        "--preset",
        preset,
        "--copy-dir",
        copy_dir,
        "--require-durable",
        "--out",
        out,
    )


def _studio_cmd(python: str, *parts: str) -> list[str]:
    return [python, "-m", "scripts.studio", *parts]


def _dedupe_receipts(steps: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for step in steps:
        for receipt in step.get("expected_receipts", []):
            if receipt not in seen:
                seen.add(receipt)
                out.append(receipt)
    return out


def _phase_summary(steps: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for step in steps:
        phase = str(step.get("phase", ""))
        out[phase] = out.get(phase, 0) + 1
    return out


def _must_precede(problems: list[str], positions: dict[str, int], before: str, after: str) -> None:
    if before in positions and after in positions and positions[before] >= positions[after]:
        problems.append(f"{before!r} must precede {after!r}")


def _step_status(step: dict[str, Any], root: Path) -> dict[str, Any]:
    receipts = [str(path) for path in step.get("expected_receipts", [])]
    receipt_records = [_receipt_status(root, receipt) for receipt in receipts]
    missing = [rec["display_path"] for rec in receipt_records if not rec["exists"]]
    problems: list[str] = []
    signals: list[dict[str, Any]] = []
    status = "pending" if missing else "complete"

    for rec in receipt_records:
        if rec["exists"] and rec["json_problem"]:
            problems.append(f"{rec['display_path']}: {rec['json_problem']}")
            status = "failed"
        signal = rec.get("signal")
        if signal:
            signals.append(signal)
            signal_status = signal.get("status")
            if signal_status in {"failed", "blocked", "running"}:
                status = _max_status(status, str(signal_status))
            elif signal_status == "dry-run" and status == "complete":
                status = "pending"

    if missing and status == "complete":
        status = "pending"
    if missing and status not in {"failed", "blocked", "running"}:
        status = "pending"

    return {
        "id": step["id"],
        "phase": step["phase"],
        "kind": step["kind"],
        "status": status,
        "cmd": list(step["cmd"]),
        "cmd_shell": shlex.join(step["cmd"]),
        "note": step.get("note", ""),
        "missing_receipts": missing,
        "receipts": receipt_records,
        "signals": signals,
        "problems": problems,
    }


def _receipt_status(root: Path, display_path: str) -> dict[str, Any]:
    path = _resolve_display_path(root, display_path)
    exists = path.exists() and path.is_file()
    json_problem = ""
    signal: dict[str, Any] | None = None
    if exists and path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            json_problem = f"invalid JSON: {e}"
        else:
            if not isinstance(data, dict):
                json_problem = "invalid JSON receipt: top-level value is not an object"
            else:
                signal = _json_signal(display_path, data)
    return {
        "display_path": display_path,
        "path": str(path),
        "exists": exists,
        "json_problem": json_problem,
        "signal": signal,
    }


def _json_signal(display_path: str, data: dict[str, Any]) -> dict[str, Any] | None:
    schema = data.get("schema")
    if display_path.endswith("daemon_state.json") and schema == "mop-long-run-daemon/v1":
        statuses = [str(job.get("status", "")) for job in data.get("jobs", {}).values()]
        summary = data.get("summary", {})
        if "running" in statuses:
            return {"status": "running", "detail": "daemon job is running", "summary": summary}
        if int(summary.get("failed", 0)) > 0:
            return {"status": "failed", "detail": "daemon has failed job(s)", "summary": summary}
        if int(summary.get("blocked", 0)) > 0:
            return {"status": "blocked", "detail": "daemon has blocked job(s)", "summary": summary}
        if int(summary.get("dry-run", 0)) > 0:
            return {
                "status": "dry-run",
                "detail": "dry-run state is not executed evidence",
                "summary": summary,
            }
        if statuses and all(status == "success" for status in statuses):
            return {"status": "complete", "detail": "daemon jobs succeeded", "summary": summary}
    if schema == "mop-artifact-bundle/v1" and data.get("all_ok") is False:
        return {
            "status": "failed",
            "detail": "artifact bundle reports problems",
            "problems": data.get("problems", []),
        }
    if schema == "mop-dr1-schedule-plan/v1" and data.get("ok_to_launch") is False:
        return {
            "status": "blocked",
            "detail": "DR1 schedule is blocked",
            "blocked_reasons": data.get("blocked_reasons", []),
        }
    if schema == "mop-dr1-source-intake/v1" and data.get("all_ok") is False:
        return {
            "status": "blocked",
            "detail": "DR1 source intake is blocked",
            "problems": data.get("problems", []),
        }
    if schema == "mop-dr1-source-card-validation/v1" and data.get("all_ok") is False:
        return {
            "status": "blocked",
            "detail": "DR1 source card validation is blocked",
            "problems": data.get("problems", []),
        }
    if display_path.endswith("encode_schedule.json") and data.get("ok_to_launch") is False:
        return {
            "status": "blocked",
            "detail": "encode schedule is blocked",
            "blocked_reasons": data.get("blocked_reasons", []),
        }
    if schema == "mop-dr1-adversarial-verification/v1" and data.get("integrity_ok") is False:
        return {
            "status": "failed",
            "detail": "DR1 verifier integrity failed",
            "problems": data.get("problems", []),
        }
    if schema == "mop-pr9-verdict-ledger/v1" and data.get("all_ok") is False:
        status = str(data.get("status") or "")
        signal_status = "failed" if status in {"config_error", "non_scoring", "indeterminate"} else "blocked"
        return {
            "status": signal_status,
            "detail": "PR9 verdict ledger is not complete/scoring",
            "ledger_status": status,
            "problems": data.get("problems", []),
        }
    if schema == "mop-process-c-license-gate/v1" and data.get("all_ok") is False:
        return {
            "status": "blocked",
            "detail": "Process C license gate is undecidable",
            "ledger_status": data.get("status"),
            "problems": data.get("problems", []),
        }
    if schema == "mop-dense-atlas-cache-gate/v1" and data.get("all_ok") is False:
        return {
            "status": "blocked",
            "detail": "dense atlas cache pair is not launchable",
            "problems": data.get("problems", []),
        }
    if schema == "mop-atlas-verdict-ledger/v1" and data.get("all_ok") is False:
        status = str(data.get("status") or "")
        signal_status = "failed" if status in {"dense_gate_invalid", "indeterminate"} else "blocked"
        return {
            "status": signal_status,
            "detail": "atlas verdict ledger is not complete/scoring",
            "ledger_status": status,
            "problems": data.get("problems", []),
        }
    if schema == "mop-studio-objective-audit/v1" and data.get("studio_10_ready") is False:
        return {
            "status": "blocked",
            "detail": "Studio objective audit is not 10/10 ready",
            "summary": data.get("summary", {}),
        }
    return None


def _resolve_display_path(root: Path, display_path: str) -> Path:
    path = Path(display_path)
    return path if path.is_absolute() else root / path


def _max_status(current: str, incoming: str) -> str:
    order = {"complete": 0, "pending": 1, "running": 2, "blocked": 3, "failed": 4}
    return incoming if order.get(incoming, 0) > order.get(current, 0) else current


def _next_step_record(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": step["id"],
        "phase": step["phase"],
        "kind": step["kind"],
        "status": step["status"],
        "cmd": step["cmd"],
        "cmd_shell": step["cmd_shell"],
        "missing_receipts": step["missing_receipts"],
        "note": step["note"],
        "signals": step["signals"],
        "problems": step["problems"],
    }
