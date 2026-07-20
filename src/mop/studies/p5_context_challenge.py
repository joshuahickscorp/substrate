
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from ..substrate.p5_context import run_p5_pilot
from .p5_context_verify import (
    CHALLENGE_CONTROLS,
    CHALLENGE_EVIDENCE_CLASS,
    CHALLENGE_PROMOTION,
    CHALLENGE_RESOURCE_CONTRACT,
    CHALLENGE_SCHEMA,
    CHALLENGE_SOURCE_PATHS,
    CLAIM_SCOPE,
    DEFAULT_CHALLENGE,
    DEFAULT_CONFIG,
    DEFAULT_PRIMARY,
    DEFAULT_PRIMARY_RUN_DIR,
    FRESH_TRAINING_SEEDS,
    PILOT_PROFILE,
    SOURCE_PATHS,
    P5VerificationRefused,
    PrimaryAudit,
    atomic_json,
    audit_primary,
    canonical_sha256,
    challenge_seed_config,
    display_path,
    file_sha256,
    source_bindings,
)

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "p5_context" / "fresh_challenge"
FORECAST_WRITE_BYTES = 8_000_000_000
REQUIRED_MEMORY_BYTES = 10_000_000_000
DISK_FLOOR_BYTES = 40_000_000_000


def _patterns(primary: PrimaryAudit) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in pattern.items() if key != "primary_ci"} for pattern in primary.patterns
    ]


def _row_problems(
    raw: dict[str, Any],
    cells: dict[int, dict[str, Any]],
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    problems: list[str] = []
    payload = dict(raw)
    declared = payload.pop("payload_sha256", None)
    if declared != canonical_sha256(payload):
        problems.append("fresh raw receipt payload digest mismatch")
    if raw.get("config_sha256") != canonical_sha256(config):
        problems.append("fresh raw receipt config digest mismatch")
    expected_sources = source_bindings(SOURCE_PATHS, repo_root)
    if raw.get("source_bindings") != expected_sources:
        problems.append("fresh raw receipt source binding drift")
    source_sha = canonical_sha256(expected_sources)
    checkpoint_sha = canonical_sha256(
        {
            "registry_sha256": raw.get("cell_registry_sha256"),
            "source_bindings_sha256": source_sha,
        }
    )
    if raw.get("source_bindings_sha256") != source_sha:
        problems.append("fresh raw aggregate source binding drift")
    if raw.get("checkpoint_requirements_sha256") != checkpoint_sha:
        problems.append("fresh raw checkpoint source identity drift")
    if raw.get("complete") is not True or raw.get("all_ok") is not True:
        problems.append("fresh raw full surface incomplete or all_ok false")
    if raw.get("problems") != [] or raw.get("resumable") is not False:
        problems.append("fresh raw full surface retains problems or resumability")
    for frames, cell in cells.items():
        if cell.get("complete") is not True or cell.get("all_ok") is not True:
            problems.append(f"fresh f{frames} cell incomplete or all_ok false")
        if cell.get("problems") != []:
            problems.append(f"fresh f{frames} cell retains problems")
    if any(
        raw.get(flag) is not False
        for flag in (
            "stopped_for_wall_budget",
            "stopped_for_disk_floor",
            "stopped_for_required_arm_refusal",
        )
    ):
        problems.append("fresh raw subrun stopped for an operational reason")
    if raw.get("required_arm_failure") is not None:
        problems.append("fresh raw subrun retains a required-arm failure")
    return problems


def _base_receipt(
    primary: PrimaryAudit,
    primary_path: Path,
    run_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema": CHALLENGE_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": CHALLENGE_EVIDENCE_CLASS,
        "source_bindings": source_bindings(CHALLENGE_SOURCE_PATHS, repo_root),
        "primary_receipt": {
            "path": display_path(primary_path, repo_root),
            "sha256": file_sha256(primary_path),
            "payload_sha256": primary.receipt["payload_sha256"],
        },
        "primary_run_dir": display_path(primary.run_dir, repo_root),
        "run_dir": display_path(run_dir, repo_root),
        "patterns": _patterns(primary),
        "fresh_training_seeds": list(FRESH_TRAINING_SEEDS),
        "fresh_seeds_disjoint_from_primary": not (set(FRESH_TRAINING_SEEDS) & set(primary.receipt["seeds"])),
        "controls": dict(CHALLENGE_CONTROLS),
        "checkpoint_globs": [
            f"{display_path(run_dir, repo_root)}/seed_*/frames/f*/seed_*/*/checkpoint.pt",
            f"{display_path(run_dir, repo_root)}/seed_*/frames/f*/seed_*/*/arm_receipt.json",
            f"{display_path(run_dir, repo_root)}/seed_*/frames/f*/seed_*/seed_result.json",
            f"{display_path(run_dir, repo_root)}/seed_*/frames/f*/cell_receipt.json",
            f"{display_path(run_dir, repo_root)}/seed_*/p5_context_receipt.json",
            f"{display_path(run_dir, repo_root)}/seed_*/resolved_config.json",
        ],
        "resource_contract": dict(CHALLENGE_RESOURCE_CONTRACT),
        "training_runs": [],
        "complete": False,
        "resumable": False,
        "verification_ready": False,
        "problems": [],
        "all_ok": False,
        "promotion": dict(CHALLENGE_PROMOTION),
        "scientific_promotion": False,
    }


def _finalize(receipt: dict[str, Any], expected_count: int) -> None:
    rows = receipt["training_runs"]
    row_problems = [f"seed {row['seed']}: {problem}" for row in rows for problem in row["problems"]]
    complete = len(rows) == expected_count and all(row["complete"] is True for row in rows)
    resumable = any(row["resumable"] is True for row in rows)
    problems = list(row_problems)
    if len(rows) != expected_count:
        problems.append(f"fresh training run coverage incomplete: {len(rows)}/{expected_count}")
    receipt.update(
        {
            "complete": complete,
            "resumable": resumable,
            "verification_ready": complete and not problems,
            "problems": problems,
            "all_ok": complete and not problems,
        }
    )
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = canonical_sha256(receipt)


def run_fresh_challenge(
    primary_path: Path = DEFAULT_PRIMARY,
    primary_run_dir: Path = DEFAULT_PRIMARY_RUN_DIR,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
    output: Path = DEFAULT_CHALLENGE,
    device: str = "cpu",
    *,
    repo_root: Path = REPO_ROOT,
    runner: Callable[..., dict[str, Any]] = run_p5_pilot,
) -> dict[str, Any]:
    primary = audit_primary(primary_path, primary_run_dir, config_path, repo_root=repo_root)
    if primary.profile != PILOT_PROFILE:
        raise P5VerificationRefused("fresh training challenge is authorized only for p5pilot")
    if not primary.patterns:
        raise P5VerificationRefused(
            "P5 primary has no favorable programmatic pattern, so fresh training is not authorized"
        )
    if set(FRESH_TRAINING_SEEDS) & set(primary.receipt["seeds"]):
        raise P5VerificationRefused("fixed P5 fresh training seeds overlap the primary pilot")
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = _base_receipt(primary, primary_path, run_dir, repo_root)
    expected_count = len(FRESH_TRAINING_SEEDS)
    for seed in FRESH_TRAINING_SEEDS:
        config = challenge_seed_config(primary.config, seed)
        subrun = run_dir / f"seed_{seed}"
        raw = runner(config, subrun, device, repo_root=repo_root)
        cells: dict[int, dict[str, Any]] = {}
        cell_bindings: dict[str, dict[str, Any]] = {}
        missing_cells: list[str] = []
        for frames in (64, 32, 16):
            cell_path = subrun / "frames" / f"f{frames}" / "cell_receipt.json"
            if not cell_path.is_file():
                missing_cells.append(f"fresh f{frames} cell receipt is missing")
                continue
            cells[frames] = json.loads(cell_path.read_text(encoding="utf-8"))
            cell_bindings[f"f{frames}"] = {
                "path": display_path(cell_path, repo_root),
                "sha256": file_sha256(cell_path),
            }
        problems = missing_cells
        if not problems:
            problems.extend(_row_problems(raw, cells, config, repo_root=repo_root))
        raw_path = subrun / "p5_context_receipt.json"
        config_file = subrun / "resolved_config.json"
        row = {
            "seed": seed,
            "raw_receipt": {
                "path": display_path(raw_path, repo_root),
                "sha256": file_sha256(raw_path) if raw_path.is_file() else None,
                "payload_sha256": raw.get("payload_sha256"),
            },
            "cell_receipts": cell_bindings,
            "resolved_config": {
                "path": display_path(config_file, repo_root),
                "sha256": file_sha256(config_file) if config_file.is_file() else None,
            },
            "complete": raw.get("complete") is True and len(cells) == 3,
            "resumable": raw.get("resumable") is True,
            "problems": problems,
            "all_ok": not problems,
        }
        receipt["training_runs"].append(row)
        _finalize(receipt, expected_count)
        atomic_json(output, receipt)
        if row["complete"] is not True or row["all_ok"] is not True:
            return receipt
    _finalize(receipt, expected_count)
    atomic_json(output, receipt)
    return receipt


def challenge_exit_code(receipt: dict[str, Any]) -> int:
    if receipt.get("complete") is True and receipt.get("all_ok") is True:
        return 0
    if receipt.get("resumable") is True:
        return 2
    return 1


__all__ = [
    "DEFAULT_RUN_DIR",
    "DISK_FLOOR_BYTES",
    "FORECAST_WRITE_BYTES",
    "REQUIRED_MEMORY_BYTES",
    "challenge_exit_code",
    "run_fresh_challenge",
]
