"""Independent integrity and adversarial verifier for the P5 context pilot.

The verifier never imports the P5 training or evaluation implementation. It reads a sealed smoke
or pilot primary, its raw per-frame receipts, and an optional separately governed fresh-training challenge.
It recomputes paired contrasts from per-seed scores, treats every threshold tie as a null, and
allows no confirmatory promotion. A directional programmatic pattern remains unverified until
three fixed, disjoint training seeds reproduce that direction through the challenge runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from ..config import REPO_ROOT

PILOT_SCHEMA = "mop-p5-context-screen/v1"
CELL_SCHEMA = "mop-p5-context-cell/v1"
SEED_SCHEMA = "mop-p5-context-seed/v1"
CHALLENGE_SCHEMA = "mop-p5-context-fresh-training-challenge/v1"
VERIFY_SCHEMA = "mop-p5-context-independent-verifier/v1"
ARM_SCHEMA = "mop-custom-substrate-arm/v1"
CHECKPOINT_SCHEMA = "mop-custom-substrate-checkpoint/v1"
P5_OBJECTIVE = "predictive"

CLAIM_SCOPE = (
    "exact-versus-factorized context pilot on deterministic programmatic video; "
    "not natural-video, memory-rung, or general-capability evidence"
)
EVIDENCE_CLASS = "R1 independently recomputed programmatic pilot evidence"
SMOKE_PROFILE = "p5smoke"
PILOT_PROFILE = "p5pilot"
SUPPORTED_PRIMARY_PROFILES = (SMOKE_PROFILE, PILOT_PROFILE)
FRESH_TRAINING_SEEDS = (5101, 5102, 5103)
FRAME_COUNTS = (64, 32, 16)
PRIMARY_FRAMES = (64, 32)
MECHANISMS = ("exact_global", "window_local", "recurrent", "hierarchical_pooled")
EXPECTED_TRANSFORMER_PARAMETERS = {16: 1_678_848, 32: 1_744_384, 64: 1_875_456}
RECURRENT_PARAMETER_DEFICIT = 512
FLOP_MATCH_TOLERANCE = 0.02
TRAINABILITY_MARGIN = 0.05
CHALLENGE_EVIDENCE_CLASS = "R1 fresh disjoint training challenge on deterministic programmatic video"
CHALLENGE_CONTROLS = {
    "shared_primary_training_contract": True,
    "matched_parameter_and_flop_contract": True,
    "same_initialization_frozen_control": True,
    "f64_trainability_gate": True,
    "threshold_tie_is_null": True,
    "isolated_full_surface_seed_subruns": True,
}
CHALLENGE_RESOURCE_CONTRACT = {
    "resource_class": "exclusive-heavy-cpu",
    "forecast_write_bytes": 8_000_000_000,
    "required_memory_bytes": 10_000_000_000,
    "disk_floor_bytes": 40_000_000_000,
    "resumable_exit_code": 2,
}
CHALLENGE_PROMOTION = {
    "confirmatory_promotable": False,
    "refused_by_construction": True,
    "scientific_capability_claim": False,
}
PILOT_PROMOTION = {
    "confirmatory_promotable": False,
    "refused_by_construction": True,
    "reason": "context pilot; confirmatory claims refused by construction",
    "category_9_possible": False,
    "category_9_statement": (
        "category 9 is impossible from this pilot: it trains deterministic programmatic "
        "video at CM7-class width on one host and cannot satisfy the P5 promotion gate "
        "(three repeated exact-model failures against the runtime envelope, measured "
        "headroom, every valid factorization attempted, and a calculated smallest rung)"
    ),
    "scope_boundary": (
        "contrasts rank context mechanisms on this deterministic programmatic task only; "
        "they license no natural-video, memory-rung, or general-capability claim"
    ),
}
SOURCE_PATHS = (
    "configs/experiment/mop_p5_context_capability.yaml",
    "scripts/p5_context_capability.py",
    "src/mop/substrate/p5_context.py",
    "src/mop/substrate/custom_workbench.py",
    "src/mop/substrate/p4_screen.py",
)
CHALLENGE_SOURCE_PATHS = SOURCE_PATHS + (
    "scripts/p5_context_fresh_challenge.py",
    "src/mop/studies/p5_context_challenge.py",
    "src/mop/studies/p5_context_verify.py",
)
VERIFIER_SOURCE_PATHS = SOURCE_PATHS + (
    "scripts/verify_p5_context_capability.py",
    "src/mop/studies/p5_context_verify.py",
)

DEFAULT_PRIMARY = REPO_ROOT / "proof" / "P5_CONTEXT_CAPABILITY_PILOT.json"
DEFAULT_PRIMARY_RUN_DIR = REPO_ROOT / "runs" / "p5_context" / "p5pilot"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "mop_p5_context_capability.yaml"
DEFAULT_CHALLENGE = REPO_ROOT / "proof" / "P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json"
DEFAULT_VERIFICATION = REPO_ROOT / "proof" / "P5_CONTEXT_CAPABILITY_VERIFICATION.json"


class P5VerificationRefused(ValueError):
    """Raised when a required P5 artifact is absent, stale, incomplete, or semantically invalid."""


@dataclass
class PrimaryAudit:
    receipt_path: Path
    run_dir: Path
    receipt: dict[str, Any]
    raw_receipt: dict[str, Any]
    config: dict[str, Any]
    profile: str
    cells: dict[int, dict[str, Any]]
    recomputed: dict[int, dict[str, Any]]
    patterns: list[dict[str, Any]]
    outcome: str
    terminal_null: bool


@dataclass
class ChallengeAudit:
    receipt_path: Path
    receipt: dict[str, Any]
    per_pattern: list[dict[str, Any]]
    artifact_evidence: dict[str, Any]
    raw_receipts: dict[int, dict[str, Any]]
    recomputed: dict[int, dict[int, dict[str, Any]]]


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: Mapping[str, Any], label: str) -> str:
    """Hash a checkpoint state independently of the P5 runner implementation."""

    if not isinstance(state, Mapping) or not state:
        raise P5VerificationRefused(f"{label} is not a nonempty tensor state")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise P5VerificationRefused(f"{label} contains a non-tensor state entry")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(
                list(tensor.shape),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _sha256_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise P5VerificationRefused(f"{label} is not a lowercase SHA-256 digest")
    return value


def _checkpoint_globs(run_dir: Path, repo_root: Path) -> list[str]:
    root = display_path(run_dir, repo_root)
    return [
        f"{root}/seed_*/frames/f*/seed_*/*/checkpoint.pt",
        f"{root}/seed_*/frames/f*/seed_*/*/arm_receipt.json",
        f"{root}/seed_*/frames/f*/seed_*/seed_result.json",
        f"{root}/seed_*/frames/f*/cell_receipt.json",
        f"{root}/seed_*/p5_context_receipt.json",
        f"{root}/seed_*/resolved_config.json",
    ]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    os.replace(temporary, path)


def display_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def resolve_bound_path(value: Any, repo_root: Path = REPO_ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise P5VerificationRefused("bound artifact path is missing")
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else repo_root / candidate
    resolved = path.resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise P5VerificationRefused(f"bound artifact escapes the repository: {value}")
    return resolved


def source_bindings(paths: Sequence[str], repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in paths:
        path = repo_root / relative
        if not path.is_file():
            raise P5VerificationRefused(f"required live P5 source is missing: {relative}")
        rows.append({"path": relative, "file_sha256": file_sha256(path)})
    return rows


def resolved_profile_config(
    config_path: Path = DEFAULT_CONFIG, profile: str = PILOT_PROFILE
) -> dict[str, Any]:
    if not config_path.is_file():
        raise P5VerificationRefused(f"P5 config is missing: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise P5VerificationRefused("P5 config must be a mapping")
    config = json.loads(json.dumps(raw))
    profiles = config.pop("profiles", None)
    if profile not in SUPPORTED_PRIMARY_PROFILES:
        raise P5VerificationRefused(f"unsupported P5 primary profile: {profile}")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise P5VerificationRefused(f"P5 profile is missing from the live config: {profile}")
    if not isinstance(config.get("training"), dict):
        raise P5VerificationRefused("P5 training config is missing")
    config["training"] = {**config["training"], **profiles[profile]}
    config["profile"] = profile
    return config


def challenge_seed_config(primary_config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    if primary_config.get("profile") != PILOT_PROFILE:
        raise P5VerificationRefused("fresh training challenge is authorized only for p5pilot")
    config = copy.deepcopy(dict(primary_config))
    config["profile"] = f"p5fresh-seed-{int(seed)}"
    config["training"]["seeds"] = [int(seed)]
    return config


def _sealed(payload: Mapping[str, Any]) -> bool:
    digest = payload.get("payload_sha256")
    if not isinstance(digest, str):
        return False
    core = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return digest == canonical_sha256(core)


def _sealed_primary_profile(receipt: Mapping[str, Any]) -> str:
    if receipt.get("schema") != PILOT_SCHEMA or receipt.get("claim_scope") != CLAIM_SCOPE:
        raise P5VerificationRefused("wrong P5 primary schema or claim scope")
    if not _sealed(receipt):
        raise P5VerificationRefused("P5 primary payload digest mismatch")
    profile = receipt.get("profile")
    if profile not in SUPPORTED_PRIMARY_PROFILES:
        raise P5VerificationRefused(f"unsupported sealed P5 primary profile: {profile}")
    return str(profile)


def _seal(payload: dict[str, Any]) -> None:
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = canonical_sha256(payload)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise P5VerificationRefused(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise P5VerificationRefused(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise P5VerificationRefused(f"{label} must be a JSON object")
    return payload


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P5VerificationRefused(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise P5VerificationRefused(f"{label} is not a finite probability")
    return number


def _paired_ci(values: Sequence[float]) -> dict[str, Any]:
    count = len(values)
    mean = sum(values) / count if values else 0.0
    if count < 2:
        return {"n": count, "mean": mean, "lo": mean, "hi": mean, "half": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    half = 1.96 * math.sqrt(variance) / math.sqrt(count)
    return {"n": count, "mean": mean, "lo": mean - half, "hi": mean + half, "half": half}


def classify_ci(lo: float, hi: float, sesoi: float) -> str:
    if lo > sesoi:
        return "meaningful_positive"
    if hi < -sesoi:
        return "meaningful_negative"
    if lo >= -sesoi and hi <= sesoi:
        return "bounded_within_sesoi"
    return "undetermined"


def _expected_flops_per_step(frames: int, mechanism: str, batch_size: int) -> int:
    tokens = frames * 32
    dim = 128
    feedforward = 4 * dim
    depth = 8 if mechanism == "recurrent" else 4
    convolution = 2 * batch_size * tokens * dim * 3 * 2 * 32**2
    if mechanism == "exact_global":
        attention = 2 * 4 * batch_size * tokens * dim * dim + 4 * batch_size * tokens * tokens * dim
        mlp = 4 * batch_size * tokens * dim * feedforward
        per_layer = attention + mlp
    elif mechanism == "recurrent":
        per_layer = 2 * 6 * batch_size * tokens * dim * dim
    else:
        window = min(512, tokens)
        attention = 2 * 4 * batch_size * tokens * dim * dim + 4 * batch_size * tokens * window * dim
        mlp = 4 * batch_size * tokens * dim * feedforward
        per_layer = attention + mlp
        if mechanism == "hierarchical_pooled":
            summaries = max(1, tokens // window)
            summary_attention = (
                2 * 4 * batch_size * summaries * dim * dim + 4 * batch_size * summaries * summaries * dim
            )
            summary_mlp = 4 * batch_size * summaries * dim * feedforward
            pooling_scatter = 4 * batch_size * tokens * dim
            per_layer += summary_attention + summary_mlp + pooling_scatter
    encoder_forward = convolution + depth * per_layer
    predictor_forward = 4 * batch_size * tokens * dim * dim
    return int(
        4 * encoder_forward + 3 * predictor_forward + 2 * batch_size * tokens * dim + 2 * depth * dim * dim
    )


def _expected_match(
    dense_steps: int,
    dense_flops: int,
    arm_flops: int,
    checkpoint_every: int,
    *,
    exact: bool,
) -> dict[str, Any]:
    target = dense_steps * dense_flops
    if exact:
        steps = dense_steps
    else:
        grain = min(5, checkpoint_every)
        steps = max(1, round(target / arm_flops / grain)) * grain
    arm_total = steps * arm_flops
    deviation = abs(arm_total - target) / target
    return {
        "steps": steps,
        "target_total_flops": target,
        "arm_total_flops": arm_total,
        "fractional_deviation": deviation,
        "tolerance_fraction": FLOP_MATCH_TOLERANCE,
        "matched_ok": deviation <= FLOP_MATCH_TOLERANCE,
    }


def _expected_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = config.get("cells")
    if not isinstance(cells, list):
        raise P5VerificationRefused("P5 config cell table is missing")
    expected = [
        {"frames": frames, "mechanism": mechanism} for frames in FRAME_COUNTS for mechanism in MECHANISMS
    ]
    parsed = [{"frames": int(row["frames"]), "mechanism": str(row["mechanism"])} for row in cells]
    if parsed != expected:
        raise P5VerificationRefused("P5 config cell table drifted from the independent design table")
    return parsed


def _validate_promotion(receipt: Mapping[str, Any], label: str) -> None:
    promotion = receipt.get("promotion")
    if promotion != PILOT_PROMOTION:
        raise P5VerificationRefused(f"{label} promotion and scope-refusal block drifted")


def _validate_primary_header(
    receipt: Mapping[str, Any], config: Mapping[str, Any], profile: str, repo_root: Path
) -> tuple[list[int], float, bool]:
    if _sealed_primary_profile(receipt) != profile or config.get("profile") != profile:
        raise P5VerificationRefused("sealed P5 profile and resolved config profile disagree")
    if receipt.get("complete") is not True:
        raise P5VerificationRefused("P5 pilot is incomplete")
    if receipt.get("all_ok") is not True:
        raise P5VerificationRefused("P5 pilot all_ok is false")
    if receipt.get("resumable") is not False:
        raise P5VerificationRefused("completed P5 pilot cannot remain resumable")
    if receipt.get("problems") != []:
        raise P5VerificationRefused("completed P5 pilot retains problems")
    for flag in (
        "stopped_for_wall_budget",
        "stopped_for_disk_floor",
        "stopped_for_required_arm_refusal",
    ):
        if receipt.get(flag) is not False:
            raise P5VerificationRefused(f"P5 pilot operational stop flag is set: {flag}")
    if receipt.get("required_arm_failure") is not None:
        raise P5VerificationRefused("P5 pilot retains a required-arm failure")
    expected_sources = source_bindings(SOURCE_PATHS, repo_root)
    if receipt.get("source_bindings") != expected_sources:
        raise P5VerificationRefused("P5 pilot source binding is stale or incomplete")
    expected_source_sha = canonical_sha256(expected_sources)
    if receipt.get("source_bindings_sha256") != expected_source_sha:
        raise P5VerificationRefused("P5 pilot aggregate source binding drifted")
    if receipt.get("config_sha256") != canonical_sha256(config):
        raise P5VerificationRefused("P5 pilot resolved config binding drifted")
    cells = _expected_cells(config)
    if receipt.get("cell_registry_sha256") != canonical_sha256(cells):
        raise P5VerificationRefused("P5 pilot cell registry binding drifted")
    expected_checkpoint_sha = canonical_sha256(
        {
            "registry_sha256": receipt["cell_registry_sha256"],
            "source_bindings_sha256": expected_source_sha,
        }
    )
    if receipt.get("checkpoint_requirements_sha256") != expected_checkpoint_sha:
        raise P5VerificationRefused("P5 pilot checkpoint source identity drifted")
    expected_serial = [f"f{row['frames']}_{row['mechanism']}" for row in cells]
    if receipt.get("serial_order") != expected_serial:
        raise P5VerificationRefused("P5 pilot serial cell order drifted")
    training = config["training"]
    seeds = [int(value) for value in training["seeds"]]
    if receipt.get("seeds") != seeds or len(seeds) != len(set(seeds)):
        raise P5VerificationRefused("P5 pilot seed identity drifted")
    if receipt.get("dense_reference_steps") != int(training["dense_steps"]):
        raise P5VerificationRefused("P5 pilot dense step contract drifted")
    sesoi = float(config["screen"]["sesoi"])
    if receipt.get("sesoi") != sesoi or sesoi <= 0.0:
        raise P5VerificationRefused("P5 pilot SESOI drifted")
    _validate_promotion(receipt, "P5 pilot")

    terminal = receipt.get("terminal_scientific_stop") is True
    expected_status = "terminal-scientific-null" if terminal else "complete"
    if receipt.get("execution_status") != expected_status:
        raise P5VerificationRefused("P5 pilot execution status is inconsistent")
    expected_reason = "f64-trainability-gate-null" if terminal else None
    if receipt.get("terminal_stop_reason") != expected_reason:
        raise P5VerificationRefused("P5 pilot terminal stop reason drifted")
    if profile == SMOKE_PROFILE and not terminal:
        raise P5VerificationRefused(
            "p5smoke is verification-eligible only as a complete terminal-scientific-null"
        )
    return seeds, sesoi, terminal


def _audit_seed_artifacts(
    seed_dir: Path,
    embedded_seed: Mapping[str, Any],
    *,
    frames: int,
    seed: int,
    mechanisms: Sequence[str],
    config_sha256: str,
    data_sha256: str,
    requirements_sha256: str,
    expected_matches: Mapping[str, Mapping[str, Any]],
    expected_flops: Mapping[str, int],
    batch_size: int,
    repo_root: Path,
) -> dict[str, Any]:
    """Join a cell seed to its durable result, arm receipts, and final checkpoints."""

    seed_path = seed_dir / "seed_result.json"
    disk_seed = _read_json(seed_path, f"f{frames} seed {seed} durable result")
    embedded = copy.deepcopy(dict(embedded_seed))
    resume_annotation = embedded.pop("resumed_from_complete_receipt", None)
    if resume_annotation not in (None, True):
        raise P5VerificationRefused(f"f{frames} seed {seed} has an invalid resume annotation")
    if disk_seed != embedded:
        raise P5VerificationRefused(
            f"f{frames} seed {seed} durable result does not exactly join its cell payload"
        )
    initial_states = disk_seed.get("initial_state_sha256")
    if not isinstance(initial_states, dict) or set(initial_states) != set(mechanisms):
        raise P5VerificationRefused(f"f{frames} seed {seed} initial-state registry drifted")

    evidence: dict[str, Any] = {
        "seed_result": {
            "path": display_path(seed_path, repo_root),
            "sha256": file_sha256(seed_path),
        },
        "arms": {},
    }
    for mechanism in mechanisms:
        arm = disk_seed["mechanisms"][mechanism]
        initial_sha = _sha256_string(
            initial_states[mechanism],
            f"f{frames} seed {seed} {mechanism} initial-state digest",
        )
        if arm.get("initial_state_sha256") != initial_sha:
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} initial-state identity drifted")
        training = arm.get("training")
        if not isinstance(training, dict):
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} training receipt is missing")
        final_sha = _sha256_string(
            training.get("final_state_sha256"),
            f"f{frames} seed {seed} {mechanism} final-state digest",
        )
        arm_dir = seed_dir / mechanism
        arm_path = arm_dir / "arm_receipt.json"
        checkpoint_path = arm_dir / "checkpoint.pt"
        arm_receipt = _read_json(arm_path, f"f{frames} seed {seed} {mechanism} arm receipt")
        expected_steps = int(expected_matches[mechanism]["steps"])
        expected_total = int(expected_matches[mechanism]["arm_total_flops"])
        expected_arm_identity = {
            "schema": ARM_SCHEMA,
            "objective": P5_OBJECTIVE,
            "seed": seed,
            "complete": True,
            "stop_reason": None,
            "requested_steps": expected_steps,
            "completed_steps": expected_steps,
            "batch_size": batch_size,
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "requirements_sha256": requirements_sha256,
            "initial_state_sha256": initial_sha,
            "final_state_sha256": final_sha,
        }
        mismatched = [key for key, value in expected_arm_identity.items() if arm_receipt.get(key) != value]
        if mismatched:
            raise P5VerificationRefused(
                f"f{frames} seed {seed} {mechanism} arm identity drifted: {mismatched}"
            )
        compute = arm_receipt.get("compute")
        if (
            not isinstance(compute, dict)
            or compute.get("estimated_flops_per_step") != expected_flops[mechanism]
            or compute.get("estimated_total_flops") != expected_total
            or not isinstance(compute.get("estimator"), str)
            or not compute["estimator"]
        ):
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} arm compute drifted")
        loss = arm_receipt.get("loss")
        if not isinstance(loss, dict) or loss.get("count") != expected_steps:
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} arm loss count drifted")
        checkpoint_binding = arm_receipt.get("checkpoint")
        if not isinstance(checkpoint_binding, dict):
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} checkpoint binding is missing")
        bound_path = resolve_bound_path(checkpoint_binding.get("path"), repo_root)
        if bound_path != checkpoint_path.resolve():
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} checkpoint path drifted")
        if not checkpoint_path.is_file():
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} checkpoint is missing")
        checkpoint_sha = file_sha256(checkpoint_path)
        if (
            checkpoint_binding.get("sha256") != checkpoint_sha
            or checkpoint_binding.get("bytes") != checkpoint_path.stat().st_size
        ):
            raise P5VerificationRefused(
                f"f{frames} seed {seed} {mechanism} checkpoint file hash or size drifted"
            )
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise P5VerificationRefused(
                f"f{frames} seed {seed} {mechanism} checkpoint is unreadable: {exc}"
            ) from exc
        if not isinstance(checkpoint, dict):
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} checkpoint is not a mapping")
        expected_checkpoint_identity = {
            "schema": CHECKPOINT_SCHEMA,
            "objective": P5_OBJECTIVE,
            "step": expected_steps,
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "requirements_sha256": requirements_sha256,
            "initial_state_sha256": initial_sha,
        }
        checkpoint_mismatched = [
            key for key, value in expected_checkpoint_identity.items() if checkpoint.get(key) != value
        ]
        if checkpoint_mismatched:
            raise P5VerificationRefused(
                f"f{frames} seed {seed} {mechanism} checkpoint identity drifted: {checkpoint_mismatched}"
            )
        checkpoint_losses = checkpoint.get("losses")
        if (
            not isinstance(checkpoint_losses, list)
            or len(checkpoint_losses) != expected_steps
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in checkpoint_losses
            )
        ):
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} checkpoint loss history drifted")
        model_sha = state_sha256(
            checkpoint.get("model", {}),
            f"f{frames} seed {seed} {mechanism} checkpoint model",
        )
        target_sha = state_sha256(
            checkpoint.get("target", {}),
            f"f{frames} seed {seed} {mechanism} checkpoint target",
        )
        if model_sha != final_sha or arm_receipt.get("target_state_sha256") != target_sha:
            raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} checkpoint state hash drifted")
        if (
            training.get("completed_steps") != arm_receipt["completed_steps"]
            or training.get("requirements_sha256") != arm_receipt["requirements_sha256"]
            or training.get("estimated_flops_per_step") != compute["estimated_flops_per_step"]
            or training.get("estimated_total_flops") != compute["estimated_total_flops"]
        ):
            raise P5VerificationRefused(
                f"f{frames} seed {seed} {mechanism} seed-to-arm training join drifted"
            )
        evidence["arms"][mechanism] = {
            "arm_receipt": {
                "path": display_path(arm_path, repo_root),
                "sha256": file_sha256(arm_path),
            },
            "checkpoint": {
                "path": display_path(checkpoint_path, repo_root),
                "sha256": checkpoint_sha,
                "model_state_sha256": model_sha,
                "target_state_sha256": target_sha,
            },
        }
    return evidence


def _recompute_cell(
    cell: Mapping[str, Any],
    *,
    frame_dir: Path,
    frames: int,
    mechanisms: Sequence[str],
    allowed_seeds: Sequence[int],
    config_sha256: str,
    registry_sha256: str,
    source_bindings_sha256: str,
    checkpoint_requirements_sha256: str,
    dense_steps: int,
    checkpoint_every: int,
    batch_size: int,
    sesoi: float,
    repo_root: Path,
    audit_artifacts: bool = True,
) -> dict[str, Any]:
    if cell.get("schema") != CELL_SCHEMA or cell.get("frames") != frames:
        raise P5VerificationRefused(f"f{frames} cell receipt schema drifted")
    if cell.get("mechanisms") != list(mechanisms):
        raise P5VerificationRefused(f"f{frames} mechanism set or order drifted")
    if cell.get("complete") is not True or cell.get("all_ok") is not True:
        raise P5VerificationRefused(f"f{frames} cell receipt is incomplete or all_ok false")
    if cell.get("problems") != []:
        raise P5VerificationRefused(f"f{frames} cell receipt retains problems")
    difficulty = cell.get("difficulty_calibration")
    if not isinstance(difficulty, dict) or difficulty.get("clears_floor") is not True:
        raise P5VerificationRefused(f"f{frames} difficulty calibration did not clear its floor")
    corpus = cell.get("corpus")
    if not isinstance(corpus, dict) or not isinstance(corpus.get("content_sha256"), str):
        raise P5VerificationRefused(f"f{frames} corpus binding is missing")
    data_sha256 = str(corpus["content_sha256"])
    expected_seeds = cell.get("expected_seeds")
    if (
        not isinstance(expected_seeds, list)
        or not expected_seeds
        or any(isinstance(value, bool) or not isinstance(value, int) for value in expected_seeds)
    ):
        raise P5VerificationRefused(f"f{frames} expected seed set is missing")
    expected = list(expected_seeds)
    if len(expected) != len(set(expected)) or not set(expected) <= set(allowed_seeds):
        raise P5VerificationRefused(f"f{frames} expected seeds drifted")
    payloads = cell.get("seed_results")
    if not isinstance(payloads, dict) or set(payloads) != {str(seed) for seed in expected}:
        raise P5VerificationRefused(f"f{frames} raw seed result coverage drifted")
    if cell.get("seeds_completed") != len(expected):
        raise P5VerificationRefused(f"f{frames} completed seed count drifted")

    expected_parameters = {
        mechanism: (
            EXPECTED_TRANSFORMER_PARAMETERS[frames] - RECURRENT_PARAMETER_DEFICIT
            if mechanism == "recurrent"
            else EXPECTED_TRANSFORMER_PARAMETERS[frames]
        )
        for mechanism in mechanisms
    }
    parameter_block = cell.get("parameters")
    recurrent_deviation = RECURRENT_PARAMETER_DEFICIT / EXPECTED_TRANSFORMER_PARAMETERS[frames]
    if (
        not isinstance(parameter_block, dict)
        or parameter_block.get("frames") != frames
        or parameter_block.get("parameters") != expected_parameters
        or parameter_block.get("tolerance_fraction") != 0.005
        or not isinstance(parameter_block.get("recurrent_fractional_deviation"), (int, float))
        or not math.isclose(
            float(parameter_block["recurrent_fractional_deviation"]),
            recurrent_deviation,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise P5VerificationRefused(f"f{frames} parameter-matching control drifted")

    compute = cell.get("compute")
    expected_flops = {
        mechanism: _expected_flops_per_step(frames, mechanism, batch_size) for mechanism in mechanisms
    }
    dense_flops = _expected_flops_per_step(frames, "exact_global", batch_size)
    if (
        not isinstance(compute, dict)
        or compute.get("dense_reference_steps") != dense_steps
        or compute.get("dense_flops_per_step") != dense_flops
    ):
        raise P5VerificationRefused(f"f{frames} dense compute reference drifted")
    compute_rows = compute.get("per_mechanism")
    if not isinstance(compute_rows, dict) or set(compute_rows) != set(mechanisms):
        raise P5VerificationRefused(f"f{frames} compute arm coverage drifted")
    expected_matches = {
        mechanism: _expected_match(
            dense_steps,
            dense_flops,
            expected_flops[mechanism],
            checkpoint_every,
            exact=mechanism == "exact_global",
        )
        for mechanism in mechanisms
    }
    for mechanism in mechanisms:
        row = compute_rows[mechanism]
        matched = row.get("matched")
        if (
            row.get("estimated_flops_per_step") != expected_flops[mechanism]
            or not isinstance(matched, dict)
            or any(matched.get(key) != value for key, value in expected_matches[mechanism].items())
            or row.get("estimated_total_flops_completed_seeds")
            != expected_matches[mechanism]["arm_total_flops"] * len(expected)
        ):
            raise P5VerificationRefused(f"f{frames} {mechanism} matched-compute control drifted")

    trained: dict[str, list[float]] = {mechanism: [] for mechanism in mechanisms}
    trained_by_seed: dict[int, dict[str, float]] = {}
    frozen: dict[str, list[float]] = {mechanism: [] for mechanism in mechanisms}
    artifact_evidence: dict[str, Any] = {}
    exact_chance: float | None = None
    for seed in expected:
        payload = payloads[str(seed)]
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != SEED_SCHEMA
            or payload.get("complete") is not True
            or payload.get("frames") != frames
            or payload.get("seed") != seed
            or payload.get("config_sha256") != config_sha256
            or payload.get("registry_sha256") != registry_sha256
            or payload.get("source_bindings_sha256") != source_bindings_sha256
            or payload.get("checkpoint_requirements_sha256") != checkpoint_requirements_sha256
            or payload.get("data_sha256") != data_sha256
        ):
            raise P5VerificationRefused(f"f{frames} seed {seed} identity or completion drifted")
        arms = payload.get("mechanisms")
        if not isinstance(arms, dict) or set(arms) != set(mechanisms):
            raise P5VerificationRefused(f"f{frames} seed {seed} mechanism coverage drifted")
        trained_by_seed[seed] = {}
        for mechanism in mechanisms:
            arm = arms[mechanism]
            if arm.get("matched") != compute_rows[mechanism]["matched"]:
                raise P5VerificationRefused(
                    f"f{frames} seed {seed} {mechanism} matched-compute identity drifted"
                )
            training = arm.get("training", {})
            if training.get("complete") is not True:
                raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} training is incomplete")
            if training.get("requirements_sha256") != checkpoint_requirements_sha256:
                raise P5VerificationRefused(
                    f"f{frames} seed {seed} {mechanism} checkpoint source binding drifted"
                )
            if (
                training.get("completed_steps") != expected_matches[mechanism]["steps"]
                or training.get("estimated_flops_per_step") != expected_flops[mechanism]
                or training.get("estimated_total_flops") != expected_matches[mechanism]["arm_total_flops"]
            ):
                raise P5VerificationRefused(f"f{frames} seed {seed} {mechanism} completed compute drifted")
            trained_score = _finite_probability(
                arm.get("evaluation", {}).get("heldout_combo_score"),
                f"f{frames} seed {seed} {mechanism} trained score",
            )
            frozen_score = _finite_probability(
                arm.get("frozen", {}).get("evaluation", {}).get("heldout_combo_score"),
                f"f{frames} seed {seed} {mechanism} frozen score",
            )
            trained[mechanism].append(trained_score)
            trained_by_seed[seed][mechanism] = trained_score
            frozen[mechanism].append(frozen_score)
            if mechanism == "exact_global" and seed == expected[0]:
                exact_chance = _finite_probability(
                    arm.get("evaluation", {}).get("chance"), f"f{frames} exact chance"
                )
        if audit_artifacts:
            artifact_evidence[str(seed)] = _audit_seed_artifacts(
                frame_dir / f"seed_{seed}",
                payload,
                frames=frames,
                seed=seed,
                mechanisms=mechanisms,
                config_sha256=config_sha256,
                data_sha256=data_sha256,
                requirements_sha256=checkpoint_requirements_sha256,
                expected_matches=expected_matches,
                expected_flops=expected_flops,
                batch_size=batch_size,
                repo_root=repo_root,
            )

    scores = {mechanism: _paired_ci(values) for mechanism, values in trained.items()}
    frozen_scores = {mechanism: _paired_ci(values) for mechanism, values in frozen.items()}
    contrasts: dict[str, dict[str, Any]] = {}
    exact = trained.get("exact_global")
    if exact is not None:
        for mechanism in mechanisms:
            if mechanism == "exact_global":
                continue
            values = [left - right for left, right in zip(exact, trained[mechanism], strict=True)]
            ci = _paired_ci(values)
            contrasts[f"exact_minus_{mechanism}"] = {
                **ci,
                "classification": classify_ci(float(ci["lo"]), float(ci["hi"]), sesoi),
            }
    if cell.get("scores") != scores or cell.get("frozen_scores") != frozen_scores:
        raise P5VerificationRefused(f"f{frames} score aggregates do not independently recompute")
    if cell.get("paired_contrasts") != contrasts:
        raise P5VerificationRefused(f"f{frames} paired contrasts do not independently recompute")
    if exact_chance is None:
        raise P5VerificationRefused(f"f{frames} exact chance is missing")
    off_ceiling = exact_chance + 0.05 <= trained["exact_global"][0] <= 0.95
    if cell.get("off_ceiling") is not off_ceiling:
        raise P5VerificationRefused(f"f{frames} ceiling classification drifted")
    return {
        "scores": scores,
        "frozen_scores": frozen_scores,
        "contrasts": contrasts,
        "trained": trained,
        "trained_by_seed": trained_by_seed,
        "frozen": frozen,
        "seeds": expected,
        "off_ceiling": off_ceiling,
        "artifact_evidence": artifact_evidence,
    }


def _validate_seed_selection_and_staging(
    receipt: Mapping[str, Any],
    cells: Mapping[int, Mapping[str, Any]],
    recomputed: Mapping[int, Mapping[str, Any]],
    *,
    configured_seeds: Sequence[int],
    terminal: bool,
    futility_margin: float,
    label: str,
) -> None:
    """Reconstruct the runner's exact per-frame stage-out and futility decisions."""

    if (
        isinstance(futility_margin, bool)
        or not isinstance(futility_margin, (int, float))
        or not math.isfinite(float(futility_margin))
        or float(futility_margin) <= 0.0
    ):
        raise P5VerificationRefused(f"{label} futility margin is invalid")
    seeds = list(configured_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise P5VerificationRefused(f"{label} configured seed set is invalid")

    expected_staging: dict[str, dict[str, Any]] = {
        "off_ceiling": {},
        "futility_truncated": {},
    }
    for frames in FRAME_COUNTS:
        frame = recomputed[frames]
        off_ceiling = frame["off_ceiling"]
        futility_evidence: dict[str, Any] | None = None
        if terminal or (len(seeds) > 1 and off_ceiling is not True):
            licensed_seeds = seeds[:1]
        else:
            licensed_seeds = seeds
            if len(seeds) > 3:
                first_three = seeds[:3]
                trained_by_seed = frame["trained_by_seed"]
                if all(seed in trained_by_seed for seed in first_three):
                    paired_mean_deltas = {
                        mechanism: sum(
                            float(trained_by_seed[seed]["exact_global"])
                            - float(trained_by_seed[seed][mechanism])
                            for seed in first_three
                        )
                        / len(first_three)
                        for mechanism in MECHANISMS
                        if mechanism != "exact_global"
                    }
                    if all(value <= 0.0 for value in paired_mean_deltas.values()) and all(
                        abs(value) < float(futility_margin) for value in paired_mean_deltas.values()
                    ):
                        futility_evidence = {
                            "paired_mean_deltas": paired_mean_deltas,
                            "futility_margin": float(futility_margin),
                            "seeds_kept": first_three,
                        }
                        licensed_seeds = first_three

        cell = cells[frames]
        if cell.get("expected_seeds") != licensed_seeds:
            raise P5VerificationRefused(f"{label} f{frames} expected seeds differ from the licensed seed set")
        staged_out = bool(len(seeds) > 1 and off_ceiling is False)
        if cell.get("staged_out") is not staged_out:
            raise P5VerificationRefused(f"{label} f{frames} staged_out decision drifted")
        futility_truncated = futility_evidence is not None
        if cell.get("futility_truncated") is not futility_truncated:
            raise P5VerificationRefused(f"{label} f{frames} futility decision drifted")
        if "futility_evidence" not in cell or cell.get("futility_evidence") != futility_evidence:
            raise P5VerificationRefused(f"{label} f{frames} futility evidence drifted")
        expected_staging["off_ceiling"][f"f{frames}"] = off_ceiling
        expected_staging["futility_truncated"][f"f{frames}"] = futility_evidence

    if receipt.get("staging") != expected_staging:
        raise P5VerificationRefused(f"{label} top-level staging authority drifted")


def _frame_summary(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "complete": cell["complete"],
        "off_ceiling": cell["off_ceiling"],
        "staged_out": cell["staged_out"],
        "futility_truncated": cell["futility_truncated"],
        "seeds_completed": cell["seeds_completed"],
        "scores": cell["scores"],
        "paired_contrasts": cell["paired_contrasts"],
        "all_ok": cell["all_ok"],
    }


def _context_response_curve(
    recomputed: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        mechanism: {
            f"f{frames}": recomputed[frames]["scores"][mechanism]
            for frames in sorted(FRAME_COUNTS)
            if mechanism in recomputed[frames]["scores"]
        }
        for mechanism in MECHANISMS
        if any(mechanism in recomputed[frames]["scores"] for frames in FRAME_COUNTS)
    }


def _patterns(recomputed: Mapping[int, Mapping[str, Any]], sesoi: float) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for frames in PRIMARY_FRAMES:
        if recomputed[frames]["off_ceiling"] is not True:
            continue
        for key, row in recomputed[frames]["contrasts"].items():
            count = int(row["n"])
            if count < 2:
                continue
            lo, hi = float(row["lo"]), float(row["hi"])
            if lo > sesoi:
                direction = "exact-over-factorized"
            elif hi < -sesoi:
                direction = "factorized-over-exact"
            else:
                continue
            mechanism = key.removeprefix("exact_minus_")
            patterns.append(
                {
                    "id": f"f{frames}-exact-minus-{mechanism}",
                    "frames": frames,
                    "mechanism": mechanism,
                    "direction": direction,
                    "primary_ci": dict(row),
                }
            )
    return patterns


def _primary_outcome(
    recomputed: Mapping[int, Mapping[str, Any]], sesoi: float, terminal: bool
) -> tuple[list[dict[str, Any]], str]:
    patterns = _patterns(recomputed, sesoi)
    if terminal:
        return patterns, "null"
    if patterns:
        return patterns, "favorable-programmatic-only"
    has_off_ceiling_contrast = any(
        recomputed[frame]["off_ceiling"] is True and int(row["n"]) >= 2
        for frame in PRIMARY_FRAMES
        for row in recomputed[frame]["contrasts"].values()
    )
    return patterns, "null" if has_off_ceiling_contrast else "mechanics"


def _validate_trainability(
    receipt: Mapping[str, Any], recomputed: Mapping[int, Mapping[str, Any]], terminal: bool
) -> None:
    gate = receipt.get("trainability_gate")
    if not isinstance(gate, dict) or gate.get("applies") is not True or gate.get("evaluated") is not True:
        raise P5VerificationRefused("P5 f64 trainability gate was not evaluated")
    trained = recomputed[64]["trained"]["exact_global"][0]
    frozen = recomputed[64]["frozen"]["exact_global"][0]
    delta = trained - frozen
    margin_value = gate.get("margin")
    delta_value = gate.get("delta")
    if (
        isinstance(margin_value, bool)
        or not isinstance(margin_value, (int, float))
        or isinstance(delta_value, bool)
        or not isinstance(delta_value, (int, float))
    ):
        raise P5VerificationRefused("P5 trainability gate margin or delta is not numeric")
    margin = float(margin_value)
    if not math.isfinite(margin) or margin != TRAINABILITY_MARGIN:
        raise P5VerificationRefused("P5 trainability gate margin drifted")
    if gate.get("trained_heldout") != trained or gate.get("frozen_heldout") != frozen:
        raise P5VerificationRefused("P5 trainability gate score binding drifted")
    if not math.isclose(float(delta_value), delta, rel_tol=0.0, abs_tol=1e-15):
        raise P5VerificationRefused("P5 trainability gate delta drifted")
    failed = delta <= margin
    expected_outcome = "null" if failed else "clears-margin"
    if gate.get("failed") is not failed or gate.get("outcome") != expected_outcome:
        raise P5VerificationRefused("P5 trainability gate decision drifted")
    if receipt.get("trainability_gate_failed") is not failed or terminal is not failed:
        raise P5VerificationRefused("P5 terminal null is inconsistent with the trainability gate")


def _audit_primary_objects(
    *,
    receipt_path: Path,
    run_dir: Path,
    receipt: dict[str, Any],
    raw_receipt: dict[str, Any],
    config: dict[str, Any],
    cells: dict[int, dict[str, Any]],
    repo_root: Path,
    audit_artifacts: bool = True,
) -> PrimaryAudit:
    profile = _sealed_primary_profile(receipt)
    seeds, sesoi, terminal = _validate_primary_header(receipt, config, profile, repo_root)
    if raw_receipt != receipt:
        raise P5VerificationRefused("published P5 pilot does not exactly match its raw run receipt")
    expected_frames = {f"f{frames}" for frames in FRAME_COUNTS}
    if set(receipt.get("frames", {})) != expected_frames or set(cells) != set(FRAME_COUNTS):
        raise P5VerificationRefused("P5 frame coverage drifted")
    registry_sha256 = str(receipt["cell_registry_sha256"])
    recomputed: dict[int, dict[str, Any]] = {}
    for frames in FRAME_COUNTS:
        recomputed[frames] = _recompute_cell(
            cells[frames],
            frame_dir=run_dir / "frames" / f"f{frames}",
            frames=frames,
            mechanisms=MECHANISMS,
            allowed_seeds=seeds,
            config_sha256=str(receipt["config_sha256"]),
            registry_sha256=registry_sha256,
            source_bindings_sha256=str(receipt["source_bindings_sha256"]),
            checkpoint_requirements_sha256=str(receipt["checkpoint_requirements_sha256"]),
            dense_steps=int(config["training"]["dense_steps"]),
            checkpoint_every=int(config["training"]["checkpoint_every"]),
            batch_size=int(config["training"]["batch_size"]),
            sesoi=sesoi,
            repo_root=repo_root,
            audit_artifacts=audit_artifacts,
        )
        if receipt["frames"][f"f{frames}"] != _frame_summary(cells[frames]):
            raise P5VerificationRefused(f"f{frames} top-level summary drifted from its raw cell")
    if receipt.get("primary_contrasts_f64") != recomputed[64]["contrasts"]:
        raise P5VerificationRefused("P5 f64 primary contrast binding drifted")
    if receipt.get("secondary_contrasts_f32") != recomputed[32]["contrasts"]:
        raise P5VerificationRefused("P5 f32 secondary contrast binding drifted")
    if receipt.get("context_response_curve") != _context_response_curve(recomputed):
        raise P5VerificationRefused("P5 context response curve drifted from raw cell scores")
    _validate_trainability(receipt, recomputed, terminal)
    _validate_seed_selection_and_staging(
        receipt,
        cells,
        recomputed,
        configured_seeds=seeds,
        terminal=terminal,
        futility_margin=config["screen"]["futility_margin"],
        label="P5 pilot",
    )
    patterns, outcome = _primary_outcome(recomputed, sesoi, terminal)
    fresh_challenge_required = bool(patterns)
    if receipt.get("fresh_challenge_required") is not fresh_challenge_required:
        raise P5VerificationRefused(
            "P5 fresh_challenge_required hint disagrees with recomputed primary patterns"
        )
    return PrimaryAudit(
        receipt_path=receipt_path,
        run_dir=run_dir,
        receipt=receipt,
        raw_receipt=raw_receipt,
        config=config,
        profile=profile,
        cells=cells,
        recomputed=recomputed,
        patterns=patterns,
        outcome=outcome,
        terminal_null=terminal,
    )


def audit_primary(
    primary_path: Path = DEFAULT_PRIMARY,
    run_dir: Path = DEFAULT_PRIMARY_RUN_DIR,
    config_path: Path = DEFAULT_CONFIG,
    *,
    repo_root: Path = REPO_ROOT,
) -> PrimaryAudit:
    receipt = _read_json(primary_path, "published P5 pilot")
    profile = _sealed_primary_profile(receipt)
    raw_receipt = _read_json(run_dir / "p5_context_receipt.json", "raw P5 pilot receipt")
    config = resolved_profile_config(config_path, profile)
    resolved = _read_json(run_dir / "resolved_config.json", "P5 resolved run config")
    if resolved != config:
        raise P5VerificationRefused("P5 raw resolved config differs from the live pilot config")
    cells = {
        frames: _read_json(
            run_dir / "frames" / f"f{frames}" / "cell_receipt.json", f"f{frames} P5 cell receipt"
        )
        for frames in FRAME_COUNTS
    }
    return _audit_primary_objects(
        receipt_path=primary_path,
        run_dir=run_dir,
        receipt=receipt,
        raw_receipt=raw_receipt,
        config=config,
        cells=cells,
        repo_root=repo_root,
    )


def _validate_binding(binding: Mapping[str, Any], path: Path, label: str) -> None:
    if binding.get("sha256") != file_sha256(path):
        raise P5VerificationRefused(f"{label} file hash binding drifted")
    payload = _read_json(path, label)
    if binding.get("payload_sha256") != payload.get("payload_sha256") or not _sealed(payload):
        raise P5VerificationRefused(f"{label} payload binding drifted")


def audit_challenge(
    primary: PrimaryAudit,
    challenge_path: Path = DEFAULT_CHALLENGE,
    *,
    repo_root: Path = REPO_ROOT,
    audit_artifacts: bool = True,
) -> ChallengeAudit:
    if primary.profile != PILOT_PROFILE:
        raise P5VerificationRefused("fresh training challenge is authorized only for p5pilot")
    receipt = _read_json(challenge_path, "P5 fresh-training challenge")
    if receipt.get("schema") != CHALLENGE_SCHEMA or receipt.get("claim_scope") != CLAIM_SCOPE:
        raise P5VerificationRefused("wrong P5 challenge schema or claim scope")
    if not _sealed(receipt):
        raise P5VerificationRefused("P5 challenge payload digest mismatch")
    expected_keys = {
        "schema",
        "claim_scope",
        "evidence_class",
        "source_bindings",
        "primary_receipt",
        "primary_run_dir",
        "run_dir",
        "patterns",
        "fresh_training_seeds",
        "fresh_seeds_disjoint_from_primary",
        "controls",
        "checkpoint_globs",
        "resource_contract",
        "training_runs",
        "complete",
        "resumable",
        "verification_ready",
        "problems",
        "all_ok",
        "promotion",
        "scientific_promotion",
        "payload_sha256",
    }
    if set(receipt) != expected_keys:
        raise P5VerificationRefused("P5 challenge receipt shape drifted from the governed runner")
    if (
        receipt.get("evidence_class") != CHALLENGE_EVIDENCE_CLASS
        or receipt.get("complete") is not True
        or receipt.get("resumable") is not False
        or receipt.get("verification_ready") is not True
        or receipt.get("problems") != []
        or receipt.get("all_ok") is not True
        or receipt.get("scientific_promotion") is not False
    ):
        raise P5VerificationRefused("P5 fresh-training challenge completion contract drifted")
    if receipt.get("source_bindings") != source_bindings(CHALLENGE_SOURCE_PATHS, repo_root):
        raise P5VerificationRefused("P5 challenge live source binding drifted")
    primary_binding = receipt.get("primary_receipt")
    expected_primary_binding = {
        "path": display_path(primary.receipt_path, repo_root),
        "sha256": file_sha256(primary.receipt_path),
        "payload_sha256": primary.receipt["payload_sha256"],
    }
    if primary_binding != expected_primary_binding:
        raise P5VerificationRefused("P5 challenge primary payload binding drifted")
    if receipt.get("primary_run_dir") != display_path(primary.run_dir, repo_root):
        raise P5VerificationRefused("P5 challenge primary run directory drifted")
    run_dir = resolve_bound_path(receipt.get("run_dir"), repo_root)
    if receipt.get("checkpoint_globs") != _checkpoint_globs(run_dir, repo_root):
        raise P5VerificationRefused("P5 challenge checkpoint globs drifted")
    if receipt.get("resource_contract") != CHALLENGE_RESOURCE_CONTRACT:
        raise P5VerificationRefused("P5 challenge resource contract drifted")
    if receipt.get("controls") != CHALLENGE_CONTROLS:
        raise P5VerificationRefused("P5 challenge control contract drifted")
    seeds = receipt.get("fresh_training_seeds")
    if seeds != list(FRESH_TRAINING_SEEDS):
        raise P5VerificationRefused("P5 challenge fresh training seeds drifted")
    if set(seeds) & set(primary.receipt["seeds"]):
        raise P5VerificationRefused("P5 challenge training seeds overlap the primary pilot")
    if receipt.get("fresh_seeds_disjoint_from_primary") is not True:
        raise P5VerificationRefused("P5 challenge did not attest seed disjointness")
    expected_patterns = [
        {key: value for key, value in pattern.items() if key != "primary_ci"} for pattern in primary.patterns
    ]
    if receipt.get("patterns") != expected_patterns:
        raise P5VerificationRefused("P5 challenge pattern set drifted from the primary pilot")
    if receipt.get("promotion") != CHALLENGE_PROMOTION:
        raise P5VerificationRefused("P5 challenge promotion refusal drifted")

    rows = receipt.get("training_runs")
    if (
        not isinstance(rows, list)
        or [int(row.get("seed", -1)) for row in rows] != list(FRESH_TRAINING_SEEDS)
        or len(rows) != len(FRESH_TRAINING_SEEDS)
    ):
        raise P5VerificationRefused("P5 challenge full-surface seed run coverage drifted")

    by_pattern: dict[str, list[dict[str, Any]]] = {str(row["id"]): [] for row in primary.patterns}
    raw_receipts: dict[int, dict[str, Any]] = {}
    recomputed_runs: dict[int, dict[int, dict[str, Any]]] = {}
    artifact_evidence: dict[str, Any] = {}
    for row in rows:
        seed = int(row["seed"])
        if set(row) != {
            "seed",
            "raw_receipt",
            "cell_receipts",
            "resolved_config",
            "complete",
            "resumable",
            "problems",
            "all_ok",
        }:
            raise P5VerificationRefused(f"P5 fresh seed {seed} row shape drifted")
        if (
            row.get("complete") is not True
            or row.get("resumable") is not False
            or row.get("problems") != []
            or row.get("all_ok") is not True
        ):
            raise P5VerificationRefused(f"P5 fresh full surface seed {seed} is not valid")
        raw_binding = row.get("raw_receipt")
        cell_bindings = row.get("cell_receipts")
        config_binding = row.get("resolved_config")
        if (
            not isinstance(raw_binding, dict)
            or not isinstance(cell_bindings, dict)
            or not isinstance(config_binding, dict)
            or set(raw_binding) != {"path", "sha256", "payload_sha256"}
            or set(config_binding) != {"path", "sha256"}
            or set(cell_bindings) != {f"f{frames}" for frames in FRAME_COUNTS}
        ):
            raise P5VerificationRefused("P5 fresh run artifact bindings are missing")
        subrun = run_dir / f"seed_{seed}"
        raw_path = resolve_bound_path(raw_binding["path"], repo_root)
        resolved_path = resolve_bound_path(config_binding["path"], repo_root)
        if raw_path != (subrun / "p5_context_receipt.json").resolve():
            raise P5VerificationRefused("P5 fresh raw receipt path escaped its seed subrun")
        if resolved_path != (subrun / "resolved_config.json").resolve():
            raise P5VerificationRefused("P5 fresh resolved config path escaped its seed subrun")
        _validate_binding(raw_binding, raw_path, "P5 fresh raw receipt")
        if config_binding.get("sha256") != file_sha256(resolved_path):
            raise P5VerificationRefused("P5 fresh resolved config file hash drifted")
        raw = _read_json(raw_path, "P5 fresh raw receipt")
        raw_receipts[seed] = raw
        config = _read_json(resolved_path, "P5 fresh resolved config")
        expected_config = challenge_seed_config(primary.config, seed)
        if config != expected_config or raw.get("config_sha256") != canonical_sha256(expected_config):
            raise P5VerificationRefused("P5 fresh run config identity drifted")
        registry_sha256 = canonical_sha256(expected_config["cells"])
        if raw.get("cell_registry_sha256") != registry_sha256:
            raise P5VerificationRefused("P5 fresh run cell registry identity drifted")
        expected_serial = [f"f{cell['frames']}_{cell['mechanism']}" for cell in expected_config["cells"]]
        if (
            raw.get("schema") != PILOT_SCHEMA
            or raw.get("claim_scope") != CLAIM_SCOPE
            or raw.get("profile") != f"p5fresh-seed-{seed}"
            or raw.get("seeds") != [seed]
            or raw.get("serial_order") != expected_serial
            or raw.get("dense_reference_steps") != int(expected_config["training"]["dense_steps"])
            or raw.get("sesoi") != float(primary.receipt["sesoi"])
            or raw.get("fresh_challenge_required") is not False
        ):
            raise P5VerificationRefused("P5 fresh raw profile, seed, or design identity drifted")
        expected_sources = source_bindings(SOURCE_PATHS, repo_root)
        expected_source_sha = canonical_sha256(expected_sources)
        expected_checkpoint_sha = canonical_sha256(
            {
                "registry_sha256": registry_sha256,
                "source_bindings_sha256": expected_source_sha,
            }
        )
        if raw.get("source_bindings") != expected_sources:
            raise P5VerificationRefused("P5 fresh raw receipt source binding drifted")
        if (
            raw.get("source_bindings_sha256") != expected_source_sha
            or raw.get("checkpoint_requirements_sha256") != expected_checkpoint_sha
        ):
            raise P5VerificationRefused("P5 fresh raw checkpoint source identity drifted")
        if raw.get("complete") is not True or raw.get("all_ok") is not True:
            raise P5VerificationRefused("P5 fresh raw full surface is incomplete or all_ok false")
        if raw.get("problems") != [] or raw.get("resumable") is not False:
            raise P5VerificationRefused("P5 fresh raw full surface retains problems or resumability")
        _validate_promotion(raw, "P5 fresh raw full surface")
        if any(
            raw.get(flag) is not False
            for flag in (
                "stopped_for_wall_budget",
                "stopped_for_disk_floor",
                "stopped_for_required_arm_refusal",
            )
        ):
            raise P5VerificationRefused("P5 fresh raw subrun stopped operationally")
        if raw.get("required_arm_failure") is not None:
            raise P5VerificationRefused("P5 fresh raw subrun retains a required-arm failure")
        recomputed_by_frame: dict[int, dict[str, Any]] = {}
        cells_by_frame: dict[int, dict[str, Any]] = {}
        for frames in FRAME_COUNTS:
            binding = cell_bindings[f"f{frames}"]
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise P5VerificationRefused("P5 fresh cell receipt binding is malformed")
            cell_path = resolve_bound_path(binding["path"], repo_root)
            expected_cell_path = subrun / "frames" / f"f{frames}" / "cell_receipt.json"
            if cell_path != expected_cell_path.resolve():
                raise P5VerificationRefused("P5 fresh cell receipt escaped its seed subrun")
            if binding.get("sha256") != file_sha256(cell_path):
                raise P5VerificationRefused("P5 fresh cell receipt file hash drifted")
            cell = _read_json(cell_path, "P5 fresh cell receipt")
            cells_by_frame[frames] = cell
            recomputed_by_frame[frames] = _recompute_cell(
                cell,
                frame_dir=subrun / "frames" / f"f{frames}",
                frames=frames,
                mechanisms=MECHANISMS,
                allowed_seeds=[seed],
                config_sha256=str(raw["config_sha256"]),
                registry_sha256=registry_sha256,
                source_bindings_sha256=str(raw.get("source_bindings_sha256")),
                checkpoint_requirements_sha256=str(raw.get("checkpoint_requirements_sha256")),
                dense_steps=int(config["training"]["dense_steps"]),
                checkpoint_every=int(config["training"]["checkpoint_every"]),
                batch_size=int(config["training"]["batch_size"]),
                sesoi=float(primary.receipt["sesoi"]),
                repo_root=repo_root,
                audit_artifacts=audit_artifacts,
            )
            if raw.get("frames", {}).get(f"f{frames}") != _frame_summary(cell):
                raise P5VerificationRefused(f"P5 fresh f{frames} top-level summary drifted")
        recomputed_runs[seed] = recomputed_by_frame
        if raw.get("context_response_curve") != _context_response_curve(recomputed_by_frame):
            raise P5VerificationRefused(f"P5 fresh seed {seed} context response curve drifted")
        artifact_evidence[str(seed)] = {
            f"f{frames}": recomputed_by_frame[frames]["artifact_evidence"] for frames in FRAME_COUNTS
        }
        terminal = raw.get("terminal_scientific_stop") is True
        _validate_trainability(raw, recomputed_by_frame, terminal)
        _validate_seed_selection_and_staging(
            raw,
            cells_by_frame,
            recomputed_by_frame,
            configured_seeds=[seed],
            terminal=terminal,
            futility_margin=config["screen"]["futility_margin"],
            label=f"P5 fresh seed {seed}",
        )
        expected_status = "terminal-scientific-null" if terminal else "complete"
        expected_reason = "f64-trainability-gate-null" if terminal else None
        if (
            raw.get("execution_status") != expected_status
            or raw.get("terminal_stop_reason") != expected_reason
        ):
            raise P5VerificationRefused("P5 fresh raw execution status drifted")
        trainability_ok = not terminal
        trainability_delta = (
            recomputed_by_frame[64]["trained"]["exact_global"][0]
            - recomputed_by_frame[64]["frozen"]["exact_global"][0]
        )
        for pattern in primary.patterns:
            recomputed = recomputed_by_frame[int(pattern["frames"])]
            delta = (
                recomputed["trained"]["exact_global"][0] - recomputed["trained"][str(pattern["mechanism"])][0]
            )
            by_pattern[str(pattern["id"])].append(
                {
                    "seed": seed,
                    "delta": delta,
                    "trainability_delta": trainability_delta,
                    "trainability_ok": trainability_ok,
                    "off_ceiling": recomputed["off_ceiling"],
                }
            )

    per_pattern: list[dict[str, Any]] = []
    sesoi = float(primary.receipt["sesoi"])
    for pattern in primary.patterns:
        units = sorted(by_pattern[str(pattern["id"])], key=lambda row: row["seed"])
        ci = _paired_ci([float(row["delta"]) for row in units])
        if pattern["direction"] == "exact-over-factorized":
            same_direction = float(ci["lo"]) > sesoi
        else:
            same_direction = float(ci["hi"]) < -sesoi
        verified = (
            same_direction
            and all(row["trainability_ok"] for row in units)
            and all(row["off_ceiling"] is True for row in units)
        )
        per_pattern.append(
            {
                "id": pattern["id"],
                "direction": pattern["direction"],
                "fresh_training_units": units,
                "fresh_ci": ci,
                "tie_is_null": True,
                "strict_direction_reproduced": same_direction,
                "programmatic_pattern_verified": verified,
                "scientific_promotion_allowed": False,
                "outcome": "favorable-programmatic-only" if verified else "null",
            }
        )
    return ChallengeAudit(
        receipt_path=challenge_path,
        receipt=receipt,
        per_pattern=per_pattern,
        artifact_evidence=artifact_evidence,
        raw_receipts=raw_receipts,
        recomputed=recomputed_runs,
    )


def _mutation_checks(
    primary: PrimaryAudit,
    challenge: ChallengeAudit | None,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def primary_case(name: str, mutate: Any) -> None:
        receipt = copy.deepcopy(primary.receipt)
        raw = copy.deepcopy(primary.raw_receipt)
        cells = copy.deepcopy(primary.cells)
        mutate(receipt, raw, cells)
        _seal(receipt)
        _seal(raw)
        try:
            _audit_primary_objects(
                receipt_path=primary.receipt_path,
                run_dir=primary.run_dir,
                receipt=receipt,
                raw_receipt=raw,
                config=copy.deepcopy(primary.config),
                cells=cells,
                repo_root=repo_root,
                audit_artifacts=False,
            )
        except (P5VerificationRefused, KeyError, TypeError, ValueError) as exc:
            rows.append({"id": name, "rejected": True, "observed_error": str(exc)})
        else:
            rows.append({"id": name, "rejected": False, "observed_error": None})

    primary_case(
        "incomplete-pilot",
        lambda receipt, raw, cells: (
            receipt.__setitem__("complete", False),
            raw.__setitem__("complete", False),
        ),
    )
    primary_case(
        "all-ok-false",
        lambda receipt, raw, cells: (receipt.__setitem__("all_ok", False), raw.__setitem__("all_ok", False)),
    )
    primary_case(
        "source-hash-drift",
        lambda receipt, raw, cells: (
            receipt["source_bindings"][0].__setitem__("file_sha256", "0" * 64),
            raw["source_bindings"][0].__setitem__("file_sha256", "0" * 64),
        ),
    )
    primary_case(
        "config-binding-drift",
        lambda receipt, raw, cells: (
            receipt.__setitem__("config_sha256", "0" * 64),
            raw.__setitem__("config_sha256", "0" * 64),
        ),
    )
    opposite_profile = SMOKE_PROFILE if primary.profile == PILOT_PROFILE else PILOT_PROFILE
    primary_case(
        "sealed-profile-config-mismatch",
        lambda receipt, raw, cells: (
            receipt.__setitem__("profile", opposite_profile),
            raw.__setitem__("profile", opposite_profile),
        ),
    )
    primary_case(
        "fresh-challenge-hint-flip",
        lambda receipt, raw, cells: (
            receipt.__setitem__("fresh_challenge_required", not bool(receipt["fresh_challenge_required"])),
            raw.__setitem__("fresh_challenge_required", not bool(raw["fresh_challenge_required"])),
        ),
    )

    def mutate_seed_selection(
        receipt: dict[str, Any], raw: dict[str, Any], cells: dict[int, dict[str, Any]]
    ) -> None:
        frames = 64
        cell = cells[frames]
        observed = list(cell["expected_seeds"])
        if len(observed) <= 1:
            cell["expected_seeds"] = []
            cell["seeds_completed"] = 0
            cell["seed_results"] = {}
            return
        selected = observed[:-1]
        cell["expected_seeds"] = selected
        cell["seeds_completed"] = len(selected)
        cell["seed_results"] = {str(seed): cell["seed_results"][str(seed)] for seed in selected}
        trained = {
            mechanism: [
                float(
                    cell["seed_results"][str(seed)]["mechanisms"][mechanism]["evaluation"][
                        "heldout_combo_score"
                    ]
                )
                for seed in selected
            ]
            for mechanism in MECHANISMS
        }
        frozen = {
            mechanism: [
                float(
                    cell["seed_results"][str(seed)]["mechanisms"][mechanism]["frozen"]["evaluation"][
                        "heldout_combo_score"
                    ]
                )
                for seed in selected
            ]
            for mechanism in MECHANISMS
        }
        cell["scores"] = {mechanism: _paired_ci(values) for mechanism, values in trained.items()}
        cell["frozen_scores"] = {mechanism: _paired_ci(values) for mechanism, values in frozen.items()}
        cell["paired_contrasts"] = {}
        for mechanism in MECHANISMS:
            if mechanism == "exact_global":
                continue
            values = [
                left - right for left, right in zip(trained["exact_global"], trained[mechanism], strict=True)
            ]
            ci = _paired_ci(values)
            cell["paired_contrasts"][f"exact_minus_{mechanism}"] = {
                **ci,
                "classification": classify_ci(float(ci["lo"]), float(ci["hi"]), float(receipt["sesoi"])),
            }
        for compute in cell["compute"]["per_mechanism"].values():
            compute["estimated_total_flops_completed_seeds"] = compute["matched"]["arm_total_flops"] * len(
                selected
            )
        cell["staged_out"] = False
        cell["futility_truncated"] = False
        cell["futility_evidence"] = None
        for target in (receipt, raw):
            target["frames"]["f64"] = _frame_summary(cell)
            target["primary_contrasts_f64"] = copy.deepcopy(cell["paired_contrasts"])
            target["staging"]["futility_truncated"]["f64"] = None

    primary_case("seed-selection-drift", mutate_seed_selection)
    primary_case(
        "confirmatory-promotion",
        lambda receipt, raw, cells: (
            receipt["promotion"].__setitem__("confirmatory_promotable", True),
            raw["promotion"].__setitem__("confirmatory_promotable", True),
        ),
    )

    def mutate_metric(receipt: dict[str, Any], raw: dict[str, Any], cells: dict[int, dict[str, Any]]) -> None:
        seed = str(cells[64]["expected_seeds"][0])
        value = cells[64]["seed_results"][seed]["mechanisms"]["exact_global"]["evaluation"]
        value["heldout_combo_score"] = min(1.0, float(value["heldout_combo_score"]) + 0.01)

    primary_case("raw-score-mutation", mutate_metric)

    def mutate_seed_source(
        receipt: dict[str, Any], raw: dict[str, Any], cells: dict[int, dict[str, Any]]
    ) -> None:
        seed = str(cells[64]["expected_seeds"][0])
        cells[64]["seed_results"][seed]["source_bindings_sha256"] = "0" * 64

    primary_case("cached-seed-source-drift", mutate_seed_source)

    def mutate_checkpoint_source(
        receipt: dict[str, Any], raw: dict[str, Any], cells: dict[int, dict[str, Any]]
    ) -> None:
        seed = str(cells[64]["expected_seeds"][0])
        cells[64]["seed_results"][seed]["mechanisms"]["exact_global"]["training"]["requirements_sha256"] = (
            "0" * 64
        )

    primary_case("checkpoint-source-drift", mutate_checkpoint_source)

    def mutate_compute(
        receipt: dict[str, Any], raw: dict[str, Any], cells: dict[int, dict[str, Any]]
    ) -> None:
        cells[64]["compute"]["per_mechanism"]["window_local"]["estimated_flops_per_step"] += 1

    primary_case("matched-compute-drift", mutate_compute)

    def promote_tie(receipt: dict[str, Any], raw: dict[str, Any], cells: dict[int, dict[str, Any]]) -> None:
        key = "exact_minus_window_local"
        target = cells[64]["paired_contrasts"][key]
        target.update(
            {
                "lo": float(receipt["sesoi"]),
                "hi": float(receipt["sesoi"]),
                "classification": "meaningful_positive",
            }
        )
        receipt["frames"]["f64"]["paired_contrasts"] = copy.deepcopy(cells[64]["paired_contrasts"])
        raw["frames"]["f64"]["paired_contrasts"] = copy.deepcopy(cells[64]["paired_contrasts"])
        receipt["primary_contrasts_f64"] = copy.deepcopy(cells[64]["paired_contrasts"])
        raw["primary_contrasts_f64"] = copy.deepcopy(cells[64]["paired_contrasts"])

    primary_case("threshold-tie-promotion", promote_tie)

    ceilinged = copy.deepcopy(primary.recomputed)
    for frames in PRIMARY_FRAMES:
        ceilinged[frames]["off_ceiling"] = False
    ceiling_patterns, ceiling_outcome = _primary_outcome(
        ceilinged,
        float(primary.receipt["sesoi"]),
        False,
    )
    if not ceiling_patterns and ceiling_outcome == "mechanics":
        rows.append(
            {
                "id": "ceilinged-contrast-promotion",
                "rejected": True,
                "observed_error": "nonterminal ceilinged contrasts classify as mechanics",
            }
        )
    else:
        rows.append(
            {
                "id": "ceilinged-contrast-promotion",
                "rejected": False,
                "observed_error": None,
            }
        )

    frames = FRAME_COUNTS[0]
    cell = primary.cells[frames]
    seed = int(cell["expected_seeds"][0])
    embedded_seed = cell["seed_results"][str(seed)]
    source_seed_dir = primary.run_dir / "frames" / f"f{frames}" / f"seed_{seed}"
    expected_flops = {
        mechanism: _expected_flops_per_step(
            frames,
            mechanism,
            int(primary.config["training"]["batch_size"]),
        )
        for mechanism in MECHANISMS
    }
    dense_flops = expected_flops["exact_global"]
    expected_matches = {
        mechanism: _expected_match(
            int(primary.config["training"]["dense_steps"]),
            dense_flops,
            expected_flops[mechanism],
            int(primary.config["training"]["checkpoint_every"]),
            exact=mechanism == "exact_global",
        )
        for mechanism in MECHANISMS
    }

    def artifact_case(name: str, setup: Any) -> None:
        with tempfile.TemporaryDirectory(dir=repo_root) as temporary:
            seed_dir = Path(temporary) / f"seed_{seed}"
            seed_dir.mkdir(parents=True)
            setup(seed_dir)
            try:
                _audit_seed_artifacts(
                    seed_dir,
                    embedded_seed,
                    frames=frames,
                    seed=seed,
                    mechanisms=MECHANISMS,
                    config_sha256=str(primary.receipt["config_sha256"]),
                    data_sha256=str(cell["corpus"]["content_sha256"]),
                    requirements_sha256=str(primary.receipt["checkpoint_requirements_sha256"]),
                    expected_matches=expected_matches,
                    expected_flops=expected_flops,
                    batch_size=int(primary.config["training"]["batch_size"]),
                    repo_root=repo_root,
                )
            except (P5VerificationRefused, KeyError, TypeError, ValueError, OSError) as exc:
                rows.append({"id": name, "rejected": True, "observed_error": str(exc)})
            else:
                rows.append({"id": name, "rejected": False, "observed_error": None})

    artifact_case("missing-seed-result-artifact", lambda seed_dir: None)

    def seed_only(seed_dir: Path) -> None:
        atomic_json(
            seed_dir / "seed_result.json",
            _read_json(source_seed_dir / "seed_result.json", "source seed result"),
        )

    artifact_case("missing-arm-receipt-artifact", seed_only)

    def arm_without_checkpoint(seed_dir: Path) -> None:
        seed_only(seed_dir)
        mechanism = MECHANISMS[0]
        arm_dir = seed_dir / mechanism
        arm = _read_json(source_seed_dir / mechanism / "arm_receipt.json", "source arm receipt")
        arm["checkpoint"]["path"] = str(arm_dir / "checkpoint.pt")
        atomic_json(arm_dir / "arm_receipt.json", arm)

    artifact_case("missing-checkpoint-artifact", arm_without_checkpoint)

    def checkpoint_hash_drift(seed_dir: Path) -> None:
        arm_without_checkpoint(seed_dir)
        mechanism = MECHANISMS[0]
        arm_dir = seed_dir / mechanism
        checkpoint = arm_dir / "checkpoint.pt"
        checkpoint.symlink_to((source_seed_dir / mechanism / "checkpoint.pt").resolve())
        arm = _read_json(arm_dir / "arm_receipt.json", "copied arm receipt")
        arm["checkpoint"]["sha256"] = "0" * 64
        atomic_json(arm_dir / "arm_receipt.json", arm)

    artifact_case("checkpoint-file-hash-drift", checkpoint_hash_drift)

    if challenge is not None:

        def challenge_case(name: str, mutate: Any) -> None:
            payload = copy.deepcopy(challenge.receipt)
            mutate(payload)
            _seal(payload)
            with tempfile.TemporaryDirectory(dir=repo_root) as temporary:
                path = Path(temporary) / "challenge-mutation.json"
                atomic_json(path, payload)
                try:
                    audit_challenge(
                        primary,
                        path,
                        repo_root=repo_root,
                        audit_artifacts=False,
                    )
                except (P5VerificationRefused, KeyError, TypeError, ValueError) as exc:
                    rows.append({"id": name, "rejected": True, "observed_error": str(exc)})
                else:
                    rows.append({"id": name, "rejected": False, "observed_error": None})

        challenge_case(
            "fresh-seed-overlap",
            lambda payload: payload["fresh_training_seeds"].__setitem__(0, int(primary.receipt["seeds"][0])),
        )
        challenge_case("fresh-run-drop", lambda payload: payload["training_runs"].pop())
        challenge_case(
            "fresh-confirmatory-promotion",
            lambda payload: payload["promotion"].__setitem__("confirmatory_promotable", True),
        )
        challenge_case(
            "challenge-shape-omission",
            lambda payload: payload.pop("resource_contract"),
        )
        fresh_seed = FRESH_TRAINING_SEEDS[0]
        fabricated_gate = copy.deepcopy(challenge.raw_receipts[fresh_seed])
        fabricated_gate["trainability_gate"]["delta"] = (
            float(fabricated_gate["trainability_gate"]["delta"]) + 0.001
        )
        try:
            _validate_trainability(
                fabricated_gate,
                challenge.recomputed[fresh_seed],
                fabricated_gate.get("terminal_scientific_stop") is True,
            )
        except (P5VerificationRefused, KeyError, TypeError, ValueError) as exc:
            rows.append(
                {
                    "id": "fresh-trainability-gate-fabrication",
                    "rejected": True,
                    "observed_error": str(exc),
                }
            )
        else:
            rows.append(
                {
                    "id": "fresh-trainability-gate-fabrication",
                    "rejected": False,
                    "observed_error": None,
                }
            )
    return rows


def build_verification(
    primary_path: Path = DEFAULT_PRIMARY,
    primary_run_dir: Path = DEFAULT_PRIMARY_RUN_DIR,
    config_path: Path = DEFAULT_CONFIG,
    challenge_path: Path = DEFAULT_CHALLENGE,
    *,
    repo_root: Path = REPO_ROOT,
    run_mutations: bool = True,
) -> dict[str, Any]:
    primary = audit_primary(primary_path, primary_run_dir, config_path, repo_root=repo_root)
    challenge: ChallengeAudit | None = None
    if primary.patterns:
        challenge = audit_challenge(primary, challenge_path, repo_root=repo_root)
    if primary.outcome == "favorable-programmatic-only":
        assert challenge is not None
        verified_patterns = [
            row for row in challenge.per_pattern if row["programmatic_pattern_verified"] is True
        ]
        outcome = "favorable-programmatic-only" if verified_patterns else "null"
    else:
        verified_patterns = []
        outcome = primary.outcome
    mutations = _mutation_checks(primary, challenge, repo_root=repo_root) if run_mutations else []
    all_mutations_rejected = bool(mutations) and all(row["rejected"] is True for row in mutations)
    problems = []
    if not run_mutations:
        problems.append("P5 adversarial mutation suite was not run")
    elif not all_mutations_rejected:
        problems.append("one or more P5 semantic mutations escaped rejection")
    all_controls_passed = primary.outcome in {"null", "favorable-programmatic-only"}
    prerequisite_ready = (
        not problems and all_controls_passed and outcome in {"null", "favorable-programmatic-only"}
    )
    receipt: dict[str, Any] = {
        "schema": VERIFY_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": EVIDENCE_CLASS,
        "verification_complete": not problems,
        "primary_profile": primary.profile,
        "source_bindings": source_bindings(VERIFIER_SOURCE_PATHS, repo_root),
        "primary_receipt": {
            "path": display_path(primary_path, repo_root),
            "sha256": file_sha256(primary_path),
            "payload_sha256": primary.receipt["payload_sha256"],
        },
        "primary_run_receipt": {
            "path": display_path(primary_run_dir / "p5_context_receipt.json", repo_root),
            "sha256": file_sha256(primary_run_dir / "p5_context_receipt.json"),
            "exactly_matches_published": True,
        },
        "config": {
            "path": display_path(config_path, repo_root),
            "sha256": file_sha256(config_path),
            "resolved_sha256": canonical_sha256(primary.config),
        },
        "independence": {
            "imports_p5_training_or_evaluator": False,
            "raw_seed_score_recompute": True,
            "checkpoint_files_opened_with_weights_only": True,
            "checkpoint_model_and_target_state_hashes_recomputed": True,
            "heldout_metrics_reexecuted_from_checkpoint": False,
            "fresh_training_required_for_each_primary_pattern": True,
            "fresh_training_seeds": list(FRESH_TRAINING_SEEDS),
            "fresh_seeds_disjoint_from_primary": not (
                set(FRESH_TRAINING_SEEDS) & set(primary.receipt["seeds"])
            ),
        },
        "primary_outcome": primary.outcome,
        "fresh_challenge_required": bool(primary.patterns),
        "terminal_null": primary.terminal_null,
        "primary_patterns": primary.patterns,
        "artifact_evidence": {
            "primary": {
                f"f{frames}": primary.recomputed[frames]["artifact_evidence"] for frames in FRAME_COUNTS
            },
            "fresh_challenge": challenge.artifact_evidence if challenge is not None else None,
        },
        "cell_receipt_evidence": {
            "primary": {
                f"f{frames}": {
                    "path": display_path(
                        primary.run_dir / "frames" / f"f{frames}" / "cell_receipt.json",
                        repo_root,
                    ),
                    "sha256": file_sha256(primary.run_dir / "frames" / f"f{frames}" / "cell_receipt.json"),
                }
                for frames in FRAME_COUNTS
            },
            "fresh_challenge": (
                {
                    str(row["seed"]): copy.deepcopy(row["cell_receipts"])
                    for row in challenge.receipt["training_runs"]
                }
                if challenge is not None
                else None
            ),
        },
        "metric_recomputation_limit": (
            "checkpoint model and target states, identities, completed steps, and compute are "
            "independently hashed and joined; heldout scores are recomputed from durable per-seed "
            "receipts but are not re-evaluated from model checkpoints"
        ),
        "fresh_challenge": (
            {
                "path": display_path(challenge.receipt_path, repo_root),
                "sha256": file_sha256(challenge.receipt_path),
                "payload_sha256": challenge.receipt["payload_sha256"],
                "per_pattern": challenge.per_pattern,
            }
            if challenge is not None
            else None
        ),
        "verified_patterns": verified_patterns,
        "outcome": outcome,
        "classification": outcome,
        "controls": {
            "same_initialization_frozen_control": True,
            "matched_parameter_and_flop_contract": True,
            "difficulty_calibration_checked": True,
            "primary_off_ceiling": {
                f"f{frames}": primary.recomputed[frames]["off_ceiling"] for frames in PRIMARY_FRAMES
            },
            "nonterminal_outcome_has_off_ceiling_multiunit_support": (
                primary.terminal_null or primary.outcome != "mechanics"
            ),
            "seed_arm_checkpoint_artifacts_exactly_joined": True,
            "raw_per_seed_contrasts_independently_recomputed": True,
            "fresh_disjoint_training_for_every_primary_pattern": (
                challenge is not None if primary.patterns else True
            ),
            "threshold_tie_is_null": True,
            "confirmatory_promotion_refused": True,
        },
        "all_controls_passed": all_controls_passed,
        "strongest_control": (
            "three disjoint fresh training seeds with matched compute and same-initialization frozen arms"
            if primary.patterns
            else "f64 same-initialization frozen trainability gate"
        ),
        "outcome_contract": {
            "allowed": ["mechanics", "null", "favorable-programmatic-only"],
            "tie_is_null": True,
            "programmatic_only": True,
            "confirmatory_promotable": False,
            "scientific_capability_claim": False,
        },
        "mutation_tests": mutations,
        "all_mutations_rejected": all_mutations_rejected,
        "problems": problems,
        "all_ok": not problems,
        "prerequisite_ready": prerequisite_ready,
        "scientific_promotion": False,
        "promotion": {
            "confirmatory_promotable": False,
            "refused_by_construction": True,
            "scientific_capability_claim": False,
        },
    }
    _seal(receipt)
    return receipt


def write_verification(
    output: Path = DEFAULT_VERIFICATION,
    primary_path: Path = DEFAULT_PRIMARY,
    primary_run_dir: Path = DEFAULT_PRIMARY_RUN_DIR,
    config_path: Path = DEFAULT_CONFIG,
    challenge_path: Path = DEFAULT_CHALLENGE,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    receipt = build_verification(
        primary_path,
        primary_run_dir,
        config_path,
        challenge_path,
        repo_root=repo_root,
        run_mutations=True,
    )
    if receipt["all_ok"] is not True:
        raise P5VerificationRefused("P5 verifier mutation suite did not fail closed")
    atomic_json(output, receipt)
    return receipt


__all__ = [
    "CHALLENGE_SCHEMA",
    "CHALLENGE_SOURCE_PATHS",
    "CLAIM_SCOPE",
    "DEFAULT_CHALLENGE",
    "DEFAULT_CONFIG",
    "DEFAULT_PRIMARY",
    "DEFAULT_PRIMARY_RUN_DIR",
    "DEFAULT_VERIFICATION",
    "FRESH_TRAINING_SEEDS",
    "PILOT_PROFILE",
    "P5VerificationRefused",
    "PrimaryAudit",
    "VERIFY_SCHEMA",
    "SMOKE_PROFILE",
    "audit_primary",
    "atomic_json",
    "build_verification",
    "canonical_sha256",
    "challenge_seed_config",
    "classify_ci",
    "display_path",
    "file_sha256",
    "source_bindings",
    "write_verification",
]
