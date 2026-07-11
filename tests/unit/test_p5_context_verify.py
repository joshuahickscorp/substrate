from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from mop.studies.p5_context_challenge import challenge_exit_code, run_fresh_challenge
from mop.studies.p5_context_verify import (
    ARM_SCHEMA,
    CELL_SCHEMA,
    CHALLENGE_CONTROLS,
    CHALLENGE_EVIDENCE_CLASS,
    CHALLENGE_PROMOTION,
    CHALLENGE_RESOURCE_CONTRACT,
    CHALLENGE_SCHEMA,
    CHALLENGE_SOURCE_PATHS,
    CHECKPOINT_SCHEMA,
    CLAIM_SCOPE,
    EXPECTED_TRANSFORMER_PARAMETERS,
    FRAME_COUNTS,
    FRESH_TRAINING_SEEDS,
    MECHANISMS,
    PILOT_PROFILE,
    PILOT_PROMOTION,
    PILOT_SCHEMA,
    SEED_SCHEMA,
    SMOKE_PROFILE,
    SOURCE_PATHS,
    VERIFIER_SOURCE_PATHS,
    VERIFY_SCHEMA,
    P5VerificationRefused,
    _expected_flops_per_step,
    _expected_match,
    atomic_json,
    audit_primary,
    build_verification,
    canonical_sha256,
    challenge_seed_config,
    classify_ci,
    display_path,
    file_sha256,
    resolved_profile_config,
    source_bindings,
    state_sha256,
)

P5_CORE_RUNTIME_SOURCES = (
    "src/mop/substrate/custom_workbench.py",
    "src/mop/substrate/p4_screen.py",
)


def _ci(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "lo": mean, "hi": mean, "half": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "lo": mean - half, "hi": mean + half, "half": half}


def _seal(payload: dict[str, Any]) -> None:
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = canonical_sha256(payload)


def _config_rows() -> list[dict[str, Any]]:
    return [{"frames": frames, "mechanism": mechanism} for frames in FRAME_COUNTS for mechanism in MECHANISMS]


def _write_repo_sources(root: Path, sesoi: float = 0.1) -> Path:
    config_path = root / SOURCE_PATHS[0]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "id": "mop_p5_context_capability",
        "name": "context_capability_pilot",
        "module": "p5_context",
        "metric": ["heldout_combo_per_context", "context_response_curve"],
        "null_hypothesis": "factorized mechanisms match exact within the SESOI",
        "tier": "cpu-now",
        "profiles": {
            "p5smoke": {
                "seeds": [0],
                "dense_steps": 12,
                "checkpoint_every": 6,
                "wall_budget_seconds": 100.0,
            },
            "p5pilot": {
                "seeds": [0, 1, 2, 3, 4],
                "dense_steps": 200,
                "checkpoint_every": 25,
                "wall_budget_seconds": 100.0,
            },
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.0005,
            "weight_decay": 0.02,
            "mask_ratio": 0.5,
            "ema_decay": 0.99,
            "variance_weight": 0.1,
        },
        "screen": {"sesoi": sesoi, "futility_margin": 0.1, "min_free_disk_gb": 40.0},
        "cells": _config_rows(),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for relative in (set(CHALLENGE_SOURCE_PATHS) | set(VERIFIER_SOURCE_PATHS)) - {SOURCE_PATHS[0]}:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source for {relative}\n", encoding="utf-8")
    return config_path


def _fixture_states(
    frames: int, seed: int, mechanism: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    index = MECHANISMS.index(mechanism)
    model = {"weight": torch.tensor([float(frames), float(seed), float(index)], dtype=torch.float64)}
    target = {"weight": torch.tensor([float(frames), float(seed), float(index) + 0.5], dtype=torch.float64)}
    return model, target


def _cell(
    frames: int,
    seeds: list[int],
    scores: dict[str, float],
    frozen_scores: dict[str, float],
    config_sha256: str,
    registry_sha256: str,
    source_bindings_sha256: str,
    checkpoint_requirements_sha256: str,
    dense_steps: int,
    checkpoint_every: int,
    batch_size: int,
    sesoi: float,
    mechanisms: tuple[str, ...] = MECHANISMS,
) -> dict[str, Any]:
    data_sha = f"{frames:064x}"[-64:]
    flops = {mechanism: _expected_flops_per_step(frames, mechanism, batch_size) for mechanism in mechanisms}
    dense_flops = _expected_flops_per_step(frames, "exact_global", batch_size)
    matches = {
        mechanism: _expected_match(
            dense_steps,
            dense_flops,
            flops[mechanism],
            checkpoint_every,
            exact=mechanism == "exact_global",
        )
        for mechanism in mechanisms
    }
    seed_results: dict[str, Any] = {}
    for seed in seeds:
        arms = {}
        initial_states = {}
        for mechanism in mechanisms:
            model_state, _ = _fixture_states(frames, seed, mechanism)
            initial_sha = canonical_sha256(
                {"frames": frames, "seed": seed, "mechanism": mechanism, "state": "initial"}
            )
            initial_states[mechanism] = initial_sha
            arms[mechanism] = {
                "initial_state_sha256": initial_sha,
                "matched": matches[mechanism],
                "frozen": {
                    "control": "exact same-architecture, same-initialization frozen encoder",
                    "evaluation": {
                        "heldout_combo_score": frozen_scores[mechanism],
                        "chance": 0.25,
                    },
                },
                "training": {
                    "complete": True,
                    "requirements_sha256": checkpoint_requirements_sha256,
                    "completed_steps": matches[mechanism]["steps"],
                    "estimated_flops_per_step": flops[mechanism],
                    "estimated_total_flops": matches[mechanism]["arm_total_flops"],
                    "final_state_sha256": state_sha256(model_state, "fixture checkpoint model"),
                },
                "evaluation": {"heldout_combo_score": scores[mechanism], "chance": 0.25},
            }
        seed_results[str(seed)] = {
            "schema": SEED_SCHEMA,
            "frames": frames,
            "seed": seed,
            "config_sha256": config_sha256,
            "data_sha256": data_sha,
            "registry_sha256": registry_sha256,
            "source_bindings_sha256": source_bindings_sha256,
            "checkpoint_requirements_sha256": checkpoint_requirements_sha256,
            "initial_state_sha256": initial_states,
            "mechanisms": arms,
            "complete": True,
        }
    aggregates = {mechanism: _ci([scores[mechanism] for _ in seeds]) for mechanism in mechanisms}
    frozen_aggregates = {
        mechanism: _ci([frozen_scores[mechanism] for _ in seeds]) for mechanism in mechanisms
    }
    contrasts = {}
    for mechanism in mechanisms:
        if mechanism == "exact_global":
            continue
        values = [scores["exact_global"] - scores[mechanism] for _ in seeds]
        row = _ci(values)
        contrasts[f"exact_minus_{mechanism}"] = {
            **row,
            "classification": classify_ci(float(row["lo"]), float(row["hi"]), sesoi),
        }
    off_ceiling = 0.30 <= scores["exact_global"] <= 0.95
    return {
        "schema": CELL_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "frames": frames,
        "mechanisms": list(mechanisms),
        "corpus": {"content_sha256": data_sha},
        "difficulty_calibration": {"clears_floor": True},
        "parameters": {
            "frames": frames,
            "parameters": {
                mechanism: (
                    EXPECTED_TRANSFORMER_PARAMETERS[frames] - 512
                    if mechanism == "recurrent"
                    else EXPECTED_TRANSFORMER_PARAMETERS[frames]
                )
                for mechanism in mechanisms
            },
            "recurrent_fractional_deviation": (
                512 / EXPECTED_TRANSFORMER_PARAMETERS[frames] if "recurrent" in mechanisms else None
            ),
            "tolerance_fraction": 0.005,
        },
        "expected_seeds": seeds,
        "seeds_completed": len(seeds),
        "seed_results": seed_results,
        "scores": aggregates,
        "frozen_scores": frozen_aggregates,
        "paired_contrasts": contrasts,
        "compute": {
            "dense_reference_steps": dense_steps,
            "dense_flops_per_step": dense_flops,
            "per_mechanism": {
                mechanism: {
                    "estimated_flops_per_step": flops[mechanism],
                    "matched": matches[mechanism],
                    "estimated_total_flops_completed_seeds": matches[mechanism]["arm_total_flops"]
                    * len(seeds),
                }
                for mechanism in mechanisms
            },
        },
        "off_ceiling": off_ceiling,
        "staged_out": not off_ceiling,
        "futility_truncated": False,
        "futility_evidence": None,
        "complete": True,
        "problems": [],
        "all_ok": True,
    }


def _summary(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cell[key]
        for key in (
            "complete",
            "off_ceiling",
            "staged_out",
            "futility_truncated",
            "seeds_completed",
            "scores",
            "paired_contrasts",
            "all_ok",
        )
    }


def _curve(cells: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        mechanism: {f"f{frames}": cells[frames]["scores"][mechanism] for frames in sorted(FRAME_COUNTS)}
        for mechanism in MECHANISMS
    }


def _materialize_cell_artifacts(frame_dir: Path, cell: dict[str, Any]) -> None:
    frames = int(cell["frames"])
    for seed_text, seed_payload in cell["seed_results"].items():
        seed = int(seed_text)
        seed_dir = frame_dir / f"seed_{seed}"
        for mechanism, arm in seed_payload["mechanisms"].items():
            training = arm["training"]
            steps = int(training["completed_steps"])
            model_state, target_state = _fixture_states(frames, seed, mechanism)
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "objective": "predictive",
                "step": steps,
                "config_sha256": seed_payload["config_sha256"],
                "data_sha256": seed_payload["data_sha256"],
                "requirements_sha256": seed_payload["checkpoint_requirements_sha256"],
                "initial_state_sha256": seed_payload["initial_state_sha256"][mechanism],
                "model": model_state,
                "target": target_state,
                "optimizer": {},
                "torch_rng_state": torch.tensor([0], dtype=torch.uint8),
                "losses": [0.5] * steps,
            }
            arm_dir = seed_dir / mechanism
            checkpoint_path = arm_dir / "checkpoint.pt"
            arm_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            checkpoint_binding = {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": file_sha256(checkpoint_path),
            }
            arm_receipt = {
                "schema": ARM_SCHEMA,
                "objective": "predictive",
                "seed": seed,
                "complete": True,
                "stop_reason": None,
                "requested_steps": steps,
                "completed_steps": steps,
                "batch_size": 4,
                "config_sha256": seed_payload["config_sha256"],
                "data_sha256": seed_payload["data_sha256"],
                "requirements_sha256": seed_payload["checkpoint_requirements_sha256"],
                "initial_state_sha256": seed_payload["initial_state_sha256"][mechanism],
                "final_state_sha256": state_sha256(model_state, "fixture checkpoint model"),
                "target_state_sha256": state_sha256(target_state, "fixture checkpoint target"),
                "checkpoint": checkpoint_binding,
                "loss": {
                    "initial": 0.5,
                    "final": 0.5,
                    "minimum": 0.5,
                    "count": steps,
                },
                "compute": {
                    "estimated_flops_per_step": training["estimated_flops_per_step"],
                    "estimated_total_flops": training["estimated_total_flops"],
                    "estimator": "fixture independent arithmetic",
                },
            }
            atomic_json(arm_dir / "arm_receipt.json", arm_receipt)
        atomic_json(seed_dir / "seed_result.json", seed_payload)


def _primary_fixture(root: Path, outcome: str, profile: str = PILOT_PROFILE) -> dict[str, Path]:
    config_path = _write_repo_sources(root)
    config = resolved_profile_config(config_path, profile)
    config_sha = canonical_sha256(config)
    registry_sha = canonical_sha256(_config_rows())
    live_sources = source_bindings(SOURCE_PATHS, root)
    source_sha = canonical_sha256(live_sources)
    checkpoint_sha = canonical_sha256({"registry_sha256": registry_sha, "source_bindings_sha256": source_sha})
    if outcome == "terminal-null":
        seeds = [0]
        scores = dict.fromkeys(MECHANISMS, 0.50)
        frozen = dict.fromkeys(MECHANISMS, 0.50)
        terminal = True
    elif outcome in {"mechanics", "ceiling-null"}:
        seeds = [0]
        scores = dict.fromkeys(MECHANISMS, 0.98)
        frozen = dict.fromkeys(MECHANISMS, 0.50)
        terminal = False
    elif outcome == "favorable":
        seeds = [0, 1, 2, 3, 4]
        scores = {
            "exact_global": 0.80,
            "window_local": 0.50,
            "recurrent": 0.80,
            "hierarchical_pooled": 0.80,
        }
        frozen = dict.fromkeys(MECHANISMS, 0.50)
        terminal = False
    else:
        seeds = [0, 1, 2]
        scores = dict.fromkeys(MECHANISMS, 0.70)
        frozen = dict.fromkeys(MECHANISMS, 0.50)
        terminal = False

    run_dir = root / "runs/p5_context" / profile
    training = config["training"]
    cells = {
        frames: _cell(
            frames,
            seeds,
            scores,
            frozen,
            config_sha,
            registry_sha,
            source_sha,
            checkpoint_sha,
            int(training["dense_steps"]),
            int(training["checkpoint_every"]),
            int(training["batch_size"]),
            0.1,
        )
        for frames in FRAME_COUNTS
    }
    if outcome == "null":
        for cell in cells.values():
            cell["futility_truncated"] = True
            cell["futility_evidence"] = {
                "paired_mean_deltas": {
                    mechanism: 0.0 for mechanism in MECHANISMS if mechanism != "exact_global"
                },
                "futility_margin": 0.1,
                "seeds_kept": [0, 1, 2],
            }
    for frames, cell in cells.items():
        frame_dir = run_dir / "frames" / f"f{frames}"
        _materialize_cell_artifacts(frame_dir, cell)
        atomic_json(frame_dir / "cell_receipt.json", cell)
    atomic_json(run_dir / "resolved_config.json", config)
    exact_delta = scores["exact_global"] - frozen["exact_global"]
    gate_failed = exact_delta <= 0.05
    receipt = {
        "schema": PILOT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "config_sha256": config_sha,
        "cell_registry_sha256": registry_sha,
        "source_bindings": live_sources,
        "source_bindings_sha256": source_sha,
        "checkpoint_requirements_sha256": checkpoint_sha,
        "profile": profile,
        "serial_order": [f"f{row['frames']}_{row['mechanism']}" for row in _config_rows()],
        "seeds": list(training["seeds"]),
        "dense_reference_steps": int(training["dense_steps"]),
        "frames": {f"f{frames}": _summary(cells[frames]) for frames in FRAME_COUNTS},
        "primary_contrasts_f64": cells[64]["paired_contrasts"],
        "secondary_contrasts_f32": cells[32]["paired_contrasts"],
        "context_response_curve": _curve(cells),
        "sesoi": 0.1,
        "trainability_gate": {
            "applies": True,
            "margin": 0.05,
            "evaluated": True,
            "trained_heldout": scores["exact_global"],
            "frozen_heldout": frozen["exact_global"],
            "delta": exact_delta,
            "failed": gate_failed,
            "outcome": "null" if gate_failed else "clears-margin",
        },
        "trainability_gate_failed": gate_failed,
        "fresh_challenge_required": outcome == "favorable",
        "staging": {
            "off_ceiling": {f"f{frames}": cells[frames]["off_ceiling"] for frames in FRAME_COUNTS},
            "futility_truncated": {
                f"f{frames}": cells[frames]["futility_evidence"] for frames in FRAME_COUNTS
            },
        },
        "promotion": dict(PILOT_PROMOTION),
        "complete": True,
        "resumable": False,
        "execution_status": "terminal-scientific-null" if terminal else "complete",
        "terminal_scientific_stop": terminal,
        "terminal_stop_reason": "f64-trainability-gate-null" if terminal else None,
        "stopped_for_wall_budget": False,
        "stopped_for_disk_floor": False,
        "stopped_for_required_arm_refusal": False,
        "required_arm_failure": None,
        "problems": [],
        "all_ok": True,
    }
    _seal(receipt)
    raw_path = run_dir / "p5_context_receipt.json"
    proof_name = (
        "P5_CONTEXT_CAPABILITY_SMOKE.json" if profile == SMOKE_PROFILE else "P5_CONTEXT_CAPABILITY_PILOT.json"
    )
    proof_path = root / "proof" / proof_name
    atomic_json(raw_path, receipt)
    atomic_json(proof_path, receipt)
    return {
        "config": config_path,
        "run_dir": run_dir,
        "primary": proof_path,
        "challenge": root / "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json",
        "verification": root / "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json",
    }


def _fresh_raw_run(
    root: Path,
    primary_config: dict[str, Any],
    seed: int,
    subrun: Path,
    delta: float,
) -> dict[str, Any]:
    config = challenge_seed_config(primary_config, seed)
    atomic_json(subrun / "resolved_config.json", config)
    config_sha = canonical_sha256(config)
    registry_sha = canonical_sha256(config["cells"])
    live_sources = source_bindings(SOURCE_PATHS, root)
    source_sha = canonical_sha256(live_sources)
    checkpoint_sha = canonical_sha256({"registry_sha256": registry_sha, "source_bindings_sha256": source_sha})
    scores = {
        "exact_global": 0.80,
        "window_local": 0.80 - delta,
        "recurrent": 0.80,
        "hierarchical_pooled": 0.80,
    }
    frozen = dict.fromkeys(MECHANISMS, 0.50)
    cells = {
        frames: _cell(
            frames,
            [seed],
            scores,
            frozen,
            config_sha,
            registry_sha,
            source_sha,
            checkpoint_sha,
            int(config["training"]["dense_steps"]),
            int(config["training"]["checkpoint_every"]),
            int(config["training"]["batch_size"]),
            0.1,
        )
        for frames in FRAME_COUNTS
    }
    for frames, cell in cells.items():
        frame_dir = subrun / "frames" / f"f{frames}"
        _materialize_cell_artifacts(frame_dir, cell)
        atomic_json(frame_dir / "cell_receipt.json", cell)
    raw = {
        "schema": PILOT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "config_sha256": config_sha,
        "cell_registry_sha256": registry_sha,
        "source_bindings": live_sources,
        "source_bindings_sha256": source_sha,
        "checkpoint_requirements_sha256": checkpoint_sha,
        "profile": f"p5fresh-seed-{seed}",
        "serial_order": [f"f{row['frames']}_{row['mechanism']}" for row in config["cells"]],
        "seeds": [seed],
        "dense_reference_steps": int(config["training"]["dense_steps"]),
        "frames": {f"f{frames}": _summary(cells[frames]) for frames in FRAME_COUNTS},
        "context_response_curve": _curve(cells),
        "fresh_challenge_required": False,
        "sesoi": 0.1,
        "staging": {
            "off_ceiling": {f"f{frames}": cells[frames]["off_ceiling"] for frames in FRAME_COUNTS},
            "futility_truncated": {f"f{frames}": None for frames in FRAME_COUNTS},
        },
        "complete": True,
        "resumable": False,
        "problems": [],
        "all_ok": True,
        "trainability_gate": {
            "applies": True,
            "margin": 0.05,
            "evaluated": True,
            "trained_heldout": scores["exact_global"],
            "frozen_heldout": frozen["exact_global"],
            "delta": scores["exact_global"] - frozen["exact_global"],
            "failed": False,
            "outcome": "clears-margin",
        },
        "trainability_gate_failed": False,
        "promotion": dict(PILOT_PROMOTION),
        "execution_status": "complete",
        "terminal_scientific_stop": False,
        "terminal_stop_reason": None,
        "stopped_for_wall_budget": False,
        "stopped_for_disk_floor": False,
        "stopped_for_required_arm_refusal": False,
        "required_arm_failure": None,
    }
    _seal(raw)
    atomic_json(subrun / "p5_context_receipt.json", raw)
    return raw


def _challenge_fixture(paths: dict[str, Path], root: Path, delta: float = 0.30) -> None:
    primary = json.loads(paths["primary"].read_text(encoding="utf-8"))
    primary_config = resolved_profile_config(paths["config"])
    patterns = [
        {
            "id": f"f{frames}-exact-minus-window_local",
            "frames": frames,
            "mechanism": "window_local",
            "direction": "exact-over-factorized",
        }
        for frames in (64, 32)
    ]
    run_root = root / "runs/p5_context/fresh_challenge"
    rows = []
    for seed in FRESH_TRAINING_SEEDS:
        subrun = run_root / f"seed_{seed}"
        raw = _fresh_raw_run(root, primary_config, seed, subrun, delta)
        raw_path = subrun / "p5_context_receipt.json"
        config_file = subrun / "resolved_config.json"
        cell_bindings = {}
        for frames in FRAME_COUNTS:
            cell_path = subrun / "frames" / f"f{frames}" / "cell_receipt.json"
            cell_bindings[f"f{frames}"] = {
                "path": display_path(cell_path, root),
                "sha256": file_sha256(cell_path),
            }
        rows.append(
            {
                "seed": seed,
                "raw_receipt": {
                    "path": display_path(raw_path, root),
                    "sha256": file_sha256(raw_path),
                    "payload_sha256": raw["payload_sha256"],
                },
                "cell_receipts": cell_bindings,
                "resolved_config": {
                    "path": display_path(config_file, root),
                    "sha256": file_sha256(config_file),
                },
                "complete": True,
                "resumable": False,
                "problems": [],
                "all_ok": True,
            }
        )
    receipt = {
        "schema": CHALLENGE_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": CHALLENGE_EVIDENCE_CLASS,
        "source_bindings": source_bindings(CHALLENGE_SOURCE_PATHS, root),
        "primary_receipt": {
            "path": display_path(paths["primary"], root),
            "sha256": file_sha256(paths["primary"]),
            "payload_sha256": primary["payload_sha256"],
        },
        "primary_run_dir": display_path(paths["run_dir"], root),
        "run_dir": display_path(run_root, root),
        "patterns": patterns,
        "fresh_training_seeds": list(FRESH_TRAINING_SEEDS),
        "fresh_seeds_disjoint_from_primary": True,
        "controls": dict(CHALLENGE_CONTROLS),
        "checkpoint_globs": [
            f"{display_path(run_root, root)}/seed_*/frames/f*/seed_*/*/checkpoint.pt",
            f"{display_path(run_root, root)}/seed_*/frames/f*/seed_*/*/arm_receipt.json",
            f"{display_path(run_root, root)}/seed_*/frames/f*/seed_*/seed_result.json",
            f"{display_path(run_root, root)}/seed_*/frames/f*/cell_receipt.json",
            f"{display_path(run_root, root)}/seed_*/p5_context_receipt.json",
            f"{display_path(run_root, root)}/seed_*/resolved_config.json",
        ],
        "resource_contract": dict(CHALLENGE_RESOURCE_CONTRACT),
        "training_runs": rows,
        "complete": True,
        "resumable": False,
        "verification_ready": True,
        "problems": [],
        "all_ok": True,
        "promotion": dict(CHALLENGE_PROMOTION),
        "scientific_promotion": False,
    }
    _seal(receipt)
    atomic_json(paths["challenge"], receipt)


@pytest.mark.parametrize(
    ("primary_outcome", "expected", "prerequisite_ready"),
    [
        ("terminal-null", "null", True),
        ("mechanics", "mechanics", False),
        ("null", "null", True),
    ],
)
def test_verifier_distinguishes_nonfavorable_outcomes_without_fresh_training(
    tmp_path: Path, primary_outcome: str, expected: str, prerequisite_ready: bool
) -> None:
    paths = _primary_fixture(tmp_path, primary_outcome)
    receipt = build_verification(
        paths["primary"],
        paths["run_dir"],
        paths["config"],
        paths["challenge"],
        repo_root=tmp_path,
    )
    assert receipt["schema"] == VERIFY_SCHEMA
    assert receipt["primary_profile"] == PILOT_PROFILE
    assert receipt["classification"] == expected
    assert receipt["prerequisite_ready"] is prerequisite_ready
    assert receipt["fresh_challenge_required"] is False
    assert receipt["verification_complete"] is True
    assert receipt["all_ok"] is True
    assert receipt["fresh_challenge"] is None
    assert receipt["scientific_promotion"] is False
    assert receipt["outcome_contract"]["tie_is_null"] is True
    assert receipt["all_mutations_rejected"] is True
    assert len(receipt["mutation_tests"]) == 18


def test_terminal_null_smoke_resolves_its_own_profile_and_is_p6_ready(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "terminal-null", SMOKE_PROFILE)
    receipt = build_verification(
        paths["primary"],
        paths["run_dir"],
        paths["config"],
        paths["challenge"],
        repo_root=tmp_path,
    )
    assert receipt["primary_profile"] == SMOKE_PROFILE
    assert receipt["primary_outcome"] == "null"
    assert receipt["classification"] == "null"
    assert receipt["terminal_null"] is True
    assert receipt["fresh_challenge_required"] is False
    assert receipt["fresh_challenge"] is None
    assert receipt["prerequisite_ready"] is True
    assert receipt["all_ok"] is True
    assert len(receipt["mutation_tests"]) == 18


def test_nonterminal_smoke_is_refused_and_never_challenge_authorized(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "mechanics", SMOKE_PROFILE)
    with pytest.raises(P5VerificationRefused, match="p5smoke.*terminal-scientific-null"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )
    smoke_root = tmp_path / "terminal-smoke"
    terminal_paths = _primary_fixture(smoke_root, "terminal-null", SMOKE_PROFILE)
    with pytest.raises(P5VerificationRefused, match="only for p5pilot"):
        run_fresh_challenge(
            terminal_paths["primary"],
            terminal_paths["run_dir"],
            terminal_paths["config"],
            smoke_root / "runs/p5_context/fresh_challenge",
            terminal_paths["challenge"],
            "cpu",
            repo_root=smoke_root,
        )


def test_verifier_refuses_favorable_primary_until_fresh_training_exists(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    with pytest.raises(P5VerificationRefused, match="challenge.*missing"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


def test_verifier_rejects_resealed_favorable_subset_of_configured_seeds(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    selected = [0, 1]
    for frames in FRAME_COUNTS:
        cell_path = paths["run_dir"] / "frames" / f"f{frames}" / "cell_receipt.json"
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        for seed in range(5):
            seed_payload = cell["seed_results"][str(seed)]
            selected_seed = seed in selected
            scores = {
                "exact_global": 0.9 if selected_seed else 0.7,
                "window_local": 0.5 if selected_seed else 0.7,
                "recurrent": 0.9 if selected_seed else 0.7,
                "hierarchical_pooled": 0.9 if selected_seed else 0.7,
            }
            for mechanism, score in scores.items():
                seed_payload["mechanisms"][mechanism]["evaluation"]["heldout_combo_score"] = score
            atomic_json(
                paths["run_dir"] / "frames" / f"f{frames}" / f"seed_{seed}" / "seed_result.json",
                seed_payload,
            )

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
        cell["scores"] = {mechanism: _ci(values) for mechanism, values in trained.items()}
        cell["frozen_scores"] = {mechanism: _ci(values) for mechanism, values in frozen.items()}
        cell["paired_contrasts"] = {}
        for mechanism in MECHANISMS:
            if mechanism == "exact_global":
                continue
            values = [
                left - right for left, right in zip(trained["exact_global"], trained[mechanism], strict=True)
            ]
            ci = _ci(values)
            cell["paired_contrasts"][f"exact_minus_{mechanism}"] = {
                **ci,
                "classification": classify_ci(float(ci["lo"]), float(ci["hi"]), 0.1),
            }
        for compute in cell["compute"]["per_mechanism"].values():
            compute["estimated_total_flops_completed_seeds"] = compute["matched"]["arm_total_flops"] * len(
                selected
            )
        cell["staged_out"] = False
        cell["futility_truncated"] = False
        cell["futility_evidence"] = None
        atomic_json(cell_path, cell)

    cells = {
        frames: json.loads(
            (paths["run_dir"] / "frames" / f"f{frames}" / "cell_receipt.json").read_text(encoding="utf-8")
        )
        for frames in FRAME_COUNTS
    }
    receipt = json.loads(paths["primary"].read_text(encoding="utf-8"))
    receipt["frames"] = {f"f{frames}": _summary(cells[frames]) for frames in FRAME_COUNTS}
    receipt["primary_contrasts_f64"] = cells[64]["paired_contrasts"]
    receipt["secondary_contrasts_f32"] = cells[32]["paired_contrasts"]
    receipt["context_response_curve"] = _curve(cells)
    receipt["trainability_gate"].update(
        {
            "trained_heldout": 0.9,
            "frozen_heldout": 0.5,
            "delta": 0.4,
            "failed": False,
            "outcome": "clears-margin",
        }
    )
    receipt["trainability_gate_failed"] = False
    receipt["fresh_challenge_required"] = True
    _seal(receipt)
    atomic_json(paths["primary"], receipt)
    atomic_json(paths["run_dir"] / "p5_context_receipt.json", receipt)

    full_five_ci = _ci([0.4, 0.4, 0.0, 0.0, 0.0])
    selected_ci = cells[64]["paired_contrasts"]["exact_minus_window_local"]
    assert float(full_five_ci["lo"]) <= 0.1
    assert float(selected_ci["lo"]) > 0.1
    with pytest.raises(P5VerificationRefused, match="licensed seed set"):
        audit_primary(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("staged-out", "staged_out decision"),
        ("futility-flag", "futility decision"),
        ("futility-evidence", "futility evidence"),
        ("top-staging", "top-level staging authority"),
    ],
)
def test_verifier_rejects_resealed_staging_authority_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    paths = _primary_fixture(tmp_path, "null")
    cell_path = paths["run_dir"] / "frames/f64/cell_receipt.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    receipt = json.loads(paths["primary"].read_text(encoding="utf-8"))
    if mutation == "staged-out":
        cell["staged_out"] = True
    elif mutation == "futility-flag":
        cell["futility_truncated"] = False
    elif mutation == "futility-evidence":
        cell["futility_evidence"]["futility_margin"] = 0.2
    else:
        receipt["staging"]["futility_truncated"]["f64"] = None
    if mutation != "top-staging":
        atomic_json(cell_path, cell)
        receipt["frames"]["f64"] = _summary(cell)
    _seal(receipt)
    atomic_json(paths["primary"], receipt)
    atomic_json(paths["run_dir"] / "p5_context_receipt.json", receipt)

    with pytest.raises(P5VerificationRefused, match=message):
        audit_primary(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            repo_root=tmp_path,
        )


def test_verifier_rejects_context_response_curve_drift(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "null")
    receipt = json.loads(paths["primary"].read_text(encoding="utf-8"))
    receipt["context_response_curve"]["exact_global"]["f64"]["mean"] = 0.01
    _seal(receipt)
    atomic_json(paths["primary"], receipt)
    atomic_json(paths["run_dir"] / "p5_context_receipt.json", receipt)

    with pytest.raises(P5VerificationRefused, match="context response curve"):
        audit_primary(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            repo_root=tmp_path,
        )


def test_fresh_disjoint_training_closes_only_a_programmatic_favorable_pattern(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    _challenge_fixture(paths, tmp_path)
    receipt = build_verification(
        paths["primary"],
        paths["run_dir"],
        paths["config"],
        paths["challenge"],
        repo_root=tmp_path,
    )
    assert receipt["primary_profile"] == PILOT_PROFILE
    assert receipt["primary_outcome"] == "favorable-programmatic-only"
    assert receipt["fresh_challenge_required"] is True
    assert receipt["classification"] == "favorable-programmatic-only"
    assert len(receipt["verified_patterns"]) == 2
    assert all(row["scientific_promotion_allowed"] is False for row in receipt["verified_patterns"])
    assert all(row["tie_is_null"] is True for row in receipt["verified_patterns"])
    assert all(row["strict_direction_reproduced"] is True for row in receipt["verified_patterns"])
    assert receipt["promotion"]["confirmatory_promotable"] is False
    assert receipt["prerequisite_ready"] is True
    assert receipt["all_mutations_rejected"] is True
    assert receipt["independence"]["checkpoint_model_and_target_state_hashes_recomputed"] is True
    assert receipt["independence"]["heldout_metrics_reexecuted_from_checkpoint"] is False
    assert receipt["artifact_evidence"]["primary"]["f64"]["0"]["arms"]["exact_global"]["checkpoint"][
        "model_state_sha256"
    ]
    assert set(receipt["cell_receipt_evidence"]["primary"]) == {"f64", "f32", "f16"}
    assert set(receipt["cell_receipt_evidence"]["fresh_challenge"]) == {
        "5101",
        "5102",
        "5103",
    }
    assert len(receipt["mutation_tests"]) == 23
    assert {row["id"] for row in receipt["mutation_tests"]} >= {
        "cached-seed-source-drift",
        "checkpoint-source-drift",
        "matched-compute-drift",
        "fresh-challenge-hint-flip",
        "seed-selection-drift",
        "sealed-profile-config-mismatch",
        "fresh-seed-overlap",
        "fresh-run-drop",
        "threshold-tie-promotion",
        "ceilinged-contrast-promotion",
        "missing-seed-result-artifact",
        "missing-arm-receipt-artifact",
        "missing-checkpoint-artifact",
        "checkpoint-file-hash-drift",
        "challenge-shape-omission",
        "fresh-trainability-gate-fabrication",
    }


def test_fresh_training_nonreplication_converts_primary_pattern_to_null(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    _challenge_fixture(paths, tmp_path, delta=0.0)
    receipt = build_verification(
        paths["primary"],
        paths["run_dir"],
        paths["config"],
        paths["challenge"],
        repo_root=tmp_path,
    )
    assert receipt["primary_outcome"] == "favorable-programmatic-only"
    assert receipt["classification"] == "null"
    assert receipt["fresh_challenge_required"] is True
    assert receipt["prerequisite_ready"] is True
    assert receipt["verified_patterns"] == []
    assert all(row["tie_is_null"] is True for row in receipt["fresh_challenge"]["per_pattern"])


def test_ceilinged_multi_seed_tie_is_mechanics_and_not_prerequisite_ready(
    tmp_path: Path,
) -> None:
    paths = _primary_fixture(tmp_path, "ceiling-null")
    receipt = build_verification(
        paths["primary"],
        paths["run_dir"],
        paths["config"],
        paths["challenge"],
        repo_root=tmp_path,
    )
    assert receipt["classification"] == "mechanics"
    assert receipt["prerequisite_ready"] is False
    assert receipt["all_controls_passed"] is False
    assert receipt["controls"]["primary_off_ceiling"] == {"f64": False, "f32": False}


def test_skipping_mutations_fails_closed(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "null")
    receipt = build_verification(
        paths["primary"],
        paths["run_dir"],
        paths["config"],
        paths["challenge"],
        repo_root=tmp_path,
        run_mutations=False,
    )
    assert receipt["all_mutations_rejected"] is False
    assert receipt["mutation_tests"] == []
    assert receipt["verification_complete"] is False
    assert receipt["all_ok"] is False
    assert receipt["prerequisite_ready"] is False
    assert receipt["problems"] == ["P5 adversarial mutation suite was not run"]


@pytest.mark.parametrize(
    ("artifact", "expected_error"),
    [
        ("seed_result.json", "durable result.*missing"),
        ("exact_global/arm_receipt.json", "arm receipt.*missing"),
        ("exact_global/checkpoint.pt", "checkpoint is missing"),
    ],
)
def test_primary_refuses_missing_training_artifacts(
    tmp_path: Path, artifact: str, expected_error: str
) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    target = paths["run_dir"] / "frames/f64/seed_0" / artifact
    target.unlink()
    with pytest.raises(P5VerificationRefused, match=expected_error):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


def test_primary_refuses_checkpoint_file_hash_drift(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "null")
    arm_path = paths["run_dir"] / "frames/f64/seed_0/exact_global/arm_receipt.json"
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    arm["checkpoint"]["sha256"] = "0" * 64
    atomic_json(arm_path, arm)
    with pytest.raises(P5VerificationRefused, match="checkpoint file hash"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


def test_challenge_refuses_shape_omission(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    _challenge_fixture(paths, tmp_path)
    challenge = json.loads(paths["challenge"].read_text(encoding="utf-8"))
    challenge.pop("resource_contract")
    _seal(challenge)
    atomic_json(paths["challenge"], challenge)
    with pytest.raises(P5VerificationRefused, match="receipt shape"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


def test_fresh_trainability_is_recomputed_from_raw_scores(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    _challenge_fixture(paths, tmp_path)
    challenge = json.loads(paths["challenge"].read_text(encoding="utf-8"))
    for row in challenge["training_runs"]:
        for frames in FRAME_COUNTS:
            binding = row["cell_receipts"][f"f{frames}"]
            cell_path = tmp_path / binding["path"]
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            seed = int(cell["expected_seeds"][0])
            payload = cell["seed_results"][str(seed)]
            trained = payload["mechanisms"]["exact_global"]["evaluation"]["heldout_combo_score"]
            payload["mechanisms"]["exact_global"]["frozen"]["evaluation"]["heldout_combo_score"] = trained
            cell["frozen_scores"]["exact_global"] = _ci([trained])
            atomic_json(
                cell_path.parent / f"seed_{seed}" / "seed_result.json",
                payload,
            )
            atomic_json(cell_path, cell)
            binding["sha256"] = file_sha256(cell_path)
    _seal(challenge)
    atomic_json(paths["challenge"], challenge)
    with pytest.raises(P5VerificationRefused, match="trainability gate score binding"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


@pytest.mark.parametrize("field", ["complete", "all_ok"])
def test_verifier_refuses_incomplete_or_all_ok_false_primary(tmp_path: Path, field: str) -> None:
    paths = _primary_fixture(tmp_path, "null")
    for path in (paths["primary"], paths["run_dir"] / "p5_context_receipt.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = False
        _seal(payload)
        atomic_json(path, payload)
    with pytest.raises(P5VerificationRefused, match="incomplete|all_ok"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


def test_verifier_rejects_resealed_fresh_challenge_authorization_hint_flip(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "null")
    for path in (paths["primary"], paths["run_dir"] / "p5_context_receipt.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fresh_challenge_required"] = True
        _seal(payload)
        atomic_json(path, payload)
    with pytest.raises(P5VerificationRefused, match="fresh_challenge_required"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


@pytest.mark.parametrize("relative", P5_CORE_RUNTIME_SOURCES)
def test_verifier_refuses_each_core_runtime_source_drift(tmp_path: Path, relative: str) -> None:
    paths = _primary_fixture(tmp_path, "null")
    assert SOURCE_PATHS[-2:] == P5_CORE_RUNTIME_SOURCES
    assert CHALLENGE_SOURCE_PATHS[: len(SOURCE_PATHS)] == SOURCE_PATHS
    assert VERIFIER_SOURCE_PATHS[: len(SOURCE_PATHS)] == SOURCE_PATHS
    (tmp_path / relative).write_text("mutated\n", encoding="utf-8")
    with pytest.raises(P5VerificationRefused, match="source binding"):
        build_verification(
            paths["primary"],
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


def test_verifier_refuses_missing_primary(tmp_path: Path) -> None:
    paths = _primary_fixture(tmp_path, "null")
    with pytest.raises(P5VerificationRefused, match="missing"):
        build_verification(
            tmp_path / "missing.json",
            paths["run_dir"],
            paths["config"],
            paths["challenge"],
            repo_root=tmp_path,
        )


def test_threshold_equality_is_a_null() -> None:
    assert classify_ci(0.1, 0.1, 0.1) == "bounded_within_sesoi"
    assert classify_ci(-0.1, -0.1, 0.1) == "bounded_within_sesoi"
    assert classify_ci(0.1000001, 0.2, 0.1) == "meaningful_positive"


def test_challenge_runner_materializes_every_fresh_seed_as_a_full_isolated_surface(
    tmp_path: Path,
) -> None:
    paths = _primary_fixture(tmp_path, "favorable")
    primary_config = resolved_profile_config(paths["config"])

    def fake_runner(
        config: dict[str, Any],
        subrun: Path,
        device: str,
        *,
        repo_root: Path,
    ) -> dict[str, Any]:
        del device
        seed = int(config["training"]["seeds"][0])
        expected = challenge_seed_config(primary_config, seed)
        assert config == expected
        return _fresh_raw_run(repo_root, primary_config, seed, subrun, 0.30)

    receipt = run_fresh_challenge(
        paths["primary"],
        paths["run_dir"],
        paths["config"],
        tmp_path / "runs/p5_context/fresh_challenge",
        paths["challenge"],
        "cpu",
        repo_root=tmp_path,
        runner=fake_runner,
    )
    assert receipt["schema"] == CHALLENGE_SCHEMA
    assert receipt["complete"] is True and receipt["all_ok"] is True
    assert len(receipt["training_runs"]) == len(FRESH_TRAINING_SEEDS)
    assert set(P5_CORE_RUNTIME_SOURCES) <= {row["path"] for row in receipt["source_bindings"]}
    for row in receipt["training_runs"]:
        raw_path = tmp_path / row["raw_receipt"]["path"]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert [binding["path"] for binding in raw["source_bindings"]] == list(SOURCE_PATHS)
    assert challenge_exit_code(receipt) == 0
    assert receipt["scientific_promotion"] is False
    assert receipt["resource_contract"]["resumable_exit_code"] == 2
    challenge_root = display_path(tmp_path / "runs/p5_context/fresh_challenge", tmp_path)
    assert receipt["checkpoint_globs"] == [
        f"{challenge_root}/seed_*/frames/f*/seed_*/*/checkpoint.pt",
        f"{challenge_root}/seed_*/frames/f*/seed_*/*/arm_receipt.json",
        f"{challenge_root}/seed_*/frames/f*/seed_*/seed_result.json",
        f"{challenge_root}/seed_*/frames/f*/cell_receipt.json",
        f"{challenge_root}/seed_*/p5_context_receipt.json",
        f"{challenge_root}/seed_*/resolved_config.json",
    ]
    assert receipt["payload_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "payload_sha256"}
    )
