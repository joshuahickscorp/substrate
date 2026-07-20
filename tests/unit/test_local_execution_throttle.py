import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import scripts.continual_million_event_rung as p6_runner
import torch
import yaml

import mop.studio.local_throttle as throttle
from mop.config import REPO_ROOT
from mop.studies import p5_context_verify as p5_verify
from mop.studio.local_throttle import (
    DECISION_SCHEMA,
    TaskDeclaration,
    ThrottleRefused,
    active_lanes,
    aggregate_admission,
    checkpoint_snapshot,
    evaluate_task,
    hysteresis_transition,
    load_policy,
)


def _snapshot(**overrides):
    payload = {
        "cpu": {
            "available": True,
            "logical_cpus": 12,
            "load_1m_per_logical_cpu": 0.20,
            "utilization_fraction": 0.20,
        },
        "memory": {
            "available": True,
            "total_bytes": int(19.3e9),
            "available_bytes": int(14e9),
            "available_percent": 70.0,
            "pressure": {"available": True, "free_percent": 68.0},
        },
        "swap": {"available": True, "used_gb": 1.0},
        "disk": {"available": True, "free_gb": 80.0},
        "processes": {
            "available": True,
            "foreground_resource_processes": [],
            "unmanaged_known_heavy": [],
        },
        "mps": {
            "telemetry_available": True,
            "available": True,
            "declared_headroom_bytes": int(14e9),
            "scope": "test",
        },
        "thermal": {"available": True, "status": "normal"},
        "power": {"available": True, "source": "AC Power", "on_ac": True},
        "missing_required_telemetry": [],
        "all_required_available": True,
    }
    payload.update(overrides)
    return payload


def _active(task, run_id="active"):
    return {
        "run_id": run_id,
        "lane": task.lane,
        "accelerator": task.accelerator,
        "cpu_cores": task.cpu_cores,
        "estimated_unified_memory_gb": task.estimated_unified_memory_gb,
        "estimated_mps_gb": task.estimated_mps_gb,
        "forecast_write_gb": task.forecast_write_gb,
        "atomic_write_gb": task.atomic_write_gb,
    }


def _write_payload_receipt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.pop("payload_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(payload))


def _native_seal(payload, field):
    payload.pop(field, None)
    payload[field] = throttle._canonical_sha256(payload)
    return payload


def _write_native_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _edcm_native_producer():
    core = {
        "schema": throttle.EDCM_RECEIPT_SCHEMA,
        "authority_sha256": "a" * 64,
        "implementation_authority_sha256": "b" * 64,
        "execution_status": "complete",
        "aggregate": {"verdict": "strong-null-not-supported"},
        "scientific_promotion": False,
    }
    payload = {**core, "deterministic_core_sha256": throttle._canonical_sha256(core)}
    return _native_seal(payload, "receipt_sha256")


def _x0_native_producer():
    return _native_seal(
        {
            "schema": throttle.X0_RECEIPT_SCHEMA,
            "implementation_authority": {"manifest_sha256": "c" * 64},
            "aggregate": {"verdict": "strong_null_not_rejected"},
            "scientific_promotion": False,
        },
        "receipt_sha256",
    )


def _seal_governor_run(root: Path, task_id: str) -> dict:
    policy = load_policy()
    task = policy.task(task_id)
    output_path = throttle._task_output_path(task)
    assert output_path is not None
    output = root / output_path
    assert output.is_file()
    snapshot = checkpoint_snapshot(task, root)
    output_sha = throttle._sha256_file(output)
    child_resource = {
        "psutil_peak_rss_bytes": 150_000_000,
        "direct_child_rusage_peak_rss_bytes": 200_000_000,
        "peak_rss_bytes": 200_000_000,
        "methods": ["psutil-process-tree", "getrusage-RUSAGE_CHILDREN"],
    }
    policy_binding = {"path": str(policy.path), "sha256": policy.sha256}
    implementation = {
        "path": "src/mop/studio/local_throttle.py",
        "sha256": throttle._sha256_file(throttle.IMPLEMENTATION_PATH),
    }
    command = list(task.command)
    command_sha = throttle._command_sha256(task.command)
    task_policy_authority = throttle._build_task_policy_authority(policy, task)
    receipt = {
        "schema": throttle.RECEIPT_SCHEMA,
        "mode": "execute",
        "run_id": f"fixture-{task_id}",
        "status": "complete",
        "policy": policy_binding,
        "implementation": implementation,
        "task_policy_authority": task_policy_authority,
        "task": throttle._json_value(throttle.asdict(task)),
        "command_executed": True,
        "invocations": [
            {
                "index": 1,
                "pid": 999_999_999,
                "command": command,
                "command_sha256": command_sha,
                "returncode": 0,
            }
        ],
        "final_returncode": 0,
        "final_checkpoint": snapshot,
        "child_resource": child_resource,
        "completion_authority": {
            "schema": throttle.COMPLETION_AUTHORITY_SCHEMA,
            "task_id": task_id,
            "task": throttle._json_value(throttle.asdict(task)),
            "command": command,
            "command_sha256": command_sha,
            "policy": policy_binding,
            "implementation": implementation,
            "task_policy_authority": task_policy_authority,
            "returncode": 0,
            "output": {"path": output_path, "sha256": output_sha},
            "final_checkpoint_aggregate_sha256": snapshot["aggregate_sha256"],
            "owned_child_active": False,
            "child_resource": child_resource,
        },
    }
    receipt["payload_sha256"] = throttle._canonical_sha256(receipt)
    receipt_path = root / "runs/local_throttle" / f"fixture-{task_id}" / "run_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return receipt


def _p5_mechanisms_by_frame(config):
    result = {frames: [] for frames in throttle.P5_FRAME_COUNTS}
    for row in config["cells"]:
        result[int(row["frames"])].append(str(row["mechanism"]))
    return result


def _p5_ci(value, count):
    return {"n": count, "mean": value, "lo": value, "hi": value, "half": 0.0}


def _p5_cell(
    *,
    frames,
    mechanisms,
    seeds,
    scores,
    frozen_scores,
    config_sha,
    registry_sha,
    source_sha,
    checkpoint_sha,
    dense_steps,
    checkpoint_every,
    batch_size,
    sesoi=0.1,
):
    expected_flops = {
        mechanism: p5_verify._expected_flops_per_step(frames, mechanism, batch_size)
        for mechanism in mechanisms
    }
    dense_flops = expected_flops["exact_global"]
    matches = {
        mechanism: p5_verify._expected_match(
            dense_steps,
            dense_flops,
            expected_flops[mechanism],
            checkpoint_every,
            exact=mechanism == "exact_global",
        )
        for mechanism in mechanisms
    }
    seed_results = {}
    for seed in seeds:
        arms = {}
        for mechanism in mechanisms:
            initial_sha = f"{frames + seed + len(mechanism):064x}"[-64:]
            final_sha = f"{frames + seed + len(mechanism) + 1000:064x}"[-64:]
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
                    "completed_steps": matches[mechanism]["steps"],
                    "estimated_flops_per_step": expected_flops[mechanism],
                    "estimated_total_flops": matches[mechanism]["arm_total_flops"],
                    "requirements_sha256": checkpoint_sha,
                    "final_state_sha256": final_sha,
                },
                "evaluation": {
                    "heldout_combo_score": scores[mechanism],
                    "chance": 0.25,
                },
            }
        seed_results[str(seed)] = {
            "schema": "mop-p5-context-seed/v1",
            "frames": frames,
            "seed": seed,
            "config_sha256": config_sha,
            "data_sha256": f"{frames:064x}"[-64:],
            "registry_sha256": registry_sha,
            "source_bindings_sha256": source_sha,
            "checkpoint_requirements_sha256": checkpoint_sha,
            "mechanisms": arms,
            "complete": True,
        }
    score_rows = {mechanism: _p5_ci(scores[mechanism], len(seeds)) for mechanism in mechanisms}
    frozen_rows = {mechanism: _p5_ci(frozen_scores[mechanism], len(seeds)) for mechanism in mechanisms}
    contrasts = {}
    for mechanism in mechanisms:
        if mechanism == "exact_global":
            continue
        delta = scores["exact_global"] - scores[mechanism]
        if len(seeds) < 2:
            classification = "undetermined"
        else:
            classification = (
                "meaningful_positive"
                if delta > sesoi
                else "meaningful_negative"
                if delta < -sesoi
                else "bounded_within_sesoi"
            )
        contrasts[f"exact_minus_{mechanism}"] = {
            **_p5_ci(delta, len(seeds)),
            "classification": classification,
        }
    return {
        "schema": "mop-p5-context-cell/v1",
        "frames": frames,
        "mechanisms": mechanisms,
        "difficulty_calibration": {"clears_floor": True},
        "parameters": {
            "frames": frames,
            "parameters": {
                mechanism: (
                    p5_verify.EXPECTED_TRANSFORMER_PARAMETERS[frames] - p5_verify.RECURRENT_PARAMETER_DEFICIT
                    if mechanism == "recurrent"
                    else p5_verify.EXPECTED_TRANSFORMER_PARAMETERS[frames]
                )
                for mechanism in mechanisms
            },
            "tolerance_fraction": 0.005,
            "recurrent_fractional_deviation": (
                p5_verify.RECURRENT_PARAMETER_DEFICIT / p5_verify.EXPECTED_TRANSFORMER_PARAMETERS[frames]
            ),
        },
        "expected_seeds": seeds,
        "seeds_completed": len(seeds),
        "seed_results": seed_results,
        "scores": score_rows,
        "frozen_scores": frozen_rows,
        "paired_contrasts": contrasts,
        "compute": {
            "dense_reference_steps": dense_steps,
            "dense_flops_per_step": dense_flops,
            "per_mechanism": {
                mechanism: {
                    "estimated_flops_per_step": expected_flops[mechanism],
                    "matched": matches[mechanism],
                    "estimated_total_flops_completed_seeds": (
                        matches[mechanism]["arm_total_flops"] * len(seeds)
                    ),
                }
                for mechanism in mechanisms
            },
        },
        "off_ceiling": True,
        "staged_out": False,
        "futility_truncated": False,
        "futility_evidence": None,
        "complete": True,
        "problems": [],
        "all_ok": True,
    }


def _write_p5_cell_artifacts(run_dir, frames, cell):
    frame_dir = run_dir / "frames" / f"f{frames}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for seed_key, seed in cell["seed_results"].items():
        seed_dir = frame_dir / f"seed_{seed_key}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for mechanism_index, (mechanism, embedded) in enumerate(seed["mechanisms"].items()):
            arm_dir = seed_dir / mechanism
            arm_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = arm_dir / "checkpoint.pt"
            model_state = {
                "weight": torch.tensor(
                    [frames, int(seed_key), mechanism_index],
                    dtype=torch.float32,
                )
            }
            target_state = {
                "weight": torch.tensor(
                    [frames + 1, int(seed_key) + 1, mechanism_index + 1],
                    dtype=torch.float32,
                )
            }
            model_sha = throttle._p5_tensor_state_sha256(model_state, "fixture model")
            target_sha = throttle._p5_tensor_state_sha256(target_state, "fixture target")
            embedded["training"]["final_state_sha256"] = model_sha
            torch.save({"model": model_state, "target": target_state}, checkpoint)
            arm_receipt = {
                "schema": "mop-custom-substrate-arm/v1",
                "objective": "predictive",
                "seed": seed["seed"],
                "complete": True,
                "config_sha256": seed["config_sha256"],
                "data_sha256": seed["data_sha256"],
                "requirements_sha256": seed["checkpoint_requirements_sha256"],
                "initial_state_sha256": embedded["initial_state_sha256"],
                "requested_steps": embedded["matched"]["steps"],
                "completed_steps": embedded["training"]["completed_steps"],
                "final_state_sha256": model_sha,
                "target_state_sha256": target_sha,
                "checkpoint": {"sha256": throttle._sha256_file(checkpoint)},
            }
            (arm_dir / "arm_receipt.json").write_text(json.dumps(arm_receipt))
        (seed_dir / "seed_result.json").write_text(json.dumps(seed))
    (frame_dir / "cell_receipt.json").write_text(json.dumps(cell))


def _p5_summary(cell):
    return throttle._p5_frame_summary(cell)


def _write_p5_screen(root, *, profile, outcome):
    config = throttle._p5_resolved_config(profile)
    config_sha = throttle._canonical_sha256(config)
    registry_sha = throttle._canonical_sha256(config["cells"])
    bindings = throttle._p5_live_bindings(throttle.P5_SOURCE_PATHS)
    source_sha = throttle._canonical_sha256(bindings)
    checkpoint_sha = throttle._canonical_sha256(
        {"registry_sha256": registry_sha, "source_bindings_sha256": source_sha}
    )
    if outcome == "favorable":
        scores = {
            "exact_global": 0.80,
            "window_local": 0.50,
            "recurrent": 0.80,
            "hierarchical_pooled": 0.80,
        }
        frozen = dict.fromkeys(throttle.P5_MECHANISMS, 0.50)
        terminal = False
    elif outcome == "terminal-null":
        scores = dict.fromkeys(throttle.P5_MECHANISMS, 0.50)
        frozen = dict.fromkeys(throttle.P5_MECHANISMS, 0.50)
        terminal = True
    else:
        scores = {
            "exact_global": 0.70,
            "window_local": 0.65,
            "recurrent": 0.70,
            "hierarchical_pooled": 0.70,
        }
        frozen = dict.fromkeys(throttle.P5_MECHANISMS, 0.50)
        terminal = False
    seeds = [int(value) for value in config["training"]["seeds"]]
    by_frame = _p5_mechanisms_by_frame(config)
    run_dir = root / "runs/p5_context" / profile
    cells = {
        frames: _p5_cell(
            frames=frames,
            mechanisms=by_frame[frames],
            seeds=seeds,
            scores=scores,
            frozen_scores=frozen,
            config_sha=config_sha,
            registry_sha=registry_sha,
            source_sha=source_sha,
            checkpoint_sha=checkpoint_sha,
            dense_steps=int(config["training"]["dense_steps"]),
            checkpoint_every=int(config["training"]["checkpoint_every"]),
            batch_size=int(config["training"]["batch_size"]),
        )
        for frames in throttle.P5_FRAME_COUNTS
    }
    for frames, cell in cells.items():
        _write_p5_cell_artifacts(run_dir, frames, cell)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config))
    receipt = {
        "schema": throttle.P5_SCREEN_SCHEMA,
        "config_sha256": config_sha,
        "cell_registry_sha256": registry_sha,
        "source_bindings": bindings,
        "source_bindings_sha256": source_sha,
        "checkpoint_requirements_sha256": checkpoint_sha,
        "profile": profile,
        "serial_order": [f"f{row['frames']}_{row['mechanism']}" for row in config["cells"]],
        "seeds": seeds,
        "frames": {f"f{frames}": _p5_summary(cells[frames]) for frames in throttle.P5_FRAME_COUNTS},
        "context_response_curve": {
            mechanism: {
                f"f{frames}": cells[frames]["scores"][mechanism]
                for frames in sorted(throttle.P5_FRAME_COUNTS)
            }
            for mechanism in throttle.P5_MECHANISMS
        },
        "primary_contrasts_f64": cells[64]["paired_contrasts"],
        "secondary_contrasts_f32": cells[32]["paired_contrasts"],
        "sesoi": 0.1,
        "staging": {
            "off_ceiling": {f"f{frames}": True for frames in throttle.P5_FRAME_COUNTS},
            "futility_truncated": {f"f{frames}": None for frames in throttle.P5_FRAME_COUNTS},
        },
        "trainability_gate": {
            "applies": True,
            "evaluated": True,
            "failed": terminal,
            "outcome": "null" if terminal else "clears-margin",
        },
        "trainability_gate_failed": terminal,
        "fresh_challenge_required": outcome == "favorable",
        "promotion": {
            "confirmatory_promotable": False,
            "refused_by_construction": True,
            "category_9_possible": False,
        },
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
    proof_name = (
        "P5_CONTEXT_CAPABILITY_SMOKE.json" if profile == "p5smoke" else "P5_CONTEXT_CAPABILITY_PILOT.json"
    )
    proof_path = root / "proof" / proof_name
    _write_payload_receipt(proof_path, receipt)
    (run_dir / "p5_context_receipt.json").write_text(json.dumps(receipt))
    _seal_governor_run(root, "p5smoke_cpu" if profile == "p5smoke" else "p5pilot_cpu")
    return receipt


def _write_p5_grid(root):
    config = throttle._p5_resolved_config("p5pilot")
    bindings = throttle._p5_live_bindings(throttle.P5_GRID_SOURCE_PATHS)
    bindings_sha = throttle._canonical_sha256(bindings)
    boundary_sha = throttle._sha256_file(throttle.P5_BOUNDARY_TRACE)
    identity = {
        "script_sha256": throttle._sha256_file(REPO_ROOT / "scripts/p5_traingrid_memory_probe.py"),
        "source_bindings": bindings,
        "source_bindings_sha256": bindings_sha,
        "boundary_trace_sha256": boundary_sha,
        "cells": config["cells"],
        "batch_rows": [4, 1],
        "repeats": 3,
        "seed": 0,
        "mask_ratio": 0.5,
        "ema_decay": 0.99,
        "child_memory_guard_gb": 12.0,
        "device": "cpu",
    }
    rows = {}
    for cell in config["cells"]:
        for batch in (4, 1):
            for repeat in range(3):
                key = f"f{cell['frames']}:{cell['mechanism']}:b{batch}:r{repeat}"
                rows[key] = {
                    "cell": f"f{cell['frames']}_{cell['mechanism']}",
                    "frames": cell["frames"],
                    "mechanism": cell["mechanism"],
                    "batch": batch,
                    "repeat": repeat,
                    "ok": True,
                    "loss_finite": True,
                    "memory_guard_exceeded": False,
                }
    progress_path = root / "proof/P5_TRAINGRID_MEMORY_TRACE.json.progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress = {
        "schema": "mop-p5-traingrid-memory-progress/v1",
        "identity": identity,
        "identity_sha256": throttle._canonical_sha256(identity),
        "rows": rows,
        "complete": True,
        "completed_rows": 72,
    }
    progress_path.write_text(json.dumps(progress))
    receipt = {
        "schema": throttle.P5_GRID_SCHEMA,
        "source_bindings": bindings,
        "source_bindings_sha256": bindings_sha,
        "claim_boundary": {
            "mechanics_only": True,
            "moves_no_category": True,
            "naive_formula_is_diagnostic_only": True,
        },
        "cited_boundary_trace": {
            "path": "proof/P5_MEMORY_BOUNDARY_TRACE.json",
            "sha256": boundary_sha,
        },
        "atomic_progress": {
            "path": "proof/P5_TRAINGRID_MEMORY_TRACE.json.progress.json",
            "sha256": throttle._sha256_file(progress_path),
            "identity_sha256": throttle._canonical_sha256(identity),
            "completed_rows": 72,
        },
        "config": {
            "cells": config["cells"],
            "batch_rows": [4, 1],
            "repeats": 3,
            "seed": 0,
            "mask_ratio": 0.5,
            "ema_decay": 0.99,
            "child_memory_guard_gb": 12.0,
            "device": "cpu",
        },
        "cells": list(rows.values()),
        "all_ok": True,
    }
    _write_payload_receipt(root / "proof/P5_TRAINGRID_MEMORY_TRACE.json", receipt)
    _seal_governor_run(root, "p5_traingrid_memory_probe_cpu")
    return receipt


def _write_p5_ancestors(root, *, pilot_outcome="null"):
    _write_p5_screen(root, profile="p5smoke", outcome="null")
    _write_p5_grid(root)
    return _write_p5_screen(root, profile="p5pilot", outcome=pilot_outcome)


def test_policy_pins_five_hour_envelope_and_exact_p4_resume_command():
    policy = load_policy()
    cpu_task = policy.task("p4_resume_cpu")
    mps_task = policy.task("p4_resume_mps")
    assert policy.limits["hard_wall_minutes"] == 300
    assert policy.limits["disk_floor_gb"] == 40.0
    assert cpu_task.wall_minutes == 300
    assert cpu_task.restart_exit_codes == (2,)
    assert cpu_task.command == (
        ".venv/bin/python",
        "scripts/p4_capability_density.py",
        "--profile",
        "p4screen",
        "--device",
        "cpu",
        "--run-dir",
        "runs/p4_screen/p4screen",
        "--out",
        "proof/P4_CAPABILITY_DENSITY_SCREEN.json",
    )
    assert mps_task.command == (
        ".venv/bin/python",
        "scripts/p4_capability_density.py",
        "--profile",
        "p4screen",
        "--device",
        "mps",
        "--run-dir",
        "runs/p4_screen/p4screen_mps_clean",
        "--out",
        "proof/P4_CAPABILITY_DENSITY_SCREEN_MPS_CLEAN.json",
    )
    p4 = yaml.safe_load((REPO_ROOT / "configs/experiment/mop_p4_capability_density_screen.yaml").read_text())
    assert p4["profiles"]["p4screen"]["wall_budget_seconds"] == 10800.0


def test_policy_pins_cpu_p5_order_and_exact_commands():
    policy = load_policy()
    assert "p5_context_fresh_challenge.py" in policy.monitor["known_heavy_markers"]
    assert policy.execution_order["p5_cpu"] == (
        "p5smoke_cpu",
        "p5_traingrid_memory_probe_cpu",
        "p5pilot_cpu",
        "p5fresh_challenge_cpu",
        "p5verify_cpu",
    )
    assert policy.execution_order["p5_pilot_null_cpu"] == (
        "p5smoke_cpu",
        "p5_traingrid_memory_probe_cpu",
        "p5pilot_cpu",
        "p5verify_pilot_null_cpu",
    )
    assert policy.task("p5smoke_cpu").depends_on == ()
    assert policy.task("p5_traingrid_memory_probe_cpu").depends_on == ("p5smoke_cpu",)
    assert policy.task("p5pilot_cpu").depends_on == ("p5_traingrid_memory_probe_cpu",)
    assert policy.task("p5fresh_challenge_cpu").depends_on == ("p5pilot_cpu",)
    assert policy.task("p5verify_cpu").depends_on == ("p5fresh_challenge_cpu",)
    assert policy.task("p5verify_pilot_null_cpu").depends_on == ("p5pilot_cpu",)
    assert (
        "runs/p5_context/p5smoke/frames/f*/cell_receipt.json" in policy.task("p5smoke_cpu").checkpoint_globs
    )
    assert (
        "runs/p5_context/p5pilot/frames/f*/cell_receipt.json" in policy.task("p5pilot_cpu").checkpoint_globs
    )
    assert policy.task("p5smoke_cpu").command == (
        ".venv/bin/python",
        "scripts/p5_context_capability.py",
        "--profile",
        "p5smoke",
        "--device",
        "cpu",
        "--run-dir",
        "runs/p5_context/p5smoke",
        "--out",
        "proof/P5_CONTEXT_CAPABILITY_SMOKE.json",
    )
    assert policy.task("p5_traingrid_memory_probe_cpu").command == (
        ".venv/bin/python",
        "scripts/p5_traingrid_memory_probe.py",
        "--out",
        "proof/P5_TRAINGRID_MEMORY_TRACE.json",
        "--repeats",
        "3",
        "--seed",
        "0",
    )
    assert policy.task("p5pilot_cpu").command == (
        ".venv/bin/python",
        "scripts/p5_context_capability.py",
        "--profile",
        "p5pilot",
        "--device",
        "cpu",
        "--run-dir",
        "runs/p5_context/p5pilot",
        "--out",
        "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
    )
    assert policy.task("p5fresh_challenge_cpu").command == (
        ".venv/bin/python",
        "scripts/p5_context_fresh_challenge.py",
        "--primary",
        "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
        "--primary-run-dir",
        "runs/p5_context/p5pilot",
        "--run-dir",
        "runs/p5_context/fresh_challenge",
        "--out",
        "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json",
        "--device",
        "cpu",
    )
    assert policy.task("p5verify_cpu").command[-2:] == (
        "--out",
        "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json",
    )
    assert "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json" in {
        requirement.path for requirement in policy.task("p5verify_cpu").prerequisites
    }
    assert policy.task("p5verify_pilot_null_cpu").command[-4:] == (
        "--primary-run-dir",
        "runs/p5_context/p5pilot",
        "--out",
        "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json",
    )


def test_task_output_path_accepts_all_three_authority_flags_and_rejects_multiplicity():
    base = load_policy().task("p5verify_cpu")
    for flag in throttle.OUTPUT_AUTHORITY_FLAGS:
        task = replace(base, command=("runner", flag, "proof/output.json"))
        assert throttle._task_output_path(task) == "proof/output.json"

    with pytest.raises(ThrottleRefused, match="exactly one output-authority"):
        throttle._task_output_path(
            replace(
                base,
                command=(
                    "runner",
                    "--out",
                    "proof/one.json",
                    "--verification-out",
                    "proof/two.json",
                ),
            )
        )
    with pytest.raises(ThrottleRefused, match="repository-relative"):
        throttle._task_output_path(replace(base, command=("runner", "--output", "../escape.json")))


def test_substrate_task_prefixes_require_completion_provenance():
    base = load_policy().task("p4_resume_cpu")
    for task_id in (
        "edcm1_official_cpu",
        "edcm1_verify_cpu",
        "escs_x0_official_cpu",
        "escs_x0_verify_cpu",
    ):
        assert throttle._requires_completion_provenance(replace(base, task_id=task_id)) is True
    assert throttle._requires_completion_provenance(base) is False


def test_policy_load_fails_closed_on_task_policy_helper_drift(monkeypatch):
    monkeypatch.setattr(throttle, "TASK_POLICY_HELPER_SHA256", "0" * 64)
    with pytest.raises(ThrottleRefused, match="helper implementation drifted"):
        load_policy()


def test_legacy_baseline_loader_rejects_a_self_consistent_splice(monkeypatch, tmp_path):
    source_path, expected_manifest, expected_governor = throttle.LEGACY_POLICY_BASELINE_BINDINGS[1]
    payload = json.loads(source_path.read_text())
    authority = payload["task_authorities"][0]
    authority["task_sha256"] = "0" * 64
    authority.pop("authority_sha256")
    authority["authority_sha256"] = throttle.task_policy.canonical_sha256(authority)
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = throttle.task_policy.canonical_sha256(payload)
    spliced = tmp_path / "spliced-baseline.json"
    spliced.write_text(json.dumps(payload))
    monkeypatch.setattr(
        throttle,
        "LEGACY_POLICY_BASELINE_BINDINGS",
        ((spliced, expected_manifest, expected_governor),),
    )

    with pytest.raises(ThrottleRefused, match="binding drifted"):
        throttle._load_legacy_policy_baselines()


def test_native_substrate_seals_fail_closed_on_payload_mutation(tmp_path):
    cases = (
        (
            throttle.ESCS_PREFLIGHT_SCHEMA,
            _native_seal(
                {
                    "schema": throttle.ESCS_PREFLIGHT_SCHEMA,
                    "scaffold_ready": True,
                    "scientific_promotion_allowed": False,
                },
                "report_sha256",
            ),
        ),
        (throttle.EDCM_RECEIPT_SCHEMA, _edcm_native_producer()),
        (throttle.X0_RECEIPT_SCHEMA, _x0_native_producer()),
    )
    for schema, payload in cases:
        assert throttle._native_schema_authority_problems(schema, payload, tmp_path) == []
        mutated = json.loads(json.dumps(payload))
        mutated["post_seal_mutation"] = True
        problems = throttle._native_schema_authority_problems(schema, mutated, tmp_path)
        assert any("drift" in problem for problem in problems)


def test_edcm_native_verifier_exactly_joins_live_producer(tmp_path):
    producer_path = tmp_path / throttle.EDCM_RECEIPT_PATH
    producer = _edcm_native_producer()
    _write_native_payload(producer_path, producer)
    producer_file = throttle._scoped_file_receipt(producer_path, tmp_path)
    verification = {
        "authority_sha256": producer["authority_sha256"],
        "implementation_authority_sha256": producer["implementation_authority_sha256"],
        "execution_status": producer["execution_status"],
        "verdict": producer["aggregate"]["verdict"],
        "verified_sources": {
            "receipt": producer_file,
            "receipt_path": throttle.EDCM_RECEIPT_PATH,
        },
        "scientific_promotion": False,
    }
    artifact = _native_seal(
        {
            "schema": throttle.EDCM_VERIFICATION_SCHEMA,
            "verification": verification,
            "scientific_promotion": False,
        },
        "verification_artifact_sha256",
    )
    assert (
        throttle._native_schema_authority_problems(
            throttle.EDCM_VERIFICATION_SCHEMA,
            artifact,
            tmp_path,
        )
        == []
    )

    spliced = json.loads(json.dumps(artifact))
    spliced["verification"]["verified_sources"]["receipt"]["sha256"] = "0" * 64
    _native_seal(spliced, "verification_artifact_sha256")
    problems = throttle._native_schema_authority_problems(
        throttle.EDCM_VERIFICATION_SCHEMA,
        spliced,
        tmp_path,
    )
    assert "EDCM verifier producer file join drift" in problems


def test_x0_native_verifier_exactly_joins_live_producer(tmp_path):
    producer_path = tmp_path / throttle.X0_RECEIPT_PATH
    producer = _x0_native_producer()
    _write_native_payload(producer_path, producer)
    artifact = _native_seal(
        {
            "schema": throttle.X0_VERIFICATION_SCHEMA,
            "producer_receipt": throttle._scoped_file_receipt(producer_path, tmp_path),
            "producer_receipt_sha256": producer["receipt_sha256"],
            "implementation_authority_sha256": producer["implementation_authority"]["manifest_sha256"],
            "primary_aggregate": producer["aggregate"],
            "scientific_promotion": False,
        },
        "verification_sha256",
    )
    assert (
        throttle._native_schema_authority_problems(
            throttle.X0_VERIFICATION_SCHEMA,
            artifact,
            tmp_path,
        )
        == []
    )

    spliced = json.loads(json.dumps(artifact))
    spliced["producer_receipt_sha256"] = "0" * 64
    _native_seal(spliced, "verification_sha256")
    problems = throttle._native_schema_authority_problems(
        throttle.X0_VERIFICATION_SCHEMA,
        spliced,
        tmp_path,
    )
    assert "X0 verifier producer receipt seal join drift" in problems


def test_governor_provenance_rejects_ambiguous_output_producers(tmp_path):
    policy = load_policy()
    producer = policy.task("p6_10k_resource_probe_cpu")
    duplicate = replace(producer, task_id="p6_duplicate_resource_probe_cpu")
    ambiguous = replace(policy, tasks={**policy.tasks, duplicate.task_id: duplicate})

    report = throttle._governor_provenance_report(
        "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
        {"schema": throttle.P6_RUNG_SCHEMA},
        ambiguous,
        tmp_path,
    )

    assert report["all_ok"] is False
    assert report["producer_task_id"] is None
    assert "maps to 2 governed producer tasks" in report["problems"][0]


def test_p5_order_fails_closed_on_missing_or_tampered_receipts(tmp_path):
    policy = load_policy()
    grid = policy.task("p5_traingrid_memory_probe_cpu")
    pilot = policy.task("p5pilot_cpu")
    assert evaluate_task(grid, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    smoke_path = tmp_path / "proof/P5_CONTEXT_CAPABILITY_SMOKE.json"
    _write_p5_screen(tmp_path, profile="p5smoke", outcome="null")
    assert evaluate_task(grid, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is True
    assert evaluate_task(pilot, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    _write_p5_grid(tmp_path)
    assert evaluate_task(pilot, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is True

    tampered = json.loads(smoke_path.read_text())
    tampered["profile"] = "p5pilot"
    smoke_path.write_text(json.dumps(tampered))
    decision = evaluate_task(grid, _snapshot(), policy, evidence_root=tmp_path)
    assert decision["allowed"] is False
    report = next(gate for gate in decision["gates"] if gate["name"] == "receipt_prerequisites")
    assert any("payload digest drift" in problem for problem in report["observed"][0]["problems"])


def test_p5_pilot_null_verifier_branch_excludes_favorable_verifier(tmp_path):
    policy = load_policy()
    _write_p5_ancestors(tmp_path, pilot_outcome="null")

    null_verifier = policy.task("p5verify_pilot_null_cpu")
    favorable_verifier = policy.task("p5verify_cpu")
    assert evaluate_task(null_verifier, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is True
    decision = evaluate_task(favorable_verifier, _snapshot(), policy, evidence_root=tmp_path)
    assert decision["allowed"] is False
    gate = next(row for row in decision["gates"] if row["name"] == "receipt_prerequisites")
    challenge = next(
        row for row in gate["observed"] if row["path"] == "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json"
    )
    assert "receipt is missing" in challenge["problems"]


def test_p5_favorable_verifier_policy_cannot_drop_challenge_provenance(tmp_path):
    raw = yaml.safe_load((REPO_ROOT / "configs/local_execution_throttle.yaml").read_text())
    prerequisites = raw["tasks"]["p5verify_cpu"]["prerequisites"]
    raw["tasks"]["p5verify_cpu"]["prerequisites"] = [
        row for row in prerequisites if row["path"] != "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json"
    ]
    path = tmp_path / "weakened-p5-challenge.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ThrottleRefused, match="exact P5 completion prerequisites drifted"):
        load_policy(path)


def test_execute_refusal_preserves_run_id_and_returns_durable_receipt(monkeypatch, tmp_path):
    policy = load_policy()
    denied = {
        "schema": "mop-local-throttle-receipt/v1",
        "admission": {"allowed": False},
        "active_lanes": [],
        "command_executed": False,
    }
    monkeypatch.setattr(throttle, "dry_run_decision", lambda *_args, **_kwargs: dict(denied))
    receipt = throttle.run_task(
        policy.task("p5smoke_cpu"),
        policy,
        run_id="p5-denied-test",
        state_root=tmp_path,
        disk_root=tmp_path,
    )
    assert receipt["mode"] == "execute-refused"
    assert receipt["status"] == "admission-refused"
    assert receipt["run_id"] == "p5-denied-test"
    assert receipt["command_executed"] is False
    persisted = json.loads((tmp_path / "p5-denied-test/run_receipt.json").read_text())
    assert persisted == receipt


def test_policy_pins_p6_progressive_order_dependencies_and_exact_commands():
    policy = load_policy()
    order = policy.execution_order["p6_cpu"]
    assert order == (
        "p6_10k_resource_probe_cpu",
        "p6_10k_replication_cpu",
        "p6_10k_verify_cpu",
        "p6_100k_replication_cpu",
        "p6_100k_verify_cpu",
        "p6_1m_replication_cpu",
        "p6_1m_verify_cpu",
    )
    assert policy.task(order[0]).depends_on == ()
    assert policy.task(order[1]).depends_on == (order[0],)
    assert policy.task(order[2]).depends_on == (order[1],)
    assert policy.task(order[3]).depends_on == (order[2],)
    assert policy.task(order[4]).depends_on == (order[3],)
    assert policy.task(order[5]).depends_on == (order[4],)
    assert policy.task(order[6]).depends_on == (order[5],)
    probe = policy.task(order[0])
    assert probe.resource_probe is True
    assert probe.requires_empty_lanes is True
    assert probe.estimated_unified_memory_gb is None
    assert probe.command == (
        ".venv/bin/python",
        "scripts/continual_million_event_rung.py",
        "--config",
        "configs/experiment/continual_million_event_rungs.yaml",
        "--rung",
        "10000",
        "--work-root",
        "runs/continual_million_event/rung_010000_probe",
        "--out",
        "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
        "--resource-probe",
        "--seed-count",
        "1",
        "--schedules",
        "abrupt",
        "--arms",
        "replay",
    )
    assert policy.task("p6_10k_verify_cpu").command == (
        ".venv/bin/python",
        "scripts/verify_continual_million_event_rung.py",
        "--source",
        "proof/P6_CONTINUAL_10K.json",
        "--out",
        "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
    )
    assert policy.task(order[5]).command[-5:] == (
        "1000000",
        "--work-root",
        "runs/continual_million_event/rung_1000000",
        "--out",
        "proof/P6_CONTINUAL_1M.json",
    )


def _write_p6_fabricated_receipt(path, *, rung, mode="replication", replication=True, rss=200_000_000):
    root = path.parents[1]
    config = yaml.safe_load(throttle.P6_RUN_CONFIG.read_text())
    evidence = throttle._p6_resource_evidence()
    preflight = json.loads(throttle.P6_PREFLIGHT.read_text())
    source_authority = throttle._p6_source_live_binding_authority(preflight)
    plan = throttle._p6_expected_plan(config, rung=rung, mode=mode)
    identity = {
        "config_sha256": evidence["config_sha256"],
        "runner_sha256": throttle._sha256_file(REPO_ROOT / "scripts/continual_million_event_rung.py"),
        "source_preflight_file_sha256": evidence["preflight_file_sha256"],
        "source_preflight_payload_sha256": evidence["preflight_payload_sha256"],
        "source_live_bindings_sha256": evidence["preflight_live_bindings_sha256"],
        "plan": plan,
        "claim_scope": throttle.P6_CLAIM_SCOPE,
    }
    work_root = root / "runs/test_p6" / path.stem
    cells = {}
    for cell in plan["cells"]:
        seed = cell["seed"]
        schedule = cell["schedule"]
        arm = cell["arm"]
        key = f"seed_{seed}/{schedule}/{arm}"
        stream_identity = throttle._canonical_sha256({"seed": seed, "schedule": schedule, "rung": rung})
        stream_sha = throttle._canonical_sha256({"stream": stream_identity})
        state = {"next_sequence": rung, "total": rung, "updates": rung * 2, "arm": arm}
        state_sha = throttle._canonical_sha256(state)
        controls = {
            "replay_enabled": arm == "replay",
            "fresh_init_on_transition": arm == "fresh-init",
            "matched_updates_per_event": 2,
            "actual_updates_per_event": 2.0,
            "fixed_topology": True,
            "reset_count": 3 if arm == "fresh-init" else 0,
        }
        metrics = {
            "retention": {"domain_zero_final_accuracy": 0.75},
            "acquisition": {"stream_accuracy": 0.75},
            "future_learnability": {"first_window_accuracy": 0.75},
            "stale_memory": {"harm_rate": 0.0},
            "deletion": {"complete": True},
            "resources": {
                "events_processed": rung,
                "updates": rung * 2,
                "updates_per_event": 2.0,
                "checkpoint_state_bytes": 22_612,
                "stream_disk_bytes": max(19_584, rung),
                "model_weights_loaded": False,
                "accelerator_required": False,
            },
        }
        checkpoint_identity = {
            "stream_identity_sha256": stream_identity,
            "stream_sha256": stream_sha,
            "arm": arm,
            "profile": plan["profile"],
            "claim_scope": throttle.P6_CLAIM_SCOPE,
        }
        checkpoint = {
            "schema": "mop-continual-smoke-checkpoint/v1",
            "identity": checkpoint_identity,
            "identity_sha256": throttle._canonical_sha256(checkpoint_identity),
            "state": state,
            "state_sha256": state_sha,
            "complete": True,
            "result": {
                "all_mechanics_ok": True,
                "metrics": metrics,
                "controls": controls,
            },
        }
        checkpoint_path = work_root / "checkpoints" / f"seed_{seed}" / schedule / f"{arm}.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True))
        cells[key] = {
            "seed": seed,
            "schedule": schedule,
            "arm": arm,
            "stream_identity_sha256": stream_identity,
            "stream_sha256": stream_sha,
            "checkpoint_sha256": throttle._sha256_file(checkpoint_path),
            "state_sha256": state_sha,
            "metrics": metrics,
            "controls": controls,
            "all_mechanics_ok": True,
            "resumed_from_atomic_checkpoint": False,
        }
    progress = {
        "schema": "mop-continual-progressive-rung-progress/v1",
        "identity": identity,
        "identity_sha256": throttle._canonical_sha256(identity),
        "cells": cells,
        "complete": True,
    }
    progress_path = work_root / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(progress, sort_keys=True))
    payload = {
        "schema": throttle.P6_RUNG_SCHEMA,
        "claim_scope": throttle.P6_CLAIM_SCOPE,
        "identity": identity,
        "identity_sha256": throttle._canonical_sha256(identity),
        "source_live_authority": source_authority,
        "mode": mode,
        "rung": rung,
        "plan": plan,
        "progress": {
            "path": str(progress_path.relative_to(root)),
            "sha256": throttle._sha256_file(progress_path),
            "resumed_existing_progress": False,
            "completed_cells": len(cells),
            "expected_cells": len(cells),
        },
        "cells": cells,
        "resource_measurement": {
            "max_rss_bytes": rss,
            "measured_after_complete": True,
            "events_per_stream": rung,
        },
        "all_mechanics_ok": True,
        "replication_execution_complete": replication,
        "independent_metric_verifier_complete": False,
        "scientific_promotion": False,
    }
    _write_payload_receipt(path, payload)
    task_id = (
        "p6_10k_resource_probe_cpu"
        if mode == "resource-probe"
        else {
            10_000: "p6_10k_replication_cpu",
            100_000: "p6_100k_replication_cpu",
            1_000_000: "p6_1m_replication_cpu",
        }[rung]
    )
    _seal_governor_run(root, task_id)
    return payload


_P6_CACHE_DIRECTORY = tempfile.TemporaryDirectory(prefix="mop-local-throttle-p6-")
_P6_CACHE_ROOT = Path(_P6_CACHE_DIRECTORY.name).resolve()
_P6_RECEIPT_CACHE: dict[tuple[int, str], Path] = {}


def _copy_p6_dependencies(root: Path) -> None:
    preflight = json.loads(throttle.P6_PREFLIGHT.read_text())
    relatives = set(throttle.P6_VERIFIER_IMPLEMENTATION_PATHS)
    relatives.add(str(preflight["config"]["path"]))
    relatives.add(str(preflight["wave_e0"]["path"]))
    relatives.update(str(row["path"]) for row in preflight["implementation"])
    for relative in sorted(relatives):
        source = REPO_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _cached_p6_receipt(rung: int, mode: str) -> Path:
    key = (rung, mode)
    cached = _P6_RECEIPT_CACHE.get(key)
    if cached is not None:
        return cached
    root = (_P6_CACHE_ROOT / f"{mode}-{rung}").resolve()
    _copy_p6_dependencies(root)
    if mode == "resource-probe":
        output = root / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
        work_root = root / "runs/continual_million_event/rung_010000_probe"
        options = {
            "resource_probe": True,
            "seed_count": 1,
            "schedules": ("abrupt",),
            "arms": ("replay",),
        }
    else:
        output_name = {
            10_000: "P6_CONTINUAL_10K.json",
            100_000: "P6_CONTINUAL_100K.json",
            1_000_000: "P6_CONTINUAL_1M.json",
        }[rung]
        output = root / "proof" / output_name
        work_root = root / f"runs/continual_million_event/rung_{rung:07d}"
        options = {}
    original_root = p6_runner.REPO_ROOT
    try:
        p6_runner.REPO_ROOT = root
        p6_runner.run_rung(
            root / "configs/experiment/continual_million_event_rungs.yaml",
            work_root,
            output,
            rung=rung,
            **options,
        )
    finally:
        p6_runner.REPO_ROOT = original_root
    _P6_RECEIPT_CACHE[key] = root
    return root


def _write_p6_receipt(
    path: Path,
    *,
    rung: int,
    mode: str = "replication",
    replication: bool = True,
    rss: int = 200_000_000,
):
    root = path.parents[1]
    cached_root = _cached_p6_receipt(rung, mode)
    shutil.copytree(cached_root, root, dirs_exist_ok=True, copy_function=shutil.copy2)
    payload = json.loads(path.read_text())
    payload["resource_measurement"]["max_rss_bytes"] = rss
    payload["resource_measurement"]["current_invocation_max_rss_bytes"] = rss
    payload["resource_measurement"]["work_root"] = str((root / payload["progress"]["path"]).parent)
    payload["replication_execution_complete"] = replication
    _write_payload_receipt(path, payload)
    task_id = (
        "p6_10k_resource_probe_cpu"
        if mode == "resource-probe"
        else {
            10_000: "p6_10k_replication_cpu",
            100_000: "p6_100k_replication_cpu",
            1_000_000: "p6_1m_replication_cpu",
        }[rung]
    )
    _seal_governor_run(root, task_id)
    return payload


def _p5_fixture_artifact_evidence(root: Path, profile: str, primary: dict) -> dict:
    result = {}
    for frames in throttle.P5_FRAME_COUNTS:
        frame_dir = root / "runs/p5_context" / profile / "frames" / f"f{frames}"
        cell = json.loads((frame_dir / "cell_receipt.json").read_text())
        by_seed = {}
        for seed in cell["expected_seeds"]:
            seed_dir = frame_dir / f"seed_{seed}"
            seed_payload = cell["seed_results"][str(seed)]
            by_seed[str(seed)] = {
                "seed_result": {
                    "path": str((seed_dir / "seed_result.json").relative_to(root)),
                    "sha256": throttle._sha256_file(seed_dir / "seed_result.json"),
                },
                "arms": {
                    mechanism: {
                        "arm_receipt": {
                            "path": str((seed_dir / mechanism / "arm_receipt.json").relative_to(root)),
                            "sha256": throttle._sha256_file(seed_dir / mechanism / "arm_receipt.json"),
                        },
                        "checkpoint": {
                            "path": str((seed_dir / mechanism / "checkpoint.pt").relative_to(root)),
                            "sha256": throttle._sha256_file(seed_dir / mechanism / "checkpoint.pt"),
                            "model_state_sha256": seed_payload["mechanisms"][mechanism]["training"][
                                "final_state_sha256"
                            ],
                            "target_state_sha256": json.loads(
                                (seed_dir / mechanism / "arm_receipt.json").read_text()
                            )["target_state_sha256"],
                        },
                    }
                    for mechanism in throttle.P5_MECHANISMS
                },
            }
        result[f"f{frames}"] = by_seed
    return result


def _write_p5_verifier(path, *, profile="p5pilot"):
    root = path.parents[1]
    if profile == "p5smoke":
        primary = _write_p5_screen(root, profile="p5smoke", outcome="terminal-null")
        primary_name = "P5_CONTEXT_CAPABILITY_SMOKE.json"
    else:
        primary = _write_p5_ancestors(root)
        primary_name = "P5_CONTEXT_CAPABILITY_PILOT.json"
    primary_path = root / "proof" / primary_name
    run_receipt = root / "runs/p5_context" / profile / "p5_context_receipt.json"
    config = throttle._p5_resolved_config(profile)
    primary_artifacts = _p5_fixture_artifact_evidence(root, profile, primary)
    payload = {
        "schema": throttle.P5_VERIFIER_SCHEMA,
        "claim_scope": throttle.P5_CLAIM_SCOPE,
        "evidence_class": throttle.P5_EVIDENCE_CLASS,
        "source_bindings": throttle._p5_live_bindings(throttle.P5_VERIFIER_SOURCE_PATHS),
        "verification_complete": True,
        "all_ok": True,
        "prerequisite_ready": True,
        "problems": [],
        "all_controls_passed": True,
        "all_mutations_rejected": True,
        "controls": {
            "same_initialization_frozen_control": True,
            "matched_parameter_and_flop_contract": True,
            "difficulty_calibration_checked": True,
            "primary_off_ceiling": {"f64": True, "f32": True},
            "nonterminal_outcome_has_off_ceiling_multiunit_support": True,
            "seed_arm_checkpoint_artifacts_exactly_joined": True,
            "raw_per_seed_contrasts_independently_recomputed": True,
            "fresh_disjoint_training_for_every_primary_pattern": True,
            "threshold_tie_is_null": True,
            "confirmatory_promotion_refused": True,
        },
        "outcome_contract": {
            "allowed": ["mechanics", "null", "favorable-programmatic-only"],
            "tie_is_null": True,
            "programmatic_only": True,
            "confirmatory_promotable": False,
            "scientific_capability_claim": False,
        },
        "classification": "null",
        "outcome": "null",
        "primary_outcome": "null",
        "primary_profile": profile,
        "terminal_null": profile == "p5smoke",
        "independence": {
            "imports_p5_training_or_evaluator": False,
            "raw_seed_score_recompute": True,
            "checkpoint_files_opened_with_weights_only": True,
            "checkpoint_model_and_target_state_hashes_recomputed": True,
            "heldout_metrics_reexecuted_from_checkpoint": False,
            "fresh_training_required_for_each_primary_pattern": True,
            "fresh_training_seeds": list(throttle.P5_FRESH_SEEDS),
            "fresh_seeds_disjoint_from_primary": True,
        },
        "artifact_evidence": {
            "primary": primary_artifacts,
            "fresh_challenge": None,
        },
        "cell_receipt_evidence": {
            "primary": {
                f"f{frames}": {
                    "path": (f"runs/p5_context/{profile}/frames/f{frames}/cell_receipt.json"),
                    "sha256": throttle._sha256_file(
                        root / f"runs/p5_context/{profile}/frames/f{frames}/cell_receipt.json"
                    ),
                }
                for frames in throttle.P5_FRAME_COUNTS
            },
            "fresh_challenge": None,
        },
        "metric_recomputation_limit": (
            "checkpoint model and target states, identities, completed steps, and compute are "
            "independently hashed and joined; heldout scores are recomputed from durable per-seed "
            "receipts but are not re-evaluated from model checkpoints"
        ),
        "primary_receipt": {
            "path": f"proof/{primary_name}",
            "sha256": throttle._sha256_file(primary_path),
            "payload_sha256": primary["payload_sha256"],
        },
        "primary_run_receipt": {
            "path": f"runs/p5_context/{profile}/p5_context_receipt.json",
            "sha256": throttle._sha256_file(run_receipt),
            "exactly_matches_published": True,
        },
        "config": {
            "path": "configs/experiment/mop_p5_context_capability.yaml",
            "sha256": throttle._sha256_file(throttle.P5_CONFIG_PATH),
            "resolved_sha256": throttle._canonical_sha256(config),
        },
        "fresh_challenge_required": False,
        "primary_patterns": [],
        "fresh_challenge": None,
        "verified_patterns": [],
        "mutation_tests": [
            {"id": mutation, "rejected": True} for mutation in sorted(throttle.P5_BASE_MUTATION_IDS)
        ],
        "promotion": {
            "confirmatory_promotable": False,
            "refused_by_construction": True,
            "scientific_capability_claim": False,
        },
        "scientific_promotion": False,
    }
    _write_payload_receipt(path, payload)
    _seal_governor_run(
        root,
        "p5verify_smoke_null_cpu" if profile == "p5smoke" else "p5verify_pilot_null_cpu",
    )
    return payload


def _write_p6_fabricated_verifier(path, *, source_path, next_rung, next_allowed=True, favorable=True):
    root = path.parents[1]
    source = json.loads(source_path.read_text())
    source_rung = source["rung"]
    seeds = source["plan"]["seeds"]
    pairs = [
        {
            "seed": seed,
            "retention_delta": 0.1 if favorable else 0.0,
            "future_first_window_delta": 0.1 if favorable else 0.0,
            "tie_is_null": not favorable,
        }
        for seed in seeds
    ]
    contrasts = [
        {
            "schedule": schedule,
            "control": control,
            "independent_units": len(seeds),
            "paired_seed_deltas": pairs,
            "retention_mean_delta": 0.1 if favorable else 0.0,
            "future_first_window_mean_delta": 0.1 if favorable else 0.0,
            "aggregate_tie_is_null": not favorable,
            "strict_joint_gain": favorable,
        }
        for schedule in ("abrupt", "gradual")
        for control in ("no-replay", "fresh-init")
    ]
    decision = {
        "primary_endpoints": [
            "retention.domain_zero_final_accuracy",
            "future_learnability.first_window_accuracy",
        ],
        "independent_unit": "seed within transition schedule",
        "controls": ["no-replay", "fresh-init"],
        "tie_rule": "an exact zero on either paired mean endpoint is a null",
        "contrasts": contrasts,
        "aggregate_tie_count": 0 if favorable else 4,
        "strict_joint_gain_all_schedules_and_controls": favorable,
        "verdict": "favorable-rung-pattern" if favorable else "null",
        "null_supported": not favorable,
        "scientific_promotion": False,
        "claim_boundary": "programmatic rung only",
    }
    mutation_rows = [
        {"mutation": f"mutation-{index}", "rejected": True, "problems": ["rejected"]} for index in range(12)
    ]
    identity = source["identity"]
    progress = source["progress"]
    payload = {
        "schema": throttle.P6_VERIFIER_SCHEMA,
        "claim_scope": throttle.P6_CLAIM_SCOPE,
        "source_rung": {
            "path": str(source_path.relative_to(root)),
            "file_sha256": throttle._sha256_file(source_path),
            "payload_sha256": source["payload_sha256"],
            "identity_sha256": source["identity_sha256"],
            "rung": source_rung,
            "mode": "replication",
        },
        "live_dependencies": {
            field: identity[field]
            for field in (
                "config_sha256",
                "runner_sha256",
                "source_preflight_file_sha256",
                "source_preflight_payload_sha256",
                "source_live_bindings_sha256",
            )
        },
        "progress_authority": {
            "path": progress["path"],
            "file_sha256": progress["sha256"],
            "identity_sha256": source["identity_sha256"],
            "complete": True,
            "cell_count": 30,
        },
        "independent_recompute": {
            "cell_count": 30,
            "metric_families": [
                "retention",
                "acquisition",
                "future_learnability",
                "stale_memory",
                "deletion",
                "resources",
            ],
            "checkpoint_state_recomputed": True,
            "controls_recomputed": True,
            "paired_metrics_recomputed": True,
            "decision": decision,
        },
        "mutation_suite": {
            "count": 12,
            "rejected": 12,
            "all_rejected": True,
            "mutations": mutation_rows,
        },
        "checks": {
            "source_payload_self_hash": True,
            "live_dependencies_current": True,
            "progress_and_checkpoints_current": True,
            "full_replication_structure_valid": True,
            "all_metrics_independently_recomputed": True,
            "all_controls_present_and_valid": True,
            "tie_is_null": True,
            "all_mutations_rejected": True,
            "scientific_promotion_blocked": True,
        },
        "verification_complete": True,
        "errors": [],
        "prerequisite": {
            "source_rung": source_rung,
            "source_rung_file_sha256": throttle._sha256_file(source_path),
            "source_identity_sha256": source["identity_sha256"],
            "verification_complete": True,
            "valid_controls": True,
            "tie_is_null": True,
            "mutation_suite_all_rejected": True,
            "next_rung": next_rung,
            "next_rung_allowed": next_allowed,
            "next_rung_reason": (
                "strict favorable programmatic pattern requires the next scale confirmation"
                if next_allowed
                else "verified tie, null, invalid evidence, or final rung does not admit scaling"
            ),
        },
        "scientific_promotion": False,
        "implementation": [
            {"path": relative, "sha256": throttle._sha256_file(REPO_ROOT / relative)}
            for relative in throttle.P6_VERIFIER_IMPLEMENTATION_PATHS
        ],
    }
    _write_payload_receipt(path, payload)
    task_id = {
        10_000: "p6_10k_verify_cpu",
        100_000: "p6_100k_verify_cpu",
        1_000_000: "p6_1m_verify_cpu",
    }[source_rung]
    _seal_governor_run(root, task_id)
    return payload


def _write_p6_verifier(
    path: Path,
    *,
    source_path: Path,
    next_rung: int | None,
    next_allowed: bool | None = None,
    favorable: bool | None = None,
):
    root = path.parents[1]
    payload = throttle.build_p6_verification_receipt(source_path, repo_root=root)
    assert payload["prerequisite"]["next_rung"] == next_rung
    if favorable is not None or next_allowed is not None:
        decision = payload["independent_recompute"]["decision"]
        desired_favorable = (
            decision["strict_joint_gain_all_schedules_and_controls"] if favorable is None else favorable
        )
        desired_next = payload["prerequisite"]["next_rung_allowed"] if next_allowed is None else next_allowed
        decision["aggregate_tie_count"] = 4
        decision["strict_joint_gain_all_schedules_and_controls"] = desired_favorable
        decision["verdict"] = "favorable-rung-pattern" if desired_favorable else "null"
        decision["null_supported"] = not desired_favorable
        payload["prerequisite"]["next_rung_allowed"] = desired_next
        payload["prerequisite"]["next_rung_reason"] = (
            "strict favorable programmatic pattern requires the next scale confirmation"
            if desired_next
            else "verified tie, null, invalid evidence, or final rung does not admit scaling"
        )
    _write_payload_receipt(path, payload)
    task_id = {
        10_000: "p6_10k_verify_cpu",
        100_000: "p6_100k_verify_cpu",
        1_000_000: "p6_1m_verify_cpu",
    }[payload["source_rung"]["rung"]]
    _seal_governor_run(root, task_id)
    return payload


def test_p6_probe_is_exclusive_and_never_overlaps_p4():
    policy = load_policy()
    probe = policy.task("p6_10k_resource_probe_cpu")
    p4 = policy.task("p4_resume_cpu")
    decision = evaluate_task(probe, _snapshot(), policy, active=[_active(p4)])
    assert decision["allowed"] is False
    exclusive = next(gate for gate in decision["gates"] if gate["name"] == "exclusive_lane")
    assert exclusive["ok"] is False


def test_p6_probe_requires_sealed_final_p5_verification(tmp_path):
    policy = load_policy()
    probe = policy.task("p6_10k_resource_probe_cpu")
    assert evaluate_task(probe, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    verifier = tmp_path / "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
    _write_p5_verifier(verifier)
    assert evaluate_task(probe, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is True

    tampered = json.loads(verifier.read_text())
    tampered["scientific_promotion"] = True
    verifier.write_text(json.dumps(tampered))
    decision = evaluate_task(probe, _snapshot(), policy, evidence_root=tmp_path)
    assert decision["allowed"] is False
    gate = next(row for row in decision["gates"] if row["name"] == "receipt_prerequisites")
    assert any("payload digest drift" in problem for problem in gate["observed"][0]["problems"])


def test_p5_verifier_rejects_any_extra_mutation_id(tmp_path):
    verifier = tmp_path / "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
    payload = _write_p5_verifier(verifier)
    payload["mutation_tests"].append({"id": "invented-extra-mutation", "rejected": True})
    _write_payload_receipt(verifier, payload)

    problems = throttle._p5_verifier_authority_problems(payload, tmp_path)
    assert "P5 verifier required mutation rejection set drift" in problems


def test_p5_fresh_pattern_decision_is_rebuilt_from_raw_cells(tmp_path):
    primary = {
        "sesoi": 0.1,
        "frames": {"f64": {"off_ceiling": True}, "f32": {"off_ceiling": False}},
        "primary_contrasts_f64": {
            "exact_minus_window_local": {
                "n": 5,
                "mean": 0.3,
                "lo": 0.2,
                "hi": 0.4,
                "half": 0.1,
            }
        },
        "secondary_contrasts_f32": {},
    }
    runs = []
    for seed in throttle.P5_FRESH_SEEDS:
        path = tmp_path / f"runs/fresh/seed_{seed}/frames/f64/cell_receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        unit = {
            "mechanisms": {
                "exact_global": {
                    "frozen": {"evaluation": {"heldout_combo_score": 0.5}},
                    "evaluation": {"heldout_combo_score": 0.8, "chance": 0.25},
                },
                "window_local": {
                    "frozen": {"evaluation": {"heldout_combo_score": 0.5}},
                    "evaluation": {"heldout_combo_score": 0.8, "chance": 0.25},
                },
            }
        }
        path.write_text(
            json.dumps(
                {
                    "difficulty_calibration": {"clears_floor": True},
                    "seed_results": {str(seed): unit},
                }
            )
        )
        runs.append(
            {
                "seed": seed,
                "cell_receipts": {
                    "f64": {"path": str(path.relative_to(tmp_path))},
                },
            }
        )

    canonical = throttle._p5_canonical_challenge_patterns(
        primary,
        {"training_runs": runs},
        tmp_path,
    )
    assert canonical[0]["strict_direction_reproduced"] is False
    assert canonical[0]["programmatic_pattern_verified"] is False
    fabricated = json.loads(json.dumps(canonical))
    fabricated[0]["strict_direction_reproduced"] = True
    fabricated[0]["programmatic_pattern_verified"] = True
    fabricated[0]["outcome"] = "favorable-programmatic-only"
    assert fabricated != canonical


def test_p5_strict_pattern_order_survives_sorted_json_round_trip():
    primary = {
        "sesoi": 0.1,
        "frames": {"f64": {"off_ceiling": True}, "f32": {"off_ceiling": True}},
        "primary_contrasts_f64": {
            "exact_minus_recurrent": {"n": 5, "lo": 0.2, "hi": 0.4},
            "exact_minus_hierarchical_pooled": {"n": 5, "lo": 0.2, "hi": 0.4},
            "exact_minus_window_local": {"n": 5, "lo": -0.05, "hi": 0.05},
        },
        "secondary_contrasts_f32": {
            "exact_minus_recurrent": {"n": 5, "lo": 0.2, "hi": 0.4},
            "exact_minus_hierarchical_pooled": {"n": 5, "lo": 0.2, "hi": 0.4},
            "exact_minus_window_local": {"n": 5, "lo": -0.05, "hi": 0.05},
        },
    }
    sorted_round_trip = json.loads(json.dumps(primary, sort_keys=True))

    patterns = throttle._p5_strict_patterns(sorted_round_trip)

    assert [row["id"] for row in patterns] == [
        "f64-exact-minus-recurrent",
        "f64-exact-minus-hierarchical_pooled",
        "f32-exact-minus-recurrent",
        "f32-exact-minus-hierarchical_pooled",
    ]


def test_p5_primary_aggregate_is_rebuilt_from_licensed_seed_scores(tmp_path):
    primary = _write_p5_ancestors(tmp_path, pilot_outcome="null")
    cell_path = tmp_path / "runs/p5_context/p5pilot/frames/f64/cell_receipt.json"
    cell = json.loads(cell_path.read_text())
    seed = str(cell["expected_seeds"][0])
    cell["seed_results"][seed]["mechanisms"]["window_local"]["evaluation"]["heldout_combo_score"] = 0.30
    cell_path.write_text(json.dumps(cell))
    seed_path = cell_path.parent / f"seed_{seed}/seed_result.json"
    seed_path.write_text(json.dumps(cell["seed_results"][seed]))

    problems = throttle._p5_screen_authority_problems(primary, tmp_path)
    assert "P5 f64 score aggregates do not independently recompute" in problems
    assert "P5 f64 paired contrasts do not independently recompute" in problems


def test_p5_pilot_null_cannot_be_resealed_as_favorable_for_p6(tmp_path):
    policy = load_policy()
    path = tmp_path / "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
    payload = _write_p5_verifier(path)
    payload["classification"] = "favorable-programmatic-only"
    payload["outcome"] = "favorable-programmatic-only"
    _write_payload_receipt(path, payload)
    _seal_governor_run(tmp_path, "p5verify_pilot_null_cpu")

    problems = throttle._p5_verifier_authority_problems(payload, tmp_path)
    assert "P5 no-pattern verdict must remain a canonical null" in problems
    decision = evaluate_task(
        policy.task("p6_10k_resource_probe_cpu"),
        _snapshot(),
        policy,
        evidence_root=tmp_path,
    )
    assert decision["allowed"] is False


def test_p5_forged_checkpoint_state_hash_cannot_release_p6(tmp_path):
    policy = load_policy()
    path = tmp_path / "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
    payload = _write_p5_verifier(path)
    payload["artifact_evidence"]["primary"]["f64"]["0"]["arms"]["exact_global"]["checkpoint"][
        "model_state_sha256"
    ] = "f" * 64
    _write_payload_receipt(path, payload)
    _seal_governor_run(tmp_path, "p5verify_pilot_null_cpu")

    decision = evaluate_task(
        policy.task("p6_10k_resource_probe_cpu"),
        _snapshot(),
        policy,
        evidence_root=tmp_path,
    )
    assert decision["allowed"] is False
    gate = next(row for row in decision["gates"] if row["name"] == "receipt_prerequisites")
    assert any("checkpoint state hash drift" in problem for problem in gate["observed"][0]["problems"])


def test_p5_primary_cell_compute_drift_after_verification_blocks_p6(tmp_path):
    policy = load_policy()
    path = tmp_path / "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
    _write_p5_verifier(path)
    cell_path = tmp_path / "runs/p5_context/p5pilot/frames/f64/cell_receipt.json"
    cell = json.loads(cell_path.read_text())
    cell["compute"]["dense_flops_per_step"] += 1
    cell_path.write_text(json.dumps(cell))

    decision = evaluate_task(
        policy.task("p6_10k_resource_probe_cpu"),
        _snapshot(),
        policy,
        evidence_root=tmp_path,
    )
    assert decision["allowed"] is False
    gate = next(row for row in decision["gates"] if row["name"] == "receipt_prerequisites")
    problems = gate["observed"][0]["problems"]
    assert any("dense compute reference drift" in problem for problem in problems)
    assert any("cell receipt binding drift" in problem for problem in problems)


def test_p6_full_10k_fails_closed_until_probe_receipt_then_uses_measured_rss(tmp_path):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    blocked = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert blocked["allowed"] is False
    failed = {gate["name"] for gate in blocked["gates"] if not gate["ok"]}
    assert {"receipt_prerequisites", "resource_measurement"} <= failed

    _write_p6_receipt(
        tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
        rung=10_000,
        mode="resource-probe",
        replication=False,
    )
    admitted = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert admitted["allowed"] is True
    resource_gate = next(gate for gate in admitted["gates"] if gate["name"] == "resource_measurement")
    assert resource_gate["observed"]["max_rss_bytes"] == 200_000_000
    assert resource_gate["observed"]["effective_unified_memory_gb"] == pytest.approx(0.25)

    receipt_path = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    tampered = json.loads(receipt_path.read_text())
    tampered["resource_measurement"]["max_rss_bytes"] = 1
    receipt_path.write_text(json.dumps(tampered))
    drifted = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert drifted["allowed"] is False
    assert any(gate["name"] == "resource_measurement" and not gate["ok"] for gate in drifted["gates"])


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "wrong-task",
        "wrong-command",
        "stale-policy",
        "stale-implementation",
        "wrong-output-hash",
        "unsealed",
        "missing-child-rss",
    ),
)
def test_p6_resource_admission_rejects_invalid_governor_provenance(tmp_path, mutation):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    output = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    _write_p6_receipt(
        output,
        rung=10_000,
        mode="resource-probe",
        replication=False,
    )
    run_path = tmp_path / "runs/local_throttle/fixture-p6_10k_resource_probe_cpu/run_receipt.json"
    if mutation == "missing":
        run_path.unlink()
    else:
        receipt = json.loads(run_path.read_text())
        if mutation == "wrong-task":
            receipt["task"]["task_id"] = "p6_100k_replication_cpu"
            receipt["completion_authority"]["task_id"] = "p6_100k_replication_cpu"
            receipt["completion_authority"]["task"]["task_id"] = "p6_100k_replication_cpu"
        elif mutation == "wrong-command":
            command = [*receipt["invocations"][0]["command"], "--invented"]
            receipt["invocations"][0]["command"] = command
            receipt["invocations"][0]["command_sha256"] = throttle._command_sha256(command)
            receipt["completion_authority"]["command"] = command
            receipt["completion_authority"]["command_sha256"] = throttle._command_sha256(command)
        elif mutation == "stale-policy":
            receipt["policy"]["sha256"] = "0" * 64
            receipt["completion_authority"]["policy"]["sha256"] = "0" * 64
        elif mutation == "stale-implementation":
            receipt["implementation"]["sha256"] = "0" * 64
            receipt["completion_authority"]["implementation"]["sha256"] = "0" * 64
        elif mutation == "wrong-output-hash":
            receipt["completion_authority"]["output"]["sha256"] = "0" * 64
        elif mutation == "missing-child-rss":
            receipt["completion_authority"].pop("child_resource")
        elif mutation == "unsealed":
            receipt["status_note"] = "post-seal mutation"
        if mutation != "unsealed":
            receipt.pop("payload_sha256", None)
            receipt["payload_sha256"] = throttle._canonical_sha256(receipt)
        run_path.write_text(json.dumps(receipt, sort_keys=True))
    decision = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert decision["allowed"] is False
    resource_gate = next(row for row in decision["gates"] if row["name"] == "resource_measurement")
    assert any("governor" in problem for problem in resource_gate["observed"]["problems"])


@pytest.mark.parametrize(
    "historic_sha",
    sorted(throttle.COMPATIBLE_GOVERNOR_IMPLEMENTATION_SHA256),
)
def test_p6_resource_admission_accepts_exact_reviewed_legacy_baseline(tmp_path, historic_sha):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    output = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    _write_p6_receipt(
        output,
        rung=10_000,
        mode="resource-probe",
        replication=False,
    )
    run_path = tmp_path / "runs/local_throttle/fixture-p6_10k_resource_probe_cpu/run_receipt.json"
    receipt = json.loads(run_path.read_text())
    receipt["implementation"]["sha256"] = historic_sha
    receipt["completion_authority"]["implementation"]["sha256"] = historic_sha
    legacy_policy = {
        "path": str(policy.path),
        "sha256": "d2d113bf77daabe977515049e226d20b5333dac2597888ae756d5fd5908dd685",
    }
    receipt["policy"] = legacy_policy
    receipt["completion_authority"]["policy"] = legacy_policy
    receipt.pop("task_policy_authority")
    receipt["completion_authority"].pop("task_policy_authority")
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = throttle._canonical_sha256(receipt)
    run_path.write_text(json.dumps(receipt, sort_keys=True))

    decision = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert decision["allowed"] is True


def test_p6_governor_provenance_accepts_additive_policy_and_marker_growth(tmp_path):
    policy = load_policy()
    output = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    _write_p6_receipt(output, rung=10_000, mode="resource-probe", replication=False)
    extended = replace(
        policy,
        monitor={
            **policy.monitor,
            "known_heavy_markers": [*policy.monitor["known_heavy_markers"], "new_substrate.py"],
        },
        sha256="d" * 64,
    )

    decision = evaluate_task(
        extended.task("p6_10k_replication_cpu"),
        _snapshot(),
        extended,
        evidence_root=tmp_path,
    )

    assert decision["allowed"] is True


@pytest.mark.parametrize("mutation", ("marker-removal", "safety-contract"))
def test_p6_governor_provenance_rejects_scoped_policy_weakening(tmp_path, mutation):
    policy = load_policy()
    output = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    _write_p6_receipt(output, rung=10_000, mode="resource-probe", replication=False)
    if mutation == "marker-removal":
        changed = replace(
            policy,
            monitor={
                **policy.monitor,
                "known_heavy_markers": policy.monitor["known_heavy_markers"][:-1],
            },
            sha256="d" * 64,
        )
        expected = "known_heavy_markers were removed"
    else:
        changed = replace(
            policy,
            limits={**policy.limits, "minimum_unified_memory_headroom_gb": 3.0},
            sha256="d" * 64,
        )
        expected = "safety contract drifted"

    decision = evaluate_task(
        changed.task("p6_10k_replication_cpu"),
        _snapshot(),
        changed,
        evidence_root=tmp_path,
    )
    resource = next(row for row in decision["gates"] if row["name"] == "resource_measurement")
    assert any(expected in problem for problem in resource["observed"]["problems"])


def test_p6_governor_provenance_rejects_unknown_authorityless_governor(tmp_path):
    policy = load_policy()
    output = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    _write_p6_receipt(output, rung=10_000, mode="resource-probe", replication=False)
    run_path = tmp_path / "runs/local_throttle/fixture-p6_10k_resource_probe_cpu/run_receipt.json"
    receipt = json.loads(run_path.read_text())
    receipt["implementation"]["sha256"] = "0" * 64
    receipt["completion_authority"]["implementation"]["sha256"] = "0" * 64
    receipt.pop("task_policy_authority")
    receipt["completion_authority"].pop("task_policy_authority")
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = throttle._canonical_sha256(receipt)
    run_path.write_text(json.dumps(receipt, sort_keys=True))

    decision = evaluate_task(
        policy.task("p6_10k_replication_cpu"),
        _snapshot(),
        policy,
        evidence_root=tmp_path,
    )
    resource = next(row for row in decision["gates"] if row["name"] == "resource_measurement")
    assert any(
        "0 reviewed legacy policy baselines" in problem for problem in resource["observed"]["problems"]
    )


def test_p6_100k_stops_on_canonical_strict_10k_null(tmp_path):
    policy = load_policy()
    task = policy.task("p6_100k_replication_cpu")
    _write_p6_receipt(tmp_path / "proof/P6_CONTINUAL_10K.json", rung=10_000)
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    verifier_path = tmp_path / "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json"
    _write_p6_verifier(
        verifier_path,
        source_path=tmp_path / "proof/P6_CONTINUAL_10K.json",
        next_rung=100_000,
        next_allowed=False,
        favorable=False,
    )
    null_decision = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert null_decision["allowed"] is False

    _write_p6_verifier(
        verifier_path,
        source_path=tmp_path / "proof/P6_CONTINUAL_10K.json",
        next_rung=100_000,
    )
    canonical_null = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert canonical_null["allowed"] is False
    verifier = json.loads(verifier_path.read_text())
    assert verifier["independent_recompute"]["decision"]["aggregate_tie_count"] == 4
    assert verifier["prerequisite"]["next_rung_allowed"] is False


def test_p6_next_rung_rejects_contradictory_null_and_any_source_join_drift(tmp_path):
    policy = load_policy()
    task = policy.task("p6_100k_replication_cpu")
    source_path = tmp_path / "proof/P6_CONTINUAL_10K.json"
    verifier_path = tmp_path / "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json"
    _write_p6_receipt(source_path, rung=10_000)
    _write_p6_verifier(verifier_path, source_path=source_path, next_rung=100_000)
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    contradictory = json.loads(verifier_path.read_text())
    decision = contradictory["independent_recompute"]["decision"]
    decision.update(
        {
            "aggregate_tie_count": 4,
            "strict_joint_gain_all_schedules_and_controls": False,
            "verdict": "null",
            "null_supported": True,
        }
    )
    contradictory["prerequisite"]["next_rung_allowed"] = True
    _write_payload_receipt(verifier_path, contradictory)
    _seal_governor_run(tmp_path, "p6_10k_verify_cpu")
    contradictory_decision = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert contradictory_decision["allowed"] is False
    prerequisite_gate = next(
        row for row in contradictory_decision["gates"] if row["name"] == "receipt_prerequisites"
    )
    verifier_report = next(
        row for row in prerequisite_gate["observed"] if row["schema"] == throttle.P6_VERIFIER_SCHEMA
    )
    assert "P6 verifier semantic canonical rebuild drift" in verifier_report["problems"]

    for field in ("file_sha256", "payload_sha256", "identity_sha256"):
        _write_p6_verifier(verifier_path, source_path=source_path, next_rung=100_000)
        drifted = json.loads(verifier_path.read_text())
        drifted["source_rung"][field] = "0" * 64
        if field == "file_sha256":
            drifted["prerequisite"]["source_rung_file_sha256"] = "0" * 64
        elif field == "identity_sha256":
            drifted["prerequisite"]["source_identity_sha256"] = "0" * 64
        _write_payload_receipt(verifier_path, drifted)
        _seal_governor_run(tmp_path, "p6_10k_verify_cpu")
        assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False


def test_p6_next_rung_rejects_stale_verifier_implementation_and_seed_tie(tmp_path):
    policy = load_policy()
    task = policy.task("p6_100k_replication_cpu")
    source_path = tmp_path / "proof/P6_CONTINUAL_10K.json"
    verifier_path = tmp_path / "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json"
    _write_p6_receipt(source_path, rung=10_000)
    _write_p6_verifier(verifier_path, source_path=source_path, next_rung=100_000)

    stale = json.loads(verifier_path.read_text())
    stale["implementation"][0]["sha256"] = "0" * 64
    _write_payload_receipt(verifier_path, stale)
    _seal_governor_run(tmp_path, "p6_10k_verify_cpu")
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    _write_p6_verifier(verifier_path, source_path=source_path, next_rung=100_000)
    tied = json.loads(verifier_path.read_text())
    pair = tied["independent_recompute"]["decision"]["contrasts"][0]["paired_seed_deltas"][0]
    pair["retention_delta"] = 0.0
    pair["tie_is_null"] = True
    _write_payload_receipt(verifier_path, tied)
    _seal_governor_run(tmp_path, "p6_10k_verify_cpu")
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False


def test_p6_policy_refuses_every_weakened_verifier_gate(tmp_path):
    source = yaml.safe_load((REPO_ROOT / "configs/local_execution_throttle.yaml").read_text())
    weakened_fields = (
        "checks.live_dependencies_current",
        "checks.progress_and_checkpoints_current",
        "checks.full_replication_structure_valid",
        "prerequisite.valid_controls",
        "prerequisite.tie_is_null",
        "prerequisite.mutation_suite_all_rejected",
        "independent_recompute.decision.strict_joint_gain_all_schedules_and_controls",
    )
    for index, field in enumerate(weakened_fields):
        weakened = json.loads(json.dumps(source))
        for task_id in ("p6_100k_replication_cpu", "p6_1m_replication_cpu"):
            verifier = next(
                row
                for row in weakened["tasks"][task_id]["prerequisites"]
                if row["schema"] == throttle.P6_VERIFIER_SCHEMA
            )
            verifier["fields"][field] = False
        path = tmp_path / f"weakened-{index}.yaml"
        path.write_text(yaml.safe_dump(weakened, sort_keys=False))
        with pytest.raises(ThrottleRefused, match="exact favorable independent verifier"):
            load_policy(path)


def test_p6_resource_receipt_rejects_minimal_shape_and_one_byte_rss(tmp_path):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    receipt_path = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    evidence = throttle._p6_resource_evidence()
    minimal = {
        "schema": throttle.P6_RUNG_SCHEMA,
        "claim_scope": throttle.P6_CLAIM_SCOPE,
        "mode": "resource-probe",
        "rung": 10_000,
        "all_mechanics_ok": True,
        "replication_execution_complete": False,
        "independent_metric_verifier_complete": False,
        "scientific_promotion": False,
        "progress": {"completed_cells": 1, "expected_cells": 1},
        "identity": {
            "config_sha256": evidence["config_sha256"],
            "runner_sha256": throttle._sha256_file(REPO_ROOT / "scripts/continual_million_event_rung.py"),
            "source_preflight_file_sha256": evidence["preflight_file_sha256"],
            "source_preflight_payload_sha256": evidence["preflight_payload_sha256"],
            "source_live_bindings_sha256": evidence["preflight_live_bindings_sha256"],
        },
        "resource_measurement": {
            "max_rss_bytes": 200_000_000,
            "measured_after_complete": True,
            "events_per_stream": 10_000,
        },
    }
    _write_payload_receipt(receipt_path, minimal)
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    _write_p6_receipt(
        receipt_path,
        rung=10_000,
        mode="resource-probe",
        replication=False,
        rss=1,
    )
    decision = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert decision["allowed"] is False
    resource_gate = next(row for row in decision["gates"] if row["name"] == "resource_measurement")
    assert any("sanity floor" in problem for problem in resource_gate["observed"]["problems"])


def test_p6_resource_receipt_rejects_identity_progress_and_source_authority_drift(tmp_path):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    receipt_path = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"

    _write_p6_receipt(receipt_path, rung=10_000, mode="resource-probe", replication=False)
    identity_drift = json.loads(receipt_path.read_text())
    identity_drift["identity_sha256"] = "0" * 64
    _write_payload_receipt(receipt_path, identity_drift)
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    _write_p6_receipt(receipt_path, rung=10_000, mode="resource-probe", replication=False)
    progress_drift = json.loads(receipt_path.read_text())
    progress_path = tmp_path / progress_drift["progress"]["path"]
    progress_path.write_text(progress_path.read_text() + "\n")
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False

    _write_p6_receipt(receipt_path, rung=10_000, mode="resource-probe", replication=False)
    source_drift = json.loads(receipt_path.read_text())
    source_drift["source_live_authority"]["bindings_sha256"] = "0" * 64
    _write_payload_receipt(receipt_path, source_drift)
    assert evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)["allowed"] is False


def test_p6_resource_probe_rejects_fully_resealed_invented_checkpoint_state(tmp_path):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    receipt_path = tmp_path / "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json"
    payload = _write_p6_receipt(
        receipt_path,
        rung=10_000,
        mode="resource-probe",
        replication=False,
    )
    key = next(iter(payload["cells"]))
    row = payload["cells"][key]
    progress_path = tmp_path / payload["progress"]["path"]
    checkpoint_path = (
        progress_path.parent / "checkpoints" / f"seed_{row['seed']}" / row["schedule"] / f"{row['arm']}.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["state"]["updates"] += 1
    invented_state_sha = throttle._canonical_sha256(checkpoint["state"])
    checkpoint["state_sha256"] = invented_state_sha
    checkpoint["result"]["state_sha256"] = invented_state_sha
    checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True))
    row["state_sha256"] = invented_state_sha
    row["checkpoint_sha256"] = throttle._sha256_file(checkpoint_path)
    progress = json.loads(progress_path.read_text())
    progress["cells"][key] = row
    progress_path.write_text(json.dumps(progress, sort_keys=True))
    payload["cells"][key] = row
    payload["progress"]["sha256"] = throttle._sha256_file(progress_path)
    _write_payload_receipt(receipt_path, payload)
    _seal_governor_run(tmp_path, "p6_10k_resource_probe_cpu")

    decision = evaluate_task(task, _snapshot(), policy, evidence_root=tmp_path)
    assert decision["allowed"] is False
    prerequisite_gate = next(row for row in decision["gates"] if row["name"] == "receipt_prerequisites")
    assert any(
        "raw-stream semantic audit failed" in problem
        for problem in prerequisite_gate["observed"][0]["problems"]
    )


def test_p6_one_million_disk_projection_fails_before_crossing_floor(tmp_path):
    policy = load_policy()
    task = policy.task("p6_1m_replication_cpu")
    decision = evaluate_task(
        task,
        _snapshot(disk={"available": True, "free_gb": 41.0}),
        policy,
        evidence_root=tmp_path,
    )
    assert decision["allowed"] is False
    disk_gate = next(gate for gate in decision["gates"] if gate["name"] == "forecasted_disk")
    assert disk_gate["critical"] is True
    assert disk_gate["observed"]["projected_free_gb"] < 40.0


def test_p6_checkpoint_snapshot_covers_exact_resume_files_and_ignores_tmp(tmp_path):
    policy = load_policy()
    task = policy.task("p6_10k_replication_cpu")
    paths = (
        "runs/continual_million_event/rung_010000/streams/seed_20260710/abrupt/chunk_000000.bin",
        "runs/continual_million_event/rung_010000/streams/seed_20260710/abrupt/manifest.json",
        "runs/continual_million_event/rung_010000/checkpoints/seed_20260710/abrupt/replay.json",
        "runs/continual_million_event/rung_010000/progress.json",
        "proof/P6_CONTINUAL_10K.json",
    )
    for value in paths:
        path = tmp_path / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    temporary = tmp_path / (
        "runs/continual_million_event/rung_010000/checkpoints/seed_20260710/abrupt/replay.json.tmp"
    )
    temporary.write_text("partial")
    snapshot = checkpoint_snapshot(task, tmp_path)
    assert {row["path"] for row in snapshot["files"]} == set(paths)
    assert all(not row["path"].endswith(".tmp") for row in snapshot["files"])


def test_p6_policy_refuses_order_and_evidence_derived_forecast_drift(tmp_path):
    source = yaml.safe_load((REPO_ROOT / "configs/local_execution_throttle.yaml").read_text())
    wrong_order = json.loads(json.dumps(source))
    wrong_order["execution_order"]["p6_cpu"][0:2] = reversed(wrong_order["execution_order"]["p6_cpu"][0:2])
    order_path = tmp_path / "wrong-order.yaml"
    order_path.write_text(yaml.safe_dump(wrong_order, sort_keys=False))
    with pytest.raises(ThrottleRefused, match="execution_order.p6_cpu"):
        load_policy(order_path)

    wrong_forecast = json.loads(json.dumps(source))
    wrong_forecast["tasks"]["p6_1m_replication_cpu"]["forecast_write_gb"] += 0.001
    forecast_path = tmp_path / "wrong-forecast.yaml"
    forecast_path.write_text(yaml.safe_dump(wrong_forecast, sort_keys=False))
    with pytest.raises(ThrottleRefused, match="not derived from the 384-event receipt"):
        load_policy(forecast_path)


def test_p6_atomic_projection_covers_one_million_raw_chunk_and_refuses_understatement(tmp_path):
    policy = load_policy()
    task = policy.task("p6_1m_replication_cpu")
    raw_chunk_bytes = 51 * 10_000
    assert task.atomic_write_gb * 1e9 >= raw_chunk_bytes
    assert policy.limits["minimum_write_reserve_gb"] == 0.5

    source = yaml.safe_load((REPO_ROOT / "configs/local_execution_throttle.yaml").read_text())
    source["tasks"]["p6_1m_replication_cpu"]["atomic_write_gb"] = raw_chunk_bytes / 1e9 - 1e-9
    path = tmp_path / "understated-atomic.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False))
    with pytest.raises(ThrottleRefused, match="atomic_write_gb is not derived"):
        load_policy(path)


def test_first_mps_heavy_lane_is_allowed_with_measured_headroom():
    policy = load_policy()
    decision = evaluate_task(policy.task("p4_resume_mps"), _snapshot(), policy)
    assert decision["schema"] == DECISION_SCHEMA
    assert decision["allowed"] is True
    assert decision["disk_forecast"]["projected_free_gb"] >= 40.0


def test_second_heavy_lane_is_always_denied():
    policy = load_policy()
    task = policy.task("p4_resume_mps")
    decision = evaluate_task(task, _snapshot(), policy, active=[_active(task)])
    assert decision["allowed"] is False
    failed = {gate["name"] for gate in decision["gates"] if not gate["ok"]}
    assert {"one_heavy", "second_lane_kind", "single_mps_owner"} <= failed


def test_second_light_lane_is_allowed_only_under_strict_headroom():
    policy = load_policy()
    heavy = policy.task("p4_resume_mps")
    light = policy.task("docs_verification")
    decision = evaluate_task(light, _snapshot(), policy, active=[_active(heavy)])
    assert decision["threshold_tier"] == "second_lane"
    assert decision["allowed"] is True


def test_blender_presence_blocks_a_second_lane_but_not_the_only_experiment_lane():
    policy = load_policy()
    heavy = policy.task("p4_resume_mps")
    light = policy.task("docs_verification")
    processes = {
        "available": True,
        "foreground_resource_processes": [{"pid": 77, "name": "Blender"}],
        "unmanaged_known_heavy": [],
    }
    snapshot = _snapshot(processes=processes)
    assert evaluate_task(heavy, snapshot, policy)["allowed"] is True
    decision = evaluate_task(light, snapshot, policy, active=[_active(heavy)])
    assert decision["allowed"] is False
    assert any(gate["name"] == "foreground_second_lane" and not gate["ok"] for gate in decision["gates"])


def test_missing_telemetry_fails_closed():
    policy = load_policy()
    snapshot = _snapshot(missing_required_telemetry=["thermal"], all_required_available=False)
    decision = evaluate_task(policy.task("docs_verification"), snapshot, policy)
    assert decision["allowed"] is False
    assert decision["gates"][0]["name"] == "required_telemetry"


def test_mps_task_fails_closed_without_working_set_telemetry():
    policy = load_policy()
    snapshot = _snapshot(mps={"available": True, "telemetry_available": False, "scope": "missing"})
    decision = evaluate_task(policy.task("p4_resume_mps"), snapshot, policy)
    assert decision["allowed"] is False
    assert any(gate["name"] == "mps_telemetry" and not gate["ok"] for gate in decision["gates"])


def test_runtime_memory_gate_does_not_reserve_the_already_running_peak_twice():
    policy = load_policy()
    memory = {
        "available": True,
        "total_bytes": int(19.3e9),
        "available_bytes": int(3e9),
        "available_percent": 25.0,
        "pressure": {"available": True, "free_percent": 60.0},
    }
    snapshot = _snapshot(memory=memory)
    task = policy.task("p4_resume_mps")
    assert evaluate_task(task, snapshot, policy)["allowed"] is False
    assert evaluate_task(task, snapshot, policy, task_already_active=True)["allowed"] is True


def test_owned_cpu_saturation_is_admission_only_not_a_runtime_self_pause():
    policy = load_policy()
    task = policy.task("p4_resume_cpu")
    cpu = {
        "available": True,
        "logical_cpus": 12,
        "load_1m_per_logical_cpu": 1.25,
        "utilization_fraction": 1.0,
    }
    snapshot = _snapshot(cpu=cpu)
    admission = evaluate_task(task, snapshot, policy)
    assert admission["allowed"] is False
    assert {gate["name"] for gate in admission["gates"] if not gate["ok"]} == {
        "cpu_load",
        "cpu_utilization",
    }
    runtime = evaluate_task(task, snapshot, policy, task_already_active=True)
    assert runtime["allowed"] is True
    for name in ("cpu_load", "cpu_utilization"):
        gate = next(value for value in runtime["gates"] if value["name"] == name)
        assert gate["ok"] is True
        assert gate["limit"] == "admission-only"


def test_owned_worker_process_group_is_not_classified_as_unmanaged_heavy(monkeypatch):
    policy = load_policy()

    class FakeProcess:
        def __init__(self, pid, ppid, command):
            self.info = {
                "pid": pid,
                "ppid": ppid,
                "name": "python3.12",
                "cmdline": command.split(),
                "username": "test-user",
            }

    processes = [
        FakeProcess(101, 100, "python scripts/p5_traingrid_memory_probe.py _child owned"),
        FakeProcess(202, 1, "python scripts/p5_traingrid_memory_probe.py _child external"),
    ]
    monkeypatch.setattr(throttle.psutil, "process_iter", lambda _fields: processes)
    monkeypatch.setattr(throttle.os, "getpgid", lambda pid: {101: 100, 202: 200}[pid])
    monkeypatch.setattr(
        throttle,
        "_frontmost_app",
        lambda: {"available": False, "name": None},
    )

    observed = throttle._processes(
        policy,
        excluded_pids={100},
        excluded_process_groups={100},
    )

    assert [row["pid"] for row in observed["unmanaged_known_heavy"]] == [202]


def test_forecasted_writes_preserve_the_40gb_floor_and_fail_critical():
    policy = load_policy()
    snapshot = _snapshot(disk={"available": True, "free_gb": 45.0})
    decision = evaluate_task(policy.task("p4_resume_mps"), snapshot, policy)
    assert decision["allowed"] is False
    assert decision["critical"] is True
    disk_gate = next(gate for gate in decision["gates"] if gate["name"] == "forecasted_disk")
    assert disk_gate["observed"]["projected_free_gb"] < 40.0


def test_admission_and_runtime_hysteresis_require_consecutive_samples():
    policy = load_policy()
    allowed = {"allowed": True, "critical": False}
    denied = {
        "allowed": False,
        "critical": False,
        "denied_reasons": ["memory pressure gate failed"],
    }
    short = aggregate_admission([allowed, allowed], 3)
    assert short["allowed"] is False
    assert short["denied_reasons"] == ["admission observed 2 of 3 required consecutive samples"]
    refused = aggregate_admission([allowed, denied, denied], 3)
    assert refused["denied_reasons"] == ["memory pressure gate failed"]
    assert refused["reason"] == "memory pressure gate failed"
    assert aggregate_admission([denied, allowed, allowed, allowed], 3)["allowed"] is True
    first = hysteresis_transition(
        "running",
        denied,
        good_count=0,
        bad_count=0,
        last_transition_monotonic=0.0,
        now_monotonic=10.0,
        policy=policy,
    )
    assert first["action"] == "none"
    second = hysteresis_transition(
        "running",
        denied,
        good_count=first["good_count"],
        bad_count=first["bad_count"],
        last_transition_monotonic=0.0,
        now_monotonic=20.0,
        policy=policy,
    )
    assert second["action"] == "pause"
    state = second
    for now in (30.0, 40.0, 90.0):
        state = hysteresis_transition(
            "paused",
            allowed,
            good_count=state["good_count"],
            bad_count=state["bad_count"],
            last_transition_monotonic=20.0,
            now_monotonic=now,
            policy=policy,
        )
    assert state["action"] == "resume"


def test_checkpoint_snapshot_hashes_final_publication_and_ignores_tmp(tmp_path):
    task = TaskDeclaration(
        task_id="fixture",
        lane="light",
        accelerator="none",
        cpu_cores=1,
        estimated_unified_memory_gb=0.1,
        estimated_mps_gb=0.0,
        resource_basis="fixture",
        forecast_write_gb=0.1,
        atomic_write_gb=0.1,
        wall_minutes=1,
        pause_safe=True,
        atomic_checkpoints=True,
        checkpoint_globs=("run/*",),
        restart_exit_codes=(),
        command=("true",),
    )
    (tmp_path / "run").mkdir()
    (tmp_path / "run/checkpoint.pt").write_bytes(b"complete")
    (tmp_path / "run/checkpoint.pt.tmp").write_bytes(b"partial")
    first = checkpoint_snapshot(task, tmp_path)
    assert [row["path"] for row in first["files"]] == ["run/checkpoint.pt"]
    assert first["file_count"] == 1
    second = checkpoint_snapshot(replace(task), tmp_path)
    assert first["aggregate_sha256"] == second["aggregate_sha256"]


def test_decision_payload_is_json_serializable():
    policy = load_policy()
    decision = evaluate_task(policy.task("p4_resume_mps"), _snapshot(), policy)
    assert json.loads(json.dumps(decision))["allowed"] is True


def test_list_cli_serializes_slotted_task_declarations(capsys):
    from scripts.local_execution_throttle import main

    assert main(["list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tasks"]["p4_resume_cpu"]["task_id"] == "p4_resume_cpu"
    assert payload["tasks"]["p4_resume_cpu"]["command"][1] == ("scripts/p4_capability_density.py")


def test_dry_receipt_binds_policy_and_implementation(monkeypatch):
    from mop.studio import local_throttle

    policy = load_policy()
    monkeypatch.setattr(local_throttle, "collect_host_telemetry", lambda *_args, **_kwargs: _snapshot())
    receipt = local_throttle.dry_run_decision(
        policy.task("p4_resume_cpu"), policy, samples=3, interval_seconds=0
    )
    assert receipt["policy"]["sha256"] == policy.sha256
    assert receipt["implementation"]["path"] == "src/mop/studio/local_throttle.py"
    assert len(receipt["implementation"]["sha256"]) == 64
    authority = receipt["task_policy_authority"]
    assert authority["schema"] == throttle.task_policy.TASK_POLICY_AUTHORITY_SCHEMA
    assert authority["task_id"] == "p4_resume_cpu"
    assert authority["task_sha256"] == throttle.task_policy.canonical_sha256(receipt["task"])
    unsealed = dict(authority)
    declared = unsealed.pop("authority_sha256")
    assert declared == throttle.task_policy.canonical_sha256(unsealed)


def test_fast_child_rusage_fallback_persists_positive_peak(monkeypatch, tmp_path):
    base_policy = load_policy()
    policy = replace(
        base_policy,
        monitor={
            **base_policy.monitor,
            "admission_good_samples": 1,
            "sample_interval_seconds": 0.01,
        },
    )
    task = TaskDeclaration(
        task_id="fast-child-fixture",
        lane="light",
        accelerator="none",
        cpu_cores=1,
        estimated_unified_memory_gb=0.1,
        estimated_mps_gb=0.0,
        resource_basis="short-lived child RSS fallback fixture",
        forecast_write_gb=0.0,
        atomic_write_gb=0.0,
        wall_minutes=1,
        pause_safe=False,
        atomic_checkpoints=False,
        checkpoint_globs=(),
        restart_exit_codes=(),
        command=(throttle.sys.executable, "-c", "pass"),
    )
    monkeypatch.setattr(throttle, "collect_host_telemetry", lambda *_args, **_kwargs: _snapshot())
    monkeypatch.setattr(throttle, "_process_tree_rss_bytes", lambda _pid: 0)
    receipt = throttle.run_task(
        task,
        policy,
        run_id="fast-child-rusage",
        state_root=tmp_path / "runs/local_throttle",
        disk_root=tmp_path,
    )
    assert receipt["status"] == "complete"
    assert receipt["child_resource"]["psutil_peak_rss_bytes"] == 0
    assert receipt["child_resource"]["direct_child_rusage_peak_rss_bytes"] > 0
    assert (
        receipt["child_resource"]["peak_rss_bytes"]
        == receipt["child_resource"]["direct_child_rusage_peak_rss_bytes"]
    )


def test_seed_boundary_yields_after_one_restart_exit_and_seals_progress(monkeypatch, tmp_path):
    base_policy = load_policy()
    policy = replace(
        base_policy,
        monitor={
            **base_policy.monitor,
            "admission_good_samples": 1,
            "sample_interval_seconds": 0.01,
        },
    )
    command_source = (
        "from pathlib import Path; import sys; "
        "Path('proof').mkdir(exist_ok=True); "
        "Path('proof/seed.checkpoint.json').write_text('{\"seed\": 1}'); "
        "sys.exit(2)"
    )
    task = TaskDeclaration(
        task_id="edcm1_official_cpu",
        lane="cpu",
        accelerator="none",
        cpu_cores=1,
        estimated_unified_memory_gb=0.1,
        estimated_mps_gb=0.0,
        resource_basis="one-invocation progress-authority fixture",
        forecast_write_gb=0.0,
        atomic_write_gb=0.0,
        wall_minutes=1,
        pause_safe=True,
        atomic_checkpoints=True,
        checkpoint_globs=("proof/seed.checkpoint.json",),
        restart_exit_codes=(2,),
        command=(throttle.sys.executable, "-c", command_source),
    )
    governor = tmp_path / "src/mop/studio/local_throttle.py"
    governor.parent.mkdir(parents=True)
    governor.write_text("progress-authority-fixture")
    monkeypatch.setattr(throttle, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(throttle, "IMPLEMENTATION_PATH", governor)
    monkeypatch.setattr(throttle, "_is_seed_boundary_task", lambda _task: True)
    monkeypatch.setattr(throttle, "collect_host_telemetry", lambda *_args, **_kwargs: _snapshot())

    receipt = throttle.run_task(
        task,
        policy,
        run_id="one-seed-boundary",
        state_root=tmp_path / "runs/local_throttle",
        disk_root=tmp_path,
    )

    assert receipt["status"] == "resumable-invocation-boundary"
    assert receipt["final_returncode"] == 2
    assert len(receipt["invocations"]) == 1
    assert receipt["progress_authority"]["schema"] == throttle.PROGRESS_AUTHORITY_SCHEMA
    assert receipt["progress_authority"]["owned_child_active"] is False
    assert (
        receipt["progress_authority"]["final_checkpoint_aggregate_sha256"]
        == (receipt["final_checkpoint"]["aggregate_sha256"])
    )
    unsealed = dict(receipt)
    declared = unsealed.pop("payload_sha256")
    assert declared == throttle._canonical_sha256(unsealed)


def test_validated_external_profile_keeps_cpu_gates_live_at_runtime(monkeypatch):
    policy = load_policy()
    task = policy.task("edcm1_official_cpu")
    unmanaged = [{"pid": 101, "name": "Python"}]
    monkeypatch.setattr(
        throttle,
        "_external_coexistence_report",
        lambda _task, _rows: {"allowed": True, "all_ok": True, "problems": []},
    )
    memory = {
        "available": True,
        "total_bytes": int(100e9),
        "available_bytes": int(45e9),
        "available_percent": 45.0,
        "pressure": {"available": True, "free_percent": 85.0},
    }
    processes = {
        "available": True,
        "foreground_resource_processes": [],
        "unmanaged_known_heavy": unmanaged,
    }
    healthy = _snapshot(memory=memory, processes=processes, swap={"available": True, "used_gb": 0.0})
    decision = evaluate_task(task, healthy, policy, task_already_active=True)
    assert decision["allowed"] is True
    assert decision["threshold_tier"] == "external_coexistence"

    saturated = _snapshot(
        cpu={
            "available": True,
            "logical_cpus": 28,
            "load_1m_per_logical_cpu": 0.90,
            "utilization_fraction": 0.90,
        },
        memory=memory,
        processes=processes,
        swap={"available": True, "used_gb": 0.0},
    )
    denied = evaluate_task(task, saturated, policy, task_already_active=True)
    assert denied["allowed"] is False
    failing = {gate["name"] for gate in denied["gates"] if not gate["ok"]}
    assert "cpu_load" in failing
    assert "cpu_utilization" not in failing

    monkeypatch.setattr(
        throttle,
        "_external_coexistence_report",
        lambda _task, _rows: {
            "profile": "hawking_v5_ultra_cpu_v1",
            "allowed": True,
            "all_ok": True,
            "problems": [],
        },
    )
    v5_residual = _snapshot(
        cpu={
            "available": True,
            "logical_cpus": 28,
            "load_1m_per_logical_cpu": 0.97,
            "utilization_fraction": 0.97,
        },
        memory=memory,
        processes=processes,
        swap={"available": True, "used_gb": 0.0},
    )
    assert evaluate_task(task, v5_residual, policy, task_already_active=True)["allowed"] is True
    v5_saturated = {
        **v5_residual,
        "cpu": {
            **v5_residual["cpu"],
            "load_1m_per_logical_cpu": 2.0,
            "utilization_fraction": 1.0,
        },
    }
    v5_decision = evaluate_task(task, v5_saturated, policy, task_already_active=True)
    assert v5_decision["allowed"] is True
    cpu_gates = {gate["name"]: gate for gate in v5_decision["gates"] if gate["name"].startswith("cpu_")}
    assert cpu_gates["cpu_load"]["limit"] == "kernel-enforced background QoS"
    assert cpu_gates["cpu_utilization"]["limit"] == "kernel-enforced background QoS"

    transient_burst = _snapshot(
        cpu={
            "available": True,
            "logical_cpus": 28,
            "load_1m_per_logical_cpu": 0.50,
            "utilization_fraction": 1.0,
        },
        memory=memory,
        processes=processes,
        swap={"available": True, "used_gb": 0.0},
    )
    burst_decision = evaluate_task(task, transient_burst, policy, task_already_active=True)
    assert burst_decision["allowed"] is True

    low_pressure_memory = {
        **memory,
        "pressure": {"available": True, "free_percent": 74.0},
    }
    low_pressure = evaluate_task(
        task,
        _snapshot(
            memory=low_pressure_memory,
            processes=processes,
            swap={"available": True, "used_gb": 0.0},
        ),
        policy,
        task_already_active=True,
    )
    pressure_gate = next(
        gate for gate in low_pressure["gates"] if gate["name"] == "external_coexistence_pressure_free"
    )
    assert pressure_gate["ok"] is False
    assert pressure_gate["limit"] == 75.0


def test_completion_reuses_admission_time_policy_and_governor_bindings(monkeypatch, tmp_path):
    base_policy = load_policy()
    policy = replace(
        base_policy,
        monitor={
            **base_policy.monitor,
            "admission_good_samples": 1,
            "sample_interval_seconds": 0.01,
        },
    )
    governor = tmp_path / "src/mop/studio/local_throttle.py"
    governor.parent.mkdir(parents=True)
    governor.write_text("admission-version")
    before_sha = throttle._sha256_file(governor)
    command_source = (
        "from pathlib import Path; "
        "Path('proof/output.json').parent.mkdir(parents=True, exist_ok=True); "
        "Path('proof/output.json').write_text('{}'); "
        "Path('src/mop/studio/local_throttle.py').write_text('completion-version')"
    )
    task = TaskDeclaration(
        task_id="p5-admission-binding-fixture",
        lane="light",
        accelerator="none",
        cpu_cores=1,
        estimated_unified_memory_gb=0.1,
        estimated_mps_gb=0.0,
        resource_basis="admission-binding fixture",
        forecast_write_gb=0.0,
        atomic_write_gb=0.0,
        wall_minutes=1,
        pause_safe=False,
        atomic_checkpoints=True,
        checkpoint_globs=("proof/output.json",),
        restart_exit_codes=(),
        command=(
            throttle.sys.executable,
            "-c",
            command_source,
            "--out",
            "proof/output.json",
        ),
    )
    monkeypatch.setattr(throttle, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(throttle, "IMPLEMENTATION_PATH", governor)
    monkeypatch.setattr(throttle, "collect_host_telemetry", lambda *_args, **_kwargs: _snapshot())

    receipt = throttle.run_task(
        task,
        policy,
        run_id="admission-binding",
        state_root=tmp_path / "runs/local_throttle",
        disk_root=tmp_path,
    )

    assert receipt["status"] == "complete"
    assert throttle._sha256_file(governor) != before_sha
    assert receipt["implementation"]["sha256"] == before_sha
    assert receipt["completion_authority"]["implementation"] == receipt["implementation"]
    assert receipt["completion_authority"]["policy"] == receipt["policy"]
    assert receipt["completion_authority"]["task_policy_authority"] == receipt["task_policy_authority"]


def test_non_pause_safe_child_is_stopped_on_critical_runtime_gate(monkeypatch, tmp_path):
    base_policy = load_policy()
    policy = replace(
        base_policy,
        monitor={
            **base_policy.monitor,
            "admission_good_samples": 1,
            "sample_interval_seconds": 0.01,
            "graceful_stop_seconds": 0.2,
        },
    )
    task = TaskDeclaration(
        task_id="non-pause-safe-fixture",
        lane="light",
        accelerator="none",
        cpu_cores=1,
        estimated_unified_memory_gb=0.1,
        estimated_mps_gb=0.0,
        resource_basis="runtime safety stop fixture",
        forecast_write_gb=0.0,
        atomic_write_gb=0.0,
        wall_minutes=1,
        pause_safe=False,
        atomic_checkpoints=False,
        checkpoint_globs=(),
        restart_exit_codes=(),
        command=(throttle.sys.executable, "-c", "import time; time.sleep(60)"),
    )
    snapshots = iter(
        (
            _snapshot(disk={"available": True, "free_gb": 80.0}),
            _snapshot(disk={"available": True, "free_gb": 39.0}),
        )
    )
    monkeypatch.setattr(throttle, "collect_host_telemetry", lambda *_args, **_kwargs: next(snapshots))

    state_root = tmp_path / "runs/local_throttle"
    receipt = throttle.run_task(
        task,
        policy,
        run_id="non-pause-safe-stop",
        state_root=state_root,
        disk_root=tmp_path,
    )

    assert receipt["status"] == "failed-dynamic-safety-stop"
    assert receipt["final_returncode"] != 0
    assert "completion_authority" not in receipt
    assert any(row["event"] == "dynamic-safety-stop" for row in receipt["events"])
    assert active_lanes(state_root) == []


def test_corrupt_active_registry_fails_closed(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "active.json").write_text("not json")
    with pytest.raises(ThrottleRefused, match="registry"):
        active_lanes(tmp_path)
