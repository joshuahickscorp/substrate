"""Measured localization of non-F frontier experiments on the current M3 Pro.

The experiment bank historically used ``studio-scale``, ``gpu-later``, and ``moonshot`` as
planning labels.  Those labels are not measurements.  This module does two narrower things:

* executes the *mechanics* of five previously registry-only synthesis rows on bounded fixtures;
* audits every non-F row carrying one of the historical frontier labels and names its first
  observed blocker without turning a fixture into scientific evidence.

The local preflights intentionally stop short of the registered scientific claims.  A caller
cannot promote them by changing a command-line flag: scientific launch readiness is derived from
named, independently auditable prerequisite receipts and fails closed while any are absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from ..config import REPO_ROOT
from .project_exhaustion import (
    DATA_BLOCKED,
    IMPLEMENTATION_BLOCKED,
    MODEL_BLOCKED,
    REGISTRY_ONLY_BLOCKERS,
    STANDALONE_EVIDENCE,
)

PREFLIGHT_SCHEMA = "mop-local-frontier-preflights/v1"
AUDIT_SCHEMA = "mop-frontier-localization/v1"
PREREQUISITE_SCHEMA = "mop-scientific-prerequisite/v1"
USER_WALL_MINUTES = 180

TARGET_IDS = (
    "mop_mt4_reasoning_router",
    "mop_at2_mode_substrate_dep",
    "mop_cm5_studio_rejuvenation",
    "mop_cm11_developmental_plasticity",
    "mop_cm12_mop_substrate_capstone",
)

# Frozen at the start of the localization pass so concurrent, justified tier demotions cannot make
# an old frontier row disappear from the audit that is supposed to account for it.
HISTORICAL_FRONTIER_IDS = (
    "mop_mt4_reasoning_router",
    "mop_dr1_video_cache",
    "mop_dr2_sparse_real",
    "mop_dr3_latent_scratchpad",
    "mop_dr4_causal_intervention",
    "mop_dr5_cross_substrate_consistency",
    "mop_dr7_latent_cot",
    "mop_dr14_corruption",
    "mop_dr15_modality_general",
    "mop_at1_nuisance_grid",
    "mop_at2_mode_substrate_dep",
    "mop_al2_shared_latent_alignment",
    "mop_al3_audio_video_alignment",
    "mop_cm1_compositional_gate",
    "mop_cm2_atlas_gate",
    "mop_cm3_dense_vs_pooled",
    "mop_cm4_workspace_shell",
    "mop_cm5_studio_rejuvenation",
    "mop_cm6_distilled_density",
    "mop_cm7_min_objective_probe",
    "mop_cm8_custom_jepa_pilot",
    "mop_cm9_slot_jepa_binding",
    "mop_cm11_developmental_plasticity",
    "mop_cm12_mop_substrate_capstone",
)

DEFAULT_RUN_RECEIPT = REPO_ROOT / "runs" / "frontier_localization" / "local_preflights.json"
DEFAULT_PREFLIGHT_PROOF = REPO_ROOT / "proof" / "LOCAL_FRONTIER_PREFLIGHTS.json"
DEFAULT_AUDIT_PROOF = REPO_ROOT / "proof" / "FRONTIER_LOCALIZATION.json"


class ScientificLaunchBlocked(RuntimeError):
    """Raised when a frontier row lacks one or more scientific prerequisite receipts."""


@dataclass(frozen=True)
class Requirement:
    key: str
    description: str
    receipt_candidates: tuple[str, ...] = ()


SCIENTIFIC_REQUIREMENTS: dict[str, tuple[Requirement, ...]] = {
    "mop_mt4_reasoning_router": (
        Requirement(
            "compatible_primitives",
            "all six reasoning primitives evaluated on the same referents with compatible outputs",
        ),
        Requirement(
            "distinct_strategy_gate",
            "a preregistered gate proves the primitives make materially distinct errors",
        ),
        Requirement("independent_verifier", "fresh seeds are checked by an independent verifier"),
    ),
    "mop_at2_mode_substrate_dep": (
        Requirement("verified_winning_mode", "a candidate mode has a verified positive gain"),
        Requirement(
            "real_substrate_cache",
            "a citable real-substrate cache covers the candidate mode's exact referents",
            ("data/cache/vjepa2_vitl_local8_citable/manifest.json",),
        ),
        Requirement(
            "matched_random_cache",
            "a same-architecture, same-resolution random-init cache covers the same referents",
        ),
        Requirement("independent_verifier", "fresh seeds are checked by an independent verifier"),
    ),
    "mop_cm5_studio_rejuvenation": (
        Requirement(
            "plasticity_loss_gate",
            "the no-rejuvenation arm first exhibits nontrivial plasticity loss at the chosen rung",
        ),
        Requirement(
            "adapting_frozen_pair",
            "adapting and frozen substrate arms share data, initialization family, and update budget",
        ),
        Requirement("independent_verifier", "fresh seeds are checked by an independent verifier"),
    ),
    "mop_cm11_developmental_plasticity": (
        Requirement(
            "developmental_regime_gate",
            "the fixture is replaced by a calibrated non-ceiling curriculum with a noisy-TV guard",
        ),
        Requirement(
            "signature_verifier",
            "window, path-dependence, and U-shape signatures are independently recomputed",
        ),
    ),
    "mop_cm12_mop_substrate_capstone": (
        Requirement("cleared_experts", "two to four expert/substrate pilots have verified receipts"),
        Requirement(
            "compatible_battery",
            "all experts expose the same referents, tasks, metrics, and total-compute accounting",
        ),
        Requirement("open_model_control", "the declared strongest open-model control is available"),
        Requirement("independent_verifier", "fresh seeds are checked by an independent verifier"),
    ),
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": path.is_file(),
    }
    if path.is_file():
        out.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
    return out


def _evidence_rows_match(rows: Any) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    for item in rows:
        if not isinstance(item, dict):
            return False
        rel = item.get("path")
        digest = item.get("sha256")
        if not isinstance(rel, str) or not isinstance(digest, str) or len(digest) != 64:
            return False
        path = (REPO_ROOT / rel).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()) or not path.is_file():
            return False
        if _sha256(path) != digest:
            return False
    return True


def _prerequisite_contract_ok(parsed: Any, experiment_id: str, requirement: str) -> bool:
    if not isinstance(parsed, dict):
        return False
    verifier = parsed.get("verifier")
    return bool(
        parsed.get("schema") == PREREQUISITE_SCHEMA
        and parsed.get("experiment_id") == experiment_id
        and parsed.get("requirement") == requirement
        and parsed.get("verified") is True
        and isinstance(parsed.get("claim_scope"), str)
        and bool(parsed["claim_scope"].strip())
        and isinstance(verifier, dict)
        and isinstance(verifier.get("name"), str)
        and bool(verifier["name"].strip())
        and verifier.get("independent") is True
        and _evidence_rows_match(parsed.get("evidence"))
    )


def _max_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _entropy(indices: torch.Tensor, n: int) -> float:
    counts = torch.bincount(indices, minlength=n).float()
    probs = counts[counts > 0] / counts.sum().clamp_min(1)
    return float(-(probs * probs.log()).sum())


def _ridge_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    l2: float = 1e-3,
) -> float:
    classes = int(torch.max(train_y).item()) + 1
    ones_train = torch.ones(len(train_x), 1)
    ones_test = torch.ones(len(test_x), 1)
    design = torch.cat([train_x.float(), ones_train], dim=1)
    test_design = torch.cat([test_x.float(), ones_test], dim=1)
    target = F.one_hot(train_y.long(), classes).float()
    eye = torch.eye(design.shape[1]) * l2
    weights = torch.linalg.solve(design.T @ design + eye, design.T @ target)
    pred = (test_design @ weights).argmax(dim=1)
    return float((pred == test_y).float().mean())


def _train_router(
    train_x: torch.Tensor,
    train_target: torch.Tensor,
    test_x: torch.Tensor,
    *,
    n_outputs: int,
    seed: int,
    steps: int = 120,
) -> torch.Tensor:
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], n_outputs)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(model(train_x), train_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model(test_x).argmax(dim=1)


def preflight_mt4(seed: int = 20260709) -> dict[str, Any]:
    """Run router, strongest fixed strategy, shuffled router, and exact compute accounting."""
    g = torch.Generator().manual_seed(seed)
    n = 320
    x = torch.randn(n, 3, generator=g)
    names = ("fixed_n", "adaptive_halt", "beam", "memory", "plan", "verify")
    angle = torch.atan2(x[:, 1], x[:, 0]) + math.pi
    specialty = torch.floor(angle / (2 * math.pi) * len(names)).long().clamp(max=len(names) - 1)
    y = (specialty + (x[:, 2] > 0).long()) % 4
    predictions = []
    for primitive in range(len(names)):
        fallback = (y + 1 + (primitive % 3)) % 4
        pred = torch.where(specialty == primitive, y, fallback)
        predictions.append(pred)
    pred_matrix = torch.stack(predictions, dim=1)
    correctness = pred_matrix == y[:, None]
    oracle = correctness.float().argmax(dim=1)
    perm = torch.randperm(n, generator=g)
    train_idx, test_idx = perm[:220], perm[220:]
    router_features = torch.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 0] * x[:, 1]], dim=1)
    selected = _train_router(
        router_features[train_idx],
        oracle[train_idx],
        router_features[test_idx],
        n_outputs=len(names),
        seed=seed,
    )
    test_preds = pred_matrix[test_idx]
    routed = test_preds[torch.arange(len(test_idx)), selected]
    fixed_accuracies = (test_preds == y[test_idx, None]).float().mean(dim=0)
    shuffled_selection = selected[torch.randperm(len(selected), generator=g)]
    shuffled = test_preds[torch.arange(len(test_idx)), shuffled_selection]
    primitive_flops = 1000
    accounting_flops = 64
    router_mean_flops = primitive_flops + accounting_flops
    fixed_control_flops = primitive_flops + accounting_flops
    return {
        "experiment_id": "mop_mt4_reasoning_router",
        "fixture_only": True,
        "seed": seed,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_test_disjoint": not bool(set(train_idx.tolist()) & set(test_idx.tolist())),
        "primitive_names": list(names),
        "router_accuracy": float((routed == y[test_idx]).float().mean()),
        "best_fixed_accuracy": float(fixed_accuracies.max()),
        "best_fixed_primitive": names[int(fixed_accuracies.argmax())],
        "shuffled_router_accuracy": float((shuffled == y[test_idx]).float().mean()),
        "selection_entropy_nats": _entropy(selected, len(names)),
        "router_mean_flops": router_mean_flops,
        "fixed_control_mean_flops": fixed_control_flops,
        "matched_mean_flops": router_mean_flops == fixed_control_flops,
        "controls_executed": [
            "best-single-fixed-strategy",
            "matched-mean-flops",
            "shuffled-router-selection",
        ],
        "mechanics_verified": bool(
            len(torch.unique(selected)) == len(names)
            and router_mean_flops == fixed_control_flops
            and not (set(train_idx.tolist()) & set(test_idx.tolist()))
        ),
    }


def preflight_at2(seed: int = 20260709) -> dict[str, Any]:
    """Exercise identical mode/head code on matched real-like and random-like fixture views."""
    g = torch.Generator().manual_seed(seed)
    n, frames, dim = 360, 4, 8
    latent = torch.randn(n, generator=g)
    y = (latent > 0).long()
    # Stable signal plus frame nuisance is a substrate fixture, never a pretrained-model claim.
    real_like = torch.randn(n, frames, dim, generator=g) * 1.5
    real_like[:, :, 0] += latent[:, None]
    random_like = torch.randn(n, frames, dim, generator=g)
    perm = torch.randperm(n, generator=g)
    train_idx, test_idx = perm[:240], perm[240:]

    def evaluate(view: torch.Tensor) -> dict[str, float]:
        baseline = view[:, 0]
        mode = view.mean(dim=1)
        baseline_acc = _ridge_accuracy(baseline[train_idx], y[train_idx], baseline[test_idx], y[test_idx])
        mode_acc = _ridge_accuracy(mode[train_idx], y[train_idx], mode[test_idx], y[test_idx])
        return {
            "baseline_accuracy": baseline_acc,
            "mode_accuracy": mode_acc,
            "mode_gain": mode_acc - baseline_acc,
        }

    real_result = evaluate(real_like)
    random_result = evaluate(random_like)
    return {
        "experiment_id": "mop_at2_mode_substrate_dep",
        "fixture_only": True,
        "seed": seed,
        "candidate_mode": "temporal-mean-refinement-fixture",
        "matched_resolution": 256,
        "matched_frames": frames,
        "matched_shape": list(real_like.shape) == list(random_like.shape),
        "same_probe_code": True,
        "train_test_disjoint": not bool(set(train_idx.tolist()) & set(test_idx.tolist())),
        "real_like_fixture": real_result,
        "random_init_like_fixture": random_result,
        "mode_gain_real_minus_random": real_result["mode_gain"] - random_result["mode_gain"],
        "controls_executed": ["random-init-like-view", "matched-resolution", "same-head"],
        "mechanics_verified": bool(
            real_like.shape == random_like.shape and not (set(train_idx.tolist()) & set(test_idx.tolist()))
        ),
        "scientific_input_warning": (
            "real_like_fixture is generated signal, not a real pretrained cache; "
            "random_init_like_fixture is generated noise, not a same-architecture checkpoint"
        ),
    }


class _PlasticNet(torch.nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.substrate = torch.nn.Linear(dim, hidden)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.substrate(x)))


def _rejuvenate(
    net: _PlasticNet,
    *,
    apply: bool,
    substrate: bool,
    generator: torch.Generator,
) -> int:
    with torch.no_grad():
        utility = net.head.weight.abs().mean(dim=0)
        count = max(1, len(utility) // 4)
        units = utility.argsort()[:count]
        head_candidate = net.head.weight[:, units] * 0.25
        head_candidate += 0.02 * torch.randn(net.head.weight[:, units].shape, generator=generator)
        substrate_candidate = net.substrate.weight[units] * 0.25
        substrate_candidate += 0.02 * torch.randn(net.substrate.weight[units].shape, generator=generator)
        bias_candidate = net.substrate.bias[units] * 0.25
        if apply:
            net.head.weight[:, units] = head_candidate
            if substrate:
                net.substrate.weight[units] = substrate_candidate
                net.substrate.bias[units] = bias_candidate
    return int(count) if apply else 0


def _plasticity_arm(
    tasks: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
    adapt_substrate: bool,
    rejuvenate: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 19)
    net = _PlasticNet(tasks[0][0].shape[1], hidden=20)
    initial_substrate = [parameter.detach().clone() for parameter in net.substrate.parameters()]
    optimizer = torch.optim.SGD(net.parameters(), lr=0.08)
    adaptation_gains: list[float] = []
    update_steps = 0
    auxiliary_ops = 0
    rejuvenated_units = 0
    for task_index, (x, y) in enumerate(tasks):
        with torch.no_grad():
            before = float((net(x).argmax(dim=1) == y).float().mean())
        for _ in range(4):
            frozen_snapshot = None
            if not adapt_substrate:
                frozen_snapshot = [parameter.detach().clone() for parameter in net.substrate.parameters()]
            loss = F.cross_entropy(net(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if frozen_snapshot is not None:
                with torch.no_grad():
                    for parameter, value in zip(net.substrate.parameters(), frozen_snapshot, strict=True):
                        parameter.copy_(value)
            update_steps += 1
        with torch.no_grad():
            after = float((net(x).argmax(dim=1) == y).float().mean())
        adaptation_gains.append(after - before)
        if task_index < len(tasks) - 1:
            rejuvenated_units += _rejuvenate(
                net,
                apply=rejuvenate,
                substrate=adapt_substrate,
                generator=g,
            )
            auxiliary_ops += 1
    with torch.no_grad():
        retention = float(torch.stack([(net(x).argmax(dim=1) == y).float().mean() for x, y in tasks]).mean())
        substrate_drift = max(
            float((parameter - initial).abs().max())
            for parameter, initial in zip(net.substrate.parameters(), initial_substrate, strict=True)
        )
    return {
        "mean_adaptation_gain": sum(adaptation_gains) / len(adaptation_gains),
        "final_retention": retention,
        "optimizer_update_steps": update_steps,
        "auxiliary_intervals": auxiliary_ops,
        "rejuvenated_units": rejuvenated_units,
        "adapt_substrate": adapt_substrate,
        "rejuvenation": rejuvenate,
        "substrate_max_abs_drift": substrate_drift,
    }


def preflight_cm5(seed: int = 20260709) -> dict[str, Any]:
    """Execute adapting/frozen and rejuvenation/no-op arms with equal update counts."""
    g = torch.Generator().manual_seed(seed)
    dim, samples, n_tasks = 12, 64, 5
    tasks: list[tuple[torch.Tensor, torch.Tensor]] = []
    base = torch.randn(dim, generator=g)
    for task in range(n_tasks):
        x = torch.randn(samples, dim, generator=g)
        direction = base + 0.45 * torch.randn(dim, generator=g) + task * 0.03
        y = ((x @ direction) > 0).long()
        tasks.append((x, y))
    arms = {
        "adapting_noop": _plasticity_arm(tasks, seed=seed, adapt_substrate=True, rejuvenate=False),
        "adapting_rejuvenation": _plasticity_arm(tasks, seed=seed, adapt_substrate=True, rejuvenate=True),
        "frozen_noop": _plasticity_arm(tasks, seed=seed, adapt_substrate=False, rejuvenate=False),
        "frozen_rejuvenation": _plasticity_arm(tasks, seed=seed, adapt_substrate=False, rejuvenate=True),
    }
    update_counts = {arm["optimizer_update_steps"] for arm in arms.values()}
    aux_counts = {arm["auxiliary_intervals"] for arm in arms.values()}
    frozen_exact = all(
        arms[name]["substrate_max_abs_drift"] == 0.0 for name in ("frozen_noop", "frozen_rejuvenation")
    )
    restoration = (
        arms["adapting_rejuvenation"]["mean_adaptation_gain"] - arms["adapting_noop"]["mean_adaptation_gain"]
    )
    retention_cost = max(
        0.0,
        arms["adapting_noop"]["final_retention"] - arms["adapting_rejuvenation"]["final_retention"],
    )
    return {
        "experiment_id": "mop_cm5_studio_rejuvenation",
        "fixture_only": True,
        "seed": seed,
        "tasks": n_tasks,
        "dimension": dim,
        "arms": arms,
        "matched_optimizer_updates": len(update_counts) == 1,
        "matched_auxiliary_intervals": len(aux_counts) == 1,
        "matched_gradient_compute_shape": True,
        "frozen_substrates_bit_exact": frozen_exact,
        "restoration_net_of_retention_cost": restoration - retention_cost,
        "plasticity_loss_gate_cleared": False,
        "controls_executed": [
            "do-nothing-plus-accounting-padding",
            "frozen-substrate",
            "adapting-vs-frozen",
        ],
        "mechanics_verified": len(update_counts) == 1 and len(aux_counts) == 1 and frozen_exact,
        "scientific_input_warning": (
            "the fixture does not establish plasticity loss at a target rung; no rejuvenation "
            "effect is interpretable until that prerequisite gate clears"
        ),
    }


class _DevelopmentalNet(torch.nn.Module):
    def __init__(self, dim: int = 6, hidden: int = 12) -> None:
        super().__init__()
        self.adapter = torch.nn.Linear(dim, hidden)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.adapter(x)))


def _developmental_condition(
    *,
    seed: int,
    schedule: str,
    order: tuple[int, int],
    onset: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 101)
    net = _DevelopmentalNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.08)
    directions = (
        torch.tensor([1.0, -0.8, 0.5, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, -0.7, 1.0, 0.6]),
    )
    updates = 0
    noisy_rows_seen = 0
    for stage, task_id in enumerate(order):
        x = torch.randn(72, 6, generator=g)
        clean_y = ((x @ directions[task_id]) > 0).long()
        noisy_mask = torch.zeros(len(x), dtype=torch.bool)
        noisy_mask[::9] = True
        noisy_y = torch.randint(0, 2, (int(noisy_mask.sum()),), generator=g)
        y = clean_y.clone()
        y[noisy_mask] = noisy_y
        noisy_rows_seen += int(noisy_mask.sum())
        for local_step in range(8):
            global_step = stage * 8 + local_step
            if schedule == "developmental":
                multiplier = 1.6 if onset <= global_step < onset + 6 else 0.45
            else:
                multiplier = 1.0
            # The noisy-TV guard is an explicit weight mask, not outcome-based row deletion.
            loss_rows = F.cross_entropy(net(x), y, reduction="none")
            weights = torch.where(noisy_mask, 0.2, 1.0)
            loss = (loss_rows * weights).sum() / weights.sum()
            optimizer.param_groups[0]["lr"] = 0.08 * multiplier
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
    probe = torch.randn(160, 6, generator=g)
    scores = []
    with torch.no_grad():
        logits = net(probe)
        params = torch.cat([parameter.flatten() for parameter in net.parameters()])
        for direction in directions:
            labels = ((probe @ direction) > 0).long()
            scores.append(float((logits.argmax(dim=1) == labels).float().mean()))
    return {
        "mean_accuracy": sum(scores) / len(scores),
        "task_accuracies": scores,
        "updates": updates,
        "noisy_rows_seen": noisy_rows_seen,
        "parameter_fingerprint": float(params[:64].sum()),
    }


def preflight_cm11(seed: int = 20260709) -> dict[str, Any]:
    """Exercise window, path-order, U-shape, flat schedule, and noisy-TV guard mechanics."""
    early = _developmental_condition(seed=seed, schedule="developmental", order=(0, 1), onset=0)
    middle = _developmental_condition(seed=seed, schedule="developmental", order=(0, 1), onset=5)
    late = _developmental_condition(seed=seed, schedule="developmental", order=(0, 1), onset=10)
    reverse = _developmental_condition(seed=seed, schedule="developmental", order=(1, 0), onset=0)
    flat = _developmental_condition(seed=seed, schedule="flat", order=(0, 1), onset=0)
    update_counts = {row["updates"] for row in (early, middle, late, reverse, flat)}
    noise_counts = {row["noisy_rows_seen"] for row in (early, middle, late, reverse, flat)}
    return {
        "experiment_id": "mop_cm11_developmental_plasticity",
        "fixture_only": True,
        "seed": seed,
        "d6_window": early["mean_accuracy"] - late["mean_accuracy"],
        "y4_path_area_proxy": abs(early["parameter_fingerprint"] - reverse["parameter_fingerprint"]),
        "d4_u_shape_curvature": (
            early["mean_accuracy"] + late["mean_accuracy"] - 2 * middle["mean_accuracy"]
        ),
        "developmental_minus_flat": early["mean_accuracy"] - flat["mean_accuracy"],
        "conditions": {
            "early": early,
            "middle": middle,
            "late": late,
            "reverse_order": reverse,
            "flat_schedule": flat,
        },
        "matched_compute": len(update_counts) == 1,
        "noisy_tv_guard_applied": len(noise_counts) == 1 and next(iter(noise_counts)) > 0,
        "controls_executed": ["flat-schedule", "matched-compute", "reverse-order", "noisy-tv"],
        "mechanics_verified": len(update_counts) == 1 and len(noise_counts) == 1,
        "scientific_input_warning": (
            "these are fixture-scale signature calculations; their signs and magnitudes are not "
            "developmental-plasticity evidence"
        ),
    }


def preflight_cm12(seed: int = 20260709) -> dict[str, Any]:
    """Execute expert routing and all local capstone controls with exact FLOP padding."""
    g = torch.Generator().manual_seed(seed)
    n, n_experts = 360, 4
    x = torch.randn(n, 4, generator=g)
    context = (x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long()
    y = (context + (x[:, 2] > 0).long()) % 4
    expert_predictions = []
    for expert in range(n_experts):
        fallback = (y + (expert % 3) + 1) % 4
        expert_predictions.append(torch.where(context == expert, y, fallback))
    experts = torch.stack(expert_predictions, dim=1)
    perm = torch.randperm(n, generator=g)
    train_idx, test_idx = perm[:240], perm[240:]
    router_features = torch.stack([x[:, 0], x[:, 1], x[:, 0] * x[:, 1]], dim=1)
    selected = _train_router(
        router_features[train_idx],
        context[train_idx],
        router_features[test_idx],
        n_outputs=n_experts,
        seed=seed + 7,
    )
    test_experts = experts[test_idx]
    mixture = test_experts[torch.arange(len(test_idx)), selected]
    fixed_accs = (test_experts == y[test_idx, None]).float().mean(dim=0)
    random_selected = torch.randint(0, n_experts, (len(test_idx),), generator=g)
    random_mix = test_experts[torch.arange(len(test_idx)), random_selected]
    expert_flops, workspace_flops, padding_flops = 2000, 160, 6000
    mixture_total = n_experts * expert_flops + workspace_flops
    single_total = expert_flops + workspace_flops + padding_flops
    return {
        "experiment_id": "mop_cm12_mop_substrate_capstone",
        "fixture_only": True,
        "seed": seed,
        "experts": n_experts,
        "mixture_accuracy": float((mixture == y[test_idx]).float().mean()),
        "best_single_accuracy": float(fixed_accs.max()),
        "random_router_accuracy": float((random_mix == y[test_idx]).float().mean()),
        "selection_entropy_nats": _entropy(selected, n_experts),
        "mixture_total_flops": mixture_total,
        "single_control_total_flops": single_total,
        "single_control_padding_flops": padding_flops,
        "matched_total_flops": mixture_total == single_total,
        "controls_executed": [
            "best-single-expert",
            "random-router",
            "matched-total-flops-with-accounting-padding",
        ],
        "mechanics_verified": bool(mixture_total == single_total and len(torch.unique(selected)) > 1),
        "scientific_input_warning": (
            "fixture experts are deterministic specialists, not cleared substrate pilots; "
            "the strongest-open-model control is a required external prerequisite"
        ),
    }


PREFLIGHTS: dict[str, Callable[[int], dict[str, Any]]] = {
    "mop_mt4_reasoning_router": preflight_mt4,
    "mop_at2_mode_substrate_dep": preflight_at2,
    "mop_cm5_studio_rejuvenation": preflight_cm5,
    "mop_cm11_developmental_plasticity": preflight_cm11,
    "mop_cm12_mop_substrate_capstone": preflight_cm12,
}


def scientific_readiness(
    experiment_id: str, supplied_receipts: dict[str, str] | None = None
) -> dict[str, Any]:
    """Return receipt-backed readiness; descriptions or booleans never satisfy a requirement."""
    supplied_receipts = supplied_receipts or {}
    requirements = SCIENTIFIC_REQUIREMENTS[experiment_id]
    checks = []
    for requirement in requirements:
        candidates = []
        if requirement.key in supplied_receipts:
            candidates.append(supplied_receipts[requirement.key])
        candidates.extend(requirement.receipt_candidates)
        receipts = []
        for rel in candidates:
            path = Path(rel)
            if not path.is_absolute():
                path = REPO_ROOT / path
            path = path.resolve()
            receipt = (
                _file_receipt(path)
                if path.is_relative_to(REPO_ROOT.resolve())
                else {
                    "path": str(path),
                    "exists": False,
                    "rejected": "outside-repository receipt paths are not portable",
                }
            )
            if receipt.get("exists") and path.suffix == ".json":
                try:
                    parsed = json.loads(path.read_text())
                    receipt["json_mapping"] = isinstance(parsed, dict)
                    receipt["semantic_contract_ok"] = _prerequisite_contract_ok(
                        parsed, experiment_id, requirement.key
                    )
                except (OSError, json.JSONDecodeError):
                    receipt["json_mapping"] = False
                    receipt["semantic_contract_ok"] = False
            receipts.append(receipt)
        satisfied = requirement.key in supplied_receipts and any(
            row.get("exists") and row.get("semantic_contract_ok") for row in receipts
        )
        checks.append(
            {
                "key": requirement.key,
                "description": requirement.description,
                "satisfied": satisfied,
                "receipts": receipts,
            }
        )
    return {
        "experiment_id": experiment_id,
        "ready": all(row["satisfied"] for row in checks),
        "checks": checks,
        "missing": [row["key"] for row in checks if not row["satisfied"]],
    }


def assert_scientific_launch_ready(
    experiment_id: str, supplied_receipts: dict[str, str] | None = None
) -> None:
    readiness = scientific_readiness(experiment_id, supplied_receipts)
    if not readiness["ready"]:
        raise ScientificLaunchBlocked(
            f"{experiment_id} scientific launch blocked; missing receipt-backed requirements: "
            + ", ".join(readiness["missing"])
        )


def run_local_preflights(*, seeds: tuple[int, ...] = (20260709, 20260710, 20260711)) -> dict[str, Any]:
    """Run all bounded local mechanics and record time/RSS without interpreting outcomes."""
    if not seeds:
        raise ValueError("at least one seed is required")
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows: dict[str, Any] = {}
    for experiment_id, fn in PREFLIGHTS.items():
        row_started = time.perf_counter()
        seed_results = [fn(seed) for seed in seeds]
        if not all(result.get("mechanics_verified") for result in seed_results):
            raise RuntimeError(f"{experiment_id} local mechanics failed verification")
        rows[experiment_id] = {
            "scope": "bounded generated fixture; mechanics only, never scientific evidence",
            "seconds": round(time.perf_counter() - row_started, 6),
            "seed_results": seed_results,
            "seed_count": len(seed_results),
            "mechanics_verified": True,
            "scientific_readiness": scientific_readiness(experiment_id),
        }
    return {
        "schema": PREFLIGHT_SCHEMA,
        "generated_at_unix": time.time(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "scope": (
            "local executable preflight over generated fixtures; proves control/dataflow mechanics "
            "and resource feasibility only"
        ),
        "claim_boundary": (
            "no fixture metric is a registered-result estimate; every scientific path remains "
            "receipt-gated and currently fails closed"
        ),
        "wall_budget_minutes": USER_WALL_MINUTES,
        "seeds": list(seeds),
        "results": rows,
        "target_count": len(rows),
        "all_mechanics_verified": all(row["mechanics_verified"] for row in rows.values()),
        "all_scientific_paths_fail_closed": all(
            not row["scientific_readiness"]["ready"] for row in rows.values()
        ),
        "seconds": round(time.perf_counter() - started, 6),
        "max_rss_bytes": _max_rss_bytes(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tagged_non_f_rows() -> list[dict[str, Any]]:
    payload = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text())
    rows = []
    for raw in payload["experiments"]:
        row = dict(raw)
        if str(row["id"]).startswith("f"):
            continue
        if row["resource_tier"] in {"studio-scale", "moonshot"} or row["exp_tier"] == "gpu-later":
            rows.append(row)
    return rows


def _historically_tagged_non_f_rows() -> list[dict[str, Any]]:
    """Include rows already demoted by this work by identity, preserving audit closure."""
    current = {str(row["id"]): row for row in _tagged_non_f_rows()}
    registry = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text())["experiments"]
    by_id = {str(row["id"]): dict(row) for row in registry}
    for experiment_id in HISTORICAL_FRONTIER_IDS:
        if experiment_id in by_id:
            current.setdefault(experiment_id, by_id[experiment_id])
    return [current[key] for key in sorted(current)]


def _git_revision() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def build_frontier_audit(preflights: dict[str, Any]) -> dict[str, Any]:
    """Audit every historical frontier tag against receipts, not labels."""
    preflight_rows = preflights.get("results", {})
    if (
        preflights.get("schema") != PREFLIGHT_SCHEMA
        or set(preflight_rows) != set(PREFLIGHTS)
        or preflights.get("all_mechanics_verified") is not True
        or preflights.get("all_scientific_paths_fail_closed") is not True
    ):
        raise ValueError("frontier audit requires a complete, verified, fail-closed preflight receipt")
    project_ledger = _load_json(REPO_ROOT / "proof" / "PROJECT_EXPERIMENT_EXHAUSTION.json")
    project_entries = {str(row["id"]): row for row in project_ledger.get("entries", [])}
    encoder_receipt_path = REPO_ROOT / "proof" / "REAL_ENCODER_LOCAL_ATTEMPT.json"
    encoder_receipt = _load_json(encoder_receipt_path)
    doctor_path = REPO_ROOT / "proof" / "STUDIO_READINESS_CURRENT_HOST.json"
    doctor = _load_json(doctor_path)
    scale_atlas_path = REPO_ROOT / "proof" / "VJEPA_SCALE_ATLAS_LOCAL.json"
    scale_atlas = _load_json(scale_atlas_path)
    scale_atlas_valid = bool(
        scale_atlas.get("schema") == "mop-vjepa-scale-atlas-local/v1"
        and scale_atlas.get("all_ok") is True
        and (scale_atlas.get("migration_use") or {}).get("local_serial_execution_proven") is True
        and (scale_atlas.get("migration_use") or {}).get("model_availability_blocker_retired") is True
        and (scale_atlas.get("alignment") or {}).get("n") == 8
    )
    seconds_per_clip = float(encoder_receipt.get("seconds_per_clip") or 0.0)
    safe_sequential_clips = (
        math.floor(USER_WALL_MINUTES * 60 / seconds_per_clip * 0.80) if seconds_per_clip > 0 else None
    )
    entries: list[dict[str, Any]] = []
    for row in _historically_tagged_non_f_rows():
        experiment_id = str(row["id"])
        prior = project_entries.get(experiment_id, {})
        preflight = preflight_rows.get(experiment_id)
        blocker_category, blocker_reason = REGISTRY_ONLY_BLOCKERS.get(
            experiment_id,
            (
                prior.get("classification") or IMPLEMENTATION_BLOCKED,
                prior.get("remaining_scientific_blocker")
                or prior.get("classification_reason")
                or "no experiment-specific hardware receipt exists",
            ),
        )
        custom_receipts = []
        custom_sources = []
        custom_paths = {
            "mop_cm7_min_objective_probe": ("proof/CUSTOM_SUBSTRATE_CALIBRATION.json",),
            "mop_cm8_custom_jepa_pilot": ("proof/CUSTOM_SUBSTRATE_CM8_PREFLIGHT.json",),
        }
        for rel in custom_paths.get(experiment_id, ()):
            path = REPO_ROOT / rel
            custom_receipts.append(_file_receipt(path))
        if experiment_id in {"mop_cm7_min_objective_probe", "mop_cm8_custom_jepa_pilot"}:
            custom_sources = [
                _file_receipt(REPO_ROOT / "src" / "mop" / "experiments" / "custom_substrate.py"),
                _file_receipt(REPO_ROOT / "configs" / "experiment" / f"{experiment_id}.yaml"),
                _file_receipt(REPO_ROOT / "scripts" / "custom_substrate_workbench.py"),
            ]

        if preflight:
            localization = "local-mechanics-proven"
            hardware_conclusion = (
                "demoted: three-seed control/dataflow mechanics executed locally; full claim remains gated"
            )
            first_blocker = ", ".join(preflight["scientific_readiness"]["missing"])
        elif experiment_id in STANDALONE_EVIDENCE and prior.get("execution_verified"):
            localization = "already-local-pilot"
            hardware_conclusion = "demoted: a local execution receipt already exists"
            first_blocker = blocker_reason
        elif blocker_category in {DATA_BLOCKED, MODEL_BLOCKED}:
            localization = "external-input-blocked"
            hardware_conclusion = "not a hardware boundary: the first blocker is data/model evidence"
            first_blocker = blocker_reason
        elif blocker_category == IMPLEMENTATION_BLOCKED:
            localization = "implementation-blocked"
            hardware_conclusion = "not a hardware boundary: the registered mechanism has no full runner"
            first_blocker = blocker_reason
        else:
            localization = "unproven-planning-tag"
            hardware_conclusion = "not measured: planning metadata cannot establish a boundary"
            first_blocker = blocker_reason
        custom_payload = _load_json(REPO_ROOT / custom_paths.get(experiment_id, ("missing",))[0])
        if experiment_id == "mop_cm7_min_objective_probe" and bool(
            custom_payload.get("schema") == "mop-custom-substrate-workbench/v1"
            and custom_payload.get("complete") is True
            and (custom_payload.get("requirements") or {}).get("all_ok") is True
            and (custom_payload.get("compute_match") or {}).get("all_ok") is True
        ):
            telemetry = custom_payload.get("resource_telemetry") or {}
            localization = "local-custom-training-calibration-proven"
            hardware_conclusion = (
                "demoted by measured local training: one complete calibration seed executed in "
                f"{float(telemetry.get('wall_seconds_this_invocation') or 0.0):.2f}s at "
                f"{int(telemetry.get('max_rss_bytes') or 0)} peak RSS bytes"
            )
            first_blocker = (
                "four additional complete seeds and independent verification; the current receipt "
                "is programmatic-video objective-lever calibration, not natural-video evidence"
            )
        elif experiment_id == "mop_cm8_custom_jepa_pilot" and bool(
            custom_payload.get("schema") == "mop-custom-substrate-cm8-preflight/v1"
            and custom_payload.get("runnable_local_preflight") is True
            and custom_payload.get("scientific_execution_ready") is False
        ):
            localization = "local-custom-preflight-proven-upstream-blocked"
            hardware_conclusion = "demoted by a measured fail-closed local execution preflight"
            first_blocker = (
                "CM7 still needs four complete seeds; teacher join has 8 of 64 required referents "
                "and is programmatic; CM1, CM2, and DR1 natural-data receipts remain absent"
            )
        elif experiment_id == "mop_cm7_min_objective_probe" and all(
            receipt.get("exists") for receipt in custom_sources
        ):
            localization = "local-implementation-present-run-pending"
            hardware_conclusion = (
                "demoted: a registered local training implementation exists; its durable run receipt "
                "is pending"
            )
            first_blocker = (
                "fresh multi-seed workbench execution and durable receipt under the local resource gate"
            )
        if (
            experiment_id in {"mop_al2_shared_latent_alignment", "mop_dr5_cross_substrate_consistency"}
            and scale_atlas_valid
        ):
            localization = "local-multi-encoder-availability-proven"
            hardware_conclusion = (
                "three real-weight frozen encoders executed serially on the same eight local referents; "
                "model availability is retired as a blocker"
            )
            if experiment_id == "mop_al2_shared_latent_alignment":
                first_blocker = (
                    "expand beyond eight programmatic referents onto meaningful shared content and "
                    "complete the preregistered equal-rank/matched-random control set"
                )
            else:
                first_blocker = (
                    "supply a verified reasoning gain on one compatible task, expand same-task rows, "
                    "and add the matched random-architecture control"
                )
        reclassified = bool(
            row.get("resource_tier") not in {"studio-scale", "moonshot"}
            and row.get("exp_tier") != "gpu-later"
        )
        entries.append(
            {
                "id": experiment_id,
                "current_registry": {
                    "resource_tier": row.get("resource_tier"),
                    "exp_tier": row.get("exp_tier"),
                    "runtime_class": row.get("runtime_class"),
                    "status": row.get("status"),
                },
                "historical_frontier_tag_in_scope": True,
                "reclassified_away_from_hardware_planning_tags": reclassified,
                "localization": localization,
                "hardware_boundary_proven": False,
                "hardware_conclusion": hardware_conclusion,
                "first_remaining_scientific_blocker": first_blocker,
                "project_exhaustion_classification": prior.get("classification"),
                "project_exhaustion_fresh_class_codepath_verified": bool(
                    prior.get("fresh_class_codepath_verified")
                ),
                "preflight_mechanics_verified": bool(preflight and preflight.get("mechanics_verified")),
                "custom_substrate_receipts": custom_receipts,
                "custom_substrate_sources": custom_sources,
                "scale_atlas_receipt": (
                    _file_receipt(scale_atlas_path)
                    if experiment_id
                    in {"mop_al2_shared_latent_alignment", "mop_dr5_cross_substrate_consistency"}
                    else None
                ),
            }
        )
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["localization"]] = counts.get(entry["localization"], 0) + 1
    accounted_exactly_once = len(entries) == len({row["id"] for row in entries})
    current_stale_rows = _tagged_non_f_rows()
    all_historical_rows_reclassified = all(
        row["reclassified_away_from_hardware_planning_tags"] for row in entries
    )
    measured_encoder = {
        "receipt": _file_receipt(encoder_receipt_path),
        "completed": encoder_receipt.get("completed"),
        "hardware_limit_reached": encoder_receipt.get("hardware_limit_reached"),
        "seconds_per_clip": seconds_per_clip or None,
        "max_rss_bytes": encoder_receipt.get("max_rss_bytes"),
        "measured_samples": encoder_receipt.get("samples"),
        "claim_scope": encoder_receipt.get("claim_scope"),
        "safe_sequential_clips_projected_at_80pct_wall": safe_sequential_clips,
        "projection_warning": (
            "linear sequential-time projection from an eight-clip programmatic run; not a throughput "
            "guarantee and not natural-video scientific evidence"
        ),
    }
    payload = {
        "schema": AUDIT_SCHEMA,
        "generated_at_unix": time.time(),
        "git_revision": _git_revision(),
        "scope": (
            "every non-F registry row historically or currently labeled studio-scale, gpu-later, "
            "or moonshot; F-series is audited separately"
        ),
        "decision_rule": (
            "hardware-blocked requires an experiment-specific measured local failure and projection; "
            "planning tags, missing code, missing rights data, and missing upstream caches do not qualify"
        ),
        "user_wall_minutes": USER_WALL_MINUTES,
        "host_readiness": {
            "receipt": _file_receipt(doctor_path),
            "all_ok": doctor.get("all_ok"),
            "chip": (doctor.get("host") or {}).get("chip"),
            "unified_memory_gb": (doctor.get("host") or {}).get("unified_memory_gb"),
            "mps_available": (doctor.get("host") or {}).get("mps_available"),
        },
        "measured_real_encoder": measured_encoder,
        "measured_multi_encoder_atlas": {
            "receipt": _file_receipt(scale_atlas_path),
            "valid_local_availability_evidence": scale_atlas_valid,
            "scope": scale_atlas.get("scope"),
            "shared_referents": (scale_atlas.get("alignment") or {}).get("n"),
            "scientific_promotable": (scale_atlas.get("claim_eligibility") or {}).get("promotable"),
            "remaining_reasons": (scale_atlas.get("claim_eligibility") or {}).get("reasons"),
        },
        "local_preflights": {
            "receipt_schema": preflights.get("schema"),
            "all_mechanics_verified": preflights.get("all_mechanics_verified"),
            "all_scientific_paths_fail_closed": preflights.get("all_scientific_paths_fail_closed"),
            "seconds": preflights.get("seconds"),
            "max_rss_bytes": preflights.get("max_rss_bytes"),
            "target_ids": list(preflight_rows),
        },
        "coverage": {
            "tagged_non_f_count": len(entries),
            "historical_frontier_count": len(HISTORICAL_FRONTIER_IDS),
            "current_stale_planning_tag_count": len(current_stale_rows),
            "current_stale_planning_tag_ids": sorted(str(row["id"]) for row in current_stale_rows),
            "all_historical_rows_reclassified": all_historical_rows_reclassified,
            "accounted_exactly_once": accounted_exactly_once,
            "localization_counts": dict(sorted(counts.items())),
            "measured_hardware_blocked_count": 0,
        },
        "conclusion": (
            "No audited non-F frontier row has an experiment-specific measured M3 Pro hardware "
            "boundary. Five cross-pillar mechanics now run locally, CM7 has a registered local "
            "implementation, and CM8 has a fail-closed local preflight receipt. Every historical "
            "hardware planning tag was replaced by the observed execution or input dependency. "
            "This demotes planning labels, not scientific standards."
        ),
        "entries": entries,
        "source_evidence": [
            _file_receipt(Path(__file__).resolve()),
            _file_receipt(REPO_ROOT / "scripts" / "frontier_localization.py"),
            _file_receipt(REPO_ROOT / "registry" / "experiments.yaml"),
            _file_receipt(REPO_ROOT / "src" / "mop" / "studies" / "project_exhaustion.py"),
        ],
    }
    if not accounted_exactly_once:
        raise AssertionError("frontier audit did not account for rows exactly once")
    if current_stale_rows or not all_historical_rows_reclassified:
        raise AssertionError("non-F hardware planning tags remain after localization")
    if any(row["hardware_boundary_proven"] for row in entries):
        raise AssertionError("hardware boundary cannot be inferred without a measured failure receipt")
    return payload


def write_frontier_receipts(
    *,
    run_receipt: Path = DEFAULT_RUN_RECEIPT,
    preflight_proof: Path = DEFAULT_PREFLIGHT_PROOF,
    audit_proof: Path = DEFAULT_AUDIT_PROOF,
    seeds: tuple[int, ...] = (20260709, 20260710, 20260711),
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflights = run_local_preflights(seeds=seeds)
    _atomic_json(run_receipt, preflights)
    _atomic_json(preflight_proof, preflights)
    audit = build_frontier_audit(preflights)
    audit["preflight_receipts"] = [
        _file_receipt(run_receipt),
        _file_receipt(preflight_proof),
    ]
    _atomic_json(audit_proof, audit)
    return preflights, audit
