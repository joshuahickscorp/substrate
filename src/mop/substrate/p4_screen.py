"""P4 capability-density response-surface screen.

A pilot screen over twelve registered architecture/budget/data cells of the locally trainable
video-token substrate.  Each cell trains the predictive objective beside its matched random-target
control from one shared per-seed initialization, plus a frozen evaluation of that exact initial
state.  The screen fits a small response surface (log2 parameters, family dummies, corpus density,
temporal extent) with seed fixed effects and seed-level bootstrap intervals, then classifies every
registered coefficient against a preregistered SESOI band.  The screen is a ranking instrument for
follow-up cells only; the promotion block refuses confirmatory claims by construction, and
programmatic-video results never become natural-video or general-capability evidence.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..config import REPO_ROOT
from ..devices import resolve
from .custom_workbench import (
    CorpusSpec,
    ModelSpec,
    ProgrammaticVideoCorpus,
    TinyVideoSubstrate,
    WorkbenchRefused,
    _atomic_json,
    _max_rss_bytes,
    _mean,
    _paired_ci,
    _stable_seed,
    _state_sha256,
    _write_arm_failure,
    build_referent_records,
    dataset_manifest,
    estimated_train_step_flops,
    evaluate_model,
    json_sha256,
    load_arm_model,
    oracle_difficulty_calibration,
    parameter_count,
    token_count,
    train_arm,
)

P4_SCREEN_SCHEMA = "mop-p4-screen/v1"
P4_CELL_SCHEMA = "mop-p4-cell/v1"
P4_SEED_SCHEMA = "mop-p4-seed/v1"

P4_OBJECTIVES = ("predictive", "random_target")
P4_FAMILIES = ("dense", "recurrent", "sparse")
P4_BUDGET_LABELS = ("0.5x", "1x", "2x", "4x", "random")
P4_CORPUS_VARIANTS = ("low", "high")
P4_HEADS = 4
P4_COMPUTE_TOLERANCE = 0.005
P4_COEFFICIENTS = ("log2_params", "family_recurrent", "family_sparse", "corpus_high", "frames_16")
PROMOTION_REFUSAL = "pilot screen; confirmatory claims refused by construction"


@dataclass(frozen=True)
class P4CellSpec:
    cell_id: str
    budget_label: str
    family: str
    dim: int
    depth: int
    experts: int
    frames: int
    corpus_variant: str

    def validate(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must be non-empty")
        if self.budget_label not in P4_BUDGET_LABELS:
            raise ValueError(f"budget_label must be one of {P4_BUDGET_LABELS}")
        if self.family not in P4_FAMILIES:
            raise ValueError(f"family must be one of {P4_FAMILIES}")
        if self.corpus_variant not in P4_CORPUS_VARIANTS:
            raise ValueError(f"corpus_variant must be one of {P4_CORPUS_VARIANTS}")
        if self.frames not in (8, 16):
            raise ValueError("frames must be 8 or 16")
        if self.dim <= 0 or self.dim % P4_HEADS:
            raise ValueError(f"dim must be positive and divisible by the fixed {P4_HEADS} heads")
        if self.depth < 1:
            raise ValueError("depth must be at least one")
        if self.family == "sparse":
            if self.experts < 2:
                raise ValueError("sparse cells need at least two experts")
        elif self.experts != 0:
            raise ValueError("experts must be zero for non-sparse families")


P4_CELLS: tuple[P4CellSpec, ...] = (
    P4CellSpec("C01", "1x", "dense", 128, 4, 0, 8, "low"),
    P4CellSpec("C02", "1x", "recurrent", 140, 6, 0, 16, "high"),
    P4CellSpec("C03", "1x", "sparse", 108, 4, 2, 16, "low"),
    P4CellSpec("C04", "2x", "dense", 184, 5, 0, 16, "high"),
    P4CellSpec("C05", "2x", "recurrent", 200, 8, 0, 8, "low"),
    P4CellSpec("C06", "2x", "sparse", 128, 4, 4, 8, "high"),
    P4CellSpec("C07", "4x", "dense", 256, 6, 0, 16, "low"),
    P4CellSpec("C08", "4x", "recurrent", 304, 8, 0, 8, "high"),
    P4CellSpec("C09", "4x", "sparse", 144, 4, 8, 8, "low"),
    P4CellSpec("C10", "0.5x", "dense", 80, 4, 0, 8, "high"),
    P4CellSpec("C11", "0.5x", "recurrent", 88, 5, 0, 16, "low"),
    P4CellSpec("C12", "0.5x", "sparse", 76, 3, 2, 16, "high"),
)

# Interleave budgets and families so an early wall-budget stop still spans the response surface.
P4_SERIAL_ORDER = ("C01", "C10", "C04", "C07", "C03", "C02", "C06", "C09", "C12", "C11", "C05", "C08")

_CORPUS_VARIANT_FIELDS: dict[str, dict[str, int]] = {
    "low": {"resolution": 256, "factor_a_levels": 4, "factor_b_levels": 4, "replicates": 9, "seed": 4407},
    "high": {"resolution": 256, "factor_a_levels": 6, "factor_b_levels": 6, "replicates": 4, "seed": 4407},
}


def corpus_spec_for_cell(
    cell: P4CellSpec,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> CorpusSpec:
    """Both variants yield 144 clips; density trades factor levels against replicates."""

    fields: dict[str, int] = {**_CORPUS_VARIANT_FIELDS[cell.corpus_variant], "frames": cell.frames}
    if overrides:
        fields.update({str(key): int(value) for key, value in overrides.items()})
    return CorpusSpec(**fields)


class GRUBlocks(nn.Module):
    """Drop-in replacement for the transformer block stack using a stacked GRU."""

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=depth, batch_first=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(hidden)
        return output


class RecurrentVideoSubstrate(TinyVideoSubstrate):
    """TinyVideoSubstrate with the transformer stack replaced by a stacked GRU."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.blocks = GRUBlocks(spec.dim, spec.depth)  # type: ignore[assignment]


class SparseGatedEncoderLayer(nn.Module):
    """Pre-norm attention plus a top-1 gated mixture of the exact dense MLP shape."""

    def __init__(self, dim: int, experts: int):
        super().__init__()
        if experts < 2:
            raise ValueError("a sparse gated layer needs at least two experts")
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, P4_HEADS, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
            for _ in range(experts)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(hidden)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        hidden = hidden + attended
        normed = self.mlp_norm(hidden)
        selected = self.gate(normed).argmax(dim=-1)
        routed = torch.zeros_like(normed)
        # Deterministic per-token top-1 routing: loop the experts with boolean masks at this scale.
        for index, expert in enumerate(self.experts):
            token_mask = selected == index
            if bool(token_mask.any()):
                routed[token_mask] = expert(normed[token_mask])
        return hidden + routed


class SparseGatedBlocks(nn.Module):
    def __init__(self, dim: int, depth: int, experts: int):
        super().__init__()
        self.layers = nn.ModuleList(SparseGatedEncoderLayer(dim, experts) for _ in range(depth))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class SparseGatedVideoSubstrate(TinyVideoSubstrate):
    """TinyVideoSubstrate with the transformer stack replaced by sparse gated layers."""

    def __init__(self, spec: ModelSpec, *, experts: int):
        super().__init__(spec)
        self.blocks = SparseGatedBlocks(spec.dim, spec.depth, experts)  # type: ignore[assignment]


def model_spec_for_cell(cell: P4CellSpec) -> ModelSpec:
    return ModelSpec(
        dim=cell.dim,
        depth=cell.depth,
        heads=P4_HEADS,
        mlp_ratio=4,
        patch_size=32,
        tubelet=2,
        max_resolution=256,
        max_frames=cell.frames,
    )


def build_substrate(cell: P4CellSpec) -> TinyVideoSubstrate:
    cell.validate()
    spec = model_spec_for_cell(cell)
    if cell.family == "dense":
        return TinyVideoSubstrate(spec)
    if cell.family == "recurrent":
        return RecurrentVideoSubstrate(spec)
    return SparseGatedVideoSubstrate(spec, experts=cell.experts)


def realized_parameter_count(cell: P4CellSpec) -> int:
    return parameter_count(build_substrate(cell))


def estimated_train_step_flops_p4(
    data: CorpusSpec,
    cell: P4CellSpec,
    *,
    batch_size: int,
    objective: str,
) -> int:
    """Family-aware analogue of the dense estimator; used only to match arms, never as energy."""

    cell.validate()
    model = model_spec_for_cell(cell)
    if cell.family == "dense":
        return estimated_train_step_flops(data, model, batch_size=batch_size, objective=objective)
    n, d, ff = token_count(data, model), model.dim, model.dim * model.mlp_ratio
    conv = 2 * batch_size * n * d * 3 * model.tubelet * model.patch_size**2
    if cell.family == "recurrent":
        # GRU gates: three input and three hidden matmuls per layer, multiply-add counted as two.
        per_layer = 2 * 6 * batch_size * n * d * d
    else:
        attention = 2 * 4 * batch_size * n * d * d + 4 * batch_size * n * n * d
        mlp = 4 * batch_size * n * d * ff
        gate = 2 * batch_size * n * d * cell.experts
        # Top-1 routing keeps exactly one dense-shaped expert active per token, so the MLP term
        # is unchanged and only the gate logits are added.
        per_layer = attention + mlp + gate
    encoder_forward = conv + model.depth * per_layer
    predictor_forward = 4 * batch_size * n * d * d
    total = 4 * encoder_forward + 3 * predictor_forward
    if objective == "reconstruction":
        total += 2 * batch_size * n * 6 * d
    total += 2 * batch_size * n * d + 2 * model.depth * d * d
    return int(total)


def sample_random_search(seed: int = 4407, n: int = 12) -> list[P4CellSpec]:
    """Deterministic uniform draws over the preregistered search box; ranking input only."""

    generator = torch.Generator().manual_seed(int(seed))
    dims = tuple(range(64, 292, 4))
    expert_choices = (2, 4, 8)
    cells: list[P4CellSpec] = []
    for index in range(int(n)):
        dim = dims[int(torch.randint(0, len(dims), (1,), generator=generator))]
        depth = int(torch.randint(2, 9, (1,), generator=generator))
        family = P4_FAMILIES[int(torch.randint(0, len(P4_FAMILIES), (1,), generator=generator))]
        expert_draw = int(torch.randint(0, len(expert_choices), (1,), generator=generator))
        experts = expert_choices[expert_draw] if family == "sparse" else 0
        frames = (8, 16)[int(torch.randint(0, 2, (1,), generator=generator))]
        variant = P4_CORPUS_VARIANTS[int(torch.randint(0, 2, (1,), generator=generator))]
        cell = P4CellSpec(
            cell_id=f"R{index + 1:02d}",
            budget_label="random",
            family=family,
            dim=dim,
            depth=depth,
            experts=experts,
            frames=frames,
            corpus_variant=variant,
        )
        cell.validate()
        cells.append(cell)
    return cells


def _verify_config_cells(config_cells: Any) -> None:
    """Refuse drift between the config cell table and the registered P4_CELLS constant."""

    if not isinstance(config_cells, list) or not config_cells:
        raise WorkbenchRefused("config cells must be a non-empty list matching the registered table")
    try:
        parsed = [
            P4CellSpec(
                cell_id=str(row["cell_id"]),
                budget_label=str(row["budget_label"]),
                family=str(row["family"]),
                dim=int(row["dim"]),
                depth=int(row["depth"]),
                experts=int(row["experts"]),
                frames=int(row["frames"]),
                corpus_variant=str(row["corpus_variant"]),
            )
            for row in config_cells
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchRefused(f"config cell table is malformed: {exc}") from exc
    if sorted(parsed, key=lambda cell: cell.cell_id) != sorted(P4_CELLS, key=lambda cell: cell.cell_id):
        raise WorkbenchRefused("config cell table drifted from the registered P4_CELLS constant")


def _flops_adapter(cell: P4CellSpec):
    """Adapt the cell-aware estimator to the train_arm flops_estimator call shape."""

    def estimator(
        data_spec: CorpusSpec,
        model_spec: ModelSpec,
        *,
        batch_size: int,
        objective: str,
        teacher_dim: int = 0,
    ) -> int:
        del model_spec, teacher_dim
        return estimated_train_step_flops_p4(data_spec, cell, batch_size=batch_size, objective=objective)

    return estimator


def _p4_compute_match(arm_totals: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Tolerance logic of the workbench compute match, restricted to the two P4 arms."""

    means = {
        objective: _mean([float(value) for value in arm_totals.get(objective, [])])
        for objective in P4_OBJECTIVES
        if arm_totals.get(objective)
    }
    if len(means) < len(P4_OBJECTIVES):
        return {
            "all_ok": False,
            "tolerance_fraction": P4_COMPUTE_TOLERANCE,
            "mean_flops": means,
            "max_fractional_deviation": None,
            "reason": "one or more trained arms are incomplete",
        }
    reference = _mean(list(means.values()))
    deviation = max(abs(value - reference) / max(1.0, reference) for value in means.values())
    return {
        "all_ok": deviation <= P4_COMPUTE_TOLERANCE,
        "tolerance_fraction": P4_COMPUTE_TOLERANCE,
        "mean_flops": means,
        "max_fractional_deviation": deviation,
        "reference_mean_flops": reference,
    }


def _fraction_table(cells: Sequence[P4CellSpec], *, batch_size: int) -> list[dict[str, Any]]:
    params = {cell.cell_id: realized_parameter_count(cell) for cell in cells}
    flops = {
        cell.cell_id: estimated_train_step_flops_p4(
            corpus_spec_for_cell(cell), cell, batch_size=batch_size, objective="predictive"
        )
        for cell in cells
    }
    reference_params = params.get("C01")
    reference_flops = flops.get("C01")
    rows: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda value: value.cell_id):
        rows.append(
            {
                **asdict(cell),
                "realized_parameters": params[cell.cell_id],
                "params_fraction_of_c01": (
                    params[cell.cell_id] / reference_params if reference_params else None
                ),
                "estimated_train_step_flops": flops[cell.cell_id],
                "flops_fraction_of_c01": (flops[cell.cell_id] / reference_flops if reference_flops else None),
            }
        )
    return rows


def _row_features(row: Mapping[str, Any]) -> list[float]:
    return [
        1.0,
        float(row["log2_params"]),
        1.0 if row["family"] == "recurrent" else 0.0,
        1.0 if row["family"] == "sparse" else 0.0,
        1.0 if row["corpus_variant"] == "high" else 0.0,
        1.0 if int(row["frames"]) == 16 else 0.0,
    ]


def _ols_coefficients(
    rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[int],
    cluster_count: int,
) -> list[float] | None:
    columns = 6 + max(0, cluster_count - 1)
    if len(rows) < columns:
        return None
    design = torch.zeros((len(rows), columns), dtype=torch.float64)
    response = torch.zeros((len(rows), 1), dtype=torch.float64)
    for index, (row, cluster) in enumerate(zip(rows, clusters, strict=True)):
        design[index, :6] = torch.tensor(_row_features(row), dtype=torch.float64)
        if cluster > 0:
            design[index, 5 + cluster] = 1.0
        response[index, 0] = float(row["score"])
    if int(torch.linalg.matrix_rank(design)) < columns:
        return None
    solution = torch.linalg.lstsq(design, response).solution.squeeze(1)
    return [float(value) for value in solution]


def _classify_against_sesoi(lo: float | None, hi: float | None, sesoi: float) -> str:
    if lo is None or hi is None:
        return "undetermined"
    if lo > sesoi:
        return "meaningful_positive"
    if hi < -sesoi:
        return "meaningful_negative"
    if lo >= -sesoi and hi <= sesoi:
        return "bounded_within_sesoi"
    return "undetermined"


def fit_response_surface(
    rows: Sequence[Mapping[str, Any]],
    *,
    sesoi: float,
    resamples: int = 1000,
) -> dict[str, Any]:
    """OLS with seed fixed effects plus a seed-level cluster bootstrap for the registered five."""

    problems: list[str] = []
    seeds_used = sorted({int(row["seed"]) for row in rows})
    positions = {seed: index for index, seed in enumerate(seeds_used)}
    clusters = [positions[int(row["seed"])] for row in rows]
    point = _ols_coefficients(rows, clusters, len(seeds_used)) if rows else None
    if point is None:
        problems.append("response surface not estimable: too few observations or rank-deficient design")
    samples: list[list[float]] = []
    if point is not None and len(seeds_used) >= 2:
        rows_by_seed: dict[int, list[Mapping[str, Any]]] = {seed: [] for seed in seeds_used}
        for row in rows:
            rows_by_seed[int(row["seed"])].append(row)
        generator = torch.Generator().manual_seed(_stable_seed("p4-response-surface-bootstrap"))
        draws = torch.randint(0, len(seeds_used), (resamples, len(seeds_used)), generator=generator)
        for resample in range(resamples):
            boot_rows: list[Mapping[str, Any]] = []
            boot_clusters: list[int] = []
            for position in range(len(seeds_used)):
                for row in rows_by_seed[seeds_used[int(draws[resample, position])]]:
                    boot_rows.append(row)
                    boot_clusters.append(position)
            fit = _ols_coefficients(boot_rows, boot_clusters, len(seeds_used))
            if fit is not None:
                samples.append(fit)
        if len(samples) < resamples // 2:
            problems.append("seed-level bootstrap degenerate: under half the resamples were estimable")
    elif point is not None:
        problems.append("seed-level bootstrap not informative: fewer than two seeds")
    coefficients: dict[str, Any] = {}
    if point is not None:
        sample_tensor = (
            torch.tensor([sample[1:6] for sample in samples], dtype=torch.float64) if samples else None
        )
        for offset, name in enumerate(P4_COEFFICIENTS):
            lo: float | None = None
            hi: float | None = None
            if sample_tensor is not None and sample_tensor.shape[0] >= 2:
                lo = float(torch.quantile(sample_tensor[:, offset], 0.025))
                hi = float(torch.quantile(sample_tensor[:, offset], 0.975))
            coefficients[name] = {
                "estimate": point[1 + offset],
                "ci_lo": lo,
                "ci_hi": hi,
                "classification": _classify_against_sesoi(lo, hi, sesoi),
            }
    return {
        "registered_coefficients": list(P4_COEFFICIENTS),
        "estimable": point is not None,
        "intercept": point[0] if point is not None else None,
        "coefficients": coefficients,
        "observations": len(rows),
        "seeds_used": seeds_used,
        "seed_fixed_effects": max(0, len(seeds_used) - 1),
        "sesoi": sesoi,
        "bootstrap": {
            "method": "seed-level cluster bootstrap with percentile intervals",
            "requested_resamples": resamples,
            "effective_resamples": len(samples),
        },
        "problems": problems,
    }


def _load_seed_result(seed_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    path = seed_dir / "seed_result.json"
    if not path.is_file():
        return None
    prior = json.loads(path.read_text())
    if not prior.get("complete"):
        return None
    mismatched = [key for key, value in identity.items() if prior.get(key) != value]
    if mismatched:
        raise WorkbenchRefused(f"completed seed result identity drift at {path}: {mismatched}")
    prior["resumed_from_complete_receipt"] = True
    return prior


def _score(payload: Mapping[str, Any], arm: str) -> float:
    if arm == "frozen_random":
        return float(payload["frozen_random"]["evaluation"]["heldout_combo_score"])
    return float(payload["arms"][arm]["evaluation"]["heldout_combo_score"])


def run_p4_screen(
    config: Mapping[str, Any],
    run_dir: Path,
    device_kind: str,
    *,
    cells: Sequence[P4CellSpec] | None = None,
    corpus_overrides: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run or resume the screen; rerunning the same command resumes from durable receipts."""

    run_dir.mkdir(parents=True, exist_ok=True)
    config_plain = json.loads(json.dumps(dict(config)))
    config_sha = json_sha256(config_plain)
    _atomic_json(run_dir / "resolved_config.json", config_plain)

    if cells is None:
        _verify_config_cells(config_plain.get("cells"))
        by_id = {cell.cell_id: cell for cell in P4_CELLS}
        ordered = [by_id[cell_id] for cell_id in P4_SERIAL_ORDER]
    else:
        ordered = list(cells)
        for cell in ordered:
            cell.validate()
        if len({cell.cell_id for cell in ordered}) != len(ordered):
            raise WorkbenchRefused("injected cell ids must be unique")
    registry_sha = json_sha256([asdict(cell) for cell in ordered])

    device = resolve(device_kind)
    training = config_plain["training"]
    screen = config_plain["screen"]
    seeds = [int(value) for value in training["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be unique")
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    sesoi = float(screen["sesoi"])
    futility_margin = float(screen["futility_margin"])
    disk_floor_bytes = int(float(screen.get("min_free_disk_gb", 40.0)) * 1_000_000_000)
    wall_budget = float(training.get("wall_budget_seconds", 10800.0))
    deadline = time.monotonic() + wall_budget
    started = time.perf_counter()

    fraction_table = _fraction_table(ordered, batch_size=batch_size)
    cell_receipts: dict[str, dict[str, Any]] = {}
    surface_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    stopped_for_wall = False
    stopped_for_disk = False
    stopped_for_required_arm = False
    required_arm_failure: dict[str, Any] | None = None
    c01_reference: float | None = None

    for serial_position, cell in enumerate(ordered):
        if shutil.disk_usage(repo_root).free < disk_floor_bytes:
            stopped_for_disk = True
            break
        if time.monotonic() >= deadline:
            stopped_for_wall = True
            break
        cell_started = time.perf_counter()
        cell_dir = run_dir / "cells" / cell.cell_id
        data_spec = corpus_spec_for_cell(cell, overrides=corpus_overrides)
        records = build_referent_records(data_spec)
        manifest = dataset_manifest(data_spec, records)
        if not manifest["disjoint_referents"]:
            raise WorkbenchRefused(f"{cell.cell_id}: dataset referents overlap across splits")
        _atomic_json(cell_dir / "dataset_manifest.json", manifest)
        corpus = ProgrammaticVideoCorpus(data_spec, records)
        difficulty = oracle_difficulty_calibration(records)
        model_spec = model_spec_for_cell(cell)
        factory = partial(build_substrate, cell)
        flops_for_cell = _flops_adapter(cell)
        params = realized_parameter_count(cell)
        tokens = token_count(data_spec, model_spec)

        seed_payloads: dict[str, dict[str, Any]] = {}
        futility_truncated = False
        futility_evidence: dict[str, Any] | None = None
        for seed_index, seed in enumerate(seeds):
            if shutil.disk_usage(repo_root).free < disk_floor_bytes:
                stopped_for_disk = True
                break
            if time.monotonic() >= deadline:
                stopped_for_wall = True
                break
            seed_dir = cell_dir / f"seed_{seed}"
            torch.manual_seed(_stable_seed(cell.cell_id, seed))
            probe = build_substrate(cell)
            initial_state = {
                key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
            }
            initial_hash = _state_sha256(initial_state)
            del probe
            identity = {
                "cell_id": cell.cell_id,
                "seed": seed,
                "config_sha256": config_sha,
                "data_sha256": manifest["content_sha256"],
                "registry_sha256": registry_sha,
                "initial_state_sha256": initial_hash,
                "requested_steps": steps,
            }
            cached = _load_seed_result(seed_dir, identity)
            if cached is not None:
                seed_payloads[str(seed)] = cached
            else:
                frozen = build_substrate(cell)
                frozen.load_state_dict(initial_state)
                frozen.to(device.device).eval()
                frozen_evaluation = evaluate_model(
                    frozen, corpus, records, device=device, batch_size=eval_batch_size
                )
                del frozen
                seed_payload: dict[str, Any] = {
                    "schema": P4_SEED_SCHEMA,
                    **identity,
                    "frozen_random": {
                        "control": "exact same-architecture, same-initialization frozen encoder",
                        "evaluation": frozen_evaluation,
                    },
                    "arms": {},
                    "complete": False,
                }
                seed_broke = False
                for objective in P4_OBJECTIVES:
                    if time.monotonic() >= deadline:
                        stopped_for_wall = True
                        seed_broke = True
                        break
                    arm_dir = seed_dir / objective
                    try:
                        arm = train_arm(
                            objective=objective,
                            seed=seed,
                            corpus=corpus,
                            records=records,
                            data_spec=data_spec,
                            model_spec=model_spec,
                            initial_state=initial_state,
                            arm_dir=arm_dir,
                            device=device,
                            steps=steps,
                            batch_size=batch_size,
                            learning_rate=float(training["learning_rate"]),
                            weight_decay=float(training["weight_decay"]),
                            mask_ratio=float(training["mask_ratio"]),
                            ema_decay=float(training["ema_decay"]),
                            variance_weight=float(training["variance_weight"]),
                            checkpoint_every=int(training["checkpoint_every"]),
                            config_sha256=config_sha,
                            data_sha256=manifest["content_sha256"],
                            requirements_sha256=registry_sha,
                            deadline=deadline,
                            disk_path=repo_root,
                            disk_floor_bytes=disk_floor_bytes,
                            model_factory=factory,
                            flops_estimator=flops_for_cell,
                        )
                    except WorkbenchRefused as exc:
                        required_arm_failure = _write_arm_failure(
                            arm_dir,
                            seed=seed,
                            objective=objective,
                            error=exc,
                            device=device,
                            kind="scientific-refusal",
                        )
                        seed_payload["arms"][objective] = {
                            "training": {
                                "complete": False,
                                "refused": True,
                                "reason": str(exc),
                                "refusal_receipt": str(arm_dir / "refusal_receipt.json"),
                            }
                        }
                        stopped_for_required_arm = True
                        seed_broke = True
                        break
                    except Exception as exc:
                        _write_arm_failure(
                            arm_dir,
                            seed=seed,
                            objective=objective,
                            error=exc,
                            device=device,
                            kind="unexpected-exception",
                        )
                        raise
                    result: dict[str, Any] = {
                        "training": {
                            "complete": bool(arm["complete"]),
                            "stop_reason": arm.get("stop_reason"),
                            "completed_steps": arm.get("completed_steps"),
                            "estimated_flops_per_step": arm["compute"]["estimated_flops_per_step"],
                            "estimated_total_flops": arm["compute"]["estimated_total_flops"],
                            "wall_seconds": arm["telemetry"]["seconds_this_invocation"],
                            "final_state_sha256": arm.get("final_state_sha256"),
                        }
                    }
                    if arm["complete"]:
                        trained = load_arm_model(arm_dir, model_spec, device=device, model_factory=factory)
                        result["evaluation"] = evaluate_model(
                            trained, corpus, records, device=device, batch_size=eval_batch_size
                        )
                        del trained
                    seed_payload["arms"][objective] = result
                    if not arm["complete"]:
                        if arm.get("stop_reason") == "disk_floor":
                            stopped_for_disk = True
                        else:
                            stopped_for_wall = True
                        seed_broke = True
                        break
                seed_payload["complete"] = not seed_broke and all(
                    seed_payload["arms"].get(objective, {}).get("training", {}).get("complete")
                    and "evaluation" in seed_payload["arms"].get(objective, {})
                    for objective in P4_OBJECTIVES
                )
                seed_payloads[str(seed)] = seed_payload
                if seed_payload["complete"]:
                    _atomic_json(seed_dir / "seed_result.json", seed_payload)
                if seed_broke:
                    break
            if seed_index == 2 and len(seeds) > 3 and c01_reference is not None:
                maybe_three = [seed_payloads.get(str(value)) for value in seeds[:3]]
                first_three = [payload for payload in maybe_three if payload is not None]
                if len(first_three) == 3 and all(payload.get("complete") for payload in first_three):
                    contrast_mean = _mean(
                        [
                            _score(payload, "predictive") - _score(payload, "frozen_random")
                            for payload in first_three
                        ]
                    )
                    raw_mean = _mean([_score(payload, "predictive") for payload in first_three])
                    if contrast_mean <= 0.0 and raw_mean < c01_reference - futility_margin:
                        futility_truncated = True
                        futility_evidence = {
                            "mean_predictive_minus_frozen": contrast_mean,
                            "mean_predictive_score": raw_mean,
                            "c01_reference_mean": c01_reference,
                            "futility_margin": futility_margin,
                            "seeds_kept": seeds[:3],
                        }
                        break

        expected_seeds = seeds[:3] if futility_truncated else seeds
        completed_keys = [
            str(value) for value in expected_seeds if seed_payloads.get(str(value), {}).get("complete")
        ]
        cell_complete = len(completed_keys) == len(expected_seeds)
        predictive_scores = [_score(seed_payloads[key], "predictive") for key in completed_keys]
        random_scores = [_score(seed_payloads[key], "random_target") for key in completed_keys]
        frozen_scores = [_score(seed_payloads[key], "frozen_random") for key in completed_keys]
        arm_totals = {
            objective: [
                int(seed_payloads[key]["arms"][objective]["training"]["estimated_total_flops"])
                for key in completed_keys
            ]
            for objective in P4_OBJECTIVES
        }
        per_step_flops = {
            objective: estimated_train_step_flops_p4(
                data_spec, cell, batch_size=batch_size, objective=objective
            )
            for objective in P4_OBJECTIVES
        }
        compute_match = _p4_compute_match(arm_totals)
        cell_problems: list[str] = []
        if not cell_complete:
            cell_problems.append("cell incomplete this invocation; rerun the same command to resume")
        if not compute_match["all_ok"]:
            cell_problems.append("predictive and random_target arms are not compute matched")
        if not difficulty.get("clears_floor"):
            cell_problems.append("programmatic difficulty oracle did not clear its floor")
        cell_receipt = {
            "schema": P4_CELL_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "claim_scope": (
                "capability-density pilot cell on deterministic programmatic video; "
                "not natural-video or general-capability evidence"
            ),
            "cell": asdict(cell),
            "serial_position": serial_position,
            "realized_parameters": params,
            "token_count": tokens,
            "corpus": {
                "spec": asdict(data_spec),
                "rows": len(records),
                "content_sha256": manifest["content_sha256"],
            },
            "difficulty_calibration": difficulty,
            "flops": {
                "per_arm_per_step": per_step_flops,
                "per_arm_total": {key: sum(values) for key, values in arm_totals.items()},
            },
            "compute_match": compute_match,
            "seed_results": seed_payloads,
            "scores": {
                "predictive": _paired_ci(predictive_scores),
                "random_target": _paired_ci(random_scores),
                "frozen_random": _paired_ci(frozen_scores),
            },
            "paired_contrasts": {
                "predictive_minus_frozen": _paired_ci(
                    [p - f for p, f in zip(predictive_scores, frozen_scores, strict=True)]
                ),
                "predictive_minus_random_target": _paired_ci(
                    [p - r for p, r in zip(predictive_scores, random_scores, strict=True)]
                ),
            },
            "futility_truncated": futility_truncated,
            "futility_evidence": futility_evidence,
            "complete": cell_complete,
            "telemetry": {
                "max_rss_bytes": _max_rss_bytes(),
                "wall_seconds_this_invocation": time.perf_counter() - cell_started,
                "device": device.kind,
            },
            "problems": cell_problems,
            "all_ok": not cell_problems,
        }
        _atomic_json(cell_dir / "cell_receipt.json", cell_receipt)
        cell_receipts[cell.cell_id] = cell_receipt
        problems.extend(f"{cell.cell_id}: {problem}" for problem in cell_problems)
        if cell.cell_id == "C01" and cell_complete and predictive_scores:
            c01_reference = _mean(predictive_scores)
        for key in completed_keys:
            surface_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "seed": int(key),
                    "score": _score(seed_payloads[key], "predictive"),
                    "log2_params": math.log2(params),
                    "family": cell.family,
                    "corpus_variant": cell.corpus_variant,
                    "frames": cell.frames,
                }
            )
        if stopped_for_wall or stopped_for_disk or stopped_for_required_arm:
            break

    complete = (
        not (stopped_for_wall or stopped_for_disk or stopped_for_required_arm)
        and all(cell_receipts.get(cell.cell_id, {}).get("complete") for cell in ordered)
    )
    if required_arm_failure is not None:
        problems.append(
            "required arm refused and aborted the screen: "
            f"seed={required_arm_failure['seed']} objective={required_arm_failure['objective']} "
            f"reason={required_arm_failure['reason']}"
        )
    if not complete:
        problems.append("screen incomplete; rerun the same command to resume")
    surface = fit_response_surface(surface_rows, sesoi=sesoi)
    problems.extend(surface["problems"])
    receipt = {
        "schema": P4_SCREEN_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_scope": (
            "capability-density pilot screen on deterministic programmatic video; "
            "not natural-video or general-capability evidence"
        ),
        "config_sha256": config_sha,
        "cell_registry_sha256": registry_sha,
        "profile": config_plain.get("profile"),
        "serial_order": [cell.cell_id for cell in ordered],
        "seeds": seeds,
        "fraction_table": fraction_table,
        "cells": {
            cell_id: {
                "complete": row["complete"],
                "futility_truncated": row["futility_truncated"],
                "realized_parameters": row["realized_parameters"],
                "seeds_completed": row["scores"]["predictive"]["n"],
                "scores": row["scores"],
                "paired_contrasts": row["paired_contrasts"],
                "compute_match_ok": row["compute_match"]["all_ok"],
                "all_ok": row["all_ok"],
            }
            for cell_id, row in cell_receipts.items()
        },
        "response_surface": surface,
        "promotion": {
            "confirmatory_promotable": False,
            "refused_by_construction": True,
            "reason": PROMOTION_REFUSAL,
            "scope_boundary": (
                "coefficients rank follow-up cells on this deterministic programmatic task only; "
                "they license no natural-video or general-capability claim"
            ),
        },
        "complete": complete,
        "resumable": not complete,
        "stopped_for_wall_budget": stopped_for_wall,
        "stopped_for_disk_floor": stopped_for_disk,
        "stopped_for_required_arm_refusal": stopped_for_required_arm,
        "required_arm_failure": required_arm_failure,
        "problems": problems,
        "all_ok": not problems,
        "resource_telemetry": {
            "wall_budget_seconds": wall_budget,
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "max_rss_bytes": _max_rss_bytes(),
            "device": device.kind,
            "free_disk_bytes_at_end": shutil.disk_usage(repo_root).free,
            "disk_floor_bytes": disk_floor_bytes,
        },
    }
    _atomic_json(run_dir / "p4_screen_receipt.json", receipt)
    return receipt
