"""P5 exact-versus-factorized context pilot.

A staged pilot over twelve registered (frame count x context mechanism) cells of the locally
trainable video-token substrate.  Every mechanism trains the predictive objective from a shared
per-seed trunk initialization at matched parameters and matched total estimated FLOPs against the
exact global-attention reference, beside a frozen evaluation of its own exact initial state.
Stage one runs seed 0 across all frame counts (descending), gates on f64 trainability, and marks
each frame count on or off ceiling; stage two runs the remaining seeds only on off-ceiling frame
counts with a three-seed futility truncation per frame count.  The pilot ranks context mechanisms
on deterministic programmatic video only; the promotion block refuses confirmatory claims by
construction and states that category 9 is impossible from this instrument.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..config import REPO_ROOT
from ..devices import resolve
from ..studies.p9_accounting import WorkloadAccountant
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
from .p4_screen import (
    GRUBlocks as GRUBlocks,  # re-exported: the recurrent stack is the exact P4 implementation
)
from .p4_screen import (
    RecurrentVideoSubstrate,
    _classify_against_sesoi,
    _load_seed_result,
)

P5_SCREEN_SCHEMA = "mop-p5-context-screen/v1"
P5_CELL_SCHEMA = "mop-p5-context-cell/v1"
P5_SEED_SCHEMA = "mop-p5-context-seed/v1"

P5_FRAME_COUNTS = (16, 32, 64)
P5_MECHANISMS = ("exact_global", "window_local", "recurrent", "hierarchical_pooled")
P5_OBJECTIVE = "predictive"
P5_DIM = 128
P5_HEADS = 4
P5_MLP_RATIO = 4
P5_PATCH_SIZE = 32
P5_TUBELET = 2
P5_RESOLUTION = 256
P5_DEPTH = 4
P5_RECURRENT_DEPTH = 8
P5_WINDOW_TOKENS = 512
P5_FLOP_MATCH_TOLERANCE = 0.02
P5_RECURRENT_PARAM_TOLERANCE = 0.005
P5_TRAINABILITY_MARGIN = 0.05
P5_CEILING_CHANCE_OFFSET = 0.05
P5_CEILING_UPPER = 0.95
PROMOTION_REFUSAL = "context pilot; confirmatory claims refused by construction"
CLAIM_SCOPE = (
    "exact-versus-factorized context pilot on deterministic programmatic video; "
    "not natural-video, memory-rung, or general-capability evidence"
)


@dataclass(frozen=True)
class P5CellSpec:
    frames: int
    mechanism: str

    @property
    def cell_id(self) -> str:
        return f"f{self.frames}_{self.mechanism}"

    def validate(self) -> None:
        if self.frames not in P5_FRAME_COUNTS:
            raise ValueError(f"frames must be one of {P5_FRAME_COUNTS}")
        if self.mechanism not in P5_MECHANISMS:
            raise ValueError(f"mechanism must be one of {P5_MECHANISMS}")


# Serial registry order: frames descending so the most expensive cells run first and the seed-0
# trainability gate reads the f64 exact arm before any cheaper work; mechanisms in fixed order.
P5_CELLS: tuple[P5CellSpec, ...] = tuple(
    P5CellSpec(frames, mechanism)
    for frames in sorted(P5_FRAME_COUNTS, reverse=True)
    for mechanism in P5_MECHANISMS
)


def _windows(hidden: torch.Tensor, window: int) -> torch.Tensor:
    """Reshape [batch, tokens, dim] into contiguous non-overlapping windows in token order.

    A sequence at or below one window stays a single window, the degenerate-global case the f16
    cells exercise; ragged token counts are refused rather than silently padded.
    """

    batch, tokens, dim = hidden.shape
    effective = min(window, tokens)
    if effective < 1 or tokens % effective:
        raise ValueError(f"{tokens} tokens do not divide into {effective}-token windows")
    return hidden.reshape(batch * (tokens // effective), effective, dim)


def _unwindows(hidden: torch.Tensor, batch: int) -> torch.Tensor:
    """Inverse of ``_windows`` for the same batch size; restores [batch, tokens, dim] token order."""

    windows, window, dim = hidden.shape
    if batch < 1 or windows % batch:
        raise ValueError(f"{windows} windows do not regroup into batch {batch}")
    return hidden.reshape(batch, (windows // batch) * window, dim)


def _encoder_layer(dim: int, heads: int, mlp_ratio: int) -> nn.TransformerEncoderLayer:
    """The exact TransformerEncoderLayer shape TinyVideoSubstrate stacks in its dense trunk."""

    return nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=dim * mlp_ratio,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class WindowedBlocks(nn.Module):
    """Dense-shaped encoder layers applied per contiguous non-overlapping 512-token window."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: int, window: int = P5_WINDOW_TOKENS):
        super().__init__()
        self.window = window
        self.layers = nn.ModuleList(_encoder_layer(dim, heads, mlp_ratio) for _ in range(depth))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[0]
        for layer in self.layers:
            hidden = _unwindows(layer(_windows(hidden, self.window)), batch)
        return hidden


class WindowedVideoSubstrate(TinyVideoSubstrate):
    """TinyVideoSubstrate with the dense stack replaced by windowed local attention."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.blocks = WindowedBlocks(spec.dim, spec.depth, spec.heads, spec.mlp_ratio)  # type: ignore[assignment]


class HierarchicalPooledBlocks(nn.Module):
    """Windowed pass plus a same-weight full-attention pass over per-window mean summaries.

    Each layer runs the windowed pass exactly as WindowedBlocks with the same layer weights, mean
    pools every window to one summary token, runs the same layer over the summaries with full
    attention, then broadcasts the transformed summaries back to token length and adds them to the
    windowed output.  Parameter count is therefore identical to the dense and windowed stacks.
    """

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: int, window: int = P5_WINDOW_TOKENS):
        super().__init__()
        self.window = window
        self.layers = nn.ModuleList(_encoder_layer(dim, heads, mlp_ratio) for _ in range(depth))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = hidden.shape
        window = min(self.window, tokens)
        for layer in self.layers:
            windowed = _unwindows(layer(_windows(hidden, self.window)), batch)
            summaries = windowed.reshape(batch, tokens // window, window, dim).mean(dim=2)
            hidden = windowed + layer(summaries).repeat_interleave(window, dim=1)
        return hidden


class HierarchicalVideoSubstrate(TinyVideoSubstrate):
    """TinyVideoSubstrate with the dense stack replaced by windowed plus pooled-summary attention."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.blocks = HierarchicalPooledBlocks(  # type: ignore[assignment]
            spec.dim, spec.depth, spec.heads, spec.mlp_ratio
        )


def model_spec_for_cell(cell: P5CellSpec, *, overrides: Mapping[str, Any] | None = None) -> ModelSpec:
    fields: dict[str, int] = {
        "dim": P5_DIM,
        "depth": P5_RECURRENT_DEPTH if cell.mechanism == "recurrent" else P5_DEPTH,
        "heads": P5_HEADS,
        "mlp_ratio": P5_MLP_RATIO,
        "patch_size": P5_PATCH_SIZE,
        "tubelet": P5_TUBELET,
        "max_resolution": P5_RESOLUTION,
        "max_frames": cell.frames,
    }
    if overrides:
        fields.update({str(key): int(value) for key, value in overrides.items()})
    return ModelSpec(**fields)


def corpus_spec_for_frames(frames: int, *, overrides: Mapping[str, Any] | None = None) -> CorpusSpec:
    """One corpus per frame count; every mechanism at that frame count shares it exactly."""

    fields: dict[str, int] = {
        "resolution": P5_RESOLUTION,
        "frames": frames,
        "factor_a_levels": 4,
        "factor_b_levels": 4,
        "replicates": 9,
        "seed": 4407,
    }
    if overrides:
        fields.update({str(key): int(value) for key, value in overrides.items()})
    return CorpusSpec(**fields)


def build_p5_substrate(
    cell: P5CellSpec,
    *,
    model_overrides: Mapping[str, Any] | None = None,
) -> TinyVideoSubstrate:
    cell.validate()
    spec = model_spec_for_cell(cell, overrides=model_overrides)
    if cell.mechanism == "exact_global":
        return TinyVideoSubstrate(spec)
    if cell.mechanism == "window_local":
        return WindowedVideoSubstrate(spec)
    if cell.mechanism == "recurrent":
        return RecurrentVideoSubstrate(spec)
    return HierarchicalVideoSubstrate(spec)


def estimated_train_step_flops_p5(
    data: CorpusSpec,
    cell: P5CellSpec,
    *,
    batch_size: int,
    objective: str,
    model_overrides: Mapping[str, Any] | None = None,
) -> int:
    """Mechanism-aware analogue of the dense estimator; used only to match arms, never as energy."""

    cell.validate()
    model = model_spec_for_cell(cell, overrides=model_overrides)
    if cell.mechanism == "exact_global":
        return estimated_train_step_flops(data, model, batch_size=batch_size, objective=objective)
    n, d, ff = token_count(data, model), model.dim, model.dim * model.mlp_ratio
    conv = 2 * batch_size * n * d * 3 * model.tubelet * model.patch_size**2
    if cell.mechanism == "recurrent":
        # GRU gates: three input and three hidden matmuls per layer, multiply-add counted as two.
        per_layer = 2 * 6 * batch_size * n * d * d
    else:
        window = min(P5_WINDOW_TOKENS, n)
        # Windowed attention replaces the dense 4*B*n*n*d score term with 4*B*n*window*d.
        attention = 2 * 4 * batch_size * n * d * d + 4 * batch_size * n * window * d
        mlp = 4 * batch_size * n * d * ff
        per_layer = attention + mlp
        if cell.mechanism == "hierarchical_pooled":
            summaries = max(1, n // window)
            summary_attention = (
                2 * 4 * batch_size * summaries * d * d + 4 * batch_size * summaries * summaries * d
            )
            summary_mlp = 4 * batch_size * summaries * d * ff
            pooling_scatter = 4 * batch_size * n * d
            per_layer += summary_attention + summary_mlp + pooling_scatter
    encoder_forward = conv + model.depth * per_layer
    predictor_forward = 4 * batch_size * n * d * d
    total = 4 * encoder_forward + 3 * predictor_forward
    if objective == "reconstruction":
        total += 2 * batch_size * n * 6 * d
    total += 2 * batch_size * n * d + 2 * model.depth * d * d
    return int(total)


def solve_matched_steps(
    dense_steps: int,
    dense_flops: int,
    arm_flops: int,
    checkpoint_every: int,
    step_granularity: int = 5,
) -> dict[str, Any]:
    """Step count whose estimated total FLOPs sit closest to the dense reference.

    Preregistration amendment (made before any pilot seed ran): matching rounds to multiples of
    ``step_granularity`` (default 5), not ``checkpoint_every``.  Checkpoint-multiple rounding left
    the recurrent arms outside the 0.02 band at every granularity-25 grid point (0.0269 at f64,
    0.0352 at f16), a rounding artifact rather than a compute mismatch.  Checkpoints still land
    every ``checkpoint_every`` steps; the matched-updates secondary endpoint still reads the
    checkpoint whose completed steps equal the dense count.  A deviation above
    ``P5_FLOP_MATCH_TOLERANCE`` refuses the matched-compute claim (``matched_ok`` false); the
    caller must record that refusal as a receipt problem rather than report matched compute.
    """

    values = (
        int(dense_steps),
        int(dense_flops),
        int(arm_flops),
        int(checkpoint_every),
        int(step_granularity),
    )
    if min(values) <= 0:
        raise ValueError("solve_matched_steps needs positive steps, FLOPs, and granularities")
    target = int(dense_steps) * int(dense_flops)
    # The matching grid never exceeds the checkpoint interval, so bounded smokes with
    # checkpoint_every 1 keep exact tiny step counts while the pilot grid stays at 5.
    grain = min(int(step_granularity), int(checkpoint_every))
    steps = max(1, round(target / int(arm_flops) / grain)) * grain
    deviation = abs(steps * int(arm_flops) - target) / target
    return {
        "steps": int(steps),
        "target_total_flops": target,
        "arm_total_flops": int(steps) * int(arm_flops),
        "fractional_deviation": deviation,
        "step_granularity": grain,
        "checkpoint_every": int(checkpoint_every),
        "tolerance_fraction": P5_FLOP_MATCH_TOLERANCE,
        "matched_ok": deviation <= P5_FLOP_MATCH_TOLERANCE,
    }


def _verify_config_cells(config_cells: Any) -> None:
    """Refuse drift between the config cell table and the registered P5_CELLS constant."""

    if not isinstance(config_cells, list) or not config_cells:
        raise WorkbenchRefused("config cells must be a non-empty list matching the registered table")
    try:
        parsed = [
            P5CellSpec(frames=int(row["frames"]), mechanism=str(row["mechanism"])) for row in config_cells
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchRefused(f"config cell table is malformed: {exc}") from exc
    key = _cell_sort_key
    if sorted(parsed, key=key) != sorted(P5_CELLS, key=key):
        raise WorkbenchRefused("config cell table drifted from the registered P5_CELLS constant")


def _cell_sort_key(cell: P5CellSpec) -> tuple[int, str]:
    return (cell.frames, cell.mechanism)


def _flops_adapter(cell: P5CellSpec, model_overrides: Mapping[str, Any] | None) -> Callable[..., int]:
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
        return estimated_train_step_flops_p5(
            data_spec, cell, batch_size=batch_size, objective=objective, model_overrides=model_overrides
        )

    return estimator


def _verify_parameter_identity(
    cells: Sequence[P5CellSpec],
    *,
    model_overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Refuse the pilot unless the transformer mechanisms are parameter identical per frame count
    and the recurrent stack sits within the 0.5 percent envelope of that reference."""

    rows: list[dict[str, Any]] = []
    for frames in sorted({cell.frames for cell in cells}, reverse=True):
        counts = {
            cell.mechanism: parameter_count(build_p5_substrate(cell, model_overrides=model_overrides))
            for cell in cells
            if cell.frames == frames
        }
        transformer = [
            counts[mechanism]
            for mechanism in ("exact_global", "window_local", "hierarchical_pooled")
            if mechanism in counts
        ]
        if len(set(transformer)) > 1:
            raise WorkbenchRefused(
                f"frames {frames}: transformer mechanisms are not parameter identical: {counts}"
            )
        deviation: float | None = None
        if "recurrent" in counts and transformer:
            deviation = abs(counts["recurrent"] - transformer[0]) / transformer[0]
            if deviation > P5_RECURRENT_PARAM_TOLERANCE:
                raise WorkbenchRefused(
                    f"frames {frames}: recurrent parameter deviation {deviation:.6f} exceeds "
                    f"the {P5_RECURRENT_PARAM_TOLERANCE} envelope"
                )
        rows.append(
            {
                "frames": frames,
                "parameters": counts,
                "recurrent_fractional_deviation": deviation,
                "tolerance_fraction": P5_RECURRENT_PARAM_TOLERANCE,
            }
        )
    return rows


def _heldout(payload: Mapping[str, Any], mechanism: str) -> float:
    return float(payload["mechanisms"][mechanism]["evaluation"]["heldout_combo_score"])


def _frozen_heldout(payload: Mapping[str, Any], mechanism: str) -> float:
    return float(payload["mechanisms"][mechanism]["frozen"]["evaluation"]["heldout_combo_score"])


def run_p5_pilot(
    config: Mapping[str, Any],
    run_dir: Path,
    device_kind: str,
    *,
    cells: Sequence[P5CellSpec] | None = None,
    corpus_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run or resume the pilot; rerunning the same command resumes from durable receipts.

    Serial order: seeds outermost, frame counts descending, mechanisms in the fixed registered
    order.  After seed 0 completes, each frame count is marked on or off ceiling and the f64
    trainability gate is evaluated; seeds 1 and later run only on off-ceiling frame counts, with a
    three-seed futility truncation per frame count.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    config_plain = json.loads(json.dumps(dict(config)))
    config_sha = json_sha256(config_plain)
    _atomic_json(run_dir / "resolved_config.json", config_plain)

    if cells is None:
        _verify_config_cells(config_plain.get("cells"))
        ordered = list(P5_CELLS)
    else:
        ordered = list(cells)
        for cell in ordered:
            cell.validate()
        if len(set(ordered)) != len(ordered):
            raise WorkbenchRefused("injected cells must be unique (frames, mechanism) pairs")
    registry_sha = json_sha256([asdict(cell) for cell in ordered])
    parameter_table = _verify_parameter_identity(ordered, model_overrides=model_overrides)

    device = resolve(device_kind)
    training = config_plain["training"]
    screen = config_plain["screen"]
    seeds = [int(value) for value in training["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be unique")
    dense_steps = int(training.get("dense_steps", 200))
    checkpoint_every = int(training["checkpoint_every"])
    batch_size = int(training["batch_size"])
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    sesoi = float(screen["sesoi"])
    futility_margin = float(screen["futility_margin"])
    disk_floor_bytes = int(float(screen.get("min_free_disk_gb", 40.0)) * 1_000_000_000)
    wall_budget = float(training.get("wall_budget_seconds", 10800.0))
    deadline = time.monotonic() + wall_budget
    started = time.perf_counter()

    frames_order = sorted({cell.frames for cell in ordered}, reverse=True)
    mechanisms_by_frames = {
        frames: [
            mechanism
            for mechanism in P5_MECHANISMS
            if any(cell.frames == frames and cell.mechanism == mechanism for cell in ordered)
        ]
        for frames in frames_order
    }
    cell_by_key = {(cell.frames, cell.mechanism): cell for cell in ordered}

    contexts: dict[int, dict[str, Any]] = {}

    def _context(frames: int) -> dict[str, Any]:
        if frames in contexts:
            return contexts[frames]
        data_spec = corpus_spec_for_frames(frames, overrides=corpus_overrides)
        records = build_referent_records(data_spec)
        manifest = dataset_manifest(data_spec, records)
        if not manifest["disjoint_referents"]:
            raise WorkbenchRefused(f"frames {frames}: dataset referents overlap across splits")
        frame_dir = run_dir / "frames" / f"f{frames}"
        _atomic_json(frame_dir / "dataset_manifest.json", manifest)
        difficulty = oracle_difficulty_calibration(records)
        _atomic_json(frame_dir / "difficulty_calibration.json", difficulty)
        dense_flops = estimated_train_step_flops_p5(
            data_spec,
            P5CellSpec(frames, "exact_global"),
            batch_size=batch_size,
            objective=P5_OBJECTIVE,
            model_overrides=model_overrides,
        )
        per_step: dict[str, int] = {}
        matched: dict[str, dict[str, Any]] = {}
        for mechanism in mechanisms_by_frames[frames]:
            flops = estimated_train_step_flops_p5(
                data_spec,
                cell_by_key[(frames, mechanism)],
                batch_size=batch_size,
                objective=P5_OBJECTIVE,
                model_overrides=model_overrides,
            )
            per_step[mechanism] = flops
            if mechanism == "exact_global":
                matched[mechanism] = {
                    "steps": dense_steps,
                    "target_total_flops": dense_steps * dense_flops,
                    "arm_total_flops": dense_steps * dense_flops,
                    "fractional_deviation": 0.0,
                    "tolerance_fraction": P5_FLOP_MATCH_TOLERANCE,
                    "matched_ok": True,
                    "dense_reference": True,
                }
            else:
                matched[mechanism] = solve_matched_steps(dense_steps, dense_flops, flops, checkpoint_every)
        contexts[frames] = {
            "dir": frame_dir,
            "data_spec": data_spec,
            "records": records,
            "manifest": manifest,
            "corpus": ProgrammaticVideoCorpus(data_spec, records),
            "difficulty": difficulty,
            "dense_flops_per_step": dense_flops,
            "per_step_flops": per_step,
            "matched": matched,
        }
        return contexts[frames]

    seed_payloads: dict[int, dict[str, dict[str, Any]]] = {frames: {} for frames in frames_order}
    off_ceiling: dict[int, bool | None] = {}
    truncated: dict[int, dict[str, Any] | None] = dict.fromkeys(frames_order)
    trainability_gate: dict[str, Any] = {
        "applies": (64, "exact_global") in cell_by_key,
        "margin": P5_TRAINABILITY_MARGIN,
        "evaluated": False,
        "failed": False,
    }
    problems: list[str] = []
    stopped_for_wall = False
    stopped_for_disk = False
    stopped_for_required_arm = False
    required_arm_failure: dict[str, Any] | None = None

    for seed_index, seed in enumerate(seeds):
        if trainability_gate["failed"]:
            break
        if shutil.disk_usage(repo_root).free < disk_floor_bytes:
            stopped_for_disk = True
            break
        if time.monotonic() >= deadline:
            stopped_for_wall = True
            break
        accountant = WorkloadAccountant(workload=f"p5_context_seed_{seed}", watch_paths={"run_dir": run_dir})
        seed_broke = False
        for frames in frames_order:
            if seed_index >= 1 and off_ceiling.get(frames) is not True:
                continue
            if seed_index >= 3 and truncated.get(frames):
                continue
            if shutil.disk_usage(repo_root).free < disk_floor_bytes:
                stopped_for_disk = True
                seed_broke = True
                break
            if time.monotonic() >= deadline:
                stopped_for_wall = True
                seed_broke = True
                break
            with accountant.phase("input"):
                context = _context(frames)
            mechanisms = mechanisms_by_frames[frames]
            seed_dir = context["dir"] / f"seed_{seed}"
            with accountant.phase("model"):
                # Shared trunk per (seed, frames): every mechanism loads the exact_global model's
                # patch_embed, mask_token, position, norm, and predictor state, while its blocks
                # initialize from the mechanism-specific stable seed.
                torch.manual_seed(_stable_seed("p5", frames, seed))
                reference = build_p5_substrate(
                    P5CellSpec(frames, "exact_global"), model_overrides=model_overrides
                )
                trunk_state = {
                    key: value.detach().cpu().clone()
                    for key, value in reference.state_dict().items()
                    if not key.startswith("blocks.")
                }
                del reference
                initial_states: dict[str, dict[str, torch.Tensor]] = {}
                for mechanism in mechanisms:
                    torch.manual_seed(_stable_seed("p5", frames, mechanism, seed))
                    probe = build_p5_substrate(
                        cell_by_key[(frames, mechanism)], model_overrides=model_overrides
                    )
                    state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
                    state.update(trunk_state)
                    initial_states[mechanism] = state
                    del probe
            identity = {
                "frames": frames,
                "seed": seed,
                "config_sha256": config_sha,
                "data_sha256": context["manifest"]["content_sha256"],
                "registry_sha256": registry_sha,
                "dense_steps": dense_steps,
                "requested_steps": {
                    mechanism: int(context["matched"][mechanism]["steps"]) for mechanism in mechanisms
                },
                "initial_state_sha256": {
                    mechanism: _state_sha256(initial_states[mechanism]) for mechanism in mechanisms
                },
            }
            cached = _load_seed_result(seed_dir, identity)
            if cached is not None:
                seed_payloads[frames][str(seed)] = cached
                continue
            seed_payload: dict[str, Any] = {
                "schema": P5_SEED_SCHEMA,
                **identity,
                "mechanisms": {},
                "complete": False,
            }
            for mechanism in mechanisms:
                if time.monotonic() >= deadline:
                    stopped_for_wall = True
                    seed_broke = True
                    break
                cell = cell_by_key[(frames, mechanism)]
                matched = context["matched"][mechanism]
                arm_dir = seed_dir / mechanism
                factory = partial(build_p5_substrate, cell, model_overrides=model_overrides)
                with accountant.phase("model"):
                    frozen = factory()
                    frozen.load_state_dict(initial_states[mechanism])
                    frozen.to(device.device).eval()
                with accountant.phase("evaluation"):
                    frozen_evaluation = evaluate_model(
                        frozen,
                        context["corpus"],
                        context["records"],
                        device=device,
                        batch_size=eval_batch_size,
                    )
                del frozen
                result: dict[str, Any] = {
                    "initial_state_sha256": identity["initial_state_sha256"][mechanism],
                    "matched": matched,
                    "frozen": {
                        "control": "exact same-architecture, same-initialization frozen encoder",
                        "evaluation": frozen_evaluation,
                    },
                }
                try:
                    with accountant.phase("model"):
                        arm = train_arm(
                            objective=P5_OBJECTIVE,
                            seed=seed,
                            corpus=context["corpus"],
                            records=context["records"],
                            data_spec=context["data_spec"],
                            model_spec=model_spec_for_cell(cell, overrides=model_overrides),
                            initial_state=initial_states[mechanism],
                            arm_dir=arm_dir,
                            device=device,
                            steps=int(matched["steps"]),
                            batch_size=batch_size,
                            learning_rate=float(training["learning_rate"]),
                            weight_decay=float(training["weight_decay"]),
                            mask_ratio=float(training["mask_ratio"]),
                            ema_decay=float(training["ema_decay"]),
                            variance_weight=float(training["variance_weight"]),
                            checkpoint_every=checkpoint_every,
                            config_sha256=config_sha,
                            data_sha256=context["manifest"]["content_sha256"],
                            requirements_sha256=registry_sha,
                            deadline=deadline,
                            disk_path=repo_root,
                            disk_floor_bytes=disk_floor_bytes,
                            model_factory=factory,
                            flops_estimator=_flops_adapter(cell, model_overrides),
                        )
                except WorkbenchRefused as exc:
                    required_arm_failure = _write_arm_failure(
                        arm_dir,
                        seed=seed,
                        objective=P5_OBJECTIVE,
                        error=exc,
                        device=device,
                        kind="scientific-refusal",
                    )
                    required_arm_failure["cell_id"] = cell.cell_id
                    result["training"] = {
                        "complete": False,
                        "refused": True,
                        "reason": str(exc),
                        "refusal_receipt": str(arm_dir / "refusal_receipt.json"),
                    }
                    seed_payload["mechanisms"][mechanism] = result
                    stopped_for_required_arm = True
                    seed_broke = True
                    break
                except Exception as exc:
                    _write_arm_failure(
                        arm_dir,
                        seed=seed,
                        objective=P5_OBJECTIVE,
                        error=exc,
                        device=device,
                        kind="unexpected-exception",
                    )
                    raise
                result["training"] = {
                    "complete": bool(arm["complete"]),
                    "stop_reason": arm.get("stop_reason"),
                    "completed_steps": arm.get("completed_steps"),
                    "estimated_flops_per_step": arm["compute"]["estimated_flops_per_step"],
                    "estimated_total_flops": arm["compute"]["estimated_total_flops"],
                    "wall_seconds": arm["telemetry"]["seconds_this_invocation"],
                    "final_state_sha256": arm.get("final_state_sha256"),
                }
                if arm["complete"]:
                    with accountant.phase("checkpoint"):
                        trained = load_arm_model(
                            arm_dir,
                            model_spec_for_cell(cell, overrides=model_overrides),
                            device=device,
                            model_factory=factory,
                        )
                    with accountant.phase("evaluation"):
                        result["evaluation"] = evaluate_model(
                            trained,
                            context["corpus"],
                            context["records"],
                            device=device,
                            batch_size=eval_batch_size,
                        )
                    del trained
                seed_payload["mechanisms"][mechanism] = result
                if not arm["complete"]:
                    if arm.get("stop_reason") == "disk_floor":
                        stopped_for_disk = True
                    else:
                        stopped_for_wall = True
                    seed_broke = True
                    break
            seed_payload["complete"] = not seed_broke and all(
                seed_payload["mechanisms"].get(mechanism, {}).get("training", {}).get("complete")
                and "evaluation" in seed_payload["mechanisms"].get(mechanism, {})
                for mechanism in mechanisms
            )
            seed_payloads[frames][str(seed)] = seed_payload
            if seed_payload["complete"]:
                _atomic_json(seed_dir / "seed_result.json", seed_payload)
            if seed_broke:
                break
        accountant.write(run_dir / "accounting" / f"{seed}.json")

        if seed_index == 0:
            for frames in frames_order:
                payload = seed_payloads[frames].get(str(seed))
                if (
                    payload is None
                    or not payload.get("complete")
                    or "exact_global" not in mechanisms_by_frames[frames]
                ):
                    off_ceiling[frames] = None
                    continue
                evaluation = payload["mechanisms"]["exact_global"]["evaluation"]
                score = float(evaluation["heldout_combo_score"])
                chance = float(evaluation["chance"])
                clears = bool(contexts[frames]["difficulty"].get("clears_floor"))
                off_ceiling[frames] = bool(
                    chance + P5_CEILING_CHANCE_OFFSET <= score <= P5_CEILING_UPPER and clears
                )
            if trainability_gate["applies"]:
                payload = seed_payloads.get(64, {}).get(str(seed))
                if payload is not None and payload.get("complete"):
                    trained_score = _heldout(payload, "exact_global")
                    frozen_score = _frozen_heldout(payload, "exact_global")
                    delta = trained_score - frozen_score
                    trainability_gate.update(
                        {
                            "evaluated": True,
                            "trained_heldout": trained_score,
                            "frozen_heldout": frozen_score,
                            "delta": delta,
                            "failed": delta <= P5_TRAINABILITY_MARGIN,
                        }
                    )
                    if trainability_gate["failed"]:
                        problems.append(
                            "trainability gate failed: seed-0 exact f64 arm does not beat its frozen "
                            f"evaluation by more than {P5_TRAINABILITY_MARGIN} (delta {delta:.4f}); "
                            "pilot stopped before any further seeds"
                        )
        if seed_index == 2 and len(seeds) > 3:
            for frames in frames_order:
                if truncated.get(frames):
                    continue
                mechanisms = mechanisms_by_frames[frames]
                factorized = [mechanism for mechanism in mechanisms if mechanism != "exact_global"]
                if "exact_global" not in mechanisms or not factorized:
                    continue
                maybe_three = [seed_payloads[frames].get(str(value)) for value in seeds[:3]]
                first_three = [payload for payload in maybe_three if payload is not None]
                if len(first_three) != 3 or any(not payload.get("complete") for payload in first_three):
                    continue
                deltas = {
                    mechanism: _mean(
                        [
                            _heldout(payload, "exact_global") - _heldout(payload, mechanism)
                            for payload in first_three
                        ]
                    )
                    for mechanism in factorized
                }
                if all(value <= 0.0 for value in deltas.values()) and all(
                    abs(value) < futility_margin for value in deltas.values()
                ):
                    truncated[frames] = {
                        "paired_mean_deltas": deltas,
                        "futility_margin": futility_margin,
                        "seeds_kept": seeds[:3],
                    }
        if stopped_for_wall or stopped_for_disk or stopped_for_required_arm:
            break

    frame_receipts: dict[str, dict[str, Any]] = {}
    parity_diagnostic: dict[str, Any] | None = None
    for serial_position, frames in enumerate(frames_order):
        context = _context(frames)
        mechanisms = mechanisms_by_frames[frames]
        payloads = seed_payloads[frames]
        if trainability_gate["failed"] or (len(seeds) > 1 and off_ceiling.get(frames) is not True):
            expected = seeds[:1]
        elif truncated.get(frames):
            expected = seeds[:3]
        else:
            expected = seeds
        completed_keys = [str(value) for value in expected if payloads.get(str(value), {}).get("complete")]
        cell_complete = len(completed_keys) == len(expected)
        scores = {
            mechanism: _paired_ci([_heldout(payloads[key], mechanism) for key in completed_keys])
            for mechanism in mechanisms
        }
        frozen_scores = {
            mechanism: _paired_ci([_frozen_heldout(payloads[key], mechanism) for key in completed_keys])
            for mechanism in mechanisms
        }
        contrasts: dict[str, dict[str, Any]] = {}
        if "exact_global" in mechanisms:
            for mechanism in mechanisms:
                if mechanism == "exact_global":
                    continue
                delta_values = [
                    _heldout(payloads[key], "exact_global") - _heldout(payloads[key], mechanism)
                    for key in completed_keys
                ]
                ci = _paired_ci(delta_values)
                contrasts[f"exact_minus_{mechanism}"] = {
                    **ci,
                    "classification": (
                        _classify_against_sesoi(ci["lo"], ci["hi"], sesoi) if ci["n"] >= 2 else "undetermined"
                    ),
                }
        compute_block = {
            "dense_reference_steps": dense_steps,
            "dense_flops_per_step": context["dense_flops_per_step"],
            "per_mechanism": {
                mechanism: {
                    "estimated_flops_per_step": context["per_step_flops"][mechanism],
                    "matched": context["matched"][mechanism],
                    "estimated_total_flops_completed_seeds": sum(
                        int(payloads[key]["mechanisms"][mechanism]["training"]["estimated_total_flops"])
                        for key in completed_keys
                    ),
                }
                for mechanism in mechanisms
            },
        }
        cell_problems: list[str] = []
        if not cell_complete:
            cell_problems.append("frame count incomplete this invocation; rerun the same command to resume")
        unmatched = [mechanism for mechanism in mechanisms if not context["matched"][mechanism]["matched_ok"]]
        if unmatched:
            cell_problems.append(
                "matched-compute claim refused: arms exceed the "
                f"{P5_FLOP_MATCH_TOLERANCE} total-FLOP deviation at this checkpoint granularity: "
                f"{unmatched}"
            )
        if not context["difficulty"].get("clears_floor"):
            cell_problems.append("programmatic difficulty oracle did not clear its floor")
        if frames == 16 and {"exact_global", "window_local"} <= set(mechanisms):
            payload_zero = payloads.get(str(seeds[0]))
            if payload_zero is not None and payload_zero.get("complete"):
                exact_zero = _heldout(payload_zero, "exact_global")
                window_zero = _heldout(payload_zero, "window_local")
                parity_diagnostic = {
                    "frames": 16,
                    "seed": seeds[0],
                    "exact_heldout": exact_zero,
                    "window_heldout": window_zero,
                    "window_minus_exact": window_zero - exact_zero,
                    "within_0p02": abs(window_zero - exact_zero) <= 0.02,
                    "hard_gate": False,
                    "note": (
                        "windowing is mathematically degenerate-global at f16 (single 512-token "
                        "window); blocks initialize differently per mechanism, so exact parity is "
                        "not expected and this delta is a diagnostic, not a gate"
                    ),
                }
        cell_receipt = {
            "schema": P5_CELL_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "claim_scope": CLAIM_SCOPE,
            "frames": frames,
            "mechanisms": mechanisms,
            "serial_position": serial_position,
            "corpus": {
                "spec": asdict(context["data_spec"]),
                "rows": len(context["records"]),
                "content_sha256": context["manifest"]["content_sha256"],
            },
            "difficulty_calibration": context["difficulty"],
            "parameters": next(row for row in parameter_table if row["frames"] == frames),
            "expected_seeds": expected,
            "seeds_completed": len(completed_keys),
            "seed_results": payloads,
            "scores": scores,
            "frozen_scores": frozen_scores,
            "paired_contrasts": contrasts,
            "compute": compute_block,
            "off_ceiling": off_ceiling.get(frames),
            "staged_out": bool(len(seeds) > 1 and off_ceiling.get(frames) is False),
            "futility_truncated": bool(truncated.get(frames)),
            "futility_evidence": truncated.get(frames),
            "parity_diagnostic": parity_diagnostic if frames == 16 else None,
            "complete": cell_complete,
            "problems": cell_problems,
            "all_ok": not cell_problems,
        }
        _atomic_json(context["dir"] / "cell_receipt.json", cell_receipt)
        frame_receipts[f"f{frames}"] = cell_receipt
        problems.extend(f"f{frames}: {problem}" for problem in cell_problems)

    complete = (
        not (stopped_for_wall or stopped_for_disk or stopped_for_required_arm)
        and not trainability_gate["failed"]
        and all(frame_receipts[f"f{frames}"]["complete"] for frames in frames_order)
    )
    if required_arm_failure is not None:
        problems.append(
            "required arm refused and aborted the pilot: "
            f"seed={required_arm_failure['seed']} cell={required_arm_failure.get('cell_id')} "
            f"reason={required_arm_failure['reason']}"
        )
    if not complete and not trainability_gate["failed"]:
        problems.append("pilot incomplete; rerun the same command to resume")

    def _contrast_tier(frames: int) -> dict[str, Any] | None:
        row = frame_receipts.get(f"f{frames}")
        if row is None or not row["paired_contrasts"]:
            return None
        if all(value["n"] == 0 for value in row["paired_contrasts"].values()):
            return None
        return row["paired_contrasts"]

    primary_contrasts = _contrast_tier(64)
    secondary_contrasts = _contrast_tier(32)
    if primary_contrasts is None:
        problems.append("f64 primary contrasts not estimable from this run")
    if secondary_contrasts is None:
        problems.append("f32 secondary contrasts not estimable from this run")
    context_response_curve = {
        mechanism: {
            f"f{frames}": frame_receipts[f"f{frames}"]["scores"][mechanism]
            for frames in sorted(frames_order)
            if mechanism in mechanisms_by_frames[frames]
        }
        for mechanism in P5_MECHANISMS
        if any(mechanism in mechanisms_by_frames[frames] for frames in frames_order)
    }
    parameters_by_frames = {row["frames"]: row["parameters"] for row in parameter_table}
    flop_table = [
        {
            "frames": frames,
            "mechanism": mechanism,
            "parameters": parameters_by_frames[frames][mechanism],
            "estimated_flops_per_step": contexts[frames]["per_step_flops"][mechanism],
            "flops_fraction_of_exact": (
                contexts[frames]["per_step_flops"][mechanism] / contexts[frames]["dense_flops_per_step"]
            ),
            "solved_steps": contexts[frames]["matched"][mechanism]["steps"],
            "fractional_total_flop_deviation": (
                contexts[frames]["matched"][mechanism]["fractional_deviation"]
            ),
            "matched_ok": contexts[frames]["matched"][mechanism]["matched_ok"],
        }
        for frames in frames_order
        for mechanism in mechanisms_by_frames[frames]
    ]
    receipt = {
        "schema": P5_SCREEN_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_scope": CLAIM_SCOPE,
        "config_sha256": config_sha,
        "cell_registry_sha256": registry_sha,
        "profile": config_plain.get("profile"),
        "serial_order": [cell.cell_id for cell in ordered],
        "seeds": seeds,
        "dense_reference_steps": dense_steps,
        "parameter_identity": parameter_table,
        "flop_table": flop_table,
        "frames": {
            key: {
                "complete": row["complete"],
                "off_ceiling": row["off_ceiling"],
                "staged_out": row["staged_out"],
                "futility_truncated": row["futility_truncated"],
                "seeds_completed": row["seeds_completed"],
                "scores": row["scores"],
                "paired_contrasts": row["paired_contrasts"],
                "all_ok": row["all_ok"],
            }
            for key, row in frame_receipts.items()
        },
        "primary_contrasts_f64": primary_contrasts,
        "secondary_contrasts_f32": secondary_contrasts,
        "context_response_curve": context_response_curve,
        "sesoi": sesoi,
        "parity_diagnostic": parity_diagnostic,
        "staging": {
            "off_ceiling": {f"f{frames}": off_ceiling.get(frames) for frames in frames_order},
            "futility_truncated": {f"f{frames}": truncated.get(frames) for frames in frames_order},
        },
        "trainability_gate": trainability_gate,
        "trainability_gate_failed": bool(trainability_gate["failed"]),
        "promotion": {
            "confirmatory_promotable": False,
            "refused_by_construction": True,
            "reason": PROMOTION_REFUSAL,
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
        },
        "complete": complete,
        "resumable": not complete and not trainability_gate["failed"],
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
    _atomic_json(run_dir / "p5_context_receipt.json", receipt)
    return receipt
