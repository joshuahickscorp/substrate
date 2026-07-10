"""P5 training-grid memory probe: cold-process training-step peaks for the twelve pilot cells.

This is the measurement companion to the P5 exact-versus-factorized context pilot
(src/mop/substrate/p5_context.py). It measures, per (frame count, context mechanism) cell at the
pilot's exact model construction (dim 128, resolution 256, patch 32, tubelet 2), the cold-process
peak RSS and wall time of one full training step (masked forward, EMA-target forward, predictor,
cosine loss, backward, gradient clip, AdamW step, EMA update) at batch 4 and batch 1, three
repeats per row, on CPU.

Claim boundary, preregistered here before any run:
- MECHANICS ONLY: this trace records the component set the P5 card's memory specification names
  (actual peak formula components, MPS recommended/current/driver memory, RSS, memory pressure,
  and three repeats). It moves no registry category and earns no memory rung by itself.
- The naive attention-score formula is a diagnostic upper bound only; the measured child-process
  peak governs. Model, gradient, optimizer, EMA-target, and input byte components are analytical
  ledger entries, never allocation evidence.
- The receipt is written ONLY when every cell finishes with a finite loss; any failed or
  non-finite cell blocks the artifact entirely (fail closed).
- It cites the existing P5 memory-boundary trace by content hash so the two instruments stay
  bound; a missing boundary trace refuses the run.
- One heavy process discipline: cells run serially in short-lived child processes; the probe
  refuses to start if free disk is below the active local floor.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "mop-p5-traingrid-memory-trace/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "proof/P5_TRAINGRID_MEMORY_TRACE.json"
BOUNDARY_TRACE = REPO_ROOT / "proof/P5_MEMORY_BOUNDARY_TRACE.json"
MIN_FREE_DISK_GB = 40.0
CHILD_MEMORY_GUARD_GB = 12.0
CHILD_TIMEOUT_SECONDS = 1800.0
BATCH_ROWS = (4, 1)
DEFAULT_REPEATS = 3
MASK_RATIO = 0.50
EMA_DECAY = 0.99


def _free_disk_gb(root: Path) -> float:
    usage = os.statvfs(root)
    return usage.f_bavail * usage.f_frsize / 1e9


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_pressure_level() -> int | None:
    """macOS kernel memory-pressure level (1 normal, 2 warn, 4 critical); None when unreadable."""

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], stderr=subprocess.DEVNULL
        )
        return int(out.strip())
    except Exception:
        return None


def _mps_memory() -> dict[str, int | None]:
    """The MPS fields the P5 card names; None off Metal or when torch cannot report them."""

    try:
        import torch

        if torch.backends.mps.is_available():
            return {
                "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
                "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
                "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            }
    except Exception:
        pass
    return {
        "recommended_max_bytes": None,
        "current_allocated_bytes": None,
        "driver_allocated_bytes": None,
    }


def _child_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one full training step in this (child) process and report measured peaks."""

    import torch

    from mop.substrate.custom_workbench import parameter_count
    from mop.substrate.p5_context import P5CellSpec, build_p5_substrate

    torch.manual_seed(int(spec["seed"]))
    cell = P5CellSpec(int(spec["frames"]), str(spec["mechanism"]))
    overrides = spec.get("model_overrides")
    batch = int(spec["batch"])

    model = build_p5_substrate(cell, model_overrides=overrides)
    target = build_p5_substrate(cell, model_overrides=overrides)
    target.load_state_dict(model.state_dict())
    target.requires_grad_(False)
    params = parameter_count(model)
    model_spec = model.spec
    tokens = (cell.frames // model_spec.tubelet) * (model_spec.max_resolution // model_spec.patch_size) ** 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    clips = torch.rand(batch, 3, cell.frames, model_spec.max_resolution, model_spec.max_resolution)
    mask = torch.rand(batch, tokens) < MASK_RATIO
    mask[:, 0] = True

    start = time.perf_counter()
    online_tokens = model.encode(clips, mask)
    with torch.no_grad():
        target_tokens = target.encode(clips)
    prediction = torch.nn.functional.normalize(model.predictor(online_tokens), dim=-1)
    target_value = torch.nn.functional.normalize(target_tokens, dim=-1)
    loss = (1.0 - (prediction * target_value).sum(dim=-1))[mask].mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    with torch.no_grad():
        for target_parameter, online_parameter in zip(target.parameters(), model.parameters(), strict=True):
            target_parameter.mul_(EMA_DECAY).add_(online_parameter, alpha=1.0 - EMA_DECAY)
    wall_seconds = time.perf_counter() - start

    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    raw_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gb = raw_maxrss * (1024 if platform.system() == "Linux" else 1) / 1e9
    return {
        "tokens": tokens,
        "parameters": params,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "wall_seconds_step": round(wall_seconds, 3),
        "peak_rss_gb": round(peak_rss_gb, 4),
        "peak_formula_components_bytes": {
            "parameters": params * 4,
            "gradients": params * 4,
            "adamw_state": 2 * params * 4,
            "ema_target_parameters": params * 4,
            "input_clips": batch * 3 * cell.frames * model_spec.max_resolution**2 * 4,
            "token_activations_one_layer": batch * tokens * model_spec.dim * 4,
            "naive_attention_score_diagnostic_only": batch * model_spec.heads * tokens * tokens * 4,
        },
        "mps_memory": _mps_memory(),
        "memory_pressure_level": _memory_pressure_level(),
        "torch_version": torch.__version__,
    }


def _child_env() -> dict[str, str]:
    """Child inherits the environment; the repo src path is guaranteed on PYTHONPATH."""

    src = str(REPO_ROOT / "src")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    return env


def run_child(spec: dict[str, Any]) -> dict[str, Any]:
    """Cold-process execution so peak RSS is per cell, not cumulative."""

    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_child", json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        env=_child_env(),
    )
    elapsed = round(time.perf_counter() - started, 3)
    cell_id = f"f{spec['frames']}_{spec['mechanism']}_b{spec['batch']}"
    if proc.returncode != 0:
        return {
            "cell": cell_id,
            "ok": False,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "wall_seconds_child": elapsed,
        }
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload.update({"cell": cell_id, "ok": True, "wall_seconds_child": elapsed})
    payload["memory_guard_exceeded"] = payload.get("peak_rss_gb", 0.0) > CHILD_MEMORY_GUARD_GB
    return payload


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        print(json.dumps(_child_payload(json.loads(sys.argv[2]))))
        return 0

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one tiny cold-child cell (dim 32, depth 1, f16 exact, batch 1, one repeat); "
        "proves the plumbing and never writes the proof artifact",
    )
    args = parser.parse_args()

    from mop.substrate.p5_context import P5_CELLS

    if not BOUNDARY_TRACE.is_file():
        print(f"refusing: required citation {BOUNDARY_TRACE} is missing")
        return 1
    boundary_sha256 = _sha256_file(BOUNDARY_TRACE)

    if args.smoke:
        row = run_child(
            {
                "frames": 16,
                "mechanism": "exact_global",
                "batch": 1,
                "seed": args.seed,
                "model_overrides": {"dim": 32, "depth": 1},
            }
        )
        row["smoke"] = True
        row["cited_boundary_trace_sha256"] = boundary_sha256
        print(json.dumps(row, indent=2, sort_keys=True))
        ok = bool(row.get("ok")) and bool(row.get("loss_finite"))
        print(f"smoke {'ok' if ok else 'FAILED'}; no proof artifact written from smoke mode")
        return 0 if ok else 1

    free_gb = _free_disk_gb(Path.cwd())
    if free_gb < MIN_FREE_DISK_GB:
        print(f"refusing: free disk {free_gb:.1f} GB below the {MIN_FREE_DISK_GB} GB floor")
        return 1

    results: list[dict[str, Any]] = []
    for cell in P5_CELLS:
        for batch in BATCH_ROWS:
            for repeat in range(args.repeats):
                row = run_child(
                    {
                        "frames": cell.frames,
                        "mechanism": cell.mechanism,
                        "batch": batch,
                        "seed": args.seed + repeat,
                        "model_overrides": None,
                    }
                )
                row.update(
                    {
                        "frames": cell.frames,
                        "mechanism": cell.mechanism,
                        "batch": batch,
                        "repeat": repeat,
                    }
                )
                results.append(row)
                status = "ok" if row.get("ok") else "FAILED"
                print(
                    f"{row['cell']} repeat {repeat}: {status} "
                    f"peak_rss_gb={row.get('peak_rss_gb')} wall_step={row.get('wall_seconds_step')}s"
                )

    all_finite = all(row.get("ok") and row.get("loss_finite") for row in results)
    if not all_finite:
        print("refusing to write the trace: one or more cells failed or produced a non-finite loss")
        return 1

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_boundary": {
            "mechanics_only": True,
            "moves_no_category": True,
            "naive_formula_is_diagnostic_only": True,
            "interpretation": (
                "Measured cold-process training-step peaks for the twelve P5 pilot cells at the "
                "pilot's exact model construction on this host. Any future memory-rung claim must "
                "cite these peaks, the P5 experiment card's estimand argument, and three repeated "
                "failures against the runtime safe envelope; this receipt alone earns nothing."
            ),
        },
        "cited_boundary_trace": {
            "path": str(BOUNDARY_TRACE.relative_to(REPO_ROOT)),
            "sha256": boundary_sha256,
        },
        "config": {
            "cells": [{"frames": cell.frames, "mechanism": cell.mechanism} for cell in P5_CELLS],
            "batch_rows": list(BATCH_ROWS),
            "repeats": args.repeats,
            "seed": args.seed,
            "mask_ratio": MASK_RATIO,
            "ema_decay": EMA_DECAY,
            "child_memory_guard_gb": CHILD_MEMORY_GUARD_GB,
            "device": "cpu",
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "free_disk_gb_at_start": round(free_gb, 2),
        },
        "cells": results,
        "all_ok": all_finite and not any(row.get("memory_guard_exceeded") for row in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.out)
    print(f"wrote {args.out} rows={len(results)} all_ok={receipt['all_ok']}")
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
