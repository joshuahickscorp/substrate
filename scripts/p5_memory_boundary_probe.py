"""P5 memory-boundary mechanics probe: exact dense context versus factorized alternatives.

This is the bounded local instrument named by EXTENDED_COMPUTE_EXECUTION_PLAN.md (P5 card and the
memory-boundary benchmark specification) and by the Studio-necessity facet in
EXTENDED_COMPUTE_DEEP_RESEARCH_2026_07.md. It measures, per (context mechanism, token count) cell,
the cold-process peak RSS and wall time of one training step (forward plus backward plus optimizer)
of a small dense spatiotemporal transformer at CM7-class width, on CPU, batch 1.

Claim boundary, preregistered here before any run:
- This receipt is MECHANICS ONLY. It produces the shape ledger and measured per-cell peaks that any
  future category 4/5/8/9 claim must cite. It cannot itself move any registry row's category.
- The naive attention-score formula (batch x heads x tokens^2 x bytes) is recorded as a diagnostic
  upper bound only; the measured child-process peak governs.
- A cell that exceeds the preregistered child memory guard is recorded as a measured boundary for
  THAT mechanism and token count on THIS host, never as evidence that the scientific question
  requires other hardware (the estimand question stays with the P5 experiment card).
- One heavy process discipline: cells run serially in short-lived child processes; the probe
  refuses to start if free disk is below the active local floor.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "mop-p5-memory-boundary-trace/v1"
DEFAULT_OUT = Path("proof/P5_MEMORY_BOUNDARY_TRACE.json")
MIN_FREE_DISK_GB = 40.0
CHILD_MEMORY_GUARD_GB = 12.0
CHILD_TIMEOUT_SECONDS = 900.0

# CM7-class width: the calibrated custom substrate is 1.646M parameters at 256 dense tokens.
# The probe holds width fixed and scales tokens, which is the P5 axis under test.
WIDTH = 192
HEADS = 4
DEPTH = 2
BATCH = 1

MECHANISMS = ("exact_math", "sdpa", "sdpa_checkpointed", "window_local")
WINDOW_TOKENS = 512

# (frames, resolution) -> tokens with tubelet 2 and patch 16, matching the ceiling formula in
# EXTENDED_COMPUTE_DEEP_RESEARCH_2026_07.md: tokens = ceil(F/t) * ceil(H/p) * ceil(W/p).
TOKEN_GRID: tuple[tuple[int, int], ...] = (
    (16, 128),
    (32, 128),
    (16, 256),
    (32, 256),
    (64, 256),
)


def grid_tokens(frames: int, resolution: int, *, tubelet: int = 2, patch: int = 16) -> int:
    return (frames // tubelet) * (resolution // patch) ** 2


@dataclass(frozen=True)
class Cell:
    mechanism: str
    frames: int
    resolution: int

    @property
    def tokens(self) -> int:
        return grid_tokens(self.frames, self.resolution)

    @property
    def cell_id(self) -> str:
        return f"{self.mechanism}_f{self.frames}_r{self.resolution}"


def _free_disk_gb(root: Path) -> float:
    usage = os.statvfs(root)
    return usage.f_bavail * usage.f_frsize / 1e9


def naive_score_bytes(tokens: int, *, heads: int = HEADS, batch: int = BATCH) -> int:
    """Diagnostic upper bound only: a materialized fp32 attention-score tensor."""
    return batch * heads * tokens * tokens * 4


def _child_payload(cell: Cell, seed: int) -> dict[str, Any]:
    """Run one training step in this (child) process and report measured peaks."""
    import torch

    torch.manual_seed(seed)
    device = torch.device("cpu")
    tokens = cell.tokens

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(WIDTH)
            self.qkv = torch.nn.Linear(WIDTH, WIDTH * 3)
            self.proj = torch.nn.Linear(WIDTH, WIDTH)
            self.norm2 = torch.nn.LayerNorm(WIDTH)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(WIDTH, WIDTH * 4),
                torch.nn.GELU(),
                torch.nn.Linear(WIDTH * 4, WIDTH),
            )

        def _attention(self, x: torch.Tensor) -> torch.Tensor:
            b, length, _ = x.shape
            qkv = self.qkv(x).reshape(b, length, 3, HEADS, WIDTH // HEADS)
            q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
            if cell.mechanism == "exact_math":
                scores = q @ k.transpose(-2, -1) / (WIDTH // HEADS) ** 0.5
                out = scores.softmax(dim=-1) @ v
            elif cell.mechanism in ("sdpa", "sdpa_checkpointed"):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            elif cell.mechanism == "window_local":
                outs = []
                for start in range(0, length, WINDOW_TOKENS):
                    stop = min(start + WINDOW_TOKENS, length)
                    outs.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop]
                        )
                    )
                out = torch.cat(outs, dim=2)
            else:
                raise ValueError(f"unknown mechanism {cell.mechanism}")
            return self.proj(out.transpose(1, 2).reshape(b, length, WIDTH))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if cell.mechanism == "sdpa_checkpointed":
                attended = torch.utils.checkpoint.checkpoint(
                    lambda t: self._attention(self.norm1(t)), x, use_reentrant=False
                )
            else:
                attended = self._attention(self.norm1(x))
            x = x + attended
            return x + self.mlp(self.norm2(x))

    model = torch.nn.Sequential(*[Block() for _ in range(DEPTH)]).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(BATCH, tokens, WIDTH, device=device)
    target = torch.randn(BATCH, tokens, WIDTH, device=device)

    start = time.perf_counter()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    optimizer.step()
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
        "torch_version": torch.__version__,
    }


def run_child(cell: Cell, seed: int) -> dict[str, Any]:
    """Cold-process execution so peak RSS is per cell, not cumulative."""
    spec = json.dumps(
        {"mechanism": cell.mechanism, "frames": cell.frames, "resolution": cell.resolution, "seed": seed}
    )
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_child", spec],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
    )
    elapsed = round(time.perf_counter() - started, 3)
    if proc.returncode != 0:
        return {
            "cell": cell.cell_id,
            "ok": False,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "wall_seconds_child": elapsed,
        }
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload.update({"cell": cell.cell_id, "ok": True, "wall_seconds_child": elapsed})
    guard_exceeded = payload.get("peak_rss_gb", 0.0) > CHILD_MEMORY_GUARD_GB
    payload["memory_guard_exceeded"] = guard_exceeded
    return payload


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        spec = json.loads(sys.argv[2])
        cell = Cell(spec["mechanism"], spec["frames"], spec["resolution"])
        print(json.dumps(_child_payload(cell, spec["seed"])))
        return 0

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8192, help="skip grid cells above this token count")
    args = parser.parse_args()

    free_gb = _free_disk_gb(Path.cwd())
    if free_gb < MIN_FREE_DISK_GB:
        print(f"refusing: free disk {free_gb:.1f} GB below the {MIN_FREE_DISK_GB} GB floor")
        return 1

    cells = [
        Cell(mechanism, frames, resolution)
        for mechanism in MECHANISMS
        for frames, resolution in TOKEN_GRID
        if grid_tokens(frames, resolution) <= args.max_tokens
    ]
    results: list[dict[str, Any]] = []
    for cell in cells:
        for repeat in range(args.repeats):
            row = run_child(cell, args.seed + repeat)
            row.update(
                {
                    "mechanism": cell.mechanism,
                    "frames": cell.frames,
                    "resolution": cell.resolution,
                    "repeat": repeat,
                    "naive_score_bytes_diagnostic_only": naive_score_bytes(cell.tokens),
                }
            )
            results.append(row)
            status = "ok" if row.get("ok") else "FAILED"
            print(
                f"{cell.cell_id} repeat {repeat}: {status} "
                f"peak_rss_gb={row.get('peak_rss_gb')} wall_step={row.get('wall_seconds_step')}s"
            )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_boundary": {
            "mechanics_only": True,
            "moves_no_category": True,
            "naive_formula_is_diagnostic_only": True,
            "interpretation": (
                "Measured cold-process training-step peaks for exact versus factorized context at "
                "CM7-class width on this host. Any future memory-rung claim must cite these peaks, "
                "the P5 experiment card's estimand argument, and three repeated failures against "
                "the runtime safe envelope; this receipt alone earns nothing."
            ),
        },
        "config": {
            "width": WIDTH,
            "heads": HEADS,
            "depth": DEPTH,
            "batch": BATCH,
            "window_tokens": WINDOW_TOKENS,
            "mechanisms": list(MECHANISMS),
            "token_grid": [
                {"frames": f, "resolution": r, "tokens": grid_tokens(f, r)} for f, r in TOKEN_GRID
            ],
            "child_memory_guard_gb": CHILD_MEMORY_GUARD_GB,
            "device": "cpu",
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "free_disk_gb_at_start": round(free_gb, 2),
        },
        "cells": results,
        "all_ok": all(row.get("ok") for row in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} cells={len(results)} all_ok={receipt['all_ok']}")
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
