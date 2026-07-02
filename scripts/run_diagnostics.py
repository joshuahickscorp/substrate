#!/usr/bin/env python
"""Run the diagnostic battery on cached/synthetic latents and print a summary. These gate the
experiments: linear-probe decodability, the noisy-TV epistemic/aleatoric guard, calibration,
the Fisher-trace critical-period signature, and the determinism sanity loop.

Usage: python scripts/run_diagnostics.py [device=cpu] [+dim=64]
"""

from __future__ import annotations

import json
import sys

import torch

from mop.config import compose
from mop.devices import resolve
from mop.diagnostics import (
    critical_period_signature,
    determinism_loop,
    fisher_trace_over_training,
    linear_probe,
    noisy_tv_diagnostic,
    reliability,
)
from mop.substrate.datasets import make_task_stream


def main(argv: list[str] | None = None) -> int:
    cfg = compose(list(sys.argv[1:] if argv is None else argv))
    dev = resolve(str(cfg.device.kind))
    dim = int(cfg.get("dim", 64))

    task = make_task_stream(n_tasks=1, dim=dim, classes_per_task=4, samples_per_task=400, separation=3.0)[0]
    probe = linear_probe(task.x, task.y)
    ntv = noisy_tv_diagnostic(dim=dim, device=dev, steps=300)

    data = [(torch.randn(16, 8), torch.randint(0, 2, (16,))) for _ in range(6)]
    trace = fisher_trace_over_training(
        lambda: torch.nn.Linear(8, 2),
        data,
        lambda m: lambda b: torch.nn.functional.cross_entropy(m(b[0]), b[1]),
        checkpoints=6,
        steps_per_ckpt=8,
    )
    sig = critical_period_signature(trace)

    probs = torch.softmax(torch.randn(500, 4), -1)
    cal = reliability(probs, torch.randint(0, 4, (500,)))
    det = determinism_loop(lambda: torch.randn(64) @ torch.randn(64), runs=5)

    out = {
        "linear_probe": {"score": probe["score"], "decodable": probe["decodable"]},
        "noisy_tv": {
            k: ntv[k]
            for k in ("noise_error_stays_high", "epistemic_collapses_on_noise", "learning_progress_separates")
        },
        "fisher_trace": {"peak_index": sig["peak_index"], "rise_then_fall": sig["rise_then_fall"]},
        "calibration_ece": cal["ece"],
        "determinism": {"bit_identical": det.bit_identical, "max_abs": det.max_abs},
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
