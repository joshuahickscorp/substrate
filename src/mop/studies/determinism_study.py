
from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import torch

from ..config import compose
from ..harness.runner import run_experiment

CONFIGS: dict[str, dict] = {
    "e1": {
        "overrides": [
            "experiment=e1_baseline",
            "device=cpu",
            "experiment.stream.n_tasks=3",
            "experiment.stream.dim=32",
            "experiment.stream.classes_per_task=3",
            "experiment.stream.samples_per_task=64",
            "experiment.train.epochs_per_task=2",
            "experiment.head.hidden=16",
        ],
        "metric": ("gate", "protected_bwt"),
    },
    "i4": {
        "overrides": [
            "experiment=i4_backprop_alts",
            "device=cpu",
            "experiment.dim=16",
            "experiment.n_classes=4",
            "experiment.samples=80",
            "experiment.hidden=12",
            "experiment.epochs=20",
            "experiment.seeds=[0,1]",
        ],
        "metric": ("ceiling_backprop_acc",),
    },
}


def _dig(d: dict, path: tuple[str, ...]) -> float:
    cur: object = d
    for k in path:
        cur = cur[k]  # type: ignore[index]
    return float(cur)  # type: ignore[arg-type]


def _run_once(overrides: list[str], metric_path: tuple[str, ...]) -> float:
    cfg = compose(list(overrides))
    with tempfile.TemporaryDirectory() as d:
        metrics = run_experiment(cfg, Path(d))
    return _dig(metrics, metric_path)


def _characterize(name: str, spec: dict, reps: int) -> dict:
    vals = [_run_once(spec["overrides"], spec["metric"]) for _ in range(reps)]
    base = vals[0]
    diffs = [abs(v - base) for v in vals[1:]]
    max_abs = max(diffs) if diffs else 0.0
    b0 = torch.tensor(base, dtype=torch.float64).view(torch.int64).item()
    identical = sum(torch.tensor(v, dtype=torch.float64).view(torch.int64).item() == b0 for v in vals)
    return {
        "name": name,
        "metric": ".".join(spec["metric"]),
        "runs": reps,
        "byte_identical": identical == reps,
        "byte_identical_rate": identical / reps,
        "max_abs": max_abs,
        "std": float(statistics.pstdev(vals)) if reps > 1 else 0.0,
        "metric_values": vals,
    }


def determinism_study(reps: int = 3, toy: bool = True) -> dict:
    assert reps >= 2, "need at least 2 reps to measure run-to-run spread"
    prev = torch.get_num_threads()
    torch.set_num_threads(1)  # single-threaded + serial: the clean determinism baseline
    try:
        configs = [_characterize(n, s, reps) for n, s in CONFIGS.items()]
    finally:
        torch.set_num_threads(prev)

    all_identical = all(c["byte_identical"] for c in configs)
    worst = max((c["max_abs"] for c in configs), default=0.0)
    verdict = "bit-identical run-to-run" if all_identical else f"near bit-identical (max abs {worst:.3e})"
    return {
        "configs": configs,
        "reps": reps,
        "toy": bool(toy),
        "num_threads": 1,
        "cpu_more_deterministic_than_metal": True,
        "note": (
            f"CPU serial single-thread is {verdict}; Apple Metal at temperature zero is only "
            "~50% byte-identical (DECISIONS.md), so CPU is far more deterministic and is the "
            "baseline the suite sizes tolerances from."
        ),
        "provenance": "provisional",
    }
