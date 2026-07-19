
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mop.diagnostics.riskcov import seed_ci  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "mot"

N_TASKS = 150  # long stream so the late window can diverge from the early window
EARLY_WINDOW = 20  # first 20 tasks define "early" learning accuracy
LATE_WINDOW = 20  # last 20 tasks define "late" learning accuracy
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 seeds for the gap CI
Z = 1.96  # 95pct normal-approx CI

DIM = 20
N_CLASSES = 8
SAMPLES_PER_TASK = 200
HIDDEN = 48
TEACHER_HIDDEN = 32
SGD_LR = 0.2  # plain SGD, fixed lr, no momentum: the baseline PR9 replaces
EPOCHS_PER_TASK = 25
BATCH = 8


class MLP(torch.nn.Module):

    def __init__(self, dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, n_classes)
        self.h: torch.Tensor | None = None

    def forward(self, x):
        self.h = F.relu(self.fc1(x))
        return self.fc2(self.h)


def _teacher(g: torch.Generator):
    W1 = torch.randn(DIM, TEACHER_HIDDEN, generator=g)
    W2 = torch.randn(TEACHER_HIDDEN, N_CLASSES, generator=g)
    return W1, W2


def _teach(X: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    return (F.relu(X @ W1) @ W2).argmax(-1)


def _task_stream(seed: int, mode: str):
    g = torch.Generator().manual_seed(seed * 100 + 7)
    fixed_teacher = _teacher(g)  # used by the stationary stream
    tasks = []
    for _ in range(N_TASKS):
        X = torch.randn(SAMPLES_PER_TASK, DIM, generator=g)
        if mode == "drift":
            W1, W2 = _teacher(g)  # NEW concept each task
        elif mode == "fixed":
            W1, W2 = fixed_teacher  # SAME concept repeated
        else:
            raise ValueError(mode)
        tasks.append((X, _teach(X, W1, W2)))
    return tasks


def _learn_on_task(model, opt, X, y) -> tuple[float, float]:
    g = torch.Generator().manual_seed(0)
    n = X.shape[0]
    best = 0.0
    dead = 0.0
    for _ in range(EPOCHS_PER_TASK):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            j = perm[i : i + BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(X[j]), y[j]).backward()
            opt.step()
        with torch.no_grad():
            out = model(X)
            best = max(best, float((out.argmax(-1) == y).float().mean()))
            dead = float((model.h.abs().sum(0) == 0).float().mean())
    return best, dead


def _run_one_seed(seed: int, mode: str) -> dict:
    torch.manual_seed(seed)
    model = MLP(DIM, HIDDEN, N_CLASSES)
    opt = torch.optim.SGD(model.parameters(), lr=SGD_LR)  # plain SGD, fixed lr, no momentum
    diag, dead = [], []
    for X, y in _task_stream(seed, mode):
        b, d = _learn_on_task(model, opt, X, y)
        diag.append(b)
        dead.append(d)
    early = sum(diag[:EARLY_WINDOW]) / EARLY_WINDOW
    late = sum(diag[-LATE_WINDOW:]) / LATE_WINDOW
    dead_early = sum(dead[:EARLY_WINDOW]) / EARLY_WINDOW
    dead_late = sum(dead[-LATE_WINDOW:]) / LATE_WINDOW
    return {
        "early": early,
        "late": late,
        "gap": early - late,
        "dead_early": dead_early,
        "dead_late": dead_late,
    }


def _run_stream(mode: str) -> dict:
    per_seed = [_run_one_seed(s, mode) for s in SEEDS]
    gaps = [r["gap"] for r in per_seed]
    ci = seed_ci(gaps, z=Z)
    fires = ci["lo"] > 0.0
    quiet = ci["lo"] <= 0.0 <= ci["hi"]
    n = len(per_seed)
    return {
        "mode": mode,
        "n_seeds": n,
        "early_learn_acc_mean": round(sum(r["early"] for r in per_seed) / n, 4),
        "late_learn_acc_mean": round(sum(r["late"] for r in per_seed) / n, 4),
        "gap_ci": ci,
        "per_seed_gap": [round(g, 4) for g in gaps],
        "dead_unit_frac_early": round(sum(r["dead_early"] for r in per_seed) / n, 4),
        "dead_unit_frac_late": round(sum(r["dead_late"] for r in per_seed) / n, 4),
        "fires": bool(fires),
        "quiet": bool(quiet),
    }


def main():
    stream1 = _run_stream("drift")  # loss-inducing, MUST fire
    stream2 = _run_stream("fixed")  # stationary, MUST NOT fire

    validated = bool(stream1["fires"] and stream2["quiet"])
    if validated:
        verdict = "VALIDATED"
    elif not stream1["fires"] and stream2["quiet"]:
        verdict = (
            "NOT_VALIDATED (NULL): instrument failed to detect loss on the positive-control "
            "stream, gap CI on stream1 includes 0"
        )
    elif stream1["fires"] and not stream2["quiet"]:
        verdict = (
            "NOT_VALIDATED: instrument FALSE-FIRES on the stationary negative-control stream, "
            "gap CI on stream2 excludes 0"
        )
    else:
        verdict = "NOT_VALIDATED: both controls failed"

    out = {
        "certificate": "plasticity_loss_gap = early_learn_acc - late_learn_acc, with 95pct seed CI",
        "mechanistic_corroborator": "dead ReLU unit fraction (rises under loss, targeted by PR9)",
        "baseline": f"fixed plain SGD (lr={SGD_LR:.3f}, no momentum, no re-init between tasks)",
        "preregistered_rule": (
            "VALIDATED iff stream1 gap CI lower bound > 0 (fires) AND stream2 gap CI contains 0 "
            "(quiet). A tie (CI includes 0) on stream1 is a NULL. Threshold fixed before running, "
            "never tuned after seeing a number."
        ),
        "design_note": (
            "The stream construction and regime (net size, lr, epochs) were chosen on separate "
            "diagnostic sweeps because an initial input-permutation design saturated both streams "
            "and never exercised the phenomenon. That is a design fix, not a threshold tune."
        ),
        "config": {
            "n_tasks": N_TASKS,
            "early_window": EARLY_WINDOW,
            "late_window": LATE_WINDOW,
            "seeds": SEEDS,
            "dim": DIM,
            "n_classes": N_CLASSES,
            "hidden": HIDDEN,
            "teacher_hidden": TEACHER_HIDDEN,
            "sgd_lr": SGD_LR,
            "epochs_per_task": EPOCHS_PER_TASK,
            "batch": BATCH,
            "samples_per_task": SAMPLES_PER_TASK,
            "z": Z,
        },
        "stream1_loss_inducing": stream1,
        "stream2_stationary": stream2,
        "validated": validated,
        "verdict": verdict,
    }
    (OUT / "plasticity_certificate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
