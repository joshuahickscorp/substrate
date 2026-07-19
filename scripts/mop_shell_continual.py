#!/usr/bin/env python

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mop.diagnostics.continual_metrics import LRIntegralAccumulator
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.metrics import ContinualResult
from mop.shell import ReplayBuffer

_ROOT = Path(__file__).resolve().parents[1]
CACHE = _ROOT / "data" / "cache" / "vjepa2_vitl_nuisance"
OUT = _ROOT / "runs" / "mot" / "shell_continual.json"
SEEDS = list(range(8))  # 8 seeds, as prescribed
N_CLASSES = 5  # 5 shapes
HIDDEN = 64  # shell trunk width
LR = 1e-2
EPOCHS_PER_TASK = 8  # optimizer passes per task (enough to overfit the 2-class window -> drift)
BATCH = 32
REPLAY_BATCH = 32
REPLAY_CAPACITY = 200  # holds the whole train set at most; reservoir eviction
TEST_FRAC = 0.30  # per-class held-out fraction
CI_Z = 1.96


def _load():
    x = torch.from_numpy(np.load(CACHE / "features.npy")).float()
    y = torch.from_numpy(np.load(CACHE / "labels_shape.npy")).long()
    return x, y


def _class_split(y: torch.Tensor, test_frac: float, g: torch.Generator):
    tr, te = [], []
    for c in y.unique().tolist():
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        cut = max(1, int(round(len(idx) * test_frac)))
        te.append(idx[:cut])
        tr.append(idx[cut:])
    return torch.cat(tr), torch.cat(te)


def _zscore_fit(train_x: torch.Tensor):
    mu = train_x.mean(0, keepdim=True)
    sd = train_x.std(0, keepdim=True) + 1e-6
    return mu, sd


class Shell(nn.Module):

    def __init__(self, dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def _tasks(n_classes: int) -> list[list[int]]:
    tasks = [[0, 1]]
    for j in range(2, n_classes):
        tasks.append([j - 1, j])
    return tasks  # T = n_classes - 1 tasks


def run_arm(x, y, *, seed: int, use_replay: bool):
    g = torch.Generator().manual_seed(seed)
    tr_idx, te_idx = _class_split(y, TEST_FRAC, g)
    mu, sd = _zscore_fit(x[tr_idx])
    xz = (x - mu) / sd

    tasks = _tasks(N_CLASSES)
    t_total = len(tasks)

    def task_pool(pool, classes):
        m = torch.zeros(len(pool), dtype=torch.bool)
        for c in classes:
            m |= y[pool] == c
        return pool[m]

    te_task = [task_pool(te_idx, t) for t in tasks]
    tr_task = [task_pool(tr_idx, t) for t in tasks]

    torch.manual_seed(seed)  # identical init across arms (only replay differs)
    shell = Shell(x.shape[1], HIDDEN, N_CLASSES)
    opt = torch.optim.SGD(shell.parameters(), lr=LR)
    lri = LRIntegralAccumulator()

    buffer = None
    if use_replay:
        buffer = ReplayBuffer(capacity=REPLAY_CAPACITY, dim=x.shape[1], prioritized=True, seed=seed)

    batch_g = torch.Generator().manual_seed(seed + 101)

    @torch.no_grad()
    def acc_on(idx):
        if len(idx) == 0:
            return 0.0
        return float((shell(xz[idx]).argmax(-1) == y[idx]).float().mean())

    R = [[0.0] * t_total for _ in range(t_total)]

    for ti in range(t_total):
        tr = tr_task[ti]
        xt, yt = xz[tr], y[tr]
        n = len(xt)
        for _ep in range(EPOCHS_PER_TASK):
            perm = torch.randperm(n, generator=batch_g)
            for b0 in range(0, n, BATCH):
                bidx = perm[b0 : b0 + BATCH]
                bx, by = xt[bidx], yt[bidx]
                opt.zero_grad()
                loss = F.cross_entropy(shell(bx), by)
                if buffer is not None and len(buffer) > 0:
                    rb = buffer.sample(REPLAY_BATCH)
                    rx, ry, rw = rb["x"], rb["y"], rb["is_weight"]
                    rl = (F.cross_entropy(shell(rx), ry, reduction="none") * rw).mean()
                    loss = loss + rl
                loss.backward()
                opt.step()
                lri.add(LR, steps=1)
        if buffer is not None:
            with torch.no_grad():
                ce = F.cross_entropy(shell(xt), yt, reduction="none")
            buffer.add(xt, yt, key=xt, priority=ce.detach().cpu())
        for tj in range(t_total):
            R[ti][tj] = acc_on(te_task[tj])

    return {"R": R, "lr_integral": lri.total()}


def main() -> int:
    t0 = time.perf_counter()
    x, y = _load()
    tasks = _tasks(N_CLASSES)
    chance = 0.5  # each task is a 2-class window; chance on its own test set is 0.5

    per_seed = []
    for s in SEEDS:
        naive = run_arm(x, y, seed=s, use_replay=False)
        replay = run_arm(x, y, seed=s, use_replay=True)

        bwt_naive = ContinualResult(R=naive["R"], chance=chance).backward_transfer()
        bwt_replay = ContinualResult(R=replay["R"], chance=chance).backward_transfer()
        acc_naive = ContinualResult(R=naive["R"], chance=chance).avg_accuracy()
        acc_replay = ContinualResult(R=replay["R"], chance=chance).avg_accuracy()

        lri_matched = abs(naive["lr_integral"] - replay["lr_integral"]) <= 0.02 * max(
            naive["lr_integral"], replay["lr_integral"], 1e-12
        )
        per_seed.append(
            {
                "seed": s,
                "bwt_naive": round(bwt_naive, 4),
                "bwt_replay": round(bwt_replay, 4),
                "bwt_delta": round(bwt_replay - bwt_naive, 4),
                "avgacc_naive": round(acc_naive, 4),
                "avgacc_replay": round(acc_replay, 4),
                "lr_integral_matched": lri_matched,
            }
        )

    bwt_deltas = [r["bwt_delta"] for r in per_seed]
    naive_bwts = [r["bwt_naive"] for r in per_seed]
    replay_bwts = [r["bwt_replay"] for r in per_seed]

    delta_ci = seed_ci(bwt_deltas, z=CI_Z)
    naive_ci = seed_ci(naive_bwts, z=CI_Z)
    replay_ci = seed_ci(replay_bwts, z=CI_Z)
    flips = sign_flip_report(bwt_deltas)
    all_matched = all(r["lr_integral_matched"] for r in per_seed)

    beats = delta_ci["lo"] > 0
    consistent = flips["consistent_sign"] == 1
    null_rejected = beats and consistent and all_matched

    if null_rejected:
        verdict = (
            "NULL REJECTED: experience replay beats naive sequential fine-tuning on BWT "
            "(delta CI lo > 0, consistent per-seed sign, matched LR-integral). Replay makes the "
            "shell measurably more moldable (less forgetting) on the frozen real substrate."
        )
    elif not all_matched:
        verdict = "NULL (compute-unmatched): LR-integrals differ between arms; comparison inadmissible."
    elif beats and not consistent:
        verdict = (
            "NULL (sign flip): mean BWT delta is positive but per-seed sign is inconsistent; a tie is a null."
        )
    elif delta_ci["lo"] <= 0 <= delta_ci["hi"]:
        verdict = "NULL (tie): BWT delta CI straddles 0; replay does not beat naive beyond seed spread."
    else:
        verdict = "NULL (replay worse or no better): BWT delta CI lower bound not above 0; null not rejected."

    result = {
        "experiment": "shell_continual_moldability_build2",
        "cache": str(CACHE),
        "substrate": "V-JEPA2 ViT-L frozen encoder, 1024-d, 200 clips (real)",
        "task_stream": "class-incremental sliding 2-class window over 5 shapes, T=4 tasks, shared 5-way MLP head",
        "mechanism": "experience replay (mop.shell.ReplayBuffer, reservoir + prioritized)",
        "baseline": "identical shell, replay off, naive sequential fine-tuning (matched arch/init/LR/steps/order)",
        "metric": "backward transfer (BWT, mop.metrics.continual); negative = forgetting",
        "seeds": SEEDS,
        "n_tasks": len(tasks),
        "chance": chance,
        "config": {
            "hidden": HIDDEN,
            "lr": LR,
            "epochs_per_task": EPOCHS_PER_TASK,
            "batch": BATCH,
            "replay_batch": REPLAY_BATCH,
            "replay_capacity": REPLAY_CAPACITY,
            "test_frac": TEST_FRAC,
        },
        "preregistered_null": "experience replay does NOT beat naive on BWT",
        "reject_rule": "BWT delta seed-CI lo > 0 AND no sign flip AND matched LR-integral; a tie is a null",
        "per_seed": per_seed,
        "bwt_naive_ci": naive_ci,
        "bwt_replay_ci": replay_ci,
        "bwt_delta_ci": delta_ci,
        "bwt_delta_sign_flips": flips,
        "lr_integral_matched_all": all_matched,
        "null_rejected": null_rejected,
        "verdict": verdict,
        "seconds": round(time.perf_counter() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
