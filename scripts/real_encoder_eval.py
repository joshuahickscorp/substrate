#!/usr/bin/env python
"""REAL-ENCODER evaluation on the cached real V-JEPA latents. Answers two real-encoder
questions (tagged real-encoder, structured-synthetic-video content):
  1. Is visual-class info linearly decodable from the real V-JEPA latent? (linear probe)
  2. Does a frozen-head continual learner forget, and does replay+EWC retain, on REAL latents?
Run after scripts/cache_real_encoder.py has populated the store.

Usage: python scripts/real_encoder_eval.py [--store data/cache/vjepa2_vitl_fpc64_256_real] [--tasks 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from devsys.config import REPO_ROOT
from devsys.devices import resolve
from devsys.diagnostics import linear_probe
from devsys.learning.backprop import Learner, TrainConfig
from devsys.metrics import ContinualResult
from devsys.seeding import seed_everything
from devsys.shell import Consolidation, ReplayBuffer
from devsys.shell.heads import ClassHead
from devsys.substrate import LatentStore, stream_from_store


def _split(task, frac=0.7):
    n = max(1, int(task.x.shape[0] * frac))
    from devsys.substrate.datasets import Task

    tr = Task(task.name, task.x[:n], task.y[:n], n_classes=task.n_classes, task_id=task.task_id)
    te = Task(task.name, task.x[n:], task.y[n:], n_classes=task.n_classes, task_id=task.task_id)
    return tr, te


def _run_arm(train, test, dim, n_classes, dev, replay=False, prioritized=True, ewc=False, seed=0):
    seed_everything(seed)
    model = ClassHead(dim, n_classes, hidden=64, depth=2)
    buf = (
        ReplayBuffer(capacity=2048, dim=dim, prioritized=prioritized, index="brute", seed=seed)
        if replay
        else None
    )
    con = (
        Consolidation(
            OmegaConf.create(
                {"method": "ewc", "ewc_lambda": 1000.0, "si_c": 0.1, "si_xi": 1e-3, "fisher_samples": 32}
            )
        )
        if ewc
        else None
    )
    lr = Learner(
        model,
        dev,
        TrainConfig(epochs_per_task=8, batch_size=16, base_lr=1e-3),
        buffer=buf,
        consolidation=con,
        seed=seed,
    )
    R = []
    for i, t in enumerate(train):
        lr.train_task(t, i / len(train), (i + 1) / len(train))
        R.append(lr.evaluate(test))
    return ContinualResult(R=R, chance=1.0 / n_classes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(REPO_ROOT / "data/cache/vjepa2_vitl_fpc64_256_real"))
    ap.add_argument("--tasks", type=int, default=3)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if not Path(a.store).exists():
        print(f"FAIL: real store missing at {a.store}; run scripts/cache_real_encoder.py first")
        return 1
    dev = resolve("cpu")
    store = LatentStore.open(Path(a.store))
    x, y = store.latents(), store.labels()
    n_classes = int(y.max()) + 1

    probe = linear_probe(x, y, classification=True, epochs=400)
    raw = stream_from_store(store, n_tasks=a.tasks)
    train = [_split(t)[0] for t in raw]
    test = [_split(t)[1] for t in raw]
    d = x.shape[1]
    # continual arms on REAL latents: naive, random replay, prioritized replay, replay+EWC.
    naive = _run_arm(train, test, d, n_classes, dev)
    rnd = _run_arm(train, test, d, n_classes, dev, replay=True, prioritized=False)
    prio = _run_arm(train, test, d, n_classes, dev, replay=True, prioritized=True)
    prot = _run_arm(train, test, d, n_classes, dev, replay=True, prioritized=True, ewc=True)

    out = {
        "provenance": "real-encoder (real V-JEPA weights, structured-synthetic video content)",
        "n_latents": len(store),
        "n_classes": n_classes,
        "linear_probe": {"acc": probe["score"], "chance": probe["chance"], "decodable": probe["decodable"]},
        "continual": {
            "naive_bwt": naive.backward_transfer(),
            "protected_bwt": prot.backward_transfer(),
            "naive_final_acc": naive.avg_accuracy(),
            "protected_final_acc": prot.avg_accuracy(),
            "protected_retains_better": prot.backward_transfer() > naive.backward_transfer(),
        },
        "replay_schemes_E2_real": {
            "naive_bwt": naive.backward_transfer(),
            "random_replay_bwt": rnd.backward_transfer(),
            "prioritized_replay_bwt": prio.backward_transfer(),
            "replay_ewc_bwt": prot.backward_transfer(),
            "replay_beats_naive": rnd.backward_transfer() > naive.backward_transfer(),
            "prioritized_beats_random": prio.backward_transfer() > rnd.backward_transfer() + 0.02,
        },
    }
    (REPO_ROOT / "runs" / "real_encoder_eval.json").parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "runs" / "real_encoder_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
