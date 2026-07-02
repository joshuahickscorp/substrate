#!/usr/bin/env python
"""PR7: fast/slow two-timescale weights (WP-06, the fast/slow half of H-MEMORY; registry
docs/mixture_of_thinking/11_experiment_registry.md, PR full schema part 2).

Thesis: a two-timescale system (a gradient-free Hebbian outer-product fast store on top of a slow
SGD head) adapts faster WITHIN a task at matched capacity and still retains after the fast store
decays (the slow path must have kept learning, not been parasitized). This supplies the System-1
mode to MoT if it survives.

Mechanism: the fast store is a WorkingMemory-style label-keyed slot matrix A [C, D] (one slot per
class, exactly the modulation.WorkingMemory read/write/decay interface, written by outer products
onehot(y) x z_hat with per-step decay instead of a learned gate, because the store must be
label-keyed and gradient-free). Read is additive: logits = slow(z) + fast_gain * (z_hat @ A.T).

Stream: domain-incremental tasks (make_task_stream, independent cluster geometry per task, shared
label space), processed ONLINE (predict-then-train per batch). At each task boundary the fast
state fully decays (A -> 0, cache cleared) and slow-only retention is measured on that task's
held-out split.

Arms (same slow head, same init, same stream order; only the fast state differs):
  fast_slow : slow SGD head + Hebbian fast store (extra state C*D floats)
  slow_only : slow SGD head alone (the ablation)
  buffer    : slow SGD head + a matched-size episodic cache, floor(C*D / (D+1)) recent (z, y)
              items read by a distance-weighted kNN vote at the same additive gain (the
              matched-capacity control: is the fast store just a small cache)

Preregistered null (fixed before running): fast_slow ties slow_only AND ties the matched-size
buffer on within-task online accuracy, or fails retention after decay. Verdict rule, uniform
across the MoT memory scripts: WIN iff the per-seed delta mean (fast_slow online accuracy minus
max(slow_only, buffer)) exceeds max(per-seed SD, MIN_MARGIN) with no sign flip AND post-decay
retention of fast_slow is within RETENTION_MARGIN of slow_only; else null_supported=True.

Usage: python scripts/mot_pr7_fast_slow.py --seeds 0-4
Writes runs/mot/pr7_fast_slow.json. No em or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from devsys.devices import DeviceInfo, resolve
from devsys.diagnostics.continual_metrics import adaptation_speed
from devsys.diagnostics.riskcov import seed_ci, sign_flip_report
from devsys.experiments.base import Experiment
from devsys.seeding import seed_everything
from devsys.substrate.datasets import make_task_stream

MIN_MARGIN = 0.02  # preregistered minimum meaningful online-accuracy delta
RETENTION_MARGIN = 0.05  # fast_slow post-decay retention may trail slow_only by at most this
ARMS = ("fast_slow", "slow_only", "buffer")


def parse_seeds(spec: str) -> list[int]:
    """'0-4' -> [0..4]; '0,2,5' -> [0,2,5]; '3' -> [3]."""
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def verdict(deltas: list[float], min_margin: float = MIN_MARGIN) -> dict:
    """The uniform preregistered rule: WIN iff mean delta > max(sd, min_margin) and no sign flip."""
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    win = ci["mean"] > max(ci["sd"], min_margin) and not flips["any_flip"]
    return {"per_seed": [round(d, 4) for d in deltas], "ci": ci, "sign_flips": flips, "win": bool(win)}


class HebbianFastStore:
    """The gradient-free fast path: per-class slots A [C, D] written by decayed outer products
    A <- decay * A + eta * onehot(y) x z_hat, read additively as z_hat @ A.T. State only, no
    trainable params; capacity = C * D floats (the number the buffer control is matched to)."""

    def __init__(self, dim: int, n_classes: int, eta: float, decay: float):
        self.a = torch.zeros(n_classes, dim)
        self.eta, self.decay = eta, decay

    @staticmethod
    def _hat(z: torch.Tensor) -> torch.Tensor:
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def read(self, z: torch.Tensor) -> torch.Tensor:
        return self._hat(z) @ self.a.t()

    def write(self, z: torch.Tensor, y: torch.Tensor) -> None:
        self.a = self.decay * self.a + self.eta * (F.one_hot(y, self.a.shape[0]).float().t() @ self._hat(z))

    def reset(self) -> None:
        self.a.zero_()


class EpisodicCache:
    """The matched-size control: a FIFO cache of floor(C*D / (D+1)) recent (z, y) items (each item
    costs D+1 floats, so the float budget equals the fast store's C*D), read by a distance-weighted
    kNN vote at the same additive gain."""

    def __init__(self, dim: int, n_classes: int, k: int):
        self.capacity = max(1, (n_classes * dim) // (dim + 1))
        self.n_classes = n_classes
        self.k = k
        self.x: list[torch.Tensor] = []
        self.y: list[int] = []

    def read(self, z: torch.Tensor) -> torch.Tensor:
        if not self.x:
            return z.new_zeros(z.shape[0], self.n_classes)
        mx = torch.stack(self.x)
        my = torch.tensor(self.y, dtype=torch.long)
        dist = torch.cdist(z, mx)
        k = min(self.k, len(self.x))
        d, i = dist.topk(k, largest=False)
        w = torch.softmax(-d, dim=-1).unsqueeze(-1)
        return (w * F.one_hot(my[i], self.n_classes).float()).sum(1)

    def write(self, z: torch.Tensor, y: torch.Tensor) -> None:
        for j in range(z.shape[0]):
            self.x.append(z[j].clone())
            self.y.append(int(y[j]))
        overflow = len(self.x) - self.capacity
        if overflow > 0:
            self.x, self.y = self.x[overflow:], self.y[overflow:]

    def reset(self) -> None:
        self.x, self.y = [], []


def chunk_curve(correct: list[float], chunk: int) -> list[float]:
    """Per-step 0/1 correctness -> chunk-mean accuracy curve (the adaptation curve)."""
    return [sum(correct[i : i + chunk]) / len(correct[i : i + chunk]) for i in range(0, len(correct), chunk)]


class MotPR7FastSlow(Experiment):
    id = "mot_pr7_fast_slow"
    metric = ("online_acc", "adaptation_steps", "post_decay_retention")
    baseline = "the same slow SGD head alone, and the same head plus a matched-size episodic cache"
    ablation = "Hebbian fast store present vs absent vs replaced by an equal-float-budget cache"
    null_hypothesis = (
        "fast weights tie the slow-only head and the matched-size cache on within-task adaptation, "
        "or the slow path fails to retain after the fast store decays"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        dim, n_classes = int(e.dim), int(e.classes_per_task)
        online = {a: [] for a in ARMS}
        adapt = {a: [] for a in ARMS}
        retention = {a: [] for a in ARMS}
        for s in seeds:
            seed_everything(s)
            tasks = make_task_stream(
                n_tasks=int(e.n_tasks),
                dim=dim,
                classes_per_task=n_classes,
                samples_per_task=int(e.samples_per_task),
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            arms: dict[str, dict] = {}
            for arm in ARMS:
                torch.manual_seed(s)  # identical slow-head init across arms
                head = nn.Linear(dim, n_classes)
                fast: HebbianFastStore | EpisodicCache | None = None
                if arm == "fast_slow":
                    fast = HebbianFastStore(dim, n_classes, eta=float(e.eta_fast), decay=float(e.fast_decay))
                elif arm == "buffer":
                    fast = EpisodicCache(dim, n_classes, k=int(e.cache_k))
                arms[arm] = {
                    "head": head,
                    "fast": fast,
                    "opt": torch.optim.SGD(head.parameters(), lr=float(e.lr_slow)),
                    "correct": {a: [] for a in range(int(e.n_tasks))},
                }
            for t, task in enumerate(tasks):
                n = task.x.shape[0]
                cut = int(n * float(e.train_frac))
                xtr, ytr, xte, yte = task.x[:cut], task.y[:cut], task.x[cut:], task.y[cut:]
                batch = int(e.batch)
                for arm in ARMS:
                    st = arms[arm]
                    head, fast, opt = st["head"], st["fast"], st["opt"]
                    for i in range(0, xtr.shape[0], batch):
                        xb, yb = xtr[i : i + batch], ytr[i : i + batch]
                        with torch.no_grad():  # predict BEFORE training on this batch
                            logits = head(xb)
                            if fast is not None:
                                logits = logits + float(e.fast_gain) * fast.read(xb)
                            st["correct"][t].extend((logits.argmax(-1) == yb).float().tolist())
                        opt.zero_grad()
                        F.cross_entropy(head(xb), yb).backward()
                        opt.step()
                        if fast is not None:
                            fast.write(xb, yb)
                    # task boundary: the fast state fully decays; retention is slow-only
                    if fast is not None:
                        fast.reset()
                    with torch.no_grad():
                        st.setdefault("retention", []).append(
                            float((head(xte).argmax(-1) == yte).float().mean())
                        )
            for arm in ARMS:
                st = arms[arm]
                all_correct = [c for t in range(int(e.n_tasks)) for c in st["correct"][t]]
                online[arm].append(sum(all_correct) / len(all_correct))
                steps = []
                for t in range(int(e.n_tasks)):
                    curve = chunk_curve(st["correct"][t], int(e.curve_chunk))
                    steps.append(adaptation_speed(curve, target_frac=0.9)["steps"])
                adapt[arm].append(sum(steps) / len(steps))
                retention[arm].append(sum(st["retention"]) / len(st["retention"]))
        deltas = [
            online["fast_slow"][i] - max(online["slow_only"][i], online["buffer"][i])
            for i in range(len(seeds))
        ]
        v = verdict(deltas)
        retention_deltas = [retention["fast_slow"][i] - retention["slow_only"][i] for i in range(len(seeds))]
        retention_ok = bool(seed_ci(retention_deltas)["mean"] >= -RETENTION_MARGIN)
        win = bool(v["win"] and retention_ok)
        out = {
            "experiment": self.id,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "capacity_note": (
                "fast store = C*D floats; cache = floor(C*D/(D+1)) items of D+1 floats (matched float "
                "budget); slow head identical across arms"
            ),
            "online_acc": {a: {"per_seed": online[a], "ci": seed_ci(online[a])} for a in ARMS},
            "adaptation_steps_mean": {a: {"per_seed": adapt[a], "ci": seed_ci(adapt[a])} for a in ARMS},
            "post_decay_retention": {
                a: {"per_seed": retention[a], "ci": seed_ci(retention[a])} for a in ARMS
            },
            "delta_online_vs_best_control": v,
            "retention_delta_vs_slow_only": {
                "per_seed": [round(d, 4) for d in retention_deltas],
                "ci": seed_ci(retention_deltas),
                "margin": RETENTION_MARGIN,
                "ok": retention_ok,
            },
            "verdict_rule": (
                "WIN iff mean per-seed delta (fast_slow online accuracy minus max(slow_only, buffer)) "
                f"> max(seed SD, {MIN_MARGIN}) with no sign flip AND mean post-decay retention delta vs "
                f"slow_only >= -{RETENTION_MARGIN}; else the preregistered null (fast store is redundant "
                "capacity or a small cache, or parasitizes the slow path) is supported"
            ),
            "null_supported": bool(not win),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        return out


def default_cfg(**overrides) -> DictConfig:
    base = {
        "seeds": [0, 1, 2, 3, 4],
        "dim": 64,
        "classes_per_task": 8,
        "n_tasks": 4,
        "samples_per_task": 640,
        "separation": 1.6,
        "train_frac": 0.8,
        "batch": 8,
        "lr_slow": 0.02,
        "eta_fast": 0.5,
        "fast_decay": 0.995,
        "fast_gain": 2.0,
        "cache_k": 5,
        "curve_chunk": 8,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR7: Hebbian fast weights + slow SGD vs slow-only and cache")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default="runs/mot/pr7_fast_slow.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds))
    result = MotPR7FastSlow().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
