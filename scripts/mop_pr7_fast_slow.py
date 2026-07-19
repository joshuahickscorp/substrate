#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.continual_metrics import adaptation_speed
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import parse_seeds, seed_everything
from mop.substrate.datasets import make_task_stream

MIN_MARGIN = 0.02  # preregistered minimum meaningful online-accuracy delta
RETENTION_MARGIN = 0.05  # fast_slow post-decay retention may trail slow_only by at most this
ARMS = ("fast_slow", "delta_rule", "slow_only", "buffer")


def verdict(deltas: list[float], min_margin: float = MIN_MARGIN) -> dict:
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    win = ci["mean"] > max(ci["sd"], min_margin) and not flips["any_flip"]
    return {"per_seed": [round(d, 4) for d in deltas], "ci": ci, "sign_flips": flips, "win": bool(win)}


class HebbianFastStore:

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


class DeltaRuleFastStore:

    def __init__(self, dim: int, n_classes: int, eta: float, decay: float):
        self.a = torch.zeros(n_classes, dim)
        self.eta, self.decay = eta, decay

    @staticmethod
    def _hat(z: torch.Tensor) -> torch.Tensor:
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def read(self, z: torch.Tensor) -> torch.Tensor:
        return self._hat(z) @ self.a.t()

    def write(self, z: torch.Tensor, y: torch.Tensor) -> None:
        zh = self._hat(z)  # [B, D]
        target = F.one_hot(y, self.a.shape[0]).float()  # [B, C]
        pred = zh @ self.a.t()  # [B, C] current store prediction
        err = target - pred  # [B, C] delta-rule error signal
        self.a = self.decay * self.a + self.eta * (err.t() @ zh)  # [C, D]

    def reset(self) -> None:
        self.a.zero_()


class EpisodicCache:

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
    return [sum(correct[i : i + chunk]) / len(correct[i : i + chunk]) for i in range(0, len(correct), chunk)]


class MotPR7FastSlow(Experiment):
    id = "mop_pr7_fast_slow"
    metric = ("online_acc", "adaptation_steps", "post_decay_retention")
    baseline = "the same slow SGD head alone, and the same head plus a matched-size episodic cache"
    ablation = (
        "Hebbian vs DeltaNet-style delta-rule fast store, present vs absent vs replaced by an "
        "equal-float-budget cache, all at matched capacity"
    )
    null_hypothesis = (
        "fast weights (Hebbian or delta-rule) tie the slow-only head and the matched-size cache on "
        "within-task adaptation, or the delta-rule merely ties the Hebbian floor, or the slow path "
        "fails to retain after the fast store decays"
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
                fast: HebbianFastStore | DeltaRuleFastStore | EpisodicCache | None = None
                if arm == "fast_slow":
                    fast = HebbianFastStore(dim, n_classes, eta=float(e.eta_fast), decay=float(e.fast_decay))
                elif arm == "delta_rule":
                    fast = DeltaRuleFastStore(
                        dim, n_classes, eta=float(e.eta_fast), decay=float(e.fast_decay)
                    )
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
        delta_deltas = [
            online["delta_rule"][i] - max(online["slow_only"][i], online["fast_slow"][i])
            for i in range(len(seeds))
        ]
        v_delta = verdict(delta_deltas)
        delta_retention_deltas = [
            retention["delta_rule"][i] - retention["slow_only"][i] for i in range(len(seeds))
        ]
        delta_retention_ok = bool(seed_ci(delta_retention_deltas)["mean"] >= -RETENTION_MARGIN)
        delta_win = bool(v_delta["win"] and delta_retention_ok)
        delta_vs_hebbian = [online["delta_rule"][i] - online["fast_slow"][i] for i in range(len(seeds))]
        v_delta_vs_hebbian = verdict(delta_vs_hebbian)
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
            "delta_rule": {
                "delta_online_vs_best_control": v_delta,
                "control_note": (
                    "delta_rule control = max(slow_only, fast_slow); the A6 preregistration requires it "
                    "to beat BOTH the slow-only baseline AND the Hebbian floor at matched capacity"
                ),
                "vs_hebbian_floor_only": v_delta_vs_hebbian,
                "retention_delta_vs_slow_only": {
                    "per_seed": [round(d, 4) for d in delta_retention_deltas],
                    "ci": seed_ci(delta_retention_deltas),
                    "margin": RETENTION_MARGIN,
                    "ok": delta_retention_ok,
                },
                "verdict_rule": (
                    "WIN iff mean per-seed delta (delta_rule online accuracy minus "
                    f"max(slow_only, fast_slow)) > max(seed SD, {MIN_MARGIN}) with no sign flip AND mean "
                    f"post-decay retention delta vs slow_only >= -{RETENTION_MARGIN}; else the "
                    "preregistered null (delta-rule ties the Hebbian floor, the deep-research-surprising "
                    "result) is supported"
                ),
                "null_supported": bool(not delta_win),
            },
            "seconds": round(time.perf_counter() - t0, 1),
        }
        return out


def default_cfg(**overrides) -> DictConfig:
    base = {
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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
    ap = argparse.ArgumentParser(
        description="PR7: delta-rule + Hebbian fast weights + slow SGD vs slow-only and cache"
    )
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default="runs/mot/pr7_delta_rule.json")
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
