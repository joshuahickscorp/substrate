
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..devices import DeviceInfo, safe_to
from ..metrics import accuracy
from ..shell import Consolidation, Neuromodulation, PlasticityController, ReplayBuffer
from ..substrate.datasets import Task


@dataclass
class TrainConfig:
    epochs_per_task: int = 3
    batch_size: int = 32
    replay_batch: int = 32
    base_lr: float = 1e-3
    adapt_threshold: float = 0.9


class Learner:
    def __init__(
        self,
        model: nn.Module,
        device: DeviceInfo,
        cfg: TrainConfig,
        buffer: ReplayBuffer | None = None,
        consolidation: Consolidation | None = None,
        plasticity: PlasticityController | None = None,
        neuromod: Neuromodulation | None = None,
        seed: int = 0,
    ):
        self.model = model.to(device.device)
        self.dev = device
        self.cfg = cfg
        self.buffer = buffer
        self.con = consolidation
        self.plast = plasticity
        self.nm = neuromod
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.base_lr)
        self.g = torch.Generator().manual_seed(seed)
        self.step = 0
        self.total_steps = 1

    def _batches(self, task: Task, bs: int):
        x, y = safe_to(task.x, self.dev.device), safe_to(task.y, self.dev.device)
        perm = torch.randperm(x.shape[0], generator=self.g)
        for i in range(0, x.shape[0], bs):
            j = perm[i : i + bs]
            yield x[j], y[j]

    def train_task(self, task: Task, progress0: float, progress1: float) -> int:
        if self.con:
            self.con.begin_task(self.model)
        adapt = -1
        local = 0
        steps_here = max(
            1, self.cfg.epochs_per_task * ((task.x.shape[0] + self.cfg.batch_size - 1) // self.cfg.batch_size)
        )
        for _ in range(self.cfg.epochs_per_task):
            for x, y in self._batches(task, self.cfg.batch_size):
                loss, acc = self._step(x, y, progress0 + (progress1 - progress0) * local / steps_here)
                if adapt < 0 and acc >= self.cfg.adapt_threshold:
                    adapt = local
                local += 1
                self.step += 1
        self._end_task(task)
        return adapt

    def _step(self, x, y, progress) -> tuple[float, float]:
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        signal = float(loss.detach())
        if self.buffer is not None and len(self.buffer) > 0:
            rb = self.buffer.sample(self.cfg.replay_batch)
            rx, ry, rw = (
                safe_to(rb["x"], self.dev.device),
                safe_to(rb["y"], self.dev.device),
                safe_to(rb["is_weight"], self.dev.device),
            )
            rl = (F.cross_entropy(self.model(rx), ry, reduction="none") * rw).mean()
            loss = loss + rl
        if self.con is not None:
            loss = loss + self.con.penalty(self.model)
        if self.plast is not None:
            loss = loss + self.plast.rigidity_penalty(self.model)
        loss.backward()
        if self.plast is not None:
            self.plast.apply_pnn_freeze(self.model)
            gate = self.nm.gate("surprise", signal) if self.nm is not None else 1.0
            lr = (
                self.plast.lr(progress, signal=self.nm.stats["surprise"].z(signal) if self.nm else 0.0) * gate
            )
            for grp in self.opt.param_groups:
                grp["lr"] = lr
        if self.con is not None:
            self.con.before_step(self.model)  # snapshot theta_{t-1} for the SI path integral
        self.opt.step()
        if self.con is not None:
            self.con.after_step(self.model)  # accumulate -grad * (theta_t - theta_{t-1})
        return float(loss.detach()), accuracy(logits.detach(), y)

    def _end_task(self, task: Task) -> None:
        if self.buffer is not None:
            with torch.no_grad():
                x = safe_to(task.x, self.dev.device)
                ce = F.cross_entropy(self.model(x), safe_to(task.y, self.dev.device), reduction="none")
            self.buffer.add(task.x, task.y, key=task.x, priority=ce.detach().cpu())
        if self.con is not None:
            xb, yb = safe_to(task.x, self.dev.device), safe_to(task.y, self.dev.device)
            batches = [(xb, yb)]

            def loss_fn(b):
                return F.cross_entropy(self.model(b[0]), b[1])

            self.con.consolidate(self.model, batches, loss_fn)
        if self.plast is not None:
            self.plast.update_rigidity(self.model)

    @torch.no_grad()
    def evaluate(self, tasks: list[Task]) -> list[float]:
        self.model.eval()
        accs = []
        for t in tasks:
            x, y = safe_to(t.x, self.dev.device), safe_to(t.y, self.dev.device)
            accs.append(accuracy(self.model(x), y))
        self.model.train()
        return accs
