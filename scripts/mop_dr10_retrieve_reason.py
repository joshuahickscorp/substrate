#!/usr/bin/env python
"""DR10: memory-first, retrieve-then-reason (WP-06, H-MEMORY; registry
docs/mixture_of_perspectives/11_experiment_registry.md, DR full schema part 2).

Thesis: kNN-retrieving similar cached latents from the episodic buffer (shell/buffer.py) and
CONDITIONING the refiner on them beats from-scratch reasoning and random retrieval at matched
compute. The frozen-encoder twist: stored latents never go stale, so the only live question is
whether the FROZEN neighbor metric (raw latent L2) is task-aligned.

Task (synthetic latents, no encoder): a fixed universe of classes with fine-grained local
sub-cluster structure in a signal subspace, plus high-variance nuisance dimensions that the raw L2
metric cannot tell apart from signal (this is what makes the null live: retrieval only helps if
raw-latent neighbors are task-aligned DESPITE the nuisance dims). A held-out set of NEW classes
never appears as a training query; they exist only as k-shot entries in the episodic buffer, so
new-class accuracy is reachable only through memory (the new-class few-shot regime of the registry
row).

Arms (identical architecture, identical meta-training compute and params; only the retrieval
channel differs, which is the matched-compute discipline):
  retrieval    : neighbors = buffer.retrieve (kNN over frozen keys)
  from_scratch : the retrieval channel is fed zeros at train AND eval (reasoning without memory)
  random       : neighbors retrieved through PERMUTED keys (label-honest values, random w.r.t. the
                 query; the registry's random-retrieval control)

Preregistered null (in code before any run): retrieval accuracy on new classes ties
max(from_scratch, random) within the verdict rule below (the frozen neighbor metric is not
task-aligned). Verdict rule, uniform across the MoT memory scripts: WIN iff the per-seed delta
mean exceeds max(per-seed SD, MIN_MARGIN) with no sign flip; else null_supported=True.

kNN FLOPs are counted locally (2 * n_mem * dim per query, the cdist mult+add count); the WP-02
attention/kNN counters are a pass-2 deliverable and are not imported here.

Usage: python scripts/mop_dr10_retrieve_reason.py --seeds 0-4
Writes runs/mot/dr10_retrieve_reason.json. No em or en dashes (BLACKHOLE.md).
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

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.continual_metrics import adaptation_speed
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import seed_everything
from mop.shell.buffer import ReplayBuffer
from mop.shell.refine import IterativeRefiner

MIN_MARGIN = 0.02  # preregistered minimum meaningful accuracy delta
ARMS = ("retrieval", "from_scratch", "random")


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


def make_universe(seed: int, e: DictConfig) -> dict:
    """Fixed class universe: n_base + n_new classes, m_sub local sub-clusters per class living in
    the first d_signal dims, plus d_nuisance high-variance nuisance dims appended to every sample
    (the same nuisance process for all classes, so it carries zero label information but dominates
    raw L2 distance when nuisance_scale is large)."""
    g = torch.Generator().manual_seed(seed)
    n_classes = int(e.n_base) + int(e.n_new)
    centers = torch.randn(n_classes, int(e.m_sub), int(e.d_signal), generator=g) * float(e.separation)
    return {"centers": centers, "g": g, "n_classes": n_classes}


def sample_classes(uni: dict, e: DictConfig, classes: list[int], per_class: int) -> tuple:
    """per_class samples from each listed class: sub-cluster center + 0.5 noise in the signal dims,
    nuisance_scale * noise in the nuisance dims."""
    g, centers = uni["g"], uni["centers"]
    xs, ys = [], []
    for c in classes:
        which = torch.randint(0, int(e.m_sub), (per_class,), generator=g)
        sig = centers[c, which] + 0.5 * torch.randn(per_class, int(e.d_signal), generator=g)
        nui = float(e.nuisance_scale) * torch.randn(per_class, int(e.d_nuisance), generator=g)
        xs.append(torch.cat([sig, nui], dim=-1))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


class RetrieveReasonNet(nn.Module):
    """The retrieval-conditioned reasoner: the query latent is shifted by a learned projection of
    the (distance-weighted) retrieved-neighbor mean and label histogram, then refined by the
    existing IterativeRefiner and read by a linear head. The from_scratch arm feeds zeros into the
    SAME conditioning path, so params and per-forward FLOPs are identical across arms."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 64, steps: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.cond = nn.Linear(dim + n_classes, dim)
        self.refiner = IterativeRefiner(dim, hidden=hidden, steps=steps)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, z: torch.Tensor, neigh_x: torch.Tensor, neigh_y: torch.Tensor) -> torch.Tensor:
        """z [B,D]; neigh_x [B,k,D]; neigh_y [B,k] (long). Zero-neighbor arms pass zeros tensors."""
        dist = (z.unsqueeze(1) - neigh_x).norm(dim=-1)  # true content distance, all arms alike
        w = torch.softmax(-dist, dim=-1).unsqueeze(-1)  # [B,k,1]
        r = (w * neigh_x).sum(1)
        h = (w * F.one_hot(neigh_y, self.n_classes).float()).sum(1)
        zc = z + self.cond(torch.cat([r, h], dim=-1))
        out, _ = self.refiner(zc)
        return self.head(out)


def build_buffer(x: torch.Tensor, y: torch.Tensor, seed: int, permute_keys: bool = False) -> ReplayBuffer:
    """The episodic store. permute_keys=True builds the random-retrieval control: values (x, y) are
    honest but the retrieval keys belong to OTHER items, so neighbors are random w.r.t. the query
    while every downstream computation stays identical."""
    buf = ReplayBuffer(capacity=x.shape[0], dim=x.shape[1], prioritized=False, eviction="fifo", seed=seed)
    if permute_keys:
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed + 13))
        buf.add(x, y, key=x[perm])
    else:
        buf.add(x, y)
    return buf


def neighbors_for(arm: str, buf: ReplayBuffer, rand_buf: ReplayBuffer, z: torch.Tensor, k: int) -> tuple:
    if arm == "from_scratch":
        b = z.shape[0]
        return z.new_zeros(b, k, z.shape[1]), torch.zeros(b, k, dtype=torch.long)
    src = rand_buf if arm == "random" else buf
    ret = src.retrieve(z, k=k)
    return ret["x"], ret["y"]


def train_arm(
    arm: str,
    net: RetrieveReasonNet,
    buf: ReplayBuffer,
    rand_buf: ReplayBuffer,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    e: DictConfig,
    seed: int,
) -> None:
    g = torch.Generator().manual_seed(seed + 31)
    opt = torch.optim.Adam(net.parameters(), lr=float(e.lr))
    n, batch = xtr.shape[0], int(e.batch)
    for _ in range(int(e.epochs)):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            nx, ny = neighbors_for(arm, buf, rand_buf, xtr[idx], int(e.k))
            opt.zero_grad()
            F.cross_entropy(net(xtr[idx], nx, ny), ytr[idx]).backward()
            opt.step()


@torch.no_grad()
def eval_arm(
    arm: str,
    net: RetrieveReasonNet,
    buf: ReplayBuffer,
    rand_buf: ReplayBuffer,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
) -> float:
    nx, ny = neighbors_for(arm, buf, rand_buf, x, k)
    return float((net(x, nx, ny).argmax(-1) == y).float().mean())


class MotDR10RetrieveReason(Experiment):
    id = "mop_dr10_retrieve_reason"
    metric = ("new_class_acc", "overall_acc", "adaptation_speed_shots", "knn_flops_per_query")
    baseline = "from-scratch reasoning (same net, zeroed retrieval channel) at matched compute and params"
    ablation = "kNN retrieval conditioning vs zeroed conditioning vs permuted-key random retrieval"
    null_hypothesis = (
        "retrieval ties max(from-scratch, random retrieval) on new-class accuracy: the frozen "
        "neighbor metric is not task-aligned"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        dim = int(e.d_signal) + int(e.d_nuisance)
        base_classes = list(range(int(e.n_base)))
        new_classes = list(range(int(e.n_base), int(e.n_base) + int(e.n_new)))
        per = {a: {"new": [], "base": [], "overall": []} for a in ARMS}
        shots_curves: list[list[float]] = []
        for s in seeds:
            seed_everything(s)
            uni = make_universe(s, e)
            n_classes = uni["n_classes"]
            # memory content: mem_per_class per base class, k_shot per new class (the few-shot part)
            mem_x, mem_y = sample_classes(uni, e, base_classes, int(e.mem_per_class))
            shot_x, shot_y = sample_classes(uni, e, new_classes, int(e.k_shot))
            buf_x = torch.cat([mem_x, shot_x])
            buf_y = torch.cat([mem_y, shot_y])
            buf = build_buffer(buf_x, buf_y, seed=s)
            rand_buf = build_buffer(buf_x, buf_y, seed=s, permute_keys=True)
            # meta-training queries: base classes only (new classes exist ONLY in memory)
            xtr, ytr = sample_classes(uni, e, base_classes, int(e.train_per_class))
            xte_b, yte_b = sample_classes(uni, e, base_classes, int(e.test_per_class))
            xte_n, yte_n = sample_classes(uni, e, new_classes, int(e.test_per_class))
            nets = {}
            for arm in ARMS:
                torch.manual_seed(s)  # identical init across arms
                net = RetrieveReasonNet(dim, n_classes, hidden=int(e.hidden), steps=int(e.refine_steps))
                train_arm(arm, net, buf, rand_buf, xtr, ytr, e, seed=s)
                nets[arm] = net
                acc_b = eval_arm(arm, net, buf, rand_buf, xte_b, yte_b, int(e.k))
                acc_n = eval_arm(arm, net, buf, rand_buf, xte_n, yte_n, int(e.k))
                n_all = xte_b.shape[0] + xte_n.shape[0]
                per[arm]["base"].append(acc_b)
                per[arm]["new"].append(acc_n)
                per[arm]["overall"].append((acc_b * xte_b.shape[0] + acc_n * xte_n.shape[0]) / n_all)
            # adaptation-vs-shots curve for the retrieval arm: new-class accuracy as the buffer
            # grows from 1..k_shot shots per new class (the registry's adaptation-speed metric)
            curve = []
            for shots in range(1, int(e.k_shot) + 1):
                keep = torch.cat(
                    [torch.arange(len(new_classes)) * int(e.k_shot) + j for j in range(shots)]
                ).sort()[0]
                sb_x = torch.cat([mem_x, shot_x[keep]])
                sb_y = torch.cat([mem_y, shot_y[keep]])
                sbuf = build_buffer(sb_x, sb_y, seed=s)
                curve.append(eval_arm("retrieval", nets["retrieval"], sbuf, rand_buf, xte_n, yte_n, int(e.k)))
            shots_curves.append(curve)
        deltas = [
            per["retrieval"]["new"][i] - max(per["from_scratch"]["new"][i], per["random"]["new"][i])
            for i in range(len(seeds))
        ]
        v = verdict(deltas)
        n_mem = int(e.n_base) * int(e.mem_per_class) + int(e.n_new) * int(e.k_shot)
        mean_curve = [
            sum(c[i] for c in shots_curves) / len(shots_curves) for i in range(len(shots_curves[0]))
        ]
        out = {
            "experiment": self.id,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "arms": {a: {k: seed_ci(vs) for k, vs in per[a].items()} for a in ARMS},
            "new_class_acc": {a: per[a]["new"] for a in ARMS},
            "overall_acc": {a: per[a]["overall"] for a in ARMS},
            "delta_retrieval_vs_best_control": v,
            "adaptation_shots_curve_mean": [round(x, 4) for x in mean_curve],
            "adaptation_speed_shots": adaptation_speed(mean_curve),
            "knn_flops_per_query": 2 * n_mem * dim,
            "verdict_rule": (
                "WIN iff mean per-seed delta (retrieval minus max(from_scratch, random) on new-class "
                f"accuracy) > max(seed SD, {MIN_MARGIN}) with no sign flip; else the preregistered null "
                "(frozen neighbor metric not task-aligned) is supported"
            ),
            "null_supported": bool(not v["win"]),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        return out


def default_cfg(**overrides) -> DictConfig:
    base = {
        "seeds": [0, 1, 2, 3, 4],
        "d_signal": 16,
        "d_nuisance": 48,
        "nuisance_scale": 2.0,
        "separation": 3.0,
        "m_sub": 4,
        "n_base": 8,
        "n_new": 4,
        "mem_per_class": 40,
        "k_shot": 5,
        "train_per_class": 60,
        "test_per_class": 40,
        "k": 10,
        "hidden": 64,
        "refine_steps": 2,
        "epochs": 20,
        "batch": 64,
        "lr": 1e-3,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="DR10: retrieve-then-reason vs from-scratch and random retrieval"
    )
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="runs/mot/dr10_retrieve_reason.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds), epochs=a.epochs)
    result = MotDR10RetrieveReason().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
