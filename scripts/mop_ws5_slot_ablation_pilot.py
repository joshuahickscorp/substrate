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
from mop.diagnostics.compute import param_count
from mop.diagnostics.continual_metrics import backward_transfer, forgetting_area
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import parse_seeds, seed_everything
from mop.shell.heads import MoEHead, moe_expert_hidden_for_dense, routing_entropy
from mop.substrate.real_latent import real_task_stream

MIN_MARGIN = 0.02  # preregistered minimum meaningful BWT delta
ARMS = ("full", "ablated")


def verdict(deltas: list[float], min_margin: float = MIN_MARGIN) -> dict:
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    win = ci["mean"] > max(ci["sd"], min_margin) and not flips["any_flip"]
    return {"per_seed": [round(d, 4) for d in deltas], "ci": ci, "sign_flips": flips, "win": bool(win)}


def split_task(x: torch.Tensor, y: torch.Tensor, train_frac: float, g: torch.Generator):
    tr, te = [], []
    for c in sorted(set(y.tolist())):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.shape[0], generator=g)]
        n_tr = min(max(1, int(round(train_frac * idx.shape[0]))), idx.shape[0] - 1)
        tr.append(idx[:n_tr])
        te.append(idx[n_tr:])
    return torch.cat(tr), torch.cat(te)


def load_split_stream(e: DictConfig, seed: int) -> list[dict]:
    tasks = real_task_stream(
        str(e.cache),
        n_tasks=int(e.n_tasks),
        incremental="class",
        seed=seed,
        data_dir=str(e.data_dir),
    )
    g = torch.Generator().manual_seed(seed + 1)
    out = []
    for t in tasks:
        tr, te = split_task(t.x, t.y, float(e.train_frac), g)
        out.append(
            {
                "xtr": t.x[tr],
                "ytr": t.y[tr],
                "xte": t.x[te],
                "yte": t.y[te],
                "n_classes": t.n_classes,
            }
        )
    return out


class SlotRoutedNet(nn.Module):

    def __init__(self, dim: int, n_classes: int, n_experts: int, expert_hidden: int, use_slot: bool):
        super().__init__()
        self.use_slot = bool(use_slot)
        self.slot = nn.Parameter(torch.zeros(dim))
        self.read = nn.Linear(dim, dim)
        self.read_gate = nn.Parameter(torch.zeros(1))
        self.moe = MoEHead(dim, n_classes, n_experts, expert_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_slot:
            x = x + torch.sigmoid(self.read_gate) * self.read(self.slot).unsqueeze(0)
        return self.moe(x)


def train_stream(model: SlotRoutedNet, stream: list[dict], e: DictConfig) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=float(e.lr))
    n_tasks = len(stream)
    acc = [[0.0] * n_tasks for _ in range(n_tasks)]
    for i, task in enumerate(stream):
        for _ in range(int(e.epochs_per_task)):
            opt.zero_grad()
            F.cross_entropy(model(task["xtr"]), task["ytr"]).backward()
            opt.step()
        with torch.no_grad():
            for j, tj in enumerate(stream):
                acc[i][j] = float((model(tj["xte"]).argmax(-1) == tj["yte"]).float().mean())
    with torch.no_grad():
        model(torch.cat([t["xte"] for t in stream]))
        ent = routing_entropy(model.moe.last_gates)
    return {
        "acc_matrix": acc,
        "bwt": backward_transfer(acc),
        "task0_forgetting_area": forgetting_area([acc[i][0] for i in range(n_tasks)]),
        "final_mean_acc": sum(acc[-1][j] for j in range(n_tasks)) / n_tasks,
        "routing_entropy": ent,
    }


class MotWS5SlotAblationPilot(Experiment):
    id = "mop_ws5_router_slot"
    metric = ("backward_transfer", "final_mean_acc", "task0_forgetting_area", "routing_entropy")
    baseline = "the identical slot-ablated network (slot read zeroed at fixed routing and capacity)"
    ablation = "shared broadcast slot read on vs off; nothing else differs"
    null_hypothesis = (
        "the shared slot adds nothing over sparse routing alone: slot-ablation ties the full model "
        "on BWT and accuracy at matched capacity; sparse routing does all the work"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        hidden = int(e.hidden)
        per: dict[str, dict[str, list[float]]] = {
            a: {"bwt": [], "final_mean_acc": [], "task0_forgetting_area": [], "routing_entropy": []}
            for a in ARMS
        }
        params: dict[str, int] = {}
        for s in seeds:
            seed_everything(s)
            stream = load_split_stream(e, s)
            dim = stream[0]["xtr"].shape[1]
            n_classes = stream[0]["n_classes"]
            expert_hidden = moe_expert_hidden_for_dense(dim, hidden, n_classes, int(e.n_experts))
            for arm in ARMS:
                torch.manual_seed(s)  # identical init across arms (identical shapes)
                net = SlotRoutedNet(dim, n_classes, int(e.n_experts), expert_hidden, arm == "full")
                params[arm] = param_count(net)
                rep = train_stream(net, stream, e)
                for k in per[arm]:
                    per[arm][k].append(rep[k])
        deltas_bwt = [per["full"]["bwt"][i] - per["ablated"]["bwt"][i] for i in range(len(seeds))]
        deltas_acc = [
            per["full"]["final_mean_acc"][i] - per["ablated"]["final_mean_acc"][i] for i in range(len(seeds))
        ]
        v_bwt = verdict(deltas_bwt)
        out = {
            "experiment": self.id,
            "pilot": True,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "param_counts": params,
            "param_match_ok": bool(params["full"] == params["ablated"]),
            "arms": {a: {k: seed_ci(vs) for k, vs in per[a].items()} for a in ARMS},
            "bwt_per_seed": {a: per[a]["bwt"] for a in ARMS},
            "delta_bwt_full_vs_ablated": v_bwt,
            "delta_acc_full_vs_ablated": verdict(deltas_acc),
            "verdict_rule": (
                "the slot WINS iff per-seed BWT delta (full minus ablated) has mean > max(seed SD, "
                f"{MIN_MARGIN}) with no sign flip; else the preregistered null (routing does all the "
                "work) is supported (PILOT: registered claim rides the DR1-scale stream)"
            ),
            "null_supported": bool(not v_bwt["win"]),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        return out


def default_cfg(**overrides) -> DictConfig:
    base = {
        "seeds": [0, 1, 2, 3, 4],
        "cache": "vjepa2_vitl_fpc64_256_real",
        "data_dir": "data/cache",
        "n_tasks": 4,
        "train_frac": 0.75,
        "hidden": 32,
        "n_experts": 4,
        "epochs_per_task": 40,
        "lr": 1e-2,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WS5 shared-slot ablation pilot on real latents")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache", default="vjepa2_vitl_fpc64_256_real")
    ap.add_argument("--out", default="runs/mot/ws5_slot_ablation_pilot.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds), cache=a.cache)
    result = MotWS5SlotAblationPilot().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
