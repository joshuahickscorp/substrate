#!/usr/bin/env python
"""DR2 (=PR3): sparse-head forgetting PILOT on real cached latents (WP-10; registry row
mop_dr2_sparse_real in registry/experiments.yaml, preregistration mirror configs/experiment/mop_dr2.yaml).

Thesis under test: the e7 sparse-head forgetting advantage (kWTA, gated MoE) persists on REAL frozen
V-JEPA 2 pooled latents (data/cache/vjepa2_vitl_fpc64_256_real, 64 clips, 8 classes) instead of the
synthetic Gaussian-cluster proxy where it was found. PILOT status: 5 seeds on a 64-clip cache is
laptop-scale evidence only; the registered claim needs the DR1-scale stream and the 30-run paired
protocol (Studio). frozen_random controls are VALID here (trained-shell dynamics metrics, per the
manifest appendix), but none are needed: the arms differ only in head architecture.

Stream: real_task_stream class-incremental over the cache (the label space grows task by task while
the single shared head is trained sequentially, the genuine interference regime). Per-task per-class
train/test split by seed.

Arms (identical latents, identical epochs and lr, identical init where shapes coincide):
  dense        : dim -> hidden (GELU) -> classes MLP, the param-matched reference
  kwta         : KWTAHead, IDENTICAL shape to dense (activation sparsity is the only change)
  moe          : MoEHead with expert width solved by moe_expert_hidden_for_dense (the e7 match rule)
  dense_sparse : dense + L1 activation penalty, coefficient picked per seed from a small grid so its
                 active fraction lands nearest the kWTA target k/hidden (the matched-activation-
                 sparsity control the registry null names; the tuned-baseline doctrine)

Active fraction definition (fixed before running): mean fraction of hidden units per sample with
|h| > 0.05 * max|h| of that sample, measured on the final task's train latents after training.

Preregistered null (in code before any run): each sparse arm ties BOTH dense controls on backward
transfer. Verdict rule, uniform across the MoT scripts: a sparse arm WINS iff its per-seed BWT delta
over the BETTER (max) of the two dense controls has mean > max(seed SD, MIN_MARGIN) with no sign
flip. null_supported is True unless at least one sparse arm wins (two comparisons, preregistered,
both reported).

Usage: python scripts/mop_dr2_sparse_real_pilot.py --seeds 0-4
Writes runs/mot/dr2_sparse_real_pilot.json. No em or en dashes (BLACKHOLE.md).
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
from mop.diagnostics.compute import param_count
from mop.diagnostics.continual_metrics import backward_transfer, forgetting_area
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import parse_seeds, seed_everything
from mop.shell.heads import KWTAHead, MoEHead, moe_expert_hidden_for_dense
from mop.substrate.real_latent import real_task_stream

MIN_MARGIN = 0.02  # preregistered minimum meaningful BWT delta
SPARSE_ARMS = ("kwta", "moe")
CONTROL_ARMS = ("dense", "dense_sparse")


def verdict(deltas: list[float], min_margin: float = MIN_MARGIN) -> dict:
    """The uniform preregistered rule: WIN iff mean delta > max(sd, min_margin) and no sign flip."""
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    win = ci["mean"] > max(ci["sd"], min_margin) and not flips["any_flip"]
    return {"per_seed": [round(d, 4) for d in deltas], "ci": ci, "sign_flips": flips, "win": bool(win)}


def split_task(x: torch.Tensor, y: torch.Tensor, train_frac: float, g: torch.Generator):
    """Per-class split so every class appears in both halves (at least one test sample per class)."""
    tr, te = [], []
    for c in sorted(set(y.tolist())):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.shape[0], generator=g)]
        n_tr = min(max(1, int(round(train_frac * idx.shape[0]))), idx.shape[0] - 1)
        tr.append(idx[:n_tr])
        te.append(idx[n_tr:])
    return torch.cat(tr), torch.cat(te)


def load_split_stream(e: DictConfig, seed: int) -> list[dict]:
    """Real class-incremental stream sliced into per-task train/test splits (seeded)."""
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


def make_dense(dim: int, hidden: int, n_classes: int) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_classes))


def active_fraction(model: nn.Module, x: torch.Tensor) -> float:
    """Preregistered activation-sparsity statistic: mean fraction of units per sample with value
    above 5 percent of that sample's max. The unit population per arm: post-kWTA hidden layer
    (kwta), post-GELU hidden layer (dense arms, the quantity the tuned L1 control is matched on),
    router gates (moe, informational: its sparsity lives in the routing, not a hidden layer)."""
    with torch.no_grad():
        if isinstance(model, KWTAHead):
            h = F.gelu(model.fc1(x))
            thresh = h.topk(model.k, dim=-1).values[..., -1:]
            h = h * (h >= thresh)
        elif isinstance(model, MoEHead):
            model(x)
            h = model.last_gates
        else:
            h = F.gelu(model[0](x))
    mask = h.abs() > 0.05 * h.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    return float(mask.float().mean())


def train_stream(model: nn.Module, stream: list[dict], e: DictConfig, l1: float = 0.0) -> dict:
    """Sequential training over the class-incremental stream; returns the acc matrix and curves.
    acc_matrix[i][j] = accuracy on task j's test split after finishing training on task i."""
    opt = torch.optim.Adam(model.parameters(), lr=float(e.lr))
    n_tasks = len(stream)
    acc = [[0.0] * n_tasks for _ in range(n_tasks)]
    for i, task in enumerate(stream):
        for _ in range(int(e.epochs_per_task)):
            opt.zero_grad()
            if l1 > 0.0 and not isinstance(model, KWTAHead | MoEHead):
                h = F.gelu(model[0](task["xtr"]))
                logits = model[2](h)
                loss = F.cross_entropy(logits, task["ytr"]) + l1 * h.abs().mean()
            else:
                loss = F.cross_entropy(model(task["xtr"]), task["ytr"])
            loss.backward()
            opt.step()
        with torch.no_grad():
            for j, tj in enumerate(stream):
                acc[i][j] = float((model(tj["xte"]).argmax(-1) == tj["yte"]).float().mean())
    return {
        "acc_matrix": acc,
        "bwt": backward_transfer(acc),
        "task0_forgetting_area": forgetting_area([acc[i][0] for i in range(n_tasks)]),
        "final_mean_acc": sum(acc[-1][j] for j in range(n_tasks)) / n_tasks,
    }


class MotDR2SparseRealPilot(Experiment):
    id = "mop_dr2_sparse_real"
    metric = ("backward_transfer", "task0_forgetting_area", "final_mean_acc", "active_fraction")
    baseline = "param-matched dense head AND dense head with matched activation sparsity (tuned L1)"
    ablation = "head architecture only: kWTA / gated MoE vs the two dense controls, same latents"
    null_hypothesis = (
        "on real latents with a D3 certificate and a paired significance test, sparse ties "
        "param-matched dense (and ties a dense head with matched activation-sparsity penalty); the "
        "synthetic win was cluster separability"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        hidden = int(e.hidden)
        l1_grid = [float(v) for v in e.l1_grid]
        per: dict[str, dict[str, list[float]]] = {
            a: {"bwt": [], "final_mean_acc": [], "task0_forgetting_area": [], "active_fraction": []}
            for a in SPARSE_ARMS + CONTROL_ARMS
        }
        chosen_l1: list[float] = []
        params: dict[str, int] = {}
        for s in seeds:
            seed_everything(s)
            stream = load_split_stream(e, s)
            dim = stream[0]["xtr"].shape[1]
            n_classes = stream[0]["n_classes"]
            target_frac = min(1.0, int(e.kwta_k) / hidden)
            expert_hidden = moe_expert_hidden_for_dense(dim, hidden, n_classes, int(e.n_experts))

            def build(arm: str, l1: float = 0.0, s=s, dim=dim, n_classes=n_classes, eh=expert_hidden):
                torch.manual_seed(s)  # identical init across arms where shapes coincide
                if arm == "kwta":
                    return KWTAHead(dim, hidden, n_classes, k=int(e.kwta_k))
                if arm == "moe":
                    return MoEHead(dim, n_classes, int(e.n_experts), eh)
                return make_dense(dim, hidden, n_classes)

            # tuned matched-activation-sparsity control: pick the grid L1 whose active fraction on
            # the final task's train latents lands nearest the kWTA target k/hidden
            candidates = []
            for l1 in l1_grid:
                m = build("dense_sparse", l1)
                rep = train_stream(m, stream, e, l1=l1)
                frac = active_fraction(m, stream[-1]["xtr"])
                candidates.append((abs(frac - target_frac), l1, rep, frac))
            _, best_l1, best_rep, best_frac = min(candidates, key=lambda c: c[0])
            chosen_l1.append(best_l1)
            reports = {"dense_sparse": (best_rep, best_frac)}
            for arm in ("dense", "kwta", "moe"):
                m = build(arm)
                params[arm] = param_count(m)
                reports[arm] = (train_stream(m, stream, e), active_fraction(m, stream[-1]["xtr"]))
            params["dense_sparse"] = params["dense"]
            for arm, (rep, frac) in reports.items():
                for k in ("bwt", "final_mean_acc", "task0_forgetting_area"):
                    per[arm][k].append(rep[k])
                per[arm]["active_fraction"].append(frac)
        # paired per-seed deltas: each sparse arm vs the BETTER of the two dense controls, on BWT
        verdicts = {}
        for arm in SPARSE_ARMS:
            deltas = [
                per[arm]["bwt"][i] - max(per["dense"]["bwt"][i], per["dense_sparse"]["bwt"][i])
                for i in range(len(seeds))
            ]
            verdicts[arm] = verdict(deltas)
        any_win = any(v["win"] for v in verdicts.values())
        moe_ratio = params["moe"] / params["dense"]
        out = {
            "experiment": self.id,
            "pilot": True,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "param_counts": params,
            "moe_param_ratio_vs_dense": round(moe_ratio, 4),
            "param_match_ok": bool(params["kwta"] == params["dense"] and abs(moe_ratio - 1) <= 0.05),
            "arms": {a: {k: seed_ci(vs) for k, vs in per[a].items()} for a in per},
            "bwt_per_seed": {a: per[a]["bwt"] for a in per},
            "chosen_l1_per_seed": chosen_l1,
            "delta_bwt_vs_best_dense_control": verdicts,
            "verdict_rule": (
                "a sparse arm WINS iff its per-seed BWT delta over max(dense, dense_sparse) has mean "
                f"> max(seed SD, {MIN_MARGIN}) with no sign flip; null_supported unless any sparse arm "
                "wins (PILOT: the registered claim needs the DR1-scale stream and 30 paired runs)"
            ),
            "null_supported": bool(not any_win),
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
        "kwta_k": 4,
        "n_experts": 4,
        "l1_grid": [1e-4, 1e-3, 1e-2],
        "epochs_per_task": 40,
        "lr": 1e-2,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DR2/PR3 sparse-head forgetting pilot on real latents")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache", default="vjepa2_vitl_fpc64_256_real")
    ap.add_argument("--out", default="runs/mot/dr2_sparse_real_pilot.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds), cache=a.cache)
    result = MotDR2SparseRealPilot().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
