#!/usr/bin/env python
"""PR8: memory-augmented retrieval head (WP-06, H-MEMORY; registry
docs/mixture_of_thinking/11_experiment_registry.md, PR full schema part 2).

Thesis: a retrieval-conditioned head over the frozen-latent KV buffer beats BOTH plain kNN AND a
matched-param parametric head on new-class few-shot, exploiting that frozen-encoder keys never go
stale (the doctrinal free lunch). The head must add value over its own two halves: if it ties kNN
the parametric part adds nothing; if it ties the parametric head the memory adds nothing.

Task (synthetic latents, no encoder): the DR10 universe, fine-grained local sub-clusters in a
signal subspace plus high-variance nuisance dims. New classes exist only as k-shot buffer entries
(never as retrieval-head training queries), so the retrieval head must GENERALIZE its learned
attention from base classes to unseen ones, which is exactly the stale-free-keys claim. The
parametric head, whose only channel is its weights, trains on the same information (base data plus
the k-shot supports as ordinary training points): information access is matched per arm, only the
storage substrate (weights vs buffer) differs.

Arms:
  retrieval_head : learned query/key attention over the k retrieved neighbors' labels, mixed with a
                   small parametric branch (trained on base queries only)
  knn            : distance-weighted vote over the SAME retrieved neighbors (no trainable part)
  parametric     : an MLP head with params matched to retrieval_head via shell/capmatch (trained on
                   base data PLUS the new-class shots, its only channel to them)
  random_key     : the retrieval head trained and evaluated through a PERMUTED-key buffer (the
                   registry's random-key retrieval control)

Preregistered null (fixed before running): retrieval_head ties kNN OR ties the parametric head at
matched capacity. Verdict rule, uniform across the MoT memory scripts: WIN iff BOTH per-seed delta
means (head minus kNN, head minus parametric, on new-class accuracy) exceed max(per-seed SD,
MIN_MARGIN) with no sign flip; else null_supported=True.

Usage: python scripts/mot_pr8_retrieval_head.py --seeds 0-4
Writes runs/mot/pr8_retrieval_head.json. No em or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
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
from devsys.shell.buffer import ReplayBuffer
from devsys.shell.capmatch import width_for_param_count
from devsys.shell.predictor import mlp

MIN_MARGIN = 0.02  # preregistered minimum meaningful accuracy delta
ARMS = ("retrieval_head", "knn", "parametric", "random_key")


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
    g = torch.Generator().manual_seed(seed)
    n_classes = int(e.n_base) + int(e.n_new)
    centers = torch.randn(n_classes, int(e.m_sub), int(e.d_signal), generator=g) * float(e.separation)
    return {"centers": centers, "g": g, "n_classes": n_classes}


def sample_classes(uni: dict, e: DictConfig, classes: list[int], per_class: int) -> tuple:
    g, centers = uni["g"], uni["centers"]
    xs, ys = [], []
    for c in classes:
        which = torch.randint(0, int(e.m_sub), (per_class,), generator=g)
        sig = centers[c, which] + 0.5 * torch.randn(per_class, int(e.d_signal), generator=g)
        nui = float(e.nuisance_scale) * torch.randn(per_class, int(e.d_nuisance), generator=g)
        xs.append(torch.cat([sig, nui], dim=-1))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def build_buffer(x: torch.Tensor, y: torch.Tensor, seed: int, permute_keys: bool = False) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=x.shape[0], dim=x.shape[1], prioritized=False, eviction="fifo", seed=seed)
    if permute_keys:
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed + 13))
        buf.add(x, y, key=x[perm])
    else:
        buf.add(x, y)
    return buf


class RetrievalHead(nn.Module):
    """The memory-augmented head: buffer.retrieve fetches k neighbors through the FROZEN raw-latent
    keys; a learned query/key projection then re-weights them (it can learn to project out the
    nuisance dims raw L2 cannot), the label votes are attention-combined, and a small parametric
    branch handles what memory does not carry. logits = vote_scale * attn_votes + mlp(z)."""

    def __init__(self, dim: int, n_classes: int, attn_dim: int, hidden: int):
        super().__init__()
        self.n_classes = n_classes
        self.attn_dim = attn_dim
        self.q = nn.Linear(dim, attn_dim)
        self.kproj = nn.Linear(dim, attn_dim)
        self.vote_scale = nn.Parameter(torch.tensor(1.0))
        self.par = mlp(dim, n_classes, hidden, depth=1, ln=True)

    def forward(self, z: torch.Tensor, neigh_x: torch.Tensor, neigh_y: torch.Tensor) -> torch.Tensor:
        """z [B,D]; neigh_x [B,k,D]; neigh_y [B,k]."""
        scores = (self.kproj(neigh_x) @ self.q(z).unsqueeze(-1)).squeeze(-1) / math.sqrt(self.attn_dim)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B,k,1]
        votes = (attn * F.one_hot(neigh_y, self.n_classes).float()).sum(1)
        return self.vote_scale * votes * float(self.n_classes) + self.par(z)


def knn_logits(z: torch.Tensor, neigh_x: torch.Tensor, neigh_y: torch.Tensor, n_classes: int):
    """Plain kNN over the same retrieved set: distance-weighted label vote (no trainable part)."""
    dist = (z.unsqueeze(1) - neigh_x).norm(dim=-1)
    w = torch.softmax(-dist, dim=-1).unsqueeze(-1)
    return (w * F.one_hot(neigh_y, n_classes).float()).sum(1)


def _train(model: nn.Module, forward, xtr, ytr, e: DictConfig, seed: int) -> None:
    g = torch.Generator().manual_seed(seed + 31)
    opt = torch.optim.Adam(model.parameters(), lr=float(e.lr))
    n, batch = xtr.shape[0], int(e.batch)
    for _ in range(int(e.epochs)):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            F.cross_entropy(forward(xtr[idx]), ytr[idx]).backward()
            opt.step()


def _head_forward(net: RetrievalHead, source: ReplayBuffer, k: int, x: torch.Tensor) -> torch.Tensor:
    ret = source.retrieve(x, k=k)
    return net(x, ret["x"], ret["y"])


def _make_parametric(dim: int, n_classes: int, width: int) -> nn.Module:
    return mlp(dim, n_classes, hidden=width, depth=1, ln=True)


@torch.no_grad()
def _arm_acc(arm: str, x: torch.Tensor, y: torch.Tensor, models: dict, k: int, n_classes: int) -> float:
    if arm == "retrieval_head":
        logits = _head_forward(models["head"], models["buf"], k, x)
    elif arm == "knn":
        ret = models["buf"].retrieve(x, k=k)
        logits = knn_logits(x, ret["x"], ret["y"], n_classes)
    elif arm == "parametric":
        logits = models["parametric"](x)
    else:
        logits = _head_forward(models["rk_head"], models["rand_buf"], k, x)
    return float((logits.argmax(-1) == y).float().mean())


def _run_seed(e: DictConfig, s: int, dim: int, base_classes: list[int], new_classes: list[int]) -> dict:
    """One seed: build the universe and buffers, train the three trainable arms, score all four,
    and trace the retrieval-head accuracy-vs-shots curve. Factored out of the seed loop so no
    closure binds a loop variable."""
    seed_everything(s)
    uni = make_universe(s, e)
    n_classes = uni["n_classes"]
    mem_x, mem_y = sample_classes(uni, e, base_classes, int(e.mem_per_class))
    shot_x, shot_y = sample_classes(uni, e, new_classes, int(e.k_shot))
    buf_x, buf_y = torch.cat([mem_x, shot_x]), torch.cat([mem_y, shot_y])
    buf = build_buffer(buf_x, buf_y, seed=s)
    rand_buf = build_buffer(buf_x, buf_y, seed=s, permute_keys=True)
    xtr, ytr = sample_classes(uni, e, base_classes, int(e.train_per_class))
    xte_b, yte_b = sample_classes(uni, e, base_classes, int(e.test_per_class))
    xte_n, yte_n = sample_classes(uni, e, new_classes, int(e.test_per_class))
    k = int(e.k)
    torch.manual_seed(s)
    head = RetrievalHead(dim, n_classes, attn_dim=int(e.attn_dim), hidden=int(e.hidden))
    _train(head, partial(_head_forward, head, buf, k), xtr, ytr, e, seed=s)
    # matched-param parametric head via the capmatch doctrine helper; its ONLY channel to the new
    # classes is training on the k-shot supports as ordinary labeled points
    target = sum(p.numel() for p in head.parameters())
    width, achieved = width_for_param_count(
        partial(_make_parametric, dim, n_classes), target, tol=float(e.capmatch_tol)
    )
    torch.manual_seed(s)
    parametric = _make_parametric(dim, n_classes, width)
    _train(parametric, parametric, torch.cat([xtr, shot_x]), torch.cat([ytr, shot_y]), e, seed=s)
    torch.manual_seed(s)
    rk_head = RetrievalHead(dim, n_classes, attn_dim=int(e.attn_dim), hidden=int(e.hidden))
    _train(rk_head, partial(_head_forward, rk_head, rand_buf, k), xtr, ytr, e, seed=s)
    models = {"head": head, "parametric": parametric, "rk_head": rk_head, "buf": buf, "rand_buf": rand_buf}
    accs = {}
    for arm in ARMS:
        acc_b = _arm_acc(arm, xte_b, yte_b, models, k, n_classes)
        acc_n = _arm_acc(arm, xte_n, yte_n, models, k, n_classes)
        n_all = xte_b.shape[0] + xte_n.shape[0]
        overall = (acc_b * xte_b.shape[0] + acc_n * xte_n.shape[0]) / n_all
        accs[arm] = {"base": acc_b, "new": acc_n, "overall": overall}
    # adaptation vs shots: retrieval-head new-class accuracy as the buffer grows 1..k_shot
    curve = []
    for shots in range(1, int(e.k_shot) + 1):
        keep = torch.cat([torch.arange(len(new_classes)) * int(e.k_shot) + j for j in range(shots)]).sort()[0]
        sbuf = build_buffer(torch.cat([mem_x, shot_x[keep]]), torch.cat([mem_y, shot_y[keep]]), seed=s)
        with torch.no_grad():
            ret = sbuf.retrieve(xte_n, k=k)
            curve.append(float((head(xte_n, ret["x"], ret["y"]).argmax(-1) == yte_n).float().mean()))
    param_counts = {
        "retrieval_head": target,
        "parametric": achieved,
        "parametric_width": width,
        "match_rel_err": round(abs(achieved - target) / target, 4),
    }
    return {"accs": accs, "curve": curve, "param_counts": param_counts}


class MotPR8RetrievalHead(Experiment):
    id = "mot_pr8_retrieval_head"
    metric = ("new_class_acc", "overall_acc", "adaptation_speed_shots", "param_counts")
    baseline = "plain kNN over the same buffer AND a matched-param parametric head (both must be beaten)"
    ablation = "learned retrieval attention vs its nonparametric and parametric halves; random-key control"
    null_hypothesis = (
        "the retrieval-conditioned head ties plain kNN (parametric part adds nothing) OR ties the "
        "matched-param parametric head (memory adds nothing)"
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
        param_counts: dict[str, int] = {}
        shots_curves: list[list[float]] = []
        for s in seeds:
            res = _run_seed(e, s, dim, base_classes, new_classes)
            for arm in ARMS:
                for key in ("new", "base", "overall"):
                    per[arm][key].append(res["accs"][arm][key])
            shots_curves.append(res["curve"])
            param_counts = res["param_counts"]
        d_knn = [per["retrieval_head"]["new"][i] - per["knn"]["new"][i] for i in range(len(seeds))]
        d_par = [per["retrieval_head"]["new"][i] - per["parametric"]["new"][i] for i in range(len(seeds))]
        v_knn, v_par = verdict(d_knn), verdict(d_par)
        win = bool(v_knn["win"] and v_par["win"])
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
            "param_counts": param_counts,
            "delta_vs_knn": v_knn,
            "delta_vs_parametric": v_par,
            "adaptation_shots_curve_mean": [round(x, 4) for x in mean_curve],
            "adaptation_speed_shots": adaptation_speed(mean_curve),
            "verdict_rule": (
                "WIN iff BOTH per-seed delta means (head minus kNN, head minus matched-param "
                f"parametric, new-class accuracy) > max(seed SD, {MIN_MARGIN}) with no sign flip; "
                "else the preregistered null (one half is redundant) is supported"
            ),
            "null_supported": bool(not win),
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
        "attn_dim": 32,
        "hidden": 32,
        "capmatch_tol": 0.02,
        "epochs": 20,
        "batch": 64,
        "lr": 1e-3,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR8: retrieval head vs plain kNN and matched-param parametric")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="runs/mot/pr8_retrieval_head.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds), epochs=a.epochs)
    result = MotPR8RetrievalHead().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
