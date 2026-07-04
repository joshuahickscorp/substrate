#!/usr/bin/env python
"""CM4: workspace-shell routing PILOT on real cached latents (WP-10; registry row
mop_cm4_workspace_shell in registry/experiments.yaml, preregistration mirror
configs/experiment/mop_cm4.yaml).

Thesis under test: a trained WorkspaceShell (context gating + working memory + head, the WP-02
composition, no new science) beats BOTH a param-matched dense shell AND matched-FLOP unrolled depth
on continual-learning dynamics over real frozen V-JEPA 2 pooled latents. A workspace win that either
control matches is capacity or depth, not routing (the registry null).

Arms (identical latents, identical epochs and lr, identical task-id information):
  workspace : WorkspaceShell with ContextGating (context = task id, known at train AND eval in this
              pilot, stated honestly: the task-agnostic version is the Studio protocol) plus
              WorkingMemory slots plus a ClassHead
  dense     : plain dense MLP head, hidden width solved by capmatch.matched_capacity against the
              workspace shell's TOTAL trainable params (tol 2 percent, by construction)
  unrolled  : weight-tied IterativeRefiner + linear head; (steps, hidden) solved so total forward
              FLOPs match the workspace shell within tol (matched_within recorded); its params are
              deliberately fewer, that is what makes it the DEPTH control, not a capacity control

Information parity (control genuineness): BOTH controls receive the SAME ground-truth task id the
workspace's context gate sees, concatenated as a one-hot to their input at train AND eval. In
class-incremental continual learning an oracle task id is itself a large anti-forgetting mechanism
(task-conditioned gating is effectively a per-task subnetwork), so task-blind controls would let a
workspace BWT win be attributable to the extra oracle input rather than to routing. The capmatch and
FLOP targets are solved over the widened control input dim (proj_dim + n_tasks).

Preprocessing (not a control): one FIXED seeded random projection 1024 -> proj_dim applied
identically to every arm and every seed, purely to keep the pilot's trainable stacks small. It is
shared input preprocessing, never a comparison arm (the vacuous-control doctrine is untouched).
frozen_random controls are VALID here (trained-shell dynamics metrics) but unused.

planning_gain (a registry metric for this row) is NOT measurable on this cache: the pooled 528K
store has no temporal successors (forward_dynamics targets are identity), so any planning delta
would be vacuous. Reported as blocked-on-DR1 rather than faked; BWT and accuracy carry the pilot.

Preregistered null (in code before any run): the workspace shell ties param-matched dense OR ties
matched-compute unrolled depth on BWT, or sign-flips across seeds. Verdict rule, uniform across the
MoT scripts: workspace WINS iff the per-seed BWT delta over max(dense, unrolled) has mean >
max(seed SD, MIN_MARGIN) with no sign flip; else null_supported=True.

Usage: python scripts/mop_cm4_workspace_pilot.py --seeds 0-2
Writes runs/mot/cm4_workspace_pilot.json. No em or en dashes (BLACKHOLE.md).
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
from mop.diagnostics.compute import linear_flops, matched_within, mlp_flops, param_count
from mop.diagnostics.continual_metrics import backward_transfer, forgetting_area
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import parse_seeds, seed_everything
from mop.shell.capmatch import width_for_param_count
from mop.shell.heads import ClassHead
from mop.shell.modulation import ContextGating, WorkingMemory
from mop.shell.refine import IterativeRefiner
from mop.shell.workspace import WorkspaceShell
from mop.substrate.real_latent import real_task_stream

MIN_MARGIN = 0.02  # preregistered minimum meaningful BWT delta
ARMS = ("workspace", "dense", "unrolled")
FLOP_TOL = 0.10  # matched_within tolerance for the unrolled-depth arm
PROJ_SEED = 1234  # the fixed preprocessing projection, shared by every arm and every seed


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
    """Real class-incremental stream, projected to proj_dim by the FIXED preprocessing map, sliced
    into per-task train/test splits (seeded)."""
    tasks = real_task_stream(
        str(e.cache),
        n_tasks=int(e.n_tasks),
        incremental="class",
        seed=seed,
        data_dir=str(e.data_dir),
    )
    proj_dim = int(e.proj_dim)
    proj = None
    if proj_dim and proj_dim < tasks[0].x.shape[1]:
        g = torch.Generator().manual_seed(PROJ_SEED)
        proj = torch.randn(tasks[0].x.shape[1], proj_dim, generator=g) / (tasks[0].x.shape[1] ** 0.5)
    g = torch.Generator().manual_seed(seed + 1)
    out = []
    for t in tasks:
        x = t.x @ proj if proj is not None else t.x
        tr, te = split_task(x, t.y, float(e.train_frac), g)
        out.append(
            {
                "xtr": x[tr],
                "ytr": t.y[tr],
                "xte": x[te],
                "yte": t.y[te],
                "n_classes": t.n_classes,
                "task_id": t.task_id,
            }
        )
    return out


def build_workspace(dim: int, n_classes: int, n_tasks: int, e: DictConfig) -> WorkspaceShell:
    return WorkspaceShell(
        dim,
        head=ClassHead(dim, n_classes, hidden=int(e.ws_head_hidden)),
        modulation={
            "context_gating": ContextGating(dim, n_tasks),
            "working_memory": WorkingMemory(dim, slots=int(e.wm_slots)),
        },
    )


def workspace_flops(dim: int, n_classes: int, e: DictConfig) -> int:
    """Forward FLOP estimate for the workspace shell: WM write+gate linears plus the head MLP.
    ContextGating is an embedding lookup plus an elementwise product, rounding error by the
    compute-module convention (linear layers are the load-bearing term)."""
    wm = linear_flops(dim, dim) + linear_flops(dim, int(e.wm_slots))
    head = mlp_flops([dim, int(e.ws_head_hidden), n_classes])
    return int(wm + head)


class UnrolledDepthNet(nn.Module):
    """Weight-tied refiner iterated `steps` times plus a linear readout: the matched-FLOP DEPTH
    control (params intentionally below the workspace shell; depth is what it buys)."""

    def __init__(self, dim: int, n_classes: int, hidden: int, steps: int):
        super().__init__()
        self.refiner = IterativeRefiner(dim, hidden=hidden, steps=steps)
        self.readout = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.refiner(x)
        return self.readout(z)


def solve_unrolled(dim: int, n_classes: int, target_flops: int, max_steps: int = 8) -> tuple[int, int]:
    """Pick (steps, hidden) whose total forward FLOPs (steps residual blocks of 4*dim*hidden plus the
    readout) land nearest target_flops, steps >= 2 so the arm is genuinely iterative."""
    head = linear_flops(dim, n_classes)
    best: tuple[float, int, int] | None = None
    for steps in range(2, max_steps + 1):
        hidden = max(1, round((target_flops - head) / (steps * 4 * dim)))
        total = steps * mlp_flops([dim, hidden, dim]) + head
        miss = abs(total - target_flops) / target_flops
        if best is None or miss < best[0]:
            best = (miss, steps, hidden)
    assert best is not None
    return best[1], best[2]


def train_arm(model: nn.Module, stream: list[dict], e: DictConfig, contextual: bool, n_tasks: int) -> dict:
    """Sequential training; acc_matrix[i][j] = acc on task j's test split after training task i.
    contextual=True feeds the task id to the workspace shell's context gate; contextual=False
    concatenates the SAME ground-truth task id as a one-hot to the control's input (information
    parity: every arm sees the task id at train and eval, the known-task-id pilot protocol)."""

    def logits(m: nn.Module, x: torch.Tensor, task_id: int) -> torch.Tensor:
        ctx = torch.full((x.shape[0],), task_id, dtype=torch.long)
        if contextual:
            return m(x, context=ctx)["head"]
        return m(torch.cat([x, F.one_hot(ctx, n_tasks).float()], dim=1))

    opt = torch.optim.Adam(model.parameters(), lr=float(e.lr))
    n_tasks = len(stream)
    acc = [[0.0] * n_tasks for _ in range(n_tasks)]
    for i, task in enumerate(stream):
        for _ in range(int(e.epochs_per_task)):
            opt.zero_grad()
            F.cross_entropy(logits(model, task["xtr"], task["task_id"]), task["ytr"]).backward()
            opt.step()
        with torch.no_grad():
            for j, tj in enumerate(stream):
                pred = logits(model, tj["xte"], tj["task_id"]).argmax(-1)
                acc[i][j] = float((pred == tj["yte"]).float().mean())
    return {
        "acc_matrix": acc,
        "bwt": backward_transfer(acc),
        "task0_forgetting_area": forgetting_area([acc[i][0] for i in range(n_tasks)]),
        "final_mean_acc": sum(acc[-1][j] for j in range(n_tasks)) / n_tasks,
    }


class MotCM4WorkspacePilot(Experiment):
    id = "mop_cm4_workspace_shell"
    metric = ("backward_transfer", "final_mean_acc", "seed_sign_flips", "planning_gain")
    baseline = "param-matched dense head (capmatch) AND matched-FLOP weight-tied unrolled depth"
    ablation = "workspace composition (context gating + working memory) vs capacity vs depth"
    null_hypothesis = (
        "the workspace shell ties param-matched dense OR ties matched-compute unrolled depth on BWT "
        "and planning, or sign-flips across seeds: the win is capacity/depth, not routing"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        per: dict[str, dict[str, list[float]]] = {
            a: {"bwt": [], "final_mean_acc": [], "task0_forgetting_area": []} for a in ARMS
        }
        match_record: dict = {}
        for s in seeds:
            seed_everything(s)
            stream = load_split_stream(e, s)
            dim = stream[0]["xtr"].shape[1]
            n_classes = stream[0]["n_classes"]
            n_tasks = len(stream)

            torch.manual_seed(s)
            ws = build_workspace(dim, n_classes, n_tasks, e)
            ws_params = param_count(ws)
            ws_flops = workspace_flops(dim, n_classes, e)
            # information parity: both controls take the ground-truth task id as a one-hot
            # concatenated to the latent input, so capmatch/FLOP targets solve over the widened dim
            ctrl_dim = dim + n_tasks

            def make_dense(w: int, dim_in=ctrl_dim, n_classes=n_classes) -> nn.Module:
                return nn.Sequential(nn.Linear(dim_in, w), nn.GELU(), nn.Linear(w, n_classes))

            w, achieved = width_for_param_count(make_dense, ws_params, tol=float(e.capmatch_tol))
            torch.manual_seed(s)
            dense = make_dense(w)

            steps, hidden = solve_unrolled(ctrl_dim, n_classes, ws_flops)
            torch.manual_seed(s)
            unrolled = UnrolledDepthNet(ctrl_dim, n_classes, hidden, steps)
            unrolled_flops = steps * mlp_flops([ctrl_dim, hidden, ctrl_dim]) + linear_flops(
                ctrl_dim, n_classes
            )

            match_record = {
                "workspace_params": ws_params,
                "dense_params": achieved,
                "dense_width": w,
                "unrolled_params": param_count(unrolled),
                "unrolled_steps": steps,
                "unrolled_hidden": hidden,
                "control_input_dim": ctrl_dim,
                "flops_matched_within": matched_within(ws_flops, unrolled_flops, tol=FLOP_TOL),
                "note": (
                    f"dense matches params (capmatch tol {float(e.capmatch_tol)}); unrolled matches "
                    "FLOPs, not params; both controls receive the same ground-truth task id as a "
                    "one-hot input (information parity with the workspace context gate)"
                ),
            }
            for arm, model in (("workspace", ws), ("dense", dense), ("unrolled", unrolled)):
                rep = train_arm(model, stream, e, contextual=(arm == "workspace"), n_tasks=n_tasks)
                for k in per[arm]:
                    per[arm][k].append(rep[k])
        deltas = [
            per["workspace"]["bwt"][i] - max(per["dense"]["bwt"][i], per["unrolled"]["bwt"][i])
            for i in range(len(seeds))
        ]
        v = verdict(deltas)
        out = {
            "experiment": self.id,
            "pilot": True,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "capacity_and_compute_matching": match_record,
            "arms": {a: {k: seed_ci(vs) for k, vs in per[a].items()} for a in ARMS},
            "bwt_per_seed": {a: per[a]["bwt"] for a in ARMS},
            "delta_bwt_ws_vs_best_control": v,
            "seed_sign_flips": v["sign_flips"],
            "planning_gain": None,
            "planning_gain_blocked_reason": (
                "the pooled 528K cache has no temporal successors (identity forward_dynamics "
                "targets), so a planning delta would be vacuous; planning waits on the DR1 cache"
            ),
            "verdict_rule": (
                "workspace WINS iff per-seed BWT delta over max(dense, unrolled) has mean > "
                f"max(seed SD, {MIN_MARGIN}) with no sign flip; else the preregistered null "
                "(capacity/depth, not routing) is supported (PILOT: registered claim is Studio-scale)"
            ),
            "null_supported": bool(not v["win"]),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        return out


def default_cfg(**overrides) -> DictConfig:
    base = {
        "seeds": [0, 1, 2],
        "cache": "vjepa2_vitl_fpc64_256_real",
        "data_dir": "data/cache",
        "n_tasks": 4,
        "train_frac": 0.75,
        "proj_dim": 64,
        "ws_head_hidden": 32,
        "wm_slots": 4,
        "capmatch_tol": 0.02,  # dense-arm param match; the doctrine 2 percent, integer widths permitting
        "epochs_per_task": 40,
        "lr": 1e-2,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CM4 workspace-shell pilot on real latents")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--cache", default="vjepa2_vitl_fpc64_256_real")
    ap.add_argument("--out", default="runs/mot/cm4_workspace_pilot.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds), cache=a.cache)
    result = MotCM4WorkspacePilot().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
