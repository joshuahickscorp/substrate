
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..diagnostics.geometry import anisotropy, effective_rank, linear_cka
from ..diagnostics.linear_probe import linear_probe
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from .base import Experiment


def _probe(x, y, nonlinear: bool, seed: int) -> dict:
    if nonlinear:
        g = torch.Generator().manual_seed(seed + 7)
        w = torch.randn(x.shape[1], x.shape[1], generator=g) / (x.shape[1] ** 0.5)
        x = torch.tanh(x @ w)
    return linear_probe(x, y, seed=seed)


class EX12(Experiment):
    id = "ex12_atlas"
    metric = ("probe_accuracy_above_chance", "effective_rank", "anisotropy", "cka_vs_random")
    baseline = "shuffle-label chance floor per factor; a frozen-random projection for the geometry"
    ablation = "linear vs nonlinear probe per factor; identity (decodable) vs random-label (not) factors"
    null_hypothesis = (
        "probe accuracy does not exceed the shuffle-label floor for a target (the factor is not in the "
        "latent), and bigger frozen perception does not raise decodability beyond the seed spread"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        rows: dict[str, dict] = {}
        for factor in ("identity", "random_label"):
            for probe_kind in ("linear", "nonlinear"):
                accs, floors = [], []
                for s in seeds:
                    seed_everything(s)
                    task = make_task_stream(
                        n_tasks=1,
                        dim=dim,
                        classes_per_task=int(e.n_classes),
                        samples_per_task=int(e.samples),
                        separation=float(e.separation),
                        seed=s,
                    )[0]
                    x = task.x
                    if factor == "identity":
                        y = task.y
                    else:
                        g = torch.Generator().manual_seed(s + 13)
                        y = torch.randint(0, int(e.n_classes), (x.shape[0],), generator=g)
                    accs.append(_probe(x, y, probe_kind == "nonlinear", s)["score"])
                    g2 = torch.Generator().manual_seed(s + 99)
                    floors.append(
                        _probe(x, y[torch.randperm(y.shape[0], generator=g2)], probe_kind == "nonlinear", s)[
                            "score"
                        ]
                    )
                am = sum(accs) / len(accs)
                fm = sum(floors) / len(floors)
                rows[f"{factor}:{probe_kind}"] = {
                    "acc_mean": round(am, 4),
                    "shuffle_floor": round(fm, 4),
                    "above_chance": bool(am > fm + 0.1),
                    "factor": factor,
                    "probe": probe_kind,
                }

        seed_everything(seeds[0])
        task = make_task_stream(
            n_tasks=1,
            dim=dim,
            classes_per_task=int(e.n_classes),
            samples_per_task=int(e.samples),
            separation=float(e.separation),
            seed=seeds[0],
        )[0]
        x = task.x
        geometry = {
            "effective_rank": round(effective_rank(x), 3),
            "anisotropy": round(anisotropy(x), 3),
            "cka_vs_random_projection": round(linear_cka(x, frozen_random_projection(x, seeds[0])), 3),
        }
        out = {
            "rows": rows,
            "geometry": geometry,
            "seeds": list(seeds),
            "identity_decodable": rows["identity:linear"]["above_chance"],
            "random_label_not_decodable": not rows["random_label:linear"]["above_chance"],
            "atlas_self_check_passed": bool(
                rows["identity:linear"]["above_chance"] and not rows["random_label:linear"]["above_chance"]
            ),
            "null_supported": not rows["random_label:linear"]["above_chance"],
        }
        return out
