"""EX12: the representational atlas. Systematizes the linear-probe spine into a decodability matrix:
for each target factor, linear AND small-nonlinear probe accuracy above the shuffle-label chance floor,
plus the geometry battery (effective rank, anisotropy, CKA vs a frozen-random projection). This is the
empirical floor under every other experiment: a mechanism cannot use information the atlas shows is not
there. Run FIRST per encoder. On the laptop it runs on a synthetic stream with a KNOWN decodable factor
(identity) and a KNOWN non-decodable factor (a random label), so the gate is self-checking.

NULL: probe accuracy does not exceed the shuffle-label floor for a target (the factor is not in the
latent, taxonomy slot 3); and bigger frozen perception does not raise decodability beyond the seed
spread (the cross-encoder null, exercised at Studio scale). cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

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
    # nonlinear probe == a small MLP; reuse linear_probe by lifting x through a fixed random feature
    # map (a cheap nonlinearity), so a factor decodable only nonlinearly shows the gap.
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
        # factors: identity (class label, must be decodable) + a random-label control (must NOT be)
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
                    # shuffle-label floor: same probe on permuted labels
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

        # geometry of the latent vs a frozen-random projection of itself
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
            # self-checking gate: identity decodable above floor, random-label NOT (the atlas is honest)
            "identity_decodable": rows["identity:linear"]["above_chance"],
            "random_label_not_decodable": not rows["random_label:linear"]["above_chance"],
            "atlas_self_check_passed": bool(
                rows["identity:linear"]["above_chance"] and not rows["random_label:linear"]["above_chance"]
            ),
            "null_supported": not rows["random_label:linear"]["above_chance"],
        }
        return out
