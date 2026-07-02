#!/usr/bin/env python
"""PR2 (WP-11, registry #7, the decisive learning-eases fork test): does the pretrained substrate ease
the plastic shell's LEARNING dynamics, or only its readout? An IDENTICAL plastic shell (same
architecture, same init seed, same LR, same steps) trains on a class-incremental stream twice, once on
real V-JEPA pooled features and once on random-init same-arch ViT-L features at matched 256px (the
non-vacuous control from cache_randominit_vitl_features.py). Scored on trained-shell dynamics only:
backward transfer (GEM BWT), adaptation speed (steps to 90 percent of final within-task accuracy), and
forgetting area, per mop.diagnostics.continual_metrics.

PREREGISTERED NULL (verbatim, registry PR2): shell adaptation speed and BWT on real V-JEPA are within
the seed spread of the same shell on random-init-ViT (matched resolution): the substrate's structure
does not ease trained-shell learning dynamics.

PREREGISTERED VERDICT RULE (fixed before any result exists): the null is REJECTED only if the BWT delta
(real minus randominit) or the adaptation-speed delta (randominit steps minus real steps) has a 5-seed
95 percent CI excluding zero from below AND a consistent per-seed sign (no flips). Forgetting-area delta
is reported as secondary, never verdict-driving. Anything else supports the null (the +0.31 was
readout-only).

Consumes the WP-11 caches (data/cache/vjepa2_vitl_nuisance, data/cache/randominit_vitl_nuisance);
never loads an encoder. Heavy queue class: run only when the encoder lane is free.

Usage: python scripts/mop_pr2_plasticity_substrates.py --seeds 0-4

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cache_randominit_vitl_features import load_feature_cache  # noqa: E402

from mop.diagnostics.continual_metrics import (  # noqa: E402
    adaptation_speed,
    backward_transfer,
    forgetting_area,
)
from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402
from mop.experiments.base import Experiment  # noqa: E402

DEFAULTS = {
    "real_cache": "data/cache/vjepa2_vitl_nuisance",
    "randinit_cache": "data/cache/randominit_vitl_nuisance",
    "seeds": (0, 1, 2, 3, 4),
    "steps_per_task": 150,
    "hidden": 32,
    "lr": 1e-2,
    "test_frac": 0.25,
    "adapt_target_frac": 0.9,
}


def _get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _zscore(train: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True) + 1e-6
    return (x - mu) / sd


def _class_split(y: torch.Tensor, test_frac: float, g: torch.Generator):
    """Per-class seeded train/test index split (stratified so every task has test samples)."""
    tr, te = [], []
    for c in y.unique().tolist():
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        cut = max(1, int(len(idx) * test_frac))
        te.append(idx[:cut])
        tr.append(idx[cut:])
    return torch.cat(tr), torch.cat(te)


def _tasks_from_classes(n_classes: int) -> list[list[int]]:
    """Consecutive class pairs (last task may be a singleton): the class-incremental stream."""
    return [list(range(i, min(i + 2, n_classes))) for i in range(0, n_classes, 2)]


def continual_stream_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    steps_per_task: int,
    hidden: int,
    lr: float,
    test_frac: float,
    adapt_target_frac: float,
) -> dict:
    """Train one plastic shell (Linear-ReLU-Linear head over frozen features) through the
    class-incremental stream; return BWT, mean adaptation steps, and task-0 forgetting area. The shell
    init, split, and task order are seeded identically across substrate arms (matched shell)."""
    n_classes = int(y.max()) + 1
    tasks = _tasks_from_classes(n_classes)
    g = torch.Generator().manual_seed(seed)
    tr_idx, te_idx = _class_split(y, test_frac, g)
    xz = _zscore(x[tr_idx], x)
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    def task_indices(pool: torch.Tensor, classes: list[int]) -> torch.Tensor:
        m = torch.zeros(len(pool), dtype=torch.bool)
        for c in classes:
            m |= y[pool] == c
        return pool[m]

    @torch.no_grad()
    def acc_on(idx: torch.Tensor) -> float:
        return float((head(xz[idx]).argmax(-1) == y[idx]).float().mean())

    acc_matrix: list[list[float]] = []
    task0_curve: list[float] = []
    adapt_steps: list[int] = []
    te_task = [task_indices(te_idx, t) for t in tasks]
    for ti, classes in enumerate(tasks):
        tr = task_indices(tr_idx, classes)
        curve: list[float] = []
        for _ in range(steps_per_task):
            opt.zero_grad()
            F.cross_entropy(head(xz[tr]), y[tr]).backward()
            opt.step()
            curve.append(acc_on(te_task[ti]))
            task0_curve.append(acc_on(te_task[0]))
        adapt_steps.append(int(adaptation_speed(curve, target_frac=adapt_target_frac)["steps"]))
        acc_matrix.append([acc_on(te) for te in te_task])
    return {
        "bwt": backward_transfer(acc_matrix),
        "adapt_steps_mean": sum(adapt_steps) / len(adapt_steps),
        "forgetting_area_task0": forgetting_area(task0_curve),
        "final_acc_mean": sum(acc_matrix[-1]) / len(acc_matrix[-1]),
        "acc_matrix": [[round(a, 4) for a in row] for row in acc_matrix],
    }


def run(cfg=None, device=None, run_dir: Path | None = None) -> dict:
    cache_real = load_feature_cache(_get(cfg, "real_cache", DEFAULTS["real_cache"]))
    cache_rand = load_feature_cache(_get(cfg, "randinit_cache", DEFAULTS["randinit_cache"]))
    y = cache_real["labels_shape"]
    if cache_rand["labels_shape"] is None or not torch.equal(y, cache_rand["labels_shape"]):
        raise ValueError("real and random-init caches disagree on labels: clip identity rule violated")
    kw = {
        k: _get(cfg, k, DEFAULTS[k])
        for k in ("steps_per_task", "hidden", "lr", "test_frac", "adapt_target_frac")
    }
    seeds = list(_get(cfg, "seeds", DEFAULTS["seeds"]))
    t0 = time.perf_counter()
    per_seed = []
    for s in seeds:
        real = continual_stream_metrics(cache_real["features"], y, seed=s, **kw)
        rand = continual_stream_metrics(cache_rand["features"], y, seed=s, **kw)
        per_seed.append(
            {
                "seed": s,
                "real": {k: v for k, v in real.items() if k != "acc_matrix"},
                "randinit": {k: v for k, v in rand.items() if k != "acc_matrix"},
                "bwt_delta": real["bwt"] - rand["bwt"],
                "speed_delta_steps": rand["adapt_steps_mean"] - real["adapt_steps_mean"],
                "fa_delta": rand["forgetting_area_task0"] - real["forgetting_area_task0"],
            }
        )
    bwt_ci = seed_ci([r["bwt_delta"] for r in per_seed])
    speed_ci = seed_ci([r["speed_delta_steps"] for r in per_seed])
    fa_ci = seed_ci([r["fa_delta"] for r in per_seed])
    bwt_flips = sign_flip_report([r["bwt_delta"] for r in per_seed])
    speed_flips = sign_flip_report([r["speed_delta_steps"] for r in per_seed])
    bwt_win = bwt_ci["lo"] > 0 and bwt_flips["consistent_sign"] == 1
    speed_win = speed_ci["lo"] > 0 and speed_flips["consistent_sign"] == 1
    null_supported = not (bwt_win or speed_win)
    out = {
        "experiment": PR2Experiment.id,
        "contract": PR2Experiment().contract(),
        "seeds": seeds,
        "per_seed": per_seed,
        "bwt_delta_ci": bwt_ci,
        "speed_delta_ci": speed_ci,
        "fa_delta_ci_secondary": fa_ci,
        "bwt_sign_flips": bwt_flips,
        "speed_sign_flips": speed_flips,
        "bwt_win": bwt_win,
        "speed_win": speed_win,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": (
            "NULL SUPPORTED: real V-JEPA learning dynamics within seed spread of random-init-ViT, the "
            "+0.31 substrate structure is readout-only here"
            if null_supported
            else "NULL REJECTED: the pretrained substrate eases trained-shell learning dynamics (BWT "
            f"win={bwt_win}, adaptation-speed win={speed_win}), the strongest keep-frozen signal"
        ),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


class PR2Experiment(Experiment):
    id = "pr2_plasticity_substrates"
    metric = ("bwt_delta", "adaptation_speed_delta", "forgetting_area_delta")
    baseline = "identical plastic shell on random-init same-arch ViT-L features at matched 256px"
    ablation = "substrate swap: pretrained V-JEPA features vs untrained same-arch features"
    null_hypothesis = (
        "Shell adaptation speed and BWT on real V-JEPA are within the seed spread of the same shell on "
        "random-init-ViT (matched resolution): the substrate's structure does not ease trained-shell "
        "learning dynamics"
    )
    tier = "cpu-now"

    def run(self, cfg, device, run_dir: Path) -> dict:
        return run(cfg, device, run_dir)


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR2 plasticity: real vs random-init-ViT substrates")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--real-cache", default=DEFAULTS["real_cache"])
    ap.add_argument("--randinit-cache", default=DEFAULTS["randinit_cache"])
    ap.add_argument("--steps-per-task", type=int, default=DEFAULTS["steps_per_task"])
    ap.add_argument("--rerun", action="store_true", help="stage-4 rerun naming (_seeds10 suffix)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {
        "seeds": _parse_seeds(a.seeds),
        "real_cache": a.real_cache,
        "randinit_cache": a.randinit_cache,
        "steps_per_task": a.steps_per_task,
    }
    out_path = a.out or (
        "runs/mot/pr2_plasticity_substrates_seeds10.json"
        if a.rerun
        else "runs/mot/pr2_plasticity_substrates.json"
    )
    result = run(cfg, "cpu", None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    keys = ("bwt_delta_ci", "speed_delta_ci", "null_supported", "verdict")
    print(json.dumps({k: result[k] for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
