"""EX8: intrinsic-motivation curriculum bake-off. Four selection signals on the noisy-TV split, where
the learnable region has reducible error and the noise region is irreducibly random. Only signals that
measure REDUCIBLE uncertainty should pick the learnable region and reject the noisy-TV; the rest chase
irreducible noise. The four: prediction-error (DA proxy), ensemble disagreement (epistemic), learning
progress (LP), and random network distillation (RND). The first three come from the same noisy-TV
diagnostic E4 uses (so the bake-off is consistent with the E4 gate); RND is added here.

NULL: prediction-error and RND chase the noisy-TV as much as a naive baseline, and only disagreement
and learning-progress reject it (the rung-6 prediction); a sharper null is that even learning-progress
ties uniform on a homogeneous stream. Taxonomy slot 10 (a signal that chases noise is mis-specified for
reducible-uncertainty selection). cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..diagnostics.noisy_tv import noisy_tv_diagnostic
from ..seeding import seed_everything
from ..shell.predictor import mlp
from .base import Experiment


def _rnd_novelty(dim: int, noise_scale: float, steps: int, batch: int, seed: int) -> tuple[float, float]:
    """RND novelty on the learnable vs noise region: a fixed random target net + a trained predictor;
    novelty is the predictor error. On the isotropic noise region the predictor covers the unbounded
    input space worse, so RND novelty stays high (it chases the noisy-TV, like prediction-error)."""
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(2, dim, generator=g)
    target = mlp(dim, dim, 128, depth=2)
    for p in target.parameters():
        p.requires_grad_(False)
    pred = mlp(dim, dim, 128, depth=2)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-3)

    def learnable(bs):
        y = torch.randint(0, 2, (bs,), generator=g)
        return centers[y] * 2.0 + 0.5 * torch.randn(bs, dim, generator=g)

    def noise(bs):
        return torch.randn(bs, dim, generator=g) * noise_scale

    for _ in range(steps):
        opt.zero_grad()
        x = torch.cat([learnable(batch), noise(batch)])
        F.mse_loss(pred(x), target(x)).backward()
        opt.step()
    with torch.no_grad():
        nl = float((pred(learnable(256)) - target(learnable(256))).abs().mean())
        nn_ = float((pred(noise(256)) - target(noise(256))).abs().mean())
    return nl, nn_


class EX8(Experiment):
    id = "ex8_curiosity_bakeoff"
    metric = ("noisy_tv_rejection", "learnable_vs_noise_value", "rejects_noise_per_signal")
    baseline = "uniform/random selection; the noisy-TV region is the decisive gate"
    ablation = "four signals: prediction-error, disagreement, learning-progress, RND"
    null_hypothesis = (
        "prediction-error and RND chase the noisy-TV as much as uniform, and only disagreement and "
        "learning-progress reject it; a sharper null is that even learning-progress ties uniform on a "
        "homogeneous stream"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, noise_scale = int(e.dim), float(e.noise_scale)
        steps, batch = int(e.steps), int(e.batch)

        agg: dict[str, dict[str, list]] = {
            s: {"learnable": [], "noise": []}
            for s in ("prediction_error", "disagreement", "learning_progress", "rnd")
        }
        for s in seeds:
            nt = noisy_tv_diagnostic(
                dim=dim,
                ensemble_size=int(e.ensemble_size),
                steps=steps,
                batch=batch,
                noise_scale=noise_scale,
                seed=s,
            )
            agg["prediction_error"]["learnable"].append(nt["raw_error"]["learnable"])
            agg["prediction_error"]["noise"].append(nt["raw_error"]["noise"])
            agg["disagreement"]["learnable"].append(nt["disagreement"]["learnable"])
            agg["disagreement"]["noise"].append(nt["disagreement"]["noise"])
            agg["learning_progress"]["learnable"].append(nt["learning_progress"]["learnable"])
            agg["learning_progress"]["noise"].append(nt["learning_progress"]["noise"])
            nl, nn_ = _rnd_novelty(dim, noise_scale, steps, batch, s)
            agg["rnd"]["learnable"].append(nl)
            agg["rnd"]["noise"].append(nn_)

        def mean(v):
            return sum(v) / len(v)

        table: dict[str, dict] = {}
        for sig, d in agg.items():
            lm, nm = mean(d["learnable"]), mean(d["noise"])
            # a signal "rejects" the noisy-TV if it does NOT rank the noise region above the learnable
            # one (it would not spend exploration budget on irreducible noise).
            table[sig] = {
                "learnable": round(lm, 4),
                "noise": round(nm, 4),
                "rejects_noise": bool(nm <= lm + 1e-4),
            }
        # the canonical pattern: chasers (prediction-error, RND) do NOT reject; epistemic + LP DO
        chasers_chase = (not table["prediction_error"]["rejects_noise"]) and (
            not table["rnd"]["rejects_noise"]
        )
        reducible_reject = (
            table["disagreement"]["rejects_noise"] and table["learning_progress"]["rejects_noise"]
        )
        return {
            "table": table,
            "seeds": list(seeds),
            "chasers_chase_noise": bool(chasers_chase),
            "reducible_signals_reject_noise": bool(reducible_reject),
            # the explicit null pattern holds: naive signals chase, reducible-uncertainty signals reject
            "null_supported": bool(chasers_chase and reducible_reject),
        }
