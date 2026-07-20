
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch import nn  # noqa: E402

from ..devices import DeviceInfo, safe_to  # noqa: E402
from ..environments import bounded_trajectory_contract  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..shell import Predictor  # noqa: E402
from ..substrate.datasets import noisy_tv_dataset  # noqa: E402
from .base import Experiment  # noqa: E402


@dataclass
class Pool:
    x: torch.Tensor  # [N, D] current latents
    xnext: torch.Tensor  # [N, D] forward-dynamics targets
    is_noise: torch.Tensor  # [N] bool, True for noisy-TV samples


def _build_pool(dim: int, n_each: int, separation: float, noise_scale: float, seed: int) -> Pool:
    d = noisy_tv_dataset(dim=dim, n=n_each, separation=separation, noise_scale=noise_scale, seed=seed)
    lrn, noi = d["learnable"], d["noise"]
    x = torch.cat([lrn.x, noi.x], dim=0)
    xnext = torch.cat([lrn.xnext, noi.xnext], dim=0)  # type: ignore[list-item]
    is_noise = torch.cat([torch.zeros(n_each, dtype=torch.bool), torch.ones(n_each, dtype=torch.bool)])
    return Pool(x, xnext, is_noise)


def _per_sample_error(model: nn.Module, x: torch.Tensor, xnext: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return ((model(x) - xnext) ** 2).mean(dim=1)


def _train_on(
    model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, xnext: torch.Tensor, steps: int
) -> None:
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(x) - xnext) ** 2).mean()
        loss.backward()
        opt.step()


def _select(
    policy: str, err: torch.Tensor, prev_err: torch.Tensor | None, k: int, g: torch.Generator
) -> torch.Tensor:
    if policy == "random":
        return torch.randperm(err.shape[0], generator=g)[:k]
    if policy == "prediction_error":
        return torch.topk(err, k).indices  # chase highest current error
    if policy == "learning_progress":
        if prev_err is None:
            return torch.randperm(err.shape[0], generator=g)[:k]  # first round: no progress signal yet
        progress = prev_err - err  # positive where error is decreasing
        return torch.topk(progress, k).indices
    raise ValueError(f"unknown policy {policy!r}")


class E5(Experiment):
    id = "e5_curiosity"
    metric = ("noise_fraction", "learnable_improvement", "lp_resists_noise", "pe_chases_noise")
    baseline = "random/uniform data selection over the fixed learnable+noise pool"
    ablation = "selection policy: random vs prediction-error curiosity vs learning-progress curiosity"
    null_hypothesis = (
        "prediction-error/RND curiosity chases the noisy-TV (high selected-noise fraction) while "
        "learning-progress curiosity does not; if even learning-progress fixates, the progress "
        "signal is miscomputed, not grounded in learnability"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seed_everything(int(cfg.seed))
        dev = device.device
        dim, n_each = int(e.dim), int(e.n_each)
        rounds, k, steps = int(e.rounds), int(e.select_k), int(e.train_steps)
        hidden, depth, lr = int(e.hidden), int(e.depth), float(e.lr)

        pool = _build_pool(dim, n_each, float(e.separation), float(e.noise_scale), int(cfg.seed))
        x = safe_to(pool.x, dev)
        xnext = safe_to(pool.xnext, dev)
        is_noise = pool.is_noise
        learn_mask = ~is_noise

        policies = ("random", "prediction_error", "learning_progress")
        results: dict[str, dict] = {}
        for policy in policies:
            seed_everything(int(cfg.seed))  # identical init/data per policy: only selection differs
            g = torch.Generator().manual_seed(int(cfg.seed))
            model = Predictor(dim=dim, hidden=hidden, depth=depth, action_dim=0).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            err = _per_sample_error(model, x, xnext)
            base_learn_err = float(err[learn_mask].mean())
            prev_err = None
            noise_hits, total_sel = 0, 0
            for _ in range(rounds):
                idx = _select(policy, err, prev_err, k, g)
                noise_hits += int(is_noise[idx.cpu()].sum())
                total_sel += idx.shape[0]
                _train_on(model, opt, x[idx], xnext[idx], steps)
                prev_err = err
                err = _per_sample_error(model, x, xnext)
            final_learn_err = float(err[learn_mask].mean())
            results[policy] = {
                "noise_fraction": noise_hits / max(1, total_sel),
                "base_learnable_err": base_learn_err,
                "final_learnable_err": final_learn_err,
                "learnable_improvement": base_learn_err - final_learn_err,
            }

        pe_nf = results["prediction_error"]["noise_fraction"]
        lp_nf = results["learning_progress"]["noise_fraction"]
        rand_nf = results["random"]["noise_fraction"]
        thresh = float(e.noise_threshold)
        pe_chases_noise = pe_nf > thresh and pe_nf > lp_nf
        lp_resists_noise = lp_nf <= thresh and lp_nf < pe_nf
        environment_contract = bounded_trajectory_contract(
            seed=int(cfg.seed),
            episodes=int(e.environment_episodes),
            horizon=int(e.environment_horizon),
        )
        out = {
            "policies": dict(results),
            "noise_fraction": {p: results[p]["noise_fraction"] for p in policies},
            "learnable_improvement": {p: results[p]["learnable_improvement"] for p in policies},
            "random_noise_fraction": rand_nf,
            "noise_threshold": thresh,
            "pe_chases_noise": pe_chases_noise,  # null check, half 1
            "lp_resists_noise": lp_resists_noise,  # null check, half 2
            "null_supported": pe_chases_noise and lp_resists_noise,
            "env_smoke_ok": environment_contract["verified"],  # compatibility key; now a full replay
            "env_rollout_ok": environment_contract["verified"],
            "env_tier": "cpu-now",
            "environment_contract": environment_contract,
            "environment_scope": "programmatic local mechanics; no natural-embodiment claim",
        }
        self._plot(results, policies, thresh, run_dir)
        return out

    def _plot(self, results, policies, thresh, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        nf = [results[p]["noise_fraction"] for p in policies]
        imp = [results[p]["learnable_improvement"] for p in policies]
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(9, 4))
        a0.bar(range(len(policies)), nf, color=["#888", "#c44", "#48c"])
        a0.axhline(thresh, color="k", linestyle="--", alpha=0.6, label="noise threshold")
        a0.set_xticks(range(len(policies)))
        a0.set_xticklabels(policies, rotation=20, ha="right", fontsize=8)
        a0.set_ylabel("selected-sample noise fraction")
        a0.set_title("noise attraction (noisy-TV)")
        a0.legend(fontsize=8)
        a1.bar(range(len(policies)), imp, color=["#888", "#c44", "#48c"])
        a1.set_xticks(range(len(policies)))
        a1.set_xticklabels(policies, rotation=20, ha="right", fontsize=8)
        a1.set_ylabel("learnable error reduction")
        a1.set_title("downstream improvement")
        fig.suptitle("E5: curiosity as self-curriculum (data-selection)")
        fig.tight_layout()
        fig.savefig(run_dir / "e5_curiosity.png", dpi=110)
        plt.close(fig)
