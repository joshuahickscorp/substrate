"""EX16: discrete codebook / VQ abstraction (and a successor-representation probe). Does a learned VQ
codebook over the frozen latents form reusable discrete concepts that decode the label better than a
RANDOM codebook, and does it add structure over the raw-latent probe. The codebook is fit by k-means
(the VQ assignment in the no-pixel-decoder setting); cluster purity against labels is the headline; a
random-codebook assignment and the raw-latent linear probe are the controls.

NULL: the codebook adds no decodable structure over raw latents (a random codebook ties the learned
one on purity, and the code-probe does not beat the raw-latent probe). Taxonomy slot 3 (no extra
structure in the pooled latent to discretize) or 4 (codebook too small). cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..diagnostics.linear_probe import linear_probe
from ..seeding import seed_everything
from .base import Experiment


def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    """Lloyd k-means; returns the hard code assignment per row. The VQ codebook in the frozen-latent
    setting (no pixel decoder, the codes ARE the abstraction)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    cent = x[idx].clone()
    codes = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, cent)
        codes = d.argmin(1)
        for c in range(k):
            m = codes == c
            if m.any():
                cent[c] = x[m].mean(0)
    return codes


def _purity(codes: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Cluster purity: the fraction of points whose cluster-majority label matches their own."""
    correct = 0
    for c in range(k):
        m = codes == c
        if m.any():
            correct += int(torch.bincount(y[m]).max())
    return correct / int(y.shape[0])


class EX16(Experiment):
    id = "ex16_codebook_sr"
    metric = ("cluster_purity", "purity_vs_random_codebook", "code_probe_vs_raw")
    baseline = "raw-latent linear probe and a random-codebook assignment at matched k"
    ablation = "learned VQ codebook vs random codebook; one-hot code probe vs raw-latent probe"
    null_hypothesis = (
        "the codebook adds no decodable structure over raw latents: a random codebook ties the learned "
        "one on purity, and the one-hot code probe does not beat the raw-latent probe"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, k = int(e.dim), int(e.codebook_size)
        learned_pur, random_pur, code_acc, raw_acc = [], [], [], []
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
            x, y = task.x, task.y
            codes = _kmeans(x, k, int(e.kmeans_iters), s)
            g = torch.Generator().manual_seed(s + 1)
            rand_codes = torch.randint(0, k, (x.shape[0],), generator=g)
            learned_pur.append(_purity(codes, y, k))
            random_pur.append(_purity(rand_codes, y, k))
            onehot = torch.zeros(x.shape[0], k)
            onehot[torch.arange(x.shape[0]), codes] = 1.0
            code_acc.append(linear_probe(onehot, y, seed=s)["score"])
            raw_acc.append(linear_probe(x, y, seed=s)["score"])

        def mean(v):
            return sum(v) / len(v)

        lp, rp = mean(learned_pur), mean(random_pur)
        ca, ra = mean(code_acc), mean(raw_acc)
        out = {
            "cluster_purity_learned": round(lp, 4),
            "cluster_purity_random": round(rp, 4),
            "purity_gain_vs_random": round(lp - rp, 4),
            "code_probe_acc": round(ca, 4),
            "raw_latent_probe_acc": round(ra, 4),
            "code_vs_raw_gain": round(ca - ra, 4),
            "codebook_size": k,
            "seeds": list(seeds),
            # the explicit null: codebook adds nothing (ties random purity AND does not beat raw probe)
            "null_supported": bool((lp - rp) <= 0.05 and (ca - ra) <= 0.0),
            "codebook_beats_random": bool((lp - rp) > 0.05),
        }
        return out
