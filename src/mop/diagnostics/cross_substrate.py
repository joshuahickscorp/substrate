from __future__ import annotations

import torch

from .geometry import linear_cka
from .linear_probe import linear_probe


def random_map_floor(
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int = 0,
    probe_epochs: int = 200,
) -> float:
    g = torch.Generator().manual_seed(seed)
    d = x.shape[1]
    m = torch.randn(d, d, generator=g) / d**0.5
    return float(linear_probe(x @ m, labels, epochs=probe_epochs, seed=seed)["score"])


def shuffled_label_null(
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int = 0,
    probe_epochs: int = 200,
) -> float:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(labels.shape[0], generator=g)
    return float(linear_probe(x, labels[perm], epochs=probe_epochs, seed=seed)["score"])


def cross_substrate_agreement(
    latents: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    seed: int = 0,
    probe_epochs: int = 200,
) -> dict:
    tags = sorted(latents)
    probe_acc = {
        t: float(linear_probe(latents[t], labels, epochs=probe_epochs, seed=seed)["score"]) for t in tags
    }
    null = {t: shuffled_label_null(latents[t], labels, seed=seed, probe_epochs=probe_epochs) for t in tags}
    floor = {t: random_map_floor(latents[t], labels, seed=seed, probe_epochs=probe_epochs) for t in tags}
    cka = {a: {b: round(linear_cka(latents[a], latents[b]), 4) for b in tags} for a in tags}
    return {
        "tags": tags,
        "probe_acc": {t: round(v, 4) for t, v in probe_acc.items()},
        "shuffled_null": {t: round(v, 4) for t, v in null.items()},
        "random_map_floor": {t: round(v, 4) for t, v in floor.items()},
        "cka": cka,
    }
