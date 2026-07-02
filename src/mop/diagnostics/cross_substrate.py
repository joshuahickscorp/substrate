"""CrossSubstrateAgreement (WP-02): operationalizes standing-control 8 (universal vs
modality/objective/architecture-specific). Input: dict[substrate_tag -> latents [N, D_tag]] with SHARED
labels (same N clips through every adapter). Output: per-substrate probe accuracy, the cross-substrate
CKA matrix, a shuffled-label probe null per substrate, and a random-map-of-equal-rank floor (a seeded
Gaussian map to the same dim, the honest 'any full-rank map decodes this much' floor, NOT a within-
latent square projection). Pure aggregation over linear_probe + geometry; requires precomputed caches,
never runs two encoders.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

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
    """Probe accuracy after a FIXED seeded Gaussian map of equal rank (D -> D). Because a full-rank
    linear map preserves linear decodability up to conditioning, this floor should sit near the raw
    probe; a large gap flags a conditioning artifact, not substrate structure."""
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
    """Probe accuracy with labels permuted once (seeded): the chance-level floor for this class
    balance. Any substrate's real probe must clear this to claim decodability."""
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
    """The full agreement report. Keys: probe_acc (tag -> float), shuffled_null (tag -> float),
    random_map_floor (tag -> float), cka (tag_a -> tag_b -> float, symmetric with 1.0 diagonal),
    tags. Verdict logic (permutation p-values, convergence call) lives in the consuming experiment,
    not here; this module only measures."""
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
