from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import torch

from .geometry import geometry_report, kernel_cka, linear_cka, neighborhood_overlap, rsa
from .seed_consistency import cross_seed_cka

SCHEMA = "mop-alignment-suite/v1"


def permutation_pvalue(observed: float, null_samples: Sequence[float], greater: bool = True) -> float:
    n = len(null_samples)
    if greater:
        hits = sum(1 for s in null_samples if s >= observed)
    else:
        hits = sum(1 for s in null_samples if s <= observed)
    return (1 + hits) / (1 + n)


def alignment_null(x: torch.Tensor, y: torch.Tensor, n_permutations: int = 200, seed: int = 0) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    out: list[float] = []
    for _ in range(n_permutations):
        perm = torch.randperm(y.shape[0], generator=g)
        out.append(linear_cka(x, y[perm]))
    return out


def alignment_suite(
    x: torch.Tensor | Mapping[str, torch.Tensor],
    y: torch.Tensor | None = None,
    *,
    n_permutations: int = 200,
    seed: int = 0,
    k: int = 5,
) -> dict:
    if isinstance(x, Mapping):
        if y is not None:
            raise ValueError("mapping mode does not accept a second representation")
        return alignment_table(x, n_permutations=n_permutations, seed=seed, k=k)
    if y is None:
        raise ValueError("pair mode requires y")
    return pair_alignment(x, y, n_permutations=n_permutations, seed=seed, k=k)


def pair_alignment(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_permutations: int = 200,
    seed: int = 0,
    k: int = 5,
) -> dict[str, Any]:
    lc = linear_cka(x, y)
    null = alignment_null(x, y, n_permutations=n_permutations, seed=seed)
    return {
        "linear_cka": round(lc, 4),
        "kernel_cka": round(kernel_cka(x, y), 4),
        "rsa": round(rsa(x, y), 4),
        "neighborhood_overlap": round(neighborhood_overlap(x, y, k=k), 4),
        "null_mean": round(sum(null) / max(len(null), 1), 4),
        "p_value": round(permutation_pvalue(lc, null), 4),
        "n_permutations": int(n_permutations),
    }


def alignment_table(
    reps: Mapping[str, torch.Tensor],
    *,
    n_permutations: int = 200,
    seed: int = 0,
    k: int = 5,
) -> dict[str, Any]:
    prepared = _prepare_reps(reps)
    tags = sorted(prepared)
    n = int(next(iter(prepared.values())).shape[0])
    self_geometry = {tag: geometry_report(prepared[tag], k=k) for tag in tags}
    pair_reports: dict[str, dict[str, Any]] = {}
    matrices: dict[str, dict[str, dict[str, float]]] = {
        "linear_cka": _identity_matrix(tags, 1.0),
        "kernel_cka": _identity_matrix(tags, 1.0),
        "rsa": _identity_matrix(tags, 1.0),
        "neighborhood_overlap": _identity_matrix(tags, 1.0),
    }

    for i, (left, right) in enumerate(combinations(tags, 2)):
        report = pair_alignment(
            prepared[left],
            prepared[right],
            n_permutations=n_permutations,
            seed=seed + i,
            k=k,
        )
        key = f"{left}__{right}"
        pair_reports[key] = {"left": left, "right": right, **report}
        for metric in matrices:
            value = float(report[metric])
            matrices[metric][left][right] = value
            matrices[metric][right][left] = value

    return {
        "schema": SCHEMA,
        "n": n,
        "tags": tags,
        "self_geometry": self_geometry,
        "pairs": pair_reports,
        "matrices": matrices,
        "n_permutations": int(n_permutations),
        "k": int(k),
        "doctrine": (
            "CKA/RSA/local-neighborhood alignment is a geometry report, not substrate-specialness proof; "
            "claims still need a random-encoder or matched-control arm."
        ),
        "warnings": _alignment_warnings(tags),
    }


def cross_seed_alignment(reps: list[torch.Tensor]) -> dict:
    return cross_seed_cka(reps)


def _prepare_reps(reps: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for raw_tag, value in reps.items():
        tag = str(raw_tag)
        if not tag:
            raise ValueError("alignment tags must be non-empty")
        if tag in out:
            raise ValueError(f"duplicate alignment tag: {tag}")
        t = torch.as_tensor(value).detach().float()
        if t.ndim < 2:
            raise ValueError(f"representation {tag!r} must have shape [N, ...]")
        out[tag] = t.flatten(1)
    if len(out) < 2:
        raise ValueError("AlignmentSuite needs at least two representations")
    ns = {int(t.shape[0]) for t in out.values()}
    if len(ns) != 1:
        sizes = {tag: int(t.shape[0]) for tag, t in out.items()}
        raise ValueError(f"all representations must share N referents, got {sizes}")
    return out


def _identity_matrix(tags: Sequence[str], value: float) -> dict[str, dict[str, float]]:
    return {tag: {other: (float(value) if other == tag else 0.0) for other in tags} for tag in tags}


def _alignment_warnings(tags: Sequence[str]) -> list[str]:
    lowered = {tag: tag.lower() for tag in tags}
    warnings: list[str] = []
    if any("projection" in tag and "random" in tag for tag in lowered.values()):
        warnings.append(
            "random projection controls are rotation/linear-map controls, not random-encoder controls"
        )
    has_random_encoder = any(
        ("random_init" in tag or "randinit" in tag or "random-encoder" in tag) for tag in lowered.values()
    )
    if not has_random_encoder:
        warnings.append("no random-encoder control tag detected")
    return warnings
