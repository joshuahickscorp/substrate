"""Representation-geometry battery (D1, the math under EX12). Characterizes WHAT the frozen latent
geometry is, independent of any downstream head: linear and kernel CKA between two representations,
RSA (representational similarity by RDM correlation), effective rank (participation ratio of the
covariance spectrum), anisotropy (top-eigenvalue dominance), intrinsic dimension (participation-ratio
estimate), and nearest-neighbour overlap. These are the cheapest, most reusable diagnostics in the
repo and bound what any mechanism can possibly use (EX12 makes the matrix explicit and permanent).

All pure torch + numpy, CPU-fine, deterministic. Every function returns a float or a small dict so a
report can table it. Nothing trains; nothing touches the encoder.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import numpy as np
import torch


def _as2d(x) -> torch.Tensor:
    """Coerce to a 2D float tensor [N, D]; flatten any trailing feature dims."""
    t = torch.as_tensor(x).detach().to(torch.float64)
    if t.dim() == 1:
        t = t.unsqueeze(1)
    if t.dim() > 2:
        t = t.reshape(t.shape[0], -1)
    return t


def _center_gram(g: torch.Tensor) -> torch.Tensor:
    """Double-center a Gram matrix (the H K H operation behind CKA / HSIC)."""
    n = g.shape[0]
    unit = torch.ones(n, n, dtype=g.dtype) / n
    return g - unit @ g - g @ unit + unit @ g @ unit


def linear_cka(x, y) -> float:
    """Linear CKA in [0,1]: 1 == identical representational structure, 0 == unrelated. Symmetric,
    invariant to orthogonal transforms and isotropic scaling. The standard representation-similarity
    measure (Kornblith et al.)."""
    x, y = _as2d(x), _as2d(y)
    kx, ky = x @ x.T, y @ y.T
    cx, cy = _center_gram(kx), _center_gram(ky)
    hsic_xy = (cx * cy).sum()
    denom = torch.sqrt((cx * cx).sum() * (cy * cy).sum()) + 1e-12
    return float((hsic_xy / denom).clamp(0.0, 1.0))


def _rbf_gram(x: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(x, x) ** 2
    med = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0, dtype=x.dtype)
    sigma2 = med if med > 0 else torch.tensor(1.0, dtype=x.dtype)
    return torch.exp(-d2 / (2.0 * sigma2))


def kernel_cka(x, y) -> float:
    """RBF-kernel CKA (median-heuristic bandwidth): captures nonlinear representational similarity that
    linear CKA misses. Same [0,1] scale."""
    x, y = _as2d(x), _as2d(y)
    cx, cy = _center_gram(_rbf_gram(x)), _center_gram(_rbf_gram(y))
    hsic_xy = (cx * cy).sum()
    denom = torch.sqrt((cx * cx).sum() * (cy * cy).sum()) + 1e-12
    return float((hsic_xy / denom).clamp(0.0, 1.0))


def rsa(x, y) -> float:
    """Representational similarity analysis: Spearman-like correlation of the two upper-triangular
    pairwise-distance matrices (RDMs). Returns Pearson correlation of ranked distances in [-1,1]."""
    x, y = _as2d(x), _as2d(y)
    n = x.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    dx = torch.cdist(x, x)[iu[0], iu[1]]
    dy = torch.cdist(y, y)[iu[0], iu[1]]
    rx = dx.argsort().argsort().to(torch.float64)  # rank transform -> Spearman
    ry = dy.argsort().argsort().to(torch.float64)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = torch.sqrt((rx * rx).sum() * (ry * ry).sum()) + 1e-12
    return float((rx * ry).sum() / denom)


def _cov_eigs(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    return eig


def effective_rank(x) -> float:
    """Participation-ratio effective rank of the feature covariance: (sum eig)^2 / sum(eig^2). Ranges
    from 1 (rank-1 collapse) to D (isotropic). Low effective rank flags representational collapse."""
    eig = _cov_eigs(_as2d(x))
    s1, s2 = eig.sum(), (eig * eig).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)


def anisotropy(x) -> float:
    """Top-eigenvalue dominance: largest_eig / sum(eig) in [1/D, 1]. High anisotropy means one
    direction dominates the representation (a known pathology of some pooled encoders)."""
    eig = _cov_eigs(_as2d(x))
    s = eig.sum()
    if s <= 0:
        return 0.0
    return float(eig.max() / s)


def intrinsic_dim(x) -> float:
    """A cheap intrinsic-dimension estimate: the participation-ratio effective rank, reported
    separately as the dimensionality the data actually occupies (alias of effective_rank, named for
    the geometry table)."""
    return effective_rank(x)


def neighborhood_overlap(x, y, k: int = 5) -> float:
    """Mean Jaccard overlap of the k-nearest-neighbour sets of the SAME points under two
    representations: 1 == identical local neighbourhoods, 0 == none shared. A local-structure analogue
    to CKA (which is global)."""
    x, y = _as2d(x), _as2d(y)
    n = x.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return 0.0

    def knn(t):
        d = torch.cdist(t, t)
        d.fill_diagonal_(float("inf"))
        return d.argsort(dim=1)[:, :k]

    nx, ny = knn(x), knn(y)
    overlaps = []
    for i in range(n):
        a, b = set(nx[i].tolist()), set(ny[i].tolist())
        overlaps.append(len(a & b) / len(a | b))
    return float(np.mean(overlaps))


def geometry_report(x, reference=None, k: int = 5) -> dict:
    """One-call geometry battery over representation `x` (optionally vs a `reference` representation of
    the same points). Returns the self-geometry (effective_rank, anisotropy, intrinsic_dim) plus the
    similarity to the reference (linear_cka, kernel_cka, rsa, neighborhood_overlap) when given."""
    out = {
        "n": int(_as2d(x).shape[0]),
        "dim": int(_as2d(x).shape[1]),
        "effective_rank": round(effective_rank(x), 4),
        "anisotropy": round(anisotropy(x), 4),
        "intrinsic_dim": round(intrinsic_dim(x), 4),
    }
    if reference is not None:
        out.update(
            linear_cka=round(linear_cka(x, reference), 4),
            kernel_cka=round(kernel_cka(x, reference), 4),
            rsa=round(rsa(x, reference), 4),
            neighborhood_overlap=round(neighborhood_overlap(x, reference, k=k), 4),
        )
    return out
