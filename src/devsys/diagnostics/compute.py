"""Compute accounting (D5). The reasoning experiments (EX17 refinement, EX18 self-verification) add
COMPUTE, so a fair comparison must match FLOPs, not just parameters. This module counts shell
parameters and estimates forward FLOPs for the small MLP/refiner shells, and provides a matched-compute
check so an arm cannot win by simply spending more compute. Estimates are explicit and dominated by the
linear layers (the load-bearing term); pointwise nonlinearities and norms are rounding error and noted.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import torch


def param_count(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count of a module (trainable by default)."""
    return int(sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)))


def linear_flops(in_dim: int, out_dim: int, batch: int = 1) -> int:
    """Forward multiply-add FLOPs for a Linear layer: 2 * batch * in * out (the 2 counts mul + add).
    Bias is +batch*out, rounding error, omitted."""
    return int(2 * batch * in_dim * out_dim)


def mlp_flops(dims: list[int], batch: int = 1) -> int:
    """Forward FLOPs for an MLP described by its layer dimensions [d_in, h1, ..., d_out]. Sums the
    per-Linear FLOPs; nonlinearities/norms are O(batch*width) and omitted as rounding error."""
    return int(sum(linear_flops(dims[i], dims[i + 1], batch) for i in range(len(dims) - 1)))


def refiner_flops(dim: int, hidden: int, steps: int, batch: int = 1) -> int:
    """Forward FLOPs for an iterative refiner of `steps` residual blocks (each a dim->hidden->dim MLP).
    This is the quantity EX17 must match against a single-pass deep MLP of equal total FLOPs."""
    per_step = mlp_flops([dim, hidden, dim], batch)
    return int(steps * per_step)


def attention_flops(n_tokens: int, dim: int, batch: int = 1) -> int:
    """Forward multiply-add FLOPs for one full self-attention layer over `n_tokens` tokens of width
    `dim`: QKV + output projections (4 Linears, 8*n*d^2) plus score and value matmuls (2 * 2*n^2*d).
    Head count does not change the total (dim is split across heads). Softmax/scale are O(n^2) and
    omitted as rounding error. Used by the MT7 scorer, WS slot attention, and any matched-compute
    check involving an attention arm (WP-02)."""
    n, d = int(n_tokens), int(dim)
    proj = 4 * linear_flops(d, d, batch=batch * n)
    scores = int(2 * batch * n * n * d) * 2  # QK^T plus attn @ V
    return int(proj + scores)


def knn_flops(n_queries: int, n_keys: int, dim: int, batch: int = 1) -> int:
    """Forward FLOPs for brute-force kNN retrieval: the query-key similarity matmul, 2*q*k*d per batch
    element (the load-bearing term). Top-k selection is O(q*k) comparisons and omitted as rounding
    error. This is the retrieval cost DR10/PR8 must charge against their matched baselines: retrieval
    compute counts, exactly like pruned search branches (WP-02)."""
    return int(2 * batch * n_queries * n_keys * dim)


def matched_within(flops_a: int, flops_b: int, tol: float = 0.10) -> dict:
    """Are two arms matched on compute within a relative tolerance. Returns {matched, ratio, tol}. A
    measured gain is only trustworthy when matched is True (else it may be bought compute, taxonomy 9)."""
    hi = max(flops_a, flops_b)
    lo = min(flops_a, flops_b)
    ratio = (hi / lo) if lo > 0 else float("inf")
    return {
        "matched": bool(ratio <= 1.0 + tol),
        "ratio": round(ratio, 4),
        "tol": tol,
        "flops": [int(flops_a), int(flops_b)],
    }


def depth_for_matched_flops(dim: int, hidden: int, refiner_steps: int) -> int:
    """The number of dim->hidden->dim blocks a single-pass MLP needs to match a refiner of
    `refiner_steps` blocks (here exactly refiner_steps, since each block has equal FLOPs). The
    compute-matched control for EX17: same total blocks, applied once each vs iterated on a residual."""
    return int(refiner_steps)


def accounting(module: torch.nn.Module, dims: list[int], steps: int = 1, batch: int = 1) -> dict:
    """A compact compute record for a shell: params + estimated forward FLOPs (one pass, `steps`
    iterations). Stamped into a run manifest so matched-compute is auditable after the fact."""
    return {
        "params": param_count(module),
        "flops_per_pass": mlp_flops(dims, batch),
        "steps": int(steps),
        "flops_total": int(steps * mlp_flops(dims, batch)),
        "batch": int(batch),
        "note": "FLOPs dominated by linear layers; nonlinearities/norms omitted as rounding error",
    }
