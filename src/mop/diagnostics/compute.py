from __future__ import annotations

import torch


def param_count(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return int(sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)))


def linear_flops(in_dim: int, out_dim: int, batch: int = 1) -> int:
    return int(2 * batch * in_dim * out_dim)


def mlp_flops(dims: list[int], batch: int = 1) -> int:
    return int(sum(linear_flops(dims[i], dims[i + 1], batch) for i in range(len(dims) - 1)))


def refiner_flops(dim: int, hidden: int, steps: int, batch: int = 1) -> int:
    per_step = mlp_flops([dim, hidden, dim], batch)
    return int(steps * per_step)


def attention_flops(n_tokens: int, dim: int, batch: int = 1) -> int:
    n, d = int(n_tokens), int(dim)
    proj = 4 * linear_flops(d, d, batch=batch * n)
    scores = int(2 * batch * n * n * d) * 2  # QK^T plus attn @ V
    return int(proj + scores)


def knn_flops(n_queries: int, n_keys: int, dim: int, batch: int = 1) -> int:
    return int(2 * batch * n_queries * n_keys * dim)


def matched_within(flops_a: int, flops_b: int, tol: float = 0.10) -> dict:
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
    return int(refiner_steps)


def accounting(module: torch.nn.Module, dims: list[int], steps: int = 1, batch: int = 1) -> dict:
    return {
        "params": param_count(module),
        "flops_per_pass": mlp_flops(dims, batch),
        "steps": int(steps),
        "flops_total": int(steps * mlp_flops(dims, batch)),
        "batch": int(batch),
        "note": "FLOPs dominated by linear layers; nonlinearities/norms omitted as rounding error",
    }
