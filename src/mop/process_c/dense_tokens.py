"""Process C dense-token module.

Process C is not licensed by default. The sanctioned pilot is narrow: a 1 to 10M trainable
object-centric module over frozen dense tokens, compared to dense tokens without slots and run only after
PR9 or DR1 licenses it. This file provides that module and its bookkeeping, not a launcher.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def param_count(module: nn.Module) -> int:
    """Trainable parameter count."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def dense_hidden_for_target_params(input_dim: int, n_classes: int, target_params: int) -> int:
    """Hidden width for mean-pooled dense-token baseline at roughly `target_params`.

    Baseline params are input_dim*hidden + hidden + hidden*n_classes + n_classes. This helper is the
    Process C control: slots must beat a dense-token arm at matched capacity, not buy the win.
    """
    denom = input_dim + n_classes + 1
    return max(1, int(round(max(1, target_params - n_classes) / denom)))


class DenseTokenSlotModule(nn.Module):
    """Learn object-centric slots from frozen dense tokens.

    Input: tokens [B, N, D]. Output: slots [B, K, H], a pooled representation [B, H], and attention
    weights [B, K, N]. The module is intentionally small and shell-side: it never touches encoder weights.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_slots: int,
        slot_dim: int,
        iterations: int = 2,
        mlp_hidden: int | None = None,
    ):
        super().__init__()
        if n_slots <= 0:
            raise ValueError("n_slots must be positive")
        if slot_dim <= 0:
            raise ValueError("slot_dim must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.input_dim = int(input_dim)
        self.n_slots = int(n_slots)
        self.slot_dim = int(slot_dim)
        self.iterations = int(iterations)
        hidden = int(mlp_hidden or slot_dim * 2)

        self.token_proj = nn.Linear(input_dim, slot_dim)
        self.slots = nn.Parameter(torch.randn(n_slots, slot_dim) * (slot_dim**-0.5))
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.update = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, slot_dim),
        )
        self.norm_tokens = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError("dense tokens must have shape [B, N, D]")
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(f"token dim {tokens.shape[-1]} != input_dim {self.input_dim}")
        b, n = tokens.shape[0], tokens.shape[1]
        x = self.norm_tokens(self.token_proj(tokens))
        slots = self.slots.unsqueeze(0).expand(b, -1, -1)
        attn = torch.empty(b, self.n_slots, n, device=tokens.device, dtype=tokens.dtype)

        for _ in range(self.iterations):
            q = self.q_proj(self.norm_slots(slots))
            k = self.k_proj(x)
            v = self.v_proj(x)
            logits = q @ k.transpose(-1, -2) / math.sqrt(self.slot_dim)
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None, :].bool(), -1e9)
            attn = torch.softmax(logits, dim=-1)
            updates = attn @ v
            slots = self.update(updates.reshape(-1, self.slot_dim), slots.reshape(-1, self.slot_dim)).view(
                b, self.n_slots, self.slot_dim
            )
            slots = slots + self.mlp(slots)

        pooled = slots.mean(dim=1)
        return {
            "slots": slots,
            "pooled": pooled,
            "attention": attn,
            "assignment_entropy": assignment_entropy(attn).to(tokens.device),
        }


class ProcessCDenseTokenClassifier(nn.Module):
    """Slot module plus a classifier head for CM9-style binding probes."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        *,
        n_slots: int,
        slot_dim: int,
        iterations: int = 2,
        mlp_hidden: int | None = None,
    ):
        super().__init__()
        self.slots = DenseTokenSlotModule(
            input_dim,
            n_slots=n_slots,
            slot_dim=slot_dim,
            iterations=iterations,
            mlp_hidden=mlp_hidden,
        )
        self.head = nn.Linear(slot_dim, n_classes)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = self.slots(tokens, mask=mask)
        out["logits"] = self.head(out["pooled"])
        return out


class DenseTokenMeanBaseline(nn.Module):
    """Dense-token control with no object slots."""

    def __init__(self, input_dim: int, n_classes: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("dense tokens must have shape [B, N, D]")
        if mask is None:
            pooled = tokens.mean(dim=1)
        else:
            weights = mask.float().unsqueeze(-1)
            pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.net(pooled)


def assignment_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention entropy over slots and batch, normalized by log(token_count)."""
    p = attn.clamp_min(1e-9)
    raw = -(p * p.log()).sum(dim=-1).mean()
    return raw / math.log(max(attn.shape[-1], 2))


def binding_specificity(before: torch.Tensor, after: torch.Tensor, target_slot: int) -> dict[str, Any]:
    """Slot-swap diagnostic: target slot should move more than non-target slots."""
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("before and after must both have shape [B, K, H]")
    k = before.shape[1]
    if target_slot < 0 or target_slot >= k:
        raise ValueError("target_slot out of range")
    delta = (after - before).norm(dim=-1).mean(dim=0)
    target = float(delta[target_slot])
    if k == 1:
        off_target = 0.0
    else:
        others = torch.cat([delta[:target_slot], delta[target_slot + 1 :]])
        off_target = float(others.max())
    return {
        "target_delta": round(target, 6),
        "max_off_target_delta": round(off_target, 6),
        "specificity_ratio": round(target / max(off_target, 1e-9), 6),
        "target_is_largest": bool(target >= off_target),
    }


def process_c_budget_report(
    module: nn.Module,
    *,
    licensed: bool,
    min_params: int = 1_000_000,
    max_params: int = 10_000_000,
) -> dict[str, Any]:
    """Budget and license gate for the sanctioned Process C pilot."""
    params = param_count(module)
    problems: list[str] = []
    if not licensed:
        problems.append("Process C not licensed by PR9/DR1 gate")
    if params < min_params:
        problems.append(f"params {params} below Process C floor {min_params}")
    if params > max_params:
        problems.append(f"params {params} above Process C ceiling {max_params}")
    return {
        "params": params,
        "min_params": int(min_params),
        "max_params": int(max_params),
        "licensed": bool(licensed),
        "within_budget": not problems,
        "problems": problems,
    }
