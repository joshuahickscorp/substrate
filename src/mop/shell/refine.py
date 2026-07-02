"""Iterative latent refinement (the EX17 primitive, the latent-reasoning line). A trainable shell
module that refines a frozen-encoder latent over N residual steps before the head reads it, optionally
with confidence-based adaptive halting (ACT-lite). The scientific question it exists to test is sharp:
does iterating computation in latent space beat a COMPUTE-MATCHED single-pass network, or was the gain
just depth. So this module is always compared against a depth-and-FLOP-matched MLP (diagnostics/compute).

This is NOT the encoder and never touches it; it operates entirely on cached latents. A fixed-step
refiner is exactly a residual network unrolled in place (so the compute-matched control is a plain MLP
of equal block count); the adaptive variant adds a halt head that can stop early per sample.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import torch
from torch import nn

from .predictor import mlp


class IterativeRefiner(nn.Module):
    """Refine a latent z over up to `steps` residual updates: z <- z + block(LayerNorm(z)). The SAME
    block is applied each step (weight-tied recurrence), which is what distinguishes iteration from
    depth, the single-pass control unties the weights across equal block count at matched FLOPs.

    halt=True adds a per-step halt head; refinement stops for a sample once its cumulative halt
    probability crosses `halt_threshold` (adaptive compute). Returns the refined latent and the number
    of steps each sample actually used (a scalar mean is also reported by callers).
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        steps: int = 4,
        halt: bool = False,
        halt_threshold: float = 0.9,
        mode: str = "residual",
        pc_rate: float = 0.5,
    ):
        super().__init__()
        if mode not in ("residual", "predictive_coding"):
            raise ValueError(f"mode must be residual or predictive_coding, got {mode!r}")
        self.dim = dim
        self.steps = int(steps)
        self.halt = bool(halt)
        self.halt_threshold = float(halt_threshold)
        self.mode = mode
        self.pc_rate = float(pc_rate)
        self.norm = nn.LayerNorm(dim)
        self.block = mlp(dim, dim, hidden, depth=1, ln=True)  # one dim->hidden->dim residual block
        self.halt_head = nn.Linear(dim, 1) if halt else None

    def _update(self, z: torch.Tensor) -> torch.Tensor:
        """One step's latent update. residual: u = block(LN(z)). predictive_coding: a one-level PC
        settle, u = pc_rate * (block(LN(z)) - z), descending the prediction-error energy toward the
        block's predicted manifold (the SAME block, so the matched-compute control is unchanged; only
        the update RULE differs, which is exactly the N3 question)."""
        pred = self.block(self.norm(z))
        if self.mode == "predictive_coding":
            return self.pc_rate * (pred - z)
        return pred

    def forward(self, z: torch.Tensor, max_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = max_steps or self.steps
        b = z.shape[0]
        used = torch.zeros(b, dtype=torch.float32, device=z.device)
        if not self.halt:
            for _ in range(n):
                z = z + self._update(z)
            used = used + n
            return z, used
        # adaptive halting: accumulate halt probability, freeze samples that have halted
        assert self.halt_head is not None  # constructed iff halt=True (the branch guard above)
        cum = torch.zeros(b, device=z.device)
        active = torch.ones(b, dtype=torch.bool, device=z.device)
        for _ in range(n):
            upd = self._update(z)
            z = torch.where(active.unsqueeze(-1), z + upd, z)
            used = used + active.float()
            p = torch.sigmoid(self.halt_head(z)).squeeze(-1)
            cum = cum + torch.where(active, p, torch.zeros_like(p))
            active = active & (cum < self.halt_threshold)
            if not active.any():
                break
        return z, used

    @torch.no_grad()
    def unroll(self, z: torch.Tensor, steps: int) -> tuple[torch.Tensor, list[float]]:
        """Unroll the refiner for `steps` (typically K >> trained N) WITHOUT halting, returning the
        final latent and the per-step mean update norm ||z_{t+1}-z_t||. A geometrically decaying norm
        sequence is the contraction signature (the refiner is an attractor); a flat or growing sequence
        is drift (unrolled depth). Used by diagnostics/convergence.py for Y1/Y2/N9."""
        norms: list[float] = []
        for _ in range(int(steps)):
            u = self._update(z)
            norms.append(float(u.norm(dim=-1).mean()))
            z = z + u
        return z, norms

    def block_count(self, max_steps: int | None = None) -> int:
        """The number of residual blocks a single FORWARD applies at full steps, the matched-compute
        control is a single-pass MLP of this many dim->hidden->dim blocks (diagnostics/compute)."""
        return int(max_steps or self.steps)


class Verifier(nn.Module):
    """A small head that scores a refined latent and proposes a correction (N11/Y9/EX18 self-correction).
    score(z) is a scalar error estimate (high == likely wrong); the experiment uses it to trigger an
    extra refine step on low-confidence samples, then checks verify-revise beats single-shot at MATCHED
    compute (else the verifier carries no usable signal, taxonomy 4). Tiny: dim->hidden->1."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(dim, 1, hidden, depth=1, ln=True)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.score(z)
