"""A3: replay-buffer compression. Quantizing stored replay latents to fewer bits trades memory for
distortion; this is the reusable retention-per-byte diagnostic (the pattern I5 rate-distortion replay
already exercises inline, generalized here so other experiments can call it directly instead of
hand-rolling the continual + quantize-on-store loop).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..seeding import seed_everything
from .substrate_ablation import quantize_dequantize


def _bwt_at_bits(tasks, dim: int, n_classes: int, bits: int, epochs: int, lr: float, seed: int) -> float:
    """Train continually over `tasks` ([.x, .y] per task), replaying each task's latents quantized to
    `bits` (bits=32 means unquantized). Returns backward transfer on task 0: acc after the full stream
    minus acc right after task 0 (negative = forgetting)."""
    from ..shell.buffer import ReplayBuffer

    seed_everything(seed)
    head = nn.Linear(dim, n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    buf = ReplayBuffer(int(1e6), dim, prioritized=False, seed=seed)
    acc_first = None
    for ti, task in enumerate(tasks):
        xt, yt = task.x, task.y
        for _ in range(epochs):
            opt.zero_grad()
            xb, yb = xt, yt
            if len(buf) > 0:
                s = buf.sample(min(len(buf), xt.shape[0]))
                xb = torch.cat([xt, s["x"]])
                yb = torch.cat([yt, s["y"]])
            F.cross_entropy(head(xb), yb).backward()
            opt.step()
        xq = xt if bits >= 32 else quantize_dequantize(xt, bits)
        buf.add(xq, yt)
        if ti == 0:
            with torch.no_grad():
                acc_first = float((head(xt).argmax(-1) == yt).float().mean())
    with torch.no_grad():
        acc_end = float((head(tasks[0].x).argmax(-1) == tasks[0].y).float().mean())
    return acc_end - (acc_first or 0.0)


def retention_per_byte(
    tasks, dim: int, n_classes: int, bits=(32, 8, 4, 2), epochs: int = 40, lr: float = 0.05, seed: int = 0
) -> dict:
    """Sweep stored-latent bit depth; return backward_transfer per bit level, bytes-per-exemplar at each
    level (dim * bits / 8), and whether a clear retention-per-byte frontier separates from the
    full-precision (32-bit) arm (low-bit replay ties full precision => memory is cheap here; collapses
    early => memory needs precision)."""
    curve = {b: round(_bwt_at_bits(tasks, dim, n_classes, b, epochs, lr, seed), 4) for b in bits}
    bytes_per_exemplar = {b: round(dim * b / 8, 2) for b in bits}
    full = curve[max(bits)]
    ties_full_precision = {b: bool(abs(curve[b] - full) < 0.05) for b in bits}
    lowest = min(bits)
    return {
        "bits": list(bits),
        "backward_transfer": curve,
        "bytes_per_exemplar": bytes_per_exemplar,
        "ties_full_precision": ties_full_precision,
        "collapses_at_lowest_bits": bool(not ties_full_precision[lowest]),
        "frontier_present": bool(any(not v for v in ties_full_precision.values())),
    }
