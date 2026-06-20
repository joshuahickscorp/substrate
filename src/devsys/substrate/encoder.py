"""The frozen perceptual substrate. V-JEPA loaded once, never trained, inference-only.

This is the ONE large object and it is held fixed: requires_grad=False, called only under
no_grad. Real weights load lazily via HuggingFace transformers or torch.hub. If weights
cannot be fetched (this session: no download), we fall back to a FROZEN deterministic
random projection so the cache pipeline and everything downstream still run end to end.
The fallback is honest: it is a real frozen encoder, just an untrained one, and the store
records `backend` so no one mistakes synthetic-substrate latents for V-JEPA latents.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
from torch import nn

from ..seeding import seed_everything


@dataclass
class EncoderSpec:
    name: str
    embed_dim: int
    dense: bool
    pool: str  # mean | cls | none
    frozen: bool = True
    backend: str = "frozen_random"  # vjepa_hf | vjepa_hub | frozen_random


class FrozenEncoder(nn.Module):
    """Wraps a frozen feature extractor. encode(clips)->latents. Never trains."""

    def __init__(self, spec: EncoderSpec, model: nn.Module | None = None):
        super().__init__()
        self.spec = spec
        self._model = model
        # frozen-random projection built lazily on first encode (we need the flat in-dim).
        # A plain frozen tensor (not a Parameter): never trains, recreated on device change.
        self._W: torch.Tensor | None = None
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: [B, C, T, H, W] (or [B, ...]) -> pooled [B, D] or dense [B, N, D]."""
        if self._model is not None:
            # V-JEPA 2 takes pixel_values_videos=[B,T,C,H,W]; other HF models take positional.
            try:
                out = self._model(pixel_values_videos=clips)
            except TypeError:
                out = self._model(clips)
            feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        else:
            b = clips.shape[0]
            flat = clips.reshape(b, -1).float()
            if self._W is None or self._W.shape[0] != flat.shape[1] or self._W.device != flat.device:
                seed_everything(0)
                w = torch.randn(flat.shape[1], self.spec.embed_dim) / math.sqrt(flat.shape[1])
                self._W = w.to(flat.device)
            feats = flat @ self._W  # [B, D]
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(self.spec.embed_dim)
        if self.spec.dense:
            return feats if feats.dim() == 3 else feats.unsqueeze(1)
        if feats.dim() == 3:  # pool token dim
            feats = feats.mean(1) if self.spec.pool == "mean" else feats[:, 0]
        return feats

    def forward(self, clips: torch.Tensor) -> torch.Tensor:  # pragma: no cover - alias
        return self.encode(clips)


def load_encoder(cfg) -> FrozenEncoder:
    """Build the encoder from an encoder config. Tries real weights, falls back to frozen
    random. cfg has: name, embed_dim, dense, pool, hf_id, hub, frozen."""
    spec = EncoderSpec(name=cfg.name, embed_dim=int(cfg.embed_dim), dense=bool(cfg.dense), pool=str(cfg.pool))
    # Real V-JEPA weights are OPT-IN (prefer_real): they need correctly-shaped real video, so the
    # toy experiments and the test suite stay on the deterministic frozen-random substrate. The
    # real-encoder caching script sets prefer_real to fetch and run the actual weights.
    model = _try_real_weights(cfg) if bool(cfg.get("prefer_real", False)) else None
    if model is not None:
        spec.backend = "vjepa_hf"
    elif bool(cfg.get("prefer_real", False)):
        warnings.warn(
            f"V-JEPA weights for {cfg.name} requested but unavailable; using FROZEN RANDOM substrate.",
            stacklevel=2,
        )
    return FrozenEncoder(spec, model)


def _try_real_weights(cfg) -> nn.Module | None:
    """Lazy, best-effort. Never raises: any failure returns None -> frozen-random path."""
    try:  # pragma: no cover - needs network + weights, deferred this session
        from transformers import AutoModel

        m = AutoModel.from_pretrained(cfg.hf_id, trust_remote_code=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
    except Exception:
        return None
