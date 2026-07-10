"""The frozen perceptual substrate. V-JEPA loaded once, never trained, inference-only.

This is the ONE large object and it is held fixed: requires_grad=False, called only under
no_grad. Real weights load lazily via HuggingFace transformers. If weights cannot be loaded,
ordinary toy runs may fall back to a FROZEN deterministic random projection so mechanics still
run end to end. Scientific and cache-building callers set ``require_real`` and fail closed instead.
Every store records ``backend`` so no one can mistake synthetic-substrate latents for V-JEPA latents.
"""

from __future__ import annotations

import hashlib
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

    @property
    def model_class_name(self) -> str:
        """Fully qualified wrapped-model class, or the explicit projection fallback identity."""
        if self._model is None:
            return "mop.substrate.encoder.FrozenEncoderProjection"
        cls = type(self._model)
        return f"{cls.__module__}.{cls.__qualname__}"


def load_encoder(cfg) -> FrozenEncoder:
    """Build the encoder from an encoder config. Tries real weights, falls back to frozen
    random. cfg has: name, embed_dim, dense, pool, hf_id, hub, frozen."""
    spec = EncoderSpec(name=cfg.name, embed_dim=int(cfg.embed_dim), dense=bool(cfg.dense), pool=str(cfg.pool))
    if bool(cfg.get("random_init", False)):
        if bool(cfg.get("prefer_real", False)) or bool(cfg.get("require_real", False)):
            raise ValueError("random_init cannot be combined with prefer_real or require_real")
        if not str(cfg.get("revision", "")).strip():
            raise ValueError("random_init requires a pinned model revision")
        model = _try_random_architecture(cfg)
        if model is None:
            raise RuntimeError(
                f"random architecture for {cfg.name} could not be constructed from its pinned config"
            )
        spec.backend = "vjepa_hf_random_init"
        return FrozenEncoder(spec, model)

    # Real V-JEPA weights are OPT-IN (prefer_real): they need correctly-shaped real video, so the
    # toy experiments and the test suite stay on the deterministic frozen-random substrate. The
    # real-encoder caching script sets prefer_real to fetch and run the actual weights.
    model = _try_real_weights(cfg) if bool(cfg.get("prefer_real", False)) else None
    if model is not None:
        spec.backend = "vjepa_hf"
    elif bool(cfg.get("prefer_real", False)):
        if bool(cfg.get("require_real", False)):
            raise RuntimeError(
                f"real weights for {cfg.name} were required but could not be loaded; "
                "refusing an unrequested frozen-random fallback"
            )
        warnings.warn(
            f"V-JEPA weights for {cfg.name} requested but unavailable; using FROZEN RANDOM substrate.",
            stacklevel=2,
        )
    return FrozenEncoder(spec, model)


def _try_random_architecture(cfg) -> nn.Module | None:
    """Build the exact pinned HF architecture with seeded random weights, without loading a shard."""
    try:  # pragma: no cover - full construction is covered by supervised scale runs
        from transformers import AutoConfig, AutoModel

        kwargs: dict[str, object] = {"trust_remote_code": True}
        revision = str(cfg.get("revision", "")).strip()
        if revision:
            kwargs["revision"] = revision
        if bool(cfg.get("local_files_only", False)):
            kwargs["local_files_only"] = True
        architecture = AutoConfig.from_pretrained(cfg.hf_id, **kwargs)
        hidden_size = getattr(architecture, "hidden_size", None)
        if hidden_size is not None and int(hidden_size) != int(cfg.embed_dim):
            raise ValueError(
                f"pinned architecture hidden_size={hidden_size} does not match embed_dim={cfg.embed_dim}"
            )
        seed_everything(int(cfg.get("random_init_seed", 0)))
        model = AutoModel.from_config(architecture, trust_remote_code=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
    except Exception:
        return None


def module_state_sha256(module: nn.Module, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Hash the exact initialized state without serializing a multi-gigabyte temporary checkpoint.

    Names, dtypes, shapes, and raw tensor bytes are included in sorted-key order. This is used for
    random-architecture controls, where a config hash plus a claimed seed is not enough to identify
    the weights that actually produced a cache.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    state = module.state_dict()
    if not state:
        raise ValueError("cannot fingerprint a module with an empty state_dict")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        raw = value.view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), chunk_bytes):
            digest.update(memoryview(raw[start : start + chunk_bytes].numpy()))
    return digest.hexdigest()


def _try_real_weights(cfg) -> nn.Module | None:
    """Lazy, best-effort. Never raises: any failure returns None -> frozen-random path."""
    try:  # pragma: no cover - needs network + weights, deferred this session
        from transformers import AutoModel

        kwargs: dict[str, object] = {"trust_remote_code": True}
        revision = str(cfg.get("revision", "")).strip()
        if revision:
            kwargs["revision"] = revision
        if bool(cfg.get("local_files_only", False)):
            kwargs["local_files_only"] = True
        m = AutoModel.from_pretrained(cfg.hf_id, **kwargs)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
    except Exception:
        return None
