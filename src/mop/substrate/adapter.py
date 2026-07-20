
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .encoder import EncoderSpec, FrozenEncoder


@dataclass(frozen=True)
class SubstrateMeta:

    tag: str
    embed_dim: int
    input_resolution: int
    frames: int
    pretrained: bool
    family: str = "unknown"
    notes: str = ""


class SubstrateAdapter(ABC):

    meta: SubstrateMeta

    @property
    def tag(self) -> str:
        return self.meta.tag

    @abstractmethod
    def extract(self, clips: torch.Tensor) -> torch.Tensor:
        pass

    def extract_batched(self, clips: torch.Tensor, batch: int = 8) -> torch.Tensor:
        with torch.no_grad():
            outs = [self.extract(clips[i : i + batch]) for i in range(0, clips.shape[0], batch)]
        return torch.cat(outs, dim=0)


class RealEncoderAdapter(SubstrateAdapter):

    def __init__(self, encoder: FrozenEncoder, *, input_resolution: int, frames: int, notes: str = ""):
        self.encoder = encoder
        self.meta = SubstrateMeta(
            tag=str(encoder.spec.name),
            embed_dim=int(encoder.spec.embed_dim),
            input_resolution=int(input_resolution),
            frames=int(frames),
            pretrained=True,
            family="real_encoder",
            notes=notes,
        )

    def extract(self, clips: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder.encode(clips)


class RandomInitViTAdapter(SubstrateAdapter):

    def __init__(self, encoder_cfg, *, input_resolution: int, frames: int, seed: int = 0):
        self.encoder_cfg = encoder_cfg
        self.seed = int(seed)
        self._encoder: FrozenEncoder | None = None
        self.meta = SubstrateMeta(
            tag=str(encoder_cfg.name) + "_randinit",
            embed_dim=int(encoder_cfg.embed_dim),
            input_resolution=int(input_resolution),
            frames=int(frames),
            pretrained=False,
            family="random_init_vit",
        )

    def _build(self) -> FrozenEncoder:
        from transformers import AutoConfig, AutoModel

        c = AutoConfig.from_pretrained(self.encoder_cfg.hf_id, trust_remote_code=True)
        torch.manual_seed(self.seed)  # deterministic untrained weights per (cfg, seed)
        with torch.no_grad():
            m = AutoModel.from_config(c, trust_remote_code=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        spec = EncoderSpec(
            name=self.meta.tag,
            embed_dim=self.meta.embed_dim,
            dense=bool(getattr(self.encoder_cfg, "dense", False)),
            pool=str(getattr(self.encoder_cfg, "pool", "mean")),
            backend="random_init_vit",
        )
        return FrozenEncoder(spec, m)

    def extract(self, clips: torch.Tensor) -> torch.Tensor:
        if self._encoder is None:
            self._encoder = self._build()
        with torch.no_grad():
            return self._encoder.encode(clips)


class RandomPixelAdapter(SubstrateAdapter):

    def __init__(self, embed_dim: int, *, ds: int = 32, tsub: int = 8, seed: int = 0):
        self.ds = int(ds)
        self.tsub = int(tsub)
        self.seed = int(seed)
        self._proj: torch.Tensor | None = None
        self.meta = SubstrateMeta(
            tag=f"random_pixel_d{ds}_t{tsub}",
            embed_dim=int(embed_dim),
            input_resolution=int(ds),
            frames=int(tsub),
            pretrained=False,
            family="random_pixel",
        )

    def _projection(self, in_dim: int) -> torch.Tensor:
        if self._proj is None or self._proj.shape[0] != in_dim:
            g = torch.Generator().manual_seed(self.seed)
            self._proj = torch.randn(in_dim, self.meta.embed_dim, generator=g) / math.sqrt(in_dim)
        return self._proj

    def extract(self, clips: torch.Tensor) -> torch.Tensor:
        n, t = clips.shape[0], clips.shape[1]
        stride = max(t // self.tsub, 1)
        c = clips[:, ::stride][:, : self.tsub]  # [N, tsub, 3, H, W]
        k = max(c.shape[-1] // self.ds, 1)
        c = F.avg_pool2d(c.reshape(-1, *c.shape[2:]), kernel_size=k)  # [N*tsub, 3, ds, ds]
        flat = c.reshape(n, -1)
        z = flat @ self._projection(flat.shape[1])
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(self.meta.embed_dim)


@dataclass
class SubstrateRegistry:

    adapters: dict[str, SubstrateAdapter] = field(default_factory=dict)

    def register(self, adapter: SubstrateAdapter) -> None:
        if adapter.tag in self.adapters:
            raise ValueError(f"duplicate substrate tag: {adapter.tag}")
        self.adapters[adapter.tag] = adapter

    def tags(self) -> list[str]:
        return sorted(self.adapters)

    def extract_all(self, clips: torch.Tensor, batch: int = 8) -> dict[str, torch.Tensor]:
        return {t: self.adapters[t].extract_batched(clips, batch=batch) for t in self.tags()}
