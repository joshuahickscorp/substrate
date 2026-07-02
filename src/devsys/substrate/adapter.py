"""SubstrateAdapter (WP-02): present ANY substrate behind ONE `extract(clips) -> [N, D]` interface so
the substrate-is-special comparison is a swap of adapter, not a rewrite. This is the gated-NEW module
from 13_code_scaffolding.md: it promotes the corrected control (real-encoder features vs random-ENCODER
or random-init-ViT features, NEVER a within-latent square projection, the vacuous-control finding) from
scripts into reusable infrastructure. Each adapter carries honest resolution metadata so the
256px-vs-32px confound is machine-readable, not a comment in a script.

Doctrine: the vacuous-control discovery means substrate-specialness claims are answered ONLY by
comparing adapters here, never via `substrate_ablation.frozen_random_projection`.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .encoder import EncoderSpec, FrozenEncoder


@dataclass(frozen=True)
class SubstrateMeta:
    """Honest-confound metadata for one substrate arm. `input_resolution` and `frames` record what the
    adapter actually consumes (e.g. real V-JEPA at 256px x 64f vs the pixel control at 32px x 8f), so
    any resolution mismatch travels with the result instead of living in a script comment."""

    tag: str
    embed_dim: int
    input_resolution: int
    frames: int
    pretrained: bool
    family: str = "unknown"
    notes: str = ""


class SubstrateAdapter(ABC):
    """ABC: one substrate = one feature extractor. Subclasses set `self.meta` in __init__."""

    meta: SubstrateMeta

    @property
    def tag(self) -> str:
        """Short unique name used as the key in cross-substrate dicts."""
        return self.meta.tag

    @abstractmethod
    def extract(self, clips: torch.Tensor) -> torch.Tensor:
        """Map clips [N, T, 3, H, W] to features [N, D]. Deterministic given construction, no grad."""

    def extract_batched(self, clips: torch.Tensor, batch: int = 8) -> torch.Tensor:
        """Convenience: run `extract` in mini-batches and concatenate (memory-bounded encodes)."""
        with torch.no_grad():
            outs = [self.extract(clips[i : i + batch]) for i in range(0, clips.shape[0], batch)]
        return torch.cat(outs, dim=0)


class RealEncoderAdapter(SubstrateAdapter):
    """The pretrained arm: wraps an already-constructed FrozenEncoder (the caller loads it, this module
    never triggers a model load itself, per the live-encode constraint)."""

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
    """The corrected control: an identical-architecture ViT with RANDOM (untrained) weights, the arm
    that isolates PRETRAINING from architecture and resolution (the p=0.029 gold-result control). The
    build (AutoConfig + AutoModel.from_config: the KB-scale config file, never the weights) is deferred
    to first `extract` so constructing an adapter never loads a model (live-encode constraint). The
    encode/pooling path is FrozenEncoder.encode, identical to the real arm; init is seeded so the same
    (cfg, seed) always yields the same untrained weights."""

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
        """Same-arch untrained ViT, lifted from scripts/substrate_vs_random_init_vit.py: from_config
        (architecture only, no weight download), frozen, wrapped so .encode pools like the real arm."""
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
    """The random-FEATURES-of-pixels control: avg-pool space to `ds` px, subsample time to `tsub`
    frames, flatten, apply one FIXED seeded random projection to `embed_dim`, renormalize. Pure and
    tiny; the honest 'does pretraining help' floor. Resolution metadata records `ds`, not the source."""

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
    """Holds constructed adapters by tag so AlignmentSuite / cross_substrate iterate generically."""

    adapters: dict[str, SubstrateAdapter] = field(default_factory=dict)

    def register(self, adapter: SubstrateAdapter) -> None:
        """Add one adapter; duplicate tags are an error (two arms must never alias)."""
        if adapter.tag in self.adapters:
            raise ValueError(f"duplicate substrate tag: {adapter.tag}")
        self.adapters[adapter.tag] = adapter

    def tags(self) -> list[str]:
        return sorted(self.adapters)

    def extract_all(self, clips: torch.Tensor, batch: int = 8) -> dict[str, torch.Tensor]:
        """tag -> [N, D] features for every registered substrate (sequential, never two encoders at
        once, per the live-encode constraint)."""
        return {t: self.adapters[t].extract_batched(clips, batch=batch) for t in self.tags()}
