from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import nn

from mop.evidence import canonical_bytes


@dataclass(frozen=True)
class ModelSpec:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    patch_size: int = 32
    tubelet: int = 2
    max_resolution: int = 256
    max_frames: int = 16

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelSpec:
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("model spec fields do not match the portable architecture")
        if any(not isinstance(value[key], int) or isinstance(value[key], bool) for key in expected):
            raise ValueError("model spec fields must be integers")
        spec = cls(**{key: int(value[key]) for key in expected})
        spec.validate()
        return spec

    def validate(self) -> None:
        if min(self.dim, self.depth, self.heads, self.mlp_ratio) <= 0:
            raise ValueError("model widths must be positive")
        if self.dim % self.heads:
            raise ValueError("model dim must be divisible by heads")
        if self.patch_size <= 0 or self.tubelet <= 0:
            raise ValueError("patch_size and tubelet must be positive")
        if self.max_resolution <= 0 or self.max_frames <= 0:
            raise ValueError("maximum input geometry must be positive")
        if self.max_resolution % self.patch_size or self.max_frames % self.tubelet:
            raise ValueError("maximum input geometry must divide exactly into patches")

    @property
    def max_tokens(self) -> int:
        return (self.max_frames // self.tubelet) * (self.max_resolution // self.patch_size) ** 2


class SubstrateOutput(NamedTuple):
    dense_spatiotemporal_tokens: torch.Tensor
    pooled_retrieval_key: torch.Tensor


class TinyVideoSubstrate(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.patch_embed = nn.Conv3d(
            3,
            spec.dim,
            kernel_size=(spec.tubelet, spec.patch_size, spec.patch_size),
            stride=(spec.tubelet, spec.patch_size, spec.patch_size),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, spec.dim))
        self.position = nn.Parameter(torch.zeros(1, spec.max_tokens, spec.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=spec.dim,
            nhead=spec.heads,
            dim_feedforward=spec.dim * spec.mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=spec.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(spec.dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(spec.dim),
            nn.Linear(spec.dim, spec.dim),
            nn.GELU(),
            nn.Linear(spec.dim, spec.dim),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, clips: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError("clips must be [batch,3,time,height,width]")
        frames, height, width = map(int, clips.shape[2:])
        if min(frames, height, width) <= 0:
            raise ValueError("clip geometry must be positive")
        if frames > self.spec.max_frames or max(height, width) > self.spec.max_resolution:
            raise ValueError("clip geometry exceeds the model maxima")
        if frames % self.spec.tubelet or height % self.spec.patch_size or width % self.spec.patch_size:
            raise ValueError("clip geometry must divide exactly into tubelets and patches")
        patches = self.patch_embed(clips).flatten(2).transpose(1, 2)
        hidden = patches + self.position[:, : patches.shape[1]]
        if mask is not None:
            if mask.dtype is not torch.bool:
                raise ValueError("token mask must have bool dtype")
            if tuple(mask.shape) != tuple(hidden.shape[:2]):
                raise ValueError("token mask shape does not match token geometry")
            hidden = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(hidden), hidden)
        return self.norm(self.blocks(hidden))

    def forward(self, clips: torch.Tensor, mask: torch.Tensor | None = None) -> SubstrateOutput:
        dense = self.encode(clips, mask)
        return SubstrateOutput(dense, dense.mean(dim=1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_bytes(list(tensor.shape)))
        array = tensor.numpy()
        digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C"))
    return digest.hexdigest()
