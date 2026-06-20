"""Frozen perceptual substrate: encoder (inference-only), latent cache, memmap store,
streams. Nothing here trains."""

from __future__ import annotations

from .cache import cache_latents, synthetic_clips
from .datasets import Task, make_task_stream, noisy_tv_dataset, stream_from_store
from .encoder import EncoderSpec, FrozenEncoder, load_encoder
from .latent_store import LatentStore, StoreMeta
from .video import iter_video_clips, preprocess_clip, read_video

__all__ = [
    "EncoderSpec",
    "FrozenEncoder",
    "load_encoder",
    "LatentStore",
    "StoreMeta",
    "cache_latents",
    "synthetic_clips",
    "Task",
    "make_task_stream",
    "noisy_tv_dataset",
    "stream_from_store",
    "preprocess_clip",
    "read_video",
    "iter_video_clips",
]
