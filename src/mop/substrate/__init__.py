from __future__ import annotations

from .cache import cache_latents, synthetic_clips
from .cache_manifest import validate_cache_manifest, write_cache_manifest
from .datasets import Task, make_task_stream, noisy_tv_dataset, stream_from_store
from .encoder import EncoderSpec, FrozenEncoder, load_encoder
from .fixtures import generate_video_corpus
from .form import (
    FORM_SCHEMA,
    FormAdapter,
    FormBatch,
    FormMatrix,
    FormMeta,
    FormRegistry,
    LatentStoreFormAdapter,
    SubstrateFormAdapter,
    TensorFormAdapter,
    apply_affine_alignment,
    build_form_matrix,
    fit_affine_alignment,
    form_audit,
)
from .latent_store import LatentStore, StoreMeta
from .real_latent import factorized_arrays, factors_meta, open_real_store, real_task_stream
from .video import iter_video_clips, preprocess_clip, read_video

__all__ = [
    "EncoderSpec",
    "FrozenEncoder",
    "load_encoder",
    "LatentStore",
    "StoreMeta",
    "cache_latents",
    "synthetic_clips",
    "write_cache_manifest",
    "validate_cache_manifest",
    "Task",
    "make_task_stream",
    "noisy_tv_dataset",
    "stream_from_store",
    "real_task_stream",
    "factorized_arrays",
    "factors_meta",
    "open_real_store",
    "preprocess_clip",
    "read_video",
    "iter_video_clips",
    "generate_video_corpus",
    "FORM_SCHEMA",
    "FormMeta",
    "FormBatch",
    "FormMatrix",
    "FormAdapter",
    "FormRegistry",
    "TensorFormAdapter",
    "LatentStoreFormAdapter",
    "SubstrateFormAdapter",
    "build_form_matrix",
    "form_audit",
    "fit_affine_alignment",
    "apply_affine_alignment",
]
