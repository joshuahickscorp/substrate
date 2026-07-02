"""Latent caching pipeline. Run the frozen encoder over a clip source ONCE, write latents
to a memmap LatentStore. After this, the whole shell trains on the cache and never touches
the encoder again (the laptop-feasibility keystone).

No real video this session, so `synthetic_clips` provides a deterministic clip source: the
round-trip (clips -> frozen encoder -> store -> read) is fully exercised and tested.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path

import torch

from ..devices import DeviceInfo, safe_to
from ..logging_utils import get_logger
from .encoder import FrozenEncoder
from .latent_store import LatentStore

log = get_logger("substrate.cache")


def synthetic_clips(
    n: int, batch: int, shape: tuple[int, ...] = (3, 4, 16, 16), n_classes: int = 4, seed: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic (clips, labels) batches. Class-conditioned mean so the frozen-random
    encoder still yields linearly-separable-ish latents (enough to develop against)."""
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(n_classes, *shape, generator=g)
    done = 0
    while done < n:
        b = min(batch, n - done)
        y = torch.randint(0, n_classes, (b,), generator=g)
        x = proto[y] + 0.3 * torch.randn(b, *shape, generator=g)
        yield x, y
        done += b


def cache_latents(
    encoder: FrozenEncoder,
    clips: Iterator[tuple[torch.Tensor, torch.Tensor]],
    root: Path,
    name: str,
    total: int,
    device: DeviceInfo,
    has_labels: bool = True,
    result_tag: str | None = None,
    seed: int = 0,
) -> LatentStore:
    """Encode a clip stream into a fresh LatentStore. Returns the finalized store. Writes a
    provenance.json (git, packages, encoder id+backend, result tag, cache id) beside the store."""
    encoder = encoder.to(device.device)
    it = iter(clips)
    first = next(it)
    clips = itertools.chain([first], it)  # restore the peeked item
    feat = encoder.encode(safe_to(first[0][:1], device.device)).reshape(1, -1)
    feat_shape = (feat.shape[1],)
    store = LatentStore.create(
        root, name, feat_shape, capacity=total, key_dim=feat_shape[0], has_labels=has_labels
    )
    pos = 0
    for x, y in clips:
        if pos >= total:
            break
        x = safe_to(x, device.device)
        z = encoder.encode(x).reshape(x.shape[0], -1).cpu()
        z = z[: total - pos]
        y = y[: z.shape[0]]
        store.write_batch(pos, z.numpy(), z.numpy(), y.numpy() if has_labels else None)
        pos += z.shape[0]
    store.finalize()
    _write_provenance(store, encoder, result_tag, seed, device)
    log.info(
        "cached %d latents dim=%d backend=%s -> %s",
        len(store),
        feat_shape[0],
        encoder.spec.backend,
        store.root,
    )
    return store


def _write_provenance(store, encoder, result_tag, seed, device) -> None:
    import json

    from ..provenance import cache_id, provenance

    backend = encoder.spec.backend
    tag = result_tag or ("real-encoder" if backend == "vjepa_hf" else "provisional")
    sample = store.latents(slice(0, min(4, len(store)))).numpy().tobytes()[:4096] if len(store) else b""
    prov = provenance(
        seed=seed,
        device=device.kind,
        encoder_id=encoder.spec.name,
        encoder_backend=backend,
        result_tag=tag,
        cache=cache_id(store.meta.name, len(store), sample),
        extra={"count": len(store), "feat_shape": store.meta.feat_shape, "dtype": store.meta.dtype},
    )
    (store.root / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
