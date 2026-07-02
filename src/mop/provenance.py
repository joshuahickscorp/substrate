"""Provenance: the one place that stamps what produced a run or a cache. Every run manifest and
every latent store records git SHA, package versions, device, seed, encoder id + backend, a
cache id, timestamps, and an honest result tag, so a number can always be traced to the exact
code, config, and substrate that made it. Nothing here is allowed to claim more than it knows:
the result tag is an explicit enum, not a vibe.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata

from .config import REPO_ROOT

# what a number was computed on, from most to least real
RESULT_TAGS = ("natural-video", "real-encoder", "structured-synthetic", "provisional")
_PKGS = ("torch", "numpy", "omegaconf", "faiss-cpu", "transformers", "torchvision", "matplotlib")


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return False


def package_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = "absent"
    return out


def validate_tag(tag: str) -> str:
    if tag not in RESULT_TAGS:
        raise ValueError(f"result_tag {tag!r} not in {RESULT_TAGS}")
    return tag


def cache_id(name: str, count: int, sample: bytes = b"") -> str:
    """A short content id for a cache: name + count + an optional content sample digest."""
    h = hashlib.sha256(f"{name}:{count}:".encode() + sample).hexdigest()
    return h[:16]


def provenance(
    *,
    seed: int,
    device: str,
    encoder_id: str = "",
    encoder_backend: str = "",
    result_tag: str = "provisional",
    cache: str = "",
    extra: dict | None = None,
) -> dict:
    """The provenance block embedded in a manifest or written beside a cache."""
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "seed": int(seed),
        "device": device,
        "encoder_id": encoder_id,
        "encoder_backend": encoder_backend,
        "cache_id": cache,
        "result_tag": validate_tag(result_tag),
        **(extra or {}),
    }
