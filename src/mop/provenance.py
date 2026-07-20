
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata

from .config import REPO_ROOT

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
