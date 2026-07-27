from __future__ import annotations

import subprocess
import sys
from importlib import metadata

from .config import REPO_ROOT

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


def package_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = "absent"
    return out
