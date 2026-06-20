"""Logging + run manifests. Every run gets a directory under runs/ holding the
resolved config snapshot, a manifest (seed, device, git, timing), and any metrics
or plots. No silent failures: errors are logged and surfaced, never swallowed.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import REPO_ROOT

_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str = "devsys", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)  # logs to stderr; data (JSON) stays clean on stdout
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(level)
        log.propagate = False
    return log


def _git_rev() -> str:
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


@dataclass
class RunManifest:
    name: str
    seed: int
    device: str
    git: str = field(default_factory=_git_rev)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / "manifest.json"
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p


def new_run_dir(name: str, root: Path | None = None) -> Path:
    """Create a unique run dir runs/<name>/<n>. No timestamp: deterministic across
    a session, monotonic per name, so reruns are easy to diff."""
    base = (root or REPO_ROOT / "runs") / name
    base.mkdir(parents=True, exist_ok=True)
    n = sum(1 for p in base.iterdir() if p.is_dir())
    d = base / f"{n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d
