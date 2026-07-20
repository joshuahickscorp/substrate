from __future__ import annotations

import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import REPO_ROOT
from .evidence import atomic_write_json
from .provenance import git_sha, package_versions

_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str = "mop", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)  # logs to stderr; data (JSON) stays clean on stdout
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(level)
        log.propagate = False
    return log


@dataclass
class RunManifest:
    name: str
    seed: int
    device: str
    git: str = field(default_factory=git_sha)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    encoder_id: str = ""
    encoder_backend: str = ""
    cache_id: str = ""
    result_tag: str = "provisional"
    packages: dict = field(default_factory=package_versions)

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / "manifest.json"
        atomic_write_json(p, asdict(self))
        return p


def new_run_dir(name: str, root: Path | None = None) -> Path:
    base = (root or REPO_ROOT / "runs") / name
    base.mkdir(parents=True, exist_ok=True)
    numeric = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    n = max(numeric, default=-1) + 1
    while True:
        d = base / f"{n:03d}"
        try:
            d.mkdir(parents=False, exist_ok=False)
            return d
        except FileExistsError:
            n += 1
