
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

UINT32_MAX = (1 << 32) - 1


def derive_seed32(seed: int, namespace: str) -> int:
    seed = int(seed)
    if 0 <= seed <= UINT32_MAX:
        return seed
    payload = json.dumps(
        {"namespace": str(namespace), "seed": seed},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(b"mop-seed32-v1\0" + payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def seed_everything(seed: int, deterministic: bool = True) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(Exception):  # not all backends support this
            torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def make_generator(seed: int, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device="cpu")  # mps has no generator; seed on cpu, move tensors after
    g.manual_seed(seed)
    return g


@dataclass(frozen=True)
class VarianceReport:
    runs: int
    max_abs: float
    mean_abs: float
    bit_identical: bool

    def tol(self, floor: float = 1e-6) -> float:
        return max(floor, self.max_abs * 4.0)


def variance_of(fn: Callable[[], torch.Tensor], runs: int = 5, seed: int = 0) -> VarianceReport:
    outs = []
    for _ in range(runs):
        seed_everything(seed)
        outs.append(fn().detach().float().cpu().reshape(-1))
    base = outs[0]
    diffs = torch.stack([(o - base).abs() for o in outs[1:]]) if runs > 1 else torch.zeros(1)
    max_abs = float(diffs.max()) if diffs.numel() else 0.0
    mean_abs = float(diffs.mean()) if diffs.numel() else 0.0
    return VarianceReport(runs=runs, max_abs=max_abs, mean_abs=mean_abs, bit_identical=max_abs == 0.0)


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]
