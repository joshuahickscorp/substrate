from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random

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
