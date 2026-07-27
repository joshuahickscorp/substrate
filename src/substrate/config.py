"""One strict configuration path and precedence policy."""

from __future__ import annotations

import hashlib
import json
import os

from substrate import evidence

CONFIG_PATH = evidence.ROOT / "configs" / "substrate" / "config.json"
ALLOWED = {"activation", "program", "data_root", "state_root"}


def load() -> dict:
    document = json.loads(CONFIG_PATH.read_text())
    unknown = sorted(set(document) - ALLOWED)
    if unknown:
        raise ValueError(f"unknown Substrate configuration fields: {unknown}")
    configured = {
        **document,
        "data_root": os.environ.get("SUBSTRATE_DATA_ROOT", document.get("data_root", "")),
        "state_root": os.environ.get("SUBSTRATE_STATE_ROOT", document.get("state_root", "")),
    }
    if configured.get("activation") is not False:
        raise ValueError("activation must remain false")
    normalized = json.dumps(configured, sort_keys=True, separators=(",", ":"))
    return {**configured, "sha256": hashlib.sha256(normalized.encode()).hexdigest()}
