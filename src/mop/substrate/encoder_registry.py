
from __future__ import annotations

from omegaconf import OmegaConf

from ..config import REPO_ROOT

VERIFIED_REAL_IDS: frozenset[str] = frozenset({"facebook/vjepa2-vitl-fpc64-256"})

_ENCODER_DIR = REPO_ROOT / "configs" / "encoder"


def list_encoders() -> list[dict]:
    out: list[dict] = []
    for f in sorted(_ENCODER_DIR.glob("*.yaml")):
        c = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
        assert isinstance(c, dict)
        out.append(
            {
                "name": str(c.get("name", f.stem)),
                "hf_id": str(c.get("hf_id", "")),
                "embed_dim": int(c.get("embed_dim", 0)),
                "dense": bool(c.get("dense", False)),
                "available": bool(c.get("available", True)),
                "prefer_real": bool(c.get("prefer_real", False)),
                "family": str(c.get("family", "unknown")),
                "training_objective": str(c.get("training_objective", "unknown")),
            }
        )
    return out


def verified_real_ids() -> frozenset[str]:
    return VERIFIED_REAL_IDS


def is_honest(enc: dict) -> bool:
    available = bool(enc.get("available", True))
    prefer_real = bool(enc.get("prefer_real", False))
    return available or not prefer_real
