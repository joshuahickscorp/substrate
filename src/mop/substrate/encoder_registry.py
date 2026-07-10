"""Encoder registry: honesty over the shipped encoder configs (Frontier 15).

A pure-config registry. It reads configs/encoder/*.yaml and answers two questions without
ever touching the network: which encoders exist, and is each one HONEST about whether it can
load real V-JEPA weights. The failure mode this guards against is a config that quietly
pretends to be real (asks for real weights it cannot fetch, or claims availability it has not
earned) and lets frozen-random latents masquerade as V-JEPA latents downstream.

The contract, mirrored by validate.validate_encoder:
  available  (default True if the key is absent) means weights are believed present on HF.
  prefer_real means "load the real weights" (load_encoder opt-in). An unavailable encoder with
  prefer_real set is dishonest (and validate blocks it). Availability is asserted, not probed:
  VERIFIED_REAL_IDS is the documented set of hf_ids known-present on HF, checked once by hand,
  recorded here so nothing in the hot path hits the network.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from ..config import REPO_ROOT

# The one live Hugging Face control ID, verified in 2026-06. Official dense ViT-B uses its pinned
# PyTorch checkpoint seam instead, so it is intentionally absent from this HF-only set. Retired
# inherited-scale configs live outside the composable encoder group and are never enumerated here.
VERIFIED_REAL_IDS: frozenset[str] = frozenset({"facebook/vjepa2-vitl-fpc64-256"})

_ENCODER_DIR = REPO_ROOT / "configs" / "encoder"


def list_encoders() -> list[dict]:
    """Every shipped encoder config as a flat dict, sorted by name. Pure config read, no network.

    Each entry: {name, hf_id, embed_dim, dense, available, prefer_real, family, training_objective}.
    `available` defaults to True when the key is absent (the shipped vjepa21_* configs set it false);
    `prefer_real` defaults to False (real weights are opt-in via load_encoder). `family` and
    `training_objective` (WP-02) type each substrate for the cross-substrate grid (AT1/AL2): a
    substrate-specialness verdict must state WHICH family and objective it is about, so both keys
    travel with every listing and default to "unknown" rather than being silently absent."""
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
    """The documented set of hf_ids known-present on HF (the 3 verified V-JEPA 2 weights).
    Returns the constant; never hits the network."""
    return VERIFIED_REAL_IDS


def is_honest(enc: dict) -> bool:
    """An encoder config is honest if it cannot silently pretend to be real.

    Honest iff EITHER it is available (claims real weights it can fetch) OR it is unavailable
    but does not ask for real weights (available False AND prefer_real False, so it stays on the
    frozen-random substrate). The remaining quadrant (unavailable AND prefer_real) is the lie:
    an encoder that would try to load real weights it cannot fetch. (validate also blocks it.)"""
    available = bool(enc.get("available", True))
    prefer_real = bool(enc.get("prefer_real", False))
    return available or not prefer_real
