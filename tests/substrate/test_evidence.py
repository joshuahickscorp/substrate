"""Legacy evidence hashing stays byte-identical across encoder paths."""

from __future__ import annotations

import hashlib
import json

import pytest

from substrate import evidence


def test_legacy_sha_obj_fast_encoder_preserves_reference_bytes() -> None:
    value = {
        "unicode": "naïve café",
        "tuple": (1, 2.0),
        "nan": float("nan"),
        "nested": {"z": [True, None], "a": {"fallback": object()}},
    }
    expected = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    assert evidence.sha_obj(value) == expected


def test_legacy_sha_obj_cycle_still_uses_reference_refusal() -> None:
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="Circular reference detected"):
        evidence.sha_obj(cyclic)
