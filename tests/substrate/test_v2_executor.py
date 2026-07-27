from __future__ import annotations

import copy

import pytest

from substrate import v2config as C
from substrate import v2executor as X


def test_frozen_configuration_matches_executable_authority():
    frozen = X.frozen_configuration()
    assert frozen["configuration_digest"] == C.configuration()["configuration_digest"]
    assert frozen["activation"] is False


def test_context_refuses_source_configuration_split_seed_and_activation():
    context = X.context()
    assert X.validate_context(context, split="development", seed=C.SPLITS["development"][0]) == context
    mutations = (
        ({**context, "source_digest": "0" * 64}, "development", C.SPLITS["development"][0]),
        ({**context, "configuration_digest": "0" * 64}, "development", C.SPLITS["development"][0]),
        ({**context, "split_digest": "0" * 64}, "development", C.SPLITS["development"][0]),
        ({**context, "activation": True}, "development", C.SPLITS["development"][0]),
        (context, "unknown", C.SPLITS["development"][0]),
        (context, "development", C.SPLITS["principal"][0]),
    )
    for supplied, split, seed in mutations:
        with pytest.raises(X.Refused):
            X.validate_context(supplied, split=split, seed=seed)


def test_receipt_identity_detects_tampering():
    document = X.receipt_body("unit", {"value": 1}, X.context())
    assert X.validate_receipt(document)
    tampered = copy.deepcopy(document)
    tampered["payload"]["value"] = 2
    assert not X.validate_receipt(tampered)
