import pytest
from omegaconf import OmegaConf

from mop.harness import validate


def test_check_all_clean_on_real_repo():
    assert validate.check_all() == []


def test_bad_device_kind_raises():
    cfg = OmegaConf.create({"device": {"kind": "tpu"}, "experiment": {"id": "x", "null_hypothesis": "n"}})
    with pytest.raises(validate.ConfigError):
        validate.validate_device(cfg)


def test_missing_null_hypothesis_raises():
    cfg = OmegaConf.create({"experiment": {"id": "x", "null_hypothesis": ""}})
    with pytest.raises(validate.ConfigError):
        validate.validate_experiment(cfg)


@pytest.mark.parametrize("field", ["strong_null", "null"])
def test_sealed_envelope_payload_declares_null_contract(field):
    cfg = OmegaConf.create({"payload": {field: "registered null"}})
    assert validate._declared_null_contract(cfg) == "registered null"


def test_unavailable_encoder_with_prefer_real_raises():
    cfg = OmegaConf.create(
        {"encoder": {"name": "unavailable", "embed_dim": 8, "available": False, "prefer_real": True}}
    )
    with pytest.raises(validate.ConfigError):
        validate.validate_encoder(cfg)


def test_unavailable_encoder_without_prefer_real_ok():
    cfg = OmegaConf.create(
        {"encoder": {"name": "unavailable", "embed_dim": 8, "available": False, "prefer_real": False}}
    )
    validate.validate_encoder(cfg)
