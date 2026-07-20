
import pytest
from omegaconf import OmegaConf

from mop import config
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
    cfg = config.compose(["encoder=vjepa21_vitl", "encoder.prefer_real=true"])
    with pytest.raises(validate.ConfigError):
        validate.validate_encoder(cfg)


def test_unavailable_encoder_without_prefer_real_ok():
    cfg = config.compose(["encoder=vjepa21_vitl"])  # prefer_real defaults false -> frozen-random, fine
    validate.validate_encoder(cfg)


def test_validate_leg_flags_problems():
    bad = {
        "tier": "Z",
        "experiment": "nope",
        "axes": {"k": [1]},
    }  # bad tier, no full grid, toy knob not in full
    probs = validate.validate_leg(bad, known_experiments={"e1_baseline"})
    assert any("tier" in p for p in probs)
    assert any("full_axes" in p for p in probs)
    assert any("unknown experiment" in p for p in probs)


def test_good_leg_passes():
    good = {
        "tier": "C",
        "experiment": "e1_baseline",
        "axes": {"k": [1]},
        "full_axes": {"k": [1, 2]},
        "full_seeds": [0, 1],
    }
    assert validate.validate_leg(good, known_experiments={"e1_baseline"}) == []
