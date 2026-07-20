
import pytest
from omegaconf import OmegaConf

from mop.config import REPO_ROOT, compose
from mop.harness.validate import ConfigError, validate_encoder
from mop.substrate import load_encoder
from mop.substrate.encoder_registry import (
    VERIFIED_REAL_IDS,
    is_honest,
    list_encoders,
    verified_real_ids,
)

_ENCODER_DIR = REPO_ROOT / "configs" / "encoder"


def _raw_configs():
    return {f.stem: OmegaConf.load(f) for f in sorted(_ENCODER_DIR.glob("*.yaml"))}


def test_list_encoders_covers_every_shipped_config():
    listed = {e["name"] for e in list_encoders()}
    on_disk = {OmegaConf.load(f).get("name", f.stem) for f in _ENCODER_DIR.glob("*.yaml")}
    assert listed == set(on_disk)
    assert listed  # not empty


def test_retired_scale_configs_are_archived_and_not_composable():
    archive = REPO_ROOT / "configs" / "legacy_encoder"
    retired = ("vjepa2_vith", "vjepa2_vitg")
    listed = {e["name"] for e in list_encoders()}
    for name in retired:
        assert (archive / f"{name}.yaml").is_file()
        assert not (_ENCODER_DIR / f"{name}.yaml").exists()
        assert name not in listed
        with pytest.raises(FileNotFoundError):
            compose([f"encoder={name}"])


def test_every_shipped_encoder_is_honest():
    for e in list_encoders():
        assert is_honest(e), f"{e['name']} is not honest: {e}"


def test_vjepa21_configs_distinguish_verified_vitb_from_larger_placeholders():
    by_name = {e["name"]: e for e in list_encoders()}
    v21 = [name for name in by_name if name.startswith("vjepa21_")]
    assert v21, "expected vjepa21_* configs to ship"
    for name in v21:
        if name == "vjepa21_vitb":
            assert by_name[name]["available"] is True
            assert OmegaConf.load(_ENCODER_DIR / "vjepa21_vitb.yaml").cache_first_only is True
        else:
            assert by_name[name]["available"] is False, f"{name} must carry available:false"
        assert by_name[name]["prefer_real"] is False


def test_verified_ids_match_non_21_config_hf_ids():
    raw = _raw_configs()
    non_21 = {str(c.get("hf_id")) for stem, c in raw.items() if not stem.startswith("vjepa21_")}
    assert verified_real_ids() == VERIFIED_REAL_IDS
    assert non_21 == set(VERIFIED_REAL_IDS)
    for stem, c in raw.items():
        if stem.startswith("vjepa21_"):
            assert str(c.get("hf_id")) not in VERIFIED_REAL_IDS


def test_unavailable_encoder_without_prefer_real_loads_frozen_random():
    raw = _raw_configs()
    unavailable = [c for stem, c in raw.items() if not bool(c.get("available", True))]
    assert unavailable, "expected at least one available:false config"
    for c in unavailable:
        cfg = OmegaConf.create(OmegaConf.to_container(c, resolve=True))
        cfg.prefer_real = False  # explicit: do not pretend to be real
        enc = load_encoder(cfg)
        assert enc.spec.backend == "frozen_random"


def test_unavailable_plus_prefer_real_is_dishonest_and_validate_blocks_it():
    bad = {"name": "fake", "embed_dim": 768, "available": False, "prefer_real": True}
    assert is_honest(bad) is False
    with pytest.raises(ConfigError):
        validate_encoder(OmegaConf.create({"encoder": bad}))


def test_unavailable_encoder_never_reports_vjepa_hf_backend():
    raw = _raw_configs()
    for c in raw.values():
        if bool(c.get("available", True)):
            continue
        cfg = OmegaConf.create(OmegaConf.to_container(c, resolve=True))
        cfg.pop("prefer_real", None)  # shipped honest default
        enc = load_encoder(cfg)
        assert enc.spec.backend != "vjepa_hf"
        assert enc.spec.backend == "frozen_random"


def test_available_defaults_true_when_key_absent():
    assert is_honest({"name": "x", "embed_dim": 1024}) is True
    raw = _raw_configs()
    absent = [stem for stem, c in raw.items() if "available" not in c]
    assert absent  # the verified vjepa2_* configs omit the key
    by_name = {e["name"]: e for e in list_encoders()}
    for stem in absent:
        assert by_name[OmegaConf.load(_ENCODER_DIR / f"{stem}.yaml").get("name")]["available"] is True
