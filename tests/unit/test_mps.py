import warnings

import torch
from omegaconf import OmegaConf

from mop import devices
from mop.config import CONFIG_DIR

MPS_YAML = CONFIG_DIR / "device" / "mps.yaml"

VIT_L_64FRAME_TOKENS = 8192


def route_encoder_forward(n_tokens: int, cap: int, mps_kind: str = "mps") -> str:
    if n_tokens > cap:
        warnings.warn(
            f"encoder forward of {n_tokens} tokens exceeds mps_safe_token_cap={cap}; routing to cpu",
            stacklevel=2,
        )
        return "cpu"
    return mps_kind


def test_cpu_has_no_amp():
    info = devices.resolve("cpu")
    assert info.kind == "cpu"
    assert info.supports_amp is False
    assert info.amp_dtype == "none"


def test_auto_resolves_mps_on_apple_silicon():
    info = devices.resolve("auto")
    if devices._mps_ok():  # Apple Silicon native: auto picks mps over cpu, fp16 autocast
        assert info.kind == "mps"
        assert info.supports_amp is True
        assert info.amp_dtype == "float16"


def test_autocast_is_noop_on_cpu():
    info = devices.resolve("cpu")
    with devices.autocast(info):  # cpu has no fp16 autocast; block must run and not downcast
        y = torch.randn(4, 4) @ torch.randn(4, 4)
    assert y.dtype == torch.float32  # stays float32: the cpu context is a true no-op


def test_autocast_does_not_raise_on_mps():
    if not devices._mps_ok():
        return  # off Apple Silicon there is no mps device to enter; nothing to assert
    info = devices.resolve("auto")
    assert info.kind == "mps"
    x = torch.randn(8, 8, device="mps")
    with devices.autocast(info):  # fp16 autocast on Metal must not raise (routing/annotation only)
        y = x @ x
    assert y.shape == (8, 8)


def test_mps_config_declares_fallback_and_token_cap():
    cfg = OmegaConf.load(MPS_YAML)
    assert cfg.kind == "mps"
    assert cfg.amp is True  # fp16 autocast opt-in
    assert cfg.pin_memory is False  # unified memory: no host/device pinning
    assert cfg.allow_cpu_fallback is True  # PYTORCH_ENABLE_MPS_FALLBACK: patchy ops run on cpu
    assert "mps_safe_token_cap" in cfg


def test_token_cap_is_positive_int_and_vit_l_exceeds_it():
    cfg = OmegaConf.load(MPS_YAML)
    cap = cfg.mps_safe_token_cap
    assert isinstance(cap, int) and cap > 0  # positive int cap
    assert cap < VIT_L_64FRAME_TOKENS


def test_route_encoder_forward_caps_to_cpu():
    cfg = OmegaConf.load(MPS_YAML)
    cap = cfg.mps_safe_token_cap
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        route = route_encoder_forward(VIT_L_64FRAME_TOKENS, cap)
    assert route == "cpu"
    assert any("mps_safe_token_cap" in str(w.message) for w in caught)  # the WARNING fired
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        small = route_encoder_forward(cap, cap)
    assert small == "mps"
    assert not caught


def test_apple_silicon_info_shape():
    info = devices.apple_silicon_info()
    assert isinstance(info["is_apple_silicon"], bool)
    if info["is_apple_silicon"]:
        assert info["performance_cores"] >= 1
