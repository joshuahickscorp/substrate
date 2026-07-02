"""Apple-Silicon-native device behavior. Guarded so the suite still passes off Apple Silicon."""

import torch

from mop import devices


def test_apple_silicon_info_shape():
    info = devices.apple_silicon_info()
    assert "is_apple_silicon" in info
    if info["is_apple_silicon"]:
        assert info["performance_cores"] >= 1
        assert info["efficiency_cores"] >= 0
        assert info["chip"]


def test_auto_prefers_mps_on_apple_silicon():
    info = devices.resolve("auto")
    if devices._mps_ok():  # on this Apple Silicon machine, auto picks mps over cpu
        assert info.kind == "mps"
        assert info.supports_amp and info.amp_dtype == "float16"


def test_cpu_has_no_amp():
    info = devices.resolve("cpu")
    assert info.kind == "cpu" and not info.supports_amp and info.amp_dtype == "none"


def test_autocast_is_noop_on_cpu():
    info = devices.resolve("cpu")
    with devices.autocast(info):  # must not raise, no fp16 on cpu
        y = torch.randn(4, 4) @ torch.randn(4, 4)
    assert y.dtype == torch.float32


def test_autocast_runs_on_mps_when_available():
    info = devices.resolve("auto")
    if info.kind == "mps":
        x = torch.randn(8, 8, device="mps")
        with devices.autocast(info):
            y = x @ x
        assert y.shape == (8, 8)
