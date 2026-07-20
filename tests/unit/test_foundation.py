import torch

from mop import config, devices, seeding


def test_device_resolves_to_real_device():
    info = devices.resolve("auto")
    assert info.kind in ("cpu", "mps", "cuda")
    assert isinstance(info.device, torch.device)
    cpu = devices.resolve("cpu")
    assert cpu.kind == "cpu"


def test_seed_everything_repeatable_on_cpu():
    seeding.seed_everything(123)
    a = torch.randn(64)
    seeding.seed_everything(123)
    b = torch.randn(64)
    assert torch.equal(a, b)  # cpu is bit-exact under fixed seed


def test_config_composition_and_overrides():
    cfg = config.compose(["seed=99"])
    assert cfg.package == "mop"
    assert cfg.device.kind == "mps"
    assert cfg.encoder.embed_dim == 1024
    assert cfg.seed == 99
    assert "experiment" not in cfg and "experiment_name" not in cfg


def test_config_group_switch():
    cfg = config.compose(["device=cuda"])
    assert cfg.device.kind == "cuda"
    assert cfg.encoder.embed_dim == 1024


def test_snapshot_roundtrip(tmp_path):
    cfg = config.compose([])
    p = config.snapshot(cfg, tmp_path / "cfg.yaml")
    assert p.exists()
    reloaded = config.compose.__globals__["OmegaConf"].load(p)
    assert reloaded.package == "mop"
