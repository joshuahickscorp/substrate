import torch

from mop import config, devices, seeding


def test_device_resolves_to_real_device():
    info = devices.resolve("auto")
    assert info.kind in ("cpu", "mps", "cuda")
    assert isinstance(info.device, torch.device)
    cpu = devices.resolve("cpu")
    assert cpu.kind == "cpu"


def test_safe_to_returns_tensor():
    info = devices.resolve("auto")
    x = torch.randn(2, 3)
    y = devices.safe_to(x, info.device)
    assert y.shape == x.shape


def test_seed_everything_repeatable_on_cpu():
    seeding.seed_everything(123)
    a = torch.randn(64)
    seeding.seed_everything(123)
    b = torch.randn(64)
    assert torch.equal(a, b)  # cpu is bit-exact under fixed seed


def test_variance_utility_quantifies_spread():
    def fn():
        return torch.randn(32) @ torch.randn(32)

    rep = seeding.variance_of(fn, runs=4, seed=7)
    assert rep.runs == 4
    assert rep.tol() > 0
    assert rep.bit_identical


def test_config_composition_and_overrides():
    cfg = config.compose(["experiment=mop_cm7_min_objective_probe", "shell.buffer.capacity=99"])
    assert cfg.package == "mop"
    assert cfg.device.kind == "mps"
    assert cfg.encoder.embed_dim == 1024
    assert cfg.shell.buffer.capacity == 99
    assert cfg.experiment.id == "mop_cm7_min_objective_probe"
    assert cfg.experiment_name == "mop_cm7_min_objective_probe"


def test_config_group_switch():
    cfg = config.compose(["device=cuda", "encoder=vjepa21_vitb"])
    assert cfg.device.kind == "cuda"
    assert cfg.encoder.embed_dim == 768
    assert cfg.encoder.dense is True


def test_snapshot_roundtrip(tmp_path):
    cfg = config.compose([])
    p = config.snapshot(cfg, tmp_path / "cfg.yaml")
    assert p.exists()
    reloaded = config.compose.__globals__["OmegaConf"].load(p)
    assert reloaded.package == "mop"
