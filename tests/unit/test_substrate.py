import pytest
import torch

from mop import devices
from mop.substrate import (
    EncoderSpec,
    FrozenEncoder,
    LatentStore,
    cache_latents,
    load_encoder,
    make_task_stream,
    noisy_tv_dataset,
    synthetic_clips,
)


def test_frozen_encoder_has_no_grad_params():
    enc = FrozenEncoder(EncoderSpec("t", 32, dense=False, pool="mean"))
    _ = enc.encode(torch.randn(2, 3, 4, 8, 8))  # init lazy layer
    assert all(not p.requires_grad for p in enc.parameters())


def test_frozen_encoder_never_accumulates_grad():
    enc = FrozenEncoder(EncoderSpec("t", 16, dense=False, pool="mean"))
    z = enc.encode(torch.randn(3, 3, 2, 8, 8))
    assert not z.requires_grad


def test_encoder_pooled_and_dense_shapes():
    pooled = FrozenEncoder(EncoderSpec("p", 24, dense=False, pool="mean"))
    z = pooled.encode(torch.randn(4, 3, 2, 8, 8))
    assert z.shape == (4, 24)
    dense = FrozenEncoder(EncoderSpec("d", 24, dense=True, pool="none"))
    zd = dense.encode(torch.randn(4, 3, 2, 8, 8))
    assert zd.dim() == 3 and zd.shape[0] == 4


def test_load_encoder_falls_back_to_frozen_random(monkeypatch):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "name": "x",
            "embed_dim": 32,
            "dense": False,
            "pool": "mean",
            "hf_id": "nope/none",
            "hub": "n",
            "frozen": True,
        }
    )
    enc = load_encoder(cfg)
    assert enc.spec.backend == "frozen_random"


def test_load_encoder_can_refuse_real_weight_fallback(monkeypatch):
    from omegaconf import OmegaConf

    import mop.substrate.encoder as encoder_module

    monkeypatch.setattr(encoder_module, "_try_real_weights", lambda cfg: None)
    cfg = OmegaConf.create(
        {
            "name": "required-real",
            "embed_dim": 32,
            "dense": False,
            "pool": "mean",
            "hf_id": "fixture/unavailable",
            "frozen": True,
            "prefer_real": True,
            "require_real": True,
        }
    )
    with pytest.raises(RuntimeError, match="refusing an unrequested frozen-random fallback"):
        load_encoder(cfg)


def test_random_architecture_is_explicit_and_cannot_mix_with_real(monkeypatch):
    from omegaconf import OmegaConf

    import mop.substrate.encoder as encoder_module

    cfg = OmegaConf.create(
        {
            "name": "fixture",
            "embed_dim": 4,
            "dense": False,
            "pool": "mean",
            "revision": "pinned-revision",
            "random_init": True,
            "random_init_seed": 7,
            "prefer_real": False,
            "require_real": False,
        }
    )
    monkeypatch.setattr(encoder_module, "_try_random_architecture", lambda ignored: torch.nn.Identity())
    encoder = load_encoder(cfg)
    assert encoder.spec.backend == "vjepa_hf_random_init"
    cfg.prefer_real = True
    with pytest.raises(ValueError, match="cannot be combined"):
        load_encoder(cfg)


def test_random_architecture_path_never_calls_from_pretrained(monkeypatch):
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoModel

    import mop.substrate.encoder as encoder_module

    calls = {"config": 0, "from_config": 0, "from_pretrained": 0}

    class Architecture:
        hidden_size = 4

    def fake_config(model_id, **kwargs):
        calls["config"] += 1
        assert model_id == "fixture/random-vit"
        assert kwargs["revision"] == "pinned-revision"
        assert kwargs["local_files_only"] is True
        return Architecture()

    def fake_from_config(architecture, **kwargs):
        calls["from_config"] += 1
        assert isinstance(architecture, Architecture)
        return torch.nn.Linear(4, 4)

    def forbidden_from_pretrained(*args, **kwargs):
        calls["from_pretrained"] += 1
        raise AssertionError("random control attempted to load pretrained weights")

    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(AutoModel, "from_config", fake_from_config)
    monkeypatch.setattr(AutoModel, "from_pretrained", forbidden_from_pretrained)
    cfg = OmegaConf.create(
        {
            "name": "fixture",
            "hf_id": "fixture/random-vit",
            "revision": "pinned-revision",
            "embed_dim": 4,
            "local_files_only": True,
            "random_init_seed": 7,
        }
    )
    model = encoder_module._try_random_architecture(cfg)
    assert model is not None
    assert calls == {"config": 1, "from_config": 1, "from_pretrained": 0}


def test_module_state_sha256_identifies_realized_random_weights():
    import mop.substrate.encoder as encoder_module

    torch.manual_seed(1)
    first = torch.nn.Linear(4, 3)
    torch.manual_seed(1)
    same = torch.nn.Linear(4, 3)
    torch.manual_seed(2)
    different = torch.nn.Linear(4, 3)
    assert encoder_module.module_state_sha256(first) == encoder_module.module_state_sha256(same)
    assert encoder_module.module_state_sha256(first) != encoder_module.module_state_sha256(different)


def test_store_roundtrip(tmp_path):
    s = LatentStore.create(tmp_path, "s", feat_shape=(8,), capacity=10, key_dim=8, has_labels=True)
    x = torch.randn(10, 8).numpy()
    y = torch.arange(10).numpy()
    s.write_batch(0, x, x, y)
    s.finalize()
    r = LatentStore.open(tmp_path / "s")
    assert len(r) == 10
    assert torch.allclose(r.latents(), torch.from_numpy(x))
    assert torch.equal(r.labels(), torch.from_numpy(y))


def test_cache_pipeline_roundtrip(tmp_path):
    enc = FrozenEncoder(EncoderSpec("c", 16, dense=False, pool="mean"))
    dev = devices.resolve("cpu")
    store = cache_latents(
        enc, synthetic_clips(n=24, batch=8, n_classes=4), tmp_path, "cache", total=24, device=dev
    )
    assert len(store) == 24
    assert store.latents().shape == (24, 16)
    assert store.labels().max() < 4


def test_make_task_stream_class_incremental():
    tasks = make_task_stream(
        n_tasks=3, dim=32, classes_per_task=2, samples_per_task=40, incremental="class", seed=1
    )
    assert len(tasks) == 3
    assert tasks[0].n_classes == 6
    assert tasks[2].y.min() >= 4
    assert tasks[0].x.shape == (40, 32)


def test_make_task_stream_forward_dynamics():
    tasks = make_task_stream(n_tasks=2, dim=16, samples_per_task=20, forward_dynamics=True)
    assert tasks[0].xnext is not None
    assert tasks[0].xnext.shape == tasks[0].x.shape


def test_noisy_tv_structure():
    d = noisy_tv_dataset(dim=32, n=64, seed=2)
    assert set(d) == {"learnable", "noise"}
    assert d["noise"].xnext.var() > d["learnable"].xnext.var()
