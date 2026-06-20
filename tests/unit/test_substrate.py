import torch

from devsys import devices
from devsys.substrate import (
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
    # encoder output is detached (no_grad); downstream cannot push grad into it
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
    # class-incremental: later tasks use higher label ids
    assert tasks[2].y.min() >= 4
    assert tasks[0].x.shape == (40, 32)


def test_make_task_stream_forward_dynamics():
    tasks = make_task_stream(n_tasks=2, dim=16, samples_per_task=20, forward_dynamics=True)
    assert tasks[0].xnext is not None
    assert tasks[0].xnext.shape == tasks[0].x.shape


def test_noisy_tv_structure():
    d = noisy_tv_dataset(dim=32, n=64, seed=2)
    assert set(d) == {"learnable", "noise"}
    # noise target variance >> learnable target variance (irreducible)
    assert d["noise"].xnext.var() > d["learnable"].xnext.var()
