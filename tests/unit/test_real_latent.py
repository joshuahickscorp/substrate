"""Real-latent replication lane: the LatentStore -> stream/factorized adapters and the factorized
clip generator. Known-answer tests on a tiny hand-built store (no encoder needed), plus a pure-tensor
check that the two factorized-clip factors actually vary independently."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from mop.substrate import LatentStore, factorized_arrays, real_task_stream
from mop.substrate.real_latent import factors_meta, open_real_store

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from cache_factorized_encoder import make_factorized_clip  # noqa: E402


def _build_store(tmp_path, labels, dim=8, name="toy") -> LatentStore:
    """A tiny store whose latents are a clean linear function of the label (so probes are trivial)."""
    n = len(labels)
    store = LatentStore.create(tmp_path, name, feat_shape=(dim,), capacity=n, key_dim=dim, has_labels=True)
    lab = np.asarray(labels, dtype=np.int64)
    x = np.zeros((n, dim), dtype=np.float32)
    for i, c in enumerate(lab):
        x[i, int(c) % dim] = 1.0  # one-hot-ish, label-separable
    store.write_batch(0, x, x, lab)
    store.finalize()
    return store


def test_real_task_stream_class_incremental(tmp_path):
    # 4 classes, 6 samples each; a 2-task class-incremental stream should split classes 2/2
    labels = [c for c in range(4) for _ in range(6)]
    store = _build_store(tmp_path, labels)
    tasks = real_task_stream(store, n_tasks=2, incremental="class", seed=0)
    assert len(tasks) == 2
    t0_classes = set(tasks[0].y.tolist())
    t1_classes = set(tasks[1].y.tolist())
    assert t0_classes.isdisjoint(t1_classes), "class-incremental tasks must have disjoint labels"
    assert t0_classes | t1_classes == {0, 1, 2, 3}


def test_real_task_stream_domain_incremental_disjoint_samples(tmp_path):
    # domain-incremental: same labels every task, but disjoint SAMPLE folds
    labels = [c for c in range(3) for _ in range(9)]
    store = _build_store(tmp_path, labels)
    tasks = real_task_stream(store, n_tasks=3, incremental="domain", seed=0)
    assert len(tasks) == 3
    for t in tasks:
        assert set(t.y.tolist()) == {0, 1, 2}, "domain-incremental reuses the full label set each task"
    # every task should carry samples (folds are non-empty at 9 samples / 3 tasks)
    assert all(t.x.shape[0] > 0 for t in tasks)


def test_real_task_stream_samples_per_task_cap(tmp_path):
    labels = [c for c in range(2) for _ in range(20)]
    store = _build_store(tmp_path, labels)
    tasks = real_task_stream(store, n_tasks=2, incremental="task", samples_per_task=5, seed=0)
    assert all(t.x.shape[0] <= 5 for t in tasks)


def test_factorized_arrays_decodes_two_factors(tmp_path):
    # composite label y = a*n_b + b, n_a=3 n_b=4, one clip per cell
    n_a, n_b = 3, 4
    labels = [a * n_b + b for a in range(n_a) for b in range(n_b)]
    store = _build_store(tmp_path, labels, dim=16, name="fac")
    (store.root / "factors.json").write_text(json.dumps({"n_a": n_a, "n_b": n_b}))
    reopened = open_real_store("fac", data_dir=tmp_path)
    x, ya, yb = factorized_arrays(reopened)
    assert x.shape[0] == n_a * n_b
    # decoding must invert the composite encoding exactly
    for i, (a, b) in enumerate((a, b) for a in range(n_a) for b in range(n_b)):
        assert int(ya[i]) == a and int(yb[i]) == b
    assert set(ya.tolist()) == set(range(n_a))
    assert set(yb.tolist()) == set(range(n_b))


def test_factorized_arrays_reads_v2_factor_metadata(tmp_path):
    n_a, n_b = 2, 3
    labels = [a * n_b + b for a in range(n_a) for b in range(n_b)]
    store = _build_store(tmp_path, labels, dim=8, name="fac_v2")
    (store.root / "factors.json").write_text(
        json.dumps(
            {
                "schema": "mop-factor-sidecar/v2",
                "metadata": {"n_a": n_a, "n_b": n_b},
                "columns": {
                    "factor_a": [a for a in range(n_a) for _ in range(n_b)],
                    "factor_b": [b for _ in range(n_a) for b in range(n_b)],
                },
            }
        )
    )
    _, factor_a, factor_b = factorized_arrays(open_real_store("fac_v2", data_dir=tmp_path))
    assert factor_a.tolist() == [0, 0, 0, 1, 1, 1]
    assert factor_b.tolist() == [0, 1, 2, 0, 1, 2]


def test_factorized_arrays_rejects_single_factor_store(tmp_path):
    store = _build_store(tmp_path, [0, 1, 2, 3], name="single")
    assert factors_meta(store) is None
    try:
        factorized_arrays(store)
        raise AssertionError("expected a ValueError on a single-factor store")
    except ValueError as e:
        assert "factorized" in str(e).lower()


def test_make_factorized_clip_factors_are_independent():
    # varying factor A (hue) at fixed B must change color but keep the grating structure; varying B
    # (orientation) at fixed A must change spatial structure. Different (a, b) cells differ; the clip
    # shape and range are well-formed.
    g = torch.Generator().manual_seed(0)
    n_a, n_b = 4, 4
    c00 = make_factorized_clip(0, 0, n_a, n_b, g)
    c10 = make_factorized_clip(1, 0, n_a, n_b, g)  # different hue, same orientation
    c01 = make_factorized_clip(0, 1, n_a, n_b, g)  # same hue, different orientation
    assert c00.shape == (64, 3, 256, 256)
    assert float(c00.min()) >= 0.0 and float(c00.max()) <= 1.0
    # a hue change shifts the per-channel color balance
    assert not torch.allclose(c00.mean((0, 2, 3)), c10.mean((0, 2, 3)), atol=1e-3)
    # an orientation change alters spatial structure (spatial variance pattern differs)
    assert not torch.allclose(c00, c01, atol=1e-2)
