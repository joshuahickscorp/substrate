import json

import numpy as np
import scripts.studio.dense_atlas_gate as dense_cli

from mop.studio.dense_atlas_gate import build_dense_atlas_cache_gate
from mop.substrate.cache_manifest import write_cache_manifest
from mop.substrate.latent_store import LatentStore


def _dense_store(root, name, *, key_offset=0.0):
    store = LatentStore.create(root, name, (4, 3), 3, 2, dtype="float32")
    latents = np.arange(36, dtype="float32").reshape(3, 4, 3)
    keys = np.arange(6, dtype="float32").reshape(3, 2) + float(key_offset)
    store.write_batch(0, latents, keys)
    store.finalize()
    write_cache_manifest(
        store.root,
        factors={"object": ["cup", "ball", "book"]},
        splits={"train": [0, 1], "val": [2]},
        full_hash_arrays=True,
    )
    return store.root


def test_dense_atlas_gate_accepts_paired_dense_real_and_control(tmp_path):
    real = _dense_store(tmp_path, "real")
    rand = _dense_store(tmp_path, "rand")
    receipt = build_dense_atlas_cache_gate(
        real_cache=real,
        randominit_cache=rand,
        min_count=3,
        min_tokens=4,
        expected_dim=3,
    )
    assert receipt["schema"] == "mop-dense-atlas-cache-gate/v1"
    assert receipt["all_ok"] is True
    assert receipt["pair"]["keys_match"] is True


def test_dense_atlas_gate_blocks_missing_randominit_cache(tmp_path):
    real = _dense_store(tmp_path, "real")
    receipt = build_dense_atlas_cache_gate(
        real_cache=real,
        randominit_cache=tmp_path / "missing",
        min_count=3,
        min_tokens=4,
        expected_dim=3,
    )
    assert receipt["all_ok"] is False
    assert any("randominit" in problem and "missing" in problem for problem in receipt["problems"])


def test_dense_atlas_gate_blocks_mismatched_referent_keys(tmp_path):
    real = _dense_store(tmp_path, "real")
    rand = _dense_store(tmp_path, "rand", key_offset=10.0)
    receipt = build_dense_atlas_cache_gate(
        real_cache=real,
        randominit_cache=rand,
        min_count=3,
        min_tokens=4,
        expected_dim=3,
    )
    assert receipt["all_ok"] is False
    assert any("referent keys" in problem for problem in receipt["problems"])


def test_dense_atlas_gate_cli_writes_blocked_receipt(tmp_path):
    out = tmp_path / "gate.json"
    rc = dense_cli.main(
        [
            "--real-cache",
            str(tmp_path / "missing-real"),
            "--randominit-cache",
            str(tmp_path / "missing-rand"),
            "--min-count",
            "3",
            "--min-tokens",
            "4",
            "--expected-dim",
            "3",
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-dense-atlas-cache-gate/v1"
    assert data["all_ok"] is False
