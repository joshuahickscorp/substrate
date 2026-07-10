import numpy as np
from scripts.quarantine_uncitable_caches import build_quarantine_receipt

from mop.substrate.cache_manifest import write_cache_manifest
from mop.substrate.latent_store import LatentStore


def _store(root, name):
    store = LatentStore.create(root, name, (3,), 3, 3)
    values = np.arange(9, dtype="float32").reshape(3, 3)
    store.write_batch(0, values, values)
    store.finalize()
    return store


def test_quarantine_is_reversible_move_and_preserves_every_hash(tmp_path):
    active = tmp_path / "active"
    quarantine = tmp_path / "quarantine"
    legacy = _store(active, "legacy")
    receipt = build_quarantine_receipt(active, quarantine, execute=True)
    row = receipt["records"][0]
    assert receipt["all_ok"] and row["action"] == "quarantined"
    assert not legacy.root.exists()
    moved = quarantine / "legacy"
    assert moved.exists()
    for file_record in row["files"]:
        path = moved / file_record["path"]
        assert path.stat().st_size == file_record["bytes"]


def test_strict_citable_programmatic_cache_stays_active(tmp_path):
    active = tmp_path / "active"
    quarantine = tmp_path / "quarantine"
    store = _store(active, "citable")
    write_cache_manifest(
        store.root,
        encoder_config={"name": "fixture-generator", "revision": "v1"},
        referents=["r0", "r1", "r2"],
        form_kind="symbolic",
        form_objective="programmatic",
        referent_scheme="fixture-id",
        full_hash_arrays=True,
    )
    receipt = build_quarantine_receipt(active, quarantine, execute=True)
    assert receipt["records"][0]["action"] == "retained-active"
    assert store.root.exists()
    assert not quarantine.exists()
