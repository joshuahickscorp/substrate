"""Adversarial integrity tests for the portable entity persistence boundary."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from substrate.product import ProductRefused
from substrate.product.contracts import EntityManifest
from substrate.product.entity import EntityStore


def _manifest() -> EntityManifest:
    return EntityManifest(
        entity_id="go-systems-engineer",
        specialty="Go distributed systems",
        selected_packs=("engineering", "research"),
    )


def _store(tmp_path: Path) -> EntityStore:
    return EntityStore.create(tmp_path / "go-systems-engineer.substrate", _manifest())


def _parallel_create(entity_root: str, start: multiprocessing.synchronize.Event, result: multiprocessing.queues.Queue) -> None:
    try:
        start.wait(timeout=10)
        store = EntityStore.create(Path(entity_root), _manifest())
        result.put(("created", store.validate()["receipt_count"]))
    except ProductRefused as exc:
        result.put(("refused", str(exc)))
    except Exception as exc:  # pragma: no cover - child failures are asserted by the parent.
        result.put(("error", repr(exc)))


def test_manifest_only_extension_tampering_fails_closed_on_load(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    # This used to be silently discarded by EntityManifest.from_dict(), leaving
    # the canonical initialization hash unchanged despite a new authority claim.
    manifest["execution_permitted"] = True
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProductRefused, match="undeclared or noncanonical fields"):
        store.load()


def test_interprocess_creation_publishes_exactly_one_complete_entity(tmp_path: Path) -> None:
    entity_root = tmp_path / "concurrent-create.substrate"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    workers = [context.Process(target=_parallel_create, args=(str(entity_root), start, results)) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    observed = [results.get(timeout=2) for _ in workers]

    assert observed.count(("created", 1)) == 1
    refusals = [message for outcome, message in observed if outcome == "refused"]
    assert len(refusals) == 1
    assert "already exists" in refusals[0]
    assert not [item for item in observed if item[0] == "error"]
    assert EntityStore(entity_root).validate()["receipt_count"] == 1
    assert not list(tmp_path.glob(".concurrent-create.substrate.staging-*"))


def test_writer_lock_symlink_is_refused_without_touching_its_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside_target = tmp_path / "outside-target"
    outside_target.write_text("must not become a lock file", encoding="utf-8")
    store.writer_lock_path.symlink_to(outside_target)

    with pytest.raises(ProductRefused, match="entity writer lock"):
        store.record("operator_note", {"note": "must not write through a link"})

    assert outside_target.read_text(encoding="utf-8") == "must not become a lock file"
    assert store.validate()["receipt_count"] == 1
