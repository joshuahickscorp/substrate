from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mop import condensation


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seal(value: dict, field: str) -> dict:
    core = {key: item for key, item in value.items() if key != field}
    return {**core, field: condensation.canonical_sha256(core)}


def _contract(*, max_line: int = 400) -> dict:
    return {
        "schema": condensation.CONTRACT_SCHEMA,
        "recommended_target_active_repo_loc": 100,
        "stretch_target_runtime_core_loc": 80,
        "elimination_gate_profile": "full",
        "fallback_ladder": [300, 200, 100],
        "measurement": {
            "max_active_line_bytes": max_line,
            "surfaces": [
                {"name": "docs", "active": False, "include": ["docs/**", "*.md"]},
                {"name": "assets", "active": False, "include": ["assets/**"]},
                {"name": "validation", "active": True, "include": ["tests/**"]},
                {"name": "laboratory", "active": True, "include": ["lab/**"]},
                {"name": "runtime", "active": True, "include": ["src/**"]},
                {"name": "other", "active": True, "default": True},
            ],
        },
        "planned_packs": [],
        "waves": [],
        "activation": {},
        "gates": {
            "quick": [],
            "full": [{"id": "suite", "argv": ["true"]}],
        },
    }


def _git(root: Path, *argv: str) -> None:
    result = subprocess.run(["git", "-C", str(root), *argv], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def _git_text(root: Path, *argv: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *argv], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _commit_all(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "MOP Test")
    _git(root, "config", "user.email", "mop-test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")


def test_contract_requires_descending_ladder_ending_at_recommended_target(tmp_path: Path) -> None:
    contract = _contract()
    path = tmp_path / "condensation.json"
    _write_json(path, contract)
    assert condensation.load_contract(path)["fallback_ladder"] == [300, 200, 100]

    contract["fallback_ladder"] = [300, 100, 200]
    _write_json(path, contract)
    with pytest.raises(condensation.CondensationError, match="fallback_ladder"):
        condensation.load_contract(path)


def test_measurement_partitions_tracked_text_and_reports_binary_separately(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "src/core.py").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "tests/test_core.py").write_text("assert True\n", encoding="utf-8")
    (tmp_path / "docs/design.md").write_text("heading\nbody\n", encoding="utf-8")
    (tmp_path / "assets/blob.bin").write_bytes(b"\x00\x01\x02")
    _commit_all(tmp_path)

    shape = condensation.measure_repository(tmp_path, _contract())

    assert shape["active_repo_LOC"] == 3
    assert shape["tracked_text_LOC"] == 5
    assert shape["binary_files"] == 1
    assert shape["surface_LOC"]["runtime"]["loc"] == 2
    assert shape["surface_LOC"]["validation"]["loc"] == 1
    assert shape["surface_LOC"]["docs"]["loc"] == 2
    assert shape["no_gaming_violations"] == []
    assert len(shape["tree_sha256"]) == 64


def test_active_binary_and_line_packing_fail_no_gaming_gate(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/packed.py").write_text("x" * 201 + "\n", encoding="utf-8")
    (tmp_path / "src/native.bin").write_bytes(b"\x00\x01")
    _commit_all(tmp_path)

    shape = condensation.measure_repository(tmp_path, _contract(max_line=200))

    assert any("exceeds 200 bytes" in problem for problem in shape["no_gaming_violations"])
    assert any("active binary file" in problem for problem in shape["no_gaming_violations"])


def test_baseline_is_recomputed_from_its_exact_source_commit(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/core.py").write_text("stable\n", encoding="utf-8")
    contract_path = tmp_path / "condensation.json"
    _write_json(contract_path, _contract())
    lock = _seal(
        {
            "schema": condensation.PACK_LOCK_SCHEMA,
            "created_at": "2026-07-16T00:00:00Z",
            "sequence": 0,
            "previous_lock_sha256": None,
            "packs": [],
        },
        "lock_sha256",
    )
    lock_path = tmp_path / "condensation.packs.lock.json"
    _write_json(lock_path, lock)
    eliminations = _seal(
        {
            "schema": condensation.ELIMINATION_LEDGER_SCHEMA,
            "created_at": "2026-07-16T00:00:00Z",
            "sequence": 0,
            "previous_ledger_sha256": None,
            "entries": [],
        },
        "ledger_sha256",
    )
    elimination_path = tmp_path / "condensation.eliminations.json"
    _write_json(elimination_path, eliminations)
    bindings = _seal(
        {
            "schema": condensation.LIVE_BINDINGS_SCHEMA,
            "observed_at": "2026-07-16T00:00:00Z",
            "bindings": [
                {
                    "path": "src/core.py",
                    "sha256": condensation.sha256_file(tmp_path / "src/core.py"),
                }
            ],
        },
        "snapshot_sha256",
    )
    binding_path = tmp_path / "condensation.live-bindings.json"
    _write_json(binding_path, bindings)
    _commit_all(tmp_path)

    baseline = condensation.build_baseline(
        tmp_path,
        contract_path=contract_path,
        lock_path=lock_path,
        elimination_ledger_path=elimination_path,
        live_bindings_path=binding_path,
    )
    verified = condensation.verify_baseline(
        tmp_path,
        baseline,
        contract=_contract(),
        contract_path=contract_path,
    )

    assert verified["all_match"] is True
    assert baseline["source_commit"] == _git_text(tmp_path, "rev-parse", "HEAD")
    assert baseline["shape"]["active_repo_LOC"] > 0
    assert len(baseline["active_inventory_sha256"]) == 64


def _make_pack(
    cache_root: Path,
    *,
    relative_path: str = "lab/example.py",
    surface: str = "laboratory",
    source_text: str = "alpha\nbeta\n",
) -> tuple[Path, dict]:
    payload = cache_root / "objects/research-v1/payload"
    payload.mkdir(parents=True)
    source = payload / relative_path
    source.parent.mkdir(parents=True)
    source.write_text(source_text, encoding="utf-8")
    file_row = {
        "path": relative_path,
        "sha256": condensation.sha256_file(source),
        "bytes": source.stat().st_size,
        "loc": len(source_text.splitlines()),
        "surface": surface,
    }
    manifest_core = {
        "schema": condensation.PACK_MANIFEST_SCHEMA,
        "pack_id": "mop-research",
        "version": "1",
        "payload_sha256": condensation.canonical_sha256([file_row]),
        "files": [file_row],
    }
    manifest = _seal(manifest_core, "manifest_sha256")
    _write_json(payload.parent / "manifest.json", manifest)
    entry = {
        "pack_id": "mop-research",
        "version": "1",
        "manifest_sha256": manifest["manifest_sha256"],
        "payload_sha256": manifest["payload_sha256"],
        "cache_relpath": "objects/research-v1",
        "mount_relpath": "research/1",
    }
    lock_core = {
        "schema": condensation.PACK_LOCK_SCHEMA,
        "created_at": "2026-07-16T00:00:00Z",
        "sequence": 0,
        "previous_lock_sha256": None,
        "packs": [entry],
    }
    lock = _seal(lock_core, "lock_sha256")
    lock_path = cache_root / "lock.json"
    _write_json(lock_path, lock)
    return lock_path, entry


def _set_validation_inventory(
    cache_root: Path,
    lock_path: Path,
    nodeids: list[str],
) -> None:
    manifest_path = cache_root / "objects/research-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodeids = sorted(nodeids)
    manifest["validation_inventory"] = {
        "payload_sha256": manifest["payload_sha256"],
        "collection_paths": ["tests"],
        "count": len(nodeids),
        "sha256": condensation.canonical_sha256(nodeids),
        "nodeids": nodeids,
    }
    manifest = _seal(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["packs"][0]["manifest_sha256"] = manifest["manifest_sha256"]
    _write_json(lock_path, _seal(lock, "lock_sha256"))


def test_checksum_locked_pack_verifies_and_hydrates_offline(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, entry = _make_pack(cache)

    verified = condensation.verify_packs(lock_path, cache)
    assert verified["pack_count"] == 1
    assert verified["owned_text_LOC"] == 2
    assert verified["relocated_LOC"] == 0

    destination = tmp_path / "hydrated"
    receipt = condensation.hydrate_packs(
        lock_path,
        cache_root=cache,
        destination_root=destination,
    )
    hydrated = destination / entry["mount_relpath"] / "lab/example.py"
    assert hydrated.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert receipt["packs"][0]["payload_sha256"] == entry["payload_sha256"]


def test_validation_pack_is_collected_from_cache_and_staged_hydration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    lock_path, entry = _make_pack(
        cache,
        relative_path="tests/test_packed.py",
        surface="validation",
        source_text="def test_packed():\n    assert True\n",
    )
    nodeid = "tests/test_packed.py::test_packed"
    _set_validation_inventory(cache, lock_path, [nodeid])
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k impossible")

    verified = condensation.verify_packs(lock_path, cache, contract=_contract())
    assert verified["validation_nodeids"] == [nodeid]

    destination = tmp_path / "hydrated"
    receipt = condensation.hydrate_packs(
        lock_path,
        cache_root=cache,
        destination_root=destination,
        contract=_contract(),
    )
    assert receipt["pack_count"] == 1
    assert (destination / entry["mount_relpath"] / "tests/test_packed.py").is_file()
    execution = condensation.run_hydrated_validation(tmp_path, verified, receipt)
    assert execution["ok"] is True
    assert execution["pack_count"] == 1


def test_validation_pack_cannot_claim_an_uncollected_nodeid(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(
        cache,
        relative_path="tests/test_packed.py",
        surface="validation",
        source_text="def test_real():\n    assert True\n",
    )
    _set_validation_inventory(cache, lock_path, ["tests/test_packed.py::test_missing"])

    with pytest.raises(condensation.CondensationError, match="validation inventory"):
        condensation.verify_packs(lock_path, cache, contract=_contract())


def test_pack_tampering_fails_closed(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(cache)
    (cache / "objects/research-v1/payload/lab/example.py").write_text(
        "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(condensation.CondensationError, match="invalid sha256"):
        condensation.verify_packs(lock_path, cache)


def test_unlisted_pack_payload_fails_closed(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(cache)
    (cache / "objects/research-v1/payload/unlisted.py").write_text(
        "not in manifest\n",
        encoding="utf-8",
    )

    with pytest.raises(condensation.CondensationError, match="inventory mismatch"):
        condensation.verify_packs(lock_path, cache)


def test_pack_mount_may_not_be_checkout_root(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(cache)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["packs"][0]["mount_relpath"] = "."
    _write_json(lock_path, _seal(lock, "lock_sha256"))

    with pytest.raises(condensation.CondensationError, match="unsafe path component"):
        condensation.hydrate_packs(
            lock_path,
            cache_root=cache,
            destination_root=tmp_path / "hydrated",
        )


def test_pack_surface_is_derived_from_its_logical_path(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(
        cache,
        relative_path="src/example.py",
        surface="docs",
    )

    with pytest.raises(condensation.CondensationError, match="surface mismatch"):
        condensation.verify_packs(lock_path, cache, contract=_contract())


def test_pack_active_line_packing_fails_closed(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(
        cache,
        relative_path="src/example.py",
        surface="runtime",
        source_text=f"{'x' * 401}\n",
    )

    with pytest.raises(condensation.CondensationError, match="packed active line"):
        condensation.verify_packs(lock_path, cache, contract=_contract())


def test_relocation_credit_requires_exact_missing_baseline_origin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src/example.py"
    source.parent.mkdir()
    source.write_text("alpha\nbeta\n", encoding="utf-8")
    _commit_all(repo)
    baseline = _git_text(repo, "rev-parse", "HEAD")
    digest = condensation.sha256_file(source)
    source.unlink()
    _git(repo, "add", "-u")
    _git(repo, "commit", "-qm", "extract")

    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(cache, relative_path="src/example.py", surface="runtime")
    manifest_path = cache / "objects/research-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["origin"] = {
        "path": "src/example.py",
        "sha256": digest,
        "loc": 2,
    }
    manifest["payload_sha256"] = condensation.canonical_sha256(manifest["files"])
    manifest = _seal(manifest, "manifest_sha256")
    _write_json(manifest_path, manifest)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["packs"][0]["manifest_sha256"] = manifest["manifest_sha256"]
    lock["packs"][0]["payload_sha256"] = manifest["payload_sha256"]
    _write_json(lock_path, _seal(lock, "lock_sha256"))

    verified = condensation.verify_packs(
        lock_path,
        cache,
        repo_root=repo,
        baseline_commit=baseline,
        contract=_contract(),
    )

    assert verified["relocated_LOC"] == 2
    assert verified["relocated_paths"] == ["src/example.py"]
    assert verified["active_owned_LOC"] == 2


def test_pack_lock_history_is_immediate_append_only_ancestry(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, _ = _make_pack(cache)
    genesis = json.loads(lock_path.read_text(encoding="utf-8"))
    history = cache / "condensation/history/pack-lock"
    _write_json(history / f"{genesis['lock_sha256']}.json", genesis)
    successor = _seal(
        {
            **{key: value for key, value in genesis.items() if key != "lock_sha256"},
            "created_at": "2026-07-16T01:00:00Z",
            "sequence": 1,
            "previous_lock_sha256": genesis["lock_sha256"],
        },
        "lock_sha256",
    )
    _write_json(lock_path, successor)

    verified = condensation.verify_packs(lock_path, cache)
    assert verified["lineage"] == [successor["lock_sha256"], genesis["lock_sha256"]]

    archived = history / f"{genesis['lock_sha256']}.json"
    archived.write_text(
        archived.read_text(encoding="utf-8").replace("2026-07-16T00:00:00Z", "tampered"),
        encoding="utf-8",
    )
    with pytest.raises(condensation.CondensationError, match="self-seal"):
        condensation.verify_packs(lock_path, cache)


def test_existing_hydration_is_verified_and_never_deleted_on_mismatch(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    lock_path, entry = _make_pack(cache)
    destination = tmp_path / "hydrated"
    mounted = destination / entry["mount_relpath"] / "lab/example.py"
    mounted.parent.mkdir(parents=True)
    mounted.write_text("local drift\n", encoding="utf-8")

    with pytest.raises(condensation.CondensationError, match="invalid sha256"):
        condensation.hydrate_packs(
            lock_path,
            cache_root=cache,
            destination_root=destination,
        )
    assert mounted.read_text(encoding="utf-8") == "local drift\n"


def test_live_binding_snapshot_detects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "src/core.py"
    source.parent.mkdir(parents=True)
    source.write_text("stable\n", encoding="utf-8")
    manifest_core = {
        "schema": condensation.LIVE_BINDINGS_SCHEMA,
        "observed_at": "2026-07-16T00:00:00Z",
        "bindings": [{"path": "src/core.py", "sha256": condensation.sha256_file(source)}],
    }
    manifest = _seal(manifest_core, "snapshot_sha256")
    manifest_path = tmp_path / "bindings.json"
    _write_json(manifest_path, manifest)

    assert condensation.verify_live_bindings(tmp_path, manifest_path)["all_match"] is True
    source.write_text("drifted\n", encoding="utf-8")
    result = condensation.verify_live_bindings(tmp_path, manifest_path)
    assert result["all_match"] is False
    assert result["mismatches"][0]["path"] == "src/core.py"


def test_empty_elimination_ledger_claims_no_deletion(tmp_path: Path) -> None:
    source = tmp_path / "src/core.py"
    source.parent.mkdir(parents=True)
    source.write_text("stable\n", encoding="utf-8")
    _commit_all(tmp_path)
    baseline = _git_text(tmp_path, "rev-parse", "HEAD")
    ledger = _seal(
        {
            "schema": condensation.ELIMINATION_LEDGER_SCHEMA,
            "created_at": "2026-07-16T00:00:00Z",
            "sequence": 0,
            "previous_ledger_sha256": None,
            "entries": [],
        },
        "ledger_sha256",
    )
    ledger_path = tmp_path / "eliminations.json"
    _write_json(ledger_path, ledger)

    verified = condensation.verify_eliminations(
        tmp_path,
        ledger_path,
        baseline_commit=baseline,
        contract=_contract(),
    )

    assert verified["eliminated_LOC"] == 0
    assert verified["before_paths"] == []


def test_elimination_credit_requires_the_immutable_full_gate_profile(tmp_path: Path) -> None:
    old = tmp_path / "src/old.py"
    old.parent.mkdir(parents=True)
    old.write_text("one\ntwo\n", encoding="utf-8")
    _commit_all(tmp_path)
    baseline = _git_text(tmp_path, "rev-parse", "HEAD")
    old_sha = condensation.sha256_file(old)
    old.unlink()
    new = tmp_path / "src/new.py"
    new.write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "condense")
    entry = {
        "id": "runtime-condense",
        "before": [{"path": "src/old.py", "sha256": old_sha, "loc": 2}],
        "after": [
            {
                "path": "src/new.py",
                "sha256": condensation.sha256_file(new),
                "loc": 1,
            }
        ],
        "eliminated_LOC": 1,
        "validation": ["looks-good"],
    }
    ledger = _seal(
        {
            "schema": condensation.ELIMINATION_LEDGER_SCHEMA,
            "created_at": "2026-07-16T00:00:00Z",
            "sequence": 0,
            "previous_ledger_sha256": None,
            "entries": [entry],
        },
        "ledger_sha256",
    )
    ledger_path = tmp_path / "eliminations.json"
    _write_json(ledger_path, ledger)

    with pytest.raises(condensation.CondensationError, match="full gate profile"):
        condensation.verify_eliminations(
            tmp_path,
            ledger_path,
            baseline_commit=baseline,
            contract=_contract(),
        )

    entry["validation"] = {"gate_profile": "full", "gate_ids": ["suite"]}
    ledger["entries"] = [entry]
    _write_json(ledger_path, _seal(ledger, "ledger_sha256"))
    verified = condensation.verify_eliminations(
        tmp_path,
        ledger_path,
        baseline_commit=baseline,
        contract=_contract(),
    )
    assert verified["eliminated_LOC"] == 1
    assert verified["required_gate_profile"] == "full"


def test_accounting_never_calls_relocated_code_eliminated() -> None:
    result = condensation.accounting(
        {
            "active_repo_LOC": 190,
            "tracked_text_LOC": 250,
            "surface_LOC": {"runtime": {"loc": 190}},
        },
        baseline={"shape": {"active_repo_LOC": 300}},
        packs={
            "active_owned_LOC": 80,
            "owned_text_LOC": 80,
            "relocated_LOC": 80,
            "non_relocated_owned_LOC": 0,
            "surface_LOC": {"runtime": 80},
        },
        eliminations={"eliminated_LOC": 30},
        contract=_contract(),
    )

    assert result["hydrated_owned_LOC"] == 270
    assert result["relocated_LOC"] == 80
    assert result["eliminated_LOC"] == 30
    assert result["added_LOC"] == 0
    assert result["unexplained_reduction_LOC"] == 0
    assert result["runtime_core_LOC"] == 190
    assert result["stretch_target_runtime_core_LOC"] == 80
    assert result["stretch_target_runtime_core_met"] is False
    assert result["next_checkpoint"] == 100


def test_added_code_is_reported_instead_of_negative_elimination() -> None:
    result = condensation.accounting(
        {
            "active_repo_LOC": 320,
            "tracked_text_LOC": 320,
            "surface_LOC": {"runtime": {"loc": 320}},
        },
        baseline={"shape": {"active_repo_LOC": 300}},
        packs={
            "active_owned_LOC": 10,
            "owned_text_LOC": 10,
            "relocated_LOC": 10,
            "non_relocated_owned_LOC": 0,
            "surface_LOC": {"runtime": 10},
        },
        eliminations={"eliminated_LOC": 0},
        contract=_contract(),
    )

    assert result["eliminated_LOC"] == 0
    assert result["added_LOC"] == 30
    assert result["hydrated_owned_LOC"] == 330


def test_next_checkpoint_descends_one_release_rung_at_a_time() -> None:
    ladder = [
        250_000,
        225_000,
        200_000,
        175_000,
        150_000,
        125_000,
        100_000,
        75_000,
        50_000,
        35_000,
        25_000,
    ]
    assert condensation.next_checkpoint(326_492, ladder) == 250_000
    assert condensation.next_checkpoint(225_000, ladder) == 200_000
    assert condensation.next_checkpoint(100_000, ladder) == 75_000
    assert condensation.next_checkpoint(25_000, ladder) is None
