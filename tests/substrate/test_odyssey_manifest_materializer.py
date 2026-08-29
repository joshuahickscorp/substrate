"""Tests for custody-facing source-bound manifest materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from substrate import odyssey_manifest_materializer as materializer
from tests.substrate.librispeech_audio_fixture import install_librispeech_audio_fixture


def _write(path: Path, value: dict | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sealed(value: dict) -> dict:
    body = dict(value)
    body["sha256"] = materializer.digest(body)
    return body


def _selection(root: Path) -> dict:
    rows = []
    for frontier in materializer.FRONTIERS:
        source = root / "inputs" / f"{frontier}.jsonl"
        rights = root / "inputs" / f"{frontier}.rights.json"
        _write(source, f"source-{frontier}".encode())
        _write(rights, {"license": "test-only"})
        rows.append(
            {
                "id": frontier,
                "assets": [
                    {
                        "path": str(source.relative_to(root)),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "role": "candidate_stimulus",
                        "rights_reference": str(rights.relative_to(root)),
                    }
                ],
            }
        )
    return _sealed(
        {
            "schema": "SUBSTRATE_ODYSSEY_SOURCE_SELECTION/v1",
            "program": materializer.PROGRAM,
            "status": "sealed",
            "frontiers": rows,
            "activation": False,
        }
    )


def _frozen() -> dict:
    return _sealed(
        {
            "schema": "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1",
            "implementation_sha256": {
                "task_bank_generator": materializer.canonical_source_digest(
                    Path(__file__).parents[2] / "src/substrate/odyssey_task_bank.py"
                )
            },
            "input_sha256": {"frontier_contract": "a" * 64, "task_bank": "b" * 64, "rendered_build_index": "c" * 64},
            "activation": False,
        }
    )


def test_builds_source_bound_candidate_and_evaluator_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_librispeech_audio_fixture(monkeypatch)
    selection = _selection(tmp_path)
    subject, artifacts = materializer.build_manifest_set(
        tmp_path,
        selection=selection,
        seed_bytes=b"custodian-secret",
        seed_provenance="operator_supplied",
        candidate_root=tmp_path / "candidate-visible",
        evaluator_root=tmp_path / "evaluator-only",
        frozen=_frozen(),
        source_commit="d" * 40,
    )

    assert subject["all_pass"] is True
    assert subject["external_activation"] is False
    assert subject["manifest_count"] == 8
    assert [row["id"] for row in subject["manifests"]] == list(materializer.FRONTIERS)
    assert all(row["task_count"] == 336 for row in subject["manifests"])
    candidate = json.loads(Path(artifacts["A"]["candidate"]).read_text())
    assert candidate["source_bundle"]["assets"][0]["role"] == "candidate_stimulus"
    assert candidate["source_bundle"]["assets"][0]["read_only"] is True
    assert "answer" not in json.dumps(candidate, sort_keys=True).casefold()
    evaluator = json.loads(Path(artifacts["A"]["evaluator"]).read_text())
    assert evaluator["answers"]


def test_rejects_drifted_source_asset_before_materializing(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    selection["frontiers"][0]["assets"][0]["sha256"] = "0" * 64
    selection = _sealed({key: value for key, value in selection.items() if key != "sha256"})

    with pytest.raises(materializer.Refused, match="absent or drifted"):
        materializer.build_manifest_set(
            tmp_path,
            selection=selection,
            seed_bytes=b"custodian-secret",
            seed_provenance="operator_supplied",
            candidate_root=tmp_path / "candidate-visible",
            evaluator_root=tmp_path / "evaluator-only",
            frozen=_frozen(),
            source_commit="d" * 40,
        )


def test_rejects_seed_inside_repository(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    selection_path = tmp_path / "selection.json"
    _write(selection_path, selection)
    _write(tmp_path / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", _frozen())
    seed_path = tmp_path / "secret-seed"
    _write(seed_path, b"custodian-secret")

    with pytest.raises(materializer.Refused, match="outside the repository"):
        materializer.materialize(
            tmp_path,
            selection_path=selection_path,
            seed_path=seed_path,
            candidate_root=tmp_path / "candidate-visible",
            evaluator_root=tmp_path / "evaluator-only",
            output_path=tmp_path / "manifest-set.json",
        )


def test_derived_seed_is_reproducible_and_declares_its_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without an operator seed the split must still be stable across runs of the
    # same frozen build, and the receipt must say the seed was derived rather
    # than withheld -- a reader cannot otherwise tell whether it was secret.
    install_librispeech_audio_fixture(monkeypatch)
    first = materializer.build_manifest_set(
        tmp_path,
        selection=_selection(tmp_path),
        seed_bytes=hashlib.sha256(materializer.DERIVED_SEED_DOMAIN + b"a" * 64).digest(),
        seed_provenance="derived_from_frozen_build",
        candidate_root=tmp_path / "candidate-visible",
        evaluator_root=tmp_path / "evaluator-only",
        frozen=_frozen(),
        source_commit="d" * 40,
    )[0]
    second = materializer.build_manifest_set(
        tmp_path,
        selection=_selection(tmp_path),
        seed_bytes=hashlib.sha256(materializer.DERIVED_SEED_DOMAIN + b"a" * 64).digest(),
        seed_provenance="derived_from_frozen_build",
        candidate_root=tmp_path / "candidate-visible-2",
        evaluator_root=tmp_path / "evaluator-only-2",
        frozen=_frozen(),
        source_commit="d" * 40,
    )[0]

    assert first["seed_provenance"] == "derived_from_frozen_build"
    assert [row["seed_commitment"] for row in first["manifests"]] == [
        row["seed_commitment"] for row in second["manifests"]
    ]

    with pytest.raises(materializer.Refused, match="unknown seed provenance"):
        materializer.build_manifest_set(
            tmp_path,
            selection=_selection(tmp_path),
            seed_bytes=b"custodian-secret",
            seed_provenance="assumed_secret",
            candidate_root=tmp_path / "candidate-visible-3",
            evaluator_root=tmp_path / "evaluator-only-3",
            frozen=_frozen(),
            source_commit="d" * 40,
        )


def test_custodian_can_source_bind_a_completed_draft_without_materializing_seeded_tasks(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    draft = {
        "schema": materializer.SOURCE_SELECTION_DRAFT_SCHEMA,
        "program": materializer.PROGRAM,
        "status": "ready_for_custodian_seal",
        "frontiers": selection["frontiers"],
        "activation": False,
        "external_activation": False,
    }
    draft_path = tmp_path / "operations" / "selection.ready.json"
    output_path = tmp_path / "operations" / "selection.sealed.json"
    _write(draft_path, draft)

    sealed = materializer.seal_source_selection(tmp_path, draft_path=draft_path, output_path=output_path)

    assert sealed["schema"] == materializer.SOURCE_SELECTION_SCHEMA
    assert sealed["status"] == "sealed"
    assert sealed["activation"] is False
    assert sealed["sha256"] == materializer.digest({key: value for key, value in sealed.items() if key != "sha256"})
    assert materializer.validate_source_selection(tmp_path, sealed)["A"][0]["role"] == "candidate_stimulus"
    assert not (tmp_path / "candidate-visible").exists()
    assert not (tmp_path / "evaluator-only").exists()


def test_source_selection_sealer_refuses_an_unreviewed_template(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    draft = {
        "schema": materializer.SOURCE_SELECTION_DRAFT_SCHEMA,
        "program": materializer.PROGRAM,
        "status": "template_unsealed",
        "frontiers": selection["frontiers"],
        "activation": False,
        "external_activation": False,
    }
    draft_path = tmp_path / "selection.template.json"
    output_path = tmp_path / "selection.sealed.json"
    _write(draft_path, draft)
    with pytest.raises(materializer.Refused, match="ready for custodian seal"):
        materializer.seal_source_selection(tmp_path, draft_path=draft_path, output_path=output_path)
