from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mop.studio import generation1_injection_builder as builder
from mop.studio import generation1_supervisor as supervisor


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(supervisor.canonical_bytes(payload) + b"\n")


def _context(*, existing: frozenset[str] = frozenset({"base"})) -> builder.InjectionContext:
    return builder.InjectionContext(
        program_id="generation1-test",
        sequence=3,
        expected_queue_head_sha256=supervisor.canonical_sha256({"queue": 2}),
        created_at="2026-07-13T04:00:00+00:00",
        existing_capsule_ids=existing,
    )


def _runner(root: Path, name: str = "runner.py") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# deterministic test authority\n", encoding="utf-8")
    return path


def _capsule(
    root: Path,
    capsule_id: str,
    *,
    runner: str = "runner.py",
) -> dict[str, object]:
    _runner(root, runner)
    return builder.build_exploratory_capsule(
        repo_root=root,
        capsule_id=capsule_id,
        command=["python", runner, "--out", f"proof/{capsule_id}.json"],
        artifact_path=f"proof/{capsule_id}.json",
        artifact_schema="mop-generation1-unit-result/v1",
        artifact_fields={"scientific_promotion": False, "status": "complete"},
        authority_paths=[runner],
        depends_on=[] if capsule_id == "base" else ["base"],
        environment={"OMP_NUM_THREADS": "1"},
        process_marker=runner,
    )


def _status() -> dict[str, object]:
    core: dict[str, object] = {
        "schema": supervisor.STATUS_SCHEMA,
        "program_id": "generation1-test",
        "created_at": "2026-07-13T00:00:00-04:00",
        "queue_head_sha256": supervisor.canonical_sha256({"queue": 2}),
        "next_injection_sequence": 3,
        "capsules": {"base": {"status": "complete"}},
    }
    return {**core, "status_sha256": supervisor.canonical_sha256(core)}


def test_request_is_deterministic_sorted_self_hashed_and_exploratory_only(tmp_path: Path) -> None:
    alpha = _capsule(tmp_path, "alpha", runner="alpha.py")
    beta = _capsule(tmp_path, "beta", runner="beta.py")

    first = builder.build_injection_request(
        context=_context(),
        capsules=[beta, alpha],
        reason="  add two bounded cognitive angles  ",
        repo_root=tmp_path,
    )
    second = builder.build_injection_request(
        context=_context(),
        capsules=[alpha, beta],
        reason="add two bounded cognitive angles",
        repo_root=tmp_path,
    )

    assert first == second
    assert first["schema"] == supervisor.INJECTION_SCHEMA
    assert first["action"] == "append-capsules"
    assert first["sequence"] == 3
    assert first["created_at"] == "2026-07-13T04:00:00+00:00"
    assert first["injection_id"].startswith("inj-000003-")
    assert [row["id"] for row in first["capsules"]] == ["alpha", "beta"]
    assert all(row["kind"] == "exploratory" for row in first["capsules"])
    core = {key: value for key, value in first.items() if key != "injection_sha256"}
    assert first["injection_sha256"] == supervisor.canonical_sha256(core)


def test_capsule_json_validation_rejects_claim_bearing_kind_and_hash_tampering(
    tmp_path: Path,
) -> None:
    capsule = _capsule(tmp_path, "angle")
    claim_bearing = copy.deepcopy(capsule)
    claim_bearing["kind"] = "aggregate"
    claim_bearing.pop("capsule_sha256")
    claim_bearing["capsule_sha256"] = supervisor.canonical_sha256(claim_bearing)

    with pytest.raises(builder.InjectionBuildError, match="must have kind 'exploratory'"):
        builder.validate_exploratory_capsule(claim_bearing, repo_root=tmp_path)

    tampered = copy.deepcopy(capsule)
    tampered["priority"] = 0
    with pytest.raises(builder.InjectionBuildError, match="self-seal mismatch"):
        builder.validate_exploratory_capsule(tampered, repo_root=tmp_path)


def test_request_rejects_existing_capsule_id_and_invalid_boundary(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path, "base")
    with pytest.raises(builder.InjectionBuildError, match="already exist"):
        builder.build_injection_request(
            context=_context(),
            capsules=[capsule],
            reason="collision must fail",
            repo_root=tmp_path,
        )

    invalid = builder.InjectionContext(
        program_id="generation1-test",
        sequence=0,
        expected_queue_head_sha256="not-a-digest",
        created_at="2026-07-13",
    )
    with pytest.raises(builder.InjectionBuildError, match="sequence must be positive"):
        builder.build_injection_request(
            context=invalid,
            capsules=[capsule],
            reason="invalid boundary",
            repo_root=tmp_path,
        )


def test_context_from_status_requires_valid_seal_and_normalizes_time(tmp_path: Path) -> None:
    status_path = tmp_path / "current_status.json"
    status = _status()
    _write_json(status_path, status)

    context = builder.context_from_status(status_path)
    assert context == _context()

    symlink = tmp_path / "status-link.json"
    symlink.symlink_to(status_path)
    with pytest.raises(builder.InjectionBuildError, match="non-symlink"):
        builder.context_from_status(symlink)

    status["next_injection_sequence"] = 4
    _write_json(status_path, status)
    with pytest.raises(builder.InjectionBuildError, match="self-seal mismatch"):
        builder.context_from_status(status_path)


def test_cli_builds_idempotent_request_from_capsule_json_and_sealed_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capsule_path = tmp_path / "capsules" / "angle.json"
    status_path = tmp_path / "current_status.json"
    output = tmp_path / "requests" / "angle.json"
    _write_json(capsule_path, _capsule(tmp_path, "angle"))
    _write_json(status_path, _status())
    arguments = [
        "--repo-root",
        str(tmp_path),
        "--status",
        str(status_path),
        "--capsule-json",
        str(capsule_path),
        "--reason",
        "new bounded exploratory angle",
        "--out",
        str(output),
    ]

    assert builder.main(arguments) == 0
    first = output.read_bytes()
    summary = json.loads(capsys.readouterr().out)
    assert summary["built"] is True
    assert summary["submitted"] is False
    assert summary["capsule_ids"] == ["angle"]

    assert builder.main(arguments) == 0
    assert output.read_bytes() == first
    request = json.loads(first)
    assert request["program_id"] == "generation1-test"
    assert request["sequence"] == 3
    assert request["expected_queue_head_sha256"] == _context().expected_queue_head_sha256


def test_cli_builds_exploratory_capsule_from_direct_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _runner(tmp_path)
    output = tmp_path / "requests" / "direct.json"
    head = supervisor.canonical_sha256({"offline": "head"})

    result = builder.main(
        [
            "--repo-root",
            str(tmp_path),
            "--program-id",
            "generation1-offline",
            "--sequence",
            "1",
            "--expected-head",
            head,
            "--created-at",
            "2026-07-13T05:00:00+01:00",
            "--capsule-id",
            "direct-angle",
            "--command-json",
            json.dumps(["python", "runner.py", "--out", "proof/direct.json"]),
            "--artifact-path",
            "proof/direct.json",
            "--artifact-schema",
            "mop-generation1-direct-result/v1",
            "--artifact-fields-json",
            json.dumps({"scientific_promotion": False}),
            "--authority",
            "runner.py",
            "--env",
            "OMP_NUM_THREADS=1",
            "--reason",
            "offline deterministic request",
            "--out",
            str(output),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["built"] is True
    request = json.loads(output.read_text(encoding="utf-8"))
    capsule = request["capsules"][0]
    assert capsule["kind"] == "exploratory"
    assert capsule["authorities"] == [
        {"path": "runner.py", "sha256": supervisor.sha256_file(tmp_path / "runner.py")}
    ]
    assert request["created_at"] == "2026-07-13T04:00:00+00:00"
    assert request["injection_sha256"] == supervisor.canonical_sha256(
        {key: value for key, value in request.items() if key != "injection_sha256"}
    )


def test_write_is_immutable_when_same_path_receives_different_request(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path, "angle")
    request = builder.build_injection_request(
        context=_context(existing=frozenset()),
        capsules=[capsule],
        reason="first request",
        repo_root=tmp_path,
    )
    output = tmp_path / "request.json"
    builder.write_injection_request(output, request)

    changed = builder.build_injection_request(
        context=_context(existing=frozenset()),
        capsules=[capsule],
        reason="different request",
        repo_root=tmp_path,
    )
    with pytest.raises(builder.InjectionBuildError, match="differs"):
        builder.write_injection_request(output, changed)
