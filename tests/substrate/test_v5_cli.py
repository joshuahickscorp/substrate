from __future__ import annotations

import json

import pytest

from substrate import v5


def _gate_fixture() -> tuple[str, dict, dict]:
    ready = "a" * 40
    source_digest = "b" * 64
    current = {
        identity: {
            "path": path,
            "sha256": f"{index:064x}",
            "configuration_digest": ("c" * 64 if identity == "configuration" else None),
        }
        for index, (identity, path) in enumerate(
            v5.v5pilot.ADMISSION_AUTHORITY_PATHS.items(),
            start=1,
        )
    }
    admission = {
        "principal_launch_authorized": True,
        "activation": False,
        "source_commit": ready,
        "source_digest": source_digest,
        "configuration_digest": "c" * 64,
        "model_registry_digest": current["model"]["sha256"],
        "corpus_catalog_digest": current["corpus"]["sha256"],
        "authority_bindings": {
            identity: {
                "path": row["path"],
                "sha256": row["sha256"],
            }
            for identity, row in current.items()
        },
    }
    return ready, admission, current


def test_principal_gate_binds_ready_commit_authorities_and_cleanliness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready, admission, current = _gate_fixture()
    monkeypatch.setattr(v5, "_tag_commit", lambda _tag: ready)
    monkeypatch.setattr(v5, "_head_commit", lambda: ready)
    monkeypatch.setattr(v5, "_load_admission", lambda: (admission, None))
    monkeypatch.setattr(
        v5,
        "_current_authority_identities",
        lambda: (current, []),
    )
    monkeypatch.setattr(
        v5.v5io,
        "source_digest",
        lambda: admission["source_digest"],
    )
    monkeypatch.setattr(
        v5.v5campaign,
        "worktree_cleanliness",
        lambda roots: {
            "allowed_roots": list(roots),
            "clean_except_allowed_roots": True,
            "undeclared_dirty_paths": [],
            "activation": False,
        },
    )

    gate = v5.principal_gate()

    assert gate["authorized"]
    assert all(gate["checks"].values())
    assert gate["ready_commit"] == gate["current_head"] == ready
    assert gate["activation"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (
            lambda admission, _current: admission.update(source_commit="d" * 40),
            "admission_source_commit_matches_ready_commit",
        ),
        (
            lambda admission, _current: admission["authority_bindings"]["pilot"].update(sha256="e" * 64),
            "pilot_authority_digest_matches",
        ),
        (
            lambda admission, _current: admission.update(configuration_digest="f" * 64),
            "configuration_identity_matches",
        ),
        (
            lambda admission, _current: admission.update(model_registry_digest="f" * 64),
            "model_identity_matches",
        ),
        (
            lambda admission, _current: admission.update(corpus_catalog_digest="f" * 64),
            "corpus_identity_matches",
        ),
    ),
)
def test_principal_gate_fails_closed_on_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    failed_check: str,
) -> None:
    ready, admission, current = _gate_fixture()
    mutation(admission, current)
    monkeypatch.setattr(v5, "_tag_commit", lambda _tag: ready)
    monkeypatch.setattr(v5, "_head_commit", lambda: ready)
    monkeypatch.setattr(v5, "_load_admission", lambda: (admission, None))
    monkeypatch.setattr(
        v5,
        "_current_authority_identities",
        lambda: (current, []),
    )
    monkeypatch.setattr(
        v5.v5io,
        "source_digest",
        lambda: admission["source_digest"],
    )
    monkeypatch.setattr(
        v5.v5campaign,
        "worktree_cleanliness",
        lambda roots: {
            "allowed_roots": list(roots),
            "clean_except_allowed_roots": True,
            "undeclared_dirty_paths": [],
            "activation": False,
        },
    )

    gate = v5.principal_gate()

    assert not gate["authorized"]
    assert gate["checks"][failed_check] is False


def test_principal_gate_rejects_head_drift_and_undeclared_dirty_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready, admission, current = _gate_fixture()
    monkeypatch.setattr(v5, "_tag_commit", lambda _tag: ready)
    monkeypatch.setattr(v5, "_head_commit", lambda: "d" * 40)
    monkeypatch.setattr(v5, "_load_admission", lambda: (admission, None))
    monkeypatch.setattr(
        v5,
        "_current_authority_identities",
        lambda: (current, []),
    )
    monkeypatch.setattr(
        v5.v5io,
        "source_digest",
        lambda: admission["source_digest"],
    )
    monkeypatch.setattr(
        v5.v5campaign,
        "worktree_cleanliness",
        lambda roots: {
            "allowed_roots": list(roots),
            "clean_except_allowed_roots": False,
            "undeclared_dirty_paths": ["src/substrate/v5.py"],
            "activation": False,
        },
    )

    gate = v5.principal_gate()

    assert not gate["authorized"]
    assert not gate["checks"]["current_head_matches_ready_commit"]
    assert not gate["checks"]["worktree_clean_except_declared_runtime_roots"]


def test_v5_status_has_exact_stage_surface(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        v5.v5principal,
        "status",
        lambda: {
            "expected": 5_760,
            "present": 0,
            "remaining": 5_760,
            "complete": False,
            "splits": {
                "principal": {"expected": 3_456, "present": 0},
                "replication": {"expected": 1_152, "present": 0},
                "open_world_review": {"expected": 1_152, "present": 0},
            },
            "activation": False,
        },
    )
    v5.main(["status"])
    report = json.loads(capsys.readouterr().out)
    assert set(report["stages"]) == {
        "acquisition",
        "preprocessing",
        "model_preparation",
        "kernel_comparison",
        "sensorium_construction",
        "micro_canaries",
        "moderate_pilot",
        "principal_campaign",
        "replication",
        "open_world_review",
        "independent_verification",
        "terminal_publication",
    }
    assert report["activation"] is False


def test_v5_preflight_and_build_commands_are_single_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        v5.v5campaign,
        "seal_preflight",
        lambda: {
            "all_pass": True,
            "preflight": {"failed": []},
            "sealed": ["SUBSTRATE_V5_PREFLIGHT.json"],
        },
    )
    with pytest.raises(SystemExit) as preflight_exit:
        v5.main(["preflight"])
    assert preflight_exit.value.code == 0
    assert json.loads(capsys.readouterr().out)["all_pass"]

    monkeypatch.setattr(
        v5.v5campaign,
        "freeze",
        lambda: {"sealed": ["SUBSTRATE_V5_SCIENTIFIC_CONSTITUTION.json"]},
    )
    monkeypatch.setattr(
        v5.v5authorities,
        "publish_construction",
        lambda: {"count": 87},
    )
    monkeypatch.setattr(
        v5.v5principal,
        "prepare",
        lambda: {"manifest": {"unit_count": 5_760}},
    )
    with pytest.raises(SystemExit) as build_exit:
        v5.main(["build"])
    assert build_exit.value.code == 0
    assert json.loads(capsys.readouterr().out)["principal_units"] == 5_760
