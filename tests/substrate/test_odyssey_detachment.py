"""Focused non-activating tests for Odyssey detachment configuration checks."""

from __future__ import annotations

import json
import plistlib
from collections.abc import Callable
from pathlib import Path

import pytest

from substrate import odyssey_detachment as detachment


def _sealed(value: dict[str, object]) -> dict[str, object]:
    document = dict(value)
    document.pop("sha256", None)
    document["sha256"] = detachment._digest(document)
    return document


def _write_json(path: Path, value: dict[str, object], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    path.chmod(mode)


def _write_plist(path: Path, value: dict[str, object], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(value, handle, sort_keys=True)
    path.chmod(mode)


def _stage_supervisor_source(root: Path) -> None:
    """Put the frozen supervisor source at the path the stage verifier checks."""
    source = Path(detachment.__file__).with_name("odyssey7d.py")
    target = root / "src/substrate/odyssey7d.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def _fixture(root: Path, launch_agents: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], dict[str, object]]:
    _stage_supervisor_source(root)
    frozen = _sealed(
        {
            "schema": detachment.FROZEN_SCHEMA,
            "program": "substrate-odyssey-r2-handoff-v1",
            "activation": False,
            "scientific_status": "frozen_waiting_for_verified_r2",
            "input_sha256": {"hardened_design": "a" * 64},
            "implementation_sha256": {
                "telegram_notifier": "b" * 64,
                "odyssey_detachment": detachment.file_digest(Path(detachment.__file__)),
                "frontier_renderer": detachment.file_digest(Path(detachment.__file__).with_name("odyssey7d.py")),
            },
        }
    )
    authority = _sealed(
        {
            "schema": detachment.AUTHORITY_SCHEMA,
            "program": {"id": detachment.PROGRAM, "launch_allowed": True, "activation": False},
            "status": "sealed_admitted",
            "activation": False,
            "external_activation": False,
            "launch_allowed": True,
            "run_id": "odyssey-detachment-fixture",
            "frozen_build_sha256": frozen["sha256"],
            "seal": {
                "status": "sealed",
                "frozen_build_sha256": frozen["sha256"],
                "protocol_digest": "c" * 64,
                "authority_source_sha256": "d" * 64,
            },
            "worker": {"run_root": str(detachment.RUN_ROOT), "argv": ["/usr/bin/true"]},
        }
    )
    _write_json(detachment.frozen_path(root), frozen)
    _write_json(detachment.authority_path(root), authority)
    for name, path in detachment.plist_paths(root, launch_agents_dir=launch_agents).items():
        _write_plist(path, detachment.expected_plists(root)[name])
    monkeypatch.setattr(detachment.odyssey_authority, "validate_current_frozen_build", lambda _root: frozen)
    monkeypatch.setattr(detachment.odyssey_authority, "verify", lambda _root, _path: {"all_pass": True})
    return authority, frozen


def _staging_frozen(root: Path, launch_agents: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install only the frozen-map fixture needed by the pre-authority stage."""
    _stage_supervisor_source(root)
    frozen = _sealed(
        {
            "schema": detachment.FROZEN_SCHEMA,
            "program": "substrate-odyssey-r2-handoff-v1",
            "activation": False,
            "scientific_status": "frozen_waiting_for_verified_r2",
            "input_sha256": {"hardened_design": "a" * 64},
            "implementation_sha256": {
                "odyssey_detachment": detachment.file_digest(Path(detachment.__file__)),
                "frontier_renderer": detachment.file_digest(Path(detachment.__file__).with_name("odyssey7d.py")),
            },
        }
    )
    _write_json(detachment.frozen_path(root), frozen)
    monkeypatch.setattr(detachment.odyssey_authority, "validate_current_frozen_build", lambda _root: frozen)
    monkeypatch.setattr(
        detachment,
        "launch_agents_root",
        lambda launch_agents_dir=None: launch_agents if launch_agents_dir is None else launch_agents_dir,
    )
    return frozen


def test_construct_receipt_requires_a_real_sealed_authority_and_never_installs_jobs(tmp_path: Path) -> None:
    launch_agents = tmp_path / "LaunchAgents"

    with pytest.raises(detachment.Refused, match="sealed Odyssey authority is unreadable"):
        detachment.construct_receipt(tmp_path, launch_agents_dir=launch_agents)

    assert not launch_agents.exists()
    assert not detachment.receipt_path(tmp_path).exists()


def test_expected_supervisor_plist_requires_current_user_caffeinate() -> None:
    expected = detachment.expected_supervisor_plist(Path("/tmp/odyssey-detachment-fixture"))

    assert expected["ProgramArguments"][:3] == [detachment.CAFFEINATE_EXECUTABLE, "-i", "-s"]
    assert expected["EnvironmentVariables"] == {
        "SUBSTRATE_ODYSSEY_SUPERVISOR": "launchd",
        detachment.POWER_ASSERTION_ENV: detachment.POWER_ASSERTION_VALUE,
    }
    assert detachment.power_resilience_contract()["mode"] == "current_user_caffeinate_child"


def test_stage_supervisor_is_private_workspace_only_and_requires_no_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    frozen = _staging_frozen(tmp_path, launch_agents, monkeypatch)

    stage = detachment.stage_supervisor(tmp_path)
    plist_path = Path(stage["staged_plist"]["path"])
    manifest_path = Path(stage["stage_manifest_path"])

    assert stage["schema"] == detachment.STAGE_SCHEMA
    assert stage["status"] == "inert_staged"
    assert stage["activation"] is False
    assert stage["external_activation"] is False
    assert stage["frozen_build"]["sha256"] == frozen["sha256"]
    assert plist_path.is_file() and manifest_path.is_file()
    assert plist_path.stat().st_mode & 0o777 == 0o600
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    assert plist_path.parent == detachment.staging_root(tmp_path)
    assert not launch_agents.exists(), "staging must never create or install a LaunchAgent"
    assert not detachment.receipt_path(tmp_path).exists(), "a stage must never become a detachment receipt"
    with plist_path.open("rb") as handle:
        assert plistlib.load(handle) == detachment.expected_supervisor_plist(tmp_path)
    assert stage["stage_contract"] == {
        "workspace_only": True,
        "requires_sealed_authority_before_install_or_activation": True,
        "command_never_calls_launchctl": True,
        "command_never_writes_launchagents": True,
        "command_never_writes_detachment_receipt": True,
        "run_at_load_is_false": True,
        "keep_alive_is_false": True,
    }
    assert detachment.verify_staged_supervisor(tmp_path) == stage


def test_stage_verifier_rejects_a_drifted_workspace_plist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    _staging_frozen(tmp_path, launch_agents, monkeypatch)
    staged = detachment.stage_supervisor(tmp_path)
    plist_path = Path(staged["staged_plist"]["path"])
    drifted = {**detachment.expected_supervisor_plist(tmp_path), "RunAtLoad": True}
    _write_plist(plist_path, drifted)

    with pytest.raises(detachment.Refused, match="does not exactly match"):
        detachment.verify_staged_supervisor(tmp_path)
    assert not launch_agents.exists()


def test_pre_authority_stage_cannot_produce_a_handoff_or_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    _staging_frozen(tmp_path, launch_agents, monkeypatch)
    detachment.stage_supervisor(tmp_path)

    with pytest.raises(detachment.Refused, match="sealed Odyssey authority is unreadable"):
        detachment.prepare_handoff(tmp_path)
    assert not launch_agents.exists()
    assert not detachment.receipt_path(tmp_path).exists()


def test_handoff_joins_only_a_verified_stage_to_a_real_sealed_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    authority, frozen = _fixture(tmp_path, launch_agents, monkeypatch)
    stage = detachment.stage_supervisor(tmp_path)

    handoff = detachment.prepare_handoff(tmp_path)

    assert handoff["schema"] == detachment.HANDOFF_SCHEMA
    assert handoff["status"] == "sealed_handoff_ready_no_activation"
    assert handoff["activation"] is False
    assert handoff["authority_sha256"] == authority["sha256"]
    assert handoff["frozen_build_sha256"] == frozen["sha256"]
    assert handoff["stage_manifest"]["sha256"] == stage["sha256"]
    assert [row["activation"] for row in handoff["ordered_external_steps"]] == [False, False, True]
    assert "supervisor_start" in handoff["forbidden_by_this_command"]


def test_receipt_binds_exact_authority_frozen_build_and_three_safe_plists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    authority, frozen = _fixture(tmp_path, launch_agents, monkeypatch)

    receipt = detachment.construct_receipt(tmp_path, launch_agents_dir=launch_agents)
    _write_json(detachment.receipt_path(tmp_path), receipt)
    verified = detachment.verify_receipt(tmp_path, launch_agents_dir=launch_agents)

    assert verified == receipt
    assert verified["authority_sha256"] == authority["sha256"]
    assert verified["frozen_build_sha256"] == frozen["sha256"]
    assert set(verified["plists"]) == {"supervisor", "run_notifier", "preflight_notifier"}
    assert verified["plists"]["supervisor"]["label"] == detachment.SUPERVISOR_LABEL
    assert verified["plists"]["run_notifier"]["label"] == detachment.RUN_NOTIFIER_LABEL
    assert verified["plists"]["preflight_notifier"]["label"] == detachment.PREFLIGHT_NOTIFIER_LABEL


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda root, agents: _write_plist(
                detachment.plist_paths(root, launch_agents_dir=agents)["run_notifier"],
                {**detachment.expected_run_notifier_plist(root), "EnvironmentVariables": {"TOKEN": "REPLACE_ME"}},
            ),
            "placeholder",
        ),
        (
            lambda root, agents: _write_plist(
                detachment.plist_paths(root, launch_agents_dir=agents)["preflight_notifier"],
                {**detachment.expected_preflight_notifier_plist(root), "RunAtLoad": False},
            ),
            "does not exactly match",
        ),
    ],
)
def test_construct_receipt_refuses_unsafe_or_drifted_plist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Callable[[Path, Path], None], match: str
) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    _fixture(tmp_path, launch_agents, monkeypatch)
    mutate(tmp_path, launch_agents)

    with pytest.raises(detachment.Refused, match=match):
        detachment.construct_receipt(tmp_path, launch_agents_dir=launch_agents)


def test_construct_receipt_refuses_authority_frozen_binding_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    authority, _frozen = _fixture(tmp_path, launch_agents, monkeypatch)
    authority["frozen_build_sha256"] = "e" * 64
    authority["seal"] = {**authority["seal"], "frozen_build_sha256": "e" * 64}
    authority = _sealed(authority)
    _write_json(detachment.authority_path(tmp_path), authority)

    with pytest.raises(detachment.Refused, match="does not bind the current frozen build"):
        detachment.construct_receipt(tmp_path, launch_agents_dir=launch_agents)


def test_verify_receipt_refuses_tampering_and_insecure_receipt_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    _fixture(tmp_path, launch_agents, monkeypatch)
    receipt = detachment.construct_receipt(tmp_path, launch_agents_dir=launch_agents)
    saved = detachment.receipt_path(tmp_path)
    _write_json(saved, receipt)

    tampered = dict(receipt)
    tampered["authority_sha256"] = "f" * 64
    _write_json(saved, tampered)
    with pytest.raises(detachment.Refused, match="self-digest"):
        detachment.verify_receipt(tmp_path, launch_agents_dir=launch_agents)

    _write_json(saved, receipt, mode=0o644)
    with pytest.raises(detachment.Refused, match="mode 0600"):
        detachment.verify_receipt(tmp_path, launch_agents_dir=launch_agents)


def test_construct_receipt_requires_authority_verification_to_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    _fixture(tmp_path, launch_agents, monkeypatch)
    monkeypatch.setattr(detachment.odyssey_authority, "verify", lambda _root, _path: {"all_pass": False})

    with pytest.raises(detachment.Refused, match="authoritative verification"):
        detachment.construct_receipt(tmp_path, launch_agents_dir=launch_agents)


def test_write_receipt_is_private_and_write_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch_agents = tmp_path / "LaunchAgents"
    _fixture(tmp_path, launch_agents, monkeypatch)

    written = detachment.write_receipt(tmp_path, launch_agents_dir=launch_agents)

    assert detachment.verify_receipt(tmp_path, launch_agents_dir=launch_agents) == written
    assert detachment.receipt_path(tmp_path).stat().st_mode & 0o777 == 0o600
    with pytest.raises(detachment.Refused, match="overwrite"):
        detachment.write_receipt(tmp_path, launch_agents_dir=launch_agents)
