"""Tests for machine subject generation and converted-gate refusals."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from tests.substrate.test_odyssey_authority import (
    _fixture_root,
    _prepared_inputs,
    _write,
)

from substrate import odyssey_authority as authority
from substrate import odyssey_machine_subjects as subjects
from substrate import odyssey_transition


@pytest.fixture(autouse=True)
def fixture_git_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authority, "_git_head", lambda _root: "fixture-head")


def test_converted_gates_are_machine_verified() -> None:
    for gate_id in ("G02", "G04", "G05", "G10", "G11"):
        assert authority.GATE_SPECS[gate_id]["kind"] == "machine_verified"


def test_g04_refuses_missing_custody_limitations_and_independence(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(
        root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    g04 = authority._read_json(root / "receipts/G04.subject.json", require_digest=True)

    missing_limits = json.loads(json.dumps(g04))
    missing_limits.pop("custody_limitations")
    with pytest.raises(authority.Refused, match="custody_limitations"):
        authority._gate_specific_checks(root, "G04", missing_limits, frozen)

    missing_independence = json.loads(json.dumps(g04))
    missing_independence.pop("custody_independence")
    with pytest.raises(authority.Refused, match="wrong fields|custody_independence"):
        authority._gate_specific_checks(root, "G04", missing_independence, frozen)

    empty_limits = json.loads(json.dumps(g04))
    empty_limits["custody_limitations"] = []
    empty_limits.pop("sha256")
    empty_limits["sha256"] = authority.digest(empty_limits)
    with pytest.raises(authority.Refused, match="custody_limitations must be a non-empty list"):
        authority._gate_specific_checks(root, "G04", empty_limits, frozen)


def test_g04_refuses_duplicate_commitment_digests(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(
        root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    g04 = json.loads(json.dumps(authority._read_json(root / "receipts/G04.subject.json", require_digest=True)))
    g04["frontiers"][2]["scorer_commitment_sha256"] = g04["frontiers"][0]["scorer_commitment_sha256"]
    with pytest.raises(authority.Refused, match="must be distinct"):
        authority._gate_specific_checks(root, "G04", g04, frozen)


def test_g04_refuses_reveal_not_chained_to_trace_lock(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(
        root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    g04 = json.loads(json.dumps(authority._read_json(root / "receipts/G04.subject.json", require_digest=True)))
    g04["day7_reveal"]["trace_lock_recipe_sha256"] = "0" * 64
    with pytest.raises(authority.Refused, match="not chained to trace lock"):
        authority._gate_specific_checks(root, "G04", g04, frozen)


def test_g10_refuses_denial_that_was_not_attempted(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(
        root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    g10 = json.loads(json.dumps(authority._read_json(root / "receipts/G10.subject.json", require_digest=True)))
    receipt_ref = g10["isolation_receipts"]["candidate_evaluator_write_denied"]
    receipt_path = root / receipt_ref["path"]
    receipt = authority._read_json(receipt_path, require_digest=True)
    receipt["attempted"] = False
    receipt.pop("sha256")
    receipt["sha256"] = authority.digest(receipt)
    receipt_path.unlink()
    _write(receipt_path, receipt)
    receipt_ref["sha256"] = authority.file_digest(receipt_path)
    with pytest.raises(authority.Refused, match="denial was not attempted"):
        authority._gate_specific_checks(root, "G10", g10, frozen)


def test_subject_refuses_stale_frozen_build_or_git_head(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(
        root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    for gate_id in ("G02", "G04", "G05", "G10", "G11"):
        subject = json.loads(
            json.dumps(authority._read_json(root / "receipts" / f"{gate_id}.subject.json", require_digest=True))
        )
        subject["frozen_build_sha256"] = "0" * 64
        subject.pop("sha256")
        subject["sha256"] = authority.digest(subject)
        with pytest.raises(authority.Refused, match="not bound to this frozen build"):
            authority._gate_specific_checks(root, gate_id, subject, frozen)

        subject = json.loads(
            json.dumps(authority._read_json(root / "receipts" / f"{gate_id}.subject.json", require_digest=True))
        )
        subject["source_commit"] = "stale-head"
        subject.pop("sha256")
        subject["sha256"] = authority.digest(subject)
        with pytest.raises(authority.Refused, match="current git HEAD"):
            authority._gate_specific_checks(root, gate_id, subject, frozen)


def test_subject_refuses_placeholder_anywhere(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    frozen = authority._read_json(
        root / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json", require_digest=True
    )
    g02 = json.loads(json.dumps(authority._read_json(root / "receipts/G02.subject.json", require_digest=True)))
    g02["selection_id"] = "REPLACE_WITH_PLACEHOLDER"
    g02.pop("sha256")
    g02["sha256"] = authority.digest(g02)
    with pytest.raises(authority.Refused, match="placeholder"):
        authority._gate_specific_checks(root, "G02", g02, frozen)


def test_each_converted_gate_seals_through_seal_machine_gate(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    for gate_id in ("G02", "G04", "G05", "G10", "G11"):
        subject = root / "receipts" / f"{gate_id}.subject.json"
        output = root / "receipts" / f"{gate_id}.machine.gate.json"
        gate = authority.seal_machine_gate(root, gate_id, subject, output)
        assert gate["status"] == "pass"
        assert gate["evidence_kind"] == "machine_verified"
        assert gate["human_attestation"] is None
        assert gate["gate_id"] == gate_id


def test_g11_generator_from_design(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    out = root / "generated/G11.subject.json"
    subject = subjects.generate_g11(root, out)
    assert subject["status"] == "pass"
    assert subject["score_weights"]["task_utility"] == 0.25
    assert subject["independent_unit_count"] == 8
    authority.seal_machine_gate(root, "G11", out, root / "generated/G11.gate.json")


def test_g02_generator_pins_the_frozen_production_adapter(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    source = root / "evidence/public-model-canary.json"
    canary = root / subjects.CANARY_DIR / "fixture.json"
    canary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, canary)
    out = root / "generated/G02.subject.json"

    subject = subjects.generate_g02(root, out)

    adapter_sha256 = odyssey_transition.canonical_source_digest(root / "src/substrate/odyssey_arms.py")
    assert subject["candidate"]["adapter_sha256"] == adapter_sha256
    assert {row["adapter_sha256"] for row in subject["controls_by_frontier"].values()} == {adapter_sha256}
    authority.seal_machine_gate(root, "G02", out, root / "generated/G02.gate.json")


def test_g04_generator_requires_g03(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    g03_src = root / "receipts/G03.subject.json"
    assert g03_src.is_file()
    out = root / "generated/G04.subject.json"
    subject = subjects.generate_g04(root, out)
    assert subject["custody_independence"] == "single_operator"
    assert authority.G04_CUSTODY_LIMITATION_STATEMENT in subject["custody_limitations"]
    authority.seal_machine_gate(root, "G04", out, root / "generated/G04.gate.json")


def test_g04_generator_refuses_without_g03(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    with pytest.raises((subjects.Refused, authority.Refused), match="G03|frozen"):
        subjects.generate_g04(root, root / "generated/G04.subject.json")


def _completed(
    returncode: int,
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_g10_refuses_when_positive_control_fails_with_traversal_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody cannot reach candidate-visible → refuse; denials not recorded."""
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    nobody_uid = 99999 if os.getuid() != 99999 else 99998
    monkeypatch.setattr(subjects, "_require_sudo_nobody", lambda: (nobody_uid, "uid=99999(nobody)"))

    def fail_positive_control(argv: list[str]) -> subprocess.CompletedProcess[str]:
        # Simulate parent-dir non-traversal: every nobody probe is EACCES.
        return _completed(1, stderr="cat: Permission denied\n")

    monkeypatch.setattr(subjects, "_run_as_nobody", fail_positive_control)

    out = root / "generated/G10.subject.json"
    with pytest.raises(
        subjects.Refused,
        match="parent-directory traversal|cannot traverse|intervening directories",
    ):
        subjects.generate_g10(root, out)

    assert not out.exists()
    observations = root / subjects.ISOLATION_ARTIFACT_ROOT / "observations"
    assert not observations.exists() or not any(observations.glob("*.json"))
    pc_path = root / subjects.ISOLATION_ARTIFACT_ROOT / subjects.POSITIVE_CONTROL_ARTIFACT
    assert pc_path.is_file()
    positive = json.loads(pc_path.read_text(encoding="utf-8"))
    assert positive["succeeded"] is False
    assert positive["attempted"] is True
    assert positive["kind"] == "candidate_visible_traversal_positive_control"


def test_g10_denial_not_attempted_when_positive_control_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denial must never be recorded as attempted:true unless really attempted."""
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    nobody_uid = 99999 if os.getuid() != 99999 else 99998
    monkeypatch.setattr(subjects, "_require_sudo_nobody", lambda: (nobody_uid, "uid=99999(nobody)"))

    call_log: list[list[str]] = []

    def record_and_fail(argv: list[str]) -> subprocess.CompletedProcess[str]:
        call_log.append(list(argv))
        return _completed(1, stderr="Permission denied")

    monkeypatch.setattr(subjects, "_run_as_nobody", record_and_fail)

    with pytest.raises(subjects.Refused, match="positive control failed"):
        subjects.generate_g10(root, root / "generated/G10.subject.json")

    # Only the positive-control cat should have run as nobody; no denial probes.
    assert len(call_log) == 1
    assert call_log[0][0] == "cat"
    assert subjects.POSITIVE_CONTROL_MARKER_NAME in call_log[0][-1]
    # No sealed denial observations under the probe tree.
    obs_dir = root / subjects.ISOLATION_ARTIFACT_ROOT / "observations"
    assert not obs_dir.exists() or list(obs_dir.glob("*")) == []


def test_g10_nobody_private_dir_verified_by_ownership_not_mkdir_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mkdir success alone is insufficient; real owner must be nobody."""
    private_dir = tmp_path / "private"
    nobody_uid = 99999 if os.getuid() != 99999 else 99998
    operator_uid = os.getuid()

    def fake_sudo(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[:1] == ["mkdir"] or (len(argv) >= 1 and argv[0] == "mkdir"):
            # mkdir as the operator (no real chown): exit 0 but ownership stays operator.
            private_dir.mkdir(mode=0o700)
            return _completed(0)
        if argv[:1] == ["chown"] or (len(argv) >= 1 and argv[0] == "chown"):
            # Pretend chown succeeded without changing ownership.
            return _completed(0)
        return _completed(1, stderr=f"unexpected sudo argv: {argv}")

    monkeypatch.setattr(subjects, "_run_sudo", fake_sudo)

    with pytest.raises(subjects.Refused, match="owner is uid|expected nobody uid"):
        subjects._create_nobody_owned_private_dir(private_dir, nobody_uid)

    assert private_dir.exists()
    assert private_dir.stat().st_uid == operator_uid
    assert private_dir.stat().st_uid != nobody_uid


def test_g10_nobody_private_dir_accepts_verified_nobody_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_dir = tmp_path / "private"
    nobody_uid = os.getuid()  # fixture: treat current uid as "nobody" for stat check

    def fake_sudo(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "mkdir":
            private_dir.mkdir(mode=0o700)
            return _completed(0)
        if argv[0] == "chown":
            return _completed(0)
        return _completed(1, stderr=f"unexpected: {argv}")

    monkeypatch.setattr(subjects, "_run_sudo", fake_sudo)
    subjects._create_nobody_owned_private_dir(private_dir, nobody_uid)
    assert private_dir.is_dir()
    assert private_dir.stat().st_uid == nobody_uid


def test_g10_positive_control_succeeds_when_nobody_can_read_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    paths = subjects._prepare_isolation_roots(root)
    marker_content = subjects.POSITIVE_CONTROL_CONTENT

    def fake_nobody(argv: list[str]) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "cat"
        path = Path(argv[1])
        return _completed(0, stdout=path.read_text(encoding="utf-8"))

    monkeypatch.setattr(subjects, "_run_as_nobody", fake_nobody)
    result = subjects._run_traversal_positive_control(root, paths, nobody_uid=99)
    assert result["succeeded"] is True
    assert result["access_result"] == "allowed"
    assert result["attempted"] is True
    marker = paths["candidate_visible_root"] / subjects.POSITIVE_CONTROL_MARKER_NAME
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == marker_content
    # World-readable marker; directory allows other-execute for traversal.
    assert marker.stat().st_mode & 0o444 == 0o444
    assert paths["candidate_visible_root"].stat().st_mode & 0o001 == 0o001


def test_g10_generator_happy_path_records_denials_only_after_positive_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When positive control passes, denial probes run and attempted stays true."""
    root = _fixture_root(tmp_path)
    _prepared_inputs(root)
    nobody_uid = 99999 if os.getuid() != 99999 else 99998
    monkeypatch.setattr(subjects, "_require_sudo_nobody", lambda: (nobody_uid, "uid=99999(nobody)"))

    private_file_holder: dict[str, Path] = {}

    def fake_nobody(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "cat":
            path = Path(argv[1])
            if path.name == subjects.POSITIVE_CONTROL_MARKER_NAME:
                return _completed(0, stdout=path.read_text(encoding="utf-8"))
            # Evaluator marker / denial targets: deny.
            return _completed(1, stderr="Permission denied")
        if argv[0] == "sh" and "-c" in argv:
            script = argv[argv.index("-c") + 1]
            if "secret" in script:
                # Create private file under operator ownership for the write probe.
                # (Real probe would be nobody-owned; evaluator write still EACCES via mock.)
                # Parse path after "echo secret > "
                target = script.split(">", 1)[1].strip().split("&&", 1)[0].strip()
                private_path = Path(target)
                private_path.parent.mkdir(parents=True, exist_ok=True)
                private_path.write_text("secret\n", encoding="utf-8")
                private_path.chmod(0o000)  # operator open should fail with EACCES
                private_file_holder["path"] = private_path
                return _completed(0)
            # write probe into evaluator-only
            return _completed(1, stderr="Permission denied")
        return _completed(1, stderr=f"unexpected nobody argv: {argv}")

    def fake_sudo(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "mkdir":
            Path(argv[-1]).mkdir(mode=0o700, parents=True, exist_ok=True)
            return _completed(0)
        if argv[0] == "chown":
            # Ownership stays operator; override stat check by monkeypatching below if needed.
            return _completed(0)
        if argv[0] == "rm":
            # Root removes a previous nobody-owned probe tree.
            shutil.rmtree(Path(argv[-1]), ignore_errors=True)
            return _completed(0)
        if argv[0] == "test" and argv[1] == "-f":
            # Root confirms the nobody-written file the operator cannot stat.
            return _completed(0 if Path(argv[-1]).is_file() else 1)
        return _completed(1, stderr=f"unexpected sudo: {argv}")

    monkeypatch.setattr(subjects, "_run_as_nobody", fake_nobody)
    monkeypatch.setattr(subjects, "_run_sudo", fake_sudo)

    # Make ownership verification accept operator uid as the expected nobody for this fixture.
    # After mkdir/chown, dir is operator-owned; patch create helper to set expected uid correctly.
    real_create = subjects._create_nobody_owned_private_dir

    def create_accepting_operator_as_nobody(private_dir: Path, expected_uid: int) -> None:
        # Create via fake_sudo path but verify against real owner (operator).
        del expected_uid
        real_create(private_dir, os.getuid())

    monkeypatch.setattr(subjects, "_create_nobody_owned_private_dir", create_accepting_operator_as_nobody)

    out = root / "generated/G10.subject.json"
    subject = subjects.generate_g10(root, out)
    assert subject["status"] == "pass"
    assert subject["isolation_mode"] == "separate_uid"

    pc = json.loads(
        (root / subjects.ISOLATION_ARTIFACT_ROOT / subjects.POSITIVE_CONTROL_ARTIFACT).read_text(encoding="utf-8")
    )
    assert pc["succeeded"] is True

    for name in (
        "candidate_evaluator_read_denied",
        "candidate_evaluator_write_denied",
        "evaluator_candidate_private_write_denied",
        "builder_evaluator_read_denied",
    ):
        receipt_ref = subject["isolation_receipts"][name]
        receipt = authority._read_json(root / receipt_ref["path"], require_digest=True)
        assert receipt["attempted"] is True
        assert receipt["access_result"] == "denied"
        assert receipt["errno_name"] in {"EACCES", "EPERM"}
