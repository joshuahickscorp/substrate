from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from mop.studio.external_coexistence import (
    HawkingSerialCPUProfile,
    validate_hawking_serial_cpu_snapshot,
)


def _profile(tmp_path: Path) -> HawkingSerialCPUProfile:
    root = tmp_path / "hawking"
    python = tmp_path / "Python"
    quantize = root / "vendor/strand-quant/target/release/quantize-model"
    quantize.parent.mkdir(parents=True)
    root.mkdir(exist_ok=True)
    python.touch()
    contents = {
        "tools/condense/studio_run.py": b"studio-run-fixture\n",
        "tools/condense/audit_ladder.py": b"audit-ladder-fixture\n",
        "tools/condense/doctor.py": b"doctor-fixture\n",
        "vendor/strand-quant/target/release/quantize-model": b"quantize-fixture\n",
    }
    expected: dict[str, str] = {}
    for relative, payload in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected[relative] = hashlib.sha256(payload).hexdigest()
    return HawkingSerialCPUProfile.create(
        root=root,
        python_executable=python,
        quantize_executable=quantize,
        expected_file_sha256=expected,
    )


def _environment() -> dict[str, str | None]:
    return {
        "CUDA_VISIBLE_DEVICES": None,
        "DOCTOR_DEVICE": "cpu",
        "PYTORCH_ENABLE_MPS_FALLBACK": None,
    }


def _audit(
    profile: HawkingSerialCPUProfile,
    *,
    pid: int = 100,
    ppid: int = 1,
    label: str = "0.5B",
) -> dict[str, Any]:
    paths = {
        "0.5B": ("scratch/qwen-05b", "reports/cron/studio_0.5B"),
        "1.5B": ("scratch/qwen-15b", "reports/cron/studio_1.5B"),
        "7B": ("scratch/qwen-7b", "reports/cron/studio_7B"),
    }
    model, report = paths[label]
    return {
        "pid": pid,
        "ppid": ppid,
        "uid": os.getuid(),
        "create_time": 1000.0 + pid,
        "exe": profile.python_executable,
        "cwd": profile.root,
        "cmdline": [
            profile.python_executable,
            "tools/condense/audit_ladder.py",
            model,
            label,
            "studio",
            report,
        ],
        "environment": _environment(),
        "rss_bytes": 2_000_000_000,
        "cpu_percent": 5.0,
    }


def _doctor(
    profile: HawkingSerialCPUProfile,
    *,
    pid: int = 101,
    ppid: int = 100,
    label: str = "0.5B",
) -> dict[str, Any]:
    return {
        "pid": pid,
        "ppid": ppid,
        "uid": os.getuid(),
        "create_time": 1000.0 + pid,
        "exe": profile.python_executable,
        "cwd": profile.root,
        "cmdline": [
            profile.python_executable,
            "tools/condense/doctor.py",
            "lora",
            f"/tmp/aud_{label}_studio_rbase.safetensors",
            "60",
            "0.0001",
            "64",
            f"/tmp/aud_{label}_studio_adapter.safetensors",
        ],
        "environment": _environment(),
        "rss_bytes": 6_000_000_000,
        "cpu_percent": 100.0,
    }


def _quantize(
    profile: HawkingSerialCPUProfile,
    *,
    pid: int = 201,
    ppid: int = 200,
    label: str = "1.5B",
    threads: str = "4",
) -> dict[str, Any]:
    return {
        "pid": pid,
        "ppid": ppid,
        "uid": os.getuid(),
        "create_time": 1000.0 + pid,
        "exe": profile.quantize_executable,
        "cwd": profile.root,
        "cmdline": [
            "vendor/strand-quant/target/release/quantize-model",
            "--in",
            f"/tmp/aud_{label}_studio_ckin.safetensors",
            "--out",
            f"/tmp/aud_{label}_studio_ckout.safetensors",
            "--bits",
            "3",
            "--rht-cols",
            "--outlier-channel",
            "1.0",
            "--outlier-bits",
            "8",
            "--threads",
            threads,
            "--rung-config",
            f"/tmp/aud_{label}_studio_rung.json",
            "--block-len",
            "256",
        ],
        "environment": _environment(),
        "rss_bytes": 10_000_000_000,
        "cpu_percent": 400.0,
    }


def _valid_snapshot(profile: HawkingSerialCPUProfile) -> list[dict[str, Any]]:
    return [
        _audit(profile),
        _doctor(profile),
        _audit(profile, pid=200, label="1.5B"),
        _quantize(profile),
    ]


def test_valid_exact_hawking_serial_cpu_snapshot_is_self_sealed(tmp_path: Path) -> None:
    profile = _profile(tmp_path)

    report = validate_hawking_serial_cpu_snapshot(_valid_snapshot(profile), profile)

    assert report["all_ok"] is True
    assert report["allowed"] is True
    assert report["problems"] == []
    assert report["scientific_promotion"] is False
    assert report["profile_authority"]["scientific_promotion"] is False
    assert len(report["profile_authority"]["expected_files"]) == 4
    assert all(row["all_ok"] for row in report["observed"]["bound_files"])
    assert len(report["report_sha256"]) == 64
    assert report["observed"]["process_count"] == 4
    assert report["observed"]["root_count"] == 2
    assert report["ownership"].startswith("observation-only")


def test_substring_spoof_is_not_an_exact_role(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    row = _audit(profile)
    row["cmdline"][1] = "tools/condense/evil-audit_ladder.py"

    report = validate_hawking_serial_cpu_snapshot([row], profile)

    assert report["allowed"] is False
    assert any("does not match an exact reviewed Hawking role" in value for value in report["problems"])


@pytest.mark.parametrize("mutation", ["cwd", "uid", "exe", "argv0"])
def test_wrong_cwd_uid_or_executable_fails_closed(tmp_path: Path, mutation: str) -> None:
    profile = _profile(tmp_path)
    row = _audit(profile)
    if mutation == "cwd":
        row["cwd"] = str(tmp_path / "hawking-spoof")
    elif mutation == "uid":
        row["uid"] = os.getuid() + 1
    elif mutation == "exe":
        row["exe"] = str(tmp_path / "Python-spoof")
    else:
        row["cmdline"][0] = str(tmp_path / "Python-spoof")

    report = validate_hawking_serial_cpu_snapshot([row], profile)

    assert report["allowed"] is False


def test_missing_creation_time_and_pid_reuse_fail_closed(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    missing = _audit(profile)
    missing.pop("create_time")
    missing_report = validate_hawking_serial_cpu_snapshot([missing], profile)
    assert missing_report["allowed"] is False
    assert any("create_time is missing" in value for value in missing_report["problems"])

    row = _audit(profile)
    reuse_report = validate_hawking_serial_cpu_snapshot(
        [row], profile, prior_identities={100: float(row["create_time"]) - 1.0}
    )
    assert reuse_report["allowed"] is False
    assert any("possible PID reuse" in value for value in reuse_report["problems"])


def test_quant_thread_cap_is_exact(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    rows = [_audit(profile, pid=200, label="1.5B"), _quantize(profile, threads="5")]

    report = validate_hawking_serial_cpu_snapshot(rows, profile)

    assert report["allowed"] is False
    assert any("quantize threads must be in [1, 4]" in value for value in report["problems"])


def test_process_and_root_caps_fail_closed(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    process_rows = [_audit(profile)]
    process_rows.extend(_doctor(profile, pid=pid) for pid in range(101, 109))
    process_report = validate_hawking_serial_cpu_snapshot(process_rows, profile)
    assert process_report["allowed"] is False
    assert any("process count exceeds" in value for value in process_report["problems"])

    root_rows = [
        _audit(profile, pid=100, label="0.5B"),
        _audit(profile, pid=200, label="1.5B"),
        _audit(profile, pid=300, label="7B"),
        _audit(profile, pid=400, label="0.5B"),
    ]
    root_report = validate_hawking_serial_cpu_snapshot(root_rows, profile)
    assert root_report["allowed"] is False
    assert any("root count exceeds" in value for value in root_report["problems"])


@pytest.mark.parametrize("resource", ["rss", "cpu"])
def test_aggregate_resource_caps_fail_closed(tmp_path: Path, resource: str) -> None:
    profile = _profile(tmp_path)
    row = _audit(profile)
    if resource == "rss":
        row["rss_bytes"] = 64_000_000_001
    else:
        row["cpu_percent"] = 2700.1

    report = validate_hawking_serial_cpu_snapshot([row], profile)

    assert report["allowed"] is False
    assert any(f"aggregate {resource.upper()} exceeds" in value for value in report["problems"])


def test_same_path_file_replacement_is_rejected(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    audit_path = Path(profile.root) / "tools/condense/audit_ladder.py"
    audit_path.write_bytes(b"same-path-replacement\n")

    report = validate_hawking_serial_cpu_snapshot(_valid_snapshot(profile), profile)

    assert report["allowed"] is False
    assert any(
        "audit_ladder.py: bound file SHA-256 does not match" in problem for problem in report["problems"]
    )


def test_same_path_symlink_replacement_is_rejected_even_with_matching_bytes(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    doctor_path = Path(profile.root) / "tools/condense/doctor.py"
    replacement = tmp_path / "doctor-replacement.py"
    replacement.write_bytes(doctor_path.read_bytes())
    doctor_path.unlink()
    doctor_path.symlink_to(replacement)

    report = validate_hawking_serial_cpu_snapshot(_valid_snapshot(profile), profile)

    assert report["allowed"] is False
    assert any("doctor.py: bound path is a symbolic link" in problem for problem in report["problems"])


@pytest.mark.parametrize(
    ("field", "value", "expected_problem"),
    [
        ("DOCTOR_DEVICE", "mps", "DOCTOR_DEVICE must be exactly cpu"),
        ("DOCTOR_DEVICE", "cuda", "DOCTOR_DEVICE must be exactly cpu"),
        ("CUDA_VISIBLE_DEVICES", "0", "CUDA device visibility is not disabled"),
        ("PYTORCH_ENABLE_MPS_FALLBACK", "1", "MPS fallback must not be enabled"),
    ],
)
def test_gpu_or_mps_environment_declarations_are_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    expected_problem: str,
) -> None:
    profile = _profile(tmp_path)
    row = _audit(profile)
    row["environment"][field] = value

    report = validate_hawking_serial_cpu_snapshot([row], profile)

    assert report["allowed"] is False
    assert any(expected_problem in problem for problem in report["problems"])


def test_full_or_unsanitized_environment_is_never_serialized(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    row = _audit(profile)
    row["environment"]["SECRET_TOKEN"] = "must-not-appear"

    report = validate_hawking_serial_cpu_snapshot([row], profile)

    assert report["allowed"] is False
    serialized = str(report)
    assert "must-not-appear" not in serialized
    assert "SECRET_TOKEN" not in serialized
    assert any("exact sanitized set" in problem for problem in report["problems"])
