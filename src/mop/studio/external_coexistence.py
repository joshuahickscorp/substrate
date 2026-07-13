"""Fail-closed identity and resource checks for unowned Hawking CPU work.

This module classifies observations only.  It grants no process ownership, may
not signal external work, and can never grant scientific promotion.  A caller
must still apply the local throttle's memory-pressure, swap, thermal, power,
disk, and task-specific admission gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "mop-external-coexistence-profile/v1"
REPORT_SCHEMA = "mop-external-coexistence-profile-report/v1"
PROFILE_NAME = "hawking_serial_cpu_v1"
DECIMAL_GB = 1_000_000_000
MAXIMUM_ROOTS = 3
MAXIMUM_PROCESSES = 8
MAXIMUM_QUANT_THREADS = 4
MAXIMUM_AGGREGATE_RSS_GB = 64.0
# Leave one logical Mac Studio CPU available to the background MOP lane. Sustained
# pressure is governed separately by the one-minute load ceiling; this cap permits
# Hawking's short all-core phase transitions without misclassifying them as memory risk.
MAXIMUM_AGGREGATE_CPU_PERCENT = 2700.0

_AUDIT_MODELS = {
    "0.5B": ("scratch/qwen-05b", "reports/cron/studio_0.5B"),
    "1.5B": ("scratch/qwen-15b", "reports/cron/studio_1.5B"),
    "7B": ("scratch/qwen-7b", "reports/cron/studio_7B"),
}
_PYTHON_ROLES = frozenset({"audit", "doctor"})
_STUDIO_RUN_SCRIPT = "tools/condense/studio_run.py"
_AUDIT_SCRIPT = "tools/condense/audit_ladder.py"
_DOCTOR_SCRIPT = "tools/condense/doctor.py"
_QUANT_ARGV0 = "vendor/strand-quant/target/release/quantize-model"
_QUANT_RELATIVE_PATH = Path(_QUANT_ARGV0)
_REQUIRED_FILE_PATHS = frozenset({_STUDIO_RUN_SCRIPT, _AUDIT_SCRIPT, _DOCTOR_SCRIPT, _QUANT_ARGV0})
LIVE_HAWKING_FILE_SHA256 = (
    (
        _AUDIT_SCRIPT,
        "1568651c1c2b9001ae6e6c9b62307ec1cf392147d9c77f8805e31fad4368cf68",
    ),
    (
        _DOCTOR_SCRIPT,
        "b47f89aba05b8b2957293d0e19cabdcc1b07d94233992c13c6c0e675a5e8f32f",
    ),
    (
        _STUDIO_RUN_SCRIPT,
        "8bdea3ee4f51207493c958fdbb086f44e2e29462c5f4da24bf5a69cb4ec1d28c",
    ),
    (
        _QUANT_ARGV0,
        "54e9689064244e3e337152e0dc77807f24dba73bd12f68a9ad88ccf3d5548416",
    ),
)
SANITIZED_ENVIRONMENT_FIELDS = (
    "CUDA_VISIBLE_DEVICES",
    "DOCTOR_DEVICE",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)
_DOCTOR_INPUT = re.compile(r"/tmp/aud_(0\.5B|1\.5B|7B)_studio_rbase\.safetensors")
_DOCTOR_OUTPUT = re.compile(r"/tmp/aud_(0\.5B|1\.5B|7B)_studio_adapter\.safetensors")
_QUANT_INPUT = re.compile(r"/tmp/aud_(0\.5B|1\.5B|7B)_studio_ckin\.safetensors")
_QUANT_OUTPUT = re.compile(r"/tmp/aud_(0\.5B|1\.5B|7B)_studio_ckout\.safetensors")
_QUANT_RUNG = re.compile(r"/tmp/aud_(0\.5B|1\.5B|7B)_studio_rung\.json")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_path(value: str | Path, label: str) -> str:
    text = os.fspath(value)
    if not text or not Path(text).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return os.path.realpath(text)


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sha256_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalized_file_hashes(
    values: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    rows = list(values.items()) if isinstance(values, Mapping) else list(values)
    if len(rows) != len(_REQUIRED_FILE_PATHS):
        raise ValueError("expected file hashes must contain exactly four bindings")
    mapping: dict[str, str] = {}
    for path, digest in rows:
        if not isinstance(path, str) or path in mapping:
            raise ValueError("expected file hash paths must be unique strings")
        mapping[path] = _sha256_digest(digest, f"expected hash for {path}")
    if set(mapping) != _REQUIRED_FILE_PATHS:
        raise ValueError("expected file hash paths do not match the reviewed Hawking files")
    return tuple(sorted(mapping.items()))


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_bound_file(root: str, relative: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(root) / relative
    problems: list[str] = []
    observed_sha256: str | None = None
    regular_file = False
    non_symlink = False
    stable_snapshot = False
    try:
        before_path = path.lstat()
        non_symlink = not stat.S_ISLNK(before_path.st_mode)
        if not non_symlink:
            problems.append("bound path is a symbolic link")
        if os.path.realpath(path) != str(path):
            problems.append("bound path traverses a symbolic link")
        regular_file = stat.S_ISREG(before_path.st_mode)
        if not regular_file:
            problems.append("bound path is not a regular file")
        if problems:
            raise OSError("bound path is not safe to open")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before_fd = os.fstat(descriptor)
            if not stat.S_ISREG(before_fd.st_mode):
                problems.append("opened binding is not a regular file")
            if (before_path.st_dev, before_path.st_ino) != (before_fd.st_dev, before_fd.st_ino):
                problems.append("bound file identity changed before hashing")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            observed_sha256 = digest.hexdigest()
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
        stable_snapshot = (
            _file_identity(before_path)
            == _file_identity(before_fd)
            == _file_identity(after_fd)
            == _file_identity(after_path)
        )
        if not stable_snapshot:
            problems.append("bound file changed while it was being hashed")
    except OSError as exc:
        if not problems:
            problems.append(f"bound file could not be snapshotted: {type(exc).__name__}")
    if observed_sha256 != expected_sha256:
        problems.append("bound file SHA-256 does not match the reviewed profile")
    return {
        "path": relative,
        "absolute_path": str(path),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "regular_file": regular_file,
        "non_symlink": non_symlink,
        "stable_snapshot": stable_snapshot,
        "problems": problems,
        "all_ok": not problems,
    }


def _bound_file_snapshots(profile: HawkingSerialCPUProfile) -> list[dict[str, Any]]:
    return [
        _snapshot_bound_file(profile.root, relative, digest)
        for relative, digest in profile.expected_file_sha256
    ]


def _sanitized_environment(value: object) -> tuple[dict[str, str | None], list[str]]:
    normalized: dict[str, str | None] = {field: None for field in SANITIZED_ENVIRONMENT_FIELDS}
    if not isinstance(value, Mapping):
        return normalized, ["sanitized environment is missing or invalid"]
    problems: list[str] = []
    if set(value) != set(SANITIZED_ENVIRONMENT_FIELDS):
        problems.append("environment fields are not the exact sanitized set")
    for field in SANITIZED_ENVIRONMENT_FIELDS:
        observed = value.get(field)
        if observed is not None and not isinstance(observed, str):
            problems.append(f"sanitized environment field {field} is not a string or null")
            continue
        normalized[field] = observed
    if normalized["DOCTOR_DEVICE"] != "cpu":
        problems.append("DOCTOR_DEVICE must be exactly cpu")
    if normalized["CUDA_VISIBLE_DEVICES"] not in {None, "", "-1"}:
        problems.append("CUDA device visibility is not disabled")
    if normalized["PYTORCH_ENABLE_MPS_FALLBACK"] not in {None, "", "0"}:
        problems.append("MPS fallback must not be enabled")
    return normalized, problems


@dataclass(frozen=True)
class HawkingSerialCPUProfile:
    """The exact external workload envelope reviewed for CPU coexistence."""

    root: str
    python_executable: str
    quantize_executable: str
    expected_uid: int
    expected_file_sha256: tuple[tuple[str, str], ...] = LIVE_HAWKING_FILE_SHA256
    maximum_roots: int = MAXIMUM_ROOTS
    maximum_processes: int = MAXIMUM_PROCESSES
    maximum_quant_threads: int = MAXIMUM_QUANT_THREADS
    maximum_aggregate_rss_gb: float = MAXIMUM_AGGREGATE_RSS_GB
    maximum_aggregate_cpu_percent: float = MAXIMUM_AGGREGATE_CPU_PERCENT

    def __post_init__(self) -> None:
        if self.root != _exact_path(self.root, "Hawking root"):
            raise ValueError("Hawking root must be an exact real path")
        if self.python_executable != _exact_path(self.python_executable, "Hawking Python executable"):
            raise ValueError("Hawking Python executable must be an exact real path")
        if self.quantize_executable != _exact_path(self.quantize_executable, "Hawking quantize executable"):
            raise ValueError("Hawking quantize executable must be an exact real path")
        if self.quantize_executable != str(Path(self.root) / _QUANT_RELATIVE_PATH):
            raise ValueError("Hawking quantize executable must be the exact file under its root")
        if (
            isinstance(self.expected_uid, bool)
            or not isinstance(self.expected_uid, int)
            or self.expected_uid < 0
        ):
            raise ValueError("expected UID must be a nonnegative integer")
        exact_limits = (
            self.maximum_roots == MAXIMUM_ROOTS
            and self.maximum_processes == MAXIMUM_PROCESSES
            and self.maximum_quant_threads == MAXIMUM_QUANT_THREADS
            and self.maximum_aggregate_rss_gb == MAXIMUM_AGGREGATE_RSS_GB
            and self.maximum_aggregate_cpu_percent == MAXIMUM_AGGREGATE_CPU_PERCENT
        )
        if not exact_limits:
            raise ValueError("Hawking serial CPU profile limits are immutable")
        if self.expected_file_sha256 != _normalized_file_hashes(self.expected_file_sha256):
            raise ValueError("expected Hawking file hashes are not canonical")

    @classmethod
    def create(
        cls,
        *,
        root: str | Path,
        python_executable: str | Path,
        expected_uid: int | None = None,
        quantize_executable: str | Path | None = None,
        expected_file_sha256: Mapping[str, str] | None = None,
    ) -> HawkingSerialCPUProfile:
        exact_root = _exact_path(root, "Hawking root")
        exact_python = _exact_path(python_executable, "Hawking Python executable")
        exact_quantize = _exact_path(
            quantize_executable or Path(exact_root) / _QUANT_RELATIVE_PATH,
            "Hawking quantize executable",
        )
        uid = os.getuid() if expected_uid is None else expected_uid
        if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
            raise ValueError("expected UID must be a nonnegative integer")
        return cls(
            root=exact_root,
            python_executable=exact_python,
            quantize_executable=exact_quantize,
            expected_uid=uid,
            expected_file_sha256=_normalized_file_hashes(
                LIVE_HAWKING_FILE_SHA256 if expected_file_sha256 is None else expected_file_sha256
            ),
        )

    def authority(self) -> dict[str, Any]:
        core: dict[str, Any] = {
            "schema": PROFILE_SCHEMA,
            "profile": PROFILE_NAME,
            "root": self.root,
            "python_executable": self.python_executable,
            "quantize_executable": self.quantize_executable,
            "expected_uid": self.expected_uid,
            "expected_files": [
                {"path": path, "sha256": digest} for path, digest in self.expected_file_sha256
            ],
            "limits": {
                "maximum_roots": self.maximum_roots,
                "maximum_processes": self.maximum_processes,
                "maximum_quant_threads": self.maximum_quant_threads,
                "maximum_aggregate_rss_gb": self.maximum_aggregate_rss_gb,
                "maximum_aggregate_cpu_percent": self.maximum_aggregate_cpu_percent,
            },
            "scientific_promotion": False,
        }
        return {**core, "profile_sha256": _canonical_sha256(core)}


def _classify_audit(argv: list[str]) -> tuple[str | None, list[str]]:
    problems: list[str] = []
    if len(argv) != 6:
        return None, ["audit argv shape is not exact"]
    if argv[1] != _AUDIT_SCRIPT:
        problems.append("audit script path is not exact")
    label = argv[3]
    expected = _AUDIT_MODELS.get(label)
    if expected is None:
        problems.append("audit model label is outside the reviewed set")
    else:
        model_path, report_path = expected
        if argv[2] != model_path:
            problems.append("audit model path does not match its label")
        if argv[4] != "studio":
            problems.append("audit mode is not studio")
        if argv[5] != report_path:
            problems.append("audit report path does not match its label")
    return label if expected is not None else None, problems


def _classify_doctor(argv: list[str]) -> tuple[str | None, list[str]]:
    if len(argv) != 8:
        return None, ["doctor argv shape is not exact"]
    problems: list[str] = []
    if argv[1] != _DOCTOR_SCRIPT:
        problems.append("doctor script path is not exact")
    if argv[2] != "lora" or argv[4:7] != ["60", "0.0001", "64"]:
        problems.append("doctor LoRA controls are not exact")
    input_match = _DOCTOR_INPUT.fullmatch(argv[3])
    output_match = _DOCTOR_OUTPUT.fullmatch(argv[7])
    if input_match is None or output_match is None:
        problems.append("doctor temporary paths are outside the reviewed grammar")
        return None, problems
    input_label = input_match.group(1)
    output_label = output_match.group(1)
    if input_label != output_label:
        problems.append("doctor input and output model labels differ")
        return None, problems
    return input_label, problems


def _classify_quantize(argv: list[str], maximum_threads: int) -> tuple[str | None, int | None, list[str]]:
    problems: list[str] = []
    if len(argv) not in {16, 18}:
        return None, None, ["quantize argv shape is not exact"]
    if argv[0] != _QUANT_ARGV0:
        problems.append("quantize argv[0] is not exact")
    expected_prefix = [
        "--in",
        argv[2],
        "--out",
        argv[4],
        "--bits",
        argv[6],
        "--rht-cols",
        "--outlier-channel",
        "1.0",
        "--outlier-bits",
        "8",
        "--threads",
        argv[13],
    ]
    if argv[1:14] != expected_prefix:
        problems.append("quantize option order or fixed controls drifted")
    input_match = _QUANT_INPUT.fullmatch(argv[2])
    output_match = _QUANT_OUTPUT.fullmatch(argv[4])
    if input_match is None or output_match is None:
        problems.append("quantize input/output paths are outside the reviewed grammar")
        model = None
    elif input_match.group(1) != output_match.group(1):
        problems.append("quantize input and output model labels differ")
        model = None
    else:
        model = input_match.group(1)
    if argv[6] not in {"1", "2", "3", "4"}:
        problems.append("quantize bit width is outside the reviewed set")
    try:
        threads = int(argv[13])
    except ValueError:
        threads = None
    if threads is None or not 1 <= threads <= maximum_threads:
        problems.append(f"quantize threads must be in [1, {maximum_threads}]")
    tail = argv[14:]
    if len(argv) == 16:
        if tail != ["--block-len", "256"]:
            problems.append("quantize terminal controls drifted")
    else:
        rung_match = _QUANT_RUNG.fullmatch(argv[15]) if len(argv) > 15 else None
        if tail[:1] != ["--rung-config"] or rung_match is None:
            problems.append("quantize rung config is outside the reviewed grammar")
        elif model is not None and rung_match.group(1) != model:
            problems.append("quantize rung config model label differs")
        if tail[2:] != ["--block-len", "256"]:
            problems.append("quantize terminal controls drifted")
    return model, threads, problems


def _argv(value: object) -> list[str] | None:
    if not isinstance(value, list | tuple) or not value:
        return None
    if any(not isinstance(part, str) or not part for part in value):
        return None
    return list(value)


def _observe_process(
    raw: Mapping[str, object],
    profile: HawkingSerialCPUProfile,
) -> dict[str, Any]:
    problems: list[str] = []
    pid = raw.get("pid")
    ppid = raw.get("ppid")
    uid = raw.get("uid")
    create_time = _numeric(raw.get("create_time"))
    rss_bytes = raw.get("rss_bytes")
    cpu_percent = _numeric(raw.get("cpu_percent"))
    exe = raw.get("exe")
    cwd = raw.get("cwd")
    argv = _argv(raw.get("cmdline"))
    environment, environment_problems = _sanitized_environment(raw.get("environment"))
    problems.extend(environment_problems)

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        problems.append("pid is missing or invalid")
    if isinstance(ppid, bool) or not isinstance(ppid, int) or ppid < 0:
        problems.append("ppid is missing or invalid")
    if isinstance(uid, bool) or not isinstance(uid, int):
        problems.append("uid is missing or invalid")
    elif uid != profile.expected_uid:
        problems.append("uid is not the current reviewed uid")
    if create_time is None or create_time <= 0:
        problems.append("create_time is missing or invalid")
    if isinstance(rss_bytes, bool) or not isinstance(rss_bytes, int) or rss_bytes < 0:
        problems.append("rss_bytes is missing or invalid")
    if cpu_percent is None or cpu_percent < 0:
        problems.append("cpu_percent is missing or invalid")
    if not isinstance(exe, str) or not exe or not Path(exe).is_absolute():
        problems.append("exe is missing or invalid")
        exact_exe = None
    else:
        exact_exe = os.path.realpath(exe)
    if not isinstance(cwd, str) or not cwd or not Path(cwd).is_absolute():
        problems.append("cwd is missing or invalid")
        exact_cwd = None
    else:
        exact_cwd = os.path.realpath(cwd)
        if exact_cwd != profile.root:
            problems.append("cwd is not the exact reviewed Hawking root")
    if argv is None:
        problems.append("cmdline is missing or invalid")

    role: str | None = None
    model: str | None = None
    quant_threads: int | None = None
    if argv is not None:
        script = argv[1] if len(argv) > 1 else None
        if script == _AUDIT_SCRIPT:
            role = "audit"
            model, grammar_problems = _classify_audit(argv)
        elif script == _DOCTOR_SCRIPT:
            role = "doctor"
            model, grammar_problems = _classify_doctor(argv)
        elif argv[0] == _QUANT_ARGV0:
            role = "quantize"
            model, quant_threads, grammar_problems = _classify_quantize(argv, profile.maximum_quant_threads)
        else:
            grammar_problems = ["cmdline does not match an exact reviewed Hawking role"]
        problems.extend(grammar_problems)
    if role in _PYTHON_ROLES and exact_exe != profile.python_executable:
        problems.append("Python executable is not exact")
    if role in _PYTHON_ROLES and argv is not None and os.path.realpath(argv[0]) != profile.python_executable:
        problems.append("Python argv[0] is not exact")
    if role == "quantize" and exact_exe != profile.quantize_executable:
        problems.append("quantize executable is not exact")

    return {
        "pid": pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
        "ppid": ppid if isinstance(ppid, int) and not isinstance(ppid, bool) else None,
        "uid": uid if isinstance(uid, int) and not isinstance(uid, bool) else None,
        "create_time": create_time,
        "exe": exact_exe,
        "cwd": exact_cwd,
        "cmdline": argv,
        "environment": environment,
        "role": role,
        "model": model,
        "quant_threads": quant_threads,
        "rss_bytes": rss_bytes if isinstance(rss_bytes, int) and not isinstance(rss_bytes, bool) else None,
        "cpu_percent": cpu_percent,
        "problems": problems,
        "all_ok": not problems,
    }


def validate_hawking_serial_cpu_snapshot(
    processes: Sequence[Mapping[str, object]],
    profile: HawkingSerialCPUProfile,
    *,
    prior_identities: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Return a deterministic self-sealed report for one complete process snapshot.

    ``prior_identities`` is optional continuity evidence.  If a PID was present
    previously, its observed creation time must remain exact; a reused PID is
    rejected rather than silently inheriting the old classification.
    """

    bound_files = _bound_file_snapshots(profile)
    raw_rows = list(processes)
    observations = [_observe_process(row, profile) for row in raw_rows]
    observations.sort(
        key=lambda row: (
            row["pid"] is None,
            row["pid"] if row["pid"] is not None else 0,
            row["create_time"] if row["create_time"] is not None else 0.0,
        )
    )
    problems: list[str] = [
        f"file {row['path']}: {problem}" for row in bound_files for problem in row["problems"]
    ]
    if not observations:
        problems.append("snapshot contains no Hawking processes")
    if len(observations) > profile.maximum_processes:
        problems.append(f"process count exceeds the {profile.maximum_processes}-process profile cap")
    for index, row in enumerate(observations):
        label = f"pid {row['pid']}" if row["pid"] is not None else f"row {index}"
        problems.extend(f"{label}: {problem}" for problem in row["problems"])

    pids = [row["pid"] for row in observations if row["pid"] is not None]
    if len(pids) != len(set(pids)):
        problems.append("snapshot contains duplicate PIDs")
    pid_set = set(pids)
    roots = [
        row
        for row in observations
        if row["pid"] is not None and row["ppid"] is not None and row["ppid"] not in pid_set
    ]
    if len(roots) > profile.maximum_roots:
        problems.append(f"root count exceeds the {profile.maximum_roots}-root profile cap")
    by_pid = {row["pid"]: row for row in observations if row["pid"] is not None}
    audit_models = [row["model"] for row in observations if row["role"] == "audit"]
    if len(audit_models) != len(set(audit_models)):
        problems.append("snapshot contains duplicate audit roots for one model")
    child_counts: dict[int, int] = {}
    for row in observations:
        if row["role"] == "audit":
            if row["ppid"] in pid_set:
                problems.append(f"pid {row['pid']}: audit process is not a snapshot root")
        elif row["role"] in {"doctor", "quantize"}:
            parent = by_pid.get(row["ppid"])
            if parent is None or parent["role"] != "audit":
                problems.append(f"pid {row['pid']}: compute parent is not an observed audit root")
            elif parent["model"] != row["model"]:
                problems.append(f"pid {row['pid']}: compute model differs from its audit parent")
            else:
                parent_pid = int(parent["pid"])
                child_counts[parent_pid] = child_counts.get(parent_pid, 0) + 1
    for parent_pid, child_count in sorted(child_counts.items()):
        if child_count > 1:
            problems.append(f"pid {parent_pid}: audit root has more than one compute child")

    prior = dict(prior_identities or {})
    continuity: list[dict[str, Any]] = []
    for row in observations:
        pid = row["pid"]
        create_time = row["create_time"]
        if pid is None or create_time is None or pid not in prior:
            continue
        expected = _numeric(prior[pid])
        exact = expected is not None and create_time == expected
        continuity.append(
            {
                "pid": pid,
                "prior_create_time": expected,
                "observed_create_time": create_time,
                "exact": exact,
            }
        )
        if not exact:
            problems.append(f"pid {pid}: creation time changed; possible PID reuse")

    rss_total = sum(int(row["rss_bytes"] or 0) for row in observations)
    cpu_total = sum(float(row["cpu_percent"] or 0.0) for row in observations)
    rss_limit = int(profile.maximum_aggregate_rss_gb * DECIMAL_GB)
    if rss_total > rss_limit:
        problems.append(f"aggregate RSS exceeds the {profile.maximum_aggregate_rss_gb:g}-GB profile cap")
    if cpu_total > profile.maximum_aggregate_cpu_percent:
        problems.append(f"aggregate CPU exceeds the {profile.maximum_aggregate_cpu_percent:g}% profile cap")

    identities = [
        {"pid": row["pid"], "create_time": row["create_time"]}
        for row in observations
        if row["pid"] is not None and row["create_time"] is not None
    ]
    core: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "profile_authority": profile.authority(),
        "observed": {
            "process_count": len(observations),
            "root_count": len(roots),
            "aggregate_rss_bytes": rss_total,
            "aggregate_rss_gb": rss_total / DECIMAL_GB,
            "aggregate_cpu_percent": cpu_total,
            "identities": identities,
            "identity_continuity": continuity,
            "bound_files": bound_files,
            "processes": observations,
        },
        "problems": problems,
        "all_ok": not problems,
        "allowed": not problems,
        "ownership": "observation-only; external processes must never receive signals",
        "scientific_promotion": False,
    }
    return {**core, "report_sha256": _canonical_sha256(core)}


__all__ = [
    "HawkingSerialCPUProfile",
    "LIVE_HAWKING_FILE_SHA256",
    "PROFILE_NAME",
    "PROFILE_SCHEMA",
    "REPORT_SCHEMA",
    "SANITIZED_ENVIRONMENT_FIELDS",
    "validate_hawking_serial_cpu_snapshot",
]
