"""Strict, read-only readiness doctor for the local and Studio execution envelopes.

The doctor distinguishes three things that older checks blurred together: software readiness,
evidence readiness, and hardware-envelope compatibility. It never downloads weights, repairs a cache,
or infers a hardware wall from a profile label. A missing package, decoder, local model snapshot, citable
cache manifest, or measured host resource is a failed check with an explicit remedy.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from omegaconf import DictConfig, OmegaConf

from .config import REPO_ROOT
from .devices import apple_silicon_info
from .studio.memory_envelope import memory_snapshot
from .studio.profiles import get_profile
from .substrate.cache_tools import validate_cache

SCHEMA = "mop-studio-readiness/v2"

# Checks are ordered from process portability through software, evidence, and launch resources.
CHECK_NAMES = (
    "python",
    "package_import",
    "torch",
    "apple_silicon",
    "memory_telemetry",
    "disk_space",
    "profile_host_match",
    "profile_floor",
    "video_backend",
    "huggingface",
    "encoders",
    "encoder_weights",
    "cache_manifests",
    "cache_write",
    "config_validation",
)


def _check(name: str, fn: Callable[[], tuple[bool, str]]) -> dict:
    """Turn a probe exception into a failed check instead of crashing the doctor."""
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    return {"name": name, "ok": bool(ok), "detail": detail}


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return ok, f"{v.major}.{v.minor}.{v.micro}; project floor is 3.11"


def _check_package_import() -> tuple[bool, str]:
    """Prove ``mop`` imports outside the repository without relying on PYTHONPATH or cwd."""
    code = (
        "import importlib.metadata,json,mop; "
        "print(json.dumps({'module':mop.__file__,'version':importlib.metadata.version('mop')}))"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tempfile.gettempdir(),
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if proc.returncode:
        error = (proc.stderr or proc.stdout).strip().splitlines()
        tail = error[-1] if error else f"exit {proc.returncode}"
        return False, f"isolated import failed: {tail}; run `uv pip install -e .`"
    payload = json.loads(proc.stdout.strip())
    return True, f"mop {payload['version']} from {payload['module']} (isolated cwd, no PYTHONPATH)"


def _check_torch() -> tuple[bool, str]:
    try:
        import torch
    except Exception as e:
        return False, f"torch not importable: {type(e).__name__}"
    avail = bool(torch.backends.mps.is_available())
    built = bool(torch.backends.mps.is_built())
    return True, f"torch {torch.__version__}; mps available={avail} built={built}"


def _check_apple_silicon() -> tuple[bool, str]:
    info = apple_silicon_info()
    if not info.get("is_apple_silicon"):
        return True, "not Apple Silicon (CPU/CUDA development remains supported)"
    chip = info.get("chip", "Apple Silicon")
    p, e = info.get("performance_cores"), info.get("efficiency_cores")
    mem = info.get("unified_memory_gb")
    ok = mem is not None and p is not None and e is not None
    return ok, f"{chip}: {p}P/{e}E cores, {mem} GB unified, mps={info.get('mps_available')}"


def _check_memory_telemetry() -> tuple[bool, str]:
    """Require system and process memory telemetry used by every scale-boundary receipt."""
    snap = memory_snapshot("studio_doctor")
    required = ("process_rss_gb", "system_total_gb", "system_available_gb")
    missing = [key for key in required if snap.get(key) is None]
    try:
        import psutil

        version = psutil.__version__
    except Exception:
        version = "absent"
    if missing:
        return False, f"psutil={version}; missing {missing}; install project dependencies"
    return (
        True,
        f"psutil={version}; RSS={snap['process_rss_gb']} GB, system="
        f"{snap['system_available_gb']}/{snap['system_total_gb']} GB available",
    )


def _check_disk_space() -> tuple[bool, str]:
    du = shutil.disk_usage(REPO_ROOT)
    free_gb = du.free / 1e9
    ok = free_gb >= 5.0
    return ok, f"{free_gb:.1f} GB free of {du.total / 1e9:.1f} GB at repository filesystem"


def _infer_profile_name() -> str:
    """Select the largest measured resource envelope this host actually satisfies."""
    host = apple_silicon_info()
    for name in ("studio-m1ultra", "studio-1tb"):
        compatible, _, _ = get_profile(name).host_compatibility(host=host)
        if compatible:
            return name
    return "m3pro-local-max"


def _check_profile_host(profile_name: str | None = None) -> tuple[bool, str]:
    profile = get_profile(profile_name or _infer_profile_name())
    ok, problems, measured = profile.host_compatibility()
    detail = (
        f"{profile.name}: measured {measured['chip']}, {measured['unified_memory_gb']} GB unified, "
        f"{measured['disk_total_gb']} GB disk; requires {profile.min_host_unified_memory_gb:.1f} GB "
        f"unified and {profile.min_host_disk_gb:.1f} GB disk"
    )
    if problems:
        detail += "; PROFILE/HOST MISMATCH: " + "; ".join(problems)
    return ok, detail


def _check_profile_floor(profile_name: str | None = None) -> tuple[bool, str]:
    profile = get_profile(profile_name or _infer_profile_name())
    ok, free_gb = profile.free_disk_ok()
    status = "profile floor ok" if ok else "PROFILE BLOCKED"
    return (
        ok,
        f"{profile.name}: {free_gb:.1f} GB free, min {profile.min_free_disk_gb:.0f} GB ({status})",
    )


def _isolated_import_probe(modules: tuple[str, ...]) -> tuple[bool, str]:
    """Import optional native modules outside the doctor process.

    Video backends load codec and accelerator libraries with process-global destructors.  A broken
    optional backend must be reported as unavailable without being able to crash the readiness
    process later during interpreter teardown.
    """

    imports = ";".join(f"importlib.import_module({module!r})" for module in modules)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", f"import importlib;{imports}"],
        cwd=tempfile.gettempdir(),
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if proc.returncode == 0:
        return True, "ok"
    lines = (proc.stderr or proc.stdout).strip().splitlines()
    detail = lines[-1] if lines else f"exit {proc.returncode}"
    return False, detail


@lru_cache(maxsize=1)
def _check_video_backend() -> tuple[bool, str]:
    """Require an importable real-video decoder while containing native-library failures."""

    errors: list[str] = []
    ok, detail = _isolated_import_probe(("decord",))
    if ok:
        return True, "present: decord (isolated import)"
    errors.append(f"decord={detail}")

    ok, detail = _isolated_import_probe(("av", "torchvision.io"))
    if ok:
        return True, "present: torchvision.io + PyAV (isolated import)"
    errors.append(f"torchvision/PyAV={detail}")
    return False, ", ".join(errors) + "; install `.[video]` before real-video work"


def _check_huggingface() -> tuple[bool, str]:
    """Require encoder libraries but make no network request; local weights are checked separately."""
    versions: list[str] = []
    for package, module in (("huggingface-hub", "huggingface_hub"), ("transformers", "transformers")):
        try:
            imported = __import__(module)
            versions.append(f"{package}={getattr(imported, '__version__', 'present')}")
        except Exception as e:
            return False, f"{package} not importable: {type(e).__name__}; install `.[encoder]`"
    return True, ", ".join(versions) + "; network not probed"


def _encoder_configs() -> list[tuple[Path, DictConfig]]:
    configs: list[tuple[Path, DictConfig]] = []
    for path in sorted((REPO_ROOT / "configs/encoder").glob("*.yaml")):
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"{path.relative_to(REPO_ROOT)} must contain a mapping")
        configs.append((path, cfg))
    return configs


def _check_encoders() -> tuple[bool, str]:
    configs = _encoder_configs()
    if not configs:
        return False, "no encoder configs found"
    rows, ok = [], True
    for path, cfg in configs:
        name = str(OmegaConf.select(cfg, "name", default=path.stem))
        dim = int(OmegaConf.select(cfg, "embed_dim", default=0))
        available = bool(OmegaConf.select(cfg, "available", default=True))
        hf_id = str(OmegaConf.select(cfg, "hf_id", default=""))
        if dim <= 0 or not hf_id:
            ok = False
        rows.append(f"{name}(d={dim},{'published' if available else 'deferred'})")
    return ok, f"{len(configs)} configs: " + ", ".join(rows)


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(roots))


def _local_weight_files(hf_id: str) -> list[Path]:
    repo_slug = "models--" + hf_id.replace("/", "--")
    patterns = ("*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth")
    found: list[Path] = []
    for root in _hf_cache_roots():
        snapshot_root = root / repo_slug / "snapshots"
        if not snapshot_root.exists():
            continue
        for pattern in patterns:
            found.extend(path for path in snapshot_root.rglob(pattern) if ".incomplete" not in path.name)
    return sorted(set(found))


@lru_cache(maxsize=32)
def _sha256_snapshot(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """Hash one immutable file snapshot while avoiding repeated multi-GB reads in one process."""
    del size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_current(path: Path) -> str:
    stat = path.stat()
    return _sha256_snapshot(
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _direct_checkpoint_path(cfg: DictConfig) -> tuple[Path | None, str | None]:
    """Resolve an official direct checkpoint without treating its sentinel hf_id as a Hub repo."""
    configured = OmegaConf.select(cfg, "checkpoint_path", default=None)
    if configured:
        path = Path(str(configured)).expanduser()
        return (path if path.is_absolute() else REPO_ROOT / path).resolve(), None

    source_url = str(OmegaConf.select(cfg, "checkpoint_url", default="")).strip()
    filename = Path(urlparse(source_url).path).name
    if not filename:
        return None, "direct checkpoint config lacks checkpoint_path and checkpoint_url filename"
    model_root = REPO_ROOT / "data" / "models"
    matches = sorted(path.resolve() for path in model_root.rglob(filename) if path.is_file())
    if not matches:
        return None, f"direct checkpoint {filename!r} is absent under {model_root}"
    if len(matches) > 1:
        return None, f"direct checkpoint {filename!r} is ambiguous under {model_root}: {matches}"
    return matches[0], None


def _check_direct_checkpoint(name: str, cfg: DictConfig) -> tuple[bool, str]:
    """Validate direct checkpoint bytes and their adjacent immutable authority receipt."""
    checkpoint, resolution_problem = _direct_checkpoint_path(cfg)
    if checkpoint is None:
        return False, str(resolution_problem)
    problems: list[str] = []
    if not checkpoint.is_file():
        return False, f"direct checkpoint missing: {checkpoint}"

    expected_size_raw = OmegaConf.select(cfg, "checkpoint_content_length", default=None)
    try:
        expected_size = int(expected_size_raw)
    except (TypeError, ValueError):
        expected_size = 0
        problems.append("config lacks a positive checkpoint_content_length")
    if expected_size <= 0:
        problems.append("config checkpoint_content_length must be positive")
    actual_size = int(checkpoint.stat().st_size)
    if expected_size > 0 and actual_size != expected_size:
        problems.append(f"file size {actual_size} does not match configured {expected_size}")

    expected_sha = str(OmegaConf.select(cfg, "checkpoint_sha256", default="")).lower()
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        problems.append("config lacks a valid checkpoint_sha256")
        expected_sha = ""
    actual_sha = _sha256_current(checkpoint)
    if expected_sha and actual_sha != expected_sha:
        problems.append("file SHA256 does not match configured checkpoint_sha256")

    receipt_path_raw = OmegaConf.select(cfg, "checkpoint_receipt_path", default=None)
    if receipt_path_raw:
        receipt_path = Path(str(receipt_path_raw)).expanduser()
        if not receipt_path.is_absolute():
            receipt_path = REPO_ROOT / receipt_path
        receipt_path = receipt_path.resolve()
    else:
        receipt_path = checkpoint.with_name(checkpoint.name + ".receipt.json")
    receipt: dict = {}
    if not receipt_path.is_file():
        problems.append(f"checkpoint receipt missing: {receipt_path}")
    else:
        try:
            loaded = json.loads(receipt_path.read_text())
            receipt = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                problems.append("checkpoint receipt must be a JSON mapping")
        except (OSError, json.JSONDecodeError):
            problems.append("checkpoint receipt is not valid JSON")
    if receipt:
        if receipt.get("schema") != "mop-vjepa21-official-checkpoint/v1":
            problems.append("checkpoint receipt schema is not the official checkpoint schema")
        if receipt.get("all_ok") is not True:
            problems.append("checkpoint receipt is not green")
        if receipt.get("size") != expected_size or receipt.get("size") != actual_size:
            problems.append("checkpoint receipt size does not match config and file")
        if receipt.get("sha256") != expected_sha or receipt.get("sha256") != actual_sha:
            problems.append("checkpoint receipt SHA256 does not match config and file")
        receipt_bindings = {
            "source_url": OmegaConf.select(cfg, "checkpoint_url", default=None),
            "source_etag": OmegaConf.select(cfg, "checkpoint_etag", default=None),
            "source_version_id": OmegaConf.select(cfg, "checkpoint_version_id", default=None),
            "repository_commit": OmegaConf.select(cfg, "official_repo_commit", default=None),
        }
        for field, expected in receipt_bindings.items():
            if expected is not None and receipt.get(field) != expected:
                problems.append(f"checkpoint receipt {field} does not match config")

    detail = (
        f"{name}: direct checkpoint {checkpoint}, {actual_size} bytes, sha256={actual_sha}, "
        f"receipt={receipt_path}"
    )
    if problems:
        detail += "; " + "; ".join(problems)
    return not problems, detail


def _check_encoder_weights(profile_name: str | None = None) -> tuple[bool, str]:
    """Require profile-relevant local weight shards without downloading them.

    Local-max requires the configured default encoder. Studio envelopes require the published
    encoder-scale grid. This keeps a missing giant model from masquerading as a laptop hardware wall.
    """
    required: list[tuple[str, str, DictConfig]] = []
    default_cfg = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    default_name = str(OmegaConf.select(default_cfg, "defaults.encoder", default=""))
    studio_profile = str(profile_name or _infer_profile_name()).startswith("studio-")
    for path, cfg in _encoder_configs():
        if not bool(OmegaConf.select(cfg, "available", default=True)):
            continue
        name = str(OmegaConf.select(cfg, "name", default=path.stem))
        if not studio_profile and name != default_name:
            continue
        source_kind = str(OmegaConf.select(cfg, "source_kind", default="huggingface"))
        required.append((name, source_kind, cfg))
    present: list[str] = []
    missing: list[str] = []
    direct_details: list[str] = []
    for name, source_kind, cfg in required:
        if source_kind == "official_pytorch_checkpoint":
            ok, detail = _check_direct_checkpoint(name, cfg)
            direct_details.append(detail)
            (present if ok else missing).append(name if ok else f"{name}({detail})")
            continue
        hf_id = str(OmegaConf.select(cfg, "hf_id", default=""))
        files = _local_weight_files(hf_id) if hf_id else []
        (present if files else missing).append(name if files else f"{name}({hf_id or 'missing hf_id'})")
    roots = ", ".join(str(root) for root in _hf_cache_roots())
    authority = "; direct authorities: " + " | ".join(direct_details) if direct_details else ""
    if missing:
        return (
            False,
            f"local weight shards {len(present)}/{len(required)}; missing {missing}; "
            f"searched HF roots {roots}{authority}",
        )
    return True, f"local weight shards present for {present}; searched HF roots {roots}{authority}"


def _check_cache_manifests() -> tuple[bool, str]:
    """Require every on-disk latent store to pass strict citable validation."""
    root = REPO_ROOT / "data" / "cache"
    stores = sorted(path for path in root.glob("*") if path.is_dir() and (path / "meta.json").exists())
    if not stores:
        return False, "0 latent stores found; a full campaign needs at least one citable cache"
    failures: list[str] = []
    for store in stores:
        problems = validate_cache(store, citable=True)
        if problems:
            failures.append(f"{store.name}: {problems[0]}")
    if failures:
        return (
            False,
            f"{len(stores) - len(failures)}/{len(stores)} stores citable; first failures: "
            + "; ".join(failures[:3]),
        )
    return True, f"{len(stores)}/{len(stores)} stores pass citable manifest and integrity checks"


def _check_cache_write() -> tuple[bool, str]:
    cache_dir = REPO_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = cache_dir / ".studio_doctor_write_test"
    probe.write_bytes(b"ok")
    wrote = probe.read_bytes() == b"ok"
    probe.unlink()
    return wrote and not probe.exists(), f"write+delete ok under {cache_dir.relative_to(REPO_ROOT)}"


def _check_config_validation() -> tuple[bool, str]:
    from .harness.validate import check_all

    problems = check_all()
    if not problems:
        return True, "0 problems (encoders, experiments, and queued legs valid)"
    head = "; ".join(f"{p['where']}: {p['problem']}" for p in problems[:3])
    more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
    return False, f"{len(problems)} problems: {head}{more}"


def _probes(profile_name: str | None = None) -> tuple[tuple[str, Callable[[], tuple[bool, str]]], ...]:
    return (
        ("python", _check_python),
        ("package_import", _check_package_import),
        ("torch", _check_torch),
        ("apple_silicon", _check_apple_silicon),
        ("memory_telemetry", _check_memory_telemetry),
        ("disk_space", _check_disk_space),
        ("profile_host_match", lambda: _check_profile_host(profile_name)),
        ("profile_floor", lambda: _check_profile_floor(profile_name)),
        ("video_backend", _check_video_backend),
        ("huggingface", _check_huggingface),
        ("encoders", _check_encoders),
        ("encoder_weights", lambda: _check_encoder_weights(profile_name)),
        ("cache_manifests", _check_cache_manifests),
        ("cache_write", _check_cache_write),
        ("config_validation", _check_config_validation),
    )


def _host_receipt() -> dict:
    info = dict(apple_silicon_info())
    du = shutil.disk_usage(REPO_ROOT)
    info.update(
        {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "disk_total_gb": round(du.total / 1e9, 3),
            "disk_free_gb": round(du.free / 1e9, 3),
        }
    )
    return info


def _classify_failures(checks: list[dict]) -> dict[str, object]:
    failed = {str(check["name"]) for check in checks if not check["ok"]}
    groups = {
        "software": {"python", "package_import", "torch", "memory_telemetry", "video_backend", "huggingface"},
        "evidence": {"encoders", "encoder_weights", "cache_manifests", "config_validation"},
        "resource_safety": {"apple_silicon", "disk_space", "profile_host_match", "profile_floor"},
    }
    return {
        "software_blockers": sorted(failed & groups["software"]),
        "evidence_blockers": sorted(failed & groups["evidence"]),
        "resource_safety_blockers": sorted(failed & groups["resource_safety"]),
        "measured_hardware_limits": [],
        "studio_only_boundary_proven": False,
        "note": (
            "A failed dependency, missing weight, uncitable cache, or free-disk safety floor is not "
            "evidence of a compute or memory hardware limit. Hardware boundaries require experiment receipts."
        ),
    }


def doctor(profile_name: str | None = None) -> dict:
    """Run all cheap probes and return a machine-readable, no-download readiness receipt."""
    resolved_profile = profile_name or _infer_profile_name()
    checks = [_check(name, fn) for name, fn in _probes(resolved_profile)]
    passed = sum(1 for check in checks if check["ok"])
    all_ok = all(check["ok"] for check in checks)
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": {
            "requested": profile_name,
            "resolved": resolved_profile,
            "envelope": get_profile(resolved_profile).as_dict(),
        },
        "host": _host_receipt(),
        "checks": checks,
        "classification": _classify_failures(checks),
        "all_ok": all_ok,
        "summary": {
            "total": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "all_ok": all_ok,
        },
    }


def render_md(report: dict) -> str:
    """Render every check plus the measured profile decision."""
    summary = report["summary"]
    profile_name = report.get("profile", {}).get("resolved", "unknown")
    head = f"CURRENT HOST READY FOR {profile_name}" if report["all_ok"] else "CURRENT HOST NOT READY"
    lines = [
        "# Host and Studio-transfer readiness doctor",
        "",
        f"**{head}** ({summary['passed']}/{summary['total']} checks passed, {summary['failed']} failed)",
        "",
        (
            f"Resolved resource envelope: `{profile_name}`. This verdict is profile readiness, "
            "not proof that a Studio-only hardware boundary has been reached."
        ),
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | {'ok' if check['ok'] else 'FAIL'} | {detail} |")
    lines.append("")
    return "\n".join(lines)
