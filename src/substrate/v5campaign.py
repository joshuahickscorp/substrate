"""Read-only preflight, prior-version integrity, and frozen authorities for v5."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

from substrate import v5config as C

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).resolve()

PRE_TAG = C.PRE_TAG
READY_TAG = C.READY_TAG
TERMINAL_TAG = C.TERMINAL_TAG
IMPLEMENTATION_BRANCH = C.IMPLEMENTATION_BRANCH

VERSION_ROOTS = {
    "v1": ("evidence/substrate/v1", "runs/substrate/v1", "artifacts/substrate/v1", "proof"),
    "v2": ("evidence/substrate/v2", "runs/substrate/v2", "artifacts/substrate/v2", "configs/substrate/v2"),
    "v3": ("evidence/substrate/v3", "runs/substrate/v3", "artifacts/substrate/v3", "configs/substrate/v3"),
    "v4": ("evidence/substrate/v4", "runs/substrate/v4", "artifacts/substrate/v4", "configs/substrate/v4"),
}

CLASSIFICATION_FILES = {
    "v1": "evidence/substrate/v1/SUBSTRATE_NOUS_CLOSURE.json",
    "v2": "evidence/substrate/v2/SUBSTRATE_V2_FINAL_CLASSIFICATION.json",
    "v3": "evidence/substrate/v3/SUBSTRATE_V3_FINAL_CLASSIFICATION.json",
    "v4": "evidence/substrate/v4/SUBSTRATE_V4_FINAL_CLASSIFICATION.json",
}

TOOL_COMMANDS = {
    "python": (sys.executable, "--version"),
    "git": ("git", "--version"),
    "gh": ("gh", "--version"),
    "clang": ("clang", "--version"),
    "gcc": ("gcc", "--version"),
    "rustc": ("rustc", "--version"),
    "cargo": ("cargo", "--version"),
    "cmake": ("cmake", "--version"),
    "ninja": ("ninja", "--version"),
    "make": ("make", "--version"),
    "ffmpeg": ("ffmpeg", "-version"),
    "ffprobe": ("ffprobe", "-version"),
    "blender": ("blender", "--version"),
    "aria2c": ("aria2c", "--version"),
    "curl": ("curl", "--version"),
    "wget": ("wget", "--version"),
}

PYTHON_CAPABILITIES = (
    "numpy",
    "torch",
    "torchvision",
    "torchaudio",
    "cv2",
    "PIL",
    "soundfile",
    "scipy",
    "trimesh",
)

MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}

CORPUS_SUFFIXES = {
    ".aac",
    ".csv",
    ".flac",
    ".glb",
    ".gltf",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".m4a",
    ".mkv",
    ".mp3",
    ".mp4",
    ".obj",
    ".parquet",
    ".ply",
    ".png",
    ".tsv",
    ".wav",
    ".webm",
}

PREFLIGHT_GENERATED_ROOTS = ("artifacts/substrate/v5",)

PRINCIPAL_RUNTIME_ROOTS = (
    "artifacts/substrate/v5",
    "cache/substrate/v5",
    "evidence/substrate/v5",
    "runs/substrate/v5",
)


def _v5io() -> ModuleType:
    """Load the v5 writer only at a write boundary.

    Importing this module and running inventory/preflight remain possible before
    the v5 I/O module has been installed.
    """

    return importlib.import_module("substrate.v5io")


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _optional_command(arguments: list[str], *, timeout: float = 8.0) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            arguments,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": type(error).__name__,
            "runtime_seconds": time.monotonic() - started,
        }
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "runtime_seconds": time.monotonic() - started,
    }


def _porcelain_entries(output: str) -> list[dict[str, str]]:
    """Parse ``git status --porcelain=v1 -z`` without trusting path quoting."""

    fields = output.split("\0")
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 4 or field[2] != " ":
            entries.append({"status": "!!", "path": field})
            continue
        status = field[:2]
        row = {"status": status, "path": field[3:]}
        if "R" in status or "C" in status:
            if index >= len(fields) or not fields[index]:
                row["source_path"] = ""
            else:
                row["source_path"] = fields[index]
                index += 1
        entries.append(row)
    return entries


def _beneath_declared_root(path: str, roots: tuple[str, ...]) -> bool:
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        return False
    posix = normalized.as_posix().removeprefix("./")
    return any(posix == root.rstrip("/") or posix.startswith(f"{root.rstrip('/')}/") for root in roots)


def worktree_cleanliness(
    allowed_roots: tuple[str, ...],
    *,
    status_output: str | None = None,
) -> dict:
    """Require every dirty path to be beneath an explicitly declared root."""

    if status_output is not None:
        status = {
            "available": True,
            "returncode": 0,
            "stdout": status_output,
            "stderr": "",
        }
    else:
        try:
            result = subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            status = {
                "available": True,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except OSError as error:
            status = {
                "available": False,
                "returncode": None,
                "stdout": "",
                "stderr": str(error),
            }
    entries = _porcelain_entries(str(status["stdout"]))
    dirty_paths = [path for row in entries for path in (row["path"], row.get("source_path")) if path]
    undeclared = sorted({path for path in dirty_paths if not _beneath_declared_root(path, allowed_roots)})
    return {
        "command_succeeded": status["returncode"] == 0,
        "allowed_roots": list(allowed_roots),
        "entries": entries,
        "dirty_paths": dirty_paths,
        "undeclared_dirty_paths": undeclared,
        "clean_except_allowed_roots": (status["returncode"] == 0 and not undeclared),
        "activation": False,
    }


def _ref_or_none(reference: str) -> str | None:
    result = _optional_command(["git", "rev-parse", "--verify", reference])
    return result["stdout"] if result["returncode"] == 0 else None


def _remote_ref(reference: str) -> str | None:
    result = _optional_command(["git", "ls-remote", "origin", reference], timeout=20.0)
    if result["returncode"] != 0 or not result["stdout"]:
        return None
    return result["stdout"].splitlines()[0].split()[0]


def _remote_tag_refs() -> dict[str, dict[str, str | None]]:
    result = _optional_command(["git", "ls-remote", "--tags", "origin"], timeout=20.0)
    if result["returncode"] != 0:
        return {}
    raw: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].startswith("refs/tags/"):
            raw[fields[1].removeprefix("refs/tags/")] = fields[0]
    names = {name.removesuffix("^{}") for name in raw}
    return {
        name: {
            "tag_object": raw.get(name),
            "peeled_commit": raw.get(f"{name}^{{}}"),
        }
        for name in names
    }


def _tag_snapshot(tag: str, remote: dict[str, dict[str, str | None]]) -> dict:
    tag_object = _ref_or_none(f"refs/tags/{tag}")
    peeled_commit = _ref_or_none(f"refs/tags/{tag}^{{}}")
    object_type = None
    if tag_object is not None:
        result = _optional_command(["git", "cat-file", "-t", tag_object])
        object_type = result["stdout"] if result["returncode"] == 0 else None
    remote_row = remote.get(tag, {})
    return {
        "name": tag,
        "tag_object": tag_object,
        "peeled_commit": peeled_commit,
        "object_type": object_type,
        "annotated": object_type == "tag",
        "remote_tag_object": remote_row.get("tag_object"),
        "remote_peeled_commit": remote_row.get("peeled_commit"),
        "tag_object_matches_remote": tag_object is not None and tag_object == remote_row.get("tag_object"),
        "peeled_commit_matches_remote": peeled_commit is not None and peeled_commit == remote_row.get("peeled_commit"),
    }


def _sha_obj(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_tree_entries(tag: str, roots: tuple[str, ...]) -> dict[str, dict[str, str]]:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", tag, "--", *roots],
        cwd=ROOT,
    )
    entries: dict[str, dict[str, str]] = {}
    for raw_line in output.split(b"\0"):
        if not raw_line:
            continue
        metadata, raw_path = raw_line.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        entries[os.fsdecode(raw_path)] = {
            "mode": mode,
            "kind": kind,
            "object": object_id,
        }
    return entries


def _local_file_paths(roots: tuple[str, ...]) -> set[str]:
    paths: set[str] = set()
    for relative_root in roots:
        root = ROOT / relative_root
        if root.is_file() or root.is_symlink():
            paths.add(relative_root)
        elif root.is_dir():
            paths.update(path.relative_to(ROOT).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink())
    return paths


def _hash_local_paths(paths: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for offset in range(0, len(paths), 200):
        batch = paths[offset : offset + 200]
        if not batch:
            continue
        output = subprocess.check_output(
            ["git", "hash-object", "--no-filters", "--", *batch],
            cwd=ROOT,
            text=True,
        ).splitlines()
        hashes.update(dict(zip(batch, output, strict=True)))
    return hashes


def _local_mode(path: Path) -> str:
    if path.is_symlink():
        return "120000"
    return "100755" if path.stat().st_mode & 0o111 else "100644"


def _tree_integrity(tag: str, roots: tuple[str, ...]) -> dict:
    expected = _git_tree_entries(tag, roots)
    local_paths = _local_file_paths(roots)
    local_hashes = _hash_local_paths(sorted(local_paths))

    objects: dict[str, dict[str, object]] = {}
    drift: list[str] = []
    for path, tag_row in expected.items():
        local_path = ROOT / path
        local_object = local_hashes.get(path)
        local_mode = _local_mode(local_path) if path in local_paths else None
        byte_identical = local_object == tag_row["object"]
        mode_identical = local_mode == tag_row["mode"]
        row = {
            "tag_mode": tag_row["mode"],
            "tag_kind": tag_row["kind"],
            "tag_object": tag_row["object"],
            "local_mode": local_mode,
            "local_object": local_object,
            "byte_identical": byte_identical,
            "mode_identical": mode_identical,
        }
        objects[path] = row
        if not byte_identical or not mode_identical:
            drift.append(path)

    unexpected = sorted(local_paths - set(expected))
    tag_manifest = {path: {"mode": row["mode"], "object": row["object"]} for path, row in expected.items()}
    local_tag_owned_manifest = {
        path: {
            "mode": _local_mode(ROOT / path),
            "object": local_hashes[path],
        }
        for path in sorted(local_paths & set(expected))
    }
    additions_manifest = {
        path: {
            "mode": _local_mode(ROOT / path),
            "object": local_hashes[path],
        }
        for path in unexpected
    }
    git_tree_objects = {root: _ref_or_none(f"{tag}:{root}") for root in roots}
    return {
        "tag": tag,
        "tag_object": _ref_or_none(f"refs/tags/{tag}"),
        "peeled_commit": _ref_or_none(f"refs/tags/{tag}^{{}}"),
        "roots": list(roots),
        "git_tree_objects": git_tree_objects,
        "tag_tree_hash": _sha_obj(tag_manifest),
        "local_tree_hash": _sha_obj(local_tag_owned_manifest),
        "tree_hashes_match": _sha_obj(tag_manifest) == _sha_obj(local_tag_owned_manifest),
        "local_additions_hash": _sha_obj(additions_manifest),
        "object_count": len(objects),
        "objects": objects,
        "drift": sorted(drift),
        "unexpected": unexpected,
        "local_additions_do_not_mutate_tag_owned_bytes": True,
        "byte_identical": not drift,
    }


def _seal_validation(version: str) -> dict:
    rows: dict[str, dict[str, object]] = {}
    for family in ("evidence", "artifacts", "configs"):
        root = ROOT / family / "substrate" / version
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            relative = path.relative_to(ROOT).as_posix()
            try:
                document = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                rows[relative] = {
                    "json_valid": False,
                    "seal_present": False,
                    "seal_valid": False,
                    "activation_false": False,
                    "error": type(error).__name__,
                }
                continue
            declared = document.get("sha256")
            expected = _sha_obj({key: value for key, value in document.items() if key != "sha256"})
            rows[relative] = {
                "json_valid": True,
                "seal_present": declared is not None,
                "seal_valid": declared is None or declared == expected,
                "activation_false": document.get("activation") is False,
            }
    failed = sorted(path for path, row in rows.items() if not row["json_valid"] or not row["seal_valid"] or not row["activation_false"])
    return {
        "documents": rows,
        "document_count": len(rows),
        "sealed_document_count": sum(bool(row["seal_present"]) for row in rows.values()),
        "failed": failed,
        "all_valid": not failed,
    }


def _classification_snapshot() -> dict:
    rows: dict[str, dict[str, object]] = {}
    for version, relative in CLASSIFICATION_FILES.items():
        path = ROOT / relative
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            rows[version] = {
                "path": relative,
                "classification": None,
                "expected": C.PRIOR_CLASSIFICATIONS[version],
                "preserved": False,
                "error": type(error).__name__,
            }
            continue
        classification = document.get("classification")
        if classification is None and isinstance(document.get("verdict"), dict):
            classification = document["verdict"].get("classification")
        rows[version] = {
            "path": relative,
            "classification": classification,
            "expected": C.PRIOR_CLASSIFICATIONS[version],
            "preserved": classification == C.PRIOR_CLASSIFICATIONS[version],
        }
    return rows


def immutability() -> dict:
    """Verify every prior tag, terminal tree, seal, and earned classification."""

    remote = _remote_tag_refs()
    tags = {tag: _tag_snapshot(tag, remote) for tag in C.PRIOR_TAGS}
    listed = set(_git("tag", "--list", "substrate-v[1-4]-*").splitlines())
    trees = {version: _tree_integrity(C.TERMINAL_TAGS[version], roots) for version, roots in VERSION_ROOTS.items()}
    seals = {version: _seal_validation(version) for version in VERSION_ROOTS}
    classifications = _classification_snapshot()
    checks = {
        "prior_tag_set_exact": listed == set(C.PRIOR_TAGS),
        "all_prior_tags_annotated": all(row["annotated"] for row in tags.values()),
        "remote_prior_tags_resolved": bool(remote) and all(tag in remote for tag in C.PRIOR_TAGS),
        "tag_objects_match_remote": all(row["tag_object_matches_remote"] for row in tags.values()),
        "peeled_commits_match_remote": all(row["peeled_commit_matches_remote"] for row in tags.values()),
        "v1_tree_byte_identical": trees["v1"]["byte_identical"],
        "v2_tree_byte_identical": trees["v2"]["byte_identical"],
        "v3_tree_byte_identical": trees["v3"]["byte_identical"],
        "v4_tree_byte_identical": trees["v4"]["byte_identical"],
        "v1_seals_valid": seals["v1"]["all_valid"],
        "v2_seals_valid": seals["v2"]["all_valid"],
        "v3_seals_valid": seals["v3"]["all_valid"],
        "v4_seals_valid": seals["v4"]["all_valid"],
        "prior_classifications_preserved": all(row["preserved"] for row in classifications.values()),
        "activation_false": C.ACTIVATION is False,
    }
    return {
        "schema": "substrate-v5-v1-v2-v3-v4-immutability/v1",
        "tag_authority": {
            "expected": list(C.PRIOR_TAGS),
            "listed": sorted(listed),
            "tags": tags,
            "remote_resolved": bool(remote),
        },
        "trees": trees,
        "seals": seals,
        "classifications": classifications,
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "all_pass": all(checks.values()),
        "read_only": True,
        "activation": False,
    }


def _tool_snapshot() -> dict:
    rows: dict[str, dict[str, object]] = {}
    for name, command in TOOL_COMMANDS.items():
        executable = command[0] if name != "python" else sys.executable
        path = shutil.which(executable) if executable != sys.executable else sys.executable
        if path is None:
            rows[name] = {"available": False, "path": None, "version": None}
            continue
        result = _optional_command(list(command), timeout=5.0)
        output = result["stdout"] or result["stderr"]
        rows[name] = {
            "available": result["returncode"] == 0,
            "path": path,
            "version": output.splitlines()[0] if output else None,
        }
    return rows


def _bounded_path_snapshot(path: Path, relevant_suffixes: set[str], *, limit: int = 2_048) -> dict:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "sampled_files": 0,
            "relevant_files": 0,
            "sampled_bytes": 0,
            "truncated": False,
            "examples": [],
        }
    files = 0
    relevant = 0
    sampled_bytes = 0
    examples: list[str] = []
    truncated = False
    for directory, _, names in os.walk(path):
        for name in sorted(names):
            candidate = Path(directory) / name
            files += 1
            with contextlib.suppress(OSError):
                sampled_bytes += candidate.stat().st_size
            if candidate.suffix.lower() in relevant_suffixes:
                relevant += 1
                if len(examples) < 40:
                    examples.append(candidate.relative_to(path).as_posix())
            if files >= limit:
                truncated = True
                break
        if truncated:
            break
    return {
        "path": str(path),
        "exists": True,
        "sampled_files": files,
        "relevant_files": relevant,
        "sampled_bytes": sampled_bytes,
        "truncated": truncated,
        "examples": examples,
    }


def _model_snapshot() -> dict:
    home = Path.home()
    candidates = (
        ROOT / "models",
        ROOT / "cache" / "substrate" / "v5",
        Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else home / ".cache" / "huggingface",
        Path(os.environ["OLLAMA_MODELS"]) if os.environ.get("OLLAMA_MODELS") else home / ".ollama" / "models",
    )
    unique = tuple(dict.fromkeys(path.resolve() for path in candidates))
    return {
        "roots": [_bounded_path_snapshot(path, MODEL_SUFFIXES) for path in unique],
        "checkpoint_suffixes": sorted(MODEL_SUFFIXES),
        "inventory_only": True,
        "models_loaded": 0,
        "models_modified": 0,
    }


def _corpus_snapshot() -> dict:
    candidates = (
        ROOT / "data",
        ROOT / "cache",
        ROOT / "models",
    )
    return {
        "roots": [_bounded_path_snapshot(path.resolve(), CORPUS_SUFFIXES) for path in candidates],
        "media_and_corpus_suffixes": sorted(CORPUS_SUFFIXES),
        "inventory_only": True,
        "files_modified": 0,
    }


def _resource_snapshot() -> dict:
    disk = shutil.disk_usage(ROOT)
    memory_result = _optional_command(["memory_pressure", "-Q"])
    memory_match = re.search(r"System-wide memory free percentage:\s*(\d+)%", memory_result["stdout"])
    swap_result = _optional_command(["sysctl", "-n", "vm.swapusage"])
    swap = {
        key: float(value)
        for key, value in re.findall(
            r"(total|used|free)\s*=\s*([0-9.]+)M",
            swap_result["stdout"],
        )
    }
    cpu_result = _optional_command(["ps", "-A", "-o", "%cpu="])
    cpu = sum(float(value) for value in cpu_result["stdout"].split() if value.replace(".", "", 1).isdigit())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "disk_total_gib": disk.total / 1024**3,
        "disk_used_gib": disk.used / 1024**3,
        "disk_available_gib": disk.free / 1024**3,
        "memory_free_percent": int(memory_match.group(1)) if memory_match else None,
        "swap_total_mib": swap.get("total"),
        "swap_used_mib": swap.get("used"),
        "swap_free_mib": swap.get("free"),
        "aggregate_cpu_percent": cpu,
        "inventory_peak_rss_mib": usage.ru_maxrss / 1024**2 if sys.platform == "darwin" else usage.ru_maxrss / 1024,
        "machine": {
            "processor": platform.processor() or "unknown",
            "logical_cores": os.cpu_count(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def _hardware_snapshot() -> dict:
    display = _optional_command(["system_profiler", "SPDisplaysDataType", "-json"], timeout=15.0)
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cores": os.cpu_count(),
        "display_inventory": (json.loads(display["stdout"]) if display["returncode"] == 0 and display["stdout"].startswith("{") else None),
        "python_modules": {name: importlib.util.find_spec(name) is not None for name in PYTHON_CAPABILITIES},
    }


def _process_snapshots() -> dict:
    result = _optional_command(
        ["ps", "axo", "pid=,ppid=,stat=,%cpu=,%mem=,rss=,etime=,command="],
    )
    hawking: list[dict[str, object]] = []
    v5_workers: list[dict[str, object]] = []
    for line in result["stdout"].splitlines():
        fields = line.strip().split(None, 7)
        if len(fields) != 8:
            continue
        pid, ppid, status, cpu, memory, rss, elapsed, command = fields
        if int(pid) == os.getpid():
            continue
        row = {
            "pid": int(pid),
            "ppid": int(ppid),
            "status": status,
            "cpu_percent": float(cpu),
            "memory_percent": float(memory),
            "rss_mib": int(rss) / 1024,
            "elapsed": elapsed,
            "command": command,
        }
        lowered = command.lower()
        if "/hawking/" in lowered or re.search(r"(^|[/\s])hawking([/\s]|$)", lowered):
            hawking.append(row)
        if re.search(r"\bsubstrate\s+v5\s+(run|resume)\b", lowered):
            v5_workers.append(row)
    return {
        "hawking": {
            "processes": hawking,
            "active_process_count": len(hawking),
            "observation_only": True,
            "signals_sent": 0,
            "processes_modified": 0,
            "controllers_modified": 0,
            "mps_adopted": False,
        },
        "v5_workers": {
            "processes": v5_workers,
            "active_process_count": len(v5_workers),
            "observation_only": True,
            "signals_sent": 0,
        },
    }


def _v5_namespace_snapshot() -> dict:
    families = ("configs", "evidence", "runs", "artifacts", "models", "data", "cache")
    rows: dict[str, dict[str, object]] = {}
    for family in families:
        path = ROOT / family / "substrate" / "v5"
        files = sorted(item.relative_to(ROOT).as_posix() for item in path.rglob("*") if item.is_file()) if path.exists() else []
        rows[family] = {
            "path": str(path),
            "exists": path.exists(),
            "file_count": len(files),
            "files": files[:100],
            "truncated": len(files) > 100,
        }
    principal = ROOT / "runs" / "substrate" / "v5" / "principal"
    principal_files = sorted(path.relative_to(ROOT).as_posix() for path in principal.rglob("*") if path.is_file()) if principal.exists() else []
    return {
        "families": rows,
        "principal_path": str(principal),
        "principal_files": principal_files,
        "principal_exists": bool(principal_files),
    }


def _network_snapshot() -> dict:
    remote = _optional_command(["git", "remote", "get-url", "origin"])
    interfaces = _optional_command(["ifconfig", "-l"])
    return {
        "origin": remote["stdout"] if remote["returncode"] == 0 else None,
        "interfaces": interfaces["stdout"].split() if interfaces["returncode"] == 0 else [],
        "proxy_configured": {key: bool(os.environ.get(key)) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")},
        "active_bandwidth_probe_performed": False,
        "download_benchmark_required_before_acquisition": True,
        "constraints": "respect source terms, bounded concurrency, resumable transfers, and host limits",
    }


def inventory() -> dict:
    """Take a read-only local capability, process, storage, and corpus snapshot."""

    processes = _process_snapshots()
    namespaces = _v5_namespace_snapshot()
    return {
        "schema": "substrate-v5-local-inventory/v1",
        "repository": str(ROOT),
        "tools": _tool_snapshot(),
        "hardware": _hardware_snapshot(),
        "resources": _resource_snapshot(),
        "storage": namespaces,
        "network": _network_snapshot(),
        "models": _model_snapshot(),
        "corpora": _corpus_snapshot(),
        "processes": processes,
        "v5_principal": {
            "files": namespaces["principal_files"],
            "workers": processes["v5_workers"]["processes"],
            "pre_existing": namespaces["principal_exists"] or processes["v5_workers"]["active_process_count"] > 0,
        },
        "read_only": True,
        "files_written": 0,
        "processes_modified": 0,
        "activation": False,
    }


def preflight(
    *,
    inventory_snapshot: dict | None = None,
    integrity_snapshot: dict | None = None,
) -> dict:
    """Resolve the v5 entry state without creating a tag, branch, or file."""

    local_inventory = inventory_snapshot if inventory_snapshot is not None else inventory()
    integrity = integrity_snapshot if integrity_snapshot is not None else immutability()
    remote_tags = _remote_tag_refs()
    pre_tag = _tag_snapshot(PRE_TAG, remote_tags)
    main = _ref_or_none("refs/heads/main")
    origin_main = _ref_or_none("refs/remotes/origin/main")
    remote_main = _remote_ref("refs/heads/main")
    head = _ref_or_none("HEAD")
    branch_result = _optional_command(["git", "branch", "--show-current"])
    branch = branch_result["stdout"] if branch_result["returncode"] == 0 else None
    cleanliness = worktree_cleanliness(PREFLIGHT_GENERATED_ROOTS)
    worktree_result = _optional_command(["git", "worktree", "list", "--porcelain"])
    ancestor_result = (
        _optional_command(["git", "merge-base", "--is-ancestor", pre_tag["peeled_commit"], head]) if pre_tag["peeled_commit"] and head else {"returncode": None}
    )
    terminal_v4 = integrity.get("tag_authority", {}).get("tags", {}).get(C.TERMINAL_TAGS["v4"], {})
    resources = local_inventory["resources"]
    hawking = local_inventory["processes"]["hawking"]
    checks = {
        "pre_tag_annotated": pre_tag["annotated"],
        "pre_tag_object_matches_remote": pre_tag["tag_object_matches_remote"],
        "pre_tag_peeled_matches_remote": pre_tag["peeled_commit_matches_remote"],
        "pre_tag_matches_v4_terminal_commit": pre_tag["peeled_commit"] == terminal_v4.get("peeled_commit"),
        "pre_tag_matches_main": pre_tag["peeled_commit"] == main,
        "main_matches_origin_main": main is not None and main == origin_main == remote_main,
        "implementation_branch_active": branch == IMPLEMENTATION_BRANCH,
        "current_head_descends_from_pre_tag": ancestor_result.get("returncode") == 0,
        "one_worktree": worktree_result["stdout"].count("worktree ") == 1,
        "v1_v2_v3_v4_immutable": integrity["all_pass"],
        "no_v5_principal_run": not local_inventory["v5_principal"]["pre_existing"],
        "no_v5_worker": not local_inventory["v5_principal"]["workers"],
        "activation_false": C.ACTIVATION is False,
        "disk_safe_for_development": resources["disk_available_gib"] >= 25,
        "hawking_observation_only": (
            hawking["observation_only"] and hawking["signals_sent"] == 0 and hawking["processes_modified"] == 0 and hawking["controllers_modified"] == 0
        ),
        "inventory_read_only": (local_inventory["read_only"] and local_inventory["files_written"] == 0 and local_inventory["processes_modified"] == 0),
        "worktree_clean_except_preflight_authorities": cleanliness["clean_except_allowed_roots"],
    }
    return {
        "schema": "substrate-v5-preflight/v1",
        "repository": str(ROOT),
        "entry": {
            "implementation_branch": IMPLEMENTATION_BRANCH,
            "branch": branch,
            "head": head,
            "main": main,
            "origin_main": origin_main,
            "remote_main": remote_main,
            "pre_tag": pre_tag,
            "ready_tag": READY_TAG,
            "terminal_tag": TERMINAL_TAG,
            "dirty": cleanliness["entries"],
            "cleanliness": cleanliness,
            "worktrees": worktree_result["stdout"].splitlines(),
        },
        "inventory": local_inventory,
        "immutability_digest": _sha_obj(integrity),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "all_pass": all(checks.values()),
        "read_only": True,
        "activation": False,
    }


def _speed_constitution() -> dict:
    return {
        "schema": "substrate-v5-speed-constitution/v1",
        "default": "bounded parallel execution when work units are independent and scientific semantics permit",
        "worker_benchmarks": [1, 2, 4, 8, 12, 16, "higher_when_safe"],
        "requirements": [
            "measure native-thread budgets per worker",
            "detect oversubscription",
            "batch small compatible operations",
            "reuse immutable models only after reset and parity are proven",
            "use content-addressed caches",
            "centralize authoritative publication",
            "preserve determinism, recovery, isolation, integrity, auditability, and claim boundaries",
        ],
        "major_operation_threshold_percent": 5,
        "activation": False,
    }


def _download_authority() -> dict:
    return {
        "schema": "substrate-v5-download-authority/v1",
        "policy": [
            "discover authoritative and permitted mirror sources",
            "record license, version, expected size, and hashes",
            "benchmark bounded segmented and parallel transfer where supported",
            "use resumable downloads",
            "stream hashes where practical",
            "download independent files concurrently",
            "pipeline or parallelize decompression when safe",
            "deduplicate by content hash",
            "preserve immutable raw source bytes",
            "separate acquisition from scientific preprocessing",
        ],
        "source_terms_override_speed": True,
        "active_downloads_during_preflight": 0,
        "activation": False,
    }


def _preflight_authorities(entry: dict, integrity: dict, local_inventory: dict) -> dict[str, dict]:
    resources = local_inventory["resources"]
    hawking = local_inventory["processes"]["hawking"]
    return {
        "SUBSTRATE_V5_PREFLIGHT.json": entry,
        "SUBSTRATE_V5_V1_V2_V3_V4_IMMUTABILITY.json": integrity,
        "SUBSTRATE_V5_HAWKING_COEXISTENCE.json": {
            "schema": "substrate-v5-hawking-coexistence/v1",
            "policy": "observe only; never signal, pause, restart, modify, or adopt Hawking",
            "snapshot": hawking,
            "resources": resources,
            "principal_requires_resource_rehearsal": True,
            "activation": False,
        },
        "SUBSTRATE_V5_LOCAL_CAPABILITY_INVENTORY.json": {
            "schema": "substrate-v5-local-capability-inventory/v1",
            "tools": local_inventory["tools"],
            "hardware": local_inventory["hardware"],
            "models": local_inventory["models"],
            "corpora": local_inventory["corpora"],
            "read_only": True,
            "activation": False,
        },
        "SUBSTRATE_V5_STORAGE_AND_NETWORK_PLAN.json": {
            "schema": "substrate-v5-storage-and-network-plan/v1",
            "storage": local_inventory["storage"],
            "network": local_inventory["network"],
            "available_gib": resources["disk_available_gib"],
            "minimum_development_free_gib": 25,
            "projected_growth_must_be_frozen_before_acquisition": True,
            "activation": False,
        },
        "SUBSTRATE_V5_SPEED_CONSTITUTION.json": _speed_constitution(),
        "SUBSTRATE_V5_DOWNLOAD_AUTHORITY.json": _download_authority(),
        "SUBSTRATE_V5_RESOURCE_TELEMETRY.json": {
            "schema": "substrate-v5-resource-telemetry/v1",
            "stage": "preflight",
            "resources": resources,
            "processes": local_inventory["processes"],
            "read_only": True,
            "activation": False,
        },
        "SUBSTRATE_V5_PERFORMANCE_LEDGER.json": {
            "schema": "substrate-v5-performance-ledger/v1",
            "stage": "preflight",
            "required_metrics": [
                "wall time",
                "CPU time and utilization",
                "accelerator utilization and memory",
                "peak RAM",
                "disk reads and writes",
                "process and thread count",
                "cache hit rate",
                "model startup and decode cost",
                "transfer cost",
                "failures and retries",
            ],
            "scientific_performance_claims": [],
            "activation": False,
        },
    }


def _seal_documents(documents: dict[str, dict], *, artifact: bool) -> list[str]:
    io = _v5io()
    if getattr(io, "ACTIVATION", False) is not False:
        raise RuntimeError("v5 I/O authority does not preserve activation=false")
    sealed: list[str] = []
    for name, document in documents.items():
        io.seal(name, document, artifact=artifact)
        sealed.append(name)
    return sealed


def seal_preflight() -> dict:
    """Inventory once, then publish the nine required entry authorities via v5io."""

    local_inventory = inventory()
    integrity = immutability()
    entry = preflight(
        inventory_snapshot=local_inventory,
        integrity_snapshot=integrity,
    )
    documents = _preflight_authorities(entry, integrity, local_inventory)
    sealed = _seal_documents(documents, artifact=True)
    return {
        "preflight": entry,
        "integrity": integrity,
        "inventory": local_inventory,
        "authorities": documents,
        "sealed": sealed,
        "all_pass": entry["all_pass"],
        "activation": False,
    }


def _classification_authority() -> dict:
    return {
        "schema": "substrate-v5-classification-authority/v1",
        "ordered_levels": list(C.CLASSIFICATIONS),
        "requirements": {
            "multimodal_cognitive_substrate": [
                "permanent state",
                "at least six integrated modalities",
                "model-neutral fabric",
                "cross-modal binding",
                "object and event continuity",
                "v4 capability preservation",
                "independent verification",
            ],
            "persistent_sensorium": [
                "continuous sensory state",
                "object permanence",
                "event continuity",
                "audiovisual binding",
                "spatial or 3D state",
                "active perception",
                "sensor interruption recovery",
            ],
            "integrated_model_organism_architecture": [
                "multimodal cognitive substrate",
                "persistent sensorium",
                "multiple independently useful models",
                "positive routing and model support",
                "model replacement continuity",
                "positive continual learning and body schema",
            ],
            "persistent_embodied_proto_nous_candidate": [
                "v4 functional proto-Nous preserved",
                "integrated model-organism architecture",
                "positive active perception, cross-modal transfer, spatial understanding, body-state knowledge, and human teaching",
                "positive long-history entity advantage",
                "cognitive integrity and terminal principal evidence",
            ],
            "multimodal_nous_ready_for_review": [
                "persistent embodied proto-Nous candidate",
                "independent replication",
                "generator-held-out multimodal evaluation",
                "multiple model families, sensor environments, and bodies",
                "zero surviving mutations",
                "complete review package",
            ],
        },
        "maximum_automatic_classification": C.CLAIM_BOUNDARY["maximum_automatic_classification"],
        "unqualified_nous": False,
        "activation": False,
    }


def freeze() -> dict:
    """Publish the campaign constitution without admitting or launching a run."""

    configuration = C.configuration()
    documents = {
        "SUBSTRATE_V5_SCIENTIFIC_CONSTITUTION.json": {
            "schema": "substrate-v5-scientific-constitution/v1",
            "objective": "a permanent multimodal cognitive substrate whose identity and state persist independently of any one model",
            "master_plan_sha256": C.MASTER_PLAN_SHA256,
            "execution_brief_sha256": C.EXECUTION_BRIEF_SHA256,
            "hypotheses": C.HYPOTHESES,
            "phases": list(C.PHASES),
            "arms": list(C.ARMS),
            "modalities": list(C.MODALITIES),
            "model_roles": list(C.MODEL_ROLES),
            "model_independence_required": True,
            "sesoi": C.SESOI,
            "statistics": C.STATISTICS,
            "principal_bounds": C.PRINCIPAL_BOUNDS,
            "freeze_rule": "no source, model, corpus, split, seed, threshold, or scientific premise changes after principal launch",
            "activation": False,
        },
        "SUBSTRATE_V5_HYPOTHESIS_GRAPH.json": {
            "schema": "substrate-v5-hypothesis-graph/v1",
            "hypotheses": C.HYPOTHESES,
            "required_count": 15,
            "activation": False,
        },
        "SUBSTRATE_V5_CLASSIFICATION_AUTHORITY.json": _classification_authority(),
        "SUBSTRATE_V5_CLAIM_BOUNDARY.json": {
            "schema": "substrate-v5-claim-boundary/v1",
            **C.CLAIM_BOUNDARY,
        },
        "SUBSTRATE_V5_STATISTICAL_AUTHORITY.json": {
            "schema": "substrate-v5-statistical-authority/v1",
            **C.STATISTICS,
            "activation": False,
        },
        "SUBSTRATE_V5_CANDIDATE_LADDERS.json": {
            "schema": "substrate-v5-candidate-ladders/v1",
            "ladders": {name: list(values) for name, values in C.CANDIDATE_LADDERS.items()},
            "selection_freezes_before_admission": True,
            "unbounded_tuning_forbidden": True,
            "activation": False,
        },
    }
    io = _v5io()
    if getattr(io, "ACTIVATION", False) is not False:
        raise RuntimeError("v5 I/O authority does not preserve activation=false")
    if hasattr(io, "config_json"):
        io.config_json("frozen_configuration.json", configuration)
        io.config_json(
            "candidate_ladders.json",
            {
                "candidate_ladders": {name: list(values) for name, values in C.CANDIDATE_LADDERS.items()},
                "activation": False,
            },
        )
    sealed = _seal_documents(documents, artifact=False)
    return {
        "configuration": configuration,
        "authorities": documents,
        "sealed": sealed,
        "activation": False,
    }
