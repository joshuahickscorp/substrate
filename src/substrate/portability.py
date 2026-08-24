"""Portability manifest, host verification, and safe restore for Substrate/Odyssey.

The working tree must remain restorable after a folder copy to another path or
machine. This module records everything required that is not inside the repo,
checks the current host against that record, and performs only the restores that
are safe without sudo or an interactive app installer.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "SUBSTRATE_PORTABILITY_MANIFEST/v1"
PROGRAM = "substrate-odyssey-portability-v1"
PLAN = Path("plans/substrate/tangible_next_launch")
MANIFEST_REL = PLAN / "SUBSTRATE_PORTABILITY_MANIFEST.json"
TOOL_INVENTORY_REL = PLAN / "ODYSSEY_TOOL_PANEL_INVENTORY.json"
FROZEN_BUILD_REL = PLAN / "ODYSSEY_FROZEN_BUILD.json"
SOURCE_SELECTION_REL = PLAN / "ODYSSEY_SOURCE_SELECTION.sealed.v2.json"
CANARY_TEMPLATE_REL = PLAN / "ODYSSEY_PUBLIC_MODEL_CANARY.template.json"
CORPUS_ACQUISITION_REL = PLAN / "ODYSSEY_CORPUS_ACQUISITION.json"
PREFETCH_REL = Path("data/substrate/tangible_sandbox/prefetch/odyssey-public-v1")

STATUS_MATCHING = "present-and-matching"
STATUS_DRIFTED = "present-but-drifted"
STATUS_MISSING = "missing"

PINNED_OLLAMA_MODELS = (
    "gpt-oss:20b",
    "qwen3:30b",
    "deepseek-r1:32b",
    "nomic-embed-text",
)

# Odyssey tool panel extras beyond pyproject [dev]. Versions are remeasured at generate time.
ODYSSEY_VENV_EXTRAS = (
    "openai-whisper",
    "python-docx",
    "openpyxl",
    "pypdf",
    "torch",
)

# Tools required outside a pure Python install. Paths may be absolute (host) or
# repo-relative (in-tree .venv). generate records measured absolute paths and digests.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "id": "Lean_or_proof_checker",
        "install_method": "elan",
        "path_candidates": [
            "~/.elan/toolchains/leanprover--lean4---v4.33.0-rc1/bin/lean",
            "~/.elan/bin/lean",
        ],
        "which_names": ["lean"],
        "version_argv": ["--version"],
        "reinstall_command": (
            "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh "
            "&& elan toolchain install leanprover/lean4:v4.33.0-rc1"
        ),
        "needs_human": True,
        "human_reason": "elan toolchain install requires network and an interactive elan bootstrap on a rebuilt machine",
    },
    {
        "id": "SMT_or_SAT_solver",
        "install_method": "homebrew_formula",
        "path_candidates": [
            "/opt/homebrew/Cellar/z3/4.16.0/bin/z3",
            "/opt/homebrew/bin/z3",
            "/usr/local/bin/z3",
        ],
        "which_names": ["z3"],
        "version_argv": ["--version"],
        "reinstall_command": "brew install z3",
        "needs_human": True,
        "human_reason": "homebrew install requires network; not executed by restore",
    },
    {
        "id": "Python_and_code_execution",
        "install_method": "venv_package",
        "path_candidates": [".venv/bin/python"],
        "which_names": [],
        "version_argv": ["--version"],
        "reinstall_command": "python -m substrate.portability restore",
        "needs_human": False,
        "human_reason": "",
        "in_repo": True,
    },
    {
        "id": "repository_tests",
        "install_method": "venv_package",
        "path_candidates": [".venv/bin/pytest"],
        "artifact_globs": [".venv/lib/python*/site-packages/pytest-*.dist-info/RECORD"],
        "which_names": [],
        "version_argv": ["--version"],
        "reinstall_command": "python -m substrate.portability restore",
        "needs_human": False,
        "human_reason": "",
        "in_repo": True,
    },
    {
        "id": "document_and_spreadsheet_tools",
        "install_method": "venv_package",
        "path_candidates": [],
        "artifact_globs": [".venv/lib/python*/site-packages/python_docx-*.dist-info/RECORD"],
        "which_names": [],
        "version_argv": None,
        "reinstall_command": "python -m substrate.portability restore",
        "needs_human": False,
        "human_reason": "",
        "in_repo": True,
        "package_names": ["python-docx", "openpyxl", "pypdf"],
    },
    {
        "id": "image_video_audio_decoders",
        "install_method": "homebrew_formula",
        "path_candidates": [
            "/opt/homebrew/Cellar/ffmpeg/8.1.2_1/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ],
        "which_names": ["ffmpeg"],
        "version_argv": ["-version"],
        "reinstall_command": "brew install ffmpeg",
        "needs_human": True,
        "human_reason": "homebrew install requires network; not executed by restore",
    },
    {
        "id": "speech_tools",
        "install_method": "venv_package",
        "path_candidates": [],
        "artifact_globs": [".venv/lib/python*/site-packages/openai_whisper-*.dist-info/RECORD"],
        "which_names": [],
        "version_argv": None,
        "reinstall_command": "python -m substrate.portability restore",
        "needs_human": False,
        "human_reason": "",
        "in_repo": True,
        "package_names": ["openai-whisper"],
    },
    {
        "id": "Blender_3D_simulation_tools",
        "install_method": "app_bundle",
        "path_candidates": [
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ],
        "which_names": ["blender"],
        "version_argv": ["--version"],
        "reinstall_command": (
            "Install Blender 4.2.1 LTS (Apple Silicon) from https://www.blender.org/download/lts/4-2/ "
            "and place the app at /Applications/Blender.app"
        ),
        "needs_human": True,
        "human_reason": "app bundle install is interactive and may require admin rights",
    },
    {
        "id": "retrieval_and_embedding_service",
        "install_method": "app_bundle",
        "path_candidates": [
            "/Applications/Ollama.app/Contents/Resources/ollama",
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
        ],
        "which_names": ["ollama"],
        "version_argv": ["--version"],
        "reinstall_command": "Install Ollama from https://ollama.com/download and ensure `ollama` is on PATH",
        "needs_human": True,
        "human_reason": "app bundle install is interactive; models are restored separately via ollama pull",
    },
]


def repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root (parent of src/substrate)."""
    env = os.environ.get("SUBSTRATE_REPOSITORY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if start is not None:
        return Path(start).resolve()
    return Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    """SHA-256 of file bytes. Follows symlinks to the target content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run(argv: list[str], *, cwd: Path | None = None, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _git_head(root: Path) -> str:
    result = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found).resolve() if found else None


def _expand_candidate(root: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _glob_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def resolve_tool_path(root: Path, spec: dict[str, Any], preferred: str | None = None) -> Path | None:
    """Find a tool artifact relative to the current root, independent of the original path."""
    ordered: list[Path] = []
    if preferred:
        ordered.append(_expand_candidate(root, preferred) if not Path(preferred).is_absolute() else Path(preferred).expanduser())
    for candidate in spec.get("path_candidates", []):
        ordered.append(_expand_candidate(root, candidate))
    for pattern in spec.get("artifact_globs", []):
        hit = _glob_first(root, pattern)
        if hit is not None:
            ordered.append(hit)
    for name in spec.get("which_names", []):
        found = _which(name)
        if found is not None:
            ordered.append(found)

    seen: set[str] = set()
    for path in ordered:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def _tool_version(path: Path, argv: list[str] | None) -> str:
    if not argv:
        return ""
    result = _run([str(path), *argv], timeout=30.0)
    text = (result.stdout or result.stderr or "").strip()
    return text.splitlines()[0] if text else ""


def _directory_byte_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                if file_path.is_file() and not file_path.is_symlink():
                    total += file_path.stat().st_size
                elif file_path.is_symlink():
                    # Count symlink targets that resolve to files inside the tree only via lstat size? Prefer target size.
                    try:
                        total += file_path.stat().st_size
                    except OSError:
                        total += file_path.lstat().st_size
            except OSError:
                continue
    return total


def _parse_manifest_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(None, 1)
    if len(parts) != 2:
        return None
    digest, rel = parts
    rel = rel.lstrip(" *")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
        return None
    return digest.lower(), rel


def _is_os_metadata_rel(rel: str) -> bool:
    """Host noise that may appear in older MANIFEST listings but is not corpus content.

    Finder .DS_Store and AppleDouble `._*` sidecars change without scientific meaning.
    They are skipped during integrity checks; every other MANIFEST entry is still enforced.
    """
    name = Path(rel).name
    return name == ".DS_Store" or name.startswith("._")


def _resolve_manifest_member(dataset_root: Path, rel: str) -> Path | None:
    """Locate a MANIFEST member under the dataset root.

    Odyssey acquisition sometimes relocates evaluator-only archives under
    ``evaluator-only/`` after the MANIFEST path was recorded at the dataset root.
    Digests are still enforced; only the path is resolved.
    """
    primary = dataset_root / rel
    if primary.is_file():
        return primary
    name = Path(rel).name
    relocated = dataset_root / "evaluator-only" / name
    if relocated.is_file():
        return relocated
    return None


def verify_dataset_manifest(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Check every content entry in an existing MANIFEST.sha256 against on-disk bytes.

    Does not regenerate the manifest. Missing or mismatched content files fail the check.
    """
    if not manifest_path.is_file():
        return {
            "ok": False,
            "status": STATUS_MISSING,
            "checked": 0,
            "mismatched": 0,
            "missing_files": 0,
            "skipped_os_metadata": 0,
            "detail": f"MANIFEST.sha256 missing at {manifest_path}",
        }
    checked = 0
    mismatched = 0
    missing_files = 0
    skipped_os_metadata = 0
    first_problems: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_manifest_line(line)
        if parsed is None:
            continue
        expected, rel = parsed
        if _is_os_metadata_rel(rel):
            skipped_os_metadata += 1
            continue
        target = _resolve_manifest_member(dataset_root, rel)
        if target is None:
            missing_files += 1
            if len(first_problems) < 5:
                first_problems.append(f"missing:{rel}")
            continue
        observed = file_sha256(target)
        checked += 1
        if observed != expected:
            mismatched += 1
            if len(first_problems) < 5:
                first_problems.append(f"drifted:{rel}")
    ok = missing_files == 0 and mismatched == 0 and checked > 0
    if missing_files and checked == 0:
        status = STATUS_MISSING
    elif mismatched or missing_files:
        status = STATUS_DRIFTED
    else:
        status = STATUS_MATCHING if ok else STATUS_MISSING
    return {
        "ok": ok,
        "status": status,
        "checked": checked,
        "mismatched": mismatched,
        "missing_files": missing_files,
        "skipped_os_metadata": skipped_os_metadata,
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else "",
        "problems": first_problems,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _venv_python(root: Path) -> Path | None:
    candidate = root / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _pip_list(python: Path) -> dict[str, str]:
    result = _run([str(python), "-m", "pip", "list", "--format=json"], timeout=120.0)
    if result.returncode != 0 or not result.stdout.strip():
        # uv-managed envs may lack pip; use uv pip list
        uv = shutil.which("uv")
        if not uv:
            return {}
        result = _run([uv, "pip", "list", "--python", str(python), "--format", "json"], timeout=120.0)
        if result.returncode != 0 or not result.stdout.strip():
            return {}
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    out: dict[str, str] = {}
    for row in rows:
        name = str(row.get("name", "")).strip()
        version = str(row.get("version", "")).strip()
        if name and version:
            out[name.lower().replace("_", "-")] = version
    return out


def _measure_python(root: Path) -> dict[str, Any]:
    python = _venv_python(root)
    interpreter_version = ""
    interpreter_path = ""
    resolved: dict[str, str] = {}
    if python is not None:
        interpreter_path = str(python.resolve())
        ver = _run([str(python), "-c", "import sys; print(sys.version.split()[0])"])
        if ver.returncode == 0:
            interpreter_version = ver.stdout.strip()
        resolved = _pip_list(python)
    else:
        ver = _run([sys.executable, "-c", "import sys; print(sys.version.split()[0])"])
        interpreter_version = ver.stdout.strip() if ver.returncode == 0 else ""
        interpreter_path = sys.executable

    lock_path = root / "uv.lock"
    lock_sha = file_sha256(lock_path) if lock_path.is_file() else ""
    return {
        "interpreter_version": interpreter_version,
        "interpreter_path": interpreter_path,
        "requires_python": ">=3.11",
        "preferred_python": "3.12",
        "lockfile": "uv.lock",
        "lockfile_sha256": lock_sha,
        "resolved_dependencies": [
            {"name": name, "version": version} for name, version in sorted(resolved.items())
        ],
        "install_base_command": 'uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"',
        "install_extras": sorted(ODYSSEY_VENV_EXTRAS),
    }


def _measure_tool(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = resolve_tool_path(root, spec)
    if path is None:
        return {
            "id": spec["id"],
            "absolute_path": "",
            "version": "",
            "artifact_sha256": "",
            "install_method": spec["install_method"],
            "reinstall_command": spec["reinstall_command"],
            "needs_human": bool(spec.get("needs_human")),
            "human_reason": spec.get("human_reason", ""),
            "in_repo": bool(spec.get("in_repo", False)),
            "present_at_generation": False,
        }
    # Prefer dist-info RECORD for venv packages when globs resolve there
    artifact = path
    for pattern in spec.get("artifact_globs", []):
        hit = _glob_first(root, pattern)
        if hit is not None and hit.is_file():
            artifact = hit
            break
    version_target = path
    # When the measured artifact is a dist-info RECORD, version comes from package metadata.
    if artifact.name == "RECORD" or artifact.suffix == ".dist-info" or "dist-info" in artifact.parts:
        version_target = path if path != artifact else path
    version = _tool_version(version_target, spec.get("version_argv")) if spec.get("version_argv") else ""
    if not version and spec.get("package_names"):
        python = _venv_python(root)
        if python is not None:
            packages = _pip_list(python)
            bits = []
            for name in spec["package_names"]:
                key = name.lower().replace("_", "-")
                if key in packages:
                    bits.append(f"{name}=={packages[key]}")
            version = "; ".join(bits)
    return {
        "id": spec["id"],
        "absolute_path": str(artifact.resolve()),
        "repo_relative_path": (
            artifact.resolve().relative_to(root.resolve()).as_posix()
            if spec.get("in_repo") and root.resolve() in artifact.resolve().parents
            else ""
        ),
        "version": version,
        "artifact_sha256": file_sha256(artifact),
        "install_method": spec["install_method"],
        "reinstall_command": spec["reinstall_command"],
        "needs_human": bool(spec.get("needs_human")),
        "human_reason": spec.get("human_reason", ""),
        "in_repo": bool(spec.get("in_repo", False)),
        "present_at_generation": True,
    }


def _ollama_model_digest(name: str) -> dict[str, Any] | None:
    """Return digest/size for an installed ollama model from local manifests/blobs."""
    # Prefer `ollama show` layer path; fall back to on-disk manifest registry.
    short = name.split(":")[0]
    tag = name.split(":")[1] if ":" in name else "latest"
    manifest_path = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / short / tag
    # nomic-embed-text is often pulled as :latest
    if not manifest_path.is_file() and tag != "latest":
        alt = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / short / "latest"
        if alt.is_file() and short == name.split(":")[0]:
            # only accept latest when caller asked for untagged or nomic-embed-text without exact tag match after show
            pass
    if not manifest_path.is_file():
        # try ollama list JSON if available
        ollama = shutil.which("ollama")
        if not ollama:
            return None
        listed = _run([ollama, "list"], timeout=30.0)
        listed_ok = listed.returncode == 0 and (
            name in listed.stdout
            or f"{name}:latest" in listed.stdout
            or name.split(":")[0] in listed.stdout
            or any(line.split()[0].startswith(name) for line in listed.stdout.splitlines()[1:] if line.strip())
        )
        if not listed_ok:
            return None
        show = _run([ollama, "show", name, "--modelfile"], timeout=60.0)
        if show.returncode != 0:
            if ":" not in name:
                show = _run([ollama, "show", f"{name}:latest", "--modelfile"], timeout=60.0)
            if show.returncode != 0:
                return None
        digests = re.findall(r"sha256-([0-9a-f]{64})", show.stdout)
        if not digests:
            digests = re.findall(r"sha256:([0-9a-f]{64})", show.stdout)
        if not digests:
            return None
        primary = digests[0]
        blob = Path.home() / ".ollama" / "models" / "blobs" / f"sha256-{primary}"
        size = blob.stat().st_size if blob.is_file() else 0
        return {
            "name": name,
            "digest": f"sha256:{primary}",
            "size_bytes": size,
            "pull_command": f"ollama pull {name}",
            "blob_path": str(blob) if blob.is_file() else "",
        }

    try:
        document = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return None
    layers = document.get("layers") or []
    digests = [str(layer.get("digest", "")) for layer in layers if layer.get("digest")]
    size = sum(int(layer.get("size") or 0) for layer in layers)
    primary = digests[0] if digests else str((document.get("config") or {}).get("digest") or "")
    blob_name = primary.replace("sha256:", "sha256-") if primary else ""
    blob = Path.home() / ".ollama" / "models" / "blobs" / blob_name if blob_name else None
    if blob is not None and blob.is_file() and size == 0:
        size = blob.stat().st_size
    return {
        "name": name if ":" in name else f"{name}:latest",
        "digest": primary,
        "size_bytes": size,
        "pull_command": f"ollama pull {name}",
        "blob_path": str(blob) if blob is not None and blob.is_file() else "",
        "layer_digests": digests,
    }


def _measure_ollama_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for name in PINNED_OLLAMA_MODELS:
        measured = _ollama_model_digest(name)
        if measured is None:
            models.append(
                {
                    "name": name,
                    "digest": "",
                    "size_bytes": 0,
                    "pull_command": f"ollama pull {name}",
                    "present_at_generation": False,
                }
            )
        else:
            measured["present_at_generation"] = True
            models.append(measured)
    return models


def _measure_corpus(root: Path) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    prefetch = root / PREFETCH_REL
    acquisition_path = root / CORPUS_ACQUISITION_REL
    acquisition_bytes: dict[str, int] = {}
    if acquisition_path.is_file():
        try:
            acquisition = _read_json(acquisition_path)
            for row in acquisition.get("acquired") or []:
                if isinstance(row, dict) and row.get("dataset") and row.get("bytes") is not None:
                    acquisition_bytes[str(row["dataset"])] = int(row["bytes"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    if prefetch.is_dir():
        for child in sorted(prefetch.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            manifest = child / "MANIFEST.sha256"
            rel_root = child.relative_to(root).as_posix()
            rel_manifest = manifest.relative_to(root).as_posix() if manifest.is_file() else ""
            # Prefer live measurement; fall back to acquisition record.
            size = _directory_byte_size(child)
            if size == 0:
                size = acquisition_bytes.get(child.name, 0)
            entry = {
                "dataset": child.name,
                "root_relative": rel_root,
                "byte_size": size,
                "manifest_sha256_path": rel_manifest,
                "has_integrity_manifest": manifest.is_file(),
            }
            if manifest.is_file():
                entry["manifest_file_sha256"] = file_sha256(manifest)
            roots.append(entry)
    return roots


def _repo_identity(root: Path) -> dict[str, Any]:
    frozen_path = root / FROZEN_BUILD_REL
    selection_path = root / SOURCE_SELECTION_REL
    frozen_sha = ""
    selection_sha = ""
    frozen_file_sha = ""
    selection_file_sha = ""
    if frozen_path.is_file():
        frozen_file_sha = file_sha256(frozen_path)
        try:
            frozen_sha = str(_read_json(frozen_path).get("sha256") or "")
        except (OSError, json.JSONDecodeError):
            frozen_sha = ""
    if selection_path.is_file():
        selection_file_sha = file_sha256(selection_path)
        try:
            selection_sha = str(_read_json(selection_path).get("sha256") or "")
        except (OSError, json.JSONDecodeError):
            selection_sha = ""
    return {
        "git_head": _git_head(root),
        "frozen_build_path": FROZEN_BUILD_REL.as_posix(),
        "frozen_build_digest": frozen_sha,
        "frozen_build_file_sha256": frozen_file_sha,
        "source_selection_path": SOURCE_SELECTION_REL.as_posix(),
        "source_selection_digest": selection_sha,
        "source_selection_file_sha256": selection_file_sha,
    }


def generate_manifest(root: Path | None = None) -> dict[str, Any]:
    """Build a portability manifest from real on-disk measurements."""
    root = repo_root(root)
    tools = [_measure_tool(root, spec) for spec in TOOL_SPECS]
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "program": PROGRAM,
        "generated_at": _utc_now(),
        "activation": False,
        "repo": _repo_identity(root),
        "python": _measure_python(root),
        "tools": tools,
        "ollama_models": _measure_ollama_models(),
        "corpus_roots": _measure_corpus(root),
        "notes": {
            "venv_shebangs": (
                ".venv embeds absolute shebangs; after a path change run "
                "`python -m substrate.portability restore` to recreate it"
            ),
            "corpus_travel": (
                "data/ is gitignored (~176 GiB). A restorable copy must include it. "
                "Integrity is checked against each dataset MANIFEST.sha256; manifests are never regenerated here."
            ),
            "ollama_travel": (
                "Ollama model weights live under ~/.ollama and do not travel with the folder. "
                "restore will `ollama pull` any missing pinned model."
            ),
            "external_tools": (
                "Lean, z3, ffmpeg, Blender, and the Ollama app live outside the repo. "
                "restore prints reinstall commands and does not run sudo or app installers."
            ),
        },
    }
    return document


def default_manifest_path(root: Path | None = None) -> Path:
    return repo_root(root) / MANIFEST_REL


def write_manifest(document: dict[str, Any], path: Path | None = None, root: Path | None = None) -> Path:
    root = repo_root(root)
    path = path or (root / MANIFEST_REL)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def load_manifest(path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root = repo_root(root)
    path = path or (root / MANIFEST_REL)
    if not path.is_file():
        raise FileNotFoundError(f"portability manifest missing: {path}")
    document = _read_json(path)
    if document.get("schema") != SCHEMA:
        raise ValueError(f"unexpected portability manifest schema: {document.get('schema')!r}")
    return document


def _classify_artifact(path: Path | None, expected_sha256: str) -> tuple[str, str, str]:
    """Return (status, observed_path, observed_sha256)."""
    if path is None or not path.is_file():
        return STATUS_MISSING, "", ""
    observed = file_sha256(path)
    if expected_sha256 and observed != expected_sha256:
        return STATUS_DRIFTED, str(path), observed
    if not expected_sha256:
        # No digest recorded (generation gap): presence alone is not matching.
        return STATUS_DRIFTED, str(path), observed
    return STATUS_MATCHING, str(path), observed


def _spec_for_id(tool_id: str) -> dict[str, Any] | None:
    for spec in TOOL_SPECS:
        if spec["id"] == tool_id:
            return spec
    return None


def verify_tool(root: Path, tool: dict[str, Any]) -> dict[str, Any]:
    spec = _spec_for_id(str(tool.get("id", ""))) or {
        "id": tool.get("id"),
        "path_candidates": [tool.get("absolute_path", "")] if tool.get("absolute_path") else [],
        "artifact_globs": [],
        "which_names": [],
        "in_repo": tool.get("in_repo", False),
        "reinstall_command": tool.get("reinstall_command", ""),
    }
    preferred = None
    if tool.get("in_repo") and tool.get("repo_relative_path"):
        preferred = str(tool["repo_relative_path"])
    elif tool.get("absolute_path"):
        preferred = str(tool["absolute_path"])
    path = resolve_tool_path(root, spec, preferred=preferred)
    # For RECORD-based tools, prefer glob artifact
    if spec.get("artifact_globs"):
        hit = _glob_first(root, spec["artifact_globs"][0]) if spec.get("artifact_globs") else None
        if hit is not None:
            path = hit
    status, observed_path, observed_sha = _classify_artifact(path, str(tool.get("artifact_sha256") or ""))
    remediation = ""
    if status != STATUS_MATCHING:
        remediation = str(tool.get("reinstall_command") or (spec.get("reinstall_command") if isinstance(spec, dict) else "") or "")
    return {
        "kind": "tool",
        "id": tool.get("id"),
        "status": status,
        "expected_path": tool.get("absolute_path") or tool.get("repo_relative_path") or "",
        "observed_path": observed_path,
        "expected_sha256": tool.get("artifact_sha256") or "",
        "observed_sha256": observed_sha,
        "remediation": remediation,
        "needs_human": bool(tool.get("needs_human")),
    }


def verify_ollama_model(model: dict[str, Any]) -> dict[str, Any]:
    name = str(model.get("name") or "")
    expected = str(model.get("digest") or "")
    measured = _ollama_model_digest(name)
    if measured is None and ":" not in name:
        measured = _ollama_model_digest(f"{name}:latest")
    if measured is None:
        return {
            "kind": "ollama_model",
            "id": name,
            "status": STATUS_MISSING,
            "expected_digest": expected,
            "observed_digest": "",
            "remediation": model.get("pull_command") or f"ollama pull {name}",
        }
    observed = str(measured.get("digest") or "")
    # Compare primary layer digest; allow sha256: vs raw
    def _norm(value: str) -> str:
        value = value.strip().lower()
        if value.startswith("sha256:"):
            return value.split(":", 1)[1]
        if value.startswith("sha256-"):
            return value.split("-", 1)[1]
        return value

    if expected and _norm(observed) != _norm(expected):
        # Also accept if expected is in layer digests
        layers = [_norm(x) for x in measured.get("layer_digests") or []]
        if _norm(expected) not in layers and _norm(expected) != _norm(observed):
            return {
                "kind": "ollama_model",
                "id": name,
                "status": STATUS_DRIFTED,
                "expected_digest": expected,
                "observed_digest": observed,
                "remediation": model.get("pull_command") or f"ollama pull {name}",
            }
    return {
        "kind": "ollama_model",
        "id": name,
        "status": STATUS_MATCHING,
        "expected_digest": expected,
        "observed_digest": observed,
        "remediation": "",
    }


def verify_corpus_root(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    rel = str(entry.get("root_relative") or "")
    dataset_root = root / rel
    manifest_rel = str(entry.get("manifest_sha256_path") or "")
    if not dataset_root.is_dir():
        return {
            "kind": "corpus",
            "id": entry.get("dataset") or rel,
            "status": STATUS_MISSING,
            "root_relative": rel,
            "remediation": (
                f"Copy the gitignored dataset directory to {rel} "
                "(include data/ when transferring the working tree)"
            ),
            "detail": {},
        }
    if not manifest_rel:
        # Presence only when no integrity manifest exists (preexisting datasets).
        expected_size = int(entry.get("byte_size") or 0)
        observed_size = _directory_byte_size(dataset_root)
        if expected_size and observed_size and abs(observed_size - expected_size) > max(4096, expected_size // 1000):
            status = STATUS_DRIFTED
            remediation = f"Dataset {entry.get('dataset')} size drifted at {rel}; re-copy data/ from the sealed corpus host"
        else:
            status = STATUS_MATCHING
            remediation = ""
        return {
            "kind": "corpus",
            "id": entry.get("dataset") or rel,
            "status": status,
            "root_relative": rel,
            "remediation": remediation,
            "detail": {"byte_size": observed_size, "expected_byte_size": expected_size, "has_integrity_manifest": False},
        }

    manifest_path = root / manifest_rel
    detail = verify_dataset_manifest(dataset_root, manifest_path)
    status = detail["status"]
    remediation = ""
    if status != STATUS_MATCHING:
        remediation = (
            f"Restore dataset files under {rel} so they match the existing {manifest_rel} "
            "(do not regenerate MANIFEST.sha256)"
        )
    return {
        "kind": "corpus",
        "id": entry.get("dataset") or rel,
        "status": status,
        "root_relative": rel,
        "remediation": remediation,
        "detail": detail,
    }


def verify_python(root: Path, python_info: dict[str, Any]) -> dict[str, Any]:
    python = _venv_python(root)
    if python is None:
        return {
            "kind": "python",
            "id": "venv_python",
            "status": STATUS_MISSING,
            "remediation": "python -m substrate.portability restore",
            "detail": {"reason": ".venv/bin/python is missing (often after a path change because shebangs are absolute)"},
        }
    ver = _run([str(python), "-c", "import sys; print(sys.version.split()[0])"])
    observed = ver.stdout.strip() if ver.returncode == 0 else ""
    expected = str(python_info.get("interpreter_version") or "")
    if expected and observed and observed != expected:
        # Minor drift of patch is still drifted
        return {
            "kind": "python",
            "id": "venv_python",
            "status": STATUS_DRIFTED,
            "remediation": "python -m substrate.portability restore",
            "detail": {"expected": expected, "observed": observed},
        }
    lock = root / "uv.lock"
    expected_lock = str(python_info.get("lockfile_sha256") or "")
    if lock.is_file() and expected_lock:
        observed_lock = file_sha256(lock)
        if observed_lock != expected_lock:
            return {
                "kind": "python",
                "id": "uv_lock",
                "status": STATUS_DRIFTED,
                "remediation": "Restore uv.lock from the portable tree; then python -m substrate.portability restore",
                "detail": {"expected_sha256": expected_lock, "observed_sha256": observed_lock},
            }
    # Check pinned Odyssey extras when recorded. Explicit empty list means "no extras required".
    packages = _pip_list(python)
    missing_extras = []
    extras = python_info.get("install_extras")
    if extras is None:
        extras = list(ODYSSEY_VENV_EXTRAS)
    for name in extras:
        key = str(name).lower().replace("_", "-")
        if key not in packages and name.lower() not in packages:
            missing_extras.append(name)
    if missing_extras:
        return {
            "kind": "python",
            "id": "venv_extras",
            "status": STATUS_MISSING,
            "remediation": "python -m substrate.portability restore",
            "detail": {"missing_packages": missing_extras},
        }
    return {
        "kind": "python",
        "id": "venv_python",
        "status": STATUS_MATCHING,
        "remediation": "",
        "detail": {"interpreter_version": observed, "package_count": len(packages)},
    }


def verify(root: Path | None = None, manifest: dict[str, Any] | None = None, *, quick_corpus: bool = False) -> dict[str, Any]:
    """Compare the host to the portability manifest.

    Reports per item present-and-matching / present-but-drifted / missing with
    remediation commands. Does not repair anything.
    """
    root = repo_root(root)
    if manifest is None:
        manifest = load_manifest(root=root)

    items: list[dict[str, Any]] = []
    items.append(verify_python(root, manifest.get("python") or {}))

    for tool in manifest.get("tools") or []:
        items.append(verify_tool(root, tool))

    for model in manifest.get("ollama_models") or []:
        items.append(verify_ollama_model(model))

    for entry in manifest.get("corpus_roots") or []:
        if quick_corpus:
            # Tests / fast path: only check root + manifest file presence/hash, not every corpus byte.
            rel = str(entry.get("root_relative") or "")
            dataset_root = root / rel
            manifest_rel = str(entry.get("manifest_sha256_path") or "")
            if not dataset_root.is_dir():
                items.append(
                    {
                        "kind": "corpus",
                        "id": entry.get("dataset") or rel,
                        "status": STATUS_MISSING,
                        "root_relative": rel,
                        "remediation": f"Copy dataset directory to {rel}",
                        "detail": {"quick": True},
                    }
                )
                continue
            if manifest_rel:
                mpath = root / manifest_rel
                if not mpath.is_file():
                    items.append(
                        {
                            "kind": "corpus",
                            "id": entry.get("dataset") or rel,
                            "status": STATUS_MISSING,
                            "root_relative": rel,
                            "remediation": f"Restore {manifest_rel}",
                            "detail": {"quick": True},
                        }
                    )
                else:
                    expected_m = str(entry.get("manifest_file_sha256") or "")
                    observed_m = file_sha256(mpath)
                    status = STATUS_MATCHING if (not expected_m or observed_m == expected_m) else STATUS_DRIFTED
                    items.append(
                        {
                            "kind": "corpus",
                            "id": entry.get("dataset") or rel,
                            "status": status,
                            "root_relative": rel,
                            "remediation": "" if status == STATUS_MATCHING else f"Restore files under {rel} to match {manifest_rel}",
                            "detail": {"quick": True, "manifest_file_sha256": observed_m},
                        }
                    )
            else:
                items.append(
                    {
                        "kind": "corpus",
                        "id": entry.get("dataset") or rel,
                        "status": STATUS_MATCHING,
                        "root_relative": rel,
                        "remediation": "",
                        "detail": {"quick": True, "has_integrity_manifest": False},
                    }
                )
        else:
            items.append(verify_corpus_root(root, entry))

    # Repo identity (informational but drift is reported)
    identity = _repo_identity(root)
    expected_repo = manifest.get("repo") or {}
    for key, label in (
        ("git_head", "git_head"),
        ("frozen_build_digest", "frozen_build_digest"),
        ("source_selection_digest", "source_selection_digest"),
    ):
        expected = str(expected_repo.get(key) or "")
        observed = str(identity.get(key) or "")
        if not expected:
            continue
        if not observed:
            status = STATUS_MISSING
        elif observed != expected:
            status = STATUS_DRIFTED
        else:
            status = STATUS_MATCHING
        items.append(
            {
                "kind": "repo",
                "id": label,
                "status": status,
                "expected": expected,
                "observed": observed,
                "remediation": "" if status == STATUS_MATCHING else f"Check out the portable tree so {label} matches the manifest",
            }
        )

    counts = {
        STATUS_MATCHING: 0,
        STATUS_DRIFTED: 0,
        STATUS_MISSING: 0,
    }
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    blockers = [item for item in items if item["status"] != STATUS_MATCHING]
    return {
        "schema": "SUBSTRATE_PORTABILITY_VERIFY/v1",
        "program": PROGRAM,
        "activation": False,
        "ok": not blockers,
        "counts": counts,
        "items": items,
        "blockers": blockers,
        "remediation_commands": sorted({item["remediation"] for item in blockers if item.get("remediation")}),
    }


def _venv_looks_current(root: Path) -> bool:
    python = _venv_python(root)
    if python is None:
        return False
    # Shebang scripts in .venv/bin should reference this root.
    pytest_script = root / ".venv" / "bin" / "pytest"
    if pytest_script.is_file():
        try:
            first = pytest_script.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except OSError:
            return False
        if first.startswith("#!") and str(root / ".venv") not in first:
            return False
    # Can import substrate from this env
    result = _run([str(python), "-c", "import substrate; print('ok')"], cwd=root, timeout=30.0)
    return result.returncode == 0 and "ok" in result.stdout


def _recreate_venv(root: Path, python_info: dict[str, Any]) -> dict[str, Any]:
    uv = shutil.which("uv")
    if not uv:
        return {
            "ok": False,
            "action": "recreate_venv",
            "executed": False,
            "detail": "uv not found on PATH",
            "print_commands": [
                "Install uv (https://docs.astral.sh/uv/) then re-run: python -m substrate.portability restore",
            ],
        }
    if _venv_looks_current(root):
        return {
            "ok": True,
            "action": "recreate_venv",
            "executed": False,
            "detail": "existing .venv is usable at the current path",
            "skipped": True,
        }

    preferred = str(python_info.get("preferred_python") or "3.12")
    print_commands: list[str] = []
    steps: list[str] = []

    # Remove broken venv so shebangs are not reused
    venv_dir = root / ".venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        steps.append("removed_stale_venv")

    created = _run([uv, "venv", "--python", preferred, str(venv_dir)], cwd=root, timeout=120.0)
    if created.returncode != 0:
        return {
            "ok": False,
            "action": "recreate_venv",
            "executed": True,
            "detail": created.stderr.strip() or created.stdout.strip() or "uv venv failed",
            "print_commands": [f'uv venv --python {preferred} .venv && uv pip install -e ".[dev]"'],
        }
    steps.append("created_venv")

    python = venv_dir / "bin" / "python"
    # Offline install first (no network except ollama pull).
    base = _run(
        [uv, "pip", "install", "--python", str(python), "--offline", "-e", ".[dev]"],
        cwd=root,
        timeout=600.0,
    )
    if base.returncode != 0:
        online = 'uv pip install --python .venv/bin/python -e ".[dev]"'
        print_commands.append(online)
        return {
            "ok": False,
            "action": "recreate_venv",
            "executed": True,
            "detail": "offline install of .[dev] failed; network install printed for the operator",
            "stderr": (base.stderr or "")[-2000:],
            "print_commands": print_commands,
            "steps": steps,
        }
    steps.append("installed_dev")

    # Prefer exact resolved set when present; else extras list.
    resolved = python_info.get("resolved_dependencies") or []
    pins = [f"{row['name']}=={row['version']}" for row in resolved if row.get("name") and row.get("version")]
    # Always ensure Odyssey extras
    extra_names = [str(x) for x in (python_info.get("install_extras") or ODYSSEY_VENV_EXTRAS)]
    if pins:
        # Filter out the editable substrate pin if present; already installed.
        pins = [p for p in pins if not p.lower().startswith("substrate==")]
        extras_cmd = [uv, "pip", "install", "--python", str(python), "--offline", *pins]
    else:
        extras_cmd = [uv, "pip", "install", "--python", str(python), "--offline", *extra_names]

    extras = _run(extras_cmd, cwd=root, timeout=600.0)
    if extras.returncode != 0:
        if pins:
            print_commands.append("uv pip install --python .venv/bin/python " + " ".join(pins))
        else:
            print_commands.append("uv pip install --python .venv/bin/python " + " ".join(extra_names))
        return {
            "ok": False,
            "action": "recreate_venv",
            "executed": True,
            "detail": "offline install of resolved dependencies failed; network install printed for the operator",
            "stderr": (extras.stderr or "")[-2000:],
            "print_commands": print_commands,
            "steps": steps,
        }
    steps.append("installed_resolved_dependencies")
    return {
        "ok": True,
        "action": "recreate_venv",
        "executed": True,
        "detail": "recreated .venv from the locked dependency set (offline)",
        "steps": steps,
        "print_commands": print_commands,
    }


def _pull_missing_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    ollama = shutil.which("ollama")
    if not ollama:
        return {
            "ok": False,
            "action": "ollama_pull",
            "executed": False,
            "pulled": [],
            "already_present": [],
            "print_commands": [
                "Install Ollama from https://ollama.com/download",
                *[m.get("pull_command") or f"ollama pull {m.get('name')}" for m in models],
            ],
        }
    pulled: list[str] = []
    already: list[str] = []
    failed: list[dict[str, str]] = []
    print_commands: list[str] = []
    for model in models:
        name = str(model.get("name") or "")
        if not name:
            continue
        check = verify_ollama_model(model)
        if check["status"] == STATUS_MATCHING:
            already.append(name)
            continue
        cmd = str(model.get("pull_command") or f"ollama pull {name}")
        # Only pull pinned names already in the manifest.
        result = _run(cmd.split(), timeout=3600.0)
        if result.returncode != 0:
            failed.append({"name": name, "detail": (result.stderr or result.stdout)[-500:]})
            print_commands.append(cmd)
        else:
            pulled.append(name)
    return {
        "ok": not failed,
        "action": "ollama_pull",
        "executed": bool(pulled) or bool(failed),
        "pulled": pulled,
        "already_present": already,
        "failed": failed,
        "print_commands": print_commands,
    }


def restore(root: Path | None = None, manifest: dict[str, Any] | None = None, *, quick_corpus: bool = False) -> dict[str, Any]:
    """Idempotent safe restore: venv, ollama pulls, corpus verify; print human-only steps."""
    root = repo_root(root)
    if manifest is None:
        manifest = load_manifest(root=root)

    actions: list[dict[str, Any]] = []
    print_only: list[dict[str, Any]] = []

    venv_result = _recreate_venv(root, manifest.get("python") or {})
    actions.append(venv_result)

    model_result = _pull_missing_models(manifest.get("ollama_models") or [])
    actions.append(model_result)

    # Corpus: verify only (never regenerate manifests, never download corpora here).
    corpus_results = []
    for entry in manifest.get("corpus_roots") or []:
        if quick_corpus:
            rel = str(entry.get("root_relative") or "")
            ok = (root / rel).is_dir()
            corpus_results.append(
                {
                    "dataset": entry.get("dataset"),
                    "ok": ok,
                    "status": STATUS_MATCHING if ok else STATUS_MISSING,
                }
            )
        else:
            result = verify_corpus_root(root, entry)
            corpus_results.append(result)
    corpus_ok = all(row.get("status") == STATUS_MATCHING or row.get("ok") is True for row in corpus_results)
    actions.append(
        {
            "ok": corpus_ok,
            "action": "verify_corpus",
            "executed": True,
            "results": corpus_results,
            "detail": "read-only verification against existing MANIFEST.sha256 files",
        }
    )

    # External tools that need a human: print only
    for tool in manifest.get("tools") or []:
        if not tool.get("needs_human"):
            continue
        check = verify_tool(root, tool)
        if check["status"] != STATUS_MATCHING:
            print_only.append(
                {
                    "id": tool.get("id"),
                    "status": check["status"],
                    "command": tool.get("reinstall_command"),
                    "reason": tool.get("human_reason") or "requires sudo, network brew, or app installer",
                }
            )

    for item in venv_result.get("print_commands") or []:
        print_only.append({"id": "venv", "command": item, "reason": "offline package cache incomplete"})
    for item in model_result.get("print_commands") or []:
        print_only.append({"id": "ollama", "command": item, "reason": "ollama pull failed or ollama missing"})

    ok = bool(venv_result.get("ok")) and bool(model_result.get("ok")) and corpus_ok and not print_only
    # print_only external tools should not make restore claim full success, but venv+models+corpus may still be fine
    ok_core = bool(venv_result.get("ok")) and bool(model_result.get("ok")) and corpus_ok
    return {
        "schema": "SUBSTRATE_PORTABILITY_RESTORE/v1",
        "program": PROGRAM,
        "activation": False,
        "ok": ok_core,
        "fully_restored": ok_core and not print_only,
        "actions": actions,
        "print_only": print_only,
        "note": "restore does not run sudo or app installers; those commands are printed for the operator",
    }


def _print(document: dict[str, Any]) -> None:
    print(json.dumps(document, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python -m substrate.portability {generate|verify|restore} [--quick-corpus] [--manifest PATH]")

    quick = False
    manifest_path: Path | None = None
    args: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--quick-corpus":
            quick = True
        elif arg == "--manifest":
            i += 1
            if i >= len(argv):
                raise SystemExit("--manifest requires a path")
            manifest_path = Path(argv[i])
        elif arg.startswith("--"):
            raise SystemExit(f"unknown option {arg}")
        else:
            args.append(arg)
        i += 1

    if not args:
        raise SystemExit("usage: python -m substrate.portability {generate|verify|restore}")

    command = args[0]
    root = repo_root()

    if command == "generate":
        document = generate_manifest(root)
        path = write_manifest(document, path=manifest_path, root=root)
        _print(
            {
                "wrote": str(path),
                "schema": document["schema"],
                "tools": len(document.get("tools") or []),
                "ollama_models": len(document.get("ollama_models") or []),
                "corpus_roots": len(document.get("corpus_roots") or []),
                "git_head": (document.get("repo") or {}).get("git_head"),
            }
        )
        return

    if command == "verify":
        manifest = load_manifest(path=manifest_path, root=root) if manifest_path or (root / MANIFEST_REL).is_file() else None
        if manifest is None:
            raise SystemExit(f"manifest not found: {root / MANIFEST_REL}")
        report = verify(root, manifest, quick_corpus=quick)
        _print(report)
        raise SystemExit(0 if report["ok"] else 1)

    if command == "restore":
        manifest = load_manifest(path=manifest_path, root=root)
        report = restore(root, manifest, quick_corpus=quick)
        _print(report)
        if report.get("print_only"):
            print("\n# Commands that need a human (not executed):", file=sys.stderr)
            for row in report["print_only"]:
                print(f"# {row.get('id')}: {row.get('command')}", file=sys.stderr)
        raise SystemExit(0 if report["ok"] else 1)

    raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
