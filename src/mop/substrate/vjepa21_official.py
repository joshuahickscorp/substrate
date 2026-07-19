
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREFLIGHT_SCHEMA = "mop-vjepa21-official-preflight/v1"
CHECKPOINT_RECEIPT_SCHEMA = "mop-vjepa21-official-checkpoint/v1"

OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/vjepa2.git"
OFFICIAL_REPOSITORY_PAGE = "https://github.com/facebookresearch/vjepa2"
OFFICIAL_REPOSITORY_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
OFFICIAL_REPOSITORY_COMMIT_DATE = "2026-03-23T10:13:05Z"

PAPER: dict[str, Any] = {
    "title": "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning",
    "arxiv_id": "2603.14482",
    "version": "v3",
    "abstract_url": "https://arxiv.org/abs/2603.14482v3",
    "pdf_url": "https://arxiv.org/pdf/2603.14482v3",
    "published": "2026-03-15T17:02:40Z",
    "updated": "2026-06-11T10:07:56Z",
    "pdf_content_length": 37_389_706,
    "pdf_sha256": "6a7be4dbfd2131ef05640be457abef4cf57e1031dc97b894eab62c58782a3cac",
    "paper_license": "CC BY-NC-ND 4.0",
}

# both repository license files.  Checkpoint licensing is not separately stated in the README, so
# weight license grant.
REPOSITORY_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "path": "README.md",
        "size": 21_574,
        "sha256": "55f8485d734a08d23170322cd13ad5ac2f4d63b3d1fd542b10ec0907042c5bb6",
        "meaning": "official release, checkpoint table, macOS warning, and repository license statement",
    },
    {
        "path": "hubconf.py",
        "size": 543,
        "sha256": "6a61c46a80c82ed10331a19822d58e9a19f062e52845e1787fc810979ff03c7b",
        "meaning": "official torch.hub entrypoints and declared inference dependencies",
    },
    {
        "path": "src/hub/backbones.py",
        "size": 10_164,
        "sha256": "391cdde1e9a1da47cb8094bbea5fbbe8acac0135b27e82f1a6ab19c0b39cc692",
        "meaning": "official 2.1 architecture/checkpoint-key mapping and checkpoint loader",
    },
    {
        "path": "app/vjepa_2_1/models/vision_transformer.py",
        "size": 18_195,
        "sha256": "d2932eabeba684d8f558302a13cfd4be70a0170ee5112f5a794652d0a29089b9",
        "meaning": "official dense encoder and final-token forward interface",
    },
    {
        "path": "app/vjepa_2_1/models/utils/modules.py",
        "size": 16_963,
        "sha256": "64be6a87bd9f18d385f4e44186db3347d1665e18a1f0511d51d3b305531562e2",
        "meaning": "official transformer block and RoPE implementation",
    },
    {
        "path": "app/vjepa_2_1/models/utils/patch_embed.py",
        "size": 1_883,
        "sha256": "29e11ab97ab3ccdef107d6a7d0d7b374b58e712076cc3561f07b7e603c9b5165",
        "meaning": "official 2D/3D tokenizer implementation",
    },
    {
        "path": "src/masks/utils.py",
        "size": 660,
        "sha256": "833f111a0fa5ffdbd3a6412e2dace2517c3c178f49c14f8bb631d9f6a070dfd0",
        "meaning": "official encoder mask helper",
    },
    {
        "path": "src/utils/tensors.py",
        "size": 1_832,
        "sha256": "782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8",
        "meaning": "official tensor initialization helper",
    },
    {
        "path": "configs/train_2_1/vitb16/pretrain-256px-16f.yaml",
        "size": 2_444,
        "sha256": "df44fd5da9f5bcf9d76dbc354e1bf3fd0eef1528e26fa216ab5c91523b01285e",
        "meaning": "official ViT-B distillation pretraining phase config",
    },
    {
        "path": "configs/train_2_1/vitb16/cooldown-256px-64f.yaml",
        "size": 2_501,
        "sha256": "303de8b7ea58a2df4acdd47052d2306b9bca6e398c3facc47d8e83bdce839e88",
        "meaning": "official ViT-B distillation cooldown phase config",
    },
    {
        "path": "requirements.txt",
        "size": 212,
        "sha256": "86df4afafef209576d308aee6a6a2d13523805c1885f3e294f3a9541e6b2be23",
        "meaning": "official broad repository dependency declaration",
    },
    {
        "path": "LICENSE",
        "size": 1_087,
        "sha256": "cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173",
        "meaning": "MIT license for the majority of repository source",
    },
    {
        "path": "APACHE-LICENSE",
        "size": 11_349,
        "sha256": "a41e2fae9915ae56028f8dbd0bf27995f1907613cbcd2d81a61b010ad34e9fe9",
        "meaning": "Apache-2.0 text for the README-listed source exceptions",
    },
)

VITB: dict[str, Any] = {
    "slug": "vjepa21_vitb",
    "official_table_name": "ViT-B/16",
    "official_parameter_label": "80M",
    "hub_entrypoint": "vjepa2_1_vit_base_384",
    "architecture": "vit_base",
    "embed_dim": 768,
    "depth": 12,
    "heads": 12,
    "patch_size": 16,
    "tubelet_size": 2,
    "configured_frames": 64,
    "resolution": 384,
    "checkpoint_key": "ema_encoder",
    "checkpoint_url": ("https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"),
    "checkpoint_content_length": 1_664_223_428,
    "checkpoint_etag": '"be0dc26f052ae6a7476714cd53176836-199"',
    "checkpoint_last_modified": "Mon, 16 Mar 2026 09:12:19 GMT",
    "checkpoint_version_id": "xJBU4AkoA4gv5boeC6gOA7eElzWcubxY",
    "checkpoint_content_type": "application/vnd.snesdev-page-table",
    "range_bytes": 65_536,
    "first_range_sha256": "61efdf5f03e06d7d4dd6c4b35566dede6523453def98d51369e616093b23fbaf",
    "last_range_sha256": "cc219dadd7ab9e7dcb37121380e1a6ec24a7d8684fef2a8438e0e64e5c23ae59",
}

MIN_FREE_DISK_BYTES = 40_000_000_000
DOWNLOAD_WORKING_HEADROOM_BYTES = 512_000_000
DEFAULT_REPOSITORY_DIR = Path("data/models/vjepa21/official_repo")
DEFAULT_CHECKPOINT = Path("data/models/vjepa21/vjepa2_1_vitb_dist_vitG_384.pt")
DEFAULT_CONFIG = Path("configs/encoder/vjepa21_vitb.yaml")
DEFAULT_PREFLIGHT_PROOF = Path("proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json")
DEFAULT_DOCTOR_RECEIPT = Path("proof/STUDIO_READINESS_CURRENT_HOST.json")
DOCTOR_MAX_AGE_SECONDS = 15 * 60


class VJEPA21IntegrationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def expected_dense_tokens(frames: int, resolution: int = 384) -> int:
    if frames <= 0 or frames % int(VITB["tubelet_size"]):
        raise ValueError("frames must be a positive multiple of the tubelet size")
    if resolution <= 0 or resolution % int(VITB["patch_size"]):
        raise ValueError("resolution must be a positive multiple of the patch size")
    return (
        frames
        // int(VITB["tubelet_size"])
        * (resolution // int(VITB["patch_size"]))
        * (resolution // int(VITB["patch_size"]))
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise VJEPA21IntegrationError(f"git {' '.join(args)} failed in {repo}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def validate_repository(repo: Path | str = DEFAULT_REPOSITORY_DIR) -> dict[str, Any]:
    root = Path(repo).resolve()
    problems: list[str] = []
    commit = remote = None
    try:
        commit = _git(root, "rev-parse", "HEAD")
        remote = _git(root, "config", "--get", "remote.origin.url")
    except VJEPA21IntegrationError as exc:
        problems.append(str(exc))
    if commit != OFFICIAL_REPOSITORY_COMMIT:
        problems.append(f"repository commit {commit!r} != pinned {OFFICIAL_REPOSITORY_COMMIT}")
    normalized_remote = str(remote or "").removesuffix("/").removesuffix(".git")
    expected_remote = OFFICIAL_REPOSITORY.removesuffix(".git")
    if normalized_remote != expected_remote:
        problems.append(f"repository remote {remote!r} != official {OFFICIAL_REPOSITORY!r}")

    artifacts: list[dict[str, Any]] = []
    for expected in REPOSITORY_ARTIFACTS:
        path = root / str(expected["path"])
        row = dict(expected)
        if not path.is_file():
            row.update({"present": False, "verified": False})
            problems.append(f"missing pinned repository artifact {expected['path']}")
        else:
            size = path.stat().st_size
            digest = sha256_file(path)
            verified = size == int(expected["size"]) and digest == expected["sha256"]
            row.update(
                {
                    "present": True,
                    "actual_size": size,
                    "actual_sha256": digest,
                    "verified": verified,
                }
            )
            if not verified:
                problems.append(f"pinned repository artifact mismatch: {expected['path']}")
        artifacts.append(row)

    backbones = root / "src/hub/backbones.py"
    active_test_base_url = False
    if backbones.is_file():
        source = backbones.read_text()
        active_test_base_url = 'VJEPA_BASE_URL = "http://localhost:8300"' in source
        if not active_test_base_url:
            problems.append("pinned hub source no longer has the audited localhost base-URL condition")
    return {
        "official_repository": OFFICIAL_REPOSITORY_PAGE,
        "local_path": str(root),
        "commit": commit,
        "expected_commit": OFFICIAL_REPOSITORY_COMMIT,
        "remote": remote,
        "artifacts": artifacts,
        "direct_loader_required": active_test_base_url,
        "torch_hub_pretrained_allowed": False,
        "torch_hub_block_reason": (
            "the pinned official src/hub/backbones.py activates http://localhost:8300; "
            "the local seam constructs the official encoder with pretrained=False semantics and "
            "strict-loads ema_encoder from the separately verified checkpoint"
        ),
        "problems": problems,
        "all_ok": not problems,
    }


def stage_repository(destination: Path | str = DEFAULT_REPOSITORY_DIR) -> dict[str, Any]:
    dest = Path(destination).resolve()
    if dest.exists():
        receipt = validate_repository(dest)
        if not receipt["all_ok"]:
            raise VJEPA21IntegrationError(
                "existing repository destination is not the pinned official checkout: "
                + "; ".join(receipt["problems"])
            )
        receipt["stage_status"] = "already-present-verified"
        return receipt
    disk_root = dest.parent
    while not disk_root.exists() and disk_root != disk_root.parent:
        disk_root = disk_root.parent
    if shutil.disk_usage(disk_root).free < MIN_FREE_DISK_BYTES:
        raise VJEPA21IntegrationError("repository stage refused below the 40 GB local disk floor")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    if partial.exists():
        raise VJEPA21IntegrationError(
            f"partial repository already exists; inspect it before retry: {partial}"
        )
    completed = subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            OFFICIAL_REPOSITORY,
            str(partial),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise VJEPA21IntegrationError(f"official repository clone failed: {completed.stderr.strip()}")
    _git(partial, "checkout", "--detach", OFFICIAL_REPOSITORY_COMMIT)
    receipt = validate_repository(partial)
    if not receipt["all_ok"]:
        raise VJEPA21IntegrationError("staged official repository failed validation")
    os.replace(partial, dest)
    receipt = validate_repository(dest)
    receipt["stage_status"] = "cloned-pinned-and-verified"
    return receipt


def _head(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"Accept-Encoding": "identity", "User-Agent": "mop-vjepa21-preflight/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return {"status": int(response.status), "final_url": response.url, "headers": headers}
    except (OSError, urllib.error.HTTPError) as exc:
        raise VJEPA21IntegrationError(f"HEAD failed for {url}: {exc}") from exc


def _range_bytes(url: str, start: int, end: int, *, timeout: float = 30.0) -> tuple[bytes, dict]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "mop-vjepa21-preflight/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(end - start + 2)
            headers = {key.lower(): value for key, value in response.headers.items()}
            meta = {"status": int(response.status), "headers": headers, "final_url": response.url}
    except (OSError, urllib.error.HTTPError) as exc:
        raise VJEPA21IntegrationError(f"range request failed for {url}: {exc}") from exc
    if len(payload) != end - start + 1:
        raise VJEPA21IntegrationError(
            f"range {start}-{end} returned {len(payload)} bytes, expected {end - start + 1}"
        )
    return payload, meta


def validate_checkpoint_remote(*, timeout: float = 30.0, verify_ranges: bool = True) -> dict[str, Any]:
    head = _head(str(VITB["checkpoint_url"]), timeout=timeout)
    headers = head["headers"]
    observed = {
        "status": head["status"],
        "final_url": head["final_url"],
        "content_length": int(headers.get("content-length", "-1")),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "version_id": headers.get("x-amz-version-id"),
        "content_type": headers.get("content-type"),
        "accept_ranges": headers.get("accept-ranges"),
    }
    expected = {
        "status": 200,
        "content_length": int(VITB["checkpoint_content_length"]),
        "etag": VITB["checkpoint_etag"],
        "last_modified": VITB["checkpoint_last_modified"],
        "version_id": VITB["checkpoint_version_id"],
        "content_type": VITB["checkpoint_content_type"],
        "accept_ranges": "bytes",
    }
    problems = [
        f"{key}: observed {observed.get(key)!r} != pinned {value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    ranges: dict[str, Any] = {"verified": False, "probed": False}
    if verify_ranges and not problems:
        width = int(VITB["range_bytes"])
        total = int(VITB["checkpoint_content_length"])
        first, first_meta = _range_bytes(str(VITB["checkpoint_url"]), 0, width - 1, timeout=timeout)
        last_start = total - width
        last, last_meta = _range_bytes(str(VITB["checkpoint_url"]), last_start, total - 1, timeout=timeout)
        first_sha = _sha256_bytes(first)
        last_sha = _sha256_bytes(last)
        first_content_range = first_meta["headers"].get("content-range")
        last_content_range = last_meta["headers"].get("content-range")
        range_ok = (
            first_meta["status"] == 206
            and last_meta["status"] == 206
            and first_content_range == f"bytes 0-{width - 1}/{total}"
            and last_content_range == f"bytes {last_start}-{total - 1}/{total}"
            and first_sha == VITB["first_range_sha256"]
            and last_sha == VITB["last_range_sha256"]
        )
        ranges = {
            "probed": True,
            "bytes_each": width,
            "first": {
                "content_range": first_content_range,
                "sha256": first_sha,
                "expected_sha256": VITB["first_range_sha256"],
            },
            "last": {
                "content_range": last_content_range,
                "sha256": last_sha,
                "expected_sha256": VITB["last_range_sha256"],
            },
            "verified": range_ok,
        }
        if not range_ok:
            problems.append("checkpoint boundary-range identity mismatch")
    return {
        "url": VITB["checkpoint_url"],
        "observed": observed,
        "expected": expected,
        "ranges": ranges,
        "official_full_sha256_published": False,
        "integrity_model": (
            "before download: pinned HTTPS URL, exact S3 version/ETag/length/time, and two 64 KiB "
            "range hashes; after download: compute a full local SHA256 and bind it to those authorities"
        ),
        "problems": problems,
        "all_ok": not problems and (not verify_ranges or bool(ranges["verified"])),
    }


def _dependency(name: str, distribution: str | None = None) -> dict[str, Any]:
    present = importlib.util.find_spec(name) is not None
    version = None
    if present:
        try:
            version = importlib.metadata.version(distribution or name)
        except importlib.metadata.PackageNotFoundError:
            version = "present-version-unavailable"
    return {"module": name, "distribution": distribution or name, "present": present, "version": version}


def dependency_report() -> dict[str, Any]:
    core = [_dependency("torch"), _dependency("timm"), _dependency("einops")]
    decode = [_dependency("torchvision"), _dependency("av"), _dependency("decord")]
    return {
        "encoder_only_required": core,
        "encoder_only_ready": all(row["present"] for row in core),
        "official_full_repository_requirement": "requirements.txt includes decord",
        "macos_native_decord_ready": bool(decode[2]["present"]),
        "project_tensor_decode_ready": bool(decode[0]["present"] and decode[1]["present"]),
        "decode": decode,
        "selected_runtime_path": (
            "decode with the project's torchvision/PyAV path (or use already-decoded frame tensors), "
            "then call the official encoder directly; do not import the official decord dataloader"
        ),
        "macos_warning": (
            "Meta's README states that decord does not support macOS and leaves alternative selection "
            "to users. This seam bypasses decord; it does not claim the full official data pipeline works."
        ),
        "accelerator_warning": (
            "Meta strongly recommends CUDA. Retained receipts verify CPU strict load and 8-frame "
            "plus 64-frame forwards on this host; MPS remains unmeasured."
        ),
    }


def disk_report(root: Path | str = ".") -> dict[str, Any]:
    path = Path(root).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    usage = shutil.disk_usage(path)
    checkpoint_bytes = int(VITB["checkpoint_content_length"])
    required_before = MIN_FREE_DISK_BYTES + checkpoint_bytes + DOWNLOAD_WORKING_HEADROOM_BYTES
    return {
        "path": str(path),
        "free_bytes": int(usage.free),
        "floor_bytes": MIN_FREE_DISK_BYTES,
        "checkpoint_bytes": checkpoint_bytes,
        "working_headroom_bytes": DOWNLOAD_WORKING_HEADROOM_BYTES,
        "required_before_download_bytes": required_before,
        "projected_free_after_checkpoint_bytes": int(usage.free) - checkpoint_bytes,
        "download_feasible_now": int(usage.free) >= required_before,
        "atomic_download_peak_note": (
            "the .part file is renamed in place, so the final checkpoint is not a second full copy"
        ),
    }


def active_heavy_lane_report() -> dict[str, Any]:
    import psutil

    patterns = (
        "custom_substrate_workbench.py cm7",
        "cache_factorized_encoder.py",
        "cache_real_encoder.py",
        "encoder_scale_probe.py",
    )
    active: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        matched = [pattern for pattern in patterns if pattern in command]
        if matched:
            active.append({"pid": int(process.info["pid"]), "matched_patterns": matched})
    return {
        "known_heavy_patterns": list(patterns),
        "active": active,
        "active_count": len(active),
        "clear_for_new_heavy_lane": not active,
        "policy": "one model/training/encoder-cache lane at a time on the current host",
    }


def doctor_receipt_report(
    path: Path | str = DEFAULT_DOCTOR_RECEIPT,
    *,
    max_age_seconds: float = DOCTOR_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    problems: list[str] = []
    receipt: dict[str, Any] = {}
    if not receipt_path.is_file():
        problems.append(f"doctor receipt missing: {receipt_path}")
    else:
        try:
            loaded = json.loads(receipt_path.read_text())
            receipt = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            problems.append("doctor receipt is not valid JSON")
    created_at = receipt.get("created_at")
    age_seconds = None
    if created_at:
        try:
            created = datetime.fromisoformat(str(created_at))
            age_seconds = (datetime.now(UTC) - created.astimezone(UTC)).total_seconds()
        except ValueError:
            problems.append("doctor receipt created_at is invalid")
    else:
        problems.append("doctor receipt lacks created_at")
    if age_seconds is not None and (age_seconds < -5 or age_seconds > max_age_seconds):
        problems.append(
            f"doctor receipt age {age_seconds:.1f}s is outside the {max_age_seconds:.1f}s freshness window"
        )
    if receipt.get("schema") != "mop-studio-readiness/v2":
        problems.append("doctor receipt schema is not mop-studio-readiness/v2")
    if receipt.get("all_ok") is not True:
        problems.append("doctor receipt is not green")
    profile_value = receipt.get("profile")
    profile: dict[str, Any] = profile_value if isinstance(profile_value, dict) else {}
    if profile.get("resolved") != "m3pro-local-max":
        problems.append("doctor receipt is not for m3pro-local-max")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path) if receipt_path.is_file() else None,
        "created_at": created_at,
        "age_seconds": age_seconds,
        "max_age_seconds": max_age_seconds,
        "profile": profile.get("resolved"),
        "summary": receipt.get("summary"),
        "problems": problems,
        "fresh_and_green": not problems,
    }


def validate_vitb_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    import yaml

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    expected = {
        "name": VITB["slug"],
        "arch": VITB["architecture"],
        "embed_dim": VITB["embed_dim"],
        "patch_size": VITB["patch_size"],
        "tubelet": VITB["tubelet_size"],
        "frames_per_clip": VITB["configured_frames"],
        "resolution": VITB["resolution"],
        "dense": True,
        "pool": "none",
        "source_kind": "official_pytorch_checkpoint",
        "official_repo_commit": OFFICIAL_REPOSITORY_COMMIT,
        "hub_entrypoint": VITB["hub_entrypoint"],
        "checkpoint_url": VITB["checkpoint_url"],
        "available": True,
        "availability_state": "local_hash_strict_load_and_8f_64f_forward_verified",
        "cache_first_only": True,
        "checkpoint_sha256": "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d",
        "prefer_real": False,
    }
    problems = [
        f"config {key}: observed {raw.get(key)!r} != pinned {value!r}"
        for key, value in expected.items()
        if raw.get(key) != value
    ]
    return {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "observed": raw,
        "expected": expected,
        "problems": problems,
        "all_ok": not problems,
    }


def checkpoint_receipt_path(checkpoint: Path | str = DEFAULT_CHECKPOINT) -> Path:
    path = Path(checkpoint)
    return path.with_name(path.name + ".receipt.json")


def validate_checkpoint_receipt(
    checkpoint: Path | str = DEFAULT_CHECKPOINT, *, rehash: bool = True
) -> dict[str, Any]:
    path = Path(checkpoint).resolve()
    receipt_path = checkpoint_receipt_path(path)
    problems: list[str] = []
    if not path.is_file():
        problems.append(f"checkpoint missing: {path}")
    if not receipt_path.is_file():
        problems.append(f"checkpoint receipt missing: {receipt_path}")
        receipt: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(receipt_path.read_text())
            receipt = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            receipt = {}
            problems.append("checkpoint receipt is not valid JSON")
    expected_fields = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "source_url": VITB["checkpoint_url"],
        "source_etag": VITB["checkpoint_etag"],
        "source_version_id": VITB["checkpoint_version_id"],
        "size": int(VITB["checkpoint_content_length"]),
        "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            problems.append(f"checkpoint receipt {key} does not match pinned authority")
    actual_sha = None
    if path.is_file():
        if path.stat().st_size != int(VITB["checkpoint_content_length"]):
            problems.append("checkpoint file size does not match pinned authority")
        if rehash:
            actual_sha = sha256_file(path)
            if actual_sha != receipt.get("sha256"):
                problems.append("checkpoint full SHA256 does not match its receipt")
    if len(str(receipt.get("sha256") or "")) != 64:
        problems.append("checkpoint receipt lacks a full local SHA256")
    return {
        "path": str(path),
        "receipt_path": str(receipt_path),
        "receipt": receipt,
        "actual_sha256": actual_sha,
        "problems": problems,
        "all_ok": not problems,
    }


def download_vitb_checkpoint(
    destination: Path | str = DEFAULT_CHECKPOINT,
    *,
    timeout: float = 90.0,
    disk_floor_bytes: int = MIN_FREE_DISK_BYTES,
    doctor_receipt: Path | str = DEFAULT_DOCTOR_RECEIPT,
) -> dict[str, Any]:
    if disk_floor_bytes < MIN_FREE_DISK_BYTES:
        raise VJEPA21IntegrationError("the 40 GB local disk floor cannot be weakened")
    heavy_lanes = active_heavy_lane_report()
    if not heavy_lanes["clear_for_new_heavy_lane"]:
        raise VJEPA21IntegrationError(
            f"checkpoint download must wait for active heavy lane(s): {heavy_lanes['active']}"
        )
    doctor = doctor_receipt_report(doctor_receipt)
    if not doctor["fresh_and_green"]:
        raise VJEPA21IntegrationError(
            "checkpoint download requires a fresh green current-host doctor: " + "; ".join(doctor["problems"])
        )
    destination = Path(destination).resolve()
    receipt_path = checkpoint_receipt_path(destination)
    if destination.exists():
        existing = validate_checkpoint_receipt(destination)
        if existing["all_ok"]:
            return existing["receipt"]
        raise VJEPA21IntegrationError(
            "checkpoint exists without a valid authority receipt; refusing adoption: "
            + "; ".join(existing["problems"])
        )
    authority = validate_checkpoint_remote(timeout=timeout, verify_ranges=True)
    if not authority["all_ok"]:
        raise VJEPA21IntegrationError("remote checkpoint authority changed; refusing download")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    current = partial.stat().st_size if partial.exists() else 0
    total = int(VITB["checkpoint_content_length"])
    if current > total:
        raise VJEPA21IntegrationError("partial checkpoint is larger than the pinned object")
    remaining = total - current
    free = shutil.disk_usage(destination.parent).free
    if free < disk_floor_bytes + remaining:
        raise VJEPA21IntegrationError(
            f"download needs {remaining} additional bytes while preserving {disk_floor_bytes}; "
            f"only {free} bytes are free"
        )
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "mop-vjepa21-download/1",
        "If-Range": str(VITB["checkpoint_etag"]),
    }
    if current:
        headers["Range"] = f"bytes={current}-"
    request = urllib.request.Request(str(VITB["checkpoint_url"]), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            expected_status = 206 if current else 200
            if int(response.status) != expected_status:
                raise VJEPA21IntegrationError(
                    f"checkpoint response status {response.status}, expected {expected_status}"
                )
            mode = "ab" if current else "wb"
            written = current
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    future_remaining = total - (written + len(chunk))
                    if shutil.disk_usage(destination.parent).free < disk_floor_bytes + future_remaining:
                        raise VJEPA21IntegrationError("disk floor projection failed during download")
                    handle.write(chunk)
                    written += len(chunk)
                    if written > total:
                        raise VJEPA21IntegrationError("checkpoint transfer exceeded pinned content length")
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, urllib.error.HTTPError) as exc:
        raise VJEPA21IntegrationError(f"checkpoint download failed: {exc}") from exc
    if partial.stat().st_size != total:
        raise VJEPA21IntegrationError(f"incomplete checkpoint: {partial.stat().st_size} of {total} bytes")
    width = int(VITB["range_bytes"])
    with partial.open("rb") as handle:
        first = handle.read(width)
        handle.seek(total - width)
        last = handle.read(width)
    if _sha256_bytes(first) != VITB["first_range_sha256"]:
        raise VJEPA21IntegrationError("downloaded checkpoint first-range hash mismatch")
    if _sha256_bytes(last) != VITB["last_range_sha256"]:
        raise VJEPA21IntegrationError("downloaded checkpoint last-range hash mismatch")
    full_sha = sha256_file(partial)
    receipt = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "source_url": VITB["checkpoint_url"],
        "source_etag": VITB["checkpoint_etag"],
        "source_version_id": VITB["checkpoint_version_id"],
        "source_last_modified": VITB["checkpoint_last_modified"],
        "size": total,
        "sha256": full_sha,
        "first_range_sha256": VITB["first_range_sha256"],
        "last_range_sha256": VITB["last_range_sha256"],
        "repository": OFFICIAL_REPOSITORY_PAGE,
        "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "checkpoint_key": VITB["checkpoint_key"],
        "official_full_sha256_published": False,
        "all_ok": True,
    }
    temporary_receipt = receipt_path.with_name(receipt_path.name + ".part")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(partial, destination)
    os.replace(temporary_receipt, receipt_path)
    return receipt


def build_vitb_encoder(
    repository: Path | str = DEFAULT_REPOSITORY_DIR,
    *,
    random_seed: int | None = None,
):
    repo_validation = validate_repository(repository)
    if not repo_validation["all_ok"]:
        raise VJEPA21IntegrationError("official repository validation failed before model construction")
    deps = dependency_report()
    if not deps["encoder_only_ready"]:
        missing = [row["module"] for row in deps["encoder_only_required"] if not row["present"]]
        raise VJEPA21IntegrationError(f"missing encoder-only dependencies: {missing}")

    import torch

    repo = str(Path(repository).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from app.vjepa_2_1.models import vision_transformer as official_vit

    with torch.random.fork_rng(devices=[]):
        if random_seed is not None:
            torch.manual_seed(int(random_seed))
        encoder = official_vit.vit_base(
            patch_size=int(VITB["patch_size"]),
            img_size=(int(VITB["resolution"]), int(VITB["resolution"])),
            num_frames=int(VITB["configured_frames"]),
            tubelet_size=int(VITB["tubelet_size"]),
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def load_vitb_encoder(
    repository: Path | str = DEFAULT_REPOSITORY_DIR,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
):
    checkpoint_validation = validate_checkpoint_receipt(checkpoint, rehash=True)
    if not checkpoint_validation["all_ok"]:
        raise VJEPA21IntegrationError(
            "checkpoint validation failed before model load: " + "; ".join(checkpoint_validation["problems"])
        )

    encoder = build_vitb_encoder(repository)
    import torch

    payload = torch.load(
        Path(checkpoint).resolve(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict) or VITB["checkpoint_key"] not in payload:
        raise VJEPA21IntegrationError(
            f"checkpoint lacks required {VITB['checkpoint_key']!r} state dictionary"
        )
    state = payload[VITB["checkpoint_key"]]
    if not isinstance(state, dict) or not state:
        raise VJEPA21IntegrationError("ema_encoder checkpoint state is not a non-empty mapping")
    cleaned = {
        str(key).replace("module.", "").replace("backbone.", ""): value for key, value in state.items()
    }
    incompatible = encoder.load_state_dict(cleaned, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise VJEPA21IntegrationError(
            f"strict state load reported missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return encoder


def build_preflight(
    *,
    repository: Path | str = DEFAULT_REPOSITORY_DIR,
    config: Path | str = DEFAULT_CONFIG,
    disk_root: Path | str = ".",
    doctor_receipt: Path | str = DEFAULT_DOCTOR_RECEIPT,
    timeout: float = 30.0,
    verify_ranges: bool = True,
) -> dict[str, Any]:
    repo = validate_repository(repository)
    remote = validate_checkpoint_remote(timeout=timeout, verify_ranges=verify_ranges)
    config_report = validate_vitb_config(config)
    dependencies = dependency_report()
    disk = disk_report(disk_root)
    doctor = doctor_receipt_report(doctor_receipt)
    heavy_lanes = active_heavy_lane_report()
    checkpoint = validate_checkpoint_receipt(DEFAULT_CHECKPOINT, rehash=False)
    checkpoint_available = bool(checkpoint["all_ok"])
    source_and_disk_preflight_ok = bool(
        repo["all_ok"]
        and remote["all_ok"]
        and config_report["all_ok"]
        and (checkpoint_available or disk["download_feasible_now"])
    )
    ready_to_download = bool(
        not checkpoint_available
        and source_and_disk_preflight_ok
        and doctor["fresh_and_green"]
        and heavy_lanes["clear_for_new_heavy_lane"]
    )
    ready_to_construct_after_download = bool(repo["all_ok"] and dependencies["encoder_only_ready"])
    ready_to_load_now = bool(ready_to_construct_after_download and checkpoint["all_ok"])
    return {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": _utc_now(),
        "official_release": {
            "released": True,
            "release_date": "2026-03-16",
            "repository": OFFICIAL_REPOSITORY_PAGE,
            "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
            "repository_commit_date": OFFICIAL_REPOSITORY_COMMIT_DATE,
            "paper": PAPER,
            "source_license": {
                "majority": "MIT",
                "apache_exceptions": (
                    "README lists three dataset augmentation/worker files; none are imported by "
                    "the encoder seam"
                ),
                "checkpoint_license_separately_stated": False,
                "checkpoint_license_caveat": (
                    "the official repository publishes the checkpoint links but does not state a separate "
                    "weight license in the pinned README; do not infer more than the published source terms"
                ),
            },
        },
        "model": {
            **VITB,
            "configured_output_shape": [1, expected_dense_tokens(64), 768],
            "first_forward_shape_8f": [1, expected_dense_tokens(8), 768],
            "input_layout": "B,C,T,H,W",
            "output_layout": "B,(T/2)*(H/16)*(W/16),D",
            "dense_fp32_bytes_per_configured_clip": expected_dense_tokens(64) * 768 * 4,
            "dense_fp16_bytes_per_configured_clip": expected_dense_tokens(64) * 768 * 2,
        },
        "repository_validation": repo,
        "checkpoint_remote_validation": remote,
        "local_checkpoint_validation": checkpoint,
        "config_validation": config_report,
        "dependencies": dependencies,
        "disk": disk,
        "fresh_doctor": doctor,
        "heavy_lane": heavy_lanes,
        "gates": {
            "ready_to_download": ready_to_download,
            "source_and_disk_preflight_ok": source_and_disk_preflight_ok,
            "download_permission_note": (
                "ready_to_download stays false when the retained checkpoint receipt is already valid; "
                "otherwise it requires official source/object/disk checks, a fresh green local doctor, "
                "and no known heavy lane"
            ),
            "ready_to_construct_after_download": ready_to_construct_after_download,
            "ready_to_load_now": ready_to_load_now,
            "required_sequence": [
                "reuse the retained hash-verified ViT-B checkpoint and prior strict-load receipts",
                "build immutable rights-clean task tensors and one canonical input manifest",
                "encode learned and official seeded-random caches serially with resumable row hashes",
                "run E6 and DR14 verification without treating larger variants as prerequisites",
            ],
        },
        "claim_boundary": {
            "model_loaded": False,
            "forward_executed": False,
            "hardware_limit_measured": False,
            "e6_scientific_compatibility_proven": False,
            "dr14_scientific_compatibility_proven": False,
            "interpretation": (
                "upstream availability and local ViT-B CPU runtime are retired; natural referents, "
                "paired task caches, and task-level scientific controls remain separate fail-closed gates"
            ),
        },
        "all_ok": source_and_disk_preflight_ok,
    }


def write_preflight(path: Path | str, receipt: dict[str, Any]) -> None:
    _atomic_json(Path(path), receipt)
