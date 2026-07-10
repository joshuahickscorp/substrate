"""Fail-closed intake for a tiny, rights-documented SANPO-Real cohort.

This module intentionally stops at acquisition and provenance. It does not decode frames, load an
encoder, tune on the official test split, or promote an F8/F16 scientific claim. The public Google
Cloud Storage object metadata is the external byte-integrity authority. Every selected GCS object is
generation-pinned, checked against its published size, MD5 and CRC32C, and assigned a local SHA256.

The default cohort is deterministic and small enough for the current-host raw-smoke envelope:

* official train sessions: first three park and first three non-park sessions become train;
* official train sessions: the fourth park and fourth non-park sessions become validation;
* official test sessions: the first park and first non-park sessions remain test-only;
* each session contributes camera_head/left frames 0, 8, ..., 56 and description.json.

"First" means lexicographic session-id order among sessions carrying all eight requested frames.
The source split lists, official repository commit, licensing statement and privacy statement are
pinned in the resulting plan and receipt.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

PLAN_SCHEMA = "mop-sanpo-real-smoke-plan/v1"
RECEIPT_SCHEMA = "mop-sanpo-real-smoke-intake/v1"
SOURCE_CARD_SCHEMA = "mop-sanpo-real-source-card/v1"
CONSUMER_SCHEMA = "mop-sanpo-real-consumer-manifest/v1"

OFFICIAL_DATASET_PAGE = "https://google-research-datasets.github.io/sanpo_dataset/"
OFFICIAL_REPOSITORY = "https://github.com/google-research-datasets/sanpo_dataset"
OFFICIAL_REPO_COMMIT = "11faca999b5c223b804cd3196541a1427834918b"
OFFICIAL_BUCKET = "gresearch"
OFFICIAL_PREFIX = "sanpo_dataset/v0/sanpo-real"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

# These are immutable bytes at OFFICIAL_REPO_COMMIT. README.md contains the official dataset-license,
# privacy, train/test split and download statements. LICENSE is the Apache-2.0 code-repository license,
# which is recorded separately so it cannot be mistaken for the dataset license.
REPOSITORY_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "name": "README.md",
        "url": (
            "https://raw.githubusercontent.com/google-research-datasets/sanpo_dataset/"
            f"{OFFICIAL_REPO_COMMIT}/README.md"
        ),
        "local_path": "official_repository/README.md",
        "size": 6755,
        "sha256": "25308388c98b559fd8ad57d29feb016b3a744f5eab74a3dee296a4f79d9f7587",
        "meaning": "official dataset description, split, privacy and dataset-license statements",
    },
    {
        "name": "LICENSE",
        "url": (
            "https://raw.githubusercontent.com/google-research-datasets/sanpo_dataset/"
            f"{OFFICIAL_REPO_COMMIT}/LICENSE"
        ),
        "local_path": "official_repository/LICENSE",
        "size": 11357,
        "sha256": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
        "meaning": "Apache-2.0 license for repository code, not the SANPO dataset license",
    },
)

FRAME_INDICES = tuple(range(0, 57, 8))
MIN_COHORT_BYTES = 250_000_000
MAX_COHORT_BYTES = 350_000_000
RAW_SMOKE_CAP_BYTES = 5_000_000_000
MIN_FREE_DISK_BYTES = 40_000_000_000
TRAIN_PER_CLASS = 3
VALIDATION_PER_CLASS = 1
TEST_PER_CLASS = 1
TARGET_SESSION_COUNT = 10

_GCS_FIELDS = "name,size,md5Hash,crc32c,generation,etag"
_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE: tuple[int, ...] | None = None


class SanpoIntakeError(RuntimeError):
    """An authority, selection, integrity, or safety gate failed."""


class GCSObjectNotFound(SanpoIntakeError):
    """A requested official GCS object does not exist."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SanpoIntakeError(f"unsafe local relative path: {value!r}")
    return path.as_posix()


def _gcs_object_url(name: str, generation: str | None = None) -> str:
    encoded = urllib.parse.quote(name, safe="")
    query = {"alt": "media"}
    if generation:
        query["generation"] = str(generation)
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{OFFICIAL_BUCKET}/o/{encoded}?"
        + urllib.parse.urlencode(query)
    )


def _normalize_metadata(raw: dict[str, Any]) -> dict[str, str | int]:
    required = ("name", "size", "md5Hash", "crc32c", "generation", "etag")
    missing = [field for field in required if raw.get(field) in (None, "")]
    if missing:
        raise SanpoIntakeError(f"GCS metadata missing authority fields {missing}: {raw.get('name')}")
    try:
        size = int(raw["size"])
        md5_bytes = base64.b64decode(str(raw["md5Hash"]), validate=True)
        crc_bytes = base64.b64decode(str(raw["crc32c"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise SanpoIntakeError(f"invalid GCS metadata encoding for {raw.get('name')}") from exc
    if size < 0 or len(md5_bytes) != 16 or len(crc_bytes) != 4:
        raise SanpoIntakeError(f"invalid GCS size/hash width for {raw.get('name')}")
    return {
        "name": str(raw["name"]),
        "size": size,
        "md5Hash": str(raw["md5Hash"]),
        "crc32c": str(raw["crc32c"]),
        "generation": str(raw["generation"]),
        "etag": str(raw["etag"]),
    }


class GCSJSONClient:
    """Minimal public GCS JSON API client with no credentials and no cloud SDK dependency."""

    def __init__(
        self,
        *,
        timeout: float = 45.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.timeout = float(timeout)
        self._opener = opener

    def _open(self, request: str | urllib.request.Request) -> Any:
        return self._opener(request, timeout=self.timeout)

    def metadata(self, name: str) -> dict[str, str | int]:
        encoded = urllib.parse.quote(name, safe="")
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{OFFICIAL_BUCKET}/o/{encoded}?"
            + urllib.parse.urlencode({"fields": _GCS_FIELDS})
        )
        try:
            with self._open(url) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise GCSObjectNotFound(name) from exc
            raise SanpoIntakeError(f"GCS metadata request failed for {name}: HTTP {exc.code}") from exc
        except (OSError, ValueError) as exc:
            raise SanpoIntakeError(f"GCS metadata request failed for {name}: {exc}") from exc
        return _normalize_metadata(raw)

    def list_metadata(
        self,
        prefix: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, str | int]]:
        items: list[dict[str, str | int]] = []
        page_token: str | None = None
        while True:
            page_size = 1000 if limit is None else max(1, min(1000, int(limit) - len(items)))
            query = {
                "prefix": prefix,
                "maxResults": str(page_size),
                "fields": f"items({_GCS_FIELDS}),nextPageToken",
            }
            if page_token:
                query["pageToken"] = page_token
            url = (
                f"https://storage.googleapis.com/storage/v1/b/{OFFICIAL_BUCKET}/o?"
                + urllib.parse.urlencode(query)
            )
            try:
                with self._open(url) as response:
                    raw = json.load(response)
            except (OSError, ValueError, urllib.error.HTTPError) as exc:
                raise SanpoIntakeError(f"GCS listing failed for {prefix}: {exc}") from exc
            items.extend(_normalize_metadata(item) for item in raw.get("items", []))
            page_token = raw.get("nextPageToken")
            if not page_token or (limit is not None and len(items) >= limit):
                break
        return items if limit is None else items[:limit]

    def verified_bytes(self, metadata: dict[str, Any], *, max_bytes: int = 1_000_000) -> bytes:
        meta = _normalize_metadata(metadata)
        if int(meta["size"]) > int(max_bytes):
            raise SanpoIntakeError(f"refusing in-memory fetch above {max_bytes} bytes: {meta['name']}")
        request = urllib.request.Request(
            _gcs_object_url(str(meta["name"]), str(meta["generation"])),
            headers={"Accept-Encoding": "identity", "User-Agent": "mop-sanpo-intake/1"},
        )
        try:
            with self._open(request) as response:
                payload = response.read(int(meta["size"]) + 1)
        except (OSError, urllib.error.HTTPError) as exc:
            raise SanpoIntakeError(f"GCS media fetch failed for {meta['name']}: {exc}") from exc
        verify_bytes(payload, meta)
        return payload


def _crc32c_table() -> tuple[int, ...]:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        values: list[int] = []
        for initial in range(256):
            value = initial
            for _ in range(8):
                value = (value >> 1) ^ (_CRC32C_POLY if value & 1 else 0)
            values.append(value & 0xFFFFFFFF)
        _CRC32C_TABLE = tuple(values)
    return _CRC32C_TABLE


def crc32c(chunks: Iterable[bytes]) -> int:
    """Return Castagnoli CRC32C for an iterable of byte chunks."""
    table = _crc32c_table()
    value = 0xFFFFFFFF
    for chunk in chunks:
        for byte in chunk:
            value = table[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def _crc32c_b64(value: int) -> str:
    return base64.b64encode(int(value).to_bytes(4, "big")).decode()


def verify_bytes(payload: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
    """Check bytes against official GCS authority and return local content hashes."""
    meta = _normalize_metadata(metadata)
    md5_digest = hashlib.md5(payload, usedforsecurity=False).digest()
    sha256 = hashlib.sha256(payload).hexdigest()
    crc = crc32c((payload,))
    problems: list[str] = []
    if len(payload) != int(meta["size"]):
        problems.append(f"size {len(payload)} != official {meta['size']}")
    if base64.b64encode(md5_digest).decode() != meta["md5Hash"]:
        problems.append("MD5 does not match official GCS md5Hash")
    if _crc32c_b64(crc) != meta["crc32c"]:
        problems.append("CRC32C does not match official GCS crc32c")
    if problems:
        raise SanpoIntakeError(f"integrity mismatch for {meta['name']}: {'; '.join(problems)}")
    return {
        "size": len(payload),
        "md5_hex": md5_digest.hex(),
        "md5_base64": base64.b64encode(md5_digest).decode(),
        "crc32c_base64": _crc32c_b64(crc),
        "sha256": sha256,
        "official_integrity_verified": True,
    }


def _iter_file(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            yield chunk


def hash_file(path: str | Path) -> dict[str, Any]:
    """Single-pass MD5, SHA256 and CRC32C for a local file."""
    local = Path(path)
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    table = _crc32c_table()
    crc = 0xFFFFFFFF
    size = 0
    with local.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            md5.update(chunk)
            sha.update(chunk)
            for byte in chunk:
                crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= 0xFFFFFFFF
    return {
        "size": size,
        "md5_hex": md5.hexdigest(),
        "md5_base64": base64.b64encode(md5.digest()).decode(),
        "crc32c_base64": _crc32c_b64(crc),
        "sha256": sha.hexdigest(),
    }


def verify_local_file(path: str | Path, authority: dict[str, Any]) -> dict[str, Any]:
    """Verify a local file against GCS metadata or a pinned repository SHA256."""
    local = Path(path)
    if not local.is_file():
        raise SanpoIntakeError(f"missing local file: {local}")
    hashes = hash_file(local)
    problems: list[str] = []
    if int(hashes["size"]) != int(authority["size"]):
        problems.append(f"size {hashes['size']} != {authority['size']}")
    if "md5Hash" in authority and hashes["md5_base64"] != authority["md5Hash"]:
        problems.append("MD5 does not match official GCS md5Hash")
    if "crc32c" in authority and hashes["crc32c_base64"] != authority["crc32c"]:
        problems.append("CRC32C does not match official GCS crc32c")
    if "sha256" in authority and hashes["sha256"] != authority["sha256"]:
        problems.append("SHA256 does not match pinned official-repository bytes")
    if problems:
        raise SanpoIntakeError(f"integrity mismatch for {local}: {'; '.join(problems)}")
    hashes["official_integrity_verified"] = True
    return hashes


def _repo_authority(opener: Callable[..., Any] = urllib.request.urlopen, timeout: float = 45.0) -> dict:
    commit_api = (
        "https://api.github.com/repos/google-research-datasets/sanpo_dataset/git/commits/"
        f"{OFFICIAL_REPO_COMMIT}"
    )
    request = urllib.request.Request(
        commit_api,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "mop-sanpo-intake/1"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            commit = json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise SanpoIntakeError(f"official repository commit verification failed: {exc}") from exc
    if commit.get("sha") != OFFICIAL_REPO_COMMIT:
        raise SanpoIntakeError("official repository API did not return the pinned commit")

    artifacts: list[dict[str, Any]] = []
    readme: bytes | None = None
    for expected in REPOSITORY_ARTIFACTS:
        req = urllib.request.Request(
            str(expected["url"]),
            headers={"Accept-Encoding": "identity", "User-Agent": "mop-sanpo-intake/1"},
        )
        try:
            with opener(req, timeout=timeout) as response:
                payload = response.read(int(expected["size"]) + 1)
        except (OSError, urllib.error.HTTPError) as exc:
            raise SanpoIntakeError(f"official repository fetch failed for {expected['name']}: {exc}") from exc
        sha = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected["size"] or sha != expected["sha256"]:
            raise SanpoIntakeError(f"pinned repository bytes changed for {expected['name']}")
        if expected["name"] == "README.md":
            readme = payload
        artifacts.append(dict(expected))

    assert readme is not None
    text = " ".join(readme.decode("utf-8").split())
    required_statements = {
        "dataset_license": "Creative Commons V4.0",
        "sharing_right": "free to share and adapt",
        "official_split": "mutually exclusive session IDs",
        "privacy_blur": "blur personally identifiable information",
        "volunteer_review": "review each video",
    }
    missing = [name for name, phrase in required_statements.items() if phrase not in text]
    if missing:
        raise SanpoIntakeError(f"official README missing required source statements: {missing}")
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_REPO_COMMIT,
        "commit_api": commit_api,
        "commit_verified": True,
        "artifacts": artifacts,
        "required_statements_verified": required_statements,
    }


def _parse_split(payload: bytes, *, name: str) -> list[str]:
    try:
        rows = [line.strip() for line in payload.decode("utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise SanpoIntakeError(f"official {name} split is not UTF-8") from exc
    if not rows or len(rows) != len(set(rows)):
        raise SanpoIntakeError(f"official {name} split is empty or contains duplicate session IDs")
    for session_id in rows:
        if "/" in session_id or session_id in {".", ".."}:
            raise SanpoIntakeError(f"unsafe session ID in official {name} split: {session_id!r}")
    return rows


def _session_attributes(payload: bytes, session_id: str) -> tuple[dict[str, Any], bool]:
    try:
        description = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SanpoIntakeError(f"invalid description.json for {session_id}") from exc
    if description.get("session_type") != "real":
        raise SanpoIntakeError(f"session {session_id} is not declared real")
    attributes = description.get("session_video_metadata")
    if not isinstance(attributes, dict) or not attributes:
        raise SanpoIntakeError(f"session {session_id} lacks high-level session_video_metadata")
    environments = attributes.get("environment_types")
    if not isinstance(environments, list) or not environments:
        raise SanpoIntakeError(f"session {session_id} lacks environment_types")
    is_park = "ENVIRONMENT_TYPE_PARK" in environments
    return attributes, is_park


def _with_local_path(metadata: dict[str, Any], local_path: str, kind: str) -> dict[str, Any]:
    return {
        **_normalize_metadata(metadata),
        "local_path": _safe_relative_path(local_path),
        "kind": kind,
        "media_url": _gcs_object_url(str(metadata["name"]), str(metadata["generation"])),
    }


def _select_sessions(
    client: GCSJSONClient,
    session_ids: list[str],
    *,
    official_split: str,
    per_class: int,
) -> tuple[dict[bool, list[dict[str, Any]]], list[dict[str, Any]]]:
    chosen: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
    audit: list[dict[str, Any]] = []
    for session_id in sorted(session_ids):
        if all(len(values) >= per_class for values in chosen.values()):
            break
        description_name = f"{OFFICIAL_PREFIX}/{session_id}/description.json"
        description_meta = client.metadata(description_name)
        description_payload = client.verified_bytes(description_meta)
        attributes, is_park = _session_attributes(description_payload, session_id)
        if len(chosen[is_park]) >= per_class:
            audit.append(
                {"session_id": session_id, "is_park": is_park, "selected": False, "reason": "quota-filled"}
            )
            continue

        frame_prefix = f"{OFFICIAL_PREFIX}/{session_id}/camera_head/left/video_frames/"
        frame_map = {
            str(item["name"]): item
            for item in client.list_metadata(frame_prefix, limit=max(FRAME_INDICES) + 1)
        }
        required_names = [f"{frame_prefix}{index:06d}.png" for index in FRAME_INDICES]
        missing = [name for name in required_names if name not in frame_map]
        if missing:
            audit.append(
                {
                    "session_id": session_id,
                    "is_park": is_park,
                    "selected": False,
                    "reason": "required-head-left-frames-missing",
                    "missing": missing,
                }
            )
            continue
        frames = [
            _with_local_path(
                frame_map[name],
                f"sessions/{session_id}/camera_head/left/video_frames/{index:06d}.png",
                "video_frame",
            )
            for index, name in zip(FRAME_INDICES, required_names, strict=True)
        ]
        selected = {
            "session_id": session_id,
            "official_split": official_split,
            "is_park": is_park,
            "high_level_attributes": attributes,
            "description": _with_local_path(
                description_meta,
                f"sessions/{session_id}/description.json",
                "session_description",
            ),
            "frames": frames,
        }
        chosen[is_park].append(selected)
        audit.append({"session_id": session_id, "is_park": is_park, "selected": True, "reason": "eligible"})
    shortages = {str(key): per_class - len(value) for key, value in chosen.items() if len(value) < per_class}
    if shortages:
        raise SanpoIntakeError(f"not enough complete {official_split} sessions by is_park: {shortages}")
    return chosen, audit


def _object_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("name", "size", "md5Hash", "crc32c", "generation", "etag", "local_path", "kind")
    }


def _plan_identity(plan: dict[str, Any]) -> dict[str, Any]:
    safety = plan["safety"]
    return {
        "schema": plan["schema"],
        "source": plan["source"],
        "selection_contract": plan["selection_contract"],
        "sessions": plan["sessions"],
        "gcs_objects": [_object_identity(item) for item in plan["gcs_objects"]],
        "repository_authority": plan["repository_authority"],
        "safety": {
            key: safety[key]
            for key in (
                "min_cohort_bytes",
                "max_cohort_bytes",
                "raw_smoke_cap_bytes",
                "min_free_disk_bytes",
                "check_before_and_after_every_object",
                "atomic_generation_pinned_downloads",
                "resumable_part_files",
            )
        },
        "claim_boundary": plan["claim_boundary"],
    }


def build_intake_plan(
    *,
    client: GCSJSONClient | None = None,
    disk_root: str | Path = ".",
    min_free_disk_bytes: int = MIN_FREE_DISK_BYTES,
    raw_smoke_cap_bytes: int = RAW_SMOKE_CAP_BYTES,
    repo_opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Build and validate the deterministic 10-session intake plan without downloading frames."""
    if min_free_disk_bytes < MIN_FREE_DISK_BYTES:
        raise SanpoIntakeError(
            f"min_free_disk_bytes cannot be lowered below the M3 floor {MIN_FREE_DISK_BYTES}"
        )
    if raw_smoke_cap_bytes > RAW_SMOKE_CAP_BYTES:
        raise SanpoIntakeError(f"raw_smoke_cap_bytes cannot exceed {RAW_SMOKE_CAP_BYTES}")
    gcs = client or GCSJSONClient()
    repo_authority = _repo_authority(repo_opener)

    split_objects: list[dict[str, Any]] = []
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "test"):
        name = f"{OFFICIAL_PREFIX}/splits/{split}_session_ids.txt"
        metadata = gcs.metadata(name)
        payload = gcs.verified_bytes(metadata)
        split_ids[split] = _parse_split(payload, name=split)
        split_objects.append(
            _with_local_path(metadata, f"official_splits/{split}_session_ids.txt", "official_split_list")
        )
    overlap = sorted(set(split_ids["train"]) & set(split_ids["test"]))
    if overlap:
        raise SanpoIntakeError(f"official train/test split overlap: {overlap[:8]}")

    train_selected, train_audit = _select_sessions(
        gcs,
        split_ids["train"],
        official_split="train",
        per_class=TRAIN_PER_CLASS + VALIDATION_PER_CLASS,
    )
    test_selected, test_audit = _select_sessions(
        gcs,
        split_ids["test"],
        official_split="test",
        per_class=TEST_PER_CLASS,
    )

    sessions: list[dict[str, Any]] = []
    for is_park in (False, True):
        for index, session in enumerate(train_selected[is_park]):
            sessions.append({**session, "role": "train" if index < TRAIN_PER_CLASS else "validation"})
        sessions.extend({**session, "role": "test"} for session in test_selected[is_park])
    role_order = {"train": 0, "validation": 1, "test": 2}
    sessions.sort(key=lambda row: (role_order[row["role"]], bool(row["is_park"]), row["session_id"]))

    gcs_objects = list(split_objects)
    for session in sessions:
        gcs_objects.append(session["description"])
        gcs_objects.extend(session["frames"])
    frame_bytes = sum(int(item["size"]) for item in gcs_objects if item["kind"] == "video_frame")
    gcs_bytes = sum(int(item["size"]) for item in gcs_objects)
    repo_bytes = sum(int(item["size"]) for item in repo_authority["artifacts"])
    total_bytes = gcs_bytes + repo_bytes
    free_before = shutil.disk_usage(disk_root).free

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_at": _utc_now(),
        "source": {
            "dataset": "SANPO-Real v0",
            "dataset_page": OFFICIAL_DATASET_PAGE,
            "repository": OFFICIAL_REPOSITORY,
            "repository_commit": OFFICIAL_REPO_COMMIT,
            "gcs_bucket": f"gs://{OFFICIAL_BUCKET}/sanpo_dataset/v0",
            "gcs_json_api": f"https://storage.googleapis.com/storage/v1/b/{OFFICIAL_BUCKET}/o",
            "dataset_license": DATASET_LICENSE,
            "dataset_license_url": DATASET_LICENSE_URL,
        },
        "selection_contract": {
            "algorithm": (
                "lexicographically first complete sessions per official split and is_park class; "
                "official-train class positions 0..2 are train, position 3 is validation; official-test "
                "position 0 is held-out test"
            ),
            "camera": "camera_head",
            "lens": "left",
            "frame_indices": list(FRAME_INDICES),
            "train_per_is_park": TRAIN_PER_CLASS,
            "validation_per_is_park": VALIDATION_PER_CLASS,
            "test_per_is_park": TEST_PER_CLASS,
            "official_test_tuning_allowed": False,
        },
        "official_split_summary": {
            "train_count": len(split_ids["train"]),
            "test_count": len(split_ids["test"]),
            "intersection_count": 0,
            "train_ids_sha256": hashlib.sha256("\n".join(split_ids["train"]).encode()).hexdigest(),
            "test_ids_sha256": hashlib.sha256("\n".join(split_ids["test"]).encode()).hexdigest(),
        },
        "sessions": sessions,
        "selection_audit": {"official_train": train_audit, "official_test": test_audit},
        "gcs_objects": gcs_objects,
        "repository_authority": repo_authority,
        "projected": {
            "sessions": len(sessions),
            "frames": sum(len(session["frames"]) for session in sessions),
            "gcs_objects": len(gcs_objects),
            "repository_objects": len(repo_authority["artifacts"]),
            "frame_bytes": frame_bytes,
            "gcs_bytes": gcs_bytes,
            "total_bytes": total_bytes,
        },
        "safety": {
            "min_cohort_bytes": MIN_COHORT_BYTES,
            "max_cohort_bytes": MAX_COHORT_BYTES,
            "raw_smoke_cap_bytes": int(raw_smoke_cap_bytes),
            "min_free_disk_bytes": int(min_free_disk_bytes),
            "free_disk_before_plan_bytes": free_before,
            "projected_free_after_bytes": free_before - total_bytes,
            "check_before_and_after_every_object": True,
            "atomic_generation_pinned_downloads": True,
            "resumable_part_files": True,
        },
        "claim_boundary": {
            "status": "staged-natural-video-input-only",
            "natural_video_source": True,
            "encoder_loaded": False,
            "scientific_promotion": False,
            "f8_f16_trusted_provenance_satisfied": False,
            "official_test_used_for_model_selection_or_tuning": False,
            "official_test_subset_selected_by_fixed_metadata_rule": True,
            "official_test_pixels_decoded_or_inspected": False,
            "statement": (
                "This stages rights-documented natural-video input. It does not by itself satisfy "
                "F8/F16 trusted-provenance authority or a natural-video scientific promotion. The "
                "official-test subset was fixed only by the preregistered split, is_park balance and "
                "frame-availability rule; its pixels remain unseen and it is forbidden for tuning."
            ),
        },
    }
    plan["plan_identity_sha256"] = _sha256_json(_plan_identity(plan))
    validate_intake_plan(plan, current_free_bytes=free_before)
    return plan


def validate_intake_plan(plan: dict[str, Any], *, current_free_bytes: int | None = None) -> None:
    """Fail if a plan weakens selection, split, authority, size, or disk invariants."""
    problems: list[str] = []
    if plan.get("schema") != PLAN_SCHEMA:
        problems.append(f"schema must be {PLAN_SCHEMA}")
    sessions = plan.get("sessions", [])
    if len(sessions) != TARGET_SESSION_COUNT:
        problems.append(f"session count {len(sessions)} != {TARGET_SESSION_COUNT}")
    counts: dict[tuple[str, bool], int] = {}
    session_ids: list[str] = []
    for session in sessions:
        role = str(session.get("role"))
        is_park = bool(session.get("is_park"))
        counts[(role, is_park)] = counts.get((role, is_park), 0) + 1
        session_id = str(session.get("session_id", ""))
        session_ids.append(session_id)
        official_split = session.get("official_split")
        if role in {"train", "validation"} and official_split != "train":
            problems.append(f"{session_id}: {role} must come from official train")
        if role == "test" and official_split != "test":
            problems.append(f"{session_id}: test must come from official test")
        if [int(Path(frame["local_path"]).stem) for frame in session.get("frames", [])] != list(
            FRAME_INDICES
        ):
            problems.append(f"{session_id}: ordered frame indices differ from contract")
        if not isinstance(session.get("high_level_attributes"), dict):
            problems.append(f"{session_id}: high-level attributes missing")
    expected_counts = {
        ("train", False): TRAIN_PER_CLASS,
        ("train", True): TRAIN_PER_CLASS,
        ("validation", False): VALIDATION_PER_CLASS,
        ("validation", True): VALIDATION_PER_CLASS,
        ("test", False): TEST_PER_CLASS,
        ("test", True): TEST_PER_CLASS,
    }
    if counts != expected_counts:
        problems.append(f"role/is_park counts {counts} != {expected_counts}")
    if len(session_ids) != len(set(session_ids)):
        problems.append("session IDs are not unique across roles")

    objects = plan.get("gcs_objects", [])
    object_names: list[str] = []
    local_paths: list[str] = []
    for item in objects:
        try:
            meta = _normalize_metadata(item)
            local_paths.append(_safe_relative_path(str(item["local_path"])))
        except (KeyError, SanpoIntakeError) as exc:
            problems.append(str(exc))
            continue
        object_names.append(str(meta["name"]))
        expected_url = _gcs_object_url(str(meta["name"]), str(meta["generation"]))
        if item.get("media_url") != expected_url:
            problems.append(f"{meta['name']}: media URL is not pinned to the recorded generation")
        if not str(meta["name"]).startswith(f"{OFFICIAL_PREFIX}/"):
            problems.append(f"object outside SANPO-Real prefix: {meta['name']}")
    if len(object_names) != len(set(object_names)):
        problems.append("GCS object names are not unique")
    if len(local_paths) != len(set(local_paths)):
        problems.append("local paths are not unique")

    projected = plan.get("projected", {})
    frame_bytes = int(projected.get("frame_bytes", -1))
    total_bytes = int(projected.get("total_bytes", -1))
    if not MIN_COHORT_BYTES <= frame_bytes <= MAX_COHORT_BYTES:
        problems.append(f"frame cohort {frame_bytes} bytes outside [{MIN_COHORT_BYTES}, {MAX_COHORT_BYTES}]")
    safety = plan.get("safety", {})
    if int(safety.get("min_free_disk_bytes", 0)) < MIN_FREE_DISK_BYTES:
        problems.append("disk floor was weakened")
    if int(safety.get("raw_smoke_cap_bytes", 0)) > RAW_SMOKE_CAP_BYTES:
        problems.append("raw-smoke cap was weakened")
    if total_bytes > int(safety.get("raw_smoke_cap_bytes", 0)):
        problems.append("projected bytes exceed raw-smoke cap")
    free = current_free_bytes
    if free is not None and int(free) - total_bytes < int(safety.get("min_free_disk_bytes", 0)):
        problems.append(
            f"projected free disk {int(free) - total_bytes} crosses floor {safety.get('min_free_disk_bytes')}"
        )
    boundary = plan.get("claim_boundary", {})
    if boundary.get("scientific_promotion") is not False:
        problems.append("intake plan may not claim scientific promotion")
    if boundary.get("f8_f16_trusted_provenance_satisfied") is not False:
        problems.append("intake plan may not satisfy the F8/F16 authority gate")
    if boundary.get("official_test_used_for_model_selection_or_tuning") is not False:
        problems.append("official test may not be used for model selection or tuning")
    if boundary.get("official_test_subset_selected_by_fixed_metadata_rule") is not True:
        problems.append("official-test metadata subset rule must be disclosed")
    if boundary.get("official_test_pixels_decoded_or_inspected") is not False:
        problems.append("official-test pixels must remain unseen during intake")
    if plan.get("repository_authority", {}).get("commit") != OFFICIAL_REPO_COMMIT:
        problems.append("official repository commit is not pinned")

    expected_identity = _sha256_json(_plan_identity(plan)) if not problems else None
    if expected_identity and plan.get("plan_identity_sha256") != expected_identity:
        problems.append("plan identity hash mismatch")
    if problems:
        raise SanpoIntakeError("invalid SANPO intake plan: " + "; ".join(problems))


def _disk_guard(
    root: Path,
    *,
    floor_bytes: int,
    additional_bytes: int = 0,
    stage: str,
) -> int:
    free = shutil.disk_usage(root).free
    if free < floor_bytes or free - max(0, int(additional_bytes)) < floor_bytes:
        raise SanpoIntakeError(
            f"disk guard failed {stage}: free={free}, additional={additional_bytes}, floor={floor_bytes}"
        )
    return free


def _atomic_json(path: Path, value: Any, *, floor_bytes: int) -> int:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _disk_guard(
        path.parent,
        floor_bytes=floor_bytes,
        additional_bytes=len(payload),
        stage=f"before generated sidecar {path.name}",
    )
    part = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with part.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()
    return _disk_guard(path.parent, floor_bytes=floor_bytes, stage=f"after generated sidecar {path.name}")


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status or 200)


def _download_atomic_resumable(
    *,
    url: str,
    destination: Path,
    authority: dict[str, Any],
    disk_floor_bytes: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = 90.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(authority["size"])
    free_before = _disk_guard(
        destination.parent,
        floor_bytes=disk_floor_bytes,
        additional_bytes=0 if destination.exists() else expected_size,
        stage=f"before {destination}",
    )
    if destination.exists():
        hashes = verify_local_file(destination, authority)
        free_after = _disk_guard(
            destination.parent, floor_bytes=disk_floor_bytes, stage=f"after reuse {destination}"
        )
        return hashes, {
            "status": "reused-verified",
            "resumed_from_bytes": expected_size,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
        }

    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_size:
        raise SanpoIntakeError(f"partial file exceeds authority size: {part}")
    remaining = expected_size - offset
    free_before = _disk_guard(
        destination.parent,
        floor_bytes=disk_floor_bytes,
        additional_bytes=remaining,
        stage=f"before transfer {destination}",
    )
    headers = {"Accept-Encoding": "identity", "User-Agent": "mop-sanpo-intake/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=timeout) as response:
            status = _response_status(response)
            if offset and status == 206:
                mode = "ab"
            elif offset and status == 200:
                mode = "wb"
                offset = 0
            elif not offset and status in {200, 206}:
                mode = "wb"
            else:
                raise SanpoIntakeError(f"unexpected HTTP status {status} for {url}")
            with part.open(mode) as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, urllib.error.HTTPError) as exc:
        raise SanpoIntakeError(f"download failed for {url}: {exc}") from exc
    if part.stat().st_size != expected_size:
        raise SanpoIntakeError(
            f"downloaded size {part.stat().st_size} != authority {expected_size}: {destination}"
        )
    hashes = verify_local_file(part, authority)
    os.replace(part, destination)
    free_after = _disk_guard(
        destination.parent, floor_bytes=disk_floor_bytes, stage=f"after transfer {destination}"
    )
    return hashes, {
        "status": "downloaded" if offset == 0 else "resumed",
        "resumed_from_bytes": offset,
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
    }


def _source_card(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SOURCE_CARD_SCHEMA,
        "created_at": _utc_now(),
        "source_id": "sanpo-real-v0-official-gcs-smoke",
        "provenance_tag": "natural-video",
        "dataset_page": OFFICIAL_DATASET_PAGE,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_repository_commit": OFFICIAL_REPO_COMMIT,
        "official_bucket": f"gs://{OFFICIAL_BUCKET}/sanpo_dataset/v0",
        "dataset_license": {
            "name": DATASET_LICENSE,
            "url": DATASET_LICENSE_URL,
            "official_statement": "SANPO is free to share and adapt for any purpose under CC V4.0.",
            "statement_source": "official_repository/README.md at the pinned commit",
            "attribution_required": True,
            "manual_terms_or_login_required": False,
        },
        "repository_code_license": {
            "name": "Apache License 2.0",
            "file": "official_repository/LICENSE",
            "applies_to": "repository code, not the dataset bytes",
        },
        "privacy": {
            "official_statements": [
                "Volunteers could review each video before upload.",
                "Videos were processed to blur PII such as faces and license plates.",
                "The official source asks users to report inadequately processed samples.",
            ],
            "contact": "sanpo_dataset@google.com",
            "residual_risk": (
                "Automated blurring can be imperfect. Keep raw data local, do not attempt re-identification, "
                "and report suspected PII failures to the official contact."
            ),
        },
        "selection": plan["selection_contract"],
        "allowed_local_use": (
            "local research, representation-learning intake, controlled feature extraction and derived "
            "aggregate receipts with attribution"
        ),
        "claim_boundary": plan["claim_boundary"],
        "legal_note": "This records the publisher's terms and provenance; it is not legal advice.",
    }


def _consumer_manifest(
    plan: dict[str, Any],
    local_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    content_entries: list[dict[str, Any]] = []
    for session in plan["sessions"]:
        description_path = session["description"]["local_path"]
        frame_entries: list[dict[str, Any]] = []
        for frame_index, frame in zip(FRAME_INDICES, session["frames"], strict=True):
            local = local_records[frame["local_path"]]
            entry = {
                "frame_index": frame_index,
                "path": frame["local_path"],
                "gcs_object": frame["name"],
                "generation": frame["generation"],
                "size": local["size"],
                "md5_base64": local["md5_base64"],
                "crc32c_base64": local["crc32c_base64"],
                "sha256": local["sha256"],
            }
            frame_entries.append(entry)
            content_entries.append(
                {key: entry[key] for key in ("path", "gcs_object", "generation", "size", "sha256")}
            )
        description_local = local_records[description_path]
        content_entries.append(
            {
                "path": description_path,
                "gcs_object": session["description"]["name"],
                "generation": session["description"]["generation"],
                "size": description_local["size"],
                "sha256": description_local["sha256"],
            }
        )
        sessions.append(
            {
                "session_id": session["session_id"],
                "official_split": session["official_split"],
                "role": session["role"],
                "test_only_no_tuning": session["role"] == "test",
                "is_park": session["is_park"],
                "high_level_attributes": session["high_level_attributes"],
                "description_path": description_path,
                "description_sha256": description_local["sha256"],
                "ordered_frames": frame_entries,
            }
        )
    content_entries.sort(key=lambda item: item["path"])
    content_set_hash = _sha256_json(content_entries)
    return {
        "schema": CONSUMER_SCHEMA,
        "created_at": _utc_now(),
        "dataset": "SANPO-Real v0",
        "root_relative_paths": True,
        "camera": "camera_head",
        "lens": "left",
        "temporal_order": list(FRAME_INDICES),
        "sessions": sessions,
        "content_entries": content_entries,
        "content_set_sha256": content_set_hash,
        "loader_contract": {
            "input_unit": "one ordered eight-frame session clip",
            "split_field": "role",
            "factor_fields": ["is_park", "high_level_attributes"],
            "test_policy": "role=test is evaluation-only and may not affect architecture or hyperparameters",
            "verification": "recompute every listed SHA256 and then content_set_sha256 before decode",
        },
        "claim_boundary": plan["claim_boundary"],
    }


def execute_intake_plan(
    plan: dict[str, Any],
    *,
    destination: str | Path,
    proof_path: str | Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = 90.0,
    download_workers: int = 1,
) -> dict[str, Any]:
    """Download or resume the plan serially, verify it, and write consumer/proof receipts."""
    root = Path(destination)
    proof = Path(proof_path)
    workers = int(download_workers)
    if workers < 1 or workers > 8:
        raise SanpoIntakeError("download_workers must be between 1 and 8")
    root.parent.mkdir(parents=True, exist_ok=True)
    floor = int(plan["safety"]["min_free_disk_bytes"])
    validate_intake_plan(plan, current_free_bytes=shutil.disk_usage(root.parent).free)
    free_start = _disk_guard(
        root.parent,
        floor_bytes=floor,
        additional_bytes=int(plan["projected"]["total_bytes"]),
        stage="before SANPO intake",
    )
    root.mkdir(parents=True, exist_ok=True)

    plan_path = root / "intake_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        identity_candidate = existing
        legacy_boundary = existing.get("claim_boundary", {})
        claim_boundary_migration = (
            legacy_boundary.get("official_test_used_for_selection_or_tuning") is False
            and "official_test_used_for_model_selection_or_tuning" not in legacy_boundary
        )
        if claim_boundary_migration:
            identity_candidate = {**existing, "claim_boundary": plan["claim_boundary"]}
        existing_stable_identity = _sha256_json(_plan_identity(identity_candidate))
        if existing_stable_identity != plan["plan_identity_sha256"]:
            raise SanpoIntakeError("destination contains a different SANPO intake plan")
        if claim_boundary_migration:
            existing["legacy_ambiguous_claim_boundary"] = existing["claim_boundary"]
            existing["claim_boundary"] = plan["claim_boundary"]
            existing["legacy_claim_plan_identity_sha256"] = existing.get("plan_identity_sha256")
            existing["plan_identity_sha256"] = existing_stable_identity
            existing["claim_boundary_migration"] = {
                "schema": "mop-sanpo-claim-boundary-migration/v1",
                "created_at": _utc_now(),
                "reason": (
                    "distinguish fixed metadata-only test subset selection from forbidden model "
                    "selection or tuning"
                ),
            }
            _atomic_json(plan_path, existing, floor_bytes=floor)
        elif existing.get("plan_identity_sha256") != existing_stable_identity:
            existing["legacy_dynamic_plan_identity_sha256"] = existing.get("plan_identity_sha256")
            existing["plan_identity_sha256"] = existing_stable_identity
            existing["identity_migration"] = {
                "schema": "mop-sanpo-plan-identity-migration/v1",
                "created_at": _utc_now(),
                "reason": "exclude live free-space observations from immutable plan identity",
            }
            _atomic_json(plan_path, existing, floor_bytes=floor)
    else:
        _atomic_json(plan_path, plan, floor_bytes=floor)

    started = time.monotonic()
    download_records: list[dict[str, Any]] = []
    local_records: dict[str, dict[str, Any]] = {}
    min_free = free_start

    for artifact in plan["repository_authority"]["artifacts"]:
        local_path = _safe_relative_path(str(artifact["local_path"]))
        hashes, transfer = _download_atomic_resumable(
            url=str(artifact["url"]),
            destination=root / local_path,
            authority=artifact,
            disk_floor_bytes=floor,
            opener=opener,
            timeout=timeout,
        )
        min_free = min(min_free, int(transfer["free_before_bytes"]), int(transfer["free_after_bytes"]))
        local_records[local_path] = hashes
        download_records.append(
            {
                "authority": "pinned-official-google-repository-commit",
                "remote": artifact["url"],
                "local_path": local_path,
                **transfer,
                **hashes,
            }
        )

    gcs_items = list(plan["gcs_objects"])
    remaining_gcs_bytes = 0
    for item in gcs_items:
        destination_path = root / _safe_relative_path(str(item["local_path"]))
        if destination_path.exists():
            continue
        part_path = destination_path.with_name(destination_path.name + ".part")
        partial_bytes = part_path.stat().st_size if part_path.exists() else 0
        remaining_gcs_bytes += max(0, int(item["size"]) - partial_bytes)
    _disk_guard(
        root,
        floor_bytes=floor,
        additional_bytes=remaining_gcs_bytes,
        stage="before bounded GCS transfer set",
    )

    def fetch_gcs(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
        local_path = _safe_relative_path(str(item["local_path"]))
        hashes, transfer = _download_atomic_resumable(
            url=str(item["media_url"]),
            destination=root / local_path,
            authority=item,
            disk_floor_bytes=floor,
            opener=opener,
            timeout=timeout,
        )
        return hashes, transfer, local_path

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sanpo-gcs") as executor:
        fetched = executor.map(fetch_gcs, gcs_items)
        for item, result in zip(gcs_items, fetched, strict=True):
            hashes, transfer, local_path = result
            min_free = min(
                min_free,
                int(transfer["free_before_bytes"]),
                int(transfer["free_after_bytes"]),
            )
            local_records[local_path] = hashes
            download_records.append(
                {
                    "authority": "official-gcs-json-api-metadata",
                    "remote": item["name"],
                    "local_path": local_path,
                    "generation": item["generation"],
                    "etag": item["etag"],
                    "official_md5_base64": item["md5Hash"],
                    "official_crc32c_base64": item["crc32c"],
                    **transfer,
                    **hashes,
                }
            )

    source_card = _source_card(plan)
    consumer = _consumer_manifest(plan, local_records)
    roles = {
        role: [session["session_id"] for session in plan["sessions"] if session["role"] == role]
        for role in ("train", "validation", "test")
    }
    split_manifest = {
        "schema": "mop-sanpo-real-explicit-splits/v1",
        "official_test_tuning_allowed": False,
        "roles": roles,
        "official_source": {
            "train": "official_splits/train_session_ids.txt",
            "test": "official_splits/test_session_ids.txt",
        },
    }
    referents = {
        "schema": "mop-sanpo-real-frame-referents/v1",
        "referents": [
            {
                "referent_id": f"sanpo-real:{session['session_id']}:head:left:{index:06d}",
                "session_id": session["session_id"],
                "official_split": session["official_split"],
                "role": session["role"],
                "is_park": session["is_park"],
                "frame_index": index,
                "path": frame["local_path"],
                "sha256": local_records[frame["local_path"]]["sha256"],
            }
            for session in plan["sessions"]
            for index, frame in zip(FRAME_INDICES, session["frames"], strict=True)
        ],
    }
    for name, payload in (
        ("source_card.json", source_card),
        ("splits.json", split_manifest),
        ("referents.json", referents),
        ("consumer_manifest.json", consumer),
    ):
        free = _atomic_json(root / name, payload, floor_bytes=floor)
        min_free = min(min_free, free)

    free_end = _disk_guard(root, floor_bytes=floor, stage="after SANPO intake")
    min_free = min(min_free, free_end)
    downloaded = [record for record in download_records if record["status"] in {"downloaded", "resumed"}]
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "mode": "executed",
        "all_ok": True,
        "download_active": False,
        "destination": str(root.resolve()),
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "source": plan["source"],
        "selection_contract": plan["selection_contract"],
        "counts": {
            "sessions": len(plan["sessions"]),
            "train_sessions": len(roles["train"]),
            "validation_sessions": len(roles["validation"]),
            "test_sessions": len(roles["test"]),
            "frames": len(referents["referents"]),
            "official_objects": len(download_records),
            "downloaded_or_resumed_objects_this_run": len(downloaded),
            "reused_objects_this_run": len(download_records) - len(downloaded),
            "download_workers": workers,
        },
        "bytes": {
            "projected_official_bytes": plan["projected"]["total_bytes"],
            "downloaded_or_resumed_final_object_bytes_this_run": sum(
                int(record["size"]) for record in downloaded
            ),
            "verified_official_bytes": sum(int(record["size"]) for record in download_records),
        },
        "disk": {
            "floor_bytes": floor,
            "free_start_bytes": free_start,
            "minimum_observed_free_bytes": min_free,
            "free_end_bytes": free_end,
            "floor_preserved": min_free >= floor,
            "checked_before_and_after_every_object": True,
        },
        "integrity": {
            "gcs_external_authority": "size + md5Hash + crc32c + generation + etag",
            "gcs_md5_verified": True,
            "gcs_crc32c_verified": True,
            "local_sha256_recorded": True,
            "repository_commit_and_sha256_verified": True,
            "content_set_sha256": consumer["content_set_sha256"],
        },
        "artifacts": {
            "plan": str(plan_path.resolve()),
            "source_card": str((root / "source_card.json").resolve()),
            "splits": str((root / "splits.json").resolve()),
            "referents": str((root / "referents.json").resolve()),
            "consumer_manifest": str((root / "consumer_manifest.json").resolve()),
        },
        "sessions": [
            {
                "session_id": session["session_id"],
                "official_split": session["official_split"],
                "role": session["role"],
                "is_park": session["is_park"],
            }
            for session in plan["sessions"]
        ],
        "download_records": download_records,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": plan["claim_boundary"],
    }
    _atomic_json(proof, receipt, floor_bytes=floor)
    return receipt


def dry_run_receipt(plan: dict[str, Any], *, destination: str | Path) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "mode": "dry-run",
        "all_ok": True,
        "download_active": False,
        "destination": str(Path(destination).resolve()),
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "projected": plan["projected"],
        "disk": plan["safety"],
        "sessions": [
            {
                "session_id": session["session_id"],
                "official_split": session["official_split"],
                "role": session["role"],
                "is_park": session["is_park"],
            }
            for session in plan["sessions"]
        ],
        "claim_boundary": plan["claim_boundary"],
    }


def verify_existing_intake(destination: str | Path) -> dict[str, Any]:
    """Re-hash every official object and the custom-substrate consumer content set."""
    root = Path(destination)
    plan_path = root / "intake_plan.json"
    if not plan_path.is_file():
        raise SanpoIntakeError(f"missing intake plan: {plan_path}")
    plan = json.loads(plan_path.read_text())
    validate_intake_plan(plan)
    records: dict[str, dict[str, Any]] = {}
    for artifact in plan["repository_authority"]["artifacts"]:
        records[artifact["local_path"]] = verify_local_file(root / artifact["local_path"], artifact)
    for item in plan["gcs_objects"]:
        records[item["local_path"]] = verify_local_file(root / item["local_path"], item)
    expected_consumer = _consumer_manifest(plan, records)
    actual_consumer = json.loads((root / "consumer_manifest.json").read_text())
    if actual_consumer.get("content_set_sha256") != expected_consumer["content_set_sha256"]:
        raise SanpoIntakeError("consumer manifest content-set hash mismatch")
    expected_roles = {
        role: [session["session_id"] for session in plan["sessions"] if session["role"] == role]
        for role in ("train", "validation", "test")
    }
    actual_splits = json.loads((root / "splits.json").read_text())
    if (
        actual_splits.get("roles") != expected_roles
        or actual_splits.get("official_test_tuning_allowed") is not False
    ):
        raise SanpoIntakeError("explicit local split manifest mismatch")
    return {
        "schema": "mop-sanpo-real-smoke-verification/v1",
        "created_at": _utc_now(),
        "all_ok": True,
        "destination": str(root.resolve()),
        "official_files_verified": len(records),
        "content_set_sha256": expected_consumer["content_set_sha256"],
        "claim_boundary": plan["claim_boundary"],
    }


def write_receipt(
    path: str | Path,
    receipt: dict[str, Any],
    *,
    floor_bytes: int = MIN_FREE_DISK_BYTES,
) -> None:
    _atomic_json(Path(path), receipt, floor_bytes=floor_bytes)


__all__ = [
    "CONSUMER_SCHEMA",
    "DATASET_LICENSE",
    "FRAME_INDICES",
    "GCSJSONClient",
    "GCSObjectNotFound",
    "MAX_COHORT_BYTES",
    "MIN_COHORT_BYTES",
    "MIN_FREE_DISK_BYTES",
    "OFFICIAL_REPO_COMMIT",
    "PLAN_SCHEMA",
    "RAW_SMOKE_CAP_BYTES",
    "RECEIPT_SCHEMA",
    "SOURCE_CARD_SCHEMA",
    "SanpoIntakeError",
    "build_intake_plan",
    "crc32c",
    "dry_run_receipt",
    "execute_intake_plan",
    "hash_file",
    "validate_intake_plan",
    "verify_bytes",
    "verify_existing_intake",
    "verify_local_file",
    "write_receipt",
]
