"""Fail-closed SANPO-Real input bridge for the portable custom substrate.

The bridge is intentionally narrower than a training dataset.  It verifies the completed SANPO
intake and exposes one content-addressed, eight-frame clip at a time.  Development code can only
iterate the six train and two validation sessions.  Official-test pixels are reachable solely from
the explicit one-shot evaluator, after an independently verified portable artifact has been selected
without using those pixels.

Importing this module does not load an encoder or model.  The integrity/decode preflight hashes all
source files but decodes only train and validation PNGs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from PIL import __version__ as PILLOW_VERSION

from ..config import REPO_ROOT
from mop.substrate.events import sha256_file

BRIDGE_PLAN_SCHEMA = "mop-sanpo-custom-substrate-bridge-plan/v1"
BRIDGE_PREFLIGHT_SCHEMA = "mop-sanpo-custom-substrate-bridge-preflight/v1"
DEVELOPMENT_SELECTION_SCHEMA = "mop-sanpo-custom-substrate-development-selection/v1"
OFFICIAL_TEST_SCHEMA = "mop-sanpo-custom-substrate-official-test-one-shot/v1"
ATTEMPT_SCHEMA = "mop-sanpo-custom-substrate-official-test-attempt/v1"

INTAKE_SCHEMA = "mop-sanpo-real-smoke-intake/v1"
VERIFICATION_SCHEMA = "mop-sanpo-real-smoke-verification/v1"
CONSUMER_SCHEMA = "mop-sanpo-real-consumer-manifest/v1"
SPLITS_SCHEMA = "mop-sanpo-real-explicit-splits/v1"
REFERENTS_SCHEMA = "mop-sanpo-real-frame-referents/v1"

FRAME_INDICES = (0, 8, 16, 24, 32, 40, 48, 56)
DEVELOPMENT_ROLES = ("train", "validation")
EXPECTED_ROLE_COUNTS = {"train": 6, "validation": 2, "test": 2}


class SanpoBridgeRefused(RuntimeError):
    """Raised when source integrity, split isolation, or evaluation policy fails closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()




def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_bytes(list(value.shape)))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SanpoBridgeRefused(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SanpoBridgeRefused(f"{label} must be a JSON object: {path}")
    return cast(dict[str, Any], value)


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SanpoBridgeRefused(f"{label} path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SanpoBridgeRefused(f"{label} path is unsafe: {value!r}")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise SanpoBridgeRefused(f"{label} escapes repository root")
    return path


def _safe_content_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SanpoBridgeRefused(f"{label} content path is missing")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SanpoBridgeRefused(f"{label} content path is unsafe: {value!r}")
    resolved_root = root.resolve()
    path = (resolved_root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(resolved_root):
        raise SanpoBridgeRefused(f"{label} content path escapes intake root")
    return path


def _require(condition: bool, problem: str) -> None:
    if not condition:
        raise SanpoBridgeRefused(problem)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _plan_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(plan)
    identity.pop("plan_identity_sha256", None)
    return identity


def load_bridge_plan(path: Path, *, repo_root: Path = REPO_ROOT) -> tuple[dict[str, Any], Path]:
    """Read a frozen bridge plan and verify its self hash and source-file hashes."""

    plan_path = path if path.is_absolute() else repo_root / path
    plan = _read_json(plan_path, "SANPO bridge plan")
    _require(plan.get("schema") == BRIDGE_PLAN_SCHEMA, "SANPO bridge plan schema mismatch")
    expected_identity = json_sha256(_plan_identity(plan))
    _require(plan.get("plan_identity_sha256") == expected_identity, "SANPO bridge plan identity drift")

    source = plan.get("source")
    _require(isinstance(source, dict), "SANPO bridge source table is missing")
    source = cast(dict[str, Any], source)
    for label in ("intake_proof", "verification_proof", "consumer_manifest", "splits", "referents"):
        record = source.get(label)
        _require(isinstance(record, dict), f"bridge source record is missing: {label}")
        record = cast(dict[str, Any], record)
        _require(_is_sha256(record.get("sha256")), f"bridge source SHA256 is malformed: {label}")
        source_path = _safe_repo_path(repo_root, record.get("path"), label)
        _require(source_path.is_file(), f"bridge source file is missing: {label}: {source_path}")
        _require(sha256_file(source_path) == record["sha256"], f"bridge source byte drift: {label}")

    preprocessing = plan.get("preprocessing")
    _require(isinstance(preprocessing, dict), "preprocessing plan is missing")
    preprocessing = cast(dict[str, Any], preprocessing)
    decoder = preprocessing.get("decoder")
    _require(isinstance(decoder, dict), "preprocessing decoder contract is missing")
    decoder = cast(dict[str, Any], decoder)
    _require(decoder.get("library") == "Pillow", "only the pinned Pillow decoder is supported")
    _require(decoder.get("version") == PILLOW_VERSION, "Pillow runtime differs from frozen bridge plan")
    _require(decoder.get("format") == "PNG" and decoder.get("mode") == "RGB", "decode contract drift")
    spatial = preprocessing.get("spatial")
    tensor = preprocessing.get("tensor")
    _require(isinstance(spatial, dict) and isinstance(tensor, dict), "preprocessing geometry is missing")
    spatial = cast(dict[str, Any], spatial)
    tensor = cast(dict[str, Any], tensor)
    _require(spatial.get("operation") == "short-side-resize-then-center-crop", "spatial operation drift")
    _require(spatial.get("interpolation") == "bilinear", "spatial interpolation drift")
    target = spatial.get("target_size")
    _require(isinstance(target, int) and 2 <= target <= 2048, "target size is invalid")
    _require(
        tensor
        == {
            "axis_order": ["batch", "channel", "time", "height", "width"],
            "batch_size": 1,
            "channels": 3,
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "normalization": "uint8-divide-by-255-no-channel-standardization",
        },
        "tensor preprocessing contract drift",
    )
    _require(preprocessing.get("temporal_indices") == list(FRAME_INDICES), "temporal contract drift")

    policy = plan.get("evaluation_policy")
    _require(isinstance(policy, dict), "evaluation policy is missing")
    policy = cast(dict[str, Any], policy)
    _require(policy.get("development_roles") == list(DEVELOPMENT_ROLES), "development roles drift")
    _require(policy.get("official_test_sealed_by_default") is True, "official test is not sealed")
    _require(policy.get("official_test_sessions") == 2, "official-test sample count drift")
    _require(policy.get("official_test_tuning_allowed") is False, "official-test tuning was enabled")
    _require(policy.get("official_test_scientific_promotion") is False, "n=2 test cannot promote")
    ledger_path = _safe_repo_path(repo_root, policy.get("one_shot_attempt_receipt"), "one-shot receipt")
    return plan, ledger_path


@dataclass(frozen=True)
class SessionRecord:
    """Verified session metadata; raw official-test frame paths remain private to the bridge."""

    session_id: str
    official_split: str
    role: str
    is_park: bool
    high_level_attributes: dict[str, Any]
    description_path: str
    description_sha256: str
    frame_indices: tuple[int, ...]
    frame_paths: tuple[str, ...]
    frame_sha256s: tuple[str, ...]
    referent_ids: tuple[str, ...]


@dataclass(frozen=True)
class SanpoClip:
    """One verified clip and its referent-preserving metadata."""

    tensor: torch.Tensor
    session: SessionRecord
    source_tensor_sha256: str


@dataclass(frozen=True)
class _OfficialTestCapability:
    plan_identity_sha256: str
    attempt_path: Path


class SanpoCustomSubstrateBridge:
    """Verified SANPO cohort with a development-only public iterator."""

    def __init__(
        self,
        plan_path: Path,
        *,
        repo_root: Path = REPO_ROOT,
        verify_content: bool = True,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.plan, self.one_shot_attempt_path = load_bridge_plan(plan_path, repo_root=self.repo_root)
        source = cast(dict[str, Any], self.plan["source"])
        self.root = _safe_repo_path(self.repo_root, source.get("root"), "SANPO intake root")
        _require(self.root.is_dir(), f"SANPO intake root is missing: {self.root}")
        self.source_paths = {
            label: _safe_repo_path(self.repo_root, cast(dict[str, Any], source[label])["path"], label)
            for label in ("intake_proof", "verification_proof", "consumer_manifest", "splits", "referents")
        }
        self._sessions, self.integrity = self._verify_sources(verify_content=verify_content)

    @property
    def plan_identity_sha256(self) -> str:
        return str(self.plan["plan_identity_sha256"])

    @property
    def content_set_sha256(self) -> str:
        return str(cast(dict[str, Any], self.plan["source"])["content_set_sha256"])

    @property
    def sessions(self) -> tuple[SessionRecord, ...]:
        """Metadata for all sessions. This does not decode or return official-test pixel bytes."""

        return self._sessions

    def _verify_sources(self, *, verify_content: bool) -> tuple[tuple[SessionRecord, ...], dict[str, Any]]:
        source = cast(dict[str, Any], self.plan["source"])
        intake = _read_json(self.source_paths["intake_proof"], "SANPO intake proof")
        verification = _read_json(self.source_paths["verification_proof"], "SANPO verification proof")
        manifest = _read_json(self.source_paths["consumer_manifest"], "SANPO consumer manifest")
        splits = _read_json(self.source_paths["splits"], "SANPO split manifest")
        referents = _read_json(self.source_paths["referents"], "SANPO referent manifest")

        expected_content_hash = source.get("content_set_sha256")
        _require(_is_sha256(expected_content_hash), "bridge content-set SHA256 is malformed")
        _require(
            intake.get("schema") == INTAKE_SCHEMA and intake.get("all_ok") is True, "intake proof failed"
        )
        _require(intake.get("mode") == "executed", "intake proof is not an executed intake")
        _require(
            verification.get("schema") == VERIFICATION_SCHEMA and verification.get("all_ok") is True,
            "verification proof failed",
        )
        _require(verification.get("mode") == "verified-existing", "verification proof was not a re-hash")
        _require(manifest.get("schema") == CONSUMER_SCHEMA, "consumer manifest schema mismatch")
        _require(splits.get("schema") == SPLITS_SCHEMA, "split manifest schema mismatch")
        _require(referents.get("schema") == REFERENTS_SCHEMA, "referent manifest schema mismatch")
        for label, receipt in (("intake", intake), ("verification", verification)):
            destination = receipt.get("destination")
            _require(isinstance(destination, str), f"{label} proof destination is missing")
            destination = cast(str, destination)
            _require(Path(destination).resolve() == self.root, f"{label} proof binds a different destination")
        _require(
            cast(dict[str, Any], intake.get("integrity", {})).get("content_set_sha256")
            == expected_content_hash,
            "intake proof content-set hash mismatch",
        )
        _require(
            verification.get("content_set_sha256") == expected_content_hash,
            "verification proof content-set hash mismatch",
        )
        _require(manifest.get("content_set_sha256") == expected_content_hash, "consumer content-set mismatch")
        _require(verification.get("official_files_verified") == 94, "verification file-count drift")

        for label, receipt in (("intake", intake), ("verification", verification)):
            boundary = receipt.get("claim_boundary")
            _require(isinstance(boundary, dict), f"{label} claim boundary is missing")
            boundary = cast(dict[str, Any], boundary)
            _require(boundary.get("scientific_promotion") is False, f"{label} proof self-promotes")
            forbidden_use = boundary.get(
                "official_test_used_for_model_selection_or_tuning",
                boundary.get("official_test_used_for_selection_or_tuning"),
            )
            _require(forbidden_use is False, f"{label} proof permits official-test tuning")

        entries = manifest.get("content_entries")
        _require(isinstance(entries, list) and len(entries) == 90, "consumer content table must have 90 rows")
        entries = cast(list[dict[str, Any]], entries)
        _require(
            entries == sorted(entries, key=lambda row: str(row.get("path"))), "content table is not sorted"
        )
        paths = [row.get("path") for row in entries]
        _require(len(paths) == len(set(paths)), "consumer content paths are duplicated")
        _require(json_sha256(entries) == expected_content_hash, "consumer content-set recomputation failed")
        by_path = {str(row["path"]): row for row in entries}

        manifest_sessions = manifest.get("sessions")
        _require(isinstance(manifest_sessions, list) and len(manifest_sessions) == 10, "session count drift")
        manifest_sessions = cast(list[dict[str, Any]], manifest_sessions)
        roles: dict[str, list[str]] = {role: [] for role in EXPECTED_ROLE_COUNTS}
        records: list[SessionRecord] = []
        expected_referents: list[dict[str, Any]] = []
        verified_files = 0
        verified_frames = 0
        verified_descriptions = 0
        verified_bytes = 0
        seen_session_ids: set[str] = set()
        seen_content_paths: set[str] = set()

        for session in manifest_sessions:
            session_id = session.get("session_id")
            role = session.get("role")
            official_split = session.get("official_split")
            _require(isinstance(session_id, str) and bool(session_id), "session ID is invalid")
            session_id = cast(str, session_id)
            _require(session_id not in seen_session_ids, f"duplicate session ID: {session_id}")
            seen_session_ids.add(session_id)
            _require(
                isinstance(role, str) and role in EXPECTED_ROLE_COUNTS,
                f"invalid session role: {session_id}: {role}",
            )
            role = cast(str, role)
            _require(
                official_split == ("test" if role == "test" else "train"),
                f"official/local split mismatch: {session_id}",
            )
            _require(
                session.get("test_only_no_tuning") is (role == "test"),
                f"test-only flag mismatch: {session_id}",
            )
            roles[role].append(session_id)
            description_path = str(session.get("description_path", ""))
            description_sha = str(session.get("description_sha256", ""))
            _require(_is_sha256(description_sha), f"description SHA256 is malformed: {session_id}")
            description_entry = by_path.get(description_path)
            _require(description_entry is not None, f"description absent from content set: {session_id}")
            description_entry = cast(dict[str, Any], description_entry)
            _require(
                description_entry.get("sha256") == description_sha, f"description SHA drift: {session_id}"
            )
            description_file = _safe_content_path(self.root, description_path, f"description {session_id}")
            _require(description_file.is_file(), f"description file missing: {session_id}")
            if verify_content:
                _require(
                    description_file.stat().st_size == description_entry.get("size"),
                    f"description size drift: {session_id}",
                )
                _require(
                    sha256_file(description_file) == description_sha, f"description bytes drift: {session_id}"
                )
                verified_files += 1
                verified_descriptions += 1
                verified_bytes += description_file.stat().st_size
            description = _read_json(description_file, f"description {session_id}")
            _require(description.get("session_type") == "real", f"non-real session in bridge: {session_id}")
            attributes = session.get("high_level_attributes")
            _require(
                isinstance(attributes, dict) and bool(attributes),
                f"session attributes missing: {session_id}",
            )
            attributes = cast(dict[str, Any], attributes)
            _require(
                description.get("session_video_metadata") == attributes,
                f"session attributes differ from description: {session_id}",
            )
            seen_content_paths.add(description_path)

            ordered = session.get("ordered_frames")
            _require(
                isinstance(ordered, list) and len(ordered) == 8, f"clip is not eight frames: {session_id}"
            )
            ordered = cast(list[dict[str, Any]], ordered)
            indices = tuple(int(frame.get("frame_index", -1)) for frame in ordered)
            _require(indices == FRAME_INDICES, f"frame order drift: {session_id}: {indices}")
            frame_paths: list[str] = []
            frame_shas: list[str] = []
            referent_ids: list[str] = []
            for frame_index, frame in zip(FRAME_INDICES, ordered, strict=True):
                frame_path = str(frame.get("path", ""))
                frame_sha = frame.get("sha256")
                _require(_is_sha256(frame_sha), f"frame SHA256 is malformed: {session_id}:{frame_index}")
                frame_entry = by_path.get(frame_path)
                _require(
                    frame_entry is not None, f"frame absent from content set: {session_id}:{frame_index}"
                )
                frame_entry = cast(dict[str, Any], frame_entry)
                for field in ("path", "gcs_object", "generation", "size", "sha256"):
                    _require(
                        frame.get(field) == frame_entry.get(field),
                        f"frame/content mismatch {field}: {session_id}:{frame_index}",
                    )
                frame_file = _safe_content_path(self.root, frame_path, f"frame {session_id}:{frame_index}")
                _require(frame_file.is_file(), f"frame file missing: {session_id}:{frame_index}")
                if verify_content:
                    _require(
                        frame_file.stat().st_size == frame.get("size"),
                        f"frame size drift: {session_id}:{frame_index}",
                    )
                    _require(
                        sha256_file(frame_file) == frame_sha, f"frame bytes drift: {session_id}:{frame_index}"
                    )
                    verified_files += 1
                    verified_frames += 1
                    verified_bytes += frame_file.stat().st_size
                referent_id = f"sanpo-real:{session_id}:head:left:{frame_index:06d}"
                frame_paths.append(frame_path)
                frame_shas.append(str(frame_sha))
                referent_ids.append(referent_id)
                seen_content_paths.add(frame_path)
                expected_referents.append(
                    {
                        "referent_id": referent_id,
                        "session_id": session_id,
                        "official_split": official_split,
                        "role": role,
                        "is_park": session.get("is_park"),
                        "frame_index": frame_index,
                        "path": frame_path,
                        "sha256": frame_sha,
                    }
                )
            records.append(
                SessionRecord(
                    session_id=session_id,
                    official_split=str(official_split),
                    role=role,
                    is_park=bool(session.get("is_park")),
                    high_level_attributes=attributes,
                    description_path=description_path,
                    description_sha256=description_sha,
                    frame_indices=indices,
                    frame_paths=tuple(frame_paths),
                    frame_sha256s=tuple(frame_shas),
                    referent_ids=tuple(referent_ids),
                )
            )

        _require({role: len(ids) for role, ids in roles.items()} == EXPECTED_ROLE_COUNTS, "role counts drift")
        _require(seen_content_paths == set(by_path), "consumer content table has unbound paths")
        _require(
            splits.get("official_test_tuning_allowed") is False, "split file enables official-test tuning"
        )
        _require(cast(dict[str, Any], splits.get("roles", {})) == roles, "explicit split IDs drift")
        _require(referents.get("referents") == expected_referents, "frame referent sidecar drift")
        manifest_boundary = manifest.get("claim_boundary")
        _require(isinstance(manifest_boundary, dict), "consumer claim boundary is missing")
        manifest_boundary = cast(dict[str, Any], manifest_boundary)
        _require(manifest_boundary.get("scientific_promotion") is False, "consumer manifest self-promotes")
        _require(
            manifest_boundary.get("official_test_used_for_model_selection_or_tuning") is False,
            "consumer manifest enables official-test tuning",
        )

        download_records = intake.get("download_records")
        _require(isinstance(download_records, list), "intake download-record table is missing")
        downloaded = {
            str(row.get("local_path")): row
            for row in cast(list[dict[str, Any]], download_records)
            if row.get("local_path") in by_path
        }
        _require(set(downloaded) == set(by_path), "intake proof does not bind every consumer file")
        for path, entry in by_path.items():
            _require(
                downloaded[path].get("sha256") == entry.get("sha256"), f"intake/content SHA mismatch: {path}"
            )

        return tuple(records), {
            "all_ok": True,
            "source_receipt_sha256": {label: sha256_file(path) for label, path in self.source_paths.items()},
            "content_set_sha256": expected_content_hash,
            "sessions_verified": len(records),
            "content_files_verified": verified_files,
            "descriptions_verified": verified_descriptions,
            "frames_verified": verified_frames,
            "content_bytes_verified": verified_bytes,
            "role_counts": {role: len(ids) for role, ids in roles.items()},
            "official_test_frames_hashed_not_decoded": EXPECTED_ROLE_COUNTS["test"] * len(FRAME_INDICES),
        }

    def _preprocess_frame(self, path: Path) -> torch.Tensor:
        target = int(
            cast(dict[str, Any], cast(dict[str, Any], self.plan["preprocessing"])["spatial"])["target_size"]
        )
        try:
            with Image.open(path) as source:
                _require(source.format == "PNG", f"non-PNG frame refused: {path}")
                image = source.convert("RGB")
                width, height = image.size
                _require(width > 0 and height > 0, f"empty image geometry: {path}")
                scale = target / min(width, height)
                resized_width = max(target, int(math.floor(width * scale + 0.5)))
                resized_height = max(target, int(math.floor(height * scale + 0.5)))
                resized = image.resize(
                    (resized_width, resized_height),
                    resample=Image.Resampling.BILINEAR,
                    reducing_gap=None,
                )
                left = (resized_width - target) // 2
                top = (resized_height - target) // 2
                cropped = resized.crop((left, top, left + target, top + target))
                pixels = np.array(cropped, dtype=np.uint8, copy=True)
        except (OSError, ValueError) as exc:
            raise SanpoBridgeRefused(f"PNG decode failed: {path}: {exc}") from exc
        _require(pixels.shape == (target, target, 3), f"decoded frame shape drift: {path}")
        return torch.from_numpy(pixels).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)

    def _load_record(self, session: SessionRecord) -> SanpoClip:
        frames = [
            self._preprocess_frame(_safe_content_path(self.root, path, f"frame {session.session_id}"))
            for path in session.frame_paths
        ]
        # [T,C,H,W] -> [B,C,T,H,W]; one session is the maximum resident clip unit.
        tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        target = int(
            cast(dict[str, Any], cast(dict[str, Any], self.plan["preprocessing"])["spatial"])["target_size"]
        )
        _require(tuple(tensor.shape) == (1, 3, 8, target, target), "custom-substrate tensor shape drift")
        _require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), "clip is non-finite")
        _require(float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0, "clip value range drift")
        return SanpoClip(tensor=tensor, session=session, source_tensor_sha256=tensor_sha256(tensor))

    def iter_development(self) -> Iterator[SanpoClip]:
        """Yield train then validation clips; official-test pixels are structurally unreachable."""

        for session in self._sessions:
            if session.role in DEVELOPMENT_ROLES:
                yield self._load_record(session)

    def load_development_session(self, session_id: str) -> SanpoClip:
        """Load one named train/validation session and refuse a test-session ID."""

        matches = [session for session in self._sessions if session.session_id == session_id]
        _require(len(matches) == 1, f"unknown SANPO session: {session_id}")
        session = matches[0]
        _require(session.role in DEVELOPMENT_ROLES, "official-test session is sealed; use one-shot evaluator")
        return self._load_record(session)

    def _claim_official_test_capability(self, *, unlock: bool) -> _OfficialTestCapability:
        _require(unlock, "official test remains sealed; pass --unlock-official-test explicitly")
        path = self.one_shot_attempt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        initial = {
            "schema": ATTEMPT_SCHEMA,
            "created_at": _utc_now(),
            "status": "claimed-before-first-official-test-decode",
            "all_ok": False,
            "plan_identity_sha256": self.plan_identity_sha256,
            "content_set_sha256": self.content_set_sha256,
            "decoded_test_sessions": 0,
            "scientific_promotion": False,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(path, flags, 0o644)
        except FileExistsError as exc:
            raise SanpoBridgeRefused(f"official-test one-shot was already claimed: {path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(initial, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return _OfficialTestCapability(self.plan_identity_sha256, path)

    def _iter_official_test(self, capability: _OfficialTestCapability) -> Iterator[SanpoClip]:
        _require(capability.plan_identity_sha256 == self.plan_identity_sha256, "test capability plan drift")
        _require(capability.attempt_path == self.one_shot_attempt_path, "test capability ledger drift")
        for session in self._sessions:
            if session.role == "test":
                yield self._load_record(session)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_preflight(
    plan_path: Path,
    proof_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify all bytes, decode development clips, and write a durable non-model proof."""

    started = time.monotonic()
    bridge = SanpoCustomSubstrateBridge(plan_path, repo_root=repo_root, verify_content=True)
    resolved_plan_path = (plan_path if plan_path.is_absolute() else repo_root / plan_path).resolve()
    implementation_path = Path(__file__).resolve()
    decoded: list[dict[str, Any]] = []
    for clip in bridge.iter_development():
        decoded.append(
            {
                "session_id": clip.session.session_id,
                "official_split": clip.session.official_split,
                "role": clip.session.role,
                "is_park": clip.session.is_park,
                "high_level_attributes": clip.session.high_level_attributes,
                "referent_ids": list(clip.session.referent_ids),
                "frame_indices": list(clip.session.frame_indices),
                "frame_sha256s": list(clip.session.frame_sha256s),
                "tensor_shape": list(clip.tensor.shape),
                "tensor_dtype": str(clip.tensor.dtype),
                "tensor_sha256": clip.source_tensor_sha256,
                "finite": bool(torch.isfinite(clip.tensor).all()),
                "value_min": float(clip.tensor.min()),
                "value_max": float(clip.tensor.max()),
            }
        )
        del clip
    roles = [row["role"] for row in decoded]
    _require(len(decoded) == 8 and set(roles) == set(DEVELOPMENT_ROLES), "development decode count drift")
    proof: dict[str, Any] = {
        "schema": BRIDGE_PREFLIGHT_SCHEMA,
        "created_at": _utc_now(),
        "all_ok": True,
        "mode": "integrity-plus-train-validation-decode-only",
        "plan_path": str(resolved_plan_path),
        "plan_identity_sha256": bridge.plan_identity_sha256,
        "implementation": {
            "bridge_module": str(implementation_path),
            "bridge_module_sha256": sha256_file(implementation_path),
            "frozen_plan_sha256": sha256_file(resolved_plan_path),
        },
        "content_set_sha256": bridge.content_set_sha256,
        "integrity": bridge.integrity,
        "preprocessing": bridge.plan["preprocessing"],
        "runtime": {
            "python": platform.python_version(),
            "pillow": PILLOW_VERSION,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu-decode-only",
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_bytes": _max_rss_bytes(),
        },
        "development_decode": {
            "roles": list(DEVELOPMENT_ROLES),
            "sessions_decoded": len(decoded),
            "frames_decoded": len(decoded) * len(FRAME_INDICES),
            "one_session_at_a_time": True,
            "clips": decoded,
        },
        "official_test_seal": {
            "sessions": EXPECTED_ROLE_COUNTS["test"],
            "frames_sha_verified": EXPECTED_ROLE_COUNTS["test"] * len(FRAME_INDICES),
            "frames_decoded": 0,
            "pixels_inspected": False,
            "used_for_selection_or_tuning": False,
            "one_shot_attempt_receipt": str(bridge.one_shot_attempt_path),
        },
        "execution_boundary": {
            "inherited_encoder_imported_or_loaded": False,
            "portable_artifact_loaded": False,
            "model_forward": False,
            "training_or_update": False,
            "scientific_promotion": False,
            "natural_video_capability_claim": False,
            "statement": (
                "This proves a verified natural-video input bridge and deterministic development decode. "
                "It is not a representation, learning, general-capability, intelligence, or sentience result."
            ),
        },
    }
    _atomic_json(proof_path if proof_path.is_absolute() else repo_root / proof_path, proof)
    return proof


def _portable_loader(path: Path, *, device: str) -> Any:
    # Lazy import is a deliberate boundary: preflight never imports or initializes model code.
    from .custom_artifact import ArtifactRefused, load_portable_artifact

    try:
        return load_portable_artifact(path, device=device)
    except ArtifactRefused as exc:
        raise SanpoBridgeRefused(f"portable artifact verification failed: {exc}") from exc


def _extract_features(
    loaded: Any,
    clips: Iterator[SanpoClip],
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    records: list[dict[str, Any]] = []
    features: dict[str, torch.Tensor] = {}
    for clip in clips:
        with torch.inference_mode():
            output = loaded.model(clip.tensor.to(device))
        pooled = getattr(output, "pooled_retrieval_key", None)
        _require(isinstance(pooled, torch.Tensor), "portable artifact lacks pooled_retrieval_key")
        pooled = cast(torch.Tensor, pooled)
        _require(tuple(pooled.shape[:1]) == (1,) and pooled.ndim == 2, "pooled-key shape drift")
        pooled = pooled[0].detach().cpu().to(dtype=torch.float32).contiguous()
        _require(bool(torch.isfinite(pooled).all()), f"non-finite output: {clip.session.session_id}")
        features[clip.session.session_id] = pooled
        records.append(
            {
                "session_id": clip.session.session_id,
                "role": clip.session.role,
                "is_park": clip.session.is_park,
                "input_tensor_sha256": clip.source_tensor_sha256,
                "pooled_shape": list(pooled.shape),
                "pooled_sha256": tensor_sha256(pooled),
                "pooled_l2_norm": float(torch.linalg.vector_norm(pooled)),
            }
        )
        del output, pooled, clip
    return records, features


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    _require(bool(torch.isfinite(norm)) and float(norm) > 0.0, "zero or non-finite representation")
    return vector / norm


def _frozen_centroids(
    sessions: Sequence[SessionRecord],
    features: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    centroids: dict[str, torch.Tensor] = {}
    for label in (False, True):
        rows = [features[row.session_id] for row in sessions if row.role == "train" and row.is_park is label]
        _require(len(rows) == 3, f"train centroid class count drift: is_park={label}")
        centroids[str(label).lower()] = _normalize(torch.stack([_normalize(row) for row in rows]).mean(0))
    return centroids


def _diagnose(
    sessions: Sequence[SessionRecord],
    features: Mapping[str, torch.Tensor],
    centroids: Mapping[str, torch.Tensor],
    *,
    role: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        if session.role != role:
            continue
        vector = _normalize(features[session.session_id])
        scores = {label: float(torch.dot(vector, centroid)) for label, centroid in centroids.items()}
        predicted = scores["true"] >= scores["false"]
        rows.append(
            {
                "session_id": session.session_id,
                "is_park": session.is_park,
                "predicted_is_park": predicted,
                "correct": predicted is session.is_park,
                "cosine_to_nonpark_centroid": scores["false"],
                "cosine_to_park_centroid": scores["true"],
            }
        )
    _require(bool(rows), f"no sessions for diagnostic role {role}")
    return {
        "role": role,
        "n": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "rows": rows,
        "interpretation": "tiny-n interface diagnostic only; never a promotion statistic",
    }


def evaluate_development_artifact(
    plan_path: Path,
    artifact_dir: Path,
    selection_receipt_path: Path,
    *,
    device: str = "cpu",
    repo_root: Path = REPO_ROOT,
    _loader: Callable[..., Any] = _portable_loader,
) -> dict[str, Any]:
    """Select one independently verified artifact using train/validation pixels only."""

    bridge = SanpoCustomSubstrateBridge(plan_path, repo_root=repo_root, verify_content=True)
    loaded = _loader(artifact_dir, device=device)
    manifest = loaded.manifest
    _require(isinstance(manifest, dict), "portable artifact manifest is missing")
    artifact_id = manifest.get("artifact_id")
    _require(_is_sha256(artifact_id), "portable artifact ID is invalid")
    evidence = manifest.get("evidence")
    _require(isinstance(evidence, dict), "portable artifact evidence is missing")
    evidence = cast(dict[str, Any], evidence)
    _require(
        evidence.get("independent_verifier_verdict") == "promote-local-objective-lever",
        "artifact was not promoted by the independent CM7 verifier",
    )
    records, features = _extract_features(loaded, bridge.iter_development(), device=device)
    centroids = _frozen_centroids(bridge.sessions, features)
    validation = _diagnose(bridge.sessions, features, centroids, role="validation")
    manifest_path = artifact_dir / "manifest.json"
    receipt: dict[str, Any] = {
        "schema": DEVELOPMENT_SELECTION_SCHEMA,
        "created_at": _utc_now(),
        "all_ok": True,
        "selection_scope": "train-plus-validation-only",
        "plan_identity_sha256": bridge.plan_identity_sha256,
        "content_set_sha256": bridge.content_set_sha256,
        "artifact": {
            "artifact_id": artifact_id,
            "artifact_dir": str(artifact_dir.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "independent_verifier_verdict": evidence["independent_verifier_verdict"],
        },
        "features": records,
        "frozen_train_centroids": {
            label: {
                "values": centroid.tolist(),
                "sha256": tensor_sha256(centroid),
            }
            for label, centroid in centroids.items()
        },
        "validation_diagnostic": validation,
        "official_test": {
            "sessions_decoded": 0,
            "pixels_used": False,
            "eligible_for_one_shot_command": True,
        },
        "claim_boundary": {
            "artifact_selected_for_interface_compatibility": True,
            "selection_depended_on_official_test": False,
            "natural_video_scientific_promotion": False,
            "general_capability_promotion": False,
            "statement": "Validation n=2 is a development diagnostic, not scientific evidence.",
        },
    }
    receipt["selection_identity_sha256"] = json_sha256(
        {
            "plan_identity_sha256": bridge.plan_identity_sha256,
            "content_set_sha256": bridge.content_set_sha256,
            "artifact": receipt["artifact"],
            "frozen_train_centroids": receipt["frozen_train_centroids"],
        }
    )
    _atomic_json(
        selection_receipt_path
        if selection_receipt_path.is_absolute()
        else repo_root / selection_receipt_path,
        receipt,
    )
    return receipt


def evaluate_official_test_once(
    plan_path: Path,
    artifact_dir: Path,
    selection_receipt_path: Path,
    *,
    unlock_official_test: bool,
    device: str = "cpu",
    repo_root: Path = REPO_ROOT,
    _loader: Callable[..., Any] = _portable_loader,
) -> dict[str, Any]:
    """Run the fixed n=2 test once, with no tuning and no possible scientific promotion."""

    _require(unlock_official_test, "official test remains sealed; pass --unlock-official-test explicitly")
    bridge = SanpoCustomSubstrateBridge(plan_path, repo_root=repo_root, verify_content=True)
    selection_path = (
        selection_receipt_path if selection_receipt_path.is_absolute() else repo_root / selection_receipt_path
    )
    selection = _read_json(selection_path, "development selection receipt")
    _require(
        selection.get("schema") == DEVELOPMENT_SELECTION_SCHEMA and selection.get("all_ok") is True,
        "development selection receipt failed",
    )
    _require(selection.get("selection_scope") == "train-plus-validation-only", "selection scope drift")
    _require(selection.get("plan_identity_sha256") == bridge.plan_identity_sha256, "selection plan drift")
    _require(selection.get("content_set_sha256") == bridge.content_set_sha256, "selection content drift")
    selection_boundary = selection.get("claim_boundary")
    _require(isinstance(selection_boundary, dict), "selection claim boundary is missing")
    selection_boundary = cast(dict[str, Any], selection_boundary)
    _require(
        selection_boundary.get("selection_depended_on_official_test") is False,
        "selection receipt used official-test data",
    )
    _require(
        cast(dict[str, Any], selection.get("official_test", {}))
        == {
            "sessions_decoded": 0,
            "pixels_used": False,
            "eligible_for_one_shot_command": True,
        },
        "selection receipt already used official-test pixels",
    )
    artifact_record = selection.get("artifact")
    _require(isinstance(artifact_record, dict), "selection artifact record is missing")
    artifact_record = cast(dict[str, Any], artifact_record)
    _require(
        Path(str(artifact_record.get("artifact_dir"))).resolve() == artifact_dir.resolve(),
        "artifact path drift",
    )
    _require(
        sha256_file(artifact_dir / "manifest.json") == artifact_record.get("manifest_sha256"),
        "artifact drift",
    )
    loaded = _loader(artifact_dir, device=device)
    _require(loaded.manifest.get("artifact_id") == artifact_record.get("artifact_id"), "artifact ID drift")
    centroid_records = selection.get("frozen_train_centroids")
    _require(isinstance(centroid_records, dict), "frozen train centroids are missing")
    centroid_records = cast(dict[str, dict[str, Any]], centroid_records)
    expected_selection_identity = json_sha256(
        {
            "plan_identity_sha256": bridge.plan_identity_sha256,
            "content_set_sha256": bridge.content_set_sha256,
            "artifact": artifact_record,
            "frozen_train_centroids": centroid_records,
        }
    )
    _require(
        selection.get("selection_identity_sha256") == expected_selection_identity,
        "development selection identity drift",
    )
    centroids: dict[str, torch.Tensor] = {}
    for label in ("false", "true"):
        record = centroid_records.get(label)
        _require(isinstance(record, dict), f"frozen centroid missing: {label}")
        record = cast(dict[str, Any], record)
        vector = torch.tensor(record.get("values"), dtype=torch.float32)
        _require(vector.ndim == 1 and bool(torch.isfinite(vector).all()), f"invalid centroid: {label}")
        _require(tensor_sha256(vector) == record.get("sha256"), f"centroid SHA drift: {label}")
        centroids[label] = vector

    capability = bridge._claim_official_test_capability(unlock=True)
    started = time.monotonic()
    try:
        feature_records, features = _extract_features(
            loaded,
            bridge._iter_official_test(capability),
            device=device,
        )
        diagnostic = _diagnose(bridge.sessions, features, centroids, role="test")
        _require(diagnostic["n"] == 2, "official-test count drift")
        result: dict[str, Any] = {
            "schema": OFFICIAL_TEST_SCHEMA,
            "created_at": _utc_now(),
            "all_ok": True,
            "status": "completed-one-shot-no-tuning",
            "plan_identity_sha256": bridge.plan_identity_sha256,
            "content_set_sha256": bridge.content_set_sha256,
            "selection_receipt_sha256": sha256_file(selection_path),
            "selection_identity_sha256": selection.get("selection_identity_sha256"),
            "artifact": artifact_record,
            "frozen_train_centroid_sha256s": {
                label: centroid_records[label]["sha256"] for label in ("false", "true")
            },
            "official_test_features": feature_records,
            "official_test_diagnostic": diagnostic,
            "elapsed_seconds": time.monotonic() - started,
            "claim_boundary": {
                "n": 2,
                "tuning_after_test_allowed": False,
                "rerun_allowed": False,
                "scientific_promotion": False,
                "generalization_claim": False,
                "statement": (
                    "This fixed two-session result is an interface smoke diagnostic only. It is too small "
                    "for promotion and may not trigger tuning, architecture changes, or a second test run."
                ),
            },
        }
    except Exception as exc:
        failure = {
            "schema": ATTEMPT_SCHEMA,
            "created_at": _utc_now(),
            "status": "consumed-and-failed",
            "all_ok": False,
            "plan_identity_sha256": bridge.plan_identity_sha256,
            "content_set_sha256": bridge.content_set_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_promotion": False,
            "rerun_allowed": False,
        }
        _atomic_json(capability.attempt_path, failure)
        raise
    _atomic_json(capability.attempt_path, result)
    return result


__all__ = [
    "ATTEMPT_SCHEMA",
    "BRIDGE_PLAN_SCHEMA",
    "BRIDGE_PREFLIGHT_SCHEMA",
    "DEVELOPMENT_SELECTION_SCHEMA",
    "FRAME_INDICES",
    "OFFICIAL_TEST_SCHEMA",
    "SanpoBridgeRefused",
    "SanpoClip",
    "SanpoCustomSubstrateBridge",
    "SessionRecord",
    "evaluate_development_artifact",
    "evaluate_official_test_once",
    "json_sha256",
    "load_bridge_plan",
    "run_preflight",
    "sha256_file",
    "tensor_sha256",
]
