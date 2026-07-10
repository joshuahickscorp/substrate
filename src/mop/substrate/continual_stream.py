"""Disk-backed deterministic event stream for bounded continual-learning studies.

The stream stores fixed-width records in atomically published chunks. Each record digest includes
the prior digest, so order, content, and resume position share one identity. This module generates
and verifies mechanics. It does not train a model or establish a continual-learning result.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .events import EntityRef, EventRef, canonical_bytes, canonical_sha256

STREAM_SPEC_SCHEMA = "mop-continual-stream-spec/v1"
STREAM_MANIFEST_SCHEMA = "mop-continual-stream-manifest/v1"
RECORD_CORE = struct.Struct("<QHHHHHB")
RECORD = struct.Struct("<QHHHHHB32s")
FLAG_ACTIVE_SECONDARY = 1
FLAG_TRANSITION = 2
FLAG_DELETE = 4


class TransitionSchedule(StrEnum):
    ABRUPT = "abrupt"
    GRADUAL = "gradual"


@dataclass(frozen=True, slots=True)
class ContinualStreamSpec:
    seed: int
    total_events: int
    chunk_events: int
    n_domains: int
    n_classes: int
    transition_schedule: TransitionSchedule
    gradual_width_events: int
    deletion_event: int
    schema: str = STREAM_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STREAM_SPEC_SCHEMA:
            raise ValueError(f"unsupported stream spec schema {self.schema!r}")
        if self.seed < 0 or self.total_events < 1 or self.chunk_events < 1:
            raise ValueError("stream seed and event counts are invalid")
        if self.n_domains < 2 or self.n_classes < 2:
            raise ValueError("stream needs at least two domains and classes")
        if self.n_domains > 65_535 or self.n_classes > 65_535:
            raise ValueError("stream domain or class count exceeds the record format")
        if self.gradual_width_events < 1 or self.gradual_width_events >= self.total_events:
            raise ValueError("gradual transition width is invalid")
        segment = self.total_events // self.n_domains
        if self.gradual_width_events >= segment:
            raise ValueError("gradual transition width must be smaller than a domain segment")
        if not 1 <= self.deletion_event < self.total_events:
            raise ValueError("deletion event must be inside the stream")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "total_events": self.total_events,
            "chunk_events": self.chunk_events,
            "n_domains": self.n_domains,
            "n_classes": self.n_classes,
            "transition_schedule": str(self.transition_schedule),
            "gradual_width_events": self.gradual_width_events,
            "deletion_event": self.deletion_event,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @property
    def stream_ref(self) -> str:
        return f"stream:{self.identity_sha256[:20]}"


@dataclass(frozen=True, slots=True)
class ContinualEvent:
    sequence: int
    domain_a: int
    domain_b: int
    blend_milli: int
    cue: int
    label: int
    flags: int
    content_sha256: str
    stream_identity_sha256: str

    @property
    def active_domain(self) -> int:
        return self.domain_b if self.flags & FLAG_ACTIVE_SECONDARY else self.domain_a

    @property
    def transition(self) -> bool:
        return bool(self.flags & FLAG_TRANSITION)

    @property
    def deletion_requested(self) -> bool:
        return bool(self.flags & FLAG_DELETE)

    @property
    def event_ref(self) -> EventRef:
        return EventRef(f"event:{self.stream_identity_sha256[:16]}/stream:{self.sequence:012d}")

    @property
    def entity_ref(self) -> EntityRef:
        return EntityRef(f"entity:{self.stream_identity_sha256[:16]}/cue:{self.cue:05d}")

    @property
    def stage(self) -> int:
        return max(self.domain_a, self.domain_b)

    def payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_ref": str(self.event_ref),
            "entity_ref": str(self.entity_ref),
            "domain_a": self.domain_a,
            "domain_b": self.domain_b,
            "active_domain": self.active_domain,
            "blend_milli": self.blend_milli,
            "cue": self.cue,
            "label": self.label,
            "transition": self.transition,
            "deletion_requested": self.deletion_requested,
            "content_sha256": self.content_sha256,
        }


def _stable_int(spec: ContinualStreamSpec, sequence: int, label: str, modulus: int) -> int:
    if modulus < 1:
        raise ValueError("stable integer modulus must be positive")
    raw = canonical_bytes(
        {
            "stream_identity_sha256": spec.identity_sha256,
            "sequence": sequence,
            "label": label,
        }
    )
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % modulus


def _schedule_fields(spec: ContinualStreamSpec, sequence: int) -> tuple[int, int, int, int]:
    segment = spec.total_events / spec.n_domains
    default_domain = min(spec.n_domains - 1, int(sequence / segment))
    if spec.transition_schedule is TransitionSchedule.ABRUPT:
        transition = sequence > 0 and int((sequence - 1) / segment) != default_domain
        return default_domain, default_domain, 0, FLAG_TRANSITION if transition else 0

    half_width = spec.gradual_width_events / 2.0
    for next_domain in range(1, spec.n_domains):
        boundary = next_domain * segment
        start = boundary - half_width
        end = boundary + half_width
        if start <= sequence < end:
            blend = int(round(1000.0 * (sequence - start) / max(1.0, end - start - 1.0)))
            blend = max(0, min(1000, blend))
            choose_secondary = _stable_int(spec, sequence, "gradual-choice", 1000) < blend
            flags = FLAG_TRANSITION | (FLAG_ACTIVE_SECONDARY if choose_secondary else 0)
            return next_domain - 1, next_domain, blend, flags
    return default_domain, default_domain, 0, 0


def _record_fields(spec: ContinualStreamSpec, sequence: int) -> tuple[int, int, int, int, int, int, int]:
    domain_a, domain_b, blend, flags = _schedule_fields(spec, sequence)
    cue = _stable_int(spec, sequence, "cue", spec.n_classes)
    active_domain = domain_b if flags & FLAG_ACTIVE_SECONDARY else domain_a
    label = (cue + active_domain) % spec.n_classes
    if sequence == spec.deletion_event:
        flags |= FLAG_DELETE
    return sequence, domain_a, domain_b, blend, cue, label, flags


def _record_digest(
    spec: ContinualStreamSpec,
    fields: tuple[int, int, int, int, int, int, int],
    previous_digest: bytes,
) -> bytes:
    return hashlib.sha256(
        bytes.fromhex(spec.identity_sha256) + previous_digest + RECORD_CORE.pack(*fields)
    ).digest()


def _decode_record(
    spec: ContinualStreamSpec, raw: bytes, previous_digest: bytes
) -> tuple[ContinualEvent, bytes]:
    if len(raw) != RECORD.size:
        raise ValueError("continual stream record has the wrong byte length")
    unpacked = RECORD.unpack(raw)
    fields = unpacked[:7]
    digest = unpacked[7]
    expected = _record_digest(spec, fields, previous_digest)
    if digest != expected:
        raise ValueError(f"continual stream record digest drift at sequence {fields[0]}")
    event = ContinualEvent(
        sequence=int(fields[0]),
        domain_a=int(fields[1]),
        domain_b=int(fields[2]),
        blend_milli=int(fields[3]),
        cue=int(fields[4]),
        label=int(fields[5]),
        flags=int(fields[6]),
        content_sha256=digest.hex(),
        stream_identity_sha256=spec.identity_sha256,
    )
    return event, digest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _spec_from_payload(payload: dict[str, Any]) -> ContinualStreamSpec:
    return ContinualStreamSpec(
        seed=int(payload["seed"]),
        total_events=int(payload["total_events"]),
        chunk_events=int(payload["chunk_events"]),
        n_domains=int(payload["n_domains"]),
        n_classes=int(payload["n_classes"]),
        transition_schedule=TransitionSchedule(str(payload["transition_schedule"])),
        gradual_width_events=int(payload["gradual_width_events"]),
        deletion_event=int(payload["deletion_event"]),
        schema=str(payload["schema"]),
    )


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def read_manifest(root: Path | str) -> dict[str, Any]:
    path = _manifest_path(Path(root))
    if not path.is_file():
        raise FileNotFoundError(f"continual stream manifest missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continual stream manifest must be a mapping")
    return payload


def stream_sha256(manifest: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "identity_sha256": manifest["identity_sha256"],
            "chunks": [
                {
                    "index": row["index"],
                    "start_sequence": row["start_sequence"],
                    "count": row["count"],
                    "sha256": row["sha256"],
                    "chain_head_sha256": row["chain_head_sha256"],
                }
                for row in manifest["chunks"]
            ],
            "chain_head_sha256": manifest["chain_head_sha256"],
            "generated_events": manifest["generated_events"],
        }
    )


def materialize_stream(
    root: Path | str,
    spec: ContinualStreamSpec,
    *,
    max_new_chunks: int | None = None,
) -> dict[str, Any]:
    """Create or resume an atomically chunked stream under an exact spec identity."""

    path = Path(root)
    manifest_path = _manifest_path(path)
    if manifest_path.is_file():
        manifest = read_manifest(path)
        if (
            manifest.get("schema") != STREAM_MANIFEST_SCHEMA
            or manifest.get("identity_sha256") != spec.identity_sha256
            or manifest.get("spec") != spec.payload()
        ):
            raise ValueError("continual stream resume identity drift")
        audit = verify_stream(path, expected_spec=spec, require_complete=False)
        if not audit["verified"]:
            raise ValueError("continual stream resume verification failed: " + "; ".join(audit["errors"]))
        if manifest.get("complete") is True:
            return manifest
    else:
        manifest = {
            "schema": STREAM_MANIFEST_SCHEMA,
            "spec": spec.payload(),
            "identity_sha256": spec.identity_sha256,
            "record_bytes": RECORD.size,
            "chunks": [],
            "generated_events": 0,
            "chain_head_sha256": "00" * 32,
            "complete": False,
            "stream_sha256": None,
        }
        _atomic_json(manifest_path, manifest)

    start = int(manifest["generated_events"])
    previous_digest = bytes.fromhex(str(manifest["chain_head_sha256"]))
    new_chunks = 0
    while start < spec.total_events:
        count = min(spec.chunk_events, spec.total_events - start)
        raw = bytearray()
        chunk_start_head = previous_digest.hex()
        for sequence in range(start, start + count):
            fields = _record_fields(spec, sequence)
            digest = _record_digest(spec, fields, previous_digest)
            raw.extend(RECORD.pack(*fields, digest))
            previous_digest = digest
        index = len(manifest["chunks"])
        filename = f"chunk_{index:06d}.bin"
        _atomic_bytes(path / filename, bytes(raw))
        manifest["chunks"].append(
            {
                "index": index,
                "path": filename,
                "start_sequence": start,
                "count": count,
                "bytes": len(raw),
                "sha256": _sha256_bytes(bytes(raw)),
                "chain_start_sha256": chunk_start_head,
                "chain_head_sha256": previous_digest.hex(),
            }
        )
        start += count
        manifest["generated_events"] = start
        manifest["chain_head_sha256"] = previous_digest.hex()
        _atomic_json(manifest_path, manifest)
        new_chunks += 1
        if max_new_chunks is not None and new_chunks >= max_new_chunks:
            return manifest

    manifest["complete"] = True
    manifest["stream_sha256"] = stream_sha256(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def verify_stream(
    root: Path | str,
    *,
    expected_spec: ContinualStreamSpec | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    path = Path(root)
    errors: list[str] = []
    try:
        manifest = read_manifest(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
        return {"verified": False, "errors": [str(exc)], "record_count": 0}
    if manifest.get("schema") != STREAM_MANIFEST_SCHEMA:
        errors.append("continual stream manifest schema drift")
    try:
        spec = _spec_from_payload(manifest["spec"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"verified": False, "errors": [f"invalid stream spec: {exc}"], "record_count": 0}
    if manifest.get("identity_sha256") != spec.identity_sha256:
        errors.append("continual stream spec identity drift")
    if expected_spec is not None and expected_spec != spec:
        errors.append("continual stream does not match the expected spec")
    if manifest.get("record_bytes") != RECORD.size:
        errors.append("continual stream record width drift")

    previous_digest = bytes(32)
    expected_sequence = 0
    chunk_rows = manifest.get("chunks", [])
    for expected_index, row in enumerate(chunk_rows):
        chunk_path = path / str(row.get("path"))
        if row.get("index") != expected_index or row.get("start_sequence") != expected_sequence:
            errors.append(f"continual stream chunk {expected_index} index drift")
        if row.get("chain_start_sha256") != previous_digest.hex():
            errors.append(f"continual stream chunk {expected_index} chain start drift")
        if not chunk_path.is_file():
            errors.append(f"continual stream chunk {expected_index} missing")
            continue
        raw = chunk_path.read_bytes()
        count = int(row.get("count", -1))
        if len(raw) != count * RECORD.size or row.get("bytes") != len(raw):
            errors.append(f"continual stream chunk {expected_index} byte count drift")
            continue
        if row.get("sha256") != _sha256_bytes(raw):
            errors.append(f"continual stream chunk {expected_index} file digest drift")
        for offset in range(count):
            record_raw = raw[offset * RECORD.size : (offset + 1) * RECORD.size]
            try:
                event, previous_digest = _decode_record(spec, record_raw, previous_digest)
            except ValueError as exc:
                errors.append(str(exc))
                break
            if event.sequence != expected_sequence:
                errors.append(f"continual stream sequence drift at {expected_sequence}")
            expected_sequence += 1
        if row.get("chain_head_sha256") != previous_digest.hex():
            errors.append(f"continual stream chunk {expected_index} chain head drift")

    if manifest.get("generated_events") != expected_sequence:
        errors.append("continual stream generated event count drift")
    if manifest.get("chain_head_sha256") != previous_digest.hex():
        errors.append("continual stream manifest chain head drift")
    complete = manifest.get("complete") is True
    if require_complete and not complete:
        errors.append("continual stream is incomplete")
    if complete:
        if expected_sequence != spec.total_events:
            errors.append("complete continual stream has the wrong event count")
        if manifest.get("stream_sha256") != stream_sha256(manifest):
            errors.append("continual stream composite digest drift")
    elif manifest.get("stream_sha256") is not None:
        errors.append("incomplete continual stream cannot publish a composite digest")
    return {
        "verified": not errors,
        "errors": errors,
        "record_count": expected_sequence,
        "complete": complete,
        "identity_sha256": spec.identity_sha256,
        "stream_sha256": manifest.get("stream_sha256"),
        "chain_head_sha256": previous_digest.hex(),
        "chunk_count": len(chunk_rows),
        "disk_bytes": sum(_sha_file_size(path / str(row.get("path"))) for row in chunk_rows),
    }


def _sha_file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def iter_stream(
    root: Path | str,
    *,
    start_sequence: int = 0,
    stop_sequence: int | None = None,
) -> Iterator[ContinualEvent]:
    path = Path(root)
    manifest = read_manifest(path)
    spec = _spec_from_payload(manifest["spec"])
    generated = int(manifest["generated_events"])
    stop = generated if stop_sequence is None else min(stop_sequence, generated)
    if start_sequence < 0 or stop < start_sequence:
        raise ValueError("continual stream iteration range is invalid")
    previous_digest = bytes(32)
    for row in manifest["chunks"]:
        raw = (path / row["path"]).read_bytes()
        count = int(row["count"])
        for offset in range(count):
            record_raw = raw[offset * RECORD.size : (offset + 1) * RECORD.size]
            event, previous_digest = _decode_record(spec, record_raw, previous_digest)
            if event.sequence >= stop:
                return
            if event.sequence >= start_sequence:
                yield event


def event_at(root: Path | str, sequence: int) -> ContinualEvent:
    events = list(iter_stream(root, start_sequence=sequence, stop_sequence=sequence + 1))
    if len(events) != 1:
        raise IndexError(f"continual stream event {sequence} is unavailable")
    return events[0]
