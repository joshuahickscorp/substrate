from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "mop-wikimedia-commons-av-intake-plan/v1"
RECEIPT_SCHEMA = "mop-wikimedia-commons-av-intake-receipt/v1"
MANIFEST_SCHEMA = "mop-wikimedia-commons-av-cohort/v1"

DEFAULT_MANIFEST = Path("configs/audiovisual/wikimedia_commons_cc0_v1.json")
DEFAULT_DESTINATION = Path("data/raw/wikimedia_commons_cc0_av_mechanics_v1")
DEFAULT_DRY_RUN_PROOF = Path("proof/WIKIMEDIA_AV_INTAKE_DRY_RUN.json")

API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
OBJECT_HOST = "upload.wikimedia.org"
MIN_FREE_DISK_BYTES = 40_000_000_000
ALLOWED_ROLES = ("train", "validation", "test")
DOWNLOAD_ROLES = ("train", "validation")
CHUNK_BYTES = 1 << 20

POST_CM7_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/studio/wikimedia_av_intake.py "
    "--execute-train-validation --confirm-cm7-complete "
    "--manifest configs/audiovisual/wikimedia_commons_cc0_v1.json "
    "--destination data/raw/wikimedia_commons_cc0_av_mechanics_v1 "
    "--proof proof/WIKIMEDIA_AV_TRAIN_VALIDATION_INTAKE.json"
)

OFFICIAL_SOURCES = {
    "commons_scope": "https://commons.wikimedia.org/wiki/Commons:Project_scope/Summary",
    "commons_licensing": "https://commons.wikimedia.org/wiki/Commons:Licensing",
    "commons_non_copyright": "https://commons.wikimedia.org/wiki/Commons:Non-copyright_restrictions",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "ravdess": "https://zenodo.org/records/1188976",
    "epic_sounds": "https://epic-kitchens.github.io/epic-sounds/site",
    "tau_av": "https://zenodo.org/records/4477542",
}

CANDIDATE_COMPARISON: tuple[dict[str, Any], ...] = (
    {
        "candidate": "Frozen Wikimedia Commons CC0 originals",
        "decision": "selected_for_mechanics",
        "synchronized_av": True,
        "authority": "per-object pageid, original URL, upload timestamp, size and Wikimedia SHA-1",
        "rights": "per-object CC0, live checked; derivative and commercial copyright reuse permitted",
        "selectivity": "individual original muxed files",
        "projected_bytes": 95_791_426,
        "split": "curator-frozen 8/2/2 before any media access",
        "privacy": "metadata-screened; manual audio-video review remains mandatory",
        "claim_boundary": "timing/shuffle mechanics only",
    },
    {
        "candidate": "RAVDESS on Zenodo",
        "decision": "reserve_candidate_not_selected",
        "synchronized_av": True,
        "authority": "Zenodo DOI and per-archive checksums",
        "rights": "CC BY-NC-SA 4.0",
        "selectivity": "actor archives are approximately 500 MB each",
        "projected_bytes": "approximately 1.5 GB for a minimal three-actor split",
        "split": "speaker-disjoint split would be curator-defined",
        "privacy": "professional identifiable actors; identity handling remains material",
        "claim_boundary": "clean speech alignment, but larger and less permissive than CC0 cohort",
    },
    {
        "candidate": "EPIC-SOUNDS plus EPIC-KITCHENS-100 video",
        "decision": "defer_natural_benchmark",
        "synchronized_av": True,
        "authority": "official Oxford/Bristol annotations and source recordings",
        "rights": "CC BY-NC 4.0 for EPIC-SOUNDS; source-video obligations also apply",
        "selectivity": "annotations are small, source RGB recordings are the acquisition unit",
        "projected_bytes": "not bounded to a sub-100 MB original-object cohort",
        "split": "official benchmark discipline available",
        "privacy": "egocentric human environments require the dataset's privacy protocol",
        "claim_boundary": "strong later natural benchmark, not the smallest local mechanics intake",
    },
    {
        "candidate": "TAU Urban Audio-Visual Scenes 2021",
        "decision": "reject_for_current_host_intake",
        "synchronized_av": True,
        "authority": "Zenodo DOI and MD5 per archive",
        "rights": "publisher-hosted record, but rights must still be checked in bundled documentation",
        "selectivity": "large audio and video ZIP parts rather than individual muxed clips",
        "projected_bytes": "107.7 GB development release",
        "split": "official development protocol",
        "privacy": "public urban scenes require review",
        "claim_boundary": "valid later benchmark; wrong granularity for the current disk envelope",
    },
    {
        "candidate": "AudioSet, VGGSound, AVE and link-list derivatives",
        "decision": "rejected",
        "synchronized_av": "nominally",
        "authority": "external YouTube links, not stable publisher-owned media objects",
        "rights": "per-video rights and redistribution are not cleared by annotation licenses",
        "selectivity": "links are selective but volatile",
        "projected_bytes": "not relevant",
        "split": "published IDs do not repair media authority or rights",
        "privacy": "heterogeneous web video",
        "claim_boundary": "must not be used for this rights-cleared intake",
    },
)


class WikimediaAVIntakeError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_values(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        raise WikimediaAVIntakeError("Wikimedia stream metadata is not a named-value list")
    result: dict[str, Any] = {}
    for item in values:
        if not isinstance(item, dict) or "name" not in item or "value" not in item:
            continue
        result[str(item["name"])] = item["value"]
    return result


def _ext_value(extmetadata: Mapping[str, Any], key: str) -> str:
    raw = extmetadata.get(key, {})
    if not isinstance(raw, Mapping):
        return ""
    return str(raw.get("value", ""))


def _normalize_live_page(page: Mapping[str, Any]) -> dict[str, Any]:
    rows = page.get("videoinfo")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise WikimediaAVIntakeError(f"missing single videoinfo row for {page.get('title')}")
    info = rows[0]
    metadata = _named_values(info.get("metadata"))
    audio = _named_values(metadata.get("audio"))
    video = _named_values(metadata.get("video"))
    extmetadata = info.get("extmetadata")
    if not isinstance(extmetadata, Mapping):
        raise WikimediaAVIntakeError(f"missing extmetadata for {page.get('title')}")
    categories = [value for value in _ext_value(extmetadata, "Categories").split("|") if value]
    return {
        "pageid": int(page["pageid"]),
        "title": str(page["title"]),
        "object_url": str(info.get("url", "")),
        "description_url": str(info.get("descriptionurl", "")),
        "size_bytes": int(info.get("size", -1)),
        "duration_seconds": float(info.get("duration", -1.0)),
        "sha1": str(info.get("sha1", "")),
        "upload_timestamp": str(info.get("timestamp", "")),
        "mime": str(info.get("mime", "")),
        "mediatype": str(info.get("mediatype", "")),
        "video": {
            "codec": str(video.get("dataformat", "")),
            "width": int(video.get("resolution_x", -1)),
            "height": int(video.get("resolution_y", -1)),
            "fps": float(video.get("frame_rate", -1.0)),
        },
        "audio": {
            "codec": str(audio.get("dataformat", "")),
            "sample_rate": int(audio.get("sample_rate", -1)),
            "channels": int(audio.get("channels", -1)),
            "language": str(audio.get("language", "")),
        },
        "license_short_name": _ext_value(extmetadata, "LicenseShortName"),
        "license_code": _ext_value(extmetadata, "License"),
        "license_url": _ext_value(extmetadata, "LicenseUrl"),
        "attribution_required": _ext_value(extmetadata, "AttributionRequired"),
        "restrictions": _ext_value(extmetadata, "Restrictions"),
        "credit": _ext_value(extmetadata, "Credit"),
        "artist": _ext_value(extmetadata, "Artist"),
        "categories": categories,
    }


class WikimediaCommonsAPI:
    def __init__(
        self,
        *,
        timeout: float = 60.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.timeout = float(timeout)
        self._opener = opener

    def pages(self, pageids: Iterable[int]) -> dict[int, dict[str, Any]]:
        ids = [int(value) for value in pageids]
        if not ids or len(ids) > 50:
            raise WikimediaAVIntakeError("metadata query requires 1 to 50 page ids")
        query = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "videoinfo",
                "pageids": "|".join(str(value) for value in ids),
                "viprop": "url|size|sha1|mime|mediatype|timestamp|metadata|extmetadata",
                "viextmetadatalanguage": "en",
            }
        )
        request = urllib.request.Request(
            f"{API_ENDPOINT}?{query}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "mop-wikimedia-av-intake/1.0 (metadata-only research preflight)",
            },
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise WikimediaAVIntakeError(f"Wikimedia metadata request failed: {exc}") from exc
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            raise WikimediaAVIntakeError("Wikimedia response omitted query.pages")
        normalized = {_page["pageid"]: _page for _page in (_normalize_live_page(row) for row in pages)}
        missing = sorted(set(ids) - set(normalized))
        extras = sorted(set(normalized) - set(ids))
        if missing or extras:
            raise WikimediaAVIntakeError(f"Wikimedia page-id mismatch: missing={missing}, extras={extras}")
        return normalized


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WikimediaAVIntakeError(f"could not read cohort manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise WikimediaAVIntakeError("cohort manifest must be a JSON object")
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    problems: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema must be {MANIFEST_SCHEMA}")
    if manifest.get("frozen_before_media_access") is not True:
        problems.append("manifest is not frozen before media access")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise WikimediaAVIntakeError("manifest objects must be a non-empty list")
    roles = Counter(str(row.get("role")) for row in objects if isinstance(row, Mapping))
    expected_counts = manifest.get("split_policy", {}).get("counts", {})
    if roles != Counter({key: int(value) for key, value in expected_counts.items()}):
        problems.append(f"role counts {dict(roles)} do not match frozen counts {expected_counts}")
    if set(roles) != set(ALLOWED_ROLES):
        problems.append(f"roles must be exactly {ALLOWED_ROLES}")

    unique_fields = ("referent_id", "capture_family", "pageid", "title", "artist", "sha1", "object_url")
    for field in unique_fields:
        values = [row.get(field) for row in objects]
        if any(value in (None, "") for value in values) or len(set(values)) != len(values):
            problems.append(f"{field} values must be present and unique")

    total_bytes = 0
    by_role: Counter[str] = Counter()
    for row in objects:
        if not isinstance(row, Mapping):
            problems.append("object rows must be mappings")
            continue
        role = str(row.get("role"))
        size = int(row.get("size_bytes", -1))
        total_bytes += size
        by_role[role] += size
        parsed = urllib.parse.urlparse(str(row.get("object_url", "")))
        if parsed.scheme != "https" or parsed.netloc != OBJECT_HOST:
            problems.append(f"{row.get('referent_id')}: object URL is not an official HTTPS object")
        description = urllib.parse.urlparse(str(row.get("description_url", "")))
        if description.scheme != "https" or description.netloc != "commons.wikimedia.org":
            problems.append(f"{row.get('referent_id')}: description URL is not an official Commons page")
        if not re.fullmatch(r"[a-z0-9-]+", str(row.get("referent_id", ""))):
            problems.append(f"{row.get('referent_id')}: unsafe referent id")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row.get("sha1", ""))):
            problems.append(f"{row.get('referent_id')}: SHA-1 is not a 40-character hex digest")
        if row.get("license_short_name") != "CC0" or row.get("license_code") != "cc0":
            problems.append(f"{row.get('referent_id')}: per-object license is not CC0")
        if row.get("credit") != "Own work":
            problems.append(f"{row.get('referent_id')}: frozen cohort requires creator-owned work")
        video = row.get("video", {})
        audio = row.get("audio", {})
        if any(float(video.get(key, -1)) <= 0 for key in ("width", "height", "fps")):
            problems.append(f"{row.get('referent_id')}: invalid video stream declaration")
        if any(int(audio.get(key, -1)) <= 0 for key in ("sample_rate", "channels")):
            problems.append(f"{row.get('referent_id')}: invalid audio stream declaration")
        if float(row.get("duration_seconds", -1)) <= 0 or size <= 0:
            problems.append(f"{row.get('referent_id')}: invalid size or duration")

    disk = manifest.get("disk_policy", {})
    if int(disk.get("min_free_disk_bytes", 0)) < MIN_FREE_DISK_BYTES:
        problems.append("disk floor was weakened below 40 GB")
    if int(disk.get("projected_total_bytes", -1)) != total_bytes:
        problems.append("projected total bytes do not equal object sum")
    if int(disk.get("projected_train_validation_bytes", -1)) != sum(by_role[role] for role in DOWNLOAD_ROLES):
        problems.append("projected train/validation bytes do not equal object sum")
    if int(disk.get("projected_locked_test_bytes", -1)) != by_role["test"]:
        problems.append("projected locked-test bytes do not equal object sum")
    if int(disk.get("atomic_reserve_multiplier", 0)) < 2:
        problems.append("atomic reserve multiplier must be at least two")

    split = manifest.get("split_policy", {})
    if split.get("official_split") is not False or split.get("frozen_before_download_or_decode") is not True:
        problems.append("curator split is not explicitly frozen and non-official")
    privacy = manifest.get("privacy_policy", {})
    if privacy.get("manual_audio_visual_review_required_after_download") is not True:
        problems.append("manual privacy review was not required")
    if privacy.get("scientific_use_before_review") is not False:
        problems.append("scientific use must be false before privacy review")
    controls = manifest.get("temporal_controls", {})
    if controls.get("no_proxy_audio") is not True or controls.get("no_caption_as_audio") is not True:
        problems.append("proxy audio or caption substitution was not forbidden")
    if controls.get("no_test_control_tuning") is not True:
        problems.append("test control tuning was not forbidden")
    fractions = controls.get("within_clip_circular_offset_fractions")
    if fractions != [0.25, 0.5, 0.75]:
        problems.append("temporal-offset fractions changed from the frozen plan")
    if problems:
        raise WikimediaAVIntakeError("invalid cohort manifest: " + "; ".join(problems))


def _compare_object(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> list[str]:
    prefix = str(expected.get("referent_id"))
    problems: list[str] = []
    for field in (
        "pageid",
        "title",
        "object_url",
        "size_bytes",
        "sha1",
        "upload_timestamp",
        "license_short_name",
        "license_code",
        "license_url",
    ):
        if observed.get(field) != expected.get(field):
            problems.append(f"{prefix}: {field} drifted")
    if observed.get("mime") != "video/webm" or observed.get("mediatype") != "VIDEO":
        problems.append(f"{prefix}: object is not an original WebM video")
    if observed.get("attribution_required") != "false":
        problems.append(f"{prefix}: CC0 attribution metadata changed")
    if observed.get("restrictions"):
        problems.append(f"{prefix}: Commons extmetadata now reports restrictions")
    categories = set(observed.get("categories", []))
    if "CC-Zero" not in categories:
        problems.append(f"{prefix}: CC-Zero category missing")
    if "Videos containing non-free audio" in categories:
        problems.append(f"{prefix}: non-free audio category present")
    if "Own work" not in str(observed.get("credit", "")):
        problems.append(f"{prefix}: live credit is no longer creator-owned work")
    if not math.isclose(
        float(observed.get("duration_seconds", -1)),
        float(expected.get("duration_seconds", -2)),
        abs_tol=0.001,
    ):
        problems.append(f"{prefix}: duration drifted")
    for stream in ("video", "audio"):
        actual_stream = observed.get(stream, {})
        expected_stream = expected.get(stream, {})
        for field, expected_value in expected_stream.items():
            actual_value = actual_stream.get(field)
            if field == "fps":
                if not math.isclose(float(actual_value), float(expected_value), abs_tol=0.001):
                    problems.append(f"{prefix}: {stream}.{field} drifted")
            elif actual_value != expected_value:
                problems.append(f"{prefix}: {stream}.{field} drifted")
    return problems


def _disk_root(path: Path) -> Path:
    root = path
    while not root.exists() and root != root.parent:
        root = root.parent
    return root


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest["schema"],
        "cohort_id": manifest["cohort_id"],
        "frozen_before_media_access": manifest["frozen_before_media_access"],
        "authority": manifest["authority"],
        "split_policy": manifest["split_policy"],
        "privacy_policy": manifest["privacy_policy"],
        "temporal_controls": manifest["temporal_controls"],
        "disk_policy": manifest["disk_policy"],
        "objects": manifest["objects"],
    }


def build_dry_run_plan(
    manifest: Mapping[str, Any],
    *,
    client: WikimediaCommonsAPI,
    disk_root: str | Path = ".",
) -> dict[str, Any]:
    validate_manifest(manifest)
    expected_rows = list(manifest["objects"])
    live = client.pages(row["pageid"] for row in expected_rows)
    problems: list[str] = []
    observed_rows: list[dict[str, Any]] = []
    for expected in expected_rows:
        observed = live[int(expected["pageid"])]
        problems.extend(_compare_object(expected, observed))
        observed_rows.append(
            {
                "referent_id": expected["referent_id"],
                "role": expected["role"],
                "event_class": expected["event_class"],
                "pageid": observed["pageid"],
                "title": observed["title"],
                "object_url": observed["object_url"],
                "size_bytes": observed["size_bytes"],
                "duration_seconds": observed["duration_seconds"],
                "sha1": observed["sha1"],
                "upload_timestamp": observed["upload_timestamp"],
                "license_short_name": observed["license_short_name"],
                "license_url": observed["license_url"],
                "video": observed["video"],
                "audio": observed["audio"],
                "authority_match": not _compare_object(expected, observed),
            }
        )

    disk_policy = manifest["disk_policy"]
    free_bytes = shutil.disk_usage(_disk_root(Path(disk_root))).free
    floor_bytes = int(disk_policy["min_free_disk_bytes"])
    multiplier = int(disk_policy["atomic_reserve_multiplier"])
    train_validation_bytes = int(disk_policy["projected_train_validation_bytes"])
    total_bytes = int(disk_policy["projected_total_bytes"])
    train_validation_reserve = multiplier * train_validation_bytes
    full_reserve = multiplier * total_bytes
    if free_bytes < floor_bytes + train_validation_reserve:
        problems.append(
            "disk guard: train/validation acquisition would cross the 40 GB floor with atomic reserve"
        )

    manifest_identity = _manifest_identity(manifest)
    manifest_sha256 = _sha256_json(manifest_identity)
    plan = {
        "schema": PLAN_SCHEMA,
        "created_at": _utc_now(),
        "cohort_id": manifest["cohort_id"],
        "mode": "metadata-only-dry-run",
        "all_ok": not problems,
        "download_active": False,
        "media_requests": 0,
        "media_bytes_requested": 0,
        "test_media_accessed": False,
        "manifest_identity_sha256": manifest_sha256,
        "manifest_identity": manifest_identity,
        "authority": {
            "api": API_ENDPOINT,
            "checked_live": True,
            "objects_expected": len(expected_rows),
            "objects_matching": sum(bool(row["authority_match"]) for row in observed_rows),
            "file_revision_fields": [
                "pageid",
                "object_url",
                "size_bytes",
                "sha1",
                "upload_timestamp",
            ],
            "rights_fields": ["license_short_name", "license_code", "license_url"],
            "stream_fields": ["audio", "video", "duration_seconds"],
        },
        "objects": observed_rows,
        "split": {
            **manifest["split_policy"],
            "ordered_referents_by_role": {
                role: [row["referent_id"] for row in expected_rows if row["role"] == role]
                for role in ALLOWED_ROLES
            },
        },
        "projected": {
            "objects": len(expected_rows),
            "source_bytes_total": total_bytes,
            "source_mib_total": total_bytes / (1024**2),
            "train_validation_bytes": train_validation_bytes,
            "train_validation_mib": train_validation_bytes / (1024**2),
            "locked_test_bytes": int(disk_policy["projected_locked_test_bytes"]),
            "locked_test_mib": int(disk_policy["projected_locked_test_bytes"]) / (1024**2),
            "duration_seconds_total": sum(float(row["duration_seconds"]) for row in expected_rows),
            "atomic_train_validation_reserve_bytes": train_validation_reserve,
            "atomic_full_cohort_reserve_bytes": full_reserve,
        },
        "safety": {
            "free_disk_bytes": free_bytes,
            "min_free_disk_bytes": floor_bytes,
            "free_after_train_validation_atomic_reserve_bytes": free_bytes - train_validation_reserve,
            "train_validation_download_allowed_by_disk": free_bytes >= floor_bytes + train_validation_reserve,
            "test_download_implemented": False,
            "active_cm7_must_be_absent": True,
        },
        "temporal_controls": manifest["temporal_controls"],
        "privacy": manifest["privacy_policy"],
        "candidate_comparison": list(CANDIDATE_COMPARISON),
        "official_sources": OFFICIAL_SOURCES,
        "post_cm7": {
            "command": POST_CM7_COMMAND,
            "downloads_roles": list(DOWNLOAD_ROLES),
            "expected_source_bytes": train_validation_bytes,
            "expected_atomic_reserve_bytes": train_validation_reserve,
            "test_bytes_remain_locked": int(disk_policy["projected_locked_test_bytes"]),
        },
        "problems": problems,
        "blockers_after_dry_run": [
            "source media bytes are not downloaded",
            "local ffprobe mux verification is not executed",
            "manual privacy, personality-rights and incidental-voice review is not complete",
            "semantic audio-video synchrony has not been manually reviewed from source bytes",
            "audio and video encoder caches over these exact referents do not exist",
            "test media stays locked until an experiment protocol binds this exact manifest hash",
        ],
        "claim_boundary": {
            "status": "mechanics-intake-plan-only" if not problems else "blocked-fail-closed",
            "scientific_promotion": False,
            "natural_audio_video_alignment_claim": False,
            "al3_ready": False,
            "dr15_ready": False,
            "rights_claim": "per-object CC0 metadata verified; non-copyright rights still require review",
            "after_download_can_support": [
                "same-container audio-video timing mechanics",
                "within-clip circular-offset controls",
                "cross-object audio derangements within frozen split",
                "exact session/referent bookkeeping",
            ],
        },
    }
    plan["plan_sha256"] = _sha256_json(_plan_hash_payload(plan))
    return plan


def _plan_hash_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"created_at", "plan_sha256", "manifest_file", "manifest_file_sha256"}
    return {key: value for key, value in plan.items() if key not in excluded}


def validate_dry_run_plan(plan: Mapping[str, Any]) -> None:
    problems: list[str] = []
    if plan.get("schema") != PLAN_SCHEMA or plan.get("mode") != "metadata-only-dry-run":
        problems.append("wrong dry-run plan schema or mode")
    manifest = plan.get("manifest_identity")
    if not isinstance(manifest, Mapping):
        problems.append("manifest identity is missing")
    else:
        try:
            validate_manifest(manifest)
        except WikimediaAVIntakeError as exc:
            problems.append(str(exc))
        if _sha256_json(_manifest_identity(manifest)) != plan.get("manifest_identity_sha256"):
            problems.append("manifest identity SHA-256 mismatch")
    if plan.get("download_active") is not False:
        problems.append("dry-run claims an active download")
    if plan.get("media_requests") != 0 or plan.get("media_bytes_requested") != 0:
        problems.append("dry-run reports media access")
    if plan.get("test_media_accessed") is not False:
        problems.append("dry-run reports test-media access")
    claim = plan.get("claim_boundary", {})
    if claim.get("scientific_promotion") is not False:
        problems.append("dry-run improperly promotes a scientific claim")
    objects = plan.get("objects", [])
    authority = plan.get("authority", {})
    if len(objects) != int(authority.get("objects_expected", -1)):
        problems.append("observed object count does not match authority summary")
    matching = sum(bool(row.get("authority_match")) for row in objects)
    if matching != int(authority.get("objects_matching", -1)):
        problems.append("authority-match count is inconsistent")
    projected = plan.get("projected", {})
    if isinstance(manifest, Mapping):
        disk = manifest.get("disk_policy", {})
        if int(projected.get("source_bytes_total", -1)) != int(disk.get("projected_total_bytes", -2)):
            problems.append("projected total bytes differ from manifest")
        if int(projected.get("train_validation_bytes", -1)) != int(
            disk.get("projected_train_validation_bytes", -2)
        ):
            problems.append("projected train/validation bytes differ from manifest")
    expected_hash = _sha256_json(_plan_hash_payload(plan))
    if plan.get("plan_sha256") != expected_hash:
        problems.append("plan SHA-256 mismatch")
    if bool(plan.get("all_ok")) != (not plan.get("problems")):
        problems.append("all_ok is inconsistent with problems")
    if problems:
        raise WikimediaAVIntakeError("invalid dry-run proof: " + "; ".join(problems))


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def _cm7_process_active() -> bool:
    try:
        import psutil

        for process in psutil.process_iter(["cmdline"]):
            command = " ".join(process.info.get("cmdline") or [])
            if "custom_substrate_workbench.py" in command and " cm7 " in f" {command} ":
                return True
    except Exception:  # noqa: BLE001
        return True
    return False


def _safe_filename(row: Mapping[str, Any]) -> str:
    return f"{row['referent_id']}.webm"


def _download_atomic(
    row: Mapping[str, Any],
    destination: Path,
    *,
    timeout: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    resume_from = temporary.stat().st_size if temporary.exists() else 0
    expected_size = int(row["size_bytes"])
    if resume_from > expected_size:
        raise WikimediaAVIntakeError(f"oversize partial file for {row['referent_id']}")
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "mop-wikimedia-av-intake/1.0 (explicit post-CM7 acquisition)",
    }
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(str(row["object_url"]), headers=headers)
    try:
        with opener(request, timeout=timeout) as response:
            status_value = getattr(response, "status", None)
            status = int(status_value if status_value is not None else response.getcode())
            if resume_from and status != 206:
                raise WikimediaAVIntakeError(
                    f"server did not honor resume range for {row['referent_id']}: HTTP {status}"
                )
            if not resume_from and status != 200:
                raise WikimediaAVIntakeError(
                    f"unexpected download status for {row['referent_id']}: HTTP {status}"
                )
            with temporary.open("ab" if resume_from else "wb") as handle:
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                    handle.write(chunk)
    except (OSError, urllib.error.HTTPError) as exc:
        raise WikimediaAVIntakeError(f"download failed for {row['referent_id']}: {exc}") from exc
    actual_size = temporary.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    with temporary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    actual_sha1 = digest.hexdigest()
    if actual_size != expected_size or actual_sha1 != row["sha1"]:
        raise WikimediaAVIntakeError(
            f"download authority mismatch for {row['referent_id']}: "
            f"size={actual_size}/{expected_size}, sha1={actual_sha1}/{row['sha1']}"
        )
    temporary.replace(destination)
    return {
        "path": str(destination),
        "size_bytes": actual_size,
        "sha1": actual_sha1,
        "sha256": _sha256_file(destination),
        "resumed_from_bytes": resume_from,
    }


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,width,height,sample_rate,channels,r_frame_rate,"
            "start_time,duration,time_base:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        return json.loads(completed.stdout)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise WikimediaAVIntakeError(f"ffprobe failed for {path}: {exc}") from exc


def _validate_ffprobe(row: Mapping[str, Any], probe: Mapping[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams", [])
    video = [value for value in streams if value.get("codec_type") == "video"]
    audio = [value for value in streams if value.get("codec_type") == "audio"]
    problems: list[str] = []
    if len(video) != 1 or len(audio) != 1:
        problems.append(f"expected one video and one audio stream, got {len(video)} and {len(audio)}")
    if video:
        expected_codec = str(row["video"]["codec"]).lower().removeprefix("v_")
        if str(video[0].get("codec_name", "")).lower() != expected_codec:
            problems.append("video codec differs from live authority")
        if int(video[0].get("width", -1)) != int(row["video"]["width"]):
            problems.append("video width differs from live authority")
        if int(video[0].get("height", -1)) != int(row["video"]["height"]):
            problems.append("video height differs from live authority")
    if audio:
        expected_codec = str(row["audio"]["codec"]).lower().removeprefix("a_")
        if str(audio[0].get("codec_name", "")).lower() != expected_codec:
            problems.append("audio codec differs from live authority")
        if int(audio[0].get("sample_rate", -1)) != int(row["audio"]["sample_rate"]):
            problems.append("audio sample rate differs from live authority")
        if int(audio[0].get("channels", -1)) != int(row["audio"]["channels"]):
            problems.append("audio channels differ from live authority")
    video_start = float(video[0].get("start_time", 0.0)) if video else math.nan
    audio_start = float(audio[0].get("start_time", 0.0)) if audio else math.nan
    if video and audio and abs(video_start - audio_start) > 0.1:
        problems.append("audio and video stream starts differ by more than 100 ms")
    duration = float(probe.get("format", {}).get("duration", -1))
    if not math.isclose(duration, float(row["duration_seconds"]), abs_tol=0.1):
        problems.append("container duration differs from live authority")
    if problems:
        raise WikimediaAVIntakeError(f"mux verification failed for {row['referent_id']}: {problems}")
    return {
        "video_stream_count": len(video),
        "audio_stream_count": len(audio),
        "duration_seconds": duration,
        "video_codec": video[0]["codec_name"],
        "audio_codec": audio[0]["codec_name"],
        "video_start_seconds": video_start,
        "audio_start_seconds": audio_start,
        "stream_start_delta_seconds": abs(video_start - audio_start),
        "same_original_mux_container": True,
    }


def execute_train_validation(
    plan: Mapping[str, Any],
    *,
    destination: str | Path,
    timeout: float = 120.0,
) -> dict[str, Any]:
    validate_dry_run_plan(plan)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("all_ok") is not True:
        raise WikimediaAVIntakeError("execution requires a fresh, passing dry-run plan")
    if _cm7_process_active():
        raise WikimediaAVIntakeError("CM7 is active; audio-video acquisition remains blocked")
    manifest = plan["manifest_identity"]
    validate_manifest(manifest)
    root = Path(destination)
    expected_rows = [row for row in manifest["objects"] if row["role"] in DOWNLOAD_ROLES]
    remaining = sum(int(row["size_bytes"]) for row in expected_rows)
    floor = int(manifest["disk_policy"]["min_free_disk_bytes"])
    multiplier = int(manifest["disk_policy"]["atomic_reserve_multiplier"])
    records: list[dict[str, Any]] = []
    for row in expected_rows:
        free = shutil.disk_usage(_disk_root(root)).free
        if free < floor + multiplier * remaining:
            raise WikimediaAVIntakeError(
                f"disk guard failed before {row['referent_id']}: free={free}, remaining={remaining}"
            )
        path = root / str(row["role"]) / _safe_filename(row)
        transfer = _download_atomic(row, path, timeout=timeout)
        mux = _validate_ffprobe(row, _ffprobe(path))
        records.append({"referent_id": row["referent_id"], "role": row["role"], **transfer, "mux": mux})
        remaining -= int(row["size_bytes"])
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "mode": "executed-train-validation-only",
        "all_ok": True,
        "download_active": False,
        "test_media_accessed": False,
        "manifest_identity_sha256": plan["manifest_identity_sha256"],
        "records": records,
        "source_bytes_downloaded": sum(int(row["size_bytes"]) for row in expected_rows),
        "locked_test_bytes": int(manifest["disk_policy"]["projected_locked_test_bytes"]),
        "privacy_review_required": True,
        "privacy_review_receipt": manifest["privacy_policy"]["review_receipt"],
        "claim_boundary": {
            "status": "bytes-and-mux-verified-privacy-pending",
            "scientific_promotion": False,
            "al3_ready": False,
            "dr15_ready": False,
            "test_media_locked": True,
        },
    }
