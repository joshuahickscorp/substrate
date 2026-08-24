"""Focused contracts for non-executing, cache-bound source observation plans."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from substrate.product import ProductRefused
from substrate.product.codec import sha256
from substrate.product.source_adapters import (
    BROWSER_OPERATIONS,
    DENIED_CAPABILITIES,
    MEDIA_OPERATIONS,
    REPOSITORY_SNAPSHOT_VIEWS,
    SOURCE_ADAPTER_PLAN_SCHEMA_VERSION,
    BrowserObservationRequest,
    CacheArtifactRef,
    LocalFileObservationRequest,
    MediaObservationRequest,
    RepositorySnapshotObservationRequest,
    parse_observation_plan,
    plan_browser_observation,
    plan_local_file_observation,
    plan_media_observation,
    plan_observation,
    plan_repository_snapshot_observation,
)
from substrate.product.tool_bundles import AdapterBinding


def _ref(digest: str = "a" * 64) -> CacheArtifactRef:
    return CacheArtifactRef(cache_id="cache-local", artifact_sha256=digest)


def _assert_quarantine_only(plan: dict[str, object]) -> None:
    assert plan["schema_version"] == SOURCE_ADAPTER_PLAN_SCHEMA_VERSION
    assert plan["execution_permitted"] is False
    assert plan["execution_refusal"] == "source adapter execution is not configured"
    assert plan["input_requirements"] == {
        "cache_zones": ["verified", "processed"],
        "descriptor_verification_status": "verified",
        "immutable_digest_match": True,
        "live_source_access": "forbidden",
    }
    assert plan["output_requirements"] == {
        "artifact_kind": "derived",
        "cache_zone": "quarantine",
        "execution_permitted": False,
        "promotion_requires": "separate-cache-attestation-and-revalidation",
    }
    assert plan["denied_capabilities"] == list(DENIED_CAPABILITIES)
    assert {"network-egress", "process-execution", "downloader-invocation", "browser-profile-access", "browser-cookies"} <= set(
        plan["denied_capabilities"]  # type: ignore[arg-type]
    )
    unhashed = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert plan["plan_sha256"] == sha256(unhashed)


def test_local_file_observation_is_digest_bound_and_quarantine_only() -> None:
    request = LocalFileObservationRequest(
        artifact=_ref(),
        views=("metadata", "text"),
        expected_media_types=("text/plain", "application/pdf"),
    )

    plan = plan_local_file_observation(request)

    _assert_quarantine_only(plan)
    assert plan["request"] == {
        "artifact": {"artifact_sha256": "a" * 64, "cache_id": "cache-local"},
        "expected_media_types": ["text/plain", "application/pdf"],
        "type": "local-file-observation",
        "views": ["metadata", "text"],
    }
    assert plan["output_descriptors"] == [
        {"media_type": "application/x-substrate-local-file-observation+json", "role": "metadata"},
        {"media_type": "application/x-substrate-local-file-observation+json", "role": "text"},
    ]
    assert "file:///" not in json.dumps(plan)


def test_repository_snapshot_observation_has_fixed_views_and_bounded_content() -> None:
    request = RepositorySnapshotObservationRequest(
        repository_id="go-raft-reference",
        snapshot=_ref("b" * 64),
        views=("tree", "commit-metadata", "text-content"),
        maximum_files=512,
        maximum_text_bytes=4 * 1024 * 1024,
    )

    plan = plan_repository_snapshot_observation(request)

    _assert_quarantine_only(plan)
    assert plan["request"] == {
        "maximum_files": 512,
        "maximum_text_bytes": 4 * 1024 * 1024,
        "repository_id": "go-raft-reference",
        "snapshot": {"artifact_sha256": "b" * 64, "cache_id": "cache-local"},
        "type": "repository-snapshot-observation",
        "views": ["tree", "commit-metadata", "text-content"],
    }
    assert [item["role"] for item in plan["output_descriptors"]] == ["tree", "commit-metadata", "text-content"]  # type: ignore[index]


def test_media_observation_covers_the_closed_multimodal_operation_set() -> None:
    request = MediaObservationRequest(
        artifact=_ref("c" * 64),
        operations=("probe", "extract-audio", "sample-frames", "subtitles", "metadata"),
        expected_media_type="video/mp4",
        frame_sample_count=12,
        maximum_audio_seconds=300,
    )

    plan = plan_media_observation(request)

    _assert_quarantine_only(plan)
    assert set(request.operations) == MEDIA_OPERATIONS
    assert [item["role"] for item in plan["output_descriptors"]] == [  # type: ignore[index]
        "probe",
        "audio-derivative",
        "frame-sample-set",
        "subtitle-tracks",
        "metadata",
    ]
    assert plan["request"]["frame_sample_count"] == 12  # type: ignore[index]
    assert plan["request"]["maximum_audio_seconds"] == 300  # type: ignore[index]


def test_browser_observation_only_consumes_a_prior_cache_capture() -> None:
    request = BrowserObservationRequest(
        capture=_ref("d" * 64),
        operations=("dom", "accessibility", "screenshot", "frame-capture", "audio-capture"),
        frame_sample_count=8,
        maximum_audio_seconds=120,
    )

    plan = plan_browser_observation(request)

    _assert_quarantine_only(plan)
    assert set(request.operations) == BROWSER_OPERATIONS
    assert plan["request"] == {
        "capture": {"artifact_sha256": "d" * 64, "cache_id": "cache-local"},
        "expected_capture_media_type": "application/x-substrate-browser-capture+json",
        "frame_sample_count": 8,
        "maximum_audio_seconds": 120,
        "operations": ["dom", "accessibility", "screenshot", "frame-capture", "audio-capture"],
        "type": "browser-observation",
    }
    assert [item["role"] for item in plan["output_descriptors"]] == [  # type: ignore[index]
        "dom-snapshot",
        "accessibility-tree",
        "screenshot",
        "frame-sample-set",
        "audio-capture",
    ]


def test_source_adapter_contract_refuses_urls_commands_profiles_and_unbounded_operations() -> None:
    with pytest.raises(ProductRefused, match="cache id must be a lowercase identifier"):
        CacheArtifactRef(cache_id="https://user:password@example.test", artifact_sha256="a" * 64)
    with pytest.raises(ProductRefused, match="repository id must be a lowercase identifier"):
        RepositorySnapshotObservationRequest(repository_id="https://example.test/repo", snapshot=_ref())
    with pytest.raises(ProductRefused, match="media operations includes an unsupported value"):
        MediaObservationRequest(
            artifact=_ref(),
            operations=("probe", "--arbitrary-flag"),
            expected_media_type="video/mp4",
        )
    with pytest.raises(ProductRefused, match="browser frame sample count must be set only for frame-capture"):
        BrowserObservationRequest(capture=_ref(), operations=("dom",), frame_sample_count=1)
    with pytest.raises(ProductRefused, match="media maximum audio seconds must be set only for extract-audio"):
        MediaObservationRequest(
            artifact=_ref(),
            operations=("metadata",),
            expected_media_type="audio/mpeg",
            maximum_audio_seconds=1,
        )
    with pytest.raises(ProductRefused, match="source adapter request is malformed"):
        plan_observation(object())  # type: ignore[arg-type]


def test_source_adapter_contract_refuses_malformed_digests_and_resource_bounds() -> None:
    with pytest.raises(ProductRefused, match="cache artifact sha256"):
        CacheArtifactRef(cache_id="cache-local", artifact_sha256="A" * 64)
    with pytest.raises(ProductRefused, match="repository maximum files must be between"):
        RepositorySnapshotObservationRequest(repository_id="repo", snapshot=_ref(), maximum_files=10_001)
    with pytest.raises(ProductRefused, match="media frame sample count must be set only for sample-frames"):
        MediaObservationRequest(
            artifact=_ref(),
            operations=("probe",),
            expected_media_type="video/mp4",
            frame_sample_count=1,
        )
    with pytest.raises(ProductRefused, match="browser operations includes an unsupported value"):
        BrowserObservationRequest(capture=_ref(), operations=("cookie-export",))


def test_observation_vocabularies_match_the_closed_tool_bundle_roles() -> None:
    media = AdapterBinding(
        adapter_role="media-observation-v1",
        tool_id="ffmpeg",
        operations=tuple(sorted(MEDIA_OPERATIONS)),
    )
    browser = AdapterBinding(
        adapter_role="browser-observation-v1",
        tool_id="chromium",
        operations=tuple(sorted(BROWSER_OPERATIONS)),
    )
    repository = AdapterBinding(
        adapter_role="repository-inspection-v1",
        tool_id="git",
        operations=tuple(sorted(REPOSITORY_SNAPSHOT_VIEWS)),
    )

    assert set(media.operations) == MEDIA_OPERATIONS
    assert set(browser.operations) == BROWSER_OPERATIONS
    assert set(repository.operations) == REPOSITORY_SNAPSHOT_VIEWS


def test_serialized_observation_plans_reparse_to_the_exact_closed_contract() -> None:
    requests = (
        LocalFileObservationRequest(artifact=_ref(), views=("metadata",), expected_media_types=("text/plain",)),
        RepositorySnapshotObservationRequest(repository_id="repo", snapshot=_ref("b" * 64), views=("tree",)),
        MediaObservationRequest(
            artifact=_ref("c" * 64),
            operations=("probe", "sample-frames"),
            expected_media_type="video/mp4",
            frame_sample_count=1,
        ),
        BrowserObservationRequest(capture=_ref("d" * 64), operations=("dom",)),
    )

    for request in requests:
        plan = plan_observation(request)
        serialized = json.loads(json.dumps(plan))
        assert parse_observation_plan(serialized) == plan


def test_observation_plan_parser_refuses_tampering_extensions_and_execution_fields() -> None:
    plan = plan_media_observation(
        MediaObservationRequest(
            artifact=_ref(),
            operations=("probe",),
            expected_media_type="video/mp4",
        )
    )

    bad_digest = deepcopy(plan)
    bad_digest["plan_sha256"] = "f" * 64
    with pytest.raises(ProductRefused, match="plan digest"):
        parse_observation_plan(bad_digest)

    command_extension = deepcopy(plan)
    command_extension["command"] = ["ffmpeg", "-i", "input.mp4"]
    with pytest.raises(ProductRefused, match="plan fields are malformed"):
        parse_observation_plan(command_extension)

    execution_enabled = deepcopy(plan)
    execution_enabled["execution_permitted"] = True
    execution_enabled["plan_sha256"] = sha256({key: value for key, value in execution_enabled.items() if key != "plan_sha256"})
    with pytest.raises(ProductRefused, match="typed contract"):
        parse_observation_plan(execution_enabled)

    altered_output = deepcopy(plan)
    altered_output["output_requirements"]["cache_zone"] = "verified"
    altered_output["plan_sha256"] = sha256({key: value for key, value in altered_output.items() if key != "plan_sha256"})
    with pytest.raises(ProductRefused, match="typed contract"):
        parse_observation_plan(altered_output)

    malformed_ref = deepcopy(plan)
    malformed_ref["request"]["artifact"]["cache_id"] = "https://credential@example.test"
    malformed_ref["plan_sha256"] = sha256({key: value for key, value in malformed_ref.items() if key != "plan_sha256"})
    with pytest.raises(ProductRefused, match="cache id must be a lowercase identifier"):
        parse_observation_plan(malformed_ref)
