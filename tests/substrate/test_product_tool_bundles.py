"""Adversarial checks for non-executing, digest-bound tool bundle manifests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from substrate.product import ProductRefused
from substrate.product.cli import main as product_main
from substrate.product.tool_bundles import (
    NETWORK_MODE_NONE,
    TOOL_BUNDLE_MANIFEST_SCHEMA_VERSION,
    AdapterBinding,
    ToolArtifact,
    ToolBundleManifest,
    parse_tool_bundle_manifest,
)


def _digest(character: str) -> str:
    return character * 64


def _artifact(
    tool_id: str,
    *,
    kind: str = "oci-image",
    version: str = "1.2.3",
    platform: str = "linux-amd64",
) -> ToolArtifact:
    digest_character = {
        "blender": "a",
        "chromium": "b",
        "clang": "c",
        "ffmpeg": "d",
        "git": "e",
        "go": "f",
        "lean": "a",
        "rustc": "b",
        "yt-dlp": "c",
        "z3": "d",
    }.get(tool_id, "a")
    return ToolArtifact(
        tool_id=tool_id,
        release_version=version,
        distribution_kind=kind,
        target_platform=platform,
        artifact_sha256=_digest(digest_character),
        artifact_size_bytes=1024,
        sbom_sha256=_digest("b"),
        notices_sha256=_digest("c"),
        license_document_sha256=_digest("d"),
        license_spdx="GPL-3.0-or-later",
        verification_receipt_sha256=_digest("e"),
    )


def _valid_manifest() -> ToolBundleManifest:
    chromium = _artifact("chromium", kind="oci-image")
    ffmpeg = _artifact("ffmpeg", kind="binary-archive")
    ytdlp = _artifact("yt-dlp", kind="binary-archive")
    return ToolBundleManifest(
        bundle_id="sensory-tools",
        version="1.0.0",
        target_platform="linux-amd64",
        tools=(chromium, ffmpeg, ytdlp),
        adapter_bindings=(
            AdapterBinding(
                adapter_role="approved-media-staging-v1",
                tool_id="yt-dlp",
                operations=("stage-approved-media",),
            ),
            AdapterBinding(
                adapter_role="browser-observation-v1",
                tool_id="chromium",
                operations=("accessibility", "dom", "screenshot"),
            ),
            AdapterBinding(
                adapter_role="media-observation-v1",
                tool_id="ffmpeg",
                operations=("extract-audio", "probe", "sample-frames"),
            ),
        ),
        capabilities=("browser-observation", "media-decode", "source-staging"),
    )


def test_manifest_is_digest_bound_and_cannot_authorize_execution() -> None:
    manifest = _valid_manifest()

    document = manifest.to_document()
    parsed = parse_tool_bundle_manifest(document)

    assert document["schema_version"] == TOOL_BUNDLE_MANIFEST_SCHEMA_VERSION
    assert document["execution_permitted"] is False
    assert document["network_mode"] == NETWORK_MODE_NONE
    assert document["manifest_sha256"] == manifest.manifest_sha256
    assert parsed == manifest
    assert [tool["distribution_kind"] for tool in document["tools"]] == ["oci-image", "binary-archive", "binary-archive"]  # type: ignore[index]
    assert all(
        {"artifact_sha256", "sbom_sha256", "notices_sha256", "license_document_sha256", "verification_receipt_sha256"}
        <= set(tool)
        for tool in document["tools"]  # type: ignore[union-attr]
    )
    assert "command" not in str(document)
    assert "--" not in str(document)


def test_manifest_refuses_tampered_content_even_when_digest_shape_is_valid() -> None:
    document = deepcopy(_valid_manifest().to_document())
    document["tools"][1]["artifact_sha256"] = _digest("f")  # type: ignore[index]

    with pytest.raises(ProductRefused, match="digest does not match"):
        parse_tool_bundle_manifest(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command", ["ffmpeg", "-i", "input.mp4"]),
        ("flags", ["--cookies-from-browser", "chrome"]),
        ("binary_path", "/usr/local/bin/ffmpeg"),
        ("image_tag", "latest"),
        ("working_directory", "../../host"),
        ("network_allowlist", ["youtube.com"]),
    ],
)
def test_manifest_refuses_unknown_command_path_flag_and_network_fields(field: str, value: object) -> None:
    document = deepcopy(_valid_manifest().to_document())
    document[field] = value

    with pytest.raises(ProductRefused, match="invalid field set"):
        parse_tool_bundle_manifest(document)


def test_manifest_refuses_network_and_execution_discrepancies() -> None:
    networked = deepcopy(_valid_manifest().to_document())
    networked["network_mode"] = "egress"

    with pytest.raises(ProductRefused, match="network mode must be none"):
        parse_tool_bundle_manifest(networked)

    executable = deepcopy(_valid_manifest().to_document())
    executable["execution_permitted"] = True

    with pytest.raises(ProductRefused, match="execution_permitted must be false"):
        parse_tool_bundle_manifest(executable)


def test_tool_artifact_requires_all_digest_bound_legal_and_verification_material() -> None:
    assert _artifact("chromium", version="126.0.6478.61").release_version == "126.0.6478.61"
    document = _artifact("ffmpeg").to_dict()
    document["sbom_sha256"] = "sha256:deadbeef"

    with pytest.raises(ProductRefused, match="tool SBOM sha256"):
        ToolArtifact.from_dict(document)

    missing_notice = _artifact("ffmpeg").to_dict()
    del missing_notice["notices_sha256"]

    with pytest.raises(ProductRefused, match="invalid field set"):
        ToolArtifact.from_dict(missing_notice)

    with pytest.raises(ProductRefused, match="tool artifact size bytes"):
        ToolArtifact(
            tool_id="ffmpeg",
            release_version="1.2.3",
            distribution_kind="binary-archive",
            target_platform="linux-amd64",
            artifact_sha256=_digest("a"),
            artifact_size_bytes=True,
            sbom_sha256=_digest("b"),
            notices_sha256=_digest("c"),
            license_document_sha256=_digest("d"),
            license_spdx="GPL-3.0-or-later",
            verification_receipt_sha256=_digest("e"),
        )


def test_adapter_bindings_are_closed_and_capabilities_must_match_exactly() -> None:
    with pytest.raises(ProductRefused, match="adapter operations includes an unsupported value"):
        AdapterBinding(
            adapter_role="media-observation-v1",
            tool_id="ffmpeg",
            operations=("--arbitrary-flag",),
        )
    with pytest.raises(ProductRefused, match="adapter tool is incompatible"):
        AdapterBinding(
            adapter_role="browser-observation-v1",
            tool_id="ffmpeg",
            operations=("dom",),
        )
    with pytest.raises(ProductRefused, match="tool id is not an approved"):
        _artifact("shell")

    manifest = _valid_manifest()
    with pytest.raises(ProductRefused, match="capabilities must exactly match"):
        ToolBundleManifest(
            bundle_id=manifest.bundle_id,
            version=manifest.version,
            target_platform=manifest.target_platform,
            tools=manifest.tools,
            adapter_bindings=manifest.adapter_bindings,
            capabilities=("browser-observation", "formal-verification", "media-decode", "source-staging"),
        )


def test_tool_bundle_cli_inspects_without_resolving_or_launching(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = tmp_path / "sensory-tools.json"
    manifest_path.write_text(json.dumps(_valid_manifest().to_document()), encoding="utf-8")

    product_main(["tool-bundle", "inspect", str(manifest_path)])

    result = json.loads(capsys.readouterr().out)
    assert result["execution_permitted"] is False
    assert result["valid"] is True
    assert result["manifest"]["bundle_id"] == "sensory-tools"
