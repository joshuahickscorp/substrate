"""Command surface for the non-executing post-Odyssey product foundation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from substrate.product.apprenticeship import HostResources, plan_apprenticeship, plan_source_acquisition
from substrate.product.backends import plan_backend_dry_run_for_entity, probe_backends
from substrate.product.cache import ArtifactStore, CacheAttestation, LocalCacheVerifierTrustStore, sign_cache_attestation
from substrate.product.codec import read_json, sha256
from substrate.product.contracts import ApprenticeshipSpec, EntityManifest, OrganRequirement, ProductRefused, ResourceBudget, SourcePolicy, SourceRequest
from substrate.product.entity import EntityStore
from substrate.product.pack_artifacts import (
    LocalPackRegistry,
    LocalTrustStore,
    build_pack_artifact,
    generate_ed25519_keypair,
    inspect_pack_artifact,
    sign_pack_artifact,
    verify_pack_artifact,
)
from substrate.product.packs import list_packs
from substrate.product.tool_bundles import parse_tool_bundle_manifest


def _print(value: dict[str, Any] | list[dict[str, Any]]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="substrate product", description="Post-Odyssey product foundation; planning and verification only.")
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("packs", help="list declarative capability packs")

    pack = commands.add_parser("pack", help="build, sign, verify, and locally install manifest-only capability packs")
    pack_commands = pack.add_subparsers(dest="pack_command", required=True)
    pack_build = pack_commands.add_parser("build", help="build an unsigned manifest-only pack artifact")
    pack_build.add_argument("pack_name")
    pack_build.add_argument("artifact_directory", type=Path)
    pack_build.add_argument("--publisher", required=True)
    pack_keygen = pack_commands.add_parser("keygen", help="generate an operator-managed Ed25519 pack-signing keypair")
    pack_keygen.add_argument("--private-key", required=True, type=Path)
    pack_keygen.add_argument("--public-key", required=True, type=Path)
    pack_sign = pack_commands.add_parser("sign", help="add a detached Ed25519 signature to an unsigned pack artifact")
    pack_sign.add_argument("artifact_directory", type=Path)
    pack_sign.add_argument("--private-key", required=True, type=Path)
    pack_inspect = pack_commands.add_parser("inspect", help="inspect a manifest-only pack artifact without trusting it")
    pack_inspect.add_argument("artifact_directory", type=Path)
    pack_trust = pack_commands.add_parser("trust", help="record a scoped local publisher trust rule")
    pack_trust.add_argument("--trust-store", required=True, type=Path)
    pack_trust.add_argument("--publisher", required=True)
    pack_trust.add_argument("--public-key", required=True, type=Path)
    pack_trust.add_argument("--pack", action="append", dest="allowed_packs", required=True)
    pack_trust.add_argument("--capability", action="append", dest="allowed_capabilities")
    pack_verify = pack_commands.add_parser("verify", help="verify a signed pack against scoped local trust")
    pack_verify.add_argument("artifact_directory", type=Path)
    pack_verify.add_argument("--trust-store", required=True, type=Path)
    pack_install = pack_commands.add_parser("install", help="install a verified manifest reference; never install tools")
    pack_install.add_argument("artifact_directory", type=Path)
    pack_install.add_argument("--trust-store", required=True, type=Path)
    pack_install.add_argument("--registry", required=True, type=Path)
    pack_remove = pack_commands.add_parser("remove", help="remove only an installed manifest reference")
    pack_remove.add_argument("--registry", required=True, type=Path)
    pack_remove.add_argument("--pack", required=True, dest="pack_name")
    pack_remove.add_argument("--version", required=True)
    pack_remove.add_argument("--manifest-sha256", required=True)

    cache = commands.add_parser("cache", help="manage local quarantine and immutable cache objects without fetching")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_init = cache_commands.add_parser("init", help="create an empty local artifact cache")
    cache_init.add_argument("cache_directory", type=Path)
    cache_init.add_argument("--capacity-bytes", required=True, type=int)
    for name, help_text in (
        ("status", "show cache zone and capacity status"),
        ("explain", "show one artifact descriptor and cache zone"),
        ("quarantine", "move an immutable artifact back to quarantine"),
        ("pin", "protect an immutable artifact from garbage collection"),
        ("gc", "remove explicitly selected unpinned cache objects"),
    ):
        cache_command = cache_commands.add_parser(name, help=help_text)
        cache_command.add_argument("cache_directory", type=Path)
        cache_command.add_argument(
            "--verifier-trust-store",
            type=Path,
            help="required when this command reads, pins, or derives from a verified cache object",
        )
        if name in {"explain", "quarantine", "pin"}:
            cache_command.add_argument("artifact_sha256")
        if name in {"quarantine", "pin"}:
            cache_command.add_argument("--reason", required=True)
        if name == "gc":
            cache_command.add_argument("--include-verified", action="store_true")
            cache_command.add_argument("--maximum-objects", type=int)
    cache_add = cache_commands.add_parser("add", help="stream a local regular file into quarantine; no remote fetch")
    cache_add.add_argument("cache_directory", type=Path)
    cache_add.add_argument("source_path", type=Path)
    cache_add.add_argument("--media-type", required=True)
    cache_add.add_argument("--source-reference-sha256", required=True)
    cache_add.add_argument("--rights-status", required=True)
    cache_add.add_argument("--expected-byte-length", type=int)
    cache_verify = cache_commands.add_parser("verify", help="promote a local-attested quarantined object after revalidation")
    cache_verify.add_argument("cache_directory", type=Path)
    cache_verify.add_argument("artifact_sha256")
    cache_verify.add_argument("--attestation", required=True, type=Path)
    cache_verify.add_argument("--verifier-trust-store", required=True, type=Path)
    cache_trust = cache_commands.add_parser("trust-verifier", help="record a local verifier key and rights scope for cache attestations")
    cache_trust.add_argument("--trust-store", required=True, type=Path)
    cache_trust.add_argument("--verifier-id", required=True)
    cache_trust.add_argument("--public-key", required=True, type=Path)
    cache_trust.add_argument("--rights-status", action="append", dest="allowed_rights_statuses", required=True)
    cache_attest = cache_commands.add_parser("attest", help="sign an attestation for one quarantined descriptor; never promotes it")
    cache_attest.add_argument("cache_directory", type=Path)
    cache_attest.add_argument("artifact_sha256")
    cache_attest.add_argument("--verifier-id", required=True)
    cache_attest.add_argument("--private-key", required=True, type=Path)
    cache_attest.add_argument("--rights-status", required=True)
    cache_attest.add_argument("--expires-at", required=True)
    cache_attest.add_argument("--issued-at")

    tool_bundle = commands.add_parser(
        "tool-bundle",
        help="inspect a digest-bound, non-executing operator tool-bundle manifest",
    )
    tool_bundle_commands = tool_bundle.add_subparsers(dest="tool_bundle_command", required=True)
    tool_bundle_inspect = tool_bundle_commands.add_parser(
        "inspect",
        help="validate a local tool-bundle manifest without resolving or launching a tool",
    )
    tool_bundle_inspect.add_argument("manifest_path", type=Path)

    init = commands.add_parser("init", help="create a portable entity directory")
    init.add_argument("directory", type=Path)
    init.add_argument("--entity-id", required=True)
    init.add_argument("--specialty", required=True)
    init.add_argument("--pack", action="append", dest="packs", required=True)
    init.add_argument("--organ", action="append", dest="organs")

    for name, help_text in (
        ("status", "show a verified entity status"),
        ("validate", "verify manifest, state, checkpoint, and receipt chain"),
        ("recover", "apply a verified pending entity transaction"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("directory", type=Path)

    plan = commands.add_parser("plan-apprenticeship", help="record a bounded worker and assimilation plan")
    plan.add_argument("directory", type=Path)
    plan.add_argument("--name", required=True)
    plan.add_argument("--objective", required=True)
    plan.add_argument("--evaluator", action="append", dest="evaluators", required=True)
    plan.add_argument("--host-cpu-cores", type=int, required=True)
    plan.add_argument("--host-memory-mib", type=int, required=True)
    plan.add_argument("--host-disk-mib", type=int, required=True)
    plan.add_argument("--worker-cpu-cores", type=int, required=True)
    plan.add_argument("--worker-memory-mib", type=int, required=True)
    plan.add_argument("--worker-disk-mib", type=int, required=True)
    plan.add_argument("--maximum-workers", type=int, required=True)
    plan.add_argument("--wall-clock-minutes", type=int, required=True)
    plan.add_argument("--source-scheme", action="append", dest="source_schemes")
    plan.add_argument("--source-domain", action="append", dest="source_domains")
    plan.add_argument("--source-file-root", action="append", dest="source_file_roots")
    plan.add_argument("--allow-download", action="store_true")

    source = commands.add_parser("plan-source", help="bind an approved source request to the active apprenticeship")
    source.add_argument("directory", type=Path)
    source.add_argument("--source-uri", required=True)
    source.add_argument("--modality", required=True)
    source.add_argument("--access-status", required=True)
    source.add_argument("--declared-rights", required=True)
    source.add_argument("--retrieval-mode", default="metadata")

    commands.add_parser("backends", help="probe local sandbox backend eligibility without executing anything")
    dry_run = commands.add_parser(
        "dry-run-backend",
        help="emit a non-executing backend plan bound to the active apprenticeship sandbox",
    )
    dry_run.add_argument("directory", type=Path)
    return root


def _source_policy(arguments: argparse.Namespace) -> SourcePolicy:
    return SourcePolicy(
        allowed_schemes=tuple(arguments.source_schemes or ("file",)),
        allowed_domains=tuple(arguments.source_domains or ()),
        allowed_file_roots=tuple(arguments.source_file_roots or ()),
        allow_download=arguments.allow_download,
    )


def main(argv: list[str] | None = None) -> None:
    root = parser()
    arguments = root.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if arguments.command == "packs":
            _print({"execution_permitted": False, "packs": list_packs()})
            return
        if arguments.command == "pack":
            if arguments.pack_command == "build":
                _print(build_pack_artifact(arguments.pack_name, arguments.artifact_directory, publisher=arguments.publisher))
                return
            if arguments.pack_command == "keygen":
                _print(generate_ed25519_keypair(arguments.private_key, arguments.public_key))
                return
            if arguments.pack_command == "sign":
                _print(sign_pack_artifact(arguments.artifact_directory, arguments.private_key))
                return
            if arguments.pack_command == "inspect":
                _print(inspect_pack_artifact(arguments.artifact_directory))
                return
            if arguments.pack_command == "trust":
                trust = LocalTrustStore(arguments.trust_store).trust(
                    publisher=arguments.publisher,
                    public_key_path=arguments.public_key,
                    allowed_pack_names=tuple(arguments.allowed_packs),
                    allowed_capabilities=tuple(arguments.allowed_capabilities or ()),
                )
                _print({"execution_permitted": False, "trust": trust.to_dict()})
                return
            if arguments.pack_command == "verify":
                _print(verify_pack_artifact(arguments.artifact_directory, LocalTrustStore(arguments.trust_store)))
                return
            if arguments.pack_command == "install":
                _print(
                    LocalPackRegistry(arguments.registry).install(
                        arguments.artifact_directory,
                        LocalTrustStore(arguments.trust_store),
                    )
                )
                return
            if arguments.pack_command == "remove":
                _print(
                    LocalPackRegistry(arguments.registry).remove(
                        pack_name=arguments.pack_name,
                        version=arguments.version,
                        manifest_sha256=arguments.manifest_sha256,
                    )
                )
                return
        if arguments.command == "cache":
            if arguments.cache_command == "init":
                _print(ArtifactStore.create(arguments.cache_directory, capacity_bytes=arguments.capacity_bytes).status())
                return
            if arguments.cache_command == "trust-verifier":
                verifier_trust = LocalCacheVerifierTrustStore(arguments.trust_store).trust(
                    verifier_id=arguments.verifier_id,
                    public_key_path=arguments.public_key,
                    allowed_rights_statuses=tuple(arguments.allowed_rights_statuses),
                )
                _print({"execution_permitted": False, "trust": verifier_trust.to_dict()})
                return
            verifier_trust_store = (
                LocalCacheVerifierTrustStore(arguments.verifier_trust_store)
                if getattr(arguments, "verifier_trust_store", None) is not None
                else None
            )
            cache_store = ArtifactStore.open(arguments.cache_directory, verifier_trust_store=verifier_trust_store)
            if arguments.cache_command == "status":
                _print(cache_store.status())
                return
            if arguments.cache_command == "add":
                _print(
                    {
                        "descriptor": cache_store.ingest_file(
                            arguments.source_path,
                            media_type=arguments.media_type,
                            source_reference_sha256=arguments.source_reference_sha256,
                            rights_status=arguments.rights_status,
                            expected_byte_length=arguments.expected_byte_length,
                        ).to_dict(),
                        "execution_permitted": False,
                    }
                )
                return
            if arguments.cache_command == "verify":
                attestation = CacheAttestation.from_dict(read_json(arguments.attestation))
                if verifier_trust_store is None:  # Defensive: argparse marks this one required.
                    raise ProductRefused("cache verification requires a local verifier trust store")
                _print(
                    {
                        "descriptor": cache_store.verify(
                            arguments.artifact_sha256,
                            attestation,
                            verifier_trust_store,
                        ).to_dict(),
                        "execution_permitted": False,
                    }
                )
                return
            if arguments.cache_command == "attest":
                explanation = cache_store.explain(arguments.artifact_sha256)
                _print(
                    {
                        "attestation": sign_cache_attestation(
                            artifact_sha256=arguments.artifact_sha256,
                            cache_id=cache_store.cache_id,
                            descriptor_sha256=sha256(explanation["descriptor"]),
                            verifier_id=arguments.verifier_id,
                            rights_status=arguments.rights_status,
                            private_key_path=arguments.private_key,
                            issued_at=arguments.issued_at,
                            expires_at=arguments.expires_at,
                        ).to_dict(),
                        "execution_permitted": False,
                    }
                )
                return
            if arguments.cache_command == "explain":
                _print(cache_store.explain(arguments.artifact_sha256))
                return
            if arguments.cache_command == "quarantine":
                _print(
                    {
                        "descriptor": cache_store.quarantine(arguments.artifact_sha256, reason=arguments.reason).to_dict(),
                        "execution_permitted": False,
                    }
                )
                return
            if arguments.cache_command == "pin":
                _print(cache_store.pin(arguments.artifact_sha256, reason=arguments.reason))
                return
            if arguments.cache_command == "gc":
                _print(cache_store.gc(include_verified=arguments.include_verified, maximum_objects=arguments.maximum_objects))
                return
        if arguments.command == "tool-bundle" and arguments.tool_bundle_command == "inspect":
            tool_bundle_manifest = parse_tool_bundle_manifest(read_json(arguments.manifest_path))
            _print({"execution_permitted": False, "manifest": tool_bundle_manifest.to_document(), "valid": True})
            return
        if arguments.command == "backends":
            probe = probe_backends()
            _print(
                {
                    "execution_permitted": False,
                    "preferred_backend": probe.preferred_backend,
                    "probe": probe.to_dict(),
                    "selection_is_authorization": False,
                    "selection_is_dry_run": True,
                    "selection_status": probe.selection_status,
                }
            )
            return
        if arguments.command == "init":
            manifest = EntityManifest(
                entity_id=arguments.entity_id,
                specialty=arguments.specialty,
                selected_packs=tuple(arguments.packs),
                organ_requirements=tuple(OrganRequirement(organ_id=organ) for organ in (arguments.organs or ())),
            )
            _print(EntityStore.create(arguments.directory, manifest).status())
            return
        store = EntityStore(arguments.directory)
        if arguments.command == "status":
            _print(store.status())
            return
        if arguments.command == "validate":
            _print(store.validate())
            return
        if arguments.command == "recover":
            _print(store.recover())
            return
        if arguments.command == "dry-run-backend":
            _print(plan_backend_dry_run_for_entity(store))
            return
        if arguments.command == "plan-apprenticeship":
            specification = ApprenticeshipSpec(
                name=arguments.name,
                objective=arguments.objective,
                evaluators=tuple(arguments.evaluators),
                source_policy=_source_policy(arguments),
                worker_budget=ResourceBudget(
                    cpu_cores=arguments.worker_cpu_cores,
                    memory_mib=arguments.worker_memory_mib,
                    disk_mib=arguments.worker_disk_mib,
                ),
                maximum_workers=arguments.maximum_workers,
                wall_clock_minutes=arguments.wall_clock_minutes,
            )
            host = HostResources(
                cpu_cores=arguments.host_cpu_cores,
                memory_mib=arguments.host_memory_mib,
                disk_mib=arguments.host_disk_mib,
            )
            _print(plan_apprenticeship(store, specification, host))
            return
        if arguments.command == "plan-source":
            request = SourceRequest(
                source_uri=arguments.source_uri,
                modality=arguments.modality,
                access_status=arguments.access_status,
                declared_rights=arguments.declared_rights,
                retrieval_mode=arguments.retrieval_mode,
            )
            _print(plan_source_acquisition(store, request))
            return
    except ProductRefused as exc:
        root.error(str(exc))
    raise AssertionError(f"unhandled product command {arguments.command!r}")


if __name__ == "__main__":
    main()
