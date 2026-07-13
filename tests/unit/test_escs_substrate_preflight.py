from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from scripts.run_escs_substrate_preflight import main as preflight_main

from mop.config import REPO_ROOT
from mop.escs.substrate_preflight import (
    PreflightBinding,
    SubstratePreflightManifest,
    _binding_integrity_problems,
    assess_substrate_preflight,
    create_substrate_preflight_manifest,
)
from mop.substrate.events import canonical_sha256


def _binding(
    role: str,
    path: str,
    schema: str,
    **required_fields: object,
) -> PreflightBinding:
    source = REPO_ROOT / path
    return PreflightBinding(
        role=role,
        path=path,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        schema=schema,
        required_fields=tuple(sorted(required_fields.items())),
        activation_authority=False,
    )


def _manifest() -> SubstratePreflightManifest:
    return create_substrate_preflight_manifest(
        manifest_id="test-quiescent-substrate-v1",
        registry_role="perspective-registry",
        assembly_role="substrate-assembly",
        topology_role="topology-grammar",
        mechanics_role="mechanics-proof",
        bindings=(
            _binding(
                "perspective-registry",
                "configs/experiment/escs_perspective_candidates.json",
                "mop-escs-perspective-candidates/v1",
                default_activation_enabled=False,
                scientific_promotion_allowed=False,
            ),
            _binding(
                "substrate-assembly",
                "configs/experiment/escs_substrate_assembly.json",
                "mop-escs-substrate-assembly/v1",
                default_quiescent=True,
                scientific_promotion_allowed=False,
            ),
            _binding(
                "topology-grammar",
                "configs/experiment/escs_g0_topology_grammar.json",
                "mop-escs-topology-grammar/v1",
                activation_enabled=False,
                scientific_promotion_allowed=False,
                status="scaffold",
            ),
            _binding(
                "mechanics-proof",
                "proof/ESCS_MECHANICS_CHASSIS.json",
                "mop-escs-mechanics-proof/v1",
                all_ok=True,
                claim_scope="scripted-mechanics-only",
                complete=True,
            ),
        ),
    )


def test_preflight_joins_every_perspective_into_one_quiescent_scaffold() -> None:
    report = assess_substrate_preflight(_manifest(), repository_root=REPO_ROOT)

    assert report.scaffold_ready is True
    assert report.problems == ()
    assert report.binding_count == report.exact_binding_count == 4
    assert report.perspective_count == report.installed_slot_count == 31
    assert report.default_quiescent is True
    assert report.topology_status == "scaffold"
    assert report.topology_implementation_complete is False
    assert report.activation_ready is False
    assert report.scientific_promotion_allowed is False


def test_digest_drift_fails_closed_without_turning_on_the_substrate() -> None:
    manifest = _manifest()
    bindings = list(manifest.bindings)
    row = bindings[0]
    bindings[0] = PreflightBinding(
        role=row.role,
        path=row.path,
        sha256="0" * 64,
        schema=row.schema,
        required_fields=row.required_fields,
        activation_authority=False,
    )
    changed = create_substrate_preflight_manifest(
        manifest_id=manifest.manifest_id,
        registry_role=manifest.registry_role,
        assembly_role=manifest.assembly_role,
        topology_role=manifest.topology_role,
        mechanics_role=manifest.mechanics_role,
        bindings=bindings,
    )

    report = assess_substrate_preflight(changed, repository_root=REPO_ROOT)

    assert report.scaffold_ready is False
    assert report.activation_ready is False
    assert any(problem.startswith("binding-digest-mismatch:") for problem in report.problems)


def test_manifest_tampering_or_activation_authority_is_rejected() -> None:
    payload = _manifest().payload()
    payload["activation_enabled"] = True
    with pytest.raises(ValueError, match="activation-disabled"):
        SubstratePreflightManifest.from_payload(payload)

    row = _manifest().bindings[0]
    with pytest.raises(ValueError, match="cannot grant activation"):
        PreflightBinding(
            role=row.role,
            path=row.path,
            sha256=row.sha256,
            schema=row.schema,
            required_fields=row.required_fields,
            activation_authority=True,
        )


def test_binding_paths_are_strictly_repository_relative() -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        PreflightBinding(
            role="escape",
            path=str(Path("..") / "escape.json"),
            sha256="0" * 64,
            schema="mop-test/v1",
            required_fields=(("all_ok", True),),
            activation_authority=False,
        )


@pytest.mark.parametrize("mutation", ["missing", "added"])
def test_mechanics_receipt_requires_exact_source_coverage(mutation: str) -> None:
    payload = json.loads((REPO_ROOT / "proof/ESCS_MECHANICS_CHASSIS.json").read_text())
    changed = copy.deepcopy(payload)
    files = changed["implementation_receipt"]["files"]
    if mutation == "missing":
        files.pop()
    else:
        files.append(copy.deepcopy(files[0]))
    changed["implementation_receipt"]["manifest_sha256"] = canonical_sha256(files)
    core = dict(changed)
    core.pop("proof_sha256")
    changed["proof_sha256"] = canonical_sha256(core)
    binding = PreflightBinding(
        role="mechanics-proof",
        path="proof/ESCS_MECHANICS_CHASSIS.json",
        sha256="0" * 64,
        schema="mop-escs-mechanics-proof/v1",
        required_fields=(("all_ok", True),),
        activation_authority=False,
    )

    problems = _binding_integrity_problems(REPO_ROOT, binding, changed)

    assert "binding-file-receipt-coverage-mismatch:mechanics-proof" in problems


def test_runner_refuses_to_overwrite_any_bound_input() -> None:
    with pytest.raises(ValueError, match="aliases a bound input authority"):
        preflight_main(
            (
                "--manifest",
                str(REPO_ROOT / "configs/experiment/escs_substrate_preflight.json"),
                "--out",
                str(REPO_ROOT / "proof/ESCS_MECHANICS_CHASSIS.json"),
            )
        )
