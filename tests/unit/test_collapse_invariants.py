from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import token
import tokenize
import tomllib
from pathlib import Path

from collapse.tools.build_ledger import _unique_object, decode_checklist, decode_reductions

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOTS = ("src", "tests", "scripts", "collapse/tools", "legacy_scaffolding")
EVIDENCE_PRIMITIVES = {
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_bytes",
    "canonical_sha256",
    "sha256_file",
    "write_canonical_json",
}


def _python_paths() -> list[Path]:
    return sorted(path for root in PYTHON_ROOTS for path in (ROOT / root).rglob("*.py"))


def test_python_is_parseable_readable_source_not_minified_or_line_packed():
    paths = _python_paths()
    assert paths
    for path in paths:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        assert max((len(line.encode()) for line in source.splitlines()), default=0) <= 4096
        semicolons = [
            row
            for row in tokenize.generate_tokens(io.StringIO(source).readline)
            if row.type == token.OP and row.string == ";"
        ]
        assert not semicolons, f"line-packed statements in {path.relative_to(ROOT)}"


def test_one_cli_registry_config_tree_and_production_evidence_authority():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"] == {"mop": "mop.harness.cli:main"}
    assert [
        path.relative_to(ROOT).as_posix() for path in (ROOT / "registry").rglob("*") if path.is_file()
    ] == ["registry/experiments.yaml"]
    assert (ROOT / "configs/config.yaml").is_file()
    assert len([path for path in ROOT.iterdir() if path.name == "configs" and path.is_dir()]) == 1

    definitions: dict[str, list[str]] = {name: [] for name in EVIDENCE_PRIMITIVES}
    for path in sorted((ROOT / "src/mop").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in definitions:
                definitions[node.name].append(path.relative_to(ROOT).as_posix())
    assert definitions == {name: ["src/mop/evidence.py"] for name in EVIDENCE_PRIMITIVES}


def test_proof_index_is_complete_content_addressed_and_deduplicated():
    payload = (ROOT / "collapse/MOP_PROOF_INDEX.json").read_bytes()
    assert payload.endswith(b"\n") and payload.count(b"\n") == 1
    index = json.loads(payload, object_pairs_hook=_unique_object)
    prior = json.loads(
        subprocess.check_output(
            ["git", "show", "mop-collapse-lowest-green-38-audit-corrected:collapse/MOP_PROOF_INDEX.json"],
            cwd=ROOT,
        ),
        object_pairs_hook=_unique_object,
    )
    assert index == prior
    indexed = {row["path"]: row for row in index["entries"]}
    actual = sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "proof").rglob("*") if path.is_file()
    )
    assert sorted(indexed) == actual
    digests: dict[str, list[str]] = {}
    for relative, row in indexed.items():
        payload = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert (row["bytes"], row["sha256"]) == (len(payload), digest)
        digests.setdefault(digest, []).append(relative)
    duplicates = {digest: paths for digest, paths in digests.items() if len(paths) > 1}
    assert index["files"] == len(actual)
    assert index["bytes"] == sum(row["bytes"] for row in indexed.values())
    assert index["duplicate_groups"] == duplicates == {}


def test_unbound_proof_json_compaction_is_semantically_exact():
    source_tag = "mop-collapse-lowest-green-35"
    target_tag = "mop-collapse-compact-unbound-proof-json"
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", source_tag, target_tag, "--", "proof"], cwd=ROOT, text=True
    ).splitlines()
    assert len(changed) == 24
    for relative in changed:
        current = subprocess.check_output(["git", "show", f"{target_tag}:{relative}"], cwd=ROOT)
        prior = subprocess.check_output(["git", "show", f"{source_tag}:{relative}"], cwd=ROOT)
        assert current.endswith(b"\n") and current.count(b"\n") == 1
        assert json.loads(current, object_pairs_hook=_unique_object) == json.loads(
            prior, object_pairs_hook=_unique_object
        )


def test_bound_proof_json_compaction_preserves_semantics_and_merkle_bindings():
    source_tag = "mop-collapse-lowest-green-37"
    paths = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "proof").rglob("*.json"))
    assert len(paths) == 59
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", source_tag, "--", "proof"], cwd=ROOT, text=True
    ).splitlines()
    assert len(changed) == 38
    current_bytes = {relative: (ROOT / relative).read_bytes() for relative in paths}
    prior_bytes = {
        relative: subprocess.check_output(["git", "show", f"{source_tag}:{relative}"], cwd=ROOT)
        for relative in paths
    }
    hash_rebinding = {
        hashlib.sha256(prior_bytes[relative]).hexdigest(): hashlib.sha256(current_bytes[relative]).hexdigest()
        for relative in changed
    }

    def rebind_hashes(value):
        if isinstance(value, dict):
            return {key: rebind_hashes(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebind_hashes(item) for item in value]
        return hash_rebinding.get(value, value)

    historical_snapshot_sources = {
        "proof/PROGRAMMATIC_FORM_CACHE.json",
        "proof/REAL_ENCODER_LOCAL_ATTEMPT.json",
    }
    attestation = "proof/CUSTOM_SUBSTRATE_PILOT_CHAIN/current_evidence_attestation.json"
    for relative in changed:
        assert current_bytes[relative].endswith(b"\n") and current_bytes[relative].count(b"\n") == 1
        prior = json.loads(prior_bytes[relative], object_pairs_hook=_unique_object)
        expected = rebind_hashes(prior)
        if relative == attestation:
            prior_snapshots = {row["source_path"]: row for row in prior["snapshot_checks"]}
            for row in expected["snapshot_checks"]:
                if row["source_path"] in historical_snapshot_sources:
                    row.update(prior_snapshots[row["source_path"]])
        assert json.loads(current_bytes[relative], object_pairs_hook=_unique_object) == expected

    prior_hashes = {
        relative: hashlib.sha256(payload).hexdigest().encode() for relative, payload in prior_bytes.items()
    }
    current_hashes = {
        relative: hashlib.sha256(payload).hexdigest().encode() for relative, payload in current_bytes.items()
    }
    historical_snapshot_edges = {(attestation, source) for source in historical_snapshot_sources}
    edges = 0
    for parent in paths:
        for child in paths:
            if parent != child and prior_hashes[child] in prior_bytes[parent]:
                edges += 1
                expected = (
                    prior_hashes[child]
                    if (parent, child) in historical_snapshot_edges
                    else current_hashes[child]
                )
                assert expected in current_bytes[parent]
    assert edges == 41


def test_bound_run_json_compaction_preserves_semantics_and_refreshes_bundle():
    source_tag = "mop-collapse-lowest-green-36"
    changed_paths = subprocess.check_output(
        ["git", "diff", "--name-only", source_tag, "--", "runs"], cwd=ROOT, text=True
    ).splitlines()
    changed = [relative for relative in changed_paths if relative.endswith(".json")]
    assert len(changed) == 7
    retired_script = "runs/mot/run_stages_1_3.sh"
    assert set(changed_paths) == {*changed, retired_script}
    assert not (ROOT / retired_script).exists()
    script = subprocess.check_output(["git", "show", f"{source_tag}:{retired_script}"], cwd=ROOT)
    assert (len(script), len(script.decode().splitlines()), hashlib.sha256(script).hexdigest()) == (
        5156,
        70,
        "8a8c814e389623b786aac1defe0ad5361eca8a9580fc49cfd6aea2f5454c588b",
    )
    index_path = "proof/ARTIFACT_INDEX/pre_studio.json"
    bundle = json.loads((ROOT / index_path).read_bytes(), object_pairs_hook=_unique_object)
    expected = json.loads(
        subprocess.check_output(["git", "show", f"{source_tag}:{index_path}"], cwd=ROOT),
        object_pairs_hook=_unique_object,
    )
    current_rows = {row["display_path"]: row for row in bundle["artifacts"]}
    expected_rows = {row["display_path"]: row for row in expected["artifacts"]}
    for relative in changed:
        current = (ROOT / relative).read_bytes()
        prior = subprocess.check_output(["git", "show", f"{source_tag}:{relative}"], cwd=ROOT)
        assert current.endswith(b"\n") and current.count(b"\n") == 1
        assert json.loads(current, object_pairs_hook=_unique_object) == json.loads(
            prior, object_pairs_hook=_unique_object
        )
        expected_rows[relative].update(size_bytes=len(current), sha256=hashlib.sha256(current).hexdigest())
        assert current_rows[relative] == expected_rows[relative]
    assert bundle == expected


def test_retired_code_and_documents_have_explicit_git_recovery():
    state = json.loads((ROOT / "MOP_COLLAPSE_STATE.json").read_text(encoding="utf-8"))
    authorities = {row["path"]: row for row in state["legacy_authorities"]["files"]}

    def recover(relative: str) -> bytes:
        row = authorities[relative]
        payload = subprocess.check_output(["git", "show", f"{row['tag']}:{relative}"], cwd=ROOT)
        assert len(payload) == row["bytes"]
        assert len(payload.decode("utf-8").splitlines()) == row["lines"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert (
            subprocess.check_output(["git", "rev-parse", f"{row['tag']}:{relative}"], text=True).strip()
            == row["git_blob"]
        )
        return payload

    code = json.loads(recover("collapse/MOP_HISTORICAL_CODE_INDEX.json"))
    assert code["source_tag"] and code["recovery"]
    assert code["clusters"]
    objects = []
    for cluster in code["clusters"]:
        assert cluster["id"] and cluster["paths"]
        tag = cluster.get("source_tag", code["source_tag"])
        objects.extend(f"{tag}:{path}" for path in cluster["paths"])
    recovered = subprocess.check_output(
        ["git", "cat-file", "--batch-check"], input="\n".join(objects) + "\n", text=True, cwd=ROOT
    )
    assert len(recovered.splitlines()) == len(objects) == 1544
    assert " missing" not in recovered

    document_payload = (ROOT / "collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json").read_bytes()
    assert document_payload.endswith(b"\n") and document_payload.count(b"\n") == 1
    documents = json.loads(document_payload, object_pairs_hook=_unique_object)
    prior_documents = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                "mop-collapse-lowest-green-38-audit-corrected:collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json",
            ],
            cwd=ROOT,
        ),
        object_pairs_hook=_unique_object,
    )
    assert documents == prior_documents
    assert documents["archive_tag"] and documents["recovery"]
    assert documents["removed_document_count"] == len(documents["removed_documents"])
    for row in documents["removed_documents"]:
        payload = subprocess.check_output(
            ["git", "show", f"{documents['archive_tag']}:{row['path']}"], cwd=ROOT
        )
        assert (len(payload), len(payload.decode().splitlines()), hashlib.sha256(payload).hexdigest()) == (
            row["bytes"],
            row["lines"],
            row["sha256"],
        )


def test_normalized_state_projects_exact_checklist_reductions_and_legacy_bytes():
    payload = (ROOT / "MOP_COLLAPSE_STATE.json").read_bytes()
    assert payload.endswith(b"\n") and payload.count(b"\n") == 1
    state = json.loads(payload, object_pairs_hook=_unique_object)
    compact_source = json.loads(
        subprocess.check_output(
            ["git", "show", "mop-collapse-lowest-green-38-audit-corrected:MOP_COLLAPSE_STATE.json"],
            cwd=ROOT,
        ),
        object_pairs_hook=_unique_object,
    )
    for key in ("meta", "current_measured", "reductions", "reduction_accounting_verified", "audit"):
        compact_source[key] = state[key]
    assert state == compact_source
    checklist = decode_checklist(state["checklist"])
    reductions = decode_reductions(state["reductions"])

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    assert len(checklist) == state["checklist"]["row_count"] == 261
    assert len({row["id"] for row in checklist}) == 261
    assert hashlib.sha256(canonical(checklist)).hexdigest() == state["checklist"]["canonical_sha256"]
    assert hashlib.sha256(canonical(reductions)).hexdigest() == state["reductions"]["canonical_sha256"]
    authorities = {row["path"]: row for row in state["legacy_authorities"]["files"]}
    legacy_state = authorities["MOP_COLLAPSE_STATE.json"]
    prior_state = json.loads(
        subprocess.check_output(["git", "show", f"{legacy_state['tag']}:{legacy_state['path']}"], cwd=ROOT)
    )
    assert checklist == prior_state["checklist"]
    legacy_log = authorities["collapse/MOP_REDUCTION_LOG.json"]
    prior = json.loads(
        subprocess.check_output(["git", "show", f"{legacy_log['tag']}:{legacy_log['path']}"], cwd=ROOT)
    )
    prior_count = len(prior["events"])
    assert reductions[:prior_count] == prior["events"]
    assert reductions[prior_count]["batch"] == "normalized_single_durable_authority"
    assert reductions[-1]["checkpoint_status"] in {"focused_green_pending_full_suite", "green"}
    try:
        _unique_object([("duplicate", 1), ("duplicate", 2)])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON keys must fail closed")


def test_no_optional_pack_or_second_runtime_authority_is_hidden_in_the_checkout():
    assert not list(ROOT.glob("condensation.packs*.json"))
    assert not (ROOT / ".mop/packs").exists()
    assert not (ROOT / "collapse/packs").exists()
