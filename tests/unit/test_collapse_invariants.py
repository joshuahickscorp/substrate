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
    index = json.loads((ROOT / "collapse/MOP_PROOF_INDEX.json").read_text(encoding="utf-8"))
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

    documents = json.loads((ROOT / "collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json").read_text(encoding="utf-8"))
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
    state = json.loads((ROOT / "MOP_COLLAPSE_STATE.json").read_text(encoding="utf-8"))
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
    assert reductions[:-1] == prior["events"]
    assert reductions[-1]["batch"] == "normalized_single_durable_authority"
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
