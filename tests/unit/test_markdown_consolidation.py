import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_docs as CD  # noqa: E402


def test_current_markdown_authority_is_exact() -> None:
    assert set(CD._project_markdown()) == set(CD.CURRENT_MD)
    assert len(CD.CURRENT_MD) == 4
    assert sum((ROOT / path).read_text().count("\n") for path in CD.CURRENT_MD) <= 8_000


def test_historical_document_index_is_sealed_and_recoverable() -> None:
    path = ROOT / "collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json"
    index = json.loads(path.read_text())
    seal = index.pop("canonical_sha256")
    canonical = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    assert hashlib.sha256(canonical).hexdigest() == seal
    assert index["archived_document_count"] == 170
    assert index["removed_document_count"] == 162
    assert index["archived_total_lines"] == 44_204
    assert index["removed_total_lines"] == 41_327
    assert all(not (ROOT / entry["path"]).exists() for entry in index["removed_documents"])
    entries = index["removed_documents"] + index["retained_source_versions"]
    assert all(len(entry["git_blob"]) == 40 and len(entry["sha256"]) == 64 for entry in entries)


def test_docs_gate_clean_after_consolidation() -> None:
    assert CD.check_docs() == []
