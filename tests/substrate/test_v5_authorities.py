from __future__ import annotations

from substrate import v5authorities


def test_v5_construction_authorities_are_active_and_evidence_bound() -> None:
    documents = v5authorities.construction_documents()
    assert len(documents) >= 80
    assert all(document["implementation_status"] == "active" for document in documents.values())
    assert all(document["evidence_routes"] for document in documents.values())
    assert all(document["activation"] is False for document in documents.values())
    assert documents["SUBSTRATE_V5_MODEL_REGISTRY.json"]["mechanism"]["count"] >= 10
    assert documents["SUBSTRATE_V5_ACQUISITION_LEDGER.json"]["mechanism"]["bytes_downloaded"] == 0
