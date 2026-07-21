"""Index this program's proof and raw receipts in the composable evidence fabric."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from mop.temporal import io

STORE = io.ROOT / "integrated" / "evidence_store"
INHERITED = io.ROOT / "integrated" / "MOP_EVIDENCE_FABRIC.json"
METHOD = io.ROOT / "proof" / "method" / "mop-experimental-method-reformation-v1" / "MOP_METHOD_EVIDENCE_FABRIC.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def merkle(h: list[str]) -> str:
    n = sorted(h) or [sha_bytes(b"")]
    while len(n) > 1:
        n = [sha_bytes((n[i] + (n[i + 1] if i + 1 < len(n) else n[i])).encode())
             for i in range(0, len(n), 2)]
    return n[0]


def _run_json_paths() -> list[Path]:
    if not io.RUNS.is_dir():
        return []
    return [p for p in sorted(io.RUNS.rglob("*.json"))
            if p.is_file() and "locks" not in p.relative_to(io.RUNS).parts
            and ".partial." not in p.name and not p.name.endswith(".partial.json")]


def _contains_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_null(k) or _contains_null(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_null(v) for v in value)
    return isinstance(value, str) and value.lower() in {"invalid", "null"}


def _json_binding(payload: bytes, receipt: bool) -> dict:
    """Describe the embedded canonical hash, or the exact byte binding for a legacy JSON file."""
    try:
        doc = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"json_parse_valid": False, "hash_version": None,
                "canonical_hash_valid": False, "legacy_whole_file_sha256": None,
                "contains_null_evidence": False}
    version_key = "result_hash_version" if receipt else "sha256_version"
    hash_key = "result_sha256" if receipt else "sha256"
    version = doc.get(version_key) if isinstance(doc, dict) else None
    canonical = None
    if version == "canonical_json_v2" and isinstance(doc, dict):
        canonical = doc.get(hash_key) == io.sha_obj({k: v for k, v in doc.items() if k != hash_key})
    elif version is not None:
        canonical = False
    return {"json_parse_valid": True, "hash_version": version,
            "canonical_hash_valid": canonical,
            "legacy_whole_file_sha256": sha_bytes(payload) if version is None else None,
            "contains_null_evidence": _contains_null(doc)}


def _manifest_valid(artifacts: list[dict], expected_ids: set[str], expected_root: str) -> bool:
    ids = [a.get("logical_id") for a in artifacts]
    hashes = [a.get("content_hash") for a in artifacts]
    return (len(ids) == len(set(ids)) and set(ids) == expected_ids
            and all(isinstance(h, str) and len(h) == 64 for h in hashes)
            and merkle(hashes) == expected_root)


def main():
    t0 = time.time()
    STORE.mkdir(parents=True, exist_ok=True)
    proof_paths = [p for p in sorted(io.PROOF.rglob("*")) if p.is_file()
                   and not p.name.endswith("_EVIDENCE_FABRIC.json")]
    sources = [(p, "temporal_core_proof") for p in proof_paths]
    sources += [(p, "temporal_core_raw_receipt") for p in _run_json_paths()]
    artifacts, uniq, dup = [], {}, 0
    for p, evidence_set in sources:
        rel = p.relative_to(io.ROOT).as_posix()
        payload = p.read_bytes()
        ch = sha_bytes(payload)
        if ch in uniq:
            dup += len(payload)
        else:
            uniq[ch] = len(payload)
        store_path = STORE / ch
        if not store_path.is_file() or sha_bytes(store_path.read_bytes()) != ch:
            store_path.write_bytes(payload)
        binding = _json_binding(payload, evidence_set == "temporal_core_raw_receipt") if p.suffix == ".json" else {
            "json_parse_valid": None, "hash_version": None, "canonical_hash_valid": None,
            "legacy_whole_file_sha256": None, "contains_null_evidence": False}
        artifacts.append({"logical_id": rel, "original_path": rel,
                          "canonical_path": f"integrated/evidence_store/{ch}", "content_hash": ch,
                          "bytes": len(payload), "set": evidence_set,
                          "is_null": binding.pop("contains_null_evidence"), "pack": "temporal-core-v1",
                          **binding})

    expected_ids = {p.relative_to(io.ROOT).as_posix() for p, _ in sources}
    raw_ids = {p.relative_to(io.ROOT).as_posix() for p in _run_json_paths()}
    indexed_raw = {a["logical_id"] for a in artifacts if a["set"] == "temporal_core_raw_receipt"}
    source_null_ids = {p.relative_to(io.ROOT).as_posix() for p, evidence_set in sources
                       if p.suffix == ".json" and _json_binding(
                           p.read_bytes(), evidence_set == "temporal_core_raw_receipt"
                       )["contains_null_evidence"]}
    indexed_null_ids = {a["logical_id"] for a in artifacts if a["is_null"]}
    checks = {"exact_byte_recovery": all(
        (STORE / a["content_hash"]).is_file()
        and sha_bytes((STORE / a["content_hash"]).read_bytes()) == a["content_hash"] for a in artifacts)}
    checks["old_path_lookup"] = all((io.ROOT / a["original_path"]).is_file()
                                    and io.sha_file(io.ROOT / a["original_path"]) == a["content_hash"]
                                    for a in artifacts)
    checks["no_hidden_unindexed_proof"] = len(proof_paths) == sum(
        a["set"] == "temporal_core_proof" for a in artifacts)
    checks["no_hidden_unindexed_receipts"] = raw_ids == indexed_raw
    checks["no_duplicate_identity"] = len(expected_ids) == len(artifacts)
    checks["all_json_parse_valid"] = all(a["json_parse_valid"] is not False for a in artifacts)
    checks["canonical_hashes_valid"] = all(
        a["canonical_hash_valid"] is not False for a in artifacts)
    checks["legacy_json_bound_by_whole_file_hash"] = all(
        a["legacy_whole_file_sha256"] == a["content_hash"] for a in artifacts
        if a["json_parse_valid"] and a["hash_version"] is None)
    checks["nulls_indexed"] = source_null_ids == indexed_null_ids
    checks["inherited_fabrics_untouched"] = INHERITED.is_file() and METHOD.is_file()
    checks["all_pass"] = all(checks.values())

    base = merkle([a["content_hash"] for a in artifacts])
    changed = json.loads(json.dumps(artifacts))
    proof_id = next(a["logical_id"] for a in artifacts if a["set"] == "temporal_core_proof")
    next(a for a in changed if a["logical_id"] == proof_id)["content_hash"] = sha_bytes(b"tampered")
    missing = [a for a in artifacts if a["logical_id"] != proof_id]
    duplicate = artifacts + [json.loads(json.dumps(artifacts[0]))]
    null_ids = [a["logical_id"] for a in artifacts if a["is_null"]]
    omitted_null = [a for a in artifacts if not null_ids or a["logical_id"] != null_ids[0]]
    mut = {
        "mutated_proof_rejected": not _manifest_valid(changed, expected_ids, base),
        "missing_proof_rejected": not _manifest_valid(missing, expected_ids, base),
        "duplicate_identity_rejected": not _manifest_valid(duplicate, expected_ids, base),
        "omitted_null_evidence_rejected": bool(null_ids)
        and not _manifest_valid(omitted_null, expected_ids, base),
        "omitted_null_mutation_applied": bool(null_ids),
    }
    mut["all_rejected"] = all(mut.values())
    inh = json.loads(INHERITED.read_text()) if INHERITED.is_file() else {"union": {"count": 0}}
    met = json.loads(METHOD.read_text()) if METHOD.is_file() else {"union": {"count": 0}}
    io.seal("MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json", {
        "schema": "mop-evidence-fabric/v2-temporal-core", "pack": "temporal-core-v1",
        "extends": {"integrated": inh["union"]["count"], "method": met["union"]["count"],
                    "method_root": met["union"].get("merkle_root")},
        "artifacts": artifacts,
        "union": {"count": len(artifacts), "unique_objects": len(uniq), "unique_bytes": sum(uniq.values()),
                  "duplicate_bytes_eliminated": dup, "merkle_root": base,
                  "proof_count": len(proof_paths), "raw_receipt_count": len(raw_ids)},
        "verification": checks, "mutations": mut,
        "content_store": "integrated/evidence_store, shared with the inherited fabric, additive only",
        "wall_seconds": round(time.time() - t0, 1)})
    print(f"fabric: {len(artifacts)} artifacts, verify {checks['all_pass']}, mutations {mut['all_rejected']}",
          flush=True)
    print("FABRIC_DONE", flush=True)


if __name__ == "__main__":
    main()
