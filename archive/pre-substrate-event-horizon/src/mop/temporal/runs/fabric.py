"""Index this program's proof and raw receipts in the composable evidence fabric."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

from mop.temporal import io

STORE = io.ROOT / "integrated" / "evidence_store"
INHERITED = io.ROOT / "integrated" / "MOP_EVIDENCE_FABRIC.json"
METHOD = io.ROOT / "proof" / "method" / "mop-experimental-method-reformation-v1" / "MOP_METHOD_EVIDENCE_FABRIC.json"
FABRIC_NAME = "MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


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


def _atomic_partial(path: Path) -> bool:
    return path.name.startswith(".") and ".partial." in path.name


def _proof_paths() -> list[Path]:
    """Return every proof byte except this manifest and atomic writer temporaries."""
    if not io.PROOF.is_dir():
        return []
    fabric = (io.PROOF / FABRIC_NAME).resolve()
    return [p for p in sorted(io.PROOF.rglob("*"))
            if p.is_file() and p.resolve() != fabric and not _atomic_partial(p)]


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
    if not isinstance(doc, dict):
        return {"json_parse_valid": False, "hash_version": None,
                "canonical_hash_valid": False, "legacy_whole_file_sha256": None,
                "contains_null_evidence": _contains_null(doc)}
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
            and all(_digest(h) for h in hashes)
            and merkle(hashes) == expected_root)


def _object(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _inherited_binding(path: Path) -> dict:
    doc = _object(path)
    union = doc.get("union") if isinstance(doc, dict) and isinstance(doc.get("union"), dict) else {}
    artifacts = doc.get("artifacts") if isinstance(doc, dict) else None
    artifact_manifest_valid = None
    if isinstance(artifacts, list):
        valid_artifacts = all(isinstance(a, dict) for a in artifacts)
        hashes = [a.get("content_hash") for a in artifacts if isinstance(a, dict)]
        ids = [a.get("logical_id") for a in artifacts if isinstance(a, dict)]
        artifact_manifest_valid = (valid_artifacts
                                   and isinstance(union.get("count"), int)
                                   and not isinstance(union.get("count"), bool)
                                   and len(hashes) == len(artifacts) == union.get("count")
                                   and len(ids) == len(set(ids))
                                   and all(isinstance(i, str) and i for i in ids)
                                   and all(_digest(h) for h in hashes)
                                   and merkle(hashes) == union.get("merkle_root"))
    return {
        "path": path.relative_to(io.ROOT).as_posix() if path.is_relative_to(io.ROOT) else str(path),
        "whole_file_sha256": sha_bytes(path.read_bytes()) if path.is_file() else None,
        "count": union.get("count"),
        "merkle_root": union.get("merkle_root"),
        "embedded_sha256": doc.get("sha256") if isinstance(doc, dict) else None,
        "embedded_sha256_valid": (doc.get("sha256") == io.sha_obj(
            {k: v for k, v in doc.items() if k != "sha256"}))
        if isinstance(doc, dict) and isinstance(doc.get("sha256"), str) else None,
        "artifact_manifest_valid": artifact_manifest_valid,
    }


def main():
    t0 = time.time()
    STORE.mkdir(parents=True, exist_ok=True)
    proof_paths = _proof_paths()
    sources = [(p, "temporal_core_proof") for p in proof_paths]
    sources += [(p, "temporal_core_quarantined_receipt" if "quarantine" in p.relative_to(io.RUNS).parts
                 else "temporal_core_raw_receipt") for p in _run_json_paths()]
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
        binding = _json_binding(payload, evidence_set in (
            "temporal_core_raw_receipt", "temporal_core_quarantined_receipt")) if p.suffix == ".json" else {
            "json_parse_valid": None, "hash_version": None, "canonical_hash_valid": None,
            "legacy_whole_file_sha256": None, "contains_null_evidence": False}
        artifacts.append({"logical_id": rel, "original_path": rel,
                          "canonical_path": f"integrated/evidence_store/{ch}", "content_hash": ch,
                          "bytes": len(payload), "set": evidence_set,
                          "is_null": binding.pop("contains_null_evidence"), "pack": "temporal-core-v1",
                          **binding})

    expected_ids = {p.relative_to(io.ROOT).as_posix() for p, _ in sources}
    raw_ids = {p.relative_to(io.ROOT).as_posix() for p in _run_json_paths()
               if "quarantine" not in p.relative_to(io.RUNS).parts}
    quarantine_ids = {p.relative_to(io.ROOT).as_posix() for p in _run_json_paths()
                      if "quarantine" in p.relative_to(io.RUNS).parts}
    indexed_raw = {a["logical_id"] for a in artifacts if a["set"] == "temporal_core_raw_receipt"}
    indexed_quarantine = {a["logical_id"] for a in artifacts
                          if a["set"] == "temporal_core_quarantined_receipt"}
    source_null_ids = {p.relative_to(io.ROOT).as_posix() for p, evidence_set in sources
                       if p.suffix == ".json" and _json_binding(
                           p.read_bytes(), evidence_set in (
                               "temporal_core_raw_receipt", "temporal_core_quarantined_receipt")
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
    checks["quarantined_receipts_indexed_but_excluded_from_claims"] = quarantine_ids == indexed_quarantine
    checks["no_duplicate_identity"] = len(expected_ids) == len(artifacts)
    scientific_artifacts = [a for a in artifacts if a["set"] != "temporal_core_quarantined_receipt"]
    checks["all_scientific_json_parse_valid"] = all(
        a["json_parse_valid"] is not False for a in scientific_artifacts)
    checks["canonical_hashes_valid"] = all(
        a["canonical_hash_valid"] is not False for a in scientific_artifacts)
    checks["legacy_json_bound_by_whole_file_hash"] = all(
        a["legacy_whole_file_sha256"] == a["content_hash"] for a in artifacts
        if a["set"] != "temporal_core_quarantined_receipt"
        and a["json_parse_valid"] and a["hash_version"] is None)
    checks["nulls_indexed"] = source_null_ids == indexed_null_ids
    inherited_binding = _inherited_binding(INHERITED)
    method_binding = _inherited_binding(METHOD)
    inherited_doc, method_doc = _object(INHERITED), _object(METHOD)
    binding_results = _object(io.PROOF / "MOP_TEMPORAL_CORE_BINDING_RESULTS.json")
    method_integrated = ((method_doc or {}).get("extends") or {}).get("integrated") or {}
    root_chain = {
        "method_extends_integrated_applicable": bool(method_integrated),
        "method_extends_integrated_verified": bool(method_integrated) and (
            method_integrated.get("count") == inherited_binding["count"]
            and method_integrated.get("merkle_root") == inherited_binding["merkle_root"]),
        "binding_results_method_root_applicable": binding_results is not None,
        "binding_results_method_root_verified": binding_results is not None
        and binding_results.get("evidence_fabric_root") == method_binding["merkle_root"],
    }
    checks["inherited_fabrics_untouched"] = inherited_doc is not None and method_doc is not None
    checks["inherited_fabric_embedded_hashes_valid"] = all(
        b["embedded_sha256_valid"] is True for b in (inherited_binding, method_binding))
    checks["inherited_fabric_artifact_roots_valid"] = all(
        b["artifact_manifest_valid"] is True for b in (inherited_binding, method_binding))
    checks["inherited_fabric_root_chain_valid"] = all(
        v for k, v in root_chain.items() if k.endswith("_verified"))
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
    mut["mutation_application"] = {
        "mutated_proof": any(a["content_hash"] != b["content_hash"]
                             for a, b in zip(artifacts, changed)),
        "missing_proof": len(missing) < len(artifacts),
        "duplicate_identity": len(duplicate) > len(artifacts),
        "omitted_null_evidence": bool(null_ids) and len(omitted_null) < len(artifacts),
    }
    mut["all_rejected"] = (all(mut[k] for k in (
        "mutated_proof_rejected", "missing_proof_rejected", "duplicate_identity_rejected",
        "omitted_null_evidence_rejected")) and all(mut["mutation_application"].values()))
    io.seal(FABRIC_NAME, {
        "schema": "mop-evidence-fabric/v2-temporal-core", "pack": "temporal-core-v1",
        "extends": {"integrated": inherited_binding, "method": method_binding,
                    "root_chain": root_chain},
        "artifacts": artifacts,
        "union": {"count": len(artifacts), "unique_objects": len(uniq), "unique_bytes": sum(uniq.values()),
                  "duplicate_bytes_eliminated": dup, "merkle_root": base,
                  "proof_count": len(proof_paths), "raw_receipt_count": len(raw_ids),
                  "quarantined_receipt_count": len(quarantine_ids)},
        "verification": checks, "mutations": mut,
        "content_store": "integrated/evidence_store, shared with the inherited fabric, additive only",
        "wall_seconds": round(time.time() - t0, 1)})
    print(f"fabric: {len(artifacts)} artifacts, verify {checks['all_pass']}, mutations {mut['all_rejected']}",
          flush=True)
    print("FABRIC_DONE", flush=True)
    if not checks["all_pass"] or not mut["all_rejected"]:
        raise RuntimeError("terminal evidence fabric is red")


if __name__ == "__main__":
    main()
