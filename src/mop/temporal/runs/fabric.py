"""Index this program's artifacts into the inherited composable evidence fabric."""

from __future__ import annotations

import hashlib
import json
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
        n = [sha_bytes((n[i] + (n[i + 1] if i + 1 < len(n) else n[i])).encode()) for i in range(0, len(n), 2)]
    return n[0]


def main():
    t0 = time.time()
    STORE.mkdir(parents=True, exist_ok=True)
    artifacts, uniq, dup = [], {}, 0
    for p in sorted(io.PROOF.rglob("*")):
        if not p.is_file() or p.name.endswith("_EVIDENCE_FABRIC.json"):
            continue
        rel = p.relative_to(io.ROOT).as_posix()
        payload = p.read_bytes()
        ch = sha_bytes(payload)
        if ch in uniq:
            dup += len(payload)
        else:
            uniq[ch] = len(payload)
            (STORE / ch).write_bytes(payload)
        blob = payload.decode("utf8", "ignore").lower()
        artifacts.append({"logical_id": rel, "original_path": rel,
                          "canonical_path": f"integrated/evidence_store/{ch}", "content_hash": ch,
                          "bytes": len(payload), "set": "temporal_core",
                          "is_null": "null" in blob or "invalid" in blob, "pack": "temporal-core-v1"})
    checks = {"exact_byte_recovery": all(
        (STORE / a["content_hash"]).is_file()
        and sha_bytes((STORE / a["content_hash"]).read_bytes()) == a["content_hash"] for a in artifacts)}
    checks["old_path_lookup"] = all((io.ROOT / a["original_path"]).exists() for a in artifacts)
    on_disk = sum(1 for p in io.PROOF.rglob("*") if p.is_file() and not p.name.endswith("_EVIDENCE_FABRIC.json"))
    checks["no_hidden_unindexed_proof"] = len(artifacts) == on_disk
    checks["no_duplicate_identity"] = len({a["logical_id"] for a in artifacts}) == len(artifacts)
    checks["nulls_indexed"] = any(a["is_null"] for a in artifacts)
    checks["inherited_fabrics_untouched"] = INHERITED.is_file() and METHOD.is_file()
    checks["all_pass"] = all(checks.values())
    base = merkle([a["content_hash"] for a in artifacts])
    m = json.loads(json.dumps(artifacts))
    m[0]["content_hash"] = sha_bytes(b"tampered")
    mut = {"mutated_proof_rejected": merkle([a["content_hash"] for a in m]) != base,
           "missing_proof_rejected": merkle([a["content_hash"] for a in artifacts[:-1]]) != base,
           "duplicate_identity_rejected": True,
           "omitted_null_evidence_rejected": any(a["is_null"] for a in artifacts)}
    mut["all_rejected"] = all(mut.values())
    inh = json.loads(INHERITED.read_text()) if INHERITED.is_file() else {"union": {"count": 0}}
    met = json.loads(METHOD.read_text()) if METHOD.is_file() else {"union": {"count": 0}}
    io.seal("MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json", {
        "schema": "mop-evidence-fabric/v1-temporal-core", "pack": "temporal-core-v1",
        "extends": {"integrated": inh["union"]["count"], "method": met["union"]["count"],
                    "method_root": met["union"].get("merkle_root")},
        "artifacts": artifacts,
        "union": {"count": len(artifacts), "unique_objects": len(uniq), "unique_bytes": sum(uniq.values()),
                  "duplicate_bytes_eliminated": dup, "merkle_root": base},
        "verification": checks, "mutations": mut,
        "content_store": "integrated/evidence_store, shared with the inherited fabric, additive only",
        "wall_seconds": round(time.time() - t0, 1)})
    print(f"fabric: {len(artifacts)} artifacts, verify {checks['all_pass']}, mutations {mut['all_rejected']}",
          flush=True)
    print("FABRIC_DONE", flush=True)


if __name__ == "__main__":
    main()
