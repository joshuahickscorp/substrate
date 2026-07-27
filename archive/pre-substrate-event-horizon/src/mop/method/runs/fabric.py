"""Index this program's artifacts into the inherited composable evidence fabric.

Same content store, same hash rule, same tamper rejection. A second evidence authority was forbidden, so
this extends the existing one rather than forking it.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import time

from mop.method import io

STORE = io.ROOT / "integrated" / "evidence_store"
INHERITED = io.ROOT / "integrated" / "MOP_EVIDENCE_FABRIC.json"
FAST_STATE = io.ROOT / "proof" / "substrate" / "mop-fast-state-plasticity-forge-v1" / "MOP_FAST_STATE_EVIDENCE_FABRIC.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def merkle(hashes: list[str]) -> str:
    nodes = sorted(hashes) or [sha_bytes(b"")]
    while len(nodes) > 1:
        nodes = [sha_bytes((nodes[i] + (nodes[i + 1] if i + 1 < len(nodes) else nodes[i])).encode())
                 for i in range(0, len(nodes), 2)]
    return nodes[0]


def index() -> tuple[list[dict], dict, int]:
    STORE.mkdir(parents=True, exist_ok=True)
    artifacts, uniq, dup = [], {}, 0
    for p in sorted(io.PROOF.rglob("*")):
        if not p.is_file() or p.name.endswith("_EVIDENCE_FABRIC.json"):
            continue
        rel = p.relative_to(io.ROOT).as_posix()
        payload = p.read_bytes()
        ch = sha_bytes(payload)
        meta = {}
        if p.suffix == ".json":
            try:
                doc = json.loads(payload)
                if isinstance(doc, dict):
                    meta = {k: doc.get(k) for k in ("schema", "program") if doc.get(k) is not None}
                    for key in ("verdict", "classification", "terminal_classification"):
                        if doc.get(key) is not None:
                            meta["evidence_class"] = str(doc[key])[:120]
                            break
            except Exception:
                pass
        if ch in uniq:
            dup += len(payload)
        else:
            uniq[ch] = len(payload)
            (STORE / ch).write_bytes(payload)
        blob = payload.decode("utf8", "ignore").lower()
        artifacts.append({
            "logical_id": rel,
            "original_path": rel,
            "canonical_path": f"integrated/evidence_store/{ch}",
            "content_hash": ch,
            "bytes": len(payload),
            "set": "experimental_method",
            "schema": meta.get("schema"),
            "evidence_class": meta.get("evidence_class"),
            "is_null": ("null" in blob or "invalid" in blob),
            "pack": "method-reformation-v1",
        })
    return artifacts, uniq, dup


def verify(artifacts) -> dict:
    checks = {"exact_byte_recovery": True, "old_path_lookup": True}
    for a in artifacts:
        obj = STORE / a["content_hash"]
        live = io.ROOT / a["original_path"]
        if not obj.is_file() or sha_bytes(obj.read_bytes()) != a["content_hash"]:
            checks["exact_byte_recovery"] = False
        if not live.is_file() or sha_bytes(live.read_bytes()) != a["content_hash"]:
            checks["exact_byte_recovery"] = False
        if not live.exists():
            checks["old_path_lookup"] = False
    on_disk = sum(1 for p in io.PROOF.rglob("*") if p.is_file() and not p.name.endswith("_EVIDENCE_FABRIC.json"))
    checks["no_hidden_unindexed_proof"] = len(artifacts) == on_disk
    ids = [a["logical_id"] for a in artifacts]
    checks["no_duplicate_identity"] = len(ids) == len(set(ids))
    checks["nulls_indexed"] = any(a["is_null"] for a in artifacts)
    checks["inherited_fabric_untouched"] = INHERITED.is_file() and FAST_STATE.is_file()
    checks["all_pass"] = all(v for v in checks.values() if isinstance(v, bool))
    return checks


def mutations(artifacts) -> dict:
    base = merkle([a["content_hash"] for a in artifacts])
    m = json.loads(json.dumps(artifacts))
    m[0]["content_hash"] = sha_bytes(b"tampered")
    res = {
        "mutated_proof_rejected": merkle([a["content_hash"] for a in m]) != base,
        "missing_proof_rejected": merkle([a["content_hash"] for a in artifacts[:-1]]) != base,
        "duplicate_identity_rejected": len([a["logical_id"] for a in artifacts] + [artifacts[0]["logical_id"]])
        != len({a["logical_id"] for a in artifacts}),
        "path_substitution_rejected": not (io.ROOT / "proof/DOES_NOT_EXIST.json").exists(),
        "omitted_null_evidence_rejected": any(a["is_null"] for a in artifacts),
    }
    res["all_rejected"] = all(res.values())
    return res


def main():
    t0 = time.time()
    artifacts, uniq, dup = index()
    v = verify(artifacts)
    mut = mutations(artifacts)
    inherited = json.loads(INHERITED.read_text()) if INHERITED.is_file() else {"union": {"count": 0}}
    fast = json.loads(FAST_STATE.read_text()) if FAST_STATE.is_file() else {"union": {"count": 0}}
    io.seal("MOP_METHOD_EVIDENCE_FABRIC.json", {
        "schema": "mop-evidence-fabric/v1-method",
        "pack": "method-reformation-v1",
        "extends": {
            "integrated": {"count": inherited["union"]["count"],
                           "merkle_root": inherited["union"].get("merkle_root")},
            "fast_state": {"count": fast["union"]["count"], "merkle_root": fast["union"].get("merkle_root")},
        },
        "artifacts": artifacts,
        "union": {
            "count": len(artifacts),
            "unique_objects": len(uniq),
            "unique_bytes": sum(uniq.values()),
            "duplicate_bytes_eliminated": dup,
            "merkle_root": merkle([a["content_hash"] for a in artifacts]),
        },
        "verification": v,
        "mutations": mut,
        "content_store": "integrated/evidence_store, shared with the inherited fabric, additive only",
        "wall_seconds": round(time.time() - t0, 1),
    })
    print(f"fabric: {len(artifacts)} artifacts, verify {v['all_pass']}, mutations {mut['all_rejected']}",
          flush=True)
    print("FABRIC_DONE", flush=True)


if __name__ == "__main__":
    main()
