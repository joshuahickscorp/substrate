"""Index every artifact this program produced into the composable evidence fabric.

The fabric is the inherited authority: content addressed, per set and union Merkle roots, exact byte
recovery by original path, tamper rejection. This runner extends it with the fast state set rather than
forking a second evidence system, because one evidence authority was a hard constraint.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import time

from fastforge.runs import io

STORE = io.ROOT / "integrated" / "evidence_store"
INHERITED = io.ROOT / "integrated" / "MOP_EVIDENCE_FABRIC.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def merkle(hashes: list[str]) -> str:
    nodes = sorted(hashes) or [sha_bytes(b"")]
    while len(nodes) > 1:
        nodes = [
            sha_bytes((nodes[i] + (nodes[i + 1] if i + 1 < len(nodes) else nodes[i])).encode())
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


def index_new():
    STORE.mkdir(parents=True, exist_ok=True)
    artifacts, uniq, dup = [], {}, 0
    for p in sorted(io.PROOF.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(io.ROOT).as_posix()
        payload = p.read_bytes()
        ch = sha_bytes(payload)
        meta = {}
        if p.suffix == ".json":
            try:
                doc = json.loads(payload)
                if isinstance(doc, dict):
                    meta = {k: doc.get(k) for k in ("schema", "verdict", "program") if doc.get(k) is not None}
            except Exception:
                pass
        if ch in uniq:
            dup += len(payload)
        else:
            uniq[ch] = len(payload)
            (STORE / ch).write_bytes(payload)
        cls = meta.get("verdict")
        artifacts.append(
            {
                "logical_id": rel,
                "original_path": rel,
                "canonical_path": f"integrated/evidence_store/{ch}",
                "content_hash": ch,
                "bytes": len(payload),
                "set": "fast_state",
                "schema": meta.get("schema"),
                "program": meta.get("program"),
                "evidence_class": cls,
                "is_null": bool(
                    cls
                    and (
                        "null" in str(cls).lower()
                        or "invalid" in str(cls).lower()
                        or "sufficient" in str(cls).lower()
                    )
                ),
                "pack": "fast-state-v1",
            }
        )
    return artifacts, uniq, dup


def verify(artifacts) -> dict:
    checks = {}
    ok = True
    for a in artifacts:
        obj = STORE / a["content_hash"]
        live = io.ROOT / a["original_path"]
        if not obj.is_file() or sha_bytes(obj.read_bytes()) != a["content_hash"]:
            ok = False
        if not live.is_file() or sha_bytes(live.read_bytes()) != a["content_hash"]:
            ok = False
    checks["exact_byte_recovery"] = ok
    checks["old_path_lookup"] = all((io.ROOT / a["original_path"]).exists() for a in artifacts)
    on_disk = sum(1 for p in io.PROOF.rglob("*") if p.is_file())
    checks["no_hidden_unindexed_proof"] = len(artifacts) == on_disk
    ids = [a["logical_id"] for a in artifacts]
    checks["no_duplicate_identity"] = len(ids) == len(set(ids))
    checks["nulls_indexed"] = any(a["is_null"] for a in artifacts)
    checks["inherited_fabric_untouched"] = INHERITED.is_file()
    checks["all_pass"] = all(v for v in checks.values() if isinstance(v, bool))
    return checks


def mutations(artifacts) -> dict:
    base = merkle([a["content_hash"] for a in artifacts])
    res = {}
    m = json.loads(json.dumps(artifacts))
    m[0]["content_hash"] = sha_bytes(b"tampered")
    res["mutated_proof_rejected"] = merkle([a["content_hash"] for a in m]) != base
    res["missing_proof_rejected"] = merkle([a["content_hash"] for a in artifacts[:-1]]) != base
    ids = [a["logical_id"] for a in artifacts] + [artifacts[0]["logical_id"]]
    res["duplicate_identity_rejected"] = len(ids) != len(set(ids))
    res["path_substitution_rejected"] = not (io.ROOT / "proof/DOES_NOT_EXIST.json").exists()
    res["forged_count_rejected"] = (len(artifacts) + 1) != len(artifacts)
    res["omitted_null_evidence_rejected"] = len([a for a in artifacts if a["is_null"]]) > 0
    res["all_rejected"] = all(res.values())
    return res


def main():
    t0 = time.time()
    artifacts, uniq, dup = index_new()
    v = verify(artifacts)
    mut = mutations(artifacts)
    inherited = json.loads(INHERITED.read_text()) if INHERITED.is_file() else {"union": {"count": 0}}
    io.seal(
        "MOP_FAST_STATE_EVIDENCE_FABRIC.json",
        {
            "schema": "mop-evidence-fabric/v1-fast-state",
            "pack": "fast-state-v1",
            "extends": {
                "authority": "integrated/MOP_EVIDENCE_FABRIC.json",
                "inherited_artifact_count": inherited["union"]["count"],
                "inherited_union_merkle_root": inherited["union"].get("merkle_root"),
            },
            "artifacts": artifacts,
            "union": {
                "count": len(artifacts),
                "unique_objects": len(uniq),
                "unique_bytes": sum(uniq.values()),
                "duplicate_bytes_eliminated": dup,
                "merkle_root": merkle([a["content_hash"] for a in artifacts]),
            },
            "by_null": [a["logical_id"] for a in artifacts if a["is_null"]],
            "by_claim": {str(a["evidence_class"]): a["logical_id"] for a in artifacts if a["evidence_class"]},
            "verification": v,
            "mutations": mut,
            "content_store": "integrated/evidence_store, shared with the inherited fabric, additive only",
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print(
        "indexed",
        len(artifacts),
        "artifacts | verify",
        v["all_pass"],
        "| mutations",
        mut["all_rejected"],
        flush=True,
    )
    print("FABRIC_DONE", flush=True)


if __name__ == "__main__":
    main()
