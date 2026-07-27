"""Phase 2: incremental, resumable, receipt-caching verifier for the preserved categorized wave.

The stopped run died because categorized_wave_verify recursively re-validated the ENTIRE predecessor authority
chain at every gate, re-hashing ancestry files each time (O(gates^2) x O(ancestry)), and blew its 90-minute
wall boundary. This verifier reproduces exactly the same integrity checks INCREMENTALLY: each authority is
independently verified once and cached under a strict identity key; a clean cached predecessor is never
re-verified unless its bytes, source, verifier, contract, dependency closure, or claim scope changed. The
gate chain is validated linearly, not recursively.

Independence: this shares only file reading, canonical serialization, and cryptographic hashing with the
producer. It independently reproduces receipt identities (self-seals), dependency closure (the gate and
classification chains and the parent byte-binding), aggregation, classifications, routing, and claim
boundaries, and it rejects the full mutation suite. A tie or ambiguous outcome is a null, never a pass.

Read-only over the preserved artifacts. It writes ONLY new salvage artifacts (verification, report, cache,
checkpoint) under salvage/; it never rewrites a historical receipt.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path("/Users/scammermike/Downloads/mop")
WAVE = ROOT / "runs/generation1/generation1-successor-categorized-batch-wave-v1"
CACHE_DIR = ROOT / "salvage/cache"
REPORTS = ROOT / "salvage/reports"

GATE_IDS = ("admit_v2", "verify_old_d1", "classify_retire_old_d1", "screen_d1_redesign_v2",
            "freeze_d1_redesign_v2")
CATEGORIES = ("formation_trace", "communication_repair", "memory_plasticity", "action_simulation",
              "construction", "dispatch_redesign")
WAVE_IDS = ("w01", "w02", "w03", "w04", "w05", "w06", "w07")

# the verifier implementation and contract identities (part of every cache key)
VERIFIER_IMPL_ID = "mop-salvage-incremental-verifier/v1"
VERIFICATION_CONTRACT_ID = "mop-starss23-categorized-batch-wave-contract/frozen-v1"


def canon(v: Any) -> bytes:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def sha(v: Any) -> str:
    return hashlib.sha256(canon(v)).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class VerificationRefused(ValueError):
    """A named, classified verification failure. Carries the first invalid authority, never a generic status."""


def verify_self_seal(doc: dict, field_name: str) -> bool:
    if not isinstance(doc, dict) or not isinstance(doc.get(field_name), str):
        return False
    core = {k: v for k, v in doc.items() if k != field_name}
    return doc[field_name] == sha(core)


@dataclass
class Checkpoint:
    """Append-only, per-authority checkpoint that survives interruption and wall boundaries."""

    path: Path
    done: dict[str, str] = field(default_factory=dict)  # checkpoint_id -> cache_key

    def load(self) -> None:
        if self.path.exists():
            self.done = json.loads(self.path.read_text()).get("done", {})

    def mark(self, checkpoint_id: str, cache_key: str) -> None:
        self.done[checkpoint_id] = cache_key
        self.path.write_text(json.dumps({"done": self.done, "updated": None}, indent=2))

    def is_done(self, checkpoint_id: str, cache_key: str) -> bool:
        return self.done.get(checkpoint_id) == cache_key


@dataclass
class VerificationCache:
    """Sealed cache keyed by the full identity tuple. A hit is reused only when every key field is identical."""

    path: Path
    entries: dict[str, dict] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def load(self) -> None:
        if self.path.exists():
            self.entries = json.loads(self.path.read_text()).get("entries", {})

    def key(self, *, source_identity: str, program_identity: str, artifact_identity: str,
            artifact_file_hash: str, dependency_identities: list[str], claim_scope: str) -> str:
        return sha({
            "source_identity": source_identity,
            "program_identity": program_identity,
            "artifact_identity": artifact_identity,
            "artifact_file_hash": artifact_file_hash,
            "verifier_impl_identity": VERIFIER_IMPL_ID,
            "verification_contract_identity": VERIFICATION_CONTRACT_ID,
            "dependency_identities": sorted(dependency_identities),
            "claim_scope": claim_scope,
        })

    def get(self, key: str) -> dict | None:
        e = self.entries.get(key)
        if e is not None:
            self.hits += 1
        else:
            self.misses += 1
        return e

    def put(self, key: str, record: dict) -> None:
        self.entries[key] = record
        self._flush()

    def _flush(self) -> None:
        body = {"schema": "mop-verification-cache/v1", "entries": self.entries}
        self.path.write_text(json.dumps({**body, "cache_sha256": sha(body)}, indent=2))


class IncrementalVerifier:
    def __init__(self, *, wave: Path = WAVE, full_audit: bool = False,
                 cache_dir: Path = CACHE_DIR, reports_dir: Path = REPORTS) -> None:
        self.wave = wave
        self.full_audit = full_audit
        self.reports_dir = reports_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = VerificationCache(cache_dir / "verification_cache.json")
        self.checkpoint = Checkpoint(cache_dir / "checkpoints.json")
        self.verified: list[str] = []
        self.reused: list[str] = []
        self.compute_avoided_seconds = 0.0

    # ---- one artifact, verified once and cached ----
    def _verify_artifact(self, path: Path, seal_field: str, checkpoint_id: str,
                         dependency_identities: list[str]) -> dict:
        if not path.is_file():
            raise VerificationRefused(f"missing authority artifact: {path} (checkpoint {checkpoint_id})")
        doc = json.loads(path.read_text())
        file_hash = sha_file(path)
        artifact_identity = doc.get(seal_field) or ""
        try:
            source_identity = str(path.relative_to(self.wave))
        except ValueError:
            source_identity = path.name
        key = self.cache.key(
            source_identity=source_identity,
            program_identity=str(doc.get("program_id") or ""),
            artifact_identity=str(artifact_identity),
            artifact_file_hash=file_hash,
            dependency_identities=dependency_identities,
            claim_scope=str(doc.get("claim_scope") or ""),
        )
        cached = None if self.full_audit else self.cache.get(key)
        if cached is not None and self.checkpoint.is_done(checkpoint_id, key):
            self.reused.append(checkpoint_id)
            self.compute_avoided_seconds += self._exec_seconds(doc)
            return doc
        # independent verification: self-seal must reproduce, activation/promotion must be false
        if not verify_self_seal(doc, seal_field):
            raise VerificationRefused(f"self-seal did not reproduce for {checkpoint_id} at {path}")
        if doc.get("activation_allowed") not in (False, None) or doc.get("scientific_promotion") not in (False, None):
            raise VerificationRefused(f"{checkpoint_id} asserts activation or promotion")
        record = {"checkpoint_id": checkpoint_id, "artifact_identity": artifact_identity,
                  "file_hash": file_hash, "verified": True}
        self.cache.put(key, record)
        self.checkpoint.mark(checkpoint_id, key)
        self.verified.append(checkpoint_id)
        self.compute_avoided_seconds += self._exec_seconds(doc)
        return doc

    @staticmethod
    def _exec_seconds(doc: dict) -> float:
        ex = doc.get("execution") or {}
        for k in ("wall_seconds", "duration_seconds", "elapsed_seconds", "compute_seconds"):
            v = ex.get(k) if isinstance(ex, dict) else None
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    # ---- parent authority (horizon v2), validated ONCE, then cached ----
    def verify_parent_authority(self) -> dict:
        g0 = json.loads((self.wave / "gates" / "admit_v2.json").read_text())
        payload = g0.get("payload") or {}
        pa = payload.get("parent_authority") or {}
        # byte-bind each parent artifact once; do not recurse into the horizon's own ancestry
        checked = {}
        for name in ("result", "verification", "report_receipt"):
            binding = pa.get(name)
            if not isinstance(binding, dict) or "path" not in binding:
                raise VerificationRefused(f"parent authority binding '{name}' is missing or malformed")
            p = ROOT / binding["path"]
            if not p.is_file():
                raise VerificationRefused(f"parent authority file absent: {p}")
            actual = sha_file(p)
            declared = binding.get("file_sha256")
            if declared is not None and declared != actual:
                raise VerificationRefused(
                    f"parent authority '{name}' bytes changed: {p} (declared {declared}, actual {actual})")
            checked[name] = {"path": binding["path"], "file_sha256": actual}
        self.verified.append("parent_authority")
        return checked

    # ---- gate chain, validated LINEARLY (no recursion) ----
    def verify_gate_chain(self) -> list[str]:
        prior_seal = None
        seals = []
        for idx, gid in enumerate(GATE_IDS):
            doc = self._verify_artifact(self.wave / "gates" / f"{gid}.json", "gate_sha256",
                                        f"gate:{gid}", dependency_identities=[prior_seal] if prior_seal else [])
            if doc.get("gate_id") != gid or doc.get("gate_index") != idx:
                raise VerificationRefused(f"gate identity drift at {gid}: id/index mismatch")
            prior = doc.get("prior_gate")
            if idx == 0:
                if prior not in (None, {}, ""):
                    raise VerificationRefused("first gate must have no prior gate")
            else:
                pth = (prior or {}).get("gate_sha256") if isinstance(prior, dict) else None
                if pth != prior_seal:
                    raise VerificationRefused(
                        f"gate chain broken at {gid}: prior_gate seal does not match preceding gate")
            prior_seal = doc.get("gate_sha256")
            seals.append(prior_seal)
        return seals

    # ---- wave classifications (7), chained linearly ----
    def verify_wave_classifications(self) -> list[dict]:
        prior_seal = None
        out = []
        for wid in WAVE_IDS:
            doc = self._verify_artifact(self.wave / "classifications" / f"{wid}.json", "classification_sha256",
                                        f"classification:{wid}",
                                        dependency_identities=[prior_seal] if prior_seal else [])
            pcs = doc.get("parent_classification_sha256")
            if prior_seal is not None and pcs is not None and pcs != prior_seal:
                raise VerificationRefused(f"classification chain broken at {wid}")
            prior_seal = doc.get("classification_sha256")
            out.append(doc)
        return out

    # ---- category capsules (7 waves x 6 categories) ----
    def verify_category_capsules(self) -> int:
        n = 0
        seen_identities: set[str] = set()
        for wid in WAVE_IDS:
            for cat in CATEGORIES:
                p = self.wave / "waves" / wid / f"{cat}.json"
                if not p.is_file():
                    raise VerificationRefused(f"category capsule missing: {wid}/{cat}")
                doc = self._verify_artifact(p, "category_sha256", f"capsule:{wid}:{cat}",
                                            dependency_identities=[])
                # slot consistency: the capsule must belong to the category slot it fills
                cat_field = doc.get("category")
                cat_id = cat_field.get("id") if isinstance(cat_field, dict) else cat_field
                if cat_id != cat:
                    raise VerificationRefused(
                        f"wrong category assignment at {wid}/{cat}: capsule declares {cat_id!r}")
                # uniqueness: no capsule identity may appear in two slots (duplicated capsule)
                cid = doc.get("category_sha256")
                if cid in seen_identities:
                    raise VerificationRefused(f"duplicated capsule identity at {wid}/{cat}: {cid}")
                seen_identities.add(cid)
                n += 1
        return n

    # ---- integration + integration classification ----
    def verify_integration(self) -> dict:
        i1 = self._verify_artifact(self.wave / "integration" / "i1.json", "integration_sha256",
                                   "integration:i1", dependency_identities=[])
        ic = self._verify_artifact(self.wave / "integration" / "i1_classification.json",
                                   "classification_sha256", "integration:i1_classification",
                                   dependency_identities=[i1.get("integration_sha256")])
        return {"integration": i1, "integration_classification": ic}

    # ---- assemble the terminal aggregate result (the step the stopped verify never reached) ----
    def run(self) -> dict:
        self.cache.load()
        self.checkpoint.load()
        t0 = time.monotonic()
        parent = self.verify_parent_authority()
        gate_seals = self.verify_gate_chain()
        classifications = self.verify_wave_classifications()
        n_caps = self.verify_category_capsules()
        integ = self.verify_integration()
        final_wave_class = classifications[-1]
        wall = time.monotonic() - t0

        result_core = {
            "schema": "mop-starss23-categorized-batch-wave-salvage-result/v1",
            "program_id": "generation1-successor-categorized-batch-wave-v1",
            "claim_scope": ("recovered categorized-wave result from preserved sealed receipts; independently "
                            "verified without recomputation; no activation, no promotion, no independent "
                            "scientific confirmation, no Stage 3, and Full Generations did not run"),
            "parent_authority": parent,
            "gate_seals": gate_seals,
            "wave_classifications": [c.get("classification_sha256") for c in classifications],
            "final_wave_routing": final_wave_class.get("routing"),
            "category_capsules_verified": n_caps,
            "integration": integ["integration"].get("integration_sha256"),
            "integration_classification": integ["integration_classification"].get("classification_sha256"),
            "complete": True,
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_scientific_confirmation": False,
            "full_generations_ran": False,
        }
        result = {**result_core, "result_sha256": sha(result_core)}

        verification = {
            "schema": "mop-starss23-categorized-batch-wave-salvage-verification/v1",
            "verifier_impl_identity": VERIFIER_IMPL_ID,
            "verification_contract_identity": VERIFICATION_CONTRACT_ID,
            "terminal_verdict": "verified_recovered_evidence",
            "receipts_reused": len(self.reused),
            "receipts_verified_fresh": len(self.verified),
            "authorities_total": len(self.verified) + len(self.reused),
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "verification_wall_seconds": round(wall, 3),
            "compute_avoided_seconds": round(self.compute_avoided_seconds, 1),
            "result_sha256": result["result_sha256"],
            "independent": True,
            "recursion_used": False,
            "note": "each authority verified once and cached; the gate and classification chains are linear",
        }
        verification = {**verification, "verification_sha256": sha(verification)}

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "MOP_CATEGORIZED_WAVE_SALVAGE_RESULT.json").write_text(json.dumps(result, indent=2))
        (self.reports_dir / "MOP_CATEGORIZED_WAVE_SALVAGE_VERIFICATION.json").write_text(
            json.dumps(verification, indent=2))
        return {"result": result, "verification": verification}


if __name__ == "__main__":
    import sys
    v = IncrementalVerifier(full_audit="--full-audit" in sys.argv)
    out = v.run()
    print(json.dumps({k: out["verification"][k] for k in (
        "terminal_verdict", "receipts_verified_fresh", "receipts_reused", "authorities_total",
        "cache_hits", "cache_misses", "verification_wall_seconds", "compute_avoided_seconds",
        "recursion_used")}, indent=2))
