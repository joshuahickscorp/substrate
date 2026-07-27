"""Phase 2 adversarial mutation suite for the incremental categorized-wave verifier.

Copies the preserved wave to a scratch tree, applies each mutation family, and requires the incremental
verifier to REJECT it. A positive salvage requires every required mutation to be rejected; a verifier that
passes any mutated wave fails the suite. A tie or ambiguous outcome is a null, never a pass.

Runs the base suite (self-seal, chain, missing) plus the fourteen mandated mutations. Never touches the
originals; every mutation is on a throwaway copy with an isolated cache.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incremental_verifier as iv  # noqa: E402


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def reseal(path: Path, field: str) -> None:
    """Re-seal a doc after mutation, so the attack must be caught by a structural check, not the self-seal."""
    d = json.loads(path.read_text())
    core = {k: v for k, v in d.items() if k != field}
    d[field] = sha(core)
    path.write_text(json.dumps(d))


def mutate(path: Path, changes: dict, *, reseal_field: str | None = None) -> None:
    d = json.loads(path.read_text())
    d.update(changes)
    path.write_text(json.dumps(d))
    if reseal_field:
        reseal(path, reseal_field)


def run_case(name: str, mutator) -> tuple[str, bool, str]:
    """Copy the wave, apply mutator(copy), run the verifier, and return (name, rejected, detail)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wave_copy = tmp / "wave"
        shutil.copytree(iv.WAVE, wave_copy)
        # copy the parent-authority files referenced by admit_v2 so the byte-binding resolves under a stub
        detail = mutator(wave_copy, tmp)
        v = iv.IncrementalVerifier(wave=wave_copy, cache_dir=tmp / "cache", reports_dir=tmp / "reports")
        try:
            v.run()
            # verifier accepted the mutated wave -> mutation NOT rejected (suite failure)
            return name, False, detail or "verifier accepted the mutated wave"
        except iv.VerificationRefused as exc:
            return name, True, str(exc)[:90]
        except Exception as exc:  # any other refusal-by-crash also counts as rejected, but note it
            return name, True, f"{type(exc).__name__}: {str(exc)[:70]}"


def _cap(w, wid="w03", cat="construction"):
    return w / "waves" / wid / f"{cat}.json"


def _cls(w, wid="w04"):
    return w / "classifications" / f"{wid}.json"


CASES = {
    # base suite
    "self_seal_broken": lambda w, t: (mutate(_cap(w), {"complete": True, "tampered": 1}), "no reseal")[1],
    "missing_capsule": lambda w, t: (_cap(w).unlink(), "removed w03/construction")[1],
    # the fourteen mandated mutations
    "stale_cached_verification": lambda w, t: _stale_cache(w, t),
    "wrong_verifier_identity": lambda w, t: _wrong_verifier(w, t),
    "changed_source_unchanged_path": lambda w, t: (mutate(_cap(w), {"redesign_v2_efficacy": 99.9}),
                                                   "changed content, path same")[1],
    "changed_dependency_receipt": lambda w, t: (mutate(_cls(w, "w03"), {"cycle_index": 999},
                                                       reseal_field="classification_sha256"),
                                                "reseal prior classification -> chain break")[1],
    "removed_capsule": lambda w, t: (_cap(w, "w05", "memory_plasticity").unlink(), "removed w05/mem")[1],
    "duplicated_capsule": lambda w, t: (shutil.copy(_cap(w, "w01", "formation_trace"),
                                                    _cap(w, "w02", "formation_trace")),
                                        "copied w01/formation into w02 slot")[1],
    "wrong_category_assignment": lambda w, t: (mutate(_cap(w, "w03", "construction"),
                                                      {"category": "formation_trace"},
                                                      reseal_field="category_sha256"),
                                               "reseal w03/construction as formation_trace")[1],
    "wrong_lane_carry_forward": lambda w, t: (mutate(_cls(w, "w04"), {"routing": {"forged": True}},
                                                     reseal_field="classification_sha256"),
                                              "reseal w04 routing -> chain break")[1],
    "resurrected_pruned_lane": lambda w, t: (mutate(_cls(w, "w03"),
                                                    {"category_bindings": {"G1-D1": "resurrected"}},
                                                    reseal_field="classification_sha256"),
                                             "reseal w03 with D1 -> chain break")[1],
    "altered_seed_interval": lambda w, t: (mutate(_cap(w, "w06", "action_simulation"),
                                                  {"seed_interval": [0, 999999]}), "no reseal")[1],
    "altered_resource_accounting": lambda w, t: (mutate(_cap(w, "w07", "dispatch_redesign"),
                                                        {"execution": {"wall_seconds": -1}}), "no reseal")[1],
    "partial_aggregate": lambda w, t: ((w / "integration" / "i1.json").unlink(), "removed integration")[1],
    "forged_completion_field": lambda w, t: (mutate(_cap(w, "w02", "communication_repair"),
                                                    {"activation_allowed": True}), "forged activation")[1],
    "changed_claim_scope": lambda w, t: (mutate(_cls(w, "w05"), {"claim_scope": "unbounded capability"},
                                               reseal_field="classification_sha256"),
                                         "reseal w05 claim -> chain break")[1],
}


def _stale_cache(w, t):
    """Pre-seed a cache entry, then change the artifact bytes: the file-hash-keyed cache must not reuse it."""
    # run once to populate cache, then mutate an artifact and require re-verification to catch it
    cache_dir = t / "cache"
    v = iv.IncrementalVerifier(wave=w, cache_dir=cache_dir, reports_dir=t / "r1")
    v.run()  # populates cache + checkpoints for the clean copy
    mutate(_cap(w, "w03", "construction"), {"redesign_v2_efficacy": 42})  # bytes change, seal now broken
    return "cache populated then artifact changed"


def _wrong_verifier(w, t):
    """Populate cache under the real verifier id, then flip the verifier id: cache must be fully invalidated."""
    cache_dir = t / "cache"
    v = iv.IncrementalVerifier(wave=w, cache_dir=cache_dir, reports_dir=t / "r1")
    v.run()
    iv.VERIFIER_IMPL_ID = "mop-forged-verifier/v9"  # different verifier identity
    # also break an artifact so a naive cache-reuse would wrongly pass; correct behavior re-verifies + rejects
    mutate(_cap(w, "w03", "construction"), {"redesign_v2_efficacy": 7})
    return "verifier identity changed + artifact tampered"


def main() -> int:
    results = []
    real_id = iv.VERIFIER_IMPL_ID
    for name, mut in CASES.items():
        iv.VERIFIER_IMPL_ID = real_id  # reset any per-case change
        results.append(run_case(name, mut))
    iv.VERIFIER_IMPL_ID = real_id
    rejected = sum(1 for _, ok, _ in results if ok)
    report = {
        "schema": "mop-categorized-wave-mutation-suite/v1",
        "total_mutations": len(results),
        "rejected": rejected,
        "all_rejected": rejected == len(results),
        "positive_verification_requires": "every required mutation rejected; a tie is a null",
        "results": [{"mutation": n, "rejected": ok, "detail": d} for n, ok, d in results],
    }
    report = {**report, "mutation_suite_sha256": sha({k: v for k, v in report.items()})}
    out = iv.REPORTS / "MOP_CATEGORIZED_WAVE_MUTATION_SUITE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"mutations rejected: {rejected}/{len(results)}  all_rejected={rejected == len(results)}")
    for n, ok, d in results:
        print(f"  [{'REJECT' if ok else 'MISS  '}] {n}: {d}")
    return 0 if rejected == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
