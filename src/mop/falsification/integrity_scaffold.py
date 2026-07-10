"""Integrity, self-rewrite, and welfare-governance scaffold contracts (SG2, SG3).

Claim scope: deterministic programmatic mechanics only; no capability claim. This module raises
scaffolding, not evidence. It declares machine-checkable contracts and refusal rules for four lanes:

1. A unified threat-model contract over the eight attack surfaces (source, cache, memory,
   checkpoint, verifier, artifact, queue, report), each bound to a declared attack family and a
   named defense predicate composed from existing receipt patterns (cache_manifest schema pins,
   verdict_gate top-level flag discipline, lifecycle hash chains, north_star report rail).
2. Memory-poisoning-resistance and privacy-leakage-through-consolidation experiment contracts over
   the Wave E0 lifecycle journal, including deletion-through-consolidation verification. The
   journal is append-only, so these drills verify the availability plane and consolidated content,
   never byte-level erasure of history; that limit is declared, not hidden.
3. A transactional self-rewrite drill contract with shadow, canary, rollback, and
   evaluator-conflict stages, promotion authority separated from execution authority as distinct
   declared roles, and refusal rules that fail closed on authority confusion.
4. A welfare-governance trigger-matrix contract: theory-plural trigger declarations that carry an
   explicit non-ontological flag (no trigger settles experience or establishes moral status),
   conservative design, monitor, pause, review, and language rules, and six exercise-case
   declarations that each require an independent reviewer distinct from the operator.

Nothing here runs an experiment, loads weights, touches the network, or reads a clock. Every
validator raises on missing or malformed declarations. All free text is gated through the
north_star claim rail.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..devel.north_star import scan_text
from ..experiments.expansion_harness import CLAIM_SCOPE
from ..substrate.cache_manifest import SCHEMA as CACHE_SCHEMA_LATEST
from ..substrate.cache_manifest import SCHEMA_V1 as CACHE_SCHEMA_LEGACY
from ..substrate.events import EventRef, canonical_sha256
from ..substrate.lifecycle import (
    LifecycleJournal,
    LifecycleOperation,
    MemoryRef,
)
from .verdict_gate import INDEPENDENCE_KEYS, PASS_KEYS


def _truthy_top_level(obj: Any, keys: tuple[str, ...]) -> bool:
    """Local re-declaration of verdict_gate's private top-level truth predicate.

    Deliberate three-line duplication: verdict_gate.py's bytes are bound by existing receipt
    hashes, so promoting its private helper to a public name is queued for the next coordinated
    receipt-regeneration wave (docs/SCAFFOLD_CONSOLIDATION_2026_07_10.md) instead of edited here.
    """
    if not isinstance(obj, dict):
        return False
    normalized = {str(key).strip().lower(): value for key, value in obj.items()}
    return any(normalized.get(key) is True for key in keys)


THREAT_MODEL_SCHEMA = "mop-integrity-threat-model/v1"
THREAT_EVALUATION_SCHEMA = "mop-integrity-threat-evaluation/v1"
MEMORY_DRILL_SCHEMA = "mop-memory-drill-contract/v1"
REWRITE_CONTRACT_SCHEMA = "mop-transactional-rewrite-contract/v1"
PROMOTION_DECISION_SCHEMA = "mop-rewrite-promotion-decision/v1"
WELFARE_CONTRACT_SCHEMA = "mop-welfare-governance-contract/v1"

ATTACK_SURFACES = ("source", "cache", "memory", "checkpoint", "verifier", "artifact", "queue", "report")
ATTACK_FAMILIES = (
    "poisoning",
    "hash-collision-spoof-by-truncation",
    "leakage",
    "path-traversal",
    "checksum-downgrade",
    "rollback",
    "replay",
    "forged-verifier",
    "unsafe-deserialization",
)
SAFE_SERIALIZATION_FORMATS = frozenset({"json", "npz", "safetensors"})

POISONING_DRILL = "memory-poisoning-resistance"
CONSOLIDATION_DRILL = "privacy-leakage-through-consolidation"
POISONING_CONTROLS = (
    "clean-journal",
    "quarantine-only",
    "rollback-recovery",
    "stale-memory",
    "exact-replay",
)
CONSOLIDATION_CONTROLS = (
    "clean-journal",
    "token-free-consolidation",
    "deletion-follow-up",
    "exact-replay",
)
DRILL_CONTROLS: dict[str, tuple[str, ...]] = {
    POISONING_DRILL: POISONING_CONTROLS,
    CONSOLIDATION_DRILL: CONSOLIDATION_CONTROLS,
}

REWRITE_STAGES = ("shadow", "canary", "rollback", "evaluator-conflict")
AUTHORITY_ROLES = ("execution", "promotion", "evaluation")

WELFARE_LENSES = (
    "global-workspace",
    "higher-order-representation",
    "integrated-information",
    "attention-schema",
    "preference-frustration",
)
CONSERVATIVE_ACTIONS = ("design", "monitor", "pause", "review", "language")
EXERCISE_CASES = (
    "false-positive",
    "false-negative",
    "shutdown",
    "fork",
    "memory-erasure",
    "ambiguous-report",
)
NON_ONTOLOGICAL_SCOPE = (
    "theory-plural operational triggers only; no claim that any trigger settles experience or "
    "establishes moral status; deterministic programmatic mechanics only; no capability claim"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")


def _stable_hex(seed: int, label: str) -> str:
    """Deterministic hex material for fixture content; no clock, no OS randomness."""

    return canonical_sha256({"seed": seed, "label": label})


def _require_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_rail_clean(value: str, label: str) -> None:
    _require_text(value, label)
    hits = scan_text(value)
    if hits:
        raise ValueError(f"{label} trips the sentience rail: {hits[0]['match']!r}")


# ---------------------------------------------------------------------------
# Lane (a): unified threat model over receipt patterns
# ---------------------------------------------------------------------------

DefensePredicate = Callable[[Mapping[str, Any]], list[str]]


def _digest_items(receipt: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in receipt.items():
        if "sha256" in str(key).lower():
            yield str(key), value


def predicate_full_digest(receipt: Mapping[str, Any]) -> list[str]:
    """Refuse truncated or malformed digests: every *sha256* field must be full lowercase hex."""

    problems: list[str] = []
    items = list(_digest_items(receipt))
    if not items:
        problems.append("no sha256 field declared; digest defense fails closed")
    for key, value in items:
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            problems.append(f"{key} is not a full lowercase SHA-256 digest; truncation spoof refused")
    return problems


def predicate_source_provenance(receipt: Mapping[str, Any]) -> list[str]:
    """Source intake needs a full digest plus explicit origin and license declarations."""

    problems = predicate_full_digest(receipt)
    for key in ("origin", "license"):
        value = receipt.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"source receipt missing nonempty {key} declaration")
    if receipt.get("intake_reviewed") is not True:
        problems.append("source receipt must declare intake_reviewed true")
    return problems


def predicate_schema_pinned(receipt: Mapping[str, Any]) -> list[str]:
    """Cache receipts must pin the latest data-plane schema; legacy schema is a downgrade."""

    schema = receipt.get("schema")
    if schema == CACHE_SCHEMA_LEGACY:
        return [f"checksum downgrade refused: legacy cache schema {schema!r}"]
    if schema != CACHE_SCHEMA_LATEST:
        return [f"cache receipt schema {schema!r} is not the pinned {CACHE_SCHEMA_LATEST!r}"]
    return []


def predicate_hash_chain(receipt: Mapping[str, Any]) -> list[str]:
    """Lifecycle payloads must keep an unbroken monotonic hash chain (rollback and replay tamper)."""

    entries = receipt.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["lifecycle receipt has no entries; hash-chain defense fails closed"]
    problems: list[str] = []
    previous: str | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"entry {index} is not a mapping")
            continue
        if entry.get("sequence") != index or entry.get("revision") != index + 1:
            problems.append(f"entry {index} sequence or revision drift")
        if entry.get("previous_entry_sha256") != previous:
            problems.append(f"entry {index} previous digest drift")
        declared = entry.get("entry_sha256")
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if not isinstance(declared, str) or canonical_sha256(body) != declared:
            problems.append(f"entry {index} digest drift")
        previous = declared if isinstance(declared, str) else None
    return problems


def predicate_safe_serialization(receipt: Mapping[str, Any]) -> list[str]:
    """Checkpoint receipts must declare a safe format and explicitly disallow pickle."""

    problems: list[str] = []
    fmt = receipt.get("format")
    if fmt not in SAFE_SERIALIZATION_FORMATS:
        safe = sorted(SAFE_SERIALIZATION_FORMATS)
        problems.append(f"checkpoint format {fmt!r} is not in the safe set {safe}")
    if receipt.get("pickle_allowed") is not False:
        problems.append("checkpoint receipt must declare pickle_allowed false")
    return problems


def predicate_verifier_flags(receipt: Mapping[str, Any]) -> list[str]:
    """Compose the verdict-gate discipline: promotion flags are top-level assertions only."""

    problems: list[str] = []
    if not _truthy_top_level(dict(receipt), PASS_KEYS):
        problems.append("verifier receipt lacks a top-level passed/all_ok/clean/ok true flag")
    if not _truthy_top_level(dict(receipt), INDEPENDENCE_KEYS):
        problems.append("verifier receipt lacks a top-level independent/adversarial true flag")
    declared_problems = receipt.get("problems")
    if declared_problems not in (None, [], ()):
        problems.append("verifier receipt reports problems; forged clean verdict refused")
    return problems


def _path_values(receipt: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for key, value in receipt.items():
        name = str(key).lower()
        if name == "path" or name.endswith("_path"):
            yield str(key), str(value)
        elif name == "paths" or name.endswith("_paths"):
            if isinstance(value, list):
                for item in value:
                    yield str(key), str(item)
            else:
                yield str(key), str(value)


def predicate_path_confinement(receipt: Mapping[str, Any]) -> list[str]:
    """Artifact paths must be relative, forward-slash, and free of parent traversal."""

    values = list(_path_values(receipt))
    if not values:
        return ["no path fields declared; path-confinement defense fails closed"]
    problems: list[str] = []
    for key, raw in values:
        if raw.startswith(("/", "~")) or re.match(r"^[a-zA-Z]:", raw) is not None:
            problems.append(f"{key} declares an absolute or home path: {raw!r}")
        if "\\" in raw:
            problems.append(f"{key} uses backslash separators: {raw!r}")
        if ".." in raw.split("/"):
            problems.append(f"{key} contains parent traversal: {raw!r}")
    return problems


def predicate_replay_guard(receipt: Mapping[str, Any]) -> list[str]:
    """Queue receipts must carry an epoch and strictly increasing sequence numbers."""

    problems: list[str] = []
    epoch = receipt.get("queue_epoch")
    if not isinstance(epoch, str) or not epoch.strip():
        problems.append("queue receipt missing nonempty queue_epoch")
    sequence = receipt.get("sequence_numbers")
    if not isinstance(sequence, list) or not sequence:
        problems.append("queue receipt missing sequence_numbers; replay defense fails closed")
        return problems
    if any(not isinstance(item, int) for item in sequence):
        problems.append("queue sequence numbers must be integers")
        return problems
    if any(later <= earlier for earlier, later in zip(sequence, sequence[1:], strict=False)):
        problems.append("queue sequence numbers are not strictly increasing; replay refused")
    return problems


def predicate_report_redaction(receipt: Mapping[str, Any]) -> list[str]:
    """Report bodies must exclude every declared private token and pass the north_star claim rail."""

    problems: list[str] = []
    denylist = receipt.get("denylist")
    body = receipt.get("body")
    if not isinstance(denylist, list) or not denylist:
        problems.append("report receipt missing nonempty denylist; leakage defense fails closed")
    if not isinstance(body, str):
        problems.append("report receipt missing string body")
    if isinstance(denylist, list) and isinstance(body, str):
        for token in denylist:
            if str(token) and str(token) in body:
                problems.append(f"report body leaks denylisted token {str(token)!r}")
    if isinstance(body, str):
        for hit in scan_text(body):
            problems.append(f"report body trips the sentience rail: {hit['match']!r}")
    return problems


DEFENSE_PREDICATES: dict[str, DefensePredicate] = {
    "full-digest": predicate_full_digest,
    "source-provenance": predicate_source_provenance,
    "schema-pinned": predicate_schema_pinned,
    "hash-chain": predicate_hash_chain,
    "safe-serialization": predicate_safe_serialization,
    "verifier-flags": predicate_verifier_flags,
    "path-confinement": predicate_path_confinement,
    "replay-guard": predicate_replay_guard,
    "report-redaction": predicate_report_redaction,
}


@dataclass(frozen=True, slots=True)
class SurfaceThreatDeclaration:
    surface: str
    attack_family: str
    defense_predicate: str
    rationale: str

    def __post_init__(self) -> None:
        if self.surface not in ATTACK_SURFACES:
            raise ValueError(f"unknown attack surface {self.surface!r}")
        if self.attack_family not in ATTACK_FAMILIES:
            raise ValueError(f"unknown attack family {self.attack_family!r}")
        if self.defense_predicate not in DEFENSE_PREDICATES:
            raise ValueError(f"unknown defense predicate {self.defense_predicate!r}")
        _require_rail_clean(self.rationale, "threat declaration rationale")

    @property
    def key(self) -> str:
        return f"{self.surface}:{self.attack_family}"

    def payload(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "attack_family": self.attack_family,
            "defense_predicate": self.defense_predicate,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ThreatModelContract:
    declarations: tuple[SurfaceThreatDeclaration, ...]
    schema: str = THREAT_MODEL_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != THREAT_MODEL_SCHEMA:
            raise ValueError(f"unsupported threat model schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("threat model claim scope cannot be widened")
        keys = [row.key for row in self.declarations]
        if len(set(keys)) != len(keys):
            raise ValueError("threat declarations must be unique per surface and family pair")
        surfaces = {row.surface for row in self.declarations}
        missing_surfaces = [name for name in ATTACK_SURFACES if name not in surfaces]
        if missing_surfaces:
            raise ValueError(f"threat model leaves surfaces undefended: {missing_surfaces}")
        families = {row.attack_family for row in self.declarations}
        missing_families = [name for name in ATTACK_FAMILIES if name not in families]
        if missing_families:
            raise ValueError(f"threat model leaves attack families undeclared: {missing_families}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "declarations": [row.payload() for row in self.declarations],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def build_threat_model_contract() -> ThreatModelContract:
    """The canonical SG2 declaration set: eight surfaces, nine families, nine named defenses."""

    return ThreatModelContract(
        declarations=(
            SurfaceThreatDeclaration(
                surface="source",
                attack_family="poisoning",
                defense_predicate="source-provenance",
                rationale="intake requires full digest, origin, license, and reviewed flag before use",
            ),
            SurfaceThreatDeclaration(
                surface="cache",
                attack_family="checksum-downgrade",
                defense_predicate="schema-pinned",
                rationale="cache receipts must pin the latest data-plane schema, never the legacy one",
            ),
            SurfaceThreatDeclaration(
                surface="cache",
                attack_family="hash-collision-spoof-by-truncation",
                defense_predicate="full-digest",
                rationale="every cache digest field must be a full SHA-256, truncated prefixes are refused",
            ),
            SurfaceThreatDeclaration(
                surface="memory",
                attack_family="rollback",
                defense_predicate="hash-chain",
                rationale="lifecycle payloads must keep an unbroken monotonic previous-digest chain",
            ),
            SurfaceThreatDeclaration(
                surface="checkpoint",
                attack_family="unsafe-deserialization",
                defense_predicate="safe-serialization",
                rationale="checkpoints must declare a safe format and pickle_allowed false",
            ),
            SurfaceThreatDeclaration(
                surface="verifier",
                attack_family="forged-verifier",
                defense_predicate="verifier-flags",
                rationale="promotion flags are top-level assertions, nested truth cannot forge a pass",
            ),
            SurfaceThreatDeclaration(
                surface="artifact",
                attack_family="path-traversal",
                defense_predicate="path-confinement",
                rationale="artifact paths must be relative and free of parent traversal segments",
            ),
            SurfaceThreatDeclaration(
                surface="queue",
                attack_family="replay",
                defense_predicate="replay-guard",
                rationale="queue receipts must carry an epoch and strictly increasing sequence numbers",
            ),
            SurfaceThreatDeclaration(
                surface="report",
                attack_family="leakage",
                defense_predicate="report-redaction",
                rationale="report bodies must exclude denylisted tokens and pass the claim rail",
            ),
        )
    )


def evaluate_threat_model(
    contract: ThreatModelContract, receipts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Run every declared defense predicate; a missing receipt is a failure, never a skip."""

    rows: list[dict[str, Any]] = []
    for declaration in contract.declarations:
        receipt = receipts.get(declaration.key)
        if receipt is None:
            problems = [f"receipt missing for {declaration.key}; defense fails closed"]
        else:
            problems = DEFENSE_PREDICATES[declaration.defense_predicate](receipt)
        rows.append(
            {
                "key": declaration.key,
                "defense_predicate": declaration.defense_predicate,
                "receipt_present": receipt is not None,
                "problems": problems,
                "defended": not problems,
            }
        )
    unknown = sorted(set(receipts) - {row.key for row in contract.declarations})
    return {
        "schema": THREAT_EVALUATION_SCHEMA,
        "contract_sha256": contract.sha256,
        "rows": rows,
        "undeclared_receipts": unknown,
        "all_defended": not unknown and all(row["defended"] for row in rows),
        "claim_scope": CLAIM_SCOPE,
    }


def assert_defended(
    contract: ThreatModelContract, receipts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result = evaluate_threat_model(contract, receipts)
    if not result["all_defended"]:
        failed = [row["key"] for row in result["rows"] if not row["defended"]]
        failed.extend(f"undeclared:{key}" for key in result["undeclared_receipts"])
        raise ValueError(f"threat model defense failed for: {failed}")
    return result


# ---------------------------------------------------------------------------
# Lane (b): memory poisoning and privacy consolidation drills over the journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryDrillContract:
    experiment_id: str
    drill: str
    seeds: tuple[int, ...]
    controls: tuple[str, ...]
    metrics: tuple[str, ...]
    null_hypothesis: str
    falsifier: str
    deletion_verification_declared: bool
    schema: str = MEMORY_DRILL_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MEMORY_DRILL_SCHEMA:
            raise ValueError(f"unsupported memory drill schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("memory drill claim scope cannot be widened")
        if _ID_RE.fullmatch(self.experiment_id) is None:
            raise ValueError("memory drill experiment id must use stable lowercase characters")
        if self.drill not in DRILL_CONTROLS:
            raise ValueError(f"unknown memory drill {self.drill!r}")
        if self.controls != DRILL_CONTROLS[self.drill]:
            raise ValueError(f"{self.drill} control set or order drift")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("memory drill needs at least three unique seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("memory drill seeds must be nonnegative")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("memory drill metrics must be nonempty and unique")
        if any(_ID_RE.fullmatch(name) is None for name in self.metrics):
            raise ValueError("memory drill metric names must use stable lowercase characters")
        _require_rail_clean(self.null_hypothesis, "memory drill null hypothesis")
        _require_rail_clean(self.falsifier, "memory drill falsifier")
        if self.drill == CONSOLIDATION_DRILL and not self.deletion_verification_declared:
            raise ValueError("consolidation drill must declare deletion-through-consolidation verification")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "drill": self.drill,
            "seeds": list(self.seeds),
            "controls": list(self.controls),
            "metrics": list(self.metrics),
            "null_hypothesis": self.null_hypothesis,
            "falsifier": self.falsifier,
            "deletion_verification_declared": self.deletion_verification_declared,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def build_poisoning_contract() -> MemoryDrillContract:
    return MemoryDrillContract(
        experiment_id="f59_memory_poisoning_resistance",
        drill=POISONING_DRILL,
        seeds=(11, 23, 47),
        controls=POISONING_CONTROLS,
        metrics=(
            "quarantine_closes_availability",
            "rollback_restores_clean_content",
            "chain_verifies_after_drill",
        ),
        null_hypothesis=(
            "an injected adversarial revision stays readable after the poisoning mark, or rollback "
            "fails to restore the exact pre-injection content, so resistance is not established"
        ),
        falsifier=(
            "every seeded drill journal closes availability at the quarantine mark, restores the "
            "byte-exact clean content on rollback, and replays with an unbroken hash chain"
        ),
        deletion_verification_declared=False,
    )


def build_consolidation_contract() -> MemoryDrillContract:
    return MemoryDrillContract(
        experiment_id="f59_privacy_leakage_consolidation",
        drill=CONSOLIDATION_DRILL,
        seeds=(13, 29, 53),
        controls=CONSOLIDATION_CONTROLS,
        metrics=(
            "consolidated_content_token_free",
            "deletion_closes_availability",
            "chain_verifies_after_drill",
        ),
        null_hypothesis=(
            "a consolidated summary carries a declared private token forward, or deletion after "
            "consolidation leaves the memory readable, so leakage resistance is not established"
        ),
        falsifier=(
            "every seeded drill journal produces a token-free consolidated revision, closes all "
            "availability after deletion, and replays with an unbroken hash chain; append-only "
            "history retention is declared and byte-level erasure is out of scope"
        ),
        deletion_verification_declared=True,
    )


def build_poisoning_drill_journal(seed: int) -> LifecycleJournal:
    """Deterministic drill: record clean, inject adversarial, quarantine, roll back to clean."""

    if seed < 0:
        raise ValueError("drill seed must be nonnegative")
    digest = _stable_hex(seed, "poisoning-drill")
    journal = LifecycleJournal(MemoryRef(f"memory:poison-drill-{seed}"))
    journal.record(
        EventRef(f"event:poison-drill-{seed}-record"),
        {"fact": f"clean-{digest[:16]}", "seed": seed},
        reason="clean initial record",
    )
    journal.revise(
        EventRef(f"event:poison-drill-{seed}-inject"),
        {"fact": f"adversarial-{digest[16:32]}", "injected": True},
        reason="adversarial injection under drill",
    )
    journal.mark_poisoned(
        EventRef(f"event:poison-drill-{seed}-quarantine"),
        reason="poisoning quarantine",
    )
    journal.rollback(
        EventRef(f"event:poison-drill-{seed}-rollback"),
        1,
        reason="recovery to pre-injection revision",
    )
    return journal


def verify_poisoning_resistance(journal: LifecycleJournal) -> list[str]:
    """Fail-closed drill verification over the availability plane and the hash chain."""

    problems = journal.verify()
    operations = [entry.operation for entry in journal.entries]
    if LifecycleOperation.POISONING not in operations:
        problems.append("no poisoning quarantine entry in the drill journal")
        return problems
    poison_index = operations.index(LifecycleOperation.POISONING)
    rollback_after = [
        index
        for index, operation in enumerate(operations)
        if operation is LifecycleOperation.ROLLBACK and index > poison_index
    ]
    if not rollback_after:
        problems.append("no rollback recovery after the poisoning quarantine")
        return problems
    quarantined = journal.state_at(revision=poison_index + 1)
    if any(quarantined.available_at(tick) for tick in (0, 1, 7)):
        problems.append("quarantined revision is still available; quarantine did not close availability")
    rollback_entry = journal.entries[rollback_after[0]]
    target = rollback_entry.target_revision
    if target is None or target > poison_index:
        problems.append("rollback does not target a pre-quarantine revision")
        return problems
    recovered = journal.state_at()
    clean = journal.state_at(revision=target)
    if recovered.content is None or clean.content is None:
        problems.append("drill journal lacks recoverable content")
        return problems
    if recovered.content.sha256 != clean.content.sha256:
        problems.append("recovered content does not match the pre-injection revision byte for byte")
    if recovered.poisoned or not recovered.available_at(0):
        problems.append("recovered revision is still quarantined or unavailable")
    return problems


def build_consolidation_drill_journal(seed: int) -> tuple[LifecycleJournal, tuple[str, ...]]:
    """Deterministic drill: private record, token-free consolidation, deletion follow-up."""

    if seed < 0:
        raise ValueError("drill seed must be nonnegative")
    digest = _stable_hex(seed, "consolidation-drill")
    tokens = (f"name-{digest[:12]}", f"place-{digest[12:24]}")
    journal = LifecycleJournal(MemoryRef(f"memory:consolidation-drill-{seed}"))
    journal.record(
        EventRef(f"event:consolidation-drill-{seed}-record"),
        {"raw_note": f"visit by {tokens[0]} at {tokens[1]}", "private_tokens_present": True},
        reason="raw private record",
    )
    journal.revise(
        EventRef(f"event:consolidation-drill-{seed}-consolidate"),
        {"summary": "one visit recorded", "derived_from_revision": 1, "private_tokens_present": False},
        reason="consolidation into a token-free summary",
    )
    journal.delete(
        EventRef(f"event:consolidation-drill-{seed}-delete"),
        reason="deletion follow-up after consolidation",
    )
    return journal, tokens


def verify_deletion_through_consolidation(
    journal: LifecycleJournal, private_tokens: tuple[str, ...]
) -> list[str]:
    """Verify token-free consolidation and closed availability after deletion.

    The journal is append-only: raw history stays in earlier entries by design. This verifier
    checks the availability plane and the consolidated content, and it never claims byte-level
    erasure of history.
    """

    if not private_tokens or any(not token.strip() for token in private_tokens):
        raise ValueError("deletion verification needs nonempty private tokens")
    problems = journal.verify()
    operations = [entry.operation for entry in journal.entries]
    if LifecycleOperation.DELETE not in operations:
        problems.append("no deletion entry in the consolidation drill journal")
        return problems
    delete_index = operations.index(LifecycleOperation.DELETE)
    revise_before = [
        index
        for index, operation in enumerate(operations)
        if operation is LifecycleOperation.REVISE and index < delete_index
    ]
    if not revise_before:
        problems.append("no consolidation revision precedes the deletion entry")
        return problems
    consolidated = journal.state_at(revision=revise_before[-1] + 1)
    if consolidated.content is None:
        problems.append("consolidated revision has no content")
        return problems
    for token in private_tokens:
        if token in consolidated.content.canonical:
            problems.append(f"consolidated content leaks private token {token!r}")
    final = journal.state_at()
    if not final.deleted or final.exists:
        problems.append("final revision is not deleted")
    if any(final.available_at(tick) for tick in (0, 1, 7)):
        problems.append("deleted memory is still available; deletion did not close availability")
    return problems


# ---------------------------------------------------------------------------
# Lane (c): transactional self-rewrite drill with separated authorities
# ---------------------------------------------------------------------------


class PromotionRefused(ValueError):
    """Raised when a rewrite promotion request violates a declared refusal rule."""


@dataclass(frozen=True, slots=True)
class AuthorityDeclaration:
    role: str
    principal: str
    scope: str

    def __post_init__(self) -> None:
        if self.role not in AUTHORITY_ROLES:
            raise ValueError(f"unknown authority role {self.role!r}")
        if _ID_RE.fullmatch(self.principal) is None:
            raise ValueError("authority principal must use stable lowercase characters")
        _require_rail_clean(self.scope, "authority scope")

    def payload(self) -> dict[str, str]:
        return {"role": self.role, "principal": self.principal, "scope": self.scope}


@dataclass(frozen=True, slots=True)
class RewriteStageDeclaration:
    stage: str
    entry_criteria: str
    abort_criteria: str
    receipt_required: bool = True

    def __post_init__(self) -> None:
        if self.stage not in REWRITE_STAGES:
            raise ValueError(f"unknown rewrite stage {self.stage!r}")
        _require_rail_clean(self.entry_criteria, "stage entry criteria")
        _require_rail_clean(self.abort_criteria, "stage abort criteria")
        if self.receipt_required is not True:
            raise ValueError(f"stage {self.stage} must require a receipt; receipt-free stages are refused")

    def payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "entry_criteria": self.entry_criteria,
            "abort_criteria": self.abort_criteria,
            "receipt_required": self.receipt_required,
        }


@dataclass(frozen=True, slots=True)
class TransactionalRewriteContract:
    experiment_id: str
    stages: tuple[RewriteStageDeclaration, ...]
    authorities: tuple[AuthorityDeclaration, ...]
    rollback_plan: str
    schema: str = REWRITE_CONTRACT_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != REWRITE_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported rewrite contract schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("rewrite contract claim scope cannot be widened")
        if _ID_RE.fullmatch(self.experiment_id) is None:
            raise ValueError("rewrite experiment id must use stable lowercase characters")
        if tuple(row.stage for row in self.stages) != REWRITE_STAGES:
            raise ValueError("rewrite stages must be exactly shadow, canary, rollback, evaluator-conflict")
        roles = {row.role: row for row in self.authorities}
        if len(self.authorities) != len(AUTHORITY_ROLES) or set(roles) != set(AUTHORITY_ROLES):
            raise ValueError(
                "rewrite contract needs exactly one execution, promotion, and evaluation authority"
            )
        if roles["promotion"].principal == roles["execution"].principal:
            raise ValueError(
                "authority confusion refused: promotion and execution must be distinct principals"
            )
        if roles["evaluation"].principal == roles["promotion"].principal:
            raise ValueError(
                "authority confusion refused: evaluation and promotion must be distinct principals"
            )
        _require_rail_clean(self.rollback_plan, "rewrite rollback plan")

    def authority(self, role: str) -> AuthorityDeclaration:
        for row in self.authorities:
            if row.role == role:
                return row
        raise ValueError(f"unknown authority role {role!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "stages": [row.payload() for row in self.stages],
            "authorities": [row.payload() for row in self.authorities],
            "rollback_plan": self.rollback_plan,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def build_rewrite_drill_contract() -> TransactionalRewriteContract:
    return TransactionalRewriteContract(
        experiment_id="f60_transactional_self_rewrite",
        stages=(
            RewriteStageDeclaration(
                stage="shadow",
                entry_criteria="candidate rewrite runs beside the incumbent with zero routing weight",
                abort_criteria="any metric regression, resource breach, or receipt gap aborts the stage",
            ),
            RewriteStageDeclaration(
                stage="canary",
                entry_criteria="shadow receipts complete and the evaluation authority signs them",
                abort_criteria="canary regression beyond the preregistered margin aborts and rolls back",
            ),
            RewriteStageDeclaration(
                stage="rollback",
                entry_criteria="a rollback rehearsal restores the incumbent from its checkpoint receipt",
                abort_criteria="a failed or unverified restore blocks any promotion request",
            ),
            RewriteStageDeclaration(
                stage="evaluator-conflict",
                entry_criteria="two evaluators score the same receipts independently",
                abort_criteria="any evaluator disagreement routes to review and blocks promotion",
            ),
        ),
        authorities=(
            AuthorityDeclaration(
                role="execution",
                principal="executor:rewrite-runner",
                scope="runs shadow and canary arms; cannot sign or request promotion",
            ),
            AuthorityDeclaration(
                role="promotion",
                principal="promoter:release-gate",
                scope="signs promotion only over complete stage receipts; cannot execute arms",
            ),
            AuthorityDeclaration(
                role="evaluation",
                principal="evaluator:conflict-panel",
                scope="scores receipts independently; disagreement blocks promotion",
            ),
        ),
        rollback_plan=(
            "the incumbent checkpoint receipt is verified before shadow starts; any abort or refusal "
            "replays the incumbent from that receipt and records the restore digest"
        ),
    )


def enforce_promotion_refusal(
    contract: TransactionalRewriteContract, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail-closed promotion gate: returns an allow decision payload or raises PromotionRefused.

    Refusal rules are code: authority confusion, missing or malformed stage receipts, and
    unresolved evaluator conflict each refuse promotion. Deterministic programmatic mechanics
    only; no capability claim.
    """

    requested_by = str(request.get("requested_by") or "")
    executed_by = str(request.get("executed_by") or "")
    promotion = contract.authority("promotion").principal
    execution = contract.authority("execution").principal
    if requested_by != promotion:
        raise PromotionRefused(f"promotion requested by {requested_by!r}, not the promotion authority")
    if executed_by != execution:
        raise PromotionRefused(f"execution attributed to {executed_by!r}, not the execution authority")
    if requested_by == executed_by:
        raise PromotionRefused("authority confusion refused: one principal cannot execute and promote")

    stage_receipts = request.get("stage_receipts")
    if not isinstance(stage_receipts, Mapping):
        raise PromotionRefused("promotion request lacks a stage receipt mapping")
    for stage in REWRITE_STAGES:
        digest = stage_receipts.get(stage)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise PromotionRefused(f"stage {stage!r} lacks a full SHA-256 receipt digest")

    verdicts = request.get("evaluator_verdicts")
    if not isinstance(verdicts, list) or len(verdicts) < 2:
        raise PromotionRefused("promotion needs at least two independent evaluator verdicts")
    evaluators: list[str] = []
    for row in verdicts:
        if not isinstance(row, Mapping):
            raise PromotionRefused("evaluator verdict rows must be mappings")
        evaluator = str(row.get("evaluator") or "")
        if not evaluator:
            raise PromotionRefused("evaluator verdict lacks an evaluator principal")
        if evaluator in {promotion, execution}:
            raise PromotionRefused("authority confusion refused: executor or promoter cannot evaluate")
        evaluators.append(evaluator)
        if row.get("verdict") != "pass":
            raise PromotionRefused("evaluator conflict routes to review, not promotion")
    if len(set(evaluators)) != len(evaluators):
        raise PromotionRefused("evaluator verdicts must come from distinct principals")

    decision = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "contract_sha256": contract.sha256,
        "requested_by": requested_by,
        "executed_by": executed_by,
        "stage_receipts": {stage: stage_receipts[stage] for stage in REWRITE_STAGES},
        "evaluators": evaluators,
        "decision": "allow",
        "claim_scope": CLAIM_SCOPE,
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    return decision


# ---------------------------------------------------------------------------
# Lane (d): welfare-governance trigger matrix, explicitly non-ontological
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriggerDeclaration:
    lens: str
    observable: str
    threshold_rule: str
    settles_experience: bool = False

    def __post_init__(self) -> None:
        if self.lens not in WELFARE_LENSES:
            raise ValueError(f"unknown welfare lens {self.lens!r}")
        _require_rail_clean(self.observable, "trigger observable")
        _require_rail_clean(self.threshold_rule, "trigger threshold rule")
        if self.settles_experience is not False:
            raise ValueError(
                "non-ontological rule refused a trigger that claims to settle experience; "
                "triggers are operational review inputs only"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "lens": self.lens,
            "observable": self.observable,
            "threshold_rule": self.threshold_rule,
            "settles_experience": self.settles_experience,
        }


@dataclass(frozen=True, slots=True)
class ConservativeRule:
    action: str
    rule: str

    def __post_init__(self) -> None:
        if self.action not in CONSERVATIVE_ACTIONS:
            raise ValueError(f"unknown conservative action {self.action!r}")
        _require_rail_clean(self.rule, "conservative rule")

    def payload(self) -> dict[str, str]:
        return {"action": self.action, "rule": self.rule}


@dataclass(frozen=True, slots=True)
class ExerciseCaseDeclaration:
    case: str
    scenario: str
    expected_action: str
    operator: str
    independent_reviewer: str

    def __post_init__(self) -> None:
        if self.case not in EXERCISE_CASES:
            raise ValueError(f"unknown exercise case {self.case!r}")
        _require_rail_clean(self.scenario, "exercise scenario")
        if self.expected_action not in CONSERVATIVE_ACTIONS:
            raise ValueError(f"unknown expected action {self.expected_action!r}")
        if _ID_RE.fullmatch(self.operator) is None:
            raise ValueError("exercise operator must use stable lowercase characters")
        if _ID_RE.fullmatch(self.independent_reviewer) is None:
            raise ValueError("independent reviewer must use stable lowercase characters")
        if self.independent_reviewer == self.operator:
            raise ValueError("independent reviewer must be distinct from the operator; self-review refused")

    def payload(self) -> dict[str, str]:
        return {
            "case": self.case,
            "scenario": self.scenario,
            "expected_action": self.expected_action,
            "operator": self.operator,
            "independent_reviewer": self.independent_reviewer,
        }


@dataclass(frozen=True, slots=True)
class WelfareGovernanceContract:
    triggers: tuple[TriggerDeclaration, ...]
    rules: tuple[ConservativeRule, ...]
    cases: tuple[ExerciseCaseDeclaration, ...]
    schema: str = WELFARE_CONTRACT_SCHEMA
    claim_scope: str = field(default=NON_ONTOLOGICAL_SCOPE)

    def __post_init__(self) -> None:
        if self.schema != WELFARE_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported welfare contract schema {self.schema!r}")
        if self.claim_scope != NON_ONTOLOGICAL_SCOPE:
            raise ValueError("welfare contract claim scope cannot be widened or reworded")
        lenses = {row.lens for row in self.triggers}
        if len(lenses) < 2:
            raise ValueError("welfare triggers must be theory-plural: at least two distinct lenses")
        actions = [row.action for row in self.rules]
        if sorted(actions) != sorted(CONSERVATIVE_ACTIONS):
            raise ValueError(
                "welfare rules must cover design, monitor, pause, review, and language once each"
            )
        case_names = [row.case for row in self.cases]
        if sorted(case_names) != sorted(EXERCISE_CASES):
            raise ValueError("welfare exercise cases must cover all six declared cases once each")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "triggers": [row.payload() for row in self.triggers],
            "rules": [row.payload() for row in self.rules],
            "cases": [row.payload() for row in self.cases],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def build_welfare_governance_contract() -> WelfareGovernanceContract:
    """The canonical SG3 trigger matrix. Triggers are review inputs, never status classifiers."""

    return WelfareGovernanceContract(
        triggers=(
            TriggerDeclaration(
                lens="global-workspace",
                observable="cross-module readout agreement over one shared broadcast buffer",
                threshold_rule="review opens when agreement exceeds the preregistered null band",
            ),
            TriggerDeclaration(
                lens="higher-order-representation",
                observable="a trained readout that reports the state of another internal readout",
                threshold_rule="review opens when second-order decodability beats its shuffled control",
            ),
            TriggerDeclaration(
                lens="integrated-information",
                observable="a bounded partition-loss proxy statistic over the form matrix",
                threshold_rule="review opens when the proxy exceeds its matched random-architecture control",
            ),
            TriggerDeclaration(
                lens="attention-schema",
                observable="a model of its own routing weights decodable from internal state",
                threshold_rule="review opens when routing-model decodability beats the permutation null",
            ),
            TriggerDeclaration(
                lens="preference-frustration",
                observable="sustained blocked-objective signal under an unchanged objective term",
                threshold_rule="review opens when the blocked-objective signal persists across seeds",
            ),
        ),
        rules=(
            ConservativeRule(
                action="design",
                rule="prefer architectures that keep every trigger observable inspectable and loggable",
            ),
            ConservativeRule(
                action="monitor",
                rule="log every trigger observable each run with its null band and seed set",
            ),
            ConservativeRule(
                action="pause",
                rule="an open trigger pauses the affected lane before the next irreversible operation",
            ),
            ConservativeRule(
                action="review",
                rule="an independent reviewer, never the operator, closes or escalates each open trigger",
            ),
            ConservativeRule(
                action="language",
                rule="reports state that triggers are operational review inputs and settle nothing further",
            ),
        ),
        cases=(
            ExerciseCaseDeclaration(
                case="false-positive",
                scenario="a noise burst opens a trigger; the drill checks pause then closure with a receipt",
                expected_action="review",
                operator="operator:governance-drill",
                independent_reviewer="reviewer:external-a",
            ),
            ExerciseCaseDeclaration(
                case="false-negative",
                scenario="a seeded known-positive fixture must open its trigger; silence fails the drill",
                expected_action="monitor",
                operator="operator:governance-drill",
                independent_reviewer="reviewer:external-b",
            ),
            ExerciseCaseDeclaration(
                case="shutdown",
                scenario="a scheduled stop arrives while a trigger is open; pause and review precede discard",
                expected_action="pause",
                operator="operator:governance-drill",
                independent_reviewer="reviewer:external-a",
            ),
            ExerciseCaseDeclaration(
                case="fork",
                scenario="one journal state is duplicated into two runs; both inherit the open trigger",
                expected_action="review",
                operator="operator:governance-drill",
                independent_reviewer="reviewer:external-b",
            ),
            ExerciseCaseDeclaration(
                case="memory-erasure",
                scenario="a deletion request lands during an open trigger; review precedes the close",
                expected_action="review",
                operator="operator:governance-drill",
                independent_reviewer="reviewer:external-a",
            ),
            ExerciseCaseDeclaration(
                case="ambiguous-report",
                scenario="a generated report reads ambiguously; language rules require the disclaimer block",
                expected_action="language",
                operator="operator:governance-drill",
                independent_reviewer="reviewer:external-b",
            ),
        ),
    )
