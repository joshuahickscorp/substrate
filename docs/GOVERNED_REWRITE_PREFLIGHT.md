# Governed rewrite transaction preflight

## What existed and what did not

The project already owned most of the ingredients, but they answered different questions.

- F8 owns the scientific learning question. It can run a plastic rewrite, frozen inherited,
  larger-shell, and random-initialization controls under a declared compute estimator. It hashes its
  evidence package and fails closed when natural data, rights, encoder, seed-plan, or shell-failure
  authority is missing. Its current local preflight proves smoke mechanics and correctly leaves the
  scientific null unevaluated.
- F20 owns prospective insufficiency detection. It compares a crisis score with raw error,
  confidence, random, and aleatoric-noise controls. Its durable structured-fixture result remains a
  null because one seed changes sign and measured avoided compute is zero.
- `FrozenJSON`, `canonical_sha256`, and `LifecycleJournal` already supply immutable identity,
  hash-linked replay, availability, conflict, poisoning, and rollback semantics.
- The local execution throttle owns resource admission, pausing, termination, and atomic resume
  authority for long-running processes. It does not authorize a candidate state rewrite.
- The provenance helper stamps environments and evidence scope. It is not an authorization system.

The missing seam was therefore narrow but important: no project-owned surface joined a rewrite
proposal, exact base identity, single-use scoped authority, independent canary and shadow evaluation,
atomic commit or rollback, and interrupted recovery into one transaction.

## Transaction design

The preflight changes one field only: `policy.adaptation_threshold`, from 3 to 2. The authority token
is bound to the exact proposal, source bundle, config payload, base state, evaluator, lifecycle memory
journal, issue and expiry ticks, and three allowed bookkeeping paths. The token is valid once. The
project-owned HMAC key is deliberately a fixture key embedded in source. It tests binding and replay
mechanics, not production cryptographic secrecy.

The evaluator has two disjoint structural-fixture surfaces:

1. An eight-case canary catches immediate capability or protected-route regressions.
2. A twelve-case shadow set tests the same candidate before commit with separate case identities.

Both surfaces record per-case predictions, targets, correctness, immutable row digests, before and
after capability, total regressions, and protected-route regressions. The canonical proposal must gain
one correct canary case and three correct shadow cases while producing zero regressions. A separately
authorized threshold-zero proposal is intentionally too broad. It fails the canary and proves rollback
without relying on an invalid token.

The evaluator reads a calibration offset from the existing lifecycle journal. A poisoned journal is
unavailable by lifecycle semantics and the transaction refuses it before evaluation.

## Interruption and atomicity drill

The canonical transaction executes its canary, atomically writes a content-addressed checkpoint,
then receives a planned interrupt before shadow evaluation. It also leaves a truncated
`state.json.tmp`, modeling a crash during a future state write. The committed `state.json` remains the
base state.

Resume requires exact source, config, proposal, authority, evaluator, memory, base-state, candidate,
canary, checkpoint, and audit-chain identity. Recovery verifies and removes the orphan partial file,
rebuilds the candidate and canary exactly, runs the shadow, consumes the token, and commits through a
same-directory temporary plus `os.replace`. The final state and both evaluation payloads must equal an
uninterrupted reference transaction.

## Adversarial and verifier coverage

Eight live transaction drills are required:

- forged authority signature
- expired authority
- replayed authority after commit
- scope escalation even when re-signed by the fixture authority
- conflicting committed base state
- evaluator rule tampering
- poisoned lifecycle memory
- an authorized proposal that fails the canary and rolls back

Every refused or rolled-back drill must leave the committed state hash unchanged. A separate verifier
reconstructs the clean journal, initial state, proposal, token, canary, shadow, committed state, and
audit semantics instead of trusting transaction status booleans. Fourteen sealed mutations target
source, base state, proposal, signature, scope, evaluator, memory, canary, shadow, commit state, audit
semantics, atomicity, resume provenance, and the claim boundary. All must be rejected.

## Resource and evidence boundary

The run is one-thread CPU, contains no model weights, performs no downloads, reads no external data,
and must stay under 10 wall seconds and 1 GiB process maximum RSS. It is deterministic structural
mechanics only.

This does not establish that a learned rewrite is useful, that an evaluator is externally independent,
that the embedded authority key is secure, or that the transaction is safe under real concurrent
processes or power loss. Promotion requires an externally governed production authority root,
independently specified evaluators, rights-clean natural workloads, preregistered protected
capabilities, real candidate artifacts, multi-process durability tests, and replicated post-commit
surveillance. A larger machine is not the current blocker for this transaction surface.

## Reproduce and verify

```bash
.venv/bin/python scripts/governed_rewrite_preflight.py
.venv/bin/python scripts/governed_rewrite_preflight.py \
  --verify proof/GOVERNED_REWRITE_PREFLIGHT.json
.venv/bin/pytest -q tests/unit/test_governed_rewrite.py
```
