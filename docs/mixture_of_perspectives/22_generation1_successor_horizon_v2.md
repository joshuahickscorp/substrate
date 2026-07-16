# Generation 1 Successor Horizon V2 Extension

> **Append-only extension:** This document adds a second bounded robustness horizon after
> [21_generation1_successor_evidence_chain.md](./21_generation1_successor_evidence_chain.md).
> It does not alter the live v4 adopter, the sealed v1 horizon, or any completed predecessor
> receipt. The categorized append-only successor is
> [23_generation1_categorized_batch_wave.md](./23_generation1_categorized_batch_wave.md).

**Status:** scaffolded as a post-v1 child; the extension parent may wait without launching compute
**Snapshot date:** 2026-07-16
**Waiting parent:** `generation1-successor-extension-chain-v1`
**Bounded child program:** `generation1-successor-horizon-v2`
**Idempotent entry command:**
`.venv/bin/python scripts/mop_generation1_successor_future_chain.py start --execute`
**Claim scope:** append-only same-code robustness and independent artifact verification; no runtime
activation, scientific promotion, independent scientific confirmation, natural-world generality, or
Stage 3 claim

The first successor horizon already provides a maximum of 234.43 serial compute-hours, or 29.30
ideal hours at eight workers. V2 adds one more finite five-epoch envelope of the same maximum size.
Across both horizons, the registered ceiling is 468.85 serial compute-hours, or 58.61 ideal
eight-worker hours. Frozen pruning may honestly shorten either program. Work is never revived merely
to consume a time budget.

## 1. Why this is a separate child

The live v4 parent binds the exact v1 horizon manifest, and the v1 manifest binds its runtime source
and throttle authorities. Rewriting those files in place would invalidate a live authority before
the first horizon launches. V2 therefore uses new files, a new program identity, a new run root, and
a separate waiting parent.

The extension parent is observation-only with respect to v4. It does not signal, suspend, restart,
or relabel v4 or any legacy queue. A claimed v4 completion is accepted only when the sibling sealed
state and status form one stable exact projection, current v4 and inherited implementation
authorities match, the three canonical legacy results pass their domain validators, and the complete
zero-injection v1 supervisor inventory and current artifacts replay without reconciliation. The
sealed state name, capsule rows, and completion counts must also describe one possible progression;
an all-complete inventory cannot be relabeled as waiting, and a downstream horizon row remains
pristine until its predecessor boundary is complete. The reader then confirms the same state/status
projection still exists, replays the artifacts a second time, confirms the projection once more,
and performs one final independent artifact-hash check. Immediately before any v2 process creation,
the extension takes the cooperative v4 lifetime lock, replays v4, independently rebuilds and
validates the v2 admission boundary, and replays v4 once more to prove it did not change across the
v1 check. The v2 status must bind the exact current generic-supervisor implementation and a positive,
finite PID/create-time identity. After persisting the sealed v2 launch intent, the complete gate is
repeated while the same lock remains held through the idempotent supervisor start.

The one-command launcher starts or resumes v4 first, then starts or resumes this lightweight
extension waiter. Both underlying starts are independently locked and idempotent, so repeating the
command cannot duplicate a legacy queue, adopter, waiter, or horizon supervisor. An exact visible
parent whose sealed status acknowledgement is still pending is retried through the same idempotent
start path, but the originally observed PID and creation time remain pinned across acknowledgement
attempts. A sealed live PID without one matching exact parent process is refused, and internally
sealed transient launch-intent snapshots are not accepted as startup acknowledgements. The launcher
reports success only after both components return self-sealed,
execution-enabled, empty-problem acknowledgements that pass each component's exact field, authority,
capsule, count, timestamp, progression, and completion validator. A reported v4 or extension
completion must also match its current durable state/status and terminal-artifact replay
byte-for-byte. The launcher fails closed on missing, stale, fabricated, replaced, unknown,
safety-drifted, failure, integrity-hold, or drained acknowledgements.

## 2. Admission boundary

V2 cannot start from partial v1 progress. Its admission receipt must validate and bind all six
terminal authorities:

1. the exact sealed `generation1-successor-horizon-v1` program manifest;
2. the generic v1 supervisor's clean, execution-enabled, zero-injection terminal status;
3. `proof/GENERATION1_SUCCESSOR_HORIZON.json`;
4. `proof/GENERATION1_SUCCESSOR_HORIZON.verification.json`;
5. the v1 generated report receipt;
6. the exact H05 classification and its ancestry.

The terminal status must contain the manifest's exact capsule inventory, with every capsule complete
and every reported artifact still matching its current bytes. Admission also rebuilds the complete
v1 verification from the bound result and raw predecessor artifacts and requires exact semantic
equality with the supplied verification. A mutually consistent set of re-sealed result,
verification, report, and H05 shells is therefore insufficient authority.

The H05 D1 route is inherited exactly. A pruned or stable-null D1 branch remains pruned. At the v2
admission boundary, the H05 mechanics survivor list is also inherited, then reduced by transitive
dependency closure. A mechanics lane enters H06 only when every declared dependency that is itself
a mechanics lane also survived H05. This can prune an integrated lane after a prerequisite lane
failed in v1, but it can never resurrect a lane removed by v1.

Dependencies on the already verified C0, C1, and C2 authorities remain satisfied by the predecessor
chain. Dependency closure applies only to the explicit successor-mechanics lane inventory.

## 3. Bounded work plan

V2 adds five ordered epochs, H06 through H10, using absolute fresh cycles 7 through 11. Those cycles
follow consolidated cycles 0 and 1 and v1 cycles 2 through 6. Their D1 and mechanics seed intervals
are disjoint from the predecessor programs by construction.

| Component | Per epoch | Five-epoch maximum |
| --- | ---: | ---: |
| D1 fresh rungs | 576 | 2,880 |
| Successor-mechanics fresh rungs | 2,380 | 11,900 |
| Raw work items | 2,956 | 14,780 |
| Planned serial compute | 46.89 hours | 234.43 hours |
| Ideal wall time at eight workers | 5.86 hours | 29.30 hours |

Each epoch contains five D1 shards, eight mechanics shards, and one classification barrier. The full
manifest contains 74 capsules: one admission capsule, 65 compute shards, five classifiers, one
aggregate, one separately authored verifier, and one report receipt. Every compute shard remains
below the 285-minute capsule boundary even when it downshifts to one worker.

## 4. Concurrency and host safety

The current reviewed host window does not authorize another physical compute lane while incumbent
successor workloads occupy the envelope. Consequently, launching the extension command starts only
a lightweight waiter.

When v2 is eventually admitted, the generic supervisor still runs one exclusive capsule at a time.
Inside a compute capsule, the frozen runner may use:

- eight workers when the host is otherwise idle and all admission gates pass;
- one worker while an exact reviewed Hawking profile is active;
- zero workers while CPU, memory, swap, thermal, disk, process identity, or coexistence authority
  gates refuse admission.

This is the maximum reviewed parallelism. Multiple top-level horizon supervisors are not authorized
by the current control plane. The extension accepts only the exact 74 base capsules from the sealed
manifest and requires the generic queue to remain at its zero-injection genesis. A re-sealed status
with extra injections, a missing capsule inventory, an active capsule, or an outstanding lane
reservation cannot claim completion.

## 5. Routing inside V2

Each epoch seals a classification before the next begins.

| Observation | Later action |
| --- | --- |
| D1 remains a stable candidate or is mixed and seed-sensitive | Continue D1 on the next fresh cycle. |
| D1 becomes a stable null | Prune all later D1 work. |
| A mechanics lane admitted through the H05 dependency closure remains clean | Continue that lane. |
| A mechanics lane warns or fails inside v2 | Prune that lane from later v2 epochs. |
| A receipt, source, seal, or verifier binding drifts | Hold the program; do not count partial work. |

Completed work is immutable. Routing changes only future eligible work, and every prune remains
visible in the aggregate.

## 6. Aggregate and verification boundary

After H10, the aggregate binds all epoch classifications, shard receipts, raw artifacts, pruning
decisions, and seed authorities. The v2 verifier reuses the separately authored streaming verifier
family against the new program identity and independently checks the v2 admission boundary, exact
predecessor manifest and terminal supervisor authority, replayed predecessor verification, routing
inventory, raw receipts, seed disjointness, aggregation, and semantic mutation rejection. Its claim
scope is an exact sealed field and cannot be widened into activation or scientific-confirmation
authority.

This is independent artifact verification. It is not a separately implemented scientific generator,
so `independent_scientific_confirmation` remains false.

## 7. Completion condition

The extension is complete only when:

- v4 and horizon v1 are terminal and clean;
- H06 through H10 each seal every eligible shard or frozen prune;
- the v2 aggregate is complete;
- the v2 independent artifact verifier is clean;
- the generated report receipt binds the exact result and verification;
- the generic supervisor status still contains the exact base capsule inventory, zero accepted
  injections, and current clean artifact reports for every completed capsule;
- every artifact still declares `activation_allowed: false` and
  `scientific_promotion: false`.

Until then, the exact description is: **the append-only v2 horizon is queued behind the current
successor evidence chain**.
