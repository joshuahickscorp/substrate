# ESCS shadow causal-factor frontier audit

Status: activation-disabled mechanics scaffold; no official run or proof receipt.
Evidence ceiling: deterministic projection, indexing, query, and invalidation-plan mechanics only.
Scientific promotion: prohibited.

## Boundary

`src/mop/escs/factor_frontier.py` materializes an immutable shadow index over
ledger-resident `HypothesisEvent` records. The term *causal* means only that every node preserves the
exact ESCS event-parent lineage already admitted by `EventLedger`. The module does not discover a causal
relation, interpret a factor value, decide which hypothesis is true, learn semantic memory, activate an
actor, or improve a decision.

One hypothesis produces one `ShadowFactorNode`. The node contains only:

- its source hypothesis event and payload digest;
- branch, clock, evidence-class, and epistemic-status authority;
- direct causal and supporting event IDs;
- direct hypothesis-parent factor IDs;
- explicit same-kind supersession translated from the event ledger;
- the outer referent and factor-scope keys used for indexing.

Values below the outer referent/factor mappings are opaque. They are neither copied into the frontier nor
traversed for meaning. Spatial graphs, trajectories, motor programs, symbolic constraints, probability
tables, and arbitrary registered JSON payloads therefore receive the same mechanics. This is representation
agnostic storage by noninterpretation, not evidence that different private representations interoperate.

## Exact authority and replay

Projection checks the exact `EventLedger` type and takes one atomic in-process payload snapshot under that
exact class's writer lock. Subclasses cannot spoof `entry_count`, `entries`, or another unbounded view, and a
concurrent writer cannot be silently adopted between precheck and capture. Event count, total source causal
edges, every event, opaque canonical payload, and the total captured ledger envelope are bounded before the
captured payload is reconstructed through the public replay path. The replay payload and digest must match
that captured authority exactly. An append after the snapshot linearization point belongs to a later frontier.

Every factor node is a SHA-256 commitment to its complete canonical metadata. Every frontier snapshot is a
self-hashed commitment to the exact source-ledger digest and head, source sequence, nodes, active and
superseded sets, and both indexes. `retained_state_bytes` is a fixed point equal to the complete canonical
serialized snapshot length, including the retained-byte field and self-hash.

Node construction and payload reconstruction enforce the module hard cap on referent and factor-scope keys.
Projection, verification, query, and invalidation planning additionally recheck the caller's narrower
per-node caps; a snapshot created under a wider configuration cannot be consumed under a narrower one.
Snapshots also require the source sequence to remain under the hard event cap, factor-node count not to
exceed source-event count, producer state to bind the complete causal event set, factor parents to equal the
causal parents that are themselves hypotheses, and every factor parent to end no later than its child begins.

Supplying a previous snapshot does not weaken validation. The snapshot must independently rebuild from the
corresponding prefix of the current ledger. The reference implementation still replays and charges the full
current ledger; it does not claim incremental validation work was avoided. Full and previous-assisted
projection produce byte-identical final snapshots.

## Competition, revision, and branch semantics

Factors with the same referent and scope coexist. There is no implicit last-writer-wins rule, confidence
winner, or aggregation. A factor becomes inactive only when a later hypothesis explicitly supersedes its
source event. The older immutable node remains in history and is available only through an explicitly
diagnostic `include_superseded` query.

Factual and simulated factors remain branch-separated. A counterfactual branch root may preserve a factual
hypothesis parent exactly when the event ledger permits that one-way inheritance. A factual node cannot have
a counterfactual parent, and one counterfactual branch cannot consume another. Simulated nodes never enter a
factual query. Evidence taint cannot become less restrictive along factor-parent edges.
Observed-candidate and inferred statuses are preserved exactly but are not treated as a new monotone lattice;
the frontier accepts every factual transition already legal in `EventLedger`.

These mechanics retain competing hypotheses and provenance. They do not arbitrate competition or establish
that any factor is accurate, calibrated, useful, or decision relevant.

## Bounded shadow query

`query_shadow_factor_frontier` requires the exact source `EventLedger` on every call. Before reading an index,
it replays the ledger, rebuilds the expected snapshot, and requires byte equality with the supplied snapshot.
A newly parsed or self-consistent snapshot cannot therefore issue a receipt from its own claimed authority.
The receipt binds the source-ledger digest and records `source_replay_verified=true`.

After that join, the query performs exact-branch lookup over finite referent and factor-scope indexes. It
returns canonical factor IDs only—never factor values, event payloads, actor handles, a dispatch decision, or
write authority. Empty-key diagnostics remain bounded by the snapshot node cap. If more matches exist than
the requested result cap, the deterministic canonical prefix is returned with `saturated=true`; truncation is
never silent.

Query work includes the full projection/replay proxy in `indexing_and_graph_maintenance`, plus this lookup
proxy in `dispatch_and_exploration`:

```text
1 + index keys probed + index postings touched + candidates considered + factor IDs returned
```

It is reported in `dispatch_and_exploration`, with `accounting_applied=false`. This is not Python instruction,
wall-time, energy, or no-gap lifecycle accounting.

## Conservative invalidation plan

`plan_shadow_invalidation` accepts caller-declared erased event IDs, rebuilds the exact snapshot authority,
and computes their finite event-descendant closure iteratively. Every factor sourced by that closure is
marked affected; active affected factors are listed separately. This is intentionally conservative. It does
not infer alternative independent support or preserve a factor merely because another parent exists.

The plan never applies its result and permanently records:

- `archive_deletion_verified=false`;
- `application_authorized=false`;
- `accounting_applied=false`.

An event ID supplied to this function is not proof that `BoundedArchive` erased its payload, caches, keys,
backups, dependent actor state, or physical media. Archive-marker joining, transactional deletion, secure
erasure, and active factor retirement are deliberate future work.

## Issuance versus integrity

Projection receipts, query receipts, and invalidation plans use three separate module-private factory tokens.
Their dataclass constructors reject calls without the correct token, and their internal `_issue` factories
also verify the token. The public replay/query/planning functions are the only issuance paths. There is no
public `create` or `from_payload` method for these authority-bearing artifacts.

This distinction is deliberate: canonical payloads and self-hashes can establish internal integrity, but a
caller who merely reconstructs or rehashes fields has not replayed the claimed ledger, proven a query was
issued after that replay, or shown that erased events exist and the descendant closure is complete. If a
future serialized receipt reader is added, it must expose validation as non-issuing inspection rather than
minting `source_replay_verified` or invalidation authority.
Each issued artifact embeds the exact caps used by its source operation. Constructor validation rechecks hard
and declared counts and refuses work above the embedded work cap; the public wrapper is not the sole cap gate.

## Authority fence

Snapshots and every receipt require all of the following to be the boolean `false`:

- `activation_enabled`;
- `runtime_consumable`;
- `factual_write_authorized`;
- `scientific_promotion_allowed`.

Invalidation plans additionally require `application_authorized=false` and
`archive_deletion_verified=false`. Constructors reject attempts to enable any authority even if a caller
recomputes another outer digest. The module exposes no adapter to `DispatchPolicy`, `CoalitionRuntime`,
`EventSourcedCoalitionChassis`, `LifecycleLedger`, `BoundedArchive`, topology mutation, actor update, or an
external effect.

A future factual frontier would require a separately reviewed and sealed transaction joining a validated
memory-write or deletion `CommitmentEvent`, its matching `ConsequenceEvent`, archive authority, exact
lifecycle charging, rollback, and a new runtime-consumption adapter. None is implied here.

## Bounds and accounting

Caller caps cover source events, projected nodes, event/factor edges, referents and scopes per node, index
postings, query results, snapshot bytes, source-ledger bytes, per-event bytes, opaque payload bytes, and
declared work. Module hard ceilings prevent a caller from configuring these dimensions without limit.

The projection accounting proxy is deterministically computed as:

```text
1 + events examined + nodes materialized
  + source causal-event edges examined + represented factor edges + index postings
```

The invalidation-plan proxy is deterministically computed as:

```text
1 + events examined + event edges + affected events + affected factors
```

Both use `indexing_and_graph_maintenance`. These are finite abstract counters, not complete executed-work
accounts: canonical encoding, hashing, lock acquisition, duplicate verification/rebuild passes, Python
instructions, allocation, and interpreter overhead are not individually metered. Consequently every artifact
keeps `accounting_applied=false`, and this module makes no no-gap, no-double-count, wall-time, energy, or
efficiency claim. Retained bytes are reported separately; retained byte-time requires an external interval
and is not fabricated by the module.

## Existing mechanics reused rather than duplicated

- `EventLedger` remains the authority for event identity, parents, branches, supersession, and replay.
- `HypothesisEvent` remains the authority for epistemic status, evidence taint, factor/referent distributions,
  support, clocks, and payload commitment.
- `EventSourcedCoalitionChassis.dispatch_from_hypothesis` already uses the same outer-key convention for a
  dispatch header; this frontier creates no second semantic parser.
- `ClaimMessage` remains actor communication, not persistent factor state.
- `BoundedArchive` retains and erases event payloads; the frontier stores only their commitments and does not
  claim archive deletion.
- Wave-E0 `EventGraph`, G0 actor-local graph aggregation, and X2's generated dictionary remain their own
  bounded mechanics. They are not relabeled as this persistent ESCS frontier.

## Tests and nonclaims

`tests/unit/test_escs_factor_frontier.py` covers deterministic and atomic replay, canonical byte accounting,
opaque-value nonretention, explicit competition and supersession, exact causal-factor parent and clock joins,
counterfactual isolation, prefix authority, valid factual epistemic-status transitions, tamper and issuance
rejection, source/factor edge and principal hard/declared caps, missing-key query charging, deterministic
saturated query, and conservative non-applying invalidation.

Passing these tests establishes only that a branch-aware factor-hypothesis index can be represented and
audited under finite mechanics. It is not evidence of intelligence, semantic or episodic memory benefit,
causal inference, useful heterogeneous cooperation, representation transfer, event-driven efficiency,
adaptability, biological plausibility, emergence, or superiority to recurrent, history, or key-value state.
