# ESCS mechanics chassis: implementation boundary

Status: integrated mechanics scaffold; no capability or efficiency result
Architecture source: `docs/mixture_of_perspectives/18_event_sourced_coalition_substrate.md`

## Purpose

The current `mop.escs` package implements deterministic boundaries needed to test the Event-Sourced
Coalition Substrate (ESCS). It makes raw admission, events, work, claims, selective activation,
endogenous reasoning, commitments, consequences, and retention inspectable and bounded. It does not
implement an intelligent system, choose meaningful events, learn useful coalitions, or establish that
ESCS is better than any alternative.

The safest description is **an evidence-neutral experimental chassis**. It can host controlled studies while
preventing several kinds of hidden computation, identity drift, epistemic laundering, and unbounded runtime
behavior. The architecture document remains the research proposal; this file describes only what the code
currently enforces.

## Implemented mechanics

- Immutable, content-addressed observation, hypothesis, commitment, and consequence records with canonical
  serialization and exact-schema parsing.
- Explicit causal parents, counterfactual branches, producer-state versions that commit to the complete
  canonical parent set, payload commitments, supersession lineage, creation costs, and clock intervals.
- An append-only, hash-linked event ledger that rejects missing parents, invalid event-kind transitions,
  unauthorized branch crossings, invalid supersession, and consequences not bound to their commitment.
- An append-only, hash-linked lifecycle ledger with separate counters for each declared work category and
  retained byte-time.
- A charged raw event-former boundary with immutable packets, bounded typed proposals or explicit
  abstention, nonzero no-packet polling, frozen policy-state checks, transitive evidence taint, and atomic
  observation/hypothesis publication.
- Typed, content-addressed claim messages whose payload remains opaque bytes; receiver-side validation is
  fail-closed on schema, integrity, provenance, branch, referent, producer state, freshness, and epistemic
  status.
- An actor protocol that separates public routing metadata and header-only readiness from payload-bearing
  activation and consequence-driven updates.
- A bounded coalition runtime with central, sharded-subscription, and peer-nomination candidate modes; hard
  candidate, coalition, beam, hop, message-edge, endogenous-round, fanout, and queue caps; explicit traces;
  an injected deterministic policy suitable for mechanics tests; ledger-resident content-addressed
  endogenous hypotheses; full trace commitments; and exact half-open retained-state accounting.
- A single-writer chassis join that derives dispatch only from a ledger-resident HypothesisEvent, records a
  CommitmentEvent before an external callback, records a matching ConsequenceEvent before actor updates,
  validates every mechanically reconstructable restart authority binding, and preserves explicit
  partial-stage status after failures.
- A bounded hot journal, content-addressed cold segments, compaction commitments, bounded retrieval cache,
  logical payload erasure, and explicit loss of exact replay authority after erasure.
- A content-addressed 31-slot perspective registry and quiescent assembly that admit heterogeneous
  candidates cheaply while keeping activation and scientific promotion under separate, hard-false
  authorities.
- A frozen, activation-disabled topology grammar plus bounded genotype validation and deterministic
  counterfactual reference semantics for its eight declared operators. These records test construction and
  replay mechanics only; they do not enact topology changes or establish useful computation.
- Two activation-disabled observers: a P9-derived causal/resource monitor that emits inert claims, and a
  scripted, oracle-tainted coalition fixture that checks removal arithmetic and shadow ranking without
  granting dispatch, effect, update, cooperation, or causal-credit authority.
- An activation-disabled shadow causal-factor frontier that projects ledger-resident hypotheses into a
  branch-separated, immutable, bounded index. It retains only outer referent/factor keys and event lineage,
  requires exact ledger replay for queries, and can emit only non-applying invalidation plans.

These pieces are mechanics. The package contains no learned event former, learned dispatcher, cognitive
actor library, learned or decision-validated causal factor frontier, causal counterfactual-credit learner,
or empirically validated structural-plasticity mechanism.

## Lifecycle and dataflow

The complete proposed lifecycle and the current implementation boundary are:

```text
raw stream
  -> source adapter creates charged RawPacket             [source adapter remains caller-audited]
  -> bounded policy proposes or abstains                  [event_former.py]
  -> ObservationEvent + HypothesisEvent atomic batch      [event_former.py, events.py, ledger.py]
  -> ledger-authoritative event-to-dispatch adapter       [chassis.py]
  -> bounded candidate lookup                            [runtime.py]
  -> header-only actor readiness + injected policy       [actors.py, runtime.py]
  -> selected actors receive payload                     [runtime.py]
  -> typed claims / action intents / endogenous hypotheses [messages.py, actors.py, runtime.py]
  -> endogenous HypothesisEvent append before redispatch [runtime.py, ledger.py]
  -> CommitmentEvent before callback                     [chassis.py]
  -> external effect callback or persistent abstention   [injected adapter]
  -> ConsequenceEvent before actor update                [chassis.py]
  -> update only actors active in the issuing trace       [runtime.py]

every charged runtime operation -> LifecycleLedger       [accounting.py]
event envelopes + replayable payloads -> BoundedArchive  [archive.py; caller still wires it]
```

`ChargedEventFormer` owns the raw-packet-to-hypothesis join. `EventSourcedCoalitionChassis` owns the
hypothesis-to-dispatch-to-commitment-to-consequence join. They share authoritative ledgers but are not one
distributed transaction. Archive publication, archive-accounting reconciliation, source-adapter isolation,
and deletion commitment/consequence orchestration remain explicit caller boundaries. A persisted
commitment is an at-most-once invocation fence: a crash after its append but before the callback can omit
the effect. Exactly-once external effects require an idempotent or transactional external adapter.

## Public module responsibilities

| Module | Responsibility | Boundary |
| --- | --- | --- |
| `mop.escs` | Export the stable event/accounting core plus charged event-former and chassis joins. | Importing the API implies no capability, learning, or efficiency result. |
| `mop.escs.events` | Construct and exactly parse the four immutable event kinds and their common envelope. | Validates individual records; it is not an event store, causal-graph engine, event former, or lifecycle orchestrator. |
| `mop.escs.ledger` | Append and replay four-stage events while validating parent existence, time, branch, supersession, and allowed kind transitions. | It is an event-history graph, not the proposed learned causal factor frontier, archive, dispatcher, or transaction coordinator. |
| `mop.escs.accounting` | Represent exact nonnegative work vectors and append/replay hash-linked charges. | Abstract counters are not FLOPs, energy, latency, or proof that all caller work was charged. |
| `mop.escs.event_former` | Validate charged raw packets, freeze injected policy state, bound proposals and idle polling, join evidence taint, and atomically publish observation/hypothesis batches. | Policy construction/training and packet-source semantics are outside this boundary; accepted policies are injected and evaluation is state-pure. |
| `mop.escs.messages` | Define finite claim schemas, content-addressed opaque messages, and fail-closed receiver validation. | Schema validity does not show that a claim is true, useful, representation-independent, or understood. |
| `mop.escs.actors` | Define dispatch, readiness, activation, action, typed endogenous-hypothesis, and update contracts without a public local-state accessor. | These are interfaces and data objects, not useful actor implementations or learned specializations. |
| `mop.escs.runtime` | Retrieve bounded candidates, enforce caps before proportional scans, activate selected actors, append actor-originated hypotheses, validate outputs, fully seal traces, charge runtime work/retention, and update prior active actors once. | The policy is injected. The runtime does not learn value of computation, coalition credit, event admission, or actor structure. |
| `mop.escs.chassis` | Derive dispatch from a ledger hypothesis, bind full trace/action authority, append commitment before callback and consequence before update, and fail closed on restart/failpoint inconsistencies. | It is a single-writer in-process join, not durable storage, a distributed transaction, an archive bridge, or an exactly-once effect system. |
| `mop.escs.archive` | Retain immutable envelope lineage separately from payloads; bound hot/cache state; compact, retrieve, audit, and logically erase payload segments. | Cold storage is not globally size-bounded, the in-memory store is not durable, and logical non-retrievability is not secure physical-media deletion. |
| `mop.escs.perspective_registry`, `mop.escs.substrate_assembly`, `mop.escs.substrate_preflight` | Bind every requested perspective to an exact quiescent slot and verify the consolidated scaffold authorities. | Inclusion is not activation, efficacy, compatibility, or scientific promotion. |
| `mop.escs.topology_grammar`, `mop.escs.g0_genotype`, `mop.escs.g0_evaluator` | Freeze a finite construction language, validate bounded actor genotypes, and replay deterministic counterfactual operator semantics. | The grammar and evaluator remain activation-disabled reference mechanics; they do not search, install, train, or promote an actor. |
| `mop.escs.causal_resource_monitor` | Observe exact event/lifecycle snapshots and emit inert resource-anomaly or same-parent simulated-contrast claims. | It cannot trigger work or establish a realized causal effect; its declared accounting is not applied to a trusted ledger. |
| `mop.escs.coalition_evidence` | Replay scripted actor-removal fixtures, decompose utility terms, and rank bounded coalitions in a nonconsumable shadow plane. | Fixture utilities are oracle-tainted and noncausal; the module grants no runtime, cooperation, action, effect, update, activation, or promotion authority. |
| `mop.escs.factor_frontier` | Project ledger hypotheses into a replay-verified, branch-aware index with explicit coexistence, supersession, bounded ID-only queries, and conservative invalidation planning. | “Causal” means preserved event lineage only; factor values stay opaque, every result is nonconsumable, and no truth, memory benefit, factual write, or causal discovery is established. |

## Minimal stable core use

The package root exposes the event/accounting core and the two integrated joins. A root observation can be
constructed and causally admitted without invoking those higher-level mechanics as follows:

```python
from mop.escs import EventLedger, ObservationEvent, WorkVector

observation = ObservationEvent.create(
    raw_packet_or_delta_refs=("packet:camera/0001",),
    adapter_version="camera-v1",
    sensor_scope={"sensor": "camera-0"},
    transport_and_detection_cost=WorkVector(raw_transport_and_adapters=4),
    clock_start_tick=0,
    clock_end_tick=0,
    source_and_provenance={"source": "sensor:camera/0"},
    measured_creation_cost=WorkVector(event_formation=1),
)
ledger = EventLedger()
ledger.append(observation)
assert ledger.verify() == []
```

This demonstrates immutable construction and causal admission only. It does not form a meaningful event,
dispatch a coalition, archive the payload, or produce a decision.

## Deterministic mechanics smoke

The complete scripted join can be exercised without a heavy experiment:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_escs_mechanics_chassis.py
PYTHONPATH=src .venv/bin/python scripts/run_escs_mechanics_chassis.py --verify-only
```

The config is `configs/experiment/escs_mechanics_chassis.json`; the default receipt is
`proof/ESCS_MECHANICS_CHASSIS.json`. It covers one no-packet poll, raw admission, the four event stages,
selective activation, commitment-before-effect, consequence-before-update, final retention, archive
publication, and exact receipt integrity. Every record remains `scripted-mechanics-only`; a green receipt is
an integration result, not empirical support for ESCS capability or efficiency.

## Invariants and accounting boundary

The chassis is intended to preserve these mechanical invariants:

1. Event, message, action, charge, segment, and snapshot identities are deterministic commitments to their
   canonical content. Tampering must be rejected rather than repaired silently.
2. Events are immutable. A revision creates another event, names causal parents, and may name superseded
   events; it does not mutate an earlier record. Parentless events use the explicit genesis state version;
   every derived event's producer-state version commits to its complete sorted parent set. The event ledger
   requires known parents and revalidates the state version, declared event-kind transition, and branch rules
   on append and replay.
3. Observation events remain factual. Simulated hypotheses and runtime outputs cannot silently enter the
   factual branch. `learned-unverified`, `scripted-mechanics-only`, and `oracle-nonpromotable` evidence taint
   is typed and transitive across packets, parents, events, messages, actions, endogenous reasoning,
   commitments, and consequences; a child cannot relabel a more restrictive ancestor downward.
4. The code rejects its declared evaluator-truth and future-information fields. This is a schema guard, not
   a proof that arbitrary payload semantics contain no leakage.
5. Within the runtime call boundary, policies and readiness calls receive headers rather than event payloads.
   Only selected actors receive the payload, actor-originated reasoning becomes a content-addressed
   HypothesisEvent before redispatch, and only actors active in an action-authorized trace receive its
   consequence update. The Python protocol cannot prevent undeclared external state; experimental actors
   require source inspection and adversarial isolation.
6. `K`, `C`, `B`, `H`, `M`, `R`, action, payload, header, pending-authority, retained-state, polling,
   proposal, parent, fanout, queue, shard, and nomination limits are hard declared caps. Runtime checks caps
   before proportional beam/fanout scans. Bounded execution does not by itself establish favorable scaling.
7. Claims are accepted only when the receiver can validate every required identity and scope. Unknown schema,
   stale producer state, expiry, corruption, wrong branch, unsupported referent, or epistemic laundering
   causes rejection.
8. Hot payload residency and the retrieval cache are byte-bounded; hot residency is also tick-bounded.
   Immutable lineage and cold payload segments are not globally size-bounded and cold payloads remain until
   explicit erasure, so total archive growth is still an experimental and operational concern.
9. Logical erasure removes the archive's retrievable payload state and permanently disables exact replay
   authority. It preserves non-content-bearing lineage and makes no claim about disks, backups, allocator
   slack, or third-party copies.
10. Lifecycle charges are append-only and replayable. Retained byte-time remains a separate dimension and is
    not added to operation-like `total_work`. Runtime retention uses one half-open frontier and requires
    explicit `finalize`; event-former retention can begin at an explicit deployment tick. Runtime authority
    binds the initial actor-state versions and the injected dispatch policy's static authority and initial
    state version; trace hashes establish integrity, not signer authenticity.
11. A factual external action is never invoked before its CommitmentEvent is admitted. Actor learning is
    never invoked before a matching ConsequenceEvent is admitted and all hypothesis/trace/action/effect/
    branch/evidence authority fields are recomputed. This is ordered fail-closed mechanics, not atomicity
    across an external system.

The authoritative operation boundary represented by `WorkVector` is:

```text
raw_transport_and_adapters
+ event_formation
+ indexing_and_graph_maintenance
+ dispatch_and_exploration
+ actor_execution
+ messages
+ counterfactual_credit
+ learning
+ archival_and_erasure
+ idle_floor
```

`retained_byte_time` is reported separately. An empty-queue runtime check incurs nonzero idle work. Archive
retention, compaction, retrieval, and erasure observations are exposed through an accounting hook, but the
caller must map them into the authoritative ledger exactly once. Event-former, runtime, endogenous-event,
dispatch-adaptation, commitment, consequence, effect-attempt, and actor-update mechanics already charge the
shared LifecycleLedger. Event-local cost fields and archive observations still need an experiment-level
no-gap/no-double-count reconciliation. Over-cap event-former failure receipts are explicitly named
*saturated rejection charges* and make that run invalid for exact-work or promotion claims; hard Python-call
preemption is not claimed. Contract-invalid chassis history can fail during authority preflight before a
trustworthy nonretrograde charge tick is available, so those rejected calls are outside the exact-work claim.
Training amortization, raw-packet construction, wall time, peak memory, and energy remain additional measured
endpoints rather than inferred properties of these counters.

The supplied ledgers and archive store are in-memory mechanics. Serializing and replaying EventLedger
preserves the invocation fence, but a newly constructed runtime does not reconstruct an old trace's pending
actor-update authority; it returns `consequence-recorded-update-unavailable` instead of guessing or applying
the update twice. A restart must inject the bounded tuple of previously trusted runtime authority IDs from a
separate durable authority record; the chassis never promotes a runtime ID merely because it appears in its
EventLedger. Production durability, transactional-outbox recovery, and archive/event atomicity remain future
systems work. A replayed commitment preserves a self-consistent runtime/sequence trace authority and
full-trace digest, but EventLedger alone does not retain the complete historical RuntimeTrace needed to prove
that a foreign trace was actually issued; experiments must archive and verify full traces separately.

## Experiment-gated mechanisms

The following remain hypotheses and must stay replaceable behind the chassis:

- learned raw event formation, sparse useful-event admission, and a genuinely low idle floor;
- header-based learned dispatch and calibrated value-of-computation estimates;
- discovery of complementary coalitions, interaction credit, exploration, and resilience credit;
- a causal factor frontier that improves prospective decisions rather than only provenance;
- useful heterogeneous actors, communication conventions, and representation transfer;
- adaptive-depth reasoning that beats equal-work and rate-matched controls;
- bounded archival state that preserves decision value under compaction, revision, poisoning, and deletion;
- actor birth, merge, pruning, repair, or novel functional roles within a frozen finite `G0` grammar;
- favorable quality, robustness, sample-efficiency, dormant-population, event-storm, and lifecycle-work
  scaling; and
- any integrated ESCS Pareto advantage over reactive, recurrent, history, memory, or fixed-coalition controls.

EDCM-1 can test structured-observation complementarity and dispatch/message mechanics. Experiment 0 owns raw
event formation and idle cost; Experiment 1 owns learned counterfactual dispatch; Experiment 2 owns causal
event state versus vector/history controls; Experiment 3 owns structural plasticity and is blocked until its
entry gate and frozen `G0` exist.

## Explicit nonclaims

This chassis is not evidence of intelligence, understanding, consciousness, generality, biological
equivalence, developmental learning, emergence, decentralization, an ecosystem or society, semantic truth,
learned event formation, useful representation-neutral communication, computational superiority, or energy
efficiency. Canonical records are not faithful explanations without causal interventions. Bounded sparse
activation is not an efficiency win until the complete control plane and lifecycle are charged. A successful
unit test is a mechanics result, never an empirical architecture result.

## Validation expectations

Before describing a revision as a mechanically usable chassis, validation should include:

- focused unit tests for event round-trip and tamper rejection, event-transition validation, event- and
  work-ledger replay and chain corruption, message integrity and laundering, runtime cap enforcement and
  inactive-actor isolation, archive compaction and corruption, logical erasure, and replay-authority loss;
- adversarial tests for future/evaluator-field injection, omitted or replaced parents, wrong branches, stale
  state versions, expired or corrupted claims, undeclared peers, oversized beams/coalitions/queues, repeated
  endogenous states, residual payload indices/caches, duplicate or forged commitment/consequence authority,
  evidence-taint downgrade, trace mutation, and double consequence application;
- deterministic replay or canonical equality wherever the API promises it;
- lint, formatting, type/import, and bytecode-compilation checks on the package and focused tests; and
- exact lifecycle-account reconciliation in every experiment, including errors, rejected work, idle checks,
  archive work, retained byte-time, and failed trials.

Those checks establish only that the chassis enforces its declared mechanics. Capability, efficiency,
robustness, interpretation, and emergence claims require preregistered controls, paired seeds, held-out world
families, decisive nulls, and independent verification defined by the architecture document.
