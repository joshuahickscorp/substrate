# ESCS scripted coalition-fixture and shadow-arbitration audit

Status: finite v1 mechanics; scripted and noncausal; runtime consumption, activation, action-effect
authority, cooperation claims, and scientific promotion prohibited.

## Decision

`src/mop/escs/coalition_evidence.py` is a nonconsumable shadow instrument for checking coalition
arithmetic and arbitration mechanics. It does not provide causal or consequence-grounded evidence.
The only admitted utilities are explicit scripted fixture values attached after native chassis
abstentions. An abstention cannot cause arbitrary utility, so all resulting records retain
`ORACLE_NONPROMOTABLE` taint and hard false causal-claim flags.

The module cannot select or activate actors, create a `DispatchDecision`, commit an action, stage an
update, invoke an effect, charge a trusted ledger, or convert its proposal into runtime authority.

## Exact fixture authority

Each fork uses native immutable `HypothesisEvent`, `RuntimeTrace`, `CommitmentEvent`,
`ConsequenceEvent`, `EventLedger`, `LifecycleLedger`, and `WorkVector` records. The common contract
binds:

- world, horizon, source state, and environment state;
- runtime identity plus the exact trace mode/caps frame;
- policy-state and pre-intervention actor-state digests;
- the complete registered actor set;
- intervention schema and scripted fixture identities;
- an integer utility key, bounds, cost-exclusion rule, and self-derived scalarizer digest;
- `utility_source=scripted-noncausal-fixture`;
- hard-false causal-effect and consequence-grounded-credit claims.

The source-state, environment-state, policy-state, and actor-state digests are fixture declarations
compared for equality across branches. They are not joins to independently supplied or replayed
native state payloads. The trace mode/caps frame and registry/assembly/binding records receive the
stronger joins described below; the declared state digests do not.

Every branch also carries a self-hashed actor-removal intervention. It binds the branch, registered,
active, and removed actors, scripted utility, realized-cost vector, observation uncertainty, and
consequence clocks. Active and removed actors must be a disjoint complete partition. The active set
must equal both the runtime-selected and trace-active set.

The finite family contains exactly the union of:

- the full actor set;
- every leave-one-out assignment;
- every leave-two-out assignment;
- every singleton assignment;
- the empty assignment.

Duplicate or extra assignments are rejected. Contracts, normalized source state, source ancestry,
mode, caps, candidate set, configuration, policy state, actor-state authority, header inputs, and
registered actors are held equal. Branch IDs, intervention hashes, trace IDs, and trace authority
sequences must be unique.

## Honest scripted consequence join

The trace must contain no action intents or message deliveries and the commitment must be the exact
native simulated-hypothesis abstention. The post-trace fixture consequence must bind:

- fixture, intervention, scalarizer, hypothesis, trace, and inert effect identities;
- the exact scripted integer utility in both observed fixture data and realized utility;
- exact registered consequence clocks and zero clock uncertainty;
- exact observation uncertainty and `WorkVector` realized cost;
- exact fixture provenance and `ORACLE_NONPROMOTABLE` evidence taint;
- hard-false causal and consequence-grounded claim flags.

The commitment and scripted consequence formation charges are checked against their exact canonical
event sizes and native parent-causal convention. These joins show that the fixture is internally
replayable; they do not turn its scripted number into an observed effect.

The single round is validated independently of the outer trace hash. Candidate and selected actors
must be canonical and remain inside `K` and `C`; every considered coalition must be canonical,
unique, candidate-bounded, `C`-bounded, and the beam must remain inside `B`. A nonempty selection
must appear in that beam. Round-local staged/consumed messages, accepted actions, and endogenous
event IDs must exactly agree with the trace collections; v1 requires all such collections and all
rejected message/action records to be empty.

## Native nonzero ledger slices

Runtime work is the half-open native interval
`[trace.ledger_start_sequence, trace.ledger_end_sequence)`. A trace may begin after routing,
retention, or unrelated prior work. Prefix and suffix entries remain hash-verified but cannot affect
trace work, actor attribution, shared work, or historical resource debit.

Every trace-slice charge remains on the fork branch and, when causal IDs are present, names the
source hypothesis. Slice charge clocks equal the source/horizon tick, and total slice work cannot
exceed the trace's native `max_episode_work`. Each active actor has exactly one activation charge
inside the slice. Commitment and consequence formation charge ticks equal their event clocks.
Focused coverage includes a trace emitted by the real `CoalitionRuntime` with a nonzero start, plus
forged large actor-labelled prefix and suffix charges that do not change debit.

## Arithmetic terms and corrected ranking

For full registered set `S`, v1 stores three different quantities rather than conflating them:

`D_i(S) = U(S) - U(S ∖ {i}) - historical_resource_debit_i`

`M_i = U({i}) - U(∅)`

`I_ij(S) = U(S) - U(S ∖ {i}) - U(S ∖ {j}) + U(S ∖ {i,j})`

`D_i(S)` is retained as a full-context difference term for fixture learning diagnostics. It is not
used as a main effect. Shadow scoring uses:

`score(C) = Σ M_i + Σ I_ij(S) - current_predicted_resource_debit(C)`

This applies the current compute/message/risk debit once. In the regression fixture:

- `D_A=440`, `D_B=380`;
- `M_A=200`, `M_B=150`;
- `I_AB=250`;
- current debit for the pair is `10`;
- pair score is `200 + 150 + 250 - 10 = 590`.

The prior calculation incorrectly summed `D_A + D_B + I_AB` and double-counted interaction and
historical cost. Exactness is inherited from the source authorities, whose actor IDs and canonical
orders are retained in the snapshot. Pairwise reconstruction is marked exact only when every source
authority contains exactly two actors and the scored coalition has at most two actors. A small
request backed by any higher-order authority remains approximate. For a two-actor score, every
retained source authority must use that same two-actor domain; one matching authority is
insufficient once other domains have been aggregated. Thus AB and AC histories cannot mark AB or BC
exact. Larger, higher-order, or cross-domain scores receive
`higher-order-interactions-unmodeled` when selected.

Self-reported readiness `expected_decision_value` remains ignored. Ranking sees only the native
header, readiness costs/risks, immutable delayed fixture terms, and exact registry/assembly
bindings. Readiness risks are bounded by an integer-micro probability ceiling before exact rational
price multiplication; very large but finite floats fail closed instead of reaching `ceil` or raw
floating-point overflow.

## Replay, leakage, and retention guards

`ExactCoalitionCredit` and `InteractionCreditSnapshot` require private factory tokens. Snapshot
creation additionally receives a one-to-one set of `ForkAuthority` records, rederives every credit,
and requires byte-exact equality. Assessment must receive those source credits and authorities
again. It reconstructs each `ForkAuthority` against the current exact registry and assembly, then
requires historical actor-to-perspective bindings to equal the current bindings, recreates the
snapshot, and requires byte equality before scoring. A semantically forged credit,
snapshot, binding, or authority with a correct new self-hash is therefore rejected downstream.
Successful proposals record `source_replay_verified=true`.
Supplied credits and authorities must exactly equal the snapshot's retained IDs with unique,
one-to-one credit-to-authority coverage; duplicate, unused, and extra authorities are rejected.

Training authority includes every source hypothesis, all of its causal ancestors, its commitment,
and the scripted consequence supplying utility, plus their payload digests. Shadow assessment
rejects a request whose event ID, direct source IDs, payload digest, or representation digest
intersects any of them. This is the strongest check possible from a payload-free
`DispatchEventHeader`; deeper request ancestry would require a separately supplied event-ledger
authority.

Forbidden-key checks normalize case, camel-case boundaries, spaces, punctuation, and separators,
so spellings such as `Future-Outcome`, `future outcome`, and `futureOutcome` cannot bypass the
fixture boundary. Separatorless aliases such as `futureoutcome`, `groundtruth`, and `oraclelabel`
are compared through underscore-free fingerprints and are also rejected.

Snapshot `retained_state_bytes` is a fixed point over the complete canonical serialized snapshot,
including full accounting and the fixed-width self-hash. The constructor rechecks equality with
`len(canonical_bytes(snapshot.payload()))`.

## Permanent proposal fence

Every proposal carries at least:

- `scripted-noncausal-fixture-only`;
- `no-action-effect-authority`;
- `accounting-unapplied`;
- `shadow-only`;
- `activation-disabled`;
- `cooperation-claim-not-authorized`;
- `scientific-promotion-blocked`.

It also records `consumable_by_runtime=false`, `scripted_fixture_only=true`, and hard-false causal,
consequence-grounded, cooperation, dispatch, commitment, effect, update, activation, application,
and promotion authority.

## Mechanism evaluation

| Criterion | v1 assessment |
| --- | --- |
| Biological plausibility | Delayed credit and transient coalitions are useful analogies; exact scripted removal forks are not a biological model. |
| Computational plausibility | High for small finite fixtures; arithmetic is exact under the frozen assignments. |
| Engineering feasibility | High in shadow mode because it reuses native immutable ESCS records and exact replay. |
| Scaling behavior | Fork coverage is quadratic plus singleton/empty controls; beam enumeration remains capped. Higher-order interaction is not identified. |
| Sample efficiency | Unknown. Fixture decomposition is more informative than one undifferentiated number but supplies no natural-data evidence. |
| Reasoning quality | Not measured. Scripted utility tests mechanics, not reasoning competence. |
| Robustness | Strong against incomplete assignments, authority splicing, ledger contamination, forged credit, temporal leakage, and normalized future-key aliases. A dishonest fixture producer can still choose arbitrary numbers. |
| Interpretability | High: main, conditional difference, pair term, and current debit remain separate. |
| Emergent behavior | None established or enabled; output is nonconsumable. |
| Computational efficiency | Declared accounting is bounded and self-hashed but remains unapplied and incomplete; it is not a measurement of all replay, hardware, or supervisory cost. Complete fixtures are appropriate for sparse tests rather than every event. |

## Remaining gate

No activation experiment should consume these terms. A future causal plane would need explicit
counterfactual action/effect authority, independently generated outcomes, held-out worlds,
matched-compute and shuffled controls, hardware or governor resource measurement, calibration, and
a separately reviewed runtime adapter. That work is intentionally absent from v1.

## Verification

Focused verification on 2026-07-12:

```text
.venv/bin/python -m pytest -q tests/unit/test_escs_coalition_evidence.py
........                                                                 [100%]

.venv/bin/ruff format --check src/mop/escs/coalition_evidence.py tests/unit/test_escs_coalition_evidence.py
2 files already formatted

.venv/bin/ruff check src/mop/escs/coalition_evidence.py tests/unit/test_escs_coalition_evidence.py
All checks passed!

.venv/bin/mypy src/mop/escs/coalition_evidence.py tests/unit/test_escs_coalition_evidence.py
Success: no issues found in 2 source files
```
