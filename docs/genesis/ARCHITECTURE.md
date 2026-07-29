# Cognitive Material Genesis — architecture

External activation is `false`. No outcome of this program assigns unqualified
Nous, consciousness, sentience, phenomenal experience, or moral status.

## What this program asks

> Can a frozen generic cognitive material develop new useful internal
> organization from previously unseen experience, and can that development
> outperform equally resourced static, replayed, precompiled, and equally
> plastic alternatives?

The inherited result is preserved unchanged: the Final Revision campaign
closed with `substrate_final_revision_complete`,
`internal_functional_nous_claim_closed`, `real_world_sandbox_ready`, and a
decisive architectural effect of 0.0 with a 95% confidence interval of [0, 0]
against an S2 monolithic deterministic state machine. This is a new question,
not a second attempt at the old one.

## The material interface

Every arm — candidate, control, baseline and instrument — implements one
interface (`genesis_material.py`). The interface is where the program's most
important guarantee lives.

```
observe(Observation)    update active state; writing durable state raises
answer(Probe) -> Answer read only; writing durable state raises
propose() -> Proposal   offer durable changes justified by experience
apply(Verdict) -> Receipt  commit admitted proposals, refuse the rest
rollback(Receipt)       undo one commit exactly
checkpoint() / restore()   exact durable identity across a round trip
freeze_mechanism(name)  disable a mechanism this arm does not own
cost() -> measured resource use
```

A material never receives an expected answer. It receives a `Verdict` carrying
`admitted`, `improvement` and `retention` — three scalars computed by an
evaluator that holds the sealed answers. There is no field, method or
serialization path by which a label reaches a material. `Unit.public()` returns
observations and probes and no sealed object at all.

## The eleven candidate materials

Each is a separate implementation with its own durable-change law, not a shared
class with a transition switch. `distinctness_report` runs every arm over an
identical observation and probe sequence with every proposal admitted and
requires the durable state digests to differ.

| | material | durable change is |
|---|---|---|
| K1 | monolithic | dense global accumulation, no structure |
| K2 | graph | a rewrite of one typed edge carrying its own value, scope and precision |
| K3 | cellular | a local neighbourhood rule inside a bounded radius |
| K4 | continuous time | decay, consolidation and expiry driven by elapsed time |
| K5 | state space | a slow update to bounded recurrence parameters |
| K6 | adaptive topology | allocate, split, merge, prune and archive under rent |
| K7 | mixed radix | per-region radix selection under the earn-your-bits rule |
| K8 | event sourced | an append-only archive, with all state a projection of it |
| K9 | predictive | a write gated on prediction residual |
| K10 | integrated | the composed mechanisms of K1–K9, with its own ablations |
| K11 | sparse fiber | interference-gated rebinding, split and fuse in a fixed bank |

K3's locality is testable rather than asserted: rewiring a non-neighbour leaves
K3's durable digest unchanged while K2's changes. K6 is the only material with
unfrozen allocation; every other arm has that mechanism frozen.

K11 was designed by the Grok `grok_original_material_author` role and adopted
after review.

## Controls and baselines

`S2_task_independent_monolithic_persistent_core` is the arm the decisive claim
must beat. It has one canonical identifier with an alias map so no analysis can
score a weaker object than the baseline the inherited null was measured
against. It is a separate implementation, not a flag on a candidate, and it
receives every opportunity a candidate receives.

Deprivation baselines each lack exactly one named opportunity: `plasticity` for
the static, replay, summary, retrieval and precompiled arms, `correct_history`
for the wrong-history arm, `history_order` for the shuffled arm,
`verified_growth` for the random-growth arm, and `development` for the record
store. The `oracle` receives the generating structure and is the headroom
reference.

## Equal opportunity is measured

Seven channels — information, compute, persistence, plasticity, sensors,
teaching, memory — each with a named measurement. Information, sensors and
teaching must be byte identical; the rest hold to a 2% relative tolerance. The
audit's load-bearing test runs one arm with more operations than another under
the same envelope and requires the audit to fail.

Equal budgets are not equal spend. The tournament measures utilisation and
reports a win bought by outspending the comparator as requiring a
compute-matched rerun.

## The three-way probe split

A developmental history is fourteen sealed units of one family, concatenated.
Its probes are partitioned by unit into three disjoint roles drawn from
separate identifier bands:

- **development** — proposals are verified against it
- **retention** — guards against forgetting
- **scoring** — the reported measure, touched by nothing else

Without that separation, plasticity reads the held-out outcome and every result
is void. `ProbeSplit` checks disjointness at construction.

## Verified plasticity

A proposal is simulated, verified, then committed or rolled back. Each proposal
is tentatively written, scored on development and retention, and reverted when
it does not pay. The improvement of a change is only observable once it has
actually been written; measuring before writing reports zero for everything and
admits nothing.

## The instruments that make a null meaningful

Two arms exist so that a failure can be interpreted:

- `record_store_null` copies observed labelled fields and abstains otherwise.
  All fourteen families hold it at or below the 0.125 chance level, so the
  measure tests development rather than storage.
- `reference_learner` implements each family's intended solution path from the
  observation stream alone, touching no sealed answer, seed or generator
  internal. It scores 1.0 on all fourteen families, so every family is
  answerable from experience and a material that fails is producing a result
  rather than hitting a broken instrument.

## Frozen analysis

The independent unit is the developmental history; episodes within a history
are not independent, so every interval resamples histories. The estimator is
the mean paired difference; the interval is a bias-corrected and accelerated
bootstrap; multiplicity is Holm-Bonferroni over the ten primary claims.

Outcome A requires the primary gate from the master plan — effect at least
0.05 with a lower bound above zero — plus replication, hidden-composition
transfer, oracle headroom, zero surviving mutations and clean-clone
reproduction. A strictly stronger secondary gate requiring the lower bound to
reach the smallest effect of interest is reported alongside; it never lowers
the bar.

## Freeze

The freeze publishes the configuration digest, generator source digest,
selected candidate, thresholds and claim boundary, then derives the principal,
replication and hidden-composition seed namespaces from the digest of the
freeze document itself. No namespace can be computed, and therefore no
principal instance generated or seen, before the freeze exists. The derivation
is public, so the commitment is checkable afterwards.
