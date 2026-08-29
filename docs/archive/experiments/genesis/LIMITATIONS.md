# Cognitive Material Genesis — limitations

External activation is `false`. No unqualified Nous is assigned.

Fifty-nine independent external reviewers graded this program between 3 and 8 out
of 20, median 5, and raised 138 blocking defects. Many were repaired. The ones
below were not, and they bound what this program's result can be read to mean.

## Seven of K10's nine composed mechanisms are inert

The integration candidate exposes its own ablations so it cannot win merely by
being larger. It did not win, and the ablations explain something sharper than
that.

Full K10 scores 0.1232. Freezing seven of its nine composed mechanisms leaves
that score **bit-identical**: append-only projection, bounded-radius
neighbourhood rule, input-dependent recurrence, per-region radix under rent,
the prediction-error gate, typed per-edge plasticity, and unfrozen
allocate/split/merge/prune all change nothing at all. Only two mechanisms move
it: freezing elapsed-time decay drops it to 0.0643, and freezing the monolithic
dense rewrite drops it to 0.0000.

So the integrated field is a dense monolithic rewrite plus time decay, wearing
seven other mechanisms as decoration. `integration_earned_its_complexity` is
published as false.

This bears directly on the headline null. A substantial part of why the
candidates do not beat an associative control is that most of their declared
machinery is not doing measurable work on these families. Whether that is a
property of the mechanisms or of these implementations of them is exactly the
question this program cannot answer, and the next one should.

## The strongest control differs from the candidates in representation, not only in architecture

This is the most important caveat on the headline result and it was raised by
the cross-examination round rather than found by the author.

It is tempting to read "the monolithic control beat every structured plastic
field" as a verdict on structure. It is not, and the numbers say so directly:
K1 is also a monolith and scores 0.0835, below the structured K2 at 0.1634. So
"monolith beats structure" fails inside the candidate set itself.

What separates the strongest control is not that it lacks structure. It is that
it stores exact content-addressable associations and answers by prefix match,
while every candidate stores lossy low-bit projections of the same experience.
That is a difference in representation fidelity, and it is confounded with the
architectural difference the program set out to measure.

Separating the two requires an ablation the program did not run: the control at
the candidates' precision, and a candidate given exact associative lookup.
Until that runs, the honest statement is that an exactly-addressed associative
core outperformed low-bit plastic fields on these families — not that
architecture does not matter.

## The canaries test mechanism hygiene, not counterfeit discrimination

Twelve canaries passed while a counterfeit tied the best candidate. They were
never going to catch it: each exercises one mechanism in isolation — a rewrite
commits, a rollback restores, growth pays rent — and none of them ranks a
counterfeit against a candidate. A suite that had run `random_growth_plastic`
alongside the candidates and required separation would have caught this before
the tournament, not after.

## The result is about these implementations, not about the architecture class

Eleven candidate materials were implemented once each, by one process, over one
working session. A null on K6 is evidence that *this* adaptive-topology
implementation did not beat an equally resourced monolith on *these* families.
It is not evidence that adaptive topology cannot help. The strongest honest
statement is about the implementations that ran.

## Scale

The tournament runs 32 developmental histories over 14 families with 23 arms.
Each history is 14 sealed units, roughly 200 observations. That is a moderate
tournament by the master plan's own sizing and it is far below the scale at
which developmental effects would be expected to separate architectures
convincingly. No meaningful-scale native training was performed: the materials
develop through verified rewrites during a run, not through a training
procedure over a large corpus.

## The attempt gap is architectural, not an artefact of the cap

The post-pilot round objected that the candidates were attempt-starved: each was
capped at thirty-two proposals per consolidation cycle while the strongest
control ran uncapped at roughly 5,300 attempts per developmental history. If the
cap were the cause, raising it should close the gap.

It was tested rather than argued. A sensitivity tournament re-ran the full
150,528 episodes with the cap raised sevenfold, to 220
(`SUBSTRATE_GENESIS_ATTEMPT_MATCHED_SENSITIVITY.json`, flagged
`is_sensitivity_analysis_not_principal`). Attempts barely moved:

| arm | attempts at cap 32 | attempts at cap 220 | score 32 | score 220 |
|---|---:|---:|---:|---:|
| K8 event sourced | 466 | 466 | 0.1625 | 0.1638 |
| K2 graph | 878 | 1308 | 0.1647 | 0.1598 |
| K1 monolithic | 325 | 324 | 0.1670 | 0.1554 |
| S2 control | 5324 | 5325 | 0.4246 | 0.4201 |

The candidates were never hitting the cap. They propose few durable changes
because their own mechanisms license few per cycle — a dense field rewrite or a
verified topology operation is simply a coarser unit of change than an
associative write. Sevenfold headroom left every score inside noise.

That converts the caveat into a finding. The attempt asymmetry is a property of
the architectures, not of the harness, and attempt-matching does not rescue the
candidates. It is also the mechanism behind the headline result: the control
wins partly because its unit of durable change is cheap enough to try thousands
of times inside the same compute budget.

The residual caveat is narrower than it was: no configuration has been found in
which a candidate both proposes at the control's rate and remains the same
material.

## Proposal counts were not equal by construction

The strongest control emits thousands of candidate durable changes per
developmental history while the candidate materials, as first delivered,
emitted roughly twenty-seven. Both operated under the same durable-write
budget, so the control explored roughly two hundred times more changes. This is
an implementation asymmetry that biases toward a false null, and it was
repaired by giving every candidate a comparable, explicitly capped and
mechanism-licensed proposal set. Results before and after that repair are both
published. Readers should treat the pre-repair numbers as uninterpretable on
this axis.

## Equal budget is not equal spend

Parity is enforced on budgets and measured on spend. An arm that spends more
inside a shared budget is more thorough, not privileged, but the comparison is
not literally compute-matched. The tournament reports the utilisation ratio and
flags a win bought by outspending the comparator as requiring a compute-matched
rerun.

## The memory-envelope frontier measured nothing, and the numbers say why

Stage 4 ran all fourteen families at all six envelopes. Every arm scored
**identically at 512 MB and at unconstrained** — not approximately, exactly, to
the last digit, at every envelope. The frontier is a flat line.

The reason is in the footprints. The largest material state observed was
141,827 bytes against a smallest envelope of 536,870,912 bytes: **0.026% of the
budget**. K6's entire durable state is 54 bytes; K8's is 108. The strongest
control, at 87,858 bytes, is the heaviest thing in the tournament and still
four orders of magnitude below the tightest constraint.

The envelope never binds, so no capability-density question is being answered.
The sixty-six exhausted cells at every envelope are *operation*-budget
exhaustions, identical across envelopes because that budget is constant.

This is published as a flat frontier rather than dressed up as a Pareto result.
A meaningful capability-density study needs materials three to four orders of
magnitude larger than these, or envelopes small enough to bite. The reported
footprint is in any case the material's own packed state, not process resident
memory.

## Continuous time is thin

Elapsed time is supplied by the harness in milliseconds attached to
observations, and the continuity lane is paced against the real clock. But only
K4 is driven by time as its primary law, and no arm is exposed to genuinely
irregular real-world event timing. Claims about continuous-time cognition rest
on a narrow instrument.

## The challenge families are synthetic and small-alphabet

Answers live in an eight-symbol alphabet, so chance is 0.125 and a lucky guess
is common. The families are hand-specified symbolic structures, not natural
data of any kind. They were validated in both directions — a record store
scores at or below chance on all fourteen, and a reference learner that reads
only the observation stream scores 1.0 on all fourteen — which makes them a
sound instrument for the question asked, but they are not the world.

One family, `task_composition_transfer`, leaked a copyable answer until it was
repaired, and one, `long_horizon_goal_recovery`, was unanswerable from
experience until it was repaired. Both were found only because the record-store
null and the reference learner were run against every family. Families are only
as trustworthy as those two instruments.

## Hidden composition is a pairwise interleave

Composition is implemented as interleaving two families in one history and
scoring probes from both. That is a real transfer test — the reference learner
drops from 1.0 to 0.6 on it — but it is a narrow notion of composition. Nothing
here tests composition of three or more systems, or composition of a learned
skill with a novel tool.

## Mutations are not fully covered

Twenty-four of the thirty-three declared mutations are injected with zero
survivors. Nine remain pending, awaiting instrumentation that would let them be
injected. A pending mutation is reported as pending, never as caught: those
nine failure modes have not been excluded.

## The exact shell is a checkpoint, not a separate authority

Identity, lineage, provenance and the claim boundary are carried in the
checkpoint and enforced by code in the same process as the approximate
mechanisms. Nothing prevents a defect in a material from corrupting them other
than the checkpoint round-trip test. A genuinely separate exact authority was
not built.

## External review role

The review process supplied proposals, not findings; where a review is cited
here, the underlying defect was reproduced before being acted on. Its median
feasibility grade of 5/20 should be read as what it is: fifty-nine independent
readers judging that this program, on the evidence available, was unlikely to
reach a defensible Outcome A.
