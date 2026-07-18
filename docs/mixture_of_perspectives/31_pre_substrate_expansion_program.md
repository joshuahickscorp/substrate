# Pre-Substrate Expansion Program: Unified Campaign Engine and the Broad Discovery Campaign

> **Append-only. Additive to the Form Substrate Program, not a competing paradigm.** This document records
> the unified campaign execution engine and the broad pre-substrate discovery campaign built to satisfy the
> mandate in `mop_expansion_bundle/{00_MOP_MASTER_GOAL_PROMPT, 01_MOP_UNIFIED_CAMPAIGN_EXECUTION,
> 02_MOP_PRE_SUBSTRATE_EXPANSION}.md`. It reconciles and extends the existing Form Substrate Program,
> F-series bank, mechanism epochs, potential atlas, exhaustion ledger, and active runs. New valid receipts
> override prose; historical sealed evidence is immutable. Inherited encoders remain frozen instruments and
> controls only; the eventual owned substrate must be earned from evidence, not shaped around an encoder.

## 1. Why this exists

The project had reached the ceiling of serial, bed-by-bed orchestration: each STARSS23 value-of-computation
bed was scaffolded, launched, and analyzed as the top-level execution unit, one at a time, behind a chain of
waiter parents. That model cannot run a broad discovery campaign across many forms, phenomena, and
mechanisms concurrently, and it cannot apply coverage pressure so that vision, action, memory, symbols, and
partner transfer stop being empty while audio scheduling is over-explored.

This program replaces that model with a durable-DAG campaign engine (`src/mop/campaign/`) and stands up the
broad pre-substrate discovery campaign on top of it. The headroom instrument from doc 30 is preserved and
integrated as one mechanism-diagnosis node inside the wider campaign, not as the session's deliverable.

The honest starting point is unchanged: the repository is at Stage 2 of 5. Stage 3 (empirically useful
active mechanisms) is not achieved. This program is discovery machinery for reaching it, not a claim to have
reached it. `activation_allowed`, `scientific_promotion`, and `independent_scientific_confirmation` are
hardcoded false across every new artifact.

## 2. The unified campaign engine

A campaign is a durable DAG of typed nodes, not a hardcoded stage list. The engine lives in
`src/mop/campaign/`:

- **`specs.py`** defines the declarative spec objects the mandate names: `CampaignSpec`,
  `ResearchQuestionSpec`, `BedSpec`, `ArmSpec`, `ReproductionSpec`, `VerificationSpec`, `ResourceRequest`,
  `Dependency`, and `DecisionRule`, plus `NodeSpec` and `ExternalDependency`. Every node carries a real
  `entrypoint` (`module:function`) resolved to executable code, workload-specific resource classes, coverage
  tags, and precommitted decision rules. A node with no importable runner cannot be marked implemented.
- **`dag.py`** computes the runnable frontier under real dependency and authority gates. A `COMPLETION`
  dependency is a run-after ordering constraint; a `SEAL` dependency needs the earlier sealed artifact; an
  `AUTHORITY` or external-boundary dependency is how work is durably queued behind the live campaign and
  auto-activated when it clears. Independent nodes (separate architectures, seeds, scoring corroborations,
  reproduction axes, unrelated waves) share one frontier and run concurrently; a serial barrier exists only
  where a later decision truly depends on an earlier sealed result.
- **`state.py`** holds durable, restart-safe state with per-node leases. A `RUNNING` node whose worker is
  dead or whose heartbeat is stale is recovered to eligible; the scheduler resumes at exact status.
- **`decisions.py`** fires precommitted decision rules after a parent seals (the criteria are fixed in the
  manifest before results are read) and applies null-safe stopping: a null seals the parent and prunes its
  positive-only follow-ups, which are skipped rather than rerun to chase a positive.
- **`broker.py`** is the one global resource broker. It reuses the reviewed `dynamic_worker_controller` for
  the CPU worker ceiling and Hawking behavior (the measured 20-worker seeded-hash optimum, shed-to-reserve
  under Hawking, the priority lever) and adds multi-class accounting: CPU-hash-heavy, CPU-light,
  native-threaded, memory-heavy, IO-heavy, and exclusive nodes consume the budget differently. It represents
  the live General Run and horizon chain as external consumers, and yields only this campaign's resources
  when Hawking appears. Beds submit to it; they do not each build a maximum pool.
- **`invariance.py`** proves receipt invariance across worker widths with a partition-independent XOR
  reduction (the merged receipt equals the XOR over all items regardless of how they were sharded), and
  records throughput per width so a materially different workload can pick its measured optimum.
- **`executor.py`** is the single shared worker fleet. It samples the broker, submits every admitted
  eligible node to one `ProcessPoolExecutor`, seals results, fires decisions, applies null-safe skips, and
  persists state after every event.
- **`coverage.py`** applies coverage pressure: a node advancing an untested form family or phenomenon is
  prioritized over the tenth variation of a covered one. This is a scheduling preference, never a change to a
  scientific verdict.
- **`nodes/framework.py`** is the reusable neutral scientific framework (deterministic seeding, control
  construction, the exact sign-flip over independent units, SESOI handling, matched-budget lifecycle
  accounting, canonical sealing). Decisive verifier logic is kept out of it: a verifier node re-derives a
  result independently rather than importing the producer's grading path.

## 3. The broad discovery campaign

`manifest.py` assembles one campaign, `mop_pre_substrate_expansion_v1`, spanning Waves A through J. It
contains real runnable science question families (each a deterministic mechanics experiment with a named
control, an exact sign-flip over independent units, a preregistered SESOI, and an honest verdict where a tie
or wrong-direction result is a legitimate null), cross-question analysis nodes, precommitted decision-branch
reproductions, and contracted external-input families whose only blocker is named data or an external
authority. The machine-readable atlas (`proof/PRE_SUBSTRATE_PHENOMENA_ATLAS.json`,
`registry/phenomena.yaml`, `registry/mechanism_candidates.yaml`) records every row with the full required
field set and links each to a real node, never a title-only concept.

The first safe local frontier runs now on this host; the contracted external families and the precommitted
branches are durably queued. The live General Run and successor horizon chain are adopted as external
resource consumers and observed, never signaled.

## 4. Mechanism cards, negative-space, and the readiness gate

`nodes/analysis.py` generates mechanism cards only from sealed results, each with a failure domain and an
M0 through M7 cross-domain evidence level; a mechanism with no known failure domain is not understood. The
negative-space synthesis clusters the record's nulls by causal failure family (weak or anti-informative
estimator, proxy signal without marginal value, pseudoreplication, architecture fragility, absent
heterogeneity headroom, replay without future learning, matched-compute collapse, and the rest) and emits
precommitted replacements so expensive dead shapes are closed rather than rerun with new names.

The executable Stage-3 readiness gate evaluates the twelve pre-substrate evidence gates over the sealed
record. None are met yet (no sealed mechanism card reaches evidence level M5 or above), so the canonical
substrate tournament stays closed. The gate exists to keep it closed until the evidence is real, not to
manufacture the feeling of progress.

## 5. Compliance and honesty posture

A hard mandate-compliance ledger (`src/mop/campaign/compliance.py`,
`proof/campaign_run/COMPLIANCE_LEDGER.json`) enumerates every hard requirement in bundle files 00, 01, 02,
and 04 and checks reality: it imports the engine, resolves runner entrypoints to real callables, counts
runnable local families, and reads the live campaign state. The compliance verifier fails if any
safely-executable hard requirement is still planned or partial; an external-blocked family is contracted,
not a failure. This is the line between honest incompleteness and scope substitution.

Every campaign artifact preserves the evidence boundary: no activation, no scientific promotion from one
run, no retrospective success criterion, no hidden scoring-unit change, no result-dependent redesign without
a fresh sealed preregistration, and no weakening of controls, matched-budget accounting, SESOI, or
independent verification. Node count is an execution envelope, not a success metric.
