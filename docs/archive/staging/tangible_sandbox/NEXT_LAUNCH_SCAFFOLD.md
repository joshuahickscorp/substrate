# Tangible Sandbox — next-launch scaffold

**Status:** provisional design record; not a result, preregistration, launch
authority, or amendment to the active R2 lane.

**Purpose:** make the next experiment a short edit-and-launch exercise once the
current run has closed, rather than another full rebuild.  The current lane,
its source identity, its storage seal, and its evidence boundary stay
unchanged.

## What we know now

- The active lane is a single-writer, real-wall-clock continuity test.  It
  exercises persistent project state across office, image, video, audio, mesh,
  spreadsheet, presentation, and telemetry artifacts; it is **not** an
  end-to-end train-on-a-large-corpus study.
- Its storage guard is dynamic rather than a fixed 250 GiB rule.  The active
  threshold is 200,008,355,840 bytes (about 186.3 GiB), with the static project
  bytes correctly counted once because they already reduce filesystem free
  space.  The plan reserves empirical lane growth, transient writes, a safety
  margin, and the protected volume floor.
- Core acquisition is complete: 18 archives and 11 repositories, totaling
  86,036,677,480 downloaded bytes.  More downloads now would create volume and
  provenance work without changing the active lane.
- The historic pre-refactor program preserved a useful campaign pattern:
  durable DAG scheduling, leases/heartbeats, fixed decision rules, and
  independent capsule parallelism.  Its resource study reported Torch
  slowdowns of 1.00, 1.03, 1.12, and 1.22 at one through four capsules.  That
  was a different workload and is a calibration lead, not permission to run
  this continuity lane concurrently.
- The executable next-launch control plane now exists outside R2's sealed
  sources. It contains editable design/data/calibration drafts, sealed pivot
  and adapter contracts, stale-evidence refusal, a deterministic post-R2
  review, receipt-invariant 1/2/4-worker calibration, fresh storage preflight,
  and a custody-handoff creator. See
  `docs/archive/staging/tangible_sandbox/NEXT_LAUNCH_RUNBOOK.md`.

## Boundary for the current R2 lane

Do not alter the current schedule, roots, worker count, evaluator separation,
or frozen source.  A clean completion and independent verification remain
required before it contributes any continuity evidence.  Until then, its
observations are operational telemetry, not a scientific outcome.

This scaffold may be edited while R2 runs because it is outside the sealed
execution path.  It must not be used to retroactively redefine R2's
hypothesis, scoring, or success criteria.

## The next experiment: a blind 24-hour shadow first

The next useful unit is not an immediate seven-day expansion.  It is a
separate, blinded 24-hour shadow run that answers whether the current
continuity result survives a new task/stimulus composition and a genuinely
independent scoring path.

### Fixed shape

| Element | Shadow-run requirement |
| --- | --- |
| Run root | New immutable run identifier and new workspace; no reuse of R2 state files. |
| Stimuli | New, manifest-pinned tasks/artifacts chosen before candidate execution. |
| Blinding | Custodian creates the seed, composition, and answer key; candidate receives only builder-visible material. |
| Evaluation | Evaluator-only root remains physically inaccessible to the candidate; scorer opens the answer key only after candidate trace lock. |
| Comparators | Keep R2 candidate, strongest fair control, budgets, and scoring definitions unless one new causal factor is separately preregistered. |
| Verification | Fresh verifier recomputes receipts and scores without importing the producer's grading path. |
| Storage | New quota/reservation and phase-specific dynamic plan; never borrow R2's protected margin. |
| Publication | Generate only after terminal trace, verification, and classification are sealed. |

### Why a blind shadow before seven days

A seven-day lane changes duration, exposure count, failure opportunity, and
operational risk at once.  If it differs from R2, we will not know whether the
change came from time, stimuli, concurrency, or reliability.  A blinded 24 h
shadow isolates the replication question cheaply.  Only after that should a
72 h intermediate duration be considered; seven days becomes a named duration
factor with its own task schedule, stop rules, storage reservation, and
precommitted analysis.

## Blindness and concurrency are separate controls

Blinding prevents information leakage; concurrency controls wall-clock time.
They need different safeguards.

### Blindness protocol

1. A custodian process writes a signed manifest containing the generator
   version, seed commitment, task IDs, split assignment, and hashes.
2. It materializes builder-visible artifacts and evaluator-only answers in
   separate roots.  The candidate account/process has no read path to the
   latter.
3. Candidate execution seals its append-only trace and artifact hashes before
   the scorer receives the decryption key or answer mapping.
4. An independent verifier recomputes the score from the sealed candidate
   trace and evaluator-only material.
5. The result record states whether every isolation check passed.  A failed
   check is invalid/diagnostic, never silently treated as a null or positive.

### Concurrency protocol

- **Current R2:** keep one worker.  It is a single-writer continuity timeline;
  parallel workers would alter the intervention rather than merely accelerate
  it.
- **Pre-run calibration:** benchmark one, two, and four *independent*,
  synthetic capsules on the actual host and storage class.  Record CPU,
  memory, I/O latency, thermal state, external disk drift, and receipt
  invariance.
- **Admission rule:** parallel execution is allowed only when each capsule has
  a distinct run root, fixed resource slice, no shared writable evaluator/data
  root, deterministic seeds, invariant merged receipts, and no breach of disk,
  memory, I/O, or thermal limits.
- **Scheduling rule:** run independent preparation and verification capsules
  concurrently only after calibration.  Keep any wall-clock longitudinal
  timeline single-writer unless the hypothesis itself is explicitly about
  multi-agent or multi-process continuity.

This preserves the good part of the old campaign engine—parallel independent
work—without turning shared state into an uncontrolled treatment variable.

## Data breadth: add episodes, not undifferentiated bytes

The existing acquisition gives a real base, but dataset size alone is not
experimental breadth.  Every added source must have a job in the causal design:
new sensory modality, longer history, disturbance/recovery, tool transition,
or held-out generalization.  Candidate and control must receive equal access
to whatever is admitted.

Create a metadata-only adoption card for each candidate source before any
large download:

```text
source/version/license/terms:
immutable URL and file hashes:
experimental question and independent unit:
modalities and novel stimulus family:
builder-visible versus evaluator-only split:
candidate/control access parity:
setup bytes, runtime working bytes, cache/reuse policy:
expected I/O and concurrency class:
holdout/scoring owner and leakage test:
accept/reject rule:
```

Priority order for a new stimulus bank:

1. Real, license-cleared multimodal project artifacts with durable history and
   recoverable disturbances.
2. Longer multi-session task sequences with held-out future work.
3. Interactive environments only when their state, actions, scoring, and
   baseline parity can be frozen and independently replayed.
4. Large training corpora only for a separately preregistered prospective
   training/curriculum arm; do not relabel the continuity lane as training.

## A prospective training arm, if wanted

Training could become a strong next family, but it is a different question:
whether a specified curriculum changes later performance beyond matched
exposure.  It needs its own manifest and at minimum:

- a training/curriculum intervention and a no-training or matched-exposure
  control;
- matched information, tokens, compute, storage, and wall-time budgets;
- held-out, evaluator-only post-training tasks;
- independent seeds as the analysis units and an a priori effect threshold;
- a separate training data license/hash ledger and a reproducible checkpoint
  policy.

That creates a clean bridge from today’s tangible continuity evidence to a
larger stimulus/training program without making either claim stand in for the
other.

## Deterministic post-R2 pivot table

The final implementation should read sealed state and select exactly one row;
it must not inspect unsealed outcome details to invent a new design.

| Required sealed condition | Automatic next state | Human-free action allowed | Must remain blocked |
| --- | --- | --- | --- |
| R2 incomplete, invalid, or verification fails | `repair_diagnosis` | Collect non-mutating diagnostics and draft failure record | New science launch, data acquisition, result classification |
| R2 valid and verification passes; blind shadow not yet run | `blind_shadow_ready` | Materialize only an already approved frozen shadow manifest and run preflight | Seven-day run, unapproved source download |
| Blind shadow valid and directionally consistent; duration calibration absent | `duration_72h_review` | Build feasibility plan and resource calibration record | Seven-day launch |
| 72 h feasibility valid, storage/concurrency gates pass, and duration preregistration sealed | `seven_day_ready` | Launch detached supervisor from frozen manifest; send lifecycle/30-minute health reports | Mid-run protocol edits or result-dependent branching |
| Any storage, isolation, or verifier guard fails | `safe_hold` | Preserve trace, alert, and release only the affected queued work | Automatic restart into a changed scientific protocol |

The implementation must be deterministic about *state transitions*, while
remaining conservative about external side effects: a new dataset, terms
acceptance, or altered scientific intervention needs an explicit approved
manifest first.

## Measurement density without false replication

The next 24-hour shadow can collect substantially more evidence than nine
coarse schedule markers without becoming a 120- or 240-hour serial campaign.
The staged density authority at
`plans/substrate/tangible_next_launch/MEASUREMENT_DENSITY.draft.json` provides
49 half-hour resource/health samples and three prespecified score dimensions at
each of the nine existing longitudinal events (27 scored observations).  It
also stages four zero- or measured-download stimulus-pack options so the
custodian can select one new, hash-pinned primary history after R2 closes.

This is greater *measurement density*, not 27 independent replications: all
observations in a single 24-hour history share its seed, writer, and exposure.
The independent unit remains a separately custodian-seeded held-out history.
Claims about breadth therefore require multiple sealed histories (the later
fabric/calibration path), while the 24-hour shadow remains a sharply
interpretable replication.  This lets us prepare the maximum usable stimulus
and scoring material now without pretending that repeated probes create new
scientific samples.

## Executable handoff

The runnable control-plane commands are deliberately separate from the R2
command surface:

```bash
python -m substrate.tangible_next status
python -m substrate.tangible_next review-r2
python -m substrate.tangible_next seal-design
python -m substrate.tangible_next run-calibration
python -m substrate.tangible_next preflight
python -m substrate.tangible_next prepare
python -m substrate.tangible_next seal-custody --handoff RUN/CUSTODY_HANDOFF.json ...
python -m substrate.tangible_next launch --handoff RUN/CUSTODY_HANDOFF.json
```

The short post-R2 session consists only of reviewing the result, selecting and
hash-pinning the new stimulus cards, recording the custodian commitment and
adapter versions, then sealing the draft. Every later command fails closed if
those bindings, storage, custody, or verification gates are missing.
`launch` is a one-shot launchd handoff: its worker retains the longitudinal
single-writer design, records heartbeat/30-minute reporting buckets, and locks
the candidate/control trace before the evaluator receives the answer mapping.

## Fill only after the active lane closes

```text
R2 terminal/verification status: {pending}
R2 trace digest: {pending}
R2 measured disk growth and external drift: {pending}
R2 maximum memory / CPU / I/O telemetry: {pending}
R2 isolation and source-identity verdict: {pending}
shadow stimulus-family gap selected: {pending}
calibration widths admitted: {pending}
next manifest identifier and digest: {pending}
```

## Preserved design sources

- `archive/pre-substrate-event-horizon/MOP_SCIENTIFIC_FRONTIER_STATE.json`
- `archive/pre-substrate-event-horizon/MOP_GENERATION3_LEDGER.md`
- `archive/pre-substrate-event-horizon/frontier/MOP_FRONTIER_PARALLEL_REPLAY.json`
- `archive/pre-substrate-event-horizon/docs/mixture_of_perspectives/31_pre_substrate_expansion_program.md`
- `proof/substrate/mop-substrate-master-v1/SUBSTRATE_LONG_RUN_RESOURCE_PLAN.json`

These record useful operational patterns and historic measurements.  They are
not imported as result evidence for Tangible Sandbox R2.
