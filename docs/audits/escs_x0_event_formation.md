# ESCS X0 charged event-formation preregistration audit

**Status:** implemented, preregistered, and intentionally unexecuted
**Evidence ceiling:** Gate-A generated candidate/control evidence only
**Scientific promotion:** blocked regardless of outcome

X0 is Experiment 0 from `docs/mixture_of_perspectives/18_event_sourced_coalition_substrate.md`.
It asks whether a learned event former can operate on raw asynchronous streams, preserve downstream
decision quality, and reduce total lifecycle work without receiving semantic labels from the evaluator.
It is independent of EDCM-1 scientifically: EDCM begins from structured observations, whereas X0 tests
the missing raw-event and idle boundary.

No official X0 receipt or verification artifact exists at scaffold time. Tiny tests exercise mechanics only
and cannot produce a scientific verdict.

## Frozen scope and arms

The canonical config is `configs/experiment/escs_x0_event_formation.json`. Its payload is content-addressed
and its authority digest is compiled into the study module. The registered arms, in fixed order, are:

1. `learned_event_former`: an injected policy trained only from visible packets and delayed public action
   value;
2. `fixed_raw_delta`;
3. `periodic`;
4. `always_on` tick processing;
5. `header_only` action cognition;
6. `novelty`;
7. `uncertainty`;
8. `shuffled_rate_matched`, with the learned arm's exact admission count in every episode; and
9. `oracle_semantic_nonpromotable`, the evaluator-only upper bound.

The oracle is never a promotion comparator. It carries `oracle_nonpromotable` standing through seed rows,
receipts, aggregates, and verification. All nonlearned arms are controls or diagnostics. Even after a
favorable fresh verifier, the learned arm is only a feature-flagged, unverified candidate for permissive
Gate-A integration.

## Raw asynchronous bed

Every arm in a paired seed receives the same immutable generated traces. Each trace includes:

- a long interval with no captured packets;
- high-rate irrelevant changes;
- sparse decision-relevant changes;
- irreducible noisy-TV packets;
- delayed public consequences;
- a correlated multi-sensor event storm containing a sparse useful event;
- uncertain source identity tokens;
- different capture, local-clock, and arrival times; and
- split-specific event and clock families.

Tune and gate use registered known families. Held-out and fresh-verifier event and clock families are each
unseen relative to tune. Producer and fresh seeds are disjoint. The generator enforces a packet cap before
any arm runs.

The generated bed is not assumed valid. All producer seeds first complete the gate split. The difficulty
gate requires useful updates to remain sparse, long idle and irrelevant high-rate regions to exist,
always-on action value to clear its floor, noisy TV/storms/uncertain identity/asynchrony to be materially
present, and every leakage check to pass. Failure stops before held-out routing and is labeled
`difficulty_or_leakage_gate_invalid_bed`; it is not a mechanism null.

## Leakage boundary

The generator emits two distinct typed products:

- `VisiblePacket`, containing raw values, source and clock metadata, prior raw values, an explicitly charged
  compact header, identity uncertainty, and payload size; and
- `EvaluatorTruth`, containing usefulness, event/referent/type identity, target action, noise/irrelevance,
  storm membership, deadline, and consequence timing.

No `EventPolicy` method receives `EvaluatorTruth`. Training receives only sequences of `VisiblePacket` and
`TrainingObservation`, whose consequence contains packet identity, delivery time, realized public action
value, and charged feedback work. Sparse exploration is selected by a deterministic packet hash before
the environment returns delayed value. The scorer may use evaluator truth for metrics; the learned policy
may not use it for training, thresholds, early stopping, or deployment.

Policy construction receives an immutable bootstrap containing only the frozen learned-policy
hyperparameters. Generator schedules, split names, world parameters, evaluator schemas, official seeds,
controls, criteria, and verdict thresholds are not passed through the policy factory boundary.

The built-in learned policy is a deliberately small injected implementation, not a privileged architecture.
It performs bounded online logistic updates from delayed public values, chooses its admission threshold from
the preregistered target rate over unlabeled tune packets, freezes before evaluation, and reports training
operations, retained bytes, threshold, and state digest. Alternate policy implementations may be passed only
through the same narrow interface and must be separately authority-bound before an official comparison.

## Fully charged endpoints

Every arm reports the same complete deterministic lifecycle vector:

- raw transport and adapters;
- idle polling, plus always-on idle processing;
- event formation, including abstention/nonadmission evaluation;
- header construction and encoded header bytes;
- indexing, queue admission/removal, overflow drops, and queue drain;
- downstream cognition;
- learned-policy updates and delayed feedback;
- retained-state byte quanta across the deployment horizon; and
- queue-retention byte-time, encoded header bytes, serialization, and receipt work.

The primary plane is held-out utility versus the weighted total of those components. Deterministic work is
not wall time, energy, or silicon cost. Empirical timing and host telemetry are nonverdict runtime evidence
and must be collected by the campaign governor if the official stage is admitted.

Seed rows additionally report raw/admitted/useful rates, event precision and recall, detection delay,
noisy-TV and irrelevant false activation, header-only action value and bytes, calibration, queue depth,
drops, unprocessed work, storm deadline misses, retained state, and exact admission digests. Useful labels
appear only in retrospective metrics.

## Verdict contract

The difficulty and leakage gates precede inference. On a valid bed, a Gate-A candidate pattern is favorable
only when all registered paired-seed confidence bounds and every-seed directions hold:

- learned utility is within `0.01` of always-on;
- fully charged lifecycle work is at least `25%` below always-on;
- Pareto contribution is positive over fixed delta, periodic, exact-rate shuffled, novelty, and uncertainty;
- irreducible noise produces no excess activation relative to the rate-matched shuffle;
- predicted event value is calibrated on unseen event and clock families;
- queues remain bounded and registered storm deadlines are met;
- the header alone does not explain the full result; and
- no semantic label or oracle mapping crosses the policy boundary.

The producer aggregate is preliminary. A favorable producer row set becomes
`fresh_verified_gate_a_candidate` only when the verifier regenerates every producer gate and held-out row
byte-for-byte and the same criteria pass on five disjoint fresh seeds. A producer/verifier mismatch is an
implementation failure, never a scientific null.

The fresh verifier also reruns the bed-difficulty and leakage gate on the disjoint fresh streams. A broken
fresh bed is `difficulty_or_leakage_gate_invalid_bed`, not a mechanism null. A cheaper header-only arm that
is noninferior inside the registered tolerance blocks a favorable learned-event-former verdict.

The strong null remains unrejected when controls match the learned arm; only the oracle works; useful rate
approaches raw rate; false negatives or delay erase utility; headers already perform the cognition without a
full-system gain; noisy TV triggers activation; storms make the queue unstable; calibration fails on unseen
families; or full boundary overhead removes the saving. That result retires integrated event-driven
efficiency while leaving structured-observation dispatch work eligible.

## Resume, integrity, and implementation authority

The runner is serial and CPU-only. Gate and held-out phases checkpoint after each complete paired-seed row.
The atomic checkpoint binds config authority, implementation-authority digest, exact seed prefixes, row
digests, and phase digests. Writes fsync the temporary file, replace atomically, and fsync the directory.
Resume accepts only an identical authority and starts at the next complete seed boundary; a partial seed is
recomputed.

Config, implementation manifest, checkpoint, producer receipt, and verifier output must be distinct paths.
Receipts and checkpoints are self-hashed and byte-capped. An official run also requires the canonical
implementation manifest and its digest supplied independently at the command line. The manifest binds the
exact byte length and SHA-256 of the config, module, runner, tests, and this audit.

The manifest review status is `preregistered-scaffold-unexecuted`. It records implementation identity, not a
scientific result or permission to bypass `GOV_1`. The one-shot campaign supervisor must admit X0 only after
the current governor/policy freeze and its CPU, memory-pressure, thermal, disk, and exclusive-heavy-lane
checks pass. Admission refusal waits without altering seeds, horizons, controls, or criteria.

After final scoped review, regenerate and externally pin the implementation digest:

```bash
PYTHONPATH=src .venv/bin/python -c \
  'from mop.studies.escs_x0_event_formation import write_implementation_authority; d = write_implementation_authority(); print(d["manifest_sha256"])'
```

The intended later producer and verifier commands are:

```bash
export ESCS_X0_IMPL_SHA256=<reviewed-manifest-sha256>

.venv/bin/python scripts/run_escs_x0_event_formation.py \
  --implementation-authority-sha256 "$ESCS_X0_IMPL_SHA256"

.venv/bin/python scripts/run_escs_x0_event_formation.py \
  --verify proof/ESCS_X0_EVENT_FORMATION.json \
  --verification-out proof/ESCS_X0_EVENT_FORMATION.verification.json \
  --implementation-authority-sha256 "$ESCS_X0_IMPL_SHA256"
```

These commands are documentation, not an instruction to launch before governor admission. No automatic
retry changes a killed premise or terminal null.

## Known limits

- All streams, action mappings, and consequences are generated.
- The built-in policy is a bounded study candidate, not a general event-discovery learner.
- Abstract lifecycle work is deterministic accounting, not empirical energy or latency.
- The verifier shares the frozen implementation but uses disjoint seeds and families; a separately
  implemented verifier would strengthen promotion evidence.
- Five paired plus five fresh seeds can support only the exact generated mechanics scope.
- Natural/session-disjoint promotion remains blocked even after a favorable verified receipt.
