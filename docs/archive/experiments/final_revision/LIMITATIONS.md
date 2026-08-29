# Final Revision Limitations

Every item here is a limitation of the frozen final-revision evidence. None is
repaired by re-running, because repairing a valid null by changing the target is
prohibited. They are recorded so that no reader mistakes the scope of the
result.

## What the decisive null does and does not show

- P3 — the selected kernel against the strongest fair equal-resource persistent
  alternative — is an exact tie in all three lanes: mean paired effect `0.0`,
  95% CI `[0.0, 0.0]`, against SESOI `0.05`. A tie is a null. No architectural
  or functional Nous advantage is claimed.
- The tie is a real behavioral result, not a scoring artifact. The selected
  `EventSourcedKernel` and the `TaskIndependentMonolithicPersistentCore`
  baseline are separately implemented, are instantiated fresh per episode, and
  are scored by per-episode recomputation against generator truth.
- Instrument liveness was checked directly rather than assumed. Degrading a
  single answer class of one system moves the reported P3 effect off zero with
  a non-degenerate interval — `+0.0903 [0.0694, 0.1111]` when the S2 baseline is
  degraded, `-0.0903 [-0.1111, -0.0694]` when the candidate is degraded, against
  `0.0 [0.0, 0.0]` unperturbed. The receipt and the script that produced it are
  `runs/substrate/final_revision/instrument_liveness_check.json` and
  `instrument_liveness_check.py`. This is a post-freeze supplementary check, not
  a required deliverable and not a scored endpoint. It changes no threshold,
  seed, split, generator or scoring rule.
- The tie is nonetheless a **ceiling tie between two compiled projections of the
  same generator vocabulary**, not a stress test of divergent mechanisms under
  open competition. Both systems answer the same seven classes correctly and
  neither is forced to diverge.

## The oracle headroom is unattempted capacity, not contested capacity

- Measured oracle headroom over the strongest baseline is approximately `0.125`
  in every lane, which clears the required `0.05` and the preferred `0.10`.
- That headroom is composed **entirely of the one answer class that neither the
  selected kernel nor the S2 baseline ever emits**. Both answer functions return
  keys 0–6; the expected map also requires key 7, so class 7 is always scored
  wrong for both systems.
- The bed is therefore formally non-saturated but the residual is unimplemented
  capacity rather than a hard frontier either architecture is straining against.
  Read the headroom as "neither system attempts this class", not as "the systems
  nearly exhausted a difficult bed".

## The transcript-replay control is not an independent control

- `full_transcript_replay` is scored through the same `_kernel_answers` function
  as the selected candidate, on the same event list. The P1 comparison against
  transcript replay is therefore zero by construction and carries no
  information.
- The informative P1 comparison is against the stateless control, where owned
  state produces a large, non-degenerate effect. The transcript arm should not
  be read as evidence that owned state beats transcript replay.

## Baseline coverage is narrower than the declared ladder

- Four declared baselines — `disconnected_model_ensemble`,
  `stateless_model_router`, `largest_model_always`, `all_models_always` — did
  not execute. They are recorded as availability sentinels scoring zero, with
  status `unavailable_no_real_model_weights_or_runtime`. They are not
  fair-competed losers.
- The claim supported by the evidence is therefore narrower than "the selected
  architecture ties the strongest fair alternative in the full declared ladder".
  It is: the selected architecture ties the strongest fair alternative **among
  the deterministic persistent baselines that actually ran**, of which S2 is the
  strongest and full transcript replay is co-strongest.
- No real model or corpus was acquired, because no bounded component tournament
  established material value under the current dependency and disk envelope.

## The continuity lane measures duration honestly but loads it thinly

- The 12-hour lane uses monotonic elapsed time across three separate
  operating-system processes, with state recovered only from the owned
  checkpoint. It contains no timestamp jumps and no sleeping.
- The dominant inner loop is a hash iteration. It is real CPU work and real
  wall-clock, but it is not continuity-relevant cognition. The
  continuity-relevant events — goal creation, model replacement, sensor
  interruption, body and tool change, conflicting correction, and a
  history-dependent new task — are a small number of appends per segment, plus
  one background consolidation per minute. The lane demonstrates that owned
  state survives process replacement over 12 hours; it does not demonstrate 12
  hours of cognitive load.
- Segment receipts are trusted if already present on disk, so the lane resumes
  rather than restarts. Nothing cross-checks a receipt against real elapsed
  time, process identity, or the checkpoint chain, so a pre-planted receipt
  would be accepted. The published receipts were produced by a live run from a
  clean checkout of the ready tag, but that provenance rests on operator
  discipline rather than on a check the verifier performs.

## "Zero mutation survivors" is narrower than it sounds

The mutation program reports zero survivors across twenty-two mutations and the
counterfeit program reports all five rejected. Both statements are true. Neither
means what a reader would reasonably assume, and the gap is large enough that it
must be stated rather than left to be discovered.

- **Every mutation is gated on a self-referential layer.** A mutation counts as
  rejected only if both a dossier check and a runtime check reject it. The
  dossier layer — `_valid_dossier`, `_mutate`, `verify_dossier` — exists only
  inside `final_revision_verification.py` and has no callers anywhere else in
  the repository. It builds a dictionary of honesty flags, flips one, and
  confirms the flag reader complains. It is never consulted by principal,
  replication, or hidden-composition scoring.
- **At least eight of the twenty-two runtime harnesses cannot fail.** Five
  (`seed_as_key`, `task_identity_leakage`, `grok_challenge_pack_leaks_answers`,
  `model_support_given_oracle_output`, `body_schema_given_oracle_affordance`)
  construct a local decision trace, inject a key, and ask a verification-only
  gate whether that key is present. `modality_aliasing` builds a list of N
  identical strings and asserts the set is smaller. `same_model_under_multiple_names`
  compares two hardcoded dictionaries. `active_perception_given_free_correct_view`
  deep-copies a payload, sets its cost to zero, and asserts the cost is not
  positive. A system with no defence at all against seed leakage, oracle
  outputs, modality aliasing or free views would pass every one of these.
- **The counterfeit program contains no systems.** `counterfeit_report` merges
  `_mutate` field changes into a synthetic dossier and calls `verify_dossier`.
  No kernel is constructed, no bed is run, nothing is scored. The five
  "counterfeits" are unions of flags. They could not have won the bed if the
  verifier had failed to reject them, because there is nothing there to score.

What the program does genuinely establish, on real production code paths:
checkpoint restore integrity under omitted goals, scene state, model competence
and self model; refusal of a rehashed checkpoint hiding a state reset; refusal
of a checkpoint with a live activation flag; refusal of learning proposals whose
data split is not construction; refusal of knowledge admission without an
undefeated warrant; typed separation of intervention from observation; and the
counterfactual declaration contract. Those are real defences with real refusals.

The frozen mutation list was fixed before principal scoring and cannot be
extended now without changing the target, so this is recorded as a limitation
rather than repaired.

## Independent recomputation re-aggregates; it does not re-execute

`recomputation_matches` in the clean clone reads the stored per-history raw
scores, recomputes the means, the paired effect, the bootstrap interval and the
Holm correction, and compares those against the sealed aggregates. It does not
call `run_discrimination_bed` and does not replay a single episode — all three
lanes recompute in well under a second, which would be impossible otherwise.

This is honest re-aggregation and it does catch arithmetic, interval and
correction errors in the published effects. It does not independently reproduce
the scores themselves. A reader should treat "independent recomputation exact"
as a statement about the statistics, not about the behavior that produced them.

## Scope of the frozen pre-launch authorities

- `SUBSTRATE_FINAL_REVISION_RESOURCE_PARITY.json`,
  `..._HEADROOM_REPORT.json` and `..._STRONGEST_BASELINE.json` record
  construction-scale (moderate pilot) numbers. This is required sequencing: the
  strongest baseline and the headroom must be selected and frozen before
  admission. The principal, replication and hidden-composition results carry
  their own principal-scale parity blocks.
- Structural memory cost is not equal between the event-sourced kernel and the
  flat monolithic state, and neither latency nor peak memory is metered per
  comparison. Parity is established over input information, history, observation
  count, tool access, model access and update-step opportunity.

## Everything else

- The controlled sensorium processes actual arrays, waveforms, depth maps,
  meshes, point clouds, telemetry and filesystem events, but it is not a
  benchmark on an external real-world corpus.
- Mechanism fixtures establish behavior only within their explicit scope. Named
  interfaces, digests, receipts and modality labels are not treated as general
  cognition.
- The Grok review programme reached 32 distinct reviewer roles against a frozen
  minimum of 24 and a preferred target of 48. The role set is frozen in
  `final_revision_config.REVIEW_CELLS`; adding roles after the candidate freeze
  would be a post-launch source edit. Grok grades are not independent external
  validation and are not a primary endpoint.
- Candidate H was admitted, implemented as `H_causal_temporal_ledger` and
  entered in the bounded tournament, where it tied the field at greater declared
  complexity. Candidate I was selected as an engineering default under a
  behavioral tie, not on positive architectural evidence.
- Real desktop, code, document, video, audio and 3D tasks remain for the next
  operator-controlled sandbox campaign. This revision only prepares and smoke
  tests their interfaces.
- External activation is false.
