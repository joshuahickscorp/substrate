# Final Revision Terminal Handoff — 2026-07-28

Uncredited local operations note. Not evidence. Does not modify the frozen
candidate. Supersedes nothing in `terminal_execution_runbook.md`; read both.

## Detached work in flight

| what | where | pid | started | expected |
|---|---|---|---|---|
| `substrate final-revision run` | `/private/tmp/substrate-fr-ready-afeb` (HEAD `afebfa2c` = ready tag, own `.venv`) | 79107 orchestrator, 79173 continuity segment | 14:45:32 | segment receipts ~18:45 / ~22:45 / ~02:45; process exits ~02:50 |
| `final_revision_cleanclone verify` | `/private/tmp/substrate-fr-cleanclone` (HEAD `afebfa2c`, own `.venv`, local `main` branch created from `origin/main`) | 38094 | 15:00 | 1–2h |

Both are `nohup`-detached and survive the controlling session.

Logs (outside every checkout, so `checkout_clean_before_launch` stays true):
`<scratchpad>/runlogs/run.log`, `run.err`, `cleanclone_verify.json`,
`cleanclone_verify.err`, `clone_pytest.log`, `clone_ruff.log`.

Continuity run identity is `e2bee8b55a12d514` =
`digest({ready_commit: afebfa2c…, source_digest: 83fd33d5…, duration_seconds: 43200.0})`.
It is identical in every checkout of the frozen source, so receipts are
portable between them — but do not move them. An earlier orchestrator (PID
36382, launched 12:47 from the main repo when HEAD was still the ready commit)
was killed at ~1h54m into segment 0; its orphaned child and empty run root were
cleared, and the lane was restarted clean at 14:45 rather than resumed, to keep
receipt provenance single-sourced.

## If the run dies overnight

The lane is resumable and the receipts are the state. Recovery is one command
from the worktree, not a restart:

```
cd /private/tmp/substrate-fr-ready-afeb
SUBSTRATE_FINAL_REVISION_CONTINUITY_SECONDS=43200 nohup .venv/bin/substrate final-revision run \
  > <scratchpad>/runlogs/run.log 2> <scratchpad>/runlogs/run.err < /dev/null &
```

`_continuity_lane` skips any segment whose receipt already exists, so it picks
up at the first missing one. The three decisive beds re-execute (~21 min wall,
they run in parallel threads) and reproduce byte-identically apart from
`runtime_seconds` — that has already been demonstrated once, see
`independent_rerun_check.json`. Do **not** hand-write a missing receipt; run the
segment.

Before trusting any receipt, run the consistency verifier:

```
.venv/bin/python runs/substrate/final_revision/continuity_receipt_consistency.py \
  /private/tmp/substrate-fr-ready-afeb/runs/substrate/final_revision/continuity/e2bee8b55a12d514
```

It cross-checks declared duration against the file's own mtime and
`process_started_unix`, consolidation cadence against duration, iteration
throughput against a plausible SHA-256 band, the checkpoint chain across
segments, and non-overlapping monotonic start times. Segment 0 passes all of
them: declared 14400.0s, observed wall 14400.0s, 239/240 consolidations,
577,506 iterations per second, constructs rather than restores.

## Overnight watch design

Two detector bugs were found and fixed in the watch itself; do not reintroduce
them.

- `pgrep -f final_revision_continuity` matches the watching shell, because the
  pattern appears in the watch script's own text. The stall detector built on it
  could never fire. Liveness now uses `kill -0 <orchestrator pid>`, and the
  segment child is found with `pgrep -P <orchestrator pid>` — a parent-child
  relationship cannot self-match.
- A watch that only greps for success markers stays silent through a wedge. The
  current watch emits on segment receipts, the result document, campaign
  completion, campaign stderr, orchestrator death, no-segment-child for four
  consecutive checks, segment CPU time unchanged across three checks, a stall
  threshold of 17400s, and a 30-minute heartbeat carrying segments-done and the
  child's cumulative CPU time. Silence is therefore never ambiguous.

## Do not

- Do not `kill` 79107 or 79173. The lane resumes from segment receipts, but the
  orchestrator does not restart itself.
- Do not create files inside `/private/tmp/substrate-fr-ready-afeb` or
  `/private/tmp/substrate-fr-cleanclone`. Untracked files break
  `checkout_clean_before_launch` and `initial_checkout_clean`.
- Do not copy `GROK_AUTHORITY`, `GROK_INVOCATION_LEDGER` or `REVIEW_ISOLATION`
  back from the ready-tag worktree. The run rewrites them at ready-tag vintage;
  branch HEAD holds the newer versions including the terminal rounds.

## Copy back from the run, and only these

    SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_AUTHORITY.json
    SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json
    SUBSTRATE_FINAL_REVISION_MUTATION_AUTHORITY.json
    SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json
    SUBSTRATE_FINAL_REVISION_COUNTERFEIT_REPORT.json

plus `artifacts/substrate/final_revision/REAL_WORLD_SANDBOX_READINESS_MANIFEST.json`.

Before copying, diff the run's regenerated `PRINCIPAL_RESULT`,
`REPLICATION_RESULT` and `HIDDEN_COMPOSITION_RESULT` against the committed
copies. They are deterministic and must be byte-identical. If they are not, stop
— that is a reproducibility failure, not a copy step.

## Terminal gate — the real defect

`record_clean_clone_verification` is the designed writer of `CLEAN_CLONE`,
`REGENERATION` and `INDEPENDENT_VERIFICATION`, and already emits
`{"complete": true, "separate_clean_process": true}`, which satisfies
`outcome_b_checks["independent_verification"]`. There is no inherent
circularity. What is broken:

1. It refuses unless `clean_report["install"]["passed"] is True`
   (`final_revision_campaign.py:2306`), and the string `install` occurs zero
   times in `final_revision_cleanclone.py`. Every possible clean report is
   refused.
2. `campaign.verify()` writes its inventory report to the same path. Its
   `complete` includes the seven documents only `publish()` writes, so any
   `verify()` after a good record destroys the strong receipt. `publish()` calls
   `verify()` at the end, so this fires on the publication path.
3. Nothing in `final_revision.py` or `cli.py` reaches
   `record_clean_clone_verification`. Zero call sites.

Repair is sealed transition 001, delegated to Grok (`fr-transition`), scoped
`verifier_orchestration_only` per the `substrate-v5-transition-001/002/003`
precedent: `final_revision_cleanclone.py` stays **untouched** so the clean
checkout keeps running ready-tag code; installation becomes an operator action
with its own verified receipt (`clean_clone_install_receipt`, which checks that
`substrate` imports from inside the clone rather than claiming an install);
`verify()` stops clobbering a `separate_clean_process` receipt; a
`record-clean-clone` CLI command is added. No threshold, seed, split, control,
generator or receipt changes. `REQUIRED_DELIVERABLES` stays at 74.

Consequence: the clean report now being produced by the **frozen** verifier is
exactly what the repaired recorder consumes. It does not need re-running.

## Nine unresolved Grok blockers

`publish()` refuses while any credited reviewer's blocking defect is unresolved.
Digests are precomputed in `<scratchpad>/blockers.json`. All nine were raised
against earlier evidence pins:

- 1, 2, 4, 8 — Candidate H "not admitted". Superseded:
  `CANDIDATE_H_ADJUDICATION.json` records 4 independent proposals, selection of
  the Intervention-Indexed Dual-Timeline Causal Ledger, implementation as
  `H_causal_temporal_ledger`, and admission to the bounded tournament where it
  tied at greater declared complexity. Blocker 4's specific claim ("plane
  counters are not a causal-temporal architecture") must be checked against the
  shipped implementation, not the placeholder it criticised.
- 1, 3, 5 — Grok minimum incomplete. Superseded: 32 distinct roles ≥ minimum 24;
  `minimum_complete`, `prefreeze_complete`, `terminal_complete` all true.
- 6, 7 — decisive receipts absent. Partly superseded now; the rows citing
  `LONG_CONTINUITY_*` cannot be applied until the campaign lands.
- 9 — publication reviewer, Outcome B machine gates. Resolve last, after every
  other receipt exists.

Apply with `substrate final-revision resolve-grok-review BATCH.json`. The batch
must pin a full 40-hex commit that actually contains every cited evidence file,
so commit the copied evidence first.

Drafted and mechanically validated:
`runs/substrate/final_revision/blocker_batch_1.json` (8 rows, every defect
digest matches the ledger, every cited path exists — applyable as soon as there
is a commit to pin) and `blocker_batch_2.json` (blocker 9 alone, cites seven
files that do not exist yet). Set `resolution_commit` from `PENDING_COMMIT` to
the real commit before applying. Full reasoning in `blocker_dispositions.md`.

Batch 2 originally cited `FINAL_CLASSIFICATION.json` as evidence; that was
removed because `publish()` writes it and `publish()` is gated on this very
blocker. Do not re-add it.

All nine dispositions are `superseded_by_later_evidence` with an explicit
argument for why none is `accepted_terminal_limit`: these nine strings are
process/status claims about H admission, Grok minimum completion and missing
receipts, not permanent scientific limits. The scientific limits were
dispositioned on other defects of the same invocations. Check that argument
before applying — it is the load-bearing claim of the batch.

## Verified limitations that must survive into the terminal report

Established today by reading source, not by accepting a report:

- `full_transcript_replay` is scored through the same `_kernel_answers` function
  as the selected candidate (`final_revision_experiment.py:869`). P1 against
  transcript replay is zero by construction and carries no information.
- Neither `_kernel_answers` nor `_monolith_answers` emits answer key 7, while
  `expected` requires it (`final_revision_experiment.py:865`). The entire
  ~0.125 oracle headroom is a class neither system attempts. The bed is formally
  non-saturated; the residual is unattempted capacity, not a contested frontier.
- `disconnected_model_ensemble`, `stateless_model_router`,
  `largest_model_always` and `all_models_always` all receive
  `unavailable_model_answers = {}` and score zero as availability sentinels.
- The P3 tie is real, not a scoring bug: an independent perturbation of either
  system moves the reported effect to ±0.153 with a non-degenerate CI.

Established by the mutation and red-team cells, each spot-checked directly:

- The mutation program's rejections come from two layers. The dossier layer
  (`_valid_dossier` / `_mutate` / `verify_dossier`) has **zero callers outside
  `final_revision_verification.py`** — verified by grep — and is never consulted
  by scoring. At least eight of the twenty-two runtime harnesses assert
  properties true by construction: five inject a key into a locally built
  decision trace and ask a verification-only gate whether the key is there;
  `modality_aliasing` builds N identical strings and asserts the set is smaller;
  `same_model_under_multiple_names` compares two hardcoded dicts;
  `active_perception_given_free_correct_view` sets a deep copy's cost to zero
  and asserts it is not positive.
- `counterfeit_report` constructs no systems at all. It merges `_mutate` field
  changes into a synthetic dossier and calls `verify_dossier`. No kernel, no
  bed, nothing scored.
- `recomputation_matches` re-aggregates stored `raw_history_scores`; it never
  calls `run_discrimination_bed`. All three lanes recompute in under a second.
- `_terminal_documents()` grants Outcome B from self-asserted fields. A red-team
  pass forged a passing terminal state in scratch in under a second, and faked
  the 12-hour lane in 0.02s by writing three segment receipts with self-declared
  durations.
- `io.contains_true_activation` matches only the exact lowercase key.
  `Activation`, `ACTIVATION` and `external_activation` are not scanned.
- `thresholds_preserved` and `challenges_preserved` are hardcoded `True`
  literals at `final_revision_campaign.py:2415-2416`, never computed.

Instrument liveness was checked rather than assumed and is sealed at
`runs/substrate/final_revision/instrument_liveness_check.json` with its script
beside it: control `0.0 [0.0, 0.0]`, S2 degraded `+0.0903 [0.0694, 0.1111]`,
candidate degraded `-0.0903 [-0.1111, -0.0694]`.

One earlier Grok finding was checked and **rejected**: that
`RESOURCE_PARITY.json` / `HEADROOM_REPORT.json` / `STRONGEST_BASELINE.json` are
"pilot-scale, not principal-scale" is correct sequencing, not a defect — the
strongest baseline and headroom must be frozen on construction data before
admission. A second was rejected in the docs work: annotated-tag object SHAs
reported as commit drift.

## Remaining sequence

1. Campaign completes → diff and copy the five documents + manifest → commit.
2. Land sealed transition 001 (own tag `substrate-final-revision-transition-001`
   and PR, per v5 precedent) with `SUBSTRATE_FINAL_REVISION_TRANSITION_001.json`.
3. `final_revision_cleanclone regenerate` in the clean clone →
   `substrate final-revision record-clean-clone CLEAN.json REGEN.json CLONE_ROOT`.
4. Resolve blocker batches 1 and 2 → commit.
5. Terminal Grok review round at the terminal evidence commit. Generator ready at
   `<scratchpad>/make_terminal_contracts.py` (12 cells, uses the sanctioned
   `final_revision_grok.prompt_for`). Record with
   `substrate final-revision record-grok-build TASK_DIR CONTRACT`. Expect fresh
   blocking defects on the class-7 and transcript-arm findings; the honest
   disposition for those is `accepted_terminal_limit`, not a post-freeze repair.
6. `substrate final-revision publish` → last 7 documents.
7. `agent/substrate-final-revision-terminal`; tags
   `substrate-final-revision-terminal`, `substrate-final-architecture-1`,
   `substrate-real-world-sandbox-ready-1`; PR #54 out of draft.

Expected outcome is **B** — `substrate_final_revision_complete`,
`internal_functional_nous_claim_closed`, `real_world_sandbox_ready`. Not A: P3
is an exact tie, so no SESOI-scale advantage exists to claim. Activation stays
false.
