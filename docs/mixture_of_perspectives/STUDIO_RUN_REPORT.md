# STUDIO RUN REPORT: the accumulating scoreboard for the goal loop

This is the single accumulating artifact the goal loop (`STUDIO_GOAL_PROMPT.md`) writes each wave, so
a dead session loses at most one wave and the next session resumes from here. House style: no em or en
dashes. A tie is a null. No positive enters a doc without an independent adversarial verification pass.

IMPORTANT PROVENANCE: waves 0 to 3 in the log below ran on the M3 Pro laptop (18 GB), NOT the M1 Ultra
Mac Studio. They executed exactly the goal-loop work that is feasible without the Studio's resources
(the facet-12 predictor lane, the b5 closure, the machine-agnostic half of WAVE 0). The Studio-specific
WAVE-0 steps are marked PENDING below and MUST run on the M1 Ultra before the spine. A session on the
real Studio resumes from the "Next move" section.

## WAVE 0 status (partial: machine-agnostic half done on M3 Pro)

- [x] Full gates green on this box: acceptance 10/10 (full pytest suite, ruff lint + format, mypy 160
      files, E1 forget-then-retain gate, diagnostics, I4 table, campaign queue dry-run, one toy Tier-C
      leg, registry 109 experiments). check_docs green.
- [x] M3-Pro Wave 0 rerun on 2026-07-08: pytest green (704 tests after the doctor-floor test),
      docs gate green, acceptance 10/10, DR1 smoke green, profile listing green, and profile-aware
      doctor green once the filesystem reclaimed above the `m3pro-local-max` 60 GB floor.
- [x] STUDIO_RUN_REPORT.md created (this file) and added to `scripts/check_docs.py` CANONICAL_MD.
- [x] Encode microbench, M3-Pro baseline: MPS vs single-worker CPU on 8 real clips (V-JEPA 2 ViT-L,
      64 frames / 256 px, transformers path). RESULT: CPU single-worker 13.69 s/clip (stable, 8 clips);
      MPS ran but took 821 s/clip on one clip (about 60x slower, memory-pressured at 18 GB). NOTE: the
      hard "Invalid buffer size" MPS overflow documented in STUDIO_HANDOFF did NOT reproduce this
      session on this path; MPS instead completed but paged badly at 18 GB. CPU is the clear M3-Pro
      winner. The Studio MUST re-measure MPS at 128 GB and 14 to 16 parallel CPU workers (128 GB
      removes the paging pressure and could make MPS the winner: MEASURE, never assume).
- [ ] PENDING (M1 Ultra): rebuild the 64-clip real cache to >= 1000 clips. The 64-clip cache
      (`data/cache/vjepa2_vitl_fpc64_256_real`, count 64, acc 1.0) is present and sufficed for all
      M3-Pro-feasible science; the >= 1000-clip rebuild needs the Studio's encode throughput.
- [ ] PENDING (M1 Ultra): the Studio-scale MPS-vs-16-worker-CPU microbench above at 128 GB.

WAVE 0 is therefore COMPLETE for the machine-agnostic half and its two remaining steps are the two
genuinely Studio-gated measurements. A session on the M1 Ultra completes those, then proceeds to the
spine.

## Axis scoreboard (the four north-star axes)

| Axis | Laptop proven | Studio ceiling | Status |
|---|---:|---:|---|
| Falsification | 10 | 10 | Held and completed (vacuous gate retired, 4 over-claims killed). B5 multi-seed re-encode is the one Studio caveat. |
| Abstraction | 6 | 9 | Four controlled wins (systematicity, pairwise + 3-way cross-substrate analogy), all on the shape factor. Beyond 6 needs DR1 real video (gates b1/b2/b3). |
| Density | 6 | 9 | Matched-compute mixture win on a constructed task (10/10 seeds). Natural-complementarity at scale needs DR1. |
| Moldability | 5 | 8 | Frozen-encoder-capped (proven). PR9 real stream then Process C are the levers. |
| Overall (mean) | ~6.75 | ~9.0 | Studio theoretical ~9.3 including the Part-2 frontier. |

Source of the laptop numbers and their adversarial proofs: `RESULTS_LEDGER.md`,
`HANDOFF.md` verdict ledger.

## Facet scoreboard (Part 2, the Studio-native frontier)

| Facet | Score | Status |
|---|---:|---|
| 12 world-model rollouts | 0 | MEASURED and walled provisionally on the M3 Pro (`RESULTS_LEDGER.md`): the frozen predictor carries a real but sub-usable rollout signal; motion-tracking is a wall/null-by-ill-posedness on synthetic near-static clips. Licensed re-test: real moving video + a readout adapter. |
| 13 closed-loop / active | 3 | Not started; needs live-encoder throughput (Studio). |
| 14 real corpora | 0 | Not started; needs 8 TB disk + licensing (Studio). |
| 15 perspective ecology | 2 | Not started; needs 10+ resident encoders (128 GB). |
| 16 developmental daemon | 0 | Not started; needs week-scale always-on (Studio). |
| 17 trainable capacity | n/a | A doctrine decision, deliberately not graded. |

## Open levers (all now Studio-gated; the M3-Pro-feasible backlog is CLOSED)

| Lever | Axis | Resource gate |
|---|---|---|
| DR1 real bound-attribute video cache (+ caption acceptance gate, local-VLM arm) | abstraction, density, coverage, plurality | 21 s/clip encode throughput, real corpora, 8 TB disk. THE spine unblocker. |
| PR9 continual-backprop on a real long stream (certificate-guarded) | moldability | 128 GB-resident long stream; decides Process C licensing. |
| Process C (1 to 10M object-centric module on dense tokens) | moldability | dense cache + GPU training. Gated by PR9 kill-switch. |
| Dense-token cache (~1.7 TB) | facet 8 | 8 TB disk. |
| Encoder-scale atlas (ViT-H/g + DINOv2 + VideoMAEv2) | facet 7 | disk + 128 GB residency; pull ViT-H/g stubs. |
| B5 multi-seed re-encode + 30-seed retrofits | falsification, statistical power | pure compute at Studio scale; LAST per doctrine. |
| Facets 13 to 17 (closed-loop, corpora, ecology, daemon) | Part-2 frontier | ride the spine's artifacts; each Studio-gated. |

M3-Pro-feasible backlog, CLOSED this session: facet 12 measured (waves 1 to 2); the pre_studio
candidate positives resolved (e7_sparse architectural, ex5 Adam-artifact refuted, ex2 promoted, b5
underpowered null, wave 3); no open non-vacuous frozen-random gap remains (`DOCTRINE_SYNTHESIS.md`).

## s/clip benchmarks

| Path | s/clip | Notes |
|---|---|---|
| M3-Pro CPU, single worker | 13.69 | 8 real clips, V-JEPA 2 ViT-L, 64 frames / 256 px, transformers path; stable ~13.5 s each. |
| M3-Pro MPS | ~821 | ran on one clip (no hard buffer error this session), 60x slower than CPU, memory-pressured at 18 GB. |
| Studio (M1 Ultra), 14 to 16 CPU workers | PENDING | ~2 s/clip aggregate projected off 13.69 single-worker; MEASURE. |
| Studio (M1 Ultra), MPS at 128 GB | PENDING | 128 GB removes the paging pressure; could win; MEASURE, never assume. |

<!-- STUDIO-WAVE0-AUTO:START -->
## Studio Wave 0 Auto Receipt

- Status: INCOMPLETE.
- Launch profile: m3pro-local-max; hardware Apple M3 Pro: 6P/6E cores, 19.3 GB unified, mps=True; disk 63.2 GB free of 494 GB at repo (basic writeability); m3pro-local-max: 63.2 GB free, min 60 GB (profile floor ok).
- MPS: torch 2.12.1; mps available=True built=True; encoders: 5 configs: vjepa21_vitb(d=768,deferred), vjepa21_vitl(d=1024,deferred), vjepa2_vitg(d=1408,real-ready), vjepa2_vith(d=1280,real-ready), vjepa2_vitl_fpc64_256(d=1024,real-ready); cache path: write+delete ok under data/cache.
- Disk recovery: ok (60.045 GB free vs floor 60.0 GB; 3 safe candidate(s), 2293 protected candidate(s), would delete 102.3 MB).
- Transfer check: ok (48/48 checks passed).
- Daemon gates: missing daemon state.
- Encode winner: cpu; CPU 16.191 s/clip; MPS available-not-timed (skip_mps; the Studio re-runs to time MPS at 128 GB) s/clip; schedule launch not-ok.
- Memory envelope: process RSS peak None GB; min system available None GB; MPS driver peak None GB.

<!-- STUDIO-WAVE0-AUTO:END -->


<!-- STUDIO-SCORECARD-AUTO:START -->
## Studio Scorecard Auto Receipt

- Status: INCOMPLETE.
- Launch: pending (Wave 0 report incomplete).

| Axis | Status | Evidence | Current score | Studio target |
|---|---|---|---:|---:|
| falsification | held | pre-Studio falsification discipline held; positives still require verdict gate plus artifact bundle | 10 | 10 |
| abstraction | pending | missing DR1 adversarial verification receipt | 6 | 9 |
| moldability | pending | PR9 result is not the DR1 real cache (data/cache/vjepa2_vitl_fpc64_256_real); local smoke is non-scoring | 5 | 8 |
| density | pending | dense/atlas cache gate is blocked; real and matched random-init dense caches are not ready | 6 | 9 |
| durability | pending | missing indexes=['wave0', 'dr1', 'pr9', 'atlas', 'spine']; failing indexes=[] | 7 | 10 |

- Next spine command: `.venv/bin/python -m scripts.studio daemon run --plan runs/studio_spine/wave0_daemon_plan.json --out-dir runs/studio_wave0 --profile studio-m1ultra --execute`.
- Process C: pending (Process C license gate is undecidable: undecidable).
- Blocking receipts: launch:Wave 0 report incomplete; abstraction:missing DR1 adversarial verification receipt; moldability:PR9 result is not the DR1 real cache (data/cache/vjepa2_vitl_fpc64_256_real); local smoke is non-scoring; density:dense/atlas cache gate is blocked; real and matched random-init dense caches are not ready; durability:missing indexes=['wave0', 'dr1', 'pr9', 'atlas', 'spine']; failing indexes=[]; next:wave0_run; process_c:Process C license gate is undecidable: undecidable.

<!-- STUDIO-SCORECARD-AUTO:END -->









## Wave log

- WAVE 0 (M3 Pro, 2026-07-03): gates 10/10, this scoreboard created, encode microbench recorded; the
  1000-clip rebuild and the 128 GB MPS microbench remain PENDING for the M1 Ultra. Artifact: this file.
- WAVE 1 (M3 Pro, 2026-07-03, commit d4abb5a): facet 12 DR13 predictor-fidelity on the real V-JEPA 2
  predictor. VERDICT NULL on usability (real beats all 3 controls by seed CI at every horizon but by
  only 5 to 7 percent; sub-usable). Instrument `scripts/mop_dr13_predictor_fidelity.py`. Adversarially
  verified 6/6. `RESULTS_LEDGER.md`.
- WAVE 2 (M3 Pro, 2026-07-03, commit f447707): facet 12 decodability-retention (does the rollout track
  object position, beating persistence?). VERDICT WALL / null-by-ill-posedness (synthetic clips move
  sub-patch; the build's motion gate was falsified by the adversarial re-derive). New finding: position
  survives the rollout in a shifted sub-space (readout-adapter target). `RESULTS_LEDGER.md` s11.
- WAVE 3 (M3 Pro, 2026-07-03, commit 4c818e4): stack-informed ORIENT audit + closed b5_degeneracy (the
  last open non-vacuous frozen-random gap) as an underpowered null. Net verdict: axis-moving M3-Pro
  science is exhausted; every remaining axis-mover is Studio-gated. `DOCTRINE_SYNTHESIS.md`,
  `STUDIO_HANDOFF.md` update.

- WAVE 4 (M3 Pro, 2026-07-03, turnkey plan Tier 1 to 2): de-risked the Studio spine before it spends
  Studio time. DR1 caption acceptance gate smoke (`scripts/studio/dr1_smoke.py`, passes on carried
  factors, refuses on a non-recoverable one); PR9 + atlas re-smoked on HEAD (reinit fires, honest
  nulls, atlas withholds on partial caches); the studio rehearsal passes 9/9 stages on HEAD; and
  preregistration frozen with 7 NULL/survival cards (facet 12, b5, ex2, e7, ex5, ex13, ex15). Plan and
  running checklist: `STUDIO_TURNKEY_PLAN.md`.
- WAVE 5 (M3 Pro, 2026-07-03, turnkey plan Tier 3 to 4): facet-12 readout-adapter scaffold
  (`scripts/mop_dr13_readout_adapter.py`, `--clip-dir` real video), preregistered + smoke-run: the
  adapter halves the visible-slot representational gap but does not transfer to the rollout (naive
  visible-slot adapter insufficient; Studio fits on rollout predictions). ViT-H / ViT-g encoder
  readiness verified (configs correct, weights are stubs to pull, vitg needs a prefer_real flag);
  turnkey pull + flip steps recorded in STUDIO_TURNKEY_PLAN.md. Remaining: T4.1 encode auto-select,
  T5.1 facet entry points (lower priority, ride the spine).

- WAVE 6 (M3 Pro, 2026-07-03, potential re-audit + gap closure): an adversarial re-audit put honest
  M3-Pro completion at 86 percent (science ~90, enablement ~72) and caught one over-claim (the facet-12
  verifier was cited but absent from the repo) plus a cross-cutting durability hole (the evidence base
  was 100 percent gitignored). Closed: the load-bearing verdict evidence is now git-tracked via a
  targeted .gitignore negation; `scripts/mop_dr13_verify.py` restored (6/6 PASS in-repo); Tier 4.1 encode
  auto-select built (`scripts/mop_encode_autoselect.py`); PR9 bare-smoke cache fallback and the stale
  rehearsal-stage phrasing fixed. Only axis-moving science (Studio-gated) and the T5 facet stubs remain.
- WAVE 7 (M3 Pro, 2026-07-08, laptop Wave 0 under the new deep audit): reread the governing audit
  stack and reran the laptop gates. Initial disk was below the `m3pro-local-max` profile floor (about
  12 GB free), so no science launched. During gate execution the filesystem reclaimed above the floor
  (64.5 GB free), enabling the guarded rehearsal: `scripts/studio_pipeline.py local-max --download-gb
  10 --time-min 90 --cache-clips 64` passed all 12 stages in
  `runs/studio_pipeline/local_max_20260708_075245`. A full Tier-C laptop run was also probed and
  correctly stopped at the run-count gate (263 run-units vs cap 64). Custom-build pass: `studio_doctor`
  now separates basic disk writeability from the active profile floor and supports an explicit
  `--profile` gate; on this laptop it reports `profile_floor` against `m3pro-local-max`, and on the
  Studio it can enforce `--profile studio-m1ultra`. Durability check: the load-bearing receipt set
  remains git-tracked (`runs/pre_studio/RESULTS_PRE_STUDIO.md`, close/census JSONs, and
  `runs/mot/dr13_predictor_fidelity.json`). Verdict: laptop Wave 0 green, local-max rehearsed, no
  axis-moving science left locally.
- WAVE 8 (M3 Pro, 2026-07-08, custom latent-cache data plane): built the first Studio cache receipt
  layer in `src/mop/substrate/cache_manifest.py`. It writes `cache_manifest.json` beside any
  `LatentStore`, records array fingerprints for pooled or dense memmaps, persists optional
  `factors.json` and `splits.json`, records an encoder config hash, and builds a small columnar index
  for factors/splits/arrays. `cache_tool.py` now has `manifest` and `validate-manifest` subcommands,
  and `validate_cache()` validates `cache_manifest.json` when present while keeping old caches valid.
  Tests cover clean receipts, sidecar tampering, factor length mismatches, duplicate/out-of-range or
  overlapping splits, and cache-tool integration. Verdict: no axis score changes locally, but DR1 and
  dense-cache artifact durability are stronger before the Studio writes terabyte-scale evidence.
- WAVE 9 (M3 Pro, 2026-07-08, custom encode scheduler): built `src/mop/studio/encode_scheduler.py`,
  a profile-aware launch planner for the Studio Wave-0 encode benchmark. It consumes measured
  CPU/MPS s/clip, applies profile-specific CPU worker defaults (`studio-m1ultra` = 16), estimates
  pooled or dense cache bytes, enforces both start and post-cache free-disk floors, checks wall clock,
  chooses MPS vs parallel CPU, and emits checkpoint cadence plus a next command. `scripts/mop_encode_autoselect.py`
  now writes `runs/mot/encode_schedule.json` alongside the backward-compatible `encode_device.json`.
  Tests cover CPU-vs-MPS choice, dense-cache disk refusal, clip-cap clamping, and failed-MPS parsing.
  Verdict: no axis score changes locally, but the M1 Ultra microbench now feeds a falsifiable launch
  contract instead of a manual operator choice.
- WAVE 10 (M3 Pro, 2026-07-08, falsification DSL / null-card generator): built
  `src/mop/falsification/null_cards.py`, `scripts/null_card_tool.py`, and
  `proof/NULL_CARDS/null_card.schema.json`. The tool generates draft null/survival cards directly
  from `registry/experiments.yaml`, validates the fenced YAML block in existing cards, supports
  strict mode that refuses TODO placeholders, and tolerates historical cards whose prose values
  contain colons. The generated contract carries the null hypothesis, baseline, falsifier/ablation,
  metric, probe dependency, seed threshold, provenance tag, verdict, taxonomy slot, and raw-run
  receipt. Tests cover registry generation, round-trip rendering, strict placeholder refusal,
  existing-card validation (`ex13_long_stream`), bad enums, missing probe dependencies, and schema
  fields. Verdict: no axis score changes locally, but Studio DR1/PR9 claims now have a code path
  for preregistered null-card receipts before any positive enters docs.
- WAVE 11 (M3 Pro, 2026-07-08, custom PerspectiveAdapter): built
  `src/mop/perspectives/adapter.py` and `tests/unit/test_perspective_adapter.py`. The new
  `PerspectiveAdapter` layer sits above `SubstrateAdapter`: tensor/store/substrate-backed views all
  yield a `PerspectiveBatch` with features, referent ids, factors, and provenance flags; the
  `PerspectiveMatrix` builder aligns every arm by referent id and refuses mismatched referent sets; and
  `perspective_audit()` names missing matched controls, supervised arms, derived arms, dimensions, and
  licenses. Verdict: no axis score changes locally, but facet-6/facet-15 perspective plurality now has
  a reusable evidence contract for DR1's identical-referent multi-arm cache instead of an informal
  collection of feature arrays.
- WAVE 12 (M3 Pro, 2026-07-08, gated Process C dense-token module): built
  `src/mop/process_c/dense_tokens.py` and `tests/unit/test_process_c_dense_tokens.py`. The module
  implements the sanctioned Process C pilot as a shell-side object-centric slot module over frozen dense
  tokens, plus a dense-without-slots mean baseline, matched-baseline width selection,
  binding-specificity reporting, and a `process_c_budget_report()` that refuses unlicensed runs and
  enforces the default 1 to 10M parameter cap. Verdict: no Process C science was run and no license is
  implied; PR9/DR1 must still fire the gate. But if the Studio licenses Process C, the first dense-token
  arm now has tensor mechanics, controls, and budget discipline in place.
- WAVE 13 (M3 Pro, 2026-07-08, long-run daemon): built `src/mop/studio/long_run.py`,
  `scripts/studio_daemon.py`, and `tests/unit/test_long_run_daemon.py`. The daemon reads a JSON job
  plan, dry-runs by default, enforces the active profile disk floor before each job, writes resumable
  `daemon_state.json`, skips completed jobs on resume, emits heartbeat events during subprocesses,
  stores per-job logs, and stops on disk-block or command failure. Verdict: no long Studio job was
  launched locally, but facet-16/week-scale execution now has a reusable supervisor instead of relying
  on session hygiene.
- WAVE 14 (M3 Pro, 2026-07-08, Studio transfer checklist): built `src/mop/studio/transfer_check.py`,
  `scripts/studio_transfer_check.py`, and `tests/unit/test_transfer_check.py`, and made the long-run
  daemon template start with `transfer_check`. The checklist is read-only: it enforces the
  `studio-m1ultra` profile, checks the governing audit/docs/scripts, parses the null-card schema,
  reports git branch/head/dirty state, confirms the pre-Studio durable receipts are present and
  git-tracked, and validates any cache manifests already present. Local smoke with `--allow-dirty`
  produced a 28/28 transfer receipt. Verdict: Studio Wave 0 now has an executable transfer receipt
  before doctor/full gates/microbench; no Studio science was run locally.
- WAVE 15 (M3 Pro, 2026-07-08, Studio Wave-0 daemon plan): extended the `scripts/studio_daemon.py`
  template into a full Wave-0 gate plan: transfer check, profile-aware doctor, profile listing, docs
  gate, full acceptance, DR1 smoke, and the `studio-m1ultra` encode microbench planning for 1000
  clips. Verdict: the Studio still must run it with `--execute` on the M1 Ultra to produce actual
  s/clip and memory-envelope evidence, but the Wave-0 command order is now an executable plan rather
  than a prose checklist.
- WAVE 16 (M3 Pro, 2026-07-08, encode memory-envelope receipt): built
  `src/mop/studio/memory_envelope.py` and wired `scripts/mop_encode_autoselect.py` so the Wave-0
  microbench records process RSS, max RSS, system available memory, and MPS allocator counters when
  available. The scheduler carries the same envelope into `encode_schedule.json`. If model files are
  absent, the script now writes a blocked JSON receipt with the memory envelope and exits nonzero,
  rather than dying before artifact creation. Local smoke hit the expected laptop wall (V-JEPA files not
  cached) and still produced valid `mop-memory-envelope/v1` receipts. Verdict: no Studio s/clip was
  measured locally, but the Studio run now has the required memory-envelope instrument.
- WAVE 17 (M3 Pro, 2026-07-08, Wave-0 report synthesizer): built `src/mop/studio/wave0_report.py`,
  `scripts/studio_wave0_report.py`, and `tests/unit/test_wave0_report.py`, and added the report step to
  the daemon template. The synthesizer reads transfer check, daemon state, encode device, and encode
  schedule receipts, writes `runs/studio_wave0/wave0_report.json`, and idempotently inserts a bounded
  auto block into `STUDIO_RUN_REPORT.md` with actual CPU/MPS s/clip, winner, blocked reasons, process
  RSS peak, minimum system memory, and MPS driver peak. Verdict: no Studio evidence was fabricated, but
  the M1 Ultra Wave 0 now has an executable path from receipts to the required scoreboard entry.
- WAVE 18 (M3 Pro, 2026-07-08, Studio-native lane manifest): built `src/mop/studio/native_lanes.py`,
  `scripts/studio_native_lanes.py`, and `tests/unit/test_native_lanes.py`. The manifest turns the Part 2
  lanes into receipt-bearing entries: runnable commands for DR13 real rollout tests, hosted-corpora
  planning/acquire, and PR9 on the DR1 cache when their inputs exist, and explicit blocked reasons for
  live-encoder doctrine, perspective ecology, and Process C licensing. Heavy lanes require
  `--include-heavy` plus concrete input paths, and acquisition requires an inspected `--plan-path`.
  Verdict: no new science, but the remaining Studio-native levers are now schedulable or mechanically
  walled instead of living only in prose.
- WAVE 19 (M3 Pro, 2026-07-08, verdict gate): built `src/mop/falsification/verdict_gate.py`,
  `scripts/verdict_gate.py`, and `tests/unit/test_verdict_gate.py`. The gate writes
  `mop-verdict-gate/v1` receipts from a strict null card, raw run JSON, and optional verifier JSON. A
  `PUBLISH-POSITIVE` verdict is refused unless the verifier receipt is separate from the run receipt and
  exposes both a pass flag and an independent/adversarial flag; nulls and ties need only the strict card
  plus raw JSON. CLI smoke against `ex2_latent_planning` intentionally refused the historical positive
  without a verifier receipt. Verdict: no new result, but the pre-ledger falsification rule is now
  executable instead of relying on memory.
- WAVE 20 (M3 Pro, 2026-07-08, artifact bundle/index): built `src/mop/studio/artifact_bundle.py`,
  `scripts/studio_artifact_bundle.py`, and `tests/unit/test_artifact_bundle.py`, then generated
  `proof/ARTIFACT_INDEX/pre_studio.json` as the first durable receipt index. The tool hashes receipt
  artifacts, validates JSON, reports git tracking, and can copy small untracked text receipts into a
  proof bundle while refusing oversized or non-text artifacts. The transfer checklist now requires the
  tool and the pre-Studio index path before Studio compute starts. Verdict: no new science, but the
  audit's gitignored-artifact risk is now a machine-checkable durability gate for Wave 0 and later waves.
- WAVE 21 (M3 Pro, 2026-07-08, daemon pre-ledger contract): extended `src/mop/studio/long_run.py`,
  `scripts/studio_daemon.py`, and `tests/unit/test_long_run_daemon.py`. Daemon plans now reject any
  `positive-ledger` job unless a `verdict-gate` job and an `artifact-bundle` job appear earlier in the
  plan; `scripts/studio_daemon.py validate --plan <plan.json>` checks this without running commands.
  Dry-run states also no longer count as completed when a later run is executed, so a rehearsal cannot
  accidentally skip the real Wave. Verdict: no science was run, but week-scale Studio jobs now carry the
  falsification and durability gates before any positive ledger mutation can happen.
- WAVE 22 (M3 Pro, 2026-07-08, DR1 schedule bridge): added `src/mop/studio/dr1_schedule.py`,
  `scripts/studio/dr1_schedule_plan.py`, and `tests/unit/test_dr1_schedule.py`, then wired
  `scripts/studio/dr1_curate_bound_video.py --device` so the measured CPU/MPS winner from
  `encode_schedule.json` is honored by the actual DR1 legs. The bridge refuses blocked schedules, splits
  DR1 encode legs by checkpoint cadence, emits a caption-gate job, merge job, A6-guard job, and optional
  long-run daemon plan. Verdict: no real video was encoded on the laptop, but DR1 now consumes the Studio
  schedule as a receipt instead of relying on a copied command string.
- WAVE 23 (M3 Pro, 2026-07-08, DR1 PerspectiveMatrix receipt): added
  `src/mop/studio/dr1_perspectives.py` and `tests/unit/test_dr1_perspectives.py`, and wired
  `dr1_curate_bound_video.py --merge --source ...` to write `perspective_matrix_receipt.json`.
  When the root merged LatentStore exists, the receipt builds the existing `PerspectiveMatrix` over
  `vision_vjepa2` and `caption_text`, verifies identical referents, records factor counts, and exposes
  the `perspective_audit()` missing-control surface. If only shard-order sidecars exist, it writes a
  blocked receipt instead of implying a merged matrix. Verdict: no science was run, but DR1 plurality now
  has a verified-or-walled receipt before any alignment or facet-15 claim can lean on it.
- WAVE 24 (M3 Pro, 2026-07-08, AlignmentSuite table): extended `src/mop/diagnostics/alignment.py` and
  `tests/unit/test_mot_shared_modules.py`. `alignment_suite(x, y)` keeps the pair report, while
  `alignment_suite({tag: tensor})` now emits a `mop-alignment-suite/v1` table with self-geometry,
  pairwise CKA/kernel-CKA/RSA/neighborhood overlap, row-shuffle p-values, metric matrices, and doctrine
  warnings when no random-encoder control tag is present. Verdict: no Studio evidence was scored, but the
  DR1 PerspectiveMatrix now has a reusable falsifiable alignment table instead of ad hoc pair calls.
- WAVE 25 (M3 Pro, 2026-07-08, claim ledger daemon plan): added `src/mop/studio/claim_plan.py`,
  `scripts/studio_claim_plan.py`, and `tests/unit/test_claim_plan.py`. Studio claim updates can now be
  emitted as daemon plans that run `verdict-gate`, then `artifact-bundle`, then the requested ledger
  command; positive claims mark the final job as `positive-ledger`, so the daemon's static contract
  rejects any hand-built positive update that omits falsification or durability gates. Verdict: no claim
  was ledgered, but the path from run receipt to doc mutation now has an executable pre-ledger spine.
- WAVE 26 (M3 Pro, 2026-07-09, exact Wave-0 launch receipt closure): integrated the disk-recovery
  receipt into the Studio transfer/daemon/report path and made the doctor emit JSON for
  `runs/studio_wave0/studio_doctor.json`. The Wave-0 daemon template now runs transfer check,
  disk recovery, JSON-backed doctor, profiles, docs gate, acceptance, DR1 smoke, encode microbench, and
  report synthesis. `studio_wave0_report.py` now records launch profile, hardware, disk/profile floor,
  MPS availability, encoder availability, cache write path, transfer status, disk-recovery summary,
  s/clip, and memory envelope in one bounded block. PR9 now writes a `mop-pr9-run-state/v1` receipt
  beside the long-stream result, recording resumable leg inventory and final verdict state; the artifact
  bundle now has a `pr9` preset for the result, run-state receipt, verdict ledger, and PR9 null card.
  Verdict: no Studio science was run on
  the laptop; this closes the remaining local launch-prep gap for exact Wave-0 receipts and PR9
  interruption/resume durability.
- WAVE 27 (M3 Pro, 2026-07-09, DR1 adversarial verification spine): added
  `src/mop/studio/dr1_verifier.py`, `scripts/studio/dr1_verify.py`, and
  `tests/unit/test_dr1_verifier.py`. The DR1 verifier reads the merge manifest, per-leg caption-gate
  sidecars, clip hashes, PerspectiveMatrix receipt, and A6 residual guard receipt; it exposes
  `independent` and `adversarial` flags, but sets positive `passed/all_ok` only when integrity is clean
  and the decisive A6 condition survives. `dr1_curate_bound_video.py --merge` now passes `--source`
  through to the PerspectiveMatrix writer, and `--a6-guard` writes `a6_residual_guard.json` beside the
  cache. The DR1 daemon plan now ends with `dr1_verify`; the artifact bundle has a `dr1` preset; and
  `proof/NULL_CARDS/mop_dr1_video_cache.md` preregisters the DR1 cache wall/null. Verdict: no Studio
  video was encoded locally, but a DR1 positive can no longer enter the spine without an independent
  adversarial receipt and a matching null card.
- WAVE 28 (M3 Pro, 2026-07-09, staged Studio spine plan): added
  `src/mop/studio/spine_plan.py`, `scripts/studio_spine_plan.py`, and
  `tests/unit/test_spine_plan.py`. The new `mop-studio-spine-plan/v1` receipt writes the Wave 0
  daemon subplan, then stages Wave 0 execution, Wave 0 bundling, DR1 source-card/intake/schedule/run/bundle, PR9
  run/bundle, dense-cache planning, paired dense real/random-init cache validation, the full atlas run, atlas
  bundling, and a final spine artifact index. The atlas command deliberately omits `--allow-partial`;
  dense/atlas stops at a missing or mismatched dense cache pair instead of promoting a partial grid. Verdict:
  no Studio science was run locally, but the real run is now a single ordered, resumable spine with
  durable receipt checkpoints rather than scattered commands.
- WAVE 29 (M3 Pro, 2026-07-09, receipt-aware spine resume): extended
  `src/mop/studio/spine_plan.py` and `scripts/studio_spine_plan.py` with `--status`. The status receipt
  reads the spine plan plus existing receipts, classifies each step as complete, pending, running,
  blocked, or failed, and prints the exact next command and missing receipts. The final spine now writes
  `runs/studio_spine/spine_status.json` before the last artifact bundle, so a resumed Studio session can
  prove where it stopped. After the atlas verdict-ledger insertion, local status on this laptop
  reports 2/20 complete, 16 pending, 2 future dense/atlas blocks, and `wave0_run` as the next pending
  command. Verdict: still no Studio science locally, but
  interruption/resume no longer depends on prose or memory.
- WAVE 30 (M3 Pro, 2026-07-09, receipt-backed Studio scorecard): added
  `src/mop/studio/scorecard.py`, `scripts/studio_scorecard.py`, and
  `tests/unit/test_studio_scorecard.py`. The scorecard reads Wave 0, DR1 verification,
  PR9 result/state/verdict ledger, dense gate, atlas result/verdict ledger, artifact indexes, and spine status, then writes
  `mop-studio-scorecard/v1` plus a bounded block in this report. It refuses to score local PR9 smoke as
  moldability evidence, refuses DR1-cache PR9 without a verdict ledger, refuses a partial atlas as
  density evidence, and keeps missing artifact indexes as durability blockers. The staged spine now runs
  scorecard synthesis before the final status and bundle. Verdict: no Studio science was run locally, but
  final score movement now has a receipt gate instead of a manual scoreboard edit.
- WAVE 31 (M3 Pro, 2026-07-09, DR1 source/license intake gate): added
  `src/mop/studio/dr1_source_intake.py`, `scripts/studio/dr1_source_intake.py`, and
  `tests/unit/test_dr1_source_intake.py`. The receipt validates the DR1 source before scheduling:
  bound-attribute class folders, per-cell floor, factor value counts, unique clip stems, `captions.json`
  coverage, natural-video provenance, license/allowed-use fields, manual terms acceptance when needed,
  source clip-count agreement, and benchmark non-overlap proof. `dr1_schedule_plan.py --source-intake`
  now refuses runnable jobs when the intake is blocked; the staged spine runs intake before
  `dr1_schedule_build`; and the DR1 artifact bundle preserves the source card plus intake receipt.
  Verdict: no Studio video was encoded locally, but the first real DR1 compute is now guarded by an
  explicit source/license receipt rather than an operator assumption.
- WAVE 32 (M3 Pro, 2026-07-09, DR1 caption null moved before schedule): extended
  `src/mop/studio/dr1_source_intake.py` so `mop-dr1-source-intake/v1` now includes a full-source
  `caption_recoverability` section. It runs the same class of cheap label-free held-out caption probe
  used by the per-leg DR1 encode gate and blocks schedule generation if any factor fails chance plus the
  preregistered margin. The per-leg gate still runs inside `dr1_curate_bound_video.py`; this earlier
  receipt exists to stop a weak caption source before the Studio even writes runnable encode jobs.
  Verdict: no Studio evidence was claimed locally, but a caption-side null is now surfaced before DR1
  scheduling and preserved as an intake receipt.
- WAVE 33 (M3 Pro, 2026-07-09, PR9 verdict ledger and null card): added
  `src/mop/studio/pr9_verdict.py`, `scripts/studio/pr9_verdict_ledger.py`,
  `tests/unit/test_pr9_verdict_ledger.py`, and `proof/NULL_CARDS/pr9_long_stream_plasticity.md`.
  The ledger reads PR9 raw result plus run-state receipt, refuses laptop smoke caches as non-scoring,
  checks the dedicated PR9 null card, classifies config error, compute mismatch, no-certificate null,
  CBP-no-win null, or candidate positive, and records whether Process C is licensed by the PR9 wall.
  The staged spine now runs `pr9_verdict_ledger` between `pr9_run` and `pr9_artifact_bundle`; the PR9
  bundle preserves the null card, result, run-state, and verdict ledger; and the scorecard requires the
  ledger before PR9 can move moldability. Verdict: no PR9 Studio evidence was run locally, but PR9 now has
  the durable verdict-ledger receipt the Studio prompt requires.
  Gates: check_docs green, strict PR9 null-card validation green, focused PR9/spine/scorecard/bundle
  tests green, ruff lint and format green, mypy green, `git diff --check` green, acceptance 10/10 green.
- WAVE 34 (M3 Pro, 2026-07-09, DR1 source-card validation receipt): added
  `scripts/studio/dr1_source_card.py` plus `mop-dr1-source-card/v1` and
  `mop-dr1-source-card-validation/v1` helpers in `src/mop/studio/dr1_source_intake.py`.
  The new receipt validates the manually supplied DR1 provenance card before source intake: source id,
  license, allowed use, natural-video tag, non-overlap proof, manual-license acceptance, optional URLs,
  and optional expected clip count. The Studio spine now runs `dr1_source_card_validate` before
  `dr1_source_intake`; the DR1 bundle preserves `dr1_source_card_validation.json`; and the transfer
  checklist now proves the CLI is present before launch. Verdict: no source data or video was encoded
  locally, but the first DR1 operator-supplied artifact is now a durable gate instead of prose.
  Gates: transfer check green at 39/39, spine plan regenerated at 19 steps, local spine status 2/19
  complete with `wave0_run` next, docs gate green, focused source-card/spine/bundle/scorecard tests
  green, ruff lint and format green, mypy green, `git diff --check` green, acceptance 10/10 green.
- WAVE 35 (M3 Pro, 2026-07-09, dense atlas paired-cache gate): added
  `src/mop/studio/dense_atlas_gate.py`, `scripts/studio/dense_atlas_gate.py`, and
  `tests/unit/test_dense_atlas_gate.py`. The new `mop-dense-atlas-cache-gate/v1` receipt validates the
  registered dense V-JEPA 2.1 real cache and matched random-init dense control before the full atlas:
  both manifests must be clean, dense-shaped, large enough, dimension-matched, count-matched, key-aligned,
  and sidecar-aligned. The spine now runs `dense_atlas_cache_gate` before `atlas_run`, the atlas bundle
  preserves the gate plus both manifests, and the scorecard refuses density movement without the gate.
  Verdict: no dense cache or atlas evidence was run locally, but the Studio can no longer launch the
  universal atlas scope on a single real dense manifest or a missing random-init control.
  Gates: transfer check green at 40/40, dense gate writes an expected local blocked receipt because the
  Studio dense pair is absent, spine plan regenerated at 19 steps, local spine status 2/19 complete with
  `wave0_run` next and one future dense block, focused dense/spine/scorecard tests green, docs gate
  green, ruff lint and format green, mypy green, `git diff --check` green, acceptance 10/10 green.
- WAVE 36 (M3 Pro, 2026-07-09, atlas verdict ledger and null card): added
  `src/mop/studio/atlas_verdict.py`, `scripts/studio/atlas_verdict_ledger.py`,
  `tests/unit/test_atlas_verdict_ledger.py`, and `proof/NULL_CARDS/atlas_dense_multiencoder.md`.
  The ledger reads the dense cache gate plus raw atlas result, checks the dedicated null card, refuses
  missing or partial registered grids as non-scoring, preserves null-supported full atlases as walls, and
  marks null rejection only as a candidate positive that still needs the generic verdict-gate path. The
  spine now runs `atlas_verdict_ledger` between `atlas_run` and `atlas_artifact_bundle`; the atlas bundle
  preserves the null card and verdict ledger; and the scorecard requires the ledger before density can
  move. Verdict: no atlas evidence was run locally, but the full atlas cannot now skip null-card verdict
  classification before score movement.
  Gates: transfer check green at 41/41, strict atlas null-card validation green, atlas verdict ledger
  writes an expected local `dense_gate_blocked` receipt because the Studio dense pair is absent, spine
  plan regenerated at 20 steps, local spine status 2/20 complete with `wave0_run` next and two future
  dense/atlas blocks, focused atlas-ledger/spine/scorecard/bundle tests green, docs gate green, ruff lint
  and format green, mypy green, `git diff --check` green, acceptance 10/10 green.
- WAVE 37 (M3 Pro, 2026-07-09, Process C license gate and null card): added
  `src/mop/studio/process_c_gate.py`, `scripts/studio/process_c_license_gate.py`,
  `tests/unit/test_process_c_license_gate.py`, and `proof/NULL_CARDS/process_c_dense_token_pilot.md`.
  The new `mop-process-c-license-gate/v1` receipt reads the PR9 verdict ledger and DR1 adversarial
  verifier, checks the Process C null card, records PR9 or DR1 licensing sources, and only sets
  `launch_allowed: true` for the sanctioned 1 to 10M dense-token pilot. A clean "not licensed" decision
  is a completed wall, not permission to train. The spine now emits this gate between
  `pr9_verdict_ledger` and `pr9_artifact_bundle`; the PR9 artifact bundle preserves the gate plus null
  card; `studio_native_lanes.py` materializes only the license-gate command when PR9/DR1 receipt paths are
  supplied; and the scorecard reports Process C status as a separate decision. Verdict: no Process C
  science was run locally, and local evidence remains launch prep only.
  Gates: transfer check green at 42/42, strict Process C null-card validation green, Process C gate writes
  an expected local `undecidable` receipt because real Studio PR9/DR1 receipts are absent, spine plan
  regenerated at 21 steps, local spine status 2/21 complete with `wave0_run` next and three future blocks,
  focused Process-C/native-lane/spine/scorecard/bundle tests green, docs gate green, ruff lint and format
  green, mypy green, `git diff --check` green, acceptance 10/10 green.
- WAVE 38 (M3 Pro, 2026-07-09, Studio objective point re-evaluation): added
  `src/mop/studio/objective_audit.py`, `scripts/studio_objective_audit.py`, and
  `tests/unit/test_studio_objective_audit.py`. The new `mop-studio-objective-audit/v1` receipt scores
  the active Studio 10/10 prompt as eight checklist points, explicitly not as scientific axis credit:
  Wave 0 launch prep, DR1, PR9, dense/atlas, Process C authorization, adversarial/null-card discipline,
  durable artifacts/reports, and Studio-native lanes. Current local reevaluation:
  `2.768/8.0` checklist credit, with 2 complete, 3 prepared, and 3 pending points. The complete points
  are local discipline surfaces only; DR1 real-video evidence, PR9 long-stream evidence, dense/atlas
  evidence, decisive Process C authorization, and final durable artifact indexes remain unearned until
  Studio receipts exist. Transfer check now covers the objective-audit CLI, native-lane CLI, facet-12
  real-video retest scripts, and atlas runner, and `studio_artifact_bundle.py --preset atlas` is exposed.
  Gates: focused objective-audit/artifact-bundle/transfer tests green; native-lane manifest refreshed
  locally at 1 ready and 7 blocked; transfer check green at 47/47; objective audit local receipt written
  and correctly not Studio-ready; docs gate green, ruff lint and format green, mypy green,
  `git diff --check` green, acceptance 10/10 green.
- WAVE 39 (M3 Pro, 2026-07-09, objective audit wired into the runnable path): updated the Wave-0 daemon
  template so it writes `runs/studio_native_lanes_manifest.json` by default after the encode microbench,
  and updated the staged spine so finalization runs `scripts/studio_objective_audit.py` after
  `spine_status_receipt` and before `spine_artifact_bundle`. Spine status now treats a not-ready
  `mop-studio-objective-audit/v1` receipt as blocked, so the final bundle cannot silently preserve an
  incomplete 10/10 state as if it were ready. Local regenerated receipts: spine plan 22 steps
  (finalize phase now 4), local spine status 2/22 complete with `wave0_run` next, and Wave-0 daemon plan
  jobs now include `native_lanes_manifest`. Verdict: still no Studio science locally, but the re-evaluation
  is now in the actual Studio command path rather than an optional side command.
  Gates: focused long-run/spine/artifact/objective tests green, docs gate green, ruff lint and format
  green, mypy green, `git diff --check` green, acceptance 10/10 green.
- WAVE 40 (M3 Pro, 2026-07-09, preserve not-ready objective audits): patched
  `scripts/studio_objective_audit.py` with `--allow-not-ready` and made the final spine use it. The
  standalone audit still exits nonzero when `studio_10_ready` is false, but the spine can now write and
  bundle a not-ready or walled objective audit instead of stopping before `spine_artifact_bundle`. Spine
  status still classifies the not-ready receipt as blocked, so durability and honesty both survive.
  Gates: focused objective-audit/spine tests green, docs gate green, ruff lint and format green, mypy
  green, `git diff --check` green, acceptance 10/10 green.
- WAVE 41 (M3 Pro, 2026-07-09, preserve incomplete scorecards): patched
  `scripts/studio_scorecard.py` with `--allow-incomplete` and made the final spine use it. The standalone
  scorecard still exits nonzero when `all_ok` is false, but the Studio spine can now write the scorecard,
  then continue to final status, objective-audit, and artifact-bundle receipts even when the scorecard
  honestly names missing DR1/PR9/dense/atlas/spine evidence. Verdict: no Studio science locally; this
  closes another launch-prep durability gap before the real M1 Ultra run. Local regenerated receipts:
  spine plan 22 steps, local spine status 2/22 complete with `wave0_run` next, scorecard `all_ok=false`,
  native lanes 1 ready / 7 blocked, and objective audit 2.768/8.0 points. Gates: focused scorecard/spine
  tests green, docs gate green, ruff lint and format green, mypy green, `git diff --check` green,
  acceptance 10/10 green.
- WAVE 42 (M3 Pro, 2026-07-09, density receipt in Wave 0): added
  `src/mop/studio/density_receipt.py`, `scripts/studio_density_receipt.py`, and focused tests. The new
  `mop-studio-density-receipt/v1` records workspace size, tracked LOC, largest files, artifact-mass
  buckets, and disk-recovery before/after cleanup deltas; the Wave-0 daemon writes it after disk
  recovery, transfer check requires the CLI, artifact-bundle presets preserve it, and the objective
  audit counts it as durable-report launch prep only. Local regenerated receipts: transfer check 48/48,
  disk recovery 60.045 GB free with 102.3 MB safe to reclaim, density receipt workspace 470.7 MB with
  102,738 tracked-source lines, Wave-0 daemon plan includes `density_receipt`, objective audit
  2.813/8.0 points. Verdict: still no Studio science locally; this closes the shared 10/10 density
  receipt requirement before the real M1 Ultra run.

## Next move (per the goal-loop protocol)

On the M1 Ultra, in order:
1. Write the staged spine contract with
   `PYTHONPATH=src:. python -m scripts.studio spine-plan --source /data/comp_video --source-card runs/studio_dr1/dr1_source_card.json --out runs/studio_spine/spine_plan.json --wave0-plan-out runs/studio_spine/wave0_daemon_plan.json`,
   then run
   `PYTHONPATH=src:. python -m scripts.studio spine-plan --status --plan runs/studio_spine/spine_plan.json --status-out runs/studio_spine/spine_status.json`
   and execute the reported next command.
2. Finish WAVE 0: run the daemon template under `studio-m1ultra` so transfer check, disk recovery,
   density receipt, doctor JSON, docs/acceptance gates, DR1 smoke, and the MPS-vs-16-worker-CPU encode
   microbench at 128 GB all produce receipts; the same daemon writes the native-lane manifest and Wave-0 report;
   rebuild the real cache to >= 1000 clips; commit or bundle the artifacts.
3. Before the DR1 source-intake step executes, write `runs/studio_dr1/dr1_source_card.json` with
   `source_id`, `license`, `allowed_use`, `provenance_tag: natural-video`, `non_overlap_proof`,
   `clip_count`, and `accepted_terms` if manual terms are required. Use
   `PYTHONPATH=src:. python scripts/studio/dr1_source_card.py validate runs/studio_dr1/dr1_source_card.json --out runs/studio_dr1/dr1_source_card_validation.json`
   before intake; if this proof is missing or blocked, the spine must stop before source traversal and
   schedule generation. The intake step also runs the full-source caption recoverability null gate; a
   failure there is a DR1 source null, not something to tune past.
4. SPINE, standing order: DR1 first (the single highest-leverage artifact; its source-intake and caption
   acceptance gates prevent wasted encode; its A6 and `dr1_verify` receipts gate any positive), PR9 second
   (certificate-guarded; `pr9_verdict_ledger` plus `process_c_license_gate` decide Process C or
   candidate-positive status), dense cache + matched random-init cache + `dense_atlas_cache_gate` + atlas encode +
   `atlas_verdict_ledger` ride the same conveyor, B5 multi-seed and seed retrofits LAST.
5. One Part-2 item is already run (facet 12, walled provisionally); the licensed re-test rides DR1's
   real moving video plus a readout adapter. The other Part-2 lanes (13 to 17) ride the spine artifacts.
6. After any completed Studio wave, run
   `PYTHONPATH=src:. python -m scripts.studio scorecard --apply --out runs/studio_scorecard.json` so
   score movement, walls, and missing receipts are recorded before any hand-written interpretation.
7. Re-evaluate objective points with
   `PYTHONPATH=src:. python -m scripts.studio objective-audit --out runs/studio_objective_audit.json --spine-status runs/studio_spine/spine_status.json --scorecard runs/studio_scorecard.json`.
   This audit is checklist credit only; it must not be used as scientific score movement.

Every wave: ORIENT (reread this scoreboard), SELECT highest axis-delta-per-hour, PREREGISTER in code,
BUILD + RUN inside `studio-m1ultra`, VERIFY (independent adversarial pass on every positive), LEDGER
here + commit, STOP-CHECK. No wave ends without a committed artifact.
