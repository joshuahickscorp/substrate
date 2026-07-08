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

- [x] Full gates green on this box: acceptance 10/10 (full pytest suite, ruff lint + format, mypy 139
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

## Next move (per the goal-loop protocol)

On the M1 Ultra, in order:
1. Finish WAVE 0: run the MPS-vs-16-worker-CPU encode microbench at 128 GB (record s/clip + winner);
   rebuild the real cache to >= 1000 clips; commit.
2. SPINE, standing order: DR1 first (the single highest-leverage artifact; its caption acceptance gate
   prevents wasted encode), PR9 second (certificate-guarded; decides Process C), dense cache + atlas
   encode ride the same conveyor, B5 multi-seed and seed retrofits LAST.
3. One Part-2 item is already run (facet 12, walled provisionally); the licensed re-test rides DR1's
   real moving video plus a readout adapter. The other Part-2 lanes (13 to 17) ride the spine artifacts.

Every wave: ORIENT (reread this scoreboard), SELECT highest axis-delta-per-hour, PREREGISTER in code,
BUILD + RUN inside `studio-m1ultra`, VERIFY (independent adversarial pass on every positive), LEDGER
here + commit, STOP-CHECK. No wave ends without a committed artifact.
