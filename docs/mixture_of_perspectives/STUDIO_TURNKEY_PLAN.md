# STUDIO TURNKEY PLAN: make the M1 Ultra runs one-command, de-risked, and maximalist

The M3 Pro cannot run the Studio science (no real corpora, no 128 GB residency, no 1.7 TB dense cache),
but it CAN make the Studio's first days turnkey: every Studio script SMOKE-VALIDATED end-to-end here so
it cannot crash on expensive Studio time, preregistration FROZEN before any Studio result exists, and
one-command entry points for the whole spine. This plan is that work. House style: no em or en dashes.
A tie is a null. Every Studio positive still gets an independent adversarial verification pass.

Governing principle: the worst Studio outcome is a script that crashes on wave 1 after a multi-hour
encode. Every item below either (a) exercises a Studio script's plumbing at smoke scale on the laptop so
that crash happens HERE for free, or (b) freezes a preregistration so a Studio result cannot be
back-fit. Priority = de-risk before decorate.

## Status (2026-07-03, M3 Pro)

- [x] T1.1 DR1 smoke: `scripts/studio/dr1_smoke.py` proves the caption acceptance gate PASSES on
      captions carrying both factors (score 1.0 vs chance 0.5) and REFUSES when a factor is not
      caption-recoverable. The spine's binding pre-encode gate is de-risked.
- [x] T1.2 PR9 + atlas re-smoked on HEAD: PR9 fires reinit (reinit_count_total 28, any_zero_reinit
      false) and returns the honest inadmissible-NULL on the tiny stream; atlas runs and WITHHOLDS the
      universal claim on incomplete caches rather than faking it.
- [x] T1.3 Studio rehearsal on HEAD: 9/9 stages pass (doctor, video_corpus, source_validation,
      decode_preprocess, cache_creation, cache_integrity, full_grid_dryrun, miniature_campaign,
      microbench). The whole workflow is validated against current HEAD.
- [x] T2.1 Preregistration frozen: 7 NULL/survival cards authored from real run data (facet 12, b5,
      ex2 survival, e7 survival, ex5, ex13, ex15), all north_star-clean, ledgered in check_docs.
- [x] T3.1 facet-12 readout-adapter scaffold: `scripts/mop_dr13_readout_adapter.py` (fit a linear
      predictor-space to encoder-space adapter on visible slots, apply to the open-loop rollout;
      `--clip-dir` for the Studio real-video re-test), preregistered and smoke-run on the laptop. Smoke
      finding: the adapter halves the visible-slot representational gap (0.727 to 0.357) but does not
      transfer to the rollout (adapted 0.82 to 1.20 vs raw 0.76 to 0.83); the naive visible-slot adapter
      is insufficient, so the Studio fits it on rollout predictions on real moving video.
- [x] T4.2 ViT-H / ViT-g encoder readiness VERIFIED: `configs/encoder/vjepa2_vith.yaml` and
      `vjepa2_vitg.yaml` are correct (verified hf_ids and embed_dims 1280 / 1408), but the HF cache
      holds only config-only STUBS (8 KB each vs 1.2 GB for ViT-L), so the Studio must PULL the real
      weights, and the vitg config has NO prefer_real flag. TURNKEY STEPS for the Studio: pull with
      `.venv/bin/hf download facebook/vjepa2-vith-fpc64-256` (and `vjepa2-vitg-fpc64-384`), then set
      `prefer_real: true` in both encoder configs (add the line to vjepa2_vitg.yaml, flip it in
      vjepa2_vith.yaml) so the atlas encoder-scale curve (facet 7) loads real weights, not the
      frozen-random fallback.
- [x] T4.1 encode auto-select: `scripts/mop_encode_autoselect.py` microbenches CPU vs MPS and writes
      `runs/mot/encode_device.json` with the winner and `runs/mot/encode_schedule.json` with the
      profile-aware launch contract (device, CPU workers, dense/pooled cache bytes, disk floors,
      wall-clock gate, checkpoint cadence, next command). `scripts/studio/dr1_source_card.py` validates
      the source/provenance card, `scripts/studio/dr1_source_intake.py` writes the source/caption/license/
      non-overlap receipt, then `scripts/studio/dr1_schedule_plan.py` consumes that receipt plus the
      schedule and emits the DR1 caption gate, checkpoint-sized encode legs, merge, A6 guard, and
      optional daemon plan. Smoke-run with `--skip-mps` (picks cpu at 16 s/clip on the laptop); the
      Studio re-runs without the flag to time MPS at 128 GB.
- [x] T4.3 perspective matrix contract: `src/mop/perspectives/adapter.py` aligns vision/language/audio/
      code/math/control arms by referent id, refuses referent drift, and audits missing matched controls
      plus supervised/derived/license flags. DR1 cache merge now writes a `perspective_matrix_receipt.json`
      when paired captions and the merged root store are present, or a blocked receipt when the root store
      is still only a shard-order manifest.
- [x] T4.4 gated Process C dense-token module: `src/mop/process_c/dense_tokens.py` provides the sanctioned
      1 to 10M object-centric slot module over frozen dense tokens, dense-without-slots baseline,
      binding-specificity report, and default unlicensed-run refusal. Do not run it until PR9 or DR1
      licenses Process C.
- [x] T4.5 long-run daemon: `python -m scripts.studio daemon` supervises a JSON job plan with dry-run default,
      profile disk gates, resumable `daemon_state.json`, per-job logs, heartbeat events, and clean stop on
      blocked/failed jobs. Plans now validate that any `positive-ledger` job is preceded by both
      `verdict-gate` and `artifact-bundle` jobs, and execute runs do not skip prior dry-run states. This
      is the facet-16 execution spine, not a science selector.
- [x] T4.6 transfer checklist command: `python -m scripts.studio transfer-check` emits a Wave-0 transfer receipt
      proving the governing audit/docs/scripts, `studio-m1ultra` profile, null-card schema, durable
      pre-Studio receipts, git state, and cache manifests before any Studio science starts.
- [x] T4.7 Wave-0 daemon template: `python -m scripts.studio daemon template` now emits the full pre-science
      sequence: transfer check, disk-recovery receipt, density/artifact-mass receipt, JSON-backed
      doctor, profiles, docs gate, acceptance, DR1 smoke, the `studio-m1ultra` encode microbench plan
      for 1000 clips, the Studio-native lane manifest, and the Wave-0 report.
- [x] T4.8 encode memory-envelope receipt: `scripts/mop_encode_autoselect.py` now records a
      `mop-memory-envelope/v1` block in both `encode_device.json` and `encode_schedule.json`, and writes
      a blocked receipt if model files are missing.
- [x] T4.9 Wave-0 report synthesizer: `python -m scripts.studio wave0-report` reads transfer/daemon/encode
      receipts plus doctor and disk-recovery receipts, writes `runs/studio_wave0/wave0_report.json`,
      and idempotently applies a bounded auto block to `STUDIO_RUN_REPORT.md` with hardware, disk, MPS,
      encoder, cache-path, transfer, s/clip, and memory-envelope values.
- [x] T5.1 native-lane manifest: `python -m scripts.studio native-lanes` lists the Part 2 Studio-native lanes
      as receipt-bearing entries, emits ready lanes into a standard daemon plan, and records explicit
      blocked reasons for live-encoder doctrine, perspective ecology, and Process C licensing.
- [x] T4.10 verdict gate: `scripts/verdict_gate.py` writes a `mop-verdict-gate/v1` receipt that refuses
      any `PUBLISH-POSITIVE` verdict unless the null card is strict, the raw run receipt exists, and a
      separate verifier receipt is both passing and independent/adversarial.
- [x] T4.11 artifact bundle/index: `python -m scripts.studio artifact-bundle` writes
      `mop-artifact-bundle/v1` indexes for pre-Studio, Wave-0, and PR9 receipt sets, hashes every
      artifact, validates JSON receipts, and can copy small untracked receipts into a durable proof
      bundle.
- [x] T4.14 disk recovery receipt: `python -m scripts.studio disk-recovery` emits a dry-run cleanup plan by
      default, classifies only known generated/cache paths, refuses tracked files, and blocks ignored
      run deletion when unbundled receipt-like text artifacts are present.
- [x] T4.26 Studio density receipt: `python -m scripts.studio density-receipt` emits
      `mop-studio-density-receipt/v1` with workspace size, tracked LOC, largest files, artifact-mass
      buckets, and disk-recovery before/after cleanup deltas. The Wave-0 daemon writes it after disk
      recovery, transfer check requires the CLI, artifact bundles preserve it, and the objective audit
      counts it as durable-report launch prep rather than science evidence.
- [x] T4.13 claim ledger daemon plan: `python -m scripts.studio claim-plan` writes a daemon-valid plan that
      runs verdict gate, artifact bundle, and only then the supplied ledger command. Positive claims use
      `kind=positive-ledger`, so the daemon rejects ungated positive doc updates.
- [x] T4.12 DR1 schedule bridge: `scripts/studio/dr1_schedule_plan.py` turns `encode_schedule.json`
      into a dry JSON plan and optional long-run daemon plan. It refuses blocked schedules, carries the
      measured CPU/MPS device into `dr1_curate_bound_video.py --device`, and makes checkpoint cadence the
      source of truth for DR1 shard legs. When given `--source-intake`, it refuses to launch unless the
      source/caption/license receipt is clean. The daemon plan now ends with `dr1_verify` after the A6 guard.
- [x] T4.15 DR1 adversarial verifier: `scripts/studio/dr1_verify.py` writes
      `mop-dr1-adversarial-verification/v1` from the DR1 merge manifest, leg sidecars,
      PerspectiveMatrix receipt, and A6 residual guard. It sets positive verifier flags only when
      artifact integrity is clean and the decisive A6 condition survives.
- [x] T4.16 Studio spine plan: `python -m scripts.studio spine-plan` writes a staged
      `mop-studio-spine-plan/v1` receipt plus the Wave 0 daemon subplan. The plan orders Wave 0,
      DR1 source-card/intake/schedule/run/bundle, PR9 run/verdict/Process C license/bundle, dense-cache
      planning, paired dense real/random-init cache validation, full atlas, scorecard, spine status,
      objective audit, and final spine artifact index. It deliberately keeps Wave 0 and DR1 as
      subdaemons so their `daemon_state.json` receipts stay resumable in their own folders.
- [x] T4.17 Studio spine status: `python -m scripts.studio spine-plan --status` reads the spine plan and
      emitted receipts, classifies each step as complete/pending/running/blocked/failed, and prints the
      exact next command plus missing receipts. The final spine now writes a status receipt before the
      last artifact bundle.
- [x] T4.18 Studio scorecard receipt: `python -m scripts.studio scorecard` writes
      `mop-studio-scorecard/v1` and updates a bounded `STUDIO_RUN_REPORT.md` block. It reads Wave 0,
      DR1 verification, PR9 result/state/verdict ledger, dense gate, atlas result/verdict ledger,
      artifact indexes, and spine status. Local PR9 smoke and partial atlas runs are explicitly
      non-scoring.
- [x] T4.19 DR1 source intake receipt: `scripts/studio/dr1_source_intake.py` writes
      `mop-dr1-source-intake/v1` before DR1 scheduling. It validates the bound-attribute source layout,
      per-cell floor, unique clip stems, `captions.json` coverage, natural-video provenance tag, license
      allowance, accepted manual terms when required, source clip-count agreement, and benchmark
      non-overlap proof. It also runs the cheap label-free caption recoverability probe over the full
      source, so a caption-side null blocks before DR1 schedule generation. The Studio spine runs this
      before any DR1 encode leg and bundles the source card, validation receipt, and intake receipt with
      DR1 artifacts.
- [x] T4.20 PR9 verdict ledger: `scripts/studio/pr9_verdict_ledger.py` writes
      `mop-pr9-verdict-ledger/v1` from the PR9 raw result and run-state receipt. It refuses laptop smoke
      caches as non-scoring, checks the dedicated PR9 null card, classifies null/no-certificate,
      null/CBP-no-win, compute mismatch, config error, or candidate positive, and records whether Process
      C is licensed. `python -m scripts.studio scorecard` now requires this ledger before PR9 can move
      moldability.
- [x] T4.21 DR1 source-card validation receipt: `scripts/studio/dr1_source_card.py` writes or validates
      `mop-dr1-source-card/v1` and emits `mop-dr1-source-card-validation/v1` before the source-intake
      gate. The spine now runs this as `dr1_source_card_validate`, so an empty license, TODO source id,
      missing manual terms acceptance, non-natural provenance tag, or missing non-overlap proof stops
      before source traversal or encode scheduling.
- [x] T4.22 Dense atlas cache gate: `scripts/studio/dense_atlas_gate.py` writes
      `mop-dense-atlas-cache-gate/v1` before the full atlas. It validates both
      `vjepa21_vitl_dense8192_real` and `vjepa21_vitl_dense8192_randominit`, requires clean
      `cache_manifest.json` receipts, dense token count, embedding dim, clip count, matching referent
      keys, and matching factor/split sidecars. The scorecard now refuses density movement without this
      gate, even if an atlas JSON exists.
- [x] T4.23 Atlas verdict ledger: `scripts/studio/atlas_verdict_ledger.py` writes
      `mop-atlas-verdict-ledger/v1` from the dense gate, atlas result, and
      `proof/NULL_CARDS/atlas_dense_multiencoder.md`. It classifies missing/partial/dense-blocked atlas
      runs as non-scoring, records null-supported walls, and leaves any null rejection as a candidate
      positive that still needs the normal verdict-gate path. `python -m scripts.studio scorecard` now requires
      this ledger before density can move.
- [x] T4.24 Process C license gate: `scripts/studio/process_c_license_gate.py` writes
      `mop-process-c-license-gate/v1` from the PR9 verdict ledger, DR1 adversarial verifier, and
      `proof/NULL_CARDS/process_c_dense_token_pilot.md`. It authorizes no training by default, records
      PR9 or DR1 licensing sources when they exist, keeps an evidence-supported "not licensed" decision
      as a completed wall, and only sets `launch_allowed: true` for the sanctioned 1 to 10M dense-token
      pilot. The Studio spine emits this receipt before PR9 bundling; the scorecard reports the Process C
      decision separately from axis movement.
- [x] T4.25 Studio objective audit: `python -m scripts.studio objective-audit` writes
      `mop-studio-objective-audit/v1`, an eight-point checklist audit of the active Studio 10/10 prompt:
      Wave 0 launch prep, DR1 real-video verification, PR9 long stream, dense cache/atlas, Process C
      authorization, adversarial/null-card discipline, durable reports/artifact indexes, and
      Studio-native lanes. It explicitly labels this as objective checklist credit, not scientific score,
      so local launch prep cannot masquerade as DR1/PR9 evidence. Transfer check now also proves the
      objective-audit CLI, native-lane CLI, facet-12 real-video retest scripts, and atlas runner are
      present before launch. `python -m scripts.studio artifact-bundle --preset atlas` is now exposed. The spine
      runs the audit with `--allow-not-ready` so a wall/not-ready audit is preserved before final bundling
      while still showing as blocked in spine status.
- [x] Re-audit gaps closed (from the 2026-07-03 potential re-audit, honest combined 86 percent):
      DURABILITY the load-bearing verdict evidence (close_*.json, frozen_random_census.json,
      census_reaudit.json, RESULTS_PRE_STUDIO.md, dr13_predictor_fidelity.json) is now GIT-TRACKED via a
      targeted .gitignore negation (was 100 percent gitignored, one disk-loss from gone); the missing
      facet-12 adversarial verifier is RESTORED as `scripts/mop_dr13_verify.py` (reproduces 6/6 PASS
      in-repo, fixing the one over-claim); PR9 bare `--smoke` now falls back to the real cache; the stale
      12-stage rehearsal phrasing is corrected to 9/9.
- [ ] T5.2 per-lane science launchers: remaining, lowest priority; `studio_native_lanes.py` now gives
      every Studio-native lane a manifest entry, but perspective extraction and Process C launchers stay
      blocked until the named receipt path exists. Process C training specifically requires
      `runs/mot/process_c_license_gate.json` with `launch_allowed: true`.

## Tier 1: de-risk the spine scripts (highest leverage)

- T1.1 DR1 smoke. `scripts/studio/dr1_curate_bound_video.py` is the spine's #1 script and has NO smoke
  mode (PR9 and atlas do). Add a `--smoke` flag that bypasses the >= 32 GB RAM guard and the
  one-encoder pgrep guard and swaps a tiny stub encoder for V-JEPA, then run the FULL pipeline on a tiny
  real-frame fixture: curate -> caption acceptance gate -> encode leg (stub) -> merge shards
  (clip_stems / clip_cells sidecars) -> A6 residual guard. Also run the NEGATIVE fixture (a factor NOT
  caption-recoverable) to confirm the gate REFUSES (a tie is a null), so the gate is proven to gate.
- T1.2 Re-smoke PR9 and atlas against current HEAD; fix any drift (they carry smoke modes already).
- T1.3 Re-run the Studio rehearsal on HEAD to confirm the conveyor before Studio time. `make rehearse`
  (`scripts/studio_rehearsal.py`, the whole workflow on tiny fixtures) passes 9/9 stages on HEAD; the
  heavier 12-stage `studio_pipeline.py local-max` rehearsal (last validated at 112053b) is the Studio's
  own WAVE-0 step (download + real encode), not re-run here.

## Tier 2: freeze preregistration (zero compute, pure honesty)

- T2.1 Author the missing NULL_CARDS: the ex-series (ex1, ex4, ex5, ex6, ex7, ex11, ex13, ex14, ex15,
  ex18), the refuted candidate positives, b5_degeneracy, and facet 12. Each card states the preregistered
  null, the required control, the decision threshold, and the honest current verdict, so no Studio result
  can be back-fit to a moved goalpost. Wire the registry `null_card` pointers to the new files.

## Tier 3: scaffold the re-tests this session discovered

- T3.1 Facet-12 real-corpora + readout-adapter script (the licensed re-test from RESULTS_LEDGER.md
  section 11): extend the decodability test with a `--clip-dir` real-video path and an ADAPTER arm (fit a
  linear map predictor-space -> encoder-space on visible slots, apply to rollout), preregistered and
  smoke-tested, so the Studio runs the facet-12 licensed re-test with one command on real moving video.

## Tier 4: encode path and conveyor polish

- T4.1 Encode auto-select. The WAVE-0 microbench measured CPU 13.69 s/clip vs MPS 821 s/clip (paged at
  18 GB) on the M3 Pro. Wire the encode path to run a tiny microbench and pick the winner automatically,
  then feed that measurement into the profile-aware scheduler, so the Studio does not hand-choose the
  device, CPU workers, checkpoint cadence, or disk reserve (and re-measures MPS at 128 GB where it may
  win). Before the DR1 schedule, write `runs/studio_dr1/dr1_source_card.json` with `source_id`,
  `license`, `allowed_use`, `provenance_tag: natural-video`, `non_overlap_proof`, optional
  `requires_manual_license`, `accepted_terms`, and `clip_count`. The helper can create the card shape:
  `PYTHONPATH=src:. python scripts/studio/dr1_source_card.py template --source-id <id> --license <name> --allowed-use <use> --non-overlap-proof <proof> --clip-count <n> --out runs/studio_dr1/dr1_source_card.json`.
  Then validate it as a durable receipt:
  `PYTHONPATH=src:. python scripts/studio/dr1_source_card.py validate runs/studio_dr1/dr1_source_card.json --out runs/studio_dr1/dr1_source_card_validation.json`.
  After the validation receipt is clean, run:
  `PYTHONPATH=src:. python scripts/studio/dr1_source_intake.py --source /data/comp_video --source-card runs/studio_dr1/dr1_source_card.json --out runs/studio_dr1/dr1_source_intake.json`.
  This command runs the caption-recoverability null gate before the schedule exists.
  After the Studio schedule exists, materialize DR1 legs with:
  `PYTHONPATH=src:. python scripts/studio/dr1_schedule_plan.py --source /data/comp_video --source-intake runs/studio_dr1/dr1_source_intake.json --daemon-out runs/studio_wave0/dr1_daemon_plan.json`.
- T4.2 Verify the ViT-H / ViT-g encoder configs and the acquisition commands are Studio-ready (facet 7
  atlas encoder-scale curve); both are config-only stubs today.
- T4.3 Perspective matrix contract. Multi-arm Studio runs must prove their arms share identical referents
  and matched controls before reporting cross-perspective structure. `PerspectiveAdapter` now supplies
  the contract; the DR1 merge pass writes the receipt beside the merged cache when `--source` is provided.
- T4.4 Process C dense-token module. Process C stays gated, but its first allowed module is ready to import:
  slot attention over frozen dense tokens, dense-without-slots baseline, binding-specificity report, and
  a budget/license refusal by default.
- T4.5 Long-run daemon. Week-scale Studio plans can now run under a dry-run-first supervisor with profile
  disk gates, heartbeat, logs, and resumable state. Validate plans before execution:
  `PYTHONPATH=src:. python -m scripts.studio daemon validate --plan <plan.json>`.
  Any `positive-ledger` job must follow both `verdict-gate` and `artifact-bundle` jobs.
- T4.6 Transfer checklist. Studio Wave 0 now starts with an executable receipt:
  `PYTHONPATH=src:. python -m scripts.studio transfer-check --profile studio-m1ultra --out runs/studio_wave0/transfer_check.json`.
- T4.7 Wave-0 daemon template. On the Studio: `PYTHONPATH=src:. python -m scripts.studio daemon template --out runs/studio_wave0_plan.json`, then inspect it and run with `--execute` only on the M1 Ultra.
- T4.8 Memory envelope. The Studio microbench must quote both actual s/clip and the emitted memory
  envelope from `runs/mot/encode_device.json` / `runs/mot/encode_schedule.json`.
- T4.9 Report synthesis. After Wave 0 on the Studio, run
  `PYTHONPATH=src:. python -m scripts.studio wave0-report --apply` to convert receipts into the scoreboard.
- T4.10 Verdict gate. Before any positive enters `STUDIO_RUN_REPORT.md` or the verdict ledger, run
  `PYTHONPATH=src:. python scripts/verdict_gate.py --null-card <card.md> --run-receipt <run.json> --verifier-receipt <verify.json> --out <gate.json>`.
  A null/tie may omit `--verifier-receipt`, but still needs the strict card and raw JSON receipt.
- T4.11 Artifact index. Pre-Studio receipt index:
  `PYTHONPATH=src:. python -m scripts.studio artifact-bundle --preset pre-studio --require-durable --out proof/ARTIFACT_INDEX/pre_studio.json`.
  After M1 Ultra Wave 0, preserve ignored run receipts with:
  `PYTHONPATH=src:. python -m scripts.studio artifact-bundle --preset wave0 --copy-dir proof/ARTIFACT_BUNDLES/wave0 --require-durable --out proof/ARTIFACT_INDEX/wave0.json`.
  After DR1, preserve the small cache sidecars without copying arrays:
  `PYTHONPATH=src:. python -m scripts.studio artifact-bundle --preset dr1 --copy-dir proof/ARTIFACT_BUNDLES/dr1 --require-durable --out proof/ARTIFACT_INDEX/dr1.json`.
  After PR9, synthesize the verdict ledger:
  `PYTHONPATH=src:. python scripts/studio/pr9_verdict_ledger.py --result runs/mot/pr9_continual_backprop.json --state runs/mot/pr9_continual_backprop.json.state.json --out runs/mot/pr9_verdict_ledger.json`.
  Then preserve the plasticity result, run-state receipt, verdict ledger, and PR9 null card with:
  `PYTHONPATH=src:. python -m scripts.studio artifact-bundle --preset pr9 --copy-dir proof/ARTIFACT_BUNDLES/pr9 --require-durable --out proof/ARTIFACT_INDEX/pr9.json`.
- T4.14 Disk recovery. Emit a dry-run receipt before launch cleanup:
  `PYTHONPATH=src:. python -m scripts.studio disk-recovery --profile studio-m1ultra --out runs/studio_wave0/disk_recovery.json`.
  Then emit the density/artifact-mass receipt:
  `PYTHONPATH=src:. python -m scripts.studio density-receipt --disk-recovery runs/studio_wave0/disk_recovery.json --out runs/studio_wave0/density_receipt.json`.
- T4.13 Claim ledger plan. For any positive doc update, generate a daemon plan instead of running the
  ledger command directly:
  `PYTHONPATH=src:. python -m scripts.studio claim-plan --null-card <card.md> --run-receipt <run.json> --verifier-receipt <verify.json> --verdict-gate-out <gate.json> --artifact-index-out <index.json> --copy-dir proof/ARTIFACT_BUNDLES/<wave> --ledger-cmd-json '["python","-m","scripts.studio","wave0-report","--apply"]' --out runs/studio_claim_plan.json`.
- T4.12 DR1 schedule bridge. Validate the generated daemon plan before execution:
  `PYTHONPATH=src:. python -m scripts.studio daemon validate --plan runs/studio_wave0/dr1_daemon_plan.json`.
- T4.15 DR1 verifier. If a DR1 daemon is resumed manually, run the verifier after A6:
  `PYTHONPATH=src:. python scripts/studio/dr1_verify.py --cache data/cache/vjepa2_vitl_comp_video`.
- T4.16 Studio spine plan. On the Studio, write the staged run contract before executing waves:
  `PYTHONPATH=src:. python -m scripts.studio spine-plan --source /data/comp_video --source-card runs/studio_dr1/dr1_source_card.json --out runs/studio_spine/spine_plan.json --wave0-plan-out runs/studio_spine/wave0_daemon_plan.json`.
  Then run `PYTHONPATH=src:. python -m scripts.studio spine-plan --status --plan runs/studio_spine/spine_plan.json --status-out runs/studio_spine/spine_status.json`
  whenever a session starts or resumes; execute the reported next command. Dense/atlas stops honestly at
  `runs/mot/dense_atlas_cache_gate.json` until the real dense cache and matched random-init dense cache
  both have valid manifests, matching referents, and enough dense tokens; the atlas step uses the full
  registered grid, never passes `--allow-partial`, and writes
  `runs/mot/atlas_verdict_ledger.json` before bundling.
- T4.18 Studio scorecard. After any completed wave, or as the final spine report step, run
  `PYTHONPATH=src:. python -m scripts.studio scorecard --apply --out runs/studio_scorecard.json`.
  This command refuses to mark Studio 10 while Wave 0, DR1, PR9 verdict ledger, dense gate, atlas
  verdict ledger, or artifact indexes are missing. A proven null or wall is preserved in the scorecard
  rather than converted into a positive. The Studio spine adds `--allow-incomplete` only for the final
  preservation pass so later status, objective-audit, and artifact-bundle receipts are still written
  when the scorecard names a real blocker.
- T5.1 Native lanes. Safe manifest:
  `PYTHONPATH=src:. python -m scripts.studio native-lanes list --profile studio-m1ultra`.
  Daemon plan from ready safe lanes:
  `PYTHONPATH=src:. python -m scripts.studio native-lanes plan --profile studio-m1ultra`.
  Heavy lanes require explicit preregistration and inputs, for example:
  `PYTHONPATH=src:. python -m scripts.studio native-lanes plan --profile studio-m1ultra --include-heavy --clip-dir <real_pt_clips> --dr1-cache data/cache/vjepa2_vitl_comp_video`.

## Tier 5: one-command entry points for the missing facets (13 to 17)

- T5.1 Scaffold the cheapest missing facet entry points with preregistered nulls and matched controls,
  so the Studio has a runnable or explicitly blocked manifest entry per facet rather than a design note.
  Default manifests emit only safe ready lanes; large downloads and science runs require `--include-heavy`
  plus concrete input paths.
- T5.2 Add the per-lane launchers only after their receipts exist. Lowest priority: the spine (DR1, PR9)
  must land first and these lanes ride its artifacts.

## Order and gates

Tier 1 first (de-risk), then Tier 2 (freeze), then Tier 3, 4, 5. Every change ships lint + type + test +
docs green and a plain commit. The Studio inherits: smoke-proven spine scripts, a frozen preregistration,
turnkey re-test scripts, an auto-selecting encode path, and this plan as the running checklist. Progress
is tracked in `STUDIO_RUN_REPORT.md`.
