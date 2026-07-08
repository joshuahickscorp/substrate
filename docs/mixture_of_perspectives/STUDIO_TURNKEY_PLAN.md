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
      wall-clock gate, checkpoint cadence, next command). Smoke-run with `--skip-mps` (picks cpu at
      16 s/clip on the laptop); the Studio re-runs without the flag to time MPS at 128 GB.
- [x] T4.3 perspective matrix contract: `src/mop/perspectives/adapter.py` aligns vision/language/audio/
      code/math/control arms by referent id, refuses referent drift, and audits missing matched controls
      plus supervised/derived/license flags. DR1 cache merge should write or consume this contract before
      AL2, A6 residualization, or facet-15 claims run.
- [x] T4.4 gated Process C dense-token module: `src/mop/process_c/dense_tokens.py` provides the sanctioned
      1 to 10M object-centric slot module over frozen dense tokens, dense-without-slots baseline,
      binding-specificity report, and default unlicensed-run refusal. Do not run it until PR9 or DR1
      licenses Process C.
- [x] T4.5 long-run daemon: `scripts/studio_daemon.py` supervises a JSON job plan with dry-run default,
      profile disk gates, resumable `daemon_state.json`, per-job logs, heartbeat events, and clean stop on
      blocked/failed jobs. This is the facet-16 execution spine, not a science selector.
- [x] T4.6 transfer checklist command: `scripts/studio_transfer_check.py` emits a Wave-0 transfer receipt
      proving the governing audit/docs/scripts, `studio-m1ultra` profile, null-card schema, durable
      pre-Studio receipts, git state, and cache manifests before any Studio science starts.
- [x] T4.7 Wave-0 daemon template: `scripts/studio_daemon.py template` now emits the full pre-science
      sequence: transfer check, doctor, profiles, docs gate, acceptance, DR1 smoke, and the
      `studio-m1ultra` encode microbench plan for 1000 clips.
- [x] T4.8 encode memory-envelope receipt: `scripts/mop_encode_autoselect.py` now records a
      `mop-memory-envelope/v1` block in both `encode_device.json` and `encode_schedule.json`, and writes
      a blocked receipt if model files are missing.
- [x] T4.9 Wave-0 report synthesizer: `scripts/studio_wave0_report.py` reads transfer/daemon/encode
      receipts, writes `runs/studio_wave0/wave0_report.json`, and idempotently applies a bounded auto
      block to `STUDIO_RUN_REPORT.md` with actual s/clip and memory-envelope values.
- [x] T5.1 native-lane manifest: `scripts/studio_native_lanes.py` lists the Part 2 Studio-native lanes
      as receipt-bearing entries, emits ready lanes into a standard daemon plan, and records explicit
      blocked reasons for live-encoder doctrine, perspective ecology, and Process C licensing.
- [x] Re-audit gaps closed (from the 2026-07-03 potential re-audit, honest combined 86 percent):
      DURABILITY the load-bearing verdict evidence (close_*.json, frozen_random_census.json,
      census_reaudit.json, RESULTS_PRE_STUDIO.md, dr13_predictor_fidelity.json) is now GIT-TRACKED via a
      targeted .gitignore negation (was 100 percent gitignored, one disk-loss from gone); the missing
      facet-12 adversarial verifier is RESTORED as `scripts/mop_dr13_verify.py` (reproduces 6/6 PASS
      in-repo, fixing the one over-claim); PR9 bare `--smoke` now falls back to the real cache; the stale
      12-stage rehearsal phrasing is corrected to 9/9.
- [ ] T5.2 per-lane science launchers: remaining, lowest priority; `studio_native_lanes.py` now gives
      every Studio-native lane a manifest entry, but perspective extraction and Process C launchers stay
      blocked until DR1, PR9, or a named wall licenses them.

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
  win).
- T4.2 Verify the ViT-H / ViT-g encoder configs and the acquisition commands are Studio-ready (facet 7
  atlas encoder-scale curve); both are config-only stubs today.
- T4.3 Perspective matrix contract. Multi-arm Studio runs must prove their arms share identical referents
  and matched controls before reporting cross-perspective structure. `PerspectiveAdapter` now supplies
  the contract; the next DR1 merge pass consumes it.
- T4.4 Process C dense-token module. Process C stays gated, but its first allowed module is ready to import:
  slot attention over frozen dense tokens, dense-without-slots baseline, binding-specificity report, and
  a budget/license refusal by default.
- T4.5 Long-run daemon. Week-scale Studio plans can now run under a dry-run-first supervisor with profile
  disk gates, heartbeat, logs, and resumable state. Next extension is making adversarial verification and
  strict null-card validation mandatory job kinds before any positive-ledger step.
- T4.6 Transfer checklist. Studio Wave 0 now starts with an executable receipt:
  `PYTHONPATH=src:. python scripts/studio_transfer_check.py --profile studio-m1ultra --out runs/studio_transfer_check.json`.
- T4.7 Wave-0 daemon template. On the Studio: `PYTHONPATH=src:. python scripts/studio_daemon.py template --out runs/studio_wave0_plan.json`, then inspect it and run with `--execute` only on the M1 Ultra.
- T4.8 Memory envelope. The Studio microbench must quote both actual s/clip and the emitted memory
  envelope from `runs/mot/encode_device.json` / `runs/mot/encode_schedule.json`.
- T4.9 Report synthesis. After Wave 0 on the Studio, run
  `PYTHONPATH=src:. python scripts/studio_wave0_report.py --apply` to convert receipts into the scoreboard.
- T5.1 Native lanes. Safe manifest:
  `PYTHONPATH=src:. python scripts/studio_native_lanes.py list --profile studio-m1ultra`.
  Daemon plan from ready safe lanes:
  `PYTHONPATH=src:. python scripts/studio_native_lanes.py plan --profile studio-m1ultra`.
  Heavy lanes require explicit preregistration and inputs, for example:
  `PYTHONPATH=src:. python scripts/studio_native_lanes.py plan --profile studio-m1ultra --include-heavy --clip-dir <real_pt_clips> --dr1-cache data/cache/vjepa2_vitl_comp_video`.

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
