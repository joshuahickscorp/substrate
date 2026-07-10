# STUDIO_HANDOFF.md

> CURRENT SUPERSESSION (2026-07-10): this file is a historical transfer and procurement scenario,
> not the active execution plan. The live 291-row requirements matrix has zero category-8 or
> category-9 rows and zero measured hardware blockers. The official dense ViT-B instrument runs
> locally, CM7 is a bound five-seed null, P4 and P5 are local, and the operational governor permits
> 300-minute legs while preserving frozen scientific shard identities. No receipt proves that a Mac
> Studio was delivered or is required. `MOP_MAXIMUM_POTENTIAL_GOAL.md`,
> `MOP_MAXIMUM_POTENTIAL_EXECUTION_PLAN.md`, and current proof artifacts supersede every lower
> sentence that treats a planning label, absent implementation, absent input, or faster runtime as
> hardware proof.

> CURRENT (2026-07-03): the MoP axis-ceiling program has since run five laptop rounds plus an adversarial
> ceiling audit. The device is at its proven maximum (~6.75/10) and the off-device agenda is now execution
> ready. For the current, prioritized expand plan (Track A moldability/PR9/Process C, Track B abstraction/DR1,
> Track C density/DR1), each translating a PROVEN laptop wall into a pre-registered off-device experiment with
> its validated method and decision gate, read `docs/mixture_of_perspectives/EXPAND_PHASE_PLAN.md` (and
> `RESULTS_LEDGER.md` for the proofs). The sections below are the earlier pre-round transfer notes.
>
> CORRECTION (2026-07-09): no tracked receipt proves that a Mac Studio has been purchased or delivered.
> The M1 Ultra 128 GB / 8 TB and M2 Max 96 GB / 2 TB descriptions below are historical procurement
> scenarios. `studio-m1ultra` and `studio-1tb` are resource-envelope slugs, not machine identity. Day 1 on
> any target host begins with the strict doctor; the host may use an envelope only when measured memory,
> disk, dependencies, local weights, and citable caches satisfy it. After that gate, type `/goal` in a session
> at the repo root (shipped at `.claude/commands/goal.md`); it reads the full handoff stack (this file,
> `docs/mixture_of_perspectives/HANDOFF.md`, `STUDIO_POTENTIAL_AUDIT.md` BOTH parts, `EXPAND_PHASE_PLAN.md`,
> `STUDIO_GOAL_PROMPT.md`) and then runs the iterative goal loop (wave 0 = transfer checklist + gates + the
> MPS-vs-parallel-CPU encode microbenchmark that decides the encode path). The audit's Part 1 grades the
> inherited program on this machine (~9.0 theoretical); Part 2 grades the Studio as a NEW instrument
> (facets 12-17: predictor rollouts, hosted real corpora, the 10-perspective ecology, the developmental
> long-run daemon; ~9.3 combined), because completing the laptop's agenda is necessary but not sufficient.

> UPDATE (2026-07-03, M3-Pro ORIENT pass, corrects staleness below): a stack-informed feasibility
> audit reconciled this doc against the ACTUAL runs/pre_studio/ state (172 files). Corrections to the
> "Studio-Gated Experiments" table and priority lists further down: ex15_rejuvenation and
> ex9_slot_attention ARE implemented now (run + grind JSONs exist: ex15_rejuvenation.json,
> ex9_slot_attention*.json), so their "no implementation exists" rows are stale; both render clean nulls
> (ex15 substrate_specific=False, ex9 slot_beats_flat=False). The two flagged live leads are CLOSED
> (DOCTRINE_SYNTHESIS.md section 3a): e7_sparse = real but architectural not substrate-specific
> (survives frozen-random, ratio 0.69), ex5_local_rules = REFUTED (Adam artifact), plus ex2_latent_planning
> PROMOTED to a real positive. The LAST open non-vacuous frozen-random gap, b5_degeneracy, was closed this
> pass (close_b5_degeneracy.py): its degenerate-retention advantage does not replicate at 5 seeds
> (underpowered null), so no open frozen-random gap remains. Facet 12 (the predictor rollout lane) was
> measured and walled provisionally (RESULTS_LEDGER.md). NET ORIENT VERDICT: axis-moving M3-Pro
> science is exhausted; every remaining axis-mover (DR1 real video, PR9 long stream, Process C trainable
> encoder, the 1.7 TB dense cache, the encoder-scale atlas, n-growth encode) is genuinely Studio-gated on a
> proven resource (21 s/clip encode, 128 GB residency, 8 TB disk, week-scale queue) or Tier R. DURABILITY
> FLAG: runs/pre_studio/RESULTS_PRE_STUDIO.md (56 KB source-of-record), the close_*.json verdicts, and
> frozen_random_census.json are on GITIGNORED disk; the interpretive verdicts survive in tracked
> DOCTRINE_SYNTHESIS.md, but the granular per-id record is one `rm -rf runs/` from loss. Recommend
> git-tracking them (a targeted .gitignore negation) on the Studio.

## What Is Done and Transfers

**100 cpu-now experiments ran for real** on the laptop against a frozen pooled substrate (synthetic Gaussian-cluster proxy for most series; real V-JEPA 2 ViT-L pooled features for a small subset). Full per-experiment JSON output lives in `runs/pre_studio/` (one file per experiment id, e.g. `runs/pre_studio/n8_object_permanence_bound.json`, plus a rollup at `runs/pre_studio/_summary.json`). Every result has an explicit declared null. The first 93's candidate positives (`null_supported=False`) went through a full adversarial verification pass, and two `seed_stability` failures (e4_neuromod, e7_sparse) were then genuinely re-run through the harness's real per-seed sweep: e4_neuromod's negative held up (30/30 runs); **e7_sparse's disqualifying objection did not, and it is now the corpus's one provisionally confirmed positive** (see below, top Studio priority). 7 more experiments (EX1, EX4, EX6, EX7, EX11, EX14, EX18) were completed afterward closing out the remaining registry-only cpu-now rows — 6 are clean nulls; EX4 failed a tuned-baseline check on direct inspection, and EX6 looked promising on a first informal read but was refuted by a full three-pass adversarial check (a variance-magnitude artifact of a hardcoded hyperparameter, confirmed by reproducing it with a trivial no-learning heuristic and an inversion test). See `RESULTS_PRE_STUDIO.md` for the full breakdown including the addendum. Nothing here needs to be redone from scratch on the Studio — it needs to be re-run at real scale with the missing controls actually wired in.

**Local HF weight cache** (`~/.cache/huggingface/hub/`), verified present:
- `models--facebook--vjepa2-vitl-fpc64-256` — canonical pooled encoder, fully present, this is what backs `configs/encoder/vjepa2_vitl_fpc64_256.yaml` and the one real-latent cache below.
- `models--facebook--vjepa2-vith-fpc64-256` — full 2.616 GB safetensors shard staged at pinned revision `b5eac870...`; local SHA-256 matches the Hub LFS digest. Strict offline real-model load passes in `proof/ENCODER_SCALE_VITH_LOAD.json`.
- `models--facebook--vjepa2-vitg-fpc64-384` — full 4.138 GB safetensors shard staged at pinned revision `12ca91694...`; local SHA-256 matches the Hub LFS digest. Strict offline real-model load passes in `proof/ENCODER_SCALE_VITG_LOAD.json`.
- `models--facebook--dinov2-large` — **fully downloaded**, `model.safetensors` present. Staged as the aux/cross-encoder for CKA/RSA geometry (d1_geometry) and atlas cross-checks; not yet wired into any experiment's encoder config, but ready to load.
- `models--OpenGVLab--VideoMAEv2-Base` — **fully downloaded** (~340 MB, `model.safetensors` present). Intended second video encoder for the atlas/CKA cross-encoder comparison; same status as DINOv2-large (staged, not yet wired in).

**Real-encoder latent cache** — `data/cache/vjepa2_vitl_fpc64_256_real/` is populated and verified: `meta.json` reports `count: 64`, `labels.npy` correctly interleaves 8 classes ([0..7] x8), and a linear probe against it scores acc=1.0 vs chance=0.125 (n=64, still small but a real, non-degenerate signal, quadrupled from an initial 16-clip build earlier in this session). Built via `python scripts/cache_real_encoder.py device=cpu +classes=8 +per_class=8 +batch=1` (~1355s total, ~21s/clip, `backend=vjepa_hf` confirming real weights were loaded, not the frozen-random fallback). **Important boundary discovered this session**: the same command with `device=mps` overflows the M3 Pro's MPS backend with `RuntimeError: Invalid buffer size` on 64-frame/256px V-JEPA 2 ViT-L attention, even at `batch=1` (a hard per-buffer ceiling, not a total-RAM limit — more system memory would not fix it). `device=cpu` succeeds, just slower (roughly 21s/clip regardless of batch, so wall-clock scales linearly with clip count); this is the clearest concrete case in this session of something the Studio's GPU genuinely unlocks (full-res real-latent extraction at MPS or CUDA speed instead of single-threaded CPU). The only other real-weight evidence is `runs/real_encoder_eval.json` (n=96, structured-synthetic content, also underpowered).

**Registry and doctrine infrastructure that transfers directly:** `registry/experiments.yaml` (116 catalogued rows, 108 implemented; every experiment's declared controls, null hypothesis, and `proof.null_card` pointer), `registry/models.yaml` (encoder availability flags), `registry/datasets.yaml`, `src/mop/diagnostics/substrate_ablation.py` (the frozen-random-projection control implementation used inconsistently across the corpus — this is the single most important piece of code to wire into every re-run), `src/mop/diagnostics/compute.py` (FLOP/matched-compute accounting, also under-wired), the 4 diagnostic modules completed this session (`difficulty_calibration.py`, `transfer_matrix.py`, `buffer_compression.py`, `latent_robustness.py`, all with known-answer tests but not yet called from any experiment's `run()`), `campaign/run_queue.yaml` and `campaign/legs/` (the tiered leg definitions), and `scripts/studio_pipeline.py` / `scripts/run_queue.py` (the gated execution conveyor).

---

## The Studio-Gated Experiments (down from 9 to 7, 2 implemented this session)

Of the original 9 rows tagged `studio-scale`, `environment-needed`, or `weights-needed`, only 1 (`ex2_latent_planning`) is now genuinely gated by something the laptop cannot supply. Two more (`ex13_long_stream`, `ex5_local_rules_scale`) were implemented for real this session — see the addendum in `RESULTS_PRE_STUDIO.md` and the priority list below. The remaining 6 are mislabeled or over-cautious: either they already run fine on the laptop today, or their real blocker is that no implementation exists yet, not a hardware ceiling.

**Implemented this session (removed from this table, now `status: implemented`, `resource_tier: cpu-now`):**
- `ex13_long_stream` — real, running results at two scales in `runs/pre_studio/ex13_long_stream.json` (shipped) and `ex13_long_stream_grind.json` (3000 tasks). Honest null: replay+EWC clearly beats naive, but the advantage does not yet clearly survive a frozen-random-substrate control.
- `ex5_local_rules_scale` — real, running results in `runs/pre_studio/ex5_local_rules_scale.json` (shipped) and `ex5_local_rules_scale_grind.json` (300 tasks). Genuine unforced positive (feedback_alignment and predictive_coding beat backprop on accuracy AND retention), adversarially checked as PLAUSIBLE-BUT-UNVERIFIED: backprop's Adam optimizer vs the local rules' plain delta updates are not step-size-matched, a specific missing control for the Studio.

| id | tier | why the laptop cannot do it | exact resume command |
|---|---|---|---|
| **ex2_latent_planning** | environment-needed (genuinely gated) | `status: deferred`, zero implementation, not wired into any campaign leg. Needs real action-conditioned rollout data from an interactive robot/ego environment for closed-loop MPC — a genuine environment gap the repo has no harness for at all, not a compute ceiling. | `python scripts/studio_pipeline.py run --gated --tiers E --full --profile studio-1tb` (once implemented and a leg is wired in). Laptop precursor available now: `y7_controllability_sysid_gate` / `d6_rollout_gate` (linear sysid on synthetic action-conditioned transitions) already answers the necessary precursor — is the latent even a controllable state — via `src/mop/diagnostics/sysid.py`. |
| e10_openended | environment-needed (label describes an unbuilt capstone) | The registered scaffold is a pure-numpy toy stub (`StubEnv`), already completed on the laptop in seconds (`runs/pre_studio/e10_openended.json`). Not Studio-gated at all as currently implemented; the real capstone (procedural env + population + rented GPU) doesn't exist in code and targets rented cloud CUDA, not the Studio, per `campaign/legs/track11/track11_e10_autotelic.yaml`'s own note. | Not applicable — not wired into `studio_pipeline.py` or `run_queue.py`. Real work is implementing the environment first. |
| e5_curiosity | cpu-now | The fixed-pool comparison and a persistent learnable-versus-noisy trajectory contract now run locally with exact replay and paired counterfactuals (`proof/LOCAL_ACTION_ENVIRONMENT.json`). | Independent/natural ecologies are a validation gate, not a local-compute blocker. |
| e6_relational | weights-needed (label correctly names the axis, not a Studio-hardware gate) | Runs end-to-end on the laptop today (`runs/pre_studio/e6_relational.json`, `runs/e6_relational/000-002`) — both encoders fall back to frozen-random because neither `vjepa2_vitl_fpc64_256.yaml` nor `vjepa21_vitl.yaml` sets `prefer_real: true`, and 2.1 dense weights don't exist anywhere yet (placeholder HF id 404s). Even the actual Mac Studio would hit the identical frozen-random fallback. | Once Meta publishes real V-JEPA 2.1 weights: set `prefer_real: true` in `configs/encoder/vjepa21_vitl.yaml`, then rerun anywhere with HF auth — laptop is sufficient given the tiny synthetic clip sizes (3x4x16x16) this experiment uses. |
| ex10_cross_modal | weights-needed (audio arm only) | No implementation exists, and `registry/models.yaml` has zero audio-model rows. But the video-only shell, matched-params control, and a synthetic paired-latent correspondence gate are all laptop-feasible on the existing frozen V-JEPA 2 pooled substrate; only the natural-AudioSet arm needs a real audio encoder sourced/licensed. | `python scripts/studio_pipeline.py run --gated --tiers C --full` (once an `ex10_cross_modal.py` scaffold exists). For the natural AudioSet arm specifically, first verify a real frozen audio encoder HF id, then `python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900 --include audioset_meta && ... acquire --execute --accept-license && ... run --gated --tiers C,E --full`. |
| ex15_rejuvenation | studio-scale (inherited tag, no hardware need, now weaker) | No implementation exists at all. The mechanism itself (shrink-and-perturb vs. no-rejuvenation, tracking `effective_rank`/dead units) needs no GPU, no encoder, no environment; `e3_plasticity.py` already proves this class of experiment runs entirely on CPU, and `ex13_long_stream.py` (implemented this session) now provides a ready-made long-stream harness with `effective_rank` tracking already wired in to build EX15 on top of directly. The studio-scale tag is even less justified now than before. | None exists yet, but is now the cheapest next laptop win in this table: clone `ex13_long_stream.py`'s harness, add a shrink-and-perturb arm at intervals through the stream, compare `effective_rank`/retention against the no-rejuvenation arm already in that module. |
| ex9_slot_attention | weights-needed (inherited tag, no dense-encoder need) | No implementation exists (`status: deferred`). The registry's own described mechanism ("slots over pooled latents vs. a flat-pooled head") never mentions a dense arm — it's a pooled-only comparison, trivially cheap, well inside laptop limits. The expected verdict (taxonomy-3, substrate blind spot) is itself the valid, citable laptop-scale result. | `python scripts/studio_pipeline.py local-max --download-gb 10 --time-min 180 --cache-clips 64` (once an `ex9_slot_attention.py` scaffold exists, mirroring `e6_relational.py`'s pooled arm). Add a dense arm and the larger-host grid command only once the upstream dense encoder is verified available. |

**Bottom line:** only `ex2_latent_planning` is truly gated on something the Studio (or a real environment) uniquely provides. `ex13_long_stream` and `ex5_local_rules_scale` are now implemented and running on the laptop. The other 6 remaining rows are gated on implementation work that can happen on the laptop first, not hardware.

---

## Prioritized First Things to Run on the Studio

### 0. Two more provisional findings from ex13_long_stream and ex5_local_rules_scale (implemented this session)
- **ex5_local_rules_scale: run the missing optimizer-matching control.** feedback_alignment and predictive_coding beat backprop on both accuracy and backward-transfer on a domain-incremental stream at two scales (80 tasks and 300 tasks), stable across a depth sweep. Adversarial check: PLAUSIBLE-BUT-UNVERIFIED, because backprop trains with Adam (adaptive, momentum) while the local rules use plain delta-rule updates at the same nominal but not necessarily matched EFFECTIVE learning rate. Rerun backprop with plain SGD (or Adam with weight decay tuned to match the local rules' effective step size) to see whether the BWT gap collapses or survives — this is now co-priority with e7_sparse's significance test. See `runs/pre_studio/ex5_local_rules_scale.json` and `_grind.json`.
- **ex13_long_stream: tighten the frozen-random control's stream length.** Replay+EWC protection clearly beats naive-sequential training (divergence +0.25 at 240 tasks, +0.53 at 3000 tasks, growing not shrinking with stream length), but that advantage does not clearly survive a frozen-random-substrate control. One real caveat found this session: the frozen-random arm ran a SHORTER stream (`n_tasks_control`, 80 or 1000) than the main arms (`n_tasks`, 240 or 3000) — rerun with all three arms at identical stream length before trusting the current "does not survive" verdict as final. See `runs/pre_studio/ex13_long_stream.json` and `_grind.json`.

### 1. Re-run the 25 candidate positives with the missing control actually wired in, at 5+ seeds
One candidate (e7_sparse) now survives its specific objection at toy scale and should be run FIRST. Of the other 24, none survived adversarial review (see `RESULTS_PRE_STUDIO.md`, CONFIRMED POSITIVES section); most failed because a registry-required control (usually `frozen_random`) was never implemented in code, not because it was implemented and failed. Priority order, based on which mechanisms still look directionally promising once the gap is closed:
- **e7_sparse: seed_stability RESOLVED this session, now the top priority.** e4_neuromod and e7_sparse were both re-run through `mop.harness.sweep.run_sweep` (the module code reads a single `cfg.seed`, not `experiment.seeds`, so a config override alone is a no-op — the real fix is going through the harness, which generates a genuine `seed=N` override per run). Full grid, 30 runs each (6 axis combos x 5 seeds): e4_neuromod's negative finding came back robustly confirmed (30/30 runs show both gates amplifying error on noise, the wrong direction; disagreement mean=20.07 std=1.25) and is done, no further work needed. e7_sparse's disqualifying objection did NOT hold up: sparse beats dense in direction in 27/30 (moe) and 26/30 (kwta) runs, mean gains 0.14-0.18. **e7_sparse is now the single strongest candidate positive in the corpus** — run it on real (not synthetic) latents with a formal significance test; see `runs/pre_studio/e7_sparse_fullgrid.json` for the raw 30-run data.
- **frozen_random missing entirely in code** (highest value to close next, `src/mop/diagnostics/substrate_ablation.py` already has the primitive): a4_cognitive_map, a5_action_loop, a7_comm_channel, a8_affordance_curiosity, b5_degeneracy_robustness, e9_local, i4_backprop_alts, s5_code_stability, y7_controllability_sysid_gate.
- **frozen_random run but failed the margin** (rerun on real, harder, non-saturated latents; the toy version is done): c1_held_out_combination, c5_transfer_matrix, d5_lp_self_curriculum, n8_object_permanence_bound (also has an explicit exact-tie result already — don't re-litigate, just confirm on real latents), n10_halting_difficulty, p2_memorize_vs_concept, s1_symbol_grounding.
- **matched_compute failures** (add a same-width/same-FLOPs control): d7_scaffolding (gate params confound), d8_imitation_conditioned_rollout (task hands the answer key to the conditioned arm — needs a harder task too).
- **tuned_baseline failure**: y6_free_energy_vs_lp (fix the null's AND-gate logic before rerunning).
- **b3_stigmergic_curriculum, ex8_curiosity_bakeoff, ex16_codebook_sr**: lower priority — these lost outright or showed only a partial split pattern even before the control gap; worth a rerun only after the higher-value items above.

### 2. Build real-latent caches at full res/frames now that MPS/CUDA buffer size is not the limiting factor
- `data/cache/vjepa2_vitl_fpc64_256_real/` is already fixed and verified this session (`count: 64`, real interleaved labels, `backend=vjepa_hf`, built at ~21s/clip on CPU over ~23 minutes). Rebuild it LARGER STILL (hundreds-thousands of clips) now that the Studio's GPU removes the MPS buffer-size ceiling that capped the laptop build to single-threaded `batch=1`/CPU throughput.
- Run the atlas/decodability battery (ex12_atlas, d1_geometry, a1_affordance_decode, c1_held_out_combination, p2_memorize_vs_concept, s1_symbol_grounding) on **real** ViT-L pooled latents at that larger scale instead of the synthetic Gaussian-cluster proxy. This is the single highest-value delta the completeness sweep identified: `real_acc` is currently referenced by only the small 16-clip cache, not the full 100-row corpus.
- DINOv2-large and VideoMAEv2-Base are already fully downloaded and staged (see above) — wire them into an encoder config and run the cross-encoder CKA/RSA geometry battery (d1_geometry) with real second and third encoder points, not just ViT-L alone.
- Every configured V-JEPA 2 scale point is acquired, hash-verified, strict-loaded, native-forward tested, and represented by an exact-referent eight-clip citable local cache. `proof/VJEPA_SCALE_ATLAS_LOCAL.json` runs the serial shared-corpus geometry/factor pilot. Model availability, native forward memory, and bounded cache construction are retired as Studio boundaries; rights-clean natural-video scale and matched random-architecture controls remain.

### 3. Run the studio-scale tranche once implemented: ex13, ex15, ex5
All three need implementation work before any hardware matters (see table above). Once `ex13_long_stream.py`, `ex15_rejuvenation.py`, and `ex5_local_rules_scale.py` land with matching `proof/NULL_CARDS/*.md` and campaign leg entries, run them via `python scripts/studio_pipeline.py run --gated --tiers C,E,R --full --profile studio-1tb`.

### 4. Run the weights-needed tranche once dense V-JEPA 2.1 exists
e6_relational, ex9_slot_attention (dense arm), and any 2.1-only rows are currently locked to a frozen-random fallback because `facebook/vjepa2.1-vitl-dense` does not exist on HF (placeholder ID, 404s). This is an upstream availability gate, not a Studio-hardware gate — flag it for periodic re-check rather than active work.

### 5. Other completeness-sweep items worth doing early and cheaply (seconds each, CPU-only, no download needed)
- `a1_frozen_random_arm`, `a2_matched_compute_arm`, and `d6_rollout_gate` are now `implemented`, pointed at the existing `substrate_ablation.py` / `compute.py` / `sysid.py` modules (no new code needed, the mechanism was already built as shared infra used across the corpus).
- The `studio_pipeline.py local-max` rehearsal has already been re-run against current HEAD this session (`runs/studio_pipeline/local_max_20260701_002111`, git `112053b`): all 12 stages pass (free_disk_killswitch, doctor, registry_validate, plan, acquire_dryrun, generate_controls, validate_source, build_cache, queue_cost_audit, microbench, gated_run, datacards_ledger). `runs/studio_pipeline/latest` points at it. `build_cache`/`gated_run` are still tagged `provisional` (frozen-random substrate only, as designed for a laptop rehearsal) — re-run again on the Studio once real latents are the default to get a `real` tag instead.
- `d3_difficulty_calibration`, `d4_transfer_matrix`, and `d6_rollout_gate` are now implemented (this session) as real modules with known-answer tests: `src/mop/diagnostics/difficulty_calibration.py`, `transfer_matrix.py`, and `sysid.py` respectively (d6 reuses the existing Y7 sysid machinery). Registry rows flipped to `implemented`. `d3_difficulty_calibration` is now also wired directly into `e7_sparse.py` (a `regime_calibration` field on the output, confirming the stream carries real decodable structure, `reference_score=1.0` vs `chance=0.167`) — this backs the e7_sparse priority-1 finding above. `d4_transfer_matrix` remains unwired; worth calling from any multi-task continual result (free T-by-T structure) as a next step.
- Run all ~30 enabled Tier-C campaign legs end-to-end via `python scripts/studio_pipeline.py run --gated --tiers C` to surface any broken sweep/gate wiring before committing real Studio time.
- Author the missing `proof/NULL_CARDS/*.md` files for ex1, ex4, ex6, ex7, ex11, ex14, ex18 (implemented this session, registry `null_card` pointers already set) and for ex5, ex13, ex15 (still unimplemented) — pure text, no compute, freezes the pre-registration before any GPU result can be back-fitted to it. Deliberately NOT done this session: the schema requires a `probe_dependency` block citing a specific `proof/atlas/` factor row per card, and a rushed pass risks shipping invalid or misleading cards.
- **Historical disk kill-switch finding**: free disk on this laptop drifted from ~63GB to ~53-58GB over the course of that session from system-level activity unrelated to this repo (confirmed: pytest tmp usage and this repo's own `runs/` growth were both under 200MB combined). The old unexplained `min_free_disk_gb=60` policy tripped and correctly exercised the stop path. It is now superseded by a derived 40GB floor: 10GB OS reserve + 25GB maximum pending download + 5GB temporary working headroom. Current free space is above that auditable requirement; the historical trip is safety-path evidence, not a present Studio boundary.

---

## Transfer Checklist

- [ ] **HF cache dir**: `~/.cache/huggingface/hub/` — pinned full-weight snapshots are present for V-JEPA 2 ViT-L, ViT-H, and ViT-g; DINOv2-large and VideoMAEv2-Base are also present. Transfer the snapshots intact. H/g strict offline-load receipts are `proof/ENCODER_SCALE_VITH_LOAD.json` and `proof/ENCODER_SCALE_VITG_LOAD.json`; a cache hit alone is not forward-pass evidence.
- [ ] **`runs/pre_studio/`**: all 100 experiment JSON files plus `_summary.json` and `RESULTS_PRE_STUDIO.md` itself — transfer as-is, this is the full record.
- [ ] **`runs/real_encoder_eval.json`**: the only current real-weight evidence (n=96); transfer for reference but treat as underpowered.
- [ ] **`runs/studio_pipeline/`**: transfer the `latest` symlink target (`local_max_20260701_002111`, current HEAD `112053b`, 12/12 stages pass) — already fresh, no re-run needed before trusting it.
- [ ] **`data/cache/vjepa2_vitl_fpc64_256_real/`**: transfer and trust — `meta.json` shows `count: 64`, real interleaved labels, `backend=vjepa_hf`. Still small (64 latents); rebuild much larger on the Studio once MPS/CUDA is not the buffer-size limit (see the real-encoder cache section above).
- [ ] **`registry/experiments.yaml`, `registry/models.yaml`, `registry/datasets.yaml`**: source of truth for every experiment's declared controls, encoder availability, and dataset licensing status — transfers directly, no changes needed to move it.
- [ ] **`campaign/run_queue.yaml`, `campaign/legs/`**: the full tiered leg definitions, including the currently-`enabled: false` legs (`track10_e5_curiosity`, `track11_e10_autotelic`, `track11_poet_envgen`, `track11_cultural_accumulation`, all Tier E/R) that should be reviewed for enabling now that the Studio removes the resource constraints that disabled them. `track04_e4_neuromod` and `track06_e7_sparse` are already `enabled: true` (Tier C) but had never actually been executed through `run_queue.py` before this session.
- [ ] **`proof/NULL_CARDS/`**: existing cards transfer directly; note the gaps. ex1, ex4, ex6, ex7, ex11, ex14, ex18 now have `proof.null_card` pointers set in the registry but the `.md` files themselves are not yet written (same for ex5, ex13, ex15, which remain unimplemented). Also missing: cards for the 25 candidate positives now marked REFUTED (a4/a5/a7/a8, b3/b5, c1/c5, d5/d7/d8, e4/e5/e7/e9, ex8/ex16, i4, n8/n10, p2, s1/s5, y6/y7) — authoring these as honest "refuted, here is why" cards is pure text, no compute, and freezes the finding before any Studio result could be back-fitted to it.
- [ ] **`src/mop/diagnostics/substrate_ablation.py`, `compute.py`, `sysid.py`, `geometry.py`, `difficulty_calibration.py`, `transfer_matrix.py`, `buffer_compression.py`, `latent_robustness.py`**: the standing-control primitives — confirm these get imported and actually called in every re-run of the 25 candidate positives, since the single most common failure mode in this corpus was a control existing in the codebase but never being wired into the specific experiment that needed it.
- [ ] **`src/mop/studio/profiles.py`**: both profiles already exist and transfer directly — `m3pro-local-max` (max_cache_clips=128, max_run_count=64, download_budget_gb=10.0, min_free_disk_gb=40.0, derived as reserve + hard-cap download + workspace) for reference, and `studio-1tb` (disk_total_gb=1000, download_budget_gb=900, max_cache_clips=2000000, allowed_tiers=[C,E]) already configured and ready to use in every `--profile studio-1tb` command referenced above.

---

## Mixture-of-Perspectives lane: Studio-gated experiments (appended from EXECUTION_MANIFEST.md)

The MoT laptop lane (runs/mot/, ~30 experiments) is complete or in flight on the M3 Pro. The rows below
are the MoT experiments the laptop cannot answer. Registry ids are from
docs/mixture_of_perspectives/11_experiment_registry.md. Every input listed as staged is already on the
laptop and transfers with data/cache/, runs/mot/, and the models/ staging directory. Each row: why the
laptop cannot do it, the staged inputs, and the slot relative to the numbered priorities above.

- **DR1 real bound-attribute video cache**: real-video curation plus a full encode pass past the
  128-clip clamp and the 21 s/clip CPU floor. Staged: clip validation pipeline (`substrate/video.py`),
  factorized cache layout. Slot: run WITH priority item 2 (real-latent caches); it is the #1
  fork-shortlist item and unblocks most rows below.
- **CM1 compositional gate on real video**: needs DR1 plus a random-init ViT-L arm at matched 256px,
  multi-seed. Staged: `scripts/substrate_vs_random_init_vit.py` logic and the laptop single-seed
  result (p=0.029). Slot: immediately after DR1; this is the C1 gate. The laptop
  compositional_under_nuisance run is descriptive only and must not close it.
- **substrate_vs_random_init_vit multi-seed rerun**: the headline number needs 5+ seeds at real scale.
  Staged: the landed single-seed json and the script itself. Slot: inside priority item 1 (rerun
  candidates with controls at 5+ seeds); the highest single-number value in the handoff.
- **DR2/PR3 sparse heads on real latents, 30-run protocol**: DR1-scale stream plus paired significance
  at 30 runs. Staged: laptop pilot `runs/mot/dr2_sparse_real_pilot.json`, kWTA/MoE heads in
  `shell/heads.py`. Slot: with priority item 1; the laptop pilot's delta decides how hot this runs.
- **MP4 router over reasoning primitives**: needs MP5-MP8 distinct strategies plus a D3
  difficulty-graded regime at scale. Staged: `runs/mot/mt5..mt8` verdicts and the PR1 verdict json.
  Slot: after the MP5-MP8 laptop verdicts transfer; skip if PR1 nulled and no laptop MT row survived.
- **DR3 latent scratchpad**: dense per-token latents plus a WM-load task from DR1. Staged: H-SLOTMEM
  design in the manifest, capmatch module. Slot: after DR1 dense tokens exist; the highest-value
  substrate-bound probe (14.6 #2).
- **DR4 causal intervention leakage**: DR1 factor-annotated clips. Staged: rollout harness scripts
  (mop_dr6/dr11/dr13). Slot: after DR1.
- **DR5 cross-substrate reasoning consistency**: two real encoder caches plus the random-init cache
  simultaneously. Staged: dinov2-small weights, randominit_vitl cache. Slot: after any laptop
  reasoning row survives Stage 4.
- **DR7 latent chain-of-thought**: DR1 multi-step relational task. Staged: Predictor chain harness
  design. Slot: after DR1.
- **DR14 dropped-channel arm**: dense latents. Staged: laptop VQ/4-bit/noise slopes in
  `runs/mot/dr14_corruption.json`. Slot: with the dense-cache decision (a deliberate budgeted choice
  on the 2TB box).
- **DR15 modality-general reasoning**: three encoder families cached at scale. Staged: qwen05b and
  wav2vec2 caches and weights. Slot: after DR5.
- **AT1 full cross-substrate nuisance grid**: the multi-encoder grid with per-substrate random-init
  controls exceeds the laptop queue budget. Staged: laptop grid pilot `runs/mot/at1_grid_pilot.json`
  and all small-model caches. Slot: with priority item 1; the pilot's nine-verdict table seeds it.
- **AT2 mode substrate-dependence**: random-init-ViT rerun of any winning mode at 256px on nuisance
  content. Staged: randominit_vitl cache, winning-mode scripts. Slot: after Stage 4 survivors
  transfer.
- **AL2 full shared-latent alignment**: second and third encoder caches on shared clips at scale.
  Staged: `runs/mot/al2_alignment_pilot.json`. Slot: with AT1.
- **AL3 audio-video temporal alignment**: aligned audio-video clips are new data plus an audio encode
  pass. Staged: wav2vec2-base weights and the preregistered sonification mapping. Slot: low priority,
  after AT1/AL2.
- **CM2 multi-substrate atlas gate**: multiple frozen substrates on real video. Staged: all staged
  encoder weights. Slot: only if CM1 FAILS; it is the swap-vs-build decider.
- **CM3 dense vs pooled compositional**: DR1 dense-token cache. Staged: laptop dense_vs_pooled probe
  result (ceilinged, commit c6efc74). Slot: only if CM1 fails; interface-vs-weights isolation.
- **CM4 workspace shell, registered claim**: DR1-scale stream and the 30-run e7/ex2 protocols. Staged:
  `runs/mot/cm4_workspace_pilot.json`. Slot: after DR2/PR3 lands.
- **CM5 studio-scale rejuvenation**: dim 256 to low thousands over thousands of tasks (memory and
  compute). Staged: ex15/b8 harness. Slot: after the plasticity laptop rows transfer; the C3 probe.
- **CM6 distilled ViT-S density**: trains a student model. Staged: teacher cache. Slot: optional,
  after any substrate is settled.
- **CM7 minimum-objective encoder probe**: trains a 1-5M encoder on pixels, out of Tier 0 by doctrine.
  Staged: the CM1 design doc and the nuisance clip generator. Slot: ONLY if the bounding
  prerequisites in 14.5 all land; a tie CLOSES the custom-encoder line.

Bottom line: DR1 is the binding constraint for two thirds of this table. Run the multi-seed
substrate_vs_random_init rerun and DR1 first, then let the CM1 verdict route everything else. Nothing
in this lane licenses custom training; CM7 is the only sanctioned training pilot and it is a
diagnostic, not a bet.
