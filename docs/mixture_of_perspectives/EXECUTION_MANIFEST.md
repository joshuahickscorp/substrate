# EXECUTION_MANIFEST.md: the Mixture-of-Perspectives operational plan

This is the single operational document for (A) the laptop scaffolding build, (B) the Studio append, and
(C) the laptop MAXIMIZATION RUN. Registry ids refer to docs/mixture_of_perspectives/11_experiment_registry.md.
All paths are repo-relative under /Users/scammermike/Downloads/brain. House style: no em or en dashes,
line length 110, preregistered nulls, non-vacuous controls (random-init or random-encoder features, never
a square latent projection), matched compute, tuned baselines, noisy-TV guard, seed stability.

---

## 1. LIVE STATE (respect this before touching anything)

In flight right now:

- `scripts/compositional_under_nuisance.py`: V-JEPA CPU encode, 200 clips, ~60 min remaining. It OWNS the
  CPU and the 18GB RAM headroom. Until it exits: NEVER run torch, NEVER load any model, no new encode.
  Its verdict is DESCRIPTIVE ONLY (random-pixel control is resolution-confounded, 256px vs 32px); it
  cannot clear gate C1 or fire off-ramp S2.
- PR1 mode-error disjointness: `scripts/pr1_mode_error_disjointness.py` writing
  `runs/pre_studio/pr1_*.json`, owned by a separate workflow. Treat as IN FLIGHT. Do not rebuild, do not
  touch its script or outputs. PR1 gates MP1/MP2/MP3 interpretation and the MP8/DR12 preconditions.

Standing rules:

- One encoder at a time. Cached-latent CPU work must not start until the in-flight encode finishes
  (18GB pool OOMs on two encoders; the registry says so explicitly for PR2/AT1/AT2/CM7).
- RAM accounting: the in-flight encode owns the 18GB pool; residual headroom for everything else is
  ~5-6GB. The light queue class therefore has a checkable definition: every light row MUST state a RAM
  ceiling that fits inside that residual, and a light row with no stated ceiling does not run while an
  encode is in flight. No light job may batch clips: 200 nuisance clips at 64x3x256x256 float32 are
  ~50MB each, ~10GB batched, which OOMs the box and kills the encode. Streaming per clip is mandatory.
- Disk is 51GB free, BELOW the 60GB floor. Large weight downloads (ViT-H, ViT-g, big LLMs) are
  FORBIDDEN. Allowed small downloads: DINOv2-S ~90MB, a <=2GB small LLM, wav2vec2-base ~400MB. All small
  downloads are STAGED: they back up and transfer to the Studio.
- No 64-frame ViT-L forward on MPS (per-buffer hang). No dense latent caching on the laptop. No real
  cache past the max_cache_clips=128 clamp without a logged override. No probe delta against
  frozen_random_projection read as a substrate claim (vacuous by construction).
- Newest gold result: `substrate_vs_random_init_vit.py` LANDED. Real V-JEPA 0.517 vs random-init ViT-L
  0.241 vs random-pixel 0.103 (chance 0.167), V-JEPA vs random-init one-sided p=0.029 at matched 256px.
  Off-ramp S1 FIRED, trigger T3 satisfied (single seed, Studio multi-seed owns the headline). The
  custom-encoder line stays UNLICENSED. Every manifest row below is a frozen-substrate row.

---

## 2. WORK PACKAGES (laptop build)

Rules: every WP lists the NEW files it creates (unique, no two WPs write the same file). ALL extensions
to existing shared files are owned by WP-02 (the integrator); no other WP edits an existing file. Queue
class: light = can run while an encode is in flight (no torch, small RAM), heavy = exclusive CPU
cached-latent job, encode = needs an encoder forward pass and queues in the encoder lane. Difficulty 1-5.

### WP-01: small-substrate downloads + caching (the staged second and third currencies)

- Experiments enabled: WS1, WS2, WS3, WS4, AT1 (laptop grid pilot), AL2 (pilot), DR5 (staging),
  DR15 (staging), AL3 (staging), AT4 (featurizer inputs).
- NEW files:
  - `scripts/stage_small_substrates.py` (hf download + sha manifest, writes `runs/mot/staging_manifest.json`)
  - `scripts/cache_dinov2s_nuisance.py` (DINOv2-S real + random-init from_config encode of 8 subsampled
    frames per nuisance clip at 224px, pooled, into `data/cache/dinov2s_nuisance_{real,randominit}`)
  - `scripts/cache_qwen_textified.py` (label-free textification of the same clips, mid-layer mean hidden
    state, real + from_config random-init, into `data/cache/qwen05b_textified_{real,randominit}`)
  - `scripts/cache_wav2vec2_sonified.py` (preregistered deterministic sonification mapping, real +
    random-init, into `data/cache/wav2vec2_sonified_{real,randominit}`)
  - `scripts/featurize_handcrafted.py` (HOG + hue histogram + frame-difference flow stats computed on
    the SAME nuisance clips as every encoder cache, into `data/cache/handcrafted_descriptors`)
  - `scripts/featurize_programmatic.py` (ground-truth scene-program vectors rederived from the
    generator's torch RNG draws, into `data/cache/programmatic_reference`)
- Exact hf ids and sizes: `facebook/dinov2-small` (~90MB), `Qwen/Qwen2.5-0.5B` (~1.0GB, fallback
  `HuggingFaceTB/SmolLM2-360M` ~0.7GB), `facebook/wav2vec2-base` (~380MB). Total ~1.5GB against 51GB
  free. Nothing larger. All three back up and transfer to the Studio.
- RULE: downloads are network-only (RAM <200MB) and may run NOW, during the in-flight encode. Every
  encode/caching script in this WP is RAM-heavy (torch + model load) and MUST queue AFTER the in-flight
  V-JEPA encode, serially, one model at a time.
- CLIP IDENTITY RULE (governs both featurizers): the nuisance clips exist only in memory. They are drawn
  by torch RNG inside `scripts/compositional_under_nuisance.py` (make_bound_nuisance_clip, torch.Generator
  draws for scale/rotation/position/motion/clutter) and are never written to disk. Numpy cannot reproduce
  torch RNG streams, so a numpy reimplementation would featurize DIFFERENT clips than every encoder cache
  and silently invalidate AT4's cross-column comparison. Both featurizers therefore MUST regenerate the
  clips by importing the same torch generator code, seed for seed. That makes them torch-CPU jobs: they
  are NOT Stage 0 jobs, they queue immediately AFTER the in-flight encode exits and BEFORE Q1.2 (AT4),
  and they MUST stream ONE clip at a time as uint8 frames (RAM <1GB, never a batched float32 tensor).
  Acceptable alternative if streaming proves awkward: a one-shot clip-dump pass right after the encode
  that persists uint8 frames to disk (~2.4GB for 200 clips at 64x3x256x256) and both featurizers read
  frames from disk; the dump counts against the 51GB free disk and is deleted after AT4 lands.
- Random controls wired in: same-arch from_config random-init for each of the three models at identical
  preprocessing (the non-vacuous control); random read-head over descriptors for the handcrafted arm.
- Depends on: nothing for downloads; encoder lane free for the caches; the in-flight encode's exit for
  the featurizers (clip identity rule above). Queue class: encode (caches), light (downloads), light
  post-encode (featurizers, torch-CPU streamed, RAM <1GB). Difficulty: 2.
- Preregistered nulls: per-substrate delta over its own random-init within seed spread
  (random-control-artifact, the expected modal verdict); sonification caveat preregistered (the mapping
  injects factor structure; the claim is only pretraining-over-random-init in this rendering).

### WP-02: integrator (all shared-file extensions + shared new modules)

- Experiments enabled: every WP below consumes at least one artifact from here.
- NEW files (shared infrastructure, owned here so no other WP collides):
  - `src/mop/substrate/adapter.py` (SubstrateAdapter ABC + RealEncoderAdapter, RandomInitViTAdapter,
    RandomPixelAdapter, honest resolution metadata; per 13_code_scaffolding.md, the one gated-NEW module)
  - `src/mop/diagnostics/alignment.py` (thin aggregator over geometry + seed_consistency, permutation
    p-value; does NOT reimplement CKA)
  - `src/mop/diagnostics/cross_substrate.py` (dict[substrate_tag -> latents] + labels in, per-substrate
    probe acc + CKA matrix + shuffled-label null + random-map-of-equal-rank floor out)
  - `src/mop/diagnostics/riskcov.py` (AUROC, equal-mass ECE, risk-coverage curve and area, Pareto
    frontier area, seed-CI and sign-flip reporting; H-RISKCOV)
  - `src/mop/diagnostics/continual_metrics.py` (factored BWT core lifted from
    buffer_compression._bwt_at_bits, plus FWT, forgetting-area, adaptation-speed, LR-integral
    accumulator; these four do not exist anywhere yet)
  - `src/mop/shell/workspace.py` (pure composition of predictor/heads/ensemble/modulation behind
    cfg.shell flags; no new science)
  - `src/mop/shell/capmatch.py` (matched-capacity constructor: solve hidden width for a target param
    count within 2%, fixed-total-params bandwidth-sweep variant; H-CAPMATCH)
  - `scripts/mop_aggregate_report.py` (reads every `runs/mot/*.json`, emits the verdict table with seed
    sign-flips, writes `runs/mot/aggregate_report.json`)
  - `tests/unit/test_mop_shared_modules.py` (tiny-tensor unit tests for the six new src modules, no
    network, no weights, per test conventions)
- EXISTING shared files this WP (and only this WP) extends:
  - `registry/experiments.yaml`: add MT/DR/PR/WS/AT/AL/CM rows with null + baseline + falsifier each
  - `configs/encoder/*.yaml` + `src/mop/substrate/encoder_registry.py`: add family and
    training_objective keys and surface them in list_encoders()
  - `src/mop/shell/heads.py`: add kWTA and MoE gated heads (the e7 head family; blocks DR2/PR3, WS5,
    CM4 and the C1-C3 routing metrics until added)
  - `src/mop/shell/__init__.py`: export the new heads and workspace
  - `src/mop/diagnostics/compute.py`: attention-op and kNN-op FLOP counters (for MP7 scorer, WS slot
    attention, DR10/PR8 retrieval)
  - `src/mop/diagnostics/__init__.py`: re-export alignment, cross_substrate, riskcov,
    continual_metrics
  - `configs/experiment/`: one yaml per new runnable row (null_hypothesis mandatory, enforced by
    validate)
- Depends on: nothing. Interface stubs land first so lanes 3-7 can build against them. Queue class:
  light (build only). Difficulty: 3.
- Preregistered nulls: n/a (infrastructure), but every registry row added here must carry its exact
  null from section 11 verbatim, and `make accept` plus `scripts/devel.py experiments` must stay green.

### WP-03: halting and verifier harness (H-HALT + H-SCORER)

- Experiments: MP5 (adaptive halting), MP6 (confidence stopping), DR9 (verify-revise), DR8 (fixed point,
  V-JEPA arm now, typing arm reruns later on the random-init cache).
- NEW files: `scripts/mop_mt5_adaptive_halting.py`, `scripts/mop_mt6_confidence_stop.py`,
  `scripts/mop_dr9_verify_revise.py`, `scripts/mop_dr8_fixed_point.py`,
  `tests/integration/test_mop_halting.py`.
- Mechanism: IterativeRefiner(halt=True) + ponder cost; fixed-depth control built at the ADAPTIVE MEAN
  via depth_for_matched_flops; free update-norm rule tuned to the same mean steps; Verifier vs shuffled
  verifier; unroll(K=64) decay curves with matched-depth control.
- Depends on: WP-02 (riskcov, compute extensions, experiment yamls). Queue class: heavy. Difficulty: 3.
- Nulls wired: MP5 adaptive ties fixed depth at equal average FLOPs or halt collapses to constant N;
  MP6 trained halt ties the free update-norm rule; DR9 verify-revise ties single-shot and trained ties
  shuffled (the ex18 result); DR8 no geometric decay and past-horizon loss rises (the n9/y1 result).

### WP-04: search and debate (MP7, MP8)

- Experiments: MP7 (beam/tree search at matched TOTAL FLOPs, pruned work counted), MP8 (latent debate,
  gated on the PR1 verdict for decorrelated seeded modules).
- NEW files: `scripts/mop_mt7_beam_search.py`, `scripts/mop_mt8_latent_debate.py`,
  `tests/integration/test_mop_search_debate.py`.
- Controls: greedy chain at matched total FLOPs, random branch scorer, oracle beam upper bound; plain
  ensemble arm and self-debate arm for MP8, seed-permuted modules.
- Depends on: WP-02, WP-03 (Verifier harness); MP8 additionally reads the PR1 verdict json (do not
  rerun PR1). Queue class: heavy. Difficulty: 3.
- Nulls: MP7 search ties deeper greedy at matched total FLOPs; MP8 debate ties max(single, ensemble)
  (the ex17 unrolled-ensemble null).

### WP-05: rollout and planning harness (H-ROLLOUT)

- Experiments: DR6 (internal simulation, ex2 extension), DR11 (Monte-Carlo rollouts), DR13
  (planning-horizon limit).
- NEW files: `scripts/mop_dr6_rollout_planning.py`, `scripts/mop_dr11_mc_rollouts.py`,
  `scripts/mop_dr13_horizon_limit.py`, `tests/integration/test_mop_rollout.py`.
- Mechanism: action-conditioned Predictor, candidate-plan search with ALL rollouts counted in FLOPs,
  action-shuffle control, rollout_gate, noisy-TV guard on the MC arm, stochastic-rollout diversity
  statistic.
- Depends on: WP-02. Queue class: heavy. Difficulty: 2 (DR13), 3 (DR6, DR11).
- Nulls: DR6 planning ties max(reactive, action-shuffle) at matched compute; DR11 MC ties the matched
  single longer rollout or wins only on aleatoric noise (noisy-TV fail); DR13 planning never beats
  reactive at any horizon, or only at H=1.

### WP-06: memory and retrieval (H-MEMORY + fast/slow)

- Experiments: DR10 (retrieve-then-reason), PR8 (memory-augmented retrieval head), PR7 (fast/slow
  two-timescale weights).
- NEW files: `scripts/mop_dr10_retrieve_reason.py`, `scripts/mop_pr8_retrieval_head.py`,
  `scripts/mop_pr7_fast_slow.py`, `tests/integration/test_mop_memory.py`.
- Mechanism: shell/buffer.py kNN + retrieval-conditioned refiner; random-retrieval and
  shuffled-neighbor controls; PR8 must beat BOTH plain kNN and the matched-param parametric head; PR7
  Hebbian outer-product fast store on top of modulation.WorkingMemory vs slow-only and matched-size
  buffer.
- Depends on: WP-02 (kNN FLOP counter, continual_metrics for adaptation speed). Queue class: heavy.
  Difficulty: 2 (DR10, PR8), 3 (PR7).
- Nulls: DR10 retrieval ties max(from-scratch, random retrieval); PR8 head ties kNN or ties parametric;
  PR7 fast weights tie slow-only and tie a matched-size replay buffer.

### WP-07: plasticity program (PR4, PR5, PR6)

- Experiments: PR4 (epistemic gate vs noisy-TV), PR5 (content-gated critical period), PR6 (offline sleep
  consolidation).
- NEW files: `scripts/mop_pr4_epistemic_gate.py`, `scripts/mop_pr5_content_gated_cp.py`,
  `scripts/mop_pr6_sleep_consolidation.py`, `tests/integration/test_mop_plasticity.py`.
- Mechanism: Ensemble.mean_and_disagreement -> Neuromodulation.gate -> PlasticityController.lr_scale
  (PR4, scored on reducible-vs-noise LR-integral); surprise-driven reopen_threshold vs cosine and tuned
  constant LR at MATCHED LR-integral (PR5); wake/sleep phase separation vs interleaved replay at matched
  total gradient steps with EWC refresh (PR6).
- Depends on: WP-02 (LR-integral accumulator, FWT, forgetting-area in continual_metrics). Queue class:
  heavy. Difficulty: 3 each.
- Nulls: PR4 gate allocates no more LR-integral to the reducible partition than ungated (the e4
  conflation, 30/30); PR5 no retention/reopening advantage at matched LR-integral (the e3/d6 negative);
  PR6 offline ties interleaved at matched steps.

### WP-08: uncertainty routing and the MT router pilots (H-ENSEMBLE consumers)

- Experiments: DR12 (disagreement-as-uncertainty), AL1 (uncertainty router with noisy-TV guard),
  MP1/MP2/MP3 (router vs best mode, vs uniform ensemble, hetero vs homogeneous; synthetic pilots today,
  real answer blocked on DR1).
- NEW files: `scripts/mop_dr12_disagreement.py`, `scripts/mop_al1_uncertainty_router.py`,
  `scripts/mop_mt123_router_pilots.py`, `tests/integration/test_mop_routing.py`.
- Mechanism: reuses PR1's calibrated difficulty grid, per-sample correctness matrix, and
  heterogeneous-vs-homogeneous oracle logic READ FROM `runs/pre_studio/pr1_*.json` (never rebuilt); the
  MT bank is {reactive readout, ex2 planner, e7 sparse head}; uncertainty admitted ONLY as router input,
  never as an LR gate.
- Depends on: WP-02; PR1 verdict on disk for gating; WP-02 heads for the e7 mode. Queue class: heavy.
  Difficulty: 3 (DR12, AL1), 4 (MP1-3 pilots, matched compute across three modes).
- Nulls: DR12 disagreement AUROC <= confidence AUROC or noisy-TV fail; AL1 router ties random episode
  selection or noisy-TV fail; MP1 routed density <= best single mode at matched compute; MP2 router <=
  uniform blend at matched total FLOPs; MP3 hetero <= homo k-copy MoE at matched params AND FLOPs. If
  PR1 reports NULL, MP1-3 still run but are REPORTED against the PR1 context (expected null,
  documented).

### WP-09: atlas guards (AT4, AT5)

- Experiments: AT4 (programmatic ceiling reference), AT5 (probe-class sweep). The two cheapest decisive
  fork-movers with zero downloads.
- NEW files: `scripts/mop_at4_programmatic_ceiling.py`, `scripts/mop_at5_probe_class_sweep.py`,
  `tests/integration/test_mop_atlas_guards.py`.
- Mechanism: AT4 scores the handcrafted-descriptor and programmatic columns (WP-01 featurizers) on the
  same atlas factors as the perceptual caches, so a perceptual tie reads substrate-bounded vs
  test-too-easy; AT5 runs every existing atlas cell under linear, MLP, and nonlinear-gain probes.
- Depends on: WP-01 featurizer outputs, WP-02. Queue class: light (numpy + small probes). Difficulty: 2.
- Nulls: AT4 programmatic never exceeds perceptual on any factor (no tie is attributable to a substrate
  bound); AT5 every cell verdict invariant to probe class.

### WP-10: real-latent pilots on the existing 528K cache (DR2/PR3, WS5, CM4)

- Experiments: DR2 (=PR3) sparse-head forgetting pilot, WS5 slot-ablation pilot, CM4 workspace-shell
  pilot. All flagged PILOT: the registered claims need the DR1-scale stream (Studio).
- NEW files: `scripts/mop_dr2_sparse_real_pilot.py`, `scripts/mop_ws5_slot_ablation_pilot.py`,
  `scripts/mop_cm4_workspace_pilot.py`, `tests/integration/test_mop_real_pilots.py`.
- Mechanism: real_latent.real_task_stream over `data/cache/vjepa2_vitl_fpc64_256_real` (count 64);
  kWTA/MoE vs param-matched dense PLUS matched-activation-sparsity dense (DR2); shared slot ablated at
  fixed routing and capacity (WS5); workspace shell vs param-matched dense AND matched-FLOP unrolled
  depth (CM4). frozen_random is VALID here (trained-shell dynamics metrics).
- Depends on: WP-02 (heads, workspace, capmatch, BWT core). Queue class: heavy. Difficulty: 3.
- Nulls: DR2 sparse ties param-matched dense and sparsity-penalized dense on real latents; WS5
  slot-ablation ties the full model; CM4 workspace ties dense or unrolled depth or sign-flips.

### WP-11: post-encode encoder-lane jobs (PR2, AT3, DR8 typing arm)

- Experiments: PR2 (plasticity real vs random-init-ViT, the decisive learning-eases fork test, #7 on the
  shortlist), AT3 (time-axis ablation via a single-frame V-JEPA encode pass), DR8 typing arm.
- NEW files: `scripts/mop_pr2_plasticity_substrates.py` (consumes the random-init-ViT feature cache;
  reuses the encode logic already in `scripts/substrate_vs_random_init_vit.py`, does not duplicate it),
  `scripts/cache_randominit_vitl_features.py` (one logged encode pass writing
  `data/cache/randominit_vitl_nuisance`), `scripts/cache_vjepa_single_frame.py` (token/frame-matched
  single-frame pass writing `data/cache/vjepa2_vitl_singleframe`), `scripts/mop_at3_time_axis.py`,
  `tests/integration/test_mop_substrate_passes.py`.
- HARD RULE: every encode here queues strictly AFTER the in-flight V-JEPA encode, serially, exclusive.
  The registry explicitly forbids running PR2 concurrently with any ViT job.
- Depends on: WP-02 (adapter.py); encoder lane free. Queue class: encode, then heavy for the probes.
  Difficulty: 3.
- Nulls: PR2 adaptation speed and BWT on real V-JEPA within seed spread of random-init-ViT at matched
  resolution (the +0.31 is readout-only); AT3 full-clip equals single-frame for every factor at matched
  token count; DR8 typing decay present on random-init-ViT too (geometry, not substrate).

### WP-12: workspace layer (WS1-WS4)

- Experiments: WS1 (agreement vs confidence, the central gate), WS2 (matched-capacity fusion
  tournament), WS3 (arbitration, ONLY if WS1 positive), WS4 (broadcast-bottleneck bandwidth sweep).
- NEW files: `scripts/mop_ws1_agreement_vs_confidence.py`, `scripts/mop_ws2_fusion_tournament.py`,
  `scripts/mop_ws3_arbitration.py`, `scripts/mop_ws4_bandwidth_sweep.py`,
  `tests/integration/test_mop_workspace.py`.
- Mechanism: dual-source (zA = V-JEPA pooled, zB = DINOv2-S pooled) triplet store; WS1 ships all five
  preregistered controls: invertible-remap vacuity guard (linear probe A -> B predictions),
  shuffled-cross-source, noisy-TV, corr(errA, errB) headline, 5-seed CI with sign-flips. WS2 arms
  concat-MLP floor, learned-linear, cross-attention, GWT broadcast at strictly matched params and FLOPs
  (capmatch). WS4 sweeps slot width at FIXED total params vs tuned dropout/weight-decay at the same
  effective bandwidth (the preregistered regularization null).
- Depends on: WP-01 (DINOv2-S caches), WP-02 (cross_substrate, capmatch, riskcov). WS3 additionally
  gated on a positive WS1. Queue class: heavy (encodes already done by WP-01). Difficulty: 3 (WS1, WS2),
  2 (WS4), 3 (WS3).
- Nulls: WS1 agreement AUROC minus confidence AUROC <= 0 in CI or any guard fails; WS2 every structured
  fusion ties concat-MLP; WS3 inverse-variance ties averaging AND disagreement ties random routing; WS4
  bottleneck ties tuned regularization at matched capacity.

### WP-13: corruption and robustness (DR14 laptop arms)

- Experiments: DR14 (VQ, 4-bit, additive-noise arms; the dropped-channel arm needs dense latents and is
  Studio).
- NEW files: `scripts/mop_dr14_corruption.py`, `tests/integration/test_mop_corruption.py`.
- Mechanism: degrade cached latents (low-rank/VQ, quantize_dequantize 4-bit, noise), run each surviving
  reasoning primitive AND the matched single-pass baseline under IDENTICAL corruption, compare
  degradation SLOPES; noisy-TV guard on the noise arm.
- Depends on: WP-02, WP-03 (primitives to test), WP-05. Queue class: heavy. Difficulty: 2.
- Null: reasoning and single-pass degrade at the same rate under every corruption.

### WP-14: laptop atlas grid pilot (AT1 pilot, AL2 pilot)

- Experiments: AT1 laptop grid pilot (V-JEPA, DINOv2-S, single-frame V-JEPA, each vs its OWN random-init
  control on the nuisance factor; registered full grid stays Studio), AL2 pilot (thin linear map per
  substrate pair vs random-map-of-equal-rank floor).
- NEW files: `scripts/mop_at1_grid_pilot.py`, `scripts/mop_al2_alignment_pilot.py`,
  `tests/integration/test_mop_atlas_grid.py`.
- Mechanism: cross_substrate.py over the WP-01 and WP-11 caches; nine-verdict decision order from
  06_cognitive_currencies_atlas.md applied top-down (random-control-artifact first).
- Depends on: WP-01 caches, WP-11 caches, WP-02. Queue class: heavy (probes only). Difficulty: 2.
- Nulls: every substrate delta over its own random-init within seed spread; learned map ties the
  random map of equal rank (alignment-artifact).

Summary: 14 WPs. WP-02 is the sole owner of every existing-file extension. No two WPs write the same
file.

---

## 3. BUILD ORDER (DAG + parallel lanes)

Dependency DAG (build-time, not run-time):

```
WP-02 (integrator, interface stubs first)
  -> WP-03 -> WP-04
  -> WP-05 -> WP-13 (also needs WP-03)
  -> WP-06
  -> WP-07
  -> WP-08 (also reads the PR1 verdict at run time)
  -> WP-10
  -> WP-11
  -> WP-12 (also needs WP-01 caches at run time)
  -> WP-14 (also needs WP-01 and WP-11 caches at run time)
WP-01 (downloads now; cache scripts author any time; encodes queue in the encoder lane)
  -> WP-09 (featurizer outputs)
  -> WP-12, WP-14 (caches)
```

Parallel lane assignment (each lane is one agent; no file collisions by construction):

- Lane A: WP-02 (stubs first, then full integrator), then WP-10.
- Lane B: WP-01 (start downloads immediately), then WP-14.
- Lane C: WP-03, then WP-04.
- Lane D: WP-05, then WP-13.
- Lane E: WP-06, then WP-07.
- Lane F: WP-08, then WP-09.
- Lane G: WP-11, then WP-12.

Lanes C-G may begin as soon as WP-02's interface stubs (module names, function signatures, config keys)
are committed; they build against stubs and go green when the integrator lands. Every WP ships its own
test file; `make accept` is the definition of done for the whole build.

---

## 4. MAXIMIZATION RUN QUEUE

No wall-clock cap. Default 5 seeds per run (seeds 0-4 via per-run `seed=<s>` overrides, never
experiment.seeds); survivors rerun at 10 seeds in Stage 4. CPU classes: light = concurrent with an
encode is fine; heavy = exclusive CPU (never two heavy at once, never heavy during an encode);
encode = the encoder lane, strictly serial, strictly after the in-flight V-JEPA job. All outputs under
`runs/mot/`. The CPU is never idle: while the encoder lane runs Stage 2, no heavy job runs, but light
jobs and code review may proceed; otherwise heavy jobs run back to back.

### Stage 0 (NOW, during the in-flight V-JEPA encode; light/network only; RAM ceiling mandatory per row)

- Q0.1 stage small substrates. cmd `python scripts/stage_small_substrates.py`
  seeds n/a, class light (network only, RAM <200MB), ~15 min, out `runs/mot/staging_manifest.json`

RECLASS NOTE: Q0.2/Q0.3 (the two featurizers) were originally slotted here but CANNOT run during the
encode. The nuisance clips they featurize are drawn in memory by torch RNG in
`scripts/compositional_under_nuisance.py` and never persisted; numpy cannot replay torch RNG streams,
and importing torch to regenerate clips is forbidden while the encode owns the CPU. They now run as
Q1.0a/Q1.0b, first thing after the encode exits and strictly before Q1.2 (WP-01 clip identity rule).

### Stage 1 (after the V-JEPA encode exits; featurizers Q1.0a/Q1.0b first, then heavy; PR1 read first)

Gate read: `runs/pre_studio/pr1_mode_error_disjointness.json`. PR1 GREEN (heterogeneous oracle gain >
homogeneous + seed SD) licenses the MT routing rows as live tests; PR1 NULL demotes MP1/MP2/MP3/MP8/DR12
to run-and-report-against-context (they still run, preregistered as expected nulls).

Priority order follows section 14.1 rankings, then gate structure. Q1.0a/Q1.0b run FIRST (they were
Stage 0 rows; reclassed per the WP-01 clip identity rule) and must land before Q1.2:

- Q1.0a programmatic featurizer (torch-CPU clip regeneration, streamed per clip, uint8, RAM <1GB).
  cmd `python scripts/featurize_programmatic.py`
  seeds n/a, class light post-encode, ~10 min, out `runs/mot/at4_programmatic_features.json`
- Q1.0b handcrafted featurizer (same streaming rule, RAM <1GB; blocks Q1.2 until landed).
  cmd `python scripts/featurize_handcrafted.py`
  seeds n/a, class light post-encode, ~20 min, out `runs/mot/at4_handcrafted_features.json`
- Q1.1 AT5 probe-class sweep. cmd `python scripts/mop_at5_probe_class_sweep.py --seeds 0-4`
  5 seeds, class light, ~30 min, out `runs/mot/at5_probe_class_sweep.json`
- Q1.2 AT4 programmatic ceiling. cmd `python scripts/mop_at4_programmatic_ceiling.py --seeds 0-4`
  5 seeds, class light, ~45 min, out `runs/mot/at4_programmatic_ceiling.json`
- Q1.3 MP5 adaptive halting. cmd `python scripts/mop_mt5_adaptive_halting.py --seeds 0-4`
  5 seeds, class heavy, ~60 min, out `runs/mot/mt5_adaptive_halting.json`
- Q1.4 MP6 confidence stopping. cmd `python scripts/mop_mt6_confidence_stop.py --seeds 0-4`
  5 seeds, class heavy, ~30 min, out `runs/mot/mt6_confidence_stop.json`
- Q1.5 DR9 verify-revise. cmd `python scripts/mop_dr9_verify_revise.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/dr9_verify_revise.json`
- Q1.6 DR8 fixed point, V-JEPA arm. cmd `python scripts/mop_dr8_fixed_point.py --cache vjepa --seeds 0-4`
  5 seeds, class heavy, ~30 min, out `runs/mot/dr8_fixed_point_vjepa.json`
- Q1.7 MP7 beam search. cmd `python scripts/mop_mt7_beam_search.py --seeds 0-4`
  5 seeds, class heavy, ~60 min, out `runs/mot/mt7_beam_search.json`
- Q1.8 DR6 rollout planning. cmd `python scripts/mop_dr6_rollout_planning.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/dr6_rollout_planning.json`
- Q1.9 DR13 horizon limit. cmd `python scripts/mop_dr13_horizon_limit.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/dr13_horizon_limit.json`
- Q1.10 DR11 MC rollouts (noisy-TV guarded). cmd `python scripts/mop_dr11_mc_rollouts.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/dr11_mc_rollouts.json`
- Q1.11 DR10 retrieve-then-reason. cmd `python scripts/mop_dr10_retrieve_reason.py --seeds 0-4`
  5 seeds, class heavy, ~30 min, out `runs/mot/dr10_retrieve_reason.json`
- Q1.12 PR8 retrieval head. cmd `python scripts/mop_pr8_retrieval_head.py --seeds 0-4`
  5 seeds, class heavy, ~30 min, out `runs/mot/pr8_retrieval_head.json`
- Q1.13 PR7 fast/slow weights. cmd `python scripts/mop_pr7_fast_slow.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/pr7_fast_slow.json`
- Q1.14 PR4 epistemic gate vs noisy-TV. cmd `python scripts/mop_pr4_epistemic_gate.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/pr4_epistemic_gate.json`
- Q1.15 PR5 content-gated critical period. cmd `python scripts/mop_pr5_content_gated_cp.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/pr5_content_gated_cp.json`
- Q1.16 PR6 sleep consolidation. cmd `python scripts/mop_pr6_sleep_consolidation.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/pr6_sleep_consolidation.json`
- Q1.17 AL1 uncertainty router. cmd `python scripts/mop_al1_uncertainty_router.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/al1_uncertainty_router.json`
- Q1.18 DR12 disagreement (PR1 context). cmd `python scripts/mop_dr12_disagreement.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/dr12_disagreement.json`
- Q1.19 MP8 latent debate (PR1 context). cmd `python scripts/mop_mt8_latent_debate.py --seeds 0-4`
  5 seeds, class heavy, ~60 min, out `runs/mot/mt8_latent_debate.json`
- Q1.20 MP1/MP2/MP3 router pilots (PR1 context, synthetic; real answer stays blocked on DR1).
  cmd `python scripts/mop_mt123_router_pilots.py --seeds 0-4`
  5 seeds, class heavy, ~90 min, out `runs/mot/mt123_router_pilots.json`
- Q1.21 DR14 corruption arms (VQ/4-bit/noise; runs after Q1.3-Q1.8 so it can test the survivors).
  cmd `python scripts/mop_dr14_corruption.py --seeds 0-2`
  3 seeds, class heavy, ~60 min, out `runs/mot/dr14_corruption.json`
- Q1.22 DR2/PR3 sparse-real pilot (528K cache). cmd `python scripts/mop_dr2_sparse_real_pilot.py --seeds 0-4`
  5 seeds, class heavy, ~60 min, out `runs/mot/dr2_sparse_real_pilot.json`
- Q1.23 WS5 slot-ablation pilot. cmd `python scripts/mop_ws5_slot_ablation_pilot.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/ws5_slot_ablation_pilot.json`
- Q1.24 CM4 workspace pilot. cmd `python scripts/mop_cm4_workspace_pilot.py --seeds 0-2`
  3 seeds, class heavy, ~60 min, out `runs/mot/cm4_workspace_pilot.json`

Stage 1 subtotal: ~19.5 h (~0.5 h light post-encode featurizers + ~19 h heavy CPU).

### Stage 2 (encoder lane, strictly serial, one model at a time, nothing heavy alongside)

- Q2.1 random-init ViT-L feature pass (PR2 prerequisite; the registry OOM rule applies).
  cmd `python scripts/cache_randominit_vitl_features.py`
  1 pass, class encode, ~90 min, out `data/cache/randominit_vitl_nuisance` +
  `runs/mot/cache_randominit_vitl.json`
- Q2.2 single-frame V-JEPA pass (AT3, token/frame matched).
  cmd `python scripts/cache_vjepa_single_frame.py`
  1 pass, class encode, ~40 min, out `data/cache/vjepa2_vitl_singleframe` +
  `runs/mot/cache_singleframe.json`
- Q2.3 DINOv2-S real + random-init encode. cmd `python scripts/cache_dinov2s_nuisance.py`
  1 pass, class encode, ~30 min, out `data/cache/dinov2s_nuisance_{real,randominit}` +
  `runs/mot/cache_dinov2s.json`
- Q2.4 Qwen2.5-0.5B textified encode + random-init. cmd `python scripts/cache_qwen_textified.py`
  1 pass, class encode, ~60 min, out `data/cache/qwen05b_textified_{real,randominit}` +
  `runs/mot/cache_qwen.json`
- Q2.5 wav2vec2-base sonified encode + random-init. cmd `python scripts/cache_wav2vec2_sonified.py`
  1 pass, class encode, ~45 min, out `data/cache/wav2vec2_sonified_{real,randominit}` +
  `runs/mot/cache_wav2vec2.json`

Stage 2 subtotal: ~4.5 h encoder lane. Light jobs (aggregation, doc updates) may run alongside ONLY
with a stated RAM ceiling <500MB each, per the section 1 RAM accounting rule; heavy jobs may NOT.

### Stage 3 (post-cache probes and the workspace layer; heavy lane resumes)

- Q3.1 PR2 plasticity real vs random-init-ViT (the decisive fork row, 14.6 #1).
  cmd `python scripts/mop_pr2_plasticity_substrates.py --seeds 0-4`
  5 seeds, class heavy, ~90 min, out `runs/mot/pr2_plasticity_substrates.json`
- Q3.2 AT3 time-axis ablation. cmd `python scripts/mop_at3_time_axis.py --seeds 0-4`
  5 seeds, class heavy, ~30 min, out `runs/mot/at3_time_axis.json`
- Q3.3 DR8 typing arm on random-init cache.
  cmd `python scripts/mop_dr8_fixed_point.py --cache randominit_vitl --seeds 0-4`
  5 seeds, class heavy, ~20 min, out `runs/mot/dr8_fixed_point_randominit.json`
- Q3.4 WS1 agreement vs confidence (central WS gate; all five guards preregistered).
  cmd `python scripts/mop_ws1_agreement_vs_confidence.py --seeds 0-4`
  5 seeds, class heavy, ~60 min, out `runs/mot/ws1_agreement_vs_confidence.json`
- Q3.5 WS2 fusion tournament. cmd `python scripts/mop_ws2_fusion_tournament.py --seeds 0-4`
  5 seeds, class heavy, ~90 min, out `runs/mot/ws2_fusion_tournament.json`
- Q3.6 WS4 bandwidth sweep. cmd `python scripts/mop_ws4_bandwidth_sweep.py --seeds 0-4`
  5 seeds, class heavy, ~90 min, out `runs/mot/ws4_bandwidth_sweep.json`
- Q3.7 WS3 arbitration, ONLY if WS1 positive (else logged SKIPPED with the WS1 verdict cited).
  cmd `python scripts/mop_ws3_arbitration.py --seeds 0-4`
  5 seeds, class heavy, ~60 min, out `runs/mot/ws3_arbitration.json`
- Q3.8 AT1 laptop grid pilot (nine-verdict classification across all cached columns).
  cmd `python scripts/mop_at1_grid_pilot.py --seeds 0-4`
  5 seeds, class heavy, ~60 min, out `runs/mot/at1_grid_pilot.json`
- Q3.9 AL2 alignment pilot (random-map-of-equal-rank floor).
  cmd `python scripts/mop_al2_alignment_pilot.py --seeds 0-4`
  5 seeds, class heavy, ~45 min, out `runs/mot/al2_alignment_pilot.json`

Stage 3 subtotal: ~9 h heavy CPU.

### Stage 4 (robustness arms and 10-seed reruns; verdict-dependent, not optional)

- Q4.1 10-seed reruns of every Stage 1-3 row whose primary delta lies within 2x the 5-seed spread or
  whose sign flipped once. cmd: same script, `--seeds 0-9 --rerun`. Budget ~8 rows x ~75 min = ~10 h,
  class heavy, outs `runs/mot/<row>_seeds10.json`.
- Q4.2 noisy-TV robustness arms rerun for every row that passed its primary null but sits within 2x
  spread on the guard (PR4, DR11, DR12, AL1, WS1). class heavy, ~2 h total, outs
  `runs/mot/<row>_noisytv_arm.json`.
- Q4.3 MP1-3 rerun against the FINAL PR1 verdict with the tuned single-mode baselines re-tuned (the
  tuned-baseline doctrine). cmd `python scripts/mop_mt123_router_pilots.py --seeds 0-9 --tuned`
  10 seeds, class heavy, ~3 h, out `runs/mot/mt123_router_pilots_tuned.json`
- Q4.4 DR14 extended corruption grid over every Stage 1 survivor. class heavy, ~2 h, out
  `runs/mot/dr14_corruption_extended.json`
- Q4.5 aggregate verdict report (sign-flips, seed CIs, gate table).
  cmd `python scripts/mop_aggregate_report.py`
  class light, ~10 min, out `runs/mot/aggregate_report.json`

Stage 4 subtotal: ~17 h heavy CPU.

### Queue totals

45 queue entries (1 + 26 + 5 + 9 + 4 grouped rerun blocks) covering 30 distinct runnable experiments.
Estimated total laptop wall time: ~0.25 h Stage 0 (concurrent with the in-flight encode) + ~19.5 h
Stage 1 + ~4.5 h Stage 2 (encoder lane) + ~9 h Stage 3 + ~17 h Stage 4 = approximately 50 hours of
scheduled machine time, of which ~45 h is exclusive heavy CPU. Sequencing keeps the CPU busy and never
oversubscribed: exactly one heavy OR one encode job at any moment, light jobs fill the encoder-lane
windows within their stated RAM ceilings.

Sequential gate logic recap: PR1 gates the MT interpretation (Stage 1); AT4/AT5 verdicts qualify every
later tie (a tie without a passing programmatic column is unreadable and must be reported as
test-too-easy); WS1 gates WS3; Stage 2 caches gate Stage 3; Stage 1-3 verdicts select the Stage 4 rerun
set. If PR1 nulls, MP1-3/MP8/DR12 are still run and reported against the PR1 context.

---

## 5. STUDIO APPEND (exact markdown to append to STUDIO_HANDOFF.md)

Append the following after the Transfer Checklist section, preceded by a `---` rule. It follows the
file's native heading and table conventions but uses house style (no em or en dashes).

```markdown
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
```

---

## 6. GO/STATUS APPEND (exact markdown for GO.md)

Append the following as a new `## ` section in GO.md between "Current validated state" and "What to do
on the Studio" (short, imperative, path-anchored, house style, no dashes):

```markdown
## MoT lane (mixture of perspectives): what it is and how to resume

A second execution lane exists: docs/mixture_of_perspectives/EXECUTION_MANIFEST.md. It is the single
operational plan for the MoT experiment bank (MT/DR/PR/WS/AT/AL/CM registry in
docs/mixture_of_perspectives/11_experiment_registry.md). The laptop ran a maximization queue into
runs/mot/ (~30 experiments, 5 seeds default, survivors at 10 seeds), with verdicts aggregated in
runs/mot/aggregate_report.json.

To resume on any machine:

1. Read docs/mixture_of_perspectives/EXECUTION_MANIFEST.md sections 1 and 4. Respect the live-state rules:
   one encoder at a time, never two heavy CPU jobs, no dense caches on a laptop.
2. Check runs/mot/ against the Stage 0-4 queue in the manifest. Any missing json is an unrun row; rerun
   it with its listed command and seeds. runs/pre_studio/pr1_mode_error_disjointness.json gates the MT
   routing rows.
3. On the Studio, execute the appended "Mixture-of-Perspectives lane" section of STUDIO_HANDOFF.md in table
   order. DR1 (real bound-attribute video cache) is the binding constraint; the multi-seed
   substrate_vs_random_init rerun owns the headline substrate number.
4. Hard invariants are unchanged: encoder FROZEN, preregistered null per row, non-vacuous controls
   (random-init same-arch, never a square latent projection), matched compute, noisy-TV guard, seed
   sweeps via per-run seed overrides. The custom-model line stays UNLICENSED until the section 8 gates
   fail on real content.
```

For STATUS.md, append one new session block at the end of the file in its native checkbox voice when
the build lands, listing: manifest written, WP-01..WP-14 built (test counts), Stage 0-4 queue progress,
and the PR1/WS1/AT4 gate verdicts as they arrive. STATUS.md is append-only; never edit prior blocks.

---

## Appendix: preregistered verdict thresholds fixed before running

- Routing rows (MP1-3, MP8, DR12): a win requires delta > 5-seed spread at matched compute AND no sign
  flip; PR1-null context demotes any win to PLAUSIBLE-BUT-UNVERIFIED pending the Studio regime.
- Halting rows (MP5, MP6): matched on MEAN FLOPs within tol 0.10 (matched_within); halting entropy ~0
  is an automatic null (constant halt head).
- Search rows (MP7, DR6, DR11, DR13): matched on TOTAL FLOPs including pruned/discarded work.
- Plasticity rows (PR4-PR8, DR2): scored on trained-shell dynamics (BWT, forgetting-area, LR-integral,
  adaptation speed); frozen_random is valid ONLY here; matched LR-integral for PR5, matched gradient
  steps for PR6, matched capacity for PR7/PR8.
- Substrate rows (PR2, AT1, AT3, DR8 typing, AL2, WS1): only random-init same-arch or random-encoder
  controls count; every verdict passes the nine-verdict decision order (random-control-artifact checked
  first); ties are reported only with a passing AT4 programmatic column and a D3 certificate.
- Noisy-TV rows (PR4, DR11, DR12, AL1, WS1, WS3): the three-boolean contract from
  diagnostics/noisy_tv.py must pass jointly; a guard failure overrides any primary win.
