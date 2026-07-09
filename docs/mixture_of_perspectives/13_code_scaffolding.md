# Section 13: Code Scaffolding (research to architecture)

This section converts the research program into a concrete module architecture and, crucially, reconciles the spec's proposed module list against the code that ALREADY EXISTS in `src/mop/`. The single most important finding of this section: most of the spec's modules are already implemented under different names, and the honest job is to EXTEND them, not to duplicate them. Building parallel `WorkspaceShell`/`ReplayMemory`/`PlasticityController` classes when `shell/modulation.py`/`shell/buffer.py`/`shell/plasticity.py` already exist would fork the codebase and split the test suite. Where a spec module has no code counterpart (`CrossSubstrateAgreement`, `MixtureArbitrator` as a first-class object), this section says NEW and justifies it.

The tree audited (read-only, no code executed): `src/mop/{substrate,shell,learning,experiments,harness,diagnostics,metrics,studies,studio,devel}` plus top-level `config.py`, `provenance.py`, `seeding.py`, `logging_utils.py`, and `scripts/`.

A note on doctrine before any module: nothing here loosens a control. Every proposed extension inherits the standing controls already wired at the harness level (frozen-random, matched-compute, tuned-baseline, noisy-TV, seed-stability, determinism, difficulty-calibration, cross-substrate). The vacuous-control discovery (frozen_random_projection is a full-rank invertible 1024x1024 map, so probe deltas are forced to 0.000) is a HARD constraint on any module whose output feeds a probe: it means the substrate specialness question is answered ONLY by `SubstrateAdapter`/`AlignmentSuite` running real-encoder-vs-random-ENCODER (or random-init-ViT) features, never by a within-latent projection control. Modules are marked accordingly.

Legend for each module: EXISTS (extend), PARTIAL (some of it exists, gap named), NEW (justify why it is not a duplicate).

---

## Registry and substrate layer

### SubstrateRegistry
- Status: EXISTS. `substrate/encoder_registry.py` (pure-config registry over `configs/encoder/*.yaml`) plus `studio/registry.py` (dataset/model acquisition registry over `registry/models.yaml`, `registry/datasets.yaml`). The spec's "SubstrateRegistry" is these two, already split by concern (which encoders exist and are honest vs which weights/datasets are acquirable).
- Purpose: answer "which frozen substrates exist, which are HONEST about having real weights" without ever touching the network. Guards the failure mode where a config silently lets frozen-random latents masquerade as V-JEPA latents.
- Inputs: `configs/encoder/*.yaml`, `registry/models.yaml`. Outputs: `list_encoders()` (name, hf_id, embed_dim, dense, available, prefer_real), `is_honest()`, `verified_real_ids()`.
- Minimal impl (present): the hand-verified `VERIFIED_REAL_IDS` frozenset plus a config read.
- Full impl (extension needed for cross-substrate): today the registry knows ONE architecture family (V-JEPA 2 ViT-L/H/g). The cross-substrate convergence control (standing control 8) needs it to enumerate DISTINCT-objective/distinct-modality encoders (e.g. a DINOv2, an image-JEPA, an audio encoder) as first-class rows, each with a `family` and `objective` field so `CrossSubstrateAgreement` can group by them. Add `family` and `training_objective` keys to the encoder schema and to `list_encoders()`.
- Dependencies: `config.py`, `omegaconf`. Used by: every cache script, `validate.validate_encoder`, `studio/*`, and (after extension) `CrossSubstrateAgreement`.
- Laptop-safe: yes (pure config). Studio-scale: yes. Prepares-for-custom-model: yes, the `family`/`objective` fields are exactly where a custom encoder row would live.

### LatentStore
- Status: EXISTS. `substrate/latent_store.py` (memmap-backed `[N, *feat]` store with keys, labels, meta, provenance sidecar), plus `substrate/cache.py` (the encode-once pipeline), `substrate/real_latent.py` (real-encoder-store to stream/factorized-arrays adapters), `substrate/storage.py`, `substrate/cache_tools.py`, and `substrate/cache_manifest.py` (Studio cache data-plane receipt).
- Purpose: the disk-backed array the whole shell trains against. Encoder is frozen, so latents are computed once and read forever (the laptop-feasibility keystone). Because the encoder never trains, stored latents never go stale.
- Inputs: a clip source through `FrozenEncoder.encode`. Outputs: `latents.npy` memmap `[N,*feat]`, `keys.npy` `[N,key_dim]`, optional `labels.npy`, `meta.json`, `provenance.json`, optional `factors.json` and `splits.json`, and `cache_manifest.json` when a Studio receipt is written.
- Minimal impl (present): synthetic clips path, pooled `[N,1024]` stores under `data/cache/`.
- Full impl (gap): the doctrine's #4 deferred prerequisite is DENSE `[N,T,P,1024]` real-video stores with non-additively bound attributes. `feat_shape` already supports dense shapes, and `cache_manifest.py` now records array fingerprints, encoder config hash, factor sidecars, split membership, and a columnar index. The remaining gap is a Studio cache script that ingests natural video with entangled color/shape/position/motion labels and writes the receipt at the end. This is now a data/acquisition and scheduler gap, not a store-code gap.
- Dependencies: numpy memmap, `provenance.py`. Used by: `real_latent`, every real-encoder experiment, `SubstrateAdapter`, `ProbeSuite`.
- Laptop-safe: yes (memmap, mmap_mode='r'). Studio-scale: yes (append-friendly capacity, finalize to true length). Prepares-for-custom-model: yes, a custom encoder writes the identical store format, so all downstream code is encoder-agnostic.

### EncodeScheduler
- Status: EXISTS. `studio/encode_scheduler.py` consumes the Wave-0 CPU/MPS benchmark, the active profile, an encoder config, a requested clip count, and a dense/pooled flag, then emits an encode launch plan. `scripts/mop_encode_autoselect.py` now writes both `runs/mot/encode_device.json` and `runs/mot/encode_schedule.json`; `scripts/studio/dr1_schedule_plan.py` turns that schedule into DR1 gate, leg, merge, and A6-guard commands after `scripts/studio/dr1_source_intake.py` proves the source receipt is clean.
- Purpose: make the Studio encode path profile-owned instead of hand-owned. The scheduler picks MPS vs parallel CPU workers from measured s/clip, estimates cache footprint, enforces the profile's start and post-cache disk floors, checks wall clock against the profile cap, and emits checkpoint cadence.
- Inputs: benchmark record (`cpu_s_per_clip`, `mps`), optional `memory_envelope`, `profile_name`, encoder config, requested clips, dense-token flag. Outputs: winner, candidates, cache estimate, disk projection, memory envelope, gates, blocked reasons, checkpoint cadence, and next command.
- Minimal impl (present): pure planning, no model load and no encode. CPU worker defaults are profile-specific (`m3pro-local-max` 1, `studio-1tb` 8, `studio-m1ultra` 16). Dense-cache disk gates use `storage.estimate_for_encoder`. `mop_encode_autoselect.py` now records process/system/MPS memory snapshots and writes a blocked JSON receipt if model files are absent instead of losing the run to a traceback.
- Full impl: `src/mop/studio/dr1_schedule.py` consumes `encode_schedule.json` directly, splits DR1 shard legs by the checkpoint cadence, carries the measured device into `dr1_curate_bound_video.py --device`, and can emit a `mop-long-run-daemon/v1` plan. The daemon supplies heartbeat/resume; the DR1 merge and A6 guard remain explicit post-encode jobs.
- Dependencies: `profiles.py`, `storage.py`, `devices.py`, `memory_envelope.py`. Used by: Studio Wave 0 microbench, DR1 cache build, dense cache planning.
- Laptop-safe: yes (pure arithmetic). Studio-scale: yes. Prepares-for-custom-model: yes, a custom encoder cache still prices through the same profile and receipt path.

### DR1SourceCard
- Status: EXISTS. `studio/dr1_source_intake.py` exposes the source-card builder/validator, and
  `scripts/studio/dr1_source_card.py` writes or validates `mop-dr1-source-card/v1`.
- Purpose: turn the human provenance handoff into a receipt before the source tree is traversed. The
  validation receipt refuses TODO/unknown source ids, licenses, allowed-use text, non-natural provenance
  tags, missing non-overlap proof, manual-license cards without accepted terms, and clip-count mismatch
  when an expected count is supplied.
- Inputs: source id, license, allowed use, provenance tag, non-overlap proof, manual-license flag,
  accepted terms, optional URLs/notes, and optional expected clip count. Outputs:
  `runs/studio_dr1/dr1_source_card.json` plus `runs/studio_dr1/dr1_source_card_validation.json`.
- Minimal/full impl: present. The CLI has `template` and `validate` modes; the Studio spine runs
  validation before `dr1_source_intake`.
- Dependencies: `provenance.RESULT_TAGS`. Used by: DR1 source intake, DR1 spine, DR1 artifact bundle.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: yes, any future source can reuse the
  same provenance card before producing cache evidence.

### DR1SourceIntake
- Status: EXISTS. `studio/dr1_source_intake.py` and `scripts/studio/dr1_source_intake.py` write a `mop-dr1-source-intake/v1` receipt before DR1 scheduling.
- Purpose: make the real-video source a launch gate, not an operator assumption. The receipt checks class-foldered bound cells, the per-cell floor, factor value counts, unique clip stems, `captions.json` coverage, the cheap label-free caption-recoverability probe, and a validated source card with license/provenance/non-overlap proof.
- Inputs: source dir, factor list, min clips per cell, and a source-card JSON. Outputs: manifest summary, cell summary, caption coverage, caption recoverability, duplicate-stem list, source-card summary, problems, and `all_ok`.
- Minimal/full impl: present. The gate is filesystem-only plus a cheap text probe; it does not decode video, download data, or load an encoder. The Studio spine runs it before `dr1_schedule_build`; `dr1_schedule_plan.py --source-intake` refuses to emit runnable jobs when it is blocked.
- Dependencies: `DR1SourceCard`, `substrate.video.validate_source`, `provenance.RESULT_TAGS`. Used by: DR1 spine, DR1 artifact bundle, and the source/license pre-compute gate.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: yes, every later source can reuse the same source-card contract before producing cache evidence.

### NullCardGenerator
- Status: EXISTS. `falsification/null_cards.py` and `scripts/null_card_tool.py` generate draft null/survival cards from `registry/experiments.yaml`, validate the fenced YAML block in a card, and expose the schema in `proof/NULL_CARDS/null_card.schema.json`.
- Purpose: make each preregistered claim's null, metric, falsifier, probe dependency, seed threshold, provenance tag, verdict, and raw-run receipt explicit before Studio compute can turn into narrative drift.
- Inputs: one experiment id from the registry, or one existing null-card markdown file. Outputs: a draft markdown card, a JSON-schema-like contract, or validation problems.
- Minimal impl (present): registry -> draft card (`generate`), card -> structural validation (`validate`), strict mode that refuses TODO placeholders, and a tolerant parser for historical cards whose prose values contain colons.
- Full impl (next): have each Studio claim writer call strict validation before adding a positive/null to docs, then auto-attach the generated receipt path to `STUDIO_RUN_REPORT.md`.
- Dependencies: `devel.registries`, `provenance.RESULT_TAGS`, `proof/NULL_CARDS/_TEMPLATE.md`. Used by: Studio DR1/PR9 preregistration, adversarial verification receipts, null-card hygiene.
- Laptop-safe: yes (pure text). Studio-scale: yes. Prepares-for-custom-model: yes, Process C pilots get the same null-card contract before training.

### SubstrateAdapter
- Status: EXISTS. `substrate/adapter.py` now defines `SubstrateAdapter`, `RealEncoderAdapter`, `RandomInitViTAdapter`, `RandomPixelAdapter`, and `SubstrateRegistry`, with tests in `tests/unit/test_mot_shared_modules.py`.
- Purpose: present ANY substrate (real frozen V-JEPA, random-init ViT, random-pixel projection, a future custom encoder) behind ONE `features(clips) -> [N,D]` interface, so the substrate-is-special comparison is a swap of adapter, not a rewrite. This is the object that makes the corrected control (real-encoder vs random-ENCODER, NOT within-latent projection) a first-class citizen.
- Inputs: clips or a `LatentStore`. Outputs: `[N,D]` (or dense) features plus a `substrate_tag` (real-encoder / random-init-vit / random-pixel / custom).
- Minimal impl: done. The corrected controls are reusable adapters with a common ABC (`extract(clips) -> tensor`, `tag -> str`).
- Full impl: next extension is to connect adapter registration to encoder-family metadata from `SubstrateRegistry`, so `AlignmentSuite`/`CrossSubstrateAgreement` can iterate substrates by family/objective rather than by hand-written scripts.
- Dependencies: `FrozenEncoder`, `LatentStore`, `SubstrateRegistry`. Used by: `AlignmentSuite`, `ProbeSuite`, `CrossSubstrateAgreement`, and the two landed/in-flight substrate scripts.
- Laptop-safe: real V-JEPA arm is CPU-bound and heavy (the running encode job); the random-pixel/random-init arms are cheap. MUST honor the hard constraint: never spawn a second torch/encoder job while the V-JEPA encode is running. Studio-scale: yes. Prepares-for-custom-model: this IS the seam a custom encoder plugs into.

### PerspectiveAdapter
- Status: EXISTS. `perspectives/adapter.py` defines `PerspectiveMeta`, `PerspectiveBatch`, `PerspectiveAdapter`, `TensorPerspectiveAdapter`, `LatentStorePerspectiveAdapter`, `SubstratePerspectiveAdapter`, `PerspectiveRegistry`, `build_perspective_matrix`, and `perspective_audit`.
- Purpose: present each named perspective (vision-static, vision-motion, language, audio, code, math, and each matched control) as features over IDENTICAL referents. This is the Studio DR1 plurality contract: the matrix builder aligns by referent id, refuses missing or extra referents, carries supervised/derived/license flags, and the audit names substantive perspectives with no matched control.
- Inputs: cached tensors, a `LatentStore`, or a `SubstrateAdapter` plus clips and referent ids. Outputs: a `PerspectiveMatrix` with `features[tag] -> [N,D]`, metadata, factors, and control mappings.
- Minimal impl: present. No model loads at construction. Cached/tensor views are immediate, store views read memmaps, and substrate-backed views call the existing `SubstrateAdapter` sequentially.
- Full impl: DR1 merge now emits `perspective_matrix_receipt.json` when `--source` provides paired captions and a root merged store exists; if the root store is absent, it writes an explicit blocked receipt instead of implying alignment. The receipt currently covers vision plus caption and reports missing matched controls. Next extension: add audio/code/math arms when their feature stores exist, then feed the receipt to `AlignmentSuite`, A6 residualization, and null-card generation before any positive enters docs.
- Dependencies: `SubstrateAdapter`, `LatentStore`, torch. Used by: DR1 multi-arm encode, AL2/cross-perspective alignment, facet-15 ecology runs, Process C dense-token pilots.
- Laptop-safe: yes for tensor/store views and random controls. Studio-scale: yes. Prepares-for-custom-model: yes, a Process C or custom-substrate arm is just another perspective with its own control and license/provenance flags.

---

## Probing and alignment layer

### AlignmentSuite
- Status: EXISTS. `diagnostics/alignment.py` wraps `diagnostics/geometry.py` (linear/kernel CKA, RSA, effective rank, anisotropy, intrinsic dim, NN-overlap) and `diagnostics/seed_consistency.py` into a pairwise report and a multi-arm `mop-alignment-suite/v1` table.
- Purpose: measure representational agreement (between two substrates, between seeds of the same shell, between a substrate and a target RDM). The load-bearing scientific use: quantify whether two DIFFERENT-objective substrates converge (universal structure) or diverge (modality/objective-specific), the newest doctrinal control.
- Inputs: two or more `[N,D]` representations of the SAME points. Outputs: pairwise CKA matrix, RSA correlations, effective-rank/anisotropy per substrate, cross-seed CKA vs frozen-random floor.
- Minimal impl: present. `alignment_suite(x, y)` keeps the historical pair report; `alignment_suite({tag: tensor})` assembles self-geometry, pair metrics, metric matrices, row-shuffle p-values, and warnings when no random-encoder control is present.
- Full impl (next): add Procrustes/CCA alignment and route DR1 `perspective_matrix_receipt.json` into an alignment receipt after the merged cache exists.
- Dependencies: `geometry`, `seed_consistency`. Used by: EX12 atlas, P5/S5/Y3 idiolect tests, `CrossSubstrateAgreement`.
- Laptop-safe: yes (CKA is a few matmuls). Studio-scale: yes. Prepares-for-custom-model: yes, comparing a custom encoder's geometry to V-JEPA is exactly a CKA/RSA call.
- Doctrine flag: CKA is rotation-invariant, so it CANNOT distinguish real from a full-rank random projection either (same vacuous-control trap as probes). The suite now warns when a random-encoder control tag is absent; any AlignmentSuite claim of specialness must use a random-ENCODER arm, not frozen_random_projection.

### ProbeSuite
- Status: EXISTS. `diagnostics/linear_probe.py`, `diagnostics/nonlinear_probe.py` (with the readout-contribution difference-in-differences index), `diagnostics/held_out_combo.py` (compositionality gate), `diagnostics/bottleneck.py` (capability-per-bit), `diagnostics/difficulty_calibration.py` (D3, certifies a regime is separable before a tie is trusted).
- Purpose: decode a target factor from a latent under controlled conditions, the single most reused diagnostic. Answers "is X in the latent at all" (linear), "is X there NONLINEARLY beyond a projection" (readout-contribution index), "does the code FACTORIZE" (held-out-combo), "in how few bits" (bottleneck).
- Inputs: `[N,D]` features, `[N]` labels, optionally a factor split. Outputs: probe accuracy, chance floor, per-arm deltas, readout-contribution index, held-out-combo gap.
- Minimal/full impl: present. The only extension is discipline, not code: the vacuous-control finding means a bare linear-probe delta vs frozen_random_projection is meaningless. ProbeSuite's default comparison arm for substrate-specialness claims must be routed through `SubstrateAdapter` (random-ENCODER features), not `substrate_ablation.frozen_random_projection`. Add a deprecation-style guard: `linear_probe` results tagged with an `is_substrate_specialness_claim` flag route to the adapter path.
- Dependencies: `seeding`, `substrate_ablation` (for the compressed/shuffled arms, which are NOT vacuous). Used by: nearly every C/P/S/I/A experiment, the census, dense-vs-pooled, compositional-binding.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: yes, the published held-out-combo BOUND is exactly the number a custom architecture must beat.

---

## Shell (the tiny trainable part)

### WorkspaceShell
- Status: PARTIAL. `shell/modulation.py` provides `WorkingMemory` (read/write slots, a minimal recurrent scratchpad) and `ContextGating` (per-context multiplicative gate); `shell/predictor.py` is the main trainable forward model; `shell/heads.py` the task heads. There is no single object called `WorkspaceShell` that composes predictor+heads+working-memory+gating into "the shell."
- Purpose: the assembled tiny trainable module (predictor, heads, ensemble, optional working memory, optional context gating) that everything trains, held to tens of millions of params. This is the "shell" of frozen-substrate-plus-tiny-trainable-shell.
- Inputs: cached latents. Outputs: predictions/logits plus optional working-memory state.
- Minimal impl: a `shell/workspace.py` composing the existing modules behind a config, so an experiment declares `shell.use_working_memory=true` instead of hand-wiring. This is pure composition of EXISTING classes; adds no new science.
- Full impl: the global-workspace-broadcast variant (bottlenecked shared latent that modules read/write) if and only if a non-ceiling test justifies it, which the corpus says does not yet exist.
- Dependencies: `predictor`, `heads`, `ensemble`, `modulation`. Used by: every trained-shell experiment.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: partly, but note a custom encoder would REPLACE the frozen substrate, not the shell; the shell composition is orthogonal.

### ReasoningLoop
- Status: EXISTS. `shell/refine.py` (`IterativeRefiner`: weight-tied residual refinement over N steps, optional ACT-lite adaptive halting) plus `diagnostics/convergence.py` (contraction-factor / basin-stability measurement) and `diagnostics/compute.py` (the matched-FLOP control). Experiments ex17_latent_reasoning, ex18_self_verification, y-series.
- Purpose: iterate computation in latent space and test whether it beats a compute-matched single-pass network, or is just unrolled depth. The corpus verdict so far: p9/ex17 gain 0.0, no fixed-point convergence, behaves like plain depth.
- Inputs: a latent `z`. Outputs: refined `z`, per-sample step count, contraction diagnostics.
- Minimal/full impl: present. Extension is only the halting/verifier arm if a real test bites.
- Dependencies: `predictor.mlp`, `convergence`, `compute`. Used by: EX17, EX18, Y1/Y2/N9.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: no, this is a shell-side algorithm, encoder-agnostic.
- Doctrine flag: any iteration gain MUST beat unrolled depth at equal FLOPs (matched-compute control), already wired.

### LatentScratchpad
- Status: PARTIAL. `shell/modulation.py::WorkingMemory` is the scratchpad primitive (gated read/write slots). No richer external-memory (differentiable-neural-computer-style addressed) scratchpad exists, and the corpus does not yet justify one.
- Purpose: a small recurrent set of read/write slots the shell can use as working memory across a latent sequence (event-segmentation / delayed-match tasks, n7_wm_delayed_match, n8_object_permanence_bound).
- Inputs: latent sequence. Outputs: slot states, read vector.
- Minimal impl: present (`WorkingMemory`). Full impl: content-addressed read/write only if a delayed-match test shows the mean-pool read is the bottleneck; unjustified today.
- Dependencies: torch. Used by: n7, n8, chunking in `modulation`.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: no.

---

## Memory and continual-learning layer

### ReplayMemory
- Status: EXISTS. `shell/buffer.py` (episodic replay buffer over frozen latents: prioritization by recency/surprise/reward/learning-progress, faiss KV index with exact brute-force fallback, reservoir/fifo/priority eviction, standard PER weights). Experiments e2_replay, ex1_generative_replay, ex13_long_stream, ex14_memory_bakeoff, n1_replay_ordering.
- Purpose: the latent hippocampus. Because the encoder is frozen, stored latents never go stale, the one place the frozen constraint actively helps.
- Inputs: latents + priorities. Outputs: sampled batches with importance weights, KV retrieval.
- Minimal/full impl: present and mature (the faiss-segfault workaround on Apple Silicon is already handled). No duplicate needed.
- Dependencies: torch, optional faiss. Used by: e2, ex1, ex13, ex14, n1. Laptop-safe: yes (brute-force fallback). Studio-scale: yes (faiss when safe). Prepares-for-custom-model: no, encoder-agnostic.

### FastWeightMemory
- Status: EXISTS. ex4_fast_weights implements the hypernetwork/fast-weight shell (h(context)->tiny head weights in one forward pass) with the collapse diagnostic and the gradient-TTA control. The mechanism is experiment-local, not yet a reusable `shell/` module.
- Purpose: in-context, zero-gradient adaptation, tested against gradient-TTA and a static head, with a collapse check (does h ignore its context).
- Minimal impl: leave in ex4. Full impl (extension): if fast weights ever beat gradient-TTA, promote to `shell/fast_weights.py`. Corpus status does not yet justify promotion.
- Dependencies: torch. Used by: ex4, ex7 (as a control arm). Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: no.

---

## Plasticity, neuromodulation, developmental layer

### PlasticityController
- Status: EXISTS. `shell/plasticity.py` (per-module LR gate: hard critical-period, soft sensitive-period, learned metaplasticity gate, signal-triggered reopening, PNN-analog rigidity). Experiments e3_plasticity, n5_fisher_reopen, ex15_rejuvenation, d6_sensitive_window.
- Purpose: gate learning rate per module (never the encoder), with the interesting variant being SIGNAL-triggered reopening (surprise/novelty above threshold reopens plasticity).
- Minimal/full impl: present. Corpus verdict: every biological-plasticity signature ties or loses its non-biological baseline; e3/n5/d6 are negatives. ex15_rejuvenation is the one scale-dependent lead (plasticity loss appears past toy scale, dim 256, thousands of tasks). No new code; the open work is running it at scale, not building a new controller.
- Dependencies: torch. Used by: e3, n5, ex15, d6. Laptop-safe: yes. Studio-scale: yes (this is where the scale lead lives). Prepares-for-custom-model: no.

### NeuromodulationGate
- Status: EXISTS. `shell/neuromod.py` (DA=surprise, ACh=expected uncertainty, NE=unexpected uncertainty/disagreement, each z-normalized by a running EMA and mapped to a multiplicative gain). Experiment e4_neuromod, n6_ach_ne_dissociation.
- Purpose: scalar self-derived signals that gate learning rate / memory write / reset. Bridge to plasticity: high surprise/disagreement reopens the critical period.
- Minimal/full impl: present. Corpus verdict: e4 is one of the STRONGEST confirmed negatives (30/30 runs amplify error on noise, wrong direction, fails the noisy-TV guard). No new code; the negative is the result.
- Dependencies: torch, `noisy_tv` (guard), `ensemble` (disagreement source). Used by: e4, n6. Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: no.

### CriticalPeriodScheduler
- Status: PARTIAL. The critical-period LOGIC is inside `PlasticityController` (hard schedule, close_at, reopen_threshold) and d6_sensitive_window. There is no standalone scheduler object; the schedule is a knob on the controller.
- Purpose: schedule opening/closing of plasticity windows over a developmental stream and test for a SUBSTRATE-SPECIFIC sensitive window (d6 result: substrate_specific_window=False).
- Minimal impl: keep as controller knobs. Full impl (extension): if a real natural-video stream ever shows a genuine window, factor the schedule into `shell/critical_period.py`. Corpus does not justify it yet (d6 flat).
- Dependencies: `plasticity`. Used by: d6, e3. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: no.

### ConsolidationEngine
- Status: EXISTS. `shell/consolidation.py` (EWC diagonal-Fisher and SI path-integral, composable, per-parameter quadratic penalty toward post-task weights). Also `diagnostics/fisher_trace.py`. Experiments e1_baseline, n4_tag_and_capture, and the continual-learning arms.
- Purpose: the weight-space dual of replay, protect important weights across tasks.
- Minimal/full impl: present. n4 tag-and-capture ties its baseline (negative). Used by: e1 gate, continual metrics. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: no.

---

## Curiosity and uncertainty layer

### CuriositySelector
- Status: EXISTS. e5_curiosity (data-selection: random vs prediction-error vs learning-progress, with noise-attraction tracking), ex8_curiosity_bakeoff, a8_affordance_curiosity, plus the mandatory `diagnostics/noisy_tv.py` guard.
- Purpose: select what to train on next; the corpus contract is that ONLY learning-progress resists the noisy-TV, prediction-error curiosity chases it.
- Minimal/full impl: present. Selection policy is experiment-local; a reusable `shell/curiosity.py` could host the three policies but is not yet justified as a shared object (each experiment configures its own). Leave as-is unless a second experiment needs the exact selector.
- Dependencies: `ensemble`, `noisy_tv`, `predictor`. Used by: e5, ex8, a8. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: no.
- Doctrine flag: noisy-TV guard is mandatory and already wired; a curiosity signal that chases irreducible noise is refuted by construction.

### UncertaintyEstimator
- Status: EXISTS. `shell/ensemble.py` (mean + disagreement = epistemic uncertainty) and `shell/heads.py::GaussianHead` (mean+logvar = aleatoric), plus `diagnostics/calibration.py`.
- Purpose: separate epistemic (ensemble spread, reducible) from aleatoric (logvar, irreducible) uncertainty, the prerequisite for a noisy-TV-passing curiosity signal.
- Minimal/full impl: present. No duplicate. Dependencies: torch. Used by: e4, e5, noisy_tv, calibration. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: no.

---

## Cross-substrate and mixture layer

### CrossSubstrateAgreement
- Status: NEW (as a first-class object). The primitives exist (`AlignmentSuite`/`geometry` CKA/RSA, `SubstrateAdapter` for multiple substrates) but there is no module that operationalizes standing-control 8 (universal vs modality/objective/architecture-specific).
- Purpose: given the SAME content encoded by multiple substrates (V-JEPA ViT-L, a different-objective encoder, random-init ViT), measure whether a decoded factor / a geometry converges across substrates (universal structure) or is specific to one training objective. This is the newest doctrinal control and has no home yet.
- Inputs: a dict `substrate_tag -> [N,D]` (from `SubstrateAdapter`), shared labels. Outputs: per-substrate probe accuracy, cross-substrate CKA matrix, a convergence verdict with a permutation p-value.
- Minimal impl: `diagnostics/cross_substrate.py` calling `AlignmentSuite` + `ProbeSuite` across the adapter set. Pure aggregation of existing calls.
- Full impl: add a null model (shuffled substrate labels) and a significance test so "converges" is certified. Register the substrate set in `SubstrateRegistry` via the new `family`/`objective` fields.
- Justification it is not a duplicate: no existing module iterates OVER substrates; every current diagnostic takes one representation. This is the missing outer loop.
- Dependencies: `SubstrateAdapter`, `AlignmentSuite`, `ProbeSuite`, `SubstrateRegistry`. Used by: a future MT/CM cross-substrate experiment (not proposed here per instructions). Laptop-safe: aggregation is cheap, but it requires MULTIPLE encoded caches; only safe if caches are precomputed (never run two encoders at once given the hard constraint). Studio-scale: yes, this is a natural Studio job. Prepares-for-custom-model: strongly, a custom encoder is just another substrate row, and its convergence with V-JEPA is the headline test.

### MixtureArbitrator
- Status: PARTIAL. The mixture-of-experts / router mechanism exists inside e7_sparse (tiny MoE with softmax router, k-WTA head) and ex9_slot_attention; there is no standalone `MixtureArbitrator` that arbitrates among heads/experts/substrates as a reusable object. Given the section title (mixture of perspectives) this is the conceptual center, but the code reality is that the only SURVIVING mixture positive (e7_sparse) is a head-architecture fact, reframed as NOT substrate-carrying-special-structure.
- Purpose: route/weight among a set of specialists (experts, heads, or substrates) and arbitrate their outputs. Honest framing: the corpus evidence for mixture is e7_sparse halving forgetting vs param-matched dense (a trained-shell metric that survives the vacuous control), which is architectural, not a claim about V-JEPA geometry.
- Inputs: latents + a set of specialist modules. Outputs: routed/weighted prediction + router entropy diagnostic.
- Minimal impl: promote e7's MoE router into `shell/mixture.py` ONLY when a second experiment reuses it; today it is a single-experiment mechanism and promoting it early would be a speculative duplicate.
- Full impl: a substrate-level arbitrator (route among V-JEPA vs a different-objective encoder per input) is the genuinely new idea the section title points at, but it is PREMATURE: no test yet shows two substrates are complementary rather than one dominating. Gate it behind CrossSubstrateAgreement showing complementary per-factor strengths.
- Dependencies: `WorkspaceShell`, `heads`, (substrate-level) `SubstrateAdapter`. Used by: e7 (head-level, present). Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: the substrate-level arbitrator is a design that could keep the frozen V-JEPA AND add a custom specialist, arbitrated per input, the least-committal answer to the reopened fork.
- Doctrine flag: any mixture win must beat a param-matched dense baseline (e7's control) and, for a substrate-level mixture, must beat the single best substrate alone (not just the average), or the arbitration bought nothing.

### ProcessCDenseTokenModule
- Status: EXISTS, GATED. `process_c/dense_tokens.py` defines `DenseTokenSlotModule`, `ProcessCDenseTokenClassifier`, `DenseTokenMeanBaseline`, matched-baseline width selection, binding-specificity reporting, and `process_c_budget_report`.
- Purpose: the sanctioned Process C pilot without overreach: a 1 to 10M object-centric trainable shell over frozen dense tokens, only run after PR9 or DR1 licenses it. This keeps Process C inside the audit's rule: remold a small module on dense tokens first, do not sneak into from-scratch V-JEPA-scale training.
- Inputs: dense frozen tokens `[B,N,D]`, optional token mask, and a task head. Outputs: slot vectors, pooled slot representation, attention over dense tokens, normalized assignment entropy, logits, and budget/license problems.
- Minimal impl: present. The slot module cross-attends learnable slots over dense tokens, updates slots with a GRU cell, and exposes attention for collapse and binding checks. The dense-token baseline mean-pools tokens and matches capacity with `dense_hidden_for_target_params`.
- Full impl (next): have CM9 or the licensed Process C launcher train this module on DR1 dense caches,
  compare to dense-without-slots and an off-the-shelf slot model, then promote
  `proof/NULL_CARDS/process_c_dense_token_pilot.md` from preregistration to a scored null/survival card.
  The laptop pass only proves tensor mechanics, budget gates, controls, and the launch-license receipt.
- Dependencies: torch. Used by: CM9 object-centric binding, Process C moldability pilot, dense-token facet-8 follow-up.
- Laptop-safe: yes for unit tests and tiny tensors. Studio-scale: yes when licensed. Prepares-for-custom-model: yes, this is the first sanctioned trainable dense-token arm.
- Doctrine flag: `process_c_budget_report` refuses unlicensed runs and enforces the 1 to 10M cap by default. A slot win must beat dense-without-slots at matched capacity and pass binding-specificity checks; a tie is a null.

### PR9RunStateReceipt
- Status: EXISTS. `scripts/studio/pr9_continual_backprop.py` writes a `mop-pr9-run-state/v1` JSON receipt beside the PR9 result.
- Purpose: make long-stream plasticity evidence resumable and durable. The PR9 script already checkpoints each `(seed, arm, lr, rate)` leg under `<out>.legs/`; the state receipt now records the cache, seed grid, CBP rate grid, expected leg count, completed legs, output path, final verdict when available, and the exact resume behavior.
- Inputs: PR9 output path, optional `--state-out`, generated leg directory, and the long-stream config. Outputs: `<out>.state.json` by default, plus the final PR9 result JSON.
- Minimal impl: present. The state is written after guards pass, after baseline LR tuning, after each plain/CBP leg, at verdict-ready time, and after the final result file is written.
- Full impl: present. The state receipt is included in the post-PR9 verdict ledger, artifact bundle, and
  scorecard path before any moldability score moves.
- Dependencies: PR9 leg cache, `StudioArtifactBundle` `pr9` preset. Used by: PR9 long-stream plasticity evidence and Process C licensing.
- Laptop-safe: yes for smoke or unit-level receipt checks. Studio-scale: yes. Prepares-for-custom-model: directly, because Process C is only allowed after PR9/DR1 evidence and this receipt proves whether PR9 completed or where it was interrupted.
- Doctrine flag: an interrupted PR9 is not a hidden failure or a positive; it is a resumable run state until the verdict gate and artifact bundle close it.

### PR9VerdictLedger
- Status: EXISTS. `studio/pr9_verdict.py` and `scripts/studio/pr9_verdict_ledger.py` synthesize
  `mop-pr9-verdict-ledger/v1` from the PR9 result plus `mop-pr9-run-state/v1`.
- Purpose: give PR9 the explicit verdict ledger required by the Studio prompt. The ledger rejects local
  smoke caches as non-scoring, requires the dedicated PR9 null card, classifies config errors,
  compute-mismatch nulls, no-certificate nulls, CBP-no-win nulls, and candidate positives, and records
  whether Process C is licensed by the PR9 wall.
- Inputs: `runs/mot/pr9_continual_backprop.json`, `runs/mot/pr9_continual_backprop.json.state.json`,
  `proof/NULL_CARDS/pr9_long_stream_plasticity.md`, and expected DR1 cache path. Outputs:
  `runs/mot/pr9_verdict_ledger.json`.
- Minimal/full impl: present. A candidate positive remains `candidate-positive-needs-verdict-gate`; the
  ledger does not publish it by itself. Nulls and walls are still durable, scoring receipts.
- Dependencies: PR9 raw result, PR9 run-state receipt, PR9 null card, ArtifactBundle. Used by:
  StudioSpinePlan, StudioScorecard, and the `pr9` artifact preset.
- Laptop-safe: yes, pure JSON/Markdown. Studio-scale: yes. Prepares-for-custom-model: directly, because
  Process C can only proceed after PR9 or DR1 licenses it with a receipt.

### ProcessCLicenseGate
- Status: EXISTS. `studio/process_c_gate.py` and `scripts/studio/process_c_license_gate.py` synthesize
  `mop-process-c-license-gate/v1`.
- Purpose: turn the "Process C only if licensed" doctrine into an artifact decision instead of a prose
  memory. The gate reads the PR9 verdict ledger and DR1 adversarial verifier, checks the dedicated
  Process C null card, records whether either source licenses the sanctioned 1 to 10M dense-token pilot,
  and leaves an evidence-supported "not licensed" state as a completed wall.
- Inputs: `runs/mot/pr9_verdict_ledger.json`,
  `data/cache/vjepa2_vitl_comp_video/dr1_verification.json`, and
  `proof/NULL_CARDS/process_c_dense_token_pilot.md`. Outputs:
  `runs/mot/process_c_license_gate.json`.
- Minimal/full impl: present. PR9 licenses only through `process_c_licensed: true`; DR1 licenses only
  when artifact integrity is clean, the verifier is independent/adversarial, and the decisive A6 survival
  flag is explicitly false. Missing or invalid upstream receipts make the gate undecidable, not licensed.
- Dependencies: PR9VerdictLedger, DR1AdversarialVerifier, NullCardGenerator, StudioArtifactBundle.
  Used by: StudioSpinePlan, StudioScorecard, StudioNativeLanes, and the `pr9` artifact preset.
- Laptop-safe: yes, pure JSON/Markdown. Studio-scale: yes. Prepares-for-custom-model: directly, by
  making the Process C launch bit receipt-backed.
- Doctrine flag: `launch_allowed: true` is the only Process C authorization bit. A clean `not_licensed`
  receipt is a wall, not a failure, and no Process C training launcher is emitted by this gate.

---

## Compression and information layer

### CompressionDoctor
- Status: EXISTS. `diagnostics/bottleneck.py` (capability-per-bit, real vs frozen-random with a bit knob), `diagnostics/buffer_compression.py`, plus the I-series experiments (i1_info_bottleneck, i2_mdl_selection, i3_compression_reasoning, i5_rate_distortion_replay, i6_mi_audit, i7_predictive_information, i9_vq_rate_distortion) and `substrate_ablation.quantize_dequantize`.
- Purpose: measure information-theoretic properties, does the real encoder pack a factor into FEWER bits than any projection (a sharp knee), or is graceful degradation a generic high-dim property (taxonomy 3).
- Minimal/full impl: present and extensive. No standalone "doctor" class needed; the functions are the doctor. Could add a thin `diagnostics/compression_report.py` aggregator mirroring how `AlignmentSuite` aggregates geometry, but this is convenience not capability.
- Dependencies: `linear_probe`, `substrate_ablation`. Used by: I-series. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: the capability-per-bit curve is a clean substrate comparison, so yes.
- Doctrine flag: the bit-knob real-vs-frozen-random comparison is subject to the vacuous-control caveat for the LINEAR probe portion; the knee must be shown against a random-ENCODER arm to be a substrate claim.

---

## Experiment governance layer

### ExperimentRegistry
- Status: EXISTS. `registry/experiments.yaml` (the machine-readable preregistration: null, metric, falsifier, taxonomy slot per row) + `src/mop/experiments/__init__.py::REGISTRY`/`get_experiment` + `experiments/base.py::Experiment` (the contract enforced at class-definition time: missing baseline/ablation/metric/null raises `TypeError`) + `devel/registries.py::validate_experiment`.
- Purpose: the preregistration and the in-code contract. An experiment cannot instantiate without a metric, baseline, ablation, and null; the yaml commits the falsifier and taxonomy slot before a run.
- Minimal/full impl: present and enforced. The new label families MT/PR/WS/CM/AT/AL/DR are free (do not collide with the 119 catalogued ids). Extension is only adding rows for future sections, not code.
- Dependencies: `omegaconf`, `base.Experiment`. Used by: every experiment, the queue, `negative_registry`. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: yes, a custom-encoder experiment is just a new row with a null and a baseline it must beat.

### NullHypothesisRegistry
- Status: EXISTS (folded into ExperimentRegistry). The null lives as `Experiment.null_hypothesis` (enforced) and as the `null` field per row in `registry/experiments.yaml`; `studies/negative_registry.py` reads the declared null and the predicted-null booleans out of each result.
- Purpose: guarantee every mechanism has a stated null and that "it did not work" lands in a named slot.
- Minimal/full impl: present. Not a separate module; the null is a required field of every experiment. No duplicate should be built. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: yes.

### NegativeResultTaxonomy
- Status: EXISTS. `studies/negative_registry.py` (runs the cheapest predicted-null ablation per experiment, records verdict confirmed/refuted/mixed and a taxonomy slot 1..10) + `proof/FAILURE_TAXONOMY.md` + `registry/experiments.yaml` `taxonomy_slot`.
- Purpose: the corpus is a rigorously honest NEGATIVE map; this is the machinery that files each refuted lever into one of ten categories rather than losing it.
- Minimal/full impl: present. No duplicate. Dependencies: `experiments`, `harness.runner`. Used by: the whole negative corpus (e4, n4, n5, n6, d4, e3, b4). Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: yes.

### VerdictGate
- Status: EXISTS. `falsification/verdict_gate.py` and `scripts/verdict_gate.py` gate a final verdict receipt before it can enter the Studio ledger.
- Purpose: make the rule "no positive without independent adversarial verification" executable. A candidate positive now needs a strict null card, a JSON raw-run receipt, and a separate verifier receipt that marks both passed and independent/adversarial. Nulls and ties still need the strict card plus raw receipt, but do not need a verifier to be honest.
- Inputs: null-card Markdown, raw-run JSON receipt, optional verifier JSON receipt, optional declared verdict override. Outputs: `mop-verdict-gate/v1` JSON with card/run/verifier hashes, pass/fail flags, and blocker reasons.
- Minimal impl: present. Positive means `PUBLISH-POSITIVE` by default. The verifier path may not equal the raw-run path, and ambiguous verifier receipts fail closed unless they expose a pass flag and an independence/adversarial flag.
- Full impl: `src/mop/studio/claim_plan.py` and `python -m scripts.studio claim-plan` now generate daemon plans that run `scripts/verdict_gate.py`, then `python -m scripts.studio artifact-bundle`, then the requested ledger command. Positive plans mark the final job as `positive-ledger`, so the daemon's static contract rejects any hand-built plan that omits the two gates.
- Dependencies: `NullCardGenerator`, JSON receipts, `ArtifactBundle`, `LongRunDaemon`. Used by: Studio ledger updates and long-run daemon plans.
- Laptop-safe: yes, pure receipt validation. Studio-scale: yes. Prepares-for-custom-model: yes, because Process C positives cannot enter docs without an independent verifier receipt.
- Doctrine flag: this is a ledger gate, not a science metric. A failed gate blocks a doc update; it does not turn a run into a null.

---

## Observability and reproducibility layer

### StudioCommandSurface
- Status: EXISTS. `scripts/studio/__main__.py` is the canonical wrapper-script surface: `python -m scripts.studio <command>`.
- Purpose: collapse top-level Studio wrapper sprawl into one command grammar while preserving historical receipt paths.
- Minimal/full impl: present. The old `scripts/studio_*.py` entrypoints remain as thin compatibility shims only; newly generated daemon and spine plans use `python -m scripts.studio ...`.
- Commands: `artifact-bundle`, `claim-plan`, `daemon`, `density-receipt`, `disk-recovery`, `doctor`, `native-lanes`, `objective-audit`, `rehearse`, `scorecard`, `spine-plan`, `transfer-check`, and `wave0-report`.
- Doctrine flag: the compatibility shims are intentionally not deleted because existing run receipts and operator notes name those paths.

### ArtifactBundle
- Status: EXISTS. `studio/artifact_bundle.py` and `python -m scripts.studio artifact-bundle` write durable artifact indexes and optional small-receipt bundles.
- Purpose: close the audit's result-durability hole. A Studio wave should not leave load-bearing JSON/Markdown only under ignored `runs/`; the bundle indexes receipt paths, hashes, sizes, JSON validity, git tracking, copy status, and whether each artifact is durable.
- Inputs: explicit paths or presets (`pre-studio`, `wave0`, `dr1`, `pr9`, `spine`), optional copy directory, copy-size cap, missing-artifact policy, and durability requirement. Outputs: `mop-artifact-bundle/v1` JSON plus optional copied small-text receipts.
- Minimal impl: present. Text receipts (`.json`, `.md`, `.txt`, `.yaml`, `.yml`, `.csv`, `.tsv`) can be copied into a durable directory; oversized or non-text artifacts are refused with a blocker reason. Large caches stay out of the bundle and must be represented by cache manifests.
- Full impl (next): after the M1 Ultra completes Wave 0, regenerate the `wave0` preset index with `--copy-dir proof/ARTIFACT_BUNDLES/wave0` and commit the index or otherwise transfer it with the Studio report.
- Dependencies: git CLI, `StudioTransferCheck` durable receipt list. Used by: pre-Studio transfer, Wave-0 artifact preservation, long-run daemon handoff bundles.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: indirectly, because Process C and dense-cache verdict receipts can be preserved without copying terabyte arrays.
- Doctrine flag: a missing artifact or failed copy is a durability blocker, not a scientific null.

### StudioTransferCheck
- Status: EXISTS. `studio/transfer_check.py` and `python -m scripts.studio transfer-check` implement the read-only Wave-0 transfer checklist.
- Purpose: prove the Studio received the governing stack and durable receipts before it spends compute. The check validates the active profile is `studio-m1ultra`, confirms the governing audit and required docs/scripts exist, including the disk-recovery CLI, parses the null-card schema, reports git branch/head/dirty state, confirms pre-Studio receipt files are present and git-tracked, and validates any cache manifests already present.
- Inputs: repo root, profile name, optional audit path, dirty-worktree policy, and receipt requirement flag. Outputs: `mop-studio-transfer-check/v1` JSON with per-check status and summary.
- Minimal impl: present. The CLI is read-only and writes a report with `--out`; dirty worktrees fail by default unless `--allow-dirty` is explicit.
- Full impl (next): after the M1 Ultra emits Wave-0 receipts, run `python -m scripts.studio artifact-bundle --preset wave0` and archive the emitted JSON beside `STUDIO_RUN_REPORT.md` on the Studio.
- Dependencies: `studio.profiles`, `substrate.cache_manifest`, git CLI. Used by: Studio Wave 0 transfer receipt and long-run daemon template.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: indirectly, by proving the receipt chain and profile envelope before Process C or dense-cache work can start.
- Doctrine flag: transfer check is not science; failing it blocks the wave instead of downgrading any scientific null.

### StudioDiskRecovery
- Status: EXISTS. `studio/disk_recovery.py` and `python -m scripts.studio disk-recovery` emit a Wave-0 disk recovery receipt.
- Purpose: make launch cleanup auditable without risking evidence loss. The planner scans only known generated/tool-cache paths by default, classifies candidates, refuses tracked files, and blocks ignored run deletion when unbundled receipt-like text artifacts are present.
- Inputs: profile name, optional scan paths, defaults toggle, execute flag, explicit allowed classes or paths, and max receipt examples. Outputs: `mop-disk-recovery-plan/v1` JSON with free-disk status, safe/protected candidates, would-delete bytes, and performed deletions when execute is explicitly allowed.
- Minimal impl: present. Dry-run is default; execute requires at least one `--allow-class` or `--allow-path`.
- Full impl (next): before large Studio waves, run the receipt, bundle any protected receipts if cleanup is needed, then rerun with explicit allow rules only for safe generated classes.
- Dependencies: `studio.profiles`, git CLI, `StudioArtifactBundle` text-extension policy. Used by: Studio Wave 0 daemon template and Wave 0 report synthesis.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: indirectly, by protecting dense-cache and PR9 receipts from cleanup loss.
- Doctrine flag: deleting an unbundled receipt is a launch failure, not a cleanup success.

### StudioDensityReceipt
- Status: EXISTS. `studio/density_receipt.py` and `python -m scripts.studio density-receipt` emit
  `mop-studio-density-receipt/v1`.
- Purpose: satisfy the shared Studio 10/10 density-receipt requirement without confusing cleanup with
  science. The receipt records workspace size, tracked text/code LOC, largest files, artifact-mass
  buckets (`runs`, `data/cache`, proof bundles/indexes/null cards), and before/after cleanup deltas from
  the disk-recovery receipt.
- Inputs: repo root, disk-recovery receipt, and largest-file limit. Outputs:
  `runs/studio_wave0/density_receipt.json`.
- Minimal/full impl: present. The module is read-only and never deletes files. Missing disk recovery is
  recorded as absent; an invalid disk-recovery schema makes the density receipt fail. The Wave-0 daemon
  writes it after disk recovery, transfer check requires the CLI, artifact-bundle presets preserve it,
  and the objective audit counts it only as durable-report launch prep.
- Dependencies: git CLI for tracked LOC. Used by: Studio Wave 0 daemon, transfer check, artifact bundle,
  and objective audit.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: indirectly, by exposing artifact mass
  before large DR1/PR9/dense-cache runs.
- Doctrine flag: density is operational hygiene. It cannot move abstraction, density/substrate, or
  moldability science scores.

### StudioWave0Report
- Status: EXISTS. `studio/wave0_report.py` and `python -m scripts.studio wave0-report` synthesize the Wave-0 launch, transfer, daemon, encode, and memory receipts into one JSON summary plus a bounded Markdown block in `STUDIO_RUN_REPORT.md`.
- Purpose: make the actual M1 Ultra hardware, disk, MPS, encoder, cache path, transfer status, s/clip, and memory envelope land in the scoreboard without a manual rewrite. The report block is bounded by markers and is replaced idempotently on rerun, so a resumed Wave 0 cannot duplicate stale rows.
- Inputs: transfer-check JSON, doctor JSON, disk-recovery JSON, daemon state JSON, `encode_device.json`, and `encode_schedule.json`. Outputs: `runs/studio_wave0/wave0_report.json` and, with `--apply`, an auto receipt block in `docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md`.
- Minimal impl: present. It records launch profile, hardware detail, profile disk floor, MPS availability, encoder config availability, cache write path, disk-recovery summary, transfer pass count, daemon job summary, CPU/MPS s/clip, winner, launch gate, blocked reasons, process RSS peak, minimum available system memory, and MPS driver peak.
- Full impl (next): after the Studio completes the >=1000-clip cache rebuild, extend the same report with cache-manifest validation and actual cache count.
- Dependencies: the JSON receipts produced by `transfer-check`, `daemon`, and `mop_encode_autoselect.py`. Used by: Studio Wave 0 ledgering.
- Laptop-safe: yes, pure JSON/Markdown. Studio-scale: yes. Prepares-for-custom-model: indirectly, by keeping the Studio scoreboard receipt-driven before DR1/PR9/Process C.
- Doctrine flag: a missing or blocked receipt renders Wave 0 incomplete. It never converts a failed gate into a scientific null.

### DR1AdversarialVerifier
- Status: EXISTS. `studio/dr1_verifier.py` and `scripts/studio/dr1_verify.py` read the completed DR1 sidecars and emit `mop-dr1-adversarial-verification/v1`.
- Purpose: make DR1 positives pass one independent receipt before they can feed downstream claims. The verifier reads the merge manifest, per-leg `cells.json` acceptance reports, clip hashes, PerspectiveMatrix receipt, and A6 residual guard receipt. It sets `independent: true` and `adversarial: true`; `passed/all_ok` are true only when integrity is clean and the decisive A6 condition survives.
- Inputs: DR1 cache directory, A6 requirement flag, PerspectiveMatrix requirement flag. Outputs: `data/cache/<dr1>/dr1_verification.json`.
- Minimal impl: present. Missing caption gates, frozen-random backends, non-contiguous legs, missing perspective receipts, and A6 collapse all refuse positive verification.
- Full impl (next): after the Studio DR1 run, feed this verifier receipt into `studio_claim_plan.py` for any DR1-dependent positive and bundle it with the `dr1` artifact preset.
- Dependencies: DR1 merge sidecars, `PerspectiveAdapter` receipt, A6 residual guard. Used by: DR1 real bound-attribute video, CM1/CM3 cache gates, and dense/atlas follow-up claims.
- Laptop-safe: yes, pure JSON checks. Studio-scale: yes. Prepares-for-custom-model: directly, because DR1 is the keep-frozen versus custom-training gate.
- Doctrine flag: artifact integrity can pass while the positive verifier fails. In that case the cache is preserved as a null or wall, not used as positive evidence.

### DenseAtlasCacheGate
- Status: EXISTS. `studio/dense_atlas_gate.py` and `scripts/studio/dense_atlas_gate.py` write
  `mop-dense-atlas-cache-gate/v1`.
- Purpose: stop the full atlas before launch unless the dense V-JEPA 2.1 real cache and its matched
  random-init dense control are both present, manifest-clean, dense-shaped, and referent-aligned.
- Inputs: real dense cache path, random-init dense cache path, min clip count, min dense-token count,
  and expected embedding dim. Outputs: `runs/mot/dense_atlas_cache_gate.json` with per-cache summaries,
  pair checks, problems, and `all_ok`.
- Minimal/full impl: present. The gate validates `cache_manifest.json`, cache integrity, dense token
  count, embedding dim, count match, referent-key fingerprint match, and factor/split sidecar match.
- Dependencies: `substrate.cache_tools`, `substrate.cache_manifest`. Used by: StudioSpinePlan,
  StudioScorecard, and the atlas artifact bundle.
- Laptop-safe: yes, read-only manifest/header checks. Studio-scale: yes. Prepares-for-custom-model:
  yes, because Process C and dense-token claims depend on the same dense-cache pair being honest.
- Doctrine flag: a single real dense cache is not enough. The matched random-init dense cache is part of
  the null, and missing or mismatched controls block density claims rather than becoming partial wins.

### AtlasVerdictLedger
- Status: EXISTS. `studio/atlas_verdict.py` and `scripts/studio/atlas_verdict_ledger.py` write
  `mop-atlas-verdict-ledger/v1`.
- Purpose: turn the raw atlas JSON into a publishability receipt before density score movement. The
  ledger checks the dedicated atlas null card, the paired dense-cache gate, full registered grid/pair
  status, and the atlas null verdict.
- Inputs: `runs/mot/dense_atlas_cache_gate.json`, `runs/mot/atlas_multi_encoder_grid.json`, and
  `proof/NULL_CARDS/atlas_dense_multiencoder.md`. Outputs: `runs/mot/atlas_verdict_ledger.json`.
- Minimal/full impl: present. Dense-gate missing/blocked, partial registered grid, missing atlas result,
  and indeterminate null state are non-scoring; null support is a publishable wall; null rejection is a
  candidate positive that still needs the generic verdict-gate and durable artifact path.
- Dependencies: DenseAtlasCacheGate, Atlas runner, NullCardGenerator. Used by: StudioSpinePlan,
  StudioScorecard, and the atlas artifact bundle.
- Laptop-safe: yes, pure JSON/Markdown. Studio-scale: yes. Prepares-for-custom-model: yes, because
  dense-token Process C claims should not inherit density movement from an ungated atlas.
- Doctrine flag: an atlas JSON alone never moves density. The ledger is the first scoring receipt.

### StudioSpinePlan
- Status: EXISTS. `studio/spine_plan.py` and `python -m scripts.studio spine-plan` emit `mop-studio-spine-plan/v1`.
- Purpose: turn the Studio order into one staged, resumable operator receipt instead of a prose list of commands. Wave 0 and DR1 remain subdaemons so their `daemon_state.json` receipts stay local to the wave.
- Inputs: DR1 source directory, profile, DR1 cache name, dense cache name, PR9 seeds, atlas seeds. Outputs: `runs/studio_spine/spine_plan.json` and `runs/studio_spine/wave0_daemon_plan.json`.
- Minimal impl: present. The staged plan orders Wave 0, Wave 0 bundling, DR1 source-card validation,
  DR1 intake/schedule/run/bundle, PR9 run/verdict/Process C license gate/bundle, dense-cache planning,
  paired dense real/random-init validation, full atlas without `--allow-partial`, atlas verdict ledger,
  atlas bundle, scorecard, spine status receipt, objective audit, and final spine artifact index.
- Full impl (next): after the actual Studio run, append any new Studio-native lanes only as receipt-bearing steps with explicit null or wall conditions.
- Dependencies: `LongRunDaemon`, `StudioArtifactBundle`, DR1 scheduler/verifier, PR9 run-state/verdict
  receipts, ProcessCLicenseGate, cache manifests, atlas runner. Used by: the M1 Ultra launch path.
- Laptop-safe: yes, plan-writing only. Studio-scale: yes. Prepares-for-custom-model: indirectly, by keeping Process C behind DR1/PR9 receipts and preserving dense-cache walls.
- Doctrine flag: the dense/atlas phase stops at missing, invalid, or mismatched dense real/control cache
  manifests and never converts a partial atlas into a scope claim.
  `python -m scripts.studio spine-plan --status` is read-only and reports the next command from receipts, so a
  resumed Studio session does not depend on memory or manual checklist reconstruction.

### StudioScorecard
- Status: EXISTS. `studio/scorecard.py` and `python -m scripts.studio scorecard` emit
  `mop-studio-scorecard/v1` and can update a bounded block in `STUDIO_RUN_REPORT.md`.
- Purpose: turn final Studio score movement into a receipt-backed synthesis instead of a manual table
  edit. The scorecard reads Wave 0, DR1 verification, PR9 result/state/verdict ledger, Process C license
  gate, dense cache gate, atlas result/verdict ledger, artifact indexes, and spine status.
- Inputs: Wave 0 report, DR1 verifier, PR9 result, PR9 run-state receipt, PR9 verdict ledger, Process C
  license gate, dense atlas cache gate, atlas result, atlas verdict ledger, artifact indexes, and spine
  status. Outputs: `runs/studio_scorecard.json` plus, with `--apply`,
  the report block.
- Minimal impl: present. Missing receipts are blockers, local PR9 smoke is non-scoring, DR1-cache PR9
  cannot score without the verdict ledger, Process C launch state is reported from its own gate,
  dense/atlas evidence cannot score without the paired cache gate and atlas verdict ledger, partial atlas
  runs are non-scoring, and proven walls stay in the scorecard instead of becoming positives. The CLI
  exits nonzero by default on incomplete receipts; the spine uses `--allow-incomplete` for its final
  preservation pass so an honest blocker does not prevent the status, objective-audit, or artifact-bundle
  receipts from being written.
- Full impl (next): after real Studio waves, extend axis details with the exact downstream verdict-gate
  receipts for any claim promoted into `HANDOFF.md` or `RESULTS_LEDGER.md`.
- Dependencies: Wave0Report, DR1AdversarialVerifier, PR9RunStateReceipt, PR9VerdictLedger,
  DenseAtlasCacheGate, AtlasVerdictLedger, ArtifactBundle, StudioSpinePlan. Used by: final Studio run
  reports and stop-checks.
- Laptop-safe: yes, pure JSON/Markdown. Studio-scale: yes. Prepares-for-custom-model: yes, because
  Process C licensing stays tied to PR9/DR1 receipt states.
- Doctrine flag: the scorecard can report a completed wall, but it cannot make a missing or partial run
  look like evidence.

### StudioObjectiveAudit
- Status: EXISTS. `studio/objective_audit.py` and `python -m scripts.studio objective-audit` emit
  `mop-studio-objective-audit/v1`.
- Purpose: re-evaluate the active Studio 10/10 prompt as requirement checklist points without
  confusing local launch prep with scientific evidence. It names which points are complete, prepared,
  pending, or blocked, and keeps the score kind explicit: checklist coverage, not an axis score.
- Inputs: Wave 0 transfer/report receipts, spine plan/status, DR1 verifier, PR9 verdict ledger, dense
  gate, atlas verdict ledger, Process C license gate, scorecard, native-lane manifest, and artifact
  indexes. Outputs: `runs/studio_objective_audit_local.json` on the laptop or
  `runs/studio_objective_audit.json` on the Studio.
- Minimal/full impl: present. The current local audit earns only launch-prep/checklist credit and leaves
  DR1, PR9, dense/atlas, Process C decisiveness, and final artifact indexes incomplete until real Studio
  receipts exist. The CLI exits nonzero by default when `studio_10_ready` is false, but the spine uses
  `--allow-not-ready` so a not-ready or walled audit receipt is still bundled.
- Dependencies: StudioScorecard, StudioSpinePlan, NativeLanes, ArtifactBundle, DR1/PR9/atlas/Process C
  verdict receipts. Used by: run-report reevaluations and final readiness decisions.
- Laptop-safe: yes, pure JSON. Studio-scale: yes. Prepares-for-custom-model: indirectly, by making the
  Process C and dense-token prerequisite gaps explicit before launch.
- Doctrine flag: a prepared point is not a scientific point. The audit is allowed to say "not ready" even
  when every local test is green. Not-ready audits must still be preserved as artifacts.

### StudioNativeLanes
- Status: EXISTS. `studio/native_lanes.py` and `python -m scripts.studio native-lanes` make the Studio-native facets machine-readable: DR13 rollout fidelity, hosted corpora, live-encoder doctrine, perspective ecology, developmental PR9, and Process C licensing.
- Purpose: turn the Part 2 audit lanes into concrete daemon jobs when their preregistered inputs exist, and into explicit blocked receipts when a prior artifact is missing. This keeps "not runnable yet" falsifiable instead of leaving it as prose.
- Inputs: profile name plus optional `clip_dir`, `dr1_cache`, `plan_path`, `encode_schedule`,
  `pr9_verdict`, and `dr1_verification` paths. Outputs: a `mop-studio-native-lanes/v1` manifest and,
  for ready lanes, a standard long-run daemon plan.
- Minimal impl: present. Heavy lanes are blocked unless `--include-heavy` is set, large acquisition
  requires an explicit inspected `--plan-path`, Process C materializes only the license-gate command
  when PR9/DR1 receipt paths are supplied, and lanes without a sanctioned launcher record the release
  condition instead of fabricating a command.
- Full impl (next): wire perspective feature extraction into the manifest once DR1 has a merged cache and matched language/audio/object-centric feature stores.
- Dependencies: `LongRunDaemon`, `PerspectiveAdapter`, `ProcessCDenseTokenModule`, DR1 cache receipts, Wave-0 encode schedule. Used by: Studio-native lane planning after Wave 0.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: directly, by keeping Process C behind an explicit PR9/DR1 licensing receipt.
- Doctrine flag: blocked lanes are walls-in-progress, not nulls. A lane can only move from blocked to ready when its named receipt exists.

### LongRunDaemon
- Status: EXISTS. `studio/long_run.py` and `python -m scripts.studio daemon` supervise a JSON job plan under an active Studio profile.
- Purpose: make week-scale Studio work boring and resumable: profile disk gate before each job, dry-run by default, state checkpoint after every transition, heartbeat events during long subprocesses, per-job stdout/stderr logs, resume-skip for completed jobs, and clean stop on blocked/failed jobs.
- Inputs: a daemon plan with `schema: mop-long-run-daemon/v1` and `jobs: [{id, cmd, cwd?, kind?}]`. Outputs: `daemon_state.json`, `logs/<job>.stdout.log`, `logs/<job>.stderr.log`, event history, and a summary by status.
- Minimal impl: present. The daemon is command-level infrastructure, not a science launcher. It does
  not choose DR1/PR9/Process C order; it enforces the profile and records execution for the chosen plan.
  The Wave-0 template now emits the transfer check, disk recovery, density receipt, doctor,
  docs/acceptance gates, DR1 smoke, encode microbench, Studio-native lane manifest, and Wave-0 report in
  one resumable plan.
- Full impl: the daemon now validates a static pre-ledger contract: any `positive-ledger` job must be ordered after both a `verdict-gate` job and an `artifact-bundle` job. Because execution stops on prior failure, this makes the falsification and durability gates mandatory before a positive doc mutation can run. `python -m scripts.studio daemon validate --plan <plan.json>` checks the contract without running the plan.
- Dependencies: `studio.profiles`, subprocess. Used by: Studio Wave 0 and week-scale gated queues.
- Laptop-safe: yes, dry-run by default and tested with injected runners. Studio-scale: yes. Prepares-for-custom-model: yes, Process C jobs can be supervised without loosening the disk/profile gates.
- Doctrine flag: a daemon failure or disk stop is a wall/blocked artifact, not a half-positive. A dry-run rehearsal is not treated as completed during a later execute run.

### MetricsLogger
- Status: EXISTS. `logging_utils.py` (`RunManifest`, `new_run_dir`, JSON-to-stdout / logs-to-stderr discipline) + `metrics/continual.py` (`ContinualResult`, BWT/forgetting) + `metrics/frontier.py` + `harness/runner.py` (writes config snapshot + manifest + metrics per run).
- Purpose: every run gets a directory with resolved config, manifest (seed, device, git, timing), and metrics; no silent failures.
- Minimal/full impl: present. Dependencies: `provenance`, `config`. Used by: every run. Laptop-safe/Studio-scale: yes. Prepares-for-custom-model: yes, metrics are encoder-agnostic.

### ReproducibilityHarness
- Status: EXISTS. `seeding.py` (`seed_everything`, `variance_of`, `VarianceReport`) + `diagnostics/determinism.py` (Metal/CPU spread, tolerance sizing) + `diagnostics/seed_consistency.py` (cross-seed CKA / code agreement, sign-flips published as instability) + `harness/sweep.py` (seed sweeps via `cfg.seed` overrides, the fix for the e4/e7 silent-no-op) + `provenance.py` (git SHA, package versions, encoder backend, result tag).
- Purpose: seed everything we can, then MEASURE run-to-run variance rather than assume bit-exactness (Apple Metal is ~50% byte-identical at temp 0); publish sign-flips as instability; stamp provenance so any number traces to exact code+config+substrate.
- Minimal/full impl: present and mature. The one live gotcha, already documented: e4/e7 seeds MUST sweep via `harness.sweep.run_sweep` because those modules read `cfg.seed`, not `experiment.seeds`.
- Dependencies: torch, `provenance`. Used by: every seeded result, the seed-variance study. Laptop-safe: yes (CPU is bit-identical). Studio-scale: yes. Prepares-for-custom-model: yes.

---

## Summary: EXISTS vs NEW

- EXISTS (extend only): SubstrateRegistry, LatentStore, EncodeScheduler, NullCardGenerator, VerdictGate, StudioClaimPlan, SubstrateAdapter, PerspectiveAdapter, AlignmentSuite, ProbeSuite, ReplayMemory, PlasticityController, NeuromodulationGate, ConsolidationEngine, CuriositySelector, UncertaintyEstimator, ReasoningLoop, CompressionDoctor, ExperimentRegistry, NullHypothesisRegistry, NegativeResultTaxonomy, ArtifactBundle, StudioTransferCheck, StudioDiskRecovery, StudioWave0Report, DR1SourceCard, DR1SourceIntake, DR1AdversarialVerifier, DenseAtlasCacheGate, AtlasVerdictLedger, StudioSpinePlan, StudioScorecard, StudioNativeLanes, LongRunDaemon, MetricsLogger, ReproducibilityHarness, ProcessCDenseTokenModule, PR9RunStateReceipt, PR9VerdictLedger.
- PARTIAL (primitives exist, needs a thin aggregator or promotion): WorkspaceShell (compose existing shell modules), LatentScratchpad (WorkingMemory exists), FastWeightMemory (ex4-local), CriticalPeriodScheduler (controller knobs), MixtureArbitrator (e7 MoE router, promote only when reused).
- NEW (justified, no duplicate): CrossSubstrateAgreement (`diagnostics/cross_substrate.py`), the missing outer loop over substrates for standing-control 8; and the substrate-LEVEL variant of MixtureArbitrator, which is the least-committal architectural answer to the reopened dense-vs-custom fork but is gated behind CrossSubstrateAgreement showing complementarity.

The load-bearing architectural consequence: the corrected-substrate research now has reusable controls, receipts, and gated dense-token machinery. The remaining axis-moving gap is still DENSE real-video bound-attribute caches plus Studio execution, not laptop science. Building parallel WorkspaceShell/ReplayMemory/PlasticityController classes would be a duplicate and is explicitly rejected.
