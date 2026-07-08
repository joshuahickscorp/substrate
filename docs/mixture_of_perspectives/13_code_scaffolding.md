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
- Status: EXISTS. `studio/encode_scheduler.py` consumes the Wave-0 CPU/MPS benchmark, the active profile, an encoder config, a requested clip count, and a dense/pooled flag, then emits an encode launch plan. `scripts/mop_encode_autoselect.py` now writes both `runs/mot/encode_device.json` and `runs/mot/encode_schedule.json`; `scripts/studio/dr1_schedule_plan.py` turns that schedule into DR1 gate, leg, merge, and A6-guard commands.
- Purpose: make the Studio encode path profile-owned instead of hand-owned. The scheduler picks MPS vs parallel CPU workers from measured s/clip, estimates cache footprint, enforces the profile's start and post-cache disk floors, checks wall clock against the profile cap, and emits checkpoint cadence.
- Inputs: benchmark record (`cpu_s_per_clip`, `mps`), optional `memory_envelope`, `profile_name`, encoder config, requested clips, dense-token flag. Outputs: winner, candidates, cache estimate, disk projection, memory envelope, gates, blocked reasons, checkpoint cadence, and next command.
- Minimal impl (present): pure planning, no model load and no encode. CPU worker defaults are profile-specific (`m3pro-local-max` 1, `studio-1tb` 8, `studio-m1ultra` 16). Dense-cache disk gates use `storage.estimate_for_encoder`. `mop_encode_autoselect.py` now records process/system/MPS memory snapshots and writes a blocked JSON receipt if model files are absent instead of losing the run to a traceback.
- Full impl: `src/mop/studio/dr1_schedule.py` consumes `encode_schedule.json` directly, splits DR1 shard legs by the checkpoint cadence, carries the measured device into `dr1_curate_bound_video.py --device`, and can emit a `mop-long-run-daemon/v1` plan. The daemon supplies heartbeat/resume; the DR1 merge and A6 guard remain explicit post-encode jobs.
- Dependencies: `profiles.py`, `storage.py`, `devices.py`, `memory_envelope.py`. Used by: Studio Wave 0 microbench, DR1 cache build, dense cache planning.
- Laptop-safe: yes (pure arithmetic). Studio-scale: yes. Prepares-for-custom-model: yes, a custom encoder cache still prices through the same profile and receipt path.

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
- Status: PARTIAL/NEW-as-aggregator. The pieces exist (`diagnostics/geometry.py` has linear/kernel CKA, RSA, effective rank, anisotropy, intrinsic dim, NN-overlap; `diagnostics/seed_consistency.py` has cross-seed CKA and Hungarian code agreement). There is no single `AlignmentSuite` that runs these across a SET of substrates/seeds and tables them.
- Purpose: measure representational agreement (between two substrates, between seeds of the same shell, between a substrate and a target RDM). The load-bearing scientific use: quantify whether two DIFFERENT-objective substrates converge (universal structure) or diverge (modality/objective-specific), the newest doctrinal control.
- Inputs: two or more `[N,D]` representations of the SAME points. Outputs: pairwise CKA matrix, RSA correlations, effective-rank/anisotropy per substrate, cross-seed CKA vs frozen-random floor.
- Minimal impl: a thin `alignment_suite(reps: dict[str, tensor]) -> dict` in `diagnostics/alignment.py` that calls the existing `geometry` and `seed_consistency` functions and assembles the table. Reuse, do not reimplement, `linear_cka`.
- Full impl: add Procrustes/CCA alignment and a permutation-test p-value on the CKA so "converged" is a certified claim not an eyeball.
- Dependencies: `geometry`, `seed_consistency`. Used by: EX12 atlas, P5/S5/Y3 idiolect tests, `CrossSubstrateAgreement`.
- Laptop-safe: yes (CKA is a few matmuls). Studio-scale: yes. Prepares-for-custom-model: yes, comparing a custom encoder's geometry to V-JEPA is exactly a CKA/RSA call.
- Doctrine flag: CKA is rotation-invariant, so it CANNOT distinguish real from a full-rank random projection either (same vacuous-control trap as probes). Any AlignmentSuite claim of specialness must use a random-ENCODER arm, not frozen_random_projection.

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
- Full impl (next): have CM9 or the licensed Process C launcher train this module on DR1 dense caches, compare to dense-without-slots and an off-the-shelf slot model, then write a null card. The laptop pass only proves tensor mechanics, budget gates, and controls.
- Dependencies: torch. Used by: CM9 object-centric binding, Process C moldability pilot, dense-token facet-8 follow-up.
- Laptop-safe: yes for unit tests and tiny tensors. Studio-scale: yes when licensed. Prepares-for-custom-model: yes, this is the first sanctioned trainable dense-token arm.
- Doctrine flag: `process_c_budget_report` refuses unlicensed runs and enforces the 1 to 10M cap by default. A slot win must beat dense-without-slots at matched capacity and pass binding-specificity checks; a tie is a null.

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
- Full impl (next): insert `scripts/verdict_gate.py` as a required daemon job before any positive-ledger doc update in DR1/PR9/Process C plans.
- Dependencies: `NullCardGenerator`, JSON receipts. Used by: Studio ledger updates and long-run daemon plans.
- Laptop-safe: yes, pure receipt validation. Studio-scale: yes. Prepares-for-custom-model: yes, because Process C positives cannot enter docs without an independent verifier receipt.
- Doctrine flag: this is a ledger gate, not a science metric. A failed gate blocks a doc update; it does not turn a run into a null.

---

## Observability and reproducibility layer

### ArtifactBundle
- Status: EXISTS. `studio/artifact_bundle.py` and `scripts/studio_artifact_bundle.py` write durable artifact indexes and optional small-receipt bundles.
- Purpose: close the audit's result-durability hole. A Studio wave should not leave load-bearing JSON/Markdown only under ignored `runs/`; the bundle indexes receipt paths, hashes, sizes, JSON validity, git tracking, copy status, and whether each artifact is durable.
- Inputs: explicit paths or presets (`pre-studio`, `wave0`), optional copy directory, copy-size cap, missing-artifact policy, and durability requirement. Outputs: `mop-artifact-bundle/v1` JSON plus optional copied small-text receipts.
- Minimal impl: present. Text receipts (`.json`, `.md`, `.txt`, `.yaml`, `.yml`, `.csv`, `.tsv`) can be copied into a durable directory; oversized or non-text artifacts are refused with a blocker reason. Large caches stay out of the bundle and must be represented by cache manifests.
- Full impl (next): after the M1 Ultra completes Wave 0, regenerate the `wave0` preset index with `--copy-dir proof/ARTIFACT_BUNDLES/wave0` and commit the index or otherwise transfer it with the Studio report.
- Dependencies: git CLI, `StudioTransferCheck` durable receipt list. Used by: pre-Studio transfer, Wave-0 artifact preservation, long-run daemon handoff bundles.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: indirectly, because Process C and dense-cache verdict receipts can be preserved without copying terabyte arrays.
- Doctrine flag: a missing artifact or failed copy is a durability blocker, not a scientific null.

### StudioTransferCheck
- Status: EXISTS. `studio/transfer_check.py` and `scripts/studio_transfer_check.py` implement the read-only Wave-0 transfer checklist.
- Purpose: prove the Studio received the governing stack and durable receipts before it spends compute. The check validates the active profile is `studio-m1ultra`, confirms the governing audit and required docs/scripts exist, parses the null-card schema, reports git branch/head/dirty state, confirms pre-Studio receipt files are present and git-tracked, and validates any cache manifests already present.
- Inputs: repo root, profile name, optional audit path, dirty-worktree policy, and receipt requirement flag. Outputs: `mop-studio-transfer-check/v1` JSON with per-check status and summary.
- Minimal impl: present. The CLI is read-only and writes a report with `--out`; dirty worktrees fail by default unless `--allow-dirty` is explicit.
- Full impl (next): after the M1 Ultra emits Wave-0 receipts, run `studio_artifact_bundle.py --preset wave0` and archive the emitted JSON beside `STUDIO_RUN_REPORT.md` on the Studio.
- Dependencies: `studio.profiles`, `substrate.cache_manifest`, git CLI. Used by: Studio Wave 0 transfer receipt and long-run daemon template.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: indirectly, by proving the receipt chain and profile envelope before Process C or dense-cache work can start.
- Doctrine flag: transfer check is not science; failing it blocks the wave instead of downgrading any scientific null.

### StudioWave0Report
- Status: EXISTS. `studio/wave0_report.py` and `scripts/studio_wave0_report.py` synthesize the Wave-0 transfer, daemon, encode, and memory receipts into one JSON summary plus a bounded Markdown block in `STUDIO_RUN_REPORT.md`.
- Purpose: make the actual M1 Ultra s/clip and memory envelope land in the scoreboard without a manual rewrite. The report block is bounded by markers and is replaced idempotently on rerun, so a resumed Wave 0 cannot duplicate stale rows.
- Inputs: transfer-check JSON, daemon state JSON, `encode_device.json`, and `encode_schedule.json`. Outputs: `runs/studio_wave0/wave0_report.json` and, with `--apply`, an auto receipt block in `docs/mixture_of_perspectives/STUDIO_RUN_REPORT.md`.
- Minimal impl: present. It records transfer pass count, daemon job summary, CPU/MPS s/clip, winner, launch gate, blocked reasons, process RSS peak, minimum available system memory, and MPS driver peak.
- Full impl (next): after the Studio completes the >=1000-clip cache rebuild, extend the same report with cache-manifest validation and actual cache count.
- Dependencies: the JSON receipts produced by `studio_transfer_check.py`, `studio_daemon.py`, and `mop_encode_autoselect.py`. Used by: Studio Wave 0 ledgering.
- Laptop-safe: yes, pure JSON/Markdown. Studio-scale: yes. Prepares-for-custom-model: indirectly, by keeping the Studio scoreboard receipt-driven before DR1/PR9/Process C.
- Doctrine flag: a missing or blocked receipt renders Wave 0 incomplete. It never converts a failed gate into a scientific null.

### StudioNativeLanes
- Status: EXISTS. `studio/native_lanes.py` and `scripts/studio_native_lanes.py` make the Studio-native facets machine-readable: DR13 rollout fidelity, hosted corpora, live-encoder doctrine, perspective ecology, developmental PR9, and Process C licensing.
- Purpose: turn the Part 2 audit lanes into concrete daemon jobs when their preregistered inputs exist, and into explicit blocked receipts when a prior artifact is missing. This keeps "not runnable yet" falsifiable instead of leaving it as prose.
- Inputs: profile name plus optional `clip_dir`, `dr1_cache`, `plan_path`, and `encode_schedule` paths. Outputs: a `mop-studio-native-lanes/v1` manifest and, for ready lanes, a standard long-run daemon plan.
- Minimal impl: present. Heavy lanes are blocked unless `--include-heavy` is set, large acquisition requires an explicit inspected `--plan-path`, and lanes without a sanctioned launcher record the release condition instead of fabricating a command.
- Full impl (next): wire perspective feature extraction into the manifest once DR1 has a merged cache and matched language/audio/object-centric feature stores.
- Dependencies: `LongRunDaemon`, `PerspectiveAdapter`, `ProcessCDenseTokenModule`, DR1 cache receipts, Wave-0 encode schedule. Used by: Studio-native lane planning after Wave 0.
- Laptop-safe: yes. Studio-scale: yes. Prepares-for-custom-model: directly, by keeping Process C behind an explicit PR9/DR1 licensing receipt.
- Doctrine flag: blocked lanes are walls-in-progress, not nulls. A lane can only move from blocked to ready when its named receipt exists.

### LongRunDaemon
- Status: EXISTS. `studio/long_run.py` and `scripts/studio_daemon.py` supervise a JSON job plan under an active Studio profile.
- Purpose: make week-scale Studio work boring and resumable: profile disk gate before each job, dry-run by default, state checkpoint after every transition, heartbeat events during long subprocesses, per-job stdout/stderr logs, resume-skip for completed jobs, and clean stop on blocked/failed jobs.
- Inputs: a daemon plan with `schema: mop-long-run-daemon/v1` and `jobs: [{id, cmd, cwd?, kind?}]`. Outputs: `daemon_state.json`, `logs/<job>.stdout.log`, `logs/<job>.stderr.log`, event history, and a summary by status.
- Minimal impl: present. The daemon is command-level infrastructure, not a science launcher. It does not choose DR1/PR9/Process C order; it enforces the profile and records execution for the chosen plan.
- Full impl: the daemon now validates a static pre-ledger contract: any `positive-ledger` job must be ordered after both a `verdict-gate` job and an `artifact-bundle` job. Because execution stops on prior failure, this makes the falsification and durability gates mandatory before a positive doc mutation can run. `scripts/studio_daemon.py validate --plan <plan.json>` checks the contract without running the plan.
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

- EXISTS (extend only): SubstrateRegistry, LatentStore, EncodeScheduler, NullCardGenerator, VerdictGate, SubstrateAdapter, PerspectiveAdapter, ProbeSuite, ReplayMemory, PlasticityController, NeuromodulationGate, ConsolidationEngine, CuriositySelector, UncertaintyEstimator, ReasoningLoop, CompressionDoctor, ExperimentRegistry, NullHypothesisRegistry, NegativeResultTaxonomy, ArtifactBundle, StudioTransferCheck, StudioWave0Report, StudioNativeLanes, LongRunDaemon, MetricsLogger, ReproducibilityHarness, ProcessCDenseTokenModule.
- PARTIAL (primitives exist, needs a thin aggregator or promotion): AlignmentSuite (geometry+seed_consistency exist), WorkspaceShell (compose existing shell modules), LatentScratchpad (WorkingMemory exists), FastWeightMemory (ex4-local), CriticalPeriodScheduler (controller knobs), MixtureArbitrator (e7 MoE router, promote only when reused).
- NEW (justified, no duplicate): CrossSubstrateAgreement (`diagnostics/cross_substrate.py`), the missing outer loop over substrates for standing-control 8; and the substrate-LEVEL variant of MixtureArbitrator, which is the least-committal architectural answer to the reopened dense-vs-custom fork but is gated behind CrossSubstrateAgreement showing complementarity.

The load-bearing architectural consequence: the corrected-substrate research now has reusable controls, receipts, and gated dense-token machinery. The remaining axis-moving gap is still DENSE real-video bound-attribute caches plus Studio execution, not laptop science. Building parallel WorkspaceShell/ReplayMemory/PlasticityController classes would be a duplicate and is explicitly rejected.
