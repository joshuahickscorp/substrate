# Section 10: Heavy vs Local Runs, the Three Compute Tiers

This section maps the whole program onto the compute it actually has (and could get), and states
the one rule that governs all of it: Brain is NOT training-compute-bound. The frozen V-JEPA 2
substrate never trains (a grad-free unit test holds in every experiment), the trainable shell is
tiny (roughly 3 to 10M params: predictor, heads, ensemble, buffer, plasticity, consolidation,
neuromod), and all learning iterates on CACHED pooled latents, never on pixels. So a bigger tier
never buys a bigger model. It buys exactly three things, in priority order: THROUGHPUT (lift the
64-frame MPS encoder block so latent caching is not stuck at the CPU floor), DATASET SCALE (a
larger permanent multi-encoder cached-latent corpus, especially the deferred prerequisite of real
natural-video with non-additively bound attributes), and STATISTICAL REPLICATION (the whole bank
at 5 to 10 seeds with real error bars).

Grounding docs for this section: APPLE_SILICON.md (the MPS-first story and the measured 64-frame
block), STUDIO_HANDOFF.md (what transfers, what is genuinely gated, the priority list), SCALING.md
(the flip-list), docs/STUDIO_MAXIMIZATION_2026_06_27.md (the corpus and encoder-scale plan), and
src/devsys/studio/profiles.py (the shipped m3pro-local-max and studio-1tb profiles).

A framing warning that governs every tier below. The single most important methodological
correction in the corpus (section 3d, the vacuous-control discovery) means MORE compute does not
by itself buy a stronger claim. frozen_random_projection is a full-rank invertible 1024x1024
matrix, so a linear or MLP probe absorbs the inverse and the delta is mathematically forced to
0.000. Scaling a probe-based experiment to 100k clips on a wider box reproduces a vacuous result
at higher confidence, not a real one. The correct substrate control (real V-JEPA vs random-ENCODER
or random-init-ViT features) and the correct binding constraint (a non-ceiling, non-additive test
bed) are DATA and DESIGN problems that a bigger machine only partly touches. Tier up when a real
control or a real test bed needs the compute, never to add seeds to a vacuous probe.

---

## Tier 0: Laptop (M3 Pro, 6P + 6E, ~18 to 19 GB unified, ~53 to 63 GB free disk)

The machine the entire catalogued corpus ran on. All ~119 experiments (117 implemented) were run
here for real. This tier is cached-latent-first and CPU-bound for the one thing it cannot do on
Metal.

### The hard ceiling (measured, not assumed)

A 64-frame ViT-L forward produces 8192 tokens and HANGS the MPS graph compiler on this M3 Pro.
Real-encoder caching therefore falls back to CPU at a MEASURED 21 to 32 s/clip (the verified
64-clip real cache took ~1355 s total, ~21 s/clip, over ~23 minutes). The same command with
device=mps overflows with `RuntimeError: Invalid buffer size` even at batch=1: this is a hard
PER-BUFFER ceiling on 64-frame/256px ViT-L attention, NOT a total-RAM limit, so more system memory
would not fix it. The knob `mps_safe_token_cap` in configs/device/mps.yaml documents the token
threshold above which an encoder forward routes to CPU. Everything else (heads, predictors,
ensembles, buffer, the whole shell) runs fine on MPS.

### Top experiments this tier enables

- The full Tier C shell bank on pooled latents: e1_baseline through e10, the continual-learning
  and plasticity series, at toy-to-moderate scale, seconds-to-minutes each on MPS.
- The two surviving positives at the scale they were found: e7_sparse (sparse/gated heads halve
  catastrophic forgetting vs param-matched dense, 30-run seed/axis sweep already run here via
  devsys.harness.sweep.run_sweep) and ex2_latent_planning's synthetic-control precursor (the
  y7_controllability_sysid_gate / d6_rollout_gate sysid on synthetic action-conditioned
  transitions).
- The corrected substrate test that LANDED here: substrate_vs_random_features.py (real V-JEPA
  decodes shape under heavy nuisance at 0.379 vs random-pixel untrained features 0.069, chance
  0.167, delta +0.31), plus the two in-flight controls (substrate_vs_random_init_vit.py and
  compositional_under_nuisance.py). Note: a full-res real-encoder cache at real scale is exactly
  what this tier is slow at (single-threaded CPU at ~21 s/clip); the +0.31 result used a small,
  RAM-forced 32px random-pixel arm, which is the honest confound (see the tier-2 justification).
- Every pure-numpy or small-latent experiment: the plasticity, consolidation, neuromod series;
  the local-rules gap study (ex5); the long-stream continual harness (ex13_long_stream, run here
  at 240 and 3000 tasks); the rejuvenation lead (ex15) once implemented (clone the ex13 harness).
- All diagnostics and standing controls: substrate_ablation.py, compute.py, sysid.py,
  difficulty_calibration.py, transfer_matrix.py, buffer_compression.py, latent_robustness.py,
  determinism.py, noisy_tv.py.

### Scaffolding / diagnostics / models supported

Full registry and doctrine infra (registry/experiments.yaml, models.yaml, datasets.yaml). The
studio_pipeline.py conveyor runs here in local-max rehearsal (all 12 gate stages pass under the
m3pro-local-max profile). Local HF weights present and usable: vjepa2-vitl-fpc64-256 (canonical),
dinov2-large (fully downloaded, staged for cross-encoder CKA/RSA), VideoMAEv2-Base (fully
downloaded, staged). The verified real cache data/cache/vjepa2_vitl_fpc64_256_real/ (count 64,
linear probe acc 1.0 vs chance 0.125).

### What NOT to attempt here

- Do NOT try the 64-frame ViT-L forward on MPS: it overflows the per-buffer ceiling regardless of
  RAM. Route encoder forwards above mps_safe_token_cap to CPU.
- Do NOT pull both ViT-H (~2.5 GB) and ViT-g (~4 GB) weight shards at once: it would breach the
  ~60 GB free-disk floor. This laptop already runs close to its own min_free_disk_gb=60 floor
  (disk drifted 63 to 53 GB in one session from unrelated system activity, and correctly tripped
  the kill switch).
- Do NOT run a second torch/encoder job while a CPU-bound V-JEPA encode is in flight: an 18 GB
  pool OOMs on two encoders.
- Do NOT cache DENSE latents here: dense ViT-L is ~32 MB/clip (8192 tokens), so 10k dense clips is
  ~313 GB, far past the disk floor and past the point of the MPS block anyway.
- Do NOT trust a probe-based "real ties frozen-random" result at any scale on this tier: it is
  vacuous by construction (invertible matrix), not a substrate finding.

### Expected bottlenecks

Encode throughput (the 21 s/clip CPU floor), free disk (~60 GB floor, already tight), sustained
thermal throttling (measured CPU timings are laptop-throttled and conservative vs a better-cooled
box), and the per-buffer MPS ceiling on large-token forwards. NOT bottlenecked on latent storage
(pooled ViT-L is ~8 KB/clip; 10k clips ~= 78 MB) or on shell compute (seconds on MPS).

### Relative cost and implementation difficulty

Cost: BASELINE (owned hardware, electricity only). Difficulty: LOW for shell/diagnostic work
(already built and green), MODERATE for a large real cache (slow, resumable per-clip, so it just
takes wall-clock days at ~21 s/clip; a 5k EPIC slice is ~29 hours CPU, a 20k slice ~5 days).

IMPORTANT (envelope reconciliation, do not read the 5k/20k line as a sanctioned default-profile
Tier-0 job): the ONLY shipped laptop profile, M3PRO_LOCAL_MAX in src/devsys/studio/profiles.py,
hard-caps max_cache_clips=128, so clamp_clips() would SILENTLY truncate a 5k or 20k request to 128.
The CPU-time arithmetic above is correct, but a real cache past 128 clips is NOT runnable under the
shipped kill-switch: it requires an explicit manual override raising max_cache_clips past 128 (which
takes it out of the default envelope), or it belongs on Tier 1, where studio-1tb's max_cache_clips=2e6
actually permits it. Treat any 5k-20k real cache as a Tier-1 job, or as a Tier-0 job ONLY under a
logged manual profile override, never as a default-profile activity.

Second, budget the RAW-VIDEO disk footprint, which the pooled-latent line (78 MB for 10k clips) hides:
the raw clips must be decoded to disk before the encoder ever runs (roughly 1 to 3 MB/clip per
STUDIO_MAXIMIZATION scale-of-100k), so a 5k slice is ~5 to 15 GB raw and a 20k slice ~20 to 60 GB raw.
Against a laptop that sits at ~53 to 63 GB free with a 60 GB min_free_disk_gb floor (a floor that has
ALREADY tripped the kill switch once), free_disk_ok() would REFUSE to start these caches: they are
disk-blocked before they are time-blocked. So even under a manual max_cache_clips override, a 5k-20k
real cache is not a laptop job at the current disk headroom; it is a Tier-1 job (900 GB budget).

### Required code modules

src/devsys/devices (resolve/autocast/safe_to), configs/device/mps.yaml and cpu.yaml,
substrate/ (video.py, cache.py, latent_store.py, storage.py), scripts/cache_real_encoder.py and
cache_video.py, the diagnostics suite, harness/cpu_pool.py (process pool sized to physical cores
with per-worker BLAS caps to avoid oversubscription), devsys.harness.sweep.run_sweep (real per-run
seed overrides; a config-only seed override is a silent no-op for modules that read cfg.seed).

### What result would justify moving to Tier 1 (the Mac Studio)

Any one of: (1) the corrected substrate finding needs its resolution confound removed, meaning a
real-encoder cache at 256px at real scale (hundreds to thousands of clips) that CPU-at-21s/clip
makes painfully slow; (2) e7_sparse's forgetting advantage needs a formal significance test on
REAL cached latents (only synthetic Gaussian-cluster so far) at 5+ seeds; (3) the deferred
prerequisite (real natural-video with non-additively bound attributes) needs building, which is
gated on encode throughput; (4) the encoder-scale question (ViT-L vs H vs g) needs the two larger
frozen encoders that will not fit disk here. Any of these is a THROUGHPUT or DATASET-SCALE gate,
which is precisely what Tier 1 buys. None of them is a "train a bigger model" gate.

---

## Tier 1: Mac Studio (Apple Silicon, M2 Max class, ~38 GPU cores, 96 GB unified, 2 TB SSD, ~400 GB/s)

The documented next box. Same Metal/MPS code, bigger chip: the scale-up is "same MPS code, bigger
unified pool", not a port. It is NOT a CUDA box. Wall-clock is explicitly a NON-constraint on this
plugged-in workstation: optimize for corpus scale, replication, and thoroughness, not speed. The
shipped profile is studio-1tb (disk_total_gb=1000, download_budget_gb=900, max_cache_clips=2e6,
max_run_count=100000, min_free_disk_gb=60, allowed_tiers={C, E}); this 2 TB machine has extra disk
headroom that the 1 TB profile already covers, and a studio-2tb profile would be a one-line numbers
change if dense caches are ever wanted.

### The first thing to verify (a hypothesis, not an assumption)

Whether the ~38-GPU-core / 96 GB Studio LIFTS the 64-frame MPS block. Procedure (APPLE_SILICON.md
"What to flip", STUDIO_MAXIMIZATION Section 1): smoke the smallest real forward
(`scripts/cache_real_encoder.py device=mps +classes=2 +per_class=1`, one [B,64,3,256,256] ->
[B,8192,1024] forward). PASS = latents return without hanging. If it passes, measure MPS s/clip
vs the 24 s/clip CPU floor (historical expectation is roughly an order of magnitude faster, i.e.
~2 to 3 s/clip, but this is a MEASUREMENT to record, not a claim). If it still hangs, fall back in
order: CPU encode via mps_safe_token_cap (fine under unlimited wall-clock), reduced-token cache on
Metal (tagged as a throughput lane, not the canonical 64-frame cache), or the opt-in MLX encode
experiment (never on the science hot path, never a yak-shave).

### Top experiments this tier enables

- The permanent multi-encoder cached-latent corpus (the single highest-value job): real
  natural-video cached through ViT-L (1024), ViT-H (1280), and ViT-g (1408), three parallel pooled
  stores over the SAME validated raw corpus. Pooled is tiny: a 100k-clip corpus across all three
  encoders is ~3 GB of latents; even 1M clips across three encoders is ~30 GB. The cost is raw
  video on disk (tens to a few hundred GB) and ENCODE TIME, not latent footprint. Because the
  encoder is frozen, this corpus NEVER goes stale: a one-time permanent asset.
- The encoder-scale experiment (the over-engineering centerpiece): does bigger frozen perception
  change WHICH shell mechanisms help? Same shell, same experiments, same clips, same seeds, only
  the latent source differs (the shell auto-resizes latent_dim per encoder config). Null: substrate
  scale does not change which mechanisms win; a sign-flip or a help-on-g-not-L is the interesting
  positive. This is science the laptop literally cannot produce.
- The corrected substrate test at 256px at REAL scale: this is the cleanest early Studio win.
  substrate_vs_random_init_vit.py (real vs random-init same-arch ViT-L, same 256px) at real scale
  isolates PRETRAINING from architecture+resolution and removes the acknowledged 256px-vs-32px
  confound behind the +0.31 nuisance-invariance result. Whether the honest delta lands near the raw
  +0.31 or the discounted ~0.21 to 0.23 is exactly what this control settles, and it wants full-res
  encode at scale.
- e7_sparse on REAL cached latents at 5+ seeds with a formal significance test (the corpus's
  strongest surviving positive, previously only synthetic Gaussian-cluster).
- The full Tier C bank at full seeds on real natural-video latents: track01 e1_gate (30 units)
  through the Level-5 headline (E2+E3+E4 combined, 30 units) and track09 e6_relational pooled
  fallback (10 units). Tier C total ~198 full-scale run-units; crossed with three encoders that is
  ~594 pooled inference runs, each seconds-to-minutes on MPS.
- The atlas/decodability battery (ex12_atlas, d1_geometry, a1_affordance_decode) on REAL ViT-L
  latents at real scale, with DINOv2-large and VideoMAEv2-Base wired in as second/third encoder
  points for cross-encoder CKA/RSA (both already downloaded, staged).
- The bleeding-edge EX bank (EX1 generative latent replay, EX3 test-time training, EX4 hypernets,
  EX5 local rules at scale, EX6 free-energy objective, EX7 MAML/Reptile, EX8 curiosity bake-off,
  EX9 slot attention over pooled sequences): all Tier C, all method-richness on the frozen-latent
  substrate, each a config plus a small shell module.
- The build-real-natural-video-with-bound-attributes prerequisite: this is where the deferred
  prerequisite for BOTH doctrinal questions can finally be built (color/shape/position/motion
  entangled, cached through the frozen encoder), so a compositional or abstraction test can at last
  be non-ceiling and non-additive.

### Scaffolding / diagnostics / models supported

The full studio_pipeline.py conveyor under studio-1tb (plan -> acquire -> validate -> cache ->
gated run -> report), with the gated kill-switch chain (validate source -> cache -> validate cache
-> linear probe -> noisy-TV/diagnostics -> E1 smoke -> Tier C toy -> Tier C full only if gates
pass). Multi-encoder cache stage (--encoder flag, three runs over one corpus). All standing
controls wired at real scale (the handoff's most important lesson: the most common corpus failure
was a control existing in code but never wired into the specific experiment). Weights to pull once
disk is a non-constraint: vjepa2-vith-fpc64-256 (~2.5 GB, currently config-only stub, unlocks the
ex12 encoder-scale falsifier) then vjepa2-vitg-fpc64-384 (~4 GB, config-only stub).

### What NOT to attempt here

- Do NOT run device=cuda work: the Studio is Apple Silicon, no CUDA. cuda is reserved for rented
  Tier R env rollouts only.
- Do NOT cache DENSE latents by default: dense ViT-L is ~32 MB/clip, a 10k dense corpus is ~313 GB.
  Dense is the E6 / V-JEPA 2.1 path and stays DEFERRED; even on 2 TB it is a deliberate, budgeted
  decision, not automatic.
- Do NOT attempt E6 dense relational, E5 live-rollout, or E10 capstone: E6 needs V-JEPA 2.1 dense
  weights (not on HF, placeholder 404s), and E5-rollout/E10 need a procedural environment to ACT in
  plus rented CUDA. These do NOT unblock just because a powerful box arrived (STUDIO_MAXIMIZATION
  Section 5). Deferred means deferred.
- Do NOT auto-pull SSv2 or Ego4D: both are status manual (Qualcomm/20BN terms, signed Ego4D
  license + AWS creds). The planner refuses them until the user completes terms and passes
  --accept-license. Full Ego4D / Ego-Exo4D are deferred (multi-TB) and never planned by default.
- Do NOT let auxiliary encoders (VideoMAEv2, DINOv2, distilled/quantized) stand in for the canonical
  frozen V-JEPA substrate in any science result (replaces_canonical: false is a hard invariant).
  They run in the OPTIMIZE lane or as cross-encoder geometry points only.
- Do NOT scale a probe-based experiment to buy confidence in a vacuous tie: the invertible-matrix
  problem does not go away with more clips. Spend the throughput on the corrected controls and the
  bound-attribute test bed instead.

### Expected bottlenecks

Encode wall-clock IF the MPS block does not lift (at the 24 s/clip CPU floor a 100k-clip corpus is
~28 days per encoder, ~83 days for all three; fine under unlimited wall-clock, but the argument for
lifting the block: at even 10x it is ~3 days per encoder). Raw video disk (a few hundred GB inside
the 900 GB usable budget). Manual-dataset access (human-in-the-loop license steps). NOT bottlenecked
on unified memory (96 GB dwarfs a tiny shell), latent storage (pooled is a few GB), or shell
compute.

Wall-clock profile reconciliation (the 48h cap vs the 83-day corpus): "wall-clock is a non-constraint"
is a POLICY posture (this plugged-in workstation may run for months), NOT a per-run license. The
shipped studio-1tb profile sets max_wall_min=60*48 = 2880 min = 48 hours, so any SINGLE run leg that
honors the profile is killed at 48h, ~40x short of the ~83-day multi-encoder corpus. This is NOT a
contradiction to fix by raising the cap: the cache is described as RESUMABLE PER-CLIP, so the corpus
job must be CHUNKED into resumable legs each under the 48h max_wall_min (each leg encodes a bounded
clip range, checkpoints, and the next leg resumes from the last cached clip). The 83-day figure is the
SUM of many sub-48h resumable legs, not one 83-day process. State this explicitly in any campaign
leg definition (campaign/legs/): the flagship corpus is a queue of resumable per-clip-range legs, so
it never trips its own 48h wall-time kill switch. Raising max_wall_min for a monolithic corpus run is
the WRONG fix (it defeats the per-clip resumability the profile is built around).

### Relative cost and implementation difficulty

Cost: MODERATE relative to the laptop (owned or borrowed workstation, still no rented compute, no
API spend); the real spend is wall-clock, which is declared free. Difficulty: LOW to build the
corpus and run the bank (config + scripts, the conveyor exists), MODERATE for the never-built EX
scaffolds (each a small shell module) and for authoring the missing null cards, and MODERATE to
HIGH for building the genuinely-new bound-attribute natural-video test bed (a data-curation and
design problem, not a compute one).

### Required code modules

Everything from Tier 0 plus: scripts/studio_pipeline.py (plan/acquire/validate/cache/run/report/
optimize/local-max lanes), scripts/studio_doctor.py, scripts/studio_rehearsal.py,
src/devsys/studio/profiles.py (studio-1tb), campaign/run_queue.yaml and campaign/legs/ (tiered leg
definitions), configs/encoder/{vjepa2_vitl_fpc64_256, vjepa2_vith, vjepa2_vitg}.yaml, the report
scaffolds (seed summaries, CIs, effect sizes, adaptation-retention frontiers, null-result registry),
and the north_star safety scanner that gates every rendered report against sentience/agency claims.

### What result would justify moving to Tier 2 (a wider-training-box)

The Studio covers essentially everything the doctrine sanctions, because the doctrine forbids
training perception. So Tier 2 is justified ONLY by a result that forces the program to question
the frozen-substrate doctrine itself. Concretely: if the corrected controls show the frozen
substrate is BOUNDED (a target that a real natural-video, non-additive, non-ceiling test needs but
that neither ViT-L nor ViT-H nor ViT-g decodes off-ceiling, i.e. the pooled/dense fork resolves
against pooling AND dense 2.1 still cannot factor it), THEN and only then does a from-scratch or
fine-tuned encoder become a justified hypothesis to test, and that needs a training box. Absent
such a bounding result, Tier 2 is out of scope by doctrine. A weaker trigger: rented CUDA for the
genuinely env-gated Tier R legs (E5 live-rollout, E10 capstone), which need a procedural environment
plus a CUDA box, is a narrow Tier-2-adjacent step that does not touch the frozen encoder.

---

## Tier 2: Wider-training-box (rented CUDA or a multi-GPU cluster, from-scratch / fine-tune training)

This tier exists to violate the frozen-substrate doctrine on purpose, under a pre-registered
hypothesis, OR to run the two genuinely environment-gated legs. It is NOT the default growth path
and must be justified by a bounding result (above), never entered to "just train something bigger".

### Top experiments this tier enables

- The ONLY doctrine-sanctioned use that touches training: Tier R environment rollouts on rented
  CUDA (device=cuda, configs/device/cuda.yaml: amp true, pin_memory true, num_workers 8,
  allow_cpu_fallback false). ex2_latent_planning's LIVE closed-loop MPC (needs real
  action-conditioned rollout data from an interactive robot/ego environment), E5's curiosity
  live-rollout arm, and the E10 open-ended capstone (procedural env + population). These are blocked
  on an ENV ADAPTER, not on GPU alone: the compute is secondary to building the environment.
- The doctrine-questioning, out-of-scope-until-justified use: training or fine-tuning an encoder
  from scratch at the SAME resolution and frame budget to test whether a task-specific perceptual
  frontend beats the frozen general V-JEPA on the bounding target the frozen substrate could not
  decode. This is a HYPOTHESIS TEST against the frozen baseline, with the frozen encoder as the
  control it must beat by a stated margin, not a replacement of the doctrine.

### Scaffolding / diagnostics / models supported

configs/device/cuda.yaml (the rented-box path), the env-adapter interface that does not yet exist
(the genuine build gap for E5-rollout/E10/ex2-live), standard multi-GPU training scaffolding (DDP,
sharded checkpointing, mixed precision), and the SAME standing controls (a from-scratch encoder must
still beat the frozen substrate, matched-compute, and a tuned baseline, and its features must beat
random-init same-arch at the same resolution).

### What NOT to attempt here

- Do NOT enter this tier without a bounding result that a frozen encoder (across all three scales,
  and dense 2.1 once it ships) provably cannot supply. Training perception is the doctrine's
  explicit out-of-scope line; crossing it requires a pre-registered null and a frozen-substrate
  baseline the trained model must beat.
- Do NOT treat "we rented a GPU" as license to abandon cached-latent-first for the shell: only the
  perceptual frontend (or the env rollout) needs the box; the shell stays tiny and latent-first.
- Do NOT run env rollouts before an env adapter exists: the blocker is the environment harness the
  repo has no code for, not the GPU. Build the adapter first.

### Expected bottlenecks

For env rollouts: the missing environment adapter (a genuine implementation gap), then rented-CUDA
cost and the sim-to-frozen-latent interface. For from-scratch training: real training-compute cost
(this is the one tier that is genuinely training-compute-bound), data licensing for a from-scratch
video corpus at scale, and the risk that the trained frontend fails to beat the frozen baseline
(the most likely outcome given the +0.31 evidence that pretraining already buys real
nuisance-invariance).

### Relative cost and implementation difficulty

Cost: HIGH and the only tier with real marginal spend (rented GPU-hours, possibly a cluster).
Difficulty: HIGH (from-scratch training pipeline, DDP, data at scale) to VERY HIGH (a procedural
interactive environment plus its adapter, which does not exist in the repo at all).

### Required code modules

configs/device/cuda.yaml (present), an env-adapter module and a rollout harness (both absent, must
be built), standard distributed-training scaffolding (absent by design; the repo trains nothing
large), and the unchanged standing-control suite so any trained-encoder claim is gated the same way
every frozen-substrate claim is.

### What result would justify staying vs abandoning this tier

Stay only if the trained frontend beats the frozen V-JEPA baseline by a stated margin on the
bounding target, survives matched-compute, and its features beat random-init same-arch at the same
resolution. If it ties the frozen substrate (the expected outcome), the result is a strong
confirmation of the frozen-substrate doctrine and the program returns to Tier 1. For the env legs:
stay only if the live-rollout MPC beats the flat reactive head and action-shuffle on TRUE dynamics
(ex2's Tier 0/1 synthetic precursor already sets that bar); otherwise the environment gap was not
worth the CUDA spend.

---

## Cross-tier summary table

| Axis | Tier 0 Laptop | Tier 1 Mac Studio | Tier 2 Wider-training-box |
|---|---|---|---|
| Hardware | M3 Pro, ~18 GB, ~53 to 63 GB free disk | M2 Max class, 96 GB, 2 TB, ~38 GPU cores | rented CUDA / cluster |
| Backend | MPS + CPU fallback | MPS (verify 64-frame lift) | CUDA |
| 64-frame ViT-L encode | CPU only, ~21 s/clip (MPS overflows) | likely MPS (hypothesis, measure); else CPU floor | GPU |
| Latent policy | pooled cached, tiny | pooled multi-encoder corpus, permanent | pooled shell unchanged; only frontend/env trains |
| Top job | shell bank + corrected substrate test | permanent corpus + encoder-scale + bound-attribute test bed | env rollouts; doctrine-questioning encoder test |
| Trains perception? | never | never | only under a pre-registered bounding hypothesis |
| Relative cost | baseline | moderate (wall-clock free) | high (real GPU-hours) |
| Impl difficulty | low | low-moderate (moderate-high for new test bed) | high to very high |
| Move-up trigger | throughput/dataset-scale/replication gate | a bounding result (frozen substrate provably insufficient) or env-gated legs | trained frontend beats frozen baseline, or abandon |

The through-line: the laptop already ran the whole catalogued corpus and found the two surviving
positives and the one valid substrate signal. The Studio is a THROUGHPUT-and-SCALE unlock for the
corrected controls, the multi-encoder corpus, and the still-unbuilt non-additive test bed, all
still on a frozen substrate. A training box is out of scope by doctrine until a bounding result
forces the question, and even then the frozen encoder is the baseline it must beat.
