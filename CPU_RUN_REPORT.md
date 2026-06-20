# CPU_RUN_REPORT

CPU-Now campaign on the Apple M3 Pro (18 GB, 12 physical cores: 6 performance + 6 efficiency).
Cost-ordered T0 -> T3, parallelized across cores, checkpointed. Tier E and R never run. Every
number is tagged: `real-encoder` (cached real V-JEPA latents) or `provisional` (the direct
synthetic latent generator, which the grids use for speed/coverage and must be re-run on real
natural-video latents later).

## The unlock (Section 3)
- Real V-JEPA 2 ViT-L weights FETCH + LOAD from HuggingFace (network reachable). hidden_size
  1024; input [B,64,3,256,256] -> last_hidden_state [B,8192,1024]. The previously-deferred real
  forward path is fixed (`pixel_values_videos=`) and exercised.
- Real-encoder cache: a 96-latent store cached through the REAL frozen encoder over structured
  synthetic video, backend=vjepa_hf, MEASURED **24.3 s/clip on CPU** (MPS hung on the 64-frame
  ViT-L; CPU works). This proves the real pipeline end to end and yields real-encoder results
  (see the REAL-ENCODER section below).
- No natural-video dataset here, so all campaign science runs `provisional`. Real-encoder
  results at useful scale are a Studio task (drop SSv2/Ego4D clips; encoding is ~30x faster on a
  GPU). Real weights are OPT-IN (`encoder.prefer_real`) so the test suite stays deterministic.

## REAL-ENCODER results [real-encoder, promoted from provisional]
A 96-clip real V-JEPA latent store was cached on CPU (backend=vjepa_hf, MEASURED **24.3 s/clip**,
~39 min total) over 6 distinct structured-synthetic visual classes, then evaluated
(`runs/real_encoder_eval.json`):
- **Linear-probe distinctiveness (the corpus's central diagnostic), on the REAL encoder:**
  acc **1.000** vs chance 0.167 at n=96 -> visual-class info IS linearly decodable from real
  V-JEPA latents. (The n=8 smoke earlier was underpowered; n=96 is a real answer.)
- **Forget-then-retain on REAL latents** (class-incremental, 3 tasks, shared head): naive
  BWT **-1.000** (catastrophic: final accuracy collapses to chance 0.167), replay+EWC BWT
  **0.000** with final accuracy **1.000** (perfect retention). The headline E1 contract --
  a naive learner forgets, replay+EWC retains -- now demonstrated on REAL V-JEPA geometry,
  not just synthetic latents.
- **E2 replay schemes on REAL latents** (promotes E2 to real, no new video): naive -1.000 ->
  random replay -0.250 -> prioritized replay -0.250 -> replay+EWC 0.000. Replay beats naive
  decisively; **prioritized ties random** (the corpus's predicted E2 half-null), now confirmed
  on REAL latents, not just synthetic.
Caveat: video CONTENT is structured-synthetic (no natural-video dataset here). Re-run on
SSv2 / Ego4D clips to remove the synthetic-content caveat; the pipeline is proven and GPU
encoding is ~30x faster.

## Parallel CPU harness (Section 2A)
`src/devsys/harness/cpu_pool.py`: process-pool over independent run-units, spawn isolation,
per-worker BLAS thread caps (OMP/MKL/OPENBLAS/VECLIB/NUMEXPR + torch.set_num_threads) so
workers x threads ~= 12, memory-aware worker cap, per-unit json checkpoint (resumable), serial
fallback on a broken pool. small mode = 12 workers x 1 thread; heavy = 4 workers x 3 threads.
Determinism leg ran single-threaded/serial for a clean baseline. T2 pool: 20 units, 0 degraded,
11.5 s wall.

## T0 (instant, complete)
- **11A determinism** [provisional]: CPU is **bit-identical** run-to-run (E1 protected_bwt and I4
  ceiling both byte_identical, rate 1.0, max_abs 0.0 over 3 reps). Confirms CPU >> Metal (~50%
  byte-identical at temp 0); CPU is the tolerance baseline.
- **11D negative-result registry** [provisional]: 8 experiments, **5 confirmed / 2 refuted / 1
  mixed**. Refuted nulls: E2 (replay does beat no-replay) and I4 (alternatives reach backprop on
  separable latents). Each verdict mapped to a negative-result taxonomy category. Table:
  `runs/negative_registry.md`.

## T1 (cheap, complete)
- **Diagnostics at real scale** [provisional]: linear-probe decodable (acc 1.0 on separable
  latents); noisy-TV battery PASSES all three contracts (noise error stays high, epistemic
  disagreement collapses on irreducible noise, learning-progress separates); calibration + Fisher
  trace computed. Plots under `runs/`.
- **11B seed-variance / power** [provisional]: SEM of the E1 protected-vs-naive gap falls
  0.094 (S=2) -> 0.039 (S=5); sign stabilizes at S=3. **Recommended seeds: headline 5, ranking 5,
  sanity 3.** This sets the Studio seed budget.

## T2 (low, complete) -- the standout CPU deliverables
- **E1 expanded** [provisional], 5 seeds x 3 stream types (mean +/- seed std):
  - domain-incremental: naive BWT -0.229 +/- 0.036, protected -0.049 +/- 0.015, gate 5/5
  - task-incremental: naive -0.229 +/- 0.036, protected -0.049 +/- 0.015, gate 5/5
  - class-incremental: naive -0.443 +/- 0.049, protected +0.091 +/- 0.021, gate 2/5 (the harder
    new-class-learnability regime: replay+EWC RETAINS strongly but the naive arm sometimes fails
    the "both learn last task" clause; documented, not a defect)
  The headline holds with proper seeds: naive forgets, replay+EWC retains.
- **I4 backprop alternatives** [provisional], 5 seeds x 3 head sizes: backprop ceiling 1.000;
  on separable latents all 6 alternatives land within the 0.03 margin (null not supported on
  accuracy), so the differentiator is locality / measured activation memory (the corpus's E9
  point), not accuracy. Table: `runs/i4_backprop_alts/*/i4_table.md`.
- **E9 streaming local** [provisional]: local rules reach within margin online AND win on
  measured activation memory (eqprop/PC retain ~0 autograd activations vs backprop's full graph).
- **8B energy associative memory** [provisional]: modern-Hopfield retrieval capacity **6x D** vs a
  feedforward autoassociator's **1x D** -- energy-based memory wins decisively on capacity.
- **8C predictive-coding vs depth** [provisional]: PC matches backprop at depth 1 (1.00 vs 1.00),
  gap **widens with depth** (1.00 vs 0.88 at d2, 1.00 vs 0.86 at d3). PC approximates backprop
  shallow, degrades deep -- the expected finding.

## T3 (affordable, drained -- 10/10 units within the 420 s budget) [provisional]
Reduced seeds + representative axis subsets (recorded in DECISIONS.md):
- E2 replay (tie_tol 0.02/0.05): frontier_auc 0.941 both -> prioritized ties random at this scale
  (the corpus's predicted E2 half-null).
- E3 plasticity (soft vs hard): frontier_auc 1.011 both -> staged ties tuned decay ("just an LR
  trick" null cannot be rejected on a frozen substrate, as predicted).
- E4 neuromod (noise_scale 4/6): null not supported -> ensemble-disagreement gating ignores
  noisy-TV while point-error chases it.
- E7 sparse (k-WTA 4/8): null not supported at toy scale (interference reduced vs matched dense).
- Consolidation (the sharp weight-space test): SI-only frontier_auc -0.064, EWC+SI -0.043, both
  below replay+EWC. On a frozen encoder, weight-space consolidation adds little over replay --
  the corpus's central prediction, now with a CPU data point.

## Studio cost projection (Section 5) [laptop-throttled, conservative]
From measured per-run-unit timings x full-scale `run_queue.yaml` run-units (full axis factorial x
5 seeds), parallelism = serial / effective_workers:
- Tier C (cached-latent, runnable now): **~0.11 h** parallel wall full-scale (cheap; basis =
  measured CPU times).
- Tier E (environment): **~1.5 h** (assumption-based, x4 run-units, 1800 s/unit assumed; needs env).
- Tier R (rented CUDA / lab-scale): **~20 h** (assumption-based, x8, 7200 s/unit assumed).
- Grand total: **~86.5 h serial, ~21.6 h parallel wall** full-scale.
Note: measured CPU times are thermal/shared-core throttle-limited and CONSERVATIVE; Studio
wall-clock will be no worse, likely much faster. E/R numbers are levers (assumptions), not
measurements, and only become real with an environment + rented CUDA. Full detail:
`runs/cpu_campaign/summary.json` (cost_projection) and `runs/cost_projection.md`.

## Legs skipped for open defects
None. ISSUES.md carries no open defects this session (the hardening wave closed them all), so no
leg was dependency-skipped.

## Next steps
Re-run on real latents (when a natural-video sample exists; encoding ~30x faster on GPU):
- Cache real-encoder latents from SSv2 / Ego4D / EPIC-KITCHENS clips (the unblock for the central
  linear-probe distinctiveness result and a real-latent E1 frontier).
- Re-run E1 expanded, the diagnostics battery, E2/E3 grids, and I4/E9 on real latents; promote
  every `provisional` tag to `real`.
Studio-only (Tier E / R, currently disabled in the queue):
- E5 curiosity env-rollout variant, E6 object-centric on real 2.1 dense features, E10 capstone +
  POET + cultural-accumulation legs. Use the cost projection to budget rented-CUDA hours first.
- Scale the seed set per 11B (headline/ranking 5, sanity 3) and the full axis factorials.

## Artifacts
`runs/cpu_campaign/{summary,t0,t1,t2,t3}.json`, `runs/negative_registry.md`,
`runs/cost_projection.md`, `runs/i4_backprop_alts/*/i4_table.md`, diagnostic + E1 plots under
`runs/`. Per-unit checkpoints in `runs/cpu_campaign/ckpt/` (the campaign is resumable).
