# Brain: Studio Maximization, Experiment Bank, and Path to Recognition (2026-06-27)

This is ONE continuous workflow document. It runs end to end, in the order the work happens:
research substrate (the frozen V-JEPA 2 perception and the cached-latent corpus it feeds) ->
experiments (the full E-series campaign and the bleeding-edge EX-series bank) -> reusable
artifacts (the representational atlas, the cached-latent corpus, the citable negative results)
-> positioning (where Brain honestly stands in the 2025-2026 landscape) -> the path to
recognition (which preprint, which workshop, which community, in which order). The experiments
are not the goal; they exist to PRODUCE citable, reusable artifacts. A continuous workflow is
essential, so this document is deliberately one narrative, not scattered files: Sections 1 to 6
build the instrument and run it, Section 7 positions the results and turns them into standing.

Target machine: Mac Studio, Apple M2 Max, 96 GB unified memory, 2 TB SSD, ~38-core GPU,
~400 GB/s bandwidth. Apple Silicon only (Metal/MPS, no CUDA). Plugged-in 24/7 workstation,
ONE project at a time at max capability. Wall-clock time is explicitly a NON-constraint:
optimize for thoroughness, corpus scale, and statistical replication, not for speed.

This is additive to the canonical docs (corpus volumes + BLACKHOLE.md + this Studio plan)
and the operational docs (README, SCALING, APPLE_SILICON, EXPERIMENTS, STATUS). Older goal and
run-report documents are consolidated in `/Users/scammermike/Downloads/PROJECT_RETROSPECTIVE_CHECKPOINTS_2026_06_28.md`.
It does not overwrite or compete with them; it is a concrete, ordered, measurable execution plan for THIS
machine. Form follows BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
No sentience, consciousness, or agency claims (north_star): the developmental framing stays
strictly measurement-based (engineered objective terms: novelty, uncertainty, learning progress,
prediction error), never experiential.

What Brain IS, stated once so nothing below overclaims: a MEASUREMENT INSTRUMENT and a RESULTS
CORPUS. It is NOT a novel mechanism (frozen encoder plus tiny trainable shell is a mature,
crowded room, Section 7.1), it is NOT the next Avalanche (that needs institutional backing and a
maintainer community a solo builder cannot spin up, Section 7.2), and it does NOT train
perception (the encoder is frozen, a grad-free unit test holds in every experiment). Its moat is
rigor and packaging on one sharp question, not compute and not a new mechanism.

## Table of contents (the full arc)

- Section 1: LIFT THE MPS BLOCKER. The throughput unlock that makes a large corpus feasible.
- Section 2: THE PERMANENT CACHED-LATENT CORPUS. The single highest-value Studio job; the
  reusable substrate every experiment and artifact below draws on.
- Section 3: THE ENCODER-SCALE EXPERIMENT. ViT-L vs ViT-H vs ViT-g; is the developmental story a
  property of the shell mechanisms or the substrate scale.
- Section 4: THE FULL CAMPAIGN AT SCALE. The whole Tier C E-series bank on real natural-video
  latents at full seeds, with real error bars.
- Section 5: WHAT STAYS DEFERRED. The honest boundary; a powerful Studio does not unblock these.
- Section 6: THE BLEEDING-EDGE EXPERIMENT BANK (EX1 to EX16). Method-richness frontier on the
  frozen-latent substrate.
- Section 7: POSITIONING AND PATH TO RECOGNITION. Where Brain honestly stands in the 2025-2026
  landscape, the leadable-or-not verdict, the wedge, the comparable-efforts gap, the
  non-experimentation moves that earn standing, and the failure modes.
- Section 8: THE BRIDGE (experiment -> artifact -> recognition). The continuous-workflow table
  that ties every experiment to the reusable artifact it produces and the recognition move that
  artifact enables.
- Section 9: FINAL EDGE PASS. The leadership mechanics, anti-claims, unfair artifacts, first
  30 days, and rare non-experiment moves that make the work recognizable outside the repo.
- Section 10: METHODOLOGY ASCENSION, THE PROOF SYSTEM. The capstone standard that makes the
  work structurally hard to dismiss: the epistemic contract, the adversarial validation layer,
  the public proof grammar (null cards, atlas, corpus card, reproduce-one-plot), the
  reproducibility gradient R0 to R5, the incumbent-resistance test, the category-ownership
  language, the embarrass-us-before-launch checklist, the 7 / 30 / 90 day plan, and the
  if-it-fails-it-still-wins path. Inline schemas, field-lists, and invalidation rules, not
  scattered files.
- Pre-Studio Scaffolding (do now on M3 Pro) + The Studio Go-Prompt. The cold-restart readiness
  verdict, the DO-NOW checklist that runs on the current 18 GB laptop (env pin, weight download,
  fully-licensed video pre-fetch, the human license-acceptance tasks, the first-artifact proof/
  tree, synthetic control fixtures, the tiny end-to-end smoke), the explicit STUDIO-ONLY boundary,
  and the single paste-able go-prompt for running the agent on the Studio.

## The framing (read first, it changes what "maximize" means)

Brain is NOT training-compute-bound. The encoder is frozen and never trained (a unit test
asserts no encoder gradient). The trainable shell is tiny (~3 to 10M params: predictor, heads,
buffer, plasticity, consolidation, neuromod). All learning iterates on CACHED latents, never on
pixels. So the Studio's value is NOT a bigger model. "Train a bigger model" misreads the
architecture and is explicitly out of scope.

The Studio buys three things, in priority order:

1. THROUGHPUT: lift the 64-frame MPS encoder block so latent caching runs on Metal, not the
   24 s/clip CPU floor. This is what makes a large corpus feasible.
2. DATASET SCALE: a large, permanent, multi-encoder cached-latent corpus over real
   natural-video. Because the encoder is frozen, cached latents never go stale: this is a
   ONE-TIME permanent asset that every future experiment reuses for free.
3. STATISTICAL REPLICATION: run the whole bank (E1 to E10 + I4 + the 14-rung ladder) at 5 to 10
   seeds on real natural-video latents, for real error bars (the repo measured SEM 0.094 at 2
   seeds falling to 0.039 at 5 seeds; sign stabilizes at 3).

Over-engineering here = corpus breadth + encoder-scale comparison (L vs H vs g) + seed
replication. Everything below is sized to that, with unlimited wall-clock as a feature.

Note on the disk: the legacy maximal-goal assumptions and the shipped `studio-1tb` profile
assume a 1 TB Studio (900 GB usable, 100 GB reserve). This machine has 2 TB. The pooled-latent corpus is so
small (see Section 2) that the 1 TB profile already has all the headroom needed; the extra
terabyte simply makes the optional dense-token caches (today deferred to E6/2.1) feasible later.
The plan uses the existing `studio-1tb` profile as-is so the safety logic is unchanged; a future
`studio-2tb` profile is a one-line numbers change, not a code change, if dense caches are wanted.

---

## Section 1: LIFT THE MPS BLOCKER (the throughput unlock)

### The blocker, measured

A 64-frame ViT-L forward produces 8192 tokens and hangs the MPS graph compiler on the M3 Pro
(APPLE_SILICON.md, "Known Metal limitation"; retrospective ledger). Real-encoder caching therefore
falls back to CPU at a MEASURED 24.3 s/clip (range 24 to 32 s/clip). `mps_safe_token_cap` in
`configs/device/mps.yaml` documents the token threshold above which an encoder forward is routed
to CPU. The shell (heads, predictors, ensembles, buffer) already runs fine on MPS; only the
large-token encoder FORWARD is affected.

### Why the Studio is expected to lift it

The M2 Max has ~38 GPU cores and 96 GB unified memory vs the M3 Pro's ~18 core / ~18 GB. The
hang is a graph-compiler / memory-pressure limit on a single very large forward, not a missing
op. More GPU cores and far more unified memory are the documented expectation for lifting it
(SCALING.md step 1, APPLE_SILICON.md "What to flip"). This is a HYPOTHESIS to verify on the
machine, not an assumption to build on. Verify before trusting MPS for large-token forwards.

### The ordered lift-and-verify procedure

Do these in order; each step has a pass/fail and a fallback. Nothing here is a code change.

1. Environment and readiness.
   - `uv pip install -e ".[dev,ann,encoder,video,apple]"`
   - `make test` (confirm green on the new box FIRST), then `make doctor`
     (python/torch/mps/disk/video/hf/encoders/cache/config readiness).
   - Confirm `devices.apple_silicon_info()` reports the M2 Max (cores, unified memory).

2. Smoke the real 64-frame ViT-L forward on Metal, smallest possible.
   - `.venv/bin/python scripts/cache_real_encoder.py device=mps +classes=2 +per_class=1`
   - This fetches real V-JEPA 2 ViT-L weights and runs ONE 64-frame forward
     ([B,64,3,256,256] -> [B,8192,1024]) on MPS. PASS = it returns latents without hanging.
     FAIL = it hangs or errors (the M3 behavior).

3. If step 2 PASSES on Metal: measure throughput vs the 24 s/clip CPU floor.
   - Time a small MPS cache (e.g. 16 to 64 clips). Record s/clip on MPS.
   - Expected: Metal should be materially faster than 24 s/clip (GPU encode is roughly an order
     of magnitude over CPU on comparable Apple Silicon; the historical CPU campaign checkpoint
     notes "~30x faster on a GPU" for the encoding step). The exact number is a MEASUREMENT to record,
     not a claim to assert. Tag it `real-encoder` once measured.
   - Cross-check fp16 vs fp32: `configs/device/mps.yaml` carries `amp: true` (fp16 autocast,
     `devices.autocast`). fp16 halves memory and is the default. Record both if time allows.

4. If step 2 still HANGS on Metal (the limit did not lift): use the documented fallbacks, in
   order of preference. The corpus build (Section 2) does NOT block on Metal: CPU encode works,
   it is just slower, and wall-clock is a non-constraint.
   - Fallback A (the safe default): keep `allow_cpu_fallback: true` and let the large encoder
     forward route to CPU via `mps_safe_token_cap`. The whole shell still runs on MPS; only the
     one-time encode is on CPU at ~24 s/clip. With unlimited wall-clock this is acceptable for
     the full corpus (Section 2 sizes the wall-clock).
   - Fallback B (reduce tokens to fit Metal): cache at a smaller frame count or resolution to
     drop below `mps_safe_token_cap`, run that on Metal, and clearly tag the cache with the
     reduced config in provenance. This is a throughput lane, not a substitute for the canonical
     64-frame cache; keep it labelled.
   - Fallback C (MLX throughput experiment, opt-in): the `apple` extra installs MLX
     (APPLE_SILICON.md). MLX can accelerate encoder inference on Apple Silicon. Treat it as a
     PROFILED-BOTTLENECK throughput experiment for the encode step ONLY, never a rewrite, never
     on the hot path of the science. Do not let it become a yak-shave (the Apple Silicon doc's explicit
     warning). Only pursue if CPU encode wall-clock is genuinely the binding constraint, which
     under "wall-clock is a non-constraint" it is not.

5. Record the verdict in a provenance-stamped report (the optimize lane is built for exactly
   this): `python scripts/studio_pipeline.py optimize --cache <cache_id>` annotates MPS vs CPU
   fallback, fp32 vs fp16, batch-size search, and cache/buffer/learner throughput. Optimization
   is NOT science, per the preserved Frontier 12 doctrine in the retrospective ledger: the
   numbers are throughput measurements, tagged as such, and never promoted to a representational claim.

### Expected outcome

A recorded MPS-vs-CPU encode throughput number with an honest tag, and a decision: if Metal
lifts the block, all caching runs on Metal (fast); if not, caching runs on CPU at the known
~24 s/clip floor (slow but fine given unlimited wall-clock). Either way the corpus gets built.
The block is a SPEED question, not a feasibility question, because cached latents are tiny and
the encode is one-time.

---

## Section 2: THE PERMANENT CACHED-LATENT CORPUS (the single highest-value Studio job)

This is the one job with the highest leverage on the whole project. Build it once; reuse forever.
Because the encoder is frozen, these latents NEVER go stale: there is no retraining that could
invalidate them. This is a permanent asset, not a run.

### The pipeline (already built, this is config + scripts, not code)

`scripts/studio_pipeline.py` is the single acquisition surface. The day-one sequence
(from SCALING.md, expanded):

```
python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900
# REVIEW runs/studio_pipeline/latest/license_ledger.md (resolve manual/blocked sources FIRST)
python scripts/studio_pipeline.py acquire  --plan runs/studio_pipeline/latest/plan.json \
    --execute --budget-gb 900 --accept-license      # REAL downloads, gated + budgeted
python scripts/studio_pipeline.py validate --plan runs/studio_pipeline/latest/plan.json
python scripts/studio_pipeline.py cache    --plan runs/studio_pipeline/latest/plan.json --execute
```

The planner is a breadth-first knapsack over `registry/datasets.yaml`. It prefers breadth (action
+ egocentric + instructional metadata + synthetic controls + local import) over one giant dataset.
Full Ego4D / Ego-Exo4D are `status: deferred` and the planner refuses them by default regardless
of acknowledgement.

### Source selection (from registry/datasets.yaml, honest status tags)

Acquire breadth-first. Sizes are the registry's honest GB estimates; subset sizes are the
`recommended_subsets` the planner prefers.

| Source | slug | status | raw GB (subset) | recommended clip subsets | provenance kind |
|---|---|---|---|---|---|
| Something-Something V2 | ssv2 | manual (Qualcomm/20BN terms) | ~20 | 1k / 5k / 20k | natural-video |
| EPIC-KITCHENS-100 (subset) | epic_kitchens_subset | available (CC BY-NC 4.0) | ~80 | 1k / 5k / 20k | natural-video |
| Kinetics-700 (subset) | kinetics700_subset | metadata-only (video via licensed mirror) | ~60 | 2k / 10k / 50k | natural-video |
| Ego4D (subset) | ego4d_subset | manual (signed Ego4D license) | ~200 | 2k / 10k | natural-video |
| Synthetic control corpus | synthetic_controls | available (generated locally) | ~0.5 | 32 / 64 / 128 (scale up) | structured-synthetic |
| Local class-folder import | local_import | available (user-provided) | user-sized | user | natural-video |
| HowTo100M / AudioSet | howto100m_meta / audioset_meta | metadata-only | ~6 / ~1 | (captions/labels only) | metadata-only |

What to actually pull, in order:

1. EPIC-KITCHENS-100 subset (status `available`, CC BY-NC 4.0, no signed terms): the egocentric
   anchor. Start at the 5k subset, then 20k. This is the cleanest first real natural-video
   source because it needs no manual access.
2. Synthetic control corpus (`available`, generated, zero license risk): scale the 9 control
   families (moving / permanence / occlusion / relation / containment / noisy-TV / navigation /
   class-incremental / domain-incremental) up from 128 to the thousands. These are CONTROLS and
   regression fixtures, not science claims, but they are essential for the linear-probe gate, the
   noisy-TV guard, and E4 (tagged structured-synthetic).
3. SSv2 (status `manual`): AFTER the user completes the Qualcomm/20BN registration and accepts
   terms. The action-motion anchor and the cleanest labeled-motion source. Then the planner will
   select it (the profile allows manual auth and the user passes `--accept-license`).
4. Ego4D subset (status `manual`): AFTER the user signs the Ego4D license and has AWS creds. The
   200 GB curated subset, NEVER the full multi-TB corpus.
5. Kinetics-700 subset: metadata-only by default (ID/label CSVs are open). Bulk video needs a
   licensed academic mirror (no scraping). Pull the CSVs now; treat video as a later manual step
   through a licensed mirror.
6. HowTo100M / AudioSet metadata: caption/narration and audio-event CSVs for the multimodal
   schema (rung 10 language-mediated, future). Metadata-only; no bulk video scraping.

### Latent store sizing (this is why the corpus is cheap and permanent)

Pooled latents are TINY. From `substrate/storage.py` (pooled = latents + duplicated keys, the
load-bearing term), float32:

- ViT-L (embed_dim 1024): ~8 KB/clip pooled. 10k clips ~= 78 MB. 100k clips ~= 0.8 GB.
- ViT-H (embed_dim 1280): ~10 KB/clip pooled. 100k clips ~= 1.0 GB.
- ViT-g (embed_dim 1408): ~11 KB/clip pooled. 100k clips ~= 1.1 GB.

So a 100k-clip natural-video corpus cached through ALL THREE encoders (L + H + g) is roughly
3 GB of pooled latents total. Even 1 million clips across three encoders is ~30 GB. The pooled
latent store is laptop-cheap and Studio-trivial; the cost is the RAW VIDEO on disk (tens to a few
hundred GB per source) and the ENCODE TIME, not the latent footprint.

Target the corpus at the union of recommended subsets:
- EPIC 20k + SSv2 20k (once unlocked) + Ego4D 10k (once unlocked) + Kinetics 50k (if mirrored) +
  synthetic controls (thousands) ~= ~100k natural-video clips plus controls.
- Pooled latent store, all three encoders: a few GB. Trivial on 2 TB. Raw video: a few hundred
  GB, comfortably inside the 900 GB usable budget.

DENSE latents are a different story and stay DEFERRED: dense ViT-L is ~32 MB/clip (8192 tokens),
so 10k dense clips ~= 313 GB. Dense is the E6 / V-JEPA 2.1 path (Section 5) and is not cached now.
Pooled is the corpus; dense waits for 2.1 and a clear reason.

### Multi-encoder caching (the substrate-scale axis)

The `cache` stage takes `--encoder`. Run it three times over the SAME validated raw corpus, once
per canonical frozen encoder, writing three parallel pooled latent stores:

```
python scripts/studio_pipeline.py cache --plan <plan> --execute --encoder vjepa2_vitl_fpc64_256
python scripts/studio_pipeline.py cache --plan <plan> --execute --encoder vjepa2_vith
python scripts/studio_pipeline.py cache --plan <plan> --execute --encoder vjepa2_vitg
```

All three are real, verified HF ids (SCALING.md, probed 2026-06): `vjepa2-vitl-fpc64-256` (1024),
`vjepa2-vith-fpc64-256` (1280), `vjepa2-vitg-fpc64-384` (1408). Each cache records `backend`
(`vjepa_hf`), encoder id, and provenance, so a reader can never confuse substrate scales or
mistake one for another. The shell reads `latent_dim` off the selected encoder config, so the
predictor and heads resize automatically with NO shell code change. This multi-encoder cache is
what makes the encoder-scale experiment (Section 3) possible at zero marginal science cost: it is
just three cache builds over one corpus.

### Encode wall-clock (sized, since it is the real cost)

At the CPU floor (24 s/clip) a 100k-clip corpus is ~28 days per encoder of pure CPU encode, ~83
days for all three. That is fine under "wall-clock is a non-constraint" but it is the argument for
lifting the MPS block (Section 1): if Metal encode is even 10x faster, the same corpus is ~3 days
per encoder. Build EPIC 5k first (a small, fast, fully-licensed slice), validate the whole
pipeline end to end on real natural-video, THEN scale to the full corpus. Caching is resumable
(per-clip manifest, skip-present), so the long encode can run unattended across days and survive
interruption.

### Verification gates on the corpus (do not trust a bad cache)

- `validate` stage: source path, class folders, empty-class detection, duplicate/short-clip
  handling, resolution/FPS/duration stats, label map persisted, corrupt files isolated.
- `cache` validation (`make cache-list`, cache_tools): shape/dtype/length/label checks, random-row
  read test, backend tag correct, stale/corrupt detection. Fails LOUDLY if the natural-video /
  structured-synthetic / provisional tag is wrong.
- Linear-probe gate (the spine): before trusting any cache for science, confirm visual-class info
  is linearly decodable from the latents (the repo measured acc 1.000 vs chance 0.167 at n=96 on
  real ViT-L latents). A cache that does not pass the probe is a data/representation problem, not
  a learning result.

---

## Section 3: THE ENCODER-SCALE EXPERIMENT (the over-engineering centerpiece)

### The question (one sentence)

Does a bigger frozen perception change WHICH developmental mechanisms help, by re-running the
exact same shell and the exact same experiment bank over ViT-L vs ViT-H vs ViT-g cached latents
and comparing the per-mechanism outcomes?

### Why this is the right over-engineering angle

It is PURE INFERENCE: the encoder is frozen, nothing trains on pixels, the shell is tiny. It costs
only the three cache builds (Section 2) plus re-running an already-built bank. It is exactly the
"throughput + dataset scale + replication" the Studio is for, and it answers a real scientific
question the M3 Pro cannot: is the developmental story a property of the SHELL mechanisms or of
the SUBSTRATE scale. Unlimited wall-clock makes it free to run all three arms at full seeds.

### Design (concrete, doctrine-respecting)

- Independent variable: frozen encoder scale (ViT-L 1024 / ViT-H 1280 / ViT-g 1408), held frozen,
  inference-only. The encoder NEVER trains in any arm (the grad-free invariant holds across all).
- Controlled: the SAME shell (predictor, heads, buffer, plasticity, consolidation, neuromod), the
  SAME experiments, the SAME natural-video clips, the SAME seeds. Only the latent source differs.
  The shell auto-resizes `latent_dim` per encoder config, so this is a config flip, not a rewrite.
- Dependent variables: for each mechanism in the bank (E2 replay, E3 staged plasticity, E4
  uncertainty gating, the E1 frontier, I4 alternatives), does the SIGN and SIZE of the effect
  hold across L / H / g? Report per-encoder frontier AUC, BWT, adaptation speed, and the
  per-experiment null verdict.
- Null hypothesis (per the doctrine contract, stated up front): substrate scale does NOT change
  which mechanisms help; every mechanism that wins (or ties) on ViT-L wins (or ties) the same way
  on ViT-H and ViT-g, within the seed spread. A REJECTION (a mechanism that helps on g but not L,
  or flips sign) is the interesting positive result and maps to negative-result taxonomy entry 3
  (frozen latent lacks/gains the info) or 9 (separating representational from compute claims).
- Linear-probe gate per encoder: before comparing mechanism outcomes, confirm the target is
  linearly decodable from EACH encoder's latents. If a variable is decodable from g but not L,
  that is itself the finding and must precede any mechanism claim.
- Determinism gate: run `diagnostics/determinism` on each encoder's cache first; trust no
  cross-encoder delta smaller than the Metal spread (the repo notes Metal ~50% byte-identical at
  temp 0; CPU is bit-identical and is the tolerance baseline).

### Protocol

1. Build the three pooled caches (Section 2) over the same validated natural-video corpus.
2. Per encoder: linear-probe gate, determinism gate, then E1 the gate (must pass before anything
   downstream is trusted).
3. Per encoder: run the full Tier C bank at full seeds (Section 4).
4. Assemble a 3-column comparison table (L / H / g) for every mechanism's metric and null verdict.
   Use the report scaffolds (Frontier 19): seed summaries, confidence intervals, effect sizes,
   adaptation-retention frontiers, per-encoder null-registry rows.
5. Honest tag: results are `real-encoder` and (once on natural video) natural-video; the
   cross-encoder comparison is a representational claim, kept separate from any throughput claim.

This is the headline Studio-unlocked science: a developmental-mechanism-by-substrate-scale matrix
that the laptop literally cannot produce (it cannot encode at scale, and cannot hold three
encoders' worth of replication).

---

## Section 4: THE FULL CAMPAIGN AT SCALE (real latents, real error bars)

### What runs

The whole Tier C bank on REAL natural-video latents at 5 to 10 seeds. From `campaign/run_queue.yaml`,
the enabled Tier C legs and their full-scale run-units:

| Leg | Experiment | run_units (full) | depends on |
|---|---|---|---|
| track01_e1_gate | E1 (GATE) | 30 | (none) |
| track02_e2_replay | E2 replay | 20 | E1 |
| track03_e3_plasticity | E3 staged plasticity | 15 | E1 |
| track04_e4_neuromod | E4 uncertainty gating | 30 | E1 |
| track05_level5_integration | Level-5 headline (E2+E3+E4) | 30 | E2, E3, E4 |
| track06_e7_sparse | E7 sparse/modular | 30 | E1, E3 |
| track07_e8_dendritic | E8 dendritic | 15 | E1 |
| track08_e9_local | E9 local learning | 9 | E1 |
| track08_i4_backprop_alts | I4 alternatives | 9 | E1 |
| track09_e6_relational | E6 relational (pooled fallback) | 10 | E1 |

Tier C total = 198 full-scale run-units (the queue's full factorial x 5 seeds). Tier E (E5
curiosity env, 10 units) and Tier R (E10 capstone + POET + cultural, 9 units) stay disabled
(Section 5), so the full queue is 217 run-units; the runnable-now C column is 198.

### Seed budget (real error bars)

The repo measured the E1 protected-vs-naive gap SEM falling 0.094 (S=2) -> 0.039 (S=5), with the
sign stabilizing at S=3 (preserved in the retrospective ledger, study 11B). Recommended seeds: headline 5, ranking 5,
sanity 3. Under unlimited wall-clock on the Studio, push the HEADLINE legs (E1, the Level-5
integration, E2, E4) to 10 seeds for tighter intervals; keep 5 for ranking legs and 3 for sanity
legs. Always run `diagnostics/determinism` FIRST to set tolerances, and trust no cross-condition
delta inside the Metal spread.

### The Level-5 headline protocol (the combined result)

The Level-5 headline is E2+E3+E4 combined: the protected arm wires replay (E2) + EWC consolidation
+ staged plasticity (E3) + uncertainty gating (E4) against the naive baseline
(`campaign/legs/track05/track05_level5_integration.yaml`). The build-order DAG is mandatory:

1. E1 must pass the gate (naive forgets, protected retains, both learn last task). On real ViT-L
   latents the repo already showed naive BWT -1.000 (collapse to chance 0.167) vs replay+EWC BWT
   0.000 (perfect retention) at n=96 structured-synthetic content; the Studio re-runs this on real
   NATURAL-video content at full seeds to remove the synthetic-content caveat.
2. E2, E3, E4 each pass their own null and (for E4) its noisy-TV test BEFORE its signal is trusted
   as a trigger in E3 or a prioritizer in E2. E4's uncertainty gating must demonstrably ignore the
   noisy-TV (ensemble disagreement collapses on irreducible noise) while point prediction-error
   chases it.
3. ONLY THEN run track05: the protected arm combines all three. Full axes are stream separation
   {0.3, 0.5, 0.7} x head depth {1, 2} x 5 seeds (full_seeds 0 to 4 in the leg). Push to more
   seeds for the headline. Report the adaptation-retention frontier and frontier AUC (the
   program's central metric) for protected vs naive, with confidence intervals.

Mechanism gates that must hold (from the bank's stated nulls, to be re-tested on real latents):
- E2: replay beats no-replay; prioritized may TIE random (the corpus's predicted E2 half-null,
  already confirmed on real ViT-L latents). Report both honestly.
- E3: staged plasticity vs constant LR AND tuned cosine decay. On a frozen substrate the
  "just an LR trick" null may not be rejectable (the repo's prediction); report the Fisher trace
  signature either way.
- E4: ensemble/distributional gating ignores noisy-TV; point-error gating does not. The probabilistic
  head's calibration (ECE) is reported.

### Per-encoder x full-bank x full-seeds

Cross Section 3 with Section 4: run the full Tier C bank at full seeds for EACH of the three
encoders. That is 198 run-units x 3 encoders, all pooled, all inference, all replicated. The
pooled caches are a few GB; the cost is purely the shell runs, which are seconds-to-minutes each
on MPS. This is the maximal honest use of an unlimited-wall-clock Studio: a fully-replicated,
multi-substrate, real-natural-video developmental campaign.

### Reporting

Use the prebuilt report scaffolds (Frontier 19): seed summaries, confidence intervals, effect
sizes, adaptation-retention frontiers, the null-result registry (every null mapped to one of the
10 taxonomy entries), the cache/source table, the gate table, the speed table, the data-card
appendix. Every number tagged by provenance (natural-video > real-encoder > structured-synthetic
> provisional). Promote `provisional` grid results to `real`/`natural-video` only after they
re-run green on the real natural-video caches. The metacognition / north-star safety scanner gates
every rendered report and refuses any affirmative sentience/consciousness/agency claim.

---

## Section 5: WHAT STAYS DEFERRED (and why, so the plan stays honest)

Deferred means deferred. These do NOT become runnable just because a powerful Studio arrived.

### E6 dense relational map (needs V-JEPA 2.1 dense weights, NOT on HF)

E6 factorizes object/relation structure from DENSE per-patch tokens. The pooled ViT-2 encoders
emit one vector per clip, so E6 "has nothing to factorize" on them (the bank's own note). E6 needs
the V-JEPA 2.1 dense encoders (`dense: true`, per-patch tokens). As of 2026-06 V-JEPA 2.1 dense is
NOT published on HuggingFace under any verified id, so `configs/encoder/vjepa21_*` are placeholders
(`available: false`) and `registry/models.yaml` carries the 2.1 dense row as `available: false`,
`result_tag: provisional`. E6 stays `2.1-only` / deferred until 2.1 ships and a real dense cache
verifies. The `track09_e6_relational` leg runs only its pooled fallback (frozen-random), which is a
scaffold, not the dense science. Also note: dense caches are ~32 MB/clip (ViT-L), so a 10k dense
corpus is ~313 GB; even when 2.1 ships, dense caching is a deliberate, budgeted, 2 TB-justified
decision, not an automatic part of the corpus.

### E5 env-rollout and E10 capstone (need a procedural environment + rented CUDA, Tier R)

E5's curiosity-as-self-curriculum ROLLOUT variant and the E10 open-ended capstone both need an
interactive environment to ACT in, not just a bigger GPU. They are blocked on an env adapter, not
on compute. The Studio is Apple Silicon (no CUDA) and cannot cheaply cover environment rollouts;
the rented-CUDA path (`device=cuda`) exists ONLY for these Tier R legs. In the queue,
`track10_e5_curiosity` (Tier E), `track11_e10_autotelic`, `track11_poet_envgen`, and
`track11_cultural_accumulation` (Tier R) are all `enabled: false`. They flip to enabled only after
an env adapter is provided AND a CUDA box is rented. The DATA-SELECTION variant of E5 (curiosity
as active curriculum over cached latents) IS runnable now on the Studio and is folded into the
curriculum-engine work (it picks the learnable-but-not-mastered family and rejects the aleatoric
noisy-TV); only the live-rollout variant is deferred.

### Manual-access datasets (need the user to complete terms)

SSv2 (Qualcomm/20BN registration + accepted terms) and Ego4D subset (signed Ego4D license + AWS
creds) are `status: manual`. The planner will not auto-select them until the user completes access
AND passes `--accept-license`. Full Ego4D and Ego-Exo4D are `status: deferred` (multi-TB, beyond
even a 2 TB local disk on the full release) and are NEVER planned by default. Kinetics-700 bulk
video needs a licensed academic mirror (no scraping); only its open ID/label CSVs are pulled by
default. LAION is `status: blocked` (the 5B release was withdrawn for a safety-reviewed
re-release); it is auxiliary, image-text, diagnostics-only, and not a video source.

### Auxiliary / approximate encoders (never replace canonical)

The optional rows in `registry/models.yaml` (VideoMAE V2, DINOv2 comparison; distilled students;
int8 quantized inference) are `available: false` and `replaces_canonical: false` (a hard
invariant). They are auxiliary baselines, speed approximations, or throughput experiments, tagged
approximate/auxiliary, and may run in the OPTIMIZE lane only. They never stand in for the frozen
V-JEPA substrate in any science result.

---

## Summary: the ordered Studio day-one to day-N sequence

```
# Day 0: readiness
uv venv --python 3.12 .venv
uv pip install -e ".[dev,ann,encoder,video,apple]"
make test            # green FIRST
make doctor          # readiness
make diag            # determinism + diagnostics; SET tolerances

# Day 0: lift-the-block verification (Section 1)
.venv/bin/python scripts/cache_real_encoder.py device=mps +classes=2 +per_class=1   # 64-frame ViT-L on Metal?
python scripts/studio_pipeline.py optimize --cache <smoke_cache>                     # record MPS vs CPU throughput

# Day 1+: the permanent corpus (Section 2), breadth-first, fully-licensed first
python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900
#   REVIEW runs/studio_pipeline/latest/license_ledger.md
python scripts/studio_pipeline.py acquire  --plan <plan> --execute --budget-gb 900 --accept-license
python scripts/studio_pipeline.py validate --plan <plan>
python scripts/studio_pipeline.py cache    --plan <plan> --execute --encoder vjepa2_vitl_fpc64_256
python scripts/studio_pipeline.py cache    --plan <plan> --execute --encoder vjepa2_vith
python scripts/studio_pipeline.py cache    --plan <plan> --execute --encoder vjepa2_vitg

# Day N: the full campaign at scale (Sections 3 + 4), per encoder, full seeds
python scripts/studio_pipeline.py run --gated --tiers C --full --profile studio-1tb
python scripts/studio_pipeline.py report

# Deferred (Section 5): E6 (waits for 2.1 dense on HF); E5 rollout + E10 (env adapter + rented CUDA, Tier R)
```

What starts automatically after data lands: the gated conveyor (validate source -> cache -> validate
cache -> linear probe -> noisy-TV/diagnostics -> E1 smoke -> Tier C toy -> Tier C full ONLY if gates
pass -> report). Gates are kill switches that STOP the run, not warnings. Nothing chains blindly.

What is real vs deferred: real now on the Studio = MPS-block verification, the multi-encoder pooled
corpus over real natural-video, the encoder-scale comparison, the full Tier C bank at full seeds.
Deferred = E6 dense (no 2.1 weights on HF), E5 rollout + E10 (env + rented CUDA), manual datasets
until the user completes terms, dense caches until there is a reason and 2.1.

---

## Section 6: THE BLEEDING-EDGE EXPERIMENT BANK (EX-series, the method-richness frontier)

### Why an EX-series and not bigger training

The frontier for THIS architecture is not a bigger model: the encoder is frozen (the grad-free unit
test holds in every experiment below, no exception), the shell stays tiny (3 to 10M params), and all
learning iterates on cached pooled latents. So the bleeding edge here is METHOD richness, corpus and
encoder scale, and replication, exactly what an unlimited-wall-clock 96 GB Studio buys. This section
adds a numbered bank of frontier experiments that adapt published ideas to the frozen-latent setting.
Each is built on the same cached-latent substrate (Section 2) and the same tiny shell, so each is a
config plus a small shell module, never a retrain of perception. Each carries the full doctrine
contract: an explicit null hypothesis, a baseline, a metric, the negative-result taxonomy entry a null
maps to, a tier, and the capacity-ladder rung it advances. Every result is provenance-tagged
(natural-video > real-encoder > structured-synthetic > provisional) and every rendered report passes
north_star.scan_text (engineering vocabulary only: novelty, uncertainty, learning progress, prediction
error). Many of these already exist as registry slugs at status studio-later or prototype-local or
blocked (registry/paradigms.yaml); the bank below promotes them to concrete, doctrine-shaped runs and
adds new ones the registry does not yet name.

### Tier legend (feasibility on this machine, restated for the bank)

- Tier C (run-now-local): pooled cached latents + tiny shell, MPS or CPU, fits in 96 GB trivially,
  runnable today. This is where the bleeding edge actually lives for Brain.
- Tier D (run-now-local, dense-pooled hybrid): runs on pooled latents now as a precursor; the FULL
  version wants V-JEPA 2.1 dense per-patch tokens (not on HF as of 2026-06), so the dense arm is
  deferred and clearly labelled. The pooled precursor is real and runnable.
- Tier R (deferred): needs a procedural environment to ACT in plus rented CUDA (device=cuda). The
  Studio is Apple Silicon, no CUDA; these flip to enabled only after an env adapter AND a rented box.
  Deferred means deferred; a powerful Studio does not unblock them.

A note that bounds every EX below: the linear-probe gate is the spine. Any EX whose mechanism needs a
variable X (controllability, object identity, action, intervention effect, cross-modal correspondence)
MUST first prove X is linearly decodable from the frozen latent (diagnostics/linear_probe). If X is
not decodable, the EX cannot make a representational claim; the finding is taxonomy entry 3 (frozen
latent lacks the info) and the mechanism is not at fault. This precedes every mechanism run, no
exception.

---

### EX1: Generative latent replay vs stored-buffer replay (the dreaming experiment)

- Capacity rung advanced: 3 (episodic memory) and 5 (consolidation). Promotes paradigm
  `generative_latent_replay` (status studio-later) to a concrete run.
- Bleeding-edge hypothesis: a tiny generator over the FROZEN pooled latent space (a small latent VAE,
  or a latent diffusion / flow-matching prior, 1 to 3M params) can synthesize pseudo-experience for
  rehearsal, replacing the stored exemplar buffer (Shin et al. deep generative replay, adapted to a
  frozen-encoder latent space where the generator never has to reconstruct pixels, only 1024-dim
  pooled vectors). Generative replay is cheap here precisely because pooled latents are tiny.
- Concrete method: train the latent generator per task on that task's cached pooled latents; during
  the continual stream, sample pseudo-latents from the generator (optionally class-conditioned on the
  task head) and interleave them with the new-task latents into the SAME predictor/head shell the E2
  buffer feeds. Hold the generator OUTSIDE the grad-free encoder boundary (it generates latents, it
  never touches the encoder). Match total replay volume to the E2 stored buffer so the comparison is
  storage-for-prior, not volume.
- Metric + baseline: BWT and the adaptation-retention frontier AUC (metrics/frontier.py), generative
  replay vs E2 stored-buffer replay at matched replay budget; report generator fidelity as the
  linear-probe accuracy of a probe trained on generated latents and tested on real latents (the
  distribution-gap check).
- Explicit null hypothesis: generated replay does NOT match stored-buffer replay on BWT at matched
  budget; the generator's distribution gap (probe transfer below a stated margin) costs retention. A
  REJECTION (generative ties or beats stored at lower storage) is the positive result.
- Negative-result category: 4 (generator/prior too weak, a capacity ablation) if fidelity is the
  bottleneck; 8 (only works combined with a small stored anchor set, the hybrid) if pure-generative
  fails but generator-plus-anchor wins.
- Tier / feasibility: Tier C. A latent VAE over 1024/1280/1408-dim vectors trains in minutes on MPS;
  the whole generator plus shell is well under 1 GB. Run all three encoders (Section 3 cross).

### EX2: Model-based planning in latent space (Dreamer/MuZero-style on frozen latents)

- Capacity rung advanced: 8 (causal and affordance sketch). Promotes paradigm `latent_rollout_mpc`
  (status studio-later) on the runnable synthetic-control families.
- Bleeding-edge hypothesis: rolling the action-conditioned latent predictor (shell/predictor.py AC
  variant) forward for a short horizon supports goal-conditioned planning that a flat reactive head
  cannot do (Hafner et al. Dreamer / Schrittwieser et al. MuZero, adapted: the learned dynamics live
  entirely in the frozen pooled latent space, no pixel decoder, no value model on images).
- Concrete method: on the synthetic control families that HAVE a defined action and transition
  (navigation, containment, the action-conditioned transitions behind rung 8), learn a latent
  dynamics model g(z, a) -> z' as a tiny shell module; plan by short-horizon rollout (MPC stub:
  sample action sequences, roll forward in latent space, pick the sequence whose terminal latent is
  nearest the goal latent). Compare against a flat reactive head trained on the same transitions.
- Metric + baseline: goal-reaching success rate and goal-latent distance reduction under the plan,
  planner vs flat reactive head; report calibrated short-horizon rollout error first (the gate: if
  rollouts are not predictive, planning is not licensed).
- Explicit null hypothesis: the learned latent dynamics model does NOT enable planning the flat shell
  cannot do; either rollout error is too high to plan against (gate fails) or the flat head ties the
  planner on the synthetic families.
- Negative-result category: 7 (needs embodiment/action; the data-selection synthetic families have
  actions but live rollout is Tier R) for the live-control case; 4 (predictor too weak to roll out)
  if the dynamics model cannot hold horizon.
- Tier / feasibility: Tier C for the SYNTHETIC-control planning (cached action-conditioned
  transitions, runnable now). The LIVE-environment planning variant (act in a real env, replan
  online) is Tier R (env adapter + rented CUDA), explicitly deferred, same boundary as E5 rollout and
  E10. Mark the synthetic arm run-now, the live arm deferred.

### EX3: Test-time training / adaptation on frozen latents

- Capacity rung advanced: 4 (plastic adaptation). Promotes paradigm `test_time_adaptation` (status
  prototype-local) to scale.
- Bleeding-edge hypothesis: a small fast-weight overlay can adapt the shell at INFERENCE from a few
  unlabeled latents of a shifted domain, with no labels and no touch to the slow weights (Sun et al.
  test-time training, with a self-supervised proxy: minimize the shell's own latent-prediction error
  on the unlabeled shifted clips).
- Concrete method: freeze the trained slow shell; at test time, take K unlabeled latents from the
  shifted domain, do a few steps of the self-supervised proxy (next-latent prediction error) to fit a
  small fast-weight overlay, then evaluate; revert the overlay to confirm the base is restored.
- Metric + baseline: error on the shifted domain after TTA vs a frozen-head baseline, and forgetting
  of the base domain (must stay bounded; revert must restore base exactly).
- Explicit null hypothesis: TTA does NOT beat the frozen head at matched parameters; the unlabeled
  proxy carries no usable adaptation signal, or adapting on it corrupts the base.
- Negative-result category: 3 (the shift is not decodable from the latent, so there is nothing to
  adapt to) or 4 (overlay too small / proxy mis-specified).
- Tier / feasibility: Tier C. Few-step inference-time adaptation on pooled latents is seconds on MPS.

### EX4: Fast-weight / hypernetwork shells, in-context plasticity vs gradient plasticity

- Capacity rung advanced: 4 (plastic adaptation) and 12 (skill library, via emitted task-specialized
  shells). Adjacent to `test_time_adaptation` but a distinct mechanism (a hypernet, not an overlay).
- Bleeding-edge hypothesis: a hypernetwork conditioned on a short context of latents can EMIT the
  predictor/head weights for that context in a single forward pass (Ba et al. fast weights /
  Ha et al. hypernetworks), achieving in-context plasticity that competes with several steps of
  gradient adaptation, at a fraction of the inference cost.
- Concrete method: train a tiny hypernet h(context_latents) -> shell_weights on the task stream; at
  evaluation, feed a few latents of the new task as context, take the emitted shell, evaluate with NO
  gradient steps. Compare head-to-head against EX3 gradient TTA and against MAML-style meta-init
  (EX7) at matched adaptation budget.
- Metric + baseline: adaptation speed (steps-to-threshold, where the hypernet target is ZERO steps)
  and accuracy on held-out tasks, hypernet vs gradient-TTA vs static head.
- Explicit null hypothesis: the hypernet's in-context plasticity does NOT match gradient plasticity;
  emitted weights underperform a few gradient steps at matched budget, OR the hypernet collapses to a
  context-independent average shell (no in-context signal).
- Negative-result category: 4 (hypernet capacity / context too weak) or 8 (only helps combined with a
  small gradient correction, the hybrid amortized-then-refine case).
- Tier / feasibility: Tier C. The hypernet is small (it emits a 3 to 10M shell, but is itself tiny);
  fits in 96 GB with room to spare.

### EX5: Local learning rules at scale (extend I4: PC, eq-prop, forward-forward, target-prop)

- Capacity rung advanced: 4 (plastic adaptation) plus the locality / continual-learning tradeoff axis.
  Promotes paradigms `predictive_coding` and `forward_forward` (status prototype-local) to the full
  natural-video, multi-encoder scale, alongside equilibrium prop and target-prop from
  learning/alternatives/.
- Bleeding-edge hypothesis: layerwise local rules (predictive coding per Rao and Ballard / Millidge
  et al., equilibrium propagation per Scellier and Bengio, forward-forward per Hinton, target-prop)
  approximate backprop on real frozen latents, and the gap-vs-depth and gap-vs-locality curves reveal
  which rule gives the best continual-learning / locality tradeoff (local rules may forget less
  because updates are local and do not perturb a global graph).
- Concrete method: extend the existing I4 harness from the toy gap study to the full multi-encoder
  natural-video probes; sweep predictor depth (the documented PC-gap-vs-depth signature) and measure
  both the accuracy gap to the backprop ceiling AND the continual-learning retention (BWT) of each
  rule on the domain-incremental stream, at matched head, data, seed, and budget.
- Metric + baseline: accuracy gap to backprop (the I4 ceiling), activation-memory and locality of
  each rule, and BWT on the continual stream; backprop is the ceiling baseline.
- Explicit null hypothesis: no local rule comes within the stated accuracy margin of backprop at
  matched budget AND none offers a continual-learning or memory advantage that justifies the gap
  (the strong I4 null, extended to retention).
- Negative-result category: 9 (separate the representational claim from the compute/locality claim,
  the hardware-incompatible-or-not distinction the taxonomy reserves for exactly this) or 4 (rule too
  weak at depth).
- Tier / feasibility: Tier C. Local rules trade activation memory for compute and run comfortably on
  pooled latents in 96 GB; the depth sweep is wall-clock, which is free.

### EX6: Active-inference / free-energy shell objective vs the standard predictor loss

- Capacity rung advanced: 6 (curiosity) and 9 (self-monitoring), via an objective that unifies
  prediction and uncertainty. A new objective the registry does not yet name.
- Bleeding-edge hypothesis: replacing the standard next-latent prediction loss with a free-energy /
  expected-free-energy shell objective (Friston-style active inference, adapted to frozen latents:
  the shell minimizes prediction error PLUS a complexity/uncertainty term, and SELECTS the next clip
  by expected information gain) yields better-calibrated uncertainty and better data selection than
  the standard predictor loss with a bolted-on curiosity head.
- Concrete method: implement the free-energy objective as a shell loss (prediction error + KL
  complexity term over the predictor's distributional head, shell/heads.py gaussian head); use
  expected free energy to pick the next clip in the curriculum engine; compare against the standard
  predictor loss with learning-progress selection (rung 6 baseline) on the same control families.
- Metric + baseline: calibration (ECE), learnable-region coverage vs noise time-share, and noisy-TV
  rejection rate, free-energy objective vs standard-predictor-plus-learning-progress.
- Explicit null hypothesis: the free-energy objective does NOT improve calibration or selection over
  the standard predictor loss; the complexity term is just a regularizer with no selection benefit,
  OR it chases the noisy-TV like point error (failing the epistemic-aleatoric split).
- Negative-result category: 8 (only helps combined with the ensemble disagreement signal) or 1
  (the biology/active-inference mapping adds complexity with no measured benefit).
- Tier / feasibility: Tier C. An objective swap on the existing shell; the noisy-TV guard
  (diagnostics/noisy_tv.py) gates it exactly as it gates E4.

### EX7: Meta-learning across the task stream (MAML / Reptile on the shell)

- Capacity rung advanced: 13 (meta-learning) and 12 (skill library). Promotes paradigm
  `meta_learned_init` (status studio-later) to a concrete run.
- Bleeding-edge hypothesis: a meta-learned shell initialization (MAML per Finn et al., or the
  first-order Reptile) adapts to each new task in the stream in fewer steps than a generic init,
  turning the long task stream itself into the meta-training distribution.
- Concrete method: outer-loop meta-train the shell init across the task stream (each task is an
  inner-loop adaptation episode on its cached latents); evaluate adaptation speed on held-out tasks
  vs a generic init and vs the EX4 hypernet. First-order Reptile is the cheap default; full MAML
  second-order is a wall-clock-free upgrade given the Studio.
- Metric + baseline: steps-to-threshold accuracy on held-out tasks, meta-init vs random init vs
  hypernet; report forward transfer (FWT) as the secondary metric.
- Explicit null hypothesis: the meta-learned init does NOT reduce adaptation steps vs a control init;
  the task family is too homogeneous for a useful shared init, or the shell is too small to benefit.
- Negative-result category: 5 (tasks too similar, no meta-structure to learn) or 4 (shell capacity).
- Tier / feasibility: Tier C. Inner/outer loops on pooled latents are cheap; second-order MAML is the
  only memory-heavier arm and still fits in 96 GB at the shell's tiny size.

### EX8: Intrinsic-motivation curriculum head-to-head (LP vs RND vs disagreement vs info-gain)

- Capacity rung advanced: 6 (curiosity). Extends the present `learning_progress_sampling` (status
  implement-now) to a four-way bake-off.
- Bleeding-edge hypothesis: among the canonical intrinsic-motivation signals, learning-progress
  (Oudeyer / Schmidhuber, already present), random network distillation (Burda et al. RND), ensemble
  disagreement (epistemic, the E4 signal), and explicit information gain, only the signals that
  measure REDUCIBLE uncertainty pick the learnable region and reject the noisy-TV; the rest chase
  irreducible noise.
- Concrete method: run all four selection signals through the same curriculum engine over the control
  families (which include an explicit noisy-TV family); measure how each spends its budget across the
  learnable-but-not-mastered family vs the aleatoric noisy-TV.
- Metric + baseline: learnable-region coverage per budget, noisy-TV time-share (must be near zero for
  a passing signal), and noisy-TV rejection rate; uniform/random sampling is the baseline.
- Explicit null hypothesis: prediction-error and RND curiosity equal random on a learnable-vs-noisy
  budget, and only learning-progress and disagreement survive (the rung-6 prediction); a SHARPER null
  is that even learning-progress ties uniform on a homogeneous stream (no curriculum benefit).
- Negative-result category: 10 (a signal that chases the noisy-TV is conceptually mis-specified for
  reducible-uncertainty selection) or 5 (stream too uniform for any curriculum to help).
- Tier / feasibility: Tier C. Four selection heads over cached latents; the noisy-TV guard is the
  decisive gate. This is one of the cheapest and most decisive EX in the bank.

### EX9: Object-centric / slot attention over pooled latents (an E6 precursor, NOT 2.1-dense)

- Capacity rung advanced: 2 (object and event permanence) and 7 (abstraction). A POOLED precursor to
  paradigm `object_slot_probe` (status blocked on dense) and to E6, runnable now without 2.1.
- Bleeding-edge hypothesis: even WITHOUT dense per-patch tokens, a slot-attention head (Locatello et
  al.) applied over a SHORT SEQUENCE of pooled per-clip latents (treating the temporal sequence as
  the set the slots compete over) can recover coarse relational / event structure that a flat head
  cannot, giving a bound on how much object-centric structure pooled latents afford before 2.1 dense
  arrives.
- Concrete method: build slots over a window of pooled latents (per-clip vectors across time), train
  the slot head on the relation-change and containment control families, and probe whether
  slot assignments decode relation identity above a flat-pooled baseline. This is explicitly a
  PRECURSOR: it cannot recover within-frame object slots (pooled discards per-patch structure, the
  blocked-paradigm note), only temporal/event structure.
- Metric + baseline: relation/event decoding accuracy from slot assignments vs a parameter-matched
  flat-pooled head; report the gap honestly as the pooled CEILING, with the dense gain reserved.
- Explicit null hypothesis: slot attention over pooled latents ties the parameter-matched flat
  baseline; pooled features carry no per-slot structure to factor (expected, and it bounds what the
  pooled substrate affords, motivating the deferred 2.1-dense arm).
- Negative-result category: 3 (frozen pooled latent lacks per-object factorization, the linear-probe
  bound) which is the EXPECTED and useful result, distinct from the dense path.
- Tier / feasibility: Tier D. The pooled precursor is run-now-local (Tier C in practice); the FULL
  within-frame slot version needs V-JEPA 2.1 dense per-patch tokens and stays deferred with E6.

### EX10: Cross-modal world model (V-JEPA video latents bound to AudioSet audio latents)

- Capacity rung advanced: 10 (language-mediated / multimodal learning) and 3 (episodic memory) via
  cross-modal retention. A genuinely novel multimodal-frozen-substrate experiment the registry does
  not yet name (adjacent to `language_conditioned_selection`, but audio not text, and a world model
  not a selector).
- Bleeding-edge hypothesis: binding frozen V-JEPA video latents with frozen audio latents from an
  AudioSet-style frozen audio encoder (a second frozen substrate, same doctrine, never trained) lets
  a small cross-modal predictor (predict audio latent from video latent and vice versa) improve
  continual RETENTION and abstraction over the unimodal shell, because the second modality is a
  partially-redundant teacher that stabilizes the shared latent map.
- Concrete method: cache AudioSet audio through a frozen audio encoder into a parallel pooled latent
  store (AudioSet is in registry/datasets.yaml as audioset_meta, metadata-only; the AUDIO encoder is
  a NEW frozen substrate that must be added to registry/models.yaml as available:false until a real
  HF id is verified, so the FULL run is gated on that). Train a tiny cross-modal predictor on the
  paired latents; measure whether cross-modal prediction as an auxiliary objective improves BWT and
  abstraction-probe accuracy on the continual stream vs the video-only shell.
- Metric + baseline: BWT and abstraction-probe accuracy, cross-modal-augmented shell vs video-only
  shell at matched params; linear-probe gate FIRST (audio-video correspondence must be decodable from
  the paired latents or there is nothing to bind, taxonomy entry 3).
- Explicit null hypothesis: cross-modal prediction does NOT improve retention or abstraction over the
  unimodal shell; the modalities are not aligned enough in the frozen latent spaces to bind (the
  correspondence is not decodable), or the auxiliary objective is just a regularizer.
- Negative-result category: 3 (cross-modal correspondence not decodable from the frozen latents) or 8
  (only helps combined with replay).
- Tier / feasibility: Tier D, run-now-local on the VIDEO side and on a structured-synthetic
  paired-control set immediately; the natural AudioSet arm is GATED on a verified frozen audio encoder
  HF id (currently none in registry/models.yaml, so honestly deferred, available:false, result_tag
  provisional until verified). The bleeding-edge claim stays provisional until the audio substrate is
  real. Pooled audio latents are tiny, so once the encoder is verified the corpus cost is trivial.

### EX11: Causal / interventional probing (do-operations on the synthetic control families)

- Capacity rung advanced: 8 (causal and affordance sketch). A new interventional probe the registry
  does not name; it extends the linear-probe spine from observational to interventional.
- Bleeding-edge hypothesis: the synthetic control families let us perform actual interventions
  (do-operations: hold one generative factor fixed and vary another, by construction), so we can ask
  whether the shell learns an INTERVENTIONAL latent map (predicts the effect of a do-operation) and
  not merely an observational correlation (Pearl-style do-calculus, operationalized on generative
  controls where the intervention is ground-truth).
- Concrete method: generate matched control clips under do(factor = value) interventions (the
  synthetic generator can fix factors deterministically by seed); train the shell to predict the
  post-intervention latent from the pre-intervention latent plus the intervention code; test whether
  it generalizes to UNSEEN intervention values, which an observational predictor cannot.
- Metric + baseline: post-intervention latent prediction error on held-out intervention values,
  interventional predictor vs an observational predictor trained only on the natural (non-intervened)
  joint distribution.
- Explicit null hypothesis: the shell cannot learn an interventional map beyond the observational one;
  it predicts seen interventions but fails to extrapolate to unseen intervention values (it learned
  correlation, not the do-effect).
- Negative-result category: 10 (interventional structure is conceptually beyond what frozen-latent
  prediction can recover) or 3 (the intervened factor is not decodable from the latent at all).
- Tier / feasibility: Tier C. Interventions are FREE on the synthetic generator (deterministic by
  seed); this is a structured-synthetic experiment by construction and runs entirely local. Provenance
  is structured-synthetic, honestly below natural-video, and the claim is bounded to controls.

### EX12: The representational atlas (what is linearly vs nonlinearly decodable, by encoder scale)

- Capacity rung advanced: 1 (sensory grounding) and 7 (abstraction); this is the BOUND on every other
  EX. Systematizes the linear-probe spine into a full atlas.
- Bleeding-edge hypothesis: a systematic sweep of WHAT is decodable from the frozen latent (class,
  action, occlusion/permanence, relation, count, intervention effect, audio correspondence) crossed
  with linear vs small-nonlinear probe AND with encoder scale (L 1024 / H 1280 / g 1408) AND with
  pooling, quantifies exactly what each frozen substrate affords, bounding every downstream mechanism.
  This is the empirical floor under the whole program: a mechanism cannot use information the atlas
  shows is not there.
- Concrete method: run linear and small-MLP probes for every target variable over every encoder's
  cache (Section 3 multi-encoder), tabulate decodability (accuracy above chance, with the shuffle-label
  ablation as the chance floor) as a function of encoder scale and probe nonlinearity. The diagonal of
  this atlas is the linear-probe gate every other EX already depends on; this experiment makes the
  whole matrix explicit and permanent.
- Metric + baseline: per-target probe accuracy above chance, linear vs nonlinear, per encoder; the
  shuffle-label control is the chance baseline (decodability must collapse under shuffle).
- Explicit null hypothesis: probe accuracy does not exceed the shuffle-label chance floor for a given
  target (that target is simply not in the latent); and the cross-encoder null: bigger frozen
  perception does not increase decodability of any target beyond the seed spread.
- Negative-result category: 3 (the target is not decodable, the cleanest taxonomy entry) and 9 (a
  decodability difference across encoders is representational, kept separate from any compute claim).
- Tier / feasibility: Tier C. Probes are the cheapest thing in the repo; the atlas is wall-clock,
  which is free. This experiment is the highest-leverage low-cost run in the bank because it bounds
  the truth-value of every mechanism EX above it. Run it FIRST per encoder.

### EX13: Long-stream continual learning (hundreds of sequential tasks, the forgetting curve)

- Capacity rung advanced: 4 (plastic adaptation) and 5 (consolidation), pushed to the stress limit.
  No single registry slug; this is the scale stress test the Studio uniquely affords.
- Bleeding-edge hypothesis: pushing the continual stream from a handful of tasks to HUNDREDS of
  sequential tasks reveals the forgetting CURVE (not a single BWT number) and the point where each
  mechanism's retention breaks; mechanisms that tie on a short stream (E2 prioritized vs random, E3
  staged vs cosine) may separate or both collapse on a long one, and loss-of-plasticity (Dohare et
  al.) and Fisher saturation (the repo's documented consolidation weakness) become visible only here.
- Concrete method: build a long domain/class-incremental stream of hundreds of tasks from the
  multi-source natural-video corpus plus the synthetic families; run E1 through E4 and the Level-5
  protected arm over the full stream; plot retention vs stream position for each mechanism, and probe
  for dead features / effective-rank collapse (the shrink_and_perturb rejuvenation diagnostic).
- Metric + baseline: the forgetting curve (retention vs task index), the stream length at which each
  mechanism crosses a retention threshold, and effective rank over the stream; the naive sequential
  arm is the lower-bound baseline.
- Explicit null hypothesis: retention is flat in stream length (no curve, mechanisms neither separate
  nor break) within the seed spread, OR every mechanism degrades identically (the protection does not
  scale, only delays the same collapse).
- Negative-result category: 8 (a mechanism only holds combined with rejuvenation at long horizon) or
  6 (the long stream is too hard for the tiny shell regardless of mechanism, a capacity bound).
- Tier / feasibility: Tier C, and this is THE experiment unlimited wall-clock unlocks: hundreds of
  sequential tasks at full seeds is days of compute the laptop cannot spend and the Studio can. The
  latent store stays a few GB; only the run-time grows, which is free here.

### EX14: Uncertainty-indexed and associative memory at corpus scale (memory-system bake-off)

- Capacity rung advanced: 3 (episodic memory). Promotes paradigms `hopfield_associative_memory` and
  `uncertainty_indexed_memory` (both prototype-local) to corpus scale.
- Bleeding-edge hypothesis: at a LARGE natural-video latent store, a modern Hopfield associative
  memory (Ramsauer et al.) stores more useful patterns per slot than a FIFO buffer, and indexing
  replay by epistemic uncertainty (ensemble disagreement) beats random eviction on retention per
  slot, both effects that may only appear at scale where the buffer is genuinely capacity-bound.
- Concrete method: at a large pooled store, compare retrieval recall@k and downstream BWT across FIFO,
  modern-Hopfield, and uncertainty-indexed eviction at matched memory budget; the corpus scale is the
  point (the prototype-local versions ran on tiny stores where capacity never bound).
- Metric + baseline: recall@k and BWT at matched buffer slots, Hopfield and uncertainty-indexed vs
  FIFO/random.
- Explicit null hypothesis: associative memory does not beat FIFO capacity, and uncertainty indexing
  ties random eviction (the registry's stated corpus null), even at scale.
- Negative-result category: 4 (index/memory structure too weak) or 5 (stream not long enough to bind
  the buffer, which EX13's long stream then addresses).
- Tier / feasibility: Tier C. The store is a few GB in 96 GB; brute and KV retrieval are fast on MPS.

### EX15: Shrink-and-perturb rejuvenation against loss of plasticity (long-horizon plasticity repair)

- Capacity rung advanced: 4 (plastic adaptation) over a long horizon. Promotes paradigm
  `shrink_and_perturb` (prototype-local) and pairs with EX13.
- Bleeding-edge hypothesis: under a long continual stream, the shell loses plasticity (dead units /
  effective-rank collapse, Dohare et al.); periodic shrink-and-perturb (Ash and Adams) restores
  plasticity without sacrificing retention, measurable only on a long enough stream.
- Concrete method: run the EX13 long stream with and without a rejuvenation schedule; track effective
  rank / dead-unit count before and after each rejuvenation and the net retention.
- Metric + baseline: recovered effective rank and dead-unit count, plus retained accuracy, rejuvenated
  vs un-rejuvenated long-stream shell.
- Explicit null hypothesis: rejuvenation does not restore plasticity, or it restores plasticity at the
  cost of retention (a wash); the frozen-latent shell does not suffer loss of plasticity at this scale.
- Negative-result category: 8 (only helps combined with consolidation) or 1 (the plasticity-loss
  mapping does not apply to a tiny frozen-latent shell, a clean negative that bounds the analogy).
- Tier / feasibility: Tier C, dependent on EX13's long stream. Pure wall-clock cost, free here.

### EX16: Discrete codebook / VQ abstraction and successor representation over latents

- Capacity rung advanced: 7 (abstraction) and 12 (skill library via reusable discrete concepts).
  Promotes paradigms `codebook_abstraction` (prototype-local) and `successor_representation`
  (studio-later).
- Bleeding-edge hypothesis: a learned VQ codebook over frozen latents (van den Oord et al.) yields
  reusable discrete concepts, and a successor representation over the resulting latent clusters
  (Dayan; Stachenfeld et al.) captures temporal/relational structure that improves next-state and
  transfer decoding over raw latents.
- Concrete method: fit a VQ codebook over the natural-video latents, measure codebook usage and
  cluster purity against labels; build a successor representation over the codes and probe next-state
  and relation decoding vs a raw-latent probe; report transfer (FWT) from SR features.
- Metric + baseline: codebook usage and cluster purity, SR-probe vs raw-latent-probe accuracy on
  next-state/relation decoding, and FWT; raw-latent features are the baseline.
- Explicit null hypothesis: the codebook adds no decodable structure over raw latents (random codebook
  assignment ties it) and the SR provides no transfer benefit over raw-latent features.
- Negative-result category: 3 (no extra structure in the pooled latent to discretize) or 4 (codebook
  too small / SR estimator weak).
- Tier / feasibility: Tier C. Codebook and SR are tiny shell modules over pooled latents.

---

### Prioritization (highest leverage first, all honest about tier)

Run-now-local (Tier C), ordered by leverage-per-cost:

1. EX12 representational atlas. Run FIRST per encoder; it bounds the truth-value of every other EX and
   is the cheapest run in the bank. Nothing downstream is trusted past what the atlas shows is there.
2. EX13 long-stream continual learning. The single experiment unlimited wall-clock most uniquely
   unlocks; the laptop cannot spend the days, the Studio can, and the latent store stays tiny.
3. EX1 generative latent replay. The headline "dreaming" question (does generative replay beat
   stored-buffer replay on BWT) is directly answerable on pooled latents and genuinely novel here.
4. EX8 intrinsic-motivation bake-off. Four selection signals, one decisive noisy-TV gate, very cheap,
   and it sharpens the rung-6 curiosity claim that is central to the north-star loop.
5. EX5 local rules at scale. Extends I4 to the full multi-encoder natural-video scale and adds the
   continual-learning/locality tradeoff axis that the toy gap study cannot reach.

EX2 (synthetic-control arm), EX3, EX4, EX6, EX7, EX11, EX14, EX15, EX16 are all additionally
run-now-local (Tier C) and slot into the campaign once the five above anchor the bank.

Deferred (honest, not unblocked by the Studio):

- EX9 full within-frame slots and the E6 dense path: needs V-JEPA 2.1 dense per-patch tokens (not on
  HF as of 2026-06). The POOLED temporal-slot precursor is run-now; the dense arm waits for 2.1.
- EX10 natural-AudioSet cross-modal: needs a VERIFIED frozen audio encoder HF id (none in
  registry/models.yaml yet). The video side and a synthetic paired-control set are run-now; the
  natural multimodal claim stays provisional/deferred until the audio substrate is real.
- EX2 LIVE-environment planning, and any EX requiring an environment to ACT in: Tier R (env adapter +
  rented CUDA). Deferred exactly as E5 rollout and E10 are; the Studio is Apple Silicon, no CUDA.

### The single most novel bleeding-edge experiment

EX10, the cross-modal frozen-substrate world model. It is the only experiment that adds a SECOND
frozen substrate (a frozen audio encoder bound to the frozen V-JEPA video encoder) and asks whether
cross-modal prediction across two frozen latent spaces improves continual retention and abstraction.
It is genuinely novel (the registry names language-conditioned SELECTION but no cross-modal world
model), it respects every invariant (both encoders frozen, the shell tiny, an explicit null, the
linear-probe correspondence gate first), and it is honestly tagged: the video side and a synthetic
paired-control arm are run-now-local, while the natural-AudioSet arm stays provisional and deferred
until a real frozen audio encoder id is verified in registry/models.yaml. It is the furthest reach
that still lands inside the doctrine.

---

## Section 7: POSITIONING AND PATH TO RECOGNITION

Sections 1 to 6 build the instrument and run the experiments. This section answers the question
those experiments exist to serve: can Brain become a niche-recognized voice, and if so, by what
honest path. "Niche leadership" here means research credibility, a referenced or adopted testbed,
and a portfolio piece that earns a young solo builder standing with labs and grad programs. It
does NOT mean a commercial product. The landscape facts below are from web research current to
2026; framing judgments that are knowledge-based and not source-backed are marked `[unverified]`.

The headline finding, stated first because it changes the whole plan: there IS a real, still-
forming community and a real publication path to join, NOT one to found. The frozen-latent
world-model niche is young and unconsolidated (a handful of serious papers as of mid-2026), the
honest-negative-results venue is live and welcoming, and both are reachable by one disciplined
person. The recognition path is JOIN and CONTRIBUTE, not LEAD and FOUND. Everywhere this document
previously implied "leadable as in found a movement," read "joinable and become a recognized
voice within an existing one." That is the honest, and still very real, ceiling.

### 7.1 The landscape, and where the white space actually is

Brain (package `mop`) sits at the intersection of four active fields. Mapping each one
honestly is the only way to find a real wedge instead of a flattering one.

Continual learning on frozen pretrained features is a MATURE, CROWDED subfield, not white space.
This is the single most important thing to get right, because a positioning that claims novelty
here is dismissed on sight. The mechanism "frozen backbone plus a tiny trainable surface, learn
continually without forgetting" is the defining template of an entire room with its own 2024
survey taxonomy. Zhou et al. ("Continual Learning with Pre-Trained Models: A Survey," IJCAI 2024,
arXiv 2401.16386) partition the field into three named groups: prompt-based, representation-based,
and model-mixture-based. The crowded canon, all verified (venue, year, arXiv, mechanism):

| Method | What it is | Frozen-backbone mechanism | Venue / year | arXiv |
|---|---|---|---|---|
| L2P | Learn a prompt pool to instruct a frozen ViT per task | Frozen ViT; small learnable prompt pool, key-query selection; only prompts + classifier train | CVPR 2022 | 2112.08654 |
| DualPrompt | Complementary general + expert prompts | Frozen ViT; task-invariant + task-specific prompts; rehearsal-free | ECCV 2022 | 2204.04799 |
| CODA-Prompt | Decomposed prompts assembled by input-conditioned attention | Frozen ViT; learnable prompt components, differentiable key-query | CVPR 2023 | 2211.13218 |
| RanPAC | Frozen random projection + class-prototype accumulation | Frozen backbone; frozen nonlinear random projection, decorrelated prototypes; no backbone updates | NeurIPS 2023 | 2307.02251 |
| SLCA | Slow (low-LR) tune + classifier alignment | Near-frozen representation; very low LR + Gaussian class modeling to realign classifier | ICCV 2023 | 2303.05118 |

Newer 2024-2026 methods EXTEND, not invent, the mechanism: EASE (adapters, CVPR 2024, arXiv
2403.12030), InfLoRA (LoRA, CVPR 2024, arXiv 2404.00228), LoRanPAC, SLCA++, SimpleCIL, C-LoRA,
and a steady stream of prompt/adapter/LoRA/prototype-on-a-frozen-encoder variants. The critical
literature is itself useful to Brain: Thede et al. ("Reflecting on the State of Rehearsal-free
Continual Learning with Pretrained Models," CoLLAs 2024, arXiv 2406.09384) and the APER/SimpleCIL
benchmark (Zhou et al., arXiv 2303.07338, IJCV 2024) both show that trivial frozen-backbone
baselines (first-task adaptation plus a nearest-mean classifier) already match many of these
methods. The safe sentence for any Brain writeup: continual learning on a frozen pretrained
encoder with a tiny trainable shell (prompts, adapters, LoRA, or a prototype/random-projection
head) is a mature, crowded subfield with its own 2024 survey taxonomy and critical literature;
the mechanism itself is not novel. Novelty, if claimed at all, must be in a specific component,
never in the frozen-encoder-plus-tiny-shell paradigm. (Note: do not quote specific citation
counts; centrality is asserted qualitatively from universal baseline use, the counts are
`[unverified]`.)

World models and the JEPA line are where the genuinely fresh, joinable community sits. The new
research finding, and the most important update over the prior brief: the V-JEPA 2 ecosystem is
NOT Meta-internal. It is a multi-lab, still-forming research line a newcomer can enter. (The other
world-model lineage, the DreamerV3 family (Nature 2025) and its continual-RL extensions Continual-
Dreamer and WMAR, is environment-and-rollout-heavy, which is exactly the regime Brain DEFERS to
Tier R, Section 5. Brain's niche is on the frozen-latent side, not the live-rollout side.)

- V-JEPA 2 (Meta, arXiv 2506.09985, June 2025) is the frozen video substrate Brain inherits: a
  lightweight attentive read-out on frozen features sets strong results on Something-Something v2
  and Epic-Kitchens, and a separate action-conditioned predictor (V-JEPA 2-AC) does control. The
  field's own framing already separates frozen vision from a learned predictor. Brain's framing is
  consistent with Meta's, which is good for credibility and bad for novelty of the CORE idea.
- The ecosystem is active and multi-lab as of mid-2026: V-JEPA 2.1 released 2026-03-16 (`facebook
  research/vjepa2` actively maintained); Apple's "Rethinking JEPA: SALT" (arXiv 2509.24317, Oct
  2025) is a credible non-Meta competitor on frozen-backbone eval; Balestriero and LeCun's
  "LeJEPA" (arXiv 2511.08544, Nov 2025, public repo `rbalestr-lab/lejepa`) and the LeWorldModel
  line are the most community-facing, hackable tooling. "World model on frozen latents" is now an
  established, named pattern (DINO-WM lineage on frozen DINOv2; "Hierarchical Planning with Latent
  World Models," arXiv 2604.03208, by the core V-JEPA 2 team incl. LeCun, Assran, Bardes, Ballas).
- Interpretability of these latents is a live 2025-2026 frontier and the MOST adjacent active work
  to Brain's linear-probe gate. The flagship is real and verified: Joseph, Garrido, Balestriero,
  Kowal, Fel, Rabbat et al. (Meta), "Interpreting Physics in Video World Models" (arXiv
  2602.07050), which linear-probes V-JEPA 2 and VideoMAEv2 latents and reports a "Physics
  Emergence Zone." A second, "What Makes Video World Model Latents Action-Relevant" (arXiv
  2606.07687), is real (Korean academic group). A solo-author preprint, "Probing the Latent
  World" (arXiv 2603.20327), is real but low-credibility (non-peer-reviewed, independent author),
  and is itself a data point: a solo outsider has ALREADY published into this exact niche, so the
  bar is reachable.

Developmental and brain-inspired AI is active but diffuse: brain-inspired CL surveys, replay-as-
memory papers, biological-mechanism taxonomies. Lots of "biology-inspired X helps" claims, very
little adversarial discipline about when biology does NOT help on a fixed substrate. That absence
is precisely the gap Brain fills.

Apple-Silicon reproducibility is real and underserved as a research-TOOLING angle, but it is a
TOOLING VIRTUE, not a research thesis (this is a sharpened verdict over the prior brief). The
documented limits are verified: FlashAttention and bitsandbytes are CUDA-only on MPS (scalastic
.io, Aug 2025); a measured performance gap vs NVIDIA from page faults, kernel-launch time, and
BLAS differences (arXiv 2501.14925, "Profiling Apple Silicon Performance for ML Training," Jan
2025); immature `torch.compile` on MPS; and cross-backend bitwise reproducibility is broadly
unachievable (floating-point non-associativity), so "reproducible on a Mac" must mean SAME-MACHINE
reproducibility, never bitwise-matches-CUDA. MLX is genuinely rising (arXiv 2510.18921, Oct 2025;
mlx-community model hub; Ollama on MLX) but inference-centric, not a training-research ecosystem.
The "GPU-poor" democratization movement is real (llama.cpp, tinygrad, local-LLM) but centered on
cheap inference and hobbyist self-hosting, and is largely invisible to top-venue reviewers who
care about results, not hardware. WHERE the angle genuinely earns credit: the ML reproducibility
crisis names COMPUTE COST as a top barrier, so "a reviewer can clone this and reproduce every
figure on one Apple Silicon machine in hours, no cloud account, on cached latents that never go
stale" directly attacks that named barrier. Lean on it as roughly 15 to 20 percent of the
positioning, a named accessibility-and-reproducibility pillar with honest caveats, never the
headline. Claiming Mac results are CUDA-equivalent, or that "runs on a Mac" validates the science,
is trivially rebutted and would convert a credible signal into a dismissible one.

The genuine white space, then, is not the mechanism (frozen plus small shell is taken) and not the
substrate (V-JEPA 2 is Meta's). It is METHOD and DISCIPLINE applied to one sharp question:

> When you hold perception fixed at a strong pretrained video substrate, which biological /
> developmental mechanisms measurably help, which are LR tricks in disguise, and which are bounded
> out by what the frozen latent threw away, each verdict carried by an explicit pre-declared null,
> a capacity ablation, and a linear-probe decodability gate.

Three reinforcing scarcities make that defensible:
1. Honest negative results as a first-class product. ICBINB and the slow-science movement exist
   precisely because the field under-publishes negatives. Brain's 10-category negative-result
   taxonomy, code-enforced null hypothesis (an `Experiment` that will not instantiate without
   `metric/baseline/ablation/null_hypothesis`), and linear-probe gate are unusually rigorous.
   Citable negatives are rare and high-value.
2. A bounded, falsifiable claim surface. Category 10 ("conceptually irrelevant to latent
   prediction, bounds the substrate") is the strongest kind of result: it tells the V-JEPA and CL
   communities what a frozen substrate cannot give you, a contribution they actively need.
3. Reproducible on commodity Apple hardware, with cached latents that never go stale (encoder
   frozen, so no retraining invalidates the cache). This lowers the cost for others to re-run the
   work, which is the precondition for adoption.

### 7.2 Verdict (honest)

As a widely-adopted benchmark replacing or rivaling Avalanche: NO. Avalanche (ContinualAI, the
incumbent PyTorch CL library, arXiv 2302.01766, JMLR vol. 24 2023) is the default substrate a
reviewer compares any new CL tooling against. Rivaling it requires institutional backing, a
maintainer community, broad task coverage, and a years-long adoption flywheel a solo 18-year-old
cannot spin up. Pretending otherwise is the fastest way to lose credibility. The right move is the
opposite: build the testbed to BENCHMARK AGAINST and interoperate with Avalanche, and inherit its
credibility rather than compete with it.

As a niche-recognized testbed and a credible referenced voice on one question: YES, conditionally.
The condition is that Brain is positioned as a MEASUREMENT INSTRUMENT and a RESULTS CORPUS, not as
a new mechanism. The mechanism room is full; the "here is what actually survives an honest null on
a frozen substrate, with the negatives reported" room is nearly empty. That is leadable-as-in-
become-a-recognized-voice by one disciplined person, because the moat is rigor and packaging, not
compute.

One-sentence wedge:

> Brain is the honest-null testbed for the question "which developmental and brain-inspired
> mechanisms genuinely help on a frozen V-JEPA 2 substrate, and which are bounded out by it,"
> producing citable negative results and a reproducible, runs-on-a-Mac harness that anyone can
> re-run on cached latents.

If even that proves too ambitious for external uptake, the honest fallback is high and real: Brain
is an A-grade portfolio and learning vehicle. The discipline (code-enforced nulls, provenance
tags, a safety scanner that refuses sentience claims, a probe-before-mechanism gate) demonstrates
research maturity that most applicants and many practitioners lack, and that alone earns standing
with labs and grad programs even with zero external adopters. The two outcomes are not mutually
exclusive: pursue recognition, bank credibility regardless.

### 7.3 Comparable-efforts gap (who else is in this exact niche)

The defensibility of the wedge rests on the niche being nearly empty, so name who is adjacent and
where the gap is. Adjacent, but not the same:
- V-JEPA interpretability authors (Joseph et al., arXiv 2602.07050) probe what is IN the latents.
  They do NOT run a continual-learning developmental-mechanism campaign on top, and they do not
  report honest nulls per mechanism. Brain's atlas (EX12) overlaps their method and extends it to
  continual-learning-relevant factors and to the cross-encoder scale axis.
- The frozen-backbone CL canon (L2P, DualPrompt, CODA-Prompt, RanPAC, SLCA) optimizes for SOTA on
  benchmarks. None of them runs a biological-mechanism-by-substrate honest-null study, and the
  critical literature (Thede et al.) shows the field WANTS exactly the trivial-baseline discipline
  Brain enforces.
- The solo "Probing the Latent World" preprint (arXiv 2603.20327) shows the niche is enterable by
  an independent author but is not occupied by a disciplined, reproducible, negative-results-first
  corpus. That is the open lane.

The gap, stated plainly: nobody is currently running a reproducible, honest-null, capacity-ablated,
probe-gated developmental-mechanism campaign on a frozen V-JEPA 2 substrate AND publishing the
negatives as first-class artifacts. That is the leadable-as-in-recognized-voice opening.

### 7.4 The non-experimentation moves that earn standing (the core ask)

Running E1 to E10 and the EX-series is necessary and not sufficient. Mindshare comes from
packaging, framing, and community contact. Prioritized for a solo builder, highest leverage first.
Each move maps to a concrete venue verified current to 2026.

Move 1: Ship ONE flagship writeup with a sharp, bounded thesis. A single preprint or long
technical report, not a sprawl. Recommended title shape: "What a frozen V-JEPA 2 substrate gives
and refuses: an honest-null study of developmental mechanisms in continual learning." Lead with
the E1 gate and the linear-probe spine, report E2 to E4 with their nulls (including the ones that
tied their baseline), and make the bounded negative (taxonomy category 10) the headline, not an
apology. This is the single artifact a lab, an admissions reader, or an arXiv endorser can
evaluate; without it there is nothing to cite. Solve the arXiv endorsement hurdle for independent
researchers early by asking an author of a cited paper. The natural venue for the workshop version
is the ICLR 2026 World Models Workshop (2nd edition; organizers incl. Mengyue Yang, Nicklas
Hansen, Dima Damen), the best single target for the V-JEPA-frontier framing.

Move 2: Package the framework as a thing others can actually run. The differentiator is the
CONTRACT, so expose the contract as the product: the abstract `Experiment` that refuses to run
without a declared null, the negative-result taxonomy, the linear-probe gate, the safety scanner.
A clean quickstart ("clone, pull cached latents, run E1 on a Mac in N minutes, get the frontier
plot") plus a documented way to add your own experiment under the same contract. Adoption requires
a five-minute path from zero to a reproduced plot. Lean on the cached-latent design; it is what
makes someone else's re-run cheap. Benchmark against Avalanche and engage the ContinualAI Slack
(continualai.org, open invite) so the testbed lands in the incumbent community rather than beside
it.

Move 3: Publish the representational atlas as a standalone reference artifact. The linear-probe
diagnostics (EX12) already measure what is and is not decodable from the frozen latent. Turn that
into a published, browsable atlas: for a fixed set of factors (identity, controllability, relation,
physics-ish properties), here is decodability from V-JEPA 2 latents, with method and seeds, across
encoder scale (L / H / g). This rides the live 2025-2026 interpretability wave (Joseph et al.,
arXiv 2602.07050) and is independently useful even to people who never touch the CL shell. It is
the most reusable, most-citable single artifact Brain can emit, and it directly substantiates the
"bounds the substrate" thesis.

Move 4: Convert one tie into a clean, standalone negative-results paper. Negatives are rare and
citable. Pick the cleanest null Brain produced (a strong candidate is E3: staged plasticity ties
tuned cosine decay, that is, "it was an LR trick on a frozen substrate") and write it up for
ICBINB ("I Can't Believe It's Not Better," icbinb.cc), which is verified live for 2026: the 2026
edition co-locates with ICLR 2026 (submission deadline 2026-01-31), selected papers have a PMLR
special-issue path (or opt out and keep it non-archival for resubmission), and it gives the
"Entropic Award" (most surprising negative) and "Didactic Award" (best-explained). It also runs a
monthly seminar series and a standing repository of unexpected negative results, and is explicitly
a welcoming community a newcomer can JOIN (no evidence a newcomer can found or chair it, so frame
this as publish-and-participate, not lead). A well-argued negative with a controlled ablation is a
stronger credibility signal for a young researcher than a marginal positive.

Move 5: Reproducibility packaging plus targeted community contact. Pin versions, publish the
cached-latent store and seeds, and document the Metal determinism caveat honestly (the repo
already notes ~50% byte-identical on Metal; surface that as a known limit, not a hidden one, and
remember "reproducible on a Mac" means same-machine, not bitwise-matches-CUDA). Then engage
narrowly: the V-JEPA / world-model latent-probing authors, the ContinualAI / Avalanche community
(Slack), the CLVision challenge (Continual Learning in Computer Vision; the 6th edition ran at
ICCV 2025 with a named ranked challenge, a low-barrier way to get a citable result), and the
ICBINB organizers. For a prestige replication path, MLRC (the ML Reproducibility Challenge) is now
an official NeurIPS 2026 track routed through TMLR and explicitly welcomes failures-to-reproduce.
A single blog post or thread framing Brain as "the honest-null Apple-Silicon CL testbed on frozen
perception," linking the atlas and the quickstart, is the lowest-cost distribution. Engagement
AFTER the artifacts exist, never before.

Deliberately deprioritized: a polished web demo, leaderboard infrastructure, or broad task-zoo
coverage. They cost a lot of solo time and move credibility less than the five moves above.

Ranked recognition paths (verified current to 2026, most-aligned first):
1. ICBINB workshop (ICLR/NeurIPS). The only venue purpose-built for clean negative results; short
   page bar; PMLR citation path; awards; active community to join. Best fit for Move 4.
2. ContinualAI / Avalanche plus the CLVision challenge. Best for ADOPTION and citations: build on
   and benchmark against Avalanche, engage the Slack, enter the annual challenge.
3. ICLR 2026 World Models Workshop. Best fit for the V-JEPA-frontier flagship (Move 1, Move 3).
4. MLRC routed to TMLR and the NeurIPS 2026 track. Highest prestige, ideal if the work is framed
   as reproducing or bounding a specific prior CL result. ReScience C is a lower-visibility
   fallback for a pure replication.

### 7.5 What would make Brain FAIL to lead

- No external adoption hook. If the only way to engage is to read the author's private repo and
  rebuild the environment, adoption is zero regardless of quality. No quickstart, no atlas, no
  published latents means good personal research that few will ever touch.
- Results not packaged for reuse. Runs that live only in `runs/` as the author's own logs are not
  a benchmark. A benchmark is a contract plus a reproducible path plus reference numbers others
  can compare against. Without that, the rigor is invisible.
- Claiming novelty in the crowded room. Positioning the frozen-encoder-plus-small-shell MECHANISM
  as new invites immediate dismissal (L2P, DualPrompt, CODA-Prompt, RanPAC, SLCA, V-JEPA 2-AC all
  precede it, Section 7.1). The novelty is the honest-null method and the bounded-substrate corpus,
  nothing else.
- Overreaching the substrate bounds. Any claim that survives only because the frozen latent
  happened to retain the needed factor, stated without the linear-probe evidence, is exactly the
  failure mode Brain exists to prevent. Discipline that lapses in the writeup destroys the one moat.
- Overclaiming the Apple-Silicon angle. Treating "runs on a Mac" as a research thesis, or implying
  Mac results are CUDA-equivalent, is trivially rebutted (Section 7.1) and converts a credible
  accessibility signal into a dismissible one. Keep it a tooling-and-reproducibility pillar.
- Any drift toward sentience / agency / consciousness language. The repo's safety scanner forbids
  it for a reason: a single such claim in a public artifact converts a credible testbed into a
  dismissible one. The "developmental" framing must stay strictly measurement-based (novelty,
  uncertainty, learning progress), never experiential.
- Sprawl over a sharp thesis. Ten half-finished experiments and three corpora documents read as
  undisciplined. One gate, one bounded claim, one atlas, one negative, shipped, beats everything
  half-done. Density is the credibility signal, consistent with the repo's own doctrine. (This
  document is itself the consolidation that fix demands: one continuous workflow, not scattered
  files.)

### 7.6 Bottom line

Brain will not become the next Avalanche, and it does not need to. Its realistic, leadable-as-in-
recognized-voice niche is to be the rigorous honest-null instrument and reference corpus for one
question the field is actively asking and under-answering: what fixed perception from V-JEPA 2
does and does not afford continual, developmental learning. The experiments are necessary but
invisible until packaged into artifacts, and the artifacts are inert until pointed at the right
venues. The new research confirms the venues are real and reachable: ICBINB for negatives, the
ICLR World Models workshop and the V-JEPA interpretability line for the frontier, ContinualAI /
Avalanche and CLVision for adoption, MLRC / TMLR for prestige. The five moves that convert private
rigor into public standing are, in order: the flagship bounded writeup, the runnable framework,
the representational atlas, a standalone negative-results paper, and reproducibility packaging with
targeted community contact. Do those and Brain becomes a recognized voice in a small, real niche
while remaining, in the worst case, an unusually credible portfolio piece.

---

## Section 8: THE BRIDGE (experiment -> artifact -> recognition)

This is the continuous workflow made explicit, and it is the load-bearing idea of the whole
document: the experiments are not the goal, they are the production line for citable, reusable
artifacts, and each artifact enables a specific recognition move from Section 7.4. Read each row
left to right as one sentence: this experiment produces this reusable artifact, which enables this
recognition move. Read the table top to bottom as the order of operations: the atlas and the
corpus come first because everything downstream depends on them, the campaign and the EX-series
produce the nulls and the headline result, and the recognition moves fire only once the artifacts
exist (engagement after the artifacts, never before).

| Experiment (Section) | Reusable artifact it produces | Recognition move it enables (Section 7.4) |
|---|---|---|
| EX12 representational atlas (S6); the linear-probe spine across L/H/g | The REPRESENTATIONAL ATLAS: a published, browsable map of what is linearly vs nonlinearly decodable from frozen V-JEPA 2 latents, by factor and encoder scale | Move 3 (standalone atlas) + the frontier flagship; lands on the live interpretability wave (Joseph et al., arXiv 2602.07050); ICLR 2026 World Models Workshop |
| Section 2 corpus build (multi-encoder pooled cache over real natural-video) | The CACHED-LATENT CORPUS: a permanent, never-stale, runs-on-a-Mac-cheap reusable substrate (a few GB across L/H/g) plus the seeds | Move 2 (runnable framework) + Move 5 (reproducibility packaging); the five-minute clone-and-reproduce path; ContinualAI adoption |
| E1 gate + E2/E3/E4 + Level-5 headline (S4) at full seeds on natural video | The HEADLINE RESULT with real error bars: protected-vs-naive frontier AUC and BWT, every mechanism carrying its pre-declared null | Move 1 (flagship bounded writeup, the E1 gate + linear-probe spine as the lead); arXiv preprint + ICLR World Models Workshop |
| E3 staged plasticity vs tuned cosine decay (S4); the cleanest tie | A CITABLE NEGATIVE RESULT: "staged plasticity was an LR trick on a frozen substrate," a controlled-ablation null (taxonomy entry mapped) | Move 4 (standalone negative-results paper); ICBINB at ICLR 2026 (Entropic/Didactic awards, PMLR path) |
| EX12 cross-encoder null + any category-10 bounded result (S3, S6) | The BOUNDED-SUBSTRATE CLAIM: a citable statement of what a frozen V-JEPA 2 latent conceptually cannot afford, the contribution the V-JEPA/CL communities need | The headline of Move 1; substantiates the wedge; engages the V-JEPA interpretability authors directly |
| EX13 long-stream continual learning (S6) at full seeds, hundreds of tasks | The FORGETTING-CURVE CORPUS: retention-vs-stream-position curves the laptop cannot produce, per mechanism | Move 2 + Move 5; a distinctive reference result; the CLVision challenge angle |
| The E-series + EX-series run under the code-enforced contract (S4, S6) | The CONTRACT-AS-PRODUCT: the `Experiment` that refuses to instantiate without metric/baseline/ablation/null, the 10-category taxonomy, the probe gate, the safety scanner | Move 2 (the framework others run under the same contract); the durable differentiator |

Three rows, read as the spine of the narrative: EX12 produces the atlas, which is the most
reusable, most-citable single artifact and the substance behind Move 3; the Section 4 campaign
produces the headline result with real error bars, which is the body of the flagship preprint
(Move 1); and the E3 tie becomes the citable negative result that fits ICBINB exactly (Move 4).

## Section 9: FINAL EDGE PASS, LEADERSHIP MECHANICS

This section is the final "above the rest" pass. It does not ask Brain to become bigger, more
agentic, or more dramatic. It asks what would make Brain unusually credible to serious people:
V-JEPA researchers, continual-learning researchers, reproducibility people, and negative-results
communities. The answer is not more spectacle. The answer is a set of artifacts that make the
project hard to dismiss because every claim is bounded, reproducible, and attached to a null.

### 9.1 The narrow leadership claim

The strongest claim is:

> Brain is the honest-null measurement instrument for what frozen V-JEPA 2 perception does and
> does not afford in continual developmental-learning mechanisms.

That is narrower than "new continual-learning framework" and stronger than "I tried some
bio-inspired mechanisms." It names a small open intersection:

- frozen world-model perception, not trained perception,
- developmental mechanisms, not generic benchmark tuning,
- continual learning, not one-shot classification,
- nulls and bounds, not only positive deltas,
- a reusable latent corpus and atlas, not private runs,
- Apple-Silicon reproducibility as access, not as the scientific thesis.

The niche is not to found a field. The niche is to become a recognized voice inside a forming
wave by publishing the thing labs rarely enjoy publishing: a clean map of which ideas survive
honest baselines and which collapse into learning-rate tricks, capacity artifacts, or substrate
blind spots.

### 9.2 The anti-claim

These claims are forbidden:

- Not "AGI", not consciousness, not sentience, not agency. The project remains measurement-based.
- Not "the next Avalanche." Avalanche is a framework ecosystem. Brain is a bounded instrument and
  corpus.
- Not "SOTA continual learning." If Brain wins a task, it reports the win, but the thesis is which
  mechanisms survive under a contract, not leaderboard conquest.
- Not "V-JEPA 2 understands the world" without probe evidence. Every affordance claim must pass
  the representational atlas.
- Not "Mac results are equivalent to CUDA." Apple Silicon is an accessibility and reproducibility
  pillar, not a universal hardware claim.
- Not "biology proves this mechanism." Brain can borrow names from developmental learning, but it
  must operationalize them as metrics, controls, and ablations.

### 9.3 The unfair artifact: the Atlas plus Null Cards bundle

The unfair artifact is not a polished demo. It is a bundled reference release:

1. **The Representational Atlas.** For each encoder scale and factor family, show linear probe,
   nonlinear probe, confidence intervals, and "not decodable" results. The atlas says what the
   frozen substrate can even support before any mechanism is allowed to claim credit.
2. **Null Cards.** Each experiment gets a one-page card: hypothesis, null, baseline, ablation,
   metric, result, taxonomy category, verdict, raw run ids. Positive results and nulls use the
   same template.
3. **The Cached-Latent Corpus Card.** Dataset sources, license state, preprocessing, encoder
   hashes, latent schema, seeds, storage size, known defects.
4. **The Experiment Contract.** A runnable interface that refuses to instantiate without metric,
   baseline, ablation, and null. This turns rigor into software, not just taste.
5. **A Reproduce-One-Plot quickstart.** Clone, fetch a tiny latent shard, run E1 or EX12, produce
   the reference plot.

The bundle is unfair because bigger labs often publish positive curves and leave negative runs in
the drawer. Brain can make the drawer the contribution.

### 9.4 The proof ladder

| Stage | Private work | Metric | Artifact | Public move | Recognition path |
|---|---|---|---|---|---|
| L0 | Verify MPS encoder throughput and latent determinism limits | clips/hour, corruption rate, same-machine stability | Studio readiness note | Internal only | Do not launch |
| L1 | Build the licensed cached-latent corpus | coverage, license status, schema validation | Corpus Card | Dataset release or artifact bundle | Reproducibility hook |
| L2 | Run EX12 atlas across L/H/g | probe scores, confidence intervals, nondecodable factors | Representational Atlas | Technical report section | V-JEPA/world-model audience |
| L3 | Run E-series at full seeds | AUC, BWT, CKA/probe-gated interpretation | Headline result | Flagship preprint | World Models workshop |
| L4 | Isolate cleanest null | tie vs tuned baseline, ablation completeness | Null Card | ICBINB submission | Negative-results credibility |
| L5 | External rerun of one plot | reproduced within tolerance | Third-party run card | README badge/table | Adoption trust |

Brain should not seek attention before L2 exists. Without the atlas, every mechanism result is
under-explained. With the atlas, even a negative result has scientific shape.

### 9.5 Scoreboards to avoid and scoreboards to own

Avoid:

- Generic CL average accuracy against full-framework incumbents.
- Task zoo breadth.
- Claims that hide the linear-probe floor.
- Handpicked positives without the null cards beside them.
- Any language implying experiential states.

Own:

- **Mechanism survival under declared nulls.**
- **Substrate affordance bounds from the atlas.**
- **Negative-result quality.**
- **Reproduce-one-plot friction.**
- **Cached-latent reuse cost.**

The scoreboard sentence:

> "Given a frozen V-JEPA 2 substrate, which developmental mechanisms produce gains that survive
> the null, the probe gate, and the capacity ablation?"

That is specific enough to be useful and small enough for one person to own.

### 9.6 The moat beyond code

The compounding assets are:

- **The atlas:** reusable by anyone studying V-JEPA latents, even if they ignore Brain's shell.
- **The null-card library:** a public archive of controlled failures and wins.
- **The experiment contract:** lowers the chance future work drifts into vibes.
- **The latent corpus:** makes reruns cheap because no one has to re-encode pixels.
- **The license ledger:** prevents the release from being weakened by provenance uncertainty.
- **The taxonomy of failure:** capacity artifact, substrate blind spot, baseline tie, objective
  mismatch, seed instability, and so on.
- **Venue alignment:** ICBINB for nulls, World Models workshop for V-JEPA framing, ContinualAI for
  adoption, MLRC/TMLR if a replication/bounding result is strong enough.
- **The safety language discipline:** no sentience claims. This protects credibility.

Better-funded groups are unlikely to package failures this way because positive novelty is their
main incentive. Brain can win attention by being the cleanest instrument, not the loudest claim.

### 9.7 First 30 days after the Studio arrives

**Machine-running tasks**

1. Run the MPS lift-and-verify procedure on ViT-L first, then H, then g if memory allows.
2. Cache a small licensed corpus shard across all available encoders and validate hashes,
   dimensions, corruption checks, and loader throughput.
3. Run EX12 on the shard before scaling. The atlas comes before the mechanism campaign.
4. Run the E1 gate and one known baseline with 5 seeds to confirm the full pipeline.
5. Scale corpus caching breadth-first only after the license ledger is clean.

**Human judgment tasks**

1. Freeze the factor list for the atlas: identity, controllability, temporal order, relation,
   object presence, motion, and any repo-specific factors already supported.
2. Define the null-card template and taxonomy before seeing the full results.
3. Choose the flagship thesis sentence and keep every result subordinate to it.
4. Decide which null would make the best ICBINB paper if it holds.
5. Review every public sentence for overclaiming or banned agency/consciousness language.

**Public artifact tasks**

1. Create `ATLAS.md` or a small static atlas page with seed-aware plots.
2. Create `NULL_CARDS/` with one card per experiment.
3. Create a tiny downloadable latent shard for the reproduce-one-plot quickstart.
4. Write the flagship report outline while runs are executing.
5. Draft the ICBINB negative-result abstract only after the cleanest null is selected.

**Stop/continue gates**

- If the atlas shows the needed factors are not decodable, do not run mechanism claims on those
  factors. Publish the bound instead.
- If E1 cannot beat the trivial baseline at full seeds, stop the campaign and fix the task or
  baseline.
- If seed variance changes the sign of results, do not write a positive result. Write the
  instability result.
- If the only interesting result needs a private dataset or unreleasable latent cache, treat it
  as internal, not a recognition path.

### 9.8 Rare high-leverage ideas, beyond "run more experiments"

1. **Null Cards as first-class artifacts.** Give failures the same visual and metadata treatment
   as wins. This is rare and directly aligned with ICBINB.
2. **A probe-before-claim rule enforced in CI.** Any experiment report that claims a latent factor
   must link to the atlas probe result. Bigger projects rarely make epistemic hygiene executable.
3. **The "mechanism obituary" appendix.** For each idea that failed, write the cleanest possible
   explanation of why. This turns dead ends into educational value.
4. **A one-plot reproduction challenge.** Ask other Mac users to reproduce exactly one atlas plot,
   not the whole system. Tiny ask, high trust.
5. **A bounded-substrate leaderboard.** Not SOTA accuracy. A table of mechanism vs null vs
   substrate affordance. It rewards rigorous interpretation, not raw score.
6. **Failure taxonomy badges.** Every null gets a category badge. Readers can scan the shape of
   failure across mechanisms.
7. **Latent corpus DOI or release tag.** The corpus, not the code, may become the reusable object.
   Make it citable if licensing allows.
8. **A "do not cite this as intelligence" note.** Put the language boundary in the public report.
   It signals maturity and protects against hype.
9. **Reference rerun notebooks.** One minimal notebook per major artifact: atlas, E1 gate, null
   card. The audience for credibility often wants plots, not a full training stack.
10. **Contact authors with a gift, not a pitch.** Send the atlas row relevant to their paper and
    ask for correction. This creates targeted community contact without self-promotion fog.

### 9.9 Final verdict

Brain can lead only in the modest but real sense of becoming a recognized instrument and reference
artifact for a narrow question: **what frozen V-JEPA 2 perception affords, refuses, and confounds
when developmental continual-learning mechanisms are tested under honest nulls.**

Leadership is real if the atlas is reusable, the null cards are rigorous, one plot is easy to
reproduce, and at least one external community can point to Brain as a clean example of bounded
negative-results methodology. It is not real if the work remains private logs, broad mechanism
claims, or SOTA-style average accuracy tables.

If the flagship mechanisms fail, the fallback value is still strong: a representational atlas, a
cached-latent corpus, a negative-results paper, and a disciplined research portfolio. The failure
case is not "nothing worked." The failure case can be "the substrate refused this family of
claims, and here is the evidence."

Single next action: **build the atlas and null-card templates before the full campaign.** Brain's
edge is not that it runs more experiments. Its edge is that every experiment becomes a reusable,
bounded artifact.
Every experiment in this document exists to land in one of those artifact rows, and every artifact
exists to fire one of the recognition moves. That is the continuous workflow: research substrate
-> experiments -> reusable artifacts -> positioning -> recognition, one line, end to end.

---

## Section 10: METHODOLOGY ASCENSION, THE PROOF SYSTEM

Sections 1 to 9 build the instrument, run it, position the results, and name the leadership
mechanics. This final section closes the arc by raising the METHODOLOGY itself to a standard that
is structurally hard to dismiss. It is not a request for more experiments; the bank above is
already deep. It is the proof system that sits UNDER every experiment and every artifact: the
rules that decide what Brain is allowed to claim, what forces a downgrade, what kills the wedge,
and what is publishable even when the result is negative. The test of this section is adversarial:
if a V-JEPA 2 researcher, a continual-learning expert, an ICBINB reviewer, a future grad advisor,
or a hostile commenter reads the work, the methodology should be unusually hard to attack. Brain
ascends here from "frozen V-JEPA continual-learning experiments" to an honest-null scientific
instrument for substrate affordance, negative results, and reusable bounds that others can cite.
Everything below is specified inline (schemas, field-lists, invalidation rules) so this stays one
consolidated document, never a scatter of side files. The single anchor sentence that governs the
whole section, stated once: Brain does not claim a mechanism helps unless it beats the declared
null, survives ablation, and the needed factor is shown decodable in the representational atlas.

### 10.1 The epistemic contract (what Brain promises, refuses, and what moves a claim)

The contract is the spine. It is the same discipline the code already enforces (an `Experiment`
that will not instantiate without metric, baseline, ablation, and null), lifted to the level of
public claims so a reader knows exactly what evidence is required before Brain says anything.

What Brain PROMISES to measure:
- Whether a target factor (identity, action, controllability, relation, permanence, count,
  intervention effect, cross-modal correspondence) is linearly or nonlinearly decodable from a
  named frozen encoder's cached latents, with a shuffle-label chance floor.
- Whether a developmental mechanism (replay, staged plasticity, uncertainty gating, generative
  replay, local rules, meta-init, and the rest of the E and EX banks) beats a tuned baseline on a
  pre-declared metric, at full seeds, on a named provenance tier.
- Whether an effect's sign and size hold across encoder scale (ViT-L / ViT-H / ViT-g) and across
  stream length, within the measured seed spread.

What Brain REFUSES to claim:
- That any mechanism "works" without the probe gate, the tuned baseline, and the ablation all
  passing first. No probe, no claim.
- That perception improved (the encoder is frozen and never trains; a grad-free unit test holds in
  every experiment).
- Any sentence implying experiential states, agency, understanding, intelligence, consciousness,
  or sentience. The vocabulary is strictly engineering: novelty, uncertainty, learning progress,
  prediction error, decodability, retention, calibration.
- That a Mac result is equivalent to CUDA, or that "runs on a Mac" validates the science.

The evidence ladder, stated as four rules a reader can hold Brain to:

| Decision | Trigger (the evidence rule) |
|---|---|
| PUBLISH a positive claim | The factor is decodable in the atlas AND the mechanism beats the tuned baseline beyond the seed spread AND the capacity ablation holds AND the sign is seed-stable. All four, or it is not a positive claim. |
| DOWNGRADE a claim | The mechanism ties the tuned baseline within the seed spread, OR the effect appears at one encoder scale but not another, OR the ablation removes most of the effect. Downgrade to a tie/bounded statement, never delete. |
| KILL the wedge | The probe gate shows the needed factor is not decodable from the frozen latent at all. The mechanism cannot make a representational claim; the result becomes a substrate bound (taxonomy entry 3), which is itself publishable. |
| PUBLISH even when NEGATIVE | A mechanism ties a tuned LR schedule, a factor is not decodable, a sign is seed-unstable, or a result is a capacity artifact. These are first-class outputs (Section 10.5), not failures to hide. |

The contract's one refusal that protects everything else: Brain does not claim a mechanism helps
unless it beats the declared null, survives ablation, and the needed factor is shown decodable in
the representational atlas. Every public sentence in every artifact traces back to this rule.

### 10.2 The adversarial validation layer (assume a hostile reader)

Every flagship claim is run through an adversary before it is written, not after a reviewer finds
the hole. The layer names the strongest baseline, the most embarrassing failure, and the critique
most likely to land, then states the test that settles it and the response in both directions.

| Field | Content |
|---|---|
| Strongest baseline | A TUNED baseline, never a strawman: tuned cosine LR decay (for plasticity claims), a tuned optimizer and capacity-matched head (for any mechanism), prioritized-vs-random replay at matched budget (for memory claims), and the trivial frozen-backbone baseline the CL critical literature uses (first-task adapt + nearest-mean, Thede et al., APER/SimpleCIL). |
| Most embarrassing failure | A published positive that turns out to be a learning-rate schedule, a capacity difference, or a seed-sign flip, that a reviewer reproduces and overturns. The whole instrument's credibility rests on this never happening in a public artifact. |
| Likely hostile critique | "Your bio-inspired mechanism is just an LR schedule." Also: "you only show the encoder where it works," "your seeds are cherry-picked," "your baseline is weak," and "this is a frozen-backbone CL paper dressed in developmental language." |
| The test that proves it right | Run the tuned cosine/optimizer/capacity ablations AND the linear-probe gate. If the mechanism still beats the tuned baseline beyond the seed spread, at more than one encoder scale, with the factor decodable, the critique is answered in the artifact itself. |
| Response if RIGHT (the mechanism survives) | Publish the survival card (Section 10.3): the mechanism, the tuned baselines it beat, the ablation, the probe dependency, the seeds, the raw run ids. Lead with the ablation, not the headline number. |
| Response if WRONG (the critique lands) | Publish the NULL CARD. "Staged plasticity ties tuned cosine decay on a frozen substrate; it was an LR trick here." This is the ICBINB headline, not an apology (Section 10.5). |

The layer's rule: a flagship claim is not ready until its null card and its survival card are BOTH
drafted, and the evidence decides which one ships. Drafting the embarrassing version first is the
cheapest insurance the instrument has.

### 10.3 The public proof grammar (the smallest unit others can cite)

The smallest public unit of Brain is not a plot and not a paragraph. It is a NULL CARD plus its
atlas dependency plus a raw run id. A reader can cite one card, reproduce it, or dispute it without
reading the whole corpus. Positive results use the SAME card template (a survival card is a null
card whose verdict is "null rejected"), so wins and nulls are visually and structurally identical,
which is the point.

The NULL_CARDS schema (one card per experiment, stored as `NULL_CARDS/<exp_id>.md`, fields exact):

```
exp_id:            E3 | EX12 | ... (stable id)
title:             one line, no agency language
hypothesis:        the mechanism claim, engineering vocabulary only
null_hypothesis:   the pre-declared null this must beat (verbatim from the experiment)
baseline:          the TUNED baseline (name the tuning: cosine decay, optimizer, capacity-matched head)
ablation:          the capacity/LR/seed ablation run and its outcome
metric:            frontier AUC | BWT | adaptation steps-to-threshold | ECE | recall@k | probe acc
probe_dependency:  REQUIRED. the atlas row (factor + encoder) the claim depends on, and its
                   decodability (acc above chance, or "not decodable -> claim void")
encoder_scale:     L | H | g | all-three; per-scale verdict if it differs
seeds:             n seeds, SEM, sign-stability verdict (stable at S>=k or unstable)
provenance_tag:    natural-video | real-encoder | structured-synthetic | provisional
result:            the numbers with confidence intervals
taxonomy_category: 1..10 (Section 10.5 map) for a null; "null rejected" for a survival card
verdict:           PUBLISH-POSITIVE | DOWNGRADE-TIE | SUBSTRATE-BOUND | SEED-UNSTABLE | CAPACITY-ARTIFACT
badges:            seed-instability | capacity-artifact | substrate-blindspot | tuned-baseline-tie (as applicable)
raw_run_id:        the run hash + config path under runs/ that produced this
repro_level:       R0..R5 (Section 10.6)
```

What makes a null card INVALID (any one of these voids it):
- No `probe_dependency`, or it cites a factor the atlas shows is not decodable, while still making
  a representational claim.
- The baseline is untuned (a strawman cosine, default optimizer, or capacity-mismatched head).
- Fewer seeds than the sign-stability threshold (the repo's measured S>=3 for sign), or no SEM.
- No `raw_run_id`, or the id does not reproduce the reported numbers within tolerance.
- A provenance tag richer than the cache actually supports (e.g. `natural-video` on a synthetic
  run), which the cache validator already refuses.
- Any sentence in the card drifting into agency/consciousness language (the north_star scanner
  refuses to render it).

How an outsider reproduces one card: clone the repo, fetch the tiny latent shard named in the
card, run the one command in `repro_cmd` (carried in the raw run config), and compare the produced
plot to the card's numbers within the stated tolerance (same-machine on Apple Silicon; see the
Metal caveat, Section 10.6). How someone submits an EXTERNAL rerun on another Mac: they run the
same one command, fill a third-party run card (the same schema plus their hardware string and the
delta to the reference), and open it as a contribution; Brain lists it next to the original. How
FAILED replications are shown: a failed rerun is not hidden; it is added as a third-party card with
`verdict: REPLICATION-FAILED` and the observed delta, and if it overturns a published claim, the
original card is marked SUPERSEDED with a link, never deleted. The archive of disagreement is part
of the credibility, not a threat to it.

### 10.4 The trust surface (what an evaluator sees first, proof-shaped)

When a serious reader arrives, the first screen must be proof, not prose. The trust surface is the
ordered set of artifacts an evaluator meets before any narrative, each one shaped so a skeptic can
test it in minutes.

| Order | Artifact | What the evaluator can do with it in minutes |
|---|---|---|
| 1 | The Representational Atlas (Section 10.9) | See exactly what each frozen encoder affords, including the "not decodable" rows, before any mechanism is allowed to claim credit. |
| 2 | The Null-Card Gallery | Scan wins and nulls in one identical template, sorted by taxonomy badge; see that nulls are shown, not buried. |
| 3 | The Reproduce-One-Plot quickstart (Section 10.9) | Clone, fetch a tiny shard, run one command, regenerate one reference plot on their own Mac. |
| 4 | The flagship bounded writeup | Read the single sharp thesis with the bounded-substrate result as the headline, not an appendix. |
| 5 | The failure taxonomy (Section 10.5) + mechanism-obituary appendix | See the shape of every failure across mechanisms, with the cleanest explanation of each dead end. |

The rule: nothing that is not proof-shaped goes above these five. A demo, a logo, or a roadmap
sits below them, if at all.

### 10.5 The negative-result strategy (how failure becomes status)

The whole instrument is designed so that a failure is an asset, not a loss. This is the rarest and
most defensible thing Brain can own, because better-funded groups are incentivized toward positive
novelty and leave their nulls in the drawer.

PUBLISH (these are deliverables):
- A mechanism ties a tuned LR schedule (e.g. staged plasticity vs tuned cosine decay on a frozen
  substrate). The clean "it was an LR trick here" result.
- A factor is not decodable from the frozen latent, so a mechanism that needs it cannot work
  (substrate bound, the contribution the V-JEPA and CL communities actively want).
- A sign is seed-unstable (the effect flips across seeds). Publish the instability, not a positive.
- A result is a capacity artifact (the effect is a parameter-count difference, removed by the
  capacity-matched ablation).

DO NOT PUBLISH:
- Any claim resting on a private, unreleasable dataset or an unreleasable latent cache. If it
  cannot be reproduced from a released shard, it is internal, not a recognition path.
- A positive whose probe dependency is missing or whose baseline is untuned (it is not yet a
  result, it is a draft).

The FAILURE_TAXONOMY (the 10 categories every null maps to, inline so no side file is needed):

| # | Category | One-line meaning |
|---|---|---|
| 1 | Biology-mapping adds no measured benefit | The developmental name adds complexity, the metric does not move. |
| 2 | Effect explained by a simpler control | A trivial baseline already captures it. |
| 3 | Frozen latent lacks (or gains) the needed factor | Substrate blind spot or substrate gift; the probe gate decides. |
| 4 | Capacity/estimator too weak | Predictor, generator, hypernet, or codebook under-sized. |
| 5 | Stream too uniform/short for structure to appear | No curriculum or meta-structure to learn at this scale. |
| 6 | Tiny shell capacity bound | The task is too hard for the shell regardless of mechanism. |
| 7 | Needs embodiment/action (Tier R) | Requires a live environment to act in; deferred, not failed here. |
| 8 | Only helps combined (hybrid) | Pure mechanism fails; mechanism-plus-anchor or plus-replay wins. |
| 9 | Representational vs compute/locality claim separated | The gain is a compute/locality property, not a representational one. |
| 10 | Conceptually beyond frozen-latent prediction | The structure (e.g. interventional) is out of reach of the substrate. |

The FAILURE_TEMPLATE is the null card itself (Section 10.3) with `verdict` in {DOWNGRADE-TIE,
SUBSTRATE-BOUND, SEED-UNSTABLE, CAPACITY-ARTIFACT} and the matching badge set. Tags/categories are
the taxonomy number plus the badges (seed-instability, capacity-artifact, substrate-blindspot,
tuned-baseline-tie). How failures alter the roadmap: a category-3 substrate bound retargets effort
to the dense V-JEPA 2.1 path (Section 5) when it ships, or to a different factor; a category-4
capacity artifact retargets to a capacity sweep before any further mechanism claim; a category-8
hybrid retargets to the combined arm as the real result. What forces a FULL pivot: if EX12 shows
the atlas factors the whole campaign depends on are broadly not decodable from any encoder scale,
the mechanism campaign stops and Brain ships the atlas-as-bound as the primary contribution
(Section 10.12).

### 10.6 The reproducibility gradient (R0 to R5, every claim tagged)

Reproducibility is not binary. Every claim, card, and atlas row carries a level so a reader knows
exactly how far it has been verified, and Brain never lets a low-level claim wear a high-level
voice.

| Level | Definition | What it takes |
|---|---|---|
| R0 | Private run | Numbers exist in `runs/` as the author's own logs. Not citable. |
| R1 | Command + config | The exact command and config are published; a reader can see how it was produced. |
| R2 | Artifact hash + metrics | The run's artifact hash and metric values are pinned and published. |
| R3 | One-command local repro, same machine class | A single command regenerates the plot on an equivalent Apple-Silicon machine. |
| R4 | Third-party Mac repro | An independent person reproduced it on their own Mac and filed a third-party card. |
| R5 | Atlas/null cited externally | Another paper or project cites the atlas row or null card as a reference. |

The honest Metal-determinism caveat, stated wherever a repro level is claimed: cross-backend
bitwise reproducibility is not achievable (floating-point non-associativity), and Metal is roughly
50% byte-identical at temperature 0 while CPU is bit-identical and is the tolerance baseline. So
"reproducible on a Mac" means SAME-MACHINE-CLASS reproducibility within a stated tolerance, never
bitwise-matches-CUDA. Every R3+ claim publishes the tolerance it was checked against. The rule: a
claim's voice cannot exceed its level. An R1 result is described as "command published," not "third
parties confirm." Most flagship claims should target R3 before launch and R4 soon after.

### 10.7 The incumbent-resistance test (could a bigger lab copy this in two weeks)

Before any claim is treated as a moat, it passes one question: could a better-funded lab copy it in
two weeks, and if so, why does Brain's version still matter? The honest answer for most single
RESULTS is yes, a bigger lab could rerun a bigger experiment. The resistance is not in any one
result; it is in the packaging discipline that incumbents are disincentivized to do.

| Test field | Brain's honest answer |
|---|---|
| Could a bigger lab copy a single claim in two weeks? | Yes. They have more compute and can rerun a bigger version of almost any one experiment. |
| Why does our version still matter? | Because the value is the corpus of nulls packaged as first-class, citable, reproducible artifacts under one contract, not any single positive curve. Labs publish positive curves and leave nulls in the drawer. |
| Barrier type | Incentive and discipline, not compute. Packaging nulls, capacity ablations, probe gates, and bounded negatives as the PRODUCT is against the positive-novelty incentive that drives funded groups. |

The sharpened sentence: labs can run bigger experiments, but they are disincentivized to package
nulls and bounded negative results as first-class outputs, so the open lane is the discipline, not
the scale. Brain's resistance compounds because the atlas, the null-card library, and the contract
get more valuable as a reference the more they are reused, and reuse is exactly what a positive-only
publishing culture does not produce.

### 10.8 Category-ownership language (the words Brain owns and the words it bans)

The instrument needs language that a skeptic, an abstract, and a homepage can all use without
overclaiming. One primary category and two secondaries, with banned phrases stated so they never
slip in.

- PRIMARY category: "honest-null substrate atlas." Brain maps what a frozen V-JEPA 2 substrate
  affords and refuses, with every mechanism verdict carried by a declared null.
- SECONDARY 1: "mechanism survival card." The unit of evidence: a mechanism that beat its tuned
  baseline, its ablation, and its probe gate, or the null that says it did not.
- SECONDARY 2: "bounded-substrate result." A citable statement of what the frozen latent
  conceptually cannot afford (taxonomy entry 10) or does not decodably contain (entry 3).

BANNED phrases (any of these in a public artifact is a defect): anything implying agency,
intelligence, understanding, consciousness, sentience, experiential states, or self; "SOTA",
"beats the leaderboard", "state of the art continual learning"; "the next Avalanche"; "AGI";
"the model understands"; "Mac-equivalent to CUDA"; "biology proves."

Three sentences, each tuned to its reader:
- HOMEPAGE sentence: "Brain is an honest-null substrate atlas for frozen V-JEPA 2 perception: it
  measures which developmental continual-learning mechanisms survive a declared null, and which are
  bounded out by what the frozen latent threw away."
- TECHNICAL-ABSTRACT sentence: "We hold perception fixed at a frozen V-JEPA 2 encoder and evaluate
  developmental continual-learning mechanisms under pre-declared nulls, tuned baselines, capacity
  ablations, and a linear-probe decodability gate, reporting both survivals and bounded negative
  results across encoder scale (ViT-L/H/g) on cached natural-video latents."
- SKEPTICAL-EXPERT sentence: "The mechanism (frozen backbone plus tiny shell) is not novel and we
  do not claim it is; the contribution is a reproducible, probe-gated, capacity-ablated corpus of
  mechanism survival cards and substrate bounds, including the nulls, which the field
  under-publishes."

### 10.9 The 10x artifact (the atlas plus the null-card gallery, specified inline)

The compounding artifact is the Representational Atlas bundled with the Null-Card Gallery. It is
the most reusable and most-citable thing Brain emits, useful even to someone who never touches the
shell. Specified inline so it stays in this one document.

The folder/file structure (a single release tree):

```
proof/
  ATLAS.md                      # the atlas index + how to read it
  atlas/
    <encoder>/<factor>.json     # one row per (encoder x factor): linear acc, nonlinear acc,
                                #   chance floor (shuffle-label), CI, seeds, repro_level, raw_run_id
    atlas_summary.csv           # the full matrix: encoder x factor x probe-type x decodability
  NULL_CARDS/
    <exp_id>.md                 # one card per experiment, the Section 10.3 schema
    third_party/<exp_id>__<who>.md   # external reruns, including REPLICATION-FAILED cards
  CORPUS_CARD.md                # the cached-latent corpus card (Section 10.10)
  REPRODUCE_ONE_PLOT.md         # the quickstart spec (below)
  FAILURE_TAXONOMY.md           # the 10 categories + badge legend (Section 10.5)
  OBITUARIES.md                 # the mechanism-obituary appendix (one entry per dead mechanism)
  DO_NOT_CITE_AS_INTELLIGENCE.md  # the language-boundary note (Section 10.11)
```

The ATLAS spec (each `atlas/<encoder>/<factor>.json` row, fields exact):

```
encoder:        vjepa2_vitl_fpc64_256 | vjepa2_vith | vjepa2_vitg
factor:         identity | action | controllability | relation | permanence | count |
                intervention_effect | cross_modal_correspondence | motion | temporal_order
linear_acc:     probe accuracy (linear), with CI
nonlinear_acc:  probe accuracy (small MLP), with CI
chance_floor:   shuffle-label accuracy (decodability must exceed this or the row reads "not decodable")
decodable:      yes | no | marginal (relative to chance_floor + seed spread)
seeds:          n, SEM
provenance_tag: natural-video | real-encoder | structured-synthetic | provisional
repro_level:    R0..R5
raw_run_id:     run hash + config path
```

The REPRODUCE_ONE_PLOT quickstart spec (the five-minute path, stated as steps):
1. `git clone` the repo and `uv pip install -e ".[dev,encoder,apple]"`.
2. Fetch the tiny named latent shard referenced in the target card (a few MB, not the full corpus).
3. Run the single `repro_cmd` from the card (e.g. `python scripts/studio_pipeline.py run --exp EX12
   --shard <shard> --seeds 5`).
4. Compare the produced plot to the card's reference numbers within the stated tolerance.
5. Optionally file a third-party card with your hardware string and the observed delta.

Minimum-viable version: the atlas for ViT-L only, on the EPIC 5k licensed shard, with the
linear-probe rows for identity/action/relation, plus null cards for E1 and EX12, plus one
reproduce-one-plot path. Gold-standard version: all three encoders, all factors including the
"not decodable" rows, linear and nonlinear probes, the full E and EX null-card gallery with
third-party reruns, a citable corpus tag, and the obituary appendix. What makes it TRUSTED: every
row carries a chance floor, a CI, a repro level, and a raw run id, and the "not decodable" rows are
shown as prominently as the decodable ones. What makes it INVALID: a missing chance floor (no
shuffle-label control), a missing repro level, a row whose raw run id does not reproduce, or any
decodability claim a mechanism card depends on that is absent from the atlas. How it COMPOUNDS: each
new encoder, factor, or third-party rerun adds a row without invalidating the others (the encoder is
frozen, so atlas rows never go stale), and external citations move rows to R5, turning the atlas
into a standing reference the field reuses.

### 10.10 The corpus card and the CI rule (provenance made executable)

The CORPUS_CARD makes the cached-latent corpus citable and prevents provenance uncertainty from
weakening the release. Specified inline:

```
CORPUS_CARD fields:
  sources:          per source: slug, license state (CC BY-NC / manual / metadata-only), subset size
  encoders:         vjepa2_vitl/h/g, verified HF ids, embed dims (1024/1280/1408)
  encoder_hashes:   the weight hash per encoder (so a reader knows the exact frozen substrate)
  preprocessing:    frame count, resolution, pooling (pooled = latents + duplicated keys)
  latent_schema:    shape, dtype, per-clip pooled size (8/10/11 KB), backend tag (vjepa_hf)
  seeds:            the seed set used across the campaign
  storage_size:     pooled store size across encoders (a few GB)
  known_defects:    corrupt-file isolation, empty-class handling, short-clip handling
  license_ledger:   the resolved license state per source (the release blocker until clean)
  repro_level:      R0..R5; corpus tag/DOI if licensing allows (target R5)
```

The CI-check rule (epistemic hygiene made executable, no new code described here, only the rule the
contract enforces): every experiment report must include a NULL, a tuned BASELINE, and an ABLATION,
and every report that claims a latent factor must carry a `probe_dependency` linking to the atlas
row that shows the factor is decodable. A report missing any of the three, or claiming a factor
with no atlas link, does not render. This is the probe-before-claim rule promoted from taste to a
gate, which is exactly the discipline bigger projects rarely make executable.

### 10.11 The embarrass-us-before-launch checklist (run before any public artifact ships)

This is the last gate before anything goes public. It is written to be run adversarially against
Brain's own work, every item a yes/no that blocks the launch if it fails.

- [ ] Is the claimed factor decodable in the atlas (probe acc above the shuffle-label floor, with CI)?
- [ ] Did the TUNED baseline (cosine/optimizer/capacity-matched) tie or lose, and is the baseline
      genuinely tuned, not a strawman?
- [ ] Does the effect survive seeds (sign-stable at S>=3, SEM reported), not a seed-sign flip?
- [ ] Is there a capacity ablation, and does the effect survive it (not a parameter-count artifact)?
- [ ] Does the null card (or survival card) state the failure or the survival clearly, with the
      taxonomy category and badges?
- [ ] Does every claim carry its repro level, and does its voice not exceed that level?
- [ ] Is the provenance tag honest (not richer than the cache supports)?
- [ ] Does the raw run id actually reproduce the reported numbers within tolerance?
- [ ] Is ANY sentence drifting into agency, intelligence, understanding, consciousness, or sentience
      language? (The north_star scanner must pass; if a human reviewer hesitates, cut the sentence.)
- [ ] Is the Apple-Silicon angle kept as an accessibility/reproducibility pillar, never a CUDA-
      equivalence or research-thesis claim?

If any box is unchecked, the artifact is not ready. The cheapest credibility Brain can buy is
catching its own embarrassing version before a reviewer does.

### 10.12 First 7 / 30 / 90 days, and the if-it-fails-it-still-wins path

The plan is sequenced so the proof surfaces exist before the big runs, and the stop/continue gates
are explicit. Each window separates machine-running, human-judgment, artifact-building, and
public-communication work, with a gate that can halt the line.

FIRST 7 DAYS (schemas and proof surfaces, not giant runs):
- Machine-running: MPS lift-and-verify on ViT-L (Section 1); cache the EPIC 5k licensed shard;
  run the linear-probe gate and EX12 on the shard only.
- Human-judgment: freeze the atlas factor list; finalize the NULL_CARDS schema, the FAILURE_TAXONOMY,
  and the CORPUS_CARD fields (all inline above); pick the flagship thesis sentence.
- Artifact-building: create the `proof/` tree (Section 10.9), write the ATLAS index and one example
  null card and the REPRODUCE_ONE_PLOT spec.
- Public-communication: none. Nothing ships before L2 (the atlas) exists.
- STOP/CONTINUE gate: if the EPIC 5k shard fails the linear-probe gate, fix the cache/representation
  before anything downstream; do not run mechanisms on a cache that fails the probe.

FIRST 30 DAYS (first serious internal proof):
- Machine-running: scale the corpus breadth-first after the license ledger is clean; run EX12 across
  L/H/g; run the E1 gate and one tuned baseline at 5 seeds to confirm the full pipeline.
- Human-judgment: choose which null would make the best ICBINB paper if it holds; review every draft
  sentence against the banned-language list.
- Artifact-building: populate `NULL_CARDS/` with one card per run; fill the atlas with real rows
  including the "not decodable" ones; draft the flagship report outline while runs execute.
- Public-communication: still internal; prepare the trust surface (Section 10.4) but do not launch.
- STOP/CONTINUE gate: if the atlas shows the needed factors are broadly not decodable, do not run
  mechanism claims on them; pivot to publishing the bound (Section 10.5 full-pivot rule).

FIRST 90 DAYS (first credible public artifact):
- Machine-running: the full Tier C campaign at full seeds, per encoder; the cleanest null isolated
  with its tuned-baseline ablation.
- Human-judgment: run the embarrass-us-before-launch checklist (Section 10.11) on every artifact.
- Artifact-building: the flagship bounded writeup, the published atlas, and one standalone null card
  ready for submission; the corpus card with its license ledger clean.
- Public-communication: ship the first credible public artifact, a bounded preprint or an ICBINB
  submission. ICBINB is verified live for ICLR 2026 (submission deadline 2026-01-31, PMLR special-
  issue path, Entropic and Didactic awards), so the 90-day target is a real, dated venue.
- STOP/CONTINUE gate: if a flagship result is seed-unstable, ship the instability result, not a
  positive; if it needs an unreleasable cache, keep it internal.

The IF-IT-FAILS-IT-STILL-WINS path: if the mechanisms mostly fail, Brain still ships four real
deliverables, and they are the point, not a consolation. (1) The Representational Atlas: a
permanent, never-stale map of what each frozen V-JEPA 2 encoder affords, reusable by anyone probing
these latents, on the live interpretability wave. (2) The Null-Card Gallery: a public archive of
controlled nulls and the tuned baselines they tied, exactly what ICBINB and the slow-science
movement want. (3) A negative-results paper: the cleanest bounded substrate result written up for
ICBINB, a stronger credibility signal for a young researcher than a marginal positive. (4) A
reusable frozen-latent corpus with a corpus card and (licensing permitting) a citable tag, which
lowers the rerun cost for everyone after. The honest failure case is not "nothing worked." It is
"the frozen substrate refused this family of claims, and here is the bounded, reproducible,
probe-gated evidence." That sentence is itself a contribution the field is asking for, and it is
the floor under the whole instrument: even at its worst, Brain ships an atlas, a null library, a
negative-results paper, and a corpus that others can cite.

---

## Pre-Studio Scaffolding (do now on M3 Pro) + The Studio Go-Prompt

This section is the transition layer. It exists so that when the Mac Studio M2 Max (96 GB / 2 TB)
arrives, the move is "just press go." Everything that does NOT need the Studio is staged NOW on the
current M3 Pro 18 GB. The constraints are unchanged and not reopened here: Apple Silicon only
(Metal/MPS, no CUDA), one project owns the machine at a time, wall-clock is cheap, human focus is
scarce. Form follows BLACKHOLE.md: no em dashes, no agency language, the frozen encoder is never
trained, the linear-probe gate precedes any mechanism claim.

### Readiness verdict (cold-restart state, honest)

The repo is dormant but structurally intact, not broken. A cold-restart audit on 2026-06-27 found:

- Environment is live. `.venv` is a uv-managed Python 3.12.13. Core deps import (torch 2.12.1,
  numpy 2.4.6, pyyaml), MPS reports available. The encoder extras are ALREADY installed in this
  venv (transformers 5.12.1, huggingface-hub 1.20.1), so the optional path is import-ready even
  though `make install` only pins `[dev,ann]`. `uv` 0.11.7 is on PATH.
- Harness runs end to end. `scripts/run_experiment.py` runs the E1 baseline on resolved MPS in
  seconds and emits real arm metrics; `studio_pipeline.py` and `run_queue.py` show clean `--help`
  with all subcommands; the studio profiles load (`studio-1tb`: 900 GB usable, min-free 60 GB).
  Pytest collects 361 tests across 34 files with ZERO collection errors. All studio/devel modules
  are present (controls, datacards, downloader, pipeline, planner, profiles, registry; north_star,
  curriculum, metacognition, ablation, registries).
- The first thing that would break is NOT code, it is ASSETS. The real V-JEPA weights are NOT
  cached. `~/.cache/huggingface/hub` holds only `config.json` stubs for vith and vitg (785 bytes
  each, no `.safetensors` blobs) and NO vitl entry at all. So any real-encoder cache step would
  trigger a multi-hundred-MB-to-GB download before it could run. The prior 96-clip real ViT-L
  latent cache survives at `data/cache/vjepa2_vitl_fpc64_256_real/` (shape (96, 1024) float32, 6
  classes, linear-probe acc 1.0 on record in `runs/real_encoder_eval.json`), but the weights that
  produced it are gone from the cache.
- The first ARTIFACT does not exist yet. There is no `proof/` tree, no `ATLAS.md`, no `NULL_CARDS/`.
  Section 10 specifies it in full but nothing is scaffolded on disk.
- Disk is the binding constraint NOW, not on the Studio. The M3 Pro root volume shows ~56 GB free
  of 460 GB. That is enough for the three pooled weight sets plus a small EPIC shard, but it is not
  comfortable, so weight downloads must be sized and ordered, and a large raw-video pull is a
  Studio job, not an M3 Pro job.
- Working tree is dirty (modified docs + cache_tools/video + tests, untracked docs/, registry/,
  studio/, devel/). This audit does NOT touch it. Do not commit or push without explicit OK
  (production-audit discipline).
- No Brain process is running. The only live Python processes are from other projects (hawking,
  tools/condense, a web http.server); nothing to stop.

Bottom line: the instrument works cold. The gap to "press go" is three asset moves (pin the env,
download the frozen weights, pre-fetch the fully-licensed video) plus two human license tasks
(SSv2, Ego4D) plus building the first-artifact scaffolding. None of that needs the Studio.

### B. PRE-STUDIO SCAFFOLDING checklist

#### [DO NOW on M3 Pro 18 GB]

Each item: what, why, command, est. size/disk/RAM/time, and whether it is safe on 18 GB.

1. Pin the environment with the real extras (so the Studio reproduces exactly).
   - What: install the package editable with the encoder + video + apple extras into the existing
     `.venv`, then freeze a lockfile next to `pyproject.toml`.
   - Why: the venv already has transformers/hf-hub, but `make install` pins only `[dev,ann]`. Pin
     the full set NOW so the Studio gets a byte-for-byte env and no surprise resolves on day one.
   - Command: `uv pip install -e ".[dev,ann,encoder,video,apple]"` then `uv pip freeze >
     requirements.lock`.
   - Size/disk/RAM/time: torchvision + mlx add ~1 to 2 GB; under 5 min; RAM trivial. SAFE on 18 GB.

2. Smoke the harness and the readiness doctor (confirm green cold).
   - What: run the test suite and the studio doctor, record both.
   - Why: prove the cold restart is green BEFORE staging assets, so any later failure is isolated.
   - Command: `make test` then `make doctor` (writes `runs/studio_doctor.md`).
   - Size/disk/RAM/time: a few minutes; no large memory. SAFE on 18 GB.

3. DOWNLOAD the real V-JEPA weights so they are cached for the Studio.
   - What: fetch the frozen encoder weights into `~/.cache/huggingface` so the Studio never waits
     on a download. Order by disk: ViT-L first (the canonical default and the only one the M3 Pro
     needs for the atlas MVP), then ViT-H, then ViT-g only if disk allows.
   - Why: this is the single highest-value pre-stage. The encoder is frozen, so a weight pulled now
     is the EXACT substrate the Studio will use; nothing about it goes stale. It removes the
     first-break asset gap.
   - Command (per encoder, real download, opt-in path is already installed):
     `.venv/bin/python -c "from huggingface_hub import snapshot_download;
     snapshot_download('facebook/vjepa2-vitl-fpc64-256')"` then repeat for
     `facebook/vjepa2-vith-fpc64-256` and `facebook/vjepa2-vitg-fpc64-384`.
   - Size/disk/RAM/time: ViT-L ~304M params (~1.2 GB fp32 on disk), ViT-H ~630M (~2.5 GB), ViT-g
     ~1B (~4 GB). All three ~8 GB on disk. Download time depends on link; RAM trivial (no model is
     instantiated, only files are fetched). SAFE on 18 GB for download; with ~56 GB free, ViT-L+H
     are comfortable, ViT-g fits but watch the margin. Do NOT run a 64-frame forward here (that is
     the MPS blocker; it is a Studio verification step, not a download step).

4. Pre-fetch the freely-licensable video sources (no signed terms).
   - What: pull the EPIC-KITCHENS-100 small subset (status `available`, CC BY-NC 4.0) and the open
     Kinetics-700 ID/label CSVs (metadata-only, freely usable). Stage a small EPIC shard only
     (target the 1k or at most 5k subset on the M3 Pro), not the full 80 GB.
   - Why: EPIC needs no manual access, so it is the cleanest first real natural-video source and
     lets the Studio validate the pipeline on real video on day one. Kinetics CSVs are tiny and
     unlock the metadata schema. This is staging, not the full corpus (that is a Studio job).
   - Command: `.venv/bin/python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 40`
     then REVIEW `runs/studio_pipeline/latest/license_ledger.md`, then
     `.venv/bin/python scripts/studio_pipeline.py acquire --plan
     runs/studio_pipeline/latest/plan.json --execute --budget-gb 40 --accept-license` (EPIC + the
     Kinetics CSVs are the freely-licensable picks; the planner refuses manual/deferred sources by
     default).
   - Size/disk/RAM/time: EPIC 1k subset is roughly a few GB of raw video; Kinetics CSVs are MB.
     Hours of download at most; RAM trivial. SAFE on 18 GB; keep the EPIC subset small so the
     ~56 GB free is not exhausted. The full EPIC 20k and the multi-source corpus are STUDIO-ONLY.

5. START the license-acceptance process for the manual sources (HUMAN task, do now).
   - What: this is a person-task the user must begin now because it has external latency. (a) SSv2:
     register at developer.qualcomm.com, accept the Something-Something V2 terms, obtain the
     download token. (b) Ego4D subset: sign the Ego4D License Agreement, receive AWS credentials,
     install the ego4d CLI.
   - Why: both are `status: manual`; the planner will NOT auto-select them until terms are accepted
     AND `--accept-license` is passed. The approvals can take days, so starting now is what makes
     the Studio able to pull them on arrival. No download happens here, only the access grant.
   - Command: none (web registration + license signing). Record the token / AWS creds somewhere the
     Studio session can read them. Do NOT pull SSv2 or Ego4D video on the M3 Pro (Studio-only).
   - Size/disk/RAM/time: zero local resources; external wall-clock of days. SAFE on 18 GB (nothing
     runs locally).

6. Build the FIRST ARTIFACT scaffolding (the proof/ tree per Section 10).
   - What: create the `proof/` directory tree and the schema-bearing stub files exactly as Section
     10.9 specifies, so the Studio fills them with real rows instead of inventing structure. Create:
     `proof/ATLAS.md` (atlas index + how-to-read), `proof/atlas/` (per-encoder/factor JSON schema
     stub + `atlas_summary.csv` header), `proof/NULL_CARDS/` with one TEMPLATE card carrying the
     exact Section 10.3 field list (and a `third_party/` subdir), `proof/CORPUS_CARD.md` (Section
     10.10 fields), `proof/REPRODUCE_ONE_PLOT.md` (the five-step quickstart spec), and stubs for
     `proof/FAILURE_TAXONOMY.md`, `proof/OBITUARIES.md`, `proof/DO_NOT_CITE_AS_INTELLIGENCE.md`.
   - Why: the first artifact is the minimum-viable representational atlas + null-card schema +
     reproduce-one-plot stub. Section 10.4 says nothing ships before the atlas exists. Scaffolding
     the schema now is pure CPU/disk work with no Studio dependency and front-loads the human-
     judgment decisions (factor list, card fields) so the Studio only has to populate.
   - Command: a small scaffolding script or by hand, mirroring the Section 10.9 tree exactly. KB of
     text files. No heavy job.
   - Size/disk/RAM/time: KB; seconds. SAFE on 18 GB.

7. Generate the synthetic control-family fixtures (CPU-cheap, zero license risk).
   - What: generate the 9 control families (moving / permanence / occlusion / relation / containment
     / noisy-TV / navigation / class-incremental / domain-incremental) at the small recommended
     sizes (32 / 64 / 128) as the regression + gate fixtures.
   - Why: these are CONTROLS and fixtures, not science claims, but they back the linear-probe gate,
     the noisy-TV guard (the E4 / EX8 decisive test), and the determinism gate. They generate
     locally with no download and no Studio. Scaling them to thousands is a Studio job; the small
     fixtures are the M3 Pro stage.
   - Command: via the controls module, e.g. through the local-max rehearsal lane or
     `studio_pipeline.py` synthetic-controls selection at small sizes.
   - Size/disk/RAM/time: tens of MB; seconds to minutes; RAM trivial. SAFE on 18 GB.

8. Run a TINY end-to-end smoke to confirm the harness still works on 18 GB.
   - What: two cheap runs. (a) The whole-pipeline rehearsal on tiny fixtures: `make rehearse` (runs
     validate / preprocess / cache / integrity / dry-run grid / miniature-run / provenance /
     microbench on tiny `.npy` fixtures, codec-free). (b) An E1 toy run on the existing
     frozen-random or the small real ViT-L cache: `make e1` (or `.venv/bin/python
     scripts/run_experiment.py experiment=e1_baseline`). Optionally a tiny real-encoder latent-cache
     smoke on a handful of clips IF the ViT-L weights are downloaded and the encode is kept tiny
     (small frame count to stay under the MPS token cap; the 64-frame forward is the Studio
     blocker and is NOT run here).
   - Why: proves the harness is end-to-end green on the current machine, so the only thing the
     Studio adds is scale, not a fix. Confirms the gate plumbing, the cache validation, and the
     provenance stamps all still fire.
   - Command: `make rehearse` then `make e1`.
   - Size/disk/RAM/time: rehearsal writes to `runs/studio_rehearsal/`, MB, a couple minutes; E1 toy
     is seconds on MPS. SAFE on 18 GB. Do NOT run the full Tier C queue or a 64-frame real encode
     here.

#### [STUDIO-ONLY, do NOT run now]

These need the M2 Max (more GPU cores + 96 GB unified memory) or the full disk, and are explicitly
deferred off the M3 Pro:

- The MPS blocker lift itself: the real 64-frame ViT-L / ViT-H / ViT-g forward on Metal
  ([B, 64, 3, 256, 256] -> [B, 8192, 1024]). This is the Section 1 verification step and the thing
  the M3 Pro cannot do (it hangs the Metal graph compiler and falls back to CPU at ~24 s/clip).
- Full multi-encoder latent caching at scale: the permanent pooled corpus over EPIC 20k + SSv2 +
  Ego4D + Kinetics, cached three times (ViT-L / H / g). The encode wall-clock and the raw-video
  disk footprint are Studio-sized.
- The full EX / E campaign at 5 to 10 seeds: the whole Tier C bank (198 run-units) per encoder at
  full seeds, the encoder-scale comparison matrix (Section 3), the Level-5 headline at full seeds
  (Section 4), and the EX1 to EX16 bank.
- Any large raw-video pull (full EPIC, SSv2 video, Ego4D subset, Kinetics mirror video) and the
  large EPIC/SSv2 subsets. The M3 Pro stages only the small fully-licensed shard.
- Dense caching (E6 / V-JEPA 2.1) stays deferred regardless of machine (no 2.1 dense weights on HF);
  Tier R (E5 rollout, E10) stays deferred (needs an env adapter + rented CUDA, out of scope here).

### C. THE STUDIO GO-PROMPT

Paste this verbatim into a fresh agent session on the Mac Studio. It is self-contained, references
the canonical doc, and carries every invariant and boundary.

```
You are operating Brain on a Mac Studio M2 Max (96 GB unified memory, 2 TB SSD, Apple Silicon,
Metal/MPS, no CUDA). This machine owns the project one task at a time. Wall-clock is cheap; human
focus is scarce; optimize for thoroughness, corpus breadth, and statistical replication, not speed.

Canonical plan: read /Users/scammermike/Downloads/brain/docs/STUDIO_MAXIMIZATION_2026_06_27.md in
full first. It is one continuous workflow: Section 1 (lift the MPS blocker), Section 2 (the
permanent multi-encoder pooled cached-latent corpus), Section 3 (the encoder-scale experiment L vs
H vs g), Section 4 (the full Tier C campaign at full seeds), Section 5 (what stays deferred),
Section 6 (the EX1 to EX16 bank), Sections 7 to 9 (positioning and recognition), Section 10 (the
proof system), and the Pre-Studio Scaffolding section (what was already staged on the M3 Pro).

Invariants you must never violate:
- The encoder is FROZEN and never trains. A grad-free unit test must hold in every experiment.
- The linear-probe gate precedes every mechanism claim. No probe, no claim. If a needed factor is
  not decodable from the frozen latent, that is a substrate bound (taxonomy entry 3), which is
  itself the result; do not make a representational claim on it.
- No agency, sentience, consciousness, understanding, or intelligence language anywhere. Vocabulary
  is strictly engineering: novelty, uncertainty, learning progress, prediction error, decodability,
  retention, calibration. The north_star scanner must pass.
- Form: no em dashes and no en dashes (commas, colons, parentheses only).
- Provenance tags are honest and never richer than the cache supports (natural-video > real-encoder
  > structured-synthetic > provisional).

Do-not-reopen constraints (out of scope, do not relitigate): hardware choice, cloud, team, CUDA.
This is Apple Silicon only. Dense caching (E6 / V-JEPA 2.1) stays deferred (no 2.1 dense weights on
HF). Tier R (E5 rollout, E10) stays deferred (needs an env adapter plus rented CUDA). Full Ego4D /
Ego-Exo4D stay deferred. The mechanism (frozen backbone plus tiny shell) is NOT claimed as novel.

Already staged on the M3 Pro (verify, do not redo): the env extras are pinned
(.[dev,ann,encoder,video,apple]) with a lockfile; the real V-JEPA weights (ViT-L, and H/g if disk
allowed) are downloaded into ~/.cache/huggingface; a small fully-licensed EPIC shard and the
Kinetics CSVs are pre-fetched; the proof/ tree (ATLAS.md, atlas/, NULL_CARDS/ with the Section 10.3
template, CORPUS_CARD.md, REPRODUCE_ONE_PLOT.md, FAILURE_TAXONOMY.md, OBITUARIES.md,
DO_NOT_CITE_AS_INTELLIGENCE.md) is scaffolded; the synthetic control fixtures are generated; and a
tiny end-to-end smoke (make rehearse, make e1) is green. The SSv2 and Ego4D license-acceptance was
started by the user; check whether the token / AWS creds are now available before planning those
sources.

Execute in this order, gating at each step (gates are kill switches that STOP the run, not
warnings):

0. Readiness: `make test` (green FIRST), `make doctor`, confirm devices report the M2 Max (cores,
   unified memory), `make diag` to set determinism tolerances.
1. Section 1, lift-and-verify the MPS blocker on the real 64-frame forward, ViT-L first
   (`scripts/cache_real_encoder.py device=mps +classes=2 +per_class=1`). PASS means it returns
   latents without hanging; measure MPS vs the 24 s/clip CPU floor and record it with an honest
   throughput tag (optimization is not science). If it still hangs, use the documented fallbacks
   (CPU encode is acceptable under unlimited wall-clock); do not yak-shave MLX.
2. Section 2, build the permanent pooled corpus breadth-first, fully-licensed first. Plan under
   studio-1tb / 900 GB, REVIEW the license ledger, acquire EPIC (scale up from the staged shard) +
   synthetic controls (scale to thousands) + SSv2 and Ego4D subset ONLY if their licenses are now
   accepted and --accept-license is passed + Kinetics CSVs. Validate, then cache the SAME validated
   corpus three times, once per encoder (vjepa2_vitl_fpc64_256, vjepa2_vith, vjepa2_vitg).
3. Run the linear-probe gate and the determinism gate on each encoder's cache. A cache that fails
   the probe is a data/representation problem; fix it before any mechanism runs.
4. Section 10 first artifact BEFORE the big campaign: run EX12 and fill the representational atlas
   (proof/atlas) for ViT-L on the EPIC shard, with the "not decodable" rows shown as prominently as
   the decodable ones, every row carrying a chance floor (shuffle-label), CI, repro level, and raw
   run id. Write the first real null cards (E1 and EX12) under proof/NULL_CARDS using the template.
   Nothing ships before the atlas exists (Section 10.4).
5. Section 3 + 4, the campaign: per encoder, run the full Tier C bank at full seeds (E1 gate first,
   then E2/E3/E4 with their nulls and the noisy-TV guard, then the Level-5 headline), then the EX
   bank. Assemble the L/H/g comparison matrix. Push headline legs (E1, Level-5, E2, E4) to 10 seeds,
   ranking legs to 5, sanity legs to 3. Every result follows R0 to R5 and the evidence ladder:
   PUBLISH-POSITIVE only if the factor is decodable AND the mechanism beats the TUNED baseline beyond
   the seed spread AND the capacity ablation holds AND the sign is seed-stable; otherwise DOWNGRADE
   to a tie, a SUBSTRATE-BOUND, a SEED-UNSTABLE, or a CAPACITY-ARTIFACT card. Nulls are first-class
   deliverables, not failures to hide.
6. Run the embarrass-us-before-launch checklist (Section 10.11) on every public artifact. Keep the
   Apple-Silicon angle as an accessibility/reproducibility pillar, never a CUDA-equivalence claim.

Report continuously into runs/ and the proof/ tree. The deliverables are the atlas, the null-card
gallery, the reusable cached-latent corpus with its corpus card, and the reproduce-one-plot path.
The experiments exist to produce those artifacts, not the other way around.
```

