# SCAFFOLD: the M3 Pro process and the Studio process, post-run and post-audit

This is the single orchestrating artifact that separates what can be done NOW on the M3 Pro (18GB) from
what REQUIRES the Studio, sequenced per the potential audit's re-ordering. It supersedes the pre-run
sequencing in EXECUTION_MANIFEST.md section 4 and the pre-audit priority list, and it is grounded in
VERIFIED disk state as of 2026-07-02 (which caches and tooling actually exist), because the audit proved
that unverified doc claims over-reach. House style: no em or en dashes.

Read order for context: M3PRO_RUN_REPORT.md (what the run found, corrected), POTENTIAL_AUDIT.md (the
3.0/10 scorecard and the re-ordered action list), SEMANTIC_POSITIONS.md (the coverage/gap map).

## State snapshot (the honest ground truth this scaffold acts on)

- PROVEN (real encoder, non-vacuous control): substrate is special because of pretraining (V-JEPA 0.517
  vs random-init same-arch ViT-L 0.241 at matched 256px, p=0.0285, single split); it factors (shape,color)
  compositionally off ceiling (held-out 0.725 = seen 0.708); PR1 licenses a router (existence result).
- SURVIVORS (hold at 10 seeds): at3 temporal-currency; at1 cross-substrate invariance (same family as
  substrate-special); pr7 two-timescale only as a LEAD.
- DEMOTED by the audit re-grade: al2 (cross-modal alignment FAILS; only vision-vision aligns; the real
  V-JEPA arm was missing) and ws2 (fails its own dual acc-AND-NLL contract). Neither is a positive.
- NULLS: the entire test-time-compute lane at matched compute (regime-correct, but the program cannot yet
  POSE a hard task: no D3 gradient, no verifier). Most plasticity mechanisms tie tuned baselines.
- IDEOLOGY SCORE: 3.0/10. Moldability is false-by-construction on a frozen substrate; multi-perspective
  thought is instrumented at 2 visual slots; density was tested once and nulled; the falsification engine
  (6/10) is the one thing keeping it honest.

Verified caches on disk (data/cache/): vjepa2_vitl_nuisance (REAL, vjepa_hf, 200 clips, the arm al2
omitted), vjepa2_vitl_singleframe, randominit_vitl_nuisance, dinov2s_nuisance_{real,randominit},
qwen05b_textified_{real,randominit} (text-of-labels, NOT parallel states), wav2vec2_sonified_{real,
randominit}, handcrafted_descriptors, programmatic_reference (shape+color+nuisance only, no count/relation),
vjepa2_vitl_fpc64_256_{real,factorized}. Verified ABSENT: any DSL/executor, D3 hardness tooling, physics/
numerosity/geometry stimulus cache, real bound-attribute natural video, a paired LLM-hidden-state cache on
identical referents.

---

## PROCESS A: the M3 Pro (18GB) process, doable NOW, in leverage order

Every item here is verified runnable on the laptop against existing caches or cheap synthetic content. No
item needs the Studio. Sequenced so the zero-compute re-grades and the one decisive build come first.

### A1. Re-grade al2 (ZERO COMPUTE, the arm exists). Highest cheap leverage.
The real V-JEPA arm al2 reported missing IS on disk as `vjepa2_vitl_nuisance` (vjepa_hf, 200 clips); al2
looked for it under a different name. Re-run al2 including that arm, and replace the ridge-R2 metric with a
kNN-topology permutation null (R2 rewards scale, not shared structure). Report the full pair census.
Expected honest verdict: two frozen VISION encoders of identical content are weakly alignable; cross-modal
(vision to text, vision to audio) alignment does NOT survive the floor. Null: no pair beats the topology
permutation. Output: runs/mot/al2_alignment_regrade.json. This settles MoP's shared-code precondition on
the laptop.

### A2. Re-grade ws2 (ZERO COMPUTE, data present). 
The seeds10 JSON already has per-arm acc and nll. Enforce the preregistered dual contract (a fusion must
beat concat-MLP on BOTH acc AND nll, with the mean-baseline guard PR1 uses). On current data no arm clears
it (gwt_broadcast wins acc, loses nll). Emit the corrected verdict (null) and a per-arm dual-metric table.
Output: runs/mot/ws2_fusion_regrade.json.

### A3. Build the D3 hardness gradient + one executable verifier + a minimal DSL/code task. THE decisive build.
This is the single highest-leverage M3 Pro item, because it converts the 24 reasoning nulls from a prose
rescue-list into a live falsification. Build: (a) a difficulty-calibrated task with a per-sample hardness
label (non-trivial-but-not-ceiling), (b) a small executable verifier (code execution is CPU-cheap), (c) a
minimal DSL over the (shape,color,motion,count) slots the caches expose. Then re-run one dead reasoning
mechanism (dr8 fixed-point, or verify-revise) WITH the verifier on the graded task. Kill-switch: if it
still ties at matched FLOPs on a genuinely hard, verifiable task, the test-time-compute branch is honestly
dead at this substrate. Output: src/mop tooling + runs/mot/d3_verifier_reasoning.json.

### A4. pr7 delta-rule upgrade (cached-latent shell). 
Replace pr7's Hebbian outer-product store with a delta-rule (least-squares, covariance-aware) update, which
the deep research showed provably dominates it. Re-run vs the slow-only baseline AND vs the Hebbian floor.
This is the one plasticity LEAD that is laptop-runnable; it either becomes a real (if modest) positive or
confirms the floor. Output: runs/mot/pr7_delta_rule.json.

### A5. Recalibrate the 5 not-evaluable degenerates against the A3 gradient.
mt5 halting, al1 uncertainty router, dr12 disagreement, ws3 arbitration all self-reported no hardness
gradient / out-of-band control. Once A3 exists, re-run them on the graded task so they produce a real
verdict instead of DEGENERATE. Output: the corresponding runs/mot/*_recal.json.

### A6. Paired vision+text on IDENTICAL referents (M3 Pro, slow but no wall-clock limit).
The Qwen cache is text-OF-LABELS, not parallel LLM states on the same clips, so it cannot test the
language-independent-abstraction north star. Generate a caption/description per clip (deterministic,
pixel-derived) and cache the LLM hidden states on THOSE, paired to the existing vjepa2_vitl_nuisance vision
cache. This is the smallest instrument for the SEM-LANG cluster. Encoder-lane cost only (a small LLM pass),
one model at a time. Output: data/cache/qwen05b_paired_states + a cross-modal alignment run using A1's
corrected metric. NOTE: this is the boundary item; if it needs more than the laptop can encode, it drops to
Process B.

### A7. Author (do not run) the Studio scripts for Process B so the Studio session is execution, not design.
Pre-write the DR1 curation pipeline, the PR9 continual-backprop harness, and the multi-encoder atlas grid,
each with a pgrep encoder-guard and resumable per-clip-range legs, so the Studio just flips a flag and
scales N.

---

## PROCESS B: the Studio process, MANDATORY (the laptop structurally cannot do these), in order

### B1. DR1: the non-additive bound-attribute natural-VIDEO cache, with count and relation slots. THE unblocker.
Why Studio: real video curation plus a full encode pass past the max_cache_clips=128 laptop clamp and the
21s/clip CPU floor; the laptop caches expose only shape+color on 200 synthetic clips. DR1 is the sole named
enabler of GATE C1 and ~70 semantic positions. Bundle the paired vision+text pass at scale if A6 could not
be done on the laptop. This is the difference between having and not having a science on the
multi-perspective ideology.

### B2. PR9 continual-backprop on a LONG real-latent stream, gated to the kill-switch.
Why Studio: it needs a stream long enough to INDUCE plasticity loss; the laptop's 4-task, 640-sample toy
regime structurally cannot, so it would re-tie by construction. PR9 is the only plasticity mechanism
certified in the literature to beat a tuned baseline on this exact failure mode. It either wins (the first
substrate-touching plasticity positive) or ties (moldability is honestly dead at this substrate). Ship the
D3-style plasticity-loss certificate (late-vs-early accuracy gap positive under the SGD baseline) or it is
vacuous.

### B3. CM1 + a small real bound-attribute video batch through the FROZEN encoder.
Why Studio: needs B1's video. Makes GATE C1 falsifiable on real non-ceiling content without violating the
frozen doctrine (a random-init same-arch arm at matched resolution is the control). Until one custom-model
gate is runnable on real content, "keep it frozen" is unfalsifiable-by-construction on the ideology core.

### B4. The full multi-encoder atlas + dense V-JEPA 2.1 (8192-token) latents.
Why Studio: multiple large encoders resident at once, MPS acceleration, and dense caches (~32MB/clip)
exceed 18GB and the disk floor. This is the at-scale AT1/AL2 grid the laptop only piloted.

### B5. ONLY THEN: multi-seed the substrate headline (the +0.276 and the compositional -0.017) and settle
dr2 sparse-real with the 30-run protocol. These refine what the program already owns; per the audit they
must NOT come first. B5 is last on purpose.

---

## PROCESS C: wider-training box (gated, do not start until Process B verdicts force it)

Custom substrate pilots (object-centric slot module on frozen dense tokens; the CM7 1-5M pixel encoder
diagnostic) run ONLY if the atlas (B4) and the compositional gate (B3) show every frozen substrate hits the
same wall. Today's evidence tilts AWAY from this (the substrate is special, not blank). A full V-JEPA-scale
from-scratch model is a ~60 GPU-year moonshot outside doctrine and is not on the roadmap.

---

## Cross-process sequencing (the audit's pivot, stated as a rule)

Do A1-A5 (the zero-compute re-grades and the D3/verifier build) BEFORE any Studio time: they are free, they
correct two laundered positives, and A3 is what makes the whole reasoning lane falsifiable. Then B1 (DR1 +
paired text) is the decisive Studio spend, because it is the only thing that gives the multi-perspective and
language-independent-abstraction ideologies an instrument. B2 (PR9) resolves moldability either way. B5
(multi-seeding) is LAST. The failure mode to avoid, named by the audit: multi-seeding a p-value already
owned while DR1 sits unbuilt, which turns the program into a permanent negative-mapping machine on two
slots. The scaffold exists to prevent exactly that.

## Decision gates (kill-switches)

- If A1 shows cross-modal alignment fails even with the V-JEPA arm and a topology null: the shared-code
  precondition for MoP is NOT met on current substrates, and the workspace line is bounded to
  same-modality fusion until B1/B4 add real cross-modal content.
- If A3's verifier still ties at matched FLOPs on a hard verifiable task: retire the custom test-time-compute
  branch (the standing kill-switch fires).
- If B2 (PR9) ties on a stream that provably induced plasticity loss: moldability is dead at a frozen
  substrate, and the evidence for un-freezing (a trainable-encoder arm, Process C) becomes real for the
  first time.
