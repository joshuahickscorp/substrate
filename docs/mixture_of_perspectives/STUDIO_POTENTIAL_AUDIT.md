# STUDIO POTENTIAL AUDIT: every facet graded on the Studio's THEORETICAL ceiling

Audit date: 2026-07-03. Companion to `POTENTIAL_AUDIT.md` (which graded the program against its ideology on
the laptop) and `AXIS_CEILING_RESULT.md` (which proved the laptop ceiling, ~6.75/10). This document answers
the inverse question: if the program moves to the Mac Studio and EVERY pre-registered bet is pursued to its
end, what is the maximum each facet can theoretically reach, and what wall stops it going higher even there.
Maximal by mandate, honest by construction: every ceiling below 10 must name its wall, every ceiling claimed
must name the bet that has to convert, and every bet inherits the falsification engine (a faked score is a 0).
House style: no em or en dashes.

## 0. The machine, honestly (M1 Ultra, 128 GB unified memory, 8 TB SSD)

The box is an M1 Ultra Mac Studio: 20-core CPU (16 performance + 4 efficiency), 48 to 64-core GPU, 32-core
Neural Engine, 800 GB/s unified memory bandwidth, 128 GB unified memory, 8 TB SSD. Against the M3 Pro laptop
(11-core, ~150 GB/s, 18 GB, and a disk that lived at its own kill-switch floor):

- CPU: per-core speed is LOWER (M1 generation, roughly 25 to 35 percent slower single-thread than M3), but
  there are 16 performance cores. The laptop's proven encode floor was 21 s/clip on ONE CPU worker
  (`STUDIO_HANDOFF.md`). Fourteen to sixteen parallel workers put the aggregate at roughly 2 s/clip even if
  MPS never works: 10,000 clips in ~6 hours, 100,000 clips in ~2.5 days. Wall-clock encode stops being the
  binding constraint; curation quality becomes it.
- GPU/MPS: the M3 Pro hit `RuntimeError: Invalid buffer size` on 64-frame 256px V-JEPA 2 ViT-L attention.
  `STUDIO_HANDOFF.md` recorded it as a per-buffer ceiling that more RAM would not fix; that claim is
  UNVERIFIED on a 128 GB device (Metal's max buffer length scales with unified memory). Day-0 microbenchmark
  decides it, never assume either way. If MPS works, encode drops toward 0.5 to 1.5 s/clip and the GPU also
  carries the local-VLM caption pass. If it does not, the 16-worker CPU path above already suffices.
- Memory: 128 GB holds EVERY staged encoder simultaneously (V-JEPA ViT-L/H/g, DINOv2-L, VideoMAEv2, wav2vec2,
  a Qwen-class 7B captioner: ~30 GB total at fp16) plus a 100k-task stream plus bootstrap buffers, no paging.
  The 800 GB/s bandwidth means the 30-seed readout sweeps and permutation floors that were minutes-per-cell
  on the laptop become in-memory batch jobs.
- Disk: 8 TB retires the disk kill-switch as a daily constraint (the laptop genuinely tripped its 60 GB floor
  mid-session). It also makes DENSE-token caches a first-class object for the first time: V-JEPA dense
  latents at ~8192 tokens x 1024 dim x fp16 are ~17 MB/clip, so a 100,000-clip dense cache is ~1.7 TB,
  comfortably inside the `studio-m1ultra` usable envelope and flatly impossible before. Pooled caches are
  ~2 KB/clip and become free.
- Enforced envelope: `src/mop/studio/profiles.py` profile `studio-m1ultra` (8 TB total, 7.2 TB usable,
  min free 250 GB, tiers C+E, week-scale wall clock). The old `studio-1tb` profile remains valid as the
  conservative envelope every earlier doc references.

The honest summary: the Studio removes the three resource walls the laptop proved (encode throughput, memory
residency, disk), and removes NONE of the two structural walls (the frozen encoder, the absence of an
interactive environment). Every grade below follows from that split.

## 1. Scoring rule

Each facet gets: the laptop-proven score (from `AXIS_CEILING_RESULT.md`), the STUDIO THEORETICAL CEILING (the
score if every named bet converts), the bet(s) that must convert, and the wall above the ceiling. Theoretical
means exactly that: the program's own history says most bets null. A null at Studio scale is TERMINAL
knowledge (there is no bigger box behind this one except rented CUDA and Process C), so every null here is a
program-routing decision, which is itself value. The expected value sits below the ceiling; the ceiling is
what we aim at.

## 2. The four north-star axes

### Falsification: 10 -> 10 (held and COMPLETED). Bet: B5.

Ceiling case. The method axis is already maxed on the laptop (vacuous frozen-random gate retired, four
over-claims killed by adversarial verifiers, meta-control audit passed). The Studio completes the two named
caveats: the B5 multi-seed re-encode of the 29-clip headline (the one test the laptop could not re-run, only
replace with the 200-clip bootstrap), and 30-seed protocols as the DEFAULT for every claim rather than a
budgeted luxury. Statistical power stops being a caveat anywhere in the corpus.
The wall above. None. 10 is the top, and the risk on this axis is EROSION under scale pressure (skipping the
adversarial pass because a run took two days), not ceiling. The standing rule transfers unchanged: every
candidate positive gets an independent adversarial verifier before it enters any doc.

### Abstraction: 6 -> 9. Bets: DR1 (both gates), the local-VLM caption arm, causal/part-whole probes.

Ceiling case. The laptop proved the substrate carries substrate-invariant compositional/analogical structure
(four controlled wins) and mapped the exact walls: 3-factor compositionality memorizes conjunctions, and
vision-to-language transfer fails ONLY because the label-free pixel-text is shape-blind (a content limit, not
a substrate limit). DR1 attacks both with real video: many genuine composable factors (object identity,
count, relation, action) instead of two, and a PAIRED language perspective built from real captions. The 128
GB unlock nobody priced in: a LOCAL 7B-class VLM captioner runs resident on this box, so the caption arm is
label-free, license-free, and re-runnable at cache scale, with the `dr1_curate_bound_video.py` acceptance
gate (attribute must be probe-recoverable from the caption BEFORE encode is spent) already in the repo. If
systematicity extends to 3+ real factors (gate b1), cross-family vision-language abstraction holds on
captions that carry the attribute (gate b2), and causal/part-whole probes register on real action content
(gate b3), the ideology's core claim (thought as an ecology of perspectives sharing abstract structure) is
demonstrated on real content across families. That is a 9.
The wall above. READOUT versus FORMATION. A frozen substrate can only ever REVEAL abstraction already latent
in its weights; it cannot form a new abstraction under experience. Band 10 (abstraction that visibly forms)
requires a trainable encoder (Process C) or is out of scope. Also gate b3's causal probes are bounded by
observational video: full interventional causality needs an environment (see facet 10).

### Density: 6 -> 9. Bets: natural-complementarity at scale, at-scale retention/byte and adaptation/update.

Ceiling case. The laptop won the thesis mechanism (matched-compute heterogeneous mixture beats every
homogeneous control, 10/10 seeds) but only on a CONSTRUCTED task, and proved the precondition: the win
appears iff a required factor sharply separates the readers. The Studio test is whether that win arises
NATURALLY: real factor structure (objects, motion, language, audio genuinely separate readers), a
data-driven mixture at n in the tens of thousands (the laptop's n=140 could not self-discover the
factorization; scale is exactly what discovery needs), four-plus perspectives on identical referents, and
the two never-scored sub-axes (retention/byte, adaptation/update) finally scored at DR1 scale. The full
capability-per-FLOP frontier across a 6-encoder atlas turns the density claim from one point into a CURVE.
If the natural win converts sign-stable at matched compute, the constructed-task caveat is removed and the
core MoP thesis is proven in the wild: 9.
The wall above. The precondition itself. If natural factor structure does not sharply separate readers, no
scale fixes it (the laptop's 40-seed re-test already hints at this), and density lands at a proven honest 6
to 7 instead. And band 10 (reasoning-per-FLOP at frontier scale) is out of scope for any single box.

### Moldability: 5 -> 8. Bets: PR9 at scale, then the Process C sanctioned pilot.

Ceiling case. The laptop proved the wall precisely: the frozen representation forces orthogonal factors to
either isolate (nothing to mold) or compete (only generic anti-forgetting repair), and well-tuned SGD showed
no plasticity loss on the SHORT real stream. PR9 at Studio scale runs the validated CBP mechanism on a REAL
long stream that can actually induce loss (millions of steps, thousands of tasks, resident in RAM), with the
validated plasticity-loss certificate guarding against the mistuned-baseline artifact the laptop caught
twice. Either outcome moves the program: a win is the first substrate-touching plasticity positive (score
7), a certified tie fires the kill-switch and LICENSES Process C. Process C is the real lever: the sanctioned
1-10M object-centric module on frozen dense tokens (dense caches now exist, facet 8), where the
representation itself can remold so factors need not compete. A 1-10M module trains in hours on this GPU at
30 seeds. If it beats the frozen shell on the exact forgetting stream the laptop built, with the matched
random-init control, that is the first genuine moldability win: 8.
The wall above. Doctrine and physics. The param cap is 1-10M by doctrine (15_custom_model_skepticism), and
child-brain moldability (bands 9-10, a substrate that reorganizes wholesale under experience) is a
from-scratch encoder, ~60 GPU-years, permanently off this box. 8 is the honest maximum of the sanctioned
path.

## 3. The expanded facets (the audit surface the laptop could not even expose)

### Facet 5, semantic coverage: ~2 -> 9. Bet: DR1 + paired language + audio.
86 positions in `SEMANTIC_POSITIONS.md`; the laptop could address ~15, and 51 are needs-new-cache. DR1 plus
the caption arm plus a wav2vec2 audio arm unblocks ~70, including every SEM-LANG cluster position. Ceiling 9
not 10: the SEM-PLAS positions that require un-freezing stay gated on Process C, and a handful require an
environment.

### Facet 6, perspective plurality: 2 (shape, color, on one cache) -> 9. Bet: DR1 multi-arm encode.
The ideology says ecology of perspectives; the laptop instrumented two visual slots plus a shape-blind
pixel-text. The Studio target: vision-static, vision-motion, language (real captions), audio, code (DSL via
`verifier_exec`), math (numerosity) on IDENTICAL referents, each with its matched random-init control. Not
10 because code/math perspectives remain synthetic-task-bound rather than natural content.

### Facet 7, substrate atlas / encoder generality: 3 encoders -> 9. Bet: B4 atlas + ex12 scale falsifier.
Every claim so far rides ViT-L (+DINOv2 partially). The Studio holds ViT-L, ViT-H, ViT-g, DINOv2-L,
VideoMAEv2, and dense V-JEPA 2.1 the day Meta ships weights, all resident at once. The ex12 encoder-scale
falsifier (does bigger frozen perception raise decodability) becomes a real curve, and every abstraction win
gets re-tested for substrate-invariance across the full atlas. Not 10: the atlas is still all
self-supervised vision plus one text model; a truly heterogeneous atlas (contrastive, supervised, multimodal
encoders) is a licensing/curation project beyond the current registry.

### Facet 8, dense-token instrumentation: 0 -> 9. Bet: the 8 TB dense cache.
Never possible before at all. A 100k-clip dense cache (~1.7 TB) unlocks the entire deferred dense lane: DR3
latent scratchpad (the highest-value substrate-bound probe per the manifest), CM3 dense-vs-pooled
compositional isolation, DR14 dropped-channel, ex9 slot attention dense arm. Not 10: dense 2.1 weights do
not exist publicly yet (upstream gate, flagged for periodic re-check).

### Facet 9, statistical power: 4 -> 10. Bet: none, this is pure compute.
Single-split p=0.0285 headlines, 63.7 percent bootstrap survival, n=64 caches: all retired. 30 seeds, 200+
clips per cell, bootstrap CIs on every number in the corpus as the default gate. This is the one facet where
the theoretical ceiling is simply purchasable.

### Facet 10, environment / interactive action: 3 -> 4 even here. Bet: none on this box.
Named so the maximal framing cannot launder it: ex2 latent planning, interventional causality, and the
autotelic capstone need an interactive environment harness that does not exist in the repo. The Studio makes
building one FEASIBLE (facet 11 buys the compute) but the gap is implementation, not hardware, and doctrine
routes the heavy version to rented CUDA (Tier R). Grading it 9 because the box is big would be exactly the
mislabeling the laptop audit caught in the old studio-gated table.

### Facet 11, autonomy / conveyor throughput: 5 -> 9. Bet: week-scale unattended queue.
The 12-stage `studio_pipeline.py` conveyor is validated end-to-end (12/12 on the laptop rehearsal). The
Studio version runs multi-day gated queues (tiers C+E, kill-switches held at 250 GB free, resumable
checkpoints, the adversarial-verifier pass wired in as a stage, full gates plus commit at every wave
boundary). The program's cadence changes from one lever per session to one WAVE per day. Not 10: unattended
means the honesty machinery must also be unattended, and the verifier-in-the-loop stage is built but not yet
battle-tested at that cadence.

## 4. The scoreboard

| Facet | Laptop proven | Studio theoretical | The bet that must convert | The wall above the ceiling |
|---|---:|---:|---|---|
| Falsification | 10 | 10 (completed) | B5 multi-seed re-encode | none; erosion risk only |
| Abstraction | 6 | 9 | DR1 gates b1/b2/b3 + VLM captions | readout vs formation (frozen) |
| Density | 6 | 9 | natural complementarity at scale | the precondition itself |
| Moldability | 5 | 8 | PR9 at scale, then Process C pilot | 1-10M doctrine cap; from-scratch is off-box |
| Semantic coverage | ~2 | 9 | DR1 + language + audio arms | SEM-PLAS needs un-freezing; env positions |
| Perspective plurality | 2 | 9 | multi-arm encode on identical referents | code/math stay synthetic |
| Substrate atlas | 3 | 9 | B4 + ex12 curve | all-SSL-vision atlas |
| Dense instrumentation | 0 | 9 | 1.7 TB dense cache | 2.1 dense weights upstream |
| Statistical power | 4 | 10 | none (purchasable) | none |
| Environment/action | 3 | 4 | none on this box | needs an env harness (Tier R) |
| Autonomy/conveyor | 5 | 9 | week-scale gated queue | unattended honesty untested |

North-star overall (mean of the four axes, the comparable number): **~9.0 theoretical** against the proven
laptop 6.75. Honest expected value given the program's own base rates: sitting near 8, with the gap being
exactly the bets that can null (natural density, PR9, gate b2). The two walls that survive even a perfect
Studio run: the frozen substrate cannot FORM (only reveal), and there is no interactive environment. Both
already have their named successor (Process C, Tier R).

## 5. Priority order (unchanged from EXPAND_PHASE_PLAN.md, now with the new facets slotted in)

1. DR1 (Tracks B and C ride the same cache), with the caption acceptance gate and the local-VLM arm.
   Facets it moves: abstraction, density, coverage, plurality. The single highest-leverage artifact.
2. PR9 at scale (Track A), certificate-guarded. Decides Process C licensing either way.
3. The dense cache + atlas encode pass (facets 7, 8) alongside DR1 (same conveyor, same clips).
4. Process C pilot IF licensed by PR9's kill-switch or DR1's b1 wall being representational.
5. B5 multi-seed + 30-seed retrofits (facet 9) LAST, per the audit's standing rule: never refine an owned
   number while an unbuilt instrument blocks an axis.

The single sentence: the Studio's theoretical ceiling is ~9.0/10 with six purchasable facets and two proven
walls, and the program is designed so that even the nulls on that path are terminal, citable knowledge.

## 6. Part 2: the Studio as a NEW instrument, not the laptop's executor

Everything above grades the Studio against the program the LAPTOP could conceive: the inherited backlog (DR1,
PR9, B4, B5) run bigger and cleaner. That is necessary and it is not enough, because the laptop's doctrine
was never chosen on the merits: cached-latent-first, the live-encoder ban, the 10 GB download cap, the
two-slot synthetic clipset, and the six-perspective plan were all DERIVED from 21 s/clip, 18 GB of memory,
and a disk living at its kill-switch floor. A device with none of those constraints licenses experiment
CLASSES the program has never priced, not just larger versions of the ones it has. This part audits the box
de novo. Same rules: every ceiling names its wall, every bet inherits the falsification engine, and none of
these lanes may jump the spine (section 7) just because they are new.

### Facet 12, world-model rollouts, the predictor half of V-JEPA 2: 0 -> 8. Bet: rollout fidelity.

Ceiling case. The entire corpus to date uses V-JEPA 2's ENCODER and discards its PREDICTOR, which is a
learned latent-space simulator of video dynamics. 128 GB holds encoder plus predictor resident, so latent
ROLLOUTS become a first-class probe: counterfactual and interventional abstraction WITHOUT building an
environment (roll the latent forward under alternative continuations and probe the divergence), the ex2
latent-planning precursor (plan by candidate-rollout search, all rollouts counted in FLOPs, exactly as
EXECUTION_MANIFEST line 171 specifies), and DR7's latent chain-of-thought harness. The acceptance test is
already in the repo: `scripts/mop_dr13_horizon_limit.py` (rollout error compounding with horizon) runs
against REAL predictor rollouts instead of synthetic transitions, cheaply, before anything rides the lane.
This is the single largest untouched capability the program already owns the weights for.
The wall above. The predictor is frozen too, and its rollout fidelity on our content is unmeasured. If
compounding error kills usable horizon at 2 to 3 steps, the lane is bounded to one-step counterfactuals, and
that bound is itself the honest ex2/DR13 verdict at real scale.

### Facet 13, closed-loop and active experiments, the live-encoder ban dissolves: 3 -> 7. Bet: s/clip.

Ceiling case. The live-encoder ban exists because encode cost 21 s/clip on one core; at the Studio's measured
1 to 2 s/clip the ban is obsolete DOCTRINE, not physics. The learner can finally choose its own data:
curiosity-driven data selection on real video (e5's rollout arm, disabled in `campaign/run_queue.yaml`
purely for resources), learning-progress self-curriculum (d5) on real latents, streaming encode where the
data distribution responds to the learner's state. Every experiment in the corpus where the data was frozen
BECAUSE encoding was slow gets a live arm.
The wall above. True interactive embodiment (a robot, a game environment with contingent physics) is still a
harness that does not exist and doctrine routes its heavy version to Tier R. 7 is closed-loop over real
video and self-selected data, not embodiment.

### Facet 14, real-corpora residency: 0 -> 9. Bet: licensing labor.

Ceiling case. `registry/datasets.yaml` already catalogs Something-Something V2 (~220k clips of ACTIONS on
objects, the exact bound-attribute causal content gate b3 wants), Kinetics-700, EPIC-KITCHENS-100, and Ego4D,
each with a license ledger and acquisition path, and the laptop could host NONE of them (10 GB cap). The 8 TB
box hosts SSv2 outright, large EPIC/Kinetics subsets, and signed-license Ego4D slices simultaneously
(`studio-m1ultra` allows manual auth and 1.5 TB per source). DR1 stops meaning "curate one cache" and starts
meaning "hosted corpora from which MANY caches are curated," with the caption acceptance gate applied per
attribute. Every abstraction and density gate gets re-posed on real human-scale content.
The wall above. License terms and annotation quality, not hardware: Kinetics bulk video needs a licensed
mirror, Ego4D needs the signed agreement, and none of that labor parallelizes onto the GPU.

### Facet 15, the perspective ecology at full width: 2 -> 9. Bet: per-perspective controls.

Ceiling case. The ideology says ecology; the plan says six perspectives; the box supports TEN-plus resident
at once: the planned six, plus segmentation (a SAM-class model as an OBJECT-CENTRIC perspective, the most
direct binding probe the program could have), depth, optical flow, audio encoders on real audio-visual
corpora (AudioSet metadata is already in the planner), and a resident 7 to 14B LLM as a SEMANTIC perspective
over captions. The mixture-of-perspectives thesis gets tested at the width its name claims, on identical
referents, with the density frontier plotted across the full ecology.
The wall above. Discipline, twice over. Each new perspective needs its matched random-init control and its
A6 residualization pass (the laptop proved apparent cross-perspective structure is usually nuisance-carried).
And supervised perspectives (SAM, depth models) change the all-self-supervised substrate claim, so they get
flagged per-perspective in every verdict. The dead test-time-compute branch STAYS dead: an LLM as a
representation source is licensed, verifier-guided iteration is not.

### Facet 16, the developmental long-run: 0 -> 8. Bet: the daemon holds honest for weeks.

Ceiling case. Moldability's ideology names DEVELOPMENTAL acquisition, and development happens over weeks,
not inside a 90-minute wall-clock cap. An always-on box runs a persistent continual-learning daemon: a shell
living on a real video stream for weeks, periodic probe batteries, the plasticity certificate sampled on a
schedule, checkpoints every 30 minutes. PR9's stream stops being simulated; it is the box's actual life.
This is the REGIME the laptop could not even sample, and it feeds every plasticity and forgetting question
the corpus has.
The wall above. The frozen encoder still caps WHAT can be molded (this facet unlocks the regime, Process C
remains the mechanism), and weeks-scale unattended operation is where honesty machinery erodes; the
adversarial pass must be wired into the daemon itself, not left to session hygiene.

### Facet 17, trainable capacity above the doctrine cap: a DECISION, deliberately not graded.

Physically, this box trains 50 to 100M-parameter modules on cached latents in days (MPS training throughput
is unspectacular but sufficient at that scale, and 128 GB removes every activation-memory excuse). Doctrine
caps the sanctioned Process C pilot at 1 to 10M. Do not sneak past that: the cap is a reasoned position
(`15_custom_model_skepticism.md`), not a hardware artifact. But the audit's job is to price the decision: if
the 1-10M pilot returns an informative result either way, the 10-100M band is a doctrine AMENDMENT to argue
with evidence in hand, on a box that can actually execute it. Named here so nobody discovers it accidentally
mid-wave.

## 7. What Part 2 changes, and what it must not change

Supersessions to the Part 1 table: environment/action rises 4 -> 7 (facet 12 gives interventional probes
without an environment, facet 13 gives closed-loop data; only embodiment stays walled at Tier R), and
abstraction's gate b3 becomes OVERDETERMINED (real action corpora and predictor counterfactuals attack the
causal wall from two independent sides). North-star overall theoretical moves from ~9.0 to ~9.3; the last
0.7 is exactly the three named successors: frozen formation (Process C amendment, facet 17), embodiment
(Tier R), from-scratch training (off-box by doctrine).

What must not change: the SPINE. DR1 and PR9 stay first, because every Part 2 lane rides their artifacts
rather than competing with them: facet 14's corpora ARE DR1's curation source, facet 12's predictor encodes
the same clips, facet 15's perspectives encode the same referents, facets 13 and 16 inherit PR9's stream
infrastructure. Exactly ONE Part 2 item is licensed to run early, because it is cheap and gates a whole
lane: the predictor-fidelity test (DR13 on real rollouts, facet 12's acceptance gate), a single afternoon
that decides whether the rollout lane exists. Everything else slots in behind the spine per section 5.

The upgraded single sentence: the Studio is not the laptop's executor but the first device on which the
program's doctrine itself (cached-only, live-ban, two slots, six perspectives, 10M params) stops being
forced, so the audit's standing job there is to RE-DERIVE the doctrine from the new constraints, wave by
wave, instead of inheriting the old ones unexamined.
