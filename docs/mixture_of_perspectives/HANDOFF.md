# HANDOFF: continue Mixture of Perspectives in a fresh context

Read this first. It is the cold-start brief for continuing the Mixture of Perspectives (MoP) program in a
new chat. Paths are relative to the repo root. House style: no em or en dashes anywhere. Never attribute
Claude in git (no Co-Authored-By, no Generated-with footer). Today is 2026-07-02.

## 0. Orientation in one paragraph (honest)

MoP is a research program that mines a FROZEN V-JEPA 2 video encoder (the substrate) with a tiny trainable
shell, scoring capability DENSITY not peak, with a falsification engine (every claim beats a non-vacuous
control: random-init same-arch encoder at matched resolution, NEVER a square latent projection; matched
compute; tuned baseline; noisy-TV; seed stability). A full laptop maximization run plus an adversarial
audit are DONE. The honest state: the substrate is genuinely special (pretraining beats a matched-arch
random-init ViT, single split p=0.0285) and factors shape from color off ceiling, and reasoning modes make
decorrelated errors (PR1 licenses a router). But almost every MECHANISM claim is a clean null, two headline
positives (al2 shared-code, ws2 structure-beats-capacity) were re-graded to NULL against their own controls,
the test-time-compute lane is DEAD even with a perfect verifier (kill-switch fired), and the potential audit
scored the program 3.0/10 on reaching its ideology: moldability is false-by-construction on a frozen
substrate, multi-perspective thought is instrumented at only 2 visual slots (shape, color), and density was
tested once and nulled. The program is one decisive build (real bound-attribute video + a paired language
cache) away from becoming falsifiable on its own ideology, and one comfortable quarter away from being a
permanent negative-mapping machine on two slots. The next chat's job is to build that instrument, not to
refine numbers already owned.

## 1. Name and folder

The program is Mixture of Perspectives (MoP), package `mop`. The repo folder is `mop` (renamed from `brain`;
if a path still says `brain`, the folder mv is the last pending step, see section 7). The old name "Mixture
of Thinking / MoT" was dropped because it collides with published prior art (Zheng 2025 Mixture-of-Thought,
Yue 2023 Mixture of Thoughts, Mixture-of-Transformers reuses MoT). "Apperception" was also rejected (Richard
Evans' Apperception Engine). Framing: perception = the frozen substrate; perspectives = the plural modes
being mixed; apperception = the shell's binding act (a concept in the docs, not the name).

## 2. What is PROVEN, NULL, and DEAD (the verdict ledger)

PROVEN (real encoder, non-vacuous control, but single-split where noted):
- Substrate is special because of pretraining: real V-JEPA 0.517 vs random-init same-arch ViT-L 0.241 at
  matched 256px, p=0.0285 (15/29 vs 7/29, ONE clip from ambiguity, not yet multi-seeded). Re-audit HOLD but
  FRAGILE: a test-clip bootstrap keeps p<0.05 in only 63.7% of resamples (one-clip swing -> p=0.088);
  direction corroborated by the 200-clip caches (delta CI lo 0.504). Multi-seeding is Studio B5, still LAST.
- Compositional factoring off ceiling: held-out (shape,color) 0.725 = seen 0.708 (within-arm property).
- PR1 router-licensing: heterogeneous oracle gain 0.155 vs homogeneous mean-copy 0.118 + spread (an
  EXISTENCE upper bound, not a trained-router win). NOTE: a TRAINED router on the real cache does NOT reach
  it (router 0.860 < best-single 0.870 < homo-bank 0.876, both gates fail; density-mechanism null, see below).
- at1 cross-substrate invariance: re-audit HARDEN (two survivors add independent evidence, per-clip
  correctness phi 0.329, not one signal double-counted).

NULL / DEMOTED (do not carry these forward as positives):
- at3 temporal currency: DEMOTED (was PROVEN). The motion_dir4 (+0.200) and speed2 (+0.245) full-vs-single
  edge is reading the INJECTED (vx,vy) draw, not integrating time: under the strong nonlinear partial-out
  (r,vx,vy,vx^2,vy^2,|v|,sin/cos angle) both collapse to chance (shrink 100% / 96.6%). See survivor_reaudit.
- A6 cross-modal shared-code: vision<->text alignment SURVIVES removing shape+color but COLLAPSES removing
  the 6 nuisance factors (0 stable survivors at minus_nuisance/minus_all), survived 3 controls. Carrier =
  spatiotemporal NUISANCE geometry, NOT the semantic abstraction. Shape-axis bet = BOUNDING NULL (shapecap
  carries shape at 0.617, killswitch did not fire, yet vision->shapecap dies at minus_all). See A6_RESULT.md.
  This supersedes the older al2 summary: cross-modal alignment is not absent, it is nuisance-carried.
- al2 shared-code: cross-modal alignment does not survive a kNN-topology permutation null AS SEMANTIC
  code; same-modality vision pairs align (propped by two V-JEPA columns). Precondition NOT met; see A6.
- ws2 structure-beats-capacity: fails its own dual acc-AND-NLL contract (was an OR-rule over-claim).
- density mechanism (trained router, real cache, SHAPE-ALONE): NULL. Router loses to a tuned best-single
  reader AND a compute-matched homogeneous bank (both gates fail). But this is the REDUNDANT-reader regime;
  Round 2 found the WIN on a COMPLEMENTARY task (see MECHANISM WINS below). See LAPTOP_LANES_RESULT.md.
- pr7 delta-rule: NULL, trails the Hebbian floor 10/10 seeds (Hebbian fast-store itself beats slow-only
  +0.029, a modest real signal, the only plasticity flicker).
- The 24-null reasoning lane, and the 4 recalibrated degenerates (mt5/al1/dr12/ws3), all NULL.

MECHANISM WINS (axis-ceiling push, 2026-07-02/03; see AXIS_CEILING_RESULT.md):
- DENSITY, the core MoP thesis (Round 2, the strongest result): on a COMPLEMENTARY composite task (color x
  motion_dir), a matched-FLOP matched-param heterogeneous FACTORED mixture beats the best single reader
  (+0.170), every homogeneous bank (+0.073 to +0.253), 10/10 seeds, mechanistic guard passed. Negative control
  (shape x motion, redundant readers) correctly TIES: falsifiable precondition = a mixture wins iff a required
  factor sharply separates the readers. runs/mot/density_mixture_win.json. FIRST matched-compute mechanism win.
- CBP plasticity repair: continual-backprop repairs provably-induced plasticity loss on the SYNTHETIC drift
  stream (SGD gap 0.513 -> ~0, 8/8 seeds; runs/mot/cbp_plasticity_repair.json). BUT it does NOT transfer to the
  REAL substrate (Round 2): well-tuned plain SGD already retains full plasticity on the real V-JEPA-latent
  stream, nothing to repair (real-substrate plasticity = NULL). So moldability stays 5, frozen-capped.
- Abstraction: count/parity decode from image/text/code/math even after area is dissociated by design (Round 2,
  stronger encoders), but the cross-perspective abstract-code bet is a BOUNDING NULL: dissociating area moves
  the confound to perimeter/spacing and a random-init encoder reproduces the alignment. Synthetic count is
  inextricably geometry-confounded; needs DR1 real video.
- FALSIFICATION reaches 10: the vacuous frozen-random gate is retired (Round 2, applied + gates green); 3
  experiment verdicts flip, each validated correct, no manufactured positive. runs/mot/falsification_vacuous_fix.diff.

- ABSTRACTION systematicity WIN (Round 3): analogical/compositional abstraction on REAL V-JEPA latents.
  A shape offset transfers across color contexts (analogy 0.336, random-init 0.0, perm p=0.000) and shape
  generalizes to NOVEL shape-color conjunctions (0.730 vs untrained-ViT collapse 0.055). Confound-corrected on
  the shape axis (color is a pixel-statistic the untrained net wins). runs/mot/abstraction_systematicity.json.
  Abstraction 3 -> 4. Held to 4: real latents but SYNTHETIC content, above 4 needs DR1 real video.

AXIS-CEILING SCORES (honest laptop maxima across 5 rounds, POTENTIAL_AUDIT addendum 1c holds current):
falsification 6->10, abstraction 2->6, density 3->6, moldability 2->5. Overall 3.0 -> ~6.75. SIX genuine
positives now (mixture-of-perspectives density win + FOUR abstraction wins [systematicity, pairwise + 3-way
cross-substrate analogy] + synthetic-stream plasticity repair) where the audit found ZERO mechanism wins.
Abstraction CLIMBED 2->3->4->5->6 (a controlled win per round), boundary now MAPPED (3-factor compositionality
breaks; vision->language fails, text is shape-blind). Moldability (5) PROVEN frozen-capped: the joint-training
ORACLE hits chance (0.300 vs 0.328) on the substrate-specific forgetting stream, frozen features cannot serve
2 orthogonal tasks in one head. Adversarial verifiers killed FOUR over-claims (R2 mistuned-baseline CBP, R3
LR-confound developmental, R3b operand-confound lang-math, R4 developmental). Moldability + abstraction-beyond-6
need the Studio (PR9 real stream, DR1 real video) or un-freezing (Process C). Every ceiling now has a
MECHANISTIC reason, not an assumption. See AXIS_CEILING_RESULT.md section 8.

INSTRUMENTATION / STUDIO-READY (built and validated on the laptop, not a science claim yet):
- Density frontier (FIRST plotted): capability/FLOP frontier {reactive, sparse} with the routed mixture
  DOMINATED; capability/param frontier DINOv2-dominant (0.861 @ 384d readout, ~3x V-JEPA/param). Two axes
  computable now; retention/byte and adaptation/update are Studio-only, unfaked. See LAPTOP_LANES_RESULT.md.
- Plasticity-loss certificate: VALIDATED (fires on drift gap +0.513 CI[0.498,0.528], dead units 0->0.75;
  quiet on stationary gap ~0). The instrument Studio PR9 needs, de-risked and turnkey. Moldability score
  itself does NOT move on the laptop (cannot induce Studio-scale loss).

DEAD (kill-switch fired, branch retired):
- Test-time compute at this substrate: even with a PERFECT executable DSL oracle verifier on a
  difficulty-graded task, verifier-guided iteration carries no usable correction signal (the shuffled
  verifier control beats the real one on the hard bin). The custom test-time-compute branch is retired.

## 3. Document map (read in this order)

- `docs/mixture_of_perspectives/HANDOFF.md` (this file).
- `docs/mixture_of_perspectives/SCAFFOLD.md` (the M3 Pro process vs Studio process, sequenced, verified
  disk state; the operational to-do).
- `docs/mixture_of_perspectives/POTENTIAL_AUDIT.md` (the 3.0/10 scorecard and the re-ordered action list).
- `docs/mixture_of_perspectives/M3PRO_RUN_REPORT.md` (the run, corrected after the audit re-grade).
- `docs/mixture_of_perspectives/SEMANTIC_POSITIONS.md` (86 positions on thought, with a coverage/gap map;
  the yardstick for the multi-perspective ideology).
- `docs/mixture_of_perspectives/DEEP_RESEARCH_2026_07.md` (the literature position; why the reasoning nulls
  are regime-correct; PR9 as the one certified plasticity baseline-beater).
- `docs/mixture_of_perspectives/MIXTURE_OF_THINKING.md` (the master thesis/definition; H1-H5 hypotheses).
- `docs/mixture_of_perspectives/EXECUTION_MANIFEST.md` (the original pre-run plan; superseded by SCAFFOLD.md).
- `docs/mixture_of_perspectives/11_experiment_registry.md` (MP/DR/PR/WS/AT/AL/CM registry).
- `DOCTRINE_SYNTHESIS.md` (the pre-studio corpus doctrine, sections 3d-3e = the substrate results).
- Results: `runs/mot/` (all mop experiment JSONs, incl. the *_regrade / *_seeds10 / process_a_report),
  `runs/pre_studio/` (the substrate result JSONs + interpreted variants).

## 4. The next work, sequenced (from SCAFFOLD.md, post-Process-A)

Process A (M3 Pro) is essentially done: al2/ws2 re-grades (confirmed NULL), the D3 hardness gradient +
executable verifier build (`src/mop/diagnostics/hardness.py`, `src/mop/shell/verifier_exec.py`,
`scripts/mop_d3_verifier_reasoning.py`; kill-switch fired), pr7 delta-rule (NULL), the 4 degenerate
recalibrations (NULL), and the pre-authored Studio scripts (`scripts/studio/`). The ONE remaining Process A
item is A6:

- A6 (M3 Pro, do this first in the next chat): the PAIRED vision+text cache on IDENTICAL referents. The
  Qwen cache is a LABEL-FREE PIXEL-DERIVED textification (a 4x4 palette-color grid + brightest-cell
  position), already paired clip-for-clip to the same vision clips; what it lacks is SHAPE, which decodes
  at chance from cheap label-free features on this clipset (shape is unverbalizable by such features here),
  so it cannot yet test the language-independent-abstraction north star. Extend the pixel-derived caption
  per clip toward a shape-bearing descriptor and cache the small-LLM hidden states on THOSE, paired to
  `data/cache/vjepa2_vitl_nuisance` (the real vision cache), then re-run al2's corrected
  (topology-permutation) metric on the vision<->text pair. This is the
  smallest instrument for the SEM-LANG cluster and the ONLY laptop-runnable probe of the multi-perspective
  ideology. Fast (short-text LLM pass, one model at a time).

Process B (Studio, mandatory; scripts pre-authored in `scripts/studio/`, RAM-guarded to refuse the laptop):
1. B1 / DR1: the non-additive bound-attribute natural-VIDEO cache with count and relation slots, plus the
   paired vision+text pass at scale. THE unblocker of GATE C1 and ~70 semantic positions. Everything hinges
   on this; it is the difference between having and not having a science on the multi-perspective ideology.
2. B2 / PR9: continual-backprop on a LONG real-latent stream with a plasticity-loss certificate. The only
   plasticity mechanism certified to beat a tuned baseline; it resolves moldability either way (win = first
   substrate-touching plasticity positive; tie = moldability honestly dead at a frozen substrate).
3. B3 / CM1: a small real bound-attribute video batch through the frozen encoder to make GATE C1 falsifiable.
4. B4: the full multi-encoder atlas + dense V-JEPA 2.1 (8192-token) latents.
5. B5 (LAST, not first): multi-seed the substrate headline p-value and settle dr2 sparse with 30 runs.

The audit's rule, do not violate it: the free re-grades and the verifier build come before Studio time, DR1
comes before multi-seeding. Multi-seeding a p-value already owned while DR1 sits unbuilt is the failure mode.

## 5. Decision gates / kill-switches

- A6/al2: if cross-modal (vision<->text) alignment fails even a topology null, the shared-code precondition
  is not met and the workspace line is bounded to same-modality fusion until B1/B4.
- D3 verifier: FIRED. Test-time compute is dead at this substrate; do not rebuild that branch.
- B2/PR9: if it ties on a stream that provably induced plasticity loss, moldability is dead at a frozen
  substrate, and the case for un-freezing (a trainable-encoder arm, the custom-model line) becomes real.
- Custom-model line (Process C, wider box): stays UNLICENSED until B3/B4 show every frozen substrate hits
  the same wall. Today's evidence tilts AWAY (the substrate is special). A from-scratch V-JEPA-scale model
  is a ~60 GPU-year moonshot outside doctrine; the only sanctioned pilot is a 1-10M object-centric module
  on frozen tokens, and only if the gates force it.

## 6. How to run and verify

- Python: `.venv/bin/python`. `import mop` REQUIRES `PYTHONPATH=/Users/scammermike/Downloads/mop/src`
  (the venv has NO `pip` and NO editable install; there is no `__editable__` finder, so `src/mop` is not on
  the path without it). Scripts self-insert the repo root for `import scripts.*`. Set `OMP_NUM_THREADS=4`.
  Canonical recipe: `PYTHONPATH=/Users/scammermike/Downloads/mop/src OMP_NUM_THREADS=4
  .venv/bin/python scripts/<x>.py`.
- Gates (run before every commit): `.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
  src/mop && .venv/bin/python -m pytest -q && .venv/bin/python scripts/check_docs.py && .venv/bin/python
  scripts/acceptance.py`. All are currently green (pytest full suite, acceptance 10/10).
- New docs must be added to the ledger in `scripts/check_docs.py` (CANONICAL_MD list) or check_docs fails.
- Experiments follow the contract (id/metric/baseline/ablation/null_hypothesis/tier, run()->dict with
  null_supported); preregister the null and verdict thresholds IN CODE before running; a tie is a null,
  never tuned toward a positive. Cached-latent caches are in `data/cache/` (LatentStore now reads both the
  native and the WP-01 provenance layout).

## 7. Pending: the folder rename (brain -> mop)

The package, scripts, configs, labels, docs folder, and prose are all renamed to mop and committed. The
only remaining step is renaming the repo FOLDER on disk from `brain` to `mop`, which changes the absolute
path and must be done from OUTSIDE the session (or as its final action), after which a new chat opens at
`/Users/scammermike/Downloads/mop`. If this handoff still lives under `.../brain/`, do the mv as the first
housekeeping step, then continue with A6.
