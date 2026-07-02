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
  matched 256px, p=0.0285 (15/29 vs 7/29, ONE clip from ambiguity, not yet multi-seeded).
- Compositional factoring off ceiling: held-out (shape,color) 0.725 = seen 0.708 (within-arm property).
- PR1 router-licensing: heterogeneous oracle gain 0.155 vs homogeneous mean-copy 0.118 + spread (an
  EXISTENCE upper bound, not a trained-router win).
- at3 temporal currency: full-clip decodes motion/speed a static frame cannot (least-surprising possible).

NULL / DEMOTED (do not carry these forward as positives):
- al2 shared-code: cross-modal alignment FAILS a kNN-topology permutation null; only same-modality vision
  "aligns" (and that was propped by two V-JEPA columns, same encoder same clips). MoP's shared-code
  precondition is NOT met on current substrates.
- ws2 structure-beats-capacity: fails its own dual acc-AND-NLL contract (was an OR-rule over-claim).
- pr7 delta-rule: NULL, trails the Hebbian floor 10/10 seeds (Hebbian fast-store itself beats slow-only
  +0.029, a modest real signal, the only plasticity flicker).
- The 24-null reasoning lane, and the 4 recalibrated degenerates (mt5/al1/dr12/ws3), all NULL.

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
  Qwen cache is text-OF-LABELS, not parallel LLM states on the same clips, so it cannot test the
  language-independent-abstraction north star. Generate a deterministic pixel-derived caption per clip and
  cache the small-LLM hidden states on THOSE, paired to `data/cache/vjepa2_vitl_nuisance` (the real vision
  cache), then re-run al2's corrected (topology-permutation) metric on the vision<->text pair. This is the
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

- Python: `.venv/bin/python`, with `PYTHONPATH=/…/mop` and `OMP_NUM_THREADS=4` for scripts that import
  `scripts.*`. The venv has no `pip`; the editable install resolves `src/mop` as package `mop`.
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
