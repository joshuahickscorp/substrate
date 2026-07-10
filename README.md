# mop

> **Status:** experimental framework and measurement instrument. Active paradigm: the Form
> Substrate Program (FORM_SUBSTRATE_PROGRAM.md). The local M3 Pro is the active execution host.
> Natural-video work is serialized locally after input, rights, identity, and control gates pass.
> No receipt proves that a Mac Studio was delivered or that one is currently required.

The Form Substrate Program: a referent-bound, form-receiving, memory-bearing,
performance-measured substrate for intelligence. Any observation family (video, audio, text,
symbols, telemetry, action traces) enters as a form over shared referents; capabilities are earned
through preregistered experiments with nulls, controls, and density accounting. Frozen inherited
encoders are measurement instruments and controls while the owned trainable substrate develops
behind evidence gates. Root docs:
FORM_SUBSTRATE_PROGRAM.md (worldview), FORM_SUBSTRATE_DOCTRINE.md (methods),
FORM_SUBSTRATE_EXPERIMENTS.md (F-series bank), FORM_SUBSTRATE_CODEMAP.md (code map),
MOP_MAXIMUM_POTENTIAL_GOAL.md (standing execution loop), MOP_POTENTIAL_ATLAS_2026_07.md (37-facet
audit), PERFORMANCE_DENSITY_DOCTRINE.md, PARADIGM_MIGRATION.md, and LEGACY_INDEX.md.

## What this is (and is not)

This is not an inherited JEPA training project. Inherited encoders remain frozen, are called under
`no_grad`, and serve as teachers, controls, and measurement instruments. A unit test asserts that no
inherited encoder parameter receives a gradient.

The owned custom substrate is a separate trainable lane with no inherited code or weights in its
artifact. It includes dense event state, prediction, memory, plasticity, action, performance, and
evidence interfaces. The five-seed CM7 objective regime has already run locally and closed with a
null. Its checkpoint and verifier contracts survive; its architecture is not treated as canonical.

Any claim about a frozen instrument first proves that the required variable is decodable from its
output. Any owned-substrate claim additionally beats frozen, random, restart, stronger-shell, and
matched-resource controls. Neither lane may claim information that its input or event identity did
not preserve.

## Cached-latent-first design

Frozen instruments encode once when the scientific view contract permits caching. Video tensors go
through the selected instrument, and outputs land in a manifest-bound memmap cache. Caches are
invalidated by checkpoint, source-byte, decode, preprocessing, view, resolution, precision, layer,
geometry, RNG, referent, event, or split drift even when weights are frozen.

Exact-architecture random controls and deterministic programmatic inputs remain available for
mechanics and learned-code controls. Their manifests record backend and evidence scope so generated
or random-control latents are never mistaken for natural learned evidence. The owned substrate may
also train directly on citable inputs without an inherited teacher when its experiment contract
licenses that path.

## Device flag (Apple-Silicon-native)

The current M3 Pro supports both CPU and Metal (MPS). Everything device-touching goes through
`devices.resolve(cfg.device.kind)`, but scientific commands pin the measured path. CPU is the active
P4 path and the verified official dense ViT-B path. MPS is used only after workload-specific
stability and numerical-parity checks. A larger host is considered only through the measured gate in
SCALING.md.

```
.venv/bin/python scripts/run_experiment.py experiment=e1_baseline device=mps
```

See APPLE_SILICON.md (the MPS-first story, the 64-frame Metal limit, MLX option) and SCALING.md.

Note on data: V-JEPA ships pretrained WEIGHTS (which load fine, see STATUS.md and the Studio
maximization plan), not a
training dataset. Its benchmarks (Something-Something-v2, Ego4D, EPIC-KITCHENS) are separate
datasets to obtain when expanding to natural-video latents. The system runs today on cached
real-encoder latents (structured-synthetic video) and the synthetic latent generator.

## Quickstart

Python 3.12 via `uv`. From the repo root:

```
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"      # add ann for hnswlib; encoder for real V-JEPA weights
make verify-install              # isolated import from /tmp, with no PYTHONPATH
make test                       # full suite (mps/cpu, seconds); count grows, see STATUS.md
make e1                         # E1 the gate: naive forgets, protected retains, both learn
make i4                         # I4 backprop-alternatives comparison (FA/DFA/FF/...)
make queue-dry                  # dry-run the campaign queue (no compute spent)
make accept                     # end-to-end acceptance check
```

`make install` does the venv + install in one step (it pulls `[dev,ann]`). Use
`make install-studio` for the full encoder, video, and Apple Silicon extras. Both installation
targets prove that `mop` imports from outside the checkout without `PYTHONPATH`; commands that still
set `PYTHONPATH` do so only for historical script compatibility.
Use `make lint` / `make types` / `make fmt` for ruff + mypy. `make diag` runs the diagnostics.
E1 must pass (the gate) before any downstream result is trusted: see EXPERIMENTS.md for the
build-order DAG (E1 gates everything; E2,E3 feed E4; E2+E3+E4 = the Level-5 headline).

## Rehearse the Mac Studio workflow (one command)

```
make rehearse        # python scripts/studio_rehearsal.py
```

This is the Mac-Studio REHEARSAL CAPSULE: it walks the entire future Studio workflow end to end
on tiny generated fixtures, with NO downloads and NO long runs, and writes
`runs/studio_rehearsal/{report.md,summary.json}`. The path it proves:

  tiny video corpus (generated .npy clips) -> source validation -> decode + preprocess ->
  cache creation -> cache integrity -> full-grid dry-run + cost agreement -> one tiny Tier C
  run -> provenance manifests -> microbenchmarks -> Markdown + JSON report.

It is a rehearsal, not a science result, and the report tags every stage real / mocked /
provisional. On THIS machine the video DECODE is mocked (no codec): the corpus is `.npy` clips
that flow through the exact same validate/decode/preprocess/cache contract; on the Studio the
same path runs over real `.mp4` with a video backend (`uv pip install -e ".[video]"`). Other
operator tools: `make doctor` (readiness), `make cache-list`, `make storage`, `make bench`,
`make report`, `make docs` (drift gate).

## The one Studio pipeline (plan -> acquire -> validate -> cache -> run -> optimize -> report)

`scripts/studio_pipeline.py` is the single acquisition surface for the future 1 TB Mac Studio,
plus a `local-max` lane that does the most real work that is SAFE on this M3 Pro today. Heavy
downloads are DRY-RUN by default and become real only under `--execute` + `--budget-gb` + (for
any source with terms) `--accept-license`. Every stage honors a device PROFILE whose kill
switches (disk, download, clip, run-count, wall-time, tier caps) are enforced, not advisory.

```
make local-max                                              # current-device maximal rehearsal (m3pro)
make studio-plan                                            # DRY-RUN plan under the 900 GB studio budget
python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900
python scripts/studio_pipeline.py acquire --plan runs/studio_pipeline/latest/plan.json   # DRY RUN
python scripts/studio_pipeline.py run --gated --tiers C --full --profile studio-1tb   # gates are kill switches
python scripts/studio_pipeline.py profiles                  # list the kill-switch envelopes
```

The `run`/`cache`/`optimize` stages default to the SAFE `m3pro-local-max` profile, so an
unqualified `run --full` fails the run-count kill switch on this laptop instead of launching a
sweep; the Studio passes `--profile studio-1tb` explicitly (as above). The 900 GB is the budget
CAP for the 1 TB Studio disk, not the planned volume: the seeded registry plans roughly 150 GB
of breadth by default (about 425 GB with `--accept-license`), and the rest is headroom for
larger subsets or future sources.

The planner is a breadth-first knapsack over `registry/datasets.yaml` (action/egocentric/
instructional/audio/synthetic/local sources, each with license, size, risk, status) and
`registry/models.yaml` (canonical V-JEPA plus clearly-tagged auxiliary/distilled/quantized
extras that NEVER replace canonical). Full Ego4D is never planned by default. `local-max` runs
real on this device: generate control corpora, validate, build a tiny real latent cache, audit
the queue/cost agreement, microbench, run one gated leg, and write a report under
`runs/studio_pipeline/`. See SCALING.md for the exact Mac Studio day-one sequence.

## Developmental capacities (sentience-ADJACENT, never sentience)

`scripts/devel.py` is a measurable developmental layer on top of the frozen substrate. It does
NOT claim sentience, consciousness, feelings, or agency: every capacity is a measurement with a
null hypothesis, and a code-level safety rail (`mop.devel.north_star`) scans every rendered
report and refuses to ship an affirmative sentience claim. The north star is the loop
`perceive -> remember -> predict -> notice surprise -> choose what to study -> adapt ->
consolidate -> abstract -> transfer -> explain what changed -> choose the next lesson`.

```
python scripts/devel.py capacities      # the 14-rung capacity ladder (registry/capacities.yaml)
python scripts/devel.py paradigms       # the paradigm frontier registry (registry/paradigms.yaml)
python scripts/devel.py ablation --scope local   # next-best experiment by info-gain per compute hour
python scripts/devel.py curriculum      # next-lesson manifest: REAL probes over controls, rejects noisy-TV
python scripts/devel.py metacognition   # self-monitoring report (gated by the safety rails)
python scripts/devel.py paperwatch      # offline literature watch
python scripts/devel.py experiments     # the preregistered experiment bank (registry/experiments.yaml)
make devel ladder curriculum
```

## The experiment bank (preregistered, machine-readable)

`registry/experiments.yaml` is the single source of truth for the whole bank: the conducted
E1-E10 + I4, the bleeding-edge EX-series (EX1-EX18), and the reusable diagnostics (D) and
ablations (A). It is a PREREGISTRATION: each row commits a null, a headline metric, a falsifier,
the controls and gates, the resource tier, the capacity-ladder/paradigm map, the proof linkage
(atlas factor, null card, R0-R5 evidence level), and the proof/FAILURE_TAXONOMY.md slot a null
maps to, BEFORE it runs. `EXPERIMENTS.md` is generated from it (`scripts/devel.py experiments
--render`), so the doc cannot drift; the validator refuses an implemented row that does not map to
real code, and moonshots stay catalogue-only until a cpu-now MVP exists. Runnable today (cpu-now,
gated on the E1 gate): EX12 atlas + geometry battery, EX17 latent iterative reasoning (a weight-tied
refiner vs a COMPUTE-MATCHED untied-depth control), EX8 intrinsic-motivation bake-off, EX16
codebook/VQ abstraction, EX3 test-time adaptation. New supporting diagnostics: `diagnostics/geometry`
(CKA/RSA/effective-rank/anisotropy), `diagnostics/compute` (FLOP accounting so matched-compute is
enforced), `diagnostics/substrate_ablation` (real vs frozen-random vs shuffled vs compressed).

The curriculum engine is real on this device: it generates control corpora, extracts frozen
latents, and ranks candidates by LEARNING PROGRESS (probe-accuracy gain) gated by a permutation
test, so it picks the learnable-but-not-mastered family and REJECTS the aleatoric noisy-TV (the
trap an error-seeking learner would chase forever). "curiosity"/"drive" are engineered objective
terms (novelty, uncertainty, learning progress), not feelings. The capacity ladder and paradigm
registry mirror the experiment contract (baseline, ablation, metric, null) so a speculative
mechanism can never be promoted to canonical science without an explicit tag.

## Repo map

```
substrate/      the frozen module + its access path (NOT trained)
  encoder.py        V-JEPA wrapper: requires_grad=False, no_grad only; lazy real weights,
                    frozen-random fallback
  cache.py          run the encoder once, write latents to a memmap
  latent_store.py   memmap-backed latent store (read path for all learning)
  datasets.py       task/class/domain-incremental streams + synthetic latent generator
shell/          the trainable shell (everything that learns)
  predictor.py      latent->latent predictor (+ action-conditioned variant)
  heads.py          task heads + probabilistic gaussian head (calibration, epistemic split)
  ensemble.py       ensemble disagreement uncertainty
  buffer.py         latent hippocampus: prioritized replay, KV faiss/brute index, eviction
  plasticity.py     staged plasticity (hard/soft/learned) + PNN rigidity + reopening
  consolidation.py  EWC (Fisher proxy) + SI (path integral), selectable + composable
  neuromod.py       DA=RPE, ACh=expected unc, NE=unexpected unc; scalar gates on lr/replay
  modulation.py     context-gating, working memory, chunking
learning/       the learning rules
  backprop.py       standard backprop trainer (accuracy ceiling); the Learner wiring
  alternatives/     FA, DFA, FF, target-prop, eq-prop, predictive-coding, local rules
metrics/        BWT/FWT/adaptation speed/avg acc (continual.py); adaptation-retention
                frontier + AUC (frontier.py, the program's central metric)
diagnostics/    linear_probe (the gate), noisy_tv, calibration, fisher_trace, determinism
experiments/    base.py (the doctrine contract), e1 harness, i4 harness, E2..E10 scaffolds
harness/        runner.py + cli.py (run, compose, campaign queue)
campaign/       synthesized training campaign (legs/tracks/tiers, run queue); see DECISIONS.md
configs/        OmegaConf group composition: device/, encoder/, shell/, experiment/
registry/       machine-readable registries: datasets.yaml (sources), models.yaml (aux encoders),
                paradigms.yaml (mechanism candidates), capacities.yaml (capacity ladder), paperwatch.yaml
studio/         the Studio acquisition layer (under src/mop/): profiles+kill-switches, dataset/
                model registry loader, 1 TB knapsack planner, dry-run downloader, data cards +
                license ledger, synthetic control expansion, the plan/acquire/validate/cache/run/
                optimize/report pipeline + local-max
devel/          the developmental capacities layer (under src/mop/): north_star + safety rails,
                paradigm/capacity/paperwatch registries, curriculum engine (learning-progress data
                selection), automated ablation/hypothesis engine, metacognition reports
scripts/        run_experiment.py, run_queue.py, acceptance.py, studio_pipeline.py, devel.py
```

## The doctrine contract

`experiments/base.Experiment` is abstract and refuses to instantiate unless the subclass
declares `metric`, `baseline`, `ablation`, and `null_hypothesis`. An experiment that does
not state its null cannot run. This is enforced in code, not convention. Every null in the
bank maps to one entry in the negative-result taxonomy (EXPERIMENTS.md), so a failed
experiment is a result, not a dead end.

## Where to read next

- ARCHITECTURE.md: the layering, the module -> corpus-lever map, the frozen-substrate
  invariant, the device boundary.
- EXPERIMENTS.md: the E1..E10 + I4 registry (metric, null, baseline, ablation, tier), the
  build-order DAG, the negative-result taxonomy, the diagnostic gates.
- SCALING.md: exactly what to flip when the Mac Studio or rented CUDA arrives, per tier,
  with first-commands-on-the-new-machine.
- runs/pre_studio/RESULTS_PRE_STUDIO.md: per-experiment interpretation of the full pre-Studio
  run, with the adversarial-verification verdict for every candidate positive.
- DOCTRINE_SYNTHESIS.md: the whole corpus read through the two central doctrinal questions
  (developmental moldability, language-independent abstraction), with the honest state of the
  evidence and the proposed next experimentation lanes.
- STUDIO_HANDOFF.md: historical transfer and procurement scenarios. Current hardware decisions are
  governed by MOP_MAXIMUM_POTENTIAL_GOAL.md and proof/EXTENDED_COMPUTE_REQUIREMENTS.json.
- STATUS.md: live build log (done / scaffolded / not started).
- DECISIONS.md, ISSUES.md: rationale log and deferred-item ledger.

## Form

Code FORM follows BLACKHOLE.md (density, flat structure, few load-bearing files, surface
every failure). No em dashes or en dashes anywhere (commas, colons, parentheses only).

## Roadmap

Brain is an experimental framework for continual and developmental learning on
top of frozen perception encoders and a trainable custom-substrate workbench. It
is a measurement instrument, not a finished result. The current M3 Pro is an
active 300-minute adaptive local execution target; frozen scientific shards keep
their registered identity. No Mac Studio purchase or hardware boundary is
assumed. Real natural-video scientific coverage is limited by rights-clean task
  data, independent units, cache materialization, and verification, not inherited-model availability.

### Now (works today)
- The full shell of mechanisms: EWC and Synaptic Intelligence consolidation,
  prioritized-experience replay, and seven alternative learning rules, each with
  real numerical tests.
- A reproducibility spine: null-hypothesis contracts, provenance manifests,
  bit-identical runs, and a proof grammar (atlas rows and null cards).
- A host-aware pipeline for plan, acquire, validate, cache, run, and report that
  is dry-run by default and gated by resource and evidence kill switches.
- A pre-Studio run across the full registered experiment bank in 9 disciplines,
  each with a pre-registered null hypothesis and an adversarial verification
  pass. Result: an honest null corpus with bounded survivors and refutations.
  The claim-level audit now separates scientific results from mechanics and
  shows zero measured hardware-blocked rows.

### Next (local first, move only a measured remainder)
- Finish P4 and P5 through the adaptive governor, run P6's progressive ladder, and admit P7/P9's
  next independently sourced trajectory and workload cohorts through their existing local harnesses.
- Extend Wave E0's verified shared event, intervention, and memory-lifecycle substrate, then run the
  P6 disk stream through its 10k, 100k, and conditional one-million-event ladder.
- Reuse P7's verified rendered action, same-parent intervention, eight-arm world-model, and exact
  compute ledgers for independently sourced action trajectories. Its tiny fixture supported the
  null and earned no capability claim.
- Reuse P9's verified causal telemetry, shifted-confounder, calibration, relief-controller, resume,
  and accounting mechanics on independent natural workloads. Keep energy unmeasured until an
  explicit wall-power boundary exists.
- Integrate the official dense ViT-B instrument into a small same-input natural
  cache and matched learned, random, handcrafted, pooled, and owned controls.
- Build a rights-aware native audiovisual cohort with original clocks, session
  units, frozen splits, and wrong-time/wrong-event controls.
- Grow the representational atlas factor by factor, including the rows that are
  not linearly decodable (the substrate's blind spots). A first real-weight
  atlas row exists; growing it is ongoing.

### Later
- Run the full E1 to E10 plus I4 campaign at real scale with seeds and error
  bars, plus an encoder-scale ablation (is the developmental story a property of
  the shell mechanisms or of substrate scale).
- Gate every claim through the proof system before it is allowed to be a claim.
