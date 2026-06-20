# EXPERIMENTS

Registry of E1-E10 + I4. Every row carries a metric, an explicit null, a baseline, and an
ablation (the doctrine contract). Tier tags: `cpu-now` (laptop, cached latents), `gpu-later`
(Studio), `env-later` (needs an environment + rollouts), `2.1-only` (needs V-JEPA 2.1 dense).

Build order (Vol I Section 8 DAG): E1 gates everything. E1 -> {E2, E3, linear-probe}.
E2, E3 feed E4. E4 must pass its noisy-TV test before its signal is trusted as a trigger
in E3 or a prioritizer in E2. E2+E3+E4 combined = the Level-5 headline. Branches: E5
(curiosity), E6 (relational, 2.1), E7 (sparse/modular), E8/E9 (dendritic/local). E10 last.

| Exp | Name | Metric | Null hypothesis | Tier |
|---|---|---|---|---|
| E1 | Baseline continual harness (GATE) | per-domain pred loss, avg acc, BWT, FWT, adaptation speed, frontier AUC | with shuffled task labels (or capacity/difficulty controls) naive and protected show no retention gap; if no forgetting, tasks too easy; if no learning, target not decodable from latent | cpu-now |
| E2 | Latent hippocampus (replay) | frontier (retention vs adaptation) at matched buffer size, BWT | prioritized replay ties random replay; replay ties no-replay (stream too short / latents not distinct) | cpu-now |
| E3 | Critical-period schedule (staged plasticity) | frontier, parameter drift, Fisher trace | staged plasticity ties constant LR AND tuned cosine decay (it was just an LR trick / nothing to shape on a frozen substrate) | cpu-now |
| E4 | Neuromodulation (uncertainty gating) | adaptation on learnable region, near-zero resource on noisy-TV, calibration | point prediction-error gating chases noisy-TV as much as ungated (conflates epistemic with aleatoric); needs ensemble/distributional | cpu-now |
| E5 | Curiosity as self-curriculum | steps-to-competence, fraction time on learnable, noise attraction | prediction-error/RND curiosity equals random on a learnable-vs-noisy env (only learning-progress should survive) | env-later (data-selection variant cpu-now) |
| E6 | Relational map over latents | recombination generalization, planning success, 2-vs-2.1 delta | structured head ties parameter-matched flat baseline; gain on 2.1 not larger than on 2 (latent lacks object factorization) | 2.1-only / gpu-later |
| E7 | Sparse / modular predictor | interference / forgetting at matched params, routing entropy | sparse/modular ties parameter-matched dense (reduction was just capacity); speedup reported separately, not required | cpu-now (speedup gpu-later) |
| E8 | Dendritic predictor | capacity-per-param, online adaptation at matched params | dendritic block ties matched MLP (analogy adds complexity, no benefit) | cpu-now |
| E9 | Local learning head | acc vs backprop, activation memory, online stability | local rules fail to reach within margin of backprop and offer no memory/stability win | cpu-now |
| E10 | Minimal open-ended JEPA (CAPSTONE) | distinct skills, reuse, archive diversity, transfer, non-collapse | single agent plateaus; open-endedness is a population/env-generation property a solo agent cannot exhibit | env-later / gpu-later |
| I4 | Backprop-alternatives comparison | accuracy gap vs backprop, locality, compute, stability | no alternative (FA, DFA, FF, target-prop, eq-prop, predictive-coding) comes within a stated accuracy margin of backprop at matched head/data/seed/budget | cpu-now |

## Negative-result taxonomy (every null attributed to one)
1 biology mapped badly; 2 needs unfrozen encoder; 3 frozen latent lacks the info (linear
probe); 4 predictor/head too weak (capacity ablation); 5 task too easy; 6 task too hard;
7 needs embodiment/action; 8 only works combined with another lever; 9 hardware-incompatible
(separate representational from compute claim); 10 conceptually irrelevant to latent
prediction (the strongest negative, bounds the substrate).

## Diagnostics gate the experiments
- linear-probe distinctiveness: before any X-dependent mechanism (E2 buffer distinctiveness,
  E4 controllability, E6 object identity).
- noisy-TV: E4 signal must pass before it is trusted (E2/E3 triggers).
- calibration: E4 probabilistic head.
- Fisher trace: E3 critical-period signature.
- determinism: run before trusting any cross-condition delta (Metal ~50% byte-identical).
