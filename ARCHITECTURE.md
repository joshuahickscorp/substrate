# ARCHITECTURE

The system is NOT a JEPA. V-JEPA is one frozen perceptual module inside it: inherited
perception, loaded once, never trained, inference-only, living under `substrate/`.
Everything else is the trainable shell and the research surface. Learning never touches
the encoder; it runs on cached latents.

Layering:

```
video/clips ──(once, frozen)──> substrate.encoder ──> latents ──> substrate.cache (memmap)
                                                                        │
   synthetic latents ───────────────────────────────────────────────────┤  (no-weights path)
                                                                        ▼
                       shell {predictor, heads, ensemble, buffer, plasticity,
                              consolidation, neuromod, modulation}
                                                                        │
                          learning {backprop | alternatives}            │
                                                                        ▼
              experiments (E1..E10, I4) ── metrics ── diagnostics ── harness/queue
```

## Module -> corpus lever map

Levers are cited by id from the Vol I 33-row ranking table and the Vol II A/B/C/D/E/F/G
dossiers. "Substrate attachment" = how a lever bolts onto the frozen module without
becoming part of it.

| Code module | Implements (lever ids) | Corpus role |
|---|---|---|
| `substrate/encoder.py` | (the frozen substrate itself) | V-JEPA 2 ViT-L default, 2.1 dense options; inference-only |
| `substrate/cache.py`, `latent_store.py` | doctrine: cache latents once, iterate on cache | makes the laptop feasible; frozen => latents never stale |
| `substrate/datasets.py` | Stream Library raw material (SSv2/Ego4D/synthetic) + synthetic generator | task/class/domain-incremental streams over latents |
| `shell/predictor.py` | latent predictor; action-conditioned AC variant | forward latent dynamics; surprise source |
| `shell/heads.py` | task heads + probabilistic (gaussian) head | calibration + epistemic/aleatoric split (C-cluster) |
| `shell/ensemble.py` | ensemble disagreement uncertainty | epistemic signal for E4 / noisy-TV |
| `shell/buffer.py` | L-LatentReplay, B1-B7 (episodic memory, prioritization, KV index, eviction) | latent hippocampus (Axis B) |
| `shell/plasticity.py` | L-StagedPlasticity, L-Metaplasticity, A1 (soft) / 6.1 (hard) / learned; PNN rigidity | critical-period schedule (Axis A) |
| `shell/consolidation.py` | L-SynapticConsolidation (EWC Fisher proxy, SI path integral) | weight-space dual of replay |
| `shell/neuromod.py` | L-Neuromodulation (DA=RPE, ACh=expected unc, NE=unexpected unc) | scalar gates on LR/replay/explore (D-cluster) |
| `shell/modulation.py` | context-gating (E8 active-dendrites), working memory, chunking (G-cluster) | optional modular control |
| `learning/backprop.py` | standard backprop trainer | accuracy ceiling for I4/E9 |
| `learning/alternatives/` | L-FeedbackAlignment, L-ForwardForward, L-EquilibriumPropagation, L-LocalLearning, predictive-coding, target-prop, DFA | I4/E9 backprop-alternatives |
| `metrics/continual.py` | BWT, FWT, adaptation speed, avg accuracy | continual-learning metrics |
| `metrics/frontier.py` | adaptation-retention frontier (Pareto), frontier AUC | the program's central metric |
| `diagnostics/linear_probe.py` | the single most important diagnostic | is X linearly decodable from frozen latent? |
| `diagnostics/noisy_tv.py` | epistemic-vs-aleatoric guard | curiosity/uncertainty must ignore irreducible noise |
| `diagnostics/calibration.py` | reliability diagram / ECE | calibration of the probabilistic head |
| `diagnostics/fisher_trace.py` | Achille/Rovere/Soatto Fisher-trace | critical-period signature for E3 |
| `diagnostics/determinism.py` | Metal nondeterminism sanity loop | size tolerances from real spread |
| `experiments/base.py` | doctrine contract enforcement | baseline+ablation+metric+null or it cannot run |
| `harness/{runner,cli,sweep,queue}.py` | run, compose sweeps, campaign queue | execution surface |

## The doctrine contract
`experiments/base.Experiment` is abstract and refuses to run unless the subclass declares
`metric`, `baseline`, `ablation`, and `null_hypothesis`. An experiment that does not
state its null cannot be instantiated. This is enforced in code, not convention.

## Device boundary (the Studio flip)
Everything that touches a device goes through `devices.resolve(cfg.device.kind)`. The same
code runs `device=mps` (laptop, toy) and `device=cuda` (Studio/rented, full) by config
alone. MPS op gaps fall back to CPU through `devices.safe_to`. See SCALING.md.

## Frozen-substrate invariant
The encoder is wrapped so its parameters are `requires_grad=False` and it is only ever
called under `torch.no_grad()`. A unit test asserts no encoder parameter ever receives a
gradient. You cannot recover information the frozen encoder discarded: any mechanism that
needs variable X must pass `diagnostics/linear_probe` first.
