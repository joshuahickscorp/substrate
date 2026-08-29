# MOP Generation 2 Scientific Frontier: Terminal Synthesis

Program `mop-generation2-scientific-frontier-v1` on branch `agent/mop-scientific-frontier` off evidence HEAD `094bdd9`.
No model or assistant attribution. A tie is a null. No positive sealed without independent adversarial verification.

## Headline
Of seven unopened mechanisms, none passed real-data admission; the one surviving mechanism (P1R) was tested against established replay methods on a stronger external source (EMNIST) with result: replication_null. No cluster, construction, or integration is licensed. Activation stays false.

## Evidence-class ladder (what each result is worth)
mechanics_robustness -> controlled_bed_plausibility -> real_benchmark_canary -> same_team_cross_architecture_confirmation -> external_method_replication -> independent_scientific_confirmation -> cluster_interaction -> construction -> integration -> activation

## The seven unopened mechanisms: all terminal, none admitted
| Lane | Mechanism | Dataset | Result | Why |
|---|---|---|---|---|
| V1 | selective verification | CIFAR-10 corrupted | architecture_dependent | Real value, avoids U1 noisy-TV trap, but only 1/3 capable estimators decodes it |
| K1 | contradiction repair | CIFAR-10 disagreement | pruned_mechanism | Architecture-robust value but fires on contradictions not warranting repair (noisy-TV) |
| M1 | messaging | split-view MNIST | pruned_mechanism | Message value predictable but fires where the message does not causally help |
| E1 | event formation | ordered KMNIST stream | pruned_mechanism | Relational boundaries help but not beyond simple change detectors by SESOI |
| C0 | trace stability | noisy KMNIST stream | pruned_mechanism | Confidence-weighted trace worse than EMA smoothing (wrong direction) |
| A1 | read affordance | gymnasium classic-control | pruned_mechanism | Ties a fitted value estimator; no incremental value |
| S1 | simulate consequence | gymnasium classic-control | pruned_mechanism | Learned-model planning worse than random |

Independent verification: all seven re-derived from sealed clause data and receipt-hash intact (MOP_FRONTIER_ADMISSION_VERIFICATION.json). None licenses a canary.

## Lane P: P1R external-method replication (the one surviving mechanism)
Terminal class: **replication_null**. On EMNIST-balanced class-incremental (a genuinely different source: NIST SD19), under matched memory (600) and matched compute (130 steps/task), over 5 torch-seeded streams:

| method | final avg accuracy |
|---|---|
| none | 0.068 |
| reservoir | 0.582 |
| gdumb | 0.594 |
| loss_based | 0.158 |
| recency | 0.136 |
| p1r | 0.104 |

Faithful P1R (learned replay-value predictor + validated toxic-value gate) reaches 0.104, far below the best established method GDumb (0.594) and reservoir, and only marginally above no-replay (0.068). P1R's per-item replay-value ranking does not yield a replay method that beats established replay. This does not contradict the three prior P1R positives, which validated P1R PREDICTING per-item replay value; Lane P shows that predictive signal does not confer a replay-method advantage on a stronger source.

### Adversarial audit (mandate-required, consequential result)
Lane P v1 returned replication_harm. A four-auditor adversarial workflow reached unanimous consensus: unfaithful_operationalization (v1 used the raw-loss control itself as the mechanism, dropped the validated toxic-value gate). One bounded bed-validity repair restored the faithful learned predictor + toxic gate; instrumentation hardened (torch-seeded, 5 seeds, true Vitter reservoir). The harm did not survive: faithful P1R is a null, not harm. Evidence: MOP_FRONTIER_P_ADVERSARIAL_AUDIT.json.

## Downstream licenses (scientific barriers)
- Action-Simulation cluster: unlicensed (A1, S1 not admitted)
- Verification-Repair-Messaging cluster: unlicensed (V1 architecture_dependent, K1/M1 pruned; no preregistered passing subset)
- Event-Trace cluster: unlicensed (E1, C0 not admitted)
- Construction: **unlicensed** (no external P1R replication, no confirmed cluster, fewer than two confirmed components)
- Integration: **unlicensed** (fewer than three confirmation-level functional domains)
- Activation: **false**

## Confirmed architecture boundary
The only mechanism with any real-data positive remains G1-P1R, as same-team cross-architecture evidence (MNIST, CIFAR-100, KMNIST), now bounded by an external-method replication_null on EMNIST. Excluded: D1, U1, N1, R1, historical I1. Not admitted: V1, K1, M1, E1, C0, A1, S1.

## Execution: concurrency correction (the missed opportunity, measured)
Scientific dependency graph: the seven admissions and Lane P are all independent after the frontier authority; the only scientific barriers are the three cluster gates and the construction/integration gates. The observed run executed the admissions largely serially and Lane P as a single monolithic loop.

Measured concurrent-training benchmark on this 28-core host (5 torch threads per capsule):
| concurrency | per-job slowdown | aggregate samples/s |
|---|---|---|
| 1 | 1.0x | 4045 |
| 2 | 1.026x | 8301 |
| 3 | 1.119x | 11669 |
| 4 | 1.223x | 14305 |

At 4 concurrent training capsules the per-job slowdown is only 1.223x for 3.5x aggregate throughput.

Schedule comparison (resource-token DAG scheduler, 28 cpu / 96 GB, measured slowdown):
- Observed wall time: 7639s total (3866s excluding the v1->audit->v2 correctness rework)
- Scientific critical path: 3141s (emnist_dl -> lane_P -> synthesis)
- Corrected parallel wall (Lane P decomposed into 5 seeds x 6 methods, seeds and method branches concurrent): **1226s**, average concurrency 3.81, peak 7
- Speedup vs observed-excluding-rework: **3.15x**; vs observed-including-rework: 6.23x

Scientific comparisons are unchanged by parallel execution: each seed binds the same initial model, data order, task order, update budget, and memory budget across methods; only wall-clock start times differ. The corrected wall time approaches the single longest scientific chain (Lane P), not the sum of independent lanes.

## Forbidden claims
No unopened mechanism is scientifically confirmed. P1R is not externally independently replicated. No cluster, construction, or integration is evidenced. The architecture is not activated.

## Remaining open scientific questions
- External INDEPENDENT replication of P1R (second team or external code) has never been achieved.
- V1 verification value across a capable estimator family (it is architecture_dependent, not fully null).
- Whether P1R replay-VALUE is better used as a sampling priority over a representative buffer than as a keep-filter (auditor hypothesis; would need a fresh preregistration, not a repair).
