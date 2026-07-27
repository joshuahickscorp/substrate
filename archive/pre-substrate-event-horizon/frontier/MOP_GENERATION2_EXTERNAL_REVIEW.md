# MOP Generation 2: External Review Package

A compact brief for an outside researcher. It states what MOP claims, what it explicitly does not, what was
independently verified, what stayed same-team, what failed, and where the uncertainty is. No dashes.

## What MOP claims (Generation 2)
1. Robust seeded experimental mechanics across many mechanism families.
2. A calibrated admission battery with construct-validity controls (noisy-TV, shuffled-target, wrong-time,
   rate-matched-random, oracle headroom, capable-architecture agreement) that provably prunes null, oracle-free,
   and misleading-signal beds before principal compute.
3. Real-data canary and same-team cross-architecture evidence for one mechanism (P1R).
4. A work-conserving resource-token DAG orchestration layer that measurably raises machine utilization.

## What MOP explicitly does NOT claim
- No externally independently replicated mechanism.
- No mechanism with incremental value beyond strong established controls that survives an external test.
- No licensed cluster, construction, integration, or activation.
- P1R is not a competitive replay method: on EMNIST it reaches approx 0.10 final accuracy vs GDumb approx 0.59.

## Independently verified vs same-team
- Independently verified (separate recomputation, receipt-hash intact): all seven Generation 2 frontier
  admissions; the Lane P bed validity (four-auditor adversarial workflow).
- Same-team only: every P1R positive (split-MNIST, CIFAR-100, KMNIST) uses in-house implementations across
  architectures; none is an external independent replication.

## What passed and what failed
| Mechanism | Best evidence | Terminal class |
|---|---|---|
| P1R | 3 same-team benchmark positives | external replication null (EMNIST) |
| N1 | MNIST canary positive | CIFAR-10 confirmation null |
| V1 | real per-item verification value | architecture_dependent (1 of 3 estimators) |
| K1, M1 | architecture-robust value | pruned (fire where the intervention does not causally help) |
| E1 | boundaries help | pruned (not beyond simple change detectors) |
| C0 | none | pruned (worse than EMA smoothing) |
| A1 | competent policy | pruned (ties a fitted value estimator) |
| S1 | none | pruned (worse than random; compounding model error) |
| U1, R1 | none | real-data canary null |
| D1, I1, G1 | none | retired or never licensed |

## The single most important result
Faithful P1R, tested against established replay methods (GDumb, reservoir) on a stronger external source under
matched memory and compute, is a replication null. Its validated per-item replay-value prediction does not yield
a replay policy that beats GDumb. This is the strongest surviving hypothesis reaching its external boundary.

Reproduction: see MOP_GENERATION2_REPRODUCTION_GUIDE.md. Evidence table: MOP_GENERATION2_EVIDENCE_TABLE.csv.
Full artifact hashes: MOP_GENERATION2_ARTIFACT_INDEX.json. Claim boundary: MOP_GENERATION2_CLAIM_BOUNDARY.json.

## Where the strongest scientific uncertainty remains
1. P1R value as a soft sampling priority over a representative buffer (does it beat GDumb, not only no-replay).
2. Whether any Generation 2 mechanism failure was a bed or estimator-capacity artifact rather than a true null.
3. External independent replication of P1R, never yet attempted.
