# MOP Generation 2 Scientific Frontier Ledger

Program `mop-generation2-scientific-frontier-v1` on `agent/mop-scientific-frontier` off `094bdd9`.

Governing graph: reconcile -> parallel admissions+data -> canaries -> confirmations -> clusters -> replicate -> construction -> integration (only on three-domain closure) -> terminal synthesis.

SESOI 0.05. A tie is a null. Wrong-direction is a failure. No positive sealed without independent adversarial verification. Activation stays false unless explicitly granted.

## Frozen evidence

- Terminal negatives: D1 (retired), U1 (real null), N1 (MNIST positive, CIFAR-10 confirmation null), R1 (KMNIST null), Cluster B (blocked by R1), historical I1 (retired).
- Surviving: P1R positive on split-MNIST / CIFAR-100 / KMNIST (same-team cross-architecture); not externally replicated, not activated.
- Unopened (mechanics-only): A1, S1, V1, K1, M1, E1, C0.

## Lanes (initial)

| Lane | Mechanism | Dataset | Units | Status |
|---|---|---|---|---|
| P | P1R external replication | EMNIST-balanced | tasks | admission_pending |
| A | A1 action | gymnasium classic-control | episodes | admission_pending |
| S | S1 simulation | gymnasium classic-control | states | admission_pending |
| V | V1 verification | CIFAR-10 corrupted | regimes | admission_pending |
| K | K1 contradiction repair | CIFAR-10 disagreement | regimes | admission_pending |
| M | M1 messaging | split-view MNIST | views | admission_pending |
| E | E1 event boundaries | ordered image stream | sessions | admission_pending |
| C | C0 trace stability | noisy image stream | segments | admission_pending |

Resume: read MOP_SCIENTIFIC_FRONTIER_STATE.json and continue each lane from next_exact_action.
