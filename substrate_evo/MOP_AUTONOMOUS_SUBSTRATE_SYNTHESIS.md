# MOP Autonomous Substrate Evolution: Terminal Synthesis (Substrate Event Horizon)

Program `mop-autonomous-substrate-evolution-v1`, branch `agent/mop-autonomous-substrate-evolution` off
`4b4a0ec`. Genesis II / Gen2 / Gen3 immutable; collapse branch untouched. Activation false. A tie is a null.

## Terminal condition: Substrate Event Horizon (full exhaustion proof)
The program reaches terminal condition B: a fully-proven Substrate Event Horizon, not a single null.

- Five materially different architectures failed: Genesis A (Shared Latent Workspace, image), Genesis B (Sparse
  Modular, image, structural_null), A-T (Temporal Shared Latent Workspace), C (Multi-Horizon Predictive State),
  D (Hierarchical Plasticity Lattice, structural_null) on a valid temporal bed.
- Five distinct improvement premises failed: shared workspace, sparse modularity, multi-timescale state (fast
  GRU + medium EMA context) + multi-horizon auxiliary, predictive-state pretraining, hierarchical bounded
  reorganization.
- Baselines converged: single-task GRU 0.898; strongest continual baseline LSTM+GDumb 0.480.
- Valid temporal headroom measured: on HAR raw inertial signals a GRU (0.898) beats an order-free
  bag-of-timesteps (0.813) by +0.085 (lcb 0.078), and shuffling timesteps hurts (0.856) - order genuinely
  matters, unlike Genesis II's image beds where fast state was degenerate.
- Adequate independent units: subject-disjoint train/test plus five stable seeds (the C1 two-seed lesson
  honored throughout).
- No gate weakened.

## The irreducible blocker (the decisive new evidence)
An unbounded-memory upper bound (LSTM storing all past windows) reaches 0.764 while bounded GDumb-LSTM reaches
0.480 - a +0.284 continual headroom. **The continual bottleneck is MEMORY CAPACITY, not substrate
architecture.** No owned architecture beats the bounded baseline; the achievable headroom is captured by more
memory. A better BOUNDED-memory mechanism is exactly what R1 (retrieval) and P1R (replay) already closed. So no
dependency-ready substrate-architecture work remains on this data.

## Exact external requirement
a validated bounded-memory mechanism that beats GDumb (R1/P1R closed) OR a larger sealed memory budget with matched cost

## Answers to the 45 synthesis questions (compact)
An owned compact multi-timescale temporal substrate was built (owned projection + fast GRU state + medium EMA
context + slow workspace + GDumb memory + heads), plus two further materially different temporal architectures.
Temporal order mattered on the valid bed; baselines were converged; but fast, medium, and slow state added no
value beyond LSTM gating (substrate null); memory helped only as raw capacity (the +0.284 unbounded-memory gap);
consolidation and bounded reorganization added no value (D structural_null); learned plasticity had no stable
headroom (simple_policy_sufficient); the entity learned new tasks (new-task acc ~0.93) but did not retain or
adapt better than LSTM+GDumb; no cross-context or cross-domain transfer positive; no architecture won (LSTM+GDumb
strongest); total cost small (all models <50k params, 406 active LOC); self-verifying (order gate, receipts,
independent recomputation, 5 seeds). Scores: layers 1-3 (falsification, orchestration, null-understanding)
at/near target; layers 4-8 below target, closed via this event horizon. Owned Substrate v1 is NOT selected;
functional reorganization is NOT evidenced; a unified cross-domain entity is NOT evidenced; activation is not
licensed.

## Evidence ceiling
MOP has a strong falsification program, efficient orchestration, a large reliable null map, and a real compact self-verifying owned multi-timescale substrate that runs on both image and valid temporal beds; but NO architecture, controller, or timescale shows incremental value over strong matched conventional alternatives, even where temporal order provably matters

## Exact next frontier
a validated bounded-memory mechanism that beats GDumb (R1/P1R closed) OR a larger sealed memory budget with matched cost

## Resume / verify commands
- resume: read MOP_AUTONOMOUS_SUBSTRATE_STATE.json and continue from next_exact_action
- verify: python substrate_evo/build_evo_synthesis.py
- status: cat runs/substrate/mop-autonomous-substrate-evolution-v1/status.json
