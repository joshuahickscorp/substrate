# MOP Generation 3 Discovery Authority

Generation 3 begins with competing scientific premises derived from the Generation 2 null map, not a large run.
No principal Generation 3 experiment is launched here. Every candidate must also satisfy the null-derived
constraints (MOP_NULL_DERIVED_CONSTRAINTS.json). House rules: no dashes, a tie is a null.

## Five candidate theses (from the null map)
1. C1 P1R-priority: replay value as a soft SAMPLING PRIORITY over a representative buffer, not a keep/eviction
   filter. Must beat GDumb, not only no-replay.
2. C2 V1-capable-family: verification value decodable by a richer capable estimator family (was
   architecture-dependent).
3. C3 S1-model-error-aware: condition simulation depth and trust on estimated rollout error to control
   compounding.
4. C4 M1-causal-message: estimate intervention-level (causal) message value, not predictability.
5. C5 E1-relational-roles: model temporal role changes and relations that simple change detectors cannot capture.

## Ranking (10 criteria) and selection
Ranked totals: C1 37, C3 34, C2 33, C5 30, C4 28.

Selection is CONDITIONED on measured gate-5 oracle headroom, not asserted:
- C1 P1R-priority: gate-5 PASSED. Over a fixed GDumb buffer, an oracle sampling priority reaches 0.536 vs
  uniform GDumb 0.368 (headroom +0.169); a simple loss-priority reaches 0.463. Value-as-sampling-priority
  beats uniform GDumb. SELECTED for implementation.
- C2 V1-capable-family: gate-5 FAILED. Zero of four capable estimators (including an MLP) robustly decode the
  verification value beyond a strict single-feature control. The architecture-dependence is not a capacity
  artifact. FALSIFIED at the precompute gate.
- C3, C4, C5: ranked but deferred; their gate-5 needs a new harness (error-aware gym, causal multi-view,
  relational multi-entity stream). C3 is the immediate next thesis to build.

So the null map yields exactly one headroom-bearing new premise (C1) and cheaply falsifies one (C2). The
precompute is doing its intended job: promoting a genuinely new premise with demonstrated headroom and killing
a relabeling before principal compute.

## C1 P1R-priority: complete precompute design (all eight gates)
1. Causal hypothesis: value-weighted replay SAMPLING over a fixed class-balanced (GDumb) buffer beats uniform
   GDumb sampling under matched memory and compute.
2. Null: value-weighted sampling ties uniform GDumb (a tie is a null).
3. Strongest established method: GDumb (uniform over a class-balanced buffer); also reservoir, uniform,
   loss-sampling.
4. Oracle: sampling weighted by each item's true retention benefit.
5. Residual oracle headroom: MEASURED present (+0.169 over uniform GDumb on EMNIST, 2 seeds, reduced config).
6. Controls: random priority and shuffled priority must not beat uniform; oracle priority should.
7. Power analysis: per-task retention units across 5+ seeds; scale until the CI half-width is below SESOI 0.05.
8. Independent units: class-incremental tasks, seed-averaged.
9. Cheapest falsification: already run (the buffer-priority-vs-uniform check).

Principal C1 run is NOT launched. It is licensed to proceed to full implementation (all replay baselines,
5+ seeds, faithful established methods) only after this authority is committed and the eight gates are on record.

## What C1 must prove before compute is trusted
That the sampling-priority advantage survives against the FULL established-method suite (GDumb, reservoir,
uniform, loss-sampling, class-balanced sampling) at matched memory and compute across seeds, not only against
uniform GDumb, and that a random/shuffled priority does not reproduce the gain.
