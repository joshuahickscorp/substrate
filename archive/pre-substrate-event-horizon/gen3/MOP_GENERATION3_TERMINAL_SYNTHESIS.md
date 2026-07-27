# MOP Generation 3 Discovery: Terminal Synthesis

Program `mop-generation3-discovery-v1`, branch `agent/mop-generation3-discovery` off `89eeca5`. Generation 2
immutable and untouched. House rules: no dashes, no git attribution, a tie is a null, no positive without
independent adversarial verification. Activation false.

## Result: both new premises fail; no new subsystem survives
Generation 3 tested one revised replay premise (C1) with precompute-suggested headroom and one genuinely new
simulation premise (C3) derived from a valid Gen2 null. Both reach terminal null boundaries. No mechanism,
cluster, construction, integration, or activation is licensed. P1R remains the strongest surviving internal
hypothesis from Generation 2, unchanged.

## C1: P1R replay value as a soft sampling priority
Premise: replay value is more useful as a soft SAMPLING PRIORITY over a fixed class-balanced buffer than as a
keep/eviction filter, with the buffer maintained independently of the priority estimator.

Terminal classification: **invalid_bed across all three tested regimes.**
- EMNIST full budget: invalid (oracle below uniform, controls misbehave, seed-unstable).
- HAR (non-image, subject-disjoint activity recognition): invalid (all methods about 0.92; GDumb-uniform
  already captures nearly all value; oracle headroom +0.006).
- EMNIST high-forgetting regime (five seeds): invalid. This is the decisive finding. The precompute's +0.169
  oracle headroom was a TWO-SEED ARTIFACT. Measured over five stable seeds at the same regime, oracle_priority
  (0.320) beats uniform GDumb (0.309) by only +0.010 (below the 0.02 validity threshold), and shuffled and
  random concentration (0.318, 0.314) match the oracle. GDumb-uniform captures essentially all recoverable
  value at every stable budget.

Separately, the LEARNED P1R priority is actively harmful where a tiny oracle edge exists: 0.18 to 0.19, worse
than uniform, loss-priority, oracle, shuffled, and random, with per-task effect lower-95-CB from -0.15 to
-0.21. Concentrating replay by a noisy learned value reduces diversity and hurts.

Meta-finding: the two-seed precompute gate produced a false-positive headroom signal that a proper five-seed
power analysis overturned. Adequate seeding (the mandate's power gate) is decisive; a cheap precompute can
over-promise.

## C3: model-error-aware bounded simulation
Premise: allocate bounded simulation only where expected decision benefit exceeds expected model-error cost;
condition depth and trust on estimated rollout error (the new premise vs Gen2 S1, which was worse than random).

Terminal classification: **precompute gate failure, no canary.** Both decisive gates fail. Only 2 of 10 env
units are informative (both CartPole; Acrobot and MountainCar are unsolvable by a linear model). On those, the
oracle error-aware allocator is WORSE than reactive (mean about -0.54); it does not beat reactive (gate 5) or
the fixed-depth planner (gate 6). Even a perfect model-error oracle cannot make learned-model simulation help
on these environments. C3 is closed before any principal canary, continuing the Gen2 S1 null. No principal
compute spent.

## Surviving subsystem
None new. P1R remains the strongest surviving internal hypothesis: same-team benchmark positives bounded by an
external replication null, with no construction, integration, or activation license.

## Forbidden claims
- P1R-priority is not a validated replay mechanism: its headroom is a seed-noise artifact and the learned
  priority is harmful on the only regimes with any oracle edge.
- C3 model-error-aware simulation has no headroom.
- No new mechanism is confirmed or activated.

## Exact next frontier
The null map's replay and simulation premises are now exhausted at this compute scale. A genuinely new premise
must target a domain with a strong non-image sequential source that has BOTH real forgetting AND oracle
headroom the established method leaves open (HAR had neither; EMNIST had forgetting but no stable headroom).
Absent such a source, the honest position is that MOP has a strong falsification program and efficient
orchestration but no mechanism with externally replicated incremental value, and further mechanism search
should pause until a qualitatively different data authority or premise is available.
