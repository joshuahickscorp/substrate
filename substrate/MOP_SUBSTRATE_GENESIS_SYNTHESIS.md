# MOP Owned Substrate Genesis II: Terminal Synthesis

Program `mop-owned-substrate-genesis-v0`, branch `agent/mop-substrate-genesis-v2` off Gen3-terminal `06860b2`.
Gen2/Gen3 immutable; collapse branch untouched. House rules: no dashes, no attribution, a tie is a null, no
positive without independent adversarial verification. Activation false.

## Result
Two materially different owned-substrate architectures were implemented, contracted, and tested on two
principal domains (one non-image) against strong matched baselines, with 5-seed inference and independent
verification. **Both architectures are `substrate_candidate_null`.** No Owned Substrate v0 candidate is
selected. The owned multi-timescale substrate, at matched budget, does not robustly beat GRU+GDumb on
cost-adjusted moldability. This continues the program-wide finding that nothing beats strong established
alternatives under honest matched conditions.

## Answers to the synthesis questions
1-3. An owned latent substrate WAS implemented: owned trainable projection + latent workspace/modules + slow
representation + fast GRU state + heads are MOP-owned; frozen providers (raw pixels, HAR sensor features)
supply only observations.
4-6. Immediate = per-sequence fast GRU state; episodic = bounded GDumb replay memory; slow = projection and
workspace/module parameters under eligibility.
7-10. Eligibility = simple param-group masks; memory = GDumb (established, no learned selector); consolidation
= EWC at task boundaries; routing = context-gain (A) or top-2 sparse module router (B).
11-15. It learns new tasks and retains prior tasks with replay, but does NOT beat the strong baselines; on HAR,
Architecture A's mean utility exceeds the best baseline by +0.112 but the 5-seed lower-95pct-CB is -0.028 (a
near-miss, a null). Future-adaptation and shift handling were within the matched-budget comparison.
16-17. Cross-context transfer is inside each domain; cross-domain moldability is `cross_domain_moldability_null`
(a transfer positive requires a substrate positive, which does not exist).
18-20. Updates localize under eligibility, but Architecture B's modular routing did NOT reorganize usefully:
B is dominated by the simpler A (decorative modularity); routing is not necessary here (`structural_null`).
21-22. Simple plasticity policies suffice; the substrate-specific plasticity headroom gate is `no_stable_headroom`
(oracle eligibility 0.664 vs simple 0.662, lcb -0.001), so no learned controller is opened.
23-26. Architecture A is the stronger of the two on every domain, but neither is a candidate. Both beat the MLP
baselines; neither robustly beats GRU+GDumb (the strongest baseline).
27. Cost: A 216862 params, B 389604 params (both < 2M); 837 LOC (< 6000 budget); CPU-first; two domains ran
concurrently.
28. What failed: Architecture B (modular routing overhead without benefit); the substrate advantage over the
strongest baseline (a HAR near-miss that did not survive the 5-seed lower-CB).
29-31. All evidence is real-domain (EMNIST, HAR) plus calibration controls; the evidence ceiling is that no
owned architecture beats GRU+GDumb on cost-adjusted moldability at matched budget.
32-34. Owned Substrate v0 is NOT a candidate (both null). No activation is licensed. For v1: a temporal-native
domain where the fast timescale carries real information (image domains make fast state degenerate), or a
domain where the strong baseline genuinely leaves stable headroom at a converged budget.

## Terminal evidence classes
- Architecture: substrate_candidate_null (both A and B, both domains); independently verified consistent.
- Cross-domain: cross_domain_moldability_null.
- Plasticity: no_stable_headroom (simple_policy_sufficient).
- Routing: structural_null (B modular routing dominated by A).
- Selection: no_substrate_candidate_selected.

## Forbidden claims
The substrate did not succeed by more params/memory/updates/data/time/frozen-encoder/state-change. No learned
controller is validated. No activation is licensed.

## Exact next frontier
The owned-substrate premise at this scale does not beat strong established continual learners under matched
budget. The honest next step is a temporal-native domain (real sequences, so the fast timescale is not
degenerate as it is on iid images) with a strong baseline that leaves stable headroom at a converged budget,
before any further learned controller. Absent a matched-budget win there, mechanism and substrate search
should pause: MOP has a strong falsification program and efficient orchestration, and a real, compact,
self-verifying owned substrate implementation, but no component or architecture with externally replicated
incremental value over strong established alternatives.

## Resume commands (if reopened)
- resume: read MOP_SUBSTRATE_GENESIS_STATE.json and continue each checklist item from next_exact_action
- status: cat runs/substrate/mop-owned-substrate-genesis-v0/status.json
- verify: python substrate/verify_substrate.py
- close-if-terminal: python substrate/build_substrate_synthesis.py --close-if-terminal
