# MOP Null-Derived Design Constraints

Every valid Generation 2 null is converted into a reusable admission constraint. These are mandatory clauses for
every future mechanism, in addition to the calibrated battery. A mechanism that cannot satisfy the constraint
its predecessor violated is a relabeling, not a new mechanism. No dashes.

| Origin | Constraint |
|---|---|
| U1 (uncertainty) | High uncertainty is not reducible uncertainty. A mechanism must discriminate reducible from irreducible error, not fire on raw uncertainty. |
| N1 (novelty) | Source-specific reducibility does not imply cross-source value. A value signal must transfer across sources before confirmation. |
| R1 (retrieval) | Learned retrieval must beat nearest-similarity by SESOI, not merely perform a successful nearest-neighbour lookup. |
| E1 (event formation) | Relational event boundaries must beat simple change detectors in downstream value, not merely segment the stream. |
| C0 (trace stability) | A stable trace must beat EMA smoothing and matched-memory buffers in downstream value. |
| A1 (affordance) | Affordance reading must beat a fitted value estimator on the same latent. |
| S1 (simulation) | Learned rollouts must control compounding model error and beat direct prediction and simple planning after charging model-error estimation. |
| M1 (messaging) | Message predictability is not causal message value. A messaging mechanism must estimate intervention-level value and beat a centralized matched-capacity control. |
| K1 (repair) | Contradiction detection is not repair necessity. A repair mechanism must estimate net corrected-decision value and avoid false-repair harm. |
| P1R (replay) | Per-item value prediction is not sufficient for a competitive replay policy. A replay mechanism must beat the best established method (GDumb), not only no-replay. |
| V1 (verification) | Incremental value must hold across a family of sufficiently capable architectures, not a single one, to be admitted. |
| Battery | Every mechanism must clear noisy-TV, shuffled-target, wrong-time, and rate-matched-random controls, and provide oracle headroom, before a canary. |

## How to use
Before a mechanism enters admission it must state, for each relevant constraint above, the specific control it
beats and by what margin. The constraint set is the accumulated cost of ten falsified premises; a new mechanism
earns attention only by expressing a premise the null map does not already forbid.
