# Failure taxonomy (the 10 categories + badge legend)

Every null maps to exactly one of these categories (Section 10.5). A null is an asset,
not a loss: better-funded groups are incentivized toward positive novelty and leave
their nulls in the drawer, so a packaged, citable corpus of nulls is the defensible
lane. Form per BLACKHOLE.md: no em dashes; engineering vocabulary only.

| # | Category | One-line meaning |
|---|---|---|
| 1 | Biology-mapping adds no measured benefit | The developmental name adds complexity, the metric does not move. |
| 2 | Effect explained by a simpler control | A trivial baseline already captures it. |
| 3 | Frozen latent lacks (or gains) the needed factor | Substrate blind spot or substrate gift; the probe gate decides. |
| 4 | Capacity/estimator too weak | Predictor, generator, hypernet, or codebook under-sized. |
| 5 | Stream too uniform/short for structure to appear | No curriculum or meta-structure to learn at this scale. |
| 6 | Tiny shell capacity bound | The task is too hard for the shell regardless of mechanism. |
| 7 | Needs embodiment/action (Tier R) | Requires a live environment to act in; deferred, not failed here. |
| 8 | Only helps combined (hybrid) | Pure mechanism fails; mechanism-plus-anchor or plus-replay wins. |
| 9 | Representational vs compute/locality claim separated | The gain is a compute/locality property, not a representational one. |
| 10 | Conceptually beyond frozen-latent prediction | The structure (e.g. interventional) is out of reach of the substrate. |

## Verdicts and badges

The failure template IS the null card (Section 10.3) with `verdict` in {DOWNGRADE-TIE,
SUBSTRATE-BOUND, SEED-UNSTABLE, CAPACITY-ARTIFACT} and the matching badge set.

| Badge | When it applies |
|---|---|
| seed-instability | the sign flips across seeds (publish the instability, not a positive) |
| capacity-artifact | the effect is a parameter-count difference, removed by the capacity-matched ablation |
| substrate-blindspot | the needed factor is not decodable from the frozen latent (taxonomy 3) |
| tuned-baseline-tie | the mechanism ties a properly tuned baseline (e.g. tuned cosine decay) |

## How a failure alters the roadmap

- Category 3 (substrate bound): retarget to the dense V-JEPA 2.1 path when it ships, or
  to a different factor.
- Category 4 (capacity artifact): retarget to a capacity sweep before any further
  mechanism claim.
- Category 8 (hybrid): retarget to the combined arm as the real result.
- Full pivot: if EX12 shows the atlas factors the whole campaign depends on are broadly
  not decodable from any encoder scale, the mechanism campaign stops and Brain ships the
  atlas-as-bound as the primary contribution.

(Stub: the populated obituary entries live in OBITUARIES.md as mechanisms are retired.)
