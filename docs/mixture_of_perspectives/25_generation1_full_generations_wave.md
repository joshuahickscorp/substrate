# Generation 1 Full Generations Wave

> **Append-only successor:** This document extends
> [23_generation1_categorized_batch_wave.md](./23_generation1_categorized_batch_wave.md)
> and preserves the recovery topology in
> [24_generation1_successor_recovery_v5.md](./24_generation1_successor_recovery_v5.md).
> It does not rewrite either sealed wave, revive the pruned old G1-P1 lane, resurrect a retired D1
> design, or widen any predecessor evidence claim.

- **Status:** scaffolded behind a clean terminal categorized-batch-wave v1 result
- **Snapshot date:** 2026-07-17
- **Waiting and adopting parent:** `generation1-full-generations-extension-chain-v1`
- **Bounded child program:** `generation1-full-generations-wave-v1`
- **Idempotent whole-chain command:**
  `.venv/bin/python scripts/mop_generation1_successor_long_chain_v3.py start --execute`

**Claim scope:** append-only same-code mechanics robustness across carried and redesigned lanes,
D1 redesign rescreening, new-lane admission with canary gating, dependency-substituted
classification, single integration route, advisory release audit, and independent structural
artifact verification; no activation, scientific promotion, natural-world generality, or independent
scientific-generator claim.

## 1. Why a full-generations wave

The first three successor programs proved fresh-seed robustness first for the whole mechanics
inventory, then for two extension horizons, and most recently for a category-organized batch wave.
The full-generations wave continues that append-only shape across fourteen further fresh mechanics
cycles, 19 through 32, mapped to epochs W08 through W21. It keeps every fail-closed evidence boundary
identical while doing three new things that the categorized wave could not:

- it carries the completed frozen D1 retirement and no-candidate redesign freeze forward without any
  recompute;
- it rescreens the append-only D1-v2 redesign catalog, still unauthorized without real
  receipt-backed candidate evidence;
- it admits three redesigned mechanism lanes (`G1-U1`, `G1-N1`, `G1-P1R`) behind an explicit W08
  canary gate.

The generic supervisor still runs one top-level category capsule at a time. Each category capsule
uses one dynamic process pool capped at sixteen workers and publishes eight deterministic,
time-balanced planning-shard descriptors. Those descriptors are planning partitions, not
worker-pinned execution partitions. The honest parallelism is inside one category, followed by a
serial classify-and-seal barrier. No later epoch can start from partial sibling receipts.

## 2. Admission boundary

Admission binds the clean, exact, zero-injection terminal authority of the categorized batch wave v1
program `generation1-successor-categorized-batch-wave-v1`. The gate byte-binds that program manifest,
the generic supervisor clean status, its result, its verification, and its report receipt, then
extracts the final (W07) classification's surviving mechanics lanes. A mutually consistent set of
re-sealed shells is insufficient authority; the parent verification is independently rebuilt from the
bound result and raw artifacts and required to match semantically before any route decision.

The full-generations wave cannot start from partial parent progress. While the categorized wave, the
two earlier horizons, or either incumbent heavy campaign remain incomplete, the long-chain command
starts only lightweight observation and waiter parents and launches no compute.

## 3. Categories and lanes

Each epoch runs the same seven legible categories in fixed order 0 through 6.

| Order | Category | Included lanes | Purpose |
| ---: | --- | --- | --- |
| 0 | Formation and trace | G1-C0, G1-E1 | Trace stability and event formation |
| 1 | Communication and repair | G1-V1, G1-M1, G1-K1 | Verification, messaging, and repair |
| 2 | Memory and plasticity | G1-R1, G1-P1R | Retention and the redesigned stable-core route |
| 3 | Action and simulation | G1-A1, G1-S1 | Causal intervention and simulated consequence |
| 4 | Construction | G1-G1 | Structured construction and topology search |
| 5 | Dispatch mechanics and gated D1-v2 efficacy | G1-D1 | Value-of-compute mechanics plus an explicit redesign boundary |
| 6 | Uncertainty and curiosity | G1-U1, G1-N1 | Calibrated abstention and reducible-novelty curiosity |

Three lanes are new to this generation and enter only through the W08 canary gate:

| New lane | Mechanism | Verified dependency | Seed bands (canary, producer, challenge) | Advisory rate (s/seed) |
| --- | --- | --- | --- | ---: |
| G1-U1 | calibrated_uncertainty | G1-C2 | 171000001, 172000001, 177000001 | 0.000154417 |
| G1-N1 | reducible_novelty | G1-C0 | 182000001, 183000001, 188000001 | 0.000124245 |
| G1-P1R | stability_plasticity_r2 | G1-C0 | 193000001, 194000001, 199000001 | 0.000229500 |

Each new lane uses 256 canary seeds, 64 rungs per phase, and 65,536 seeds per rung. The seed bands
are disjoint from every sealed mechanics band by construction. The advisory rates are the measured
per-seed wall times times a 1.5x conservative planning factor, rounded to six significant digits;
observed throughput replaces them once real receipts exist.

### 3.1 The G1-P1 to G1-P1R substitution

The old G1-P1 `stability_plasticity` lane is retired and pruned. It never appears in the
full-generations categories. Its retirement is not a fresh decision here: it was recorded when the
old lane failed its canary gate in the mechanics-extended screen. In
`proof/GENERATION1_SUCCESSOR_MECHANICS_EXTENDED.json`, the G1-P1 lane carries
`canary_gate_passed: false`, `execution_decision: "prune_after_canary"`, an empty producer and
challenge (`long_work_executed: false`), and appears in the top-level `pruned_lanes` list. That
mechanics-extended canary prune of old P1 is the authority this wave names when it substitutes the
lane rather than reviving it.

`G1-P1R` (`stability_plasticity_r2`) is the doc-20 redesigned successor: a stable core with
orthogonal per-task adapters and selective consolidation driven by an honest recurrence signal that
the old bed never exposed, which is exactly why the old lane pruned. It enters under new authority
and must pass its own W08 canary partition. The old `stability_plasticity` files are untouched; the
redesign lives in the separate `stability_plasticity_r2` mechanism modules.

### 3.2 Vectorized construction execution

The `G1-G1` construction lane is by a wide margin the most expensive lane in the wave: its charged
`construction_search` bed accounts for roughly three quarters of the whole serial compute envelope.
This wave executes that one lane through a proven numpy-vectorized runner
(`src/mop/mechanisms/construction_search_vec_runner.py`, over
`src/mop/mechanisms/construction_search_vec_impl.py`) instead of the scalar bed. The vectorized runner
is an independent reimplementation that mints a receipt byte-identical to the sealed scalar runner for
every seed. The equivalence is proven at the receipt level over more than sixteen hundred real G1-G1
seeds (the canary, producer, and challenge bands carried through every fresh cycle) and again at the
rung level, where the wave's vectorized rung folds to the same `result_sha256` the scalar rung folds
to. Only the construction lane is routed this way; every other lane keeps its existing path unchanged.
The receipt, its folded digest, the verdict tally, and the seal are all identical, so no evidence
class, seed, threshold, or control moves.

Because the receipts are identical, the speedup is purely a pacing and ETA effect and never a receipt
value: the mechanics rung JSON carries no timing at all. On the review host the vectorized runner is
measured at 6.82x the scalar runner across the real 256-seed canary band; the wave applies a
conservative 6.7x planning factor to the construction pacing seconds only. Both vectorized modules are
pinned as runtime authorities in the sealed manifest, so the program authority chain freezes the exact
construction code that mints the receipts.

## 4. The five serial admission gates

The wave opens with five gates in fixed order before any epoch compute.

| Order | Gate id | What it binds and seals |
| ---: | --- | --- |
| 0 | `admit_wave_v1` | Byte-binds the categorized wave v1 manifest, supervisor status, result, verification, and report receipt; rebuilds the parent verification; admits the surviving lanes. |
| 1 | `carry_d1_freeze` | Re-validates the parent's sealed D1 registration and no-candidate freeze from the v1 result with no D1 recompute and no re-retirement; seals `carried_forward=true`, `old_d1_retired=true`, `resurrection=false`, `execution_authorized=false`. |
| 2 | `rescreen_d1_redesign` | Calls the D1 redesign-v2 candidate catalog, seals candidate ids and a recomputed `candidate_evidence_count` (expected 0); `execution_authorized=false` always; if evidence ever exceeds 0 it records `eligible_for_future_authority=true` but still never executes. |
| 3 | `admit_new_lanes` | Seals the three new LaneSpecs, their bed and runner module paths and sha256s, the W08 canary policy, and the G1-P1 to G1-P1R dependency substitution with the justification string naming the old canary prune receipt path. |
| 4 | `freeze_routing` | Seals the initial route as the surviving v1 lanes union {G1-U1, G1-N1, G1-P1R}, and the initial I1 eligibility from the substituted dependency set. |

Gate 1 and gate 2 are strictly byte-binding and catalog replays. They never re-run D1 compute, never
re-retire the old design, and can never fabricate efficacy evidence.

## 5. One readable dependency graph

```text
clean categorized-batch-wave v1
  -> gate 0 admit_wave_v1 (bind manifest, status, result, verification, report; admit survivors)
  -> gate 1 carry_d1_freeze (carry completed D1 retirement and no-candidate freeze; no recompute)
  -> gate 2 rescreen_d1_redesign (recompute candidate evidence; still unauthorized)
  -> gate 3 admit_new_lanes (seal G1-U1, G1-N1, G1-P1R; canary policy; P1 -> P1R substitution)
  -> gate 4 freeze_routing (route_seed = survivors union new lanes; i1 initial eligibility)
  -> W08 seven categories (new-lane canary partition) -> W08 classify + seal
  -> W09 seven categories -> W09 classify + seal
  -> ...
  -> W21 seven categories -> W21 classify + seal
  -> G1-I1 dependency gate (substituted deps) + integration batch or null-safe prune
  -> integration classify + seal
  -> aggregate
  -> independent structural verifier
  -> report receipt
  -> advisory release audit (never gates activation, promotion, or confirmation)
```

`G1-I1` is evaluated physically once after W21 at cycle 32. It executes only if gate-4 eligibility
holds and its substituted dependency closure over `G1-E1`, `G1-D1`, `G1-M1`, `G1-V1`, `G1-R1`, and
`G1-P1R` survives W21. Otherwise it seals a null-safe pruned receipt with zero receipts. Other
category results stay visible in the aggregate and are never silently promoted into new causal
premises for I1.

## 6. Hour ceiling and inventory

The manifest contains 123 top-level capsules: five gates, fourteen epochs of seven category compute
capsules plus one serial classification barrier each, one integration compute capsule and one
integration classification, one aggregate, one independently authored verifier, one report receipt,
and one advisory release-audit capsule.

The sealed manifest is `configs/campaign/generation1_full_generations_wave_v1.json` with
`program_sha256 = 5efc3bd48d6b490b79cbb04ab9e1e50c3f63550db16976796618abc7ab8537a3`. It binds the
throttle policy `configs/local_execution_throttle_v6_full_generations.yaml`, and every category and
integration compute capsule declares a sixteen-core idle-host pool while the serial gate, classifier,
aggregate, verifier, report, and release-audit capsules stay single-core.

Across the executable mechanics capsules the program can schedule up to 35,255 checkpointed fresh
work items: 2,509 category items per epoch across fourteen epochs, plus 129 substituted-dependency
integration items.

| Route | Maximum serial compute | Ideal wall time at sixteen workers |
| --- | ---: | ---: |
| Executable full-generations mechanics | approximately 136.2 hours | approximately 8.5 hours |

The construction lane's vectorized execution (section 3.2) drives this envelope down from the earlier
scalar ceiling of approximately 401.5 serial hours (25.1 ideal-worker hours). Construction was about
77.7 percent of that scalar envelope; at the conservative 6.7x construction planning factor its share
falls from roughly 311.9 to 46.6 serial hours, so the whole-program envelope drops to approximately
136.2 serial hours (8.5 ideal-worker hours), an overall 2.95x pacing reduction. The item counts, seeds,
receipts, and every seal are unchanged; only the planned pacing and ETA figures move.

These are ceilings, not forced runtime promises. Classifiers may honestly prune a failed mechanism,
a failed new-lane canary, or a no-candidate D1 route. Work is never revived merely to consume an hour
budget.

## 7. Advisory result-aware reprofiler

A separate result-aware reprofiler reads completed sealed aggregates and deterministically
re-derives observed per-mechanism throughput, then recommends an idle-host worker count. On the
current review host it recommends six workers, bound by the memory term rather than the core or
hard-ceiling terms. The recommendation is deterministic given host cores, memory, and the per-capsule
memory cap; the same inputs always yield the same six-worker advisory.

This is advisory operational telemetry only. It is read-only over completed programs, every artifact
it seals carries `advisory: true`, and it can never change an evidence class, a seed, a threshold, a
control, or any sealed receipt. The sixteen-worker planning envelope in section 6 is the manifest
ceiling; the six-worker default advisory only informs how an operator might schedule an admitted run.

## 8. Frozen routing rules

Each epoch seals a classification before the next begins. Completed work is immutable; routing
changes only future eligible work, and every prune stays visible in the aggregate.

| Observation | Later action |
| --- | --- |
| A carried lane admitted through the parent survivor set remains clean | Continue that lane on the next fresh cycle. |
| A new lane's W08 canary partition is all mechanics-ok | Admit that lane from W09 onward. |
| A new lane's W08 canary partition contains any non-mechanics-ok verdict | Prune that lane from W09 onward; no downstream epoch runs it. |
| A carried or admitted lane warns or fails inside an epoch | Prune that lane from later epochs. |
| A substituted I1 dependency lane fails before W21 completes | Seal the I1 null-safe prune with zero receipts. |
| The carried D1 retirement or the D1-v2 rescreen changes premise | No effect; D1 stays retired and unauthorized. Never resurrect, never execute. |
| A receipt, source, seal, or verifier binding drifts | Hold the program; do not count partial work. |

## 9. Host and launch safety

The full-generations compute class is idle-host-only and uses at most sixteen internal workers. It is
not added to the Hawking coexistence whitelist. While incumbent heavy workloads or any earlier
program remain incomplete, the long-chain v3 command starts only lightweight observation parents.

The long-chain v3 launcher starts or resumes, in order, the append-only recovery v5 chain and its
downstream waiters, then the full-generations extension waiter
`generation1-full-generations-extension-chain-v1`. Every component is independently locked and
idempotent. Repeating the command cannot duplicate a legacy queue, a horizon supervisor, a
categorized supervisor, a full-generations supervisor, or any waiter. The extension waiter cannot
create its child process until the complete categorized wave v1 state, status, program, terminal
artifacts, and process authority replay without reconciliation.

## 10. Verification boundary

After W21, the substituted-dependency integration route, and the advisory release audit, the
aggregate binds all epoch classifications, shard receipts, raw artifacts, pruning decisions, and seed
authorities. The independent verifier reuses the separately authored streaming verifier family
against the new program identity and independently checks the admission boundary, the exact parent
manifest and terminal supervisor authority, the replayed parent verification, the new-lane admission
and canary policy, the routing inventory, the raw receipts, seed disjointness, the substituted I1
route, and semantic mutation rejection. Its claim scope is an exact sealed field set and cannot be
widened into activation or scientific-confirmation authority.

This is independent artifact verification. It is not a separately implemented scientific generator,
so `independent_scientific_confirmation` remains false, and every artifact still declares
`activation_allowed: false` and `scientific_promotion: false`. The release-audit capsule is advisory:
it captures the release-audit module's full JSON report and exit status without gating any
activation, promotion, or confirmation, and only a crash of the audit module itself can fail it.

## 11. Completion condition

The full-generations wave is complete only when:

- the categorized wave v1 and every earlier program are terminal and clean;
- the five admission gates are sealed with the carried D1 freeze, the rescreen, the new-lane
  admission, and the frozen routing;
- W08 through W21 each seal every eligible category shard, the serial classifier, and every honest
  canary or lane prune;
- the physically last substituted-dependency integration batch is sealed or null-safe pruned;
- the aggregate is complete;
- the independent artifact verifier is clean;
- the generated report receipt binds the exact result and verification;
- the advisory release-audit capsule is sealed with `advisory: true`;
- the generic supervisor status still contains the exact 123-capsule inventory, zero accepted
  injections, and current clean artifact reports for every completed capsule;
- every artifact still declares `activation_allowed: false` and `scientific_promotion: false`.

Until then, the exact description is: **the full-generations wave is queued behind the categorized
batch wave and the earlier successor horizons, and may only wait without launching compute**.
