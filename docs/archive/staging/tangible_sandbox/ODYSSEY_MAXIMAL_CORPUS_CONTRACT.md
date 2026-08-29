# Odyssey maximal corpus contract

## Decision

The 7-day Odyssey needs a **replayable, independently held-out task bank for
each of its eight frontiers**. A large undifferentiated training scrape would
not improve the primary claim: it would obscure rights, contaminate held-out
evaluation, and make candidate/control parity impossible. The maximal useful
corpus is therefore a shared, read-only resource lake plus separately seeded,
evaluator-only task manifests.

No item below is adopted into an Odyssey arm until its rights, exact revision
or object manifest, candidate/control parity, and evaluator-only split are
sealed after R2.

## Resource lake

| Frontier | Maximal useful resource | Status | Predictable storage budget | What is actually scored |
| --- | --- | --- | ---: | --- |
| A: integrated continuity | STSC recompositions + project-state histories + selected cross-modal packets | Local base present | 8 GiB generated deltas | continuity, correction, and recovery across days |
| B: formal mathematics | MATH cache; pinned proof tasks and verifier traces | MATH prefetched | 4 GiB task/trace lake | derivation persistence and verified transfer |
| C: formal logic | pinned rule/constraint/proof instances, countermodels, solver traces | plan only | 4 GiB | stateful counterexample and proof repair |
| D: software | existing SWE-bench data + pinned permissively licensed repositories + test traces | metadata local | 24 GiB | issue history, patching, and test-grounded recovery |
| E: philosophy/self-model | curator-authored prompts, commitment ledgers, blinded revisions, licensed source excerpts only | plan only | 2 GiB | perspective tracking and belief/commitment revision, not a claim of selfhood |
| F: sound/speech | retained LibriSpeech and license-filtered FSD50K, with timestamped repair tasks | 80.13 GiB local | 0 GiB new raw data; 4 GiB derivatives | temporal grounding and post-disturbance recovery |
| G: vision/spatial/embodied | MOVi-A physical video/state corpus, CLEVR, plus deterministic Blender/Kubric rollouts | active public prefetch | 82.93 GB raw cache + 24 GiB generated rollouts | action/state prediction, spatial repair, and replayable counterfactuals |
| H: science/multimodal | ARC plus predeclared telemetry/table/image packets and causal-inference tasks | ARC prefetched | 6 GiB | hypothesis revision against mixed evidence |

The post-R2 target is therefore roughly **76 GiB of deterministic/generated
task material** in addition to retained data and the public prefetch. This is
an upper budget, not a blind download target; it is generated only after an
8-cell pilot measures actual storage amplification.

## Embodiment/proprioception lane

MOVi-A supplies an unusually complete observational state: RGB, depth,
segmentation, optical flow, object position, velocity, quaternion, mass,
friction, restitution, camera state, and collision events. It is the shared
perception/physical-state base. It does **not** by itself make an active-body
experiment.

For active proprioception, the correct addition is a deterministic simulator
pack rather than an uncontrolled video corpus:

1. Freeze the Blender/Kubric generator revision and simulator configuration.
2. Precommit seeds, action distributions, episode lengths, and domain-randomization ranges.
3. Store action vectors, joint/object state, camera transforms, observations,
   contacts, rewards only when relevant, and reset/replay receipts.
4. Create candidate-visible trajectories and evaluator-only counterfactual
   continuations from disjoint seed ranges.
5. Measure bytes per episode in a small no-score pilot; project the full 24
   GiB allocation with a 2x transient-write allowance, then set the final
   storage guard from that measurement.

This gives real controllable action/state evidence, supports blind transfer,
and makes storage predictable. Downloading an opaque robotics archive before
its action semantics, licensing, and replay contract are known would add
bytes without adding valid Odyssey evidence.

## Staging sequence

1. **Current public prefetch:** MATH, ARC, SWE-bench multimodal metadata,
   MOVi-A 128px, and CLEVR. The downloader has a hard floor equal to the live
   R2 guard plus 100 GiB user reserve.
2. **After R2 verification:** run a non-scoring 8-cell resource rehearsal.
   It measures working-set RAM, bytes per checkpoint, bytes per generated
   episode, and bandwidth interference. It may not alter candidate/control
   content based on observed scores.
3. **After rehearsal:** materialize only the stated generated budgets and
   seal task manifests, custody splits, and hashes.
4. **Separate consent queue:** real-world video, Common Voice, hosted
   simulator assets, OSWorld, WorkArena, MLE-bench, and any robotics set with
   data-use terms or credentials. Nothing in this queue is accepted merely
   by downloading it.

## Storage calculation

Synthetic output is predictable enough to budget, but the final number must
be measured, not guessed:

```text
final_required_free = protected_dynamic_floor
                    + P95(8-cell 7-day-equivalent growth)
                    + 2 * largest measured transient write
                    + terminal allowance
                    + user reserve
```

The shared raw lake is read-only and counted once through actual free space;
only newly selected generated data, frontier-private deltas, and terminal
allowances are added. A later storage expansion may increase capacity, but it
does not remove the requirement to recompute this formula immediately before
launch.
