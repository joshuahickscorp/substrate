# Local action environment and trajectory contract

## Outcome

The repository now has a deterministic persistent local action environment. It removes the
environment-adapter implementation gap for bounded mechanics on this device. It does **not** turn a
programmatic grid into natural embodiment, natural affordance evidence, or an open-ended ecology.

Canonical artifacts:

- implementation: `src/mop/environments/persistent_grid.py`;
- profile: `configs/environment/local_persistent_grid.yaml`;
- runner: `scripts/local_action_environment.py`;
- durable proof: `proof/LOCAL_ACTION_ENVIRONMENT.json`;
- full per-seed trajectories: `runs/local_action_environment/trajectory_seed_*.json`.

The three-seed run completed in about two seconds and remained below 263 MB peak RSS. All mechanics
checks passed.

## What a trajectory contains

Every actual row binds:

1. the observation before the decision;
2. the action actually selected by a named deterministic policy;
3. the next observation and structured consequence;
4. action cost, reward, movement/blockage, goal and noisy-TV events;
5. episode-start and episode-end flags;
6. stable world, episode, event, entity, state, and branch references.

Before the chosen action is committed, all four alternatives are evaluated from the **same cloned
state**. The bundle therefore contains a four-branch paired counterfactual group per actual event.
World specifications and complete bundles have canonical SHA-256 identifiers. The verifier rebuilds
the world from the recorded seed, replays every chosen action, recomputes every alternative branch,
and compares the canonical bytes. Mutating a consequence or reference fails verification.

The 20260709-20260711 run produced 782 actual transitions and 3,128 cloned-state alternatives across
72 episodes. Each actual sequence used all four action classes; every counterfactual group was
complete; all episode splits were disjoint.

## What changed by experiment

| Row | What is local now | What is still outside the proof |
|---|---|---|
| F6 | observation -> chosen action -> next-state records, costs, boundaries, exact action-blind and action-shuffle construction | natural sensorimotor or physical embodiment |
| F15 | paired alternative consequences, state-conditioned affordances, stable entity refs, passive and shuffled-consequence controls | affordances of natural objects under real interventions |
| E5 | actual learnable and noisy-TV trajectory regions with replay-stable hidden sensor variation | generalization to independent ecologies or natural trajectories |
| CM10 | train/test episode split, same-shape true/blind/shuffled forward models, matched planner calls, and execution on true local dynamics; P7 now adds rendering, same-parent branches, eight arms, and equal-core ledgers | independently sourced trajectories, an exact-referent action control, and predeclared replication |
| E10 | persistent action mechanics | population search, environment generation, sustained non-plateau evaluation, and cross-environment transfer |

E5 is therefore `cpu-now`. CM10 is no longer blocked on an environment adapter; it is now
upstream-evidence gated. E10 remains deferred because a persistent finite world is not an
open-ended environment generator.

## CM10 pilot interpretation

On every seed, true actions improved held-out one-step R2 over the stronger blind/shuffled action
control by about 0.073-0.080. The action-conditioned planner also beat both action controls on local
goal completion by 0.33-0.67. However, the exact-state reactive control scored 0.83-1.00 while the
learned planner scored 0.33-0.67. The current programmatic pilot therefore **does not** satisfy
CM10's registered claim. It proves that the causal and control plumbing works, and it supplies a
useful negative: action conditioning is necessary here but not sufficient to beat the strongest
reactive policy.

The receipt sets `scientific_ready: false` and `scientific_promotion_allowed: false`. A larger box
cannot fix those missing referents or controls.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/local_action_environment.py
.venv/bin/python -m pytest \
  tests/unit/test_persistent_grid_environment.py \
  tests/unit/test_local_action_environment.py \
  tests/integration/test_e5_curiosity.py \
  tests/integration/test_e10_openended.py \
  tests/integration/test_f_missing_lanes.py -q
```

Changing the seed, world geometry, horizon, policy, costs, or event semantics changes the content
hash. A result may use this adapter as a mechanics fixture, but natural or embodied promotion must
name and hash an independent evidence source.
