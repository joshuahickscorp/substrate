# P7 rendered action-world-model shared-mechanics audit

## Audit outcome

There was a real local implementation gap, but it was smaller than “build a new world-model
platform.” The repository already had the causal core:

- `persistent_grid.py` owns deterministic hidden-state transitions, observation vectors, stable
  world/episode/event/entity/state references, and four action branches evaluated before the chosen
  action mutates the cursor;
- Wave E0 owns immutable state/intervention/consequence bytes and exact branch replay digests;
- F6 already executes action-conditioned, action-blind, and action-shuffled transition models;
- the local CM10 pilot already fits a small forward model, plans through it, and executes the selected
  action on true local dynamics;
- E5 already exposes the learnable/noisy-TV split; and
- the shell already has a small action-conditioned `Predictor` independent of an inherited encoder.

Creating another environment, event graph, or generic action predictor would have duplicated those
surfaces. The missing shared seam was a deterministic raster observation bound to the exact hidden
state, plus one comparison contract covering rendered compact transitions, object-centered
transitions, direct reactive and recurrent controls, an oracle, the two required action controls,
and a matched-depth control under one held-out-intervention and cost ledger. P7 implements only that
seam.

## What P7 adds

The new renderer produces a 12 by 12 RGB frame directly from a six by six persistent-grid state.
Walls, goal, agent, agent-on-goal, and the noisy-TV cell have explicit programmatic colours. No image
library, encoder, checkpoint, download, accelerator, or inherited latent is involved. Noisy-TV pixels
are derived from the world hash, complete hidden-state payload, and pixel coordinate, so a repeated
render is byte exact while a state or seed mutation changes the image.

Each selected pre-action event retains all four branches. For every branch, the fixture binds:

1. the persistent event, group, branch, before-state, and after-state references;
2. the complete before and after hidden-state payloads;
3. the action, consequence, chosen flag, and vector-observation hashes;
4. before and after raster shapes, byte counts, and SHA-256 digests; and
5. a Wave E0 `BranchMeta` digest over one immutable parent state, the action intervention, and the
   resulting consequence state.

The verifier reconstructs the exact world from the recorded seed and layout, recollects the original
persistent trajectory, reselects the same budgeted event roots, rerenders every frame, and compares
canonical bytes. It rejects outer-hash-repaired mutations to a raster digest, hidden state, action,
branch consequence, environment seed, or declared budget.

## Independent units and exact budgets

The fixture uses three independent seed/layout pairs. Layout identity includes the grid size, ordered
wall cells, and noisy-TV cell; world identity additionally includes the seed and horizon. Episode
ranges are disjoint by purpose:

- 12 episodes supply training event roots;
- six different episodes supply held-out interventions and multi-step rollouts; and
- four further episodes are used only for executed planning on true dynamics.

Per unit, the fixed data budget is 32 training event roots and 16 held-out event roots. Four actions
per root yield 128 training branch actions and 64 held-out branch actions. Forty-eight held-out rows
are unchosen same-parent interventions. Eight rollout roots are required without crossing an episode
boundary. The branch materialization creates 192 before/after render pairs, or 384 render receipts.
Planning has 32 maximum real-action opportunities per arm and unit; early terminal states reduce real
actions but do not reduce the declared equal-compute arms’ inference budget.

## Arm contract

| Arm | Input and objective | Transition output | Role |
|---|---|---:|---|
| `reactive_rendered` | palette-decoded current agent, goal, walls, and legal moves | no | direct rendered baseline |
| `model_free_recurrent` | compact render history and prior action, direct action-value fit | no | recurrent model-free baseline |
| `compact_latent_transition` | 12 generic global RGB moments plus action | yes | primary compact rendered world model |
| `object_centered_transition` | agent/goal/affordance slots decoded from pixels plus action | yes | primary object-centered world model |
| `oracle_state` | exact hidden state and the true simulator | exact | upper control, never a learned claim |
| `action_blind` | compact render latent with the four action inputs zeroed | yes | negative action control |
| `action_shuffled` | compact render latent trained against a deterministic nonidentity action permutation | yes | negative correspondence control |
| `matched_depth_reactive` | identical predictor shape and search depth, trained for immediate potential while its state is held fixed | no | unrolled reactive depth control |

The five learned predictor controls use the existing shell `Predictor` with 12-dimensional input
state, four action inputs, two 24-wide hidden layers, the same initialization seed, 60 full-batch
updates, and a same-shape four-coordinate decoder. They therefore have identical trainable parameter
counts. At planning time, each scores every length-three four-action sequence. Terminal episodes are
compute padded. Model forward calls, decoder calls, linear-layer active MACs, and trainable parameters
are exact within this group. The proof does not pretend optimizer targets, activation constants, or
nonlinear scoring semantics are identical.

The recurrent baseline is deliberately model free: it fits direct branch action values from a leaky
history state and never predicts a next observation. The reactive and matched-depth controls likewise
receive `prediction.applicable: false`; inventing prediction numbers for non-transition models would
make the comparison less trustworthy.

## Metrics and interpretation boundary

Transition arms report held-out one-step coordinate R2, agent-position RMSE, exact agent-cell
accuracy, unchosen-intervention accuracy, deterministic rendered-pixel MSE, multiclass Brier score,
and expected calibration error. The same arms report agent RMSE, exact cell accuracy, and render MSE
at horizons one, two, and three. Every arm reports executed true-dynamics goal success, return, action
cost, real actions, padded opportunities, and success/return deltas against the reactive rendered
baseline. Cost records include parameters, update count, training rows, inference calls, decoder
calls, and active linear MACs.

These are fixture diagnostics, not evidence that one architecture understands or imagines a natural
world. The oracle sees hidden state. The rendered scene is tiny and programmatic. The compact and
object features are deterministic engineering choices, not learned visual perception. Planning and
prediction episodes share a world generator even though their episode identities are disjoint.

## Measured bounded result

The fixture is a useful negative. Across layouts alpha, beta, and gamma, the reactive rendered policy
reached the goal on 1.00, 0.75, and 1.00 of planning episodes; the oracle reached 1.00 on all three.
Both primary learned transition arms reached 0.00 on all three. The action-shuffled control reached
0.25, 0.25, and 0.00, while action-blind, matched-depth reactive, and model-free recurrent controls
reached 0.00 throughout.

Object-centered one-step exact agent-cell accuracy was 0.09375, 0.359375, and 0.171875. It was better
than the compact rendered transition on each unit (0.00, 0.03125, and 0.046875), but it did not turn
that prediction advantage into planning benefit. The strongest action control scored 0.00, 0.109375,
and 0.09375 on one-step cell accuracy. Therefore neither primary arm beat both the prediction control
and the strongest non-oracle planning control. The fixture null is supported and the scientific
verdict remains `not-eligible`.

Every equal-core arm has 1,360 trainable parameters including its same-shape coordinate decoder. Per
unit it performs 6,144 predictor calls, 2,048 decoder calls, and 7,667,712 counted linear-layer MACs
during planning. These exact equalities make the negative interpretable as a mechanics result: it is
not explained by a missing arm or a shallower control. They do not make the tiny dataset externally
valid.

The current registered inherited action-conditioned control is intentionally not run. This preflight
forbids model weights and downloads, and there is no exact-referent inherited control package whose
training and observation contract would make that comparison fair. Substituting an arbitrary model
or random initialization would manufacture a result. Scientific promotion therefore remains blocked
until an independently sourced action-conditioned trajectory set, exact-referent control, held-out
interventions, and predeclared replication exist.

## What this changes in the project

P7 demotes “rendered action-world-model comparison software” from an external-compute blocker to a
local, executable mechanics surface. Future custom-substrate candidates can plug a rendered latent or
object state into the same exact branches, budgets, horizons, planning executor, calibration metrics,
and cost ledger. The inherited platform is not the substrate here: it is absent. The durable platform
seam is the project’s own event identity, render contract, action interface, transition API, replay
verification, and falsification boundary.

P7 does not demote external validity. A larger machine cannot turn this grid into natural evidence,
and a perfect fixture score would not close that gap. What the scaffold does is make future compute
earned: additional scale is justified only after a candidate passes these local causal, control,
replay, calibration, and cost mechanics.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/p7_action_world_model_preflight.py
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_action_world_model.py -q
```

The runner is bounded to one CPU thread, three units, 45 seconds, one GiB maximum RSS, no accelerator,
no downloads, and no model weights. It writes only the durable proof receipt and refuses to promote a
scientific claim.
