# Odyssey action and training boundary

External corpora and live input play different roles.

- **External corpus:** supplies fixed observations and task material. MOVi and
  CLEVR provide visual/physical observations; MATH provides problems; SWE-bench
  provides issue/patch metadata. They can support a separately predeclared
  training intervention, but must not leak evaluator-only answers.
- **Deterministic simulator:** supplies action-conditioned feedback. For the
  embodied frontier, precommitted actions enter a frozen Blender/Kubric scene;
  the simulator returns observation and state receipts. This is the valid
  proprioception analogue because the feedback depends on the selected action.
- **Uncontrolled live input:** is not part of the blinded Odyssey. It would
  make stimuli non-replayable, compromise parity, and prevent a clean
  candidate/control comparison.

The implementation sequence is therefore: install/freeze the simulation
runtime after R2; measure a no-score episode; commit seeds, action policy
family, reset policy, output schema, and disjoint evaluator continuations;
then generate the capped task bank. Passive MOVi trajectories are used for
state-prediction and calibration, while simulator episodes test active action
and recovery.
