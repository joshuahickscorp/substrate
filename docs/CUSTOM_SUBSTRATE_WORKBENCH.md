# Custom substrate workbench

## What changed

The project now has a locally trainable, teacher-independent video substrate rather than only an
inherited-encoder attachment point. The implementation is intentionally small, 1 to 5 million
trainable parameters, and intended to answer engineering questions cheaply on the M3 Pro. It is not a
claim of a brain, sentience, general intelligence, or natural-video capability.

The inherited encoder has one remaining role in this lane: an optional frozen measurement teacher
supplied through a citable cache. No inherited model code or weights are imported by the mandatory
training arms. Removing or swapping that cache does not prevent CM7 from running.

The machine-readable design source is
`configs/custom_substrate/requirements.yaml`. Every run hashes that ledger and every evidence file it
names. The ledger converts the F campaign and `PROJECT_EXPERIMENT_EXHAUSTION` findings into concrete
requirements:

| Requirement | Campaign input | Workbench consequence |
|---|---|---|
| immutable referents | F1, F5, F14 | content-addressed split and checkpoint identities; explicit teacher joins |
| off-ceiling factor structure | F2, F3, F9 | dense video tokens plus held-out-combination factor probes |
| cross-form transfer | F1, F4 | train-only ridge alignment to deterministic oracle-form tokens |
| predictive/intervention readiness | F6, F18 | masked latent prediction and retained spatiotemporal tokens |
| disciplined plasticity/replay | F7, F8, F11, F14, F16 | EMA targets, exact resume, immutable replay export contract |
| density and cost | F13 | parameter envelope, matched FLOPs, RSS/time/checkpoint accounting |
| failure awareness | F17, F20, project exhaustion | scope labels and fail-closed CM8 promotion |
| teacher independence | real local encoder receipt | cache-only optional distillation, never a platform dependency |

This is the concrete bridge from the inherited-encoder stepping stone to a later independent
substrate. The bridge carries requirements, controls, targets, and measurements forward, not the old
architecture.

## Architecture

`TinyVideoSubstrate` uses a 3D tubelet/patch embedding, learned spatiotemporal positions, a small
Transformer encoder, dense token output, a pooled retrieval key, and one identical predictor head in
every training arm. The default 256px, eight-frame, 128-wide, four-layer model has about 1.7 million
trainable parameters. It can later accept action conditioning, replay, memory, or multi-form alignment
without changing referent identity.

CM7 compares four arms that share the exact same initial state, architecture, batches, update count,
resolution, and estimated core FLOPs:

1. predictive: masked token prediction against an EMA target on the same view;
2. invariance: token prediction across deterministic nuisance views of one referent;
3. reconstruction: masked prediction of fixed low-order patch statistics;
4. random target: a compute-matched negative control using permuted target tokens.

An exact frozen copy of the same initialization is evaluated as a separate architecture control. An
optional `teacher_distill` arm can join frozen-teacher targets by exact referent id. It refuses to run
if the cache is invalid or a batch referent has no target.

The programmatic corpus uses hue and orientation/drift as independent factors, with phase, texture,
gain, and pixel noise as nuisance. Entire factor combinations, not individual replicas, are held out.
The train, validation, and test referents are therefore immutable and combination-disjoint. A
programmatic oracle must solve the same held-out split before an encoder result is interpreted.

## Receipts and resume

Every run writes:

- `resolved_config.json` and its content hash;
- `requirements_audit.json` with current hashes of every source finding;
- `dataset_manifest.json` with generator hash, factor rows, referents, splits, and split hashes;
- `teacher_audit.json`, even when no teacher is configured;
- one checkpoint and arm receipt per seed/objective;
- `workbench_receipt.json` with evaluations, controls, compute matching, resource telemetry, and
  promotion refusals.

A checkpoint contains the online model, EMA target, optimizer, step, losses, RNG state, and the
config/data/requirements/initialization hashes. Resume refuses any mismatch. A completed arm is reused
only when its checkpoint hash is unchanged and it already reached the requested step count.

## Local execution

The harness profile is a bounded one-seed 256px execution. The long local profile uses five seeds,
1,000 updates per arm, eight replicas per factor combination, and a 180-minute wall budget:

```bash
.venv/bin/python scripts/custom_substrate_workbench.py cm7 --profile local180 --device mps
```

The wall budget is per invocation. If it is reached, the command exits resumable; running the same
command again continues only the unfinished arm from its last content-addressed checkpoint. It never
silently changes resolution, seed count, or model size to finish sooner.

The shorter calibration profile is:

```bash
.venv/bin/python scripts/custom_substrate_workbench.py cm7 --profile local30 --device mps
```

CM8 is now locally inspectable without pretending its upstream evidence exists:

```bash
.venv/bin/python scripts/custom_substrate_workbench.py cm8-preflight
```

That command performs no training and does not load teacher weights. It audits the CM7 receipt, the
same-referent teacher cache, and upstream gates. The current eight-row programmatic teacher is useful
for wiring, but is intentionally below the 64-row minimum and is not natural video, so CM8 scientific
promotion remains refused.

## Promotion boundary

A CM7 local result can say only that objective choice is or is not a live lever on this deterministic
programmatic task. Promotion requires five complete seeds, clean requirements, a 1 to 5M model,
off-ceiling oracle calibration, matched trained-arm FLOPs, and a conservative margin over both the
random-target and frozen-initialization controls.

CM8 additionally requires a sufficiently large exact same-referent citable teacher cache,
rights-cleared natural-video evidence, and the CM1, CM2, and DR1 upstream gates. Programmatic teacher
data can validate mechanics and inform the next architecture, but it cannot satisfy those scientific
requirements.
