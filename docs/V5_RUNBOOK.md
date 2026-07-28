# Substrate v5 ready-stage runbook

All commands run from the authoritative repository:

```bash
cd /Users/scammermike/Downloads/substrate
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

Use `.venv/bin/substrate` when the virtual environment is not activated.

## Inspect without launching

```bash
.venv/bin/substrate v5 status
git rev-parse HEAD
git tag --list 'substrate-v5*'
jq '{all_pass,total,passed,failed,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_CHEAP_CANARIES.json
jq '{passed,independent_histories,focused_arm_count,episodes,modality_count,model_equivalent_count,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_MODERATE_PILOT.json
jq '{admitted,principal_launch_authorized,gates,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_ADMISSION.json
```

`status` distinguishes acquisition, preprocessing, model preparation, kernel
comparison, sensorium construction, canaries, pilot, principal, replication,
open-world review, independent verification, and terminal publication.

## One v5 command family

```bash
substrate v5 preflight
substrate v5 acquire
substrate v5 inventory
substrate v5 build
substrate v5 canaries
substrate v5 pilot
substrate v5 rehearse
substrate v5 run
substrate v5 status
substrate v5 stop
substrate v5 resume
substrate v5 verify
```

- `preflight` verifies prior-version immutability, repository identity,
  resources, and `activation=false`.
- `inventory` reports local capabilities without admitting them.
- `acquire` publishes acquisition authorities. The ready-stage result performs
  no network download, downloads zero bytes, and admits no external object.
- `build` freezes configuration, publishes construction authorities, and
  materializes the 5,760-unit principal manifest.
- `canaries` executes and seals all 50 deterministic canaries.
- `pilot` runs the 16-history, 14-arm, 89,600-episode moderate pilot, failure
  rehearsal, resource measurement, local transfer benchmark, kernel selection,
  and principal admission.
- `rehearse` reruns and publishes the 15 contained failure injections.
- `run` executes principal, replication, and open-world splits only after the
  automatic principal gate passes.
- `stop` creates `runs/substrate/v5/state/stop`.
- `resume` removes that switch, rechecks the principal gate, and executes the
  deterministic DAG.
- `verify` consumes raw receipts, independently recomputes effects, injects
  mutations, performs clean-clone checks, and assigns only the classification
  supported by terminal evidence.

## Reproduce the ready-stage evidence

These commands write new named authorities and immutable content-addressed
copies. Preserve existing evidence before reproducing measured fields such as
elapsed time or resource use.

```bash
.venv/bin/substrate v5 preflight
.venv/bin/substrate v5 inventory
.venv/bin/substrate v5 acquire
.venv/bin/substrate v5 build
.venv/bin/substrate v5 canaries
.venv/bin/substrate v5 pilot
.venv/bin/substrate v5 rehearse
.venv/bin/substrate v5 status
```

Verify the implementation separately:

```bash
.venv/bin/python -m pytest -q tests/substrate
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy
```

Do not use `substrate v5 verify` as a ready-stage smoke test: terminal
verification requires complete principal, replication, and open-world receipts
plus the ready tag for clean-clone reproduction.

## Automatic principal gate

Both `run` and `resume` fail closed unless all five checks are true:

1. `substrate-v5-sensorium-ready` resolves to a commit;
2. `SUBSTRATE_V5_ADMISSION.json` exists;
3. `principal_launch_authorized` is exactly `true`;
4. admission records `activation` exactly `false`; and
5. the admission source digest matches the ready source.

Inspect the gate:

```bash
.venv/bin/substrate v5 status | jq '.principal_gate'
```

The ready-stage checkout has green admission, but the principal gate is not yet
authorized because the ready tag and matching ready-source freeze are pending.
Once the ready publication is complete, the campaign command is:

```bash
.venv/bin/substrate v5 run
```

Do not create or move the ready tag merely to bypass the gate. Do not change
source, models, corpora, splits, seeds, thresholds, controls, or scientific
premises after principal launch.

## Stop and recovery

The stop switch is checked when `run` or `resume` enters the principal runner. It
is not a process signal and does not claim to terminate an already-running
worker. If a foreground campaign is active, stop it through the shell that owns
that process, then preserve all emitted receipts and logs.

Use this recovery sequence:

1. Run `substrate v5 status` and record the failing stage, source commit, source
   digest, unit identity, receipt, checkpoint, and error output.
2. Run `substrate v5 stop` before any new launch. Do not delete or overwrite a
   failed receipt, checkpoint, content-addressed object, or failure record.
3. Classify the event as operational failure, implementation defect, instrument
   defect, scientific null, no headroom, or unavailable dependency.
4. For a corrupt permanent-entity checkpoint, let restore fail closed. Recover
   from a previously verified sealed checkpoint or exact deterministic replay;
   never hand-edit state or hashes.
5. For a missing or stale model, preserve entity state and checkpoint identity.
   Do not silently substitute a different model. No external checkpoint is
   currently admitted.
6. For a missing corpus or cache, regenerate only from the frozen local
   generator identity. No external corpus is currently admitted.
7. Repair only demonstrated software or instrument defects, add a regression
   test, and use the sealed transition procedure if principal has launched.
   Scientific nulls and no-headroom results are not tuning invitations.
8. Re-run tests, lint, affected canaries or rehearsal, and admission. Confirm the
   ready-source digest again before `substrate v5 resume`.

The contained pilot rehearsal detected and recovered all 15 declared scenarios:
model crash, sensor loss, process restart, stale checkpoint, corrupt time,
corrupt frame, partial 3D state, failed download, partial extraction, worker and
supervisor death, duplicate publication, disk and memory pressure, and
stop/resume. It signaled or modified no pre-existing process.

`activation` must remain `false` throughout recovery. Recovery never authorizes
external action, threshold changes, data substitution, or a stronger scientific
claim.
