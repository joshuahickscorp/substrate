# Substrate v5 terminal runbook

All commands run from the authoritative repository:

```bash
cd /Users/scammermike/Downloads/substrate
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

Use `.venv/bin/substrate` when the virtual environment is not activated.

## Inspect terminal state

```bash
.venv/bin/substrate v5 status
git rev-parse HEAD
git tag --list 'substrate-v5*'

jq '{expected_units,published_units,all_terminal,sensory_events_or_cognitive_episodes,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_PRINCIPAL_AUTHORITY.json
jq '{all_pass,all_primary_hypotheses_pass,replication_pass,open_world_pass,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_INDEPENDENT_VERIFICATION.json
jq '{total,detected,survived,zero_survived,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_MUTATION_REPORT.json
jq '{all_pass,exact_reproduction,normalized_double_regeneration_exact,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_CLEAN_CLONE.json
jq '{classification,gates,unqualified_nous,activation}' \
  evidence/substrate/v5/SUBSTRATE_V5_FINAL_CLASSIFICATION.json
```

The terminal state is 5,760/5,760 units: 3,456 principal, 1,152 replication,
and 1,152 generator-held-out open-world units. Independent verification passed,
all 21 mutations were detected, clean-clone reproduction passed, and activation
is `false`.

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
- `acquire` publishes acquisition authorities. The terminal campaign performed
  no network download, downloaded zero bytes, and admitted no external object.
- `build` freezes configuration, publishes construction authorities, and
  materializes the 5,760-unit manifest.
- `canaries` executes and seals all 50 deterministic canaries.
- `pilot` runs the 16-history, 14-arm, 89,600-episode moderate pilot, resource
  measurement, local transfer benchmark, kernel selection, and admission.
- `rehearse` executes the 15 contained failure injections.
- `run` executes principal, replication, and open-world splits after the frozen
  principal gate passes.
- `stop` creates `runs/substrate/v5/state/stop`.
- `resume` removes that switch, rechecks the gate, and continues the
  deterministic DAG without republishing terminal units.
- `verify` consumes raw receipts, independently regenerates work units,
  recomputes effects, injects mutations, performs clean-clone checks, builds the
  review package, and assigns only the supported classification.

## Reproduce from the frozen ready source

The authoritative implementation is frozen by:

```text
tag: substrate-v5-sensorium-ready
commit: 9988a70e418998fcab7b3bb869fba06dd273c811
```

Use an isolated clone or worktree. Verification intentionally resolves the
immutable ready source for raw regeneration and clean-clone expectations:

```bash
git clone <repository-url> substrate-v5-review
cd substrate-v5-review
git checkout substrate-v5-terminal

uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

.venv/bin/substrate v5 status
```

The full verifier requires the raw run tree. Validate
`artifacts/substrate/v5/review/RAW_RECEIPT_INDEX.json`, reconstruct the raw
receipt layout from `RAW_RECEIPTS.jsonl.gz` in an isolated review workspace,
then run `.venv/bin/substrate v5 verify`. The published package contains the
compressed raw receipts, indexes, effect ledgers, controls, identity/learning
evidence, mutations, and reproduction result without committing the
approximately 1.1 GiB operational run directory.

Review the compact evidence without rerunning the campaign:

```bash
gzip -cd artifacts/substrate/v5/review/RAW_RECEIPTS.jsonl.gz | head
jq '.' artifacts/substrate/v5/review/AUTHORITY_INDEX.json
jq '.' artifacts/substrate/v5/review/EFFECT_LEDGER.json
jq '.' artifacts/substrate/v5/review/KNOWN_LIMITATIONS.json
```

## Verify the repository

```bash
.venv/bin/python -m pytest -q tests/substrate
.venv/bin/ruff check src tests
.venv/bin/ruff format --check \
  src/substrate/v5verify.py tests/substrate/test_v5_verify.py
.venv/bin/python - <<'PY'
from substrate import audit

report = audit.run()
assert report["all_pass"], report
print(report["all_pass"])
PY
```

The terminal clean-clone authority records the exact command results and
normalized double-regeneration digest. The independent verification authority
contains the principal, replication, and open-world effects.

## Principal gate

`run` and `resume` fail closed unless the ready tag, admission authority, source
identity, frozen authority bindings, worktree policy, and `activation=false`
checks agree. Inspect the current gate with:

```bash
.venv/bin/substrate v5 status | jq '.principal_gate'
```

After terminal publication the current checkout is intentionally newer than the
ready commit, so the gate reports that HEAD differs from the ready source. This
does not invalidate completed receipts. Reproduction and verification use the
ready tag recorded by the sealed authorities; do not move that tag or alter
source identity to make a post-publication gate green.

## Stop, recovery, and resume

The stop switch is checked when `run` or `resume` enters the principal runner.
It is not a process signal and does not claim to terminate an already-running
worker. Preserve every emitted receipt, checkpoint, object, and failure record.

For recovery:

1. Run `substrate v5 status` and record the stage, ready source, unit identity,
   receipt, checkpoint, and error.
2. Run `substrate v5 stop` before a new launch.
3. Classify the event as operational failure, implementation defect, instrument
   defect, scientific null, no headroom, or unavailable dependency.
4. Recover corrupt state only from a verified sealed checkpoint or exact
   deterministic replay; never hand-edit state or hashes.
5. Do not silently substitute a model, corpus, checkpoint, seed, threshold, or
   control.
6. Repair only demonstrated software or instrument defects, add a regression
   test, and use a sealed transition if the principal campaign has launched.
7. Re-run tests, lint, the affected canary/rehearsal, admission, and source
   identity checks before `substrate v5 resume`.

The failure rehearsal detected, recovered, and contained all 15 declared
scenarios without signaling or modifying pre-existing processes.

## Sealed post-launch transition

After principal launch, scientific inputs and premises are frozen. A legitimate
verifier-only defect repair must:

1. preserve the ready commit and ready tag;
2. leave source models, corpora, splits, seeds, thresholds, controls, principal
   receipts, and claim boundaries unchanged;
3. identify the exact implementation or instrument defect;
4. add a regression test that fails before the repair;
5. publish a numbered `SUBSTRATE_V5_TRANSITION_NNN.json` authority;
6. pass repository tests, lint, formatting, activation checks, and authoritative
   CI;
7. merge a dedicated PR and create a matching immutable annotated transition
   tag; and
8. carry the ready-source identity into all regenerated expectations.

Transitions 001-003 followed this process. They repaired verifier command
expansion, ready-source regeneration binding, and mutation-fixture source
identity respectively. They did not change the principal science or outcomes.

## Claim and action boundary

The exact classification `multimodal_nous_ready_for_review` means eligible for
external review only. It is never an unqualified Nous declaration. No command
licenses uncontrolled external action. The model-equivalents and environments
are deterministic bounded scientific fixtures, and activation must remain
`false` throughout inspection, recovery, and reproduction.
