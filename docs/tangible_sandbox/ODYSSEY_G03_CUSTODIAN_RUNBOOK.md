# Odyssey G03 source selection and manifest materialization

This is a custody-facing preparation runbook, not a launch command. It never
selects a corpus, model, candidate, control, task, answer, or seed on its own.
It produces no G03 gate receipt by itself and cannot start the scientific
worker.

## Preconditions

Use this only after the current Odyssey frozen build and R2-to-Odyssey
transition receipt validate, and only after the responsible humans have made
the source, rights, arm, and custody decisions that G02/G04/G05/G10 require.
The materializer is a mechanical source-binding step; it is not a substitute
for any of those human attestations.

Before filling the template, the custodian must have all of the following for
each ordered frontier A through H:

- one or more locally present, candidate-visible source assets;
- the SHA-256 of each exact asset;
- a local, reviewable rights record for each asset;
- a distinct candidate-visible role for each asset; and
- confirmation that no candidate-visible asset contains evaluator answers,
  scorer material, hidden assignments, or result-dependent selection data.

The eight source rows must remain ordered A–H. Do not add evaluator-only
answers or the custodian seed to this file. Keep the seed as a non-empty file
outside the repository, accessible only to the custodian account.

## Deterministic handoff

1. Copy and complete
   `plans/substrate/tangible_next_launch/ODYSSEY_SOURCE_SELECTION.template.json`
   into a new, root-relative draft path. Replace every placeholder, retain
   `activation:false` and `external_activation:false`, and set
   `status` to `ready_for_custodian_seal`. The draft is still only a human
   proposal; it is not a gate receipt.

2. On the custodian account, source-bind the completed draft without changing
   any selection:

   ```bash
   PYTHONPATH=src ./.venv/bin/python -m substrate.odyssey_manifest_materializer \
     --root /Users/scammermike/Downloads/substrate \
     --seal-source-selection \
     --draft operations/odyssey/custody/ODYSSEY_SOURCE_SELECTION.ready.json \
     --out operations/odyssey/custody/ODYSSEY_SOURCE_SELECTION.sealed.json
   ```

   The command is write-once. It verifies source bytes and rights-reference
   paths, writes a canonical self-digested selection, and does not inspect or
   create a seed, evaluator answer, candidate/control adapter, or authority.

3. Materialize the source-bound candidate/evaluator pairs on the separate
   custodian account. The candidate and evaluator roots must be disjoint. For
   a real G10 boundary, the evaluator root must additionally be protected by
   a real separate UID or mounted/ACL-isolated filesystem; directory names or
   modes alone are not sufficient.

   ```bash
   PYTHONPATH=src ./.venv/bin/python -m substrate.odyssey_manifest_materializer \
     --root /Users/scammermike/Downloads/substrate \
     --selection operations/odyssey/custody/ODYSSEY_SOURCE_SELECTION.sealed.json \
     --seed-file /outside/repository/odyssey-custodian-seed \
     --candidate-root run/odyssey/candidate-visible \
     --evaluator-root run/odyssey/evaluator-only \
     --out evidence/substrate/odyssey/manifests/ODYSSEY_FRONTIER_MANIFEST_SET.json
   ```

4. Independently inspect the resulting eight candidate manifests and preserve
   the evaluator-only counterparts outside candidate access. Only then may the
   resulting manifest-set subject be supplied to the G03 machine-gate sealer.
   A valid G03 receipt still does not make G04, G06–G11, or launch true.

If a source, rights record, frozen build, or seed-access boundary changes,
stop. Create a new reviewed selection and materialize a new immutable manifest
set; never overwrite a prior receipt or revise it in place.
