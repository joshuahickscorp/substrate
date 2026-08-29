# Tangible Sandbox next-launch runbook

This control plane is separate from the running R2 source identity. It makes a
blinded 24-hour shadow the default next scientific unit; it does not edit R2 or
turn an incomplete R2 trace into evidence.

## Handoff sequence

```bash
python -m substrate.tangible_next status
python -m substrate.tangible_next review-r2
# Review and complete the three draft JSON files under plans/substrate/tangible_next_launch/
python -m substrate.tangible_next seal-design
python -m substrate.tangible_next run-calibration
python -m substrate.tangible_next preflight
python -m substrate.tangible_next prepare
# Custodian materializes task/answer manifests, then:
python -m substrate.tangible_next seal-custody --handoff RUN/CUSTODY_HANDOFF.json \
  --task-manifest RUN/builder_visible/TASK_MANIFEST.json \
  --answer-manifest RUN/evaluator_only/ANSWER_MANIFEST.json \
  --seed-file /outside/repository/custodian-seed
python -m substrate.tangible_next launch --handoff RUN/CUSTODY_HANDOFF.json
```

`review-r2` deliberately rejects the historic `not_run` evidence and any live
or incomplete longitudinal state. `seal-design` rejects placeholders, missing
custody commitment, unfrozen data cards, or an unapproved causal change.
`seal-custody` verifies that task identities match the sealed 24-hour schedule,
the seed matches its sealed commitment, and answer material stays evaluator
only. `launch` creates a one-shot launchd job; its worker locks the complete
candidate/control trace before it invokes the evaluator.

## Adapter contract

The final short scaffolding session binds three versioned commands in the
sealed design: candidate, matched control, and independent evaluator. Each
must accept a JSON request path and write exactly one JSON receipt path. A
receipt must contain: `task_id`, `run_id`, `input_manifest_sha256`,
`output_artifacts`, `elapsed_seconds`, `resource_usage`, and
`activation:false`. The evaluator command is not released the answer mapping
until the candidate trace digest has been sealed by the custodian.

Use `python -m substrate.tangible_next validate-receipt REQUEST RECEIPT` to
check the contract before the command is admitted. The contract itself is
sealed in `plans/substrate/tangible_next_launch/ADAPTER_CONTRACT.sealed.json`.

The generic control plane intentionally does not invent these scientific
adapters. Their exact behavior depends on the R2 result and the newly selected
stimulus bank; binding a placeholder would look launch-ready while invalidating
the blind comparison.

## What may run in parallel

Only the synthetic resource-calibration capsules and independent preparation
or verification work may run concurrently after their admission check. The
24-hour continuity timeline remains one dedicated writer. Every parallel
capsule receives its own root and may not share writable evaluator or data
state.
