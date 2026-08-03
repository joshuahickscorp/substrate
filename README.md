# Substrate

Python research program with one question: does an entity made of many cooperating parts with persistent memory beat a
much simpler program on the same information and the same budget? So far the measured answer is no.

## How it works

- Campaigns are preregistered. Thresholds, seeds, controls and claims are frozen and hashed before any result exists,
  and every arm runs at equal budget against controls up to and including an oracle.
- Entity state (identity, goals, memories, world, body) lives in an append-only hash-chained log. Checkpoints refuse to
  load on a broken seal, on backwards time, or on any attempt to switch the entity on for real. Input is synthetic
  sensor events of eight kinds: text, image, video, motion, audio, speech, depth/3D, body/tool.
- `verify` recomputes results from the sealed receipts rather than the summary files. Older campaigns are offline seeded
  simulations; Tangible Sandbox R2 adds 235 tasks over pinned public benchmarks and a frozen local qwen3:8b.

## Run

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"   # or: make install
substrate test             # test suite
substrate verify           # audit, then recompute from sealed receipts
substrate sandbox status   # Tangible Sandbox R2 terminal state
```

`substrate run` starts a real campaign, so leave it alone. `substrate v5 verify` needs an uncommitted ~1.1 GiB raw run
tree, so it fails from a fresh clone ([docs/V5_RUNBOOK.md](docs/V5_RUNBOOK.md)).

## Results

[Genesis II](docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md), seeded simulation, 4,245,640 episodes, 128
histories, BCa bootstrap: the selected material beat the S2 monolithic baseline by 0.393415, 95% CI [0.372210, 0.414509].
But the preregistered simplicity rule picked an associative monolith, not a field, and the strongest field trailed the
equally plastic monolith by 0.004167, so no compositional advantage was shown. Four of ten primary claims passed, and
the parent Genesis field still loses to S2: effect -0.247768, 95% CI [-0.256737, -0.238393].

[Tangible Sandbox R2](evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_TERMINAL_REPORT.md) ended at Outcome C,
`terminal_tangible_sandbox_null`. At 64 histories with 1024 model calls and 2048 tool calls per arm, the entity trailed
a plain project-state database by 0.5 on the custom STSC-1 corpus, and matched the controls on the public benchmark
floor (effect 0.0). Nothing terminal is claimed either way: the fresh 24-hour continuity lane never ran, because the
campaign filesystem sat below its protected disk floor with no writable alternate volume.

## Limits

- No claim about consciousness, sentience, feeling, personhood or moral status: none of it is claimed and none of it
  follows. "Nous" is this project's name for the property under test, not a claim to have it.
- Nothing here is a trained model. Modules are hand-specified deterministic fixtures with no training data, and R2
  borrows a frozen local model only as a replaceable organ.
- Every task sits in a frozen benchmark or local sandbox with no deployment authority, so real-world ability is not
  evidenced. Activation is `false` throughout, asserted by CI.
- The closure package's reviewers are internal simulations, not outside readers (`external_independence_claimed: false`).

## What's next

The missing terminal gate is a fresh 24-hour continuity trace, blocked on disk space, so nothing downstream runs. Next up
is outside review, a clean-clone `substrate v5 verify`, and lint, type and docs cleanup: see [ROADMAP.md](ROADMAP.md).
