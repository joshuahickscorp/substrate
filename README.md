# Substrate

Substrate is an offline Python program that runs repeatable experiments on a simulated software entity.
The entity keeps memory across tasks, acts inside two fake environments (a desktop-style sandbox and a
seeded 3D simulator), and every result it produces is written to a hash-sealed JSON file that can be
re-checked later.

It exists to test one question: does building the entity out of many cooperating parts with persistent
memory beat a much simpler program given the same information and budget? So far the measured answer is no.

## What it does

- Runs seeded simulations. Same seed, same result, every time. Nothing is downloaded and nothing touches
  the network.
- Feeds the entity synthetic sensor events of eight kinds: text, image, video, motion, audio, speech,
  depth/3D, and body/tool.
- Stores the entity's identity, goals, memories, world state and body state in an append-only log where
  each entry is hashed against the previous one. A checkpoint refuses to load on a broken seal, on time
  running backwards, or if anything tried to switch the entity on for real.
- Routes work to 13 small hand-written Python modules standing in for models (an object detector, a
  speech grounder, a depth estimator, a planner, and so on). Nobody trained these. They are fixtures.
- Re-checks its own claims by recomputing results from the sealed receipts rather than trusting the
  summary files. Six finished campaigns ship with their evidence and review packages.

## Install

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"   # or: make install
```

## Use

```bash
substrate status                            # what has been run
substrate test                              # 396 tests (about 4 minutes)
substrate verify                            # audit, then recompute results from sealed receipts
substrate audit                             # structural audit only
substrate nous-closure status               # state of the most recent campaign
substrate nous-closure verify --no-publish  # re-check that campaign's 68 sealed authorities
```

`make test`, `make accept` (= `substrate verify`) and `make audit` wrap these. Three things to know.
Commands in the `v2`/`v3`/`v4`/`v5` families write into the tracked `evidence/` and `artifacts/` trees,
so they dirty your working copy; the `nous-closure` family takes `--no-publish` and does not.
`substrate v5 verify` does not work from a fresh clone, because it needs a ~1.1 GiB raw run tree that is
not committed — see [docs/V5_RUNBOOK.md](docs/V5_RUNBOOK.md) to rebuild it. And do not run
`substrate run`: that is the deliberate boundary that starts a real campaign.

## The latest result was a null

The most recent campaign, Nous Closure, ended at `terminal_closed_null` and scored 10.0 out of 20 on its
own checklist. Two tests were run.

1. A stateless direct policy scored `0.95125` over 32 unseen histories; the full v5 entity scored
   `0.93796875`. A perfect answer would have been only `0.04875` better than the simple policy — less
   than the `0.05` the project had committed in advance to treat as meaningful, so the test had no room
   to show anything.
2. On a sandbox of 12 task families with real files and media, the modular entity and an independently
   written single-file state machine both scored `1.00000000`. The paired difference was `0.00000000`,
   95% CI `[0.00000000, 0.00000000]`. A tie.

Persistent memory does do something: a fresh stateless control was clearly worse. What was not shown is
the bigger claim, that the elaborate modular design beats a simple state machine given equal information,
tools and budget. 25 of 25 deliberately planted defects were caught, none survived, and a clean-clone
check passed. After the null the program stopped instead of looking for a friendlier test, so the 12-hour,
principal, replication and open-world runs were never launched.

The earlier v5 campaign did meet its own targets: 5,760 work units, 576,000 events, 15 effects recomputed
from raw receipts and all clearing `0.05` in all three splits, 21 of 21 planted defects caught. Those
numbers describe the sandbox, not the world.

## What this is not

- Not a claim about consciousness, phenomenal experience, sentience, feeling, suffering, desire,
  personhood, life, or moral status. None of that is claimed, and none of it follows from any result here.
- Not a trained model. The 13 modules are hand-written deterministic Python fixtures with no training
  data, not downloaded pretrained checkpoints. Zero models, zero corpora and zero bytes were ever fetched.
- Not evidence of general real-world ability. Every task is bounded, offline, and inside a simulator.
- Not externally reviewed. The reviewers that graded the closure package are internal simulations, which
  the package states itself (`external_independence_claimed: false`).
- Not switched on. Activation is `false` throughout and CI asserts it stays false. The entity never acts
  outside its sandboxes.

"Nous" is this project's own name for the property being tested. Naming it is not claiming it.

## Evidence

- [Latest campaign report](artifacts/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_TERMINAL_REPORT.md)
  and its [limitations](artifacts/substrate/nous_closure/external_review/LIMITATIONS.md)
- [Review package for outside readers](artifacts/substrate/nous_closure/external_review/README.md)
- [v5 report](artifacts/substrate/v5/SUBSTRATE_V5_TERMINAL_REPORT.md),
  [v5 scientific status](docs/V5_SCIENTIFIC_STATUS.md), [v5 architecture](docs/V5_ARCHITECTURE.md)

## What's next

See [ROADMAP.md](ROADMAP.md).
