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
  summary files. Nine finished campaigns ship with their evidence and review packages.

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

## The latest result is an architectural null

The most recent campaign, Cognitive Material Genesis II, ended at Outcome B:
`cognitive_material_genesis_ii_complete`, with status `compositional_advantage_unproven`. It repaired
the earlier material's associative write granularity and substantially improved learning, but the
preregistered simplicity rule selected an associative monolith rather than a field.

Against exact S2, the selected monolith's principal paired effect was `0.393415`, 95% CI
`[0.372210, 0.414509]`; replication was `0.415282`, and hidden composition was `0.316777`. The
decisive campaign covered 4,245,640 episodes. But the strongest field trailed the strongest equally
plastic monolith by `-0.004167` in the representation/architecture factorial, so the experiment did
not establish a field, compositional, or low-precision architectural advantage.

Four of ten primary claims passed. All 17 deliberately planted defects were detected, none survived,
and a clean clone exactly recomputed the principal, replication, and hidden-composition results. The
selected non-continuous-time material also passed 250,000 events, 16 process interruptions, 32 scheduled
checkpoints, four migrations, and four model and body replacements.

The parent Cognitive Material Genesis result remains authoritative: its selected field lost to S2 by
`-0.247768`, 95% CI `[-0.256737, -0.238393]`. Genesis II explains part of that failure as representation
and update-granularity cost; it does not erase the negative result. Earlier Nous Closure and v5 results
also remain preserved under their original tags.

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

- [Cognitive Material Genesis II terminal report](docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md)
  and [tangible-sandbox handoff](docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_HANDOFF.md)
- [Genesis II sealed classification](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_FINAL_CLASSIFICATION.json)
  and [limitations](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_LIMITATIONS.json)
- [Latest campaign report](artifacts/substrate/nous_closure/SUBSTRATE_NOUS_CLOSURE_TERMINAL_REPORT.md)
  and its [limitations](artifacts/substrate/nous_closure/external_review/LIMITATIONS.md)
- [Review package for outside readers](artifacts/substrate/nous_closure/external_review/README.md)
- [v5 report](artifacts/substrate/v5/SUBSTRATE_V5_TERMINAL_REPORT.md),
  [v5 scientific status](docs/V5_SCIENTIFIC_STATUS.md), [v5 architecture](docs/V5_ARCHITECTURE.md)

## What's next

See [ROADMAP.md](ROADMAP.md).
