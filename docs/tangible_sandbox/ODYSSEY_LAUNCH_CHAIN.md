# Odyssey launch chain

Dependency-ordered plan from the current 5/15 gate state to a detached 7-day run.
Fail-closed at every step: a gate that does not genuinely pass halts the chain.

## Phase 0 — in flight

| lane | produces | blocks |
|---|---|---|
| `odyssey-rehearsal` (Grok) | `odyssey_rehearsal.py` — G06/G07/G08/G09 producers | phase 3 |
| `odyssey-custody` (Grok) | G02/G04/G05/G10/G11 converted to machine-verified | phase 3 |
| `odyssey-corpus` (Grok, gate) | corpora for E/F + enrichment, rights records, hash manifests | G03, and G07 |
| corpus-audit (agent) | `ODYSSEY_SOURCE_SELECTION.sealed.json` | G03 |

Already landed: optional/derived custodian seed in
`odyssey_manifest_materializer.py`, so G03 needs no human secret.

## Phase 1 — integrate

1. Review both Grok diffs. Rehearsal lane gets adversarial review: its gates are a
   *structural* oracle, so a harness emitting plausible constants would turn four
   gates green while measuring nothing. Verify each metric traces to a syscall,
   child process, or real file.
2. Merge `grok/odyssey-rehearsal-*` and `grok/odyssey-custody-*`.
3. Re-run the corpus audit's selection seal against the enlarged corpus.
4. `ruff` + full test suite.
5. Commit.

## Phase 2 — refreeze

Refreezing changes `frozen_build_sha256`. **Every existing gate receipt dies with
it** — `_require_frozen_subject_binding` demands the subject name the current
frozen build *and* current git HEAD. So G01/G12/G13/G14/G15 all re-run after this
point, not before.

1. Regenerate the frozen build and rendered index.
2. Re-run the public model canary (G02/G05 bind to its receipt).

## Phase 3 — gates, in dependency order

```
G15 protocol digests        (needs refreeze)
G01 R2 transition
G13 clean clone + CI        (needs committed HEAD)
G12 mutation suite
G14 telegram probe
G03 frontier manifests      (needs sealed source selection + corpus)
G05 model/tool panel        (needs canary receipt)
G02 arm selection           (needs canary receipt + G05)
G11 statistics authority
G10 isolation probes        (real cross-uid denial, sudo -u nobody)
G08 memory broker canary    (cheap, pure-function)
G06 width-8 admission       (63 paired cell runs, serialized ollama — long pole)
G07 storage rehearsal       (MUST run after corpus download completes)
G09 durability + recovery   (16 real kill/restart/restore cycles)
G04 custody commitments     (needs G03 manifests)
```

**Ordering constraint discovered, not optional:** G07 measures real device free
space and real private growth. A concurrent 100 GiB corpus download would corrupt
that measurement into a false failure — or worse, a false pass on a machine that
is actually full. Corpus acquisition must be finished and settled before G07 runs.

**G06 may honestly fail.** If this machine cannot admit width 8 without swap, the
frozen design already answers it: `reduced_width_policy` says a narrower run takes
a *different diagnostic identifier and makes no full Odyssey claim*. That is a
pre-frozen decision, not a new one, and the chain follows it rather than retrying
until the number looks better.

## Phase 4 — seal and launch

1. `preflight` over all 15 gates.
2. `seal` — only if every gate genuinely passes.
3. Launch detached under the OS supervisor:

```
PYTHONPATH=src .venv/bin/python -m substrate.odyssey7d supervise \
  --root . --authority plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json
```

4. Confirm the supervisor is detached and shell-independent, Telegram reporting on
   the 30-minute cadence, and the 75/80/82/85 GiB broker thresholds live.

## What still stops the chain

- Any gate that fails on real measurement. The chain reports and halts; it does
  not adjust a threshold, retry until green, or relax a validator.
- A frontier left with no licensed source material after acquisition.
- Free space that cannot hold the 120 GiB private-write cap over the 25 GiB floor.

## Custody, recorded

Human custody was removed by operator decision: every scored task is verifier-
scored, so no withheld answer key needs a human holder. The property lost is real
and is written into the sealed authority — early reveal is now *detectable* via
pre-published commitments, not *impossible*. No claim of independent custody or
double-blind reveal survives this change, and nothing downstream may assert one.
