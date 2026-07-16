# Generation 1 Successor Recovery v5

> **Canonical recovery note:** This document supersedes the active launch guidance in
> [21_generation1_successor_evidence_chain.md](./21_generation1_successor_evidence_chain.md) and
> preserves the categorized program in
> [23_generation1_categorized_batch_wave.md](./23_generation1_categorized_batch_wave.md).
> The v4 state, status, and adoption receipts remain immutable historical evidence.

- **Status:** validated append-only recovery scaffold
- **Snapshot date:** 2026-07-16
- **Adopter:** `generation1-successor-evidence-chain-v5`
- **Post-adopter waiter:** `generation1-successor-extension-chain-v2`
- **Categorized waiter:** `generation1-categorized-batch-extension-chain-v1`
- **Categorized child:** `generation1-successor-categorized-batch-wave-v1`
- **One-command launch:**
  `.venv/bin/python scripts/mop_generation1_successor_long_chain_v2.py start --execute`
- **Claim scope:** generated same-code robustness, mechanics classification, redesign
  preregistration, and structural artifact verification only; no runtime activation, scientific
  promotion, natural-world generality, or independent scientific-generator claim

## 1. Result interpretation

The frozen D1 producer/challenge campaign is complete across all 576 registered rungs and
1,658,880 evaluated cells. The exact frozen centroid design fails its static-margin and
context-route-gap criteria in both phases. Its honest route is `null_safe_prune`:

- preserve the complete aggregate and byte-bound structural verification;
- retire the exact centroid design as an efficacy candidate;
- retain it as a comparison control;
- spend no additional efficacy seeds on that retired premise;
- require any future D1 efficacy work to enter through a new append-only redesign authority.

This is a clean, bounded, nonconfirmatory result. It is not an independently generated scientific
null and does not authorize weaker thresholds or activation.

The successor-mechanics and consolidated-final campaigns remain independent live predecessors.
Their partial receipts are operational progress, not terminal evidence. V5 observes them without
signals and waits for their exact terminal artifacts.

## 2. Why v5 exists

The immutable v4 adopter entered `integrity_hold` after reporting an unexpected process-group
member in the consolidated-final campaign. The underlying campaign remained healthy.

The failure was reproduced at the process boundary:

1. `ProcessPoolExecutor` creates a worker with the exact Python `multiprocessing.spawn_main`
   command.
2. The worker calls `setproctitle` before its durable `mop-final-*` label is fully visible.
3. For roughly half a second, macOS process metadata exposes a joined spawn command in `argv[0]`
   while later argv slots contain transitional overwritten data.
4. V4 recognizes the original four-argument spawn command and the final worker label, but not that
   exact intermediate representation, so one valid worker can cause a permanent hold.

V5 never rewrites or clears the v4 hold. It uses a fresh root, schemas, process label, adoption
receipts, implementation authority, locks, and detached entrypoint.

## 3. Bounded process-title stabilization

V5 retains v4's exact parent, cwd, process-group, restart-command, worker-label, resource-tracker,
and spawn-worker rules. It adds one provisional classification for the reproduced transition only.

A provisional child must satisfy every condition:

- direct child of the exact adopted parent;
- member of that parent's exact process group;
- exact repository cwd;
- finite positive PID creation identity;
- executable independently resolved to the repository virtualenv Python;
- exactly four argv slots;
- full `argv[0]` match to the joined canonical `spawn_main` command and
  `--multiprocessing-fork` flag.

When that exact shape appears, V5 re-samples the complete process table every 50 milliseconds for
at most 41 attempts. The parent identity must remain unchanged. Each provisional child must become
an already allowed spawn worker, become an exact labelled worker, or disappear with an exact
`gone` identity probe. Two consecutive clean classifications are required.

Any unrelated group member, wrong executable, wrong cwd, wrong PPID or PGID, changed PID identity,
arbitrary resolved command, inaccessible disappearance, parent replacement, or persistent
transition fails closed. No grace-path action sends a signal, launches a process, or mutates a
legacy queue.

Raw transitional argv is never written to state, status, logs, or adoption receipts because
setproctitle can temporarily expose overwritten environment-shaped strings. Durable receipts record
only the exact adopted parent identity and the bounded policy name.

## 4. Append-only one-command topology

```text
completed frozen D1
        +
live successor mechanics
        +
live consolidated final campaign
        |
        v
generation1-successor-evidence-chain-v5
  - independently validates completed D1
  - adopts mechanics/final observation-only
  - waits for exact terminal results
        |
        v
generation1-successor-horizon-v1
        |
        v
generation1-successor-extension-chain-v2
        |
        v
generation1-successor-horizon-v2
        |
        v
generation1-categorized-batch-extension-chain-v1
        |
        v
generation1-successor-categorized-batch-wave-v1
```

`generation1-successor-future-chain-v2` starts or acknowledges v5 before starting extension v2.
`generation1-successor-long-chain-v2` then starts or acknowledges the existing categorized waiter.
Each durable component has its own lock and exact visible-parent acknowledgement. Repeating the
whole-chain command resumes the same roots and cannot duplicate a legacy queue, horizon
supervisor, categorized supervisor, or waiter.

The historical roots remain evidence only:

- `generation1-successor-evidence-chain-v4`: immutable `integrity_hold`;
- `generation1-successor-extension-chain-v1`: immutable failure propagated from v4;
- `generation1-successor-long-chain-v1`: historical start-lock root.

No v2 launcher treats those unsafe terminal states as success. It starts fresh versioned parents
that independently re-establish current authority.

## 5. Categorized serial and parallel work

The final categorized program remains one readable serial wave with safe parallel work inside each
category:

- seven ordered waves;
- six scientific categories per wave;
- up to eight internal workers in one admitted category capsule;
- a serial classify-and-seal barrier after every wave;
- physically last, dependency-gated G1-I1 integration;
- aggregate, independent structural verifier, and report receipt.

The 59 top-level capsules cover formation and trace, communication and repair, memory and
plasticity, action and simulation, construction search, dispatch redesign, and final integration.
The program can schedule up to 15,886 checkpointed fresh mechanics items.

Its maximum categorized mechanics ceiling is approximately 196.18 serial hours or 24.52 ideal
hours at eight workers. Including the two preceding successor horizons raises the append-only
mechanics ceiling to approximately 481.04 serial hours or 60.13 ideal eight-worker hours before
registered pruning. These are compute ceilings, not promises to consume time after a valid null or
failed mechanism route.

## 6. Launch and monitoring boundary

The complete append-only chain starts with:

```bash
.venv/bin/python scripts/mop_generation1_successor_long_chain_v2.py start --execute
```

The command may immediately start lightweight observation parents while the two incumbent heavy
campaigns continue. It cannot start horizon-v1 until every legacy prerequisite is terminal and
valid. Extension v2 cannot start horizon-v2 until v5 completes and replays. The categorized waiter
cannot launch categorized compute until horizon-v2 completes and the host admission gate is clean.

Notifications remain advisory. They can tell an operator that a result is ready for interpretation,
but they do not count as receipts, alter a frozen route, or authorize a later capsule.
