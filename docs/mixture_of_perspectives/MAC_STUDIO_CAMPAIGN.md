# Mac Studio campaign supervisor

`scripts/mop_campaign.py` is the durable control plane for the local P5-to-substrate sequence. It
observes and adopts an already-running matching governor leg without signaling it, then schedules
only named tasks through `scripts/local_execution_throttle.py`. The initial DAG is:

`p5fresh_challenge_cpu` → `p5verify_cpu` → controlled substrate-baseline transition.

The supervisor has a nonblocking lifetime `flock`, atomic self-hashed state/status, immutable hourly
snapshots, bounded retry/backoff and resumable-leg accounting. A stop request is a drain: no new work
starts, and an adopted/owned governor remains untouched until it exits. Policy or governor source
drift fails closed everywhere except the final transition node.

Validate or inspect without launching anything:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_campaign.py validate \
  --plan configs/campaign/mac_studio_local.json
PYTHONPATH=src:. .venv/bin/python scripts/mop_campaign.py run --once \
  --plan configs/campaign/mac_studio_local.json
```

Start the detached execution campaign (new session plus `caffeinate -ims` when available):

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_campaign.py start --execute \
  --plan configs/campaign/mac_studio_local.json
```

Use `status`, `wait`, or `stop` with the same `--plan`. `stop` writes a sealed drain request; it does
not send a process signal.

After both P5 artifacts have valid governor-provenance joins and all governor lanes are empty, the
baseline transition can be made explicit:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_campaign.py mark-ready \
  --plan configs/campaign/mac_studio_local.json \
  --reason "reviewed substrate baseline transition"
```

The immutable marker self-hashes and binds the campaign/plan plus expected old and new policy and
governor implementation hashes. A policy-only transition is adopted at that DAG node. If governor
code changed, the old process exits in `migration_restart_required`; run `start --execute` once more
so the freshly loaded process can verify and adopt the exact authorized hash. Any unmarked or
mismatched change remains `policy_drift_hold`.

For a staged change, create the marker before replacing either live file by also passing
`--expected-new-policy-sha256` and/or `--expected-new-governor-sha256`. The marker waits until the
on-disk authorities exactly equal those hashes; it never authorizes a wildcard or nearest match.

## Post-transition long run

`configs/campaign/mac_studio_substrate_phase1.json` is the next detached campaign, prepared but
deliberately unloadable under the live P5 policy. The controlled transition must first install the
reviewed task overlay and the compatible governor implementation. That refusal is the boundary that
prevents an in-flight P5 receipt from being reinterpreted under a newer authority.

After the transition is sealed and the new policy validates, the phase-one campaign executes one
dependency-ordered chain without operator polling:

```text
EDCM-1 producer -> independent regeneration
  -> X0 producer -> disjoint fresh verification
  -> P6 10k resource probe -> 10k replication -> verifier
  -> P6 100k replication -> verifier
  -> P6 1m replication -> verifier
```

Every node is joined to its declared output and a sealed governor completion receipt. The P6 scale
steps require a favorable, independently rebuilt predecessor verdict; a null therefore stops the
ladder instead of consuming more compute. EDCM and X0 terminal nulls remain valid results and do not
silently become positive substrate premises.

Once the migrated policy is current, validate and start it with:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_campaign.py validate \
  --plan configs/campaign/mac_studio_substrate_phase1.json
PYTHONPATH=src:. .venv/bin/python scripts/mop_campaign.py start --execute \
  --plan configs/campaign/mac_studio_substrate_phase1.json
```

The supervisor still emits immutable hourly snapshots and otherwise remains detached. Heavy work is
exclusive because wall time, memory, and lifecycle work are scientific endpoints; safe internal CPU
or MPS acceleration remains available to each admitted task. Light substrate construction can proceed
concurrently only when it cannot mutate the active task, policy, receipt, or checkpoint namespace.
