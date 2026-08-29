# substrate-sandbox

Policy and validation types for Substrate's sandbox **body plan**.

This crate is the future native policy core for Substrate's capability sandbox.
It models grants, portable mounts, network mode, resource budgets, and tool
roles in a closed, typed API. Every accepted dry-run plan keeps
`execution_permitted == false`.

## What this is

- A **policy contract** (`SandboxBodyPlan`) and fail-closed validators
- Explicit capability grants (`Capability`)
- Portable mount sources only (no host-path mounts in the public API)
- Tool **roles** with required capabilities, not executable names or paths
- Standard library only: no third-party crates, no `unsafe`, no FFI

## What this is not

- Not a security sandbox, seccomp profile, or container runtime
- Not a process supervisor, launcher, or resource enforcer
- Does not spawn processes, open sockets, traverse filesystems, download
  sources, automate browsers, decode media, or install tools
- Does not vendor or assert licenses for upstream adapters (Chromium, FFmpeg,
  Git, yt-dlp, and similar remain operator-provided, license-reviewed choices
  selected by signed packs elsewhere)

A later broker may enforce these plans when launching a sandbox. This package
only defines and checks the plan.

## Schema

`SCHEMA_VERSION` is exactly `substrate-sandbox-body-v1`.

## Quick start

```rust
use substrate_sandbox::{SandboxBodyPlan, ValidationError};

let plan = SandboxBodyPlan::dry_run_default();
assert_eq!(plan.execution_permitted, false);
plan.validate_dry_run().expect("default plan is valid");

// Execution remains forbidden under dry-run validation:
let mut bad = plan.clone();
bad.execution_permitted = true;
assert!(matches!(
    bad.validate_dry_run(),
    Err(ValidationError::ExecutionPermitted)
));
```

## Portable mounts

Only three mount sources exist, each with a fixed access mode:

| Source | Access |
|--------|--------|
| `content-addressed-inputs` | read-only |
| `task-workspace` | ephemeral-write |
| `untrusted-output` | quarantine-write |

There is no host-path source in the typed API.

## Dry-run validation

`validate_dry_run` / `validate` reject (fail closed):

- schema mismatch
- `execution_permitted == true`
- any network mode other than deny / no-network
- missing, duplicate, or wrong-access mounts
- invalid (non-positive) resource limits
- tool roles missing required capability grants
- tool roles whose required capabilities are unapproved under dry-run
- browser or media tool use without the explicit capability

## License

MIT OR Apache-2.0
