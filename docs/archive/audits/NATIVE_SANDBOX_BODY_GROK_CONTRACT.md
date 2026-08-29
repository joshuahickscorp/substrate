# Native sandbox body-policy package contract

## Objective

Create a self-contained Rust crate at `native/substrate-sandbox/` that is the
future native policy core for Substrate's capability sandbox.  It must model
the body contract used by the Python product runtime but must not launch,
supervise, download, inspect, or install anything.  This is a policy and
validation package only.

## Context

Substrate is a portable developmental runtime.  Its product layer currently
has a Python reference implementation and a deliberately non-executing
sandbox plan (`execution_permitted: false`).  A later broker may use Rust for
process supervision, resource accounting, I/O, and sandbox launch, but those
operations are out of scope here.  The crate must be useful without requiring
the current untracked Python modules to be present in this isolated worktree.

Upstream tools such as Chromium, FFmpeg, Git, and yt-dlp are not to be copied,
vendored, disguised, downloaded, or executed.  They will eventually be
operator-provided, license-reviewed adapters selected by signed packs.  This
crate only defines the grants that a future adapter must receive.

## Permitted modifications

Create or edit only:

- `native/substrate-sandbox/Cargo.toml`
- `native/substrate-sandbox/README.md`
- `native/substrate-sandbox/src/lib.rs`
- `native/substrate-sandbox/tests/**` if integration tests are useful

Do not modify any existing Python, documentation, CI, dependency, Git, or
Odyssey files.  Do not create files outside `native/substrate-sandbox/`.

## Required public contract

Use Rust stable and the standard library only: no third-party crate
dependencies, no build script, no unsafe code, no FFI.

Expose a small well-documented API, including:

1. `SCHEMA_VERSION` exactly `"substrate-sandbox-body-v1"`.
2. A closed `Capability` enum covering exactly these explicit grants:
   `Network`, `FilesystemWrite`, `Subprocess`, `Browser`, `MediaDecode`,
   `ContainerNesting`, `Gpu`, `Microphone`, `Camera`, and `Desktop`.
3. A closed mount/source model that permits only the portable non-host sources
   `content-addressed-inputs`, `task-workspace`, and `untrusted-output`, with
   their required access modes respectively read-only, ephemeral-write, and
   quarantine-write.  It must be impossible to construct an accepted host-path
   mount through the public typed API.
4. `ResourceBudget` with strictly positive CPU, memory MiB, disk MiB, process,
   thread, wall-clock-second, and output-byte limits.
5. `NetworkMode` where the dry-run validation accepts only a deny/no-network
   mode.
6. `ToolAdapterKind` or an equivalent closed tool-role enumeration that can
   express at least repository inspection, browser observation, and media
   observation.  It must state the required capabilities but not an executable
   name or path.
7. `SandboxBodyPlan` (or an equivalently named type) that contains schema
   version, `execution_permitted`, network mode, mounts, grants, resource
   budget, and tool roles.
8. A fail-closed `validate_dry_run`/`validate` method returning a typed error:
   it must reject execution permission, egress, wrong/missing/duplicate mount
   sources, wrong mount access, missing tool-role capability grants, a tool
   role that implies unapproved capability, invalid resource limits, a schema
   mismatch, and browser/media use without their explicit capability.
9. A constructor for a safe default dry-run plan that has all three portable
   mounts, no network, no capabilities, no tools, and execution forbidden.

## Non-goals and prohibitions

- No process spawning (`std::process`, command invocation, or shelling out).
- No sockets, HTTP, filesystem traversal, child supervision, container API,
  browser automation, media decode, Git operation, credential handling, or
  source downloading.
- No claims that the crate is a security sandbox yet.
- No execution authorization: every valid plan must retain
  `execution_permitted == false`.
- Do not add a license assertion for upstream software or imply that the
  future adapters are vendored.

## Tests and evidence

Add unit/integration tests proving the valid default plan and every stated
refusal.  Run and report exactly:

```bash
cargo fmt --check --manifest-path native/substrate-sandbox/Cargo.toml
cargo test --manifest-path native/substrate-sandbox/Cargo.toml
cargo clippy --manifest-path native/substrate-sandbox/Cargo.toml -- -D warnings
```

Do not commit, push, merge, publish, delete worktrees, or modify remote state.
Return a concise report with changed files, test output, and any limitation.
