# The custom sandbox body

The sandbox is Substrate's controlled sensory and work boundary: the place a
future specialist can receive approved evidence, carry out bounded work, and
return results for verification. It is not an unrestricted desktop, browser,
terminal, or downloader.

The implementation here is intentionally staged. The current product code can
plan and validate the body, but it cannot launch it. Every current plan keeps
`execution_permitted: false`.

## Current native package

`src/native/substrate-sandbox` is a Rust crate named `substrate-sandbox`. Its
`substrate-sandbox-body-v1` contract represents:

- a closed capability vocabulary;
- three fixed portable mount sources and their fixed access modes;
- network mode, resource limits, and typed tool roles;
- fail-closed validation of dry-run plans, including capability and mount
  mismatches.

It has no third-party crate dependencies and explicitly does not spawn
processes, access the network, manage filesystems, automate a browser, decode
media, fetch sources, install software, or launch a container. A valid Rust
body plan is therefore useful as a stable broker/guest protocol, not as a
claim of operational isolation.

## Required task shape

When an executable backend is introduced, it should enforce this fixed data
shape:

```text
trusted host
  ├─ /inputs  : immutable, content-addressed, read-only
  ├─ /work    : task-scoped and ephemeral
  └─ /output  : untrusted, quarantine-only
```

The public plan intentionally has no host-path mount source. A worker must not
inherit the user’s home directory, credentials, SSH agent, Git identity,
browser profile, clipboard, Docker socket, local network reachability, or
ambient environment. It should receive an explicit resource grant and no
network access unless a later broker attaches a source-specific, bounded
egress grant.

## Eyes and ears, safely

Substrate can eventually have useful multimodal inputs without making every
tool omnipotent:

- A repository adapter can present an approved immutable repository snapshot
  and return a quarantined patch or test receipt.
- A media adapter can turn approved, rights-cleared source bytes into explicit
  derived artifacts such as a transcript, frame sample, waveform, or metadata
  record.
- A browser adapter can observe an approved source through a fresh task-only
  browser context, inside the outer worker boundary.
- A simulation adapter can present an approved scene or solver result while
  keeping actuators, LAN access, and host devices outside the task grant.

Each observation must retain its source/policy receipt, input object digest,
tool or image digest, transformation recipe identifier, output digest, and
verification result. The authoritative entity can then decide whether the
evidence changes durable competence or an active project.

## Performance direction

The performance work should concentrate on the narrow trusted path: streaming
hashes, cache addressing, validated launch specifications, bounded IPC,
resource accounting, and deterministic receipts. Rust is a good fit for that
path. It does not make sense to fork or mechanically rewrite mature upstream
projects just to make them “native to Substrate.”

The accompanying Python tool-bundle contract records the exact
operator-provided upstream artifact that a future Rust broker would be allowed
to bind: a digest-pinned OCI image or binary archive with SBOM, notices,
license, and verification-receipt digests. Its adapter roles describe fixed
operations such as media probing, frame sampling, browser observation, or
repository inspection. It carries no command line, executable path, image tag,
or network permission.

Large work should be parallelized as independent disposable workers. The
entity remains a single ordered developmental timeline: workers return
evidence, and one controlled assimilation writer applies any state change.
More host RAM or CPU can increase throughput under the same wall-clock budget;
it does not relax provenance, source policy, or sandbox limits.

## Preconditions for execution

Before any body plan can become executable, the product needs a reviewed
backend that enforces all of the following in practice:

1. VM/container isolation appropriate to the platform, with resource and
   process limits.
2. No-network default plus explicit, validated brokered egress controls.
3. Exact read-only inputs, ephemeral work, and quarantine-only output paths.
4. Typed argument-vector adapters rather than model-provided shell commands.
5. Pinned tool/image provenance, SBOM and license review, health checks, and
   revocation/update handling.
6. Output hashing, inspection, signed locally trusted attestation, promotion,
   and audit receipts.
7. Fault-injection, escape-resistance, and policy-denial tests.

Until those conditions are implemented and independently verified, local
backend discovery and the Rust body plan remain dry-run tools only.
