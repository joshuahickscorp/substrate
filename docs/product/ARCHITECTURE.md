# Post-Odyssey product architecture

Status: foundation only. This document describes a product track that is
separate from the Tangible Sandbox and Odyssey experiments. It does not change
their launch gates, activate a worker, or make a capability claim.

Substrate is being shaped as a portable developmental runtime: an entity has
durable identity, specialty, state, receipts, and unfinished work; compatible
model organs and task tools are replaceable components around that state. It
is not a claim that the entity is conscious, sentient, autonomous in the open
world, or generally capable.

```text
portable entity
  identity · developmental state · receipts · active plans
                         |
trusted product control plane
  policy · scheduling · pack trust · provenance · assimilation
                         |
signed capability declarations       local artifact cache
  no tools installed                  quarantine → verify → promote
                         |                         |
                future sandbox broker / task worker
  fixed mounts · limits · typed tool roles · explicit egress grants
                         |
            approved external tools and model organs
```

Only the entity and control-plane records are portable. Model weights,
credentials, browser profiles, source bytes, executable images, and tool
binaries stay outside the entity directory. A task worker, when one exists,
will receive selected immutable inputs and return untrusted output; it will
not write developmental state directly. A single authoritative assimilation
path decides what becomes durable state.

## Implemented foundation

The current Python product modules provide deterministic state, policy, and
evidence handling:

- `contracts.py`, `entity.py`, and `apprenticeship.py` define portable entity
  state, receipts, bounded apprenticeship plans, a typed cache-attested
  source-evidence bridge, and a single-writer update discipline.
- `packs.py` declares built-in capability requirements and emits
  non-executing sandbox plans.
- `pack_artifacts.py` builds, signs, verifies, trusts, and locally registers
  manifest-only capability packs. A valid signature is still insufficient
  without an explicit, locally scoped trust rule.
- `cache.py` provides a local content-addressed quarantine boundary for
  operator-supplied regular files. A signed, expiry-bound attestation from a
  locally trusted verifier, digest/length/type revalidation, durable
  transition recovery, and provenance are required before promotion.
- `sources.py` records constrained source-acquisition plans and receipts;
  `source_adapters.py` adds cache-digest-bound, non-executing observation
  plans for local files, repository snapshots, media, and browser captures;
  `backends.py` only probes backend eligibility and emits dry-run bindings.
- `tool_bundles.py` defines immutable, non-executing manifests for
  operator-provided upstream tool artifacts. They pin an OCI image or binary
  archive digest alongside SBOM, notices, license, and verification material,
  but do not download, install, or launch a tool.

`src/native/substrate-sandbox` is a small Rust policy package. It defines the
closed, typed `substrate-sandbox-body-v1` dry-run plan contract for mounts,
resource limits, capability grants, and tool roles. It is deliberately not a
process launcher, container runtime, browser controller, downloader, or media
processor.

## Capability and data lifecycle

1. An operator chooses a declared pack and builds a manifest-only artifact.
2. The artifact is signed with an operator-managed Ed25519 key, then checked
   against local publisher, pack-name, capability, runtime, and host scope.
3. Local installation records a verified manifest reference only. It does not
   download, copy, install, or execute a tool.
4. A separately supplied local file enters cache quarantine. Its descriptor
   records a SHA-256 digest, byte length, declared media type, safe source
   reference digest, retrieval time, rights status, and any derived lineage.
5. A signed attestation bound to the exact cache identity and quarantined
   descriptor, plus repeat validation, can promote the immutable object to
   `verified` (raw input) or `processed` (derived output). The cache writes
   its own verification receipt; an input JSON file alone is never a receipt.
6. A typed local cache-attestation authority may bind one approved source
   object to an active entity plan and record sanitized evidence; it neither
   launches a worker nor grants source/tool authority.
7. A future broker may materialize only approved immutable inputs for a
   disposable worker, then must quarantine its outputs again.

At every implemented step, `execution_permitted` remains `false`. The plans
are contracts for future enforcement, not enforcement itself.

## Trust boundaries

The future runtime is intentionally layered rather than a giant repository of
copied tools:

| Boundary | Owns | Must not receive |
| --- | --- | --- |
| Portable entity | identity, development, receipts, pack/organ requirements | credentials, raw media, tool binaries |
| Trusted host/control plane | policies, trust decisions, cache promotion, eventual scheduling | untrusted worker output as durable state |
| Disposable worker | immutable inputs and one ephemeral work area | host paths, Docker socket, Git identity, browser profile, secrets |
| Brokered source/tool adapter | a narrow approved operation | arbitrary shell text, open-ended network, inherited user session |

A browser context does not itself create an outer security boundary, and a
manifest does not become a sandbox merely because it names a browser or a
media utility. The outer isolation, resource controls, safe I/O paths, and
promotion process must be independently implemented and reviewed.

## Product direction

Rust is the preferred native implementation language for a narrow
security-sensitive broker, portable guest protocol, streaming validation, and
high-throughput artifact handling. It is not a reason to rewrite Chromium,
FFmpeg, yt-dlp, Blender, Git, or other upstream projects. Those remain
operator-provided or image-provided dependencies with their own version,
license, SBOM, provenance, and security review.

The next executable increment is not broad web scraping. It is a reviewed,
bounded task backend that enforces the existing plan: no network by default,
portable fixed mounts, validated argument vectors, resource limits, typed
operations, output quarantine, and auditable receipts. Source-specific
browser, repository, and media adapters can follow only through that broker.

The detailed product contracts are kept beside this overview: [entity
format](ENTITY_FORMAT.md), [capability packs](CAPABILITY_PACKS.md),
[assimilation](ASSIMILATION.md), [source adapters](SOURCE_ADAPTERS.md),
[sandbox body](SANDBOX.md), [security](SECURITY.md), and
[portability](PORTABILITY.md).
