# Product foundation after Odyssey

This document describes a new, deliberately bounded product foundation. It is
not part of the Tangible Sandbox R2 campaign or the Odyssey experiment, and it
does not change any experiment, claim, launch gate, or activation boundary.

It also does not claim consciousness, sentience, agency in the open world, or
AGI. The initial implementation is a portable state format and capability
control plane. It plans and verifies work; it does not execute a browser,
download media, install packages, run a model, or create a worker process.

## Product boundary

Odyssey evaluates whether a controlled experimental subject can preserve and
use multimodal stimuli over time. The product foundation asks a separate
engineering question: how could a specialist keep durable developmental state
while using approved models and tools in a controlled world?

```text
portable entity state
  identity · specialty · developmental state · receipt ledger
                         |
                 capability broker
  policy · provenance · scheduler · artifact promotion
                         |
      future disposable task sandbox / worker
  approved tool image · read-only inputs · ephemeral work
                         |
           quarantine → hash → verify → assimilate
```

Workers can gather evidence in parallel. They are not represented as one mind
and they never write entity state. One authoritative assimilation writer
validates receipts and records what becomes durable developmental state.

## What is implemented now

`substrate product` is independent from all existing campaign commands:

```bash
substrate product packs
substrate product init ./engineer.substrate \
  --entity-id go-systems-engineer \
  --specialty "Go distributed systems" \
  --pack engineering --pack research --organ primary-reasoning
substrate product plan-apprenticeship ./engineer.substrate \
  --name go-systems-foundation \
  --objective "Build and verify bounded Go systems exercises" \
  --evaluator hidden-go-suite \
  --host-cpu-cores 20 --host-memory-mib 65536 --host-disk-mib 262144 \
  --worker-cpu-cores 2 --worker-memory-mib 4096 --worker-disk-mib 8192 \
  --maximum-workers 8 --wall-clock-minutes 480 \
  --source-file-root /approved-corpus
substrate product plan-source ./engineer.substrate \
  --source-uri file:///approved-corpus/go-concurrency.txt \
  --modality text --access-status user-provided \
  --declared-rights "operator-provided material" --retrieval-mode import
substrate product backends
substrate product dry-run-backend ./engineer.substrate
substrate product validate ./engineer.substrate
```

`backends` and `dry-run-backend` remain non-executing. They discover local
backend candidates from platform facts and executable path lookup only
(`container`, `docker`). They never launch a VM or container, contact a daemon,
open a socket, install software, or mutate the host. Discovering a binary path
does not prove daemon readiness, image availability, safe configuration, or
permission to run.

Selection priority is eligible Apple Container (`container` on Darwin/Apple
Silicon), otherwise Docker as a compatibility fallback when `docker` is on
PATH, otherwise an explicit unavailable result. `plan_sandbox` stays portable
and host-agnostic (`backend: unconfigured`). The dry-run layer separately binds
to its verified digest, repeats the no-network/default non-host-mount posture,
declares the resource budget, and always sets `execution_permitted: false`.
Selection is a recommendation, never authorization.

An entity directory is portable and intentionally small:

```text
engineer.substrate/
  entity.json                 identity, specialty, pack and replaceable organ requirements
  developmental-state.json    current phase, competence, unfinished work
  receipts.jsonl              logically append-only, hash-linked developmental receipts
  checkpoint.json             state and ledger-tail integrity binding
  .writer.lock                single-writer coordination after the first update
```

Creation additionally uses a sibling, parent-directory coordination lock so
two local creators cannot publish different entities at the same target. That
lock is not portable entity content.

Model weights, downloaded media, source documents, browser profiles, tool
binaries, and executable images are not stored in the entity directory. The
format has no credentials field, and source URIs with credentials, query
strings, or fragments are refused before a plan is persisted; operators must
also never put secrets in its free-form metadata. The product foundation now
includes a separate content-addressed cache. It admits local regular files into
quarantine and needs a signed, locally trusted verifier attestation before an
object can enter a verified or processed zone; future workers will receive
only those admitted immutable inputs.

An organ requirement is a stable protocol-and-modality declaration, not a
checkpoint or provider commitment. The same entity can later use a compatible
local model, remote model gateway, formal solver, or multimodal organ without
putting its weights or credentials in the portable entity bundle.

The hash chain detects ordinary alteration and accidental corruption. It does
not by itself prove authenticity against someone who can rewrite every local
file and recompute every digest. Production portability needs an external
signature, transparency log, or operator-held root of trust.

Entity updates take an advisory writer lock, stage the next state, ledger, and
checkpoint as one verified pending transaction, then publish the three files.
Operations derived from a verified snapshot also use its checkpoint digest as a
compare-and-commit revision guard, so an assimilation cannot land after a plan
or policy has been replaced. A caller that loses this race must reload and
re-evaluate its work.
If an interruption leaves a pending transaction, normal loading refuses it;
an operator must run `substrate product recover <entity-directory>` to apply
the verified staged transaction explicitly. The first format rewrites the
small JSON-lines ledger for atomic publication, so it favors correctness over
high-volume ingestion. A production cache needs segmented receipt logs and
periodic signed checkpoints. The scaffold fsyncs each replacement and its
parent directory on supported POSIX filesystems; production still needs a
fault-injection and power-loss durability campaign before making stronger
crash-recovery claims.

## Capability packs are declarations, not installers

The initial product-facing packs are `engineering`, `formal-math`, `research`,
`media`, `3d`, `browser`, `desktop`, `data-science`, and `robotics`.
`mathematics` and `three-d` remain legacy aliases. Each declares:

- tool requirements rather than installed binaries;
- allowed media modalities and source-policy requirements;
- per-worker CPU, memory, and disk profile;
- default network posture, currently `none`;
- filesystem and isolation expectations.

For example, `media` names FFmpeg and yt-dlp only as optional image or host
requirements. It never invokes either. A video request is a provenance-bearing
plan that needs an explicit allowlisted source policy and rights/access basis.
It is not a generic "scrape YouTube" feature. YouTube's terms restrict
automated access and downloading unless it is permitted by the service, rights
holder, or applicable law; the product should default to approved APIs,
operator-provided media, captions, and licensed datasets instead.
[YouTube Terms of Service](https://www.youtube.com/t/terms)

When media execution exists, pin the whole worker image, its tool versions,
SBOM, licenses, and entrypoints. FFmpeg licensing depends on its exact build
configuration, so record it with the transform receipt rather than assuming a
single license posture. [FFmpeg legal guidance](https://ffmpeg.org/legal.html)

At the broker/cache boundary, source receipts retain the source URI, declared
rights/access status, retrieval method and time, content SHA-256, and
processing history. Portable entity plans and assimilation receipts retain
only safe origin labels and digests: no raw URI/path, rights text, or detailed
processing history enters the entity ledger. Raw content and derivatives should
use OCI-style content descriptors and immutable digests. [OCI image descriptor specification](https://github.com/opencontainers/image-spec/blob/main/descriptor.md)
The `plan_acquisition` API creates the corresponding hashable, non-executing
source-adapter plan before any future broker is allowed to acquire content.

Source policy is intentionally strict in this first version: a local `file:`
request must be an absolute path under an operator-declared non-root directory;
a remote request must use an approved scheme and exact approved authority, with
no explicit port. File authorities, URL credentials, query strings, and
fragments are rejected. This means a future YouTube adapter should pass a
sanitized source reference or opaque approved ID to the broker instead of
persisting a user-session URL.

Planning a source records a `source_acquisition_planned` receipt. A later
source receipt must name that plan's digest, match its policy and request, and
pass a typed `CacheAttestedEvidenceAuthority` before it can be assimilated.
That authority delegates to the cache's signed, locally trusted attestation
check for the exact source-plan digest and content object; a caller-supplied
callback is refused. Syntax and a claimed digest are not proof that bytes were
fetched, rights were checked, or a content store contains them.

## The actual sandbox comes later

A capability manifest is not a security boundary. The future backend should be
a brokered capability plane:

- The trusted host owns entity state, policies, source grants, and the content
  store.
- A worker receives read-only, digest-verified inputs and one disposable
  output area. It gets no Docker socket, host mount, SSH agent, browser
  profile, cookies, credentials, or inherited user environment.
- Network is off by default. Approved retrieval uses a brokered allowlist with
  domain, method, byte, rate, and time limits, while refusing localhost, LAN,
  metadata, and rebinding destinations.
- Worker output is quarantined, hashed, inspected, and explicitly promoted by
  the trusted host. Workers do not directly mutate reusable artifacts or
  developmental state.
- A broker builds validated argument vectors from declared operations. Model
  output is never passed directly to an unrestricted shell.

On this Apple Silicon macOS workstation, do not use the deprecated
`sandbox-exec`/SBPL interface as the product boundary. A future proof of
concept should prefer Apple's OCI `container` runtime, which starts each
container in a lightweight Linux VM. Docker Desktop can be a compatibility
adapter, but shared paths require the same careful treatment as host mounts.
Apple notes that guest memory may not return to macOS until a container is
restarted, so media and browser workers should be short-lived and recycled.
[Apple Container technical overview](https://github.com/apple/container/blob/main/docs/technical-overview.md)
and [command reference](https://github.com/apple/container/blob/main/docs/command-reference.md)

On Linux, rootless containers are useful defense in depth but are not enough
alone: the backend needs cgroups v2, dropped capabilities, seccomp, and an
LSM where available. [Docker rootless mode](https://docs.docker.com/engine/security/rootless/),
[cgroups v2](https://docs.kernel.org/admin-guide/cgroup-v2.html), and
[seccomp filters](https://docs.kernel.org/userspace-api/seccomp_filter.html)
describe those controls. Browser contexts isolate web session state cheaply,
but Chromium itself must still run inside the task VM; a fresh context is not
the outer sandbox. [Chromium sandbox design](https://chromium.googlesource.com/chromium/src/+/HEAD/docs/design/sandbox.md)

## Performance and host-specific scheduling

The planner takes an explicit host resource snapshot and a per-worker vector:

```text
workers = min(
  caller maximum,
  (host CPU - control-plane reserve - assimilation reserve) / worker CPU,
  (host RAM - control-plane reserve - assimilation reserve) / worker RAM,
  (host disk - control-plane reserve - assimilation reserve) / worker disk,
)
```

The result is a hard upper bound after a default 1-core / 1-GiB / 1-GiB broker
and assimilation reserve; it is not a promise that every remaining core should
be used. A selected pack also refuses a worker budget smaller than its declared
minimum profile.
On this 28-core / 96 GiB Mac, an initial global ceiling around 20 cores and
64 GiB leaves room for the desktop, local model inference, and the broker.
Use separate queues for light retrieval, compilation, browser work, media
transcode, simulation, and formal verification; their real resource profiles
will differ. A later scheduler should admit CPU, RAM, PID, disk, I/O, wall
time, egress bytes, and risk tier together rather than treating worker count as
the only limit.

## Language and packaging decision

The current scaffold remains Python because its job is typed contracts,
deterministic storage, policy, and planning in the existing repository. Do not
add a second language merely to claim performance.

When a native broker or high-throughput artifact/media path is actually needed,
Rust is the better general second language than Go: it is well suited to a
small security-sensitive supervisor, explicit resource/FD handling, safe
concurrency, streaming digests, and a portable guest-agent protocol. On macOS,
a thin Swift bridge may also be useful if the backend directly adopts Apple's
Containerization or Virtualization APIs. Neither Rust nor Go is the sandbox;
the VM, operating system policy, resource limits, and brokered data flow are
the sandbox.

## Backend discovery and dry-run (current boundary)

`src/substrate/product/backends.py` provides deterministic local backend
discovery over `platform.system`, `platform.machine`, and `shutil.which`. It
prefers Apple Container, falls back to Docker, or reports no eligible backend.
It validates the active apprenticeship and sandbox-plan digests before it emits
an entity-bound dry-run plan. This is still not a trusted execution adapter:
there is no image pull, worker launch, daemon access, or host mutation.

## Next implementation increments

1. Complete the signed-manifest, local-trust, and content-addressed-cache
   foundation with pack-artifact references, SBOM/notice descriptors,
   verifier-key rotation/revocation, and fault-injection coverage.
2. Implement a reviewed local task backend with no-network default and
   validated argument-vector adapters for compilers, formal tools, browser,
   and media processing.
3. Build a bounded coding apprenticeship evaluation with hidden tasks and
   controls before making any specialist-capability claim.
