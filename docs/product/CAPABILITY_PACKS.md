# Capability packs

Capability packs describe what a future sandboxed task may need. They are not
tool bundles, installers, execution permits, or grants of open-world access.
Every built-in pack currently defaults to `network_mode: none` and produces
only non-executing plans.

## Built-in declarations

| Pack | Intended inputs and roles | Default worker floor |
| --- | --- | --- |
| `engineering` | repositories, text, compiler/test/repository roles | 2 CPU / 4 GiB / 8 GiB disk |
| `research` | documents, repositories, text, document-observation and retrieval roles | 1 CPU / 2 GiB / 4 GiB disk |
| `media` | approved audio, images, video, media-observation roles | 4 CPU / 8 GiB / 16 GiB disk |
| `browser` | approved document, image, text, video observation | 2 CPU / 4 GiB / 8 GiB disk |
| `formal-math` | documents and text, Lean/symbolic/solver roles | 2 CPU / 4 GiB / 4 GiB disk |
| `3d` | images, scenes, video, geometry/simulation roles | 4 CPU / 12 GiB / 16 GiB disk |
| `desktop` | a future isolated virtual desktop observation role | 2 CPU / 4 GiB / 8 GiB disk |
| `data-science` | approved datasets, documents, repositories, text | 4 CPU / 8 GiB / 16 GiB disk |
| `robotics` | simulation scenes, telemetry, images, video | 4 CPU / 12 GiB / 16 GiB disk |

`mathematics` and `three-d` remain accepted legacy identifiers for existing
portable entities; new product-facing examples should use `formal-math` and
`3d`.

The resource figures are planning minima, not a promise to consume all host
resources. A scheduler must reserve control-plane capacity and enforce CPU,
memory, disk, process, I/O, time, and egress limits together.

## What a pack artifact contains

The artifact schema is `substrate-capability-pack-v1`. Its signed manifest
describes the following without embedding executable content:

- pack name, semantic version, publisher, supported host identities, and the
  required product schema;
- an explicit, closed capability set and matching denial canary;
- the three portable mount grants only: read-only content-addressed inputs,
  ephemeral task workspace, and quarantine-only output;
- a resource class, typed source-adapter and tool-adapter roles, and required
  model-organ protocol names;
- operator-provided-or-verified-image binary requirements;
- empty network grants until an independently reviewed broker exists;
- license-review metadata and optional immutable artifact descriptors.

Signature, local trust, and installation-record schemas are respectively
`substrate-capability-pack-signature-v1`,
`substrate-capability-pack-trust-v1`, and
`substrate-capability-pack-install-v1`.

An Ed25519 signature binds canonical manifest data. A local trust decision
must additionally bind the publisher key to the allowed pack names and
capabilities. Successful verification only verifies the declaration; it does
not authorize execution, retrieval, installation of a binary, or network use.

## Media and browser scope

The `media` declaration may name FFmpeg and yt-dlp as optional
operator-provided or verified-image requirements. The `browser` declaration
may name Chromium and a browser-automation role. None of those tools are
vendored, rewritten, downloaded, run, or assumed licensed by this repository.

Future adapters must pin their exact binary/image digest, version, SBOM,
license posture, entrypoint, and transformation receipt. They must also apply
source policy and rights checks before acquiring content. In particular, a
media adapter must not bypass DRM, access controls, service terms, rate limits,
or rights restrictions; it is not a generic downloader or scraping feature.

`src/substrate/product/tool_bundles.py` now supplies the pre-execution
manifest shape for that pinning. `substrate-tool-bundle-manifest-v1` permits
only an operator-provided OCI image or binary-archive digest plus its byte
size, SBOM, notices, license document, SPDX identifier, and prior verification
receipt digest. It binds each approved utility to a closed adapter role and
closed operation list. It rejects commands, flags, executable or host paths,
image tags, credentials, URLs, and network grants, and always retains
`execution_permitted: false`. The manifest is not a tool installer and is not
a signature or authorization by itself.

## Installation and removal boundary

The pack CLI is intentionally limited to local manifest lifecycle work:

```text
substrate product pack build
substrate product pack keygen
substrate product pack sign
substrate product pack inspect
substrate product pack trust
substrate product pack verify
substrate product pack install
substrate product pack remove
```

`install` records a verified manifest reference. `remove` removes that local
registry reference only; it does not remove shared cache objects or arbitrary
software from the host.

The adjacent tool-bundle surface is intentionally inspection-only:

```text
substrate product tool-bundle inspect ./sensory-tools.json
```

It rehashes and validates a local manifest; it does not resolve its digests,
pull an image, unpack an archive, or start a process.
