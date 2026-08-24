# Source adapters and provenance

Source adapters are the future boundary between Substrate and material it is
allowed to observe. They should turn a narrow approved request into a
provenance-bearing artifact flow; they should not give a model an unrestricted
web shell.

## Current boundary

The current source layer records non-executing acquisition plans and validates
their policy shape. The current artifact cache accepts only a local
operator-supplied regular file. It does not fetch a URL, invoke yt-dlp, launch
a browser, extract an archive, decode media, or run a repository tool.

`src/substrate/product/source_adapters.py` adds a second, deliberately narrower
contract for *observing already admitted evidence*. It accepts only a cache
identity plus immutable artifact digest — never a live URL, host path, command,
flag, browser profile, cookie, credential, or executable selector. Its closed
v1 requests cover:

- local-file views: metadata, text, and document structure;
- immutable repository snapshot views: tree, commit metadata, file metadata,
  and bounded text content;
- approved media observations: probe, metadata, subtitles, bounded frame
  sampling, and bounded audio extraction; and
- already-captured browser bundles: DOM, accessibility tree, screenshot,
  bounded frame capture, and bounded audio capture.

Each `substrate-source-adapter-plan-v1` plan is digest-bound,
`execution_permitted: false`, requires an immutable verified or processed
cache input, and declares that every prospective derivative returns to
quarantine. It explicitly denies
network egress, process and shell execution, arbitrary flags, downloader
invocation, host-path access, credentials, browser profiles/cookies, and cache
promotion. It is a stable input/output protocol for the future broker, not a
back door to a local browser or media stack.

Before a serialized plan crosses into a future broker, it must be decoded with
the product's strict JSON reader and passed through
`parse_observation_plan`. That parser re-hashes the plan, rebuilds its closed
request type, and requires every declared input/output/denial field to match
the typed contract. It does not open a cache object or execute a tool.

The cache recognizes `application/x-substrate-…+json` only when its complete,
bounded blob is a strict UTF-8 JSON object with duplicate keys and nonstandard
constants refused. Ordinary `application/json` is not thereby promoted to a
typed observation artifact; schema-specific validation stays with the source
adapter/output receipt contract.

Source requests are deliberately conservative. Credentials, query strings,
fragments, explicit ports, and non-approved source locations are refused.
Local `file:` sources must be absolute and live beneath an operator-declared,
non-root file root. Remote sources need an approved scheme and exact approved
authority. A future adapter should persist a sanitized reference or opaque
approved identifier rather than a credential-bearing session URL.

## Required provenance record

Every acquired or derived artifact needs enough information for later review:

- policy/plan digest and a safe source-reference digest;
- declared access and rights status, retrieval method, and timestamp;
- byte digest, byte length, and observed media type;
- the exact input object digests for any derivative;
- a typed transformation recipe and pinned tool/image digest;
- explicit verifier/attestation decision, promotion state, and quarantine
  reason where applicable.

The artifact cache uses `substrate-artifact-cache-v1`; its attestation record
uses `substrate-artifact-cache-attestation-v1`. An attestation is an Ed25519
signature over the cache identity, exact quarantined-descriptor digest,
artifact digest, rights decision, and expiry; a matching local verifier trust
rule is also required. Raw objects remain in quarantine until that check and
repeat validation promote them to the verified zone. Derived objects use a
separate processed zone, retain typed lineage, and are revoked when an input
is revoked. Cache objects are not entity state, and a portable entity stores
only safe digests and receipts rather than arbitrary local paths.

The present local control path is deliberately explicit:

```text
substrate product cache trust-verifier
substrate product cache add
substrate product cache attest
substrate product cache verify
```

`attest` produces a signed verifier statement but cannot promote an object;
`verify` checks that statement against a separate local trust store and emits
the cache's verification receipt during promotion.

## Adapter progression

| Adapter type | Safe first operation | Not an allowed shortcut |
| --- | --- | --- |
| Repository | operator imports a reviewed immutable snapshot | cloning arbitrary URLs, preserving host hooks/credentials |
| Document | operator imports a licensed or user-provided file | unbounded document crawling or opaque archive expansion |
| Media | broker processes approved source bytes into bounded derivatives | blanket video downloading, DRM/access-control bypass, rights-free scraping |
| Browser | brokered observation of an approved source in a task-only context | user-session reuse, open-ended browsing, host browser automation |
| Simulation | read approved scene/telemetry inputs and produce quarantined results | access to physical actuators, LAN devices, or host services |

For a source that needs network access, a later broker must validate a
source-specific grant before any connection: scheme, exact authority, method,
redirect and DNS-rebinding policy, byte/rate/time quotas, and a denylist for
localhost, LAN, metadata endpoints, and unapproved destinations. There is no
general “download anything it finds” permission.

## Media and YouTube

yt-dlp and FFmpeg appear only as optional future media requirements in the
`media` pack. They are not included, converted, launched, or treated as a
license for acquisition. A lawful future use must be limited to material the
operator is permitted to access and process, using a source policy and a
provenance record. Depending on the source, a better first route may be an
operator-provided file, licensed dataset, approved caption/transcript export,
or an official API.

The tool-bundle contract can record a digest-pinned `yt-dlp` or FFmpeg artifact
and its legal/verification material, but it remains non-executing and has
`network_mode: none`. A later source-specific broker — not the manifest and
not a model-generated command line — must decide whether a permitted staging
operation receives any bounded egress grant.

## Assimilation rule

Observing an item is not equivalent to learning from it. A worker can return
evidence and an explanation, but it cannot directly modify the entity. The
implemented source-evidence bridge accepts only a sealed acquisition plan and
a typed cache-attested authority; it records sanitized provenance through the
single-writer path. Evaluator-backed competence, procedure, project, and
unfinished-work updates remain future work.
