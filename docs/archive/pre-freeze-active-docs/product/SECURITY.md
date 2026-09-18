# Product security boundary

Substrate's product foundation is deliberately fail-closed. It is not an
authorization to execute a browser, downloader, compiler, container, or
model-controlled shell command.

## Implemented controls

- Capability packs are canonical manifest data signed with Ed25519 and scoped
  by a separate local trust rule. A pack signature does not install a binary
  or authorize execution.
- The cache admits only local regular files, hashes them while copying, starts
  in quarantine, and rechecks bytes and media type before promotion.
- Promotion requires an unexpired signed verifier statement bound to the cache
  identity and exact quarantined descriptor plus a locally trusted verifier
  key. The cache writes its own promotion receipt and retains the signed
  quarantined descriptor, so a later relabeling of the source, rights, media
  type, bytes, or processing lineage fails closed.
- A durable transition marker permits recovery after a failed cache-zone
  rename. A derived object keeps forward and reverse lineage; revoking an
  input or its pinned tool revokes downstream derivatives.
- Every implemented read path that accepts a verified cache object rehashes
  bytes, rechecks media type, revalidates the attestation signature against an
  explicit local verifier trust store, and refuses an expired attestation.
  A reopened cache therefore needs its `LocalCacheVerifierTrustStore` supplied
  before `status`, `explain`, `pin`, or derivative admission may represent a
  verified object. The CLI exposes this as `--verifier-trust-store`. Trusted
  consumption additionally requires a complete verified lineage closure.
- Source contracts reject credential-bearing and ambiguous URI forms. There is
  no source fetcher in the current product code.

## Explicitly absent

- No executable sandbox broker, worker, egress proxy, secret injection, or
  host integration exists.
- No yt-dlp, FFmpeg, Chromium, Git, Blender, compiler, model, or source
  repository is bundled, rewritten, installed, or invoked.
- Tool-bundle manifests are self-hashed declarations, not signed installed-tool
  authority. There is not yet a trusted tool-install receipt that binds a
  cache artifact to its tool id, platform, SBOM/license material, adapter role,
  and permitted operation; there is also no single cross-language role registry
  shared by packs, tool bundles, and the Rust body policy.
- Cache file modes are not treated as access control against the cache owner.
  Consumption revalidates bytes and the receipt-to-provenance binding; a
  future privileged cache service must still strengthen the storage boundary.
- `cache quarantine` remains a monotonic recovery operation: it can move an
  internally coherent but untrusted verified-looking entry back to quarantine
  without a trust store. It never reports that entry as verified evidence.

## Future execution gate

Execution may be considered only after an independently reviewed broker
enforces short-lived isolation, fixed mounts, resource and process limits,
network denial by default, validated argument vectors, source-specific egress,
complete receipts, output quarantine, and escape/fault-injection canaries. The
gate must also add a locally trusted signed tool inventory/install receipt and
a generated or otherwise single-source closed adapter-role registry before a
tool bundle can bind to any worker plan.
