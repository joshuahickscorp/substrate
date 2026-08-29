# Product Genesis security audit — revision 1

## Role

Resume the prior read-only security audit. Do not edit files, create a
worktree, install packages, or use web/network tools.

## Scope

Review the current versions of only:

- `src/substrate/product/cache.py`
- `src/substrate/product/pack_artifacts.py`
- `src/substrate/product/source_adapters.py`
- `src/substrate/product/tool_bundles.py`
- `native/substrate-sandbox/**`
- `tests/substrate/test_product_pack_cache.py`
- `tests/substrate/test_source_adapters.py`
- `tests/substrate/test_product_tool_bundles.py`

## Remediation to verify

The prior report identified two P1 concerns. Verify the actual code rather
than trusting this summary:

1. Every cache path that accepts a verified object now rehashes its blob and
   rechecks observed media type. `explain`, lineage traversal, status, GC, and
   source consumption must not treat post-promotion byte replacement as valid.
2. Derived-object revocation now follows both `input_sha256` and
   `tool_artifact_sha256`. Re-promotion of a derived object must refuse if any
   recursive input or tool lineage is no longer verified.

Also check the added pack key read hardening and exact local trust-rule field
validation. All surfaces must remain non-executing: no process, browser,
downloader, media tool, container, or network action may be present.

## Deliverable

Return only:

- residual P0/P1 findings, if any, with exact file/line references;
- a concise list of verified remediations;
- any P2 limitation that must remain explicit in docs; and
- final verdict: `pass with stated limitations` or `not ready`.

