# Real-World Sandbox Readiness Protocol

The readiness package is an interface and containment contract for the next
campaign. It does not activate the system or authorize any external action.

Each task requires an objective, explicit consent receipt, tool allowlist,
privacy fields, and resource budget. Sensor packets require a content digest,
provenance, logical time, and the applicable consent receipt. Proposed tool
actions are separate from execution, default to operator approval, declare
reversibility, and remain unauthorized in this package.

Replaceable model adapters expose identity, health, bounded inference requests,
structured responses, cost, latency, support status, and limitations. Model
outputs remain proposals or evidence; unsupported output cannot become
knowledge.

Receipts record events, failures, and the number of external actions executed.
The bounded smoke test must report zero external executions and
`activation=false`.
