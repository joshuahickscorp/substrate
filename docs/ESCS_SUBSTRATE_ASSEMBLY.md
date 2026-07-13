# ESCS permissive substrate assembly

`configs/experiment/escs_substrate_assembly.json` projects the complete perspective-candidate registry
into one content-addressed ESCS assembly. All 31 requested facets have a stable slot. This is the low
Gate-A integration boundary: inclusion is cheap and reversible, while activation and scientific promotion
remain separate authorities.

The projection preserves evidence rather than flattening it:

- verified mechanics become inert infrastructure slots;
- toy positives become inert feature candidates;
- controlled nulls remain inert controls;
- pending, blocked, and untested mechanisms become sandbox stubs;
- failed mechanisms remain explicit exclusions.

Every slot binds the source candidate digest, interface, trigger authority, effect boundary, and required
guards. Every slot is disabled, the assembly defaults to quiescence, and neither a slot nor the assembly can
grant scientific promotion. Imagination and simulation remain counterfactual-only. Novelty, uncertainty,
and curiosity remain unable to trigger work. Unknown or silently changed candidates invalidate the registry
join.

This lets the substrate carry heterogeneous perspectives without paying their runtime cost or pretending
they are supported. Later experiment receipts may authorize exact feature flags or control runs, but only
through a new content-addressed registry/assembly revision and the existing runtime, commitment,
consequence, accounting, and rollback boundaries.

## Consolidated preflight

`configs/experiment/escs_substrate_preflight.json` joins the registry, the 31-slot assembly, the
mechanics chassis proof, the disabled G0 grammar, EDCM-1, and the X0-X3 implementation authorities by
exact file digest, schema, and semantic fields. `scripts/run_escs_substrate_preflight.py` independently
rebuilds those joins and publishes `proof/ESCS_SUBSTRATE_PREFLIGHT.json`.

The current preflight is scaffold-ready with all declared bindings exact and all 31 perspective slots
installed. It also records the intentionally negative activation facts: runtime activation is not ready,
G0 evaluator construction is incomplete, and scientific promotion is unavailable. A valid preflight is
therefore consolidated integration evidence, not a capability result. Any artifact drift, registry splice,
schema mismatch, enabled slot, or enabled topology grammar fails the report closed.
