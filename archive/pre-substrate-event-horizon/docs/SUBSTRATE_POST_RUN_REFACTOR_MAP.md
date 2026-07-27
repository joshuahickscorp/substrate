# Substrate post run refactor map

Written after the long run reached terminal, deliberately not before it. Every item here was noticed while
building or certifying and left alone, because a rename before terminal evidence invalidates the receipts
that reference it.

Nothing in this list is a defect. Defects were repaired as they were found and carry corrections. These
are shape problems: places where the code works and reads worse than it should.

## 1. Two module names for one concept

`body.py` declares the contract, `bodies.py` implements three of them. The one letter difference is a trap
for anyone reading an import line quickly.

**Do:** merge into `bodies.py` with the contract at the top, or rename to `body_contract.py` and
`body_adapters.py`. Twelve import sites.

**Blocked on:** nothing after this run. The artifacts move with the modules and would need one reseal.

## 2. `graph.py` and `campaign.py` overlap

`graph.NODES` is the materialized program graph, `campaign.STAGES` is an earlier stage list, and
`longrun.UNIT_LIST` is the frozen run. Three structures, two of them now historical.

**Do:** retire `campaign.STAGES` in favour of `graph.NODES`, and keep `longrun.UNIT_LIST` as the frozen
subset. The campaign driver becomes a thin executor over graph nodes.

**Blocked on:** `campaign.py` is referenced by item L1 and its receipts. Retiring it needs the item
retired or repointed, which is an append only change to the item table.

## 3. `certify.py` and `audit.py` split by accident, not by design

The audit is structural and static, certification is behavioural and runs fixtures. That is a real
distinction, but `certify.body_canaries` is neither: it is a conformance check that belongs beside the
bodies.

**Do:** move `body_canaries` into `bodies.py`, leave `certify.py` with the runtime and session work.

## 4. `program.ITEMS` is 67 entries in one tuple

It is readable in order and unreadable at a glance. The category and batch fields already imply a
grouping that the file does not use.

**Do:** split into per category tuples concatenated at the end, so a reader looking for the memory items
finds them together. Purely mechanical; no field changes.

## 5. `runtime.step` is long

Eleven stages, each guarded by an ablation check, in one method. The guards are repetitive in a way that
invites a decorator, and a decorator here would hide the stage boundaries the trace depends on.

**Do:** extract each stage to a private method taking and returning the trace, keeping the sequence
visible in `step`. Do not introduce a stage registry: the explicit order is the thing being certified.

## 6. Duplicated fixture construction

`certify.positive_fixture`, `divergence._observation` and `bodies.compare` each build a stream of
observations from the same three fields.

**Do:** one `fixtures.py` with the observation shape declared once.

## 7. The `SESOI` constant is declared in eight modules

Every one is 0.05 and they agree, which is luck rather than design.

**Do:** one declaration in `io.py`, imported. Check first that no module intends a different threshold;
`sx2` and `worldbed` both reason about it explicitly.

## 8. Test file boundaries drifted

`test_final_program.py` now holds graph, authority, temporal, bodies, sessions, goals, grounding and
divergence tests. It grew by accretion.

**Do:** split by subject to match the module names.

## What not to do

- Do not rename anything under `proof/` or `runs/`. Those names are in sealed receipts and in the
  temporal program's evidence.
- Do not merge `mop.cognition.io` into `mop.method.io` or `mop.temporal.io`. The three are deliberate
  siblings, each stamping its own program, and the older two sit behind sealed evidence.
- Do not generalise `longrun.py` into a scheduler framework. It is nineteen units and a claim check, and
  the temporal supervisor already exists for the case that needs more.

## Order

1, 6 and 7 are mechanical and safe. 3, 4 and 8 are moves with no logic change. 5 is a real edit and wants
its own commit and a full certification rerun. 2 is last: it retires a component and needs the item table
updated append only.
