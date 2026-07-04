# CONDENSE_DOCS_REVIEW

Staged doc deletions for the grader (regenerable or byte-identical duplicates only). Never auto-committed.

## Status: RESULT-docs merge DONE (iter 11, 5047dd5); nothing else staged for deletion

No documentation deletion is staged by the committed condense run. No byte-identical duplicate markdown exists
(the only md5 collision in the tree is between empty package `__init__.py` files, which are code, not docs).
The committed continuation stayed code-structural.

The 4 standalone RESULT docs (A6_RESULT, LAPTOP_LANES_RESULT, AXIS_CEILING_RESULT, ROLLOUT_LANE_RESULT)
were merged VERBATIM into docs/mixture_of_perspectives/RESULTS_LEDGER.md and removed, with all ~13 inbound
references rewritten and check_docs CANONICAL_MD updated (iter 11, committed 5047dd5, gates green). Content
preservation was verified line-by-line and the broken-ref check is empty. This was user (grader) authorized.
The numbered MoP section docs remain a candidate for a later merge (higher cross-ref count).

## Deferred doc-consolidation opportunities (merge-only, for a future docs-track run; NO content loss)

These are candidates the grader (or a later docs-only run) could pursue by MERGING content under H2 anchors
and adding an index entry, never by deleting unique content. Each requires the check_docs CANONICAL_MD ledger
(scripts/check_docs.py, lines ~32-68) updated in the SAME commit, because that gate enforces the markdown
ledger.

- docs/mixture_of_perspectives/ numbered section files (01_thesis, 03_thinking_modes, 04_reasoning_program,
  05_plasticity_program, ...) could merge into a few thematic files under H2 anchors. ~12 files -> ~4.
- The standalone lane RESULT docs (A6_RESULT, LAPTOP_LANES_RESULT, AXIS_CEILING_RESULT, ROLLOUT_LANE_RESULT)
  could merge into one RESULTS_LEDGER.md with a section per lane, content preserved verbatim. 4+ -> 1.

Both are content-preserving merges, not deletions, so nothing is staged HERE for a delete-review; they are
recorded so the opportunity is not lost.
