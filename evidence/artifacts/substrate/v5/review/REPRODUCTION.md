# Substrate v5 reproduction

1. Check out the frozen V5 ready or terminal tag with prior tags available.
2. Validate `RAW_RECEIPT_INDEX.json` and decompress `RAW_RECEIPTS.jsonl.gz`.
3. Install with `python -m pip install '.[dev]'`.
4. Run the test suite, Ruff, `substrate v5 status`, and `substrate v5 verify`.
5. Regenerate twice and compare normalized canonical digests.

Activation must remain `false`; review readiness is not an unqualified Nous claim.
