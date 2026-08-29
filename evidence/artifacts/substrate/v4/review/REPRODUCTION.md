# Substrate v4 reproduction

1. Check out `substrate-v4-terminal` with tags available.
2. Decompress `RAW_RECEIPTS.jsonl.gz`; write each embedded `document` to its recorded `path`.
3. Install with `python -m pip install '.[dev]'`.
4. Run `substrate test`, `substrate v4 status`, and `substrate v4 verify`.
5. Compare regenerated seals and endpoint ledgers against `REVIEW_INDEX.json`.

Activation must remain `false`. The review-candidate tag is not an unqualified Nous claim.
