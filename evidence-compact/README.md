# Compact evidence pack

This directory is the full source `evidence/` tree in a smaller, portable
representation. It is not a summary or a model/runtime cache.

- `MANIFEST.json` records the source inventory and counts.
- `json-index.jsonl.zst` maps every JSON source path to its canonical blob and
  preserves the original source byte count and SHA-256.
- `json-blobs.jsonl.zst` stores each unique JSON value once as sorted,
  whitespace-free JSON.
- `raw-files.tar.zst` preserves non-JSON files and JSON files that could not be
  safely canonicalized byte-for-byte.

The canonical JSON form preserves the JSON values and keys but not original
whitespace or key order. The original source SHA-256 remains in the index so
the transformation is auditable. The source evidence is not connected to a
runtime, model, external data volume, or launcher.

To inspect the pack without unpacking it:

```sh
zstd -dc json-index.jsonl.zst | head
zstd -dc json-blobs.jsonl.zst | head
zstd -dc raw-files.tar.zst | tar -tf - | head
```
