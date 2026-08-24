# Portable entity format

An entity is persistent developmental state, not a model checkpoint or a
sandbox instance. The current portable layout is intentionally small:

```text
entity/
  entity.json
  developmental-state.json
  receipts.jsonl
  checkpoint.json
```

The entity manifest records an identifier, specialty, selected capability-pack
identifiers, and optional organ requirements. State records competence,
unfinished work, and the active apprenticeship binding. Receipts are
hash-linked, and a checkpoint binds the manifest, state, and ledger tail.

The directory does not carry raw source bytes, credentials, browser state,
tool images, host paths, or model weights. Cache object digests and compatible
organ requirements are the portable references. Source plans and assimilated
source receipts retain safe origin labels and digests rather than raw
URI/path/rights text. A POSIX writer lock, a sibling creation lock, and a
recoverable pending transaction protect the current reference implementation;
they do not turn local hash chains into a remote trust service.
