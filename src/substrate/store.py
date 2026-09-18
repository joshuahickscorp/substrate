"""Transactional entity state, compact evidence and restartable work in one store.

WAL permits readers alongside one writer on a local filesystem. Every causal
entity mutation is a delta plus a hash-linked receipt, not a rewrite of history.
Evaluator databases are physically separate from actor databases; visibility is
not implemented by asking an LLM to ignore columns.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
import stat
import tempfile
import time
import uuid
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterator

from .core import (MAX_JSON, SCHEMA, ZERO, Conflict, Refused, bytes_sha, canonical, clone,
                   digest, fields, identifier, integer, numeric, parse, require, sha)

DDL = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value BLOB NOT NULL);
CREATE TABLE objects(id TEXT PRIMARY KEY, codec TEXT NOT NULL, raw_size INTEGER NOT NULL,
                     payload BLOB NOT NULL);
CREATE TABLE refs(parent TEXT NOT NULL REFERENCES objects(id),
                  child TEXT NOT NULL REFERENCES objects(id), PRIMARY KEY(parent,child));
CREATE TABLE entities(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, head TEXT NOT NULL,
                      manifest TEXT NOT NULL REFERENCES objects(id), frozen INTEGER NOT NULL);
CREATE TABLE records(entity TEXT NOT NULL REFERENCES entities(id), namespace TEXT NOT NULL,
                     key TEXT NOT NULL, object TEXT NOT NULL REFERENCES objects(id),
                     PRIMARY KEY(entity,namespace,key));
CREATE TABLE events(entity TEXT NOT NULL REFERENCES entities(id), revision INTEGER NOT NULL,
                    id TEXT NOT NULL UNIQUE, previous TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL REFERENCES objects(id), created REAL NOT NULL,
                    PRIMARY KEY(entity,revision));
CREATE TABLE jobs(id TEXT PRIMARY KEY, entity TEXT NOT NULL, phase TEXT NOT NULL,
                  payload TEXT NOT NULL REFERENCES objects(id), state TEXT NOT NULL,
                  owner TEXT, fence INTEGER NOT NULL DEFAULT 0, expires REAL,
                  attempts INTEGER NOT NULL DEFAULT 0, result TEXT REFERENCES objects(id));
CREATE TABLE dependencies(job TEXT NOT NULL REFERENCES jobs(id),
                          prerequisite TEXT NOT NULL REFERENCES jobs(id),
                          PRIMARY KEY(job,prerequisite));
CREATE INDEX jobs_ready ON jobs(state,phase);
CREATE TABLE effects(id TEXT PRIMARY KEY, job TEXT NOT NULL REFERENCES jobs(id),
                     request TEXT NOT NULL REFERENCES objects(id), state TEXT NOT NULL,
                     reserved_calls INTEGER NOT NULL, reserved_tokens INTEGER NOT NULL,
                     response TEXT REFERENCES objects(id));
CREATE TABLE counters(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE outcomes(id TEXT PRIMARY KEY, entity TEXT NOT NULL, unit TEXT NOT NULL,
                      arm TEXT NOT NULL, phase TEXT NOT NULL, family TEXT NOT NULL,
                      case_id TEXT NOT NULL, summary TEXT NOT NULL REFERENCES objects(id));
CREATE INDEX outcomes_analysis ON outcomes(phase,arm,unit);
"""


def references(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            result.add(digest(value["$ref"]))
        else:
            for child in value.values():
                result.update(references(child))
    elif isinstance(value, list):
        for child in value:
            result.update(references(child))
    return result


class Store:
    def __init__(self, path: Path, *, create: bool = False, role: str = "actor",
                 max_payload_bytes: int = 256 * 1024 * 1024):
        self.path = Path(path)
        self.limit = integer(max_payload_bytes, 1024)
        require(role in {"actor", "evaluator"}, "unknown store role")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                         getattr(os, "O_NOFOLLOW", 0), 0o600)
            os.close(fd)
        info = self.path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not self.path.is_symlink(),
                "store must be an operator-owned regular file")
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA trusted_schema=OFF")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        if create:
            self.db.executescript(DDL)
            with self.transaction():
                self.db.executemany("INSERT INTO meta VALUES (?,?)", [
                    ("schema", canonical(SCHEMA)), ("role", canonical(role)),
                    ("created", canonical(time.time()))])
                self.db.executemany("INSERT INTO counters VALUES (?,0)",
                                    [("payload_bytes",), ("calls",), ("tokens",)])
        require(self.meta("schema") == SCHEMA and self.meta("role") == role,
                "store schema or actor/evaluator role mismatch")
        self.role = role

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        require(not self.db.in_transaction, "nested store transaction is not supported")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def meta(self, key: str) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return parse(row[0]) if row else None

    def set_meta(self, key: str, value: Any) -> None:
        require(self.db.in_transaction, "metadata mutation requires a transaction")
        self.db.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (identifier(key), canonical(value)))

    def counter(self, key: str) -> int:
        row = self.db.execute("SELECT value FROM counters WHERE key=?", (key,)).fetchone()
        return row[0] if row else 0

    def add_counter(self, key: str, amount: int) -> None:
        require(self.db.in_transaction and type(amount) is int, "counter mutation requires a transaction")
        self.db.execute("INSERT INTO counters VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=value+excluded.value",
                        (identifier(key), amount))

    def put(self, value: Any) -> str:
        """Deduplicate canonical JSON; zlib level 1 is chosen for ingestion, not maximum ratio."""
        require(self.db.in_transaction, "object publication requires a transaction")
        raw = canonical(value)
        require(len(raw) <= MAX_JSON, "object too large; segment it explicitly")
        key = bytes_sha(raw)
        if self.db.execute("SELECT 1 FROM objects WHERE id=?", (key,)).fetchone():
            return key
        children = references(value)
        for child in children:
            require(self.db.execute("SELECT 1 FROM objects WHERE id=?", (child,)).fetchone() is not None,
                    "object reference is missing")
        compressed = zlib.compress(raw, 1) if len(raw) >= 512 else raw
        use_compression = len(compressed) + 32 < len(raw)
        payload = compressed if use_compression else raw
        require(self.counter("payload_bytes") + len(payload) <= self.limit, "retained-payload budget exhausted")
        self.db.execute("INSERT INTO objects VALUES (?,?,?,?)",
                        (key, "zlib" if use_compression else "raw", len(raw), payload))
        self.db.executemany("INSERT INTO refs VALUES (?,?)", [(key, child) for child in sorted(children)])
        self.add_counter("payload_bytes", len(payload))
        return key

    def get(self, key: str) -> Any:
        row = self.db.execute("SELECT * FROM objects WHERE id=?", (digest(key),)).fetchone()
        require(row is not None, "missing content-addressed object")
        size = integer(row["raw_size"], 0, MAX_JSON)
        if row["codec"] == "zlib":
            decoder = zlib.decompressobj()
            try:
                raw = decoder.decompress(row["payload"], size + 1)
            except zlib.error as exc:
                raise Refused("invalid compressed object") from exc
            require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                    "compressed object is truncated, oversized or has trailing data")
        else:
            require(row["codec"] == "raw", "unknown object codec")
            raw = bytes(row["payload"])
        require(len(raw) == size and bytes_sha(raw) == key, "object integrity failure")
        return parse(raw)

    def _create_entity(self, entity: str, constitution: dict, *, lineage: dict | None = None) -> str:
        require(self.db.in_transaction, "entity creation requires a transaction")
        identifier(entity)
        fields(constitution, {"self_modify", "external_authority", "goal"})
        require(constitution["self_modify"] is False and constitution["external_authority"] == "operator-permit",
                "constitution cannot grant its own execution or self-modification authority")
        manifest = {"schema": SCHEMA, "entity": entity, "constitution": clone(constitution),
                    "lineage": clone(lineage or {}), "created": time.time()}
        key = self.put(manifest)
        self.db.execute("INSERT INTO entities VALUES (?,0,?,?,0)", (entity, ZERO, key))
        return self._commit(entity, ZERO, "born", {"manifest": {"$ref": key}}, [])

    def create_entity(self, entity: str, constitution: dict, *, lineage: dict | None = None) -> str:
        with self.transaction():
            return self._create_entity(entity, constitution, lineage=lineage)

    def entity(self, entity: str) -> dict:
        row = self.db.execute("SELECT * FROM entities WHERE id=?", (identifier(entity),)).fetchone()
        require(row is not None, "unknown entity")
        return dict(row)

    def head(self, entity: str) -> str:
        return self.entity(entity)["head"]

    def record(self, entity: str, namespace: str, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT object FROM records WHERE entity=? AND namespace=? AND key=?",
                              (entity, identifier(namespace), identifier(key))).fetchone()
        return self.get(row[0]) if row else clone(default)

    def records(self, entity: str, namespace: str) -> dict[str, Any]:
        return {row["key"]: self.get(row["object"]) for row in self.db.execute(
            "SELECT key,object FROM records WHERE entity=? AND namespace=? ORDER BY key", (entity, namespace))}

    def iter_records(self, entity: str, namespace: str):
        """Stream content-addressed records without loading a whole personal corpus."""
        for row in self.db.execute("SELECT key,object FROM records WHERE entity=? AND namespace=? ORDER BY key",
                                   (entity, identifier(namespace))):
            yield row["key"], self.get(row["object"])

    def _commit(self, entity: str, expected: str, kind: str, payload: Any,
                changes: list[tuple[str, str, Any]], *, allow_frozen: bool = False) -> str:
        require(self.db.in_transaction, "entity commit requires a transaction")
        current = self.entity(entity)
        if current["head"] != expected:
            raise Conflict("entity revision changed")
        require(allow_frozen or not current["frozen"], "entity is frozen; use a declared experimental fork")
        operations = []
        for namespace, key, value in changes:
            identifier(namespace)
            identifier(key)
            object_id = self.put(value)
            operations.append([namespace, key, {"$ref": object_id}])
            self.db.execute("INSERT INTO records VALUES (?,?,?,?) ON CONFLICT(entity,namespace,key) "
                            "DO UPDATE SET object=excluded.object", (entity, namespace, key, object_id))
        object_id = self.put({"data": payload, "changes": operations})
        unsigned = {"entity": entity, "revision": current["revision"] + 1, "previous": expected,
                    "kind": identifier(kind), "payload": object_id, "created": time.time()}
        event_id = sha(unsigned)
        self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                        (entity, unsigned["revision"], event_id, expected, kind, object_id, unsigned["created"]))
        self.db.execute("UPDATE entities SET revision=?,head=? WHERE id=?",
                        (unsigned["revision"], event_id, entity))
        return event_id

    def commit(self, entity: str, expected: str, kind: str, payload: Any,
               changes: list[tuple[str, str, Any]]) -> str:
        with self.transaction():
            return self._commit(entity, expected, kind, payload, changes)

    def freeze(self, entity: str, expected: str) -> str:
        with self.transaction():
            head = self._commit(entity, expected, "frozen", {}, [])
            self.db.execute("UPDATE entities SET frozen=1 WHERE id=?", (entity,))
            return head

    def continue_entity(self, entity: str, expected: str, receipt: dict) -> str:
        """Storage primitive; caller must verify an operator continuity attestation first."""
        with self.transaction():
            row = self.entity(entity)
            require(row["frozen"] and row["head"] == expected, "continuation needs the pinned frozen checkpoint")
            self.db.execute("UPDATE entities SET frozen=0 WHERE id=?", (entity,))
            return self._commit(entity, expected, "continuation-authorized", receipt, [])

    def fork(self, parent: str, child: str, *, intervention: str,
             omit_namespaces: tuple[str, ...] = ()) -> str:
        """An ablation is a new branch with explicit parentage, never a hidden edit."""
        with self.transaction():
            require(self.entity(parent)["frozen"], "experimental forks require a frozen parent")
            parent_head = self.head(parent)
            manifest = self.get(self.entity(parent)["manifest"])
            self._create_entity(child, manifest["constitution"], lineage={"parent": parent, "head": parent_head,
                                                                         "intervention": intervention})
            changes = []
            for row in self.db.execute("SELECT * FROM records WHERE entity=? ORDER BY namespace,key", (parent,)):
                if row["namespace"] not in omit_namespaces:
                    changes.append((row["namespace"], row["key"], self.get(row["object"])))
            return self._commit(child, self.head(child), "inherit", {"parent": parent, "head": parent_head,
                                                                     "omit": list(omit_namespaces)}, changes)

    def audit(self, entity: str) -> dict:
        """Linear offline audit. Normal steps never rescan all prior episodes."""
        previous, count, reconstructed = ZERO, 0, {}
        for row in self.db.execute("SELECT * FROM events WHERE entity=? ORDER BY revision", (entity,)):
            count += 1
            unsigned = {key: row[key] for key in ("entity", "revision", "previous", "kind", "payload", "created")}
            require(row["revision"] == count and row["previous"] == previous and row["id"] == sha(unsigned),
                    "entity event-chain corruption")
            payload = self.get(row["payload"])
            for namespace, key, ref in payload["changes"]:
                self.get(ref["$ref"])
                reconstructed[(namespace, key)] = ref["$ref"]
            previous = row["id"]
        current = self.entity(entity)
        live = {(r["namespace"], r["key"]): r["object"] for r in self.db.execute(
            "SELECT * FROM records WHERE entity=?", (entity,))}
        require(previous == current["head"] and count == current["revision"] and reconstructed == live,
                "materialized entity state disagrees with history")
        self.get(current["manifest"])
        return {"entity": entity, "head": previous, "events": count, "records": len(live), "audit": "intact"}

    def enqueue(self, jobs: list[dict]) -> None:
        with self.transaction():
            for job in jobs:
                fields(job, {"id", "entity", "phase", "payload", "depends_on"})
                key = self.put(job["payload"])
                existing = self.db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
                if existing:
                    require(existing["payload"] == key and existing["entity"] == job["entity"]
                            and existing["phase"] == job["phase"], "job identity reused for different work")
                    previous_dependencies = {row[0] for row in self.db.execute(
                        "SELECT prerequisite FROM dependencies WHERE job=?", (job["id"],))}
                    require(previous_dependencies == set(job["depends_on"]), "job dependencies cannot be rewritten")
                else:
                    self.db.execute("INSERT INTO jobs(id,entity,phase,payload,state) VALUES (?,?,?,?, 'pending')",
                                    (identifier(job["id"]), identifier(job["entity"]), identifier(job["phase"]), key))
            for job in jobs:
                require(job["id"] not in job["depends_on"], "job depends on itself")
                self.db.executemany("INSERT OR IGNORE INTO dependencies VALUES (?,?)",
                                    [(job["id"], dep) for dep in job["depends_on"]])
            rows = self.db.execute("SELECT job,prerequisite FROM dependencies").fetchall()
            graph: dict[str, set[str]] = {}
            for row in rows:
                graph.setdefault(row[0], set()).add(row[1])
            visiting, done = set(), set()

            def visit(node):
                require(node not in visiting, "cyclic job graph")
                if node in done:
                    return
                visiting.add(node)
                for dependency in graph.get(node, ()):
                    visit(dependency)
                visiting.remove(node)
                done.add(node)

            for node in graph:
                visit(node)

    def claim(self, owner: str, *, lease_seconds: float = 600, max_running: int = 64) -> dict | None:
        numeric(lease_seconds, 1, 86_400)
        integer(max_running, 1, 64)
        now = time.time()
        with self.transaction():
            expired = self.db.execute("SELECT id FROM jobs WHERE state='running' AND expires<?", (now,)).fetchall()
            for row in expired:
                # A crashed run can have a sent-but-unrecorded request. Never silently replay it.
                self.db.execute("UPDATE jobs SET state='uncertain',owner=NULL WHERE id=?", (row[0],))
            if self.db.execute("SELECT COUNT(*) FROM jobs WHERE state='running'").fetchone()[0] >= max_running:
                return None
            row = self.db.execute("""
                SELECT j.* FROM jobs j WHERE j.state='pending'
                AND NOT EXISTS(SELECT 1 FROM dependencies d JOIN jobs p ON p.id=d.prerequisite
                               WHERE d.job=j.id AND p.state!='done')
                AND NOT EXISTS(SELECT 1 FROM jobs active WHERE active.entity=j.entity
                               AND active.state IN ('running','uncertain'))
                ORDER BY j.id LIMIT 1
            """).fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE jobs SET state='running',owner=?,fence=fence+1,expires=?,attempts=attempts+1 WHERE id=?",
                            (identifier(owner), now + lease_seconds, row["id"]))
            result = dict(self.db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())
            result["document"] = self.get(result["payload"])
            return result

    def guard_job(self, job: dict) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
        if not row or row["state"] != "running" or row["owner"] != job["owner"] or row["fence"] != job["fence"]:
            raise Conflict("stale worker fence")
        if row["expires"] <= time.time():
            raise Conflict("worker lease expired")
        return row

    def heartbeat(self, job: dict, lease_seconds: float = 600) -> None:
        numeric(lease_seconds, 1, 86_400)
        with self.transaction():
            self.guard_job(job)
            self.db.execute("UPDATE jobs SET expires=? WHERE id=?", (time.time() + lease_seconds, job["id"]))

    def complete(self, job: dict, result: Any, *, state: str = "done") -> None:
        require(state in {"done", "failed", "uncertain"}, "invalid completion state")
        with self.transaction():
            self.guard_job(job)
            require(state != "done" or not self.db.execute(
                "SELECT 1 FROM effects WHERE job=? AND state IN ('reserved','sent','uncertain')", (job["id"],)).fetchone(),
                "unsettled effect cannot complete a job")
            self.db.execute("UPDATE jobs SET state=?,result=?,owner=NULL WHERE id=?",
                            (state, self.put(result), job["id"]))

    def reserve_effect(self, job: dict, request: Any, *, max_tokens: int,
                       call_limit: int, token_limit: int) -> str:
        integer(max_tokens, 1)
        with self.transaction():
            self.guard_job(job)
            require(self.counter("calls") + 1 <= call_limit, "model call budget exhausted")
            require(self.counter("tokens") + max_tokens <= token_limit, "model token budget exhausted")
            request_id = self.put(request)
            key = sha({"job": job["id"], "fence": job["fence"], "request": request_id,
                       "nonce": uuid.uuid4().hex})
            self.db.execute("INSERT INTO effects VALUES (?,?,?,'reserved',1,?,NULL)",
                            (key, job["id"], request_id, max_tokens))
            self.add_counter("calls", 1)
            self.add_counter("tokens", max_tokens)
            return key

    def sent_effect(self, job: dict, effect: str) -> None:
        with self.transaction():
            self.guard_job(job)
            count = self.db.execute("UPDATE effects SET state='sent' WHERE id=? AND job=? AND state='reserved'",
                                    (effect, job["id"])).rowcount
            require(count == 1, "effect was already dispatched or is not owned by this job")

    def settle_effect(self, job: dict, effect: str, response: Any, *, actual_tokens: int | None) -> None:
        with self.transaction():
            self.guard_job(job)
            row = self.db.execute("SELECT * FROM effects WHERE id=? AND job=?", (effect, job["id"])).fetchone()
            require(row is not None and row["state"] == "sent", "effect settlement is not admissible")
            # Missing provider usage retains the entire reservation, rather than inventing zero cost.
            if actual_tokens is not None:
                integer(actual_tokens)
                require(actual_tokens <= row["reserved_tokens"], "provider exceeded its reserved token envelope")
                self.add_counter("tokens", actual_tokens - row["reserved_tokens"])
            self.db.execute("UPDATE effects SET state='complete',response=? WHERE id=?", (self.put(response), effect))

    def add_outcome(self, record: dict, trace: dict | None = None, *, retain_success_every: int = 32) -> str:
        """Every compact outcome is retained. Trace sampling never selects statistical rows."""
        integer(retain_success_every, 1)
        fields(record, {"id", "entity", "unit", "arm", "phase", "family", "case_id", "metrics"})
        row_id = digest(record["id"])
        with self.transaction():
            summary = clone(record)
            keep = record["metrics"].get("correct") is False or int(row_id[:16], 16) % retain_success_every == 0
            if trace is not None and (keep or record["phase"] == "learn"):
                summary["trace"] = {"$ref": self.put(trace)}
            summary["trace_retention"] = "full" if "trace" in summary else "manifest-only"
            summary["trace_digest"] = sha(trace) if trace is not None else None
            key = self.put(summary)
            existing = self.db.execute("SELECT summary FROM outcomes WHERE id=?", (row_id,)).fetchone()
            if existing:
                require(existing[0] == key, "outcome ID collision with different content")
            else:
                self.db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?)",
                                tuple(record[k] for k in ("id", "entity", "unit", "arm", "phase", "family", "case_id"))
                                + (key,))
            return key

    def outcomes(self, *, phase: str | None = None) -> Iterator[dict]:
        query = "SELECT summary FROM outcomes" + (" WHERE phase=?" if phase else "") + " ORDER BY id"
        for row in self.db.execute(query, (phase,) if phase else ()):
            yield self.get(row[0])

    def storage(self) -> dict:
        names = [self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")]
        return {"payload_bytes": self.counter("payload_bytes"),
                "disk_bytes": sum(p.stat().st_size for p in names if p.exists()),
                "objects": self.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0],
                "sqlite_pages": self.db.execute("PRAGMA page_count").fetchone()[0],
                "note": "payload limit excludes SQLite indexes/WAL; provision and monitor disk separately"}

    def export_entity(self, entity: str, destination: Path) -> str:
        """Export only this entity's reachable objects; no credentials, leases or other entities."""
        require(not destination.exists(), "bundle destination already exists")
        with self.transaction():
            self.audit(entity)
            entry = self.entity(entity)
            events = [dict(row) for row in self.db.execute(
                "SELECT * FROM events WHERE entity=? ORDER BY revision", (entity,))]
            records = [dict(row) for row in self.db.execute(
                "SELECT * FROM records WHERE entity=? ORDER BY namespace,key", (entity,))]
            roots = {entry["manifest"], *(r["payload"] for r in events), *(r["object"] for r in records)}
            pending, reachable = list(roots), set()
            while pending:
                key = pending.pop()
                if key in reachable:
                    continue
                reachable.add(key)
                pending.extend(row[0] for row in self.db.execute("SELECT child FROM refs WHERE parent=?", (key,)))
            manifest = {"schema": "substrate-bundle-v2", "entity": entry, "records": records,
                        "events": events, "objects": sorted(reachable), "contains_host_authority": False}
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".bundle-", dir=destination.parent)
            os.close(fd)
            try:
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
                    bundle.writestr("manifest.json", canonical(manifest))
                    for key in sorted(reachable):
                        bundle.writestr("objects/" + key, canonical(self.get(key)))
                with open(temporary, "rb") as stream:
                    os.fsync(stream.fileno())
                os.link(temporary, destination)
            finally:
                os.unlink(temporary)
        return sha(manifest)

    def restore_entity(self, archive: Path, *, manifest_pin: str, max_bytes: int = 256 * 1024 * 1024) -> str:
        """A pin must come from the operator, not from an untrusted bundle itself."""
        digest(manifest_pin)
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            names = [x.filename for x in infos]
            require(len(names) == len(set(names)) and len(names) <= 100_000, "duplicate or excessive bundle entries")
            require(sum(x.file_size for x in infos) <= max_bytes, "bundle expansion exceeds budget")
            require("manifest.json" in names, "bundle manifest missing")
            manifest = parse(bundle.read("manifest.json"), max_bytes)
            fields(manifest, {"schema", "entity", "records", "events", "objects", "contains_host_authority"})
            require(sha(manifest) == manifest_pin and manifest["schema"] == "substrate-bundle-v2"
                    and manifest["contains_host_authority"] is False, "bundle pin or schema mismatch")
            expected = {"manifest.json", *("objects/" + digest(key) for key in manifest["objects"])}
            require(set(names) == expected, "unexpected bundle content")
            values = {key: parse(bundle.read("objects/" + key)) for key in manifest["objects"]}
            require(all(sha(value) == key for key, value in values.items()), "bundle object digest mismatch")
            entry = manifest["entity"]
            entity = identifier(entry["id"])
            require(not self.db.execute("SELECT 1 FROM entities WHERE id=?", (entity,)).fetchone(),
                    "restore never overwrites a live entity")
            with self.transaction():
                pending = dict(values)
                while pending:
                    ready = [key for key, value in pending.items() if all(
                        self.db.execute("SELECT 1 FROM objects WHERE id=?", (child,)).fetchone()
                        for child in references(value))]
                    require(bool(ready), "bundle has cyclic or missing object references")
                    for key in ready:
                        require(self.put(pending.pop(key)) == key, "object restore mismatch")
                self.db.execute("INSERT INTO entities VALUES (?,?,?,?,?)",
                                tuple(entry[k] for k in ("id", "revision", "head", "manifest", "frozen")))
                for row in manifest["records"]:
                    require(row["entity"] == entity, "foreign record in bundle")
                    self.db.execute("INSERT INTO records VALUES (?,?,?,?)",
                                    tuple(row[k] for k in ("entity", "namespace", "key", "object")))
                for row in manifest["events"]:
                    require(row["entity"] == entity, "foreign event in bundle")
                    self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                                    tuple(row[k] for k in ("entity", "revision", "id", "previous", "kind", "payload", "created")))
                self.audit(entity)
        return entity
