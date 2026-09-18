"""Portable lineage, conservative v1 migration and Hawking training preparation.

Preparing a dataset does not train an organ. An entity procedure is not a whole NR,
and an NR manifest is not a loadable NX. All state/model changes require new
artifact identities and independent qualification before deployment.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
from collections import Counter
from pathlib import Path

from .cognition import Entity
from .core import (Trust, bytes_sha, canonical, clone, digest, fields, identifier, read_json,
                   require, sha, write_json)
from .store import Store

import json
import math
import shutil
import struct
from collections import defaultdict
from .core import Refused, integer, numeric, scope_matches
from .language import Program, compile_program


def migrate_v1(store: Store, document: dict, *, pin: str, name: str | None = None) -> dict:
    """Archival migration, not silent semantic conversion of old qualifications.

    Verify the original checkpoint and receipt chain without importing its code.
    Preserve every legacy state item/receipt, reactivate only project text, and
    leave skills, beliefs, organs and permissions quarantined pending review.
    """
    require(store.role == "actor", "migration requires an actor store")
    fields(document, {"checkpoint", "manifest", "manifest_sha256", "receipts", "schema_version", "state", "sha256"})
    digest(pin)
    require(document["schema_version"] == "substrate-product-v1", "unsupported predecessor schema")
    require(document["sha256"] == pin == sha({k: v for k, v in document.items() if k != "sha256"}), "predecessor pin mismatch")
    require(document["manifest_sha256"] == sha(document["manifest"]), "predecessor manifest differs")
    receipts = document["receipts"]
    require(type(receipts) is list and bool(receipts), "predecessor receipt ledger is empty")
    require(receipts[0].get("kind") == "entity_initialized"
            and receipts[0].get("payload") == {"manifest_sha256": sha(document["manifest"])},
            "predecessor initialization does not bind the manifest")
    previous = None
    for index, receipt in enumerate(receipts, 1):
        require(receipt.get("schema_version") == "substrate-product-v1" and receipt.get("sequence") == index
                and receipt.get("previous_sha256") == previous, "predecessor receipt ordering differs")
        require(receipt.get("sha256") == sha({k: v for k, v in receipt.items() if k != "sha256"}), "predecessor receipt hash differs")
        previous = receipt["sha256"]
    expected = {"developmental_state_sha256": sha(document["state"]), "ledger_length": len(receipts),
                "ledger_tail_sha256": previous, "manifest_sha256": sha(document["manifest"]), "schema_version": "substrate-product-v1"}
    require(document["checkpoint"] == expected, "predecessor checkpoint differs")
    entity_id = identifier(name or document["manifest"]["entity_id"])
    require(document["state"].get("entity_id") == document["manifest"]["entity_id"], "predecessor identity mismatch")
    store.create_entity(entity_id, {"self_modify": False, "external_authority": "operator-permit",
                                    "goal": "Continue a verified predecessor through explicit semantic requalification"},
                        lineage={"predecessor_schema": document["schema_version"], "predecessor_pin": pin,
                                 "predecessor_entity": document["manifest"]["entity_id"], "migration": "archival-not-equivalence"})
    changes = [("legacy-manifest", "manifest", document["manifest"]), ("legacy-manifest", "checkpoint", expected)]
    state = document["state"]
    for namespace, value in state.items():
        if isinstance(value, dict):
            for key, item in value.items():
                changes.append(("legacy-state", sha([namespace, key]), {"namespace": namespace, "key": key, "value": item}))
        else:
            changes.append(("legacy-state", sha(namespace), {"namespace": namespace, "value": value}))
    for receipt in receipts:
        changes.append(("legacy-ledger", receipt["sha256"], receipt))
    for start in range(0, len(changes), 64):
        store.commit(entity_id, store.head(entity_id), "legacy-archive-import", {"pin": pin, "offset": start}, changes[start:start + 64])
    cognition = state.get("cognition", {})
    projects = cognition.get("projects", {})
    projected = []
    for key, project in projects.items():
        if type(project) is dict and type(project.get("goal")) is str and type(project.get("next_action")) is str:
            projected.append(("projects", sha(key), {"legacy_name": key, "goal": project["goal"],
                "next_action": project["next_action"], "status": "blocked", "reason": "migration review required",
                "created": project.get("created_at"), "updated": project.get("updated_at")}))
    if projected:
        store.commit(entity_id, store.head(entity_id), "legacy-projects-projected", {"pin": pin}, projected)
    report = {"entity": entity_id, "predecessor_pin": pin, "archived_records": len(changes),
              "projects_preserved": len(projected), "qualified_skills_inherited": 0, "host_permissions_inherited": 0,
              "semantic_equivalence": "not-established", "status": "quarantined-migration"}
    store.commit(entity_id, store.head(entity_id), "migration-complete", report, [("migration", "v1", report)])
    return report


def training_rows(entities: list[Entity], *, allowed_sources: dict[str, str], purpose: str) -> tuple[list[dict], dict]:
    """Only independently admitted DEVELOPMENT records; no qualification/final feedback."""
    require(type(purpose) is str and bool(purpose.strip()), "training purpose must be explicit")
    require(bool(allowed_sources), "source/license allowlist is required")
    for source, license_id in allowed_sources.items():
        digest(source)
        require(type(license_id) is str and bool(license_id.strip()), "source license must be recorded")
    rows, excluded, seen = [], Counter(), set()
    for entity in entities:
        require(entity.store.entity(entity.id)["frozen"], "training extraction needs frozen source entities")
        for experience_id, experience in entity.all("experiences").items():
            if experience["split"] != "train":
                excluded["not-training"] += 1
                continue
            if not entity._certificate_live(experience["certificate"]):
                excluded["authority-expired-or-revoked"] += 1
                continue
            prediction = entity.get("predictions", experience["prediction"])
            observation = entity.get("observations", prediction["observation"])
            source = observation["source"]
            if source not in allowed_sources:
                excluded["source-not-licensed"] += 1
                continue
            identity = sha({"entity": entity.id, "case": experience["case_id"], "target": experience["actual"]})
            if identity in seen:
                excluded["duplicate-entity-case-target"] += 1
                continue
            seen.add(identity)
            # All arms of the same developmental unit stay in the same fitting split.
            fold = "validation" if int(sha({"unit": experience["unit"], "purpose": purpose})[:16], 16) % 5 == 0 else "train"
            rows.append({"schema": "substrate-experience-export-v1", "id": identity, "fold": fold,
                         "group": experience["unit"], "family": experience["family"], "scope": experience["scope"],
                         "input": experience["input"], "target": experience["actual"],
                         "prediction": prediction["expected"], "prediction_correct": experience["correct"],
                         "mechanism": prediction["mechanism"], "control_arm": entity.id.rsplit(".", 1)[-1],
                         "provenance": {"entity": entity.id, "head": entity.store.head(entity.id), "experience": experience_id,
                                        "certificate": sha(experience["certificate"]), "source": source,
                                        "license": allowed_sources[source]}, "purpose": purpose})
    return sorted(rows, key=lambda row: row["id"]), dict(excluded)


def export_training(entities: list[Entity], destination: Path, *, allowed_sources: dict[str, str], purpose: str) -> dict:
    """Deterministic gzip shards, unit-grouped splits and a frozen lineage manifest."""
    require(not destination.exists(), "training export never overwrites a dataset")
    rows, excluded = training_rows(entities, allowed_sources=allowed_sources, purpose=purpose)
    require(bool(rows), "no licensed, currently verified training experience is available")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".training-", dir=destination.parent))
    shards, counts = [], Counter(row["fold"] for row in rows)
    try:
        for offset in range(0, len(rows), 128):
            part = rows[offset:offset + 128]
            raw = b"".join(canonical(row) + b"\n" for row in part)
            packed = gzip.compress(raw, compresslevel=1, mtime=0)
            name = bytes_sha(packed) + ".jsonl.gz"
            path = staging / name
            with path.open("xb") as stream:
                stream.write(packed)
                stream.flush()
                os.fsync(stream.fileno())
            shards.append({"file": name, "packed_sha256": bytes_sha(packed), "raw_sha256": bytes_sha(raw),
                           "records": len(part), "raw_bytes": len(raw), "packed_bytes": len(packed)})
        manifest = {"schema": "substrate-hawking-experience-v1", "purpose": purpose, "shards": shards,
                    "counts": dict(counts), "excluded": excluded, "source_license_allowlist": allowed_sources,
                    "source_entities": [{"entity": entity.id, "head": entity.store.head(entity.id)} for entity in entities],
                    "split_unit": "whole-developmental-history", "heldout_odyssey_feedback_included": False,
                    "ready_for_fitting_review": bool(counts["train"] and counts["validation"]),
                    "models_trained": 0, "capability_claim": None}
        write_json(staging / "manifest.json", manifest)
        os.rename(staging, destination)
        return manifest
    except BaseException:
        import shutil
        shutil.rmtree(staging)
        raise


def hawking_request(*, dataset_manifest: dict, source_model: dict, role_contract: dict, machine_contract: str | None) -> dict:
    """A preparation artifact, not a fictional call into an uninspected Hawking ABI."""
    fields(source_model, {"artifact", "revision", "tokenizer", "representation"})
    for name in ("artifact", "tokenizer"):
        digest(source_model[name])
    require(source_model["representation"] in {"model", "NR"}, "a frozen NX is not a mutable training target")
    fields(role_contract, {"roles", "behavior", "state_schema", "retention_suite", "transfer_suite"})
    for name in ("behavior", "state_schema", "retention_suite", "transfer_suite"):
        digest(role_contract[name])
    if machine_contract is not None:
        digest(machine_contract)
    return {"schema": "substrate-hawking-development-request-v1", "status": "prepared-not-executed",
            "dataset": sha(dataset_manifest), "source_model": clone(source_model), "role_contract": clone(role_contract),
            "stages": ["licensed verified experience", "bounded teacher/no-teacher and memory/compile/train controls",
                       "mutable model or NR fitting", "capability and retention freeze", "Gravity representation search",
                       "accepted NR", "machine-bound NX compilation", "independent organ requalification", "shadow attachment"],
            "target_machine_contract": machine_contract, "updates_frozen_nx_in_place": False,
            "entity_checkpoint_replacement": False, "adapter_status": "site-specific Hawking mapping required"}


# Whole-entity representation and qualified physical-region realization.
WORKING = {"cognitive-state", "attention", "metacognition"}
EXPERIENCE = {"observations", "predictions", "experiences", "failures", "experience-index", "cycle-feedback"}
ORGANIZATION = {"beliefs", "skills", "materials", "projects", "inquiries", "competence", "mechanism-competence", "organs", "skill-index"}


def file_digest(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), "artifact must be an explicit regular file")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _reachable(store, roots: set[str]) -> set[str]:
    found, pending = set(), list(roots)
    while pending:
        key = pending.pop()
        if key not in found:
            found.add(key)
            pending.extend(row[0] for row in store.db.execute("SELECT child FROM refs WHERE parent=?", (key,)))
    return found


def phenotype(entity) -> dict:
    """Snapshot accounting; do not call this linear scan on every thought."""
    store = entity.store
    with store.transaction():
        entry = store.entity(entity.id)
        rows = list(store.db.execute("SELECT namespace,key,object FROM records WHERE entity=?", (entity.id,)))
        current = _reachable(store, {r["object"] for r in rows} | {entry["manifest"]})
        history = _reachable(store, current | {r[0] for r in store.db.execute(
            "SELECT payload FROM events WHERE entity=?", (entity.id,))})
        allocation: dict[str, str] = {entry["manifest"]: "organization"}
        namespace_counts = defaultdict(int)
        for row in sorted(rows, key=lambda r: (r["namespace"], r["key"])):
            ns = row["namespace"]
            category = "working-state" if ns in WORKING else "experience" if ns in EXPERIENCE else "organization" if ns in ORGANIZATION else "other"
            namespace_counts[ns] += 1
            for key in _reachable(store, {row["object"]}):
                allocation.setdefault(key, category)
        totals = defaultdict(lambda: {"objects": 0, "raw_bytes": 0, "packed_bytes": 0})
        for key in sorted(history):
            raw_size, packed = store.db.execute("SELECT raw_size,LENGTH(payload) FROM objects WHERE id=?", (key,)).fetchone()
            category = allocation.get(key, "historical-evidence")
            totals[category]["objects"] += 1
            totals[category]["raw_bytes"] += raw_size
            totals[category]["packed_bytes"] += packed
        organs = entity.all("organs")
        programs = [v for v in entity.all("skills").values() if v["status"] == "qualified"]
        instructions = sum(len(compile_program(v["program"]).instructions) for v in programs)
        return {"schema": "substrate-phenotype-v1", "entity": entity.id, "head": entry["head"],
            "revision": entry["revision"], "categories": dict(totals), "namespaces": dict(namespace_counts),
            "live_objects": len(current), "reachable_objects": len(history),
            "native_instructions": instructions, "qualified_skill_records": len(programs),
            "external_organ_artifacts": sorted({v["artifact"] for v in organs.values()}),
            "external_organ_bytes": None, "resident_process_bytes": None, "device_resident_bytes": None,
            "size_is_intelligence": False, "shared_object_allocation": "once, first namespace in stable order",
            "scope": "entity-reachable compressed payload; excludes SQLite indexes/WAL, runtime, model files and measured residency"}


def growth(before: dict, after: dict) -> dict:
    require(before["schema"] == after["schema"] == "substrate-phenotype-v1"
            and before["entity"] == after["entity"], "growth requires the same entity and schema")
    require(before["revision"] <= after["revision"], "growth snapshots are reversed")
    categories = set(before["categories"]) | set(after["categories"])
    delta = {name: {metric: after["categories"].get(name, {}).get(metric, 0) - before["categories"].get(name, {}).get(metric, 0)
                    for metric in ("objects", "raw_bytes", "packed_bytes")} for name in sorted(categories)}
    return {"entity": before["entity"], "from": before["head"], "to": after["head"], "delta": delta,
            "native_instruction_delta": after["native_instructions"] - before["native_instructions"],
            "capability_delta": "requires-matched-independent-evaluation", "lineage_continuity": "verify-event-chain-separately",
            "interpretation": "growth, plateau or shrinkage can coexist with improved capability; bytes alone do not decide"}


def safetensors_manifest(path: Path, *, expected: str | None = None) -> dict:
    """Read bounded metadata and stream a digest; never instantiate arbitrary model objects."""
    require(path.is_file() and not path.is_symlink(), "tensor artifact must be a regular file")
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        require(2 <= length <= min(16 * 1024 * 1024, size - 8), "safetensors header bound")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, "duplicate safetensors header key")
                result[key] = value
            return result
        raw_header = stream.read(length)
        header = json.loads(raw_header, object_pairs_hook=unique)
        hasher, payload_bytes = hashlib.sha256(prefix + raw_header), 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
            payload_bytes += len(block)
        require(size == 8 + length + payload_bytes, "tensor file size changed during inspection")
        identity = hasher.hexdigest()
    require(type(header) is dict, "safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    require(type(metadata) is dict and all(type(k) is str and type(v) is str for k, v in metadata.items()), "invalid tensor metadata")
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
              "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
              "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1}
    rows, offsets = [], []
    for name, value in header.items():
        fields(value, {"dtype", "shape", "data_offsets"})
        require(value["dtype"] in widths, "unsupported tensor dtype; use a separately qualified format adapter")
        require(type(value["shape"]) is list and len(value["shape"]) <= 32, "invalid tensor shape")
        count = math.prod(integer(x, 0, 2**40) for x in value["shape"])
        require(count <= 2**63, "tensor element bound")
        require(type(value["data_offsets"]) is list and len(value["data_offsets"]) == 2, "invalid data offsets")
        low, high = value["data_offsets"]
        integer(low, 0, size - 8 - length)
        integer(high, low, size - 8 - length)
        require(high - low == count * widths[value["dtype"]], "tensor shape/dtype/byte size differs")
        if high > low:
            offsets.append((low, high))
        rows.append({"name": name, "dtype": value["dtype"], "shape": value["shape"], "elements": count, "bytes": high - low})
    cursor = 0
    for low, high in sorted(offsets):
        require(low == cursor, "tensor buffers overlap or contain gaps")
        cursor = high
    require(cursor == size - 8 - length, "unaccounted tensor payload")
    require(expected is None or identity == digest(expected), "tensor artifact pin differs")
    return {"schema": "substrate-tensor-inventory-v1", "artifact": identity, "file_bytes": size,
            "elements": sum(r["elements"] for r in rows), "tensors": sorted(rows, key=lambda r: r["name"]),
            "metadata": metadata, "semantics": "tensor elements are not automatically neural parameters or intelligence"}


def export_state_tensors(entity, path: Path) -> dict:
    """Optional official safetensors writer for actual learned sufficient statistics.

    JSON/event state remains authoritative. These arrays are not relabeled as
    neural weights; no random placeholder tensors or fictional brain are emitted.
    """
    require(entity.store.entity(entity.id)["frozen"], "tensor export requires a frozen checkpoint")
    require(not path.exists(), "tensor export never overwrites")
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise Refused("optional reviewed numpy/safetensors dependencies are required for tensor export") from exc
    attention = entity.all("attention")
    keys = sorted((family, channel) for family, channels in attention.items() for channel in channels)
    competence = entity.all("competence")
    families = sorted(competence)
    tensors = {
        "attention_counts": np.asarray([[attention[f][c]["attempts"], attention[f][c]["wins"]] for f, c in keys], dtype=np.int64).reshape((-1, 2)),
        "competence_counts": np.asarray([[competence[f]["attempts"], competence[f]["wins"]] for f in families], dtype=np.int64).reshape((-1, 2)),
        "calibration_error": np.asarray([competence[f]["brier_sum"] for f in families], dtype=np.float64),
    }
    require(bool(keys or families), "no learned sufficient statistics to export")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tensors-", dir=path.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata={"entity": entity.id, "head": entity.store.head(entity.id),
            "role": "learned-sufficient-statistics-not-neural-weights", "families": json.dumps(families),
            "attention_keys": json.dumps(keys)})
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return safetensors_manifest(path)


def portable_organization(entity, destination: Path, *, runtime: str, assets: dict[str, Path] | None = None,
                          tensor_state: bool = False) -> dict:
    """Atomic whole-entity package, with explicit external dependency closure.

    The NR-like representation includes all durable records and event history;
    it is not limited to a neural weight checkpoint or to a selected kernel.
    Site Hawking adapters must map this declared contract to their actual ABI.
    """
    require(entity.store.entity(entity.id)["frozen"], "portable organization requires a frozen checkpoint")
    digest(runtime)
    require(not destination.exists(), "portable export never overwrites")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".organization-", dir=destination.parent))
    try:
        snapshot = phenotype(entity)
        pin = entity.store.export_entity(entity.id, staging / "entity.bundle")
        records = [dict(r) for r in entity.store.db.execute(
            "SELECT namespace,key,object FROM records WHERE entity=? ORDER BY namespace,key", (entity.id,))]
        inventory = []
        for expected, path in sorted((assets or {}).items()):
            digest(expected)
            require(file_digest(path) == expected, "external asset digest differs")
            target = staging / "assets" / expected
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            require(file_digest(target) == expected, "copied asset changed during export")
            inventory.append({"artifact": expected, "file": "assets/" + expected, "bytes": target.stat().st_size})
        tensors = export_state_tensors(entity, staging / "state.safetensors") if tensor_state else None
        supplied = {a["artifact"] for a in inventory}
        missing = sorted(set(snapshot["external_organ_artifacts"]) - supplied)
        document = {"schema": "substrate-organization-nr-v1", "entity": entity.id, "head": snapshot["head"],
            "bundle": {"file": "entity.bundle", "manifest_pin": pin, "sha256": file_digest(staging / "entity.bundle")},
            "state_index": records, "runtime_contract": runtime, "assets": inventory, "state_tensors": tensors,
            "mutable_state_schema": "substrate-v2-records-and-events", "organ_artifacts_missing": missing,
            "neural_transitive_closure": "site-adapter-must-verify-tokenizer-config-kernels-and-state",
            "manifestation": "portable-hybrid-cognitive-organization", "hawking_abi_qualified": False,
            "loadable_without_external_assets": not snapshot["external_organ_artifacts"], "phenotype": snapshot,
            "contains_credentials": False, "full_entity_included": True}
        require(entity.store.head(entity.id) == snapshot["head"] and entity.store.entity(entity.id)["frozen"],
                "entity changed during export; abandon the mixed snapshot")
        write_json(staging / "organization.nr.json", document)
        os.rename(staging, destination)
        return {"directory": str(destination), "nr_pin": sha(document), "missing_organs": missing,
                "whole_entity_preserved": True, "hawking_native_execution": "requires-qualified-site-adapter"}
    except BaseException:
        shutil.rmtree(staging)
        raise


def lower_native_regions(entity, *, nr: dict, nr_pin: str, runtime: str, machine: str) -> dict:
    """Produce executable closed-DAG regions; not a fictitious whole-model compiler."""
    require(entity.store.entity(entity.id)["frozen"], "lowering requires a frozen source")
    for value in (nr_pin, runtime, machine):
        digest(value)
    require(sha(nr) == nr_pin and nr["schema"] == "substrate-organization-nr-v1"
            and nr["entity"] == entity.id and nr["head"] == entity.store.head(entity.id)
            and nr["runtime_contract"] == runtime and nr["full_entity_included"] is True,
            "native image must bind the actual whole-organization source and runtime")
    entity.store.audit(entity.id)
    require(machine == entity.host, "requalify source skills on the target host before local lowering")
    regions = {}
    for key, row in entity.all("skills").items():
        if entity.skill_ready(row, row["family"], row["scope"]):
            program = compile_program(row["program"])
            regions[key] = {"ir": program.document, "instructions": clone(program.instructions),
                            "output": program.output, "output_type": program.output_type,
                            "scope": row["scope"], "family": row["family"], "source_qualification": sha(row["certificate"])}
    require(bool(regions), "no currently qualified native regions exist")
    return {"schema": "substrate-native-image-v1", "nr": nr_pin, "entity": entity.id, "head": entity.store.head(entity.id),
            "machine": machine, "runtime": runtime, "regions": regions, "status": "candidate",
            "external_mutable_state": "entity.bundle continuation under a separate state contract",
            "scope": "closed-program-regions; full entity remains in the portable organization",
            "hawking_nx_abi": "unqualified"}


class NativeImage:
    """Load once, execute qualified precompiled regions without a model or rediscovery."""
    def __init__(self, document: dict, *, pin: str, runtime: str, machine: str, trust, certificate: dict):
        require(sha(document) == digest(pin) and document["schema"] == "substrate-native-image-v1", "native image pin/schema differs")
        require(document["runtime"] == digest(runtime) and document["machine"] == digest(machine), "native image host/runtime differs")
        require(type(document["regions"]) is dict and 1 <= len(document["regions"]) <= 4096, "native region count bound")
        self.document, self.trust, self.certificate, self.pin = clone(document), trust, certificate, pin
        self.programs = {}
        self._qualified()
        for key, row in document["regions"].items():
            program = compile_program(row["ir"])
            require(clone(program.instructions) == row["instructions"] and program.output == row["output"]
                    and program.output_type == row["output_type"], "native image lowering differs from its source IR")
            self.programs[key] = Program(program.document, tuple((op, tuple(refs), literal) for op, refs, literal in row["instructions"]),
                                         row["output"], row["output_type"], program.digest)

    def _qualified(self) -> None:
        findings = self.trust.verify(self.certificate, purpose="native-image", domain="cognition",
            subject={"image": self.pin, "nr": self.document["nr"], "machine": self.document["machine"], "runtime": self.document["runtime"]},
            producer="native-lowerer")
        require(findings.get("executed") is True and findings.get("fidelity_passed") is True
                and findings.get("scope") == sha(sorted(self.document["regions"])), "native regions not independently qualified")

    def run(self, skill: str, inputs: dict, *, family: str, scope: dict):
        self._qualified()
        row = self.document["regions"].get(skill)
        require(row and row["family"] == family and scope_matches(row["scope"], scope), "native region outside qualified scope")
        return self.programs[skill].run(inputs)


def restore_organization(store: Store, directory: Path, *, nr_pin: str, runtime: str) -> dict:
    """Verify a complete pinned portable package, then restore its frozen entity.

    Credentials, writer authority and new-host organ qualification are never
    inherited. External model files remain in the package; this does not launch
    an organ, execute a native image or claim a Hawking ABI was matched.
    """
    require(directory.is_dir() and not directory.is_symlink(), "organization directory must be explicit")
    document = read_json(directory / "organization.nr.json")
    require(sha(document) == digest(nr_pin) and document["schema"] == "substrate-organization-nr-v1",
            "portable organization pin/schema differs")
    require(document["runtime_contract"] == digest(runtime), "runtime differs; explicit migration is required")
    require(document["full_entity_included"] is True and document["contains_credentials"] is False,
            "incomplete or authority-bearing package")
    bundle = directory / "entity.bundle"
    require(document["bundle"]["file"] == "entity.bundle" and file_digest(bundle) == document["bundle"]["sha256"],
            "entity bundle differs")
    for row in document["assets"]:
        expected = digest(row["artifact"])
        require(row["file"] == "assets/" + expected and not (directory / "assets").is_symlink(), "unsafe asset path")
        require(file_digest(directory / row["file"]) == expected
                and (directory / row["file"]).stat().st_size == row["bytes"], "asset payload differs")
    if document["state_tensors"] is not None:
        require(safetensors_manifest(directory / "state.safetensors", expected=document["state_tensors"]["artifact"])
                == document["state_tensors"], "learned tensor state differs")
    import zipfile
    with zipfile.ZipFile(bundle) as archive:
        require(archive.getinfo("manifest.json").file_size <= 64 * 1024 * 1024, "entity manifest bound")
        manifest = json.loads(archive.read("manifest.json"))
    entry = manifest["entity"]
    require(entry["id"] == document["entity"] and entry["head"] == document["head"] and entry["frozen"],
            "portable wrapper and entity checkpoint differ")
    projected = [{key: row[key] for key in ("namespace", "key", "object")} for row in manifest["records"]]
    require(projected == document["state_index"], "portable state index differs")
    restored = store.restore_entity(bundle, manifest_pin=document["bundle"]["manifest_pin"])
    return {"entity": restored, "nr": nr_pin, "head": document["head"], "frozen": True,
            "organ_artifacts_missing": document["organ_artifacts_missing"], "host_authority_inherited": False,
            "hawking_abi_qualified": False}
