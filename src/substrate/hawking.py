"""Pinned Hawking imports and a whole-system inheritance boundary.

Hawking's NR deny-list, legacy seal encoding, NX genome comparison, Nova
lineage fields and Bridge surface are adapted from Joshua Hicks' MIT-licensed
source at UPSTREAM_COMMIT. See HAWKING_LICENSE.txt. No benchmark constants,
hard-coded GPU measurements, model-specific training goals or campaign launchers
are imported. A passed schema check is never an execution qualification.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .core import (canonical, clone, digest, fields, identifier, integer, numeric, parse,
                   read_json, regular_bytes, require, sha, write_json)
from .organs import JsonTransport

UPSTREAM_REPOSITORY = "joshuahickscorp/hawking"
UPSTREAM_COMMIT = "6a86a5cd49fbe9211ec11e133e9d70a4e2044238"
UPSTREAM_IMPORTS = {
    "tools/nr_container.py": "10007c2d679ec560a17d8b1a14321db5fc8d560e",
    "tools/nx_genome.py": "00228167d694b44c0f76e71d36aab451b8e29cd8",
    "tools/future/gravity_nova_lineage.py": "8dffa027ec7db9c005fbf6c0d55952fd5f523c23",
    "crates/hawking-adapters/src/bridge_surface.rs": "09c3e0dbc788ca72eefacf15dac35cef907d7647",
    "crates/hawking-serve/src/http.rs": "1a95da64dfb90ce51f44de7f1888f672a03cb031",
}
# The final architecture deliberately requests every relevant Hawking subsystem.
# There is no fixed organ-only port limit and no permanently required external
# Hawking service. A mature implementation can be inherited in its entirety;
# source, runtime, model, transport and state closures must remain inspectable.
INHERITANCE = {
    "scope": "all-relevant-source-runtime-and-research-machinery",
    "whole_system_eligible": True, "permanent_partial_port_cap": None,
    "external_hawking_service_required": False, "production_parent_mutable": False,
    "capabilities": ["inference", "perception", "training", "representation", "compilation", "execution",
                     "device-discovery", "memory", "transport", "state", "qualification", "instrumentation"],
    "future_bindings": "explicit-versioned-capability-negotiation-not-assumed-ABI",
    "transition_contract": "state,observation,resources -> action,state,learning-consequences",
}
MACHINE_SPECIFIC = {
    "kernel", "kernels", "kernel_name", "shader", "metallib", "threadgroup",
    "threadgroup_size", "tg_size", "grid", "dispatch", "dispatches", "device",
    "device_id", "machine_genome", "gpu", "residency_plan", "cache_plan",
    "schedule", "ps_per_element", "gb_s", "tps", "token_ns", "occupancy",
    "register_pressure", "simd_width", "r_tiling", "k_amortization",
}
GENOME_FIELDS = ("chipset", "gpu_cores", "unified_memory_bytes", "metal_family", "measured_roof_gb_s")
BRIDGE_SURFACE = {
    "POST /v1/chat/completions": "live", "POST /v1/completions": "live", "GET /v1/models": "live",
    "GET /healthz": "live", "GET /metrics": "live", "POST /v1/embeddings": "partial",
    "POST /v1/hawking/tokens": "live", "POST /v1/hawking/generate": "live",
    "GET /v1/hawking/context": "live", "GET /v1/hawking/surface": "live",
    "POST /v1/responses": "not_implemented", "POST /v1/messages": "not_implemented",
    "MCP": "partial", "ACP": "not_implemented", "SDK Transport -> HCLI": "not_implemented",
}


def seal_payload_sha256_v1(document):
    """Preserve Hawking's actual legacy sorted/default-separator JSON byte rule."""
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def walk_keys(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{path}.{key}" if path else key
            yield key, location
            yield from walk_keys(child, location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_keys(child, f"{path}[{index}]")


def validate_nr(document: dict) -> dict:
    require(type(document) is dict, "NR must be an object")
    require(document.get("nr_kind") == "hawking.nos.noetic_representation" and document.get("nr_version") == "1.0.0", "unknown Hawking NR revision")
    for key, path in walk_keys(document):
        require(key.lower() not in MACHINE_SPECIFIC, f"machine-specific NR field: {path}")
    for key in ("semantic_provenance", "representation", "kernel_requirements"):
        require(key in document, "missing portable NR section")
    provenance = document["semantic_provenance"]
    require(type(provenance) is dict and all(k in provenance for k in ("parent_model", "parent_revision", "parameter_count")), "NR parent provenance incomplete")
    require(type(document["representation"]) is dict and type(document["kernel_requirements"]) is list, "NR section type differs")
    for requirement in document["kernel_requirements"]:
        require(type(requirement) is dict and not {"implementation", "kernel"} & requirement.keys(), "NR requirement contains a machine binding")
    return {"schema_valid": True, "legacy_seal": seal_payload_sha256_v1(document), "execution_qualified": False}


def genome_digest(genome: dict) -> str:
    require(all(key in genome for key in GENOME_FIELDS), "incomplete measured machine genome")
    numeric(genome["measured_roof_gb_s"], .000001)
    integer(genome["gpu_cores"], 1)
    integer(genome["unified_memory_bytes"], 1)
    return seal_payload_sha256_v1({key: genome[key] for key in GENOME_FIELDS})


def validate_nx(document: dict, *, nr_bytes: bytes, machine: dict) -> dict:
    """Validate real G103/G104 wire fields, without reproducing one host's measurements."""
    nr = parse(nr_bytes)
    validate_nr(nr)
    require(document.get("nx_kind") == "hawking.nos.noetic_executable_genome" and document.get("nx_version") == "1.0.0", "unknown Hawking NX revision")
    current = genome_digest(machine)
    require(machine.get("genome_digest") == current, "supplied machine genome digest is inconsistent")
    compiled = document["compiled_for_machine_genome"]
    require(compiled.get("genome_digest") == genome_digest(compiled) == current, "NX machine genome mismatch")
    lowered = document["lowers_nr"]
    nr_hash = hashlib.sha256(nr_bytes).hexdigest()
    require(lowered.get("nr_content_sha256") == nr_hash and lowered.get("nr_kind") == nr["nr_kind"], "NX does not bind these exact NR bytes")
    expected = [{k: v for k, v in row.items() if k != "note"} for row in nr["kernel_requirements"]]
    require(lowered.get("requirements_the_nx_satisfies") == expected, "NX requirement closure differs")
    return {"schema_valid": True, "nr_content_sha256": nr_hash, "machine": current,
            "execution_qualified": False, "runtime_payloads_verified": False}


def validate_nova(lineage: dict, *, parent_identity: str, descendant: str) -> dict:
    """Port the pure upstream Nova admission fields; avoid model-specific defaults."""
    require(lineage.get("schema") == "hawking.gravity.nova_lineage.v1", "unknown Nova lineage")
    parent, child = lineage["parent"], lineage["descendant"]
    require(parent["identity"] == parent_identity and child.get("identity"), "Nova lineage identity differs")
    digest(parent["artifact_hash"])
    require(digest(child["artifact_hash"]) == digest(descendant) != parent["artifact_hash"], "Nova must create a new descendant")
    require(bool(lineage.get("objective")), "Nova objective missing")
    change = lineage["transformation"]
    for key in ("kind", "changed_tensors", "train_set", "freeze_set", "data_class", "teacher", "optimization_objective", "representation_constraints"):
        require(change.get(key) is not None and change.get(key) != "" and change.get(key) != [], "Nova transformation incomplete")
    require(set(change["train_set"]).isdisjoint(change["freeze_set"]), "Nova train/freeze sets overlap")
    for axis in ("capability", "epistemic", "authorization", "physical", "destructive_controls"):
        row = lineage["measurements"][axis]
        require(all(key in row and row[key] is not None for key in ("before", "after", "receipt")), "Nova evidence axis missing")
    rollback = lineage["rollback"]
    require(rollback.get("reversible") is True and rollback.get("parent_immutable") is True and rollback.get("recipe"), "Nova rollback closure missing")
    return {"schema_valid": True, "descendant": descendant, "promotion_authorized": False}


class HawkingClient:
    """Read/query the real inspected Bridge, not an invented training REST API."""
    def __init__(self, base: str, *, token_env: str | None = None, timeout: float = 60):
        url = urlsplit(base)
        require(url.hostname in {"localhost", "127.0.0.1", "::1"} and url.path in {"", "/"}, "Hawking discovery is loopback-only")
        self.base, self.token_env, self.timeout = base.rstrip("/"), token_env, timeout

    def surface(self) -> dict:
        document = JsonTransport(self.base + "/v1/hawking/surface", token_env=self.token_env, timeout=self.timeout).get()
        require(document.get("schema") == "hawking.bridge.surface.v1" and type(document.get("endpoints")) is list, "unknown Bridge capability response")
        return {"document": document, "declared_not_qualified": True, "reviewed_source": UPSTREAM_COMMIT}

    def query(self, endpoint: str, payload: dict, *, surface: dict, allow_partial: bool = False) -> dict:
        advertised = {row["endpoint"]: row["status"] for row in surface["document"]["endpoints"]}
        require(endpoint in BRIDGE_SURFACE and endpoint.startswith("POST "), "unknown inspected query endpoint")
        require(advertised.get(endpoint) == "live" or (allow_partial and advertised.get(endpoint) == "partial"), "endpoint unavailable")
        require(BRIDGE_SURFACE[endpoint] != "not_implemented", "source does not implement this endpoint")
        return JsonTransport(self.base + endpoint.split(" ", 1)[1], token_env=self.token_env, timeout=self.timeout).post(payload)


def port_hawking_source(root: Path, destination: Path, *, commit: str = UPSTREAM_COMMIT,
                       max_bytes: int = 4 * 1024**3, prefixes: tuple[str, ...] = ()) -> dict:
    """Stage pinned committed source without importing it or building anything.

    With no prefix filter, every tracked ordinary file is eligible: this is not
    an organ-only donor mechanism. Gitlinks/symlinks stop the operation rather
    than silently producing an incomplete source closure. Untracked secrets and
    the donor's live working state are never copied. The result is quarantined.
    """
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "exact Git revision required")
    integer(max_bytes, 1024)
    require(root.is_dir() and not destination.exists(), "existing donor and fresh destination required")
    require(root.resolve() not in destination.resolve().parents, "port destination cannot be inside donor")
    env = {"PATH": os.environ.get("PATH", ""), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0",
           "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0"}

    def git(*args):
        result = subprocess.run(["git", "--no-pager", "-C", str(root), *args], env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=120, check=False)
        require(result.returncode == 0, "cannot resolve pinned donor source")
        return result.stdout

    require(git("rev-parse", commit + "^{commit}").decode().strip() == commit, "donor commit unavailable")
    listing = git("ls-tree", "-r", "-z", "--full-tree", commit)
    entries, total = [], 0
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        meta, name = raw.split(b"\t", 1)
        mode, kind, blob = meta.decode().split()
        path = PurePosixPath(name.decode("utf-8"))
        require(not path.is_absolute() and ".." not in path.parts and ".git" not in path.parts, "unsafe donor path")
        if prefixes and not any(str(path) == p or str(path).startswith(p.rstrip("/") + "/") for p in prefixes) and str(path) not in {"LICENSE", "Cargo.toml", "Cargo.lock", "pyproject.toml"}:
            continue
        require(mode in {"100644", "100755"} and kind == "blob", "donor link/submodule requires explicit closure resolution")
        entries.append((str(path), blob, mode))
    require(entries, "empty source selection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hawking-port-", dir=destination.parent))
    inventory = {}
    try:
        for name, blob, mode in entries:
            size = int(git("cat-file", "-s", blob))
            total += size
            require(total <= max_bytes, "donor port byte budget exhausted")
            content = git("cat-file", "blob", blob)
            require(len(content) == size and hashlib.sha1(b"blob " + str(size).encode() + b"\0" + content).hexdigest() == blob, "Git payload identity differs")
            path = staging / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(int(mode, 8) & 0o777)
            inventory[name] = {"git_blob": blob, "mode": mode, "sha256": hashlib.sha256(content).hexdigest(), "bytes": size}
        result = {"schema": "substrate-hawking-source-port-v1", "repository": UPSTREAM_REPOSITORY,
                  "commit": commit, "source_inventory": inventory, "bytes": total,
                  "scope": "entire-committed-tree" if not prefixes else "operator-selected-not-build-closure",
                  "inheritance": INHERITANCE, "status": "quarantined-source-not-built-or-qualified",
                  "external_build_dependencies_included": False, "live_worktree_copied": False}
        write_json(staging / "port.json", result)
        os.rename(staging, destination)
        return {"commit": commit, "files": len(inventory), "bytes": total, "manifest": sha(result), "qualified": False}
    except BaseException:
        shutil.rmtree(staging)
        raise


class CapabilityPort:
    """Substrate-side versioned extension ABI; future Hawking services must map to it.

    No operation is claimed to exist in today's Hawking just because it appears
    in INHERITANCE. A supplied independently qualified provider can expose any
    needed subsystem. Transport credentials stay host-local, outside the entity.
    """
    def __init__(self, entity, binding: dict, certificate: dict):
        fields(binding, {"schema", "id", "artifact", "endpoint", "operations", "host", "state_schema", "token_env"})
        require(binding["schema"] == "substrate-capability-port-v1" and binding["host"] == entity.host, "provider host/ABI mismatch")
        digest(binding["artifact"])
        digest(binding["state_schema"])
        identifier(binding["id"])
        require(type(binding["operations"]) is dict and binding["operations"], "provider has no operations")
        for name, contract in binding["operations"].items():
            identifier(name)
            fields(contract, {"input", "output", "effects", "max_tokens", "timeout"})
            integer(contract["max_tokens"], 1, 1000000)
            numeric(contract["timeout"], .1, 3600)
            digest(contract["input"])
            digest(contract["output"])
            require(contract["effects"] in {"read", "propose", "mutate-external"}, "unknown provider effect")
        findings = entity.trust.verify(certificate, purpose="capability-port", domain="cognition", subject=binding, producer=binding["id"])
        require(findings.get("executed") is True and findings.get("passed") is True, "provider declarations are not qualifications")
        self.entity, self.binding, self.certificate = entity, clone(binding), clone(certificate)
        JsonTransport(binding["endpoint"], token_env=binding["token_env"])

    def invoke(self, job: dict, operation: str, payload: dict, permit: dict, budget) -> dict:
        b, e = self.binding, self.entity
        require(e._certificate_live(self.certificate), "provider qualification expired")
        require(operation in b["operations"], "provider operation not advertised")
        request = {"schema": "substrate-capability-call-v1", "provider": b["artifact"], "operation": operation,
                   "contract": b["operations"][operation], "state_schema": b["state_schema"],
                   "entity": e.id, "head": e.store.head(e.id), "payload": clone(payload)}
        findings = e.trust.verify(permit, purpose="capability-call", domain="cognition", subject=request, producer="substrate-dispatch")
        require(findings.get("approved") is True, "external capability action not approved")
        limits = b["operations"][operation]
        require(limits["timeout"] <= budget.wall_seconds, "operation exceeds supplied wall budget")
        require(time.time() + limits["timeout"] < min(permit["expires"], self.certificate["expires"]), "permission does not cover deadline")
        effect = e.store.reserve_effect(job, request, max_tokens=limits["max_tokens"], call_limit=budget.calls, token_limit=budget.tokens)
        e.store.sent_effect(job, effect)
        response = JsonTransport(b["endpoint"], token_env=b["token_env"], timeout=limits["timeout"]).post(request)
        fields(response, {"schema", "request", "provider", "state_schema", "output", "cost"})
        require(response["schema"] == "substrate-capability-result-v1" and response["request"] == sha(request)
                and response["provider"] == b["artifact"] and response["state_schema"] == b["state_schema"], "provider response identity differs")
        cost = response["cost"]
        require(type(cost) is dict, "provider cost must distinguish unknown usage")
        actual = integer(cost["tokens"], 0, limits["max_tokens"]) if cost.get("tokens") is not None else None
        e.store.settle_effect(job, effect, response, actual_tokens=actual)
        return {**response, "independently_verified_output": False, "output_schema_validation": "provider-contract-not-locally-resolved"}


def organ_candidate(nr_path: Path, specification: dict, *, nx_path: Path | None = None,
                    machine: dict | None = None) -> dict:
    """Map inspected Hawking file contracts into the existing Substrate organ API.

    The caller supplies complete-artifact and behavioral identities. Checking
    these metadata documents cannot prove that a running service loaded those
    payloads. The normal independent organ-binding gate is still mandatory.
    """
    from dataclasses import asdict
    from .organs import Organ
    fields(specification, {"id", "model", "artifact", "nr_artifact", "behavior_contract", "state_schema",
                            "tokenizer", "endpoint", "roles"}, {"token_env", "qualification"})
    raw = regular_bytes(nr_path, 4 * 1024**2)
    validate_nr(parse(raw))
    nr_pin = hashlib.sha256(raw).hexdigest()
    nr = {"schema": "substrate-hawking-nr-bridge-v1", "artifact": digest(specification["nr_artifact"]),
          "behavior_contract": digest(specification["behavior_contract"]), "state_schema": digest(specification["state_schema"]),
          "tokenizer": digest(specification["tokenizer"]),
          "lineage": [{"nr_content_sha256": nr_pin, "import_contract_source": UPSTREAM_COMMIT}]}
    nx, nx_pin = None, None
    if nx_path is not None:
        document_bytes = regular_bytes(nx_path, 4 * 1024**2)
        document = parse(document_bytes)
        require(machine is not None, "NX import needs an observed machine genome")
        checked = validate_nx(document, nr_bytes=raw, machine=machine)
        nx_pin = hashlib.sha256(document_bytes).hexdigest()
        nx = {"schema": "substrate-hawking-nx-bridge-v1", "artifact": digest(specification["artifact"]),
              "nr": nr["artifact"], "machine_contract": checked["machine"], "behavior_contract": nr["behavior_contract"],
              "state_schema": nr["state_schema"], "execution_plan": sha(document),
              "qualification": digest(specification.get("qualification"))}
    organ = Organ(specification["id"], specification["model"], specification["artifact"], specification["endpoint"],
                  tuple(specification["roles"]), representation="nx" if nx else "nr", nr=nr, nx=nx,
                  token_env=specification.get("token_env"))
    return {"organ": asdict(organ), "source_nr_file": nr_pin, "source_nx_file": nx_pin,
            "status": "metadata-candidate-needs-independent-organ-binding", "complete_payload_verified": False,
            "server_artifact_verified": False, "execution_qualified": False}
