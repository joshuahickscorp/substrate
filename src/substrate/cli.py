"""One operational interface. Plan and preflight never launch an experiment.

Only explicitly selected worker/service commands perform external execution.
No command downloads models, starts tests or manufactures scientific gate passes.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .cognition import Entity
from .core import (Refused, Trust, attest, canonical, fields, integer, parse, read_json, regular_bytes,
                   require, sha, source_digest, write_json, inspect_body, digest)
from .odyssey import Gate, Runner, draft_plan, initialize, preflight, run_subject, status, acquisition_report
from .organs import OracleClient
from .sandbox import Executor, SandboxConfig
from .statistics import analyze, read_score_directory
from .store import Store
from .worlds import Evaluator
from .handoff import (export_training, migrate_v1, hawking_request, phenotype, growth, safetensors_manifest,
                      portable_organization, lower_native_regions, restore_organization)
from .research import program_plan, OdysseyLedger
from .local import configure_local, execute_local


def private_key(path: Path) -> Ed25519PrivateKey:
    info = path.lstat()
    require(info.st_mode & 0o077 == 0, "private key permissions must exclude group and other users")
    raw = regular_bytes(path, 16384)
    key = serialization.load_pem_private_key(raw, password=None)
    require(isinstance(key, Ed25519PrivateKey), "Ed25519 private key required")
    return key


def serve_evaluator(evaluator: Evaluator, *, gate: Gate, port: int, token_env: str) -> None:
    """Authenticated loopback service; deploy behind reviewed TLS for a separate host.

    A distinct process alone is not OS isolation: use a different OS identity or
    machine, and deny actor access to the evaluator key, database and seed paths.
    """
    integer(port, 1024, 65535)
    token = os.environ.get(token_env)
    require(token is not None and len(token) >= 32, "evaluator bearer token must have at least 32 characters")
    gate.check()
    started = time.monotonic()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return  # Never log prompts, tokens, private labels or Authorization headers.

        def do_POST(self):
            code, response = 200, None
            try:
                require(time.monotonic() - started < gate.plan["budget"]["wall_seconds"], "evaluator wall budget exhausted")
                require(time.time() < gate.permit["expires"], "run permit expired")
                require(self.path == "/" and self.headers.get("Transfer-Encoding") is None, "unsupported request framing")
                require(hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token), "unauthorized")
                length = self.headers.get("Content-Length", "")
                require(length.isdigit() and 0 < int(length) <= 4 * 1024 * 1024, "request size bound")
                request = parse(self.rfile.read(int(length)))
                fields(request, {"method", "payload"})
                response = evaluator.dispatch(request["method"], request["payload"])
            except (Refused, KeyError, TypeError, ValueError):
                code, response = 400, {"error": "contract-refused", "details": "inspect evaluator-owned audit state"}
            except Exception:
                code, response = 500, {"error": "evaluator-failure", "details": "operator review required"}
            raw = canonical(response)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

    class Server(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(15)
            return connection, address

    with Server(("127.0.0.1", port), Handler) as server:
        server.timeout = 1
        while time.monotonic() - started < gate.plan["budget"]["wall_seconds"] and time.time() < gate.permit["expires"]:
            server.handle_request()


def common(parser, *, permit: bool = False):
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--trust", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    if permit:
        parser.add_argument("--permit", type=Path, required=True)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="substrate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("plan", help="emit a draft; no network or experiment")
    draft.add_argument("--kind", choices=("pilot", "main"), default="pilot")
    draft.add_argument("--output", type=Path, required=True)
    body = sub.add_parser("body", help="inspect host facts without running models")
    body.add_argument("--machine-tag", required=True)
    body.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("preflight", help="inspect source-bound readiness without contacting services")
    common(check)
    check.add_argument("--output", type=Path)
    approval = sub.add_parser("approve-run", help="explicit operator approval only; never signs scientific gates")
    common(approval)
    approval.add_argument("--key", type=Path, required=True)
    approval.add_argument("--signer", required=True)
    approval.add_argument("--ttl", type=float, default=3600)
    approval.add_argument("--output", type=Path, required=True)
    for command in ("init", "worker", "fetch-scores"):
        child = sub.add_parser(command)
        common(child, permit=True)
        child.add_argument("--db", type=Path, required=True)
        if command == "worker":
            child.add_argument("--owner", required=True)
            child.add_argument("--max-jobs", type=int, default=1)
        if command == "fetch-scores":
            child.add_argument("--output", type=Path, required=True)
    server = sub.add_parser("serve-evaluator", help="explicit independent evaluator process; requires authorization")
    common(server, permit=True)
    server.add_argument("--db", type=Path, required=True)
    server.add_argument("--key", type=Path, required=True)
    server.add_argument("--seed", type=Path, required=True)
    server.add_argument("--signer", required=True)
    server.add_argument("--token-env", default="SUBSTRATE_ORACLE_TOKEN")
    server.add_argument("--port", type=int, default=8091)
    for command in ("status", "audit", "bundle"):
        child = sub.add_parser(command)
        child.add_argument("--db", type=Path, required=True)
        if command != "status":
            child.add_argument("--entity", required=True)
        if command == "bundle":
            child.add_argument("--output", type=Path, required=True)
    for command in ("restore", "migrate-v1"):
        child = sub.add_parser(command)
        child.add_argument("--db", type=Path, required=True)
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--pin", required=True)
    analysis = sub.add_parser("analyze", help="verify a complete signed score bundle and analyze histories")
    analysis.add_argument("--plan", type=Path, required=True)
    analysis.add_argument("--trust", type=Path, required=True)
    analysis.add_argument("--scores", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--db", type=Path, help="include acquisition and qualification accounting")
    training = sub.add_parser("prepare-training", help="export licensed verified experience; does not train models")
    training.add_argument("--db", type=Path, required=True)
    training.add_argument("--trust", type=Path, required=True)
    training.add_argument("--host", required=True)
    training.add_argument("--entity", action="append", required=True)
    training.add_argument("--sources", type=Path, required=True)
    training.add_argument("--purpose", required=True)
    training.add_argument("--output", type=Path, required=True)
    request = sub.add_parser("hawking-request", help="prepare a reviewed integration request; never invokes training")
    request.add_argument("--dataset", type=Path, required=True)
    request.add_argument("--source-model", type=Path, required=True)
    request.add_argument("--role-contract", type=Path, required=True)
    request.add_argument("--machine-contract")
    request.add_argument("--output", type=Path, required=True)
    fork = sub.add_parser("fork", help="create a lineage-bound branch from a frozen entity")
    fork.add_argument("--db", type=Path, required=True)
    fork.add_argument("--parent", required=True)
    fork.add_argument("--child", required=True)
    fork.add_argument("--intervention", required=True)
    fork.add_argument("--omit-namespace", action="append", default=[])
    resume = sub.add_parser("continue-entity", help="resume the same identity with an independently signed handoff")
    resume.add_argument("--db", type=Path, required=True)
    resume.add_argument("--entity", required=True)
    resume.add_argument("--trust", type=Path, required=True)
    resume.add_argument("--host", required=True)
    resume.add_argument("--receipt", type=Path, required=True)
    restore_org = sub.add_parser("organization-restore", help="verify the whole package; restore frozen without host authority")
    restore_org.add_argument("--db", type=Path, required=True)
    restore_org.add_argument("--input", type=Path, required=True)
    restore_org.add_argument("--nr-pin", required=True)
    restore_org.add_argument("--runtime", required=True)
    charter = sub.add_parser("odyssey-charter", help="one Odyssey and its full target; no run")
    charter.add_argument("--output", type=Path, required=True)
    tensor = sub.add_parser("tensor-inspect", help="bounded safetensors metadata and streamed content hash")
    tensor.add_argument("--input", type=Path, required=True)
    tensor.add_argument("--pin")
    tensor.add_argument("--output", type=Path, required=True)
    delta = sub.add_parser("growth", help="compare storage snapshots, not infer an intelligence score")
    delta.add_argument("--before", type=Path, required=True)
    delta.add_argument("--after", type=Path, required=True)
    delta.add_argument("--output", type=Path, required=True)
    for command in ("phenotype", "organization-export", "native-lower"):
        child = sub.add_parser(command)
        for name in ("entity", "host"):
            child.add_argument("--" + name, required=True)
        for name in ("db", "trust", "output"):
            child.add_argument("--" + name, type=Path, required=True)
        if command != "phenotype":
            child.add_argument("--runtime", required=True, help="pinned runtime content/behavior contract hash")
        if command == "organization-export":
            child.add_argument("--assets", type=Path, help="explicit JSON mapping content hashes to local artifact paths")
            child.add_argument("--tensor-state", action="store_true", help="optional official safetensors writer")
        if command == "native-lower":
            child.add_argument("--nr-pin", required=True)
            child.add_argument("--nr", type=Path, required=True, help="actual organization.nr.json source")
    for command in ("program-init", "program-admit", "program-assess"):
        child = sub.add_parser(command, help="one Odyssey ledger; chapter evidence is not a consciousness verdict")
        for name in ("db", "trust", "charter", "output"):
            child.add_argument("--" + name, type=Path, required=True)
        if command == "program-admit":
            child.add_argument("--record", type=Path, required=True)
            child.add_argument("--certificate", type=Path, required=True)
    configure_local(sub)
    return parser


def execute(args) -> dict:
    command = args.command
    if command in {"local-init", "local", "hawking-port", "hawking-contract"}:
        return execute_local(args)
    if command == "organization-restore":
        with Store(args.db, create=True) as store:
            return restore_organization(store, args.input, nr_pin=args.nr_pin, runtime=args.runtime)
    if command in {"odyssey-charter", "tensor-inspect", "growth"}:
        if command == "odyssey-charter":
            result = program_plan()
        elif command == "tensor-inspect":
            result = safetensors_manifest(args.input, expected=args.pin)
        else:
            result = growth(read_json(args.before), read_json(args.after))
        write_json(args.output, result)
        return result
    if command in {"program-init", "program-admit", "program-assess"}:
        with Store(args.db, create=command == "program-init") as store:
            if command != "program-init":
                require(store.db.execute("SELECT 1 FROM entities WHERE id='odyssey-program'").fetchone(),
                        "initialize a separate program ledger explicitly")
            ledger = OdysseyLedger(store, Trust.load(args.trust), read_json(args.charter))
            if command == "program-admit":
                ledger.admit(read_json(args.record), read_json(args.certificate))
            result = ledger.assessment()
        write_json(args.output, result)
        return result
    if command in {"phenotype", "organization-export", "native-lower"}:
        with Store(args.db) as store:
            entity = Entity(store, args.entity, Trust.load(args.trust), host=args.host)
            if command == "organization-export":
                assets = {key: Path(value) for key, value in read_json(args.assets).items()} if args.assets else None
                return portable_organization(entity, args.output, runtime=args.runtime, assets=assets, tensor_state=args.tensor_state)
            result = (phenotype(entity) if command == "phenotype" else
                      lower_native_regions(entity, nr=read_json(args.nr), nr_pin=args.nr_pin, runtime=args.runtime, machine=args.host))
        write_json(args.output, result)
        return result
    if command == "hawking-request":
        request = hawking_request(dataset_manifest=read_json(args.dataset), source_model=read_json(args.source_model),
                                 role_contract=read_json(args.role_contract), machine_contract=args.machine_contract)
        write_json(args.output, request)
        return {"request": str(args.output), "status": request["status"]}
    if command in {"fork", "continue-entity"}:
        with Store(args.db) as store:
            if command == "fork":
                head = store.fork(args.parent, args.child, intervention=args.intervention,
                                  omit_namespaces=tuple(args.omit_namespace))
                return {"entity": args.child, "head": head, "same_identity": False}
            entity = Entity(store, args.entity, Trust.load(args.trust), host=args.host)
            return {"entity": args.entity, "head": entity.resume(read_json(args.receipt)), "same_identity": True}
    if command == "body":
        profile = inspect_body(args.machine_tag)
        write_json(args.output, profile)
        return profile
    if command == "plan":
        plan = draft_plan(kind=args.kind)
        write_json(args.output, plan)
        return {"draft": str(args.output), "plan": sha(plan), "executed": False}
    if command in {"preflight", "approve-run", "init", "worker", "fetch-scores", "serve-evaluator"}:
        plan, bindings, trust = read_json(args.plan), read_json(args.bindings), Trust.load(args.trust)
        check = preflight(plan, bindings, args.root, trust)
        if command == "preflight":
            if args.output:
                write_json(args.output, check)
            return check
        if command == "approve-run":
            require(not check["missing"], "cannot approve incomplete preflight")
            permit = attest(private_key(args.key), signer=args.signer, producer="odyssey-runner", purpose="run",
                             domain="odyssey", subject=check["subject"], evidence=sha(check), findings={"approved": True}, ttl=args.ttl)
            trust.verify(permit, purpose="run", domain="odyssey", subject=check["subject"], producer="odyssey-runner")
            write_json(args.output, permit)
            return {"permit": str(args.output), "execution_started": False}
        gate = Gate(plan, bindings, args.root, trust, read_json(args.permit))
        if command == "serve-evaluator":
            key = private_key(args.key)
            require(args.signer in trust.rules, "evaluator signer is not in the reviewed trust registry")
            public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            require(base64.b64encode(public).decode() == trust.rules[args.signer].public_key, "evaluator key/trust mismatch")
            require(args.seed.lstat().st_mode & 0o077 == 0, "private evaluator seed requires owner-only permissions")
            secret = regular_bytes(args.seed, 64)
            with Store(args.db, create=not args.db.exists(), role="evaluator", max_payload_bytes=plan["budget"]["retained_bytes"]) as store:
                evaluator = Evaluator(store, plan, secret, key, args.signer, bindings["host"],
                                      Executor(SandboxConfig(**bindings["sandbox"]), main=plan["kind"] == "main"))
                serve_evaluator(evaluator, gate=gate, port=args.port, token_env=args.token_env)
            return {"service_stopped": True}
        with Store(args.db, create=command == "init", max_payload_bytes=plan["budget"]["retained_bytes"]) as store:
            if command == "init":
                return initialize(store, gate)
            if command == "worker":
                return Runner(store, gate).worker(args.owner, max_jobs=args.max_jobs)
            require(status(store)["complete"], "fetch final scores only after all planned jobs are done")
            require(not args.output.exists(), "score directory already exists")
            oracle = OracleClient(**bindings["evaluator"])
            signed = oracle.call("report", {})
            fields(signed, {"document", "certificate"})
            document = signed["document"]
            fields(document, {"schema", "plan", "shards", "expected", "unit_of_replication"})
            require(document["schema"] == "odyssey-scores-v2" and document["plan"] == sha(plan), "score plan differs")
            findings = trust.verify(signed["certificate"], purpose="report", domain="micro-world", subject=document,
                                    producer="odyssey-runner")
            require(findings.get("complete") is True and findings.get("rows") == document["expected"], "report incomplete")
            require(type(document["shards"]) is list and 1 <= len(document["shards"]) <= 7813, "score shard bound")
            require(len(document["shards"]) == len(set(document["shards"])), "duplicate score shards")
            for key in document["shards"]:
                digest(key)
            args.output.mkdir(parents=True)
            for key in signed["document"]["shards"]:
                chunk = oracle.call("scores", {"shard": key})
                require(chunk["id"] == sha(chunk["rows"]) == key, "score shard digest mismatch")
                write_json(args.output / "shards" / (key + ".json"), chunk["rows"])
            write_json(args.output / "manifest.json", signed)
            return {"scores": str(args.output), "manifest": sha(signed["document"])}
    if command in {"status", "audit", "bundle", "restore", "migrate-v1", "prepare-training"}:
        with Store(args.db, create=command in {"restore", "migrate-v1"}) as store:
            if command == "status":
                return status(store)
            if command == "audit":
                return store.audit(args.entity)
            if command == "bundle":
                return {"bundle": str(args.output), "manifest_pin": store.export_entity(args.entity, args.output)}
            if command == "restore":
                return {"entity": store.restore_entity(args.input, manifest_pin=args.pin), "host_authority_inherited": False}
            if command == "migrate-v1":
                return migrate_v1(store, read_json(args.input, 256 * 1024 * 1024), pin=args.pin)
            entities = [Entity(store, name, Trust.load(args.trust), host=args.host) for name in args.entity]
            return export_training(entities, args.output, allowed_sources=read_json(args.sources), purpose=args.purpose)
    if command == "analyze":
        signed, shards = read_score_directory(args.scores)
        acquisition = None
        if args.db:
            with Store(args.db) as store:
                require(store.meta("run_plan") == read_json(args.plan), "acquisition store belongs to another plan")
                acquisition = acquisition_report(store)
        result = analyze(read_json(args.plan), signed, shards, Trust.load(args.trust), acquisition=acquisition)
        write_json(args.output, result)
        return {"analysis": str(args.output), "study_kind": result["study_kind"]}
    raise Refused("unknown command")


def main() -> None:
    parser = arguments()
    try:
        result = execute(parser.parse_args())
    except (Refused, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"refused: {exc}\n")
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
