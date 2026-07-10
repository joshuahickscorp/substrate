from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.custom_substrate_finalize import (
    CHAIN_SCHEMA,
    RAW_SCHEMA,
    freeze_raw,
    materialize_proof_chain,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def test_materialize_proof_chain_copies_exact_bytes_and_rewrites_links(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    links: dict[str, dict[str, str]] = {}
    for role, filename in {
        "raw_training_receipt": "raw_workbench_receipt.json",
        "current_evidence_attestation": "current_evidence_attestation.json",
        "environment_receipt": "environment_receipt.json",
        "independent_verifier": "independent_verifier.json",
    }.items():
        raw = (json.dumps({"role": role}, sort_keys=True) + "\n").encode()
        (run_dir / filename).write_bytes(raw)
        links[role] = {"path": filename, "sha256": _sha256(raw)}
    composite = {
        "schema": RAW_SCHEMA,
        "receipt_chain_schema": CHAIN_SCHEMA,
        "receipt_chain": links,
        "authoritative_promotion": {"verdict": "not-promoted"},
    }
    (run_dir / "workbench_receipt.json").write_text(json.dumps(composite))

    proof = tmp_path / "proof" / "CUSTOM_SUBSTRATE_PILOT.json"
    durable = materialize_proof_chain(run_dir, proof)

    for role, link in durable["receipt_chain"].items():
        target = Path(link["path"])
        assert target.is_file()
        assert _sha256(target.read_bytes()) == links[role]["sha256"]
        assert target.parent.name == "CUSTOM_SUBSTRATE_PILOT_CHAIN"
    assert json.loads(proof.read_text())["receipt_chain"] == durable["receipt_chain"]


def test_materialize_proof_chain_replaces_only_the_previously_bound_target(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    filename = "independent_verifier.json"
    first_raw = b'{"all_ok": true}\n'
    (run_dir / filename).write_bytes(first_raw)
    composite = {
        "schema": RAW_SCHEMA,
        "receipt_chain_schema": CHAIN_SCHEMA,
        "receipt_chain": {"independent_verifier": {"path": filename, "sha256": _sha256(first_raw)}},
        "authoritative_promotion": {"verdict": "not-promoted"},
    }
    (run_dir / "workbench_receipt.json").write_text(json.dumps(composite))
    proof = tmp_path / "proof" / "CUSTOM_SUBSTRATE_PILOT.json"
    materialize_proof_chain(run_dir, proof)

    second_raw = b'{"all_ok": false}\n'
    (run_dir / filename).write_bytes(second_raw)
    composite["receipt_chain"]["independent_verifier"]["sha256"] = _sha256(second_raw)
    (run_dir / "workbench_receipt.json").write_text(json.dumps(composite))
    with pytest.raises(RuntimeError, match="different hash"):
        materialize_proof_chain(run_dir, proof)

    durable = materialize_proof_chain(run_dir, proof, replace_existing=True)
    target = Path(durable["receipt_chain"]["independent_verifier"]["path"])
    assert target.read_bytes() == second_raw


def test_freeze_raw_can_reuse_only_an_exact_bound_finalized_raw_receipt(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = {"schema": RAW_SCHEMA, "complete": True, "seed_results": {}}
    raw_bytes = _json_bytes(raw)
    raw_hash = _sha256(raw_bytes)
    (run_dir / "raw_workbench_receipt.json").write_bytes(raw_bytes)
    composite = {
        **raw,
        "receipt_chain_schema": CHAIN_SCHEMA,
        "receipt_chain": {
            "raw_training_receipt": {
                "path": "raw_workbench_receipt.json",
                "sha256": raw_hash,
            }
        },
        "authoritative_promotion": {"verdict": "not-promoted"},
    }
    (run_dir / "workbench_receipt.json").write_bytes(_json_bytes(composite))

    loaded, loaded_hash = freeze_raw(run_dir, allow_finalized=True)
    assert loaded == raw
    assert loaded_hash == raw_hash

    (run_dir / "raw_workbench_receipt.json").write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="does not match the finalized binding"):
        freeze_raw(run_dir, allow_finalized=True)
