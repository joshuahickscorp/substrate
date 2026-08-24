"""Tool-bearing Odyssey registry, broker, parity, and refusal tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate import odyssey_arms as arms
from substrate.odyssey_tools import (
    FRONTIER_CANARY_OPERATION,
    FRONTIER_OPERATIONS,
    REGISTRY_OPERATIONS,
    TOOL_RECEIPT_SCHEMA,
    ToolBroker,
    ToolBudget,
    ToolRefused,
    assert_budget_parity,
    declared_operations_for_task,
    discover_tool_inventory,
    make_tool_request,
    run_frontier_canary,
    verify_tool_receipt,
)


@pytest.fixture(scope="module")
def inventory() -> dict:
    return discover_tool_inventory()


@pytest.fixture
def budget() -> ToolBudget:
    return ToolBudget()


def _broker(tmp_path: Path, *, frontier: str, arm: str, budget: ToolBudget, inventory: dict) -> ToolBroker:
    return ToolBroker(
        root=tmp_path,
        lane_id=frontier,
        arm=arm,
        budget=budget,
        inventory=inventory,
        peer_budget_sha256=budget.budget_sha256(),
    )


def test_registry_is_closed_and_covers_frontiers() -> None:
    assert len(REGISTRY_OPERATIONS) >= 16
    for frontier, ops in FRONTIER_OPERATIONS.items():
        assert frontier in "ABCDEFGH"
        assert ops
        assert ops <= REGISTRY_OPERATIONS
        assert FRONTIER_CANARY_OPERATION[frontier] in ops


def test_every_registry_operation_executes_with_provenance(
    tmp_path: Path, budget: ToolBudget, inventory: dict
) -> None:
    """Each closed registry operation runs for real and returns admitted digests."""
    # Map each operation to a frontier that declares it.
    owners = {
        "repo.inspect": "A",
        "repo.test": "D",
        "repo.patch": "D",
        "formal.check_lean": "B",
        "formal.solve_smt": "C",
        "formal.countermodel": "C",
        "media.probe": "F",
        "media.extract_frames": "G",
        "media.transcode_audio": "F",
        "media.transcribe": "F",
        "document.render": "A",
        "document.extract_structure": "E",
        "three_d.build_scene": "G",
        "three_d.render": "G",
        "three_d.depth": "G",
        "three_d.move_object": "G",
        "three_d.set_camera": "G",
        "three_d.inspect_mesh": "G",
        "compute.python": "H",
        "compute.sympy": "H",
        "source.read_cached": "A",
    }
    assert set(owners) == set(REGISTRY_OPERATIONS)

    # Seed an admitted artifact for source.read_cached.
    seed_broker = _broker(tmp_path, frontier="A", arm="candidate", budget=budget, inventory=inventory)
    seed_req = make_tool_request(
        lane_id="A",
        arm="candidate",
        task_id="seed-doc",
        operation="document.render",
        frontier="A",
        declared_operations=FRONTIER_OPERATIONS["A"],
        budget=budget,
    )
    seed_resp = seed_broker.execute(seed_req)
    assert seed_resp.admitted
    seed_digest = seed_resp.output_digests[0]

    for operation, frontier in sorted(owners.items()):
        broker = _broker(tmp_path, frontier=frontier, arm="candidate", budget=budget, inventory=inventory)
        params: dict = {}
        inputs: list[str] = []
        if operation == "source.read_cached":
            inputs = [seed_digest]
        elif operation == "repo.test":
            params = {"mode": "pass"}
        elif operation == "three_d.build_scene":
            params = {"seed_id": "canary_occlusion_v1"}
        elif operation == "three_d.render":
            params = {"seed_id": "canary_occlusion_v1", "backend": "spatial3d"}
        elif operation == "three_d.depth":
            # depth requires a prior scene; build inline via render auto-build path
            # by first ensuring build_scene ran on this task sandbox.
            params = {"camera_id": "cam_front"}
        elif operation == "three_d.move_object":
            params = {"object_id": "occluder", "translation": [0.0, 0.05, 0.0]}
        elif operation == "three_d.set_camera":
            params = {"camera_id": "cam_side"}
        elif operation == "three_d.inspect_mesh":
            params = {"object_id": "occluder", "seed_id": "canary_occlusion_v1"}
        req = make_tool_request(
            lane_id=frontier,
            arm="candidate",
            task_id=f"reg-{operation.replace('.', '-')}",
            operation=operation,
            frontier=frontier,
            declared_operations=FRONTIER_OPERATIONS[frontier],
            budget=budget,
            parameters=params,
            input_artifact_digests=inputs,
        )
        resp = broker.execute(req)
        assert resp.admitted, f"{operation} refused: {resp.detail}"
        assert resp.status == "ok"
        assert resp.error_class == "ok"
        assert resp.output_digests
        assert all(len(d) == 64 for d in resp.output_digests)
        assert resp.tool_revision.get("tool_id")
        assert resp.provenance.get("request_sha256")
        assert resp.receipt_sha256


def test_undeclared_operation_is_refused(tmp_path: Path, budget: ToolBudget, inventory: dict) -> None:
    from dataclasses import replace

    broker = _broker(tmp_path, frontier="C", arm="candidate", budget=budget, inventory=inventory)
    req = make_tool_request(
        lane_id="C",
        arm="candidate",
        task_id="undeclared",
        operation="compute.sympy",  # not on C surface
        frontier="C",
        declared_operations=FRONTIER_OPERATIONS["C"],
        budget=budget,
    )
    resp = broker.execute(req)
    assert not resp.admitted
    assert resp.error_class == "undeclared_operation"

    # Non-registry operation name is refused even if somehow declared.
    forged = replace(req, operation="shell.exec", declared_operations=frozenset({"shell.exec"}))
    resp2 = broker.execute(forged)
    assert not resp2.admitted
    assert resp2.error_class == "undeclared_operation"


def test_evaluator_only_material_is_refused(tmp_path: Path, budget: ToolBudget, inventory: dict) -> None:
    broker = _broker(tmp_path, frontier="B", arm="candidate", budget=budget, inventory=inventory)
    req = make_tool_request(
        lane_id="B",
        arm="candidate",
        task_id="eval-leak",
        operation="formal.check_lean",
        frontier="B",
        declared_operations=FRONTIER_OPERATIONS["B"],
        budget=budget,
        parameters={"source": "theorem t : True := by\n  -- evaluator answer key\n  trivial\n"},
    )
    resp = broker.execute(req)
    assert not resp.admitted
    assert resp.error_class == "evaluator_isolation"


def test_candidate_control_budget_parity_refuses_asymmetry(budget: ToolBudget) -> None:
    shared = assert_budget_parity(budget, ToolBudget.from_dict(budget.to_dict()))
    assert shared == budget.budget_sha256()
    other = ToolBudget(cpu_ms=budget.cpu_ms + 1)
    with pytest.raises(ToolRefused, match="asymmetry"):
        assert_budget_parity(budget, other)


def test_broker_refuses_peer_budget_mismatch(tmp_path: Path, budget: ToolBudget, inventory: dict) -> None:
    with pytest.raises(ToolRefused, match="asymmetry"):
        ToolBroker(
            root=tmp_path,
            lane_id="A",
            arm="candidate",
            budget=budget,
            inventory=inventory,
            peer_budget_sha256="0" * 64,
        )


def test_forged_tool_receipt_is_refused(tmp_path: Path, budget: ToolBudget, inventory: dict) -> None:
    broker = _broker(tmp_path, frontier="H", arm="candidate", budget=budget, inventory=inventory)
    req = make_tool_request(
        lane_id="H",
        arm="candidate",
        task_id="forge-base",
        operation="compute.python",
        frontier="H",
        declared_operations=FRONTIER_OPERATIONS["H"],
        budget=budget,
        parameters={"expression": "1+1"},
    )
    resp = broker.execute(req)
    assert resp.admitted
    good = {
        "schema": TOOL_RECEIPT_SCHEMA,
        "status": "ok",
        "operation": resp.operation,
        "output_digests": resp.output_digests,
        "tool_revision": resp.tool_revision,
        "provenance": resp.provenance,
        "error_class": "ok",
        "admitted": True,
        "receipt_sha256": resp.receipt_sha256,
    }
    # Tamper output digests without updating self-digest.
    forged = dict(good)
    forged["output_digests"] = ["a" * 64]
    with pytest.raises(ToolRefused, match="forged tool receipt"):
        verify_tool_receipt(forged, cache=broker.cache)
    # Tamper self-digest claim.
    forged2 = dict(good)
    forged2["receipt_sha256"] = "b" * 64
    with pytest.raises(ToolRefused, match="forged tool receipt"):
        verify_tool_receipt(forged2, cache=broker.cache)


def test_quarantine_bypass_is_refused(tmp_path: Path, budget: ToolBudget, inventory: dict) -> None:
    broker = _broker(tmp_path, frontier="A", arm="candidate", budget=budget, inventory=inventory)
    # Ingest without admit by writing a fake digest read.
    with pytest.raises(ToolRefused, match="quarantine bypass|cache lookup"):
        broker.cache.read_admitted("c" * 64, expected_lane="A")


def test_cross_lane_cache_access_is_refused(tmp_path: Path, budget: ToolBudget, inventory: dict) -> None:
    a = _broker(tmp_path, frontier="A", arm="candidate", budget=budget, inventory=inventory)
    req = make_tool_request(
        lane_id="A",
        arm="candidate",
        task_id="lane-a",
        operation="compute.python" if "compute.python" in FRONTIER_OPERATIONS["A"] else "repo.inspect",
        frontier="A",
        declared_operations=FRONTIER_OPERATIONS["A"],
        budget=budget,
        parameters={"expression": "2+2"} if "compute.python" in FRONTIER_OPERATIONS["A"] else {},
    )
    # A does not declare compute.python — use repo.inspect
    req = make_tool_request(
        lane_id="A",
        arm="candidate",
        task_id="lane-a",
        operation="repo.inspect",
        frontier="A",
        declared_operations=FRONTIER_OPERATIONS["A"],
        budget=budget,
    )
    resp = a.execute(req)
    assert resp.admitted
    b = _broker(tmp_path, frontier="B", arm="candidate", budget=budget, inventory=inventory)
    with pytest.raises(ToolRefused, match="cross-lane"):
        # Caller must name the broker's own lane; a foreign expected_lane is refused.
        b.cache.read_admitted(resp.output_digests[0], expected_lane="A")
    # Digest from lane A is absent from lane B even when expected_lane matches B.
    with pytest.raises(ToolRefused, match="absent|cross-lane|quarantine bypass|cache lookup"):
        b.cache.read_admitted(resp.output_digests[0], expected_lane="B")


@pytest.mark.parametrize("frontier", sorted(FRONTIER_OPERATIONS))
def test_per_frontier_real_tool_invocation(
    frontier: str, tmp_path: Path, budget: ToolBudget, inventory: dict
) -> None:
    operation = FRONTIER_CANARY_OPERATION[frontier]
    for arm in ("candidate", "control"):
        broker = _broker(tmp_path, frontier=frontier, arm=arm, budget=budget, inventory=inventory)
        req = make_tool_request(
            lane_id=frontier,
            arm=arm,
            task_id=f"frontier-{frontier}-{arm}",
            operation=operation,
            frontier=frontier,
            declared_operations=FRONTIER_OPERATIONS[frontier],
            budget=budget,
        )
        resp = broker.execute(req)
        assert resp.admitted, f"{frontier}/{arm}/{operation}: {resp.detail}"
        assert resp.output_digests
        assert resp.tool_revision.get("artifact_sha256")
        # Persist a small proof row for the canary evidence binder.
        proof = {
            "frontier": frontier,
            "arm": arm,
            "operation": operation,
            "tool_revision": resp.tool_revision,
            "artifact_digest": resp.output_digests[0],
            "receipt_sha256": resp.receipt_sha256,
        }
        out = tmp_path / "artifacts/substrate/odyssey7d/frontier-proofs" / f"{frontier}-{arm}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(proof, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_arms_prompt_permits_only_declared_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_chat(**kwargs: object) -> tuple[dict, dict]:
        calls.append(dict(kwargs))
        return (
            {"summary": "visible-only answer"},
            {
                "prompt_eval_count": 5,
                "eval_count": 7,
                "generation_num_predict": arms.GENERATION_NUM_PREDICT,
                "substantive_max_tokens": arms.MAX_OUTPUT_TOKENS,
                "transport_gate": "pass",
            },
        )

    monkeypatch.setattr(arms, "_ollama_chat", fake_chat)
    task = {
        "schema": "SUBSTRATE_ODYSSEY_TEST_TASK/v1",
        "activation": False,
        "program": arms.PROGRAM,
        "frontier": "A",
        "task_id": "A-0000",
        "request": "Summarize the visible project note.",
        "required_receipt": ["summary"],
    }
    path = tmp_path / "runs/substrate/odyssey7d/v1/arms/A/candidate/requests/000-retrieval.json"
    receipt = tmp_path / "runs/substrate/odyssey7d/v1/arms/A/candidate/receipts/000-retrieval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "schema": arms.REQUEST_SCHEMA,
        "activation": False,
        "authority_sha256": "a" * 64,
        "run_id": "odyssey-tools-fixture",
        "frontier": "A",
        "role": "candidate",
        "cycle": 0,
        "phase": "retrieval",
        "task": task,
        "candidate_manifest_sha256": "b" * 64,
        "receipt_path": str(receipt.relative_to(tmp_path)),
    }
    request["request_sha256"] = arms.digest(request)
    path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    arms.run(
        tmp_path,
        role="candidate",
        model="gpt-oss:20b",
        state_root="runs/substrate/odyssey7d/v1/private-state/A/candidate",
        self_sha256=arms.file_digest(Path(arms.__file__).resolve()),
        request_path=path,
    )
    system = calls[0]["messages"][0]["content"]
    assert "Do not use external tools" not in system
    assert "Bounded tool operations permitted" in system
    assert "repo.inspect" in system
    user = json.loads(calls[0]["messages"][1]["content"])
    assert "allowed_operations" in user
    assert set(user["allowed_operations"]) == set(FRONTIER_OPERATIONS["A"])


def test_declared_operations_reject_foreign_registry_name() -> None:
    with pytest.raises(ToolRefused):
        declared_operations_for_task({"allowed_operations": ["shell.exec"]}, frontier="A")
    with pytest.raises(ToolRefused):
        declared_operations_for_task({"allowed_operations": ["formal.check_lean"]}, frontier="A")


def test_public_tool_bearing_canary_writes_evidence(tmp_path: Path, budget: ToolBudget) -> None:
    # Use a longer wall budget so Blender can finish under load.
    long_budget = ToolBudget(wall_seconds=180)
    document = run_frontier_canary(tmp_path, budget=long_budget)
    assert document["all_admitted"] is True
    assert len(document["rows"]) == 16
    out = tmp_path / "artifacts/substrate/odyssey7d/tool-bearing-canary/TOOL_BEARING_CANARY.json"
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["sha256"] == document["sha256"]
    by_frontier = {row["frontier"]: row for row in loaded["rows"] if row["role"] == "candidate"}
    for frontier in "ABCDEFGH":
        row = by_frontier[frontier]
        assert row["admitted"] is True
        assert row["output_digests"]
        assert row["tool_revision"]["artifact_sha256"]
