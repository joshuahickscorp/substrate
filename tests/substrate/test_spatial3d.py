"""Native spatial3d backend: determinism, occlusion, canaries, evaluator isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from substrate import spatial3d
from substrate.odyssey_tools import (
    FRONTIER_OPERATIONS,
    ToolBroker,
    ToolBudget,
    discover_tool_inventory,
    make_tool_request,
)


@pytest.fixture(scope="module")
def inventory() -> dict:
    return discover_tool_inventory()


@pytest.fixture
def budget() -> ToolBudget:
    return ToolBudget()


def test_renderer_identity() -> None:
    assert spatial3d.RENDERER_ID == "substrate_spatial3d"
    assert spatial3d.RENDERER_VERSION


def test_all_seven_canaries_pass() -> None:
    doc = spatial3d.run_all_canaries()
    assert doc["canary_count"] == 7
    assert doc["all_passed"] is True
    for row in doc["rows"]:
        assert row["passed"], row
        assert row["deterministic"] is True
        assert row["requires_render"] is True


def test_determinism_byte_identical_rgb_and_depth() -> None:
    seed_id = "canary_occlusion_v1"
    a = spatial3d.render_scene(spatial3d.build_scene_from_seed(seed_id), camera_id="cam_front")
    b = spatial3d.render_scene(spatial3d.build_scene_from_seed(seed_id), camera_id="cam_front")
    assert a.rgb_png == b.rgb_png
    assert a.depth_f32 == b.depth_f32
    assert hashlib.sha256(a.rgb_png).hexdigest() == hashlib.sha256(b.rgb_png).hexdigest()
    assert hashlib.sha256(a.depth_f32).hexdigest() == hashlib.sha256(b.depth_f32).hexdigest()


def test_occlusion_zbuffer_two_viewpoints() -> None:
    scene = spatial3d.build_scene_from_seed("canary_occlusion_v1")
    front = spatial3d.render_scene(scene, camera_id="cam_front")
    side = spatial3d.render_scene(scene, camera_id="cam_side")
    assert front.pixel_object_coverage.get("blue_pyramid", 0) < 8
    assert side.pixel_object_coverage.get("blue_pyramid", 0) >= 20
    assert front.rgb_png != side.rgb_png
    assert front.depth_f32 != side.depth_f32


def test_two_viewpoints_produce_depth_artifacts() -> None:
    scene = spatial3d.build_scene_from_seed("canary_depth_ordering_v1")
    r0 = spatial3d.render_scene(scene, camera_id="cam_front")
    r1 = spatial3d.render_scene(scene, camera_id="cam_top")
    assert r0.width == scene.width and r0.height == scene.height
    assert len(r0.depth_f32) == scene.width * scene.height * 4
    assert len(r1.depth_f32) == scene.width * scene.height * 4
    assert r0.rgb_png.startswith(b"\x89PNG")
    assert r1.rgb_png.startswith(b"\x89PNG")


def test_evaluator_answer_not_in_candidate_sensor() -> None:
    scene = spatial3d.build_scene_from_seed("canary_occlusion_v1")
    rendered = spatial3d.render_scene(scene, camera_id="cam_front")
    sensor = spatial3d.candidate_sensor_bundle(rendered)
    blob = json.dumps(sensor, sort_keys=True)
    answer = spatial3d.evaluator_answer_state("canary_occlusion_v1")
    assert "hidden_from_camera" not in blob
    assert "answer" not in blob
    assert answer["target"] == "blue_pyramid"
    # Candidate sensor must not embed the GT answer payload.
    assert "cam_front_visible" not in blob


def test_scene_graph_write_uses_evaluator_namespace(tmp_path: Path) -> None:
    scene = spatial3d.build_scene_from_seed("canary_relative_position_v1")
    eval_dir = tmp_path / "evaluator"
    digests = spatial3d.write_evaluator_state(eval_dir, scene, "canary_relative_position_v1")
    assert (eval_dir / "scene_graph.json").is_file()
    assert (eval_dir / "answer_state.json").is_file()
    assert len(digests["scene_graph_sha256"]) == 64
    # Runtime path is separate.
    runtime = tmp_path / "work" / "scene_runtime.json"
    spatial3d.write_runtime_scene(runtime, scene)
    loaded = spatial3d.read_runtime_scene(runtime)
    assert loaded.canonical_digest() == scene.canonical_digest()


def test_broker_parity_candidate_control_identical_scene_inputs(
    tmp_path: Path, budget: ToolBudget, inventory: dict
) -> None:
    digests: dict[str, list[str]] = {}
    for arm in ("candidate", "control"):
        broker = ToolBroker(
            root=tmp_path,
            lane_id="G",
            arm=arm,
            budget=budget,
            inventory=inventory,
            peer_budget_sha256=budget.budget_sha256(),
        )
        req = make_tool_request(
            lane_id="G",
            arm=arm,
            task_id=f"parity-g-{arm}",
            operation="three_d.render",
            frontier="G",
            declared_operations=FRONTIER_OPERATIONS["G"],
            budget=budget,
            parameters={"seed_id": "canary_occlusion_v1", "camera_id": "cam_front", "backend": "spatial3d"},
        )
        resp = broker.execute(req)
        assert resp.admitted, resp.detail
        digests[arm] = list(resp.output_digests)
        assert resp.tool_revision["tool_id"] == spatial3d.RENDERER_ID
        assert resp.tool_revision["version"] == spatial3d.RENDERER_VERSION
    # Byte-identical admitted artifacts for candidate and control on the same seed.
    assert digests["candidate"] == digests["control"]


def test_broker_motion_and_camera_ops(tmp_path: Path, budget: ToolBudget, inventory: dict) -> None:
    broker = ToolBroker(
        root=tmp_path,
        lane_id="G",
        arm="candidate",
        budget=budget,
        inventory=inventory,
        peer_budget_sha256=budget.budget_sha256(),
    )
    build = make_tool_request(
        lane_id="G",
        arm="candidate",
        task_id="g-motion",
        operation="three_d.build_scene",
        frontier="G",
        declared_operations=FRONTIER_OPERATIONS["G"],
        budget=budget,
        parameters={"seed_id": "canary_object_permanence_v1"},
    )
    assert broker.execute(build).admitted
    r1 = broker.execute(
        make_tool_request(
            lane_id="G",
            arm="candidate",
            task_id="g-motion",
            operation="three_d.render",
            frontier="G",
            declared_operations=FRONTIER_OPERATIONS["G"],
            budget=budget,
            parameters={"camera_id": "cam_main"},
        )
    )
    assert r1.admitted
    move = broker.execute(
        make_tool_request(
            lane_id="G",
            arm="candidate",
            task_id="g-motion",
            operation="three_d.move_object",
            frontier="G",
            declared_operations=FRONTIER_OPERATIONS["G"],
            budget=budget,
            parameters={"object_id": "mover", "translation": [2.0, 0.0, 0.0]},
        )
    )
    assert move.admitted
    r2 = broker.execute(
        make_tool_request(
            lane_id="G",
            arm="candidate",
            task_id="g-motion",
            operation="three_d.render",
            frontier="G",
            declared_operations=FRONTIER_OPERATIONS["G"],
            budget=budget,
            parameters={"camera_id": "cam_main"},
        )
    )
    assert r2.admitted
    # Digests change after motion (scene input to the second render differs).
    assert r1.output_digests != r2.output_digests
    cam = broker.execute(
        make_tool_request(
            lane_id="G",
            arm="candidate",
            task_id="g-motion",
            operation="three_d.set_camera",
            frontier="G",
            declared_operations=FRONTIER_OPERATIONS["G"],
            budget=budget,
            parameters={"camera_id": "cam_alt"},
        )
    )
    assert cam.admitted
    depth = broker.execute(
        make_tool_request(
            lane_id="G",
            arm="candidate",
            task_id="g-motion",
            operation="three_d.depth",
            frontier="G",
            declared_operations=FRONTIER_OPERATIONS["G"],
            budget=budget,
            parameters={},
        )
    )
    assert depth.admitted
    assert depth.output_digests


def test_evaluator_path_token_refused_in_parameters(
    tmp_path: Path, budget: ToolBudget, inventory: dict
) -> None:
    broker = ToolBroker(
        root=tmp_path,
        lane_id="G",
        arm="candidate",
        budget=budget,
        inventory=inventory,
        peer_budget_sha256=budget.budget_sha256(),
    )
    req = make_tool_request(
        lane_id="G",
        arm="candidate",
        task_id="eval-leak-g",
        operation="three_d.render",
        frontier="G",
        declared_operations=FRONTIER_OPERATIONS["G"],
        budget=budget,
        parameters={"seed_id": "evaluator_secret_seed"},
    )
    resp = broker.execute(req)
    assert not resp.admitted
    assert resp.error_class == "evaluator_isolation"


def test_committed_seeds_roundtrip(tmp_path: Path) -> None:
    paths = spatial3d.write_committed_seeds(tmp_path / "seeds")
    assert any(p.name == "SEED_INDEX.json" for p in paths)
    index = json.loads((tmp_path / "seeds" / "SEED_INDEX.json").read_text(encoding="utf-8"))
    assert set(index["seeds"]) == set(spatial3d.CANARY_SEED_IDS)
    for seed_id in spatial3d.CANARY_SEED_IDS:
        public = json.loads((tmp_path / "seeds" / f"{seed_id}.json").read_text(encoding="utf-8"))
        assert "answer" not in public
        answer = json.loads(
            (tmp_path / "seeds" / "evaluator" / seed_id / "answer_state.json").read_text(encoding="utf-8")
        )
        assert answer.get("requires_render") is True
