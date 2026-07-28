"""Seeded local body and environment contracts for Substrate v5.

Physics state and rendering are deliberately separate.  Public observations have
no physical object or target identifiers; an oracle view is available only
through an explicit post-commitment method for instrument validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from substrate.v5sensorium import Modality, SensoriumError, Vec3


class EnvironmentError(ValueError):
    """An action, checkpoint, or observation violates the environment contract."""


def _digest(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _assert_no_hidden_ids(value: object) -> None:
    forbidden = {
        "answer",
        "answer_id",
        "oracle",
        "oracle_id",
        "outcome",
        "physical_id",
        "private_target",
        "target",
        "target_id",
        "truth",
        "truth_id",
    }
    if isinstance(value, Mapping):
        leaked = forbidden & {str(key).lower() for key in value}
        if leaked:
            raise EnvironmentError(f"hidden target authority leaked into observation: {sorted(leaked)}")
        for child in value.values():
            _assert_no_hidden_ids(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_no_hidden_ids(child)


@dataclass(frozen=True)
class BodyContract:
    identity: str
    sensors: tuple[Modality, ...]
    actuators: tuple[str, ...]
    coordinate_frames: tuple[str, ...]
    latency_ms: Mapping[str, float]
    action_cost: Mapping[str, float]
    failure_modes: tuple[str, ...]
    reach: float
    field_of_view_degrees: float
    tool_access: tuple[str, ...]
    energy_budget: float
    checkpoint_format: str

    def __post_init__(self) -> None:
        if not self.identity or not self.sensors or not self.actuators or not self.coordinate_frames:
            raise EnvironmentError("body identity, sensors, actuators, and coordinate frames are required")
        if set(self.latency_ms) != set(self.actuators) or set(self.action_cost) != set(self.actuators):
            raise EnvironmentError("every actuator needs explicit latency and cost")
        if any(value < 0.0 for value in (*self.latency_ms.values(), *self.action_cost.values())):
            raise EnvironmentError("body latency and action cost must be non-negative")
        if self.reach < 0.0 or not 0.0 < self.field_of_view_degrees <= 360.0 or self.energy_budget <= 0.0:
            raise EnvironmentError("body geometry or energy budget is invalid")

    def check_action(self, action: str) -> None:
        if action not in self.actuators:
            raise EnvironmentError(f"body {self.identity!r} cannot execute {action!r}")


class DesktopBodyContract(BodyContract):
    """Mouse/keyboard/tool body for a sandboxed 2D desktop."""

    def __init__(self) -> None:
        actuators = ("move_pointer", "click", "type_text", "inspect", "wait")
        super().__init__(
            identity="desktop-browser-body-v5",
            sensors=(Modality.IMAGE, Modality.TEXT, Modality.BODY_TOOL),
            actuators=actuators,
            coordinate_frames=("desktop_pixels", "pointer"),
            latency_ms={action: value for action, value in zip(actuators, (8.0, 25.0, 40.0, 12.0, 1.0), strict=True)},
            action_cost={action: value for action, value in zip(actuators, (0.02, 0.08, 0.10, 0.04, 0.0), strict=True)},
            failure_modes=("pointer_out_of_bounds", "occluded_control", "disabled_control", "tool_unavailable"),
            reach=1_000_000.0,
            field_of_view_degrees=90.0,
            tool_access=("sandbox_inspector", "local_text_entry"),
            energy_budget=100.0,
            checkpoint_format="substrate.v5.desktop-body/1",
        )


class Simulator3DBodyContract(BodyContract):
    """Pose/depth/tool body for a deterministic room simulator."""

    def __init__(self) -> None:
        actuators = ("move_body", "rotate_view", "reach", "request_depth", "inspect", "wait")
        super().__init__(
            identity="simulator-3d-body-v5",
            sensors=(Modality.IMAGE, Modality.VIDEO, Modality.MOTION, Modality.DEPTH_3D, Modality.AUDIO, Modality.BODY_TOOL),
            actuators=actuators,
            coordinate_frames=("world", "body", "camera"),
            latency_ms={action: value for action, value in zip(actuators, (60.0, 20.0, 80.0, 25.0, 30.0, 1.0), strict=True)},
            action_cost={action: value for action, value in zip(actuators, (0.20, 0.05, 0.30, 0.12, 0.08, 0.0), strict=True)},
            failure_modes=("collision", "out_of_reach", "outside_room", "sensor_dropout", "tool_unavailable"),
            reach=1.5,
            field_of_view_degrees=72.0,
            tool_access=("depth_query", "local_scene_inspector"),
            energy_budget=120.0,
            checkpoint_format="substrate.v5.simulator-body/1",
        )


@dataclass(frozen=True)
class EnvironmentContract:
    identity: str
    state_schema: str
    sensors: tuple[Modality, ...]
    actions: tuple[str, ...]
    deterministic_transitions: bool
    seeded: bool
    action_costs: Mapping[str, float]
    failure_modes: tuple[str, ...]
    hidden_state_schema: str
    oracle_schema: str
    checkpoint_schema: str
    reset_semantics: str
    render_identity: str
    physics_identity: str

    def __post_init__(self) -> None:
        if set(self.action_costs) != set(self.actions):
            raise EnvironmentError("each environment action requires a declared cost")
        if self.render_identity == self.physics_identity:
            raise EnvironmentError("render and physics identities must remain separate")


@dataclass(frozen=True)
class ActionReceipt:
    environment_identity: str
    action: str
    parameters: Mapping[str, Any]
    cost: float
    timestamp: float
    success: bool
    failure: str | None
    state_digest_before: str
    state_digest_after: str


@dataclass(frozen=True)
class CommitmentToken:
    """An instance-issued capability proving a decision preceded oracle access."""

    environment_identity: str
    committed_at_tick: int
    state_digest: str
    decision_digest: str
    token_sha256: str


class DesktopEnvironment:
    """A small desktop world whose controls have private physics identities."""

    width = 800
    height = 600

    def __init__(self, seed: int, *, render_variant: str = "flat") -> None:
        self.seed = int(seed)
        self.render_variant = render_variant
        self.body = DesktopBodyContract()
        self.contract = EnvironmentContract(
            identity="desktop-sandbox-v5",
            state_schema="substrate.v5.desktop-state/1",
            sensors=self.body.sensors,
            actions=self.body.actuators,
            deterministic_transitions=True,
            seeded=True,
            action_costs=self.body.action_cost,
            failure_modes=self.body.failure_modes,
            hidden_state_schema="substrate.v5.desktop-physics-private/1",
            oracle_schema="substrate.v5.desktop-oracle/1",
            checkpoint_schema=self.body.checkpoint_format,
            reset_semantics="restore the exact seed-derived initial physics state",
            render_identity=f"desktop-render:{render_variant}",
            physics_identity="desktop-hitbox-physics-v1",
        )
        self._initial_truth = self._make_truth()
        self._truth: dict[str, Any] = {}
        self._tick = 0
        self._energy = self.body.energy_budget
        self._oracle_commitments: dict[str, CommitmentToken] = {}
        self.reset()

    def _make_truth(self) -> dict[str, Any]:
        offset_x = int(_digest(self.seed, "x")[:4], 16) % 80
        offset_y = int(_digest(self.seed, "y")[:4], 16) % 60
        return {
            "cursor": [20.0, 20.0],
            "focus": None,
            "typed": "",
            "controls": [
                {
                    "physical_id": "control-a",
                    "bounds": [100 + offset_x, 120 + offset_y, 220 + offset_x, 180 + offset_y],
                    "role": "button",
                    "label": "Continue",
                    "enabled": True,
                    "z": 1,
                },
                {
                    "physical_id": "control-b",
                    "bounds": [280 + offset_x, 120 + offset_y, 520 + offset_x, 180 + offset_y],
                    "role": "text_field",
                    "label": "Notes",
                    "enabled": True,
                    "z": 1,
                },
            ],
            "click_count": 0,
        }

    def reset(self) -> dict[str, Any]:
        self._truth = copy.deepcopy(self._initial_truth)
        self._tick = 0
        self._energy = self.body.energy_budget
        self._oracle_commitments.clear()
        return self.observe()

    def _physics_digest(self) -> str:
        return _digest(self.contract.physics_identity, self._truth, self._tick, self._energy)

    def _style(self, role: str, index: int) -> dict[str, Any]:
        token = int(_digest(self.seed, self.render_variant, role, index)[:8], 16)
        return {
            "fill_rgb": ((token >> 16) & 255, (token >> 8) & 255, token & 255),
            "border": 1 + token % 3,
            "glyph": ("square", "round", "line")[token % 3],
        }

    def render(self) -> dict[str, Any]:
        # No physical ID appears in this observation.  The order is a z-order, not identity.
        elements = [
            {
                "bounds": tuple(control["bounds"]),
                "role": control["role"],
                "visible_text": control["label"],
                "enabled": control["enabled"],
                "style": self._style(control["role"], index),
            }
            for index, control in enumerate(sorted(self._truth["controls"], key=lambda control: control["z"]))
        ]
        return {
            "render_identity": self.contract.render_identity,
            "viewport": (self.width, self.height),
            "cursor": tuple(self._truth["cursor"]),
            "elements": elements,
            "visible_typed_text": self._truth["typed"],
        }

    def observe(self) -> dict[str, Any]:
        observation = {
            "environment": self.contract.identity,
            "sequence": self._tick,
            "timestamp": self._tick * 0.05,
            "coordinate_frame": "desktop_pixels",
            "modalities": tuple(modality.value for modality in self.contract.sensors),
            "render": self.render(),
            "body": {"energy_remaining": self._energy, "available_actions": self.contract.actions},
        }
        _assert_no_hidden_ids(observation)
        return observation

    def _control_at(self, x: float, y: float) -> dict[str, Any] | None:
        rows = sorted(self._truth["controls"], key=lambda control: control["z"], reverse=True)
        return next(
            (
                control
                for control in rows
                if control["bounds"][0] <= x <= control["bounds"][2]
                and control["bounds"][1] <= y <= control["bounds"][3]
            ),
            None,
        )

    def step(self, action: str, parameters: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ActionReceipt]:
        parameters = dict(parameters or {})
        self.body.check_action(action)
        before = self._physics_digest()
        cost = self.contract.action_costs[action]
        success = True
        failure = None
        if self._energy < cost:
            success = False
            failure = "energy_exhausted"
        elif action == "move_pointer":
            x, y = float(parameters["x"]), float(parameters["y"])
            if not 0.0 <= x <= self.width or not 0.0 <= y <= self.height:
                success = False
                failure = "pointer_out_of_bounds"
            else:
                self._truth["cursor"] = [x, y]
        elif action == "click":
            control = self._control_at(*self._truth["cursor"])
            if control is None:
                success = False
                failure = "no_control_at_pointer"
            elif not control["enabled"]:
                success = False
                failure = "disabled_control"
            else:
                self._truth["focus"] = control["physical_id"]
                self._truth["click_count"] += 1
        elif action == "type_text":
            focused = next(
                (control for control in self._truth["controls"] if control["physical_id"] == self._truth["focus"]),
                None,
            )
            if focused is None or focused["role"] != "text_field":
                success = False
                failure = "no_text_field_focused"
            else:
                self._truth["typed"] += str(parameters.get("text", ""))
        elif action in {"inspect", "wait"}:
            pass
        if success:
            self._energy -= cost
        self._tick += 1
        after = self._physics_digest()
        receipt = ActionReceipt(
            environment_identity=self.contract.identity,
            action=action,
            parameters=parameters,
            cost=cost if success else 0.0,
            timestamp=self._tick * 0.05,
            success=success,
            failure=failure,
            state_digest_before=before,
            state_digest_after=after,
        )
        return self.observe(), receipt

    def checkpoint(self) -> dict[str, Any]:
        body = {
            "schema": self.contract.checkpoint_schema,
            "environment": self.contract.identity,
            "seed": self.seed,
            "render_variant": self.render_variant,
            "tick": self._tick,
            "energy": self._energy,
            "physics": copy.deepcopy(self._truth),
        }
        body["digest"] = _digest(body)
        return body

    def restore(self, checkpoint: Mapping[str, Any]) -> DesktopEnvironment:
        supplied = dict(checkpoint)
        digest = supplied.pop("digest", None)
        if digest != _digest(supplied):
            raise EnvironmentError("desktop checkpoint digest mismatch")
        if supplied["schema"] != self.contract.checkpoint_schema or supplied["environment"] != self.contract.identity:
            raise EnvironmentError("desktop checkpoint identity mismatch")
        if supplied["seed"] != self.seed or supplied["render_variant"] != self.render_variant:
            raise EnvironmentError("desktop checkpoint belongs to another environment instance")
        self._tick = int(supplied["tick"])
        self._energy = float(supplied["energy"])
        self._truth = copy.deepcopy(supplied["physics"])
        self._oracle_commitments.clear()
        return self

    def commit_decision(self, decision: Mapping[str, Any]) -> CommitmentToken:
        normalized = dict(decision)
        if not normalized:
            raise EnvironmentError("oracle commitment requires a nonempty decision")
        _assert_no_hidden_ids(normalized)
        state_digest = self._physics_digest()
        decision_digest = _digest(normalized)
        token_sha256 = _digest(
            "oracle-commitment",
            self.contract.identity,
            self.seed,
            self.render_variant,
            self._tick,
            state_digest,
            decision_digest,
        )
        token = CommitmentToken(
            environment_identity=self.contract.identity,
            committed_at_tick=self._tick,
            state_digest=state_digest,
            decision_digest=decision_digest,
            token_sha256=token_sha256,
        )
        self._oracle_commitments[token_sha256] = token
        return token

    def reveal_physics_after_commitment(
        self,
        commitment: CommitmentToken | None = None,
    ) -> dict[str, Any]:
        """Return the private oracle layer; never include it in ``observe``."""

        if commitment is None:
            raise EnvironmentError("oracle reveal requires a prior commitment token")
        issued = self._oracle_commitments.pop(commitment.token_sha256, None)
        if issued is not commitment or commitment.environment_identity != self.contract.identity:
            raise EnvironmentError("oracle commitment token is invalid, foreign, or already consumed")
        return {
            "revealed_after_commitment": True,
            "commitment": {
                "committed_at_tick": commitment.committed_at_tick,
                "decision_digest": commitment.decision_digest,
                "token_sha256": commitment.token_sha256,
            },
            "physics_identity": self.contract.physics_identity,
            "state": copy.deepcopy(self._truth),
        }


class Simulator3DEnvironment:
    """Deterministic room physics with viewpoint-dependent, identity-free rendering."""

    room_bounds = ((-5.0, 5.0), (-5.0, 5.0), (0.0, 3.0))

    def __init__(self, seed: int, *, render_variant: str = "matte") -> None:
        self.seed = int(seed)
        self.render_variant = render_variant
        self.body = Simulator3DBodyContract()
        self.contract = EnvironmentContract(
            identity="room-3d-simulator-v5",
            state_schema="substrate.v5.room-state/1",
            sensors=self.body.sensors,
            actions=self.body.actuators,
            deterministic_transitions=True,
            seeded=True,
            action_costs=self.body.action_cost,
            failure_modes=self.body.failure_modes,
            hidden_state_schema="substrate.v5.room-physics-private/1",
            oracle_schema="substrate.v5.room-oracle/1",
            checkpoint_schema=self.body.checkpoint_format,
            reset_semantics="restore exact seed-derived geometry and body state",
            render_identity=f"room-render:{render_variant}",
            physics_identity="sphere-aabb-room-physics-v1",
        )
        self._initial_truth = self._make_truth()
        self._truth: dict[str, Any] = {}
        self._tick = 0
        self._energy = self.body.energy_budget
        self._depth_requested = False
        self._oracle_commitments: dict[str, CommitmentToken] = {}
        self.reset()

    def _make_truth(self) -> dict[str, Any]:
        shifts = [
            (int(_digest(self.seed, index, axis)[:4], 16) % 101 - 50) / 200.0
            for index in range(3)
            for axis in ("x",)
        ]
        return {
            "body_position": [0.0, -3.5, 0.9],
            "camera_yaw_degrees": 0.0,
            "objects": [
                {
                    "physical_id": "physics-object-0",
                    "position": [-1.2 + shifts[0], 0.2, 0.55],
                    "radius": 0.35,
                    "appearance": [0.9, 0.2, 0.1],
                    "velocity": [0.0, 0.0, 0.0],
                },
                {
                    "physical_id": "physics-object-1",
                    "position": [0.9 + shifts[1], 1.0, 0.45],
                    "radius": 0.28,
                    "appearance": [0.1, 0.7, 0.4],
                    "velocity": [0.0, 0.0, 0.0],
                },
                {
                    "physical_id": "physics-object-2",
                    "position": [1.8 + shifts[2], -0.1, 0.70],
                    "radius": 0.42,
                    "appearance": [0.2, 0.3, 0.9],
                    "velocity": [0.0, 0.0, 0.0],
                },
            ],
            "contact_events": [],
        }

    def reset(self) -> dict[str, Any]:
        self._truth = copy.deepcopy(self._initial_truth)
        self._tick = 0
        self._energy = self.body.energy_budget
        self._depth_requested = False
        self._oracle_commitments.clear()
        return self.observe()

    def _physics_digest(self) -> str:
        return _digest(self.contract.physics_identity, self._truth, self._tick, self._energy)

    @staticmethod
    def _rotate_z(point: Vec3, degrees: float) -> Vec3:
        angle = math.radians(degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        return (
            cosine * point[0] - sine * point[1],
            sine * point[0] + cosine * point[1],
            point[2],
        )

    def _camera_coordinates(self, point: Vec3) -> Vec3:
        body = self._truth["body_position"]
        relative = (point[0] - body[0], point[1] - body[1], point[2] - body[2])
        return self._rotate_z(relative, -self._truth["camera_yaw_degrees"])

    def render(self) -> dict[str, Any]:
        visible = []
        for physical in self._truth["objects"]:
            camera = self._camera_coordinates(tuple(physical["position"]))
            forward = camera[1]
            if forward <= 0.05:
                continue
            angle = abs(math.degrees(math.atan2(camera[0], forward)))
            if angle > self.body.field_of_view_degrees / 2.0:
                continue
            scale = 180.0 / forward
            style_jitter = (int(_digest(self.seed, self.render_variant, physical["physical_id"])[:4], 16) % 21 - 10) / 255.0
            visible.append(
                {
                    "screen_center": (400.0 + camera[0] * scale, 300.0 - camera[2] * scale),
                    "apparent_radius": physical["radius"] * scale,
                    "appearance": tuple(
                        max(0.0, min(1.0, channel + style_jitter)) for channel in physical["appearance"]
                    ),
                    "depth": forward if self._depth_requested else None,
                },
            )
        # Screen position gives association evidence, but no private or stable object ID.
        return {
            "render_identity": self.contract.render_identity,
            "camera_frame": "camera",
            "camera_yaw_degrees": self._truth["camera_yaw_degrees"],
            "detections": sorted(visible, key=lambda row: (row["screen_center"][0], row["screen_center"][1])),
            "depth_available": self._depth_requested,
        }

    def observe(self) -> dict[str, Any]:
        observation = {
            "environment": self.contract.identity,
            "sequence": self._tick,
            "timestamp": self._tick * 0.1,
            "coordinate_frames": ("world", "body", "camera"),
            "modalities": tuple(modality.value for modality in self.contract.sensors),
            "render": self.render(),
            "body": {
                "position": tuple(self._truth["body_position"]),
                "camera_yaw_degrees": self._truth["camera_yaw_degrees"],
                "energy_remaining": self._energy,
                "reach": self.body.reach,
                "available_actions": self.contract.actions,
            },
            "recent_contact": bool(self._truth["contact_events"][-1:]),
        }
        _assert_no_hidden_ids(observation)
        return observation

    def _inside_room(self, position: Vec3) -> bool:
        return all(lower <= position[index] <= upper for index, (lower, upper) in enumerate(self.room_bounds))

    def _nearest_object(self, position: Vec3) -> tuple[dict[str, Any] | None, float]:
        rows = [
            (
                math.dist(position, tuple(physical["position"])),
                physical,
            )
            for physical in self._truth["objects"]
        ]
        if not rows:
            return None, math.inf
        distance, physical = min(rows, key=lambda row: row[0])
        return physical, distance

    def step(self, action: str, parameters: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ActionReceipt]:
        parameters = dict(parameters or {})
        self.body.check_action(action)
        before = self._physics_digest()
        cost = self.contract.action_costs[action]
        success = True
        failure = None
        if self._energy < cost:
            success = False
            failure = "energy_exhausted"
        elif action == "move_body":
            delta = tuple(float(parameters.get(axis, 0.0)) for axis in ("dx", "dy", "dz"))
            current = tuple(self._truth["body_position"])
            destination = tuple(current[index] + delta[index] for index in range(3))
            if not self._inside_room(destination):
                success = False
                failure = "outside_room"
            else:
                physical, distance = self._nearest_object(destination)  # noqa: F841
                if distance < 0.45:
                    success = False
                    failure = "collision"
                else:
                    self._truth["body_position"] = list(destination)
        elif action == "rotate_view":
            degrees = float(parameters.get("degrees", 0.0))
            self._truth["camera_yaw_degrees"] = (self._truth["camera_yaw_degrees"] + degrees) % 360.0
        elif action == "reach":
            point = tuple(float(parameters[axis]) for axis in ("x", "y", "z"))
            physical, distance = self._nearest_object(point)
            body_distance = math.dist(tuple(self._truth["body_position"]), point)
            if physical is None or distance > physical["radius"]:
                success = False
                failure = "no_object_at_position"
            elif body_distance > self.body.reach:
                success = False
                failure = "out_of_reach"
            else:
                self._truth["contact_events"].append(
                    {"tick": self._tick, "position": list(point), "impulse": float(parameters.get("impulse", 0.0))},
                )
        elif action == "request_depth":
            self._depth_requested = True
        elif action in {"inspect", "wait"}:
            pass
        if success:
            self._energy -= cost
        self._tick += 1
        after = self._physics_digest()
        receipt = ActionReceipt(
            environment_identity=self.contract.identity,
            action=action,
            parameters=parameters,
            cost=cost if success else 0.0,
            timestamp=self._tick * 0.1,
            success=success,
            failure=failure,
            state_digest_before=before,
            state_digest_after=after,
        )
        return self.observe(), receipt

    def checkpoint(self) -> dict[str, Any]:
        body = {
            "schema": self.contract.checkpoint_schema,
            "environment": self.contract.identity,
            "seed": self.seed,
            "render_variant": self.render_variant,
            "tick": self._tick,
            "energy": self._energy,
            "depth_requested": self._depth_requested,
            "physics": copy.deepcopy(self._truth),
        }
        body["digest"] = _digest(body)
        return body

    def restore(self, checkpoint: Mapping[str, Any]) -> Simulator3DEnvironment:
        supplied = dict(checkpoint)
        digest = supplied.pop("digest", None)
        if digest != _digest(supplied):
            raise EnvironmentError("3D checkpoint digest mismatch")
        if supplied["schema"] != self.contract.checkpoint_schema or supplied["environment"] != self.contract.identity:
            raise EnvironmentError("3D checkpoint identity mismatch")
        if supplied["seed"] != self.seed or supplied["render_variant"] != self.render_variant:
            raise EnvironmentError("3D checkpoint belongs to another environment instance")
        self._tick = int(supplied["tick"])
        self._energy = float(supplied["energy"])
        self._depth_requested = bool(supplied["depth_requested"])
        self._truth = copy.deepcopy(supplied["physics"])
        self._oracle_commitments.clear()
        return self

    def commit_decision(self, decision: Mapping[str, Any]) -> CommitmentToken:
        normalized = dict(decision)
        if not normalized:
            raise EnvironmentError("oracle commitment requires a nonempty decision")
        _assert_no_hidden_ids(normalized)
        state_digest = self._physics_digest()
        decision_digest = _digest(normalized)
        token_sha256 = _digest(
            "oracle-commitment",
            self.contract.identity,
            self.seed,
            self.render_variant,
            self._tick,
            state_digest,
            decision_digest,
        )
        token = CommitmentToken(
            environment_identity=self.contract.identity,
            committed_at_tick=self._tick,
            state_digest=state_digest,
            decision_digest=decision_digest,
            token_sha256=token_sha256,
        )
        self._oracle_commitments[token_sha256] = token
        return token

    def reveal_physics_after_commitment(
        self,
        commitment: CommitmentToken | None = None,
    ) -> dict[str, Any]:
        if commitment is None:
            raise EnvironmentError("oracle reveal requires a prior commitment token")
        issued = self._oracle_commitments.pop(commitment.token_sha256, None)
        if issued is not commitment or commitment.environment_identity != self.contract.identity:
            raise EnvironmentError("oracle commitment token is invalid, foreign, or already consumed")
        return {
            "revealed_after_commitment": True,
            "commitment": {
                "committed_at_tick": commitment.committed_at_tick,
                "decision_digest": commitment.decision_digest,
                "token_sha256": commitment.token_sha256,
            },
            "physics_identity": self.contract.physics_identity,
            "state": copy.deepcopy(self._truth),
        }


def deterministic_environment_fixture(seed: int = 5517) -> dict[str, Any]:
    """Exercise replay determinism and render/physics separation for both bodies."""

    desktop_a, desktop_b = DesktopEnvironment(seed), DesktopEnvironment(seed)
    desktop_trace_a = [
        desktop_a.step("move_pointer", {"x": 150.0, "y": 160.0})[0],
        desktop_a.step("click")[0],
    ]
    desktop_trace_b = [
        desktop_b.step("move_pointer", {"x": 150.0, "y": 160.0})[0],
        desktop_b.step("click")[0],
    ]
    room_a, room_b = Simulator3DEnvironment(seed), Simulator3DEnvironment(seed)
    room_trace_a = [room_a.step("rotate_view", {"degrees": 15.0})[0], room_a.step("request_depth")[0]]
    room_trace_b = [room_b.step("rotate_view", {"degrees": 15.0})[0], room_b.step("request_depth")[0]]
    room_other_render = Simulator3DEnvironment(seed, render_variant="wireframe")
    room_other_render.step("rotate_view", {"degrees": 15.0})
    room_other_render.step("request_depth")
    room_a_commitment = room_a.commit_decision({"prediction": "physics unchanged"})
    other_render_commitment = room_other_render.commit_decision({"prediction": "physics unchanged"})
    same_physics = (
        room_a.reveal_physics_after_commitment(room_a_commitment)["state"]
        == room_other_render.reveal_physics_after_commitment(other_render_commitment)["state"]
    )
    different_render = room_a.render()["render_identity"] != room_other_render.render()["render_identity"]
    return {
        "desktop_deterministic": desktop_trace_a == desktop_trace_b and desktop_a.checkpoint() == desktop_b.checkpoint(),
        "room_deterministic": room_trace_a == room_trace_b and room_a.checkpoint() == room_b.checkpoint(),
        "physics_independent_of_render": same_physics,
        "render_variants_distinct": different_render,
        "observations_hide_physical_ids": "physical_id"
        not in json.dumps((desktop_trace_a, room_trace_a), sort_keys=True),
    }


def validate_coordinate_frame(frame: str, allowed: tuple[str, ...]) -> None:
    """Shared boundary check for body adapters."""

    if frame not in allowed:
        raise SensoriumError(f"coordinate frame {frame!r} is not in the body contract")


__all__ = [
    "ActionReceipt",
    "BodyContract",
    "CommitmentToken",
    "DesktopBodyContract",
    "DesktopEnvironment",
    "EnvironmentContract",
    "EnvironmentError",
    "Simulator3DBodyContract",
    "Simulator3DEnvironment",
    "deterministic_environment_fixture",
    "validate_coordinate_frame",
]
