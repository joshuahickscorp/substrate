"""Native deterministic 3D scene graph + z-buffer renderer for Odyssey Frontier G.

Primary Frontier-G backend.  Pure Python + stdlib; no Blender dependency.
Blender remains an optional higher-fidelity comparison path elsewhere.

Invariants
----------
* Real geometry: meshes with vertices/triangles, per-object transforms, unique ids.
* Explicit camera intrinsics (fx, fy, cx, cy, width, height) and extrinsics (pose).
* First-class depth / z-buffer output; near surfaces occlude far ones.
* Deterministic regeneration from a pinned seed: same seed → byte-identical
  RGB PNG and depth buffer.
* Evaluator-only scene graph and answer state are physically separate from
  candidate-visible sensor outputs (RGB, depth, mesh-inspect summaries).
* No network, no sudo, sandbox-portable.

Version pin: RENDERER_ID / RENDERER_VERSION appear on every receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Identity / version (honest recording)
# ---------------------------------------------------------------------------

RENDERER_ID = "substrate_spatial3d"
RENDERER_VERSION = "1.0.0"
SCENE_SCHEMA = "SUBSTRATE_SPATIAL3D_SCENE/v1"
SEED_SCHEMA = "SUBSTRATE_SPATIAL3D_SEED/v1"
ANSWER_SCHEMA = "SUBSTRATE_SPATIAL3D_ANSWER_STATE/v1"
CANARY_SCHEMA = "SUBSTRATE_SPATIAL3D_CANARY/v1"
DEPTH_MEDIA = "application/x-substrate-depth-f32"
META_MEDIA = "application/x-substrate-tool-result+json"

# Depth encoding: positive eye-space distance along camera forward; background = +inf sentinel.
DEPTH_FAR_SENTINEL = 1.0e6
DEFAULT_WIDTH = 96
DEFAULT_HEIGHT = 96
DEFAULT_FOV_DEG = 50.0
DEFAULT_NEAR = 0.05
DEFAULT_FAR = 100.0


class Spatial3DError(RuntimeError):
    """Raised for invalid scene/camera/object operations."""


# ---------------------------------------------------------------------------
# Math primitives (row-vector convention; column-major storage as nested tuples)
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Sequence[float]) -> tuple[float, float, float]:
    n = _norm(v)
    if n < 1e-12:
        raise Spatial3DError("cannot normalize near-zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v: Sequence[float], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def _mat4_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j]
    return out


def _mat4_vec4(m: Sequence[Sequence[float]], v: Sequence[float]) -> tuple[float, float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    )


def _mat4_translate(t: Sequence[float]) -> list[list[float]]:
    m = _mat4_identity()
    m[0][3] = float(t[0])
    m[1][3] = float(t[1])
    m[2][3] = float(t[2])
    return m


def _mat4_scale(s: Sequence[float]) -> list[list[float]]:
    m = _mat4_identity()
    m[0][0] = float(s[0])
    m[1][1] = float(s[1])
    m[2][2] = float(s[2])
    return m


def _mat4_rotate_xyz(rx: float, ry: float, rz: float) -> list[list[float]]:
    """Intrinsic XYZ Euler rotations (radians)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = [[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]]
    ry_m = [[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]]
    rz_m = [[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return _mat4_mul(rz_m, _mat4_mul(ry_m, rx_m))


def _look_at(
    eye: Sequence[float],
    target: Sequence[float],
    up: Sequence[float] = (0.0, 1.0, 0.0),
) -> list[list[float]]:
    """World-to-camera matrix (OpenGL-ish: camera looks down -Z in camera space)."""
    forward = _normalize(_sub(target, eye))  # world direction camera faces
    # Camera basis: right, up, -forward (so -Z is look direction)
    z_axis = _scale(forward, -1.0)  # camera +Z points opposite look
    x_axis = _normalize(_cross(up, z_axis))
    y_axis = _cross(z_axis, x_axis)
    # Rotation part: rows are camera axes
    rot = [
        [x_axis[0], x_axis[1], x_axis[2], 0.0],
        [y_axis[0], y_axis[1], y_axis[2], 0.0],
        [z_axis[0], z_axis[1], z_axis[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # Translate by -eye in camera space
    tx = -_dot(x_axis, eye)
    ty = -_dot(y_axis, eye)
    tz = -_dot(z_axis, eye)
    rot[0][3] = tx
    rot[1][3] = ty
    rot[2][3] = tz
    return rot


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> list[list[float]]:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = [[0.0] * 4 for _ in range(4)]
    m[0][0] = f / aspect
    m[1][1] = f
    m[2][2] = (far + near) / (near - far)
    m[2][3] = (2.0 * far * near) / (near - far)
    m[3][2] = -1.0
    return m


# ---------------------------------------------------------------------------
# Mesh primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mesh:
    """Triangle mesh in object-local space.  Vertices are (x,y,z); indices are triples."""

    name: str
    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]

    def transformed(self, matrix: Sequence[Sequence[float]]) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
        """Return world-space triangles as 3-tuples of positions."""
        world: list[tuple[float, float, float]] = []
        for vx, vy, vz in self.vertices:
            x, y, z, w = _mat4_vec4(matrix, (vx, vy, vz, 1.0))
            if abs(w) > 1e-12:
                inv = 1.0 / w
                world.append((x * inv, y * inv, z * inv))
            else:
                world.append((x, y, z))
        tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
        for i0, i1, i2 in self.triangles:
            tris.append((world[i0], world[i1], world[i2]))
        return tris


def mesh_box() -> Mesh:
    # Unit cube centered at origin, half-extent 0.5 → side length 1.
    v = (
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    )
    t = (
        (0, 1, 2),
        (0, 2, 3),  # -Z
        (4, 6, 5),
        (4, 7, 6),  # +Z
        (0, 4, 5),
        (0, 5, 1),  # -Y
        (2, 6, 7),
        (2, 7, 3),  # +Y
        (0, 3, 7),
        (0, 7, 4),  # -X
        (1, 5, 6),
        (1, 6, 2),  # +X
    )
    return Mesh(name="box", vertices=v, triangles=t)


def mesh_plane() -> Mesh:
    v = (
        (-0.5, 0.0, -0.5),
        (0.5, 0.0, -0.5),
        (0.5, 0.0, 0.5),
        (-0.5, 0.0, 0.5),
    )
    t = ((0, 1, 2), (0, 2, 3))
    return Mesh(name="plane", vertices=v, triangles=t)


def mesh_wedge() -> Mesh:
    """Right triangular prism (wedge) for asymmetric silhouette tests."""
    v = (
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, 0.5, 0.5),
    )
    t = (
        (0, 1, 2),
        (0, 2, 3),  # bottom
        (0, 4, 1),  # front slope edge
        (3, 2, 5),  # back
        (0, 3, 5),
        (0, 5, 4),  # left
        (1, 4, 5),
        (1, 5, 2),  # slope
    )
    return Mesh(name="wedge", vertices=v, triangles=t)


def mesh_pyramid() -> Mesh:
    v = (
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        (0.0, 0.5, 0.0),
    )
    t = (
        (0, 1, 2),
        (0, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    )
    return Mesh(name="pyramid", vertices=v, triangles=t)


_MESH_LIBRARY: dict[str, Mesh] = {
    "box": mesh_box(),
    "cube": mesh_box(),
    "plane": mesh_plane(),
    "wedge": mesh_wedge(),
    "pyramid": mesh_pyramid(),
}


def get_mesh(shape: str) -> Mesh:
    key = shape.strip().lower()
    if key not in _MESH_LIBRARY:
        raise Spatial3DError(f"unknown mesh shape {shape!r}")
    return _MESH_LIBRARY[key]


# ---------------------------------------------------------------------------
# Scene graph
# ---------------------------------------------------------------------------


@dataclass
class SceneObject:
    object_id: str
    shape: str
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotation_xyz: tuple[float, float, float]  # radians
    color: tuple[int, int, int]  # 0..255
    support_parent: str | None = None  # optional containment/support relation

    def world_matrix(self) -> list[list[float]]:
        return _mat4_mul(
            _mat4_translate(self.position),
            _mat4_mul(_mat4_rotate_xyz(*self.rotation_xyz), _mat4_scale(self.scale)),
        )

    def to_public(self) -> dict[str, Any]:
        """Candidate-safe identity listing — no pose, no color ground truth required."""
        return {"object_id": self.object_id, "shape": self.shape}

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "shape": self.shape,
            "position": list(self.position),
            "scale": list(self.scale),
            "rotation_xyz": list(self.rotation_xyz),
            "color": list(self.color),
            "support_parent": self.support_parent,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneObject:
        return cls(
            object_id=str(data["object_id"]),
            shape=str(data["shape"]),
            position=_as_vec3(data["position"]),
            scale=_as_vec3(data.get("scale", (1.0, 1.0, 1.0))),
            rotation_xyz=_as_vec3(data.get("rotation_xyz", (0.0, 0.0, 0.0))),
            color=_as_rgb(data.get("color", (200, 200, 200))),
            support_parent=(str(data["support_parent"]) if data.get("support_parent") is not None else None),
        )


@dataclass
class Camera:
    camera_id: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_y_deg: float = DEFAULT_FOV_DEG
    near: float = DEFAULT_NEAR
    far: float = DEFAULT_FAR

    def world_to_camera(self) -> list[list[float]]:
        return _look_at(self.position, self.look_at, self.up)

    def intrinsics(self, width: int, height: int) -> dict[str, float]:
        aspect = width / height
        f = 1.0 / math.tan(math.radians(self.fov_y_deg) * 0.5)
        # Pixel focal length: NDC y maps [-1,1] → [0,H]; fy = f * (H/2)
        fy = f * (height * 0.5)
        fx = (f / aspect) * (width * 0.5)
        return {
            "fx": fx,
            "fy": fy,
            "cx": width * 0.5,
            "cy": height * 0.5,
            "width": float(width),
            "height": float(height),
            "fov_y_deg": self.fov_y_deg,
            "near": self.near,
            "far": self.far,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "position": list(self.position),
            "look_at": list(self.look_at),
            "up": list(self.up),
            "fov_y_deg": self.fov_y_deg,
            "near": self.near,
            "far": self.far,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Camera:
        return cls(
            camera_id=str(data["camera_id"]),
            position=_as_vec3(data["position"]),
            look_at=_as_vec3(data["look_at"]),
            up=_as_vec3(data.get("up", (0.0, 1.0, 0.0))),
            fov_y_deg=float(data.get("fov_y_deg", DEFAULT_FOV_DEG)),
            near=float(data.get("near", DEFAULT_NEAR)),
            far=float(data.get("far", DEFAULT_FAR)),
        )


@dataclass
class Scene:
    """Full scene graph.  Evaluator-only when serialized under the evaluator mount."""

    seed_id: str
    seed: int
    width: int
    height: int
    objects: dict[str, SceneObject] = field(default_factory=dict)
    cameras: dict[str, Camera] = field(default_factory=dict)
    active_camera: str = ""
    background: tuple[int, int, int] = (24, 28, 36)
    light_dir: tuple[float, float, float] = _normalize((0.4, 0.85, 0.35))
    ambient: float = 0.28
    diffuse: float = 0.72
    canary_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_object(self, obj: SceneObject) -> None:
        if obj.object_id in self.objects:
            raise Spatial3DError(f"duplicate object_id {obj.object_id!r}")
        self.objects[obj.object_id] = obj

    def add_camera(self, cam: Camera) -> None:
        if cam.camera_id in self.cameras:
            raise Spatial3DError(f"duplicate camera_id {cam.camera_id!r}")
        self.cameras[cam.camera_id] = cam
        if not self.active_camera:
            self.active_camera = cam.camera_id

    def set_active_camera(self, camera_id: str) -> None:
        if camera_id not in self.cameras:
            raise Spatial3DError(f"unknown camera {camera_id!r}")
        self.active_camera = camera_id

    def move_object(
        self,
        object_id: str,
        *,
        translation: Sequence[float] | None = None,
        position: Sequence[float] | None = None,
        rotation_delta: Sequence[float] | None = None,
    ) -> None:
        if object_id not in self.objects:
            raise Spatial3DError(f"unknown object {object_id!r}")
        obj = self.objects[object_id]
        if position is not None:
            obj.position = _as_vec3(position)
        if translation is not None:
            t = _as_vec3(translation)
            obj.position = (obj.position[0] + t[0], obj.position[1] + t[1], obj.position[2] + t[2])
        if rotation_delta is not None:
            r = _as_vec3(rotation_delta)
            obj.rotation_xyz = (
                obj.rotation_xyz[0] + r[0],
                obj.rotation_xyz[1] + r[1],
                obj.rotation_xyz[2] + r[2],
            )

    def public_object_catalog(self) -> list[dict[str, Any]]:
        """Candidate-visible identities only (no poses)."""
        return [self.objects[k].to_public() for k in sorted(self.objects)]

    def public_camera_catalog(self) -> list[dict[str, Any]]:
        """Candidate-visible camera ids and intrinsics (no extrinsics poses)."""
        rows = []
        for cid in sorted(self.cameras):
            cam = self.cameras[cid]
            rows.append(
                {
                    "camera_id": cid,
                    "intrinsics": cam.intrinsics(self.width, self.height),
                    "active": cid == self.active_camera,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENE_SCHEMA,
            "renderer": RENDERER_ID,
            "renderer_version": RENDERER_VERSION,
            "seed_id": self.seed_id,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "active_camera": self.active_camera,
            "background": list(self.background),
            "light_dir": list(self.light_dir),
            "ambient": self.ambient,
            "diffuse": self.diffuse,
            "canary_id": self.canary_id,
            "metadata": dict(self.metadata),
            "objects": [self.objects[k].to_dict() for k in sorted(self.objects)],
            "cameras": [self.cameras[k].to_dict() for k in sorted(self.cameras)],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scene:
        scene = cls(
            seed_id=str(data["seed_id"]),
            seed=int(data["seed"]),
            width=int(data.get("width", DEFAULT_WIDTH)),
            height=int(data.get("height", DEFAULT_HEIGHT)),
            background=_as_rgb(data.get("background", (24, 28, 36))),
            light_dir=_normalize(_as_vec3(data.get("light_dir", (0.4, 0.85, 0.35)))),
            ambient=float(data.get("ambient", 0.28)),
            diffuse=float(data.get("diffuse", 0.72)),
            canary_id=(str(data["canary_id"]) if data.get("canary_id") is not None else None),
            metadata=dict(data.get("metadata") or {}),
        )
        for row in data.get("objects") or []:
            scene.add_object(SceneObject.from_dict(row))
        for row in data.get("cameras") or []:
            scene.add_camera(Camera.from_dict(row))
        active = data.get("active_camera") or scene.active_camera
        if active:
            scene.set_active_camera(str(active))
        return scene

    def canonical_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def _as_vec3(value: object) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise Spatial3DError("expected a length-3 vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_rgb(value: object) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise Spatial3DError("expected an RGB triple")
    return (
        int(_clamp(int(value[0]), 0, 255)),
        int(_clamp(int(value[1]), 0, 255)),
        int(_clamp(int(value[2]), 0, 255)),
    )


# ---------------------------------------------------------------------------
# PNG writer (RGBA8, deterministic zlib level 9)
# ---------------------------------------------------------------------------


def write_png_rgba(path: Path, width: int, height: int, pixels: bytes) -> None:
    """Write an RGBA8 PNG.  ``pixels`` is length width*height*4, row-major top-down."""
    if len(pixels) != width * height * 4:
        raise Spatial3DError("pixel buffer size mismatch")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw_rows = b"".join(b"\x00" + pixels[row * width * 4 : (row + 1) * width * 4] for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw_rows, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def write_depth_f32(path: Path, width: int, height: int, depth: Sequence[float]) -> None:
    """Little-endian float32 depth buffer, row-major top-down.  Background = DEPTH_FAR_SENTINEL.

    Not cache-admissible as raw bytes (product cache refuses opaque binary).  Used for
    evaluator/local evidence; the admitted sensor form is ``write_depth_u16_png``.
    """
    if len(depth) != width * height:
        raise Spatial3DError("depth buffer size mismatch")
    path.write_bytes(struct.pack(f"<{width * height}f", *depth))


def depth_f32_bytes(width: int, height: int, depth: Sequence[float]) -> bytes:
    if len(depth) != width * height:
        raise Spatial3DError("depth buffer size mismatch")
    return struct.pack(f"<{width * height}f", *depth)


# Depth PNG encoding: millimeters as uint16, 0 = background/invalid, max 65535 mm (~65 m).
DEPTH_U16_SCALE_MM = 1000.0
DEPTH_U16_MAX_MM = 65535


def encode_depth_u16(depth: Sequence[float]) -> list[int]:
    """Convert eye-space meters → uint16 millimeters (0 = invalid/background)."""
    out: list[int] = []
    for d in depth:
        if d >= DEPTH_FAR_SENTINEL * 0.5 or d <= 0.0:
            out.append(0)
        else:
            mm = int(round(d * DEPTH_U16_SCALE_MM))
            out.append(1 if mm < 1 else (DEPTH_U16_MAX_MM if mm > DEPTH_U16_MAX_MM else mm))
    return out


def write_depth_u16_png(path: Path, width: int, height: int, depth: Sequence[float]) -> bytes:
    """Write a 16-bit greyscale PNG of metric depth (millimeters).  Returns file bytes.

    This is the cache-admissible first-class depth artifact (sniffed as image/png).
    """
    if len(depth) != width * height:
        raise Spatial3DError("depth buffer size mismatch")
    samples = encode_depth_u16(depth)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    # 16-bit greyscale, big-endian samples per PNG spec; filter byte 0 per row.
    rows = []
    for row in range(height):
        row_bytes = bytearray([0])
        base = row * width
        for col in range(width):
            row_bytes.extend(struct.pack(">H", samples[base + col]))
        rows.append(bytes(row_bytes))
    raw = b"".join(rows)
    # color type 0 = greyscale, bit depth 16
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)
    return png


def depth_to_vis_png(path: Path, width: int, height: int, depth: Sequence[float]) -> None:
    """Greyscale visualization of finite depths (for human inspection only)."""
    finite = [d for d in depth if d < DEPTH_FAR_SENTINEL * 0.5]
    if not finite:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = min(finite), max(finite)
        if hi - lo < 1e-9:
            hi = lo + 1.0
    pixels = bytearray()
    span = hi - lo
    for d in depth:
        if d >= DEPTH_FAR_SENTINEL * 0.5:
            pixels.extend((8, 8, 12, 255))
        else:
            t = (d - lo) / span
            g = int(_clamp(255.0 * (1.0 - t), 0.0, 255.0))
            pixels.extend((g, g, g, 255))
    write_png_rgba(path, width, height, bytes(pixels))


# ---------------------------------------------------------------------------
# Z-buffer rasterizer
# ---------------------------------------------------------------------------


@dataclass
class RenderResult:
    rgb_png: bytes
    depth_f32: bytes
    depth_u16_png: bytes
    depth_vis_png: bytes
    width: int
    height: int
    camera_id: str
    seed_id: str
    scene_digest: str
    visible_object_ids: list[str]
    pixel_object_coverage: dict[str, int]
    renderer: str = RENDERER_ID
    renderer_version: str = RENDERER_VERSION


def render_scene(scene: Scene, *, camera_id: str | None = None) -> RenderResult:
    """Rasterize RGB + depth with a z-buffer.  Deterministic for a fixed scene state."""
    cid = camera_id or scene.active_camera
    if not cid or cid not in scene.cameras:
        raise Spatial3DError("scene has no active camera")
    cam = scene.cameras[cid]
    w, h = scene.width, scene.height
    if w < 8 or h < 8 or w > 1024 or h > 1024:
        raise Spatial3DError("unsupported render resolution")

    view = cam.world_to_camera()
    proj = _perspective(cam.fov_y_deg, w / h, cam.near, cam.far)
    vp = _mat4_mul(proj, view)

    # Buffers
    bg = scene.background
    rgba = bytearray([bg[0], bg[1], bg[2], 255] * (w * h))
    zbuf = [DEPTH_FAR_SENTINEL] * (w * h)
    idbuf = [-1] * (w * h)  # object index or -1
    object_ids = sorted(scene.objects)
    id_of = {oid: i for i, oid in enumerate(object_ids)}
    colors = [scene.objects[oid].color for oid in object_ids]
    light = scene.light_dir
    ambient = scene.ambient
    diffuse = scene.diffuse

    # Gather triangles sorted by object id for determinism
    for oid in object_ids:
        obj = scene.objects[oid]
        mesh = get_mesh(obj.shape)
        matrix = obj.world_matrix()
        world_tris = mesh.transformed(matrix)
        oid_index = id_of[oid]
        base_color = colors[oid_index]
        for tri in world_tris:
            _raster_triangle(
                tri,
                vp=vp,
                view=view,
                width=w,
                height=h,
                near=cam.near,
                far=cam.far,
                rgba=rgba,
                zbuf=zbuf,
                idbuf=idbuf,
                oid_index=oid_index,
                base_color=base_color,
                light=light,
                ambient=ambient,
                diffuse=diffuse,
            )

    coverage: dict[str, int] = {oid: 0 for oid in object_ids}
    for idx in idbuf:
        if idx >= 0:
            coverage[object_ids[idx]] += 1
    visible = [oid for oid in object_ids if coverage[oid] > 0]

    # Build PNG / depth bytes via temp files (deterministic zlib PNG writer).
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        rgb_path = tdir / "rgb.png"
        depth_path = tdir / "depth.f32"
        depth_u16_path = tdir / "depth_u16.png"
        vis_path = tdir / "depth_vis.png"
        write_png_rgba(rgb_path, w, h, bytes(rgba))
        write_depth_f32(depth_path, w, h, zbuf)
        depth_u16_png = write_depth_u16_png(depth_u16_path, w, h, zbuf)
        depth_to_vis_png(vis_path, w, h, zbuf)
        rgb_png = rgb_path.read_bytes()
        depth_f32 = depth_path.read_bytes()
        depth_vis = vis_path.read_bytes()

    return RenderResult(
        rgb_png=rgb_png,
        depth_f32=depth_f32,
        depth_u16_png=depth_u16_png,
        depth_vis_png=depth_vis,
        width=w,
        height=h,
        camera_id=cid,
        seed_id=scene.seed_id,
        scene_digest=scene.canonical_digest(),
        visible_object_ids=visible,
        pixel_object_coverage=coverage,
    )


def _raster_triangle(
    tri: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    *,
    vp: Sequence[Sequence[float]],
    view: Sequence[Sequence[float]],
    width: int,
    height: int,
    near: float,
    far: float,
    rgba: bytearray,
    zbuf: list[float],
    idbuf: list[int],
    oid_index: int,
    base_color: tuple[int, int, int],
    light: Sequence[float],
    ambient: float,
    diffuse: float,
) -> None:
    # Project three vertices to clip → NDC → screen; keep eye-space depth for z-buffer.
    screen: list[tuple[float, float, float, float]] = []  # sx, sy, eye_z, inv_w
    eye_pts: list[tuple[float, float, float]] = []
    for p in tri:
        ex, ey, ez, _ = _mat4_vec4(view, (p[0], p[1], p[2], 1.0))
        # OpenGL camera looks down -Z; positive depth distance = -ez when ez < 0
        eye_depth = -ez
        if eye_depth <= near or eye_depth >= far:
            # Still project; clipping handled per-pixel by depth range
            pass
        cx, cy, cz, cw = _mat4_vec4(vp, (p[0], p[1], p[2], 1.0))
        if abs(cw) < 1e-12:
            return
        inv_w = 1.0 / cw
        ndc_x = cx * inv_w
        ndc_y = cy * inv_w
        sx = (ndc_x * 0.5 + 0.5) * width
        sy = (1.0 - (ndc_y * 0.5 + 0.5)) * height  # top-down rows
        screen.append((sx, sy, eye_depth, inv_w))
        eye_pts.append((ex, ey, ez))

    # Back-face cull in eye space (optional but stable)
    e0 = _sub(eye_pts[1], eye_pts[0])
    e1 = _sub(eye_pts[2], eye_pts[0])
    normal_eye = _cross(e0, e1)
    # Camera looks down -Z; front faces have normal pointing toward camera (+dot with -forward = -nz)
    if normal_eye[2] <= 1e-12:
        # Facing away or edge-on
        pass  # keep double-sided for robustness on thin objects
    nlen = _norm(normal_eye)
    if nlen < 1e-12:
        return
    # Lighting uses world-space normal from the world triangle.
    w0 = _sub(tri[1], tri[0])
    w1 = _sub(tri[2], tri[0])
    wn = _cross(w0, w1)
    wn_len = _norm(wn)
    if wn_len < 1e-12:
        return
    wn_u = (wn[0] / wn_len, wn[1] / wn_len, wn[2] / wn_len)
    ndl = max(0.0, _dot(wn_u, light))
    shade = ambient + diffuse * ndl
    r = int(_clamp(base_color[0] * shade, 0, 255))
    g = int(_clamp(base_color[1] * shade, 0, 255))
    b = int(_clamp(base_color[2] * shade, 0, 255))

    (x0, y0, z0, _), (x1, y1, z1, _), (x2, y2, z2, _) = screen
    min_x = max(0, int(math.floor(min(x0, x1, x2))))
    max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
    min_y = max(0, int(math.floor(min(y0, y1, y2))))
    max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
    if min_x > max_x or min_y > max_y:
        return

    area = _edge(x0, y0, x1, y1, x2, y2)
    if abs(area) < 1e-12:
        return
    inv_area = 1.0 / area

    for py in range(min_y, max_y + 1):
        cy = py + 0.5
        for px in range(min_x, max_x + 1):
            cx = px + 0.5
            w0b = _edge(x1, y1, x2, y2, cx, cy) * inv_area
            w1b = _edge(x2, y2, x0, y0, cx, cy) * inv_area
            w2b = _edge(x0, y0, x1, y1, cx, cy) * inv_area
            if w0b < 0.0 or w1b < 0.0 or w2b < 0.0:
                # Also accept the opposite winding (double-sided)
                if w0b > 0.0 or w1b > 0.0 or w2b > 0.0:
                    continue
                # all non-positive: still inside opposite winding if sums to 1
                if not (w0b <= 0.0 and w1b <= 0.0 and w2b <= 0.0):
                    continue
            # Barycentric depth (eye-space distance)
            depth = w0b * z0 + w1b * z1 + w2b * z2
            if depth <= near or depth >= far:
                continue
            idx = py * width + px
            if depth < zbuf[idx]:
                zbuf[idx] = depth
                idbuf[idx] = oid_index
                o = idx * 4
                rgba[o] = r
                rgba[o + 1] = g
                rgba[o + 2] = b
                rgba[o + 3] = 255


def _edge(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)


# ---------------------------------------------------------------------------
# Public scene seeds + canaries
# ---------------------------------------------------------------------------

# Seven public canary seed ids (committed definitions below).
CANARY_SEED_IDS: tuple[str, ...] = (
    "canary_relative_position_v1",
    "canary_support_containment_v1",
    "canary_occlusion_v1",
    "canary_viewpoint_change_v1",
    "canary_object_permanence_v1",
    "canary_depth_ordering_v1",
    "canary_active_camera_v1",
)

CANARY_CAPABILITIES: dict[str, str] = {
    "canary_relative_position_v1": "relative_position",
    "canary_support_containment_v1": "support_or_containment",
    "canary_occlusion_v1": "occlusion",
    "canary_viewpoint_change_v1": "viewpoint_change",
    "canary_object_permanence_v1": "object_permanence_after_motion",
    "canary_depth_ordering_v1": "depth_ordering",
    "canary_active_camera_v1": "active_camera_selection",
}


def _seed_defs() -> dict[str, dict[str, Any]]:
    """Pinned public seed definitions.  Geometry is fully specified (no RNG)."""
    W, H = DEFAULT_WIDTH, DEFAULT_HEIGHT
    return {
        # 1. Relative position: red left of blue from cam_front; green above floor.
        "canary_relative_position_v1": {
            "schema": SEED_SCHEMA,
            "seed_id": "canary_relative_position_v1",
            "seed": 1701,
            "canary_id": "relative_position",
            "width": W,
            "height": H,
            "objects": [
                {
                    "object_id": "floor",
                    "shape": "plane",
                    "position": [0.0, -0.5, 0.0],
                    "scale": [6.0, 1.0, 6.0],
                    "color": [70, 78, 90],
                },
                {
                    "object_id": "red_block",
                    "shape": "box",
                    "position": [-1.2, 0.0, 0.0],
                    "scale": [0.8, 0.8, 0.8],
                    "color": [220, 48, 48],
                },
                {
                    "object_id": "blue_block",
                    "shape": "box",
                    "position": [1.2, 0.0, 0.0],
                    "scale": [0.8, 0.8, 0.8],
                    "color": [48, 96, 220],
                },
                {
                    "object_id": "green_block",
                    "shape": "box",
                    "position": [0.0, 1.1, 0.0],
                    "scale": [0.6, 0.6, 0.6],
                    "color": [48, 200, 80],
                },
            ],
            "cameras": [
                {
                    "camera_id": "cam_front",
                    "position": [0.0, 1.5, 5.5],
                    "look_at": [0.0, 0.3, 0.0],
                    "fov_y_deg": 45.0,
                },
                {
                    "camera_id": "cam_side",
                    "position": [5.5, 1.5, 0.0],
                    "look_at": [0.0, 0.3, 0.0],
                    "fov_y_deg": 45.0,
                },
            ],
            "active_camera": "cam_front",
            "answer": {
                "schema": ANSWER_SCHEMA,
                "canary_id": "relative_position",
                "capability": "relative_position",
                "from_camera": "cam_front",
                "left_object": "red_block",
                "right_object": "blue_block",
                "above_object": "green_block",
                "below_object": "floor",
                "question": "From cam_front, which object is on the left: red_block or blue_block?",
                "answer": "red_block",
                "requires_render": True,
                "rationale": "Screen-space centroid of red is left of blue under cam_front projection.",
            },
        },
        # 2. Support / containment: small box on table; wedge "inside" U-ish pair of walls.
        "canary_support_containment_v1": {
            "schema": SEED_SCHEMA,
            "seed_id": "canary_support_containment_v1",
            "seed": 1702,
            "canary_id": "support_containment",
            "width": W,
            "height": H,
            "objects": [
                {
                    "object_id": "table",
                    "shape": "box",
                    "position": [0.0, -0.15, 0.0],
                    "scale": [3.0, 0.3, 2.0],
                    "color": [140, 110, 70],
                },
                {
                    "object_id": "cup",
                    "shape": "box",
                    "position": [0.0, 0.45, 0.0],
                    "scale": [0.5, 0.5, 0.5],
                    "color": [200, 60, 60],
                    "support_parent": "table",
                },
                {
                    "object_id": "wall_left",
                    "shape": "box",
                    "position": [-0.9, 0.5, 0.0],
                    "scale": [0.2, 1.2, 1.2],
                    "color": [90, 100, 120],
                },
                {
                    "object_id": "wall_right",
                    "shape": "box",
                    "position": [0.9, 0.5, 0.0],
                    "scale": [0.2, 1.2, 1.2],
                    "color": [90, 100, 120],
                },
                {
                    "object_id": "token",
                    "shape": "wedge",
                    "position": [0.0, 0.35, 0.0],
                    "scale": [0.35, 0.35, 0.35],
                    "color": [40, 180, 220],
                    "support_parent": "table",
                },
            ],
            "cameras": [
                {
                    "camera_id": "cam_elevated",
                    "position": [3.5, 3.0, 4.0],
                    "look_at": [0.0, 0.3, 0.0],
                    "fov_y_deg": 40.0,
                },
                {
                    "camera_id": "cam_low",
                    "position": [0.0, 0.8, 4.5],
                    "look_at": [0.0, 0.3, 0.0],
                    "fov_y_deg": 45.0,
                },
            ],
            "active_camera": "cam_elevated",
            "answer": {
                "schema": ANSWER_SCHEMA,
                "canary_id": "support_containment",
                "capability": "support_or_containment",
                "supported_object": "cup",
                "support_surface": "table",
                "contained_object": "token",
                "container_sides": ["wall_left", "wall_right"],
                "question": "Which object rests on the table (support_parent=table): cup or wall_left?",
                "answer": "cup",
                "requires_render": True,
                "rationale": "Cup is centered above table top; walls stand beside. Visible stack in render.",
            },
        },
        # 3. Occlusion: tall red hides small blue from cam_front; cam_side sees blue.
        "canary_occlusion_v1": {
            "schema": SEED_SCHEMA,
            "seed_id": "canary_occlusion_v1",
            "seed": 1703,
            "canary_id": "occlusion",
            "width": W,
            "height": H,
            "objects": [
                {
                    "object_id": "floor",
                    "shape": "plane",
                    "position": [0.0, -0.5, 0.0],
                    "scale": [8.0, 1.0, 8.0],
                    "color": [60, 66, 78],
                },
                {
                    "object_id": "occluder",
                    "shape": "box",
                    "position": [0.0, 0.5, 0.5],
                    "scale": [1.6, 1.8, 0.4],
                    "color": [220, 50, 50],
                },
                {
                    "object_id": "blue_pyramid",
                    "shape": "pyramid",
                    "position": [0.0, 0.2, -1.2],
                    "scale": [0.7, 0.7, 0.7],
                    "color": [50, 90, 230],
                },
            ],
            "cameras": [
                {
                    "camera_id": "cam_front",
                    "position": [0.0, 1.0, 4.5],
                    "look_at": [0.0, 0.4, 0.0],
                    "fov_y_deg": 40.0,
                },
                {
                    "camera_id": "cam_side",
                    "position": [4.5, 1.2, -0.5],
                    "look_at": [0.0, 0.3, -0.5],
                    "fov_y_deg": 40.0,
                },
            ],
            "active_camera": "cam_front",
            "answer": {
                "schema": ANSWER_SCHEMA,
                "canary_id": "occlusion",
                "capability": "occlusion",
                "occluder": "occluder",
                "target": "blue_pyramid",
                "hidden_from_camera": "cam_front",
                "visible_from_camera": "cam_side",
                "question": "Is blue_pyramid visible from cam_front? From cam_side?",
                "answer": {"cam_front_visible": False, "cam_side_visible": True},
                "requires_render": True,
                "rationale": "Z-buffer: occluder covers target from cam_front; side view clears the line of sight.",
            },
        },
        # 4. Viewpoint change: asymmetric wedge; front vs back silhouette differs.
        "canary_viewpoint_change_v1": {
            "schema": SEED_SCHEMA,
            "seed_id": "canary_viewpoint_change_v1",
            "seed": 1704,
            "canary_id": "viewpoint_change",
            "width": W,
            "height": H,
            "objects": [
                {
                    "object_id": "floor",
                    "shape": "plane",
                    "position": [0.0, -0.5, 0.0],
                    "scale": [6.0, 1.0, 6.0],
                    "color": [55, 60, 70],
                },
                {
                    "object_id": "marker_near",
                    "shape": "box",
                    "position": [0.0, 0.0, 1.5],
                    "scale": [0.5, 0.5, 0.5],
                    "color": [240, 200, 40],
                },
                {
                    "object_id": "marker_far",
                    "shape": "pyramid",
                    "position": [0.0, 0.2, -1.5],
                    "scale": [0.8, 0.9, 0.8],
                    "color": [40, 200, 180],
                },
            ],
            "cameras": [
                {
                    "camera_id": "cam_a",
                    "position": [0.0, 1.2, 5.0],
                    "look_at": [0.0, 0.2, 0.0],
                    "fov_y_deg": 40.0,
                },
                {
                    "camera_id": "cam_b",
                    "position": [0.0, 1.2, -5.0],
                    "look_at": [0.0, 0.2, 0.0],
                    "fov_y_deg": 40.0,
                },
            ],
            "active_camera": "cam_a",
            "answer": {
                "schema": ANSWER_SCHEMA,
                "canary_id": "viewpoint_change",
                "capability": "viewpoint_change",
                "cam_a_nearest": "marker_near",
                "cam_b_nearest": "marker_far",
                "question": "Which marker is nearest under cam_a vs cam_b?",
                "answer": {"cam_a": "marker_near", "cam_b": "marker_far"},
                "requires_render": True,
                "rationale": "Opposite cameras reverse depth order of the two markers.",
            },
        },
        # 5. Object permanence after motion: move blue; it remains in second render.
        "canary_object_permanence_v1": {
            "schema": SEED_SCHEMA,
            "seed_id": "canary_object_permanence_v1",
            "seed": 1705,
            "canary_id": "object_permanence",
            "width": W,
            "height": H,
            "objects": [
                {
                    "object_id": "floor",
                    "shape": "plane",
                    "position": [0.0, -0.5, 0.0],
                    "scale": [8.0, 1.0, 8.0],
                    "color": [50, 55, 65],
                },
                {
                    "object_id": "anchor",
                    "shape": "box",
                    "position": [-1.5, 0.0, 0.0],
                    "scale": [0.7, 0.7, 0.7],
                    "color": [200, 80, 80],
                },
                {
                    "object_id": "mover",
                    "shape": "box",
                    "position": [0.0, 0.0, 0.0],
                    "scale": [0.7, 0.7, 0.7],
                    "color": [60, 120, 230],
                },
            ],
            "cameras": [
                {
                    "camera_id": "cam_main",
                    "position": [0.0, 2.0, 6.0],
                    "look_at": [0.0, 0.0, 0.0],
                    "fov_y_deg": 45.0,
                },
                {
                    "camera_id": "cam_alt",
                    "position": [4.0, 2.0, 4.0],
                    "look_at": [0.5, 0.0, 0.0],
                    "fov_y_deg": 45.0,
                },
            ],
            "active_camera": "cam_main",
            "motion": {
                "object_id": "mover",
                "translation": [2.0, 0.0, 0.0],
            },
            "answer": {
                "schema": ANSWER_SCHEMA,
                "canary_id": "object_permanence",
                "capability": "object_permanence_after_motion",
                "moved_object": "mover",
                "translation": [2.0, 0.0, 0.0],
                "question": "After translating mover by +2 on X, is mover still visible?",
                "answer": {"still_present": True, "object_id": "mover"},
                "requires_render": True,
                "rationale": "Second render after motion still has non-zero pixel coverage for mover.",
            },
        },
        # 6. Depth ordering: near yellow in front of far cyan from cam_front.
        "canary_depth_ordering_v1": {
            "schema": SEED_SCHEMA,
            "seed_id": "canary_depth_ordering_v1",
            "seed": 1706,
            "canary_id": "depth_ordering",
            "width": W,
            "height": H,
            "objects": [
                {
                    "object_id": "floor",
                    "shape": "plane",
                    "position": [0.0, -0.5, 0.0],
                    "scale": [8.0, 1.0, 8.0],
                    "color": [48, 52, 62],
                },
                {
                    "object_id": "near_block",
                    "shape": "box",
                    "position": [0.0, 0.0, 1.5],
                    "scale": [0.9, 0.9, 0.9],
                    "color": [240, 200, 40],
                },
                {
                    "object_id": "far_block",
                    "shape": "box",
                    "position": [0.0, 0.0, -1.5],
                    "scale": [1.2, 1.2, 1.2],
                    "color": [40, 200, 200],
                },
            ],
            "cameras": [
                {
                    "camera_id": "cam_front",
                    "position": [0.0, 1.0, 5.0],
                    "look_at": [0.0, 0.2, 0.0],
                    "fov_y_deg": 40.0,
                },
                {
                    "camera_id": "cam_top",
                    "position": [0.0, 6.0, 0.1],
                    "look_at": [0.0, 0.0, 0.0],
                    "fov_y_deg": 50.0,
                },
            ],
            "active_camera": "cam_front",
            "answer": {
                "schema": ANSWER_SCHEMA,
                "canary_id": "depth_ordering",
                "capability": "depth_ordering",
                "nearer": "near_block",
                "farther": "far_block",
                "from_camera": "cam_front",
                "question": "From cam_front depth buffer, which object is nearer?",
                "answer": "near_block",
                "requires_render": True,
                "rationale": "Min finite depth over near_block pixels is smaller than far_block.",
            },
        },
        # 7. Active camera selection: red only in cam0 FOV; blue only in cam1 FOV.
        "canary_active_camera_v1": {
            "schema": SEED_SCHEMA,
            "seed_id": "canary_active_camera_v1",
            "seed": 1707,
            "canary_id": "active_camera",
            "width": W,
            "height": H,
            "objects": [
                {
                    "object_id": "floor",
                    "shape": "plane",
                    "position": [0.0, -0.5, 0.0],
                    "scale": [10.0, 1.0, 10.0],
                    "color": [45, 50, 60],
                },
                {
                    "object_id": "left_only",
                    "shape": "box",
                    "position": [-2.5, 0.2, 0.0],
                    "scale": [0.8, 0.8, 0.8],
                    "color": [230, 50, 50],
                },
                {
                    "object_id": "right_only",
                    "shape": "box",
                    "position": [2.5, 0.2, 0.0],
                    "scale": [0.8, 0.8, 0.8],
                    "color": [50, 90, 230],
                },
            ],
            "cameras": [
                {
                    "camera_id": "cam_left",
                    "position": [-2.5, 1.5, 4.5],
                    "look_at": [-2.5, 0.2, 0.0],
                    "fov_y_deg": 35.0,
                },
                {
                    "camera_id": "cam_right",
                    "position": [2.5, 1.5, 4.5],
                    "look_at": [2.5, 0.2, 0.0],
                    "fov_y_deg": 35.0,
                },
            ],
            "active_camera": "cam_left",
            "answer": {
                "schema": ANSWER_SCHEMA,
                "canary_id": "active_camera",
                "capability": "active_camera_selection",
                "cam_left_sees": "left_only",
                "cam_right_sees": "right_only",
                "question": "After set_camera(cam_right), which colored block dominates the frame?",
                "answer": "right_only",
                "requires_render": True,
                "rationale": "Narrow FOV cameras are aimed at opposite objects; active camera selects the view.",
            },
        },
    }


def list_public_seeds() -> list[str]:
    return list(_seed_defs().keys())


def load_seed_definition(seed_id: str) -> dict[str, Any]:
    defs = _seed_defs()
    if seed_id not in defs:
        raise Spatial3DError(f"unknown seed_id {seed_id!r}")
    # Return a deep-ish copy via JSON round-trip for isolation.
    return json.loads(json.dumps(defs[seed_id], sort_keys=True))


def build_scene_from_seed(seed_id: str, *, seed_override: int | None = None) -> Scene:
    """Build a Scene from a pinned public seed.  seed_override only recolors nothing —
    geometry is fully pinned; the integer seed is recorded for digests/parity."""
    definition = load_seed_definition(seed_id)
    scene = Scene(
        seed_id=str(definition["seed_id"]),
        seed=int(seed_override if seed_override is not None else definition["seed"]),
        width=int(definition.get("width", DEFAULT_WIDTH)),
        height=int(definition.get("height", DEFAULT_HEIGHT)),
        canary_id=str(definition.get("canary_id") or ""),
        metadata={
            "motion": definition.get("motion"),
            "seed_schema": definition.get("schema"),
        },
    )
    for row in definition["objects"]:
        scene.add_object(SceneObject.from_dict(row))
    for row in definition["cameras"]:
        scene.add_camera(Camera.from_dict(row))
    scene.set_active_camera(str(definition.get("active_camera") or next(iter(scene.cameras))))
    return scene


def evaluator_answer_state(seed_id: str) -> dict[str, Any]:
    """Return evaluator-only ground truth for a seed.  Never expose to candidate sensors."""
    definition = load_seed_definition(seed_id)
    answer = dict(definition.get("answer") or {})
    answer["seed_id"] = seed_id
    answer["seed"] = definition["seed"]
    answer["renderer"] = RENDERER_ID
    answer["renderer_version"] = RENDERER_VERSION
    return answer


def write_evaluator_state(evaluator_dir: Path, scene: Scene, seed_id: str) -> dict[str, str]:
    """Persist scene graph + answer state under an evaluator-only directory.

    Path components must include a forbidden token (``evaluator``) so arm tool
    parameters cannot legally name this namespace.
    """
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    graph_path = evaluator_dir / "scene_graph.json"
    answer_path = evaluator_dir / "answer_state.json"
    graph_path.write_text(json.dumps(scene.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    answer = evaluator_answer_state(seed_id)
    answer_path.write_text(json.dumps(answer, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "scene_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "answer_state_sha256": hashlib.sha256(answer_path.read_bytes()).hexdigest(),
    }


def write_runtime_scene(path: Path, scene: Scene) -> str:
    """Serialize runtime scene state for multi-op sandbox continuity (not candidate-facing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(scene.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_runtime_scene(path: Path) -> Scene:
    if not path.is_file():
        raise Spatial3DError(f"runtime scene missing at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Scene.from_dict(data)


# ---------------------------------------------------------------------------
# Canary verification (machine-checkable, uses renders + evaluator answers)
# ---------------------------------------------------------------------------


def _mean_depth_for_object(result: RenderResult, object_id: str) -> float | None:
    """Mean finite depth of pixels covered by object_id (from a second id-aware render).

    RenderResult only stores coverage counts; callers that need per-object depth
    should use ``render_with_object_depths``.
    """
    return None


def render_with_object_depths(scene: Scene, *, camera_id: str | None = None) -> tuple[RenderResult, dict[str, float]]:
    """Render and compute mean eye-space depth per visible object (evaluator helper)."""
    result = render_scene(scene, camera_id=camera_id)
    # Re-derive depths by a lightweight second pass over the depth buffer is not
    # object-aware; re-rasterize id+depth for mean computation.
    cid = camera_id or scene.active_camera
    cam = scene.cameras[cid]
    w, h = scene.width, scene.height
    view = cam.world_to_camera()
    proj = _perspective(cam.fov_y_deg, w / h, cam.near, cam.far)
    vp = _mat4_mul(proj, view)
    zbuf = [DEPTH_FAR_SENTINEL] * (w * h)
    idbuf = [-1] * (w * h)
    object_ids = sorted(scene.objects)
    id_of = {oid: i for i, oid in enumerate(object_ids)}
    dummy_rgba = bytearray([0, 0, 0, 255] * (w * h))
    for oid in object_ids:
        obj = scene.objects[oid]
        mesh = get_mesh(obj.shape)
        for tri in mesh.transformed(obj.world_matrix()):
            _raster_triangle(
                tri,
                vp=vp,
                view=view,
                width=w,
                height=h,
                near=cam.near,
                far=cam.far,
                rgba=dummy_rgba,
                zbuf=zbuf,
                idbuf=idbuf,
                oid_index=id_of[oid],
                base_color=obj.color,
                light=scene.light_dir,
                ambient=scene.ambient,
                diffuse=scene.diffuse,
            )
    sums = {oid: 0.0 for oid in object_ids}
    counts = {oid: 0 for oid in object_ids}
    for i, idx in enumerate(idbuf):
        if idx >= 0 and zbuf[i] < DEPTH_FAR_SENTINEL * 0.5:
            oid = object_ids[idx]
            sums[oid] += zbuf[i]
            counts[oid] += 1
    means = {oid: sums[oid] / counts[oid] for oid in object_ids if counts[oid] > 0}
    return result, means


def _screen_centroid(scene: Scene, object_id: str, camera_id: str) -> tuple[float, float] | None:
    """Approximate projected centroid of object origin (not full mesh) for relative tests."""
    if object_id not in scene.objects or camera_id not in scene.cameras:
        return None
    obj = scene.objects[object_id]
    cam = scene.cameras[camera_id]
    view = cam.world_to_camera()
    proj = _perspective(cam.fov_y_deg, scene.width / scene.height, cam.near, cam.far)
    vp = _mat4_mul(proj, view)
    p = obj.position
    cx, cy, cz, cw = _mat4_vec4(vp, (p[0], p[1], p[2], 1.0))
    if abs(cw) < 1e-12:
        return None
    inv = 1.0 / cw
    ndc_x = cx * inv
    ndc_y = cy * inv
    sx = (ndc_x * 0.5 + 0.5) * scene.width
    sy = (1.0 - (ndc_y * 0.5 + 0.5)) * scene.height
    return (sx, sy)


def verify_canary(seed_id: str) -> dict[str, Any]:
    """Run one public canary end-to-end; return a machine-checkable result row."""
    if seed_id not in CANARY_CAPABILITIES:
        raise Spatial3DError(f"{seed_id!r} is not a public canary seed")
    scene = build_scene_from_seed(seed_id)
    answer = evaluator_answer_state(seed_id)
    capability = CANARY_CAPABILITIES[seed_id]
    started = __import__("time").perf_counter()
    evidence: dict[str, Any] = {
        "seed_id": seed_id,
        "capability": capability,
        "renderer": RENDERER_ID,
        "renderer_version": RENDERER_VERSION,
        "scene_digest": scene.canonical_digest(),
    }
    passed = False
    detail: dict[str, Any] = {}

    if capability == "relative_position":
        r0 = render_scene(scene, camera_id="cam_front")
        r1 = render_scene(scene, camera_id="cam_side")
        c_red = _screen_centroid(scene, "red_block", "cam_front")
        c_blue = _screen_centroid(scene, "blue_block", "cam_front")
        c_green = _screen_centroid(scene, "green_block", "cam_front")
        assert c_red and c_blue and c_green
        left_is_red = c_red[0] < c_blue[0]
        green_above_floor = c_green[1] < _screen_centroid(scene, "floor", "cam_front")[1]  # type: ignore[index]
        # Also require both colored blocks visible in the RGB buffer.
        visible_ok = "red_block" in r0.visible_object_ids and "blue_block" in r0.visible_object_ids
        passed = left_is_red and green_above_floor and visible_ok and answer["answer"] == "red_block"
        detail = {
            "cam_front_rgb_sha256": hashlib.sha256(r0.rgb_png).hexdigest(),
            "cam_front_depth_sha256": hashlib.sha256(r0.depth_f32).hexdigest(),
            "cam_side_rgb_sha256": hashlib.sha256(r1.rgb_png).hexdigest(),
            "cam_side_depth_sha256": hashlib.sha256(r1.depth_f32).hexdigest(),
            "red_centroid_x": c_red[0],
            "blue_centroid_x": c_blue[0],
            "left_is_red": left_is_red,
            "coverage": r0.pixel_object_coverage,
        }
        evidence["viewpoints"] = 2

    elif capability == "support_or_containment":
        r0 = render_scene(scene, camera_id="cam_elevated")
        r1 = render_scene(scene, camera_id="cam_low")
        cup = scene.objects["cup"]
        table = scene.objects["table"]
        # Support: cup bottom above table top and horizontally over table.
        table_top = table.position[1] + table.scale[1] * 0.5
        cup_bottom = cup.position[1] - cup.scale[1] * 0.5
        supported = cup_bottom >= table_top - 0.05 and abs(cup.position[0] - table.position[0]) < table.scale[0] * 0.5
        token = scene.objects["token"]
        wl, wr = scene.objects["wall_left"], scene.objects["wall_right"]
        contained = wl.position[0] < token.position[0] < wr.position[0]
        visible = "cup" in r0.visible_object_ids and "table" in r0.visible_object_ids
        passed = supported and contained and visible and answer["answer"] == "cup"
        detail = {
            "cam_elevated_rgb_sha256": hashlib.sha256(r0.rgb_png).hexdigest(),
            "cam_elevated_depth_sha256": hashlib.sha256(r0.depth_f32).hexdigest(),
            "cam_low_rgb_sha256": hashlib.sha256(r1.rgb_png).hexdigest(),
            "supported": supported,
            "contained": contained,
            "coverage": r0.pixel_object_coverage,
        }
        evidence["viewpoints"] = 2

    elif capability == "occlusion":
        r_front = render_scene(scene, camera_id="cam_front")
        r_side = render_scene(scene, camera_id="cam_side")
        front_hidden = "blue_pyramid" not in r_front.visible_object_ids or r_front.pixel_object_coverage.get("blue_pyramid", 0) < 8
        side_visible = r_side.pixel_object_coverage.get("blue_pyramid", 0) >= 20
        occluder_front = r_front.pixel_object_coverage.get("occluder", 0) >= 50
        expected = answer["answer"]
        passed = (
            front_hidden
            and side_visible
            and occluder_front
            and expected["cam_front_visible"] is False
            and expected["cam_side_visible"] is True
        )
        detail = {
            "cam_front_rgb_sha256": hashlib.sha256(r_front.rgb_png).hexdigest(),
            "cam_front_depth_sha256": hashlib.sha256(r_front.depth_f32).hexdigest(),
            "cam_side_rgb_sha256": hashlib.sha256(r_side.rgb_png).hexdigest(),
            "cam_side_depth_sha256": hashlib.sha256(r_side.depth_f32).hexdigest(),
            "front_coverage": r_front.pixel_object_coverage,
            "side_coverage": r_side.pixel_object_coverage,
            "front_hidden": front_hidden,
            "side_visible": side_visible,
        }
        evidence["viewpoints"] = 2
        evidence["occlusion_proof"] = True

    elif capability == "viewpoint_change":
        r_a, means_a = render_with_object_depths(scene, camera_id="cam_a")
        r_b, means_b = render_with_object_depths(scene, camera_id="cam_b")
        near_a = min(
            ((oid, d) for oid, d in means_a.items() if oid.startswith("marker_")),
            key=lambda x: x[1],
        )[0]
        near_b = min(
            ((oid, d) for oid, d in means_b.items() if oid.startswith("marker_")),
            key=lambda x: x[1],
        )[0]
        # RGB digests must differ across viewpoints.
        digests_differ = r_a.rgb_png != r_b.rgb_png
        expected = answer["answer"]
        passed = near_a == expected["cam_a"] and near_b == expected["cam_b"] and digests_differ
        detail = {
            "cam_a_rgb_sha256": hashlib.sha256(r_a.rgb_png).hexdigest(),
            "cam_b_rgb_sha256": hashlib.sha256(r_b.rgb_png).hexdigest(),
            "cam_a_depth_sha256": hashlib.sha256(r_a.depth_f32).hexdigest(),
            "cam_b_depth_sha256": hashlib.sha256(r_b.depth_f32).hexdigest(),
            "means_a": means_a,
            "means_b": means_b,
            "nearest_a": near_a,
            "nearest_b": near_b,
        }
        evidence["viewpoints"] = 2

    elif capability == "object_permanence_after_motion":
        r_before = render_scene(scene, camera_id="cam_main")
        motion = scene.metadata.get("motion") or {}
        scene.move_object(str(motion["object_id"]), translation=motion["translation"])
        r_after = render_scene(scene, camera_id="cam_main")
        r_alt = render_scene(scene, camera_id="cam_alt")
        still = r_after.pixel_object_coverage.get("mover", 0) >= 20
        moved = r_before.rgb_png != r_after.rgb_png
        anchor_stable = abs(
            r_before.pixel_object_coverage.get("anchor", 0) - r_after.pixel_object_coverage.get("anchor", 0)
        ) < max(30, r_before.pixel_object_coverage.get("anchor", 0) * 0.5)
        passed = still and moved and anchor_stable
        detail = {
            "before_rgb_sha256": hashlib.sha256(r_before.rgb_png).hexdigest(),
            "after_rgb_sha256": hashlib.sha256(r_after.rgb_png).hexdigest(),
            "after_depth_sha256": hashlib.sha256(r_after.depth_f32).hexdigest(),
            "alt_rgb_sha256": hashlib.sha256(r_alt.rgb_png).hexdigest(),
            "before_coverage": r_before.pixel_object_coverage,
            "after_coverage": r_after.pixel_object_coverage,
            "still_present": still,
            "rgb_changed": moved,
        }
        evidence["viewpoints"] = 2
        evidence["motion"] = motion

    elif capability == "depth_ordering":
        r0, means = render_with_object_depths(scene, camera_id="cam_front")
        r1 = render_scene(scene, camera_id="cam_top")
        nearer = min(
            ((oid, d) for oid, d in means.items() if oid.endswith("_block")),
            key=lambda x: x[1],
        )[0]
        passed = nearer == answer["answer"] == "near_block" and means["near_block"] < means["far_block"]
        detail = {
            "cam_front_rgb_sha256": hashlib.sha256(r0.rgb_png).hexdigest(),
            "cam_front_depth_sha256": hashlib.sha256(r0.depth_f32).hexdigest(),
            "cam_top_rgb_sha256": hashlib.sha256(r1.rgb_png).hexdigest(),
            "mean_depths": means,
            "nearer": nearer,
        }
        evidence["viewpoints"] = 2

    elif capability == "active_camera_selection":
        scene.set_active_camera("cam_left")
        r_left = render_scene(scene)
        scene.set_active_camera("cam_right")
        r_right = render_scene(scene)
        left_dom = r_left.pixel_object_coverage.get("left_only", 0) > r_left.pixel_object_coverage.get("right_only", 0)
        right_dom = r_right.pixel_object_coverage.get("right_only", 0) > r_right.pixel_object_coverage.get("left_only", 0)
        digests_differ = r_left.rgb_png != r_right.rgb_png
        passed = left_dom and right_dom and digests_differ and answer["answer"] == "right_only"
        detail = {
            "cam_left_rgb_sha256": hashlib.sha256(r_left.rgb_png).hexdigest(),
            "cam_left_depth_sha256": hashlib.sha256(r_left.depth_f32).hexdigest(),
            "cam_right_rgb_sha256": hashlib.sha256(r_right.rgb_png).hexdigest(),
            "cam_right_depth_sha256": hashlib.sha256(r_right.depth_f32).hexdigest(),
            "left_coverage": r_left.pixel_object_coverage,
            "right_coverage": r_right.pixel_object_coverage,
        }
        evidence["viewpoints"] = 2

    else:
        raise Spatial3DError(f"unhandled capability {capability!r}")

    elapsed = __import__("time").perf_counter() - started
    # Determinism sub-check: rebuild + re-render active camera twice
    scene_a = build_scene_from_seed(seed_id)
    scene_b = build_scene_from_seed(seed_id)
    ra = render_scene(scene_a)
    rb = render_scene(scene_b)
    deterministic = ra.rgb_png == rb.rgb_png and ra.depth_f32 == rb.depth_f32

    return {
        "schema": CANARY_SCHEMA,
        "seed_id": seed_id,
        "capability": capability,
        "passed": bool(passed and deterministic),
        "deterministic": deterministic,
        "requires_render": True,
        "answer_consulted": True,
        "evaluator_answer": answer,
        "evidence": evidence,
        "detail": detail,
        "elapsed_seconds": round(elapsed, 6),
        "renderer": RENDERER_ID,
        "renderer_version": RENDERER_VERSION,
    }


def run_all_canaries() -> dict[str, Any]:
    """Execute all seven public canaries; return aggregate document."""
    rows = [verify_canary(seed_id) for seed_id in CANARY_SEED_IDS]
    doc = {
        "schema": "SUBSTRATE_SPATIAL3D_CANARY_SUITE/v1",
        "renderer": RENDERER_ID,
        "renderer_version": RENDERER_VERSION,
        "canary_count": len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "rows": rows,
        "seed_ids": list(CANARY_SEED_IDS),
    }
    doc["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in doc.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return doc


def write_committed_seeds(out_dir: Path) -> list[Path]:
    """Materialize public seed JSON files under plans/ or artifacts/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for seed_id, definition in _seed_defs().items():
        # Public seed: strip evaluator answer into sibling evaluator file.
        public = {k: v for k, v in definition.items() if k != "answer"}
        public_path = out_dir / f"{seed_id}.json"
        public_path.write_text(json.dumps(public, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        paths.append(public_path)
        eval_dir = out_dir / "evaluator" / seed_id
        eval_dir.mkdir(parents=True, exist_ok=True)
        answer_path = eval_dir / "answer_state.json"
        answer_path.write_text(json.dumps(definition["answer"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
        paths.append(answer_path)
    index = {
        "schema": "SUBSTRATE_SPATIAL3D_SEED_INDEX/v1",
        "renderer": RENDERER_ID,
        "renderer_version": RENDERER_VERSION,
        "seeds": list(CANARY_SEED_IDS),
        "capabilities": dict(CANARY_CAPABILITIES),
    }
    index_path = out_dir / "SEED_INDEX.json"
    index_path.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    paths.append(index_path)
    return paths


def inspect_mesh_public(scene: Scene, object_id: str) -> dict[str, Any]:
    """Mesh statistics without world pose of the full scene graph.

    Returns local mesh topology + axis-aligned local bbox.  Deliberately omits
    other objects' transforms so this is not a scene-graph leak.
    """
    if object_id not in scene.objects:
        raise Spatial3DError(f"unknown object {object_id!r}")
    obj = scene.objects[object_id]
    mesh = get_mesh(obj.shape)
    xs = [v[0] for v in mesh.vertices]
    ys = [v[1] for v in mesh.vertices]
    zs = [v[2] for v in mesh.vertices]
    return {
        "object_id": object_id,
        "shape": obj.shape,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "local_bbox": [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)],
        "renderer": RENDERER_ID,
        "renderer_version": RENDERER_VERSION,
        # Note: world pose intentionally omitted from candidate-facing inspect.
        "world_pose_included": False,
    }


def candidate_sensor_bundle(result: RenderResult) -> dict[str, Any]:
    """Metadata a candidate may see alongside RGB/depth artifacts — no scene graph."""
    return {
        "renderer": result.renderer,
        "renderer_version": result.renderer_version,
        "seed_id": result.seed_id,
        "camera_id": result.camera_id,
        "width": result.width,
        "height": result.height,
        "rgb_sha256": hashlib.sha256(result.rgb_png).hexdigest(),
        "depth_sha256": hashlib.sha256(result.depth_u16_png).hexdigest(),
        "depth_f32_sha256": hashlib.sha256(result.depth_f32).hexdigest(),
        "depth_encoding": {
            "admitted_format": "png_uint16_greyscale_millimeters",
            "scale_mm": DEPTH_U16_SCALE_MM,
            "zero_means": "background_or_invalid",
        },
        "scene_digest": result.scene_digest,
        # Visibility flags without poses — still a sensor summary, not GT answers.
        "visible_object_ids": list(result.visible_object_ids),
    }


__all__ = (
    "ANSWER_SCHEMA",
    "CANARY_CAPABILITIES",
    "CANARY_SEED_IDS",
    "DEPTH_FAR_SENTINEL",
    "RENDERER_ID",
    "RENDERER_VERSION",
    "SCENE_SCHEMA",
    "SEED_SCHEMA",
    "Camera",
    "RenderResult",
    "Scene",
    "SceneObject",
    "Spatial3DError",
    "build_scene_from_seed",
    "candidate_sensor_bundle",
    "evaluator_answer_state",
    "get_mesh",
    "inspect_mesh_public",
    "list_public_seeds",
    "load_seed_definition",
    "read_runtime_scene",
    "render_scene",
    "render_with_object_depths",
    "run_all_canaries",
    "verify_canary",
    "write_committed_seeds",
    "write_depth_f32",
    "write_depth_u16_png",
    "write_evaluator_state",
    "write_png_rgba",
    "write_runtime_scene",
)
