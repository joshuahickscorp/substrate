"""Array and waveform sensorium for the Substrate final revision."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from substrate import final_revision_io as io

MODALITIES = (
    "image",
    "video",
    "audio",
    "speech",
    "depth",
    "mesh",
    "point_cloud",
    "tool_telemetry",
    "filesystem_events",
)


@dataclass(frozen=True)
class MediaPacket:
    modality: str
    payload: Any
    timestamp: float
    provenance: str
    metadata: Mapping[str, Any] | None = None


def _array(value: Any, *, dimensions: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if dimensions is not None and array.ndim not in dimensions:
        raise io.Refused(f"expected dimensions {dimensions}, received shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise io.Refused("sensor arrays must be numeric")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise io.Refused("sensor arrays must be finite")
    return array


def _payload_digest(value: Any) -> str:
    hasher = hashlib.sha256()

    def update(child: Any) -> None:
        if isinstance(child, np.ndarray):
            hasher.update(str(child.dtype).encode())
            hasher.update(str(child.shape).encode())
            hasher.update(np.ascontiguousarray(child).tobytes())
        elif isinstance(child, Mapping):
            for key in sorted(child, key=str):
                hasher.update(str(key).encode())
                update(child[key])
        elif isinstance(child, (list, tuple)):
            for item in child:
                update(item)
        else:
            hasher.update(repr(child).encode())

    update(value)
    return hasher.hexdigest()


def _gray(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.shape[-1] not in {1, 3, 4}:
        raise io.Refused("image/video channel dimension must be 1, 3, or 4")
    return np.mean(array[..., :3], axis=-1)


def _centroid(weights: np.ndarray) -> list[float]:
    positive = np.maximum(weights - float(np.min(weights)), 0.0)
    total = float(np.sum(positive))
    if total <= 1e-12:
        return [float((weights.shape[1] - 1) / 2), float((weights.shape[0] - 1) / 2)]
    y, x = np.indices(weights.shape)
    return [float(np.sum(x * positive) / total), float(np.sum(y * positive) / total)]


def _best_translation(first: np.ndarray, second: np.ndarray, radius: int = 2) -> tuple[int, int, float]:
    best = (0, 0, float("inf"))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(first, (dy, dx), axis=(0, 1))
            error = float(np.mean(np.abs(second - shifted)))
            if error < best[2]:
                best = (dx, dy, error)
    return best


class Sensorium:
    """Process real media structures without accepting hidden semantic labels."""

    def process(self, packet: MediaPacket) -> dict[str, Any]:
        if packet.modality not in MODALITIES:
            raise io.Refused(f"unknown modality {packet.modality!r}")
        if not packet.provenance:
            raise io.Refused("media provenance is required")
        handler = getattr(self, f"_process_{packet.modality}")
        features = handler(packet.payload, dict(packet.metadata or {}))
        return {
            "modality": packet.modality,
            "timestamp": float(packet.timestamp),
            "provenance": packet.provenance,
            "content_digest": _payload_digest(packet.payload),
            "features": features,
            "feature_digest": io.digest(features),
            "hidden_label_used": False,
            "activation": False,
        }

    def _process_image(self, payload: Any, _metadata: dict[str, Any]) -> dict[str, Any]:
        image = _array(payload, dimensions=(2, 3))
        gray = _gray(image)
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        return {
            "shape": list(image.shape),
            "mean": float(np.mean(gray)),
            "standard_deviation": float(np.std(gray)),
            "intensity_centroid_xy": _centroid(gray),
            "horizontal_edge_energy": float(np.mean(np.abs(dx))),
            "vertical_edge_energy": float(np.mean(np.abs(dy))),
        }

    def _process_video(self, payload: Any, _metadata: dict[str, Any]) -> dict[str, Any]:
        video = _array(payload, dimensions=(3, 4))
        if video.shape[0] < 2:
            raise io.Refused("video requires at least two frames")
        frames = np.asarray([_gray(frame) for frame in video])
        temporal = np.diff(frames, axis=0)
        tracks = [_centroid(frame) for frame in frames]
        translations = [_best_translation(frames[index], frames[index + 1]) for index in range(len(frames) - 1)]
        compensated = []
        for index, (dx, dy, _error) in enumerate(translations):
            shifted = np.roll(frames[index], (dy, dx), axis=(0, 1))
            compensated.append(float(np.mean(np.abs(frames[index + 1] - shifted))))
        return {
            "shape": list(video.shape),
            "frame_count": int(video.shape[0]),
            "motion_energy": float(np.mean(np.abs(temporal))),
            "object_track_centroids_xy": tracks,
            "camera_translation_proxies_xy": [[row[0], row[1]] for row in translations],
            "camera_fit_errors": [row[2] for row in translations],
            "residual_object_motion_energy": float(np.mean(compensated)),
            "event_boundaries": [
                index + 1
                for index, value in enumerate(np.mean(np.abs(temporal), axis=(1, 2)))
                if value > float(np.median(np.mean(np.abs(temporal), axis=(1, 2))))
            ],
        }

    def _audio_features(self, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
        if waveform.ndim != 1 or waveform.size < 16:
            raise io.Refused("audio must be a one-dimensional waveform with at least 16 samples")
        if sample_rate <= 0:
            raise io.Refused("sample_rate must be positive")
        window = max(8, min(512, waveform.size // 8))
        energies = np.asarray([np.sqrt(np.mean(waveform[index : index + window] ** 2)) for index in range(0, waveform.size - window + 1, window)])
        spectrum = np.abs(np.fft.rfft(waveform))
        frequencies = np.fft.rfftfreq(waveform.size, 1.0 / sample_rate)
        denominator = float(np.sum(spectrum))
        centroid = float(np.sum(frequencies * spectrum) / denominator) if denominator else 0.0
        onset_threshold = float(np.median(energies) + np.std(energies)) if energies.size else 0.0
        return {
            "samples": int(waveform.size),
            "sample_rate_hz": int(sample_rate),
            "duration_seconds": float(waveform.size / sample_rate),
            "rms": float(np.sqrt(np.mean(waveform**2))),
            "zero_crossing_rate": float(np.mean(np.signbit(waveform[:-1]) != np.signbit(waveform[1:]))),
            "spectral_centroid_hz": centroid,
            "energy_windows": [float(value) for value in energies],
            "onset_windows": [int(index) for index, value in enumerate(energies) if value > onset_threshold],
        }

    def _process_audio(self, payload: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        waveform = _array(payload, dimensions=(1,))
        return self._audio_features(waveform, int(metadata.get("sample_rate", 16_000)))

    def _process_speech(self, payload: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        waveform = _array(payload, dimensions=(1,))
        features = self._audio_features(waveform, int(metadata.get("sample_rate", 16_000)))
        energy = np.asarray(features["energy_windows"])
        threshold = float(np.median(energy)) if energy.size else 0.0
        voiced = energy > threshold
        segments: list[list[int]] = []
        start: int | None = None
        for index, active in enumerate(voiced):
            if active and start is None:
                start = index
            if start is not None and (not active or index == len(voiced) - 1):
                end = index if not active else index + 1
                segments.append([start, end])
                start = None
        features["waveform_segments"] = segments
        features["transcript"] = None
        features["speech_content_claimed"] = False
        return features

    def _process_depth(self, payload: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        depth = _array(payload, dimensions=(2,))
        if np.any(depth <= 0):
            raise io.Refused("depth values must be positive")
        height, width = depth.shape
        focal = float(metadata.get("focal_length_px", max(height, width)))
        cy = float(metadata.get("principal_y", (height - 1) / 2))
        cx = float(metadata.get("principal_x", (width - 1) / 2))
        y, x = np.indices(depth.shape)
        points = np.stack(((x - cx) * depth / focal, (y - cy) * depth / focal, depth), axis=-1).reshape(-1, 3)
        return {
            "shape": list(depth.shape),
            "minimum_depth": float(np.min(depth)),
            "maximum_depth": float(np.max(depth)),
            "median_depth": float(np.median(depth)),
            "point_count": int(points.shape[0]),
            "point_centroid_xyz": [float(value) for value in np.mean(points, axis=0)],
            "geometry_extent_xyz": [float(value) for value in np.ptp(points, axis=0)],
            "focal_length_px": focal,
        }

    def _process_mesh(self, payload: Any, _metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise io.Refused("mesh payload requires vertices and faces")
        vertices = _array(payload.get("vertices"), dimensions=(2,))
        faces = np.asarray(payload.get("faces"), dtype=np.int64)
        if vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise io.Refused("mesh vertices and triangular faces have invalid shape")
        if faces.size and (int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices)):
            raise io.Refused("mesh face index is out of range")
        areas = []
        for face in faces:
            a, b, c = vertices[face]
            areas.append(float(np.linalg.norm(np.cross(b - a, c - a)) / 2))
        return {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "centroid_xyz": [float(value) for value in np.mean(vertices, axis=0)],
            "extent_xyz": [float(value) for value in np.ptp(vertices, axis=0)],
            "surface_area": float(sum(areas)),
        }

    def _process_point_cloud(self, payload: Any, _metadata: dict[str, Any]) -> dict[str, Any]:
        points = _array(payload, dimensions=(2,))
        if points.shape[1] != 3 or len(points) < 2:
            raise io.Refused("point cloud must have shape [N,3] with N >= 2")
        centered = points - np.mean(points, axis=0)
        covariance = centered.T @ centered / max(1, len(points) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        return {
            "points": int(len(points)),
            "centroid_xyz": [float(value) for value in np.mean(points, axis=0)],
            "extent_xyz": [float(value) for value in np.ptp(points, axis=0)],
            "covariance_eigenvalues": [float(value) for value in eigenvalues],
        }

    def _process_tool_telemetry(self, payload: Any, _metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or not payload:
            raise io.Refused("tool telemetry must be a nonempty row sequence")
        rows = [dict(row) for row in payload]
        if not all({"time", "value"} <= set(row) for row in rows):
            raise io.Refused("telemetry rows require time and value")
        times = _array([row["time"] for row in rows], dimensions=(1,))
        values = _array([row["value"] for row in rows], dimensions=(1,))
        if np.any(np.diff(times) < 0):
            raise io.Refused("telemetry time must be monotonic")
        return {
            "samples": len(rows),
            "duration": float(times[-1] - times[0]),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "failures": [str(row["status"]) for row in rows if row.get("status") not in {None, "ok"}],
        }

    def _process_filesystem_events(self, payload: Any, _metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or not payload:
            raise io.Refused("filesystem events must be a nonempty row sequence")
        rows = [dict(row) for row in payload]
        required = {"sequence", "operation", "path_digest"}
        if not all(required <= set(row) for row in rows):
            raise io.Refused("filesystem event fields are incomplete")
        sequences = [int(row["sequence"]) for row in rows]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise io.Refused("filesystem sequences must be unique and ordered")
        return {
            "events": len(rows),
            "operations": {operation: sum(row["operation"] == operation for row in rows) for operation in sorted({row["operation"] for row in rows})},
            "path_digests": sorted({str(row["path_digest"]) for row in rows}),
            "ordered": True,
        }

    @staticmethod
    def bind(receipts: Sequence[Mapping[str, Any]], *, tolerance_seconds: float) -> dict[str, Any]:
        if len(receipts) < 2:
            raise io.Refused("cross-modal binding requires at least two receipts")
        modalities = [str(row["modality"]) for row in receipts]
        if len(set(modalities)) != len(modalities):
            raise io.Refused("cross-modal binding requires distinct modalities")
        times = [float(row["timestamp"]) for row in receipts]
        span = max(times) - min(times)
        return {
            "modalities": modalities,
            "content_digests": [str(row["content_digest"]) for row in receipts],
            "time_span_seconds": span,
            "within_tolerance": span <= tolerance_seconds,
            "distinct_information": len({str(row["content_digest"]) for row in receipts}) == len(receipts),
            "binding_digest": io.digest(
                {
                    "modalities": modalities,
                    "content_digests": [str(row["content_digest"]) for row in receipts],
                    "time_span_seconds": span,
                }
            ),
            "activation": False,
        }


def controlled_media() -> dict[str, MediaPacket]:
    image = np.zeros((16, 16), dtype=np.float64)
    image[5:9, 3:7] = 1.0
    video = np.zeros((5, 16, 16), dtype=np.float64)
    for index in range(5):
        video[index, 5:8, 2 + index : 5 + index] = 1.0
    sample_rate = 8000
    timeline = np.arange(0, 0.2, 1 / sample_rate)
    audio = 0.2 * np.sin(2 * np.pi * 440 * timeline)
    audio[600:620] += 0.8
    speech = np.zeros(1600, dtype=np.float64)
    speech[200:600] = 0.3 * np.sin(2 * np.pi * 180 * np.arange(400) / sample_rate)
    speech[900:1400] = 0.25 * np.sin(2 * np.pi * 240 * np.arange(500) / sample_rate)
    depth = np.tile(np.linspace(1.0, 2.0, 16), (16, 1))
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    point_cloud = np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 2], [0.5, 0.5, 1.5]], dtype=np.float64)
    return {
        "image": MediaPacket("image", image, 1.00, "generated://final-revision/image"),
        "video": MediaPacket("video", video, 1.02, "generated://final-revision/video"),
        "audio": MediaPacket("audio", audio, 1.01, "generated://final-revision/audio", {"sample_rate": sample_rate}),
        "speech": MediaPacket("speech", speech, 1.015, "generated://final-revision/speech", {"sample_rate": sample_rate}),
        "depth": MediaPacket("depth", depth, 1.00, "generated://final-revision/depth", {"focal_length_px": 16}),
        "mesh": MediaPacket("mesh", {"vertices": vertices, "faces": faces}, 1.00, "generated://final-revision/mesh"),
        "point_cloud": MediaPacket("point_cloud", point_cloud, 1.00, "generated://final-revision/point-cloud"),
        "tool_telemetry": MediaPacket(
            "tool_telemetry",
            [{"time": 0.0, "value": 0.0, "status": "ok"}, {"time": 0.1, "value": 0.5, "status": "ok"}, {"time": 0.2, "value": 0.2, "status": "stall"}],
            1.03,
            "generated://final-revision/tool",
        ),
        "filesystem_events": MediaPacket(
            "filesystem_events",
            [
                {"sequence": 1, "operation": "create", "path_digest": io.digest("project/a")},
                {"sequence": 2, "operation": "modify", "path_digest": io.digest("project/a")},
            ],
            1.04,
            "generated://final-revision/filesystem",
        ),
    }


def structural_sensorium_report() -> dict[str, Any]:
    sensorium = Sensorium()
    packets = controlled_media()
    receipts = {name: sensorium.process(packet) for name, packet in packets.items()}
    corrupted_image = np.asarray(packets["image"].payload).copy()
    corrupted_image[:, :] = 0.0
    corruption_receipt = sensorium.process(MediaPacket("image", corrupted_image, 1.0, "generated://final-revision/image-corrupted"))
    audiovisual = sensorium.bind([receipts["video"], receipts["audio"], receipts["speech"]], tolerance_seconds=0.05)
    distinct = len({receipt["content_digest"] for receipt in receipts.values()}) == len(receipts)
    return {
        "modalities": list(receipts),
        "receipts": receipts,
        "real_structures": {
            "image_array": True,
            "video_frames": True,
            "motion_from_frame_differences": True,
            "audio_waveform": True,
            "speech_waveform": True,
            "depth_map": True,
            "mesh_vertices_and_faces": True,
            "point_cloud_xyz": True,
            "tool_telemetry": True,
            "filesystem_events": True,
        },
        "modality_content_digests_distinct": distinct,
        "corruption_changes_image_features": corruption_receipt["feature_digest"] != receipts["image"]["feature_digest"],
        "cross_modal_timing": audiovisual,
        "hidden_labels_used": False,
        "limitations": [
            "camera-motion separation is a bounded translation proxy, not a learned optical-flow claim",
            "speech processing detects waveform segments but does not claim transcription",
            "generated controlled media establish structure handling, not open-world perception",
        ],
        "activation": False,
    }
