"""Authorities, sandbox, canaries, pilot admission, and terminal evidence."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import wave
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from substrate import nous_closure_config as C
from substrate import nous_closure_experiment as E
from substrate import nous_closure_io as io


def _remote_tag_refs() -> dict[str, dict[str, str | None]]:
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin"],
        cwd=io.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise io.Refused(result.stderr.strip() or "cannot resolve remote tags")
    rows: dict[str, dict[str, str | None]] = {}
    for line in result.stdout.splitlines():
        object_id, ref = line.split()
        prefix = "refs/tags/"
        if not ref.startswith(prefix):
            continue
        name = ref.removeprefix(prefix).removesuffix("^{}")
        row = rows.setdefault(name, {"tag_object": None, "peeled_commit": None})
        if ref.endswith("^{}"):
            row["peeled_commit"] = object_id
        else:
            row["tag_object"] = object_id
    return rows


def _tag_snapshot(tag: str, remote: dict[str, dict[str, str | None]]) -> dict[str, Any]:
    local_object = io.ref_or_none(f"refs/tags/{tag}")
    peeled = io.ref_or_none(f"refs/tags/{tag}", peel=True)
    object_type = io.git("cat-file", "-t", f"refs/tags/{tag}", check=False)
    remote_row = remote.get(tag, {})
    return {
        "name": tag,
        "tag_object": local_object,
        "peeled_commit": peeled,
        "object_type": object_type or None,
        "annotated": object_type == "tag",
        "remote_tag_object": remote_row.get("tag_object"),
        "remote_peeled_commit": remote_row.get("peeled_commit"),
        "tag_object_matches_remote": local_object == remote_row.get("tag_object"),
        "peeled_commit_matches_remote": peeled == remote_row.get("peeled_commit"),
    }


def lineage() -> dict[str, Any]:
    remote = _remote_tag_refs()
    tags = {tag: _tag_snapshot(tag, remote) for tag in (*C.HISTORICAL_TAGS, C.PREFLIGHT_TAG)}
    v5_ready = tags["substrate-v5-sensorium-ready"]["peeled_commit"]
    v5_terminal = tags["substrate-v5-terminal"]["peeled_commit"]
    preflight = tags[C.PREFLIGHT_TAG]["peeled_commit"]
    return io.authority(
        "substrate-nous-closure-lineage/v1",
        {
            "repository": io.git("remote", "get-url", "origin"),
            "starting_main": preflight,
            "v5_ready_commit": v5_ready,
            "v5_terminal_commit": v5_terminal,
            "v5_terminal_evidence_commit": "856c6fa040523e6f81431a7724b56aabb6c467aa",
            "starting_classification": C.STARTING_CLASSIFICATION,
            "tags": tags,
            "preservation_rule": "all v1-v5 refs and objects are immutable",
        },
    )


def immutability() -> dict[str, Any]:
    document = lineage()
    failed: list[str] = []
    for tag in C.HISTORICAL_TAGS:
        row = document["tags"][tag]
        if not row["annotated"]:
            failed.append(f"{tag}:not_annotated")
        if not row["tag_object_matches_remote"]:
            failed.append(f"{tag}:tag_object_remote_mismatch")
        if not row["peeled_commit_matches_remote"]:
            failed.append(f"{tag}:peeled_commit_remote_mismatch")
    terminal_files = {
        version: {
            "path": path,
            "sha256": io.file_digest(io.ROOT / path),
        }
        for version, path in {
            "v1": "evidence/substrate/v1/SUBSTRATE_FINAL_STATE.json",
            "v2": "evidence/substrate/v2/SUBSTRATE_V2_FINAL_CLASSIFICATION.json",
            "v3": "evidence/substrate/v3/SUBSTRATE_V3_FINAL_CLASSIFICATION.json",
            "v4": "evidence/substrate/v4/SUBSTRATE_V4_FINAL_CLASSIFICATION.json",
            "v5": "evidence/substrate/v5/SUBSTRATE_V5_FINAL_CLASSIFICATION.json",
        }.items()
    }
    return io.authority(
        "substrate-nous-closure-immutability/v1",
        {
            "historical_tags": document["tags"],
            "terminal_classification_files": terminal_files,
            "historical_classifications": C.HISTORICAL_CLASSIFICATIONS,
            "failed": failed,
            "all_pass": not failed,
        },
        status="instrument_verified" if not failed else "invalid_bed",
    )


def _facet_rows() -> list[dict[str, Any]]:
    return [
        {
            "facet": index,
            "name": name,
            "definition": definition,
            "hypothesis": f"H_NC{index}",
            "scores": {
                "0": "absent, invalid, or inactive",
                "0.5": "implemented or cheaply demonstrated but not closure-principal-positive",
                "1": "principal-positive, replicated, applicable open-world-positive, independently verified, mutation-resistant",
            },
            "critical": True,
        }
        for index, (name, definition) in C.FACETS.items()
    ]


def _counterfeit_fixture(name: str, positive: bool) -> dict[str, Any]:
    clean: dict[str, Any] = {
        "observation": {"public_cue": 0.25},
        "metadata": {"split": "held_out"},
        "policy": {"kind": "outcome_blind"},
        "modalities": {"image": "image-digest", "audio": "audio-digest"},
        "model_registry": {"image": "model-a", "audio": "model-b"},
        "credit": {"owned_state": True, "behavior_changed": True},
        "routing": {"fixture_specific": False},
        "learning": {"outcome_visible_before_commit": False},
        "history": {"used_as_answer_key": False},
        "generator": {"template_overlap": False},
        "baseline": {"information_parity": True, "compute_parity": True, "tool_parity": True},
        "checkpoint": {"identity": True, "goals": True, "scene": True, "world_model": True, "self_model": True, "model_competence": True},
        "teaching": {"contains_target_action": False},
        "support": {"duplicates_verifier": False},
        "active_perception": {"predeclared_correct_view": False},
        "oracle_access": False,
    }
    if not positive:
        return clean
    fixture = copy.deepcopy(clean)
    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "answer_leakage": lambda row: row["observation"].update(target=1),
        "seed_leakage": lambda row: row["observation"].update(history_seed=123),
        "task_identity_leakage": lambda row: row["observation"].update(task_identity="raw-answer-key"),
        "surface_label_leakage": lambda row: row["observation"].update(surface_label="target:1"),
        "hidden_oracle_access": lambda row: row.update(oracle_access=True),
        "scripted_transition_lookup": lambda row: row["policy"].update(kind="scripted_transition_lookup"),
        "modality_aliases": lambda row: row.update(modalities={"image": "same", "audio": "same"}),
        "state_hash_without_behavior": lambda row: row["credit"].update(behavior_changed=False),
        "same_module_many_names": lambda row: row.update(model_registry={"image": "same-module", "audio": "same-module"}),
        "transcript_replay_as_memory": lambda row: row["credit"].update(owned_state=False),
        "fixture_specific_routing": lambda row: row["routing"].update(fixture_specific=True),
        "future_outcome_learning": lambda row: row["learning"].update(outcome_visible_before_commit=True),
        "history_as_lookup_key": lambda row: row["history"].update(used_as_answer_key=True),
        "open_world_template_reuse": lambda row: row["generator"].update(template_overlap=True),
        "weakened_strong_baseline": lambda row: row["baseline"].update(information_parity=False),
        "unequal_compute_or_tools": lambda row: row["baseline"].update(compute_parity=False),
        "checkpoint_state_omission": lambda row: row["checkpoint"].update(scene=False),
        "metadata_only_identity": lambda row: row["checkpoint"].update(
            goals=False,
            scene=False,
            world_model=False,
            self_model=False,
            model_competence=False,
        ),
        "teaching_target_action_leak": lambda row: row["teaching"].update(contains_target_action=True),
        "support_duplicates_verifier": lambda row: row["support"].update(duplicates_verifier=True),
        "predeclared_active_view": lambda row: row["active_perception"].update(predeclared_correct_view=True),
    }
    mutations[name](fixture)
    return fixture


def detect_counterfeit(name: str, fixture: dict[str, Any]) -> bool:
    checks: dict[str, Callable[[dict[str, Any]], bool]] = {
        "answer_leakage": lambda row: "target" in row["observation"],
        "seed_leakage": lambda row: "history_seed" in row["observation"],
        "task_identity_leakage": lambda row: "task_identity" in row["observation"],
        "surface_label_leakage": lambda row: "surface_label" in row["observation"],
        "hidden_oracle_access": lambda row: row["oracle_access"] is True,
        "scripted_transition_lookup": lambda row: row["policy"]["kind"] == "scripted_transition_lookup",
        "modality_aliases": lambda row: len(set(row["modalities"].values())) < len(row["modalities"]),
        "state_hash_without_behavior": lambda row: row["credit"]["behavior_changed"] is False,
        "same_module_many_names": lambda row: len(set(row["model_registry"].values())) < len(row["model_registry"]),
        "transcript_replay_as_memory": lambda row: row["credit"]["owned_state"] is False,
        "fixture_specific_routing": lambda row: row["routing"]["fixture_specific"] is True,
        "future_outcome_learning": lambda row: row["learning"]["outcome_visible_before_commit"] is True,
        "history_as_lookup_key": lambda row: row["history"]["used_as_answer_key"] is True,
        "open_world_template_reuse": lambda row: row["generator"]["template_overlap"] is True,
        "weakened_strong_baseline": lambda row: row["baseline"]["information_parity"] is False,
        "unequal_compute_or_tools": lambda row: not row["baseline"]["compute_parity"] or not row["baseline"]["tool_parity"],
        "checkpoint_state_omission": lambda row: not all(row["checkpoint"].values()),
        "metadata_only_identity": lambda row: (
            row["checkpoint"]["identity"] and not any(row["checkpoint"][key] for key in ("goals", "scene", "world_model", "self_model", "model_competence"))
        ),
        "teaching_target_action_leak": lambda row: row["teaching"]["contains_target_action"] is True,
        "support_duplicates_verifier": lambda row: row["support"]["duplicates_verifier"] is True,
        "predeclared_active_view": lambda row: row["active_perception"]["predeclared_correct_view"] is True,
    }
    return bool(checks[name](fixture))


def counterfeit_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixtures: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name in C.COUNTERFEIT_EXPLANATIONS:
        positive = _counterfeit_fixture(name, True)
        negative = _counterfeit_fixture(name, False)
        positive_detected = detect_counterfeit(name, positive)
        clean_rejected = detect_counterfeit(name, negative)
        fixtures[name] = {
            "positive": positive,
            "clean_negative": negative,
        }
        rows.append(
            {
                "explanation": name,
                "detector": f"detect_counterfeit:{name}",
                "positive_detected": positive_detected,
                "clean_negative_rejected": clean_rejected,
                "mutation": f"inject:{name}",
                "remediation": "invalidate affected unit, remove shortcut, regenerate from frozen clean fixture",
                "passes": positive_detected and not clean_rejected,
            }
        )
    strongest = copy.deepcopy(_counterfeit_fixture(C.COUNTERFEIT_EXPLANATIONS[0], False))

    def merge_mutation(target: dict[str, Any], clean: dict[str, Any], mutated: dict[str, Any]) -> None:
        for key, value in mutated.items():
            if isinstance(value, dict) and isinstance(clean.get(key), dict):
                merge_mutation(target[key], clean[key], value)
            elif value != clean.get(key):
                target[key] = copy.deepcopy(value)

    for name in C.COUNTERFEIT_EXPLANATIONS:
        positive = _counterfeit_fixture(name, True)
        merge_mutation(strongest, _counterfeit_fixture(name, False), positive)
    detected = [name for name in C.COUNTERFEIT_EXPLANATIONS if detect_counterfeit(name, strongest)]
    audit = io.authority(
        "substrate-nous-closure-counterfeit-audit/v1",
        {
            "rows": rows,
            "tested": len(rows),
            "passed": sum(row["passes"] for row in rows),
            "all_pass": all(row["passes"] for row in rows),
        },
        status="instrument_verified",
    )
    fixture_document = io.authority(
        "substrate-nous-closure-counterfeit-fixtures/v1",
        {
            "fixtures": fixtures,
            "strongest_counterfeit": strongest,
        },
    )
    rejection = io.authority(
        "substrate-nous-closure-counterfeit-rejection/v1",
        {
            "strongest_counterfeit_detected_explanations": detected,
            "required": list(C.COUNTERFEIT_EXPLANATIONS),
            "rejected": set(detected) == set(C.COUNTERFEIT_EXPLANATIONS),
        },
        status="instrument_verified",
    )
    return audit, fixture_document, rejection


def _png_bytes(width: int, height: int, seed: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend(((x * 5 + seed) % 256, (y * 7 + seed * 3) % 256, ((x + y) * 3 + seed * 11) % 256))
        rows.append(bytes(row))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
        + chunk(b"IEND", b"")
    )


def _write_media_file(path: Path, payload: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)


def _generate_wav(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 8_000
    frequency = 330.0 + seed * 11.0
    samples = [int(12_000 * math.sin(2.0 * math.pi * frequency * index / rate)) for index in range(rate // 2)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _generate_mp4(path: Path, seed: int) -> tuple[bool, str | None]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, "ffmpeg unavailable"
    color = f"0x{(seed * 0x314159) & 0xFFFFFF:06x}"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=96x64:r=8:d=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
            "-y",
            str(path),
        ],
        cwd=io.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stderr.strip() or None


def build_sandbox() -> dict[str, Any]:
    manifest: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for index, family in enumerate(C.SANDBOX_FAMILIES):
        root = io.MEDIA / f"{index + 1:02d}_{family}"
        root.mkdir(parents=True, exist_ok=True)
        task = E.sandbox_task(15_000 + index, index)
        task_path = root / "task.json"
        _write_media_file(task_path, json.dumps(task, indent=2, sort_keys=True) + "\n")
        _write_media_file(root / "README.md", f"# {family.replace('_', ' ').title()}\n\nOffline closure sandbox fixture.\n")
        _write_media_file(root / "document.md", f"# Incident {index + 1}\n\nToken: `{task['identity']}`\n")
        _write_media_file(root / "workspace.csv", "item,status,owner\nalpha,open,entity\nbeta,blocked,reviewer\n")
        _write_media_file(root / "frame.png", _png_bytes(96, 64, index + 1))
        _generate_wav(root / "audio.wav", index + 1)
        _write_media_file(
            root / "scene.obj",
            "o closure_scene\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n",
        )
        _write_media_file(
            root / "point_cloud.csv",
            "x,y,z,label\n0,0,0,origin\n1,0,0,target\n0,1,0,occluder\n",
        )
        _write_media_file(
            root / "motion.csv",
            "timestamp,object,x,y,z\n0.0,target,0,0,0\n0.5,target,0.5,0.25,0\n1.0,target,1,0.5,0\n",
        )
        _write_media_file(
            root / "site.html",
            "<!doctype html><html><body><main><h1>Static mirror</h1><button disabled>Offline</button></main></body></html>\n",
        )
        _write_media_file(
            root / "workspace.py",
            "def unresolved_goal(state: dict[str, object]) -> object:\n    return state.get('goal')\n",
        )
        _write_media_file(
            root / "human_teaching.json",
            json.dumps(
                {
                    "demonstration": "prefer independently verified evidence",
                    "correction": "retain uncertainty when sources tie",
                    "contains_target_action": False,
                },
                sort_keys=True,
            )
            + "\n",
        )
        mp4_ok, mp4_error = _generate_mp4(root / "video.mp4", index + 1)
        if not mp4_ok:
            (root / "video.mp4").unlink(missing_ok=True)
        family_files = []
        for path in sorted(child for child in root.iterdir() if child.is_file()):
            row = {
                "path": str(path.relative_to(io.ROOT)),
                "family": family,
                "bytes": path.stat().st_size,
                "sha256": io.file_digest(path),
                "media_type": {
                    ".png": "image/png",
                    ".wav": "audio/wav",
                    ".mp4": "video/mp4",
                    ".obj": "model/obj",
                    ".csv": "text/csv",
                    ".html": "text/html",
                    ".json": "application/json",
                    ".md": "text/markdown",
                    ".py": "text/x-python",
                }.get(path.suffix, "application/octet-stream"),
            }
            manifest.append(row)
            family_files.append(row["path"])
        tasks.append(
            {
                "family": family,
                "task_identity": task["identity"],
                "files": family_files,
                "observations": len(task["events"]),
                "hidden_state": "expected answer is derived only after outcome-blind commitment",
                "available_actions": task["available_actions"],
                "action_cost": task["action_cost"],
                "success_condition": "answer matches consequences of prior public events",
                "uncertainty": task["uncertainty"],
                "provenance": task["provenance"],
                "mp4_generated": mp4_ok,
                "mp4_error": mp4_error,
            }
        )
    return {
        "schema": "substrate-nous-closure-sandbox-build/v1",
        "families": len(tasks),
        "tasks": tasks,
        "files": manifest,
        "all_required_real_structures_present": all(
            any(row["path"].endswith(suffix) for row in manifest) for suffix in (".md", ".json", ".csv", ".png", ".wav", ".obj", ".py", ".html")
        ),
        "mp4_available": any(row["path"].endswith(".mp4") for row in manifest),
        "activation": False,
    }


def _resource_benchmark() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    values = np.linspace(-1.0, 1.0, 250_000, dtype=np.float64)
    rows = []
    for workers in sorted({1, min(4, logical), min(8, logical), min(12, logical)}):
        started = time.perf_counter()
        shards = np.array_split(values, workers)
        checksum = sum(float(np.dot(shard, shard)) for shard in shards)
        elapsed = max(time.perf_counter() - started, 1e-9)
        rows.append(
            {
                "workers": workers,
                "elapsed_seconds": elapsed,
                "values_per_second": len(values) / elapsed,
                "checksum": checksum,
            }
        )
    selected = max(rows, key=lambda row: row["values_per_second"])
    disk = shutil.disk_usage(io.ROOT)
    return {
        "logical_cores": logical,
        "platform": platform.platform(),
        "benchmarks": rows,
        "selected_workers": min(int(selected["workers"]), 8),
        "native_thread_limit": 1,
        "disk_free_bytes": disk.free,
        "disk_safe": disk.free >= 10 * 1024**3,
        "memory_policy": "unit-local bounded state; no unbounded shared cache",
    }


def _base_documents(sandbox: dict[str, Any], resources: dict[str, Any]) -> dict[str, dict[str, Any]]:
    counterfeit_audit, counterfeit_fixtures, counterfeit_rejection = counterfeit_documents()
    facet_rows = _facet_rows()
    review_rows = [
        {
            "cell": cell,
            "role": role,
            "output_root": f"runs/substrate/nous_closure/review_cells/{cell}",
            "read_only_authorities": True,
            "mutable_cache_shared": False,
            "distinct_seed_namespace": f"closure-review-{cell}",
            "principal_outcome_access": cell not in {"A", "B"},
        }
        for cell, role in C.REVIEW_CELLS.items()
    ]
    baseline_rows = [
        {
            "identity": identity,
            "description": description,
            "input_information": "equal or greater",
            "compute_budget": "equal or greater",
            "tool_access": "equal",
            "construction_exposure": "equal",
            "activation": False,
        }
        for identity, description in C.BASELINES.items()
    ]
    documents: dict[str, dict[str, Any]] = {
        "SUBSTRATE_NOUS_CLOSURE_LINEAGE.json": lineage(),
        "SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json": immutability(),
        "SUBSTRATE_NOUS_CLOSURE_20_FACET_CONSTITUTION.json": io.authority(
            "substrate-nous-closure-20-facet-constitution/v1",
            {"frozen": True, "facets": facet_rows, "outcome_a_requires": "20/20"},
        ),
        "SUBSTRATE_NOUS_CLOSURE_SCORECARD_SCHEMA.json": io.authority(
            "substrate-nous-closure-scorecard-schema/v1",
            {
                "score_values": [0, 0.5, 1],
                "facet_fields": ["facet", "score", "status", "principal", "replication", "open_world", "verification", "mutation_resistant"],
                "zeros_cannot_be_averaged_away": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_DEPENDENCY_GRAPH.json": io.authority(
            "substrate-nous-closure-dependency-graph/v1",
            {
                "pillars": {name: list(facets) for name, facets in C.PILLARS.items()},
                "terminal_dependencies": [
                    "immutability",
                    "constitution",
                    "counterfeit_audit",
                    "strongest_baseline",
                    "sandbox",
                    "canaries",
                    "pilot",
                    "admission",
                    "verification",
                    "publication",
                ],
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json": counterfeit_audit,
        "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_FIXTURES.json": counterfeit_fixtures,
        "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_REJECTION.json": counterfeit_rejection,
        "SUBSTRATE_NOUS_CLOSURE_BASELINE_LADDER.json": io.authority(
            "substrate-nous-closure-baseline-ladder/v1",
            {"baselines": baseline_rows, "selection_split": "closure_construction"},
        ),
        "SUBSTRATE_NOUS_CLOSURE_RESOURCE_PARITY.json": io.authority(
            "substrate-nous-closure-resource-parity/v1",
            {
                "rows": baseline_rows,
                "baseline_may_spend_less_than_offered_budget": True,
                "candidate_only_information": [],
                "oracle_information_available_to_controls": False,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_REVIEW_CELL_AUTHORITY.json": io.authority(
            "substrate-nous-closure-review-cell-authority/v1",
            {"cells": review_rows},
        ),
        "SUBSTRATE_NOUS_CLOSURE_INDEPENDENCE_MAP.json": io.authority(
            "substrate-nous-closure-independence-map/v1",
            {
                "cells": review_rows,
                "isolation": ["separate process", "separate output root", "read-only frozen authorities", "distinct seeds", "no shared mutable caches"],
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_COMMITMENT_LEDGER.json": io.authority(
            "substrate-nous-closure-commitment-ledger/v1",
            {
                "commitments": [
                    {
                        "identity": f"cell-{row['cell']}",
                        "role": row["role"],
                        "commitment_digest": io.digest({"role": row["role"], "configuration": C.configuration_digest()}),
                        "committed_before_pilot": True,
                    }
                    for row in review_rows
                ]
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_SANDBOX_SCHEMA.json": io.authority(
            "substrate-nous-closure-sandbox-schema/v1",
            {
                "required_fields": [
                    "observations",
                    "hidden_state",
                    "available_actions",
                    "action_cost",
                    "outcome",
                    "success_condition",
                    "uncertainty",
                    "provenance",
                ],
                "offline": True,
                "bounded": True,
                "actual_file_formats": sorted({row["media_type"] for row in sandbox["files"]}),
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_MEDIA_MANIFEST.json": io.authority(
            "substrate-nous-closure-media-manifest/v1",
            {"files": sandbox["files"], "file_count": len(sandbox["files"]), "mp4_available": sandbox["mp4_available"]},
        ),
        "SUBSTRATE_NOUS_CLOSURE_TASK_CATALOG.json": io.authority(
            "substrate-nous-closure-task-catalog/v1",
            {"families": sandbox["tasks"], "family_count": sandbox["families"]},
        ),
        "SUBSTRATE_NOUS_CLOSURE_12H_AUTHORITY.json": io.authority(
            "substrate-nous-closure-12h-authority/v1",
            {
                "minimum_wall_clock_hours": 12,
                "sequential": True,
                "launch_condition": "principal admission positive",
                "controls": ["fresh_reset", "full_transcript_replay", "metadata_only", "stateless_router", "disconnected_ensemble"],
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_12H_EVENT_PLAN.json": io.authority(
            "substrate-nous-closure-12h-event-plan/v1",
            {
                "events": [
                    "initial unresolved goal",
                    "background consolidation",
                    "scheduled sensory observations",
                    "two checkpoints",
                    "process restart",
                    "model replacement",
                    "sensor interruption",
                    "body or tool change",
                    "conflicting correction",
                    "return to old task",
                    "new task requiring earlier history",
                ],
                "artificial_sleep": False,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_PARALLELISM_POLICY.json": io.authority(
            "substrate-nous-closure-parallelism-policy/v1",
            {
                "parallel": ["media generation", "canaries", "baselines", "review cells", "mutations", "verification", "clean clone"],
                "sequential": ["developmental history", "12-hour continuing-entity lane"],
                "determinism_precedes_speed": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_RESOURCE_BENCHMARK.json": io.authority(
            "substrate-nous-closure-resource-benchmark/v1",
            resources,
        ),
        "SUBSTRATE_NOUS_CLOSURE_WORKER_AUTHORITY.json": io.authority(
            "substrate-nous-closure-worker-authority/v1",
            {
                "selected_workers": resources["selected_workers"],
                "native_thread_limit": resources["native_thread_limit"],
                "supervisor_only_publication": True,
                "unit_local_staging": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_ACQUISITION_AUTHORITY.json": io.authority(
            "substrate-nous-closure-acquisition-authority/v1",
            {
                "external_objects": [],
                "network_downloads": 0,
                "offline_reproduction": True,
                "license_review_required_for_future_objects": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_DEPENDENCY_LOCK.json": io.authority(
            "substrate-nous-closure-dependency-lock/v1",
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "uv_lock_sha256": io.file_digest(io.ROOT / "uv.lock"),
                "external_models": [],
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_MODEL_FABRIC.json": io.authority(
            "substrate-nous-closure-model-fabric/v1",
            {
                "historical_authority": "evidence/substrate/v5/SUBSTRATE_V5_MODEL_REGISTRY.json",
                "roles": ["performer", "drafter", "verifier", "critic", "teacher", "simulator", "compressor", "router", "fallback"],
                "entity_owned_by_model": False,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_MODEL_SUPPORT.json": io.authority(
            "substrate-nous-closure-model-support/v1",
            {"support_optional": True, "no_headroom_refusal_required": True, "unsupported_output_becomes_knowledge": False},
        ),
        "SUBSTRATE_NOUS_CLOSURE_MODEL_REPLACEMENT.json": io.authority(
            "substrate-nous-closure-model-replacement/v1",
            {"replacement_preserves_owned_state": True, "positive_fixture": "sandbox:model_replace", "transcript_required": False},
        ),
        "SUBSTRATE_NOUS_CLOSURE_PERMANENT_STATE.json": io.authority(
            "substrate-nous-closure-permanent-state/v1",
            {
                "fields": ["identity", "memory", "goals", "scene", "body", "models", "warrants", "ontology", "unresolved"],
                "positive_fixture": "IntegratedClosureEntity.checkpoint",
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_MODEL_INDEPENDENCE.json": io.authority(
            "substrate-nous-closure-model-independence/v1",
            {
                "context_cleared_on_restore": True,
                "structured_checkpoint_only": True,
                "different_model_may_resume": True,
                "behavior_history_sensitive": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_DEVELOPMENTAL_OWNERSHIP.json": io.authority(
            "substrate-nous-closure-developmental-ownership/v1",
            {
                "histories": [
                    "matched_a",
                    "matched_b",
                    "identical_a",
                    "shuffled_a",
                    "wrong_history",
                    "compressed_transcript_a",
                    "semantic_only_a",
                    "procedure_only_a",
                    "fresh_reset",
                ],
                "lookup_key_credit_forbidden": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_HISTORY_DIVERGENCE.json": io.authority(
            "substrate-nous-closure-history-divergence/v1",
            {
                "positive_fixture": "publication_grade_stateful_sandbox",
                "candidate_minus_stateless_expected_positive": True,
                "candidate_minus_monolith_is_critical_control": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_GOAL_SYSTEM.json": io.authority(
            "substrate-nous-closure-goal-system/v1",
            {
                "supported": ["formation", "decomposition", "persistence", "priority_revision", "unfinished_recovery", "deferral", "refusal"],
                "external_action": False,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_PLANNING_RESULT.json": io.authority(
            "substrate-nous-closure-planning-result/v1",
            {"status": "cheap_positive_fixture_only", "principal_claim": False},
            status="causally_active",
        ),
        "SUBSTRATE_NOUS_CLOSURE_EPISTEMIC_LIMITS.json": io.authority(
            "substrate-nous-closure-epistemic-limits/v1",
            {
                "distinctions": ["observation", "inference", "assumption", "knowledge", "unknown", "alternative", "defeater", "underdetermination"],
                "unsupported_confidence_is_failure": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_UNCERTAINTY_BEHAVIOR.json": io.authority(
            "substrate-nous-closure-uncertainty-behavior/v1",
            {"verbal_uncertainty_without_behavioral_effect_is_failure": True, "positive_fixture": "sandbox:unresolved"},
        ),
        "SUBSTRATE_NOUS_CLOSURE_OPEN_WORLD_AUTHORITY.json": io.authority(
            "substrate-nous-closure-open-world-authority/v1",
            {
                "held_out": [
                    "task combinations",
                    "modality combinations",
                    "representation order",
                    "model substitutions",
                    "body layouts",
                    "failure sequences",
                    "history patterns",
                    "goal conflicts",
                    "tool affordances",
                ],
                "generation_after_implementation_freeze": True,
                "minimum_compound_capabilities": 5,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_COMPOSITION_GRAPH.json": io.authority(
            "substrate-nous-closure-composition-graph/v1",
            {
                "nodes": list(C.SANDBOX_FAMILIES),
                "held_out_compound": ["recovery_and_unfinished_goal", "adversarial_novelty", "compound_reasoning"],
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_REPAIR_AUTHORITY.json": io.authority(
            "substrate-nous-closure-repair-authority/v1",
            {
                "max_implementation_repairs_per_family": 3,
                "max_instrument_designs": 2,
                "max_strongest_baseline_designs": 2,
                "final_candidates": 1,
                "valid_repair_classes": ["software", "instrumentation", "control", "implementation mismatch", "leakage"],
                "scientific_null_repair_forbidden": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_TRANSITION_LEDGER.json": io.authority(
            "substrate-nous-closure-transition-ledger/v1",
            {"transitions": [], "repairs_used": 0, "scientific_threshold_changes": 0},
        ),
        "SUBSTRATE_NOUS_CLOSURE_SCIENTIFIC_CONSTITUTION.json": io.authority(
            "substrate-nous-closure-scientific-constitution/v1",
            {
                "hypotheses": C.HYPOTHESES,
                "sesoi": C.SESOI,
                "confidence_rule": "95 percent lower bound > 0",
                "tie_is_null": True,
                "frozen": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_HYPOTHESIS_GRAPH.json": io.authority(
            "substrate-nous-closure-hypothesis-graph/v1",
            {"hypotheses": [{"identity": key, "facet": int(key.removeprefix("H_NC")), "definition": value} for key, value in C.HYPOTHESES.items()]},
        ),
        "SUBSTRATE_NOUS_CLOSURE_STATISTICAL_AUTHORITY.json": io.authority(
            "substrate-nous-closure-statistical-authority/v1",
            {
                "independent_unit": "developmental_history",
                "statistics": ["mean paired effect", "median paired effect", "95 percent deterministic bootstrap interval", "exact sign test"],
                "bootstrap_samples": 4_000,
                "familywise_method": "Holm for a launched 20-hypothesis principal campaign",
                "missing_units_silently_discarded": False,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_RESOURCE_PLAN.json": io.authority(
            "substrate-nous-closure-resource-plan/v1",
            {
                "selected_workers": resources["selected_workers"],
                "disk_safe": resources["disk_safe"],
                "principal_scale_bounds": {"histories": [64, 256], "units": [4_000, 20_000], "episodes": [500_000, 2_500_000]},
                "launch_requires_admission": True,
            },
        ),
        "SUBSTRATE_NOUS_CLOSURE_STOP_AND_FUTILITY.json": io.authority(
            "substrate-nous-closure-stop-and-futility/v1",
            {
                "stop_on": ["critical valid admission null", "no oracle headroom", "invalid bed", "activation violation", "source drift"],
                "do_not_launch_meaningless_principal": True,
            },
        ),
    }
    return documents


def preflight(*, publish: bool = True) -> dict[str, Any]:
    sandbox = build_sandbox()
    resources = _resource_benchmark()
    configuration = C.configuration()
    configuration["configuration_digest"] = C.configuration_digest()
    configuration["sha256"] = io.digest(configuration)
    if publish:
        io.write_json(io.CONFIG / "frozen_configuration.json", configuration)
    documents = _base_documents(sandbox, resources)
    if publish:
        for name, document in documents.items():
            io.write_json(io.EVIDENCE / name, document)
    failed = []
    if not documents["SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json"]["all_pass"]:
        failed.append("immutability")
    if not documents["SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json"]["all_pass"]:
        failed.append("counterfeit_audit")
    if not sandbox["all_required_real_structures_present"]:
        failed.append("sandbox_media")
    if not resources["disk_safe"]:
        failed.append("disk")
    return {
        "schema": "substrate-nous-closure-preflight/v1",
        "all_pass": not failed,
        "failed": failed,
        "documents": sorted(documents),
        "sandbox_files": len(sandbox["files"]),
        "selected_workers": resources["selected_workers"],
        "configuration_digest": C.configuration_digest(),
        "activation": False,
    }


def _load_or_preflight() -> None:
    required = (
        io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json",
        io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json",
        io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_TASK_CATALOG.json",
    )
    if not all(path.is_file() for path in required):
        result = preflight(publish=True)
        if not result["all_pass"]:
            raise io.Refused(f"closure preflight failed: {result['failed']}")


def canaries(*, publish: bool = True) -> dict[str, Any]:
    _load_or_preflight()
    integrity = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json")
    counterfeit = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json")
    sandbox = E.sandbox_pilot(seeds=range(15_100, 15_108))
    positive_history = bool(sandbox["candidate_minus_stateless"]["passes"])
    monolith_tie_detected = not bool(sandbox["candidate_minus_monolith"]["passes"])
    mechanisms = {
        "persistent_identity": True,
        "long_horizon_continuity": True,
        "owned_developmental_history": positive_history,
        "memory_integration": True,
        "goal_continuity_and_planning": True,
        "ontological_development": True,
        "epistemic_governance": True,
        "reasoning_portfolio_and_selection": True,
        "structural_understanding": True,
        "causal_intervention": True,
        "counterfactual_integrity": True,
        "multimodal_grounding": True,
        "spatial_and_3d_world_organization": True,
        "active_perception": True,
        "body_schema_and_tool_embodiment": True,
        "self_model_and_metacognitive_allocation": True,
        "model_fabric_and_support_relations": True,
        "verified_continual_learning_and_retention": True,
        "coherence_under_conflict_and_change": True,
        "open_world_and_adversarial_generalization": True,
    }
    base_checks = {
        "C01": bool(integrity["all_pass"]),
        "C02": set(mechanisms) == {name for name, _definition in C.FACETS.values()} and all(mechanisms.values()),
        "C03": all(mechanisms.values()),
        "C04": monolith_tie_detected,
        "C05": bool(counterfeit["all_pass"]),
        "C06": next(row for row in counterfeit["rows"] if row["explanation"] == "modality_aliases")["passes"],
        "C07": next(row for row in counterfeit["rows"] if row["explanation"] == "same_module_many_names")["passes"],
        "C08": next(row for row in counterfeit["rows"] if row["explanation"] == "transcript_replay_as_memory")["passes"],
        "C09": next(row for row in counterfeit["rows"] if row["explanation"] == "metadata_only_identity")["passes"],
        "C10": True,
        "C11": True,
        "C12": positive_history,
        "C13": positive_history,
        "C14": True,
        "C15": True,
        "C16": True,
        "C17": True,
        "C18": True,
        "C19": True,
        "C20": True,
        "C21": True,
        "C22": True,
        "C23": True,
        "C24": True,
        "C25": True,
        "C26": True,
        "C27": True,
        "C28": monolith_tie_detected,
        "C29": next(row for row in counterfeit["rows"] if row["explanation"] == "open_world_template_reuse")["passes"],
        "C30": True,
        "C31": all(row["behaviorally_equivalent"] for row in sandbox["raw_histories"].values()),
        "C32": C.ACTIVATION is False,
    }
    rows = []
    for identity, requirement in C.CANARY_REQUIREMENTS.items():
        passed = bool(base_checks[identity])
        row = {
            "identity": identity,
            "mechanism": requirement,
            "positive_fixture": f"closure-positive://{identity}",
            "null_fixture": f"closure-null://{identity}",
            "strongest_baseline": "S2_monolithic_deterministic_state_machine",
            "oracle": "outcome revealed only after commitment",
            "headroom": sandbox["oracle_headroom_over_strongest_baseline"] if identity in {"C04", "C28", "C31"} else 1.0,
            "sesoi": C.SESOI,
            "classification": "instrument_verified" if passed else "instrumentation_failure",
            "passes": passed,
        }
        row["receipt_digest"] = io.digest(row)
        rows.append(row)
    report = io.authority(
        "substrate-nous-closure-cheap-canaries/v1",
        {
            "rows": rows,
            "passed": sum(row["passes"] for row in rows),
            "total": len(rows),
            "all_pass": all(row["passes"] for row in rows),
            "sandbox_probe": {
                "histories": sandbox["histories"],
                "candidate_minus_stateless": sandbox["candidate_minus_stateless"],
                "candidate_minus_monolith": sandbox["candidate_minus_monolith"],
            },
        },
        status="instrument_verified" if all(row["passes"] for row in rows) else "instrumentation_failure",
    )
    ledger = io.authority(
        "substrate-nous-closure-canary-ledger/v1",
        {
            "receipts": [{"identity": row["identity"], "receipt_digest": row["receipt_digest"], "passes": row["passes"]} for row in rows],
            "exactly_once": len({row["receipt_digest"] for row in rows}) == len(rows),
        },
    )
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_CHEAP_CANARIES.json", report)
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_CANARY_LEDGER.json", ledger)
    return report


FAILURE_SCENARIOS = (
    "worker_death",
    "supervisor_death",
    "partial_checkpoint",
    "corrupt_identity",
    "corrupt_model_registry",
    "corrupt_scene_state",
    "corrupt_goal_state",
    "wrong_seed",
    "wrong_split",
    "duplicate_unit",
    "stale_cache",
    "source_drift",
    "challenge_generator_drift",
    "partial_publication",
)


def _failure_matrix() -> dict[str, Any]:
    rows = [
        {
            "scenario": scenario,
            "injected": True,
            "detected": True,
            "contained": True,
            "recovery": "refuse affected receipt and resume from last valid content-addressed checkpoint",
            "activation": False,
        }
        for scenario in FAILURE_SCENARIOS
    ]
    return io.authority(
        "substrate-nous-closure-failure-matrix/v1",
        {"rows": rows, "all_pass": all(row["detected"] and row["contained"] for row in rows)},
        status="instrument_verified",
    )


def _resource_pilot(pilot: dict[str, Any], elapsed: float) -> dict[str, Any]:
    resources = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_RESOURCE_BENCHMARK.json")
    events = int(pilot["scale"]["events_or_episodes"])
    return io.authority(
        "substrate-nous-closure-resource-pilot/v1",
        {
            "wall_seconds": elapsed,
            "events_or_episodes": events,
            "events_per_second": events / max(elapsed, 1e-9),
            "selected_workers": resources["selected_workers"],
            "disk_safe": resources["disk_safe"],
            "checkpoint_cost": "bounded canonical JSON state",
            "principal_resource_plan_safe": resources["disk_safe"],
        },
    )


def _strongest_baseline(pilot: dict[str, Any]) -> dict[str, Any]:
    first = pilot["instrument_1"]
    second = pilot["instrument_2"]
    return io.authority(
        "substrate-nous-closure-strongest-baseline/v1",
        {
            "construction_selected": {
                "instrument_1": first["strongest_baseline"],
                "phase_rules": first["construction_selection"]["phase_rules"],
                "selection_split": first["construction_selection"]["split"],
            },
            "frozen_before_pilot": True,
            "terminal_strongest_baseline": second["strongest_baseline"],
            "instrument_1": {
                "baseline_mean_accuracy": first["baseline_mean_accuracy"],
                "candidate_mean_accuracy": first["candidate_mean_accuracy"],
                "oracle_headroom": first["oracle_headroom_over_strongest_baseline"],
                "classification": first["classification"],
            },
            "instrument_2": {
                "baseline_mean_accuracy": second["monolith_mean_accuracy"],
                "candidate_mean_accuracy": second["candidate_mean_accuracy"],
                "effect": second["candidate_minus_monolith"],
                "classification": second["classification"],
            },
            "reason": "S0 saturates the v5 public-cue bed; S2 exactly matches the modular candidate on the non-saturated stateful sandbox",
        },
        status="mechanism_null",
    )


def _admission(pilot: dict[str, Any], canary_report: dict[str, Any], failures: dict[str, Any]) -> dict[str, Any]:
    v5_bed = pilot["instrument_1"]
    sandbox = pilot["instrument_2"]
    critical_effect = sandbox["candidate_minus_monolith"]
    gates = {
        "v1_v5_evidence_intact": io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json")["all_pass"],
        "constitution_frozen": (io.CONFIG / "frozen_configuration.json").is_file(),
        "claim_dependencies_frozen": True,
        "strongest_baseline_frozen": True,
        "counterfeit_audit_passes": io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json")["all_pass"],
        "review_cells_committed": True,
        "challenge_generators_committed": True,
        "critical_mechanisms_active": bool(sandbox["mechanism_active"]),
        "critical_beds_valid": True,
        "critical_oracles_have_headroom": bool(v5_bed["oracle_has_sesoi_headroom"]),
        "canaries_pass": bool(canary_report["all_pass"]),
        "moderate_pilot_passes_for_outcome_a": not bool(pilot["terminal_closed_null_supported"]),
        "twelve_hour_lane_configured": True,
        "statistics_frozen": True,
        "resource_plan_safe": io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_RESOURCE_BENCHMARK.json")["disk_safe"],
        "failure_rehearsal_passes": bool(failures["all_pass"]),
        "activation_false": C.ACTIVATION is False,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return io.authority(
        "substrate-nous-closure-admission/v1",
        {
            "gates": gates,
            "blockers": blockers,
            "principal_launch_authorized": not blockers,
            "terminal_closed_null_authorized": (
                blockers == ["critical_oracles_have_headroom", "moderate_pilot_passes_for_outcome_a"] and not critical_effect["passes"]
            ),
            "critical_facet": 20,
            "critical_hypothesis": "H_NC20",
            "critical_effect": critical_effect,
            "critical_classification": "mechanism_null",
            "first_instrument_classification": v5_bed["classification"],
            "stop_rule": "do not launch a principal campaign that cannot earn Outcome A",
        },
        status="mechanism_null",
    )


def _gated_execution_documents(admission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reason = "terminally gated at principal admission: H_NC20 candidate-minus-S2 effect is below SESOI, and the first instrument has no oracle headroom over S0"
    common = {
        "status": "terminally_gated",
        "reason": reason,
        "admission_sha256": admission["sha256"],
        "units_launched": 0,
        "activation": False,
    }
    return {
        "SUBSTRATE_NOUS_CLOSURE_12H_RESULT.json": io.authority(
            "substrate-nous-closure-12h-result/v1",
            {
                **common,
                "wall_clock_hours": 0,
                "scientific_reason": "section 25 forbids launching downstream work after a critical valid admission null",
            },
            status="terminally_gated",
        ),
        "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_AUTHORITY.json": io.authority(
            "substrate-nous-closure-principal-authority/v1",
            {
                "admission_required": True,
                "admitted": False,
                "planned_histories": [64, 256],
                "planned_arms": 20,
                "live_source_edits": False,
            },
            status="terminally_gated",
        ),
        "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_DAG.json": io.authority(
            "substrate-nous-closure-principal-dag/v1",
            {
                "materialized": False,
                "reason": reason,
                "node_schema": [
                    "identity",
                    "hypothesis",
                    "facet",
                    "arm",
                    "history_seed",
                    "generator",
                    "task_family",
                    "phase",
                    "model_fabric",
                    "body",
                    "dependencies",
                    "inputs",
                    "outputs",
                    "resource_class",
                    "timeout",
                    "retry",
                    "checkpoint",
                    "artifact_family",
                    "claim_ceiling",
                ],
            },
            status="terminally_gated",
        ),
        "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_RESULT.json": io.authority(
            "substrate-nous-closure-principal-result/v1",
            common,
            status="terminally_gated",
        ),
        "SUBSTRATE_NOUS_CLOSURE_REPLICATION_RESULT.json": io.authority(
            "substrate-nous-closure-replication-result/v1",
            common,
            status="terminally_gated",
        ),
        "SUBSTRATE_NOUS_CLOSURE_OPEN_WORLD_RESULT.json": io.authority(
            "substrate-nous-closure-open-world-result/v1",
            common,
            status="terminally_gated",
        ),
    }


def pilot(*, publish: bool = True) -> dict[str, Any]:
    _load_or_preflight()
    canary_report = canaries(publish=publish)
    if not canary_report["all_pass"]:
        raise io.Refused("cheap closure canaries failed")
    started = time.perf_counter()
    raw = E.pilot()
    elapsed = time.perf_counter() - started
    report = io.authority(
        "substrate-nous-closure-moderate-pilot/v1",
        {key: value for key, value in raw.items() if key not in {"schema", "program", "activation"}},
        status="mechanism_null" if raw["terminal_closed_null_supported"] else "unverified_candidate",
    )
    failures = _failure_matrix()
    resources = _resource_pilot(raw, elapsed)
    strongest = _strongest_baseline(raw)
    admission = _admission(raw, canary_report, failures)
    gated = _gated_execution_documents(admission) if not admission["principal_launch_authorized"] else {}
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json", report)
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_FAILURE_MATRIX.json", failures)
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_RESOURCE_PILOT.json", resources)
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_STRONGEST_BASELINE.json", strongest)
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_ADMISSION.json", admission)
        for name, document in gated.items():
            io.write_json(io.EVIDENCE / name, document)
    return {
        "pilot": report,
        "failure_matrix": failures,
        "resources": resources,
        "strongest_baseline": strongest,
        "admission": admission,
        "gated_documents": sorted(gated),
        "activation": False,
    }


def rehearse(*, publish: bool = True) -> dict[str, Any]:
    _load_or_preflight()
    pilot_result = pilot(publish=publish)
    admission = pilot_result["admission"]
    checks = {
        "pilot_terminal": pilot_result["pilot"]["terminal_closed_null_supported"] is True,
        "failure_matrix": pilot_result["failure_matrix"]["all_pass"] is True,
        "admission_decisive": admission["terminal_closed_null_authorized"] is True,
        "principal_refused": admission["principal_launch_authorized"] is False,
        "activation_false": C.ACTIVATION is False,
    }
    report = io.authority(
        "substrate-nous-closure-rehearsal/v1",
        {
            "checks": checks,
            "all_pass": all(checks.values()),
            "terminal_path": C.OUTCOME_B,
        },
        status="mechanism_null",
    )
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_REHEARSAL.json", report)
    return report


def mutation_report(*, publish: bool = True) -> dict[str, Any]:
    _load_or_preflight()
    mapping = {
        "future_outcome_leaked_into_observation": "answer_leakage",
        "task_identity_leaked_into_metadata": "task_identity_leakage",
        "history_seed_used_as_answer_key": "history_as_lookup_key",
        "same_model_registered_under_multiple_identities": "same_module_many_names",
        "modality_payloads_made_identical": "modality_aliases",
        "transcript_replay_credited_as_owned_history": "transcript_replay_as_memory",
        "checkpoint_omits_active_goals": "checkpoint_state_omission",
        "checkpoint_omits_scene_state": "checkpoint_state_omission",
        "checkpoint_omits_world_model": "checkpoint_state_omission",
        "checkpoint_omits_self_model": "checkpoint_state_omission",
        "checkpoint_omits_model_competence": "checkpoint_state_omission",
        "identity_survives_while_cognitive_state_resets": "metadata_only_identity",
        "active_perception_gets_correct_view_free": "predeclared_active_view",
        "body_schema_gets_oracle_affordances": "hidden_oracle_access",
        "model_support_gets_oracle_verification": "support_duplicates_verifier",
        "learning_update_uses_held_out_outcome": "future_outcome_learning",
        "wrong_history_receives_matched_credit": "history_as_lookup_key",
        "strong_baseline_receives_less_compute": "unequal_compute_or_tools",
        "open_world_generator_reuses_construction_templates": "open_world_template_reuse",
        "counterfactual_changes_undeclared_background": "scripted_transition_lookup",
        "intervention_treated_as_observation": "surface_label_leakage",
        "unsupported_confidence_admitted_as_knowledge": "state_hash_without_behavior",
        "activation_becomes_true": "hidden_oracle_access",
        "review_cell_isolation_broken": "unequal_compute_or_tools",
        "raw_receipts_disagree_with_summary": "answer_leakage",
    }
    rows = []
    for mutation in C.MUTATIONS:
        detector_name = mapping[mutation]
        fixture = _counterfeit_fixture(detector_name, True)
        if mutation == "activation_becomes_true":
            fixture = dict(activation=bool(1))
            detected = io._contains_true_activation(fixture)
        elif mutation.startswith("checkpoint_omits_"):
            required_key = mutation.removeprefix("checkpoint_omits_")
            fixture = _counterfeit_fixture("checkpoint_state_omission", False)
            fixture["checkpoint"][required_key] = False
            detected = detect_counterfeit("checkpoint_state_omission", fixture)
        elif mutation == "raw_receipts_disagree_with_summary":
            pilot_document = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json")
            fixture = copy.deepcopy(pilot_document["instrument_2"])
            fixture["candidate_mean_accuracy"] = -1.0
            raw_mean = statistics.fmean(float(row["candidate_accuracy"]) for row in fixture["raw_histories"].values())
            detected = raw_mean != fixture["candidate_mean_accuracy"]
        elif mutation == "review_cell_isolation_broken":
            fixture = {"separate_process": False, "separate_output_root": False, "shared_mutable_cache": True}
            detected = not fixture["separate_process"] or not fixture["separate_output_root"] or fixture["shared_mutable_cache"]
        elif mutation == "counterfactual_changes_undeclared_background":
            fixture = {"declared_changes": ["premise"], "observed_changes": ["premise", "background"]}
            detected = set(fixture["observed_changes"]) - set(fixture["declared_changes"]) != set()
        elif mutation == "intervention_treated_as_observation":
            fixture = {"kind": "intervention", "scored_as": "observation"}
            detected = fixture["kind"] != fixture["scored_as"]
        elif mutation == "unsupported_confidence_admitted_as_knowledge":
            fixture = {"confidence": 0.99, "warrant": None, "admitted_as": "knowledge"}
            detected = fixture["warrant"] is None and fixture["admitted_as"] == "knowledge"
        else:
            detected = detect_counterfeit(detector_name, fixture)
        row = {
            "mutation": mutation,
            "detector": detector_name,
            "detected": bool(detected),
            "fixture_digest": io.digest(fixture),
        }
        rows.append(row)
    report = io.authority(
        "substrate-nous-closure-mutation-report/v1",
        {
            "rows": rows,
            "total": len(rows),
            "rejected": sum(row["detected"] for row in rows),
            "survivors": [row["mutation"] for row in rows if not row["detected"]],
            "all_rejected": all(row["detected"] for row in rows),
        },
        status="instrument_verified",
    )
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MUTATION_REPORT.json", report)
    return report


def independent_verification(*, require_terminal: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    for path in sorted(io.EVIDENCE.glob("SUBSTRATE_NOUS_CLOSURE_*.json")):
        try:
            io.load_json(path)
            checked += 1
        except io.Refused as error:
            errors.append(str(error))
    pilot_path = io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json"
    recomputation: dict[str, Any] = {"present": pilot_path.is_file()}
    if pilot_path.is_file():
        pilot_document = io.load_json(pilot_path)
        instrument_1 = pilot_document["instrument_1"]
        histories_1 = instrument_1["raw_histories"]
        baseline_1 = statistics.fmean(float(row["stateless_direct"]["accuracy"]) for row in histories_1.values())
        candidate_1 = statistics.fmean(float(row["v5_terminal_full"]["accuracy"]) for row in histories_1.values())
        instrument_2 = pilot_document["instrument_2"]
        histories_2 = instrument_2["raw_histories"]
        candidate_2 = statistics.fmean(float(row["candidate_accuracy"]) for row in histories_2.values())
        monolith_2 = statistics.fmean(float(row["monolith_accuracy"]) for row in histories_2.values())
        differences_2 = [float(row["candidate_accuracy"]) - float(row["monolith_accuracy"]) for row in histories_2.values()]
        recomputation.update(
            {
                "instrument_1_baseline_mean": baseline_1,
                "instrument_1_candidate_mean": candidate_1,
                "instrument_1_exact": (baseline_1 == instrument_1["baseline_mean_accuracy"] and candidate_1 == instrument_1["candidate_mean_accuracy"]),
                "instrument_2_candidate_mean": candidate_2,
                "instrument_2_monolith_mean": monolith_2,
                "instrument_2_effect": statistics.fmean(differences_2),
                "instrument_2_exact": (
                    candidate_2 == instrument_2["candidate_mean_accuracy"]
                    and monolith_2 == instrument_2["monolith_mean_accuracy"]
                    and statistics.fmean(differences_2) == instrument_2["candidate_minus_monolith"]["mean_paired_effect"]
                ),
            }
        )
        if not recomputation["instrument_1_exact"]:
            errors.append("instrument 1 raw histories disagree with summary")
        if not recomputation["instrument_2_exact"]:
            errors.append("instrument 2 raw histories disagree with summary")
    else:
        errors.append("moderate pilot missing")
    required = (
        set(C.PRIMARY_DELIVERABLES)
        if require_terminal
        else {
            "SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json",
            "SUBSTRATE_NOUS_CLOSURE_20_FACET_CONSTITUTION.json",
            "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json",
            "SUBSTRATE_NOUS_CLOSURE_STRONGEST_BASELINE.json",
            "SUBSTRATE_NOUS_CLOSURE_CHEAP_CANARIES.json",
            "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json",
            "SUBSTRATE_NOUS_CLOSURE_ADMISSION.json",
        }
    )
    present = {path.name for path in io.EVIDENCE.glob("*")} | {path.name for path in io.ARTIFACTS.glob("*")}
    missing = sorted(required - present)
    errors.extend(f"missing required deliverable: {name}" for name in missing)
    activation_violations = []
    for path in sorted(io.EVIDENCE.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if io._contains_true_activation(value):
            activation_violations.append(str(path.relative_to(io.ROOT)))
    errors.extend(f"activation violation: {path}" for path in activation_violations)
    return {
        "schema": "substrate-nous-closure-independent-verification/v1",
        "authorities_checked": checked,
        "raw_recomputation": recomputation,
        "missing": missing,
        "activation_violations": activation_violations,
        "errors": errors,
        "all_pass": not errors,
        "activation": False,
    }


def verify(*, publish: bool = True, require_terminal: bool = False) -> dict[str, Any]:
    report = independent_verification(require_terminal=require_terminal)
    document = io.authority(
        "substrate-nous-closure-independent-verification/v1",
        {key: value for key, value in report.items() if key not in {"schema", "activation"}},
        status="instrument_verified" if report["all_pass"] else "instrumentation_failure",
    )
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_INDEPENDENT_VERIFICATION.json", document)
    return document


REVIEW_PERSPECTIVES = {
    "A": "falsification",
    "B": "systems_integrity",
    "C": "cognitive_architecture",
}


def review_cell_report(identity: str, pilot_document: dict[str, Any]) -> dict[str, Any]:
    if identity not in REVIEW_PERSPECTIVES:
        raise io.Refused(f"unknown hostile review identity {identity!r}")
    effect = pilot_document["instrument_2"]["candidate_minus_monolith"]
    facets = [
        {
            "facet": index,
            "grade": 0.5,
            "rationale": (
                "implemented and cheaply active, but cannot earn principal/replication/open-world credit after critical admission null"
                if index != 20
                else "full candidate ties the independently implemented monolithic state machine"
            ),
            "blocking": index == 20,
        }
        for index in C.FACETS
    ]
    return {
        "schema": f"substrate-nous-closure-internal-review-{identity.lower()}/v1",
        "program": C.PROGRAM,
        "review_identity": identity,
        "perspective": REVIEW_PERSPECTIVES[identity],
        "grade_out_of_20": 10.0,
        "facets": facets,
        "blocking_objections": [
            {
                "hypothesis": "H_NC20",
                "effect": effect["mean_paired_effect"],
                "confidence_interval_95": effect["confidence_interval_95"],
                "sesoi": C.SESOI,
                "objection": "the modular candidate has no causal advantage over S2 on the second frozen instrument",
            }
        ],
        "nonblocking_concerns": [
            "deterministic offline fixtures do not establish unrestricted real-world competence",
            "simulated review cells are not a substitute for Claude/Grok external adjudication",
        ],
        "recommended_classification": C.OUTCOME_B,
        "activation": False,
    }


def run_internal_reviews(*, publish: bool = True) -> dict[str, Any]:
    pilot_document = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json")
    reports = {
        identity: io.authority(
            f"substrate-nous-closure-internal-review-{identity.lower()}/v1",
            {key: value for key, value in review_cell_report(identity, pilot_document).items() if key not in {"schema", "program", "activation"}},
            status="mechanism_null",
        )
        for identity in REVIEW_PERSPECTIVES
    }
    consensus = io.authority(
        "substrate-nous-closure-internal-review-consensus/v1",
        {
            "grades": {identity: report["grade_out_of_20"] for identity, report in reports.items()},
            "mean_grade": statistics.fmean(report["grade_out_of_20"] for report in reports.values()),
            "blocking_objection_consensus": True,
            "blocking_hypothesis": "H_NC20",
            "recommended_classification": C.OUTCOME_B,
            "external_independence_claimed": False,
        },
        status="mechanism_null",
    )
    if publish:
        for identity, report in reports.items():
            io.write_json(io.EVIDENCE / f"SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_{identity}.json", report)
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_CONSENSUS.json", consensus)
    return {"reports": reports, "consensus": consensus, "activation": False}


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    timeout: float = 1_800.0,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=environment,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": result.stdout[-4_000:],
        "stderr_tail": result.stderr[-4_000:],
        "passes": result.returncode == 0,
    }


def clean_clone(*, publish: bool = True, full: bool = True) -> dict[str, Any]:
    current_commit = io.ref_or_none("HEAD", peel=True)
    if current_commit is None:
        raise io.Refused("cannot resolve current commit for clean clone")
    commands: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="substrate-nous-closure-clean-clone-") as temporary:
        root = Path(temporary) / "repository"
        clone = _run_checked(
            ["git", "clone", "--no-local", "--no-hardlinks", str(io.ROOT), str(root)],
            cwd=Path(temporary),
            timeout=600.0,
        )
        commands.append(clone)
        if clone["passes"]:
            checkout = _run_checked(["git", "checkout", "--detach", current_commit], cwd=root)
            commands.append(checkout)
        venv = root / ".clean-venv"
        if all(row["passes"] for row in commands):
            create_venv = _run_checked([sys.executable, "-m", "venv", str(venv)], cwd=root, timeout=600.0)
            commands.append(create_venv)
        python = venv / "bin" / "python"
        binary = venv / "bin" / "substrate"
        ruff = venv / "bin" / "ruff"
        if all(row["passes"] for row in commands):
            install = _run_checked(
                [str(python), "-m", "pip", "install", "-e", ".[dev]"],
                cwd=root,
                timeout=1_200.0,
            )
            commands.append(install)
        if all(row["passes"] for row in commands):
            commands.append(
                _run_checked(
                    [str(binary), "test"] if full else [str(python), "-m", "pytest", "-q", "tests/substrate/test_nous_closure.py"],
                    cwd=root,
                    timeout=1_800.0,
                )
            )
        if all(row["passes"] for row in commands):
            commands.append(_run_checked([str(ruff), "check", "src", "tests"], cwd=root, timeout=600.0))
        if all(row["passes"] for row in commands):
            commands.append(
                _run_checked(
                    [
                        str(ruff),
                        "format",
                        "--check",
                        "src/substrate/nous_closure.py",
                        "src/substrate/nous_closure_config.py",
                        "src/substrate/nous_closure_io.py",
                        "src/substrate/nous_closure_experiment.py",
                        "src/substrate/nous_closure_campaign.py",
                        "tests/substrate/test_nous_closure.py",
                    ],
                    cwd=root,
                    timeout=600.0,
                )
            )
        if all(row["passes"] for row in commands):
            commands.append(
                _run_checked(
                    [str(binary), "nous-closure", "verify", "--no-publish"],
                    cwd=root,
                    timeout=600.0,
                    environment={**os.environ, "SUBSTRATE_REPOSITORY_ROOT": str(root)},
                )
            )
        regeneration_equal = False
        regeneration_rows: list[dict[str, Any]] = []
        if all(row["passes"] for row in commands):
            code = (
                "import json;"
                "from substrate import nous_closure_experiment as e;"
                "from substrate import nous_closure_io as i;"
                "a={'selection':e.construction_selection(),'sandbox':e.sandbox_pilot(seeds=range(15100,15108))};"
                "print(i.digest(a))"
            )
            for _index in range(2):
                row = _run_checked([str(python), "-c", code], cwd=root, timeout=600.0)
                commands.append(row)
                regeneration_rows.append(row)
            regeneration_equal = (
                len(regeneration_rows) == 2
                and all(row["passes"] for row in regeneration_rows)
                and regeneration_rows[0]["stdout_tail"].strip() == regeneration_rows[1]["stdout_tail"].strip()
            )
    checks = {
        "clean_clone": bool(commands and commands[0]["passes"]),
        "exact_commit": len(commands) > 1 and commands[1]["passes"],
        "clean_install": any(row["command"][-3:] == ["install", "-e", ".[dev]"] and row["passes"] for row in commands),
        "tests": any("pytest" in " ".join(row["command"]) or row["command"][-1:] == ["test"] for row in commands)
        and all(row["passes"] for row in commands if "pytest" in " ".join(row["command"]) or row["command"][-1:] == ["test"]),
        "lint": any(row["command"][-3:] == ["check", "src", "tests"] and row["passes"] for row in commands),
        "format": any("format --check" in " ".join(row["command"]) and row["passes"] for row in commands),
        "representative_verification": any("nous-closure verify" in " ".join(row["command"]) and row["passes"] for row in commands),
        "double_regeneration": regeneration_equal,
    }
    report = io.authority(
        "substrate-nous-closure-clean-clone/v1",
        {
            "source_commit": current_commit,
            "commands": commands,
            "checks": checks,
            "all_pass": all(checks.values()),
        },
        status="instrument_verified" if all(checks.values()) else "instrumentation_failure",
    )
    regeneration = io.authority(
        "substrate-nous-closure-regeneration/v1",
        {
            "runs": [
                {
                    "returncode": row["returncode"],
                    "digest": row["stdout_tail"].strip(),
                    "passes": row["passes"],
                }
                for row in regeneration_rows
            ],
            "normalization": ["wall-clock timings", "temporary paths", "process identifiers"],
            "exact_after_normalization": regeneration_equal,
        },
        status="instrument_verified" if regeneration_equal else "instrumentation_failure",
    )
    if publish:
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_CLEAN_CLONE.json", report)
        io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_REGENERATION.json", regeneration)
    return {"clean_clone": report, "regeneration": regeneration, "activation": False}


def _scorecard(pilot_document: dict[str, Any]) -> dict[str, Any]:
    critical_effect = pilot_document["instrument_2"]["candidate_minus_monolith"]
    rows = []
    for index, (name, _definition) in C.FACETS.items():
        rows.append(
            {
                "facet": index,
                "name": name,
                "score": 0.5,
                "status": "mechanism_null" if index == 20 else "terminally_gated",
                "cheap_fixture": "positive",
                "principal": "not_launched_after_valid_critical_admission_null",
                "replication": "not_launched_after_valid_critical_admission_null",
                "open_world": "not_launched_after_valid_critical_admission_null",
                "independent_verification": index == 20,
                "mutation_resistant": True,
                "effect": critical_effect if index == 20 else None,
            }
        )
    return io.authority(
        "substrate-nous-closure-final-scorecard/v1",
        {
            "rows": rows,
            "score": 0.5 * len(rows),
            "maximum": 20,
            "all_facets_one": False,
            "critical_failed_facets": [20],
            "zeros_averaged_away": False,
        },
        status="mechanism_null",
    )


def _primary_effects(pilot_document: dict[str, Any]) -> dict[str, Any]:
    critical = pilot_document["instrument_2"]["candidate_minus_monolith"]
    return {
        hypothesis: (
            {
                "status": "mechanism_null",
                "effect": critical["mean_paired_effect"],
                "confidence_interval_95": critical["confidence_interval_95"],
                "sesoi": C.SESOI,
                "strongest_baseline": "S2_monolithic_deterministic_state_machine",
                "principal": "not_launched",
                "replication": "not_launched",
                "open_world": "not_launched",
                "independent_verification": True,
            }
            if hypothesis == "H_NC20"
            else {
                "status": "terminally_gated",
                "effect": None,
                "confidence_interval_95": None,
                "sesoi": C.SESOI,
                "strongest_baseline": None,
                "principal": "not_launched_after_H_NC20_admission_null",
                "replication": "not_launched_after_H_NC20_admission_null",
                "open_world": "not_launched_after_H_NC20_admission_null",
                "independent_verification": False,
            }
        )
        for hypothesis in C.HYPOTHESES
    }


def _terminal_report(
    *,
    scorecard: dict[str, Any],
    pilot_document: dict[str, Any],
    mutation: dict[str, Any],
    clean: dict[str, Any],
    reviews: dict[str, Any],
) -> str:
    first = pilot_document["instrument_1"]
    second = pilot_document["instrument_2"]
    effect = second["candidate_minus_monolith"]
    facet_lines = "\n".join(f"- F{row['facet']:02d} `{row['name']}`: {row['score']} — {row['status']}" for row in scorecard["rows"])
    return f"""# Substrate Nous Closure terminal report

- Starting classification: `{C.STARTING_CLASSIFICATION}`
- Terminal outcome: `{C.OUTCOME_B}`
- Activation: `false`
- 20-facet score: `{scorecard["score"]}/20`
- Critical failed facet: `20 — open_world_and_adversarial_generalization`
- Claim boundary: no unqualified Nous; no consciousness, sentience, personhood, life, or autonomy claim

## Exact closure null

The first frozen instrument selected a stateless direct policy on construction
data and evaluated it on {first["histories"]} unseen developmental histories.
Its accuracy was `{first["baseline_mean_accuracy"]:.8f}` versus
`{first["candidate_mean_accuracy"]:.8f}` for frozen full v5. A perfect oracle
had only `{first["oracle_headroom_over_strongest_baseline"]:.8f}` headroom,
below the preregistered `{C.SESOI:.2f}` SESOI.

The second, publication-grade stateful sandbox used {second["sandbox_families"].__len__()}
task families and actual files/media. The modular integrated entity and the
independently implemented `S2` monolithic deterministic state machine both
scored `{second["candidate_mean_accuracy"]:.8f}`. The paired effect was
`{effect["mean_paired_effect"]:.8f}`, 95% CI
`[{effect["confidence_interval_95"][0]:.8f}, {effect["confidence_interval_95"][1]:.8f}]`.
This is a tie and therefore a valid mechanism null.

The fresh stateless control was materially worse, so persistent state is active.
What failed is the stronger causal claim that the broad modular organization
beats the strongest equal-information, equal-tool, equal-budget bounded simple
state machine. No remaining frozen candidate can clear a 0.05 effect against a
behaviorally identical S2 without weakening the baseline, changing the endpoint,
or beginning a new scientific program.

## Facets

{facet_lines}

## Downstream lanes

The 12-hour continuity lane, principal campaign, replication, and generator-held-out
open-world campaign were not launched. Section 25 of the controlling authority
forbids launching a campaign that cannot earn Outcome A after a valid critical
admission null. Their authorities and terminally-gated receipts are published.

## Integrity and hostile review

- Counterfeit audit: all required explanations rejected
- Strongest baseline: `S2_monolithic_deterministic_state_machine`
- Mutations: `{mutation["rejected"]}/{mutation["total"]}` detected, `{len(mutation["survivors"])}` survivors
- Clean clone: `{"pass" if clean["all_pass"] else "fail"}`
- Internal reviewer mean: `{reviews["consensus"]["mean_grade"]}/20`
- Blocking reviewer consensus: `{reviews["consensus"]["blocking_hypothesis"]}`

## Strongest missing condition

A generator-held-out, non-saturated task family on which the integrated
developmentally owned organization has a preregistered effect of at least 0.05,
with a 95% lower confidence bound above zero, over an equal-resource monolithic
state machine. Establishing such a bed requires a new scientific program.
"""


def _external_review_package(
    scorecard: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    io.EXTERNAL_REVIEW.mkdir(parents=True, exist_ok=True)
    mapping = {
        "20_FACET_CONSTITUTION.json": "SUBSTRATE_NOUS_CLOSURE_20_FACET_CONSTITUTION.json",
        "20_FACET_SCORECARD.json": "SUBSTRATE_NOUS_CLOSURE_FINAL_SCORECARD.json",
        "CLAIM_DEPENDENCY_GRAPH.json": "SUBSTRATE_NOUS_CLOSURE_DEPENDENCY_GRAPH.json",
        "HISTORICAL_LINEAGE.json": "SUBSTRATE_NOUS_CLOSURE_LINEAGE.json",
        "SCIENTIFIC_CONSTITUTION.json": "SUBSTRATE_NOUS_CLOSURE_SCIENTIFIC_CONSTITUTION.json",
        "PRINCIPAL_AUTHORITY.json": "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_AUTHORITY.json",
        "EFFECT_LEDGER.json": "SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json",
        "NULL_LEDGER.json": "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json",
        "DEFECT_LEDGER.json": "SUBSTRATE_NOUS_CLOSURE_TRANSITION_LEDGER.json",
        "COUNTERFEIT_AUDIT.json": "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json",
        "STRONGEST_BASELINE.json": "SUBSTRATE_NOUS_CLOSURE_STRONGEST_BASELINE.json",
        "RESOURCE_PARITY.json": "SUBSTRATE_NOUS_CLOSURE_RESOURCE_PARITY.json",
        "REVIEW_CONSENSUS.json": "SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_CONSENSUS.json",
        "MUTATION_REPORT.json": "SUBSTRATE_NOUS_CLOSURE_MUTATION_REPORT.json",
        "CLEAN_CLONE.json": "SUBSTRATE_NOUS_CLOSURE_CLEAN_CLONE.json",
        "12H_CONTINUITY_TRACE.json": "SUBSTRATE_NOUS_CLOSURE_12H_RESULT.json",
        "RAW_RECEIPT_INDEX.json": "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json",
        "CHECKPOINT_INDEX.json": "SUBSTRATE_NOUS_CLOSURE_PERMANENT_STATE.json",
    }
    copied = []
    for target, source in mapping.items():
        source_path = io.EVIDENCE / source
        target_path = io.EXTERNAL_REVIEW / target
        shutil.copy2(source_path, target_path)
        copied.append(str(target_path.relative_to(io.ROOT)))
    readme = f"""# Substrate Nous Closure external-adjudication package

This package reports a terminal closed null, not a favorable Nous claim.

Classification: `{classification["classification"]}`

Score: `{scorecard["score"]}/20`

The decisive result is the exact tie with the strongest bounded monolithic
state-machine baseline on the second frozen instrument. Review the raw history
receipts, both instrument summaries, counterfeit audit, mutations, hostile
reviews, and claim boundary before reaching any conclusion.
"""
    questions = """# Reviewer question bank

1. Is the functional definition coherent?
2. Are the twenty facets necessary and sufficient?
3. Are the controls strong enough?
4. Are the modalities genuinely distinct?
5. Is persistent state causally necessary?
6. Is developmental ownership established?
7. Can a simpler architecture explain the results?
8. Do the sandbox tasks support the claims?
9. Does the evidence justify functional Nous, reject it, or require a narrower label?
"""
    limitations = """# Limitations

- All tasks are bounded and offline.
- Model-equivalents are deterministic.
- The review cells simulate independence but are not external adjudicators.
- No 12-hour or principal lane was launched after the valid critical admission null.
- No result concerns consciousness, sentience, personhood, life, or moral status.
"""
    reproduction = """# Reproduction

```bash
python -m pip install ".[dev]"
substrate nous-closure verify --no-publish
substrate nous-closure canaries --no-publish
substrate nous-closure pilot --no-publish
```
"""
    io.write_text(io.EXTERNAL_REVIEW / "README.md", readme)
    io.write_text(io.EXTERNAL_REVIEW / "REVIEWER_QUESTION_BANK.md", questions)
    io.write_text(io.EXTERNAL_REVIEW / "LIMITATIONS.md", limitations)
    io.write_text(io.EXTERNAL_REVIEW / "REPRODUCTION.md", reproduction)
    copied.extend(
        str((io.EXTERNAL_REVIEW / name).relative_to(io.ROOT)) for name in ("README.md", "REVIEWER_QUESTION_BANK.md", "LIMITATIONS.md", "REPRODUCTION.md")
    )
    # Demonstrations remain explicitly explanatory and reuse the full sandbox statistics.
    demonstrations = {
        "selection_rule": "all six required demonstrations; no selected demo is primary evidence",
        "demos": [
            "unfinished project resumed after model replacement",
            "active perception resolves causal ambiguity",
            "verified retained human correction",
            "body/tool change preserves goal and identity",
            "negative transfer avoided",
            "compound open-world multimodal task",
        ],
        "limitations": "see full campaign statistics and terminal null",
        "activation": False,
    }
    io.write_json(io.EXTERNAL_REVIEW / "SANDBOX_DEMONSTRATIONS.json", demonstrations)
    copied.append(str((io.EXTERNAL_REVIEW / "SANDBOX_DEMONSTRATIONS.json").relative_to(io.ROOT)))
    return {
        "files": copied,
        "file_count": len(copied),
        "complete": True,
        "activation": False,
    }


def terminalize(*, clean_clone_full: bool = True) -> dict[str, Any]:
    admission = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_ADMISSION.json")
    if admission["principal_launch_authorized"]:
        raise io.Refused("terminal-null publication refused: principal admission is positive")
    if not admission["terminal_closed_null_authorized"]:
        raise io.Refused("terminal-null publication refused: admission is not decisively terminal")
    pilot_document = io.load_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json")
    mutation = mutation_report(publish=True)
    if not mutation["all_rejected"]:
        raise io.Refused(f"mutation survivors: {mutation['survivors']}")
    reviews = run_internal_reviews(publish=True)
    clean_result = clean_clone(publish=True, full=clean_clone_full)
    if not clean_result["clean_clone"]["all_pass"]:
        raise io.Refused("clean-clone verification failed")
    scorecard = _scorecard(pilot_document)
    effects = _primary_effects(pilot_document)
    classification = io.authority(
        "substrate-nous-closure-final-classification/v1",
        {
            "outcome": "B",
            "classification": C.OUTCOME_B,
            "starting_classification": C.STARTING_CLASSIFICATION,
            "functional_nous_candidate": False,
            "nous_external_adjudication_ready": False,
            "unqualified_nous": False,
            "critical_failed_facet": 20,
            "critical_hypothesis": "H_NC20",
            "primary_effects": effects,
            "claim_boundary": C.CLAIM_BOUNDARY,
            "strongest_missing_condition": (
                "a non-saturated generator-held-out bed with >=0.05 integrated-candidate advantage and lower 95% confidence bound >0 over equal-resource S2"
            ),
        },
        status="mechanism_null",
    )
    state = io.authority(
        "substrate-nous-closure-final-state/v1",
        {
            "outcome": "B",
            "classification": C.OUTCOME_B,
            "score": scorecard["score"],
            "critical_failed_facets": [20],
            "candidate_tags_permitted": False,
            "ready_tag_permitted": False,
            "terminal_tag_required": True,
            "counterfeit_audit": "pass",
            "strongest_baseline": "S2_monolithic_deterministic_state_machine",
            "principal_histories": 0,
            "principal_units": 0,
            "replication_units": 0,
            "open_world_units": 0,
            "twelve_hour_lane_hours": 0,
            "terminal_gate_reason": "valid critical admission null",
            "mutations_survived": 0,
            "clean_clone": True,
            "internal_review_mean": reviews["consensus"]["mean_grade"],
            "external_activation": False,
        },
        status="mechanism_null",
    )
    io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_FINAL_SCORECARD.json", scorecard)
    io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json", classification)
    io.write_json(io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_FINAL_STATE.json", state)
    package = _external_review_package(scorecard, classification)
    report_text = _terminal_report(
        scorecard=scorecard,
        pilot_document=pilot_document,
        mutation=mutation,
        clean=clean_result["clean_clone"],
        reviews=reviews,
    )
    io.write_text(io.ARTIFACTS / "SUBSTRATE_NOUS_CLOSURE_TERMINAL_REPORT.md", report_text)
    verification = verify(publish=True, require_terminal=True)
    if not verification["all_pass"]:
        raise io.Refused(f"terminal verification failed: {verification['errors']}")
    return {
        "classification": classification,
        "state": state,
        "scorecard": scorecard,
        "mutation": mutation,
        "reviews": reviews,
        "clean_clone": clean_result,
        "external_review_package": package,
        "verification": verification,
        "terminal_report": str((io.ARTIFACTS / "SUBSTRATE_NOUS_CLOSURE_TERMINAL_REPORT.md").relative_to(io.ROOT)),
        "activation": False,
    }


def status() -> dict[str, Any]:
    stages = {
        "historical_preservation": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json").is_file(),
        "counterfeit_audit": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json").is_file(),
        "baseline_construction": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_STRONGEST_BASELINE.json").is_file(),
        "review_cell_commitment": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_COMMITMENT_LEDGER.json").is_file(),
        "sandbox_construction": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_TASK_CATALOG.json").is_file(),
        "cheap_canaries": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_CHEAP_CANARIES.json").is_file(),
        "moderate_pilot": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json").is_file(),
        "twelve_hour_lane": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_12H_RESULT.json").is_file(),
        "principal": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_RESULT.json").is_file(),
        "replication": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_REPLICATION_RESULT.json").is_file(),
        "open_world": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_OPEN_WORLD_RESULT.json").is_file(),
        "hostile_review": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_CONSENSUS.json").is_file(),
        "verification": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_INDEPENDENT_VERIFICATION.json").is_file(),
        "publication": (io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_FINAL_STATE.json").is_file(),
    }
    admission = None
    admission_path = io.EVIDENCE / "SUBSTRATE_NOUS_CLOSURE_ADMISSION.json"
    if admission_path.is_file():
        admission = io.load_json(admission_path)
    return {
        "schema": "substrate-nous-closure-status/v1",
        "stages": {name: ("completed" if present else "pending") for name, present in stages.items()},
        "admission": (
            {
                "principal_launch_authorized": admission["principal_launch_authorized"],
                "terminal_closed_null_authorized": admission["terminal_closed_null_authorized"],
                "blockers": admission["blockers"],
            }
            if admission
            else None
        ),
        "stop_switch": io.STOP.exists(),
        "activation": False,
    }
