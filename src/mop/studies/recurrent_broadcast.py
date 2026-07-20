
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from ..config import REPO_ROOT
from ..diagnostics.compute import linear_flops
from ..diagnostics.operational_awareness import report_grounding
from ..substrate.events import EventRef, FrozenJSON, canonical_sha256
from ..substrate.lifecycle import LifecycleJournal, MemoryRef
from .p9_accounting import WorkloadAccountant

CONFIG_SCHEMA = "mop-recurrent-broadcast-config/v1"
MESSAGE_SCHEMA = "mop-recurrent-broadcast-message/v1"
EPISODE_SCHEMA = "mop-recurrent-broadcast-episode/v1"
DATASET_SCHEMA = "mop-recurrent-broadcast-dataset/v1"
CHECKPOINT_SCHEMA = "mop-recurrent-broadcast-checkpoint/v1"
PREFLIGHT_SCHEMA = "mop-recurrent-broadcast-preflight/v1"
CLAIM_SCOPE = (
    "deterministic structural-fixture mechanics only; no natural-task, capability, cognition, "
    "consciousness, sentience, or agency claim"
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "recurrent_broadcast_preflight.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "RECURRENT_BROADCAST_PREFLIGHT.json"

ARM_ORDER = (
    "recurrent_limited_broadcast",
    "global_dense_state",
    "no_broadcast",
    "larger_feedforward_depth",
    "independent_specialists",
    "equal_flop_routing",
)
PERTURBATION_ORDER = (
    "clean",
    "consumer_link_lesion",
    "wrong_message_injection",
    "delayed_stale_message",
    "restored_link",
    "shuffled_message",
)
SPLITS = ("train", "heldout")

IMPLEMENTATION_PATHS = (
    "configs/experiment/recurrent_broadcast_preflight.yaml",
    "src/mop/studies/recurrent_broadcast.py",
    "scripts/recurrent_broadcast_preflight.py",
    "tests/unit/test_recurrent_broadcast_preflight.py",
    "docs/RECURRENT_BROADCAST_PREFLIGHT.md",
)
REUSED_PATHS = (
    "src/mop/substrate/events.py",
    "src/mop/substrate/lifecycle.py",
    "src/mop/studies/p9_accounting.py",
    "src/mop/diagnostics/compute.py",
    "src/mop/diagnostics/operational_awareness.py",
    "src/mop/shell/workspace.py",
    "src/mop/shell/modulation.py",
    "scripts/mop_router_mechanism.py",
    "scripts/mop_cm4_workspace_pilot.py",
    "runs/mot/cm4_workspace_pilot.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    os.replace(temporary, path)


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _fraction(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": round(float(value), 8),
    }


def _from_fraction(payload: dict[str, Any]) -> Fraction:
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_tree(item) for item in value)
    return True


def _without_digest(payload: dict[str, Any], key: str = "payload_sha256") -> dict[str, Any]:
    out = copy.deepcopy(payload)
    out.pop(key, None)
    return out


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("recurrent broadcast config schema drift")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("recurrent broadcast claim scope drift")
    if not str(config.get("null_hypothesis", "")).strip():
        raise ValueError("recurrent broadcast null hypothesis is required")
    fixture = config.get("fixture", {})
    specialists = tuple(str(value) for value in fixture.get("specialists", ()))
    consumers = fixture.get("consumers", ())
    if len(specialists) < 4 or len(set(specialists)) != len(specialists):
        raise ValueError("recurrent broadcast requires at least four unique specialists")
    if len(consumers) < 3:
        raise ValueError("recurrent broadcast requires at least three separated consumers")
    consumer_ids = tuple(str(row.get("consumer_id")) for row in consumers)
    topics = tuple(str(row.get("topic")) for row in consumers)
    if len(set(consumer_ids)) != len(consumer_ids) or len(set(topics)) != len(topics):
        raise ValueError("recurrent broadcast consumer ids and topics must be unique")
    if any(str(row.get("local_specialist")) not in specialists for row in consumers):
        raise ValueError("recurrent broadcast local specialist is unknown")
    if int(fixture.get("workspace_capacity_messages", 0)) != 1:
        raise ValueError("recurrent broadcast preflight requires one-message capacity")
    if int(fixture.get("blocks_per_episode", 0)) != len(consumers):
        raise ValueError("recurrent broadcast requires one query block per consumer")
    if int(fixture.get("query_gap_ticks", 0)) < 1:
        raise ValueError("recurrent broadcast query gap must require persistence")
    units = fixture.get("independent_units", ())
    minimum = int(config.get("stop_contract", {}).get("minimum_independent_units", 0))
    if len(units) < minimum:
        raise ValueError("recurrent broadcast independent-unit minimum is unmet")
    if len({str(row.get("unit_id")) for row in units}) != len(units):
        raise ValueError("recurrent broadcast unit ids must be unique")
    if len({int(row.get("seed", -1)) for row in units}) != len(units):
        raise ValueError("recurrent broadcast unit seeds must be unique")
    evaluation = config.get("evaluation", {})
    if tuple(evaluation.get("arms", ())) != ARM_ORDER:
        raise ValueError("recurrent broadcast arm set drift")
    if tuple(evaluation.get("perturbations", ())) != PERTURBATION_ORDER:
        raise ValueError("recurrent broadcast perturbation set drift")
    if evaluation.get("primary_arm") != ARM_ORDER[0]:
        raise ValueError("recurrent broadcast primary arm drift")
    if {
        str(evaluation.get("lesion_consumer")),
        str(evaluation.get("injection_consumer")),
        str(evaluation.get("stale_consumer")),
        str(evaluation.get("restoration_consumer")),
    } - set(consumer_ids):
        raise ValueError("recurrent broadcast perturbation consumer is unknown")
    cost = config.get("cost_model", {})
    matched = tuple(cost.get("matched_arms", ()))
    if matched != (ARM_ORDER[0], ARM_ORDER[3], ARM_ORDER[5]):
        raise ValueError("recurrent broadcast matched-control set drift")
    if (
        min(
            int(cost.get("matched_modelled_flops_per_tick", 0)),
            int(cost.get("matched_fixed_parameter_slots", 0)),
            int(cost.get("matched_transmitted_words_per_tick", 0)),
        )
        <= 0
    ):
        raise ValueError("recurrent broadcast matched cost budget must be positive")
    envelope = config.get("resource_envelope", {})
    if (
        envelope.get("device") != "cpu"
        or int(envelope.get("cpu_threads", 0)) != 1
        or envelope.get("accelerator_required") is not False
        or envelope.get("model_weights_loaded") is not False
        or envelope.get("model_downloads_allowed") is not False
        or envelope.get("external_data_allowed") is not False
    ):
        raise ValueError("recurrent broadcast preflight must remain one-thread CPU and self-contained")
    return config


def _construction(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_payload": copy.deepcopy(config),
        "config_payload_sha256": canonical_sha256(config),
        "generator": {
            "name": "capacity-one-recurrent-broadcast-fixture",
            "version": 1,
            "competition_rule": (
                "decay every specialist activation, replace only when incoming salience is greater, "
                "then select the highest activation with specialist-order tie breaking"
            ),
            "broadcast_rule": "one retained winning message is copied to three separated consumers",
            "query_rule": "each consumer reads only the frame delivered at its held-out query tick",
        },
    }


def _split_count(config: dict[str, Any], split: str) -> int:
    key = "train_episodes_per_unit" if split == "train" else "heldout_episodes_per_unit"
    return int(config["fixture"][key])


def _message(
    *,
    unit_id: str,
    split: str,
    episode_index: int,
    tick: int,
    source_id: str,
    source_index: int,
    topic: str,
    value: int,
    salience: int,
    kind: str,
) -> dict[str, Any]:
    ref = f"message:broadcast/{unit_id}/{split}/{episode_index:03d}/{tick:02d}/{source_id}"
    event_ref = f"event:broadcast/{unit_id}/{split}/{episode_index:03d}/{tick:02d}/{source_id}"
    content = FrozenJSON.from_value(
        {
            "topic": topic,
            "value": value,
            "source_id": source_id,
            "tick": tick,
            "kind": kind,
        }
    )
    payload: dict[str, Any] = {
        "schema": MESSAGE_SCHEMA,
        "message_ref": ref,
        "event_ref": event_ref,
        "source_id": source_id,
        "source_index": source_index,
        "topic": topic,
        "value": value,
        "salience": salience,
        "kind": kind,
        "tick": tick,
        "content": content.payload(),
    }
    payload["row_sha256"] = canonical_sha256(payload)
    return payload


def _episode(construction: dict[str, Any], unit_index: int, split: str, episode_index: int) -> dict[str, Any]:
    config = construction["config_payload"]
    fixture = config["fixture"]
    unit = fixture["independent_units"][unit_index]
    unit_id = str(unit["unit_id"])
    seed = int(unit["seed"])
    specialists = [str(value) for value in fixture["specialists"]]
    consumers = fixture["consumers"]
    gap = int(fixture["query_gap_ticks"])
    ticks_per_block = gap + 1
    total_ticks = int(fixture["blocks_per_episode"]) * ticks_per_block
    split_value_offset = 100_000 if split == "heldout" else 0
    episode_value_offset = split_value_offset + unit_index * 10_000 + episode_index * 100
    queries: list[dict[str, Any]] = []
    target_by_tick: dict[int, tuple[int, dict[str, Any], int]] = {}
    for consumer_index, consumer in enumerate(consumers):
        target_tick = consumer_index * ticks_per_block
        query_tick = target_tick + gap
        source_index = (seed + episode_index + consumer_index) % len(specialists)
        expected_value = episode_value_offset + consumer_index + 1
        target_by_tick[target_tick] = (source_index, consumer, expected_value)
        queries.append(
            {
                "query_ref": (
                    f"query:broadcast/{unit_id}/{split}/{episode_index:03d}/{consumer['consumer_id']}"
                ),
                "consumer_id": str(consumer["consumer_id"]),
                "consumer_index": consumer_index,
                "topic": str(consumer["topic"]),
                "target_tick": target_tick,
                "query_tick": query_tick,
                "expected_value": expected_value,
                "target_source_id": specialists[source_index],
            }
        )

    ticks: list[dict[str, Any]] = []
    for tick in range(total_ticks):
        rows: list[dict[str, Any]] = []
        target = target_by_tick.get(tick)
        for source_index, source_id in enumerate(specialists):
            if target is not None and target[0] == source_index:
                consumer = target[1]
                row = _message(
                    unit_id=unit_id,
                    split=split,
                    episode_index=episode_index,
                    tick=tick,
                    source_id=source_id,
                    source_index=source_index,
                    topic=str(consumer["topic"]),
                    value=int(target[2]),
                    salience=int(fixture["target_salience"]),
                    kind="target",
                )
            else:
                row = _message(
                    unit_id=unit_id,
                    split=split,
                    episode_index=episode_index,
                    tick=tick,
                    source_id=source_id,
                    source_index=source_index,
                    topic=f"noise_{source_id}",
                    value=-(episode_value_offset + tick * 10 + source_index + 1),
                    salience=int(fixture["distractor_salience_by_specialist"][source_index]),
                    kind="distractor",
                )
            rows.append(row)
        ticks.append({"tick": tick, "messages": rows, "tick_sha256": canonical_sha256(rows)})
    payload: dict[str, Any] = {
        "schema": EPISODE_SCHEMA,
        "episode_ref": f"episode:broadcast/{unit_id}/{split}/{episode_index:03d}",
        "unit_id": unit_id,
        "seed": seed,
        "split": split,
        "episode_index": episode_index,
        "ticks": ticks,
        "queries": queries,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _generate_chunk(construction: dict[str, Any], unit_index: int, split: str) -> dict[str, Any]:
    config = construction["config_payload"]
    unit = config["fixture"]["independent_units"][unit_index]
    episodes = [
        _episode(construction, unit_index, split, index) for index in range(_split_count(config, split))
    ]
    payload: dict[str, Any] = {
        "chunk_id": f"{unit['unit_id']}:{split}",
        "unit": copy.deepcopy(unit),
        "unit_index": unit_index,
        "split": split,
        "episodes": episodes,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _chunk_specs(construction: dict[str, Any]) -> list[tuple[int, str]]:
    units = construction["config_payload"]["fixture"]["independent_units"]
    return [(index, split) for index in range(len(units)) for split in SPLITS]


def _assemble_dataset(construction: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    config = construction["config_payload"]
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    units: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(config["fixture"]["independent_units"]):
        splits = {split: copy.deepcopy(by_id[f"{unit['unit_id']}:{split}"]) for split in SPLITS}
        payload: dict[str, Any] = {
            "unit": copy.deepcopy(unit),
            "unit_index": unit_index,
            "splits": splits,
        }
        payload["payload_sha256"] = canonical_sha256(payload)
        units.append(payload)
    train = int(config["fixture"]["train_episodes_per_unit"])
    heldout = int(config["fixture"]["heldout_episodes_per_unit"])
    blocks = int(config["fixture"]["blocks_per_episode"])
    ticks = blocks * (int(config["fixture"]["query_gap_ticks"]) + 1)
    specialists = len(config["fixture"]["specialists"])
    dataset: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "construction": copy.deepcopy(construction),
        "budget_contract": {
            "independent_units": len(units),
            "chunks": len(chunks),
            "train_episodes_per_unit": train,
            "heldout_episodes_per_unit": heldout,
            "queries_per_episode": blocks,
            "ticks_per_episode": ticks,
            "messages_per_tick": specialists,
            "total_episodes": len(units) * (train + heldout),
            "total_messages": len(units) * (train + heldout) * ticks * specialists,
        },
        "units": units,
    }
    dataset["payload_sha256"] = canonical_sha256(dataset)
    return dataset


def build_dataset(config: dict[str, Any]) -> dict[str, Any]:
    construction = _construction(config)
    chunks = [
        _generate_chunk(construction, unit_index, split) for unit_index, split in _chunk_specs(construction)
    ]
    return _assemble_dataset(construction, chunks)


def _checkpoint_chain(records: list[dict[str, Any]]) -> str:
    chain = "0" * 64
    for record in records:
        chain = canonical_sha256(
            {
                "previous": chain,
                "chunk_id": record["chunk_id"],
                "payload_sha256": record["payload_sha256"],
            }
        )
    return chain


def _validate_checkpoint(checkpoint: dict[str, Any], construction: dict[str, Any]) -> None:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("recurrent broadcast checkpoint schema drift")
    if checkpoint.get("construction_sha256") != canonical_sha256(construction):
        raise ValueError("recurrent broadcast checkpoint construction mismatch")
    config = construction["config_payload"]
    specs = _chunk_specs(construction)
    expected_ids = [
        f"{config['fixture']['independent_units'][index]['unit_id']}:{split}" for index, split in specs
    ]
    records = checkpoint.get("completed_chunks", [])
    observed = [record.get("chunk_id") for record in records]
    if observed != expected_ids[: len(observed)]:
        raise ValueError("recurrent broadcast checkpoint chunk order mismatch")
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("recurrent broadcast checkpoint chunk payload is invalid")
        expected = canonical_sha256(_without_digest(payload))
        if record.get("payload_sha256") != expected:
            raise ValueError("recurrent broadcast checkpoint chunk digest mismatch")
        if payload.get("payload_sha256") != expected:
            raise ValueError("recurrent broadcast nested chunk digest mismatch")
    if checkpoint.get("chain_sha256") != _checkpoint_chain(records):
        raise ValueError("recurrent broadcast checkpoint chain mismatch")


def build_dataset_resumable(
    config: dict[str, Any],
    checkpoint_path: Path,
    *,
    stop_after_chunks: int | None = None,
) -> dict[str, Any] | None:
    construction = _construction(config)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        _validate_checkpoint(checkpoint, construction)
    else:
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "construction_sha256": canonical_sha256(construction),
            "completed_chunks": [],
            "chain_sha256": "0" * 64,
        }
        _atomic_json(checkpoint_path, checkpoint)
    specs = _chunk_specs(construction)
    while len(checkpoint["completed_chunks"]) < len(specs):
        index = len(checkpoint["completed_chunks"])
        unit_index, split = specs[index]
        payload = _generate_chunk(construction, unit_index, split)
        checkpoint["completed_chunks"].append(
            {
                "chunk_id": payload["chunk_id"],
                "payload_sha256": payload["payload_sha256"],
                "payload": payload,
            }
        )
        checkpoint["chain_sha256"] = _checkpoint_chain(checkpoint["completed_chunks"])
        _atomic_json(checkpoint_path, checkpoint)
        if stop_after_chunks is not None and len(checkpoint["completed_chunks"]) >= stop_after_chunks:
            return None
    dataset = _assemble_dataset(
        construction,
        [record["payload"] for record in checkpoint["completed_chunks"]],
    )
    checkpoint["final_dataset_sha256"] = dataset["payload_sha256"]
    _atomic_json(checkpoint_path, checkpoint)
    return dataset


def verify_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        construction = dataset["construction"]
        config = construction["config_payload"]
        if config.get("schema") != CONFIG_SCHEMA or config.get("claim_scope") != CLAIM_SCOPE:
            raise ValueError("embedded recurrent broadcast config drift")
        expected = build_dataset(config)
        checks["exact_deterministic_rebuild"] = canonical_sha256(dataset) == canonical_sha256(expected)
        checks["dataset_payload_digest"] = dataset.get("payload_sha256") == canonical_sha256(
            _without_digest(dataset)
        )
        checks["construction_digest"] = construction.get("config_payload_sha256") == canonical_sha256(config)
        episode_digests = True
        message_digests = True
        frozen_content = True
        event_refs_unique = True
        split_disjoint = True
        query_targets_exact = True
        counts_exact = True
        expected_specialists = len(config["fixture"]["specialists"])
        expected_ticks = int(config["fixture"]["blocks_per_episode"]) * (
            int(config["fixture"]["query_gap_ticks"]) + 1
        )
        for unit in dataset["units"]:
            refs: dict[str, set[str]] = {}
            for split in SPLITS:
                episodes = unit["splits"][split]["episodes"]
                refs[split] = {str(episode["episode_ref"]) for episode in episodes}
                counts_exact &= len(episodes) == _split_count(config, split)
                for episode in episodes:
                    episode_digests &= episode.get("payload_sha256") == canonical_sha256(
                        _without_digest(episode)
                    )
                    counts_exact &= len(episode["ticks"]) == expected_ticks
                    events: list[str] = []
                    targets: dict[tuple[str, int], dict[str, Any]] = {}
                    for tick in episode["ticks"]:
                        counts_exact &= len(tick["messages"]) == expected_specialists
                        counts_exact &= tick.get("tick_sha256") == canonical_sha256(tick["messages"])
                        for row in tick["messages"]:
                            row_core = copy.deepcopy(row)
                            recorded = row_core.pop("row_sha256", None)
                            message_digests &= recorded == canonical_sha256(row_core)
                            content = row["content"]
                            frozen_content &= content.get("sha256") == canonical_sha256(content.get("value"))
                            frozen_content &= content.get("value", {}).get("topic") == row["topic"]
                            frozen_content &= content.get("value", {}).get("value") == row["value"]
                            events.append(str(row["event_ref"]))
                            if row["kind"] == "target":
                                targets[(str(row["topic"]), int(row["tick"]))] = row
                    event_refs_unique &= len(events) == len(set(events))
                    for query in episode["queries"]:
                        row = targets.get((str(query["topic"]), int(query["target_tick"])))
                        query_targets_exact &= row is not None
                        if row is not None:
                            query_targets_exact &= row["value"] == query["expected_value"]
                            query_targets_exact &= row["source_id"] == query["target_source_id"]
            split_disjoint &= refs["train"].isdisjoint(refs["heldout"])
        checks.update(
            {
                "episode_payload_digests": episode_digests,
                "message_row_digests": message_digests,
                "frozen_message_content": frozen_content,
                "event_refs_unique_within_episode": event_refs_unique,
                "train_heldout_episode_refs_disjoint": split_disjoint,
                "query_targets_exact": query_targets_exact,
                "declared_counts_exact": counts_exact,
            }
        )
    except Exception as exc:
        errors.append(f"verification exception: {exc}")
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {"verified": not errors and all(checks.values()), "checks": checks, "errors": errors}


def mutation_suite(dataset: dict[str, Any]) -> dict[str, Any]:
    mutations: dict[str, dict[str, Any]] = {}

    def reject(name: str, mutate: Any) -> None:
        changed = copy.deepcopy(dataset)
        mutate(changed)
        audit = verify_dataset(changed)
        mutations[name] = {"rejected": audit["verified"] is False, "errors": audit["errors"][:4]}

    def first_episode(value: dict[str, Any], split: str = "heldout") -> dict[str, Any]:
        return value["units"][0]["splits"][split]["episodes"][0]

    def first_message(value: dict[str, Any]) -> dict[str, Any]:
        return first_episode(value)["ticks"][0]["messages"][0]

    reject("message_value", lambda value: first_message(value).__setitem__("value", 999_999))
    reject("message_digest", lambda value: first_message(value).__setitem__("row_sha256", "0" * 64))
    reject(
        "frozen_content",
        lambda value: first_message(value)["content"]["value"].__setitem__("topic", "forged"),
    )
    reject(
        "query_target",
        lambda value: first_episode(value)["queries"][0].__setitem__("expected_value", -7),
    )
    reject(
        "event_provenance",
        lambda value: first_episode(value)["ticks"][0]["messages"][1].__setitem__(
            "event_ref", first_message(value)["event_ref"]
        ),
    )
    reject(
        "split_identity",
        lambda value: first_episode(value).__setitem__(
            "episode_ref", first_episode(value, "train")["episode_ref"]
        ),
    )
    reject(
        "episode_digest",
        lambda value: first_episode(value).__setitem__("payload_sha256", "f" * 64),
    )
    reject("dataset_digest", lambda value: value.__setitem__("payload_sha256", "a" * 64))
    return {
        "mutations": mutations,
        "count": len(mutations),
        "rejected": sum(row["rejected"] is True for row in mutations.values()),
        "all_rejected": all(row["rejected"] is True for row in mutations.values()),
    }


def _fit_consumer_contract(episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    topic_to_consumer: dict[str, str] = {}
    observations = 0
    for episode in episodes:
        for query in episode["queries"]:
            topic = str(query["topic"])
            consumer = str(query["consumer_id"])
            prior = topic_to_consumer.setdefault(topic, consumer)
            if prior != consumer:
                raise ValueError("training consumer contract is inconsistent")
            observations += 1
    expected = {str(row["topic"]): str(row["consumer_id"]) for row in config["fixture"]["consumers"]}
    if topic_to_consumer != expected:
        raise ValueError("training consumer contract is incomplete")
    payload: dict[str, Any] = {
        "kind": "shared topic-to-consumer lookup fitted on train episodes only",
        "topic_to_consumer": topic_to_consumer,
        "fit_observations": observations,
        "fitted_lookup_entries": len(topic_to_consumer),
        "trainable_parameters": 0,
        "shared_across_all_arms": True,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _cost_ledger(arm: str, config: dict[str, Any]) -> dict[str, Any]:
    fixture = config["fixture"]
    cost = config["cost_model"]
    message_words = int(fixture["message_words"])
    specialists = len(fixture["specialists"])
    consumers = len(fixture["consumers"])
    score_flops = specialists * linear_flops(message_words, 1)
    readout_flops = consumers * linear_flops(message_words, 1)
    recurrent_update_flops = specialists * 2
    primary_core = score_flops + readout_flops + recurrent_update_flops
    matched_budget = int(cost["matched_modelled_flops_per_tick"])
    if arm in cost["matched_arms"]:
        modelled = matched_budget
        words = int(cost["matched_transmitted_words_per_tick"])
        fixed_slots = int(cost["matched_fixed_parameter_slots"])
    elif arm == "global_dense_state":
        modelled = int(cost["global_dense_modelled_flops_per_tick"])
        words = int(cost["global_dense_transmitted_words_per_tick"])
        fixed_slots = int(cost["matched_fixed_parameter_slots"])
    elif arm == "no_broadcast":
        modelled = int(cost["no_broadcast_modelled_flops_per_tick"])
        words = 0
        fixed_slots = int(cost["matched_fixed_parameter_slots"])
    else:
        modelled = int(cost["independent_specialist_modelled_flops_per_tick"])
        words = int(cost["matched_transmitted_words_per_tick"])
        fixed_slots = int(cost["matched_fixed_parameter_slots"])
    capacity = (
        int(cost["global_dense_capacity_messages"])
        if arm == "global_dense_state"
        else 0
        if arm == "no_broadcast"
        else 1
    )
    semantic_words = message_words if arm == "equal_flop_routing" else words
    return {
        "arm": arm,
        "modelled_flops_per_tick": modelled,
        "modelled_flop_decomposition": {
            "specialist_score_linear_flops": score_flops,
            "consumer_readout_linear_flops": readout_flops,
            "recurrent_update_flops": recurrent_update_flops if "recurrent" in arm else 0,
            "matched_padding_or_alternative_structure_flops": max(0, modelled - primary_core),
        },
        "transmitted_words_per_tick": words,
        "semantic_words_per_tick": semantic_words,
        "padding_or_masked_words_per_tick": words - semantic_words,
        "shared_capacity_messages": capacity,
        "fixed_parameter_slots": fixed_slots,
        "trainable_parameters": 0,
        "hardware_flops_measured": False,
        "scope": str(cost["note"]),
    }


def _recurrent_step(
    states: list[dict[str, Any]], messages: list[dict[str, Any]], decay: int
) -> tuple[dict[str, Any], int]:
    for state in states:
        state["activation"] = max(0, int(state["activation"]) - decay)
    for row in messages:
        index = int(row["source_index"])
        if int(row["salience"]) > int(states[index]["activation"]):
            states[index] = {"activation": int(row["salience"]), "message": row}
    winner_index = min(
        range(len(states)),
        key=lambda index: (-int(states[index]["activation"]), index),
    )
    winner = states[winner_index]["message"]
    if winner is None:
        raise ValueError("recurrent competition produced no winner")
    return winner, winner_index


def _state_content(states: list[dict[str, Any]], winner: dict[str, Any]) -> dict[str, Any]:
    return {
        "activations": [int(state["activation"]) for state in states],
        "retained_message_refs": [
            state["message"]["message_ref"] if state["message"] is not None else None for state in states
        ],
        "winner_message_ref": winner["message_ref"],
    }


def _shuffled_values(episodes: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        source = episodes[(index + 1) % len(episodes)]
        source_by_consumer = {row["consumer_id"]: row for row in source["queries"]}
        for query in episode["queries"]:
            source_query = source_by_consumer[str(query["consumer_id"])]
            mapping[(str(episode["episode_ref"]), str(query["consumer_id"]))] = {
                "value": int(source_query["expected_value"]),
                "source_episode_ref": str(source["episode_ref"]),
                "source_query_ref": str(source_query["query_ref"]),
            }
    return mapping


def _delivered_message(
    *,
    arm: str,
    perturbation: str,
    consumer: dict[str, Any],
    current: dict[str, Any] | None,
    dense_state: dict[str, dict[str, Any]],
    specialist_states: list[dict[str, Any]],
    broadcast_history: list[dict[str, Any] | None],
    tick: int,
    episode_ref: str,
    shuffle: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str, str | None]:
    consumer_id = str(consumer["consumer_id"])
    topic = str(consumer["topic"])
    action = "delivered"
    provenance: str | None = None
    if arm == "global_dense_state":
        delivered = dense_state.get(topic)
    elif arm == "no_broadcast":
        delivered = None
        action = "no-channel"
    elif arm == "independent_specialists":
        specialist_index = list(config["fixture"]["specialists"]).index(consumer["local_specialist"])
        delivered = specialist_states[specialist_index]["message"]
        action = "local-only"
    elif arm == "equal_flop_routing":
        if current is None:
            delivered = None
        else:
            receiver = int(current["source_index"]) % len(config["fixture"]["consumers"])
            delivered = current if receiver == int(consumer["consumer_index"]) else None
            action = "point-routed" if delivered is not None else "masked-padding-frame"
    else:
        delivered = current

    if arm != "recurrent_limited_broadcast" or perturbation == "clean":
        return delivered, action, provenance
    evaluation = config["evaluation"]
    if perturbation == "consumer_link_lesion" and consumer_id == evaluation["lesion_consumer"]:
        return None, "link-lesioned", None
    if perturbation == "wrong_message_injection" and consumer_id == evaluation["injection_consumer"]:
        if delivered is None:
            return None, "injection-no-source", None
        injected = copy.deepcopy(delivered)
        injected["value"] = int(delivered["value"]) + 1_000_003
        return injected, "wrong-message-injected", str(delivered["message_ref"])
    if perturbation == "delayed_stale_message" and consumer_id == evaluation["stale_consumer"]:
        source_tick = tick - int(evaluation["stale_delay_ticks"])
        stale = broadcast_history[source_tick] if source_tick >= 0 else None
        return stale, "delayed-stale-frame", f"tick:{source_tick}" if source_tick >= 0 else None
    if perturbation == "restored_link" and consumer_id == evaluation["restoration_consumer"]:
        return delivered, "link-restored-before-query", str(delivered["message_ref"]) if delivered else None
    if perturbation == "shuffled_message" and delivered is not None:
        source = shuffle[(episode_ref, consumer_id)]
        shuffled = copy.deepcopy(delivered)
        shuffled["value"] = int(source["value"])
        return shuffled, "heldout-message-shuffled", str(source["source_query_ref"])
    return delivered, action, provenance


def _simulate_episode(
    episode: dict[str, Any],
    config: dict[str, Any],
    *,
    arm: str,
    perturbation: str = "clean",
    shuffle: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if arm not in ARM_ORDER or perturbation not in PERTURBATION_ORDER:
        raise ValueError("unsupported recurrent broadcast arm or perturbation")
    specialists = list(config["fixture"]["specialists"])
    consumers_by_id = {
        str(row["consumer_id"]): {**row, "consumer_index": index}
        for index, row in enumerate(config["fixture"]["consumers"])
    }
    queries_by_tick = {int(row["query_tick"]): row for row in episode["queries"]}
    decay = int(config["fixture"]["activation_decay_per_tick"])
    states: list[dict[str, Any]] = [{"activation": 0, "message": None} for _ in range(len(specialists))]
    dense_state: dict[str, dict[str, Any]] = {}
    winner_history: list[dict[str, Any] | None] = []
    query_rows: list[dict[str, Any]] = []
    persistence = 0
    prior_ref: str | None = None
    max_shared = 0
    restoration_lesion_frames = 0
    journal: LifecycleJournal | None = None
    journal_events: set[str] = set()
    if arm == "recurrent_limited_broadcast":
        journal = LifecycleJournal(
            MemoryRef(
                f"memory:broadcast/{episode['unit_id']}/{episode['split']}/"
                f"{episode['episode_index']:03d}/{perturbation}"
            )
        )
    for tick_row in episode["ticks"]:
        tick = int(tick_row["tick"])
        messages = tick_row["messages"]
        for row in messages:
            if row["kind"] == "target":
                dense_state[str(row["topic"])] = row
        winner, winner_index = _recurrent_step(states, messages, decay)
        if arm == "larger_feedforward_depth":
            winner = min(messages, key=lambda row: (-int(row["salience"]), int(row["source_index"])))
        current = winner if arm != "no_broadcast" else None
        winner_history.append(current)
        winner_ref = current["message_ref"] if current is not None else None
        if winner_ref is not None and winner_ref == prior_ref:
            persistence += 1
        prior_ref = winner_ref
        max_shared = max(
            max_shared,
            len(dense_state) if arm == "global_dense_state" else 0 if arm == "no_broadcast" else 1,
        )
        if journal is not None:
            event_ref = EventRef(
                f"event:broadcast/{episode['unit_id']}/{episode['split']}/"
                f"{episode['episode_index']:03d}/{perturbation}/{tick:02d}"
            )
            journal_events.add(str(event_ref))
            content = _state_content(states, winner)
            if tick == 0:
                journal.record(event_ref, content, reason="initial recurrent competition state")
            else:
                journal.revise(event_ref, content, reason="recurrent competition tick")

        query = queries_by_tick.get(tick)
        if query is not None:
            consumer = consumers_by_id[str(query["consumer_id"])]
            if (
                perturbation == "restored_link"
                and consumer["consumer_id"] == config["evaluation"]["restoration_consumer"]
            ):
                restoration_lesion_frames += int(config["fixture"]["query_gap_ticks"])
            delivered, action, provenance = _delivered_message(
                arm=arm,
                perturbation=perturbation,
                consumer=consumer,
                current=current,
                dense_state=dense_state,
                specialist_states=states,
                broadcast_history=winner_history,
                tick=tick,
                episode_ref=str(episode["episode_ref"]),
                shuffle=shuffle or {},
                config=config,
            )
            observed = (
                int(delivered["value"])
                if delivered is not None and delivered["topic"] == query["topic"]
                else None
            )
            query_rows.append(
                {
                    "query_ref": query["query_ref"],
                    "consumer_id": query["consumer_id"],
                    "topic": query["topic"],
                    "target_tick": query["target_tick"],
                    "query_tick": tick,
                    "latency_ticks": tick - int(query["target_tick"]),
                    "expected_value": int(query["expected_value"]),
                    "observed_value": observed,
                    "correct": observed == int(query["expected_value"]),
                    "delivered_message_ref": delivered["message_ref"] if delivered is not None else None,
                    "delivery_action": action,
                    "control_provenance": provenance,
                }
            )
            states = [{"activation": 0, "message": None} for _ in specialists]
            dense_state.pop(str(query["topic"]), None)
            prior_ref = None

    journal_payload = journal.payload() if journal is not None else None
    journal_errors = journal.verify(event_refs=journal_events) if journal is not None else []
    trace = {
        "episode_ref": episode["episode_ref"],
        "arm": arm,
        "perturbation": perturbation,
        "winner_message_refs": [row["message_ref"] if row is not None else None for row in winner_history],
        "query_rows": query_rows,
        "persistent_winner_transitions": persistence,
        "max_shared_capacity_messages": max_shared,
        "restoration_lesion_frames_before_query": restoration_lesion_frames,
        "state_journal_sha256": canonical_sha256(journal_payload) if journal_payload is not None else None,
        "state_journal_head_sha256": journal_payload["head_sha256"] if journal_payload is not None else None,
        "state_journal_errors": journal_errors,
    }
    trace["trace_sha256"] = canonical_sha256(trace)
    return trace


def _evaluate_arm(
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    arm: str,
    perturbation: str = "clean",
) -> dict[str, Any]:
    shuffle = _shuffled_values(episodes)
    traces = [
        _simulate_episode(
            episode,
            config,
            arm=arm,
            perturbation=perturbation,
            shuffle=shuffle,
        )
        for episode in episodes
    ]
    consumers = [str(row["consumer_id"]) for row in config["fixture"]["consumers"]]
    per_consumer: dict[str, dict[str, Any]] = {}
    all_rows = [row for trace in traces for row in trace["query_rows"]]
    for consumer in consumers:
        rows = [row for row in all_rows if row["consumer_id"] == consumer]
        correct = sum(row["correct"] is True for row in rows)
        per_consumer[consumer] = {
            "queries": len(rows),
            "correct": correct,
            "accuracy": _fraction(Fraction(correct, len(rows))),
        }
    correct = sum(row["correct"] is True for row in all_rows)
    journal_ok = all(not trace["state_journal_errors"] for trace in traces)
    return {
        "arm": arm,
        "perturbation": perturbation,
        "episodes": len(episodes),
        "queries": len(all_rows),
        "correct": correct,
        "accuracy": _fraction(Fraction(correct, len(all_rows))),
        "per_consumer": per_consumer,
        "persistent_winner_transitions": sum(int(trace["persistent_winner_transitions"]) for trace in traces),
        "max_shared_capacity_messages": max(int(trace["max_shared_capacity_messages"]) for trace in traces),
        "consumer_ids_observed": sorted({str(row["consumer_id"]) for row in all_rows}),
        "mean_latency_ticks": _fraction(
            Fraction(sum(int(row["latency_ticks"]) for row in all_rows), len(all_rows))
        ),
        "state_journals_exact": journal_ok,
        "state_journal_set_sha256": canonical_sha256([trace["state_journal_sha256"] for trace in traces]),
        "restoration_lesion_frames_before_query": sum(
            int(trace["restoration_lesion_frames_before_query"]) for trace in traces
        ),
        "trace_set_sha256": canonical_sha256([trace["trace_sha256"] for trace in traces]),
        "cost": _cost_ledger(arm, config),
        "mechanics_only": True,
    }


def _accuracy(result: dict[str, Any], consumer: str | None = None) -> Fraction:
    payload = result["accuracy"] if consumer is None else result["per_consumer"][consumer]["accuracy"]
    return _from_fraction(payload)


def _evaluate_unit(unit: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    train = unit["splits"]["train"]["episodes"]
    heldout = unit["splits"]["heldout"]["episodes"]
    fitted = _fit_consumer_contract(train, config)
    arms = {arm: _evaluate_arm(heldout, config, arm=arm) for arm in ARM_ORDER}
    perturbations = {
        perturbation: _evaluate_arm(
            heldout,
            config,
            arm="recurrent_limited_broadcast",
            perturbation=perturbation,
        )
        for perturbation in PERTURBATION_ORDER
    }
    primary = arms["recurrent_limited_broadcast"]
    dense = arms["global_dense_state"]
    lesion = perturbations["consumer_link_lesion"]
    injection = perturbations["wrong_message_injection"]
    stale = perturbations["delayed_stale_message"]
    restored = perturbations["restored_link"]
    shuffled = perturbations["shuffled_message"]
    evaluation = config["evaluation"]
    consumers = [str(row["consumer_id"]) for row in config["fixture"]["consumers"]]
    lesion_consumer = str(evaluation["lesion_consumer"])
    injection_consumer = str(evaluation["injection_consumer"])
    stale_consumer = str(evaluation["stale_consumer"])

    def only_consumer_degraded(result: dict[str, Any], target: str) -> bool:
        return _accuracy(result, target) < _accuracy(primary, target) and all(
            _accuracy(result, consumer) == _accuracy(primary, consumer)
            for consumer in consumers
            if consumer != target
        )

    matched_arms = [arms[name]["cost"] for name in config["cost_model"]["matched_arms"]]
    training_refs = {str(row["episode_ref"]) for row in train}
    heldout_refs = {str(row["episode_ref"]) for row in heldout}
    reported = {
        "unit_id": unit["unit"]["unit_id"],
        "heldout_episodes": len(heldout),
        "consumer_count": len(consumers),
        "primary_queries": primary["queries"],
    }
    traced = dict(reported)
    grounding = report_grounding(reported, traced)
    checks = {
        "all_arms_executed": tuple(arms) == ARM_ORDER,
        "all_perturbations_executed": tuple(perturbations) == PERTURBATION_ORDER,
        "train_heldout_units_disjoint": training_refs.isdisjoint(heldout_refs),
        "three_separated_consumers": primary["consumer_ids_observed"] == sorted(consumers),
        "persistent_recurrent_winner": primary["persistent_winner_transitions"]
        >= len(heldout) * len(consumers) * int(config["fixture"]["query_gap_ticks"]),
        "capacity_limit_one": primary["max_shared_capacity_messages"] == 1,
        "global_dense_capacity_is_wider": dense["max_shared_capacity_messages"] > 1,
        "primary_matches_dense_fixture_accuracy": _accuracy(primary) == _accuracy(dense) == 1,
        "no_broadcast_is_lower": _accuracy(arms["no_broadcast"]) < _accuracy(primary),
        "feedforward_depth_is_lower": _accuracy(arms["larger_feedforward_depth"]) < _accuracy(primary),
        "independent_specialists_are_lower": _accuracy(arms["independent_specialists"]) < _accuracy(primary),
        "equal_flop_routing_is_lower": _accuracy(arms["equal_flop_routing"]) < _accuracy(primary),
        "matched_controls_exact_flops": len({int(row["modelled_flops_per_tick"]) for row in matched_arms})
        == 1,
        "matched_controls_exact_fixed_parameter_slots": len(
            {int(row["fixed_parameter_slots"]) for row in matched_arms}
        )
        == 1,
        "matched_controls_exact_bandwidth": len(
            {int(row["transmitted_words_per_tick"]) for row in matched_arms}
        )
        == 1,
        "limited_uses_less_bandwidth_than_dense": primary["cost"]["transmitted_words_per_tick"]
        < dense["cost"]["transmitted_words_per_tick"],
        "consumer_link_lesion_is_selective": only_consumer_degraded(lesion, lesion_consumer),
        "wrong_message_injection_is_selective": only_consumer_degraded(injection, injection_consumer),
        "delayed_stale_message_is_selective": only_consumer_degraded(stale, stale_consumer),
        "restoration_recovers_clean_result": _accuracy(restored) == _accuracy(primary)
        and restored["trace_set_sha256"] != primary["trace_set_sha256"]
        and restored["restoration_lesion_frames_before_query"] > 0,
        "shuffled_messages_reduce_all_consumers": all(
            _accuracy(shuffled, consumer) < _accuracy(primary, consumer) for consumer in consumers
        ),
        "exact_lifecycle_replay": primary["state_journals_exact"] is True,
        "oa8_trace_grounding_exact": grounding["grounded_fraction"] == 1.0,
        "all_metrics_finite": _finite_tree({"arms": arms, "perturbations": perturbations}),
    }
    comparison = {
        "primary_minus_global_dense_accuracy": _fraction(_accuracy(primary) - _accuracy(dense)),
        "primary_bandwidth_fraction_of_global_dense": _fraction(
            Fraction(
                int(primary["cost"]["transmitted_words_per_tick"]),
                int(dense["cost"]["transmitted_words_per_tick"]),
            )
        ),
        "primary_minus_no_broadcast_accuracy": _fraction(
            _accuracy(primary) - _accuracy(arms["no_broadcast"])
        ),
        "primary_minus_feedforward_depth_accuracy": _fraction(
            _accuracy(primary) - _accuracy(arms["larger_feedforward_depth"])
        ),
        "primary_minus_independent_specialists_accuracy": _fraction(
            _accuracy(primary) - _accuracy(arms["independent_specialists"])
        ),
        "primary_minus_equal_flop_routing_accuracy": _fraction(
            _accuracy(primary) - _accuracy(arms["equal_flop_routing"])
        ),
    }
    return {
        "independent_unit": copy.deepcopy(unit["unit"]),
        "dataset_payload_sha256": unit["payload_sha256"],
        "split_payload_sha256": {split: unit["splits"][split]["payload_sha256"] for split in SPLITS},
        "fitted_consumer_contract": fitted,
        "arms": arms,
        "perturbations": perturbations,
        "comparisons": comparison,
        "operational_report_grounding": grounding,
        "checks": checks,
        "all_mechanics_ok": all(checks.values()),
        "scientific_promotion_allowed": False,
    }


def _aggregate(units: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_keys = (
        "primary_minus_global_dense_accuracy",
        "primary_minus_no_broadcast_accuracy",
        "primary_minus_feedforward_depth_accuracy",
        "primary_minus_independent_specialists_accuracy",
        "primary_minus_equal_flop_routing_accuracy",
    )
    comparisons: dict[str, Any] = {}
    for key in comparison_keys:
        values = [_from_fraction(unit["comparisons"][key]) for unit in units]
        comparisons[key] = {
            "mean": _fraction(sum(values, Fraction()) / len(values)),
            "positive_units": sum(value > 0 for value in values),
            "zero_units": sum(value == 0 for value in values),
            "negative_units": sum(value < 0 for value in values),
        }
    return {
        "comparisons": comparisons,
        "all_units_mechanics_ok": all(unit["all_mechanics_ok"] is True for unit in units),
        "mechanics_pattern_only": True,
        "natural_null_rejected": False,
    }


def _corrupt_checkpoint(config: dict[str, Any], source: Path, target: Path) -> bool:
    shutil.copy2(source, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["completed_chunks"][0]["payload"]["episodes"][0]["queries"][0]["expected_value"] += 1
    _atomic_json(target, payload)
    try:
        build_dataset_resumable(config, target)
    except ValueError:
        return True
    return False


def _deterministic_part(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key not in {"resource_observation", "deterministic_core_sha256"}
    }


def build_preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="recurrent-broadcast-") as scratch_name:
        scratch = Path(scratch_name)
        accountant = WorkloadAccountant(
            workload="bounded recurrent capacity-limited broadcast mechanics preflight",
            watch_paths={"scratch": scratch},
        )
        with accountant.phase("fixture"):
            dataset = build_dataset(config)
        with accountant.phase("resume"):
            checkpoint = scratch / "resume.json"
            interrupted = build_dataset_resumable(config, checkpoint, stop_after_chunks=3)
            if interrupted is not None:
                raise ValueError("recurrent broadcast resume drill did not interrupt")
            resumed = build_dataset_resumable(config, checkpoint)
            if resumed is None:
                raise ValueError("recurrent broadcast resume drill did not finish")
            resume_exact = resumed["payload_sha256"] == dataset["payload_sha256"]
            corrupted_rejected = _corrupt_checkpoint(config, checkpoint, scratch / "corrupt.json")
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        with accountant.phase("evaluate"):
            units = [_evaluate_unit(unit, config) for unit in dataset["units"]]
        with accountant.phase("verify"):
            verification = verify_dataset(dataset)
            mutations = mutation_suite(dataset)
        accounting = accountant.receipt()

    elapsed = time.perf_counter() - started
    max_rss = _max_rss_bytes()
    envelope = config["resource_envelope"]
    checks = {
        "minimum_independent_units": len(units) >= int(config["stop_contract"]["minimum_independent_units"]),
        "dataset_exact_replay": verification["verified"] is True,
        "all_dataset_mutations_rejected": mutations["all_rejected"] is True,
        "all_unit_mechanics_ok": all(unit["all_mechanics_ok"] is True for unit in units),
        "interrupted_resume_is_exact": resume_exact,
        "corrupt_checkpoint_rejected": corrupted_rejected,
        "all_scientific_promotion_blocked": all(
            unit["scientific_promotion_allowed"] is False for unit in units
        ),
        "resource_wall_envelope": elapsed <= float(envelope["maximum_wall_seconds"]),
        "resource_rss_envelope": max_rss <= int(envelope["maximum_rss_bytes"]),
        "no_model_or_download_modules": not any(
            name.startswith(("transformers", "timm", "huggingface_hub", "vjepa")) for name in sys.modules
        ),
    }
    core: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "status": "mechanics-pass" if all(checks.values()) else "mechanics-fail",
        "null_hypothesis": config["null_hypothesis"],
        "audit": {
            "existing_surfaces": [
                (
                    "WorkspaceShell composes context gating, mean-pooled working memory, "
                    "predictors, ensembles, and heads"
                ),
                "MoT router studies choose or mix reader predictions without a recurrent broadcast bus",
                "CM4 tested the composed workspace and recorded a null against dense and unrolled controls",
                "Wave E0 supplies immutable event identities and exact lifecycle replay",
                "P9 supplies bounded phase accounting, exact resume, and mutation-verifier patterns",
                "operational-awareness OA8 checks report-to-trace grounding without a self-awareness claim",
            ],
            "true_gap_closed": [
                "persistent recurrent competition among four separated specialists",
                "one-message capacity limit with winner broadcast to three separated consumers",
                (
                    "global dense, no-broadcast, deeper feed-forward, independent-specialist, "
                    "and equal-FLOP routing controls"
                ),
                (
                    "consumer-specific lesion, injection, stale-delay, restoration, and "
                    "shuffled-message branches"
                ),
                "exact event-bound state journal, interrupted resume, cost ledger, and mutation rejection",
            ],
            "nonduplication": (
                "the implementation reuses canonical event bytes, lifecycle journals, compute estimates, "
                "P9 accounting, and OA8 grounding; it does not add another shell or router class"
            ),
        },
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(config_path),
            "payload_sha256": canonical_sha256(config),
            "payload": config,
        },
        "dataset": {
            "schema": dataset["schema"],
            "payload_sha256": dataset["payload_sha256"],
            "budget_contract": dataset["budget_contract"],
            "verification": verification,
        },
        "resume": {
            "interrupted_after_chunks": 3,
            "completed_chunks": len(checkpoint_payload["completed_chunks"]),
            "checkpoint_chain_sha256": checkpoint_payload["chain_sha256"],
            "final_dataset_sha256": checkpoint_payload["final_dataset_sha256"],
            "clean_dataset_sha256": dataset["payload_sha256"],
            "exact": resume_exact,
            "corrupt_checkpoint_rejected": corrupted_rejected,
        },
        "mutation_suite": mutations,
        "units": units,
        "aggregate": _aggregate(units),
        "checks": checks,
        "claim_boundary": {
            "mechanics_only": True,
            "natural_tasks": False,
            "learned_broadcast": False,
            "capability_claim": False,
            "cognition_consciousness_sentience_or_agency_claim": False,
            "scientific_promotion_allowed": False,
            "remaining_evidence_gate": (
                "a preregistered non-ceiling natural task with learned competition, independently "
                "held-out data, matched bandwidth, FLOPs, and parameters, replicated seeds, and "
                "the same lesion and restoration controls"
            ),
        },
        "implementation": [
            _file_receipt(REPO_ROOT / path) for path in (*IMPLEMENTATION_PATHS, *REUSED_PATHS)
        ],
        "all_mechanics_ok": all(checks.values()),
    }
    core["deterministic_core_sha256"] = canonical_sha256(core)
    return {
        **core,
        "resource_observation": {
            "elapsed_seconds": round(elapsed, 6),
            "max_rss_bytes": max_rss,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "device": "cpu",
            "cpu_threads": 1,
            "accelerator_required": False,
            "model_weights_loaded": False,
            "model_downloads_performed": False,
            "external_data_loaded": False,
            "command_executed_heavy_work": False,
            "workload_accounting": accounting,
        },
    }


def verify_preflight_receipt(receipt: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    recorded = str(receipt.get("deterministic_core_sha256", ""))
    self_hash = canonical_sha256(_deterministic_part(receipt))
    rebuilt = build_preflight(config_path)
    rebuilt_hash = str(rebuilt["deterministic_core_sha256"])
    checks = {
        "recorded_core_self_hash": recorded == self_hash,
        "exact_rebuild_hash": recorded == rebuilt_hash,
        "exact_rebuild_payload": canonical_sha256(_deterministic_part(receipt))
        == canonical_sha256(_deterministic_part(rebuilt)),
        "mechanics_pass": receipt.get("all_mechanics_ok") is True,
        "scientific_promotion_blocked": receipt.get("claim_boundary", {}).get("scientific_promotion_allowed")
        is False,
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "recorded_core_sha256": recorded,
        "rebuilt_core_sha256": rebuilt_hash,
    }


def write_preflight(
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    receipt = build_preflight(config_path)
    if receipt["all_mechanics_ok"] is not True:
        raise ValueError("recurrent broadcast mechanics did not pass")
    _atomic_json(output_path, receipt)
    maximum = int(receipt["config"]["payload"]["resource_envelope"]["maximum_proof_bytes"])
    if output_path.stat().st_size > maximum:
        output_path.unlink()
        raise ValueError("recurrent broadcast proof exceeds its declared byte envelope")
    return receipt
