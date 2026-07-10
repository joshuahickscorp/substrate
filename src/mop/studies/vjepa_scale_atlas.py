"""Local, referent-exact atlas over serially cached V-JEPA scale points.

This is deliberately a migration instrument, not a claim that three pretrained encoders are a custom
substrate.  It establishes which geometry and factor signals survive scale, then exposes those signals as
requirements for locally trained candidates.  Programmatic clips and missing random-architecture controls
keep the result non-promotable even when a permutation test is small.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..diagnostics.alignment import alignment_table
from ..substrate.cache_manifest import validate_cache_manifest
from ..substrate.latent_store import LatentStore
from ..substrate.real_latent import factorized_arrays

SCHEMA = "mop-vjepa-scale-atlas-local/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_cache(path: Path) -> dict[str, Any]:
    problems = validate_cache_manifest(path, citable=True)
    if problems:
        raise ValueError(f"uncitable cache {path}: {'; '.join(problems)}")
    store = LatentStore.open(path)
    x, factor_a, factor_b = factorized_arrays(store)
    referents = [str(value) for value in _json(path / "referents.json")]
    splits = _json(path / "splits.json")
    manifest_path = path / "cache_manifest.json"
    manifest = _json(manifest_path)
    objective = (manifest.get("form") or {}).get("objective")
    identity_name = "initialization_receipt.json" if objective == "random-control" else "encoder_receipt.json"
    identity = _json(path / identity_name)
    run_receipt = _json(path / "run_receipt.json") if (path / "run_receipt.json").exists() else {}
    return {
        "path": path,
        "latents": torch.as_tensor(x).float(),
        "factor_a": torch.as_tensor(factor_a).long(),
        "factor_b": torch.as_tensor(factor_b).long(),
        "referents": referents,
        "splits": splits,
        "manifest": manifest,
        "objective": objective,
        "identity": identity,
        "run_receipt": run_receipt,
        "manifest_sha256": _sha256(manifest_path),
    }


def validate_shared_referents(rows: dict[str, dict[str, Any]]) -> None:
    """Require identical ordered referents, factors, and splits across every scale point."""
    if len(rows) < 2:
        raise ValueError("the scale atlas requires at least two citable caches")
    first_tag = sorted(rows)[0]
    first = rows[first_tag]
    for tag, row in rows.items():
        if row["referents"] != first["referents"]:
            raise ValueError(f"referent order mismatch: {first_tag} vs {tag}")
        if not torch.equal(row["factor_a"], first["factor_a"]):
            raise ValueError(f"factor_a mismatch: {first_tag} vs {tag}")
        if not torch.equal(row["factor_b"], first["factor_b"]):
            raise ValueError(f"factor_b mismatch: {first_tag} vs {tag}")
        if row["splits"] != first["splits"]:
            raise ValueError(f"frozen split mismatch: {first_tag} vs {tag}")
    split_values = first["splits"]
    if not isinstance(split_values, dict) or not split_values.get("train"):
        raise ValueError("scale atlas requires a non-empty frozen train split")
    heldout = [index for key in ("val", "test") for index in split_values.get(key, [])]
    if not heldout:
        raise ValueError("scale atlas requires a non-empty frozen val or test split")
    assigned = [index for values in split_values.values() for index in values]
    expected = list(range(len(first["referents"])))
    if sorted(assigned) != expected:
        raise ValueError("frozen splits must assign every referent exactly once")


def frozen_split_probe(
    x: torch.Tensor,
    labels: torch.Tensor,
    splits: dict[str, list[int]],
    *,
    seed: int,
    epochs: int = 300,
    shuffle_train_labels: bool = False,
) -> dict[str, Any]:
    """Train only on the manifest train rows and evaluate on the frozen val plus test rows."""
    train_idx = torch.tensor(splits["train"], dtype=torch.long)
    heldout_values = [index for key in ("val", "test") for index in splits.get(key, [])]
    heldout_idx = torch.tensor(heldout_values, dtype=torch.long)
    x = torch.as_tensor(x).detach().float().flatten(1)
    labels = torch.as_tensor(labels).detach().long()
    all_classes = sorted(set(labels.tolist()))
    train_classes = sorted(set(labels[train_idx].tolist()))
    if train_classes != all_classes:
        raise ValueError(f"frozen train split lacks factor classes: train={train_classes}, all={all_classes}")
    n_classes = max(all_classes) + 1
    mean = x[train_idx].mean(0, keepdim=True)
    scale = x[train_idx].std(0, unbiased=False, keepdim=True).clamp_min(1e-5)
    x_train = (x[train_idx] - mean) / scale
    x_heldout = (x[heldout_idx] - mean) / scale
    y_train = labels[train_idx]
    if shuffle_train_labels:
        generator = torch.Generator().manual_seed(seed + 91_337)
        y_train = y_train[torch.randperm(y_train.shape[0], generator=generator)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = torch.nn.Linear(x.shape[1], n_classes)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
        for _ in range(epochs):
            optimizer.zero_grad()
            F.cross_entropy(head(x_train), y_train).backward()
            optimizer.step()
        with torch.no_grad():
            prediction = head(x_heldout).argmax(-1)
            score = float((prediction == labels[heldout_idx]).float().mean())
    heldout_counts = torch.bincount(labels[heldout_idx], minlength=n_classes)
    return {
        "score": round(score, 4),
        "train_n": int(train_idx.numel()),
        "heldout_n": int(heldout_idx.numel()),
        "uniform_chance": round(1.0 / n_classes, 4),
        "heldout_majority_baseline": round(float(heldout_counts.max() / heldout_counts.sum()), 4),
        "train_label_shuffled": shuffle_train_labels,
    }


_ARCHITECTURE_FIELDS = (
    "arch",
    "embed_dim",
    "patch_size",
    "tubelet",
    "frames_per_clip",
    "resolution",
    "dense",
    "pool",
)


def _architecture_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    config = row["manifest"]["encoder_config"]
    return tuple(config.get(field) for field in _ARCHITECTURE_FIELDS)


def _control_matches(rows: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    learned = {tag: row for tag, row in rows.items() if row["objective"] != "random-control"}
    random_rows = {tag: row for tag, row in rows.items() if row["objective"] == "random-control"}
    matches: dict[str, list[dict[str, Any]]] = {}
    for learned_tag, learned_row in learned.items():
        learned_identity = learned_row["identity"]
        records: list[dict[str, Any]] = []
        for control_tag, control_row in random_rows.items():
            control_identity = control_row["identity"]
            identity_match = all(
                str(learned_identity.get(field) or "") == str(control_identity.get(field) or "")
                for field in ("model_id", "revision")
            )
            architecture_match = _architecture_signature(learned_row) == _architecture_signature(control_row)
            if not (identity_match and architecture_match):
                continue
            learned_stimulus = (learned_row["run_receipt"].get("stimulus") or {}).get("set_sha256")
            control_stimulus = (control_row["run_receipt"].get("stimulus") or {}).get("set_sha256")
            records.append(
                {
                    "tag": control_tag,
                    "architecture_exact": True,
                    "seed": control_identity.get("seed"),
                    "state_dict_sha256": control_identity.get("state_dict_sha256"),
                    "stimulus_hash_exact": bool(
                        learned_stimulus and control_stimulus and learned_stimulus == control_stimulus
                    ),
                    "stimulus_hash_limitation": (
                        None
                        if learned_stimulus and control_stimulus
                        else "one or both cache runs predate per-input stimulus hashes"
                    ),
                }
            )
        matches[learned_tag] = records
    return matches


def build_local_scale_atlas(
    caches: dict[str, Path | str],
    *,
    seeds: tuple[int, ...] = (17, 29, 43, 59, 71),
    permutations: int = 500,
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("scale atlas seeds must be a non-empty unique tuple")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    rows = {tag: _load_cache(Path(path)) for tag, path in caches.items()}
    validate_shared_referents(rows)
    first = rows[sorted(rows)[0]]
    factor_reports: dict[str, list[dict[str, Any]]] = {"factor_a": [], "factor_b": []}
    for seed in seeds:
        for factor_name in factor_reports:
            labels = first[factor_name]
            probe_acc = {
                tag: frozen_split_probe(row["latents"], labels, first["splits"], seed=seed)
                for tag, row in rows.items()
            }
            shuffled_null = {
                tag: frozen_split_probe(
                    row["latents"],
                    labels,
                    first["splits"],
                    seed=seed,
                    shuffle_train_labels=True,
                )
                for tag, row in rows.items()
            }
            factor_reports[factor_name].append(
                {
                    "seed": seed,
                    "protocol": "manifest train only; manifest val plus test held out",
                    "probe": probe_acc,
                    "shuffled_train_label_null": shuffled_null,
                }
            )

    learned_rows = {tag: row for tag, row in rows.items() if row["objective"] != "random-control"}
    if len(learned_rows) < 2:
        raise ValueError("scale atlas requires at least two learned scale points")
    learned_reps = {tag: row["latents"] for tag, row in learned_rows.items()}
    alignment = alignment_table(
        learned_reps,
        n_permutations=permutations,
        seed=seeds[0],
        k=max(1, min(3, len(first["referents"]) - 1)),
    )
    pair_min_p = min(float(pair["p_value"]) for pair in alignment["pairs"].values())
    pair_count = len(alignment["pairs"])
    control_matches = _control_matches(rows)
    matched_architecture = bool(control_matches) and all(control_matches.values())
    matched_stimulus = matched_architecture and all(
        any(record["stimulus_hash_exact"] for record in records) for records in control_matches.values()
    )
    random_seeds_per_architecture = {
        tag: sorted(
            {
                int(record["seed"])
                for record in records
                if isinstance(record.get("seed"), int) and not isinstance(record.get("seed"), bool)
            }
        )
        for tag, records in control_matches.items()
    }
    reasons = [
        "programmatic native-resolution renders rather than rights-clean natural video",
        f"n={len(first['referents'])} is a mechanics and scale-signal pilot",
    ]
    if not matched_architecture:
        reasons.append("matched random-architecture cache is not present for every learned scale")
    if not matched_stimulus:
        reasons.append("byte-identical stimuli are not proven for every learned/control pair")
    if matched_architecture and any(len(values) < 3 for values in random_seeds_per_architecture.values()):
        reasons.append("fewer than three random-initialization seeds per architecture")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": (
            "serial local V-JEPA scale atlas on shared generative referents rendered at each model's "
            "native resolution"
        ),
        "caches": {
            tag: {
                "path": str(row["path"]),
                "manifest_sha256": row["manifest_sha256"],
                "backend": row["manifest"]["encoder_config"].get("actual_backend"),
                "model_id": row["manifest"]["encoder_config"].get("hf_id"),
                "revision": row["manifest"]["encoder_config"].get("revision"),
                "shape": list(row["latents"].shape),
                "objective": row["objective"],
                "random_init_seed": row["identity"].get("seed"),
                "state_dict_sha256": row["identity"].get("state_dict_sha256"),
            }
            for tag, row in rows.items()
        },
        "referent_contract": {
            "exact_order_match": True,
            "count": len(first["referents"]),
            "referent_sha256": hashlib.sha256("\n".join(first["referents"]).encode("utf-8")).hexdigest(),
            "factors_match": True,
            "splits_match": True,
            "pixel_bytes_verified_across_caches": matched_stimulus,
            "limitation": (
                "referent IDs and generative factors match exactly, but native-resolution renders are "
                "not the same pixel tensor and old runs lack input hashes"
            ),
        },
        "seeds": list(seeds),
        "factor_reports": factor_reports,
        "alignment": alignment,
        "pairwise_permutation_min_p": pair_min_p,
        "pairwise_permutation_min_p_interpretation": "raw exploratory minimum, not a familywise p-value",
        "pairwise_permutation_min_bonferroni": min(1.0, pair_min_p * pair_count),
        "controls": {
            "row_permutation": True,
            "shuffled_train_labels_on_frozen_split": True,
            "full_rank_random_map": False,
            "matched_random_architecture": matched_architecture,
            "matched_stimulus_hashes": matched_stimulus,
            "matches": control_matches,
            "random_seeds_per_architecture": random_seeds_per_architecture,
            "warning": (
                "the former full-rank random-map arm was removed from claim evidence because an "
                "invertible linear map preserves linear decodability; matched random architectures are "
                "the relevant learned-code control"
            ),
        },
        "claim_eligibility": {
            "promotable": False,
            "reasons": reasons,
        },
        "migration_use": {
            "model_availability_blocker_retired": True,
            "local_serial_execution_proven": True,
            "requirements_export": [
                "preserve exact referent identity across encoder replacements",
                "measure factor retention separately for hue and orientation-motion",
                "compare pairwise geometry against row permutation",
                (
                    "require matched random-architecture and byte-hashed stimulus controls before "
                    "learned-code claims"
                ),
                "price throughput and memory per native resolution",
            ],
        },
        "pair_tags": [list(pair) for pair in combinations(sorted(learned_rows), 2)],
        "artifact_validation": {
            "all_caches_citable": True,
            "ordered_referents_factors_and_splits_match": True,
            "meaning_of_all_ok": (
                "receipt mechanics are valid; scientific promotability is separate and false"
            ),
        },
        "all_ok": True,
    }


def write_local_scale_atlas(receipt: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
