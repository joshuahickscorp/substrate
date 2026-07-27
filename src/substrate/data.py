"""Substrate data custody and deterministic cache restoration.

Only datasets already admitted by the sealed custody authority are available here. Restoration downloads
the same public archives and applies the frozen window, split, normalization, and stream construction
rules. It does not run an experiment or change any scientific premise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = ROOT / "evidence" / "substrate" / "v1" / "SUBSTRATE_DATA_CUSTODY_AUTHORITY.json"


class Refused(RuntimeError):
    """A data action outside the admitted custody authority."""


def authority() -> dict:
    return json.loads(AUTHORITY.read_text())


def root() -> Path:
    override = os.environ.get("SUBSTRATE_DATA_ROOT")
    configured = override or authority()["canonical_root"]
    return Path(configured).expanduser()


def cache_path(dataset: str) -> Path:
    """Resolve the admitted cache, preferring the operator-owned custody root.

    The repository copy is a byte-identical, content-addressed portability fallback for clean-clone
    verification. It is not an additional dataset, a new trial, or a scientific authority.
    """

    try:
        record = authority()["datasets"][dataset]
    except KeyError as exc:
        raise Refused(f"dataset {dataset!r} is not admitted") from exc
    external = root() / record["cache"]
    if external.is_file():
        selected = external
    else:
        bundled = ROOT / record.get("bundled_cache", "")
        selected = bundled if bundled.is_file() else external
    if selected.is_file():
        observed = hashlib.sha256(selected.read_bytes()).hexdigest()
        expected = record.get("cache_sha256")
        if expected and observed != expected:
            raise Refused(f"{dataset!r} cache failed custody hash verification")
    return selected


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        base = target.resolve()
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if destination != base and base not in destination.parents:
                raise Refused(f"archive member escapes the custody root: {member.filename!r}")
        bundle.extractall(target)


def _download(dataset: str) -> Path:
    try:
        record = authority()["datasets"][dataset]
    except KeyError as exc:
        raise Refused(f"dataset {dataset!r} is not admitted") from exc
    destination = root() / record["raw_subdirectory"]
    if any(destination.rglob("*")):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f"substrate-{dataset}-"))
    archive = temporary_dir / f"{dataset}.zip"
    try:
        with urllib.request.urlopen(record["archive_url"], timeout=120) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        _safe_extract(archive, destination)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return destination


def _windows(signal: np.ndarray, activity: np.ndarray, subject, window: int, stride: int, decimation: int):
    features, labels, units = [], [], []
    for start in range(0, len(signal) - window, stride):
        segment = activity[start : start + window]
        if segment[0] == 0 or not np.all(segment == segment[0]):
            continue
        features.append(signal[start : start + window : decimation])
        labels.append(int(segment[0]))
        units.append(subject)
    return features, labels, units


def _stream_from(features, labels, units, per_stream: int, n_streams: int, decimation: int, seed: int):
    by_unit: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        by_unit.setdefault(str(unit), []).append(index)
    usable = [unit for unit, indexes in by_unit.items() if len(indexes) >= per_stream]
    random = np.random.default_rng(seed)
    streams, targets, source_units = [], [], []
    for _ in range(n_streams):
        if not usable:
            break
        unit = usable[int(random.integers(len(usable)))]
        selected = random.choice(by_unit[unit], per_stream, replace=False)
        streams.append(np.concatenate([features[index] for index in selected])[::decimation])
        targets.append(int(labels[selected[-1]]))
        source_units.append(unit)
    if not streams:
        raise Refused("the admitted source produced no usable streams")
    return np.stack(streams).astype(np.float32), np.array(targets), np.array(source_units)


def _publish_cache(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def build_harth(*, download: bool = False, per_stream: int = 3, n_train: int = 4000) -> Path:
    cache = root() / authority()["datasets"]["harth"]["cache"]
    if cache.is_file():
        return cache
    raw = root() / authority()["datasets"]["harth"]["raw_subdirectory"]
    if not any(raw.rglob("*.csv")):
        if not download:
            raise Refused("HARTH raw data is absent; pass download=True to restore the admitted public archive")
        raw = _download("harth")

    all_features, all_labels, all_units = [], [], []
    columns = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
    for path in sorted(raw.rglob("*.csv")):
        rows, labels = [], []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(columns) <= set(reader.fieldnames or []):
                continue
            for row in reader:
                try:
                    rows.append([float(row[column]) for column in columns])
                    labels.append(int(float(row["label"])))
                except (ValueError, KeyError, TypeError):
                    continue
        if len(rows) < 2000:
            continue
        features, targets, units = _windows(
            np.asarray(rows, dtype=np.float32),
            np.asarray(labels, dtype=int),
            path.stem,
            250,
            250,
            5,
        )
        all_features += features
        all_labels += targets
        all_units += units

    labels = np.array(all_labels)
    if not labels.size:
        raise Refused("HARTH archive contained no admissible labeled windows")
    counts = np.bincount(labels)
    kept = [int(label) for label in np.argsort(-counts)[:6] if counts[label] > 50]
    remap = {label: index for index, label in enumerate(sorted(kept))}
    selected = [index for index, label in enumerate(labels) if label in remap]
    features = [all_features[index] for index in selected]
    labels = np.array([remap[int(labels[index])] for index in selected])
    units = np.array([all_units[index] for index in selected])
    mean = np.mean([feature.mean(0) for feature in features], 0)
    deviation = np.std([feature.mean(0) for feature in features], 0) + 1e-6
    features = [(feature - mean) / deviation for feature in features]
    unique_units = np.unique(units)
    test_units = set(unique_units[: max(1, len(unique_units) // 3)].tolist())
    training = [index for index, unit in enumerate(units) if unit not in test_units]
    testing = [index for index, unit in enumerate(units) if unit in test_units]
    xtr, ytr, utr = _stream_from([features[i] for i in training], labels[training], units[training], per_stream, n_train, 2, 0)
    xte, yte, ute = _stream_from([features[i] for i in testing], labels[testing], units[testing], per_stream, n_train // 3, 2, 1)
    return _publish_cache(cache, {"Xtr": xtr, "Ytr": ytr, "Utr": utr, "Xte": xte, "Yte": yte, "Ute": ute})


def build_pamap2(*, download: bool = False, per_stream: int = 3, n_train: int = 4000) -> Path:
    cache = root() / authority()["datasets"]["pamap2"]["cache"]
    if cache.is_file():
        return cache
    raw = root() / authority()["datasets"]["pamap2"]["raw_subdirectory"]
    if not any(raw.rglob("subject*.dat")):
        if not download:
            raise Refused("PAMAP2 raw data is absent; pass download=True to restore the admitted public archive")
        raw = _download("pamap2")
    # The UCI download currently wraps the dataset in a second ZIP. This is packaging only: extract the
    # nested archive through the same traversal guard and keep the frozen scientific builder unchanged.
    if not any(raw.rglob("subject*.dat")):
        for nested in sorted(raw.rglob("*.zip")):
            _safe_extract(nested, raw)

    columns = [4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46]
    all_features, all_labels, all_units = [], [], []
    for path in sorted(raw.rglob("subject*.dat")):
        subject = int(path.name.split("subject")[1][:3])
        values = np.loadtxt(path)
        activity = values[:, 1].astype(int)
        signal = values[:, columns]
        for index in range(signal.shape[1]):
            column = signal[:, index]
            missing = np.isnan(column)
            if missing.any():
                present = np.where(~missing)[0]
                if len(present):
                    column[missing] = np.interp(np.where(missing)[0], present, column[present])
                signal[:, index] = column
        features, labels, units = _windows(np.nan_to_num(signal).astype(np.float32), activity, subject, 256, 256, 4)
        all_features += features
        all_labels += labels
        all_units += units

    labels = np.array(all_labels)
    if not labels.size:
        raise Refused("PAMAP2 archive contained no admissible labeled windows")
    unique = np.unique(labels)
    kept = unique[np.argsort(-np.bincount(labels)[unique])][:6]
    selected = [index for index, label in enumerate(labels) if label in kept]
    remap = {int(label): index for index, label in enumerate(sorted(kept))}
    features = [all_features[index] for index in selected]
    labels = np.array([remap[int(labels[index])] for index in selected])
    units = np.array([all_units[index] for index in selected])
    mean = np.mean([feature.mean(0) for feature in features], 0)
    deviation = np.std([feature.mean(0) for feature in features], 0) + 1e-6
    features = [(feature - mean) / deviation for feature in features]
    unique_units = np.unique(units)
    test_units = set(unique_units[: max(1, len(unique_units) // 3)].tolist())
    training = [index for index, unit in enumerate(units) if unit not in test_units]
    testing = [index for index, unit in enumerate(units) if unit in test_units]
    xtr, ytr, utr = _stream_from([features[i] for i in training], labels[training], units[training], per_stream, n_train, 2, 0)
    xte, yte, ute = _stream_from([features[i] for i in testing], labels[testing], units[testing], per_stream, n_train // 3, 2, 1)
    return _publish_cache(cache, {"Xtr": xtr, "Ytr": ytr, "Utr": utr, "Xte": xte, "Yte": yte, "Ute": ute})


def inspect() -> dict:
    datasets = {}
    for name, record in authority()["datasets"].items():
        external = root() / record["cache"]
        cache = cache_path(name)
        row = {
            "cache": str(cache),
            "source": "operator_custody" if cache == external else "repository_fallback",
            "present": cache.is_file(),
            "sha256": None,
            "bytes": 0,
            "arrays": {},
        }
        if cache.is_file():
            row["sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
            row["bytes"] = cache.stat().st_size
            with np.load(cache) as arrays:
                row["arrays"] = {key: list(arrays[key].shape) for key in sorted(arrays.files)}
        datasets[name] = row
    missing = sorted(name for name, row in datasets.items() if not row["present"])
    return {
        "schema": "substrate-data-status/v1",
        "root": str(root()),
        "datasets": datasets,
        "missing": missing,
        "all_present": not missing,
        "activation": False,
    }


def restore() -> dict:
    paths = {
        "harth": build_harth(download=True),
        "pamap2": build_pamap2(download=True),
    }
    return {
        "restored": {name: str(path) for name, path in paths.items()},
        "status": inspect(),
        "scientific_run_launched": False,
        "activation": False,
    }


def seal_authority() -> Path:
    from substrate import evidence as io

    document = {key: value for key, value in authority().items() if key not in {"program", "source_commit", "sha256"}}
    return io.seal("SUBSTRATE_DATA_CUSTODY_AUTHORITY.json", document)


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "status"
    if command == "seal":
        path = seal_authority()
        sealed = path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path)
        print(json.dumps({"sealed": sealed, "status": inspect()}, indent=2))
    elif command == "status":
        print(json.dumps(inspect(), indent=2))
    elif command == "restore":
        print(json.dumps(restore(), indent=2))
    else:
        raise ValueError(argv)


if __name__ == "__main__":
    main()
