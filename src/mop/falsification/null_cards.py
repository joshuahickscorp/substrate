"""Null-card generator and validator.

The experiment registry is the preregistration source of truth. This module turns one registry
row into a draft null/survival card and validates the machine-readable YAML block in an existing
card. It is deliberately small: the goal is to remove prose drift before Studio-scale runs, not to
invent a second registry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..devel.registries import load_experiments
from ..provenance import RESULT_TAGS

REQUIRED_FIELDS = (
    "exp_id",
    "title",
    "hypothesis",
    "null_hypothesis",
    "baseline",
    "ablation",
    "metric",
    "probe_dependency",
    "encoder_scale",
    "seeds",
    "provenance_tag",
    "result",
    "taxonomy_category",
    "verdict",
    "badges",
    "raw_run_id",
    "repro_level",
)
PROBE_FIELDS = ("factor", "encoder", "atlas_row", "decodable", "acc_above_chance")
SEED_FIELDS = ("n", "sem", "sign_stability")
VERDICTS = (
    "PUBLISH-POSITIVE",
    "DOWNGRADE-TIE",
    "SUBSTRATE-BOUND",
    "SEED-UNSTABLE",
    "CAPACITY-ARTIFACT",
    "REPLICATION-FAILED",
)
DECODABLE = ("yes", "no", "marginal", "TODO")
_YAML_BLOCK = re.compile(r"```yaml\s*(.*?)```", re.S)
_TODO = re.compile(r"\bTODO\b|<[^>]+>")


def experiment_by_id(exp_id: str) -> dict[str, Any]:
    """Return one experiment row from registry/experiments.yaml."""
    for row in load_experiments():
        if str(row.get("id")) == exp_id:
            return row
    raise KeyError(f"unknown experiment id {exp_id!r}")


def generate_from_experiment(exp_id: str, *, encoder: str = "vjepa2_vitl_fpc64_256") -> dict[str, Any]:
    """Generate a draft card spec from the preregistered experiment row."""
    row = experiment_by_id(exp_id)
    proof = dict(row.get("proof") or {})
    factor = str(proof.get("atlas_factor") or "TODO")
    metrics = list(row.get("metrics") or [])
    controls = list(row.get("controls") or [])
    return {
        "exp_id": row["id"],
        "title": str(row["name"]),
        "hypothesis": str(row.get("mechanism") or row.get("question")),
        "null_hypothesis": str(row["null_hypothesis"]),
        "baseline": ", ".join(str(c) for c in controls) if controls else "TODO: tuned baseline",
        "ablation": str(row["falsifier"]),
        "metric": str(metrics[0]) if metrics else "TODO",
        "probe_dependency": {
            "factor": factor,
            "encoder": encoder,
            "atlas_row": f"proof/atlas/{encoder}/{factor}.json" if factor != "TODO" else "TODO",
            "decodable": "TODO",
            "acc_above_chance": "TODO",
        },
        "encoder_scale": "L",
        "seeds": {"n": 3, "sem": "TODO", "sign_stability": "TODO"},
        "provenance_tag": "provisional",
        "result": "TODO: fill with numbers and confidence intervals after the run",
        "taxonomy_category": int(row["taxonomy_slot"]),
        "verdict": "DOWNGRADE-TIE",
        "badges": ["preregistered"],
        "raw_run_id": "TODO: runs/<path> plus config",
        "repro_level": str(proof.get("evidence_level") or "R0"),
    }


def render_card(card: dict[str, Any]) -> str:
    """Render a card as Markdown with one machine-readable YAML block."""
    title = str(card["title"])
    exp_id = str(card["exp_id"])
    yaml_block = _dump_yaml(card)
    return "\n".join(
        [
            f"# Null card draft: {exp_id}",
            "",
            "Generated from `registry/experiments.yaml`. Replace TODO fields before treating this as a",
            "completed proof card. A tie is a null. The encoder is frozen and never trained.",
            "",
            "## Claim Under Test",
            "",
            title,
            "",
            "## Machine-Readable Card",
            "",
            "```yaml",
            yaml_block.rstrip(),
            "```",
            "",
        ]
    )


def extract_card_yaml(text: str) -> dict[str, Any]:
    """Extract the first fenced YAML block from a null card."""
    m = _YAML_BLOCK.search(text)
    if not m:
        raise ValueError("no fenced yaml block found")
    block = m.group(1)
    try:
        data = OmegaConf.to_container(OmegaConf.create(block), resolve=True)
    except Exception:
        data = _loose_card_yaml(block)
    if not isinstance(data, dict):
        raise ValueError("yaml block is not a mapping")
    return dict(data)


def load_card(path: Path | str) -> dict[str, Any]:
    return extract_card_yaml(Path(path).read_text())


def validate_card(card: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Validate the machine-readable card fields. strict=True refuses TODO placeholders."""
    problems: list[str] = []
    exp_id = str(card.get("exp_id", "<no-exp-id>"))
    for field in REQUIRED_FIELDS:
        if field not in card or card.get(field) in (None, "", []):
            problems.append(f"{exp_id}: missing required field {field!r}")
    if problems:
        return problems

    probe = card.get("probe_dependency")
    if not isinstance(probe, dict):
        problems.append(f"{exp_id}: probe_dependency must be a mapping")
    else:
        for field in PROBE_FIELDS:
            if field not in probe or probe.get(field) == "":
                problems.append(f"{exp_id}: probe_dependency missing {field!r}")
        decodable = str(probe.get("decodable"))
        if decodable not in DECODABLE:
            problems.append(f"{exp_id}: decodable {decodable!r} not in {DECODABLE}")

    seeds = card.get("seeds")
    if not isinstance(seeds, dict):
        problems.append(f"{exp_id}: seeds must be a mapping")
    else:
        for field in SEED_FIELDS:
            if field not in seeds or seeds.get(field) == "":
                problems.append(f"{exp_id}: seeds missing {field!r}")
        try:
            n_value = seeds.get("n", 0)
            if int(n_value if n_value is not None else 0) < 3:
                problems.append(f"{exp_id}: seeds.n must be >= 3")
        except Exception:
            problems.append(f"{exp_id}: seeds.n must be an integer")

    tag = str(card.get("provenance_tag"))
    if tag not in RESULT_TAGS:
        problems.append(f"{exp_id}: provenance_tag {tag!r} not in {RESULT_TAGS}")
    verdict = str(card.get("verdict"))
    if verdict not in VERDICTS:
        problems.append(f"{exp_id}: verdict {verdict!r} not in {VERDICTS}")
    repro = str(card.get("repro_level"))
    if not re.fullmatch(r"R[0-5]", repro):
        problems.append(f"{exp_id}: repro_level {repro!r} must be R0..R5")
    if not isinstance(card.get("badges"), list):
        problems.append(f"{exp_id}: badges must be a list")

    tax = card.get("taxonomy_category")
    if tax != "null rejected":
        try:
            tax_value = tax if tax is not None else 0
            if int(tax_value) not in range(1, 11):
                problems.append(f"{exp_id}: taxonomy_category {tax!r} not in 1..10")
        except Exception:
            problems.append(f"{exp_id}: taxonomy_category {tax!r} not in 1..10 or 'null rejected'")

    if strict:
        for path, value in _walk(card):
            if isinstance(value, str) and _TODO.search(value):
                problems.append(f"{exp_id}: strict placeholder at {'.'.join(path)}")
    return problems


def schema() -> dict[str, Any]:
    """A compact JSON-schema-like receipt for tools and humans."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "MOP null card",
        "type": "object",
        "required": list(REQUIRED_FIELDS),
        "properties": {
            "probe_dependency": {"type": "object", "required": list(PROBE_FIELDS)},
            "seeds": {"type": "object", "required": list(SEED_FIELDS)},
            "provenance_tag": {"enum": list(RESULT_TAGS)},
            "verdict": {"enum": list(VERDICTS)},
        },
    }


def _dump_yaml(card: dict[str, Any]) -> str:
    return OmegaConf.to_yaml(OmegaConf.create(json.loads(json.dumps(card, default=str))))


def _loose_card_yaml(block: str) -> dict[str, Any]:
    """Parse historical card YAML that left colon-heavy prose unquoted.

    This is not a general YAML parser. It accepts the null-card subset: top-level key/value pairs,
    one-level nested mappings, and list items. New generated cards still use real YAML.
    """
    root: dict[str, Any] = {}
    current: str | None = None
    for raw in block.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if line.startswith("- "):
            if current is not None:
                if not isinstance(root.get(current), list):
                    root[current] = []
                root[current].append(_scalar(line[2:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip("'\"")
        value = value.strip()
        if indent == 0:
            if value == "":
                root[key] = {}
                current = key
            else:
                root[key] = _scalar(value)
                current = key
        elif current is not None and isinstance(root.get(current), dict):
            root[current][key] = _scalar(value)
    return root


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in ("null", "None", "~"):
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _walk(obj: Any, prefix: tuple[str, ...] = ()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk(value, (*prefix, str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk(value, (*prefix, str(i)))
    else:
        yield prefix, obj
