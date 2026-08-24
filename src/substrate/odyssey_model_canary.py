"""Deterministic public-only local-model canaries for Odyssey base selection.

This module is deliberately not an Odyssey worker.  It can inventory locally
available Ollama models and evaluate only the frozen, public prompt set.  It
never creates hidden task seeds or evaluator material, chooses an arm, seals a
human gate, or installs/starts a supervisor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from substrate import odyssey_density as density
from substrate import odyssey_transition

PROGRAM = "substrate-odyssey-7d-v1"
PLAN = Path("plans/substrate/tangible_next_launch")
TEMPLATE = PLAN / "ODYSSEY_PUBLIC_MODEL_CANARY.template.json"
SCHEMA = "SUBSTRATE_ODYSSEY_PUBLIC_MODEL_CANARY/v1"
GIB = 1024**3
OLLAMA = "http://127.0.0.1:11434"
FINAL_PATTERN = re.compile(r"(?im)^\s*FINAL\s*:\s*([^\r\n]+)\s*$")
# Re-export the measured gateway pin so G02/G05 runtime identity and arms share
# one contract value (never a silent operator-only environment default).
PINNED_OLLAMA_NUM_PARALLEL = density.PINNED_OLLAMA_NUM_PARALLEL
GATEWAY_REVISION = density.GATEWAY_REVISION


class Refused(RuntimeError):
    """The public canary cannot safely produce a selection receipt."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, *, require_digest: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} must contain a JSON object")
    if require_digest:
        unsigned = dict(value)
        claimed = unsigned.pop("sha256", None)
        if not isinstance(claimed, str) or claimed != digest(unsigned):
            raise Refused(f"{path} has an invalid self-digest")
    return value


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    if path.exists():
        raise Refused(f"refusing to overwrite public model-canary receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise Refused(f"{label} must be a sha256 string")
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized.lower()):
        raise Refused(f"{label} must be a sha256 string")
    return normalized.lower()


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Refused(f"{label} must be non-empty text")
    return value.strip()


def _int(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise Refused(f"{label} must be an integer at least {minimum}")
    return value


def _api(path: str, *, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    body = canonical(payload) if payload is not None else None
    request = urllib.request.Request(
        f"{OLLAMA}{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- fixed local endpoint
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        raise Refused(f"Ollama API {path} failed: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"Ollama API {path} did not return an object")
    return value


def _pageout_bytes() -> int:
    """Return the cumulative macOS pageout counter in bytes, not pages."""
    completed = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise Refused("cannot sample macOS pageouts")
    pageouts = re.search(r"Pageouts:\s+(\d+)", completed.stdout)
    page_size = re.search(r"page size of (\d+) bytes", completed.stdout, re.IGNORECASE)
    if pageouts is None or page_size is None:
        raise Refused("macOS pageout counter is unavailable")
    return int(pageouts.group(1)) * int(page_size.group(1))


def _runtime() -> dict[str, str]:
    completed = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise Refused("cannot resolve local Ollama runtime version")
    version = _api("/api/version").get("version")
    # Bind the parallel-slot pin into the runtime identity that G02 seals.  A
    # later service mismatch is then identity drift, not a silent slowdown.
    try:
        density.assert_ollama_num_parallel_pinned(require_running=True)
    except density.DensityRefused as error:
        raise Refused(str(error)) from error
    identity = density.gateway_runtime_identity(
        cli=completed.stdout.strip(),
        api_version=_nonempty(version, label="Ollama API version"),
    )
    return {"id": "ollama", "version": identity["api"], "sha256": digest(identity)}


def _template(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    if frozen.get("schema") != "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1" or frozen.get("activation") is not False:
        raise Refused("current Odyssey frozen build is not inactive and valid")
    inputs = frozen.get("input_sha256")
    implementation = frozen.get("implementation_sha256")
    if not isinstance(inputs, dict) or not isinstance(implementation, dict):
        raise Refused("current Odyssey frozen build lacks source maps")
    source_inputs = odyssey_transition.build_inputs(root)
    source_implementation = odyssey_transition.implementation_inputs(root)
    if set(inputs) != set(source_inputs) or set(implementation) != set(source_implementation):
        raise Refused("current Odyssey frozen source maps do not match the controller")
    for name, path in source_inputs.items():
        if not path.is_file() or inputs[name] != file_digest(path):
            raise Refused(f"frozen Odyssey input drift: {name}")
    for name, path in source_implementation.items():
        if not path.is_file() or implementation[name] != file_digest(path):
            raise Refused(f"frozen Odyssey implementation drift: {name}")
    template = _read_json(root / TEMPLATE)
    if inputs.get("public_model_canary_template") != file_digest(root / TEMPLATE):
        raise Refused("public model-canary template is not bound to the frozen build")
    required = {
        "schema",
        "program",
        "status",
        "activation",
        "external_activation",
        "visibility",
        "neutral_organ_prompt",
        "reasoning_effort_policy",
        "conversation_policy",
        "max_output_tokens",
        "candidate_aliases",
        "model_service_cap_bytes",
        "required_concurrent_clients",
        "selection_rule",
        "hidden_seed_policy",
        "case_set",
    }
    if set(template) != required:
        raise Refused("public model-canary template has an unexpected shape")
    if (
        template.get("schema") != "SUBSTRATE_ODYSSEY_PUBLIC_MODEL_CANARY_TEMPLATE/v1"
        or template.get("program") != PROGRAM
        or template.get("status") != "template_unsealed"
        or template.get("activation") is not False
        or template.get("external_activation") is not False
        or template.get("visibility") != "public_only"
        or not isinstance(template.get("neutral_organ_prompt"), str)
        or not template["neutral_organ_prompt"].strip()
        or template.get("reasoning_effort_policy") != "fixed_default_no_frontier_override"
        or template.get("conversation_policy") != "fresh_request_context_per_case; keep_alive_only_for_weights; no_cross_case_state"
        or template.get("max_output_tokens") != 512
        or template.get("model_service_cap_bytes") != 24 * GIB
        or template.get("required_concurrent_clients") != 8
        or template.get("selection_rule")
        != "highest_public_canary_score_then_lower_service_peak_then_lower_median_latency_then_lexical_weight_digest"
    ):
        raise Refused("public model-canary template does not retain the frozen selection policy")
    aliases = template.get("candidate_aliases")
    if not isinstance(aliases, list) or len(aliases) < 1 or len(set(aliases)) != len(aliases):
        raise Refused("public model-canary template candidate aliases are invalid")
    if not all(isinstance(alias, str) and alias.strip() for alias in aliases):
        raise Refused("public model-canary template candidate aliases are invalid")
    cases = template.get("case_set")
    if not isinstance(cases, list) or len(cases) != 16:
        raise Refused("public model-canary template must retain exactly sixteen public cases")
    expected_ids = [f"{frontier}{ordinal}" for frontier in "ABCDEFGH" for ordinal in (1, 2)]
    if [case.get("id") if isinstance(case, dict) else None for case in cases] != expected_ids:
        raise Refused("public model-canary cases must retain ordered A1-H2 identifiers")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "frontier", "seed", "prompt", "answer"}:
            raise Refused("public model-canary case has an unexpected shape")
        if case["frontier"] != case["id"][0]:
            raise Refused("public model-canary case does not bind its frontier")
        _int(case["seed"], label=f"public canary {case['id']}.seed", minimum=1)
        _nonempty(case["prompt"], label=f"public canary {case['id']}.prompt")
        _nonempty(case["answer"], label=f"public canary {case['id']}.answer")
    return frozen, template


def _normal_answer(value: str) -> str:
    final = FINAL_PATTERN.findall(value)
    selected = final[-1] if final else value.strip().splitlines()[-1] if value.strip() else ""
    return re.sub(r"\s+", "", selected.strip().casefold().rstrip(". "))


def _tag_record(name: str) -> dict[str, Any]:
    values = _api("/api/tags").get("models")
    if not isinstance(values, list):
        raise Refused("Ollama tag inventory is malformed")
    for row in values:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    raise Refused(f"requested public-canary model is not locally present: {name}")


def _base_model(name: str, tag: dict[str, Any], runtime: dict[str, str]) -> dict[str, str]:
    shown = _api("/api/show", payload={"model": name, "verbose": True})
    model_info = shown.get("model_info")
    if not isinstance(model_info, dict) or not model_info:
        raise Refused(f"Ollama show response lacks model info for {name}")
    tokenizer = {key: value for key, value in model_info.items() if isinstance(key, str) and key.startswith("tokenizer.")}
    if not tokenizer:
        raise Refused(f"Ollama show response lacks tokenizer metadata for {name}")
    details = shown.get("details") if isinstance(shown.get("details"), dict) else tag.get("details")
    if not isinstance(details, dict):
        details = {}
    quantization = details.get("quantization_level") or model_info.get("general.file_type")
    return {
        "id": name,
        "revision": f"ollama:{_sha256(tag.get('digest'), label=f'{name} digest')[:16]}",
        "weight_sha256": _sha256(tag.get("digest"), label=f"{name} digest"),
        "tokenizer_sha256": digest(tokenizer),
        "runtime_sha256": runtime["sha256"],
        "quantization": _nonempty(str(quantization) if quantization is not None else None, label=f"{name} quantization"),
    }


def _service_bytes(name: str, *, fallback: int) -> int:
    values = _api("/api/ps").get("models")
    if not isinstance(values, list):
        raise Refused("Ollama resident-model inventory is malformed")
    for row in values:
        if isinstance(row, dict) and row.get("name") == name:
            observed = [fallback]
            for field in ("size", "size_vram"):
                value = row.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    observed.append(value)
            return max(observed)
    return fallback


def _chat(
    name: str,
    case: dict[str, Any],
    *,
    system_prompt: str,
    max_output_tokens: int = 64,
    seed_offset: int = 0,
) -> dict[str, Any]:
    started = time.monotonic()
    response = _api(
        "/api/chat",
        payload={
            "model": name,
            "stream": False,
            "keep_alive": "30m",
            # Public canary scoring is over the final answer only.  Disable
            # optional reasoning traces so every candidate receives the same
            # final-output budget; GPT-OSS ignores boolean think values.
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": case["prompt"]},
            ],
            "options": {
                "temperature": 0,
                "seed": int(case["seed"]) + seed_offset,
                "num_predict": max_output_tokens,
            },
        },
        timeout=600.0,
    )
    latency_ms = (time.monotonic() - started) * 1000.0
    message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise Refused(f"Ollama returned no textual response for public canary {case['id']}")
    answer = _normal_answer(content)
    return {
        "id": case["id"],
        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "answer": answer,
        "passed": answer == _normal_answer(str(case["answer"])),
        "latency_ms": round(latency_ms, 3),
    }


def _unload(name: str) -> None:
    try:
        _api("/api/generate", payload={"model": name, "prompt": "", "stream": False, "keep_alive": 0}, timeout=30.0)
    except Refused:
        # Unloading is a courtesy to the next candidate.  A failed unload is
        # captured implicitly by the next candidate's observed service peak.
        return


def _candidate(name: str, *, template: dict[str, Any], runtime: dict[str, str]) -> dict[str, Any]:
    tag = _tag_record(name)
    model_size = _int(tag.get("size"), label=f"{name} model size", minimum=1)
    base = _base_model(name, tag, runtime)
    pageouts_before = _pageout_bytes()
    case_results: list[dict[str, Any]] = []
    errors: list[str] = []
    service_peak = model_size
    for case_index, case in enumerate(template["case_set"]):
        try:
            case_results.append(
                _chat(
                    name,
                    case,
                    system_prompt=template["neutral_organ_prompt"],
                    max_output_tokens=int(template.get("max_output_tokens", 64)),
                )
            )
        except Refused as error:
            errors.append(str(error))
            case_results.append(
                {
                    "id": case["id"],
                    "response_sha256": None,
                    "answer": None,
                    "passed": False,
                    "latency_ms": None,
                }
            )
        if case_index == 0:
            # A single public warm-up is enough to measure the resident model
            # service.  Bodies above the frozen 24 GiB cap are recorded as
            # ineligible immediately; there is no value in queuing another
            # fifteen prompts or a width-eight probe for a body that cannot
            # be admitted.
            service_peak = _service_bytes(name, fallback=model_size)
            if service_peak > template["model_service_cap_bytes"]:
                errors.append(
                    f"shared model service footprint exceeds {template['model_service_cap_bytes']} bytes"
                )
                break
    if len(case_results) < len(template["case_set"]):
        seen = {row["id"] for row in case_results}
        for case in template["case_set"]:
            if case["id"] not in seen:
                case_results.append(
                    {
                        "id": case["id"],
                        "response_sha256": None,
                        "answer": None,
                        "passed": False,
                        "latency_ms": None,
                    }
                )
    concurrent_results: list[dict[str, Any]] = []
    if service_peak <= template["model_service_cap_bytes"]:
        concurrent_case = template["case_set"][0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=template["required_concurrent_clients"]) as pool:
            futures = [
                pool.submit(
                    _chat,
                    name,
                    concurrent_case,
                    system_prompt=template["neutral_organ_prompt"],
                    max_output_tokens=int(template.get("max_output_tokens", 64)),
                    seed_offset=index + 1000,
                )
                for index in range(template["required_concurrent_clients"])
            ]
            for future in futures:
                try:
                    concurrent_results.append(future.result())
                except Refused as error:
                    errors.append(str(error))
    pageouts_after = _pageout_bytes()
    service_peak = _service_bytes(name, fallback=model_size)
    latencies = [row["latency_ms"] for row in case_results if isinstance(row.get("latency_ms"), (int, float))]
    passed = sum(row["passed"] is True for row in case_results)
    total = len(case_results)
    width_eight = {
        "requests": template["required_concurrent_clients"],
        "completed": len(concurrent_results),
        "all_responses_valid": len(concurrent_results) == template["required_concurrent_clients"]
        and all(isinstance(row.get("answer"), str) and bool(row["answer"]) for row in concurrent_results),
    }
    eligible = (
        not errors
        and width_eight["all_responses_valid"] is True
        and service_peak <= template["model_service_cap_bytes"]
        and pageouts_after - pageouts_before == 0
        and len(latencies) == total
    )
    _unload(name)
    return {
        "base_model": base,
        "model_size_bytes": model_size,
        "service_peak_bytes": service_peak,
        "swap_pageout_delta_bytes": pageouts_after - pageouts_before,
        "width_eight": width_eight,
        "canary": {
            "total": total,
            "passed": passed,
            "median_latency_ms": round(float(statistics.median(latencies)), 3) if latencies else None,
            "case_results": case_results,
        },
        "errors": errors,
        "eligible": eligible,
    }


def _winner(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in candidates if row.get("eligible") is True]
    if not eligible:
        return None

    def key(row: dict[str, Any]) -> tuple[Fraction, int, float, str]:
        canary = row["canary"]
        total = int(canary["total"])
        passed = int(canary["passed"])
        latency = canary["median_latency_ms"]
        if not isinstance(latency, (int, float)):
            raise Refused("eligible public-canary candidate lacks median latency")
        base = row["base_model"]
        return (-Fraction(passed, total), int(row["service_peak_bytes"]), float(latency), base["weight_sha256"])

    return sorted(eligible, key=key)[0]


def inventory(root: Path, models: list[str] | None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    frozen, template = _template(root)
    names = models or list(template["candidate_aliases"])
    runtime = _runtime()
    rows: list[dict[str, Any]] = []
    for name in names:
        try:
            tag = _tag_record(name)
            rows.append({"name": name, "present": True, "base_model": _base_model(name, tag, runtime), "size_bytes": tag.get("size")})
        except Refused as error:
            rows.append({"name": name, "present": False, "reason": str(error)})
    return {
        "schema": "SUBSTRATE_ODYSSEY_PUBLIC_MODEL_INVENTORY/v1",
        "program": PROGRAM,
        "activation": False,
        "external_activation": False,
        "scientific_evidence": False,
        "frozen_build_sha256": frozen["sha256"],
        "canary_template_sha256": file_digest(root / TEMPLATE),
        "runtime": runtime,
        "models": rows,
    }


def run(root: Path, output_path: Path, models: list[str] | None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    output_path = (root / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
    if not _inside(root, output_path):
        raise Refused("public model-canary receipt must stay inside the repository root")
    frozen, template = _template(root)
    frozen_names = list(template["candidate_aliases"])
    names = models or frozen_names
    # The receipt chooses only after every predeclared public body is measured.
    # A subset could make a weaker model win simply because a stronger staged
    # candidate was omitted, so allow no caller-defined cohort at run time.
    if names != frozen_names:
        raise Refused("public model-canary run requires the exact frozen candidate order")
    runtime = _runtime()
    candidates: list[dict[str, Any]] = []
    for name in names:
        try:
            candidates.append(_candidate(name, template=template, runtime=runtime))
        except Refused as error:
            candidates.append(
                {
                    "base_model": {
                        "id": name,
                        "revision": "unavailable",
                        "weight_sha256": "0" * 64,
                        "tokenizer_sha256": "0" * 64,
                        "runtime_sha256": runtime["sha256"],
                        "quantization": "unavailable",
                    },
                    "model_size_bytes": 0,
                    "service_peak_bytes": 0,
                    "swap_pageout_delta_bytes": 0,
                    "width_eight": {"requests": template["required_concurrent_clients"], "completed": 0, "all_responses_valid": False},
                    "canary": {"total": len(template["case_set"]), "passed": 0, "median_latency_ms": None, "case_results": []},
                    "errors": [str(error)],
                    "eligible": False,
                }
            )
    selected = _winner(candidates)
    checks = {
        "frozen_template_bound": True,
        "public_only": template["visibility"] == "public_only",
        "all_candidates_accounted": len(candidates) == len(names),
        "all_configured_candidates_eligible": len(candidates) == len(names)
        and all(candidate["eligible"] is True for candidate in candidates),
        "no_hidden_seed_commitments": True,
        "selection_rule_applied": selected is not None,
        "selected_candidate_eligible": selected is not None and selected["eligible"] is True,
        "shared_service_footprint_within_24_gib": selected is not None
        and selected["service_peak_bytes"] <= template["model_service_cap_bytes"],
        "no_swap": selected is not None and selected["swap_pageout_delta_bytes"] == 0,
        "width_eight_admitted": selected is not None and selected["width_eight"]["all_responses_valid"] is True,
    }
    # The frozen rule excludes bodies that fail the resource screen; it does
    # not require every predeclared body to be eligible.  Keep the historical
    # all_configured_candidates_eligible diagnostic for auditability, but do
    # not let a correctly excluded body block selection.  Legacy test fixtures
    # without the frozen output-budget field retain their original all-pass
    # semantics.
    pass_checks = [value for name, value in checks.items() if name != "all_configured_candidates_eligible"]
    all_pass = all(pass_checks) if "max_output_tokens" in template else all(checks.values())
    organ_prompt = template.get(
        "neutral_organ_prompt",
        "This is a public deterministic model canary. Reply with exactly one final line in the form FINAL: <answer>. Do not use tools.",
    )
    body = {
        "schema": SCHEMA,
        "program": PROGRAM,
        "status": "pass" if all_pass else "fail",
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
        "scientific_evidence": False,
        "evidence_scope": "frozen_public_model_selection_canaries_only",
        "completed_at": datetime.now(UTC).isoformat(),
        "frozen_build_sha256": frozen["sha256"],
        "canary_template_sha256": file_digest(root / TEMPLATE),
        "runtime": runtime,
        "model_service_cap_bytes": template["model_service_cap_bytes"],
        "required_concurrent_clients": template["required_concurrent_clients"],
        "selection_rule": template["selection_rule"],
        "neutral_organ_prompt_sha256": hashlib.sha256(organ_prompt.encode("utf-8")).hexdigest(),
        "reasoning_effort_policy": template.get("reasoning_effort_policy", "legacy_fixture"),
        "conversation_policy": template.get("conversation_policy", "legacy_fixture"),
        "max_output_tokens": int(template.get("max_output_tokens", 64)),
        "hidden_seed_commitments_materialized": False,
        "candidates": candidates,
        "selected_base_model": selected["base_model"] if selected is not None else None,
        "checks": checks,
        "all_pass": all_pass,
        "non_claims": [
            "No hidden task seed, answer, scorer, or evaluator material was read or created.",
            "This receipt selects only a candidate base-model body under the frozen public rule.",
            "This receipt does not select a treatment arm, control arm, corpus, custodian, or scientific outcome.",
            "This command never seals G02 and never starts an Odyssey worker.",
        ],
    }
    body["sha256"] = digest(body)
    _write_json(output_path, body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory(args.root, args.models)
        else:
            if args.out is None:
                raise Refused("run requires --out")
            result = run(args.root, args.out, args.models)
    except Refused as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("all_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
