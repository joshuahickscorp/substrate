"""Focused contract tests for the production Odyssey candidate/control arms."""

from __future__ import annotations

import inspect
import json
import math
import sqlite3
from pathlib import Path

import pytest

from substrate import odyssey_arms as arms


def _request(root: Path, *, role: str, cycle: int = 0, task: dict | None = None) -> tuple[Path, dict]:
    task = task or {
        "schema": "SUBSTRATE_ODYSSEY_TEST_TASK/v1",
        "activation": False,
        "program": arms.PROGRAM,
        "frontier": "A",
        "task_id": f"A-{cycle:04d}",
        "request": "Summarize the visible project note.",
        "required_receipt": ["summary"],
    }
    path = root / "runs/substrate/odyssey7d/v1/arms/A" / role / "requests" / f"{cycle:03d}-retrieval.json"
    receipt = root / "runs/substrate/odyssey7d/v1/arms/A" / role / "receipts" / f"{cycle:03d}-retrieval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "schema": arms.REQUEST_SCHEMA,
        "activation": False,
        "authority_sha256": "a" * 64,
        "run_id": "odyssey-arms-fixture",
        "frontier": "A",
        "role": role,
        "cycle": cycle,
        "phase": "retrieval",
        "task": task,
        "candidate_manifest_sha256": "b" * 64,
        "receipt_path": str(receipt.relative_to(root)),
    }
    request["request_sha256"] = arms.digest(request)
    path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    return path, request


def _self_sha() -> str:
    return arms.file_digest(Path(arms.__file__).resolve())


def _fake_model(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_chat(**kwargs: object) -> tuple[dict, dict]:
        calls.append(dict(kwargs))
        return (
            {
                "summary": "visible-only answer",
                "memory_updates": [{"kind": "fact", "key": "visible-note", "value": "remembered", "confidence": 0.8}],
                "unfinished": ["follow up"],
                "competence": {"summarization": "observed"},
            },
            {
                "prompt_eval_count": 5,
                "eval_count": 7,
                "total_duration_ns": 12,
                "load_duration_ns": 2,
                "eval_duration_ns": 10,
                "generation_num_predict": arms.GENERATION_NUM_PREDICT,
                "substantive_max_tokens": arms.MAX_OUTPUT_TOKENS,
                "transport_gate": "pass",
            },
        )

    monkeypatch.setattr(arms, "_ollama_chat", fake_chat)
    return calls


def test_candidate_is_mutable_associative_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_model(monkeypatch)
    request_path, _request_body = _request(tmp_path, role="candidate")
    state_root = "runs/substrate/odyssey7d/v1/private-state/A/candidate"

    first = arms.run(
        tmp_path,
        role="candidate",
        model="gpt-oss:20b",
        state_root=state_root,
        self_sha256=_self_sha(),
        request_path=request_path,
    )
    second = arms.run(
        tmp_path,
        role="candidate",
        model="gpt-oss:20b",
        state_root=state_root,
        self_sha256=_self_sha(),
        request_path=request_path,
    )

    assert len(calls) == 1
    assert first == second
    assert first["state_change"]["mode"] == "flat_exact_associative_monolith"
    assert first["state_change"]["associations_written"] >= 3
    assert first["state_before_sha256"] != first["state_after_sha256"]
    output_path = tmp_path / first["output_artifacts"][0]["path"]
    assert output_path.is_file()
    assert first["output_artifacts"][0]["sha256"] == arms.file_digest(output_path)
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert first["response_sha256"] == output["sha256"]
    database = tmp_path / state_root / "state.sqlite3"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM associations").fetchone()[0] >= 3


def test_control_is_append_only_and_cannot_share_candidate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_model(monkeypatch)
    candidate_request, _ = _request(tmp_path, role="candidate")
    shared_root = "runs/substrate/odyssey7d/v1/private-state/A/shared"
    arms.run(
        tmp_path,
        role="candidate",
        model="gpt-oss:20b",
        state_root=shared_root,
        self_sha256=_self_sha(),
        request_path=candidate_request,
    )
    control_request, _ = _request(tmp_path, role="control")
    with pytest.raises(arms.Refused, match="metadata drifted: role"):
        arms.run(
            tmp_path,
            role="control",
            model="gpt-oss:20b",
            state_root=shared_root,
            self_sha256=_self_sha(),
            request_path=control_request,
        )

    control_root = "runs/substrate/odyssey7d/v1/private-state/A/control"
    receipt = arms.run(
        tmp_path,
        role="control",
        model="gpt-oss:20b",
        state_root=control_root,
        self_sha256=_self_sha(),
        request_path=control_request,
    )

    assert len(calls) == 2
    assert arms.MAX_OUTPUT_TOKENS == 64
    assert arms.GENERATION_NUM_PREDICT == 1024
    assert arms.GENERATION_NUM_PREDICT > arms.MAX_OUTPUT_TOKENS
    assert arms.ARM_TRANSPORT_ATTEMPTS == 1
    assert calls[0]["seed"] == calls[1]["seed"]
    assert calls[0]["messages"] == calls[1]["messages"]
    assert calls[0]["timeout_seconds"] == calls[1]["timeout_seconds"] == arms.ARM_REQUEST_TIMEOUT_SECONDS
    assert receipt["state_change"] == {
        "mode": "append_only_history_retrieval",
        "associations_written": 0,
        "association_revisions": 0,
        "history_events_appended": 1,
    }
    with sqlite3.connect(tmp_path / control_root / "state.sqlite3") as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "history" in tables
        assert "associations" not in tables


def test_request_rejects_evaluator_surface_and_adapter_drift(tmp_path: Path) -> None:
    task = {
        "schema": "SUBSTRATE_ODYSSEY_TEST_TASK/v1",
        "activation": False,
        "program": arms.PROGRAM,
        "frontier": "A",
        "task_id": "A-0000",
        "evaluator_key": "must not be visible",
        "required_receipt": ["summary"],
    }
    request_path, _ = _request(tmp_path, role="candidate", task=task)

    with pytest.raises(arms.Refused, match="evaluator-only"):
        arms.validate_request(tmp_path, request_path, expected_role="candidate")
    with pytest.raises(arms.Refused, match="source drifted"):
        arms.run(
            tmp_path,
            role="candidate",
            model="gpt-oss:20b",
            state_root="runs/substrate/odyssey7d/v1/private-state/A/candidate",
            self_sha256="0" * 64,
            request_path=request_path,
        )


# ---------------------------------------------------------------------------
# Protocol v2 — transport vs semantic gates and adversarial content
# ---------------------------------------------------------------------------


def test_generation_budget_is_transport_only_and_shared() -> None:
    options = arms._generation_options(seed=7)
    assert options == {"temperature": 0, "seed": 7, "num_predict": 1024}
    assert arms.MAX_OUTPUT_TOKENS == 64
    assert options["num_predict"] == arms.GENERATION_NUM_PREDICT
    assert options["num_predict"] != arms.MAX_OUTPUT_TOKENS


def test_transport_strips_markdown_fence_only() -> None:
    body = '```json\n{"summary": "fenced"}\n```'
    parsed, cleaned, fence_stripped = arms._assert_transport_content(body)
    assert fence_stripped is True
    assert parsed == {"summary": "fenced"}
    assert cleaned == '{"summary": "fenced"}'


@pytest.mark.parametrize(
    "content",
    [
        'Leading analysis\n{"summary": "x"}',
        '{"summary": "x"}\ntrailing prose',
        "",
        "   ",
        '{"summary": "x", "summary": "y"}',  # duplicate keys — second wins under naive parse; we refuse
        '{"summary": NaN}',
        '{"summary": Infinity}',
        '{"summary": -Infinity}',
        "<|channel|>final<|message|>" + '{"summary": "x"}',
        "<think>secret</think>" + '{"summary": "x"}',
        "not json at all",
        '{"summary": "truncated',
        '["not", "an", "object"]',
    ],
)
def test_transport_rejects_adversarial_payloads(content: str) -> None:
    with pytest.raises(arms.Refused, match="transport:"):
        arms._assert_transport_content(content)


def test_transport_accepts_escaped_quotes_and_unicode() -> None:
    content = '{"summary": "he said \\"hello\\" — café 测试"}'
    parsed, _cleaned, fence_stripped = arms._assert_transport_content(content)
    assert fence_stripped is False
    assert parsed["summary"] == 'he said "hello" — café 测试'


def test_semantic_rejects_missing_null_empty_and_oversize() -> None:
    with pytest.raises(arms.Refused, match="semantic:.*missing"):
        arms._validate_semantic_response({"other": 1}, ["summary"])
    with pytest.raises(arms.Refused, match="semantic:.*null"):
        arms._validate_semantic_response({"summary": None}, ["summary"])
    with pytest.raises(arms.Refused, match="semantic:.*empty"):
        arms._validate_semantic_response({"summary": "   "}, ["summary"])
    with pytest.raises(arms.Refused, match="semantic:.*non-finite"):
        arms._validate_semantic_response({"summary": math.nan}, ["summary"])
    huge = {"summary": "word " * 80}
    with pytest.raises(arms.Refused, match="semantic:.*substantive"):
        arms._validate_semantic_response(huge, ["summary"])


def test_semantic_accepts_bounded_model_answer() -> None:
    count = arms._validate_semantic_response({"summary": "short visible answer"}, ["summary"])
    assert 1 <= count <= arms.MAX_OUTPUT_TOKENS


def test_transport_repair_never_fills_missing_fields() -> None:
    """Fence strip is allowed; inventing required fields is not."""
    parsed, _, _ = arms._assert_transport_content('```\n{"note": "only"}\n```')
    with pytest.raises(arms.Refused, match="semantic:.*missing"):
        arms._validate_semantic_response(parsed, ["summary"])


def test_run_records_semantic_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_model(monkeypatch)
    request_path, _ = _request(tmp_path, role="candidate")
    receipt = arms.run(
        tmp_path,
        role="candidate",
        model="gpt-oss:20b",
        state_root="runs/substrate/odyssey7d/v1/private-state/A/candidate-telemetry",
        self_sha256=_self_sha(),
        request_path=request_path,
    )
    usage = receipt["resource_usage"]
    assert usage["semantic_gate"] == "pass"
    assert usage["substantive_token_count"] >= 1
    assert usage["substantive_max_tokens"] == 64
    assert usage["generation_num_predict"] == 1024


def test_run_refuses_semantic_failure_without_substituting_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def empty_answer_chat(**_kwargs: object) -> tuple[dict, dict]:
        return {"summary": ""}, {"transport_gate": "pass", "generation_num_predict": 1024}

    monkeypatch.setattr(arms, "_ollama_chat", empty_answer_chat)
    request_path, _ = _request(tmp_path, role="candidate")
    with pytest.raises(arms.Refused, match="semantic:"):
        arms.run(
            tmp_path,
            role="candidate",
            model="gpt-oss:20b",
            state_root="runs/substrate/odyssey7d/v1/private-state/A/candidate-empty",
            self_sha256=_self_sha(),
            request_path=request_path,
        )
    # No receipt may be minted with an adapter-fabricated answer.
    receipt_path = tmp_path / "runs/substrate/odyssey7d/v1/arms/A/candidate/receipts/000-retrieval.json"
    assert not receipt_path.exists()


# ---------------------------------------------------------------------------
# Injection tests — adapter must not manufacture a pass
# ---------------------------------------------------------------------------


def test_injection_adapter_source_has_no_gold_or_evaluator_substitution() -> None:
    source = Path(arms.__file__).read_text(encoding="utf-8")
    forbidden = (
        "gold_answer",
        "GOLD_ANSWER",
        "evaluator_answer",
        "copy_candidate_to_control",
        "fabricate_evidence",
        "substitute_answer",
        "scorer_answer",
    )
    for token in forbidden:
        assert token not in source, f"adapter source contains injection affordance {token!r}"
    # Thinking must never be used as the answer body.
    assert 'message["thinking"]' not in source or "thinking_char_count" in source
    assert "thinking" in source  # telemetry allowed
    assert 'message.get("content")' in source or 'message["content"]' in source


def test_injection_semantic_gate_does_not_default_required_fields() -> None:
    source = inspect.getsource(arms._validate_semantic_response)
    assert "setdefault" not in source
    assert "get(" not in source or "response[field]" in source
    # Explicitly index required fields; no silent defaults.
    assert "response[field]" in source


def test_injection_arms_share_identical_budget_constants() -> None:
    """Unequal retry/timeout/predict between arms is a parity violation."""
    source = inspect.getsource(arms._ollama_chat)
    assert "GENERATION_NUM_PREDICT" in source
    assert "ARM_REQUEST_TIMEOUT_SECONDS" in source
    assert "ARM_TRANSPORT_ATTEMPTS" in source
    # No role parameter and no role-conditional branching in the HTTP seam.
    signature = inspect.signature(arms._ollama_chat)
    assert "role" not in signature.parameters
    assert "if role" not in source
    assert 'role == "candidate"' not in source
    assert 'role == "control"' not in source


def test_injection_retry_constant_forbids_task_leaking_retry_ladder() -> None:
    assert arms.ARM_TRANSPORT_ATTEMPTS == 1
    source = Path(arms.__file__).read_text(encoding="utf-8")
    assert "for attempt" not in source
    assert "retry_prompt" not in source
    assert "with_hint" not in source


def test_paired_seed_identical_for_roles(tmp_path: Path) -> None:
    cand_path, cand = _request(tmp_path, role="candidate")
    ctrl_path, ctrl = _request(tmp_path, role="control")
    # Same pairing identity aside from role field still yields same seed because
    # seed is derived from authority/run/frontier/cycle/phase/task/manifest only.
    assert arms._paired_seed(cand) == arms._paired_seed(ctrl)
    assert cand_path != ctrl_path
