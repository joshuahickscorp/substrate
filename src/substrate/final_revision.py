"""Single command family for the Substrate final revision."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from substrate import final_revision_campaign as campaign
from substrate import final_revision_field_campaign as field_campaign
from substrate import final_revision_grok as grok
from substrate import final_revision_io as io


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _exit(value: dict[str, Any], passed: bool) -> None:
    _print(value)
    raise SystemExit(0 if passed else 1)


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    publish = "--no-publish" not in arguments
    arguments = [argument for argument in arguments if argument != "--no-publish"]
    command = arguments.pop(0) if arguments else "status"
    if command == "preflight":
        report = campaign.preflight(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "research":
        report = campaign.research(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "grok-review":
        report = campaign.grok_review(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "record-grok-build":
        if len(arguments) != 2:
            raise SystemExit("record-grok-build requires TASK_DIRECTORY CONTRACT_PATH")
        record = grok.grok_build_record(Path(arguments[0]), Path(arguments[1]))
        ledger = campaign.record_grok_invocation(record)
        report = campaign.grok_review(publish=publish)
        _print(
            {
                "recorded": record["invocation_id"],
                "role": record["role"],
                "round": record["round"],
                "output_digest": record["output_digest"],
                "validated_output_count": report["ledger"]["validated_output_count"],
                "ledger_digest": ledger["sha256"],
                "activation": False,
            }
        )
        return
    if command == "record-grok-build-rejected":
        if len(arguments) != 2:
            raise SystemExit("record-grok-build-rejected requires TASK_DIRECTORY CONTRACT_PATH")
        record = grok.grok_build_rejected_record(Path(arguments[0]), Path(arguments[1]))
        ledger = campaign.record_grok_rejected_attempt(record)
        report = campaign.grok_review(publish=publish)
        _print(
            {
                "recorded_rejected_attempt": record["invocation_id"],
                "role": record["role"],
                "round": record["round"],
                "rejection_reason": record["rejection_reason"],
                "validated_output_count": report["ledger"]["validated_output_count"],
                "ledger_digest": ledger["sha256"],
                "activation": False,
            }
        )
        return
    if command == "resolve-grok-review":
        if len(arguments) != 1:
            raise SystemExit("resolve-grok-review requires RESOLUTION_BATCH_JSON")
        resolution_batch = json.loads(Path(arguments[0]).read_text())
        result = campaign.resolve_grok_blockers(resolution_batch)
        report = campaign.grok_review(publish=publish)
        _print(
            {
                **result,
                "validated_output_count": report["ledger"]["validated_output_count"],
                "activation": False,
            }
        )
        return
    if command == "adjudicate-grok-blockers":
        if len(arguments) != 1:
            raise SystemExit("adjudicate-grok-blockers requires RESOLUTION_COMMIT")
        result = campaign.adjudicate_grok_blockers(arguments[0])
        if result["all_pass"]:
            report = campaign.grok_review(publish=publish)
            result["validated_output_count"] = report["ledger"]["validated_output_count"]
        _exit(result, bool(result["all_pass"]))
    if command == "reproduce-null":
        report = campaign.reproduce_null(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "tournament":
        report = campaign.tournament(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "acquire":
        report = campaign.acquire(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "canaries":
        report = campaign.canaries(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "pilot":
        report = campaign.pilot(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "field-scaffold":
        report = field_campaign.publish_scaffold()
        _exit(report, bool(report["mechanisms_pass"]))
    if command == "field-foundation":
        report = field_campaign.publish_scaffold()
        _exit(report, bool(report["all_pass"]))
    if command == "field-status":
        _print(field_campaign.status())
        return
    if command == "field-grok-contracts":
        if len(arguments) != 1:
            raise SystemExit("field-grok-contracts requires EVIDENCE_COMMIT")
        _print(field_campaign.write_grok_contracts(arguments[0]))
        return
    if command == "record-field-grok-build":
        if len(arguments) != 2:
            raise SystemExit("record-field-grok-build requires TASK_DIRECTORY CONTRACT_PATH")
        report = field_campaign.record_grok_build(Path(arguments[0]), Path(arguments[1]))
        _print(
            {
                "recorded": report["credited_invocations"][-1]["invocation_id"],
                "role": report["credited_invocations"][-1]["role"],
                "distinct_credited_role_count": report["distinct_credited_role_count"],
                "complete": report["complete"],
                "activation": False,
            }
        )
        return
    if command == "record-field-grok-build-rejected":
        if len(arguments) != 2:
            raise SystemExit("record-field-grok-build-rejected requires TASK_DIRECTORY CONTRACT_PATH")
        report = field_campaign.record_grok_build_rejected(Path(arguments[0]), Path(arguments[1]))
        _print(
            {
                "recorded_rejected_attempt": report["rejected_uncredited_invocations"][-1]["invocation_id"],
                "role": report["rejected_uncredited_invocations"][-1]["role"],
                "rejection_reason": report["rejected_uncredited_invocations"][-1]["rejection_reason"],
                "complete": report["complete"],
                "activation": False,
            }
        )
        return
    if command == "record-field-grok-failed-launch":
        if len(arguments) != 2:
            raise SystemExit("record-field-grok-failed-launch requires TASK_DIRECTORY CONTRACT_PATH")
        report = field_campaign.record_grok_failed_launch(Path(arguments[0]), Path(arguments[1]))
        _print(
            {
                "recorded_failed_launch": report["rejected_uncredited_invocations"][-1]["invocation_id"],
                "role": report["rejected_uncredited_invocations"][-1]["role"],
                "rejection_reason": report["rejected_uncredited_invocations"][-1]["rejection_reason"],
                "complete": report["complete"],
                "activation": False,
            }
        )
        return
    if command == "adjudicate-field-grok":
        if len(arguments) != 1:
            raise SystemExit("adjudicate-field-grok requires RESOLUTIONS_JSON")
        resolutions = json.loads(Path(arguments[0]).read_text())
        if not isinstance(resolutions, dict):
            raise SystemExit("field Grok resolutions must be a JSON object")
        report = field_campaign.adjudicate_grok_recommendations(resolutions)
        _print(
            {
                "disposition_count": len(report["dispositions"]),
                "complete": report["complete"],
                "activation": False,
            }
        )
        return
    if command == "freeze":
        report = campaign.freeze(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "run":
        report = campaign.run(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "status":
        _print(campaign.status())
        return
    if command == "stop":
        _print({"stopped": True, "stop_switch": str(io.stop()), "activation": False})
        return
    if command == "resume":
        io.resume()
        _print({"resumed": True, "status": campaign.status(), "activation": False})
        return
    if command == "verify":
        report = campaign.verify(publish=publish)
        _exit(report, bool(report["all_pass"]))
    if command == "publish":
        report = campaign.publish(publish_files=publish)
        _exit(report, bool(report["all_pass"]))
    raise SystemExit(f"unknown final-revision command {command!r}")
