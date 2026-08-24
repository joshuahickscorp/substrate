"""Provenance-aware acquisition planning without transport or downloader execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from substrate.product.codec import sha256
from substrate.product.contracts import (
    PRODUCT_SCHEMA_VERSION,
    ProductRefused,
    SourcePolicy,
    SourceReceipt,
    SourceRequest,
    _parse_source_uri,
)

_ACQUISITION_PLAN_FIELDS = frozenset(
    {
        "adapter",
        "execution_permitted",
        "execution_refusal",
        "plan_sha256",
        "required_receipt_fields",
        "schema_version",
        "source_policy_sha256",
        "source_request",
    }
)
_REQUIRED_RECEIPT_FIELDS = (
    "request.source_request_sha256",
    "request.source_origin",
    "request.access_status",
    "request.declared_rights_sha256",
    "received_at",
    "retrieval_method",
    "content_sha256",
    "acquisition_plan_sha256",
    "processing_history_sha256",
)


def _adapter_for(request: SourceRequest) -> str:
    """Name the future broker interface without making a network call."""

    _require_source_request(request)
    scheme = _parse_source_uri(request.source_uri).scheme.lower()
    if scheme == "file":
        return "operator-import"
    return "brokered-source-adapter"


def _require_source_request(request: SourceRequest) -> None:
    """Keep direct internal adapter selection fail-closed too."""

    if not isinstance(request, SourceRequest):
        raise ProductRefused("source request is malformed")


def _require_source_policy(source_policy: SourcePolicy) -> None:
    """Refuse malformed public planner arguments before dereferencing them."""

    if not isinstance(source_policy, SourcePolicy):
        raise ProductRefused("source policy is malformed")


def plan_acquisition(request: SourceRequest, source_policy: SourcePolicy) -> dict[str, Any]:
    """Produce a hashable acquisition plan after applying the source policy.

    The return value is deliberately insufficient to execute. A trusted future
    broker must interpret it, enforce egress/mount limits, and return a receipt.
    """

    _require_source_request(request)
    _require_source_policy(source_policy)
    source_policy.assert_permits(request)
    plan = {
        "adapter": _adapter_for(request),
        "execution_permitted": False,
        "execution_refusal": "no source adapter backend is configured",
        "required_receipt_fields": list(_REQUIRED_RECEIPT_FIELDS),
        "schema_version": PRODUCT_SCHEMA_VERSION,
        # An entity ledger may retain this plan.  Preserve a binding to the
        # live request and policy, not their raw URL/path/rights text.
        "source_policy_sha256": sha256(source_policy.to_dict()),
        "source_request": request.to_provenance_dict(),
    }
    plan["plan_sha256"] = sha256(plan)
    return plan


def validate_acquisition_plan(
    plan: Mapping[str, Any],
    source_policy: SourcePolicy,
    *,
    request: SourceRequest,
) -> dict[str, Any]:
    """Validate one sealed, non-executing plan against the active policy.

    A ledger hash protects stored rows from accidental alteration, but this
    additional validation prevents a caller from treating a look-alike row as
    an authorized acquisition plan during assimilation.
    """

    _require_source_request(request)
    _require_source_policy(source_policy)
    if not isinstance(plan, Mapping) or set(plan) != _ACQUISITION_PLAN_FIELDS:
        raise ProductRefused("source acquisition plan is malformed")
    canonical_plan = dict(plan)
    plan_sha256 = canonical_plan.pop("plan_sha256", None)
    if not isinstance(plan_sha256, str) or plan_sha256 != sha256(canonical_plan):
        raise ProductRefused("source acquisition plan digest does not match its contents")
    if canonical_plan.get("schema_version") != PRODUCT_SCHEMA_VERSION:
        raise ProductRefused("source acquisition plan has an unsupported schema version")
    if canonical_plan.get("execution_permitted") is not False:
        raise ProductRefused("source acquisition plan cannot permit execution")
    if canonical_plan.get("execution_refusal") != "no source adapter backend is configured":
        raise ProductRefused("source acquisition plan has an invalid execution refusal")
    if canonical_plan.get("adapter") != _adapter_for(request):
        raise ProductRefused("source acquisition plan adapter does not match its source request")
    if canonical_plan.get("required_receipt_fields") != list(_REQUIRED_RECEIPT_FIELDS):
        raise ProductRefused("source acquisition plan has invalid required receipt fields")
    if canonical_plan.get("source_request") != request.to_provenance_dict():
        raise ProductRefused("source acquisition plan does not bind its source request")
    if canonical_plan.get("source_policy_sha256") != sha256(source_policy.to_dict()):
        raise ProductRefused("source acquisition plan does not bind the active source policy")
    return dict(plan)


def validate_receipt(receipt: SourceReceipt, source_policy: SourcePolicy) -> dict[str, Any]:
    """Return a validated receipt envelope suitable for controlled assimilation."""

    if not isinstance(receipt, SourceReceipt):
        raise ProductRefused("source receipt is malformed")
    _require_source_policy(source_policy)
    source_policy.assert_permits(receipt.request)
    if receipt.retrieval_method != _adapter_for(receipt.request):
        raise ProductRefused("source receipt retrieval method does not match its declared source adapter")
    envelope = {"receipt": receipt.to_provenance_dict(), "schema_version": PRODUCT_SCHEMA_VERSION, "structurally_valid": True}
    envelope["receipt_sha256"] = sha256(envelope)
    return envelope
