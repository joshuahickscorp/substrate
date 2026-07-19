"""Seal the broad-campaign launch manifest and the Horizon 2 terminal transition receipt.

The launch manifest binds the exact commit, the campaign schema and digest, node specifications, resource
classes, the dynamic throttling policy, the run root, Telegram configuration identity (service names only,
no secrets), and the rollback and restart contract. The transition receipt independently audits the
Horizon 2 boundary: if Horizon 2 is not terminal, it records that the legacy General Run drain is DEFERRED
(gated on real completion), never skipped, and states exactly what the new campaign does and does not
inherit. No scientific score is read here.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from mop.campaign.manifest import build_campaign
from mop.campaign.specs import ResourceClass
from mop.substrate.events import canonical_bytes, canonical_sha256

REPO = Path(__file__).resolve().parents[1]
OP_ROOT = REPO / "proof" / "campaign_run"
HORIZON_STATE = REPO / "runs/generation1/generation1-successor-horizon-v2/current_status.json"
HORIZON_PROGRAM = REPO / "runs/generation1/generation1-successor-horizon-v2/program_state.json"


def _git_head() -> str:
    return (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        or "unknown"
    )


def _count_live(markers: tuple[str, ...]) -> int:
    try:
        import psutil  # type: ignore
    except Exception:
        return -1
    n = 0
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            hay = " ".join([proc.info.get("name") or "", " ".join(proc.info.get("cmdline") or [])])
            if any(m in hay for m in markers):
                n += 1
        except Exception:
            continue
    return n


def build_launch_manifest() -> dict[str, Any]:
    camp = build_campaign()
    content: dict[str, Any] = {
        "schema": "mop-launch-manifest/v1",
        "program_id": "mop-research",
        "process_label": "mop:research:orchestrator",
        "launch_commit": _git_head(),
        "campaign_id": camp.campaign_id,
        "campaign_schema": camp.schema,
        "campaign_digest": camp.digest(),
        "n_nodes": len(camp.nodes),
        "n_runnable_local": len([n for n in camp.nodes if not n.is_blocked]),
        "n_contracted_external": len([n for n in camp.nodes if n.is_blocked]),
        "node_ids": [n.node_id for n in camp.nodes],
        "resource_classes": [rc.value for rc in ResourceClass],
        "worker_ceiling": 20,
        "throttle_policy": {
            "controller": "mop.studio.dynamic_worker_controller",
            "measured_hash_heavy_optimum_workers": 20,
            "hawking_reserve": "shed worker count + gentlest QoS; never signal Hawking",
            "admission_sequence": "staged 1->4->8->16->20 under live checks",
            "native_thread_pinning": [
                "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "VECLIB_MAXIMUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
            ],
        },
        "run_root": "runs/campaign/mop_research",
        "op_root": "proof/campaign_run",
        "external_dependencies": [e.payload() for e in camp.external_dependencies],
        "telegram_identity": {
            "token_service": "com.hawking.doctorv5.telegram.bot-token",
            "chat_service": "com.hawking.doctorv5.telegram.chat-id",
            "secrets_stored_here": False,
        },
        "mechanism_card_versions": "mop-campaign-mechanism-cards/v1",
        "readiness_gate_version": "mop-campaign-readiness/v1",
        "producer_verifier_authority": "verifier nodes re-derive independently of producer paths",
        "rollback_contract": {
            "stop": "PYTHONPATH=src .venv/bin/python scripts/mop_research_orchestrator.py --stop",
            "restart": "PYTHONPATH=src .venv/bin/python scripts/mop_research_orchestrator.py --detach",
            "rollback_commit": _git_head(),
            "durable_state": "runs/campaign/mop_research/state.json (atomic, restart-safe, lease recovery)",
        },
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**content, "seal": {"sha256": canonical_sha256(content)}}


def build_transition_receipt() -> dict[str, Any]:
    horizon: dict[str, Any] = {}
    for path in (HORIZON_PROGRAM, HORIZON_STATE):
        try:
            horizon.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    state = horizon.get("state")
    finished = horizon.get("finished_at")
    terminal = (finished not in (None, "")) or state in ("complete", "finished", "drained")
    live_mechanics = _count_live(("mop-final-mechanic", "mop-g1-horizon", "mop-supervisor:generation1"))
    general_run_alive = _count_live(("general-run:adopter",)) > 0

    content: dict[str, Any] = {
        "schema": "mop-horizon2-transition-receipt/v1",
        "horizon_v2_state": state,
        "horizon_v2_finished_at": finished,
        "horizon_v2_terminal": bool(terminal),
        "live_horizon_worker_and_supervisor_count": live_mechanics,
        "general_run_adopter_alive": general_run_alive,
        "verdict": "horizon_2_terminal" if terminal else "horizon_2_still_running",
        "legacy_general_run_drain": "permitted" if terminal else "DEFERRED_gated_on_horizon2_terminal",
        "drain_reason": (
            "Horizon 2 is terminal; the successor boundary is satisfied and the legacy scheduler may drain"
            if terminal
            else "Horizon 2 is state=running with live workers and supervisor active; draining the legacy "
            "General Run or signaling the horizon supervisor now would destroy in-flight evidence. The drain is "
            "deferred, not skipped: it auto-permits when Horizon 2 reaches a clean terminal authority."
        ),
        "new_campaign_inherits": (
            ["the sealed Horizon 2 terminal authority as external:horizon-v2-complete"]
            if terminal
            else [
                "nothing from Horizon 2 yet; it is incomplete. The new orchestrator coexists and adopts it "
                "as an external resource consumer only."
            ]
        ),
        "new_campaign_does_not_inherit": [
            "any unsealed Horizon 2 result",
            "any authority to promote or activate",
            "ownership of the live General Run or horizon supervisor processes",
        ],
        "safety": "no live process was signaled, killed, drained, or restarted by this transition audit",
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**content, "seal": {"sha256": canonical_sha256(content)}}


def main() -> int:
    OP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_launch_manifest()
    (OP_ROOT / "LAUNCH_MANIFEST.json").write_bytes(canonical_bytes(manifest))
    receipt = build_transition_receipt()
    (OP_ROOT / "HORIZON2_TRANSITION_RECEIPT.json").write_bytes(canonical_bytes(receipt))
    print(
        "launch manifest sealed:", manifest["seal"]["sha256"][:16], "commit", manifest["launch_commit"][:12]
    )
    print("campaign digest:", manifest["campaign_digest"][:16], "nodes", manifest["n_nodes"])
    print("transition receipt:", receipt["verdict"], "| drain:", receipt["legacy_general_run_drain"])
    print(
        "  horizon_v2_terminal:",
        receipt["horizon_v2_terminal"],
        "live horizon procs:",
        receipt["live_horizon_worker_and_supervisor_count"],
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
