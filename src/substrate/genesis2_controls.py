"""Genesis II representation controls.

The exact arm is the inherited S2 implementation.  The low-bit arm is a
separate subclass whose transition, proposal, retrieval, and write opportunity
are identical; only the representation committed for an associative value is
changed.  That is the controlled contrast required by the representation–
architecture factorial.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from substrate import genesis2_config as C2
from substrate import genesis2_microstore as MS2
from substrate import genesis_controls as G1
from substrate import genesis_k_structural as G1S
from substrate.genesis_material import Opportunity, Proposal, register

_ACTIVATION = False


@dataclass
class S2LowBitConstrained(G1.S2TaskIndependentMonolithicPersistentCore):
    """S2 with its associative payloads represented in the ternary alphabet."""

    def _propose(self) -> Iterable[Proposal]:
        for proposal in super()._propose():
            low_value = tuple(MS2.quantize(value, "ternary") for value in proposal.delta)
            current = self.table.get(proposal.target)
            if current is not None and tuple(current["value"]) == low_value:
                continue
            yield replace(
                proposal,
                proposal_id=f"s2-low-bit:{proposal.proposal_id}",
                delta=low_value,
                precision_request="ternary",
                cost_bytes=MS2.entry_bytes(low_value, "ternary"),
            )

    def _durable_state(self) -> Any:
        state = dict(super()._durable_state())
        state["form"] = "monolithic_deterministic_state_machine_low_bit"
        state["representation"] = "ternary"
        state["activation"] = _ACTIVATION
        return state


def _build_low_bit(opportunity: Opportunity, **_options: Any) -> S2LowBitConstrained:
    return S2LowBitConstrained(
        name=C2.S2_LOW_BIT_ID,
        mechanism="monolithic_deterministic_task_independent_persistent_core_low_bit",
        _opportunity=opportunity,
    )


register(C2.S2_LOW_BIT_ID, _build_low_bit)


def _build_prior_field(opportunity: Opportunity, **_options: Any) -> G1S.K8_event_sourced_plastic_field:
    """Carry the selected Genesis I material forward without changing its law."""
    material = G1S.K8_event_sourced_plastic_field(
        name="L0_prior_selected_field",
        mechanism=G1S.MECH_K8,
        _opportunity=opportunity,
    )
    material._sync_resident()
    return material


register("L0_prior_selected_field", _build_prior_field)


def demo() -> None:
    from substrate.genesis_material import Observation, Verdict, equal_opportunity

    observations = [Observation(0, "scope", (1, 7), teaching=True)]
    opportunity = equal_opportunity(
        envelope="1GB",
        observations=observations,
        sensor_channels=("scope",),
        operation_budget=1_000,
        durable_write_budget=100,
    )
    material = _build_low_bit(opportunity)
    material.observe(observations[0])
    proposals = material.propose()
    assert proposals and all(value in (-1, 0, 1) for proposal in proposals for value in proposal.delta)
    material.apply([Verdict(proposal.proposal_id, True, 1.0, 1.0) for proposal in proposals])
    assert all(value in (-1, 0, 1) for row in material.table.values() for value in row["value"])
    assert not material.propose(), "a quantized value was proposed again against its unquantized staging value"
    print("genesis2 S2 representation control self-check passed")


if __name__ == "__main__":
    demo()
