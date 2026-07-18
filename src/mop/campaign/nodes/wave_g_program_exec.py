"""Wave G program-execution bed: a sequential state tracker versus a bag-of-instructions control.

The question is whether tracking register state through a program's actual instruction order buys exact
final-state accuracy over an order-agnostic bag-of-instructions predictor that sees only the multiset of
instructions. Each experimental unit is one tiny register-machine program over R registers. An exact
interpreter executes the program in its true order and defines the ground-truth final register vector.

Two arms predict that final vector from the same initial state and the same transition function; the only
thing that varies is whether the program's order is used:

  candidate  a state tracker that folds the transition function over the instructions in their true order.
             It reproduces the exact interpreter and so recovers the ground-truth final state.
  bag        the NAMED control: a bag-of-instructions predictor with no sequential state. Its output is a
             function of the instruction multiset alone; it reconstructs a final state by executing the same
             instructions in a fixed canonical (sorted) order, which discards the program's real ordering.

Score per unit is exact final-state accuracy: the fraction of the R registers whose predicted final value
matches ground truth. The paired delta per program is candidate minus bag (positive favors the tracker).
Programs whose instructions all commute leave the bag control exact, giving a per-unit tie; those are
legitimate nulls and are not tuned away. The order-dependent instructions (assignment, register-to-register
move, and add) are what a bag cannot place correctly; a per-unit covariate records how many the program
contains so the effect can be read against program structure.

Honesty. A tie or a wrong-direction result is a legitimate null; nothing here is tuned toward a positive.
One run is evidence level M1: consistent with, never a scientific confirmation.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner

N_PROGRAMS = 10  # independent experimental units; within the exact sign-flip enumeration cap
R_REGISTERS = 6  # registers in each tiny machine
MIN_LEN = 10  # shortest program (instructions)
MAX_LEN = 16  # longest program (instructions)
INIT_MAX = 3  # initial register values are drawn from [0, INIT_MAX]
CONST_MAX = 4  # SET constants are drawn from [0, CONST_MAX]
SESOI = 1.0 / R_REGISTERS  # one register of the R-register final state: the smallest structural step
ROUND = 8

OP_INC = 0  # r[a] += 1        (commutative)
OP_DEC = 1  # r[a] -= 1        (commutative)
OP_SET = 2  # r[a] = b         (order dependent: overwrites)
OP_MOV = 3  # r[a] = r[b]      (order dependent: reads another register)
OP_ADD = 4  # r[a] += r[b]     (order dependent: reads another register)
N_OPS = 5
ORDER_DEPENDENT = (OP_SET, OP_MOV, OP_ADD)


def _r(value: Any) -> float:
    """Cast any numpy or python scalar to a plain rounded float so the sealed JSON stays canonical."""

    return round(float(value), ROUND)


def _generate_program(gen: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Draw one initial register vector and a random instruction list. Deterministic given the generator."""

    init = gen.integers(0, INIT_MAX + 1, size=R_REGISTERS).astype(np.int64)
    length = int(gen.integers(MIN_LEN, MAX_LEN + 1))
    program: list[tuple[int, int, int]] = []
    for _ in range(length):
        op = int(gen.integers(0, N_OPS))
        a = int(gen.integers(0, R_REGISTERS))
        if op == OP_SET:
            b = int(gen.integers(0, CONST_MAX + 1))
        elif op in (OP_MOV, OP_ADD):
            b = int(gen.integers(0, R_REGISTERS))
        else:
            b = 0
        program.append((op, a, b))
    return init, program


def _interpret(init: np.ndarray, program: list[tuple[int, int, int]]) -> np.ndarray:
    """The exact register-machine interpreter. Executes the given instruction list in the order supplied."""

    r = init.astype(np.int64).copy()
    for op, a, b in program:
        if op == OP_INC:
            r[a] += 1
        elif op == OP_DEC:
            r[a] -= 1
        elif op == OP_SET:
            r[a] = b
        elif op == OP_MOV:
            r[a] = r[b]
        else:  # OP_ADD
            r[a] += r[b]
    return r


def _final_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Exact final-state accuracy: the fraction of registers whose predicted value matches ground truth."""

    return float(np.mean(pred == truth))


def _simulate_program(seed: int, idx: int) -> dict[str, Any]:
    """Build one program, score both arms against the exact interpreter, and return the per-unit record."""

    gen = rng(seed, "wave_g_program_exec", "program", idx)
    init, program = _generate_program(gen)

    truth_final = _interpret(init, program)

    # candidate: an order-aware sequential fold. It runs the true order and so reproduces ground truth.
    candidate_final = _interpret(init, program)

    # bag control: no sequential state. Its output depends only on the instruction multiset, so we execute
    # the same instructions in a fixed canonical (sorted) order. Sorting a list is invariant to the program's
    # actual ordering, which is exactly the ordering information the bag model is defined to discard.
    bag_program = sorted(program)
    bag_final = _interpret(init, bag_program)

    candidate_acc = _final_accuracy(candidate_final, truth_final)
    bag_acc = _final_accuracy(bag_final, truth_final)
    n_order_dep = sum(1 for op, _a, _b in program if op in ORDER_DEPENDENT)

    return {
        "program": idx,
        "length": len(program),
        "n_order_dependent": n_order_dep,
        "instructions": [[op, a, b] for op, a, b in program],
        "candidate_final_accuracy": _r(candidate_acc),
        "bag_final_accuracy": _r(bag_acc),
        "delta_candidate_vs_bag": _r(candidate_acc - bag_acc),
    }


@register_runner("wave_g.program_execution_state")
def wave_g_program_exec_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Sequential register-state tracking versus an order-agnostic bag-of-instructions control."""

    n_programs = int(params.get("n_programs", N_PROGRAMS))
    per_unit = [_simulate_program(ctx.seed, i) for i in range(n_programs)]

    deltas = [u["delta_candidate_vs_bag"] for u in per_unit]
    sign_flip = exact_sign_flip_one_sided(deltas)

    verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], SESOI)
    is_null = verdict != "survives"

    content = honest_envelope(
        ctx.node_id,
        "mop-campaign-wave_g_program_exec/v1",
        {
            "form_family": "symbolic",
            "phenomenon": "predictive_state",
            "mechanism_family": "transition_state",
            "unit_class": "register_machine_program",
            "evidence_level": "M1",
        },
    )
    content.update(
        {
            "design": {
                "n_programs": n_programs,
                "registers": R_REGISTERS,
                "program_length_range": [MIN_LEN, MAX_LEN],
                "initial_register_range": [0, INIT_MAX],
                "set_constant_range": [0, CONST_MAX],
                "instruction_set": ["INC", "DEC", "SET", "MOV", "ADD"],
                "order_dependent_ops": ["SET", "MOV", "ADD"],
                "score": "exact_final_state_accuracy_over_registers",
                "ground_truth": "exact_interpreter_true_order",
            },
            "control": (
                "bag_of_instructions is an order-agnostic predictor with no sequential state: its output "
                "is a function of the instruction multiset alone, obtained by executing the same "
                "instructions in a fixed canonical (sorted) order. Sorting is invariant to the program's "
                "real ordering, so any final-state mismatch is exactly the ordering information a bag "
                "model cannot access."
            ),
            "per_unit": per_unit,
            "primary_deltas": deltas,
            "sign_flip": sign_flip,
            "sesoi": SESOI,
            "verdict": verdict,
            "alternative_explanation": (
                "The candidate coincides with the exact interpreter by construction, so the measured "
                "effect is the bag control's order-blind loss rather than any nontrivial cleverness in the "
                "tracker. One might worry the bag loses only because the canonical sort happens to differ "
                "from the true order; but the bag output is invariant to the program's actual ordering (a "
                "function of the multiset), so a mismatch demonstrates that final state genuinely depends "
                "on order, which is the information the tracker uses and the bag discards. A different "
                "canonical order would move which registers err while preserving the order-invariance that "
                "defines the control."
            ),
            "failure_domain": (
                "Commutative-heavy programs. When a program touches each register only through INC and DEC "
                "(or adds that never read a later-mutated register), reordering leaves the final state "
                "unchanged, the bag control is already exact, and the tracker's advantage vanishes to a "
                "tie. Very short programs with few order-dependent instructions sit near this boundary."
            ),
        }
    )

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "n_units_favorable": sign_flip["n_units_favorable"],
            "mean_order_dependent": _r(np.mean([u["n_order_dependent"] for u in per_unit])),
        },
    )
