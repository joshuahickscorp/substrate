"""Bounded typed programs: learned procedures are data, never host Python code.

A validated expression is compiled once into a shared-expression DAG. Reuse
executes that DAG without a model call. This is an interpretable IR compiler,
not a JIT, a learned universal language, or a Hawking NR/NX compiler.
"""
from __future__ import annotations

import itertools
import time
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Iterable

from .core import Refused, canonical, clone, fields, identifier, integer, numeric, require, sha

IR_SCHEMA = "substrate-ir-v2"
MAX_NODES = 256
MAX_ITEMS = 256
ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "floordiv": 2, "mod": 2,
         "min": 2, "max": 2, "lt": 2, "le": 2, "eq": 2, "and": 2, "or": 2,
         "not": 1, "neg": 1, "abs": 1, "if": 3, "len": 1, "sum": 1,
         "sort": 1, "reverse": 1, "unique": 1, "at": 2, "take": 2, "concat": 2}
NUMERIC = {"int", "number"}


def _scalar(value: Any) -> str:
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        integer(abs(value), 0, 2**53)
        return "int"
    numeric(value, -1e50, 1e50)
    return "number"


def validate_input(spec: dict, value: Any) -> Any:
    kind = spec["type"]
    if kind == "bool":
        require(type(value) is bool, "expected boolean input")
    elif kind in NUMERIC:
        if kind == "int":
            require(type(value) is int, "expected integer input")
        numeric(value, spec["min"], spec["max"])
    elif kind == "list-int":
        require(type(value) is list and len(value) <= spec["max_items"], "list input bound exceeded")
        for item in value:
            integer(abs(item) if type(item) is int else item, 0, 2**53)
    else:
        raise Refused("unsupported input type")
    return value


def _bound(value: Any) -> Any:
    if type(value) is list:
        require(len(value) <= MAX_ITEMS, "intermediate list bound exceeded")
        for item in value:
            _scalar(item)
    else:
        _scalar(value)
    return value


def apply(op: str, args: list[Any]) -> Any:
    """Closed instruction set. No reflection, paths, imports, network or file access."""
    a = args[0] if args else None
    b = args[1] if len(args) > 1 else None
    if op == "add":
        result = a + b
    elif op == "sub":
        result = a - b
    elif op == "mul":
        result = a * b
    elif op == "div":
        require(b != 0, "division by zero")
        result = a / b
    elif op == "floordiv":
        require(b != 0, "division by zero")
        result = a // b
    elif op == "mod":
        require(b != 0, "modulo by zero")
        result = a % b
    elif op == "min":
        result = min(a, b)
    elif op == "max":
        result = max(a, b)
    elif op == "lt":
        result = a < b
    elif op == "le":
        result = a <= b
    elif op == "eq":
        result = type(a) is type(b) and a == b
    elif op == "and":
        result = a and b
    elif op == "or":
        result = a or b
    elif op == "not":
        result = not a
    elif op == "neg":
        result = -a
    elif op == "abs":
        result = abs(a)
    elif op == "len":
        result = len(a)
    elif op == "sum":
        result = sum(a)
    elif op == "sort":
        result = sorted(a)
    elif op == "reverse":
        result = list(reversed(a))
    elif op == "unique":
        result = list(dict.fromkeys(a))
    elif op == "at":
        require(0 <= b < len(a), "index out of range")
        result = a[b]
    elif op == "take":
        require(0 <= b <= MAX_ITEMS, "slice bound exceeded")
        result = a[:b]
    elif op == "concat":
        require(len(a) + len(b) <= MAX_ITEMS, "concatenation bound exceeded")
        result = a + b
    else:
        raise Refused("unknown instruction")
    return _bound(result)


def result_type(op: str, types: list[str]) -> str:
    if op in {"add", "sub", "mul", "div", "floordiv", "mod", "min", "max", "neg", "abs"}:
        require(all(t in NUMERIC for t in types), "arithmetic requires numbers")
        if op in {"floordiv", "mod"}:
            require(all(t == "int" for t in types), "integer arithmetic requires integers")
        return "number" if "number" in types or op == "div" else "int"
    if op in {"lt", "le"}:
        require(all(t in NUMERIC for t in types), "comparison requires numbers")
        return "bool"
    if op == "eq":
        require(types[0] == types[1], "equality requires matching types")
        return "bool"
    if op in {"and", "or", "not"}:
        require(all(t == "bool" for t in types), "logical instruction requires booleans")
        return "bool"
    if op == "if":
        require(types[0] == "bool" and types[1] == types[2], "if needs bool and matching branch types")
        return types[1]
    if op in {"len", "sum", "sort", "reverse", "unique"}:
        require(types == ["list-int"], "list instruction requires a list of integers")
        return "int" if op in {"len", "sum"} else "list-int"
    if op in {"at", "take"}:
        require(types == ["list-int", "int"], "index/slice type mismatch")
        return "int" if op == "at" else "list-int"
    if op == "concat":
        require(types == ["list-int", "list-int"], "concat requires integer lists")
        return "list-int"
    raise Refused("unsupported type rule")


@dataclass(frozen=True)
class Program:
    document: dict
    instructions: tuple[tuple[str, tuple[int, ...], Any], ...]
    output: int
    output_type: str
    digest: str

    def run(self, inputs: dict) -> Any:
        require(type(inputs) is dict and set(inputs) == set(self.document["inputs"]), "program input names differ")
        for name, spec in self.document["inputs"].items():
            validate_input(spec, inputs[name])
        memo: dict[int, Any] = {}

        def execute(index: int) -> Any:
            if index in memo:
                return memo[index]
            op, refs, literal = self.instructions[index]
            if op == "var":
                value = inputs[literal]
            elif op == "const":
                value = literal
            elif op == "if":
                value = execute(refs[1] if execute(refs[0]) else refs[2])
            elif op == "and":
                value = execute(refs[0]) and execute(refs[1])
            elif op == "or":
                value = execute(refs[0]) or execute(refs[1])
            else:
                value = apply(op, [execute(ref) for ref in refs])
            memo[index] = _bound(value)
            return value

        return clone(execute(self.output))

    def cost(self) -> dict:
        return {"ir_bytes": len(canonical(self.document)), "instructions": len(self.instructions),
                "model_calls": 0, "kind": "structural-not-measured"}


def compile_program(document: dict) -> Program:
    fields(document, {"schema", "inputs", "body"})
    require(document["schema"] == IR_SCHEMA, "unsupported IR schema")
    require(type(document["inputs"]) is dict and 0 < len(document["inputs"]) <= 32, "declare 1-32 inputs")
    require(len(canonical(document)) <= 64 * 1024, "program byte bound exceeded")
    for name, spec in document["inputs"].items():
        identifier(name)
        require(type(spec) is dict and "type" in spec, "input specification missing type")
        kind = spec["type"]
        if kind in NUMERIC:
            fields(spec, {"type", "min", "max"})
            numeric(spec["min"], -1e50, 1e50)
            numeric(spec["max"], spec["min"], 1e50)
            if kind == "int":
                require(type(spec["min"]) is int and type(spec["max"]) is int, "integer input bounds required")
        elif kind == "bool":
            fields(spec, {"type"})
        else:
            fields(spec, {"type", "max_items"})
            require(kind == "list-int", "unsupported input type")
            integer(spec["max_items"], 0, MAX_ITEMS)
    instructions, seen = [], {}
    visited = 0

    def visit(node, depth=0):
        nonlocal visited
        visited += 1
        require(depth <= 32 and visited <= MAX_NODES, "program structural bound exceeded")
        require(type(node) is list and len(node) >= 2 and type(node[0]) is str, "invalid expression")
        key = sha(node)
        if key in seen:
            return seen[key]
        op = node[0]
        if op == "var":
            require(len(node) == 2 and type(node[1]) is str and node[1] in document["inputs"], "unknown input")
            kind, refs, literal = document["inputs"][node[1]]["type"], (), node[1]
        elif op == "const":
            require(len(node) == 2, "constant arity")
            kind, refs, literal = _scalar(node[1]), (), node[1]
        else:
            require(op in ARITY and len(node) == ARITY[op] + 1, "unknown instruction or invalid arity")
            children = [visit(child, depth + 1) for child in node[1:]]
            kind = result_type(op, [child[1] for child in children])
            refs, literal = tuple(child[0] for child in children), None
        index = len(instructions)
        instructions.append((op, refs, literal))
        seen[key] = index, kind
        return index, kind

    output, output_type = visit(document["body"])
    return Program(clone(document), tuple(instructions), output, output_type, sha(document))


def make_program(inputs: dict, body: list) -> dict:
    document = {"schema": IR_SCHEMA, "inputs": clone(inputs), "body": clone(body)}
    compile_program(document)
    return document


def equivalent(actual: Any, expected: Any, *, tolerance: float = 0.0) -> bool:
    numeric(tolerance, 0, 1)
    if type(actual) is bool or type(expected) is bool:
        return type(actual) is type(expected) and actual == expected
    if type(actual) in (int, float) and type(expected) in (int, float):
        return abs(numeric(actual) - numeric(expected)) <= tolerance
    return sha(actual) == sha(expected)


def induce(samples: list[dict], inputs: dict, *, max_nodes: int = 7,
           max_candidates: int = 5000, seconds: float = 0.25,
           operators: tuple[str, ...] = ("add", "sub", "mul", "min", "max")) -> dict:
    """Bounded bottom-up search with behavioral deduplication on training ONLY.

    A shortest observed fit is a candidate, not a generalization certificate.
    Search constants are fixed independently of held-out answers.
    """
    require(2 <= len(samples) <= 64, "induction needs 2-64 training examples")
    integer(max_nodes, 1, 15)
    integer(max_candidates, 1, 100_000)
    numeric(seconds, 0.001, 60)
    require(all(op in {"add", "sub", "mul", "min", "max"} for op in operators), "unsupported search grammar")
    require(all(spec["type"] in NUMERIC for spec in inputs.values()), "enumerator supports scalar numeric inputs")
    started, examined = time.monotonic(), 0
    target = [sample["output"] for sample in samples]
    by_size: dict[int, list[tuple[list, list]]] = {1: []}
    signatures = set()
    candidate = None

    def remember(body, values, size):
        nonlocal examined, candidate
        if examined >= max_candidates or time.monotonic() - started >= seconds:
            return
        examined += 1
        signature = sha(values)
        if signature not in signatures:
            signatures.add(signature)
            by_size.setdefault(size, []).append((body, values))
            if all(equivalent(a, b) for a, b in zip(values, target)):
                candidate = make_program(inputs, body)

    for name in inputs:
        for sample in samples:
            validate_input(inputs[name], sample["input"][name])
        remember(["var", name], [s["input"][name] for s in samples], 1)
    for constant in (-2, -1, 0, 1, 2):
        remember(["const", constant], [constant] * len(samples), 1)
    for size in range(3, max_nodes + 1, 2):
        if candidate is not None:
            break
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            for (left, lv), (right, rv), op in itertools.product(
                    tuple(by_size.get(left_size, ())), tuple(by_size.get(right_size, ())), operators):
                if examined >= max_candidates or time.monotonic() - started >= seconds or candidate is not None:
                    break
                try:
                    values = [apply(op, [a, b]) for a, b in zip(lv, rv)]
                except (Refused, ArithmeticError):
                    examined += 1
                    continue
                remember([op, left, right], values, size)
            if examined >= max_candidates or time.monotonic() - started >= seconds or candidate is not None:
                break
        if examined >= max_candidates or time.monotonic() - started >= seconds:
            break
    return {"program": candidate, "status": "candidate" if candidate else "no-fit-within-budget",
            "examined": examined, "elapsed_ns": int((time.monotonic() - started) * 1e9),
            "training_digest": sha(samples), "qualified": False}


def mine_material(programs: Iterable[dict], *, minimum_uses: int = 2) -> list[dict]:
    """Propose recurring subexpressions. No automatic claim that a useful concept was learned."""
    integer(minimum_uses, 2)
    occurrences: dict[str, dict] = {}
    for document in programs:
        program = compile_program(document)
        local = set()

        def visit(node):
            if node[0] in {"var", "const"}:
                return
            key = sha(node)
            row = occurrences.setdefault(key, {"expression": clone(node), "programs": []})
            if key not in local:
                row["programs"].append(program.digest)
                local.add(key)
            for child in node[1:]:
                visit(child)

        visit(document["body"])
    return [{"kind": "abstraction-candidate", "id": key, **row, "status": "unqualified"}
            for key, row in sorted(occurrences.items()) if len(set(row["programs"])) >= minimum_uses]


def abstract_material(programs: Iterable[dict]) -> list[dict]:
    """Anti-unify whole numeric programs by variable and literal roles, not by their answers.

    Candidate grammar is deliberately bounded. Two distinct qualified programs
    must support the same structural template; reusing it still requires new
    training witnesses and independent held-out qualification.
    """
    groups = {}
    for document in programs:
        program = compile_program(document)
        variables, parameters = [], []
        def normalize(node):
            if node[0] == "var":
                name = node[1]
                require(document["inputs"][name]["type"] in NUMERIC, "numeric abstraction only")
                if name not in variables:
                    variables.append(name)
                return ["argument", variables.index(name), document["inputs"][name]["type"]]
            if node[0] == "const":
                kind = _scalar(node[1])
                require(kind in NUMERIC, "numeric abstraction only")
                slot = (kind, sha(node[1]))
                if slot not in parameters:
                    parameters.append(slot)
                return ["parameter", parameters.index(slot), kind]
            return [node[0], *[normalize(child) for child in node[1:]]]
        try:
            template = normalize(program.document["body"])
        except Refused:
            continue
        if not (1 <= len(variables) <= 4 and len(parameters) <= 3):
            continue
        key = sha(template)
        row = groups.setdefault(key, {"id": key, "kind": "parameterized-abstraction", "template": template,
              "arguments": len(variables), "parameters": len(parameters), "programs": [], "status": "unqualified"})
        row["programs"].append(program.digest)
    return [row for _, row in sorted(groups.items()) if len(set(row["programs"])) >= 2]


def instantiate_material(material: dict, inputs: dict, names: tuple, constants: tuple) -> dict:
    require(material["kind"] == "parameterized-abstraction", "unknown reusable template")
    require(len(names) == material["arguments"] and len(constants) == material["parameters"], "template binding differs")
    def bind(node, depth=0):
        require(type(node) is list and bool(node) and depth <= 32, "template shape bound")
        if node[0] == "argument":
            integer(node[1], 0, len(names) - 1)
            require(inputs[names[node[1]]]["type"] == node[2], "argument type mismatch")
            return ["var", names[node[1]]]
        if node[0] == "parameter":
            integer(node[1], 0, len(constants) - 1)
            value = constants[node[1]]
            require(_scalar(value) == node[2], "parameter type mismatch")
            return ["const", value]
        require(node[0] in ARITY and len(node) == ARITY[node[0]] + 1, "unknown template operation")
        return [node[0], *[bind(child, depth + 1) for child in node[1:]]]
    return compile_program(make_program(inputs, bind(material["template"]))).document


def transfer_material(materials: list[dict], samples: list[dict], inputs: dict, *, max_candidates=512, seconds=.05) -> dict:
    """Search acquired templates before primitive enumeration; count this acquisition work too."""
    integer(max_candidates, 1, 100000)
    numeric(seconds, .000001, 60)
    require(len(materials) <= 64 and 2 <= len(samples) <= 64, "material or witness bound")
    started, examined, found, origin = time.monotonic(), 0, None, None
    for row in materials:
        if row.get("kind") != "parameterized-abstraction" or row.get("status") == "suspended":
            continue
        integer(row["arguments"], 1, 4)
        integer(row["parameters"], 0, 3)
        for names in itertools.permutations(sorted(inputs), row["arguments"]):
            for constants in itertools.product((-2, -1, 0, 1, 2), repeat=row["parameters"]):
                if examined >= max_candidates or time.monotonic() - started >= seconds:
                    return {"program": None, "examined": examined, "origin": None, "qualified": False}
                examined += 1
                try:
                    candidate = instantiate_material(row, inputs, names, constants)
                    compiled = compile_program(candidate)
                    if all(equivalent(compiled.run(s["input"]), s["output"]) for s in samples):
                        found, origin = candidate, row["id"]
                        break
                except (Refused, ArithmeticError, KeyError, TypeError):
                    continue
            if found:
                break
        if found:
            break
    return {"program": found, "examined": examined, "origin": origin, "qualified": False}


class CausalModel:
    """Explicit acyclic structural equations with fixed exogenous variables.

    Interventions replace equations. Counterfactuals require caller-supplied
    exogenous values; observational fit is not silently treated as abduction.
    """
    def __init__(self, document: dict):
        fields(document, {"inputs", "equations"})
        require(type(document["inputs"]) is dict and type(document["equations"]) is dict,
                "causal model needs input and equation dictionaries")
        require(0 < len(document["inputs"]) + len(document["equations"]) <= 64, "causal model size bound")
        require(not set(document["inputs"]) & set(document["equations"]), "endogenous/exogenous collision")
        self.document = clone(document)
        self.programs = {identifier(name): compile_program(value) for name, value in document["equations"].items()}
        available = set(document["inputs"])
        self.order = []
        while len(self.order) != len(self.programs):
            ready = sorted(name for name, program in self.programs.items()
                           if name not in available and set(program.document["inputs"]) <= available)
            require(bool(ready), "cyclic equations or missing causal variable")
            self.order.extend(ready)
            available.update(ready)
        self.digest = sha(document)

    def predict(self, exogenous: dict, *, do: dict | None = None) -> dict:
        require(set(exogenous) == set(self.document["inputs"]), "complete exogenous state required")
        values = clone(exogenous)
        for name, spec in self.document["inputs"].items():
            validate_input(spec, values[name])
        intervention = do or {}
        require(set(intervention) <= set(self.programs), "intervention must name endogenous variables")
        for name in self.order:
            program = self.programs[name]
            if name in intervention:
                value = _bound(intervention[name])
                require(_scalar(value) == program.output_type, "intervention type mismatch")
            else:
                value = program.run({key: values[key] for key in program.document["inputs"]})
            values[name] = value
        return values

    def counterfactual(self, exogenous: dict, intervention: dict) -> dict:
        return {"factual": self.predict(exogenous), "counterfactual": self.predict(exogenous, do=intervention),
                "assumption": "same explicit exogenous state; no inferred abduction"}


@lru_cache(maxsize=256)
def cached_program(encoded: bytes) -> Program:
    """Bounded process-local DAG cache, not an answer cache or entity-owned authority."""
    from .core import parse
    return compile_program(parse(encoded))
