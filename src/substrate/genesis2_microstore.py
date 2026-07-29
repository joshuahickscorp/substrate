"""The associative microstore and the structural consolidator above it.

Genesis I lost to S2 because S2 performed thousands of cheap exact associative
writes while the field proposed a few coarse durable rewrites.  This module is
the direct answer to that finding: it gives every Genesis II candidate the same
cheap exact write S2 has, and then asks whether anything above it earns its
cost.

Three things live here.

``Microstore`` is content-addressable storage with exact and low-bit modes,
bounded collision detection, provenance pointers, scope, expiry, confidence, a
measured write cost, and rollback.  Nothing in it knows what a challenge family
is; it stores ``(scope, key) -> value`` and reports what it cost.

``Rule`` and ``induce`` are the structural consolidator.  A rule is a typed
regularity fitted to repeated associations inside one scope -- constant, affine,
offset map, modular fold, threshold.  Fitting is arithmetic over observed pairs
only; a rule never reads a sealed answer and never stores a probe outcome.  The
point of a rule is that it answers keys the microstore has never seen, which is
exactly the thing an exact associative monolith cannot do.

``compile_rule`` freezes a verified rule into a monitored procedure with a cheap
answer path and a reliability counter that returns control to flexible reasoning
when the procedure starts failing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# The answer alphabet the challenge generators use.  Rules are fitted modulo
# this value because that is the arithmetic the world actually runs on.
MODULUS = 8

#: Low-bit alphabets.  ``exact`` stores the integer unchanged.
RADIX_ALPHABETS: dict[str, tuple[int, ...] | None] = {
    "exact": None,
    "ternary": (-1, 0, 1),
    "quinary": (-2, -1, 0, 1, 2),
    "seven_state": (-3, -2, -1, 0, 1, 2, 3),
}

#: Bits each mode spends per stored component.  Exact storage is charged the
#: full machine word; a low-bit mode is charged the bits its alphabet needs.
RADIX_BITS: dict[str, int] = {
    "exact": 64,
    "ternary": 2,
    "quinary": 3,
    "seven_state": 3,
}

#: Fixed overhead per entry: address, provenance head, confidence, expiry.
ENTRY_OVERHEAD_BYTES = 24

#: A scope must show at least this many distinct pairs before a rule may be
#: fitted to it.  Two points determine a line, so two points prove nothing.
MINIMUM_RULE_SUPPORT = 3

#: A compiled procedure that falls below this reliability over its audit window
#: is decompiled and the material returns to flexible reasoning.
PROCEDURE_RELIABILITY_FLOOR = 0.6
PROCEDURE_AUDIT_WINDOW = 16

_ACTIVATION = False


class MicrostoreRefused(RuntimeError):
    """A write or a fit was attempted in a way that would void the result."""


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def quantize(value: int, mode: str) -> int:
    """Project one integer into the alphabet of ``mode``."""
    alphabet = RADIX_ALPHABETS.get(mode)
    if alphabet is None:
        return int(value)
    return min(alphabet, key=lambda level: (abs(level - int(value)), level))


def entry_bytes(value: Sequence[int], mode: str) -> int:
    """Measured residency of one stored value under ``mode``."""
    bits = RADIX_BITS.get(mode, 64) * max(1, len(value))
    return ENTRY_OVERHEAD_BYTES + (bits + 7) // 8


# --------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------


@dataclass
class Entry:
    """One association, with everything the constitution requires it to carry."""

    scope: str
    key: tuple[int, ...]
    value: tuple[int, ...]
    mode: str
    confidence: int
    provenance: tuple[int, ...]
    expiry: int
    collisions: int = 0
    writes: int = 1

    def expired(self, now: int) -> bool:
        return self.expiry >= 0 and now > self.expiry

    def document(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "key": list(self.key),
            "value": list(self.value),
            "mode": self.mode,
            "confidence": int(self.confidence),
            "provenance": list(self.provenance),
            "expiry": int(self.expiry),
            "collisions": int(self.collisions),
            "writes": int(self.writes),
        }

    @staticmethod
    def restore(document: Mapping[str, Any]) -> Entry:
        return Entry(
            scope=str(document["scope"]),
            key=tuple(int(x) for x in document["key"]),
            value=tuple(int(x) for x in document["value"]),
            mode=str(document["mode"]),
            confidence=int(document["confidence"]),
            provenance=tuple(int(x) for x in document["provenance"]),
            expiry=int(document["expiry"]),
            collisions=int(document.get("collisions", 0)),
            writes=int(document.get("writes", 1)),
        )


def address_of(scope: str, key: Sequence[int]) -> str:
    return f"{scope}|{','.join(str(int(component)) for component in key)}"


# --------------------------------------------------------------------------
# Microstore
# --------------------------------------------------------------------------


@dataclass
class Microstore:
    """High-throughput content-addressable learning with a measured cost.

    ``default_mode`` decides the representation of a write that does not name
    one, which is how the representation arm of the factorial is set: the same
    material with ``exact`` and with ``ternary`` differs in nothing else.
    """

    default_mode: str = "exact"
    #: Bounded collision detection.  An address that receives a different value
    #: is a collision; the count is kept per address and the confidence of a
    #: colliding entry falls rather than the newest write silently winning.
    collision_ceiling: int = 8
    entries: dict[str, Entry] = field(default_factory=dict)
    undo: dict[str, tuple[str, Entry | None]] = field(default_factory=dict)
    total_collisions: int = 0
    total_writes: int = 0
    total_bytes_written: int = 0
    total_bytes_read: int = 0

    # -- reads ----------------------------------------------------------

    def read(self, scope: str, key: Sequence[int], *, now: int = -1) -> Entry | None:
        entry = self.entries.get(address_of(scope, key))
        if entry is None:
            return None
        if now >= 0 and entry.expired(now):
            return None
        self.total_bytes_read += entry_bytes(entry.value, entry.mode)
        return entry

    def scope_pairs(self, scope: str) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Every ``(key, value)`` pair stored under one scope, ordered."""
        rows = [(entry.key, entry.value) for entry in self.entries.values() if entry.scope == scope]
        rows.sort()
        return rows

    def scopes(self) -> tuple[str, ...]:
        return tuple(sorted({entry.scope for entry in self.entries.values()}))

    def nearest(self, scope: str | None, key: Sequence[int]) -> Entry | None:
        """Closest stored key, by Hamming then absolute distance.

        ``scope=None`` searches every scope. A probe channel usually names no
        stored scope at all -- the generators ask on ``survivor`` about facts
        taught on ``store`` -- so a scope-local search returns nothing and the
        global search is the one that actually answers.
        """
        probe = tuple(int(component) for component in key)
        best: Entry | None = None
        best_cost: tuple[int, int, tuple[int, ...]] | None = None
        for entry in self.entries.values():
            if scope is not None and entry.scope != scope:
                continue
            width = min(len(probe), len(entry.key))
            hamming = sum(1 for index in range(width) if probe[index] != entry.key[index])
            hamming += abs(len(probe) - len(entry.key))
            distance = sum(abs(probe[index] - entry.key[index]) for index in range(width))
            cost = (hamming, distance, entry.key)
            if best_cost is None or cost < best_cost:
                best, best_cost = entry, cost
        if best is not None:
            self.total_bytes_read += entry_bytes(best.value, best.mode)
        return best

    # -- writes ---------------------------------------------------------

    def write(
        self,
        scope: str,
        key: Sequence[int],
        value: Sequence[int],
        *,
        mode: str | None = None,
        provenance: Sequence[int] = (),
        expiry: int = -1,
        confidence: int = 128,
        undo_token: str | None = None,
    ) -> int:
        """Store one association and return its measured byte cost."""
        mode = mode or self.default_mode
        if mode not in RADIX_ALPHABETS:
            raise MicrostoreRefused(f"unknown representation mode {mode!r}")
        address = address_of(scope, key)
        stored = tuple(quantize(int(component), mode) for component in value)
        prior = self.entries.get(address)
        if undo_token is not None and undo_token not in self.undo:
            self.undo[undo_token] = (address, None if prior is None else Entry.restore(prior.document()))
        collisions = 0
        if prior is not None:
            collisions = prior.collisions + (1 if tuple(prior.value) != stored else 0)
            if collisions > self.collision_ceiling:
                # A hot address that keeps changing is not a fact; it is noise.
                # Refuse the write rather than let the newest observation win.
                self.total_collisions += 1
                return 0
            if tuple(prior.value) != stored:
                self.total_collisions += 1
                confidence = max(0, min(int(confidence), prior.confidence) - 16)
            else:
                confidence = min(255, prior.confidence + 16)
        cost = entry_bytes(stored, mode)
        self.entries[address] = Entry(
            scope=str(scope),
            key=tuple(int(component) for component in key),
            value=stored,
            mode=mode,
            confidence=int(max(0, min(255, confidence))),
            provenance=tuple(int(index) for index in provenance)[:4],
            expiry=int(expiry),
            collisions=collisions,
            writes=1 if prior is None else prior.writes + 1,
        )
        self.total_writes += 1
        self.total_bytes_written += cost
        return cost

    def demote(self, address: str, mode: str, *, undo_token: str | None = None) -> int:
        """Reduce the precision of one entry.  Returns the bytes reclaimed."""
        entry = self.entries.get(address)
        if entry is None:
            return 0
        if undo_token is not None and undo_token not in self.undo:
            self.undo[undo_token] = (address, Entry.restore(entry.document()))
        before = entry_bytes(entry.value, entry.mode)
        entry.value = tuple(quantize(component, mode) for component in entry.value)
        entry.mode = mode
        return max(0, before - entry_bytes(entry.value, entry.mode))

    def evict(self, address: str, *, undo_token: str | None = None) -> int:
        """Remove one entry.  Returns the bytes reclaimed."""
        entry = self.entries.get(address)
        if entry is None:
            return 0
        if undo_token is not None and undo_token not in self.undo:
            self.undo[undo_token] = (address, Entry.restore(entry.document()))
        del self.entries[address]
        return entry_bytes(entry.value, entry.mode)

    def rollback(self, undo_token: str) -> bool:
        """Undo one recorded change exactly.  P2 requires the benefit to vanish."""
        record = self.undo.pop(undo_token, None)
        if record is None:
            return False
        address, prior = record
        if prior is None:
            self.entries.pop(address, None)
        else:
            self.entries[address] = prior
        return True

    def forget_undo(self, undo_token: str) -> None:
        self.undo.pop(undo_token, None)

    # -- accounting -----------------------------------------------------

    def resident_bytes(self) -> int:
        return sum(entry_bytes(entry.value, entry.mode) for entry in self.entries.values())

    def measurement(self) -> dict[str, int]:
        return {
            "entries": len(self.entries),
            "writes": self.total_writes,
            "collisions": self.total_collisions,
            "bytes_written": self.total_bytes_written,
            "bytes_read": self.total_bytes_read,
            "resident_bytes": self.resident_bytes(),
        }

    def document(self) -> dict[str, Any]:
        return {
            "default_mode": self.default_mode,
            "entries": {address: entry.document() for address, entry in sorted(self.entries.items())},
            "activation": _ACTIVATION,
        }

    def restore(self, document: Mapping[str, Any]) -> None:
        self.default_mode = str(document.get("default_mode", "exact"))
        self.entries = {str(address): Entry.restore(row) for address, row in document.get("entries", {}).items()}
        self.undo.clear()

    def digest(self) -> str:
        return _digest(self.document())


# --------------------------------------------------------------------------
# Structural consolidation: typed rules fitted to repeated associations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rule:
    """One typed regularity over a scope's associations.

    A rule is the meso-structure of the master plan: it is what repeated
    micro-associations consolidate into, and it earns its cost by answering
    keys no association covers.
    """

    kind: str
    scope: str
    params: tuple[int, ...]
    support: int
    arity: int

    def apply(self, key: Sequence[int]) -> tuple[int, ...] | None:
        components = tuple(int(component) for component in key)
        if self.kind == "constant":
            return self.params[: self.arity] or None
        if not components:
            return None
        head = components[0]
        if self.kind == "affine":
            slope, intercept = self.params[0], self.params[1]
            return ((slope * head + intercept) % MODULUS,)
        if self.kind == "offset_map":
            offset, modulus = self.params[0], self.params[1]
            if modulus <= 0:
                return None
            return ((head + offset) % modulus,)
        if self.kind == "modular_fold":
            bias = self.params[0]
            total = bias
            for component in components:
                total = (total + component) % MODULUS
            return (total,)
        if self.kind == "threshold":
            threshold, low, high = self.params[0], self.params[1], self.params[2]
            probe = components[1] if len(components) > 1 else head
            return (high if probe >= threshold else low,)
        if self.kind == "parameterised_affine":
            # The functional form is learned; its arguments arrive with the
            # query. This is what using a tool is: the material knows that a
            # tool applies a scale and an offset without knowing, until asked,
            # which scale and which offset this call carries.
            argument, scale, offset = self.params[0], self.params[1], self.params[2]
            width = max(argument, scale, offset) + 1
            if len(components) < width:
                return None
            return ((components[scale] * components[argument] + components[offset]) % MODULUS,)
        return None

    def document(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scope": self.scope,
            "params": list(self.params),
            "support": int(self.support),
            "arity": int(self.arity),
        }

    @staticmethod
    def restore(document: Mapping[str, Any]) -> Rule:
        return Rule(
            kind=str(document["kind"]),
            scope=str(document["scope"]),
            params=tuple(int(x) for x in document["params"]),
            support=int(document["support"]),
            arity=int(document["arity"]),
        )


def _fit_constant(pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[int, ...] | None:
    values = {pair[1] for pair in pairs}
    if len(values) != 1:
        return None
    return next(iter(values))


def _fit_affine(pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[int, int] | None:
    """Solve y = (a*x + b) mod MODULUS from two points, then verify on all."""
    points = [(pair[0][0] % MODULUS, pair[1][0] % MODULUS) for pair in pairs if pair[0] and pair[1]]
    if len(points) < MINIMUM_RULE_SUPPORT:
        return None
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            x0, y0 = points[left]
            x1, y1 = points[right]
            dx = (x1 - x0) % MODULUS
            if dx == 0:
                continue
            try:
                inverse = pow(dx, -1, MODULUS)
            except ValueError:
                continue
            slope = ((y1 - y0) * inverse) % MODULUS
            intercept = (y0 - slope * x0) % MODULUS
            if all((slope * x + intercept) % MODULUS == y for x, y in points):
                return slope, intercept
    return None


def _fit_offset_map(pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[int, int] | None:
    """Solve y = (x + c) mod m for a modulus large enough to be a re-encoding."""
    points = [(pair[0][0], pair[1][0]) for pair in pairs if pair[0] and pair[1]]
    if len(points) < MINIMUM_RULE_SUPPORT:
        return None
    span = max(max(abs(x), abs(y)) for x, y in points) + 1
    for modulus in (span, span + 1, 2 * span):
        if modulus <= 1:
            continue
        offsets = {(y - x) % modulus for x, y in points}
        if len(offsets) == 1:
            offset = next(iter(offsets))
            if offset % modulus == 0:
                continue  # the identity is not a learned re-encoding
            return offset, modulus
    return None


def _fit_modular_fold(pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[int] | None:
    """Solve y = (bias + sum(key)) mod MODULUS."""
    points = [(pair[0], pair[1][0] % MODULUS) for pair in pairs if pair[0] and pair[1]]
    if len(points) < MINIMUM_RULE_SUPPORT:
        return None
    biases = {(value - sum(key)) % MODULUS for key, value in points}
    if len(biases) != 1:
        return None
    return (next(iter(biases)),)


def _fit_threshold(pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[int, int, int] | None:
    """Solve y = high if key[1] >= t else low, for a binary-valued scope."""
    points = [(pair[0], pair[1][0]) for pair in pairs if len(pair[0]) >= 2 and pair[1]]
    if len(points) < MINIMUM_RULE_SUPPORT:
        return None
    values = sorted({value for _, value in points})
    if len(values) != 2:
        return None
    low, high = values
    probes = sorted({key[1] for key, _ in points})
    for threshold in probes:
        if all((high if key[1] >= threshold else low) == value for key, value in points):
            return threshold, low, high
    return None


def fit_parameterised_affine(
    rows: Sequence[Sequence[int]],
    values: Sequence[int],
    *,
    max_width: int = 6,
) -> tuple[int, int, int] | None:
    """Find slots (argument, scale, offset) with value = (scale*argument + offset).

    The search is over slot assignments, not over constants, so what is learned
    is the shape of the computation rather than one instance of it. It has to
    hold for every observed row: a coincidence on one row is not a procedure.
    """
    usable = [(tuple(int(x) for x in row), int(value) % MODULUS) for row, value in zip(rows, values, strict=True)]
    usable = [(row, value) for row, value in usable if len(row) >= 2]
    if len(usable) < MINIMUM_RULE_SUPPORT:
        return None
    width = min(max_width, min(len(row) for row, _ in usable))
    for argument in range(width):
        for scale in range(width):
            for offset in range(width):
                if len({argument, scale, offset}) < 3:
                    continue
                if all((row[scale] * row[argument] + row[offset]) % MODULUS == value for row, value in usable):
                    return argument, scale, offset
    return None


def parameterise_affine(
    rule: Rule,
    *,
    argument_slot: int,
    scale_slot: int,
    offset_slot: int,
) -> Rule:
    """Compile an observed affine relation into a query-argument procedure.

    The observation-side rule has already established that the source behaves
    affinely.  This transformation does not retain its fitted constants.  It
    records where a later query supplies the argument, scale, and offset, so
    the same learned functional form can be applied to a new invocation.
    """
    if rule.kind != "affine":
        raise MicrostoreRefused("only an observed affine rule can be parameterised")
    slots = (int(argument_slot), int(scale_slot), int(offset_slot))
    if min(slots) < 0 or len(set(slots)) != 3:
        raise MicrostoreRefused("parameterised affine slots must be distinct non-negative indices")
    return Rule(
        kind="parameterised_affine",
        scope=rule.scope,
        params=slots,
        support=rule.support,
        arity=1,
    )


#: Fit order is cheapest and most constrained first.  A constant that explains
#: the scope is preferred to an affine map that also explains it, because the
#: constant carries less structure and therefore less risk.
_FITTERS: tuple[tuple[str, Any, int], ...] = (
    ("constant", _fit_constant, 0),
    ("threshold", _fit_threshold, 3),
    ("modular_fold", _fit_modular_fold, 1),
    ("affine", _fit_affine, 2),
    ("offset_map", _fit_offset_map, 2),
)


def induce(scope: str, pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> list[Rule]:
    """Fit every typed regularity that explains a scope's associations.

    Fitting reads observed ``(key, value)`` pairs only.  It never reads a sealed
    answer, a probe outcome, or an evaluator verdict; a rule that does not in
    fact help is rolled back by the harness like any other proposal.
    """
    distinct = sorted({(tuple(key), tuple(value)) for key, value in pairs})
    if len(distinct) < MINIMUM_RULE_SUPPORT:
        return []
    rules: list[Rule] = []
    for kind, fitter, _params in _FITTERS:
        fitted = fitter(distinct)
        if fitted is None:
            continue
        arity = len(distinct[0][1]) if kind == "constant" else 1
        rules.append(
            Rule(
                kind=kind,
                scope=scope,
                params=tuple(int(component) for component in fitted),
                support=len(distinct),
                arity=arity,
            )
        )
    return rules


# --------------------------------------------------------------------------
# Cognitive compilation
# --------------------------------------------------------------------------


@dataclass
class Procedure:
    """A verified rule frozen into a monitored cheap answer path.

    ``hits`` and ``misses`` are the reliability window.  A procedure that falls
    below the floor is decompiled and the material returns to flexible
    reasoning, which is claim C18 and the ``procedure_loses_accuracy`` mutation.
    """

    rule: Rule
    hits: int = 0
    misses: int = 0
    retired: bool = False

    def reliability(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 1.0
        return self.hits / total

    def live(self) -> bool:
        if self.retired:
            return False
        if self.hits + self.misses < PROCEDURE_AUDIT_WINDOW:
            return True
        return self.reliability() >= PROCEDURE_RELIABILITY_FLOOR

    def observe(self, *, hit: bool) -> None:
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        if not self.live():
            self.retired = True

    def document(self) -> dict[str, Any]:
        return {
            "rule": self.rule.document(),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "retired": bool(self.retired),
        }

    @staticmethod
    def restore(document: Mapping[str, Any]) -> Procedure:
        return Procedure(
            rule=Rule.restore(document["rule"]),
            hits=int(document.get("hits", 0)),
            misses=int(document.get("misses", 0)),
            retired=bool(document.get("retired", False)),
        )


def compile_rule(rule: Rule) -> Procedure:
    return Procedure(rule=rule)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


def demo() -> None:
    """Runnable self-check for the guarantees this module is responsible for."""
    store = Microstore(default_mode="exact")
    cost = store.write("tool_use", (1, 2), (5,), provenance=(0,), undo_token="w0")
    assert cost > 0
    exact_entry = store.read("tool_use", (1, 2))
    assert exact_entry is not None and exact_entry.value == (5,)

    # Exact keeps the integer; a low-bit mode projects it into its alphabet.
    low = Microstore(default_mode="ternary")
    low.write("tool_use", (1, 2), (5,))
    low_entry = low.read("tool_use", (1, 2))
    assert low_entry is not None and low_entry.value == (1,), low_entry
    assert entry_bytes((5,), "ternary") < entry_bytes((5,), "exact")

    # Rollback restores the prior state exactly.
    before = store.digest()
    store.write("tool_use", (3, 4), (6,), undo_token="w1")
    assert store.digest() != before
    assert store.rollback("w1") is True
    assert store.digest() == before

    # A colliding address loses confidence rather than silently taking the new value.
    store.write("noisy", (1,), (1,))
    first_entry = store.read("noisy", (1,))
    assert first_entry is not None
    first = first_entry.confidence
    store.write("noisy", (1,), (2,))
    collided = store.read("noisy", (1,))
    assert collided is not None and collided.confidence < first
    assert store.total_collisions >= 1

    # Affine induction generalises to a key that was never written.
    pairs = [((x,), ((3 * x + 2) % MODULUS,)) for x in range(4)]
    rules = induce("tool_use", pairs)
    affine = [rule for rule in rules if rule.kind == "affine"]
    assert affine, [rule.kind for rule in rules]
    unseen = 6
    assert affine[0].apply((unseen,)) == (((3 * unseen + 2) % MODULUS),), affine[0].apply((unseen,))
    parameterised = parameterise_affine(affine[0], argument_slot=1, scale_slot=2, offset_slot=3)
    assert parameterised.apply((99, unseen, 3, 2)) == (((3 * unseen + 2) % MODULUS),)

    # Threshold induction recovers a revised category boundary.
    threshold_pairs = [((0, value), (1 if value >= 5 else 0,)) for value in range(8)]
    threshold = [rule for rule in induce("category", threshold_pairs) if rule.kind == "threshold"]
    assert threshold and threshold[0].apply((0, 7)) == (1,) and threshold[0].apply((0, 1)) == (0,)

    # Offset maps recover a sensor re-encoding.
    offset_pairs = [((state,), ((state + 3) % 11,)) for state in range(5)]
    offset = [rule for rule in induce("sensor", offset_pairs) if rule.kind == "offset_map"]
    assert offset and offset[0].apply((4,)) == ((4 + 3) % 11,)

    # A scope with too little support yields nothing at all.
    assert induce("thin", [((0,), (1,)), ((1,), (2,))]) == []

    # A procedure retires once it stops paying, and never before its window.
    procedure = compile_rule(affine[0])
    for _ in range(PROCEDURE_AUDIT_WINDOW):
        procedure.observe(hit=False)
    assert procedure.retired is True and procedure.live() is False

    # Precision demotion reclaims bytes and eviction reclaims more.
    reclaimed = store.demote(address_of("tool_use", (1, 2)), "ternary")
    assert reclaimed > 0
    assert store.evict(address_of("tool_use", (1, 2))) > 0

    # Checkpoint and restore reproduce the developed state exactly.
    snapshot = store.document()
    replica = Microstore()
    replica.restore(snapshot)
    assert replica.digest() == store.digest()

    print(f"genesis2 microstore self-check passed: {len(rules)} rule kinds fitted on the affine scope")


if __name__ == "__main__":
    demo()
