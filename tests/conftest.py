"""Fixtures are artificial contract instruments, never Odyssey qualification receipts."""
import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from substrate.cognition import Entity
from substrate.core import Trust, TrustRule, attest, sha
from substrate.store import Store

HOST = sha("fixture-host")
SCOPE = {"task": "fixture-task", "world": "fixture-world"}
INPUTS = {"x": {"type": "int", "min": -1000000, "max": 1000000}}


@pytest.fixture
def key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def trust(key):
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return Trust({"oracle": TrustRule(base64.b64encode(public).decode(),
        ("belief", "experience", "skill", "organ", "organ-binding", "world", "report", "run", "continuity", "chapter", "agency", "native-image", "apprenticeship"),
        ("micro-world", "cognition", "odyssey", "entity"))})


@pytest.fixture
def actor(tmp_path, trust):
    with Store(tmp_path / "actor.sqlite", create=True) as store:
        yield Entity.create(store, "fixture", trust, host=HOST)


def certificate(key, entity, purpose, proposal, findings, *, producer="tester", domain="micro-world"):
    return attest(key, signer="oracle", producer=producer, purpose=purpose, domain=domain,
                  subject=entity.subject(purpose, proposal), evidence=sha(findings), findings=findings, ttl=3600)


def experience(entity, key, x, *, expected=None, actual=None, skill=None):
    actual = 2 * x + 1 if actual is None else actual
    public = {"case_id": sha(["train", x]), "domain": "micro-world", "family": "affine", "split": "train",
              "input": {"x": x}, "source": sha("fixture-source"), "scope": SCOPE}
    observed = entity.observe(public)
    predicted = entity.predict(observed, expected, producer="tester", mechanism="fixture", confidence=.5, skill_id=skill)
    proposal = {"prediction": predicted, "case_id": public["case_id"], "unit": "unit-0", "actual": actual, "producer": "tester"}
    signed = certificate(key, entity, "experience", proposal, {"outcome_verified": True, "case_id": public["case_id"]})
    return entity.assimilate(proposal, signed)


def native_skill(entity, key):
    from substrate.language import make_program
    witnesses = [experience(entity, key, x, expected=2 * x + 1) for x in range(4)]
    program = make_program(INPUTS, ["add", ["mul", ["const", 2], ["var", "x"]], ["const", 1]])
    skill = entity.propose_skill(program, family="affine", domain="micro-world", scope=SCOPE, witnesses=witnesses, producer="tester")
    row = entity.get("skills", skill)
    proposal = {"skill": skill, "record": sha(row), "host": HOST}
    findings = {"passed": True, "split": "qualification", "program": row["program_digest"], "host": HOST,
                "scope": sha(SCOPE), "cases": [sha(["qualification", i]) for i in range(8)],
                "input_digests": [sha({"x": 100 + i}) for i in range(8)], "oracle": sha("fixture-oracle"),
                "cost": {"elapsed_ns": 1000, "model_calls": 0}}
    signed = certificate(key, entity, "skill", proposal, findings)
    entity.qualify_skill(skill, signed)
    return skill
