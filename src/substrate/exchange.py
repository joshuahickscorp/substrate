"""Consent-aware portable skill exchange with quarantine, not automatic mind merging.

A public capsule is deliberately not a full personal checkpoint. Export requires
an explicit source-rights closure and disclosure review. Imports cannot install
credentials, change constitutions, become qualified skills, or execute code.
"""
from __future__ import annotations

from .core import canonical, clone, digest, fields, identifier, require, sha
from .language import compile_program


def capsule(entity, *, skills: list[str], rights: dict, author: str) -> dict:
    require(entity.store.entity(entity.id)["frozen"], "publish from a frozen entity")
    require(type(skills) is list and 1 <= len(set(skills)) <= 128, "select 1..128 skills")
    programs, source_closure = [], set()
    for key in skills:
        row = entity.get("skills", digest(key))
        require(row and entity.skill_ready(row, row["family"], row["scope"]), "only currently qualified source skills can be proposed for sharing")
        require(not row["dependencies"], "export dependent beliefs through a separately reviewed whole-entity transfer")
        source_closure.update(row.get("source_ancestry", []))
        for witness in row["witnesses"]:
            experience = entity.get("experiences", witness)
            prediction = entity.get("predictions", experience["prediction"])
            observation = entity.get("observations", prediction["observation"])
            source_closure.add(observation["source"])
        # Even scalar constants can reveal private facts: permission below covers derived payloads too.
        programs.append({"source_skill": key, "family": row["family"], "domain": row["domain"],
                         "scope": clone(row["scope"]), "program": clone(row["program"]),
                         "source_qualification_digest": sha(row["certificate"]), "import_status": "unqualified"})
    require(source_closure and source_closure <= rights.keys(), "export provenance closure is incomplete")
    selected = {}
    for source in sorted(source_closure):
        require(not entity.get("revoked-sources", source), "revoked source cannot be exported")
        row = rights[source]
        fields(row, {"license", "share_derived", "privacy_review", "classification", "review_digest"})
        require(row["share_derived"] is True and row["privacy_review"] is True and row["classification"] == "public",
                "private or unreviewed derived information is not publishable")
        digest(row["review_digest"])
        require(type(row["license"]) is str and row["license"], "redistribution license required")
        selected[source] = clone(row)
    document = {"schema": "substrate-skill-capsule-v1", "author": identifier(author),
                "lineage": {"entity": entity.id, "head": entity.store.head(entity.id)},
                "programs": programs, "source_rights": selected, "raw_personal_data_included": False,
                "authority_inherited": False, "automatic_merge": False}
    require(len(canonical(document)) <= 4 * 1024 * 1024, "capsule exceeds import bound")
    return document


def import_capsule(entity, document: dict, certificate: dict) -> str:
    fields(document, {"schema", "author", "lineage", "programs", "source_rights", "raw_personal_data_included",
                      "authority_inherited", "automatic_merge"})
    require(document["schema"] == "substrate-skill-capsule-v1" and document["raw_personal_data_included"] is False
            and document["authority_inherited"] is False and document["automatic_merge"] is False, "unsupported exchange semantics")
    require(type(document["programs"]) is list and 1 <= len(document["programs"]) <= 128, "capsule program bound")
    require(type(document["source_rights"]) is dict and document["source_rights"], "source rights closure missing")
    for source, rights in document["source_rights"].items():
        digest(source)
        require(rights.get("share_derived") is True and rights.get("privacy_review") is True
                and rights.get("classification") == "public" and rights.get("license"), "unreviewed source rights")
        digest(rights.get("review_digest"))
        require(not entity.get("revoked-sources", source), "capsule source revoked")
    for row in document["programs"]:
        fields(row, {"source_skill", "family", "domain", "scope", "program", "source_qualification_digest", "import_status"})
        require(row["import_status"] == "unqualified", "foreign qualification cannot grant local execution")
        compile_program(row["program"])
        digest(row["source_skill"])
        digest(row["source_qualification_digest"])
    findings = entity.trust.verify(certificate, purpose="exchange", domain="cognition", subject=document, producer=document["author"])
    require(findings.get("rights_reviewed") is True and findings.get("privacy_reviewed") is True, "exchange requires independent disclosure review")
    key = sha(document)
    entity._commit("capsule-quarantined", {"capsule": key}, [("quarantine-capsules", key,
                   {"document": clone(document), "certificate": clone(certificate), "status": "quarantined"})],
                   base=entity.store.head(entity.id))
    return key


def propose_imported_skill(entity, capsule_id: str, index: int, witnesses: list[str]) -> str:
    """Import only via local witnesses and the normal independent qualification path."""
    row = entity.get("quarantine-capsules", digest(capsule_id))
    require(row and entity._certificate_live(row["certificate"]), "capsule review unavailable")
    programs = row["document"]["programs"]
    require(type(index) is int and 0 <= index < len(programs), "capsule index outside range")
    require(not any(entity.get("revoked-sources", key) for key in row["document"]["source_rights"]), "capsule source revoked")
    program = programs[index]
    return entity.propose_skill(program["program"], family=program["family"], domain=program["domain"],
                                scope=program["scope"], witnesses=witnesses, producer=row["document"]["author"],
                                source_ancestry=sorted(row["document"]["source_rights"]))
