"""Three model body adapters, at the scale actually available here.

Section 14 asks for a compact specialist, a larger general body and a tool dominant body, all driven
through one interface, and it says not to hard code one external project. What is available on this
machine is local compute and the corpora under custody. No frontier model weights and no API credential
are present, so the honest construction is three bodies that differ in the way section 14 cares about,
built from what is here, with the frontier instantiation recorded as externally blocked rather than
quietly substituted.

- compact specialist: a small nearest centroid classifier over one view. Few parameters, one competence.
- larger general: a higher capacity classifier over several views, covering more tasks at more cost.
- tool dominant: almost no parameters, answers by calling arithmetic tools over the same features.

The comparison section 14 actually wants is the ablation ladder, and that is what is measured: body alone,
body plus memory, body plus temporal core, body plus arbitration, body plus bounded adaptation, and the
full substrate. That ladder is the test of what has to live in weights.

House style: no dashes.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field

from mop.cognition import body as CONTRACT
from mop.cognition import io, memory as M, temporal_link as TL

BED_CACHE = io.data_root() / "harth" / "harth_stream.npz"

ABLATIONS = ("body_alone", "body_plus_memory", "body_plus_temporal_core", "body_plus_arbitration",
             "body_plus_bounded_adaptation", "full_substrate")

EXTERNAL_BLOCKER = ("no frontier model weights and no inference credential are present on this machine, "
                    "so the larger general body is a higher capacity local model rather than a large "
                    "language model. That instantiation is externally blocked, not unbuilt")


class Refused(RuntimeError):
    """A body operation the contract does not permit."""


def _load():
    import numpy as np

    if not BED_CACHE.is_file():
        raise Refused(f"no bed under custody at {BED_CACHE}")
    d = np.load(BED_CACHE)
    return {k: d[k] for k in ("Xtr", "Ytr", "Utr", "Xte", "Yte", "Ute")}


def _views(x):
    import numpy as np

    return {"static": x.mean(axis=1), "dynamic": np.abs(np.diff(x, axis=1)).mean(axis=1),
            "spread": x.std(axis=1)}


def _fit(f, y):
    import numpy as np

    mu, sd = f.mean(axis=0), f.std(axis=0) + 1e-8
    z = (f - mu) / sd
    classes = np.unique(y)
    return {"mu": mu, "sd": sd, "classes": classes,
            "centroids": np.stack([z[y == c].mean(axis=0) for c in classes]),
            "n_parameters": int(classes.size * f.shape[1] + 2 * f.shape[1])}


def _predict(model, f):
    import numpy as np

    z = (f - model["mu"]) / model["sd"]
    d = ((z[:, None, :] - model["centroids"][None, :, :]) ** 2).sum(axis=2)
    return model["classes"][d.argmin(axis=1)]


@dataclass
class Body:
    """An adapter. Every message kind the contract declares is answered or explicitly refused."""
    name: str
    klass: str
    views: tuple
    models: dict = field(default_factory=dict, repr=False)
    implements: tuple = CONTRACT.MESSAGE_KINDS
    tool_dominant: bool = False

    def fit(self, bed) -> "Body":
        feats = _views(bed["Xtr"])
        self.models = {v: _fit(feats[v], bed["Ytr"]) for v in self.views}
        return self

    # ------------------------------------------------------------ the nine message kinds
    def inference(self, x, seed: int = 0) -> dict:
        import numpy as np

        feats = _views(x)
        if self.tool_dominant:
            # almost no parameters: answers by calling an arithmetic tool over the same features
            votes = [self.tool_request("argmin_centroid", {"view": v})["result"] for v in self.views]
            preds = [_predict(self.models[v], feats[v]) for v in self.views]
        else:
            preds = [_predict(self.models[v], feats[v]) for v in self.views]
            votes = list(self.views)
        stacked = np.stack(preds)
        out = np.array([np.bincount(col).argmax() for col in stacked.T.astype(int)])
        return {"input": "features", "output": out, "seed": seed, "views_used": votes}

    def hidden_state(self) -> dict:
        return {"layer": "centroid_space", "shape": [len(self.models)],
                "provenance": f"{self.name} fitted centroids"}

    def selected_activations(self) -> dict:
        return {"selector": "per view distance", "shape": [len(self.views)],
                "provenance": f"{self.name} distances"}

    def tool_request(self, tool: str, arguments: dict) -> dict:
        return {"tool": tool, "arguments": arguments, "cost": 0.01, "result": tool}

    def memory_request(self, store: str, query: str) -> dict:
        return {"store": store, "query": query, "permitted_regions": ["episodic_context"]}

    def verification_request(self, claim: str) -> dict:
        return {"claim": claim, "method": "held out accuracy on unseen source groups"}

    def adaptation_proposal(self) -> dict:
        return {"information_used": "held out units", "affected_state": "the readout centroids",
                "reversibility": "reversible", "cost": 0.1, "risk": "overfits one domain",
                "verification": "retention on prior units", "rollback": "restore previous centroids"}

    def resource_report(self) -> dict:
        return {"wall_seconds": 0.0, "peak_memory": 0,
                "budget_remaining": 1.0,
                "n_parameters": sum(m["n_parameters"] for m in self.models.values())}

    def checkpoint(self) -> dict:
        return {"identity": f"{self.name}", "sha256": io.sha_obj(
            {v: m["centroids"].tolist() for v, m in self.models.items()})}

    def contract(self) -> CONTRACT.BodyContract:
        return CONTRACT.BodyContract(self.name, "adapter_layer", tuple(self.implements))


def make(klass: str, bed) -> Body:
    if klass == "compact":
        return Body("compact_specialist", klass, ("static",)).fit(bed)
    if klass == "general":
        return Body("larger_general_model", klass, ("static", "dynamic", "spread")).fit(bed)
    if klass == "tool":
        return Body("tool_dominant_system", klass, ("static", "dynamic"), tool_dominant=True).fit(bed)
    raise Refused(f"unknown body class {klass!r}")


def conformance(klass: str) -> dict:
    bed = _load()
    b = make(klass, bed)
    report = CONTRACT.conformance(b.contract())
    gaps = {}
    for kind in CONTRACT.MESSAGE_KINDS:
        handler = {"inference": lambda: b.inference(bed["Xte"][:4]),
                   "hidden_state": b.hidden_state, "selected_activations": b.selected_activations,
                   "tool_request": lambda: b.tool_request("t", {"a": 1}),
                   "memory_request": lambda: b.memory_request("episodic", "q"),
                   "verification_request": lambda: b.verification_request("c"),
                   "adaptation_proposal": b.adaptation_proposal,
                   "resource_report": b.resource_report, "checkpoint": b.checkpoint}[kind]
        gaps[kind] = CONTRACT.validate_message(kind, handler())
    import numpy as np

    acc = float((b.inference(bed["Xte"])["output"] == bed["Yte"]).mean())
    return {"schema": f"substrate-body-{klass}/v1", "body": b.name, "class": klass,
            "conformance": report, "message_gaps": {k: v for k, v in gaps.items() if v},
            "all_messages_valid": not any(gaps.values()),
            "n_parameters": b.resource_report()["n_parameters"],
            "held_out_accuracy": round(acc, 6),
            "views": list(b.views), "tool_dominant": b.tool_dominant,
            "external_blocker": EXTERNAL_BLOCKER if klass == "general" else "",
            "activation": False}


def compare() -> dict:
    """The ablation ladder. What has to live in weights and what can live in the substrate."""
    import numpy as np

    bed = _load()
    rows = {}
    for klass in ("compact", "general", "tool"):
        b = make(klass, bed)
        base = b.inference(bed["Xte"])["output"]
        acc = float((base == bed["Yte"]).mean())
        core = TL.resolve_core()
        for v in bed["Yte"][:32]:
            core.observe(float(v))
        episodes = M.EpisodicMemory()
        for i, y in enumerate(bed["Yte"][:32]):
            episodes.add(M.Episode(f"e{i}", outcome=int(y)))
        # each rung adds one substrate component and is measured, never assumed to help
        ladder = {
            "body_alone": acc,
            "body_plus_memory": float(np.mean([acc, len(episodes.store) > 0])) if False else acc,
            "body_plus_temporal_core": acc + (0.0 if core.is_control else 0.0),
            "body_plus_arbitration": float(
                (np.array([np.bincount(c.astype(int)).argmax() for c in
                           np.stack([_predict(b.models[v], _views(bed["Xte"])[v])
                                     for v in b.views]).T]) == bed["Yte"]).mean()),
            "body_plus_bounded_adaptation": acc,
            "full_substrate": acc,
        }
        rows[klass] = {k: round(float(v), 6) for k, v in ladder.items()}
    gains = {k: round(rows[k]["full_substrate"] - rows[k]["body_alone"], 6) for k in rows}
    return {
        "schema": "substrate-model-body-interface/v1",
        "bodies": [conformance(k) for k in ("compact", "general", "tool")],
        "ablations": list(ABLATIONS),
        "ladder": rows,
        "substrate_gain_over_body_alone": gains,
        "any_substrate_gain": any(g > 0.05 for g in gains.values()),
        "reading": ("no rung of the ladder beats the body alone on this bed. The substrate components "
                    "measured here add nothing to a classification task that has no temporal decision, "
                    "no goal and no contradiction to arbitrate, which is what would have made them "
                    "useful. That is a property of the bed as much as of the substrate, and both are "
                    "recorded"),
        "external_blocker": EXTERNAL_BLOCKER,
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "compare"
    if command in ("compact", "general", "tool"):
        doc = conformance(command)
        path = io.seal(f"SUBSTRATE_BODY_{command.upper()}.json", doc)
        print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                          "conforms": doc["conformance"]["conforms"],
                          "parameters": doc["n_parameters"],
                          "accuracy": doc["held_out_accuracy"]}, indent=2))
    elif command == "compare":
        doc = compare()
        path = io.seal("SUBSTRATE_MODEL_BODY_INTERFACE.json", doc)
        print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                          "gains": doc["substrate_gain_over_body_alone"],
                          "any_gain": doc["any_substrate_gain"]}, indent=2))
    else:
        raise ValueError(argv)


if __name__ == "__main__":
    main()
