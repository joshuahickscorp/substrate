from devsys.studies.negative_registry import (
    SIGNALS,
    TOY,
    _bool_signals,
    _verdict,
    negative_registry,
    to_markdown,
)


def test_verdict_logic_is_non_vacuous():
    assert _verdict({"a": True, "b": True}) == "refuted"
    assert _verdict({"a": False, "b": False}) == "confirmed"
    assert _verdict({"a": True, "b": False}) == "mixed"


def test_bool_signals_inverts_e4_and_unwraps_i4_list():
    # E4: point-error CHASING noise means the gate did NOT separate => lever-win is the inverse.
    e4 = _bool_signals("e4_neuromod", {"point_error_chases_noise": True, "disagreement_ignores_noise": True})
    assert e4["point_error_separates_noise"] is False
    assert e4["disagreement_ignores_noise"] is True
    # I4: result exposes a list, not a bool; non-empty => some rule matched backprop.
    assert (
        _bool_signals("i4_backprop_alts", {"rules_within_margin": ["fa"]})["any_rule_within_margin"] is True
    )
    assert _bool_signals("i4_backprop_alts", {"rules_within_margin": []})["any_rule_within_margin"] is False


def test_registry_runs_all_experiments_and_records_real_booleans():
    reg = negative_registry(toy=True)
    assert reg["provenance"] == "provisional"
    # one entry per experiment the leg covers, none silently dropped
    assert len(reg["entries"]) == len(TOY)
    assert {e["experiment"] for e in reg["entries"]} == set(TOY)

    for e in reg["entries"]:
        assert e["verdict"] in ("confirmed", "refuted", "mixed", "error")
        assert e["null_hypothesis"]  # pulled off the Experiment.null_hypothesis attr, non-empty
        assert e["provenance"] == "provisional"
        if e["verdict"] != "error":
            # the observed booleans are the actual null check the experiment computed
            assert e["observed"], e["experiment"]
            assert e["taxonomy_category"] in range(1, 11)
            # verdict is CONSISTENT with the observed lever-beats-baseline booleans
            bools = [v for v in e["observed"].values() if isinstance(v, bool)]
            if e["experiment"] != "e1_baseline":
                assert e["verdict"] == _verdict({f"s{i}": b for i, b in enumerate(bools)})

    # summary tallies match the entries exactly (not a stale or hardcoded count)
    s = reg["summary"]
    counts = {
        k: sum(1 for e in reg["entries"] if e["verdict"] == k)
        for k in ("confirmed", "refuted", "mixed", "error")
    }
    assert s == counts
    assert sum(s.values()) == len(TOY)


def test_e1_gate_drives_a_named_taxonomy_leaf():
    # E1 is special-cased: at toy scale the gate should not pass, landing the null in a real
    # taxonomy slot (3 latent-not-decodable or 5 task-too-easy), never None.
    reg = negative_registry(toy=True)
    e1 = next(e for e in reg["entries"] if e["experiment"] == "e1_baseline")
    assert "gate_passed" in e1["observed"]
    assert e1["taxonomy_category"] in (3, 5)
    # confirmed when the gate did not pass, refuted when it did
    assert e1["verdict"] == ("refuted" if e1["observed"]["gate_passed"] else "confirmed")


def test_signals_cover_every_non_e1_experiment():
    # the SIGNALS table (taxonomy + lever booleans) covers every experiment except the
    # special-cased E1: a missing mapping would silently skip an experiment's null.
    assert set(SIGNALS) | {"e1_baseline"} == set(TOY)
    for _keys, cat, label in SIGNALS.values():
        assert cat in range(1, 11) and label


def test_markdown_table_renders_every_entry():
    reg = negative_registry(toy=True)
    md = to_markdown(reg)
    assert md.startswith("# Leg 11D")
    assert "| Exp | Verdict |" in md
    for e in reg["entries"]:
        assert e["experiment"] in md
    assert "—" not in md and "–" not in md  # NO em/en dash rule
