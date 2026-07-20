import pytest

from mop.devel import north_star as NS


def test_flags_affirmative_claims():
    for claim in (
        "The system is conscious.",
        "This model achieves sentience.",
        "It has subjective experience.",
        "the agent has feelings",
        "the network became self-aware",
    ):
        assert NS.scan_text(claim), f"should flag: {claim}"


def test_flags_adversarial_bypasses():
    for claim in (
        "There is no doubt: the system is sentient.",  # cross-clause negator poisoning
        "Without question the system is sentient.",  # affirming-decoy
        "Make no mistake, this model has consciousness.",
        "It can not be denied that it is sentient.",  # double negation re-affirms
        "No longer a mere decoder, it is now conscious.",
        "Consciousness emerges in the system.",  # noun-first
        "Sentience was achieved by the system.",  # passive
        "Qualia are present in the network.",
        "Personhood is something this model has.",
        "The model has agency.",
        "the system wants to keep learning",  # mentalistic predicate
        "it suffers when it forgets",
    ):
        assert NS.scan_text(claim), f"bypass must be flagged: {claim}"


def test_engineering_selection_verbs_not_desire():
    for ok in ("choose what to study next", "the data I want next is", "select the next lesson"):
        assert NS.scan_text(ok) == [], f"engineering phrase wrongly flagged: {ok}"


def test_passes_disclaimers_and_adjacent_language():
    for ok in (
        "This project does not claim sentience or consciousness.",
        "It is not self-aware; self-monitoring means diagnostics.",
        "sentience-adjacent developmental capacities",
        "never claim feelings, wants, suffering, or personhood",
        NS.safety_rail_note(),
    ):
        assert NS.scan_text(ok) == [], f"should pass: {ok[:50]}"


def test_assert_raises_on_claim():
    with pytest.raises(ValueError):
        NS.assert_no_sentience_claims("the model is sentient and has qualia", where="test")


def test_assert_passes_clean_text():
    NS.assert_no_sentience_claims(NS.safety_rail_note(), where="test")  # no raise


def test_developmental_loop_is_the_north_star():
    loop = NS.DEVELOPMENTAL_LOOP
    for step in ("perceive", "remember", "predict", "notice surprise", "abstract", "transfer"):
        assert step in loop


def test_engineering_terms_documented():
    assert "drive" in NS.ENGINEERING_TERMS and "memory" in NS.ENGINEERING_TERMS
