from ub_cse_bot.guardrails.scope import ScopeClassifier


def test_keyword_in_scope() -> None:
    c = ScopeClassifier(llm=None)
    d = c.classify("What are the prerequisites for CSE 574?")
    assert d.label == "in_scope"


def test_keyword_small_talk() -> None:
    c = ScopeClassifier(llm=None)
    assert c.classify("hi").label == "small_talk"


def test_out_of_scope_blocklist() -> None:
    c = ScopeClassifier(llm=None)
    d = c.classify("Where's the best pizza in Buffalo?")
    assert d.label == "out_of_scope"
    assert "UB" in d.redirect or "CSE" in d.redirect
